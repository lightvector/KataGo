"""
Tests for the multi-GPU training-path optimizations:

  1. Model.compile_per_block_trunk (KATAGO_COMPILE_PER_BLOCK): the per-block compiled trunk
     gives the same outputs and parameter gradients as compiling the whole model as one graph.
     Needs one CUDA device and compiles the model (about a minute).
  2. MuonWithAuxAdam with bf16 update gathering (KATAGO_MUON_GATHER_BF16_UPDATES): parameters end
     up bitwise identical to the fp32 parameter-gather path and bitwise identical across ranks.
     Needs two CUDA devices and runs a 2-rank NCCL process group.

Skipped when the hardware is not available.
"""

import copy
import os

import pytest
import torch

cuda_count = torch.cuda.device_count() if torch.cuda.is_available() else 0

MODEL_KIND = "b11c768h12nbt3tflrs-fson-silu"
DATA = os.path.join(os.path.dirname(__file__), "..", "testdata", "benchmark_data_1024.npz")


def _grad_rel(ga, gb):
    d = sum((ga[n] - gb[n]).double().square().sum() for n in ga).sqrt()
    a = sum(gb[n].double().square().sum() for n in gb).sqrt()
    return (d / a).item()


@pytest.mark.skipif(cuda_count < 1, reason="requires a CUDA device")
def test_per_block_compiled_trunk_matches_whole_model():
    from torch.amp import autocast
    from katago.train import modelconfigs
    from katago.train.model_pytorch import Model
    from benchmark_fresh_model import load_batch

    torch._dynamo.config.recompile_limit = max(torch._dynamo.config.recompile_limit, 32)
    dev = torch.device("cuda")
    torch.manual_seed(3)
    cfg = modelconfigs.config_of_name[MODEL_KIND].copy()
    base = Model(cfg, 19)
    base.initialize()
    base.to(dev)
    base.train()
    base.attn_logit_penalty_cap = 250.0
    base.attn_logit_penalty_batch_frac = 0.25
    assert base.supports_per_block_compile()
    batch = load_batch(DATA, 32, 19, cfg, dev)

    def run(per_block):
        model = copy.deepcopy(base)
        model.compile_per_block_trunk = per_block
        compiled = torch.compile(model)
        for _ in range(2):  # second iteration exercises the steady-state compiled path
            model.zero_grad(set_to_none=True)
            with autocast("cuda", dtype=torch.float16):
                outs = compiled(batch["binaryInputNCHW"], batch["globalInputNC"])
            flat = [o.float() for heads in outs for o in heads]
            loss = sum(o.square().mean() for o in flat) + 1e-3 * model.attn_logit_penalty_per_sample.mean()
            loss.backward()
        grads = {n: p.grad.detach().clone() for n, p in model.named_parameters()}
        return [o.detach() for o in flat], grads

    outs_whole, grads_whole = run(False)
    outs_pb, grads_pb = run(True)
    out_diff = max(((a - b).norm() / (b.norm() + 1e-12)).item() for a, b in zip(outs_pb, outs_whole))
    assert out_diff < 1e-2, out_diff
    assert _grad_rel(grads_pb, grads_whole) < 1e-2


def _muon_gather_worker(rank, world, port, result_queue):
    import torch.distributed as dist
    from muon.muon import MuonWithAuxAdam
    from katago.train import modelconfigs
    from katago.train.model_pytorch import Model
    from benchmark_fresh_model import build_train_param_groups

    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(port)
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", rank=rank, world_size=world)
    try:
        torch.manual_seed(7)
        cfg = modelconfigs.config_of_name[MODEL_KIND].copy()
        base = Model(cfg, 19)
        base.initialize()
        base.cuda()

        def run_mode(gather_bf16, steps):
            model = copy.deepcopy(base)
            opt = MuonWithAuxAdam(build_train_param_groups(model), adjust_lr_fn="match_rms_adamw")
            assert opt.gather_bf16_updates, "expected the bf16 update gather to be eligible for this model"
            opt.gather_bf16_updates = gather_bf16
            for g in opt.param_groups:
                g["lr"] = 1e-3
                g["weight_decay"] = 1e-2
            gen = torch.Generator(device="cuda").manual_seed(1234)  # identical grads on every rank
            for _ in range(steps):
                for p in model.parameters():
                    p.grad = torch.randn(p.shape, generator=gen, device="cuda", dtype=p.dtype) * 1e-2
                opt.step()
            return {n: p.detach().clone() for n, p in model.named_parameters()}

        # The first call of a freshly compiled Newton-Schulz shape has been seen to differ slightly
        # from later calls, so warm the shapes up before comparing, and check that repeating the
        # same mode is itself bitwise stable so a failure below is attributable to the gather mode.
        run_mode(False, 1)
        params_fp32_gather = run_mode(False, 4)
        params_fp32_gather_again = run_mode(False, 4)
        n_unstable = sum(int(not torch.equal(params_fp32_gather[k], params_fp32_gather_again[k])) for k in params_fp32_gather)
        params_bf16_gather = run_mode(True, 4)
        n_diff = sum(int(not torch.equal(params_fp32_gather[k], params_bf16_gather[k])) for k in params_fp32_gather)
        flat = torch.cat([params_bf16_gather[k].reshape(-1) for k in sorted(params_bf16_gather)])
        from_rank0 = flat.clone()
        dist.broadcast(from_rank0, src=0)
        cross_rank_equal = torch.equal(flat, from_rank0)
        result_queue.put((rank, n_unstable, n_diff, cross_rank_equal))
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(cuda_count < 2, reason="requires two CUDA devices")
def test_muon_bf16_update_gather_is_bitwise_identical():
    import torch.multiprocessing as mp

    ctx = mp.get_context("spawn")
    result_queue = ctx.Queue()
    world = 2
    ctx_procs = mp.spawn(_muon_gather_worker, nprocs=world, args=(world, 23517, result_queue), join=False, start_method="spawn")
    results = [result_queue.get(timeout=1200) for _ in range(world)]
    ctx_procs.join()
    for rank, n_unstable, n_diff, cross_rank_equal in results:
        assert n_unstable == 0, f"rank {rank}: {n_unstable} parameter tensors differ between two identical fp32-gather runs"
        assert n_diff == 0, f"rank {rank}: {n_diff} parameter tensors differ between gather modes"
        assert cross_rank_equal, f"rank {rank}: parameters differ from rank 0"
