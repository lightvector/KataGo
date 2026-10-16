#!/usr/bin/python3
"""Preflight check of the compiled training paths on this machine.

Compiles a model exactly the way train.py compiles it (torch.compile, per-block trunk, DDP when
several GPUs are given, the fused Triton kernels) and checks, with freed GPU memory NaN-poisoned
first so that any kernel reading uninitialized memory shows up, that:

  1. at a comparison batch size small enough for an eager fp32 reference to fit in memory, the
     compiled gradients are finite and match that reference at least as closely as the plain
     eager AMP path does,
  2. at the training batch size, the compiled gradients are finite and, when an eager AMP
     reference fits in memory, not grossly different from it, and
  3. one Muon step leaves every rank's parameters finite and bitwise identical across ranks.

Run it once per new machine, PyTorch version, model config, batch size, or GPU count before
starting a training run, with the same batch size per GPU and GPU count as the run, and with
-attn-logit-penalty-cap if the run uses it. It takes a few minutes, mostly compilation. Exit
status is nonzero on failure.

Every rank loads the same batch, so the DDP-averaged gradient equals the single-GPU gradient
and can be compared against the local eager reference on each rank.

Examples:
  python check_training_paths.py -model-kind b11c768h12nbt3tflrs-fson-silu -batch-size 96 -multi-gpus 0,1
  python check_training_paths.py -model-kind b10c384h6nbttflrs-fson-silu-rsnh -batch-size 256 -gpu 0
"""

import argparse
import gc
import os
import sys

import torch
from torch.amp import autocast

from katago.train import modelconfigs
from katago.train import trainloop_helpers
from katago.train.model_pytorch import Model, TransformerAttentionBlock, TransformerFFNBlock
from benchmark_fresh_model import load_batch, build_train_param_groups, make_optimizer

DEFAULT_DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "testdata", "benchmark_data_1024.npz")


def main():
    parser = argparse.ArgumentParser(description="Check the compiled training paths against eager on this machine")
    parser.add_argument("-model-kind", required=True, help="Model config name, e.g. b11c768h12nbt3tflrs-fson-silu")
    parser.add_argument("-batch-size", type=int, required=True, help="Batch size per GPU, as in the training run")
    parser.add_argument("-compare-batch-size", type=int, default=32,
                        help="Batch size for the comparison against an eager fp32 reference, which needs several times the memory of a training step")
    parser.add_argument("-data", default=DEFAULT_DATA, help="npz file to take the batches from")
    parser.add_argument("-pos-len", type=int, default=19)
    parser.add_argument("-gpu", type=int, default=0)
    parser.add_argument("-multi-gpus", type=str, default=None, help="Comma-separated GPU ids for a DDP check, e.g. 0,1")
    parser.add_argument("-master-port", type=int, default=23456)
    parser.add_argument("-use-bf16", action="store_true", help="Check under bf16 autocast instead of fp16")
    parser.add_argument("-attn-logit-penalty-cap", type=float, default=None, help="As in train.py, if the run uses it")
    parser.add_argument("-attn-logit-penalty-batch-frac", type=float, default=1.0, help="As in train.py")
    parser.add_argument("-no-poison", action="store_true", help="Do not NaN-fill freed GPU memory before the compiled runs")
    parser.add_argument("-tolerance-factor", type=float, default=1.5,
                        help="At the comparison batch size, the compiled gradient error relative to fp32 may be at most this times the eager AMP error, plus -tolerance-floor")
    parser.add_argument("-tolerance-floor", type=float, default=1e-2,
                        help="Absolute slack added to the comparison tolerance. Matters under bf16, where the eager error is a few 1e-3 and the per-block compiled trunk differs from eager by a similar amount")
    parser.add_argument("-gross-tolerance", type=float, default=0.5,
                        help="At the training batch size, the relative difference between compiled and eager AMP gradients must stay below this")
    args = vars(parser.parse_args())

    gpu_ids = [int(x) for x in args["multi_gpus"].split(",")] if args["multi_gpus"] is not None else [args["gpu"]]
    args["gpu_ids"] = gpu_ids
    world_size = len(gpu_ids)
    if world_size > 1:
        torch.multiprocessing.spawn(worker, nprocs=world_size, args=(world_size, args))
    else:
        worker(0, 1, args)


def grad_rel(ga, gb):
    assert set(ga) == set(gb), "gradient sets differ"
    diff = sum((ga[n] - gb[n]).square().sum() for n in ga).sqrt()
    norm = sum(gb[n].square().sum() for n in gb).sqrt()
    return (diff / (norm + 1e-30)).item()


def worst_params(ga, gb, k=3):
    per_param = sorted(
        ((((ga[n] - gb[n]).norm() / (gb[n].norm() + 1e-30)).item(), n) for n in ga), reverse=True
    )
    return ", ".join(f"{n} {v:.2e}" for v, n in per_param[:k])


def poison_freed_memory(device):
    """Leave NaN-filled blocks of many sizes in the caching allocator's pool, so that later
    allocations that reuse them start out as NaN. The blocks must stay cached: releasing them
    with torch.cuda.empty_cache() would hand fresh memory to later allocations instead."""
    cached_free = torch.cuda.memory_reserved(device) - torch.cuda.memory_allocated(device)
    free_bytes, total_bytes = torch.cuda.mem_get_info(device)
    # The poisoned blocks stay in the cache as separate segments that cannot merge, so keep their
    # total modest or a later large allocation may find no block big enough when memory is tight.
    budget = min(cached_free + int(free_bytes * 0.4), total_bytes // 4)
    junk = []
    size = 1 << 20
    while size <= budget // 4:
        junk.append(torch.empty(size // 2, dtype=torch.float16, device=device))
        junk[-1].fill_(float("nan"))
        budget -= size
        size *= 2
    for _ in range(8):
        junk.append(torch.empty(budget // 8 // 2, dtype=torch.float16, device=device))
        junk[-1].fill_(float("nan"))
    del junk


def worker(rank, world_size, args):
    device = torch.device(f"cuda:{args['gpu_ids'][rank]}")
    torch.cuda.set_device(device)
    if world_size > 1:
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = str(args["master_port"])
        torch.distributed.init_process_group("nccl", rank=rank, world_size=world_size)

    def log(msg):
        print(f"[rank {rank}] {msg}", flush=True)

    torch.set_float32_matmul_precision("high")
    torch.backends.cudnn.benchmark = True
    torch._dynamo.config.recompile_limit = max(torch._dynamo.config.recompile_limit, 64)
    # Training compiles once at a fixed batch size. Compiling a second batch size would by default
    # switch dynamo to a dynamic batch dimension, which changes inductor's layout decisions and
    # can hide problems that only the static graph has, so give each batch size its own static graph.
    torch._dynamo.config.automatic_dynamic_shapes = False
    amp_dtype = torch.bfloat16 if args["use_bf16"] else torch.float16
    poison = not args["no_poison"]

    model_config = modelconfigs.config_of_name[args["model_kind"]].copy()
    torch.manual_seed(20260904)
    raw_model = Model(model_config, args["pos_len"])
    raw_model.initialize()
    raw_model.to(device)
    raw_model.train()
    penalty_cap = args["attn_logit_penalty_cap"]
    if penalty_cap is not None:
        raw_model.attn_logit_penalty_cap = penalty_cap
        raw_model.attn_logit_penalty_batch_frac = args["attn_logit_penalty_batch_frac"]
    grad_params = {n for n, p in raw_model.named_parameters() if p.requires_grad}
    compare_batch_size = min(args["compare_batch_size"], args["batch_size"])
    compare_batch = load_batch(args["data"], compare_batch_size, args["pos_len"], model_config, device)
    full_batch = load_batch(args["data"], args["batch_size"], args["pos_len"], model_config, device)

    def run(model_fn, batch, use_autocast):
        raw_model.zero_grad(set_to_none=True)
        with autocast("cuda", dtype=amp_dtype, enabled=use_autocast):
            outputs = model_fn(batch["binaryInputNCHW"], batch["globalInputNC"])
        # A loss on the raw outputs rather than the training loss, so every head and every
        # trunk parameter receives gradient regardless of the batch's targets.
        loss = sum(o.float().square().mean() for heads in outputs for o in heads)
        if penalty_cap is not None:
            loss = loss + 1e-3 * raw_model.attn_logit_penalty_per_sample.mean()
        loss.backward()
        grads = {n: p.grad.detach().clone().float() for n, p in raw_model.named_parameters() if p.grad is not None}
        missing = grad_params - set(grads)
        assert not missing, f"parameters without gradient: {sorted(missing)[:5]}"
        return grads

    def nonfinite_params(grads):
        return [n for n, g in grads.items() if not torch.isfinite(g).all()]

    grads_fp32 = run(raw_model, compare_batch, use_autocast=False)
    err_eager = grad_rel(run(raw_model, compare_batch, use_autocast=True), grads_fp32)

    # DDP lays out its own gradient buffers, so leave no gradients from the eager runs behind.
    raw_model.zero_grad(set_to_none=True)
    compiled_model = trainloop_helpers.wrap_model_for_training(raw_model, device, world_size, no_compile=False)
    fused_rope = any(m.fused_rope_backward for m in raw_model.modules() if isinstance(m, TransformerAttentionBlock))
    fused_swiglu = any(m.fused_swiglu_kernel for m in raw_model.modules() if isinstance(m, TransformerFFNBlock))
    log(f"model={args['model_kind']} batch={args['batch_size']}/gpu compare_batch={compare_batch_size} "
        f"world_size={world_size} amp={amp_dtype} penalty_cap={penalty_cap} per_block_trunk={raw_model.compile_per_block_trunk} "
        f"fused_rope={fused_rope} fused_swiglu={fused_swiglu}")

    ok = True
    for iteration in range(2):
        if poison:
            poison_freed_memory(device)
        grads_compiled = run(compiled_model, compare_batch, use_autocast=True)
        nonfinite = nonfinite_params(grads_compiled)
        err_compiled = grad_rel(grads_compiled, grads_fp32)
        passed = len(nonfinite) == 0 and err_compiled <= args["tolerance_factor"] * err_eager + args["tolerance_floor"]
        ok = ok and passed
        log(f"compiled run {iteration} at batch {compare_batch_size}: gradient error vs fp32 {err_compiled:.3e} "
            f"(eager AMP {err_eager:.3e}), {len(nonfinite)} nonfinite -> {'OK' if passed else 'FAIL'}")
        if not passed:
            log(f"  nonfinite: {nonfinite[:5]}")
            log(f"  worst parameters: {worst_params(grads_compiled, grads_fp32)}")
    del grads_fp32, grads_compiled

    # The eager AMP pass at the training batch size needs more memory than a compiled step and
    # may not fit. Without it only finiteness is checked at that batch size.
    try:
        grads_eager_full = run(raw_model, full_batch, use_autocast=True)
    except torch.OutOfMemoryError:
        grads_eager_full = None
    if grads_eager_full is None:
        # Release what the failed attempt left behind (done outside the except block, whose
        # traceback would keep the attempt's activations alive), including the penalty tensors
        # the model keeps as attributes together with their autograd graph.
        raw_model.zero_grad(set_to_none=True)
        for attr in ("attn_logit_penalty_per_sample", "attn_logit_ub_batch_max"):
            if hasattr(raw_model, attr):
                setattr(raw_model, attr, None)
        gc.collect()
        torch.cuda.empty_cache()
        log(f"eager AMP reference at batch {args['batch_size']} does not fit in memory, checking finiteness only there")
    for iteration in range(2):
        if poison:
            poison_freed_memory(device)
        grads_compiled = run(compiled_model, full_batch, use_autocast=True)
        nonfinite = nonfinite_params(grads_compiled)
        passed = len(nonfinite) == 0
        detail = ""
        if grads_eager_full is not None:
            err_full = grad_rel(grads_compiled, grads_eager_full)
            passed = passed and err_full < args["gross_tolerance"]
            detail = f", difference vs eager AMP {err_full:.3e}"
        ok = ok and passed
        log(f"compiled run {iteration} at batch {args['batch_size']}: {len(nonfinite)} nonfinite{detail} -> {'OK' if passed else 'FAIL'}")
        if not passed:
            log(f"  nonfinite: {nonfinite[:5]}")
            if grads_eager_full is not None:
                log(f"  worst parameters: {worst_params(grads_compiled, grads_eager_full)}")
    del grads_eager_full, grads_compiled

    # One optimizer step with the full-batch compiled gradients, then every rank must hold the
    # same parameters.
    optimizer = make_optimizer("muon", build_train_param_groups(raw_model), world_size, 5, False)
    for group in optimizer.param_groups:
        group["lr"] = 1e-3
        group["weight_decay"] = 1e-2
    optimizer.step()
    flat = torch.cat([p.detach().reshape(-1) for _, p in sorted(raw_model.named_parameters())])
    step_finite = bool(torch.isfinite(flat).all())
    if world_size > 1:
        reference = flat.clone()
        torch.distributed.broadcast(reference, src=0)
        same_as_rank0 = bool(torch.equal(flat, reference))
    else:
        same_as_rank0 = True
    step_ok = step_finite and same_as_rank0
    ok = ok and step_ok
    log(f"muon step: parameters finite={step_finite} identical to rank 0={same_as_rank0} -> {'OK' if step_ok else 'FAIL'}")

    if world_size > 1:
        status = torch.tensor([0 if ok else 1], device=device)
        torch.distributed.all_reduce(status)
        ok = status.item() == 0
        torch.distributed.destroy_process_group()
    if rank == 0:
        print("PREFLIGHT " + ("PASSED" if ok else "FAILED"), flush=True)
    if not ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
