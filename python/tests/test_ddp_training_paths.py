"""
Tests for the multi-GPU training-path optimizations:

  1. Model.compile_per_block_trunk (KATAGO_COMPILE_PER_BLOCK): the per-block compiled trunk
     gives the same outputs and parameter gradients as compiling the whole model as one graph.
     Needs one CUDA device and compiles the model (about a minute).
  2. MuonWithAuxAdam with bf16 update gathering (KATAGO_MUON_GATHER_BF16_UPDATES), for plain Muon
     and for Aurora: parameters end up bitwise identical to the fp32 parameter-gather path and
     bitwise identical across ranks. Needs two CUDA devices and runs a 2-rank NCCL process group.
  3. The batched compiled Aurora path matches the per-parameter scalar Aurora path about as closely
     as batched Muon matches scalar Muon, on synthetic and on real training gradients, and keeps
     Aurora's uniform row norms. Needs one CUDA device.
  4. On real training gradients, the batched Aurora polar factor is about as close as the scalar
     path, or closer, to an fp32 computation of the same iteration. Needs one CUDA device.
  5. Aurora ignores NorMuon on the batched path, as it does on the scalar path.

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


def _make_base_model(seed):
    from katago.train import modelconfigs
    from katago.train.model_pytorch import Model

    torch._dynamo.config.recompile_limit = max(torch._dynamo.config.recompile_limit, 32)
    torch.manual_seed(seed)
    cfg = modelconfigs.config_of_name[MODEL_KIND].copy()
    base = Model(cfg, 19)
    base.initialize()
    base.cuda()
    return base, cfg


def _real_gradients(base, cfg, num_batches, batch_size=16):
    """Gradients of the real training loss of `base` on `num_batches` distinct minibatches of the
    test data, computed in eager fp32 with TF32 off. Returns a list of {name: grad} dicts."""
    from katago.train.metrics_pytorch import Metrics
    from benchmark_fresh_model import load_batch

    model = copy.deepcopy(base)
    model.train()
    metrics_obj = Metrics(1, model)
    full = load_batch(DATA, num_batches * batch_size, 19, cfg, torch.device("cuda"))
    prev_tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        grads = []
        for i in range(num_batches):
            batch = {k: v[i * batch_size:(i + 1) * batch_size] for k, v in full.items()}
            model.zero_grad(set_to_none=True)
            outputs = model(batch["binaryInputNCHW"], batch["globalInputNC"])
            metrics = metrics_obj.metrics_dict_batchwise(
                model, model.postprocess_output(outputs), extra_outputs=None, batch=batch, is_training=True,
                soft_policy_weight_scale=1.0, disable_optimistic_policy=False, meta_kata_only_soft_policy=False,
                value_loss_scale=1.0, td_value_loss_scales=[0.4, 1.0, 1.0], seki_loss_scale=0.35,
                variance_time_loss_scale=0.5, main_loss_scale=1.0,
                intermediate_loss_scale=0.5 if model.get_has_intermediate_head() else None,
            )
            metrics["loss_sum"].backward()
            grads.append({n: p.grad.detach().clone() for n, p in model.named_parameters() if p.grad is not None})
        return grads
    finally:
        torch.backends.cuda.matmul.allow_tf32 = prev_tf32


def _synthetic_uneven_gradients(base, num_steps):
    """Random gradients with fixed log-normal scales on rows and on columns. Isotropic random
    gradients have rows of statistically equal norm, where Aurora's row preconditioning barely
    changes the Muon polar factor. Uneven rows and columns, as in real gradients, make it differ."""
    scale_gen = torch.Generator(device="cuda").manual_seed(99)
    scales = {}
    for name, p in base.named_parameters():
        scale = torch.exp(torch.randn(p.shape[0], generator=scale_gen, device="cuda")).view(-1, *([1] * (p.dim() - 1)))
        if p.dim() >= 2:
            col_shape = [1, p.shape[1]] + [1] * (p.dim() - 2)
            scale = scale * torch.exp(torch.randn(p.shape[1], generator=scale_gen, device="cuda")).view(col_shape)
        scales[name] = scale
    gen = torch.Generator(device="cuda").manual_seed(4321)
    return [
        {name: torch.randn(p.shape, generator=gen, device="cuda", dtype=p.dtype) * 1e-2 * scales[name]
         for name, p in base.named_parameters()}
        for _ in range(num_steps)
    ]


def _ns5_reference(G, steps, dtype):
    """zeropower_via_newtonschulz5 with the same coefficients and steps, computed in `dtype`. This is
    the same deliberately non-converged iteration, not the exact polar factor."""
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.to(dtype)
    if G.size(-2) > G.size(-1):
        X = X.mT
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    for _ in range(steps):
        A = X @ X.mT
        B = b * A + c * A @ A
        X = a * X + B @ X
    if G.size(-2) > G.size(-1):
        X = X.mT
    return X


def _aurora_polar_reference(update, dtype, ns_steps=5, pp_iterations=2, pp_beta=0.5, eps=1e-7):
    """muon._aurora_polar with every step, including Newton-Schulz, computed in `dtype`."""
    m, n = update.size(-2), update.size(-1)
    if m == n:
        return _ns5_reference(update, ns_steps, dtype)
    transposed = m < n
    if transposed:
        update = update.mT
        m, n = n, m
    G = update.to(dtype)
    row_norm = G.norm(dim=-1, keepdim=True).clamp_(min=eps)
    D = 1.0 / row_norm
    for k in range(pp_iterations):
        U = _ns5_reference(D * G, ns_steps, dtype)
        if k < pp_iterations - 1:
            row_sq = U.pow(2).sum(dim=-1, keepdim=True).clamp_(min=eps * eps)
            D = D * ((n / m) / row_sq).pow(pp_beta)
    if transposed:
        U = U.mT
    return U


def _global_rel(outs, refs):
    d = sum((o.double() - r.double()).square().sum() for o, r in zip(outs, refs)).sqrt()
    a = sum(r.double().square().sum() for r in refs).sqrt()
    return (d / a).item()


@pytest.mark.skipif(cuda_count < 1, reason="requires a CUDA device")
def test_aurora_polar_accuracy_against_fp32_reference():
    """On real training gradients of every Muon-group matrix, the batched compiled Aurora polar
    factor is as close to an fp32 computation of the same iteration as the per-parameter eager bf16
    path is. Plain Muon's Newton-Schulz is measured the same way for comparison. An fp64 spot check
    confirms that fp32 is accurate enough to serve as the reference."""
    from muon.muon import _aurora_polar, _aurora_polar_compiled, zeropower_via_newtonschulz5, zeropower_via_newtonschulz5_compiled
    from benchmark_fresh_model import build_train_param_groups

    base, cfg = _make_base_model(seed=11)
    grads = _real_gradients(base, cfg, num_batches=1)[0]
    muon_names = {id(p) for g in build_train_param_groups(base) if g["use_muon"] for p in g["params"]}
    matrices = [grads[n].reshape(p.shape[0], -1) for n, p in base.named_parameters() if id(p) in muon_names]
    assert len(matrices) > 100

    def batched(fn, mats, **kwargs):
        # Same grouping as the optimizer's batched path: wide orientation, stacked per shape.
        by_shape = {}
        for i, m in enumerate(mats):
            normalized = m.mT if m.shape[0] > m.shape[1] else m
            by_shape.setdefault(tuple(normalized.shape), []).append((i, m.shape[0] > m.shape[1], normalized))
        out = [None] * len(mats)
        for entries in by_shape.values():
            result = fn(torch.stack([e[2] for e in entries]), **kwargs)
            for (i, was_transposed, _), r in zip(entries, result.unbind(0)):
                out[i] = r.mT if was_transposed else r
        return out

    prev_tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        aurora_ref = [_aurora_polar_reference(m, torch.float32) for m in matrices]
        muon_ref = [_ns5_reference(m, 5, torch.float32) for m in matrices]
        spot = [0, len(matrices) // 2, len(matrices) - 1]
        fp32_vs_fp64 = _global_rel([aurora_ref[i] for i in spot], [_aurora_polar_reference(matrices[i], torch.float64) for i in spot])
    finally:
        torch.backends.cuda.matmul.allow_tf32 = prev_tf32

    batched(_aurora_polar_compiled, matrices)  # warm up the compiled shapes
    batched(zeropower_via_newtonschulz5_compiled, matrices, steps=5)
    aurora_batched_err = _global_rel(batched(_aurora_polar_compiled, matrices), aurora_ref)
    aurora_scalar_err = _global_rel([_aurora_polar(m) for m in matrices], aurora_ref)
    muon_batched_err = _global_rel(batched(zeropower_via_newtonschulz5_compiled, matrices, steps=5), muon_ref)
    muon_scalar_err = _global_rel([zeropower_via_newtonschulz5(m, steps=5) for m in matrices], muon_ref)
    aurora_ref_vs_muon_ref = _global_rel(muon_ref, aurora_ref)
    print(f"error vs fp32 reference: aurora batched {aurora_batched_err:.3e} scalar {aurora_scalar_err:.3e}, "
          f"muon batched {muon_batched_err:.3e} scalar {muon_scalar_err:.3e}; fp32 vs fp64 {fp32_vs_fp64:.3e}; "
          f"aurora reference vs muon reference {aurora_ref_vs_muon_ref:.3e}")

    assert fp32_vs_fp64 < 1e-4, fp32_vs_fp64
    assert aurora_batched_err <= 1.25 * aurora_scalar_err + 1e-3, (aurora_batched_err, aurora_scalar_err)
    assert aurora_batched_err < 0.1, aurora_batched_err
    # The batched path must be computing Aurora and not plain Muon.
    assert aurora_batched_err < 0.5 * aurora_ref_vs_muon_ref, (aurora_batched_err, aurora_ref_vs_muon_ref)


@pytest.mark.skipif(cuda_count < 1, reason="requires a CUDA device")
@pytest.mark.parametrize("grad_source", ["synthetic", "real"])
def test_batched_aurora_matches_scalar_aurora(grad_source):
    """The batched compiled Aurora path gives the same updates as the per-parameter scalar path, up
    to kernel rounding. The tolerance is set by how closely batched Muon matches scalar Muon on the
    same parameters and gradients, since that is the same kind of change for plain Muon."""
    from muon.muon import SingleDeviceMuonWithAuxAdam
    from benchmark_fresh_model import build_train_param_groups

    base, cfg = _make_base_model(seed=5)
    init = {n: p.detach().clone() for n, p in base.named_parameters()}
    steps = 4
    if grad_source == "real":
        grad_seq = _real_gradients(base, cfg, num_batches=steps)
    else:
        grad_seq = _synthetic_uneven_gradients(base, steps)

    def run(use_aurora, batched, steps=steps):
        model = copy.deepcopy(base)
        groups = build_train_param_groups(model)
        opt = SingleDeviceMuonWithAuxAdam(groups, adjust_lr_fn="match_rms_adamw", use_aurora=use_aurora)
        assert opt.use_batched_muon_ns, "expected the batched path to be the default"
        opt.use_batched_muon_ns = batched
        for g in opt.param_groups:
            g["lr"] = 1e-3
            g["weight_decay"] = 0.0
        muon_ids = {id(p) for g in groups if g["use_muon"] for p in g["params"]}
        for step in range(steps):
            for name, p in model.named_parameters():
                g = grad_seq[step].get(name)
                p.grad = g.clone() if g is not None else torch.zeros_like(p)
            opt.step()
        return {n: (p.detach() - init[n]) for n, p in model.named_parameters() if id(p) in muon_ids}

    # Warm up the compiled shapes first, since the first call of a freshly compiled shape has been
    # seen to differ slightly from later calls.
    run(True, True, steps=1)
    run(False, True, steps=1)
    aurora_scalar = run(True, False)
    aurora_batched = run(True, True)
    muon_scalar = run(False, False)
    muon_batched = run(False, True)
    assert len(aurora_scalar) > 100

    err_aurora = _grad_rel(aurora_batched, aurora_scalar)
    err_muon = _grad_rel(muon_batched, muon_scalar)
    aurora_vs_muon = _grad_rel(aurora_batched, muon_batched)
    print(f"batched vs scalar: aurora {err_aurora:.3e}, muon {err_muon:.3e}; aurora vs muon {aurora_vs_muon:.3e}")
    # On real gradients the eager scalar path is itself far less accurate than the batched path
    # (see test_aurora_polar_accuracy_against_fp32_reference), so the batched-vs-scalar difference is
    # mostly the scalar path's own error. That test also checks against a Muon fallback.
    assert err_aurora <= 2.0 * err_muon + 1e-3, (err_aurora, err_muon)

    # Aurora's defining property: in the tall orientation of each non-square matrix, the rows of its
    # update have more uniform norms than the rows of the Muon update.
    def row_norm_spread(deltas):
        spreads = []
        for d in deltas.values():
            m = d.reshape(d.shape[0], -1)
            if m.shape[0] == m.shape[1]:
                continue
            if m.shape[0] < m.shape[1]:
                m = m.mT
            r = m.norm(dim=1)
            spreads.append((r.std() / r.mean()).item())
        return sum(spreads) / len(spreads)

    # Single steps, since a sum of several updates with equal row norms does not have equal row norms.
    spread_aurora_batched = row_norm_spread(run(True, True, steps=1))
    spread_aurora_scalar = row_norm_spread(run(True, False, steps=1))
    spread_muon = row_norm_spread(run(False, True, steps=1))
    print(f"row norm spread: aurora batched {spread_aurora_batched:.3f}, scalar {spread_aurora_scalar:.3f}, muon {spread_muon:.3f}")
    assert spread_aurora_batched < 0.5 * spread_muon, (spread_aurora_batched, spread_muon)
    assert spread_aurora_batched == pytest.approx(spread_aurora_scalar, rel=0.1)


@pytest.mark.skipif(cuda_count < 1, reason="requires a CUDA device")
def test_aurora_ignores_normuon_on_batched_path():
    """With use_normuon also set, batched Aurora gives the same updates as plain batched Aurora.
    The two runs take different branches that round the scale into the update in a different order,
    so they agree to bf16 rounding rather than bitwise. NorMuon's row normalization would differ by
    far more than that."""
    from muon.muon import SingleDeviceMuonWithAuxAdam
    from benchmark_fresh_model import build_train_param_groups

    base, _ = _make_base_model(seed=13)
    init = {n: p.detach().clone() for n, p in base.named_parameters()}
    grad_seq = _synthetic_uneven_gradients(base, 2)

    def run(use_normuon):
        model = copy.deepcopy(base)
        groups = build_train_param_groups(model)
        opt = SingleDeviceMuonWithAuxAdam(groups, adjust_lr_fn="match_rms_adamw", use_aurora=True, use_normuon=use_normuon)
        for g in opt.param_groups:
            g["lr"] = 1e-3
            g["weight_decay"] = 0.0
        muon_ids = {id(p) for g in groups if g["use_muon"] for p in g["params"]}
        for step_grads in grad_seq:
            for name, p in model.named_parameters():
                p.grad = step_grads[name].clone()
            opt.step()
        return {n: (p.detach() - init[n]) for n, p in model.named_parameters() if id(p) in muon_ids}

    run(False)  # warm up the compiled shapes
    plain = run(False)
    with_normuon = run(True)
    err = _grad_rel(with_normuon, plain)
    print(f"aurora with normuon flag vs plain aurora: {err:.3e}")
    assert err < 1e-2, err


def _muon_gather_worker(rank, world, port, result_queue, use_aurora=False):
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
            opt = MuonWithAuxAdam(build_train_param_groups(model), adjust_lr_fn="match_rms_adamw", use_aurora=use_aurora)
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
@pytest.mark.parametrize("use_aurora", [False, True])
def test_muon_bf16_update_gather_is_bitwise_identical(use_aurora):
    import torch.multiprocessing as mp

    ctx = mp.get_context("spawn")
    result_queue = ctx.Queue()
    world = 2
    port = 23518 if use_aurora else 23517
    ctx_procs = mp.spawn(_muon_gather_worker, nprocs=world, args=(world, port, result_queue, use_aurora), join=False, start_method="spawn")
    results = [result_queue.get(timeout=1200) for _ in range(world)]
    ctx_procs.join()
    for rank, n_unstable, n_diff, cross_rank_equal in results:
        assert n_unstable == 0, f"rank {rank}: {n_unstable} parameter tensors differ between two identical fp32-gather runs"
        assert n_diff == 0, f"rank {rank}: {n_diff} parameter tensors differ between gather modes"
        assert cross_rank_equal, f"rank {rank}: parameters differ from rank 0"
