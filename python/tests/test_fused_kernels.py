"""
Tests for the fused training-time kernels/paths in the transformer blocks:

  1. katago.train.fused_swiglu: the Triton dual-GEMM + SwiGLU op matches the plain
     two-GEMM + pointwise reference in forward and backward (fp16 and bf16), to within
     the error of the fp16 reference itself relative to an fp64 computation.
  2. TransformerAttentionBlock.fused_qkv_proj and TransformerFFNBlock.fused_gate_proj
     (one GEMM over concatenated weights) give the same outputs and parameter gradients as
     the separate-projection path, in fp32.
  3. TransformerFFNBlock under fp16 autocast with the fused SwiGLU kernel matches the
     autocast path without it, relative to an fp32 reference.
  4. katago.train.fused_rope: the learnable-RoPE op with the fused Triton backward matches
     autograd of the plain rotation, to within rounding (the kernels keep cos/sin in fp32).

These need a CUDA device and are skipped otherwise.
"""

import os

import pytest
import torch

cuda_available = torch.cuda.is_available()
pytestmark = pytest.mark.skipif(not cuda_available, reason="requires CUDA")


def _rel(a, b):
    return ((a.double() - b.double()).norm() / (b.double().norm() + 1e-30)).item()


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("m", [4096, 4096 + 37])  # second one exercises the row mask
def test_fused_swiglu_matches_reference(dtype, m):
    from katago.train.fused_swiglu import fused_swiglu, reference_swiglu

    torch.manual_seed(0)
    dev = torch.device("cuda")
    k, n = 384, 1152
    x32 = torch.randn(m, k, device=dev)
    w1_32 = torch.randn(n, k, device=dev) / k**0.5
    w2_32 = torch.randn(n, k, device=dev) / k**0.5
    dh = torch.randn(m, n, device=dev).to(dtype)

    def inputs(cast):
        return [t.to(dtype).to(cast).clone().requires_grad_(True) for t in (x32, w1_32, w2_32)]

    xf, w1f, w2f = inputs(dtype)
    hf = fused_swiglu(xf, w1f, w2f)
    hf.backward(dh)

    xr, w1r, w2r = inputs(dtype)
    hr = reference_swiglu(xr, w1r, w2r)
    hr.backward(dh)

    x64, w164, w264 = inputs(torch.float64)
    h64 = reference_swiglu(x64, w164, w264)
    h64.backward(dh.double())

    # The fused op keeps the intermediates in fp32, so it should be no worse than the
    # low-precision reference relative to fp64 (allow 1.5x slack for summation order).
    for name, f, r, exact in (
        ("out", hf, hr, h64),
        ("dx", xf.grad, xr.grad, x64.grad),
        ("dw1", w1f.grad, w1r.grad, w164.grad),
        ("dw2", w2f.grad, w2r.grad, w264.grad),
    ):
        err_f = _rel(f, exact)
        err_r = _rel(r, exact)
        assert err_f <= 1.5 * err_r + 1e-6, f"{name}: fused err {err_f} vs reference err {err_r}"


def _make_blocks(dev):
    from katago.train import modelconfigs
    from katago.train.model_pytorch import TransformerAttentionBlock, TransformerFFNBlock

    config = modelconfigs.config_of_name["b11c768h12nbt3tflrs-fson-silu"].copy()
    pos_len = 19
    torch.manual_seed(0)
    attn = TransformerAttentionBlock("attn", 384, config, "silu", pos_len, use_rope=True).to(dev)
    ffn = TransformerFFNBlock("ffn", 384, config, "silu", use_swiglu=True).to(dev)
    # Give the (identity-initialized) norms and zero-initialized projections nontrivial values.
    with torch.no_grad():
        for p in list(attn.parameters()) + list(ffn.parameters()):
            p.add_(torch.randn_like(p) * 0.05)
    return attn, ffn


def _board_inputs(dev, batch=6, pos_len=19):
    torch.manual_seed(1)
    x = torch.randn(batch, 384, pos_len, pos_len, device=dev)
    mask = torch.ones(batch, 1, pos_len, pos_len, device=dev)
    mask[0, :, 9:, :] = 0.0  # one smaller board
    mask[0, :, :, 9:] = 0.0
    mask_sum_hw = mask.sum(dim=(2, 3), keepdim=True)
    mask_sum = mask.sum()
    return x, mask, mask_sum_hw, mask_sum


def _run_blocks(attn, ffn, fused, dev, amp=False, use_kernel=False):
    attn.fused_qkv_proj = fused
    ffn.fused_gate_proj = fused
    ffn.fused_swiglu_kernel = use_kernel
    x, mask, mask_sum_hw, mask_sum = _board_inputs(dev)
    for p in list(attn.parameters()) + list(ffn.parameters()):
        p.grad = None
    with torch.amp.autocast("cuda", dtype=torch.float16, enabled=amp):
        out = attn(x, mask=mask, mask_sum_hw=mask_sum_hw, mask_sum=mask_sum, extra_outputs=None)
        out = out + ffn(out, mask=mask, mask_sum_hw=mask_sum_hw, mask_sum=mask_sum, extra_outputs=None)
    out = out.float()
    out.square().mean().backward()
    grads = {n: p.grad.detach().clone() for n, p in list(attn.named_parameters()) + list(ffn.named_parameters())}
    return out.detach(), grads


def _grad_rel(ga, gb):
    d = sum((ga[n] - gb[n]).double().square().sum() for n in ga).sqrt()
    a = sum(gb[n].double().square().sum() for n in gb).sqrt()
    return (d / a).item()


def test_fused_projections_match_separate_fp32():
    dev = torch.device("cuda")
    prev = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        attn, ffn = _make_blocks(dev)
        out_sep, grads_sep = _run_blocks(attn, ffn, fused=False, dev=dev)
        out_fused, grads_fused = _run_blocks(attn, ffn, fused=True, dev=dev)
    finally:
        torch.backends.cuda.matmul.allow_tf32 = prev
    assert _rel(out_fused, out_sep) < 1e-5
    assert _grad_rel(grads_fused, grads_sep) < 1e-5


def test_fused_swiglu_kernel_in_ffn_block_under_autocast():
    dev = torch.device("cuda")
    prev = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        attn, ffn = _make_blocks(dev)
        out32, grads32 = _run_blocks(attn, ffn, fused=True, dev=dev, amp=False)
        out_amp, grads_amp = _run_blocks(attn, ffn, fused=True, dev=dev, amp=True, use_kernel=False)
        out_k, grads_k = _run_blocks(attn, ffn, fused=True, dev=dev, amp=True, use_kernel=True)
    finally:
        torch.backends.cuda.matmul.allow_tf32 = prev
    # Kernel path is no worse than the plain autocast path, relative to fp32.
    assert _rel(out_k, out32) <= 1.5 * _rel(out_amp, out32) + 1e-5
    assert _grad_rel(grads_k, grads32) <= 1.5 * _grad_rel(grads_amp, grads32) + 1e-5
    # And it is genuinely close to the plain autocast path.
    assert _rel(out_k, out_amp) < 1e-2


@pytest.mark.parametrize("dtype,batch,pos_len,heads,head_dim,v_head_dim", [
    (torch.float32, 6, 19, 12, 32, 32),
    (torch.float16, 6, 19, 12, 32, 32),
    (torch.float16, 5, 19, 6, 32, 32),   # odd batch exercises the batch-chunk tail of the backward
    (torch.bfloat16, 3, 9, 6, 32, 16),   # value head dim != query head dim, small board
])
def test_fused_rope_backward_matches_autograd(dtype, batch, pos_len, heads, head_dim, v_head_dim):
    from katago.train.fused_rope import learnable_rope_qkv
    from katago.train.model_pytorch import compute_learnable_rope_cos_sin, apply_learnable_rotary_emb

    torch.manual_seed(0)
    dev = torch.device("cuda")
    seq = pos_len * pos_len

    def reference(qkv, freqs):
        hd = heads * head_dim
        q = qkv[..., :hd].view(batch, seq, heads, head_dim)
        k = qkv[..., hd:2 * hd].view(batch, seq, heads, head_dim)
        v = qkv[..., 2 * hd:].view(batch, seq, heads, v_head_dim)
        s_idx = torch.arange(seq, device=dev)
        cos, sin = compute_learnable_rope_cos_sin((s_idx % pos_len).float(), (s_idx // pos_len).float(), freqs)
        q, k = apply_learnable_rotary_emb(q, k, cos, sin, cos, sin)
        return q, k, v.permute(0, 2, 1, 3)

    qkv0 = torch.randn(batch, seq, 2 * heads * head_dim + heads * v_head_dim, device=dev, dtype=dtype)
    freqs0 = torch.randn(heads, head_dim // 2, 2, device=dev) * 0.3
    gq = torch.randn(batch, heads, seq, head_dim, device=dev, dtype=dtype)
    gk = torch.randn_like(gq)
    gv = torch.randn(batch, heads, seq, v_head_dim, device=dev, dtype=dtype)

    def run(fn):
        qkv = qkv0.clone().requires_grad_(True)
        freqs = freqs0.clone().requires_grad_(True)
        q, k, v = fn(qkv, freqs)
        # Consumers see (B, H, S, D), as attention does.
        loss = (q.permute(0, 2, 1, 3).float() * gq.float()).sum() + (k.permute(0, 2, 1, 3).float() * gk.float()).sum() + (v.float() * gv.float()).sum()
        loss.backward()
        return [q.detach(), k.detach(), v.detach()], qkv.grad.clone(), freqs.grad.clone()

    outs_r, gqkv_r, gf_r = run(reference)

    def fused(a, b):
        q, k, v = learnable_rope_qkv(a, b, pos_len, heads, head_dim, v_head_dim)  # (B, H, S, D) each
        return q.permute(0, 2, 1, 3), k.permute(0, 2, 1, 3), v

    outs_f, gqkv_f, gf_f = run(fused)
    tol = {torch.float32: 1e-5, torch.float16: 2e-3, torch.bfloat16: 1.5e-2}[dtype]
    for a, b in zip(outs_f, outs_r):
        assert _rel(a, b) < tol
    assert _rel(gqkv_f, gqkv_r) < tol
    assert _rel(gf_f, gf_r) < tol


@pytest.mark.parametrize("model_kind,batch,per_block", [
    ("b10c384h6nbttflrs-fson-silu", 128, True),
    ("b10c384h6nbttflrs-fson-silu", 128, False),
])
def test_compiled_model_matches_eager(model_kind, batch, per_block, monkeypatch):
    """A model compiled the way training compiles it gives the same gradients as eager, with the
    per-block trunk (fused kernels on) and as a single graph (fused attention kernels turned off by
    configure_model_for_compile). Guards against inductor copying the RoPE custom op outputs into
    channels-last order for flex attention in a graph with convolutions, which made the flex attention
    backward return garbage (NaN with dirty memory) for this 6-head model at batch sizes >= 128."""
    from torch.amp import autocast
    from katago.train import modelconfigs
    from katago.train.model_pytorch import Model, TransformerAttentionBlock
    from katago.train.trainloop_helpers import wrap_model_for_training
    from benchmark_fresh_model import load_batch

    monkeypatch.setenv("KATAGO_COMPILE_PER_BLOCK", "1" if per_block else "0")
    dev = torch.device("cuda")
    torch.manual_seed(3)
    cfg = modelconfigs.config_of_name[model_kind].copy()
    model = Model(cfg, 19)
    model.initialize()
    model.to(dev)
    model.train()
    data = load_batch(os.path.join(os.path.dirname(__file__), "..", "testdata", "benchmark_data_1024.npz"), batch, 19, cfg, dev)
    # Fill freed memory with NaN so that any read of uninitialized memory shows up.
    junk = torch.empty(torch.cuda.mem_get_info()[0] // 4, dtype=torch.float16, device=dev)
    junk.fill_(float("nan"))
    del junk

    def run(fn, amp):
        model.zero_grad(set_to_none=True)
        with autocast("cuda", dtype=torch.float16, enabled=amp):
            outs = fn(data["binaryInputNCHW"], data["globalInputNC"])
        loss = sum(o.float().square().mean() for heads in outs for o in heads)
        loss.backward()
        return {n: p.grad.detach().clone() for n, p in model.named_parameters() if p.grad is not None}

    grads_fp32 = run(model, amp=False)
    grads_eager = run(model, amp=True)
    compiled = wrap_model_for_training(model, dev, world_size=1, no_compile=False)
    assert model.compile_per_block_trunk == per_block
    fused_rope_active = any(m.fused_rope_backward for m in model.modules() if isinstance(m, TransformerAttentionBlock))
    assert fused_rope_active == per_block
    grads_compiled = run(compiled, amp=True)
    grads_compiled = run(compiled, amp=True)
    nonfinite = [n for n, g in grads_compiled.items() if not torch.isfinite(g).all()]
    assert not nonfinite, nonfinite[:5]
    # No worse than eager fp16 relative to fp32 (both differ from it by fp16 rounding noise).
    err_compiled = _grad_rel(grads_compiled, grads_fp32)
    err_eager = _grad_rel(grads_eager, grads_fp32)
    assert err_compiled <= 1.5 * err_eager + 1e-4, (err_compiled, err_eager)
