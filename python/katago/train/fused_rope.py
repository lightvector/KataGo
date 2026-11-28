"""Learnable 2D RoPE over a packed Q/K/V projection as a pair of Triton kernels (custom ops).

Forward: one kernel reads the packed projection (B, S, Cq+Ck+Cv), rotates Q and K, and writes
Q, K, V as dense (B, H, S, head dim) tensors, i.e. the layout attention consumes. Backward: one
kernel, per (head, position tile, batch chunk), reads the three attention gradients
(B, H, S, head dim) plus the saved packed projection, and writes the un-rotated Q/K gradients and
the V gradient directly into the packed gradient buffer (B, S, Cq+Ck+Cv) that the projection
GEMM's backward consumes. The frequency gradients are accumulated per program into a small fp32
partials buffer and summed afterwards, so the result is deterministic (no atomics).

Without these kernels, autograd of the plain PyTorch rotation produces a batch-reduction kernel
for the frequency gradients plus a separate permute/concatenate kernel for the Q/K/V gradients,
each re-reading the same tensors. At trunk width 384 that is ~0.6 ms per attention layer of pure
memory traffic.

Both directions are torch custom ops (opaque to torch.compile), and the forward op carries the
autograd rule that calls the backward op. Keeping the forward as plain PyTorch inside a
torch.autograd.Function so that inductor can fuse it does not work: under torch.compile the
backward custom op then receives stale (zero) gradient buffers in some graphs.

The rotation is computed in fp32 and rounded once on store, whichever dtype the projection has.
The plain PyTorch path instead rounds cos/sin to the input dtype first when
KATAGO_LEARNED_ROPE_CAST_TO_INPUT_DTYPE is set, so the two differ at fp16 rounding level.

Inductor caveat: when this op's outputs feed flex attention in a compiled graph that also
contains convolutions, inductor's convolution layout optimization copies the (4D) outputs into
channels-last order for the flex kernels, and the flex attention backward template then produces
garbage gradients (observed on PyTorch 2.10 with 6 heads of dim 32). The plain PyTorch rotation
is not affected because its rotation intermediates are 5D, which the layout optimization leaves
alone. trainloop_helpers.configure_model_for_compile therefore uses this op only with the
per-block compiled trunk, whose graphs contain no convolutions.

Only the n_rep == 1 (no grouped-query) case without register tokens is handled. Callers fall back
to the plain path otherwise. The math matches apply_learnable_rotary_emb (interleaved pairs,
angle = fx*x + fy*y).
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _rope_qkv_fwd_kernel(
    QKV, FREQS, Q_OUT, K_OUT, V_OUT,
    S, W,
    H: tl.constexpr, D: tl.constexpr, DV: tl.constexpr,
    BLOCK_S: tl.constexpr,
):
    # QKV: (B, S, C) contiguous, C = 2*H*D + H*DV, columns [q | k | v]. FREQS: (H, P, 2) fp32.
    # Q_OUT, K_OUT: (B, H, S, D) contiguous; V_OUT: (B, H, S, DV) contiguous.
    P: tl.constexpr = D // 2
    C: tl.constexpr = 2 * H * D + H * DV
    h = tl.program_id(0)
    sb = tl.program_id(1)
    b = tl.program_id(2)

    offs_s = sb * BLOCK_S + tl.arange(0, BLOCK_S)
    mask_s = offs_s < S
    offs_p = tl.arange(0, P)
    offs_d = tl.arange(0, D)
    offs_dv = tl.arange(0, DV)
    mask_sd = mask_s[:, None] & (offs_d[None, :] < D)
    mask_sv = mask_s[:, None] & (offs_dv[None, :] < DV)

    fx = tl.load(FREQS + (h * P + offs_p) * 2)
    fy = tl.load(FREQS + (h * P + offs_p) * 2 + 1)
    s_x = (offs_s % W).to(tl.float32)
    s_y = (offs_s // W).to(tl.float32)
    angle = s_x[:, None] * fx[None, :] + s_y[:, None] * fy[None, :]
    c = tl.cos(angle)
    sn = tl.sin(angle)

    x_off = offs_s[:, None] * C + offs_d[None, :]
    o_off = offs_s[:, None] * D + offs_d[None, :]
    x_base = QKV + b * S * C
    o_base = (b * H + h) * S

    x = tl.load(x_base + h * D + x_off, mask=mask_sd, other=0.0).to(tl.float32)
    x0, x1 = tl.split(tl.reshape(x, (BLOCK_S, P, 2)))
    y = tl.reshape(tl.join(x0 * c - x1 * sn, x0 * sn + x1 * c), (BLOCK_S, D))
    tl.store(Q_OUT + o_base * D + o_off, y.to(Q_OUT.dtype.element_ty), mask=mask_sd)

    x = tl.load(x_base + H * D + h * D + x_off, mask=mask_sd, other=0.0).to(tl.float32)
    x0, x1 = tl.split(tl.reshape(x, (BLOCK_S, P, 2)))
    y = tl.reshape(tl.join(x0 * c - x1 * sn, x0 * sn + x1 * c), (BLOCK_S, D))
    tl.store(K_OUT + o_base * D + o_off, y.to(K_OUT.dtype.element_ty), mask=mask_sd)

    v = tl.load(x_base + 2 * H * D + h * DV + offs_s[:, None] * C + offs_dv[None, :], mask=mask_sv, other=0.0)
    tl.store(V_OUT + o_base * DV + offs_s[:, None] * DV + offs_dv[None, :], v, mask=mask_sv)


@triton.jit
def _rope_qkv_bwd_kernel(
    GQ, GK, GV, QKV, FREQS, GRAD_QKV, PARTIALS,
    B, S, W, num_s_blocks,
    gq_sb, gq_sh, gq_ss,
    gk_sb, gk_sh, gk_ss,
    gv_sb, gv_sh, gv_ss,
    H: tl.constexpr, D: tl.constexpr, DV: tl.constexpr,
    BLOCK_S: tl.constexpr, BLOCK_B: tl.constexpr,
):
    # GQ, GK: (B, H, S, D) and GV: (B, H, S, DV), with arbitrary (b, h, s) strides and unit d stride.
    # QKV, GRAD_QKV: (B, S, C) contiguous with C = 2*H*D + H*DV, columns [q | k | v].
    # FREQS: (H, P, 2) fp32 with P = D // 2. PARTIALS: (num_b_chunks * num_s_blocks, H, P, 2) fp32,
    # one slot per program.
    P: tl.constexpr = D // 2
    C: tl.constexpr = 2 * H * D + H * DV
    h = tl.program_id(0)
    sb = tl.program_id(1)
    bc = tl.program_id(2)

    offs_s = sb * BLOCK_S + tl.arange(0, BLOCK_S)
    mask_s = offs_s < S
    offs_p = tl.arange(0, P)
    offs_dv = tl.arange(0, DV)
    mask_sv = mask_s[:, None] & (offs_dv[None, :] < DV)

    fx = tl.load(FREQS + (h * P + offs_p) * 2)
    fy = tl.load(FREQS + (h * P + offs_p) * 2 + 1)
    s_x = (offs_s % W).to(tl.float32)
    s_y = (offs_s // W).to(tl.float32)
    angle = s_x[:, None] * fx[None, :] + s_y[:, None] * fy[None, :]
    c = tl.cos(angle)
    sn = tl.sin(angle)

    # Full-width (BLOCK_S, D) tiles are loaded contiguously and split into the interleaved
    # (even, odd) pair components with tl.split. Results are re-interleaved with tl.join.
    offs_d = tl.arange(0, D)
    mask_sd = mask_s[:, None] & (offs_d[None, :] < D)
    gq_off = offs_s[:, None] * gq_ss + offs_d[None, :]
    gk_off = offs_s[:, None] * gk_ss + offs_d[None, :]
    gv_off = offs_s[:, None] * gv_ss + offs_dv[None, :]
    x_off = offs_s[:, None] * C + offs_d[None, :]
    v_off = offs_s[:, None] * C + offs_dv[None, :]

    dtheta = tl.zeros((BLOCK_S, P), dtype=tl.float32)
    for bi in range(BLOCK_B):
        b = bc * BLOCK_B + bi
        valid = b < B
        m_sd = mask_sd & valid
        m_sv = mask_sv & valid
        gq_base = GQ + b * gq_sb + h * gq_sh
        gk_base = GK + b * gk_sb + h * gk_sh
        gv_base = GV + b * gv_sb + h * gv_sh
        x_base = QKV + b * S * C
        o_base = GRAD_QKV + b * S * C

        # Q
        g = tl.load(gq_base + gq_off, mask=m_sd, other=0.0).to(tl.float32)
        x = tl.load(x_base + h * D + x_off, mask=m_sd, other=0.0).to(tl.float32)
        g0, g1 = tl.split(tl.reshape(g, (BLOCK_S, P, 2)))
        x0, x1 = tl.split(tl.reshape(x, (BLOCK_S, P, 2)))
        gx0 = g0 * c + g1 * sn
        gx1 = g1 * c - g0 * sn
        dtheta += g1 * (x0 * c - x1 * sn) - g0 * (x0 * sn + x1 * c)
        gx = tl.reshape(tl.join(gx0, gx1), (BLOCK_S, D))
        tl.store(o_base + h * D + x_off, gx.to(GRAD_QKV.dtype.element_ty), mask=m_sd)

        # K (columns offset by H*D)
        g = tl.load(gk_base + gk_off, mask=m_sd, other=0.0).to(tl.float32)
        x = tl.load(x_base + H * D + h * D + x_off, mask=m_sd, other=0.0).to(tl.float32)
        g0, g1 = tl.split(tl.reshape(g, (BLOCK_S, P, 2)))
        x0, x1 = tl.split(tl.reshape(x, (BLOCK_S, P, 2)))
        gx0 = g0 * c + g1 * sn
        gx1 = g1 * c - g0 * sn
        dtheta += g1 * (x0 * c - x1 * sn) - g0 * (x0 * sn + x1 * c)
        gx = tl.reshape(tl.join(gx0, gx1), (BLOCK_S, D))
        tl.store(o_base + H * D + h * D + x_off, gx.to(GRAD_QKV.dtype.element_ty), mask=m_sd)

        # V (plain relayout, columns offset by 2*H*D)
        gv = tl.load(gv_base + gv_off, mask=m_sv, other=0.0)
        tl.store(o_base + 2 * H * D + h * DV + v_off, gv, mask=m_sv)

    gfx = tl.sum(dtheta * s_x[:, None], axis=0)
    gfy = tl.sum(dtheta * s_y[:, None], axis=0)
    pidx = bc * num_s_blocks + sb
    tl.store(PARTIALS + ((pidx * H + h) * P + offs_p) * 2, gfx)
    tl.store(PARTIALS + ((pidx * H + h) * P + offs_p) * 2 + 1, gfy)


# Tuned on RTX PRO 6000 Blackwell for B=224, S=361, H=12, D=32. Both kernels are memory-bound,
# and most configs are within a few percent of each other.
_FWD_CONFIG = dict(BLOCK_S=32, num_warps=4, num_stages=2)
_BWD_CONFIG = dict(BLOCK_S=32, BLOCK_B=2, num_warps=4, num_stages=2)


def _check_dims(num_heads, head_dim, v_head_dim, freqs):
    assert head_dim % 2 == 0 and (head_dim & (head_dim - 1)) == 0, "head dim must be a power of two"
    assert (v_head_dim & (v_head_dim - 1)) == 0, "value head dim must be a power of two"
    assert freqs.shape == (num_heads, head_dim // 2, 2)


def _round_up(n, multiple):
    return -(-n // multiple) * multiple


def _check_int32_addressable(t, padded_sizes):
    """The kernels compute element offsets in int32 for whole tiles, so every tensor they index
    must fit with its tiled dimensions rounded up to the tile size (masked positions included)."""
    max_offset = sum((size - 1) * stride for size, stride in zip(padded_sizes, t.stride()))
    assert max_offset < 2**31, f"tensor of shape {tuple(t.shape)} too large for int32 kernel indexing"


def _fwd_impl(qkv, freqs, pos_len, num_heads, head_dim, v_head_dim):
    batch_size, seq_len, total_channels = qkv.shape
    assert total_channels == 2 * num_heads * head_dim + num_heads * v_head_dim
    _check_dims(num_heads, head_dim, v_head_dim, freqs)
    qkv = qkv.contiguous()
    _check_int32_addressable(qkv, (batch_size, _round_up(seq_len, _FWD_CONFIG["BLOCK_S"]), total_channels))
    freqs32 = freqs.detach().to(torch.float32).contiguous()
    q_out = torch.empty((batch_size, num_heads, seq_len, head_dim), dtype=qkv.dtype, device=qkv.device)
    k_out = torch.empty_like(q_out)
    v_out = torch.empty((batch_size, num_heads, seq_len, v_head_dim), dtype=qkv.dtype, device=qkv.device)
    cfg = _FWD_CONFIG
    grid = (num_heads, triton.cdiv(seq_len, cfg["BLOCK_S"]), batch_size)
    _rope_qkv_fwd_kernel[grid](
        qkv, freqs32, q_out, k_out, v_out,
        seq_len, pos_len,
        H=num_heads, D=head_dim, DV=v_head_dim,
        BLOCK_S=cfg["BLOCK_S"], num_warps=cfg["num_warps"], num_stages=cfg["num_stages"],
    )
    return q_out, k_out, v_out


def _bwd_impl(gq, gk, gv, qkv, freqs, pos_len, num_heads, head_dim, v_head_dim):
    batch_size, seq_len, total_channels = qkv.shape
    assert total_channels == 2 * num_heads * head_dim + num_heads * v_head_dim
    _check_dims(num_heads, head_dim, v_head_dim, freqs)
    for g, last in ((gq, head_dim), (gk, head_dim), (gv, v_head_dim)):
        assert g.shape == (batch_size, num_heads, seq_len, last) and g.dtype == qkv.dtype
    # The gradients are read through their own strides. They have shape (B, H, S, D) but may be
    # laid out in memory as (B, S, H, D), as when attention's backward returns a permuted view.
    # Only a strided head dimension forces a copy.
    def unit_last_stride(t):
        return t if t.stride(-1) == 1 else t.contiguous()
    gq = unit_last_stride(gq)
    gk = unit_last_stride(gk)
    gv = unit_last_stride(gv)
    qkv = qkv.contiguous()
    cfg = _BWD_CONFIG
    padded_batch = _round_up(batch_size, cfg["BLOCK_B"])
    padded_seq = _round_up(seq_len, cfg["BLOCK_S"])
    for g, last in ((gq, head_dim), (gk, head_dim), (gv, v_head_dim)):
        _check_int32_addressable(g, (padded_batch, num_heads, padded_seq, last))
    _check_int32_addressable(qkv, (padded_batch, padded_seq, total_channels))
    freqs32 = freqs.detach().to(torch.float32).contiguous()

    num_s_blocks = triton.cdiv(seq_len, cfg["BLOCK_S"])
    num_b_chunks = triton.cdiv(batch_size, cfg["BLOCK_B"])
    grad_qkv = torch.empty_like(qkv)
    partials = torch.empty(
        (num_b_chunks * num_s_blocks, num_heads, head_dim // 2, 2), dtype=torch.float32, device=qkv.device
    )
    grid = (num_heads, num_s_blocks, num_b_chunks)
    _rope_qkv_bwd_kernel[grid](
        gq, gk, gv, qkv, freqs32, grad_qkv, partials,
        batch_size, seq_len, pos_len, num_s_blocks,
        gq.stride(0), gq.stride(1), gq.stride(2),
        gk.stride(0), gk.stride(1), gk.stride(2),
        gv.stride(0), gv.stride(1), gv.stride(2),
        H=num_heads, D=head_dim, DV=v_head_dim,
        BLOCK_S=cfg["BLOCK_S"], BLOCK_B=cfg["BLOCK_B"],
        num_warps=cfg["num_warps"], num_stages=cfg["num_stages"],
    )
    grad_freqs = partials.sum(dim=0).to(freqs.dtype)
    return grad_qkv, grad_freqs


@torch.library.custom_op("katago::rope_qkv_backward", mutates_args=())
def rope_qkv_backward(
    gq: torch.Tensor, gk: torch.Tensor, gv: torch.Tensor, qkv: torch.Tensor, freqs: torch.Tensor,
    pos_len: int, num_heads: int, head_dim: int, v_head_dim: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    return _bwd_impl(gq, gk, gv, qkv, freqs, pos_len, num_heads, head_dim, v_head_dim)


@rope_qkv_backward.register_fake
def _(gq, gk, gv, qkv, freqs, pos_len, num_heads, head_dim, v_head_dim):
    # The real op always returns a contiguous gradient (it copies a non-contiguous qkv first).
    return torch.empty_like(qkv, memory_format=torch.contiguous_format), torch.empty_like(freqs)


@torch.library.custom_op("katago::rope_qkv_forward", mutates_args=())
def rope_qkv_forward(
    qkv: torch.Tensor, freqs: torch.Tensor, pos_len: int, num_heads: int, head_dim: int, v_head_dim: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return _fwd_impl(qkv, freqs, pos_len, num_heads, head_dim, v_head_dim)


@rope_qkv_forward.register_fake
def _(qkv, freqs, pos_len, num_heads, head_dim, v_head_dim):
    batch_size, seq_len, _ = qkv.shape
    q = qkv.new_empty((batch_size, num_heads, seq_len, head_dim))
    return q, torch.empty_like(q), qkv.new_empty((batch_size, num_heads, seq_len, v_head_dim))


def _setup_context(ctx, inputs, output):
    qkv, freqs, pos_len, num_heads, head_dim, v_head_dim = inputs
    ctx.save_for_backward(qkv, freqs)
    ctx.pos_len = pos_len
    ctx.num_heads = num_heads
    ctx.head_dim = head_dim
    ctx.v_head_dim = v_head_dim


def _backward(ctx, gq, gk, gv):
    qkv, freqs = ctx.saved_tensors
    grad_qkv, grad_freqs = rope_qkv_backward(
        gq, gk, gv, qkv, freqs, ctx.pos_len, ctx.num_heads, ctx.head_dim, ctx.v_head_dim,
    )
    return grad_qkv, grad_freqs, None, None, None, None


rope_qkv_forward.register_autograd(_backward, setup_context=_setup_context)


def learnable_rope_qkv(qkv, freqs, pos_len, num_heads, head_dim, v_head_dim):
    """qkv: (B, S, 2*H*D + H*DV) packed [q | k | v] projection; freqs: (H, D//2, 2).

    Returns (q_rot, k_rot, v), each dense (B, H, S, head dim), ready for attention.
    """
    return rope_qkv_forward(qkv, freqs, pos_len, num_heads, head_dim, v_head_dim)


# ---- RoPE with query-key norm folded in --------------------------------------------------------
#
# With attention_qk_norm the model applies an RMSNorm over the head dimension to the rotated Q and
# K. As separate ops that is two more read+write passes over Q and K in the forward and two in the
# backward, all memory-bound, plus one more saved copy of Q and K. The rotation does not change a
# vector's norm, so the kernels below normalize in registers while they rotate:
#   out = rot(x) * rsqrt(mean(x^2) + eps) * weight
# and the backward kernel applies the norm's gradient before un-rotating, accumulating the
# gradients of the two norm weights per program next to the frequency gradients.
#
# Everything is computed in fp32 from the packed projection and rounded once on store. The
# separate-op path rounds the rotated Q/K to the input dtype before normalizing, so the two differ
# at fp16/bf16 rounding level. The plain kernels above are untouched (models without QK norm run
# exactly the code they ran before).


@triton.jit
def _rope_qkn_fwd_kernel(
    QKV, FREQS, QW, KW, Q_OUT, K_OUT, V_OUT,
    S, W, EPS,
    H: tl.constexpr, D: tl.constexpr, DV: tl.constexpr,
    BLOCK_S: tl.constexpr,
):
    # As _rope_qkv_fwd_kernel, plus QW, KW: (D,) fp32 RMSNorm weights for Q and K.
    P: tl.constexpr = D // 2
    C: tl.constexpr = 2 * H * D + H * DV
    h = tl.program_id(0)
    sb = tl.program_id(1)
    b = tl.program_id(2)

    offs_s = sb * BLOCK_S + tl.arange(0, BLOCK_S)
    mask_s = offs_s < S
    offs_p = tl.arange(0, P)
    offs_d = tl.arange(0, D)
    offs_dv = tl.arange(0, DV)
    mask_sd = mask_s[:, None] & (offs_d[None, :] < D)
    mask_sv = mask_s[:, None] & (offs_dv[None, :] < DV)

    fx = tl.load(FREQS + (h * P + offs_p) * 2)
    fy = tl.load(FREQS + (h * P + offs_p) * 2 + 1)
    s_x = (offs_s % W).to(tl.float32)
    s_y = (offs_s // W).to(tl.float32)
    angle = s_x[:, None] * fx[None, :] + s_y[:, None] * fy[None, :]
    c = tl.cos(angle)
    sn = tl.sin(angle)
    qw = tl.load(QW + offs_d)
    kw = tl.load(KW + offs_d)

    x_off = offs_s[:, None] * C + offs_d[None, :]
    o_off = offs_s[:, None] * D + offs_d[None, :]
    x_base = QKV + b * S * C
    o_base = (b * H + h) * S

    x = tl.load(x_base + h * D + x_off, mask=mask_sd, other=0.0).to(tl.float32)
    n = tl.rsqrt(tl.sum(x * x, axis=1) / D + EPS)
    x0, x1 = tl.split(tl.reshape(x, (BLOCK_S, P, 2)))
    y = tl.reshape(tl.join(x0 * c - x1 * sn, x0 * sn + x1 * c), (BLOCK_S, D))
    y = y * n[:, None] * qw[None, :]
    tl.store(Q_OUT + o_base * D + o_off, y.to(Q_OUT.dtype.element_ty), mask=mask_sd)

    x = tl.load(x_base + H * D + h * D + x_off, mask=mask_sd, other=0.0).to(tl.float32)
    n = tl.rsqrt(tl.sum(x * x, axis=1) / D + EPS)
    x0, x1 = tl.split(tl.reshape(x, (BLOCK_S, P, 2)))
    y = tl.reshape(tl.join(x0 * c - x1 * sn, x0 * sn + x1 * c), (BLOCK_S, D))
    y = y * n[:, None] * kw[None, :]
    tl.store(K_OUT + o_base * D + o_off, y.to(K_OUT.dtype.element_ty), mask=mask_sd)

    v = tl.load(x_base + 2 * H * D + h * DV + offs_s[:, None] * C + offs_dv[None, :], mask=mask_sv, other=0.0)
    tl.store(V_OUT + o_base * DV + offs_s[:, None] * DV + offs_dv[None, :], v, mask=mask_sv)


@triton.jit
def _rope_qkn_bwd_kernel(
    GQ, GK, GV, QKV, FREQS, QW, KW, GRAD_QKV, PARTIALS, PARTIALS_W,
    B, S, W, num_s_blocks, EPS,
    gq_sb, gq_sh, gq_ss,
    gk_sb, gk_sh, gk_ss,
    gv_sb, gv_sh, gv_ss,
    H: tl.constexpr, D: tl.constexpr, DV: tl.constexpr,
    BLOCK_S: tl.constexpr, BLOCK_B: tl.constexpr,
):
    # As _rope_qkv_bwd_kernel, with GQ and GK the gradients with respect to the normalized,
    # rotated Q and K. PARTIALS_W: (num_b_chunks * num_s_blocks * H, 2, D) fp32, one slot per
    # program, holding the gradient partials of the Q and K norm weights.
    P: tl.constexpr = D // 2
    C: tl.constexpr = 2 * H * D + H * DV
    h = tl.program_id(0)
    sb = tl.program_id(1)
    bc = tl.program_id(2)

    offs_s = sb * BLOCK_S + tl.arange(0, BLOCK_S)
    mask_s = offs_s < S
    offs_p = tl.arange(0, P)
    offs_dv = tl.arange(0, DV)
    mask_sv = mask_s[:, None] & (offs_dv[None, :] < DV)

    fx = tl.load(FREQS + (h * P + offs_p) * 2)
    fy = tl.load(FREQS + (h * P + offs_p) * 2 + 1)
    s_x = (offs_s % W).to(tl.float32)
    s_y = (offs_s // W).to(tl.float32)
    angle = s_x[:, None] * fx[None, :] + s_y[:, None] * fy[None, :]
    c = tl.cos(angle)
    sn = tl.sin(angle)

    offs_d = tl.arange(0, D)
    mask_sd = mask_s[:, None] & (offs_d[None, :] < D)
    gq_off = offs_s[:, None] * gq_ss + offs_d[None, :]
    gk_off = offs_s[:, None] * gk_ss + offs_d[None, :]
    gv_off = offs_s[:, None] * gv_ss + offs_dv[None, :]
    x_off = offs_s[:, None] * C + offs_d[None, :]
    v_off = offs_s[:, None] * C + offs_dv[None, :]
    qw = tl.load(QW + offs_d)
    kw = tl.load(KW + offs_d)

    dtheta = tl.zeros((BLOCK_S, P), dtype=tl.float32)
    dqw = tl.zeros((D,), dtype=tl.float32)
    dkw = tl.zeros((D,), dtype=tl.float32)
    for bi in range(BLOCK_B):
        b = bc * BLOCK_B + bi
        valid = b < B
        m_sd = mask_sd & valid
        m_sv = mask_sv & valid
        gq_base = GQ + b * gq_sb + h * gq_sh
        gk_base = GK + b * gk_sb + h * gk_sh
        gv_base = GV + b * gv_sb + h * gv_sh
        x_base = QKV + b * S * C
        o_base = GRAD_QKV + b * S * C

        # Q: out = r * n * w with r = rot(x), n = rsqrt(mean(x^2) + eps).
        # d/dr = n * (g*w) - n^3 * r * mean(g*w*r), d/dw = sum(g * r * n).
        g = tl.load(gq_base + gq_off, mask=m_sd, other=0.0).to(tl.float32)
        x = tl.load(x_base + h * D + x_off, mask=m_sd, other=0.0).to(tl.float32)
        n = tl.rsqrt(tl.sum(x * x, axis=1) / D + EPS)
        x0, x1 = tl.split(tl.reshape(x, (BLOCK_S, P, 2)))
        r0 = x0 * c - x1 * sn
        r1 = x0 * sn + x1 * c
        dqw += tl.sum(g * tl.reshape(tl.join(r0, r1), (BLOCK_S, D)) * n[:, None], axis=0)
        g0, g1 = tl.split(tl.reshape(g * qw[None, :], (BLOCK_S, P, 2)))
        coef = n * n * n * (tl.sum(g0 * r0 + g1 * r1, axis=1) / D)
        d0 = n[:, None] * g0 - coef[:, None] * r0
        d1 = n[:, None] * g1 - coef[:, None] * r1
        dtheta += d1 * r0 - d0 * r1
        gx = tl.reshape(tl.join(d0 * c + d1 * sn, d1 * c - d0 * sn), (BLOCK_S, D))
        tl.store(o_base + h * D + x_off, gx.to(GRAD_QKV.dtype.element_ty), mask=m_sd)

        # K (columns offset by H*D)
        g = tl.load(gk_base + gk_off, mask=m_sd, other=0.0).to(tl.float32)
        x = tl.load(x_base + H * D + h * D + x_off, mask=m_sd, other=0.0).to(tl.float32)
        n = tl.rsqrt(tl.sum(x * x, axis=1) / D + EPS)
        x0, x1 = tl.split(tl.reshape(x, (BLOCK_S, P, 2)))
        r0 = x0 * c - x1 * sn
        r1 = x0 * sn + x1 * c
        dkw += tl.sum(g * tl.reshape(tl.join(r0, r1), (BLOCK_S, D)) * n[:, None], axis=0)
        g0, g1 = tl.split(tl.reshape(g * kw[None, :], (BLOCK_S, P, 2)))
        coef = n * n * n * (tl.sum(g0 * r0 + g1 * r1, axis=1) / D)
        d0 = n[:, None] * g0 - coef[:, None] * r0
        d1 = n[:, None] * g1 - coef[:, None] * r1
        dtheta += d1 * r0 - d0 * r1
        gx = tl.reshape(tl.join(d0 * c + d1 * sn, d1 * c - d0 * sn), (BLOCK_S, D))
        tl.store(o_base + H * D + h * D + x_off, gx.to(GRAD_QKV.dtype.element_ty), mask=m_sd)

        # V (plain relayout, columns offset by 2*H*D)
        gv = tl.load(gv_base + gv_off, mask=m_sv, other=0.0)
        tl.store(o_base + 2 * H * D + h * DV + v_off, gv, mask=m_sv)

    gfx = tl.sum(dtheta * s_x[:, None], axis=0)
    gfy = tl.sum(dtheta * s_y[:, None], axis=0)
    pidx = bc * num_s_blocks + sb
    tl.store(PARTIALS + ((pidx * H + h) * P + offs_p) * 2, gfx)
    tl.store(PARTIALS + ((pidx * H + h) * P + offs_p) * 2 + 1, gfy)
    tl.store(PARTIALS_W + ((pidx * H + h) * 2) * D + offs_d, dqw)
    tl.store(PARTIALS_W + ((pidx * H + h) * 2 + 1) * D + offs_d, dkw)


def _qkn_fwd_impl(qkv, freqs, q_weight, k_weight, eps, pos_len, num_heads, head_dim, v_head_dim):
    batch_size, seq_len, total_channels = qkv.shape
    assert total_channels == 2 * num_heads * head_dim + num_heads * v_head_dim
    _check_dims(num_heads, head_dim, v_head_dim, freqs)
    assert q_weight.shape == (head_dim,) and k_weight.shape == (head_dim,)
    qkv = qkv.contiguous()
    _check_int32_addressable(qkv, (batch_size, _round_up(seq_len, _FWD_CONFIG["BLOCK_S"]), total_channels))
    freqs32 = freqs.detach().to(torch.float32).contiguous()
    qw32 = q_weight.detach().to(torch.float32).contiguous()
    kw32 = k_weight.detach().to(torch.float32).contiguous()
    q_out = torch.empty((batch_size, num_heads, seq_len, head_dim), dtype=qkv.dtype, device=qkv.device)
    k_out = torch.empty_like(q_out)
    v_out = torch.empty((batch_size, num_heads, seq_len, v_head_dim), dtype=qkv.dtype, device=qkv.device)
    cfg = _FWD_CONFIG
    grid = (num_heads, triton.cdiv(seq_len, cfg["BLOCK_S"]), batch_size)
    _rope_qkn_fwd_kernel[grid](
        qkv, freqs32, qw32, kw32, q_out, k_out, v_out,
        seq_len, pos_len, float(eps),
        H=num_heads, D=head_dim, DV=v_head_dim,
        BLOCK_S=cfg["BLOCK_S"], num_warps=cfg["num_warps"], num_stages=cfg["num_stages"],
    )
    return q_out, k_out, v_out


def _qkn_bwd_impl(gq, gk, gv, qkv, freqs, q_weight, k_weight, eps, pos_len, num_heads, head_dim, v_head_dim):
    batch_size, seq_len, total_channels = qkv.shape
    assert total_channels == 2 * num_heads * head_dim + num_heads * v_head_dim
    _check_dims(num_heads, head_dim, v_head_dim, freqs)
    for g, last in ((gq, head_dim), (gk, head_dim), (gv, v_head_dim)):
        assert g.shape == (batch_size, num_heads, seq_len, last) and g.dtype == qkv.dtype

    def unit_last_stride(t):
        return t if t.stride(-1) == 1 else t.contiguous()
    gq = unit_last_stride(gq)
    gk = unit_last_stride(gk)
    gv = unit_last_stride(gv)
    qkv = qkv.contiguous()
    cfg = _BWD_CONFIG
    padded_batch = _round_up(batch_size, cfg["BLOCK_B"])
    padded_seq = _round_up(seq_len, cfg["BLOCK_S"])
    for g, last in ((gq, head_dim), (gk, head_dim), (gv, v_head_dim)):
        _check_int32_addressable(g, (padded_batch, num_heads, padded_seq, last))
    _check_int32_addressable(qkv, (padded_batch, padded_seq, total_channels))
    freqs32 = freqs.detach().to(torch.float32).contiguous()
    qw32 = q_weight.detach().to(torch.float32).contiguous()
    kw32 = k_weight.detach().to(torch.float32).contiguous()

    num_s_blocks = triton.cdiv(seq_len, cfg["BLOCK_S"])
    num_b_chunks = triton.cdiv(batch_size, cfg["BLOCK_B"])
    grad_qkv = torch.empty_like(qkv)
    partials = torch.empty(
        (num_b_chunks * num_s_blocks, num_heads, head_dim // 2, 2), dtype=torch.float32, device=qkv.device
    )
    partials_w = torch.empty(
        (num_b_chunks * num_s_blocks * num_heads, 2, head_dim), dtype=torch.float32, device=qkv.device
    )
    grid = (num_heads, num_s_blocks, num_b_chunks)
    _rope_qkn_bwd_kernel[grid](
        gq, gk, gv, qkv, freqs32, qw32, kw32, grad_qkv, partials, partials_w,
        batch_size, seq_len, pos_len, num_s_blocks, float(eps),
        gq.stride(0), gq.stride(1), gq.stride(2),
        gk.stride(0), gk.stride(1), gk.stride(2),
        gv.stride(0), gv.stride(1), gv.stride(2),
        H=num_heads, D=head_dim, DV=v_head_dim,
        BLOCK_S=cfg["BLOCK_S"], BLOCK_B=cfg["BLOCK_B"],
        num_warps=cfg["num_warps"], num_stages=cfg["num_stages"],
    )
    grad_freqs = partials.sum(dim=0).to(freqs.dtype)
    grad_w = partials_w.sum(dim=0)
    # Separate tensors: custom op outputs must not alias each other.
    return grad_qkv, grad_freqs, grad_w[0].to(q_weight.dtype, copy=True), grad_w[1].to(k_weight.dtype, copy=True)


@torch.library.custom_op("katago::rope_qkn_backward", mutates_args=())
def rope_qkn_backward(
    gq: torch.Tensor, gk: torch.Tensor, gv: torch.Tensor, qkv: torch.Tensor, freqs: torch.Tensor,
    q_weight: torch.Tensor, k_weight: torch.Tensor, eps: float,
    pos_len: int, num_heads: int, head_dim: int, v_head_dim: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    return _qkn_bwd_impl(gq, gk, gv, qkv, freqs, q_weight, k_weight, eps, pos_len, num_heads, head_dim, v_head_dim)


@rope_qkn_backward.register_fake
def _(gq, gk, gv, qkv, freqs, q_weight, k_weight, eps, pos_len, num_heads, head_dim, v_head_dim):
    return (
        torch.empty_like(qkv, memory_format=torch.contiguous_format), torch.empty_like(freqs),
        torch.empty_like(q_weight), torch.empty_like(k_weight),
    )


@torch.library.custom_op("katago::rope_qkn_forward", mutates_args=())
def rope_qkn_forward(
    qkv: torch.Tensor, freqs: torch.Tensor, q_weight: torch.Tensor, k_weight: torch.Tensor, eps: float,
    pos_len: int, num_heads: int, head_dim: int, v_head_dim: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return _qkn_fwd_impl(qkv, freqs, q_weight, k_weight, eps, pos_len, num_heads, head_dim, v_head_dim)


@rope_qkn_forward.register_fake
def _(qkv, freqs, q_weight, k_weight, eps, pos_len, num_heads, head_dim, v_head_dim):
    batch_size, seq_len, _ = qkv.shape
    q = qkv.new_empty((batch_size, num_heads, seq_len, head_dim))
    return q, torch.empty_like(q), qkv.new_empty((batch_size, num_heads, seq_len, v_head_dim))


def _qkn_setup_context(ctx, inputs, output):
    qkv, freqs, q_weight, k_weight, eps, pos_len, num_heads, head_dim, v_head_dim = inputs
    ctx.save_for_backward(qkv, freqs, q_weight, k_weight)
    ctx.eps = eps
    ctx.pos_len = pos_len
    ctx.num_heads = num_heads
    ctx.head_dim = head_dim
    ctx.v_head_dim = v_head_dim


def _qkn_backward(ctx, gq, gk, gv):
    qkv, freqs, q_weight, k_weight = ctx.saved_tensors
    grad_qkv, grad_freqs, grad_qw, grad_kw = rope_qkn_backward(
        gq, gk, gv, qkv, freqs, q_weight, k_weight, ctx.eps,
        ctx.pos_len, ctx.num_heads, ctx.head_dim, ctx.v_head_dim,
    )
    return grad_qkv, grad_freqs, grad_qw, grad_kw, None, None, None, None, None


rope_qkn_forward.register_autograd(_qkn_backward, setup_context=_qkn_setup_context)


def learnable_rope_qkn(qkv, freqs, q_weight, k_weight, eps, pos_len, num_heads, head_dim, v_head_dim):
    """As learnable_rope_qkv, with an RMSNorm over the head dimension (weights q_weight, k_weight
    of shape (head_dim,), epsilon eps) applied to the rotated Q and K inside the same kernels."""
    return rope_qkn_forward(qkv, freqs, q_weight, k_weight, eps, pos_len, num_heads, head_dim, v_head_dim)
