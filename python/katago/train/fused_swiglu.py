"""Fused SwiGLU feed-forward input projection: h = silu(x @ W1^T) * (x @ W2^T) in one Triton kernel.

Motivation: in the nested-bottleneck transformer blocks the mid channel count is small (384 in
tf3-b11c768, for example), so the FFN GEMMs are memory-bandwidth-bound rather than compute-bound.
Computing the two projections as ordinary GEMMs writes both ffn-width intermediates to HBM, then
a separate pointwise kernel reads them back and writes the gated result. This kernel computes
both GEMM tiles in registers and writes only the gated result, and the backward recomputes the
two intermediates tile-wise from the saved input instead of reading saved copies. Net effect
is far less memory traffic (and no saved intermediates) at the price of recomputing both
projections in backward, which is cheap on a bandwidth-bound workload.

Numerics: accumulation is fp32 in the GEMM and the gating is applied in fp32 before the
single rounding to the output dtype, so results are at least as accurate as the unfused
path (which rounds the two intermediates to fp16 or bf16 first).

Exposed as torch custom ops (katago::fused_swiglu_fwd / katago::fused_swiglu_bwd_intermediates)
with an autograd rule, so they are opaque to torch.compile and work under DDP like any other op.
They have no autocast rule: the inputs must already be fp16 or bf16, and TransformerFFNBlock
casts them first. K (input channels) and N (ffn channels) must be multiples of the kernel tile
sizes, REQUIRED_K_MULTIPLE and REQUIRED_N_MULTIPLE below. TransformerFFNBlock falls back to
ordinary matmuls otherwise.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _dual_gemm_swiglu_fwd_kernel(
    A, W1, W2, H,
    M, N, K,
    stride_am, stride_ak,
    stride_wn, stride_wk,
    stride_hm, stride_hn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    # Grouped (swizzled) program ordering for L2 reuse of A across N tiles.
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    mask_m = offs_m < M

    a_ptrs = A + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    # W is (N, K) as in torch.nn.Linear, loaded here as a (BLOCK_K, BLOCK_N) tile of W^T.
    w1_ptrs = W1 + offs_k[:, None] * stride_wk + offs_n[None, :] * stride_wn
    w2_ptrs = W2 + offs_k[:, None] * stride_wk + offs_n[None, :] * stride_wn

    acc1 = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    acc2 = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for _ in range(0, tl.cdiv(K, BLOCK_K)):
        a = tl.load(a_ptrs, mask=mask_m[:, None], other=0.0)
        w1 = tl.load(w1_ptrs)
        w2 = tl.load(w2_ptrs)
        acc1 = tl.dot(a, w1, acc1)
        acc2 = tl.dot(a, w2, acc2)
        a_ptrs += BLOCK_K * stride_ak
        w1_ptrs += BLOCK_K * stride_wk
        w2_ptrs += BLOCK_K * stride_wk

    h = acc1 * tl.sigmoid(acc1) * acc2
    h_ptrs = H + offs_m[:, None] * stride_hm + offs_n[None, :] * stride_hn
    tl.store(h_ptrs, h.to(H.dtype.element_ty), mask=mask_m[:, None])


@triton.jit
def _dual_gemm_swiglu_bwd_kernel(
    A, W1, W2, DH, D13,
    M, N, K,
    stride_am, stride_ak,
    stride_wn, stride_wk,
    stride_dhm, stride_dhn,
    stride_dm, stride_dn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    # Recompute the two projection tiles, then write the gradients w.r.t. both
    # intermediates into the concatenated (M, 2N) buffer D13 = [d_x1 | d_gate].
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    mask_m = offs_m < M

    a_ptrs = A + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    w1_ptrs = W1 + offs_k[:, None] * stride_wk + offs_n[None, :] * stride_wn
    w2_ptrs = W2 + offs_k[:, None] * stride_wk + offs_n[None, :] * stride_wn

    acc1 = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    acc2 = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for _ in range(0, tl.cdiv(K, BLOCK_K)):
        a = tl.load(a_ptrs, mask=mask_m[:, None], other=0.0)
        w1 = tl.load(w1_ptrs)
        w2 = tl.load(w2_ptrs)
        acc1 = tl.dot(a, w1, acc1)
        acc2 = tl.dot(a, w2, acc2)
        a_ptrs += BLOCK_K * stride_ak
        w1_ptrs += BLOCK_K * stride_wk
        w2_ptrs += BLOCK_K * stride_wk

    dh_ptrs = DH + offs_m[:, None] * stride_dhm + offs_n[None, :] * stride_dhn
    dh = tl.load(dh_ptrs, mask=mask_m[:, None], other=0.0).to(tl.float32)
    s = tl.sigmoid(acc1)
    silu = acc1 * s
    dsilu = s * (1.0 + acc1 * (1.0 - s))
    d_x1 = dh * acc2 * dsilu
    d_gate = dh * silu
    d1_ptrs = D13 + offs_m[:, None] * stride_dm + offs_n[None, :] * stride_dn
    d2_ptrs = d1_ptrs + N * stride_dn
    tl.store(d1_ptrs, d_x1.to(D13.dtype.element_ty), mask=mask_m[:, None])
    tl.store(d2_ptrs, d_gate.to(D13.dtype.element_ty), mask=mask_m[:, None])


# Tile configuration, chosen by sweeping BLOCK_M/N/K, warps and stages on an RTX PRO 6000
# Blackwell (sm_120) for M ~ 80k tokens, K = 384, N = 1152. Two accumulators per program
# means BLOCK_N=64 with 4 warps has the register footprint of a normal 128x128 GEMM tile,
# and 128x128 tiles spill and run ~2x slower. Other GPUs may prefer different tiles.
_FWD_CONFIG = dict(BLOCK_M=128, BLOCK_N=64, BLOCK_K=32, GROUP_M=8, num_warps=4, num_stages=3)
_BWD_CONFIG = dict(BLOCK_M=128, BLOCK_N=64, BLOCK_K=32, GROUP_M=8, num_warps=4, num_stages=4)
# Neither kernel masks the K loop or the N tile, so K and N must be multiples of BLOCK_K and BLOCK_N.
# _check_inputs enforces one shared constraint, so keep the two configs' BLOCK_K/BLOCK_N equal or
# generalize it.
assert _FWD_CONFIG["BLOCK_K"] == _BWD_CONFIG["BLOCK_K"] and _FWD_CONFIG["BLOCK_N"] == _BWD_CONFIG["BLOCK_N"]
REQUIRED_K_MULTIPLE = _FWD_CONFIG["BLOCK_K"]
REQUIRED_N_MULTIPLE = _FWD_CONFIG["BLOCK_N"]


def is_supported_shape(in_channels: int, ffn_channels: int) -> bool:
    return in_channels % REQUIRED_K_MULTIPLE == 0 and ffn_channels % REQUIRED_N_MULTIPLE == 0


def _check_inputs(x: torch.Tensor, w13: torch.Tensor):
    assert x.dim() == 2 and w13.dim() == 2
    assert x.dtype == w13.dtype and x.dtype in (torch.float16, torch.bfloat16)
    assert x.is_cuda and w13.is_cuda
    assert x.shape[1] == w13.shape[1]
    assert w13.shape[0] % 2 == 0
    n = w13.shape[0] // 2
    k = x.shape[1]
    assert k % REQUIRED_K_MULTIPLE == 0, f"K={k} must be a multiple of {REQUIRED_K_MULTIPLE}"
    assert n % REQUIRED_N_MULTIPLE == 0, f"ffn dim {n} must be a multiple of {REQUIRED_N_MULTIPLE}"
    assert x.stride(1) == 1 and w13.stride(1) == 1
    # The kernels compute element offsets in int32 for whole tiles, so the row count is rounded up
    # to the tile height. The largest indexed tensors are x (through its strides) and the
    # contiguous (M, 2N) backward buffer.
    assert _FWD_CONFIG["BLOCK_M"] == _BWD_CONFIG["BLOCK_M"]
    padded_m = -(-x.shape[0] // _FWD_CONFIG["BLOCK_M"]) * _FWD_CONFIG["BLOCK_M"]
    assert (padded_m - 1) * x.stride(0) + k < 2**31 and padded_m * 2 * n < 2**31, \
        f"M={x.shape[0]}, K={k}, N={n} too large for int32 kernel indexing"


def _fwd_impl(x: torch.Tensor, w13: torch.Tensor) -> torch.Tensor:
    _check_inputs(x, w13)
    m, k = x.shape
    n = w13.shape[0] // 2
    w1 = w13[:n]
    w2 = w13[n:]
    h = torch.empty((m, n), dtype=x.dtype, device=x.device)
    cfg = _FWD_CONFIG
    grid = (triton.cdiv(m, cfg["BLOCK_M"]) * triton.cdiv(n, cfg["BLOCK_N"]),)
    _dual_gemm_swiglu_fwd_kernel[grid](
        x, w1, w2, h,
        m, n, k,
        x.stride(0), x.stride(1),
        w13.stride(0), w13.stride(1),
        h.stride(0), h.stride(1),
        **cfg,
    )
    return h


def _bwd_intermediates_impl(x: torch.Tensor, w13: torch.Tensor, dh: torch.Tensor) -> torch.Tensor:
    """Return d13 = [d_x1 | d_gate] of shape (M, 2N), recomputing x1/gate tile-wise."""
    _check_inputs(x, w13)
    m, k = x.shape
    n = w13.shape[0] // 2
    assert dh.shape == (m, n) and dh.dtype == x.dtype
    dh = dh.contiguous()
    w1 = w13[:n]
    w2 = w13[n:]
    d13 = torch.empty((m, 2 * n), dtype=x.dtype, device=x.device)
    cfg = _BWD_CONFIG
    grid = (triton.cdiv(m, cfg["BLOCK_M"]) * triton.cdiv(n, cfg["BLOCK_N"]),)
    _dual_gemm_swiglu_bwd_kernel[grid](
        x, w1, w2, dh, d13,
        m, n, k,
        x.stride(0), x.stride(1),
        w13.stride(0), w13.stride(1),
        dh.stride(0), dh.stride(1),
        d13.stride(0), d13.stride(1),
        **cfg,
    )
    return d13


@torch.library.custom_op("katago::fused_swiglu_fwd", mutates_args=())
def fused_swiglu_fwd(x: torch.Tensor, w13: torch.Tensor) -> torch.Tensor:
    return _fwd_impl(x, w13)


@fused_swiglu_fwd.register_fake
def _(x, w13):
    return x.new_empty((x.shape[0], w13.shape[0] // 2))


@torch.library.custom_op("katago::fused_swiglu_bwd_intermediates", mutates_args=())
def fused_swiglu_bwd_intermediates(x: torch.Tensor, w13: torch.Tensor, dh: torch.Tensor) -> torch.Tensor:
    return _bwd_intermediates_impl(x, w13, dh)


@fused_swiglu_bwd_intermediates.register_fake
def _(x, w13, dh):
    return x.new_empty((x.shape[0], w13.shape[0]))


def _setup_context(ctx, inputs, output):
    x, w13 = inputs
    ctx.save_for_backward(x, w13)


def _backward(ctx, dh):
    x, w13 = ctx.saved_tensors
    d13 = fused_swiglu_bwd_intermediates(x, w13, dh)
    # Ordinary GEMMs (cuBLAS) for the input and weight gradients, exactly as the
    # unfused concatenated-weight path would do them.
    dx = d13 @ w13
    dw13 = d13.t() @ x
    return dx, dw13


fused_swiglu_fwd.register_autograd(_backward, setup_context=_setup_context)


def fused_swiglu(x: torch.Tensor, w1: torch.Tensor, w2: torch.Tensor) -> torch.Tensor:
    """h = silu(x @ w1^T) * (x @ w2^T) for x of shape (..., K), w1/w2 of shape (N, K).

    Low precision only (fp16/bf16): callers must have applied the autocast dtype already.
    """
    lead_shape = x.shape[:-1]
    x2 = x.reshape(-1, x.shape[-1])
    if x2.stride(-1) != 1:
        x2 = x2.contiguous()
    w13 = torch.cat([w1, w2], dim=0)
    h = fused_swiglu_fwd(x2, w13)
    return h.reshape(*lead_shape, h.shape[-1])


def reference_swiglu(x: torch.Tensor, w1: torch.Tensor, w2: torch.Tensor) -> torch.Tensor:
    x1 = torch.nn.functional.linear(x, w1)
    gate = torch.nn.functional.linear(x, w2)
    return torch.nn.functional.silu(x1) * gate
