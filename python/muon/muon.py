import logging
import os
from dataclasses import dataclass
from itertools import repeat
from typing import List, Sequence, Tuple

import torch
import torch.distributed as dist


def _env_flag(name: str, default: bool) -> bool:
    """Read a boolean environment flag, accepting only the exact values 0 and 1."""
    value = os.environ.get(name)
    if value is None:
        return default
    if value == "0":
        return False
    if value == "1":
        return True
    raise ValueError(f"Environment variable {name} must be exactly '0' or '1', got {value!r}")


@torch.no_grad()
def floored_weight_decay_(w: torch.Tensor, a: float, floor_norm: float):
    """Weight decay by factor (1 - a) that leaves each output channel a norm floor.

    Output channel j is w[j], indexed by dim 0, which is one output row for nn.Linear [out, in]
    and conv [out, in, kh, kw] weights. Let r be the norm of w[j]. Only the part of r above
    floor_norm decays. The new norm is r - a * (r - floor_norm) when r > floor_norm, and r is
    unchanged otherwise. That is, w[j] *= 1 - a * max(0, r - floor_norm) / r.

    The floor stops channels that receive little gradient from being decayed all the way to zero.
    An FFN hidden channel or a q/k pair whose weights on both sides of the multiplication reach
    exactly zero gets exactly zero gradient forever after, and Newton-Schulz maps zero gradient
    rows to zero update rows, so under Muon that state is permanent. Held at the floor the
    channel still produces a small activation, so the weights that read it keep receiving
    gradient and it can be recruited again.

    floor_norm <= 0 or a tensor with fewer than 2 dims gets plain decay. Norms are accumulated in
    fp32 and the only temporaries are three vectors of length w.shape[0]. Rows at or below the
    floor are scaled by exactly 1, and the divisor is never below floor_norm, so an all-zero row
    causes no division by zero.

    This standalone multiply is the reference definition of the decay. The optimizers below do not
    call it. They fold the same per-channel decay into the parameter update instead, see
    _apply_updates_with_weight_decay, because a standalone multiply of fp32 weights by a factor
    within about 3e-8 of 1 rounds to no change at all.
    """
    if w.dim() < 2 or floor_norm <= 0.0:
        w.mul_(1.0 - a)
        return
    reduce_dims = tuple(range(1, w.dim()))
    r = torch.linalg.vector_norm(w, dim=reduce_dims, dtype=torch.float32)
    excess = (r - floor_norm).clamp_(min=0.0)
    scale = excess.div_(r.clamp_(min=floor_norm)).mul_(-a).add_(1.0)
    w.mul_(scale.to(w.dtype).view(-1, *([1] * (w.dim() - 1))))


def zeropower_via_newtonschulz5(G, steps: int):
    """
    Newton-Schulz iteration to compute the zeroth power / orthogonalization of G. We opt to use a
    quintic iteration whose coefficients are selected to maximize the slope at zero. For the purpose
    of minimizing steps, it turns out to be empirically effective to keep increasing the slope at
    zero even beyond the point where the iteration no longer converges all the way to one everywhere
    on the interval. This iteration therefore does not produce UV^T but rather something like US'V^T
    where S' is diagonal with S_{ii}' ~ Uniform(0.5, 1.5), which turns out not to hurt model
    performance at all relative to UV^T, where USV^T = G is the SVD.
    """
    assert G.ndim >= 2 # batched Muon implementation by @scottjmaddox, and put into practice in the record by @YouJiacheng
    a, b, c = (3.4445, -4.7750,  2.0315)
    X = G.bfloat16()
    if G.size(-2) > G.size(-1):
        X = X.mT

    # Ensure spectral norm is at most 1
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    # Perform the NS iterations
    for _ in range(steps):
        A = X @ X.mT
        B = b * A + c * A @ A # quintic computation strategy adapted from suggestion by @jxbz, @leloykun, and @YouJiacheng
        X = a * X + B @ X

    if G.size(-2) > G.size(-1):
        X = X.mT
    return X


# Polar Express: per-iteration coefficients with safety factor for numerical stability.
# See https://arxiv.org/abs/2505.16932
_POLAR_EXPRESS_COEFFS = [
    (8.28721201814563, -23.595886519098837, 17.300387312530933),
    (4.107059111542203, -2.9478499167379106, 0.5448431082926601),
    (3.9486908534822946, -2.908902115962949, 0.5518191394370137),
    (3.3184196573706015, -2.488488024314874, 0.51004894012372),
    (2.300652019954817, -1.6689039845747493, 0.4188073119525673),
    (1.891301407787398, -1.2679958271945868, 0.37680408948524835),
    (1.8750014808534479, -1.2500016453999487, 0.3750001645474248),
    (1.875, -1.25, 0.375),  # subsequent coeffs equal this numerically
]
# Apply safety factor for numerical stability (but exclude the last polynomial)
_POLAR_EXPRESS_COEFFS = [
    (a / 1.01, b / 1.01**3, c / 1.01**5)
    for (a, b, c) in _POLAR_EXPRESS_COEFFS[:-1]
] + [_POLAR_EXPRESS_COEFFS[-1]]


def zeropower_via_polar_express(G, steps: int):
    """
    Polar Express iteration for orthogonalization: same NS5 structure but with per-iteration
    coefficients that are tuned for faster convergence, and a slightly more conservative safety
    factor in the initial normalization.
    """
    assert G.ndim >= 2
    X = G.bfloat16()
    if G.size(-2) > G.size(-1):
        X = X.mT

    # Ensure spectral norm is at most 1, with 1.01 safety factor
    X = X / (X.norm(dim=(-2, -1), keepdim=True) * 1.01 + 1e-7)
    # Per-iteration coefficients, repeating the last entry if more steps than coefficients
    hs = _POLAR_EXPRESS_COEFFS[:steps] + list(
        repeat(_POLAR_EXPRESS_COEFFS[-1], steps - len(_POLAR_EXPRESS_COEFFS))
    )
    for a, b, c in hs:
        A = X @ X.mT
        B = b * A + c * A @ A
        X = a * X + B @ X

    if G.size(-2) > G.size(-1):
        X = X.mT
    return X


def _aurora_polar(update, ns_steps=5, pp_iterations=2, pp_beta=0.5, eps=1e-7, use_polar_express=False):
    """
    Leverage-uniform polar factor for Aurora optimizer.
    For non-square matrices, iteratively applies diagonal preconditioning to equalize
    row norms before orthogonalization. For square matrices, falls back to standard polar.
    See https://tilderesearch.com/blog/aurora
    """
    m, n = update.size(-2), update.size(-1)
    if m == n:
        # Square: no leverage freedom to exploit, standard polar.
        if use_polar_express:
            return zeropower_via_polar_express(update, steps=ns_steps)
        else:
            return zeropower_via_newtonschulz5(update, steps=ns_steps)

    # For wide matrices, transpose to tall, apply, transpose back.
    transposed = m < n
    if transposed:
        update = update.mT
        m, n = n, m

    G32 = update.to(torch.float32)
    target_row_sq = n / m
    row_norm = G32.norm(dim=-1, keepdim=True).clamp_(min=eps)
    D = 1.0 / row_norm
    for k in range(pp_iterations):
        scaled = D * G32
        if use_polar_express:
            U = zeropower_via_polar_express(scaled, steps=ns_steps)
        else:
            U = zeropower_via_newtonschulz5(scaled, steps=ns_steps)
        if k < pp_iterations - 1:
            row_sq = U.to(torch.float32).pow(2).sum(dim=-1, keepdim=True).clamp_(min=eps * eps)
            D = D * (target_row_sq / row_sq).pow(pp_beta)

    if transposed:
        U = U.mT
    return U


def muon_update(grad, momentum, ns_steps=5, beta=0.95, nesterov=True, adjust_lr_fn="match_rms_adamw",
                normuon_v=None, normuon_beta2=0.95, normuon_eps=1e-8, use_polar_express=False):
    """
    Compute the Muon (or NorMuon) update for a single parameter.

    If normuon_v is provided, applies NorMuon: neuron-wise (row-wise) adaptive normalization
    after orthogonalization, with dynamic lr scaling. See https://arxiv.org/abs/2510.05491

    Args:
        grad: The gradient tensor.
        momentum: The first-order momentum buffer (updated in-place).
        ns_steps: Number of Newton-Schulz iterations.
        beta: First-order momentum decay.
        nesterov: Whether to use Nesterov momentum.
        adjust_lr_fn: Learning rate scaling mode. NorMuon requires "match_rms_adamw".
        normuon_v: Per-row second-order momentum buffer (shape (m,)), or None for standard Muon.
        normuon_beta2: Second-order momentum decay for NorMuon.
        normuon_eps: Epsilon for numerical stability in NorMuon row normalization.
        use_polar_express: Use Polar Express iteration instead of standard NS5.
    """
    momentum.lerp_(grad, 1 - beta)
    update = grad.lerp_(momentum, beta) if nesterov else momentum
    if update.ndim == 4: # for the case of conv filters
        update = update.view(len(update), -1)
    if use_polar_express:
        update = zeropower_via_polar_express(update, steps=ns_steps)
    else:
        update = zeropower_via_newtonschulz5(update, steps=ns_steps)

    if normuon_v is not None:
        assert adjust_lr_fn == "match_rms_adamw", \
            f"NorMuon requires adjust_lr_fn='match_rms_adamw', got '{adjust_lr_fn}'"
        # Update per-row second-order momentum: v = beta2 * v + (1 - beta2) * mean_cols(O^2)
        normuon_v.lerp_(update.square().mean(dim=-1).to(normuon_v.dtype), 1 - normuon_beta2)
        # Row-wise normalization: O_hat = O / (sqrt(v) + eps)
        update = update / (normuon_v.sqrt().unsqueeze(-1) + normuon_eps)
        # Dynamic lr scaling: 0.1825 * sqrt(m*n) / ||O_hat||_F
        # Official paper has it as 0.2 here but NorMuon's perfect rowwise normalization produces
        # very slightly larger steps than Muon so we compensate that by scaling with 0.1825 instead of 0.2.
        update *= 0.1825 * (update.size(-2) * update.size(-1))**0.5 / (update.norm() + 1e-30)
    elif adjust_lr_fn == "match_rms_adamw":
        if use_polar_express:
            update *= 0.1825 * max(update.size(-2), update.size(-1))**0.5
        else:
            update *= 0.2 * max(update.size(-2), update.size(-1))**0.5
    elif adjust_lr_fn == "original":
        update *= max(1, update.size(-2) / update.size(-1))**0.5
    else:
        raise AssertionError(f"Unexpected value {adjust_lr_fn=}")
    return update


def aurora_update(grad, momentum, ns_steps=5, beta=0.95, nesterov=True, adjust_lr_fn="match_rms_adamw",
                  pp_iterations=2, pp_beta=0.5, eps=1e-7, use_polar_express=False):
    """
    Compute the Aurora update for a single parameter.
    Aurora is a leverage-aware optimizer that extends Muon with diagonal preconditioning
    to achieve uniform row norms in the polar factor for non-square matrices, preventing
    neuron death in MLP layers.
    See https://tilderesearch.com/blog/aurora

    Args:
        grad: The gradient tensor.
        momentum: The first-order momentum buffer (updated in-place).
        ns_steps: Number of Newton-Schulz iterations per polar call.
        beta: First-order momentum decay.
        nesterov: Whether to use Nesterov momentum.
        adjust_lr_fn: Learning rate scaling mode.
        pp_iterations: Number of preconditioning-polar iterations (K in the paper).
        pp_beta: Damping parameter for the diagonal preconditioner EMA.
        eps: Epsilon for numerical stability.
        use_polar_express: Use Polar Express iteration instead of standard NS5.
    """
    momentum.lerp_(grad, 1 - beta)
    update = grad.lerp_(momentum, beta) if nesterov else momentum
    if update.ndim == 4:  # for the case of conv filters
        update = update.view(len(update), -1)

    update = _aurora_polar(update, ns_steps=ns_steps, pp_iterations=pp_iterations,
                           pp_beta=pp_beta, eps=eps, use_polar_express=use_polar_express)

    if adjust_lr_fn == "match_rms_adamw":
        if use_polar_express:
            update *= 0.1825 * max(update.size(-2), update.size(-1))**0.5
        else:
            update *= 0.2 * max(update.size(-2), update.size(-1))**0.5
    elif adjust_lr_fn == "original":
        update *= max(1, update.size(-2) / update.size(-1))**0.5
    else:
        raise AssertionError(f"Unexpected value {adjust_lr_fn=}")
    return update


def adam_update(grad, buf1, buf2, step, betas, eps):
    buf1.lerp_(grad, 1 - betas[0])
    buf2.lerp_(grad.square(), 1 - betas[1])
    buf1c = buf1 / (1 - betas[0]**step)
    buf2c = buf2 / (1 - betas[1]**step)
    return buf1c / (buf2c.sqrt() + eps)


# Compiled variants used by the batched Newton-Schulz path.
# The number of distinct (batch, rows, cols) shapes per model is small, so per-shape compilation settles quickly.
# These functions accept stacked (B, m, n) input since the underlying implementations already support batched matrices.
# torch.compile mode for the batched Newton-Schulz / Polar Express iterations. cuBLAS picks a
# weak kernel for the small stacked bf16 matmuls involved (e.g. a stack of 32 matrices of size
# 384 x 384), and letting inductor autotune the batched matmul against its Triton templates roughly
# halves the optimizer step on RTX PRO 6000 Blackwell. The autotuning adds some seconds of compile
# time per distinct stacked shape. Set KATAGO_MUON_NS_COMPILE_MODE=default to skip the autotuning.
_NS_COMPILE_MODE = os.environ.get("KATAGO_MUON_NS_COMPILE_MODE", "max-autotune-no-cudagraphs")
if _NS_COMPILE_MODE not in ("default", "max-autotune-no-cudagraphs", "max-autotune"):
    raise ValueError(f"KATAGO_MUON_NS_COMPILE_MODE must be default|max-autotune-no-cudagraphs|max-autotune, got {_NS_COMPILE_MODE!r}")
zeropower_via_newtonschulz5_compiled = torch.compile(zeropower_via_newtonschulz5, mode=_NS_COMPILE_MODE)
zeropower_via_polar_express_compiled = torch.compile(zeropower_via_polar_express, mode=_NS_COMPILE_MODE)


DEFAULT_DISTRIBUTED_BUCKET_CAP_BYTES = 16 * 1024 * 1024


@dataclass(frozen=True)
class _MuonBucketSegment:
    param_index: int
    param_offset: int
    packed_offset: int
    numel: int


@dataclass(frozen=True)
class _MuonFlatBucketPlan:
    collective_numel: int
    owner_numels: Tuple[int, ...]
    segments_by_owner: Tuple[Tuple[_MuonBucketSegment, ...], ...]


@dataclass
class _MuonDistributedLayout:
    params: Tuple[torch.Tensor, ...]
    buckets: Tuple[_MuonFlatBucketPlan, ...]
    send_buffer: torch.Tensor
    gathered_buffer: torch.Tensor


def _build_muon_flat_bucket_plan(
    owner_param_numels: Sequence[Sequence[Tuple[int, int]]],
    bucket_cap_numel: int,
) -> Tuple[_MuonFlatBucketPlan, ...]:
    """Build equal-sized all-gather buckets from one parameter stream per owner.

    Each input item is ``(param_index, numel)``.
    A parameter may be split across buckets, but every element appears exactly once and in parameter-stream order.
    The collective size of a bucket is the largest owner payload in that bucket.
    Shorter owner payloads are padded by the caller.
    """
    if bucket_cap_numel <= 0:
        raise ValueError(f"bucket_cap_numel must be positive, got {bucket_cap_numel}")
    if len(owner_param_numels) <= 0:
        raise ValueError("owner_param_numels must contain at least one owner")

    normalized_streams: List[Tuple[Tuple[int, int], ...]] = []
    for stream in owner_param_numels:
        normalized_stream = []
        for param_index, numel in stream:
            if param_index < 0:
                raise ValueError(f"param_index must be nonnegative, got {param_index}")
            if numel < 0:
                raise ValueError(f"parameter numel must be nonnegative, got {numel}")
            if numel > 0:
                normalized_stream.append((param_index, numel))
        normalized_streams.append(tuple(normalized_stream))

    stream_indices = [0 for _ in normalized_streams]
    param_offsets = [0 for _ in normalized_streams]
    buckets: List[_MuonFlatBucketPlan] = []

    while any(stream_indices[owner] < len(normalized_streams[owner]) for owner in range(len(normalized_streams))):
        owner_numels: List[int] = []
        segments_by_owner: List[Tuple[_MuonBucketSegment, ...]] = []

        for owner, stream in enumerate(normalized_streams):
            packed_offset = 0
            owner_segments: List[_MuonBucketSegment] = []
            while packed_offset < bucket_cap_numel and stream_indices[owner] < len(stream):
                param_index, param_numel = stream[stream_indices[owner]]
                param_offset = param_offsets[owner]
                take = min(param_numel - param_offset, bucket_cap_numel - packed_offset)
                assert take > 0
                owner_segments.append(_MuonBucketSegment(
                    param_index=param_index,
                    param_offset=param_offset,
                    packed_offset=packed_offset,
                    numel=take,
                ))
                packed_offset += take
                param_offset += take
                if param_offset == param_numel:
                    stream_indices[owner] += 1
                    param_offsets[owner] = 0
                else:
                    param_offsets[owner] = param_offset

            owner_numels.append(packed_offset)
            segments_by_owner.append(tuple(owner_segments))

        collective_numel = max(owner_numels)
        assert collective_numel > 0
        buckets.append(_MuonFlatBucketPlan(
            collective_numel=collective_numel,
            owner_numels=tuple(owner_numels),
            segments_by_owner=tuple(segments_by_owner),
        ))

    return tuple(buckets)


class _MuonWithAuxAdamBase(torch.optim.Optimizer):
    """Shared implementation for the single-device and distributed variants.

    Optimization toggles (set the environment variable to 0 to disable for
    debugging or exact regression comparison against the historical kernels):
      KATAGO_MUON_BATCHED_NS (default 1): stack Muon updates with the same
        matrix shape (up to KATAGO_MUON_NS_BATCH_SIZE, default 32) into a single
        compiled Newton-Schulz iteration rather than one launch sequence per
        parameter. Same update equations, but not bitwise identical to the
        scalar launches.
      KATAGO_AUX_ADAM_FOREACH (default 1): use torch._foreach multi-tensor
        kernels for the auxiliary Adam parameter groups.
      KATAGO_MUON_GATHER_BF16_UPDATES (default 1, distributed only): after the
        sharded Newton-Schulz, all-gather the bf16 orthogonalized updates (the
        precision Newton-Schulz already produces) and apply the fp32 scale /
        weight decay / update step identically on every rank, instead of
        all-gathering the updated fp32 parameters. This halves synchronization
        bytes, and parameters stay bitwise identical across ranks and to the
        fp32-gather path. Plain Muon with adjust_lr_fn "match_rms_adamw" on the
        batched Newton-Schulz path only. NorMuon, Aurora, and the scalar path
        keep the parameter all-gather.
    """
    def __init__(self, param_groups, adjust_lr_fn="match_rms_adamw", adam_betas=(0.95, 0.995), adam_eps=1e-6,
                 use_normuon=False, normuon_beta2=0.95, normuon_eps=1e-8,
                 use_aurora=False, aurora_pp_iterations=2, aurora_pp_beta=0.5, aurora_eps=1e-7,
                 ns_steps=5, use_polar_express=False, sort_muon_params=False):
        self.use_normuon = use_normuon
        self.normuon_beta2 = normuon_beta2
        self.normuon_eps = normuon_eps
        self.use_aurora = use_aurora
        self.aurora_pp_iterations = aurora_pp_iterations
        self.aurora_pp_beta = aurora_pp_beta
        self.aurora_eps = aurora_eps
        self.ns_steps = ns_steps
        self.use_polar_express = use_polar_express
        # Aurora's data-dependent preconditioning loop stays on the scalar path.
        self.use_batched_muon_ns = _env_flag("KATAGO_MUON_BATCHED_NS", default=True) and not use_aurora
        self.use_foreach_aux_adam = _env_flag("KATAGO_AUX_ADAM_FOREACH", default=True)
        self.gather_bf16_updates_requested = _env_flag("KATAGO_MUON_GATHER_BF16_UPDATES", default=True)
        self.muon_ns_batch_size = int(os.environ.get("KATAGO_MUON_NS_BATCH_SIZE", "32"))
        if self.muon_ns_batch_size <= 0:
            raise ValueError(f"KATAGO_MUON_NS_BATCH_SIZE must be positive, got {self.muon_ns_batch_size}")
        # Parameter -> per-output-channel weight decay floor norm, see floored_weight_decay_.
        # Empty means plain decay for everything. Not saved in the optimizer state dict, the
        # caller recomputes the floors from stored initialization statistics at every startup.
        self.weight_decay_floor_norms = {}
        if self.use_batched_muon_ns:
            logging.info(f"Muon: using batched Newton-Schulz with batch size {self.muon_ns_batch_size}")
        if self.use_foreach_aux_adam:
            logging.info("Muon: using foreach kernels for auxiliary Adam parameter groups")
        for group in param_groups:
            assert "use_muon" in group
            if group["use_muon"]:
                # Both the distributed and the single-device optimizer sort, so that the position
                # of each parameter in its group, which is what the optimizer state dict is keyed
                # by, is the same whichever class wrote a checkpoint. The distributed class also
                # relies on this order to balance round-robin parameter ownership across ranks.
                if sort_muon_params:
                    group["params"] = sorted(group["params"], key=lambda x: x.size(), reverse=True)
                # defaults
                group["lr"] = group.get("lr", 0.02)
                group["momentum"] = group.get("momentum", 0.95)
                group["weight_decay"] = group.get("weight_decay", 0)
                group["adjust_lr_fn"] = group.get("adjust_lr_fn", adjust_lr_fn)
            else:
                # defaults
                group["lr"] = group.get("lr", 3e-4)
                group["betas"] = group.get("betas", adam_betas)
                group["eps"] = group.get("eps", adam_eps)
                group["weight_decay"] = group.get("weight_decay", 0)
        super().__init__(param_groups, dict())

    def load_state_dict(self, state_dict):
        """Load optimizer state, but drop it with a warning if the saved per-parameter tensors do
        not match the shapes of the parameters at their positions. That happens for checkpoints
        written before the single-device optimizer sorted its Muon groups the same way as the
        distributed one, and the mismatch cannot be repaired from the saved positions alone.
        Momentum and Adam moments are short-lived, so starting them fresh is harmless."""
        saved_groups = state_dict["param_groups"]
        if len(saved_groups) != len(self.param_groups) or any(
            len(saved["params"]) != len(group["params"]) for saved, group in zip(saved_groups, self.param_groups)
        ):
            logging.warning("Optimizer state dict has a different parameter group layout than this optimizer, dropping optimizer state")
            return
        param_of_id = {}
        for saved, group in zip(saved_groups, self.param_groups):
            for param_id, p in zip(saved["params"], group["params"]):
                param_of_id[param_id] = p
        for param_id, state in state_dict["state"].items():
            p = param_of_id.get(param_id)
            if p is None:
                logging.warning(f"Optimizer state refers to unknown parameter id {param_id}, dropping optimizer state")
                return
            for key, value in state.items():
                if not isinstance(value, torch.Tensor):
                    continue
                expected = (p.shape[0],) if key == "normuon_v" else tuple(p.shape)
                if tuple(value.shape) != expected:
                    logging.warning(
                        f"Optimizer state {key} of shape {tuple(value.shape)} does not match its parameter of shape "
                        f"{expected}, so the parameter order differs from when the checkpoint was written "
                        "(e.g. a single-GPU Muon checkpoint from before Muon groups were sorted). Dropping optimizer state."
                    )
                    return
        super().load_state_dict(state_dict)

    def set_weight_decay_floor_norms(self, floor_norms):
        """Set per-parameter weight decay floors, a dict from parameter to floor norm (see
        floored_weight_decay_). Replaces any floors set before. Parameters absent from the dict,
        or with a floor of 0 or less, get plain weight decay. Every rank must set identical floors."""
        params = set()
        for group in self.param_groups:
            params.update(group["params"])
        for p in floor_norms:
            if p not in params:
                raise ValueError("Weight decay floor given for a tensor that is not an optimizer parameter")
        self.weight_decay_floor_norms = {p: float(f) for p, f in floor_norms.items() if float(f) > 0.0}

    def _apply_updates_with_weight_decay(self, params, updates, lr, weight_decay):
        """Apply p -= lr * (update + d * p) to each parameter, where d is weight_decay for a plain
        parameter and weight_decay * max(0, r - floor) / r per output channel of norm r for a
        parameter with a registered floor. This is the same decay as floored_weight_decay_ with
        a = lr * weight_decay, folded into the update before the single add to the parameter.

        The decay is folded in rather than applied as a separate multiply because the parameters
        are fp32. A separate multiply by (1 - a) is quantized to the fp32 spacing below 1.0, about
        6e-8, so it does nothing when a is below about 3e-8 and is off by tens of percent for a
        up to about 1e-7, which is where a = lr * weight_decay lands at the low end of the learning
        rate schedule and during warmup. Added to an update that is much larger than the fp32
        spacing of the parameter, the decay term survives rounding in expectation. It is still
        lost for an element whose update is zero, since nothing then separates the rounded sum
        from the parameter itself. The expectation argument also assumes the update values are
        spread evenly between fp32 grid points of the parameter. A Muon update is a bf16 value
        times a scale, so its values lie on a coarse comb, and the realized decay is biased by an
        amount that depends on the learning rate, about 1 percent at a = 1e-8 and larger still
        for smaller a, where the decay is in any case negligible.

        The per-channel factors of all floored parameters are computed with multi-tensor kernels,
        so the per-tensor launches are only the row norm and the broadcast multiply-add. The
        updates are consumed and may be modified in place. An update whose dtype differs from its
        parameter is converted first, so the decay term is never rounded to a narrower dtype.
        """
        if len(params) == 0:
            return
        updates = [u if u.dtype == p.dtype else u.to(p.dtype) for p, u in zip(params, updates)]
        if weight_decay != 0.0:
            if len(self.weight_decay_floor_norms) == 0:
                torch._foreach_add_(updates, params, alpha=weight_decay)
            else:
                is_floored = [p.dim() >= 2 and p in self.weight_decay_floor_norms for p in params]
                plain_params = [p for p, f in zip(params, is_floored) if not f]
                plain_updates = [u for u, f in zip(updates, is_floored) if not f]
                floored_params = [p for p, f in zip(params, is_floored) if f]
                floored_updates = [u for u, f in zip(updates, is_floored) if f]
                if len(plain_params) > 0:
                    torch._foreach_add_(plain_updates, plain_params, alpha=weight_decay)
                if len(floored_params) > 0:
                    floors = [self.weight_decay_floor_norms[p] for p in floored_params]
                    norms = [
                        torch.linalg.vector_norm(p, dim=tuple(range(1, p.dim())), dtype=torch.float32)
                        for p in floored_params
                    ]
                    fractions = torch._foreach_sub(norms, floors)
                    torch._foreach_clamp_min_(fractions, 0.0)
                    torch._foreach_clamp_min_(norms, floors)
                    torch._foreach_div_(fractions, norms)
                    torch._foreach_mul_(fractions, weight_decay)
                    torch._foreach_addcmul_(floored_updates, floored_params, [
                        f.to(p.dtype).view(-1, *([1] * (p.dim() - 1))) for f, p in zip(fractions, floored_params)
                    ])
        torch._foreach_add_(params, updates, alpha=-lr)

    def _ensure_muon_state(self, p):
        if p.grad is None:
            # continue
            p.grad = torch.zeros_like(p)  # Force synchronization
        state = self.state[p]
        if len(state) == 0:
            state["momentum_buffer"] = torch.zeros_like(p)
            if self.use_normuon:
                state["normuon_v"] = torch.zeros(p.shape[0], device=p.device, dtype=p.dtype)
        return state

    def _step_muon_param_scalar(self, group, p):
        state = self._ensure_muon_state(p)
        if self.use_aurora:
            update = aurora_update(
                p.grad, state["momentum_buffer"],
                ns_steps=self.ns_steps,
                beta=group["momentum"], adjust_lr_fn=group["adjust_lr_fn"],
                pp_iterations=self.aurora_pp_iterations,
                pp_beta=self.aurora_pp_beta, eps=self.aurora_eps,
                use_polar_express=self.use_polar_express,
            )
        else:
            update = muon_update(
                p.grad, state["momentum_buffer"],
                ns_steps=self.ns_steps,
                beta=group["momentum"], adjust_lr_fn=group["adjust_lr_fn"],
                normuon_v=state.get("normuon_v"),
                normuon_beta2=self.normuon_beta2, normuon_eps=self.normuon_eps,
                use_polar_express=self.use_polar_express,
            )
        self._apply_updates_with_weight_decay([p], [update.reshape(p.shape)], group["lr"], group["weight_decay"])

    def _update_scale(self, matrix_shape):
        """The match_rms_adamw scale for an orthogonalized update of the given 2D shape."""
        m, n = matrix_shape
        return (0.1825 if self.use_polar_express else 0.2) * max(m, n)**0.5

    def _apply_muon_updates(self, group, params, updates, chunk_size=64):
        """Apply weight decay and the scaled update for `params` given their bf16 orthogonalized
        `updates` (same shapes as the params), matching the arithmetic of _step_muon_params_batched:
        scaled = update.to(p.dtype) * scale, then _apply_updates_with_weight_decay.

        Every rank applies this to every Muon parameter, so group["lr"], group["weight_decay"] and
        the weight decay floors must be identical on all ranks for the parameters to stay
        identical across ranks.

        The parameters are processed in chunks so that the fp32 scaled updates of only one chunk are
        alive at a time (the same elementwise ops in the same order, so the result is unchanged).
        """
        for start in range(0, len(params), chunk_size):
            chunk_params = params[start:start + chunk_size]
            scaled = []
            for p, update in zip(chunk_params, updates[start:start + chunk_size]):
                matrix_shape = (len(p), p.numel() // len(p)) if p.ndim == 4 else tuple(p.shape)
                scaled.append(update.to(p.dtype) * self._update_scale(matrix_shape))
            self._apply_updates_with_weight_decay(chunk_params, scaled, group["lr"], group["weight_decay"])

    def _step_muon_params_batched(self, group, param_indices, deferred_updates=None):
        """Same equations as the scalar path, but Newton-Schulz iterations for
        same-shape matrices are launched as one batched computation, and the
        elementwise momentum/Nesterov/scale/weight-decay/apply passes use
        multi-tensor (foreach / stacked) kernels.

        Per-parameter momentum/Nesterov mutation semantics are preserved.
        Only independent computations are regrouped.

        If `deferred_updates` (a dict) is given, the plain-Muon match_rms_adamw path does not
        modify the parameters. Instead it stores each parameter's bf16 orthogonalized update (in the
        parameter's shape) into the dict for a later _apply_muon_updates, so the same update can be
        shared across ranks before being applied.
        """
        params = group["params"]
        chosen = []
        for param_index in param_indices:
            p = params[param_index]
            state = self._ensure_muon_state(p)
            chosen.append((p, state))
        if len(chosen) == 0:
            return

        # Momentum + Nesterov for all parameters in a few multi-tensor launches,
        # matching muon_update: momentum.lerp_(grad, 1-beta); grad.lerp_(momentum, beta).
        grads = [p.grad for p, state in chosen]
        momenta = [state["momentum_buffer"] for p, state in chosen]
        torch._foreach_lerp_(momenta, grads, 1 - group["momentum"])
        torch._foreach_lerp_(grads, momenta, group["momentum"])

        entries_by_shape = {}
        for p, state in chosen:
            update = p.grad
            matrix = update.view(len(update), -1) if update.ndim == 4 else update
            assert matrix.ndim == 2
            # Normalize orientation to rows <= cols so that transposed shape pairs share a batch.
            # Zeropower of the transpose is the transpose of zeropower, so this is equivalent to the scalar path.
            was_transposed = matrix.shape[0] > matrix.shape[1]
            normalized = matrix.mT if was_transposed else matrix
            key = (normalized.device, normalized.dtype, normalized.shape[0], normalized.shape[1])
            entries_by_shape.setdefault(key, []).append((p, normalized, was_transposed, state))

        apply_params = []
        apply_updates = []
        for entries in entries_by_shape.values():
            for chunk_begin in range(0, len(entries), self.muon_ns_batch_size):
                chunk = entries[chunk_begin:chunk_begin + self.muon_ns_batch_size]
                stacked = torch.stack([entry[1] for entry in chunk], dim=0)
                if self.use_polar_express:
                    orthogonalized = zeropower_via_polar_express_compiled(stacked, steps=self.ns_steps)
                else:
                    orthogonalized = zeropower_via_newtonschulz5_compiled(stacked, steps=self.ns_steps)

                if not self.use_normuon and group["adjust_lr_fn"] == "match_rms_adamw":
                    if deferred_updates is not None:
                        for (p, _, was_transposed, state), update in zip(chunk, orthogonalized.unbind(dim=0)):
                            if was_transposed:
                                update = update.mT
                            # Always copy: the update must outlive later invocations of the compiled
                            # Newton-Schulz, whose output buffer can be reused. Under a compile mode
                            # that uses CUDA graphs (max-autotune) it lives in a CUDA-graph private pool.
                            deferred_updates[p] = update.reshape(p.shape).clone(memory_format=torch.contiguous_format)
                        continue
                    # The adjust-lr scale depends only on the (shared) matrix
                    # shape, so scale the whole stacked chunk in one launch,
                    # in parameter dtype for the foreach apply below.
                    scale = self._update_scale((orthogonalized.shape[-2], orthogonalized.shape[-1]))
                    scaled = orthogonalized.to(chunk[0][0].dtype) * scale
                    for (p, _, was_transposed, state), update in zip(chunk, scaled.unbind(dim=0)):
                        if was_transposed:
                            update = update.mT
                        apply_params.append(p)
                        apply_updates.append(update.reshape(p.shape))
                    continue

                for (p, _, was_transposed, state), update in zip(chunk, orthogonalized.unbind(dim=0)):
                    if was_transposed:
                        update = update.mT
                    normuon_v = state.get("normuon_v")
                    if normuon_v is not None:
                        assert group["adjust_lr_fn"] == "match_rms_adamw", \
                            f"NorMuon requires adjust_lr_fn='match_rms_adamw', got '{group['adjust_lr_fn']}'"
                        normuon_v.lerp_(update.square().mean(dim=-1).to(normuon_v.dtype), 1 - self.normuon_beta2)
                        update = update / (normuon_v.sqrt().unsqueeze(-1) + self.normuon_eps)
                        update = update * (0.1825 * (update.size(-2) * update.size(-1))**0.5 / (update.norm() + 1e-30))
                    elif group["adjust_lr_fn"] == "match_rms_adamw":
                        if self.use_polar_express:
                            update = update * (0.1825 * max(update.size(-2), update.size(-1))**0.5)
                        else:
                            update = update * (0.2 * max(update.size(-2), update.size(-1))**0.5)
                    elif group["adjust_lr_fn"] == "original":
                        update = update * (max(1, update.size(-2) / update.size(-1))**0.5)
                    else:
                        raise AssertionError(f"Unexpected value adjust_lr_fn={group['adjust_lr_fn']}")
                    apply_params.append(p)
                    apply_updates.append(update.reshape(p.shape).to(p.dtype))

        self._apply_updates_with_weight_decay(apply_params, apply_updates, group["lr"], group["weight_decay"])

    def _step_muon_group(self, group, param_indices):
        if self.use_batched_muon_ns:
            self._step_muon_params_batched(group, param_indices)
        else:
            for param_index in param_indices:
                self._step_muon_param_scalar(group, group["params"][param_index])

    def _ensure_adam_state(self, p):
        if p.grad is None:
            # continue
            p.grad = torch.zeros_like(p)  # Force synchronization
        state = self.state[p]
        if len(state) == 0:
            state["exp_avg"] = torch.zeros_like(p)
            state["exp_avg_sq"] = torch.zeros_like(p)
            state["step"] = 0
        return state

    def _step_adam_group(self, group):
        if self.use_foreach_aux_adam:
            self._step_adam_group_foreach(group)
            return
        for p in group["params"]:
            state = self._ensure_adam_state(p)
            state["step"] += 1
            update = adam_update(p.grad, state["exp_avg"], state["exp_avg_sq"],
                                 state["step"], group["betas"], group["eps"])
            self._apply_updates_with_weight_decay([p], [update], group["lr"], group["weight_decay"])

    def _step_adam_group_foreach(self, group):
        """Update an auxiliary Adam group with one multi-tensor launch per operation.
        Same update equations as adam_update."""
        entries_by_step = {}
        for p in group["params"]:
            state = self._ensure_adam_state(p)
            state["step"] += 1
            entries_by_step.setdefault(state["step"], []).append((
                p, p.grad, state["exp_avg"], state["exp_avg_sq"],
            ))

        beta1, beta2 = group["betas"]
        adam_lr = group["lr"]
        for step, entries in entries_by_step.items():
            params, grads, exp_avgs, exp_avg_sqs = map(list, zip(*entries))
            torch._foreach_lerp_(exp_avgs, grads, 1 - beta1)
            grads_sq = torch._foreach_mul(grads, grads)
            torch._foreach_lerp_(exp_avg_sqs, grads_sq, 1 - beta2)

            bias_correction1 = 1 - beta1**step
            bias_correction2_sqrt = (1 - beta2**step) ** 0.5
            denominators = torch._foreach_sqrt(exp_avg_sqs)
            torch._foreach_div_(denominators, bias_correction2_sqrt)
            torch._foreach_add_(denominators, group["eps"])
            updates = torch._foreach_div(exp_avgs, denominators)

            # The bias correction is applied through the step size, so the weight decay is scaled
            # up by the same factor to keep the decay per step at adam_lr * weight_decay.
            self._apply_updates_with_weight_decay(
                params, updates, adam_lr / bias_correction1, group["weight_decay"] * bias_correction1
            )


class MuonWithAuxAdam(_MuonWithAuxAdamBase):
    """
    Distributed Muon variant that can be used for all parameters in the network, since it runs an
    internal AdamW for the parameters that are not compatible with Muon. The user must manually
    specify which parameters shall be optimized with Muon and which with Adam by passing in a
    list of param_groups with the `use_muon` flag set.

    The point of this class is to allow the user to have a single optimizer in their code, rather
    than having both a Muon and an Adam which each need to be stepped.

    Muon parameter ownership is sharded round-robin across ranks. After each step the ranks are
    synchronized with reusable flat buckets (one all-gather per ~16 MiB bucket) rather than one
    collective per parameter, in one of two ways. By default (KATAGO_MUON_GATHER_BF16_UPDATES=1,
    plain Muon only) each rank gathers the bf16 orthogonalized updates of the other ranks and
    applies the fp32 update arithmetic to every parameter itself, which requires identical lr and
    weight decay on every rank. Otherwise the owners' updated fp32 parameters are gathered.

    Set use_normuon=True to enable NorMuon (neuron-wise normalized Muon), which adds row-wise
    adaptive learning rates after orthogonalization. See https://arxiv.org/abs/2510.05491

    Set use_aurora=True to enable Aurora (leverage-aware optimizer), which uses diagonal
    preconditioning to achieve uniform row norms in the polar factor for non-square matrices.
    See https://tilderesearch.com/blog/aurora

    You can see an example usage below:

    https://github.com/KellerJordan/modded-nanogpt/blob/master/records/052525_MuonWithAuxAdamExample/b01550f9-03d8-4a9c-86fe-4ab434f1c5e0.txt#L470
    ```
    hidden_matrix_params = [p for n, p in model.blocks.named_parameters() if p.ndim >= 2 and "embed" not in n]
    embed_params = [p for n, p in model.named_parameters() if "embed" in n]
    scalar_params = [p for p in model.parameters() if p.ndim < 2]
    head_params = [model.lm_head.weight]

    from muon import MuonWithAuxAdam
    adam_groups = [dict(params=head_params, lr=0.22), dict(params=embed_params, lr=0.6), dict(params=scalar_params, lr=0.04)]
    adam_groups = [dict(**g, betas=(0.8, 0.95), eps=1e-10, use_muon=False) for g in adam_groups]
    muon_group = dict(params=hidden_matrix_params, lr=0.05, momentum=0.95, use_muon=True)
    param_groups = [*adam_groups, muon_group]
    optimizer = MuonWithAuxAdam(param_groups)
    ```
    """
    def __init__(self, param_groups, adjust_lr_fn="match_rms_adamw", adam_betas=(0.95, 0.995), adam_eps=1e-6,
                 use_normuon=False, normuon_beta2=0.95, normuon_eps=1e-8,
                 use_aurora=False, aurora_pp_iterations=2, aurora_pp_beta=0.5, aurora_eps=1e-7,
                 ns_steps=5, use_polar_express=False,
                 distributed_bucket_cap_bytes=DEFAULT_DISTRIBUTED_BUCKET_CAP_BYTES):
        self.distributed_bucket_cap_bytes = int(distributed_bucket_cap_bytes)
        if self.distributed_bucket_cap_bytes <= 0:
            raise ValueError(f"distributed_bucket_cap_bytes must be positive, got {distributed_bucket_cap_bytes}")
        self._muon_distributed_layouts = None
        self._muon_update_layouts = None
        super().__init__(
            param_groups, adjust_lr_fn=adjust_lr_fn, adam_betas=adam_betas, adam_eps=adam_eps,
            use_normuon=use_normuon, normuon_beta2=normuon_beta2, normuon_eps=normuon_eps,
            use_aurora=use_aurora, aurora_pp_iterations=aurora_pp_iterations,
            aurora_pp_beta=aurora_pp_beta, aurora_eps=aurora_eps,
            ns_steps=ns_steps, use_polar_express=use_polar_express,
            sort_muon_params=True,
        )
        # bf16 update gathering requires every Muon group to take the batched plain-Muon path.
        self.gather_bf16_updates = (
            self.gather_bf16_updates_requested
            and self.use_batched_muon_ns
            and not self.use_normuon
            and not self.use_aurora
            and all(group["adjust_lr_fn"] == "match_rms_adamw" for group in self.param_groups if group["use_muon"])
        )
        if self.gather_bf16_updates:
            logging.info("Muon DDP: synchronizing bf16 orthogonalized updates instead of fp32 parameters")

    def _initialize_muon_distributed_layouts(self):
        self._muon_distributed_layouts = self._build_muon_distributed_layouts(buffer_dtype=None)

    def _build_muon_distributed_layouts(self, buffer_dtype):
        """Build the flat all-gather bucket layouts over all Muon parameters.

        buffer_dtype None means the buffers hold parameter values (parameter dtype). Otherwise the
        buffers hold per-element update values of that dtype with the same numel structure.
        """
        world_size = dist.get_world_size()

        # Insertion order follows parameter traversal and is therefore identical
        # across ranks even though each rank's CUDA device index is different.
        layout_builders = {}
        for group in self.param_groups:
            if not group["use_muon"]:
                continue
            for local_index, param in enumerate(group["params"]):
                if not param.is_contiguous():
                    raise ValueError(
                        "Distributed Muon parameter synchronization requires contiguous parameters, "
                        f"got shape={tuple(param.shape)} stride={param.stride()}"
                    )
                key = (param.device, param.dtype)
                if key not in layout_builders:
                    layout_builders[key] = {
                        "params": [],
                        "owner_param_numels": [[] for _ in range(world_size)],
                    }
                builder = layout_builders[key]
                param_index = len(builder["params"])
                builder["params"].append(param)
                owner = local_index % world_size
                builder["owner_param_numels"][owner].append((param_index, param.numel()))

        layouts = []
        total_buckets = 0
        total_workspace_bytes = 0
        for builder in layout_builders.values():
            params = tuple(builder["params"])
            if len(params) <= 0:
                continue
            dtype = params[0].dtype if buffer_dtype is None else buffer_dtype
            element_size = torch.empty((), dtype=dtype).element_size()
            bucket_cap_numel = max(1, self.distributed_bucket_cap_bytes // element_size)
            buckets = _build_muon_flat_bucket_plan(
                builder["owner_param_numels"],
                bucket_cap_numel,
            )
            if len(buckets) <= 0:
                continue
            max_collective_numel = max(bucket.collective_numel for bucket in buckets)
            send_buffer = torch.empty(
                max_collective_numel,
                dtype=dtype,
                device=params[0].device,
            )
            gathered_buffer = torch.empty(
                world_size * max_collective_numel,
                dtype=dtype,
                device=params[0].device,
            )
            layouts.append(_MuonDistributedLayout(
                params=params,
                buckets=buckets,
                send_buffer=send_buffer,
                gathered_buffer=gathered_buffer,
            ))
            total_buckets += len(buckets)
            total_workspace_bytes += (world_size + 1) * max_collective_numel * element_size

        logging.info(
            "Muon DDP flat %s synchronization: %d bucket(s), %.1f MiB reusable workspace per rank",
            "parameter" if buffer_dtype is None else f"{buffer_dtype} update",
            total_buckets,
            total_workspace_bytes / (1024.0 * 1024.0),
        )
        return tuple(layouts)

    def state_dict_for_checkpoint(self):
        """Full optimizer state dict including Muon states owned by other ranks.

        Muon momentum (and NorMuon second-moment) buffers are sharded: each rank
        only steps params[rank::world_size] within each Muon group, so a plain
        rank-0 state_dict() would silently drop the other ranks' buffers and
        they would reinitialize to zero on resume. This gathers every rank's
        owned Muon states onto rank 0.

        COLLECTIVE: every rank must call this at the same point. Returns the
        complete state dict on rank 0 and None on other ranks.
        """
        optimizer_state_dict = self.state_dict()
        rank = dist.get_rank()
        world_size = dist.get_world_size()

        local_muon_state = {}
        for group in optimizer_state_dict["param_groups"]:
            if not group.get("use_muon", False):
                continue
            for local_index, param_id in enumerate(group["params"]):
                if local_index % world_size != rank:
                    continue
                if param_id not in optimizer_state_dict["state"]:
                    continue
                state = optimizer_state_dict["state"][param_id]
                local_muon_state[param_id] = {
                    key: (value.detach().cpu() if isinstance(value, torch.Tensor) else value)
                    for key, value in state.items()
                }

        gathered_muon_state = [None for _ in range(world_size)] if rank == 0 else None
        dist.gather_object(local_muon_state, gathered_muon_state, dst=0)
        if rank != 0:
            return None
        for rank_state in gathered_muon_state:
            if rank_state is None:
                continue
            for param_id, state in rank_state.items():
                optimizer_state_dict["state"][param_id] = state
        return optimizer_state_dict

    def load_state_dict_for_checkpoint(self, state_dict):
        """Load a checkpoint produced by state_dict_for_checkpoint, keeping only
        the locally-owned Muon states on this rank. Because the checkpoint holds
        every rank's Muon states, resuming with a different world size works as
        long as the parameter groups and their ordering are unchanged."""
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        muon_non_local_param_ids = set()
        for group in state_dict["param_groups"]:
            if not group.get("use_muon", False):
                continue
            for local_index, param_id in enumerate(group["params"]):
                if local_index % world_size != rank:
                    muon_non_local_param_ids.add(param_id)
        filtered_state = {
            param_id: state
            for param_id, state in state_dict["state"].items()
            if param_id not in muon_non_local_param_ids
        }
        self.load_state_dict({
            "state": filtered_state,
            "param_groups": state_dict["param_groups"],
        })

    def _synchronize_muon_parameters(self):
        if self._muon_distributed_layouts is None:
            self._initialize_muon_distributed_layouts()
        self._all_gather_over_layouts(
            self._muon_distributed_layouts,
            lambda param: param.detach(),
        )

    def _synchronize_muon_updates(self, updates, owned_params):
        """Fill `updates` (param -> bf16 update, with every parameter in `owned_params` present on
        entry) for every Muon parameter by all-gathering the owned updates."""
        if self._muon_update_layouts is None:
            self._muon_update_layouts = self._build_muon_distributed_layouts(buffer_dtype=torch.bfloat16)
        for layout in self._muon_update_layouts:
            for param in layout.params:
                if param in owned_params:
                    # Sending an absent update would broadcast uninitialized memory to every rank.
                    assert param in updates, "owned Muon parameter has no deferred update to gather"
                    assert updates[param].dtype == torch.bfloat16 and updates[param].shape == param.shape
                elif param not in updates:
                    updates[param] = torch.empty(param.shape, dtype=torch.bfloat16, device=param.device)
        self._all_gather_over_layouts(
            self._muon_update_layouts,
            lambda param: updates[param],
        )

    def _all_gather_over_layouts(self, layouts, tensor_of_param):
        """Bucketed all-gather: each rank sends its owned slices of tensor_of_param(param) and
        writes the other ranks' slices into its copies of their parameters' tensors."""
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        for layout in layouts:
            for bucket in layout.buckets:
                collective_numel = bucket.collective_numel
                owner_numel = bucket.owner_numels[rank]
                send = layout.send_buffer[:collective_numel]

                local_parts = [
                    tensor_of_param(layout.params[segment.param_index]).view(-1)[
                        segment.param_offset:segment.param_offset + segment.numel
                    ]
                    for segment in bucket.segments_by_owner[rank]
                ]
                if len(local_parts) == 1:
                    send[:owner_numel].copy_(local_parts[0])
                elif len(local_parts) > 1:
                    torch.cat(local_parts, dim=0, out=send[:owner_numel])
                else:
                    assert owner_numel == 0
                if owner_numel < collective_numel:
                    send[owner_numel:].zero_()

                gathered = layout.gathered_buffer[:world_size * collective_numel]
                dist.all_gather_into_tensor(gathered, send)
                gathered_by_owner = gathered.view(world_size, collective_numel)

                destination_parts = []
                source_parts = []
                for owner in range(world_size):
                    if owner == rank:
                        continue
                    for segment in bucket.segments_by_owner[owner]:
                        destination_parts.append(
                            tensor_of_param(layout.params[segment.param_index]).view(-1)[
                                segment.param_offset:segment.param_offset + segment.numel
                            ]
                        )
                        source_parts.append(
                            gathered_by_owner[owner, segment.packed_offset:segment.packed_offset + segment.numel]
                        )
                if len(destination_parts) > 0:
                    torch._foreach_copy_(destination_parts, source_parts)

    @torch.no_grad()
    def step(self, closure=None):

        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        rank = dist.get_rank()
        world_size = dist.get_world_size()
        deferred_updates = {} if self.gather_bf16_updates else None
        owned_params = set()
        for group in self.param_groups:
            if group["use_muon"]:
                # Round-robin ownership matching the historical layout. All
                # ranks must agree on this assignment; parameters this rank
                # does not own are filled in by the flat synchronization below.
                param_indices = range(rank, len(group["params"]), world_size)
                if deferred_updates is not None:
                    owned_params.update(group["params"][i] for i in param_indices)
                    self._step_muon_params_batched(group, param_indices, deferred_updates=deferred_updates)
                else:
                    self._step_muon_group(group, param_indices)
            else:
                self._step_adam_group(group)

        if deferred_updates is not None:
            # Share the bf16 orthogonalized updates, then every rank applies the identical
            # fp32 arithmetic to every Muon parameter.
            self._synchronize_muon_updates(deferred_updates, owned_params)
            for group in self.param_groups:
                if group["use_muon"] and len(group["params"]) > 0:
                    params = list(group["params"])
                    self._apply_muon_updates(group, params, [deferred_updates[p] for p in params])
        else:
            self._synchronize_muon_parameters()

        return loss


class SingleDeviceMuonWithAuxAdam(_MuonWithAuxAdamBase):
    """
    Non-distributed variant of MuonWithAuxAdam.
    """
    def __init__(self, param_groups, adjust_lr_fn="match_rms_adamw", adam_betas=(0.95, 0.995), adam_eps=1e-6,
                 use_normuon=False, normuon_beta2=0.95, normuon_eps=1e-8,
                 use_aurora=False, aurora_pp_iterations=2, aurora_pp_beta=0.5, aurora_eps=1e-7,
                 ns_steps=5, use_polar_express=False):
        super().__init__(
            param_groups, adjust_lr_fn=adjust_lr_fn, adam_betas=adam_betas, adam_eps=adam_eps,
            use_normuon=use_normuon, normuon_beta2=normuon_beta2, normuon_eps=normuon_eps,
            use_aurora=use_aurora, aurora_pp_iterations=aurora_pp_iterations,
            aurora_pp_beta=aurora_pp_beta, aurora_eps=aurora_eps,
            ns_steps=ns_steps, use_polar_express=use_polar_express,
            sort_muon_params=True,
        )

    @torch.no_grad()
    def step(self, closure=None):

        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            if group["use_muon"]:
                self._step_muon_group(group, range(len(group["params"])))
            else:
                self._step_adam_group(group)

        return loss
