from itertools import repeat

import torch
import torch.distributed as dist


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


def adam_update(grad, buf1, buf2, step, betas, eps):
    buf1.lerp_(grad, 1 - betas[0])
    buf2.lerp_(grad.square(), 1 - betas[1])
    buf1c = buf1 / (1 - betas[0]**step)
    buf2c = buf2 / (1 - betas[1]**step)
    return buf1c / (buf2c.sqrt() + eps)


class MuonWithAuxAdam(torch.optim.Optimizer):
    """
    Distributed Muon variant that can be used for all parameters in the network, since it runs an
    internal AdamW for the parameters that are not compatible with Muon. The user must manually
    specify which parameters shall be optimized with Muon and which with Adam by passing in a
    list of param_groups with the `use_muon` flag set.

    The point of this class is to allow the user to have a single optimizer in their code, rather
    than having both a Muon and an Adam which each need to be stepped.

    Set use_normuon=True to enable NorMuon (neuron-wise normalized Muon), which adds row-wise
    adaptive learning rates after orthogonalization. See https://arxiv.org/abs/2510.05491

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
                 ns_steps=5, use_polar_express=False):
        self.use_normuon = use_normuon
        self.normuon_beta2 = normuon_beta2
        self.normuon_eps = normuon_eps
        self.ns_steps = ns_steps
        self.use_polar_express = use_polar_express
        for group in param_groups:
            assert "use_muon" in group
            if group["use_muon"]:
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

    @torch.no_grad()
    def step(self, closure=None):

        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            if group["use_muon"]:
                params = group["params"]
                params_pad = params + [torch.empty_like(params[-1])] * (dist.get_world_size() - len(params) % dist.get_world_size())
                for base_i in range(len(params))[::dist.get_world_size()]:
                    if base_i + dist.get_rank() < len(params):
                        p = params[base_i + dist.get_rank()]
                        if p.grad is None:
                            # continue
                            p.grad = torch.zeros_like(p)  # Force synchronization
                        state = self.state[p]
                        if len(state) == 0:
                            state["momentum_buffer"] = torch.zeros_like(p)
                            if self.use_normuon:
                                state["normuon_v"] = torch.zeros(p.shape[0], device=p.device, dtype=p.dtype)
                        update = muon_update(
                            p.grad, state["momentum_buffer"],
                            ns_steps=self.ns_steps,
                            beta=group["momentum"], adjust_lr_fn=group["adjust_lr_fn"],
                            normuon_v=state.get("normuon_v"),
                            normuon_beta2=self.normuon_beta2, normuon_eps=self.normuon_eps,
                            use_polar_express=self.use_polar_express,
                        )
                        p.mul_(1 - group["lr"] * group["weight_decay"])
                        p.add_(update.reshape(p.shape), alpha=-group["lr"])
                    dist.all_gather(params_pad[base_i:base_i + dist.get_world_size()], params_pad[base_i + dist.get_rank()])
            else:
                for p in group["params"]:
                    if p.grad is None:
                        # continue
                        p.grad = torch.zeros_like(p)  # Force synchronization
                    state = self.state[p]
                    if len(state) == 0:
                        state["exp_avg"] = torch.zeros_like(p)
                        state["exp_avg_sq"] = torch.zeros_like(p)
                        state["step"] = 0
                    state["step"] += 1
                    update = adam_update(p.grad, state["exp_avg"], state["exp_avg_sq"],
                                         state["step"], group["betas"], group["eps"])
                    p.mul_(1 - group["lr"] * group["weight_decay"])
                    p.add_(update, alpha=-group["lr"])

        return loss


class SingleDeviceMuonWithAuxAdam(torch.optim.Optimizer):
    """
    Non-distributed variant of MuonWithAuxAdam.
    """
    def __init__(self, param_groups, adjust_lr_fn="match_rms_adamw", adam_betas=(0.95, 0.995), adam_eps=1e-6,
                 use_normuon=False, normuon_beta2=0.95, normuon_eps=1e-8,
                 ns_steps=5, use_polar_express=False):
        self.use_normuon = use_normuon
        self.normuon_beta2 = normuon_beta2
        self.normuon_eps = normuon_eps
        self.ns_steps = ns_steps
        self.use_polar_express = use_polar_express
        for group in param_groups:
            assert "use_muon" in group
            if group["use_muon"]:
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

    @torch.no_grad()
    def step(self, closure=None):

        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            if group["use_muon"]:
                for p in group["params"]:
                    if p.grad is None:
                        # continue
                        p.grad = torch.zeros_like(p)  # Force synchronization
                    state = self.state[p]
                    if len(state) == 0:
                        state["momentum_buffer"] = torch.zeros_like(p)
                        if self.use_normuon:
                            state["normuon_v"] = torch.zeros(p.shape[0], device=p.device, dtype=p.dtype)
                    update = muon_update(
                        p.grad, state["momentum_buffer"],
                        ns_steps=self.ns_steps,
                        beta=group["momentum"], adjust_lr_fn=group["adjust_lr_fn"],
                        normuon_v=state.get("normuon_v"),
                        normuon_beta2=self.normuon_beta2, normuon_eps=self.normuon_eps,
                        use_polar_express=self.use_polar_express,
                    )
                    p.mul_(1 - group["lr"] * group["weight_decay"])
                    p.add_(update.reshape(p.shape), alpha=-group["lr"])
            else:
                for p in group["params"]:
                    if p.grad is None:
                        # continue
                        p.grad = torch.zeros_like(p)  # Force synchronization
                    state = self.state[p]
                    if len(state) == 0:
                        state["exp_avg"] = torch.zeros_like(p)
                        state["exp_avg_sq"] = torch.zeros_like(p)
                        state["step"] = 0
                    state["step"] += 1
                    update = adam_update(p.grad, state["exp_avg"], state["exp_avg_sq"],
                                         state["step"], group["betas"], group["eps"])
                    p.mul_(1 - group["lr"] * group["weight_decay"])
                    p.add_(update, alpha=-group["lr"])

        return loss
