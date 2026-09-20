"""
Tests for the per-output-channel weight decay floor: floored_weight_decay_ in muon/muon.py and
the floors registered on MuonWithAuxAdam optimizers. CPU only.
"""

import pytest
import torch

from muon.muon import floored_weight_decay_, SingleDeviceMuonWithAuxAdam


def _row_norms(w):
    return torch.linalg.vector_norm(w, dim=tuple(range(1, w.dim())))


def test_floored_decay_matches_formula_and_leaves_floor_rows_alone():
    torch.manual_seed(0)
    a = 1e-2
    floor = 0.8
    w = torch.randn(8, 5)
    w[0].zero_()          # all-zero channel: must stay zero, no NaN
    w[1] *= 0.1           # well below the floor: unchanged
    w[2] *= floor / w[2].norm()  # exactly at the floor: unchanged
    before = w.clone()
    r_before = _row_norms(before)

    floored_weight_decay_(w, a, floor)

    assert torch.isfinite(w).all()
    r_after = _row_norms(w)
    expected = torch.where(r_before > floor, r_before - a * (r_before - floor), r_before)
    assert torch.allclose(r_after, expected, rtol=1e-6, atol=1e-7)
    assert torch.equal(w[0], before[0])
    assert torch.equal(w[1], before[1])
    assert torch.equal(w[2], before[2])
    # Directions are preserved for rows that decayed.
    for j in range(3, 8):
        cos = torch.dot(w[j], before[j]) / (w[j].norm() * before[j].norm())
        assert cos > 1 - 1e-6


def test_floored_decay_plain_cases():
    torch.manual_seed(1)
    a = 3e-3
    w = torch.randn(6, 4)
    ref = w * (1 - a)
    floored_weight_decay_(w, a, 0.0)
    assert torch.allclose(w, ref)

    bias = torch.randn(7)
    ref = bias * (1 - a)
    floored_weight_decay_(bias, a, 5.0)
    assert torch.allclose(bias, ref)


def test_floored_decay_conv_shape_reduces_over_all_non_output_dims():
    torch.manual_seed(2)
    a = 1e-2
    w = torch.randn(4, 3, 3, 3)
    floor = 1.0
    w[0] *= 0.1 / w[0].norm()  # below the floor
    before = w.clone()
    floored_weight_decay_(w, a, floor)
    assert torch.equal(w[0], before[0])
    r_before = _row_norms(before)
    r_after = _row_norms(w)
    expected = torch.where(r_before > floor, r_before - a * (r_before - floor), r_before)
    assert torch.allclose(r_after, expected, rtol=1e-6, atol=1e-7)


def _make_optimizer(muon_params, adam_params, floors):
    groups = [
        {"params": muon_params, "use_muon": True, "lr": 1e-3, "weight_decay": 0.5},
        {"params": adam_params, "use_muon": False, "lr": 1e-3, "weight_decay": 0.5},
    ]
    opt = SingleDeviceMuonWithAuxAdam(groups, adjust_lr_fn="match_rms_adamw")
    # The batched Newton-Schulz path is torch.compiled, and inductor's CPU backend fails on the
    # tiny matrices used here. The scalar path applies the same decay code.
    opt.use_batched_muon_ns = False
    opt.set_weight_decay_floor_norms(floors)
    return opt


def test_apply_updates_with_weight_decay_routes_mixed_lists_like_the_reference():
    """_apply_updates_with_weight_decay on a mix of floored 2-D, floored 4-D, unfloored 2-D and
    1-D tensors must give the same result as floored_weight_decay_ / plain decay on each tensor
    followed by the plain update step, and the update must not be rounded to a narrower dtype."""
    torch.manual_seed(5)
    lin_f = torch.nn.Parameter(torch.randn(12, 6))
    conv_f = torch.nn.Parameter(torch.randn(5, 2, 3, 3))
    lin_u = torch.nn.Parameter(torch.randn(7, 4))
    bias = torch.nn.Parameter(torch.randn(9))
    with torch.no_grad():
        lin_f[0].zero_()
        conv_f[1] *= 1e-3
    lr = 1e-2
    weight_decay = 0.2
    a = lr * weight_decay
    floors = {lin_f: 0.7 * _row_norms(lin_f.detach()).median().item(), conv_f: 1.5}
    opt = _make_optimizer([lin_f, conv_f, lin_u], [bias], floors)
    params = [lin_f, conv_f, lin_u, bias]
    updates = [torch.randn_like(p) for p in params]
    # A bf16 update must be converted to fp32 before the decay term is added to it.
    updates[2] = updates[2].bfloat16()

    expected = {}
    for p, u in zip(params, updates):
        e = p.detach().clone()
        floored_weight_decay_(e, a, floors.get(p, 0.0))
        e.add_(u.to(e.dtype), alpha=-lr)
        expected[p] = e
    with torch.no_grad():
        opt._apply_updates_with_weight_decay(params, updates, lr, weight_decay)
    for p in params:
        assert torch.allclose(p.detach(), expected[p], rtol=1e-6, atol=1e-6)
    # The all-zero row of lin_f has zero decay, so it changes by exactly -lr * update.
    assert torch.equal(lin_f.detach()[0], -lr * updates[0][0])

    # Zero decay is the plain update step.
    before = [p.detach().clone() for p in params]
    updates = [torch.randn_like(p) for p in params]
    with torch.no_grad():
        opt._apply_updates_with_weight_decay(params, updates, lr, 0.0)
    for p, b, u in zip(params, before, updates):
        assert torch.allclose(p.detach(), b - lr * u, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("use_floors", [False, True])
def test_tiny_weight_decay_is_not_rounded_away(use_floors):
    """With a = lr * weight_decay below the fp32 spacing near 1.0, a standalone multiply by
    (1 - a) would leave fp32 weights unchanged. Folded into a nonzero update, the decay must
    still be applied in expectation. Muon and Adam updates depend only on the gradients and
    optimizer state, not on the weights, so two optimizers fed identical gradients with and
    without weight decay differ only by the accumulated decay."""
    torch.manual_seed(6)
    lr = 1e-3
    weight_decay = 1e-5
    a = lr * weight_decay
    assert a < 3e-8
    n = 40
    init = {
        "lin": torch.randn(256, 256),
        "conv": torch.randn(64, 16, 3, 3),
        "vec": torch.randn(4096),
    }
    # Some floored rows well above the floor, some at or below it.
    init["lin"][:8] *= 0.3
    floor = 0.5 * _row_norms(init["lin"]).median().item() if use_floors else 0.0

    grads = [{k: torch.randn_like(v) for k, v in init.items()} for _ in range(n)]

    def run(wd):
        torch.manual_seed(7)
        lin = torch.nn.Parameter(init["lin"].clone())
        conv = torch.nn.Parameter(init["conv"].clone())
        vec = torch.nn.Parameter(init["vec"].clone())
        groups = [
            {"params": [lin, conv], "use_muon": True, "lr": lr, "weight_decay": wd},
            {"params": [vec], "use_muon": False, "lr": lr, "weight_decay": wd},
        ]
        opt = SingleDeviceMuonWithAuxAdam(groups, adjust_lr_fn="match_rms_adamw")
        opt.use_batched_muon_ns = False
        if floor > 0.0:
            opt.set_weight_decay_floor_norms({lin: floor})
        for step_grads in grads:
            lin.grad = step_grads["lin"].clone()
            conv.grad = step_grads["conv"].clone()
            vec.grad = step_grads["vec"].clone()
            opt.step()
        return {"lin": lin.detach(), "conv": conv.detach(), "vec": vec.detach()}

    with_wd = run(weight_decay)
    without_wd = run(0.0)

    def fitted_decay(diff, ref):
        """Least-squares coefficient c in diff = -c * ref, in float64."""
        d = diff.double().flatten()
        r = ref.double().flatten()
        return -(d * r).sum().item() / (r * r).sum().item()

    for key in ("conv", "vec"):
        c = fitted_decay(with_wd[key] - without_wd[key], without_wd[key])
        assert c == pytest.approx(n * a, rel=0.1), key

    diff = with_wd["lin"] - without_wd["lin"]
    ref = without_wd["lin"]
    if not use_floors:
        assert fitted_decay(diff, ref) == pytest.approx(n * a, rel=0.1)
        return
    # Rows at or below the floor get a decay term of exactly zero, so the two runs agree exactly.
    # Rows well above the floor decay by the fraction of their norm above it. The per-row fit has
    # a few percent of rounding scatter, so the tight check is on the average over rows.
    r = _row_norms(ref)
    ratios = []
    for j in range(ref.shape[0]):
        if r[j] <= floor:
            assert torch.equal(diff[j], torch.zeros_like(diff[j]))
        elif r[j] > 1.5 * floor:
            expected = n * a * (r[j].item() - floor) / r[j].item()
            ratio = fitted_decay(diff[j], ref[j]) / expected
            assert ratio == pytest.approx(1.0, rel=0.5)
            ratios.append(ratio)
    assert (r <= floor).sum() > 0 and len(ratios) > 0
    assert sum(ratios) / len(ratios) == pytest.approx(1.0, rel=0.1)


def test_optimizer_floors_hold_rows_at_floor_with_zero_gradient():
    """With zero gradients Muon's update is zero, so the only change per step is weight decay,
    and the floored rows must follow r_n = floor + (r_0 - floor) * (1 - a)^n exactly."""
    torch.manual_seed(3)
    lin1 = torch.nn.Parameter(torch.randn(16, 8))
    lin2 = torch.nn.Parameter(torch.randn(8, 16))
    bias = torch.nn.Parameter(torch.randn(8))
    with torch.no_grad():
        lin1[0] *= 1e-3  # a nearly dead channel that must not shrink further
    r0_1 = _row_norms(lin1.detach()).clone()
    r0_2 = _row_norms(lin2.detach()).clone()
    b0 = bias.detach().clone()

    floor1 = 0.5 * r0_1.median().item()
    opt = _make_optimizer([lin1, lin2], [bias], {lin1: floor1})
    a = 1e-3 * 0.5
    n = 20
    for _ in range(n):
        for p in (lin1, lin2, bias):
            p.grad = torch.zeros_like(p)
        opt.step()

    r1 = _row_norms(lin1.detach())
    expected1 = torch.where(r0_1 > floor1, floor1 + (r0_1 - floor1) * (1 - a) ** n, r0_1)
    assert torch.allclose(r1, expected1, rtol=1e-5, atol=1e-6)
    assert r1[0].item() == pytest.approx(r0_1[0].item())
    # lin2 has no floor and the aux Adam bias has no floor: plain decay.
    assert torch.allclose(_row_norms(lin2.detach()), r0_2 * (1 - a) ** n, rtol=1e-5, atol=1e-6)
    assert torch.allclose(bias.detach(), b0 * (1 - a) ** n, rtol=1e-5, atol=1e-6)


def test_optimizer_without_floors_is_plain_decay_and_rejects_foreign_tensors():
    torch.manual_seed(4)
    lin1 = torch.nn.Parameter(torch.randn(16, 8))
    lin2 = torch.nn.Parameter(torch.randn(8, 16))
    bias = torch.nn.Parameter(torch.randn(8))
    r0 = _row_norms(lin1.detach()).clone()
    opt = _make_optimizer([lin1, lin2], [bias], {})
    for p in (lin1, lin2, bias):
        p.grad = torch.zeros_like(p)
    opt.step()
    assert torch.allclose(_row_norms(lin1.detach()), r0 * (1 - 1e-3 * 0.5), rtol=1e-6, atol=1e-7)

    with pytest.raises(ValueError):
        opt.set_weight_decay_floor_norms({torch.nn.Parameter(torch.randn(2, 2)): 1.0})
