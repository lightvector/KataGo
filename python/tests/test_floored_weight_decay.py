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


def test_apply_weight_decay_routes_mixed_lists_like_the_reference():
    """_apply_weight_decay on a mix of floored 2-D, floored 4-D, unfloored 2-D and 1-D tensors
    must give the same result as floored_weight_decay_ / plain decay on each tensor."""
    torch.manual_seed(5)
    lin_f = torch.nn.Parameter(torch.randn(12, 6))
    conv_f = torch.nn.Parameter(torch.randn(5, 2, 3, 3))
    lin_u = torch.nn.Parameter(torch.randn(7, 4))
    bias = torch.nn.Parameter(torch.randn(9))
    with torch.no_grad():
        lin_f[0].zero_()
        conv_f[1] *= 1e-3
    a = 2e-3
    floors = {lin_f: 0.7 * _row_norms(lin_f.detach()).median().item(), conv_f: 1.5}
    opt = _make_optimizer([lin_f, conv_f, lin_u], [bias], floors)

    expected = {}
    for p in (lin_f, conv_f, lin_u, bias):
        e = p.detach().clone()
        floored_weight_decay_(e, a, floors.get(p, 0.0))
        expected[p] = e
    with torch.no_grad():
        opt._apply_weight_decay([lin_f, conv_f, lin_u, bias], a)
    for p in (lin_f, conv_f, lin_u, bias):
        assert torch.allclose(p.detach(), expected[p], rtol=1e-6, atol=1e-7)
    assert torch.equal(lin_f.detach()[0], torch.zeros(6))

    # Zero decay is a no-op.
    before = [p.detach().clone() for p in (lin_f, conv_f, lin_u, bias)]
    with torch.no_grad():
        opt._apply_weight_decay([lin_f, conv_f, lin_u, bias], 0.0)
    for p, b in zip((lin_f, conv_f, lin_u, bias), before):
        assert torch.equal(p.detach(), b)


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
