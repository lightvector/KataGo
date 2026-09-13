"""
Tests that the single-device and distributed Muon optimizers key their state by the same
parameter order, and that a state dict whose per-parameter tensors do not fit the parameters
is dropped instead of crashing. CPU only.
"""

import copy

import torch

from muon.muon import SingleDeviceMuonWithAuxAdam


def _params():
    torch.manual_seed(0)
    small = torch.nn.Parameter(torch.randn(4, 8))
    big = torch.nn.Parameter(torch.randn(32, 16))
    mid = torch.nn.Parameter(torch.randn(16, 8))
    bias = torch.nn.Parameter(torch.randn(8))
    return small, big, mid, bias


def _optimizer(small, big, mid, bias):
    groups = [
        {"params": [small, big, mid], "use_muon": True, "lr": 1e-3, "weight_decay": 0.0},
        {"params": [bias], "use_muon": False, "lr": 1e-3, "weight_decay": 0.0},
    ]
    opt = SingleDeviceMuonWithAuxAdam(groups, adjust_lr_fn="match_rms_adamw")
    opt.use_batched_muon_ns = False
    return opt


def _step(opt, params):
    for p in params:
        p.grad = torch.randn_like(p)
    opt.step()


def test_single_device_sorts_muon_params_by_size():
    small, big, mid, bias = _params()
    opt = _optimizer(small, big, mid, bias)
    assert [p.shape for p in opt.param_groups[0]["params"]] == [big.shape, mid.shape, small.shape]
    assert opt.param_groups[1]["params"] == [bias]


def test_matching_state_dict_loads_and_mismatched_one_is_dropped(caplog):
    params = _params()
    opt = _optimizer(*params)
    _step(opt, params)
    saved = copy.deepcopy(opt.state_dict())

    fresh = _optimizer(*_params())
    fresh.load_state_dict(saved)
    for p_saved, p_fresh in zip(opt.param_groups[0]["params"], fresh.param_groups[0]["params"]):
        assert torch.equal(opt.state[p_saved]["momentum_buffer"], fresh.state[p_fresh]["momentum_buffer"])

    # A checkpoint written under a different parameter order: the per-position tensors no
    # longer fit the parameters at those positions.
    permuted = copy.deepcopy(saved)
    ids = permuted["param_groups"][0]["params"]
    permuted["state"][ids[0]], permuted["state"][ids[2]] = permuted["state"][ids[2]], permuted["state"][ids[0]]
    fresh = _optimizer(*_params())
    with caplog.at_level("WARNING"):
        fresh.load_state_dict(permuted)
    assert len(fresh.state) == 0
    assert any("Dropping optimizer state" in rec.getMessage() for rec in caplog.records)

    # After the drop the optimizer still steps normally.
    _step(fresh, [p for g in fresh.param_groups for p in g["params"]])
    assert len(fresh.state) == 4
