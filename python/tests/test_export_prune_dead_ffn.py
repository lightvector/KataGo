"""prune_dead_ffn_channels in export_model_pytorch.py: removes all-zero FFN hidden channels,
chooses widths within the scratch budget, keeps the last FFN block at full width, and leaves the
model's outputs unchanged."""
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
# export_model_pytorch parses its command line at import.
sys.argv = [sys.argv[0], "-export-dir", "/nonexistent", "-model-name", "x", "-filename-prefix", "x"]
import export_model_pytorch as export  # noqa: E402
from katago.train import modelconfigs  # noqa: E402
from katago.train.model_pytorch import Model, TransformerFFNBlock  # noqa: E402


def _small_model(num_blocks=2, ffn_channels=None):
    config = modelconfigs.config_of_name["b11c768h12nbt3tflrs-fson-silu"].copy()
    config["block_kind"] = config["block_kind"][:num_blocks]
    if ffn_channels is not None:
        config["transformer_ffn_channels"] = ffn_channels
    torch.manual_seed(0)
    model = Model(config, pos_len=19)
    model.initialize()
    model.eval()
    return model


def _random_batch(model):
    torch.manual_seed(1)
    num_bin = modelconfigs.get_num_bin_input_features(model.config)
    num_glob = modelconfigs.get_num_global_input_features(model.config)
    bin_input = (torch.rand(2, num_bin, 19, 19) < 0.3).float()
    bin_input[:, 0] = 1.0
    return bin_input, torch.randn(2, num_glob)


def _flatten(outputs):
    """Model outputs are nested tuples of tensors."""
    if isinstance(outputs, torch.Tensor):
        return [outputs]
    return [t for o in outputs for t in _flatten(o)]


def _zero_channels(block, idx):
    with torch.no_grad():
        block.ffn_linear1.weight[idx] = 0.0
        block.ffn_linear_gate.weight[idx] = 0.0
        block.ffn_linear2.weight[:, idx] = 0.0


def _ffn_blocks(model):
    return [b for _, b in export.ffn_blocks_in_export_order(model)]


def test_granularity():
    assert export.ffn_width_granularity(1152) == 64
    assert export.ffn_width_granularity(1056) == 32
    assert export.ffn_width_granularity(384) == 64
    assert export.ffn_width_granularity(1000) == 8
    assert export.ffn_width_granularity(1001) == 1


def test_choose_ffn_widths_minimizes_total_width_within_budget():
    # Budget 1.5 * 1152 = 1728 = 27 units of 64; the full width 1152 takes 18, leaving 9 units
    # (576 channels) for other widths. {512} alone serves both groups better than any pair
    # of smaller widths, e.g. {128, 448} would send the 500-live blocks to 1152.
    widths = export.choose_ffn_widths([100, 100, 500, 500, 1152], 1152, 64, 1.5)
    assert widths == [512, 512, 512, 512, 1152]
    # With a generous budget every block gets the smallest multiple of 64 that fits.
    widths = export.choose_ffn_widths([100, 100, 500, 500, 1152, 0], 1152, 64, 100.0)
    assert widths == [128, 128, 512, 512, 1152, 64]
    # With no budget beyond the full width, nothing is pruned.
    assert export.choose_ffn_widths([100, 500], 1152, 64, 1.0) == [1152, 1152]
    # A budget below the full width is an error.
    with pytest.raises(Exception, match="scratch-factor"):
        export.choose_ffn_widths([100], 1152, 64, 0.5)
    # 2.3 * 100 evaluates to 229.99999999999997, and the full 230 must still be granted so that
    # {50, 80, 100} fits.
    assert export.choose_ffn_widths([50, 80], 100, 1, 2.3) == [50, 80]
    # Odd widths with granularity 1 stay fast because only the live counts are candidates.
    assert export.choose_ffn_widths([3, 500, 999], 1001, 1, 2.0) == [3, 500, 1001]


def test_prune_keeps_function_and_respects_budget():
    model = _small_model()
    blocks = _ffn_blocks(model)
    assert len(blocks) == 6 and all(b.ffn_dim == 1152 for b in blocks)
    # Block 0: 100 dead -> 1052 live -> 1088 (36 dead channels retained).
    # Block 1: 64 dead -> exactly 1088.
    # Block 2: one channel zero on the input side only, must not count as dead.
    # Block 3: one channel with a tiny but nonzero weight, must not count as dead.
    # Block 4: all dead.
    # Block 5: 500 dead, but it is the last block and stays at full width.
    _zero_channels(blocks[0], torch.arange(5, 105))
    _zero_channels(blocks[1], torch.arange(0, 1152, 18))
    with torch.no_grad():
        blocks[2].ffn_linear1.weight[7] = 0.0
        blocks[2].ffn_linear_gate.weight[7] = 0.0
        _zero_channels(blocks[3], torch.tensor([9]))
        blocks[3].ffn_linear2.weight[0, 9] = 1e-12
    _zero_channels(blocks[4], torch.arange(1152))
    _zero_channels(blocks[5], torch.arange(500))
    bin_input, glob_input = _random_batch(model)
    with torch.no_grad():
        before = model(bin_input, glob_input)

    total_before, total_after, total_live = export.prune_dead_ffn_channels(model, 2.0)

    assert total_before == 6 * 1152
    assert total_live == 1052 + 1088 + 1152 + 1152 + 0 + 652
    # Budget 2 * 1152: besides 1152 itself the set can hold widths summing to 1152, and {64, 1088}
    # is the cheapest choice for these blocks.
    assert [b.ffn_dim for b in blocks] == [1088, 1088, 1152, 1152, 64, 1152]
    assert total_after == sum(b.ffn_dim for b in blocks)
    for b in blocks:
        assert b.ffn_linear1.weight.shape == (b.ffn_dim, b.c_main)
        assert b.ffn_linear_gate.weight.shape == (b.ffn_dim, b.c_main)
        assert b.ffn_linear2.weight.shape == (b.c_main, b.ffn_dim)
        assert b.ffn_linear1.out_features == b.ffn_dim and b.ffn_linear2.in_features == b.ffn_dim
    # The retained dead channels in block 0 are the lowest-index dead ones: 5..40 survive as
    # zeros and 41..104 are gone.
    assert torch.equal(blocks[0].ffn_linear1.weight[5:41], torch.zeros(36, blocks[0].c_main))
    assert blocks[0].ffn_linear1.weight[41].abs().max() > 0
    assert torch.equal(blocks[4].ffn_linear2.weight, torch.zeros(blocks[4].c_main, 64))
    with torch.no_grad():
        after = model(bin_input, glob_input)
    assert len(_flatten(after)) == len(_flatten(before)) > 0
    for x, y in zip(_flatten(before), _flatten(after)):
        assert torch.allclose(x, y, rtol=1e-5, atol=1e-6)


def test_prune_uses_smaller_granularity_for_odd_width():
    model = _small_model(num_blocks=1, ffn_channels=1056)
    blocks = _ffn_blocks(model)
    assert len(blocks) == 3
    _zero_channels(blocks[0], torch.arange(1000))
    _zero_channels(blocks[1], torch.arange(500))
    export.prune_dead_ffn_channels(model, 100.0)
    assert [b.ffn_dim for b in blocks] == [64, 576, 1056]


def test_prune_leaves_fully_live_model_alone():
    model = _small_model()
    blocks = _ffn_blocks(model)
    weights = [b.ffn_linear1.weight.detach().clone() for b in blocks]
    total_before, total_after, total_live = export.prune_dead_ffn_channels(model, 2.0)
    assert total_before == total_after == total_live == 6 * 1152
    for b, w in zip(blocks, weights):
        assert torch.equal(b.ffn_linear1.weight, w)
