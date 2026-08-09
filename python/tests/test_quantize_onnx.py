"""Focused tests for the research ONNX quantization exporter.

The synthetic graphs below mirror the names emitted by onnxmodelbuilder.cpp.  In
particular, an nbt3 block has three attention/FFN pairs, so the target b15 model
has 15 * 3 * (4 attention projections + 3 FFN projections) = 315 eligible
weight projections.  Attention score/value MatMuls deliberately have no
initializer and must remain unquantized.
"""

from __future__ import annotations

import numpy as np
import pytest


onnx = pytest.importorskip("onnx")
quantize_onnx = pytest.importorskip(
    "katago.quantization", reason="KataGo quantization helpers have not been added yet"
)


def _make_target_projection_graph(*, nhwc: bool):
    helper = onnx.helper
    tensor_proto = onnx.TensorProto
    nodes = []
    initializers = []
    expected = []
    current = "input"

    def add_weight_node(name: str, op_type: str) -> None:
        nonlocal current
        weight_name = name + (".Wnhwc" if nhwc else ".W")
        if op_type == "MatMul":
            weight = helper.make_tensor(weight_name, tensor_proto.FLOAT, [1, 1], [1.0])
        else:
            weight = helper.make_tensor(
                weight_name, tensor_proto.FLOAT, [1, 1, 1, 1], [1.0]
            )
        output = name + "/output"
        nodes.append(
            helper.make_node(op_type, [current, weight_name], [output], name=name)
        )
        initializers.append(weight)
        current = output

    projection_op = "MatMul" if nhwc else "Conv"
    suffix = "/nhwc" if nhwc else ""
    for outer_idx in range(15):
        for pair_idx in range(3):
            attn_idx = pair_idx * 2
            ffn_idx = attn_idx + 1
            attn_base = f"model.blocks.{outer_idx}.blockstack.{attn_idx}"
            ffn_base = f"model.blocks.{outer_idx}.blockstack.{ffn_idx}"
            for projection in ("q_proj", "k_proj", "v_proj", "out_proj"):
                name = f"{attn_base}.{projection}{suffix}"
                add_weight_node(name, projection_op)
                expected.append(name)
            for projection in ("ffn_linear1", "ffn_linear_gate", "ffn_linear2"):
                name = f"{ffn_base}.{projection}{suffix}"
                add_weight_node(name, projection_op)
                expected.append(name)

    # Activation x activation attention MatMuls are precision-sensitive and are
    # never weight projections, despite living in the transformer scope.
    nodes.append(
        helper.make_node(
            "MatMul",
            [current, current],
            ["scores"],
            name="model.blocks.0.blockstack.0/scores",
        )
    )
    nodes.append(
        helper.make_node(
            "MatMul",
            ["scores", current],
            ["sv"],
            name="model.blocks.0.blockstack.0/sv",
        )
    )

    # These all have constant weights, but are outside the deliberately narrow
    # transformer-projection scope.
    add_weight_node("model.conv_spatial", "Conv")
    add_weight_node("model.blocks.0.normactconvp.conv" + suffix, projection_op)
    add_weight_node("model.policy_head.conv1p", "Conv")
    add_weight_node("model.value_head.linear2" + suffix, projection_op)

    # A projection-looking activation MatMul and a non-projection op are both
    # negative controls for name-only selection.
    nodes.append(
        helper.make_node(
            "MatMul",
            [current, current],
            ["fake_projection"],
            name="model.blocks.0.blockstack.0.q_proj/activation_only",
        )
    )
    fake_weight = helper.make_tensor("fake_add.W", tensor_proto.FLOAT, [1], [1.0])
    initializers.append(fake_weight)
    nodes.append(
        helper.make_node(
            "Add",
            [current, "fake_add.W"],
            ["final"],
            name="model.blocks.0.blockstack.0.q_proj" + suffix,
        )
    )

    graph = helper.make_graph(
        nodes,
        "target_projection_selection",
        [helper.make_tensor_value_info("input", tensor_proto.FLOAT, [None, 1, 1, 1])],
        [helper.make_tensor_value_info("final", tensor_proto.FLOAT, None)],
        initializer=initializers,
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 20)])
    return model, expected


@pytest.mark.parametrize("nhwc", [False, True])
def test_selects_exactly_315_b15_nbt3_weight_projections(nhwc):
    model, expected = _make_target_projection_graph(nhwc=nhwc)

    selected = quantize_onnx.select_quantizable_nodes(model, scope="transformer")

    assert selected.selected_names == sorted(expected)
    assert len(selected.selected_names) == 315
    assert selected.selected_by_op_type == {"MatMul" if nhwc else "Conv": 315}


@pytest.mark.parametrize("nhwc", [False, True])
def test_all_weighted_matches_reference_matmul_conv_scope(nhwc):
    """Match the executable scope in zml24's reference INT8 script.

    That script passes every weighted MatMul and Conv to ORT and excludes only
    activation-only attention MatMuls. Its declared stem/head skip patterns are
    not wired into the quantization call.
    """

    model, transformer = _make_target_projection_graph(nhwc=nhwc)
    suffix = "/nhwc" if nhwc else ""
    expected = transformer + [
        "model.conv_spatial",
        "model.blocks.0.normactconvp.conv" + suffix,
        "model.policy_head.conv1p",
        "model.value_head.linear2" + suffix,
    ]

    selected = quantize_onnx.select_quantizable_nodes(model, scope="all-weighted")

    assert selected.selected_names == sorted(expected)
    assert len(selected.selected_names) == 319
    assert "model.blocks.0.blockstack.0/scores" not in selected.selected_names
    assert "model.blocks.0.blockstack.0/sv" not in selected.selected_names
    assert not any("activation_only" in name for name in selected.selected_names)


def test_only_node_regex_restricts_selection_to_one_outer_block_ffn():
    model, _ = _make_target_projection_graph(nhwc=True)

    selected = quantize_onnx.select_quantizable_nodes(
        model,
        scope="transformer",
        only_regexes=(r"^model\.blocks\.7\..*ffn_",),
    )
    assert len(selected.selected_names) == 9
    assert all(
        name.startswith("model.blocks.7.") and "ffn_" in name
        for name in selected.selected_names
    )


def _make_five_output_contract_without_provenance():
    helper = onnx.helper
    tp = onnx.TensorProto
    inputs = [
        helper.make_tensor_value_info("InputMask", tp.FLOAT, [None, 1, 19, 19]),
        helper.make_tensor_value_info("InputSpatial", tp.FLOAT, [None, 22, 19, 19]),
        helper.make_tensor_value_info("InputGlobal", tp.FLOAT, [None, 19, 1, 1]),
    ]
    outputs = [
        helper.make_tensor_value_info("OutputPolicyPass", tp.FLOAT, [None, 4, 1, 1]),
        helper.make_tensor_value_info("OutputPolicy", tp.FLOAT, [None, 4, 19, 19]),
        helper.make_tensor_value_info("OutputValue", tp.FLOAT, [None, 3, 1, 1]),
        helper.make_tensor_value_info("OutputScoreValue", tp.FLOAT, [None, 6, 1, 1]),
        helper.make_tensor_value_info("OutputOwnership", tp.FLOAT, [None, 1, 19, 19]),
    ]
    return helper.make_model(helper.make_graph([], "derived", inputs, outputs))


def test_incremental_parent_may_lack_provenance_but_not_five_output_contract():
    model = _make_five_output_contract_without_provenance()
    with pytest.raises(ValueError, match="producer_name=katago"):
        quantize_onnx.validate_katago_io_contract(model)

    specs = quantize_onnx.validate_katago_io_contract(
        model, require_producer_metadata=False
    )
    assert [spec.name for spec in specs] == [
        "InputMask",
        "InputSpatial",
        "InputGlobal",
    ]

    model.graph.output[4].name = "WrongOwnership"
    with pytest.raises(ValueError, match="Expected the five raw outputs"):
        quantize_onnx.validate_katago_io_contract(
            model, require_producer_metadata=False
        )


def _input_specs(height: int, width: int):
    return [
        quantize_onnx.InputSpec(
            "InputMask", (None, 1, height, width), onnx.TensorProto.FLOAT
        ),
        quantize_onnx.InputSpec(
            "InputSpatial", (None, 22, height, width), onnx.TensorProto.FLOAT
        ),
        quantize_onnx.InputSpec(
            "InputGlobal", (None, 19, 1, 1), onnx.TensorProto.FLOAT
        ),
    ]


def _write_training_npz(path, spatial: np.ndarray, global_input: np.ndarray) -> None:
    packed = np.packbits(
        spatial.reshape(spatial.shape[0], spatial.shape[1], -1), axis=2
    )
    np.savez_compressed(
        path,
        binaryInputNCHWPacked=packed,
        globalInputNC=global_input,
    )


def _concatenate_inputs(dataset):
    return {
        name: np.concatenate([batch[name] for batch in dataset.batches], axis=0)
        for name in ("InputMask", "InputSpatial", "InputGlobal")
    }


def _make_identifiable_positions(sample_count: int, height: int, width: int):
    spatial = np.zeros((sample_count, 22, height, width), dtype=np.uint8)
    for row in range(sample_count):
        # Different channels and rows exercise both pack-byte boundaries and
        # the final padding bits when H*W is not divisible by eight.
        flat = spatial[row].reshape(22, -1)
        for channel in range(22):
            flat[channel, (row * 3 + channel * 5) % (height * width)] = 1
    global_input = np.ones((sample_count, 19), dtype=np.float32)
    global_input[:, 18] = np.arange(sample_count, dtype=np.float32)
    return spatial, global_input


def test_training_npz_unpack_and_full_history_round_trip(tmp_path):
    height, width, sample_count = 3, 5, 7
    spatial, global_input = _make_identifiable_positions(sample_count, height, width)
    path = tmp_path / "positions.npz"
    _write_training_npz(path, spatial, global_input)

    dataset = quantize_onnx.load_position_dataset(
        [str(path)],
        _input_specs(height, width),
        sample_count=sample_count,
        batch_size=3,
        seed=12345,
        history_mode="full",
        symmetry_mode="none",
    )
    actual = _concatenate_inputs(dataset)

    assert [batch["InputSpatial"].shape[0] for batch in dataset.batches] == [3, 3, 1]
    assert dataset.sample_count == sample_count
    assert dataset.history_mode == "full"
    assert all(value.dtype == np.float32 for value in actual.values())
    assert all(value.flags.c_contiguous for value in actual.values())

    # The sampler intentionally shuffles rows.  The last global feature is a
    # stable row id, allowing an exact comparison without relying on order.
    row_ids = actual["InputGlobal"][:, 18, 0, 0].astype(np.int64)
    np.testing.assert_array_equal(actual["InputSpatial"], spatial[row_ids])
    np.testing.assert_array_equal(actual["InputMask"], spatial[row_ids, 0:1])
    np.testing.assert_array_equal(
        actual["InputGlobal"][:, :, 0, 0], global_input[row_ids]
    )


def test_none_history_matches_katago_history_matrix_semantics(tmp_path):
    height, width, sample_count = 3, 5, 8
    spatial, global_input = _make_identifiable_positions(sample_count, height, width)
    path = tmp_path / "positions.npz"
    _write_training_npz(path, spatial, global_input)

    dataset = quantize_onnx.load_position_dataset(
        [str(path)],
        _input_specs(height, width),
        sample_count=sample_count,
        batch_size=sample_count,
        seed=9,
        history_mode="none",
        symmetry_mode="none",
    )
    actual = _concatenate_inputs(dataset)
    row_ids = actual["InputGlobal"][:, 18, 0, 0].astype(np.int64)
    transformed = actual["InputSpatial"]

    np.testing.assert_array_equal(transformed[:, 9:14], 0.0)
    np.testing.assert_array_equal(transformed[:, 14], spatial[row_ids, 14])
    np.testing.assert_array_equal(transformed[:, 15], spatial[row_ids, 14])
    np.testing.assert_array_equal(transformed[:, 16], spatial[row_ids, 14])
    np.testing.assert_array_equal(actual["InputGlobal"][:, :5], 0.0)
    np.testing.assert_array_equal(
        actual["InputGlobal"][:, 5:, 0, 0], global_input[row_ids, 5:]
    )


def test_training_history_is_seeded_and_prefix_shaped(tmp_path):
    height, width, sample_count = 3, 5, 256
    spatial, global_input = _make_identifiable_positions(sample_count, height, width)
    path = tmp_path / "positions.npz"
    _write_training_npz(path, spatial, global_input)

    kwargs = dict(
        paths=[str(path)],
        input_specs=_input_specs(height, width),
        sample_count=sample_count,
        batch_size=31,
        seed=20260808,
        history_mode="training",
        symmetry_mode="none",
    )
    first = quantize_onnx.load_position_dataset(**kwargs)
    second = quantize_onnx.load_position_dataset(**kwargs)
    first_inputs = _concatenate_inputs(first)
    second_inputs = _concatenate_inputs(second)

    assert first.position_sha256 == second.position_sha256
    assert first.selection_sha256 == second.selection_sha256
    for name in first_inputs:
        np.testing.assert_array_equal(first_inputs[name], second_inputs[name])

    history_flags = first_inputs["InputGlobal"][:, :5, 0, 0]
    assert np.any(history_flags == 0.0)
    assert np.any(np.all(history_flags == 1.0, axis=1))
    # Included history is always a prefix: 1,1,...,1,0,...,0.
    assert np.all(np.diff(history_flags, axis=1) <= 0.0)


def test_calibration_reader_supports_modelopt_reader_protocol(tmp_path):
    height, width, sample_count = 3, 5, 7
    spatial, global_input = _make_identifiable_positions(sample_count, height, width)
    path = tmp_path / "positions.npz"
    _write_training_npz(path, spatial, global_input)
    dataset = quantize_onnx.load_position_dataset(
        [str(path)],
        _input_specs(height, width),
        sample_count=sample_count,
        batch_size=3,
        seed=12,
        history_mode="full",
        symmetry_mode="none",
    )
    reader = quantize_onnx.ArrayCalibrationDataReader(dataset)

    assert len(reader) == 3
    assert reader.get_first() is dataset.batches[0]
    assert reader.get_next() is dataset.batches[0]
    reader.rewind()
    assert reader.get_next() is dataset.batches[0]
    reader.set_range(1, 3)
    assert len(reader) == 2
    assert reader.get_first() is dataset.batches[1]
    ranged = list(reader)
    assert len(ranged) == 2
    assert all(
        actual is expected for actual, expected in zip(ranged, dataset.batches[1:3])
    )


def _expected_spatial_symmetry(value: np.ndarray, symmetry: int) -> np.ndarray:
    if symmetry == 0:
        return value
    if symmetry == 1:
        return np.rot90(value, k=1, axes=(-2, -1))
    if symmetry == 2:
        return np.rot90(value, k=2, axes=(-2, -1))
    if symmetry == 3:
        return np.rot90(value, k=3, axes=(-2, -1))
    if symmetry == 4:
        return np.swapaxes(value, -2, -1)
    if symmetry == 5:
        return np.flip(value, axis=-1)
    if symmetry == 6:
        return np.flip(np.swapaxes(value, -2, -1), axis=(-2, -1))
    if symmetry == 7:
        return np.flip(value, axis=-2)
    raise AssertionError(symmetry)


def test_all_symmetries_expand_each_source_position_in_numbered_order(tmp_path):
    height = width = 5
    spatial, global_input = _make_identifiable_positions(4, height, width)
    path = tmp_path / "positions.npz"
    _write_training_npz(path, spatial, global_input)

    kwargs = dict(
        paths=[str(path)],
        input_specs=_input_specs(height, width),
        sample_count=3,
        batch_size=10,
        seed=42,
        history_mode="full",
        symmetry_mode="all",
    )
    first = quantize_onnx.load_position_dataset(**kwargs)
    second = quantize_onnx.load_position_dataset(**kwargs)
    actual = _concatenate_inputs(first)

    assert first.base_sample_count == 3
    assert first.sample_count == 24
    assert [batch["InputSpatial"].shape[0] for batch in first.batches] == [10, 10, 4]
    assert first.symmetry_mode == "all"
    assert first.symmetry_counts == {str(symmetry): 3 for symmetry in range(8)}
    assert first.symmetry_sha256 == second.symmetry_sha256
    assert first.selection_sha256 == second.selection_sha256
    assert first.position_sha256 == second.position_sha256

    row_ids = actual["InputGlobal"][:, 18, 0, 0].astype(np.int64).reshape(3, 8)
    for base_index in range(3):
        assert np.all(row_ids[base_index] == row_ids[base_index, 0])
        source_row = row_ids[base_index, 0]
        for symmetry in range(8):
            output_index = base_index * 8 + symmetry
            np.testing.assert_array_equal(
                actual["InputSpatial"][output_index],
                _expected_spatial_symmetry(spatial[source_row], symmetry),
            )
            np.testing.assert_array_equal(
                actual["InputMask"][output_index],
                actual["InputSpatial"][output_index, 0:1],
            )
            np.testing.assert_array_equal(
                actual["InputGlobal"][output_index, :, 0, 0], global_input[source_row]
            )


def test_random_symmetries_are_seeded_and_transform_mask_with_spatial(tmp_path):
    height = width = 5
    sample_count = 64
    spatial, global_input = _make_identifiable_positions(sample_count, height, width)
    path = tmp_path / "positions.npz"
    _write_training_npz(path, spatial, global_input)
    kwargs = dict(
        paths=[str(path)],
        input_specs=_input_specs(height, width),
        sample_count=sample_count,
        batch_size=17,
        seed=20260808,
        history_mode="full",
        symmetry_mode="random",
    )

    first = quantize_onnx.load_position_dataset(**kwargs)
    second = quantize_onnx.load_position_dataset(**kwargs)
    first_inputs = _concatenate_inputs(first)
    second_inputs = _concatenate_inputs(second)

    assert first.sample_count == first.base_sample_count == sample_count
    assert sum(first.symmetry_counts.values()) == sample_count
    assert sum(count > 0 for count in first.symmetry_counts.values()) > 1
    assert first.symmetry_sha256 == second.symmetry_sha256
    assert first.position_sha256 == second.position_sha256
    for name in first_inputs:
        np.testing.assert_array_equal(first_inputs[name], second_inputs[name])

    row_ids = first_inputs["InputGlobal"][:, 18, 0, 0].astype(np.int64)
    for output_index, source_row in enumerate(row_ids):
        possible = [
            _expected_spatial_symmetry(spatial[source_row], symmetry)
            for symmetry in range(8)
        ]
        assert any(
            np.array_equal(first_inputs["InputSpatial"][output_index], candidate)
            for candidate in possible
        )
        np.testing.assert_array_equal(
            first_inputs["InputMask"][output_index],
            first_inputs["InputSpatial"][output_index, 0:1],
        )
        np.testing.assert_array_equal(
            first_inputs["InputGlobal"][output_index, :, 0, 0], global_input[source_row]
        )


def test_symmetry_augmentation_rejects_rectangular_graph(tmp_path):
    spatial, global_input = _make_identifiable_positions(2, 3, 5)
    path = tmp_path / "positions.npz"
    _write_training_npz(path, spatial, global_input)

    with pytest.raises(ValueError, match="square ONNX board"):
        quantize_onnx.load_position_dataset(
            [str(path)],
            _input_specs(3, 5),
            sample_count=2,
            batch_size=2,
            seed=1,
            history_mode="full",
            symmetry_mode="random",
        )


def test_expanded_inputs_apply_requested_history_transform(tmp_path):
    height = width = 3
    spatial, global_input = _make_identifiable_positions(4, height, width)
    path = tmp_path / "expanded.npz"
    np.savez_compressed(
        path,
        InputMask=spatial[:, 0:1].astype(np.float32),
        InputSpatial=spatial.astype(np.float32),
        InputGlobal=global_input[:, :, None, None],
    )

    dataset = quantize_onnx.load_position_dataset(
        [str(path)],
        _input_specs(height, width),
        sample_count=4,
        batch_size=4,
        seed=33,
        history_mode="none",
        symmetry_mode="none",
    )
    actual = _concatenate_inputs(dataset)

    np.testing.assert_array_equal(actual["InputSpatial"][:, 9:14], 0.0)
    np.testing.assert_array_equal(actual["InputGlobal"][:, :5], 0.0)


def test_source_file_limit_is_seeded_and_bounds_npz_decompression(tmp_path):
    height = width = 3
    for file_index in range(5):
        spatial, global_input = _make_identifiable_positions(10, height, width)
        global_input[:, 18] = file_index * 100 + np.arange(10, dtype=np.float32)
        _write_training_npz(tmp_path / f"shard-{file_index}.npz", spatial, global_input)

    kwargs = dict(
        paths=[str(tmp_path)],
        input_specs=_input_specs(height, width),
        sample_count=8,
        batch_size=4,
        seed=99,
        history_mode="full",
        symmetry_mode="none",
        max_source_files=2,
    )
    first = quantize_onnx.load_position_dataset(**kwargs)
    second = quantize_onnx.load_position_dataset(**kwargs)

    assert first.available_source_file_count == 5
    assert first.selected_source_file_count == 2
    assert first.max_source_files == 2
    assert len(first.source_files) == 2
    assert first.selection_sha256 == second.selection_sha256
    assert first.position_sha256 == second.position_sha256


def _raw_outputs(sample_count=2, height=2, width=2, policy_channels=2):
    return {
        "OutputPolicyPass": np.zeros(
            (sample_count, policy_channels, 1, 1), dtype=np.float32
        ),
        "OutputPolicy": np.zeros(
            (sample_count, policy_channels, height, width), dtype=np.float32
        ),
        "OutputValue": np.zeros((sample_count, 3, 1, 1), dtype=np.float32),
        "OutputScoreValue": np.zeros((sample_count, 6, 1, 1), dtype=np.float32),
        "OutputOwnership": np.zeros((sample_count, 1, height, width), dtype=np.float32),
    }


def test_validation_metrics_mask_offboard_policy_and_ownership():
    reference = _raw_outputs()
    candidate = {name: value.copy() for name, value in reference.items()}
    mask = np.ones((2, 1, 2, 2), dtype=np.float32)
    mask[0, 0, 1, 1] = 0.0
    candidate["OutputPolicy"][0, :, 1, 1] = 1000.0
    candidate["OutputOwnership"][0, 0, 1, 1] = 1000.0

    metrics = quantize_onnx.compute_validation_metrics(
        reference, candidate, {"InputMask": mask}
    )

    assert metrics["policy"]["kl"]["max"] == pytest.approx(0.0, abs=1.0e-12)
    assert metrics["outputs"]["OutputOwnership"]["max_abs"] == 0.0
    assert metrics["outputs"]["OutputOwnership"]["unmasked"]["max_abs"] == 1000.0
    assert metrics["outputs"]["OutputOwnership"]["offboard"]["max_abs"] == 1000.0
    # The generic raw-output audit intentionally remains unmasked, making it
    # possible to spot unexpectedly large off-board activations too.
    assert metrics["outputs"]["OutputPolicy"]["max_abs"] == 1000.0


def test_four_channel_q_outputs_use_raw_error_not_policy_kl():
    reference = _raw_outputs(policy_channels=4)
    candidate = {name: value.copy() for name, value in reference.items()}
    mask = np.ones((2, 1, 2, 2), dtype=np.float32)
    candidate["OutputPolicy"][:, 2] += 3.0
    candidate["OutputPolicyPass"][:, 2] += 3.0
    candidate["OutputPolicy"][:, 3] -= 2.0
    candidate["OutputPolicyPass"][:, 3] -= 2.0

    metrics = quantize_onnx.compute_validation_metrics(
        reference, candidate, {"InputMask": mask}
    )

    assert metrics["policy"]["kl"]["max"] == pytest.approx(0.0, abs=1.0e-12)
    assert set(metrics["policy"]["per_channel"]) == {
        "policy",
        "shortterm_optimistic",
    }
    assert metrics["policy"]["quantitative"]["q_value"]["rmse"] == pytest.approx(3.0)
    assert metrics["policy"]["quantitative"]["q_score"]["max_abs"] == pytest.approx(2.0)
    gated = quantize_onnx.evaluate_accuracy_gates(
        metrics,
        {
            "max_q_value_rmse": 2.9,
            "max_q_score_max_abs": 2.1,
        },
    )
    assert gated["status"] == "failed"
    assert (
        next(
            check
            for check in gated["checks"]
            if check["threshold"] == "max_q_value_rmse"
        )["passed"]
        is False
    )


def test_validation_metrics_and_accuracy_gates_detect_real_drift():
    reference = _raw_outputs()
    candidate = {name: value.copy() for name, value in reference.items()}
    mask = np.ones((2, 1, 2, 2), dtype=np.float32)
    candidate["OutputPolicyPass"][0, 0, 0, 0] = 1.0
    candidate["OutputValue"][0, 1, 0, 0] = 1.0
    candidate["OutputScoreValue"][0, 2, 0, 0] = 0.25
    candidate["OutputOwnership"][0, 0, 0, 0] = 0.1

    metrics = quantize_onnx.compute_validation_metrics(
        reference, candidate, {"InputMask": mask}
    )
    assert metrics["policy"]["kl"]["mean"] > 0.0
    assert metrics["value"]["kl"]["mean"] > 0.0
    assert metrics["outputs"]["OutputScoreValue"]["max_abs"] == 0.25
    assert metrics["outputs"]["OutputOwnership"]["rmse"] > 0.0

    failed = quantize_onnx.evaluate_accuracy_gates(
        metrics,
        {
            "max_policy_kl_mean": 0.0,
            "max_value_kl_mean": 0.0,
            "max_scorevalue_max_abs": 0.1,
            "max_ownership_rmse": 0.001,
        },
    )
    assert failed["status"] == "failed"
    assert any(not check["passed"] for check in failed["checks"])

    permissive = quantize_onnx.evaluate_accuracy_gates(
        metrics,
        {
            "max_policy_kl_mean": 1.0,
            "max_value_kl_mean": 1.0,
            "max_scorevalue_max_abs": 1.0,
            "max_ownership_rmse": 1.0,
        },
    )
    assert permissive["status"] == "passed"


def test_nonfinite_candidate_always_fails_without_optional_thresholds():
    reference = _raw_outputs()
    candidate = {name: value.copy() for name, value in reference.items()}
    candidate["OutputScoreValue"][0, 0, 0, 0] = np.nan
    metrics = quantize_onnx.compute_validation_metrics(
        reference,
        candidate,
        {"InputMask": np.ones((2, 1, 2, 2), dtype=np.float32)},
    )

    result = quantize_onnx.evaluate_accuracy_gates(metrics, {})

    assert result["status"] == "failed"
    assert result["numeric_thresholds_configured"] is False
    assert result["checks"][-1]["threshold"] == "candidate_nonfinite"
    assert result["checks"][-1]["passed"] is False
