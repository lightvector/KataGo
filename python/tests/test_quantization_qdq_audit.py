"""Adversarial tests for the TensorRT explicit Q/DQ graph audit."""

from __future__ import annotations

import numpy as np
import pytest


onnx = pytest.importorskip("onnx")
quantization = pytest.importorskip("katago.quantization")


def _make_qdq_model(
    *,
    op_type: str = "MatMul",
    output_channels: int = 3,
    trans_b: int = 0,
    scalar_weight_scale: bool = False,
):
    helper = onnx.helper
    tensor_proto = onnx.TensorProto

    if op_type == "Conv":
        input_shape = [None, 4, 5, 5]
        weight_shape = [output_channels, 4, 1, 1]
        weight_axis = 0
    elif op_type == "Gemm" and trans_b:
        input_shape = [None, 4]
        weight_shape = [output_channels, 4]
        weight_axis = 0
    else:
        input_shape = [None, 4]
        weight_shape = [4, output_channels]
        weight_axis = 1

    weight_scale_shape = [] if scalar_weight_scale else [output_channels]
    weight_scale_values = [0.05] if scalar_weight_scale else [0.05] * output_channels
    weight_zero_shape = [] if scalar_weight_scale else [output_channels]
    weight_zero_values = [0] if scalar_weight_scale else [0] * output_channels

    initializers = [
        helper.make_tensor(
            "weight",
            tensor_proto.FLOAT,
            weight_shape,
            np.linspace(-1.0, 1.0, int(np.prod(weight_shape)), dtype=np.float32),
        ),
        helper.make_tensor("act_scale", tensor_proto.FLOAT, [], [0.1]),
        helper.make_tensor("act_zp", tensor_proto.INT8, [], [0]),
        helper.make_tensor(
            "weight_scale", tensor_proto.FLOAT, weight_scale_shape, weight_scale_values
        ),
        helper.make_tensor(
            "weight_zp", tensor_proto.INT8, weight_zero_shape, weight_zero_values
        ),
    ]
    nodes = [
        helper.make_node(
            "QuantizeLinear",
            ["input", "act_scale", "act_zp"],
            ["input_q"],
            name="act_q",
        ),
        helper.make_node(
            "DequantizeLinear",
            ["input_q", "act_scale", "act_zp"],
            ["input_dq"],
            name="act_dq",
        ),
        helper.make_node(
            "QuantizeLinear",
            ["weight", "weight_scale", "weight_zp"],
            ["weight_q"],
            name="weight_q",
            axis=weight_axis,
        ),
        helper.make_node(
            "DequantizeLinear",
            ["weight_q", "weight_scale", "weight_zp"],
            ["weight_dq"],
            name="weight_dq",
            axis=weight_axis,
        ),
    ]
    attributes = {"transB": trans_b} if op_type == "Gemm" else {}
    nodes.append(
        helper.make_node(
            op_type,
            ["input_dq", "weight_dq"],
            ["output"],
            name="selected",
            **attributes,
        )
    )
    graph = helper.make_graph(
        nodes,
        "qdq_audit",
        [helper.make_tensor_value_info("input", tensor_proto.FLOAT, input_shape)],
        [helper.make_tensor_value_info("output", tensor_proto.FLOAT, None)],
        initializer=initializers,
    )
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 21)])


def _node(model, name: str):
    return next(node for node in model.graph.node if node.name == name)


def _replace_initializer(model, name: str, values, dtype) -> None:
    replacement = onnx.numpy_helper.from_array(
        np.asarray(values, dtype=dtype), name=name
    )
    initializer = next(item for item in model.graph.initializer if item.name == name)
    initializer.CopyFrom(replacement)


def _set_axis(node, axis: int) -> None:
    for attribute in node.attribute:
        if attribute.name == "axis":
            attribute.i = axis
            return
    node.attribute.append(onnx.helper.make_attribute("axis", axis))


def _add_initializer(model, name: str, values, dtype) -> None:
    model.graph.initializer.append(
        onnx.numpy_helper.from_array(np.asarray(values, dtype=dtype), name=name)
    )


def _audit(tmp_path, model):
    path = tmp_path / "model.onnx"
    onnx.save(model, path)
    return quantization.audit_qdq_model(str(path), ["selected"])


@pytest.mark.parametrize(
    ("op_type", "trans_b", "expected_axis"),
    [("MatMul", 0, 1), ("Conv", 0, 0), ("Gemm", 0, 1), ("Gemm", 1, 0)],
)
def test_accepts_valid_qdq_and_operator_weight_axis(
    tmp_path, op_type, trans_b, expected_axis
):
    report = _audit(tmp_path, _make_qdq_model(op_type=op_type, trans_b=trans_b))

    assert report["errors"] == []
    assert report["qdq_pair_count"] == 2
    assert report["selected_nodes_with_qdq_inputs"] == 1
    assert report["selected_chain_qtypes"] == {
        "selected": {"activation": "INT8", "weight": "INT8"}
    }
    assert (
        report["selected_input_chains"]["selected"]["weight"]["effective_axis"]
        == expected_axis
    )


def test_allows_scalar_scale_for_single_output_channel(tmp_path):
    model = _make_qdq_model(op_type="Conv", output_channels=1, scalar_weight_scale=True)

    report = _audit(tmp_path, model)

    assert report["errors"] == []
    assert report["selected_input_chains"]["selected"]["weight"]["scale_elements"] == 1


def test_reads_scale_and_zero_point_from_external_data_relative_to_model(tmp_path):
    model = _make_qdq_model(output_channels=3)
    _replace_initializer(model, "act_scale", 0.1, np.float32)
    _replace_initializer(model, "act_zp", 0, np.int8)
    _replace_initializer(model, "weight_scale", [0.05] * 3, np.float32)
    _replace_initializer(model, "weight_zp", [0] * 3, np.int8)
    model_dir = tmp_path / "nested"
    model_dir.mkdir()
    path = model_dir / "model.onnx"
    onnx.save_model(
        model,
        path,
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location="payload.bin",
        size_threshold=0,
    )
    unloaded = onnx.load(path, load_external_data=False)
    unloaded_by_name = {
        initializer.name: initializer for initializer in unloaded.graph.initializer
    }
    assert all(
        unloaded_by_name[name].data_location == onnx.TensorProto.EXTERNAL
        for name in ("act_scale", "act_zp", "weight_scale", "weight_zp")
    )

    report = quantization.audit_qdq_model(str(path), ["selected"])

    assert report["errors"] == []


def test_accepts_equal_qdq_constants_with_different_names(tmp_path):
    model = _make_qdq_model()
    _add_initializer(model, "act_scale_dq", 0.1, np.float32)
    _add_initializer(model, "act_zp_dq", 0, np.int8)
    _node(model, "act_dq").input[1] = "act_scale_dq"
    _node(model, "act_dq").input[2] = "act_zp_dq"

    assert _audit(tmp_path, model)["errors"] == []


def test_accepts_scale_from_constant_node(tmp_path):
    model = _make_qdq_model()
    scale_tensor = onnx.numpy_helper.from_array(np.asarray(0.1, dtype=np.float32))
    model.graph.node.insert(
        0,
        onnx.helper.make_node(
            "Constant",
            [],
            ["act_scale_constant"],
            name="act_scale_constant",
            value=scale_tensor,
        ),
    )
    _node(model, "act_q").input[1] = "act_scale_constant"
    _node(model, "act_dq").input[1] = "act_scale_constant"

    assert _audit(tmp_path, model)["errors"] == []


@pytest.mark.parametrize(
    ("values", "expected_fragment"),
    [
        ([np.nan], "non-finite"),
        ([-0.1], "strictly positive"),
        ([0.0], "strictly positive"),
    ],
)
def test_rejects_nonfinite_or_nonpositive_scale(tmp_path, values, expected_fragment):
    model = _make_qdq_model()
    _replace_initializer(model, "act_scale", values, np.float32)

    report = _audit(tmp_path, model)

    assert any(expected_fragment in error for error in report["errors"])


def test_rejects_nonconstant_scale(tmp_path):
    model = _make_qdq_model()
    _node(model, "act_q").input[1] = "input"
    _node(model, "act_dq").input[1] = "input"

    report = _audit(tmp_path, model)

    assert any("not a readable constant" in error for error in report["errors"])


def test_rejects_different_qdq_scales(tmp_path):
    model = _make_qdq_model()
    _add_initializer(model, "different_scale", 0.2, np.float32)
    _node(model, "act_dq").input[1] = "different_scale"

    report = _audit(tmp_path, model)

    assert any("Q/DQ scales differ" in error for error in report["errors"])


def test_rejects_nonzero_and_different_qdq_zero_points(tmp_path):
    model = _make_qdq_model()
    _add_initializer(model, "different_zp", 1, np.int8)
    _node(model, "act_dq").input[2] = "different_zp"

    report = _audit(tmp_path, model)

    assert any("all-zero" in error for error in report["errors"])
    assert any("Q/DQ zero points differ" in error for error in report["errors"])


def test_rejects_zero_point_with_wrong_target_dtype(tmp_path):
    model = _make_qdq_model()
    _replace_initializer(model, "weight_zp", [0, 0, 0], np.uint8)
    _node(model, "weight_q").attribute.append(
        onnx.helper.make_attribute("output_dtype", onnx.TensorProto.INT8)
    )

    report = _audit(tmp_path, model)

    assert any(
        "zero point type UINT8 does not match quantized type INT8" in error
        for error in report["errors"]
    )


def test_rejects_dq_not_directly_fed_by_q_and_reports_orphans(tmp_path):
    model = _make_qdq_model()
    _node(model, "act_dq").input[0] = "input"

    report = _audit(tmp_path, model)

    assert "act_q" in report["orphan_quantize_linear_nodes"]
    assert "act_dq" in report["orphan_dequantize_linear_nodes"]
    assert report["selected_nodes_without_qdq_inputs"] == ["selected"]
    assert any(
        "not produced directly by QuantizeLinear" in error for error in report["errors"]
    )


def test_rejects_qdq_axis_mismatch_using_default_axis_one(tmp_path):
    model = _make_qdq_model(op_type="MatMul")
    _set_axis(_node(model, "weight_dq"), 0)

    report = _audit(tmp_path, model)

    assert any(
        "Q/DQ effective axes differ (1 vs 0)" in error for error in report["errors"]
    )


@pytest.mark.parametrize(
    ("op_type", "trans_b", "wrong_axis", "expected_axis"),
    [("MatMul", 0, 0, 1), ("Conv", 0, 1, 0), ("Gemm", 0, 0, 1), ("Gemm", 1, 1, 0)],
)
def test_rejects_wrong_weight_axis_for_operator(
    tmp_path, op_type, trans_b, wrong_axis, expected_axis
):
    model = _make_qdq_model(op_type=op_type, trans_b=trans_b)
    _set_axis(_node(model, "weight_q"), wrong_axis)
    _set_axis(_node(model, "weight_dq"), wrong_axis)

    report = _audit(tmp_path, model)

    assert any(
        f"does not match expected axis {expected_axis}" in error
        for error in report["errors"]
    )


def test_normalizes_negative_axis_before_comparison(tmp_path):
    model = _make_qdq_model(op_type="MatMul")
    _set_axis(_node(model, "weight_q"), -1)
    _set_axis(_node(model, "weight_dq"), -1)

    report = _audit(tmp_path, model)

    assert report["errors"] == []
    assert report["selected_input_chains"]["selected"]["weight"]["effective_axis"] == 1


def test_rejects_weight_scale_length_not_equal_to_output_channels(tmp_path):
    model = _make_qdq_model(output_channels=3)
    _replace_initializer(model, "weight_scale", [0.05, 0.05], np.float32)
    _replace_initializer(model, "weight_zp", [0, 0], np.int8)

    report = _audit(tmp_path, model)

    assert any(
        "scale length 2 does not match output channels 3" in error
        for error in report["errors"]
    )


def test_rejects_per_channel_activation_scale(tmp_path):
    model = _make_qdq_model()
    _replace_initializer(model, "act_scale", [0.1, 0.1], np.float32)
    _replace_initializer(model, "act_zp", [0, 0], np.int8)

    report = _audit(tmp_path, model)

    assert any(
        "activation scale is not per-tensor" in error for error in report["errors"]
    )


def test_reports_selected_chain_qtype_and_rejects_mixed_types(tmp_path):
    model = _make_qdq_model()
    _replace_initializer(model, "weight_zp", [0, 0, 0], np.uint8)

    report = _audit(tmp_path, model)

    assert report["selected_chain_qtypes"]["selected"] == {
        "activation": "INT8",
        "weight": "UINT8",
    }
    assert any(
        "unsupported quantized type UINT8" in error for error in report["errors"]
    )
    assert any("selected input chains mix" in error for error in report["errors"])


def test_rejects_nonselected_weighted_node_with_two_qdq_inputs(tmp_path):
    model = _make_qdq_model()
    model.graph.node.append(
        onnx.helper.make_node(
            "MatMul",
            ["input_dq", "weight_dq"],
            ["unexpected_output"],
            name="unselected",
        )
    )

    report = _audit(tmp_path, model)

    assert report["unexpected_fully_quantized_weighted_nodes"] == ["unselected"]
    assert any("non-selected weighted op" in error for error in report["errors"])


def _append_unquantized_outer_p(model):
    weight = onnx.numpy_helper.from_array(
        np.arange(6, dtype=np.float32).reshape(3, 2), name="outer_p_weight"
    )
    model.graph.initializer.append(weight)
    model.graph.node.append(
        onnx.helper.make_node(
            "MatMul",
            ["output", "outer_p_weight"],
            ["outer_p_output"],
            name="model.blocks.0.normactconvp.conv/nhwc",
        )
    )
    model.graph.output[0].name = "outer_p_output"


def _quantize_outer_p_in_place(model):
    outer = _node(model, "model.blocks.0.normactconvp.conv/nhwc")
    activation_scale = onnx.numpy_helper.from_array(
        np.asarray(0.2, dtype=np.float32), name="outer_p_act_scale"
    )
    activation_zp = onnx.numpy_helper.from_array(
        np.asarray(0, dtype=np.int8), name="outer_p_act_zp"
    )
    weight_scale = onnx.numpy_helper.from_array(
        np.asarray([0.03, 0.04], dtype=np.float32), name="outer_p_weight_scale"
    )
    weight_zp = onnx.numpy_helper.from_array(
        np.asarray([0, 0], dtype=np.int8), name="outer_p_weight_zp"
    )
    model.graph.initializer.extend(
        [activation_scale, activation_zp, weight_scale, weight_zp]
    )
    nodes = [
        onnx.helper.make_node(
            "QuantizeLinear",
            ["output", "outer_p_act_scale", "outer_p_act_zp"],
            ["outer_p_act_q_value"],
            name="outer_p_act_q",
        ),
        onnx.helper.make_node(
            "DequantizeLinear",
            ["outer_p_act_q_value", "outer_p_act_scale", "outer_p_act_zp"],
            ["outer_p_act_dq_value"],
            name="outer_p_act_dq",
        ),
        onnx.helper.make_node(
            "QuantizeLinear",
            ["outer_p_weight", "outer_p_weight_scale", "outer_p_weight_zp"],
            ["outer_p_weight_q_value"],
            name="outer_p_weight_q",
            axis=1,
        ),
        onnx.helper.make_node(
            "DequantizeLinear",
            ["outer_p_weight_q_value", "outer_p_weight_scale", "outer_p_weight_zp"],
            ["outer_p_weight_dq_value"],
            name="outer_p_weight_dq",
            axis=1,
        ),
    ]
    outer.input[0] = "outer_p_act_dq_value"
    outer.input[1] = "outer_p_weight_dq_value"
    outer_index = list(model.graph.node).index(outer)
    for offset, node in enumerate(nodes):
        model.graph.node.insert(outer_index + offset, node)


def test_incremental_qdq_snapshot_finds_only_existing_parent_and_freezes_bytes(
    tmp_path,
):
    parent = _make_qdq_model()
    _append_unquantized_outer_p(parent)
    parent_path = tmp_path / "parent.onnx"
    onnx.save_model(
        parent,
        parent_path,
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location="parent.data",
        size_threshold=0,
    )

    selection = quantization.select_quantizable_nodes(
        parent,
        scope="all-weighted",
        only_regexes=(r"normactconvp\.conv",),
    )
    assert selection.selected_names == ["model.blocks.0.normactconvp.conv/nhwc"]
    state = quantization.capture_existing_qdq_state(str(parent_path))
    assert state["selected_names"] == ["selected"]
    assert state["weighted_node_count"] == 1
    assert state["qdq_node_count"] == 4
    assert set(state["initializers"]) == {
        "act_scale",
        "act_zp",
        "weight",
        "weight_scale",
        "weight_zp",
    }

    candidate = onnx.load_model(parent_path, load_external_data=True)
    _quantize_outer_p_in_place(candidate)
    candidate_path = tmp_path / "candidate.onnx"
    onnx.save_model(
        candidate,
        candidate_path,
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location="candidate.data",
        size_threshold=0,
    )

    preservation = quantization.compare_existing_qdq_state(str(candidate_path), state)
    assert preservation["status"] == "passed"
    union_audit = quantization.audit_qdq_model(
        str(candidate_path),
        ["selected", "model.blocks.0.normactconvp.conv/nhwc"],
    )
    assert union_audit["errors"] == []
    assert union_audit["selected_nodes_with_qdq_inputs"] == 2


def test_incremental_qdq_snapshot_detects_old_scale_byte_change(tmp_path):
    parent = _make_qdq_model()
    _append_unquantized_outer_p(parent)
    parent_path = tmp_path / "parent.onnx"
    onnx.save_model(parent, parent_path)
    state = quantization.capture_existing_qdq_state(str(parent_path))

    changed = onnx.load_model(parent_path)
    _replace_initializer(changed, "act_scale", 0.125, np.float32)
    changed_path = tmp_path / "changed.onnx"
    onnx.save_model(changed, changed_path)

    comparison = quantization.compare_existing_qdq_state(str(changed_path), state)
    assert comparison["status"] == "failed"
    assert any(
        "initializers: changed ['act_scale']" in difference
        for difference in comparison["differences"]
    )
