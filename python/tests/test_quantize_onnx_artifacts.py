"""Offline tests for safe ONNX artifact staging and replacement."""

from __future__ import annotations

import shutil
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


onnx = pytest.importorskip("onnx")
exporter = pytest.importorskip("quantize_onnx")


def _write_model(
    path: Path,
    value: float,
    *,
    external_location: str | None = None,
) -> None:
    helper = onnx.helper
    tensor_proto = onnx.TensorProto
    weight = onnx.numpy_helper.from_array(
        np.full((1, 16), value, dtype=np.float32), name="weight"
    )
    model = helper.make_model(
        helper.make_graph(
            [helper.make_node("Add", ["input", "weight"], ["output"], name="add")],
            "artifact-test",
            [helper.make_tensor_value_info("input", tensor_proto.FLOAT, [1, 16])],
            [helper.make_tensor_value_info("output", tensor_proto.FLOAT, [1, 16])],
            [weight],
        ),
        opset_imports=[helper.make_opsetid("", 17)],
    )
    if external_location is None:
        onnx.save_model(model, str(path))
    else:
        onnx.save_model(
            model,
            str(path),
            save_as_external_data=True,
            all_tensors_to_one_file=True,
            location=external_location,
            size_threshold=0,
        )


def _weight_value(path: Path) -> float:
    model = onnx.load(str(path), load_external_data=True)
    return float(onnx.numpy_helper.to_array(model.graph.initializer[0]).flat[0])


def _write_fp8_qdq_matmul(path: Path) -> None:
    helper = onnx.helper
    tp = onnx.TensorProto
    activation_scale = onnx.numpy_helper.from_array(
        np.asarray(0.027776, dtype=np.float32), name="activation_scale"
    )
    weight_scale = onnx.numpy_helper.from_array(
        np.asarray([0.01, 0.02, 0.03], dtype=np.float32), name="weight_scale"
    )
    weight = onnx.numpy_helper.from_array(
        np.arange(6, dtype=np.float32).reshape(2, 3), name="weight"
    )
    nodes = [
        helper.make_node(
            "QuantizeLinear",
            ["input", "activation_scale"],
            ["input_q"],
            name="input_q",
            output_dtype=tp.FLOAT8E4M3FN,
        ),
        helper.make_node(
            "DequantizeLinear",
            ["input_q", "activation_scale"],
            ["input_dq"],
            name="input_dq",
        ),
        helper.make_node(
            "QuantizeLinear",
            ["weight", "weight_scale"],
            ["weight_q"],
            name="weight_q",
            axis=1,
            output_dtype=tp.FLOAT8E4M3FN,
        ),
        helper.make_node(
            "DequantizeLinear",
            ["weight_q", "weight_scale"],
            ["weight_dq"],
            name="weight_dq",
            axis=1,
        ),
        helper.make_node(
            "MatMul",
            ["input_dq", "weight_dq"],
            ["output"],
            name="model.blocks.0.blockstack.1.ffn_linear1/nhwc",
        ),
    ]
    model = helper.make_model(
        helper.make_graph(
            nodes,
            "fp8-scale-test",
            [helper.make_tensor_value_info("input", tp.FLOAT, [None, 2])],
            [helper.make_tensor_value_info("output", tp.FLOAT, [None, 3])],
            [activation_scale, weight_scale, weight],
        ),
        opset_imports=[helper.make_opsetid("", 21)],
    )
    onnx.save_model(model, str(path))


def _write_incremental_int8_parent(path: Path) -> None:
    helper = onnx.helper
    tp = onnx.TensorProto
    initializers = [
        onnx.numpy_helper.from_array(
            np.arange(6, dtype=np.float32).reshape(2, 3), name="old_weight"
        ),
        onnx.numpy_helper.from_array(np.asarray(0.1, dtype=np.float32), name="old_as"),
        onnx.numpy_helper.from_array(np.asarray(0, dtype=np.int8), name="old_az"),
        onnx.numpy_helper.from_array(
            np.asarray([0.02, 0.03, 0.04], dtype=np.float32), name="old_ws"
        ),
        onnx.numpy_helper.from_array(
            np.asarray([0, 0, 0], dtype=np.int8), name="old_wz"
        ),
        onnx.numpy_helper.from_array(
            np.arange(3 * 512, dtype=np.float32).reshape(3, 512), name="new_weight"
        ),
    ]
    nodes = [
        helper.make_node(
            "QuantizeLinear", ["input", "old_as", "old_az"], ["old_aq"], name="old_aq"
        ),
        helper.make_node(
            "DequantizeLinear",
            ["old_aq", "old_as", "old_az"],
            ["old_adq"],
            name="old_adq",
        ),
        helper.make_node(
            "QuantizeLinear",
            ["old_weight", "old_ws", "old_wz"],
            ["old_wq"],
            name="old_wq",
            axis=1,
        ),
        helper.make_node(
            "DequantizeLinear",
            ["old_wq", "old_ws", "old_wz"],
            ["old_wdq"],
            name="old_wdq",
            axis=1,
        ),
        helper.make_node("MatMul", ["old_adq", "old_wdq"], ["old_output"], name="old"),
        helper.make_node(
            "MatMul",
            ["old_output", "new_weight"],
            ["output"],
            name="model.blocks.0.normactconvp.conv/nhwc",
        ),
    ]
    model = helper.make_model(
        helper.make_graph(
            nodes,
            "incremental-parent",
            [helper.make_tensor_value_info("input", tp.FLOAT, [None, 2])],
            [helper.make_tensor_value_info("output", tp.FLOAT, [None, 512])],
            initializers,
        ),
        opset_imports=[helper.make_opsetid("", 21)],
    )
    model.ir_version = 10
    onnx.save_model(
        model,
        path,
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location=path.name + ".parent.data",
        size_threshold=0,
    )


def _add_incremental_qdq(model) -> None:
    outer = next(
        node
        for node in model.graph.node
        if node.name == "model.blocks.0.normactconvp.conv/nhwc"
    )
    model.graph.initializer.extend(
        [
            onnx.numpy_helper.from_array(
                np.asarray(0.2, dtype=np.float32), name="new_as"
            ),
            onnx.numpy_helper.from_array(np.asarray(0, dtype=np.int8), name="new_az"),
            onnx.numpy_helper.from_array(
                np.asarray([0.05, 0.06], dtype=np.float32), name="new_ws"
            ),
            onnx.numpy_helper.from_array(
                np.asarray([0, 0], dtype=np.int8), name="new_wz"
            ),
        ]
    )
    new_nodes = [
        onnx.helper.make_node(
            "QuantizeLinear",
            ["old_output", "new_as", "new_az"],
            ["new_aq"],
            name="new_aq",
        ),
        onnx.helper.make_node(
            "DequantizeLinear",
            ["new_aq", "new_as", "new_az"],
            ["new_adq"],
            name="new_adq",
        ),
        onnx.helper.make_node(
            "QuantizeLinear",
            ["new_weight", "new_ws", "new_wz"],
            ["new_wq"],
            name="new_wq",
            axis=1,
        ),
        onnx.helper.make_node(
            "DequantizeLinear",
            ["new_wq", "new_ws", "new_wz"],
            ["new_wdq"],
            name="new_wdq",
            axis=1,
        ),
    ]
    outer.input[0] = "new_adq"
    outer.input[1] = "new_wdq"
    index = list(model.graph.node).index(outer)
    for offset, node in enumerate(new_nodes):
        model.graph.node.insert(index + offset, node)


def test_direct_amax_fp8_scale_rewrite_uses_separate_activation_headroom(
    tmp_path: Path,
) -> None:
    path = tmp_path / "fp8.onnx"
    _write_fp8_qdq_matmul(path)
    original = onnx.load(str(path))
    original_values = {
        item.name: onnx.numpy_helper.to_array(item).copy()
        for item in original.graph.initializer
    }

    details = exporter._rewrite_fp8_direct_amax_scales(
        str(path),
        ["model.blocks.0.blockstack.1.ffn_linear1/nhwc"],
        activation_qmax=224.0,
    )

    rewritten = onnx.load(str(path))
    values = {
        item.name: onnx.numpy_helper.to_array(item)
        for item in rewritten.graph.initializer
    }
    legacy_qmax = 127.0**2 / 448.0
    np.testing.assert_allclose(
        values["activation_scale"],
        original_values["activation_scale"] * legacy_qmax / 224.0,
    )
    np.testing.assert_allclose(
        values["weight_scale"],
        original_values["weight_scale"] * legacy_qmax / 448.0,
    )
    np.testing.assert_array_equal(values["weight"], original_values["weight"])
    assert details["scale_initializer_count"] == {"activation": 1, "weight": 1}
    assert details["activation_qmax"] == 224.0


def test_copy_artifact_stages_external_data_without_touching_source(
    tmp_path: Path,
) -> None:
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    source = source_dir / "model.onnx"
    _write_model(source, 3.0, external_location="weights.bin")
    expected = exporter.artifact_manifest(str(source))

    staging_dir = tmp_path / "staging"
    staged = Path(exporter._copy_onnx_artifact(str(source), str(staging_dir)))

    assert staged == staging_dir / source.name
    assert (staging_dir / "weights.bin").is_file()
    assert _weight_value(staged) == pytest.approx(3.0)

    # Simulate ModelOpt's in-place shape-inference rewrite of its input path.
    staged_model = onnx.load(str(staged), load_external_data=False)
    staged_model.producer_name = "mutated-staging-copy"
    onnx.save_model(staged_model, str(staged))

    assert exporter._artifact_integrity(expected)["status"] == "passed"


def test_promote_replaces_only_exact_previous_artifact(tmp_path: Path) -> None:
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    final_model = output_dir / "network.int8.qdq.onnx"
    _write_model(final_model, 1.0, external_location="old.weights.bin")
    old_sidecar = output_dir / "old.weights.bin"
    unrelated = output_dir / "do-not-delete.weights.bin"
    unrelated.write_bytes(b"unrelated")

    staging_dir = output_dir / ".katago-quant-test"
    staging_dir.mkdir()
    staged_model = staging_dir / "candidate.0123456789abcdef.onnx"
    new_location = "candidate.0123456789abcdef.onnx_data"
    _write_model(staged_model, 9.0, external_location=new_location)

    details = exporter._promote_staged_artifact(
        str(staged_model), str(final_model), overwrite=True
    )

    new_sidecar = output_dir / new_location
    assert _weight_value(final_model) == pytest.approx(9.0)
    assert new_sidecar.is_file()
    assert not old_sidecar.exists()
    assert unrelated.read_bytes() == b"unrelated"
    assert str(old_sidecar.resolve()) in details["removed_previous_external_data"]
    assert details["promoted_external_data"] == [str(new_sidecar.resolve())]


def test_promote_without_overwrite_has_no_side_effects(tmp_path: Path) -> None:
    final_model = tmp_path / "existing.onnx"
    _write_model(final_model, 2.0, external_location="existing.weights.bin")
    expected = exporter.artifact_manifest(str(final_model))

    staging_dir = tmp_path / "staging"
    staging_dir.mkdir()
    staged_model = staging_dir / "candidate.onnx"
    _write_model(staged_model, 7.0, external_location="candidate.weights.bin")

    with pytest.raises(FileExistsError, match="--overwrite"):
        exporter._promote_staged_artifact(
            str(staged_model), str(final_model), overwrite=False
        )

    assert exporter._artifact_integrity(expected)["status"] == "passed"
    assert staged_model.is_file()
    assert (staging_dir / "candidate.weights.bin").is_file()


def test_quantize_one_never_passes_source_artifact_to_modelopt(tmp_path: Path) -> None:
    source = tmp_path / "source.onnx"
    _write_model(source, 4.0)
    expected = exporter.artifact_manifest(str(source))
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    output = output_dir / "safe.int8.qdq.onnx"
    seen: dict[str, Path] = {}

    def mutating_quantizer(**kwargs) -> None:
        staged_source = Path(kwargs["onnx_path"]).resolve()
        seen["source"] = staged_source
        assert staged_source != source.resolve()
        staged_model = onnx.load(str(staged_source), load_external_data=False)
        staged_model.producer_name = "modelopt-mutated-this-file"
        onnx.save_model(staged_model, str(staged_source))
        shutil.copy2(staged_source, kwargs["output_path"])

    args = SimpleNamespace(
        overwrite=True,
        output_dir=str(output_dir),
        output_prefix="safe",
        calibration_method="entropy",
        calibration_eps="cpu",
        keep_intermediate_files=False,
        high_precision="fp32",
        calibrate_per_node=False,
    )
    dataset = SimpleNamespace(batches=[{}])

    exporter._quantize_one(
        "int8",
        str(source),
        str(output),
        dataset,
        [],
        [],
        args,
        mutating_quantizer,
    )

    assert seen["source"].parent != source.parent
    assert output.is_file()
    assert exporter._artifact_integrity(expected)["status"] == "passed"
    assert not list(output_dir.glob(".katago-quant-*"))


def test_incremental_quantize_one_passes_only_new_allowlist_and_preserves_parent(
    tmp_path: Path,
) -> None:
    source = tmp_path / "parent.onnx"
    _write_incremental_int8_parent(source)
    snapshot = exporter.capture_existing_qdq_state(str(source))
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    output = output_dir / "incremental.int8.qdq.onnx"
    selected = ["model.blocks.0.normactconvp.conv/nhwc"]

    def incremental_quantizer(**kwargs) -> None:
        raise AssertionError("preserve mode must bypass ModelOpt's Q/DQ short circuit")

    args = SimpleNamespace(
        overwrite=True,
        output_dir=str(output_dir),
        output_prefix="incremental",
        calibration_method="entropy",
        calibration_eps="cpu",
        keep_intermediate_files=False,
        high_precision="fp32",
        calibrate_per_node=False,
    )
    details = exporter._quantize_one(
        "int8",
        str(source),
        str(output),
        SimpleNamespace(
            batches=[
                {"input": np.asarray([[1.0, -2.0], [0.5, 3.0]], dtype=np.float32)},
                {"input": np.asarray([[-1.5, 0.25], [2.0, -0.75]], dtype=np.float32)},
            ]
        ),
        selected,
        ["MatMul"],
        args,
        incremental_quantizer,
        snapshot,
    )

    incremental = details["incremental_quantizer"]
    assert incremental["backend"] == "onnxruntime-node-filtered-qdq"
    assert incremental["newly_quantized_nodes"] == selected
    assert incremental["calibration_tensor_names"] == ["old_output", "output"]
    assert incremental["materialized_parent_raw_bytes"] > 0
    assert incremental["self_contained_output"] is True
    assert incremental["configuration"] == {
        "calibration_method": "entropy",
        "activation_type": "QInt8",
        "weight_type": "QInt8",
        "activation_symmetric": True,
        "weight_symmetric": True,
        "per_channel_weights": True,
        "reduce_range": False,
        "quantize_bias": False,
        "output_quantization": False,
    }
    assert details["incremental_union_selected_count"] == 2
    assert details["existing_qdq_preservation"]["status"] == "passed"
    assert details["incremental_staged_qdq_audit"]["errors"] == []
    assert output.is_file()
    output_members = exporter._external_artifact_members(
        str(output), require_exists=True
    )
    assert output_members
    parent_members = exporter._external_artifact_members(
        str(source), require_exists=True
    )
    assert parent_members
    assert not any(
        output_member.samefile(parent_member)
        for _, output_member in output_members
        for _, parent_member in parent_members
    )

    # Prove the promoted result is not accidentally dependent on the parent payload.
    for _, parent_member in parent_members:
        parent_member.unlink()
    loaded = onnx.load_model(output, load_external_data=True)
    onnx.checker.check_model(loaded, full_check=False)


def test_incremental_quantize_one_refuses_noop_clone_and_cleans_staging(
    tmp_path: Path,
    monkeypatch,
) -> None:
    source = tmp_path / "parent.onnx"
    _write_incremental_int8_parent(source)
    snapshot = exporter.capture_existing_qdq_state(str(source))
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    output = output_dir / "bad.int8.qdq.onnx"

    def no_change_runner(source_path, output_path, *args, **kwargs):
        shutil.copy2(source_path, output_path)
        raise RuntimeError(
            "Incremental ORT quantization produced no usable change; refusing to promote a parent clone"
        )

    monkeypatch.setattr(exporter, "_run_incremental_ort_quantization", no_change_runner)

    args = SimpleNamespace(
        overwrite=True,
        output_dir=str(output_dir),
        output_prefix="bad",
        calibration_method="entropy",
        calibration_eps="cpu",
        keep_intermediate_files=False,
        high_precision="fp32",
        calibrate_per_node=False,
    )
    with pytest.raises(RuntimeError, match="produced no usable change"):
        exporter._quantize_one(
            "int8",
            str(source),
            str(output),
            SimpleNamespace(batches=[{}]),
            ["model.blocks.0.normactconvp.conv/nhwc"],
            ["MatMul"],
            args,
            lambda **kwargs: None,
            snapshot,
        )

    assert not output.exists()
    assert not list(output_dir.glob(".katago-quant-*"))


@pytest.mark.parametrize(
    ("mode_report", "expected"),
    [
        ({}, False),
        ({"accuracy_gate": {"status": "passed"}}, False),
        ({"accuracy_gate": {"status": "failed"}}, True),
        ({"trtexec": {"passed": True}}, False),
        ({"trtexec": {"passed": False}}, True),
    ],
)
def test_mode_report_failed(mode_report, expected: bool) -> None:
    assert exporter._mode_report_failed(mode_report) is expected
