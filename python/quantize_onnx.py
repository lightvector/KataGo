#!/usr/bin/env python3
"""Export calibrated INT8/FP8 Q/DQ variants of KataGo's emitted ONNX graph.

The input must be the ONNX file produced by the TensorRT backend when
``trtDumpDebugPlanToDir`` is set.  Current KataGo releases do not load the
resulting Q/DQ ONNX files directly; they are research artifacts for TensorRT
``--stronglyTyped`` builds and accuracy/throughput experiments.
"""

from __future__ import annotations

import argparse
import gc
import datetime
import importlib.metadata
import inspect
import logging
import os
import platform
import re
import shutil
import subprocess
import sys
import tempfile
import traceback
import uuid
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np
import onnx

from katago.quantization import (
    ArrayCalibrationDataReader,
    EXPECTED_OUTPUT_NAMES,
    PositionDataset,
    artifact_manifest,
    audit_qdq_model,
    capture_existing_qdq_state,
    compare_existing_qdq_state,
    compute_validation_metrics,
    concatenate_batches,
    evaluate_accuracy_gates,
    json_dump,
    load_position_dataset,
    resolve_npz_files,
    select_quantizable_nodes,
    sha256_files,
    validate_katago_io_contract,
)


MODEL_OPT_VERSION = "0.45.0"


def _build_parser() -> argparse.ArgumentParser:
    description = """
Quantize the exact ONNX inference graph emitted by KataGo's TensorRT backend.
Only transformer q/k/v/out and SwiGLU FFN projections are selected by default;
attention score/value matmuls, norms, stem, outer bottlenecks, trunk tip, and
all output heads stay at the requested high precision.
"""
    parser = argparse.ArgumentParser(
        description=description, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("-onnx-input", "--onnx-input", required=True)
    parser.add_argument("-output-dir", "--output-dir", required=True)
    parser.add_argument(
        "--output-prefix",
        help="Artifact prefix; defaults to the source ONNX stem",
    )
    parser.add_argument(
        "--formats",
        nargs="+",
        choices=("int8", "fp8"),
        default=("int8", "fp8"),
    )
    parser.add_argument(
        "--calibration-data",
        nargs="+",
        required=True,
        help="KataGo training NPZ files/directories/globs, or expanded Input* NPZ files",
    )
    parser.add_argument(
        "--validation-data",
        nargs="+",
        help="Disjoint held-out NPZ files/directories/globs",
    )
    parser.add_argument("--calibration-samples", type=int, default=2048)
    parser.add_argument("--validation-samples", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=20260808)
    parser.add_argument(
        "--max-source-files",
        type=int,
        default=64,
        help=(
            "Bound NPZ shards decompressed per dataset; 0 samples uniformly from every shard. "
            "Official shuffled shards make the bounded default much faster with little bias"
        ),
    )
    parser.add_argument(
        "--history-mode",
        choices=("training", "full", "none"),
        default="training",
        help="Deterministic training-style history truncation, or force all/no history planes",
    )
    parser.add_argument(
        "--symmetry-mode",
        choices=("random", "all", "none"),
        default="random",
        help=(
            "Apply one seeded D4 symmetry per source position, expand each source position "
            "to all eight symmetries, or preserve stored orientation"
        ),
    )
    parser.add_argument(
        "--allow-data-overlap",
        action="store_true",
        help="Allow calibration and validation to use any of the same NPZ shards",
    )
    parser.add_argument(
        "--skip-validation",
        action="store_true",
        help="Export without held-out numerical comparison (never sufficient for release)",
    )

    parser.add_argument(
        "--scope", choices=("transformer", "all-weighted"), default="transformer"
    )
    parser.add_argument(
        "--include-node-regex",
        action="append",
        default=[],
        help="Add weighted nodes matching this regex to the selected scope",
    )
    parser.add_argument(
        "--exclude-node-regex",
        action="append",
        default=[],
        help="Remove selected weighted nodes matching this regex",
    )
    parser.add_argument(
        "--only-node-regex",
        action="append",
        default=[],
        help=(
            "Restrict the selected scope to nodes matching at least one regex. "
            "Useful for reproducible block-level sensitivity searches"
        ),
    )
    parser.add_argument(
        "--expected-quantized-nodes",
        type=int,
        help="Fail if node selection differs (315 for b15c1024h16nbt3tflrs)",
    )
    parser.add_argument(
        "--preserve-existing-qdq",
        action="store_true",
        help=(
            "Incrementally calibrate only the newly selected INT8 nodes while strictly "
            "preserving every pre-existing weighted Q/DQ chain and referenced initializer"
        ),
    )
    parser.add_argument(
        "--expected-existing-qdq-nodes",
        type=int,
        help=(
            "With --preserve-existing-qdq, fail unless exactly this many fully quantized "
            "weighted nodes already exist in the input graph"
        ),
    )
    parser.add_argument(
        "--calibration-method",
        choices=("entropy", "max"),
        default="entropy",
        help="INT8 calibration method",
    )
    parser.add_argument(
        "--fp8-calibration-method",
        choices=("max", "entropy"),
        default="max",
        help="FP8 calibration method; ModelOpt 0.45 supports conversion from max calibration",
    )
    parser.add_argument(
        "--fp8-scale-mode",
        choices=("direct-amax", "modelopt-legacy"),
        default="direct-amax",
        help=(
            "Use direct E4M3 amax/qmax scales, consistent with ModelOpt's Torch exporter, "
            "or retain ModelOpt 0.45's lossy INT8-to-FP8 conversion for comparison"
        ),
    )
    parser.add_argument(
        "--fp8-activation-qmax",
        type=float,
        default=448.0,
        help=(
            "Effective positive E4M3 activation range for direct-amax scaling. "
            "Lower values reserve headroom for calibration-set outliers; weights always use 448"
        ),
    )
    parser.add_argument(
        "--allow-fp8-nonmax-calibration",
        action="store_true",
        help="Permit an unsupported research experiment using non-max FP8 calibration",
    )
    parser.add_argument(
        "--calibration-eps",
        default="cuda:0,cpu",
        help="Ordered ModelOpt calibration execution providers",
    )
    parser.add_argument(
        "--high-precision",
        choices=("fp32", "fp16"),
        default="fp32",
        help="Fallback precision; fp32 isolates quantization error",
    )
    parser.add_argument(
        "--allow-global-fp16-fallback",
        action="store_true",
        help=(
            "Acknowledge that ModelOpt converts all non-quantized fallback ops to FP16; "
            "this is not the same as KataGo's selective FP32 precision pins"
        ),
    )
    parser.add_argument(
        "--calibrate-per-node",
        action="store_true",
        help="Lower calibration memory at substantial runtime cost",
    )
    parser.add_argument("--keep-intermediate-files", action="store_true")
    parser.add_argument(
        "--allow-unpinned-modelopt",
        action="store_true",
        help=f"Permit a ModelOpt version other than the tested {MODEL_OPT_VERSION}",
    )

    parser.add_argument(
        "--validation-ep",
        default="cuda:0",
        help="ONNX Runtime provider: cpu, cuda:N, or trt",
    )
    parser.add_argument(
        "--allow-validation-ep-fallback",
        action="store_true",
        help="Allow an unavailable requested CUDA/TRT EP to fall back entirely to CPU",
    )
    parser.add_argument("--ort-intra-op-threads", type=int, default=0)
    parser.add_argument("--max-policy-kl-mean", type=float)
    parser.add_argument("--max-policy-kl-p99", type=float)
    parser.add_argument("--max-value-kl-mean", type=float)
    parser.add_argument("--max-ownership-rmse", type=float)
    parser.add_argument("--max-scorevalue-max-abs", type=float)
    parser.add_argument("--max-score-mean-max-abs", type=float)
    parser.add_argument("--max-score-mean-sq-max-abs", type=float)
    parser.add_argument("--max-lead-max-abs", type=float)
    parser.add_argument("--max-q-value-rmse", type=float)
    parser.add_argument("--max-q-value-max-abs", type=float)
    parser.add_argument("--max-q-score-rmse", type=float)
    parser.add_argument("--max-q-score-max-abs", type=float)
    parser.add_argument("--min-policy-top1-agreement", type=float)

    parser.add_argument(
        "--trtexec",
        help="Optionally parse/build each Q/DQ graph using this trtexec executable",
    )
    parser.add_argument("--trt-opt-batch", type=int, default=32)
    parser.add_argument("--trt-max-batch", type=int, default=64)
    parser.add_argument("--trt-workspace-mib", type=int, default=0)
    parser.add_argument("--trt-timeout-seconds", type=int, default=3600)

    parser.add_argument(
        "--skip-onnx-check",
        action="store_true",
        help="Skip ONNX checker on source/output",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow ModelOpt/report files at the selected paths to be replaced",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Attempt the other format if one quantization path fails",
    )
    return parser


def _setup_logging(output_dir: str, output_prefix: str) -> str:
    log_path = os.path.join(output_dir, f"{output_prefix}.quantization.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_path, mode="w", encoding="utf-8"),
        ],
        force=True,
    )
    return log_path


def _git_revision() -> Optional[str]:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parent.parent,
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        ).stdout.strip()
    except Exception:
        return None


def _gpu_summary() -> List[Dict[str, str]]:
    executable = shutil.which("nvidia-smi")
    if executable is None:
        return []
    try:
        output = subprocess.run(
            [
                executable,
                "--query-gpu=index,name,compute_cap,memory.total,driver_version",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=15,
        ).stdout
        result = []
        for line in output.splitlines():
            values = [value.strip() for value in line.split(",")]
            if len(values) == 5:
                result.append(
                    dict(
                        index=values[0],
                        name=values[1],
                        compute_capability=values[2],
                        memory_mib=values[3],
                        driver_version=values[4],
                    )
                )
        return result
    except Exception:
        return []


def _optional_package_versions() -> Dict[str, Optional[str]]:
    distributions = (
        "onnxruntime-gpu",
        "onnxruntime",
        "nvidia-modelopt",
        "onnx-graphsurgeon",
        "polygraphy",
        "tensorrt",
    )
    versions: Dict[str, Optional[str]] = {}
    for distribution in distributions:
        try:
            versions[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            versions[distribution] = None
    return versions


def _fp8_hardware_summary(gpus: Sequence[Mapping[str, str]]) -> Dict[str, Any]:
    capable_indices: List[str] = []
    for gpu in gpus:
        try:
            major_text, minor_text = gpu["compute_capability"].split(".", 1)
            capability = int(major_text) * 10 + int(minor_text)
        except (KeyError, TypeError, ValueError):
            continue
        if capability >= 89:
            capable_indices.append(str(gpu["index"]))
    return {
        "minimum_compute_capability": "8.9 (Ada or newer)",
        "capable_gpu_indices": capable_indices,
        "hardware_acceleration_detected": bool(capable_indices),
    }


def _load_modelopt(allow_unpinned: bool):
    try:
        import modelopt  # type: ignore[import-not-found]
        from modelopt.onnx.quantization import quantize  # type: ignore[import-not-found]
    except ImportError as exc:
        raise RuntimeError(
            "NVIDIA Model Optimizer ONNX support is required. Install "
            f"python/requirements-quantization.txt (nvidia-modelopt[onnx]=={MODEL_OPT_VERSION})."
        ) from exc
    version = getattr(modelopt, "__version__", "unknown")
    if version != MODEL_OPT_VERSION and not allow_unpinned:
        raise RuntimeError(
            f"Expected NVIDIA Model Optimizer {MODEL_OPT_VERSION}, found {version}. "
            "Use --allow-unpinned-modelopt only for an intentional compatibility experiment."
        )
    return quantize, version


def _preload_ort_gpu_dependencies() -> None:
    """Load CUDA/cuDNN wheels before ORT creates a GPU execution provider.

    Recent ``onnxruntime-gpu`` wheels intentionally do not depend on the large
    NVIDIA runtime wheels unless the ``[cuda,cudnn]`` extra is installed.  Even
    when those wheels are present, their library directories are not normally
    on Linux's loader path.  ORT 1.21+ exposes this helper so a session does not
    silently fall back to CPU merely because (for example) libcublasLt.so.12
    was not preloaded.
    """

    import onnxruntime as ort  # type: ignore[import-untyped]

    preload = getattr(ort, "preload_dlls", None)
    if callable(preload):
        preload(directory="")


def _provider_spec(requested: str) -> tuple[List[Any], str]:
    import onnxruntime as ort  # type: ignore[import-untyped]

    if requested != "cpu":
        _preload_ort_gpu_dependencies()
    available = set(ort.get_available_providers())
    if requested == "cpu":
        name = "CPUExecutionProvider"
        providers: List[Any] = [name]
    elif requested.startswith("cuda:"):
        name = "CUDAExecutionProvider"
        device_id = int(requested.split(":", 1)[1])
        providers = [(name, {"device_id": device_id}), "CPUExecutionProvider"]
    elif requested == "trt":
        name = "TensorrtExecutionProvider"
        providers = [name, "CUDAExecutionProvider", "CPUExecutionProvider"]
    else:
        raise ValueError(f"Unknown validation EP: {requested}")
    if name not in available:
        raise RuntimeError(
            f"Requested {name}, but ONNX Runtime only has {sorted(available)}"
        )
    return providers, name


def _run_ort_model(
    model_path: str,
    dataset: PositionDataset,
    requested_ep: str,
    intra_op_threads: int,
    allow_ep_fallback: bool,
) -> tuple[List[Dict[str, np.ndarray]], Dict[str, Any]]:
    import onnxruntime as ort  # type: ignore[import-untyped]

    options = ort.SessionOptions()
    # Use the same unoptimized execution graph for the FP32 reference and every
    # candidate. ORT 1.22's optimizer may incorrectly rewrite an FP8 Q/DQ
    # MatMul into MatMulIntegerToFloat, whose input contract is INT8. Running
    # the explicit Q/DQ nodes without graph rewrites correctly emulates FP8 for
    # held-out error measurement.
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    if intra_op_threads > 0:
        options.intra_op_num_threads = intra_op_threads
    providers, required_provider = _provider_spec(requested_ep)
    logging.info("Loading %s with ONNX Runtime providers %s", model_path, providers)
    session = ort.InferenceSession(
        model_path, sess_options=options, providers=providers
    )
    active_providers = list(session.get_providers())
    fell_back_entirely = required_provider not in active_providers
    if fell_back_entirely and not allow_ep_fallback:
        raise RuntimeError(
            f"Requested {required_provider}, but the created ONNX Runtime session only has "
            f"{active_providers}. This commonly means a CUDA/cuDNN DLL failed to load. "
            "Fix the environment or explicitly use --allow-validation-ep-fallback."
        )
    actual_outputs = [output.name for output in session.get_outputs()]
    if actual_outputs != list(EXPECTED_OUTPUT_NAMES):
        raise RuntimeError(
            f"Unexpected output contract in {model_path}: {actual_outputs}"
        )
    results: List[Dict[str, np.ndarray]] = []
    for batch_index, batch in enumerate(dataset.batches):
        values = session.run(list(EXPECTED_OUTPUT_NAMES), batch)
        results.append(dict(zip(EXPECTED_OUTPUT_NAMES, values)))
        if (batch_index + 1) % 10 == 0 or batch_index + 1 == len(dataset.batches):
            logging.info(
                "Validated %d/%d batches", batch_index + 1, len(dataset.batches)
            )
    del session
    return results, {
        "requested": requested_ep,
        "required_provider": required_provider,
        "configured_providers": providers,
        "active_providers": active_providers,
        "entire_session_fallback": fell_back_entirely,
        "fallback_allowed": allow_ep_fallback,
        "graph_optimization": "ORT_DISABLE_ALL",
        "note": (
            "Provider registration is verified. Individual unsupported nodes may still fall "
            "back unless ONNX Runtime is configured separately to forbid per-node CPU fallback."
        ),
    }


def _validation_metrics(
    reference_batches: Sequence[Mapping[str, np.ndarray]],
    candidate_batches: Sequence[Mapping[str, np.ndarray]],
    dataset: PositionDataset,
) -> Dict[str, Any]:
    if len(reference_batches) != len(candidate_batches):
        raise RuntimeError("Reference and candidate validation batch counts differ")
    reference = concatenate_batches(reference_batches, EXPECTED_OUTPUT_NAMES)
    candidate = concatenate_batches(candidate_batches, EXPECTED_OUTPUT_NAMES)
    feed = concatenate_batches(dataset.batches, ["InputMask"])
    return compute_validation_metrics(reference, candidate, feed)


def _dynamic_shape_string(input_specs, batch: int) -> str:
    items = []
    for spec in input_specs:
        dims = [batch if axis == 0 else dim for axis, dim in enumerate(spec.shape)]
        if any(dim is None for dim in dims):
            raise ValueError(
                f"trtexec validation needs fixed non-batch dimensions: {spec}"
            )
        items.append(f"{spec.name}:" + "x".join(str(int(dim)) for dim in dims))
    return ",".join(items)


def _run_trtexec(
    executable: str,
    model_path: str,
    input_specs,
    opt_batch: int,
    max_batch: int,
    workspace_mib: int,
    timeout_seconds: int,
    overwrite: bool,
) -> Dict[str, Any]:
    resolved = shutil.which(executable) or (
        executable if os.path.isfile(executable) else None
    )
    if resolved is None:
        raise FileNotFoundError(f"Could not find trtexec: {executable}")
    if not (1 <= opt_batch <= max_batch):
        raise ValueError("Require 1 <= --trt-opt-batch <= --trt-max-batch")
    engine_path = str(Path(model_path).with_suffix(".engine"))
    layer_info_path = str(Path(model_path).with_suffix(".layers.json"))
    for generated_path in (engine_path, layer_info_path):
        if os.path.exists(generated_path):
            if not overwrite:
                raise FileExistsError(
                    f"Refusing to replace TensorRT artifact {generated_path}; use --overwrite"
                )
            os.remove(generated_path)
    command = [
        resolved,
        f"--onnx={model_path}",
        "--stronglyTyped",
        "--skipInference",
        "--builderOptimizationLevel=5",
        "--profilingVerbosity=detailed",
        "--dumpLayerInfo",
        f"--exportLayerInfo={layer_info_path}",
        f"--saveEngine={engine_path}",
        f"--minShapes={_dynamic_shape_string(input_specs, 1)}",
        f"--optShapes={_dynamic_shape_string(input_specs, opt_batch)}",
        f"--maxShapes={_dynamic_shape_string(input_specs, max_batch)}",
    ]
    if workspace_mib > 0:
        command.append(f"--memPoolSize=workspace:{workspace_mib}MiB")
    logging.info("Building strongly typed TensorRT network: %s", " ".join(command))
    version_process = subprocess.run(
        [resolved, "--version"],
        capture_output=True,
        text=True,
        timeout=min(timeout_seconds, 30),
    )
    process = subprocess.run(
        command,
        capture_output=True,
        text=True,
        timeout=timeout_seconds,
    )
    combined = (process.stdout or "") + "\n" + (process.stderr or "")
    tail = combined.splitlines()[-250:]
    passed = process.returncode == 0
    generated = {}
    for label, path in (("engine", engine_path), ("layer_info", layer_info_path)):
        if os.path.isfile(path):
            generated[label] = {
                "path": path,
                "size": int(os.path.getsize(path)),
                "sha256": sha256_files([path]),
            }
    return {
        "command": command,
        "returncode": int(process.returncode),
        "passed": passed,
        "output_tail": tail,
        "version_output": (
            (version_process.stdout or "") + "\n" + (version_process.stderr or "")
        ).strip(),
        "artifacts": generated,
        "precision_audit": (
            "Layer information is exported for inspection. A successful build alone does not "
            "prove that every selected GEMM ran in INT8/FP8 or establish TensorRT accuracy."
        ),
    }


def _accuracy_thresholds(args: argparse.Namespace) -> Dict[str, Optional[float]]:
    return {
        "max_policy_kl_mean": args.max_policy_kl_mean,
        "max_policy_kl_p99": args.max_policy_kl_p99,
        "max_value_kl_mean": args.max_value_kl_mean,
        "max_ownership_rmse": args.max_ownership_rmse,
        "max_scorevalue_max_abs": args.max_scorevalue_max_abs,
        "max_score_mean_max_abs": args.max_score_mean_max_abs,
        "max_score_mean_sq_max_abs": args.max_score_mean_sq_max_abs,
        "max_lead_max_abs": args.max_lead_max_abs,
        "max_q_value_rmse": args.max_q_value_rmse,
        "max_q_value_max_abs": args.max_q_value_max_abs,
        "max_q_score_rmse": args.max_q_score_rmse,
        "max_q_score_max_abs": args.max_q_score_max_abs,
        "min_policy_top1_agreement": args.min_policy_top1_agreement,
    }


def _external_artifact_members(
    model_path: str, *, require_exists: bool
) -> List[tuple[Path, Path]]:
    """Return safe ``(relative location, absolute path)`` external-data members.

    ONNX external-data locations are interpreted relative to the model file.  A
    quantization artifact must never be allowed to escape that directory: these
    paths are later copied, replaced, and removed without globs.
    """

    primary = Path(model_path).resolve()
    base = primary.parent
    model = onnx.load(str(primary), load_external_data=False)
    members: Dict[str, tuple[Path, Path]] = {}
    for initializer in model.graph.initializer:
        if initializer.data_location != onnx.TensorProto.EXTERNAL:
            continue
        locations = [
            entry.value
            for entry in initializer.external_data
            if entry.key == "location"
        ]
        if len(locations) != 1 or not locations[0]:
            raise ValueError(
                f"External initializer {initializer.name!r} must have exactly one location"
            )
        relative = Path(locations[0])
        if relative.is_absolute() or relative.drive or ".." in relative.parts:
            raise ValueError(
                f"Unsafe external-data location {locations[0]!r} in {primary}"
            )
        absolute = (base / relative).resolve()
        try:
            normalized_relative = absolute.relative_to(base)
        except ValueError as exc:
            raise ValueError(
                f"External-data location {locations[0]!r} escapes {base}"
            ) from exc
        if absolute == primary:
            raise ValueError(f"External data aliases its ONNX file: {absolute}")
        key = os.path.normcase(str(absolute))
        members[key] = (normalized_relative, absolute)

    result = [members[key] for key in sorted(members)]
    if require_exists:
        missing = [str(absolute) for _, absolute in result if not absolute.is_file()]
        if missing:
            raise FileNotFoundError(f"Missing ONNX external data files: {missing}")
    return result


def _copy_onnx_artifact(source_model_path: str, destination_dir: str) -> str:
    """Copy an ONNX file and every referenced external-data file as one artifact."""

    source = Path(source_model_path).resolve()
    destination = Path(destination_dir).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    target_model = destination / source.name
    if target_model.exists():
        raise FileExistsError(f"Staging model already exists: {target_model}")

    shutil.copy2(source, target_model)
    try:
        for relative, external_source in _external_artifact_members(
            str(source), require_exists=True
        ):
            external_target = (destination / relative).resolve()
            try:
                external_target.relative_to(destination)
            except ValueError as exc:
                raise ValueError(f"Unsafe staging target: {external_target}") from exc
            if external_target == target_model:
                raise ValueError(
                    f"External data aliases staged model: {external_target}"
                )
            external_target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(external_source, external_target)
    except Exception:
        if target_model.exists():
            target_model.unlink()
        raise
    return str(target_model)


def _promote_staged_artifact(
    staged_model_path: str, final_model_path: str, *, overwrite: bool
) -> Dict[str, Any]:
    """Commit a complete staged artifact, swapping the primary ONNX file last.

    The staged output name is unique, therefore its external-data names are also
    unique.  Moving those files first leaves an existing artifact valid until
    the atomic primary-file replacement.  Only sidecars referenced by the old
    primary file are removed afterwards.
    """

    staged_model = Path(staged_model_path).resolve()
    final_model = Path(final_model_path).resolve()
    final_dir = final_model.parent
    if not staged_model.is_file():
        raise FileNotFoundError(f"Staged ONNX output is missing: {staged_model}")
    if final_model.exists() and not overwrite:
        raise FileExistsError(f"Refusing to replace {final_model}; use --overwrite")

    old_external: List[Path] = []
    if final_model.exists():
        old_external = [
            absolute
            for _, absolute in _external_artifact_members(
                str(final_model), require_exists=False
            )
        ]

    new_members = _external_artifact_members(str(staged_model), require_exists=True)
    staged_dir = staged_model.parent
    new_targets: List[tuple[Path, Path]] = []
    old_keys = {os.path.normcase(str(path)) for path in old_external}
    for relative, staged_external in new_members:
        target = (final_dir / relative).resolve()
        try:
            target.relative_to(final_dir)
        except ValueError as exc:
            raise ValueError(f"Unsafe promoted external-data target: {target}") from exc
        # Reusing the old sidecar name would invalidate the old primary before
        # it is atomically swapped. Staged names are deliberately unique.
        if target.exists() or os.path.normcase(str(target)) in old_keys:
            raise FileExistsError(
                f"Staged external-data name is not unique and cannot be safely promoted: {target}"
            )
        try:
            staged_external.relative_to(staged_dir)
        except ValueError as exc:
            raise ValueError(
                f"Staged external data escapes its directory: {staged_external}"
            ) from exc
        new_targets.append((staged_external, target))

    moved_targets: List[Path] = []
    primary_promoted = False
    try:
        for source, target in new_targets:
            target.parent.mkdir(parents=True, exist_ok=True)
            os.replace(source, target)
            moved_targets.append(target)
        os.replace(staged_model, final_model)
        primary_promoted = True
    finally:
        if not primary_promoted:
            for target in moved_targets:
                try:
                    target.unlink()
                except FileNotFoundError:
                    pass

    new_keys = {os.path.normcase(str(target)) for _, target in new_targets}
    removed_old: List[str] = []
    for old_path in old_external:
        if os.path.normcase(str(old_path)) in new_keys:
            continue
        try:
            old_path.unlink()
            removed_old.append(str(old_path))
        except FileNotFoundError:
            pass

    return {
        "promoted_external_data": [str(target) for _, target in new_targets],
        "removed_previous_external_data": removed_old,
    }


def _artifact_integrity(expected_manifest: Mapping[str, Any]) -> Dict[str, Any]:
    """Re-hash an artifact using its originally resolved member list."""

    expected_files = [str(item["path"]) for item in expected_manifest["files"]]
    try:
        missing = [path for path in expected_files if not os.path.isfile(path)]
        if missing:
            raise FileNotFoundError(f"Artifact members disappeared: {missing}")
        current_hash = sha256_files(expected_files)
        current_size = int(sum(os.path.getsize(path) for path in expected_files))
        passed = current_hash == expected_manifest["sha256"] and current_size == int(
            expected_manifest["total_size"]
        )
        return {
            "status": "passed" if passed else "failed",
            "expected_sha256": expected_manifest["sha256"],
            "current_sha256": current_hash,
            "expected_total_size": int(expected_manifest["total_size"]),
            "current_total_size": current_size,
            "files": expected_files,
        }
    except Exception as exc:
        return {
            "status": "failed",
            "expected_sha256": expected_manifest.get("sha256"),
            "error": str(exc),
            "files": expected_files,
        }


def _mode_report_failed(mode_report: Mapping[str, Any]) -> bool:
    accuracy_gate = mode_report.get("accuracy_gate")
    if accuracy_gate is not None and accuracy_gate.get("status") != "passed":
        return True
    trtexec = mode_report.get("trtexec")
    return trtexec is not None and not bool(trtexec.get("passed"))


def _rewrite_fp8_direct_amax_scales(
    model_path: str,
    selected_names: Sequence[str],
    *,
    activation_qmax: float,
) -> Dict[str, Any]:
    """Correct ModelOpt 0.45's INT8-to-FP8 scale conversion.

    ModelOpt's ONNX PTQ path first calibrates symmetric INT8 (scale roughly
    ``amax / 127``), then multiplies that scale by ``448 / 127``.  The result
    uses only about 36 representable E4M3 levels.  Direct E4M3 calibration is
    ``amax / qmax``; ModelOpt's own Torch/ONNX exporters use ``qmax=448``.

    Since max calibration already encoded ``amax`` in every generated scale,
    multiplying the legacy scales by ``127**2 / (448*qmax)`` recovers the
    direct-amax result without rerunning calibration.  Weight scales always use
    the full exact range; activation qmax is exposed so held-out headroom can be
    measured rather than guessed.
    """

    if not (1.0 <= activation_qmax <= 448.0) or not np.isfinite(activation_qmax):
        raise ValueError("--fp8-activation-qmax must be finite and in [1, 448]")
    model = onnx.load(model_path, load_external_data=False)
    producers = {
        output: node for node in model.graph.node for output in node.output if output
    }
    selected = set(selected_names)
    roles: Dict[str, set[str]] = {}
    for node in model.graph.node:
        if node.name not in selected:
            continue
        if len(node.input) < 2:
            raise RuntimeError(f"Selected node lacks data/weight inputs: {node.name}")
        for input_index, role in ((0, "activation"), (1, "weight")):
            dq = producers.get(node.input[input_index])
            if dq is None or dq.op_type != "DequantizeLinear" or len(dq.input) < 2:
                raise RuntimeError(
                    f"Selected {role} input is not produced by DequantizeLinear: {node.name}"
                )
            roles.setdefault(dq.input[1], set()).add(role)

    initializer_by_name = {item.name: item for item in model.graph.initializer}
    legacy_effective_qmax = (127.0 * 127.0) / 448.0
    factors = {
        "activation": legacy_effective_qmax / float(activation_qmax),
        "weight": legacy_effective_qmax / 448.0,
    }
    counts = {"activation": 0, "weight": 0}
    scale_elements = {"activation": 0, "weight": 0}
    base_dir = str(Path(model_path).resolve().parent)
    for scale_name, scale_roles in sorted(roles.items()):
        if len(scale_roles) != 1:
            raise RuntimeError(
                f"FP8 scale {scale_name!r} is shared across incompatible roles: "
                f"{sorted(scale_roles)}"
            )
        role = next(iter(scale_roles))
        initializer = initializer_by_name.get(scale_name)
        if initializer is None:
            raise RuntimeError(f"FP8 scale is not a constant initializer: {scale_name}")
        values = onnx.numpy_helper.to_array(initializer, base_dir=base_dir)
        if values.size == 0 or not np.all(np.isfinite(values)) or np.any(values <= 0.0):
            raise RuntimeError(f"FP8 scale must be finite and positive: {scale_name}")
        corrected = np.asarray(values * factors[role], dtype=values.dtype)
        if not np.all(np.isfinite(corrected)) or np.any(corrected <= 0.0):
            raise RuntimeError(f"Corrected FP8 scale is invalid: {scale_name}")
        initializer.CopyFrom(onnx.numpy_helper.from_array(corrected, name=scale_name))
        counts[role] += 1
        scale_elements[role] += int(values.size)

    if counts["activation"] == 0 or counts["weight"] == 0:
        raise RuntimeError(
            f"Did not find both activation and weight FP8 scales: {counts}"
        )
    temporary = model_path + ".direct-amax.tmp.onnx"
    onnx.save_model(model, temporary)
    os.replace(temporary, model_path)
    return {
        "mode": "direct-amax",
        "formula": "legacy_scale * 127^2 / (448 * qmax)",
        "modelopt_legacy_effective_qmax": legacy_effective_qmax,
        "activation_qmax": float(activation_qmax),
        "weight_qmax": 448.0,
        "scale_initializer_count": counts,
        "scale_element_count": scale_elements,
        "multipliers": factors,
        "note": (
            "Accuracy-recovery experiment correcting ModelOpt 0.45 ONNX PTQ to direct "
            "E4M3 amax scaling; modelopt-legacy remains available for A/B comparison."
        ),
    }


def _require_callable_parameters(
    callable_object: Any, label: str, required: Sequence[str], runtime_version: str
) -> str:
    """Fail clearly when a newer ORT changes an internal API we deliberately pin."""

    try:
        signature = inspect.signature(callable_object)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(
            f"Cannot inspect {label} with ONNX Runtime {runtime_version}; "
            "incremental Q/DQ is disabled rather than guessing an internal API"
        ) from exc
    missing = sorted(set(required) - set(signature.parameters))
    if missing:
        raise RuntimeError(
            f"Unsupported ONNX Runtime {runtime_version} {label} signature {signature}; "
            f"missing required parameters {missing}. Test and update the pinned incremental path."
        )
    return str(signature)


def _incremental_calibration_providers(
    specification: str,
) -> tuple[List[Any], List[str]]:
    import onnxruntime as ort  # type: ignore[import-untyped]

    available = set(ort.get_available_providers())
    providers: List[Any] = []
    skipped: List[str] = []
    for raw in specification.split(","):
        token = raw.strip()
        if not token:
            continue
        lowered = token.lower()
        if lowered == "cpu":
            provider: Any = "CPUExecutionProvider"
        elif lowered.startswith("cuda"):
            device_id = int(token.split(":", 1)[1]) if ":" in token else 0
            provider = ("CUDAExecutionProvider", {"device_id": device_id})
        elif lowered.startswith("dml"):
            device_id = int(token.split(":", 1)[1]) if ":" in token else 0
            provider = ("DmlExecutionProvider", {"device_id": device_id})
        elif lowered == "trt":
            provider = "TensorrtExecutionProvider"
        else:
            raise ValueError(f"Unsupported incremental calibration EP {token!r}")
        provider_name = provider[0] if isinstance(provider, tuple) else provider
        if provider_name not in available:
            skipped.append(provider_name)
            continue
        if provider not in providers:
            providers.append(provider)
    if not providers:
        raise RuntimeError(
            "None of the requested incremental calibration execution providers are "
            f"available; requested={specification!r}, available={sorted(available)}"
        )
    return providers, sorted(set(skipped))


def _node_filtered_calibration_tensors(
    model: onnx.ModelProto, selected_names: Sequence[str]
) -> tuple[set[str], Dict[str, Any]]:
    """ModelOpt 0.45's node-name calibration selection without global monkeypatching."""

    selected = set(selected_names)
    value_infos = {value.name: value for value in model.graph.value_info}
    value_infos.update({value.name: value for value in model.graph.output})
    value_infos.update({value.name: value for value in model.graph.input})
    initializer_names = {item.name for item in model.graph.initializer}
    tensors: set[str] = set()
    found: set[str] = set()
    allowed_types = {onnx.TensorProto.FLOAT, onnx.TensorProto.FLOAT16}
    for node in model.graph.node:
        if node.name not in selected:
            continue
        found.add(node.name)
        for tensor_name in list(node.input) + list(node.output):
            value = value_infos.get(tensor_name)
            if value is None or tensor_name in initializer_names:
                continue
            tensor_type = value.type.tensor_type
            if tensor_type.elem_type in allowed_types:
                tensors.add(tensor_name)
    missing = sorted(selected - found)
    if missing:
        raise RuntimeError(
            f"Incremental calibration nodes disappeared after shape inference: {missing[:10]}"
        )
    if not tensors:
        raise RuntimeError(
            "Incremental calibration selected no floating activation tensors"
        )
    return tensors, value_infos


def _selected_nodes_with_direct_qdq(
    model_path: str, selected_names: Sequence[str]
) -> List[str]:
    model = onnx.load(model_path, load_external_data=False)
    producers = {
        output: node for node in model.graph.node for output in node.output if output
    }
    result: List[str] = []
    selected = set(selected_names)
    for node in model.graph.node:
        if node.name not in selected or len(node.input) < 2:
            continue
        complete = True
        for input_name in node.input[:2]:
            dq_node = producers.get(input_name)
            if (
                dq_node is None
                or dq_node.op_type != "DequantizeLinear"
                or not dq_node.input
            ):
                complete = False
                break
            q_node = producers.get(dq_node.input[0])
            if q_node is None or q_node.op_type != "QuantizeLinear":
                complete = False
                break
        if complete:
            result.append(node.name)
    return sorted(result)


def _run_incremental_ort_quantization(
    source_path: str,
    output_path: str,
    calibration_dataset: PositionDataset,
    selected_names: Sequence[str],
    selected_op_types: Sequence[str],
    args: argparse.Namespace,
    staging_dir: str,
) -> Dict[str, Any]:
    """Quantize only new nodes in an existing Q/DQ graph.

    ModelOpt 0.45's public INT8 entry point skips every graph containing Q/DQ.
    This follows its ORT configuration while using local calibrator subclasses,
    avoiding ModelOpt's process-global ORT monkeypatches.
    """

    import onnxruntime as ort  # type: ignore[import-untyped]
    from onnxruntime.quantization.calibrate import (  # type: ignore[import-untyped]
        EntropyCalibrater,
        HistogramCollector,
        MinMaxCalibrater,
        TensorsData,
    )
    from onnxruntime.quantization.qdq_quantizer import (  # type: ignore[import-untyped]
        QDQQuantizer,
    )
    from onnxruntime.quantization.quant_utils import (  # type: ignore[import-untyped]
        QuantType,
        add_infer_metadata,
    )

    runtime_version = str(ort.__version__)
    signatures = {
        "QDQQuantizer": _require_callable_parameters(
            QDQQuantizer,
            "QDQQuantizer",
            (
                "model",
                "per_channel",
                "reduce_range",
                "weight_qType",
                "activation_qType",
                "tensors_range",
                "nodes_to_quantize",
                "nodes_to_exclude",
                "op_types_to_quantize",
                "extra_options",
            ),
            runtime_version,
        ),
        "EntropyCalibrater": _require_callable_parameters(
            EntropyCalibrater,
            "EntropyCalibrater",
            (
                "model_path",
                "op_types_to_calibrate",
                "augmented_model_path",
                "use_external_data_format",
                "symmetric",
                "num_bins",
                "num_quantized_bins",
            ),
            runtime_version,
        ),
        "MinMaxCalibrater": _require_callable_parameters(
            MinMaxCalibrater,
            "MinMaxCalibrater",
            (
                "model_path",
                "op_types_to_calibrate",
                "augmented_model_path",
                "use_external_data_format",
                "symmetric",
            ),
            runtime_version,
        ),
    }
    if args.calibrate_per_node:
        raise RuntimeError(
            "--calibrate-per-node is not supported with --preserve-existing-qdq; "
            "the incremental calibrator is already restricted to the selected nodes"
        )
    providers, skipped_providers = _incremental_calibration_providers(
        args.calibration_eps
    )

    selected_set = set(selected_names)

    class _NodeFilterMixin:
        def select_tensors_to_calibrate(self, model):
            return _node_filtered_calibration_tensors(model, selected_names)

    class _StreamingEntropy(_NodeFilterMixin, EntropyCalibrater):
        def collect_data(self, data_reader):
            collected = False
            output_names = [item.name for item in self.infer_session.get_outputs()]
            input_names = {item.name for item in self.infer_session.get_inputs()}
            while True:
                inputs = data_reader.get_next()
                if not inputs:
                    break
                outputs = self.infer_session.run(None, inputs)
                values = {
                    name: [np.copy(value) if name in input_names else value]
                    for name, value in zip(output_names, outputs)
                    if name in self.tensors_to_calibrate
                }
                if not self.collector:
                    self.collector = HistogramCollector(
                        method=self.method,
                        symmetric=self.symmetric,
                        num_bins=self.num_bins,
                        num_quantized_bins=self.num_quantized_bins,
                        percentile=self.percentile,
                        scenario=self.scenario,
                    )
                self.collector.collect(values)
                collected = True
            if not collected:
                raise ValueError(
                    "No incremental entropy calibration data was collected"
                )

    class _StreamingMinMax(_NodeFilterMixin, MinMaxCalibrater):
        def collect_data(self, data_reader):
            collected = False
            while True:
                inputs = data_reader.get_next()
                if not inputs:
                    break
                self.intermediate_outputs.append(self.infer_session.run(None, inputs))
                result = self.compute_data()
                if not isinstance(result, TensorsData):
                    raise TypeError(
                        f"Expected TensorsData from incremental min/max calibration, got {type(result)}"
                    )
                self.clear_collected_data()
                collected = True
            if not collected:
                raise ValueError(
                    "No incremental min/max calibration data was collected"
                )

    calibration_dir = os.path.join(staging_dir, "incremental-calibration")
    os.makedirs(calibration_dir, exist_ok=False)
    augmented_path = os.path.join(calibration_dir, "augmented.onnx")
    calibrator_cls = (
        _StreamingEntropy if args.calibration_method == "entropy" else _StreamingMinMax
    )
    calibrator_kwargs: Dict[str, Any] = dict(
        model_path=source_path,
        # Selection is by node name in the local mixin, matching ModelOpt 0.45.
        op_types_to_calibrate=list(selected_names),
        augmented_model_path=augmented_path,
        use_external_data_format=True,
        symmetric=False,
    )
    if args.calibration_method == "entropy":
        calibrator_kwargs.update(num_bins=128, num_quantized_bins=128)
    calibrator = calibrator_cls(**calibrator_kwargs)
    calibrator.augment_graph()
    calibrator.set_execution_providers(providers)
    reader = ArrayCalibrationDataReader(calibration_dataset)
    reader.rewind()
    calibrator.collect_data(reader)
    tensors_range = calibrator.compute_data()
    if not isinstance(tensors_range, TensorsData):
        raise TypeError(
            f"Expected TensorsData from incremental calibration, got {type(tensors_range)}"
        )
    calibration_tensor_names = sorted(tensors_range.data)
    expected_tensors, _ = _node_filtered_calibration_tensors(
        calibrator.model, selected_names
    )
    unexpected_calibrated = sorted(set(calibration_tensor_names) - expected_tensors)
    if unexpected_calibrated:
        raise RuntimeError(
            "Incremental calibrator collected tensors outside the selected nodes: "
            f"{unexpected_calibrated[:10]}"
        )

    # The calibrator's ModelProto may retain external-data locations relative
    # to its temporary augmented path. Keep only its proven shape information,
    # then reload and fully materialize the immutable parent from its real path.
    # A b15 parent is ~813 MiB, comfortably below the 64 GiB research host, and
    # the resulting output is self-contained instead of sharing a fragile link.
    inferred_value_info = [
        value.SerializeToString() for value in calibrator.model.graph.value_info
    ]
    calibrator.infer_session = None
    del calibrator
    gc.collect()
    shutil.rmtree(calibration_dir)

    quantization_model = onnx.load(source_path, load_external_data=True)
    existing_value_info = {value.name for value in quantization_model.graph.value_info}
    for payload in inferred_value_info:
        value = onnx.ValueInfoProto()
        value.ParseFromString(payload)
        if value.name not in existing_value_info:
            quantization_model.graph.value_info.append(value)
            existing_value_info.add(value.name)
    add_infer_metadata(quantization_model)
    onnx.external_data_helper.convert_model_from_external_data(quantization_model)
    remaining_external = [
        initializer.name
        for initializer in quantization_model.graph.initializer
        if initializer.data_location == onnx.TensorProto.EXTERNAL
        or initializer.external_data
    ]
    if remaining_external:
        raise RuntimeError(
            "Failed to materialize parent external data before incremental Q/DQ: "
            f"{remaining_external[:10]}"
        )
    materialized_raw_bytes = int(
        sum(
            len(initializer.raw_data)
            for initializer in quantization_model.graph.initializer
        )
    )

    graph_op_types = sorted({node.op_type for node in quantization_model.graph.node})
    extra_options = {
        "QuantizeBias": False,
        "ActivationSymmetric": True,
        "WeightSymmetric": True,
        "OpTypesToExcludeOutputQuantization": graph_op_types,
        "AddQDQPairToWeight": True,
        "QDQOpTypePerChannelSupportToAxis": {"Conv": 0, "ConvTranspose": 1},
        "DedicatedQDQPair": False,
        "ForceQuantizeNoInputCheck": True,
        "QDQDisableWeightAdjustForInt32Bias": True,
    }
    quantizer = QDQQuantizer(
        model=quantization_model,
        per_channel=True,
        reduce_range=False,
        weight_qType=QuantType.QInt8,
        activation_qType=QuantType.QInt8,
        tensors_range=tensors_range,
        nodes_to_quantize=list(selected_names),
        nodes_to_exclude=[],
        op_types_to_quantize=list(selected_op_types),
        extra_options=extra_options,
    )
    quantizer.quantize_model()
    save_method = quantizer.model.save_model_to_file
    signatures["save_model_to_file"] = _require_callable_parameters(
        save_method,
        "ONNXModel.save_model_to_file",
        ("output_path", "use_external_data_format"),
        runtime_version,
    )
    save_method(output_path=output_path, use_external_data_format=True)
    del quantizer
    del quantization_model
    gc.collect()

    newly_quantized = _selected_nodes_with_direct_qdq(output_path, selected_names)
    if newly_quantized != sorted(selected_set):
        raise RuntimeError(
            "Incremental ORT quantization produced no usable change; refusing to promote "
            f"a parent clone. expected={sorted(selected_set)}, actual={newly_quantized}"
        )
    return {
        "backend": "onnxruntime-node-filtered-qdq",
        "onnxruntime_version": runtime_version,
        "api_signatures": signatures,
        "providers": providers,
        "skipped_unavailable_providers": skipped_providers,
        "calibration_tensor_count": len(calibration_tensor_names),
        "calibration_tensor_names": calibration_tensor_names,
        "newly_quantized_nodes": newly_quantized,
        "materialized_parent_raw_bytes": materialized_raw_bytes,
        "self_contained_output": True,
        "configuration": {
            "calibration_method": args.calibration_method,
            "activation_type": "QInt8",
            "weight_type": "QInt8",
            "activation_symmetric": True,
            "weight_symmetric": True,
            "per_channel_weights": True,
            "reduce_range": False,
            "quantize_bias": False,
            "output_quantization": False,
        },
    }


def _quantize_one(
    mode: str,
    source_path: str,
    output_path: str,
    calibration_dataset: PositionDataset,
    selected_names: Sequence[str],
    selected_op_types: Sequence[str],
    args: argparse.Namespace,
    quantize,
    preservation_snapshot: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    if any(
        item.strip().lower() != "cpu"
        for item in args.calibration_eps.split(",")
        if item.strip()
    ):
        _preload_ort_gpu_dependencies()
    if os.path.exists(output_path) and not args.overwrite:
        raise FileExistsError(f"Refusing to replace {output_path}; use --overwrite")
    staging_dir = tempfile.mkdtemp(
        prefix=f".katago-quant-{args.output_prefix}.{mode}-",
        dir=args.output_dir,
    )
    details: Dict[str, Any] = {}
    try:
        # ModelOpt shape-infers its input in place, so normal PTQ requires a
        # complete clone. The incremental ORT path is read-only and uses the
        # original parent directly, avoiding another ~800 MiB full315 copy.
        staged_source = (
            source_path
            if preservation_snapshot is not None
            else _copy_onnx_artifact(source_path, staging_dir)
        )
        token = uuid.uuid4().hex
        staged_output = os.path.join(
            staging_dir,
            f"{Path(output_path).stem}.{token}.onnx",
        )
        reader = ArrayCalibrationDataReader(calibration_dataset)
        exact_node_patterns = [f"^{re.escape(name)}$" for name in selected_names]
        calibration_method = (
            args.fp8_calibration_method if mode == "fp8" else args.calibration_method
        )
        kwargs = dict(
            # ModelOpt 0.45 shape-infers this path in place, so it must always
            # receive the complete staged copy rather than the user's dump.
            onnx_path=staged_source,
            quantize_mode=mode,
            calibration_data_reader=reader,
            calibration_method=calibration_method,
            calibration_eps=[
                item.strip() for item in args.calibration_eps.split(",") if item.strip()
            ],
            op_types_to_quantize=list(selected_op_types),
            nodes_to_quantize=exact_node_patterns,
            use_external_data_format=True,
            keep_intermediate_files=args.keep_intermediate_files,
            output_path=staged_output,
            log_level="INFO",
            log_file=os.path.join(
                args.output_dir, f"{args.output_prefix}.{mode}.modelopt.log"
            ),
            high_precision_dtype=args.high_precision,
            mha_accumulation_dtype="fp32",
            disable_mha_qdq=True,
            use_zero_point=False,
            passes=[],
            simplify=False,
            calibrate_per_node=args.calibrate_per_node,
            direct_io_types=False,
            # The selected KataGo projections operate on H*W tokens and are not
            # GEMV even at batch 1. Keep ModelOpt's generic shape heuristic from
            # silently overriding the explicit, audited node allowlist.
            enable_gemv_detection_for_trt=False,
        )
        if mode == "fp8":
            kwargs["opset"] = 21
        logging.info(
            "Quantizing %d %s nodes to %s with %s calibration and %s fallback in %s",
            len(selected_names),
            "/".join(selected_op_types),
            mode.upper(),
            calibration_method,
            args.high_precision.upper(),
            staging_dir,
        )
        if preservation_snapshot is not None:
            details["incremental_quantizer"] = _run_incremental_ort_quantization(
                staged_source,
                staged_output,
                calibration_dataset,
                selected_names,
                selected_op_types,
                args,
                staging_dir,
            )
        else:
            quantize(**kwargs)
        if not os.path.isfile(staged_output):
            raise RuntimeError(f"ModelOpt returned without creating {staged_output}")
        if mode == "fp8" and args.fp8_scale_mode == "direct-amax":
            details["fp8_scale_rewrite"] = _rewrite_fp8_direct_amax_scales(
                staged_output,
                selected_names,
                activation_qmax=args.fp8_activation_qmax,
            )
        if preservation_snapshot is not None:
            union_names = sorted(
                set(preservation_snapshot["selected_names"]) | set(selected_names)
            )
            staged_union_audit = audit_qdq_model(staged_output, union_names)
            if staged_union_audit["errors"]:
                raise RuntimeError(
                    "Incremental Q/DQ union audit failed before artifact promotion: "
                    f"{staged_union_audit['errors'][:10]}"
                )
            preservation = compare_existing_qdq_state(
                staged_output, preservation_snapshot
            )
            if preservation["status"] != "passed":
                raise RuntimeError(
                    "ModelOpt changed pre-existing Q/DQ state; refusing to promote artifact: "
                    f"{preservation['differences'][:10]}"
                )
            details["incremental_union_selected_count"] = len(union_names)
            details["incremental_union_selected_names"] = union_names
            details["incremental_staged_qdq_audit"] = staged_union_audit
            details["existing_qdq_preservation"] = preservation
        details.update(
            _promote_staged_artifact(
                staged_output,
                output_path,
                overwrite=args.overwrite,
            )
        )
        if args.keep_intermediate_files:
            details["modelopt_intermediate_directory"] = staging_dir
        return details
    finally:
        if not args.keep_intermediate_files:
            staging_path = Path(staging_dir).resolve()
            output_root = Path(args.output_dir).resolve()
            if staging_path.parent != output_root or not staging_path.name.startswith(
                ".katago-quant-"
            ):
                raise RuntimeError(
                    f"Refusing to clean unsafe staging path: {staging_path}"
                )
            try:
                shutil.rmtree(staging_path)
            except FileNotFoundError:
                pass


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    args.onnx_input = str(Path(args.onnx_input).resolve())
    args.output_dir = str(Path(args.output_dir).resolve())
    if args.max_source_files < 0:
        parser.error("--max-source-files must be nonnegative (0 means unlimited)")
    if args.expected_existing_qdq_nodes is not None and not args.preserve_existing_qdq:
        parser.error("--expected-existing-qdq-nodes requires --preserve-existing-qdq")
    if args.preserve_existing_qdq:
        if set(args.formats) != {"int8"}:
            parser.error(
                "--preserve-existing-qdq currently supports exactly --formats int8"
            )
        if not args.only_node_regex:
            parser.error(
                "--preserve-existing-qdq requires --only-node-regex so ModelOpt receives "
                "an explicit incremental node allowlist"
            )
        if args.expected_existing_qdq_nodes is None:
            parser.error(
                "--preserve-existing-qdq requires --expected-existing-qdq-nodes"
            )
        if args.expected_existing_qdq_nodes <= 0:
            parser.error("--expected-existing-qdq-nodes must be positive")
    if (
        "fp8" in args.formats
        and args.fp8_calibration_method != "max"
        and not args.allow_fp8_nonmax_calibration
    ):
        parser.error(
            "ModelOpt 0.45 FP8 conversion is documented for max-calibrated INT8 scales. "
            "Use --fp8-calibration-method max, or explicitly acknowledge an unsupported "
            "experiment with --allow-fp8-nonmax-calibration."
        )
    if not np.isfinite(args.fp8_activation_qmax) or not (
        1.0 <= args.fp8_activation_qmax <= 448.0
    ):
        parser.error("--fp8-activation-qmax must be finite and in [1, 448]")
    if (
        "fp8" in args.formats
        and args.fp8_scale_mode == "direct-amax"
        and args.fp8_calibration_method != "max"
    ):
        parser.error("--fp8-scale-mode direct-amax requires max calibration")
    if args.high_precision == "fp16" and not args.allow_global_fp16_fallback:
        parser.error(
            "--high-precision fp16 globally converts fallback ops, including sensitive heads/norms. "
            "Use FP32 for quantization-only error, or explicitly add "
            "--allow-global-fp16-fallback for a separately gated experiment."
        )
    if not os.path.isfile(args.onnx_input):
        parser.error(f"ONNX input does not exist: {args.onnx_input}")
    os.makedirs(args.output_dir, exist_ok=True)
    if args.output_prefix is None:
        args.output_prefix = Path(args.onnx_input).stem
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._+-]*", args.output_prefix):
        parser.error(
            "--output-prefix must be a filename-safe model name containing only "
            "letters, digits, dot, underscore, plus, or minus"
        )
    planned_output_paths = {
        mode: str(
            (Path(args.output_dir) / f"{args.output_prefix}.{mode}.qdq.onnx").resolve()
        )
        for mode in dict.fromkeys(args.formats)
    }
    source_key = os.path.normcase(args.onnx_input)
    colliding_formats = [
        mode
        for mode, path in planned_output_paths.items()
        if os.path.normcase(path) == source_key
    ]
    if colliding_formats:
        parser.error(
            "Quantized output would overwrite the source ONNX artifact for format(s) "
            f"{colliding_formats}. Choose a different --output-dir or --output-prefix."
        )
    report_path = os.path.join(
        args.output_dir, f"{args.output_prefix}.quantization-report.json"
    )
    if os.path.exists(report_path) and not args.overwrite:
        raise FileExistsError(f"Refusing to replace {report_path}; use --overwrite")
    log_path = _setup_logging(args.output_dir, args.output_prefix)

    gpu_summary = _gpu_summary()
    report: Dict[str, Any] = {
        "schema_version": 1,
        "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "status": "running",
        "research_only": True,
        "runtime_integration": (
            "Current KataGo loads FP32 .bin.gz and builds its own ONNX. These Q/DQ ONNX artifacts "
            "must be benchmarked with a standalone strongly typed TensorRT build until a separate "
            "runtime integration is implemented."
        ),
        "command": [sys.executable, str(Path(__file__).resolve())]
        + list(argv or sys.argv[1:]),
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "onnx": onnx.__version__,
            "packages": _optional_package_versions(),
            "git_revision": _git_revision(),
            "gpus": gpu_summary,
            "fp8_hardware": _fp8_hardware_summary(gpu_summary),
        },
        "options": {
            key: value
            for key, value in vars(args).items()
            if key not in ("calibration_data", "validation_data")
        },
        "log_path": log_path,
        "formats": {},
    }

    source_manifest: Optional[Dict[str, Any]] = None
    existing_qdq_state: Optional[Dict[str, Any]] = None
    try:
        if not args.skip_onnx_check:
            logging.info("Running ONNX checker on source graph")
            onnx.checker.check_model(args.onnx_input, full_check=False)
        logging.info("Loading source ONNX: %s", args.onnx_input)
        source_model = onnx.load(args.onnx_input, load_external_data=False)
        input_specs = validate_katago_io_contract(
            source_model,
            require_producer_metadata=not args.preserve_existing_qdq,
        )
        source_manifest = artifact_manifest(args.onnx_input, source_model)
        report["source"] = source_manifest

        if args.preserve_existing_qdq:
            logging.info("Strictly auditing pre-existing weighted Q/DQ state")
            existing_qdq_state = capture_existing_qdq_state(args.onnx_input)
            if existing_qdq_state["selected_count"] != args.expected_existing_qdq_nodes:
                raise RuntimeError(
                    "Found "
                    f"{existing_qdq_state['selected_count']} pre-existing fully quantized "
                    f"weighted nodes, expected {args.expected_existing_qdq_nodes}. "
                    "Refusing incremental quantization of a changed parent graph."
                )
            existing_audit = audit_qdq_model(
                args.onnx_input, existing_qdq_state["selected_names"]
            )
            if existing_audit["errors"]:
                raise RuntimeError(
                    "Pre-existing Q/DQ semantic audit failed: "
                    f"{existing_audit['errors'][:10]}"
                )
            non_int8_types = {
                qtype: count
                for qtype, count in existing_audit["quantized_tensor_types"].items()
                if qtype != "INT8"
            }
            if non_int8_types or not existing_audit["quantized_tensor_types"].get(
                "INT8"
            ):
                raise RuntimeError(
                    "--preserve-existing-qdq requires a purely INT8 parent graph; "
                    f"quantized tensor types={existing_audit['quantized_tensor_types']}"
                )
            report["existing_qdq"] = {
                "state": existing_qdq_state,
                "audit": existing_audit,
            }

        selection = select_quantizable_nodes(
            source_model,
            scope=args.scope,
            include_regexes=args.include_node_regex,
            exclude_regexes=args.exclude_node_regex,
            only_regexes=args.only_node_regex,
        )
        report["node_selection"] = selection.manifest()
        logging.info(
            "Selected %d/%d weighted nodes: %s",
            len(selection.selected_names),
            len(selection.weighted_candidate_names),
            selection.selected_by_op_type,
        )
        if (
            args.expected_quantized_nodes is not None
            and len(selection.selected_names) != args.expected_quantized_nodes
        ):
            raise RuntimeError(
                f"Selected {len(selection.selected_names)} nodes, expected "
                f"{args.expected_quantized_nodes}. Refusing to quantize a changed graph."
            )
        if existing_qdq_state is not None:
            overlap = sorted(
                set(existing_qdq_state["selected_names"])
                & set(selection.selected_names)
            )
            if overlap:
                raise RuntimeError(
                    "Incremental selection overlaps pre-existing Q/DQ nodes: "
                    f"{overlap[:10]}"
                )
            union_names = sorted(
                set(existing_qdq_state["selected_names"])
                | set(selection.selected_names)
            )
            report["union_node_selection"] = {
                "existing_count": len(existing_qdq_state["selected_names"]),
                "incremental_count": len(selection.selected_names),
                "selected_count": len(union_names),
                "selected_names": union_names,
            }
        # The b15 graph contains roughly 680 MB of FP32 initializers. ModelOpt
        # loads its own graph copies, so release this inspection copy before
        # calibration/quantization to keep peak host memory under control.
        del source_model
        gc.collect()

        calibration_files = resolve_npz_files(args.calibration_data)
        validation_files = (
            resolve_npz_files(args.validation_data) if args.validation_data else []
        )
        overlap = sorted(
            set(map(os.path.normcase, calibration_files))
            & set(map(os.path.normcase, validation_files))
        )
        if overlap and not args.allow_data_overlap:
            raise RuntimeError(
                "Calibration and validation share NPZ shards. Use disjoint held-out files; "
                f"overlap={overlap[:10]}"
            )
        if not args.skip_validation and not validation_files:
            raise RuntimeError(
                "--validation-data is required unless --skip-validation is explicitly set"
            )

        logging.info("Sampling %d calibration positions", args.calibration_samples)
        calibration_dataset = load_position_dataset(
            calibration_files,
            input_specs,
            sample_count=args.calibration_samples,
            batch_size=args.batch_size,
            seed=args.seed,
            history_mode=args.history_mode,
            symmetry_mode=args.symmetry_mode,
            max_source_files=args.max_source_files,
        )
        report["calibration_dataset"] = calibration_dataset.manifest()

        validation_dataset: Optional[PositionDataset] = None
        reference_batches: Optional[List[Dict[str, np.ndarray]]] = None
        if not args.skip_validation:
            logging.info(
                "Sampling %d held-out validation positions", args.validation_samples
            )
            validation_dataset = load_position_dataset(
                validation_files,
                input_specs,
                sample_count=args.validation_samples,
                batch_size=args.batch_size,
                seed=args.seed + 1,
                history_mode=args.history_mode,
                symmetry_mode=args.symmetry_mode,
                max_source_files=args.max_source_files,
            )
            report["validation_dataset"] = validation_dataset.manifest()
            logging.info("Computing FP32 reference outputs once")
            reference_batches, reference_execution = _run_ort_model(
                args.onnx_input,
                validation_dataset,
                args.validation_ep,
                args.ort_intra_op_threads,
                args.allow_validation_ep_fallback,
            )
            report["reference_execution"] = reference_execution

        quantize, modelopt_version = _load_modelopt(args.allow_unpinned_modelopt)
        report["environment"]["nvidia_modelopt"] = modelopt_version
        thresholds = _accuracy_thresholds(args)
        if not any(value is not None for value in thresholds.values()):
            logging.warning(
                "No numerical accuracy thresholds were supplied. Metrics will be reported, "
                "but the artifact must not be called release-qualified."
            )

        any_failure = False
        for mode in dict.fromkeys(args.formats):
            output_path = planned_output_paths[mode]
            mode_report: Dict[str, Any] = {"status": "running", "path": output_path}
            mode_report["calibration_method"] = (
                args.fp8_calibration_method
                if mode == "fp8"
                else args.calibration_method
            )
            report["formats"][mode] = mode_report
            json_dump(report_path, report)
            try:
                quant_details = _quantize_one(
                    mode,
                    args.onnx_input,
                    output_path,
                    calibration_dataset,
                    selection.selected_names,
                    sorted(selection.selected_by_op_type),
                    args,
                    quantize,
                    existing_qdq_state,
                )
                mode_report.update(quant_details)
                if not args.skip_onnx_check:
                    logging.info("Running ONNX checker on %s", output_path)
                    onnx.checker.check_model(output_path, full_check=False)
                mode_report["artifact"] = artifact_manifest(output_path)
                audited_names = (
                    report["union_node_selection"]["selected_names"]
                    if existing_qdq_state is not None
                    else selection.selected_names
                )
                mode_report["qdq_audit"] = audit_qdq_model(output_path, audited_names)
                if mode_report["qdq_audit"]["non_float_io"]:
                    raise RuntimeError(
                        f"Quantization changed graph I/O types: "
                        f"{mode_report['qdq_audit']['non_float_io']}"
                    )
                if mode_report["qdq_audit"]["quantize_linear_count"] == 0:
                    raise RuntimeError(
                        "Quantized graph contains no QuantizeLinear nodes"
                    )
                audit = mode_report["qdq_audit"]
                if audit["errors"]:
                    raise RuntimeError(
                        f"Q/DQ semantic audit failed: {audit['errors'][:10]}"
                    )
                if audit["selected_nodes_missing_after_quantization"]:
                    raise RuntimeError(
                        "ModelOpt removed or renamed explicitly selected nodes: "
                        f"{audit['selected_nodes_missing_after_quantization'][:10]}"
                    )
                if audit["selected_nodes_without_qdq_inputs"]:
                    raise RuntimeError(
                        "Some selected nodes do not have Q/DQ on both data and weight inputs: "
                        f"{audit['selected_nodes_without_qdq_inputs'][:10]}"
                    )
                if audit["scale_granularity_errors"]:
                    raise RuntimeError(
                        "Unexpected Q/DQ scale granularity: "
                        f"{audit['scale_granularity_errors'][:10]}"
                    )
                expected_qtype = "INT8" if mode == "int8" else "FLOAT8E4M3FN"
                unexpected_qtypes = {
                    qtype: count
                    for qtype, count in audit["quantized_tensor_types"].items()
                    if qtype != expected_qtype
                }
                if unexpected_qtypes:
                    raise RuntimeError(
                        f"{mode.upper()} graph contains unexpected quantized tensor types: "
                        f"{unexpected_qtypes}"
                    )

                if validation_dataset is not None and reference_batches is not None:
                    candidate_batches, candidate_execution = _run_ort_model(
                        output_path,
                        validation_dataset,
                        args.validation_ep,
                        args.ort_intra_op_threads,
                        args.allow_validation_ep_fallback,
                    )
                    mode_report["validation_execution"] = candidate_execution
                    metrics = _validation_metrics(
                        reference_batches, candidate_batches, validation_dataset
                    )
                    mode_report["validation"] = metrics
                    mode_report["accuracy_gate"] = evaluate_accuracy_gates(
                        metrics, thresholds
                    )

                if args.trtexec:
                    mode_report["trtexec"] = _run_trtexec(
                        args.trtexec,
                        output_path,
                        input_specs,
                        args.trt_opt_batch,
                        args.trt_max_batch,
                        args.trt_workspace_mib,
                        args.trt_timeout_seconds,
                        args.overwrite,
                    )
                mode_failed = _mode_report_failed(mode_report)
                any_failure = any_failure or mode_failed
                mode_report["status"] = "failed" if mode_failed else "complete"
            except Exception as exc:
                any_failure = True
                mode_report["status"] = "error"
                mode_report["error"] = str(exc)
                mode_report["traceback"] = traceback.format_exc()
                logging.exception("%s quantization failed", mode.upper())
                json_dump(report_path, report)
                if not args.continue_on_error:
                    raise
            finally:
                json_dump(report_path, report)

        assert source_manifest is not None
        report["source_integrity"] = _artifact_integrity(source_manifest)
        if report["source_integrity"]["status"] != "passed":
            raise RuntimeError(
                "Source ONNX artifact changed during quantization; see source_integrity"
            )
        report["status"] = "failed" if any_failure else "complete"
        json_dump(report_path, report)
        logging.info("Wrote reproducibility/accuracy report: %s", report_path)
        return 2 if any_failure else 0
    except Exception as exc:
        if source_manifest is not None and "source_integrity" not in report:
            report["source_integrity"] = _artifact_integrity(source_manifest)
        report["status"] = "error"
        report["error"] = str(exc)
        if (
            source_manifest is not None
            and report["source_integrity"]["status"] != "passed"
            and "Source ONNX artifact changed" not in report["error"]
        ):
            report["error"] += (
                "; source ONNX artifact also changed during the failed run; "
                "see source_integrity"
            )
        report["traceback"] = traceback.format_exc()
        json_dump(report_path, report)
        logging.exception("Quantization pipeline failed")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
