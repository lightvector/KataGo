"""Utilities for accuracy-first quantization of KataGo ONNX graphs.

This module intentionally operates on the ONNX graph emitted by KataGo's
TensorRT backend.  That graph is the inference graph KataGo actually builds,
including its five raw output tensors.  It avoids maintaining a second,
slightly different PyTorch-to-ONNX exporter.

The public helpers are kept independent of NVIDIA Model Optimizer so that data
loading, node selection, graph auditing, and accuracy metrics can be tested
without installing the optional quantization toolchain.
"""

from __future__ import annotations

import glob
import hashlib
import json
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import onnx
from onnx import TensorProto


EXPECTED_INPUT_NAMES = ("InputMask", "InputSpatial", "InputGlobal")
OPTIONAL_INPUT_NAMES = ("InputMeta",)
EXPECTED_OUTPUT_NAMES = (
    "OutputPolicyPass",
    "OutputPolicy",
    "OutputValue",
    "OutputScoreValue",
    "OutputOwnership",
)

_OUTPUT_CHANNEL_LABELS = {
    "OutputPolicyPass": ("policy", "shortterm_optimistic", "q_value", "q_score"),
    "OutputPolicy": ("policy", "shortterm_optimistic", "q_value", "q_score"),
    "OutputValue": ("win", "loss", "no_result"),
    "OutputScoreValue": (
        "score_mean",
        "score_mean_sq",
        "lead",
        "variance_time",
        "shortterm_value_error",
        "shortterm_score_error",
    ),
    "OutputOwnership": ("ownership",),
}

_TRANSFORMER_PROJECTION_RE = re.compile(
    r"(?:^|[./])(?:q_proj|k_proj|v_proj|out_proj|"
    r"ffn_linear1|ffn_linear_gate|ffn_linear2)(?=$|[./])"
)
_SUPPORTED_WEIGHTED_OPS = frozenset(("Conv", "MatMul", "Gemm"))
_CONSTANT_PASSTHROUGH_OPS = frozenset(("Cast", "Identity", "Reshape", "Transpose"))


@dataclass(frozen=True)
class InputSpec:
    name: str
    shape: Tuple[Optional[int], ...]
    elem_type: int


@dataclass
class PositionDataset:
    batches: List[Dict[str, np.ndarray]]
    sample_count: int
    base_sample_count: int
    batch_size: int
    seed: int
    history_mode: str
    symmetry_mode: str
    symmetry_counts: Dict[str, int]
    symmetry_sha256: str
    position_sha256: str
    selection_sha256: str
    available_source_file_count: int
    selected_source_file_count: int
    max_source_files: int
    source_files: List[Dict[str, Any]]
    input_shapes: Dict[str, List[int]]

    def manifest(self) -> Dict[str, Any]:
        return {
            "sample_count": self.sample_count,
            "base_sample_count": self.base_sample_count,
            "batch_count": len(self.batches),
            "batch_size": self.batch_size,
            "seed": self.seed,
            "history_mode": self.history_mode,
            "symmetry_mode": self.symmetry_mode,
            "symmetry_counts": self.symmetry_counts,
            "symmetry_sha256": self.symmetry_sha256,
            "position_sha256": self.position_sha256,
            "selection_sha256": self.selection_sha256,
            "available_source_file_count": self.available_source_file_count,
            "selected_source_file_count": self.selected_source_file_count,
            "max_source_files": self.max_source_files,
            "source_files": self.source_files,
            "input_shapes": self.input_shapes,
        }


@dataclass
class NodeSelection:
    scope: str
    selected_names: List[str]
    selected_by_op_type: Dict[str, int]
    weighted_candidate_names: List[str]
    weighted_candidates_by_op_type: Dict[str, int]
    rejected: List[Dict[str, str]]

    def manifest(self) -> Dict[str, Any]:
        return {
            "scope": self.scope,
            "selected_count": len(self.selected_names),
            "selected_by_op_type": self.selected_by_op_type,
            "selected_names": self.selected_names,
            "weighted_candidate_count": len(self.weighted_candidate_names),
            "weighted_candidates_by_op_type": self.weighted_candidates_by_op_type,
            "rejected": self.rejected,
        }


class ArrayCalibrationDataReader:
    """Small ORT/ModelOpt-compatible reader backed by immutable numpy batches."""

    def __init__(self, dataset: PositionDataset):
        self.dataset = dataset
        self._start = 0
        self._end = len(dataset.batches)
        self._index = self._start

    def get_next(self) -> Optional[Dict[str, np.ndarray]]:
        if self._index >= self._end:
            return None
        batch = self.dataset.batches[self._index]
        self._index += 1
        return batch

    def rewind(self) -> None:
        self._index = self._start

    def get_first(self) -> Dict[str, np.ndarray]:
        if self._start >= self._end:
            raise RuntimeError("Calibration dataset is empty")
        return self.dataset.batches[self._start]

    def set_range(self, start_index: int, end_index: int) -> None:
        if not (0 <= start_index <= end_index <= len(self.dataset.batches)):
            raise ValueError(
                f"Invalid calibration batch range [{start_index}, {end_index})"
            )
        self._start = start_index
        self._end = end_index
        self.rewind()

    def __len__(self) -> int:
        return self._end - self._start

    def __iter__(self):
        return self

    def __next__(self) -> Dict[str, np.ndarray]:
        value = self.get_next()
        if value is None:
            raise StopIteration
        return value


def _dim_value(dim: onnx.TensorShapeProto.Dimension) -> Optional[int]:
    if dim.HasField("dim_value") and dim.dim_value > 0:
        return int(dim.dim_value)
    return None


def read_input_specs(model: onnx.ModelProto) -> List[InputSpec]:
    initializer_names = {initializer.name for initializer in model.graph.initializer}
    specs: List[InputSpec] = []
    for value_info in model.graph.input:
        if value_info.name in initializer_names:
            continue
        tensor_type = value_info.type.tensor_type
        specs.append(
            InputSpec(
                name=value_info.name,
                shape=tuple(_dim_value(dim) for dim in tensor_type.shape.dim),
                elem_type=int(tensor_type.elem_type),
            )
        )
    return specs


def validate_katago_io_contract(
    model: onnx.ModelProto, *, require_producer_metadata: bool = True
) -> List[InputSpec]:
    specs = read_input_specs(model)
    names = [spec.name for spec in specs]
    missing_inputs = [name for name in EXPECTED_INPUT_NAMES if name not in names]
    unknown_inputs = [
        name
        for name in names
        if name not in EXPECTED_INPUT_NAMES + OPTIONAL_INPUT_NAMES
    ]
    if missing_inputs or unknown_inputs:
        raise ValueError(
            "Expected KataGo TensorRT-dump inputs "
            f"{EXPECTED_INPUT_NAMES} plus optional {OPTIONAL_INPUT_NAMES}; "
            f"got {names}. Missing={missing_inputs}, unknown={unknown_inputs}"
        )
    for spec in specs:
        if spec.elem_type != TensorProto.FLOAT:
            raise ValueError(
                f"Input {spec.name} must remain FLOAT, got ONNX elem_type={spec.elem_type}"
            )
        if len(spec.shape) != 4:
            raise ValueError(f"Input {spec.name} must be rank 4, got {spec.shape}")

    spec_by_name = {spec.name: spec for spec in specs}
    expected_channels = {"InputMask": 1, "InputSpatial": 22, "InputGlobal": 19}
    for name, channels in expected_channels.items():
        if spec_by_name[name].shape[1] != channels:
            raise ValueError(
                f"{name} must have {channels} channels for the current KataGo input format, "
                f"got {spec_by_name[name].shape}"
            )
    spatial_shape = spec_by_name["InputSpatial"].shape
    if spatial_shape[2] is None or spatial_shape[3] is None:
        raise ValueError("KataGo ONNX must have fixed board height and width")
    if spec_by_name["InputMask"].shape[2:] != spatial_shape[2:]:
        raise ValueError("InputMask and InputSpatial board dimensions differ")
    for name in ("InputGlobal", "InputMeta"):
        if name in spec_by_name and spec_by_name[name].shape[2:] != (1, 1):
            raise ValueError(
                f"{name} must use NC11 layout, got {spec_by_name[name].shape}"
            )

    output_names = [value_info.name for value_info in model.graph.output]
    if output_names != list(EXPECTED_OUTPUT_NAMES):
        raise ValueError(
            "Expected the five raw outputs from KataGo's ONNX emitter in order "
            f"{EXPECTED_OUTPUT_NAMES}, got {output_names}"
        )

    output_shapes = {
        value_info.name: tuple(
            _dim_value(dim) for dim in value_info.type.tensor_type.shape.dim
        )
        for value_info in model.graph.output
    }
    for name, shape in output_shapes.items():
        if len(shape) != 4:
            raise ValueError(f"Output {name} must be rank 4, got {shape}")
    policy_channels = output_shapes["OutputPolicy"][1]
    if policy_channels not in (1, 2, 4):
        raise ValueError(f"Unexpected policy channel count: {policy_channels}")
    if output_shapes["OutputPolicyPass"][1:] != (policy_channels, 1, 1):
        raise ValueError("OutputPolicyPass does not match OutputPolicy channels")
    if output_shapes["OutputPolicy"][2:] != spatial_shape[2:]:
        raise ValueError("OutputPolicy board dimensions do not match InputSpatial")
    if output_shapes["OutputValue"][1:] != (3, 1, 1):
        raise ValueError(
            f"OutputValue must be [N,3,1,1], got {output_shapes['OutputValue']}"
        )
    if output_shapes["OutputScoreValue"][1:] != (6, 1, 1):
        raise ValueError(
            f"OutputScoreValue must be [N,6,1,1], got {output_shapes['OutputScoreValue']}"
        )
    if output_shapes["OutputOwnership"][1:] != (1,) + spatial_shape[2:]:
        raise ValueError("OutputOwnership board dimensions do not match InputSpatial")

    metadata = {entry.key: entry.value for entry in model.metadata_props}
    if require_producer_metadata and (
        model.producer_name != "katago" or "modelVersion" not in metadata
    ):
        raise ValueError(
            "The source must be the ONNX emitted by KataGo (producer_name=katago with modelVersion metadata)"
        )
    return specs


def resolve_npz_files(paths: Sequence[str]) -> List[str]:
    """Resolve files, directories, and glob expressions deterministically."""

    found: List[Path] = []
    for raw in paths:
        expanded = os.path.expandvars(os.path.expanduser(raw))
        candidate = Path(expanded)
        if candidate.is_dir():
            found.extend(candidate.rglob("*.npz"))
        elif candidate.is_file():
            if candidate.suffix.lower() != ".npz":
                raise ValueError(f"Calibration input is not an .npz file: {candidate}")
            found.append(candidate)
        else:
            matches = [Path(path) for path in glob.glob(expanded, recursive=True)]
            found.extend(
                path
                for path in matches
                if path.is_file() and path.suffix.lower() == ".npz"
            )

    unique: Dict[str, Path] = {}
    for path in found:
        resolved = path.resolve()
        unique[os.path.normcase(str(resolved))] = resolved
    result = [str(path) for _, path in sorted(unique.items(), key=lambda item: item[0])]
    if not result:
        raise ValueError(f"No .npz files found in: {list(paths)}")
    return result


def _npz_sample_count(path: str) -> Tuple[int, str]:
    with np.load(path, allow_pickle=False) as data:
        if "InputSpatial" in data:
            return int(data["InputSpatial"].shape[0]), "onnx-inputs"
        if "binaryInputNCHWPacked" in data and "globalInputNC" in data:
            return int(data["globalInputNC"].shape[0]), "katago-training"
        raise ValueError(
            f"{path} is neither an ONNX-input NPZ nor a KataGo training NPZ. "
            "Expected InputSpatial or binaryInputNCHWPacked/globalInputNC."
        )


def _make_history_matrices() -> Tuple[np.ndarray, np.ndarray]:
    """Numpy equivalent of data_processing_pytorch.build_history_matrices."""

    diagonal = np.array(
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 1, 1],
        dtype=np.float32,
    )
    h_base = np.diag(diagonal)
    h_base[14, 15] = 1.0
    h_base[14, 16] = 1.0

    builders = np.zeros((5, 22, 22), dtype=np.float32)
    builders[0, 9, 9] = 1.0
    builders[0, 14, 15] = -1.0
    builders[0, 14, 16] = -1.0
    builders[0, 15, 15] = 1.0
    builders[0, 15, 16] = 1.0
    builders[1, 10, 10] = 1.0
    builders[1, 15, 16] = -1.0
    builders[1, 16, 16] = 1.0
    builders[2, 11, 11] = 1.0
    builders[3, 12, 12] = 1.0
    builders[4, 13, 13] = 1.0
    return h_base.reshape(1, 22, 22), builders


def _history_inclusion(
    sample_count: int, history_mode: str, rng: np.random.Generator
) -> np.ndarray:
    if history_mode == "full":
        return np.ones((sample_count, 5), dtype=np.float32)
    if history_mode == "none":
        return np.zeros((sample_count, 5), dtype=np.float32)
    if history_mode != "training":
        raise ValueError(f"Unknown history mode: {history_mode}")

    # Match KataGo training's 2% chance of stopping at each successive history
    # plane, but use an explicit generator so the exported scales are reproducible.
    should_stop = rng.random((sample_count, 5)) >= 0.98
    return (np.cumsum(should_stop, axis=1, dtype=np.int32) == 0).astype(np.float32)


def _apply_history_selection(
    spatial: np.ndarray, global_input: np.ndarray, include_history: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    if spatial.shape[1] != 22 or global_input.shape[1] != 19:
        raise ValueError(
            f"History transform expects 22 spatial and 19 global features, got "
            f"{spatial.shape[1]} and {global_input.shape[1]}"
        )
    h_base, h_builder = _make_history_matrices()
    matrices = h_base + np.einsum("bi,ijk->bjk", include_history, h_builder)
    spatial = np.einsum("bijk,bil->bljk", spatial, matrices).astype(
        np.float32, copy=False
    )
    multiplier = np.pad(include_history, ((0, 0), (0, 14)), constant_values=1.0)
    global_input = (global_input * multiplier).astype(np.float32, copy=False)
    return spatial, global_input


def _apply_spatial_symmetry(value: np.ndarray, symmetry: int) -> np.ndarray:
    """Apply KataGo's numbered D4 symmetry to the final two tensor axes."""

    if symmetry == 0:
        return value
    if symmetry == 1:
        return np.flip(np.swapaxes(value, -2, -1), axis=-2)
    if symmetry == 2:
        return np.flip(value, axis=(-2, -1))
    if symmetry == 3:
        return np.flip(np.swapaxes(value, -2, -1), axis=-1)
    if symmetry == 4:
        return np.swapaxes(value, -2, -1)
    if symmetry == 5:
        return np.flip(value, axis=-1)
    if symmetry == 6:
        return np.flip(np.swapaxes(value, -2, -1), axis=(-2, -1))
    if symmetry == 7:
        return np.flip(value, axis=-2)
    raise ValueError(f"Symmetry must be in [0,7], got {symmetry}")


def _augment_symmetries(
    inputs: Mapping[str, np.ndarray],
    symmetry_mode: str,
    seed: int,
) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
    """Apply deterministic inference symmetries and return effective symmetry ids.

    ``random`` applies one seeded symmetry to each source position. ``all``
    expands every source position into eight adjacent rows in symmetry order
    0..7. Non-spatial inputs are repeated but never transformed.
    """

    if symmetry_mode not in ("random", "all", "none"):
        raise ValueError(f"Unknown symmetry mode: {symmetry_mode}")
    spatial = np.asarray(inputs["InputSpatial"])
    if spatial.shape[-2] != spatial.shape[-1] and symmetry_mode != "none":
        raise ValueError(
            "Random/all D4 symmetry augmentation requires a square ONNX board; "
            "use --symmetry-mode none for a rectangular graph"
        )
    base_count = int(spatial.shape[0])
    spatial_names = frozenset(("InputMask", "InputSpatial"))

    if symmetry_mode == "none":
        none_symmetry_ids = np.zeros(base_count, dtype=np.uint8)
        return (
            {name: np.ascontiguousarray(value) for name, value in inputs.items()},
            none_symmetry_ids,
        )

    if symmetry_mode == "random":
        symmetry_rng = np.random.default_rng(
            np.random.SeedSequence([int(seed), 0x4B415441])
        )
        random_symmetry_ids = symmetry_rng.integers(
            0, 8, size=base_count, dtype=np.uint8
        )
        augmented: Dict[str, np.ndarray] = {}
        for name, value in inputs.items():
            value = np.asarray(value)
            if name not in spatial_names:
                augmented[name] = np.ascontiguousarray(value)
                continue
            transformed = np.empty_like(value)
            for symmetry in range(8):
                indices = np.flatnonzero(random_symmetry_ids == symmetry)
                if indices.size > 0:
                    transformed[indices] = _apply_spatial_symmetry(
                        value[indices], symmetry
                    )
            augmented[name] = np.ascontiguousarray(transformed)
        return augmented, random_symmetry_ids

    all_symmetry_ids = np.tile(np.arange(8, dtype=np.uint8), base_count)
    augmented = {}
    for name, value in inputs.items():
        value = np.asarray(value)
        if name not in spatial_names:
            augmented[name] = np.ascontiguousarray(np.repeat(value, 8, axis=0))
            continue
        transformed = np.empty((base_count, 8) + value.shape[1:], dtype=value.dtype)
        for symmetry in range(8):
            transformed[:, symmetry] = _apply_spatial_symmetry(value, symmetry)
        augmented[name] = np.ascontiguousarray(
            transformed.reshape((base_count * 8,) + value.shape[1:])
        )
    return augmented, all_symmetry_ids


def _normalize_nc11(value: np.ndarray, name: str) -> np.ndarray:
    value = np.asarray(value, dtype=np.float32)
    if value.ndim == 2:
        value = value[:, :, None, None]
    if value.ndim != 4 or value.shape[2:] != (1, 1):
        raise ValueError(
            f"{name} must have shape [N,C] or [N,C,1,1], got {value.shape}"
        )
    return np.ascontiguousarray(value)


def _extract_training_rows(
    data: Mapping[str, np.ndarray],
    row_indices: np.ndarray,
    specs: Sequence[InputSpec],
    include_history: np.ndarray,
) -> Dict[str, np.ndarray]:
    spatial_spec = next(spec for spec in specs if spec.name == "InputSpatial")
    if spatial_spec.shape[2] is None or spatial_spec.shape[3] is None:
        raise ValueError("KataGo ONNX must have fixed board dimensions")
    height = int(spatial_spec.shape[2])
    width = int(spatial_spec.shape[3])
    packed = np.asarray(data["binaryInputNCHWPacked"][row_indices])
    spatial = np.unpackbits(packed, axis=2)
    padded_area = ((height * width + 7) // 8) * 8
    if spatial.shape[2] != padded_area:
        raise ValueError(
            f"Packed spatial width is {spatial.shape[2]} bits, expected {padded_area} "
            f"for {height}x{width}"
        )
    spatial = spatial[:, :, : height * width].reshape(
        len(row_indices), spatial.shape[1], height, width
    )
    spatial = spatial.astype(np.float32, copy=False)
    global_input = np.asarray(data["globalInputNC"][row_indices], dtype=np.float32)
    spatial, global_input = _apply_history_selection(
        spatial, global_input, include_history
    )

    result: Dict[str, np.ndarray] = {
        "InputMask": np.ascontiguousarray(spatial[:, 0:1]),
        "InputSpatial": np.ascontiguousarray(spatial),
        "InputGlobal": _normalize_nc11(global_input, "InputGlobal"),
    }
    if any(spec.name == "InputMeta" for spec in specs):
        if "metadataInputNC" not in data:
            raise ValueError(
                "The ONNX graph requires InputMeta but metadataInputNC is absent"
            )
        result["InputMeta"] = _normalize_nc11(
            np.asarray(data["metadataInputNC"][row_indices], dtype=np.float32),
            "InputMeta",
        )
    return result


def _extract_onnx_input_rows(
    data: Mapping[str, np.ndarray],
    row_indices: np.ndarray,
    specs: Sequence[InputSpec],
    include_history: np.ndarray,
) -> Dict[str, np.ndarray]:
    extracted: Dict[str, np.ndarray] = {}
    for spec in specs:
        if spec.name not in data:
            raise ValueError(f"Expanded input NPZ is missing {spec.name}")
        extracted[spec.name] = np.asarray(
            data[spec.name][row_indices], dtype=np.float32
        )

    spatial = extracted["InputSpatial"]
    global_input = extracted["InputGlobal"]
    if global_input.ndim == 4 and global_input.shape[2:] == (1, 1):
        global_input = global_input[:, :, 0, 0]
    spatial, global_input = _apply_history_selection(
        spatial, global_input, include_history
    )
    extracted["InputSpatial"] = spatial
    extracted["InputGlobal"] = global_input

    result: Dict[str, np.ndarray] = {}
    for spec in specs:
        value = extracted[spec.name]
        if spec.name in ("InputGlobal", "InputMeta"):
            result[spec.name] = _normalize_nc11(value, spec.name)
        else:
            result[spec.name] = np.ascontiguousarray(value)
    return result


def _limit_source_files(
    files: Sequence[str],
    counts: Sequence[int],
    formats: Sequence[str],
    sample_count: int,
    max_source_files: int,
    seed: int,
) -> Tuple[List[str], List[int], List[str]]:
    """Bound expensive NPZ decompressions while retaining seeded shard sampling."""

    if max_source_files < 0:
        raise ValueError("max_source_files must be nonnegative (0 means unlimited)")
    if max_source_files == 0 or len(files) <= max_source_files:
        return list(files), list(counts), list(formats)

    limit = min(max_source_files, len(files))
    largest = sorted(range(len(files)), key=lambda index: (-counts[index], index))[
        :limit
    ]
    largest_capacity = sum(counts[index] for index in largest)
    if largest_capacity < sample_count:
        raise ValueError(
            f"No {limit} source files contain the requested {sample_count} positions; "
            "increase --max-source-files or use 0 for exact full-corpus sampling"
        )

    rng = np.random.default_rng(np.random.SeedSequence([int(seed), 0x53484152]))
    probabilities = np.asarray(counts, dtype=np.float64)
    probabilities /= np.sum(probabilities)
    selected = rng.choice(
        len(files), size=limit, replace=False, p=probabilities
    ).tolist()
    # Highly uneven custom shards can produce a subset too small for the
    # requested number of unique rows. Fall back to the largest eligible set;
    # official shuffled shards are nearly equal-sized and do not hit this path.
    if sum(counts[index] for index in selected) < sample_count:
        selected = largest
    selected.sort()
    return (
        [files[index] for index in selected],
        [int(counts[index]) for index in selected],
        [formats[index] for index in selected],
    )


def _validate_batch_shapes(
    batch: Mapping[str, np.ndarray], specs: Sequence[InputSpec]
) -> None:
    expected_names = [spec.name for spec in specs]
    if list(batch.keys()) != expected_names:
        raise ValueError(
            f"Input order mismatch: expected {expected_names}, got {list(batch.keys())}"
        )
    batch_size: Optional[int] = None
    for spec in specs:
        value = batch[spec.name]
        if value.dtype != np.float32:
            raise ValueError(f"{spec.name} must be float32, got {value.dtype}")
        if len(value.shape) != len(spec.shape):
            raise ValueError(
                f"{spec.name} rank mismatch: expected {spec.shape}, got {value.shape}"
            )
        for axis, (actual, expected) in enumerate(zip(value.shape, spec.shape)):
            if axis == 0:
                continue
            if expected is not None and actual != expected:
                raise ValueError(
                    f"{spec.name} shape mismatch at axis {axis}: expected {spec.shape}, got {value.shape}"
                )
        if batch_size is None:
            batch_size = int(value.shape[0])
        elif value.shape[0] != batch_size:
            raise ValueError("All ONNX inputs must have the same batch dimension")


def load_position_dataset(
    paths: Sequence[str],
    input_specs: Sequence[InputSpec],
    sample_count: int,
    batch_size: int,
    seed: int,
    history_mode: str = "training",
    symmetry_mode: str = "random",
    max_source_files: int = 64,
) -> PositionDataset:
    """Uniformly sample real positions and materialize deterministic input batches.

    ``sample_count`` is the number of unique source positions. In ``all``
    symmetry mode, each source position produces eight effective samples.
    """

    if sample_count <= 0 or batch_size <= 0:
        raise ValueError("sample_count and batch_size must be positive")
    files = resolve_npz_files(paths)
    available_source_file_count = len(files)
    counts: List[int] = []
    formats: List[str] = []
    for path in files:
        count, data_format = _npz_sample_count(path)
        if count <= 0:
            continue
        counts.append(count)
        formats.append(data_format)
    if len(counts) != len(files):
        raise ValueError("Empty .npz files are not supported")
    if len(set(formats)) != 1:
        raise ValueError(
            "Do not mix expanded ONNX-input NPZs with KataGo training NPZs"
        )
    files, counts, formats = _limit_source_files(
        files, counts, formats, sample_count, max_source_files, seed
    )
    total = int(sum(counts))
    if sample_count > total:
        raise ValueError(
            f"Requested {sample_count} positions but only {total} are available"
        )

    sampling_rng = np.random.default_rng(np.random.SeedSequence([int(seed), 0x504F53]))
    history_rng = np.random.default_rng(np.random.SeedSequence([int(seed), 0x484953]))
    selected_global = sampling_rng.choice(total, size=sample_count, replace=False)
    include_history = _history_inclusion(sample_count, history_mode, history_rng)
    cumulative = np.cumsum(np.array([0] + counts, dtype=np.int64))

    positions_by_file: Dict[int, List[Tuple[int, int]]] = {}
    for output_position, global_row in enumerate(selected_global.tolist()):
        file_index = int(np.searchsorted(cumulative, global_row, side="right") - 1)
        local_row = int(global_row - cumulative[file_index])
        positions_by_file.setdefault(file_index, []).append(
            (output_position, local_row)
        )

    combined: Dict[str, Optional[np.ndarray]] = {
        spec.name: None for spec in input_specs
    }
    source_manifest: List[Dict[str, Any]] = []
    selection_hasher = hashlib.sha256()
    for file_index, path in enumerate(files):
        pairs = positions_by_file.get(file_index, [])
        stat = os.stat(path)
        source_entry: Dict[str, Any] = {
            "path": str(Path(path).resolve()),
            "size": int(stat.st_size),
            "mtime_ns": int(stat.st_mtime_ns),
            "available_positions": counts[file_index],
            "selected_positions": len(pairs),
        }
        source_manifest.append(source_entry)
        if not pairs:
            continue

        output_positions = np.array([pair[0] for pair in pairs], dtype=np.int64)
        local_rows = np.array([pair[1] for pair in pairs], dtype=np.int64)
        selection_hasher.update(str(Path(path).resolve()).encode("utf-8"))
        selection_hasher.update(local_rows.tobytes(order="C"))
        with np.load(path, allow_pickle=False) as data:
            if formats[file_index] == "katago-training":
                extracted = _extract_training_rows(
                    data,
                    local_rows,
                    input_specs,
                    include_history[output_positions],
                )
            else:
                extracted = _extract_onnx_input_rows(
                    data,
                    local_rows,
                    input_specs,
                    include_history[output_positions],
                )

        for spec in input_specs:
            value = extracted[spec.name]
            if combined[spec.name] is None:
                combined[spec.name] = np.empty(
                    (sample_count,) + tuple(value.shape[1:]), dtype=np.float32
                )
            combined[spec.name][output_positions] = value  # type: ignore[index]

    concrete = {
        name: np.ascontiguousarray(value)
        for name, value in combined.items()
        if value is not None
    }
    if len(concrete) != len(input_specs):
        raise RuntimeError("Failed to materialize every model input")

    concrete, symmetry_ids = _augment_symmetries(concrete, symmetry_mode, seed)
    effective_sample_count = int(symmetry_ids.size)
    symmetry_hasher = hashlib.sha256()
    symmetry_hasher.update(symmetry_mode.encode("utf-8"))
    symmetry_hasher.update(symmetry_ids.tobytes(order="C"))
    symmetry_sha256 = symmetry_hasher.hexdigest()
    selection_hasher.update(b"\x00symmetry\x00")
    selection_hasher.update(symmetry_mode.encode("utf-8"))
    selection_hasher.update(symmetry_ids.tobytes(order="C"))
    symmetry_counts = {
        str(symmetry): int(np.count_nonzero(symmetry_ids == symmetry))
        for symmetry in range(8)
    }

    position_hasher = hashlib.sha256()
    for spec in input_specs:
        position_hasher.update(spec.name.encode("utf-8"))
        position_hasher.update(concrete[spec.name].tobytes(order="C"))

    batches: List[Dict[str, np.ndarray]] = []
    for start in range(0, effective_sample_count, batch_size):
        end = min(start + batch_size, effective_sample_count)
        batch = {
            spec.name: np.ascontiguousarray(concrete[spec.name][start:end])
            for spec in input_specs
        }
        _validate_batch_shapes(batch, input_specs)
        batches.append(batch)

    return PositionDataset(
        batches=batches,
        sample_count=effective_sample_count,
        base_sample_count=sample_count,
        batch_size=batch_size,
        seed=seed,
        history_mode=history_mode,
        symmetry_mode=symmetry_mode,
        symmetry_counts=symmetry_counts,
        symmetry_sha256=symmetry_sha256,
        position_sha256=position_hasher.hexdigest(),
        selection_sha256=selection_hasher.hexdigest(),
        available_source_file_count=available_source_file_count,
        selected_source_file_count=len(files),
        max_source_files=max_source_files,
        source_files=source_manifest,
        input_shapes={name: list(value.shape) for name, value in concrete.items()},
    )


def _constant_tensor_names(model: onnx.ModelProto) -> set[str]:
    constants = {initializer.name for initializer in model.graph.initializer}
    changed = True
    while changed:
        changed = False
        for node in model.graph.node:
            if not node.output or any(output in constants for output in node.output):
                continue
            if node.op_type == "Constant":
                constants.update(node.output)
                changed = True
            elif node.op_type in _CONSTANT_PASSTHROUGH_OPS and node.input:
                if all(
                    input_name in constants for input_name in node.input if input_name
                ):
                    constants.update(node.output)
                    changed = True
    return constants


def _node_has_weight(node: onnx.NodeProto, constant_names: set[str]) -> bool:
    if node.op_type == "Conv":
        return len(node.input) >= 2 and node.input[1] in constant_names
    if node.op_type == "Gemm":
        return len(node.input) >= 2 and node.input[1] in constant_names
    if node.op_type == "MatMul":
        return any(input_name in constant_names for input_name in node.input)
    return False


def select_quantizable_nodes(
    model: onnx.ModelProto,
    scope: str = "transformer",
    include_regexes: Sequence[str] = (),
    exclude_regexes: Sequence[str] = (),
    only_regexes: Sequence[str] = (),
) -> NodeSelection:
    """Select weighted nodes while excluding attention activation matmuls by construction."""

    if scope not in ("transformer", "all-weighted"):
        raise ValueError(f"Unknown quantization scope: {scope}")
    include_patterns = [re.compile(pattern) for pattern in include_regexes]
    exclude_patterns = [re.compile(pattern) for pattern in exclude_regexes]
    only_patterns = [re.compile(pattern) for pattern in only_regexes]
    constants = _constant_tensor_names(model)

    selected: List[str] = []
    candidates: List[str] = []
    selected_counts: Dict[str, int] = {}
    candidate_counts: Dict[str, int] = {}
    rejected: List[Dict[str, str]] = []
    for index, node in enumerate(model.graph.node):
        if node.op_type not in _SUPPORTED_WEIGHTED_OPS:
            continue
        node_name = node.name or f"__unnamed_{node.op_type}_{index}"
        if not node.name:
            raise ValueError(
                f"Quantizable {node.op_type} node at index {index} has no name; "
                "node-level reproducible selection is impossible"
            )
        if not _node_has_weight(node, constants):
            rejected.append(
                {"name": node_name, "reason": "no constant weight (activation matmul)"}
            )
            continue
        candidates.append(node_name)
        candidate_counts[node.op_type] = candidate_counts.get(node.op_type, 0) + 1

        selected_by_scope = scope == "all-weighted" or bool(
            _TRANSFORMER_PROJECTION_RE.search(node_name)
        )
        if include_patterns and any(
            pattern.search(node_name) for pattern in include_patterns
        ):
            selected_by_scope = True
        if not selected_by_scope:
            rejected.append({"name": node_name, "reason": "outside selected scope"})
            continue
        if only_patterns and not any(
            pattern.search(node_name) for pattern in only_patterns
        ):
            rejected.append({"name": node_name, "reason": "outside node restriction"})
            continue
        if any(pattern.search(node_name) for pattern in exclude_patterns):
            rejected.append({"name": node_name, "reason": "matched exclusion regex"})
            continue
        selected.append(node_name)
        selected_counts[node.op_type] = selected_counts.get(node.op_type, 0) + 1

    if not selected:
        raise ValueError(
            "No weighted nodes were selected. Ensure this is the official KataGo-emitted ONNX "
            "and inspect node names before overriding the selection regexes."
        )
    return NodeSelection(
        scope=scope,
        selected_names=sorted(selected),
        selected_by_op_type=dict(sorted(selected_counts.items())),
        weighted_candidate_names=sorted(candidates),
        weighted_candidates_by_op_type=dict(sorted(candidate_counts.items())),
        rejected=rejected,
    )


def _external_locations(model_path: str, model: onnx.ModelProto) -> List[str]:
    base = Path(model_path).resolve().parent
    result: set[str] = set()
    for initializer in model.graph.initializer:
        if initializer.data_location != TensorProto.EXTERNAL:
            continue
        for entry in initializer.external_data:
            if entry.key == "location":
                result.add(str((base / entry.value).resolve()))
    return sorted(result)


def sha256_files(paths: Sequence[str]) -> str:
    hasher = hashlib.sha256()
    for path in sorted(paths, key=os.path.normcase):
        resolved = str(Path(path).resolve())
        hasher.update(Path(resolved).name.encode("utf-8"))
        with open(resolved, "rb") as handle:
            while True:
                chunk = handle.read(8 * 1024 * 1024)
                if not chunk:
                    break
                hasher.update(chunk)
    return hasher.hexdigest()


def artifact_manifest(
    model_path: str, model: Optional[onnx.ModelProto] = None
) -> Dict[str, Any]:
    if model is None:
        model = onnx.load(model_path, load_external_data=False)
    files = [str(Path(model_path).resolve())] + _external_locations(model_path, model)
    missing = [path for path in files if not os.path.isfile(path)]
    if missing:
        raise FileNotFoundError(f"Missing ONNX external data files: {missing}")
    return {
        "path": str(Path(model_path).resolve()),
        "files": [{"path": path, "size": int(os.path.getsize(path))} for path in files],
        "total_size": int(sum(os.path.getsize(path) for path in files)),
        "sha256": sha256_files(files),
        "opsets": {
            item.domain or "ai.onnx": int(item.version) for item in model.opset_import
        },
        "metadata": {item.key: item.value for item in model.metadata_props},
    }


@dataclass(frozen=True)
class _ConstantValue:
    array: np.ndarray
    data_type: int
    shape: Tuple[int, ...]
    source: str


def _node_attribute_int(node: onnx.NodeProto, name: str, default: int) -> int:
    for attribute in node.attribute:
        if attribute.name == name:
            return int(attribute.i)
    return default


def _tensor_shape_from_value_info(
    value: onnx.ValueInfoProto,
) -> Optional[Tuple[Optional[int], ...]]:
    tensor_type = value.type.tensor_type
    if not tensor_type.HasField("shape"):
        return None
    return tuple(
        int(dimension.dim_value) if dimension.HasField("dim_value") else None
        for dimension in tensor_type.shape.dim
    )


def audit_qdq_model(model_path: str, selected_names: Sequence[str]) -> Dict[str, Any]:
    """Deeply audit the explicit Q/DQ contract expected by TensorRT.

    The audit deliberately checks graph semantics rather than only counting Q/DQ
    nodes.  In particular, every selected weighted input must follow
    ``QuantizeLinear -> DequantizeLinear -> weighted op`` and use symmetric
    INT8 or FP8 quantization.  The returned ``errors`` list is the authoritative
    pass/fail signal; the older summary fields are retained for report and CLI
    compatibility.
    """

    model = onnx.load(model_path, load_external_data=False)
    q_nodes = [node for node in model.graph.node if node.op_type == "QuantizeLinear"]
    dq_nodes = [node for node in model.graph.node if node.op_type == "DequantizeLinear"]
    initializer_by_name = {
        initializer.name: initializer for initializer in model.graph.initializer
    }
    producer_by_output = {
        output: node for node in model.graph.node for output in node.output if output
    }
    consumers_by_input: Dict[str, List[onnx.NodeProto]] = {}
    for node in model.graph.node:
        for input_name in node.input:
            if input_name:
                consumers_by_input.setdefault(input_name, []).append(node)

    constant_node_by_output = {
        output: node
        for node in model.graph.node
        if node.op_type == "Constant"
        for output in node.output
        if output
    }
    constant_cache: Dict[str, Optional[_ConstantValue]] = {}

    def node_label(node: onnx.NodeProto) -> str:
        if node.name:
            return node.name
        output = node.output[0] if node.output else "no-output"
        return f"{node.op_type}[{output}]"

    def constant_value(name: str) -> Optional[_ConstantValue]:
        if name in constant_cache:
            return constant_cache[name]
        tensor = initializer_by_name.get(name)
        source = "initializer"
        if tensor is None:
            constant_node = constant_node_by_output.get(name)
            if constant_node is None:
                constant_cache[name] = None
                return None
            source = f"Constant node {node_label(constant_node)}"
            for attribute in constant_node.attribute:
                if attribute.name == "value" and attribute.HasField("t"):
                    tensor = attribute.t
                    break
            if tensor is None:
                constant_cache[name] = None
                return None
        try:
            array = np.asarray(
                onnx.numpy_helper.to_array(
                    tensor, base_dir=str(Path(model_path).resolve().parent)
                )
            )
        except Exception:
            constant_cache[name] = None
            return None
        value = _ConstantValue(
            array=array,
            data_type=int(tensor.data_type),
            shape=tuple(int(dimension) for dimension in tensor.dims),
            source=source,
        )
        constant_cache[name] = value
        return value

    tensor_shapes: Dict[str, Tuple[Optional[int], ...]] = {
        initializer.name: tuple(int(dimension) for dimension in initializer.dims)
        for initializer in model.graph.initializer
    }
    for value in (
        list(model.graph.input)
        + list(model.graph.value_info)
        + list(model.graph.output)
    ):
        shape = _tensor_shape_from_value_info(value)
        if shape is not None:
            tensor_shapes[value.name] = shape
    for output_name in constant_node_by_output:
        value = constant_value(output_name)
        if value is not None:
            tensor_shapes[output_name] = tuple(value.shape)

    error_groups: Dict[str, set[str]] = {
        "qdq_chain": set(),
        "scale": set(),
        "zero_point": set(),
        "axis": set(),
        "granularity": set(),
        "qtype": set(),
        "unexpected_quantization": set(),
    }

    def add_error(category: str, message: str) -> None:
        error_groups[category].add(message)

    def dtype_name(data_type: Optional[int]) -> str:
        if data_type is None:
            return "UNSPECIFIED"
        try:
            return TensorProto.DataType.Name(int(data_type))
        except ValueError:
            return f"UNKNOWN({data_type})"

    allowed_scale_types = {
        TensorProto.FLOAT,
        TensorProto.FLOAT16,
        TensorProto.BFLOAT16,
    }

    def validate_scale(node: onnx.NodeProto) -> Optional[_ConstantValue]:
        label = node_label(node)
        if len(node.input) < 2 or not node.input[1]:
            add_error("scale", f"{label}: missing scale input")
            return None
        value = constant_value(node.input[1])
        if value is None:
            add_error(
                "scale", f"{label}: scale {node.input[1]!r} is not a readable constant"
            )
            return None
        if value.data_type not in allowed_scale_types:
            add_error(
                "scale",
                f"{label}: scale has non-floating type {dtype_name(value.data_type)}",
            )
        if value.array.size == 0:
            add_error("scale", f"{label}: scale is empty")
            return value
        try:
            numeric = np.asarray(value.array, dtype=np.float64)
        except (TypeError, ValueError):
            add_error(
                "scale", f"{label}: scale cannot be interpreted as floating point"
            )
            return value
        if not np.all(np.isfinite(numeric)):
            add_error("scale", f"{label}: scale contains non-finite values")
        if np.any(numeric <= 0.0):
            add_error("scale", f"{label}: scale must be strictly positive")
        return value

    def validate_zero_point(
        node: onnx.NodeProto, target_type: Optional[int]
    ) -> Optional[_ConstantValue]:
        label = node_label(node)
        if len(node.input) < 3 or not node.input[2]:
            add_error(
                "zero_point", f"{label}: missing zero point for symmetric quantization"
            )
            return None
        value = constant_value(node.input[2])
        if value is None:
            add_error(
                "zero_point",
                f"{label}: zero point {node.input[2]!r} is not a readable constant",
            )
            return None
        if target_type is not None and value.data_type != target_type:
            add_error(
                "zero_point",
                f"{label}: zero point type {dtype_name(value.data_type)} does not match "
                f"quantized type {dtype_name(target_type)}",
            )
        try:
            all_zero = value.array.size > 0 and bool(np.all(value.array == 0))
        except (TypeError, ValueError):
            all_zero = False
        if not all_zero:
            add_error(
                "zero_point",
                f"{label}: zero point must be a non-empty all-zero constant",
            )
        return value

    qtype_by_output: Dict[str, Optional[int]] = {}

    def quantized_type(node: onnx.NodeProto) -> Optional[int]:
        output_dtype: Optional[int] = None
        for attribute in node.attribute:
            if attribute.name == "output_dtype":
                output_dtype = int(attribute.i)
        zero_point_type: Optional[int] = None
        if len(node.input) >= 3 and node.input[2]:
            zero_point = constant_value(node.input[2])
            if zero_point is not None:
                zero_point_type = zero_point.data_type
        if (
            output_dtype is not None
            and zero_point_type is not None
            and output_dtype != zero_point_type
        ):
            add_error(
                "qtype",
                f"{node_label(node)}: output_dtype {dtype_name(output_dtype)} does not match "
                f"zero point type {dtype_name(zero_point_type)}",
            )
        return output_dtype if output_dtype is not None else zero_point_type

    scale_by_output: Dict[str, Optional[_ConstantValue]] = {}
    zero_point_by_output: Dict[str, Optional[_ConstantValue]] = {}
    q_types: Dict[str, int] = {}
    constant_names = _constant_tensor_names(model)
    weight_q_count = 0
    activation_q_count = 0
    granularity_errors: List[str] = []

    for node in q_nodes:
        if not node.output:
            add_error("qdq_chain", f"{node_label(node)}: QuantizeLinear has no output")
            continue
        output_name = node.output[0]
        qtype = quantized_type(node)
        qtype_by_output[output_name] = qtype
        type_name = dtype_name(qtype)
        q_types[type_name] = q_types.get(type_name, 0) + 1
        scale = validate_scale(node)
        scale_by_output[output_name] = scale
        zero_point_by_output[output_name] = validate_zero_point(node, qtype)

        is_weight = bool(node.input) and node.input[0] in constant_names
        if is_weight:
            weight_q_count += 1
        else:
            activation_q_count += 1
            if scale is not None and scale.array.size != 1:
                message = f"{node_label(node)}: activation scale is not per-tensor"
                granularity_errors.append(message)
                add_error("granularity", message)

    for node in dq_nodes:
        if not node.output:
            add_error(
                "qdq_chain", f"{node_label(node)}: DequantizeLinear has no output"
            )
            continue
        output_name = node.output[0]
        scale_by_output[output_name] = validate_scale(node)
        upstream = producer_by_output.get(node.input[0]) if node.input else None
        upstream_type = (
            qtype_by_output.get(upstream.output[0])
            if upstream is not None
            and upstream.op_type == "QuantizeLinear"
            and upstream.output
            else None
        )
        zero_point_by_output[output_name] = validate_zero_point(node, upstream_type)

    graph_outputs = {value.name for value in model.graph.output}
    orphan_q_nodes: set[str] = set()
    orphan_dq_nodes: set[str] = set()
    for node in q_nodes:
        if not node.output:
            orphan_q_nodes.add(node_label(node))
            continue
        dq_consumers = [
            consumer
            for consumer in consumers_by_input.get(node.output[0], [])
            if consumer.op_type == "DequantizeLinear"
        ]
        if not dq_consumers:
            orphan_q_nodes.add(node_label(node))
            add_error(
                "qdq_chain",
                f"{node_label(node)}: QuantizeLinear output has no DQ consumer",
            )

    pair_by_dq_output: Dict[str, Dict[str, Any]] = {}
    for dq_node in dq_nodes:
        dq_label = node_label(dq_node)
        if not dq_node.input:
            orphan_dq_nodes.add(dq_label)
            add_error(
                "qdq_chain", f"{dq_label}: DequantizeLinear has no quantized input"
            )
            continue
        q_node = producer_by_output.get(dq_node.input[0])
        if q_node is None or q_node.op_type != "QuantizeLinear" or not q_node.output:
            orphan_dq_nodes.add(dq_label)
            add_error(
                "qdq_chain",
                f"{dq_label}: DQ input is not produced directly by QuantizeLinear",
            )
            continue
        if not dq_node.output:
            orphan_dq_nodes.add(dq_label)
            continue

        q_output = q_node.output[0]
        dq_output = dq_node.output[0]
        q_scale = scale_by_output.get(q_output)
        dq_scale = scale_by_output.get(dq_output)
        if q_scale is not None and dq_scale is not None:
            if (
                q_scale.data_type != dq_scale.data_type
                or q_scale.array.shape != dq_scale.array.shape
                or not np.array_equal(q_scale.array, dq_scale.array)
            ):
                add_error(
                    "scale", f"{node_label(q_node)} -> {dq_label}: Q/DQ scales differ"
                )

        q_zero = zero_point_by_output.get(q_output)
        dq_zero = zero_point_by_output.get(dq_output)
        if q_zero is not None and dq_zero is not None:
            if (
                q_zero.data_type != dq_zero.data_type
                or q_zero.array.shape != dq_zero.array.shape
                or not np.array_equal(q_zero.array, dq_zero.array)
            ):
                add_error(
                    "zero_point",
                    f"{node_label(q_node)} -> {dq_label}: Q/DQ zero points differ",
                )

        source_name = q_node.input[0] if q_node.input else ""
        source_shape = tensor_shapes.get(source_name)
        rank = len(source_shape) if source_shape is not None else None
        q_axis = _node_attribute_int(q_node, "axis", 1)
        dq_axis = _node_attribute_int(dq_node, "axis", 1)

        def normalize_axis(axis: int) -> int:
            return axis + rank if rank is not None and axis < 0 else axis

        effective_q_axis = normalize_axis(q_axis)
        effective_dq_axis = normalize_axis(dq_axis)
        if rank is not None and not 0 <= effective_q_axis < rank:
            add_error(
                "axis",
                f"{node_label(q_node)}: effective axis {effective_q_axis} is invalid for rank {rank}",
            )
        if rank is not None and not 0 <= effective_dq_axis < rank:
            add_error(
                "axis",
                f"{dq_label}: effective axis {effective_dq_axis} is invalid for rank {rank}",
            )
        if effective_q_axis != effective_dq_axis:
            add_error(
                "axis",
                f"{node_label(q_node)} -> {dq_label}: Q/DQ effective axes differ "
                f"({effective_q_axis} vs {effective_dq_axis})",
            )

        pair_by_dq_output[dq_output] = {
            "q_node": q_node,
            "dq_node": dq_node,
            "qtype": qtype_by_output.get(q_output),
            "scale": q_scale,
            "effective_axis": effective_q_axis,
            "source_tensor": source_name,
            "source_shape": source_shape,
        }

        if not consumers_by_input.get(dq_output) and dq_output not in graph_outputs:
            orphan_dq_nodes.add(dq_label)
            add_error("qdq_chain", f"{dq_label}: DequantizeLinear output is unused")

    model_nodes_by_name: Dict[str, List[onnx.NodeProto]] = {}
    for node in model.graph.node:
        if node.name:
            model_nodes_by_name.setdefault(node.name, []).append(node)
    selected_missing = sorted(
        name for name in selected_names if name not in model_nodes_by_name
    )
    for name in selected_missing:
        add_error("qdq_chain", f"selected node {name!r} is missing after quantization")

    selected_without_qdq: set[str] = set()
    selected_qdq_input_counts: Dict[str, int] = {}
    selected_chain_qtypes: Dict[str, Dict[str, str]] = {}
    selected_input_chains: Dict[str, Dict[str, Any]] = {}
    selected_set = set(selected_names)
    supported_selected_qtypes = {
        TensorProto.INT8,
        getattr(TensorProto, "FLOAT8E4M3FN", 17),
    }

    for selected_name in selected_names:
        nodes = model_nodes_by_name.get(selected_name, [])
        if not nodes:
            continue
        if len(nodes) != 1:
            add_error(
                "qdq_chain", f"selected node {selected_name!r} is not uniquely named"
            )
            selected_without_qdq.add(selected_name)
            continue
        node = nodes[0]
        weighted_inputs = list(node.input[:2])
        direct_dq_count = sum(
            1
            for input_name in weighted_inputs
            if input_name in producer_by_output
            and producer_by_output[input_name].op_type == "DequantizeLinear"
        )
        selected_qdq_input_counts[selected_name] = direct_dq_count
        if len(weighted_inputs) != 2 or direct_dq_count != 2:
            selected_without_qdq.add(selected_name)
            add_error(
                "qdq_chain",
                f"{selected_name}: selected weighted op does not have direct DQ on both inputs",
            )

        role_details: Dict[str, Any] = {}
        role_qtypes: Dict[str, str] = {}
        resolved_types: List[int] = []
        for input_index, role in ((0, "activation"), (1, "weight")):
            if input_index >= len(node.input):
                continue
            input_name = node.input[input_index]
            pair = pair_by_dq_output.get(input_name)
            if pair is None:
                selected_without_qdq.add(selected_name)
                add_error(
                    "qdq_chain",
                    f"{selected_name}: {role} DQ is not fed directly by QuantizeLinear",
                )
                continue
            qtype = pair["qtype"]
            role_qtypes[role] = dtype_name(qtype)
            if qtype is None:
                add_error(
                    "qtype", f"{selected_name}: {role} quantized type is unspecified"
                )
            else:
                resolved_types.append(qtype)
                if qtype not in supported_selected_qtypes:
                    add_error(
                        "qtype",
                        f"{selected_name}: {role} uses unsupported quantized type {dtype_name(qtype)}",
                    )
            scale = pair["scale"]
            scale_elements = int(scale.array.size) if scale is not None else None
            role_details[role] = {
                "q_node": node_label(pair["q_node"]),
                "dq_node": node_label(pair["dq_node"]),
                "qtype": dtype_name(qtype),
                "scale_elements": scale_elements,
                "effective_axis": pair["effective_axis"],
                "source_tensor": pair["source_tensor"],
                "source_shape": list(pair["source_shape"])
                if pair["source_shape"] is not None
                else None,
            }

            if role == "activation":
                if scale_elements is not None and scale_elements != 1:
                    message = f"{selected_name}: activation scale is not per-tensor"
                    granularity_errors.append(message)
                    add_error("granularity", message)
                continue

            source_shape = pair["source_shape"]
            if pair["source_tensor"] not in constant_names:
                add_error(
                    "granularity",
                    f"{selected_name}: weight Q input {pair['source_tensor']!r} is not constant",
                )
                continue
            if source_shape is None or any(
                dimension is None for dimension in source_shape
            ):
                add_error(
                    "granularity",
                    f"{selected_name}: weight shape is not statically known",
                )
                continue

            expected_axis: Optional[int] = None
            output_channels: Optional[int] = None
            if node.op_type == "MatMul":
                if len(source_shape) != 2:
                    add_error(
                        "axis",
                        f"{selected_name}: MatMul weight must be rank 2, got {source_shape}",
                    )
                else:
                    expected_axis = 1
                    output_channels = int(source_shape[1])
            elif node.op_type == "Conv":
                if not source_shape:
                    add_error("axis", f"{selected_name}: Conv weight has no dimensions")
                else:
                    expected_axis = 0
                    output_channels = int(source_shape[0])
            elif node.op_type == "Gemm":
                if len(source_shape) != 2:
                    add_error(
                        "axis",
                        f"{selected_name}: Gemm weight must be rank 2, got {source_shape}",
                    )
                else:
                    trans_b = _node_attribute_int(node, "transB", 0)
                    expected_axis = 0 if trans_b else 1
                    output_channels = int(source_shape[expected_axis])

            if expected_axis is not None and pair["effective_axis"] != expected_axis:
                add_error(
                    "axis",
                    f"{selected_name}: {node.op_type} weight axis {pair['effective_axis']} "
                    f"does not match expected axis {expected_axis}",
                )
            if (
                output_channels is not None
                and scale_elements is not None
                and scale_elements != output_channels
            ):
                message = (
                    f"{selected_name}: weight scale length {scale_elements} does not match "
                    f"output channels {output_channels}"
                )
                granularity_errors.append(message)
                add_error("granularity", message)

        if len(set(resolved_types)) > 1:
            add_error(
                "qtype",
                f"{selected_name}: selected input chains mix "
                f"{sorted(dtype_name(item) for item in set(resolved_types))}",
            )
        selected_chain_qtypes[selected_name] = role_qtypes
        selected_input_chains[selected_name] = role_details

    unexpected_fully_quantized_weighted: List[str] = []
    for node in model.graph.node:
        if node.op_type not in _SUPPORTED_WEIGHTED_OPS or node.name in selected_set:
            continue
        if len(node.input) < 2:
            continue
        pairs = [pair_by_dq_output.get(input_name) for input_name in node.input[:2]]
        if any(pair is None for pair in pairs):
            continue
        if not any(
            pair["source_tensor"] in constant_names
            for pair in pairs
            if pair is not None
        ):
            continue
        label = node_label(node)
        unexpected_fully_quantized_weighted.append(label)
        add_error(
            "unexpected_quantization",
            f"{label}: non-selected weighted op unexpectedly has Q/DQ on both inputs",
        )

    io_types = {
        "inputs": {
            value.name: TensorProto.DataType.Name(value.type.tensor_type.elem_type)
            for value in model.graph.input
        },
        "outputs": {
            value.name: TensorProto.DataType.Name(value.type.tensor_type.elem_type)
            for value in model.graph.output
        },
    }
    bad_io = {
        side: {name: dtype for name, dtype in values.items() if dtype != "FLOAT"}
        for side, values in io_types.items()
    }
    bad_io = {side: values for side, values in bad_io.items() if values}
    if bad_io:
        add_error("qdq_chain", f"graph I/O types are not all FLOAT: {bad_io}")

    error_categories = {
        category: sorted(messages)
        for category, messages in error_groups.items()
        if messages
    }
    errors = [
        f"{category}: {message}"
        for category, messages in error_categories.items()
        for message in messages
    ]

    return {
        "errors": errors,
        "error_count": len(errors),
        "error_categories": error_categories,
        "quantize_linear_count": len(q_nodes),
        "dequantize_linear_count": len(dq_nodes),
        "qdq_pair_count": len(pair_by_dq_output),
        "quantized_tensor_types": dict(sorted(q_types.items())),
        "weight_quantize_linear_count": weight_q_count,
        "activation_quantize_linear_count": activation_q_count,
        "scale_granularity_errors": sorted(set(granularity_errors)),
        "orphan_quantize_linear_nodes": sorted(orphan_q_nodes),
        "orphan_dequantize_linear_nodes": sorted(orphan_dq_nodes),
        "unexpected_fully_quantized_weighted_nodes": sorted(
            unexpected_fully_quantized_weighted
        ),
        "io_types": io_types,
        "non_float_io": bad_io,
        "selected_nodes_missing_after_quantization": selected_missing,
        "selected_nodes_without_qdq_inputs": sorted(selected_without_qdq),
        "selected_nodes_with_qdq_inputs": len(selected_qdq_input_counts)
        - len(selected_without_qdq),
        "selected_qdq_input_counts": dict(sorted(selected_qdq_input_counts.items())),
        "selected_chain_qtypes": dict(sorted(selected_chain_qtypes.items())),
        "selected_input_chains": dict(sorted(selected_input_chains.items())),
    }


def _protobuf_sha256(message: Any) -> str:
    """Hash one protobuf without depending on map/dictionary iteration order."""

    try:
        payload = message.SerializeToString(deterministic=True)
    except TypeError:  # pragma: no cover - compatibility with old protobuf releases
        payload = message.SerializeToString()
    return hashlib.sha256(payload).hexdigest()


def _initializer_content_manifest(
    initializer: onnx.TensorProto, model_path: str, roles: Sequence[str]
) -> Dict[str, Any]:
    """Return a location-independent hash of an initializer's exact tensor bytes."""

    array = np.asarray(
        onnx.numpy_helper.to_array(
            initializer, base_dir=str(Path(model_path).resolve().parent)
        )
    )
    contiguous = np.ascontiguousarray(array)
    byte_view = contiguous.view(np.uint8).reshape(-1)
    return {
        "data_type": int(initializer.data_type),
        "data_type_name": TensorProto.DataType.Name(int(initializer.data_type)),
        "dims": [int(value) for value in initializer.dims],
        "byte_count": int(byte_view.size),
        "byte_sha256": hashlib.sha256(byte_view).hexdigest(),
        "roles": sorted(set(roles)),
    }


def capture_existing_qdq_state(
    model_path: str, selected_names: Optional[Sequence[str]] = None
) -> Dict[str, Any]:
    """Fingerprint pre-existing fully-quantized weighted operators.

    This is intentionally stricter than :func:`audit_qdq_model`.  It records
    the weighted nodes, their direct Q/DQ chains, every scale/zero-point
    constant node, and every referenced initializer (including quantized
    weights).  Initializer hashes are based on the resolved tensor bytes, so
    moving an external-data artifact does not create a false mismatch.
    """

    model = onnx.load(model_path, load_external_data=False)
    producer_by_output = {
        output: node for node in model.graph.node for output in node.output if output
    }
    initializer_by_name = {
        initializer.name: initializer for initializer in model.graph.initializer
    }
    constant_names = _constant_tensor_names(model)
    node_by_name: Dict[str, onnx.NodeProto] = {}
    for index, node in enumerate(model.graph.node):
        if not node.name:
            continue
        if node.name in node_by_name:
            raise ValueError(f"ONNX graph has duplicate node name {node.name!r}")
        node_by_name[node.name] = node

    def direct_pair(input_name: str) -> Optional[Tuple[onnx.NodeProto, onnx.NodeProto]]:
        dq_node = producer_by_output.get(input_name)
        if (
            dq_node is None
            or dq_node.op_type != "DequantizeLinear"
            or not dq_node.input
        ):
            return None
        q_node = producer_by_output.get(dq_node.input[0])
        if q_node is None or q_node.op_type != "QuantizeLinear" or not q_node.input:
            return None
        return q_node, dq_node

    discovered: List[str] = []
    for node in model.graph.node:
        if (
            node.op_type not in _SUPPORTED_WEIGHTED_OPS
            or not node.name
            or len(node.input) < 2
        ):
            continue
        pairs = [direct_pair(input_name) for input_name in node.input[:2]]
        if any(pair is None for pair in pairs):
            continue
        assert all(pair is not None for pair in pairs)
        if not any(
            pair[0].input[0] in constant_names for pair in pairs if pair is not None
        ):
            continue
        discovered.append(node.name)

    if selected_names is None:
        protected_names = sorted(discovered)
    else:
        protected_names = sorted(set(selected_names))
        missing = sorted(set(protected_names) - set(discovered))
        if missing:
            raise ValueError(
                "Previously quantized weighted nodes no longer have two direct Q/DQ inputs: "
                f"{missing[:10]}"
            )

    if not protected_names:
        raise ValueError("No pre-existing fully-quantized weighted nodes were found")

    weighted_nodes: Dict[str, Dict[str, Any]] = {}
    qdq_nodes: Dict[str, Dict[str, Any]] = {}
    constant_nodes: Dict[str, Dict[str, Any]] = {}
    initializer_roles: Dict[str, set[str]] = {}

    def node_key(node: onnx.NodeProto) -> str:
        if node.name:
            return node.name
        if node.output:
            return f"{node.op_type}[{node.output[0]}]"
        raise ValueError(
            f"Protected {node.op_type} node has neither a name nor an output"
        )

    def record_constant(name: str, role: str) -> None:
        if name in initializer_by_name:
            initializer_roles.setdefault(name, set()).add(role)
            return
        node = producer_by_output.get(name)
        if node is None or node.op_type != "Constant":
            return
        key = node_key(node)
        entry = constant_nodes.setdefault(
            key,
            {
                "op_type": node.op_type,
                "protobuf_sha256": _protobuf_sha256(node),
                "roles": [],
            },
        )
        entry["roles"] = sorted(set(entry["roles"]) | {role})

    for weighted_name in protected_names:
        node = node_by_name.get(weighted_name)
        if node is None:
            raise ValueError(f"Protected weighted node {weighted_name!r} is missing")
        weighted_nodes[weighted_name] = {
            "op_type": node.op_type,
            "protobuf_sha256": _protobuf_sha256(node),
            "inputs": list(node.input),
            "outputs": list(node.output),
        }
        for input_index, input_name in enumerate(node.input[:2]):
            pair = direct_pair(input_name)
            if pair is None:
                raise ValueError(
                    f"Protected weighted node {weighted_name!r} input {input_index} lost Q/DQ"
                )
            q_node, dq_node = pair
            chain_role = f"{weighted_name}:input{input_index}"
            for kind, chain_node in (("Q", q_node), ("DQ", dq_node)):
                key = node_key(chain_node)
                entry = qdq_nodes.setdefault(
                    key,
                    {
                        "op_type": chain_node.op_type,
                        "protobuf_sha256": _protobuf_sha256(chain_node),
                        "roles": [],
                    },
                )
                if entry["protobuf_sha256"] != _protobuf_sha256(chain_node):
                    raise ValueError(f"Protected Q/DQ identity {key!r} is ambiguous")
                entry["roles"] = sorted(set(entry["roles"]) | {f"{chain_role}:{kind}"})

            if q_node.input:
                record_constant(q_node.input[0], f"{chain_role}:source")
            for index, role in ((1, "scale"), (2, "zero_point")):
                if index < len(q_node.input) and q_node.input[index]:
                    record_constant(q_node.input[index], f"{chain_role}:Q:{role}")
                if index < len(dq_node.input) and dq_node.input[index]:
                    record_constant(dq_node.input[index], f"{chain_role}:DQ:{role}")

    initializers = {
        name: _initializer_content_manifest(
            initializer_by_name[name], model_path, sorted(roles)
        )
        for name, roles in sorted(initializer_roles.items())
    }
    sections = {
        "weighted_nodes": weighted_nodes,
        "qdq_nodes": qdq_nodes,
        "constant_nodes": constant_nodes,
        "initializers": initializers,
    }
    aggregate_payload = json.dumps(
        sections, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return {
        "schema_version": 1,
        "selected_count": len(protected_names),
        "selected_names": protected_names,
        "discovered_fully_quantized_weighted_count": len(discovered),
        "discovered_fully_quantized_weighted_names": sorted(discovered),
        "weighted_nodes": weighted_nodes,
        "qdq_nodes": qdq_nodes,
        "constant_nodes": constant_nodes,
        "initializers": initializers,
        "weighted_node_count": len(weighted_nodes),
        "qdq_node_count": len(qdq_nodes),
        "constant_node_count": len(constant_nodes),
        "initializer_count": len(initializers),
        "aggregate_sha256": hashlib.sha256(aggregate_payload).hexdigest(),
    }


def compare_existing_qdq_state(
    model_path: str, expected: Mapping[str, Any]
) -> Dict[str, Any]:
    """Verify that an incremental PTQ pass preserved every protected byte."""

    actual = capture_existing_qdq_state(model_path, expected["selected_names"])
    differences: List[str] = []
    for section in ("weighted_nodes", "qdq_nodes", "constant_nodes", "initializers"):
        expected_items = expected.get(section, {})
        actual_items = actual.get(section, {})
        missing = sorted(set(expected_items) - set(actual_items))
        added = sorted(set(actual_items) - set(expected_items))
        changed = sorted(
            name
            for name in set(expected_items) & set(actual_items)
            if expected_items[name] != actual_items[name]
        )
        if missing:
            differences.append(f"{section}: missing {missing[:10]}")
        if added:
            differences.append(f"{section}: added {added[:10]}")
        if changed:
            differences.append(f"{section}: changed {changed[:10]}")
    return {
        "status": "passed" if not differences else "failed",
        "differences": differences,
        "expected_aggregate_sha256": expected.get("aggregate_sha256"),
        "actual_aggregate_sha256": actual["aggregate_sha256"],
        "actual": actual,
    }


def _softmax_log_probs(
    logits: np.ndarray, axis: int = -1
) -> Tuple[np.ndarray, np.ndarray]:
    logits64 = np.asarray(logits, dtype=np.float64)
    maximum = np.max(logits64, axis=axis, keepdims=True)
    shifted = logits64 - maximum
    log_sum = np.log(np.sum(np.exp(shifted), axis=axis, keepdims=True))
    log_probs = shifted - log_sum
    return np.exp(log_probs), log_probs


def _summary(values: np.ndarray) -> Dict[str, float]:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return {"mean": 0.0, "p99": 0.0, "max": 0.0, "nonfinite": int(values.size)}
    return {
        "mean": float(np.mean(finite)),
        "p99": float(np.quantile(finite, 0.99)),
        "max": float(np.max(finite)),
        "nonfinite": int(values.size - finite.size),
    }


def _output_error(reference: np.ndarray, candidate: np.ndarray) -> Dict[str, Any]:
    reference64 = np.asarray(reference, dtype=np.float64)
    candidate64 = np.asarray(candidate, dtype=np.float64)
    if reference64.shape != candidate64.shape:
        raise ValueError(
            f"Output shape mismatch: {reference64.shape} vs {candidate64.shape}"
        )
    finite_reference = np.isfinite(reference64)
    finite_candidate = np.isfinite(candidate64)
    finite = finite_reference & finite_candidate
    diff = np.where(finite, candidate64 - reference64, 0.0)
    abs_diff = np.abs(diff[finite])
    reference_finite = reference64[finite]
    squared_error = float(np.sum(diff[finite] * diff[finite]))
    reference_energy = float(np.sum(reference_finite * reference_finite))
    return {
        "shape": list(reference64.shape),
        "element_count": int(reference64.size),
        "max_abs": float(np.max(abs_diff)) if abs_diff.size else 0.0,
        "mean_abs": float(np.mean(abs_diff)) if abs_diff.size else 0.0,
        "p99_abs": float(np.quantile(abs_diff, 0.99)) if abs_diff.size else 0.0,
        "rmse": math.sqrt(squared_error / max(1, int(np.count_nonzero(finite)))),
        "rel_l2": math.sqrt(squared_error / max(reference_energy, 1.0e-30)),
        "reference_nonfinite": int(
            reference64.size - np.count_nonzero(finite_reference)
        ),
        "candidate_nonfinite": int(
            candidate64.size - np.count_nonzero(finite_candidate)
        ),
    }


def _masked_ownership_error(
    reference: np.ndarray, candidate: np.ndarray, mask: np.ndarray
) -> Dict[str, Any]:
    reference = np.asarray(reference)
    candidate = np.asarray(candidate)
    valid = np.asarray(mask) > 0.5
    valid = np.broadcast_to(valid, reference.shape)
    inboard = _output_error(reference[valid], candidate[valid])
    unmasked = _output_error(reference, candidate)
    offboard = _output_error(reference[~valid], candidate[~valid])
    # Quality gates use in-board ownership, but non-finite values anywhere in
    # the tensor remain fatal and off-board drift stays visible in the report.
    inboard["reference_nonfinite"] = unmasked["reference_nonfinite"]
    inboard["candidate_nonfinite"] = unmasked["candidate_nonfinite"]
    inboard["unmasked"] = unmasked
    inboard["offboard"] = offboard
    return inboard


def _per_channel_errors(
    name: str,
    reference: np.ndarray,
    candidate: np.ndarray,
    mask: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    reference = np.asarray(reference)
    candidate = np.asarray(candidate)
    if reference.ndim < 2:
        return {}
    labels = _OUTPUT_CHANNEL_LABELS.get(name, ())
    result: Dict[str, Any] = {}
    for channel in range(reference.shape[1]):
        label = labels[channel] if channel < len(labels) else str(channel)
        if mask is not None:
            channel_mask = np.asarray(mask) > 0.5
            channel_mask = np.broadcast_to(
                channel_mask, reference[:, channel : channel + 1].shape
            )
            metric = _output_error(
                reference[:, channel : channel + 1][channel_mask],
                candidate[:, channel : channel + 1][channel_mask],
            )
        else:
            metric = _output_error(reference[:, channel], candidate[:, channel])
        result[label] = metric
    return result


def _policy_metrics(
    reference_outputs: Mapping[str, np.ndarray],
    candidate_outputs: Mapping[str, np.ndarray],
    mask: np.ndarray,
) -> Dict[str, Any]:
    ref_board = np.asarray(reference_outputs["OutputPolicy"], dtype=np.float64)
    cand_board = np.asarray(candidate_outputs["OutputPolicy"], dtype=np.float64)
    ref_pass = np.asarray(reference_outputs["OutputPolicyPass"], dtype=np.float64)
    cand_pass = np.asarray(candidate_outputs["OutputPolicyPass"], dtype=np.float64)
    n, channels = ref_board.shape[:2]
    ref_board = ref_board.reshape(n, channels, -1)
    cand_board = cand_board.reshape(n, channels, -1)
    valid = np.asarray(mask).reshape(n, -1) > 0.5
    ref_pass = ref_pass.reshape(n, channels, -1)
    cand_pass = cand_pass.reshape(n, channels, -1)
    if ref_pass.shape[2] != 1 or cand_pass.shape[2] != 1:
        raise ValueError("OutputPolicyPass must contain exactly one logit per channel")
    per_channel: Dict[str, Any] = {}
    # Channels 0 and (when present) 1 are policy-logit distributions. Channels
    # 2/3 in four-channel models are per-move Q values and must never be judged
    # by a shift-invariant softmax KL.
    policy_channel_count = min(channels, 2)
    labels = _OUTPUT_CHANNEL_LABELS["OutputPolicy"]
    for channel in range(policy_channel_count):
        ref_board_channel = np.where(valid, ref_board[:, channel], -1.0e30)
        cand_board_channel = np.where(valid, cand_board[:, channel], -1.0e30)
        ref_logits = np.concatenate((ref_board_channel, ref_pass[:, channel]), axis=1)
        cand_logits = np.concatenate(
            (cand_board_channel, cand_pass[:, channel]), axis=1
        )
        ref_probs, ref_log_probs = _softmax_log_probs(ref_logits)
        _, cand_log_probs = _softmax_log_probs(cand_logits)
        kl = np.sum(ref_probs * (ref_log_probs - cand_log_probs), axis=1)
        agreement = np.argmax(ref_logits, axis=1) == np.argmax(cand_logits, axis=1)
        per_channel[labels[channel]] = {
            "kl": _summary(kl),
            "top1_agreement": float(np.mean(agreement)),
        }

    primary = per_channel["policy"]
    result: Dict[str, Any] = {
        "kl": primary["kl"],
        "top1_agreement": primary["top1_agreement"],
        "per_channel": per_channel,
    }
    if channels == 4:
        quantitative: Dict[str, Any] = {}
        for channel, label in ((2, "q_value"), (3, "q_score")):
            reference_values = np.concatenate(
                (ref_board[:, channel][valid], ref_pass[:, channel].reshape(-1))
            )
            candidate_values = np.concatenate(
                (cand_board[:, channel][valid], cand_pass[:, channel].reshape(-1))
            )
            quantitative[label] = _output_error(reference_values, candidate_values)
        result["quantitative"] = quantitative
    return result


def _value_metrics(reference: np.ndarray, candidate: np.ndarray) -> Dict[str, Any]:
    reference = np.asarray(reference, dtype=np.float64).reshape(reference.shape[0], -1)
    candidate = np.asarray(candidate, dtype=np.float64).reshape(candidate.shape[0], -1)
    if reference.shape[1] != 3:
        raise ValueError(f"OutputValue must have 3 channels, got {reference.shape}")
    ref_probs, ref_log_probs = _softmax_log_probs(reference)
    _, cand_log_probs = _softmax_log_probs(candidate)
    kl = np.sum(ref_probs * (ref_log_probs - cand_log_probs), axis=1)
    agreement = np.argmax(reference, axis=1) == np.argmax(candidate, axis=1)
    return {"kl": _summary(kl), "top1_agreement": float(np.mean(agreement))}


def compute_validation_metrics(
    reference_outputs: Mapping[str, np.ndarray],
    candidate_outputs: Mapping[str, np.ndarray],
    feed: Mapping[str, np.ndarray],
) -> Dict[str, Any]:
    missing = [
        name
        for name in EXPECTED_OUTPUT_NAMES
        if name not in reference_outputs or name not in candidate_outputs
    ]
    if missing:
        raise ValueError(f"Missing outputs for validation: {missing}")
    if "InputMask" not in feed:
        raise ValueError("InputMask is required for policy and ownership metrics")

    output_metrics: Dict[str, Any] = {}
    for name in EXPECTED_OUTPUT_NAMES:
        if name == "OutputOwnership":
            output_metrics[name] = _masked_ownership_error(
                reference_outputs[name], candidate_outputs[name], feed["InputMask"]
            )
            output_metrics[name]["per_channel"] = _per_channel_errors(
                name,
                reference_outputs[name],
                candidate_outputs[name],
                feed["InputMask"],
            )
        else:
            output_metrics[name] = _output_error(
                reference_outputs[name], candidate_outputs[name]
            )
            output_metrics[name]["per_channel"] = _per_channel_errors(
                name, reference_outputs[name], candidate_outputs[name]
            )
    return {
        "sample_count": int(np.asarray(feed["InputMask"]).shape[0]),
        "outputs": output_metrics,
        "policy": _policy_metrics(
            reference_outputs, candidate_outputs, feed["InputMask"]
        ),
        "value": _value_metrics(
            reference_outputs["OutputValue"], candidate_outputs["OutputValue"]
        ),
    }


def _get_nested(mapping: Mapping[str, Any], dotted_path: str) -> float:
    current: Any = mapping
    for key in dotted_path.split("."):
        current = current[key]
    return float(current)


def evaluate_accuracy_gates(
    metrics: Mapping[str, Any], thresholds: Mapping[str, Optional[float]]
) -> Dict[str, Any]:
    """Apply only user-specified gates; non-finite outputs always fail."""

    checks: List[Dict[str, Any]] = []
    mapping = {
        "max_policy_kl_mean": ("policy.kl.mean", "max"),
        "max_policy_kl_p99": ("policy.kl.p99", "max"),
        "max_value_kl_mean": ("value.kl.mean", "max"),
        "max_ownership_rmse": ("outputs.OutputOwnership.rmse", "max"),
        "max_scorevalue_max_abs": ("outputs.OutputScoreValue.max_abs", "max"),
        "max_score_mean_max_abs": (
            "outputs.OutputScoreValue.per_channel.score_mean.max_abs",
            "max",
        ),
        "max_score_mean_sq_max_abs": (
            "outputs.OutputScoreValue.per_channel.score_mean_sq.max_abs",
            "max",
        ),
        "max_lead_max_abs": (
            "outputs.OutputScoreValue.per_channel.lead.max_abs",
            "max",
        ),
        "max_q_value_rmse": ("policy.quantitative.q_value.rmse", "max"),
        "max_q_value_max_abs": ("policy.quantitative.q_value.max_abs", "max"),
        "max_q_score_rmse": ("policy.quantitative.q_score.rmse", "max"),
        "max_q_score_max_abs": ("policy.quantitative.q_score.max_abs", "max"),
        "min_policy_top1_agreement": ("policy.top1_agreement", "min"),
    }
    for threshold_name, (metric_path, direction) in mapping.items():
        threshold = thresholds.get(threshold_name)
        if threshold is None:
            continue
        try:
            value = _get_nested(metrics, metric_path)
        except (KeyError, TypeError):
            checks.append(
                {
                    "threshold": threshold_name,
                    "metric": metric_path,
                    "value": None,
                    "limit": float(threshold),
                    "passed": False,
                    "reason": "metric is unavailable for this model/output contract",
                }
            )
            continue
        passed = value <= threshold if direction == "max" else value >= threshold
        checks.append(
            {
                "threshold": threshold_name,
                "metric": metric_path,
                "value": value,
                "limit": float(threshold),
                "passed": bool(passed),
            }
        )

    reference_nonfinite = sum(
        int(output["reference_nonfinite"]) for output in metrics["outputs"].values()
    )
    candidate_nonfinite = sum(
        int(output["candidate_nonfinite"]) for output in metrics["outputs"].values()
    )
    checks.append(
        {
            "threshold": "reference_nonfinite",
            "metric": "sum(outputs.*.reference_nonfinite)",
            "value": reference_nonfinite,
            "limit": 0,
            "passed": reference_nonfinite == 0,
        }
    )
    checks.append(
        {
            "threshold": "candidate_nonfinite",
            "metric": "sum(outputs.*.candidate_nonfinite)",
            "value": candidate_nonfinite,
            "limit": 0,
            "passed": candidate_nonfinite == 0,
        }
    )
    configured = any(thresholds.get(name) is not None for name in mapping)
    return {
        "status": "passed" if all(check["passed"] for check in checks) else "failed",
        "numeric_thresholds_configured": configured,
        "checks": checks,
    }


def concatenate_batches(
    batches: Sequence[Mapping[str, np.ndarray]], names: Sequence[str]
) -> Dict[str, np.ndarray]:
    return {
        name: np.concatenate([np.asarray(batch[name]) for batch in batches], axis=0)
        for name in names
    }


def json_dump(path: str, value: Mapping[str, Any]) -> None:
    temporary_path = path + ".tmp"
    with open(temporary_path, "w", encoding="utf-8") as handle:
        json.dump(
            value,
            handle,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary_path, path)
