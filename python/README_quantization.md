# Experimental INT8 and FP8 ONNX export

`quantize_onnx.py` creates calibrated, explicit Q/DQ ONNX graphs for TensorRT research. It is deliberately accuracy-first: by default it quantizes only the weight projections inside KataGo transformer attention and SwiGLU blocks. Attention QK/AV matmuls, Softmax, RMSNorm, the input stem, outer bottleneck projections, trunk tip, and all policy/value heads remain FP32.

This is not a replacement for a KataGo `.bin.gz`. The current runtime reads FP32 `.bin.gz` weights and emits its own ONNX graph, so the generated Q/DQ files must initially be built and benchmarked outside KataGo with a strongly typed TensorRT network.

## 1. Dump the exact inference ONNX

Add the following to a TensorRT benchmark config:

```ini
trtDumpDebugPlanToDir = /path/to/onnx-dump
trtTransformerNHWC = true
```

Then run the normal KataGo benchmark with the exported `.bin.gz`. The backend writes files such as `plan_19x19_fp16_max.onnx`. Prefer the same exact/max-board variant that will be benchmarked; the `max` graph is useful when calibration includes masked smaller boards.

Using this backend-emitted graph avoids maintaining a second PyTorch ONNX implementation and guarantees that quantization targets the same raw five-output graph used by KataGo.

## 2. Install the pinned quantizer

```bash
python -m venv .venv-quantization
source .venv-quantization/bin/activate
# Windows PowerShell: .venv-quantization\Scripts\Activate.ps1
python -m pip install -r python/requirements-quantization.txt
```

Use a fresh Python 3.10-3.14 environment: Model Optimizer has a large PyTorch/ONNX dependency set that should not mutate the environment used for training. The tested version is NVIDIA Model Optimizer 0.45.0. The script refuses another version unless `--allow-unpinned-modelopt` is supplied, because graph rewrites and Q/DQ placement can change between versions.

## 3. Calibrate and validate

Use different shuffled-data shards for calibration and validation:

```bash
python python/quantize_onnx.py \
  --onnx-input /path/to/onnx-dump/plan_19x19_fp16_max.onnx \
  --output-dir /path/to/quantized-b15 \
  --output-prefix b15c1024h16nbt3tflrs-fson-silu \
  --calibration-data /data/calibration \
  --validation-data /data/held-out \
  --calibration-samples 2048 \
  --validation-samples 512 \
  --batch-size 32 \
  --max-source-files 64 \
  --formats int8 fp8 \
  --expected-quantized-nodes 315 \
  --validation-ep cuda:0
```

The reader samples positions from official shuffled training `.npz` files, expands the packed 22 spatial features, supplies `InputMask`, reshapes the 19 global features to NC11, and applies deterministic training-style history truncation. Since opening a compressed shard may decompress its complete arrays, the default deterministically limits each dataset to 64 weighted-random shards before sampling positions. This keeps a thousand-shard corpus fast; use `--max-source-files 0` when exact full-corpus uniform sampling is more important. The selected-shard count and paths are recorded.

It also accepts `.npz` files that already contain the exact `InputMask`, `InputSpatial`, `InputGlobal`, and optional `InputMeta` arrays. The requested history transform is applied to those inputs too; use `--history-mode full` when they are already in the exact as-stored inference state.

By default, `--symmetry-mode random` deterministically chooses one of KataGo's eight board symmetries for every selected source position. `InputMask` and `InputSpatial` receive the same transform; global and metadata inputs are unchanged. Use `--symmetry-mode all` for exhaustive coverage: `--calibration-samples 2048` still means 2048 unique source positions, but the quantizer receives 16384 effective rows (and validation expands the same way). Use `none` only when preserving stored orientation is intentional, such as with already-augmented input files. The manifest records the base/effective counts, symmetry histogram, and symmetry hash.

Defaults are intentionally conservative:

- signed symmetric INT8 or FP8 E4M3 Q/DQ through Model Optimizer;
- quantized activations per tensor and weights using Model Optimizer's TensorRT-oriented handling;
- FP32 model inputs, outputs, and fallback operations;
- entropy calibration for INT8 and max calibration for FP8 over real positions (Model Optimizer 0.45's FP8 conversion requires max-calibrated scales);
- seeded random D4 symmetry augmentation matching KataGo's training-style input distribution;
- no latency-only autotuning;
- held-out primary-policy KL/top-move agreement, value KL, raw Q-value/Q-score drift, per-channel score-value error, masked and unmasked ownership error, per-output p99/max/RMSE/relative-L2, and non-finite checks;
- a semantic Q/DQ audit covering Q-to-DQ chains, data types, scale/zero-point values, axes, channel counts, and unexpected quantization;
- a JSON manifest with immutable source/output hashes, sampled-position hashes, selected node names, package/GPU versions, actual ONNX Runtime providers, and validation results.

Model Optimizer works only on a complete staging copy and writes each result as a staged artifact before promotion. This prevents its shape inference from changing the dumped source ONNX and prevents repeated `--overwrite` runs from appending another copy of a large external weight sidecar.

Keep `--high-precision fp32` for the first experiments. Model Optimizer's FP16 option converts the entire non-quantized fallback graph, whereas KataGo's current TensorRT backend selectively keeps norms, trunk tip, and heads in FP32. The script therefore rejects global FP16 fallback unless `--allow-global-fp16-fallback` is also supplied, and any such variant must be evaluated as a separate mixed-precision experiment rather than attributed solely to INT8/FP8.

Numerical release limits are model- and experiment-dependent, so the script does not invent them. Add explicit gates after an FP32/FP16 baseline is established, for example:

```bash
  --max-policy-kl-mean ... \
  --max-policy-kl-p99 ... \
  --max-value-kl-mean ... \
  --max-ownership-rmse ... \
  --max-score-mean-max-abs ... \
  --max-score-mean-sq-max-abs ... \
  --max-lead-max-abs ... \
  --max-q-value-rmse ... \
  --max-q-score-rmse ... \
  --min-policy-top1-agreement ...
```

Without explicit limits, the report is descriptive and must not be treated as release qualification. A separate held-out corpus, TensorRT build, throughput tests at production batch sizes, exhaustive symmetry validation (`--symmetry-mode all` or an equivalent independent check), `testgpuerror`, and self-play are still required.

## TensorRT validation

Pass a TensorRT 10.16 `trtexec` executable to add a parser/build check:

```bash
  --trtexec /path/to/trtexec --trt-opt-batch 32 --trt-max-batch 64
```

The script uses `--stronglyTyped` and does not add `--fp16`, `--int8`, or `--fp8`; precision is encoded by the Q/DQ graph. It saves the engine plus detailed TensorRT layer information for precision inspection. Build success is only a parser/builder check: verify selected layer precision from that report and eventually run held-out inputs through TensorRT before making accuracy or speed claims.

ONNX Runtime numerical comparison disables graph optimization for both reference and candidate, avoiding its FP8 rewrite bug. It also verifies that the requested CUDA/TRT provider actually survived session creation, rather than silently reporting a full CPU fallback. Individual unsupported nodes can still fall back and are called out in the manifest.

INT8 can be researched on Ampere and newer. Hardware FP8 acceleration requires Ada or newer, so RTX 4090/5090 are suitable and RTX 3090 is not.

## Background

The initial KataGo-specific comparison was zml24's [`nano/quantize_int8.py`](https://github.com/zml24/KataGo_Transformer/blob/main/nano/quantize_int8.py). This implementation keeps its useful static-Q/DQ and real-position calibration ideas, but uses the current official KataGo input/output contract, deterministic disjoint data, an effective projection allowlist, NVIDIA Model Optimizer for both INT8 and FP8, and multi-output accuracy gates.

For a direct coverage comparison, `--scope all-weighted` implements the
reference script's actual executable allowlist: on the KataGo-emitted graph,
every `MatMul` or `Conv` with a constant weight is eligible, while
activation-by-activation attention MatMuls (`QK^T` and attention-by-`V`)
remain unquantized. Weighted `Gemm` is supported too if a graph contains it.
This includes the stem, outer bottleneck projections, and policy/value heads.
The reference file declares additional stem/head skip patterns, but does not
pass those patterns to its quantization call. Use `--scope transformer` for
the narrower accuracy-first 315-projection b15 scope; use `all-weighted` for
the aggressive 358-node b15 comparison and validate all five raw heads before
promotion.

TensorRT explicit-quantization semantics and supported Q/DQ patterns are documented in NVIDIA's [Working with Quantized Types](https://docs.nvidia.com/deeplearning/tensorrt/10.x.x/inference-library/work-quantized-types.html), and Model Optimizer provides the maintained [ONNX PTQ implementation](https://github.com/NVIDIA/Model-Optimizer/tree/main/examples/onnx_ptq).
