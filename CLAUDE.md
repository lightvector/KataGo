# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

KataGo is a strong open-source Go/Baduk AI engine. It uses AlphaZero-style self-play training with many enhancements, supports MCTS (Monte-Carlo Tree Search) with graph search, and provides both GTP and JSON analysis interfaces.

This fork (`feature/onnx-coreml-backend`) explores ONNX and CoreML backend support for Apple devices.

## Build Commands

All C++ builds happen from the `cpp/` directory. CMake 3.18.2+ and C++17 required.

### macOS with Metal backend (Apple Silicon)
```bash
cd cpp
cmake . -G Ninja -DUSE_BACKEND=METAL
ninja
```
Requires: Ninja generator, Swift 5.9+, AppleClang, macOS 13.0+

### macOS with OpenCL backend
```bash
cd cpp
cmake . -G Ninja -DUSE_BACKEND=OPENCL
ninja
```

### Eigen (CPU-only) backend
```bash
cd cpp
cmake . -DUSE_BACKEND=EIGEN -DUSE_AVX2=1
make -j$(nproc)
```

### ONNX Runtime backend (cross-platform)
```bash
cd cpp
cmake . -G Ninja -DUSE_BACKEND=ONNX \
  -DONNXRUNTIME_ROOT=/path/to/onnxruntime \
  -DONNXRUNTIME_BUILD_DIR=/path/to/onnxruntime/build/MacOS/RelWithDebInfo \
  -DNO_GIT_REVISION=1
ninja
```
Requires: ONNX Runtime v1.25.0. Loads `.bin.gz` model files (builds ONNX graph dynamically from ModelDesc).

Config key `onnxProvider` selects the execution provider:
- `cpu` (default) — CPU execution provider, works everywhere
- `coreml` — CoreML execution provider (macOS only, Apple Silicon)

### Other backends
- CUDA: `-DUSE_BACKEND=CUDA` (requires CUDA 11+, CUDNN)
- TensorRT: `-DUSE_BACKEND=TENSORRT` (requires TensorRT 8.5+)
- Dummy (no neural net, for testing non-NN code paths): omit `-DUSE_BACKEND`

### Additional CMake flags
- `-DBUILD_DISTRIBUTED=1` — distributed training support (requires OpenSSL, libzip, git repo)
- `-DNO_GIT_REVISION=1` — skip embedding git hash
- `-DUSE_BIGGER_BOARDS_EXPENSIVE=1` — allow boards up to 50x50 (slower, more memory)

## Running Tests

From the `cpp/` directory after building:

```bash
# Unit tests (board algorithms, rules, SGF, datastructures) — no GPU/neural net needed
./katago runtests

# Output tests (deterministic, no neural net)
./katago runoutputtests

# NN layer tests (requires built backend)
./katago runnnlayertests

# Full search tests with neural net (downloads models, slow)
bash runsearchtests.sh

# GTP/analysis/command integration tests (requires models)
bash runcmdtests.sh
```

Test models live in `cpp/tests/models/`. The search/cmd test scripts use deterministic seeds (`nnRandSeed=forTesting`, `searchRandSeed=forTesting`, `forDeterministicTesting=true`).

## CI

GitHub Actions (`.github/workflows/build.yml`) builds and runs `./katago runtests` on Linux (OpenCL), macOS (OpenCL+Ninja), and Windows (OpenCL+MSVC).

## Architecture

### C++ Source (`cpp/`)

Dependency order from lowest to highest:

- **`core/`** — Standard library extensions: hashing, RNG, string utils, filesystem, threading, logging
- **`game/`** — Board (`board.h`), game state with history/ko/scoring (`boardhistory.h`), rules for 10+ rulesets (`rules.h`), graph hash for MCGS (`graphhash.h`)
- **`neuralnet/`** — Neural net abstraction layer
  - `nninterface.h` — Backend-agnostic interface (key types: `LoadedModel`, `ComputeContext`, `ComputeHandle`, `InputBuffers`)
  - `desc.h` — Model descriptor (layer structure, weights). All backends deserialize from this
  - `nninputs.h` — Input feature computation (board state → tensor)
  - `nneval.h` — Thread-safe batched evaluation used by search
  - Backend implementations: `cudabackend.cpp`, `trtbackend.cpp`, `openclbackend.cpp`, `eigenbackend.cpp`, `metalbackend.{cpp,swift}`, `onnxbackend.cpp`, `dummybackend.cpp`
- **`search/`** — Multithreaded MCTS with graph search, symmetry pruning, pondering, time control
  - `search.h` — Main search class
  - `asyncbot.h` — Thread-safe wrapper with pondering support
  - `searchparams.h` — All tunable search parameters
- **`dataio/`** — SGF I/O (`sgf.h`), model loading (`loadmodel.h`), training data writing
- **`program/`** — Config parsing, NN setup, match/selfplay orchestration
- **`command/`** — Top-level subcommands: `gtp.cpp`, `analysis.cpp`, `benchmark.cpp`, `selfplay.cpp`, `match.cpp`, etc.
- **`main.cpp`** — Entry point, dispatches to subcommands. Version string is hardcoded here.

### Adding a new neural net backend

Implement the functions declared in `neuralnet/nninterface.h` (roughly: `globalInitialize`, `loadModelFile`, `createComputeContext`, `createComputeHandle`, `getOutput`, `freeXxx`). Add the backend source files to `CMakeLists.txt` under a new `USE_BACKEND` option. The `desc.h` model descriptor is backend-agnostic — your backend reads weights from it.

### Python Source (`python/`)

Training pipeline (not needed for inference):
- `train.py` — Main training script (PyTorch)
- `shuffle.py` — Data shuffling for training
- `export_model.py` — Export PyTorch checkpoints to KataGo `.bin.gz` format
- `katago/train/model_pytorch.py` — Neural net architecture definition
- `play.py` — Minimal GTP engine using raw PyTorch model (useful reference for direct inference)

## Key Subcommands

```
katago gtp          # GTP engine for GUIs
katago analysis     # JSON batch analysis engine
katago benchmark    # Performance testing
katago match        # Self-play matches with batching
katago selfplay     # Generate training data
katago contribute   # Distributed training client
katago genconfig    # Interactive config generator
```

## Model Formats

- `.bin.gz` — Binary format (current standard)
- `.txt.gz` — Text format (older, still supported for tests)
- `.onnx` — ONNX format (for ONNX backend, loaded directly via ONNX Runtime)
- Models are loaded via `dataio/loadmodel.h`, deserialized into `neuralnet/desc.h` structures

## Rules Support

KataGo supports Chinese, Japanese, Korean, AGA, New Zealand, Stone Scoring, and more — configured via the `Rules` struct in `game/rules.h`. Board sizes 7-19 standard (up to 50 with compile flag).
