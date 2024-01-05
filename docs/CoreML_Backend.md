# Documentation for Metal and CoreML Backends in KataGo
KataGo harnesses the advanced capabilities of Apple Silicon through the integration of the [Metal Performance Shaders Graph](https://developer.apple.com/documentation/metalperformanceshadersgraph) and [CoreML](https://developer.apple.com/documentation/coreml). This integration empowers KataGo with GPU acceleration and compatibility with the [Neural Engine](https://machinelearning.apple.com/research/neural-engine-transformers), ensuring exceptional performance levels.

## Essential Software Installation
Before proceeding, ensure that the indispensable build tool, [Ninja](https://ninja-build.org) is installed. Execute the following command to install Ninja:
```
brew install ninja
```
This command installs [Ninja](https://ninja-build.org) onto your system.

## Source Code Acquisition
For the creation of a KataGo executable and corresponding CoreML models, initiate by downloading the source code. Build KataGo equipped with the Metal and CoreML backends by executing:
```
wget https://github.com/ChinChangYang/KataGo/archive/metal-coreml-stable.tar.gz
tar -zxvf metal-coreml-stable.tar.gz
```
This command retrieves the `metal-coreml-stable` source code version and decompresses the tarball into the `KataGo-metal-coreml-stable` directory.

## Preparing the Workspace
Transition into the workspace directory where the KataGo models and executable will be built:
```
cd KataGo-metal-coreml-stable
```

## Compiling KataGo
Utilize [CMake](https://cmake.org) in conjunction with [Ninja](https://ninja-build.org) for compiling KataGo with the Metal and CoreML backends:
```
cd cpp
mv CMakeLists.txt-macos CMakeLists.txt
mkdir -p build
cd build
cmake -G Ninja -DNO_GIT_REVISION=1 -DCMAKE_BUILD_TYPE=Release ../
ninja
```
Executing these commands compiles KataGo in the `cpp/build` directory.

## Download the KataGo model
Acquire the KataGo model in binary format suitable for the Metal backend:
```
wget https://github.com/ChinChangYang/KataGo/releases/download/v1.13.2-coreml2/kata1-b18c384nbt-s8341979392-d3881113763.bin.gz
wget https://github.com/ChinChangYang/KataGo/releases/download/v1.13.2-coreml2/KataGoModel19x19fp16v14s8341979392.mlpackage.zip
unzip KataGoModel19x19fp16v14s8341979392.mlpackage.zip
```

## Organizing Binary and CoreML Model
Optionally, relocate the binary model to the run directory. However, it is essential to link the CoreML model in the run directory to ensure its accessibility by the CoreML backend:
```
ln -s KataGoModel19x19fp16v14s8341979392.mlpackage KataGoModel19x19fp16.mlpackage
```

## Utilization of KataGo
KataGo can be operated in several modes, thanks to its extensive command options. Here are three primary use cases:

**Benchmark**

To conduct a benchmark, use the `benchmark` command, specify the binary model location, and apply the `coreml_example.cfg` configuration:
```
./katago benchmark -model kata1-b18c384nbt-s8341979392-d3881113763.bin.gz -config ../configs/misc/coreml_example.cfg -t 32 -v 1600
```
This command activates the benchmark mode utilizing both Metal and CoreML backends.

**GTP**

For running the GTP protocol, utilize the `gtp` command, specify the binary model location, and use the `coreml_example.cfg` configuration:
```
./katago gtp -model kata1-b18c384nbt-s8341979392-d3881113763.bin.gz -config ../configs/misc/coreml_example.cfg
```
This enables the GTP protocol leveraging Metal and CoreML backends.

**Analysis**

Activate the analysis engine with the `analysis` command, specify the binary model location, and use the `coreml_analysis.cfg` configuration:
```
./katago analysis -model kata1-b18c384nbt-s8341979392-d3881113763.bin.gz -config ../configs/misc/coreml_analysis.cfg
```
This initiates the analysis mode, taking advantage of both Metal and CoreML backends.

## Updating the CoreML model

### Prerequisite Software Installation

Before initiating the update process, it is crucial to install the required software. Start by installing `miniconda`, then create and activate a Python environment specifically for `coremltools`. Follow these commands:

```
brew install miniconda
conda create -n coremltools python=3.8
conda activate coremltools
pip install coremltools torch
```

This sequence first installs `miniconda`. Subsequently, a dedicated environment named `coremltools` is created using Python version 3.8. Finally, within this environment, `coremltools` and `torch` are installed, setting the stage for the model update process.

### Downloading the Checkpoint File

The next step involves acquiring the latest and most robust network checkpoint from the KataGo Networks. Navigate to [KataGo Networks](https://katagotraining.org/networks/) and select the strongest confidently-rated network available. For instance, if `kata1-b18c384nbt-s8526915840-d3929217702` is the latest, download the corresponding `.zip` file, such as `kata1-b18c384nbt-s8526915840-d3929217702.zip`. Upon downloading, unzip the file to access the `model.ckpt` checkpoint file.

### Converting the Checkpoint File

**To Binary Model**

Utilize the `export_model_pytorch.py` script to transform the checkpoint file into a binary model compatible with the Metal backend:

```
python python/export_model_pytorch.py -checkpoint model.ckpt -export-dir model -model-name model -filename-prefix model -use-swa
gzip model/model.bin
```

Executing this command sequence generates a compressed binary model file named `model.bin.gz`.

**To CoreML Model**

Similarly, for converting the checkpoint file into a CoreML model, the `convert_coreml_pytorch.py` script is employed:

```
python python/convert_coreml_pytorch.py -checkpoint model.ckpt -use-swa
```

This script outputs the CoreML model directory `KataGoModel19x19fp16.mlpackage`, specifically tailored for the CoreML backend.

However, it's important to note a specific scenario: If KataGo has been compiled with the option `COMPILE_MAX_BOARD_LEN=29` to support larger 29x29 board sizes, the CoreML model conversion requires an additional parameter. In such cases, include the `-pos-len 29` option in the script command to ensure compatibility with the larger board size. The command modifies as follows:

```
python python/convert_coreml_pytorch.py -checkpoint model.ckpt -use-swa -pos-len 29
```

This adjustment in the command results in the creation of a distinct CoreML model directory, `KataGoModel29x29fp16.mlpackage`, specifically tailored for KataGo versions supporting board sizes up to 29x29.

### Reorganizing the Models

Post-conversion, it is advisable to reorganize the models for optimal accessibility. While relocating the binary model to the run directory is optional, linking the CoreML model within this directory is essential for its effective utilization by the CoreML backend.
