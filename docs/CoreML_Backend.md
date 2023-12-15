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
wget https://github.com/ChinChangYang/KataGo/archive/refs/tags/v1.13.2-coreml2.tar.gz
tar -zxvf v1.13.2-coreml2.tar.gz
```
This command retrieves the `v1.13.2-coreml2` source code version and decompresses the tarball into the `KataGo-1.13.2-coreml2` directory.

## Preparing the Workspace
Transition into the workspace directory where the KataGo models and executable will be built:
```
cd KataGo-1.13.2-coreml2
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
