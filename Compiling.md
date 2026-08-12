
# Compiling KataGo
KataGo is written in C++. It should compile on Linux or OSX via g++ that supports at least C++14, or on Windows via MSVC 15 (2017) and later or MinGW. Other compilers and systems have not been tested yet. This is recommended if you want to run the full KataGo self-play training loop on your own and/or do your own research and experimentation, or if you want to run KataGo on an operating system for which there is no precompiled executable available.

### Building for Distributed
As also mentioned in the instructions below but repeated here for visibility, if you also are building KataGo with the intent to use it in distributed training on https://katagotraining.org, then keep in mind:
* You'll need to specify `-DBUILD_DISTRIBUTED=1` or `BUILD_DISTRIBUTED` and have OpenSSL installed.
* Building will need to happen within a Git clone of the KataGo repo, rather than a zipped copy of the source (such as what you might download from a packaged release).
* The version will need to be supported for distributed training. **The `master` branch will NOT work** - instead please use the either latest release tag or the tip of the `stable` branch, these should both work.
* Please do NOT attempt to bypass any versioning or safety checks - if you feel you need to do so, please first reach out by opening an issue or messaging in [discord](https://discord.gg/bqkZAz3). There is an alternate site [test.katagodistributed.org](test.katagodistributed.org) you can use if you are working on KataGo development or want to test things more freely, ask in the KataGo channel of discord to set up a test account.

## Linux
   * TLDR (if you have a working GPU):
     ```
     git clone https://github.com/lightvector/KataGo.git
     cd KataGo/cpp
     # If you get missing library errors, install the appropriate packages using your system package manager and try again.
     # -DBUILD_DISTRIBUTED=1 is only needed if you want to contribute back to public training.
     cmake . -DUSE_BACKEND=OPENCL -DBUILD_DISTRIBUTED=1
     make -j 4
     ```
   * TLDR (building the slow pure-CPU version):
     ```
     git clone https://github.com/lightvector/KataGo.git
     cd KataGo/cpp
     # If you get missing library errors, install the appropriate packages using your system package manager and try again.
     cmake . -DUSE_BACKEND=EIGEN -DUSE_AVX2=1
     make -j 4
     ```
   * Requirements
      * CMake with a minimum version of 3.18.2 - for example `sudo apt install cmake` on Debian, or download from https://cmake.org/download/ if that doesn't give you a recent-enough version.
      * Some version of g++ that supports at least C++14.
      * If using the OpenCL backend, a modern GPU that supports OpenCL 1.2 or greater, or else something like [this](https://software.intel.com/en-us/opencl-sdk) for CPU. But if using CPU, Eigen should be better.
      * If using the CUDA backend, CUDA 11 or later and a compatible version of CUDNN based on your CUDA version (https://developer.nvidia.com/cuda-toolkit) (https://developer.nvidia.com/cudnn) and a GPU capable of supporting them.
      * If using the TensorRT backend, in addition to a compatible CUDA Toolkit (https://developer.nvidia.com/cuda-toolkit), you also need TensorRT (https://developer.nvidia.com/tensorrt) that is at least version 8.5.
      * If using the Eigen backend, Eigen3. With Debian packages, (i.e. apt or apt-get), this should be `libeigen3-dev`.
      * If using the ONNX backend, ONNX Runtime headers/libs and ONNX protobuf dependencies (`onnx/onnx-ml.pb.h`, `onnx_proto`, `protobuf-lite`) for `.bin.gz` model conversion support.
      * zlib, libzip. With Debian packages (i.e. apt or apt-get), these should be `zlib1g-dev`, `libzip-dev`.
      * If you want to do self-play training and research, probably Google perftools `libgoogle-perftools-dev` for TCMalloc or some other better malloc implementation. For unknown reasons, the allocation pattern in self-play with large numbers of threads and parallel games causes a lot of memory fragmentation under glibc malloc that will eventually run your machine out of memory, but better mallocs handle it fine.
      * If compiling to contribute to public distributed training runs, OpenSSL is required (`libssl-dev`).
   * Clone this repo:
      * `git clone https://github.com/lightvector/KataGo.git`
   * Compile using CMake and make in the cpp directory:
      * `cd KataGo/cpp`
      * `cmake . -DUSE_BACKEND=OPENCL` or `cmake . -DUSE_BACKEND=CUDA` or `cmake . -DUSE_BACKEND=TENSORRT` or `cmake . -DUSE_BACKEND=EIGEN` or `cmake . -DUSE_BACKEND=ONNX` depending on which backend you want.
         * Specify also `-DUSE_TCMALLOC=1` if using TCMalloc.
         * Compiling will also call git commands to embed the git hash into the compiled executable, specify also `-DNO_GIT_REVISION=1` to disable it if this is causing issues for you.
         * Specify `-DUSE_AVX2=1` to also compile Eigen with AVX2 and FMA support, which will make it incompatible with old CPUs but much faster. (If you want to go further, you can also add `-DCMAKE_CXX_FLAGS='-march=native'` which will specialize to precisely your machine's CPU, but the exe might not run on other machines at all).
         * Specify `-DBUILD_DISTRIBUTED=1` to compile with support for contributing data to public distributed training runs.
            * If building distributed, you will also need to build with Git revision support, including building within a clone of the repo, as opposed to merely an unzipped copy of its source.
            * Only builds from specific tagged versions or branches can contribute, in particular, instead of the `master` branch, use either the latest [release](https://github.com/lightvector/KataGo/releases) tag or the tip of the `stable` branch. To minimize the chance of any data incompatibilities or bugs, please do NOT attempt to contribute with custom changes or circumvent these limitations.
      * `make`
   * Done! You should now have a compiled `katago` executable in your working directory.
   * Pre-trained neural nets are available at [the main training website](https://katagotraining.org/).
   * You will probably want to edit `configs/gtp_example.cfg` (see "Tuning for Performance" above).
   * If using OpenCL, you will want to verify that KataGo is picking up the correct device when you run it (e.g. some systems may have both an Intel CPU OpenCL and GPU OpenCL, if KataGo appears to pick the wrong one, you can correct this by specifying `openclGpuToUse` in `configs/gtp_example.cfg`).

##### ONNX Runtime Backend (Linux)
The ONNX backend uses ONNX Runtime for inference, and supports both:
* `.onnx` models loaded directly.
* `.bin.gz` KataGo models via internal conversion to ONNX graph (requires ONNX protobuf dependencies in CMake).

##### Linux Intel NPU (OpenVINO EP) Setup
1. Install Intel NPU driver on Linux:
   * https://github.com/intel/linux-npu-driver
2. Install OpenVINO via system package manager (APT example):
   * https://docs.openvino.ai/2025/get-started/install-openvino/install-openvino-apt.html
3. Build ONNX Runtime with OpenVINO EP for NPU (same ORT flow as Windows):
   * https://onnxruntime.ai/docs/build/eps.html#openvino
   * Set OpenVINO EP build option so `use_openvino` is `NPU` (for example `--use_openvino NPU` in ORT build.py).

##### Prepare `ONNXRUNTIME_ROOT` in KataGo (Linux)
Use package root:
* `cpp/external/onnxruntime-linux-x64-openvino`

Linux one-to-one mapping (`<ORT_PACKAGE_ROOT>` -> KataGo):

Include files:

| Source (`<ORT_PACKAGE_ROOT>`) | Destination (KataGo) |
| --- | --- |
| `include/core/*` | `cpp/external/onnxruntime-linux-x64-openvino/include/core/` |
| `include/cpu_provider_factory.h` | `cpp/external/onnxruntime-linux-x64-openvino/include/cpu_provider_factory.h` |
| `include/provider_options.h` | `cpp/external/onnxruntime-linux-x64-openvino/include/provider_options.h` |
| `include/onnxruntime_c_api.h` | `cpp/external/onnxruntime-linux-x64-openvino/include/onnxruntime_c_api.h` |
| `include/onnxruntime_cxx_api.h` | `cpp/external/onnxruntime-linux-x64-openvino/include/onnxruntime_cxx_api.h` |
| `include/onnxruntime_cxx_inline.h` | `cpp/external/onnxruntime-linux-x64-openvino/include/onnxruntime_cxx_inline.h` |
| `include/onnxruntime_env_config_keys.h` | `cpp/external/onnxruntime-linux-x64-openvino/include/onnxruntime_env_config_keys.h` |
| `include/onnxruntime_ep_c_api.h` | `cpp/external/onnxruntime-linux-x64-openvino/include/onnxruntime_ep_c_api.h` |
| `include/onnxruntime_ep_device_ep_metadata_keys.h` | `cpp/external/onnxruntime-linux-x64-openvino/include/onnxruntime_ep_device_ep_metadata_keys.h` |
| `include/onnxruntime_float16.h` | `cpp/external/onnxruntime-linux-x64-openvino/include/onnxruntime_float16.h` |
| `include/onnxruntime_lite_custom_op.h` | `cpp/external/onnxruntime-linux-x64-openvino/include/onnxruntime_lite_custom_op.h` |
| `include/onnxruntime_run_options_config_keys.h` | `cpp/external/onnxruntime-linux-x64-openvino/include/onnxruntime_run_options_config_keys.h` |
| `include/onnxruntime_session_options_config_keys.h` | `cpp/external/onnxruntime-linux-x64-openvino/include/onnxruntime_session_options_config_keys.h` |

Library/config/pkgconfig files:

| Source (`<ORT_PACKAGE_ROOT>`) | Destination (KataGo) |
| --- | --- |
| `lib/libonnxruntime_providers_openvino.so` | `cpp/external/onnxruntime-linux-x64-openvino/lib/libonnxruntime_providers_openvino.so` |
| `lib/libonnxruntime_providers_shared.so` | `cpp/external/onnxruntime-linux-x64-openvino/lib/libonnxruntime_providers_shared.so` |
| `lib/libonnxruntime.so.1.24.3` | `cpp/external/onnxruntime-linux-x64-openvino/lib/libonnxruntime.so.1.24.3` |
| `lib/libonnxruntime.so.1` (symlink to `.1.24.3`) | `cpp/external/onnxruntime-linux-x64-openvino/lib/libonnxruntime.so.1` |
| `lib/libonnxruntime.so` (symlink to `.1`) | `cpp/external/onnxruntime-linux-x64-openvino/lib/libonnxruntime.so` |
| `lib/cmake/onnxruntime/onnxruntimeConfig.cmake` | `cpp/external/onnxruntime-linux-x64-openvino/lib/cmake/onnxruntime/onnxruntimeConfig.cmake` |
| `lib/cmake/onnxruntime/onnxruntimeConfigVersion.cmake` | `cpp/external/onnxruntime-linux-x64-openvino/lib/cmake/onnxruntime/onnxruntimeConfigVersion.cmake` |
| `lib/cmake/onnxruntime/onnxruntimeTargets.cmake` | `cpp/external/onnxruntime-linux-x64-openvino/lib/cmake/onnxruntime/onnxruntimeTargets.cmake` |
| `lib/cmake/onnxruntime/onnxruntimeTargets-release.cmake` | `cpp/external/onnxruntime-linux-x64-openvino/lib/cmake/onnxruntime/onnxruntimeTargets-release.cmake` |
| `lib/pkgconfig/libonnxruntime.pc` | `cpp/external/onnxruntime-linux-x64-openvino/lib/pkgconfig/libonnxruntime.pc` |

##### Minimal KataGo Build Commands (Linux, ONNX backend)
On Linux, `KATAGO_AUTO_FETCH_DEPS=ON` can auto-fetch missing `zlib`, `onnx`, and `protobuf` dependencies via vcpkg into `cpp/build/deps/vcpkg`.

```bash
cmake -S cpp -B cpp/build -G Ninja -DUSE_BACKEND=ONNX -DONNXRUNTIME_ROOT=cpp/external/onnxruntime-linux-x64-openvino
cmake --build cpp/build -j
```

If you want to disable auto-fetch and provide dependencies manually:
* `-DKATAGO_AUTO_FETCH_DEPS=OFF`
* plus `-DONNX_INCLUDE_DIR=... -DONNX_PROTO_LIB=... -DPROTOBUF_INCLUDE_DIR=... -DPROTOBUF_LIB=... -DZLIB_INCLUDE_DIR=... -DZLIB_LIBRARY=...`

Typical run config for Intel NPU:
* `onnxProvider = openvino`
* `onnxOpenVINODeviceType = NPU`
* `onnxOpenVINOEnableNPUFastCompile = true` (optional; may be ignored on ORT builds that do not support this key)

Multi-device assignment is mainly for `onnxProvider=cuda/tensorrt/migraphx` (`onnxDeviceToUseThread*`).
For `onnxProvider=openvino` on Intel NPU, a single device is typically used.


## Windows
   * TLDR:
      * Building from source on Windows is actually a bit tricky, depending on what version you're building, there's not necessarily a super-fast way.
   * Requirements
      * CMake with a minimum version of 3.18.2, GUI version strongly recommended (https://cmake.org/download/)
      * Microsoft Visual Studio for C++. Version 15 (2017) has been tested and should work, MinGW version also should work but only with Eigen and OpenCL backends (CUDA and TensorRT MinGW backends are [not supported by NVIDIA](https://forums.developer.nvidia.com/t/cuda-with-mingw-how-to-get-cuda-running-under-mingw)).
      * If using the OpenCL backend, a modern GPU that supports OpenCL 1.2 or greater, or else something like [this](https://software.intel.com/en-us/opencl-sdk) for CPU. But if using CPU, Eigen should be better.
      * If using the CUDA backend, CUDA 11 or later and a compatible version of CUDNN based on your CUDA version (https://developer.nvidia.com/cuda-toolkit) (https://developer.nvidia.com/cudnn) and a GPU capable of supporting them. I'm unsure how version compatibility works with CUDA, there's a good chance that later versions than these work just as well, but they have not been tested.
      * If using the TensorRT backend, in addition to a compatible CUDA Toolkit (https://developer.nvidia.com/cuda-toolkit), you also need TensorRT (https://developer.nvidia.com/tensorrt) that is at least version 8.5.
      * If using the Eigen backend, Eigen3, version 3.3.x. (http://eigen.tuxfamily.org/index.php?title=Main_Page#Download).
      * If using the ONNX backend, ONNX Runtime package (headers + import libs + runtime DLLs).
      * On Windows, missing `zlib` and ONNX model-conversion dependencies (`onnx`, `protobuf`) can be auto-fetched by CMake into `cpp/build/deps/vcpkg` (default `KATAGO_AUTO_FETCH_DEPS=ON`).
      * libzip (optional, needed only for self-play training) - for example https://github.com/kiyolee/libzip-win-build
      * For MinGW it's recommended to use [MSYS2](https://www.msys2.org/) building platform to get necessary zlib and libzip dependencies:
        * Install MSYS2 according to the instruction on the official site
        * Run `mingw64.exe` app from Console
        * Install zlib/libzip dependencies using pacman package manager:
          * `pacman -S mingw-w64-x86_64-libzip`
          * `pacman -S mingw-w64-x86_64-xz`
          * `pacman -S mingw-w64-x86_64-bzip2`
          * `pacman -S mingw-w64-x86_64-zstd`
      * If compiling to contribute to public distributed training runs, OpenSSL is required (https://www.openssl.org/, https://wiki.openssl.org/index.php/Compilation_and_Installation).
   * Download/clone this repo to some folder `KataGo`.
   * Configure using CMake GUI and compile in an IDE:
      * Select `KataGo/cpp` as the source code directory in [CMake GUI](https://cmake.org/runningcmake/).
      * Set the build directory to wherever you would like the built executable to be produced.
      * Click "Configure". For the generator select your generator (MSVC or MinGW), and also select "x64" for the optional platform if you're on 64-bit windows, don't use win32.
      * If you get errors where CMake has not automatically found ZLib, point it to the appropriate places according to the error messages:
        * `ZLIB_INCLUDE_DIR` - point this to the directory containing `zlib.h` and other headers
        * `ZLIB_LIBRARY` - point this to the `libz.lib` (`libz.a` for MinGW) resulting from building zlib. Note that "*_LIBRARY" expects to be pointed to the ".lib" file, whereas the ".dll" file is the file that needs to be included with KataGo at runtime.
        * For MinGW zlib/libzip CMake options should look like the following way:
          ```
          -DZLIB_INCLUDE_DIR="C:/msys64/mingw64/include"
          -DZLIB_LIBRARY="C:/msys64/mingw64/lib/libz.a"
          -DLIBZIP_INCLUDE_DIR_ZIP:PATH="C:/msys64/mingw64/include"
          -DLIBZIP_INCLUDE_DIR_ZIPCONF:PATH="C:/msys64/mingw64/include"
          -DLIBZIP_LIBRARY:FILEPATH="C:/msys64/mingw64/lib/libzip.dll.a"
          ```
      * Also set `USE_BACKEND` to `OPENCL`, or `CUDA`, or `TENSORRT`, or `EIGEN`, or `ONNX` depending on what backend you want to use.
      * Set any other options you want and re-run "Configure" again as needed after setting them. Such as:
         * `NO_GIT_REVISION` if you don't have Git or if cmake is not finding it.
         * `NO_LIBZIP` if you don't care about running self-play training and you don't have libzip.
         * `USE_AVX2` if you want to compile with AVX2 and FMA instructions, which will fail on some CPUs but speed up Eigen greatly on CPUs that support them.
         * `BUILD_DISTRIBUTED` to compile with support for contributing data to public distributed training runs.
            * If building distributed, you will also need to build with Git revision support, including building within a clone of the repo, as opposed to merely an unzipped copy of its source.
            * Only builds from specific tagged versions or branches can contribute, in particular, instead of the `master` branch, use either the latest [release](https://github.com/lightvector/KataGo/releases) tag or the tip of the `stable` branch. To minimize the chance of any data incompatibilities or bugs, please do NOT attempt to contribute with custom changes or circumvent these limitations.
      * Once running "Configure" looks good, run "Generate" and then open the project in Visual Studio or CLion and build it as usual.
   * For MinGW it's recommended to configure the project in the following ways:
     * Use the default MinGW toolchain in [CLion IDE](https://www.jetbrains.com/clion/) (free for Non-Commercial use)
     * Use [MSYS2](https://www.msys2.org/) MinGW toolchain. Befor configuring, install gcc compiler using pacman package manager: `pacman -S mingw-w64-x86_64-gcc`
   * Done! You should now have a compiled `katago.exe` executable in your working directory.
   * Note: You may need to copy the ".dll" files corresponding to the various ".lib" (".a") files you compiled with into the directory containing katago.exe.
     * MinGW has different dlls. If you use pacman, the necessary dlls (`libbz2-1.dll`, `libzip.dll`, `libzstd.dll`, `liblzma-5.dll`) should be copied from MinGW bin directory (like `C:\msys64\mingw64\bin`).
   * Note: If you had to update or install CUDA or GPU drivers, you will likely need to reboot before they will work.
   * Pre-trained neural nets are available at [the main training website](https://katagotraining.org/).
   * You will probably want to edit `configs/gtp_example.cfg` (see "Tuning for Performance" above).
   * If using OpenCL, you will want to verify that KataGo is picking up the correct device (e.g. some systems may have both an Intel CPU OpenCL and GPU OpenCL, if KataGo appears to pick the wrong one, you can correct this by specifying `openclGpuToUse` in `configs/gtp_example.cfg`).

##### ONNX Runtime Backend
The ONNX backend uses ONNX Runtime for inference, and supports both:
* `.onnx` models loaded directly.
* `.bin.gz` KataGo models via internal conversion to ONNX graph (requires ONNX protobuf dependencies in CMake).

##### Windows Intel NPU (OpenVINO EP) Setup
1. Install Visual Studio 2026 Community or Visual Studio 2026 Build Tools:
   * https://visualstudio.microsoft.com/zh-hans/downloads/
   * In installer workloads, select **Desktop development with C++**.
2. Install Intel NPU driver:
   * https://www.intel.com/content/www/us/en/download/794734/intel-npu-driver-windows.html
3. Install OpenVINO 2026 archive package on Windows:
   * https://docs.openvino.ai/2026/get-started/install-openvino/install-openvino-archive-windows.html
   * Typical install root looks like: `C:\Program Files (x86)\Intel\openvino_2026.0`
4. Add these to System PATH:
   * `C:\Program Files (x86)\Intel\openvino_2026.0\runtime\bin\intel64\Release`
   * `C:\Program Files (x86)\Intel\openvino_2026.0\runtime\3rdparty\tbb\bin`
5. Build ONNX Runtime with OpenVINO EP for NPU (follow official docs):
   * https://onnxruntime.ai/docs/build/eps.html#openvino
   * Set OpenVINO EP build option so `use_openvino` is `NPU` (for example `--use_openvino NPU` in ORT build.py).

##### Prepare `ONNXRUNTIME_ROOT` in KataGo (Windows)
Use package root:
* `cpp/external/onnxruntime-win-x64-openvino`

Windows one-to-one mapping (`<ORT_PACKAGE_ROOT>` -> KataGo):

Windows and Linux generally share the same ORT include/config layout.
The main differences are binary file names/extensions (`.dll/.lib` vs `.so`).

Include files:

| Source (`<ORT_PACKAGE_ROOT>`) | Destination (KataGo) |
| --- | --- |
| `include/core/*` | `cpp/external/onnxruntime-win-x64-openvino/include/core/` |
| `include/cpu_provider_factory.h` | `cpp/external/onnxruntime-win-x64-openvino/include/cpu_provider_factory.h` |
| `include/provider_options.h` | `cpp/external/onnxruntime-win-x64-openvino/include/provider_options.h` |
| `include/onnxruntime_c_api.h` | `cpp/external/onnxruntime-win-x64-openvino/include/onnxruntime_c_api.h` |
| `include/onnxruntime_cxx_api.h` | `cpp/external/onnxruntime-win-x64-openvino/include/onnxruntime_cxx_api.h` |
| `include/onnxruntime_cxx_inline.h` | `cpp/external/onnxruntime-win-x64-openvino/include/onnxruntime_cxx_inline.h` |
| `include/onnxruntime_env_config_keys.h` | `cpp/external/onnxruntime-win-x64-openvino/include/onnxruntime_env_config_keys.h` |
| `include/onnxruntime_ep_c_api.h` | `cpp/external/onnxruntime-win-x64-openvino/include/onnxruntime_ep_c_api.h` |
| `include/onnxruntime_ep_device_ep_metadata_keys.h` | `cpp/external/onnxruntime-win-x64-openvino/include/onnxruntime_ep_device_ep_metadata_keys.h` |
| `include/onnxruntime_float16.h` | `cpp/external/onnxruntime-win-x64-openvino/include/onnxruntime_float16.h` |
| `include/onnxruntime_lite_custom_op.h` | `cpp/external/onnxruntime-win-x64-openvino/include/onnxruntime_lite_custom_op.h` |
| `include/onnxruntime_run_options_config_keys.h` | `cpp/external/onnxruntime-win-x64-openvino/include/onnxruntime_run_options_config_keys.h` |
| `include/onnxruntime_session_options_config_keys.h` | `cpp/external/onnxruntime-win-x64-openvino/include/onnxruntime_session_options_config_keys.h` |

Library/config/pkgconfig files:

| Source (`<ORT_PACKAGE_ROOT>`) | Destination (KataGo) |
| --- | --- |
| `lib/onnxruntime.lib` | `cpp/external/onnxruntime-win-x64-openvino/lib/onnxruntime.lib` |
| `lib/onnxruntime.dll` | `cpp/external/onnxruntime-win-x64-openvino/lib/onnxruntime.dll` |
| `lib/onnxruntime_providers_shared.dll` | `cpp/external/onnxruntime-win-x64-openvino/lib/onnxruntime_providers_shared.dll` |
| `lib/onnxruntime_providers_openvino.dll` | `cpp/external/onnxruntime-win-x64-openvino/lib/onnxruntime_providers_openvino.dll` |
| `lib/onnxruntime_providers_shared.lib` (optional import lib) | `cpp/external/onnxruntime-win-x64-openvino/lib/onnxruntime_providers_shared.lib` |
| `lib/onnxruntime_providers_openvino.lib` (optional import lib) | `cpp/external/onnxruntime-win-x64-openvino/lib/onnxruntime_providers_openvino.lib` |
| `lib/cmake/onnxruntime/onnxruntimeConfig.cmake` | `cpp/external/onnxruntime-win-x64-openvino/lib/cmake/onnxruntime/onnxruntimeConfig.cmake` |
| `lib/cmake/onnxruntime/onnxruntimeConfigVersion.cmake` | `cpp/external/onnxruntime-win-x64-openvino/lib/cmake/onnxruntime/onnxruntimeConfigVersion.cmake` |
| `lib/cmake/onnxruntime/onnxruntimeTargets.cmake` | `cpp/external/onnxruntime-win-x64-openvino/lib/cmake/onnxruntime/onnxruntimeTargets.cmake` |
| `lib/cmake/onnxruntime/onnxruntimeTargets-release.cmake` | `cpp/external/onnxruntime-win-x64-openvino/lib/cmake/onnxruntime/onnxruntimeTargets-release.cmake` |
| `lib/pkgconfig/libonnxruntime.pc` (optional) | `cpp/external/onnxruntime-win-x64-openvino/lib/pkgconfig/libonnxruntime.pc` |

##### Minimal KataGo Build Commands (Windows, ONNX backend)
On Windows, `KATAGO_AUTO_FETCH_DEPS=ON` by default, so missing `zlib`, `onnx`, and `protobuf` dependencies are auto-fetched via vcpkg into `cpp/build/deps/vcpkg`.

```
cmake -S cpp -B cpp/build -G "Visual Studio 18 2026" -A x64 -DUSE_BACKEND=ONNX -DONNXRUNTIME_ROOT=cpp/external/onnxruntime-win-x64-openvino
cmake --build cpp/build --config Release -j
```

If you want to disable auto-fetch and provide dependencies manually:
* `-DKATAGO_AUTO_FETCH_DEPS=OFF`
* plus `-DONNX_INCLUDE_DIR=... -DONNX_PROTO_LIB=... -DPROTOBUF_INCLUDE_DIR=... -DPROTOBUF_LIB=... -DZLIB_INCLUDE_DIR=... -DZLIB_LIBRARY=...`

Typical run config for Intel NPU:
* `onnxProvider = openvino`
* `onnxOpenVINODeviceType = NPU`
* `onnxOpenVINOEnableNPUFastCompile = true` (optional; may be ignored on ORT builds that do not support this key)

Multi-device assignment is mainly for `onnxProvider=cuda/tensorrt/migraphx` (`onnxDeviceToUseThread*`).
For `onnxProvider=openvino` on Intel NPU, a single device is typically used.

##### Windows AMD NPU (VitisAI EP) Setup
This branch defaults to the `onnxruntime-win-x64-vitisai` ONNX Runtime package (built with the VitisAI execution provider) rather than the OpenVINO one used above. It targets AMD Ryzen AI NPUs via the [AMD Ryzen AI / VitisAI SDK](https://ryzenai.docs.amd.com/).

1. Install the Ryzen AI SDK (provides `onnxruntime_providers_vitisai.dll`'s runtime dependencies, `vaip_config.json`, and NPU `xclbin` firmware images). Typical install path: `C:\Program Files\RyzenAI\<version>`.
2. Prepare `cpp/external/onnxruntime-win-x64-vitisai/{include,lib,bin}` with an ONNX Runtime build that includes the VitisAI EP (`onnxruntime_providers_vitisai.dll`/`.lib`), matching the ORT ABI version bundled with your Ryzen AI SDK.
3. CMake auto-detects the newest `RyzenAI\<version>` folder under `%ProgramFiles%` at configure time (validated by the presence of `voe-*-win_amd64\vaip_config.json` inside it -- a stale `PATH` entry pointing at an uninstalled version is not trusted). Override with `-DRYZENAI_ROOT=...` or the `RYZENAI_ROOT` environment variable if auto-detection picks the wrong install or you have a nonstandard layout.
4. The build copies `vaip_config.json`, the `xclbins/` firmware directory, `xrt_coreutil.dll`, and the VitisAI EP's runtime dependency DLLs (`dyn_dispatch_core.dll`, `vaiml.dll`, `flexmlrt.dll`, etc., all from the Ryzen AI SDK's `deployment/` folder) next to `katago.exe` automatically. The ONNX Runtime core DLLs (`onnxruntime.dll`, `onnxruntime_providers_shared.dll`, `onnxruntime_providers_vitisai.dll`) come from `ONNXRUNTIME_ROOT` instead, to keep the ABI consistent with what KataGo was linked against.

```
cmake -S cpp -B cpp/build -G "Visual Studio 18 2026" -A x64 -DUSE_BACKEND=ONNX
cmake --build cpp/build --config Release -j
```

(`ONNXRUNTIME_ROOT` defaults to `cpp/external/onnxruntime-win-x64-vitisai` on this branch, so it does not need to be passed explicitly unless overriding.)

Typical run config for AMD Ryzen AI NPU:
* `onnxProvider = vitisai`
* `onnxVitisAIConfigFile = ...` (optional; defaults to the auto-detected `vaip_config.json` baked in at build time)
* `onnxVitisAICacheDir = ...` (optional; defaults to `<katagodata>/vitisaicache` -- the NPU model compile is slow, on the order of minutes, so this cache avoids recompiling on every launch)
* `onnxVitisAIDisableCPUFallback = true` (default; fails loudly at session creation if any node can't run on the NPU, rather than silently falling back to CPU for that node)

VitisAI only accelerates a quantized (INT8 QDQ) ONNX graph -- FP32 nodes fall back to CPU. To quantize a KataGo model:
1. `katago exportonnx -model <model>.bin.gz -xlen 19 -ylen 19 -output model-fp32.onnx` -- export a fixed-size FP32 ONNX graph.
2. `katago dumpcalibrationdata -model <model>.bin.gz -sgfdir <dir of real games> -output calib.npz` -- sample calibration input tensors from real games (reuses the same feature-encoding path as inference).
3. `python python/quantize_vitisai.py --input model-fp32.onnx --calibration calib.npz --output model-int8.onnx` -- offline quantize via AMD's `amd-quark` (`quark.onnx`), bundled with the Ryzen AI SDK's Python environment.
4. Point `nnModelFile` at `model-int8.onnx` and set `onnxProvider = vitisai`.

This quantization step is entirely offline/manual -- `katago.exe` does not invoke it automatically.

##### Windows AMD NPU with Python-Embedded ONNX Runtime (Workaround)

On some Ryzen AI SDK versions (observed with 1.7.1), the native C++ VitisAI execution provider path can fail during session creation for quantized models with many `EPContext` nodes, producing repeated errors such as `Failed to create runner: Failed to open library '...\xrt_core.dll'`. This appears to be an SDK/ORT C++ API initialization issue: the C++ API creates a fresh runner per subgraph, while Python's `onnxruntime.InferenceSession` reuses a single XRT context.

This branch provides an optional workaround that embeds a Python interpreter and routes VitisAI inference through Python's `onnxruntime.InferenceSession`. Enable it with:

```
cmake -S cpp -B cpp/build -G "Visual Studio 18 2026" -A x64 -DUSE_BACKEND=ONNX -DPYTHON_ONNXRUNTIME=ON
cmake --build cpp/build --config Release -j
```

Requirements:
* A Python installation with development headers/libs that matches the Ryzen AI SDK's recommended environment (e.g., the `ryzen-ai-1.7.1` conda env with Python 3.12). CMake uses `CONDA_PREFIX` if set, or you can point it with `Python3_ROOT_DIR`.
* The `onnxruntime` Python package installed in that environment, with a version matching the SDK's bundled ONNX Runtime ABI.
* The Ryzen AI SDK installed (auto-detected as described above).

The build bakes in the path to the Python interpreter (`KATAGO_PYTHON_EXE_PATH`). At runtime, `katago.exe` initializes Python from that installation and imports `numpy` and `onnxruntime`.

Deployment/runtime notes:
* The directory containing `katago.exe` should also contain the matching `python312.dll` (or whichever Python version you built against) and the conda `onnxruntime.dll`. This prevents Windows from loading the older `C:\Windows\System32\onnxruntime.dll` (1.17.1), which causes API version mismatch errors such as `TarWriter` failures.
* The conda `...\Lib\site-packages\onnxruntime\capi` directory must be on `PATH` so the VitisAI EP DLLs (`onnxruntime_vitisai_ep.dll`, etc.) are found.
* `PATH` must also include the Ryzen AI SDK's `xrt` and `voe-*-win_amd64` directories.

Example wrapper (`sabaki_katago.bat`) that keeps these PATH changes process-local:

```bat
@echo off
setlocal
set PATH=C:\Users\<user>\miniconda3\envs\ryzen-ai-1.7.1;%PATH%
set PATH=C:\Users\<user>\miniconda3\envs\ryzen-ai-1.7.1\Lib\site-packages\onnxruntime\capi;%PATH%
set PATH=C:\Program Files\RyzenAI\1.7.1\xrt;%PATH%
set PATH=C:\Program Files\RyzenAI\1.7.1\voe-4.0-win_amd64;%PATH%
cd /d "C:\Envs\katago-v1.16.5-vitisai1.7.1-windows-x64"
katago.exe gtp -config gtp_npu.cfg -model kata1-zhizi-b40c768nbt-s11272M-d5935M-mish-int8_sdk_ctx.onnx
endlocal
```

Caveats:
* The Python C API holds the GIL around bookkeeping calls. `session.run` releases it internally, but multi-threaded throughput has not been fully benchmarked.
* First inference can take 15–30 seconds while the NPU session is created and the model is compiled/cached.
* `katago benchmark` may time out because it creates a second session while the first is still active. This workaround is mainly intended for interactive GTP play.

## MacOS
   * TLDR (Metal backend - recommended for most users, hybrid CPU+GPU+Neural Engine for maximum throughput):
     ```
     git clone https://github.com/lightvector/KataGo.git
     cd KataGo/cpp
     # If you get missing library errors, install the appropriate packages using your system package manager and try again.
     # -DBUILD_DISTRIBUTED=1 is only needed if you want to contribute back to public training.
     cmake -G Ninja -DUSE_BACKEND=METAL -DBUILD_DISTRIBUTED=1
     ninja
     ```
   * Requirements
      * [Homebrew](https://brew.sh): `/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"`
      * CMake with a minimum version of 3.18.2: `brew install cmake`.
      * AppleClang and Swift compilers: `xcode-select --install`.
      * If using the Metal backend, [Ninja](https://ninja-build.org): `brew install ninja`
      * If using the Metal backend, protobuf and abseil: `brew install protobuf abseil`
      * libzip: `brew install libzip`.
      * If you want to do self-play training and research, probably Google perftools `brew install gperftools` for TCMalloc or some other better malloc implementation. For unknown reasons, the allocation pattern in self-play with large numbers of threads and parallel games causes a lot of memory fragmentation under glibc malloc that will eventually run your machine out of memory, but better mallocs handle it fine.
      * If compiling to contribute to public distributed training runs, OpenSSL is required (`brew install openssl`).
   * Clone this repo:
      * `git clone https://github.com/lightvector/KataGo.git`
   * Compile using CMake and make in the cpp directory:
      * `cd KataGo/cpp`
      * `cmake . -G Ninja -DUSE_BACKEND=METAL` or `cmake . -DUSE_BACKEND=OPENCL` or `cmake . -DUSE_BACKEND=EIGEN` depending on which backend you want.
         * Specify also `-DUSE_TCMALLOC=1` if using TCMalloc.
         * Compiling will also call git commands to embed the git hash into the compiled executable, specify also `-DNO_GIT_REVISION=1` to disable it if this is causing issues for you.
         * Specify `-DUSE_AVX2=1` to also compile Eigen with AVX2 and FMA support, which will make it incompatible with old CPUs but much faster. Intel-based Macs with new processors support AVX2, but Apple Silicon Macs do not support AVX2 natively. (If you want to go further, you can also add `-DCMAKE_CXX_FLAGS='-march=native'` which will specialize to precisely your machine's CPU, but the exe might not run on other machines at all).
         * Specify `-DBUILD_DISTRIBUTED=1` to compile with support for contributing data to public distributed training runs.
            * If building distributed, you will also need to build with Git revision support, including building within a clone of the repo, as opposed to merely an unzipped copy of its source.
            * Only builds from specific tagged versions or branches can contribute, in particular, instead of the `master` branch, use either the latest [release](https://github.com/lightvector/KataGo/releases) tag or the tip of the `stable` branch. To minimize the chance of any data incompatibilities or bugs, please do NOT attempt to contribute with custom changes or circumvent these limitations.
      * `ninja` for Metal backend, or `make` for other backends.
   * Done! You should now have a compiled `katago` executable in your working directory.
   * Pre-trained neural nets are available at [the main training website](https://katagotraining.org/).
   * You will probably want to edit `configs/gtp_example.cfg` (see "Tuning for Performance" above).
   * If using OpenCL, you will want to verify that KataGo is picking up the correct device when you run it (e.g. some systems may have both an Intel CPU OpenCL and GPU OpenCL, if KataGo appears to pick the wrong one, you can correct this by specifying `openclGpuToUse` in `configs/gtp_example.cfg`).
