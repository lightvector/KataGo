
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
      * If using the TensorRT backend, in addition to a compatible CUDA Toolkit (https://developer.nvidia.com/cuda-toolkit), you also need TensorRT (https://developer.nvidia.com/tensorrt) version 10 or newer (as of KataGo v1.17.0, TensorRT versions older than 10 are no longer supported).
      * If using the ROCm backend, ROCm 6.4 or later (https://rocm.docs.amd.com/projects/install-on-linux/en/latest/) and a GPU capable of supporting it. Install the ROCm developer packages, not just the ROCm runtime packages.
      * If using the Eigen backend, Eigen3. With Debian packages, (i.e. apt or apt-get), this should be `libeigen3-dev`.
      * zlib, libzip. With Debian packages (i.e. apt or apt-get), these should be `zlib1g-dev`, `libzip-dev`.
      * If you want to do self-play training and research, probably Google perftools `libgoogle-perftools-dev` for TCMalloc or some other better malloc implementation. For unknown reasons, the allocation pattern in self-play with large numbers of threads and parallel games causes a lot of memory fragmentation under glibc malloc that will eventually run your machine out of memory, but better mallocs handle it fine.
      * If compiling to contribute to public distributed training runs, OpenSSL is required (`libssl-dev`).
   * Clone this repo:
      * `git clone https://github.com/lightvector/KataGo.git`
   * Compile using CMake and make in the cpp directory:
      * `cd KataGo/cpp`
      * `cmake . -DUSE_BACKEND=OPENCL` or `cmake . -DUSE_BACKEND=CUDA` or `cmake . -DUSE_BACKEND=TENSORRT` or `cmake . -DUSE_BACKEND=EIGEN` or `cmake . -DUSE_BACKEND=ROCM` depending on which backend you want.
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

   * **ROCm backend (Linux) - additional notes:**
      * Install ROCm following the [official guide](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/). Install the full developer stack (not just runtime): `sudo apt install rocm-dev miopen-hip-dev hipblas-dev rocblas-dev`.
      * Build:
        ```
        cd KataGo/cpp
        mkdir build && cd build
        cmake .. -DUSE_BACKEND=ROCM -DCMAKE_BUILD_TYPE=Release
        make -j$(nproc)
        ```
        No `-DCMAKE_PREFIX_PATH` is needed in the common case: the build auto-detects the ROCm
        install location, preferring the newer `/opt/rocm/core-<version>/` layout used since
        ROCm ~7.9 (picking the highest version present) and falling back to the older flat
        `/opt/rocm/` layout. Pass `-DCMAKE_PREFIX_PATH=...` explicitly to override - this is what to
        use if your ROCm lives somewhere else entirely. It should name the install **prefix**, i.e.
        the directory containing `include/hip/hip_runtime.h`, not a subdirectory of it.
      * Always configure in a clean build directory. A `CMakeCache.txt` left over from an earlier
        configure (especially one for a different backend) suppresses the automatic selection of
        `hipcc` and of the GPU architecture list, which otherwise shows up as strange compile errors
        like `unrecognized command-line option '-Xarch_host'`.
      * GPU architecture: by default the build targets a broad set of AMD GPU architectures
        (Vega 20, CDNA, and all RDNA generations) in a single "fat" binary, probing the installed compiler for
        which ones it actually supports, so the resulting `katago` runs on more than just the
        machine it was built on. Pass `-DCMAKE_HIP_ARCHITECTURES=gfx1100` (replace with your GPU's
        gfx target) to build for only your own GPU instead, which is faster to compile.
      * The build bakes the resolved ROCm lib directory into the binary's RPATH, so the built
        `katago` doesn't depend on `/opt/rocm` still pointing at the same install later (e.g. after
        installing a different ROCm version) or on `LD_LIBRARY_PATH` being set when run. The
        generic `/opt/rocm/lib` path is included as a fallback after the pinned path, so the binary
        can also run on a machine with a different (soname-compatible) ROCm version installed.
      * On first run, MIOpen will search for optimal convolution algorithms for your specific GPU and network size. This may take up to a minute and results are cached (compiled kernels in `~/.cache/miopen/`, the tuning find-db in `~/.config/miopen/`) for subsequent runs.
      * **Transformer/attention models (model version 17+):** supported on all architectures via a
        built-in kernel. If a matching version of AMD's Composable Kernel (CK, packaged as
        `composablekernel-dev` / `amdrocm-ck*`) is also installed, the build additionally enables a fused-attention fast path
        (measured ~2x faster on a gfx1100/RX 7900 XTX) for CDNA and RDNA3/RDNA3.5/RDNA4 GPUs -
        RDNA1/RDNA2 always use the built-in kernel, as does any GPU if the fused path isn't
        available or is explicitly disabled with `rocmDisableFusedAttention = true` in the config.
        See `cpp/external/composable_kernel_fmha/README.md` for details.

## Windows
   * TLDR:
      * Building from source on Windows is actually a bit tricky, depending on what version you're building, there's not necessarily a super-fast way.
   * Requirements
      * CMake with a minimum version of 3.18.2, GUI version strongly recommended (https://cmake.org/download/)
      * Microsoft Visual Studio for C++. Version 15 (2017) has been tested and should work, MinGW version also should work but only with Eigen and OpenCL backends (CUDA and TensorRT MinGW backends are [not supported by NVIDIA](https://forums.developer.nvidia.com/t/cuda-with-mingw-how-to-get-cuda-running-under-mingw)).
      * If using the OpenCL backend, a modern GPU that supports OpenCL 1.2 or greater, or else something like [this](https://software.intel.com/en-us/opencl-sdk) for CPU. But if using CPU, Eigen should be better.
      * If using the CUDA backend, CUDA 11 or later and a compatible version of CUDNN based on your CUDA version (https://developer.nvidia.com/cuda-toolkit) (https://developer.nvidia.com/cudnn) and a GPU capable of supporting them. I'm unsure how version compatibility works with CUDA, there's a good chance that later versions than these work just as well, but they have not been tested.
      * If using the TensorRT backend, in addition to a compatible CUDA Toolkit (https://developer.nvidia.com/cuda-toolkit), you also need TensorRT (https://developer.nvidia.com/tensorrt) version 10 or newer (as of KataGo v1.17.0, TensorRT versions older than 10 are no longer supported).
      * If using the Eigen backend, Eigen3, version 3.3.x. (http://eigen.tuxfamily.org/index.php?title=Main_Page#Download).
      * zlib. Easy way to build zlib on Windows is to use vcpkg. Run in Powershell:
         * git clone https://github.com/microsoft/vcpkg.git
         * cd .\vcpkg\
         * .\bootstrap-vcpkg.bat
         * .\vcpkg.exe install zlib:x64-windows
         * Set CMake ZLIB_LIBRARY to vcpkg\installed\x64-windows\lib\zlib.lib and ZLIB_INCLUDE_DIRECTORY to vcpkg\installed\x64-windows\include.
         * Copy zlib1.dll from vcpkg\installed\x64-windows\bin to Katago folder after you've built Katago executable.
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
      * Also set `USE_BACKEND` to `OPENCL`, or `CUDA`, or `TENSORRT`, or `EIGEN` depending on what backend you want to use.
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

   * **ROCm backend (Windows) - building via AMD TheRock:**
      * The ROCm (MIOpen) backend supports Windows via [AMD TheRock](https://github.com/ROCm/TheRock) (tested with TheRock 7.13 / ROCm 7.13, RX 7900 XTX / gfx1100), including transformer/attention models (model version 17+) and the optional Composable Kernel (CK) fused-attention fast path.
      * **Prerequisites:**
         * Download [AMD TheRock](https://github.com/ROCm/TheRock) and extract it to e.g. `C:\TheRock\build`, adjusting the paths below if you extract elsewhere.
         * Install **Visual Studio Build Tools or Community** with the "Desktop development with C++" workload, for the MSVC toolchain and Windows SDK the HIP compiler needs. If several toolsets are installed side by side, `CMakeLists.txt` probes them at configure time, newest first, and picks the first one the HIP compiler actually accepts (see "Fully automatic" below), no manual toolset selection needed. A very new MSVC STL can be ahead of TheRock's bundled clang, so if configure reports that no installed toolset works, add an older one such as **MSVC v143 (14.3x/14.4x)** from the Visual Studio Installer's "Individual Components" tab.
         * Install [Ninja](https://ninja-build.org) build tool: `winget install Ninja-build.Ninja`.
         * Set the following **system environment variables** (via System Properties -> Advanced -> Environment Variables):
           ```
           HIP_PATH=C:/TheRock/build
           HIP_PLATFORM=amd
           HIP_DEVICE_LIB_PATH=C:/TheRock/build/lib/llvm/amdgcn/bitcode
           LLVM_PATH=C:/TheRock/build/lib/llvm
           ```
         * Add to system `PATH`:
           ```
           C:\TheRock\build\bin
           C:\TheRock\build\lib\llvm\bin
           ```
         * Reboot after setting environment variables so they take effect system-wide.
      * **Build - fully automatic**, just like Linux:
        ```
        cd KataGo/cpp
        mkdir build
        cd build
        cmake .. -G Ninja -DUSE_BACKEND=ROCM -DCMAKE_BUILD_TYPE=Release
        ninja
        ```
        No manual environment setup, no `-D` flags, no `vcvarsall`, and no external package manager
        install are needed beyond the prerequisites above. `CMakeLists.txt` handles the rest of the
        Windows-specific setup automatically at configure/build time:
         * **MSVC toolset selection:** if more than one MSVC toolset is installed side by side, a
           newer one can conflict with TheRock's bundled clang (newer MSVC STL headers are not yet
           compatible with it). `CMakeLists.txt` enumerates every installed toolset via
           `vswhere` and probes them newest-first with a real compile until one works, with
           no user action needed.
         * **zlib:** TheRock's Windows package ships `zlib.h` but (as of 7.13) no longer ships a
           linkable `.lib`. `CMakeLists.txt` automatically bootstraps a local
           [vcpkg](https://github.com/microsoft/vcpkg) clone under `<build dir>/deps/vcpkg` (this
           needs internet access and `git` on `PATH` the first time; subsequent reconfigures reuse
           the same local install) and builds zlib through it, via the
           `KATAGO_AUTO_FETCH_DEPS` mechanism (`KATAGO_DEPS_DIR` overrides the location, e.g. to
           share fetched deps across multiple build directories).
         * **Runtime DLLs:** all the ROCm/HIP DLLs (`amdhip64_7.dll`, `MIOpen.dll`, `hipblas.dll`,
           `rocblas.dll` + its `library/` subfolder, `libhipblaslt.dll` + its `library/` subfolder,
           `amdocl64.dll`, `hiprtc*.dll`, `amd_comgr*.dll`) and the vcpkg zlib runtime DLL are
           automatically copied next to `katago.exe` as a post-build step - nothing to copy by hand.
        `ck_tile` headers for the optional fused-attention path are still auto-detected from
        `HIP_PATH` the same way as on Linux.
      * **First-run note:** MIOpen will search for optimal convolution algorithms on the first run. This may take 45+ seconds per network configuration and results are cached in `%USERPROFILE%\.miopen\` for subsequent runs. Do not terminate the process during this initial tuning.
      * **Performance note:** GPU utilization on Windows may be somewhat lower than on Linux due to the Windows Driver Model (WDDM) adding overhead to GPU kernel submissions. This is a known limitation of ROCm on Windows. For example, the CK fused-attention path measured ~2x faster than the built-in kernel on Linux (gfx1100), but only ~1.3x faster on Windows on the same GPU.

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

## ONNX Runtime backend (optional)
The `ONNX` backend loads the usual KataGo neural net files, but runs them using [ONNX Runtime](https://onnxruntime.ai/), which supports many kinds of hardware through its "execution providers". This is mainly useful for hardware that has no native KataGo backend, such as Intel GPUs and NPUs (via the OpenVINO provider). If you have an NVIDIA GPU, use KataGo's native CUDA or TensorRT backends instead - they will almost certainly be faster.

### Execution providers

Set `onnxProvider` in your config to choose the execution provider:

| Provider | Status | Platform | ONNX Runtime needed | Also needs at runtime |
|---|---|---|---|---|
| `openvino` | Tested (Windows, Intel GPU and NPU) | Windows / Linux | built from source with `--use_openvino GPU` | OpenVINO runtime, TBB |
| `cpu` | Tested | All | official prebuilt package | - |
| `cuda` | Lightly tested (Linux) | Windows / Linux | official prebuilt GPU package (`gpu_cuda12`), or built with `--use_cuda` | CUDA, cuDNN 9 |
| `tensorrt` | Lightly tested (Linux) | Windows / Linux | official prebuilt GPU package (`gpu_cuda12`), or built with `--use_tensorrt` | CUDA, TensorRT 10 |
| `migraphx` | Untested | Linux (AMD) | built from source with `--use_migraphx` | MIGraphX |
| `directml` | Tested (Windows) | Windows | Microsoft.ML.OnnxRuntime.DirectML package | DirectML |
| `coreml` | Not working yet | macOS | built from source with `--use_coreml` | CoreML |

"Tested" means KataGo's neural net tests give correct results with this provider. "Lightly tested" means correctness was checked but performance was not. "Untested" providers are implemented but have never been run - if you try one, sanity-check its results, for example against the `cpu` provider. The `directml` provider may also be slow, because it prefers fixed tensor sizes while KataGo's search varies its batch size.

> **Note**: This backend is more involved to set up than KataGo's other backends. Most execution providers are not included in the official prebuilt ONNX Runtime packages - in particular, using the OpenVINO provider requires building ONNX Runtime from source with that provider enabled.

### Requirements
   * Everything KataGo normally needs (CMake, a C++17 compiler, zlib).
   * ONNX Runtime including the execution provider you want - either an official prebuilt package, or built from source with the provider enabled (see the table above and https://onnxruntime.ai/docs/install/).
   * Protobuf 3.x, such that CMake's `find_package(Protobuf)` succeeds. On Linux the usual system packages work. If you build ONNX Runtime from source, the protobuf inside its build tree also works.
   * For the OpenVINO provider, the OpenVINO toolkit.

### Compile
   * Point CMake at your ONNX Runtime install and select the backend:
     ```
     cmake -S KataGo/cpp -B KataGo/cpp/build -DUSE_BACKEND=ONNX -DONNXRUNTIME_ROOT=<path-to-onnxruntime>
     cmake --build KataGo/cpp/build -j
     ```
   * `ONNXRUNTIME_ROOT` is the directory containing ONNX Runtime's `include/` and `lib/`.
   * If CMake does not find protobuf on its own, also pass `-DProtobuf_PROTOC_EXECUTABLE=<protoc>`, `-DProtobuf_INCLUDE_DIR=<include-dir>`, and `-DProtobuf_LIBRARY=<library>`.
   * As with other backends, `-DNO_GIT_REVISION=1` avoids embedding the git hash, and `-DBUILD_DISTRIBUTED=1` enables contributing to distributed training.

### Runtime
   * The `onnxruntime` shared library must be next to the executable or on your library path.
   * For the OpenVINO provider, the OpenVINO runtime DLLs (`openvino.dll`, `openvino_intel_gpu_plugin.dll`, `tbb12.dll`, `cache.json`, etc.) must also be next to the executable or on the system path.
   * For the DirectML provider, `DirectML.dll` 1.8.0 or newer (from the Microsoft.AI.DirectML package) must be next to `onnxruntime.dll`. Without it, Windows 10 falls back to its much older inbox DirectML and the provider fails at startup.
   * Choose the provider and its options with the `onnx*` keys in your config, e.g. `onnxProvider = openvino` and `onnxOpenVINODeviceType = GPU`. The ONNX section of `configs/gtp_example.cfg` documents all the options.

### Working with .onnx files
This backend and the TensorRT backend can also write out the ONNX graph they build (`katago dumponnx`) and load a `.onnx` file as a model in place of the `.bin.gz`, including one produced by other tooling. See **[ONNX_Model_Files.md](docs/ONNX_Model_Files.md)** for the commands and the model file format.
