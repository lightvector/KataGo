
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
      * If using the ROCm backend, ROCm 6.4 or later and a GPU capable of supporting them. More information about installation(https://rocm.docs.amd.com/projects/install-on-linux/en/latest/) and please install all possiable ROCm developer packages, instead of just ROCm runtime packages.
      * If using the Eigen backend, Eigen3. With Debian packages, (i.e. apt or apt-get), this should be `libeigen3-dev`.
      * zlib, libzip. With Debian packages (i.e. apt or apt-get), these should be `zlib1g-dev`, `libzip-dev`.
      * If you want to do self-play training and research, probably Google perftools `libgoogle-perftools-dev` for TCMalloc or some other better malloc implementation. For unknown reasons, the allocation pattern in self-play with large numbers of threads and parallel games causes a lot of memory fragmentation under glibc malloc that will eventually run your machine out of memory, but better mallocs handle it fine.
      * If compiling to contribute to public distributed training runs, OpenSSL is required (`libssl-dev`).
   * Clone this repo:
      * `git clone https://github.com/lightvector/KataGo.git`
   * Compile using CMake and make in the cpp directory:
      * `cd KataGo/cpp`
      * `cmake . -DUSE_BACKEND=OPENCL` or `cmake . -DUSE_BACKEND=CUDA` or `cmake . -DUSE_BACKEND=TENSORRT` or `cmake . -DUSE_BACKEND=EIGEN` or `cmake . -DUSE_BACKEND=ROCM`depending on which backend you want.
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

## MacOS
   * TLDR:
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
   * If you want to run `synchronous_loop.sh` on macOS, do the following steps:
      * Install GNU coreutils `brew install coreutils` to support a `head` tool that can take negative numbers (`head -n -5` in `train.sh`)
      * Install GNU findutils `brew install findutils` to support a `find` tool that supports `-printf` option, that's used by `export_model_for_selfplay.sh`. After that, fix `find` with `gfind` in the script.
        Note: you can try to avoid fixing `export_model_for_selfplay.sh` by adjusting `PATH` with the installed findutils: `export PATH="/opt/homebrew/opt/findutils/libexec/gnubin:$PATH"` or by using the alias `alias find="gfind"`. However, it works not always.