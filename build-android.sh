#!/usr/bin/env bash
# Cross-build KataGo for arm64 Android using the GB10's NATIVE arm64 clang-17 (matches
# NDK 26 = clang 17) aimed at the NDK sysroot + NDK compiler-rt. Same target ISA as the
# host → no qemu, full-speed compile. Usage: ./build-android.sh [EIGEN|OPENCL]
set -e
BACKEND=${1:-EIGEN}
NDK=/home/taro/android-sdk/ndk/26.3.11579264
PRE=$NDK/toolchains/llvm/prebuilt/linux-x86_64
SYS=$PRE/sysroot
RES=$PRE/lib/clang/17
TGT=aarch64-linux-android28
LIBDIR=$SYS/usr/lib/aarch64-linux-android
CF="--target=$TGT --sysroot=$SYS -resource-dir=$RES -include endian.h"
LF="$CF -rtlib=compiler-rt -unwindlib=libunwind -static-libstdc++ -fuse-ld=lld -L$LIBDIR/28 -L$LIBDIR"
BUILD=build-android-$(echo "$BACKEND" | tr 'A-Z' 'a-z')

EXTRA=()
if [ "$BACKEND" = "OPENCL" ]; then
  # Expose ONLY the CL/ headers (not all of /usr/include, which would leak glibc
  # headers into the bionic build) via an isolated include root.
  CLINC=/home/taro/code/katago/cl-include
  mkdir -p "$CLINC"; ln -sfn /usr/include/CL "$CLINC/CL"
  CF="$CF -I$CLINC"
  LF="$LF -I$CLINC"
  EXTRA+=( -DOpenCL_INCLUDE_DIR="$CLINC"
           -DOpenCL_LIBRARY=/home/taro/code/katago/android-libs/libOpenCL.so )
fi

cmake -S cpp -B "$BUILD" -GNinja \
  -DCMAKE_SYSTEM_NAME=Linux -DCMAKE_SYSTEM_PROCESSOR=aarch64 \
  -DCMAKE_C_COMPILER=clang-17 -DCMAKE_CXX_COMPILER=clang++-17 \
  -DCMAKE_C_FLAGS="$CF" -DCMAKE_CXX_FLAGS="$CF" \
  -DCMAKE_EXE_LINKER_FLAGS="$LF" \
  -DUSE_BACKEND="$BACKEND" \
  -DEIGEN3_INCLUDE_DIRS=/usr/include/eigen3 \
  -DZLIB_INCLUDE_DIR="$SYS/usr/include" -DZLIB_LIBRARY="$LIBDIR/28/libz.so" \
  -DCMAKE_THREAD_LIBS_INIT="" -DCMAKE_HAVE_LIBC_PTHREAD=ON \
  -DBUILD_DISTRIBUTED=0 -DUSE_TCMALLOC=0 \
  "${EXTRA[@]}"

cmake --build "$BUILD" -j"$(nproc)"
echo "=== built: $BUILD/katago ==="; file "$BUILD/katago"
