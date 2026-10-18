// FP16 GEMM through Composable Kernel's DPP GEMM kernels (rocmckgemm.hip), which on RDNA2
// (gfx103x, no matrix cores) run the transformer projection shapes 5-20% faster than the rocBLAS /
// Tensile kernels hipBLAS picks. Compiled only for the gfx103x archs of the build (see the CK GEMM
// section of CMakeLists.txt), so callers must check rocmCkGemmSupportsArch first.

#ifndef NEURALNET_ROCMCKGEMM_H_
#define NEURALNET_ROCMCKGEMM_H_

#include "../neuralnet/rocmincludes.h"

// Whether this build has CK GEMM kernels for the device architecture (hipDeviceProp_t::gcnArchName).
bool rocmCkGemmSupportsArch(const char* gcnArchName);

// C[M][N] = A[M][K] * B[N][K]^T with all three row-major at leading dimensions lda/ldb/ldc
// (elements), FP32 accumulation. In KataGo terms: A is NHWC activations (M = tokens, K = input
// channels), B the weights as [outC][inC], C the NHWC output.
//
// Picks the fastest kernel instance for this problem by timing each candidate on the given buffers
// (overwriting C) and synchronizing the stream. Returns -1 if no instance supports the problem,
// else the instance, with its time (best of several runs) in *bestMs.
int rocmCkGemmTune(
  const half* A, int lda, const half* B, int ldb, half* C, int ldc, int M, int N, int K, hipStream_t stream,
  float* bestMs);
// Runs the instance returned by rocmCkGemmTune for the same problem shape.
void rocmCkGemmRun(
  int instance, const half* A, int lda, const half* B, int ldb, half* C, int ldc, int M, int N, int K, hipStream_t stream);

#endif  // NEURALNET_ROCMCKGEMM_H_
