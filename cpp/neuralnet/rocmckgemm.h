// FP16 GEMM through Composable Kernel's DPP GEMM kernels (rocmckgemm.hip), which on RDNA2
// (gfx103x, no matrix cores) run many transformer projection shapes faster than the rocBLAS /
// Tensile kernels hipBLAS picks. Compiled only for the gfx103x archs of the build (see the CK GEMM
// section of CMakeLists.txt), so callers must check rocmCkGemmSupportsArch first.

#ifndef NEURALNET_ROCMCKGEMM_H_
#define NEURALNET_ROCMCKGEMM_H_

#include "../neuralnet/rocmincludes.h"

// Whether this build has CK GEMM kernels for the device architecture (hipDeviceProp_t::gcnArchName).
bool rocmCkGemmSupportsArch(const char* gcnArchName);

// out[t][o] = sum_i in[t][i] * w[o][i] for t < tokens, o < outChannels, i < inChannels, with in
// (NHWC activations), w (weights, out-major) and out (NHWC) row-major at row strides inStride,
// wStride, outStride (elements), and FP32 accumulation.
//
// The kernel instances, which perform differently per problem shape, so callers time them.
int rocmCkGemmNumInstances();
// Whether instance (in [0, rocmCkGemmNumInstances())) supports this problem.
bool rocmCkGemmSupports(int instance, int tokens, int outChannels, int inChannels, int inStride, int wStride, int outStride);
// Runs instance on the problem, which it must support. With accumulate, adds the product into out
// instead, rounded to FP16 first and then added in FP16: the same rounding as the transformer
// residual adds on masked boards, where the projection is written to FP16 and added into the
// trunk by the next RMSNorm, but not hipBLAS's beta = 1, which adds in FP32 and rounds once.
// Pointers must be 16-byte aligned.
void rocmCkGemmRun(
  int instance, bool accumulate,
  const half* in, int inStride, const half* w, int wStride, half* out, int outStride,
  int tokens, int outChannels, int inChannels, hipStream_t stream);

#endif  // NEURALNET_ROCMCKGEMM_H_
