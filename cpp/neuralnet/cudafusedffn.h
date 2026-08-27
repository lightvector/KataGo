// Optional CUTLASS-based fused transformer FFN for the CUDA backend: one kernel computing
// SiLU(A @ W1) * (A @ Wgate) without writing the two intermediate GEMM outputs to global
// memory. Implemented in cudafusedffn.cu, which is compiled only when the build finds the
// vendored CUTLASS (CMake defines USE_CUTLASS_FUSED_FFN), so call sites must be guarded on
// that define.

#ifndef NEURALNET_CUDAFUSEDFFN_H_
#define NEURALNET_CUDAFUSEDFFN_H_

#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace CudaFusedFFN {
  // Whether the fused FFN kernel actually works on the current device: runs a tiny dual GEMM
  // and verifies it executed, guarding against the same hazard as
  // flashAttentionMmaSupportedOnCurrentDevice in cudaflashmma.cuh (pre-sm_80 PTX JIT-compiling
  // to an empty stub) as well as any failure to launch with its large dynamic shared memory
  // requirement. Synchronous and slightly costly, so call once per handle creation.
  bool supportedOnCurrentDevice();

  // Whether the kernel supports an FFN with weight matrices [N, K] (activations are [M, K]
  // with M varying per forward and not affecting support). Depends only on shape and
  // alignment, so callers may decide at model load time whether the fused path will be used
  // and commit to weight layouts accordingly.
  bool supportsShape(int N, int K);

  // out = SiLU(A @ W1) * (A @ Wgate), FP16 io and FP16 accumulation (matching the unfused
  // cublasHgemm path). A is [M, K] row-major (NHWC tokens), w1/wGate are packed out-major
  // ([N, K] row-major), out is [M, N] row-major. All pointers must be 16-byte aligned.
  // The caller must have verified supportedOnCurrentDevice() once and supportsShape(N, K)
  // beforehand, so any failure here is a genuine error and throws std::runtime_error.
  void runSwiGLU(
    const half* A, const half* w1, const half* wGate, half* out,
    int M, int N, int K, cudaStream_t stream
  );
}

#endif  // NEURALNET_CUDAFUSEDFFN_H_
