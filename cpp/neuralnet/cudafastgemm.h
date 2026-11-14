// Autotuned FP16 GEMM for the CUDA backend. The first time a problem shape is seen, it times a
// small set of CUTLASS tensor-core kernels (plain and stream-K tilings) against cublasHgemm on
// the real operands, and from then on dispatches to the fastest. KataGo's transformer GEMMs have
// narrow outputs (384-1152 columns) for which cuBLAS's heuristic tiling often leaves SMs idle or
// underoccupied on mid-size GPUs.
//
// Compiled only with the vendored CUTLASS (same condition as cudafusedffn.cu, guarded by
// USE_CUTLASS_FUSED_FFN at call sites). The kernels need sm_80+: callers must only use this
// after CudaFusedFFN::supportedOnCurrentDevice() succeeded, since CUTLASS's below-arch fallback
// is a device-side trap.

#ifndef NEURALNET_CUDAFASTGEMM_H_
#define NEURALNET_CUDAFASTGEMM_H_

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

namespace CudaFastGemm {
  // False if the KATAGO_CUDA_DISABLE_FAST_GEMM env var is set to a nonzero value (for A/B testing).
  bool enabled();

  // Column-major (cuBLAS convention) C[m x n] = W[m x k] * X[k x n], plus C if accumulate.
  // FP16 inputs/outputs, FP16 accumulation, matching cublasHgemm. Leading dimensions are packed
  // (m, k, m). The first call for a new (device, m, n, k, accumulate) synchronizes the stream to
  // time the candidates, so it should happen during warmup and never during CUDA graph capture.
  // Set KATAGO_CUDA_FAST_GEMM_LOG=1 to print each autotuning decision to stderr.
  void hgemm(
    cublasHandle_t cublas, cudaStream_t stream,
    int m, int n, int k,
    const half* W, const half* X, half* C, bool accumulate
  );
}

#endif  // NEURALNET_CUDAFASTGEMM_H_
