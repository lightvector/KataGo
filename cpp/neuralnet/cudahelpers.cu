// CUDA wrapper for the shared CUDA/ROCm GPU kernels.
// All kernel code lives in cudaandrocmhelpers.inc, which is shared with the ROCm backend
// (rocmhelpers.hip). See the comment at the top of that file for the macro contract.

#include "../neuralnet/cudahelpers.h"

// Evaluated per device architecture during nvcc's per-arch device compilation passes, so the
// half-precision kernel bodies are compiled exactly for the archs that support them.
#if __CUDA_ARCH__ >= 530
#define KATAGO_GPU_SUPPORTS_FP16
#endif

#define KATAGO_GPU_CUDA 1
#define KATAGO_GPU_SINCOSF __sincosf

// Tensor-core flash attention is CUDA-only (mma.sync/ldmatrix/cp.async PTX). The shared .inc
// provides a fallback stub for builds without it.
#include "../neuralnet/cudaflashmma.cuh"
#define KATAGO_HAS_FLASH_MMA 1

#include "../neuralnet/cudaandrocmhelpers.inc"
