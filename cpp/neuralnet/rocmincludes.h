#ifndef NEURALNET_ROCMINCLUDES_H
#define NEURALNET_ROCMINCLUDES_H

//Note: unlike the CUDA backend (which defines CUDA_API_PER_THREAD_DEFAULT_STREAM here), the ROCm
//backend currently runs all work on the shared legacy null stream. HIP's analog would be defining
//HIP_API_PER_THREAD_DEFAULT_STREAM (or compiling with -fgpu-default-stream=per-thread), but
//flipping it also changes which stream MIOpen/hipBLAS calls run on relative to our own kernels
//and memcpys, so it needs a careful audit of stream synchronization before enabling. Correctness
//is unaffected either way. Multiple server threads sharing one GPU just serialize their GPU work.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

// hipBLAS 2.x (ROCm 6.x) declared hipblasGemmEx's type arguments as the now-removed
// hipblasDatatype_t. Defining HIPBLAS_V2 first selects the modern spelling (hipDataType plus
// hipblasComputeType_t) that hipBLAS 3.0+ uses unconditionally, so one call site compiles
// against both. hipBLAS 3.x defines this
// macro itself and nothing in it is conditional on the macro, so predefining it is a no-op there.
#define HIPBLAS_V2
#include <hipblas/hipblas.h>
#include <miopen/miopen.h>


#endif //NEURALNET_ROCMINCLUDES_H
