#ifndef NEURALNET_ROCMINCLUDES_H
#define NEURALNET_ROCMINCLUDES_H

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

//The shared kernel launcher declarations (cudaandrocmhelpers.h) are written with the CUDA
//stream type spelling. This alias must live here rather than in rocmcudanames.h because
//rocmhelpers.hip reaches those declarations without including rocmcudanames.h. All backend work
//for one compute handle runs on that handle's non-blocking stream (see OwnedComputeStream in
//cudaandrocmbackend.inc), so multiple NN server threads sharing one GPU overlap their GPU work.
using cudaStream_t = hipStream_t;

// hipBLAS 2.x (ROCm 6.x) declared hipblasGemmEx's type arguments as the now-removed
// hipblasDatatype_t. Defining HIPBLAS_V2 first selects the modern spelling (hipDataType plus
// hipblasComputeType_t) that hipBLAS 3.0+ uses unconditionally, so one call site compiles
// against both. hipBLAS 3.x defines this
// macro itself and nothing in it is conditional on the macro, so predefining it is a no-op there.
#define HIPBLAS_V2
#include <hipblas/hipblas.h>
#include <miopen/miopen.h>


#endif //NEURALNET_ROCMINCLUDES_H
