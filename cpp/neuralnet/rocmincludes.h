#ifndef NEURALNET_ROCMINCLUDES_H
#define NEURALNET_ROCMINCLUDES_H

//Ensure that CUDA_API_PER_THREAD_DEFAULT_STREAM is always defined
//before any cuda headers are included so that we get the desired threading behavior for CUDA.

#define CUDA_API_PER_THREAD_DEFAULT_STREAM
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

#include <hipblas/hipblas.h>
#include <miopen/miopen.h>


#endif //NEURALNET_ROCMINCLUDES_H
