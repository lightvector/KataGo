#ifndef NEURALNET_CUDAINCLUDES_H
#define NEURALNET_CUDAINCLUDES_H

//Ensure that CUDA_API_PER_THREAD_DEFAULT_STREAM is always defined
//before any cuda headers are included so that we get the desired threading behavior for CUDA.

#define CUDA_API_PER_THREAD_DEFAULT_STREAM
#include <cuda.h>
#include <cuda_fp16.h>

#include <cublas_v2.h>
#include <cudnn.h>


#endif //NEURALNET_CUDAINCLUDES_H
