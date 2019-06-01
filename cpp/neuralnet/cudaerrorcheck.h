#ifndef NEURALNET_CUDAERRORCHECK_H_
#define NEURALNET_CUDAERRORCHECK_H_

#define CUDA_API_PER_THREAD_DEFAULT_STREAM

#include <cuda.h>
#include <cublas_v2.h>
#include <cudnn.h>

#include "../core/global.h"

static void checkCudaError(const cudaError_t status, const char* opName, const char* file, const char* func, int line) {
  if(status != cudaSuccess)
    throw StringError(std::string("CUDA Error, for ") + opName + " file " + file + ", func " + func + ", line " + Global::intToString(line) + ", error " + cudaGetErrorString(status));
}
#define CUDA_ERR(opName,x) { checkCudaError((x),opName,__FILE__,#x,__LINE__); }

static const char* cublasGetErrorString(const cublasStatus_t status)
{
  switch(status)
  {
  case CUBLAS_STATUS_SUCCESS: return "CUBLAS_STATUS_SUCCESS";

  case CUBLAS_STATUS_ALLOC_FAILED: return "CUBLAS_STATUS_ALLOC_FAILED";
  case CUBLAS_STATUS_ARCH_MISMATCH: return "CUBLAS_STATUS_ARCH_MISMATCH";
  case CUBLAS_STATUS_EXECUTION_FAILED: return "CUBLAS_STATUS_EXECUTION_FAILED";
  case CUBLAS_STATUS_INTERNAL_ERROR: return "CUBLAS_STATUS_INTERNAL_ERROR";
  case CUBLAS_STATUS_INVALID_VALUE: return "CUBLAS_STATUS_INVALID_VALUE";
  case CUBLAS_STATUS_LICENSE_ERROR: return "CUBLAS_STATUS_LICENSE_ERROR";
  case CUBLAS_STATUS_MAPPING_ERROR: return "CUBLAS_STATUS_MAPPING_ERROR";
  case CUBLAS_STATUS_NOT_INITIALIZED: return "CUBLAS_STATUS_NOT_INITIALIZED";
  case CUBLAS_STATUS_NOT_SUPPORTED: return "CUBLAS_STATUS_NOT_SUPPORTED";
  default:
    return "UNKNOWN CUBLAS ERROR";
  }
}

static void checkCublasError(const cublasStatus_t status, const char* opName, const char* file, const char* func, int line) {
  (void)checkCublasError;
  if(status != CUBLAS_STATUS_SUCCESS)
    throw StringError(std::string("CUBLAS Error, for ") + opName + " file " + file + ", func " + func + ", line " + Global::intToString(line) + ", error " + cublasGetErrorString(status));
}
#define CUBLAS_ERR(opName,x) { checkCublasError((x),opName,__FILE__,#x,__LINE__); }

static void checkCudnnError(const cudnnStatus_t status, const char* opName, const char* file, const char* func, int line) {
  (void)checkCudnnError;
  if(status != CUDNN_STATUS_SUCCESS)
    throw StringError(std::string("CUDNN Error, for ") + opName + " file " + file + ", func " + func  + ", line " + Global::intToString(line) + ", error " + cudnnGetErrorString(status));
}
#define CUDNN_ERR(opName,x) { checkCudnnError((x),opName,__FILE__,#x,__LINE__); }

#endif  // NEURALNET_CUDAERRORCHECK_H_
