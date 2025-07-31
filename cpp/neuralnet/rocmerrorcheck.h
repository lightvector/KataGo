#ifndef NEURALNET_ROCMERRORCHECK_H_
#define NEURALNET_ROCMERRORCHECK_H_

#include "../neuralnet/rocmincludes.h"
#include "../core/global.h"

// ---------- HIP runtime ----------
static inline void checkCudaError(hipError_t status,
                                 const char* opName,
                                 const char* file,
                                 const char* func,
                                 int line) {
  if(status != hipSuccess)
    throw StringError(std::string("HIP Error @") + opName + " " +
                      file + ":" + func + ":" + Global::intToString(line) +
                      " : " + hipGetErrorString(status));
}
#define CUDA_ERR(opName,x)   checkCudaError((x),opName,__FILE__,#x,__LINE__)

// ---------- hipBLAS ----------
static inline const char* cublasGetErrorString(hipblasStatus_t s) {
  switch(s) {
    case HIPBLAS_STATUS_SUCCESS:          return "HIPBLAS_STATUS_SUCCESS";
    case HIPBLAS_STATUS_ALLOC_FAILED:     return "HIPBLAS_STATUS_ALLOC_FAILED";
    case HIPBLAS_STATUS_MAPPING_ERROR:    return "HIPBLAS_STATUS_MAPPING_ERROR";
    case HIPBLAS_STATUS_EXECUTION_FAILED: return "HIPBLAS_STATUS_EXECUTION_FAILED";
    case HIPBLAS_STATUS_INTERNAL_ERROR:   return "HIPBLAS_STATUS_INTERNAL_ERROR";
    case HIPBLAS_STATUS_INVALID_VALUE:    return "HIPBLAS_STATUS_INVALID_VALUE";
    case HIPBLAS_STATUS_NOT_INITIALIZED:  return "HIPBLAS_STATUS_NOT_INITIALIZED";
    case HIPBLAS_STATUS_NOT_SUPPORTED:    return "HIPBLAS_STATUS_NOT_SUPPORTED";
    default:                              return "HIPBLAS_STATUS_UNKNOWN";
  }
}
static inline void checkCublasError(hipblasStatus_t status,
                                     const char* opName,
                                     const char* file,
                                     const char* func,
                                     int line) {
  if(status != HIPBLAS_STATUS_SUCCESS)
    throw StringError(std::string("hipBLAS Error @") + opName + " " +
                      file + ":" + func + ":" + Global::intToString(line) +
                      " : " + cublasGetErrorString(status));
}
#define CUBLAS_ERR(opName,x) checkCublasError((x),opName,__FILE__,#x,__LINE__)

// ---------- MIOpen ----------
static inline void checkCudnnError(miopenStatus_t status,
                                    const char* opName,
                                    const char* file,
                                    const char* func,
                                    int line) {
  if(status != miopenStatusSuccess)
    throw StringError(std::string("MIOpen Error @") + opName + " " +
                      file + ":" + func + ":" + Global::intToString(line) +
                      " : " + miopenGetErrorString(status));
}
#define CUDNN_ERR(opName,x) checkCudnnError((x),opName,__FILE__,#x,__LINE__)

#endif // NEURALNET_ROCMERRORCHECK_H_
