// Maps the CUDA spellings used by the shared backend implementation file
// (cudaandrocmbackend.inc) to their HIP/hipBLAS/MIOpen equivalents, so that the code shared
// with the CUDA backend can be written once, with CUDA spellings, and compiled unchanged by hip
// clang. Only the names that the SHARED regions of that file actually use are mapped here. Code
// inside its KATAGO_GPU_HIP regions uses native HIP/MIOpen spellings directly, and cuDNN/cuBLAS
// API with no direct MIOpen/hipBLAS equivalent (convolution setup, algo search, ...) exists only
// inside KATAGO_GPU_CUDA regions and is deliberately absent here.
//
// Functions are static inline forwarders rather than #defines so that types are checked and
// nothing leaks into other headers. hipMalloc also has a templated overload that would make
// function-pointer aliases ambiguous. Adding a new CUDA API call to shared code means adding
// its mapping here - the HIP build error is the reminder.

#ifndef NEURALNET_ROCMCUDANAMES_H_
#define NEURALNET_ROCMCUDANAMES_H_

#include "../neuralnet/rocmincludes.h"

// Types
using cudaDeviceProp = hipDeviceProp_t;
using cublasHandle_t = hipblasHandle_t;
using cudnnHandle_t = miopenHandle_t;
using cudnnStatus_t = miopenStatus_t;
using cudnnTensorDescriptor_t = miopenTensorDescriptor_t;

// Enum constants
constexpr auto cudaSuccess = hipSuccess;
constexpr auto cudaMemcpyHostToDevice = hipMemcpyHostToDevice;
constexpr auto cudaMemcpyDeviceToHost = hipMemcpyDeviceToHost;
constexpr auto cudaStreamNonBlocking = hipStreamNonBlocking;
constexpr auto cudaHostAllocPortable = hipHostMallocPortable;
constexpr auto CUBLAS_OP_N = HIPBLAS_OP_N;

// Runtime API
//void** only (not a template): the real cudaMalloc is looser (it has a T** overload), but
//keeping the shim strict means a shared call site that compiles here also compiles on CUDA.
static inline hipError_t cudaMalloc(void** ptr, size_t size) { return hipMalloc(ptr, size); }
static inline hipError_t cudaFree(void* ptr) { return hipFree(ptr); }
static inline hipError_t cudaMemcpy(void* dst, const void* src, size_t size, hipMemcpyKind kind) {
  return hipMemcpy(dst, src, size, kind);
}
static inline hipError_t cudaDeviceSynchronize() { return hipDeviceSynchronize(); }
static inline hipError_t cudaDeviceReset() { return hipDeviceReset(); }
static inline hipError_t cudaGetDeviceCount(int* count) { return hipGetDeviceCount(count); }
static inline hipError_t cudaGetDevice(int* device) { return hipGetDevice(device); }
static inline hipError_t cudaGetDeviceProperties(hipDeviceProp_t* prop, int device) {
  return hipGetDeviceProperties(prop, device);
}
static inline hipError_t cudaPeekAtLastError() { return hipPeekAtLastError(); }
static inline hipError_t cudaGetLastError() { return hipGetLastError(); }
static inline hipError_t cudaStreamCreateWithFlags(hipStream_t* stream, unsigned int flags) {
  return hipStreamCreateWithFlags(stream, flags);
}
static inline hipError_t cudaStreamDestroy(hipStream_t stream) { return hipStreamDestroy(stream); }
static inline hipError_t cudaStreamSynchronize(hipStream_t stream) { return hipStreamSynchronize(stream); }
static inline hipError_t cudaMemcpyAsync(void* dst, const void* src, size_t size, hipMemcpyKind kind, hipStream_t stream) {
  return hipMemcpyAsync(dst, src, size, kind, stream);
}
static inline hipError_t cudaHostAlloc(void** ptr, size_t size, unsigned int flags) {
  return hipHostMalloc(ptr, size, flags);
}
static inline hipError_t cudaFreeHost(void* ptr) { return hipHostFree(ptr); }

// hipBLAS
static inline hipblasStatus_t cublasCreate(hipblasHandle_t* handle) { return hipblasCreate(handle); }
static inline hipblasStatus_t cublasDestroy(hipblasHandle_t handle) { return hipblasDestroy(handle); }
static inline hipblasStatus_t cublasSetStream(hipblasHandle_t handle, hipStream_t stream) {
  return hipblasSetStream(handle, stream);
}
static inline hipblasStatus_t cublasSgemm(
  hipblasHandle_t handle, hipblasOperation_t transa, hipblasOperation_t transb,
  int m, int n, int k,
  const float* alpha, const float* A, int lda, const float* B, int ldb,
  const float* beta, float* C, int ldc
) {
  return hipblasSgemm(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
}
static inline hipblasStatus_t cublasHgemm(
  hipblasHandle_t handle, hipblasOperation_t transa, hipblasOperation_t transb,
  int m, int n, int k,
  const hipblasHalf* alpha, const hipblasHalf* A, int lda, const hipblasHalf* B, int ldb,
  const hipblasHalf* beta, hipblasHalf* C, int ldc
) {
  return hipblasHgemm(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
}
static inline hipblasStatus_t cublasSgemmStridedBatched(
  hipblasHandle_t handle, hipblasOperation_t transa, hipblasOperation_t transb,
  int m, int n, int k,
  const float* alpha, const float* A, int lda, long long strideA,
  const float* B, int ldb, long long strideB,
  const float* beta, float* C, int ldc, long long strideC,
  int batchCount
) {
  return hipblasSgemmStridedBatched(handle, transa, transb, m, n, k, alpha, A, lda, strideA, B, ldb, strideB, beta, C, ldc, strideC, batchCount);
}
static inline hipblasStatus_t cublasHgemmStridedBatched(
  hipblasHandle_t handle, hipblasOperation_t transa, hipblasOperation_t transb,
  int m, int n, int k,
  const hipblasHalf* alpha, const hipblasHalf* A, int lda, long long strideA,
  const hipblasHalf* B, int ldb, long long strideB,
  const hipblasHalf* beta, hipblasHalf* C, int ldc, long long strideC,
  int batchCount
) {
  return hipblasHgemmStridedBatched(handle, transa, transb, m, n, k, alpha, A, lda, strideA, B, ldb, strideB, beta, C, ldc, strideC, batchCount);
}

// MIOpen, tensor descriptor lifetime only. Descriptor SETUP has no cuDNN-compatible signature
// and lives in the KATAGO_GPU_HIP regions of cudaandrocmbackend.inc.
static inline miopenStatus_t cudnnCreate(miopenHandle_t* handle) { return miopenCreate(handle); }
static inline miopenStatus_t cudnnDestroy(miopenHandle_t handle) { return miopenDestroy(handle); }
static inline miopenStatus_t cudnnSetStream(miopenHandle_t handle, hipStream_t stream) {
  return miopenSetStream(handle, stream);
}
static inline miopenStatus_t cudnnCreateTensorDescriptor(miopenTensorDescriptor_t* desc) {
  return miopenCreateTensorDescriptor(desc);
}
static inline miopenStatus_t cudnnDestroyTensorDescriptor(miopenTensorDescriptor_t desc) {
  return miopenDestroyTensorDescriptor(desc);
}

#endif  // NEURALNET_ROCMCUDANAMES_H_
