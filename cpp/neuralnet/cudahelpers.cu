
#define CUDA_API_PER_THREAD_DEFAULT_STREAM

#include <cublas_v2.h>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/generate.h>
#include <thrust/reduce.h>
#include <thrust/functional.h>
#include <thrust/sequence.h>
#include <thrust/iterator/discard_iterator.h>

#include "../neuralnet/cudahelpers.h"

__global__
void cudaChannelConcatKernel(
  float* inA,
  float* inB,
  float* out,
  int chwA,
  int chwB,
  int numBlocksA,
  int numBlocksB,
  int n
) {
  if(blockIdx.x < numBlocksA) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if(index < chwA) {
      int nchwA = n*chwA;
      int chwOut = (chwA+chwB);

      int aIdx = index;
      int outIdx = index;
      while(aIdx < nchwA) {
        out[outIdx] = inA[aIdx];
        aIdx += chwA;
        outIdx += chwOut;
      }
    }
  }
  else {
    int index = (blockIdx.x - numBlocksA) * blockDim.x + threadIdx.x;
    if(index < chwB) {
      int nchwB = n*chwB;
      int chwOut = (chwA+chwB);

      int bIdx = index;
      int outIdx = chwA+index;
      while(bIdx < nchwB) {
        out[outIdx] = inB[bIdx];
        bIdx += chwB;
        outIdx += chwOut;
      }
    }
  }
}

void customCudaChannelConcat(float* inA, float* inB, float* out, int chwA, int chwB, int n) {
  int blockSize = 128;
  int numBlocksA = (chwA + blockSize-1) / blockSize;
  int numBlocksB = (chwB + blockSize-1) / blockSize;
  int numBlocks = numBlocksA + numBlocksB;
  cudaChannelConcatKernel<<<numBlocks, blockSize>>>(inA,inB,out,chwA,chwB,numBlocksA,numBlocksB,n);
}



template <typename T>
struct linear_index_to_row_index : public thrust::unary_function<T,T> {
  T c;
  __host__ __device__ linear_index_to_row_index(T c) : c(c) {}
  __host__ __device__ T operator()(T i) { return i / c; }
};

void customCudaPoolRowsSum(float* in, float* out, int n, int c) {

  thrust::device_ptr<float> inThrust = thrust::device_pointer_cast(in);
  thrust::device_ptr<float> outThrust = thrust::device_pointer_cast(out);

  thrust::reduce_by_key(
    thrust::make_transform_iterator(thrust::counting_iterator<int>(0), linear_index_to_row_index<int>(c)),
    thrust::make_transform_iterator(thrust::counting_iterator<int>(0), linear_index_to_row_index<int>(c)) + (n*c),
    inThrust,
    thrust::make_discard_iterator(),
    outThrust
  );

}

void customCudaPoolRowsMax(float* in, float* out, int n, int c) {

  thrust::device_ptr<float> inThrust = thrust::device_pointer_cast(in);
  thrust::device_ptr<float> outThrust = thrust::device_pointer_cast(out);

  thrust::reduce_by_key(
    thrust::make_transform_iterator(thrust::counting_iterator<int>(0), linear_index_to_row_index<int>(c)),
    thrust::make_transform_iterator(thrust::counting_iterator<int>(0), linear_index_to_row_index<int>(c)) + (n*c),
    inThrust,
    thrust::make_discard_iterator(),
    outThrust,
    thrust::equal_to<int>(),
    thrust::maximum<float>()
  );

}
