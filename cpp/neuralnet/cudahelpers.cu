
#define CUDA_API_PER_THREAD_DEFAULT_STREAM

#include <cublas_v2.h>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/generate.h>
#include <thrust/reduce.h>
#include <thrust/functional.h>
#include <thrust/sequence.h>
#include <thrust/iterator/discard_iterator.h>

#include <stdexcept>

#include "../neuralnet/cudahelpers.h"

__global__
void channelConcatKernel(
  const float* inA,
  const float* inB,
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

void customCudaChannelConcat(const float* inA, const float* inB, float* out, int chwA, int chwB, int n) {
  //TODO maybe tune this number, it varies by GPU
  int blockSize = 256;
  int numBlocksA = (chwA + blockSize-1) / blockSize;
  int numBlocksB = (chwB + blockSize-1) / blockSize;
  int numBlocks = numBlocksA + numBlocksB;
  channelConcatKernel<<<numBlocks, blockSize>>>(inA,inB,out,chwA,chwB,numBlocksA,numBlocksB,n);
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

__global__
void nchwTransposeKernel(const float *in, float* out, int xSize, int ySize, int tileDim, int tileStride, int xySize)
{
  //+1 avoids bank conflicts
  extern __shared__ float tile[];
  int tileDimP1 = tileDim+1;

  int xIdx = blockIdx.x * tileDim + threadIdx.x;
  int yIdx = blockIdx.y * tileDim + threadIdx.y;
  int nc = blockIdx.z;
  if(xIdx < xSize) {
    for(int j = 0; j < tileDim && yIdx+j < ySize; j += tileStride) {
      int inIdx = xIdx + xSize * (yIdx+j) + xySize * nc;
      tile[(threadIdx.y+j)*tileDimP1 + threadIdx.x] = in[inIdx];
    }
  }

  __syncthreads();

  //Transpose idx
  int outXIdx = blockIdx.y * tileDim + threadIdx.x;
  int outYIdx = blockIdx.x * tileDim + threadIdx.y;

  if(outXIdx < ySize) {
    for(int j = 0; j < tileDim && outYIdx+j < xSize; j += tileStride) {
      int outIdx = outXIdx + ySize * (outYIdx+j) + xySize * nc;
      out[outIdx] = tile[threadIdx.x*tileDimP1 + threadIdx.y+j];
    }
  }
}

__global__
void nhwcTransposeKernel(const float *in, float* out, int xSize, int ySize, int cSize, int tileDim, int tileStride, int xycSize)
{
  //+1 reduces bank conflicts
  extern __shared__ float tile[];
  int tileDimP1 = tileDim+1;

  int xIdx = blockIdx.x * tileDim + threadIdx.x;
  int yIdx = blockIdx.y * tileDim + threadIdx.y;
  int cIdx = threadIdx.z;
  int n = blockIdx.z;
  if(xIdx < xSize) {
    for(int j = 0; j < tileDim && yIdx+j < ySize; j += tileStride) {
      int inIdx = cIdx + cSize * (xIdx + xSize * (yIdx+j)) + xycSize * n;
      tile[cIdx + cSize * ((threadIdx.y+j)*tileDimP1 + threadIdx.x)] = in[inIdx];
    }
  }

  __syncthreads();

  //Transpose idx
  int outXIdx = blockIdx.y * tileDim + threadIdx.x;
  int outYIdx = blockIdx.x * tileDim + threadIdx.y;

  if(outXIdx < ySize) {
    for(int j = 0; j < tileDim && outYIdx+j < xSize; j += tileStride) {
      int outIdx = cIdx + cSize * (outXIdx + ySize * (outYIdx+j)) + xycSize * n;
      out[outIdx] = tile[cIdx + cSize * (threadIdx.x*tileDimP1 + threadIdx.y+j)];
    }
  }
}

void customCudaNCHWTranspose(const float *in, float* out, int xSize, int ySize, int ncSize) {
  //TODO maybe tune these numbers, it varies by GPU
  //The first one should be the warp size, since it's set to what we need to avoid bank conflicts?
  //Or is it better to just make it xSize, to reduce overhead on top of 19x19?
  int targetNumThreads = 256;
  int tileDim = 32;
  int tileStride = targetNumThreads/tileDim;
  dim3 grid((xSize+tileDim-1)/tileDim,(ySize+tileDim-1)/tileDim,ncSize);
  dim3 threads(tileDim,tileStride,1);
  int sharedMemSize = sizeof(float)*tileDim*(tileDim+1);
  nchwTransposeKernel<<<grid,threads,sharedMemSize>>>(in,out,xSize,ySize,tileDim,tileStride,xSize*ySize);
}

void customCudaNHWCTranspose(const float *in, float* out, int xSize, int ySize, int cSize, int nSize) {
  if(cSize > 64)
    throw std::runtime_error("customCudaNHWCTranspose: cSize too large");
  //TODO maybe tune these numbers, it varies by GPU
  int targetNumThreads = 256;

  int tileDim = 1;
  while(tileDim * 2 * cSize <= targetNumThreads)
    tileDim *= 2;

  int tileStride = 1;
  if(tileDim > 32) {
    tileStride = tileDim / 32;
    tileDim = 32;
  }
  dim3 grid((xSize+tileDim-1)/tileDim,(ySize+tileDim-1)/tileDim,nSize);
  dim3 threads(tileDim,tileStride,cSize);
  int sharedMemSize = sizeof(float)*tileDim*(tileDim+1)*cSize;
  nhwcTransposeKernel<<<grid,threads,sharedMemSize>>>(in,out,xSize,ySize,cSize,tileDim,tileStride,xSize*ySize*cSize);
}

__global__
void mirrorKernel(const float *in, float* out, int batchSize, int mSize, int subSize)
{
  int subIdx = blockIdx.x * blockDim.x + threadIdx.x;
  int mIdx = blockIdx.y * blockDim.y + threadIdx.y;
  int batchIdx = blockIdx.z;
  if(subIdx < subSize && mIdx < mSize) {
    int inIdx = subIdx + subSize * (mIdx + mSize * batchIdx);
    int outIdx = subIdx + subSize * ((mSize-mIdx-1) + mSize * batchIdx);
    out[outIdx] = in[inIdx];
  }
}


void customCudaMirror(const float *in, float* out, int batchSize, int mSize, int subSize) {
  //TODO maybe tune these numbers, it varies by GPU
  int targetNumThreads = 256;

  int subThreads;
  int subBlocks;
  int mThreads;
  int mBlocks;

  if(subSize > targetNumThreads) {
    subThreads = targetNumThreads/2;
    subBlocks = (subSize + subThreads - 1) / subThreads;
    mThreads = 1;
    mBlocks = mSize;
  }
  else if(subSize > targetNumThreads/2) {
    subThreads = subSize;
    subBlocks = 1;
    mThreads = 1;
    mBlocks = mSize;
  }
  else {
    subThreads = subSize;
    subBlocks = 1;
    mThreads = targetNumThreads / subSize;
    mBlocks = (mSize + mThreads - 1) / mThreads;
  }

  dim3 grid(subBlocks,mBlocks,batchSize);
  dim3 threads(subThreads,mThreads,1);
  mirrorKernel<<<grid,threads>>>(in,out,batchSize,mSize,subSize);
}

void customCudaMirrorNCHW(const float *in, float* out, int batchSize, int cSize, int ySize, int xSize, bool mirrorY, bool mirrorX) {
  if(mirrorY && mirrorX)
    customCudaMirror(in,out,batchSize*cSize,ySize*xSize,1);
  else if(mirrorY)
    customCudaMirror(in,out,batchSize*cSize,ySize,xSize);
  else if(mirrorX)
    customCudaMirror(in,out,batchSize*cSize*ySize,xSize,1);
  else
    cudaMemcpyAsync(out,in,sizeof(float)*batchSize*cSize*ySize*xSize,cudaMemcpyDeviceToDevice);
}

void customCudaMirrorNHWC(const float *in, float* out, int batchSize, int ySize, int xSize, int cSize, bool mirrorY, bool mirrorX) {
  if(mirrorY && mirrorX)
    customCudaMirror(in,out,batchSize,ySize*xSize,cSize);
  else if(mirrorY)
    customCudaMirror(in,out,batchSize,ySize,xSize*cSize);
  else if(mirrorX)
    customCudaMirror(in,out,batchSize*ySize,xSize,cSize);
  else
    cudaMemcpyAsync(out,in,sizeof(float)*batchSize*ySize*xSize*cSize,cudaMemcpyDeviceToDevice);
}
