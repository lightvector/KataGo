
#define CUDA_API_PER_THREAD_DEFAULT_STREAM

#include <cuda_fp16.h>
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

#if __CUDA_ARCH__ >= 530
#define CUDA_SUPPORTS_FP16
#endif

//TODO maybe tune this number, it varies by GPU
static const int targetNumThreads = 256;

//--------------------------------------------------------------------------------------------------------------

template <typename T>
__global__
void channelConcatKernel(
  const T* inA,
  const T* inB,
  T* out,
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

template <typename T>
void customCudaChannelConcatTemplate(const T* inA, const T* inB, T* out, int chwA, int chwB, int n) {
  int blockSize = targetNumThreads;
  int numBlocksA = (chwA + blockSize-1) / blockSize;
  int numBlocksB = (chwB + blockSize-1) / blockSize;
  int numBlocks = numBlocksA + numBlocksB;
  channelConcatKernel<<<numBlocks, blockSize>>>(inA,inB,out,chwA,chwB,numBlocksA,numBlocksB,n);
}
template void customCudaChannelConcatTemplate<float>(const float* inA, const float* inB, float* out, int chwA, int chwB, int n);
template void customCudaChannelConcatTemplate<half>(const half* inA, const half* inB, half* out, int chwA, int chwB, int n);

void customCudaChannelConcat(const float* inA, const float* inB, float* out, int chwA, int chwB, int n) {
  customCudaChannelConcatTemplate<float>(inA,inB,out,chwA,chwB,n);
}
void customCudaChannelConcat(const half* inA, const half* inB, half* out, int chwA, int chwB, int n) {
  customCudaChannelConcatTemplate<half>(inA,inB,out,chwA,chwB,n);
}

//--------------------------------------------------------------------------------------------------------------

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

//--------------------------------------------------------------------------------------------------------------

__global__
void nchwTransposeKernel(const float *in, float* out, int xSize, int ySize, int tileDim, int tileStride, int xySize)
{
  //+1 avoids bank conflicts
  extern __shared__ float tileNCHW[];
  int tileDimP1 = tileDim+1;

  int xIdx = blockIdx.x * tileDim + threadIdx.x;
  int yIdx = blockIdx.y * tileDim + threadIdx.y;
  int nc = blockIdx.z;
  if(xIdx < xSize) {
    for(int j = 0; j < tileDim && yIdx+j < ySize; j += tileStride) {
      int inIdx = xIdx + xSize * (yIdx+j) + xySize * nc;
      tileNCHW[(threadIdx.y+j)*tileDimP1 + threadIdx.x] = in[inIdx];
    }
  }

  __syncthreads();

  //Transpose idx
  int outXIdx = blockIdx.y * tileDim + threadIdx.x;
  int outYIdx = blockIdx.x * tileDim + threadIdx.y;

  if(outXIdx < ySize) {
    for(int j = 0; j < tileDim && outYIdx+j < xSize; j += tileStride) {
      int outIdx = outXIdx + ySize * (outYIdx+j) + xySize * nc;
      out[outIdx] = tileNCHW[threadIdx.x*tileDimP1 + threadIdx.y+j];
    }
  }
}

__global__
void nhwcTransposeKernel(const float *in, float* out, int xSize, int ySize, int cSize, int tileDim, int tileStride, int xycSize)
{
  //+1 reduces bank conflicts
  extern __shared__ float tileNHWC[];
  int tileDimP1 = tileDim+1;

  int xIdx = blockIdx.x * tileDim + threadIdx.x;
  int yIdx = blockIdx.y * tileDim + threadIdx.y;
  int cIdx = threadIdx.z;
  int n = blockIdx.z;
  if(xIdx < xSize) {
    for(int j = 0; j < tileDim && yIdx+j < ySize; j += tileStride) {
      int inIdx = cIdx + cSize * (xIdx + xSize * (yIdx+j)) + xycSize * n;
      tileNHWC[cIdx + cSize * ((threadIdx.y+j)*tileDimP1 + threadIdx.x)] = in[inIdx];
    }
  }

  __syncthreads();

  //Transpose idx
  int outXIdx = blockIdx.y * tileDim + threadIdx.x;
  int outYIdx = blockIdx.x * tileDim + threadIdx.y;

  if(outXIdx < ySize) {
    for(int j = 0; j < tileDim && outYIdx+j < xSize; j += tileStride) {
      int outIdx = cIdx + cSize * (outXIdx + ySize * (outYIdx+j)) + xycSize * n;
      out[outIdx] = tileNHWC[cIdx + cSize * (threadIdx.x*tileDimP1 + threadIdx.y+j)];
    }
  }
}

__global__
void nchwTransposeHalfKernel(const half *in, half* out, int xSize, int ySize, int tileDim, int tileStride, int xySize)
{
  //+1 avoids bank conflicts
  extern __shared__ half tileNCHWHALF[];
  int tileDimP1 = tileDim+1;

  int xIdx = blockIdx.x * tileDim + threadIdx.x;
  int yIdx = blockIdx.y * tileDim + threadIdx.y;
  int nc = blockIdx.z;
  if(xIdx < xSize) {
    for(int j = 0; j < tileDim && yIdx+j < ySize; j += tileStride) {
      int inIdx = xIdx + xSize * (yIdx+j) + xySize * nc;
      tileNCHWHALF[(threadIdx.y+j)*tileDimP1 + threadIdx.x] = in[inIdx];
    }
  }

  __syncthreads();

  //Transpose idx
  int outXIdx = blockIdx.y * tileDim + threadIdx.x;
  int outYIdx = blockIdx.x * tileDim + threadIdx.y;

  if(outXIdx < ySize) {
    for(int j = 0; j < tileDim && outYIdx+j < xSize; j += tileStride) {
      int outIdx = outXIdx + ySize * (outYIdx+j) + xySize * nc;
      out[outIdx] = tileNCHWHALF[threadIdx.x*tileDimP1 + threadIdx.y+j];
    }
  }
}

__global__
void nhwcTransposeHalfKernel(const half *in, half* out, int xSize, int ySize, int cSize, int tileDim, int tileStride, int xycSize)
{
  //+1 reduces bank conflicts
  extern __shared__ half tileNHWCHALF[];
  int tileDimP1 = tileDim+1;

  int xIdx = blockIdx.x * tileDim + threadIdx.x;
  int yIdx = blockIdx.y * tileDim + threadIdx.y;
  int cIdx = threadIdx.z;
  int n = blockIdx.z;
  if(xIdx < xSize) {
    for(int j = 0; j < tileDim && yIdx+j < ySize; j += tileStride) {
      int inIdx = cIdx + cSize * (xIdx + xSize * (yIdx+j)) + xycSize * n;
      tileNHWCHALF[cIdx + cSize * ((threadIdx.y+j)*tileDimP1 + threadIdx.x)] = in[inIdx];
    }
  }

  __syncthreads();

  //Transpose idx
  int outXIdx = blockIdx.y * tileDim + threadIdx.x;
  int outYIdx = blockIdx.x * tileDim + threadIdx.y;

  if(outXIdx < ySize) {
    for(int j = 0; j < tileDim && outYIdx+j < xSize; j += tileStride) {
      int outIdx = cIdx + cSize * (outXIdx + ySize * (outYIdx+j)) + xycSize * n;
      out[outIdx] = tileNHWCHALF[cIdx + cSize * (threadIdx.x*tileDimP1 + threadIdx.y+j)];
    }
  }
}

static void sharedNCHWTranspose(const void *in, void* out, int xSize, int ySize, int ncSize, bool isHalf) {
  if(ncSize > 65536)
    throw std::runtime_error("customCudaNCHWTranspose: ncSize too large");

  //TODO maybe tune these numbers, it varies by GPU
  //The first one should be the warp size, since it's set to what we need to avoid bank conflicts?
  //Or is it better to just make it xSize, to reduce overhead on top of 19x19?
  int tileDim = 32;
  int tileStride = targetNumThreads/tileDim;
  dim3 grid((xSize+tileDim-1)/tileDim,(ySize+tileDim-1)/tileDim,ncSize);
  dim3 threads(tileDim,tileStride,1);
  if(isHalf) {
    int sharedMemSize = sizeof(half)*tileDim*(tileDim+1);
    nchwTransposeHalfKernel<<<grid,threads,sharedMemSize>>>((const half*)in,(half*)out,xSize,ySize,tileDim,tileStride,xSize*ySize);
  }
  else {
    int sharedMemSize = sizeof(float)*tileDim*(tileDim+1);
    nchwTransposeKernel<<<grid,threads,sharedMemSize>>>((const float*)in,(float*)out,xSize,ySize,tileDim,tileStride,xSize*ySize);
  }
}
void customCudaNCHWTranspose(const float *in, float* out, int xSize, int ySize, int ncSize) {
  sharedNCHWTranspose(in,out,xSize,ySize,ncSize,false);
}
void customCudaNCHWTranspose(const half *in, half* out, int xSize, int ySize, int ncSize) {
  sharedNCHWTranspose(in,out,xSize,ySize,ncSize,true);
}

void sharedNHWCTranspose(const void *in, void* out, int xSize, int ySize, int cSize, int nSize, bool isHalf) {
  if(cSize > 64)
    throw std::runtime_error("customCudaNHWCTranspose: cSize too large");

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

  if(isHalf) {
    int sharedMemSize = sizeof(half)*tileDim*(tileDim+1)*cSize;
    nhwcTransposeHalfKernel<<<grid,threads,sharedMemSize>>>((const half*)in,(half*)out,xSize,ySize,cSize,tileDim,tileStride,xSize*ySize*cSize);
  }
  else {
    int sharedMemSize = sizeof(float)*tileDim*(tileDim+1)*cSize;
    nhwcTransposeKernel<<<grid,threads,sharedMemSize>>>((const float*)in,(float*)out,xSize,ySize,cSize,tileDim,tileStride,xSize*ySize*cSize);
  }
}
void customCudaNHWCTranspose(const float *in, float* out, int xSize, int ySize, int cSize, int nSize) {
  sharedNHWCTranspose(in,out,xSize,ySize,cSize,nSize,false);
}
void customCudaNHWCTranspose(const half *in, half* out, int xSize, int ySize, int cSize, int nSize) {
  sharedNHWCTranspose(in,out,xSize,ySize,cSize,nSize,true);
}

//--------------------------------------------------------------------------------------------------------------


template <typename T>
__global__
void mirrorKernel(const T *in, T* out, int mSize, int subSize)
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

template <typename T>
void customCudaMirrorTemplate(const T *in, T* out, int batchSize, int mSize, int subSize) {
  if(batchSize > 65536)
    throw std::runtime_error("customCudaMirror: batchSize too large");
  if(mSize > 65536)
    throw std::runtime_error("customCudaMirror: mSize too large");

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
  mirrorKernel<<<grid,threads>>>(in,out,mSize,subSize);
}

template <typename T>
void customCudaMirrorNCHWTemplate(const T *in, T* out, int batchSize, int cSize, int ySize, int xSize, bool mirrorY, bool mirrorX) {
  if(mirrorY && mirrorX)
    customCudaMirrorTemplate(in,out,batchSize*cSize,ySize*xSize,1);
  else if(mirrorY)
    customCudaMirrorTemplate(in,out,batchSize*cSize,ySize,xSize);
  else if(mirrorX)
    customCudaMirrorTemplate(in,out,batchSize*cSize*ySize,xSize,1);
  else
    cudaMemcpyAsync(out,in,sizeof(T)*batchSize*cSize*ySize*xSize,cudaMemcpyDeviceToDevice);
}

template <typename T>
void customCudaMirrorNHWCTemplate(const T *in, T* out, int batchSize, int ySize, int xSize, int cSize, bool mirrorY, bool mirrorX) {
  if(mirrorY && mirrorX)
    customCudaMirrorTemplate(in,out,batchSize,ySize*xSize,cSize);
  else if(mirrorY)
    customCudaMirrorTemplate(in,out,batchSize,ySize,xSize*cSize);
  else if(mirrorX)
    customCudaMirrorTemplate(in,out,batchSize*ySize,xSize,cSize);
  else
    cudaMemcpyAsync(out,in,sizeof(T)*batchSize*ySize*xSize*cSize,cudaMemcpyDeviceToDevice);
}

void customCudaMirror(const float *in, float* out, int batchSize, int mSize, int subSize) {
  customCudaMirrorTemplate<float>(in,out,batchSize,mSize,subSize);
}
void customCudaMirrorNCHW(const float *in, float* out, int batchSize, int cSize, int ySize, int xSize, bool mirrorY, bool mirrorX) {
  customCudaMirrorNCHWTemplate<float>(in,out,batchSize,cSize,ySize,xSize,mirrorY,mirrorX);
}
void customCudaMirrorNHWC(const float *in, float* out, int batchSize, int ySize, int xSize, int cSize, bool mirrorY, bool mirrorX) {
  customCudaMirrorNHWCTemplate<float>(in,out,batchSize,ySize,xSize,cSize,mirrorY,mirrorX);
}

void customCudaMirror(const half *in, half* out, int batchSize, int mSize, int subSize) {
  customCudaMirrorTemplate<half>(in,out,batchSize,mSize,subSize);
}
void customCudaMirrorNCHW(const half *in, half* out, int batchSize, int cSize, int ySize, int xSize, bool mirrorY, bool mirrorX) {
  customCudaMirrorNCHWTemplate<half>(in,out,batchSize,cSize,ySize,xSize,mirrorY,mirrorX);
}
void customCudaMirrorNHWC(const half *in, half* out, int batchSize, int ySize, int xSize, int cSize, bool mirrorY, bool mirrorX) {
  customCudaMirrorNHWCTemplate<half>(in,out,batchSize,ySize,xSize,cSize,mirrorY,mirrorX);
}


//--------------------------------------------------------------------------------------------------------------

__global__
void copyToHalfKernel(const float *in, half* out, int n)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if(idx < n) {
    out[idx] = __float2half(in[idx]);
  }
}
__global__
void copyFromHalfKernel(const half *in, float* out, int n)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if(idx < n) {
    out[idx] = __half2float(in[idx]);
  }
}

void customCudaCopyToHalf(const float* in, half* out, int n) {
  int blockSize = targetNumThreads;
  int numBlocks = (n+blockSize-1)/blockSize;
  copyToHalfKernel<<<numBlocks, blockSize>>>(in,out,n);
}
void customCudaCopyFromHalf(const half* in, float* out, int n) {
  int blockSize = targetNumThreads;
  int numBlocks = (n+blockSize-1)/blockSize;
  copyFromHalfKernel<<<numBlocks, blockSize>>>(in,out,n);
}

//--------------------------------------------------------------------------------------------------------------


#ifdef CUDA_SUPPORTS_FP16
__global__
void addTensorInplaceHalfKernel(half *buf, const half* biases, int nSize)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if(idx < nSize) {
    buf[idx] = __hadd(buf[idx],biases[idx]);
  }
}
#else
__global__
void addTensorInplaceHalfKernel(half *buf, const half* biases, int nSize)
{
  //Do nothing, FP16 not supported
}
#endif
void customCudaAddTensorInplace(half* buf, const half* biases, int nSize) {
  int blockSize = targetNumThreads;
  int numBlocks = (nSize+blockSize-1)/blockSize;
  addTensorInplaceHalfKernel<<<numBlocks, blockSize>>>(buf,biases,nSize);
}

//--------------------------------------------------------------------------------------------------------------


__global__
void addCBiasInplaceNCKernel(float *buf, const float* biases, int nSize, int cSize)
{
  int cIdx = blockIdx.x * blockDim.x + threadIdx.x;
  int nIdx = blockIdx.y * blockDim.y + threadIdx.y;
  if(cIdx < cSize && nIdx < nSize) {
    int idx = nIdx * cSize + cIdx;
    buf[idx] = buf[idx] + biases[cIdx];
  }
}
#ifdef CUDA_SUPPORTS_FP16
__global__
void addCBiasInplaceNCHalfKernel(half *buf, const half* biases, int nSize, int cSize)
{
  int cIdx = blockIdx.x * blockDim.x + threadIdx.x;
  int nIdx = blockIdx.y * blockDim.y + threadIdx.y;
  if(cIdx < cSize && nIdx < nSize) {
    int idx = nIdx * cSize + cIdx;
    buf[idx] = __hadd(buf[idx],biases[cIdx]);
  }
}
#else
__global__
void addCBiasInplaceNCHalfKernel(half *buf, const half* biases, int nSize, int cSize)
{
  //Do nothing, FP16 not supported
}
#endif

void sharedAddCBiasInplaceNC(void* buf, const void* biases, int nSize, int cSize, bool isHalf) {
  int cThreads;
  int cBlocks;
  int nThreads;
  int nBlocks;

  if(cSize > targetNumThreads) {
    cThreads = targetNumThreads/2;
    cBlocks = (cSize + cThreads - 1) / cThreads;
    nThreads = 1;
    nBlocks = nSize;
  }
  else if(cSize > targetNumThreads/2) {
    cThreads = cSize;
    cBlocks = 1;
    nThreads = 1;
    nBlocks = nSize;
  }
  else {
    cThreads = cSize;
    cBlocks = 1;
    nThreads = targetNumThreads / cSize;
    nBlocks = (nSize + nThreads - 1) / nThreads;
  }

  if(nBlocks > 65536)
    throw std::runtime_error("customCudaAddCBiasInplaceNC: nSize too large given cSize");

  dim3 grid(cBlocks,nBlocks,1);
  dim3 threads(cThreads,nThreads,1);

  if(isHalf)
    addCBiasInplaceNCHalfKernel<<<grid,threads>>>((half*)buf,(const half*)biases,nSize,cSize);
  else
    addCBiasInplaceNCKernel<<<grid,threads>>>((float*)buf,(const float*)biases,nSize,cSize);
}

void customCudaAddCBiasInplaceNC(float* buf, const float* biases, int nSize, int cSize) {
  sharedAddCBiasInplaceNC(buf,biases,nSize,cSize,false);
}
void customCudaAddCBiasInplaceNC(half* buf, const half* biases, int nSize, int cSize) {
  sharedAddCBiasInplaceNC(buf,biases,nSize,cSize,true);
}

//--------------------------------------------------------------------------------------------------------------

__global__
void addNCBiasInplaceNCHWKernel(float *buf, const float* biases, int cSize, int sSize)
{
  int sIdx = blockIdx.x * blockDim.x + threadIdx.x;
  int cIdx = blockIdx.y * blockDim.y + threadIdx.y;
  int nIdx = blockIdx.z;
  if(cIdx < cSize && sIdx < sSize) {
    int ncIdx = nIdx * cSize + cIdx;
    int idx = ncIdx * sSize + sIdx;
    buf[idx] = buf[idx] + biases[ncIdx];
  }
}
#ifdef CUDA_SUPPORTS_FP16
__global__
void addNCBiasInplaceNCHWHalfKernel(half *buf, const half* biases, int cSize, int sSize)
{
  int sIdx = blockIdx.x * blockDim.x + threadIdx.x;
  int cIdx = blockIdx.y * blockDim.y + threadIdx.y;
  int nIdx = blockIdx.z;
  if(cIdx < cSize && sIdx < sSize) {
    int ncIdx = nIdx * cSize + cIdx;
    int idx = ncIdx * sSize + sIdx;
    buf[idx] = __hadd(buf[idx],biases[ncIdx]);
  }
}
#else
__global__
void addNCBiasInplaceNCHWHalfKernel(half *buf, const half* biases, int cSize, int sSize) {
  //Do nothing, FP16 not supported
}
#endif

void sharedAddNCBiasInplaceNCHW(void *buf, const void* biases, int nSize, int cSize, int xySize, bool isHalf) {
  if(nSize > 65536)
    throw std::runtime_error("customCudaAddNCBiasInplaceNCHW: nSize too large");
  if(cSize > 65536)
    throw std::runtime_error("customCudaAddNCBiasInplaceNCHW: cSize too large");

  int sSize = xySize;
  int sThreads;
  int sBlocks;
  int cThreads;
  int cBlocks;

  if(sSize > targetNumThreads) {
    sThreads = targetNumThreads/2;
    sBlocks = (sSize + sThreads - 1) / sThreads;
    cThreads = 1;
    cBlocks = cSize;
  }
  else if(sSize > targetNumThreads/2) {
    sThreads = sSize;
    sBlocks = 1;
    cThreads = 1;
    cBlocks = cSize;
  }
  else {
    sThreads = sSize;
    sBlocks = 1;
    cThreads = targetNumThreads / sSize;
    cBlocks = (cSize + cThreads - 1) / cThreads;
  }

  dim3 grid(sBlocks,cBlocks,nSize);
  dim3 threads(sThreads,cThreads,1);
  if(isHalf)
    addNCBiasInplaceNCHWHalfKernel<<<grid,threads>>>((half*)buf,(const half*)biases,cSize,sSize);
  else
    addNCBiasInplaceNCHWKernel<<<grid,threads>>>((float*)buf,(const float*)biases,cSize,sSize);
}

void customCudaAddNCBiasInplaceNCHW(float *buf, const float* biases, int nSize, int cSize, int xySize) {
  sharedAddNCBiasInplaceNCHW(buf,biases,nSize,cSize,xySize,false);
}
void customCudaAddNCBiasInplaceNCHW(half *buf, const half* biases, int nSize, int cSize, int xySize) {
  sharedAddNCBiasInplaceNCHW(buf,biases,nSize,cSize,xySize,true);
}

//--------------------------------------------------------------------------------------------------------------

#ifdef CUDA_SUPPORTS_FP16
__global__
void applyCScaleBiasNCHWHalfKernel(const half *in, half* out, const half* scale, const half* biases, int cSize, int sSize)
{
  int sIdx = blockIdx.x * blockDim.x + threadIdx.x;
  int cIdx = blockIdx.y * blockDim.y + threadIdx.y;
  int nIdx = blockIdx.z;
  if(cIdx < cSize && sIdx < sSize) {
    int idx = (nIdx * cSize + cIdx) * sSize + sIdx;
    out[idx] = __hfma(in[idx],scale[cIdx],biases[cIdx]);
  }
}
#else
__global__
void applyCScaleBiasNCHWHalfKernel(const half *in, half* out, const half* scale, const half* biases, int cSize, int sSize)
{
  //Do nothing, FP16 not supported
}
#endif

void customCudaApplyCScaleBiasNCHW(const half* in, half* out, const half* scale, const half* biases, int nSize, int cSize, int xySize) {
  if(nSize > 65536)
    throw std::runtime_error("customCudaApplyCScaleBiasNCHW: nSize too large");
  if(cSize > 65536)
    throw std::runtime_error("customCudaApplyCScaleBiasNCHW: cSize too large");

  int sSize = xySize;
  int sThreads;
  int sBlocks;
  int cThreads;
  int cBlocks;

  if(sSize > targetNumThreads) {
    sThreads = targetNumThreads/2;
    sBlocks = (sSize + sThreads - 1) / sThreads;
    cThreads = 1;
    cBlocks = cSize;
  }
  else if(sSize > targetNumThreads/2) {
    sThreads = sSize;
    sBlocks = 1;
    cThreads = 1;
    cBlocks = cSize;
  }
  else {
    sThreads = sSize;
    sBlocks = 1;
    cThreads = targetNumThreads / sSize;
    cBlocks = (cSize + cThreads - 1) / cThreads;
  }

  dim3 grid(sBlocks,cBlocks,nSize);
  dim3 threads(sThreads,cThreads,1);
  applyCScaleBiasNCHWHalfKernel<<<grid,threads>>>(in,out,scale,biases,cSize,sSize);
}
