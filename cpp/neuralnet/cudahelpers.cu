
#include "../neuralnet/cudahelpers.h"

#include <stdexcept>

#if __CUDA_ARCH__ >= 530
#define CUDA_SUPPORTS_FP16
#endif

//TODO maybe tune this number, it varies by GPU
static const int targetNumThreads = 512;

void splitThreadsAcrossDim01(int dim0Size, int dim1Size, int& threads0, int& blocks0, int& threads1, int& blocks1) {
  if(dim0Size > targetNumThreads) {
    threads0 = targetNumThreads/2;
    blocks0 = (dim0Size + threads0 - 1) / threads0;
    threads1 = 1;
    blocks1 = dim1Size;
  }
  else if(dim0Size > targetNumThreads/2) {
    threads0 = dim0Size;
    blocks0 = 1;
    threads1 = 1;
    blocks1 = dim1Size;
  }
  else {
    threads0 = dim0Size;
    blocks0 = 1;
    threads1 = targetNumThreads / dim0Size;
    blocks1 = (dim1Size + threads1 - 1) / threads1;
  }
}

__forceinline__ __device__ float mishf(float a) {
  return a * tanhf(a < 20.0f ? log1pf(expf(a)) : a);
}

#ifdef CUDA_SUPPORTS_FP16
__forceinline__ __device__ half mishh(half h) {
  float a = __half2float(h);
  return __float2half(a * tanhf(a < 20.0f ? log1pf(expf(a)) : a));
}
#endif

#define SQRT2 1.41421356237f

__forceinline__ __device__ float geluf(float a) {
  return a * tanhf(a < 20.0f ? log1pf(expf(a)) : a);
  return a * 0.5 * (1.0 + erff(a / SQRT2));
}

#ifdef CUDA_SUPPORTS_FP16
__forceinline__ __device__ half geluh(half h) {
  float a = __half2float(h);
  return __float2half(a * 0.5 * (1.0 + erff(a / SQRT2)));
}
#endif

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
__global__
void extractChannel0KernelNHWC(const T *in, T* out, int nhwSize, int cSize)
{
  int nhwIdx = blockIdx.x * blockDim.x + threadIdx.x;
  if(nhwIdx < nhwSize) {
    out[nhwIdx] = in[nhwIdx*cSize];
  }
}
template <typename T>
void customCudaChannel0ExtractNHWCTemplate(const T *in, T* out, int n, int hw, int c) {
  int nhw = n*hw;
  int blockSize = targetNumThreads;
  int numBlocks = (nhw+blockSize-1)/blockSize;
  extractChannel0KernelNHWC<<<numBlocks,blockSize>>>(in,out,nhw,c);
}

template <typename T>
__global__
void extractChannel0KernelNCHW(const T *in, T* out, int nSize, int cSize, int hwSize)
{
  int hwIdx = blockIdx.x * blockDim.x + threadIdx.x;
  int nIdx = blockIdx.y * blockDim.y + threadIdx.y;
  if(hwIdx < hwSize && nIdx < nSize) {
    out[nIdx * hwSize + hwIdx] = in[nIdx * cSize * hwSize + hwIdx];
  }
}
template <typename T>
void customCudaChannel0ExtractNCHWTemplate(const T *in, T* out, int nSize, int cSize, int hwSize) {
  int hwThreads;
  int hwBlocks;
  int nThreads;
  int nBlocks;
  splitThreadsAcrossDim01(hwSize, nSize, hwThreads, hwBlocks, nThreads, nBlocks);

  if(nBlocks > 65536)
    throw std::runtime_error("customCudaChannel0ExtractNCHW: nSize too large given hwSize");

  dim3 grid(hwBlocks,nBlocks,1);
  dim3 threads(hwThreads,nThreads,1);
  extractChannel0KernelNCHW<<<grid,threads>>>(in,out,nSize,cSize,hwSize);
}

void customCudaChannel0ExtractNCHW(const float* in, float* out, int n, int c, int hw) {
  customCudaChannel0ExtractNCHWTemplate<float>(in,out,n,c,hw);
}
void customCudaChannel0ExtractNCHW(const half* in, half* out, int n, int c, int hw) {
  customCudaChannel0ExtractNCHWTemplate<half>(in,out,n,c,hw);
}
void customCudaChannel0ExtractNHWC(const float* in, float* out, int n, int hw, int c) {
  customCudaChannel0ExtractNHWCTemplate<float>(in,out,n,hw,c);
}
void customCudaChannel0ExtractNHWC(const half* in, half* out, int n, int hw, int c) {
  customCudaChannel0ExtractNHWCTemplate<half>(in,out,n,hw,c);
}

//--------------------------------------------------------------------------------------------------------------

__global__
void sumChannelsNCHWKernel(const float* in, float* out, int cSize, int xySize, float scaleSum)
{
  extern __shared__ float sumPoolNCHWShared[];
  int xyId = threadIdx.x;
  int xyBlockDim = blockDim.x;
  int cId = threadIdx.y;
  int cBlockDim = blockDim.y;
  int cIdx = blockIdx.y * cBlockDim + cId;
  int nIdx = blockIdx.z;

  int xycSize = xySize*cSize;
  int sharedIdx = xyId + cId * xyBlockDim;

  float acc = 0.0f;
  if(cIdx < cSize) {
    int xyIdx = xyId;
    while(xyIdx < xySize) {
      acc += in[xyIdx + cIdx * xySize + nIdx * xycSize];
      xyIdx += xyBlockDim;
    }
    sumPoolNCHWShared[sharedIdx] = acc;
  }
  __syncthreads();

  for(int s = xyBlockDim>>1; s > 0; s >>= 1) {
    if(xyId < s) {
      sumPoolNCHWShared[sharedIdx] += sumPoolNCHWShared[sharedIdx + s];
    }
    __syncthreads();
  }
  if(xyId == 0 && cIdx < cSize)
    out[cIdx + nIdx * cSize] = sumPoolNCHWShared[sharedIdx] * scaleSum;
}
__global__
void valueHeadPoolChannelsNCHWKernel(const float* in, float* out, int nSize, int cSize, int xySize, const float* maskSum)
{
  extern __shared__ float sumPoolNCHWShared[];
  int xyId = threadIdx.x;
  int xyBlockDim = blockDim.x;
  int cId = threadIdx.y;
  int cBlockDim = blockDim.y;
  int cIdx = blockIdx.y * cBlockDim + cId;
  int nIdx = blockIdx.z;

  int xycSize = xySize*cSize;
  int sharedIdx = xyId + cId * xyBlockDim;

  float acc = 0.0f;
  if(cIdx < cSize) {
    int xyIdx = xyId;
    while(xyIdx < xySize) {
      acc += in[xyIdx + cIdx * xySize + nIdx * xycSize];
      xyIdx += xyBlockDim;
    }
    sumPoolNCHWShared[sharedIdx] = acc;
  }
  __syncthreads();

  for(int s = xyBlockDim>>1; s > 0; s >>= 1) {
    if(xyId < s) {
      sumPoolNCHWShared[sharedIdx] += sumPoolNCHWShared[sharedIdx + s];
    }
    __syncthreads();
  }
  if(xyId == 0 && cIdx < cSize) {
    float sum = sumPoolNCHWShared[sharedIdx];
    float div = maskSum[nIdx];
    float sqrtdiv = sqrt(div);
    float mean = sum/div;
    out[cIdx + nIdx * cSize*3] = mean;
    out[cIdx + nIdx * cSize*3 + cSize] = mean * (sqrtdiv - 14.0f) * 0.1f;
    out[cIdx + nIdx * cSize*3 + cSize*2] = mean * ((sqrtdiv - 14.0f) * (sqrtdiv - 14.0f) * 0.01f - 0.1f);
  }
}
__global__
void gPoolChannelsNCHWKernel(const float* in, float* out, int cSize, int xySize, const float* maskSum, int sharedMemElts)
{
  extern __shared__ float poolNCHWShared[];
  float* sumShared = (float*)poolNCHWShared;
  float* maxShared = (float*)poolNCHWShared + sharedMemElts;

  int xyId = threadIdx.x;
  int xyBlockDim = blockDim.x;
  int cId = threadIdx.y;
  int cBlockDim = blockDim.y;
  int cIdx = blockIdx.y * cBlockDim + cId;
  int nIdx = blockIdx.z;

  int xycSize = xySize*cSize;
  int sharedIdx = xyId + cId * xyBlockDim;

  if(cIdx < cSize) {
    float accSum = 0.0f;
    float accMax = -1.0f;
    int xyIdx = xyId;
    while(xyIdx < xySize) {
      float a = in[xyIdx + cIdx * xySize + nIdx * xycSize];
      accSum += a;
      accMax = fmaxf(accMax, a);
      xyIdx += xyBlockDim;
    }
    sumShared[sharedIdx] = accSum;
    maxShared[sharedIdx] = accMax;
  }
  __syncthreads();

  for(int s = xyBlockDim>>1; s > 0; s >>= 1) {
    if(xyId < s) {
      sumShared[sharedIdx] += sumShared[sharedIdx + s];
      maxShared[sharedIdx] = fmaxf(maxShared[sharedIdx], maxShared[sharedIdx + s]);
    }
    __syncthreads();
  }
  if(xyId == 0 && cIdx < cSize) {
    float sum = sumShared[sharedIdx];
    float div = maskSum[nIdx];
    float sqrtdiv = sqrt(div);
    float mean = sum/div;

    out[cIdx + nIdx * (cSize*3)] = mean;
    out[cIdx + nIdx * (cSize*3) + cSize] = mean * (sqrtdiv - 14.0f) * 0.1f;
    out[cIdx + nIdx * (cSize*3) + cSize*2] = maxShared[sharedIdx];
  }
}
__global__
void gPoolChannelsNCHWMaskKernel(const float* in, float* out, int cSize, int xySize, const float* mask, const float* maskSum, int sharedMemElts)
{
  extern __shared__ float poolNCHWShared[];
  float* sumShared = (float*)poolNCHWShared;
  float* maxShared = (float*)poolNCHWShared + sharedMemElts;

  int xyId = threadIdx.x;
  int xyBlockDim = blockDim.x;
  int cId = threadIdx.y;
  int cBlockDim = blockDim.y;
  int cIdx = blockIdx.y * cBlockDim + cId;
  int nIdx = blockIdx.z;

  int xycSize = xySize*cSize;
  int sharedIdx = xyId + cId * xyBlockDim;

  if(cIdx < cSize) {
    float accSum = 0.0f;
    float accMax = -1.0f;
    int xyIdx = xyId;
    while(xyIdx < xySize) {
      float a = in[xyIdx + cIdx * xySize + nIdx * xycSize];
      accSum += a;
      // Init to -1.0 above and + mask - 1.0 is because it will effectively make all padded space into -1.0
      // which is lower than the lowest value that any current activation function will produce.
      // so the max over all valid spaces will the same as the mask over all spaces including padding
      // We're relying on all padded space being equal to 0 because this gpool only ever follows a BN+Activate with a mask.
      accMax = fmaxf(accMax, a + (mask[xyIdx + nIdx * xySize] - 1.0f));
      xyIdx += xyBlockDim;
    }
    sumShared[sharedIdx] = accSum;
    maxShared[sharedIdx] = accMax;
  }
  __syncthreads();

  for(int s = xyBlockDim>>1; s > 0; s >>= 1) {
    if(xyId < s) {
      sumShared[sharedIdx] += sumShared[sharedIdx + s];
      maxShared[sharedIdx] = fmaxf(maxShared[sharedIdx], maxShared[sharedIdx + s]);
    }
    __syncthreads();
  }
  if(xyId == 0 && cIdx < cSize) {
    float sum = sumShared[sharedIdx];
    float div = maskSum[nIdx];
    float sqrtdiv = sqrt(div);
    float mean = sum/div;

    out[cIdx + nIdx * (cSize*3)] = mean;
    out[cIdx + nIdx * (cSize*3) + cSize] = mean * (sqrtdiv - 14.0f) * 0.1f;
    out[cIdx + nIdx * (cSize*3) + cSize*2] = maxShared[sharedIdx];
  }
}

void customCudaPoolRowsSumNCHW(const float* in, float* out, int nSize, int cSize, int xySize, float scaleSum) {
  if(nSize > 65536)
    throw std::runtime_error("customCudaPoolRowsSumNCHW: nSize too large");
  if(cSize > 65536)
    throw std::runtime_error("customCudaPoolRowsSumNCHW: cSize too large");

  //Use up as many threads as possible along the xy dimension.
  int xyThreads = 1;
  while(xyThreads < targetNumThreads && xyThreads < xySize/2)
    xyThreads *= 2;

  //Distribute the extra threads along the c dimension.
  int cThreads = (targetNumThreads < xyThreads) ? 1 : (targetNumThreads / xyThreads);
  int cBlocks = (cSize + cThreads - 1) / cThreads;

  //We need one shared memory spot per thread
  int sharedMemSize = sizeof(float) * cThreads * xyThreads;

  dim3 grid(1,cBlocks,nSize);
  dim3 threads(xyThreads,cThreads,1);
  sumChannelsNCHWKernel<<<grid,threads,sharedMemSize>>>(in,out,cSize,xySize,scaleSum);
}
void customCudaValueHeadPoolNCHW(const float* in, float* out, int nSize, int cSize, int xySize, const float* maskSum) {
  if(nSize > 65536)
    throw std::runtime_error("customCudaValueHeadPoolNCHW: nSize too large");
  if(cSize > 65536)
    throw std::runtime_error("customCudaValueHeadPoolNCHW: cSize too large");

  //Use up as many threads as possible along the xy dimension.
  int xyThreads = 1;
  while(xyThreads < targetNumThreads && xyThreads < xySize/2)
    xyThreads *= 2;

  //Distribute the extra threads along the c dimension.
  int cThreads = (targetNumThreads < xyThreads) ? 1 : (targetNumThreads / xyThreads);
  int cBlocks = (cSize + cThreads - 1) / cThreads;

  //We need one shared memory spot per thread
  int sharedMemSize = sizeof(float) * cThreads * xyThreads;

  dim3 grid(1,cBlocks,nSize);
  dim3 threads(xyThreads,cThreads,1);
  valueHeadPoolChannelsNCHWKernel<<<grid,threads,sharedMemSize>>>(in,out,nSize,cSize,xySize,maskSum);
}
void customCudaPoolRowsGPoolNCHW(const float* in, float* out, int nSize, int cSize, int xySize, const float* mask, const float* maskSum) {
  if(nSize > 65536)
    throw std::runtime_error("customCudaPoolRowsGPoolNCHW: nSize too large");
  if(cSize > 65536)
    throw std::runtime_error("customCudaPoolRowsGPoolNCHW: cSize too large");

  //Use up as many threads as possible along the xy dimension.
  int xyThreads = 1;
  while(xyThreads < targetNumThreads && xyThreads < xySize/2)
    xyThreads *= 2;

  //Distribute the extra threads along the c dimension.
  int cThreads = (targetNumThreads < xyThreads) ? 1 : (targetNumThreads / xyThreads);
  int cBlocks = (cSize + cThreads - 1) / cThreads;

  //We need one shared memory spot per thread, and then we double it because we need both sum and max.
  //We also make sure it's a power of two to address any alignment concerns.
  int sharedMemElts = 128;
  while(sharedMemElts < cThreads * xyThreads)
    sharedMemElts *= 2;
  int sharedMemSize = sizeof(float) * sharedMemElts * 2;

  dim3 grid(1,cBlocks,nSize);
  dim3 threads(xyThreads,cThreads,1);
  if(mask != NULL)
    gPoolChannelsNCHWMaskKernel<<<grid,threads,sharedMemSize>>>(in,out,cSize,xySize,mask,maskSum,sharedMemElts);
  else
    gPoolChannelsNCHWKernel<<<grid,threads,sharedMemSize>>>(in,out,cSize,xySize,maskSum,sharedMemElts);
}

//--------------------------------------------------------------------------------------------------------------

__global__
void gPoolChannelsNCHWHalfKernel(const half* in, half* out, int cSize, int xySize, const float* maskSum, int sharedMemElts)
{
#ifdef CUDA_SUPPORTS_FP16
  extern __shared__ float poolNCHWShared[];
  float* sumShared = (float*)poolNCHWShared;
  float* maxShared = (float*)poolNCHWShared + sharedMemElts;

  int xyId = threadIdx.x;
  int xyBlockDim = blockDim.x;
  int cId = threadIdx.y;
  int cBlockDim = blockDim.y;
  int cIdx = blockIdx.y * cBlockDim + cId;
  int nIdx = blockIdx.z;

  int xycSize = xySize*cSize;
  int sharedIdx = xyId + cId * xyBlockDim;

  if(cIdx < cSize) {
    float accSum = 0.0f;
    float accMax = -1.0f;
    int xyIdx = xyId;
    while(xyIdx < xySize) {
      float a = __half2float(in[xyIdx + cIdx * xySize + nIdx * xycSize]);
      accSum += a;
      accMax = fmaxf(accMax, a);
      xyIdx += xyBlockDim;
    }
    sumShared[sharedIdx] = accSum;
    maxShared[sharedIdx] = accMax;
  }
  __syncthreads();

  for(int s = xyBlockDim>>1; s > 0; s >>= 1) {
    if(xyId < s) {
      sumShared[sharedIdx] += sumShared[sharedIdx + s];
      maxShared[sharedIdx] = fmaxf(maxShared[sharedIdx], maxShared[sharedIdx + s]);
    }
    __syncthreads();
  }
  if(xyId == 0 && cIdx < cSize) {
    float sum = sumShared[sharedIdx];
    float div = maskSum[nIdx];
    float sqrtdiv = sqrt(div);
    float mean = sum/div;

    out[cIdx + nIdx * (cSize*3)] = __float2half(mean);
    out[cIdx + nIdx * (cSize*3) + cSize] = __float2half(mean * (sqrtdiv - 14.0f) * 0.1f);
    out[cIdx + nIdx * (cSize*3) + cSize*2] = __float2half(maxShared[sharedIdx]);
  }
#else
  //Do nothing, FP16 not supported
#endif
}
__global__
void gPoolChannelsNCHWHalfMaskKernel(const half* in, half* out, int cSize, int xySize, const half* mask, const float* maskSum, int sharedMemElts)
{
#ifdef CUDA_SUPPORTS_FP16
  extern __shared__ float poolNCHWShared[];
  float* sumShared = (float*)poolNCHWShared;
  float* maxShared = (float*)poolNCHWShared + sharedMemElts;

  int xyId = threadIdx.x;
  int xyBlockDim = blockDim.x;
  int cId = threadIdx.y;
  int cBlockDim = blockDim.y;
  int cIdx = blockIdx.y * cBlockDim + cId;
  int nIdx = blockIdx.z;

  int xycSize = xySize*cSize;
  int sharedIdx = xyId + cId * xyBlockDim;

  if(cIdx < cSize) {
    float accSum = 0.0f;
    float accMax = -1.0f;
    int xyIdx = xyId;
    while(xyIdx < xySize) {
      float a = __half2float(in[xyIdx + cIdx * xySize + nIdx * xycSize]);
      accSum += a;
      // Init to -1.0 above and + mask - 1.0 is because it will effectively make all padded space into -1.0
      // which is lower than the lowest value that any current activation function will produce.
      // so the max over all valid spaces will the same as the mask over all spaces including padding
      accMax = fmaxf(accMax, a + (__half2float(mask[xyIdx + nIdx * xySize]) - 1.0f));
      xyIdx += xyBlockDim;
    }
    sumShared[sharedIdx] = accSum;
    maxShared[sharedIdx] = accMax;
  }
  __syncthreads();

  for(int s = xyBlockDim>>1; s > 0; s >>= 1) {
    if(xyId < s) {
      sumShared[sharedIdx] += sumShared[sharedIdx + s];
      maxShared[sharedIdx] = fmaxf(maxShared[sharedIdx], maxShared[sharedIdx + s]);
    }
    __syncthreads();
  }
  if(xyId == 0 && cIdx < cSize) {
    float sum = sumShared[sharedIdx];
    float div = maskSum[nIdx];
    float sqrtdiv = sqrt(div);
    float mean = sum/div;

    out[cIdx + nIdx * (cSize*3)] = __float2half(mean);
    out[cIdx + nIdx * (cSize*3) + cSize] = __float2half(mean * (sqrtdiv - 14.0f) * 0.1f);
    out[cIdx + nIdx * (cSize*3) + cSize*2] = __float2half(maxShared[sharedIdx]);
  }
#else
  //Do nothing, FP16 not supported
#endif
}

void customCudaPoolRowsGPoolNCHW(const half* in, half* out, int nSize, int cSize, int xySize, const half* mask, const float* maskSum) {
  if(nSize > 65536)
    throw std::runtime_error("customCudaPoolRowsGPoolNCHW: nSize too large");
  if(cSize > 65536)
    throw std::runtime_error("customCudaPoolRowsGPoolNCHW: cSize too large");

  //Use up as many threads as possible along the xy dimension.
  int xyThreads = 1;
  while(xyThreads < targetNumThreads && xyThreads < xySize/2)
    xyThreads *= 2;

  //Distribute the extra threads along the c dimension.
  int cThreads = (targetNumThreads < xyThreads) ? 1 : (targetNumThreads / xyThreads);
  int cBlocks = (cSize + cThreads - 1) / cThreads;

  //We need one shared memory spot per thread, and then we double it because we need both sum and max.
  //We also make sure it's a power of two to address any alignment concerns.
  int sharedMemElts = 128;
  while(sharedMemElts < cThreads * xyThreads)
    sharedMemElts *= 2;
  int sharedMemSize = sizeof(float) * sharedMemElts * 2;

  dim3 grid(1,cBlocks,nSize);
  dim3 threads(xyThreads,cThreads,1);
  if(mask != NULL)
    gPoolChannelsNCHWHalfMaskKernel<<<grid,threads,sharedMemSize>>>(in,out,cSize,xySize,mask,maskSum,sharedMemElts);
  else
    gPoolChannelsNCHWHalfKernel<<<grid,threads,sharedMemSize>>>(in,out,cSize,xySize,maskSum,sharedMemElts);
}



//--------------------------------------------------------------------------------------------------------------

__global__
void sumChannelsNHWCKernel(const float* in, float* out, int xySize, int cSize, float scaleSum)
{
  extern __shared__ float sumPoolNHWCShared[];
  int cId = threadIdx.x;
  int cBlockDim = blockDim.x;
  int xyId = threadIdx.y;
  int xyBlockDim = blockDim.y;

  int cIdx = blockIdx.x * cBlockDim + cId;
  int nIdx = blockIdx.z;
  int sharedIdx = cId + cBlockDim * xyId;
  int xycSize = xySize*cSize;

  sumPoolNHWCShared[sharedIdx] = 0;

  if(cIdx < cSize) {
    int xyIdx = xyId;
    while(xyIdx < xySize) {
      sumPoolNHWCShared[sharedIdx] += in[cIdx + xyIdx * cSize + nIdx * xycSize];
      xyIdx += xyBlockDim;
    }
  }
  __syncthreads();

  for(int s = xyBlockDim>>1; s > 0; s >>= 1) {
    if(xyId < s) {
      sumPoolNHWCShared[sharedIdx] += sumPoolNHWCShared[sharedIdx + cBlockDim * s];
    }
    __syncthreads();
  }
  if(xyId == 0 && cIdx < cSize)
    out[cIdx + nIdx * cSize] = sumPoolNHWCShared[sharedIdx] * scaleSum;
}
__global__
void valueHeadPoolChannelsNHWCKernel(const float* in, float* out, int nSize, int xySize, int cSize, const float* maskSum)
{
  extern __shared__ float sumPoolNHWCShared[];
  int cId = threadIdx.x;
  int cBlockDim = blockDim.x;
  int xyId = threadIdx.y;
  int xyBlockDim = blockDim.y;

  int cIdx = blockIdx.x * cBlockDim + cId;
  int nIdx = blockIdx.z;
  int sharedIdx = cId + cBlockDim * xyId;
  int xycSize = xySize*cSize;

  sumPoolNHWCShared[sharedIdx] = 0;

  if(cIdx < cSize) {
    int xyIdx = xyId;
    while(xyIdx < xySize) {
      sumPoolNHWCShared[sharedIdx] += in[cIdx + xyIdx * cSize + nIdx * xycSize];
      xyIdx += xyBlockDim;
    }
  }
  __syncthreads();

  for(int s = xyBlockDim>>1; s > 0; s >>= 1) {
    if(xyId < s) {
      sumPoolNHWCShared[sharedIdx] += sumPoolNHWCShared[sharedIdx + cBlockDim * s];
    }
    __syncthreads();
  }
  if(xyId == 0 && cIdx < cSize) {
    float sum = sumPoolNHWCShared[sharedIdx];
    float div = maskSum[nIdx];
    float sqrtdiv = sqrt(div);
    float mean = sum/div;
    out[cIdx + nIdx * cSize*3] = mean;
    out[cIdx + nIdx * cSize*3 + cSize] = mean * (sqrtdiv - 14.0f) * 0.1f;
    out[cIdx + nIdx * cSize*3 + cSize*2] = mean * ((sqrtdiv - 14.0f) * (sqrtdiv - 14.0f) * 0.01f - 0.1f);
  }
}
__global__
void gPoolChannelsNHWCKernel(const float* in, float* out, int xySize, int cSize, const float* maskSum, int sharedMemElts)
{
  extern __shared__ float poolNHWCShared[];
  float* sumShared = (float*)poolNHWCShared;
  float* maxShared = (float*)poolNHWCShared + sharedMemElts;

  int cId = threadIdx.x;
  int cBlockDim = blockDim.x;
  int xyId = threadIdx.y;
  int xyBlockDim = blockDim.y;

  int cIdx = blockIdx.x * cBlockDim + cId;
  int nIdx = blockIdx.z;
  int sharedIdx = cId + cBlockDim * xyId;
  int xycSize = xySize*cSize;

  sumShared[sharedIdx] = 0;
  maxShared[sharedIdx] = -1.0f;

  if(cIdx < cSize) {
    int xyIdx = xyId;
    while(xyIdx < xySize) {
      float a = in[cIdx + xyIdx * cSize + nIdx * xycSize];
      sumShared[sharedIdx] += a;
      maxShared[sharedIdx] = fmaxf(maxShared[sharedIdx], a);
      xyIdx += xyBlockDim;
    }
  }
  __syncthreads();

  for(int s = xyBlockDim>>1; s > 0; s >>= 1) {
    if(xyId < s) {
      sumShared[sharedIdx] += sumShared[sharedIdx + cBlockDim * s];
      maxShared[sharedIdx] = fmaxf(maxShared[sharedIdx],maxShared[sharedIdx + cBlockDim * s]);
    }
    __syncthreads();
  }
  if(xyId == 0 && cIdx < cSize) {
    float sum = sumShared[sharedIdx];
    float div = maskSum[nIdx];
    float sqrtdiv = sqrt(div);
    float mean = sum/div;

    out[cIdx + nIdx * (cSize*3)] = mean;
    out[cIdx + nIdx * (cSize*3) + cSize] = mean * (sqrtdiv - 14.0f) * 0.1f;
    out[cIdx + nIdx * (cSize*3) + cSize*2] = maxShared[sharedIdx];
  }
}
__global__
void gPoolChannelsNHWCMaskKernel(const float* in, float* out, int xySize, int cSize, const float* mask, const float* maskSum, int sharedMemElts)
{
  extern __shared__ float poolNHWCShared[];
  float* sumShared = (float*)poolNHWCShared;
  float* maxShared = (float*)poolNHWCShared + sharedMemElts;

  int cId = threadIdx.x;
  int cBlockDim = blockDim.x;
  int xyId = threadIdx.y;
  int xyBlockDim = blockDim.y;

  int cIdx = blockIdx.x * cBlockDim + cId;
  int nIdx = blockIdx.z;
  int sharedIdx = cId + cBlockDim * xyId;
  int xycSize = xySize*cSize;

  sumShared[sharedIdx] = 0;
  maxShared[sharedIdx] = -1.0f;

  if(cIdx < cSize) {
    int xyIdx = xyId;
    while(xyIdx < xySize) {
      float a = in[cIdx + xyIdx * cSize + nIdx * xycSize];
      sumShared[sharedIdx] += a;
      // Init to -1.0 above and + mask - 1.0 is because it will effectively make all padded space into -1.0
      // which is lower than the lowest value that any current activation function will produce.
      // so the max over all valid spaces will the same as the mask over all spaces including padding
      maxShared[sharedIdx] = fmaxf(maxShared[sharedIdx], a + (mask[xyIdx + nIdx * xySize] - 1.0f));
      xyIdx += xyBlockDim;
    }
  }
  __syncthreads();

  for(int s = xyBlockDim>>1; s > 0; s >>= 1) {
    if(xyId < s) {
      sumShared[sharedIdx] += sumShared[sharedIdx + cBlockDim * s];
      maxShared[sharedIdx] = fmaxf(maxShared[sharedIdx],maxShared[sharedIdx + cBlockDim * s]);
    }
    __syncthreads();
  }
  if(xyId == 0 && cIdx < cSize) {
    float sum = sumShared[sharedIdx];
    float div = maskSum[nIdx];
    float sqrtdiv = sqrt(div);
    float mean = sum/div;

    out[cIdx + nIdx * (cSize*3)] = mean;
    out[cIdx + nIdx * (cSize*3) + cSize] = mean * (sqrtdiv - 14.0f) * 0.1f;
    out[cIdx + nIdx * (cSize*3) + cSize*2] = maxShared[sharedIdx];
  }
}


void customCudaPoolRowsSumNHWC(const float* in, float* out, int nSize, int xySize, int cSize, float scaleSum) {
  if(nSize > 65536)
    throw std::runtime_error("customCudaPoolRowsSumNHWC: nSize too large");
  if(cSize > 65536)
    throw std::runtime_error("customCudaPoolRowsSumNHWC: cSize too large");

  //Use up to two warps worth of threads along the channel dimension, which is the
  //most compact
  int cThreads = 1;
  while(cThreads < 64 && cThreads < cSize/2)
    cThreads *= 2;
  int cBlocks = (cSize + cThreads - 1) / cThreads;

  //Distribute the extra threads to perform parallel reduction along the xy dimension.
  int xyThreads = (targetNumThreads < cThreads) ? 1 : (targetNumThreads / cThreads);

  //We need one shared memory spot per thread
  int sharedMemSize = sizeof(float) * cThreads * xyThreads;

  dim3 grid(cBlocks,1,nSize);
  dim3 threads(cThreads,xyThreads,1);
  sumChannelsNHWCKernel<<<grid,threads,sharedMemSize>>>(in,out,xySize,cSize,scaleSum);
}

void customCudaValueHeadPoolNHWC(const float* in, float* out, int nSize, int xySize, int cSize, const float* maskSum) {
  if(nSize > 65536)
    throw std::runtime_error("customCudaValueHeadPoolNHWC: nSize too large");
  if(cSize > 65536)
    throw std::runtime_error("customCudaValueHeadPoolNHWC: cSize too large");

  //Use up to two warps worth of threads along the channel dimension, which is the
  //most compact
  int cThreads = 1;
  while(cThreads < 64 && cThreads < cSize/2)
    cThreads *= 2;
  int cBlocks = (cSize + cThreads - 1) / cThreads;

  //Distribute the extra threads to perform parallel reduction along the xy dimension.
  int xyThreads = (targetNumThreads < cThreads) ? 1 : (targetNumThreads / cThreads);

  //We need one shared memory spot per thread
  int sharedMemSize = sizeof(float) * cThreads * xyThreads;

  dim3 grid(cBlocks,1,nSize);
  dim3 threads(cThreads,xyThreads,1);
  valueHeadPoolChannelsNHWCKernel<<<grid,threads,sharedMemSize>>>(in,out,nSize,xySize,cSize,maskSum);
}

void customCudaPoolRowsGPoolNHWC(const float* in, float* out, int nSize, int xySize, int cSize, const float* mask, const float* maskSum) {
  if(nSize > 65536)
    throw std::runtime_error("customCudaPoolRowsGPoolNHWC: nSize too large");
  if(cSize > 65536)
    throw std::runtime_error("customCudaPoolRowsGPoolNHWC: cSize too large");

  //Use up to two warps worth of threads along the channel dimension, which is the
  //most compact
  int cThreads = 1;
  while(cThreads < 64 && cThreads < cSize/2)
    cThreads *= 2;
  int cBlocks = (cSize + cThreads - 1) / cThreads;

  //Distribute the extra threads to perform parallel reduction along the xy dimension.
  int xyThreads = (targetNumThreads < cThreads) ? 1 : (targetNumThreads / cThreads);

  //We need one shared memory spot per thread, and then we double it because we need both sum and max.
  //We also make sure it's a power of two to address any alignment concerns.
  int sharedMemElts = 128;
  while(sharedMemElts < cThreads * xyThreads)
    sharedMemElts *= 2;
  int sharedMemSize = sizeof(float) * sharedMemElts * 2;

  dim3 grid(cBlocks,1,nSize);
  dim3 threads(cThreads,xyThreads,1);
  if(mask != NULL)
    gPoolChannelsNHWCMaskKernel<<<grid,threads,sharedMemSize>>>(in,out,xySize,cSize,mask,maskSum,sharedMemElts);
  else
    gPoolChannelsNHWCKernel<<<grid,threads,sharedMemSize>>>(in,out,xySize,cSize,maskSum,sharedMemElts);
}

//--------------------------------------------------------------------------------------------------------------

__global__
void gPoolChannelsNHWCHalfKernel(const half* in, half* out, int xySize, int cSize, const float* maskSum, int sharedMemElts)
{
#ifdef CUDA_SUPPORTS_FP16
  extern __shared__ float poolNHWCShared[];
  float* sumShared = (float*)poolNHWCShared;
  float* maxShared = (float*)poolNHWCShared + sharedMemElts;

  int cId = threadIdx.x;
  int cBlockDim = blockDim.x;
  int xyId = threadIdx.y;
  int xyBlockDim = blockDim.y;

  int cIdx = blockIdx.x * cBlockDim + cId;
  int nIdx = blockIdx.z;
  int sharedIdx = cId + cBlockDim * xyId;
  int xycSize = xySize*cSize;

  sumShared[sharedIdx] = 0;
  maxShared[sharedIdx] = -1.0f;

  if(cIdx < cSize) {
    int xyIdx = xyId;
    while(xyIdx < xySize) {
      float a = __half2float(in[cIdx + xyIdx * cSize + nIdx * xycSize]);
      sumShared[sharedIdx] += a;
      maxShared[sharedIdx] = fmaxf(maxShared[sharedIdx], a);
      xyIdx += xyBlockDim;
    }
  }
  __syncthreads();

  for(int s = xyBlockDim>>1; s > 0; s >>= 1) {
    if(xyId < s) {
      sumShared[sharedIdx] += sumShared[sharedIdx + cBlockDim * s];
      maxShared[sharedIdx] = fmaxf(maxShared[sharedIdx],maxShared[sharedIdx + cBlockDim * s]);
    }
    __syncthreads();
  }
  if(xyId == 0 && cIdx < cSize) {
    float sum = sumShared[sharedIdx];
    float div = maskSum[nIdx];
    float sqrtdiv = sqrt(div);
    float mean = sum/div;

    out[cIdx + nIdx * (cSize*3)] = __float2half(mean);
    out[cIdx + nIdx * (cSize*3) + cSize] = __float2half(mean * (sqrtdiv - 14.0f) * 0.1f);
    out[cIdx + nIdx * (cSize*3) + cSize*2] = __float2half(maxShared[sharedIdx]);
  }
#else
  //Do nothing, FP16 not supported
#endif
}
__global__
void gPoolChannelsNHWCHalfMaskKernel(const half* in, half* out, int xySize, int cSize, const half* mask, const float* maskSum, int sharedMemElts)
{
#ifdef CUDA_SUPPORTS_FP16
  extern __shared__ float poolNHWCShared[];
  float* sumShared = (float*)poolNHWCShared;
  float* maxShared = (float*)poolNHWCShared + sharedMemElts;

  int cId = threadIdx.x;
  int cBlockDim = blockDim.x;
  int xyId = threadIdx.y;
  int xyBlockDim = blockDim.y;

  int cIdx = blockIdx.x * cBlockDim + cId;
  int nIdx = blockIdx.z;
  int sharedIdx = cId + cBlockDim * xyId;
  int xycSize = xySize*cSize;

  sumShared[sharedIdx] = 0;
  maxShared[sharedIdx] = -1.0f;

  if(cIdx < cSize) {
    int xyIdx = xyId;
    while(xyIdx < xySize) {
      float a = __half2float(in[cIdx + xyIdx * cSize + nIdx * xycSize]);
      sumShared[sharedIdx] += a;
      // Init to -1.0 above and + mask - 1.0 is because it will effectively make all padded space into -1.0
      // which is lower than the lowest value that any current activation function will produce.
      // so the max over all valid spaces will the same as the mask over all spaces including padding
      maxShared[sharedIdx] = fmaxf(maxShared[sharedIdx], a + (__half2float(mask[xyIdx + nIdx * xySize]) - 1.0f));
      xyIdx += xyBlockDim;
    }
  }
  __syncthreads();

  for(int s = xyBlockDim>>1; s > 0; s >>= 1) {
    if(xyId < s) {
      sumShared[sharedIdx] += sumShared[sharedIdx + cBlockDim * s];
      maxShared[sharedIdx] = fmaxf(maxShared[sharedIdx],maxShared[sharedIdx + cBlockDim * s]);
    }
    __syncthreads();
  }
  if(xyId == 0 && cIdx < cSize) {
    float sum = sumShared[sharedIdx];
    float div = maskSum[nIdx];
    float sqrtdiv = sqrt(div);
    float mean = sum/div;

    out[cIdx + nIdx * (cSize*3)] = __float2half(mean);
    out[cIdx + nIdx * (cSize*3) + cSize] = __float2half(mean * (sqrtdiv - 14.0f) * 0.1f);
    out[cIdx + nIdx * (cSize*3) + cSize*2] = __float2half(maxShared[sharedIdx]);
  }
#else
  //Do nothing, FP16 not supported
#endif
}

void customCudaPoolRowsGPoolNHWC(const half* in, half* out, int nSize, int xySize, int cSize, const half* mask, const float* maskSum) {
  if(nSize > 65536)
    throw std::runtime_error("customCudaPoolRowsGPoolNHWC: nSize too large");
  if(cSize > 65536)
    throw std::runtime_error("customCudaPoolRowsGPoolNHWC: cSize too large");

  //Use up to two warps worth of threads along the channel dimension, which is the
  //most compact
  int cThreads = 1;
  while(cThreads < 64 && cThreads < cSize/2)
    cThreads *= 2;
  int cBlocks = (cSize + cThreads - 1) / cThreads;

  //Distribute the extra threads to perform parallel reduction along the xy dimension.
  int xyThreads = (targetNumThreads < cThreads) ? 1 : (targetNumThreads / cThreads);

  //We need one shared memory spot per thread, and then we double it because we need both sum and max.
  //We also make sure it's a power of two to address any alignment concerns.
  int sharedMemElts = 128;
  while(sharedMemElts < cThreads * xyThreads)
    sharedMemElts *= 2;
  int sharedMemSize = sizeof(float) * sharedMemElts * 2;

  dim3 grid(cBlocks,1,nSize);
  dim3 threads(cThreads,xyThreads,1);
  if(mask != NULL)
    gPoolChannelsNHWCHalfMaskKernel<<<grid,threads,sharedMemSize>>>(in,out,xySize,cSize,mask,maskSum,sharedMemElts);
  else
    gPoolChannelsNHWCHalfKernel<<<grid,threads,sharedMemSize>>>(in,out,xySize,cSize,maskSum,sharedMemElts);
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


__global__
void addTensorInplaceHalfKernel(half *buf, const half* biases, int nSize)
{
#ifdef CUDA_SUPPORTS_FP16
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if(idx < nSize) {
    buf[idx] = __hadd(buf[idx],biases[idx]);
  }
#else
  //Do nothing, FP16 not supported
#endif
}
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
__global__
void addCBiasInplaceNCHalfKernel(half *buf, const half* biases, int nSize, int cSize)
{
#ifdef CUDA_SUPPORTS_FP16
  int cIdx = blockIdx.x * blockDim.x + threadIdx.x;
  int nIdx = blockIdx.y * blockDim.y + threadIdx.y;
  if(cIdx < cSize && nIdx < nSize) {
    int idx = nIdx * cSize + cIdx;
    buf[idx] = __hadd(buf[idx],biases[cIdx]);
  }
#else
  //Do nothing, FP16 not supported
#endif
}

__global__
void addCBiasInplaceNCKernelRelu(float *buf, const float* biases, int nSize, int cSize)
{
  int cIdx = blockIdx.x * blockDim.x + threadIdx.x;
  int nIdx = blockIdx.y * blockDim.y + threadIdx.y;
  if(cIdx < cSize && nIdx < nSize) {
    int idx = nIdx * cSize + cIdx;
    buf[idx] = fmaxf(buf[idx] + biases[cIdx],0.0f);
  }
}
__global__
void addCBiasInplaceNCHalfKernelRelu(half *buf, const half* biases, int nSize, int cSize)
{
#ifdef CUDA_SUPPORTS_FP16
  int cIdx = blockIdx.x * blockDim.x + threadIdx.x;
  int nIdx = blockIdx.y * blockDim.y + threadIdx.y;
  if(cIdx < cSize && nIdx < nSize) {
    int idx = nIdx * cSize + cIdx;
    const half halfzero = __float2half(0.0f);
    half a = __hadd(buf[idx],biases[cIdx]);
    buf[idx] = __hgt(a,halfzero) ? a : halfzero;
  }
#else
  //Do nothing, FP16 not supported
#endif
}

__global__
void addCBiasInplaceNCKernelMish(float *buf, const float* biases, int nSize, int cSize)
{
  int cIdx = blockIdx.x * blockDim.x + threadIdx.x;
  int nIdx = blockIdx.y * blockDim.y + threadIdx.y;
  if(cIdx < cSize && nIdx < nSize) {
    int idx = nIdx * cSize + cIdx;
    buf[idx] = mishf(buf[idx] + biases[cIdx]);
  }
}
__global__
void addCBiasInplaceNCHalfKernelMish(half *buf, const half* biases, int nSize, int cSize)
{
#ifdef CUDA_SUPPORTS_FP16
  int cIdx = blockIdx.x * blockDim.x + threadIdx.x;
  int nIdx = blockIdx.y * blockDim.y + threadIdx.y;
  if(cIdx < cSize && nIdx < nSize) {
    int idx = nIdx * cSize + cIdx;
    half a = __hadd(buf[idx],biases[cIdx]);
    buf[idx] = mishh(a);
  }
#else
  //Do nothing, FP16 not supported
#endif
}

__global__
void addCBiasInplaceNCKernelGelu(float *buf, const float* biases, int nSize, int cSize)
{
  int cIdx = blockIdx.x * blockDim.x + threadIdx.x;
  int nIdx = blockIdx.y * blockDim.y + threadIdx.y;
  if(cIdx < cSize && nIdx < nSize) {
    int idx = nIdx * cSize + cIdx;
    buf[idx] = geluf(buf[idx] + biases[cIdx]);
  }
}
__global__
void addCBiasInplaceNCHalfKernelGelu(half *buf, const half* biases, int nSize, int cSize)
{
#ifdef CUDA_SUPPORTS_FP16
  int cIdx = blockIdx.x * blockDim.x + threadIdx.x;
  int nIdx = blockIdx.y * blockDim.y + threadIdx.y;
  if(cIdx < cSize && nIdx < nSize) {
    int idx = nIdx * cSize + cIdx;
    half a = __hadd(buf[idx],biases[cIdx]);
    buf[idx] = geluh(a);
  }
#else
  //Do nothing, FP16 not supported
#endif
}

void sharedAddCBiasInplaceNC(void* buf, const void* biases, int nSize, int cSize, bool isHalf, int activation) {
  int cThreads;
  int cBlocks;
  int nThreads;
  int nBlocks;
  splitThreadsAcrossDim01(cSize, nSize, cThreads, cBlocks, nThreads, nBlocks);

  if(nBlocks > 65536)
    throw std::runtime_error("customCudaAddCBiasInplaceNC: nSize too large given cSize");

  dim3 grid(cBlocks,nBlocks,1);
  dim3 threads(cThreads,nThreads,1);

  if(activation == ACTIVATION_IDENTITY) {
    if(isHalf)
      addCBiasInplaceNCHalfKernel<<<grid,threads>>>((half*)buf,(const half*)biases,nSize,cSize);
    else
      addCBiasInplaceNCKernel<<<grid,threads>>>((float*)buf,(const float*)biases,nSize,cSize);
  }
  else if(activation == ACTIVATION_RELU) {
    if(isHalf)
      addCBiasInplaceNCHalfKernelRelu<<<grid,threads>>>((half*)buf,(const half*)biases,nSize,cSize);
    else
      addCBiasInplaceNCKernelRelu<<<grid,threads>>>((float*)buf,(const float*)biases,nSize,cSize);
  }
  else if(activation == ACTIVATION_MISH) {
    if(isHalf)
      addCBiasInplaceNCHalfKernelMish<<<grid,threads>>>((half*)buf,(const half*)biases,nSize,cSize);
    else
      addCBiasInplaceNCKernelMish<<<grid,threads>>>((float*)buf,(const float*)biases,nSize,cSize);
  }
  else if(activation == ACTIVATION_GELU) {
    if(isHalf)
      addCBiasInplaceNCHalfKernelGelu<<<grid,threads>>>((half*)buf,(const half*)biases,nSize,cSize);
    else
      addCBiasInplaceNCKernelGelu<<<grid,threads>>>((float*)buf,(const float*)biases,nSize,cSize);
  }
  else {
    throw std::runtime_error("customCudaAddCBiasInplaceNC: unsupported activation");
  }
}

void customCudaAddCBiasInplaceNC(float* buf, const float* biases, int nSize, int cSize, int activation) {
  sharedAddCBiasInplaceNC(buf,biases,nSize,cSize,false,activation);
}
void customCudaAddCBiasInplaceNC(half* buf, const half* biases, int nSize, int cSize, int activation) {
  sharedAddCBiasInplaceNC(buf,biases,nSize,cSize,true,activation);
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
__global__
void addNCBiasInplaceNCHWHalfKernel(half *buf, const half* biases, int cSize, int sSize)
{
#ifdef CUDA_SUPPORTS_FP16
  int sIdx = blockIdx.x * blockDim.x + threadIdx.x;
  int cIdx = blockIdx.y * blockDim.y + threadIdx.y;
  int nIdx = blockIdx.z;
  if(cIdx < cSize && sIdx < sSize) {
    int ncIdx = nIdx * cSize + cIdx;
    int idx = ncIdx * sSize + sIdx;
    buf[idx] = __hadd(buf[idx],biases[ncIdx]);
  }
#else
  //Do nothing, FP16 not supported
#endif
}

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
  splitThreadsAcrossDim01(sSize, cSize, sThreads, sBlocks, cThreads, cBlocks);

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

__global__
void addNCBiasInplaceNHWCKernel(float *buf, const float* biases, int sSize, int cSize)
{
  int cIdx = blockIdx.x * blockDim.x + threadIdx.x;
  int sIdx = blockIdx.y * blockDim.y + threadIdx.y;
  int nIdx = blockIdx.z;
  if(cIdx < cSize && sIdx < sSize) {
    int ncIdx = nIdx * cSize + cIdx;
    int idx = (nIdx * sSize + sIdx) * cSize + cIdx;
    buf[idx] = buf[idx] + biases[ncIdx];
  }
}
__global__
void addNCBiasInplaceNHWCHalfKernel(half *buf, const half* biases, int sSize, int cSize)
{
#ifdef CUDA_SUPPORTS_FP16
  int cIdx = blockIdx.x * blockDim.x + threadIdx.x;
  int sIdx = blockIdx.y * blockDim.y + threadIdx.y;
  int nIdx = blockIdx.z;
  if(cIdx < cSize && sIdx < sSize) {
    int ncIdx = nIdx * cSize + cIdx;
    int idx = (nIdx * sSize + sIdx) * cSize + cIdx;
    buf[idx] = __hadd(buf[idx],biases[ncIdx]);
  }
#else
  //Do nothing, FP16 not supported
#endif
}

void sharedAddNCBiasInplaceNHWC(void *buf, const void* biases, int nSize, int xySize, int cSize, bool isHalf) {
  if(nSize > 65536)
    throw std::runtime_error("customCudaAddNCBiasInplaceNHWC: nSize too large");
  if(xySize > 65536)
    throw std::runtime_error("customCudaAddNCBiasInplaceNHWC: xySize too large");

  int sSize = xySize;
  int cThreads;
  int cBlocks;
  int sThreads;
  int sBlocks;
  splitThreadsAcrossDim01(cSize, sSize, cThreads, cBlocks, sThreads, sBlocks);

  dim3 grid(cBlocks,sBlocks,nSize);
  dim3 threads(cThreads,sThreads,1);
  if(isHalf)
    addNCBiasInplaceNHWCHalfKernel<<<grid,threads>>>((half*)buf,(const half*)biases,sSize,cSize);
  else
    addNCBiasInplaceNHWCKernel<<<grid,threads>>>((float*)buf,(const float*)biases,sSize,cSize);
}

void customCudaAddNCBiasInplaceNHWC(float *buf, const float* biases, int nSize, int xySize, int cSize) {
  sharedAddNCBiasInplaceNHWC(buf,biases,nSize,xySize,cSize,false);
}
void customCudaAddNCBiasInplaceNHWC(half *buf, const half* biases, int nSize, int xySize, int cSize) {
  sharedAddNCBiasInplaceNHWC(buf,biases,nSize,xySize,cSize,true);
}

//--------------------------------------------------------------------------------------------------------------

__global__
void applyCScaleBiasNCHWKernel(const float *in, float* out, const float* scale, const float* biases, int cSize, int sSize)
{
  int sIdx = blockIdx.x * blockDim.x + threadIdx.x;
  int cIdx = blockIdx.y * blockDim.y + threadIdx.y;
  int nIdx = blockIdx.z;
  if(cIdx < cSize && sIdx < sSize) {
    int idx = (nIdx * cSize + cIdx) * sSize + sIdx;
    out[idx] = in[idx] * scale[cIdx] + biases[cIdx];
  }
}
__global__
void applyCScaleBiasNCHWReluKernel(const float *in, float* out, const float* scale, const float* biases, int cSize, int sSize)
{
  int sIdx = blockIdx.x * blockDim.x + threadIdx.x;
  int cIdx = blockIdx.y * blockDim.y + threadIdx.y;
  int nIdx = blockIdx.z;
  if(cIdx < cSize && sIdx < sSize) {
    int idx = (nIdx * cSize + cIdx) * sSize + sIdx;
    out[idx] = fmaxf(in[idx] * scale[cIdx] + biases[cIdx],0.0f);
  }
}
__global__
void applyCScaleBiasNCHWMishKernel(const float *in, float* out, const float* scale, const float* biases, int cSize, int sSize)
{
  int sIdx = blockIdx.x * blockDim.x + threadIdx.x;
  int cIdx = blockIdx.y * blockDim.y + threadIdx.y;
  int nIdx = blockIdx.z;
  if(cIdx < cSize && sIdx < sSize) {
    int idx = (nIdx * cSize + cIdx) * sSize + sIdx;
    out[idx] = mishf(in[idx] * scale[cIdx] + biases[cIdx]);
  }
}
__global__
void applyCScaleBiasNCHWGeluKernel(const float *in, float* out, const float* scale, const float* biases, int cSize, int sSize)
{
  int sIdx = blockIdx.x * blockDim.x + threadIdx.x;
  int cIdx = blockIdx.y * blockDim.y + threadIdx.y;
  int nIdx = blockIdx.z;
  if(cIdx < cSize && sIdx < sSize) {
    int idx = (nIdx * cSize + cIdx) * sSize + sIdx;
    out[idx] = geluf(in[idx] * scale[cIdx] + biases[cIdx]);
  }
}
__global__
void applyCScaleBiasNCHWMaskKernel(const float *in, float* out, const float* scale, const float* biases, const float* mask, int cSize, int sSize)
{
  int sIdx = blockIdx.x * blockDim.x + threadIdx.x;
  int cIdx = blockIdx.y * blockDim.y + threadIdx.y;
  int nIdx = blockIdx.z;
  if(cIdx < cSize && sIdx < sSize) {
    int idx = (nIdx * cSize + cIdx) * sSize + sIdx;
    out[idx] = (in[idx] * scale[cIdx] + biases[cIdx]) * mask[nIdx*sSize+sIdx];
  }
}
__global__
void applyCScaleBiasNCHWReluMaskKernel(const float *in, float* out, const float* scale, const float* biases, const float* mask, int cSize, int sSize)
{
  int sIdx = blockIdx.x * blockDim.x + threadIdx.x;
  int cIdx = blockIdx.y * blockDim.y + threadIdx.y;
  int nIdx = blockIdx.z;
  if(cIdx < cSize && sIdx < sSize) {
    int idx = (nIdx * cSize + cIdx) * sSize + sIdx;
    out[idx] = fmaxf(in[idx] * scale[cIdx] + biases[cIdx],0.0f) * mask[nIdx*sSize+sIdx];
  }
}
__global__
void applyCScaleBiasNCHWMishMaskKernel(const float *in, float* out, const float* scale, const float* biases, const float* mask, int cSize, int sSize)
{
  int sIdx = blockIdx.x * blockDim.x + threadIdx.x;
  int cIdx = blockIdx.y * blockDim.y + threadIdx.y;
  int nIdx = blockIdx.z;
  if(cIdx < cSize && sIdx < sSize) {
    int idx = (nIdx * cSize + cIdx) * sSize + sIdx;
    out[idx] = mishf(in[idx] * scale[cIdx] + biases[cIdx]) * mask[nIdx*sSize+sIdx];
  }
}
__global__
void applyCScaleBiasNCHWGeluMaskKernel(const float *in, float* out, const float* scale, const float* biases, const float* mask, int cSize, int sSize)
{
  int sIdx = blockIdx.x * blockDim.x + threadIdx.x;
  int cIdx = blockIdx.y * blockDim.y + threadIdx.y;
  int nIdx = blockIdx.z;
  if(cIdx < cSize && sIdx < sSize) {
    int idx = (nIdx * cSize + cIdx) * sSize + sIdx;
    out[idx] = geluf(in[idx] * scale[cIdx] + biases[cIdx]) * mask[nIdx*sSize+sIdx];
  }
}
__global__
void applyCScaleBiasNCHWHalfKernel(const half *in, half* out, const half* scale, const half* biases, int cSize, int sSize)
{
#ifdef CUDA_SUPPORTS_FP16
  int sIdx = blockIdx.x * blockDim.x + threadIdx.x;
  int cIdx = blockIdx.y * blockDim.y + threadIdx.y;
  int nIdx = blockIdx.z;
  if(cIdx < cSize && sIdx < sSize) {
    int idx = (nIdx * cSize + cIdx) * sSize + sIdx;
    out[idx] = __hfma(in[idx],scale[cIdx],biases[cIdx]);
  }
#else
  //Do nothing, FP16 not supported
#endif
}
__global__
void applyCScaleBiasNCHWReluHalfKernel(const half *in, half* out, const half* scale, const half* biases, int cSize, int sSize)
{
#ifdef CUDA_SUPPORTS_FP16
  int sIdx = blockIdx.x * blockDim.x + threadIdx.x;
  int cIdx = blockIdx.y * blockDim.y + threadIdx.y;
  int nIdx = blockIdx.z;
  if(cIdx < cSize && sIdx < sSize) {
    int idx = (nIdx * cSize + cIdx) * sSize + sIdx;
    half a = __hfma(in[idx],scale[cIdx],biases[cIdx]);
    const half halfzero = __float2half(0.0f);
    out[idx] = __hgt(a,halfzero) ? a : halfzero;
  }
#else
  //Do nothing, FP16 not supported
#endif
}
__global__
void applyCScaleBiasNCHWMishHalfKernel(const half *in, half* out, const half* scale, const half* biases, int cSize, int sSize)
{
#ifdef CUDA_SUPPORTS_FP16
  int sIdx = blockIdx.x * blockDim.x + threadIdx.x;
  int cIdx = blockIdx.y * blockDim.y + threadIdx.y;
  int nIdx = blockIdx.z;
  if(cIdx < cSize && sIdx < sSize) {
    int idx = (nIdx * cSize + cIdx) * sSize + sIdx;
    half a = __hfma(in[idx],scale[cIdx],biases[cIdx]);
    out[idx] = mishh(a);
  }
#else
  //Do nothing, FP16 not supported
#endif
}
__global__
void applyCScaleBiasNCHWGeluHalfKernel(const half *in, half* out, const half* scale, const half* biases, int cSize, int sSize)
{
#ifdef CUDA_SUPPORTS_FP16
  int sIdx = blockIdx.x * blockDim.x + threadIdx.x;
  int cIdx = blockIdx.y * blockDim.y + threadIdx.y;
  int nIdx = blockIdx.z;
  if(cIdx < cSize && sIdx < sSize) {
    int idx = (nIdx * cSize + cIdx) * sSize + sIdx;
    half a = __hfma(in[idx],scale[cIdx],biases[cIdx]);
    out[idx] = geluh(a);
  }
#else
  //Do nothing, FP16 not supported
#endif
}
__global__
void applyCScaleBiasNCHWMaskHalfKernel(const half *in, half* out, const half* scale, const half* biases, const half* mask, int cSize, int sSize)
{
#ifdef CUDA_SUPPORTS_FP16
  int sIdx = blockIdx.x * blockDim.x + threadIdx.x;
  int cIdx = blockIdx.y * blockDim.y + threadIdx.y;
  int nIdx = blockIdx.z;
  if(cIdx < cSize && sIdx < sSize) {
    int idx = (nIdx * cSize + cIdx) * sSize + sIdx;
    out[idx] = __hmul(__hfma(in[idx],scale[cIdx],biases[cIdx]),mask[nIdx*sSize+sIdx]);
  }
#else
  //Do nothing, FP16 not supported
#endif
}
__global__
void applyCScaleBiasNCHWReluMaskHalfKernel(const half *in, half* out, const half* scale, const half* biases, const half* mask, int cSize, int sSize)
{
#ifdef CUDA_SUPPORTS_FP16
  int sIdx = blockIdx.x * blockDim.x + threadIdx.x;
  int cIdx = blockIdx.y * blockDim.y + threadIdx.y;
  int nIdx = blockIdx.z;
  if(cIdx < cSize && sIdx < sSize) {
    int idx = (nIdx * cSize + cIdx) * sSize + sIdx;
    half a = __hmul(__hfma(in[idx],scale[cIdx],biases[cIdx]),mask[nIdx*sSize+sIdx]);
    const half halfzero = __float2half(0.0f);
    out[idx] = __hgt(a,halfzero) ? a : halfzero;
  }
#else
  //Do nothing, FP16 not supported
#endif
}
__global__
void applyCScaleBiasNCHWMishMaskHalfKernel(const half *in, half* out, const half* scale, const half* biases, const half* mask, int cSize, int sSize)
{
#ifdef CUDA_SUPPORTS_FP16
  int sIdx = blockIdx.x * blockDim.x + threadIdx.x;
  int cIdx = blockIdx.y * blockDim.y + threadIdx.y;
  int nIdx = blockIdx.z;
  if(cIdx < cSize && sIdx < sSize) {
    int idx = (nIdx * cSize + cIdx) * sSize + sIdx;
    half a = __hmul(__hfma(in[idx],scale[cIdx],biases[cIdx]),mask[nIdx*sSize+sIdx]);
    out[idx] = mishh(a);
  }
#else
  //Do nothing, FP16 not supported
#endif
}
__global__
void applyCScaleBiasNCHWGeluMaskHalfKernel(const half *in, half* out, const half* scale, const half* biases, const half* mask, int cSize, int sSize)
{
#ifdef CUDA_SUPPORTS_FP16
  int sIdx = blockIdx.x * blockDim.x + threadIdx.x;
  int cIdx = blockIdx.y * blockDim.y + threadIdx.y;
  int nIdx = blockIdx.z;
  if(cIdx < cSize && sIdx < sSize) {
    int idx = (nIdx * cSize + cIdx) * sSize + sIdx;
    half a = __hmul(__hfma(in[idx],scale[cIdx],biases[cIdx]),mask[nIdx*sSize+sIdx]);
    out[idx] = geluh(a);
  }
#else
  //Do nothing, FP16 not supported
#endif
}

void sharedApplyCScaleBiasNCHW(const void* in, void* out, const void* scale, const void* biases, const void* mask, int nSize, int cSize, int xySize, bool isHalf, int activation) {
  if(nSize > 65536)
    throw std::runtime_error("customCudaApplyCScaleBiasNCHW: nSize too large");
  if(cSize > 65536)
    throw std::runtime_error("customCudaApplyCScaleBiasNCHW: cSize too large");

  int sSize = xySize;
  int sThreads;
  int sBlocks;
  int cThreads;
  int cBlocks;
  splitThreadsAcrossDim01(sSize, cSize, sThreads, sBlocks, cThreads, cBlocks);

  dim3 grid(sBlocks,cBlocks,nSize);
  dim3 threads(sThreads,cThreads,1);
  if(mask == NULL) {
    if(activation == ACTIVATION_IDENTITY) {
      if(isHalf)
        applyCScaleBiasNCHWHalfKernel<<<grid,threads>>>((const half*)in,(half*)out,(const half*)scale,(const half*)biases,cSize,sSize);
      else
        applyCScaleBiasNCHWKernel<<<grid,threads>>>((const float*)in,(float*)out,(const float*)scale,(const float*)biases,cSize,sSize);
    }
    else if(activation == ACTIVATION_RELU) {
      if(isHalf)
        applyCScaleBiasNCHWReluHalfKernel<<<grid,threads>>>((const half*)in,(half*)out,(const half*)scale,(const half*)biases,cSize,sSize);
      else
        applyCScaleBiasNCHWReluKernel<<<grid,threads>>>((const float*)in,(float*)out,(const float*)scale,(const float*)biases,cSize,sSize);
    }
    else if(activation == ACTIVATION_MISH) {
      if(isHalf)
        applyCScaleBiasNCHWMishHalfKernel<<<grid,threads>>>((const half*)in,(half*)out,(const half*)scale,(const half*)biases,cSize,sSize);
      else
        applyCScaleBiasNCHWMishKernel<<<grid,threads>>>((const float*)in,(float*)out,(const float*)scale,(const float*)biases,cSize,sSize);
    }
    else if(activation == ACTIVATION_GELU) {
      if(isHalf)
        applyCScaleBiasNCHWGeluHalfKernel<<<grid,threads>>>((const half*)in,(half*)out,(const half*)scale,(const half*)biases,cSize,sSize);
      else
        applyCScaleBiasNCHWGeluKernel<<<grid,threads>>>((const float*)in,(float*)out,(const float*)scale,(const float*)biases,cSize,sSize);
    }
    else {
      throw std::runtime_error("customCudaApplyCScaleBiasNCHW: unsupported activation");
    }
  }
  else {
    if(activation == ACTIVATION_IDENTITY) {
      if(isHalf)
        applyCScaleBiasNCHWMaskHalfKernel<<<grid,threads>>>((const half*)in,(half*)out,(const half*)scale,(const half*)biases,(const half*)mask,cSize,sSize);
      else
        applyCScaleBiasNCHWMaskKernel<<<grid,threads>>>((const float*)in,(float*)out,(const float*)scale,(const float*)biases,(const float*)mask,cSize,sSize);
    }
    else if(activation == ACTIVATION_RELU) {
      if(isHalf)
        applyCScaleBiasNCHWReluMaskHalfKernel<<<grid,threads>>>((const half*)in,(half*)out,(const half*)scale,(const half*)biases,(const half*)mask,cSize,sSize);
      else
        applyCScaleBiasNCHWReluMaskKernel<<<grid,threads>>>((const float*)in,(float*)out,(const float*)scale,(const float*)biases,(const float*)mask,cSize,sSize);
    }
    else if(activation == ACTIVATION_MISH) {
      if(isHalf)
        applyCScaleBiasNCHWMishMaskHalfKernel<<<grid,threads>>>((const half*)in,(half*)out,(const half*)scale,(const half*)biases,(const half*)mask,cSize,sSize);
      else
        applyCScaleBiasNCHWMishMaskKernel<<<grid,threads>>>((const float*)in,(float*)out,(const float*)scale,(const float*)biases,(const float*)mask,cSize,sSize);
    }
    else if(activation == ACTIVATION_GELU) {
      if(isHalf)
        applyCScaleBiasNCHWGeluMaskHalfKernel<<<grid,threads>>>((const half*)in,(half*)out,(const half*)scale,(const half*)biases,(const half*)mask,cSize,sSize);
      else
        applyCScaleBiasNCHWGeluMaskKernel<<<grid,threads>>>((const float*)in,(float*)out,(const float*)scale,(const float*)biases,(const float*)mask,cSize,sSize);
    }
    else {
      throw std::runtime_error("customCudaApplyCScaleBiasNCHW: unsupported activation");
    }
  }
}

void customCudaApplyCScaleBiasNCHW(const float* in, float* out, const float* scale, const float* biases, const float* mask, int nSize, int cSize, int xySize, int activation) {
  sharedApplyCScaleBiasNCHW(in,out,scale,biases,mask,nSize,cSize,xySize,false,activation);
}
void customCudaApplyCScaleBiasNCHW(const half* in, half* out, const half* scale, const half* biases, const half* mask, int nSize, int cSize, int xySize, int activation) {
  sharedApplyCScaleBiasNCHW(in,out,scale,biases,mask,nSize,cSize,xySize,true,activation);
}


//--------------------------------------------------------------------------------------------------------------

__global__
void applyCScaleBiasNHWCKernel(const float* in, float* out, const float* scale, const float* biases, int sSize, int cSize)
{
  int cIdx = blockIdx.x * blockDim.x + threadIdx.x;
  int sIdx = blockIdx.y * blockDim.y + threadIdx.y;
  int nIdx = blockIdx.z;
  if(cIdx < cSize && sIdx < sSize) {
    int idx = (nIdx * sSize + sIdx) * cSize + cIdx;
    out[idx] = in[idx] * scale[cIdx] + biases[cIdx];
  }
}
__global__
void applyCScaleBiasNHWCReluKernel(const float* in, float* out, const float* scale, const float* biases, int sSize, int cSize)
{
  int cIdx = blockIdx.x * blockDim.x + threadIdx.x;
  int sIdx = blockIdx.y * blockDim.y + threadIdx.y;
  int nIdx = blockIdx.z;
  if(cIdx < cSize && sIdx < sSize) {
    int idx = (nIdx * sSize + sIdx) * cSize + cIdx;
    out[idx] = fmaxf(in[idx] * scale[cIdx] + biases[cIdx],0.0f);
  }
}
__global__
void applyCScaleBiasNHWCMishKernel(const float* in, float* out, const float* scale, const float* biases, int sSize, int cSize)
{
  int cIdx = blockIdx.x * blockDim.x + threadIdx.x;
  int sIdx = blockIdx.y * blockDim.y + threadIdx.y;
  int nIdx = blockIdx.z;
  if(cIdx < cSize && sIdx < sSize) {
    int idx = (nIdx * sSize + sIdx) * cSize + cIdx;
    out[idx] = mishf(in[idx] * scale[cIdx] + biases[cIdx]);
  }
}
__global__
void applyCScaleBiasNHWCGeluKernel(const float* in, float* out, const float* scale, const float* biases, int sSize, int cSize)
{
  int cIdx = blockIdx.x * blockDim.x + threadIdx.x;
  int sIdx = blockIdx.y * blockDim.y + threadIdx.y;
  int nIdx = blockIdx.z;
  if(cIdx < cSize && sIdx < sSize) {
    int idx = (nIdx * sSize + sIdx) * cSize + cIdx;
    out[idx] = geluf(in[idx] * scale[cIdx] + biases[cIdx]);
  }
}
__global__
void applyCScaleBiasNHWCMaskKernel(const float* in, float* out, const float* scale, const float* biases, const float* mask, int sSize, int cSize)
{
  int cIdx = blockIdx.x * blockDim.x + threadIdx.x;
  int sIdx = blockIdx.y * blockDim.y + threadIdx.y;
  int nIdx = blockIdx.z;
  if(cIdx < cSize && sIdx < sSize) {
    int idx = (nIdx * sSize + sIdx) * cSize + cIdx;
    out[idx] = (in[idx] * scale[cIdx] + biases[cIdx]) * mask[nIdx*sSize+sIdx];
  }
}
__global__
void applyCScaleBiasNHWCReluMaskKernel(const float* in, float* out, const float* scale, const float* biases, const float* mask, int sSize, int cSize)
{
  int cIdx = blockIdx.x * blockDim.x + threadIdx.x;
  int sIdx = blockIdx.y * blockDim.y + threadIdx.y;
  int nIdx = blockIdx.z;
  if(cIdx < cSize && sIdx < sSize) {
    int idx = (nIdx * sSize + sIdx) * cSize + cIdx;
    out[idx] = fmaxf(in[idx] * scale[cIdx] + biases[cIdx],0.0f) * mask[nIdx*sSize+sIdx];
  }
}
__global__
void applyCScaleBiasNHWCMishMaskKernel(const float* in, float* out, const float* scale, const float* biases, const float* mask, int sSize, int cSize)
{
  int cIdx = blockIdx.x * blockDim.x + threadIdx.x;
  int sIdx = blockIdx.y * blockDim.y + threadIdx.y;
  int nIdx = blockIdx.z;
  if(cIdx < cSize && sIdx < sSize) {
    int idx = (nIdx * sSize + sIdx) * cSize + cIdx;
    out[idx] = mishf(in[idx] * scale[cIdx] + biases[cIdx]) * mask[nIdx*sSize+sIdx];
  }
}
__global__
void applyCScaleBiasNHWCGeluMaskKernel(const float* in, float* out, const float* scale, const float* biases, const float* mask, int sSize, int cSize)
{
  int cIdx = blockIdx.x * blockDim.x + threadIdx.x;
  int sIdx = blockIdx.y * blockDim.y + threadIdx.y;
  int nIdx = blockIdx.z;
  if(cIdx < cSize && sIdx < sSize) {
    int idx = (nIdx * sSize + sIdx) * cSize + cIdx;
    out[idx] = geluf(in[idx] * scale[cIdx] + biases[cIdx]) * mask[nIdx*sSize+sIdx];
  }
}
__global__
void applyCScaleBiasNHWCHalfKernel(const half* in, half* out, const half* scale, const half* biases, int sSize, int cSize)
{
#ifdef CUDA_SUPPORTS_FP16
  int cIdx = blockIdx.x * blockDim.x + threadIdx.x;
  int sIdx = blockIdx.y * blockDim.y + threadIdx.y;
  int nIdx = blockIdx.z;
  if(cIdx < cSize && sIdx < sSize) {
    int idx = (nIdx * sSize + sIdx) * cSize + cIdx;
    out[idx] = __hfma(in[idx],scale[cIdx],biases[cIdx]);
  }
#else
  //Do nothing, FP16 not supported
#endif
}
__global__
void applyCScaleBiasNHWCReluHalfKernel(const half* in, half* out, const half* scale, const half* biases, int sSize, int cSize)
{
#ifdef CUDA_SUPPORTS_FP16
  int cIdx = blockIdx.x * blockDim.x + threadIdx.x;
  int sIdx = blockIdx.y * blockDim.y + threadIdx.y;
  int nIdx = blockIdx.z;
  if(cIdx < cSize && sIdx < sSize) {
    int idx = (nIdx * sSize + sIdx) * cSize + cIdx;
    half a = __hfma(in[idx],scale[cIdx],biases[cIdx]);
    const half halfzero = __float2half(0.0f);
    out[idx] = __hgt(a,halfzero) ? a : halfzero;
  }
#else
  //Do nothing, FP16 not supported
#endif
}
__global__
void applyCScaleBiasNHWCMishHalfKernel(const half* in, half* out, const half* scale, const half* biases, int sSize, int cSize)
{
#ifdef CUDA_SUPPORTS_FP16
  int cIdx = blockIdx.x * blockDim.x + threadIdx.x;
  int sIdx = blockIdx.y * blockDim.y + threadIdx.y;
  int nIdx = blockIdx.z;
  if(cIdx < cSize && sIdx < sSize) {
    int idx = (nIdx * sSize + sIdx) * cSize + cIdx;
    half a = __hfma(in[idx],scale[cIdx],biases[cIdx]);
    out[idx] = mishh(a);
  }
#else
  //Do nothing, FP16 not supported
#endif
}
__global__
void applyCScaleBiasNHWCGeluHalfKernel(const half* in, half* out, const half* scale, const half* biases, int sSize, int cSize)
{
#ifdef CUDA_SUPPORTS_FP16
  int cIdx = blockIdx.x * blockDim.x + threadIdx.x;
  int sIdx = blockIdx.y * blockDim.y + threadIdx.y;
  int nIdx = blockIdx.z;
  if(cIdx < cSize && sIdx < sSize) {
    int idx = (nIdx * sSize + sIdx) * cSize + cIdx;
    half a = __hfma(in[idx],scale[cIdx],biases[cIdx]);
    out[idx] = geluh(a);
  }
#else
  //Do nothing, FP16 not supported
#endif
}
__global__
void applyCScaleBiasNHWCMaskHalfKernel(const half* in, half* out, const half* scale, const half* biases, const half* mask, int sSize, int cSize)
{
#ifdef CUDA_SUPPORTS_FP16
  int cIdx = blockIdx.x * blockDim.x + threadIdx.x;
  int sIdx = blockIdx.y * blockDim.y + threadIdx.y;
  int nIdx = blockIdx.z;
  if(cIdx < cSize && sIdx < sSize) {
    int idx = (nIdx * sSize + sIdx) * cSize + cIdx;
    out[idx] = __hmul(__hfma(in[idx],scale[cIdx],biases[cIdx]),mask[nIdx*sSize+sIdx]);
  }
#else
  //Do nothing, FP16 not supported
#endif
}
__global__
void applyCScaleBiasNHWCReluMaskHalfKernel(const half* in, half* out, const half* scale, const half* biases, const half* mask, int sSize, int cSize)
{
#ifdef CUDA_SUPPORTS_FP16
  int cIdx = blockIdx.x * blockDim.x + threadIdx.x;
  int sIdx = blockIdx.y * blockDim.y + threadIdx.y;
  int nIdx = blockIdx.z;
  if(cIdx < cSize && sIdx < sSize) {
    int idx = (nIdx * sSize + sIdx) * cSize + cIdx;
    half a = __hmul(__hfma(in[idx],scale[cIdx],biases[cIdx]),mask[nIdx*sSize+sIdx]);
    const half halfzero = __float2half(0.0f);
    out[idx] = __hgt(a,halfzero) ? a : halfzero;
  }
#else
  //Do nothing, FP16 not supported
#endif
}
__global__
void applyCScaleBiasNHWCMishMaskHalfKernel(const half* in, half* out, const half* scale, const half* biases, const half* mask, int sSize, int cSize)
{
#ifdef CUDA_SUPPORTS_FP16
  int cIdx = blockIdx.x * blockDim.x + threadIdx.x;
  int sIdx = blockIdx.y * blockDim.y + threadIdx.y;
  int nIdx = blockIdx.z;
  if(cIdx < cSize && sIdx < sSize) {
    int idx = (nIdx * sSize + sIdx) * cSize + cIdx;
    half a = __hmul(__hfma(in[idx],scale[cIdx],biases[cIdx]),mask[nIdx*sSize+sIdx]);
    out[idx] = mishh(a);
  }
#else
  //Do nothing, FP16 not supported
#endif
}
__global__
void applyCScaleBiasNHWCGeluMaskHalfKernel(const half* in, half* out, const half* scale, const half* biases, const half* mask, int sSize, int cSize)
{
#ifdef CUDA_SUPPORTS_FP16
  int cIdx = blockIdx.x * blockDim.x + threadIdx.x;
  int sIdx = blockIdx.y * blockDim.y + threadIdx.y;
  int nIdx = blockIdx.z;
  if(cIdx < cSize && sIdx < sSize) {
    int idx = (nIdx * sSize + sIdx) * cSize + cIdx;
    half a = __hmul(__hfma(in[idx],scale[cIdx],biases[cIdx]),mask[nIdx*sSize+sIdx]);
    out[idx] = geluh(a);
  }
#else
  //Do nothing, FP16 not supported
#endif
}

void sharedApplyCScaleBiasNHWC(const void* in, void* out, const void* scale, const void* biases, const void* mask, int nSize, int xySize, int cSize, bool isHalf, int activation) {
  if(nSize > 65536)
    throw std::runtime_error("customCudaApplyCScaleBiasNHWC: nSize too large");
  if(xySize > 65536)
    throw std::runtime_error("customCudaApplyCScaleBiasNHWC: xySize too large");

  int sSize = xySize;
  int cThreads;
  int cBlocks;
  int sThreads;
  int sBlocks;
  splitThreadsAcrossDim01(cSize, sSize, cThreads, cBlocks, sThreads, sBlocks);

  dim3 grid(cBlocks,sBlocks,nSize);
  dim3 threads(cThreads,sThreads,1);
  if(mask == NULL) {
    if(activation == ACTIVATION_IDENTITY) {
      if(isHalf)
        applyCScaleBiasNHWCHalfKernel<<<grid,threads>>>((const half*)in,(half*)out,(const half*)scale,(const half*)biases,sSize,cSize);
      else
        applyCScaleBiasNHWCKernel<<<grid,threads>>>((const float*)in,(float*)out,(const float*)scale,(const float*)biases,sSize,cSize);
    }
    else if(activation == ACTIVATION_RELU) {
      if(isHalf)
        applyCScaleBiasNHWCReluHalfKernel<<<grid,threads>>>((const half*)in,(half*)out,(const half*)scale,(const half*)biases,sSize,cSize);
      else
        applyCScaleBiasNHWCReluKernel<<<grid,threads>>>((const float*)in,(float*)out,(const float*)scale,(const float*)biases,sSize,cSize);
    }
    else if(activation == ACTIVATION_MISH) {
      if(isHalf)
        applyCScaleBiasNHWCMishHalfKernel<<<grid,threads>>>((const half*)in,(half*)out,(const half*)scale,(const half*)biases,sSize,cSize);
      else
        applyCScaleBiasNHWCMishKernel<<<grid,threads>>>((const float*)in,(float*)out,(const float*)scale,(const float*)biases,sSize,cSize);
    }
    else if(activation == ACTIVATION_GELU) {
      if(isHalf)
        applyCScaleBiasNHWCGeluHalfKernel<<<grid,threads>>>((const half*)in,(half*)out,(const half*)scale,(const half*)biases,sSize,cSize);
      else
        applyCScaleBiasNHWCGeluKernel<<<grid,threads>>>((const float*)in,(float*)out,(const float*)scale,(const float*)biases,sSize,cSize);
    }
    else {
      throw std::runtime_error("customCudaApplyCScaleBiasNHWC: unsupported activation");
    }
  }
  else {
    if(activation == ACTIVATION_IDENTITY) {
      if(isHalf)
        applyCScaleBiasNHWCMaskHalfKernel<<<grid,threads>>>((const half*)in,(half*)out,(const half*)scale,(const half*)biases,(const half*)mask,sSize,cSize);
      else
        applyCScaleBiasNHWCMaskKernel<<<grid,threads>>>((const float*)in,(float*)out,(const float*)scale,(const float*)biases,(const float*)mask,sSize,cSize);
    }
    else if(activation == ACTIVATION_RELU) {
      if(isHalf)
        applyCScaleBiasNHWCReluMaskHalfKernel<<<grid,threads>>>((const half*)in,(half*)out,(const half*)scale,(const half*)biases,(const half*)mask,sSize,cSize);
      else
        applyCScaleBiasNHWCReluMaskKernel<<<grid,threads>>>((const float*)in,(float*)out,(const float*)scale,(const float*)biases,(const float*)mask,sSize,cSize);
    }
    else if(activation == ACTIVATION_MISH) {
      if(isHalf)
        applyCScaleBiasNHWCMishMaskHalfKernel<<<grid,threads>>>((const half*)in,(half*)out,(const half*)scale,(const half*)biases,(const half*)mask,sSize,cSize);
      else
        applyCScaleBiasNHWCMishMaskKernel<<<grid,threads>>>((const float*)in,(float*)out,(const float*)scale,(const float*)biases,(const float*)mask,sSize,cSize);
    }
    else if(activation == ACTIVATION_GELU) {
      if(isHalf)
        applyCScaleBiasNHWCGeluMaskHalfKernel<<<grid,threads>>>((const half*)in,(half*)out,(const half*)scale,(const half*)biases,(const half*)mask,sSize,cSize);
      else
        applyCScaleBiasNHWCGeluMaskKernel<<<grid,threads>>>((const float*)in,(float*)out,(const float*)scale,(const float*)biases,(const float*)mask,sSize,cSize);
    }
    else {
      throw std::runtime_error("customCudaApplyCScaleBiasNHWC: unsupported activation");
    }
  }
}

void customCudaApplyCScaleBiasNHWC(const float* in, float* out, const float* scale, const float* biases, const float* mask, int nSize, int xySize, int cSize, int activation) {
  sharedApplyCScaleBiasNHWC(in,out,scale,biases,mask,nSize,xySize,cSize,false,activation);
}
void customCudaApplyCScaleBiasNHWC(const half* in, half* out, const half* scale, const half* biases, const half* mask, int nSize, int xySize, int cSize, int activation) {
  sharedApplyCScaleBiasNHWC(in,out,scale,biases,mask,nSize,xySize,cSize,true,activation);
}


//--------------------------------------------------------------------------------------------------------------

template <typename T>
__global__
void dilationTransposeNHWCKernel(
  const T* in,
  T* out,
  int nSize,
  int hSize,
  int wSize,
  int cSize,
  int hOuterSize,
  int wOuterSize
) {
  int cIdx = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  int hoIdx = y / wOuterSize;
  int woIdx = y % wOuterSize;
  int n = blockIdx.z;

  if(cIdx >= cSize || hoIdx >= hOuterSize)
    return;

  // Not sure if this is exactly ideal, but we go ahead and don't bother with shared memory, relying on
  // coalescing to work across the C dimension to get good performance as long as C is a decent power of 2.
  for(int hiIdx = 0; hiIdx < 3; hiIdx++) {
    for(int wiIdx = 0; wiIdx < 3; wiIdx++) {
      int hInIdx = hoIdx * 3 + hiIdx;
      int wInIdx = woIdx * 3 + wiIdx;
      int outIdx = ((((n * 3 + hiIdx) * 3 + wiIdx) * hOuterSize + hoIdx) * wOuterSize + woIdx) * cSize + cIdx;

      if(hInIdx >= hSize || wInIdx >= wSize) {
        out[outIdx] = (T)0;
      }
      else {
        int inIdx = ((n * hSize + hInIdx) * wSize + wInIdx) * cSize + cIdx;
        out[outIdx] = in[inIdx];
      }
    }
  }
}

template <typename T>
void dilationTransposeNHWCTemplate(const T* in, T* out, int nSize, int hSize, int wSize, int cSize) {
  if(nSize > 65536)
    throw std::runtime_error("customCudaDilationTransposeNHWC: nSize too large");
  if(hSize*wSize > 65536)
    throw std::runtime_error("customCudaDilationTransposeNHWC: hSize*wSize too large");

  int hOuterSize = (hSize + 2) / 3;
  int wOuterSize = (wSize + 2) / 3;

  int cThreads = 1;
  while(cThreads < targetNumThreads && cThreads < cSize)
    cThreads *= 2;
  int cBlocks = (cSize + cThreads - 1) / cThreads;

  dim3 grid(cBlocks, hOuterSize * wOuterSize, nSize);
  dim3 threads(cThreads, 1, 1);

  dilationTransposeNHWCKernel<<<grid,threads>>>(in, out, nSize, hSize, wSize, cSize, hOuterSize, wOuterSize);
}

template <typename T>
__global__
void dilationUntransposeNHWCKernel(
  const T* in,
  T* out,
  int nSize,
  int hSize,
  int wSize,
  int cSize,
  int hOuterSize,
  int wOuterSize
) {
  int cIdx = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  int hoIdx = y / wOuterSize;
  int woIdx = y % wOuterSize;
  int n = blockIdx.z;

  if(cIdx >= cSize || hoIdx >= hOuterSize)
    return;

  // Not sure if this is exactly ideal, but we go ahead and don't bother with shared memory, relying on
  // coalescing to work across the C dimension to get good performance as long as C is a decent power of 2.
  for(int hiIdx = 0; hiIdx < 3; hiIdx++) {
    for(int wiIdx = 0; wiIdx < 3; wiIdx++) {
      int hOutIdx = hoIdx * 3 + hiIdx;
      int wOutIdx = woIdx * 3 + wiIdx;

      if(hOutIdx < hSize && wOutIdx < wSize) {
        int inIdx = ((((n * 3 + hiIdx) * 3 + wiIdx) * hOuterSize + hoIdx) * wOuterSize + woIdx) * cSize + cIdx;
        int outIdx = ((n * hSize + hOutIdx) * wSize + wOutIdx) * cSize + cIdx;
        out[outIdx] = in[inIdx];
      }
    }
  }
}

template <typename T>
void dilationUntransposeNHWCTemplate(const T* in, T* out, int nSize, int hSize, int wSize, int cSize) {
  if(nSize > 65536)
    throw std::runtime_error("customCudaDilationUntransposeNHWC: nSize too large");
  if(hSize*wSize > 65536)
    throw std::runtime_error("customCudaDilationUntransposeNHWC: hSize*wSize too large");

  int hOuterSize = (hSize + 2) / 3;
  int wOuterSize = (wSize + 2) / 3;

  int cThreads = 1;
  while(cThreads < targetNumThreads && cThreads < cSize)
    cThreads *= 2;
  int cBlocks = (cSize + cThreads - 1) / cThreads;

  dim3 grid(cBlocks, hOuterSize * wOuterSize, nSize);
  dim3 threads(cThreads, 1, 1);

  dilationUntransposeNHWCKernel<<<grid,threads>>>(in, out, nSize, hSize, wSize, cSize, hOuterSize, wOuterSize);
}

// See header file for semantics
void customCudaDilationTransposeNHWC(const float* in, float* out, int nSize, int hSize, int wSize, int cSize) {
  dilationTransposeNHWCTemplate<float>(in, out, nSize, hSize, wSize, cSize);
}

void customCudaDilationTransposeNHWC(const half* in, half* out, int nSize, int hSize, int wSize, int cSize) {
  dilationTransposeNHWCTemplate<half>(in, out, nSize, hSize, wSize, cSize);
}
// Inverse transforms
void customCudaDilationUntransposeNHWC(const float* in, float* out, int nSize, int hSize, int wSize, int cSize) {
  dilationUntransposeNHWCTemplate<float>(in, out, nSize, hSize, wSize, cSize);
}

void customCudaDilationUntransposeNHWC(const half* in, half* out, int nSize, int hSize, int wSize, int cSize) {
  dilationUntransposeNHWCTemplate<half>(in, out, nSize, hSize, wSize, cSize);
}

//--------------------------------------------------------------------------------------------------------------

// For floats, this is 32kb. Max shared memory on CUDA per block is 48kb.
static constexpr int MAX_DT_TILE_SIZE = 8192;

template <typename T>
__global__
void dilationTransposeNCHWKernel(
  const T* in,
  T* out,
  int nSize,
  int cSize,
  int hSize,
  int wSize,
  int hOuterSize,
  int wOuterSize,
  int cProcessedPerBlock,
  int eltsInputPerThread,
  int eltsOutputPerThread
) {
  extern __shared__ double tileShared[]; // use double just for alignment
  T *tile = reinterpret_cast<T*>(tileShared);

  int x = threadIdx.x;
  int threadsPerBlock = blockDim.x;
  int cIdxBase = blockIdx.x * cProcessedPerBlock;
  int n = blockIdx.y;

  // A block of threads reads a chunk in[n, cIdxBase:cIdxBase+cProcessedPerBlock, :, :] sequentially (i.e. grid-stridedly) from in's perspective.
  // And writes it in the same order into tile[0:cProcessedPerBlock, :, :]

  int hwSize = hSize*wSize;
  int inBase = (n * cSize + cIdxBase) * hwSize;
  for(int i = 0; i < eltsInputPerThread; i++) {
    int inOffset = i*threadsPerBlock+x;
    int cOffset = inOffset / hwSize;
    int cIdx = cIdxBase + cOffset;
    if(cIdx < cSize && cOffset < cProcessedPerBlock)
      tile[inOffset] = in[inBase + inOffset];
  }

  __syncthreads();

  // Now, the block of threads writes to out[n,0:3,0:3,cIdxBase:cIdxBase+cProcessedPerBlock, :, :] sequentially (i.e. grid-stridedly) from out's perspective.
  // by reading from tile[0:cProcessedPerBlock, :, :] doing the index computations necessary to figure out where in the tile to look.
  int howoSize = hOuterSize*wOuterSize;
  int cphowoSize = cProcessedPerBlock * howoSize;
  for(int i = 0; i < eltsOutputPerThread; i++) {
    int outCounter = i*threadsPerBlock+x;
    int hiwiIdx = outCounter / cphowoSize;
    if(hiwiIdx < 9) {
      int cphowoIdx = outCounter % cphowoSize;
      int cOffset = cphowoIdx / howoSize;
      int cIdx = cIdxBase + cOffset;
      if(cIdx < cSize && cOffset < cProcessedPerBlock) {
        int howoIdx = cphowoIdx % howoSize;
        int hoIdx = howoIdx / wOuterSize;
        int woIdx = howoIdx % wOuterSize;

        int hInIdx = hoIdx * 3 + hiwiIdx / 3;
        int wInIdx = woIdx * 3 + hiwiIdx % 3;

        int outIdx = ((n * 9 + hiwiIdx) * cSize + cIdx) * howoSize + howoIdx;
        if(hInIdx >= hSize || wInIdx >= wSize) {
          out[outIdx] = (T)0;
        }
        else {
          out[outIdx] = tile[(cOffset * hSize + hInIdx) * wSize + wInIdx];
        }
      }
    }
  }
}

template <typename T>
void dilationTransposeNCHWTemplate(const T* in, T* out, int nSize, int cSize, int hSize, int wSize) {
  if(nSize > 65536)
    throw std::runtime_error("customCudaDilationTransposeNCHW: nSize too large");
  if(hSize * wSize > MAX_DT_TILE_SIZE)
    throw std::runtime_error("customCudaDilationTransposeNCHW: hSize*wSize too large");

  int hOuterSize = (hSize + 2) / 3;
  int wOuterSize = (wSize + 2) / 3;

  // Reduce threads if it would be overkill for the total number of values per batch, but otherwise go up to 512
  int threadsPerBlock = 32;
  while(threadsPerBlock < cSize * hOuterSize * wOuterSize * 9 && threadsPerBlock < 512) {
    threadsPerBlock *= 2;
  }

  // Process as many channels as we can at a time subject to shared memory tile size limit
  int cProcessedPerBlock = 1;
  while(cProcessedPerBlock * 2 < cSize && cProcessedPerBlock * 2 * hSize * wSize < MAX_DT_TILE_SIZE) {
    cProcessedPerBlock *= 2;
  }

  size_t sharedMemSize = std::max((size_t)64, sizeof(T) * cProcessedPerBlock * hSize * wSize);

  int eltsInputPerThread = (cProcessedPerBlock * hSize * wSize + (threadsPerBlock-1)) / threadsPerBlock;
  int eltsOutputPerThread = (cProcessedPerBlock * hOuterSize * wOuterSize * 9 + (threadsPerBlock-1)) / threadsPerBlock;

  int numStepsToProcessAllOfC = (cSize+cProcessedPerBlock-1)/cProcessedPerBlock;
  dim3 grid(numStepsToProcessAllOfC, nSize, 1);
  dim3 threads(threadsPerBlock, 1, 1);

  dilationTransposeNCHWKernel<<<grid,threads,sharedMemSize>>>(in, out, nSize, cSize, hSize, wSize, hOuterSize, wOuterSize, cProcessedPerBlock, eltsInputPerThread, eltsOutputPerThread);
}

template <typename T>
__global__
void dilationUntransposeNCHWKernel(
  const T* in,
  T* out,
  int nSize,
  int cSize,
  int hSize,
  int wSize,
  int hOuterSize,
  int wOuterSize,
  int cProcessedPerBlock,
  int eltsInputPerThread,
  int eltsOutputPerThread
) {
  extern __shared__ double tileShared[]; // use double just for alignment
  T *tile = reinterpret_cast<T*>(tileShared);

  int x = threadIdx.x;
  int threadsPerBlock = blockDim.x;
  int cIdxBase = blockIdx.x * cProcessedPerBlock;
  int n = blockIdx.y;

  // A block of threads reads from in[n,0:3,0:3,cIdxBase:cIdxBase+cProcessedPerBlock, :, :] sequentially (i.e. grid-stridedly) from out's perspective.
  // writing to tile[0:cProcessedPerBlock, :, :] doing the index computations necessary to figure out where in the tile to look.

  int howoSize = hOuterSize*wOuterSize;
  int cphowoSize = cProcessedPerBlock * howoSize;
  for(int i = 0; i < eltsInputPerThread; i++) {
    int inCounter = i*threadsPerBlock+x;
    int hiwiIdx = inCounter / cphowoSize;
    if(hiwiIdx < 9) {
      int cphowoIdx = inCounter % cphowoSize;
      int cOffset = cphowoIdx / howoSize;
      int cIdx = cIdxBase + cOffset;
      if(cIdx < cSize && cOffset < cProcessedPerBlock) {
        int howoIdx = cphowoIdx % howoSize;
        int hoIdx = howoIdx / wOuterSize;
        int woIdx = howoIdx % wOuterSize;

        int hOutIdx = hoIdx * 3 + hiwiIdx / 3;
        int wOutIdx = woIdx * 3 + hiwiIdx % 3;

        int inIdx = ((n * 9 + hiwiIdx) * cSize + cIdx) * howoSize + howoIdx;
        if(hOutIdx >= hSize || wOutIdx >= wSize) {
          // pass
        }
        else {
          tile[(cOffset * hSize + hOutIdx) * wSize + wOutIdx] = in[inIdx];
        }
      }
    }
  }

  __syncthreads();

  // Now, the block of threads writes to out[n, cIdxBase:cIdxBase+cProcessedPerBlock, :, :] sequentially (i.e. grid-stridedly) from in's perspective.
  // And writes it in the same order from tile[0:cProcessedPerBlock, :, :]

  int hwSize = hSize*wSize;
  int outBase = (n * cSize + cIdxBase) * hwSize;
  for(int i = 0; i < eltsInputPerThread; i++) {
    int outOffset = i*threadsPerBlock+x;
    int cOffset = outOffset / hwSize;
    int cIdx = cIdxBase + cOffset;
    if(cIdx < cSize && cOffset < cProcessedPerBlock)
      out[outBase + outOffset] = tile[outOffset];
  }
}

template <typename T>
void dilationUntransposeNCHWTemplate(const T* in, T* out, int nSize, int cSize, int hSize, int wSize) {
  if(nSize > 65536)
    throw std::runtime_error("customCudaDilationUntransposeNCHW: nSize too large");
  if(hSize * wSize > MAX_DT_TILE_SIZE)
    throw std::runtime_error("customCudaDilationUntransposeNCHW: hSize*wSize too large");

  int hOuterSize = (hSize + 2) / 3;
  int wOuterSize = (wSize + 2) / 3;

  // Reduce threads if it would be overkill for the total number of values per batch, but otherwise go up to 512
  int threadsPerBlock = 32;
  while(threadsPerBlock < cSize * hOuterSize * wOuterSize * 9 && threadsPerBlock < 512) {
    threadsPerBlock *= 2;
  }

  // Process as many channels as we can at a time subject to shared memory tile size limit
  int cProcessedPerBlock = 1;
  while(cProcessedPerBlock * 2 < cSize && cProcessedPerBlock * 2 * hSize * wSize < MAX_DT_TILE_SIZE) {
    cProcessedPerBlock *= 2;
  }

  size_t sharedMemSize = std::max((size_t)64, sizeof(T) * cProcessedPerBlock * hSize * wSize);

  int eltsInputPerThread = (cProcessedPerBlock * hOuterSize * wOuterSize * 9 + (threadsPerBlock-1)) / threadsPerBlock;
  int eltsOutputPerThread = (cProcessedPerBlock * hSize * wSize + (threadsPerBlock-1)) / threadsPerBlock;

  int numStepsToProcessAllOfC = (cSize+cProcessedPerBlock-1)/cProcessedPerBlock;
  dim3 grid(numStepsToProcessAllOfC, nSize, 1);
  dim3 threads(threadsPerBlock, 1, 1);

  dilationUntransposeNCHWKernel<<<grid,threads,sharedMemSize>>>(in, out, nSize, cSize, hSize, wSize, hOuterSize, wOuterSize, cProcessedPerBlock, eltsInputPerThread, eltsOutputPerThread);
}

// See header file for semantics
void customCudaDilationTransposeNCHW(const float* in, float* out, int nSize, int cSize, int hSize, int wSize) {
  dilationTransposeNCHWTemplate<float>(in, out, nSize, cSize, hSize, wSize);
}

void customCudaDilationTransposeNCHW(const half* in, half* out, int nSize, int cSize, int hSize, int wSize) {
  dilationTransposeNCHWTemplate<half>(in, out, nSize, cSize, hSize, wSize);
}
// Inverse transforms
void customCudaDilationUntransposeNCHW(const float* in, float* out, int nSize, int cSize, int hSize, int wSize) {
  dilationUntransposeNCHWTemplate<float>(in, out, nSize, cSize, hSize, wSize);
}

void customCudaDilationUntransposeNCHW(const half* in, half* out, int nSize, int cSize, int hSize, int wSize) {
  dilationUntransposeNCHWTemplate<half>(in, out, nSize, cSize, hSize, wSize);
}

//--------------------------------------------------------------------------------------------------------------

__global__
void dilationFillMaskFloatKernel(
  float* out,
  int nSize,
  int hSize,
  int wSize,
  int hOuterSize,
  int wOuterSize,
  int eltsOutputPerThread
) {
  int x = threadIdx.x;
  int threadsPerBlock = blockDim.x;
  int n = blockIdx.y;

  // Now, the block of threads writes to out[n,0:3,0:3, :, :] sequentially (i.e. grid-stridedly) from out's perspective.
  int howoSize = hOuterSize*wOuterSize;
  for(int i = 0; i < eltsOutputPerThread; i++) {
    int outCounter = i*threadsPerBlock+x;
    int hiwiIdx = outCounter / howoSize;
    if(hiwiIdx < 9) {
      int howoIdx = outCounter % howoSize;
      int hoIdx = howoIdx / wOuterSize;
      int woIdx = howoIdx % wOuterSize;

      int hInIdx = hoIdx * 3 + hiwiIdx / 3;
      int wInIdx = woIdx * 3 + hiwiIdx % 3;

      int outIdx = (n * 9 + hiwiIdx) * howoSize + howoIdx;
      if(hInIdx >= hSize || wInIdx >= wSize) {
        out[outIdx] = 0.0f;
      }
      else {
        out[outIdx] = 1.0f;
      }
    }
  }
}
__global__
void dilationFillMaskHalfKernel(
  half* out,
  int nSize,
  int hSize,
  int wSize,
  int hOuterSize,
  int wOuterSize,
  int eltsOutputPerThread
) {
#ifdef CUDA_SUPPORTS_FP16
  int x = threadIdx.x;
  int threadsPerBlock = blockDim.x;
  int n = blockIdx.y;

  // Now, the block of threads writes to out[n,0:3,0:3, :, :] sequentially (i.e. grid-stridedly) from out's perspective.
  int howoSize = hOuterSize*wOuterSize;
  for(int i = 0; i < eltsOutputPerThread; i++) {
    int outCounter = i*threadsPerBlock+x;
    int hiwiIdx = outCounter / howoSize;
    if(hiwiIdx < 9) {
      int howoIdx = outCounter % howoSize;
      int hoIdx = howoIdx / wOuterSize;
      int woIdx = howoIdx % wOuterSize;

      int hInIdx = hoIdx * 3 + hiwiIdx / 3;
      int wInIdx = woIdx * 3 + hiwiIdx % 3;

      int outIdx = (n * 9 + hiwiIdx) * howoSize + howoIdx;
      if(hInIdx >= hSize || wInIdx >= wSize) {
        out[outIdx] = (half)0;
      }
      else {
        out[outIdx] = __float2half(1.0f);
      }
    }
  }
#else
  //Do nothing, FP16 not supported
#endif
}

void customCudaDilationFillMask(float* out, int nSize, int hSize, int wSize) {
  if(nSize > 65536)
    throw std::runtime_error("customCudaDilationFillMask: nSize too large");
  if(hSize * wSize > MAX_DT_TILE_SIZE)
    throw std::runtime_error("customCudaDilationFillMask: hSize*wSize too large");

  int hOuterSize = (hSize + 2) / 3;
  int wOuterSize = (wSize + 2) / 3;

  // Reduce threads if it would be overkill for the total number of values per batch, but otherwise go up to 512
  int threadsPerBlock = 32;
  while(threadsPerBlock < hOuterSize * wOuterSize * 9 && threadsPerBlock < 512) {
    threadsPerBlock *= 2;
  }

  int eltsOutputPerThread = (hOuterSize * wOuterSize * 9 + (threadsPerBlock-1)) / threadsPerBlock;

  dim3 grid(1, nSize, 1);
  dim3 threads(threadsPerBlock, 1, 1);

  dilationFillMaskFloatKernel<<<grid,threads>>>(out, nSize, hSize, wSize, hOuterSize, wOuterSize, eltsOutputPerThread);
}

void customCudaDilationFillMask(half* out, int nSize, int hSize, int wSize) {
  if(nSize > 65536)
    throw std::runtime_error("customCudaDilationFillMask: nSize too large");
  if(hSize * wSize > MAX_DT_TILE_SIZE)
    throw std::runtime_error("customCudaDilationFillMask: hSize*wSize too large");

  int hOuterSize = (hSize + 2) / 3;
  int wOuterSize = (wSize + 2) / 3;

  // Reduce threads if it would be overkill for the total number of values per batch, but otherwise go up to 512
  int threadsPerBlock = 32;
  while(threadsPerBlock < hOuterSize * wOuterSize * 9 && threadsPerBlock < 512) {
    threadsPerBlock *= 2;
  }

  int eltsOutputPerThread = (hOuterSize * wOuterSize * 9 + (threadsPerBlock-1)) / threadsPerBlock;

  dim3 grid(1, nSize, 1);
  dim3 threads(threadsPerBlock, 1, 1);

  dilationFillMaskHalfKernel<<<grid,threads>>>(out, nSize, hSize, wSize, hOuterSize, wOuterSize, eltsOutputPerThread);
}

