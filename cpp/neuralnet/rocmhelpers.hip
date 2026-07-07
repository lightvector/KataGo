#include "hip/hip_runtime.h"

#include "../neuralnet/rocmhelpers.h"

#include <stdexcept>
#include <string>

#if defined(__HIP_ARCH_HAS_FP16__) || (defined(__HIP_DEVICE_COMPILE__) && (__HIP_ARCH_GFX803__ || __HIP_ARCH_GFX900__))
#define HIP_SUPPORTS_FP16
#endif

//TODO maybe tune this number, it varies by GPU
static const int targetNumThreads = 512;

// The custom kernels below compute flattened element indices in 32-bit int. Guard the launchers:
// throw loudly if the total element count does not fit in a positive int, rather than silently
// overflowing into an out-of-bounds access.
static void checkBufferIndexFitsInt(int64_t a, int64_t b, int64_t c, const char* whatKernel) {
  int64_t total = a * b * c;
  if(total >= (int64_t)2147483647)
    throw std::runtime_error(
      std::string(whatKernel) + ": total element count " + std::to_string(total) +
      " exceeds the 32-bit index limit used by this kernel");
}

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
__forceinline__ __device__ float mishf_scale8(float a) {
  return a < 2.5f ? a * tanhf(log1pf(expf(a*8.0f))) : a;
}
__forceinline__ __device__ float siluf(float a) {
  return a / (1.0f + expf(-a));
}

#ifdef HIP_SUPPORTS_FP16
__forceinline__ __device__ half mishh(half h) {
  float a = __half2float(h);
  return __float2half(a * tanhf(a < 20.0f ? log1pf(expf(a)) : a));
}
__forceinline__ __device__ half mishh_scale8(half h) {
  float a = __half2float(h);
  return __float2half(a < 2.5f ? a * tanhf(log1pf(expf(a*8.0f))) : a);
}
__forceinline__ __device__ half siluh(half h) {
  float a = __half2float(h);
  return __float2half(a / (1.0f + expf(-a)));
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
#ifdef HIP_SUPPORTS_FP16
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
#ifdef HIP_SUPPORTS_FP16
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
#ifdef HIP_SUPPORTS_FP16
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
#ifdef HIP_SUPPORTS_FP16
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


//--------------------------------------------------------------------------------------------------------------
// Element-wise tensor add: buf[i] += toAdd[i], for float and half

__global__
void addTensorsInplaceKernel(float *buf, const float* toAdd, int nSize)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if(idx < nSize) {
    buf[idx] += toAdd[idx];
  }
}
void customCudaAddTensorsInplace(float* buf, const float* toAdd, int nSize) {
  int blockSize = targetNumThreads;
  int numBlocks = (nSize+blockSize-1)/blockSize;
  addTensorsInplaceKernel<<<numBlocks, blockSize>>>(buf,toAdd,nSize);
}

__global__
void addTensorsInplaceHalfKernel(half *buf, const half* toAdd, int nSize)
{
#ifdef HIP_SUPPORTS_FP16
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if(idx < nSize) {
    buf[idx] = __hadd(buf[idx],toAdd[idx]);
  }
#else
  //Do nothing, FP16 not supported
#endif
}
void customCudaAddTensorsInplace(half* buf, const half* toAdd, int nSize) {
  int blockSize = targetNumThreads;
  int numBlocks = (nSize+blockSize-1)/blockSize;
  addTensorsInplaceHalfKernel<<<numBlocks, blockSize>>>(buf,toAdd,nSize);
}

//--------------------------------------------------------------------------------------------------------------

__global__
void addTensorInplaceHalfKernel(half *buf, const half* biases, int nSize)
{
#ifdef HIP_SUPPORTS_FP16
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
#ifdef HIP_SUPPORTS_FP16
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
#ifdef HIP_SUPPORTS_FP16
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
#ifdef HIP_SUPPORTS_FP16
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
void addCBiasInplaceNCKernelMishScale8(float *buf, const float* biases, int nSize, int cSize)
{
  int cIdx = blockIdx.x * blockDim.x + threadIdx.x;
  int nIdx = blockIdx.y * blockDim.y + threadIdx.y;
  if(cIdx < cSize && nIdx < nSize) {
    int idx = nIdx * cSize + cIdx;
    buf[idx] = mishf_scale8(buf[idx] + biases[cIdx]);
  }
}
__global__
void addCBiasInplaceNCHalfKernelMishScale8(half *buf, const half* biases, int nSize, int cSize)
{
#ifdef HIP_SUPPORTS_FP16
  int cIdx = blockIdx.x * blockDim.x + threadIdx.x;
  int nIdx = blockIdx.y * blockDim.y + threadIdx.y;
  if(cIdx < cSize && nIdx < nSize) {
    int idx = nIdx * cSize + cIdx;
    half a = __hadd(buf[idx],biases[cIdx]);
    buf[idx] = mishh_scale8(a);
  }
#else
  //Do nothing, FP16 not supported
#endif
}
__global__
void addCBiasInplaceNCKernelSilu(float *buf, const float* biases, int nSize, int cSize)
{
  int cIdx = blockIdx.x * blockDim.x + threadIdx.x;
  int nIdx = blockIdx.y * blockDim.y + threadIdx.y;
  if(cIdx < cSize && nIdx < nSize) {
    int idx = nIdx * cSize + cIdx;
    buf[idx] = siluf(buf[idx] + biases[cIdx]);
  }
}
__global__
void addCBiasInplaceNCHalfKernelSilu(half *buf, const half* biases, int nSize, int cSize)
{
#ifdef HIP_SUPPORTS_FP16
  int cIdx = blockIdx.x * blockDim.x + threadIdx.x;
  int nIdx = blockIdx.y * blockDim.y + threadIdx.y;
  if(cIdx < cSize && nIdx < nSize) {
    int idx = nIdx * cSize + cIdx;
    half a = __hadd(buf[idx],biases[cIdx]);
    buf[idx] = siluh(a);
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
  else if(activation == ACTIVATION_MISH_SCALE8) {
    if(isHalf)
      addCBiasInplaceNCHalfKernelMishScale8<<<grid,threads>>>((half*)buf,(const half*)biases,nSize,cSize);
    else
      addCBiasInplaceNCKernelMishScale8<<<grid,threads>>>((float*)buf,(const float*)biases,nSize,cSize);
  }
  else if(activation == ACTIVATION_SILU) {
    if(isHalf)
      addCBiasInplaceNCHalfKernelSilu<<<grid,threads>>>((half*)buf,(const half*)biases,nSize,cSize);
    else
      addCBiasInplaceNCKernelSilu<<<grid,threads>>>((float*)buf,(const float*)biases,nSize,cSize);
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
#ifdef HIP_SUPPORTS_FP16
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
#ifdef HIP_SUPPORTS_FP16
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
void applyCScaleBiasNCHWMishScale8Kernel(const float *in, float* out, const float* scale, const float* biases, int cSize, int sSize)
{
  int sIdx = blockIdx.x * blockDim.x + threadIdx.x;
  int cIdx = blockIdx.y * blockDim.y + threadIdx.y;
  int nIdx = blockIdx.z;
  if(cIdx < cSize && sIdx < sSize) {
    int idx = (nIdx * cSize + cIdx) * sSize + sIdx;
    out[idx] = mishf_scale8(in[idx] * scale[cIdx] + biases[cIdx]);
  }
}
__global__
void applyCScaleBiasNCHWSiluKernel(const float *in, float* out, const float* scale, const float* biases, int cSize, int sSize)
{
  int sIdx = blockIdx.x * blockDim.x + threadIdx.x;
  int cIdx = blockIdx.y * blockDim.y + threadIdx.y;
  int nIdx = blockIdx.z;
  if(cIdx < cSize && sIdx < sSize) {
    int idx = (nIdx * cSize + cIdx) * sSize + sIdx;
    out[idx] = siluf(in[idx] * scale[cIdx] + biases[cIdx]);
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
void applyCScaleBiasNCHWMishScale8MaskKernel(const float *in, float* out, const float* scale, const float* biases, const float* mask, int cSize, int sSize)
{
  int sIdx = blockIdx.x * blockDim.x + threadIdx.x;
  int cIdx = blockIdx.y * blockDim.y + threadIdx.y;
  int nIdx = blockIdx.z;
  if(cIdx < cSize && sIdx < sSize) {
    int idx = (nIdx * cSize + cIdx) * sSize + sIdx;
    out[idx] = mishf_scale8(in[idx] * scale[cIdx] + biases[cIdx]) * mask[nIdx*sSize+sIdx];
  }
}
__global__
void applyCScaleBiasNCHWSiluMaskKernel(const float *in, float* out, const float* scale, const float* biases, const float* mask, int cSize, int sSize)
{
  int sIdx = blockIdx.x * blockDim.x + threadIdx.x;
  int cIdx = blockIdx.y * blockDim.y + threadIdx.y;
  int nIdx = blockIdx.z;
  if(cIdx < cSize && sIdx < sSize) {
    int idx = (nIdx * cSize + cIdx) * sSize + sIdx;
    out[idx] = siluf(in[idx] * scale[cIdx] + biases[cIdx]) * mask[nIdx*sSize+sIdx];
  }
}
__global__
void applyCScaleBiasNCHWHalfKernel(const half *in, half* out, const half* scale, const half* biases, int cSize, int sSize)
{
#ifdef HIP_SUPPORTS_FP16
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
#ifdef HIP_SUPPORTS_FP16
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
#ifdef HIP_SUPPORTS_FP16
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
void applyCScaleBiasNCHWMishScale8HalfKernel(const half *in, half* out, const half* scale, const half* biases, int cSize, int sSize)
{
#ifdef HIP_SUPPORTS_FP16
  int sIdx = blockIdx.x * blockDim.x + threadIdx.x;
  int cIdx = blockIdx.y * blockDim.y + threadIdx.y;
  int nIdx = blockIdx.z;
  if(cIdx < cSize && sIdx < sSize) {
    int idx = (nIdx * cSize + cIdx) * sSize + sIdx;
    half a = __hfma(in[idx],scale[cIdx],biases[cIdx]);
    out[idx] = mishh_scale8(a);
  }
#else
  //Do nothing, FP16 not supported
#endif
}
__global__
void applyCScaleBiasNCHWSiluHalfKernel(const half *in, half* out, const half* scale, const half* biases, int cSize, int sSize)
{
#ifdef HIP_SUPPORTS_FP16
  int sIdx = blockIdx.x * blockDim.x + threadIdx.x;
  int cIdx = blockIdx.y * blockDim.y + threadIdx.y;
  int nIdx = blockIdx.z;
  if(cIdx < cSize && sIdx < sSize) {
    int idx = (nIdx * cSize + cIdx) * sSize + sIdx;
    half a = __hfma(in[idx],scale[cIdx],biases[cIdx]);
    out[idx] = siluh(a);
  }
#else
  //Do nothing, FP16 not supported
#endif
}
__global__
void applyCScaleBiasNCHWMaskHalfKernel(const half *in, half* out, const half* scale, const half* biases, const half* mask, int cSize, int sSize)
{
#ifdef HIP_SUPPORTS_FP16
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
#ifdef HIP_SUPPORTS_FP16
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
#ifdef HIP_SUPPORTS_FP16
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
void applyCScaleBiasNCHWMishScale8MaskHalfKernel(const half *in, half* out, const half* scale, const half* biases, const half* mask, int cSize, int sSize)
{
#ifdef HIP_SUPPORTS_FP16
  int sIdx = blockIdx.x * blockDim.x + threadIdx.x;
  int cIdx = blockIdx.y * blockDim.y + threadIdx.y;
  int nIdx = blockIdx.z;
  if(cIdx < cSize && sIdx < sSize) {
    int idx = (nIdx * cSize + cIdx) * sSize + sIdx;
    half a = __hmul(__hfma(in[idx],scale[cIdx],biases[cIdx]),mask[nIdx*sSize+sIdx]);
    out[idx] = mishh_scale8(a);
  }
#else
  //Do nothing, FP16 not supported
#endif
}

__global__
void applyCScaleBiasNCHWSiluMaskHalfKernel(const half *in, half* out, const half* scale, const half* biases, const half* mask, int cSize, int sSize)
{
#ifdef HIP_SUPPORTS_FP16
  int sIdx = blockIdx.x * blockDim.x + threadIdx.x;
  int cIdx = blockIdx.y * blockDim.y + threadIdx.y;
  int nIdx = blockIdx.z;
  if(cIdx < cSize && sIdx < sSize) {
    int idx = (nIdx * cSize + cIdx) * sSize + sIdx;
    half a = __hmul(__hfma(in[idx],scale[cIdx],biases[cIdx]),mask[nIdx*sSize+sIdx]);
    out[idx] = siluh(a);
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
    else if(activation == ACTIVATION_MISH_SCALE8) {
      if(isHalf)
        applyCScaleBiasNCHWMishScale8HalfKernel<<<grid,threads>>>((const half*)in,(half*)out,(const half*)scale,(const half*)biases,cSize,sSize);
      else
        applyCScaleBiasNCHWMishScale8Kernel<<<grid,threads>>>((const float*)in,(float*)out,(const float*)scale,(const float*)biases,cSize,sSize);
    }
    else if(activation == ACTIVATION_SILU) {
      if(isHalf)
        applyCScaleBiasNCHWSiluHalfKernel<<<grid,threads>>>((const half*)in,(half*)out,(const half*)scale,(const half*)biases,cSize,sSize);
      else
        applyCScaleBiasNCHWSiluKernel<<<grid,threads>>>((const float*)in,(float*)out,(const float*)scale,(const float*)biases,cSize,sSize);
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
    else if(activation == ACTIVATION_MISH_SCALE8) {
      if(isHalf)
        applyCScaleBiasNCHWMishScale8MaskHalfKernel<<<grid,threads>>>((const half*)in,(half*)out,(const half*)scale,(const half*)biases,(const half*)mask,cSize,sSize);
      else
        applyCScaleBiasNCHWMishScale8MaskKernel<<<grid,threads>>>((const float*)in,(float*)out,(const float*)scale,(const float*)biases,(const float*)mask,cSize,sSize);
    }
    else if(activation == ACTIVATION_SILU) {
      if(isHalf)
        applyCScaleBiasNCHWSiluMaskHalfKernel<<<grid,threads>>>((const half*)in,(half*)out,(const half*)scale,(const half*)biases,(const half*)mask,cSize,sSize);
      else
        applyCScaleBiasNCHWSiluMaskKernel<<<grid,threads>>>((const float*)in,(float*)out,(const float*)scale,(const float*)biases,(const float*)mask,cSize,sSize);
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
void applyCScaleBiasNHWCMishScale8Kernel(const float* in, float* out, const float* scale, const float* biases, int sSize, int cSize)
{
  int cIdx = blockIdx.x * blockDim.x + threadIdx.x;
  int sIdx = blockIdx.y * blockDim.y + threadIdx.y;
  int nIdx = blockIdx.z;
  if(cIdx < cSize && sIdx < sSize) {
    int idx = (nIdx * sSize + sIdx) * cSize + cIdx;
    out[idx] = mishf_scale8(in[idx] * scale[cIdx] + biases[cIdx]);
  }
}
__global__
void applyCScaleBiasNHWCSiluKernel(const float* in, float* out, const float* scale, const float* biases, int sSize, int cSize)
{
  int cIdx = blockIdx.x * blockDim.x + threadIdx.x;
  int sIdx = blockIdx.y * blockDim.y + threadIdx.y;
  int nIdx = blockIdx.z;
  if(cIdx < cSize && sIdx < sSize) {
    int idx = (nIdx * sSize + sIdx) * cSize + cIdx;
    out[idx] = siluf(in[idx] * scale[cIdx] + biases[cIdx]);
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
void applyCScaleBiasNHWCMishScale8MaskKernel(const float* in, float* out, const float* scale, const float* biases, const float* mask, int sSize, int cSize)
{
  int cIdx = blockIdx.x * blockDim.x + threadIdx.x;
  int sIdx = blockIdx.y * blockDim.y + threadIdx.y;
  int nIdx = blockIdx.z;
  if(cIdx < cSize && sIdx < sSize) {
    int idx = (nIdx * sSize + sIdx) * cSize + cIdx;
    out[idx] = mishf_scale8(in[idx] * scale[cIdx] + biases[cIdx]) * mask[nIdx*sSize+sIdx];
  }
}
__global__
void applyCScaleBiasNHWCSiluMaskKernel(const float* in, float* out, const float* scale, const float* biases, const float* mask, int sSize, int cSize)
{
  int cIdx = blockIdx.x * blockDim.x + threadIdx.x;
  int sIdx = blockIdx.y * blockDim.y + threadIdx.y;
  int nIdx = blockIdx.z;
  if(cIdx < cSize && sIdx < sSize) {
    int idx = (nIdx * sSize + sIdx) * cSize + cIdx;
    out[idx] = siluf(in[idx] * scale[cIdx] + biases[cIdx]) * mask[nIdx*sSize+sIdx];
  }
}
__global__
void applyCScaleBiasNHWCHalfKernel(const half* in, half* out, const half* scale, const half* biases, int sSize, int cSize)
{
#ifdef HIP_SUPPORTS_FP16
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
#ifdef HIP_SUPPORTS_FP16
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
#ifdef HIP_SUPPORTS_FP16
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
void applyCScaleBiasNHWCMishScale8HalfKernel(const half* in, half* out, const half* scale, const half* biases, int sSize, int cSize)
{
#ifdef HIP_SUPPORTS_FP16
  int cIdx = blockIdx.x * blockDim.x + threadIdx.x;
  int sIdx = blockIdx.y * blockDim.y + threadIdx.y;
  int nIdx = blockIdx.z;
  if(cIdx < cSize && sIdx < sSize) {
    int idx = (nIdx * sSize + sIdx) * cSize + cIdx;
    half a = __hfma(in[idx],scale[cIdx],biases[cIdx]);
    out[idx] = mishh_scale8(a);
  }
#else
  //Do nothing, FP16 not supported
#endif
}
__global__
void applyCScaleBiasNHWCSiluHalfKernel(const half* in, half* out, const half* scale, const half* biases, int sSize, int cSize)
{
#ifdef HIP_SUPPORTS_FP16
  int cIdx = blockIdx.x * blockDim.x + threadIdx.x;
  int sIdx = blockIdx.y * blockDim.y + threadIdx.y;
  int nIdx = blockIdx.z;
  if(cIdx < cSize && sIdx < sSize) {
    int idx = (nIdx * sSize + sIdx) * cSize + cIdx;
    half a = __hfma(in[idx],scale[cIdx],biases[cIdx]);
    out[idx] = siluh(a);
  }
#else
  //Do nothing, FP16 not supported
#endif
}
__global__
void applyCScaleBiasNHWCMaskHalfKernel(const half* in, half* out, const half* scale, const half* biases, const half* mask, int sSize, int cSize)
{
#ifdef HIP_SUPPORTS_FP16
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
#ifdef HIP_SUPPORTS_FP16
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
#ifdef HIP_SUPPORTS_FP16
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
void applyCScaleBiasNHWCMishScale8MaskHalfKernel(const half* in, half* out, const half* scale, const half* biases, const half* mask, int sSize, int cSize)
{
#ifdef HIP_SUPPORTS_FP16
  int cIdx = blockIdx.x * blockDim.x + threadIdx.x;
  int sIdx = blockIdx.y * blockDim.y + threadIdx.y;
  int nIdx = blockIdx.z;
  if(cIdx < cSize && sIdx < sSize) {
    int idx = (nIdx * sSize + sIdx) * cSize + cIdx;
    half a = __hmul(__hfma(in[idx],scale[cIdx],biases[cIdx]),mask[nIdx*sSize+sIdx]);
    out[idx] = mishh_scale8(a);
  }
#else
  //Do nothing, FP16 not supported
#endif
}
__global__
void applyCScaleBiasNHWCSiluMaskHalfKernel(const half* in, half* out, const half* scale, const half* biases, const half* mask, int sSize, int cSize)
{
#ifdef HIP_SUPPORTS_FP16
  int cIdx = blockIdx.x * blockDim.x + threadIdx.x;
  int sIdx = blockIdx.y * blockDim.y + threadIdx.y;
  int nIdx = blockIdx.z;
  if(cIdx < cSize && sIdx < sSize) {
    int idx = (nIdx * sSize + sIdx) * cSize + cIdx;
    half a = __hmul(__hfma(in[idx],scale[cIdx],biases[cIdx]),mask[nIdx*sSize+sIdx]);
    out[idx] = siluh(a);
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
    else if(activation == ACTIVATION_MISH_SCALE8) {
      if(isHalf)
        applyCScaleBiasNHWCMishScale8HalfKernel<<<grid,threads>>>((const half*)in,(half*)out,(const half*)scale,(const half*)biases,sSize,cSize);
      else
        applyCScaleBiasNHWCMishScale8Kernel<<<grid,threads>>>((const float*)in,(float*)out,(const float*)scale,(const float*)biases,sSize,cSize);
    }
    else if(activation == ACTIVATION_SILU) {
      if(isHalf)
        applyCScaleBiasNHWCSiluHalfKernel<<<grid,threads>>>((const half*)in,(half*)out,(const half*)scale,(const half*)biases,sSize,cSize);
      else
        applyCScaleBiasNHWCSiluKernel<<<grid,threads>>>((const float*)in,(float*)out,(const float*)scale,(const float*)biases,sSize,cSize);
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
    else if(activation == ACTIVATION_MISH_SCALE8) {
      if(isHalf)
        applyCScaleBiasNHWCMishScale8MaskHalfKernel<<<grid,threads>>>((const half*)in,(half*)out,(const half*)scale,(const half*)biases,(const half*)mask,sSize,cSize);
      else
        applyCScaleBiasNHWCMishScale8MaskKernel<<<grid,threads>>>((const float*)in,(float*)out,(const float*)scale,(const float*)biases,(const float*)mask,sSize,cSize);
    }
    else if(activation == ACTIVATION_SILU) {
      if(isHalf)
        applyCScaleBiasNHWCSiluMaskHalfKernel<<<grid,threads>>>((const half*)in,(half*)out,(const half*)scale,(const half*)biases,(const half*)mask,sSize,cSize);
      else
        applyCScaleBiasNHWCSiluMaskKernel<<<grid,threads>>>((const float*)in,(float*)out,(const float*)scale,(const float*)biases,(const float*)mask,sSize,cSize);
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

//==============================================================================================
// Transformer support kernels
//==============================================================================================

//--------------------------------------------------------------------------------------------------------------
// RoPE: apply rotary position embeddings in-place.
// buf: [totalDim, seqLen*batchSize] column-major (totalDim = numBufHeads*qHeadDim, fast-moving).
// Each thread handles one (pair, xy, n, h) combination.

__global__
void applyRoPEKernel(
  float* buf, const float* cosTable, const float* sinTable,
  int batchSize, int seqLen, int numBufHeads, int numKVHeads, int qHeadDim, int totalDim, int numPairs, int learnableRope
) {
  int xy = blockIdx.x;
  int n = blockIdx.y;
  int hp = threadIdx.x;
  int totalHP = numBufHeads * numPairs;
  if(xy >= seqLen || n >= batchSize || hp >= totalHP)
    return;

  int h = hp / numPairs;
  int pairIdx = hp % numPairs;
  int c0 = h * qHeadDim + 2 * pairIdx;
  int c1 = c0 + 1;
  size_t col = (size_t)n * seqLen + xy;
  size_t idx0 = c0 + col * totalDim;
  size_t idx1 = c1 + col * totalDim;

  int tableIdx;
  if(learnableRope) {
    int kvh = h * numKVHeads / numBufHeads;
    tableIdx = (kvh * numPairs + pairIdx) * seqLen + xy;
  } else {
    tableIdx = pairIdx * seqLen + xy;
  }

  float cosVal = cosTable[tableIdx];
  float sinVal = sinTable[tableIdx];
  float x0 = buf[idx0];
  float x1 = buf[idx1];
  buf[idx0] = x0 * cosVal - x1 * sinVal;
  buf[idx1] = x0 * sinVal + x1 * cosVal;
}

__global__
void applyRoPEHalfKernel(
  half* buf, const half* cosTable, const half* sinTable,
  int batchSize, int seqLen, int numBufHeads, int numKVHeads, int qHeadDim, int totalDim, int numPairs, int learnableRope
) {
#ifdef HIP_SUPPORTS_FP16
  int xy = blockIdx.x;
  int n = blockIdx.y;
  int hp = threadIdx.x;
  int totalHP = numBufHeads * numPairs;
  if(xy >= seqLen || n >= batchSize || hp >= totalHP)
    return;

  int h = hp / numPairs;
  int pairIdx = hp % numPairs;
  int c0 = h * qHeadDim + 2 * pairIdx;
  int c1 = c0 + 1;
  size_t col = (size_t)n * seqLen + xy;
  size_t idx0 = c0 + col * totalDim;
  size_t idx1 = c1 + col * totalDim;

  int tableIdx;
  if(learnableRope) {
    int kvh = h * numKVHeads / numBufHeads;
    tableIdx = (kvh * numPairs + pairIdx) * seqLen + xy;
  } else {
    tableIdx = pairIdx * seqLen + xy;
  }

  float cosVal = __half2float(cosTable[tableIdx]);
  float sinVal = __half2float(sinTable[tableIdx]);
  float x0 = __half2float(buf[idx0]);
  float x1 = __half2float(buf[idx1]);
  buf[idx0] = __float2half(x0 * cosVal - x1 * sinVal);
  buf[idx1] = __float2half(x0 * sinVal + x1 * cosVal);
#else
  //Do nothing, FP16 not supported
#endif
}

// Learnable RoPE, table-free variant: recompute cos/sin in-kernel from the per-head frequencies
// instead of reading a precomputed [numKVHeads, numPairs, seqLen] cos/sin table.
// freqs layout: (numKVHeads, numPairs, 2) flattened; [...,0]=freqX (width/x), [...,1]=freqY (height/y).
__global__
void applyRoPELearnableRecomputeKernel(
  float* buf, const float* freqs,
  int batchSize, int seqLen, int numBufHeads, int numKVHeads, int qHeadDim, int totalDim, int numPairs, int nnXLen
) {
  int xy = blockIdx.x;
  int n = blockIdx.y;
  int hp = threadIdx.x;
  int totalHP = numBufHeads * numPairs;
  if(xy >= seqLen || n >= batchSize || hp >= totalHP)
    return;

  int h = hp / numPairs;
  int pairIdx = hp % numPairs;
  int c0 = h * qHeadDim + 2 * pairIdx;
  int c1 = c0 + 1;
  size_t col = (size_t)n * seqLen + xy;
  size_t idx0 = c0 + col * totalDim;
  size_t idx1 = c1 + col * totalDim;

  int kvh = h * numKVHeads / numBufHeads;
  int x = xy % nnXLen;
  int y = xy / nnXLen;
  float freqX = freqs[(kvh * numPairs + pairIdx) * 2 + 0];
  float freqY = freqs[(kvh * numPairs + pairIdx) * 2 + 1];
  float angle = (float)x * freqX + (float)y * freqY;
  float sinVal, cosVal;
  sincosf(angle, &sinVal, &cosVal);

  float x0 = buf[idx0];
  float x1 = buf[idx1];
  buf[idx0] = x0 * cosVal - x1 * sinVal;
  buf[idx1] = x0 * sinVal + x1 * cosVal;
}

__global__
void applyRoPELearnableRecomputeHalfKernel(
  half* buf, const float* freqs,
  int batchSize, int seqLen, int numBufHeads, int numKVHeads, int qHeadDim, int totalDim, int numPairs, int nnXLen
) {
#ifdef HIP_SUPPORTS_FP16
  int xy = blockIdx.x;
  int n = blockIdx.y;
  int hp = threadIdx.x;
  int totalHP = numBufHeads * numPairs;
  if(xy >= seqLen || n >= batchSize || hp >= totalHP)
    return;

  int h = hp / numPairs;
  int pairIdx = hp % numPairs;
  int c0 = h * qHeadDim + 2 * pairIdx;
  int c1 = c0 + 1;
  size_t col = (size_t)n * seqLen + xy;
  size_t idx0 = c0 + col * totalDim;
  size_t idx1 = c1 + col * totalDim;

  int kvh = h * numKVHeads / numBufHeads;
  int x = xy % nnXLen;
  int y = xy / nnXLen;
  float freqX = freqs[(kvh * numPairs + pairIdx) * 2 + 0];
  float freqY = freqs[(kvh * numPairs + pairIdx) * 2 + 1];
  float angle = (float)x * freqX + (float)y * freqY;
  float sinVal, cosVal;
  sincosf(angle, &sinVal, &cosVal);

  float x0 = __half2float(buf[idx0]);
  float x1 = __half2float(buf[idx1]);
  buf[idx0] = __float2half(x0 * cosVal - x1 * sinVal);
  buf[idx1] = __float2half(x0 * sinVal + x1 * cosVal);
#else
  //Do nothing, FP16 not supported
#endif
}

// One block per (xy, n). threadIdx.x = h*numPairs + pairIdx covers all channel pairs for the position.
void customCudaApplyRoPE(
  float* buf, const float* cosTable, const float* sinTable,
  int batchSize, int seqLen, int numBufHeads, int numKVHeads, int qHeadDim, int numPairs, bool learnableRope
) {
  int totalDim = numBufHeads * qHeadDim;
  int totalHP = numBufHeads * numPairs;
  if(totalHP > 1024)
    throw std::runtime_error("customCudaApplyRoPE: numHeads*qHeadDim/2 (" + std::to_string(totalHP) + ") exceeds the 1024 threads/block limit");
  int threads = ((totalHP + 31) / 32) * 32;
  dim3 blocks(seqLen, batchSize, 1);
  applyRoPEKernel<<<blocks, threads>>>(
    buf, cosTable, sinTable, batchSize, seqLen, numBufHeads, numKVHeads, qHeadDim, totalDim, numPairs, learnableRope ? 1 : 0
  );
}
void customCudaApplyRoPE(
  half* buf, const half* cosTable, const half* sinTable,
  int batchSize, int seqLen, int numBufHeads, int numKVHeads, int qHeadDim, int numPairs, bool learnableRope
) {
  int totalDim = numBufHeads * qHeadDim;
  int totalHP = numBufHeads * numPairs;
  if(totalHP > 1024)
    throw std::runtime_error("customCudaApplyRoPE: numHeads*qHeadDim/2 (" + std::to_string(totalHP) + ") exceeds the 1024 threads/block limit");
  int threads = ((totalHP + 31) / 32) * 32;
  dim3 blocks(seqLen, batchSize, 1);
  applyRoPEHalfKernel<<<blocks, threads>>>(
    buf, cosTable, sinTable, batchSize, seqLen, numBufHeads, numKVHeads, qHeadDim, totalDim, numPairs, learnableRope ? 1 : 0
  );
}

void customCudaApplyRoPELearnableRecompute(
  float* buf, const float* freqs,
  int batchSize, int seqLen, int numBufHeads, int numKVHeads, int qHeadDim, int numPairs, int nnXLen
) {
  int totalDim = numBufHeads * qHeadDim;
  int totalHP = numBufHeads * numPairs;
  if(totalHP > 1024)
    throw std::runtime_error("customCudaApplyRoPELearnableRecompute: numHeads*qHeadDim/2 (" + std::to_string(totalHP) + ") exceeds the 1024 threads/block limit");
  int threads = ((totalHP + 31) / 32) * 32;
  dim3 blocks(seqLen, batchSize, 1);
  applyRoPELearnableRecomputeKernel<<<blocks, threads>>>(
    buf, freqs, batchSize, seqLen, numBufHeads, numKVHeads, qHeadDim, totalDim, numPairs, nnXLen
  );
}
void customCudaApplyRoPELearnableRecompute(
  half* buf, const float* freqs,
  int batchSize, int seqLen, int numBufHeads, int numKVHeads, int qHeadDim, int numPairs, int nnXLen
) {
  int totalDim = numBufHeads * qHeadDim;
  int totalHP = numBufHeads * numPairs;
  if(totalHP > 1024)
    throw std::runtime_error("customCudaApplyRoPELearnableRecompute: numHeads*qHeadDim/2 (" + std::to_string(totalHP) + ") exceeds the 1024 threads/block limit");
  int threads = ((totalHP + 31) / 32) * 32;
  dim3 blocks(seqLen, batchSize, 1);
  applyRoPELearnableRecomputeHalfKernel<<<blocks, threads>>>(
    buf, freqs, batchSize, seqLen, numBufHeads, numKVHeads, qHeadDim, totalDim, numPairs, nnXLen
  );
}

//--------------------------------------------------------------------------------------------------------------
// FlashAttention-style scaled dot product attention with online softmax (tiled).
// Grid: (numQGroups, batchSize * numHeads), block: BLOCK_Q threads.
// Layout: BSHD row-major. Templated on qHeadDim/vHeadDim so inner loops unroll.

template<int qHeadDim, int vHeadDim, int BLOCK_Q, int BLOCK_KV, int Q_PER_THREAD, typename T>
__device__ __forceinline__
void flashAttentionTiledImpl(
  const T* Q, const T* K, const T* V, const T* mask, T* output,
  int seqLen, int numHeads, int numKVHeads, float scale
) {
  const int tid = threadIdx.x;
  const int qBlockStart = blockIdx.x * (BLOCK_Q * Q_PER_THREAD);
  const int bh = blockIdx.y;
  const int n = bh / numHeads;
  const int h = bh % numHeads;
  const int kvh = h * numKVHeads / numHeads;

  const int qTotalDim = numHeads * qHeadDim;
  const int kTotalDim = numKVHeads * qHeadDim;
  const int vTotalDim = numKVHeads * vHeadDim;
  const int oTotalDim = numHeads * vHeadDim;

  constexpr int K_TILE_STRIDE = qHeadDim;
  constexpr int V_TILE_STRIDE = vHeadDim;
  __shared__ float kTile[BLOCK_KV * K_TILE_STRIDE];
  __shared__ float vTile[BLOCK_KV * V_TILE_STRIDE];
  __shared__ float kMaskTile[BLOCK_KV];

  float qReg[Q_PER_THREAD * qHeadDim];
  float qMask[Q_PER_THREAD];
  float runningMax[Q_PER_THREAD];
  float runningSum[Q_PER_THREAD];
  float acc[Q_PER_THREAD * vHeadDim];

  // Load Q for the Q_PER_THREAD positions this thread owns.
  #pragma unroll
  for(int qi = 0; qi < Q_PER_THREAD; qi++) {
    int qPos = qBlockStart + qi * BLOCK_Q + tid;
    qMask[qi] = 0.0f;
    if(qPos < seqLen) {
      if(mask != NULL) {
        qMask[qi] = (float)mask[n * seqLen + qPos];
      } else {
        qMask[qi] = 1.0f;
      }
      if(qMask[qi] != 0.0f) {
        const T* qPtr = Q + ((size_t)n * seqLen + qPos) * qTotalDim + h * qHeadDim;
        #pragma unroll
        for(int d = 0; d < qHeadDim; d++) qReg[qi * qHeadDim + d] = (float)qPtr[d];
      }
    }
    runningMax[qi] = -1e30f;
    runningSum[qi] = 0.0f;
    #pragma unroll
    for(int d = 0; d < vHeadDim; d++) acc[qi * vHeadDim + d] = 0.0f;
  }

  // Iterate over K/V in BLOCK_KV-row tiles.
  for(int kvStart = 0; kvStart < seqLen; kvStart += BLOCK_KV) {
    // Cooperatively load K tile: BLOCK_KV rows of qHeadDim values (stride K_TILE_STRIDE).
    #pragma unroll
    for(int t = tid; t < BLOCK_KV * qHeadDim; t += BLOCK_Q) {
      int tileKPos = t / qHeadDim;
      int tileD = t % qHeadDim;
      int globalKPos = kvStart + tileKPos;
      float v = 0.0f;
      if(globalKPos < seqLen) {
        const T* kPtr = K + ((size_t)n * seqLen + globalKPos) * kTotalDim + kvh * qHeadDim;
        v = (float)kPtr[tileD];
      }
      kTile[tileKPos * K_TILE_STRIDE + tileD] = v;
    }
    // Cooperatively load V tile: BLOCK_KV rows of vHeadDim values (stride V_TILE_STRIDE).
    #pragma unroll
    for(int t = tid; t < BLOCK_KV * vHeadDim; t += BLOCK_Q) {
      int tileKPos = t / vHeadDim;
      int tileD = t % vHeadDim;
      int globalKPos = kvStart + tileKPos;
      float v = 0.0f;
      if(globalKPos < seqLen) {
        const T* vPtr = V + ((size_t)n * seqLen + globalKPos) * vTotalDim + kvh * vHeadDim;
        v = (float)vPtr[tileD];
      }
      vTile[tileKPos * V_TILE_STRIDE + tileD] = v;
    }
    // Cooperatively load mask tile.
    for(int t = tid; t < BLOCK_KV; t += BLOCK_Q) {
      int globalKPos = kvStart + t;
      float m = 0.0f;
      if(globalKPos < seqLen) {
        m = (mask != NULL) ? (float)mask[n * seqLen + globalKPos] : 1.0f;
      }
      kMaskTile[t] = m;
    }
    __syncthreads();

    int kvEnd = min(BLOCK_KV, seqLen - kvStart);

    // Each thread updates its Q_PER_THREAD queries against the shared K/V tile.
    #pragma unroll
    for(int qi = 0; qi < Q_PER_THREAD; qi++) {
      int qPos = qBlockStart + qi * BLOCK_Q + tid;
      if(qPos >= seqLen || qMask[qi] == 0.0f) continue;

      for(int tk = 0; tk < kvEnd; tk++) {
        if(kMaskTile[tk] == 0.0f) continue;

        float dot = 0.0f;
        #pragma unroll
        for(int d = 0; d < qHeadDim; d++) {
          dot += qReg[qi * qHeadDim + d] * kTile[tk * K_TILE_STRIDE + d];
        }
        dot *= scale;

        float newMax = fmaxf(runningMax[qi], dot);
        float expOldMax = expf(runningMax[qi] - newMax);
        float expCur = expf(dot - newMax);

        #pragma unroll
        for(int d = 0; d < vHeadDim; d++) {
          acc[qi * vHeadDim + d] = acc[qi * vHeadDim + d] * expOldMax + expCur * vTile[tk * V_TILE_STRIDE + d];
        }
        runningSum[qi] = runningSum[qi] * expOldMax + expCur;
        runningMax[qi] = newMax;
      }
    }
    __syncthreads();
  }

  // Write outputs.
  #pragma unroll
  for(int qi = 0; qi < Q_PER_THREAD; qi++) {
    int qPos = qBlockStart + qi * BLOCK_Q + tid;
    if(qPos >= seqLen) continue;
    T* outRow = output + ((size_t)n * seqLen + qPos) * oTotalDim + h * vHeadDim;
    if(qMask[qi] == 0.0f) {
      #pragma unroll
      for(int d = 0; d < vHeadDim; d++) outRow[d] = (T)0.0f;
    } else {
      float invSum = (runningSum[qi] > 0.0f) ? (1.0f / runningSum[qi]) : 0.0f;
      #pragma unroll
      for(int d = 0; d < vHeadDim; d++) outRow[d] = (T)(acc[qi * vHeadDim + d] * invSum);
    }
  }
}

template<int qHeadDim, int vHeadDim, int BLOCK_Q, int BLOCK_KV, int Q_PER_THREAD>
__global__
void flashAttentionKernelFloat(
  const float* Q, const float* K, const float* V, const float* mask, float* output,
  int seqLen, int numHeads, int numKVHeads, float scale
) {
  flashAttentionTiledImpl<qHeadDim, vHeadDim, BLOCK_Q, BLOCK_KV, Q_PER_THREAD, float>(
    Q, K, V, mask, output, seqLen, numHeads, numKVHeads, scale);
}

template<int qHeadDim, int vHeadDim, int BLOCK_Q, int BLOCK_KV, int Q_PER_THREAD>
__global__
void flashAttentionKernelHalf(
  const half* Q, const half* K, const half* V, const half* mask, half* output,
  int seqLen, int numHeads, int numKVHeads, float scale
) {
#ifdef HIP_SUPPORTS_FP16
  flashAttentionTiledImpl<qHeadDim, vHeadDim, BLOCK_Q, BLOCK_KV, Q_PER_THREAD, half>(
    Q, K, V, mask, output, seqLen, numHeads, numKVHeads, scale);
#endif
}

#define FA_LAUNCH_FLOAT(QD, VD, BQ, BKV, QPT)                                   \
  do {                                                                          \
    int totalQPerBlock = (BQ) * (QPT);                                          \
    dim3 grid((seqLen + totalQPerBlock - 1) / totalQPerBlock, batchSize * numHeads); \
    flashAttentionKernelFloat<(QD), (VD), (BQ), (BKV), (QPT)><<<grid, (BQ)>>>( \
      Q, K, V, mask, output, seqLen, numHeads, numKVHeads, scale);              \
  } while(0)

#define FA_LAUNCH_HALF(QD, VD, BQ, BKV, QPT)                                    \
  do {                                                                          \
    int totalQPerBlock = (BQ) * (QPT);                                          \
    dim3 grid((seqLen + totalQPerBlock - 1) / totalQPerBlock, batchSize * numHeads); \
    flashAttentionKernelHalf<(QD), (VD), (BQ), (BKV), (QPT)><<<grid, (BQ)>>>(  \
      Q, K, V, mask, output, seqLen, numHeads, numKVHeads, scale);              \
  } while(0)

// Only (qHeadDim,vHeadDim) pairs known to be used by KataGo's transformer models so far are
// instantiated; add more shapes as needed (mirrors the CUDA backend's supported-shape list).
void customCudaFlashAttention(
  const float* Q, const float* K, const float* V, const float* mask, float* output,
  int batchSize, int seqLen, int numHeads, int numKVHeads, int qHeadDim, int vHeadDim
) {
  if(batchSize * numHeads > 65536)
    throw std::runtime_error("customCudaFlashAttention: batchSize * numHeads too large");
  float scale = 1.0f / sqrtf((float)qHeadDim);
  if(qHeadDim == 32 && vHeadDim == 32) FA_LAUNCH_FLOAT(32, 32, 128, 32, 1);
  else if(qHeadDim == 32 && vHeadDim == 16) FA_LAUNCH_FLOAT(32, 16, 128, 32, 1);
  else if(qHeadDim == 64 && vHeadDim == 64) FA_LAUNCH_FLOAT(64, 64, 128, 32, 1);
  else if(qHeadDim == 64 && vHeadDim == 32) FA_LAUNCH_FLOAT(64, 32, 128, 32, 1);
  else if(qHeadDim == 32 && vHeadDim == 64) FA_LAUNCH_FLOAT(32, 64, 128, 32, 1);
  else throw std::runtime_error("customCudaFlashAttention: unsupported (qHeadDim,vHeadDim) combination");
}
void customCudaFlashAttention(
  const half* Q, const half* K, const half* V, const half* mask, half* output,
  int batchSize, int seqLen, int numHeads, int numKVHeads, int qHeadDim, int vHeadDim
) {
  if(batchSize * numHeads > 65536)
    throw std::runtime_error("customCudaFlashAttention: batchSize * numHeads too large");
  float scale = 1.0f / sqrtf((float)qHeadDim);
  if(qHeadDim == 32 && vHeadDim == 32) FA_LAUNCH_HALF(32, 32, 128, 32, 1);
  else if(qHeadDim == 32 && vHeadDim == 16) FA_LAUNCH_HALF(32, 16, 128, 32, 1);
  else if(qHeadDim == 64 && vHeadDim == 64) FA_LAUNCH_HALF(64, 64, 128, 32, 1);
  else if(qHeadDim == 64 && vHeadDim == 32) FA_LAUNCH_HALF(64, 32, 128, 32, 1);
  else if(qHeadDim == 32 && vHeadDim == 64) FA_LAUNCH_HALF(32, 64, 128, 32, 1);
  else throw std::runtime_error("customCudaFlashAttention: unsupported (qHeadDim,vHeadDim) combination");
}

#undef FA_LAUNCH_FLOAT
#undef FA_LAUNCH_HALF

//--------------------------------------------------------------------------------------------------------------
// Convert mask [batchSize, seqLen] (0/1) into a fully-materialized additive attention bias of shape
// [batchSize, seqLen, seqLen]: bias[b,q,k] = (mask[b,k] != 0 ? 0 : -1e4).

__global__
void maskToAttnBiasFullKernel(const float* mask, float* outBias, int seqLen) {
  int k = blockIdx.x * blockDim.x + threadIdx.x;
  int q = blockIdx.y;
  int b = blockIdx.z;
  if(k >= seqLen)
    return;
  float m = mask[b * seqLen + k];
  outBias[((size_t)b * seqLen + q) * seqLen + k] = (m != 0.0f) ? 0.0f : -1e4f;
}

__global__
void maskToAttnBiasFullHalfKernel(const half* mask, half* outBias, int seqLen) {
#ifdef HIP_SUPPORTS_FP16
  int k = blockIdx.x * blockDim.x + threadIdx.x;
  int q = blockIdx.y;
  int b = blockIdx.z;
  if(k >= seqLen)
    return;
  float m = __half2float(mask[b * seqLen + k]);
  outBias[((size_t)b * seqLen + q) * seqLen + k] = __float2half((m != 0.0f) ? 0.0f : -1e4f);
#endif
}

void customCudaMaskToAttnBiasFull(const float* mask, float* outBias, int batchSize, int seqLen) {
  if(batchSize <= 0 || seqLen <= 0)
    return;
  int threads = 128;
  dim3 blocks((seqLen + threads - 1) / threads, seqLen, batchSize);
  maskToAttnBiasFullKernel<<<blocks, threads>>>(mask, outBias, seqLen);
}
void customCudaMaskToAttnBiasFull(const half* mask, half* outBias, int batchSize, int seqLen) {
  if(batchSize <= 0 || seqLen <= 0)
    return;
  int threads = 128;
  dim3 blocks((seqLen + threads - 1) / threads, seqLen, batchSize);
  maskToAttnBiasFullHalfKernel<<<blocks, threads>>>(mask, outBias, seqLen);
}

//--------------------------------------------------------------------------------------------------------------
// SwiGLU: out[i] = SiLU(a[i]) * b[i]

__global__
void swiGLUKernel(const float* a, const float* b, float* out, int size)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if(idx < size) {
    out[idx] = siluf(a[idx]) * b[idx];
  }
}

__global__
void swiGLUHalfKernel(const half* a, const half* b, half* out, int size)
{
#ifdef HIP_SUPPORTS_FP16
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if(idx < size) {
    float av = __half2float(a[idx]);
    float bv = __half2float(b[idx]);
    out[idx] = __float2half(siluf(av) * bv);
  }
#else
  //Do nothing, FP16 not supported
#endif
}

void customCudaSwiGLU(const float* a, const float* b, float* out, int size) {
  if(size <= 0)
    return;
  int threads = targetNumThreads;
  int blocks = (size + threads - 1) / threads;
  swiGLUKernel<<<blocks, threads>>>(a, b, out, size);
}
void customCudaSwiGLU(const half* a, const half* b, half* out, int size) {
  if(size <= 0)
    return;
  int threads = targetNumThreads;
  int blocks = (size + threads - 1) / threads;
  swiGLUHalfKernel<<<blocks, threads>>>(a, b, out, size);
}

//--------------------------------------------------------------------------------------------------------------
// Masked residual add: trunk[i] += residual[i] * mask[spatial_idx]
// NCHW: trunk/residual [n, c, xy], mask [n, xy]
// NHWC: trunk/residual [n, xy, c], mask [n, xy]

__global__
void maskedResidualAddNCHWKernel(float* trunk, const float* residual, const float* mask, int cSize, int xySize)
{
  int xyIdx = blockIdx.x * blockDim.x + threadIdx.x;
  int cIdx = blockIdx.y * blockDim.y + threadIdx.y;
  int nIdx = blockIdx.z;
  if(xyIdx >= xySize || cIdx >= cSize)
    return;
  int idx = (nIdx * cSize + cIdx) * xySize + xyIdx;
  float m = (mask != NULL) ? mask[nIdx * xySize + xyIdx] : 1.0f;
  trunk[idx] += residual[idx] * m;
}

__global__
void maskedResidualAddNCHWHalfKernel(half* trunk, const half* residual, const half* mask, int cSize, int xySize)
{
#ifdef HIP_SUPPORTS_FP16
  int xyIdx = blockIdx.x * blockDim.x + threadIdx.x;
  int cIdx = blockIdx.y * blockDim.y + threadIdx.y;
  int nIdx = blockIdx.z;
  if(xyIdx >= xySize || cIdx >= cSize)
    return;
  int idx = (nIdx * cSize + cIdx) * xySize + xyIdx;
  float m = (mask != NULL) ? __half2float(mask[nIdx * xySize + xyIdx]) : 1.0f;
  trunk[idx] = __float2half(__half2float(trunk[idx]) + __half2float(residual[idx]) * m);
#else
  //Do nothing, FP16 not supported
#endif
}

void customCudaMaskedResidualAddNCHW(float* trunk, const float* residual, const float* mask, int nSize, int cSize, int xySize) {
  if(nSize > 65536)
    throw std::runtime_error("customCudaMaskedResidualAddNCHW: nSize too large");
  checkBufferIndexFitsInt(nSize, cSize, xySize, "customCudaMaskedResidualAddNCHW");
  int xyThreads, xyBlocks, cThreads, cBlocks;
  splitThreadsAcrossDim01(xySize, cSize, xyThreads, xyBlocks, cThreads, cBlocks);
  dim3 grid(xyBlocks, cBlocks, nSize);
  dim3 threads(xyThreads, cThreads, 1);
  maskedResidualAddNCHWKernel<<<grid, threads>>>(trunk, residual, mask, cSize, xySize);
}
void customCudaMaskedResidualAddNCHW(half* trunk, const half* residual, const half* mask, int nSize, int cSize, int xySize) {
  if(nSize > 65536)
    throw std::runtime_error("customCudaMaskedResidualAddNCHW: nSize too large");
  checkBufferIndexFitsInt(nSize, cSize, xySize, "customCudaMaskedResidualAddNCHW");
  int xyThreads, xyBlocks, cThreads, cBlocks;
  splitThreadsAcrossDim01(xySize, cSize, xyThreads, xyBlocks, cThreads, cBlocks);
  dim3 grid(xyBlocks, cBlocks, nSize);
  dim3 threads(xyThreads, cThreads, 1);
  maskedResidualAddNCHWHalfKernel<<<grid, threads>>>(trunk, residual, mask, cSize, xySize);
}

__global__
void maskedResidualAddNHWCKernel(float* trunk, const float* residual, const float* mask, int xySize, int cSize)
{
  int cIdx = blockIdx.x * blockDim.x + threadIdx.x;
  int xyIdx = blockIdx.y * blockDim.y + threadIdx.y;
  int nIdx = blockIdx.z;
  if(cIdx >= cSize || xyIdx >= xySize)
    return;
  int idx = (nIdx * xySize + xyIdx) * cSize + cIdx;
  float m = (mask != NULL) ? mask[nIdx * xySize + xyIdx] : 1.0f;
  trunk[idx] += residual[idx] * m;
}

__global__
void maskedResidualAddNHWCHalfKernel(half* trunk, const half* residual, const half* mask, int xySize, int cSize)
{
#ifdef HIP_SUPPORTS_FP16
  int cIdx = blockIdx.x * blockDim.x + threadIdx.x;
  int xyIdx = blockIdx.y * blockDim.y + threadIdx.y;
  int nIdx = blockIdx.z;
  if(cIdx >= cSize || xyIdx >= xySize)
    return;
  int idx = (nIdx * xySize + xyIdx) * cSize + cIdx;
  float m = (mask != NULL) ? __half2float(mask[nIdx * xySize + xyIdx]) : 1.0f;
  trunk[idx] = __float2half(__half2float(trunk[idx]) + __half2float(residual[idx]) * m);
#else
  //Do nothing, FP16 not supported
#endif
}

void customCudaMaskedResidualAddNHWC(float* trunk, const float* residual, const float* mask, int nSize, int xySize, int cSize) {
  if(nSize > 65536)
    throw std::runtime_error("customCudaMaskedResidualAddNHWC: nSize too large");
  checkBufferIndexFitsInt(nSize, xySize, cSize, "customCudaMaskedResidualAddNHWC");
  int cThreads, cBlocks, xyThreads, xyBlocks;
  splitThreadsAcrossDim01(cSize, xySize, cThreads, cBlocks, xyThreads, xyBlocks);
  dim3 grid(cBlocks, xyBlocks, nSize);
  dim3 threads(cThreads, xyThreads, 1);
  maskedResidualAddNHWCKernel<<<grid, threads>>>(trunk, residual, mask, xySize, cSize);
}
void customCudaMaskedResidualAddNHWC(half* trunk, const half* residual, const half* mask, int nSize, int xySize, int cSize) {
  if(nSize > 65536)
    throw std::runtime_error("customCudaMaskedResidualAddNHWC: nSize too large");
  checkBufferIndexFitsInt(nSize, xySize, cSize, "customCudaMaskedResidualAddNHWC");
  int cThreads, cBlocks, xyThreads, xyBlocks;
  splitThreadsAcrossDim01(cSize, xySize, cThreads, cBlocks, xyThreads, xyBlocks);
  dim3 grid(cBlocks, xyBlocks, nSize);
  dim3 threads(cThreads, xyThreads, 1);
  maskedResidualAddNHWCHalfKernel<<<grid, threads>>>(trunk, residual, mask, xySize, cSize);
}

//--------------------------------------------------------------------------------------------------------------
// RMSNorm with gamma/beta/activation (for trunk tip and transformer pre-norm, non-spatial mode).
// NHWC: input/output [n, xy, c], gamma/beta [c], mask [n, xy]
// Each block handles one (n, xy) position. Scalar (non-vectorized) kernels only, for portability
// across AMD wavefront sizes (no warp-shuffle assumptions).

__global__
void rmsNormGammaBetaNHWCKernel(
  const float* in, float* out, const float* gamma, const float* beta, const float* mask,
  int nSize, int xySize, int cSize, float epsilon, int activation
) {
  extern __shared__ float rmsShared[];
  int pos = blockIdx.x; // n * xySize + xy
  int tid = threadIdx.x;
  int n = pos / xySize;
  int xy = pos % xySize;
  if(n >= nSize)
    return;

  float maskVal = (mask != NULL) ? mask[n * xySize + xy] : 1.0f;

  const float* inRow = in + (size_t)pos * cSize;

  float acc = 0.0f;
  for(int c = tid; c < cSize; c += blockDim.x) {
    float val = inRow[c] * maskVal;
    acc += val * val;
  }
  rmsShared[tid] = acc;
  __syncthreads();
  for(int s = blockDim.x / 2; s > 0; s >>= 1) {
    if(tid < s) rmsShared[tid] += rmsShared[tid + s];
    __syncthreads();
  }
  float rms = rsqrtf(rmsShared[0] / (float)cSize + epsilon);

  float* outRow = out + (size_t)pos * cSize;
  for(int c = tid; c < cSize; c += blockDim.x) {
    float val = inRow[c] * maskVal * rms * gamma[c] + beta[c];
    if(activation == ACTIVATION_RELU) val = fmaxf(val, 0.0f);
    else if(activation == ACTIVATION_MISH) val = mishf(val);
    else if(activation == ACTIVATION_SILU) val = siluf(val);
    val *= maskVal;
    outRow[c] = val;
  }
}

__global__
void rmsNormGammaBetaNHWCHalfKernel(
  const half* in, half* out, const half* gamma, const half* beta, const half* mask,
  int nSize, int xySize, int cSize, float epsilon, int activation
) {
#ifdef HIP_SUPPORTS_FP16
  extern __shared__ float rmsShared[];
  int pos = blockIdx.x;
  int tid = threadIdx.x;
  int n = pos / xySize;
  int xy = pos % xySize;
  if(n >= nSize)
    return;

  float maskVal = (mask != NULL) ? __half2float(mask[n * xySize + xy]) : 1.0f;

  const half* inRow = in + (size_t)pos * cSize;

  float acc = 0.0f;
  for(int c = tid; c < cSize; c += blockDim.x) {
    float val = __half2float(inRow[c]) * maskVal;
    acc += val * val;
  }
  rmsShared[tid] = acc;
  __syncthreads();
  for(int s = blockDim.x / 2; s > 0; s >>= 1) {
    if(tid < s) rmsShared[tid] += rmsShared[tid + s];
    __syncthreads();
  }
  float rms = rsqrtf(rmsShared[0] / (float)cSize + epsilon);

  half* outRow = out + (size_t)pos * cSize;
  for(int c = tid; c < cSize; c += blockDim.x) {
    float val = __half2float(inRow[c]) * maskVal * rms * __half2float(gamma[c]) + __half2float(beta[c]);
    if(activation == ACTIVATION_RELU) val = fmaxf(val, 0.0f);
    else if(activation == ACTIVATION_MISH) val = mishf(val);
    else if(activation == ACTIVATION_SILU) val = siluf(val);
    val *= maskVal;
    outRow[c] = __float2half(val);
  }
#else
  //Do nothing, FP16 not supported
#endif
}

void customCudaRMSNormGammaBetaNHWC(
  const float* in, float* out, const float* gamma, const float* beta, const float* mask,
  int nSize, int xySize, int cSize, float epsilon, int activation
) {
  checkBufferIndexFitsInt(nSize, xySize, cSize, "customCudaRMSNormGammaBetaNHWC");
  int totalPositions = nSize * xySize;
  if(totalPositions <= 0)
    return;
  int threads = 1;
  while(threads < cSize && threads < targetNumThreads) threads *= 2;
  int sharedMem = threads * sizeof(float);
  rmsNormGammaBetaNHWCKernel<<<totalPositions, threads, sharedMem>>>(
    in, out, gamma, beta, mask, nSize, xySize, cSize, epsilon, activation);
}
void customCudaRMSNormGammaBetaNHWC(
  const half* in, half* out, const half* gamma, const half* beta, const half* mask,
  int nSize, int xySize, int cSize, float epsilon, int activation
) {
  checkBufferIndexFitsInt(nSize, xySize, cSize, "customCudaRMSNormGammaBetaNHWC");
  int totalPositions = nSize * xySize;
  if(totalPositions <= 0)
    return;
  int threads = 1;
  while(threads < cSize && threads < targetNumThreads) threads *= 2;
  int sharedMem = threads * sizeof(float);
  rmsNormGammaBetaNHWCHalfKernel<<<totalPositions, threads, sharedMem>>>(
    in, out, gamma, beta, mask, nSize, xySize, cSize, epsilon, activation);
}

// NCHW variant: input/output [n, c, xy], gamma/beta [c], mask [n, xy]
__global__
void rmsNormGammaBetaNCHWKernel(
  const float* in, float* out, const float* gamma, const float* beta, const float* mask,
  int nSize, int cSize, int xySize, float epsilon, int activation
) {
  extern __shared__ float rmsShared[];
  int pos = blockIdx.x; // n * xySize + xy
  int tid = threadIdx.x;
  int n = pos / xySize;
  int xy = pos % xySize;
  if(n >= nSize)
    return;

  float maskVal = (mask != NULL) ? mask[n * xySize + xy] : 1.0f;

  float acc = 0.0f;
  for(int c = tid; c < cSize; c += blockDim.x) {
    float val = in[(n * cSize + c) * xySize + xy] * maskVal;
    acc += val * val;
  }
  rmsShared[tid] = acc;
  __syncthreads();
  for(int s = blockDim.x / 2; s > 0; s >>= 1) {
    if(tid < s) rmsShared[tid] += rmsShared[tid + s];
    __syncthreads();
  }
  float rms = rsqrtf(rmsShared[0] / (float)cSize + epsilon);

  for(int c = tid; c < cSize; c += blockDim.x) {
    float val = in[(n * cSize + c) * xySize + xy] * maskVal * rms * gamma[c] + beta[c];
    if(activation == ACTIVATION_RELU) val = fmaxf(val, 0.0f);
    else if(activation == ACTIVATION_MISH) val = mishf(val);
    else if(activation == ACTIVATION_SILU) val = siluf(val);
    val *= maskVal;
    out[(n * cSize + c) * xySize + xy] = val;
  }
}

__global__
void rmsNormGammaBetaNCHWHalfKernel(
  const half* in, half* out, const half* gamma, const half* beta, const half* mask,
  int nSize, int cSize, int xySize, float epsilon, int activation
) {
#ifdef HIP_SUPPORTS_FP16
  extern __shared__ float rmsShared[];
  int pos = blockIdx.x;
  int tid = threadIdx.x;
  int n = pos / xySize;
  int xy = pos % xySize;
  if(n >= nSize)
    return;

  float maskVal = (mask != NULL) ? __half2float(mask[n * xySize + xy]) : 1.0f;

  float acc = 0.0f;
  for(int c = tid; c < cSize; c += blockDim.x) {
    float val = __half2float(in[(n * cSize + c) * xySize + xy]) * maskVal;
    acc += val * val;
  }
  rmsShared[tid] = acc;
  __syncthreads();
  for(int s = blockDim.x / 2; s > 0; s >>= 1) {
    if(tid < s) rmsShared[tid] += rmsShared[tid + s];
    __syncthreads();
  }
  float rms = rsqrtf(rmsShared[0] / (float)cSize + epsilon);

  for(int c = tid; c < cSize; c += blockDim.x) {
    float val = __half2float(in[(n * cSize + c) * xySize + xy]) * maskVal * rms * __half2float(gamma[c]) + __half2float(beta[c]);
    if(activation == ACTIVATION_RELU) val = fmaxf(val, 0.0f);
    else if(activation == ACTIVATION_MISH) val = mishf(val);
    else if(activation == ACTIVATION_SILU) val = siluf(val);
    val *= maskVal;
    out[(n * cSize + c) * xySize + xy] = __float2half(val);
  }
#else
  //Do nothing, FP16 not supported
#endif
}

void customCudaRMSNormGammaBetaNCHW(
  const float* in, float* out, const float* gamma, const float* beta, const float* mask,
  int nSize, int cSize, int xySize, float epsilon, int activation
) {
  checkBufferIndexFitsInt(nSize, cSize, xySize, "customCudaRMSNormGammaBetaNCHW");
  int totalPositions = nSize * xySize;
  if(totalPositions <= 0)
    return;
  int threads = 1;
  while(threads < cSize && threads < targetNumThreads) threads *= 2;
  int sharedMem = threads * sizeof(float);
  rmsNormGammaBetaNCHWKernel<<<totalPositions, threads, sharedMem>>>(
    in, out, gamma, beta, mask, nSize, cSize, xySize, epsilon, activation);
}
void customCudaRMSNormGammaBetaNCHW(
  const half* in, half* out, const half* gamma, const half* beta, const half* mask,
  int nSize, int cSize, int xySize, float epsilon, int activation
) {
  checkBufferIndexFitsInt(nSize, cSize, xySize, "customCudaRMSNormGammaBetaNCHW");
  int totalPositions = nSize * xySize;
  if(totalPositions <= 0)
    return;
  int threads = 1;
  while(threads < cSize && threads < targetNumThreads) threads *= 2;
  int sharedMem = threads * sizeof(float);
  rmsNormGammaBetaNCHWHalfKernel<<<totalPositions, threads, sharedMem>>>(
    in, out, gamma, beta, mask, nSize, cSize, xySize, epsilon, activation);
}

//--------------------------------------------------------------------------------------------------------------
// Spatial RMSNorm: normalize over all C*H*W per batch element.
// Three-pass, deterministic:
//   Pass 1 (SumSq):  grid (numBlocksPerBatch, nSize). Many blocks per batch element grid-stride over
//                    the flat C*xy range, reduce in-block, write one partial per block into partialBuf.
//   Pass 2 (Reduce): grid (nSize). One block per batch element sums its numBlocksPerBatch partials
//                    (fixed order) into sumSqBuf[n].
//   Pass 3 (Apply):  grid (numApplyBlocks, nSize). Normalize + activation + remask.

static const int SPATIAL_RMSNORM_BLOCKS_PER_BATCH = 8;
static_assert(CUDA_SPATIAL_RMSNORM_SUMSQ_STRIDE == SPATIAL_RMSNORM_BLOCKS_PER_BATCH + 1,
  "CUDA_SPATIAL_RMSNORM_SUMSQ_STRIDE must equal SPATIAL_RMSNORM_BLOCKS_PER_BATCH + 1");

static int spatialRMSNormBlocksPerBatch(int totalElems) {
  int maxUseful = (totalElems + targetNumThreads - 1) / targetNumThreads;
  if(maxUseful < 1) maxUseful = 1;
  int b = SPATIAL_RMSNORM_BLOCKS_PER_BATCH;
  if(b > maxUseful) b = maxUseful;
  return b;
}

template<bool IS_NHWC>
__global__
void spatialRMSNormSumSqKernel(
  const float* in, const float* mask, float* partialBuf,
  int totalElems, int cSize, int xySize, int numBlocksPerBatch, int partialStride
) {
  extern __shared__ float srmsShared[];
  int n = blockIdx.y;
  int blk = blockIdx.x;
  int tid = threadIdx.x;

  const float* inRow = in + (size_t)n * totalElems;

  float acc = 0.0f;
  for(int i = blk * blockDim.x + tid; i < totalElems; i += blockDim.x * numBlocksPerBatch) {
    int xy = IS_NHWC ? (i / cSize) : (i % xySize);
    float m = (mask != NULL) ? mask[n * xySize + xy] : 1.0f;
    float val = inRow[i] * m;
    acc += val * val;
  }
  srmsShared[tid] = acc;
  __syncthreads();
  for(int s = blockDim.x / 2; s > 0; s >>= 1) {
    if(tid < s) srmsShared[tid] += srmsShared[tid + s];
    __syncthreads();
  }
  if(tid == 0) partialBuf[n * partialStride + blk] = srmsShared[0];
}

template<bool IS_NHWC>
__global__
void spatialRMSNormSumSqHalfKernel(
  const half* in, const half* mask, float* partialBuf,
  int totalElems, int cSize, int xySize, int numBlocksPerBatch, int partialStride
) {
#ifdef HIP_SUPPORTS_FP16
  extern __shared__ float srmsShared[];
  int n = blockIdx.y;
  int blk = blockIdx.x;
  int tid = threadIdx.x;

  const half* inRow = in + (size_t)n * totalElems;

  float acc = 0.0f;
  for(int i = blk * blockDim.x + tid; i < totalElems; i += blockDim.x * numBlocksPerBatch) {
    int xy = IS_NHWC ? (i / cSize) : (i % xySize);
    float m = (mask != NULL) ? __half2float(mask[n * xySize + xy]) : 1.0f;
    float val = __half2float(inRow[i]) * m;
    acc += val * val;
  }
  srmsShared[tid] = acc;
  __syncthreads();
  for(int s = blockDim.x / 2; s > 0; s >>= 1) {
    if(tid < s) srmsShared[tid] += srmsShared[tid + s];
    __syncthreads();
  }
  if(tid == 0) partialBuf[n * partialStride + blk] = srmsShared[0];
#else
  //Do nothing, FP16 not supported
#endif
}

__global__
void spatialRMSNormReduceKernel(
  const float* partialBuf, float* sumSqBuf, int numBlocksPerBatch, int partialStride
) {
  int n = blockIdx.x;
  if(threadIdx.x != 0)
    return;
  float total = 0.0f;
  const float* row = partialBuf + (size_t)n * partialStride;
  for(int b = 0; b < numBlocksPerBatch; b++)
    total += row[b];
  sumSqBuf[n * partialStride + numBlocksPerBatch] = total;
}

__global__
void spatialRMSNormApplyNHWCKernel(
  const float* in, float* out, const float* gamma, const float* beta, const float* mask,
  const float* maskSum, const float* sumSqBuf,
  int totalElems, int cSize, int xySize, float epsilon, int activation, int numBlocksPerBatch, int partialStride
) {
  int n = blockIdx.y;
  float mSum = maskSum[n];
  float totalSize = mSum * (float)cSize;
  float rms = rsqrtf(sumSqBuf[n * partialStride + numBlocksPerBatch] / totalSize + epsilon);

  const float* inRow = in + (size_t)n * totalElems;
  float* outRow = out + (size_t)n * totalElems;

  for(int i = blockIdx.x * blockDim.x + threadIdx.x; i < totalElems; i += blockDim.x * gridDim.x) {
    int xy = i / cSize;
    int c = i - xy * cSize;
    float maskVal = (mask != NULL) ? mask[n * xySize + xy] : 1.0f;
    float val = inRow[i] * maskVal * rms * gamma[c] + beta[c];
    if(activation == ACTIVATION_RELU) val = fmaxf(val, 0.0f);
    else if(activation == ACTIVATION_MISH) val = mishf(val);
    else if(activation == ACTIVATION_SILU) val = siluf(val);
    val *= maskVal;
    outRow[i] = val;
  }
}

__global__
void spatialRMSNormApplyNHWCHalfKernel(
  const half* in, half* out, const half* gamma, const half* beta, const half* mask,
  const float* maskSum, const float* sumSqBuf,
  int totalElems, int cSize, int xySize, float epsilon, int activation, int numBlocksPerBatch, int partialStride
) {
#ifdef HIP_SUPPORTS_FP16
  int n = blockIdx.y;
  float mSum = maskSum[n];
  float totalSize = mSum * (float)cSize;
  float rms = rsqrtf(sumSqBuf[n * partialStride + numBlocksPerBatch] / totalSize + epsilon);

  const half* inRow = in + (size_t)n * totalElems;
  half* outRow = out + (size_t)n * totalElems;

  for(int i = blockIdx.x * blockDim.x + threadIdx.x; i < totalElems; i += blockDim.x * gridDim.x) {
    int xy = i / cSize;
    int c = i - xy * cSize;
    float maskVal = (mask != NULL) ? __half2float(mask[n * xySize + xy]) : 1.0f;
    float val = __half2float(inRow[i]) * maskVal * rms * __half2float(gamma[c]) + __half2float(beta[c]);
    if(activation == ACTIVATION_RELU) val = fmaxf(val, 0.0f);
    else if(activation == ACTIVATION_MISH) val = mishf(val);
    else if(activation == ACTIVATION_SILU) val = siluf(val);
    val *= maskVal;
    outRow[i] = __float2half(val);
  }
#else
  //Do nothing, FP16 not supported
#endif
}

__global__
void spatialRMSNormApplyNCHWKernel(
  const float* in, float* out, const float* gamma, const float* beta, const float* mask,
  const float* maskSum, const float* sumSqBuf,
  int totalElems, int cSize, int xySize, float epsilon, int activation, int numBlocksPerBatch, int partialStride
) {
  int n = blockIdx.y;
  float mSum = maskSum[n];
  float totalSize = mSum * (float)cSize;
  float rms = rsqrtf(sumSqBuf[n * partialStride + numBlocksPerBatch] / totalSize + epsilon);

  const float* inRow = in + (size_t)n * totalElems;
  float* outRow = out + (size_t)n * totalElems;

  for(int i = blockIdx.x * blockDim.x + threadIdx.x; i < totalElems; i += blockDim.x * gridDim.x) {
    int c = i / xySize;
    int xy = i - c * xySize;
    float maskVal = (mask != NULL) ? mask[n * xySize + xy] : 1.0f;
    float val = inRow[i] * maskVal * rms * gamma[c] + beta[c];
    if(activation == ACTIVATION_RELU) val = fmaxf(val, 0.0f);
    else if(activation == ACTIVATION_MISH) val = mishf(val);
    else if(activation == ACTIVATION_SILU) val = siluf(val);
    val *= maskVal;
    outRow[i] = val;
  }
}

__global__
void spatialRMSNormApplyNCHWHalfKernel(
  const half* in, half* out, const half* gamma, const half* beta, const half* mask,
  const float* maskSum, const float* sumSqBuf,
  int totalElems, int cSize, int xySize, float epsilon, int activation, int numBlocksPerBatch, int partialStride
) {
#ifdef HIP_SUPPORTS_FP16
  int n = blockIdx.y;
  float mSum = maskSum[n];
  float totalSize = mSum * (float)cSize;
  float rms = rsqrtf(sumSqBuf[n * partialStride + numBlocksPerBatch] / totalSize + epsilon);

  const half* inRow = in + (size_t)n * totalElems;
  half* outRow = out + (size_t)n * totalElems;

  for(int i = blockIdx.x * blockDim.x + threadIdx.x; i < totalElems; i += blockDim.x * gridDim.x) {
    int c = i / xySize;
    int xy = i - c * xySize;
    float maskVal = (mask != NULL) ? __half2float(mask[n * xySize + xy]) : 1.0f;
    float val = __half2float(inRow[i]) * maskVal * rms * __half2float(gamma[c]) + __half2float(beta[c]);
    if(activation == ACTIVATION_RELU) val = fmaxf(val, 0.0f);
    else if(activation == ACTIVATION_MISH) val = mishf(val);
    else if(activation == ACTIVATION_SILU) val = siluf(val);
    val *= maskVal;
    outRow[i] = __float2half(val);
  }
#else
  //Do nothing, FP16 not supported
#endif
}

static int spatialRMSNormApplyBlocks(int totalElems, int threads) {
  int blocks = (totalElems + threads - 1) / threads;
  if(blocks < 1) blocks = 1;
  if(blocks > 256) blocks = 256;
  return blocks;
}

void customCudaSpatialRMSNormNHWC(
  const float* in, float* out, const float* gamma, const float* beta, const float* mask, const float* maskSum,
  int nSize, int xySize, int cSize, float epsilon, int activation, float* sumSqBuf
) {
  if(nSize <= 0)
    return;
  if(nSize > 65536)
    throw std::runtime_error("customCudaSpatialRMSNormNHWC: nSize too large");
  checkBufferIndexFitsInt(nSize, xySize, cSize, "customCudaSpatialRMSNormNHWC");
  int totalElems = xySize * cSize;
  int numBlocksPerBatch = spatialRMSNormBlocksPerBatch(totalElems);
  int partialStride = SPATIAL_RMSNORM_BLOCKS_PER_BATCH + 1;

  int threads1 = targetNumThreads;
  int sharedMem1 = threads1 * sizeof(float);
  dim3 grid1(numBlocksPerBatch, nSize);
  spatialRMSNormSumSqKernel<true><<<grid1, threads1, sharedMem1>>>(
    in, mask, sumSqBuf, totalElems, cSize, xySize, numBlocksPerBatch, partialStride);

  spatialRMSNormReduceKernel<<<nSize, 1>>>(sumSqBuf, sumSqBuf, numBlocksPerBatch, partialStride);

  int threads2 = targetNumThreads;
  int applyBlocks = spatialRMSNormApplyBlocks(totalElems, threads2);
  dim3 grid2(applyBlocks, nSize);
  spatialRMSNormApplyNHWCKernel<<<grid2, threads2>>>(
    in, out, gamma, beta, mask, maskSum, sumSqBuf, totalElems, cSize, xySize, epsilon, activation, numBlocksPerBatch, partialStride);
}
void customCudaSpatialRMSNormNHWC(
  const half* in, half* out, const half* gamma, const half* beta, const half* mask, const float* maskSum,
  int nSize, int xySize, int cSize, float epsilon, int activation, float* sumSqBuf
) {
  if(nSize <= 0)
    return;
  if(nSize > 65536)
    throw std::runtime_error("customCudaSpatialRMSNormNHWC: nSize too large");
  checkBufferIndexFitsInt(nSize, xySize, cSize, "customCudaSpatialRMSNormNHWC");
  int totalElems = xySize * cSize;
  int numBlocksPerBatch = spatialRMSNormBlocksPerBatch(totalElems);
  int partialStride = SPATIAL_RMSNORM_BLOCKS_PER_BATCH + 1;

  int threads1 = targetNumThreads;
  int sharedMem1 = threads1 * sizeof(float);
  dim3 grid1(numBlocksPerBatch, nSize);
  spatialRMSNormSumSqHalfKernel<true><<<grid1, threads1, sharedMem1>>>(
    in, mask, sumSqBuf, totalElems, cSize, xySize, numBlocksPerBatch, partialStride);

  spatialRMSNormReduceKernel<<<nSize, 1>>>(sumSqBuf, sumSqBuf, numBlocksPerBatch, partialStride);

  int threads2 = targetNumThreads;
  int applyBlocks = spatialRMSNormApplyBlocks(totalElems, threads2);
  dim3 grid2(applyBlocks, nSize);
  spatialRMSNormApplyNHWCHalfKernel<<<grid2, threads2>>>(
    in, out, gamma, beta, mask, maskSum, sumSqBuf, totalElems, cSize, xySize, epsilon, activation, numBlocksPerBatch, partialStride);
}

void customCudaSpatialRMSNormNCHW(
  const float* in, float* out, const float* gamma, const float* beta, const float* mask, const float* maskSum,
  int nSize, int cSize, int xySize, float epsilon, int activation, float* sumSqBuf
) {
  if(nSize <= 0)
    return;
  if(nSize > 65536)
    throw std::runtime_error("customCudaSpatialRMSNormNCHW: nSize too large");
  checkBufferIndexFitsInt(nSize, cSize, xySize, "customCudaSpatialRMSNormNCHW");
  int totalElems = cSize * xySize;
  int numBlocksPerBatch = spatialRMSNormBlocksPerBatch(totalElems);
  int partialStride = SPATIAL_RMSNORM_BLOCKS_PER_BATCH + 1;

  int threads1 = targetNumThreads;
  int sharedMem1 = threads1 * sizeof(float);
  dim3 grid1(numBlocksPerBatch, nSize);
  spatialRMSNormSumSqKernel<false><<<grid1, threads1, sharedMem1>>>(
    in, mask, sumSqBuf, totalElems, cSize, xySize, numBlocksPerBatch, partialStride);

  spatialRMSNormReduceKernel<<<nSize, 1>>>(sumSqBuf, sumSqBuf, numBlocksPerBatch, partialStride);

  int threads2 = targetNumThreads;
  int applyBlocks = spatialRMSNormApplyBlocks(totalElems, threads2);
  dim3 grid2(applyBlocks, nSize);
  spatialRMSNormApplyNCHWKernel<<<grid2, threads2>>>(
    in, out, gamma, beta, mask, maskSum, sumSqBuf, totalElems, cSize, xySize, epsilon, activation, numBlocksPerBatch, partialStride);
}
void customCudaSpatialRMSNormNCHW(
  const half* in, half* out, const half* gamma, const half* beta, const half* mask, const float* maskSum,
  int nSize, int cSize, int xySize, float epsilon, int activation, float* sumSqBuf
) {
  if(nSize <= 0)
    return;
  if(nSize > 65536)
    throw std::runtime_error("customCudaSpatialRMSNormNCHW: nSize too large");
  checkBufferIndexFitsInt(nSize, cSize, xySize, "customCudaSpatialRMSNormNCHW");
  int totalElems = cSize * xySize;
  int numBlocksPerBatch = spatialRMSNormBlocksPerBatch(totalElems);
  int partialStride = SPATIAL_RMSNORM_BLOCKS_PER_BATCH + 1;

  int threads1 = targetNumThreads;
  int sharedMem1 = threads1 * sizeof(float);
  dim3 grid1(numBlocksPerBatch, nSize);
  spatialRMSNormSumSqHalfKernel<false><<<grid1, threads1, sharedMem1>>>(
    in, mask, sumSqBuf, totalElems, cSize, xySize, numBlocksPerBatch, partialStride);

  spatialRMSNormReduceKernel<<<nSize, 1>>>(sumSqBuf, sumSqBuf, numBlocksPerBatch, partialStride);

  int threads2 = targetNumThreads;
  int applyBlocks = spatialRMSNormApplyBlocks(totalElems, threads2);
  dim3 grid2(applyBlocks, nSize);
  spatialRMSNormApplyNCHWHalfKernel<<<grid2, threads2>>>(
    in, out, gamma, beta, mask, maskSum, sumSqBuf, totalElems, cSize, xySize, epsilon, activation, numBlocksPerBatch, partialStride);
}
