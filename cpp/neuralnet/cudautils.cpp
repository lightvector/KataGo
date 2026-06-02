#include "../neuralnet/cudautils.h"

#include <iomanip>
#include "../neuralnet/cudaerrorcheck.h"
#include "../neuralnet/cudaincludes.h"
#include "../neuralnet/cudahelpers.h"
#include "../neuralnet/debugprint.h"

#include "../external/half-2.2.0/include/half.hpp"

//------------------------
#include "../core/using.h"
//------------------------

using half_t = half_float::half;

void CudaUtils::mallocOnDevice(const string& name, int numWeights, void*& deviceBuf, bool useFP16) {
  if(useFP16) {
    size_t halfBytes = numWeights * sizeof(half_t);
    CUDA_ERR(name.c_str(),cudaMalloc(&deviceBuf, halfBytes));
  }
  else {
    size_t floatBytes = numWeights * sizeof(float);
    CUDA_ERR(name.c_str(),cudaMalloc(&deviceBuf, floatBytes));
  }
}

void CudaUtils::mallocAndCopyToDevice(const string& name, const vector<float>& weights, void*& deviceBuf, bool useFP16) {
  size_t numWeights = weights.size();
  if(useFP16) {
    size_t halfBytes = numWeights * sizeof(half_t);
    vector<half_t> weightsHalf(weights.size());
    for(size_t i = 0; i<weights.size(); i++)
      weightsHalf[i] = half_float::half_cast<half_t>(weights[i]);
    CUDA_ERR(name.c_str(),cudaMalloc(&deviceBuf, halfBytes));
    CUDA_ERR(name.c_str(),cudaMemcpy(deviceBuf, weightsHalf.data(), halfBytes, cudaMemcpyHostToDevice));
  }
  else {
    size_t floatBytes = numWeights * sizeof(float);
    CUDA_ERR(name.c_str(),cudaMalloc(&deviceBuf, floatBytes));
    CUDA_ERR(name.c_str(),cudaMemcpy(deviceBuf, weights.data(), floatBytes, cudaMemcpyHostToDevice));
  }
}

void CudaUtils::mallocAndCopyToDevice(const string& name, const float* weights, int numWeights, void*& deviceBuf, bool useFP16) {
  if(useFP16) {
    size_t halfBytes = numWeights * sizeof(half_t);
    vector<half_t> weightsHalf(numWeights);
    for(int i = 0; i<numWeights; i++)
      weightsHalf[i] = half_float::half_cast<half_t>(weights[i]);
    CUDA_ERR(name.c_str(),cudaMalloc(&deviceBuf, halfBytes));
    CUDA_ERR(name.c_str(),cudaMemcpy(deviceBuf, weightsHalf.data(), halfBytes, cudaMemcpyHostToDevice));
  }
  else {
    size_t floatBytes = numWeights * sizeof(float);
    CUDA_ERR(name.c_str(),cudaMalloc(&deviceBuf, floatBytes));
    CUDA_ERR(name.c_str(),cudaMemcpy(deviceBuf, weights, floatBytes, cudaMemcpyHostToDevice));
  }
}

//Only use in testing, allocates an intermediate buffer in the case of FP16 which will be very slow.
void CudaUtils::expensiveCopyFromDevice(const string& name, float* weights, int numWeights, const void* deviceBuf, bool useFP16) {
  if(useFP16) {
    vector<half_t> weightsHalf(numWeights);
    size_t halfBytes = numWeights * sizeof(half_t);
    CUDA_ERR(name.c_str(),cudaMemcpy(weightsHalf.data(), deviceBuf, halfBytes, cudaMemcpyDeviceToHost));
    for(int i = 0; i<numWeights; i++)
      weights[i] = weightsHalf[i];
  }
  else {
    size_t floatBytes = numWeights * sizeof(float);
    CUDA_ERR(name.c_str(),cudaMemcpy(weights, deviceBuf, floatBytes, cudaMemcpyDeviceToHost));
  }
}

void CudaUtils::debugPrint3D(
  const string& name, const void* deviceBuf,
  int batchSize, int cSize, int spatialSize, bool useNHWC, bool useFP16,
  const void* maskBuf
) {
  int totalSize = batchSize * cSize * spatialSize;
  vector<float> values(totalSize);
  expensiveCopyFromDevice(name, values.data(), totalSize, deviceBuf, useFP16);

  vector<float> maskValues;
  float* maskPtr = nullptr;
  if(maskBuf != nullptr) {
    maskValues.resize(batchSize * spatialSize);
    expensiveCopyFromDevice(name + ":mask", maskValues.data(), batchSize * spatialSize, maskBuf, useFP16);
    maskPtr = maskValues.data();
  }

  if(useNHWC) {
    DebugPrint::print3DSummary(name, values.data(), batchSize, spatialSize, cSize, "NSC", batchSize, spatialSize, maskPtr);
#ifdef DEBUG_INTERMEDIATE_VALUES_VERBOSE
    DebugPrint::print3DVerbose(name, values.data(), batchSize, spatialSize, cSize, "NSC");
#endif
  }
  else {
    DebugPrint::print3DSummary(name, values.data(), batchSize, cSize, spatialSize, "NCS", batchSize, spatialSize, maskPtr);
#ifdef DEBUG_INTERMEDIATE_VALUES_VERBOSE
    DebugPrint::print3DVerbose(name, values.data(), batchSize, cSize, spatialSize, "NCS");
#endif
  }
}

void CudaUtils::debugPrint2D(const string& name, const void* deviceBuf, int batchSize, int cSize, bool useFP16) {
  vector<float> values(batchSize * cSize);
  expensiveCopyFromDevice(name, values.data(), values.size(), deviceBuf, useFP16);
  DebugPrint::print2DSummary(name, values.data(), batchSize, cSize);
#ifdef DEBUG_INTERMEDIATE_VALUES_VERBOSE
  DebugPrint::print2DVerbose(name, values.data(), batchSize, cSize);
#endif
}

void CudaUtils::checkBufferSize(int batchSize, int xSize, int ySize, int channels) {
  if((int64_t)batchSize * xSize * ySize * channels >= (int64_t)1 << 31)
    throw StringError("Batch size too large, resulting GPU buffers might exceed 2^31 entries which is not currently supported");
}

void CudaUtils::hostMallocZeroOneBufs(void*& zeroBuf, void*& oneBuf, bool useFP16) {
  if(!useFP16) {
    zeroBuf = malloc(sizeof(float));
    oneBuf = malloc(sizeof(float));
    *((float*)zeroBuf) = 0.0f;
    *((float*)oneBuf) = 1.0f;
  }
  else {
    //Convert to FP16 on the device, then copy back so we have it in host memory
    float zero = 0.0f;
    float one = 1.0f;
    void* zeroTmp;
    void* oneTmp;
    mallocAndCopyToDevice("Buffers",&zero,1,zeroTmp,useFP16);
    mallocAndCopyToDevice("Buffers",&one,1,oneTmp,useFP16);
    zeroBuf = malloc(sizeof(half_t));
    oneBuf = malloc(sizeof(half_t));
    CUDA_ERR("Buffers",cudaMemcpy(zeroBuf,zeroTmp,sizeof(half_t),cudaMemcpyDeviceToHost));
    CUDA_ERR("Buffers",cudaMemcpy(oneBuf,oneTmp,sizeof(half_t),cudaMemcpyDeviceToHost));
    cudaFree(zeroTmp);
    cudaFree(oneTmp);
  }
}
