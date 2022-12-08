#ifndef NEURALNET_CUDAUTILS_H
#define NEURALNET_CUDAUTILS_H

#include "../core/global.h"

namespace CudaUtils {
  void mallocOnDevice(const std::string& name, int numWeights, void*& deviceBuf, bool useFP16);
  void mallocAndCopyToDevice(const std::string& name, const std::vector<float>& weights, void*& deviceBuf, bool useFP16);
  void mallocAndCopyToDevice(const std::string& name, const float* weights, int numWeights, void*& deviceBuf, bool useFP16);

  //Only use in testing, allocates an intermediate buffer in the case of FP16 which will be very slow.
  void expensiveCopyFromDevice(const std::string& name, float* weights, int numWeights, const void* deviceBuf, bool useFP16);

  void debugPrint2D(const std::string& name, const void* deviceBuf, int batchSize, int cSize, bool useFP16);
  void debugPrint4D(const std::string& name, const void* deviceBuf, int batchSize, int cSize, int xSize, int ySize, bool useNHWC, bool useFP16);

  void checkBufferSize(int batchSize, int xSize, int ySize, int channels);
  void hostMallocZeroOneBufs(void*& zeroBuf, void*& oneBuf, bool useFP16);
}

#endif // NEURALNET_CUDAUTILS_H
