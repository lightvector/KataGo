#ifndef NEURALNET_NNINTERFACE_H_
#define NEURALNET_NNINTERFACE_H_

#include "../core/global.h"
#include "../core/hash.h"
#include "../core/logger.h"
#include "../neuralnet/nninputs.h"

struct LocalGpuHandle; //Not thread-safe, each handle should only be used by one thread
struct LoadedModel;
struct InputBuffers;

//Generic interface to neural net inference.
//There are two backends - a Tensorflow backend, and a CUDA backend.
//Some parameters to these functions only apply for one backend or another.

namespace NeuralNet {
  void globalInitialize(
    const std::string& tensorflowGpuVisibleGpuList,
    double tensorflowPerProcessGpuMemoryFraction
  );
  void globalCleanup();

  LoadedModel* loadModelFile(const std::string& file, int modelFileIdx);
  void freeLoadedModel(LoadedModel* loadedModel);

  int getModelVersion(const LoadedModel* loadedModel);

  //Any given thread should only ever create one of these at a time.
  //When using the CUDA backend, will mutably set the GPU that this thread is associated with to the specified index.
  //If logger is specified, may output some info messages to it.
  //If requireExactNNLen is true, the backend is allowed to assume that all boards to evaluate will be of size exactly
  //equal to (nnXLen,nnYLen) rather than smaller, and skip any masking operations.
  LocalGpuHandle* createLocalGpuHandle(
    const LoadedModel* loadedModel,
    Logger* logger,
    int maxBatchSize,
    int nnXLen,
    int nnYLen,
    bool requireExactNNLen,
    bool inputsUseNHWC,
    int cudaGpuIdxForThisThread,
    bool cudaUseFP16,
    bool cudaUseNHWC
  );
  void freeLocalGpuHandle(LocalGpuHandle* gpuHandle);

  InputBuffers* createInputBuffers(const LoadedModel* loadedModel, int maxBatchSize, int nnXLen, int nnYLen);
  void freeInputBuffers(InputBuffers* buffers);

  float* getRowInplace(InputBuffers* buffers, int rowIdx);
  float* getRowGlobalInplace(InputBuffers* buffers, int rowIdx);
  bool* getSymmetriesInplace(InputBuffers* buffers);

  int getRowLen(const InputBuffers* buffers);
  int getRowGlobalLen(const InputBuffers* buffers);

  void getOutput(LocalGpuHandle* gpuHandle, InputBuffers* buffers, int numFilledRows, std::vector<NNOutput*>& outputs);
}

//Model versions
namespace NNModelVersion {
  extern const int latestModelVersionImplemented;
  extern const int defaultModelVersion;

  //Which V* feature version from NNInputs does a given model version consume?
  int getInputsVersion(int modelVersion);
  //Convenience functions, feeds forward the number of features and the size of the row vector that the net takes as input
  int getNumSpatialFeatures(int modelVersion);
  int getNumGlobalFeatures(int modelVersion);
  int getRowSize(int modelVersion);
}


#endif  // NEURALNET_NNINTERFACE_H_
