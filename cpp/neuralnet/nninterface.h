#ifndef NNINTERFACE_H
#define NNINTERFACE_H

#include "../core/global.h"
#include "../core/hash.h"
#include "../neuralnet/nninputs.h"

struct LocalGpuHandle; //Not thread-safe, each handle should only be used by one thread
struct LoadedModel;
struct InputBuffers;

namespace NeuralNet {
  void globalInitialize(
    const string& tensorflowGpuVisibleDeviceList,
    double tensorflowPerProcessGpuMemoryFraction
  );
  void globalCleanup();
  
  LocalGpuHandle* createLocalGpuHandle();
  void freeLocalGpuHandle(LocalGpuHandle* gpuHandle);

  LoadedModel* loadModelFile(const string& file, int modelFileIdx);
  void freeLoadedModel(LoadedModel* loadedModel);
  
  InputBuffers* createInputBuffers(const LoadedModel* loadedModel, int maxBatchSize);
  void freeInputBuffers(InputBuffers* buffers);
  
  float* getRowInplace(InputBuffers* buffers, int rowIdx);
  bool* getSymmetriesInplace(InputBuffers* buffers);
  
  void getOutput(LocalGpuHandle* gpuHandle, InputBuffers* buffers, int numFilledRows, vector<NNOutput*>& outputs);

}



#endif
