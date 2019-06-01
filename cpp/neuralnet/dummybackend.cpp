#include "../neuralnet/nninterface.h"
#include "../neuralnet/nninputs.h"

using namespace std;

void NeuralNet::globalInitialize(const string& tensorflowGpuVisibleGpuList,
                                 double tensorflowPerProcessGpuMemoryFraction) {
  //Do nothing, calling this is okay even if there is no neural net
  //as long as we don't attempt to actually load a net file and use one.
}

void NeuralNet::globalCleanup() {
  //Do nothing, calling this is okay even if there is no neural net
  //as long as we don't attempt to actually load a net file and use one.
}

LoadedModel* NeuralNet::loadModelFile(const string& file, int modelFileIdx) {
  throw StringError("Dummy neural net backend: NeuralNet::loadModelFile unimplemented");
}

void NeuralNet::freeLoadedModel(LoadedModel* loadedModel) {
  throw StringError("Dummy neural net backend: NeuralNet::freeLoadedModel unimplemented");
}

int NeuralNet::getModelVersion(const LoadedModel* loadedModel) {
  throw StringError("Dummy neural net backend: NeuralNet::getModelVersion unimplemented");
}

LocalGpuHandle* NeuralNet::createLocalGpuHandle(
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
) {
  throw StringError("Dummy neural net backend: NeuralNet::createLocalGpuHandle unimplemented");
}

void NeuralNet::freeLocalGpuHandle(LocalGpuHandle* gpuHandle) {
  throw StringError("Dummy neural net backend: NeuralNet::freeLocalGpuHandle unimplemented");
}

InputBuffers* NeuralNet::createInputBuffers(
  const LoadedModel* loadedModel,
  int maxBatchSize,
  int nnXLen, int nnYLen
) {
  throw StringError("Dummy neural net backend: NeuralNet::createInputBuffers unimplemented");
}

void NeuralNet::freeInputBuffers(InputBuffers* buffers) {
  throw StringError("Dummy neural net backend: NeuralNet::freeInputBuffers unimplemented");
}

float* NeuralNet::getRowInplace(InputBuffers* buffers, int rowIdx) {
  throw StringError("Dummy neural net backend: NeuralNet::getRowInplace unimplemented");
}

float* NeuralNet::getRowGlobalInplace(InputBuffers* buffers, int rowIdx) {
  throw StringError("Dummy neural net backend: NeuralNet::getRowGlobalInplace unimplemented");
}

bool* NeuralNet::getSymmetriesInplace(InputBuffers* buffers) {
  throw StringError("Dummy neural net backend: NeuralNet::getSymmetriesInplace unimplemented");
}

int NeuralNet::getRowLen(const InputBuffers* buffers) {
  throw StringError("Dummy neural net backend: NeuralNet::getRowLen unimplemented");
}

int NeuralNet::getRowGlobalLen(const InputBuffers* buffers) {
  throw StringError("Dummy neural net backend: NeuralNet::getRowGlobalLen unimplemented");
}

void NeuralNet::getOutput(
  LocalGpuHandle* gpuHandle,
  InputBuffers* buffers,
  int numFilledRows,
  vector<NNOutput*>& outputs
) {
  throw StringError("Dummy neural net backend: NeuralNet::getOutput unimplemented");
}
