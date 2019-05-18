#include "../neuralnet/nninterface.h"
#include "../neuralnet/nninputs.h"

using namespace std;

void NeuralNet::globalInitialize(const string& tensorflowGpuVisibleGpuList,
                                 double tensorflowPerProcessGpuMemoryFraction) {
  assert(false);
}

void NeuralNet::globalCleanup() {
  assert(false);
}

LoadedModel* NeuralNet::loadModelFile(const string& file, int modelFileIdx) {
  assert(false);
  return nullptr;
}

void NeuralNet::freeLoadedModel(LoadedModel* loadedModel) {
  assert(false);
}

int NeuralNet::getModelVersion(const LoadedModel* loadedModel) {
  assert(false);
  return 0;
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
  assert(false);
  return nullptr;
}

void NeuralNet::freeLocalGpuHandle(LocalGpuHandle* gpuHandle) {
  assert(false);
}

InputBuffers* NeuralNet::createInputBuffers(const LoadedModel* loadedModel,
                                            int maxBatchSize,
                                            int nnXLen, int nnYLen) {
  assert(false);
  return nullptr;
}

void NeuralNet::freeInputBuffers(InputBuffers* buffers) {
  assert(false);
}

float* NeuralNet::getRowInplace(InputBuffers* buffers, int rowIdx) {
  assert(false);
  return nullptr;
}

float* NeuralNet::getRowGlobalInplace(InputBuffers* buffers, int rowIdx) {
  assert(false);
  return nullptr;
}

bool* NeuralNet::getSymmetriesInplace(InputBuffers* buffers) {
  assert(false);
  return nullptr;
}

int NeuralNet::getRowLen(const InputBuffers* buffers) {
  assert(false);
  return 0;
}

int NeuralNet::getRowGlobalLen(const InputBuffers* buffers) {
  assert(false);
  return 0;
}

void NeuralNet::getOutput(LocalGpuHandle* gpuHandle,
                          InputBuffers* buffers,
                          int numFilledRows,
                          vector<NNOutput*>& outputs) {
  assert(false);
}
