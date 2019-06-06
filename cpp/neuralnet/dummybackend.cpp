#include "../neuralnet/nninterface.h"

#include "../neuralnet/nninputs.h"

using namespace std;

void NeuralNet::globalInitialize() {
  // Do nothing, calling this is okay even if there is no neural net
  // as long as we don't attempt to actually load a net file and use one.
}

void NeuralNet::globalCleanup() {
  // Do nothing, calling this is okay even if there is no neural net
  // as long as we don't attempt to actually load a net file and use one.
}

LoadedModel* NeuralNet::loadModelFile(const string& file, int modelFileIdx) {
  (void)file;
  (void)modelFileIdx;
  throw StringError("Dummy neural net backend: NeuralNet::loadModelFile unimplemented");
}

void NeuralNet::freeLoadedModel(LoadedModel* loadedModel) {
  (void)loadedModel;
  throw StringError("Dummy neural net backend: NeuralNet::freeLoadedModel unimplemented");
}

int NeuralNet::getModelVersion(const LoadedModel* loadedModel) {
  (void)loadedModel;
  throw StringError("Dummy neural net backend: NeuralNet::getModelVersion unimplemented");
}

ComputeHandle* NeuralNet::createComputeHandle(
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
  (void)loadedModel;
  (void)logger;
  (void)maxBatchSize;
  (void)nnXLen;
  (void)nnYLen;
  (void)requireExactNNLen;
  (void)inputsUseNHWC;
  (void)cudaGpuIdxForThisThread;
  (void)cudaUseFP16;
  (void)cudaUseNHWC;
  throw StringError("Dummy neural net backend: NeuralNet::createLocalGpuHandle unimplemented");
}

void NeuralNet::freeComputeHandle(ComputeHandle* gpuHandle) {
  if(gpuHandle != NULL)
    throw StringError("Dummy neural net backend: NeuralNet::freeLocalGpuHandle unimplemented");
}

InputBuffers* NeuralNet::createInputBuffers(const LoadedModel* loadedModel, int maxBatchSize, int nnXLen, int nnYLen) {
  (void)loadedModel;
  (void)maxBatchSize;
  (void)nnXLen;
  (void)nnYLen;
  throw StringError("Dummy neural net backend: NeuralNet::createInputBuffers unimplemented");
}

void NeuralNet::freeInputBuffers(InputBuffers* buffers) {
  if(buffers != NULL)
    throw StringError("Dummy neural net backend: NeuralNet::freeInputBuffers unimplemented");
}

float* NeuralNet::getBatchEltSpatialInplace(InputBuffers* buffers, int nIdx) {
  (void)buffers;
  (void)nIdx;
  throw StringError("Dummy neural net backend: NeuralNet::getBatchEltSpatialInplace unimplemented");
}

float* NeuralNet::getBatchEltGlobalInplace(InputBuffers* buffers, int nIdx) {
  (void)buffers;
  (void)nIdx;
  throw StringError("Dummy neural net backend: NeuralNet::getBatchEltGlobalInplace unimplemented");
}

bool* NeuralNet::getSymmetriesInplace(InputBuffers* buffers) {
  (void)buffers;
  throw StringError("Dummy neural net backend: NeuralNet::getSymmetriesInplace unimplemented");
}

int NeuralNet::getBatchEltSpatialLen(const InputBuffers* buffers) {
  (void)buffers;
  throw StringError("Dummy neural net backend: NeuralNet::getBatchEltSpatialLen unimplemented");
}

int NeuralNet::getBatchEltGlobalLen(const InputBuffers* buffers) {
  (void)buffers;
  throw StringError("Dummy neural net backend: NeuralNet::getBatchEltGlobalLen unimplemented");
}

void NeuralNet::getOutput(
  ComputeHandle* gpuHandle,
  InputBuffers* buffers,
  int numBatchEltsFilled,
  vector<NNOutput*>& outputs) {
  (void)gpuHandle;
  (void)buffers;
  (void)numBatchEltsFilled;
  (void)outputs;
  throw StringError("Dummy neural net backend: NeuralNet::getOutput unimplemented");
}
