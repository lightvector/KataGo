#ifndef coremlbackend_h
#define coremlbackend_h

#include "../neuralnet/modelversion.h"
#include "../neuralnet/nneval.h"
#include "../neuralnet/nninputs.h"
#include "../neuralnet/nninterface.h"

using namespace std;

void createCoreMLContext();
void destroyCoreMLContext();

int createCoreMLBackend(int modelXLen,
                        int modelYLen,
                        int serverThreadIdx,
                        bool useFP16);

void freeCoreMLBackend(int modelIndex);
int getCoreMLBackendNumSpatialFeatures(int modelIndex);
int getCoreMLBackendNumGlobalFeatures(int modelIndex);
int getCoreMLBackendVersion(int modelIndex);

void getCoreMLHandleOutput(
  float* userInputBuffer,
  float* userInputGlobalBuffer,
  float* policyOutput,
  float* valueOutput,
  float* ownershipOutput,
  float* miscValuesOutput,
  float* moreMiscValuesOutput,
  int modelIndex);

void getCoreMLOutput(
  ComputeHandle* gpuHandle,
  InputBuffers* inputBuffers,
  int numBatchEltsFilled,
  NNResultBuf** inputBufs,
  std::vector<NNOutput*>& outputs);

#endif /* coremlbackend_h */
