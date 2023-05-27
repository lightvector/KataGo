#ifndef coremlbackend_h
#define coremlbackend_h

#include "../neuralnet/modelversion.h"
#include "../neuralnet/nneval.h"
#include "../neuralnet/nninputs.h"
#include "../neuralnet/nninterface.h"

using namespace std;

namespace CoreMLProcess {
  int calculateIndex(const int y, const int x, const int xLen);
  void processValue(const InputBuffers* inputBuffers, NNOutput* currentOutput, const size_t row);

  void processOwnership(
    const InputBuffers* inputBuffers,
    NNOutput* currentOutput,
    const ComputeHandle* gpuHandle,
    const int symmetry,
    const size_t row);

  void
  processScoreValues(const InputBuffers* inputBuffers, NNOutput* currentOutput, const int version, const size_t row);

  void getCoreMLOutput(
    ComputeHandle* gpuHandle,
    InputBuffers* inputBuffers,
    int numBatchEltsFilled,
    NNResultBuf** inputBufs,
    vector<NNOutput*>& outputs);

  void createCoreMLContext();
  void destroyCoreMLContext();

  int createCoreMLBackend(int modelXLen, int modelYLen, int serverThreadIdx, bool useFP16);

  void freeCoreMLBackend(int modelIndex);
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
};

#endif /* coremlbackend_h */
