#ifndef coremlbackend_h
#define coremlbackend_h

#include "../neuralnet/modelversion.h"
#include "../neuralnet/nneval.h"
#include "../neuralnet/nninputs.h"
#include "../neuralnet/nninterface.h"

using namespace std;

namespace CoreMLProcess {
  size_t calculateBufferOffset(size_t row, size_t singleResultElts, size_t resultChannels);
  int calculateIndex(const int y, const int x, const int xLen);
  float policyOptimismCalc(const double policyOptimism, const float p, const float pOpt);

  float assignPolicyValue(
    const size_t policyResultChannels,
    const double policyOptimism,
    const float* targetBuffer,
    const size_t outputIdx,
    const size_t singleModelPolicyResultElts);

  void processPolicy(
    InputBuffers* inputBuffers,
    NNOutput* currentOutput,
    const ComputeHandle* gpuHandle,
    NNResultBuf* inputBuf,
    size_t row);

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

};

#endif /* coremlbackend_h */
