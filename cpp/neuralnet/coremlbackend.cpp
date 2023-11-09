#ifdef USE_COREML_BACKEND

#include "../neuralnet/modelversion.h"
#include "../neuralnet/nneval.h"
#include "../neuralnet/nninputs.h"
#include "../neuralnet/nninterface.h"
#include "../neuralnet/metalbackend.h"
#include "../neuralnet/coremlbackend.h"


using namespace std;

//--------------------------------------------------------------

size_t CoreMLProcess::calculateBufferOffset(size_t row, size_t singleResultElts, size_t resultChannels) {
  return row * singleResultElts * resultChannels;
}

int CoreMLProcess::calculateIndex(const int y, const int x, const int xLen) {
  return (y * xLen) + x;
}

float CoreMLProcess::policyOptimismCalc(const double policyOptimism, const float p, const float pOpt) {
  return MetalProcess::policyOptimismCalc(policyOptimism, p, pOpt);
}

float CoreMLProcess::assignPolicyValue(
  const size_t policyResultChannels,
  const double policyOptimism,
  const float* targetBuffer,
  const size_t outputIdx,
  const size_t singleModelPolicyResultElts) {
  return (policyResultChannels == 1)
           ? targetBuffer[outputIdx]
           : policyOptimismCalc(
               policyOptimism, targetBuffer[outputIdx], targetBuffer[outputIdx + singleModelPolicyResultElts]);
}

void CoreMLProcess::processPolicy(
  InputBuffers* inputBuffers,
  NNOutput* currentOutput,
  const ComputeHandle* gpuHandle,
  NNResultBuf* inputBuf,
  size_t row) {
  const int gpuHandleXLen = gpuHandle->nnXLen;
  const int gpuHandleYLen = gpuHandle->nnYLen;
  const int modelXLen = gpuHandle->modelXLen;
  auto& inputBuffersRef = *inputBuffers;
  const size_t targetBufferOffset =
    calculateBufferOffset(row, inputBuffersRef.singleModelPolicyResultElts, inputBuffersRef.policyResultChannels);
  const size_t currentBufferOffset =
    calculateBufferOffset(row, inputBuffersRef.singlePolicyProbsElts, inputBuffersRef.policyResultChannels);
  float* targetBuffer = &inputBuffersRef.policyResults[targetBufferOffset];
  float* currentBuffer = &inputBuffersRef.policyProbsBuffer[currentBufferOffset];
  const auto symmetry = inputBuf->symmetry;
  const auto policyOptimism = inputBuf->policyOptimism;

  auto processBuffer = [&](int y, int x) {
    int outputIdx = calculateIndex(y, x, modelXLen);
    int probsIdx = calculateIndex(y, x, gpuHandleXLen);

    currentBuffer[probsIdx] = assignPolicyValue(
      inputBuffersRef.policyResultChannels,
      policyOptimism,
      targetBuffer,
      outputIdx,
      inputBuffersRef.singleModelPolicyResultElts);
  };

  for(int y = 0; y < gpuHandleYLen; y++) {
    for(int x = 0; x < gpuHandleXLen; x++) {
      processBuffer(y, x);
    }
  }

  assert(inputBuffersRef.singleModelPolicyResultElts > 0);
  assert(inputBuffersRef.singlePolicyProbsElts > 0);
  size_t endOfModelPolicyIdx = inputBuffersRef.singleModelPolicyResultElts - 1;
  size_t endOfPolicyProbsIdx = inputBuffersRef.singlePolicyProbsElts - 1;

  currentOutput->policyProbs[endOfPolicyProbsIdx] = assignPolicyValue(
    inputBuffersRef.policyResultChannels,
    policyOptimism,
    targetBuffer,
    endOfModelPolicyIdx,
    inputBuffersRef.singleModelPolicyResultElts);

  SymmetryHelpers::copyOutputsWithSymmetry(
    currentBuffer, currentOutput->policyProbs, 1, gpuHandleYLen, gpuHandleXLen, symmetry);
}

void CoreMLProcess::processValue(
  const InputBuffers* inputBuffers,
  NNOutput* currentOutput,
  const size_t row) {
  MetalProcess::processValue(inputBuffers, currentOutput, row);
}

void CoreMLProcess::processOwnership(
  const InputBuffers* inputBuffers,
  NNOutput* currentOutput,
  const ComputeHandle* gpuHandle,
  const int symmetry,
  const size_t row) {
  // If there's no ownership map, we have nothing to do
  if(currentOutput->whiteOwnerMap == nullptr) {
    return;
  }

  // Extract useful values from buffers and GPU handle
  const int nnXLen = gpuHandle->nnXLen;
  const int nnYLen = gpuHandle->nnYLen;
  const int modelXLen = gpuHandle->modelXLen;

  const size_t singleOwnershipResultElts = inputBuffers->singleNnOwnershipResultElts;
  const size_t singleOwnerMapElts = inputBuffers->singleOwnerMapElts;

  // Calculate starting points in the buffers
  const float* ownershipOutputBuf = &inputBuffers->ownershipResults[row * singleOwnershipResultElts];
  float* ownerMapBuf = &inputBuffers->ownerMapBuffer[row * singleOwnerMapElts];

  // Copy data from ownership output buffer to owner map buffer
  for(int y = 0; y < nnYLen; y++) {
    for(int x = 0; x < nnXLen; x++) {
      int outputIdx = calculateIndex(y, x, modelXLen);
      int ownerMapIdx = calculateIndex(y, x, nnXLen);
      ownerMapBuf[ownerMapIdx] = ownershipOutputBuf[outputIdx];
    }
  }

  // Apply symmetry to the owner map buffer and copy it to the output's whiteOwnerMap
  SymmetryHelpers::copyOutputsWithSymmetry(ownerMapBuf, currentOutput->whiteOwnerMap, 1, nnYLen, nnXLen, symmetry);
}

void CoreMLProcess::processScoreValues(
  const InputBuffers* inputBuffers,
  NNOutput* currentOutput,
  const int version,
  const size_t row) {
  const size_t singleScoreValuesResultElts = inputBuffers->singleScoreValuesResultElts;
  const size_t scoreValuesOutputBufOffset = row * singleScoreValuesResultElts;
  const float* scoreValuesOutputBuf = &inputBuffers->scoreValuesResults[scoreValuesOutputBufOffset];
  const size_t singleMoreMiscValuesResultElts = inputBuffers->singleMoreMiscValuesResultElts;
  const size_t moreMiscValuesOutputBufOffset = row * singleMoreMiscValuesResultElts;
  const float* moreMiscValuesOutputBuf = &inputBuffers->moreMiscValuesResults[moreMiscValuesOutputBufOffset];

  currentOutput->whiteScoreMean = scoreValuesOutputBuf[0];
  currentOutput->whiteScoreMeanSq = currentOutput->whiteScoreMean * currentOutput->whiteScoreMean;
  currentOutput->whiteLead = currentOutput->whiteScoreMean;
  currentOutput->varTimeLeft = 0.0f;
  currentOutput->shorttermWinlossError = 0.0f;
  currentOutput->shorttermScoreError = 0.0f;

  if(version >= 4) {
    currentOutput->whiteScoreMean = scoreValuesOutputBuf[0];
    currentOutput->whiteScoreMeanSq = scoreValuesOutputBuf[1];
    currentOutput->whiteLead = (version >= 8) ? scoreValuesOutputBuf[2] : currentOutput->whiteScoreMean;
    currentOutput->varTimeLeft = (version >= 9) ? scoreValuesOutputBuf[3] : currentOutput->varTimeLeft;
    currentOutput->shorttermWinlossError =
      (version >= 9) ? moreMiscValuesOutputBuf[0] : currentOutput->shorttermWinlossError;
    currentOutput->shorttermScoreError = (version >= 9) ? moreMiscValuesOutputBuf[1] : currentOutput->shorttermScoreError;
  }
}

void CoreMLProcess::getCoreMLOutput(
  ComputeHandle* gpuHandle,
  InputBuffers* inputBuffers,
  int numBatchEltsFilled,
  NNResultBuf** inputBufs,
  vector<NNOutput*>& outputs) {
  int batchSize = numBatchEltsFilled;
  int nnXLen = gpuHandle->nnXLen;
  int nnYLen = gpuHandle->nnYLen;
  int modelXLen = gpuHandle->modelXLen;
  int modelYLen = gpuHandle->modelYLen;
  int version = gpuHandle->modelVersion;
  int numSpatialFeatures = NNModelVersion::getNumSpatialFeatures(version);
  int numGlobalFeatures = NNModelVersion::getNumGlobalFeatures(version);

  assert(batchSize <= inputBuffers->maxBatchSize);
  assert(batchSize > 0);
  assert((numSpatialFeatures * modelXLen * modelYLen) == inputBuffers->singleInputElts);
  assert(numGlobalFeatures == inputBuffers->singleInputGlobalElts);
  assert(version == getCoreMLBackendVersion(gpuHandle->modelIndex));

  size_t policyResultChannels = inputBuffers->policyResultChannels;
  size_t singleSpatialElts = inputBuffers->singleSpatialElts;
  size_t singleInputElts = inputBuffers->singleInputElts;
  size_t singleInputGlobalElts = inputBuffers->singleInputGlobalElts;
  size_t singlePolicyResultElts = inputBuffers->singleModelPolicyResultElts;
  size_t singleValueResultElts = inputBuffers->singleValueResultElts;
  size_t singleOwnershipResultElts = inputBuffers->singleModelOwnershipResultElts;
  size_t singleScoreValuesResultElts = inputBuffers->singleScoreValuesResultElts;
  size_t singleMoreMiscValuesResultElts = inputBuffers->singleMoreMiscValuesResultElts;

  assert(singleInputElts == (modelXLen * modelYLen * 22));
  assert(singleInputGlobalElts == 19);
  assert(singlePolicyResultElts == ((modelXLen * modelYLen) + 1));
  assert(singleValueResultElts == 3);
  assert(singleOwnershipResultElts == (modelXLen * modelYLen));
  assert(singleScoreValuesResultElts == 10);
  assert(singleMoreMiscValuesResultElts == 8);

  // Get CoreML backend output
  for(size_t row = 0; row < batchSize; row++) {
    float* rowSpatialBuffer = &inputBuffers->rowSpatialBuffer[singleSpatialElts * row];
    float* rowSpatialInput = &inputBuffers->userInputBuffer[singleInputElts * row];
    float* rowGlobalInput = &inputBuffers->userInputGlobalBuffer[singleInputGlobalElts * row];
    float* policyOutputBuf = &inputBuffers->policyResults[row * (singlePolicyResultElts * policyResultChannels)];
    float* valueOutputBuf = &inputBuffers->valueResults[row * singleValueResultElts];
    float* ownershipOutputBuf = &inputBuffers->ownershipResults[row * singleOwnershipResultElts];
    float* miscValuesOutputBuf = &inputBuffers->scoreValuesResults[row * singleScoreValuesResultElts];
    float* moreMiscValuesOutputBuf = &inputBuffers->moreMiscValuesResults[row * singleMoreMiscValuesResultElts];

    const float* rowGlobal = inputBufs[row]->rowGlobal;
    const float* rowSpatial = inputBufs[row]->rowSpatial;

    std::copy(&rowGlobal[0], &rowGlobal[numGlobalFeatures], rowGlobalInput);

    assert(gpuHandle->inputsUseNHWC == false);

    SymmetryHelpers::copyInputsWithSymmetry(
      rowSpatial,
      rowSpatialBuffer,
      1,
      nnYLen,
      nnXLen,
      numSpatialFeatures,
      gpuHandle->inputsUseNHWC,
      inputBufs[row]->symmetry);

    for(int c = 0; c < numSpatialFeatures; c++) {
      for(int y = 0; y < nnYLen; y++) {
        for(int x = 0; x < nnXLen; x++) {
          int bufferIdx = (c * nnYLen * nnXLen) + (y * nnXLen) + x;
          int inputIdx = (c * modelYLen * modelXLen) + (y * modelXLen) + x;
          rowSpatialInput[inputIdx] = rowSpatialBuffer[bufferIdx];
        }
      }
    }

    getCoreMLHandleOutput(
      rowSpatialInput,
      rowGlobalInput,
      policyOutputBuf,
      valueOutputBuf,
      ownershipOutputBuf,
      miscValuesOutputBuf,
      moreMiscValuesOutputBuf,
      gpuHandle->modelIndex);
  }

  // Fill results by CoreML model output
  for(size_t row = 0; row < batchSize; row++) {
    NNOutput* output = outputs[row];
    CoreMLProcess::processPolicy(inputBuffers, output, gpuHandle, inputBufs[row], row);
    CoreMLProcess::processValue(inputBuffers, output, row);
    CoreMLProcess::processOwnership(inputBuffers, output, gpuHandle, inputBufs[row]->symmetry, row);
    CoreMLProcess::processScoreValues(inputBuffers, output, version, row);
  }
}

#endif  // USE_COREML_BACKEND
