#ifdef USE_METAL_BACKEND

#include "../neuralnet/modelversion.h"
#include "../neuralnet/nneval.h"
#include "../neuralnet/nninputs.h"
#include "../neuralnet/nninterface.h"
#include "../neuralnet/metalbackend.h"

using namespace std;

//---------------------------------------------------------------------------------------------------------

void NeuralNet::globalInitialize() {
  // Do nothing, calling this is okay even if there is no neural net
  // as long as we don't attempt to actually load a net file and use one.
}

void NeuralNet::globalCleanup() {
  // Do nothing, calling this is okay even if there is no neural net
  // as long as we don't attempt to actually load a net file and use one.
}

//------------------------------------------------------------------------------

struct LoadedModel {
  ModelDesc modelDesc;

  LoadedModel(const string& fileName, const string& expectedSha256) {
    ModelDesc::loadFromFileMaybeGZipped(fileName, modelDesc, expectedSha256);
  }

  LoadedModel() = delete;
  LoadedModel(const LoadedModel&) = delete;
  LoadedModel& operator=(const LoadedModel&) = delete;
};

LoadedModel* NeuralNet::loadModelFile(const string& file, const string& expectedSha256) {
  LoadedModel* loadedModel = new LoadedModel(file, expectedSha256);
  return loadedModel;
}

void NeuralNet::freeLoadedModel(LoadedModel* loadedModel) {
  delete loadedModel;
}

string NeuralNet::getModelName(const LoadedModel* loadedModel) {
  return loadedModel->modelDesc.name;
}

int NeuralNet::getModelVersion(const LoadedModel* loadedModel) {
  return loadedModel->modelDesc.version;
}

Rules NeuralNet::getSupportedRules(const LoadedModel* loadedModel, const Rules& desiredRules, bool& supported) {
  return loadedModel->modelDesc.getSupportedRules(desiredRules, supported);
}

struct ComputeContext {
  int nnXLen;
  int nnYLen;

  ComputeContext(int nnX, int nnY) {
    nnXLen = nnX;
    nnYLen = nnY;
  }

  ~ComputeContext() {}

  ComputeContext() = delete;
  ComputeContext(const ComputeContext&) = delete;
  ComputeContext& operator=(const ComputeContext&) = delete;
};

ComputeContext* NeuralNet::createComputeContext(
  const vector<int>& gpuIdxs,
  Logger* logger,
  int nnXLen,
  int nnYLen,
  const string& openCLTunerFile,
  const string& homeDataDirOverride,
  bool openCLReTunePerBoardSize,
  enabled_t useFP16Mode,
  enabled_t useNHWCMode,
  const LoadedModel* loadedModel) {

  (void)gpuIdxs;
  (void)logger;
  (void)openCLTunerFile;
  (void)homeDataDirOverride;
  (void)openCLReTunePerBoardSize;
  (void)loadedModel;

  createMetalContext(nnXLen, nnYLen, useFP16Mode, useNHWCMode);

  return new ComputeContext(nnXLen, nnYLen);
}

void NeuralNet::freeComputeContext(ComputeContext* computeContext) {
  delete computeContext;
}

//--------------------------------------------------------------

struct ComputeHandle {
  int nnXLen;
  int nnYLen;
  int maxBatchSize;
  int inputsUseNHWC;
  int gpuIndex;
  int version;

  ComputeHandle(ComputeContext* context,
                const LoadedModel* loadedModel,
                int maxBatchSize,
                int inputsUseNHWC,
                int gpuIdx,
                int serverThreadIdx) {
    const ModelDesc* modelDesc = &loadedModel->modelDesc;

    nnXLen = context->nnXLen;
    nnYLen = context->nnYLen;
    this->maxBatchSize = maxBatchSize;
    this->inputsUseNHWC = inputsUseNHWC;
    gpuIndex = gpuIdx;
    version = modelDesc->version;

    createMetalHandle(gpuIdx, modelDesc, maxBatchSize, serverThreadIdx);
  }

  ~ComputeHandle() {}

  void apply(
    float* userInputBuffer,
    float* userInputGlobalBuffer,
    float* policyOutput,
    float* valueOutput,
    float* ownershipOutput,
    float* miscValuesOutput,
    float* moreMiscValuesOutput) {

    getMetalHandleOutput(
      userInputBuffer,
      userInputGlobalBuffer,
      policyOutput,
      valueOutput,
      ownershipOutput,
      miscValuesOutput,
      moreMiscValuesOutput,
      gpuIndex);
  }

  ComputeHandle() = delete;
  ComputeHandle(const ComputeHandle&) = delete;
  ComputeHandle& operator=(const ComputeHandle&) = delete;
};

ComputeHandle* NeuralNet::createComputeHandle(
  ComputeContext* context,
  const LoadedModel* loadedModel,
  Logger* logger,
  int maxBatchSize,
  bool requireExactNNLen,
  bool inputsUseNHWC,
  int gpuIdxForThisThread,
  int serverThreadIdx) {
  auto deviceStr = [&]() {
    if(gpuIdxForThisThread < 0) {
      return string("");
    } else {
      return " Device " + Global::intToString(gpuIdxForThisThread);
    }
  };

  if(logger != NULL) {
    logger->write(
      "Metal backend thread " + Global::intToString(serverThreadIdx) + ":" + deviceStr() + " Model version " +
      Global::intToString(loadedModel->modelDesc.version));

    logger->write(
      "Metal backend thread " + Global::intToString(serverThreadIdx) + ":" + deviceStr() +
      " Model name: " + loadedModel->modelDesc.name);
  }

  // Current implementation always tolerates excess nn len
  (void)requireExactNNLen;
  ComputeHandle* handle = new ComputeHandle(context, loadedModel, maxBatchSize, inputsUseNHWC, gpuIdxForThisThread, serverThreadIdx);

  if(logger != NULL) {
    logger->write("Metal backend thread " + Global::intToString(serverThreadIdx) + ":" + deviceStr());
  }
  return handle;
}

void NeuralNet::freeComputeHandle(ComputeHandle* handle) {
  delete handle;
}

//------------------------------------------------------------------------------

void NeuralNet::printDevices() {
  MetalDevices* metalDevices = new MetalDevices();
  metalDevices->printDevices();
  delete metalDevices;
}

//--------------------------------------------------------------

struct InputBuffers {
  int maxBatchSize;
  size_t policyResultChannels;

  size_t singleInputElts;
  size_t singleInputGlobalElts;
  size_t singlePolicyResultElts;
  size_t singleValueResultElts;
  size_t singleOwnershipResultElts;
  size_t singleMiscValuesResultElts;
  size_t singleMoreMiscValuesResultElts;

  size_t userInputBufferElts;
  size_t userInputGlobalBufferElts;
  size_t policyResultBufferElts;
  size_t valueResultBufferElts;
  size_t ownershipResultBufferElts;
  size_t miscValuesResultBufferElts;
  size_t moreMiscValuesResultsBufferElts;

  float* userInputBuffer;        // Host pointer
  float* userInputGlobalBuffer;  // Host pointer

  float* policyResults;
  float* valueResults;
  float* ownershipResults;
  float* miscValuesResults;
  float* moreMiscValuesResults;

  InputBuffers(const LoadedModel* loadedModel, int maxBatchSz, int nnXLen, int nnYLen) {
    const ModelDesc& m = loadedModel->modelDesc;

    int xSize = nnXLen;
    int ySize = nnYLen;

    maxBatchSize = maxBatchSz;
    policyResultChannels = 2;
    singleInputElts = (size_t)m.numInputChannels * xSize * ySize;
    singleInputGlobalElts = (size_t)m.numInputGlobalChannels;
    singlePolicyResultElts = (size_t)((xSize * ySize) + 1);
    singleValueResultElts = (size_t)m.numValueChannels;
    singleOwnershipResultElts = (size_t)m.numOwnershipChannels * xSize * ySize;
    singleMiscValuesResultElts = 10;
    singleMoreMiscValuesResultElts = 8;

    assert(NNModelVersion::getNumSpatialFeatures(m.version) == m.numInputChannels);
    assert(NNModelVersion::getNumGlobalFeatures(m.version) == m.numInputGlobalChannels);
    assert(singleInputElts == (361 * 22));
    assert(singleInputGlobalElts == 19);
    assert(singlePolicyResultElts == 362);
    assert(singleValueResultElts == 3);
    assert(singleOwnershipResultElts == 361);

    // swa_model_bin_inputs shape: [1, 361, 22]
    userInputBufferElts = (size_t)maxBatchSize * singleInputElts;

    // swa_model_global_inputs shape: [1, 19]
    userInputGlobalBufferElts = (size_t)maxBatchSize * singleInputGlobalElts;

    // swa_model_policy_output shape: [1, 362, 2]
    policyResultBufferElts = (size_t)maxBatchSize * singlePolicyResultElts * policyResultChannels;

    // swa_model_value_output shape: [1, 3]
    valueResultBufferElts = (size_t)maxBatchSize * singleValueResultElts;

    // swa_model_ownership_output shape: [1, 19, 19]
    ownershipResultBufferElts = (size_t)maxBatchSize * singleOwnershipResultElts;

    // swa_model_miscvalues_output shape: [1, 10]
    miscValuesResultBufferElts = (size_t)maxBatchSize * singleMiscValuesResultElts;

    // swa_model_moremiscvalues_output shape: [1, 8]
    moreMiscValuesResultsBufferElts = (size_t)maxBatchSize * singleMoreMiscValuesResultElts;

    userInputBuffer = new float[userInputBufferElts];
    userInputGlobalBuffer = new float[userInputGlobalBufferElts];
    policyResults = new float[policyResultBufferElts];
    valueResults = new float[valueResultBufferElts];
    ownershipResults = new float[ownershipResultBufferElts];
    miscValuesResults = new float[miscValuesResultBufferElts];
    moreMiscValuesResults = new float[moreMiscValuesResultsBufferElts];
  }

  ~InputBuffers() {
    delete[] userInputBuffer;
    delete[] userInputGlobalBuffer;
    delete[] policyResults;
    delete[] valueResults;
    delete[] ownershipResults;
    delete[] miscValuesResults;
    delete[] moreMiscValuesResults;
  }

  InputBuffers() = delete;
  InputBuffers(const InputBuffers&) = delete;
  InputBuffers& operator=(const InputBuffers&) = delete;
};

InputBuffers* NeuralNet::createInputBuffers(const LoadedModel* loadedModel, int maxBatchSize, int nnXLen, int nnYLen) {
  return new InputBuffers(loadedModel, maxBatchSize, nnXLen, nnYLen);
}

void NeuralNet::freeInputBuffers(InputBuffers* inputBuffers) {
  delete inputBuffers;
}

void NeuralNet::getOutput(
  ComputeHandle* gpuHandle,
  InputBuffers* inputBuffers,
  int numBatchEltsFilled,
  NNResultBuf** inputBufs,
  vector<NNOutput*>& outputs) {

  int batchSize = numBatchEltsFilled;
  int nnXLen = gpuHandle->nnXLen;
  int nnYLen = gpuHandle->nnYLen;
  int version = gpuHandle->version;
  int numSpatialFeatures = NNModelVersion::getNumSpatialFeatures(version);
  int numGlobalFeatures = NNModelVersion::getNumGlobalFeatures(version);

  assert(batchSize <= inputBuffers->maxBatchSize);
  assert(batchSize > 0);
  assert((numSpatialFeatures * nnXLen * nnYLen) == inputBuffers->singleInputElts);
  assert(numGlobalFeatures == inputBuffers->singleInputGlobalElts);

  size_t policyResultChannels = inputBuffers->policyResultChannels;
  size_t singleInputElts = inputBuffers->singleInputElts;
  size_t singleInputGlobalElts = inputBuffers->singleInputGlobalElts;
  size_t singlePolicyResultElts = inputBuffers->singlePolicyResultElts;
  size_t singleValueResultElts = inputBuffers->singleValueResultElts;
  size_t singleOwnershipResultElts = inputBuffers->singleOwnershipResultElts;
  size_t singleMiscValuesResultElts = inputBuffers->singleMiscValuesResultElts;
  size_t singleMoreMiscValuesResultElts = inputBuffers->singleMoreMiscValuesResultElts;

  assert(policyResultChannels == 2);
  assert(singleInputElts == (361 * 22));
  assert(singleInputGlobalElts == 19);
  assert(singlePolicyResultElts == 362);
  assert(singleValueResultElts == 3);
  assert(singleOwnershipResultElts == 361);
  assert(singleMiscValuesResultElts == 10);
  assert(singleMoreMiscValuesResultElts == 8);

  for(size_t row = 0; row < batchSize; row++) {
    float* rowSpatialInput = &inputBuffers->userInputBuffer[singleInputElts * row];
    float* rowGlobalInput = &inputBuffers->userInputGlobalBuffer[singleInputGlobalElts * row];
    float* policyOutputBuf = &inputBuffers->policyResults[row * (singlePolicyResultElts * policyResultChannels)];
    float* valueOutputBuf = &inputBuffers->valueResults[row * singleValueResultElts];
    float* ownershipOutputBuf = &inputBuffers->ownershipResults[row * singleOwnershipResultElts];
    float* miscValuesOutputBuf = &inputBuffers->miscValuesResults[row * singleMiscValuesResultElts];
    float* moreMiscValuesOutputBuf = &inputBuffers->moreMiscValuesResults[row * singleMoreMiscValuesResultElts];

    const float* rowGlobal = inputBufs[row]->rowGlobal;
    const float* rowSpatial = inputBufs[row]->rowSpatial;

    copy(&rowGlobal[0], &rowGlobal[numGlobalFeatures], rowGlobalInput);

    assert(gpuHandle->inputsUseNHWC == false);

    SymmetryHelpers::copyInputsWithSymmetry(
      rowSpatial,
      rowSpatialInput,
      1,
      nnYLen,
      nnXLen,
      numSpatialFeatures,
      gpuHandle->inputsUseNHWC,
      inputBufs[row]->symmetry);

    gpuHandle->apply(
      rowSpatialInput,
      rowGlobalInput,
      policyOutputBuf,
      valueOutputBuf,
      ownershipOutputBuf,
      miscValuesOutputBuf,
      moreMiscValuesOutputBuf);
  }

  for(size_t row = 0; row < batchSize; row++) {
    NNOutput* output = outputs[row];

    assert(output->nnXLen == nnXLen);
    assert(output->nnYLen == nnYLen);

    float* policyOutputBuf = &inputBuffers->policyResults[row * (singlePolicyResultElts * policyResultChannels)];

    // Extract policy0_output
    for(size_t i = 0; i < singlePolicyResultElts; i++) {
      policyOutputBuf[i] = policyOutputBuf[i * policyResultChannels];
    }

    // These are not actually correct, the client does the postprocessing to turn them into
    // policy probabilities and white game outcome probabilities
    // Also we don't fill in the nnHash here either
    SymmetryHelpers::copyOutputsWithSymmetry(
      policyOutputBuf, output->policyProbs, 1, nnYLen, nnXLen, inputBufs[row]->symmetry);

    output->policyProbs[singlePolicyResultElts - 1] = policyOutputBuf[singlePolicyResultElts - 1];

    const float* valueOutputBuf = &inputBuffers->valueResults[row * singleValueResultElts];

    output->whiteWinProb = valueOutputBuf[0];
    output->whiteLossProb = valueOutputBuf[1];
    output->whiteNoResultProb = valueOutputBuf[2];

    if(output->whiteOwnerMap != NULL) {
      const float* ownershipOutputBuf = &inputBuffers->ownershipResults[row * singleOwnershipResultElts];

      SymmetryHelpers::copyOutputsWithSymmetry(
        ownershipOutputBuf, output->whiteOwnerMap, 1, nnYLen, nnXLen, inputBufs[row]->symmetry);
    }

    const float* miscValuesOutputBuf = &inputBuffers->miscValuesResults[row * singleMiscValuesResultElts];
    const float* moreMiscValuesOutputBuf = &inputBuffers->moreMiscValuesResults[row * singleMoreMiscValuesResultElts];

    if(version >= 9) {
      output->whiteScoreMean = miscValuesOutputBuf[0];
      output->whiteScoreMeanSq = miscValuesOutputBuf[1];
      output->whiteLead = miscValuesOutputBuf[2];
      output->varTimeLeft = miscValuesOutputBuf[3];
      output->shorttermWinlossError = moreMiscValuesOutputBuf[0];
      output->shorttermScoreError = moreMiscValuesOutputBuf[1];
    } else if(version >= 8) {
      output->whiteScoreMean = miscValuesOutputBuf[0];
      output->whiteScoreMeanSq = miscValuesOutputBuf[1];
      output->whiteLead = miscValuesOutputBuf[2];
      output->varTimeLeft = miscValuesOutputBuf[3];
      output->shorttermWinlossError = 0;
      output->shorttermScoreError = 0;
    } else if(version >= 4) {
      output->whiteScoreMean = miscValuesOutputBuf[0];
      output->whiteScoreMeanSq = miscValuesOutputBuf[1];
      output->whiteLead = output->whiteScoreMean;
      output->varTimeLeft = 0;
      output->shorttermWinlossError = 0;
      output->shorttermScoreError = 0;
    } else {
      assert(version >= 3);
      output->whiteScoreMean = miscValuesOutputBuf[0];
      // Version 3 neural nets don't have any second moment output, implicitly already folding it in, so we just use the
      // mean squared
      output->whiteScoreMeanSq = output->whiteScoreMean * output->whiteScoreMean;
      output->whiteLead = output->whiteScoreMean;
      output->varTimeLeft = 0;
      output->shorttermWinlossError = 0;
      output->shorttermScoreError = 0;
    }
  }
}

bool NeuralNet::testEvaluateConv(
  const ConvLayerDesc* desc,
  int batchSize,
  int nnXLen,
  int nnYLen,
  bool useFP16,
  bool useNHWC,
  const vector<float>& inputBuffer,
  vector<float>& outputBuffer) {

  size_t numOutputFloats = (size_t)batchSize * nnXLen * nnYLen * desc->outChannels;
  outputBuffer.resize(numOutputFloats);

  testMetalEvaluateConv(desc,
                        nnXLen,
                        nnYLen,
                        batchSize,
                        useFP16,
                        useNHWC,
                        (float*)inputBuffer.data(),
                        (float*)outputBuffer.data());
  return true;
}

// Mask should be in 'NHW' format (no "C" channel).
bool NeuralNet::testEvaluateBatchNorm(
  const BatchNormLayerDesc* desc,
  int batchSize,
  int nnXLen,
  int nnYLen,
  bool useFP16,
  bool useNHWC,
  const vector<float>& inputBuffer,
  const vector<float>& maskBuffer,
  vector<float>& outputBuffer) {

  size_t numOutputFloats = (size_t)batchSize * nnXLen * nnYLen * desc->numChannels;
  outputBuffer.resize(numOutputFloats);

  testMetalEvaluateBatchNorm(desc,
                             nnXLen,
                             nnYLen,
                             batchSize,
                             useFP16,
                             useNHWC,
                             (float*)inputBuffer.data(),
                             (float*)maskBuffer.data(),
                             (float*)outputBuffer.data());
  return true;
}

bool NeuralNet::testEvaluateResidualBlock(
  const ResidualBlockDesc* desc,
  int batchSize,
  int nnXLen,
  int nnYLen,
  bool useFP16,
  bool useNHWC,
  const vector<float>& inputBuffer,
  const vector<float>& maskBuffer,
  vector<float>& outputBuffer) {

  size_t numOutputFloats = (size_t)batchSize * nnXLen * nnYLen * desc->finalConv.outChannels;
  outputBuffer.resize(numOutputFloats);

  testMetalEvaluateResidualBlock(desc,
                                 batchSize,
                                 nnXLen,
                                 nnYLen,
                                 useFP16,
                                 useNHWC,
                                 (float*)inputBuffer.data(),
                                 (float*)maskBuffer.data(),
                                 (float*)outputBuffer.data());
  return true;
}

bool NeuralNet::testEvaluateGlobalPoolingResidualBlock(
  const GlobalPoolingResidualBlockDesc* desc,
  int batchSize,
  int nnXLen,
  int nnYLen,
  bool useFP16,
  bool useNHWC,
  const vector<float>& inputBuffer,
  const vector<float>& maskBuffer,
  vector<float>& outputBuffer) {

  size_t numOutputFloats = (size_t)batchSize * nnXLen * nnYLen * desc->finalConv.outChannels;
  outputBuffer.resize(numOutputFloats);

  testMetalEvaluateGlobalPoolingResidualBlock(desc,
                                              batchSize,
                                              nnXLen,
                                              nnYLen,
                                              useFP16,
                                              useNHWC,
                                              (float*)inputBuffer.data(),
                                              (float*)maskBuffer.data(),
                                              (float*)outputBuffer.data());
  return true;
}

#endif  // USE_METAL_BACKEND
