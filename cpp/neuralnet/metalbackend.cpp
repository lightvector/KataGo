#ifdef USE_COREML_BACKEND

#include "../neuralnet/modelversion.h"
#include "../neuralnet/nneval.h"
#include "../neuralnet/nninputs.h"
#include "../neuralnet/nninterface.h"
#include "../neuralnet/metalbackend.h"
#include "../neuralnet/coremlbackend.h"

using namespace std;

//---------------------------------------------------------------------------------------------------------

void NeuralNet::globalInitialize() {
  initCoreMLBackends();
}

void NeuralNet::globalCleanup() {
  // Do nothing, calling this is okay even if there is no neural net
  // as long as we don't attempt to actually load a net file and use one.
}

//------------------------------------------------------------------------------

struct LoadedModel {
  ModelDesc modelDesc;
  CoreMLLoadedModel coreMLLoadedModel;

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
  enabled_t useFP16Mode;

  ComputeContext(int nnX, int nnY, enabled_t useFP16Mode, enabled_t useNHWCMode) {
    this->useFP16Mode = useFP16Mode;
    createMetalContext(nnX, nnY, useFP16Mode, useNHWCMode);
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

  return new ComputeContext(nnXLen, nnYLen, useFP16Mode, useNHWCMode);
}

void NeuralNet::freeComputeContext(ComputeContext* computeContext) {
  delete computeContext;
}

//--------------------------------------------------------------

struct ComputeHandle {
  int nnXLen;
  int nnYLen;
  bool inputsUseNHWC;
  int gpuIndex;
  int version;
  CoreMLComputeHandle* coreMLComputeHandle = NULL;

  ComputeHandle(ComputeContext* context,
                const LoadedModel* loadedModel,
                int maxBatchSize,
                bool inputsUseNHWC,
                int gpuIdx,
                int serverThreadIdx) {
    const ModelDesc* modelDesc = &loadedModel->modelDesc;

    nnXLen = getMetalContextXLen();
    nnYLen = getMetalContextYLen();
    this->inputsUseNHWC = inputsUseNHWC;
    gpuIndex = gpuIdx;
    version = modelDesc->version;

    /* Use FP16 mode if the model supports it and the user has not explicitly
     * disabled it. */
    bool useFP16 = context->useFP16Mode != enabled_t::False;

    coreMLComputeHandle = new CoreMLComputeHandle(&loadedModel->coreMLLoadedModel,
                                                  nnXLen,
                                                  nnYLen,
                                                  gpuIdx,
                                                  inputsUseNHWC,
                                                  serverThreadIdx,
                                                  useFP16);

    if(!(coreMLComputeHandle->isCoreML)) {
      createMetalHandle(gpuIdx, modelDesc, maxBatchSize, serverThreadIdx);
    }
  }

  ~ComputeHandle() {
    freeCoreMLBackend(gpuIndex);

    if(coreMLComputeHandle != NULL) {
      delete coreMLComputeHandle;
    }
  }

  void apply(float* userInputBuffer,
             float* userInputGlobalBuffer,
             float* policyOutput,
             float* policyPassOutput,
             float* valueOutput,
             float* ownershipOutput,
             float* scoreValueOutput) {

    getMetalHandleOutput(userInputBuffer,
                         userInputGlobalBuffer,
                         policyOutput,
                         policyPassOutput,
                         valueOutput,
                         ownershipOutput,
                         scoreValueOutput,
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

  // Current implementation always tolerates excess nn len
  (void)requireExactNNLen;
  ComputeHandle* handle = new ComputeHandle(context, loadedModel, 1, inputsUseNHWC, gpuIdxForThisThread, serverThreadIdx);

  return handle;
}

void NeuralNet::freeComputeHandle(ComputeHandle* handle) {
  delete handle;
}

//------------------------------------------------------------------------------

void NeuralNet::printDevices() {
  printMetalDevices();
}

//--------------------------------------------------------------

struct InputBuffers {
  int maxBatchSize;
  size_t policyResultChannels;

  size_t singleInputElts;
  size_t singleInputGlobalElts;
  size_t singlePolicyResultElts;
  size_t singlePolicyPassResultElts;
  size_t singleValueResultElts;
  size_t singleOwnershipResultElts;
  size_t singleScoreValuesResultElts;

  size_t userInputBufferElts;
  size_t userInputGlobalBufferElts;
  size_t policyResultBufferElts;
  size_t policyPassResultBufferElts;
  size_t valueResultBufferElts;
  size_t ownershipResultBufferElts;
  size_t scoreValuesResultBufferElts;

  float* userInputBuffer;        // Host pointer
  float* userInputGlobalBuffer;  // Host pointer

  float* policyResults;
  float* policyPassResults;
  float* valueResults;
  float* ownershipResults;
  float* scoreValuesResults;

  CoreMLInputBuffers* coreMLInputBuffers;

  InputBuffers(const LoadedModel* loadedModel, int maxBatchSz, int nnXLen, int nnYLen) {
    const ModelDesc& m = loadedModel->modelDesc;

    int xSize = nnXLen;
    int ySize = nnYLen;

    maxBatchSize = maxBatchSz;
    policyResultChannels = 1;
    singleInputElts = (size_t)m.numInputChannels * xSize * ySize;
    singleInputGlobalElts = (size_t)m.numInputGlobalChannels;
    singlePolicyResultElts = (size_t)(xSize * ySize);
    singlePolicyPassResultElts = (size_t)1;
    singleValueResultElts = (size_t)m.numValueChannels;
    singleOwnershipResultElts = (size_t)m.numOwnershipChannels * xSize * ySize;
    singleScoreValuesResultElts = 6;

    assert(NNModelVersion::getNumSpatialFeatures(m.version) == m.numInputChannels);
    assert(NNModelVersion::getNumGlobalFeatures(m.version) == m.numInputGlobalChannels);
    assert(singleValueResultElts == 3);

    userInputBufferElts = (size_t)maxBatchSize * singleInputElts;
    userInputGlobalBufferElts = (size_t)maxBatchSize * singleInputGlobalElts;
    policyResultBufferElts = (size_t)maxBatchSize * singlePolicyResultElts * policyResultChannels;
    policyPassResultBufferElts = (size_t)maxBatchSize * singlePolicyPassResultElts;
    valueResultBufferElts = (size_t)maxBatchSize * singleValueResultElts;
    ownershipResultBufferElts = (size_t)maxBatchSize * singleOwnershipResultElts;
    scoreValuesResultBufferElts = (size_t)maxBatchSize * singleScoreValuesResultElts;

    userInputBuffer = new float[userInputBufferElts];
    userInputGlobalBuffer = new float[userInputGlobalBufferElts];
    policyResults = new float[policyResultBufferElts];
    policyPassResults = new float[policyPassResultBufferElts];
    valueResults = new float[valueResultBufferElts];
    ownershipResults = new float[ownershipResultBufferElts];
    scoreValuesResults = new float[scoreValuesResultBufferElts];
    coreMLInputBuffers = new CoreMLInputBuffers(&loadedModel->coreMLLoadedModel, maxBatchSize, nnXLen, nnYLen);
  }

  ~InputBuffers() {
    delete[] userInputBuffer;
    delete[] userInputGlobalBuffer;
    delete[] policyResults;
    delete[] policyPassResults;
    delete[] valueResults;
    delete[] ownershipResults;
    delete[] scoreValuesResults;
    delete coreMLInputBuffers;
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

void getMetalHandleOutput(
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
  size_t singlePolicyPassResultElts = inputBuffers->singlePolicyPassResultElts;
  size_t singleValueResultElts = inputBuffers->singleValueResultElts;
  size_t singleOwnershipResultElts = inputBuffers->singleOwnershipResultElts;
  size_t singleScoreValuesResultElts = inputBuffers->singleScoreValuesResultElts;

  assert(policyResultChannels == 1);
  assert(singleValueResultElts == 3);
  assert(singleScoreValuesResultElts == 6);

  for(size_t row = 0; row < batchSize; row++) {
    float* rowSpatialInput = &inputBuffers->userInputBuffer[singleInputElts * row];
    float* rowGlobalInput = &inputBuffers->userInputGlobalBuffer[singleInputGlobalElts * row];
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

    float* policyOutputBuf = &inputBuffers->policyResults[row * (singlePolicyResultElts * policyResultChannels)];
    float* policyPassOutputBuf = &inputBuffers->policyPassResults[row * singlePolicyPassResultElts];
    float* valueOutputBuf = &inputBuffers->valueResults[row * singleValueResultElts];
    float* ownershipOutputBuf = &inputBuffers->ownershipResults[row * singleOwnershipResultElts];
    float* scoreValuesOutputBuf = &inputBuffers->scoreValuesResults[row * singleScoreValuesResultElts];

    gpuHandle->apply(rowSpatialInput,
                     rowGlobalInput,
                     policyOutputBuf,
                     policyPassOutputBuf,
                     valueOutputBuf,
                     ownershipOutputBuf,
                     scoreValuesOutputBuf);
  }

  for(size_t row = 0; row < batchSize; row++) {
    NNOutput* output = outputs[row];

    assert(output->nnXLen == nnXLen);
    assert(output->nnYLen == nnYLen);

    float* policyOutputBuf = &inputBuffers->policyResults[row * (singlePolicyResultElts * policyResultChannels)];

    // These are not actually correct, the client does the postprocessing to turn them into
    // policy probabilities and white game outcome probabilities
    // Also we don't fill in the nnHash here either
    SymmetryHelpers::copyOutputsWithSymmetry(
      policyOutputBuf, output->policyProbs, 1, nnYLen, nnXLen, inputBufs[row]->symmetry);

    output->policyProbs[singlePolicyResultElts] = inputBuffers->policyPassResults[row * singlePolicyPassResultElts];

    const float* valueOutputBuf = &inputBuffers->valueResults[row * singleValueResultElts];

    output->whiteWinProb = valueOutputBuf[0];
    output->whiteLossProb = valueOutputBuf[1];
    output->whiteNoResultProb = valueOutputBuf[2];

    if(output->whiteOwnerMap != NULL) {
      const float* ownershipOutputBuf = &inputBuffers->ownershipResults[row * singleOwnershipResultElts];

      SymmetryHelpers::copyOutputsWithSymmetry(
        ownershipOutputBuf, output->whiteOwnerMap, 1, nnYLen, nnXLen, inputBufs[row]->symmetry);
    }

    const float* scoreValuesOutputBuf = &inputBuffers->scoreValuesResults[row * singleScoreValuesResultElts];

    if(version >= 9) {
      output->whiteScoreMean = scoreValuesOutputBuf[0];
      output->whiteScoreMeanSq = scoreValuesOutputBuf[1];
      output->whiteLead = scoreValuesOutputBuf[2];
      output->varTimeLeft = scoreValuesOutputBuf[3];
      output->shorttermWinlossError = scoreValuesOutputBuf[4];
      output->shorttermScoreError = scoreValuesOutputBuf[5];
    } else if(version >= 8) {
      output->whiteScoreMean = scoreValuesOutputBuf[0];
      output->whiteScoreMeanSq = scoreValuesOutputBuf[1];
      output->whiteLead = scoreValuesOutputBuf[2];
      output->varTimeLeft = scoreValuesOutputBuf[3];
      output->shorttermWinlossError = 0;
      output->shorttermScoreError = 0;
    } else if(version >= 4) {
      output->whiteScoreMean = scoreValuesOutputBuf[0];
      output->whiteScoreMeanSq = scoreValuesOutputBuf[1];
      output->whiteLead = output->whiteScoreMean;
      output->varTimeLeft = 0;
      output->shorttermWinlossError = 0;
      output->shorttermScoreError = 0;
    } else {
      assert(version >= 3);
      output->whiteScoreMean = scoreValuesOutputBuf[0];
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

void NeuralNet::getOutput(
  ComputeHandle* gpuHandle,
  InputBuffers* inputBuffers,
  int numBatchEltsFilled,
  NNResultBuf** inputBufs,
  vector<NNOutput*>& outputs) {

  if (gpuHandle->coreMLComputeHandle->isCoreML) {
    getCoreMLHandleOutput(gpuHandle->coreMLComputeHandle,
                          inputBuffers->coreMLInputBuffers,
                          numBatchEltsFilled,
                          inputBufs,
                          outputs);
  } else {
    getMetalHandleOutput(gpuHandle, inputBuffers, numBatchEltsFilled, inputBufs, outputs);
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

#endif  // USE_COREML_BACKEND
