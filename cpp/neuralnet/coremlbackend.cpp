#ifdef USE_COREML_BACKEND

#include "../neuralnet/coremlbackend.h"
#include "../neuralnet/modelversion.h"
#include "../neuralnet/nneval.h"
#include "../neuralnet/nninputs.h"
#include "../neuralnet/nninterface.h"

using namespace std;

//---------------------------------------------------------------------------------------------------------

void NeuralNet::globalInitialize() {
  initCoreMLBackends();
}

void NeuralNet::globalCleanup() {}

//------------------------------------------------------------------------------

struct LoadedModel {
  int modelXLen;
  int modelYLen;
  ModelDesc modelDesc;

  LoadedModel() {
    modelXLen = COMPILE_MAX_BOARD_LEN;
    modelYLen = COMPILE_MAX_BOARD_LEN;
    modelDesc.name = "CoreML model";
    modelDesc.version = createCoreMLBackend(0, COMPILE_MAX_BOARD_LEN, COMPILE_MAX_BOARD_LEN);
    modelDesc.numInputChannels = 22;
    modelDesc.numInputGlobalChannels = 19;
    modelDesc.numValueChannels = 3;
    modelDesc.numOwnershipChannels = 1;
    modelDesc.numScoreValueChannels = 18;
  }

  LoadedModel(const LoadedModel&) = delete;
  LoadedModel& operator=(const LoadedModel&) = delete;
};

LoadedModel* NeuralNet::loadModelFile(const string& file, const string& expectedSha256) {
  LoadedModel* loadedModel = new LoadedModel();
  (void)file;
  (void)expectedSha256;

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
  const std::vector<int>& gpuIdxs,
  Logger* logger,
  int nnXLen,
  int nnYLen,
  const string& openCLTunerFile,
  const string& homeDataDirOverride,
  bool openCLReTunePerBoardSize,
  enabled_t useFP16Mode,
  enabled_t useNHWCMode,
  const LoadedModel* loadedModel) {
  if(gpuIdxs.size() <= 0) {
    throw StringError("NeuralNet::createComputeContext - specified no gpus to use");
  }

  (void)logger;
  (void)openCLTunerFile;
  (void)homeDataDirOverride;
  (void)openCLReTunePerBoardSize;
  (void)useFP16Mode;
  (void)useNHWCMode;
  (void)loadedModel;

  return new ComputeContext(nnXLen, nnYLen);
}

void NeuralNet::freeComputeContext(ComputeContext* computeContext) {
  delete computeContext;
}

//--------------------------------------------------------------

struct ComputeHandle {
  int nnXLen;
  int nnYLen;
  int modelXLen;
  int modelYLen;
  bool inputsUseNHWC;
  int version;
  int gpuIndex;

  ComputeHandle(ComputeContext* context, const LoadedModel* loadedModel, int gpuIdx, bool inputsNHWC) {
    nnXLen = context->nnXLen;
    nnYLen = context->nnYLen;
    modelXLen = loadedModel->modelXLen;
    modelYLen = loadedModel->modelYLen;
    gpuIndex = gpuIdx;
    inputsUseNHWC = inputsNHWC;

    version = createCoreMLBackend(gpuIdx, loadedModel->modelXLen, loadedModel->modelYLen);
  }

  ~ComputeHandle() {
    freeCoreMLBackend(gpuIndex);
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

  // Current implementation always tolerates excess nn len
  (void)requireExactNNLen;
  ComputeHandle* handle = new ComputeHandle(context, loadedModel, gpuIdxForThisThread, inputsUseNHWC);

  if(logger != NULL) {
    logger->write("CoreML backend thread " + Global::intToString(serverThreadIdx) + ":" + deviceStr());
  }

  (void)maxBatchSize;

  return handle;
}

void NeuralNet::freeComputeHandle(ComputeHandle* handle) {
  delete handle;
}

//------------------------------------------------------------------------------

struct DeviceInfo {
  int gpuIdx;
  std::string name;
  int defaultDesirability;

  static std::vector<DeviceInfo> getAllDeviceInfosOnSystem();
};

//------------------------------------------------------------------------------

vector<DeviceInfo> DeviceInfo::getAllDeviceInfosOnSystem() {
  int numDevicesTotal = 2;
  vector<DeviceInfo> allDeviceInfos;

  for(int gpuIdx = 0; gpuIdx < numDevicesTotal; gpuIdx++) {
    DeviceInfo info;

    info.gpuIdx = gpuIdx;
    info.name = "KataGo CoreML package";
    info.defaultDesirability = 100;
    allDeviceInfos.push_back(info);
  }

  return allDeviceInfos;
}

//------------------------------------------------------------------------------

void NeuralNet::printDevices() {
  vector<DeviceInfo> devices = DeviceInfo::getAllDeviceInfosOnSystem();
  for(int i = 0; i < devices.size(); i++) {
    const DeviceInfo& device = devices[i];
    string msg = "Found CoreML Device " + Global::intToString(device.gpuIdx) + ": " + device.name + " (score " +
                 Global::intToString(device.defaultDesirability) + ")";
    cout << msg << endl;
  }
}

//--------------------------------------------------------------

struct InputBuffers {
  int maxBatchSize;
  int modelXLen;
  int modelYLen;

  size_t policyResultChannels;

  size_t singleSpatialElts;
  size_t singleInputElts;
  size_t singleInputGlobalElts;
  size_t singlePolicyResultElts;
  size_t singlePolicyProbsElts;
  size_t singleValueResultElts;
  size_t singleOwnershipResultElts;
  size_t singleOwnerMapElts;
  size_t singleMiscValuesResultElts;
  size_t singleMoreMiscValuesResultElts;

  size_t rowSpatialBufferElts;
  size_t userInputBufferElts;
  size_t userInputGlobalBufferElts;
  size_t policyResultBufferElts;
  size_t policyProbsBufferElts;
  size_t valueResultBufferElts;
  size_t ownershipResultBufferElts;
  size_t ownerMapBufferElts;
  size_t miscValuesResultBufferElts;
  size_t moreMiscValuesResultsBufferElts;

  float* rowSpatialBuffer;
  float* userInputBuffer;        // Host pointer
  float* userInputGlobalBuffer;  // Host pointer

  float* policyResults;
  float* policyProbsBuffer;
  float* valueResults;
  float* ownershipResults;
  float* ownerMapBuffer;
  float* miscValuesResults;
  float* moreMiscValuesResults;

  InputBuffers(const LoadedModel* loadedModel, int maxBatchSz, int nnXLen, int nnYLen) {
    const ModelDesc& m = loadedModel->modelDesc;

    modelXLen = COMPILE_MAX_BOARD_LEN;
    modelYLen = COMPILE_MAX_BOARD_LEN;
    maxBatchSize = maxBatchSz;
    policyResultChannels = 2;
    singleSpatialElts = (size_t)m.numInputChannels * nnXLen * nnYLen;
    singleInputElts = (size_t)m.numInputChannels * modelXLen * modelYLen;
    singleInputGlobalElts = (size_t)m.numInputGlobalChannels;
    singlePolicyResultElts = (size_t)((modelXLen * modelYLen) + 1);
    singlePolicyProbsElts = (size_t)((nnXLen * nnYLen) + 1);
    singleValueResultElts = (size_t)m.numValueChannels;
    singleOwnershipResultElts = (size_t)m.numOwnershipChannels * modelXLen * modelYLen;
    singleOwnerMapElts = (size_t)m.numOwnershipChannels * nnXLen * nnYLen;
    singleMiscValuesResultElts = 10;
    singleMoreMiscValuesResultElts = 8;

    assert(NNModelVersion::getNumSpatialFeatures(m.version) == m.numInputChannels);
    assert(NNModelVersion::getNumGlobalFeatures(m.version) == m.numInputGlobalChannels);
    assert(singleInputElts == (modelXLen * modelYLen * 22));
    assert(singleInputGlobalElts == 19);
    assert(singleValueResultElts == 3);
    assert(singleOwnershipResultElts == (modelXLen * modelYLen));

    rowSpatialBufferElts = (size_t)maxBatchSize * singleSpatialElts;

    // swa_model_bin_inputs shape: [1, 361, 22]
    userInputBufferElts = (size_t)maxBatchSize * singleInputElts;

    // swa_model_global_inputs shape: [1, 19]
    userInputGlobalBufferElts = (size_t)maxBatchSize * singleInputGlobalElts;

    // swa_model_policy_output shape: [1, 362, 2]
    policyResultBufferElts = (size_t)maxBatchSize * singlePolicyResultElts * policyResultChannels;

    policyProbsBufferElts = (size_t)maxBatchSize * singlePolicyProbsElts;

    // swa_model_value_output shape: [1, 3]
    valueResultBufferElts = (size_t)maxBatchSize * singleValueResultElts;

    // swa_model_ownership_output shape: [1, 19, 19]
    ownershipResultBufferElts = (size_t)maxBatchSize * singleOwnershipResultElts;

    ownerMapBufferElts = (size_t)maxBatchSize * singleOwnerMapElts;

    // swa_model_miscvalues_output shape: [1, 10]
    miscValuesResultBufferElts = (size_t)maxBatchSize * singleMiscValuesResultElts;

    // swa_model_moremiscvalues_output shape: [1, 8]
    moreMiscValuesResultsBufferElts = (size_t)maxBatchSize * singleMoreMiscValuesResultElts;

    rowSpatialBuffer = new float[rowSpatialBufferElts];
    userInputBuffer = new float[userInputBufferElts];
    userInputGlobalBuffer = new float[userInputGlobalBufferElts];
    policyResults = new float[policyResultBufferElts];
    policyProbsBuffer = new float[policyProbsBufferElts];
    valueResults = new float[valueResultBufferElts];
    ownershipResults = new float[ownershipResultBufferElts];
    ownerMapBuffer = new float[ownerMapBufferElts];
    miscValuesResults = new float[miscValuesResultBufferElts];
    moreMiscValuesResults = new float[moreMiscValuesResultsBufferElts];

    memset(&userInputBuffer[0], 0, userInputBufferElts * sizeof(userInputBuffer[0]));
  }

  ~InputBuffers() {
    delete[] rowSpatialBuffer;
    delete[] userInputBuffer;
    delete[] userInputGlobalBuffer;
    delete[] policyResults;
    delete[] policyProbsBuffer;
    delete[] valueResults;
    delete[] ownershipResults;
    delete[] ownerMapBuffer;
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
  int modelXLen = gpuHandle->modelXLen;
  int modelYLen = gpuHandle->modelYLen;
  int version = gpuHandle->version;
  int numSpatialFeatures = NNModelVersion::getNumSpatialFeatures(version);
  int numGlobalFeatures = NNModelVersion::getNumGlobalFeatures(version);

  assert(batchSize <= inputBuffers->maxBatchSize);
  assert(batchSize > 0);
  assert((numSpatialFeatures * modelXLen * modelYLen) == inputBuffers->singleInputElts);
  assert(numGlobalFeatures == inputBuffers->singleInputGlobalElts);

  size_t policyResultChannels = inputBuffers->policyResultChannels;
  size_t singleSpatialElts = inputBuffers->singleSpatialElts;
  size_t singleInputElts = inputBuffers->singleInputElts;
  size_t singleInputGlobalElts = inputBuffers->singleInputGlobalElts;
  size_t singlePolicyResultElts = inputBuffers->singlePolicyResultElts;
  size_t singlePolicyProbsElts = inputBuffers->singlePolicyProbsElts;
  size_t singleValueResultElts = inputBuffers->singleValueResultElts;
  size_t singleOwnershipResultElts = inputBuffers->singleOwnershipResultElts;
  size_t singleOwnerMapElts = inputBuffers->singleOwnerMapElts;
  size_t singleMiscValuesResultElts = inputBuffers->singleMiscValuesResultElts;
  size_t singleMoreMiscValuesResultElts = inputBuffers->singleMoreMiscValuesResultElts;

  assert(policyResultChannels == 2);
  assert(singleInputElts == (modelXLen * modelYLen * 22));
  assert(singleInputGlobalElts == 19);
  assert(singlePolicyResultElts == ((modelXLen * modelYLen) + 1));
  assert(singleValueResultElts == 3);
  assert(singleOwnershipResultElts == (modelXLen * modelYLen));
  assert(singleMiscValuesResultElts == 10);
  assert(singleMoreMiscValuesResultElts == 8);

  // Get CoreML backend output
  for(size_t row = 0; row < batchSize; row++) {
    float* rowSpatialBuffer = &inputBuffers->rowSpatialBuffer[singleSpatialElts * row];
    float* rowSpatialInput = &inputBuffers->userInputBuffer[singleInputElts * row];
    float* rowGlobalInput = &inputBuffers->userInputGlobalBuffer[singleInputGlobalElts * row];
    float* policyOutputBuf = &inputBuffers->policyResults[row * (singlePolicyResultElts * policyResultChannels)];
    float* valueOutputBuf = &inputBuffers->valueResults[row * singleValueResultElts];
    float* ownershipOutputBuf = &inputBuffers->ownershipResults[row * singleOwnershipResultElts];
    float* miscValuesOutputBuf = &inputBuffers->miscValuesResults[row * singleMiscValuesResultElts];
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

    getCoreMLBackendOutput(
      rowSpatialInput,
      rowGlobalInput,
      policyOutputBuf,
      valueOutputBuf,
      ownershipOutputBuf,
      miscValuesOutputBuf,
      moreMiscValuesOutputBuf,
      gpuHandle->gpuIndex);
  }

  // Fill results by CoreML model output
  for(size_t row = 0; row < batchSize; row++) {
    NNOutput* output = outputs[row];
    assert(output->nnXLen == nnXLen);
    assert(output->nnYLen == nnYLen);

    float* policyOutputBuf = &inputBuffers->policyResults[row * (singlePolicyResultElts * policyResultChannels)];
    float* policyProbsBuf = &inputBuffers->policyProbsBuffer[row * singlePolicyProbsElts];

    // Extract policy0_output
    for(size_t i = 0; i < singlePolicyResultElts; i++) {
      policyOutputBuf[i] = policyOutputBuf[i * policyResultChannels];
    }

    for(int y = 0; y < nnYLen; y++) {
      for(int x = 0; x < nnXLen; x++) {
        int outputIdx = (y * modelXLen) + x;
        int probsIdx = (y * nnXLen) + x;
        policyProbsBuf[probsIdx] = policyOutputBuf[outputIdx];
      }
    }

    // These are not actually correct, the client does the postprocessing to turn them into
    // policy probabilities and white game outcome probabilities
    // Also we don't fill in the nnHash here either
    SymmetryHelpers::copyOutputsWithSymmetry(
      policyProbsBuf, output->policyProbs, 1, nnYLen, nnXLen, inputBufs[row]->symmetry);

    output->policyProbs[singlePolicyProbsElts - 1] = policyOutputBuf[singlePolicyResultElts - 1];

    const float* valueOutputBuf = &inputBuffers->valueResults[row * singleValueResultElts];

    output->whiteWinProb = valueOutputBuf[0];
    output->whiteLossProb = valueOutputBuf[1];
    output->whiteNoResultProb = valueOutputBuf[2];

    if(output->whiteOwnerMap != NULL) {
      const float* ownershipOutputBuf = &inputBuffers->ownershipResults[row * singleOwnershipResultElts];
      float* ownerMapBuf = &inputBuffers->ownerMapBuffer[row * singleOwnerMapElts];

      for(int y = 0; y < nnYLen; y++) {
        for(int x = 0; x < nnXLen; x++) {
          int outputIdx = (y * modelXLen) + x;
          int ownerMapIdx = (y * nnXLen) + x;
          ownerMapBuf[ownerMapIdx] = ownershipOutputBuf[outputIdx];
        }
      }

      SymmetryHelpers::copyOutputsWithSymmetry(
        ownerMapBuf, output->whiteOwnerMap, 1, nnYLen, nnXLen, inputBufs[row]->symmetry);
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
    } else if(version >= 3) {
      output->whiteScoreMean = miscValuesOutputBuf[0];
      // Version 3 neural nets don't have any second moment output, implicitly already folding it in, so we just use the
      // mean squared
      output->whiteScoreMeanSq = output->whiteScoreMean * output->whiteScoreMean;
      output->whiteLead = output->whiteScoreMean;
      output->varTimeLeft = 0;
      output->shorttermWinlossError = 0;
      output->shorttermScoreError = 0;
    } else {
      ASSERT_UNREACHABLE;
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
  const std::vector<float>& inputBuffer,
  std::vector<float>& outputBuffer) {
  (void)desc;
  (void)batchSize;
  (void)nnXLen;
  (void)nnYLen;
  (void)useFP16;
  (void)useNHWC;
  (void)inputBuffer;
  (void)outputBuffer;
  return false;
}

bool NeuralNet::testEvaluateBatchNorm(
  const BatchNormLayerDesc* desc,
  int batchSize,
  int nnXLen,
  int nnYLen,
  bool useFP16,
  bool useNHWC,
  const std::vector<float>& inputBuffer,
  const std::vector<float>& maskBuffer,
  std::vector<float>& outputBuffer) {
  (void)desc;
  (void)batchSize;
  (void)nnXLen;
  (void)nnYLen;
  (void)useFP16;
  (void)useNHWC;
  (void)inputBuffer;
  (void)maskBuffer;
  (void)outputBuffer;
  return false;
}

bool NeuralNet::testEvaluateResidualBlock(
  const ResidualBlockDesc* desc,
  int batchSize,
  int nnXLen,
  int nnYLen,
  bool useFP16,
  bool useNHWC,
  const std::vector<float>& inputBuffer,
  const std::vector<float>& maskBuffer,
  std::vector<float>& outputBuffer) {
  (void)desc;
  (void)batchSize;
  (void)nnXLen;
  (void)nnYLen;
  (void)useFP16;
  (void)useNHWC;
  (void)inputBuffer;
  (void)maskBuffer;
  (void)outputBuffer;
  return false;
}

bool NeuralNet::testEvaluateGlobalPoolingResidualBlock(
  const GlobalPoolingResidualBlockDesc* desc,
  int batchSize,
  int nnXLen,
  int nnYLen,
  bool useFP16,
  bool useNHWC,
  const std::vector<float>& inputBuffer,
  const std::vector<float>& maskBuffer,
  std::vector<float>& outputBuffer) {
  (void)desc;
  (void)batchSize;
  (void)nnXLen;
  (void)nnYLen;
  (void)useFP16;
  (void)useNHWC;
  (void)inputBuffer;
  (void)maskBuffer;
  (void)outputBuffer;
  return false;
}

#endif  // USE_COREML_BACKEND
