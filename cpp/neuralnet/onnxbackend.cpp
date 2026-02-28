// ONNX Runtime backend for KataGo.
// Loads standard .bin.gz model files, builds an ONNX graph from ModelDesc at
// load time, and runs inference via ONNX Runtime with a configurable execution
// provider (CPU, CoreML, etc.) selected at runtime via the onnxProvider config key.

#include "../neuralnet/nninterface.h"
#include "../neuralnet/nneval.h"
#include "../neuralnet/nninputs.h"
#include "../neuralnet/modelversion.h"
#include "../neuralnet/onnxmodelbuilder.h"

#include <onnxruntime_cxx_api.h>
#ifdef __APPLE__
#include <coreml_provider_factory.h>
#endif

using namespace std;

//--------------------------------------------------------------

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
  return new LoadedModel(file, expectedSha256);
}

void NeuralNet::freeLoadedModel(LoadedModel* loadedModel) {
  delete loadedModel;
}

const ModelDesc& NeuralNet::getModelDesc(const LoadedModel* loadedModel) {
  return loadedModel->modelDesc;
}

//--------------------------------------------------------------

struct ComputeContext {
  Ort::Env env;
  int nnXLen;
  int nnYLen;
  string providerName;

  ComputeContext(int xLen, int yLen, const string& provider)
    : env(ORT_LOGGING_LEVEL_WARNING, "KataGoOnnx"),
      nnXLen(xLen),
      nnYLen(yLen),
      providerName(provider)
  {}
};

//--------------------------------------------------------------

struct ComputeHandle {
  ComputeContext* context;
  std::unique_ptr<Ort::Session> session;
  int modelVersion;
  int numInputChannels;
  int numInputGlobalChannels;
  int numPolicyChannels;
  int numValueChannels;
  int numScoreValueChannels;
  int numOwnershipChannels;
  int policyResultLen; // H*W+1

  // Input/output names (stored for session->Run)
  vector<string> inputNames;
  vector<string> outputNames;
  vector<const char*> inputNamePtrs;
  vector<const char*> outputNamePtrs;

  ComputeHandle(ComputeContext* ctx, const LoadedModel& loadedModel, Logger* logger)
    : context(ctx),
      modelVersion(loadedModel.modelDesc.modelVersion),
      numInputChannels(loadedModel.modelDesc.numInputChannels),
      numInputGlobalChannels(loadedModel.modelDesc.numInputGlobalChannels),
      numPolicyChannels(loadedModel.modelDesc.numPolicyChannels),
      numValueChannels(loadedModel.modelDesc.numValueChannels),
      numScoreValueChannels(loadedModel.modelDesc.numScoreValueChannels),
      numOwnershipChannels(loadedModel.modelDesc.numOwnershipChannels),
      policyResultLen(ctx->nnXLen * ctx->nnYLen + 1)
  {
    if(logger != NULL)
      logger->write("ONNX backend: building ONNX graph from model weights...");

    // Build ONNX model bytes from ModelDesc
    string onnxBytes = OnnxModelBuilder::buildOnnxModel(loadedModel.modelDesc, ctx->nnXLen, ctx->nnYLen);

    if(logger != NULL)
      logger->write("ONNX backend: ONNX graph built (" + Global::uint64ToString(onnxBytes.size()) + " bytes), creating session...");

    Ort::SessionOptions sessionOpts;
    sessionOpts.SetIntraOpNumThreads(1);

    // Select execution provider based on providerName
    const string& provider = ctx->providerName;
    if(provider == "coreml") {
#ifdef __APPLE__
      uint32_t coremlFlags = COREML_FLAG_CREATE_MLPROGRAM;
      Ort::ThrowOnError(OrtSessionOptionsAppendExecutionProvider_CoreML(sessionOpts, coremlFlags));
      if(logger != NULL)
        logger->write("ONNX backend: CoreML execution provider enabled (MLProgram mode)");
#else
      throw StringError("ONNX backend: CoreML is only available on Apple platforms");
#endif
    } else if(provider == "cpu" || provider.empty()) {
      if(logger != NULL)
        logger->write("ONNX backend: using CPU execution provider");
    } else {
      throw StringError("ONNX backend: unknown onnxProvider '" + provider + "', expected 'cpu' or 'coreml'");
    }

    // Create session from in-memory bytes
    session = std::make_unique<Ort::Session>(ctx->env, onnxBytes.data(), onnxBytes.size(), sessionOpts);

    // Query and store input names
    Ort::AllocatorWithDefaultOptions allocator;
    size_t numInputs = session->GetInputCount();
    for(size_t i = 0; i < numInputs; i++) {
      Ort::AllocatedStringPtr name = session->GetInputNameAllocated(i, allocator);
      inputNames.push_back(name.get());
    }
    for(auto& n : inputNames)
      inputNamePtrs.push_back(n.c_str());

    // Query and store output names
    size_t numOutputs = session->GetOutputCount();
    for(size_t i = 0; i < numOutputs; i++) {
      Ort::AllocatedStringPtr name = session->GetOutputNameAllocated(i, allocator);
      outputNames.push_back(name.get());
    }
    for(auto& n : outputNames)
      outputNamePtrs.push_back(n.c_str());

    if(logger != NULL)
      logger->write("ONNX backend: session created, inputs=" + Global::uint64ToString(numInputs) +
                     " outputs=" + Global::uint64ToString(numOutputs));
  }

  ComputeHandle() = delete;
  ComputeHandle(const ComputeHandle&) = delete;
  ComputeHandle& operator=(const ComputeHandle&) = delete;
};

//--------------------------------------------------------------

struct InputBuffers {
  int maxBatchSize;

  size_t singleInputElts;
  size_t singleInputGlobalElts;

  vector<float> spatialInput;
  vector<float> globalInput;

  InputBuffers(const LoadedModel* loadedModel, int maxBatchSz, int nnXLen, int nnYLen) {
    const ModelDesc& m = loadedModel->modelDesc;
    maxBatchSize = maxBatchSz;
    singleInputElts = (size_t)m.numInputChannels * nnXLen * nnYLen;
    singleInputGlobalElts = (size_t)m.numInputGlobalChannels;
    spatialInput.resize(singleInputElts * maxBatchSize, 0.0f);
    globalInput.resize(singleInputGlobalElts * maxBatchSize, 0.0f);
  }

  ~InputBuffers() {}

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

//--------------------------------------------------------------

void NeuralNet::globalInitialize() {
}

void NeuralNet::globalCleanup() {
}

//--------------------------------------------------------------

ComputeContext* NeuralNet::createComputeContext(
  const std::vector<int>& gpuIdxs,
  Logger* logger,
  int nnXLen,
  int nnYLen,
  const string& backendExtraParam,
  const string& homeDataDirOverride,
  bool openCLReTunePerBoardSize,
  enabled_t useFP16Mode,
  enabled_t useNHWCMode,
  const LoadedModel* loadedModel
) {
  (void)gpuIdxs;
  (void)homeDataDirOverride;
  (void)openCLReTunePerBoardSize;
  (void)useFP16Mode;
  (void)useNHWCMode;
  (void)loadedModel;

  string providerName = backendExtraParam.empty() ? "cpu" : backendExtraParam;

  if(logger != NULL)
    logger->write("ONNX backend: creating compute context for " +
                   Global::intToString(nnXLen) + "x" + Global::intToString(nnYLen) +
                   " with provider '" + providerName + "'");

  return new ComputeContext(nnXLen, nnYLen, providerName);
}

void NeuralNet::freeComputeContext(ComputeContext* computeContext) {
  delete computeContext;
}

//--------------------------------------------------------------

ComputeHandle* NeuralNet::createComputeHandle(
  ComputeContext* context,
  const LoadedModel* loadedModel,
  Logger* logger,
  int maxBatchSize,
  bool requireExactNNLen,
  bool inputsUseNHWC,
  int gpuIdxForThisThread,
  int serverThreadIdx
) {
  (void)maxBatchSize;
  (void)requireExactNNLen;
  (void)gpuIdxForThisThread;

  if(inputsUseNHWC)
    throw StringError("ONNX backend: inputsUseNHWC = true not supported, must use NCHW");

  if(logger != NULL) {
    logger->write("ONNX backend thread " + Global::intToString(serverThreadIdx) +
                  ": Model version " + Global::intToString(loadedModel->modelDesc.modelVersion));
    logger->write("ONNX backend thread " + Global::intToString(serverThreadIdx) +
                  ": Model name: " + loadedModel->modelDesc.name);
  }

  return new ComputeHandle(context, *loadedModel, logger);
}

void NeuralNet::freeComputeHandle(ComputeHandle* computeHandle) {
  delete computeHandle;
}

bool NeuralNet::isUsingFP16(const ComputeHandle* handle) {
  (void)handle;
  return false;
}

//--------------------------------------------------------------

// Helper to find the index of a name in a vector, checking multiple alternatives.
static int findNameIndex(const vector<string>& names, const vector<string>& targets) {
  for(size_t i = 0; i < names.size(); i++) {
    for(const auto& t : targets) {
      if(names[i] == t)
        return (int)i;
    }
  }
  return -1;
}

void NeuralNet::getOutput(
  ComputeHandle* computeHandle,
  InputBuffers* inputBuffers,
  int numBatchEltsFilled,
  NNResultBuf** inputBufs,
  vector<NNOutput*>& outputs
) {
  assert(numBatchEltsFilled <= inputBuffers->maxBatchSize);
  assert(numBatchEltsFilled > 0);
  const int batchSize = numBatchEltsFilled;
  const int nnXLen = computeHandle->context->nnXLen;
  const int nnYLen = computeHandle->context->nnYLen;
  const int numSpatialFeatures = computeHandle->numInputChannels;
  const int numGlobalFeatures = computeHandle->numInputGlobalChannels;
  const int numPolicyChannels = computeHandle->numPolicyChannels;

  // Fill input buffers
  for(int nIdx = 0; nIdx < batchSize; nIdx++) {
    float* rowSpatialInput = inputBuffers->spatialInput.data() + (inputBuffers->singleInputElts * nIdx);
    float* rowGlobalInput = inputBuffers->globalInput.data() + (inputBuffers->singleInputGlobalElts * nIdx);

    const float* rowGlobal = inputBufs[nIdx]->rowGlobalBuf.data();
    const float* rowSpatial = inputBufs[nIdx]->rowSpatialBuf.data();
    std::copy(rowGlobal, rowGlobal + numGlobalFeatures, rowGlobalInput);
    SymmetryHelpers::copyInputsWithSymmetry(rowSpatial, rowSpatialInput, 1, nnYLen, nnXLen, numSpatialFeatures, false, inputBufs[nIdx]->symmetry);
  }

  // Create ONNX tensors
  Ort::MemoryInfo memInfo = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

  std::array<int64_t, 4> spatialShape = {batchSize, numSpatialFeatures, nnYLen, nnXLen};
  Ort::Value spatialTensor = Ort::Value::CreateTensor<float>(
    memInfo, inputBuffers->spatialInput.data(), inputBuffers->singleInputElts * batchSize,
    spatialShape.data(), spatialShape.size()
  );

  std::array<int64_t, 2> globalShape = {batchSize, numGlobalFeatures};
  Ort::Value globalTensor = Ort::Value::CreateTensor<float>(
    memInfo, inputBuffers->globalInput.data(), inputBuffers->singleInputGlobalElts * batchSize,
    globalShape.data(), globalShape.size()
  );

  // Match input ordering
  int spatialIdx = findNameIndex(computeHandle->inputNames, {"input_spatial"});
  int globalIdx = findNameIndex(computeHandle->inputNames, {"input_global"});
  if(spatialIdx < 0 || globalIdx < 0)
    throw StringError("ONNX backend: could not find expected input names");

  vector<Ort::Value> inputTensors;
  inputTensors.reserve(computeHandle->inputNames.size());
  for(size_t i = 0; i < computeHandle->inputNames.size(); i++) {
    if((int)i == spatialIdx)
      inputTensors.push_back(std::move(spatialTensor));
    else if((int)i == globalIdx)
      inputTensors.push_back(std::move(globalTensor));
    else {
      std::array<int64_t, 1> emptyShape = {0};
      inputTensors.push_back(Ort::Value::CreateTensor<float>(memInfo, nullptr, 0, emptyShape.data(), 1));
    }
  }

  // Run inference
  auto outputTensors = computeHandle->session->Run(
    Ort::RunOptions{nullptr},
    computeHandle->inputNamePtrs.data(),
    inputTensors.data(),
    inputTensors.size(),
    computeHandle->outputNamePtrs.data(),
    computeHandle->outputNamePtrs.size()
  );

  // Find output indices
  int policyOutputIdx = findNameIndex(computeHandle->outputNames, {"out_policy"});
  int valueOutputIdx = findNameIndex(computeHandle->outputNames, {"out_value"});
  int miscvalueOutputIdx = findNameIndex(computeHandle->outputNames, {"out_miscvalue"});
  int moremiscvalueOutputIdx = findNameIndex(computeHandle->outputNames, {"out_moremiscvalue"});
  int ownershipOutputIdx = findNameIndex(computeHandle->outputNames, {"out_ownership"});

  const float* policyData = (policyOutputIdx >= 0) ? outputTensors[policyOutputIdx].GetTensorData<float>() : nullptr;
  const float* valueData = (valueOutputIdx >= 0) ? outputTensors[valueOutputIdx].GetTensorData<float>() : nullptr;
  const float* miscvalueData = (miscvalueOutputIdx >= 0) ? outputTensors[miscvalueOutputIdx].GetTensorData<float>() : nullptr;
  const float* moremiscvalueData = (moremiscvalueOutputIdx >= 0) ? outputTensors[moremiscvalueOutputIdx].GetTensorData<float>() : nullptr;
  const float* ownershipData = (ownershipOutputIdx >= 0) ? outputTensors[ownershipOutputIdx].GetTensorData<float>() : nullptr;

  assert((int)outputs.size() == batchSize);

  const int policyResultLen = computeHandle->policyResultLen;
  const int spatialPolicyLen = nnXLen * nnYLen;
  float policyProbsTmp[NNPos::MAX_NN_POLICY_SIZE];

  for(int row = 0; row < batchSize; row++) {
    NNOutput* output = outputs[row];
    assert(output->nnXLen == nnXLen);
    assert(output->nnYLen == nnYLen);
    float policyOptimism = (float)inputBufs[row]->policyOptimism;

    // Policy: [N, C, H*W+1]
    if(policyData != nullptr) {
      const float* policyRowBase = policyData + row * numPolicyChannels * policyResultLen;
      float* policyProbs = output->policyProbs;

      if(numPolicyChannels >= 2) {
        const float* ch0 = policyRowBase;
        const float* ch1 = policyRowBase + policyResultLen;
        for(int i = 0; i < spatialPolicyLen; i++) {
          float p = ch0[i];
          float pOpt = ch1[i];
          policyProbsTmp[i] = p + (pOpt - p) * policyOptimism;
        }
        SymmetryHelpers::copyOutputsWithSymmetry(policyProbsTmp, policyProbs, 1, nnYLen, nnXLen, inputBufs[row]->symmetry);
        policyProbs[spatialPolicyLen] = ch0[spatialPolicyLen] + (ch1[spatialPolicyLen] - ch0[spatialPolicyLen]) * policyOptimism;
      } else {
        assert(numPolicyChannels == 1);
        const float* ch0 = policyRowBase;
        SymmetryHelpers::copyOutputsWithSymmetry(ch0, policyProbs, 1, nnYLen, nnXLen, inputBufs[row]->symmetry);
        policyProbs[spatialPolicyLen] = ch0[spatialPolicyLen];
      }
    }

    // Value: [N, 3]
    if(valueData != nullptr) {
      int numVC = computeHandle->numValueChannels;
      assert(numVC == 3);
      output->whiteWinProb = valueData[row * numVC];
      output->whiteLossProb = valueData[row * numVC + 1];
      output->whiteNoResultProb = valueData[row * numVC + 2];
    }

    // MiscValue: [N, numScoreValueChannels]
    if(miscvalueData != nullptr) {
      int miscStride = computeHandle->numScoreValueChannels;
      output->whiteScoreMean = miscvalueData[row * miscStride + 0];
      output->whiteScoreMeanSq = miscvalueData[row * miscStride + 1];
      output->whiteLead = miscvalueData[row * miscStride + 2];
      output->varTimeLeft = miscvalueData[row * miscStride + 3];
    } else {
      output->whiteScoreMean = 0;
      output->whiteScoreMeanSq = 0;
      output->whiteLead = 0;
      output->varTimeLeft = 0;
    }

    if(moremiscvalueData != nullptr) {
      int moreMiscStride = 8;
      output->shorttermWinlossError = moremiscvalueData[row * moreMiscStride + 0];
      output->shorttermScoreError = moremiscvalueData[row * moreMiscStride + 1];
    } else {
      output->shorttermWinlossError = 0;
      output->shorttermScoreError = 0;
    }

    // Ownership: [N, 1, H, W]
    if(output->whiteOwnerMap != NULL && ownershipData != nullptr) {
      assert(computeHandle->numOwnershipChannels == 1);
      const float* ownershipRowBuf = ownershipData + row * nnXLen * nnYLen;
      SymmetryHelpers::copyOutputsWithSymmetry(ownershipRowBuf, output->whiteOwnerMap, 1, nnYLen, nnXLen, inputBufs[row]->symmetry);
    }
  }
}

void NeuralNet::printDevices() {
}

//--------------------------------------------------------------
// FOR TESTING — all return false (not implemented for this backend)

bool NeuralNet::testEvaluateConv(
  const ConvLayerDesc* desc, int batchSize, int nnXLen, int nnYLen,
  bool useFP16, bool useNHWC, const std::vector<float>& inputBuffer, std::vector<float>& outputBuffer
) {
  (void)desc; (void)batchSize; (void)nnXLen; (void)nnYLen;
  (void)useFP16; (void)useNHWC; (void)inputBuffer; (void)outputBuffer;
  return false;
}

bool NeuralNet::testEvaluateBatchNorm(
  const BatchNormLayerDesc* desc, int batchSize, int nnXLen, int nnYLen,
  bool useFP16, bool useNHWC, const std::vector<float>& inputBuffer,
  const std::vector<float>& maskBuffer, std::vector<float>& outputBuffer
) {
  (void)desc; (void)batchSize; (void)nnXLen; (void)nnYLen;
  (void)useFP16; (void)useNHWC; (void)inputBuffer; (void)maskBuffer; (void)outputBuffer;
  return false;
}

bool NeuralNet::testEvaluateResidualBlock(
  const ResidualBlockDesc* desc, int batchSize, int nnXLen, int nnYLen,
  bool useFP16, bool useNHWC, const std::vector<float>& inputBuffer,
  const std::vector<float>& maskBuffer, std::vector<float>& outputBuffer
) {
  (void)desc; (void)batchSize; (void)nnXLen; (void)nnYLen;
  (void)useFP16; (void)useNHWC; (void)inputBuffer; (void)maskBuffer; (void)outputBuffer;
  return false;
}

bool NeuralNet::testEvaluateGlobalPoolingResidualBlock(
  const GlobalPoolingResidualBlockDesc* desc, int batchSize, int nnXLen, int nnYLen,
  bool useFP16, bool useNHWC, const std::vector<float>& inputBuffer,
  const std::vector<float>& maskBuffer, std::vector<float>& outputBuffer
) {
  (void)desc; (void)batchSize; (void)nnXLen; (void)nnYLen;
  (void)useFP16; (void)useNHWC; (void)inputBuffer; (void)maskBuffer; (void)outputBuffer;
  return false;
}
