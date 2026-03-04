// ONNX Runtime backend for KataGo.
// Loads standard .bin.gz model files (builds ONNX graph from ModelDesc) or
// raw .onnx model files directly, and runs inference via ONNX Runtime with a
// configurable execution provider (CPU, CoreML, etc.) selected at runtime via
// the onnxProvider config key.

#include "../neuralnet/nninterface.h"
#include "../neuralnet/nneval.h"
#include "../neuralnet/nninputs.h"
#include "../neuralnet/modelversion.h"
#include "../neuralnet/onnxmodelbuilder.h"

#include <onnxruntime_cxx_api.h>
#ifdef __APPLE__
#include <coreml_provider_factory.h>
#endif

#include <fstream>

using namespace std;

//--------------------------------------------------------------

// Auto-detect modelVersion from introspected channel counts.
//
// Detection is based on channel-count heuristics for raw .onnx files where the
// model version is not encoded in the file.  The mapping assumes V7 inputs
// (22 spatial + 19 global channels) and distinguishes versions by the number of
// score-value and policy output channels:
//   - 4 score-value channels                    → version 8
//   - 6 score-value channels, 1 policy channel  → version 10
//   - 6 score-value channels, 2 policy channels → version 15
//
// If the heuristic picks the wrong version, set the `onnxModelVersion` config
// key to the correct value (>= 0) to override auto-detection.
static int detectModelVersion(
  int numInputChannels, int numInputGlobalChannels,
  int numPolicyChannels, int numScoreValueChannels,
  int configModelVersion
) {
  if(configModelVersion >= 0)
    return configModelVersion;

  // inputsVersion 7 → models 8-16: 22 spatial + 19 global
  if(numInputChannels == NNInputs::NUM_FEATURES_SPATIAL_V7 &&
     numInputGlobalChannels == NNInputs::NUM_FEATURES_GLOBAL_V7) {
    if(numScoreValueChannels == 6 && numPolicyChannels == 2)
      return 15;
    if(numScoreValueChannels == 6 && numPolicyChannels == 1)
      return 10;
    if(numScoreValueChannels == 4)
      return 8;
    // Default for V7 inputs
    return 15;
  }
  // Older input versions — fall back to a reasonable default
  return NNModelVersion::defaultModelVersion;
}

struct LoadedModel {
  ModelDesc modelDesc;
  bool isRawOnnx;
  string rawOnnxBytes;

  // Constructor for .bin.gz files
  LoadedModel(const string& fileName, const string& expectedSha256, bool rawOnnx)
    : isRawOnnx(rawOnnx)
  {
    if(!rawOnnx) {
      ModelDesc::loadFromFileMaybeGZipped(fileName, modelDesc, expectedSha256);
      return;
    }

    // Read raw .onnx file bytes
    {
      std::ifstream in(fileName, std::ios::binary | std::ios::ate);
      if(!in.good())
        throw StringError("ONNX backend: could not open raw ONNX file: " + fileName);
      std::streamsize size = in.tellg();
      if(size < 0)
        throw StringError("ONNX backend: could not determine size of ONNX file: " + fileName);
      in.seekg(0, std::ios::beg);
      rawOnnxBytes.resize(size);
      if(!in.read(rawOnnxBytes.data(), size))
        throw StringError("ONNX backend: failed to read raw ONNX file: " + fileName);
    }

    // Create a temporary CPU session to introspect shapes
    Ort::Env tmpEnv(ORT_LOGGING_LEVEL_WARNING, "KataGoOnnxIntrospect");
    Ort::SessionOptions tmpOpts;
    tmpOpts.SetIntraOpNumThreads(1);
    Ort::Session tmpSession(tmpEnv, rawOnnxBytes.data(), rawOnnxBytes.size(), tmpOpts);

    Ort::AllocatorWithDefaultOptions allocator;

    // Introspect inputs by name first, falling back to shape-based heuristic
    int numInputChannels = 0;
    int numInputGlobalChannels = 0;
    int numInputMetaChannels = 0;
    size_t numInputs = tmpSession.GetInputCount();
    for(size_t i = 0; i < numInputs; i++) {
      Ort::AllocatedStringPtr namePtr = tmpSession.GetInputNameAllocated(i, allocator);
      string name = namePtr.get();
      auto typeInfo = tmpSession.GetInputTypeInfo(i);
      auto tensorInfo = typeInfo.GetTensorTypeAndShapeInfo();
      auto shape = tensorInfo.GetShape();
      if(name.find("spatial") != string::npos) {
        if(shape.size() >= 2)
          numInputChannels = (int)shape[1];
      } else if(name.find("global") != string::npos) {
        if(shape.size() >= 2)
          numInputGlobalChannels = (int)shape[1];
      } else if(name.find("meta") != string::npos) {
        if(shape.size() >= 2)
          numInputMetaChannels = (int)shape[1];
      } else if(shape.size() == 4) {
        // Shape-based fallback: [N, C, H, W] — spatial input
        numInputChannels = (int)shape[1];
      } else if(shape.size() == 2) {
        // Shape-based fallback: [N, C] — first 2D is global, second is meta
        if(numInputGlobalChannels == 0)
          numInputGlobalChannels = (int)shape[1];
        else
          numInputMetaChannels = (int)shape[1];
      } else {
        cerr << "ONNX backend warning: unrecognized input tensor '" << name
             << "' with " << shape.size() << "D shape, ignoring" << "\n";
      }
    }

    // Introspect outputs
    int numPolicyChannels = 0;
    int numValueChannels = 0;
    int numScoreValueChannels = 0;
    int numOwnershipChannels = 0;
    size_t numOutputs = tmpSession.GetOutputCount();
    for(size_t i = 0; i < numOutputs; i++) {
      Ort::AllocatedStringPtr namePtr = tmpSession.GetOutputNameAllocated(i, allocator);
      string name = namePtr.get();
      auto typeInfo = tmpSession.GetOutputTypeInfo(i);
      auto tensorInfo = typeInfo.GetTensorTypeAndShapeInfo();
      auto shape = tensorInfo.GetShape();

      if(name.find("policy") != string::npos) {
        // Policy: [N, C, H*W+1] → dim 1 is policy channels
        if(shape.size() >= 2)
          numPolicyChannels = (int)shape[1];
      } else if(name.find("miscvalue") != string::npos) {
        // MiscValue: [N, numScoreValueChannels] — check before "value" since "miscvalue" contains "value"
        if(shape.size() >= 2)
          numScoreValueChannels = (int)shape[1];
      } else if(name.find("value") != string::npos) {
        // Value: [N, 3]
        if(shape.size() >= 2)
          numValueChannels = (int)shape[1];
      } else if(name.find("ownership") != string::npos) {
        // Ownership: [N, 1, H, W]
        if(shape.size() >= 2)
          numOwnershipChannels = (int)shape[1];
      }
    }

    // Populate ModelDesc metadata (weights are in the ONNX graph, not in modelDesc)
    modelDesc.numInputChannels = numInputChannels;
    modelDesc.numInputGlobalChannels = numInputGlobalChannels;
    modelDesc.numInputMetaChannels = numInputMetaChannels;
    modelDesc.numPolicyChannels = numPolicyChannels;
    modelDesc.numValueChannels = numValueChannels;
    modelDesc.numScoreValueChannels = numScoreValueChannels;
    modelDesc.numOwnershipChannels = numOwnershipChannels;

    // Extract filename stem as model name
    {
      size_t lastSlash = fileName.find_last_of("/\\");
      string basename = (lastSlash != string::npos) ? fileName.substr(lastSlash + 1) : fileName;
      size_t dotPos = basename.find('.');
      modelDesc.name = (dotPos != string::npos) ? basename.substr(0, dotPos) : basename;
    }

    // Model version: auto-detect with possible config override (applied later)
    modelDesc.modelVersion = detectModelVersion(
      numInputChannels, numInputGlobalChannels,
      numPolicyChannels, numScoreValueChannels,
      -1  // No config override at load time; applied in createComputeHandle if needed
    );

    // postProcessParams gets default values from its constructor (already set)
  }

  LoadedModel() = delete;
  LoadedModel(const LoadedModel&) = delete;
  LoadedModel& operator=(const LoadedModel&) = delete;
};

LoadedModel* NeuralNet::loadModelFile(const string& file, const string& expectedSha256) {
  bool isRawOnnx = Global::isSuffix(file, ".onnx");
  return new LoadedModel(file, expectedSha256, isRawOnnx);
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

  // Configurable input/output node names
  string inputSpatialName;
  string inputGlobalName;
  string inputMetaName;
  string outputPolicyName;
  string outputValueName;
  string outputMiscvalueName;
  string outputOwnershipName;

  // Config override for model version (-1 means auto-detect)
  int configModelVersion;

  ComputeContext(int xLen, int yLen, const string& provider)
    : env(ORT_LOGGING_LEVEL_WARNING, "KataGoOnnx"),
      nnXLen(xLen),
      nnYLen(yLen),
      providerName(provider),
      inputSpatialName("input_spatial"),
      inputGlobalName("input_global"),
      inputMetaName("input_meta"),
      outputPolicyName("out_policy"),
      outputValueName("out_value"),
      outputMiscvalueName("out_miscvalue"),
      outputOwnershipName("out_ownership"),
      configModelVersion(-1)
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
  int numInputMetaChannels;
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
      numInputMetaChannels(loadedModel.modelDesc.numInputMetaChannels),
      policyResultLen(ctx->nnXLen * ctx->nnYLen + 1)
  {
    // Apply config model version override if set
    if(ctx->configModelVersion >= 0)
      modelVersion = ctx->configModelVersion;

    const char* onnxData;
    size_t onnxSize;
    string builtOnnxBytes;
    if(loadedModel.isRawOnnx) {
      if(logger != NULL)
        logger->write("ONNX backend: using raw ONNX model (" +
                       Global::uint64ToString(loadedModel.rawOnnxBytes.size()) + " bytes)");
      onnxData = loadedModel.rawOnnxBytes.data();
      onnxSize = loadedModel.rawOnnxBytes.size();
    } else {
      if(logger != NULL)
        logger->write("ONNX backend: building ONNX graph from model weights...");
      builtOnnxBytes = OnnxModelBuilder::buildOnnxModel(loadedModel.modelDesc, ctx->nnXLen, ctx->nnYLen);
      if(logger != NULL)
        logger->write("ONNX backend: ONNX graph built (" + Global::uint64ToString(builtOnnxBytes.size()) + " bytes)");
      onnxData = builtOnnxBytes.data();
      onnxSize = builtOnnxBytes.size();
    }

    if(logger != NULL)
      logger->write("ONNX backend: creating session...");

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
    session = std::make_unique<Ort::Session>(ctx->env, onnxData, onnxSize, sessionOpts);

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
  size_t singleInputMetaElts;

  vector<float> spatialInput;
  vector<float> globalInput;
  vector<float> metaInput;

  InputBuffers(const LoadedModel* loadedModel, int maxBatchSz, int nnXLen, int nnYLen) {
    const ModelDesc& m = loadedModel->modelDesc;
    maxBatchSize = maxBatchSz;
    singleInputElts = (size_t)m.numInputChannels * nnXLen * nnYLen;
    singleInputGlobalElts = (size_t)m.numInputGlobalChannels;
    singleInputMetaElts = (size_t)m.numInputMetaChannels;
    spatialInput.resize(singleInputElts * maxBatchSize, 0.0f);
    globalInput.resize(singleInputGlobalElts * maxBatchSize, 0.0f);
    if(m.numInputMetaChannels > 0)
      metaInput.resize(singleInputMetaElts * maxBatchSize, 0.0f);
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

  // Parse backendExtraParam as "key=value;key=value;..."
  string providerName = "cpu";
  map<string, string> params;
  if(!backendExtraParam.empty()) {
    vector<string> parts = Global::split(backendExtraParam, ';');
    for(const string& part : parts) {
      size_t eq = part.find('=');
      if(eq != string::npos) {
        string key = Global::trim(part.substr(0, eq));
        string val = Global::trim(part.substr(eq + 1));
        params[key] = val;
      } else {
        // Legacy: bare string is provider name
        string trimmed = Global::trim(part);
        if(!trimmed.empty())
          providerName = trimmed;
      }
    }
    if(params.count("provider"))
      providerName = params["provider"];
  }

  if(logger != NULL)
    logger->write("ONNX backend: creating compute context for " +
                   Global::intToString(nnXLen) + "x" + Global::intToString(nnYLen) +
                   " with provider '" + providerName + "'");

  ComputeContext* ctx = new ComputeContext(nnXLen, nnYLen, providerName);

  // Apply configured node names
  if(params.count("inputSpatial")) ctx->inputSpatialName = params["inputSpatial"];
  if(params.count("inputGlobal")) ctx->inputGlobalName = params["inputGlobal"];
  if(params.count("inputMeta")) ctx->inputMetaName = params["inputMeta"];
  if(params.count("outputPolicy")) ctx->outputPolicyName = params["outputPolicy"];
  if(params.count("outputValue")) ctx->outputValueName = params["outputValue"];
  if(params.count("outputMiscvalue")) ctx->outputMiscvalueName = params["outputMiscvalue"];
  if(params.count("outputOwnership")) ctx->outputOwnershipName = params["outputOwnership"];
  if(params.count("modelVersion")) {
    int v = Global::stringToInt(params["modelVersion"]);
    if(v >= 0)
      ctx->configModelVersion = v;
  }

  return ctx;
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

    if(computeHandle->numInputMetaChannels > 0) {
      float* rowMetaInput = inputBuffers->metaInput.data() + (inputBuffers->singleInputMetaElts * nIdx);
      const float* rowMeta = inputBufs[nIdx]->rowMetaBuf.data();
      std::copy(rowMeta, rowMeta + computeHandle->numInputMetaChannels, rowMetaInput);
    }
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

  // Match input ordering using configured node names
  const ComputeContext* ctx = computeHandle->context;
  int spatialIdx = findNameIndex(computeHandle->inputNames, {ctx->inputSpatialName});
  int globalIdx = findNameIndex(computeHandle->inputNames, {ctx->inputGlobalName});
  if(spatialIdx < 0 || globalIdx < 0)
    throw StringError("ONNX backend: could not find expected input names");

  int metaIdx = -1;
  Ort::Value metaTensor(nullptr);
  if(computeHandle->numInputMetaChannels > 0) {
    metaIdx = findNameIndex(computeHandle->inputNames, {ctx->inputMetaName});
    if(metaIdx < 0)
      throw StringError("ONNX backend: model has metadata channels but could not find input_meta");
    std::array<int64_t, 2> metaShape = {batchSize, computeHandle->numInputMetaChannels};
    metaTensor = Ort::Value::CreateTensor<float>(
      memInfo, inputBuffers->metaInput.data(), inputBuffers->singleInputMetaElts * batchSize,
      metaShape.data(), metaShape.size()
    );
  }

  vector<Ort::Value> inputTensors;
  inputTensors.reserve(computeHandle->inputNames.size());
  for(size_t i = 0; i < computeHandle->inputNames.size(); i++) {
    if((int)i == spatialIdx)
      inputTensors.push_back(std::move(spatialTensor));
    else if((int)i == globalIdx)
      inputTensors.push_back(std::move(globalTensor));
    else if((int)i == metaIdx)
      inputTensors.push_back(std::move(metaTensor));
    else {
      throw StringError("ONNX backend: unexpected input node '" + computeHandle->inputNames[i] +
                         "' — only spatial, global, and meta inputs are supported");
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

  // Find output indices using configured node names
  int policyOutputIdx = findNameIndex(computeHandle->outputNames, {ctx->outputPolicyName});
  int valueOutputIdx = findNameIndex(computeHandle->outputNames, {ctx->outputValueName});
  int miscvalueOutputIdx = findNameIndex(computeHandle->outputNames, {ctx->outputMiscvalueName});
  int ownershipOutputIdx = findNameIndex(computeHandle->outputNames, {ctx->outputOwnershipName});

  if(policyOutputIdx < 0)
    throw StringError("ONNX backend: could not find policy output node '" + ctx->outputPolicyName + "'");
  if(valueOutputIdx < 0)
    throw StringError("ONNX backend: could not find value output node '" + ctx->outputValueName + "'");
  if(miscvalueOutputIdx < 0)
    throw StringError("ONNX backend: could not find miscvalue output node '" + ctx->outputMiscvalueName + "'");
  if(ownershipOutputIdx < 0)
    throw StringError("ONNX backend: could not find ownership output node '" + ctx->outputOwnershipName + "'");

  const float* policyData = outputTensors[policyOutputIdx].GetTensorData<float>();
  const float* valueData = outputTensors[valueOutputIdx].GetTensorData<float>();
  const float* miscvalueData = outputTensors[miscvalueOutputIdx].GetTensorData<float>();
  const float* ownershipData = outputTensors[ownershipOutputIdx].GetTensorData<float>();

  assert(policyData != nullptr);
  assert(valueData != nullptr);
  assert(miscvalueData != nullptr);
  assert(ownershipData != nullptr);
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
    {
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
    {
      int numVC = computeHandle->numValueChannels;
      assert(numVC == 3);
      output->whiteWinProb = valueData[row * numVC];
      output->whiteLossProb = valueData[row * numVC + 1];
      output->whiteNoResultProb = valueData[row * numVC + 2];
    }

    // MiscValue: [N, numScoreValueChannels] — version-dependent interpretation
    {
      int numScoreValueChannels = computeHandle->numScoreValueChannels;
      if(computeHandle->modelVersion >= 9) {
        assert(numScoreValueChannels >= 6);
        output->whiteScoreMean = miscvalueData[row * numScoreValueChannels];
        output->whiteScoreMeanSq = miscvalueData[row * numScoreValueChannels + 1];
        output->whiteLead = miscvalueData[row * numScoreValueChannels + 2];
        output->varTimeLeft = miscvalueData[row * numScoreValueChannels + 3];
        output->shorttermWinlossError = miscvalueData[row * numScoreValueChannels + 4];
        output->shorttermScoreError = miscvalueData[row * numScoreValueChannels + 5];
      }
      else if(computeHandle->modelVersion >= 8) {
        assert(numScoreValueChannels >= 4);
        output->whiteScoreMean = miscvalueData[row * numScoreValueChannels];
        output->whiteScoreMeanSq = miscvalueData[row * numScoreValueChannels + 1];
        output->whiteLead = miscvalueData[row * numScoreValueChannels + 2];
        output->varTimeLeft = miscvalueData[row * numScoreValueChannels + 3];
        output->shorttermWinlossError = 0;
        output->shorttermScoreError = 0;
      }
      else if(computeHandle->modelVersion >= 4) {
        assert(numScoreValueChannels >= 2);
        output->whiteScoreMean = miscvalueData[row * numScoreValueChannels];
        output->whiteScoreMeanSq = miscvalueData[row * numScoreValueChannels + 1];
        output->whiteLead = output->whiteScoreMean;
        output->varTimeLeft = 0;
        output->shorttermWinlossError = 0;
        output->shorttermScoreError = 0;
      }
      else if(computeHandle->modelVersion >= 3) {
        assert(numScoreValueChannels >= 1);
        output->whiteScoreMean = miscvalueData[row * numScoreValueChannels];
        output->whiteScoreMeanSq = output->whiteScoreMean * output->whiteScoreMean;
        output->whiteLead = output->whiteScoreMean;
        output->varTimeLeft = 0;
        output->shorttermWinlossError = 0;
        output->shorttermScoreError = 0;
      }
      else {
        ASSERT_UNREACHABLE;
      }
    }

    // Ownership: [N, 1, H, W]
    if(output->whiteOwnerMap != NULL) {
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
