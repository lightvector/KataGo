// ONNX Runtime backend for KataGo.
//
// Loads standard .bin.gz KataGo model files, converts the ModelDesc to a serialized
// ONNX ModelProto via the same OnnxModelBuilder that the TensorRT backend uses, and
// hands the bytes to an Ort::Session. Inference is run through ONNX Runtime with a
// configurable execution provider, or EP (CPU, OpenVINO, CUDA, TensorRT, MIGraphX,
// CoreML, DirectML), selected at runtime via the onnxProvider config key. Not every
// provider is tested, and most need an ONNX Runtime package or build that includes
// them - see Compiling.md "Execution providers".
//
// The IO tensor protocol is identical to the TensorRT ONNX-emitter path (see
// onnxmodelbuilder.h): four NCHW float32 inputs declared in the order InputSpatial,
// InputGlobal, InputMeta, InputMask, and five NCHW float32 outputs OutputPolicyPass /
// OutputPolicy / OutputValue / OutputScoreValue / OutputOwnership, all raw logits.
// getOutput below reproduces the TensorRT backend's post-processing exactly (per-row
// optimism blend, inverse-symmetry, version-branched score-value decode) so that the
// same downstream decode path is shared.

#ifdef USE_ONNX_BACKEND

#include "../neuralnet/nninterface.h"
#include "../neuralnet/nneval.h"
#include "../neuralnet/nninputs.h"
#include "../neuralnet/modelversion.h"
#include "../neuralnet/onnxmodelbuilder.h"

#include <onnxruntime_cxx_api.h>
#ifdef __APPLE__
#include <coreml_provider_factory.h>
#endif
#ifdef _WIN32
// dml_provider_factory.h is only shipped by DirectML-enabled ONNX Runtime packages such as
// Microsoft.ML.OnnxRuntime.DirectML, not by the stock CPU prebuilt. Guard on availability so
// that building against an ORT without it still compiles, with the DirectML provider then
// failing at runtime with a clear error instead of at compile time.
#if __has_include(<dml_provider_factory.h>)
#include <dml_provider_factory.h>
#define KATAGO_ONNX_HAS_DML_PROVIDER_FACTORY 1
#endif
#endif

#include <unordered_map>
#include <fstream>
#include <cstdlib>
#include <mutex>

using namespace std;

//--------------------------------------------------------------

// ONNX execution providers this backend knows how to wire up. Being listed here means the
// wiring exists, not that the provider is tested - see Compiling.md "Execution providers".
// Exposing a new provider takes an entry here plus an AppendExecutionProvider_* branch in
// ComputeHandle.
static const char* const kKnownProviders[] = {
  "cpu", "openvino", "cuda", "tensorrt", "migraphx", "coreml", "directml",
};

//--------------------------------------------------------------

struct LoadedModel {
  ModelDesc modelDesc;
  // One-time scale8 transform (see maybeApplyScale8), called from createComputeContext.
  //
  // It MUST run inside createComputeContext rather than lazily at compute-handle creation:
  // applyScale8ToReduceActivations() multiplies postProcessParams.outputScaleMultiplier by 8
  // to compensate for the 1/8-scaled graph outputs, and NNEvaluator snapshots
  // postProcessParams immediately after createComputeContext returns (nneval.cpp). A later
  // application would leave NNEvaluator decoding 1/8-scale outputs with the stale
  // multiplier. Running here also happens-before the server threads spawn and read
  // modelDesc in OnnxModelBuilder::build(). The mutex only keeps the transform idempotent
  // if multiple contexts are ever created on one model.
  mutable bool scale8Resolved;
  mutable std::mutex scale8Mutex;

  LoadedModel(const string& fileName, const string& expectedSha256) {
    if(Global::isSuffix(fileName, ".onnx"))
      throw StringError(
        "ONNX backend: loading a raw .onnx file is not supported by this backend. "
        "Feed a standard KataGo .bin.gz model instead (this backend builds the ONNX "
        "graph from the model weights internally).");
    ModelDesc::loadFromFileMaybeGZipped(fileName, modelDesc, expectedSha256);
    scale8Resolved = false;
  }

  // Apply the scale8 FP16-range workaround exactly once per model, unless skipped via
  // onnxSkipScale8. See the comment on scale8Resolved for why this must run at
  // createComputeContext time.
  void maybeApplyScale8(bool skip) const {
    std::lock_guard<std::mutex> lock(scale8Mutex);
    if(!scale8Resolved) {
      if(!skip)
        const_cast<LoadedModel*>(this)->modelDesc.applyScale8ToReduceActivations();
      scale8Resolved = true;
    }
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
  string openvinoDeviceType;
  string openvinoCacheDir;
  // Optional OpenVINO provider options (empty = not passed to ORT)
  string openvinoPrecision;      // FP16 / FP32 / ACCURACY
  string openvinoNumStreams;     // 1-8
  string openvinoNumOfThreads;   // positive int (infer requests per session)
  string openvinoModelPriority;  // LOW / MEDIUM / HIGH / DEFAULT
  bool transformerNHWC;         // run the trunk block stack channel-last (NHWC)
  bool skipScale8;              // skip the scale8 FP16-range workaround (see createComputeContext)

  // Per-thread device type (index = serverThreadIdx). Filled with openvinoDeviceType
  // by default, and individual entries are replaced by onnxOpenVINODeviceTypeThread<N>.
  std::vector<std::string> perThreadDeviceType;

  // Per-device-type EP option overrides.
  // Outer key = short device name ("NPU", "GPU", "CPU").
  // Inner key = ORT EP option key ("num_streams", "precision", ...).
  std::unordered_map<std::string, std::unordered_map<std::string, std::string>> deviceConfigOverrides;

  ComputeContext(int xLen, int yLen)
    : env(ORT_LOGGING_LEVEL_WARNING, "KataGoOnnx"),
      nnXLen(xLen),
      nnYLen(yLen),
      providerName("cpu"),
      openvinoDeviceType("GPU"),
      openvinoCacheDir(""),
      openvinoPrecision(""),
      openvinoNumStreams(""),
      openvinoNumOfThreads(""),
      openvinoModelPriority(""),
      transformerNHWC(true),
      skipScale8(false)
  {}
};

ComputeContext* NeuralNet::createComputeContext(
  const std::vector<int>& gpuIdxs,
  Logger* logger,
  int nnXLen,
  int nnYLen,
  const string& homeDataDirOverride,
  enabled_t useFP16Mode,
  const LoadedModel* loadedModel,
  ConfigParser& cfg
) {
  (void)gpuIdxs;
  (void)homeDataDirOverride;
  // The emitted ONNX graph is fp32, and inference precision is chosen internally by the
  // execution provider (e.g. OpenVINO downcasts to FP16 per onnxOpenVINOPrecision). KataGo's
  // global useFP16 flag therefore cannot force FP16 here, so fail loudly instead of silently
  // ignoring a request. useFP16 = false is honored, though: for the OpenVINO provider it
  // forces precision = FP32 below.
  if(useFP16Mode == enabled_t::True)
    throw StringError(
      "ONNX backend: useFP16 = true is not supported and cannot be honored. "
      "Precision is controlled by the execution provider; for the OpenVINO provider set "
      "onnxOpenVINOPrecision (e.g. FP16/FP32/ACCURACY). Leave useFP16 unset/auto, or set it "
      "to false to force full FP32.");

  ComputeContext* ctx = new ComputeContext(nnXLen, nnYLen);

  // Provider selection. Defaults to CPU. OpenVINO is the EP used for Intel Arc GPUs.
  string providerName = cfg.contains("onnxProvider") ? cfg.getString("onnxProvider") : "cpu";
  ctx->providerName = Global::toLower(providerName);

  // OpenVINO EP options.
  ctx->openvinoDeviceType = cfg.contains("onnxOpenVINODeviceType") ? cfg.getString("onnxOpenVINODeviceType") : "GPU";
  ctx->openvinoCacheDir = cfg.contains("onnxOpenVINOCacheDir") ? cfg.getString("onnxOpenVINOCacheDir") : "";
  ctx->openvinoPrecision = cfg.contains("onnxOpenVINOPrecision") ? cfg.getString("onnxOpenVINOPrecision") : "";
  ctx->openvinoNumStreams = cfg.contains("onnxOpenVINONumStreams") ? cfg.getString("onnxOpenVINONumStreams") : "";
  ctx->openvinoNumOfThreads = cfg.contains("onnxOpenVINONumOfThreads") ? cfg.getString("onnxOpenVINONumOfThreads") : "";
  ctx->openvinoModelPriority = cfg.contains("onnxOpenVINOModelPriority") ? cfg.getString("onnxOpenVINOModelPriority") : "";

  // useFP16 = false is an explicit request for full FP32 on every other backend. The only
  // provider here that downcasts an fp32 graph by default is OpenVINO (GPU/NPU run FP16
  // unless told otherwise), so honor the request by forcing its precision option to FP32
  // when the user has not explicitly set onnxOpenVINOPrecision themselves.
  if(useFP16Mode == enabled_t::False && ctx->providerName == "openvino" && ctx->openvinoPrecision.empty()) {
    ctx->openvinoPrecision = "FP32";
    if(logger != NULL)
      logger->write("ONNX backend: useFP16 = false, forcing OpenVINO precision = FP32");
  }

  // Trunk layout for transformer models. Default NHWC (channel-last), matching the TensorRT
  // backend's trtTransformerNHWC default. NHWC is markedly faster for transformer trunks on
  // OpenVINO GPU/NPU, and is ignored entirely for models without transformer blocks.
  ctx->transformerNHWC = cfg.contains("onnxTransformerNHWC") ? cfg.getBool("onnxTransformerNHWC") : true;

  // Skip the scale8 FP16-range workaround. Default false, meaning the workaround is applied.
  // See the onnxSkipScale8 documentation in configs/gtp_example.cfg for the tradeoff.
  ctx->skipScale8 = cfg.contains("onnxSkipScale8") ? cfg.getBool("onnxSkipScale8") : false;

  // Must happen here rather than at compute-handle creation. See LoadedModel::scale8Resolved.
  loadedModel->maybeApplyScale8(ctx->skipScale8);

  // --- Per-thread device type assignment ---
  // Pre-parse onnxOpenVINODeviceTypeThread<N> keys so ComputeHandle can look up
  // the device type for each server thread without reaching back into ConfigParser.
  {
    int numThreads = 1;
    if(cfg.contains("numNNServerThreadsPerModel"))
      numThreads = cfg.getInt("numNNServerThreadsPerModel", 1, 1024);
    ctx->perThreadDeviceType.resize(numThreads, ctx->openvinoDeviceType);
    for(int t = 0; t < numThreads; t++) {
      string key = "onnxOpenVINODeviceTypeThread" + Global::intToString(t);
      if(cfg.contains(key))
        ctx->perThreadDeviceType[t] = cfg.getString(key);
    }
  }

  // --- Per-device-type EP option overrides ---
  // onnxOpenVINODeviceConfig_<Device>_<OptionSuffix> = value
  // e.g. onnxOpenVINODeviceConfig_NPU_NumStreams = 4
  //         maps to deviceConfigOverrides["NPU"]["num_streams"] = "4"
  {
    static const char* knownDevices[] = {"NPU", "GPU", "CPU"};
    struct OptMapping { const char* cfgSuffix; const char* ortKey; };
    static const OptMapping epOptMappings[] = {
      {"NumStreams",    "num_streams"},
      {"Precision",     "precision"},
      {"NumOfThreads",  "num_of_threads"},
      {"ModelPriority", "model_priority"},
      {"CacheDir",      "cache_dir"},
    };
    for(const char* dev : knownDevices) {
      string devPrefix = string("onnxOpenVINODeviceConfig_") + dev + "_";
      for(const auto& m : epOptMappings) {
        string key = devPrefix + m.cfgSuffix;
        if(cfg.contains(key))
          ctx->deviceConfigOverrides[dev][m.ortKey] = cfg.getString(key);
      }
    }
  }

  {
    bool knownProvider = false;
    for(const char* p : kKnownProviders) {
      if(ctx->providerName == p) {
        knownProvider = true;
        break;
      }
    }
    if(!knownProvider)
      throw StringError(
        "ONNX backend: unknown onnxProvider '" + ctx->providerName +
        "'. Known providers: cpu, openvino, cuda, tensorrt, migraphx, coreml, directml.");
  }

  if(logger != NULL)
    logger->write("ONNX backend: creating compute context for " +
                  Global::intToString(nnXLen) + "x" + Global::intToString(nnYLen) +
                  " with provider '" + ctx->providerName + "'");

  return ctx;
}

void NeuralNet::freeComputeContext(ComputeContext* computeContext) {
  delete computeContext;
}

//--------------------------------------------------------------
// Helper: extract a short device name from an OpenVINO device_type
// string for matching onnxOpenVINODeviceConfig_<Device>_<Option> keys.
//
//   "NPU"          -> "NPU"
//   "GPU" / "GPU.0" / "GPU.1"  -> "GPU"
//   "CPU"          -> "CPU"
//   "AUTO:GPU,CPU" -> "GPU"
//   "MULTI:GPU,NPU" -> "GPU"
//   "HETERO:GPU,CPU" -> "GPU"
//--------------------------------------------------------------
static std::string extractShortDeviceName(const std::string& deviceType) {
  std::string upper = Global::toUpper(deviceType);

  size_t colonPos = upper.find(':');
  if(colonPos != std::string::npos) {
    std::string prefix = upper.substr(0, colonPos);
    if(prefix == "AUTO" || prefix == "MULTI" || prefix == "HETERO") {
      std::string afterColon = upper.substr(colonPos + 1);
      size_t commaPos = afterColon.find(',');
      if(commaPos != std::string::npos)
        afterColon = afterColon.substr(0, commaPos);
      size_t dotPos = afterColon.find('.');
      if(dotPos != std::string::npos)
        afterColon = afterColon.substr(0, dotPos);
      return afterColon;
    }
  }

  size_t dotPos = upper.find('.');
  if(dotPos != std::string::npos)
    upper = upper.substr(0, dotPos);

  return upper;
}

//--------------------------------------------------------------

struct ComputeHandle {
  ComputeContext* ctx;
  std::unique_ptr<Ort::Session> session;
  int modelVersion;
  int numInputChannels;
  int numInputGlobalChannels;
  int numInputMetaChannels;
  int numPolicyChannels;
  int numValueChannels;
  int numScoreValueChannels;
  int numOwnershipChannels;

  // Queried graph input/output names (and raw-char pointer views for Run).
  vector<string> inputNames;
  vector<string> outputNames;
  vector<const char*> inputNamePtrs;
  vector<const char*> outputNamePtrs;

  ComputeHandle(ComputeContext* context, const LoadedModel& loadedModel, Logger* logger, int deviceIdxForThread, int serverThreadIdx, bool requireExactNNLen)
    : ctx(context),
      modelVersion(loadedModel.modelDesc.modelVersion),
      numInputChannels(loadedModel.modelDesc.numInputChannels),
      numInputGlobalChannels(loadedModel.modelDesc.numInputGlobalChannels),
      numInputMetaChannels(loadedModel.modelDesc.numInputMetaChannels),
      numPolicyChannels(loadedModel.modelDesc.numPolicyChannels),
      numValueChannels(loadedModel.modelDesc.numValueChannels),
      numScoreValueChannels(loadedModel.modelDesc.numScoreValueChannels),
      numOwnershipChannels(loadedModel.modelDesc.numOwnershipChannels)
  {
    if(logger != NULL)
      logger->write("ONNX backend: building ONNX graph from model weights...");

    // Reuse the same ONNX emitter as the TensorRT backend. The serialized ModelProto is a
    // standard ONNX graph that Ort::Session can parse directly. The TRT-only FP32 node-name
    // lists in the Result are ignored, since ORT has no per-node precision API.
    // TODO: every server thread re-runs this build, transiently duplicating the fully
    // weight-baked serialized proto (hundreds of MB for large nets) across N spawning
    // threads. The bytes are identical per (nnXLen, nnYLen, requireExactNNLen,
    // transformerNHWC), so they could be built once in the ComputeContext and shared.
    OnnxModelBuilder::Result onnxResult = OnnxModelBuilder::build(
      loadedModel.modelDesc, ctx->nnXLen, ctx->nnYLen, requireExactNNLen, ctx->transformerNHWC, logger);
    const string& onnxBytes = onnxResult.serializedModel;
    (void)onnxResult.trunkTipAndHeadNodeNames;
    (void)onnxResult.rmsNormNodeNames;

    if(logger != NULL)
      logger->write("ONNX backend: ONNX graph built (" + Global::uint64ToString(onnxBytes.size()) + " bytes)");

    // Dump the ONNX model to a file when KATAGO_DUMP_ONNX is set (debug aid).
    {
      const char* dumpPath = getenv("KATAGO_DUMP_ONNX");
      if(dumpPath != nullptr && dumpPath[0] != '\0') {
        ofstream dumpFile(dumpPath, ios::binary);
        if(dumpFile.is_open()) {
          dumpFile.write(onnxBytes.data(), (streamsize)onnxBytes.size());
          dumpFile.close();
          if(logger != NULL)
            logger->write(string("ONNX backend: dumped ONNX model to ") + dumpPath +
                          " (" + Global::uint64ToString(onnxBytes.size()) + " bytes)");
        } else if(logger != NULL) {
          logger->write(string("ONNX backend: WARNING - could not open dump path ") + dumpPath);
        }
      }
    }

    Ort::SessionOptions sessionOpts;

    // Select execution provider based on providerName.
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
    }
    else if(provider == "cuda") {
      OrtCUDAProviderOptions cudaOpts{};
      cudaOpts.device_id = (unsigned int)(deviceIdxForThread >= 0 ? deviceIdxForThread : 0);
      sessionOpts.AppendExecutionProvider_CUDA(cudaOpts);
      if(logger != NULL)
        logger->write("ONNX backend: CUDA execution provider enabled, device_id=" + Global::intToString((int)cudaOpts.device_id));
    }
    else if(provider == "tensorrt") {
      OrtTensorRTProviderOptions trtOpts{};
      trtOpts.device_id = (unsigned int)(deviceIdxForThread >= 0 ? deviceIdxForThread : 0);
      sessionOpts.AppendExecutionProvider_TensorRT(trtOpts);
      if(logger != NULL)
        logger->write("ONNX backend: TensorRT execution provider enabled, device_id=" + Global::intToString((int)trtOpts.device_id));
    }
    else if(provider == "migraphx") {
      OrtMIGraphXProviderOptions migraphxOpts{};
      migraphxOpts.device_id = (unsigned int)(deviceIdxForThread >= 0 ? deviceIdxForThread : 0);
      sessionOpts.AppendExecutionProvider_MIGraphX(migraphxOpts);
      if(logger != NULL)
        logger->write("ONNX backend: MIGraphX execution provider enabled, device_id=" + Global::intToString((int)migraphxOpts.device_id));
    }
    else if(provider == "openvino") {
      // The OpenVINO EP runs the graph nodes itself and manages its own inference threads via the
      // num_of_threads provider option, leaving ORT's intra-op pool with only the few EP-external
      // nodes. With one ORT session per nn-server thread, leaving the default intra-op thread count
      // would oversubscribe the CPU with N x M worker pools. Pin it to 1 for this provider only.
      sessionOpts.SetIntraOpNumThreads(1);

      // --- Determine this thread's device_type ---
      string threadDeviceType = ctx->openvinoDeviceType;  // global default
      if(serverThreadIdx >= 0 && serverThreadIdx < (int)ctx->perThreadDeviceType.size())
        threadDeviceType = ctx->perThreadDeviceType[serverThreadIdx];

      // --- Look up per-device-type EP option overrides ---
      string shortDev = extractShortDeviceName(threadDeviceType);
      const std::unordered_map<std::string, std::string>* devOverrides = nullptr;
      {
        auto it = ctx->deviceConfigOverrides.find(shortDev);
        if(it != ctx->deviceConfigOverrides.end())
          devOverrides = &it->second;
      }
      auto resolveOpt = [&](const char* ortKey, const std::string& globalVal) -> std::string {
        if(devOverrides) {
          auto it = devOverrides->find(ortKey);
          if(it != devOverrides->end())
            return it->second;
        }
        return globalVal;
      };

      // --- Build EP option map ---
      std::unordered_map<std::string, std::string> openvinoOpts;

      // Map the per-thread device index (from the gpuToUse*/deviceToUse* config keys) into OpenVINO's
      // device_type suffix, e.g. GPU -> GPU.1. The OpenVINO EP selects devices via device_type
      // ("GPU.0", "GPU.1", ...). The legacy device_id provider option is deprecated and only accepts
      // a bare device name, so passing a numeric index there would throw at session creation.
      string deviceType = threadDeviceType;
      if(deviceIdxForThread > 0 && deviceType.find('.') == string::npos && deviceType.find(':') == string::npos)
        deviceType += "." + Global::intToString(deviceIdxForThread);
      else if(deviceIdxForThread > 0 && logger != NULL)
        logger->write(
          "ONNX backend: device index " + Global::intToString(deviceIdxForThread) +
          " ignored for device_type '" + deviceType +
          "' because it already selects a specific device (\"GPU.1\"-style suffix) or is a "
          "composite/qualified device string (AUTO:/MULTI:/HETERO:). "
          "Select the device explicitly in onnxOpenVINODeviceType or the per-thread onnxOpenVINODeviceTypeThread<N> override.");
      openvinoOpts["device_type"] = deviceType;

      auto setIfNotEmpty = [&](const char* ortKey, const std::string& globalVal) {
        std::string val = resolveOpt(ortKey, globalVal);
        if(!val.empty())
          openvinoOpts[ortKey] = val;
      };
      setIfNotEmpty("cache_dir",      ctx->openvinoCacheDir);
      setIfNotEmpty("precision",      ctx->openvinoPrecision);
      setIfNotEmpty("num_streams",    ctx->openvinoNumStreams);
      setIfNotEmpty("num_of_threads", ctx->openvinoNumOfThreads);
      setIfNotEmpty("model_priority", ctx->openvinoModelPriority);

      // Some ORT OpenVINO builds reject optional keys (cache_dir, precision, num_streams,
      // num_of_threads, model_priority). Retry with only the core device keys if optional keys
      // are rejected, so that setting onnxOpenVINOCacheDir on an EP that doesn't support it
      // degrades gracefully instead of crashing.
      static const char* optionalKeys[] = {
        "cache_dir", "precision", "num_streams", "num_of_threads", "model_priority"
      };
      try {
        sessionOpts.AppendExecutionProvider_OpenVINO_V2(openvinoOpts);
      }
      catch(const Ort::Exception& e) {
        bool hadOptionalKeys = false;
        for(const char* k : optionalKeys) {
          if(openvinoOpts.count(k) > 0) {
            hadOptionalKeys = true;
            break;
          }
        }
        if(!hadOptionalKeys)
          throw;

        if(logger != NULL) {
          logger->write(
            string("ONNX backend: OpenVINO optional provider options rejected, retrying without optional keys. Error: ") +
            e.what()
          );
        }
        for(const char* k : optionalKeys)
          openvinoOpts.erase(k);
        sessionOpts.AppendExecutionProvider_OpenVINO_V2(openvinoOpts);
      }

      if(logger != NULL) {
        string extras;
        for(const char* k : optionalKeys) {
          if(openvinoOpts.count(k) > 0)
            extras += string(", ") + k + "=" + openvinoOpts[k];
        }
        logger->write(
          "ONNX backend: OpenVINO execution provider enabled for thread " + Global::intToString(serverThreadIdx) +
          ", device_type=" + deviceType + extras
        );
      }
    }
    else if(provider == "directml") {
#ifdef _WIN32
#if defined(KATAGO_ONNX_HAS_DML_PROVIDER_FACTORY)
      // DirectML does not support memory-pattern optimization and requires sequential
      // execution mode (see the ORT DirectML EP docs). With one session per nn-server
      // thread the single-Run restriction of a DML session is satisfied naturally.
      sessionOpts.DisableMemPattern();
      sessionOpts.SetExecutionMode(ORT_SEQUENTIAL);

      // Prefer the OrtDmlApi route: the plain OrtSessionOptionsAppendExecutionProvider_DML
      // export in dml_provider_factory.h is deprecated. Check the status by hand rather
      // than via Ort::ThrowOnError so that a DML-less ORT build produces this friendly
      // error instead of a generic Ort::Exception.
      const OrtDmlApi* dmlApi = nullptr;
      {
        const OrtApi* ortApi = OrtGetApiBase()->GetApi(ORT_API_VERSION);
        OrtStatus* status = ortApi->GetExecutionProviderApi("DML", ORT_API_VERSION, (const void**)&dmlApi);
        if(status != nullptr || dmlApi == nullptr) {
          string detail = status != nullptr ? ortApi->GetErrorMessage(status) : "GetExecutionProviderApi returned null";
          if(status != nullptr)
            ortApi->ReleaseStatus(status);
          throw StringError(
            "ONNX backend: DirectML execution provider is not available in this ONNX Runtime build: " + detail);
        }
      }

      int dmlDeviceId = deviceIdxForThread >= 0 ? deviceIdxForThread : 0;
      Ort::ThrowOnError(dmlApi->SessionOptionsAppendExecutionProvider_DML(sessionOpts, dmlDeviceId));
      if(logger != NULL)
        logger->write("ONNX backend: DirectML execution provider enabled, device_id=" + Global::intToString(dmlDeviceId));
#else
      throw StringError(
        "ONNX backend: DirectML is not available in this ONNX Runtime build: the ORT install "
        "tree does not ship dml_provider_factory.h. Compile against the "
        "Microsoft.ML.OnnxRuntime.DirectML package to use the DirectML execution provider.");
#endif
#else
      throw StringError("ONNX backend: DirectML is only available on Windows");
#endif
    }
    else if(provider == "cpu" || provider.empty()) {
      if(logger != NULL)
        logger->write("ONNX backend: using CPU execution provider");
    }
    else {
      throw StringError("ONNX backend: unknown onnxProvider '" + provider + "'");
    }

    session = std::make_unique<Ort::Session>(ctx->env, onnxBytes.data(), onnxBytes.size(), sessionOpts);

    Ort::AllocatorWithDefaultOptions allocator;
    size_t numInputs = session->GetInputCount();
    for(size_t i = 0; i < numInputs; i++) {
      Ort::AllocatedStringPtr name = session->GetInputNameAllocated(i, allocator);
      inputNames.push_back(name.get());
    }
    for(auto& n : inputNames)
      inputNamePtrs.push_back(n.c_str());

    size_t numOutputs = session->GetOutputCount();
    for(size_t i = 0; i < numOutputs; i++) {
      Ort::AllocatedStringPtr name = session->GetOutputNameAllocated(i, allocator);
      outputNames.push_back(name.get());
    }
    for(auto& n : outputNames)
      outputNamePtrs.push_back(n.c_str());

    if(logger != NULL) {
      // The graph input/output orders are identical for every server thread, so log them once.
      if(serverThreadIdx <= 0) {
        string inList = "ONNX backend: graph input order:";
        for(size_t i = 0; i < inputNames.size(); i++)
          inList += " [" + Global::uint64ToString(i) + "]" + inputNames[i];
        logger->write(inList);
        string outList = "ONNX backend: graph output order:";
        for(size_t i = 0; i < outputNames.size(); i++)
          outList += " [" + Global::uint64ToString(i) + "]" + outputNames[i];
        logger->write(outList);
      }
      logger->write("ONNX backend: session created, inputs=" + Global::uint64ToString(numInputs) +
                     " outputs=" + Global::uint64ToString(numOutputs));
    }
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
  int serverThreadIdx
) {
  // ONNX Runtime sessions support dynamic batch sizes, but the InputBuffers maxBatchSize
  // field still enforces the upper bound at inference time.
  (void)maxBatchSize;
  if(inputsUseNHWC)
    throw StringError("ONNX backend: inputsUseNHWC = true not supported, must use NCHW");

  if(logger != NULL) {
    logger->write("ONNX backend thread " + Global::intToString(serverThreadIdx) +
                  ": Model version " + Global::intToString(loadedModel->modelDesc.modelVersion));
    logger->write("ONNX backend thread " + Global::intToString(serverThreadIdx) +
                  ": Model name: " + loadedModel->modelDesc.name +
                  " (" + loadedModel->modelDesc.getShortInfoString() + ")");
    string deviceInfo =
      context->providerName == "openvino"
      ? (serverThreadIdx >= 0 && serverThreadIdx < (int)context->perThreadDeviceType.size()
         ? context->perThreadDeviceType[serverThreadIdx]
         : context->openvinoDeviceType)
      : Global::intToString(gpuIdxForThisThread);
    logger->write("ONNX backend thread " + Global::intToString(serverThreadIdx) +
                  ": provider=" + context->providerName + " deviceIdx=" + deviceInfo);
  }

  return new ComputeHandle(context, *loadedModel, logger, gpuIdxForThisThread, serverThreadIdx, requireExactNNLen);
}

void NeuralNet::freeComputeHandle(ComputeHandle* computeHandle) {
  delete computeHandle;
}

bool NeuralNet::isUsingFP16(const ComputeHandle* handle) {
  (void)handle;
  // The emitted ONNX graph is fp32, and precision is delegated to the execution provider,
  // which may downcast internally, so from KataGo's perspective this is fp32.
  return false;
}

bool NeuralNet::setIsWarmup(const ComputeHandle* handle, bool isWarmup) {
  (void)handle;
  (void)isWarmup;
  return false;
}

//--------------------------------------------------------------

struct InputBuffers {
  int maxBatchSize;

  size_t singleMaskElts;
  size_t singleInputElts;
  size_t singleInputGlobalElts;
  size_t singleInputMetaElts;

  size_t singlePolicyPassResultElts;
  size_t singlePolicyResultElts;
  size_t singleValueResultElts;
  size_t singleScoreValueResultElts;
  size_t singleOwnershipResultElts;

  vector<float> maskInput;
  vector<float> spatialInput;
  vector<float> globalInput;
  vector<float> metaInput;

  InputBuffers(const LoadedModel* loadedModel, int maxBatchSz, int nnXLen, int nnYLen) {
    const ModelDesc& m = loadedModel->modelDesc;

    if(nnXLen > NNPos::MAX_BOARD_LEN)
      throw StringError(
        Global::strprintf("nnXLen (%d) is greater than NNPos::MAX_BOARD_LEN (%d)", nnXLen, NNPos::MAX_BOARD_LEN));
    if(nnYLen > NNPos::MAX_BOARD_LEN)
      throw StringError(
        Global::strprintf("nnYLen (%d) is greater than NNPos::MAX_BOARD_LEN (%d)", nnYLen, NNPos::MAX_BOARD_LEN));

    maxBatchSize = maxBatchSz;
    singleMaskElts = (size_t)nnXLen * nnYLen;
    singleInputElts = (size_t)m.numInputChannels * nnXLen * nnYLen;
    singleInputGlobalElts = (size_t)m.numInputGlobalChannels;
    singleInputMetaElts = (size_t)m.numInputMetaChannels;
    singlePolicyPassResultElts = (size_t)m.numPolicyChannels;
    singlePolicyResultElts = (size_t)m.numPolicyChannels * nnXLen * nnYLen;
    singleValueResultElts = (size_t)m.numValueChannels;
    singleScoreValueResultElts = (size_t)m.numScoreValueChannels;
    singleOwnershipResultElts = (size_t)m.numOwnershipChannels * nnXLen * nnYLen;

    testAssert(NNModelVersion::getNumSpatialFeatures(m.modelVersion) == m.numInputChannels);
    testAssert(NNModelVersion::getNumGlobalFeatures(m.modelVersion) == m.numInputGlobalChannels);
    if(m.numInputMetaChannels > 0)
      testAssert(SGFMetadata::METADATA_INPUT_NUM_CHANNELS == m.numInputMetaChannels);

    maskInput.assign(singleMaskElts * maxBatchSize, 0.0f);
    spatialInput.assign(singleInputElts * maxBatchSize, 0.0f);
    globalInput.assign(singleInputGlobalElts * maxBatchSize, 0.0f);
    if(singleInputMetaElts > 0)
      metaInput.assign(singleInputMetaElts * maxBatchSize, 0.0f);
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

//--------------------------------------------------------------

void NeuralNet::globalInitialize() {
}

void NeuralNet::globalCleanup() {
}

//--------------------------------------------------------------

// Find the index of a name in the graph's name list, matching any of the target alternatives.
static int findNameIndex(const vector<string>& names, std::initializer_list<const char*> targets) {
  for(size_t i = 0; i < names.size(); i++) {
    for(const char* t : targets) {
      if(names[i] == t)
        return (int)i;
    }
  }
  return -1;
}

void NeuralNet::getOutput(
  ComputeHandle* gpuHandle,
  InputBuffers* inputBuffers,
  int numBatchEltsFilled,
  NNResultBuf** inputBufs,
  vector<NNOutput*>& outputs
) {
  assert(numBatchEltsFilled <= inputBuffers->maxBatchSize);
  assert(numBatchEltsFilled > 0);

  const int batchSize = numBatchEltsFilled;
  const int nnXLen = gpuHandle->ctx->nnXLen;
  const int nnYLen = gpuHandle->ctx->nnYLen;
  const int modelVersion = gpuHandle->modelVersion;

  const int numSpatialFeatures = NNModelVersion::getNumSpatialFeatures(modelVersion);
  const int numGlobalFeatures = NNModelVersion::getNumGlobalFeatures(modelVersion);
  const int numMetaFeatures = (int)inputBuffers->singleInputMetaElts;
  assert(numSpatialFeatures * nnXLen * nnYLen == inputBuffers->singleInputElts);
  assert(numGlobalFeatures == inputBuffers->singleInputGlobalElts);

  // Fill host input buffers, mirroring the TensorRT backend exactly:
  //  - global / meta are straight copies (no symmetry)
  //  - spatial is symmetry-transformed (NCHW, useNHWC=false)
  //  - mask = channel 0 of the symmetry-transformed spatial input
  for(int nIdx = 0; nIdx < batchSize; nIdx++) {
    float* rowMaskInput = inputBuffers->maskInput.data() + inputBuffers->singleMaskElts * nIdx;
    float* rowSpatialInput = inputBuffers->spatialInput.data() + inputBuffers->singleInputElts * nIdx;
    float* rowGlobalInput = inputBuffers->globalInput.data() + inputBuffers->singleInputGlobalElts * nIdx;

    const float* rowGlobal = inputBufs[nIdx]->rowGlobalBuf.data();
    const float* rowSpatial = inputBufs[nIdx]->rowSpatialBuf.data();
    std::copy(rowGlobal, rowGlobal + numGlobalFeatures, rowGlobalInput);
    SymmetryHelpers::copyInputsWithSymmetry(
      rowSpatial, rowSpatialInput, 1, nnYLen, nnXLen, numSpatialFeatures, false, inputBufs[nIdx]->symmetry);
    std::copy(rowSpatialInput, rowSpatialInput + inputBuffers->singleMaskElts, rowMaskInput);

    if(numMetaFeatures > 0) {
      float* rowMetaInput = inputBuffers->metaInput.data() + inputBuffers->singleInputMetaElts * nIdx;
      const float* rowMeta = inputBufs[nIdx]->rowMetaBuf.data();
      testAssert(inputBufs[nIdx]->hasRowMeta);
      std::copy(rowMeta, rowMeta + numMetaFeatures, rowMetaInput);
    }
    else {
      testAssert(!inputBufs[nIdx]->hasRowMeta);
    }
  }

  // Build Ort::Value views over the host buffers. These stay in CPU memory - the execution
  // provider copies to device internally and returns outputs in CPU memory.
  Ort::MemoryInfo memInfo = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

  std::array<int64_t, 4> maskShape = {batchSize, 1, nnYLen, nnXLen};
  Ort::Value maskTensor = Ort::Value::CreateTensor<float>(
    memInfo, inputBuffers->maskInput.data(), inputBuffers->singleMaskElts * batchSize,
    maskShape.data(), maskShape.size());

  std::array<int64_t, 4> spatialShape = {batchSize, numSpatialFeatures, nnYLen, nnXLen};
  Ort::Value spatialTensor = Ort::Value::CreateTensor<float>(
    memInfo, inputBuffers->spatialInput.data(), inputBuffers->singleInputElts * batchSize,
    spatialShape.data(), spatialShape.size());

  std::array<int64_t, 4> globalShape = {batchSize, numGlobalFeatures, 1, 1};
  Ort::Value globalTensor = Ort::Value::CreateTensor<float>(
    memInfo, inputBuffers->globalInput.data(), inputBuffers->singleInputGlobalElts * batchSize,
    globalShape.data(), globalShape.size());

  Ort::Value metaTensor(nullptr);
  std::array<int64_t, 4> metaShape;
  if(numMetaFeatures > 0) {
    metaShape = {batchSize, numMetaFeatures, 1, 1};
    metaTensor = Ort::Value::CreateTensor<float>(
      memInfo, inputBuffers->metaInput.data(), inputBuffers->singleInputMetaElts * batchSize,
      metaShape.data(), metaShape.size());
  }

  // Bind tensors in the graph's declared input order (ORT matches by pointer array + name array).
  int maskIdx = findNameIndex(gpuHandle->inputNames, {"InputMask"});
  int spatialIdx = findNameIndex(gpuHandle->inputNames, {"InputSpatial"});
  int globalIdx = findNameIndex(gpuHandle->inputNames, {"InputGlobal"});
  if(maskIdx < 0 || spatialIdx < 0 || globalIdx < 0)
    throw StringError("ONNX backend: graph is missing expected inputs InputMask/InputSpatial/InputGlobal");
  int metaIdx = -1;
  if(numMetaFeatures > 0) {
    metaIdx = findNameIndex(gpuHandle->inputNames, {"InputMeta"});
    if(metaIdx < 0)
      throw StringError("ONNX backend: model has metadata channels but the graph has no InputMeta input");
  }

  vector<Ort::Value> inputTensors;
  inputTensors.reserve(gpuHandle->inputNames.size());
  for(size_t i = 0; i < gpuHandle->inputNames.size(); i++) {
    if((int)i == maskIdx)
      inputTensors.push_back(std::move(maskTensor));
    else if((int)i == spatialIdx)
      inputTensors.push_back(std::move(spatialTensor));
    else if((int)i == globalIdx)
      inputTensors.push_back(std::move(globalTensor));
    else if((int)i == metaIdx)
      inputTensors.push_back(std::move(metaTensor));
    else
      throw StringError("ONNX backend: unexpected graph input '" + gpuHandle->inputNames[i] +
                        "' -- only InputMask/InputSpatial/InputGlobal/InputMeta are supported");
  }

  auto outputTensors = gpuHandle->session->Run(
    Ort::RunOptions{nullptr},
    gpuHandle->inputNamePtrs.data(),
    inputTensors.data(),
    inputTensors.size(),
    gpuHandle->outputNamePtrs.data(),
    gpuHandle->outputNamePtrs.size());

  int policyPassIdx = findNameIndex(gpuHandle->outputNames, {"OutputPolicyPass"});
  int policyIdx = findNameIndex(gpuHandle->outputNames, {"OutputPolicy"});
  int valueIdx = findNameIndex(gpuHandle->outputNames, {"OutputValue"});
  int scoreValueIdx = findNameIndex(gpuHandle->outputNames, {"OutputScoreValue"});
  int ownershipIdx = findNameIndex(gpuHandle->outputNames, {"OutputOwnership"});
  if(policyPassIdx < 0 || policyIdx < 0 || valueIdx < 0 || scoreValueIdx < 0 || ownershipIdx < 0)
    throw StringError(
      "ONNX backend: graph is missing expected outputs "
      "(OutputPolicyPass/OutputPolicy/OutputValue/OutputScoreValue/OutputOwnership)");

  const float* policyPassData = outputTensors[policyPassIdx].GetTensorData<float>();
  const float* policyData = outputTensors[policyIdx].GetTensorData<float>();
  const float* valueData = outputTensors[valueIdx].GetTensorData<float>();
  const float* scoreValueData = outputTensors[scoreValueIdx].GetTensorData<float>();
  const float* ownershipData = outputTensors[ownershipIdx].GetTensorData<float>();

  assert(policyPassData != nullptr);
  assert(policyData != nullptr);
  assert(valueData != nullptr);
  assert(scoreValueData != nullptr);
  assert(ownershipData != nullptr);
  assert((int)outputs.size() == batchSize);

  const int numPolicyChannels = (int)inputBuffers->singlePolicyPassResultElts;
  assert(inputBuffers->singlePolicyResultElts == (size_t)numPolicyChannels * nnXLen * nnYLen);
  const int numValueChannels = (int)inputBuffers->singleValueResultElts;
  const int numScoreValueChannels = (int)inputBuffers->singleScoreValueResultElts;

  // Per-row decode, reproducing the TensorRT backend's post-processing exactly.
  // Outputs are raw logits, so the client applies softmax / tanh / etc.
  float policyProbsTmp[NNPos::MAX_NN_POLICY_SIZE];

  for(int row = 0; row < batchSize; row++) {
    NNOutput* output = outputs[row];
    assert(output->nnXLen == nnXLen);
    assert(output->nnYLen == nnYLen);
    float policyOptimism = (float)inputBufs[row]->policyOptimism;

    // Policy: OutputPolicyPass is [N, numPolicyChannels, 1, 1] and OutputPolicy is [N, numPolicyChannels, H, W].
    {
      const float* policyPassSrcBuf = policyPassData + row * numPolicyChannels;
      const float* policySrcBuf = policyData + row * numPolicyChannels * nnXLen * nnYLen;
      float* policyProbs = output->policyProbs;

      if(numPolicyChannels == 2 || (numPolicyChannels == 4 && modelVersion >= 16)) {
        // NCHW: channel 0 = base logits, channel 1 = optimism logits.
        for(int i = 0; i < nnXLen * nnYLen; i++) {
          float p = policySrcBuf[i];
          float pOpt = policySrcBuf[i + nnXLen * nnYLen];
          policyProbsTmp[i] = p + (pOpt - p) * policyOptimism;
        }
        SymmetryHelpers::copyOutputsWithSymmetry(
          policyProbsTmp, policyProbs, 1, nnYLen, nnXLen, inputBufs[row]->symmetry);
        policyProbs[nnXLen * nnYLen] =
          policyPassSrcBuf[0] + (policyPassSrcBuf[1] - policyPassSrcBuf[0]) * policyOptimism;
      }
      else {
        assert(numPolicyChannels == 1);
        SymmetryHelpers::copyOutputsWithSymmetry(
          policySrcBuf, policyProbs, 1, nnYLen, nnXLen, inputBufs[row]->symmetry);
        policyProbs[nnXLen * nnYLen] = policyPassSrcBuf[0];
      }
    }

    // Value: [N, 3, 1, 1] raw categorical logits (win/loss/noresult).
    {
      assert(numValueChannels == 3);
      output->whiteWinProb = valueData[row * numValueChannels];
      output->whiteLossProb = valueData[row * numValueChannels + 1];
      output->whiteNoResultProb = valueData[row * numValueChannels + 2];
    }

    // Ownership: [N, 1, H, W] raw, inverse-symmetried back to canonical orientation.
    if(output->whiteOwnerMap != NULL) {
      assert(inputBuffers->singleOwnershipResultElts == (size_t)nnXLen * nnYLen);
      const float* ownershipSrcBuf = ownershipData + row * nnXLen * nnYLen;
      SymmetryHelpers::copyOutputsWithSymmetry(
        ownershipSrcBuf, output->whiteOwnerMap, 1, nnYLen, nnXLen, inputBufs[row]->symmetry);
    }

    // ScoreValue: [N, numScoreValueChannels, 1, 1] raw, version-dependent channel interpretation.
    {
      if(modelVersion >= 9) {
        assert(numScoreValueChannels == 6);
        output->whiteScoreMean = scoreValueData[row * numScoreValueChannels];
        output->whiteScoreMeanSq = scoreValueData[row * numScoreValueChannels + 1];
        output->whiteLead = scoreValueData[row * numScoreValueChannels + 2];
        output->varTimeLeft = scoreValueData[row * numScoreValueChannels + 3];
        output->shorttermWinlossError = scoreValueData[row * numScoreValueChannels + 4];
        output->shorttermScoreError = scoreValueData[row * numScoreValueChannels + 5];
      }
      else if(modelVersion >= 8) {
        assert(numScoreValueChannels == 4);
        output->whiteScoreMean = scoreValueData[row * numScoreValueChannels];
        output->whiteScoreMeanSq = scoreValueData[row * numScoreValueChannels + 1];
        output->whiteLead = scoreValueData[row * numScoreValueChannels + 2];
        output->varTimeLeft = scoreValueData[row * numScoreValueChannels + 3];
        output->shorttermWinlossError = 0;
        output->shorttermScoreError = 0;
      }
      else if(modelVersion >= 4) {
        assert(numScoreValueChannels == 2);
        output->whiteScoreMean = scoreValueData[row * numScoreValueChannels];
        output->whiteScoreMeanSq = scoreValueData[row * numScoreValueChannels + 1];
        output->whiteLead = output->whiteScoreMean;
        output->varTimeLeft = 0;
        output->shorttermWinlossError = 0;
        output->shorttermScoreError = 0;
      }
      else if(modelVersion >= 3) {
        assert(numScoreValueChannels == 1);
        output->whiteScoreMean = scoreValueData[row * numScoreValueChannels];
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
  }
}

std::string NeuralNet::getRuntimeBackendDetail(ConfigParser& cfg) {
  // Report which execution provider will run under this backend, matching the parsing in
  // createComputeContext. Different providers are effectively different backends with
  // different numerics and maturity, so e.g. the distributed training server wants to be
  // able to tell them apart. Sanitized to lowercase alphanumeric since this is user
  // config that gets reported verbatim to the server.
  string provider = cfg.contains("onnxProvider") ? Global::toLower(cfg.getString("onnxProvider")) : "cpu";
  string detail;
  for(char c : provider) {
    if((c >= 'a' && c <= 'z') || (c >= '0' && c <= '9'))
      detail += c;
    if(detail.size() >= 32)
      break;
  }
  return detail;
}

void NeuralNet::printDevices() {
  cout << "ONNX backend: device enumeration is execution-provider-specific." << endl;
  cout << "Providers other than cpu need an ONNX Runtime package or build that includes them," << endl;
  cout << "and not all of them are tested. For the status of each provider and what it needs, see:" << endl;
  cout << "https://github.com/lightvector/KataGo/blob/master/Compiling.md#execution-providers" << endl;
  cout << "Set onnxProvider (e.g. 'openvino') plus provider-specific options in the config." << endl;
  cout << endl;
  cout << "OpenVINO provider options:" << endl;
  cout << "  onnxOpenVINODeviceType = GPU            (default; CPU, GPU, NPU, GPU.0, GPU.1, etc.)" << endl;
  cout << "  Also supports OpenVINO multi-device strings:" << endl;
  cout << "    AUTO:GPU,CPU  MULTI:GPU,NPU  HETERO:GPU,CPU" << endl;
  cout << endl;
  cout << "  Multi-device per-thread assignment:" << endl;
  cout << "    onnxOpenVINODeviceTypeThread0 = NPU" << endl;
  cout << "    onnxOpenVINODeviceTypeThread1 = GPU" << endl;
  cout << endl;
  cout << "  Per-device-type provider tuning (optional):" << endl;
  cout << "    onnxOpenVINODeviceConfig_NPU_NumStreams = 4" << endl;
  cout << "    onnxOpenVINODeviceConfig_GPU_NumStreams = 2" << endl;
}

//--------------------------------------------------------------
// The layer-level test entry points are not implemented for this backend. Returning
// false tells the test harness this configuration is unsupported rather than failing.
// The TensorRT backend does the same.

bool NeuralNet::testEvaluateConv(
  const ConvLayerDesc*, int, int, int, bool, bool,
  const std::vector<float>&, std::vector<float>&
) {
  return false;
}

bool NeuralNet::testEvaluateBatchNorm(
  const BatchNormLayerDesc*, int, int, int, bool, bool,
  const std::vector<float>&, const std::vector<float>&, std::vector<float>&
) {
  return false;
}

bool NeuralNet::testEvaluateResidualBlock(
  const ResidualBlockDesc*, int, int, int, bool, bool,
  const std::vector<float>&, const std::vector<float>&, std::vector<float>&
) {
  return false;
}

bool NeuralNet::testEvaluateGlobalPoolingResidualBlock(
  const GlobalPoolingResidualBlockDesc*, int, int, int, bool, bool,
  const std::vector<float>&, const std::vector<float>&, std::vector<float>&
) {
  return false;
}

#endif  // USE_ONNX_BACKEND
