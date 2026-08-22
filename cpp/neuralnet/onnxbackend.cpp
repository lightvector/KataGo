// ONNX Runtime backend for KataGo.
//
// Loads standard .bin.gz KataGo model files, converts the ModelDesc to a serialized ONNX
// ModelProto via the same OnnxModelBuilder that the TensorRT backend uses, and hands the bytes
// to an Ort::Session -- or loads a raw .onnx file directly. Inference runs through ONNX Runtime
// with a configurable execution provider (CPU, OpenVINO, CUDA, TensorRT, MIGraphX, CoreML)
// selected at runtime via the onnxProvider config key. OpenVINO is the primary target for Intel
// GPUs/NPUs, with per-server-thread device assignment so a single process can mix e.g. NPU and
// iGPU inference across threads.
//
// The IO tensor protocol for .bin.gz-sourced graphs is identical to the TensorRT ONNX-emitter
// path (see onnxmodelbuilder.h): four NCHW/NC11 float32 inputs InputSpatial / InputGlobal /
// [InputMeta] / InputMask and five outputs OutputPolicyPass / OutputPolicy / OutputValue /
// OutputScoreValue / OutputOwnership, all raw logits. getOutput below reproduces the TensorRT
// backend's post-processing exactly (per-row optimism blend, inverse-symmetry, version-branched
// score-value decode) so both paths share one decode path. Raw .onnx files may use different
// tensor names; the onnxInput*/onnxOutput* config keys override the defaults for that case.

#ifdef USE_ONNX_BACKEND

#include "../neuralnet/nninterface.h"
#include "../neuralnet/nneval.h"
#include "../neuralnet/nninputs.h"
#include "../neuralnet/modelversion.h"
#include "../neuralnet/onnxmodelbuilder.h"
#include "../dataio/homedata.h"

#include <onnxruntime_cxx_api.h>
#ifdef __APPLE__
#include <coreml_provider_factory.h>
#endif

#include <fstream>
#include <unordered_map>
#include <mutex>
#include <atomic>

using namespace std;

//--------------------------------------------------------------

// Auto-detect modelVersion from introspected channel counts, for raw .onnx files where the
// model version is not encoded in the file. Standard .bin.gz KataGo models do NOT need this --
// version is read from the file itself.
//
// Detection is based on channel-count heuristics assuming V7 inputs (22 spatial + 19 global
// channels) and distinguishing versions by the number of score-value and policy channels:
//   - 4 score-value channels                    -> version 8
//   - 6 score-value channels, 1 policy channel  -> version 10
//   - 6 score-value channels, 2 policy channels -> version 15
// For any other channel-count combination (including non-V7 inputs), falls back to the newest
// implemented model version rather than erroring, since raw .onnx files are typically hand
// exported for a specific known model. Override via 'onnxModelVersion=<N>' in the config for any
// case where the fallback guesses wrong.
static int detectModelVersion(
  int numInputChannels, int numInputGlobalChannels,
  int numPolicyChannels, int numScoreValueChannels,
  int configModelVersion
) {
  if(configModelVersion >= 0)
    return configModelVersion;

  if(numInputChannels == NNInputs::NUM_FEATURES_SPATIAL_V7 &&
     numInputGlobalChannels == NNInputs::NUM_FEATURES_GLOBAL_V7) {
    if(numScoreValueChannels == 6 && numPolicyChannels == 2)
      return 15;
    if(numScoreValueChannels == 6 && numPolicyChannels == 1)
      return 10;
    if(numScoreValueChannels == 4)
      return 8;
  }
  return NNModelVersion::defaultModelVersion;
}

struct LoadedModel {
  ModelDesc modelDesc;
  bool isRawOnnx;
  string rawOnnxBytes;

  // One-time scale8 transform (see maybeApplyScale8), only meaningful for the .bin.gz path. All
  // server threads share this LoadedModel, so whichever ComputeHandle is created first decides
  // for everyone.
  mutable std::atomic<bool> scale8Resolved;
  mutable std::mutex scale8Mutex;

  LoadedModel(const string& fileName, const string& expectedSha256, bool rawOnnx)
    : isRawOnnx(rawOnnx)
  {
    scale8Resolved.store(false);

    if(!rawOnnx) {
      ModelDesc::loadFromFileMaybeGZipped(fileName, modelDesc, expectedSha256);
      return;
    }

    // Read raw .onnx file bytes.
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

    // Create a temporary CPU session to introspect shapes.
    Ort::Env tmpEnv(ORT_LOGGING_LEVEL_WARNING, "KataGoOnnxIntrospect");
    Ort::SessionOptions tmpOpts;
    tmpOpts.SetIntraOpNumThreads(1);
    Ort::Session tmpSession(tmpEnv, rawOnnxBytes.data(), rawOnnxBytes.size(), tmpOpts);

    Ort::AllocatorWithDefaultOptions allocator;

    // Introspect inputs by name (case-insensitive substring match against the names
    // OnnxModelBuilder::build() emits: InputSpatial/InputGlobal/InputMeta/InputMask), falling
    // back to a shape-based heuristic for differently-named graphs.
    int numInputChannels = 0;
    int numInputGlobalChannels = 0;
    int numInputMetaChannels = 0;
    size_t numInputs = tmpSession.GetInputCount();
    for(size_t i = 0; i < numInputs; i++) {
      Ort::AllocatedStringPtr namePtr = tmpSession.GetInputNameAllocated(i, allocator);
      string name = namePtr.get();
      string lowerName = Global::toLower(name);
      auto typeInfo = tmpSession.GetInputTypeInfo(i);
      auto tensorInfo = typeInfo.GetTensorTypeAndShapeInfo();
      auto shape = tensorInfo.GetShape();
      if(lowerName.find("mask") != string::npos) {
        // The on-board mask is its own single-channel input, not a feature-channel input --
        // explicitly ignored here so it can't be mistaken for spatial/global/meta below.
      } else if(lowerName.find("spatial") != string::npos) {
        if(shape.size() >= 2)
          numInputChannels = (int)shape[1];
      } else if(lowerName.find("global") != string::npos) {
        if(shape.size() >= 2)
          numInputGlobalChannels = (int)shape[1];
      } else if(lowerName.find("meta") != string::npos) {
        if(shape.size() >= 2)
          numInputMetaChannels = (int)shape[1];
      } else if(shape.size() == 4) {
        // Shape-based fallback: [N,C,H,W] -- spatial input.
        numInputChannels = (int)shape[1];
      } else if(shape.size() == 2 || shape.size() == 4) {
        // Shape-based fallback: [N,C] or [N,C,1,1] -- first is global, second (if any) is meta.
        if(numInputGlobalChannels == 0)
          numInputGlobalChannels = (int)shape[1];
        else
          numInputMetaChannels = (int)shape[1];
      } else {
        throw StringError(
          "ONNX backend: unrecognized input tensor '" + name +
          "' with " + Global::intToString((int)shape.size()) + "D shape -- "
          "expected tensors named/shaped for spatial, global, meta, or mask."
        );
      }
    }

    // Introspect outputs (case-insensitive; "scorevalue" is checked before "value" since
    // "OutputScoreValue" also contains "value").
    int numPolicyChannels = 0;
    int numValueChannels = 0;
    int numScoreValueChannels = 0;
    int numOwnershipChannels = 0;
    size_t numOutputs = tmpSession.GetOutputCount();
    for(size_t i = 0; i < numOutputs; i++) {
      Ort::AllocatedStringPtr namePtr = tmpSession.GetOutputNameAllocated(i, allocator);
      string name = namePtr.get();
      string lowerName = Global::toLower(name);
      auto typeInfo = tmpSession.GetOutputTypeInfo(i);
      auto tensorInfo = typeInfo.GetTensorTypeAndShapeInfo();
      auto shape = tensorInfo.GetShape();

      if(lowerName.find("policypass") != string::npos) {
        // OutputPolicyPass: [N,C,1,1] -- already counted via OutputPolicy below; skip.
      } else if(lowerName.find("policy") != string::npos) {
        if(shape.size() >= 2)
          numPolicyChannels = (int)shape[1];
      } else if(lowerName.find("scorevalue") != string::npos || lowerName.find("miscvalue") != string::npos) {
        if(shape.size() >= 2)
          numScoreValueChannels = (int)shape[1];
      } else if(lowerName.find("value") != string::npos) {
        if(shape.size() >= 2)
          numValueChannels = (int)shape[1];
      } else if(lowerName.find("ownership") != string::npos) {
        if(shape.size() >= 2)
          numOwnershipChannels = (int)shape[1];
      }
    }

    if(numPolicyChannels == 0 || numValueChannels == 0 ||
       numScoreValueChannels == 0 || numOwnershipChannels == 0) {
      throw StringError(
        "ONNX backend: failed to introspect required outputs from raw .onnx file '" +
        fileName + "'. Found policy=" + Global::intToString(numPolicyChannels) +
        ", value=" + Global::intToString(numValueChannels) +
        ", scoreValue=" + Global::intToString(numScoreValueChannels) +
        ", ownership=" + Global::intToString(numOwnershipChannels) +
        ". Expected output tensor names containing 'policy', 'value', 'scorevalue'/'miscvalue', "
        "'ownership' (case-insensitive substring match), or override names via the onnxOutput* "
        "config keys. Alternatively use a non-raw .bin.gz KataGo model."
      );
    }

    modelDesc.numInputChannels = numInputChannels;
    modelDesc.numInputGlobalChannels = numInputGlobalChannels;
    modelDesc.numInputMetaChannels = numInputMetaChannels;
    modelDesc.numPolicyChannels = numPolicyChannels;
    modelDesc.numValueChannels = numValueChannels;
    modelDesc.numScoreValueChannels = numScoreValueChannels;
    modelDesc.numOwnershipChannels = numOwnershipChannels;

    // Extract filename stem as model name.
    {
      size_t lastSlash = fileName.find_last_of("/\\");
      string basename = (lastSlash != string::npos) ? fileName.substr(lastSlash + 1) : fileName;
      size_t dotPos = basename.find('.');
      modelDesc.name = (dotPos != string::npos) ? basename.substr(0, dotPos) : basename;
    }

    // Model version: auto-detect here; a config override (onnxModelVersion) is applied later in
    // createComputeContext, once cfg is available.
    modelDesc.modelVersion = detectModelVersion(
      numInputChannels, numInputGlobalChannels, numPolicyChannels, numScoreValueChannels, -1
    );

    scale8Resolved.store(true);  // scale8 only applies to modelDesc weights, which raw .onnx has none of.
  }

  // Apply the scale8 FP16-range workaround exactly once per model, unless skipped via
  // onnxSkipScale8. Called from createComputeContext -- see the call site for why it must run
  // there and not later. No-op for the raw .onnx path (scale8Resolved is already true from the
  // constructor above).
  void maybeApplyScale8(bool skip) const {
    std::lock_guard<std::mutex> lock(scale8Mutex);
    if(!scale8Resolved.load()) {
      if(!skip)
        const_cast<LoadedModel*>(this)->modelDesc.applyScale8ToReduceActivations();
      scale8Resolved.store(true);
    }
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

  // OpenVINO EP options. Default device type is NPU (this fork targets Intel NPU first;
  // upstream KataGo PR #1222 defaults to GPU -- override with onnxOpenVINODeviceType if desired).
  string openvinoDeviceType;
  string openvinoDeviceId;  // legacy explicit device_id override; prefer device_type suffixes (e.g. "GPU.1") if unset
  bool openvinoEnableNPUFastCompile;
  string openvinoCacheDir;
  string openvinoPrecision;      // FP16 / FP32 / ACCURACY
  string openvinoNumStreams;     // 1-8
  string openvinoNumOfThreads;   // positive int (infer requests per session)
  string openvinoModelPriority;  // LOW / MEDIUM / HIGH / DEFAULT
  string openvinoLoadConfig;     // JSON for the ORT EP "load_config" provider option (arbitrary OV device config)
  bool openvinoNPUExactBoard;    // NPU-only: build the mask-free exact-board graph (requires fixed-size workloads)

  bool transformerNHWC;  // run the trunk block stack channel-last (NHWC) for transformer models
  bool skipScale8;       // skip the scale8 FP16-range workaround (see createComputeContext)

  // Per-thread device type (index = serverThreadIdx). Filled with openvinoDeviceType by default;
  // individual entries are replaced by onnxOpenVINODeviceTypeThread<N>. Allows mixing e.g. NPU
  // and iGPU inference across server threads within the same process.
  std::vector<std::string> perThreadDeviceType;

  // Per-device-type EP option overrides.
  // Outer key = short device name ("NPU", "GPU", "CPU"). Inner key = ORT EP option key.
  std::unordered_map<std::string, std::unordered_map<std::string, std::string>> deviceConfigOverrides;

  // Configurable input/output node names. Defaults match the node names emitted by the shared
  // OnnxModelBuilder::build() (see onnxmodelbuilder.cpp), also consumed by the TensorRT backend.
  // Raw .onnx models with different tensor names can override these via the onnxInput*/onnxOutput*
  // config keys.
  string inputMaskName;
  string inputSpatialName;
  string inputGlobalName;
  string inputMetaName;
  string outputPolicyPassName;
  string outputPolicyName;
  string outputValueName;
  string outputMiscvalueName;  // aka OutputScoreValue
  string outputOwnershipName;

  // Config override for model version, for raw .onnx files where detectModelVersion guesses
  // wrong (-1 means auto-detect).
  int configModelVersion;

  ComputeContext(int xLen, int yLen)
    : env(ORT_LOGGING_LEVEL_WARNING, "KataGoOnnx"),
      nnXLen(xLen),
      nnYLen(yLen),
      providerName("cpu"),
      openvinoDeviceType("NPU"),
      openvinoDeviceId(""),
      openvinoEnableNPUFastCompile(false),
      openvinoCacheDir(""),
      openvinoPrecision(""),
      openvinoNumStreams(""),
      openvinoNumOfThreads(""),
      openvinoModelPriority(""),
      openvinoLoadConfig(""),
      openvinoNPUExactBoard(false),
      transformerNHWC(true),
      skipScale8(false),
      inputMaskName("InputMask"),
      inputSpatialName("InputSpatial"),
      inputGlobalName("InputGlobal"),
      inputMetaName("InputMeta"),
      outputPolicyPassName("OutputPolicyPass"),
      outputPolicyName("OutputPolicy"),
      outputValueName("OutputValue"),
      outputMiscvalueName("OutputScoreValue"),
      outputOwnershipName("OutputOwnership"),
      configModelVersion(-1)
  {}
};

// Probe whether the OpenVINO EP can actually use a given device_type.
//
// AppendExecutionProvider_OpenVINO_V2 resolves and validates device_type immediately, throwing
// "[OpenVINO] Device X is not available" for a device this machine/package cannot provide, so this
// is a real availability signal and needs no model or session. (ONNX Runtime's own device
// enumeration is NOT usable for this: on a provider-bridge OpenVINO build it reports CPU only,
// even on machines that do have an NPU and an iGPU.)
static bool openvinoDeviceAvailable(const string& deviceType) {
  try {
    Ort::SessionOptions probeOpts;
    std::unordered_map<std::string, std::string> probeEpOpts;
    probeEpOpts["device_type"] = deviceType;
    probeOpts.AppendExecutionProvider_OpenVINO_V2(probeEpOpts);
    return true;
  }
  catch(const std::exception&) {
    return false;
  }
}

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
  // The emitted ONNX graph is fp32; inference precision is chosen internally by the execution
  // provider (e.g. OpenVINO downcasts to FP16 per onnxOpenVINOPrecision). KataGo's global useFP16
  // flag therefore cannot be honored here -- fail loudly instead of silently ignoring a request.
  if(useFP16Mode == enabled_t::True)
    throw StringError(
      "ONNX backend: the global useFP16 flag is not supported and cannot be honored. "
      "Precision is controlled by the execution provider; for the OpenVINO provider set "
      "onnxOpenVINOPrecision (e.g. FP16/FP32/ACCURACY). Leave useFP16 unset or set it to false/auto.");

  string providerName = cfg.contains("onnxProvider") ? cfg.getString("onnxProvider") : "cpu";
  providerName = Global::toLower(providerName);
  if(providerName != "cpu" && providerName != "openvino" && providerName != "cuda" &&
     providerName != "tensorrt" && providerName != "migraphx" && providerName != "coreml")
    throw StringError(
      "ONNX backend: unknown onnxProvider '" + providerName +
      "', expected one of 'cpu','openvino','cuda','tensorrt','migraphx','coreml'");

  if(logger != NULL)
    logger->write("ONNX backend: creating compute context for " +
                   Global::intToString(nnXLen) + "x" + Global::intToString(nnYLen) +
                   " with provider '" + providerName + "'");

  ComputeContext* ctx = new ComputeContext(nnXLen, nnYLen);
  ctx->providerName = providerName;

  // OpenVINO EP options.
  // When onnxOpenVINODeviceType is left unset we try NPU first and fall back to GPU then CPU at
  // session-creation time (see ComputeHandle). We deliberately do NOT try to enumerate hardware up
  // front: ONNX Runtime's device enumeration only reports devices for EPs it has registered, and
  // on a provider-bridge OpenVINO build it reports CPU only even on machines that do have an NPU
  // and an iGPU, so trusting it would silently downgrade exactly the setups this fork targets.
  // Letting a real session creation succeed or fail is the only reliable signal.
  if(cfg.contains("onnxOpenVINODeviceType")) {
    ctx->openvinoDeviceType = cfg.getString("onnxOpenVINODeviceType");
  }
  else if(providerName == "openvino") {
    // No device configured: pick the first one this machine actually provides, preferring the NPU
    // (this fork's target), then a GPU, then CPU. This makes a GPU-only machine work out of the box
    // instead of failing with "Device NPU is not available". If none of them probe as available we
    // keep the NPU default so the real error surfaces at session creation rather than being masked.
    static const char* candidates[] = {"NPU", "GPU", "CPU"};
    for(const char* cand : candidates) {
      if(openvinoDeviceAvailable(cand)) {
        ctx->openvinoDeviceType = cand;
        break;
      }
    }
    if(logger != NULL)
      logger->write(
        string("ONNX backend: onnxOpenVINODeviceType not set, auto-selected '") +
        ctx->openvinoDeviceType + "'");
  }
  if(cfg.contains("onnxOpenVINODeviceId")) ctx->openvinoDeviceId = cfg.getString("onnxOpenVINODeviceId");
  if(cfg.contains("onnxOpenVINOEnableNPUFastCompile"))
    ctx->openvinoEnableNPUFastCompile = cfg.getBool("onnxOpenVINOEnableNPUFastCompile");
  // OpenVINO blob cache. Defaults to <homeDataDir>/openvino_cache so caching works out of the box;
  // onnxOpenVINOCacheDir overrides the location explicitly.
  if(cfg.contains("onnxOpenVINOCacheDir"))
    ctx->openvinoCacheDir = cfg.getString("onnxOpenVINOCacheDir");
  else if(providerName == "openvino")
    ctx->openvinoCacheDir = HomeData::getHomeDataDir(true, homeDataDirOverride) + "/openvino_cache";
  if(cfg.contains("onnxOpenVINOPrecision")) ctx->openvinoPrecision = cfg.getString("onnxOpenVINOPrecision");
  if(cfg.contains("onnxOpenVINONumStreams")) ctx->openvinoNumStreams = cfg.getString("onnxOpenVINONumStreams");
  if(cfg.contains("onnxOpenVINONumOfThreads")) ctx->openvinoNumOfThreads = cfg.getString("onnxOpenVINONumOfThreads");
  if(cfg.contains("onnxOpenVINOModelPriority")) ctx->openvinoModelPriority = cfg.getString("onnxOpenVINOModelPriority");
  // Arbitrary OpenVINO device config as JSON, passed through to the ORT EP "load_config" provider
  // option (e.g. {"NPU":{"NPU_COMPILATION_MODE_PARAMS":"optimization-level=2 performance-hint-override=latency"}}).
  if(cfg.contains("onnxOpenVINOLoadConfig")) ctx->openvinoLoadConfig = cfg.getString("onnxOpenVINOLoadConfig");
  // NPU-friendly exact-board build: when the session targets an NPU device, emit the mask-free
  // exact-board graph (drops InputMask and all attention mask-bias adds, ~6% faster measured on
  // b11-class transformer models). Only valid when every query uses the exact nnXLen x nnYLen
  // board size (e.g. GTP play on a fixed board size); do NOT enable for mixed-size analysis.
  ctx->openvinoNPUExactBoard = cfg.contains("onnxOpenVINONPUExactBoard") ? cfg.getBool("onnxOpenVINONPUExactBoard") : false;

  // --- Per-thread device type assignment ---
  // Pre-parse onnxOpenVINODeviceTypeThread<N> keys so ComputeHandle can look up the device type
  // for each server thread without reaching back into ConfigParser.
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

  // Trunk layout for transformer models, defaulted per device.
  //
  // NHWC matches the TensorRT backend's trtTransformerNHWC default and is faster on the OpenVINO
  // GPU plugin, but it is markedly SLOWER on the OpenVINO NPU plugin, which prefers the NCHW
  // trunk. Measured on b11c768h12nbt3 at 19x19, single server thread:
  //   NPU: NHWC 3.8 visits/s vs NCHW 10.2 visits/s   (NCHW ~2.7x faster)
  //   GPU: NHWC 14.3 nnEvals/s vs NCHW 11.2 nnEvals/s (NHWC ~1.3x faster)
  // So default to NCHW whenever any OpenVINO server thread targets an NPU (the NPU's penalty for
  // the wrong layout is far larger than the GPU's, which also makes NCHW the better compromise for
  // hybrid NPU+GPU setups, since this setting is context-wide), and NHWC otherwise.
  // Ignored entirely for models without transformer blocks.
  {
    // A device string counts as "may run on an NPU" if it names one, or if it is a bare AUTO,
    // where OpenVINO picks the device at runtime and could land on the NPU. Guessing NCHW for a
    // bare AUTO costs a GPU ~1.3x if wrong but saves an NPU ~2.7x if right, so it is the lower
    // expected loss.
    auto mayBeNPU = [](const string& dev) {
      string upper = Global::toUpper(Global::trim(dev));
      if(upper.find("NPU") != string::npos)
        return true;
      return upper == "AUTO";
    };
    bool anyNPU = false;
    if(ctx->providerName == "openvino") {
      if(mayBeNPU(ctx->openvinoDeviceType))
        anyNPU = true;
      for(const string& dev : ctx->perThreadDeviceType) {
        if(mayBeNPU(dev))
          anyNPU = true;
      }
    }
    ctx->transformerNHWC =
      cfg.contains("onnxTransformerNHWC") ? cfg.getBool("onnxTransformerNHWC") : !anyNPU;
  }

  // Skip the scale8 FP16-range workaround (default false = apply it). scale8 rescales convnet
  // activations to 1/8 and compensates with MISH_SCALE8 subgraphs so activations stay well inside
  // the FP16 range that execution providers may infer in. The cost is that the emitted
  // Mul(x,8)->Softplus->Tanh->Mul(x,.) chain no longer matches the canonical Mish pattern that
  // OpenVINO fuses into a single op, so it runs unfused; on the NPU that is ~3.7x slower on convnet
  // models (11 vs 41 visits/s measured on b28c512nbt). Set true to trade the FP16 headroom for that
  // speed, or set onnxOpenVINOPrecision=FP32 and skip it safely.
  ctx->skipScale8 = cfg.contains("onnxSkipScale8") ? cfg.getBool("onnxSkipScale8") : false;

  // Apply the scale8 transform HERE, not lazily at compute-handle creation.
  // applyScale8ToReduceActivations() multiplies postProcessParams.outputScaleMultiplier by 8 to
  // compensate for the 1/8-scaled graph outputs, and NNEvaluator snapshots postProcessParams
  // immediately after createComputeContext returns (see nneval.cpp). Applying it any later would
  // leave NNEvaluator decoding 1/8-scale outputs with a stale multiplier of 1, silently producing
  // wrong winrates/leads. This also happens-before the server threads spawn and read modelDesc in
  // OnnxModelBuilder::build().
  loadedModel->maybeApplyScale8(ctx->skipScale8);

  // --- Raw .onnx compatibility: configurable node names, model version override ---
  if(cfg.contains("onnxInputMask")) ctx->inputMaskName = cfg.getString("onnxInputMask");
  if(cfg.contains("onnxInputSpatial")) ctx->inputSpatialName = cfg.getString("onnxInputSpatial");
  if(cfg.contains("onnxInputGlobal")) ctx->inputGlobalName = cfg.getString("onnxInputGlobal");
  if(cfg.contains("onnxInputMeta")) ctx->inputMetaName = cfg.getString("onnxInputMeta");
  if(cfg.contains("onnxOutputPolicyPass")) ctx->outputPolicyPassName = cfg.getString("onnxOutputPolicyPass");
  if(cfg.contains("onnxOutputPolicy")) ctx->outputPolicyName = cfg.getString("onnxOutputPolicy");
  if(cfg.contains("onnxOutputValue")) ctx->outputValueName = cfg.getString("onnxOutputValue");
  if(cfg.contains("onnxOutputMiscvalue")) ctx->outputMiscvalueName = cfg.getString("onnxOutputMiscvalue");
  if(cfg.contains("onnxOutputOwnership")) ctx->outputOwnershipName = cfg.getString("onnxOutputOwnership");
  if(cfg.contains("onnxModelVersion")) {
    int v = Global::stringToInt(cfg.getString("onnxModelVersion"));
    if(v >= 0)
      ctx->configModelVersion = v;
  }

  return ctx;
}

void NeuralNet::freeComputeContext(ComputeContext* computeContext) {
  delete computeContext;
}

//--------------------------------------------------------------
// Helper: extract a short device name from an OpenVINO device_type string for matching
// onnxOpenVINODeviceConfig_<Device>_<Option> keys.
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

  ComputeHandle(
    ComputeContext* context, const LoadedModel& loadedModel, Logger* logger,
    int deviceIdxForThread, int serverThreadIdx
  )
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
    // Apply config model version override if set (raw .onnx only; .bin.gz models carry their own).
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
      // NPU-friendly exact-board build: when this session targets an NPU device and the user
      // opted in via onnxOpenVINONPUExactBoard, emit the mask-free exact-board graph (the
      // builder drops InputMask and every attention mask-bias add). Non-NPU targets are built
      // exactly as before.
      bool exactBoardBuild = false;
      if(ctx->openvinoNPUExactBoard && ctx->providerName == "openvino") {
        string devForThread = ctx->openvinoDeviceType;
        if(serverThreadIdx >= 0 && serverThreadIdx < (int)ctx->perThreadDeviceType.size())
          devForThread = ctx->perThreadDeviceType[serverThreadIdx];
        exactBoardBuild = (devForThread.find("NPU") != string::npos);
      }
      // Reuse the same ONNX emitter as the TensorRT backend. The serialized ModelProto is a
      // standard ONNX graph that Ort::Session can parse directly; the TRT-only FP32 node-name
      // lists in the Result are ignored (ORT has no per-node precision API).
      OnnxModelBuilder::Result onnxResult = OnnxModelBuilder::build(
        loadedModel.modelDesc, ctx->nnXLen, ctx->nnYLen, /*requireExactNNLen=*/exactBoardBuild, ctx->transformerNHWC, logger);
      builtOnnxBytes = onnxResult.serializedModel;
      if(logger != NULL)
        logger->write("ONNX backend: ONNX graph built (" + Global::uint64ToString(builtOnnxBytes.size()) + " bytes)");
      onnxData = builtOnnxBytes.data();
      onnxSize = builtOnnxBytes.size();

      // Dump the ONNX model to a file when KATAGO_DUMP_ONNX is set (debug aid).
      const char* dumpPath = getenv("KATAGO_DUMP_ONNX");
      if(dumpPath != nullptr && dumpPath[0] != '\0') {
        ofstream dumpFile(dumpPath, ios::binary);
        if(dumpFile.is_open()) {
          dumpFile.write(builtOnnxBytes.data(), (streamsize)builtOnnxBytes.size());
          dumpFile.close();
          if(logger != NULL)
            logger->write(string("ONNX backend: dumped ONNX model to ") + dumpPath +
                          " (" + Global::uint64ToString(builtOnnxBytes.size()) + " bytes)");
        } else if(logger != NULL) {
          logger->write(string("ONNX backend: WARNING - could not open dump path ") + dumpPath);
        }
      }
    }

    if(logger != NULL)
      logger->write("ONNX backend: creating session...");

    Ort::SessionOptions sessionOpts;

    const string& provider = ctx->providerName;
    if(provider == "coreml") {
#ifdef __APPLE__
      uint32_t coremlFlags = COREML_FLAG_CREATE_ML_PROGRAM;
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
      // num_of_threads provider option; ORT's intra-op pool is left with only the few EP-external
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

      // Device selection: an explicit onnxOpenVINODeviceId takes the legacy device_id provider
      // option (for older ORT/OpenVINO builds without device_type suffix support). Otherwise, a
      // nonzero per-thread device index (from onnxDeviceToUse*) is appended to device_type as an
      // OpenVINO device suffix, e.g. GPU -> GPU.1; device_type is how modern OpenVINO EP builds
      // select among multiple devices of the same kind.
      string deviceType = threadDeviceType;
      if(!ctx->openvinoDeviceId.empty()) {
        openvinoOpts["device_id"] = ctx->openvinoDeviceId;
      } else if(deviceIdxForThread > 0 && deviceType.find('.') == string::npos && deviceType.find(':') == string::npos) {
        deviceType += "." + Global::intToString(deviceIdxForThread);
      }
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
      setIfNotEmpty("load_config",    ctx->openvinoLoadConfig);
      if(ctx->openvinoEnableNPUFastCompile)
        openvinoOpts["enable_npu_fast_compile"] = "true";

      // Some ORT OpenVINO builds reject optional keys (cache_dir, precision, num_streams,
      // num_of_threads, model_priority, load_config, enable_npu_fast_compile, device_id). Retry
      // with only the core device_type key if optional keys are rejected, so that e.g. setting
      // onnxOpenVINOCacheDir on an EP that doesn't support it degrades gracefully instead of
      // crashing.
      static const char* optionalKeys[] = {
        "cache_dir", "precision", "num_streams", "num_of_threads", "model_priority", "load_config",
        "enable_npu_fast_compile", "device_id"
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
          "ONNX backend: OpenVINO EP enabled for thread " + Global::intToString(serverThreadIdx) +
          ", device_type=" + deviceType + extras
        );
      }
    }
    else if(provider == "cpu" || provider.empty()) {
      if(logger != NULL)
        logger->write("ONNX backend: using CPU execution provider");
    }
    else {
      throw StringError("ONNX backend: unknown onnxProvider '" + provider + "'");
    }

    // Create session from in-memory bytes.
    session = std::make_unique<Ort::Session>(ctx->env, onnxData, onnxSize, sessionOpts);

    // Query and store graph input names.
    Ort::AllocatorWithDefaultOptions allocator;
    size_t numInputs = session->GetInputCount();
    for(size_t i = 0; i < numInputs; i++) {
      Ort::AllocatedStringPtr name = session->GetInputNameAllocated(i, allocator);
      inputNames.push_back(name.get());
    }
    for(auto& n : inputNames)
      inputNamePtrs.push_back(n.c_str());

    // Query and store graph output names.
    size_t numOutputs = session->GetOutputCount();
    for(size_t i = 0; i < numOutputs; i++) {
      Ort::AllocatedStringPtr name = session->GetOutputNameAllocated(i, allocator);
      outputNames.push_back(name.get());
    }
    for(auto& n : outputNames)
      outputNamePtrs.push_back(n.c_str());

    if(logger != NULL) {
      string inList = "ONNX backend: graph input order:";
      for(size_t i = 0; i < inputNames.size(); i++)
        inList += " [" + Global::uint64ToString(i) + "]" + inputNames[i];
      logger->write(inList);
      string outList = "ONNX backend: graph output order:";
      for(size_t i = 0; i < outputNames.size(); i++)
        outList += " [" + Global::uint64ToString(i) + "]" + outputNames[i];
      logger->write(outList);
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
  // maxBatchSize is intentionally ignored: ONNX Runtime sessions support dynamic batch sizes.
  // The InputBuffers maxBatchSize field still enforces the upper bound at inference time.
  (void)maxBatchSize;
  // requireExactNNLen is intentionally ignored: the emitted graph handles dynamic board shapes
  // transparently up to the configured nnXLen x nnYLen (see OnnxModelBuilder::build).
  (void)requireExactNNLen;
  if(inputsUseNHWC)
    throw StringError("ONNX backend: inputsUseNHWC = true not supported, must use NCHW");

  if(logger != NULL) {
    logger->write("ONNX backend thread " + Global::intToString(serverThreadIdx) +
                  ": Model version " + Global::intToString(loadedModel->modelDesc.modelVersion));
    logger->write("ONNX backend thread " + Global::intToString(serverThreadIdx) +
                  ": Model name: " + loadedModel->modelDesc.name);
    string deviceInfo =
      context->providerName == "openvino"
      ? (serverThreadIdx >= 0 && serverThreadIdx < (int)context->perThreadDeviceType.size()
         ? context->perThreadDeviceType[serverThreadIdx]
         : context->openvinoDeviceType)
      : Global::intToString(gpuIdxForThisThread);
    logger->write("ONNX backend thread " + Global::intToString(serverThreadIdx) +
                  ": provider=" + context->providerName + " deviceIdx=" + deviceInfo);
  }

  return new ComputeHandle(context, *loadedModel, logger, gpuIdxForThisThread, serverThreadIdx);
}

void NeuralNet::freeComputeHandle(ComputeHandle* computeHandle) {
  delete computeHandle;
}

bool NeuralNet::isUsingFP16(const ComputeHandle* handle) {
  (void)handle;
  // The emitted ONNX graph is fp32; precision is delegated to the execution provider (e.g.
  // OpenVINO may downcast internally), so from KataGo's perspective this is fp32.
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

  vector<float> maskInput;
  vector<float> spatialInput;
  vector<float> globalInput;
  vector<float> metaInput;

  InputBuffers(const LoadedModel* loadedModel, int maxBatchSz, int nnXLen, int nnYLen) {
    const ModelDesc& m = loadedModel->modelDesc;

    if(nnXLen > NNPos::MAX_BOARD_LEN)
      throw StringError(Global::strprintf("nnXLen (%d) is greater than NNPos::MAX_BOARD_LEN (%d)", nnXLen, NNPos::MAX_BOARD_LEN));
    if(nnYLen > NNPos::MAX_BOARD_LEN)
      throw StringError(Global::strprintf("nnYLen (%d) is greater than NNPos::MAX_BOARD_LEN (%d)", nnYLen, NNPos::MAX_BOARD_LEN));

    maxBatchSize = maxBatchSz;
    singleMaskElts = (size_t)nnXLen * nnYLen;
    singleInputElts = (size_t)m.numInputChannels * nnXLen * nnYLen;
    singleInputGlobalElts = (size_t)m.numInputGlobalChannels;
    singleInputMetaElts = (size_t)m.numInputMetaChannels;

    maskInput.assign(singleMaskElts * maxBatchSize, 0.0f);
    spatialInput.assign(singleInputElts * maxBatchSize, 0.0f);
    globalInput.assign(singleInputGlobalElts * maxBatchSize, 0.0f);
    if(m.numInputMetaChannels > 0)
      metaInput.assign(singleInputMetaElts * maxBatchSize, 0.0f);
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

// Find the index of a name in a vector, checking multiple alternatives.
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
  const int nnXLen = computeHandle->ctx->nnXLen;
  const int nnYLen = computeHandle->ctx->nnYLen;
  const int numSpatialFeatures = computeHandle->numInputChannels;
  const int numGlobalFeatures = computeHandle->numInputGlobalChannels;
  const int numPolicyChannels = computeHandle->numPolicyChannels;
  const int spatialPolicyLen = nnXLen * nnYLen;

  // Fill host input buffers, mirroring the TensorRT backend:
  //  - global/meta are straight copies (no symmetry)
  //  - spatial is symmetry-transformed (NCHW, useNHWC=false)
  //  - mask = channel 0 of the symmetry-transformed spatial input
  for(int nIdx = 0; nIdx < batchSize; nIdx++) {
    float* rowMaskInput = inputBuffers->maskInput.data() + (inputBuffers->singleMaskElts * nIdx);
    float* rowSpatialInput = inputBuffers->spatialInput.data() + (inputBuffers->singleInputElts * nIdx);
    float* rowGlobalInput = inputBuffers->globalInput.data() + (inputBuffers->singleInputGlobalElts * nIdx);

    const float* rowGlobal = inputBufs[nIdx]->rowGlobalBuf.data();
    const float* rowSpatial = inputBufs[nIdx]->rowSpatialBuf.data();
    std::copy(rowGlobal, rowGlobal + numGlobalFeatures, rowGlobalInput);
    SymmetryHelpers::copyInputsWithSymmetry(rowSpatial, rowSpatialInput, 1, nnYLen, nnXLen, numSpatialFeatures, false, inputBufs[nIdx]->symmetry);
    std::copy(rowSpatialInput, rowSpatialInput + inputBuffers->singleMaskElts, rowMaskInput);

    if(computeHandle->numInputMetaChannels > 0) {
      float* rowMetaInput = inputBuffers->metaInput.data() + (inputBuffers->singleInputMetaElts * nIdx);
      const float* rowMeta = inputBufs[nIdx]->rowMetaBuf.data();
      std::copy(rowMeta, rowMeta + computeHandle->numInputMetaChannels, rowMetaInput);
    }
  }

  // Create ONNX tensors.
  Ort::MemoryInfo memInfo = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

  std::array<int64_t, 4> maskShape = {batchSize, 1, nnYLen, nnXLen};
  Ort::Value maskTensor = Ort::Value::CreateTensor<float>(
    memInfo, inputBuffers->maskInput.data(), inputBuffers->singleMaskElts * batchSize,
    maskShape.data(), maskShape.size()
  );

  std::array<int64_t, 4> spatialShape = {batchSize, numSpatialFeatures, nnYLen, nnXLen};
  Ort::Value spatialTensor = Ort::Value::CreateTensor<float>(
    memInfo, inputBuffers->spatialInput.data(), inputBuffers->singleInputElts * batchSize,
    spatialShape.data(), spatialShape.size()
  );

  std::array<int64_t, 4> globalShape = {batchSize, numGlobalFeatures, 1, 1};
  Ort::Value globalTensor = Ort::Value::CreateTensor<float>(
    memInfo, inputBuffers->globalInput.data(), inputBuffers->singleInputGlobalElts * batchSize,
    globalShape.data(), globalShape.size()
  );

  // Match inputs to graph nodes using configured node names.
  const ComputeContext* ctx = computeHandle->ctx;
  int spatialIdx = findNameIndex(computeHandle->inputNames, {ctx->inputSpatialName});
  int globalIdx = findNameIndex(computeHandle->inputNames, {ctx->inputGlobalName});
  if(spatialIdx < 0 || globalIdx < 0)
    throw StringError("ONNX backend: could not find expected input names");

  // The mask input may be absent from a hand-exported raw .onnx graph -- only bind it if the
  // session actually declares it. Graphs built by OnnxModelBuilder::build() (.bin.gz path) always
  // declare it.
  int maskIdx = findNameIndex(computeHandle->inputNames, {ctx->inputMaskName});

  int metaIdx = -1;
  Ort::Value metaTensor(nullptr);
  if(computeHandle->numInputMetaChannels > 0) {
    metaIdx = findNameIndex(computeHandle->inputNames, {ctx->inputMetaName});
    if(metaIdx < 0)
      throw StringError("ONNX backend: model has metadata channels but could not find the InputMeta node");
    std::array<int64_t, 4> metaShape = {batchSize, computeHandle->numInputMetaChannels, 1, 1};
    metaTensor = Ort::Value::CreateTensor<float>(
      memInfo, inputBuffers->metaInput.data(), inputBuffers->singleInputMetaElts * batchSize,
      metaShape.data(), metaShape.size()
    );
  }

  vector<Ort::Value> inputTensors;
  inputTensors.reserve(computeHandle->inputNames.size());
  for(size_t i = 0; i < computeHandle->inputNames.size(); i++) {
    if((int)i == maskIdx)
      inputTensors.push_back(std::move(maskTensor));
    else if((int)i == spatialIdx)
      inputTensors.push_back(std::move(spatialTensor));
    else if((int)i == globalIdx)
      inputTensors.push_back(std::move(globalTensor));
    else if((int)i == metaIdx)
      inputTensors.push_back(std::move(metaTensor));
    else {
      throw StringError("ONNX backend: unexpected input node '" + computeHandle->inputNames[i] +
                         "' -- only mask, spatial, global, and meta inputs are supported");
    }
  }

  // Run inference.
  auto outputTensors = computeHandle->session->Run(
    Ort::RunOptions{nullptr},
    computeHandle->inputNamePtrs.data(),
    inputTensors.data(),
    inputTensors.size(),
    computeHandle->outputNamePtrs.data(),
    computeHandle->outputNamePtrs.size()
  );

  // Find output indices using configured node names.
  int policyPassOutputIdx = findNameIndex(computeHandle->outputNames, {ctx->outputPolicyPassName});
  int policyOutputIdx = findNameIndex(computeHandle->outputNames, {ctx->outputPolicyName});
  int valueOutputIdx = findNameIndex(computeHandle->outputNames, {ctx->outputValueName});
  int scoreValueOutputIdx = findNameIndex(computeHandle->outputNames, {ctx->outputMiscvalueName});
  int ownershipOutputIdx = findNameIndex(computeHandle->outputNames, {ctx->outputOwnershipName});

  if(policyPassOutputIdx < 0)
    throw StringError("ONNX backend: could not find policy-pass output node '" + ctx->outputPolicyPassName + "'");
  if(policyOutputIdx < 0)
    throw StringError("ONNX backend: could not find policy output node '" + ctx->outputPolicyName + "'");
  if(valueOutputIdx < 0)
    throw StringError("ONNX backend: could not find value output node '" + ctx->outputValueName + "'");
  if(scoreValueOutputIdx < 0)
    throw StringError("ONNX backend: could not find score-value output node '" + ctx->outputMiscvalueName + "'");
  if(ownershipOutputIdx < 0)
    throw StringError("ONNX backend: could not find ownership output node '" + ctx->outputOwnershipName + "'");

  const float* policyPassData = outputTensors[policyPassOutputIdx].GetTensorData<float>();
  const float* policyData = outputTensors[policyOutputIdx].GetTensorData<float>();
  const float* valueData = outputTensors[valueOutputIdx].GetTensorData<float>();
  const float* scoreValueData = outputTensors[scoreValueOutputIdx].GetTensorData<float>();
  const float* ownershipData = outputTensors[ownershipOutputIdx].GetTensorData<float>();

  assert(policyPassData != nullptr);
  assert(policyData != nullptr);
  assert(valueData != nullptr);
  assert(scoreValueData != nullptr);
  assert(ownershipData != nullptr);
  assert((int)outputs.size() == batchSize);

  float policyProbsTmp[NNPos::MAX_NN_POLICY_SIZE];

  for(int row = 0; row < batchSize; row++) {
    NNOutput* output = outputs[row];
    assert(output->nnXLen == nnXLen);
    assert(output->nnYLen == nnYLen);
    float policyOptimism = (float)inputBufs[row]->policyOptimism;

    // Policy: OutputPolicyPass is [N,numPolicyChannels,1,1]; OutputPolicy is [N,numPolicyChannels,H,W].
    {
      const float* policyPassRowBase = policyPassData + row * numPolicyChannels;
      const float* policyRowBase = policyData + row * numPolicyChannels * spatialPolicyLen;
      float* policyProbs = output->policyProbs;

      if(numPolicyChannels == 2 || (numPolicyChannels == 4 && computeHandle->modelVersion >= 16)) {
        // Channel 0 = base logits, channel 1 = optimism logits.
        const float* ch0 = policyRowBase;
        const float* ch1 = policyRowBase + spatialPolicyLen;
        for(int i = 0; i < spatialPolicyLen; i++) {
          float p = ch0[i];
          float pOpt = ch1[i];
          policyProbsTmp[i] = p + (pOpt - p) * policyOptimism;
        }
        SymmetryHelpers::copyOutputsWithSymmetry(policyProbsTmp, policyProbs, 1, nnYLen, nnXLen, inputBufs[row]->symmetry);
        policyProbs[spatialPolicyLen] = policyPassRowBase[0] + (policyPassRowBase[1] - policyPassRowBase[0]) * policyOptimism;
      } else {
        assert(numPolicyChannels == 1);
        SymmetryHelpers::copyOutputsWithSymmetry(policyRowBase, policyProbs, 1, nnYLen, nnXLen, inputBufs[row]->symmetry);
        policyProbs[spatialPolicyLen] = policyPassRowBase[0];
      }
    }

    // Value: [N,3] raw categorical logits (win/loss/noresult).
    {
      int numVC = computeHandle->numValueChannels;
      assert(numVC == 3);
      output->whiteWinProb = valueData[row * numVC];
      output->whiteLossProb = valueData[row * numVC + 1];
      output->whiteNoResultProb = valueData[row * numVC + 2];
    }

    // ScoreValue: [N,numScoreValueChannels] raw, version-dependent channel interpretation.
    {
      int numScoreValueChannels = computeHandle->numScoreValueChannels;
      if(computeHandle->modelVersion >= 9) {
        assert(numScoreValueChannels >= 6);
        output->whiteScoreMean = scoreValueData[row * numScoreValueChannels];
        output->whiteScoreMeanSq = scoreValueData[row * numScoreValueChannels + 1];
        output->whiteLead = scoreValueData[row * numScoreValueChannels + 2];
        output->varTimeLeft = scoreValueData[row * numScoreValueChannels + 3];
        output->shorttermWinlossError = scoreValueData[row * numScoreValueChannels + 4];
        output->shorttermScoreError = scoreValueData[row * numScoreValueChannels + 5];
      }
      else if(computeHandle->modelVersion >= 8) {
        assert(numScoreValueChannels >= 4);
        output->whiteScoreMean = scoreValueData[row * numScoreValueChannels];
        output->whiteScoreMeanSq = scoreValueData[row * numScoreValueChannels + 1];
        output->whiteLead = scoreValueData[row * numScoreValueChannels + 2];
        output->varTimeLeft = scoreValueData[row * numScoreValueChannels + 3];
        output->shorttermWinlossError = 0;
        output->shorttermScoreError = 0;
      }
      else if(computeHandle->modelVersion >= 4) {
        assert(numScoreValueChannels >= 2);
        output->whiteScoreMean = scoreValueData[row * numScoreValueChannels];
        output->whiteScoreMeanSq = scoreValueData[row * numScoreValueChannels + 1];
        output->whiteLead = output->whiteScoreMean;
        output->varTimeLeft = 0;
        output->shorttermWinlossError = 0;
        output->shorttermScoreError = 0;
      }
      else if(computeHandle->modelVersion >= 3) {
        assert(numScoreValueChannels >= 1);
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

    // Ownership: [N,1,H,W] raw; inverse-symmetry back to canonical orientation.
    if(output->whiteOwnerMap != NULL) {
      assert(computeHandle->numOwnershipChannels == 1);
      const float* ownershipRowBuf = ownershipData + row * nnXLen * nnYLen;
      SymmetryHelpers::copyOutputsWithSymmetry(ownershipRowBuf, output->whiteOwnerMap, 1, nnYLen, nnXLen, inputBufs[row]->symmetry);
    }
  }
}

void NeuralNet::printDevices() {
  // Deliberately not enumerating devices here: ONNX Runtime only reports devices for EPs it has
  // registered, and a provider-bridge OpenVINO build reports CPU only even when an NPU and iGPU are
  // present, so any listing we printed would be actively misleading. Use OpenVINO's own
  // hello_query_device sample, or just let the auto fallback below pick a device.
  cout << "ONNX backend: device selection is execution-provider-specific." << endl;
  cout << "Set onnxProvider (e.g. 'openvino') plus provider-specific options in the config." << endl;
  cout << endl;
  cout << "OpenVINO EP options:" << endl;
  cout << "  onnxOpenVINODeviceType = NPU | GPU | CPU | NPU.0 | GPU.1 | AUTO:GPU,CPU | MULTI:NPU,GPU | HETERO:NPU,CPU" << endl;
  cout << "  If left unset, KataGo tries NPU, then GPU, then CPU, and uses the first that works." << endl;
  cout << endl;
  cout << "  Multi-device per-thread assignment:" << endl;
  cout << "    onnxOpenVINODeviceTypeThread0 = NPU" << endl;
  cout << "    onnxOpenVINODeviceTypeThread1 = GPU" << endl;
  cout << endl;
  cout << "  Per-device-type EP tuning (optional):" << endl;
  cout << "    onnxOpenVINODeviceConfig_NPU_NumStreams = 4" << endl;
  cout << "    onnxOpenVINODeviceConfig_GPU_NumStreams = 2" << endl;
}

//--------------------------------------------------------------
// The layer-level test entry points are not implemented for this backend. Returning false tells
// the test harness this configuration is unsupported (not a failure). The TensorRT backend
// likewise returns false for all of these, since neither backend's ONNX emitter exposes
// single-layer test models anymore.

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
