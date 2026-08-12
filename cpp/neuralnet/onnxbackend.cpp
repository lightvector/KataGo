// ONNX Runtime backend for KataGo.
// Loads standard .bin.gz model files (builds ONNX graph from ModelDesc) or
// raw .onnx model files directly, and runs inference via ONNX Runtime with a
// configurable execution provider (CPU, CUDA, TensorRT, MIGraphX, VitisAI, CoreML)
// selected at
// runtime via the onnxProvider config key.

#include "../neuralnet/nninterface.h"
#include "../neuralnet/nneval.h"
#include "../neuralnet/nninputs.h"
#include "../neuralnet/modelversion.h"
#include "../neuralnet/onnxmodelbuilder.h"
#include "../core/makedir.h"
#include "../dataio/homedata.h"

#include <onnxruntime_cxx_api.h>
#ifdef __APPLE__
#include <coreml_provider_factory.h>
#endif

#include <onnx/onnx_pb.h>

#include <fstream>
#include <unordered_map>
#include <atomic>
#include <mutex>

#ifdef _WIN32
#include <objbase.h> // For CoInitializeEx/CoUninitialize
#endif

#ifdef ENABLE_PYTHON_ONNXRUNTIME
#include <Python.h>
#endif

using namespace std;

//--------------------------------------------------------------

// Convert a narrow (ASCII) path string to ONNX Runtime's ORTCHAR_T path type.
static std::basic_string<ORTCHAR_T> toOrtPath(const string& path) {
#ifdef _WIN32
  return std::basic_string<ORTCHAR_T>(path.begin(), path.end());
#else
  return path;
#endif
}

// Extract concrete dimensions from an ONNX TensorShapeProto. Symbolic dimensions
// (e.g. "batch") are treated as unknown and reported as -1.
static vector<int64_t> getOnnxTensorShape(const onnx::TensorShapeProto& shapeProto) {
  vector<int64_t> shape;
  shape.reserve(shapeProto.dim_size());
  for(const auto& dim : shapeProto.dim()) {
    if(dim.has_dim_value())
      shape.push_back(dim.dim_value());
    else
      shape.push_back(-1);
  }
  return shape;
}

//--------------------------------------------------------------

// Auto-detect modelVersion from introspected channel counts.
//
// Detection is based on channel-count heuristics for raw .onnx files where the
// model version is not encoded in the file.  The mapping assumes V7 inputs
// (22 spatial + 19 global channels) and distinguishes versions by the number of
// score-value and policy output channels:
//   - 4 score-value channels                    -> version 8
//   - 6 score-value channels, 1 policy channel  -> version 10
//   - 6 score-value channels, 2 policy channels -> version 15
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

  // inputsVersion 7 -> models 8-16: 22 spatial + 19 global
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
  // Older input versions -- fall back to a reasonable default
  return NNModelVersion::defaultModelVersion;
}

struct LoadedModel {
  ModelDesc modelDesc;
  bool isRawOnnx;
  string rawOnnxBytes;
  string rawOnnxFileName; // only valid when isRawOnnx is true

  // One-time scale8 transform (see maybeApplyScale8), only meaningful for the .bin.gz path. All
  // server threads share this LoadedModel, so whichever ComputeHandle is created first decides
  // for everyone.
  mutable std::atomic<bool> scale8Resolved;
  mutable std::mutex scale8Mutex;

  // Constructor for .bin.gz files
  LoadedModel(const string& fileName, const string& expectedSha256, bool rawOnnx)
    : isRawOnnx(rawOnnx)
  {
    scale8Resolved.store(false);

    if(!rawOnnx) {
      ModelDesc::loadFromFileMaybeGZipped(fileName, modelDesc, expectedSha256);
      return;
    }

    rawOnnxFileName = fileName;

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

    // Parse the ONNX ModelProto directly to introspect input/output shapes. This avoids
    // creating a temporary CPU-only ONNX Runtime session, which cannot load EP context cache
    // models that contain EPContext nodes (the CPU EP has no kernel for them).
    onnx::ModelProto modelProto;
    if(!modelProto.ParseFromArray(rawOnnxBytes.data(), (int)rawOnnxBytes.size()))
      throw StringError("ONNX backend: failed to parse raw ONNX file as ModelProto: " + fileName);

    const onnx::GraphProto& graph = modelProto.graph();

    // Introspect inputs by name first, falling back to shape-based heuristic
    int numInputChannels = 0;
    int numInputGlobalChannels = 0;
    int numInputMetaChannels = 0;
    for(const auto& input : graph.input()) {
      string name = input.name();
      if(!input.type().has_tensor_type())
        continue;
      auto shape = getOnnxTensorShape(input.type().tensor_type().shape());
      // Name-based matching must be case-insensitive: the graph node names emitted by
      // OnnxModelBuilder::build() (and expected by default elsewhere in this file) are
      // PascalCase ("InputSpatial", "InputGlobal", "InputMask", "InputMeta"), not lowercase.
      string lowerName = name;
      for(auto& c : lowerName) c = (char)tolower((unsigned char)c);
      if(lowerName.find("mask") != string::npos) {
        // The on-board mask is its own single-channel input, not part of the spatial feature
        // channel count -- explicitly ignored here so it can't clobber numInputChannels below.
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
        // Shape-based fallback: [N, C, H, W] -- spatial input
        numInputChannels = (int)shape[1];
      } else if(shape.size() == 2) {
        // Shape-based fallback: [N, C] -- first 2D is global, second is meta
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
    for(const auto& output : graph.output()) {
      string name = output.name();
      if(!output.type().has_tensor_type())
        continue;
      auto shape = getOnnxTensorShape(output.type().tensor_type().shape());

      // Case-insensitive for the same reason as the input-side matching above; also match
      // "scorevalue" (the actual "OutputScoreValue" node name), not "miscvalue".
      string lowerName = name;
      for(auto& c : lowerName) c = (char)tolower((unsigned char)c);
      if(lowerName.find("policy") != string::npos) {
        // Policy: [N, C, H, W] -> dim 1 is policy channels
        if(shape.size() >= 2)
          numPolicyChannels = (int)shape[1];
      } else if(lowerName.find("scorevalue") != string::npos) {
        // ScoreValue (aka MiscValue): [N, numScoreValueChannels]
        if(shape.size() >= 2)
          numScoreValueChannels = (int)shape[1];
      } else if(lowerName.find("value") != string::npos) {
        // Value: [N, 3]
        if(shape.size() >= 2)
          numValueChannels = (int)shape[1];
      } else if(lowerName.find("ownership") != string::npos) {
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

    scale8Resolved.store(true);  // scale8 only applies to modelDesc weights, which raw .onnx has none of.
  }

  // Apply the scale8 FP16-range workaround exactly once per model, unless skipped via
  // onnxSkipScale8. Must run before any ComputeHandle builds the ONNX graph from modelDesc.
  // No-op for the raw .onnx path (scale8Resolved is already true from the constructor above).
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

  bool transformerNHWC;  // run the trunk block stack channel-last (NHWC) for transformer models
  bool skipScale8;        // skip the scale8 FP16-range workaround (see createComputeContext)

  // VitisAI (AMD Ryzen AI NPU) EP options.
  string vitisaiConfigFile;
  string vitisaiCacheDir;
  bool vitisaiDisableCPUFallback;
  bool vitisaiUsePythonRuntime;

  // ONNX Runtime EP context cache options (VitisAI). See
  // https://onnxruntime.ai/docs/execution-providers/EP-Context-Design.html
  bool vitisaiUseEpContextCache;
  string vitisaiEpContextFilePath;
  int vitisaiEpContextEmbedMode;

  // Configurable input/output node names. Defaults match the node names emitted by the shared
  // OnnxModelBuilder::build() (see onnxmodelbuilder.cpp) used for .bin.gz -> ONNX conversion,
  // which is also what trtbackend.cpp consumes. Raw .onnx models can override these if they use
  // different names.
  string inputMaskName;
  string inputSpatialName;
  string inputGlobalName;
  string inputMetaName;
  string outputPolicyPassName;
  string outputPolicyName;
  string outputValueName;
  string outputMiscvalueName;
  string outputOwnershipName;

  // Config override for model version (-1 means auto-detect)
  int configModelVersion;

  ComputeContext(int xLen, int yLen, const string& provider)
    // TEMP DEBUG: KATAGO_ONNX_VERBOSE_EP_LOG=1 bumps the ORT Env's own logging level to VERBOSE to
    // surface ONNX Runtime's node-placement dump (VerifyEachNodeIsAssignedToAnEp).
    : env(std::getenv("KATAGO_ONNX_VERBOSE_EP_LOG") != nullptr ? ORT_LOGGING_LEVEL_VERBOSE : ORT_LOGGING_LEVEL_WARNING, "KataGoOnnx"),
      nnXLen(xLen),
      nnYLen(yLen),
      providerName(provider),
      transformerNHWC(true),
      skipScale8(false),
      vitisaiConfigFile(
#ifdef KATAGO_VITISAI_DEFAULT_CONFIG_FILE
        KATAGO_VITISAI_DEFAULT_CONFIG_FILE
#endif
      ),
      vitisaiCacheDir(""),
      vitisaiDisableCPUFallback(true),
#ifdef ENABLE_PYTHON_ONNXRUNTIME
      // When Python embedding is compiled in, default to the Python onnxruntime path for
      // VitisAI. This works around a RyzenAI SDK bug where the native C/C++ API fails to
      // create runners for EP context cache models. The user can still opt out by setting
      // onnxVitisAIUsePythonRuntime=false in the config.
      vitisaiUsePythonRuntime(true),
#else
      vitisaiUsePythonRuntime(false),
#endif
      vitisaiUseEpContextCache(false),
      vitisaiEpContextFilePath(""),
      vitisaiEpContextEmbedMode(1),
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

//--------------------------------------------------------------

#ifdef ENABLE_PYTHON_ONNXRUNTIME
namespace {

struct PythonOnnxState {
  static std::atomic<bool> initialized;
  static std::mutex initMutex;
  static PyObject* npModule;
  static PyObject* ortModule;
  static PyObject* npFloat32;
  static PyObject* npFrombuffer;
};

std::atomic<bool> PythonOnnxState::initialized{false};
std::mutex PythonOnnxState::initMutex;
PyObject* PythonOnnxState::npModule = nullptr;
PyObject* PythonOnnxState::ortModule = nullptr;
PyObject* PythonOnnxState::npFloat32 = nullptr;
PyObject* PythonOnnxState::npFrombuffer = nullptr;

static void pythonPrintError() {
  if(PyErr_Occurred())
    PyErr_Print();
}

static void ensurePythonInitialized() {
  std::lock_guard<std::mutex> lock(PythonOnnxState::initMutex);
  if(PythonOnnxState::initialized)
    return;

#ifdef KATAGO_PYTHON_EXE_PATH
  // Point the embedded interpreter at the Python installation that was detected at build time
  // (e.g. the ryzen-ai conda environment). This ensures Python finds its stdlib, site-packages,
  // and the onnxruntime package even when katago.exe is launched from a directory that does not
  // contain a full Python installation. The buffers must remain valid for the interpreter lifetime.
  {
    static std::wstring pyExeW;
    static std::wstring pyHomeW;
    if(pyExeW.empty()) {
      const char* pyExe = KATAGO_PYTHON_EXE_PATH;
      int wlen = MultiByteToWideChar(CP_UTF8, 0, pyExe, -1, nullptr, 0);
      if(wlen > 0) {
        pyExeW.resize(wlen);
        MultiByteToWideChar(CP_UTF8, 0, pyExe, -1, pyExeW.data(), wlen);
        Py_SetProgramName(pyExeW.c_str());
      }
      // Derive PYTHONHOME from the directory containing python.exe.
      const char* lastSlash = strrchr(pyExe, '/');
      if(!lastSlash) lastSlash = strrchr(pyExe, '\\');
      std::string pyHome(pyExe, lastSlash ? (size_t)(lastSlash - pyExe) : strlen(pyExe));
      wlen = MultiByteToWideChar(CP_UTF8, 0, pyHome.c_str(), -1, nullptr, 0);
      if(wlen > 0) {
        pyHomeW.resize(wlen);
        MultiByteToWideChar(CP_UTF8, 0, pyHome.c_str(), -1, pyHomeW.data(), wlen);
        Py_SetPythonHome(pyHomeW.c_str());
      }
    }
  }
#endif

  Py_Initialize();
  if(!Py_IsInitialized())
    throw StringError("ONNX backend: Py_Initialize failed");
  PythonOnnxState::npModule = PyImport_ImportModule("numpy");
  PythonOnnxState::ortModule = PyImport_ImportModule("onnxruntime");
  if(!PythonOnnxState::npModule || !PythonOnnxState::ortModule) {
    pythonPrintError();
    throw StringError("ONNX backend: failed to import numpy or onnxruntime");
  }
  PythonOnnxState::npFloat32 = PyObject_GetAttrString(PythonOnnxState::npModule, "float32");
  PythonOnnxState::npFrombuffer = PyObject_GetAttrString(PythonOnnxState::npModule, "frombuffer");
  if(!PythonOnnxState::npFloat32 || !PythonOnnxState::npFrombuffer) {
    pythonPrintError();
    throw StringError("ONNX backend: failed to get numpy helpers");
  }
  PythonOnnxState::initialized = true;
}

static PyObject* buildPythonProviderOptions(const ComputeContext* ctx) {
  PyObject* opts = PyDict_New();
  PyDict_SetItemString(opts, "config_file", PyUnicode_FromString(ctx->vitisaiConfigFile.c_str()));
  return opts;
}

static PyObject* buildPythonSessionOptions(const ComputeContext* ctx) {
  PyObject* cls = PyObject_GetAttrString(PythonOnnxState::ortModule, "SessionOptions");
  PyObject* opts = PyObject_CallObject(cls, nullptr);
  PyObject_CallMethod(opts, "__setattr__", "si", "intra_op_num_threads", 1);
  Py_DECREF(cls);
  return opts;
}

static PyObject* createPythonSession(const ComputeContext* ctx, const string& modelPath) {
  ensurePythonInitialized();
  PyGILState_STATE gil = PyGILState_Ensure();
  PyObject* sessOptions = buildPythonSessionOptions(ctx);
  PyObject* providerOpts = buildPythonProviderOptions(ctx);
  PyObject* providerOptionsList = PyList_New(1);
  PyList_SET_ITEM(providerOptionsList, 0, providerOpts);
  PyObject* providersList = PyList_New(1);
  PyList_SET_ITEM(providersList, 0, PyUnicode_FromString("VitisAIExecutionProvider"));
  PyObject* sessionCls = PyObject_GetAttrString(PythonOnnxState::ortModule, "InferenceSession");
  PyObject* modelPathObj = PyUnicode_FromString(modelPath.c_str());
  PyObject* kwargs = PyDict_New();
  PyDict_SetItemString(kwargs, "sess_options", sessOptions);
  PyDict_SetItemString(kwargs, "providers", providersList);
  PyDict_SetItemString(kwargs, "provider_options", providerOptionsList);
  PyObject* session = PyObject_Call(sessionCls, PyTuple_Pack(1, modelPathObj), kwargs);
  Py_DECREF(sessOptions); Py_DECREF(providersList); Py_DECREF(providerOptionsList);
  Py_DECREF(sessionCls); Py_DECREF(modelPathObj); Py_DECREF(kwargs);
  if(!session) {
    pythonPrintError();
    PyGILState_Release(gil);
    throw StringError("ONNX backend: failed to create Python InferenceSession");
  }
  PyGILState_Release(gil);
  return session;
}

static PyObject* createPythonSessionFromBytes(const ComputeContext* ctx, const char* data, size_t size) {
  ensurePythonInitialized();
  PyGILState_STATE gil = PyGILState_Ensure();
  PyObject* sessOptions = buildPythonSessionOptions(ctx);
  PyObject* providerOpts = buildPythonProviderOptions(ctx);
  PyObject* providerOptionsList = PyList_New(1);
  PyList_SET_ITEM(providerOptionsList, 0, providerOpts);
  PyObject* providersList = PyList_New(1);
  PyList_SET_ITEM(providersList, 0, PyUnicode_FromString("VitisAIExecutionProvider"));
  PyObject* sessionCls = PyObject_GetAttrString(PythonOnnxState::ortModule, "InferenceSession");
  PyObject* modelBytes = PyBytes_FromStringAndSize(data, size);
  PyObject* kwargs = PyDict_New();
  PyDict_SetItemString(kwargs, "sess_options", sessOptions);
  PyDict_SetItemString(kwargs, "providers", providersList);
  PyDict_SetItemString(kwargs, "provider_options", providerOptionsList);
  PyObject* session = PyObject_Call(sessionCls, PyTuple_Pack(1, modelBytes), kwargs);
  Py_DECREF(sessOptions); Py_DECREF(providersList); Py_DECREF(providerOptionsList);
  Py_DECREF(sessionCls); Py_DECREF(modelBytes); Py_DECREF(kwargs);
  if(!session) {
    pythonPrintError();
    PyGILState_Release(gil);
    throw StringError("ONNX backend: failed to create Python InferenceSession from bytes");
  }
  PyGILState_Release(gil);
  return session;
}

// Create a NumPy float32 array (reshaped) that views an existing C++ float buffer.
// The returned object is a new reference; the caller must Py_DECREF it.
static PyObject* numpyArrayFromBuffer(float* data, size_t numFloats, const vector<int64_t>& shape) {
  PyObject* memview = PyMemoryView_FromMemory(
    reinterpret_cast<char*>(data), static_cast<Py_ssize_t>(numFloats * sizeof(float)), PyBUF_WRITE);
  if(!memview)
    return nullptr;
  PyObject* arr = PyObject_CallFunctionObjArgs(
    PythonOnnxState::npFrombuffer, memview, PythonOnnxState::npFloat32, nullptr);
  Py_DECREF(memview);
  if(!arr)
    return nullptr;
  PyObject* shapeTuple = PyTuple_New(shape.size());
  for(size_t i = 0; i < shape.size(); i++)
    PyTuple_SET_ITEM(shapeTuple, i, PyLong_FromLongLong(shape[i]));
  PyObject* reshaped = PyObject_CallMethod(arr, "reshape", "O", shapeTuple);
  Py_DECREF(arr); Py_DECREF(shapeTuple);
  return reshaped;
}

} // namespace
#endif // ENABLE_PYTHON_ONNXRUNTIME

//--------------------------------------------------------------

struct ComputeHandle {
  ComputeContext* context;
  std::unique_ptr<Ort::Session> session;
  int modelVersion;

#ifdef ENABLE_PYTHON_ONNXRUNTIME
  bool usePythonRuntime;
  PyObject* pySession;
  std::vector<vector<int64_t>> outputShapes;
#endif

#ifdef _WIN32
  // COM initialization state for this thread. Some NPU runtime components (XRT/
  // VitisAI) appear to require COM on the thread that creates/runs the session.
  HRESULT comInitResult;
#endif

  ~ComputeHandle() {
#ifdef ENABLE_PYTHON_ONNXRUNTIME
    if(usePythonRuntime && pySession) {
      PyGILState_STATE gil = PyGILState_Ensure();
      Py_DECREF(pySession);
      PyGILState_Release(gil);
    }
#endif
#ifdef _WIN32
    if(comInitResult == S_OK || comInitResult == S_FALSE)
      CoUninitialize();
#endif
  }
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

  ComputeHandle(ComputeContext* ctx, const LoadedModel& loadedModel, Logger* logger, int deviceIdxForThread)
    : context(ctx),
      modelVersion(loadedModel.modelDesc.modelVersion),
#ifdef ENABLE_PYTHON_ONNXRUNTIME
      usePythonRuntime(false),
      pySession(nullptr),
#endif
      numInputChannels(loadedModel.modelDesc.numInputChannels),
      numInputGlobalChannels(loadedModel.modelDesc.numInputGlobalChannels),
      numPolicyChannels(loadedModel.modelDesc.numPolicyChannels),
      numValueChannels(loadedModel.modelDesc.numValueChannels),
      numScoreValueChannels(loadedModel.modelDesc.numScoreValueChannels),
      numOwnershipChannels(loadedModel.modelDesc.numOwnershipChannels),
      numInputMetaChannels(loadedModel.modelDesc.numInputMetaChannels),
      policyResultLen(ctx->nnXLen * ctx->nnYLen + 1)
#ifdef _WIN32
      , comInitResult(S_OK)
#endif
  {
#ifdef _WIN32
    comInitResult = CoInitializeEx(nullptr, COINIT_MULTITHREADED);
#endif

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
      OnnxModelBuilder::Result onnxResult = OnnxModelBuilder::build(
        loadedModel.modelDesc, ctx->nnXLen, ctx->nnYLen,
        /*requireExactNNLen=*/false, ctx->transformerNHWC, logger, /*emitFusedMishOp=*/false
      );
      builtOnnxBytes = std::move(onnxResult.serializedModel);
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
    sessionOpts.SetIntraOpNumThreads(1);

    // Select execution provider based on providerName
    const string& provider = ctx->providerName;

#ifdef ENABLE_PYTHON_ONNXRUNTIME
    // When Python embedding is compiled in, VitisAI defaults to the Python onnxruntime path
    // to work around the RyzenAI 1.7.1 SDK C/C++ API bug. The user can opt out via
    // onnxVitisAIUsePythonRuntime=false in the config.
    usePythonRuntime = (provider == "vitisai" && ctx->vitisaiUsePythonRuntime);
#endif

    if(provider == "coreml") {
#ifdef __APPLE__
      uint32_t coremlFlags = COREML_FLAG_CREATE_MLPROGRAM;
      Ort::ThrowOnError(OrtSessionOptionsAppendExecutionProvider_CoreML(sessionOpts, coremlFlags));
      if(logger != NULL)
        logger->write("ONNX backend: CoreML execution provider enabled (MLProgram mode)");
#else
      throw StringError("ONNX backend: CoreML is only available on Apple platforms");
#endif
    } else if(provider == "cuda") {
      OrtCUDAProviderOptions cudaOpts{};
      cudaOpts.device_id = deviceIdxForThread >= 0 ? deviceIdxForThread : 0;
      sessionOpts.AppendExecutionProvider_CUDA(cudaOpts);
      if(logger != NULL)
        logger->write("ONNX backend: CUDA execution provider enabled, device_id=" + Global::intToString(cudaOpts.device_id));
    } else if(provider == "tensorrt") {
      OrtTensorRTProviderOptions trtOpts{};
      trtOpts.device_id = deviceIdxForThread >= 0 ? deviceIdxForThread : 0;
      sessionOpts.AppendExecutionProvider_TensorRT(trtOpts);
      if(logger != NULL)
        logger->write("ONNX backend: TensorRT execution provider enabled, device_id=" + Global::intToString(trtOpts.device_id));
    } else if(provider == "migraphx") {
      OrtMIGraphXProviderOptions migraphxOpts{};
      migraphxOpts.device_id = deviceIdxForThread >= 0 ? deviceIdxForThread : 0;
      sessionOpts.AppendExecutionProvider_MIGraphX(migraphxOpts);
      if(logger != NULL)
        logger->write("ONNX backend: MIGraphX execution provider enabled, device_id=" + Global::intToString(migraphxOpts.device_id));
    } else if(provider == "vitisai") {
#ifdef ENABLE_PYTHON_ONNXRUNTIME
      if(usePythonRuntime) {
        if(logger != NULL)
          logger->write("ONNX backend: using Python-embedded onnxruntime path for VitisAI");
      } else
#endif
      {
        std::unordered_map<std::string, std::string> vitisOpts;
        if(!ctx->vitisaiConfigFile.empty())
          vitisOpts["config_file"] = ctx->vitisaiConfigFile;
        else
          throw StringError(
            "ONNX backend: onnxProvider=vitisai requires a VitisAI EP config file (vaip_config.json, "
            "shipped with the AMD Ryzen AI / VitisAI SDK). Set 'onnxVitisAIConfigFile' in your config, "
            "or rebuild with a locatable Ryzen AI installation so a default gets baked in."
          );
        // The VitisAI EP's enable_cache_file_io_in_mem provider option defaults to 1 (in-memory
        // only -- nothing is ever written to cache_dir), so it must be explicitly set to 0 to make
        // the compiled model actually persist to disk across process launches. Without this, every
        // katago.exe launch pays the full NPU compile cost again (can take on the order of minutes).
        // For pre-built EP context cache models, leave the cache options at their defaults; the
        // compiled context is already embedded in the ONNX model and does not need to be persisted.
        bool isContextModel = loadedModel.isRawOnnx &&
                              loadedModel.rawOnnxFileName.find("_ctx.onnx") != string::npos;
        if(!isContextModel) {
          vitisOpts["cache_dir"] = ctx->vitisaiCacheDir;
          vitisOpts["enable_cache_file_io_in_mem"] = "0";
        }
        // Use the generic provider-registration path rather than the VitisAI-specific C++ helper.
        // The generic path loads the provider by name and also registers any associated custom-op
        // libraries (e.g. onnxruntime_vitis_ai_custom_ops.dll), which is required for EP context
        // cache models containing EPContext nodes.
        sessionOpts.AppendExecutionProvider("VitisAIExecutionProvider", vitisOpts);
        if(ctx->vitisaiDisableCPUFallback) {
          // Force a hard failure at session-creation time if any node can't be claimed by VitisAI,
          // instead of ONNX Runtime silently assigning it to the always-registered CPU EP. Without
          // this, "session created successfully" is not evidence that anything actually runs on the
          // NPU. Set onnxVitisAIDisableCPUFallback=false to relax this if partial CPU fallback is
          // acceptable for your use case.
          sessionOpts.AddConfigEntry("session.disable_cpu_ep_fallback", "1");
        }
        if(ctx->vitisaiUseEpContextCache) {
          // ONNX Runtime EP context cache: ask the VitisAI EP to dump a self-contained context model
          // (compiled subgraphs embedded as EPContext nodes). This may avoid per-subgraph repeated
          // initialization of the NPU runtime. See https://onnxruntime.ai/docs/execution-providers/EP-Context-Design.html
          string ctxFilePath = ctx->vitisaiEpContextFilePath;
          if(ctxFilePath.empty())
            ctxFilePath = ctx->vitisaiCacheDir + "/katago_vitisai_ctx.onnx";
          sessionOpts.AddConfigEntry("ep.context_enable", "1");
          sessionOpts.AddConfigEntry("ep.context_file_path", ctxFilePath.c_str());
          sessionOpts.AddConfigEntry("ep.context_embed_mode", Global::intToString(ctx->vitisaiEpContextEmbedMode).c_str());
          if(logger != NULL) {
            logger->write(
              "ONNX backend: VitisAI EP context cache enabled, context_file_path=" + ctxFilePath +
              ", embed_mode=" + Global::intToString(ctx->vitisaiEpContextEmbedMode)
            );
          }
        }
        if(logger != NULL) {
          logger->write(
            "ONNX backend: VitisAI execution provider enabled, config_file=" + ctx->vitisaiConfigFile +
            ", cache_dir=" + ctx->vitisaiCacheDir +
            ", disable_cpu_ep_fallback=" + Global::boolToString(ctx->vitisaiDisableCPUFallback)
          );
        }
      }
    } else if(provider == "cpu" || provider.empty()) {
      if(logger != NULL)
        logger->write("ONNX backend: using CPU execution provider");
    } else {
      throw StringError("ONNX backend: unknown onnxProvider '" + provider + "', expected 'cpu', 'coreml', 'cuda', 'tensorrt', 'migraphx', or 'vitisai'");
    }

    // Create session. For raw .onnx files we have the original file path, so prefer loading
    // from disk: some EPs (notably VitisAI when consuming an EP context cache model) need the
    // model file path to resolve embedded context binaries or to correctly claim EPContext nodes.
    // For .bin.gz-derived models we only have in-memory bytes.
#ifdef ENABLE_PYTHON_ONNXRUNTIME
    if(usePythonRuntime) {
      if(loadedModel.isRawOnnx && !loadedModel.rawOnnxFileName.empty()) {
        if(logger != NULL)
          logger->write("ONNX backend: creating Python session from file path: " + loadedModel.rawOnnxFileName);
        pySession = createPythonSession(ctx, loadedModel.rawOnnxFileName);
      } else {
        if(logger != NULL)
          logger->write("ONNX backend: creating Python session from in-memory bytes");
        pySession = createPythonSessionFromBytes(ctx, onnxData, onnxSize);
      }

      // Query input/output names from the Python session and store output shapes.
      PyGILState_STATE gil = PyGILState_Ensure();
      PyObject* pyInputs = PyObject_CallMethod(pySession, "get_inputs", nullptr);
      if(!pyInputs) { pythonPrintError(); PyGILState_Release(gil); throw StringError("ONNX backend: Python get_inputs failed"); }
      Py_ssize_t numInputs = PySequence_Size(pyInputs);
      for(Py_ssize_t i = 0; i < numInputs; i++) {
        PyObject* inp = PySequence_GetItem(pyInputs, i);
        PyObject* nameObj = PyObject_GetAttrString(inp, "name");
        inputNames.push_back(PyUnicode_AsUTF8(nameObj));
        Py_DECREF(nameObj); Py_DECREF(inp);
      }
      Py_DECREF(pyInputs);

      PyObject* pyOutputs = PyObject_CallMethod(pySession, "get_outputs", nullptr);
      if(!pyOutputs) { pythonPrintError(); PyGILState_Release(gil); throw StringError("ONNX backend: Python get_outputs failed"); }
      Py_ssize_t numOutputs = PySequence_Size(pyOutputs);
      for(Py_ssize_t i = 0; i < numOutputs; i++) {
        PyObject* out = PySequence_GetItem(pyOutputs, i);
        PyObject* nameObj = PyObject_GetAttrString(out, "name");
        outputNames.push_back(PyUnicode_AsUTF8(nameObj));
        PyObject* shapeObj = PyObject_GetAttrString(out, "shape");
        Py_ssize_t shapeLen = PySequence_Size(shapeObj);
        vector<int64_t> shape;
        for(Py_ssize_t s = 0; s < shapeLen; s++) {
          PyObject* dim = PySequence_GetItem(shapeObj, s);
          if(PyLong_Check(dim)) shape.push_back(PyLong_AsLongLong(dim));
          else shape.push_back(-1); // symbolic dimension (batch)
          Py_DECREF(dim);
        }
        outputShapes.push_back(shape);
        Py_DECREF(nameObj); Py_DECREF(shapeObj); Py_DECREF(out);
      }
      Py_DECREF(pyOutputs);
      PyGILState_Release(gil);

      for(auto& n : inputNames) inputNamePtrs.push_back(n.c_str());
      for(auto& n : outputNames) outputNamePtrs.push_back(n.c_str());

      if(logger != NULL) {
        logger->write("ONNX backend: Python session created, inputs=" + Global::int64ToString(numInputs) +
                       " outputs=" + Global::int64ToString(numOutputs));
        // Log which onnxruntime Python package is being used. This is useful for diagnosing
        // DLL shadowing: if the directory containing katago.exe also contains an onnxruntime.dll
        // (or XRT/VitisAI DLLs) from a different build, the embedded Python path may load the
        // wrong one and fail with xrt_core.dll / runner-creation errors.
        PyGILState_STATE gil = PyGILState_Ensure();
        PyObject* ortModule = PyImport_ImportModule("onnxruntime");
        if(ortModule) {
          PyObject* fileObj = PyObject_GetAttrString(ortModule, "__file__");
          if(fileObj) {
            logger->write(string("ONNX backend: Python onnxruntime loaded from: ") + PyUnicode_AsUTF8(fileObj));
            Py_DECREF(fileObj);
          }
          Py_DECREF(ortModule);
        }
        PyGILState_Release(gil);
      }
    } else
#endif
    {
      if(loadedModel.isRawOnnx && !loadedModel.rawOnnxFileName.empty()) {
        if(logger != NULL)
          logger->write("ONNX backend: creating session from file path: " + loadedModel.rawOnnxFileName);
        session = std::make_unique<Ort::Session>(ctx->env, toOrtPath(loadedModel.rawOnnxFileName).c_str(), sessionOpts);
      } else {
        if(logger != NULL)
          logger->write("ONNX backend: creating session from in-memory bytes");
        session = std::make_unique<Ort::Session>(ctx->env, onnxData, onnxSize, sessionOpts);
      }

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
  const string& homeDataDirOverride,
  enabled_t useFP16Mode,
  const LoadedModel* loadedModel,
  ConfigParser& cfg
) {
  (void)gpuIdxs;
  (void)loadedModel;
  // The emitted ONNX graph is fp32; inference precision is chosen internally by the execution
  // provider. KataGo's global useFP16 flag therefore cannot be honored here - fail loudly instead
  // of silently ignoring a request.
  if(useFP16Mode == enabled_t::True)
    throw StringError(
      "ONNX backend: the global useFP16 flag is not supported and cannot be honored. "
      "Precision is controlled by the execution provider. Leave useFP16 unset or set it to false/auto.");

  string providerName = cfg.contains("onnxProvider") ? Global::toLower(cfg.getString("onnxProvider")) : "cpu";

  if(logger != NULL)
    logger->write("ONNX backend: creating compute context for " +
                   Global::intToString(nnXLen) + "x" + Global::intToString(nnYLen) +
                   " with provider '" + providerName + "'");

  ComputeContext* ctx = new ComputeContext(nnXLen, nnYLen, providerName);

  // Apply configured node names / options, read directly off cfg.
  if(cfg.contains("onnxInputMask")) ctx->inputMaskName = cfg.getString("onnxInputMask");
  if(cfg.contains("onnxInputSpatial")) ctx->inputSpatialName = cfg.getString("onnxInputSpatial");
  if(cfg.contains("onnxInputGlobal")) ctx->inputGlobalName = cfg.getString("onnxInputGlobal");
  if(cfg.contains("onnxInputMeta")) ctx->inputMetaName = cfg.getString("onnxInputMeta");
  if(cfg.contains("onnxOutputPolicyPass")) ctx->outputPolicyPassName = cfg.getString("onnxOutputPolicyPass");
  if(cfg.contains("onnxOutputPolicy")) ctx->outputPolicyName = cfg.getString("onnxOutputPolicy");
  if(cfg.contains("onnxOutputValue")) ctx->outputValueName = cfg.getString("onnxOutputValue");
  if(cfg.contains("onnxOutputMiscvalue")) ctx->outputMiscvalueName = cfg.getString("onnxOutputMiscvalue");
  if(cfg.contains("onnxOutputOwnership")) ctx->outputOwnershipName = cfg.getString("onnxOutputOwnership");

  // Trunk layout for transformer models. Default NHWC (channel-last), matching the TensorRT
  // backend's trtTransformerNHWC default; ignored entirely for models without transformer blocks.
  ctx->transformerNHWC = cfg.contains("onnxTransformerNHWC") ? cfg.getBool("onnxTransformerNHWC") : true;

  // Skip the scale8 FP16-range workaround (default false = apply it). scale8 keeps convnet
  // activations 8x smaller so they stay inside typical FP16 execution-provider ranges; the cost is
  // MISH_SCALE8 subgraphs that can block fused-Mish optimizations on some providers. Keep on
  // (default); set true only for FP32 precision or workloads where FP16 overflow is not a
  // practical risk.
  ctx->skipScale8 = cfg.contains("onnxSkipScale8") ? cfg.getBool("onnxSkipScale8") : false;

  if(cfg.contains("onnxVitisAIConfigFile")) ctx->vitisaiConfigFile = cfg.getString("onnxVitisAIConfigFile");
  if(cfg.contains("onnxVitisAICacheDir")) ctx->vitisaiCacheDir = cfg.getString("onnxVitisAICacheDir");
  if(cfg.contains("onnxVitisAIDisableCPUFallback"))
    ctx->vitisaiDisableCPUFallback = cfg.getBool("onnxVitisAIDisableCPUFallback");
  if(cfg.contains("onnxVitisAIUsePythonRuntime"))
    ctx->vitisaiUsePythonRuntime = cfg.getBool("onnxVitisAIUsePythonRuntime");
  if(cfg.contains("onnxVitisAIUseEpContextCache"))
    ctx->vitisaiUseEpContextCache = cfg.getBool("onnxVitisAIUseEpContextCache");
  if(cfg.contains("onnxVitisAIEpContextFilePath"))
    ctx->vitisaiEpContextFilePath = cfg.getString("onnxVitisAIEpContextFilePath");
  if(cfg.contains("onnxVitisAIEpContextEmbedMode"))
    ctx->vitisaiEpContextEmbedMode = Global::stringToInt(cfg.getString("onnxVitisAIEpContextEmbedMode"));
  if(providerName == "vitisai" && ctx->vitisaiCacheDir.empty()) {
    // No explicit override: default to a fixed subdir of katagodata, same convention as
    // trtbackend.cpp's "trtcache" (see HomeData::getHomeDataDir), so the (slow, ~minutes-long)
    // NPU compile only has to happen once per model rather than on every process launch.
    string homeDataDir = HomeData::getHomeDataDir(true, homeDataDirOverride);
    string cacheDir = homeDataDir + "/vitisaicache";
    MakeDir::make(cacheDir);
    ctx->vitisaiCacheDir = cacheDir;
  }
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
  if(inputsUseNHWC)
    throw StringError("ONNX backend: inputsUseNHWC = true not supported, must use NCHW");

  // Apply the scale8 FP16-range workaround exactly once per model (unless onnxSkipScale8),
  // before this handle builds the ONNX graph from modelDesc.
  loadedModel->maybeApplyScale8(context->skipScale8);

  if(logger != NULL) {
    logger->write("ONNX backend thread " + Global::intToString(serverThreadIdx) +
                  ": Model version " + Global::intToString(loadedModel->modelDesc.modelVersion));
    logger->write("ONNX backend thread " + Global::intToString(serverThreadIdx) +
                  ": Model name: " + loadedModel->modelDesc.name);
    string deviceInfo =
      context->providerName == "vitisai"
      ? "n/a (single NPU device)"
      : Global::intToString(gpuIdxForThisThread);
    logger->write("ONNX backend thread " + Global::intToString(serverThreadIdx) +
                  ": provider=" + context->providerName +
                  " deviceIdx=" + deviceInfo);
  }

  return new ComputeHandle(context, *loadedModel, logger, gpuIdxForThisThread);
}

void NeuralNet::freeComputeHandle(ComputeHandle* computeHandle) {
  delete computeHandle;
}

bool NeuralNet::isUsingFP16(const ComputeHandle* handle) {
  (void)handle;
  return false;
}

bool NeuralNet::setIsWarmup(const ComputeHandle* handle, bool isWarmup) {
  (void)handle;
  (void)isWarmup;
  return false;
}

//--------------------------------------------------------------

static int findNameIndex(const vector<string>& names, const vector<string>& targets);

#ifdef ENABLE_PYTHON_ONNXRUNTIME
// Run inference through the embedded Python onnxruntime.InferenceSession and copy the
// resulting NumPy arrays back into KataGo's output buffers. This path is used for the
// VitisAI provider when the public C/C++ API path fails (RyzenAI 1.7.1 SDK bug).
static void getOutputPython(
  ComputeHandle* computeHandle,
  InputBuffers* inputBuffers,
  int numBatchEltsFilled,
  NNResultBuf** inputBufs,
  vector<NNOutput*>& outputs
) {
  const int batchSize = numBatchEltsFilled;
  const int nnXLen = computeHandle->context->nnXLen;
  const int nnYLen = computeHandle->context->nnYLen;
  const int numSpatialFeatures = computeHandle->numInputChannels;
  const int numGlobalFeatures = computeHandle->numInputGlobalChannels;
  const int numInputMetaChannels = computeHandle->numInputMetaChannels;
  const ComputeContext* ctx = computeHandle->context;

  // Fill input buffers (same logic as the Ort::Session path).
  for(int nIdx = 0; nIdx < batchSize; nIdx++) {
    float* rowSpatialInput = inputBuffers->spatialInput.data() + (inputBuffers->singleInputElts * nIdx);
    float* rowGlobalInput = inputBuffers->globalInput.data() + (inputBuffers->singleInputGlobalElts * nIdx);
    const float* rowGlobal = inputBufs[nIdx]->rowGlobalBuf.data();
    const float* rowSpatial = inputBufs[nIdx]->rowSpatialBuf.data();
    std::copy(rowGlobal, rowGlobal + numGlobalFeatures, rowGlobalInput);
    SymmetryHelpers::copyInputsWithSymmetry(rowSpatial, rowSpatialInput, 1, nnYLen, nnXLen, numSpatialFeatures, false, inputBufs[nIdx]->symmetry);
    if(numInputMetaChannels > 0) {
      float* rowMetaInput = inputBuffers->metaInput.data() + (inputBuffers->singleInputMetaElts * nIdx);
      const float* rowMeta = inputBufs[nIdx]->rowMetaBuf.data();
      std::copy(rowMeta, rowMeta + numInputMetaChannels, rowMetaInput);
    }
  }

  int spatialIdx = findNameIndex(computeHandle->inputNames, {ctx->inputSpatialName});
  int globalIdx = findNameIndex(computeHandle->inputNames, {ctx->inputGlobalName});
  if(spatialIdx < 0 || globalIdx < 0)
    throw StringError("ONNX backend (Python path): could not find expected input names");

  const int spatialPolicyLen = nnXLen * nnYLen;
  int maskIdx = findNameIndex(computeHandle->inputNames, {ctx->inputMaskName});
  vector<float> maskBuf;
  if(maskIdx >= 0) {
    maskBuf.resize((size_t)batchSize * spatialPolicyLen);
    for(int r = 0; r < batchSize; r++) {
      const float* rowSpatial = inputBuffers->spatialInput.data() + inputBuffers->singleInputElts * r;
      std::copy(rowSpatial, rowSpatial + spatialPolicyLen, maskBuf.data() + (size_t)r * spatialPolicyLen);
    }
  }

  int metaIdx = -1;
  if(numInputMetaChannels > 0) {
    metaIdx = findNameIndex(computeHandle->inputNames, {ctx->inputMetaName});
    if(metaIdx < 0)
      throw StringError("ONNX backend (Python path): model has metadata channels but could not find " + ctx->inputMetaName);
  }

  PyGILState_STATE gil = PyGILState_Ensure();

  // Build input_feed dict.
  PyObject* inputFeed = PyDict_New();
  for(size_t i = 0; i < computeHandle->inputNames.size(); i++) {
    float* data = nullptr;
    size_t numFloats = 0;
    vector<int64_t> shape;
    if((int)i == spatialIdx) {
      data = inputBuffers->spatialInput.data();
      numFloats = inputBuffers->singleInputElts * batchSize;
      shape = {batchSize, numSpatialFeatures, nnYLen, nnXLen};
    } else if((int)i == globalIdx) {
      data = inputBuffers->globalInput.data();
      numFloats = inputBuffers->singleInputGlobalElts * batchSize;
      shape = {batchSize, numGlobalFeatures, 1, 1};
    } else if((int)i == metaIdx) {
      data = inputBuffers->metaInput.data();
      numFloats = inputBuffers->singleInputMetaElts * batchSize;
      shape = {batchSize, numInputMetaChannels, 1, 1};
    } else if((int)i == maskIdx) {
      data = maskBuf.data();
      numFloats = maskBuf.size();
      shape = {batchSize, 1, nnYLen, nnXLen};
    } else {
      PyGILState_Release(gil);
      throw StringError("ONNX backend (Python path): unexpected input node '" + computeHandle->inputNames[i] +
                         "' -- only mask, spatial, global, and meta inputs are supported");
    }
    PyObject* arr = numpyArrayFromBuffer(data, numFloats, shape);
    if(!arr) {
      pythonPrintError();
      PyGILState_Release(gil);
      throw StringError("ONNX backend (Python path): failed to create numpy array for input '" + computeHandle->inputNames[i] + "'");
    }
    PyDict_SetItemString(inputFeed, computeHandle->inputNames[i].c_str(), arr);
    Py_DECREF(arr);
  }

  // Run inference.
  PyObject* runResult = PyObject_CallMethod(computeHandle->pySession, "run", "OO", Py_None, inputFeed);
  Py_DECREF(inputFeed);
  if(!runResult) {
    pythonPrintError();
    PyGILState_Release(gil);
    throw StringError("ONNX backend (Python path): session.run failed");
  }

  // Copy outputs and wrap them in Ort::Value tensors so the rest of getOutput can stay unchanged.
  Py_ssize_t numOutputs = PyList_Size(runResult);
  vector<Ort::Value> outputTensors;
  outputTensors.reserve(numOutputs);
  vector<vector<float>> outputData(numOutputs);
  Ort::MemoryInfo memInfo = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

  for(Py_ssize_t i = 0; i < numOutputs; i++) {
    PyObject* outArr = PyList_GET_ITEM(runResult, i);
    PyObject* bytes = PyObject_CallMethod(outArr, "tobytes", nullptr);
    if(!bytes) {
      pythonPrintError();
      Py_DECREF(runResult);
      PyGILState_Release(gil);
      throw StringError("ONNX backend (Python path): failed to get output bytes");
    }
    char* buf = nullptr;
    Py_ssize_t len = 0;
    PyBytes_AsStringAndSize(bytes, &buf, &len);
    size_t numFloats = len / sizeof(float);
    outputData[i].resize(numFloats);
    std::memcpy(outputData[i].data(), buf, len);
    Py_DECREF(bytes);

    // Use the stored output shape, replacing any symbolic dimension with the current batch size.
    vector<int64_t> shape = computeHandle->outputShapes[i];
    int64_t product = 1;
    for(size_t s = 0; s < shape.size(); s++) {
      if(shape[s] < 0) shape[s] = batchSize;
      product *= shape[s];
    }
    // Fallback: if stored shape is empty or inconsistent, flatten to [batchSize, numFloats/batchSize].
    if(shape.empty() || product != (int64_t)numFloats) {
      shape = {(int64_t)batchSize, (int64_t)(numFloats / batchSize)};
    }
    outputTensors.push_back(
      Ort::Value::CreateTensor<float>(memInfo, outputData[i].data(), outputData[i].size(), shape.data(), shape.size())
    );
  }
  Py_DECREF(runResult);
  PyGILState_Release(gil);

  // The remainder of output extraction is identical to the C++ API path.
  int policyPassOutputIdx = findNameIndex(computeHandle->outputNames, {ctx->outputPolicyPassName});
  int policyOutputIdx = findNameIndex(computeHandle->outputNames, {ctx->outputPolicyName});
  int valueOutputIdx = findNameIndex(computeHandle->outputNames, {ctx->outputValueName});
  int miscvalueOutputIdx = findNameIndex(computeHandle->outputNames, {ctx->outputMiscvalueName});
  int ownershipOutputIdx = findNameIndex(computeHandle->outputNames, {ctx->outputOwnershipName});

  if(policyPassOutputIdx < 0)
    throw StringError("ONNX backend (Python path): could not find policy-pass output node '" + ctx->outputPolicyPassName + "'");
  if(policyOutputIdx < 0)
    throw StringError("ONNX backend (Python path): could not find policy output node '" + ctx->outputPolicyName + "'");
  if(valueOutputIdx < 0)
    throw StringError("ONNX backend (Python path): could not find value output node '" + ctx->outputValueName + "'");
  if(miscvalueOutputIdx < 0)
    throw StringError("ONNX backend (Python path): could not find miscvalue output node '" + ctx->outputMiscvalueName + "'");
  if(ownershipOutputIdx < 0)
    throw StringError("ONNX backend (Python path): could not find ownership output node '" + ctx->outputOwnershipName + "'");

  const float* policyPassData = outputTensors[policyPassOutputIdx].GetTensorData<float>();
  const float* policyData = outputTensors[policyOutputIdx].GetTensorData<float>();
  const float* valueData = outputTensors[valueOutputIdx].GetTensorData<float>();
  const float* miscvalueData = outputTensors[miscvalueOutputIdx].GetTensorData<float>();
  const float* ownershipData = outputTensors[ownershipOutputIdx].GetTensorData<float>();

  assert(policyPassData != nullptr);
  assert(policyData != nullptr);
  assert(valueData != nullptr);
  assert(miscvalueData != nullptr);
  assert(ownershipData != nullptr);
  assert((int)outputs.size() == batchSize);

  float policyProbsTmp[NNPos::MAX_NN_POLICY_SIZE];

  for(int row = 0; row < batchSize; row++) {
    NNOutput* output = outputs[row];
    assert(output->nnXLen == nnXLen);
    assert(output->nnYLen == nnYLen);
    float policyOptimism = (float)inputBufs[row]->policyOptimism;

    {
      const float* policyRowBase = policyData + (size_t)row * computeHandle->numPolicyChannels * spatialPolicyLen;
      const float* policyPassRowBase = policyPassData + (size_t)row * computeHandle->numPolicyChannels;
      float* policyProbs = output->policyProbs;

      if(computeHandle->numPolicyChannels >= 2) {
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
        assert(computeHandle->numPolicyChannels == 1);
        const float* ch0 = policyRowBase;
        SymmetryHelpers::copyOutputsWithSymmetry(ch0, policyProbs, 1, nnYLen, nnXLen, inputBufs[row]->symmetry);
        policyProbs[spatialPolicyLen] = policyPassRowBase[0];
      }
    }

    {
      int numVC = computeHandle->numValueChannels;
      assert(numVC == 3);
      output->whiteWinProb = valueData[row * numVC];
      output->whiteLossProb = valueData[row * numVC + 1];
      output->whiteNoResultProb = valueData[row * numVC + 2];
    }

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

    if(output->whiteOwnerMap != NULL) {
      assert(computeHandle->numOwnershipChannels == 1);
      const float* ownershipRowBuf = ownershipData + row * nnXLen * nnYLen;
      SymmetryHelpers::copyOutputsWithSymmetry(ownershipRowBuf, output->whiteOwnerMap, 1, nnYLen, nnXLen, inputBufs[row]->symmetry);
    }
  }
}
#endif // ENABLE_PYTHON_ONNXRUNTIME

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

#ifdef ENABLE_PYTHON_ONNXRUNTIME
  if(computeHandle->usePythonRuntime) {
    getOutputPython(computeHandle, inputBuffers, numBatchEltsFilled, inputBufs, outputs);
    return;
  }
#endif

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

  // NC11 (rank 4), matching OnnxModelBuilder::build()'s addInputNC11("InputGlobal", ...).
  std::array<int64_t, 4> globalShape = {batchSize, numGlobalFeatures, 1, 1};
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

  const int spatialPolicyLen = nnXLen * nnYLen;

  // InputMask (the on-board mask, [N,1,H,W]) is required by graphs built by OnnxModelBuilder::build()
  // (used for .bin.gz models), but may be absent from hand-exported raw .onnx models - only require
  // it if the session actually declares it. It's channel 0 of the spatial input (KataGo convention),
  // but not contiguous across rows within the spatial buffer, so gather it into its own buffer.
  int maskIdx = findNameIndex(computeHandle->inputNames, {ctx->inputMaskName});
  vector<float> maskBuf;
  Ort::Value maskTensor(nullptr);
  if(maskIdx >= 0) {
    maskBuf.resize((size_t)batchSize * spatialPolicyLen);
    for(int r = 0; r < batchSize; r++) {
      const float* rowSpatial = inputBuffers->spatialInput.data() + inputBuffers->singleInputElts * r;
      std::copy(rowSpatial, rowSpatial + spatialPolicyLen, maskBuf.data() + (size_t)r * spatialPolicyLen);
    }
    std::array<int64_t, 4> maskShape = {batchSize, 1, nnYLen, nnXLen};
    maskTensor = Ort::Value::CreateTensor<float>(
      memInfo, maskBuf.data(), maskBuf.size(), maskShape.data(), maskShape.size()
    );
  }

  int metaIdx = -1;
  Ort::Value metaTensor(nullptr);
  if(computeHandle->numInputMetaChannels > 0) {
    metaIdx = findNameIndex(computeHandle->inputNames, {ctx->inputMetaName});
    if(metaIdx < 0)
      throw StringError("ONNX backend: model has metadata channels but could not find " + ctx->inputMetaName);
    // NC11 (rank 4), matching trtbackend.cpp's InputMeta declaration.
    std::array<int64_t, 4> metaShape = {batchSize, computeHandle->numInputMetaChannels, 1, 1};
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
    else if((int)i == maskIdx)
      inputTensors.push_back(std::move(maskTensor));
    else {
      throw StringError("ONNX backend: unexpected input node '" + computeHandle->inputNames[i] +
                         "' -- only mask, spatial, global, and meta inputs are supported");
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

  // Find output indices using configured node names. OutputPolicyPass ([N,C]) and OutputPolicy
  // ([N,C,H,W]) are separate tensors in graphs built by OnnxModelBuilder::build() - the pass logit
  // isn't appended to the spatial policy tensor.
  int policyPassOutputIdx = findNameIndex(computeHandle->outputNames, {ctx->outputPolicyPassName});
  int policyOutputIdx = findNameIndex(computeHandle->outputNames, {ctx->outputPolicyName});
  int valueOutputIdx = findNameIndex(computeHandle->outputNames, {ctx->outputValueName});
  int miscvalueOutputIdx = findNameIndex(computeHandle->outputNames, {ctx->outputMiscvalueName});
  int ownershipOutputIdx = findNameIndex(computeHandle->outputNames, {ctx->outputOwnershipName});

  if(policyPassOutputIdx < 0)
    throw StringError("ONNX backend: could not find policy-pass output node '" + ctx->outputPolicyPassName + "'");
  if(policyOutputIdx < 0)
    throw StringError("ONNX backend: could not find policy output node '" + ctx->outputPolicyName + "'");
  if(valueOutputIdx < 0)
    throw StringError("ONNX backend: could not find value output node '" + ctx->outputValueName + "'");
  if(miscvalueOutputIdx < 0)
    throw StringError("ONNX backend: could not find miscvalue output node '" + ctx->outputMiscvalueName + "'");
  if(ownershipOutputIdx < 0)
    throw StringError("ONNX backend: could not find ownership output node '" + ctx->outputOwnershipName + "'");

  const float* policyPassData = outputTensors[policyPassOutputIdx].GetTensorData<float>();
  const float* policyData = outputTensors[policyOutputIdx].GetTensorData<float>();
  const float* valueData = outputTensors[valueOutputIdx].GetTensorData<float>();
  const float* miscvalueData = outputTensors[miscvalueOutputIdx].GetTensorData<float>();
  const float* ownershipData = outputTensors[ownershipOutputIdx].GetTensorData<float>();

  assert(policyPassData != nullptr);
  assert(policyData != nullptr);
  assert(valueData != nullptr);
  assert(miscvalueData != nullptr);
  assert(ownershipData != nullptr);
  assert((int)outputs.size() == batchSize);

  float policyProbsTmp[NNPos::MAX_NN_POLICY_SIZE];

  for(int row = 0; row < batchSize; row++) {
    NNOutput* output = outputs[row];
    assert(output->nnXLen == nnXLen);
    assert(output->nnYLen == nnYLen);
    float policyOptimism = (float)inputBufs[row]->policyOptimism;

    // Policy: OutputPolicy is [N, C, H*W] (channel-major, NCHW), OutputPolicyPass is [N, C]
    // (one pass logit per channel). These are two separate tensors, not a single [N,C,H*W+1].
    {
      const float* policyRowBase = policyData + (size_t)row * numPolicyChannels * spatialPolicyLen;
      const float* policyPassRowBase = policyPassData + (size_t)row * numPolicyChannels;
      float* policyProbs = output->policyProbs;

      if(numPolicyChannels >= 2) {
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
        const float* ch0 = policyRowBase;
        SymmetryHelpers::copyOutputsWithSymmetry(ch0, policyProbs, 1, nnYLen, nnXLen, inputBufs[row]->symmetry);
        policyProbs[spatialPolicyLen] = policyPassRowBase[0];
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

    // MiscValue: [N, numScoreValueChannels] -- version-dependent interpretation
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
  cout << "ONNX backend: device enumeration is provider-specific." << endl;
  cout << "Use onnxProvider plus provider-specific settings in config." << endl;
}

//--------------------------------------------------------------
// FOR TESTING -- all return false (not implemented for this backend)

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
