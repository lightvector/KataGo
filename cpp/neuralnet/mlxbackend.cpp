#ifdef USE_MLX_BACKEND

/**
 * MLX backend for KataGo.
 * Uses Apple's MLX framework for neural network inference on Apple Silicon.
 * Supports FP16 (half precision) and FP32 computation with NHWC memory layout.
 * FP16 Winograd uses selective fp32 accumulation at the matmul reduction and
 * BatchNorm intermediate for numerical stability.
 * `mlxUseFP16 = auto` resolves to fp16.
 */

#include "../neuralnet/nninterface.h"
#include "../neuralnet/desc.h"
#include "../neuralnet/modelversion.h"
#include "../neuralnet/nninputs.h"
#include "../neuralnet/nneval.h"
#include "../neuralnet/activations.h"
#include "../neuralnet/mlxwinograd.h"
#include "../neuralnet/mlxwinotuner.h"
#include "../core/global.h"
#include "../core/test.h"

#include <mlx/mlx.h>
#include <KataGoSwift/KataGoSwift-swift.h>
#include <katagocoreml/KataGoConverter.hpp>
#include <ghc/filesystem.hpp>
#include <chrono>
#include <unistd.h>  // For getpid()
#include <iostream>
#include <cstring>
#include <cstdlib>     // malloc / std::getenv
#include <memory>
#include <mutex>
#include <optional>
#include <map>
#include <tuple>
#include <random>
#include <array>
#include <cmath>

// Test-only free functions, defined in mlxtests.cpp. Invoked once per
// process from testEvaluateConv via the ranMLXAuxTests guard.
void runMLXWinogradTests();
void runMLXWinotunerTests();

namespace mx = mlx::core;

// Type alias for compiled inference functions
using CompiledInferenceFunc = std::function<std::vector<mx::array>(const std::vector<mx::array>&)>;

// Cache key: (batchSize, nnXLen, nnYLen, useMask, hasMeta, useFP16)
using CompileCacheKey = std::tuple<int, int, int, bool, bool, bool>;
using namespace std;

// MUX modes: gpuIdx selects per-thread execution path.
// Same convention the Metal backend uses (METAL_MUX_GPU / METAL_MUX_ANE).
static constexpr int MLX_MUX_GPU = 0;    // MLX/GPU - default
static constexpr int MLX_MUX_ANE = 100;  // CoreML on CPU+ANE via katagocoreml + KataGoSwift

// Serializes ComputeHandle construction across server threads. The CoreML
// converter (katagocoreml::KataGoConverter::convert) holds process-global
// MIL writer state that is not reentrant; without this lock, 2+ ANE threads
// racing at startup corrupt the .mlpackage and throw "Metadata written to
// different offset than expected." Mirrors metalbackend.cpp's
// computeHandleMutex.
static std::mutex computeHandleMutex;

// Serializes MLX/GPU graph construction and command encoding across NN server
// threads. MLX's GPU streams have no per-stream worker thread (scheduler.h
// pushes nullptr for gpu streams); encoding runs inline on the calling thread
// and every handle shares MLX's single global default GPU stream. Two server
// threads encoding concurrently therefore open two compute command encoders on
// the same MTLCommandBuffer, which aborts with the Metal assertion "A command
// encoder is already encoding to this command buffer". So applyCompiled holds
// this lock across input-array creation, the compiled-graph call, and
// mx::async_eval (which performs all Metal command encoding synchronously on
// the calling thread before returning). The wait for GPU completion
// (array::wait, a per-array shared-event block) and the result readback
// (data<float>() on already-materialized buffers) do not encode, so they run
// OUTSIDE the lock: with numNNServerThreadsPerModel=2, one server thread
// encodes its batch while the other waits on the GPU, keeping the GPU fed
// back-to-back instead of idling during encode + readback.
static std::mutex mlxGpuEvalMutex;

//------------------------------------------------------------------------------
// CoreML Model Conversion - reuses katagocoreml library, mirrors metalbackend.cpp
//------------------------------------------------------------------------------

namespace gfs = ghc::filesystem;

namespace CoreMLConversion {

// Get temp directory for model conversion. Identical path to Metal's
// getTempDirectory() in metalbackend.cpp so a .mlpackage produced by either
// backend can be reused by the other on a same-model run.
static string getTempDirectory() {
  gfs::path tempDir = gfs::temp_directory_path() / "katago_coreml";
  std::error_code ec;
  gfs::create_directories(tempDir, ec);
  if(ec) {
    throw runtime_error("Failed to create temp directory: " + ec.message());
  }
  return tempDir.string();
}

// Generate unique temporary path for model conversion
static string generateTempPath(int serverThreadIdx) {
  auto now = chrono::steady_clock::now().time_since_epoch().count();
  return getTempDirectory() + "/model_" + to_string(getpid()) + "_" +
         to_string(serverThreadIdx) + "_" + to_string(now) + ".mlpackage";
}

// CoreML model metadata constants
static const string COREML_MODEL_AUTHOR = "KataGo";
static const string COREML_MODEL_LICENSE = "See original model file for license terms";

// Convert KataGo model to CoreML in temp directory, returns path to .mlpackage.
// The caller (Swift side) is responsible for deleting the temp file after loading:
// see deleteSourceModel in metalbackend.swift, invoked via `defer` from
// createCoreMLComputeHandle.
static string convertModelToTemp(
  const string& modelPath,
  int boardX,
  int boardY,
  bool useFP16,
  bool optimizeMask,
  int maxBatchSize,
  int serverThreadIdx
) {
  // maxBatchSize is validated upstream: cfg.getInt("nnMaxBatchSize", 1, 65536) in setup.cpp
  // and NNEvaluator constructor throws if maxBatchSize <= 0. Assert for defensive documentation.
  assert(maxBatchSize >= 1);

  string tempPath = generateTempPath(serverThreadIdx);
  cerr << "MLX backend " << serverThreadIdx << ": Converting model to " << tempPath << endl;

  katagocoreml::ConversionOptions opts;
  opts.board_x_size = boardX;
  opts.board_y_size = boardY;
  opts.compute_precision = useFP16 ? "FLOAT16" : "FLOAT32";
  opts.optimize_identity_mask = optimizeMask;
  opts.min_batch_size = 1;
  opts.max_batch_size = maxBatchSize;
  opts.author = COREML_MODEL_AUTHOR;
  opts.license = COREML_MODEL_LICENSE;

  try {
    katagocoreml::KataGoConverter::convert(modelPath, tempPath, opts);
  } catch(const exception& e) {
    // Clean up partial conversion on failure
    std::error_code ec;
    gfs::remove_all(tempPath, ec);
    if(ec) {
      cerr << "MLX backend " << serverThreadIdx << ": Warning: Failed to clean up partial conversion at " << tempPath << ": " << ec.message() << endl;
    }
    throw runtime_error(string("MLX backend ") + to_string(serverThreadIdx) + ": Core ML model conversion failed: " + e.what());
  }

  cerr << "MLX backend " << serverThreadIdx << ": Conversion completed" << endl;
  return tempPath;
}

}  // namespace CoreMLConversion

// LoadedModel / ModelDesc ---------------------------------------------------------------------------------------------

struct LoadedModel {
  ModelDesc modelDesc;
  // Source path of the .bin.gz, retained for CoreML/ANE mux: the katagocoreml
  // converter needs the on-disk source to produce a .mlpackage. The MLX GPU
  // path does not read this field.
  string modelPath;

  LoadedModel(const string& fileName, const string& expectedSha256) {
    ModelDesc::loadFromFileMaybeGZipped(fileName, modelDesc, expectedSha256);
    modelPath = fileName;
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

const ModelDesc& NeuralNet::getModelDesc(const LoadedModel* loadedModel) {
  return loadedModel->modelDesc;
}

// Helpers --------------------------------------------------------------------------------------------------------------

// Convert convolution weights from OIHW to OHWI (MLX conv2d weight format)
static mx::array convertConvWeightsOIHWtoOHWI(const vector<float>& weights,
                                               int outChannels, int inChannels,
                                               int kH, int kW) {
  // Original: [outC, inC, kH, kW] - stored in column-major order
  // Target: [outC, kH, kW, inC]
  vector<float> converted(weights.size());
  for (int oc = 0; oc < outChannels; oc++) {
    for (int ic = 0; ic < inChannels; ic++) {
      for (int h = 0; h < kH; h++) {
        for (int w = 0; w < kW; w++) {
          int srcIdx = ((oc * inChannels + ic) * kH + h) * kW + w;
          int dstIdx = ((oc * kH + h) * kW + w) * inChannels + ic;
          converted[dstIdx] = weights[srcIdx];
        }
      }
    }
  }
  mx::Shape shape = {outChannels, kH, kW, inChannels};
  return mx::array(converted.data(), shape, mx::float32);
}

// Convert array to compute dtype. Lazy form for the inference hot path
// (each call's astype goes into the compiled trace; evaluating eagerly
// would force a stream sync per inference).
static mx::array toComputeDtype(const mx::array& arr, bool useFP16) {
  return useFP16 ? mx::astype(arr, mx::float16) : arr;
}

// Convert array to compute dtype and materialize the result.
//
// Use this for STATIC layer weights cached on a shared Model (the
// `cachedModels` map below shares a single Model instance across all
// MLX/GPU server threads). Without the eval, fp16 weights are
// unevaluated AsType primitives stamped with the constructor thread's
// MLX Stream; any other thread that later evals a compiled graph that
// captures these weights throws "There is no Stream(gpu, N) in current
// thread." with N = the constructor thread's stream index. MLX
// 0.31.2's command encoders live in `thread_local` storage inside
// mlx-core's metal/device.cpp, so a stream created on thread A is
// unreachable from thread B.
static mx::array toComputeDtypeMaterialized(const mx::array& arr, bool useFP16) {
  if(!useFP16) return arr;
  mx::array result = mx::astype(arr, mx::float16);
  mx::eval(result);
  return result;
}

// Mish activation: x * tanh(softplus(x)) = x * tanh(log(1 + exp(x)))
//
// Numerical stability: softplus is computed via logaddexp(0, x), which MLX
// implements as max(0, x) + log1p(exp(-|x|)) (see mlx/backend/cpu/binary_ops.h
// LogAddExp). The exp argument is always in (-inf, 0], so exp(-|x|) lies in
// (0, 1] and cannot overflow in either FP32 or FP16. This is why MLX does
// not need the ACTIVATION_MISH_SCALE8 variant that CUDA/OpenCL/TensorRT apply
// at model load (each backend calls modelDesc.applyScale8ToReduceActivations,
// implemented in desc.cpp) to keep Mish inside FP16
// representable range: those backends compute softplus via a path that
// overflows for x >~ 11 in FP16 (since exp(11.09) >~ 65504 = FP16 max).
// Cross-backend validation against an Eigen FP32 reference confirms FP16
// MLX is within typical half-precision tolerance with no Mish-overflow
// artifacts (see testgpuerror workflow in CLAUDE.md).
static mx::array applyMish(const mx::array& x) {
  // softplus(x) = log(1 + exp(x)) = log(exp(0) + exp(x)) = logaddexp(0, x).
  // MLX's logaddexp uses max(0,x) + log1p(exp(-|x|)) -- overflow-free.
  mx::array softplus = mx::logaddexp(mx::array(0.0f), x);
  return x * mx::tanh(softplus);
}

// Apply activation function
static mx::array applyActivation(const mx::array& x, int activationType) {
  switch(activationType) {
    case ACTIVATION_RELU:
      return mx::maximum(x, mx::array(0.0f));
    case ACTIVATION_MISH:
      return applyMish(x);
    case ACTIVATION_SILU:
      // SiLU (swish): x * sigmoid(x) = x / (1 + exp(-x)). Matches Eigen's
      // ACTIVATION_SILU (x / ((-x).exp() + 1)). MLX's sigmoid is overflow-safe.
      return x * mx::sigmoid(x);
    case ACTIVATION_MISH_SCALE8:
      // ACTIVATION_MISH_SCALE8 is an FP16-numerics workaround applied in-place
      // at model load by CUDA/OpenCL/TensorRT (see desc.cpp:applyScale8To-
      // ReduceActivations). MLX does not call that transform because its
      // logaddexp-based softplus is already overflow-free in FP16 (see
      // applyMish above), so we should never see this enum here. If a model
      // ever ships with MISH_SCALE8 baked in on disk, fail loudly rather than
      // silently fall through to identity. Mirrors Eigen/Metal behavior.
      testAssert(false);
      return x;  // unreached; satisfies compiler
    case ACTIVATION_IDENTITY:
    default:
      return x;
  }
}

// Fused matmul + bias: result = input @ weights + bias
// Uses addmm for better performance (single kernel instead of matmul + add)
static mx::array matmulBias(const mx::array& input, const mx::array& weights, const mx::array& bias) {
  // addmm(c, a, b, alpha, beta) = alpha * (a @ b) + beta * c
  return mx::addmm(bias, input, weights, 1.0f, 1.0f);
}

// Winograd is on by default; KATAGO_MLX_WINOGRAD=0 forces mx::conv2d
// (A/B correctness testing and runtime safety valve).
static bool mlxWinogradEnabled() {
  static const bool enabled = [](){
    const char* e = std::getenv("KATAGO_MLX_WINOGRAD");
    return !(e != nullptr && std::string(e) == "0");
  }();
  return enabled;
}

// Tuner is on by default; KATAGO_MLX_WINOTUNER=0 forces baked defaults.
static bool mlxWinotunerEnabled() {
  static const bool enabled = [](){
    const char* e = std::getenv("KATAGO_MLX_WINOTUNER");
    return !(e != nullptr && std::string(e) == "0");
  }();
  return enabled;
}
// Layers --------------------------------------------------------------------------------------------------------------

struct ConvLayer {
  const string name;
  const int convYSize;
  const int convXSize;
  const int inChannels;
  const int outChannels;
  const int dilationY;
  const int dilationX;
  const bool useFP16;
  const bool useWinograd;
  mx::array weights;            // OHWI format (only built when !useWinograd)
  mx::array winogradWeights;    // 4x4 domain U, valid only if useWinograd
  const MLXWinograd::InputTransform    winoInCfg;
  const MLXWinograd::OutputUntransform winoOutCfg;

  ConvLayer() = delete;
  ConvLayer(const ConvLayer&) = delete;
  ConvLayer& operator=(const ConvLayer&) = delete;

  ConvLayer(const ConvLayerDesc& desc,
            const MLXWinograd::InputTransform& inCfg,
            const MLXWinograd::OutputUntransform& outCfg,
            bool useFP16_ = false)
    : name(desc.name),
      convYSize(desc.convYSize),
      convXSize(desc.convXSize),
      inChannels(desc.inChannels),
      outChannels(desc.outChannels),
      dilationY(desc.dilationY),
      dilationX(desc.dilationX),
      useFP16(useFP16_),
      // Winograd path runs in fp16 too (no `!useFP16` gate).
      useWinograd(mlxWinogradEnabled()
                  && convYSize==3 && convXSize==3
                  && dilationY==1 && dilationX==1),
      weights(useWinograd ? mx::array(0.0f) : toComputeDtypeMaterialized(convertConvWeightsOIHWtoOHWI(desc.weights, outChannels, inChannels, convYSize, convXSize), useFP16_)),
      winogradWeights(useWinograd
        ? MLXWinograd::makeWinogradWeights(desc.weights, outChannels, inChannels, useFP16_)
        : mx::array(0.0f))
      ,winoInCfg(inCfg)
      ,winoOutCfg(outCfg)
  {}

  mx::array apply(const mx::array& input) const {
    if(useWinograd) {
      return MLXWinograd::winogradConv2d(input, winogradWeights, outChannels, winoInCfg, winoOutCfg, useFP16);
    }
    // MLX conv2d: input NHWC, weights OHWI
    // Compute padding to maintain spatial dimensions (same padding)
    int padY = (convYSize - 1) * dilationY / 2;
    int padX = (convXSize - 1) * dilationX / 2;

    return mx::conv2d(
      input,
      weights,
      /*stride=*/std::make_pair(1, 1),
      /*padding=*/std::make_pair(padY, padX),
      /*dilation=*/std::make_pair(dilationY, dilationX),
      /*groups=*/1
    );
  }
};

struct BatchNormLayer {
  const string name;
  const int numChannels;
  const int activation;
  const bool useFP16;
  mx::array mergedScale; // Shape: [C], always fp32
  mx::array mergedBias;  // Shape: [C], always fp32

  BatchNormLayer() = delete;
  BatchNormLayer(const BatchNormLayer&) = delete;
  BatchNormLayer& operator=(const BatchNormLayer&) = delete;

  // mergedScale/mergedBias storage is always fp32 to preserve dynamic
  // range across the 25-block-deep b18c384 chain. The `useFP16` parameter
  // is intentionally ignored.
  static mx::array createArray1D(const std::vector<float>& data, int size, bool /*useFP16*/) {
    mx::Shape shape = {size};
    return mx::array(data.data(), shape, mx::float32);
  }

  static std::vector<float> getMergedScale(const BatchNormLayerDesc& desc) {
    // If mergedScale is already computed, use it
    if(!desc.mergedScale.empty()) {
      return desc.mergedScale;
    }
    // Otherwise compute from mean/variance/scale/bias (for tests)
    std::vector<float> mergedScale(desc.numChannels);
    for(int c = 0; c < desc.numChannels; c++) {
      mergedScale[c] = desc.scale[c] / sqrt(desc.variance[c] + desc.epsilon);
    }
    return mergedScale;
  }

  static std::vector<float> getMergedBias(const BatchNormLayerDesc& desc) {
    // If mergedBias is already computed, use it
    if(!desc.mergedBias.empty()) {
      return desc.mergedBias;
    }
    // Otherwise compute from mean/variance/scale/bias (for tests)
    std::vector<float> mergedBias(desc.numChannels);
    for(int c = 0; c < desc.numChannels; c++) {
      float ms = desc.scale[c] / sqrt(desc.variance[c] + desc.epsilon);
      mergedBias[c] = desc.bias[c] - ms * desc.mean[c];
    }
    return mergedBias;
  }

  BatchNormLayer(const BatchNormLayerDesc& desc, int activationType, bool useFP16_ = false)
    : name(desc.name),
      numChannels(desc.numChannels),
      activation(activationType),
      useFP16(useFP16_),
      mergedScale(createArray1D(getMergedScale(desc), desc.numChannels, useFP16_)),
      mergedBias(createArray1D(getMergedBias(desc), desc.numChannels, useFP16_))
  {}

  mx::array apply(const mx::array& input, const mx::array& mask, bool useMask) const {
    // input: NHWC [N, H, W, C] in compute dtype (fp16 or fp32).
    // mask: NHW1 [N, H, W, 1] in compute dtype.
    // mergedScale/mergedBias are always fp32; MLX type promotion lifts the
    // multiply-add-activation chain to fp32 automatically (selective fp32
    // accumulation — defense against inf/nan in deep stacks).
    // Mask multiply runs while activated is still fp32 (safe because mask is
    // binary 0/1, so fp32*fp16 and fp16*fp16 round to bit-equal results).
    // The single trailing astype-to-fp16 covers both useMask branches.
    mx::array normalized = input * mergedScale + mergedBias;
    mx::array activated = applyActivation(normalized, activation);
    if(useMask)
      activated = activated * mask;
    // Cast back to fp16 so downstream layers see the expected compute dtype.
    if(useFP16) activated = mx::astype(activated, mx::float16);
    return activated;
  }
};

struct MatMulLayer {
  const string name;
  const int inChannels;
  const int outChannels;
  mx::array weights; // [inC, outC]

  MatMulLayer() = delete;
  MatMulLayer(const MatMulLayer&) = delete;
  MatMulLayer& operator=(const MatMulLayer&) = delete;

  static mx::array createWeights(const MatMulLayerDesc& desc, bool useFP16) {
    if(desc.inChannels > 0 && desc.outChannels > 0) {
      // Original weights: [inC, outC] (column-major)
      mx::Shape shape = {desc.inChannels, desc.outChannels};
      mx::array arr = mx::array(desc.weights.data(), shape, mx::float32);
      return toComputeDtypeMaterialized(arr, useFP16);
    }
    std::vector<float> dummy = {0.0f};
    mx::Shape shape = {1};
    return mx::array(dummy.data(), shape, mx::float32);
  }

  MatMulLayer(const MatMulLayerDesc& desc, bool useFP16 = false)
    : name(desc.name),
      inChannels(desc.inChannels),
      outChannels(desc.outChannels),
      weights(createWeights(desc, useFP16))
  {}

  mx::array apply(const mx::array& input) const {
    // input: [N, inC]
    // output: [N, outC]
    return mx::matmul(input, weights);
  }
};

struct MatBiasLayer {
  const string name;
  const int numChannels;
  mx::array bias;

  MatBiasLayer() = delete;
  MatBiasLayer(const MatBiasLayer&) = delete;
  MatBiasLayer& operator=(const MatBiasLayer&) = delete;

  static mx::array createBias(const MatBiasLayerDesc& desc, bool useFP16) {
    mx::Shape shape = {desc.numChannels};
    mx::array arr = mx::array(desc.weights.data(), shape, mx::float32);
    return toComputeDtypeMaterialized(arr, useFP16);
  }

  MatBiasLayer(const MatBiasLayerDesc& desc, bool useFP16 = false)
    : name(desc.name),
      numChannels(desc.numChannels),
      bias(createBias(desc, useFP16))
  {}

  mx::array apply(const mx::array& input) const {
    return input + bias;
  }
};

// --------------------------------------------------------------------------------------------------------------
// Transformer layers (RMSNorm / attention / FFN).
//
// MLX operates on NHWC arrays [N, H, W, C]; C is the fastest (last) axis. The
// Eigen ground-truth (eigenbackend.cpp) operates on (C, W, H, N) col-major and
// reshapes spatial dims to a sequence. The math below mirrors Eigen exactly
// but stays in MLX's native NHWC layout: the spatial dims H,W are adjacent and
// C is last, so [N,H,W,C] reshapes contiguously to [N, seq, C] with
// seq = H*W = nnYLen*nnXLen. The mask [N,H,W,1] reshapes to [N, seq, 1].
//
// All RMSNorm gamma/beta/weight buffers stay fp32 (like BatchNormLayer) to
// preserve dynamic range; MLX type promotion lifts the normalize chain to fp32
// and the trailing astype returns to compute dtype.
// --------------------------------------------------------------------------------------------------------------

// Lightweight RMSNorm used inside transformer blocks (weight only, no bias, no
// spatial mode, no activation). Mirrors Eigen TransformerRMSNormLayer
// (eigenbackend.cpp 866-918).
struct TransformerRMSNormLayer {
  const string name;
  const int numChannels;
  const float epsilon;
  const bool useFP16;
  mx::array weight;  // [C], always fp32

  TransformerRMSNormLayer() = delete;
  TransformerRMSNormLayer(const TransformerRMSNormLayer&) = delete;
  TransformerRMSNormLayer& operator=(const TransformerRMSNormLayer&) = delete;

  static mx::array createWeight(const TransformerRMSNormDesc& desc) {
    mx::Shape shape = {desc.numChannels};
    return mx::array(desc.weight.data(), shape, mx::float32);
  }

  TransformerRMSNormLayer(const TransformerRMSNormDesc& desc, bool useFP16_ = false)
    : name(desc.name),
      numChannels(desc.numChannels),
      epsilon(desc.epsilon),
      useFP16(useFP16_),
      weight(createWeight(desc))
  {}

  // input/output NHWC [N, H, W, C] in compute dtype. mask NHW1 [N, H, W, 1].
  // Per-position RMSNorm across channels: out = x * rsqrt(mean(x^2) + eps) * weight,
  // then masked to zero on invalid positions (Eigen zeroes masked positions).
  mx::array apply(const mx::array& input, const mx::array& mask, bool useMask) const {
    // Variance reduction in fp32 (Eigen computes sumSq in fp32, eigenbackend.cpp
    // 904-910). mx::mean over an fp16 operand would accumulate the per-channel
    // sum of squares in fp16; promote to fp32 first so the reduction is
    // overflow/precision-safe before rsqrt.
    mx::array inputF32 = useFP16 ? mx::astype(input, mx::float32) : input;
    std::vector<int> chAxis = {3};
    mx::array meanSq = mx::mean(inputF32 * inputF32, chAxis, /*keepdims=*/true);  // [N,H,W,1], fp32
    mx::array rms = mx::rsqrt(meanSq + mx::array(epsilon));
    mx::array normalized = input * rms * weight;  // weight fp32 promotes chain to fp32
    if(useMask)
      normalized = normalized * mask;
    if(useFP16) normalized = mx::astype(normalized, mx::float16);
    return normalized;
  }
};

// Full-featured RMSNorm for the trunk tip: gamma/beta, optional spatial mode,
// optional activation. Mirrors Eigen RMSNormLayer (eigenbackend.cpp 922-1022).
struct TransformerTrunkRMSNormLayer {
  const string name;
  const int numChannels;
  const float epsilon;
  const bool spatial;
  const int activation;
  const bool useFP16;
  mx::array gamma;  // [C], always fp32
  mx::array beta;   // [C], always fp32

  TransformerTrunkRMSNormLayer() = delete;
  TransformerTrunkRMSNormLayer(const TransformerTrunkRMSNormLayer&) = delete;
  TransformerTrunkRMSNormLayer& operator=(const TransformerTrunkRMSNormLayer&) = delete;

  static mx::array createVec(const std::vector<float>& data, int size) {
    mx::Shape shape = {size};
    return mx::array(data.data(), shape, mx::float32);
  }

  TransformerTrunkRMSNormLayer(const RMSNormLayerDesc& desc, int activation_, bool useFP16_ = false)
    : name(desc.name),
      numChannels(desc.numChannels),
      epsilon(desc.epsilon),
      spatial(desc.spatial),
      activation(activation_),
      useFP16(useFP16_),
      gamma(createVec(desc.gamma, desc.numChannels)),
      beta(createVec(desc.beta, desc.numChannels))
  {}

  // input/output NHWC [N, H, W, C]. mask NHW1 [N, H, W, 1]. maskSum N111.
  mx::array apply(const mx::array& input, const mx::array& mask, const mx::array& maskSum, bool useMask) const {
    // The variance reduction (sum/mean of squares) MUST run in fp32 even when
    // the compute dtype is fp16. Eigen computes sumSq in fp32 (eigenbackend.cpp
    // 968-1002). In fp16 the spatial branch sums x^2 over up to
    // nnXLen*nnYLen*numChannels (e.g. 19*19*256 ≈ 92k) elements; an fp16
    // accumulator saturates at 65504 and the rsqrt then yields 0/NaN/Inf which
    // poisons the whole net (observed ~50% winrate / 99% top-policy error on the
    // SiLU + spatial-rmsnorm-tip transformer). Promoting input to fp32 for the
    // reduction makes the reduction overflow-free; the result rms is fp32 and the
    // downstream normalize chain stays fp32 (gamma/beta are fp32 too) until the
    // single trailing astype back to fp16.
    mx::array inputF32 = useFP16 ? mx::astype(input, mx::float32) : input;
    mx::array rms = mx::array(0.0f);
    if(!spatial) {
      // Non-spatial: per-position RMS across channels.
      std::vector<int> chAxis = {3};
      mx::array meanSq = mx::mean(inputF32 * inputF32, chAxis, /*keepdims=*/true);  // [N,H,W,1], fp32
      rms = mx::rsqrt(meanSq + mx::array(epsilon));
    }
    else {
      // Spatial: per-batch RMS across all valid spatial positions AND channels.
      // sumSq over valid positions only (Eigen skips masked positions);
      // totalElts = count * numChannels with count = maskSum (valid spatial count).
      std::vector<int> spatialChAxes = {1, 2, 3};
      mx::array x2 = inputF32 * inputF32;  // fp32 reduction operand (overflow-safe)
      if(useMask)
        x2 = x2 * mask;  // mask is [N,H,W,1], broadcasts over channels
      mx::array sumSq = mx::sum(x2, spatialChAxes, /*keepdims=*/true);  // [N,1,1,1], fp32
      // maskSum is [N,1,1,1] (valid spatial position count); when !useMask it is
      // a constant nnXLen*nnYLen full array, equally valid.
      mx::array totalElts = maskSum * mx::array((float)numChannels);
      rms = mx::rsqrt(sumSq / totalElts + mx::array(epsilon));  // [N,1,1,1], fp32
    }
    mx::array normalized = input * rms * gamma + beta;  // gamma/beta fp32 promote chain
    normalized = applyActivation(normalized, activation);
    if(useMask)
      normalized = normalized * mask;
    if(useFP16) normalized = mx::astype(normalized, mx::float16);
    return normalized;
  }
};

// Transformer attention block: GQA + optional RoPE + masked scaled-dot-product
// attention. Mirrors Eigen TransformerAttentionBlock (eigenbackend.cpp 1307-1588).
struct TransformerAttentionBlock {
  const string name;
  const int numHeads;
  const int numKVHeads;
  const int qHeadDim;
  const int vHeadDim;
  const bool useRope;
  const bool learnableRope;
  const int inChannels;
  const bool useFP16;
  const int nnXLen;
  const int nnYLen;

  const TransformerRMSNormLayer preLN;
  const MatMulLayer qProj;
  const MatMulLayer kProj;
  const MatMulLayer vProj;
  const MatMulLayer outProj;

  // Precomputed RoPE cos/sin tables, materialized as MLX arrays (always fp32).
  // Learnable layout (after reshaping for our use): [numKVHeads, numPairs, seq].
  // Fixed layout: [numPairs, seq]. ropeNumPairs = qHeadDim/2.
  int ropeNumPairs;
  mx::array ropeCos;  // valid iff useRope
  mx::array ropeSin;  // valid iff useRope

  TransformerAttentionBlock() = delete;
  TransformerAttentionBlock(const TransformerAttentionBlock&) = delete;
  TransformerAttentionBlock& operator=(const TransformerAttentionBlock&) = delete;

  static mx::array makeRopeTable(const std::vector<float>& table, bool learnable, int numKVHeads, int numPairs, int seq) {
    if(table.empty())
      return mx::array(0.0f);
    if(learnable) {
      mx::Shape shape = {numKVHeads, numPairs, seq};
      return mx::array(table.data(), shape, mx::float32);
    }
    mx::Shape shape = {numPairs, seq};
    return mx::array(table.data(), shape, mx::float32);
  }

  TransformerAttentionBlock(const TransformerAttentionDesc& desc, int nnX, int nnY, bool useFP16_ = false)
    : name(desc.name),
      numHeads(desc.numHeads),
      numKVHeads(desc.numKVHeads),
      qHeadDim(desc.qHeadDim),
      vHeadDim(desc.vHeadDim),
      useRope(desc.useRope),
      learnableRope(desc.learnableRope),
      inChannels(desc.qProj.inChannels),
      useFP16(useFP16_),
      nnXLen(nnX),
      nnYLen(nnY),
      preLN(desc.preLN, useFP16_),
      qProj(desc.qProj, useFP16_),
      kProj(desc.kProj, useFP16_),
      vProj(desc.vProj, useFP16_),
      outProj(desc.outProj, useFP16_),
      ropeNumPairs(0),
      ropeCos(mx::array(0.0f)),
      ropeSin(mx::array(0.0f))
  {
    if(useRope) {
      ropeNumPairs = qHeadDim / 2;
      int seq = nnX * nnY;
      int paddedNNXYLen = seq;
      std::vector<float> cosTable, sinTable;
      desc.computeRopeCosSin(nnX, nnY, paddedNNXYLen, cosTable, sinTable);
      ropeCos = makeRopeTable(cosTable, learnableRope, numKVHeads, ropeNumPairs, seq);
      ropeSin = makeRopeTable(sinTable, learnableRope, numKVHeads, ropeNumPairs, seq);
    }
  }

  // Apply RoPE to a projection laid out as [N, seq, numBufHeads, headDim].
  // RoPE rotates interleaved channel pairs (2p, 2p+1) within each head.
  // cos/sin tables map per (kvHead-of-this-head, pair, seq). Returns rotated
  // array in the same shape. headDim == qHeadDim here.
  mx::array applyRope(const mx::array& proj, int numBufHeads) const {
    int seq = nnXLen * nnYLen;
    int batchSize = proj.shape()[0];
    // Split even/odd channel pairs. proj: [N, seq, numBufHeads, qHeadDim].
    // Reshape to [N, seq, numBufHeads, ropeNumPairs, 2] then take [...,0]/[...,1].
    mx::Shape pairShape = {batchSize, seq, numBufHeads, ropeNumPairs, 2};
    mx::array pairs = mx::reshape(proj, pairShape);
    mx::array x0 = mx::squeeze(mx::slice(pairs, {0,0,0,0,0}, {batchSize, seq, numBufHeads, ropeNumPairs, 1}), std::vector<int>{4});  // [N,seq,H,pairs]
    mx::array x1 = mx::squeeze(mx::slice(pairs, {0,0,0,0,1}, {batchSize, seq, numBufHeads, ropeNumPairs, 2}), std::vector<int>{4});  // [N,seq,H,pairs]

    // Build cos/sin broadcastable to [N, seq, numBufHeads, ropeNumPairs].
    // Source tables: learnable [numKVHeads, pairs, seq]; fixed [pairs, seq].
    // Target per-head index kvh = h * numKVHeads / numBufHeads (Eigen mapping).
    mx::array cosB = mx::array(0.0f);
    mx::array sinB = mx::array(0.0f);
    if(learnableRope) {
      // Expand each KV head to the Q heads that map to it. For head h,
      // kvh = h * numKVHeads / numBufHeads. With numBufHeads a multiple of
      // numKVHeads, this is a contiguous block expansion: repeat each kv head
      // (numBufHeads / numKVHeads) times along the head axis.
      int groupSize = numBufHeads / numKVHeads;
      // ropeCos: [numKVHeads, pairs, seq] -> [numBufHeads, pairs, seq]
      mx::array cosHeads = mx::repeat(ropeCos, groupSize, /*axis=*/0);  // [numBufHeads, pairs, seq]
      mx::array sinHeads = mx::repeat(ropeSin, groupSize, /*axis=*/0);
      // -> [seq, numBufHeads, pairs] then expand batch axis
      cosHeads = mx::transpose(cosHeads, std::vector<int>{2, 0, 1});  // [seq, numBufHeads, pairs]
      sinHeads = mx::transpose(sinHeads, std::vector<int>{2, 0, 1});
      cosB = mx::expand_dims(cosHeads, 0);  // [1, seq, numBufHeads, pairs]
      sinB = mx::expand_dims(sinHeads, 0);
    }
    else {
      // ropeCos: [pairs, seq] -> [seq, pairs], broadcast over heads.
      mx::array cosSP = mx::transpose(ropeCos, std::vector<int>{1, 0});  // [seq, pairs]
      mx::array sinSP = mx::transpose(ropeSin, std::vector<int>{1, 0});
      cosB = mx::expand_dims(mx::expand_dims(cosSP, 1), 0);  // [1, seq, 1, pairs]
      sinB = mx::expand_dims(mx::expand_dims(sinSP, 1), 0);
    }
    if(useFP16) {
      cosB = mx::astype(cosB, mx::float16);
      sinB = mx::astype(sinB, mx::float16);
    }

    mx::array r0 = x0 * cosB - x1 * sinB;  // [N,seq,H,pairs]
    mx::array r1 = x0 * sinB + x1 * cosB;
    // Re-interleave: stack along new last axis -> [N,seq,H,pairs,2] -> [N,seq,H,qHeadDim].
    mx::array stacked = mx::stack(std::vector<mx::array>{r0, r1}, /*axis=*/4);
    mx::Shape outShape = {batchSize, seq, numBufHeads, ropeNumPairs * 2};
    return mx::reshape(stacked, outShape);
  }

  // input/output NHWC [N, H, W, C]. mask NHW1 [N, H, W, 1].
  mx::array apply(const mx::array& trunk, const mx::array& mask, bool useMask) const {
    int batchSize = trunk.shape()[0];
    int seq = nnXLen * nnYLen;
    int kvGroupSize = numHeads / numKVHeads;
    float scale = 1.0f / sqrtf((float)qHeadDim);

    // Step 1: preLN RMSNorm (masks output to zero on invalid positions).
    mx::array normed = preLN.apply(trunk, mask, useMask);  // [N,H,W,C]

    // Flatten spatial dims to a sequence: [N, seq, C].
    mx::Shape seqShape = {batchSize, seq, inChannels};
    mx::array normedSeq = mx::reshape(normed, seqShape);

    // Step 2: Q/K/V projections. weights are [inC, outC]; x[N,seq,inC] @ W.
    mx::array q = mx::matmul(normedSeq, qProj.weights);  // [N, seq, numHeads*qHeadDim]
    mx::array k = mx::matmul(normedSeq, kProj.weights);  // [N, seq, numKVHeads*qHeadDim]
    mx::array v = mx::matmul(normedSeq, vProj.weights);  // [N, seq, numKVHeads*vHeadDim]

    // Reshape to per-head: [N, seq, numHeads, qHeadDim] etc.
    q = mx::reshape(q, mx::Shape{batchSize, seq, numHeads, qHeadDim});
    k = mx::reshape(k, mx::Shape{batchSize, seq, numKVHeads, qHeadDim});
    v = mx::reshape(v, mx::Shape{batchSize, seq, numKVHeads, vHeadDim});

    // Step 3: RoPE on Q and K.
    if(useRope) {
      q = applyRope(q, numHeads);
      k = applyRope(k, numKVHeads);
    }

    // Move head axis ahead of seq: [N, numHeads, seq, headDim].
    q = mx::transpose(q, std::vector<int>{0, 2, 1, 3});  // [N, numHeads, seq, qHeadDim]
    k = mx::transpose(k, std::vector<int>{0, 2, 1, 3});  // [N, numKVHeads, seq, qHeadDim]
    v = mx::transpose(v, std::vector<int>{0, 2, 1, 3});  // [N, numKVHeads, seq, vHeadDim]

    // Expand KV heads to match Q heads (GQA): repeat each kv head kvGroupSize
    // times so head h uses kv head h/kvGroupSize (Eigen kvh = h / kvGroupSize).
    if(kvGroupSize > 1) {
      k = mx::repeat(k, kvGroupSize, /*axis=*/1);  // [N, numHeads, seq, qHeadDim]
      v = mx::repeat(v, kvGroupSize, /*axis=*/1);  // [N, numHeads, seq, vHeadDim]
    }

    // Step 4: scores = scale * Q @ K^T -> [N, numHeads, seq(query), seq(key)].
    // Keep the whole attention chain that follows (scores, softmax, attn@V,
    // out-proj, and the residual stream) in the compute dtype. MLX C++ has no
    // weak scalars: mx::array(scale) is a strong float32 array, so
    // `matmul(q,kT) * mx::array(scale)` would promote scores -- and everything
    // downstream, since each transformer block adds its output into the trunk --
    // to fp32. Cast scale to the compute dtype so the fp16 path stays fp16
    // end-to-end (maximize-fp16 on the GPU path; fp16 attention accuracy is gated
    // by testgpuerror). The pooling/BN/RMSNorm layers handle this same MLX
    // promotion rule by casting their fp32 intermediates back with a trailing
    // astype; attention does it here at the source instead.
    mx::array kT = mx::transpose(k, std::vector<int>{0, 1, 3, 2});  // [N, numHeads, qHeadDim, seq]
    mx::array scaleArr = useFP16 ? mx::astype(mx::array(scale), mx::float16) : mx::array(scale);
    mx::array scores = mx::matmul(q, kT) * scaleArr;

    // Masked softmax over the key axis (last). Keys with mask==0 get -inf so
    // they contribute 0; fully-masked query rows are zeroed afterward to match
    // Eigen (which zeroes masked query rows entirely).
    if(useMask) {
      // keyMask: [N, 1, 1, seq] broadcasting over heads and query positions.
      // Use 1e4 (not 1e9): representable in fp16, and exp(-1e4) underflows to 0
      // cleanly, avoiding inf arithmetic. Scores are O(1)*scale, so 1e4 fully
      // suppresses masked keys. The board is never fully masked, so no query row
      // is all-masked -> softmax never sees an all -inf row (no NaN).
      mx::array maskSeq = mx::reshape(mask, mx::Shape{batchSize, 1, 1, seq});  // [N,1,1,seq]
      mx::array neg = (mx::array(1.0f) - maskSeq) * mx::array(1e4f);
      if(useFP16) neg = mx::astype(neg, mx::float16);
      scores = scores - neg;
    }
    mx::array weights = mx::softmax(scores, /*axis=*/3, /*precise=*/true);  // [N, numHeads, seq, seq]

    // attnOut = weights @ V -> [N, numHeads, seq(query), vHeadDim].
    mx::array attn = mx::matmul(weights, v);

    if(useMask) {
      // Zero out fully-masked query rows (Eigen zeroes masked queries). With a
      // masked-key softmax a masked query still produces a normalized row, but
      // its residual contribution is gated by the trunk residual mask below, so
      // this extra gating is for parity; multiply by query mask.
      mx::array qMask = mx::reshape(mask, mx::Shape{batchSize, 1, seq, 1});  // [N,1,seq,1]
      attn = attn * qMask;
    }

    // Merge heads back: [N, numHeads, seq, vHeadDim] -> [N, seq, numHeads*vHeadDim].
    attn = mx::transpose(attn, std::vector<int>{0, 2, 1, 3});  // [N, seq, numHeads, vHeadDim]
    attn = mx::reshape(attn, mx::Shape{batchSize, seq, numHeads * vHeadDim});

    // Step 5: output projection -> [N, seq, outC] (outC == inChannels).
    mx::array out = mx::matmul(attn, outProj.weights);  // [N, seq, inChannels]
    mx::array outSpatial = mx::reshape(out, mx::Shape{batchSize, nnYLen, nnXLen, inChannels});

    // Step 6: residual + mask. (Eigen adds outProj * maskVal to trunk.)
    if(useMask)
      outSpatial = outSpatial * mask;
    return trunk + outSpatial;
  }
};

// Transformer FFN block: SwiGLU. Mirrors Eigen TransformerFFNBlock
// (eigenbackend.cpp 1592-1707). Non-SwiGLU is unsupported (Eigen throws too).
struct TransformerFFNBlock {
  const string name;
  const int numChannels;
  const int ffnChannels;
  const bool useSwiGLU;
  const bool useFP16;
  const int nnXLen;
  const int nnYLen;

  const TransformerRMSNormLayer preLN;
  const MatMulLayer linear1;
  unique_ptr<MatMulLayer> linearGate;
  const MatMulLayer linear2;

  TransformerFFNBlock() = delete;
  TransformerFFNBlock(const TransformerFFNBlock&) = delete;
  TransformerFFNBlock& operator=(const TransformerFFNBlock&) = delete;

  TransformerFFNBlock(const TransformerFFNDesc& desc, int nnX, int nnY, bool useFP16_ = false)
    : name(desc.name),
      numChannels(desc.numChannels),
      ffnChannels(desc.ffnChannels),
      useSwiGLU(desc.useSwiGLU),
      useFP16(useFP16_),
      nnXLen(nnX),
      nnYLen(nnY),
      preLN(desc.preLN, useFP16_),
      linear1(desc.linear1, useFP16_),
      linear2(desc.linear2, useFP16_)
  {
    if(useSwiGLU)
      linearGate = make_unique<MatMulLayer>(desc.linearGate, useFP16_);
    else
      throw StringError("MLX backend: Non-SwiGLU transformer FFN is not supported");
  }

  // input/output NHWC [N, H, W, C]. mask NHW1 [N, H, W, 1].
  mx::array apply(const mx::array& trunk, const mx::array& mask, bool useMask) const {
    int batchSize = trunk.shape()[0];
    int seq = nnXLen * nnYLen;

    // Step 1: preLN RMSNorm.
    mx::array normed = preLN.apply(trunk, mask, useMask);  // [N,H,W,C]
    mx::array normedSeq = mx::reshape(normed, mx::Shape{batchSize, seq, numChannels});

    // Step 2/3: SwiGLU = SiLU(linear1(x)) * linearGate(x).
    mx::array a = mx::matmul(normedSeq, linear1.weights);     // [N, seq, ffnChannels]
    mx::array gate = mx::matmul(normedSeq, linearGate->weights);
    mx::array swiglu = (a * mx::sigmoid(a)) * gate;          // SiLU(a) * gate

    // Step 4: linear2 -> [N, seq, numChannels].
    mx::array out = mx::matmul(swiglu, linear2.weights);
    mx::array outSpatial = mx::reshape(out, mx::Shape{batchSize, nnYLen, nnXLen, numChannels});

    // Step 5: residual + mask.
    if(useMask)
      outSpatial = outSpatial * mask;
    return trunk + outSpatial;
  }
};

// Global pooling: computes [mean, mean * (sqrt(maskSum) - 14) * 0.1, max] concatenated along channel axis
static mx::array applyGlobalPooling(const mx::array& input, const mx::array& mask, const mx::array& maskSum, bool useMask) {
  // input: NHWC [N, H, W, C]
  // mask: NHW1 [N, H, W, 1]
  // maskSum: N111 [N, 1, 1, 1]

  // Keep the whole pooling in the input's compute dtype (fp16 in fp16 mode) so
  // the pooled output matches the trunk and the downstream gpool-bias matmul
  // stays fp16. maskSum is the valid-position count (<=361 for 19x19), exact in
  // fp16. The fp32-accumulation variant measured negligible accuracy gain
  // (PR#1199 M1), so prefer fp16 consistency (minimize fp32 when useFP16).
  const auto dt = input.dtype();
  std::vector<int> spatialAxes = {1, 2};
  mx::array spatialSum = mx::sum(input, spatialAxes, /*keepdims=*/true); // [N, 1, 1, C], dt
  mx::array maskSumDt = mx::astype(maskSum, dt);

  // Mean = sum / maskSum
  mx::array mean = spatialSum / maskSumDt; // [N, 1, 1, C], dt

  // sqrt(maskSum) - 14) * 0.1   (scalar literals promote to fp32; cast result back to dt)
  mx::array sqrtMaskSum = mx::sqrt(maskSumDt);
  mx::array scaleFactor = (sqrtMaskSum - mx::array(14.0f)) * mx::array(0.1f);
  mx::array meanScaled = mx::astype(mean * scaleFactor, dt); // dt

  // Max - masked positions pushed down in fp32 (1e9 overflows fp16 -> inf/NaN),
  // then cast back to dt. Skip the mask adjustment when useMask=false.
  mx::array maxVal = useMask
    ? mx::max(input - (mx::array(1.0f) - mask) * mx::array(1e9f), spatialAxes, /*keepdims=*/true)
    : mx::max(input, spatialAxes, /*keepdims=*/true);
  maxVal = mx::astype(maxVal, dt); // dt

  // Concatenate along channel axis (axis 3 for NHWC); all components are dt
  std::vector<mx::array> concatInputs = {mean, meanScaled, maxVal};
  return mx::concatenate(concatInputs, /*axis=*/3);
}

// Value head pooling: computes [mean, mean * (sqrt(maskSum) - 14) * 0.1, mean * ((sqrt-14)^2 * 0.01 - 0.1)]
static mx::array applyValueHeadPooling(const mx::array& input, const mx::array& maskSum) {
  // input: NHWC [N, H, W, C]
  // maskSum: N111 [N, 1, 1, 1]

  // fp16-consistent (see applyGlobalPooling): keep the value-head pooling in the
  // compute dtype so the v2 matmul stays fp16.
  const auto dt = input.dtype();
  std::vector<int> spatialAxes = {1, 2};
  mx::array spatialSum = mx::sum(input, spatialAxes, /*keepdims=*/true); // dt
  mx::array maskSumDt = mx::astype(maskSum, dt);
  mx::array mean = spatialSum / maskSumDt; // dt

  mx::array sqrtMaskSum = mx::sqrt(maskSumDt);
  mx::array diff = sqrtMaskSum - mx::array(14.0f);
  mx::array meanScaled1 = mx::astype(mean * diff * mx::array(0.1f), dt);
  mx::array meanScaled2 = mx::astype(mean * (diff * diff * mx::array(0.01f) - mx::array(0.1f)), dt);

  std::vector<mx::array> concatInputs = {mean, meanScaled1, meanScaled2};
  return mx::concatenate(concatInputs, /*axis=*/3);
}

// Residual Block
// Map KataGo activation enum -> kWinoOutputSourceBNAct ACT id; -1 = unsupported
// (caller must fall back to the unfused path).
static int mapEpilogueAct(int activation) {
  switch(activation) {
    case ACTIVATION_IDENTITY: return 0;
    case ACTIVATION_MISH:     return 1;
    case ACTIVATION_RELU:     return 2;
    default:                  return -1;   // SILU etc.: not fused
  }
}

// conv -> BN+activation, fused into the Winograd untransform when the conv is
// Winograd, the batch is unmasked, and the activation is supported. Otherwise
// the exact pre-existing decomposition.
static mx::array fusedConvBNAct(const ConvLayer& conv, const BatchNormLayer& bn,
                                const mx::array& x, const mx::array& mask, bool useMask) {
  int act = mapEpilogueAct(bn.activation);
  if(conv.useWinograd && !useMask && act >= 0) {
    return MLXWinograd::winogradConv2d(
      x, conv.winogradWeights, conv.outChannels, conv.winoInCfg, conv.winoOutCfg, conv.useFP16,
      MLXWinograd::Epilogue::bnAct(bn.mergedScale, bn.mergedBias, act));
  }
  return bn.apply(conv.apply(x), mask, useMask);
}

// conv -> (residual + conv), fused into the Winograd untransform when the conv
// is Winograd and the batch is unmasked. Otherwise the exact decomposition.
static mx::array fusedConvResidual(const ConvLayer& conv, const mx::array& x,
                                   const mx::array& resid, bool useMask) {
  if(conv.useWinograd && !useMask) {
    return MLXWinograd::winogradConv2d(
      x, conv.winogradWeights, conv.outChannels, conv.winoInCfg, conv.winoOutCfg, conv.useFP16,
      MLXWinograd::Epilogue::residual(resid));
  }
  return resid + conv.apply(x);
}

struct ResidualBlock {
  const string name;
  const BatchNormLayer preBN;
  const ConvLayer regularConv;
  const BatchNormLayer midBN;
  const ConvLayer finalConv;

  ResidualBlock() = delete;
  ResidualBlock(const ResidualBlock&) = delete;
  ResidualBlock& operator=(const ResidualBlock&) = delete;

  ResidualBlock(const ResidualBlockDesc& desc,
                const MLXWinograd::InputTransform& inCfg,
                const MLXWinograd::OutputUntransform& outCfg,
                bool useFP16 = false)
    : name(desc.name),
      preBN(desc.preBN, desc.preActivation.activation, useFP16),
      regularConv(desc.regularConv, inCfg, outCfg, useFP16),
      midBN(desc.midBN, desc.midActivation.activation, useFP16),
      finalConv(desc.finalConv, inCfg, outCfg, useFP16)
  {}

  mx::array apply(const mx::array& input, const mx::array& mask, bool useMask) const {
    mx::array out = preBN.apply(input, mask, useMask);
    out = fusedConvBNAct(regularConv, midBN, out, mask, useMask);  // fuse regularConv -> midBN
    return fusedConvResidual(finalConv, out, input, useMask);      // fuse finalConv -> input + out
  }
};

// Global Pooling Residual Block
struct GlobalPoolingResidualBlock {
  const string name;
  const BatchNormLayer preBN;
  const ConvLayer regularConv;
  const ConvLayer gpoolConv;
  const BatchNormLayer gpoolBN;
  const MatMulLayer gpoolToBiasMul;
  const BatchNormLayer midBN;
  const ConvLayer finalConv;

  GlobalPoolingResidualBlock() = delete;
  GlobalPoolingResidualBlock(const GlobalPoolingResidualBlock&) = delete;
  GlobalPoolingResidualBlock& operator=(const GlobalPoolingResidualBlock&) = delete;

  GlobalPoolingResidualBlock(const GlobalPoolingResidualBlockDesc& desc,
                             const MLXWinograd::InputTransform& inCfg,
                             const MLXWinograd::OutputUntransform& outCfg,
                             bool useFP16 = false)
    : name(desc.name),
      preBN(desc.preBN, desc.preActivation.activation, useFP16),
      regularConv(desc.regularConv, inCfg, outCfg, useFP16),
      gpoolConv(desc.gpoolConv, inCfg, outCfg, useFP16),
      gpoolBN(desc.gpoolBN, desc.gpoolActivation.activation, useFP16),
      gpoolToBiasMul(desc.gpoolToBiasMul, useFP16),
      midBN(desc.midBN, desc.midActivation.activation, useFP16),
      finalConv(desc.finalConv, inCfg, outCfg, useFP16)
  {}

  mx::array apply(const mx::array& input, const mx::array& mask, const mx::array& maskSum, bool useMask) const {
    mx::array preOut = preBN.apply(input, mask, useMask);

    // Regular path
    mx::array regularOut = regularConv.apply(preOut);

    // Global pooling path
    mx::array gpoolOut = gpoolConv.apply(preOut);
    gpoolOut = gpoolBN.apply(gpoolOut, mask, useMask);
    mx::array pooled = applyGlobalPooling(gpoolOut, mask, maskSum, useMask);

    // Squeeze spatial dims for matmul: [N, 1, 1, C*3] -> [N, C*3]
    std::vector<int> squeezeAxes = {1, 2};
    mx::array pooledFlat = mx::squeeze(pooled, squeezeAxes);
    mx::array bias = gpoolToBiasMul.apply(pooledFlat);

    // Add bias to regular path (broadcast): [N, outC] -> [N, 1, 1, outC]
    mx::Shape biasShape = {static_cast<int>(bias.shape()[0]), 1, 1, static_cast<int>(bias.shape()[1])};
    bias = mx::reshape(bias, biasShape);
    mx::array combined = regularOut + bias;

    combined = midBN.apply(combined, mask, useMask);
    mx::array finalOut = finalConv.apply(combined);

    return input + finalOut;
  }
};

// Nested Bottleneck Residual Block (simplified - forward declaration for recursive types)
struct NestedBottleneckResidualBlock;

// Block variant type for trunk
struct BlockVariant {
  enum Type { REGULAR, GLOBAL_POOLING, NESTED_BOTTLENECK, TRANSFORMER_ATTENTION, TRANSFORMER_FFN };
  Type type;
  unique_ptr<ResidualBlock> regular;
  unique_ptr<GlobalPoolingResidualBlock> globalPooling;
  unique_ptr<NestedBottleneckResidualBlock> nestedBottleneck;
  unique_ptr<TransformerAttentionBlock> attention;
  unique_ptr<TransformerFFNBlock> ffn;

  BlockVariant(const ResidualBlockDesc& desc,
               const MLXWinograd::InputTransform& inCfg,
               const MLXWinograd::OutputUntransform& outCfg,
               bool useFP16 = false)
    : type(REGULAR), regular(make_unique<ResidualBlock>(desc, inCfg, outCfg, useFP16)) {}

  BlockVariant(const GlobalPoolingResidualBlockDesc& desc,
               const MLXWinograd::InputTransform& inCfg,
               const MLXWinograd::OutputUntransform& outCfg,
               bool useFP16 = false)
    : type(GLOBAL_POOLING), globalPooling(make_unique<GlobalPoolingResidualBlock>(desc, inCfg, outCfg, useFP16)) {}

  // Transformer blocks have no convolutions, so they take board dims (nnX, nnY)
  // instead of Winograd transform configs.
  BlockVariant(const TransformerAttentionDesc& desc, int nnX, int nnY, bool useFP16 = false)
    : type(TRANSFORMER_ATTENTION), attention(make_unique<TransformerAttentionBlock>(desc, nnX, nnY, useFP16)) {}

  BlockVariant(const TransformerFFNDesc& desc, int nnX, int nnY, bool useFP16 = false)
    : type(TRANSFORMER_FFN), ffn(make_unique<TransformerFFNBlock>(desc, nnX, nnY, useFP16)) {}

  // Forward declaration - defined after NestedBottleneckResidualBlock.
  // Takes nnX/nnY so any transformer blocks nested inside the bottleneck can
  // precompute their RoPE tables.
  BlockVariant(const NestedBottleneckResidualBlockDesc& desc,
               const MLXWinograd::InputTransform& inCfg,
               const MLXWinograd::OutputUntransform& outCfg,
               int nnX,
               int nnY,
               bool useFP16);

  mx::array apply(const mx::array& input, const mx::array& mask, const mx::array& maskSum, bool useMask) const;
};

struct NestedBottleneckResidualBlock {
  const string name;
  const BatchNormLayer preBN;
  const ConvLayer preConv;
  vector<BlockVariant> blocks;
  const BatchNormLayer postBN;
  const ConvLayer postConv;

  NestedBottleneckResidualBlock() = delete;
  NestedBottleneckResidualBlock(const NestedBottleneckResidualBlock&) = delete;
  NestedBottleneckResidualBlock& operator=(const NestedBottleneckResidualBlock&) = delete;

  NestedBottleneckResidualBlock(const NestedBottleneckResidualBlockDesc& desc,
                                const MLXWinograd::InputTransform& inCfg,
                                const MLXWinograd::OutputUntransform& outCfg,
                                int nnX,
                                int nnY,
                                bool useFP16 = false)
    : name(desc.name),
      preBN(desc.preBN, desc.preActivation.activation, useFP16),
      preConv(desc.preConv, inCfg, outCfg, useFP16),
      postBN(desc.postBN, desc.postActivation.activation, useFP16),
      postConv(desc.postConv, inCfg, outCfg, useFP16)
  {
    // Mirror parseResidualBlockStack (desc.cpp), which accepts the same five
    // block kinds inside a nested bottleneck as in the trunk - including a
    // nested bottleneck within a nested bottleneck. Keep this in sync with the
    // trunk's block loop below; an unhandled kind is a loud bug, not a silent
    // no-op block.
    for(size_t i = 0; i < desc.blocks.size(); i++) {
      int blockKind = desc.blocks[i].first;
      if(blockKind == ORDINARY_BLOCK_KIND) {
        blocks.emplace_back(*static_cast<ResidualBlockDesc*>(desc.blocks[i].second.get()), inCfg, outCfg, useFP16);
      }
      else if(blockKind == GLOBAL_POOLING_BLOCK_KIND) {
        blocks.emplace_back(*static_cast<GlobalPoolingResidualBlockDesc*>(desc.blocks[i].second.get()), inCfg, outCfg, useFP16);
      }
      else if(blockKind == NESTED_BOTTLENECK_BLOCK_KIND) {
        blocks.emplace_back(*static_cast<NestedBottleneckResidualBlockDesc*>(desc.blocks[i].second.get()), inCfg, outCfg, nnX, nnY, useFP16);
      }
      else if(blockKind == TRANSFORMER_ATTENTION_BLOCK_KIND) {
        blocks.emplace_back(*static_cast<TransformerAttentionDesc*>(desc.blocks[i].second.get()), nnX, nnY, useFP16);
      }
      else if(blockKind == TRANSFORMER_FFN_BLOCK_KIND) {
        blocks.emplace_back(*static_cast<TransformerFFNDesc*>(desc.blocks[i].second.get()), nnX, nnY, useFP16);
      }
      else {
        ASSERT_UNREACHABLE;
      }
    }
  }

  mx::array apply(const mx::array& input, const mx::array& mask, const mx::array& maskSum, bool useMask) const {
    mx::array out = preBN.apply(input, mask, useMask);
    out = preConv.apply(out);
    for(const auto& block : blocks) {
      out = block.apply(out, mask, maskSum, useMask);
    }
    out = postBN.apply(out, mask, useMask);
    return fusedConvResidual(postConv, out, input, useMask);       // fuse postConv -> input + out
  }
};

// Define BlockVariant constructor for NestedBottleneckResidualBlock now that it's complete
BlockVariant::BlockVariant(const NestedBottleneckResidualBlockDesc& desc,
                           const MLXWinograd::InputTransform& inCfg,
                           const MLXWinograd::OutputUntransform& outCfg,
                           int nnX,
                           int nnY,
                           bool useFP16)
  : type(NESTED_BOTTLENECK), nestedBottleneck(make_unique<NestedBottleneckResidualBlock>(desc, inCfg, outCfg, nnX, nnY, useFP16)) {}

mx::array BlockVariant::apply(const mx::array& input, const mx::array& mask, const mx::array& maskSum, bool useMask) const {
  switch(type) {
    case REGULAR:
      return regular->apply(input, mask, useMask);
    case GLOBAL_POOLING:
      return globalPooling->apply(input, mask, maskSum, useMask);
    case NESTED_BOTTLENECK:
      return nestedBottleneck->apply(input, mask, maskSum, useMask);
    case TRANSFORMER_ATTENTION:
      return attention->apply(input, mask, useMask);
    case TRANSFORMER_FFN:
      return ffn->apply(input, mask, useMask);
    default:
      // All BlockVariant::Type values are handled above. Reaching the default
      // means the tagged union holds an unrecognized type - fail loudly rather
      // than silently returning the input (an identity no-op block). The
      // default also satisfies -Wswitch-default (see cpp/CMakeLists.txt).
      ASSERT_UNREACHABLE;
  }
}

// SGF Metadata Encoder
struct SGFMetadataEncoder {
  const int metaEncoderVersion;
  const int numInputMetaChannels;
  const MatMulLayer mul1;
  const MatBiasLayer bias1;
  const int act1;
  const MatMulLayer mul2;
  const MatBiasLayer bias2;
  const int act2;
  const MatMulLayer mul3;

  SGFMetadataEncoder() = delete;
  SGFMetadataEncoder(const SGFMetadataEncoder&) = delete;
  SGFMetadataEncoder& operator=(const SGFMetadataEncoder&) = delete;

  SGFMetadataEncoder(const SGFMetadataEncoderDesc& desc, bool useFP16 = false)
    : metaEncoderVersion(desc.metaEncoderVersion),
      numInputMetaChannels(desc.numInputMetaChannels),
      mul1(desc.mul1, useFP16),
      bias1(desc.bias1, useFP16),
      act1(desc.act1.activation),
      mul2(desc.mul2, useFP16),
      bias2(desc.bias2, useFP16),
      act2(desc.act2.activation),
      mul3(desc.mul3, useFP16)
  {}

  mx::array apply(const mx::array& metaInput) const {
    // Fuse matmul + bias with addmm for better performance
    mx::array out = matmulBias(metaInput, mul1.weights, bias1.bias);
    out = applyActivation(out, act1);
    out = matmulBias(out, mul2.weights, bias2.bias);
    out = applyActivation(out, act2);
    out = mul3.apply(out);  // Last layer has no bias
    return out;
  }
};

// Trunk
struct Trunk {
  const string name;
  const int trunkNumChannels;
  const int trunkNormKind;
  const ConvLayer initialConv;
  const MatMulLayer initialMatMul;
  unique_ptr<SGFMetadataEncoder> sgfMetadataEncoder;
  vector<BlockVariant> blocks;
  // Exactly one of these is populated depending on trunkNormKind. For
  // TRUNK_NORM_KIND_RMSNORM the trunkTipBN desc on disk is empty (size-0
  // weights), so constructing a BatchNormLayer from it would create empty
  // arrays that broadcast against the spatial trunk and crash. Mirrors Eigen's
  // Trunk (eigenbackend.cpp 1875-1880) which selects BN vs RMSNorm at load.
  unique_ptr<BatchNormLayer> trunkTipBN;
  unique_ptr<TransformerTrunkRMSNormLayer> trunkTipRMSNorm;

  Trunk() = delete;
  Trunk(const Trunk&) = delete;
  Trunk& operator=(const Trunk&) = delete;

  Trunk(const TrunkDesc& desc,
        const MLXWinograd::InputTransform& inCfg,
        const MLXWinograd::OutputUntransform& outCfg,
        int nnX,
        int nnY,
        bool useFP16 = false)
    : name(desc.name),
      trunkNumChannels(desc.trunkNumChannels),
      trunkNormKind(desc.trunkNormKind),
      initialConv(desc.initialConv, inCfg, outCfg, useFP16),
      initialMatMul(desc.initialMatMul, useFP16)
  {
    // Trunk tip normalization: BatchNorm for standard nets, RMSNorm for
    // transformer nets. Only the desc matching trunkNormKind was parsed.
    if(desc.trunkNormKind == TRUNK_NORM_KIND_STANDARD) {
      trunkTipBN = make_unique<BatchNormLayer>(desc.trunkTipBN, desc.trunkTipActivation.activation, useFP16);
    }
    else {
      trunkTipRMSNorm = make_unique<TransformerTrunkRMSNormLayer>(desc.trunkTipRMSNorm, desc.trunkTipActivation.activation, useFP16);
    }

    if(desc.sgfMetadataEncoder.metaEncoderVersion > 0 && desc.sgfMetadataEncoder.numInputMetaChannels > 0) {
      sgfMetadataEncoder = make_unique<SGFMetadataEncoder>(desc.sgfMetadataEncoder, useFP16);
    }

    for(size_t i = 0; i < desc.blocks.size(); i++) {
      int blockKind = desc.blocks[i].first;
      if(blockKind == ORDINARY_BLOCK_KIND) {
        blocks.emplace_back(*static_cast<ResidualBlockDesc*>(desc.blocks[i].second.get()), inCfg, outCfg, useFP16);
      }
      else if(blockKind == GLOBAL_POOLING_BLOCK_KIND) {
        blocks.emplace_back(*static_cast<GlobalPoolingResidualBlockDesc*>(desc.blocks[i].second.get()), inCfg, outCfg, useFP16);
      }
      else if(blockKind == NESTED_BOTTLENECK_BLOCK_KIND) {
        blocks.emplace_back(*static_cast<NestedBottleneckResidualBlockDesc*>(desc.blocks[i].second.get()), inCfg, outCfg, nnX, nnY, useFP16);
      }
      else if(blockKind == TRANSFORMER_ATTENTION_BLOCK_KIND) {
        blocks.emplace_back(*static_cast<TransformerAttentionDesc*>(desc.blocks[i].second.get()), nnX, nnY, useFP16);
      }
      else if(blockKind == TRANSFORMER_FFN_BLOCK_KIND) {
        blocks.emplace_back(*static_cast<TransformerFFNDesc*>(desc.blocks[i].second.get()), nnX, nnY, useFP16);
      }
      else {
        // parseResidualBlockStack (desc.cpp) rejects any other kind at load,
        // so reaching here means a new block kind was added without backend
        // support - fail loudly instead of silently dropping the block.
        ASSERT_UNREACHABLE;
      }
    }
  }

  mx::array apply(
    const mx::array& input,
    const mx::array& inputGlobal,
    const mx::array* inputMeta,
    const mx::array& mask,
    const mx::array& maskSum,
    bool useMask
  ) const {
    // Initial conv
    mx::array trunk = initialConv.apply(input);

    // Add global input bias
    mx::array globalBias = initialMatMul.apply(inputGlobal);
    // Reshape from [N, C] to [N, 1, 1, C] for broadcasting
    mx::Shape globalBiasShape = {static_cast<int>(globalBias.shape()[0]), 1, 1, static_cast<int>(globalBias.shape()[1])};
    globalBias = mx::reshape(globalBias, globalBiasShape);
    trunk = trunk + globalBias;

    // Add SGF metadata if present
    if(sgfMetadataEncoder && inputMeta != nullptr) {
      mx::array metaBias = sgfMetadataEncoder->apply(*inputMeta);
      mx::Shape metaBiasShape = {static_cast<int>(metaBias.shape()[0]), 1, 1, static_cast<int>(metaBias.shape()[1])};
      metaBias = mx::reshape(metaBias, metaBiasShape);
      trunk = trunk + metaBias;
    }

    // Apply mask - skip when useMask=false (all positions valid)
    if(useMask)
      trunk = trunk * mask;

    // Apply residual blocks
    for(const auto& block : blocks) {
      trunk = block.apply(trunk, mask, maskSum, useMask);
    }

    // Final trunk-tip normalization + activation: BatchNorm or RMSNorm.
    if(trunkNormKind == TRUNK_NORM_KIND_STANDARD) {
      trunk = trunkTipBN->apply(trunk, mask, useMask);
    }
    else {
      trunk = trunkTipRMSNorm->apply(trunk, mask, maskSum, useMask);
    }

    return trunk;
  }
};

// Policy Head
struct PolicyHead {
  const string name;
  const int modelVersion;
  const ConvLayer p1Conv;
  const ConvLayer g1Conv;
  const BatchNormLayer g1BN;
  const MatMulLayer gpoolToBiasMul;
  const BatchNormLayer p1BN;
  const ConvLayer p2Conv;
  const MatMulLayer gpoolToPassMul;
  // v15+ two-layer pass head: gpoolToPassMul (input -> hidden) ->
  // gpoolToPassBias -> passActivation -> gpoolToPassMul2 (hidden -> output).
  // Pre-v15 models use a single matmul (gpoolToPassMul: input -> output) and
  // these three fields stay empty / zero. Mirrors the v15+ branch of
  // PolicyHeadDesc::PolicyHeadDesc in desc.cpp and Metal's
  // policyHeadDescToSwift in metalbackend.cpp.
  const std::optional<MatBiasLayer> gpoolToPassBias;
  const int passActivationType;
  const std::optional<MatMulLayer> gpoolToPassMul2;

  PolicyHead() = delete;
  PolicyHead(const PolicyHead&) = delete;
  PolicyHead& operator=(const PolicyHead&) = delete;

  PolicyHead(const PolicyHeadDesc& desc,
             const MLXWinograd::InputTransform& inCfg,
             const MLXWinograd::OutputUntransform& outCfg,
             bool useFP16 = false)
    : name(desc.name),
      modelVersion(desc.modelVersion),
      p1Conv(desc.p1Conv, inCfg, outCfg, useFP16),
      g1Conv(desc.g1Conv, inCfg, outCfg, useFP16),
      g1BN(desc.g1BN, desc.g1Activation.activation, useFP16),
      gpoolToBiasMul(desc.gpoolToBiasMul, useFP16),
      p1BN(desc.p1BN, desc.p1Activation.activation, useFP16),
      p2Conv(desc.p2Conv, inCfg, outCfg, useFP16),
      gpoolToPassMul(desc.gpoolToPassMul, useFP16),
      gpoolToPassBias(desc.modelVersion >= 15
        ? std::optional<MatBiasLayer>(std::in_place, desc.gpoolToPassBias, useFP16)
        : std::nullopt),
      passActivationType(desc.modelVersion >= 15 ? desc.passActivation.activation : 0),
      gpoolToPassMul2(desc.modelVersion >= 15
        ? std::optional<MatMulLayer>(std::in_place, desc.gpoolToPassMul2, useFP16)
        : std::nullopt)
  {}

  std::pair<mx::array, mx::array> apply(
    const mx::array& trunk,
    const mx::array& mask,
    const mx::array& maskSum,
    bool useMask
  ) const {
    // Policy conv
    mx::array p1Out = p1Conv.apply(trunk);

    // Global pooling path
    mx::array g1Out = g1Conv.apply(trunk);
    g1Out = g1BN.apply(g1Out, mask, useMask);
    mx::array pooled = applyGlobalPooling(g1Out, mask, maskSum, useMask);
    std::vector<int> squeezeAxes = {1, 2};
    mx::array pooledFlat = mx::squeeze(pooled, squeezeAxes);

    // Add bias from global pooling
    mx::array bias = gpoolToBiasMul.apply(pooledFlat);
    mx::Shape biasShape = {static_cast<int>(bias.shape()[0]), 1, 1, static_cast<int>(bias.shape()[1])};
    bias = mx::reshape(bias, biasShape);
    p1Out = p1Out + bias;

    p1Out = p1BN.apply(p1Out, mask, useMask);

    // Final policy conv
    mx::array policy = p2Conv.apply(p1Out);

    // Pass policy: pre-v15 is a single matmul (pooled -> output). v15+ is a
    // two-layer MLP (pooled -> hidden, + bias, activation, hidden -> output).
    // Mirrors the v15+ branch of PolicyHeadDesc::PolicyHeadDesc in desc.cpp
    // and Metal's policyHeadDescToSwift in metalbackend.cpp.
    mx::array policyPass = gpoolToPassMul.apply(pooledFlat);
    if(modelVersion >= 15) {
      policyPass = gpoolToPassBias->apply(policyPass);
      policyPass = applyActivation(policyPass, passActivationType);
      policyPass = gpoolToPassMul2->apply(policyPass);
    }

    return {policyPass, policy};
  }
};

// Value Head
struct ValueHead {
  const string name;
  const int modelVersion;
  const ConvLayer v1Conv;
  const BatchNormLayer v1BN;
  const MatMulLayer v2Mul;
  const MatBiasLayer v2Bias;
  const int v2Activation;
  const MatMulLayer v3Mul;
  const MatBiasLayer v3Bias;
  const MatMulLayer sv3Mul;
  const MatBiasLayer sv3Bias;
  const ConvLayer vOwnershipConv;

  ValueHead() = delete;
  ValueHead(const ValueHead&) = delete;
  ValueHead& operator=(const ValueHead&) = delete;

  ValueHead(const ValueHeadDesc& desc,
            const MLXWinograd::InputTransform& inCfg,
            const MLXWinograd::OutputUntransform& outCfg,
            bool useFP16 = false)
    : name(desc.name),
      modelVersion(desc.modelVersion),
      v1Conv(desc.v1Conv, inCfg, outCfg, useFP16),
      v1BN(desc.v1BN, desc.v1Activation.activation, useFP16),
      v2Mul(desc.v2Mul, useFP16),
      v2Bias(desc.v2Bias, useFP16),
      v2Activation(desc.v2Activation.activation),
      v3Mul(desc.v3Mul, useFP16),
      v3Bias(desc.v3Bias, useFP16),
      sv3Mul(desc.sv3Mul, useFP16),
      sv3Bias(desc.sv3Bias, useFP16),
      vOwnershipConv(desc.vOwnershipConv, inCfg, outCfg, useFP16)
  {}

  std::tuple<mx::array, mx::array, mx::array> apply(
    const mx::array& trunk,
    const mx::array& mask,
    const mx::array& maskSum,
    bool useMask
  ) const {
    mx::array v1Out = v1Conv.apply(trunk);
    v1Out = v1BN.apply(v1Out, mask, useMask);

    // Value head pooling (only uses maskSum, not mask)
    mx::array v1Mean = applyValueHeadPooling(v1Out, maskSum);
    std::vector<int> squeezeAxes = {1, 2};
    mx::array v1MeanFlat = mx::squeeze(v1Mean, squeezeAxes);

    // Fuse matmul + bias with addmm for better performance
    mx::array v2Out = matmulBias(v1MeanFlat, v2Mul.weights, v2Bias.bias);
    v2Out = applyActivation(v2Out, v2Activation);

    mx::array value = matmulBias(v2Out, v3Mul.weights, v3Bias.bias);
    mx::array scoreValue = matmulBias(v2Out, sv3Mul.weights, sv3Bias.bias);

    mx::array ownership = vOwnershipConv.apply(v1Out);

    return {value, scoreValue, ownership};
  }
};

// Model
struct Model {
  const string name;
  const int modelVersion;
  const int numInputChannels;
  const int numInputGlobalChannels;
  const int numInputMetaChannels;
  const int numPolicyChannels;
  // Pass-policy output width. For v15+ models the pass head is two-layer:
  // gpoolToPassMul (input -> hidden) -> bias -> activation -> gpoolToPassMul2
  // (hidden -> output). The actual final output width — and the per-row stride
  // extractOutputs in metalbackend.swift uses for its writes
  // (batchIndex * numPolicyChannels) — is gpoolToPassMul2.outChannels, which
  // PolicyHeadDesc::PolicyHeadDesc in desc.cpp validates equals
  // numPolicyChannels. Pre-v15 models have a single matmul (gpoolToPassMul:
  // input -> output) and the output width is gpoolToPassMul.outChannels =
  // numPolicyChannels (also validated in PolicyHeadDesc::PolicyHeadDesc).
  // Using gpoolToPassMul.outChannels for v15+ was the prior bug: it is the
  // hidden width, not the output width, and rows >= 1 in batched ANE reads
  // landed on uninitialized memory.
  const int numPolicyPassChannels;
  const int numValueChannels;
  const int numScoreValueChannels;
  const int numOwnershipChannels;
  const bool useFP16;

  const Trunk trunk;
  const PolicyHead policyHead;
  const ValueHead valueHead;

  Model() = delete;
  Model(const Model&) = delete;
  Model& operator=(const Model&) = delete;

  // nnX/nnY (board dims) are needed by transformer attention blocks to
  // precompute RoPE cos/sin tables, which are position-dependent. The MLX
  // compiled graph is keyed by nnXLen/nnYLen so a Model built for one board
  // size is never reused for another.
  Model(const ModelDesc& desc, const MLXWinogradTuneParams& tuneParams, int nnX, int nnY, bool useFP16_ = false)
    : name(desc.name),
      modelVersion(desc.modelVersion),
      numInputChannels(desc.numInputChannels),
      numInputGlobalChannels(desc.numInputGlobalChannels),
      numInputMetaChannels(desc.numInputMetaChannels),
      numPolicyChannels(desc.numPolicyChannels),
      numPolicyPassChannels(desc.modelVersion >= 15
                              ? desc.policyHead.gpoolToPassMul2.outChannels
                              : desc.policyHead.gpoolToPassMul.outChannels),
      numValueChannels(desc.numValueChannels),
      numScoreValueChannels(desc.numScoreValueChannels),
      numOwnershipChannels(desc.numOwnershipChannels),
      useFP16(useFP16_),
      trunk(desc.trunk, tuneParams.inputTransform, tuneParams.outputUntransform, nnX, nnY, useFP16_),
      policyHead(desc.policyHead, tuneParams.inputTransform, tuneParams.outputUntransform, useFP16_),
      valueHead(desc.valueHead, tuneParams.inputTransform, tuneParams.outputUntransform, useFP16_)
  {}

  // Apply model inference with mx::array inputs directly (for compiled execution)
  // inputs: [input, inputGlobal, mask, maskSum] or [input, inputGlobal, mask, maskSum, inputMeta]
  // outputs: [policy, policyPass, value, scoreValue, ownership]
  std::vector<mx::array> applyArrays(
    const std::vector<mx::array>& inputs,
    bool useMask
  ) const {
    // Convert inputs to compute dtype if FP16 is enabled
    mx::array input = toComputeDtype(inputs[0], useFP16);
    mx::array inputGlobalArr = toComputeDtype(inputs[1], useFP16);
    mx::array mask = toComputeDtype(inputs[2], useFP16);
    // maskSum stays FP32 - small scalar, negligible impact
    const mx::array& maskSum = inputs[3];
    unique_ptr<mx::array> inputMeta;
    if(inputs.size() > 4) {
      inputMeta = make_unique<mx::array>(toComputeDtype(inputs[4], useFP16));
    }
    const mx::array* inputMetaPtr = inputMeta.get();

    // Apply trunk
    mx::array trunkOut = trunk.apply(input, inputGlobalArr, inputMetaPtr, mask, maskSum, useMask);

    // Apply policy head
    auto [policyPass, policy] = policyHead.apply(trunkOut, mask, maskSum, useMask);

    // Apply value head
    auto [value, scoreValue, ownership] = valueHead.apply(trunkOut, mask, maskSum, useMask);

    // Convert outputs back to FP32 for interface compatibility
    if(useFP16) {
      policy = mx::astype(policy, mx::float32);
      policyPass = mx::astype(policyPass, mx::float32);
      value = mx::astype(value, mx::float32);
      scoreValue = mx::astype(scoreValue, mx::float32);
      ownership = mx::astype(ownership, mx::float32);
    }

    return {policy, policyPass, value, scoreValue, ownership};
  }

  // Create a compiled inference function for the given configuration
  // hasMeta is used as part of the cache key but not needed in the function itself
  CompiledInferenceFunc createCompiledFunc(bool useMask, bool /*hasMeta*/) const {
    // Create lambda that captures this model
    auto inferenceFunc = [this, useMask](const std::vector<mx::array>& inputs) -> std::vector<mx::array> {
      return this->applyArrays(inputs, useMask);
    };

    // Wrap in std::function and compile
    std::function<std::vector<mx::array>(const std::vector<mx::array>&)> func = inferenceFunc;
    return mx::compile(func, /*shapeless=*/false);
  }

  // Apply model using a pre-compiled inference function
  void applyCompiled(
    const CompiledInferenceFunc& compiledFunc,
    const float* inputSpatial,
    const float* inputGlobal,
    const float* inputMeta,
    int batchSize,
    int nnXLen,
    int nnYLen,
    bool requireExactNNLen,
    float* policyOut,
    float* policyPassOut,
    float* valueOut,
    float* scoreValueOut,
    float* ownershipOut
  ) const {
    std::vector<mx::array> outputs;
    {
      // Serialize graph construction + command encoding only. async_eval
      // encodes all GPU work synchronously on this thread before returning;
      // the completion wait and result readback below run outside the lock
      // so another server thread can encode its batch while this one waits.
      // See mlxGpuEvalMutex for the full rationale.
      std::lock_guard<std::mutex> gpuLock(mlxGpuEvalMutex);

      // Create input tensors - NHWC format
      mx::Shape inputShape = {batchSize, nnYLen, nnXLen, numInputChannels};
      mx::array input = mx::array(inputSpatial, inputShape, mx::float32);
      mx::Shape globalShape = {batchSize, numInputGlobalChannels};
      mx::array inputGlobalArr = mx::array(inputGlobal, globalShape, mx::float32);

      // Extract mask from first channel of input
      mx::Shape sliceStart = {0, 0, 0, 0};
      mx::Shape sliceEnd = {batchSize, nnYLen, nnXLen, 1};
      mx::array mask = mx::slice(input, sliceStart, sliceEnd);

      // Compute mask sum
      std::vector<int> sumAxes = {1, 2};
      mx::array maskSum = requireExactNNLen
        ? mx::full({batchSize, 1, 1, 1}, static_cast<float>(nnXLen * nnYLen))
        : mx::sum(mask, sumAxes, /*keepdims=*/true);

      // Build input vector for compiled function
      std::vector<mx::array> inputs = {input, inputGlobalArr, mask, maskSum};

      // Add metadata if present
      if(numInputMetaChannels > 0 && inputMeta != nullptr) {
        mx::Shape metaShape = {batchSize, numInputMetaChannels};
        inputs.push_back(mx::array(inputMeta, metaShape, mx::float32));
      }

      // Call compiled function and encode the GPU work
      outputs = compiledFunc(inputs);
      mx::async_eval(outputs);
    }

    // Wait for GPU completion. array::wait() blocks on the array's shared
    // completion event without touching the command stream, so it is safe
    // (and intended) to run while another thread holds mlxGpuEvalMutex to
    // encode the next batch.
    for(mx::array& out : outputs)
      out.wait();

    // Extract results - outputs are [policy, policyPass, value, scoreValue, ownership]
    mx::array& policy = outputs[0];
    mx::array& policyPass = outputs[1];
    mx::array& value = outputs[2];
    mx::array& scoreValue = outputs[3];
    mx::array& ownership = outputs[4];

    // Copy results to output buffers
    memcpy(policyOut, policy.data<float>(), batchSize * numPolicyChannels * nnXLen * nnYLen * sizeof(float));
    memcpy(policyPassOut, policyPass.data<float>(), batchSize * numPolicyPassChannels * sizeof(float));
    memcpy(valueOut, value.data<float>(), batchSize * numValueChannels * sizeof(float));
    memcpy(scoreValueOut, scoreValue.data<float>(), batchSize * numScoreValueChannels * sizeof(float));
    memcpy(ownershipOut, ownership.data<float>(), batchSize * numOwnershipChannels * nnXLen * nnYLen * sizeof(float));
  }
};

// Forward declaration needed by the helpers below (struct is defined in the
// "ComputeContext and ComputeHandle" section that follows).
struct ComputeContext;

//------------------------------------------------------------------------------
// CoreML/ANE compute handle helpers - mirrors convertAndCreateCoreMLOnlyHandle
// in metalbackend.cpp
//------------------------------------------------------------------------------

// Note: KataGoSwift::MetalComputeContext is the Swift-side context type. Its
// name is misleading in this file (MLX, not Metal) but we reuse it as-is per
// the design decision to leave KataGoSwift unchanged. It carries only
// (nnXLen, nnYLen, useFP16).

// Helper: convert model and create CoreML-only compute handle (for mux ANE thread)
static swift::Optional<KataGoSwift::CoreMLComputeHandle> convertAndCreateCoreMLOnlyHandleMLX(
  ComputeContext* context,
  const LoadedModel* loadedModel,
  bool requireExactNNLen,
  int maxBatchSize,
  int serverThreadIdx
);

// Helper: create CoreML-only handle when gpuIdx == MLX_MUX_ANE.
// Returns Optional::none() for the GPU path. Emits the same FP16-only-ANE
// warning Metal emits when useFP16=false is combined with the ANE mux.
static swift::Optional<KataGoSwift::CoreMLComputeHandle> createCoreMLOnlyHandleIfNeededMLX(
  ComputeContext* context,
  const LoadedModel* loadedModel,
  bool requireExactNNLen,
  int maxBatchSize,
  int gpuIdx,
  int serverThreadIdx
);

// ComputeContext and ComputeHandle ------------------------------------------------------------------------------------

// Session-scoped, cross-context Winograd-tune memo. The main net and the human
// SL net are separate NNEvaluators with separate ComputeContexts (gtp.cpp builds
// both), yet on iOS MLX/GPU both are b18c384 with identical 3x3-conv shapes, so
// their optimal transform geometry is the same — they should tune ONCE per
// engine session, not twice. A per-context memo can't span the two contexts;
// this process-global one does. It is scoped to a session by being cleared when
// the last ComputeContext is freed (a session == the window in which any context
// is alive), so a forced re-tune in a later session runs afresh instead of
// reusing a stale entry. Sequential context creation in gtp.cpp means the
// check/tune/store below needs no lock held across the (long) tune itself.
//
// All state (the map, its mutex, and the live-context refcount that drives the
// session-scoped clear) lives in this one unit so the locking obligation and the
// "clear on last release" invariant are defined in one place rather than spread
// across createComputeContext / freeComputeContext / the ComputeHandle ctor.
struct WinogradTuneMemo {
  // Returns the memoized params for this shape key, or nullopt on a miss.
  std::optional<MLXWinogradTuneParams> tryGet(const std::string& key) {
    std::lock_guard<std::mutex> lk(mutex);
    auto it = memo.find(key);
    if(it == memo.end())
      return std::nullopt;
    return it->second;
  }
  void put(const std::string& key, const MLXWinogradTuneParams& params) {
    std::lock_guard<std::mutex> lk(mutex);
    memo[key] = params;
  }
  // retain/release track how many ComputeContexts are alive. The memo is a
  // session-scoped cache: when the last context goes away the session is over,
  // so drop everything (a later forced re-tune must not reuse this session's
  // results).
  void retain() {
    std::lock_guard<std::mutex> lk(mutex);
    liveContexts++;
  }
  void release() {
    std::lock_guard<std::mutex> lk(mutex);
    if(--liveContexts <= 0) {
      liveContexts = 0;
      memo.clear();
    }
  }
private:
  std::mutex mutex;
  std::map<std::string, MLXWinogradTuneParams> memo;
  int liveContexts = 0;
};
static WinogradTuneMemo g_winoTuneMemo;

struct ComputeContext {
  const int nnXLen;
  const int nnYLen;
  const enabled_t useFP16Mode;
  std::string homeDataDirOverride;
  Logger* logger;

  std::mutex cachedModelsMutex;
  std::map<std::string, std::shared_ptr<const Model>> cachedModels;
  std::map<std::string, int> cachedModelsRefCount;

  // True only when EVERY configured device index is MLX_MUX_ANE (set in
  // createComputeContext). The ANE path re-reads the model from disk in
  // convertModelToTemp and otherwise reads only scalar dims, so when no thread
  // will ever build the MLX/GPU model we can free the in-memory FP32 weights via
  // ModelDesc::releaseWeights(). Must stay false if any thread uses MLX_MUX_GPU,
  // which would read the freed weights.
  bool aneOnly = false;

  // MLX/GPU Winograd autotuner controls, plumbed from the app via
  // -override-config (mlxTunerFull / mlxReTune), read in createComputeContext.
  // tunerFull=true selects the wide grid (thorough but much slower); reTune=true
  // forces a fresh tune that ignores and overwrites the cached file. Consumed by
  // the GPU ComputeHandle ctor's loadOrAutoTune call; ignored on the ANE path.
  bool tunerFull = false;
  bool tunerReTune = false;

  ComputeContext() = delete;
  ComputeContext(const ComputeContext&) = delete;
  ComputeContext& operator=(const ComputeContext&) = delete;

  ComputeContext(int nnX, int nnY, enabled_t fp16Mode,
                 const std::string& homeDataDirOverride_, Logger* logger_)
    : nnXLen(nnX),
      nnYLen(nnY),
      useFP16Mode(fp16Mode),
      homeDataDirOverride(homeDataDirOverride_),
      logger(logger_),
      cachedModelsMutex(),
      cachedModels(),
      cachedModelsRefCount()
  {}

  ~ComputeContext() {
    assert(cachedModels.size() == 0);
  }
};

// Resolve the Winograd transform geometry for a GPU-path model load: honor the
// cross-context session memo (g_winoTuneMemo) first, then fall back to
// loadOrAutoTune (disk cache or a fresh sweep), storing the result in the memo.
// Returns default-constructed (identity) transforms when Winograd or the
// winotuner is disabled. Reads only ComputeContext + LoadedModel state and
// touches no ComputeHandle, so it lives as a free function the ctor calls.
static MLXWinogradTuneParams resolveTuneParams(
    ComputeContext* context, const LoadedModel& loadedModel, bool useFP16) {
  MLXWinogradTuneParams tuneParams;
  if(!(mlxWinogradEnabled() && mlxWinotunerEnabled()))
    return tuneParams;

  // Shape diagnostic: print the model's 3x3 conv shape distribution before
  // calling the tuner so the log carries this signal on every load, including
  // cache-hit runs where loadOrAutoTune short-circuits.
  if(context->logger != NULL) {
    context->logger->write(
        MLXWinogradTuner::formatConv3x3Distribution(loadedModel.modelDesc));
  }
  MLXWinogradTuner::ModelInfoForTuning mi;
  mi.trunkNumChannels   = loadedModel.modelDesc.trunk.trunkNumChannels;
  mi.modelVersion       = loadedModel.modelDesc.modelVersion;
  auto [inHist, outHist] =
      MLXWinogradTuner::buildConv3x3Histograms(loadedModel.modelDesc);
  mi.conv3x3InputHistogram  = std::move(inHist);
  mi.conv3x3OutputHistogram = std::move(outHist);

  // Cross-context shape memo (see g_winoTuneMemo): if a same-shape GPU
  // handle already tuned this session, reuse its result and skip the sweep
  // entirely. This is what keeps the main + human b18c384 nets at a single
  // tune instead of two — halving model-load tuning time at zero quality
  // cost (identical shape ⇒ identical optimal geometry).
  const std::string shapeKey =
      std::to_string(mi.trunkNumChannels)
      + "_" + std::to_string(context->nnXLen)
      + "x" + std::to_string(context->nnYLen)
      + (useFP16 ? "_fp16" : "_fp32")
      + (context->tunerFull ? "_full" : "_fast");
  if(auto memoized = g_winoTuneMemo.tryGet(shapeKey)) {
    if(context->logger != NULL)
      context->logger->write("Reusing MLX Winograd tuning for shape " + shapeKey
                             + " (already tuned this session)");
    return *memoized;
  }

  tuneParams = MLXWinogradTuner::loadOrAutoTune(
      /*tunerFile=*/"",
      context->homeDataDirOverride,
      MLXWinogradTuner::detectGpuName(),
      context->nnXLen, context->nnYLen,
      // Tuner times the Winograd input/output transform kernels at this
      // batch size only (the matmul stage is untuned). Probed re-tuning
      // at 8/16/32/64: the winning configs do differ per batch size, but
      // end-to-end throughput stayed flat within ~1.5% run-to-run noise.
      // OpenCL's tuner pins a single batch size too. Not worth
      // parameterizing.
      /*batchSize=*/8,
      mi,
      context->logger,
      // full / reTune come from the app's MLX/GPU tuning UI via
      // -override-config (mlxTunerFull / mlxReTune), read into the
      // ComputeContext in createComputeContext. Defaults are false/false:
      // load a valid cache, or on a miss tune the coarse "fast" grid once.
      // full=true selects the wide grid (cached under a distinct "_full"
      // file); reTune=true forces a fresh tune that overwrites the current
      // mode's cache. `./katago tuner` (optionally -full) remains a separate
      // always-overwrite entry point.
      /*full=*/context->tunerFull,
      /*reTune=*/context->tunerReTune,
      /*useFP16=*/useFP16);
  g_winoTuneMemo.put(shapeKey, tuneParams);
  return tuneParams;
}

struct ComputeHandle {
  ComputeContext* context;
  bool inputsUseNHWC;
  bool requireExactNNLen;
  bool useFP16;
  int gpuIdx;
  std::string modelCacheKey;  // assigned in ctor body after loadOrAutoTune
  std::shared_ptr<const Model> model;
  const int modelVersion;

  // ModelDesc fields cached on both paths so getOutput does not have to
  // dereference `model` (which is nullptr on the ANE path). Populated in
  // the constructor body for both MLX_MUX_GPU and MLX_MUX_ANE.
  int numInputChannels;
  int numPolicyChannels;
  int numPolicyPassChannels;
  int numValueChannels;
  int numScoreValueChannels;
  int numOwnershipChannels;

  // Compiled function cache - keyed by (batchSize, nnXLen, nnYLen, useMask, hasMeta, useFP16).
  // Populated only on the MLX/GPU path; the ANE path uses coremlOnlyHandle instead.
  mutable std::mutex compiledFuncsMutex;
  mutable std::map<CompileCacheKey, CompiledInferenceFunc> compiledFuncs;

  // CoreML-only handle (Swift). Populated iff gpuIdx == MLX_MUX_ANE; otherwise none().
  // Exactly one of {model populated (MLX/GPU path) OR coremlOnlyHandle has value (ANE path)}.
  swift::Optional<KataGoSwift::CoreMLComputeHandle> coremlOnlyHandle;

  ComputeHandle() = delete;
  ComputeHandle(const ComputeHandle&) = delete;
  ComputeHandle& operator=(const ComputeHandle&) = delete;

  static std::string makeCacheKey(const LoadedModel& loadedModel,
                                  const MLXWinogradTuneParams& tuneParams,
                                  bool useFP16) {
    return loadedModel.modelDesc.name + "-" + loadedModel.modelDesc.sha256
      + (useFP16 ? "-fp16" : "-fp32")
      + (mlxWinogradEnabled() ? "-wg" : "-nowg")
      + "-it" + std::to_string(tuneParams.inputTransform.tg0)
      + "x"   + std::to_string(tuneParams.inputTransform.tg1)
      + "x"   + std::to_string(tuneParams.inputTransform.wpt)
      + "x"   + std::to_string(tuneParams.inputTransform.vw)
      + "g"   + std::to_string((int)tuneParams.inputTransform.gridOrder)
      + "-ou" + std::to_string(tuneParams.outputUntransform.tg0)
      + "x"   + std::to_string(tuneParams.outputUntransform.tg1)
      + "x"   + std::to_string(tuneParams.outputUntransform.wpt);
  }

  ComputeHandle(ComputeContext* ctx,
                const LoadedModel& loadedModel,
                bool iNHWC,
                bool requireExactNNLen_,
                bool useFP16_,
                int gpuIdx_,
                int maxBatchSize,
                int serverThreadIdx)
    : context(ctx),
      inputsUseNHWC(iNHWC),
      requireExactNNLen(requireExactNNLen_),
      useFP16(useFP16_),
      gpuIdx(gpuIdx_),
      modelCacheKey(),
      model(nullptr),
      modelVersion(loadedModel.modelDesc.modelVersion),
      compiledFuncsMutex(),
      compiledFuncs(),
      coremlOnlyHandle(createCoreMLOnlyHandleIfNeededMLX(
        ctx, &loadedModel, requireExactNNLen_, maxBatchSize, gpuIdx_, serverThreadIdx))
  {
    // Cache ModelDesc fields used by both paths in getOutput.
    numInputChannels = loadedModel.modelDesc.numInputChannels;
    numPolicyChannels = loadedModel.modelDesc.numPolicyChannels;
    // See Model::numPolicyPassChannels comment for the v15+ two-layer pass head
    // rationale: the per-row stride must match the *final* pass output width
    // (gpoolToPassMul2.outChannels for v15+, gpoolToPassMul.outChannels otherwise),
    // not the hidden width.
    numPolicyPassChannels =
      loadedModel.modelDesc.modelVersion >= 15
        ? loadedModel.modelDesc.policyHead.gpoolToPassMul2.outChannels
        : loadedModel.modelDesc.policyHead.gpoolToPassMul.outChannels;
    numValueChannels = loadedModel.modelDesc.numValueChannels;
    numScoreValueChannels = loadedModel.modelDesc.numScoreValueChannels;
    numOwnershipChannels = loadedModel.modelDesc.numOwnershipChannels;

    if(gpuIdx_ == MLX_MUX_ANE) {
      // ANE path: MLX inference state is intentionally left uninitialized.
      checkExactlyOnePath();
      return;
    }

    // GPU path: resolve the Winograd geometry (session memo → disk cache →
    // fresh sweep; see resolveTuneParams), then build/cache the Model below.
    MLXWinogradTuneParams tuneParams = resolveTuneParams(context, loadedModel, useFP16_);

    modelCacheKey = makeCacheKey(loadedModel, tuneParams, useFP16_);

    std::lock_guard<std::mutex> lock(context->cachedModelsMutex);
    if(context->cachedModels.find(modelCacheKey) == context->cachedModels.end()) {
      context->cachedModels[modelCacheKey] =
          std::make_shared<const Model>(loadedModel.modelDesc, tuneParams,
                                        context->nnXLen, context->nnYLen, useFP16_);
    }
    model = context->cachedModels[modelCacheKey];
    context->cachedModelsRefCount[modelCacheKey] += 1;

    // GPU path invariant check.
    checkExactlyOnePath();
  }

  // Invariant: exactly one inference path is live — the MLX/GPU `model` OR the
  // CoreML/ANE `coremlOnlyHandle`, never both or neither. Reads members set in
  // the init list (gpuIdx, coremlOnlyHandle) and `model` as assigned so far, so
  // it is valid at the end of either ctor path.
  void checkExactlyOnePath() const {
    bool hasMLX = (model != nullptr);
    bool hasCoreML = static_cast<bool>(coremlOnlyHandle);
    if(hasMLX == hasCoreML) {
      throw runtime_error(
        string("MLX backend: Logic error - expected exactly one compute handle, got ") +
        (hasMLX && hasCoreML ? "both" : "neither") +
        " (gpuIdx=" + to_string(gpuIdx) + ")");
    }
  }

  ~ComputeHandle() {
    // Only the GPU path populated the cachedModels map; ANE path's destructor
    // is a no-op for the MLX-side state. Swift ARC releases coremlOnlyHandle
    // automatically when the swift::Optional member is destroyed.
    if(gpuIdx == MLX_MUX_ANE)
      return;

    std::lock_guard<std::mutex> lock(context->cachedModelsMutex);
    context->cachedModelsRefCount[modelCacheKey] -= 1;
    assert(context->cachedModelsRefCount[modelCacheKey] >= 0);
    if(context->cachedModelsRefCount[modelCacheKey] == 0) {
      context->cachedModelsRefCount.erase(modelCacheKey);
      context->cachedModels.erase(modelCacheKey);
    }
  }

  // Get or create compiled inference function for the given configuration.
  // GPU path only — must not be called on an ANE-mux handle.
  const CompiledInferenceFunc& getCompiledFunc(int batchSize, int nnXLen, int nnYLen, bool useMask, bool hasMeta) const {
    assert(gpuIdx == MLX_MUX_GPU);
    CompileCacheKey key = std::make_tuple(batchSize, nnXLen, nnYLen, useMask, hasMeta, useFP16);

    std::lock_guard<std::mutex> lock(compiledFuncsMutex);
    auto it = compiledFuncs.find(key);
    if(it != compiledFuncs.end()) {
      return it->second;
    }

    // Create and cache compiled function
    compiledFuncs[key] = model->createCompiledFunc(useMask, hasMeta);
    return compiledFuncs[key];
  }
};

// InputBuffers --------------------------------------------------------------------------------------------------------

struct InputBuffers {
  int maxBatchSize;

  size_t singleInputElts;
  size_t singleInputGlobalElts;
  size_t singleInputMetaElts;

  size_t singlePolicyPassResultElts;
  size_t singlePolicyResultElts;
  size_t singleValueResultElts;
  size_t singleScoreValueResultElts;
  size_t singleOwnershipResultElts;
  size_t singleMaskElts;

  std::vector<float> spatialInput;
  std::vector<float> globalInput;
  std::vector<float> metaInput;
  std::vector<float> userInputMaskBuffer;
  // NCHW staging buffer for the ANE/CoreML dispatch path. The Swift
  // CoreMLComputeHandle.apply() allocates MLMultiArray with shape
  // [1, C, H, W] and memcpys each row's bytes, so it strictly requires
  // NCHW. spatialInput stays NHWC for the MLX/GPU path; rows are
  // transposed into this buffer inside getOutput before dispatch. The
  // MLX/GPU path never reads this buffer.
  std::vector<float> userInputBufferNCHW;
  std::vector<float> policyResults;
  std::vector<float> policyPassResults;
  std::vector<float> valueResults;
  std::vector<float> scoreValueResults;
  std::vector<float> ownershipResults;

  InputBuffers(const LoadedModel* loadedModel, int maxBatchSz, int nnXLen, int nnYLen) {
    const ModelDesc& m = loadedModel->modelDesc;

    maxBatchSize = maxBatchSz;
    singleInputElts = m.numInputChannels * nnXLen * nnYLen;
    singleInputGlobalElts = m.numInputGlobalChannels;
    singleInputMetaElts = m.numInputMetaChannels;

    // See Model::numPolicyPassChannels comment: pass output width is
    // gpoolToPassMul2.outChannels for v15+, gpoolToPassMul.outChannels otherwise.
    // Must match ComputeHandle::numPolicyPassChannels (assertion in getOutput).
    singlePolicyPassResultElts = (size_t)(
      m.modelVersion >= 15
        ? m.policyHead.gpoolToPassMul2.outChannels
        : m.policyHead.gpoolToPassMul.outChannels);
    singlePolicyResultElts = (size_t)(m.numPolicyChannels * nnXLen * nnYLen);
    singleValueResultElts = (size_t)m.numValueChannels;
    singleScoreValueResultElts = (size_t)m.numScoreValueChannels;
    singleOwnershipResultElts = (size_t)m.numOwnershipChannels * nnXLen * nnYLen;
    singleMaskElts = (size_t)nnXLen * nnYLen;

    assert(NNModelVersion::getNumSpatialFeatures(m.modelVersion) == m.numInputChannels);
    assert(NNModelVersion::getNumGlobalFeatures(m.modelVersion) == m.numInputGlobalChannels);
    if(m.numInputMetaChannels > 0) {
      assert(SGFMetadata::METADATA_INPUT_NUM_CHANNELS == m.numInputMetaChannels);
    }

    spatialInput.resize(m.numInputChannels * nnXLen * nnYLen * maxBatchSize);
    globalInput.resize(m.numInputGlobalChannels * maxBatchSize);
    if(m.numInputMetaChannels > 0)
      metaInput.resize(m.numInputMetaChannels * maxBatchSize);
    else
      metaInput.resize(1);

    policyResults.resize(singlePolicyResultElts * maxBatchSize);
    policyPassResults.resize(singlePolicyPassResultElts * maxBatchSize);
    valueResults.resize(singleValueResultElts * maxBatchSize);
    scoreValueResults.resize(singleScoreValueResultElts * maxBatchSize);
    ownershipResults.resize(singleOwnershipResultElts * maxBatchSize);
    userInputMaskBuffer.resize(singleMaskElts * maxBatchSize);
    userInputBufferNCHW.resize(singleInputElts * maxBatchSize);
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

// NeuralNet Interface -------------------------------------------------------------------------------------------------

void NeuralNet::globalInitialize() {
  // MLX initializes automatically
}

void NeuralNet::globalCleanup() {
  // MLX cleans up automatically
}

// Helper implementations (forward-declared before ComputeContext; defined here
// after ComputeContext and LoadedModel are both fully visible).

static swift::Optional<KataGoSwift::CoreMLComputeHandle> convertAndCreateCoreMLOnlyHandleMLX(
  ComputeContext* context,
  const LoadedModel* loadedModel,
  bool requireExactNNLen,
  int maxBatchSize,
  int serverThreadIdx
) {
  int nnXLen = context->nnXLen;
  int nnYLen = context->nnYLen;
  bool useFP16 = (context->useFP16Mode != enabled_t::False);
  bool optimizeMask = requireExactNNLen;

  // Convert model to CoreML format in temp directory
  string coremlModelPath = CoreMLConversion::convertModelToTemp(
    loadedModel->modelPath,
    nnXLen,
    nnYLen,
    useFP16,
    optimizeMask,
    maxBatchSize,
    serverThreadIdx
  );

  // ANE-only context: the converter has just re-read the model from disk
  // (modelPath) into CoreML form, and nothing afterward reads the in-memory
  // weight arrays — the ComputeHandle ctor takes the MLX_MUX_ANE early-return
  // before building any MLX/GPU model, and only scalar dims are read later
  // (numInputChannels, ..., which releaseWeights() preserves). Runs under
  // computeHandleMutex (held by createComputeHandle), so it is not racy;
  // releaseWeights() is idempotent across the per-thread ANE handles.
  if(context->aneOnly) {
    const_cast<LoadedModel*>(loadedModel)->modelDesc.releaseWeights();
  }

  // The Swift createCoreMLComputeHandle entry point expects a
  // MetalComputeContext. Construct one on-the-fly from MLX's context values.
  auto swiftContext = KataGoSwift::createMetalComputeContext(
    static_cast<int32_t>(nnXLen),
    static_cast<int32_t>(nnYLen),
    useFP16);

  // Create CoreML-only compute handle (CPU+ANE) — same Swift entry point Metal uses.
  return KataGoSwift::createCoreMLComputeHandle(
    swift::String(coremlModelPath),
    serverThreadIdx,
    requireExactNNLen,
    loadedModel->modelDesc.numInputChannels,
    loadedModel->modelDesc.numInputGlobalChannels,
    loadedModel->modelDesc.numInputMetaChannels,
    loadedModel->modelDesc.numPolicyChannels,
    loadedModel->modelDesc.numValueChannels,
    loadedModel->modelDesc.numScoreValueChannels,
    loadedModel->modelDesc.numOwnershipChannels,
    swiftContext
  );
}

static swift::Optional<KataGoSwift::CoreMLComputeHandle> createCoreMLOnlyHandleIfNeededMLX(
  ComputeContext* context,
  const LoadedModel* loadedModel,
  bool requireExactNNLen,
  int maxBatchSize,
  int gpuIdx,
  int serverThreadIdx
) {
  if(gpuIdx != MLX_MUX_ANE) {
    return swift::Optional<KataGoSwift::CoreMLComputeHandle>::none();
  }

  if(context->useFP16Mode == enabled_t::False) {
    // Honor the user's explicit FP32 request even on an ANE thread: the ANE
    // is FP16-only, so CoreML falls back to CPU. Result is correct (and
    // deterministic) FP32 CoreML inference, just much slower than GPU.
    cerr << "MLX backend " << serverThreadIdx << ": Note: ANE thread with mlxUseFP16=false: "
         << "the ANE is FP16-only, so CoreML will run this thread on CPU (FP32). "
         << "This is significantly slower than the GPU path; if you wanted ANE acceleration, "
         << "remove mlxUseFP16=false." << endl;
  }

  cerr << "MLX backend " << serverThreadIdx << ": Mux ANE mode - using CoreML (CPU+ANE)" << endl;
  return convertAndCreateCoreMLOnlyHandleMLX(context, loadedModel, requireExactNNLen, maxBatchSize, serverThreadIdx);
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
  (void)loadedModel;

  // aneOnly drives the ANE-path weight release in convertAndCreateCoreMLOnlyHandleMLX.
  // INVARIANT: gpuIdxs must be the complete, deduplicated set of device indices any
  // thread will use under this context. Free the in-memory weights only when every one
  // is MLX_MUX_ANE; if a thread later used an MLX_MUX_GPU index not represented here it
  // would read freed weights.
  bool aneOnly = !gpuIdxs.empty();
  for(int idx : gpuIdxs) {
    if(idx != MLX_MUX_ANE) { aneOnly = false; break; }
  }

  // MLX requires NHWC inputs; this is enforced per-handle via inputsUseNHWC in
  // createComputeHandle (the old context-level useNHWCMode param was removed
  // upstream when createComputeContext was consolidated onto ConfigParser).
  ComputeContext* context = new ComputeContext(nnXLen, nnYLen, useFP16Mode, homeDataDirOverride, logger);
  context->aneOnly = aneOnly;
  // Track live contexts so the cross-context tune memo can be cleared when this
  // engine session ends (see g_winoTuneMemo).
  g_winoTuneMemo.retain();
  // MLX/GPU Winograd autotuner controls (app sets these via -override-config).
  // Read here so the GPU ComputeHandle ctor can honor them; harmless on the ANE
  // path, which returns before the tuner. Calling getBool marks the keys "used"
  // so ConfigParser doesn't flag them as unused overrides.
  context->tunerFull   = cfg.contains("mlxTunerFull") ? cfg.getBool("mlxTunerFull") : false;
  context->tunerReTune = cfg.contains("mlxReTune")    ? cfg.getBool("mlxReTune")    : false;
  return context;
}

void NeuralNet::freeComputeContext(ComputeContext* computeContext) {
  delete computeContext;
  // When the last context of this engine session goes away, release() drops the
  // cross-context tune memo so the next session (e.g. a forced re-tune) starts
  // fresh rather than reusing this session's results.
  g_winoTuneMemo.release();
}

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
  // Auto resolves to fp16. The original acceptance gate (MLX-fp16 paired-t
  // beat both Metal-fp16 and MLX-fp32 with non-overlapping CIs, and
  // testgpuerror accuracy exit=0) is preserved in the traceability commit.
  // Users who need bit-for-bit fp32 reproducibility set `mlxUseFP16 = false`
  // explicitly.
  bool useFP16 = (context->useFP16Mode != enabled_t::False);

  // gpuIdx == -1 is the "no preference" sentinel from upstream; map to default GPU.
  int gpuIdx = (gpuIdxForThisThread == -1) ? MLX_MUX_GPU : gpuIdxForThisThread;
  if(gpuIdx != MLX_MUX_GPU && gpuIdx != MLX_MUX_ANE) {
    throw StringError(
      "MLX backend: Invalid mlxDeviceToUseThread value " + std::to_string(gpuIdx) +
      " for server thread " + std::to_string(serverThreadIdx) +
      ". The MLX backend only supports " + std::to_string(MLX_MUX_GPU) +
      " (GPU via MLX) or " + std::to_string(MLX_MUX_ANE) +
      " (ANE via CoreML).");
  }

  if(logger != NULL) {
    logger->write("MLX backend thread " + Global::intToString(serverThreadIdx) + ": Model version " + Global::intToString(loadedModel->modelDesc.modelVersion));
    logger->write("MLX backend thread " + Global::intToString(serverThreadIdx) + ": Model name: " + loadedModel->modelDesc.name);
    logger->write("MLX backend thread " + Global::intToString(serverThreadIdx) + ": FP16 = " + (useFP16 ? "true" : "false"));
    logger->write("MLX backend thread " + Global::intToString(serverThreadIdx) + ": gpuIdx = " + Global::intToString(gpuIdx));
  }

  if(!inputsUseNHWC)
    throw StringError("MLX backend: inputsUseNHWC = false unsupported");

  // Serialize handle construction: see computeHandleMutex declaration above.
  std::lock_guard<std::mutex> lock(computeHandleMutex);
  return new ComputeHandle(context, *loadedModel, inputsUseNHWC, requireExactNNLen, useFP16,
                           gpuIdx, maxBatchSize, serverThreadIdx);
}

void NeuralNet::freeComputeHandle(ComputeHandle* gpuHandle) {
  delete gpuHandle;
}

bool NeuralNet::isUsingFP16(const ComputeHandle* handle) {
  return handle->useFP16;
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
  const int modelVersion = computeHandle->modelVersion;

  const int numSpatialFeatures = NNModelVersion::getNumSpatialFeatures(modelVersion);
  const int numGlobalFeatures = NNModelVersion::getNumGlobalFeatures(modelVersion);
  const int numMetaFeatures = inputBuffers->singleInputMetaElts;
  assert(numSpatialFeatures == computeHandle->numInputChannels);
  assert(numSpatialFeatures * nnXLen * nnYLen == inputBuffers->singleInputElts);
  assert(numGlobalFeatures == inputBuffers->singleInputGlobalElts);
  const int numPolicyChannels = computeHandle->numPolicyChannels;

  // Copy input data to buffers
  for(int nIdx = 0; nIdx < batchSize; nIdx++) {
    float* rowSpatialInput = inputBuffers->spatialInput.data() + (inputBuffers->singleInputElts * nIdx);
    float* rowGlobalInput = inputBuffers->globalInput.data() + (inputBuffers->singleInputGlobalElts * nIdx);
    float* rowMetaInput = inputBuffers->metaInput.data() + (inputBuffers->singleInputMetaElts * nIdx);

    const float* rowGlobal = inputBufs[nIdx]->rowGlobalBuf.data();
    const float* rowSpatial = inputBufs[nIdx]->rowSpatialBuf.data();
    const float* rowMeta = inputBufs[nIdx]->rowMetaBuf.data();
    const bool hasRowMeta = inputBufs[nIdx]->hasRowMeta;

    std::copy(rowGlobal, rowGlobal + numGlobalFeatures, rowGlobalInput);

    if(numMetaFeatures > 0) {
      testAssert(rowMeta != NULL);
      testAssert(hasRowMeta);
      std::copy(rowMeta, rowMeta + numMetaFeatures, rowMetaInput);
    }
    else {
      testAssert(!hasRowMeta);
    }

    SymmetryHelpers::copyInputsWithSymmetry(rowSpatial, rowSpatialInput, 1, nnYLen, nnXLen, numSpatialFeatures, computeHandle->inputsUseNHWC, inputBufs[nIdx]->symmetry);

    // ANE/CoreML path needs an NCHW spatial buffer because the Swift
    // CoreMLComputeHandle.apply() allocates MLMultiArray with shape
    // [1, C, H, W] and raw memcpys C*H*W floats per row. spatialInput
    // is NHWC (required by the MLX/GPU path's mx::array shape), so we
    // transpose each row into userInputBufferNCHW here. The validity
    // mask (channel 0) sits at the start of the converted row, so it
    // collapses to a contiguous memcpy into userInputMaskBuffer.
    //
    // When the mlpackage was converted with optimize_identity_mask=true
    // (i.e., requireExactNNLen=true) the ANE model ignores the mask
    // buffer, but populating it unconditionally costs essentially
    // nothing (one memcpy of H*W floats) and avoids a silent-
    // misprediction footgun when optimize_identity_mask=false.
    //
    // The MLX/GPU path slices channel 0 itself via mx::slice and does
    // not read userInputMaskBuffer or userInputBufferNCHW.
    if(computeHandle->coremlOnlyHandle) {
      const int C = computeHandle->numInputChannels;
      const size_t HW = inputBuffers->singleMaskElts;  // nnXLen * nnYLen
      float* rowNCHW = inputBuffers->userInputBufferNCHW.data()
                     + inputBuffers->singleInputElts * nIdx;
      const float* rowNHWC = rowSpatialInput;  // [H*W, C]
      for(int c = 0; c < C; c++) {
        float* dstCh = rowNCHW + (size_t)c * HW;
        for(size_t hw = 0; hw < HW; hw++) {
          dstCh[hw] = rowNHWC[hw * C + c];
        }
      }
      float* dstMask = inputBuffers->userInputMaskBuffer.data()
                     + inputBuffers->singleMaskElts * nIdx;
      std::memcpy(dstMask, rowNCHW, HW * sizeof(float));
    }
  }

  // Dispatch to appropriate path based on mux mode.
  if(computeHandle->coremlOnlyHandle) {
    // ANE path: dispatch through the Swift CoreMLComputeHandle. Swift
    // creates MLMultiArray(shape: [1, C, H, W]) per row and memcpys
    // C*H*W floats — strict NCHW. We pass userInputBufferNCHW (rows
    // transposed from NHWC in the loop above) instead of spatialInput.
    // The mask is the contiguous H*W float prefix of each NCHW row,
    // already lifted into userInputMaskBuffer above. The mlpackage
    // ignores the mask buffer iff it was converted with
    // optimize_identity_mask=true.
    computeHandle->coremlOnlyHandle.get().apply(
      inputBuffers->userInputBufferNCHW.data(),
      inputBuffers->globalInput.data(),
      inputBuffers->metaInput.data(),  // always non-null (resized to at least 1 in InputBuffers ctor)
      inputBuffers->userInputMaskBuffer.data(),
      inputBuffers->policyResults.data(),
      inputBuffers->policyPassResults.data(),
      inputBuffers->valueResults.data(),
      inputBuffers->scoreValueResults.data(),
      inputBuffers->ownershipResults.data(),
      batchSize);
  } else {
    // GPU path: run the MLX compiled function exactly as before.
    const bool useMask = !computeHandle->requireExactNNLen;
    const bool hasMeta = (numMetaFeatures > 0);
    const CompiledInferenceFunc& compiledFunc = computeHandle->getCompiledFunc(batchSize, nnXLen, nnYLen, useMask, hasMeta);

    computeHandle->model->applyCompiled(
      compiledFunc,
      inputBuffers->spatialInput.data(),
      inputBuffers->globalInput.data(),
      (numMetaFeatures > 0 ? inputBuffers->metaInput.data() : nullptr),
      batchSize,
      nnXLen,
      nnYLen,
      computeHandle->requireExactNNLen,
      inputBuffers->policyResults.data(),
      inputBuffers->policyPassResults.data(),
      inputBuffers->valueResults.data(),
      inputBuffers->scoreValueResults.data(),
      inputBuffers->ownershipResults.data()
    );
  }

  assert(inputBuffers->singlePolicyPassResultElts == (size_t)computeHandle->numPolicyPassChannels);
  assert(inputBuffers->singlePolicyResultElts == numPolicyChannels * nnXLen * nnYLen);
  assert(outputs.size() == batchSize);

  float policyProbsTmp[NNPos::MAX_NN_POLICY_SIZE];

  float* policyData = inputBuffers->policyResults.data();
  float* policyPassData = inputBuffers->policyPassResults.data();
  float* valueData = inputBuffers->valueResults.data();
  float* scoreValueData = inputBuffers->scoreValueResults.data();
  float* ownershipData = inputBuffers->ownershipResults.data();

  for(int row = 0; row < batchSize; row++) {
    NNOutput* output = outputs[row];
    assert(output->nnXLen == nnXLen);
    assert(output->nnYLen == nnYLen);
    float policyOptimism = (float)inputBufs[row]->policyOptimism;

    const float* policyPassSrcBuf = policyPassData + row * computeHandle->numPolicyPassChannels;
    const float* policySrcBuf = policyData + row * numPolicyChannels * nnXLen * nnYLen;
    float* policyProbs = output->policyProbs;

    // Handle policy optimism (version >= 12). The optimism mix uses
    // channel 0 (p) and channel 1 (pOpt) of the policy output; v16+
    // channels 2-3 are ignored here, matching MetalProcess::processOptimism
    // in metalbackend.cpp.
    //
    // MLX/GPU writes NHWC: channels are interleaved per spatial position.
    // CoreML/ANE writes NCHW (MLMultiArray shape [1, C, H, W], contiguous
    // memcpy in metalbackend.swift copyMultiArray): channel 0 occupies the
    // first HW floats, channel 1 the next HW, etc. Stride differs per path.
    if(numPolicyChannels == 2 || (numPolicyChannels == 4 && modelVersion >= 16)) {
      const int HW = nnXLen * nnYLen;
      const bool isNCHW = (bool)computeHandle->coremlOnlyHandle;
      const int strideI   = isNCHW ? 1  : numPolicyChannels;
      const int strideOpt = isNCHW ? HW : 1;
      for(int i = 0; i < HW; i++) {
        float p    = policySrcBuf[i * strideI];
        float pOpt = policySrcBuf[i * strideI + strideOpt];
        policyProbsTmp[i] = p + (pOpt - p) * policyOptimism;
      }
      SymmetryHelpers::copyOutputsWithSymmetry(policyProbsTmp, policyProbs, 1, nnYLen, nnXLen, inputBufs[row]->symmetry);
      policyProbs[nnXLen * nnYLen] = policyPassSrcBuf[0] + (policyPassSrcBuf[1] - policyPassSrcBuf[0]) * policyOptimism;
    }
    else {
      assert(numPolicyChannels == 1);
      SymmetryHelpers::copyOutputsWithSymmetry(policySrcBuf, policyProbs, 1, nnYLen, nnXLen, inputBufs[row]->symmetry);
      policyProbs[inputBuffers->singlePolicyResultElts] = policyPassSrcBuf[0];
    }

    int numValueChannels = computeHandle->numValueChannels;
    assert(numValueChannels == 3);
    output->whiteWinProb = valueData[row * numValueChannels];
    output->whiteLossProb = valueData[row * numValueChannels + 1];
    output->whiteNoResultProb = valueData[row * numValueChannels + 2];

    if(output->whiteOwnerMap != NULL) {
      const float* ownershipSrcBuf = ownershipData + row * nnXLen * nnYLen;
      assert(computeHandle->numOwnershipChannels == 1);
      SymmetryHelpers::copyOutputsWithSymmetry(ownershipSrcBuf, output->whiteOwnerMap, 1, nnYLen, nnXLen, inputBufs[row]->symmetry);
    }

    if(modelVersion >= 9) {
      int numScoreValueChannels = computeHandle->numScoreValueChannels;
      assert(numScoreValueChannels == 6);
      output->whiteScoreMean = scoreValueData[row * numScoreValueChannels];
      output->whiteScoreMeanSq = scoreValueData[row * numScoreValueChannels + 1];
      output->whiteLead = scoreValueData[row * numScoreValueChannels + 2];
      output->varTimeLeft = scoreValueData[row * numScoreValueChannels + 3];
      output->shorttermWinlossError = scoreValueData[row * numScoreValueChannels + 4];
      output->shorttermScoreError = scoreValueData[row * numScoreValueChannels + 5];
    }
    else if(modelVersion >= 8) {
      int numScoreValueChannels = computeHandle->numScoreValueChannels;
      assert(numScoreValueChannels == 4);
      output->whiteScoreMean = scoreValueData[row * numScoreValueChannels];
      output->whiteScoreMeanSq = scoreValueData[row * numScoreValueChannels + 1];
      output->whiteLead = scoreValueData[row * numScoreValueChannels + 2];
      output->varTimeLeft = scoreValueData[row * numScoreValueChannels + 3];
      output->shorttermWinlossError = 0;
      output->shorttermScoreError = 0;
    }
    else if(modelVersion >= 4) {
      int numScoreValueChannels = computeHandle->numScoreValueChannels;
      assert(numScoreValueChannels == 2);
      output->whiteScoreMean = scoreValueData[row * numScoreValueChannels];
      output->whiteScoreMeanSq = scoreValueData[row * numScoreValueChannels + 1];
      output->whiteLead = output->whiteScoreMean;
      output->varTimeLeft = 0;
      output->shorttermWinlossError = 0;
      output->shorttermScoreError = 0;
    }
    else if(modelVersion >= 3) {
      int numScoreValueChannels = computeHandle->numScoreValueChannels;
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

void NeuralNet::printDevices() {
  cout << "MLX Backend (Apple Silicon)" << endl;
  cout << "Default device: " << mx::default_device() << endl;
}

// FOR TESTING ---------------------------------------------------------------------------------------------------------

bool NeuralNet::testEvaluateConv(
  const ConvLayerDesc* desc,
  int batchSize,
  int nnXLen,
  int nnYLen,
  bool useFP16,
  bool useNHWC,
  const vector<float>& inputBuffer,
  vector<float>& outputBuffer
) {
  // Run MLX-specific aux tests (Winograd kernel + tuner) exactly once per
  // process, on the first invocation of testEvaluateConv. This is the
  // MLX-side hook reachable from Tests::runNNLayerTests through
  // testConvLayer, allowing testnn.cpp to stay backend-agnostic.
  // The flag is set BEFORE the calls so a propagating exception does not
  // cause the aux tests to re-run on subsequent conv configs.
  static bool ranMLXAuxTests = false;
  if(!ranMLXAuxTests) {
    ranMLXAuxTests = true;
    runMLXWinogradTests();
    runMLXWinotunerTests();
  }

  if(!useNHWC) {
    return false; // MLX only supports NHWC
  }

  size_t numOutputFloats = (size_t)batchSize * nnXLen * nnYLen * desc->outChannels;
  outputBuffer.resize(numOutputFloats);

  MLXWinograd::InputTransform   defaultInCfg;
  MLXWinograd::OutputUntransform defaultOutCfg;
  ConvLayer layer(*desc, defaultInCfg, defaultOutCfg, useFP16);
  mx::Shape inputShape = {batchSize, nnYLen, nnXLen, desc->inChannels};
  mx::array input = mx::array(inputBuffer.data(), inputShape, mx::float32);
  mx::array computeInput = toComputeDtype(input, useFP16);
  mx::array output = layer.apply(computeInput);
  if(useFP16) output = mx::astype(output, mx::float32);
  mx::eval(output);

  memcpy(outputBuffer.data(), output.data<float>(), numOutputFloats * sizeof(float));
  return true;
}

bool NeuralNet::testEvaluateBatchNorm(
  const BatchNormLayerDesc* desc,
  int batchSize,
  int nnXLen,
  int nnYLen,
  bool useFP16,
  bool useNHWC,
  const vector<float>& inputBuffer,
  const vector<float>& maskBuffer,
  vector<float>& outputBuffer
) {
  if(!useNHWC) {
    return false;
  }

  size_t numOutputFloats = (size_t)batchSize * nnXLen * nnYLen * desc->numChannels;
  outputBuffer.resize(numOutputFloats);

  BatchNormLayer layer(*desc, ACTIVATION_IDENTITY, useFP16);
  mx::Shape inputShape = {batchSize, nnYLen, nnXLen, desc->numChannels};
  mx::Shape maskShape = {batchSize, nnYLen, nnXLen, 1};
  mx::array input = mx::array(inputBuffer.data(), inputShape, mx::float32);
  mx::array mask = mx::array(maskBuffer.data(), maskShape, mx::float32);
  mx::array computeInput = toComputeDtype(input, useFP16);
  mx::array computeMask = toComputeDtype(mask, useFP16);
  mx::array output = layer.apply(computeInput, computeMask, /*useMask=*/true);
  if(useFP16) output = mx::astype(output, mx::float32);
  mx::eval(output);

  memcpy(outputBuffer.data(), output.data<float>(), numOutputFloats * sizeof(float));
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
  vector<float>& outputBuffer
) {
  if(!useNHWC) {
    return false;
  }

  size_t numOutputFloats = (size_t)batchSize * nnXLen * nnYLen * desc->preBN.numChannels;
  outputBuffer.resize(numOutputFloats);

  MLXWinograd::InputTransform   defaultInCfg;
  MLXWinograd::OutputUntransform defaultOutCfg;
  ResidualBlock block(*desc, defaultInCfg, defaultOutCfg, useFP16);
  mx::Shape inputShape = {batchSize, nnYLen, nnXLen, desc->preBN.numChannels};
  mx::Shape maskShape = {batchSize, nnYLen, nnXLen, 1};
  mx::array input = mx::array(inputBuffer.data(), inputShape, mx::float32);
  mx::array mask = mx::array(maskBuffer.data(), maskShape, mx::float32);
  mx::array computeInput = toComputeDtype(input, useFP16);
  mx::array computeMask = toComputeDtype(mask, useFP16);
  mx::array output = block.apply(computeInput, computeMask, /*useMask=*/true);
  if(useFP16) output = mx::astype(output, mx::float32);
  mx::eval(output);

  memcpy(outputBuffer.data(), output.data<float>(), numOutputFloats * sizeof(float));
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
  vector<float>& outputBuffer
) {
  if(!useNHWC) {
    return false;
  }

  size_t numOutputFloats = (size_t)batchSize * nnXLen * nnYLen * desc->preBN.numChannels;
  outputBuffer.resize(numOutputFloats);

  MLXWinograd::InputTransform   defaultInCfg;
  MLXWinograd::OutputUntransform defaultOutCfg;
  GlobalPoolingResidualBlock block(*desc, defaultInCfg, defaultOutCfg, useFP16);
  mx::Shape inputShape = {batchSize, nnYLen, nnXLen, desc->preBN.numChannels};
  mx::Shape maskShape = {batchSize, nnYLen, nnXLen, 1};
  mx::array input = mx::array(inputBuffer.data(), inputShape, mx::float32);
  mx::array mask = mx::array(maskBuffer.data(), maskShape, mx::float32);
  mx::array computeInput = toComputeDtype(input, useFP16);
  mx::array computeMask = toComputeDtype(mask, useFP16);
  std::vector<int> sumAxes = {1, 2};
  // maskSum stays FP32 for precision
  mx::array maskSum = mx::sum(mask, sumAxes, /*keepdims=*/true);
  mx::array output = block.apply(computeInput, computeMask, maskSum, /*useMask=*/true);
  if(useFP16) output = mx::astype(output, mx::float32);
  mx::eval(output);

  memcpy(outputBuffer.data(), output.data<float>(), numOutputFloats * sizeof(float));
  return true;
}

// Directly-asserting unit test for BatchNormLayer fp16 mode.
// Declared here because BatchNormLayer is not in any public header.
// Called from runMLXWinogradTests() in mlxtests.cpp.
void runMLXBatchNormFP16Test() {
  namespace mxc = mx;  // reuse the file-scope `mx` alias
  using std::cout;
  using std::endl;

  int N=1,H=5,W=5,C=4;
  std::vector<float> mean(C, 0.0f), variance(C, 1.0f), scale(C, 1.0f), bias(C, 0.0f);
  BatchNormLayerDesc bnDesc;
  bnDesc.name = "bnFP16Test";
  bnDesc.numChannels = C;
  bnDesc.epsilon = 1e-5f;
  bnDesc.mean = mean;
  bnDesc.variance = variance;
  bnDesc.scale = scale;
  bnDesc.bias = bias;
  BatchNormLayer bn(bnDesc, ACTIVATION_IDENTITY, /*useFP16=*/true);

  // mergedScale/mergedBias must be fp32 even in fp16 mode.
  testAssert(bn.mergedScale.dtype() == mxc::float32);
  testAssert(bn.mergedBias.dtype()  == mxc::float32);

  // apply() must return fp16 when useFP16=true.
  std::vector<float> inV((size_t)N*H*W*C, 0.5f);
  std::vector<float> maskV((size_t)N*H*W*1, 1.0f);
  mxc::array inArrF32(inV.data(), {N,H,W,C}, mxc::float32);
  mxc::array inArr = mxc::astype(inArrF32, mxc::float16);
  mxc::array maskArrF32(maskV.data(), {N,H,W,1}, mxc::float32);
  mxc::array maskArr = mxc::astype(maskArrF32, mxc::float16);
  mxc::array out = bn.apply(inArr, maskArr, /*useMask=*/true);
  mxc::eval(out);
  testAssert(out.dtype() == mxc::float16);
  cout << "  BatchNormLayer fp16: mergedScale/Bias fp32, output fp16 OK" << endl;
}

// Directly-asserting unit test for ConvLayer fp16 Winograd path.
// Declared here because ConvLayer is not in any public header.
// Called from runMLXWinogradTests() in mlxtests.cpp.
void runMLXConvLayerFP16WinogradTest() {
  namespace mxc = mx;  // reuse the file-scope `mx` alias
  using std::cout;
  using std::endl;

  int N=1,H=19,W=19,Cin=8,Cout=16;
  std::mt19937 grng(779);
  std::uniform_real_distribution<float> gdist(-1.f,1.f);
  std::vector<float> in((size_t)N*H*W*Cin); for(auto&x:in)x=gdist(grng);
  std::vector<float> w((size_t)Cout*Cin*9); for(auto&x:w)x=gdist(grng);
  auto refv = MLXWinograd::cpuConv2d3x3(in,N,H,W,Cin,w,Cout);

  ConvLayerDesc convDesc;
  convDesc.name = "convFP16WinogradTest";
  convDesc.convYSize = 3;
  convDesc.convXSize = 3;
  convDesc.inChannels = Cin;
  convDesc.outChannels = Cout;
  convDesc.dilationY = 1;
  convDesc.dilationX = 1;
  convDesc.weights = w;

  MLXWinograd::InputTransform inCfg;
  MLXWinograd::OutputUntransform outCfg;
  ConvLayer conv(convDesc, inCfg, outCfg, /*useFP16=*/true);
  testAssert(conv.useWinograd);  // fp16 still picks Winograd

  mxc::array inArrF32(in.data(),{N,H,W,Cin},mxc::float32);
  mxc::array inArr = mxc::astype(inArrF32, mxc::float16);
  mxc::array o = conv.apply(inArr);
  mxc::eval(o);
  testAssert(o.dtype() == mxc::float16);
  mxc::array oF32 = mxc::astype(o, mxc::float32);
  mxc::eval(oF32);
  const float* od = oF32.data<float>();
  double maxErr=0.0;
  for(size_t i=0;i<refv.size();i++)
    maxErr=std::max(maxErr,(double)std::fabs(refv[i]-od[i]));
  cout<<"  ConvLayer fp16 winograd maxErr="<<maxErr<<endl;
  testAssert(maxErr < 5e-2);
}

// Directly-asserting unit test for the transformer layer path
// (TransformerRMSNormLayer + TransformerAttentionBlock). These structs are
// file-local to this TU, so the test lives here (mirroring the BatchNorm/Conv
// tests above); it is forward-declared and called from runMLXWinogradTests().
//
// Guards covered:
//   1. RMSNorm fp32 numeric correctness + weight stays fp32.
//   2. RMSNorm fp16 dtype + closeness + finiteness.
//   3. Attention OUTPUT dtype-preservation under useFP16 (THE "§1" guard: a
//      stray mx::array(scale) is a strong fp32 array in MLX C++, which silently
//      promoted the attention output -- and the whole residual stream -- to
//      fp32). Plus fp16-vs-fp32 closeness, mask on/off, and a fixed-RoPE variant.
//   4. Structural residual anchor: outProj all-zeros => apply() == trunk exactly.
void runMLXTransformerLayerFP16Test() {
  namespace mxc = mx;  // reuse the file-scope `mx` alias
  using std::cout;
  using std::endl;

  std::mt19937 rng(20260607u);
  std::uniform_real_distribution<float> dist(-0.3f, 0.3f);  // keep fp16 well-conditioned
  auto fillRand = [&](std::vector<float>& v){ for(auto& x : v) x = dist(rng); };

  // ---- (1)+(2) RMSNorm correctness + dtype ----
  {
    const int N=1,H=3,W=3,C=8;
    const float eps=1e-5f;
    std::vector<float> weightV(C); fillRand(weightV);
    // Bias the weights to ~1.0 so the normalize is not degenerate.
    for(auto& x : weightV) x += 1.0f;

    TransformerRMSNormDesc desc;
    desc.name = "rmsnormFP16Test";
    desc.numChannels = C;
    desc.epsilon = eps;
    desc.weight = weightV;

    std::vector<float> inV((size_t)N*H*W*C); fillRand(inV);
    std::vector<float> maskV((size_t)N*H*W*1, 1.0f);

    mxc::array inArrF32(inV.data(), {N,H,W,C}, mxc::float32);
    mxc::array maskF32(maskV.data(), {N,H,W,1}, mxc::float32);

    // fp32 layer; useMask=false.
    TransformerRMSNormLayer rmsF32(desc, /*useFP16=*/false);
    testAssert(rmsF32.weight.dtype() == mxc::float32);
    mxc::array outF32 = rmsF32.apply(inArrF32, maskF32, /*useMask=*/false);
    mxc::eval(outF32);
    const float* op = outF32.data<float>();

    // CPU reference: per position, ms = mean_c(x^2); r = 1/sqrt(ms+eps);
    // out_c = x_c * r * weight_c.
    double maxErr=0.0;
    for(int pos=0; pos<N*H*W; pos++) {
      double ms=0.0;
      for(int c=0;c<C;c++){ float x=inV[(size_t)pos*C+c]; ms += (double)x*x; }
      ms /= C;
      double r = 1.0/std::sqrt(ms+eps);
      for(int c=0;c<C;c++){
        double ref = inV[(size_t)pos*C+c]*r*weightV[c];
        maxErr = std::max(maxErr, std::fabs(ref - (double)op[(size_t)pos*C+c]));
      }
    }
    cout<<"  TransformerRMSNorm fp32 maxErr="<<maxErr<<endl;
    testAssert(maxErr < 1e-4);

    // fp16 layer: dtype, closeness, finiteness.
    TransformerRMSNormLayer rmsF16(desc, /*useFP16=*/true);
    testAssert(rmsF16.weight.dtype() == mxc::float32);  // weight stays fp32
    mxc::array inArrF16 = mxc::astype(inArrF32, mxc::float16);
    mxc::array maskF16 = mxc::astype(maskF32, mxc::float16);
    mxc::array outF16 = rmsF16.apply(inArrF16, maskF16, /*useMask=*/false);
    mxc::eval(outF16);
    testAssert(outF16.dtype() == mxc::float16);
    mxc::array outF16to32 = mxc::astype(outF16, mxc::float32);
    mxc::eval(outF16to32);
    const float* op16 = outF16to32.data<float>();
    double maxErr16=0.0; bool allFinite=true;
    for(size_t i=0;i<(size_t)N*H*W*C;i++){
      if(!std::isfinite(op16[i])) allFinite=false;
      maxErr16 = std::max(maxErr16, std::fabs((double)op[i] - (double)op16[i]));
    }
    cout<<"  TransformerRMSNorm fp16 dtype OK, vs-fp32 maxErr="<<maxErr16<<endl;
    testAssert(allFinite);
    testAssert(maxErr16 < 5e-2);
  }

  // Build a TransformerAttentionDesc with the standard projection shapes.
  // qProj  : [inChannels, numHeads*qHeadDim]
  // kProj  : [inChannels, numKVHeads*qHeadDim]
  // vProj  : [inChannels, numKVHeads*vHeadDim]
  // outProj: [numHeads*vHeadDim, inChannels]
  // preLN  : TransformerRMSNormDesc(numChannels=inChannels)
  auto makeMatMulDesc = [&](MatMulLayerDesc& d, const std::string& nm,
                            int inC, int outC, bool zero){
    d.name = nm; d.inChannels = inC; d.outChannels = outC;
    d.weights.assign((size_t)inC*outC, 0.0f);
    if(!zero) fillRand(d.weights);
  };
  auto buildAttnDesc = [&](TransformerAttentionDesc& desc,
                           int inChannels, int numHeads, int numKVHeads,
                           int qHeadDim, int vHeadDim,
                           bool useRope, bool zeroOutProj){
    desc.name = "attnFP16Test";
    desc.numHeads = numHeads;
    desc.numKVHeads = numKVHeads;
    desc.qHeadDim = qHeadDim;
    desc.vHeadDim = vHeadDim;
    desc.useRope = useRope;
    desc.learnableRope = false;       // fixed RoPE only here
    desc.ropeTheta = 10000.0f;        // standard base; only read when useRope
    // preLN.
    desc.preLN.name = "attnPreLN";
    desc.preLN.numChannels = inChannels;
    desc.preLN.epsilon = 1e-5f;
    std::vector<float> lnW(inChannels); fillRand(lnW);
    for(auto& x : lnW) x += 1.0f;
    desc.preLN.weight = lnW;
    // Projections.
    makeMatMulDesc(desc.qProj,  "qProj",  inChannels, numHeads*qHeadDim,   false);
    makeMatMulDesc(desc.kProj,  "kProj",  inChannels, numKVHeads*qHeadDim, false);
    makeMatMulDesc(desc.vProj,  "vProj",  inChannels, numKVHeads*vHeadDim, false);
    makeMatMulDesc(desc.outProj,"outProj",numHeads*vHeadDim, inChannels,   zeroOutProj);
  };

  // ---- (3) Attention dtype preservation (THE §1 guard) + fp16-vs-fp32 ----
  {
    const int N=1, nnX=3, nnY=3;  // seq=9
    const int inChannels=8, numHeads=2, numKVHeads=1, qHeadDim=4, vHeadDim=4;

    // Fixed weights (one RNG draw) reused by the fp16 and fp32 blocks so the two
    // can be compared. Build the desc once for fp16 and an identical one for fp32.
    auto runVariant = [&](bool useRope){
      // descF16 and descF32 must get BIT-IDENTICAL weights so the only difference
      // between the two blocks is the compute dtype. buildOne therefore seeds a
      // FRESH RNG from the same base seed on every call (a single shared RNG would
      // hand the second desc a different draw, i.e. a different random network --
      // the comparison would then measure network divergence, not fp16 error).
      const unsigned baseSeed = 424242u + (useRope?1u:0u);
      auto buildOne = [&](TransformerAttentionDesc& desc){
        std::mt19937 localRng(baseSeed);
        std::uniform_real_distribution<float> ld(-0.3f, 0.3f);
        auto lfill = [&](std::vector<float>& v){ for(auto& x:v) x=ld(localRng); };
        desc.name = "attnFP16Test";
        desc.numHeads = numHeads; desc.numKVHeads = numKVHeads;
        desc.qHeadDim = qHeadDim; desc.vHeadDim = vHeadDim;
        desc.useRope = useRope; desc.learnableRope = false; desc.ropeTheta = 10000.0f;
        desc.preLN.name = "attnPreLN"; desc.preLN.numChannels = inChannels; desc.preLN.epsilon = 1e-5f;
        std::vector<float> lnW(inChannels); lfill(lnW); for(auto& x:lnW) x+=1.0f; desc.preLN.weight = lnW;
        auto mk = [&](MatMulLayerDesc& d, const std::string& nm, int inC, int outC){
          d.name=nm; d.inChannels=inC; d.outChannels=outC;
          d.weights.assign((size_t)inC*outC,0.0f); lfill(d.weights);
        };
        mk(desc.qProj,  "qProj",  inChannels, numHeads*qHeadDim);
        mk(desc.kProj,  "kProj",  inChannels, numKVHeads*qHeadDim);
        mk(desc.vProj,  "vProj",  inChannels, numKVHeads*vHeadDim);
        mk(desc.outProj,"outProj",numHeads*vHeadDim, inChannels);
      };

      TransformerAttentionDesc descF16; buildOne(descF16);
      TransformerAttentionDesc descF32; buildOne(descF32);  // same RNG seed => same weights

      std::vector<float> trunkV((size_t)N*nnY*nnX*inChannels);
      { std::mt19937 tr(99u); std::uniform_real_distribution<float> td(-0.3f,0.3f);
        for(auto& x:trunkV) x=td(tr); }
      std::vector<float> maskV((size_t)N*nnY*nnX*1, 1.0f);
      // For the useMask=true case, mask out two positions.
      maskV[2]=0.0f; maskV[5]=0.0f;

      mxc::array trunkF32(trunkV.data(), {N,nnY,nnX,inChannels}, mxc::float32);
      mxc::array maskF32(maskV.data(), {N,nnY,nnX,1}, mxc::float32);
      mxc::array trunkF16 = mxc::astype(trunkF32, mxc::float16);
      mxc::array maskF16  = mxc::astype(maskF32,  mxc::float16);

      TransformerAttentionBlock blkF16(descF16, nnX, nnY, /*useFP16=*/true);
      TransformerAttentionBlock blkF32(descF32, nnX, nnY, /*useFP16=*/false);

      for(bool useMask : {true, false}) {
        mxc::array oF16 = blkF16.apply(trunkF16, maskF16, useMask);
        mxc::eval(oF16);
        // THE guard: output dtype must stay fp16 (a stray fp32 scalar promotes it).
        testAssert(oF16.dtype() == mxc::float16);
        mxc::array oF16to32 = mxc::astype(oF16, mxc::float32);
        mxc::eval(oF16to32);
        const float* p16 = oF16to32.data<float>();
        bool finite=true; for(size_t i=0;i<trunkV.size();i++) if(!std::isfinite(p16[i])) finite=false;
        testAssert(finite);

        mxc::array oF32 = blkF32.apply(trunkF32, maskF32, useMask);
        mxc::eval(oF32);
        const float* p32 = oF32.data<float>();
        double maxErr=0.0, maxAbs=0.0;
        for(size_t i=0;i<trunkV.size();i++) {
          maxErr = std::max(maxErr, std::fabs((double)p16[i]-(double)p32[i]));
          maxAbs = std::max(maxAbs, std::fabs((double)p32[i]));
        }
        double relErr = maxErr / (maxAbs + 1e-6);
        cout<<"  TransformerAttention fp16 dtype OK (useRope="<<useRope
            <<" useMask="<<useMask<<") vs-fp32 maxErr="<<maxErr
            <<" maxAbs="<<maxAbs<<" relErr="<<relErr<<endl;
        // The dtype assert above is the precise §1 regression guard. This is a
        // secondary numeric check: fp16 and fp32 run the SAME weights, so this
        // measures genuine fp16 accuracy of the whole chain (preLN -> Q/K/V
        // matmuls -> masked softmax -> @V -> out-proj). The bound is RELATIVE to
        // the output scale because the attention-delta magnitude is set by the
        // (random) projection scale. fp16 here lands a few % relative; 0.10 keeps
        // comfortable margin while still catching a real fp16/fp32 divergence.
        testAssert(relErr < 0.10);
      }
    };

    runVariant(/*useRope=*/false);
    // Fixed RoPE setup is straightforward (computeRopeCosSin reads only ropeTheta
    // and qHeadDim for the non-learnable path, and qHeadDim=4 is even), so cover
    // useRope=true too. Learnable RoPE is NOT covered here (it needs ropeFreqs).
    runVariant(/*useRope=*/true);
  }

  // ---- (4) Structural residual anchor: outProj all zeros => apply() == trunk ----
  {
    const int N=1, nnX=3, nnY=3;
    const int inChannels=8, numHeads=2, numKVHeads=1, qHeadDim=4, vHeadDim=4;
    TransformerAttentionDesc desc;
    buildAttnDesc(desc, inChannels, numHeads, numKVHeads, qHeadDim, vHeadDim,
                  /*useRope=*/false, /*zeroOutProj=*/true);

    std::vector<float> trunkV((size_t)N*nnY*nnX*inChannels); fillRand(trunkV);
    std::vector<float> maskV((size_t)N*nnY*nnX*1, 1.0f);
    mxc::array trunkF32(trunkV.data(), {N,nnY,nnX,inChannels}, mxc::float32);
    mxc::array maskF32(maskV.data(), {N,nnY,nnX,1}, mxc::float32);

    TransformerAttentionBlock blk(desc, nnX, nnY, /*useFP16=*/false);
    // useMask=false to avoid mask interactions: zero out-proj => added spatial
    // term is exactly 0, so apply() must return exactly trunk.
    mxc::array out = blk.apply(trunkF32, maskF32, /*useMask=*/false);
    mxc::eval(out);
    const float* op = out.data<float>();
    double maxErr=0.0;
    for(size_t i=0;i<trunkV.size();i++)
      maxErr = std::max(maxErr, std::fabs((double)trunkV[i]-(double)op[i]));
    cout<<"  TransformerAttention zero-outProj residual maxErr="<<maxErr<<endl;
    testAssert(maxErr < 1e-3);
  }

  cout << "  TransformerLayer fp16 tests OK" << endl;
}

#endif // USE_MLX_BACKEND
