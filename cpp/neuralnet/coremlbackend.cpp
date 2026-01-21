#ifdef USE_COREML_BACKEND

#include "../neuralnet/modelversion.h"
#include "../neuralnet/nneval.h"
#include "../neuralnet/nninputs.h"
#include "../neuralnet/nninterface.h"
#include "../neuralnet/coremlbackend.h"

#include <katagocoreml/KataGoConverter.hpp>
#include <ghc/filesystem.hpp>
#include <mutex>
#include <chrono>
#include <cassert>
#include <unistd.h>  // For getpid()

using namespace std;

//------------------------------------------------------------------------------
// CoreML Model Conversion - Native C++ using katagocoreml library
//------------------------------------------------------------------------------

namespace gfs = ghc::filesystem;

// Minimum batch sizes for hybrid execution mode.
// Hybrid splits batches between CoreML (CPU+ANE) and MPSGraph (GPU).
// When batch is too small to split, prefer MPSGraph-only for stability:
// MPSGraph has more predictable latency and avoids CoreML dispatch overhead.
static constexpr int MIN_COREML_BATCH = 1;
static constexpr int MIN_MPSGRAPH_BATCH = 1;

namespace CoreMLConversion {

// Get temp directory for model conversion
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

// Convert KataGo model to CoreML in temp directory, returns path to .mlpackage
// The caller (Swift side) is responsible for deleting the temp file after loading
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
  cerr << "Core ML backend " << serverThreadIdx << ": Converting model to " << tempPath << endl;

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
      cerr << "Core ML backend " << serverThreadIdx << ": Warning: Failed to clean up partial conversion at " << tempPath << ": " << ec.message() << endl;
    }
    throw runtime_error(string("Core ML model conversion failed: ") + e.what());
  }

  cerr << "Core ML backend " << serverThreadIdx << ": Conversion completed" << endl;
  return tempPath;
}

}  // namespace CoreMLConversion

//------------------------------------------------------------------------------
// Model Descriptor Conversion - C++ to Swift types for MPSGraph
//------------------------------------------------------------------------------

namespace CoreMLProcess {

/// Converts a ConvLayerDesc instance from C++ to Swift
SWConvLayerDesc convLayerDescToSwift(const ConvLayerDesc* desc) {
  return createSWConvLayerDesc(
    desc->convYSize,
    desc->convXSize,
    desc->inChannels,
    desc->outChannels,
    desc->dilationY,
    desc->dilationX,
    (float*)desc->weights.data());
}

/// Converts a BatchNormLayerDesc instance from C++ to Swift
SWBatchNormLayerDesc batchNormLayerDescToSwift(const BatchNormLayerDesc* desc) {
  return createSWBatchNormLayerDesc(
    desc->numChannels,
    (float*)desc->mergedScale.data(),
    (float*)desc->mergedBias.data());
}

/// Convert an activation layer description from C++ to Swift
ActivationKind activationLayerDescToSwift(const ActivationLayerDesc* desc) {
  switch(desc->activation) {
    case ACTIVATION_RELU:
      return ActivationKind::relu();
    case ACTIVATION_MISH:
      return ActivationKind::mish();
    case ACTIVATION_MISH_SCALE8:
      return ActivationKind::identity(); // Metal/CoreML does not use scaled mish
    case ACTIVATION_IDENTITY:
      return ActivationKind::identity();
    default:
      return ActivationKind::identity();
  }
}

/// Convert a matrix multiplication layer description from C++ to Swift
SWMatMulLayerDesc matMulLayerDescToSwift(const MatMulLayerDesc* desc) {
  return createSWMatMulLayerDesc(
    desc->inChannels,
    desc->outChannels,
    (float*)desc->weights.data());
}

/// Convert a matrix bias layer description from C++ to Swift
SWMatBiasLayerDesc matBiasLayerDescToSwift(const MatBiasLayerDesc* desc) {
  return createSWMatBiasLayerDesc(desc->numChannels, (float*)desc->weights.data());
}

/// Convert a residual block description from C++ to Swift
SWResidualBlockDesc residualBlockDescToSwift(const ResidualBlockDesc* desc) {
  SWBatchNormLayerDesc preBN = batchNormLayerDescToSwift(&desc->preBN);
  ActivationKind preActivationKind = activationLayerDescToSwift(&desc->preActivation);
  SWConvLayerDesc regularConv = convLayerDescToSwift(&desc->regularConv);
  SWBatchNormLayerDesc midBN = batchNormLayerDescToSwift(&desc->midBN);
  ActivationKind midActivationKind = activationLayerDescToSwift(&desc->midActivation);
  SWConvLayerDesc finalConv = convLayerDescToSwift(&desc->finalConv);

  return createSWResidualBlockDesc(
    preBN,
    preActivationKind,
    regularConv,
    midBN,
    midActivationKind,
    finalConv);
}

/// Convert a global pooling residual block description from C++ to Swift
SWGlobalPoolingResidualBlockDesc globalPoolingResidualBlockDescToSwift(const GlobalPoolingResidualBlockDesc* desc) {
  SWBatchNormLayerDesc preBN = batchNormLayerDescToSwift(&desc->preBN);
  ActivationKind preActivationKind = activationLayerDescToSwift(&desc->preActivation);
  SWConvLayerDesc regularConv = convLayerDescToSwift(&desc->regularConv);
  SWConvLayerDesc gpoolConv = convLayerDescToSwift(&desc->gpoolConv);
  SWBatchNormLayerDesc gpoolBN = batchNormLayerDescToSwift(&desc->gpoolBN);
  ActivationKind gpoolActivationKind = activationLayerDescToSwift(&desc->gpoolActivation);
  SWMatMulLayerDesc gpoolToBiasMul = matMulLayerDescToSwift(&desc->gpoolToBiasMul);
  SWBatchNormLayerDesc midBN = batchNormLayerDescToSwift(&desc->midBN);
  ActivationKind midActivationKind = activationLayerDescToSwift(&desc->midActivation);
  SWConvLayerDesc finalConv = convLayerDescToSwift(&desc->finalConv);

  return createSWGlobalPoolingResidualBlockDesc(
    preBN,
    preActivationKind,
    regularConv,
    gpoolConv,
    gpoolBN,
    gpoolActivationKind,
    gpoolToBiasMul,
    midBN,
    midActivationKind,
    finalConv);
}

// Forward declaration for mutual recursion
swift::Array<BlockDescriptor> residualBlocksToSwift(const vector<pair<int, unique_ptr_void>>& blocks);

/// Convert a nested bottleneck residual block description from C++ to Swift
SWNestedBottleneckResidualBlockDesc nestedBottleneckResidualBlockDescToSwift(const NestedBottleneckResidualBlockDesc* desc) {
  SWBatchNormLayerDesc preBN = batchNormLayerDescToSwift(&desc->preBN);
  ActivationKind preActivationKind = activationLayerDescToSwift(&desc->preActivation);
  SWConvLayerDesc preConv = convLayerDescToSwift(&desc->preConv);
  auto swBlocks = residualBlocksToSwift(desc->blocks);
  SWBatchNormLayerDesc postBN = batchNormLayerDescToSwift(&desc->postBN);
  ActivationKind postActivationKind = activationLayerDescToSwift(&desc->postActivation);
  SWConvLayerDesc postConv = convLayerDescToSwift(&desc->postConv);

  return createSWNestedBottleneckResidualBlockDesc(
    preBN,
    preActivationKind,
    preConv,
    swBlocks,
    postBN,
    postActivationKind,
    postConv);
}

/// Convert residual blocks from C++ to Swift
swift::Array<BlockDescriptor> residualBlocksToSwift(const vector<pair<int, unique_ptr_void>>& blocks) {
  auto builder = createBlockDescriptorBuilder();

  for(size_t i = 0; i < blocks.size(); i++) {
    void* blockDesc = blocks[i].second.get();

    if(blocks[i].first == GLOBAL_POOLING_BLOCK_KIND) {
      BlockDescriptor descriptor = globalPoolingResidualBlockDescToSwift((GlobalPoolingResidualBlockDesc*)blockDesc);
      builder.enque(descriptor);
    } else if(blocks[i].first == NESTED_BOTTLENECK_BLOCK_KIND) {
      BlockDescriptor descriptor = nestedBottleneckResidualBlockDescToSwift((NestedBottleneckResidualBlockDesc*)blockDesc);
      builder.enque(descriptor);
    } else {
      BlockDescriptor descriptor = residualBlockDescToSwift((ResidualBlockDesc*)blockDesc);
      builder.enque(descriptor);
    }
  }

  return builder.getBlockDescriptors();
}

/// Convert a SGF metadata encoder description from C++ to Swift
swift::Optional<SWSGFMetadataEncoderDesc> sGFMetadataEncoderDescToSwift(const SGFMetadataEncoderDesc* desc) {
  SWMatMulLayerDesc mul1 = matMulLayerDescToSwift(&desc->mul1);
  SWMatBiasLayerDesc bias1 = matBiasLayerDescToSwift(&desc->bias1);
  ActivationKind act1 = activationLayerDescToSwift(&desc->act1);
  SWMatMulLayerDesc mul2 = matMulLayerDescToSwift(&desc->mul2);
  SWMatBiasLayerDesc bias2 = matBiasLayerDescToSwift(&desc->bias2);
  ActivationKind act2 = activationLayerDescToSwift(&desc->act2);
  SWMatMulLayerDesc mul3 = matMulLayerDescToSwift(&desc->mul3);

  return createSWSGFMetadataEncoderDesc(
    desc->metaEncoderVersion,
    desc->numInputMetaChannels,
    mul1,
    bias1,
    act1,
    mul2,
    bias2,
    act2,
    mul3);
}

/// Convert a trunk description from C++ to Swift
SWTrunkDesc trunkDescToSwift(const TrunkDesc* trunk) {
  SWConvLayerDesc initialConv = convLayerDescToSwift(&trunk->initialConv);
  SWMatMulLayerDesc initialMatMul = matMulLayerDescToSwift(&trunk->initialMatMul);
  auto sgfMetadataEncoder = sGFMetadataEncoderDescToSwift(&trunk->sgfMetadataEncoder);
  auto swBlocks = residualBlocksToSwift(trunk->blocks);
  SWBatchNormLayerDesc trunkTipBN = batchNormLayerDescToSwift(&trunk->trunkTipBN);
  ActivationKind trunkTipActivation = activationLayerDescToSwift(&trunk->trunkTipActivation);

  return createSWTrunkDesc(
    trunk->modelVersion,
    trunk->trunkNumChannels,
    trunk->midNumChannels,
    trunk->regularNumChannels,
    trunk->gpoolNumChannels,
    initialConv,
    initialMatMul,
    sgfMetadataEncoder,
    swBlocks,
    trunkTipBN,
    trunkTipActivation);
}

/// Convert a policy head description from C++ to Swift
SWPolicyHeadDesc policyHeadDescToSwift(const PolicyHeadDesc* policyHead) {
  SWConvLayerDesc p1Conv = convLayerDescToSwift(&policyHead->p1Conv);
  SWConvLayerDesc g1Conv = convLayerDescToSwift(&policyHead->g1Conv);
  SWBatchNormLayerDesc g1BN = batchNormLayerDescToSwift(&policyHead->g1BN);
  ActivationKind g1Activation = activationLayerDescToSwift(&policyHead->g1Activation);
  SWMatMulLayerDesc gpoolToBiasMul = matMulLayerDescToSwift(&policyHead->gpoolToBiasMul);
  SWBatchNormLayerDesc p1BN = batchNormLayerDescToSwift(&policyHead->p1BN);
  ActivationKind p1Activation = activationLayerDescToSwift(&policyHead->p1Activation);
  SWConvLayerDesc p2Conv = convLayerDescToSwift(&policyHead->p2Conv);
  SWMatMulLayerDesc gpoolToPassMul = matMulLayerDescToSwift(&policyHead->gpoolToPassMul);
  SWMatBiasLayerDesc gpoolToPassBias = matBiasLayerDescToSwift(&policyHead->gpoolToPassBias);
  ActivationKind passActivation = activationLayerDescToSwift(&policyHead->passActivation);
  SWMatMulLayerDesc gpoolToPassMul2 = matMulLayerDescToSwift(&policyHead->gpoolToPassMul2);

  return createSWPolicyHeadDesc(
    policyHead->modelVersion,
    p1Conv,
    g1Conv,
    g1BN,
    g1Activation,
    gpoolToBiasMul,
    p1BN,
    p1Activation,
    p2Conv,
    gpoolToPassMul,
    gpoolToPassBias,
    passActivation,
    gpoolToPassMul2);
}

/// Convert a value head description from C++ to Swift
SWValueHeadDesc valueHeadDescToSwift(const ValueHeadDesc* valueHead) {
  SWConvLayerDesc v1Conv = convLayerDescToSwift(&valueHead->v1Conv);
  SWBatchNormLayerDesc v1BN = batchNormLayerDescToSwift(&valueHead->v1BN);
  ActivationKind v1Activation = activationLayerDescToSwift(&valueHead->v1Activation);
  SWMatMulLayerDesc v2Mul = matMulLayerDescToSwift(&valueHead->v2Mul);
  SWMatBiasLayerDesc v2Bias = matBiasLayerDescToSwift(&valueHead->v2Bias);
  ActivationKind v2Activation = activationLayerDescToSwift(&valueHead->v2Activation);
  SWMatMulLayerDesc v3Mul = matMulLayerDescToSwift(&valueHead->v3Mul);
  SWMatBiasLayerDesc v3Bias = matBiasLayerDescToSwift(&valueHead->v3Bias);
  SWMatMulLayerDesc sv3Mul = matMulLayerDescToSwift(&valueHead->sv3Mul);
  SWMatBiasLayerDesc sv3Bias = matBiasLayerDescToSwift(&valueHead->sv3Bias);
  SWConvLayerDesc vOwnershipConv = convLayerDescToSwift(&valueHead->vOwnershipConv);

  return createSWValueHeadDesc(
    valueHead->modelVersion,
    v1Conv,
    v1BN,
    v1Activation,
    v2Mul,
    v2Bias,
    v2Activation,
    v3Mul,
    v3Bias,
    sv3Mul,
    sv3Bias,
    vOwnershipConv);
}

/// Convert a model description from C++ to Swift
SWModelDesc modelDescToSwift(const ModelDesc* modelDesc) {
  return createSWModelDesc(
    modelDesc->modelVersion,
    swift::String(modelDesc->name),
    modelDesc->numInputChannels,
    modelDesc->numInputGlobalChannels,
    modelDesc->numInputMetaChannels,
    modelDesc->numValueChannels,
    modelDesc->numScoreValueChannels,
    modelDesc->numOwnershipChannels,
    modelDesc->numPolicyChannels,
    trunkDescToSwift(&modelDesc->trunk),
    policyHeadDescToSwift(&modelDesc->policyHead),
    valueHeadDescToSwift(&modelDesc->valueHead));
}

}  // namespace CoreMLProcess

//------------------------------------------------------------------------------
// LoadedModel implementation
//------------------------------------------------------------------------------

LoadedModel::LoadedModel(const string& fileName, const string& expectedSha256) {
  modelPath = fileName;
  ModelDesc::loadFromFileMaybeGZipped(fileName, modelDesc, expectedSha256);
}

//------------------------------------------------------------------------------
// NeuralNet namespace - Global functions
//------------------------------------------------------------------------------

void NeuralNet::globalInitialize() {
  // No global initialization needed for Core ML
}

void NeuralNet::globalCleanup() {
  // No cleanup needed - temp files are deleted immediately after loading
}

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

//------------------------------------------------------------------------------
// ComputeContext implementation
//------------------------------------------------------------------------------

ComputeContext::ComputeContext(int nnX, int nnY, enabled_t useFP16Mode, enabled_t useNHWCMode):
coremlContext(createCoreMLComputeContext(nnX, nnY, useFP16Mode != enabled_t::False)) {
  this->useFP16Mode = useFP16Mode;
  this->nnXLen = nnX;
  this->nnYLen = nnY;
  (void)useNHWCMode;
}

ComputeContext::~ComputeContext() {
}

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

//------------------------------------------------------------------------------
// ComputeHandle implementation
//------------------------------------------------------------------------------

static mutex computeHandleMutex;

// Helper function to convert model and create hybrid compute handle
// This is needed because Swift Optional doesn't support assignment in C++
static swift::Optional<KataGoCoreML::HybridComputeHandle> convertAndCreateHybridHandle(
  ComputeContext* context,
  const LoadedModel* loadedModel,
  bool requireExactNNLen,
  int maxBatchSize,
  int serverThreadIdx
) {
  auto coremlContext = context->coremlContext;
  int nnXLen = coremlContext.getNnXLen();
  int nnYLen = coremlContext.getNnYLen();
  bool useFP16 = (context->useFP16Mode != enabled_t::False);
  bool optimizeMask = requireExactNNLen;

  // Convert model to CoreML format in temp directory
  // The Swift side will delete the temp file after loading
  string coremlModelPath = CoreMLConversion::convertModelToTemp(
    loadedModel->modelPath,
    nnXLen,
    nnYLen,
    useFP16,
    optimizeMask,
    maxBatchSize,
    serverThreadIdx
  );

  // Convert model descriptor to Swift format for MPSGraph path
  SWModelDesc swModelDesc = CoreMLProcess::modelDescToSwift(&loadedModel->modelDesc);

  // Create hybrid compute handle (CoreML on CPU+ANE, MPSGraph on GPU)
  return createHybridComputeHandle(
    swift::String(coremlModelPath),
    swModelDesc,
    serverThreadIdx,
    requireExactNNLen,
    loadedModel->modelDesc.numInputChannels,
    loadedModel->modelDesc.numInputGlobalChannels,
    loadedModel->modelDesc.numInputMetaChannels,
    loadedModel->modelDesc.numPolicyChannels,
    loadedModel->modelDesc.numValueChannels,
    loadedModel->modelDesc.numScoreValueChannels,
    loadedModel->modelDesc.numOwnershipChannels,
    coremlContext
  );
}

// Helper function to create hybrid handle if FP16 mode with sufficient batch size, otherwise returns none
static swift::Optional<KataGoCoreML::HybridComputeHandle> createHybridHandleIfNeeded(
  ComputeContext* context,
  const LoadedModel* loadedModel,
  bool requireExactNNLen,
  int maxBatchSize,
  int serverThreadIdx
) {
  if(context->useFP16Mode == enabled_t::False) {
    // FP32 mode - don't create hybrid handle
    return swift::Optional<KataGoCoreML::HybridComputeHandle>::none();
  }

  // Hybrid mode splits batches: CoreML takes max(1, ...), MPSGraph takes remainder
  // Minimum samples for meaningful split = 1 (CoreML) + 1 (MPSGraph) = 2
  // If batch can't be split, prefer MPSGraph-only for stability
  if(maxBatchSize < MIN_COREML_BATCH + MIN_MPSGRAPH_BATCH) {
    return swift::Optional<KataGoCoreML::HybridComputeHandle>::none();
  }

  // FP16 mode with sufficient batch size: Use hybrid execution (CoreML on CPU+ANE, MPSGraph on GPU)
  return convertAndCreateHybridHandle(context, loadedModel, requireExactNNLen, maxBatchSize, serverThreadIdx);
}

// Helper function to create MPSGraph-only handle when needed
// Used when: (1) useFP16=false to avoid slow FP32 CoreML, or (2) batch too small for hybrid split
static swift::Optional<KataGoCoreML::MPSGraphModelHandle> createMPSGraphHandleIfNeeded(
  ComputeContext* context,
  const LoadedModel* loadedModel,
  bool requireExactNNLen,
  int maxBatchSize,
  int serverThreadIdx
) {
  // Use MPSGraph-only when:
  // 1. FP32 mode (CoreML FP32 on CPU+ANE is slow), OR
  // 2. Batch too small to split (hybrid requires minCoreML + minMPSGraph samples)
  bool batchTooSmallForHybrid = maxBatchSize < MIN_COREML_BATCH + MIN_MPSGRAPH_BATCH;

  if(context->useFP16Mode != enabled_t::False && !batchTooSmallForHybrid) {
    // FP16 mode with sufficient batch - hybrid handle will be created instead
    return swift::Optional<KataGoCoreML::MPSGraphModelHandle>::none();
  }

  // Log reason for MPSGraph-only mode
  if(batchTooSmallForHybrid) {
    cerr << "Core ML backend " << serverThreadIdx << ": Batch size " << maxBatchSize
         << " too small for hybrid split - using MPSGraph GPU-only" << endl;
  } else {
    cerr << "Core ML backend " << serverThreadIdx << ": FP32 mode - using MPSGraph GPU-only (skipping CoreML converter)" << endl;
  }

  // Convert model descriptor to Swift format for MPSGraph path
  // Note: No CoreML conversion needed - MPSGraph reads weights directly
  SWModelDesc swModelDesc = CoreMLProcess::modelDescToSwift(&loadedModel->modelDesc);

  // Create MPSGraph-only handle (GPU only)
  return createMPSGraphOnlyHandle(
    swModelDesc,
    serverThreadIdx,
    requireExactNNLen,
    context->coremlContext
  );
}

ComputeHandle::ComputeHandle(
  ComputeContext* context,
  const LoadedModel* loadedModel,
  bool inputsUseNHWC,
  int gpuIdx,
  int serverThreadIdx,
  bool requireExactNNLen,
  int maxBatchSize):
hybridHandle(createHybridHandleIfNeeded(context, loadedModel, requireExactNNLen, maxBatchSize, serverThreadIdx)),
mpsGraphOnlyHandle(createMPSGraphHandleIfNeeded(context, loadedModel, requireExactNNLen, maxBatchSize, serverThreadIdx)) {
  assert(((hybridHandle && !mpsGraphOnlyHandle) || (!hybridHandle && mpsGraphOnlyHandle)) &&
         "Exactly one of hybridHandle or mpsGraphOnlyHandle must be valid");

  const ModelDesc* modelDesc = &loadedModel->modelDesc;
  auto coremlContext = context->coremlContext;

  nnXLen = coremlContext.getNnXLen();
  nnYLen = coremlContext.getNnYLen();
  gpuIndex = gpuIdx;
  version = modelDesc->modelVersion;
  metaEncoderVersion = modelDesc->metaEncoderVersion;
  this->inputsUseNHWC = inputsUseNHWC;
  this->requireExactNNLen = requireExactNNLen;
  useFP16 = (context->useFP16Mode != enabled_t::False);
}

ComputeHandle::~ComputeHandle() {
}

ComputeHandle* NeuralNet::createComputeHandle(
  ComputeContext* context,
  const LoadedModel* loadedModel,
  Logger* logger,
  int maxBatchSize,
  bool requireExactNNLen,
  bool inputsUseNHWC,
  int gpuIdxForThisThread,
  int serverThreadIdx) {

  (void)logger;

  int gpuIdx = (gpuIdxForThisThread == -1) ? 0 : gpuIdxForThisThread;
  ComputeHandle* handle = nullptr;

  {
    lock_guard<mutex> lock(computeHandleMutex);
    handle = new ComputeHandle(context, loadedModel, inputsUseNHWC, gpuIdx, serverThreadIdx, requireExactNNLen, maxBatchSize);
  }

  return handle;
}

void NeuralNet::freeComputeHandle(ComputeHandle* handle) {
  delete handle;
}

bool NeuralNet::isUsingFP16(const ComputeHandle* handle) {
  return handle->useFP16;
}

//------------------------------------------------------------------------------
// Device information
//------------------------------------------------------------------------------

void NeuralNet::printDevices() {
  printCoreMLDevices();
}

//------------------------------------------------------------------------------
// InputBuffers implementation
//------------------------------------------------------------------------------

InputBuffers::InputBuffers(const LoadedModel* loadedModel, int maxBatchSz, int nnXLen, int nnYLen) {
  const ModelDesc& m = loadedModel->modelDesc;

  maxBatchSize = maxBatchSz;
  policyResultChannels = m.policyHead.p2Conv.outChannels;

  assert(((m.modelVersion < 16) || (policyResultChannels == 4)) &&
         ((m.modelVersion >= 16) || (m.modelVersion < 12) || (policyResultChannels == 2)) &&
         ((m.modelVersion >= 12) || (policyResultChannels == 1)));

  singleSpatialElts = (size_t)m.numInputChannels * nnXLen * nnYLen;
  singleInputElts = (size_t)m.numInputChannels * nnXLen * nnYLen;
  singleInputGlobalElts = (size_t)m.numInputGlobalChannels;
  singleInputMetaElts = (size_t)m.numInputMetaChannels;
  singlePolicyResultElts = (size_t)(nnXLen * nnYLen);
  singlePolicyPassResultElts = 1;
  singlePolicyProbsElts = (size_t)((nnXLen * nnYLen) + 1);
  singleValueResultElts = (size_t)m.numValueChannels;
  singleOwnershipResultElts = (size_t)m.numOwnershipChannels * nnXLen * nnYLen;
  singleOwnerMapElts = (size_t)m.numOwnershipChannels * nnXLen * nnYLen;
  singleScoreValuesResultElts = (size_t)m.numScoreValueChannels;
  singleMaskElts = (size_t)nnXLen * nnYLen;

  assert(NNModelVersion::getNumSpatialFeatures(m.modelVersion) == m.numInputChannels);
  assert(NNModelVersion::getNumGlobalFeatures(m.modelVersion) == m.numInputGlobalChannels);
  assert(singleValueResultElts == 3);

  rowSpatialBufferElts = (size_t)maxBatchSz * singleSpatialElts;
  userInputBufferElts = (size_t)maxBatchSize * singleInputElts;
  userInputGlobalBufferElts = (size_t)maxBatchSize * singleInputGlobalElts;
  userInputMetaBufferElts = (size_t)maxBatchSize * singleInputMetaElts;
  policyResultBufferElts = (size_t)maxBatchSize * singlePolicyResultElts * policyResultChannels;
  policyPassResultBufferElts = (size_t)maxBatchSize * singlePolicyPassResultElts * policyResultChannels;
  policyProbsBufferElts = (size_t)maxBatchSize * singlePolicyProbsElts * policyResultChannels;
  valueResultBufferElts = (size_t)maxBatchSize * singleValueResultElts;
  ownershipResultBufferElts = (size_t)maxBatchSize * singleOwnershipResultElts;
  ownerMapBufferElts = (size_t)maxBatchSz * singleOwnerMapElts;
  scoreValuesResultBufferElts = (size_t)maxBatchSize * singleScoreValuesResultElts;
  userInputMaskBufferElts = (size_t)maxBatchSize * singleMaskElts;

  rowSpatialBuffer = new float[rowSpatialBufferElts];
  userInputBuffer = new float[userInputBufferElts];
  memset(&userInputBuffer[0], 0, userInputBufferElts * sizeof(userInputBuffer[0]));

  userInputGlobalBuffer = new float[userInputGlobalBufferElts];
  userInputMetaBuffer = new float[userInputMetaBufferElts];
  policyResults = new float[policyResultBufferElts];
  policyPassResults = new float[policyPassResultBufferElts];
  policyProbsBuffer = new float[policyProbsBufferElts];
  valueResults = new float[valueResultBufferElts];
  ownershipResults = new float[ownershipResultBufferElts];
  ownerMapBuffer = new float[ownerMapBufferElts];
  scoreValuesResults = new float[scoreValuesResultBufferElts];
  userInputMaskBuffer = new float[userInputMaskBufferElts];
  memset(&userInputMaskBuffer[0], 0, userInputMaskBufferElts * sizeof(userInputMaskBuffer[0]));
}

InputBuffers::~InputBuffers() {
  delete[] rowSpatialBuffer;
  delete[] userInputBuffer;
  delete[] userInputGlobalBuffer;
  delete[] userInputMetaBuffer;
  delete[] policyResults;
  delete[] policyPassResults;
  delete[] policyProbsBuffer;
  delete[] valueResults;
  delete[] ownershipResults;
  delete[] ownerMapBuffer;
  delete[] scoreValuesResults;
  delete[] userInputMaskBuffer;
}

InputBuffers* NeuralNet::createInputBuffers(const LoadedModel* loadedModel, int maxBatchSize, int nnXLen, int nnYLen) {
  return new InputBuffers(loadedModel, maxBatchSize, nnXLen, nnYLen);
}

void NeuralNet::freeInputBuffers(InputBuffers* inputBuffers) {
  delete inputBuffers;
}

//------------------------------------------------------------------------------
// CoreMLProcess namespace - Helper functions
//------------------------------------------------------------------------------

void CoreMLProcess::copyRowData(float* dest, const float* src, size_t numElements) {
  copy(src, src + numElements, dest);
}

void CoreMLProcess::convertNCHW(
    float* rowSpatialInput,
    const int C,
    const int H,
    const int W,
    const bool inputsUseNHWC) {

  if ((!inputsUseNHWC) || (C <= 0) || (H <= 0) || (W <= 0)) {
    return;
  }

  const int totalSize = H * W * C;

  if (totalSize <= 1)
    return;

  const int HW = H * W;

  auto get_nchw_target_index = [C, W, HW](int nhwc_index) -> int {
    int c = nhwc_index % C;
    int temp = nhwc_index / C;
    int x = temp % W;
    int y = temp / W;
    return (c * HW) + (y * W) + x;
  };

  std::vector<bool> processed(totalSize, false);

  for (int i = 0; i < totalSize; ++i) {
    if (processed[i])
      continue;

    int target_i = get_nchw_target_index(i);

    if (target_i == i) {
      processed[i] = true;
      continue;
    }

    int current_idx = i;
    float value_in_hand = rowSpatialInput[i];

    while (true) {
      int target_idx = get_nchw_target_index(current_idx);
      float value_at_target = rowSpatialInput[target_idx];
      rowSpatialInput[target_idx] = value_in_hand;
      processed[target_idx] = true;
      value_in_hand = value_at_target;
      current_idx = target_idx;

      if (current_idx == i)
        break;
    }
  }
}

void CoreMLProcess::processRowData(size_t row, ComputeHandle* gpuHandle, InputBuffers* inputBuffers, NNResultBuf** inputBufs) {
  int nnXLen = gpuHandle->nnXLen;
  int nnYLen = gpuHandle->nnYLen;
  int numSpatialFeatures = NNModelVersion::getNumSpatialFeatures(gpuHandle->version);

  float* rowSpatialInput = &inputBuffers->userInputBuffer[inputBuffers->singleSpatialElts * row];
  float* rowGlobalInput = &inputBuffers->userInputGlobalBuffer[inputBuffers->singleInputGlobalElts * row];
  float* rowMetaInput = &inputBuffers->userInputMetaBuffer[inputBuffers->singleInputMetaElts * row];
  const float* rowGlobal = inputBufs[row]->rowGlobalBuf.data();
  const float* rowSpatial = inputBufs[row]->rowSpatialBuf.data();
  const float* rowMeta = inputBufs[row]->rowMetaBuf.data();

  CoreMLProcess::copyRowData(rowGlobalInput, rowGlobal, inputBuffers->singleInputGlobalElts);
  CoreMLProcess::copyRowData(rowMetaInput, rowMeta, inputBuffers->singleInputMetaElts);

  SymmetryHelpers::copyInputsWithSymmetry(
    rowSpatial,
    rowSpatialInput,
    1,
    nnYLen,
    nnXLen,
    numSpatialFeatures,
    gpuHandle->inputsUseNHWC,
    inputBufs[row]->symmetry);

  CoreMLProcess::convertNCHW(
    rowSpatialInput,
    numSpatialFeatures,
    nnYLen,
    nnXLen,
    gpuHandle->inputsUseNHWC);

  // Copy first channel of spatial input (mask) to dedicated mask buffer
  // After NCHW conversion, the first nnXLen*nnYLen elements are the mask channel
  float* rowMaskInput = &inputBuffers->userInputMaskBuffer[inputBuffers->singleMaskElts * row];
  copy(rowSpatialInput, rowSpatialInput + inputBuffers->singleMaskElts, rowMaskInput);
}

float CoreMLProcess::policyOptimismCalc(const double policyOptimism, const float p, const float pOpt) {
  return p + ((pOpt - p) * policyOptimism);
}

void CoreMLProcess::processOptimism(
  InputBuffers* inputBuffers,
  NNOutput* currentOutput,
  const double policyOptimism,
  size_t row) {
  auto& buffers = *inputBuffers;
  const auto singlePolicyResultElts = buffers.singlePolicyResultElts;
  float* targetBuffer = &buffers.policyProbsBuffer[row * singlePolicyResultElts];
  float* policyOutputBuf = &buffers.policyResults[row * singlePolicyResultElts * buffers.policyResultChannels];

  for(size_t i = 0; i < singlePolicyResultElts; ++i) {
    const float p = policyOutputBuf[i];
    const float pOpt = policyOutputBuf[i + singlePolicyResultElts];
    targetBuffer[i] = CoreMLProcess::policyOptimismCalc(policyOptimism, p, pOpt);
  }

  const auto p = buffers.policyPassResults[row * buffers.policyResultChannels];
  const auto pOpt = buffers.policyPassResults[row * buffers.policyResultChannels + 1];
  currentOutput->policyProbs[buffers.singlePolicyProbsElts - 1] = CoreMLProcess::policyOptimismCalc(policyOptimism, p, pOpt);
}

void CoreMLProcess::processPolicy(
  InputBuffers* inputBuffers,
  NNOutput* currentOutput,
  const ComputeHandle* gpuHandle,
  NNResultBuf* inputBuf,
  size_t row) {
  auto& buffers = *inputBuffers;
  float* targetBuffer = &buffers.policyResults[row * buffers.singlePolicyResultElts * buffers.policyResultChannels];
  const auto policyOptimism = inputBuf->policyOptimism;

  if(buffers.policyResultChannels == 1) {
    currentOutput->policyProbs[buffers.singlePolicyProbsElts - 1] =
      buffers.policyPassResults[row * buffers.policyResultChannels];
  } else {
    CoreMLProcess::processOptimism(inputBuffers, currentOutput, policyOptimism, row);
    targetBuffer = &buffers.policyProbsBuffer[row * buffers.singlePolicyResultElts];
  }

  SymmetryHelpers::copyOutputsWithSymmetry(
    targetBuffer, currentOutput->policyProbs, 1, gpuHandle->nnYLen, gpuHandle->nnXLen, inputBuf->symmetry);
}

void CoreMLProcess::processValue(
  const InputBuffers* inputBuffers,
  NNOutput* currentOutput,
  const size_t row) {
  const size_t singleValueResultElts = inputBuffers->singleValueResultElts;
  assert(singleValueResultElts == 3);
  const float* valueOutputBuf = &inputBuffers->valueResults[row * singleValueResultElts];
  currentOutput->whiteWinProb = valueOutputBuf[0];
  currentOutput->whiteLossProb = valueOutputBuf[1];
  currentOutput->whiteNoResultProb = valueOutputBuf[2];
}

void CoreMLProcess::processOwnership(
  const InputBuffers* inputBuffers,
  NNOutput* currentOutput,
  const ComputeHandle* gpuHandle,
  const int symmetry,
  const size_t row) {
  const int nnXLen = gpuHandle->nnXLen;
  const int nnYLen = gpuHandle->nnYLen;
  const size_t singleOwnershipResultElts = inputBuffers->singleOwnershipResultElts;
  const size_t ownershipOutputBufOffset = row * singleOwnershipResultElts;

  if(currentOutput->whiteOwnerMap != nullptr) {
    const float* ownershipOutputBuf = &inputBuffers->ownershipResults[ownershipOutputBufOffset];
    SymmetryHelpers::copyOutputsWithSymmetry(
      ownershipOutputBuf, currentOutput->whiteOwnerMap, 1, nnYLen, nnXLen, symmetry);
  }
}

void CoreMLProcess::processScoreValues(
  const InputBuffers* inputBuffers,
  NNOutput* currentOutput,
  const int modelVersion,
  const size_t row) {
  const size_t offset = row * inputBuffers->singleScoreValuesResultElts;
  const float* currentScoreValueData = &inputBuffers->scoreValuesResults[offset];

  if(modelVersion >= 9) {
    size_t numScoreValueChannels = inputBuffers->singleScoreValuesResultElts;
    assert(numScoreValueChannels == 6);
    currentOutput->whiteScoreMean = currentScoreValueData[0];
    currentOutput->whiteScoreMeanSq = currentScoreValueData[1];
    currentOutput->whiteLead = currentScoreValueData[2];
    currentOutput->varTimeLeft = currentScoreValueData[3];
    currentOutput->shorttermWinlossError = currentScoreValueData[4];
    currentOutput->shorttermScoreError = currentScoreValueData[5];
  }
  else if(modelVersion >= 8) {
    size_t numScoreValueChannels = inputBuffers->singleScoreValuesResultElts;
    assert(numScoreValueChannels == 4);
    currentOutput->whiteScoreMean = currentScoreValueData[0];
    currentOutput->whiteScoreMeanSq = currentScoreValueData[1];
    currentOutput->whiteLead = currentScoreValueData[2];
    currentOutput->varTimeLeft = currentScoreValueData[3];
    currentOutput->shorttermWinlossError = 0;
    currentOutput->shorttermScoreError = 0;
  }
  else if(modelVersion >= 4) {
    size_t numScoreValueChannels = inputBuffers->singleScoreValuesResultElts;
    assert(numScoreValueChannels == 2);
    currentOutput->whiteScoreMean = currentScoreValueData[0];
    currentOutput->whiteScoreMeanSq = currentScoreValueData[1];
    currentOutput->whiteLead = currentOutput->whiteScoreMean;
    currentOutput->varTimeLeft = 0;
    currentOutput->shorttermWinlossError = 0;
    currentOutput->shorttermScoreError = 0;
  }
  else {
    assert(modelVersion >= 3);
    size_t numScoreValueChannels = inputBuffers->singleScoreValuesResultElts;
    assert(numScoreValueChannels == 1);
    currentOutput->whiteScoreMean = currentScoreValueData[0];
    currentOutput->whiteScoreMeanSq = currentOutput->whiteScoreMean * currentOutput->whiteScoreMean;
    currentOutput->whiteLead = currentOutput->whiteScoreMean;
    currentOutput->varTimeLeft = 0;
    currentOutput->shorttermWinlossError = 0;
    currentOutput->shorttermScoreError = 0;
  }
}

void CoreMLProcess::processRow(
  size_t row,
  const ComputeHandle* gpuHandle,
  InputBuffers* inputBuffers,
  NNResultBuf** inputBufs,
  vector<NNOutput*>& outputs) {
  NNOutput* currentOutput = outputs[row];
  assert(currentOutput->nnXLen == gpuHandle->nnXLen);
  assert(currentOutput->nnYLen == gpuHandle->nnYLen);
  CoreMLProcess::processPolicy(inputBuffers, currentOutput, gpuHandle, inputBufs[row], row);
  CoreMLProcess::processValue(inputBuffers, currentOutput, row);
  CoreMLProcess::processOwnership(inputBuffers, currentOutput, gpuHandle, inputBufs[row]->symmetry, row);
  CoreMLProcess::processScoreValues(inputBuffers, currentOutput, gpuHandle->version, row);
}

void CoreMLProcess::getCoreMLOutput(
  ComputeHandle* gpuHandle,
  InputBuffers* inputBuffers,
  int numBatchEltsFilled,
  NNResultBuf** inputBufs,
  vector<NNOutput*>& outputs) {
  assert(numBatchEltsFilled > 0);

  int batchSize = numBatchEltsFilled;

  assert(batchSize <= inputBuffers->maxBatchSize);
  assert((NNModelVersion::getNumSpatialFeatures(gpuHandle->version) * gpuHandle->nnXLen * gpuHandle->nnYLen) <= (int)inputBuffers->singleInputElts);
  assert(NNModelVersion::getNumGlobalFeatures(gpuHandle->version) == (int)inputBuffers->singleInputGlobalElts);

  if(gpuHandle->metaEncoderVersion > 0) {
    assert(SGFMetadata::METADATA_INPUT_NUM_CHANNELS == (int)inputBuffers->singleInputMetaElts);
  }

  assert(inputBuffers->singleValueResultElts == 3);

  for(int row = 0; row < batchSize; row++) {
    CoreMLProcess::processRowData(row, gpuHandle, inputBuffers, inputBufs);
  }

  // Dispatch to appropriate handle based on mode
  if(gpuHandle->hybridHandle) {
    // FP16 mode: Use hybrid execution (CoreML on CPU+ANE, MPSGraph on GPU)
    // Mask buffer has correct stride (singleMaskElts = H*W per batch element)
    // When requireExactNNLen is true, mask operations can be optimized (optimize_identity_mask)
    gpuHandle->hybridHandle.get().apply(
      inputBuffers->userInputBuffer,
      inputBuffers->userInputGlobalBuffer,
      inputBuffers->userInputMetaBuffer,
      inputBuffers->userInputMaskBuffer,  // Dedicated mask buffer with correct stride
      inputBuffers->policyResults,
      inputBuffers->policyPassResults,
      inputBuffers->valueResults,
      inputBuffers->scoreValuesResults,
      inputBuffers->ownershipResults,
      batchSize);
  } else if(gpuHandle->mpsGraphOnlyHandle) {
    // FP32 mode: Use MPSGraph only (GPU-only)
    // Note: MPSGraphModelHandle.apply() doesn't take maskInput - it extracts from spatial input
    gpuHandle->mpsGraphOnlyHandle.get().apply(
      inputBuffers->userInputBuffer,
      inputBuffers->userInputGlobalBuffer,
      inputBuffers->userInputMetaBuffer,
      inputBuffers->policyResults,
      inputBuffers->policyPassResults,
      inputBuffers->valueResults,
      inputBuffers->scoreValuesResults,
      inputBuffers->ownershipResults,
      batchSize);
  } else {
    throw runtime_error("Core ML backend: No valid compute handle available");
  }

  for(int row = 0; row < batchSize; row++) {
    CoreMLProcess::processRow(row, gpuHandle, inputBuffers, inputBufs, outputs);
  }
}

void NeuralNet::getOutput(
  ComputeHandle* gpuHandle,
  InputBuffers* inputBuffers,
  int numBatchEltsFilled,
  NNResultBuf** inputBufs,
  vector<NNOutput*>& outputs) {

  CoreMLProcess::getCoreMLOutput(gpuHandle, inputBuffers, numBatchEltsFilled, inputBufs, outputs);
}

//------------------------------------------------------------------------------
// Test functions - not supported for Core ML backend
//------------------------------------------------------------------------------

bool NeuralNet::testEvaluateConv(
  const ConvLayerDesc* desc,
  int batchSize,
  int nnXLen,
  int nnYLen,
  bool useFP16,
  bool useNHWC,
  const vector<float>& inputBuffer,
  vector<float>& outputBuffer) {
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
  const vector<float>& inputBuffer,
  const vector<float>& maskBuffer,
  vector<float>& outputBuffer) {
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
  const vector<float>& inputBuffer,
  const vector<float>& maskBuffer,
  vector<float>& outputBuffer) {
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
  const vector<float>& inputBuffer,
  const vector<float>& maskBuffer,
  vector<float>& outputBuffer) {
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

#endif // USE_COREML_BACKEND
