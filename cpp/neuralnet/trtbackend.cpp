#ifdef USE_TENSORRT_BACKEND

#define CUDA_API_PER_THREAD_DEFAULT_STREAM
#include <NvInfer.h>
#include <cuda_runtime_api.h>

#include "../core/fileutils.h"
#include "../core/makedir.h"
#include "../core/sha2.h"
#include "../dataio/homedata.h"
#include "../neuralnet/desc.h"
#include "../neuralnet/modelversion.h"
#include "../neuralnet/nneval.h"
#include "../neuralnet/nninputs.h"
#include "../neuralnet/nninterface.h"

using namespace std;
using namespace nvinfer1;

// Define this to print out some of the intermediate values of the neural net
//#define DEBUG_INTERMEDIATE_VALUES

// Define this to use plan cache instead of timing cache, which enables instant
// initialization at the cost of excessive disk space usage
//#define CACHE_TENSORRT_PLAN

static void checkCudaError(const cudaError_t status, const char* opName, const char* file, const char* func, int line) {
  if(status != cudaSuccess)
    throw StringError(
      string("CUDA Error, for ") + opName + " file " + file + ", func " + func + ", line " + Global::intToString(line) +
      ", error " + cudaGetErrorString(status));
}
#define CUDA_ERR(opName, x) \
  { checkCudaError((x), opName, __FILE__, #x, __LINE__); }

void NeuralNet::globalInitialize() {
  // Empty for TensorRT backend
}

void NeuralNet::globalCleanup() {
  // Empty for TensorRT backend
}

struct ComputeContext {
  int nnXLen;
  int nnYLen;
  enabled_t useFP16Mode;
  string homeDataDirOverride;
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
  (void)openCLReTunePerBoardSize;
  (void)loadedModel;

  if(useNHWCMode == enabled_t::True) {
    throw StringError("TensorRT backend: useNHWC = false required, other configurations not supported");
  }

  ComputeContext* context = new ComputeContext();
  context->nnXLen = nnXLen;
  context->nnYLen = nnYLen;
  context->useFP16Mode = useFP16Mode;
  context->homeDataDirOverride = homeDataDirOverride;
  return context;
}

void NeuralNet::freeComputeContext(ComputeContext* computeContext) {
  delete computeContext;
}

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

ModelPostProcessParams NeuralNet::getPostProcessParams(const LoadedModel* loadedModel) {
  return loadedModel->modelDesc.postProcessParams;
}

struct TRTModel {
  int nnXLen;
  int nnYLen;
  int maxBatchSize;
  bool requireExactNNLen;

  // TensorRT keeps only reference to weights before engine is built
  const LoadedModel* rawModel;
  vector<unique_ptr<float[]>> extraWeights;

  int version;
  uint8_t tuneHash[32];
  IOptimizationProfile* profile;
  unique_ptr<INetworkDefinition> network;
  vector<pair<string, string>> debugOutputs;

  TRTModel() = default;
  TRTModel(TRTModel&&) = default;
  TRTModel(const TRTModel&) = delete;
  TRTModel& operator=(TRTModel&&) = default;
  TRTModel& operator=(const TRTModel&) = delete;
};

struct ModelParser {
  unique_ptr<TRTModel> model;

  ITensor* inputMask;
  ITensor* inputFeature;
  ITensor* inputGlobalFeature;

  ILayer* maskSumLayer;
  ILayer* maskScaleLayer;
  ILayer* maskQuadLayer;

  string tuneDesc;  // Serves as a hash of the network architecture specific to tuning

  ModelParser() = default;
  ModelParser(const ModelParser&) = delete;
  ModelParser& operator=(const ModelParser&) = delete;

  static constexpr int tuneSalt = 3;  // Bump this when between katago versions we want to forcibly drop old timing caches and plan caches.

  unique_ptr<TRTModel> build(
    unique_ptr<INetworkDefinition> net,
    IOptimizationProfile* profile,
    const LoadedModel* rawModel,
    int nnXLen,
    int nnYLen,
    int maxBatchSize,
    bool requireExactNNLen) {
    model = make_unique<TRTModel>();

    model->nnXLen = nnXLen;
    model->nnYLen = nnYLen;
    model->profile = profile;
    model->network = move(net);
    model->rawModel = rawModel;
    model->maxBatchSize = maxBatchSize;
    model->requireExactNNLen = requireExactNNLen;

    auto& network = model->network;
    auto modelDesc = &model->rawModel->modelDesc;

    tuneDesc = Global::strprintf(
      R"|("salt"(%d)"model"(%d,%d,%d,%d,%d,%d))|",
      tuneSalt,
      modelDesc->version,
      modelDesc->numInputChannels,
      modelDesc->numInputGlobalChannels,
      modelDesc->numValueChannels,
      modelDesc->numScoreValueChannels,
      modelDesc->numOwnershipChannels);

    model->version = modelDesc->version;
    network->setName(modelDesc->name.c_str());

    initInputs();
    initMaskProcLayers();

    auto trunk = buildTrunk(&modelDesc->trunk);
    buildPolicyHead(trunk->getOutput(0), &modelDesc->policyHead);
    buildValueHead(trunk->getOutput(0), &modelDesc->valueHead);

    SHA2::get256(tuneDesc.c_str(), model->tuneHash);

    return move(model);
  }

  void markDebugOutput(ITensor* tensor, const string& description, bool force2D = false) {
#ifdef DEBUG_INTERMEDIATE_VALUES
    auto& network = model->network;
    ILayer* debugOutputLayer = nullptr;
    if(force2D) {
      auto layer = network->addShuffle(*tensor);
      layer->setReshapeDimensions({2, {0, -1}});
      debugOutputLayer = layer;
    } else {
      debugOutputLayer = network->addIdentity(*tensor);
    }
    debugOutputLayer->setOutputType(0, DataType::kFLOAT);
    string debugOutputName = "DBG" + to_string(hash<string>{}(description));
    auto debugOutput = debugOutputLayer->getOutput(0);
    network->markOutput(*debugOutput);
    debugOutput->setName(debugOutputName.c_str());
    debugOutput->setType(DataType::kFLOAT);
    debugOutput->setAllowedFormats(1U << static_cast<int>(TensorFormat::kLINEAR));
    model->debugOutputs.push_back(pair<string, string>(debugOutputName, description));
#else
    (void)tensor;
    (void)description;
    (void)force2D;
#endif
  }

  void initInputs() {
    auto profile = model->profile;
    auto& network = model->network;
    auto modelDesc = &model->rawModel->modelDesc;

    int nnXLen = model->nnXLen;
    int nnYLen = model->nnYLen;
    int numInputChannels = modelDesc->numInputChannels;
    int numInputGlobalChannels = modelDesc->numInputGlobalChannels;

    int numFeatures = NNModelVersion::getNumSpatialFeatures(model->version);
    if(numInputChannels != numFeatures)
      throw StringError(Global::strprintf(
        "Neural net numInputChannels (%d) was not the expected number based on version (%d)",
        numInputChannels,
        numFeatures));
    int numGlobalFeatures = NNModelVersion::getNumGlobalFeatures(model->version);
    if(numInputGlobalChannels != numGlobalFeatures)
      throw StringError(Global::strprintf(
        "Neural net numInputGlobalChannels (%d) was not the expected number based on version (%d)",
        numInputGlobalChannels,
        numGlobalFeatures));

    if(nnXLen > NNPos::MAX_BOARD_LEN)
      throw StringError(
        Global::strprintf("nnXLen (%d) is greater than NNPos::MAX_BOARD_LEN (%d)", nnXLen, NNPos::MAX_BOARD_LEN));
    if(nnYLen > NNPos::MAX_BOARD_LEN)
      throw StringError(
        Global::strprintf("nnYLen (%d) is greater than NNPos::MAX_BOARD_LEN (%d)", nnYLen, NNPos::MAX_BOARD_LEN));

    inputMask = network->addInput("InputMask", DataType::kFLOAT, {4, {-1, 1, nnYLen, nnXLen}});
    inputMask->setAllowedFormats(1U << static_cast<int>(TensorFormat::kLINEAR));
    profile->setDimensions("InputMask", OptProfileSelector::kMIN, Dims4(1, 1, nnYLen, nnXLen));
    profile->setDimensions("InputMask", OptProfileSelector::kOPT, Dims4(model->maxBatchSize, 1, nnYLen, nnXLen));
    profile->setDimensions("InputMask", OptProfileSelector::kMAX, Dims4(model->maxBatchSize, 1, nnYLen, nnXLen));

    inputFeature = network->addInput("InputFeature", DataType::kFLOAT, {4, {-1, numInputChannels, nnYLen, nnXLen}});
    inputFeature->setAllowedFormats(1U << static_cast<int>(TensorFormat::kLINEAR));
    profile->setDimensions("InputFeature", OptProfileSelector::kMIN, Dims4(1, numInputChannels, nnYLen, nnXLen));
    profile->setDimensions(
      "InputFeature", OptProfileSelector::kOPT, Dims4(model->maxBatchSize, numInputChannels, nnYLen, nnXLen));
    profile->setDimensions(
      "InputFeature", OptProfileSelector::kMAX, Dims4(model->maxBatchSize, numInputChannels, nnYLen, nnXLen));

    inputGlobalFeature =
      network->addInput("InputGlobalFeature", DataType::kFLOAT, {4, {-1, numInputGlobalChannels, 1, 1}});
    inputFeature->setAllowedFormats(1U << static_cast<int>(TensorFormat::kLINEAR));
    profile->setDimensions("InputGlobalFeature", OptProfileSelector::kMIN, Dims4(1, numInputGlobalChannels, 1, 1));
    profile->setDimensions(
      "InputGlobalFeature", OptProfileSelector::kOPT, Dims4(model->maxBatchSize, numInputGlobalChannels, 1, 1));
    profile->setDimensions(
      "InputGlobalFeature", OptProfileSelector::kMAX, Dims4(model->maxBatchSize, numInputGlobalChannels, 1, 1));

    markDebugOutput(inputFeature, "Initial bin features");
  }

  void initMaskProcLayers() {
    int nnXLen = model->nnXLen;
    int nnYLen = model->nnYLen;
    auto& network = model->network;

    if(!model->requireExactNNLen) {
      maskSumLayer = network->addReduce(*inputMask, ReduceOperation::kSUM, 1U << 2 | 1U << 3, true);
      maskSumLayer->setName("InputMask/sum");
      maskSumLayer->setPrecision(DataType::kFLOAT);

      auto maskWidthLayer = network->addUnary(*maskSumLayer->getOutput(0), UnaryOperation::kSQRT);
      maskWidthLayer->setName("InputMask/width");
      maskWidthLayer->setPrecision(DataType::kFLOAT);

      auto maskScaleWeightsShift = make_unique<float[]>(1);
      auto maskScaleWeightsScale = make_unique<float[]>(1);
      maskScaleWeightsShift[0] = -1.4f;
      maskScaleWeightsScale[0] = 0.1f;
      maskScaleLayer = network->addScale(
        *maskWidthLayer->getOutput(0),
        ScaleMode::kUNIFORM,
        {DataType::kFLOAT, maskScaleWeightsShift.get(), 1},
        {DataType::kFLOAT, maskScaleWeightsScale.get(), 1},
        {DataType::kFLOAT, nullptr, 0});
      maskScaleLayer->setName("InputMask/scale");
      maskScaleLayer->setPrecision(DataType::kFLOAT);
      model->extraWeights.push_back(move(maskScaleWeightsShift));
      model->extraWeights.push_back(move(maskScaleWeightsScale));

      auto maskCenterSquareWeightsShift = make_unique<float[]>(1);
      auto maskCenterSquareWeightsPower = make_unique<float[]>(1);
      maskCenterSquareWeightsShift[0] = -14.0f;
      maskCenterSquareWeightsPower[0] = 2.0f;
      auto maskCenterSquareLayer = network->addScale(
        *maskWidthLayer->getOutput(0),
        ScaleMode::kUNIFORM,
        {DataType::kFLOAT, maskCenterSquareWeightsShift.get(), 1},
        {DataType::kFLOAT, nullptr, 0},
        {DataType::kFLOAT, maskCenterSquareWeightsPower.get(), 1});
      maskCenterSquareLayer->setName("InputMask/centersquare");
      maskCenterSquareLayer->setPrecision(DataType::kFLOAT);
      model->extraWeights.push_back(move(maskCenterSquareWeightsShift));
      model->extraWeights.push_back(move(maskCenterSquareWeightsPower));

      auto maskQuadWeightsShift = make_unique<float[]>(1);
      auto maskQuadWeightsScale = make_unique<float[]>(1);
      maskQuadWeightsShift[0] = -0.1f;
      maskQuadWeightsScale[0] = 0.01f;
      maskQuadLayer = network->addScale(
        *maskCenterSquareLayer->getOutput(0),
        ScaleMode::kUNIFORM,
        {DataType::kFLOAT, maskQuadWeightsShift.get(), 1},
        {DataType::kFLOAT, maskQuadWeightsScale.get(), 1},
        {DataType::kFLOAT, nullptr, 0});
      maskQuadLayer->setName("InputMask/quad");
      maskQuadLayer->setPrecision(DataType::kFLOAT);
      model->extraWeights.push_back(move(maskQuadWeightsShift));
      model->extraWeights.push_back(move(maskQuadWeightsScale));
    } else {
      float maskWidth = sqrtf(nnXLen * nnYLen);

      auto maskScaleLayerWeights = make_unique<float[]>(1);
      maskScaleLayerWeights[0] = maskWidth * 0.1f - 1.4f;
      maskScaleLayer = network->addConstant({4, {1, 1, 1, 1}}, {DataType::kFLOAT, maskScaleLayerWeights.get(), 1});
      maskScaleLayer->setName("InputMask/scale");
      model->extraWeights.push_back(move(maskScaleLayerWeights));

      auto maskQuadLayerWeights = make_unique<float[]>(1);
      maskQuadLayerWeights[0] = (maskWidth - 14.0f) * (maskWidth - 14.0f) * 0.01f - 0.1f;
      maskQuadLayer = network->addConstant({4, {1, 1, 1, 1}}, {DataType::kFLOAT, maskQuadLayerWeights.get(), 1});
      maskQuadLayer->setName("InputMask/quad");
      model->extraWeights.push_back(move(maskQuadLayerWeights));
    }
  }

  ILayer* buildTrunk(const TrunkDesc* desc) {
    auto& network = model->network;

    string name = desc->name;
    int numChannels = desc->trunkNumChannels;

    tuneDesc += Global::strprintf(
      R"|("%s"(%d,%d,%d,%d,%d))|",
      desc->name.c_str(),
      desc->numBlocks,
      desc->trunkNumChannels,
      desc->midNumChannels,
      desc->regularNumChannels,
      desc->gpoolNumChannels);

    auto initialConvLayer = buildConvLayer(inputFeature, &desc->initialConv);
    auto initialMatMulLayer = buildMatMulLayer(inputGlobalFeature, &desc->initialMatMul);

    auto initialConv = initialConvLayer->getOutput(0);
    auto initialMatMul = initialMatMulLayer->getOutput(0);

    assert(initialConv->getDimensions().d[1] == numChannels);
    assert(initialMatMul->getDimensions().d[1] == numChannels);

    markDebugOutput(initialConvLayer->getOutput(0), "After initial conv");

    auto initialBiasLayer = network->addElementWise(*initialConv, *initialMatMul, ElementWiseOperation::kSUM);
    auto initiaBiasLayerName = name + "/initbias";
    initialBiasLayer->setName(initiaBiasLayerName.c_str());

    assert(desc->blocks.size() == desc->numBlocks);
    auto trunkScratchLayer = buildResidualBlockStack(initialBiasLayer->getOutput(0), desc->blocks, "trunk");

    auto trunkTipBatchNormLayer = buildBatchNormLayer(trunkScratchLayer->getOutput(0), &desc->trunkTipBN);
    auto trunkTipActivationLayer =
      buildActivationLayer(trunkTipBatchNormLayer->getOutput(0), &desc->trunkTipActivation);
    auto trunkTipMaskLayer = applyMaskLayer(trunkTipActivationLayer);

    markDebugOutput(trunkTipMaskLayer->getOutput(0), "Trunk tip");

    return trunkTipMaskLayer;
  }

  ILayer* buildResidualBlockStack(
    ITensor* input,
    const std::vector<std::pair<int, unique_ptr_void>>& blocks,
    const string& name) {
    ILayer* trunkScratchLayer = model->network->addIdentity(*input);
    auto trunkScratchLayerName = name + "/scratch";
    trunkScratchLayer->setName(trunkScratchLayerName.c_str());

    for(int i = 0; i < blocks.size(); i++) {
      markDebugOutput(trunkScratchLayer->getOutput(0), name + " before block " + to_string(i));
      if(blocks[i].first == ORDINARY_BLOCK_KIND) {
        auto blockDesc = static_cast<ResidualBlockDesc*>(blocks[i].second.get());
        trunkScratchLayer = buildResidualBlock(trunkScratchLayer->getOutput(0), blockDesc);
      } else if(blocks[i].first == GLOBAL_POOLING_BLOCK_KIND) {
        auto blockDesc = static_cast<GlobalPoolingResidualBlockDesc*>(blocks[i].second.get());
        trunkScratchLayer = buildGlobalPoolingResidualBlock(trunkScratchLayer->getOutput(0), blockDesc);
      } else if(blocks[i].first == NESTED_BOTTLENECK_BLOCK_KIND) {
        auto blockDesc = static_cast<NestedBottleneckResidualBlockDesc*>(blocks[i].second.get());
        trunkScratchLayer = buildNestedBottleneckResidualBlock(trunkScratchLayer->getOutput(0), blockDesc);
      } else {
        ASSERT_UNREACHABLE;
      }
    }

    return trunkScratchLayer;
  }

  void buildPolicyHead(ITensor* input, const PolicyHeadDesc* desc) {
    auto& network = model->network;
    string name = desc->name;

    auto p1ConvLayer = buildConvLayer(input, &desc->p1Conv);
    auto g1ConvLayer = buildConvLayer(input, &desc->g1Conv);
    auto g1BatchNormLayer = buildBatchNormLayer(g1ConvLayer->getOutput(0), &desc->g1BN);
    auto g1ActivationLayer = buildActivationLayer(g1BatchNormLayer->getOutput(0), &desc->g1Activation);
    auto g1MaskLayer = applyMaskLayer(g1ActivationLayer);
    auto g1CastLayer = applyCastLayer(g1MaskLayer, DataType::kFLOAT);
    auto gpoolLayer = applyGPoolLayer(g1CastLayer, true);
    auto gpoolToBiasMulLayer = buildMatMulLayer(gpoolLayer->getOutput(0), &desc->gpoolToBiasMul, true);
    auto p1CastLayer = applyCastLayer(p1ConvLayer, DataType::kFLOAT);
    auto gpoolBiasLayer = network->addElementWise(
      *p1CastLayer->getOutput(0), *gpoolToBiasMulLayer->getOutput(0), ElementWiseOperation::kSUM);
    auto gpoolBiasLayerName = name + "/gpbias";
    gpoolBiasLayer->setName(gpoolBiasLayerName.c_str());
    gpoolBiasLayer->setPrecision(DataType::kFLOAT);
    auto p1BatchNormLayer = buildBatchNormLayer(gpoolBiasLayer->getOutput(0), &desc->p1BN, true);
    auto p1ActivationLayer = buildActivationLayer(p1BatchNormLayer->getOutput(0), &desc->p1Activation, true);
    auto p1MaskLayer = applyMaskLayer(p1ActivationLayer, true);

    markDebugOutput(p1ConvLayer->getOutput(0), "p1 pre-gpool-sum");
    markDebugOutput(g1ConvLayer->getOutput(0), "g1 pre-gpool");
    markDebugOutput(gpoolLayer->getOutput(0), "g1 pooled", true);
    markDebugOutput(gpoolToBiasMulLayer->getOutput(0), "g1 biases", true);
    markDebugOutput(gpoolBiasLayer->getOutput(0), "p1 after-gpool-sum");

    // So that mask layer can be omitted
    assert(desc->p2Conv.convXSize == 1);
    assert(desc->p2Conv.convYSize == 1);

    auto p2ConvLayer = buildConvLayer(p1MaskLayer->getOutput(0), &desc->p2Conv, true);
    p2ConvLayer->setPrecision(DataType::kFLOAT);
    auto gpoolToPassMulLayer = buildMatMulLayer(gpoolLayer->getOutput(0), &desc->gpoolToPassMul, true);
    gpoolToPassMulLayer->setPrecision(DataType::kFLOAT);

    auto outputPolicyPass = gpoolToPassMulLayer->getOutput(0);
    network->markOutput(*outputPolicyPass);
    outputPolicyPass->setName("OutputPolicyPass");
    outputPolicyPass->setType(DataType::kFLOAT);
    outputPolicyPass->setAllowedFormats(1U << static_cast<int>(TensorFormat::kLINEAR));

    auto outputPolicy = p2ConvLayer->getOutput(0);
    network->markOutput(*outputPolicy);
    outputPolicy->setName("OutputPolicy");
    outputPolicy->setType(DataType::kFLOAT);
    outputPolicy->setAllowedFormats(1U << static_cast<int>(TensorFormat::kLINEAR));
  }

  void buildValueHead(ITensor* input, const ValueHeadDesc* desc) {
    auto& network = model->network;

    auto v1ConvLayer = buildConvLayer(input, &desc->v1Conv);
    auto v1BatchNormLayer = buildBatchNormLayer(v1ConvLayer->getOutput(0), &desc->v1BN);
    auto v1ActivationLayer = buildActivationLayer(v1BatchNormLayer->getOutput(0), &desc->v1Activation);
    auto v1MaskLayer = applyMaskLayer(v1ActivationLayer);
    auto v1CastLayer = applyCastLayer(v1MaskLayer, DataType::kFLOAT);

    markDebugOutput(v1ConvLayer->getOutput(0), "v1");

    auto gpoolLayer = applyGPoolLayer(v1CastLayer, true, true);
    auto v2MulLayer = buildMatMulLayer(gpoolLayer->getOutput(0), &desc->v2Mul, true);
    auto v2BiasLayer = buildMatBiasLayer(v2MulLayer->getOutput(0), &desc->v2Bias, true);
    auto v2ActivationLayer = buildActivationLayer(v2BiasLayer->getOutput(0), &desc->v2Activation, true);

    markDebugOutput(gpoolLayer->getOutput(0), "v1 pooled", true);
    markDebugOutput(v2ActivationLayer->getOutput(0), "v2", true);

    auto v3MulLayer = buildMatMulLayer(v2ActivationLayer->getOutput(0), &desc->v3Mul, true);
    auto v3BiasLayer = buildMatBiasLayer(v3MulLayer->getOutput(0), &desc->v3Bias, true);

    auto sv3MulLayer = buildMatMulLayer(v2ActivationLayer->getOutput(0), &desc->sv3Mul, true);
    auto sv3BiasLayer = buildMatBiasLayer(sv3MulLayer->getOutput(0), &desc->sv3Bias, true);

    // So that mask layer can be omitted
    assert(desc->vOwnershipConv.convXSize == 1);
    assert(desc->vOwnershipConv.convYSize == 1);

    auto vOwnershipConvLayer = buildConvLayer(v1MaskLayer->getOutput(0), &desc->vOwnershipConv);
    auto vOwnershipCastLayer = applyCastLayer(vOwnershipConvLayer, DataType::kFLOAT);

    auto outputValue = v3BiasLayer->getOutput(0);
    network->markOutput(*outputValue);
    outputValue->setName("OutputValue");
    outputValue->setType(DataType::kFLOAT);
    outputValue->setAllowedFormats(1U << static_cast<int>(TensorFormat::kLINEAR));

    auto outputScoreValue = sv3BiasLayer->getOutput(0);
    network->markOutput(*outputScoreValue);
    outputScoreValue->setName("OutputScoreValue");
    outputScoreValue->setType(DataType::kFLOAT);
    outputScoreValue->setAllowedFormats(1U << static_cast<int>(TensorFormat::kLINEAR));

    auto outputOwnership = vOwnershipCastLayer->getOutput(0);
    network->markOutput(*outputOwnership);
    outputOwnership->setName("OutputOwnership");
    outputOwnership->setType(DataType::kFLOAT);
    outputOwnership->setAllowedFormats(1U << static_cast<int>(TensorFormat::kLINEAR));

    auto modelDesc = &model->rawModel->modelDesc;
    assert(outputValue->getDimensions().d[1] == modelDesc->numValueChannels);
    assert(outputScoreValue->getDimensions().d[1] == modelDesc->numScoreValueChannels);
    assert(outputOwnership->getDimensions().d[1] == modelDesc->numOwnershipChannels);
  }

  ILayer* buildResidualBlock(ITensor* input, const ResidualBlockDesc* desc) {
    auto preBatchNormLayer = buildBatchNormLayer(input, &desc->preBN);
    auto preActivationLayer = buildActivationLayer(preBatchNormLayer->getOutput(0), &desc->preActivation);
    auto preMaskLayer = applyMaskLayer(preActivationLayer);
    auto regularConvLayer = buildConvLayer(preMaskLayer->getOutput(0), &desc->regularConv);
    auto midBatchNormLayer = buildBatchNormLayer(regularConvLayer->getOutput(0), &desc->midBN);
    auto midActivationLayer = buildActivationLayer(midBatchNormLayer->getOutput(0), &desc->midActivation);
    auto midMaskLayer = applyMaskLayer(midActivationLayer);
    auto finalConvLayer = buildConvLayer(midMaskLayer->getOutput(0), &desc->finalConv);

    auto mergeLayer = model->network->addElementWise(*input, *finalConvLayer->getOutput(0), ElementWiseOperation::kSUM);
    mergeLayer->setName(desc->name.c_str());

    return mergeLayer;
  }

  ILayer* buildGlobalPoolingResidualBlock(ITensor* input, const GlobalPoolingResidualBlockDesc* desc) {
    auto& network = model->network;
    string name = desc->name;

    auto preBatchNormLayer = buildBatchNormLayer(input, &desc->preBN);
    auto preActivationLayer = buildActivationLayer(preBatchNormLayer->getOutput(0), &desc->preActivation);
    auto preMaskLayer = applyMaskLayer(preActivationLayer);

    auto regularConvLayer = buildConvLayer(preMaskLayer->getOutput(0), &desc->regularConv);
    auto gpoolConvLayer = buildConvLayer(preMaskLayer->getOutput(0), &desc->gpoolConv);
    auto gpoolBatchNormLayer = buildBatchNormLayer(gpoolConvLayer->getOutput(0), &desc->gpoolBN);
    auto gpoolActivationLayer = buildActivationLayer(gpoolBatchNormLayer->getOutput(0), &desc->gpoolActivation);
    auto gpoolMaskLayer = applyMaskLayer(gpoolActivationLayer);
    auto gpoolLayer = applyGPoolLayer(gpoolMaskLayer);
    auto gpoolToBiasMulLayer = buildMatMulLayer(gpoolLayer->getOutput(0), &desc->gpoolToBiasMul);
    auto gpoolBiasLayer = network->addElementWise(
      *regularConvLayer->getOutput(0), *gpoolToBiasMulLayer->getOutput(0), ElementWiseOperation::kSUM);
    auto gpoolBiasLayerName = name + "/gpbias";
    gpoolBiasLayer->setName(gpoolBiasLayerName.c_str());

    auto midBatchNormLayer = buildBatchNormLayer(gpoolBiasLayer->getOutput(0), &desc->midBN);
    auto midActivationLayer = buildActivationLayer(midBatchNormLayer->getOutput(0), &desc->midActivation);
    auto midMaskLayer = applyMaskLayer(midActivationLayer);

    auto finalConvLayer = buildConvLayer(midMaskLayer->getOutput(0), &desc->finalConv);

    auto mergeLayer = network->addElementWise(*input, *finalConvLayer->getOutput(0), ElementWiseOperation::kSUM);
    mergeLayer->setName(name.c_str());

    return mergeLayer;
  }

  ILayer* buildNestedBottleneckResidualBlock(ITensor* input, const NestedBottleneckResidualBlockDesc* desc) {
    assert(desc->blocks.size() == desc->numBlocks);

    auto preBatchNormLayer = buildBatchNormLayer(input, &desc->preBN);
    auto preActivationLayer = buildActivationLayer(preBatchNormLayer->getOutput(0), &desc->preActivation);
    auto preMaskLayer = applyMaskLayer(preActivationLayer);
    auto preConvLayer = buildConvLayer(preMaskLayer->getOutput(0), &desc->preConv);
    auto stackLayer = buildResidualBlockStack(preConvLayer->getOutput(0), desc->blocks, desc->name);
    auto postBatchNormLayer = buildBatchNormLayer(stackLayer->getOutput(0), &desc->postBN);
    auto postActivationLayer = buildActivationLayer(postBatchNormLayer->getOutput(0), &desc->postActivation);
    auto postMaskLayer = applyMaskLayer(postActivationLayer);
    auto postConvLayer = buildConvLayer(postMaskLayer->getOutput(0), &desc->postConv);

    auto mergeLayer = model->network->addElementWise(*input, *postConvLayer->getOutput(0), ElementWiseOperation::kSUM);
    mergeLayer->setName(desc->name.c_str());

    return mergeLayer;
  }

  ILayer* buildMatMulLayer(ITensor* input, const MatMulLayerDesc* desc, bool forceFP32 = false) {
    int numInChannels = desc->inChannels;
    int numOutChannels = desc->outChannels;

    tuneDesc += Global::strprintf(R"|("%s"(%d,%d))|", desc->name.c_str(), desc->inChannels, desc->outChannels);

    assert(desc->weights.size() == numInChannels * numOutChannels);
    assert(input->getDimensions().d[1] == numInChannels);

    // Transpose from model's CK to TensorRT's KC
    auto transposedWeights = make_unique<float[]>(desc->weights.size());
    for(int ic = 0; ic < numInChannels; ic++) {
      for(int oc = 0; oc < numOutChannels; oc++) {
        transposedWeights[oc * numInChannels + ic] = desc->weights[ic * numOutChannels + oc];
      }
    }

    // For convenience, both I/O tensors have 3 dimentions (in addition to batch), so that
    // matmul is mathmatically equivalent to a 2D convolution of 1x1 features and 1x1 kernels.
    auto matMulLayer = model->network->addConvolutionNd(
      *input,
      desc->outChannels,
      {2, {1, 1}},
      {DataType::kFLOAT, transposedWeights.get(), static_cast<int64_t>(desc->weights.size())},
      {DataType::kFLOAT, nullptr, 0});
    matMulLayer->setName(desc->name.c_str());

    if(forceFP32) {
      matMulLayer->setPrecision(DataType::kFLOAT);
    }

    model->extraWeights.push_back(move(transposedWeights));

    return matMulLayer;
  }

  ILayer* buildMatBiasLayer(ITensor* input, const MatBiasLayerDesc* desc, bool forceFP32 = false) {
    int numChannels = desc->numChannels;

    tuneDesc += Global::strprintf(R"|("%s"(%d))|", desc->name.c_str(), desc->numChannels);

    assert(desc->weights.size() == numChannels);
    assert(input->getDimensions().d[1] == numChannels);

    auto matBiasLayer = model->network->addScale(
      *input,
      ScaleMode::kCHANNEL,
      {DataType::kFLOAT, desc->weights.data(), static_cast<int64_t>(numChannels)},
      {DataType::kFLOAT, nullptr, 0},
      {DataType::kFLOAT, nullptr, 0});
    matBiasLayer->setName(desc->name.c_str());

    if(forceFP32) {
      matBiasLayer->setPrecision(DataType::kFLOAT);
    }

    return matBiasLayer;
  }

  ILayer* buildConvLayer(ITensor* input, const ConvLayerDesc* desc, bool forceFP32 = false) {
    int convXSize = desc->convXSize;
    int convYSize = desc->convYSize;
    int dilationX = desc->dilationX;
    int dilationY = desc->dilationY;
    int numInChannels = desc->inChannels;
    int numOutChannels = desc->outChannels;

    tuneDesc += Global::strprintf(
      R"|("%s"(%d,%d,%d,%d,%d,%d))|",
      desc->name.c_str(),
      desc->convXSize,
      desc->convYSize,
      desc->inChannels,
      desc->outChannels,
      desc->dilationX,
      desc->dilationY);

    assert(desc->weights.size() == convYSize * convXSize * numInChannels * numOutChannels);
    assert(input->getDimensions().d[1] == numInChannels);

    auto convLayer = model->network->addConvolutionNd(
      *input,
      desc->outChannels,
      {2, {convYSize, convXSize}},
      {DataType::kFLOAT, desc->weights.data(), static_cast<int64_t>(desc->weights.size())},
      {DataType::kFLOAT, nullptr, 0});
    convLayer->setDilationNd({2, {dilationY, dilationX}});
    convLayer->setPaddingMode(PaddingMode::kSAME_UPPER);
    convLayer->setName(desc->name.c_str());

    if(forceFP32) {
      convLayer->setPrecision(DataType::kFLOAT);
    }

    return convLayer;
  }

  ILayer* buildBatchNormLayer(ITensor* input, const BatchNormLayerDesc* desc, bool forceFP32 = false) {
    int numChannels = desc->numChannels;
    float epsilon = desc->epsilon;

    tuneDesc += Global::strprintf(R"|("%s"(%d))|", desc->name.c_str(), desc->numChannels);

    assert(desc->mean.size() == numChannels);
    assert(desc->variance.size() == numChannels);
    assert(desc->scale.size() == numChannels);
    assert(desc->bias.size() == numChannels);
    assert(input->getDimensions().d[1] == numChannels);

    auto mergedScale = make_unique<float[]>(numChannels);
    auto mergedBias = make_unique<float[]>(numChannels);
    for(int i = 0; i < numChannels; i++) {
      mergedScale[i] = desc->scale[i] / sqrtf(desc->variance[i] + epsilon);
      mergedBias[i] = desc->bias[i] - mergedScale[i] * desc->mean[i];
    }

    auto bnLayer = model->network->addScale(
      *input,
      ScaleMode::kCHANNEL,
      {DataType::kFLOAT, mergedBias.get(), static_cast<int64_t>(numChannels)},
      {DataType::kFLOAT, mergedScale.get(), static_cast<int64_t>(numChannels)},
      {DataType::kFLOAT, nullptr, 0});
    bnLayer->setName(desc->name.c_str());

    if(forceFP32) {
      bnLayer->setPrecision(DataType::kFLOAT);
    }

    model->extraWeights.push_back(move(mergedScale));
    model->extraWeights.push_back(move(mergedBias));

    return bnLayer;
  }

  ILayer* buildActivationLayer(ITensor* input, const ActivationLayerDesc* desc, bool forceFP32 = false) {
    tuneDesc += Global::strprintf(R"|("%s"(%d))|", desc->name.c_str(), desc->activation);
    if(desc->activation == ACTIVATION_IDENTITY) {
      auto activationLayer = model->network->addIdentity(*input);
      activationLayer->setName(desc->name.c_str());
      if(forceFP32) {
        activationLayer->setPrecision(DataType::kFLOAT);
      }
      return activationLayer;
    } else if(desc->activation == ACTIVATION_RELU) {
      auto activationLayer = model->network->addActivation(*input, ActivationType::kRELU);
      activationLayer->setName(desc->name.c_str());
      if(forceFP32) {
        activationLayer->setPrecision(DataType::kFLOAT);
      }
      return activationLayer;
    } else if(desc->activation == ACTIVATION_MISH) {
      auto softplusLayer = model->network->addActivation(*input, ActivationType::kSOFTPLUS);
      auto softplusLayerName = desc->name + "/softplus";
      softplusLayer->setName(softplusLayerName.c_str());
      auto tanhLayer = model->network->addActivation(*softplusLayer->getOutput(0), ActivationType::kTANH);
      auto tanhLayerName = desc->name + "/tanh";
      tanhLayer->setName(tanhLayerName.c_str());
      auto mergeLayer = model->network->addElementWise(*input, *tanhLayer->getOutput(0), ElementWiseOperation::kPROD);
      mergeLayer->setName(desc->name.c_str());
      if(forceFP32) {
        softplusLayer->setPrecision(DataType::kFLOAT);
        tanhLayer->setPrecision(DataType::kFLOAT);
        mergeLayer->setPrecision(DataType::kFLOAT);
      }
      return mergeLayer;
    } else {
      ASSERT_UNREACHABLE;
    }
  }

  ILayer* applyGPoolLayer(ILayer* inputLayer, bool forceFP32 = false, bool isValueHead = false) {
    auto& network = model->network;
    string name = inputLayer->getName();

    ILayer* gpoolSumLayer = nullptr;
    ILayer* gpoolMeanLayer = nullptr;
    if(!model->requireExactNNLen) {
      gpoolSumLayer = network->addReduce(*inputLayer->getOutput(0), ReduceOperation::kSUM, 1U << 2 | 1U << 3, true);
      auto gpoolSumLayerName = name + "/gpsum";
      gpoolSumLayer->setName(gpoolSumLayerName.c_str());
      gpoolMeanLayer =
        network->addElementWise(*gpoolSumLayer->getOutput(0), *maskSumLayer->getOutput(0), ElementWiseOperation::kDIV);
    } else {
      gpoolMeanLayer = network->addReduce(*inputLayer->getOutput(0), ReduceOperation::kAVG, 1U << 2 | 1U << 3, true);
    }
    auto gpoolMeanLayerName = name + "/gpmean";
    gpoolMeanLayer->setName(gpoolMeanLayerName.c_str());

    auto gpoolMeanScaleLayer = network->addElementWise(
      *gpoolMeanLayer->getOutput(0), *maskScaleLayer->getOutput(0), ElementWiseOperation::kPROD);
    auto gpoolMeanScaleLayerName = name + "/gpmeanscale";
    gpoolMeanScaleLayer->setName(gpoolMeanScaleLayerName.c_str());

    ILayer* gpoolMaskAddLayer = nullptr;
    ILayer* gpoolMaskShiftLayer = nullptr;
    ILayer* gpoolConcatInputLayer3 = nullptr;
    if(isValueHead) {
      auto gpoolMeanQuadLayer = network->addElementWise(
        *gpoolMeanLayer->getOutput(0), *maskQuadLayer->getOutput(0), ElementWiseOperation::kPROD);
      auto gpoolMeanQuadLayerName = name + "/gpmeanquad";
      gpoolMeanQuadLayer->setName(gpoolMeanQuadLayerName.c_str());
      gpoolConcatInputLayer3 = gpoolMeanQuadLayer;
    } else if(!model->requireExactNNLen) {
      // All activation functions we use right now are always greater than -1.0, and map 0 -> 0.
      // So off-board areas will equal 0, and then this max is mask-safe if we assign -1.0 to off-board areas.
      auto gpoolMaskShiftWeights = make_unique<float[]>(1);
      gpoolMaskShiftWeights[0] = -1.0f;
      gpoolMaskShiftLayer = network->addScale(
        *inputMask,
        ScaleMode::kUNIFORM,
        {DataType::kFLOAT, gpoolMaskShiftWeights.get(), 1},
        {DataType::kFLOAT, nullptr, 0},
        {DataType::kFLOAT, nullptr, 0});
      auto gpoolMaskShiftLayerName = name + "/gpmaskshift";
      gpoolMaskShiftLayer->setName(gpoolMaskShiftLayerName.c_str());
      model->extraWeights.push_back(move(gpoolMaskShiftWeights));
      gpoolMaskAddLayer = network->addElementWise(
        *inputLayer->getOutput(0), *gpoolMaskShiftLayer->getOutput(0), ElementWiseOperation::kSUM);
      auto gpoolMaskAddLayerName = name + "/gpmaskadd";
      gpoolMaskAddLayer->setName(gpoolMaskAddLayerName.c_str());
      auto gpoolMaxLayer =
        network->addReduce(*gpoolMaskAddLayer->getOutput(0), ReduceOperation::kMAX, 1U << 2 | 1U << 3, true);
      auto gpoolMaxLayerName = name + "/gpmax";
      gpoolMaxLayer->setName(gpoolMaxLayerName.c_str());
      gpoolConcatInputLayer3 = gpoolMaxLayer;
    } else {
      auto gpoolMaxLayer =
        network->addReduce(*inputLayer->getOutput(0), ReduceOperation::kMAX, 1U << 2 | 1U << 3, true);
      auto gpoolMaxLayerName = name + "/gpmax";
      gpoolMaxLayer->setName(gpoolMaxLayerName.c_str());
      gpoolConcatInputLayer3 = gpoolMaxLayer;
    }

    ITensor* gpoolConcatInputs[] = {
      gpoolMeanLayer->getOutput(0), gpoolMeanScaleLayer->getOutput(0), gpoolConcatInputLayer3->getOutput(0)};
    auto gpoolConcatLayer = network->addConcatenation(gpoolConcatInputs, 3);
    auto gpoolConcatLayerName = name + "/gpconcat";
    gpoolConcatLayer->setAxis(1);
    gpoolConcatLayer->setName(gpoolConcatLayerName.c_str());

    if(forceFP32) {
      if(gpoolSumLayer) {
        gpoolSumLayer->setPrecision(DataType::kFLOAT);
      }
      if(gpoolMaskAddLayer) {
        gpoolMaskAddLayer->setPrecision(DataType::kFLOAT);
      }
      if(gpoolMaskShiftLayer) {
        gpoolMaskShiftLayer->setPrecision(DataType::kFLOAT);
      }
      gpoolMeanLayer->setPrecision(DataType::kFLOAT);
      gpoolMeanScaleLayer->setPrecision(DataType::kFLOAT);
      gpoolConcatInputLayer3->setPrecision(DataType::kFLOAT);
      gpoolConcatLayer->setPrecision(DataType::kFLOAT);
    }

    return gpoolConcatLayer;
  }

  ILayer* applyMaskLayer(ILayer* inputLayer, bool forceFP32 = false) {
    if(!model->requireExactNNLen) {
      auto maskLayer =
        model->network->addElementWise(*inputLayer->getOutput(0), *inputMask, ElementWiseOperation::kPROD);
      auto maskLayerName = string(inputLayer->getName()) + "/mask";
      maskLayer->setName(maskLayerName.c_str());
      if(forceFP32) {
        maskLayer->setPrecision(DataType::kFLOAT);
      }
      return maskLayer;
    } else {
      return inputLayer;
    }
  }

  ILayer* applyCastLayer(ILayer* inputLayer, DataType dataType) {
    auto castLayer = model->network->addIdentity(*inputLayer->getOutput(0));
    castLayer->setOutputType(0, dataType);
    auto castLayerName = string(inputLayer->getName()) + "/cast";
    castLayer->setName(castLayerName.c_str());
    return castLayer;
  }
};

struct TRTLogger : ILogger {
  Logger* logger;
  Severity level;

  TRTLogger() {
    logger = nullptr;
    level = Severity::kERROR;
  }

  TRTLogger(const TRTLogger&) = delete;
  TRTLogger& operator=(const TRTLogger&) = delete;

  void log(Severity severity, const char* msg) noexcept override {
    if(logger && severity <= level)
      logger->write("TensorRT backend: " + string(msg));
    if(severity == Severity::kERROR && logger && !logger->isLoggingToStderr() && !logger->isLoggingToStdout()) {
      std::cerr << ("TensorRT backend: " + string(msg)) << std::endl;
    }
  }

  void setLogger(Logger* externalLogger) { logger = externalLogger; }
};

struct ComputeHandle {
  ComputeContext* ctx;

  bool usingFP16;
  int maxBatchSize;
  int modelVersion;
  vector<pair<string, string>> debugOutputs;

  TRTLogger trtLogger;
  map<string, void*> buffers;
  unique_ptr<ICudaEngine> engine;
  unique_ptr<IExecutionContext> exec;

  ComputeHandle(
    Logger* logger,
    const cudaDeviceProp* prop,
    ComputeContext* context,
    const LoadedModel* loadedModel,
    int maxBatchSz,
    bool requireExactNNLen) {
    ctx = context;

    maxBatchSize = maxBatchSz;
    modelVersion = loadedModel->modelDesc.version;

    // Certain minor versions of TensorRT uses a global logger, which is bad.
    // Since TensorRT maintains ABI compatibility between minor versions, a dynamic library mismatch
    // does not necessarily generate a dynamic link error, therefore, an extra check is required.
    if(getInferLibVersion() / 100 != NV_TENSORRT_VERSION / 100) {
      throw StringError("TensorRT backend: detected incompatible version of TensorRT library");
    }

    trtLogger.setLogger(logger);

    auto builder = unique_ptr<IBuilder>(createInferBuilder(trtLogger));
    if(!builder) {
      throw StringError("TensorRT backend: failed to create builder");
    }
    auto config = unique_ptr<IBuilderConfig>(builder->createBuilderConfig());
    if(!config) {
      throw StringError("TensorRT backend: failed to create builder config");
    }

    usingFP16 = false;
    if(builder->platformHasFastFp16()) {
      if(ctx->useFP16Mode == enabled_t::True || ctx->useFP16Mode == enabled_t::Auto) {
        config->setFlag(BuilderFlag::kFP16);
        usingFP16 = true;
      }
    } else if(ctx->useFP16Mode == enabled_t::True) {
      throw StringError("CUDA device does not support useFP16=true");
    }
    config->setFlag(BuilderFlag::kPREFER_PRECISION_CONSTRAINTS);

    auto network = unique_ptr<INetworkDefinition>(
      builder->createNetworkV2(1U << static_cast<int>(NetworkDefinitionCreationFlag::kEXPLICIT_BATCH)));
    if(!network) {
      throw StringError("TensorRT backend: failed to create network definition");
    }
    auto profile = builder->createOptimizationProfile();
    if(!profile) {
      throw StringError("TensorRT backend: failed to create optimization profile");
    }
    auto modelParser = make_unique<ModelParser>();
    auto model = modelParser->build(
      move(network), profile, loadedModel, ctx->nnXLen, ctx->nnYLen, maxBatchSize, requireExactNNLen);
    debugOutputs = model->debugOutputs;
    config->addOptimizationProfile(profile);

    // This is to avoid external tactic sources and tactics that have shape switching overhead
    if(prop->major < 8) {
      config->setTacticSources(
        1U << static_cast<uint32_t>(TacticSource::kJIT_CONVOLUTIONS) |
        1U << static_cast<uint32_t>(TacticSource::kEDGE_MASK_CONVOLUTIONS));
    } else {
      config->setTacticSources(1U << static_cast<uint32_t>(TacticSource::kJIT_CONVOLUTIONS));
    }

    // So that there are no concurrent kernel executions probably from other parts of code while profiling
    // See CUDA Runtime API document for more details related to NULL stream and synchronization behaviors
    config->setProfileStream(cudaStreamLegacy);

    // Typical runtime allocation is much less than the 1 GiB specified below
    config->setMemoryPoolLimit(MemoryPoolType::kWORKSPACE, 1U << 30);

    string plan;
    {
      static mutex tuneMutex;
      tuneMutex.lock();

      auto cacheDir = HomeData::getHomeDataDir(true, ctx->homeDataDirOverride);
      cacheDir += "/trtcache";
      MakeDir::make(cacheDir);

      uint8_t deviceHash[32];
      SHA2::get256(prop->name, deviceHash);

      // Truncated to 4 bytes
      char deviceIdent[4 * 2 + 1];
      for(int i = 0; i < 4; i++) {
        sprintf(deviceIdent + i * 2, "%02x", static_cast<unsigned char>(deviceHash[i]));
      }
      deviceIdent[sizeof(deviceIdent) - 1] = 0;

#ifdef CACHE_TENSORRT_PLAN
      auto planCacheFile = Global::strprintf(
        "%s/trt-%d_gpu-%s_net-%s_%d_%s%dx%d_batch%d_fp%d",
        cacheDir.c_str(),
        getInferLibVersion(),
        deviceIdent,
        loadedModel->modelDesc.name.c_str(),
        ModelParser::tuneSalt,
        requireExactNNLen ? "exact" : "max",
        ctx->nnYLen,
        ctx->nnXLen,
        maxBatchSize,
        usingFP16 ? 16 : 32
      );
      string paramStr = Global::strprintf(
        "_%d_%s_%d_%s_%d_%d_%d_%d",
        getInferLibVersion(),
        deviceIdent,
        ModelParser::tuneSalt,
        requireExactNNLen ? "exact" : "max",
        ctx->nnYLen,
        ctx->nnXLen,
        maxBatchSize,
        usingFP16 ? 16 : 32
      );
      try {
        plan = FileUtils::readFileBinary(planCacheFile);
      } catch(const StringError& e) {
        (void)e;
      };

      if(plan.size() > 0) {
        if(plan.size() < 64 + paramStr.size()) {
          logger->write("Could not parse plan, unexpected size in " + planCacheFile);
          plan.clear();
        }
        else {
          string cachedParamStr = plan.substr(plan.size()-paramStr.size());
          string modelHash = plan.substr(plan.size()-64-paramStr.size(),64);
          if(modelHash != loadedModel->modelDesc.sha256) {
            logger->write("Plan cache is corrupted or is for the wrong model in " + planCacheFile);
            plan.clear();
          }
          else if(cachedParamStr != paramStr) {
            logger->write("Plan cache is corrupted or is for the wrong parameters in " + planCacheFile);
            plan.clear();
          }
          else {
            plan.erase(plan.size()-64-paramStr.size());
          }
        }
      }

      if(plan.size() <= 0) {
        logger->write("Creating new plan cache");
        auto planBuffer = unique_ptr<IHostMemory>(builder->buildSerializedNetwork(*model->network, *config));
        if(!planBuffer) {
          throw StringError("TensorRT backend: failed to create plan");
        }
        plan.insert(
          plan.end(),
          static_cast<char*>(planBuffer->data()),
          static_cast<char*>(planBuffer->data()) + planBuffer->size()
        );
        if(loadedModel->modelDesc.sha256.size() != 64) {
          throw StringError("Unexpected model hash size");
        }
        plan.insert(
          plan.end(),
          loadedModel->modelDesc.sha256.begin(),
          loadedModel->modelDesc.sha256.end()
        );
        plan.insert(
          plan.end(),
          paramStr.begin(),
          paramStr.end()
        );
        ofstream ofs;
        FileUtils::open(ofs, planCacheFile, ios::out | ios::binary);
        ofs.write(plan.data(), plan.size());
        ofs.close();
        logger->write("Saved new plan cache to " + planCacheFile);
        plan.erase(plan.size()-64-paramStr.size());
        tuneMutex.unlock();
      } else {
        tuneMutex.unlock();
        logger->write("Using existing plan cache at " + planCacheFile);
      }
#else
      // Truncated to 6 bytes
      char tuneIdent[6 * 2 + 1];
      for(int i = 0; i < 6; i++) {
        sprintf(tuneIdent + i * 2, "%02x", static_cast<unsigned char>(model->tuneHash[i]));
      }
      tuneIdent[sizeof(tuneIdent) - 1] = 0;

      auto timingCacheFile = Global::strprintf(
        "%s/trt-%d_gpu-%s_tune-%s_%s%dx%d_batch%d_fp%d",
        cacheDir.c_str(),
        getInferLibVersion(),
        deviceIdent,
        tuneIdent,
        requireExactNNLen ? "exact" : "max",
        ctx->nnYLen,
        ctx->nnXLen,
        maxBatchSize,
        usingFP16 ? 16 : 32);

      string timingCacheBlob;
      try {
        timingCacheBlob = FileUtils::readFileBinary(timingCacheFile);
      } catch(const StringError& e) {
        (void)e;
      };
      if(timingCacheBlob.size() > 0)
        logger->write("Using existing timing cache at " + timingCacheFile);
      else
        logger->write("Creating new timing cache");

      auto timingCache =
        unique_ptr<ITimingCache>(config->createTimingCache(timingCacheBlob.data(), timingCacheBlob.size()));
      auto invalidTimingCache = !config->setTimingCache(*timingCache, false);
      if(invalidTimingCache) {
        logger->write("Invalid timing cache, using new one instead");
        timingCache.reset(config->createTimingCache(nullptr, 0));
        config->setTimingCache(*timingCache, false);
      }

      unique_ptr<IHostMemory> planBuffer;
      if(invalidTimingCache || !timingCacheBlob.size()) {
        planBuffer.reset(builder->buildSerializedNetwork(*model->network, *config));
        if(!planBuffer) {
          throw StringError("TensorRT backend: failed to create plan");
        }
        auto serializedTimingCache = unique_ptr<IHostMemory>(config->getTimingCache()->serialize());
        ofstream ofs;
        FileUtils::open(ofs, timingCacheFile, ios::out | ios::binary);
        ofs.write(static_cast<char*>(serializedTimingCache->data()), serializedTimingCache->size());
        ofs.close();
        logger->write("Saved new timing cache to " + timingCacheFile);
        tuneMutex.unlock();
      } else {
        tuneMutex.unlock();
        planBuffer.reset(builder->buildSerializedNetwork(*model->network, *config));
        if(!planBuffer) {
          throw StringError("TensorRT backend: failed to create plan");
        }
      }
      plan.insert(
        plan.end(),
        static_cast<char*>(planBuffer->data()),
        static_cast<char*>(planBuffer->data()) + planBuffer->size());
#endif
    }

    auto runtime = unique_ptr<IRuntime>(createInferRuntime(trtLogger));
    if(!runtime) {
      throw StringError("TensorRT backend: failed to create runtime");
    }

    engine.reset(runtime->deserializeCudaEngine(plan.data(), plan.size()));
    if(!engine) {
      throw StringError("TensorRT backend: failed to create cuda engine");
    }
    exec.reset(engine->createExecutionContext());
    if(!exec) {
      throw StringError("TensorRT backend: failed to create execution context");
    }

    for(int i = 0; i < engine->getNbIOTensors(); i++) {
      void* buffer = nullptr;
      auto name = engine->getIOTensorName(i);
      auto dims = engine->getTensorShape(name);
      size_t bytes = accumulate(dims.d + 1, dims.d + dims.nbDims, maxBatchSize * sizeof(float), multiplies<size_t>());
      CUDA_ERR("ComputeHandle", cudaMalloc(&buffer, bytes));
      buffers.emplace(make_pair(name, buffer));
      exec->setTensorAddress(name, buffer);
    }

    exec->setOptimizationProfileAsync(0, cudaStreamPerThread);
    cudaStreamSynchronize(cudaStreamPerThread);
  }

  ~ComputeHandle() {
    for(auto ptr: buffers) {
      CUDA_ERR("~ComputeHandle", cudaFree(ptr.second));
    }
  }

  ComputeHandle() = delete;
  ComputeHandle(const ComputeHandle&) = delete;
  ComputeHandle& operator=(const ComputeHandle&) = delete;

  void* getBuffer(const char* name) {
    auto search = buffers.find(name);
    if(search != buffers.end()) {
      return search->second;
    } else {
      throw StringError(Global::strprintf("ComputeHandle: unknown tensor name %s", name));
    }
  }

  size_t getBufferBytes(const char* name) {
    auto dims = engine->getTensorShape(name);
    if(dims.nbDims != -1) {
      return accumulate(dims.d + 1, dims.d + dims.nbDims, maxBatchSize * sizeof(float), multiplies<size_t>());
    } else {
      throw StringError(Global::strprintf("ComputeHandle: unknown tensor name %s", name));
    }
  }

  size_t getBufferRowElts(const char* name) {
    auto dims = engine->getTensorShape(name);
    if(dims.nbDims != -1) {
      return accumulate(dims.d + 1, dims.d + dims.nbDims, 1, multiplies<size_t>());
    } else {
      throw StringError(Global::strprintf("ComputeHandle: unknown tensor name %s", name));
    }
  }

  Dims getBufferDynamicShape(const char* name, int batchSize) {
    auto dims = engine->getTensorShape(name);
    if(dims.nbDims != -1) {
      dims.d[0] = batchSize;
      return dims;
    } else {
      throw StringError(Global::strprintf("ComputeHandle: unknown tensor name %s", name));
    }
  }

  void printDebugOutput(int batchSize) {
    for(auto& debugOutput: debugOutputs) {
      auto name = debugOutput.first;
      auto desc = debugOutput.second;
      auto dims = getBufferDynamicShape(name.c_str(), batchSize);

      vector<float> values(accumulate(dims.d, dims.d + dims.nbDims, 1, multiplies<size_t>()));
      CUDA_ERR(
        "printDebugOutput",
        cudaMemcpy(values.data(), getBuffer(name.c_str()), values.size() * sizeof(float), cudaMemcpyDeviceToHost));

      cout << "=========================================================" << endl;
      cout << desc << endl;
      int i = 0;
      if(dims.nbDims == 2) {
        for(int n = 0; n < dims.d[0]; n++) {
          cout << "-(n=" << n << ")--------------------" << endl;
          for(int c = 0; c < dims.d[1]; c++) {
            cout << values[i++] << " ";
          }
          cout << endl;
        }
        cout << endl;
      } else if(dims.nbDims == 4) {
        for(int n = 0; n < dims.d[0]; n++) {
          cout << "-(n=" << n << ")--------------------" << endl;
          for(int c = 0; c < dims.d[1]; c++) {
            cout << "(c=" << c << ")" << endl;
            for(int y = 0; y < dims.d[2]; y++) {
              for(int x = 0; x < dims.d[3]; x++)
                cout << values[i++] << " ";
              cout << endl;
            }
            cout << endl;
          }
        }
      }
      cout << "=========================================================" << endl;
    }
  }
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
  if(inputsUseNHWC) {
    throw StringError("TensorRT backend: inputsUseNHWC = false required, other configurations not supported");
  }

  // Use whatever CUDA believes GPU 0 to be.
  if(gpuIdxForThisThread == -1)
    gpuIdxForThisThread = 0;
  CUDA_ERR("createComputeHandle", cudaSetDevice(gpuIdxForThisThread));

  cudaDeviceProp prop;
  CUDA_ERR("createComputeHandle", cudaGetDeviceProperties(&prop, gpuIdxForThisThread));

  if(logger != NULL) {
    logger->write(
      "TensorRT backend thread " + Global::intToString(serverThreadIdx) + ": Found GPU " + string(prop.name) +
      " memory " + Global::uint64ToString(prop.totalGlobalMem) + " compute capability major " +
      Global::intToString(prop.major) + " minor " + Global::intToString(prop.minor));
    logger->write(
      "TensorRT backend thread " + Global::intToString(serverThreadIdx) + ": Initializing (may take a long time)");
  }

  auto handle = new ComputeHandle(logger, &prop, context, loadedModel, maxBatchSize, requireExactNNLen);

  if(logger != NULL) {
    logger->write(
      "TensorRT backend thread " + Global::intToString(serverThreadIdx) + ": Model version " +
      Global::intToString(loadedModel->modelDesc.version) + " useFP16 = " + Global::boolToString(handle->usingFP16));
    logger->write(
      "TensorRT backend thread " + Global::intToString(serverThreadIdx) +
      ": Model name: " + loadedModel->modelDesc.name);
  }

  return handle;
}

void NeuralNet::freeComputeHandle(ComputeHandle* gpuHandle) {
  delete gpuHandle;
}

bool NeuralNet::isUsingFP16(const ComputeHandle* gpuHandle) {
  return gpuHandle->usingFP16;
}

void NeuralNet::printDevices() {
  int numDevices = 0;
  CUDA_ERR("printDevices", cudaGetDeviceCount(&numDevices));
  for(int i = 0; i < numDevices; i++) {
    cudaDeviceProp prop;
    CUDA_ERR("printDevices", cudaGetDeviceProperties(&prop, i));
    cout << "Found GPU device " << i << ": " << prop.name << endl;
  }
}

struct InputBuffers {
  int maxBatchSize;

  size_t singleMaskElts;
  size_t singleMaskBytes;
  size_t singleFeatureElts;
  size_t singleFeatureBytes;
  size_t singleGlobalFeatureElts;
  size_t singleGlobalFeatureBytes;
  size_t singlePolicyPassResultElts;
  size_t singlePolicyPassResultBytes;
  size_t singlePolicyResultElts;
  size_t singlePolicyResultBytes;
  size_t singleValueResultElts;
  size_t singleValueResultBytes;
  size_t singleScoreValueResultElts;
  size_t singleScoreValueResultBytes;
  size_t singleOwnershipResultElts;
  size_t singleOwnershipResultBytes;

  size_t maskInputBufferBytes;
  size_t featureInputBufferBytes;
  size_t globalFeatureInputBufferBytes;
  size_t policyPassResultBufferBytes;
  size_t policyResultBufferBytes;
  size_t valueResultBufferBytes;
  size_t scoreValueResultBufferBytes;
  size_t ownershipResultBufferBytes;

  unique_ptr<float[]> maskInputs;           // Host pointer
  unique_ptr<float[]> featureInputs;        // Host pointer
  unique_ptr<float[]> globalFeatureInputs;  // Host pointer
  unique_ptr<float[]> policyPassResults;    // Host pointer
  unique_ptr<float[]> policyResults;        // Host pointer
  unique_ptr<float[]> valueResults;         // Host pointer
  unique_ptr<float[]> scoreValueResults;    // Host pointer
  unique_ptr<float[]> ownershipResults;     // Host pointer

  InputBuffers(const LoadedModel* loadedModel, int maxBatchSz, int nnXLen, int nnYLen) {
    const ModelDesc& m = loadedModel->modelDesc;

    if(nnXLen > NNPos::MAX_BOARD_LEN)
      throw StringError(
        Global::strprintf("nnXLen (%d) is greater than NNPos::MAX_BOARD_LEN (%d)", nnXLen, NNPos::MAX_BOARD_LEN));
    if(nnYLen > NNPos::MAX_BOARD_LEN)
      throw StringError(
        Global::strprintf("nnYLen (%d) is greater than NNPos::MAX_BOARD_LEN (%d)", nnYLen, NNPos::MAX_BOARD_LEN));

    const int policyChannels = m.version >= 12 ? 2 : 1;
    assert(policyChannels == m.policyHead.p2Conv.outChannels);

    maxBatchSize = maxBatchSz;
    singleMaskElts = nnXLen * nnYLen;
    singleMaskBytes = singleMaskElts * sizeof(float);
    singleFeatureElts = m.numInputChannels * nnXLen * nnYLen;
    singleFeatureBytes = singleFeatureElts * sizeof(float);
    singleGlobalFeatureElts = m.numInputGlobalChannels;
    singleGlobalFeatureBytes = singleGlobalFeatureElts * sizeof(float);
    singlePolicyPassResultElts = (size_t)policyChannels;
    singlePolicyPassResultBytes = singlePolicyPassResultElts * sizeof(float);
    singlePolicyResultElts = (size_t)policyChannels * nnXLen * nnYLen;
    singlePolicyResultBytes = singlePolicyResultElts * sizeof(float);
    singleValueResultElts = m.numValueChannels;
    singleValueResultBytes = singleValueResultElts * sizeof(float);
    singleScoreValueResultElts = m.numScoreValueChannels;
    singleScoreValueResultBytes = singleScoreValueResultElts * sizeof(float);
    singleOwnershipResultElts = m.numOwnershipChannels * nnXLen * nnYLen;
    singleOwnershipResultBytes = singleOwnershipResultElts * sizeof(float);

    assert(NNModelVersion::getNumSpatialFeatures(m.version) == m.numInputChannels);
    assert(NNModelVersion::getNumGlobalFeatures(m.version) == m.numInputGlobalChannels);

    maskInputBufferBytes = maxBatchSize * singleMaskBytes;
    featureInputBufferBytes = maxBatchSize * singleFeatureBytes;
    globalFeatureInputBufferBytes = maxBatchSize * singleGlobalFeatureBytes;
    policyPassResultBufferBytes = maxBatchSize * singlePolicyPassResultBytes;
    policyResultBufferBytes = maxBatchSize * singlePolicyResultBytes;
    valueResultBufferBytes = maxBatchSize * singleValueResultBytes;
    scoreValueResultBufferBytes = maxBatchSize * singleScoreValueResultBytes;
    ownershipResultBufferBytes = maxBatchSize * singleOwnershipResultBytes;

    maskInputs = make_unique<float[]>(maxBatchSize * singleMaskElts);
    featureInputs = make_unique<float[]>(maxBatchSize * singleFeatureElts);
    globalFeatureInputs = make_unique<float[]>(maxBatchSize * singleGlobalFeatureElts);
    policyPassResults = make_unique<float[]>(maxBatchSize * singlePolicyPassResultElts);
    policyResults = make_unique<float[]>(maxBatchSize * singlePolicyResultElts);
    valueResults = make_unique<float[]>(maxBatchSize * singleValueResultElts);
    scoreValueResults = make_unique<float[]>(maxBatchSize * singleScoreValueResultElts);
    ownershipResults = make_unique<float[]>(maxBatchSize * singleOwnershipResultElts);
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
  assert(numBatchEltsFilled <= inputBuffers->maxBatchSize);
  assert(numBatchEltsFilled > 0);

  int batchSize = numBatchEltsFilled;
  int nnXLen = gpuHandle->ctx->nnXLen;
  int nnYLen = gpuHandle->ctx->nnYLen;
  int version = gpuHandle->modelVersion;

  int numSpatialFeatures = NNModelVersion::getNumSpatialFeatures(version);
  int numGlobalFeatures = NNModelVersion::getNumGlobalFeatures(version);

  for(int nIdx = 0; nIdx < batchSize; nIdx++) {
    float* rowMaskInput = &inputBuffers->maskInputs[inputBuffers->singleMaskElts * nIdx];
    float* rowFeatureInput = &inputBuffers->featureInputs[inputBuffers->singleFeatureElts * nIdx];
    float* rowGlobalFeatureInput = &inputBuffers->globalFeatureInputs[inputBuffers->singleGlobalFeatureElts * nIdx];

    const float* rowFeature = inputBufs[nIdx]->rowSpatial;
    const float* rowGlobalFeature = inputBufs[nIdx]->rowGlobal;
    SymmetryHelpers::copyInputsWithSymmetry(
      rowFeature, rowFeatureInput, 1, nnYLen, nnXLen, numSpatialFeatures, false, inputBufs[nIdx]->symmetry);
    copy(rowGlobalFeature, rowGlobalFeature + numGlobalFeatures, rowGlobalFeatureInput);
    copy(rowFeatureInput, rowFeatureInput + inputBuffers->singleMaskElts, rowMaskInput);
  }

  assert(inputBuffers->singleMaskElts == gpuHandle->getBufferRowElts("InputMask"));
  assert(inputBuffers->singleFeatureElts == gpuHandle->getBufferRowElts("InputFeature"));
  assert(inputBuffers->singleGlobalFeatureElts == gpuHandle->getBufferRowElts("InputGlobalFeature"));
  assert(inputBuffers->singlePolicyPassResultElts == gpuHandle->getBufferRowElts("OutputPolicyPass"));
  assert(inputBuffers->singlePolicyResultElts == gpuHandle->getBufferRowElts("OutputPolicy"));
  assert(inputBuffers->singleValueResultElts == gpuHandle->getBufferRowElts("OutputValue"));
  assert(inputBuffers->singleScoreValueResultElts == gpuHandle->getBufferRowElts("OutputScoreValue"));
  assert(inputBuffers->singleOwnershipResultElts == gpuHandle->getBufferRowElts("OutputOwnership"));

  assert(inputBuffers->maskInputBufferBytes == gpuHandle->getBufferBytes("InputMask"));
  assert(inputBuffers->featureInputBufferBytes == gpuHandle->getBufferBytes("InputFeature"));
  assert(inputBuffers->globalFeatureInputBufferBytes == gpuHandle->getBufferBytes("InputGlobalFeature"));
  assert(inputBuffers->policyPassResultBufferBytes == gpuHandle->getBufferBytes("OutputPolicyPass"));
  assert(inputBuffers->policyResultBufferBytes == gpuHandle->getBufferBytes("OutputPolicy"));
  assert(inputBuffers->valueResultBufferBytes == gpuHandle->getBufferBytes("OutputValue"));
  assert(inputBuffers->scoreValueResultBufferBytes == gpuHandle->getBufferBytes("OutputScoreValue"));
  assert(inputBuffers->ownershipResultBufferBytes == gpuHandle->getBufferBytes("OutputOwnership"));

  const int policyChannels = version >= 12 ? 2 : 1;
  assert(inputBuffers->singlePolicyPassResultElts == policyChannels);
  assert(inputBuffers->singlePolicyResultElts == policyChannels * nnXLen * nnYLen);

  // Transfers from host memory to device memory are asynchronous with respect to the host
  CUDA_ERR(
    "getOutput",
    cudaMemcpyAsync(
      gpuHandle->getBuffer("InputMask"),
      inputBuffers->maskInputs.get(),
      inputBuffers->singleMaskBytes * batchSize,
      cudaMemcpyHostToDevice));
  CUDA_ERR(
    "getOutput",
    cudaMemcpyAsync(
      gpuHandle->getBuffer("InputFeature"),
      inputBuffers->featureInputs.get(),
      inputBuffers->singleFeatureBytes * batchSize,
      cudaMemcpyHostToDevice));
  CUDA_ERR(
    "getOutput",
    cudaMemcpyAsync(
      gpuHandle->getBuffer("InputGlobalFeature"),
      inputBuffers->globalFeatureInputs.get(),
      inputBuffers->singleGlobalFeatureBytes * batchSize,
      cudaMemcpyHostToDevice));

  auto maskInputDims = gpuHandle->getBufferDynamicShape("InputMask", batchSize);
  auto featureInputDims = gpuHandle->getBufferDynamicShape("InputFeature", batchSize);
  auto globalFeatureInputDims = gpuHandle->getBufferDynamicShape("InputGlobalFeature", batchSize);

  gpuHandle->exec->setInputShape("InputMask", maskInputDims);
  gpuHandle->exec->setInputShape("InputFeature", featureInputDims);
  gpuHandle->exec->setInputShape("InputGlobalFeature", globalFeatureInputDims);

  gpuHandle->exec->enqueueV3(cudaStreamPerThread);

  CUDA_ERR(
    "getOutput",
    cudaMemcpy(
      inputBuffers->policyPassResults.get(),
      gpuHandle->getBuffer("OutputPolicyPass"),
      inputBuffers->singlePolicyPassResultBytes * batchSize,
      cudaMemcpyDeviceToHost));
  CUDA_ERR(
    "getOutput",
    cudaMemcpy(
      inputBuffers->policyResults.get(),
      gpuHandle->getBuffer("OutputPolicy"),
      inputBuffers->singlePolicyResultBytes * batchSize,
      cudaMemcpyDeviceToHost));
  CUDA_ERR(
    "getOutput",
    cudaMemcpy(
      inputBuffers->valueResults.get(),
      gpuHandle->getBuffer("OutputValue"),
      inputBuffers->singleValueResultBytes * batchSize,
      cudaMemcpyDeviceToHost));
  CUDA_ERR(
    "getOutput",
    cudaMemcpy(
      inputBuffers->scoreValueResults.get(),
      gpuHandle->getBuffer("OutputScoreValue"),
      inputBuffers->singleScoreValueResultBytes * batchSize,
      cudaMemcpyDeviceToHost));
  CUDA_ERR(
    "getOutput",
    cudaMemcpy(
      inputBuffers->ownershipResults.get(),
      gpuHandle->getBuffer("OutputOwnership"),
      inputBuffers->singleOwnershipResultBytes * batchSize,
      cudaMemcpyDeviceToHost));

  gpuHandle->printDebugOutput(batchSize);

  assert(outputs.size() == batchSize);

  float policyProbsTmp[NNPos::MAX_NN_POLICY_SIZE];

  for(int row = 0; row < batchSize; row++) {
    NNOutput* output = outputs[row];

    assert(output->nnXLen == nnXLen);
    assert(output->nnYLen == nnYLen);
    float policyOptimism = (float)inputBufs[row]->policyOptimism;

    const float* policyPassSrcBuf = &inputBuffers->policyPassResults[row * inputBuffers->singlePolicyPassResultElts];
    const float* policySrcBuf = &inputBuffers->policyResults[row * inputBuffers->singlePolicyResultElts];
    float* policyProbs = output->policyProbs;

    // These are in logits, the client does the postprocessing to turn them into
    // policy probabilities and white game outcome probabilities
    // Also we don't fill in the nnHash here either
    // Handle version >= 12 policy optimism
    if(policyChannels == 2) {
      // TRT is all NCHW
      for(int i = 0; i<nnXLen*nnYLen; i++) {
        float p = policySrcBuf[i];
        float pOpt = policySrcBuf[i+nnXLen*nnYLen];
        policyProbsTmp[i] = p + (pOpt-p) * policyOptimism;
      }
      SymmetryHelpers::copyOutputsWithSymmetry(policyProbsTmp, policyProbs, 1, nnYLen, nnXLen, inputBufs[row]->symmetry);
      policyProbs[nnXLen*nnYLen] = policyPassSrcBuf[0] + (policyPassSrcBuf[1] - policyPassSrcBuf[0]) * policyOptimism;
    }
    else {
      assert(policyChannels == 1);
      SymmetryHelpers::copyOutputsWithSymmetry(policySrcBuf, policyProbs, 1, nnYLen, nnXLen, inputBufs[row]->symmetry);
      policyProbs[nnXLen * nnYLen] = policyPassSrcBuf[0];
    }

    int numValueChannels = inputBuffers->singleValueResultElts;
    assert(numValueChannels == 3);
    output->whiteWinProb = inputBuffers->valueResults[row * numValueChannels];
    output->whiteLossProb = inputBuffers->valueResults[row * numValueChannels + 1];
    output->whiteNoResultProb = inputBuffers->valueResults[row * numValueChannels + 2];

    // As above, these are NOT actually from white's perspective, but rather the player to move.
    // As usual the client does the postprocessing.
    if(output->whiteOwnerMap != NULL) {
      const float* ownershipSrcBuf = &inputBuffers->ownershipResults[row * nnXLen * nnYLen];
      assert(inputBuffers->singleOwnershipResultElts == nnXLen * nnYLen);
      SymmetryHelpers::copyOutputsWithSymmetry(
        ownershipSrcBuf, output->whiteOwnerMap, 1, nnYLen, nnXLen, inputBufs[row]->symmetry);
    }

    int numScoreValueChannels = inputBuffers->singleScoreValueResultElts;
    if(version >= 9) {
      assert(numScoreValueChannels == 6);
      output->whiteScoreMean = inputBuffers->scoreValueResults[row * numScoreValueChannels];
      output->whiteScoreMeanSq = inputBuffers->scoreValueResults[row * numScoreValueChannels + 1];
      output->whiteLead = inputBuffers->scoreValueResults[row * numScoreValueChannels + 2];
      output->varTimeLeft = inputBuffers->scoreValueResults[row * numScoreValueChannels + 3];
      output->shorttermWinlossError = inputBuffers->scoreValueResults[row * numScoreValueChannels + 4];
      output->shorttermScoreError = inputBuffers->scoreValueResults[row * numScoreValueChannels + 5];
    } else if(version >= 8) {
      assert(numScoreValueChannels == 4);
      output->whiteScoreMean = inputBuffers->scoreValueResults[row * numScoreValueChannels];
      output->whiteScoreMeanSq = inputBuffers->scoreValueResults[row * numScoreValueChannels + 1];
      output->whiteLead = inputBuffers->scoreValueResults[row * numScoreValueChannels + 2];
      output->varTimeLeft = inputBuffers->scoreValueResults[row * numScoreValueChannels + 3];
      output->shorttermWinlossError = 0;
      output->shorttermScoreError = 0;
    } else if(version >= 4) {
      assert(numScoreValueChannels == 2);
      output->whiteScoreMean = inputBuffers->scoreValueResults[row * numScoreValueChannels];
      output->whiteScoreMeanSq = inputBuffers->scoreValueResults[row * numScoreValueChannels + 1];
      output->whiteLead = output->whiteScoreMean;
      output->varTimeLeft = 0;
      output->shorttermWinlossError = 0;
      output->shorttermScoreError = 0;
    } else if(version >= 3) {
      assert(numScoreValueChannels == 1);
      output->whiteScoreMean = inputBuffers->scoreValueResults[row * numScoreValueChannels];
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

#endif  // USE_TENSORRT_BACKEND
