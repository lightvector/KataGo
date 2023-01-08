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

struct TRTModel {
  int nnXLen;
  int nnYLen;
  bool requireExactNNLen;

  // TensorRT keeps only reference to weights before engine is built
  const LoadedModel* rawModel;
  vector<unique_ptr<float[]>> extraWeights;

  int version;
  uint8_t tuneHash[32];
  unique_ptr<INetworkDefinition> network;
  vector<pair<string, Dims>> debugOutputs;

  TRTModel() = default;
  TRTModel(TRTModel&&) = default;
  TRTModel(const TRTModel&) = delete;
  TRTModel& operator=(TRTModel&&) = default;
  TRTModel& operator=(const TRTModel&) = delete;
};

struct ModelBuilder {
  unique_ptr<TRTModel> model;

  ITensor* inputFeature;
  ITensor* inputGlobalFeature;

  ILayer* maskExtractLayer;
  ILayer* maskExtractSumLayer;
  ILayer* maskExtractLinLayer;
  ILayer* maskExtractQuadLayer;

  string tuneDesc;  // Serves as a hash of the network architecture specific to tuning

  ModelBuilder() = default;
  ModelBuilder(const ModelBuilder&) = delete;
  ModelBuilder& operator=(const ModelBuilder&) = delete;

  unique_ptr<TRTModel> build(
    unique_ptr<INetworkDefinition> net,
    const LoadedModel* rawModel,
    int nnXLen,
    int nnYLen,
    bool requireExactNNLen) {
    model = make_unique<TRTModel>();

    model->nnXLen = nnXLen;
    model->nnYLen = nnYLen;
    model->rawModel = rawModel;
    model->network = move(net);
    model->requireExactNNLen = requireExactNNLen;

    auto& network = model->network;
    auto modelDesc = &model->rawModel->modelDesc;

    int modelSalt = 1;  // Bump this when between katago versions we want to forcibly drop old timing caches.
    tuneDesc = Global::strprintf(
      R"("model"(%d,%d,%d,%d,%d,%d,%d))",
      modelSalt,
      modelDesc->version,
      modelDesc->numInputChannels,
      modelDesc->numInputGlobalChannels,
      modelDesc->numValueChannels,
      modelDesc->numScoreValueChannels,
      modelDesc->numOwnershipChannels);

    model->version = modelDesc->version;
    network->setName(modelDesc->name.c_str());

    initInputs();
    initMaskLayers();

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
      layer->setReshapeDimensions({1, {-1}});
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
    model->debugOutputs.push_back(pair<string, Dims>(description, debugOutput->getDimensions()));
#else
    (void)tensor;
    (void)description;
    (void)force2D;
#endif
  }

  void initInputs() {
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

    assert(network->hasImplicitBatchDimension());

    inputFeature = network->addInput("InputFeature", DataType::kFLOAT, {3, {numInputChannels, nnYLen, nnXLen}});
    inputFeature->setAllowedFormats(1U << static_cast<int>(TensorFormat::kLINEAR));

    inputGlobalFeature = network->addInput("InputGlobalFeature", DataType::kFLOAT, {3, {numInputGlobalChannels, 1, 1}});
    inputFeature->setAllowedFormats(1U << static_cast<int>(TensorFormat::kLINEAR));

    markDebugOutput(inputFeature, "Initial bin features");
  }

  void initMaskLayers() {
    int nnXLen = model->nnXLen;
    int nnYLen = model->nnYLen;
    auto& network = model->network;

    if(!model->requireExactNNLen) {
      maskExtractLayer = network->addSlice(*inputFeature, {3, {0, 0, 0}}, {3, {1, nnYLen, nnXLen}}, {3, {1, 1, 1}});
      maskExtractLayer->setName("MaskExtract");
      maskExtractLayer->setPrecision(DataType::kFLOAT);

      maskExtractSumLayer =
        network->addReduce(*maskExtractLayer->getOutput(0), ReduceOperation::kSUM, 1U << 1 | 1U << 2, true);
      maskExtractSumLayer->setName("MaskExtract/Sum");
      maskExtractSumLayer->setPrecision(DataType::kFLOAT);

      auto maskExtractWidthLayer = network->addUnary(*maskExtractSumLayer->getOutput(0), UnaryOperation::kSQRT);
      maskExtractWidthLayer->setName("MaskExtract/Width");
      maskExtractWidthLayer->setPrecision(DataType::kFLOAT);

      auto maskExtractLinWeightsShift = make_unique<float[]>(1);
      auto maskExtractLinWeightsScale = make_unique<float[]>(1);
      maskExtractLinWeightsShift[0] = -1.4f;
      maskExtractLinWeightsScale[0] = 0.1f;
      maskExtractLinLayer = network->addScale(
        *maskExtractWidthLayer->getOutput(0),
        ScaleMode::kUNIFORM,
        {DataType::kFLOAT, maskExtractLinWeightsShift.get(), 1},
        {DataType::kFLOAT, maskExtractLinWeightsScale.get(), 1},
        {DataType::kFLOAT, nullptr, 0});
      maskExtractLinLayer->setName("MaskExtract/Lin");
      maskExtractLinLayer->setPrecision(DataType::kFLOAT);
      model->extraWeights.push_back(move(maskExtractLinWeightsShift));
      model->extraWeights.push_back(move(maskExtractLinWeightsScale));

      auto maskExtractCenterSquareWeightsShift = make_unique<float[]>(1);
      auto maskExtractCenterSquareWeightsPower = make_unique<float[]>(1);
      maskExtractCenterSquareWeightsShift[0] = -14.0f;
      maskExtractCenterSquareWeightsPower[0] = 2.0f;
      auto maskExtractCenterSquareLayer = network->addScale(
        *maskExtractWidthLayer->getOutput(0),
        ScaleMode::kUNIFORM,
        {DataType::kFLOAT, maskExtractCenterSquareWeightsShift.get(), 1},
        {DataType::kFLOAT, nullptr, 0},
        {DataType::kFLOAT, maskExtractCenterSquareWeightsPower.get(), 1});
      maskExtractCenterSquareLayer->setName("MaskExtract/CenterSquare");
      maskExtractCenterSquareLayer->setPrecision(DataType::kFLOAT);
      model->extraWeights.push_back(move(maskExtractCenterSquareWeightsShift));
      model->extraWeights.push_back(move(maskExtractCenterSquareWeightsPower));

      auto maskExtractQuadWeightsShift = make_unique<float[]>(1);
      auto maskExtractQuadWeightsScale = make_unique<float[]>(1);
      maskExtractQuadWeightsShift[0] = -0.1f;
      maskExtractQuadWeightsScale[0] = 0.01f;
      maskExtractQuadLayer = network->addScale(
        *maskExtractCenterSquareLayer->getOutput(0),
        ScaleMode::kUNIFORM,
        {DataType::kFLOAT, maskExtractQuadWeightsShift.get(), 1},
        {DataType::kFLOAT, maskExtractQuadWeightsScale.get(), 1},
        {DataType::kFLOAT, nullptr, 0});
      maskExtractQuadLayer->setName("MaskExtract/Quad");
      maskExtractQuadLayer->setPrecision(DataType::kFLOAT);
      model->extraWeights.push_back(move(maskExtractQuadWeightsShift));
      model->extraWeights.push_back(move(maskExtractQuadWeightsScale));
    } else {
      float maskExtractWidth = sqrtf(nnXLen * nnYLen);

      auto linLayerWeights = make_unique<float[]>(1);
      linLayerWeights[0] = maskExtractWidth * 0.1f - 1.4f;
      maskExtractLinLayer = network->addConstant({3, {1, 1, 1}}, {DataType::kFLOAT, linLayerWeights.get(), 1});
      maskExtractLinLayer->setName("MaskExtract/Lin");
      model->extraWeights.push_back(move(linLayerWeights));

      auto quadLayerWeights = make_unique<float[]>(1);
      quadLayerWeights[0] = (maskExtractWidth - 14.0f) * (maskExtractWidth - 14.0f) * 0.01f - 0.1f;
      maskExtractQuadLayer = network->addConstant({3, {1, 1, 1}}, {DataType::kFLOAT, quadLayerWeights.get(), 1});
      maskExtractQuadLayer->setName("MaskExtract/Quad");
      model->extraWeights.push_back(move(quadLayerWeights));
    }
  }

  ILayer* buildTrunk(const TrunkDesc* desc) {
    auto& network = model->network;

    string name = desc->name;
    int numChannels = desc->trunkNumChannels;

    tuneDesc += Global::strprintf(
      R"("%s"(%d,%d,%d,%d,%d))",
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

    assert(initialConv->getDimensions().d[0] == numChannels);
    assert(initialMatMul->getDimensions().d[0] == numChannels);

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
    auto p2ConvReshapeLayer = network->addShuffle(*p2ConvLayer->getOutput(0));
    auto p2ConvReshapeLayerName = string(p2ConvLayer->getName()) + "/reshape";
    p2ConvReshapeLayer->setName(p2ConvReshapeLayerName.c_str());
    p2ConvReshapeLayer->setReshapeDimensions({1, {-1}});
    p2ConvReshapeLayer->setPrecision(DataType::kFLOAT);

    markDebugOutput(p2ConvReshapeLayer->getOutput(0), "p2");

    auto gpoolToPassMulLayer = buildMatMulLayer(gpoolLayer->getOutput(0), &desc->gpoolToPassMul, true);
    auto gpoolToPassMulReshapeLayer = network->addShuffle(*gpoolToPassMulLayer->getOutput(0));
    auto gpoolToPassMulReshapeLayerName = string(gpoolToPassMulLayer->getName()) + "/reshape";
    gpoolToPassMulReshapeLayer->setName(gpoolToPassMulReshapeLayerName.c_str());
    gpoolToPassMulReshapeLayer->setReshapeDimensions({1, {-1}});
    gpoolToPassMulReshapeLayer->setPrecision(DataType::kFLOAT);

    markDebugOutput(gpoolToPassMulReshapeLayer->getOutput(0), "p2pass");

    ITensor* concatInputs[] = {p2ConvReshapeLayer->getOutput(0), gpoolToPassMulReshapeLayer->getOutput(0)};
    auto concatLayer = network->addConcatenation(concatInputs, 2);
    auto concatLayerName = name + "/concat";
    concatLayer->setName(concatLayerName.c_str());
    concatLayer->setPrecision(DataType::kFLOAT);

    auto outputPolicy = concatLayer->getOutput(0);
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
    assert(outputValue->getDimensions().d[0] == modelDesc->numValueChannels);
    assert(outputScoreValue->getDimensions().d[0] == modelDesc->numScoreValueChannels);
    assert(outputOwnership->getDimensions().d[0] == modelDesc->numOwnershipChannels);
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

    tuneDesc += Global::strprintf(R"("%s"(%d,%d))", desc->name.c_str(), desc->inChannels, desc->outChannels);

    assert(desc->weights.size() == numInChannels * numOutChannels);
    assert(input->getDimensions().d[0] == numInChannels);

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

    tuneDesc += Global::strprintf(R"("%s"(%d))", desc->name.c_str(), desc->numChannels);

    assert(desc->weights.size() == numChannels);
    assert(input->getDimensions().d[0] == numChannels);

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
      R"("%s"(%d,%d,%d,%d,%d,%d))",
      desc->name.c_str(),
      desc->convXSize,
      desc->convYSize,
      desc->inChannels,
      desc->outChannels,
      desc->dilationX,
      desc->dilationY);

    assert(desc->weights.size() == convYSize * convXSize * numInChannels * numOutChannels);
    assert(input->getDimensions().d[0] == numInChannels);

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

    tuneDesc += Global::strprintf(R"("%s"(%d))", desc->name.c_str(), desc->numChannels);

    assert(desc->mean.size() == numChannels);
    assert(desc->variance.size() == numChannels);
    assert(desc->scale.size() == numChannels);
    assert(desc->bias.size() == numChannels);
    assert(input->getDimensions().d[0] == numChannels);

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
    tuneDesc += Global::strprintf(R"("%s"(%d))", desc->name.c_str(), desc->activation);
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
      gpoolSumLayer = network->addReduce(*inputLayer->getOutput(0), ReduceOperation::kSUM, 1U << 1 | 1U << 2, true);
      auto gpoolSumLayerName = name + "/gpsum";
      gpoolSumLayer->setName(gpoolSumLayerName.c_str());
      gpoolMeanLayer = network->addElementWise(
        *gpoolSumLayer->getOutput(0), *maskExtractSumLayer->getOutput(0), ElementWiseOperation::kDIV);
    } else {
      gpoolMeanLayer = network->addReduce(*inputLayer->getOutput(0), ReduceOperation::kAVG, 1U << 1 | 1U << 2, true);
    }
    auto gpoolMeanLayerName = name + "/gpmean";
    gpoolMeanLayer->setName(gpoolMeanLayerName.c_str());

    auto gpoolLinMeanLayer = network->addElementWise(
      *gpoolMeanLayer->getOutput(0), *maskExtractLinLayer->getOutput(0), ElementWiseOperation::kPROD);
    auto gpoolLinMeanLayerName = name + "/gplinmean";
    gpoolLinMeanLayer->setName(gpoolLinMeanLayerName.c_str());

    ILayer* gpoolMaskAddLayer = nullptr;
    ILayer* gpoolMaskShiftLayer = nullptr;
    ILayer* gpoolConcatInputLayer3 = nullptr;
    if(isValueHead) {
      auto gpoolQuadMeanLayer = network->addElementWise(
        *gpoolMeanLayer->getOutput(0), *maskExtractQuadLayer->getOutput(0), ElementWiseOperation::kPROD);
      auto gpoolQuadMeanLayerName = name + "/gpquadmean";
      gpoolQuadMeanLayer->setName(gpoolQuadMeanLayerName.c_str());
      gpoolConcatInputLayer3 = gpoolQuadMeanLayer;
    } else if(!model->requireExactNNLen) {
      // All activation functions we use right now are always greater than -1.0, and map 0 -> 0.
      // So off-board areas will equal 0, and then this max is mask-safe if we assign -1.0 to off-board areas.
      auto gpoolMaskShiftWeights = make_unique<float[]>(1);
      gpoolMaskShiftWeights[0] = -1.0f;
      gpoolMaskShiftLayer = network->addScale(
        *maskExtractLayer->getOutput(0),
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
        network->addReduce(*gpoolMaskAddLayer->getOutput(0), ReduceOperation::kMAX, 1U << 1 | 1U << 2, true);
      auto gpoolMaxLayerName = name + "/gpmax";
      gpoolMaxLayer->setName(gpoolMaxLayerName.c_str());
      gpoolConcatInputLayer3 = gpoolMaxLayer;
    } else {
      auto gpoolMaxLayer =
        network->addReduce(*inputLayer->getOutput(0), ReduceOperation::kMAX, 1U << 1 | 1U << 2, true);
      auto gpoolMaxLayerName = name + "/gpmax";
      gpoolMaxLayer->setName(gpoolMaxLayerName.c_str());
      gpoolConcatInputLayer3 = gpoolMaxLayer;
    }

    ITensor* gpoolConcatInputs[] = {
      gpoolMeanLayer->getOutput(0), gpoolLinMeanLayer->getOutput(0), gpoolConcatInputLayer3->getOutput(0)};
    auto gpoolConcatLayer = network->addConcatenation(gpoolConcatInputs, 3);
    auto gpoolConcatLayerName = name + "/gpconcat";
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
      gpoolLinMeanLayer->setPrecision(DataType::kFLOAT);
      gpoolConcatInputLayer3->setPrecision(DataType::kFLOAT);
      gpoolConcatLayer->setPrecision(DataType::kFLOAT);
    }

    return gpoolConcatLayer;
  }

  ILayer* applyMaskLayer(ILayer* inputLayer, bool forceFP32 = false) {
    if(!model->requireExactNNLen) {
      auto maskLayer = model->network->addElementWise(
        *inputLayer->getOutput(0), *maskExtractLayer->getOutput(0), ElementWiseOperation::kPROD);
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
  int modelVersion;
  vector<pair<string, Dims>> debugOutputs;

  TRTLogger trtLogger;
  unique_ptr<ICudaEngine> engine;
  unique_ptr<IExecutionContext> exec;
  vector<void*> buffers;
  vector<size_t> bufferBytes;
  vector<size_t> bufferRowElts;

  ComputeHandle(
    Logger* logger,
    const cudaDeviceProp* prop,
    ComputeContext* context,
    const LoadedModel* loadedModel,
    int maxBatchSize,
    bool requireExactNNLen) {
    ctx = context;

    modelVersion = loadedModel->modelDesc.version;

    // Certain minor versions of TensorRT uses a global logger, which is bad.
    // Since TensorRT maintains ABI compatibility between minor versions, a dynamic library mismatch
    // does not necessarily generate a dynamic link error, therefore, an extra check is required.
    if(getInferLibVersion() != NV_TENSORRT_VERSION) {
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

    auto network = unique_ptr<INetworkDefinition>(builder->createNetworkV2(0U));
    if(!network) {
      throw StringError("TensorRT backend: failed to create network definition");
    }
    auto modelBuilder = make_unique<ModelBuilder>();
    auto model = modelBuilder->build(move(network), loadedModel, ctx->nnXLen, ctx->nnYLen, requireExactNNLen);

    debugOutputs = model->debugOutputs;

    bool saveTimingCache;
    string timingCacheFile;
    unique_ptr<ITimingCache> timingCache;
    {
      string cacheDir = HomeData::getHomeDataDir(true, ctx->homeDataDirOverride);
      cacheDir += "/trtcache";
      MakeDir::make(cacheDir);

      char uuid[sizeof(prop->uuid.bytes) * 2 + 1];
      for(int i = 0; i < sizeof(prop->uuid.bytes); i++) {
        sprintf(uuid + i * 2, "%02x", static_cast<unsigned char>(prop->uuid.bytes[i]));
      }
      uuid[sizeof(uuid) - 1] = 0;

      // Truncated to 4 bytes
      char tuneHash[4 * 2 + 1];
      for(int i = 0; i < 4; i++) {
        sprintf(tuneHash + i * 2, "%02x", static_cast<unsigned char>(model->tuneHash[i]));
      }
      tuneHash[sizeof(tuneHash) - 1] = 0;

      timingCacheFile = Global::strprintf(
        "%s/gpu-%s_tune-%s_%dx%d%s_batch%d_fp%d",
        cacheDir.c_str(),
        uuid,
        tuneHash,
        ctx->nnYLen,
        ctx->nnXLen,
        requireExactNNLen ? "-exact" : "",
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

      timingCache.reset(config->createTimingCache(timingCacheBlob.data(), timingCacheBlob.size()));
      bool invalidTimingCache = !config->setTimingCache(*timingCache, false);
      if(invalidTimingCache) {
        logger->write("Invalid timing cache, using new one instead");
        timingCache.reset(config->createTimingCache(nullptr, 0));
        config->setTimingCache(*timingCache, false);
      }
      saveTimingCache = invalidTimingCache || !timingCacheBlob.size();
    }

    // So that there are no concurrent kernel executions probably from other parts of code while profiling
    // See CUDA Runtime API document for more details related to NULL stream and synchronization behaviors
    config->setProfileStream(cudaStreamLegacy);

    // Typical runtime allocation is much less than the 1 GiB specified below
    config->setMemoryPoolLimit(MemoryPoolType::kWORKSPACE, 1U << 30);

    builder->setMaxBatchSize(maxBatchSize);

    auto plan = unique_ptr<IHostMemory>(builder->buildSerializedNetwork(*model->network, *config));
    if(!plan) {
      throw StringError("TensorRT backend: failed to create plan");
    }

    if(saveTimingCache) {
      auto serializedTimingCache = unique_ptr<IHostMemory>(config->getTimingCache()->serialize());
      ofstream ofs;
      FileUtils::open(ofs, timingCacheFile, ios::out | ios::binary);
      ofs.write(static_cast<char*>(serializedTimingCache->data()), serializedTimingCache->size());
      ofs.close();
      logger->write("Saved new timing cache to " + timingCacheFile);
    }

    auto runtime = unique_ptr<IRuntime>(createInferRuntime(trtLogger));
    if(!runtime) {
      throw StringError("TensorRT backend: failed to create runtime");
    }

    engine.reset(runtime->deserializeCudaEngine(plan->data(), plan->size()));
    if(!engine) {
      throw StringError("TensorRT backend: failed to create cuda engine");
    }
    exec.reset(engine->createExecutionContext());
    if(!exec) {
      throw StringError("TensorRT backend: failed to create execution context");
    }

    int numBindings = engine->getNbBindings();
    buffers.resize(numBindings, nullptr);
    bufferBytes.resize(numBindings, 0);
    bufferRowElts.resize(numBindings, 0);
    for(int i = 0; i < numBindings; i++) {
      auto dims = engine->getBindingDimensions(i);
      size_t elts = accumulate(dims.d, dims.d + dims.nbDims, 1, multiplies<int>());
      size_t bytes = maxBatchSize * elts * sizeof(float);
      bufferRowElts[i] = elts;
      bufferBytes[i] = bytes;
      CUDA_ERR("ComputeHandle", cudaMalloc(&buffers[i], bytes));
    }
  }

  ~ComputeHandle() {
    for(auto ptr: buffers) {
      CUDA_ERR("~ComputeHandle", cudaFree(ptr));
    }
  }

  ComputeHandle() = delete;
  ComputeHandle(const ComputeHandle&) = delete;
  ComputeHandle& operator=(const ComputeHandle&) = delete;

  void* getBuffer(const char* name) {
    int index = engine->getBindingIndex(name);
    if(index == -1) {
      throw StringError(Global::strprintf("ComputeHandle: unknown binding name %s", name));
    }
    return buffers[index];
  }

  size_t getBufferBytes(const char* name) {
    int index = engine->getBindingIndex(name);
    if(index == -1) {
      throw StringError(Global::strprintf("ComputeHandle: unknown binding name %s", name));
    }
    return bufferBytes[index];
  }

  size_t getBufferRowElts(const char* name) {
    int index = engine->getBindingIndex(name);
    if(index == -1) {
      throw StringError(Global::strprintf("ComputeHandle: unknown binding name %s", name));
    }
    return bufferRowElts[index];
  }

  void printDebugOutput(int batchSize) {
    for(auto& debugOutput: debugOutputs) {
      Dims dims = debugOutput.second;
      string desc = debugOutput.first;
      string name = "DBG" + to_string(hash<string>{}(debugOutput.first));

      vector<float> values(batchSize * getBufferRowElts(name.c_str()));
      CUDA_ERR(
        "printDebugOutput",
        cudaMemcpy(values.data(), getBuffer(name.c_str()), values.size() * sizeof(float), cudaMemcpyDeviceToHost));

      cout << "=========================================================" << endl;
      cout << desc << endl;
      int i = 0;
      if(dims.nbDims == 1) {
        for(int n = 0; n < batchSize; n++) {
          cout << "-(n=" << n << ")--------------------" << endl;
          for(int c = 0; c < dims.d[0]; c++) {
            cout << values[i++] << " ";
          }
          cout << endl;
        }
        cout << endl;
      } else if(dims.nbDims == 3) {
        for(int n = 0; n < batchSize; n++) {
          cout << "-(n=" << n << ")--------------------" << endl;
          for(int c = 0; c < dims.d[0]; c++) {
            cout << "(c=" << c << ")" << endl;
            for(int y = 0; y < dims.d[1]; y++) {
              for(int x = 0; x < dims.d[2]; x++)
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

bool NeuralNet::isUsingFP16(const ComputeHandle* handle) {
  return handle->usingFP16;
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

  size_t singleInputElts;
  size_t singleInputBytes;
  size_t singleInputGlobalElts;
  size_t singleInputGlobalBytes;
  size_t singlePolicyResultElts;
  size_t singlePolicyResultBytes;
  size_t singleValueResultElts;
  size_t singleValueResultBytes;
  size_t singleScoreValueResultElts;
  size_t singleScoreValueResultBytes;
  size_t singleOwnershipResultElts;
  size_t singleOwnershipResultBytes;

  size_t inputBufferBytes;
  size_t inputGlobalBufferBytes;
  size_t policyResultBufferBytes;
  size_t valueResultBufferBytes;
  size_t scoreValueResultBufferBytes;
  size_t ownershipResultBufferBytes;

  unique_ptr<float[]> inputBuffer;        // Host pointer
  unique_ptr<float[]> inputGlobalBuffer;  // Host pointer
  unique_ptr<float[]> policyResults;      // Host pointer
  unique_ptr<float[]> valueResults;       // Host pointer
  unique_ptr<float[]> scoreValueResults;  // Host pointer
  unique_ptr<float[]> ownershipResults;   // Host pointer

  InputBuffers(const LoadedModel* loadedModel, int maxBatchSz, int nnXLen, int nnYLen) {
    const ModelDesc& m = loadedModel->modelDesc;

    int xSize = nnXLen;
    int ySize = nnYLen;

    if(nnXLen > NNPos::MAX_BOARD_LEN)
      throw StringError(
        Global::strprintf("nnXLen (%d) is greater than NNPos::MAX_BOARD_LEN (%d)", nnXLen, NNPos::MAX_BOARD_LEN));
    if(nnYLen > NNPos::MAX_BOARD_LEN)
      throw StringError(
        Global::strprintf("nnYLen (%d) is greater than NNPos::MAX_BOARD_LEN (%d)", nnYLen, NNPos::MAX_BOARD_LEN));

    maxBatchSize = maxBatchSz;
    singleInputElts = m.numInputChannels * xSize * ySize;
    singleInputBytes = singleInputElts * sizeof(float);
    singleInputGlobalElts = m.numInputGlobalChannels;
    singleInputGlobalBytes = singleInputGlobalElts * sizeof(float);
    singlePolicyResultElts = NNPos::getPolicySize(xSize, ySize);
    singlePolicyResultBytes = singlePolicyResultElts * sizeof(float);
    singleValueResultElts = m.numValueChannels;
    singleValueResultBytes = singleValueResultElts * sizeof(float);
    singleScoreValueResultElts = m.numScoreValueChannels;
    singleScoreValueResultBytes = singleScoreValueResultElts * sizeof(float);
    singleOwnershipResultElts = m.numOwnershipChannels * xSize * ySize;
    singleOwnershipResultBytes = singleOwnershipResultElts * sizeof(float);

    assert(NNModelVersion::getNumSpatialFeatures(m.version) == m.numInputChannels);
    assert(NNModelVersion::getNumGlobalFeatures(m.version) == m.numInputGlobalChannels);

    inputBufferBytes = maxBatchSize * singleInputBytes;
    inputGlobalBufferBytes = maxBatchSize * singleInputGlobalBytes;
    policyResultBufferBytes = maxBatchSize * singlePolicyResultBytes;
    valueResultBufferBytes = maxBatchSize * singleValueResultBytes;
    scoreValueResultBufferBytes = maxBatchSize * singleScoreValueResultBytes;
    ownershipResultBufferBytes = maxBatchSize * singleOwnershipResultBytes;

    inputBuffer = make_unique<float[]>(maxBatchSize * singleInputElts);
    inputGlobalBuffer = make_unique<float[]>(maxBatchSize * singleInputGlobalElts);
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
    float* rowSpatialInput = &inputBuffers->inputBuffer[inputBuffers->singleInputElts * nIdx];
    float* rowGlobalInput = &inputBuffers->inputGlobalBuffer[inputBuffers->singleInputGlobalElts * nIdx];

    const float* rowGlobal = inputBufs[nIdx]->rowGlobal;
    const float* rowSpatial = inputBufs[nIdx]->rowSpatial;
    copy(rowGlobal, rowGlobal + numGlobalFeatures, rowGlobalInput);
    SymmetryHelpers::copyInputsWithSymmetry(
      rowSpatial, rowSpatialInput, 1, nnYLen, nnXLen, numSpatialFeatures, false, inputBufs[nIdx]->symmetry);
  }

  assert(inputBuffers->singleInputElts == gpuHandle->getBufferRowElts("InputFeature"));
  assert(inputBuffers->singleInputGlobalElts == gpuHandle->getBufferRowElts("InputGlobalFeature"));
  assert(inputBuffers->singlePolicyResultElts == gpuHandle->getBufferRowElts("OutputPolicy"));
  assert(inputBuffers->singleValueResultElts == gpuHandle->getBufferRowElts("OutputValue"));
  assert(inputBuffers->singleScoreValueResultElts == gpuHandle->getBufferRowElts("OutputScoreValue"));
  assert(inputBuffers->singleOwnershipResultElts == gpuHandle->getBufferRowElts("OutputOwnership"));

  assert(inputBuffers->inputBufferBytes == gpuHandle->getBufferBytes("InputFeature"));
  assert(inputBuffers->inputGlobalBufferBytes == gpuHandle->getBufferBytes("InputGlobalFeature"));
  assert(inputBuffers->policyResultBufferBytes == gpuHandle->getBufferBytes("OutputPolicy"));
  assert(inputBuffers->valueResultBufferBytes == gpuHandle->getBufferBytes("OutputValue"));
  assert(inputBuffers->scoreValueResultBufferBytes == gpuHandle->getBufferBytes("OutputScoreValue"));
  assert(inputBuffers->ownershipResultBufferBytes == gpuHandle->getBufferBytes("OutputOwnership"));

  // Transfers from host memory to device memory are asynchronous with respect to the host
  CUDA_ERR(
    "getOutput",
    cudaMemcpyAsync(
      gpuHandle->getBuffer("InputFeature"),
      inputBuffers->inputBuffer.get(),
      inputBuffers->singleInputBytes * batchSize,
      cudaMemcpyHostToDevice));
  CUDA_ERR(
    "getOutput",
    cudaMemcpyAsync(
      gpuHandle->getBuffer("InputGlobalFeature"),
      inputBuffers->inputGlobalBuffer.get(),
      inputBuffers->singleInputGlobalBytes * batchSize,
      cudaMemcpyHostToDevice));

  gpuHandle->exec->enqueue(batchSize, gpuHandle->buffers.data(), cudaStreamPerThread, nullptr);

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

  for(int row = 0; row < batchSize; row++) {
    NNOutput* output = outputs[row];

    assert(output->nnXLen == nnXLen);
    assert(output->nnYLen == nnYLen);

    const float* policySrcBuf = &inputBuffers->policyResults[row * inputBuffers->singlePolicyResultElts];
    float* policyProbs = output->policyProbs;

    // These are not actually correct, the client does the postprocessing to turn them into
    // policy probabilities and white game outcome probabilities
    // Also we don't fill in the nnHash here either
    SymmetryHelpers::copyOutputsWithSymmetry(policySrcBuf, policyProbs, 1, nnYLen, nnXLen, inputBufs[row]->symmetry);
    policyProbs[inputBuffers->singlePolicyResultElts - 1] = policySrcBuf[inputBuffers->singlePolicyResultElts - 1];

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
