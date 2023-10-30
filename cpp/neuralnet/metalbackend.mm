#import "metalbackend.h"
#import "metalswift.h"

using namespace katago;

/// Converts a ConvLayerDesc instance from C++ to Swift by creating a new SWConvLayerDesc instance with the same properties.
/// - Parameter desc: The ConvLayerDesc instance to convert.
/// - Returns: A SWConvLayerDesc instance with the same properties as the input ConvLayerDesc.
static SWConvLayerDesc convLayerDescToSwift(const ConvLayerDesc * desc) {

    SWConvLayerDesc swDesc = createSWConvLayerDesc(desc->convYSize,
                                                   desc->convXSize,
                                                   desc->inChannels,
                                                   desc->outChannels,
                                                   desc->dilationY,
                                                   desc->dilationX,
                                                   (float*)desc->weights.data());

    return swDesc;
}

/// Converts a BatchNormLayerDesc instance from C++ to Swift by creating a new SWBatchNormLayerDesc instance with the same properties.
/// - Parameter desc: The BatchNormLayerDesc instance to convert.
/// - Returns: A SWBatchNormLayerDesc instance with the same properties as the input BatchNormLayerDesc.
static SWBatchNormLayerDesc batchNormLayerDescToSwift(const BatchNormLayerDesc * desc) {

    SWBatchNormLayerDesc swDesc =
    createSWBatchNormLayerDesc(desc->numChannels,
                               desc->epsilon,
                               desc->hasScale,
                               desc->hasBias,
                               (float*)desc->mean.data(),
                               (float*)desc->variance.data(),
                               (float*)desc->scale.data(),
                               (float*)desc->bias.data());

    return swDesc;
}

/// Convert an activation layer description from C++ to Swift
/// - Parameter desc: An activation layer description
static ActivationKind activationLayerDescToSwift(const ActivationLayerDesc * desc) {

    switch (desc->activation) {
        case ACTIVATION_RELU:
            return ActivationKind::relu();
        case ACTIVATION_MISH:
            return ActivationKind::mish();
        default:
            return ActivationKind::identity();
    }
}

/// Convert a residual block description from C++ to Swift
/// - Parameter desc: A residual block description
/// - Returns: The residual block description converted to SWResidualBlockDesc
static SWResidualBlockDesc residualBlockDescToSwift(const ResidualBlockDesc * desc) {

    SWBatchNormLayerDesc preBN = batchNormLayerDescToSwift(&desc->preBN);
    ActivationKind preActivationKind = activationLayerDescToSwift(&desc->preActivation);
    SWConvLayerDesc regularConv = convLayerDescToSwift(&desc->regularConv);
    SWBatchNormLayerDesc midBN = batchNormLayerDescToSwift(&desc->midBN);
    ActivationKind midActivationKind = activationLayerDescToSwift(&desc->midActivation);
    SWConvLayerDesc finalConv = convLayerDescToSwift(&desc->finalConv);

    SWResidualBlockDesc swDesc =
    createSWResidualBlockDesc(preBN,
                              preActivationKind,
                              regularConv,
                              midBN,
                              midActivationKind,
                              finalConv);

    return swDesc;
}

/// Convert a matrix multiplication layer description from C++ to Swift
/// - Parameter desc: A matrix multiplication layer description
/// - Returns: The matrix multiplication layer description converted to SWMatMulLayerDesc
static SWMatMulLayerDesc matMulLayerDescToSwift(const MatMulLayerDesc * desc) {

    SWMatMulLayerDesc swDesc = createSWMatMulLayerDesc(desc->inChannels,
                                                       desc->outChannels,
                                                       (float*)desc->weights.data());

    return swDesc;
}

/// Convert a global pooling residual block description from C++ to Swift
/// - Parameter desc: A global pooling residual block description
/// - Returns: The global pooling residual block description converted to SWGlobalPoolingResidualBlockDesc
static SWGlobalPoolingResidualBlockDesc globalPoolingResidualBlockDescToSwift(const GlobalPoolingResidualBlockDesc* desc) {

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

    SWGlobalPoolingResidualBlockDesc swDesc =
    createSWGlobalPoolingResidualBlockDesc(preBN,
                                           preActivationKind,
                                           regularConv,
                                           gpoolConv,
                                           gpoolBN,
                                           gpoolActivationKind,
                                           gpoolToBiasMul,
                                           midBN,
                                           midActivationKind,
                                           finalConv);

    return swDesc;
}

static swift::Array<BlockDescriptor> residualBlocksToSwift(const std::vector<std::pair<int, unique_ptr_void>>& blocks);
static SWNestedBottleneckResidualBlockDesc nestedBottleneckResidualBlockDescToSwift(const NestedBottleneckResidualBlockDesc* desc);

/// Convert residual blocks from C++ to Swift
/// - Parameters:
///   - blocks: Residual blocks
///   - swBlocks: A pointer to an array of BlockDescriptor
static swift::Array<BlockDescriptor> residualBlocksToSwift(const std::vector<std::pair<int, unique_ptr_void>>& blocks) {

    auto builder = createBlockDescriptorBuilder();

    for (int i = 0; i < blocks.size(); i++) {

        void * blockDesc = blocks[i].second.get();

        if (blocks[i].first == GLOBAL_POOLING_BLOCK_KIND) {
            BlockDescriptor descriptor = globalPoolingResidualBlockDescToSwift((GlobalPoolingResidualBlockDesc*)blockDesc);
            builder.enque(descriptor);
        } else if (blocks[i].first == NESTED_BOTTLENECK_BLOCK_KIND) {
            BlockDescriptor descriptor = nestedBottleneckResidualBlockDescToSwift((NestedBottleneckResidualBlockDesc*)blockDesc);
            builder.enque(descriptor);
        } else {
            BlockDescriptor descriptor = residualBlockDescToSwift((ResidualBlockDesc*)blockDesc);
            builder.enque(descriptor);
        }
    }

    return builder.getBlockDescriptors();
}

/// Convert a nested bottleneck residual block description from C++ to Swift
/// - Parameter desc: A nested bottleneck residual block description
static SWNestedBottleneckResidualBlockDesc nestedBottleneckResidualBlockDescToSwift(const NestedBottleneckResidualBlockDesc* desc) {

    SWBatchNormLayerDesc preBN = batchNormLayerDescToSwift(&desc->preBN);
    ActivationKind preActivationKind = activationLayerDescToSwift(&desc->preActivation);
    SWConvLayerDesc preConv = convLayerDescToSwift(&desc->preConv);
    auto swBlocks = residualBlocksToSwift(desc->blocks);
    SWBatchNormLayerDesc postBN = batchNormLayerDescToSwift(&desc->postBN);
    ActivationKind postActivationKind = activationLayerDescToSwift(&desc->postActivation);
    SWConvLayerDesc postConv = convLayerDescToSwift(&desc->postConv);

    SWNestedBottleneckResidualBlockDesc swDesc =
    createSWNestedBottleneckResidualBlockDesc(preBN,
                                              preActivationKind,
                                              preConv,
                                              swBlocks,
                                              postBN,
                                              postActivationKind,
                                              postConv);

    return swDesc;
}

/// Convert a trunk description from C++ to Swift
/// - Parameter trunk: A trunk description
/// - Returns: The trunk description converted to SWTrunkDesc
static SWTrunkDesc trunkDescToSwift(const TrunkDesc * trunk) {

    SWConvLayerDesc initialConv = convLayerDescToSwift(&trunk->initialConv);
    SWMatMulLayerDesc initialMatMul = matMulLayerDescToSwift(&trunk->initialMatMul);
    auto swBlocks = residualBlocksToSwift(trunk->blocks);
    SWBatchNormLayerDesc trunkTipBN = batchNormLayerDescToSwift(&trunk->trunkTipBN);
    ActivationKind trunkTipActivation = activationLayerDescToSwift(&trunk->trunkTipActivation);

    SWTrunkDesc swTrunkDesc = createSWTrunkDesc(trunk->version,
                                                trunk->trunkNumChannels,
                                                trunk->midNumChannels,
                                                trunk->regularNumChannels,
                                                trunk->gpoolNumChannels,
                                                initialConv,
                                                initialMatMul,
                                                swBlocks,
                                                trunkTipBN,
                                                trunkTipActivation);

    return swTrunkDesc;
}

/// Convert a policy head description from C++ to Swift
/// - Parameter policyHead: A policy head description
/// - Returns: The policy head description converted to SWPolicyHeadDesc
static SWPolicyHeadDesc policyHeadDescToSwift(const PolicyHeadDesc * policyHead) {

    SWConvLayerDesc p1Conv = convLayerDescToSwift(&policyHead->p1Conv);
    SWConvLayerDesc g1Conv = convLayerDescToSwift(&policyHead->g1Conv);
    SWBatchNormLayerDesc g1BN = batchNormLayerDescToSwift(&policyHead->g1BN);
    ActivationKind g1Activation = activationLayerDescToSwift(&policyHead->g1Activation);
    SWMatMulLayerDesc gpoolToBiasMul = matMulLayerDescToSwift(&policyHead->gpoolToBiasMul);
    SWBatchNormLayerDesc p1BN = batchNormLayerDescToSwift(&policyHead->p1BN);
    ActivationKind p1Activation = activationLayerDescToSwift(&policyHead->p1Activation);
    SWConvLayerDesc p2Conv = convLayerDescToSwift(&policyHead->p2Conv);
    SWMatMulLayerDesc gpoolToPassMul = matMulLayerDescToSwift(&policyHead->gpoolToPassMul);

    SWPolicyHeadDesc swPolicyHead = createSWPolicyHeadDesc(policyHead->version,
                                                           p1Conv,
                                                           g1Conv,
                                                           g1BN,
                                                           g1Activation,
                                                           gpoolToBiasMul,
                                                           p1BN,
                                                           p1Activation,
                                                           p2Conv,
                                                           gpoolToPassMul);

    return swPolicyHead;
}

/// Convert a matrix bias layer description from C++ to Swift
/// - Parameter desc: A matrix bias layer description
/// - Returns: The matrix bias layer description converted to SWMatBiasLayerDesc
static SWMatBiasLayerDesc matBiasLayerDescToSwift(const MatBiasLayerDesc * desc) {

    SWMatBiasLayerDesc swDesc = createSWMatBiasLayerDesc(desc->numChannels, (float*)desc->weights.data());

    return swDesc;
}

/// Convert a value head description from C++ to Swift
/// - Parameter valueHead: A value head description
/// - Returns: The value head description converted to SWValueHeadDesc
static SWValueHeadDesc valueHeadDescToSwift(const ValueHeadDesc * valueHead) {

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

    SWValueHeadDesc swDesc = createSWValueHeadDesc(valueHead->version,
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

    return swDesc;
}

/// Create a Metal context
/// - Parameters:
///   - nnXLen: The width of the neural network input
///   - nnYLen: The height of the neural network input
///   - inputUseFP16Mode: Whether to use FP16 mode
///   - inputUseNHWCMode: Whether to use NHWC mode
void MetalProcess::createMetalContext(int nnXLen,
                                      int nnYLen,
                                      enabled_t inputUseFP16Mode,
                                      enabled_t inputUseNHWCMode) {
    SWEnable useFP16Mode =
    (inputUseFP16Mode == enabled_t::False) ? SWEnable::False() :
    (inputUseFP16Mode == enabled_t::True) ? SWEnable::True() :
    SWEnable::Auto();

    SWEnable useNHWCMode =
    (inputUseNHWCMode == enabled_t::False) ? SWEnable::False() :
    (inputUseNHWCMode == enabled_t::True) ? SWEnable::True() :
    SWEnable::Auto();

    createMetalComputeContext(nnXLen, nnYLen, useFP16Mode, useNHWCMode);
}

/// Create a Metal handle
/// - Parameters:
///   - gpuIdxForThisThread: The GPU index for this thread
///   - desc: The model description
///   - serverThreadIdx: The server thread index
void MetalProcess::createMetalHandle(int gpuIdxForThisThread,
                                     const ModelDesc* desc,
                                     int serverThreadIdx) {

    SWModelDesc swModelDesc = createSWModelDesc(desc->version,
                                                swift::String(desc->name),
                                                desc->numInputChannels,
                                                desc->numInputGlobalChannels,
                                                desc->numValueChannels,
                                                desc->numScoreValueChannels,
                                                desc->numOwnershipChannels,
                                                trunkDescToSwift(&desc->trunk),
                                                policyHeadDescToSwift(&desc->policyHead),
                                                valueHeadDescToSwift(&desc->valueHead));

    createMetalComputeHandle(gpuIdxForThisThread, swModelDesc, serverThreadIdx);
}

/// Evaluate a convolutional layer using Metal API for testing purposes
/// - Parameters:
///   - desc: The convolutional layer description
///   - nnXLen: The width of the neural network input
///   - nnYLen: The height of the neural network input
///   - batchSize: The batch size
///   - input: The pointer to the input
///   - output: The pointer to the output
void testMetalEvaluateConv(const ConvLayerDesc* desc,
                           int nnXLen,
                           int nnYLen,
                           int batchSize,
                           float* input,
                           float* output) {
    testConvLayer(convLayerDescToSwift(desc), nnXLen, nnYLen, batchSize, input, output);
}

/// Evaluate a batch normalization layer using Metal API for testing purposes
/// - Parameters:
///   - desc: The batch normalization layer description
///   - nnXLen: The width of the neural network input
///   - nnYLen: The height of the neural network input
///   - batchSize: The batch size
///   - input: The pointer to the input
///   - mask: The pointer to the mask
///   - output: The pointer to the output
void testMetalEvaluateBatchNorm(const BatchNormLayerDesc* desc,
                                int nnXLen,
                                int nnYLen,
                                int batchSize,
                                float* input,
                                float* mask,
                                float* output) {
    testBatchNormLayer(batchNormLayerDescToSwift(desc), nnXLen, nnYLen, batchSize, input, mask, output);
}

/// Evaluate a residual block using Metal API for testing purposes
/// - Parameters:
///   - desc: The residual block description
///   - batchSize: The batch size
///   - nnXLen: The width of the neural network input
///   - nnYLen: The height of the neural network input
///   - input: The pointer to the input
///   - mask: The pointer to the mask
///   - output: The pointer to the output
void testMetalEvaluateResidualBlock(const ResidualBlockDesc* desc,
                                    int batchSize,
                                    int nnXLen,
                                    int nnYLen,
                                    float* input,
                                    float* mask,
                                    float* output) {
    testResidualBlock(residualBlockDescToSwift(desc), batchSize, nnXLen, nnYLen, input, mask, output);
}

/// Evaluate a global pooling residual block using Metal API for testing purposes
/// - Parameters:
///   - desc: The global pooling residual block description
///   - batchSize: The batch size
///   - nnXLen: The width of the neural network input
///   - nnYLen: The height of the neural network input
///   - input: The pointer to the input
///   - mask: The pointer to the mask
///   - output: The pointer to the output
void testMetalEvaluateGlobalPoolingResidualBlock(const GlobalPoolingResidualBlockDesc* desc,
                                                 int batchSize,
                                                 int nnXLen,
                                                 int nnYLen,
                                                 float* input,
                                                 float* mask,
                                                 float* output) {
    testGlobalPoolingResidualBlock(globalPoolingResidualBlockDescToSwift(desc),
                                   batchSize,
                                   nnXLen,
                                   nnYLen,
                                   input,
                                   mask,
                                   output);
}
