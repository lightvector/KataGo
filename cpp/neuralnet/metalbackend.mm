#import "metalbackend.h"
#import "metalswift.h"

/// Converts a ConvLayerDesc instance from C++ to Swift by creating a new SWConvLayerDesc instance with the same properties.
/// - Parameter desc: The ConvLayerDesc instance to convert.
/// - Returns: A SWConvLayerDesc instance with the same properties as the input ConvLayerDesc.
static SWConvLayerDesc * convLayerDescToSwift(const ConvLayerDesc * desc) {

    SWConvLayerDesc * swDesc =
    [[SWConvLayerDesc alloc] initWithConvYSize:[NSNumber numberWithInt:desc->convYSize]
                                     convXSize:[NSNumber numberWithInt:desc->convXSize]
                                    inChannels:[NSNumber numberWithInt:desc->inChannels]
                                   outChannels:[NSNumber numberWithInt:desc->outChannels]
                                     dilationY:desc->dilationY
                                     dilationX:desc->dilationX
                                       weights:(float*)desc->weights.data()];

    return swDesc;
}

/// Converts a BatchNormLayerDesc instance from C++ to Swift by creating a new SWBatchNormLayerDesc instance with the same properties.
/// - Parameter desc: The BatchNormLayerDesc instance to convert.
/// - Returns: A SWBatchNormLayerDesc instance with the same properties as the input BatchNormLayerDesc.
static SWBatchNormLayerDesc * batchNormLayerDescToSwift(const BatchNormLayerDesc * desc) {

    SWBatchNormLayerDesc * swDesc =
    [[SWBatchNormLayerDesc alloc] initWithNumChannels:[NSNumber numberWithInt:desc->numChannels]
                                              epsilon:desc->epsilon
                                             hasScale:[NSNumber numberWithBool:desc->hasScale]
                                              hasBias:[NSNumber numberWithBool:desc->hasBias]
                                                 mean:(float*)desc->mean.data()
                                             variance:(float*)desc->variance.data()
                                                scale:(float*)desc->scale.data()
                                                 bias:(float*)desc->bias.data()];

    return swDesc;
}

/// Convert an activation layer description from C++ to Swift
/// - Parameter desc: An activation layer description
static ActivationKind activationLayerDescToSwift(const ActivationLayerDesc * desc) {

    ActivationKind activationKind;

    switch (desc->activation) {
        case ACTIVATION_RELU:
            activationKind = ActivationKindRelu;
            break;
        case ACTIVATION_MISH:
            activationKind = ActivationKindMish;
            break;
        default:
            activationKind = ActivationKindIdentity;
            break;
    }

    return activationKind;
}

/// Convert a residual block description from C++ to Swift
/// - Parameter desc: A residual block description
/// - Returns: The residual block description converted to SWResidualBlockDesc
static SWResidualBlockDesc * residualBlockDescToSwift(const ResidualBlockDesc * desc) {

    SWBatchNormLayerDesc * preBN = batchNormLayerDescToSwift(&desc->preBN);
    ActivationKind preActivationKind = activationLayerDescToSwift(&desc->preActivation);
    SWConvLayerDesc * regularConv = convLayerDescToSwift(&desc->regularConv);
    SWBatchNormLayerDesc * midBN = batchNormLayerDescToSwift(&desc->midBN);
    ActivationKind midActivationKind = activationLayerDescToSwift(&desc->midActivation);
    SWConvLayerDesc * finalConv = convLayerDescToSwift(&desc->finalConv);

    SWResidualBlockDesc * swDesc = [[SWResidualBlockDesc alloc] initWithPreBN:preBN
                                                                preActivation:preActivationKind
                                                                  regularConv:regularConv
                                                                        midBN:midBN
                                                                midActivation:midActivationKind
                                                                    finalConv:finalConv];

    return swDesc;
}

/// Convert a matrix multiplication layer description from C++ to Swift
/// - Parameter desc: A matrix multiplication layer description
/// - Returns: The matrix multiplication layer description converted to SWMatMulLayerDesc
static SWMatMulLayerDesc * matMulLayerDescToSwift(const MatMulLayerDesc * desc) {

    SWMatMulLayerDesc * swDesc =
    [[SWMatMulLayerDesc alloc] initInChannels:[NSNumber numberWithInt:desc->inChannels]
                                  outChannels:[NSNumber numberWithInt:desc->outChannels]
                                      weights:(float*)desc->weights.data()];

    return swDesc;
}

/// Convert a global pooling residual block description from C++ to Swift
/// - Parameter desc: A global pooling residual block description
/// - Returns: The global pooling residual block description converted to SWGlobalPoolingResidualBlockDesc
static SWGlobalPoolingResidualBlockDesc* globalPoolingResidualBlockDescToSwift(const GlobalPoolingResidualBlockDesc* desc) {

    SWBatchNormLayerDesc * preBN = batchNormLayerDescToSwift(&desc->preBN);
    ActivationKind preActivationKind = activationLayerDescToSwift(&desc->preActivation);
    SWConvLayerDesc * regularConv = convLayerDescToSwift(&desc->regularConv);
    SWConvLayerDesc * gpoolConv = convLayerDescToSwift(&desc->gpoolConv);
    SWBatchNormLayerDesc * gpoolBN = batchNormLayerDescToSwift(&desc->gpoolBN);
    ActivationKind gpoolActivationKind = activationLayerDescToSwift(&desc->gpoolActivation);
    SWMatMulLayerDesc * gpoolToBiasMul = matMulLayerDescToSwift(&desc->gpoolToBiasMul);
    SWBatchNormLayerDesc * midBN = batchNormLayerDescToSwift(&desc->midBN);
    ActivationKind midActivationKind = activationLayerDescToSwift(&desc->midActivation);
    SWConvLayerDesc * finalConv = convLayerDescToSwift(&desc->finalConv);

    SWGlobalPoolingResidualBlockDesc * swDesc =
    [[SWGlobalPoolingResidualBlockDesc alloc] initWithPreBN:preBN
                                              preActivation:preActivationKind
                                                regularConv:regularConv
                                                  gpoolConv:gpoolConv
                                                    gpoolBN:gpoolBN
                                            gpoolActivation:gpoolActivationKind
                                             gpoolToBiasMul:gpoolToBiasMul
                                                      midBN:midBN
                                              midActivation:midActivationKind
                                                  finalConv:finalConv];

    return swDesc;
}

static void residualBlocksToSwift(const std::vector<std::pair<int, unique_ptr_void>>& blocks, NSMutableArray<BlockDescriptor *> * swBlocks);
static SWNestedBottleneckResidualBlockDesc* nestedBottleneckResidualBlockDescToSwift(const NestedBottleneckResidualBlockDesc* desc);

/// Convert residual blocks from C++ to Swift
/// - Parameters:
///   - blocks: Residual blocks
///   - swBlocks: A pointer to an array of BlockDescriptor
static void residualBlocksToSwift(const std::vector<std::pair<int, unique_ptr_void>>& blocks, NSMutableArray<BlockDescriptor *> * swBlocks) {

    for (int i = 0; i < blocks.size(); i++) {

        BlockDescriptor * swBlockDesc;
        void * blockDesc = blocks[i].second.get();

        if (blocks[i].first == GLOBAL_POOLING_BLOCK_KIND) {
            swBlockDesc = globalPoolingResidualBlockDescToSwift((GlobalPoolingResidualBlockDesc*)blockDesc);
        } else if (blocks[i].first == NESTED_BOTTLENECK_BLOCK_KIND) {
            swBlockDesc = nestedBottleneckResidualBlockDescToSwift((NestedBottleneckResidualBlockDesc*)blockDesc);
        } else {
            swBlockDesc = residualBlockDescToSwift((ResidualBlockDesc*)blockDesc);
        }

        [swBlocks addObject:swBlockDesc];
    }
}

/// Convert a nested bottleneck residual block description from C++ to Swift
/// - Parameter desc: A nested bottleneck residual block description
static SWNestedBottleneckResidualBlockDesc* nestedBottleneckResidualBlockDescToSwift(const NestedBottleneckResidualBlockDesc* desc) {

    SWBatchNormLayerDesc * preBN = batchNormLayerDescToSwift(&desc->preBN);
    ActivationKind preActivationKind = activationLayerDescToSwift(&desc->preActivation);
    SWConvLayerDesc * preConv = convLayerDescToSwift(&desc->preConv);
    NSMutableArray<BlockDescriptor *> * swBlocks = [[NSMutableArray alloc] init];
    residualBlocksToSwift(desc->blocks, swBlocks);
    SWBatchNormLayerDesc * postBN = batchNormLayerDescToSwift(&desc->postBN);
    ActivationKind postActivationKind = activationLayerDescToSwift(&desc->postActivation);
    SWConvLayerDesc * postConv = convLayerDescToSwift(&desc->postConv);

    SWNestedBottleneckResidualBlockDesc * swDesc =
    [[SWNestedBottleneckResidualBlockDesc alloc] initWithPreBN:preBN
                                                 preActivation:preActivationKind
                                                       preConv:preConv
                                              blockDescriptors:swBlocks
                                                        postBN:postBN
                                                postActivation:postActivationKind
                                                      postConv:postConv];

    return swDesc;
}

/// Convert a trunk description from C++ to Swift
/// - Parameter trunk: A trunk description
/// - Returns: The trunk description converted to SWTrunkDesc
static SWTrunkDesc * trunkDescToSwift(const TrunkDesc * trunk) {

    SWConvLayerDesc * initialConv = convLayerDescToSwift(&trunk->initialConv);
    SWMatMulLayerDesc * initialMatMul = matMulLayerDescToSwift(&trunk->initialMatMul);
    NSMutableArray<BlockDescriptor *> * swBlocks = [[NSMutableArray alloc] init];
    residualBlocksToSwift(trunk->blocks, swBlocks);
    SWBatchNormLayerDesc * trunkTipBN = batchNormLayerDescToSwift(&trunk->trunkTipBN);
    ActivationKind trunkTipActivation = activationLayerDescToSwift(&trunk->trunkTipActivation);

    SWTrunkDesc * swTrunkDesc =
    [[SWTrunkDesc alloc] initWithVersion:trunk->version
                        trunkNumChannels:[NSNumber numberWithInt:trunk->trunkNumChannels]
                          midNumChannels:[NSNumber numberWithInt:trunk->midNumChannels]
                      regularNumChannels:[NSNumber numberWithInt:trunk->regularNumChannels]
                        gpoolNumChannels:[NSNumber numberWithInt:trunk->gpoolNumChannels]
                             initialConv:initialConv
                           initialMatMul:initialMatMul
                        blockDescriptors:swBlocks
                              trunkTipBN:trunkTipBN
                      trunkTipActivation:trunkTipActivation];

    return swTrunkDesc;
}

/// Convert a policy head description from C++ to Swift
/// - Parameter policyHead: A policy head description
/// - Returns: The policy head description converted to SWPolicyHeadDesc
static SWPolicyHeadDesc * policyHeadDescToSwift(const PolicyHeadDesc * policyHead) {

    SWConvLayerDesc * p1Conv = convLayerDescToSwift(&policyHead->p1Conv);
    SWConvLayerDesc * g1Conv = convLayerDescToSwift(&policyHead->g1Conv);
    SWBatchNormLayerDesc * g1BN = batchNormLayerDescToSwift(&policyHead->g1BN);
    ActivationKind g1Activation = activationLayerDescToSwift(&policyHead->g1Activation);
    SWMatMulLayerDesc * gpoolToBiasMul = matMulLayerDescToSwift(&policyHead->gpoolToBiasMul);
    SWBatchNormLayerDesc * p1BN = batchNormLayerDescToSwift(&policyHead->p1BN);
    ActivationKind p1Activation = activationLayerDescToSwift(&policyHead->p1Activation);
    SWConvLayerDesc * p2Conv = convLayerDescToSwift(&policyHead->p2Conv);
    SWMatMulLayerDesc * gpoolToPassMul = matMulLayerDescToSwift(&policyHead->gpoolToPassMul);

    SWPolicyHeadDesc * swPolicyHead =
    [[SWPolicyHeadDesc alloc] initWithVersion:policyHead->version
                                       p1Conv:p1Conv
                                       g1Conv:g1Conv
                                         g1BN:g1BN
                                 g1Activation:g1Activation
                               gpoolToBiasMul:gpoolToBiasMul
                                         p1BN:p1BN
                                 p1Activation:p1Activation
                                       p2Conv:p2Conv
                               gpoolToPassMul:gpoolToPassMul];

    return swPolicyHead;
}

/// Convert a matrix bias layer description from C++ to Swift
/// - Parameter desc: A matrix bias layer description
/// - Returns: The matrix bias layer description converted to SWMatBiasLayerDesc
static SWMatBiasLayerDesc * matBiasLayerDescToSwift(const MatBiasLayerDesc * desc) {
    SWMatBiasLayerDesc * swDesc =
    [[SWMatBiasLayerDesc alloc] initWithNumChannels:[NSNumber numberWithInt:desc->numChannels]
                                            weights:(float*)desc->weights.data()];

    return swDesc;
}

/// Convert a value head description from C++ to Swift
/// - Parameter valueHead: A value head description
/// - Returns: The value head description converted to SWValueHeadDesc
static SWValueHeadDesc * valueHeadDescToSwift(const ValueHeadDesc * valueHead) {

    SWConvLayerDesc * v1Conv = convLayerDescToSwift(&valueHead->v1Conv);
    SWBatchNormLayerDesc * v1BN = batchNormLayerDescToSwift(&valueHead->v1BN);
    ActivationKind v1Activation = activationLayerDescToSwift(&valueHead->v1Activation);
    SWMatMulLayerDesc * v2Mul = matMulLayerDescToSwift(&valueHead->v2Mul);
    SWMatBiasLayerDesc * v2Bias = matBiasLayerDescToSwift(&valueHead->v2Bias);
    ActivationKind v2Activation = activationLayerDescToSwift(&valueHead->v2Activation);
    SWMatMulLayerDesc * v3Mul = matMulLayerDescToSwift(&valueHead->v3Mul);
    SWMatBiasLayerDesc * v3Bias = matBiasLayerDescToSwift(&valueHead->v3Bias);
    SWMatMulLayerDesc * sv3Mul = matMulLayerDescToSwift(&valueHead->sv3Mul);
    SWMatBiasLayerDesc * sv3Bias = matBiasLayerDescToSwift(&valueHead->sv3Bias);
    SWConvLayerDesc * vOwnershipConv = convLayerDescToSwift(&valueHead->vOwnershipConv);

    SWValueHeadDesc * swDesc =
    [[SWValueHeadDesc alloc] initWithVersion:valueHead->version
                                      v1Conv:v1Conv
                                        v1BN:v1BN
                                v1Activation:v1Activation
                                       v2Mul:v2Mul
                                      v2Bias:v2Bias
                                v2Activation:v2Activation
                                       v3Mul:v3Mul
                                      v3Bias:v3Bias
                                      sv3Mul:sv3Mul
                                     sv3Bias:sv3Bias
                              vOwnershipConv:vOwnershipConv];

    return swDesc;
}

/// Print the list of available Metal devices
void MetalProcess::printMetalDevices(void) {
    [MetalBackend printDevices];
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
    SWEnable useFP16Mode;
    SWEnable useNHWCMode;

    if (inputUseFP16Mode == enabled_t::False) {
        useFP16Mode = SWEnableFalse;
    } else if (inputUseFP16Mode == enabled_t::True) {
        useFP16Mode = SWEnableTrue;
    } else {
        useFP16Mode = SWEnableAuto;
    }

    if (inputUseNHWCMode == enabled_t::False) {
        useNHWCMode = SWEnableFalse;
    } else if (inputUseNHWCMode == enabled_t::True) {
        useNHWCMode = SWEnableTrue;
    } else {
        useNHWCMode = SWEnableAuto;
    }

    [MetalComputeContext createInstanceWithNnXLen:[NSNumber numberWithInt:nnXLen]
                                           nnYLen:[NSNumber numberWithInt:nnYLen]
                                      useFP16Mode:useFP16Mode
                                      useNHWCMode:useNHWCMode];
}

/// Destroy the Metal context
void MetalProcess::destroyMetalContext(void) {
    [MetalComputeContext destroyInstance];
}

/// Get x length of the Metal context
int MetalProcess::getMetalContextXLen(void) {
    return (int)[MetalBackend getContextXLen];
}

/// Get y length of the Metal context
int MetalProcess::getMetalContextYLen(void) {
    return (int)[MetalBackend getContextYLen];
}

/// Create a Metal handle
/// - Parameters:
///   - gpuIdxForThisThread: The GPU index for this thread
///   - desc: The model description
///   - serverThreadIdx: The server thread index
void MetalProcess::createMetalHandle(int gpuIdxForThisThread,
                                const ModelDesc* desc,
                                int serverThreadIdx) {
    NSString * name = [NSString stringWithUTF8String:desc->name.c_str()];

    SWModelDesc * swModelDesc =
    [[SWModelDesc alloc] initWithVersion:desc->version
                                    name:name
                        numInputChannels:[NSNumber numberWithInt:desc->numInputChannels]
                  numInputGlobalChannels:[NSNumber numberWithInt:desc->numInputGlobalChannels]
                        numValueChannels:[NSNumber numberWithInt:desc->numValueChannels]
                   numScoreValueChannels:[NSNumber numberWithInt:desc->numScoreValueChannels]
                    numOwnershipChannels:[NSNumber numberWithInt:desc->numOwnershipChannels]
                                   trunk:trunkDescToSwift(&desc->trunk)
                              policyHead:policyHeadDescToSwift(&desc->policyHead)
                               valueHead:valueHeadDescToSwift(&desc->valueHead)];

    [MetalComputeHandle createInstanceAt:gpuIdxForThisThread
                              descriptor:swModelDesc
                         serverThreadIdx:serverThreadIdx];
}

/// Get output from a Metal handle
/// - Parameters:
///   - userInputBuffer: The user input buffer
///   - userInputGlobalBuffer: The user input global buffer
///   - policyOutput: The policy output
///   - policyPassOutput: The policy pass output
///   - valueOutput: The value output
///   - ownershipOutput: The ownership output
///   - scoreValueOutput: The score value output
///   - gpuIdx: The GPU index
///   - batchSize: The batch size
void MetalProcess::getMetalHandleOutput(float* userInputBuffer,
                                   float* userInputGlobalBuffer,
                                   float* policyOutput,
                                   float* policyPassOutput,
                                   float* valueOutput,
                                   float* ownershipOutput,
                                   float* scoreValueOutput,
                                   int gpuIdx,
                                   int batchSize) {
    [MetalBackend getOutputWithUserInputBuffer:userInputBuffer
                         userInputGlobalBuffer:userInputGlobalBuffer
                                  policyOutput:policyOutput
                              policyPassOutput:policyPassOutput
                                   valueOutput:valueOutput
                               ownershipOutput:ownershipOutput
                              scoreValueOutput:scoreValueOutput
                                        gpuIdx:gpuIdx
                                     batchSize:batchSize];
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
    [ConvLayer testWithDescriptor:convLayerDescToSwift(desc)
                           nnXLen:[NSNumber numberWithInt:nnXLen]
                           nnYLen:[NSNumber numberWithInt:nnYLen]
                        batchSize:[NSNumber numberWithInt:batchSize]
                            input:input
                           output:output];
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
    [BatchNormLayer testWithDescriptor:batchNormLayerDescToSwift(desc)
                                nnXLen:[NSNumber numberWithInt:nnXLen]
                                nnYLen:[NSNumber numberWithInt:nnYLen]
                             batchSize:[NSNumber numberWithInt:batchSize]
                                 input:input
                                  mask:mask
                                output:output];
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
    [ResidualBlock testWithDescriptor:residualBlockDescToSwift(desc)
                            batchSize:[NSNumber numberWithInt:batchSize]
                               nnXLen:[NSNumber numberWithInt:nnXLen]
                               nnYLen:[NSNumber numberWithInt:nnYLen]
                                input:input
                                 mask:mask
                               output:output];
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
    [GlobalPoolingResidualBlock testWithDescriptor:globalPoolingResidualBlockDescToSwift(desc)
                                         batchSize:[NSNumber numberWithInt:batchSize]
                                            nnXLen:[NSNumber numberWithInt:nnXLen]
                                            nnYLen:[NSNumber numberWithInt:nnYLen]
                                             input:input
                                              mask:mask
                                            output:output];
}
