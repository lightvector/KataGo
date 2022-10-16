#import "metalbackend.h"
#import "metalswift.h"

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

static SWResidualBlockDesc * residualBlockDescToSwift(const ResidualBlockDesc * desc) {

    SWBatchNormLayerDesc * preBN = batchNormLayerDescToSwift(&desc->preBN);
    SWConvLayerDesc * regularConv = convLayerDescToSwift(&desc->regularConv);
    SWBatchNormLayerDesc * midBN = batchNormLayerDescToSwift(&desc->midBN);
    SWConvLayerDesc * finalConv = convLayerDescToSwift(&desc->finalConv);

    SWResidualBlockDesc * swDesc = [[SWResidualBlockDesc alloc] initWithPreBN:preBN
                                                                preActivation:nil
                                                                  regularConv:regularConv
                                                                        midBN:midBN
                                                                midActivation:nil
                                                                    finalConv:finalConv];

    return swDesc;
}

static SWMatMulLayerDesc * matMulLayerDescToSwift(const MatMulLayerDesc * desc) {

    SWMatMulLayerDesc * swDesc =
    [[SWMatMulLayerDesc alloc] initInChannels:[NSNumber numberWithInt:desc->inChannels]
                                  outChannels:[NSNumber numberWithInt:desc->outChannels]
                                      weights:(float*)desc->weights.data()];

    return swDesc;
}

static SWGlobalPoolingResidualBlockDesc* globalPoolingResidualBlockDescToSwift(const GlobalPoolingResidualBlockDesc* desc) {

    SWBatchNormLayerDesc * preBN = batchNormLayerDescToSwift(&desc->preBN);
    SWConvLayerDesc * regularConv = convLayerDescToSwift(&desc->regularConv);
    SWConvLayerDesc * gpoolConv = convLayerDescToSwift(&desc->gpoolConv);
    SWBatchNormLayerDesc * gpoolBN = batchNormLayerDescToSwift(&desc->gpoolBN);
    SWMatMulLayerDesc * gpoolToBiasMul = matMulLayerDescToSwift(&desc->gpoolToBiasMul);
    SWBatchNormLayerDesc * midBN = batchNormLayerDescToSwift(&desc->midBN);
    SWConvLayerDesc * finalConv = convLayerDescToSwift(&desc->finalConv);

    SWGlobalPoolingResidualBlockDesc * swDesc =
    [[SWGlobalPoolingResidualBlockDesc alloc] initWithPreBN:preBN
                                              preActivation:nil
                                                regularConv:regularConv
                                                  gpoolConv:gpoolConv
                                                    gpoolBN:gpoolBN
                                            gpoolActivation:nil
                                             gpoolToBiasMul:gpoolToBiasMul
                                                      midBN:midBN
                                              midActivation:nil
                                                  finalConv:finalConv];

    return swDesc;
}

static SWTrunkDesc * trunkDescToSwift(const TrunkDesc * trunk) {

    SWConvLayerDesc * initialConv = convLayerDescToSwift(&trunk->initialConv);
    SWMatMulLayerDesc * initialMatMul = matMulLayerDescToSwift(&trunk->initialMatMul);

    const std::vector<std::pair<int, unique_ptr_void>>& blocks = trunk->blocks;
    NSMutableArray<BlockDescriptor *> * swBlocks = [[NSMutableArray alloc] init];

    for (int i = 0; i < blocks.size(); i++) {

        BlockDescriptor * blockDesc;

        if (blocks[i].first == ORDINARY_BLOCK_KIND) {
            ResidualBlockDesc * residualBlockDesc = (ResidualBlockDesc*)blocks[i].second.get();
            SWResidualBlockDesc * swResidualBlockDesc = residualBlockDescToSwift(residualBlockDesc);

            blockDesc = [[BlockDescriptor alloc] initWithKind:BlockKindOrdinary
                                                     ordinary:swResidualBlockDesc
                                                globalPooling:nil];
        } else {
            GlobalPoolingResidualBlockDesc * residualBlockDesc = (GlobalPoolingResidualBlockDesc*)blocks[i].second.get();
            SWGlobalPoolingResidualBlockDesc * swResidualBlockDesc = globalPoolingResidualBlockDescToSwift(residualBlockDesc);

            blockDesc = [[BlockDescriptor alloc] initWithKind:BlockKindGlobalPooling
                                                     ordinary:nil
                                                globalPooling:swResidualBlockDesc];
        }

        [swBlocks addObject:blockDesc];
    }

    SWBatchNormLayerDesc * trunkTipBN = batchNormLayerDescToSwift(&trunk->trunkTipBN);

    SWTrunkDesc * swTrunkDesc =
    [[SWTrunkDesc alloc] initWithVersion:trunk->version
                               numBlocks:trunk->numBlocks
                        trunkNumChannels:[NSNumber numberWithInt:trunk->trunkNumChannels]
                          midNumChannels:[NSNumber numberWithInt:trunk->midNumChannels]
                      regularNumChannels:[NSNumber numberWithInt:trunk->regularNumChannels]
                      dilatedNumChannels:[NSNumber numberWithInt:trunk->dilatedNumChannels]
                        gpoolNumChannels:[NSNumber numberWithInt:trunk->gpoolNumChannels]
                             initialConv:initialConv
                           initialMatMul:initialMatMul
                                  blocks:swBlocks
                              trunkTipBN:trunkTipBN];

    return swTrunkDesc;
}

static SWPolicyHeadDesc * policyHeadDescToSwift(const PolicyHeadDesc * policyHead) {

    SWConvLayerDesc * p1Conv = convLayerDescToSwift(&policyHead->p1Conv);
    SWConvLayerDesc * g1Conv = convLayerDescToSwift(&policyHead->g1Conv);
    SWBatchNormLayerDesc * g1BN = batchNormLayerDescToSwift(&policyHead->g1BN);
    SWMatMulLayerDesc * gpoolToBiasMul = matMulLayerDescToSwift(&policyHead->gpoolToBiasMul);
    SWBatchNormLayerDesc * p1BN = batchNormLayerDescToSwift(&policyHead->p1BN);
    SWConvLayerDesc * p2Conv = convLayerDescToSwift(&policyHead->p2Conv);
    SWMatMulLayerDesc * gpoolToPassMul = matMulLayerDescToSwift(&policyHead->gpoolToPassMul);

    SWPolicyHeadDesc * swPolicyHead =
    [[SWPolicyHeadDesc alloc] initWithVersion:policyHead->version
                                       p1Conv:p1Conv
                                       g1Conv:g1Conv
                                         g1BN:g1BN
                               gpoolToBiasMul:gpoolToBiasMul
                                         p1BN:p1BN
                                       p2Conv:p2Conv
                               gpoolToPassMul:gpoolToPassMul];

    return swPolicyHead;
}

static SWMatBiasLayerDesc * matBiasLayerDescToSwift(const MatBiasLayerDesc * desc) {
    SWMatBiasLayerDesc * swDesc =
    [[SWMatBiasLayerDesc alloc] initWithNumChannels:[NSNumber numberWithInt:desc->numChannels]
                                            weights:(float*)desc->weights.data()];

    return swDesc;
}

static SWValueHeadDesc * valueHeadDescToSwift(const ValueHeadDesc * valueHead) {

    SWConvLayerDesc * v1Conv = convLayerDescToSwift(&valueHead->v1Conv);
    SWBatchNormLayerDesc * v1BN = batchNormLayerDescToSwift(&valueHead->v1BN);
    SWMatMulLayerDesc * v2Mul = matMulLayerDescToSwift(&valueHead->v2Mul);
    SWMatBiasLayerDesc * v2Bias = matBiasLayerDescToSwift(&valueHead->v2Bias);
    SWMatMulLayerDesc * v3Mul = matMulLayerDescToSwift(&valueHead->v3Mul);
    SWMatBiasLayerDesc * v3Bias = matBiasLayerDescToSwift(&valueHead->v3Bias);
    SWMatMulLayerDesc * sv3Mul = matMulLayerDescToSwift(&valueHead->sv3Mul);
    SWMatBiasLayerDesc * sv3Bias = matBiasLayerDescToSwift(&valueHead->sv3Bias);
    SWConvLayerDesc * vOwnershipConv = convLayerDescToSwift(&valueHead->vOwnershipConv);

    SWValueHeadDesc * swDesc =
    [[SWValueHeadDesc alloc] initWithVersion:valueHead->version
                                      v1Conv:v1Conv
                                        v1BN:v1BN
                                       v2Mul:v2Mul
                                      v2Bias:v2Bias
                                       v3Mul:v3Mul
                                      v3Bias:v3Bias
                                      sv3Mul:sv3Mul
                                     sv3Bias:sv3Bias
                              vOwnershipConv:vOwnershipConv];

    return swDesc;
}

MetalDevices::MetalDevices(void) {}
MetalDevices::~MetalDevices(void) {}
void MetalDevices::printDevices(void) {}

void createMetalContext(int nnXLen,
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

    [ComputeContext createInstanceWithNnXLen:[NSNumber numberWithInt:nnXLen]
                                      nnYLen:[NSNumber numberWithInt:nnYLen]
                                 useFP16Mode:useFP16Mode
                                 useNHWCMode:useNHWCMode];
}

void createMetalHandle(int gpuIdxForThisThread,
                       const ModelDesc* desc,
                       int batchSize,
                       int serverThreadIdx) {
    SWModelDesc * swModelDesc =
    [[SWModelDesc alloc] initWithVersion:desc->version
                        numInputChannels:[NSNumber numberWithInt:desc->numInputChannels]
                  numInputGlobalChannels:[NSNumber numberWithInt:desc->numInputGlobalChannels]
                        numValueChannels:[NSNumber numberWithInt:desc->numValueChannels]
                   numScoreValueChannels:[NSNumber numberWithInt:desc->numScoreValueChannels]
                    numOwnershipChannels:[NSNumber numberWithInt:desc->numOwnershipChannels]
                                   trunk:trunkDescToSwift(&desc->trunk)
                              policyHead:policyHeadDescToSwift(&desc->policyHead)
                               valueHead:valueHeadDescToSwift(&desc->valueHead)];

    [ComputeHandle createInstanceAt:gpuIdxForThisThread
                         descriptor:swModelDesc
                          batchSize:[NSNumber numberWithInt:batchSize]
                    serverThreadIdx:serverThreadIdx];
}

void getMetalHandleOutput(float* userInputBuffer,
                          float* userInputGlobalBuffer,
                          float* policyOutput,
                          float* valueOutput,
                          float* ownershipOutput,
                          float* miscValuesOutput,
                          float* moreMiscValuesOutput,
                          int gpuIdx) {
    // FIXME: to be done
    KataGoGraph* graph = [KataGoGraph getGraphWithGpuIndex:[NSNumber numberWithInt:gpuIdx]];

    [graph runWithUserInputBuffer:userInputBuffer
            userInputGlobalBuffer:userInputGlobalBuffer
                     policyOutput:policyOutput
                      valueOutput:valueOutput
                  ownershipOutput:ownershipOutput
                 miscValuesOutput:miscValuesOutput
             moreMiscValuesOutput:moreMiscValuesOutput];
}

void testMetalEvaluateConv(const ConvLayerDesc* desc,
                           int nnXLen,
                           int nnYLen,
                           int batchSize,
                           bool useFP16,
                           bool useNHWC,
                           float* input,
                           float* output) {
    [ConvLayer testWithDescriptor:convLayerDescToSwift(desc)
                           nnXLen:[NSNumber numberWithInt:nnXLen]
                           nnYLen:[NSNumber numberWithInt:nnYLen]
                        batchSize:[NSNumber numberWithInt:batchSize]
                          useFP16:useFP16
                          useNHWC:useNHWC
                            input:input
                           output:output];
}

void testMetalEvaluateBatchNorm(const BatchNormLayerDesc* desc,
                                int nnXLen,
                                int nnYLen,
                                int batchSize,
                                bool useFP16,
                                bool useNHWC,
                                float* input,
                                float* mask,
                                float* output) {
    [BatchNormLayer testWithDescriptor:batchNormLayerDescToSwift(desc)
                                nnXLen:[NSNumber numberWithInt:nnXLen]
                                nnYLen:[NSNumber numberWithInt:nnYLen]
                             batchSize:[NSNumber numberWithInt:batchSize]
                               useFP16:useFP16
                               useNHWC:useNHWC
                                 input:input
                                  mask:mask
                                output:output];
}

void testMetalEvaluateResidualBlock(const ResidualBlockDesc* desc,
                                    int batchSize,
                                    int nnXLen,
                                    int nnYLen,
                                    bool useFP16,
                                    bool useNHWC,
                                    float* input,
                                    float* mask,
                                    float* output) {
    [ResidualBlock testWithDescriptor:residualBlockDescToSwift(desc)
                            batchSize:[NSNumber numberWithInt:batchSize]
                               nnXLen:[NSNumber numberWithInt:nnXLen]
                               nnYLen:[NSNumber numberWithInt:nnYLen]
                              useFP16:useFP16
                              useNHWC:useNHWC
                                input:input
                                 mask:mask
                               output:output];
}

void testMetalEvaluateGlobalPoolingResidualBlock(const GlobalPoolingResidualBlockDesc* desc,
                                                 int batchSize,
                                                 int nnXLen,
                                                 int nnYLen,
                                                 bool useFP16,
                                                 bool useNHWC,
                                                 float* input,
                                                 float* mask,
                                                 float* output) {
    [GlobalPoolingResidualBlock testWithDescriptor:globalPoolingResidualBlockDescToSwift(desc)
                                         batchSize:[NSNumber numberWithInt:batchSize]
                                            nnXLen:[NSNumber numberWithInt:nnXLen]
                                            nnYLen:[NSNumber numberWithInt:nnYLen]
                                           useFP16:useFP16
                                           useNHWC:useNHWC
                                             input:input
                                              mask:mask
                                            output:output];
}
