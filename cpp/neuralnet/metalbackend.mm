#import "metalbackend.h"
#import "metalswift.h"

MetalDevices::MetalDevices(void) {}
MetalDevices::~MetalDevices(void) {}
void MetalDevices::printDevices(void) {}

void createMetalHandle(int gpuIdx,
                       int nnXLen,
                       int nnYLen,
                       int version,
                       int numInputChannels,
                       int numInputGlobalChannels,
                       int numValueChannels,
                       int numScoreValueChannels,
                       int numOwnershipChannels) {
    [KataGoGraph initGraphWithGpuIndex:[NSNumber numberWithInt:gpuIdx]
                                nnXLen:[NSNumber numberWithInt:nnXLen]
                                nnYLen:[NSNumber numberWithInt:nnYLen]
                               version:[NSNumber numberWithInt:version]
                      numInputChannels:[NSNumber numberWithInt:numInputChannels]
                numInputGlobalChannels:[NSNumber numberWithInt:numInputGlobalChannels]
                      numValueChannels:[NSNumber numberWithInt:numValueChannels]
                 numScoreValueChannels:[NSNumber numberWithInt:numScoreValueChannels]
                  numOwnershipChannels:[NSNumber numberWithInt:numOwnershipChannels]];
}

void getMetalHandleOutput(float* userInputBuffer,
                          float* userInputGlobalBuffer,
                          float* policyOutput,
                          float* valueOutput,
                          float* ownershipOutput,
                          float* miscValuesOutput,
                          float* moreMiscValuesOutput,
                          int gpuIdx) {
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
    SWConvLayerDesc * swDesc;

    swDesc = [[SWConvLayerDesc alloc] initWithConvYSize:[NSNumber numberWithInt:desc->convYSize]
                                                 convXSize:[NSNumber numberWithInt:desc->convXSize]
                                                inChannels:[NSNumber numberWithInt:desc->inChannels]
                                               outChannels:[NSNumber numberWithInt:desc->outChannels]
                                                 dilationY:desc->dilationY
                                                 dilationX:desc->dilationX
                                                   weights:(float*)desc->weights.data()];

    [ConvLayer testWithDescriptor:swDesc
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
    SWBatchNormLayerDesc * swDesc;

    swDesc = [[SWBatchNormLayerDesc alloc] initWithNumChannels:[NSNumber numberWithInt:desc->numChannels]
                                                          epsilon:desc->epsilon
                                                         hasScale:[NSNumber numberWithBool:desc->hasScale]
                                                          hasBias:[NSNumber numberWithBool:desc->hasBias]
                                                             mean:(float*)desc->mean.data()
                                                         variance:(float*)desc->variance.data()
                                                            scale:(float*)desc->scale.data()
                                                             bias:(float*)desc->bias.data()];

    [BatchNormLayer testWithDescriptor:swDesc
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
    SWResidualBlockDesc * swDesc;
    SWBatchNormLayerDesc * preBN;
    SWConvLayerDesc * regularConv;
    SWBatchNormLayerDesc * midBN;
    SWConvLayerDesc * finalConv;

    preBN = [[SWBatchNormLayerDesc alloc] initWithNumChannels:[NSNumber numberWithInt:desc->preBN.numChannels]
                                                      epsilon:desc->preBN.epsilon
                                                     hasScale:[NSNumber numberWithBool:desc->preBN.hasScale]
                                                      hasBias:[NSNumber numberWithBool:desc->preBN.hasBias]
                                                         mean:(float*)desc->preBN.mean.data()
                                                     variance:(float*)desc->preBN.variance.data()
                                                        scale:(float*)desc->preBN.scale.data()
                                                         bias:(float*)desc->preBN.bias.data()];

    regularConv = [[SWConvLayerDesc alloc] initWithConvYSize:[NSNumber numberWithInt:desc->regularConv.convYSize]
                                                   convXSize:[NSNumber numberWithInt:desc->regularConv.convXSize]
                                                  inChannels:[NSNumber numberWithInt:desc->regularConv.inChannels]
                                                 outChannels:[NSNumber numberWithInt:desc->regularConv.outChannels]
                                                   dilationY:desc->regularConv.dilationY
                                                   dilationX:desc->regularConv.dilationX
                                                     weights:(float*)desc->regularConv.weights.data()];

    midBN = [[SWBatchNormLayerDesc alloc] initWithNumChannels:[NSNumber numberWithInt:desc->midBN.numChannels]
                                                      epsilon:desc->midBN.epsilon
                                                     hasScale:[NSNumber numberWithBool:desc->midBN.hasScale]
                                                      hasBias:[NSNumber numberWithBool:desc->midBN.hasBias]
                                                         mean:(float*)desc->midBN.mean.data()
                                                     variance:(float*)desc->midBN.variance.data()
                                                        scale:(float*)desc->midBN.scale.data()
                                                         bias:(float*)desc->midBN.bias.data()];

    finalConv = [[SWConvLayerDesc alloc] initWithConvYSize:[NSNumber numberWithInt:desc->finalConv.convYSize]
                                                 convXSize:[NSNumber numberWithInt:desc->finalConv.convXSize]
                                                inChannels:[NSNumber numberWithInt:desc->finalConv.inChannels]
                                               outChannels:[NSNumber numberWithInt:desc->finalConv.outChannels]
                                                 dilationY:desc->finalConv.dilationY
                                                 dilationX:desc->finalConv.dilationX
                                                   weights:(float*)desc->finalConv.weights.data()];

    swDesc = [[SWResidualBlockDesc alloc] initWithPreBN:preBN
                                          preActivation:nil
                                            regularConv:regularConv
                                                  midBN:midBN
                                          midActivation:nil
                                              finalConv:finalConv];

    [ResidualBlock testWithDescriptor:swDesc
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

    SWGlobalPoolingResidualBlockDesc * swDesc;
    SWBatchNormLayerDesc * preBN;
    SWConvLayerDesc * regularConv;
    SWConvLayerDesc * gpoolConv;
    SWBatchNormLayerDesc * gpoolBN;
    SWMatMulLayerDesc * gpoolToBiasMul;
    SWBatchNormLayerDesc * midBN;
    SWConvLayerDesc * finalConv;

    preBN = [[SWBatchNormLayerDesc alloc] initWithNumChannels:[NSNumber numberWithInt:desc->preBN.numChannels]
                                                      epsilon:desc->preBN.epsilon
                                                     hasScale:[NSNumber numberWithBool:desc->preBN.hasScale]
                                                      hasBias:[NSNumber numberWithBool:desc->preBN.hasBias]
                                                         mean:(float*)desc->preBN.mean.data()
                                                     variance:(float*)desc->preBN.variance.data()
                                                        scale:(float*)desc->preBN.scale.data()
                                                         bias:(float*)desc->preBN.bias.data()];

    regularConv = [[SWConvLayerDesc alloc] initWithConvYSize:[NSNumber numberWithInt:desc->regularConv.convYSize]
                                                   convXSize:[NSNumber numberWithInt:desc->regularConv.convXSize]
                                                  inChannels:[NSNumber numberWithInt:desc->regularConv.inChannels]
                                                 outChannels:[NSNumber numberWithInt:desc->regularConv.outChannels]
                                                   dilationY:desc->regularConv.dilationY
                                                   dilationX:desc->regularConv.dilationX
                                                     weights:(float*)desc->regularConv.weights.data()];

    gpoolConv = [[SWConvLayerDesc alloc] initWithConvYSize:[NSNumber numberWithInt:desc->gpoolConv.convYSize]
                                                 convXSize:[NSNumber numberWithInt:desc->gpoolConv.convXSize]
                                                inChannels:[NSNumber numberWithInt:desc->gpoolConv.inChannels]
                                               outChannels:[NSNumber numberWithInt:desc->gpoolConv.outChannels]
                                                 dilationY:desc->gpoolConv.dilationY
                                                 dilationX:desc->gpoolConv.dilationX
                                                   weights:(float*)desc->gpoolConv.weights.data()];

    gpoolBN = [[SWBatchNormLayerDesc alloc] initWithNumChannels:[NSNumber numberWithInt:desc->gpoolBN.numChannels]
                                                        epsilon:desc->gpoolBN.epsilon
                                                       hasScale:[NSNumber numberWithBool:desc->gpoolBN.hasScale]
                                                        hasBias:[NSNumber numberWithBool:desc->gpoolBN.hasBias]
                                                           mean:(float*)desc->gpoolBN.mean.data()
                                                       variance:(float*)desc->gpoolBN.variance.data()
                                                          scale:(float*)desc->gpoolBN.scale.data()
                                                           bias:(float*)desc->gpoolBN.bias.data()];

    gpoolToBiasMul = [[SWMatMulLayerDesc alloc] initInChannels:[NSNumber numberWithInt:desc->gpoolToBiasMul.inChannels]
                                                   outChannels:[NSNumber numberWithInt:desc->gpoolToBiasMul.outChannels]
                                                       weights:(float*)desc->gpoolToBiasMul.weights.data()];

    midBN = [[SWBatchNormLayerDesc alloc] initWithNumChannels:[NSNumber numberWithInt:desc->midBN.numChannels]
                                                      epsilon:desc->midBN.epsilon
                                                     hasScale:[NSNumber numberWithBool:desc->midBN.hasScale]
                                                      hasBias:[NSNumber numberWithBool:desc->midBN.hasBias]
                                                         mean:(float*)desc->midBN.mean.data()
                                                     variance:(float*)desc->midBN.variance.data()
                                                        scale:(float*)desc->midBN.scale.data()
                                                         bias:(float*)desc->midBN.bias.data()];

    finalConv = [[SWConvLayerDesc alloc] initWithConvYSize:[NSNumber numberWithInt:desc->finalConv.convYSize]
                                                 convXSize:[NSNumber numberWithInt:desc->finalConv.convXSize]
                                                inChannels:[NSNumber numberWithInt:desc->finalConv.inChannels]
                                               outChannels:[NSNumber numberWithInt:desc->finalConv.outChannels]
                                                 dilationY:desc->finalConv.dilationY
                                                 dilationX:desc->finalConv.dilationX
                                                   weights:(float*)desc->finalConv.weights.data()];

    swDesc = [[SWGlobalPoolingResidualBlockDesc alloc] initWithPreBN:preBN
                                                       preActivation:nil
                                                         regularConv:regularConv
                                                           gpoolConv:gpoolConv
                                                             gpoolBN:gpoolBN
                                                     gpoolActivation:nil
                                                      gpoolToBiasMul:gpoolToBiasMul
                                                               midBN:midBN
                                                       midActivation:nil
                                                           finalConv:finalConv];

    [GlobalPoolingResidualBlock testWithDescriptor:swDesc
                                         batchSize:[NSNumber numberWithInt:batchSize]
                                            nnXLen:[NSNumber numberWithInt:nnXLen]
                                            nnYLen:[NSNumber numberWithInt:nnYLen]
                                           useFP16:useFP16
                                           useNHWC:useNHWC
                                             input:input
                                              mask:mask
                                            output:output];
}
