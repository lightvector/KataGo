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

void testMetalEvaluateConv(int convXSize,
                           int convYSize,
                           int inChannels,
                           int outChannels,
                           int dilationX,
                           int dilationY,
                           int nnXLen,
                           int nnYLen,
                           int batchSize,
                           bool useFP16,
                           bool useNHWC,
                           float* weights,
                           float* input,
                           float* output) {
    [ConvLayer testWithConvXSize:[NSNumber numberWithInt:convXSize]
                       convYSize:[NSNumber numberWithInt:convYSize]
                      inChannels:[NSNumber numberWithInt:inChannels]
                     outChannels:[NSNumber numberWithInt:outChannels]
                       dilationX:[NSNumber numberWithInt:dilationX]
                       dilationY:[NSNumber numberWithInt:dilationY]
                          nnXLen:[NSNumber numberWithInt:nnXLen]
                          nnYLen:[NSNumber numberWithInt:nnYLen]
                       batchSize:[NSNumber numberWithInt:batchSize]
                         useFP16:[NSNumber numberWithBool:useFP16]
                         useNHWC:[NSNumber numberWithBool:useNHWC]
                         weights:weights
                           input:input
                          output:output];
}
