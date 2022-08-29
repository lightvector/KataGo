#import <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>
#import "metalbackend.h"

@interface KataGoGraph : NSObject {
@private
  id<MTLDevice> device;
  id<MTLCommandQueue> commandQueue;
  dispatch_semaphore_t doubleBufferingSemaphore;
  MPSGraph* graph;
  MPSGraphTensor* sourcePlaceholderTensor;
}

-(nonnull instancetype) initWithDevice:(nonnull id <MTLDevice>) inputDevice
                                nnXLen:(int)nnXLen
                                nnYLen:(int)nnYLen
                               version:(int)version
                      numInputChannels:(int)numInputChannels
                numInputGlobalChannels:(int)numInputGlobalChannels
                      numValueChannels:(int)numValueChannels
                 numScoreValueChannels:(int)numScoreValueChannels
                  numOwnershipChannels:(int)numOwnershipChannels;
@end

@implementation KataGoGraph

-(nonnull instancetype) initWithDevice:(nonnull id <MTLDevice>) inputDevice
                                nnXLen:(int)nnXLen
                                nnYLen:(int)nnYLen
                               version:(int)version
                      numInputChannels:(int)numInputChannels
                numInputGlobalChannels:(int)numInputGlobalChannels
                      numValueChannels:(int)numValueChannels
                 numScoreValueChannels:(int)numScoreValueChannels
                  numOwnershipChannels:(int)numOwnershipChannels {
  self = [super init];
  device = inputDevice;
  commandQueue = [device newCommandQueue];
  doubleBufferingSemaphore = dispatch_semaphore_create(2);
  graph = [MPSGraph alloc];
  return self;
}

-(void) encodeInferenceBatch:(nonnull float*)userInputBuffer
       userInputGlobalBuffer:(nonnull float*)userInputGlobalBuffer
                policyOutput:(nonnull float*)policyOutput
                 valueOutput:(nonnull float*)valueOutput
             ownershipOutput:(nonnull float*)ownershipOutput
            miscValuesOutput:(nonnull float*)miscValuesOutput
        moreMiscValuesOutput:(nonnull float*)moreMiscValuesOutput
{
  MPSGraphTensor* labelsPlaceholderTensor = [MPSGraphTensor alloc];
  MPSGraphTensorData* sourceTensorData = [MPSGraphTensorData alloc];
  MPSGraphTensorData* labelsTensorData = [MPSGraphTensorData alloc];
  NSArray<MPSGraphTensor*>* targetTensors = [NSArray alloc];
  NSArray<MPSGraphOperation*>* targetOperations = [NSArray alloc];

  dispatch_semaphore_wait(doubleBufferingSemaphore, DISPATCH_TIME_FOREVER);
  MPSCommandBuffer* commandBuffer = [MPSCommandBuffer commandBufferFromCommandQueue:commandQueue];
  MPSGraphExecutionDescriptor* executionDesc = [MPSGraphExecutionDescriptor alloc];
  executionDesc.completionHandler = ^(MPSGraphTensorDataDictionary* resultsDictionary, NSError* error) {
    dispatch_semaphore_signal(doubleBufferingSemaphore);
  };

  MPSGraphTensorDataDictionary* feeds = @{
    sourcePlaceholderTensor : sourceTensorData,
    labelsPlaceholderTensor : labelsTensorData
  };

  MPSGraphTensorDataDictionary* fetch = [graph encodeToCommandBuffer:commandBuffer
                                                               feeds:feeds
                                                       targetTensors:targetTensors
                                                    targetOperations:targetOperations
                                                 executionDescriptor:executionDesc];

  [commandBuffer commit];
  [commandBuffer waitUntilCompleted];
}

-(MPSGraphTensor*) placeholderWithShape:(int)nnXLen
                                 nnYLen:(int)nnYLen
                       numInputChannels:(int)numInputChannels
                 numInputGlobalChannels:(int)numInputGlobalChannels
                                   name:(nonnull NSString*)name
{
  int channels = numInputChannels + numInputGlobalChannels;
  MPSShape* shape = @[@(-1), @(channels), @(nnYLen), @(nnXLen)];

  sourcePlaceholderTensor = [graph placeholderWithShape:shape
                                                   name:name];

  return sourcePlaceholderTensor;
}

@end

MetalDevices::MetalDevices(void) {
}

MetalDevices::~MetalDevices(void) {}
void MetalDevices::printDevices(void) {}

MetalHandle::MetalHandle() {}
MetalHandle::~MetalHandle(void) {}

void MetalHandle::init(int nnXLen,
                       int nnYLen,
                       int versionIn,
                       int numInputChannels,
                       int numInputGlobalChannels,
                       int numValueChannels,
                       int numScoreValueChannels,
                       int numOwnershipChannels) {
  this->version = versionIn;
  id<MTLDevice> device = MTLCreateSystemDefaultDevice();

  kataGoGraph = [[KataGoGraph alloc] initWithDevice:device
                                             nnXLen:nnXLen
                                             nnYLen:nnYLen
                                            version:version
                                   numInputChannels:numInputChannels
                             numInputGlobalChannels:numInputGlobalChannels
                                   numValueChannels:numValueChannels
                              numScoreValueChannels:numScoreValueChannels
                               numOwnershipChannels:numOwnershipChannels];
}

void* MetalHandle::placeholderWithShape(int nnXLen,
                                        int nnYLen,
                                        int numInputChannels,
                                        int numInputGlobalChannels,
                                        string name) {
  NSString* nsName = [NSString stringWithUTF8String:name.c_str()];

  return [(id)kataGoGraph placeholderWithShape:nnXLen
                                        nnYLen:nnYLen
                              numInputChannels:numInputChannels
                        numInputGlobalChannels:numInputGlobalChannels
                                          name:nsName];
}

void MetalHandle::apply(float* userInputBuffer,
                        float* userInputGlobalBuffer,
                        float* policyOutput,
                        float* valueOutput,
                        float* ownershipOutput,
                        float* miscValuesOutput,
                        float* moreMiscValuesOutput) {
  [(id)kataGoGraph encodeInferenceBatch:userInputBuffer
                  userInputGlobalBuffer:userInputGlobalBuffer
                           policyOutput:policyOutput
                            valueOutput:valueOutput
                        ownershipOutput:ownershipOutput
                       miscValuesOutput:miscValuesOutput
                   moreMiscValuesOutput:moreMiscValuesOutput];
}
