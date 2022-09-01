#import <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>
#import "metalbackend.h"

@interface KataGoGraph : NSObject {
@private
  int nnXLen;
  int nnYLen;
  id<MTLDevice> device;
  id<MTLCommandQueue> commandQueue;
  dispatch_semaphore_t doubleBufferingSemaphore;
  MPSGraph* graph;
  MPSGraphTensor* bin_inputs;
  MPSGraphTensor* global_inputs;
  MPSGraphTensor* symmetries;
  MPSGraphTensor* include_history;
  MPSGraphTensor* policy_output;
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
                                nnXLen:(int)inputXLen
                                nnYLen:(int)inputYLen
                               version:(int)version
                      numInputChannels:(int)numInputChannels
                numInputGlobalChannels:(int)numInputGlobalChannels
                      numValueChannels:(int)numValueChannels
                 numScoreValueChannels:(int)numScoreValueChannels
                  numOwnershipChannels:(int)numOwnershipChannels {
  self = [super init];
  device = inputDevice;
  nnXLen = inputXLen;
  nnYLen = inputYLen;
  commandQueue = [device newCommandQueue];
  doubleBufferingSemaphore = dispatch_semaphore_create(2);
  
  [self initKataGoGraph:version
                 nnXLen:nnXLen
                 nnYLen:nnYLen
       numInputChannels:numInputChannels
 numInputGlobalChannels:numInputGlobalChannels
       numValueChannels:numValueChannels
  numScoreValueChannels:numScoreValueChannels
   numOwnershipChannels:numOwnershipChannels];
  
  return self;
}

-(void) initKataGoGraph:(int)version
                 nnXLen:(int)nnXLen
                 nnYLen:(int)nnYLen
       numInputChannels:(int)numInputChannels
 numInputGlobalChannels:(int)numInputGlobalChannels
       numValueChannels:(int)numValueChannels
  numScoreValueChannels:(int)numScoreValueChannels
   numOwnershipChannels:(int)numOwnershipChannels
{
  int num_bin_input_features = numInputChannels;
  int num_global_input_features = numInputGlobalChannels;
  MPSShape* bin_input_shape = @[@(nnXLen * nnYLen), @(num_bin_input_features)];
  MPSShape* global_input_shape = @[@(num_global_input_features)];
  MPSShape* symmetries_shape = @[@(3)];
  MPSShape* include_history_shape = @[@(5)];
  
  MPSShape* shape;
  
  graph = [MPSGraph alloc];
  
  bin_inputs = [graph placeholderWithShape:bin_input_shape
                                      name:@"bin_inputs"];
  
  global_inputs = [graph placeholderWithShape:global_input_shape
                                         name:@"global_inputs"];
  
  symmetries = [graph placeholderWithShape:symmetries_shape
                                      name:@"symmetries"];
  
  include_history = [graph placeholderWithShape:include_history_shape
                                           name:@"include_history"];
  
  shape = @[@(-1), @(nnXLen * nnYLen), @(num_bin_input_features)];
  
  MPSGraphTensor* cur_layer = [graph reshapeTensor:bin_inputs
                                         withShape:shape
                                              name:@"model.py:940"];
  
  policy_output = cur_layer;
}

-(void) encodeInferenceBatch:(nonnull float*)userInputBuffer
       userInputGlobalBuffer:(nonnull float*)userInputGlobalBuffer
                policyOutput:(nonnull float*)policyOutput
                 valueOutput:(nonnull float*)valueOutput
             ownershipOutput:(nonnull float*)ownershipOutput
            miscValuesOutput:(nonnull float*)miscValuesOutput
        moreMiscValuesOutput:(nonnull float*)moreMiscValuesOutput
{
  MPSGraphTensorData* bin_inputs_data = [MPSGraphTensorData alloc];
  MPSGraphTensorData* global_inputs_data = [MPSGraphTensorData alloc];
  MPSGraphTensorData* symmetries_data = [MPSGraphTensorData alloc];
  MPSGraphTensorData* include_history_data = [MPSGraphTensorData alloc];
  NSArray<MPSGraphTensor*>* targetTensors = @[policy_output];
  
  dispatch_semaphore_wait(doubleBufferingSemaphore, DISPATCH_TIME_FOREVER);
  MPSCommandBuffer* commandBuffer = [MPSCommandBuffer commandBufferFromCommandQueue:commandQueue];
  MPSGraphExecutionDescriptor* executionDesc = [MPSGraphExecutionDescriptor alloc];
  
  executionDesc.completionHandler = ^(MPSGraphTensorDataDictionary* resultsDictionary, NSError* error) {
    dispatch_semaphore_signal(doubleBufferingSemaphore);
  };
  
  MPSGraphTensorDataDictionary* feeds = @{
    bin_inputs: bin_inputs_data,
    global_inputs: global_inputs_data,
    symmetries: symmetries_data,
    include_history: include_history_data
  };
  
  MPSGraphTensorDataDictionary* fetch = [graph encodeToCommandBuffer:commandBuffer
                                                               feeds:feeds
                                                       targetTensors:targetTensors
                                                    targetOperations:@[]
                                                 executionDescriptor:executionDesc];
  
  [commandBuffer commit];
  [commandBuffer waitUntilCompleted];

  int policySize = (nnXLen * nnYLen) + 1;

  for (NSUInteger index = 0; index < policySize; index++) {
    [[fetch[policy_output] mpsndarray] readBytes:&policyOutput[index]
                                     strideBytes:nil];
  }
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
                       const ModelDesc* modelDesc) {
  version = modelDesc->version;
  id<MTLDevice> device = MTLCreateSystemDefaultDevice();
  
  kataGoGraph = [[KataGoGraph alloc] initWithDevice:device
                                             nnXLen:nnXLen
                                             nnYLen:nnYLen
                                            version:version
                                   numInputChannels:modelDesc->numInputChannels
                             numInputGlobalChannels:modelDesc->numInputGlobalChannels
                                   numValueChannels:modelDesc->numValueChannels
                              numScoreValueChannels:modelDesc->numScoreValueChannels
                               numOwnershipChannels:modelDesc->numOwnershipChannels];
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
