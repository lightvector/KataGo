#import <Foundation/Foundation.h>
#import <CoreML/MLMultiArray.h>
#import "coremlmodel.h"
#import "coremlbackend.h"

// This is the CoreMLBackend class.
@implementation CoreMLBackend

/// Handle CoreMLBackend dictionary with a command, and return the CoreMLBackend dictionary.
/// - Parameter command: "clear" to remove all objects from the dictionary"; otherwise, do nothing.
+ (NSMutableDictionary * _Nonnull)handleBackendsWithCommand:(NSString * _Nonnull) command {
  // This is the CoreMLBackend dictionary.
  static NSMutableDictionary * backends = nil;

  @synchronized (self) {
    if (backends == nil) {
      // Two threads run with two CoreML backends in parallel.
      backends = [NSMutableDictionary dictionaryWithCapacity:2];
    }
  }

  if ([command isEqualToString:@"clear"]) {
    @synchronized (self) {
      [backends removeAllObjects];
    }
  }

  return backends;
}

// This is the CoreMLBackend dictionary getter method.
// It is a singleton object that is used to store the CoreML models.
+ (NSMutableDictionary * _Nonnull)getBackends {
  return [CoreMLBackend handleBackendsWithCommand:@"get"];
}

// This is the CoreMLBackend dictionary clear method.
// It is used to clear the CoreMLBackend dictionary.
+ (void)clearBackends {
  [CoreMLBackend handleBackendsWithCommand:@"clear"];
}

/// Get the next model index
+ (NSNumber * _Nonnull)getNextModelIndex {
  // This is the CoreMLBackend index.
  static NSNumber * modelIndex = nil;

  @synchronized (self) {
    if (modelIndex == nil) {
      // The first CoreMLBackend index is 0.
      modelIndex = [NSNumber numberWithInt:0];
    } else {
      // The next CoreMLBackend index is the current index + 1.
      modelIndex = [NSNumber numberWithInt:[modelIndex intValue] + 1];
    }
  }

  // The CoreMLBackend index is returned.
  return modelIndex;
}

// This is the CoreMLBackend getter method.
+ (CoreMLBackend * _Nonnull)getBackendAt:(NSNumber * _Nonnull)index {
  NSMutableDictionary * backends = [CoreMLBackend getBackends];

  return backends[index];
}

/// This is the CoreMLBackend factory method, which is used to create a CoreMLBackend object. The CoreMLBackend object is stored in the dictionary.
/// - Parameters:
///   - xLen: x-direction length
///   - yLen: y-direction length
///   - useFP16: use FP16 or not
/// - Returns: model index
+ (NSNumber * _Nonnull)initWithModelXLen:(NSNumber * _Nonnull)xLen
                               modelYLen:(NSNumber * _Nonnull)yLen
                                 useFP16:(NSNumber * _Nonnull)useFP16 {
  // The CoreMLBackend dictionary is retrieved.
  NSMutableDictionary * backends = [CoreMLBackend getBackends];

  // The next ML model index is retrieved.
  NSNumber * modelIndex = [CoreMLBackend getNextModelIndex];

  @synchronized (self) {
    // Get the model string
    string modelString = CoreMLProcess::getModelName(useFP16.boolValue);

    // Create the model name
    NSString * modelName = [NSString stringWithUTF8String:modelString.c_str()];

    // Compile the model in Application Support
    MLModel * mlmodel = [KataGoModel compileAppMLModelWithModelName:modelName];

    if (mlmodel == nil) {
      // Compile the model in Bundle
      mlmodel = [KataGoModel compileBundleMLModelWithModelName:modelName];
    }

    assert(mlmodel != nil);

    // The CoreMLBackend object is created.
    backends[modelIndex] = [[CoreMLBackend alloc] initWithMLModel:mlmodel
                                                             xLen:xLen
                                                             yLen:yLen];
  }

  // The ML model index is returned.
  return modelIndex;
}

// This is the CoreMLBackend destruction method.
// It is used to destroy a CoreMLBackend object.
// The CoreMLBackend object is removed from the dictionary.
+ (void)releaseWithIndex:(NSNumber * _Nonnull)index {
  NSMutableDictionary * backends = [CoreMLBackend getBackends];

  @synchronized (self) {
    backends[index] = nil;
  }
}

// This is the CoreMLBackend constructor.
- (nullable instancetype)initWithMLModel:(MLModel * _Nonnull)model
                                    xLen:(NSNumber * _Nonnull)xLen
                                    yLen:(NSNumber * _Nonnull)yLen {
  self = [super init];
  _model = [[KataGoModel alloc] initWithMLModel:model];
  _xLen = xLen;
  _yLen = yLen;

  // The model version must be at least 8.
  _version = model.modelDescription.metadata[MLModelVersionStringKey];
  NSAssert1(_version.intValue >= 8, @"version must not be smaller than 8: %@", _version);

  // The number of spatial features must be 22.
  _numSpatialFeatures = [NSNumber numberWithInt:22];

  // The number of global features must be 19.
  _numGlobalFeatures = [NSNumber numberWithInt:19];

  return self;
}

@synthesize numSpatialFeatures = _numSpatialFeatures;
@synthesize numGlobalFeatures = _numGlobalFeatures;
@synthesize version = _version;

// Get the model's output.
- (void)getOutputWithBinInputs:(void * _Nonnull)binInputs
                  globalInputs:(void * _Nonnull)globalInputs
                 policyOutputs:(void * _Nonnull)policyOutputs
                  valueOutputs:(void * _Nonnull)valueOutputs
              ownershipOutputs:(void * _Nonnull)ownershipOutputs
              miscValueOutputs:(void * _Nonnull)miscValuesOutputs
          moreMiscValueOutputs:(void * _Nonnull)moreMiscValuesOutputs {
  @autoreleasepool {
    // Strides are used to access the data in the MLMultiArray.
    NSArray * strides = @[[NSNumber numberWithInt:(_numSpatialFeatures.intValue) * (_yLen.intValue) * (_xLen.intValue)],
                          [NSNumber numberWithInt:(_yLen.intValue) * (_xLen.intValue)],
                          _yLen,
                          @1];

    // Create the MLMultiArray for the spatial features.
    MLMultiArray * bin_inputs_array = [[MLMultiArray alloc] initWithDataPointer:binInputs
                                                                          shape:@[@1, _numSpatialFeatures, _yLen, _xLen]
                                                                       dataType:MLMultiArrayDataTypeFloat
                                                                        strides:strides
                                                                    deallocator:nil
                                                                          error:nil];

    // Create the MLMultiArray for the global features.
    MLMultiArray * global_inputs_array = [[MLMultiArray alloc] initWithDataPointer:globalInputs
                                                                             shape:@[@1, _numGlobalFeatures]
                                                                          dataType:MLMultiArrayDataTypeFloat
                                                                           strides:@[_numGlobalFeatures, @1]
                                                                       deallocator:nil
                                                                             error:nil];

    KataGoModelInput * input =
    [[KataGoModelInput alloc] initWithInput_spatial:bin_inputs_array
                                       input_global:global_inputs_array];

    MLPredictionOptions * options = [[MLPredictionOptions alloc] init];

    KataGoModelOutput * output = [_model predictionFromFeatures:input
                                                        options:options
                                                          error:nil];
  
    // Copy the output to the output buffers.
    for (int i = 0; i < output.output_policy.count; i++) {
      ((float *)policyOutputs)[i] = output.output_policy[i].floatValue;
    }

    for (int i = 0; i < output.out_value.count; i++) {
      ((float *)valueOutputs)[i] = output.out_value[i].floatValue;
    }

    for (int i = 0; i < output.out_ownership.count; i++) {
      ((float *)ownershipOutputs)[i] = output.out_ownership[i].floatValue;
    }

    for (int i = 0; i < output.out_miscvalue.count; i++) {
      ((float *)miscValuesOutputs)[i] = output.out_miscvalue[i].floatValue;
    }

    for (int i = 0; i < output.out_moremiscvalue.count; i++) {
      ((float *)moreMiscValuesOutputs)[i] = output.out_moremiscvalue[i].floatValue;
    }

  }
}

@end

/// Create the CoreMLBackend context.
void CoreMLProcess::createCoreMLContext() {
  (void)[CoreMLBackend getBackends];
}

/// Destroy the CoreMLBackend context.
void CoreMLProcess::destroyCoreMLContext() {
  (void)[CoreMLBackend clearBackends];
}

/// Create the CoreMLBackend instance.
/// - Parameters:
///   - modelXLen: model x-direction length
///   - modelYLen: model y-direction length
///   - serverThreadIdx: server thread index
///   - useFP16: use FP16 or not
/// - Returns: model index
int CoreMLProcess::createCoreMLBackend(int modelXLen, int modelYLen, int serverThreadIdx, bool useFP16) {
  // Load the model.
  NSNumber * modelIndex = [CoreMLBackend initWithModelXLen:[NSNumber numberWithInt:modelXLen]
                                                 modelYLen:[NSNumber numberWithInt:modelYLen]
                                                   useFP16:[NSNumber numberWithBool:useFP16]];

  NSLog(@"CoreML backend thread %d: #%@-%dx%d useFP16 %d", serverThreadIdx, modelIndex, modelXLen, modelYLen, useFP16);

  // Return the model index.
  return modelIndex.intValue;
}

// Reset the CoreMLBackend instance.
void CoreMLProcess::freeCoreMLBackend(int modelIndex) {
  [CoreMLBackend releaseWithIndex:[NSNumber numberWithInt:modelIndex]];
}

/// Get the model's version.
/// - Parameter modelIndex: model index
int CoreMLProcess::getCoreMLBackendVersion(int modelIndex) {
  return [[[CoreMLBackend getBackendAt:[NSNumber numberWithInt:modelIndex]] version] intValue];
}

// Get the model's output.
void CoreMLProcess::getCoreMLHandleOutput(float* userInputBuffer,
                                          float* userInputGlobalBuffer,
                                          float* policyOutput,
                                          float* valueOutput,
                                          float* ownershipOutput,
                                          float* miscValuesOutput,
                                          float* moreMiscValuesOutput,
                                          int modelIndex) {
  CoreMLBackend* model = [CoreMLBackend getBackendAt:[NSNumber numberWithInt:modelIndex]];

  [model getOutputWithBinInputs:userInputBuffer
                   globalInputs:userInputGlobalBuffer
                  policyOutputs:policyOutput
                   valueOutputs:valueOutput
               ownershipOutputs:ownershipOutput
               miscValueOutputs:miscValuesOutput
           moreMiscValueOutputs:moreMiscValuesOutput];
}
