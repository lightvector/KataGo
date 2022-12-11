#import <Foundation/Foundation.h>
#import <CoreML/MLMultiArray.h>
#import "coremlmodel.h"

// This is the CoreMLBackend class.
@implementation CoreMLBackend

// This is the CoreMLBackend dictionary getter method.
// It is a singleton object that is used to store the CoreML models.
+ (NSMutableDictionary * _Nonnull)getBackends {
  // This is the CoreMLBackend dictionary.
  static NSMutableDictionary * backends = nil;

  @synchronized (self) {
    if (backends == nil) {
      // Two threads run with two CoreML backends in parallel.
      backends = [NSMutableDictionary dictionaryWithCapacity:2];
    }
  }

  return backends;
}

// This is the CoreMLBackend getter method.
// If the backend is not in the dictionary, it is initialized.
+ (CoreMLBackend * _Nonnull)getBackendAt:(NSNumber * _Nonnull)index {
  NSMutableDictionary * backends = [CoreMLBackend getBackends];

  return backends[index];
}

// This is the CoreMLBackend factory method.
// It is used to create a CoreMLBackend object.
// The CoreMLBackend object is stored in the dictionary.
// The CoreMLBackend object is initialized with the CoreML model.
// The ML model version is returned.
+ (NSNumber * _Nonnull)initWithIndex:(NSNumber * _Nonnull)index
                           modelXLen:(NSNumber * _Nonnull)xLen
                           modelYLen:(NSNumber * _Nonnull)yLen {
  NSMutableDictionary * backends = [CoreMLBackend getBackends];

  @synchronized (self) {
    if (backends[index] == nil) {
      MLModel * mlmodel = [KataGoModel compileMLModelWithXLen:xLen
                                                         yLen:yLen];

      backends[index] = [[CoreMLBackend alloc] initWithMLModel:mlmodel
                                                          xLen:xLen
                                                          yLen:yLen];
    }
  }

  return ((CoreMLBackend *)backends[index])->_model.model.modelDescription.metadata[MLModelVersionStringKey];
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

// Get the model's output.
- (void)getOutputWithBinInputs:(void * _Nonnull)binInputs
                  globalInputs:(void * _Nonnull)globalInputs
                  policyOutput:(void * _Nonnull)policyOutput
                   valueOutput:(void * _Nonnull)valueOutput
               ownershipOutput:(void * _Nonnull)ownershipOutput
              miscValuesOutput:(void * _Nonnull)miscValuesOutput
          moreMiscValuesOutput:(void * _Nonnull)moreMiscValuesOutput {
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
      ((float *)policyOutput)[i] = output.output_policy[i].floatValue;
    }

    for (int i = 0; i < output.out_value.count; i++) {
      ((float *)valueOutput)[i] = output.out_value[i].floatValue;
    }

    for (int i = 0; i < output.out_ownership.count; i++) {
      ((float *)ownershipOutput)[i] = output.out_ownership[i].floatValue;
    }

    for (int i = 0; i < output.out_miscvalue.count; i++) {
      ((float *)miscValuesOutput)[i] = output.out_miscvalue[i].floatValue;
    }

    for (int i = 0; i < output.out_moremiscvalue.count; i++) {
      ((float *)moreMiscValuesOutput)[i] = output.out_moremiscvalue[i].floatValue;
    }

  }
}

@end

// Initialize the CoreMLBackend dictionary.
void initCoreMLBackends() {
  (void)[CoreMLBackend getBackends];
}

// Create the CoreMLBackend instance.
// The ML model version is returned.
int createCoreMLBackend(int modelIndex, int modelXLen, int modelYLen, int serverThreadIdx) {
  NSLog(@"Metal backend thread %d: CoreML-#%d-%dx%d", serverThreadIdx, modelIndex, modelXLen, modelYLen);

  NSNumber * version = [CoreMLBackend initWithIndex:[NSNumber numberWithInt:modelIndex]
                                          modelXLen:[NSNumber numberWithInt:modelXLen]
                                          modelYLen:[NSNumber numberWithInt:modelYLen]];

  return version.intValue;
}

// Reset the CoreMLBackend instance.
void freeCoreMLBackend(int modelIndex) {
  [CoreMLBackend releaseWithIndex:[NSNumber numberWithInt:modelIndex]];
}

// Get the model's number of spatial features.
int getCoreMLBackendNumSpatialFeatures(int modelIndex) {
  return [[[CoreMLBackend getBackendAt:[NSNumber numberWithInt:modelIndex]] numSpatialFeatures] intValue];
}

// Get the model's number of global features.
int getCoreMLBackendNumGlobalFeatures(int modelIndex) {
  return [[[CoreMLBackend getBackendAt:[NSNumber numberWithInt:modelIndex]] numGlobalFeatures] intValue];
}

// Get the model's output.
void getCoreMLBackendOutput(float* userInputBuffer,
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
                   policyOutput:policyOutput
                    valueOutput:valueOutput
                ownershipOutput:ownershipOutput
               miscValuesOutput:miscValuesOutput
           moreMiscValuesOutput:moreMiscValuesOutput];
}
