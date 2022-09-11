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
+ (void)initWithIndex:(NSNumber * _Nonnull)index
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

  _includeHistory = [[MLMultiArray alloc] initWithShape:@[@1, @5]
                                               dataType:MLMultiArrayDataTypeFloat
                                                  error:nil];

  for (int x = 0; x < 5; x++) {
    NSNumber *xSubscript = [NSNumber numberWithInt:x];

    // Set the value of the array at the subscript.
    [_includeHistory setObject:@1.0
             forKeyedSubscript:@[@0, xSubscript]];
  }

  _symmetries = [[MLMultiArray alloc] initWithShape:@[@3]
                                           dataType:MLMultiArrayDataTypeFloat
                                              error:nil];

  for (int x = 0; x < 3; x++) {
    NSNumber *xSubscript = [NSNumber numberWithInt:x];

    // Set the value of the array at the subscript.
    [_symmetries setObject:@0
         forKeyedSubscript:@[xSubscript]];
  }

  return self;
}

// Get the model's output.
- (void)getOutputWithBinInputs:(void * _Nonnull)binInputs
                  globalInputs:(void * _Nonnull)globalInputs
                  policyOutput:(void * _Nonnull)policyOutput
                   valueOutput:(void * _Nonnull)valueOutput
               ownershipOutput:(void * _Nonnull)ownershipOutput
              miscValuesOutput:(void * _Nonnull)miscValuesOutput
          moreMiscValuesOutput:(void * _Nonnull)moreMiscValuesOutput {
  @autoreleasepool {
    NSNumber * boardSize = [NSNumber numberWithInt:(_xLen.intValue * _yLen.intValue)];

    MLMultiArray * bin_inputs_array = [[MLMultiArray alloc] initWithDataPointer:binInputs
                                                                          shape:@[@1, boardSize, @22]
                                                                       dataType:MLMultiArrayDataTypeFloat
                                                                        strides:@[@1, @1, boardSize]
                                                                    deallocator:nil
                                                                          error:nil];

    MLMultiArray * global_inputs_array = [[MLMultiArray alloc] initWithDataPointer:globalInputs
                                                                             shape:@[@1, @19]
                                                                          dataType:MLMultiArrayDataTypeFloat
                                                                           strides:@[@1, @1]
                                                                       deallocator:nil
                                                                             error:nil];

    KataGoModelInput * input =
    [[KataGoModelInput alloc] initWithSwa_model_bin_inputs:bin_inputs_array
                                   swa_model_global_inputs:global_inputs_array
                                 swa_model_include_history:_includeHistory
                                      swa_model_symmetries:_symmetries];

    MLPredictionOptions * options = [[MLPredictionOptions alloc] init];

    KataGoModelOutput * output = [_model predictionFromFeatures:input
                                                        options:options
                                                          error:nil];

    // Copy the output to the output pointer.
    for (int i = 0; i < output.swa_model_policy_output.count; i++) {
      ((float *)policyOutput)[i] = output.swa_model_policy_output[i].floatValue;
    }

    for (int i = 0; i < output.swa_model_value_output.count; i++) {
      ((float *)valueOutput)[i] = output.swa_model_value_output[i].floatValue;
    }

    for (int i = 0; i < output.swa_model_ownership_output.count; i++) {
      ((float *)ownershipOutput)[i] = output.swa_model_ownership_output[i].floatValue;
    }

    for (int i = 0; i < output.swa_model_miscvalues_output.count; i++) {
      ((float *)miscValuesOutput)[i] = output.swa_model_miscvalues_output[i].floatValue;
    }

    for (int i = 0; i < output.swa_model_moremiscvalues_output.count; i++) {
      ((float *)moreMiscValuesOutput)[i] = output.swa_model_moremiscvalues_output[i].floatValue;
    }

  }
}

@end

// Initialize the CoreMLBackend dictionary.
void initCoreMLBackends() {
  (void)[CoreMLBackend getBackends];
}

// Create the CoreMLBackend instance.
void createCoreMLBackend(int modelIndex, int modelXLen, int modelYLen) {
  [CoreMLBackend initWithIndex:[NSNumber numberWithInt:modelIndex]
                     modelXLen:[NSNumber numberWithInt:modelXLen]
                     modelYLen:[NSNumber numberWithInt:modelYLen]];
}

// Reset the CoreMLBackend instance.
void freeCoreMLBackend(int modelIndex) {
  [CoreMLBackend releaseWithIndex:[NSNumber numberWithInt:modelIndex]];
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
