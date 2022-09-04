#import <Foundation/Foundation.h>
#import <CoreML/MLMultiArray.h>
#import "coremlmodel.h"

// This is the CoreMLBackend dictionary.
// It is a singleton object that is used to store the CoreML model.
// Two threads run with two CoreML models in parallel.
static NSMutableDictionary * models = [NSMutableDictionary dictionaryWithCapacity:2];

// This is the CoreMLBackend class.
@implementation CoreMLBackend

// This is the CoreMLBackend getter method.
// If the model is not in the dictionary, it is initialized.
+ (CoreMLBackend * _Nonnull)getModelAt:(NSNumber * _Nonnull)index {
  return models[index];
}

// This is the CoreMLBackend constructor.
- (nullable instancetype)init {
  self = [super init];
  NSError *error = nil;
  _model = [[KataGoModel alloc] init];
  
  _includeHistory = [[MLMultiArray alloc] initWithShape:@[@1, @5]
                                               dataType:MLMultiArrayDataTypeFloat
                                                  error:&error];

  for (int x = 0; x < 5; x++) {
    NSNumber *xSubscript = [NSNumber numberWithInt:x];
    
    // Set the value of the array at the subscript.
    [_includeHistory setObject:@1.0
             forKeyedSubscript:@[@0, xSubscript]];
  }

  _symmetries = [[MLMultiArray alloc] initWithShape:@[@3]
                                           dataType:MLMultiArrayDataTypeFloat
                                              error:&error];

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
    NSError *error = nil;

    MLMultiArray * bin_inputs_array = [[MLMultiArray alloc] initWithDataPointer:binInputs
                                                                          shape:@[@1, @361, @22]
                                                                       dataType:MLMultiArrayDataTypeFloat
                                                                        strides:@[@1, @1, @361]
                                                                    deallocator:nil
                                                                          error:&error];

    MLMultiArray * global_inputs_array = [[MLMultiArray alloc] initWithDataPointer:globalInputs
                                                                             shape:@[@1, @19]
                                                                          dataType:MLMultiArrayDataTypeFloat
                                                                           strides:@[@1, @1]
                                                                       deallocator:nil
                                                                             error:&error];

    KataGoModelInput * input =
    [[KataGoModelInput alloc] initWithSwa_model_bin_inputs:bin_inputs_array
                                   swa_model_global_inputs:global_inputs_array
                                 swa_model_include_history:_includeHistory
                                      swa_model_symmetries:_symmetries];

    MLPredictionOptions * options = [[MLPredictionOptions alloc] init];

    KataGoModelOutput * output = [_model predictionFromFeatures:input
                                                        options:options
                                                          error:&error];

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

    [output release];
    [options release];
    [input release];
    [global_inputs_array release];
    [bin_inputs_array release];
  }
}

@end

// Initialize the CoreMLBackend class.
void initCoreMLBackend(int modelIndex) {
  NSNumber * index = [NSNumber numberWithInt:modelIndex];
  models[index] = [[CoreMLBackend alloc] init];
}

void resetCoreMLBackend(int modelIndex) {
  NSNumber * index = [NSNumber numberWithInt:modelIndex];
  [models[index] release];
  models[index] = nil;
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
  @autoreleasepool {
    CoreMLBackend* model = [CoreMLBackend getModelAt:[NSNumber numberWithInt:modelIndex]];

    [model getOutputWithBinInputs:userInputBuffer
                     globalInputs:userInputGlobalBuffer
                     policyOutput:policyOutput
                      valueOutput:valueOutput
                  ownershipOutput:ownershipOutput
                 miscValuesOutput:miscValuesOutput
             moreMiscValuesOutput:moreMiscValuesOutput];
  }
}
