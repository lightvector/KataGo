#import <Foundation/Foundation.h>
#import <CoreML/CoreML.h>
#include <stdint.h>
#include <os/log.h>

#if ! __has_feature(objc_arc)
#error This code must be compiled with Objective-C ARC! Did you compile with -fobjc-arc?
#endif

NS_ASSUME_NONNULL_BEGIN


/// Model Prediction Input Type
API_AVAILABLE(macos(12.0), ios(15.0), watchos(8.0), tvos(15.0)) __attribute__((visibility("hidden")))
@interface KataGoModelInput : NSObject<MLFeatureProvider>

/// swa_model_bin_inputs as 1 × 361 × 22 3-dimensional array of floats
@property (readwrite, nonatomic, strong) MLMultiArray * swa_model_bin_inputs;

/// swa_model_global_inputs as 1 by 19 matrix of floats
@property (readwrite, nonatomic, strong) MLMultiArray * swa_model_global_inputs;

/// swa_model_include_history as 1 by 5 matrix of floats
@property (readwrite, nonatomic, strong) MLMultiArray * swa_model_include_history;

/// swa_model_symmetries as 3 element vector of floats
@property (readwrite, nonatomic, strong) MLMultiArray * swa_model_symmetries;
- (instancetype)init NS_UNAVAILABLE;
- (instancetype)initWithSwa_model_bin_inputs:(MLMultiArray *)swa_model_bin_inputs swa_model_global_inputs:(MLMultiArray *)swa_model_global_inputs swa_model_include_history:(MLMultiArray *)swa_model_include_history swa_model_symmetries:(MLMultiArray *)swa_model_symmetries NS_DESIGNATED_INITIALIZER;

@end


/// Model Prediction Output Type
API_AVAILABLE(macos(12.0), ios(15.0), watchos(8.0), tvos(15.0)) __attribute__((visibility("hidden")))
@interface KataGoModelOutput : NSObject<MLFeatureProvider>

/// swa_model_miscvalues_output as multidimensional array of floats
@property (readwrite, nonatomic, strong) MLMultiArray * swa_model_miscvalues_output;

/// swa_model_moremiscvalues_output as multidimensional array of floats
@property (readwrite, nonatomic, strong) MLMultiArray * swa_model_moremiscvalues_output;

/// swa_model_ownership_output as multidimensional array of floats
@property (readwrite, nonatomic, strong) MLMultiArray * swa_model_ownership_output;

/// swa_model_policy_output as multidimensional array of floats
@property (readwrite, nonatomic, strong) MLMultiArray * swa_model_policy_output;

/// swa_model_value_output as multidimensional array of floats
@property (readwrite, nonatomic, strong) MLMultiArray * swa_model_value_output;
- (instancetype)init NS_UNAVAILABLE;
- (instancetype)initWithSwa_model_miscvalues_output:(MLMultiArray *)swa_model_miscvalues_output swa_model_moremiscvalues_output:(MLMultiArray *)swa_model_moremiscvalues_output swa_model_ownership_output:(MLMultiArray *)swa_model_ownership_output swa_model_policy_output:(MLMultiArray *)swa_model_policy_output swa_model_value_output:(MLMultiArray *)swa_model_value_output NS_DESIGNATED_INITIALIZER;

@end


/// Class for model loading and prediction
API_AVAILABLE(macos(12.0), ios(15.0), watchos(8.0), tvos(15.0)) __attribute__((visibility("hidden")))
@interface KataGoModel : NSObject
@property (readonly, nonatomic, nullable) MLModel * model;

/**
    Compile the MLModel
 */
+ (nullable MLModel *)compileMLModelWithXLen:(NSNumber * _Nonnull)xLen yLen:(NSNumber * _Nonnull)yLen;

/**
    URL of the underlying .mlmodelc directory.
*/
+ (nullable NSURL *)URLOfModelInThisBundle;

/**
    Initialize KataGoModel instance from an existing MLModel object.

    Usually the application does not use this initializer unless it makes a subclass of KataGoModel.
    Such application may want to use `-[MLModel initWithContentsOfURL:configuration:error:]` and `+URLOfModelInThisBundle` to create a MLModel object to pass-in.
*/
- (instancetype)initWithMLModel:(MLModel *)model NS_DESIGNATED_INITIALIZER;

/**
    Initialize KataGoModel instance with the model in this bundle.
*/
- (nullable instancetype)init;

/**
    Initialize KataGoModel instance with the model in this bundle.

    @param configuration The model configuration object
    @param error If an error occurs, upon return contains an NSError object that describes the problem. If you are not interested in possible errors, pass in NULL.
*/
- (nullable instancetype)initWithConfiguration:(MLModelConfiguration *)configuration error:(NSError * _Nullable __autoreleasing * _Nullable)error;

/**
    Initialize KataGoModel instance from the model URL.

    @param modelURL URL to the .mlmodelc directory for KataGoModel.
    @param error If an error occurs, upon return contains an NSError object that describes the problem. If you are not interested in possible errors, pass in NULL.
*/
- (nullable instancetype)initWithContentsOfURL:(NSURL *)modelURL error:(NSError * _Nullable __autoreleasing * _Nullable)error;

/**
    Initialize KataGoModel instance from the model URL.

    @param modelURL URL to the .mlmodelc directory for KataGoModel.
    @param configuration The model configuration object
    @param error If an error occurs, upon return contains an NSError object that describes the problem. If you are not interested in possible errors, pass in NULL.
*/
- (nullable instancetype)initWithContentsOfURL:(NSURL *)modelURL configuration:(MLModelConfiguration *)configuration error:(NSError * _Nullable __autoreleasing * _Nullable)error;

/**
    Make a prediction using the standard interface
    @param input an instance of KataGoModelInput to predict from
    @param options prediction options
    @param error If an error occurs, upon return contains an NSError object that describes the problem. If you are not interested in possible errors, pass in NULL.
    @return the prediction as KataGoModelOutput
*/
- (nullable KataGoModelOutput *)predictionFromFeatures:(KataGoModelInput *)input options:(MLPredictionOptions *)options error:(NSError * _Nullable __autoreleasing * _Nullable)error;

@end

NS_ASSUME_NONNULL_END

/// Class for CoreML backend
@interface CoreMLBackend : NSObject

/// CoreML model instance
@property (readonly) KataGoModel * _Nonnull model;

/// Board x length
@property (readonly) NSNumber * _Nonnull xLen;

/// Board y length
@property (readonly) NSNumber * _Nonnull yLen;

/// swa_model_include_history
@property (readonly) MLMultiArray * _Nonnull includeHistory;

/// swa_model_symmetries
@property (readonly) MLMultiArray * _Nonnull symmetries;

/**
    Get CoreML backend with model index
    @param index model index
*/
+ (CoreMLBackend * _Nonnull)getBackendAt:(NSNumber * _Nonnull)index;

/**
    Initialize CoreML backend with model index
    @param index model index
    @param xLen x-direction length
    @param yLen y-direction length
    @return Model version
*/
+ (NSNumber * _Nonnull)initWithIndex:(NSNumber * _Nonnull)index
                           modelXLen:(NSNumber * _Nonnull)xLen
                           modelYLen:(NSNumber * _Nonnull)yLen;

/**
    Initialize CoreML backend
*/
- (nullable instancetype)initWithMLModel:(MLModel * _Nonnull)model
                                    xLen:(NSNumber * _Nonnull)xLen
                                    yLen:(NSNumber * _Nonnull)yLen;

/**
    Get output from CoreML model
    @param binInputs bin inputs
    @param globalInputs global inputs
    @param policyOutputs policy outputs
    @param valueOutputs value outputs
    @param ownershipOutputs ownership outputs
    @param miscValueOutputs misc value outputs
    @param miscOwnershipOutputs misc ownership outputs
*/
- (void)getOutputWithBinInputs:(void * _Nonnull)binInputs
                  globalInputs:(void * _Nonnull)globalInputs
                  policyOutput:(void * _Nonnull)policyOutput
                   valueOutput:(void * _Nonnull)valueOutput
               ownershipOutput:(void * _Nonnull)ownershipOutput
              miscValuesOutput:(void * _Nonnull)miscValuesOutput
          moreMiscValuesOutput:(void * _Nonnull)moreMiscValuesOutput;
@end
