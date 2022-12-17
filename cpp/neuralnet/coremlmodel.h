#import <Foundation/Foundation.h>
#import <CoreML/CoreML.h>
#include <stdint.h>
#include <os/log.h>

#if ! __has_feature(objc_arc)
#error This code must be compiled with Objective-C ARC! Did you compile with -fobjc-arc?
#endif

NS_ASSUME_NONNULL_BEGIN


/// Model Prediction Input Type
API_AVAILABLE(macos(10.15), ios(13.0), watchos(6.0), tvos(13.0)) __attribute__((visibility("hidden")))
@interface KataGoModelInput : NSObject<MLFeatureProvider>

/// input_spatial as 1 × 22 × 19 × 19 4-dimensional array of floats
@property (readwrite, nonatomic, strong) MLMultiArray * input_spatial;

/// input_global as 1 by 19 matrix of floats
@property (readwrite, nonatomic, strong) MLMultiArray * input_global;
- (instancetype)init NS_UNAVAILABLE;
- (instancetype)initWithInput_spatial:(MLMultiArray *)input_spatial input_global:(MLMultiArray *)input_global NS_DESIGNATED_INITIALIZER;

@end


/// Model Prediction Output Type
API_AVAILABLE(macos(10.15), ios(13.0), watchos(6.0), tvos(13.0)) __attribute__((visibility("hidden")))
@interface KataGoModelOutput : NSObject<MLFeatureProvider>

/// output_policy as multidimensional array of floats
@property (readwrite, nonatomic, strong) MLMultiArray * output_policy;

/// out_value as multidimensional array of floats
@property (readwrite, nonatomic, strong) MLMultiArray * out_value;

/// out_miscvalue as multidimensional array of floats
@property (readwrite, nonatomic, strong) MLMultiArray * out_miscvalue;

/// out_moremiscvalue as multidimensional array of floats
@property (readwrite, nonatomic, strong) MLMultiArray * out_moremiscvalue;

/// out_ownership as multidimensional array of floats
@property (readwrite, nonatomic, strong) MLMultiArray * out_ownership;
- (instancetype)init NS_UNAVAILABLE;
- (instancetype)initWithOutput_policy:(MLMultiArray *)output_policy out_value:(MLMultiArray *)out_value out_miscvalue:(MLMultiArray *)out_miscvalue out_moremiscvalue:(MLMultiArray *)out_moremiscvalue out_ownership:(MLMultiArray *)out_ownership NS_DESIGNATED_INITIALIZER;

@end


/// Class for model loading and prediction
API_AVAILABLE(macos(10.15), ios(13.0), watchos(6.0), tvos(13.0)) __attribute__((visibility("hidden")))
@interface KataGoModel : NSObject
@property (readonly, nonatomic, nullable) MLModel * model;

/**
    Compile the MLModel
 */
+ (nullable MLModel *)compileMLModelWithXLen:(NSNumber * _Nonnull)xLen
                                        yLen:(NSNumber * _Nonnull)yLen
                                     useFP16:(NSNumber * _Nonnull)useFP16;

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

/// Model version
@property (readonly) NSNumber * _Nonnull version;

/// Number of spatial features
@property (readonly) NSNumber * _Nonnull numSpatialFeatures;

/// Number of global features
@property (readonly) NSNumber * _Nonnull numGlobalFeatures;

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
    @param useFP16 use FP16 or not
    @return Model version
*/
+ (NSNumber * _Nonnull)initWithIndex:(NSNumber * _Nonnull)index
                           modelXLen:(NSNumber * _Nonnull)xLen
                           modelYLen:(NSNumber * _Nonnull)yLen
                             useFP16:(NSNumber * _Nonnull)useFP16;

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
