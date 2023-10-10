#import <Foundation/Foundation.h>
#import <CoreML/CoreML.h>
#import <CommonCrypto/CommonDigest.h>
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

/// This is an initializer method in Objective-C that has been marked as unavailable.
- (instancetype)init NS_UNAVAILABLE;

/// Initializes a KataGoModelInput object and returns it. This method is marked with the NS_DESIGNATED_INITIALIZER macro, indicating that it is the primary designated initializer for the KataGoModelInput class.
/// - Parameters:
///   - input_spatial: an MLMultiArray representing a 4-dimensional array of floats with dimensions 1 × 22 × 19 × 19
///   - input_global: an MLMultiArray representing a 1-dimensional array of floats with size 19
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

/// This is an initializer method in Objective-C that has been marked as unavailable.
- (instancetype)init NS_UNAVAILABLE;

/// Initializes a KataGoModelOutput object and returns it. This method is marked with the NS_DESIGNATED_INITIALIZER macro, indicating that it is the primary designated initializer for the KataGoModelOutput class.
/// - Parameters:
///   - output_policy: The policy output of the model as an MLMultiArray containing multidimensional arrays of floats
///   - out_value: The value output of the model as an MLMultiArray containing multidimensional arrays of floats
///   - out_miscvalue: The miscellaneous value output of the model as an MLMultiArray containing multidimensional arrays of floats
///   - out_moremiscvalue: The more miscellaneous value output of the model as an MLMultiArray containing multidimensional arrays of floats
///   - out_ownership: The ownership output of the model as an MLMultiArray containing multidimensional arrays of floats
- (instancetype)initWithOutput_policy:(MLMultiArray *)output_policy out_value:(MLMultiArray *)out_value out_miscvalue:(MLMultiArray *)out_miscvalue out_moremiscvalue:(MLMultiArray *)out_moremiscvalue out_ownership:(MLMultiArray *)out_ownership NS_DESIGNATED_INITIALIZER;

@end


/// A class representing a compiled MLModel for loading and prediction of KataGoModel
API_AVAILABLE(macos(10.15), ios(13.0), watchos(6.0), tvos(13.0)) __attribute__((visibility("hidden")))
@interface KataGoModel : NSObject

/// The underlying MLModel object for this KataGoModel instance.
@property (readonly, nonatomic, nullable) MLModel * model;

/// Compile the MLModel for KataGoModel and returns the compiled model.
/// - Parameters:
///   - xLen: The X dimension of the input_spatial MLMultiArray.
///   - yLen: The Y dimension of the input_spatial MLMultiArray.
///   - useFP16: A boolean NSNumber that specifies whether to use 16-bit floating point precision for the input and output tensors of the compiled model.
+ (nullable MLModel *)compileMLModelWithXLen:(NSNumber *)xLen
                                        yLen:(NSNumber *)yLen
                                     useFP16:(NSNumber *)useFP16;

/// Compile the MLModel for KataGoModel and returns the compiled model.
/// - Parameters:
///   - modelName: The name of the MLModel.
+ (nullable MLModel *)compileMLModelWithModelName:(NSString *)modelName;

/// Returns the URL of the underlying .mlmodelc directory for KataGoModel.
+ (nullable NSURL *)URLOfModelInThisBundle;

/// Initializes a KataGoModel instance from an existing MLModel object.
/// Usually the application does not use this initializer unless it makes a subclass of KataGoModel.
/// Such application may want to use `-[MLModel initWithContentsOfURL:configuration:error:]` and `+URLOfModelInThisBundle` to create a MLModel object to pass-in.
/// @param model An MLModel object that will be used as the underlying model for this KataGoModel instance.
- (instancetype)initWithMLModel:(MLModel *)model NS_DESIGNATED_INITIALIZER;

///  Initializes a KataGoModel instance with the model in this bundle.
- (nullable instancetype)init;

/// Initializes a KataGoModel instance from a model URL.
/// @param modelURL URL to the .mlmodelc directory for KataGoModel.
/// @param error If an error occurs, upon return contains an NSError object that describes the problem. If you are not interested in possible errors, pass in NULL.
- (nullable instancetype)initWithContentsOfURL:(NSURL *)modelURL error:(NSError * _Nullable __autoreleasing * _Nullable)error;

/// Initializes a KataGoModel instance from a model URL with the specified configuration.
/// @param modelURL URL to the .mlmodelc directory for KataGoModel.
/// @param configuration The model configuration object.
/// @param error If an error occurs, upon return contains an NSError object that describes the problem. If you are not interested in possible errors, pass in NULL.
- (nullable instancetype)initWithContentsOfURL:(NSURL *)modelURL configuration:(MLModelConfiguration *)configuration error:(NSError * _Nullable __autoreleasing * _Nullable)error;

/// Make a prediction using the standard interface.
/// @param input An instance of KataGoModelInput to predict from.
/// @param options Prediction options.
/// @param error If an error occurs, upon return contains an NSError object that describes the problem. If you are not interested in possible errors, pass in NULL.
- (nullable KataGoModelOutput *)predictionFromFeatures:(KataGoModelInput *)input options:(MLPredictionOptions *)options error:(NSError * _Nullable __autoreleasing * _Nullable)error;

@end

/// A class that provides a CoreML backend for the application.
@interface CoreMLBackend : NSObject

/// The CoreML model instance used for prediction.
@property (readonly) KataGoModel * model;

/// The length of the board in the x-direction.
@property (readonly) NSNumber * xLen;

/// The length of the board in the y-direction.
@property (readonly) NSNumber * _Nonnull yLen;

/// The version number of the model.
@property (readonly) NSNumber * _Nonnull version;

/// The number of spatial features in the input.
@property (readonly) NSNumber * _Nonnull numSpatialFeatures;

/// The number of global features in the input.
@property (readonly) NSNumber * _Nonnull numGlobalFeatures;

/// Returns a CoreML backend instance for the model at the specified index.
/// - Parameter index: The index of the model to use.
+ (CoreMLBackend *)getBackendAt:(NSNumber *)index;

/// Returns the index for the next model.
+ (NSNumber *)getNextModelIndex;

/// Initializes the CoreML backend with the specified parameters.
/// @param xLen The length of the board in the x-direction.
/// @param yLen The length of the board in the y-direction.
/// @param useFP16 Whether to use 16-bit floating-point precision or not.
+ (NSNumber *)initWithModelXLen:(NSNumber *)xLen
                      modelYLen:(NSNumber *)yLen
                        useFP16:(NSNumber *)useFP16;

/// Initializes the CoreML backend with the specified ML model and parameters.
/// @param model The ML model to use for prediction.
/// @param xLen The length of the board in the x-direction.
/// @param yLen The length of the board in the y-direction.
- (nullable instancetype)initWithMLModel:(MLModel *)model
                                    xLen:(NSNumber *)xLen
                                    yLen:(NSNumber *)yLen;

/// Returns the output of the CoreML model for the specified inputs.
/// @param binInputs The binary inputs.
/// @param globalInputs The global inputs.
/// @param policyOutputs The policy outputs.
/// @param valueOutputs The value outputs.
/// @param ownershipOutputs The ownership outputs.
/// @param miscValueOutputs The miscellaneous value outputs.
/// @param moreMiscValueOutputs The more miscellaneous value outputs.
- (void)getOutputWithBinInputs:(void *)binInputs
                  globalInputs:(void *)globalInputs
                 policyOutputs:(void *)policyOutputs
                  valueOutputs:(void *)valueOutputs
              ownershipOutputs:(void *)ownershipOutputs
              miscValueOutputs:(void *)miscValueOutputs
          moreMiscValueOutputs:(void *)moreMiscValueOutputs;
@end

NS_ASSUME_NONNULL_END
