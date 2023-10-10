#import "coremlmodel.h"

@implementation KataGoModelInput

- (instancetype)initWithInput_spatial:(MLMultiArray *)input_spatial input_global:(MLMultiArray *)input_global {
  self = [super init];
  if (self) {
    _input_spatial = input_spatial;
    _input_global = input_global;
  }
  return self;
}

- (NSSet<NSString *> *)featureNames {
  return [NSSet setWithArray:@[@"input_spatial", @"input_global"]];
}

- (nullable MLFeatureValue *)featureValueForName:(NSString *)featureName {
  if ([featureName isEqualToString:@"input_spatial"]) {
    return [MLFeatureValue featureValueWithMultiArray:_input_spatial];
  }
  if ([featureName isEqualToString:@"input_global"]) {
    return [MLFeatureValue featureValueWithMultiArray:_input_global];
  }
  return nil;
}

@end

@implementation KataGoModelOutput

- (instancetype)initWithOutput_policy:(MLMultiArray *)output_policy out_value:(MLMultiArray *)out_value out_miscvalue:(MLMultiArray *)out_miscvalue out_moremiscvalue:(MLMultiArray *)out_moremiscvalue out_ownership:(MLMultiArray *)out_ownership {
  self = [super init];
  if (self) {
    _output_policy = output_policy;
    _out_value = out_value;
    _out_miscvalue = out_miscvalue;
    _out_moremiscvalue = out_moremiscvalue;
    _out_ownership = out_ownership;
  }
  return self;
}

- (NSSet<NSString *> *)featureNames {
  return [NSSet setWithArray:@[@"output_policy", @"out_value", @"out_miscvalue", @"out_moremiscvalue", @"out_ownership"]];
}

- (nullable MLFeatureValue *)featureValueForName:(NSString *)featureName {
  if ([featureName isEqualToString:@"output_policy"]) {
    return [MLFeatureValue featureValueWithMultiArray:_output_policy];
  }
  if ([featureName isEqualToString:@"out_value"]) {
    return [MLFeatureValue featureValueWithMultiArray:_out_value];
  }
  if ([featureName isEqualToString:@"out_miscvalue"]) {
    return [MLFeatureValue featureValueWithMultiArray:_out_miscvalue];
  }
  if ([featureName isEqualToString:@"out_moremiscvalue"]) {
    return [MLFeatureValue featureValueWithMultiArray:_out_moremiscvalue];
  }
  if ([featureName isEqualToString:@"out_ownership"]) {
    return [MLFeatureValue featureValueWithMultiArray:_out_ownership];
  }
  return nil;
}

@end

@implementation KataGoModel

/// Compile MLModel from the bundle resource
/// - Parameters:
///   - xLen: x-direction of the board
///   - yLen: y-direction of the board
///   - useFP16: use FP16 or FP32
/// - Returns: compiled MLModel
+ (nullable MLModel *)compileMLModelWithXLen:(NSNumber * _Nonnull)xLen
                                        yLen:(NSNumber * _Nonnull)yLen
                                     useFP16:(NSNumber * _Nonnull)useFP16 {

  // Set compute precision name based on useFP16
  NSString *precisionName = useFP16.boolValue ? @"fp16" : @"fp32";

  // Set model name based on xLen, yLen, and precisionName
  NSString *modelName = [NSString stringWithFormat:@"KataGoModel%dx%d%@", xLen.intValue, yLen.intValue, precisionName];

  // Compile MLModel with the model name
  MLModel *model = [KataGoModel compileMLModelWithModelName:modelName];

  return model;
}


/// Compile the MLModel for KataGoModel and returns the compiled model.
/// - Parameters:
///   - modelName: The name of the MLModel.
+ (nullable MLModel *)compileMLModelWithModelName:(NSString * _Nonnull)modelName {

  // Get compiled model name
  NSString *compiledModelName = [NSString stringWithFormat:@"%@.mlmodelc", modelName];

  // Set the directory for KataGo models
  NSString *directory = @"KataGoModels";

  // Get path component
  NSString *pathComponent = [NSString stringWithFormat:@"%@/%@", directory, compiledModelName];

  // Get default file manager
  NSFileManager *fileManager = [NSFileManager defaultManager];

  // Get application support directory
  // Create the directory if it does not already exist
  NSURL *appSupportURL = [fileManager URLForDirectory:NSApplicationSupportDirectory
                                             inDomain:NSUserDomainMask
                                    appropriateForURL:nil
                                               create:true
                                                error:nil];

  // Create the URL for the permanent compiled model file
  NSURL *permanentURL = [appSupportURL URLByAppendingPathComponent:pathComponent];

  // Initialize model
  MLModel *model = nil;

  // Set model type name
  NSString *typeName = @"mlpackage";

  // Get model path from bundle resource
  NSString *modelPath = [[NSBundle bundleForClass:[self class]] pathForResource:modelName
                                                                         ofType:typeName];

  // Get model URL
  NSURL *modelURL = [NSURL fileURLWithPath:modelPath];

  // Get model data
  NSData *modelData = [NSData dataWithContentsOfURL:modelURL];

  // Initialize hash data
  NSMutableData *hashData = [NSMutableData dataWithLength:CC_SHA256_DIGEST_LENGTH];

  // Get SHA256 data
  CC_SHA256(modelData.bytes, (CC_LONG)modelData.length, hashData.mutableBytes);

  // Get hash digest
  NSString *digest = [hashData base64EncodedStringWithOptions:0];

  // Set digest path
  NSString *savedDigestPath = [NSString stringWithFormat:@"%@/%@.digest", directory, modelName];

  // Get digest URL
  NSURL *savedDigestURL = [appSupportURL URLByAppendingPathComponent:savedDigestPath];

  // Get saved digest
  NSString *savedDigest = [NSString stringWithContentsOfURL:savedDigestURL encoding:NSUTF8StringEncoding error:nil];

  // Check permanent compiled model is reachable
  BOOL reachableModel = [permanentURL checkResourceIsReachableAndReturnError:nil];

  if (!reachableModel) {
    NSLog(@"INFO: Compiling model because it is not reachable");
  }

  // Check the saved digest is changed or not
  BOOL isChangedDigest = ![digest isEqualToString:savedDigest];

  if (isChangedDigest) {
    NSLog(@"INFO: Compiling model because the digest has changed");
  }

  // Model should be compiled if the compiled model is not reachable or the digest changes
  BOOL shouldCompile = !reachableModel || isChangedDigest;

  if (shouldCompile) {
    if (nil == modelPath) {
      // If model is not found in bundle resource, return nil
      NSLog(@"ERROR: Could not load %@.%@ in the bundle resource", modelName, typeName);
      return model;
    } else {
      // If model is found in bundle resource, compile it and return the compiled model
      NSLog(@"INFO: Compiling model at %@", modelURL);

      // Compile the model
      NSURL *compiledURL = [MLModel compileModelAtURL:modelURL
                                                error:nil];

      NSLog(@"INFO: Copying model to the permanent location %@", permanentURL);

      // Create the directory for KataGo models
      BOOL success = [fileManager createDirectoryAtURL:[appSupportURL URLByAppendingPathComponent:directory]
                           withIntermediateDirectories:true
                                            attributes:nil
                                                 error:nil];

      assert(success);

      // Copy the file to the to the permanent location, replacing it if necessary
      success = [fileManager replaceItemAtURL:permanentURL
                                withItemAtURL:compiledURL
                               backupItemName:nil
                                      options:NSFileManagerItemReplacementUsingNewMetadataOnly
                             resultingItemURL:nil
                                        error:nil];

      assert(success);

      // Update the digest
      success = [digest writeToURL:savedDigestURL
                        atomically:YES
                          encoding:NSUTF8StringEncoding
                             error:nil];

      assert(success);
    }
  }

  // Initialize the model configuration
  MLModelConfiguration *configuration = [[MLModelConfiguration alloc] init];

  // Set the compute units to CPU and Neural Engine
  configuration.computeUnits = MLComputeUnitsCPUAndNeuralEngine;

  // Set the model display name
  configuration.modelDisplayName = modelName;

  NSLog(@"INFO: Creating model with contents %@", permanentURL);

  // Create the model
  model = [MLModel modelWithContentsOfURL:permanentURL
                            configuration:configuration
                                    error:nil];

  assert(model != nil);

  NSLog(@"INFO: Created model: %@", model.modelDescription.metadata[MLModelDescriptionKey]);

  // Return the model
  return model;
}


/**
 URL of the underlying .mlmodelc directory.
 */
+ (nullable NSURL *)URLOfModelInThisBundle {
  NSString *assetPath = [[NSBundle bundleForClass:[self class]] pathForResource:@"KataGoModel" ofType:@"mlmodelc"];
  if (nil == assetPath) { os_log_error(OS_LOG_DEFAULT, "Could not load KataGoModel.mlmodelc in the bundle resource"); return nil; }
  return [NSURL fileURLWithPath:assetPath];
}


/**
 Initialize KataGoModel instance from an existing MLModel object.

 Usually the application does not use this initializer unless it makes a subclass of KataGoModel.
 Such application may want to use `-[MLModel initWithContentsOfURL:configuration:error:]` and `+URLOfModelInThisBundle` to create a MLModel object to pass-in.
 */
- (instancetype)initWithMLModel:(MLModel *)model {
  self = [super init];
  if (!self) { return nil; }
  _model = model;
  if (_model == nil) { return nil; }
  return self;
}


/**
 Initialize KataGoModel instance with the model in this bundle.
 */
- (nullable instancetype)init {
  return [self initWithContentsOfURL:(NSURL * _Nonnull)self.class.URLOfModelInThisBundle error:nil];
}


/**
 Initialize KataGoModel instance from the model URL.

 @param modelURL URL to the .mlmodelc directory for KataGoModel.
 @param error If an error occurs, upon return contains an NSError object that describes the problem. If you are not interested in possible errors, pass in NULL.
 */
- (nullable instancetype)initWithContentsOfURL:(NSURL *)modelURL error:(NSError * _Nullable __autoreleasing * _Nullable)error {
  MLModel *model = [MLModel modelWithContentsOfURL:modelURL error:error];
  if (model == nil) { return nil; }
  return [self initWithMLModel:model];
}


/**
 Initialize KataGoModel instance from the model URL.

 @param modelURL URL to the .mlmodelc directory for KataGoModel.
 @param configuration The model configuration object
 @param error If an error occurs, upon return contains an NSError object that describes the problem. If you are not interested in possible errors, pass in NULL.
 */
- (nullable instancetype)initWithContentsOfURL:(NSURL *)modelURL configuration:(MLModelConfiguration *)configuration error:(NSError * _Nullable __autoreleasing * _Nullable)error {
  MLModel *model = [MLModel modelWithContentsOfURL:modelURL configuration:configuration error:error];
  if (model == nil) { return nil; }
  return [self initWithMLModel:model];
}

- (nullable KataGoModelOutput *)predictionFromFeatures:(KataGoModelInput *)input options:(MLPredictionOptions *)options error:(NSError * _Nullable __autoreleasing * _Nullable)error {
  id<MLFeatureProvider> outFeatures = [_model predictionFromFeatures:input options:options error:error];
  if (!outFeatures) { return nil; }
  return [[KataGoModelOutput alloc] initWithOutput_policy:(MLMultiArray *)[outFeatures featureValueForName:@"output_policy"].multiArrayValue out_value:(MLMultiArray *)[outFeatures featureValueForName:@"out_value"].multiArrayValue out_miscvalue:(MLMultiArray *)[outFeatures featureValueForName:@"out_miscvalue"].multiArrayValue out_moremiscvalue:(MLMultiArray *)[outFeatures featureValueForName:@"out_moremiscvalue"].multiArrayValue out_ownership:(MLMultiArray *)[outFeatures featureValueForName:@"out_ownership"].multiArrayValue];
}

@end
