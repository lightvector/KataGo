#import "coremlmodel.h"

@implementation KataGoModelInput

- (instancetype)initWithSwa_model_bin_inputs:(MLMultiArray *)swa_model_bin_inputs swa_model_global_inputs:(MLMultiArray *)swa_model_global_inputs swa_model_include_history:(MLMultiArray *)swa_model_include_history swa_model_symmetries:(MLMultiArray *)swa_model_symmetries {
  self = [super init];
  if (self) {
    _swa_model_bin_inputs = swa_model_bin_inputs;
    _swa_model_global_inputs = swa_model_global_inputs;
    _swa_model_include_history = swa_model_include_history;
    _swa_model_symmetries = swa_model_symmetries;
  }
  return self;
}

- (NSSet<NSString *> *)featureNames {
  return [NSSet setWithArray:@[@"swa_model_bin_inputs", @"swa_model_global_inputs", @"swa_model_include_history", @"swa_model_symmetries"]];
}

- (nullable MLFeatureValue *)featureValueForName:(NSString *)featureName {
  if ([featureName isEqualToString:@"swa_model_bin_inputs"]) {
    return [MLFeatureValue featureValueWithMultiArray:_swa_model_bin_inputs];
  }
  if ([featureName isEqualToString:@"swa_model_global_inputs"]) {
    return [MLFeatureValue featureValueWithMultiArray:_swa_model_global_inputs];
  }
  if ([featureName isEqualToString:@"swa_model_include_history"]) {
    return [MLFeatureValue featureValueWithMultiArray:_swa_model_include_history];
  }
  if ([featureName isEqualToString:@"swa_model_symmetries"]) {
    return [MLFeatureValue featureValueWithMultiArray:_swa_model_symmetries];
  }
  return nil;
}

@end

@implementation KataGoModelOutput

- (instancetype)initWithSwa_model_miscvalues_output:(MLMultiArray *)swa_model_miscvalues_output swa_model_moremiscvalues_output:(MLMultiArray *)swa_model_moremiscvalues_output swa_model_ownership_output:(MLMultiArray *)swa_model_ownership_output swa_model_policy_output:(MLMultiArray *)swa_model_policy_output swa_model_value_output:(MLMultiArray *)swa_model_value_output {
  self = [super init];
  if (self) {
    _swa_model_miscvalues_output = swa_model_miscvalues_output;
    _swa_model_moremiscvalues_output = swa_model_moremiscvalues_output;
    _swa_model_ownership_output = swa_model_ownership_output;
    _swa_model_policy_output = swa_model_policy_output;
    _swa_model_value_output = swa_model_value_output;
  }
  return self;
}

- (NSSet<NSString *> *)featureNames {
  return [NSSet setWithArray:@[@"swa_model_miscvalues_output", @"swa_model_moremiscvalues_output", @"swa_model_ownership_output", @"swa_model_policy_output", @"swa_model_value_output"]];
}

- (nullable MLFeatureValue *)featureValueForName:(NSString *)featureName {
  if ([featureName isEqualToString:@"swa_model_miscvalues_output"]) {
    return [MLFeatureValue featureValueWithMultiArray:_swa_model_miscvalues_output];
  }
  if ([featureName isEqualToString:@"swa_model_moremiscvalues_output"]) {
    return [MLFeatureValue featureValueWithMultiArray:_swa_model_moremiscvalues_output];
  }
  if ([featureName isEqualToString:@"swa_model_ownership_output"]) {
    return [MLFeatureValue featureValueWithMultiArray:_swa_model_ownership_output];
  }
  if ([featureName isEqualToString:@"swa_model_policy_output"]) {
    return [MLFeatureValue featureValueWithMultiArray:_swa_model_policy_output];
  }
  if ([featureName isEqualToString:@"swa_model_value_output"]) {
    return [MLFeatureValue featureValueWithMultiArray:_swa_model_value_output];
  }
  return nil;
}

@end

@implementation KataGoModel

/**
 Compile the MLModel
 */
+ (nullable MLModel *)compileMLModelWithXLen:(NSNumber * _Nonnull)xLen yLen:(NSNumber * _Nonnull)yLen {
  NSString *modelName = [NSString stringWithFormat:@"KataGoModel%dx%d", xLen.intValue, yLen.intValue];

  NSString *modelPath = [[NSBundle bundleForClass:[self class]] pathForResource:modelName
                                                                         ofType:@"mlpackage"];

  if (nil == modelPath) {
    NSLog(@"ERROR: Could not load KataGoModel.mlpackage in the bundle resource");

    return nil;
  }

  NSURL *modelUrl = [NSURL fileURLWithPath:modelPath];

  NSLog(@"INFO: Loading KataGo Model from %@", modelUrl);

  NSURL *compiledUrl = [MLModel compileModelAtURL:modelUrl
                                            error:nil];

  MLModel *model = [MLModel modelWithContentsOfURL:compiledUrl error:nil];

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
 Initialize KataGoModel instance with the model in this bundle.

 @param configuration The model configuration object
 @param error If an error occurs, upon return contains an NSError object that describes the problem. If you are not interested in possible errors, pass in NULL.
 */
- (nullable instancetype)initWithConfiguration:(MLModelConfiguration *)configuration error:(NSError * _Nullable __autoreleasing * _Nullable)error {
  return [self initWithContentsOfURL:(NSURL * _Nonnull)self.class.URLOfModelInThisBundle configuration:configuration error:error];
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
  return [[KataGoModelOutput alloc] initWithSwa_model_miscvalues_output:(MLMultiArray *)[outFeatures featureValueForName:@"swa_model_miscvalues_output"].multiArrayValue swa_model_moremiscvalues_output:(MLMultiArray *)[outFeatures featureValueForName:@"swa_model_moremiscvalues_output"].multiArrayValue swa_model_ownership_output:(MLMultiArray *)[outFeatures featureValueForName:@"swa_model_ownership_output"].multiArrayValue swa_model_policy_output:(MLMultiArray *)[outFeatures featureValueForName:@"swa_model_policy_output"].multiArrayValue swa_model_value_output:(MLMultiArray *)[outFeatures featureValueForName:@"swa_model_value_output"].multiArrayValue];
}

@end
