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

/**
 Compile the MLModel
 */
+ (nullable MLModel *)compileMLModelWithXLen:(NSNumber * _Nonnull)xLen yLen:(NSNumber * _Nonnull)yLen {
  NSString *modelName = [NSString stringWithFormat:@"KataGoModel%dx%dv11", xLen.intValue, yLen.intValue];

  NSString *typeName = @"mlmodel";

  NSString *modelPath = [[NSBundle bundleForClass:[self class]] pathForResource:modelName
                                                                         ofType:typeName];

  if (nil == modelPath) {
    NSLog(@"ERROR: Could not load %@.%@ in the bundle resource", modelName, typeName);

    return nil;
  }

  NSURL *modelUrl = [NSURL fileURLWithPath:modelPath];

  NSLog(@"INFO: Loading KataGo Model from %@", modelUrl);

  NSURL *compiledUrl = [MLModel compileModelAtURL:modelUrl
                                            error:nil];

  MLModel *model = [MLModel modelWithContentsOfURL:compiledUrl error:nil];

  NSLog(@"Loaded KataGo Model: %@", model.modelDescription.metadata[MLModelDescriptionKey]);

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
