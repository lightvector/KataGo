#include "../neuralnet/nninputs.h"
#include "../neuralnet/modelversion.h"

//Old model versions, no longer supported:
//0 = V1 features, with old head architecture using crelus (no longer supported)
//1 = V1 features, with new head architecture, no crelus
//2 = V2 features, no internal architecture change.

//Supported model versions:
//3 = V3 features, many architecture changes for new selfplay loop, including multiple board sizes
//4 = V3 features, scorebelief head
//5 = V4 features, changed current territory feature to just indicate pass-alive
//6 = V5 features, disable fancy features
//7 = V6 features, support new rules configurations
//8 = V7 features, unbalanced training, button go, lead and variance time
//9 = V7 features, shortterm value error
//10 = V7 features, shortterm value error done more properly
//11 = V7 features, supports mish activations by desc actually reading the activations
//12 = V7 features, optimisic policy head
//13 = V7 features, Adjusted scaling on shortterm score variance, and made C++ side read in scalings.
//14 = V7 features, Squared softplus for error variance predictions
//15 = V7 features, Extra nonlinearity for pass output
//16 = V7 features, Q value predictions in the policy head

static void fail(int modelVersion) {
  throw StringError("NNModelVersion: Model version not currently implemented or supported: " + Global::intToString(modelVersion));
}

static_assert(NNModelVersion::oldestModelVersionImplemented == 3, "");
static_assert(NNModelVersion::oldestInputsVersionImplemented == 3, "");
static_assert(NNModelVersion::latestModelVersionImplemented == 16, "");
static_assert(NNModelVersion::latestInputsVersionImplemented == 7, "");

int NNModelVersion::getInputsVersion(int modelVersion) {
  if(modelVersion >= 8 && modelVersion <= 16)
    return 7;
  else if(modelVersion == 7)
    return 6;
  else if(modelVersion == 6)
    return 5;
  else if(modelVersion == 5)
    return 4;
  else if(modelVersion == 3 || modelVersion == 4)
    return 3;

  fail(modelVersion);
  return -1;
}

int NNModelVersion::getNumSpatialFeatures(int modelVersion) {
  if(modelVersion >= 8 && modelVersion <= 16)
    return NNInputs::NUM_FEATURES_SPATIAL_V7;
  else if(modelVersion == 7)
    return NNInputs::NUM_FEATURES_SPATIAL_V6;
  else if(modelVersion == 6)
    return NNInputs::NUM_FEATURES_SPATIAL_V5;
  else if(modelVersion == 5)
    return NNInputs::NUM_FEATURES_SPATIAL_V4;
  else if(modelVersion == 3 || modelVersion == 4)
    return NNInputs::NUM_FEATURES_SPATIAL_V3;

  fail(modelVersion);
  return -1;
}

int NNModelVersion::getNumGlobalFeatures(int modelVersion) {
  if(modelVersion >= 8 && modelVersion <= 16)
    return NNInputs::NUM_FEATURES_GLOBAL_V7;
  else if(modelVersion == 7)
    return NNInputs::NUM_FEATURES_GLOBAL_V6;
  else if(modelVersion == 6)
    return NNInputs::NUM_FEATURES_GLOBAL_V5;
  else if(modelVersion == 5)
    return NNInputs::NUM_FEATURES_GLOBAL_V4;
  else if(modelVersion == 3 || modelVersion == 4)
    return NNInputs::NUM_FEATURES_GLOBAL_V3;

  fail(modelVersion);
  return -1;
}

int NNModelVersion::getNumInputMetaChannels(int metaEncoderVersion) {
  if(metaEncoderVersion == 0)
    return 0;
  if(metaEncoderVersion == 1)
    return 192;
  throw StringError("NNModelVersion: metaEncoderVersion not currently implemented or supported: " + Global::intToString(metaEncoderVersion));
}
