#include "../neuralnet/nninputs.h"
#include "../neuralnet/modelversion.h"

//Old model versions, no longer supported:
//0 = V1 features, with old head architecture using crelus (no longer supported)
//1 = V1 features, with new head architecture, no crelus
//2 = V2 features, no internal architecture change.

//Supported model versions:
//3 = V3 features, many architecture changes for new selfplay loop
//4 = V3 features, scorebelief head
//5 = V4 features, changed current territory feature to just indicate pass-alive
//6 = V5 features, disable fancy features

static void fail(int modelVersion) {
  throw StringError("NNModelVersion: Model version not currently implemented or supported: " + Global::intToString(modelVersion));
}

static_assert(NNModelVersion::latestModelVersionImplemented == 6, "");
static_assert(NNModelVersion::latestInputsVersionImplemented == 5, "");

int NNModelVersion::getInputsVersion(int modelVersion) {
  if(modelVersion == 3 || modelVersion == 4)
    return 3;
  else if(modelVersion == 5)
    return 4;
  else if(modelVersion == 6)
    return 5;

  fail(modelVersion);
  return -1;
}

int NNModelVersion::getNumSpatialFeatures(int modelVersion) {
  if(modelVersion == 3 || modelVersion == 4)
    return NNInputs::NUM_FEATURES_SPATIAL_V3;
  else if(modelVersion == 5)
    return NNInputs::NUM_FEATURES_SPATIAL_V4;
  else if(modelVersion == 6)
    return NNInputs::NUM_FEATURES_SPATIAL_V5;

  fail(modelVersion);
  return -1;
}

int NNModelVersion::getNumGlobalFeatures(int modelVersion) {
  if(modelVersion == 3 || modelVersion == 4)
    return NNInputs::NUM_FEATURES_GLOBAL_V3;
  else if(modelVersion == 5)
    return NNInputs::NUM_FEATURES_GLOBAL_V4;
  else if(modelVersion == 6)
    return NNInputs::NUM_FEATURES_GLOBAL_V5;

  fail(modelVersion);
  return -1;
}
