#include "../neuralnet/modelversion.h"

#include "../core/global.h"
#include "../neuralnet/nninputs.h"

// Model versions:
// 0 = V1 features, with old head architecture using crelus (no longer
// supported) 1 = V1 features, with new head architecture, no crelus 2 = V2
// features, no internal architecture change. 3 = V3 features, many architecture
// changes for new selfplay loop 4 = V3 features, scorebelief head 5 = V4
// features, changed current territory feature to just indicate pass-alive 6 = V5
// features, disable fancy features

static void fail(int modelVersion) {
  throw StringError(
      "NNModelVersion: Model version not currently implemented or supported: " +
      Global::intToString(modelVersion));
}

const int NNModelVersion::latestModelVersionImplemented = 6;
const int NNModelVersion::defaultModelVersion = 5;

int NNModelVersion::getInputsVersion(int modelVersion) {
  if (modelVersion == 0 || modelVersion == 1)
    return 1;
  else if (modelVersion == 2)
    return 2;
  else if (modelVersion == 3 || modelVersion == 4)
    return 3;
  else if (modelVersion == 5)
    return 4;
  else if (modelVersion == 6)
    return 5;

  fail(modelVersion);
  return -1;
}

int NNModelVersion::getNumSpatialFeatures(int modelVersion) {
  if (modelVersion == 0 || modelVersion == 1)
    return NNInputs::NUM_FEATURES_V1;
  else if (modelVersion == 2)
    return NNInputs::NUM_FEATURES_V2;
  else if (modelVersion == 3 || modelVersion == 4)
    return NNInputs::NUM_FEATURES_BIN_V3;
  else if (modelVersion == 5)
    return NNInputs::NUM_FEATURES_BIN_V4;
  else if (modelVersion == 6)
    return NNInputs::NUM_FEATURES_BIN_V5;

  fail(modelVersion);
  return -1;
}

int NNModelVersion::getNumGlobalFeatures(int modelVersion) {
  if (modelVersion == 0 || modelVersion == 1 || modelVersion == 2)
    return 0;
  else if (modelVersion == 3 || modelVersion == 4)
    return NNInputs::NUM_FEATURES_GLOBAL_V3;
  else if (modelVersion == 5)
    return NNInputs::NUM_FEATURES_GLOBAL_V4;
  else if (modelVersion == 6)
    return NNInputs::NUM_FEATURES_GLOBAL_V5;

  fail(modelVersion);
  return -1;
}

int NNModelVersion::getRowSize(int modelVersion) {
  if (modelVersion == 0 || modelVersion == 1)
    return NNInputs::ROW_SIZE_V1;
  else if (modelVersion == 2)
    return NNInputs::ROW_SIZE_V2;
  else if (modelVersion >= 3)
    throw StringError(
        "NNModelVersion::getRowSize - row size not meaningful for version >= "
        "3");

  fail(modelVersion);
  return -1;
}
