#include "../neuralnet/nninterface.h"

//Model versions:
//0 = V1 features, with old head architecture using crelus (no longer supported)
//1 = V1 features, with new head architecture, no crelus
//2 = V2 features, no internal architecture change.

static void fail(int modelVersion) {
  throw StringError("NNModelVersion: Model version not currently implemented or supported: " + Global::intToString(modelVersion));
}

const int NNModelVersion::latestModelVersionImplemented = 2;

int NNModelVersion::getInputsVersion(int modelVersion) {
  if(modelVersion == 0 || modelVersion == 1)
    return 1;
  else if(modelVersion == 2)
    return 2;

  fail(modelVersion);
  return -1;
}

int NNModelVersion::getNumFeatures(int modelVersion) {
  if(modelVersion == 0 || modelVersion == 1)
    return NNInputs::NUM_FEATURES_V1;
  else if(modelVersion == 2)
    return NNInputs::NUM_FEATURES_V2;

  fail(modelVersion);
  return -1;
}
int NNModelVersion::getRowSize(int modelVersion) {
  if(modelVersion == 0 || modelVersion == 1)
    return NNInputs::ROW_SIZE_V1;
  else if(modelVersion == 2)
    return NNInputs::ROW_SIZE_V2;

  fail(modelVersion);
  return -1;
}



