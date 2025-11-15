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

static void fail(const int modelVersion, const bool dotsGame) {
  throw StringError("NNModelVersion: Model version not currently implemented or supported: " + Global::intToString(modelVersion) +
    (dotsGame ? " (Dots game)" : ""));
}

static_assert(NNModelVersion::oldestModelVersionImplemented == 3);
static_assert(NNModelVersion::oldestInputsVersionImplemented == 3);
static_assert(NNModelVersion::latestModelVersionImplemented == 16);
static_assert(NNModelVersion::latestInputsVersionImplemented == 7);

int NNModelVersion::getInputsVersion(const int modelVersion, const bool dotsGame) {
  switch(modelVersion) {
    case 3:
    case 4:
      if (!dotsGame) {
        return 3;
      }
      break;
    case 5:
      if (!dotsGame) {
        return 4;
      }
      break;
    case 6:
      if (!dotsGame) {
        return 5;
      }
      break;
    case 7:
      if (!dotsGame) {
        return 6;
      }
      break;
    default:
      if (modelVersion <= latestModelVersionImplemented) {
        return 7;
      }
      break;
  }

  fail(modelVersion, dotsGame);
  return -1;
}

int NNModelVersion::getNumSpatialFeatures(const int modelVersion, const bool dotsGame) {
  switch(modelVersion) {
    case 3:
    case 4:
      if (!dotsGame) {
        return NNInputs::NUM_FEATURES_SPATIAL_V3;
      }
      break;
    case 5:
      if (!dotsGame) {
        return NNInputs::NUM_FEATURES_SPATIAL_V4;
      }
      break;
    case 6:
      if (!dotsGame) {
        return NNInputs::NUM_FEATURES_SPATIAL_V5;
      }
      break;
    case 7:
      if (!dotsGame) {
        return NNInputs::NUM_FEATURES_SPATIAL_V6;
      }
      break;
    default:
      if (modelVersion <= latestModelVersionImplemented) {
        return dotsGame ? NNInputs::NUM_FEATURES_SPATIAL_V7_DOTS : NNInputs::NUM_FEATURES_SPATIAL_V7;
      }
      break;
  }

  fail(modelVersion, dotsGame);
  return -1;
}

int NNModelVersion::getNumGlobalFeatures(const int modelVersion, const bool dotsGame) {
  switch(modelVersion) {
    case 3:
    case 4:
      if (!dotsGame) {
        return NNInputs::NUM_FEATURES_GLOBAL_V3;
      }
      break;
    case 5:
      if (!dotsGame) {
        return NNInputs::NUM_FEATURES_GLOBAL_V4;
      }
      break;
    case 6:
      if (!dotsGame) {
        return NNInputs::NUM_FEATURES_GLOBAL_V5;
      }
      break;
    case 7:
      if (!dotsGame) {
        return NNInputs::NUM_FEATURES_GLOBAL_V6;
      }
      break;
    default:
      if (modelVersion <= latestModelVersionImplemented) {
        return dotsGame ? NNInputs::NUM_FEATURES_GLOBAL_V7_DOTS : NNInputs::NUM_FEATURES_GLOBAL_V7;
      }
      break;
  }

  fail(modelVersion, dotsGame);
  return -1;
}

int NNModelVersion::getNumInputMetaChannels(int metaEncoderVersion) {
  if(metaEncoderVersion == 0)
    return 0;
  if(metaEncoderVersion == 1)
    return 192;
  throw StringError("NNModelVersion: metaEncoderVersion not currently implemented or supported: " + Global::intToString(metaEncoderVersion));
}
