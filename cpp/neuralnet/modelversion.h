#ifndef NEURALNET_MODELVERSION_H_
#define NEURALNET_MODELVERSION_H_

// Model versions
namespace NNModelVersion {

  extern const int latestModelVersionImplemented;
  extern const int defaultModelVersion;

  // Which V* feature version from NNInputs does a given model version consume?
  int getInputsVersion(int modelVersion);

  // Convenience functions, feeds forward the number of features and the size of
  // the row vector that the net takes as input
  int getNumSpatialFeatures(int modelVersion);
  int getNumGlobalFeatures(int modelVersion);
  int getRowSize(int modelVersion);

}  // namespace NNModelVersion

#endif  // NEURALNET_MODELVERSION_H_
