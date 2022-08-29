#pragma once

#include <string>

using namespace std;

class MetalDevices {
public:
  MetalDevices();
  ~MetalDevices();
  void printDevices();
};

class MetalHandle {
public:
  MetalHandle();
  ~MetalHandle();

  void init(int nnXLen,
            int nnYLen,
            int versionIn,
            int numInputChannels,
            int numInputGlobalChannels,
            int numValueChannels,
            int numScoreValueChannels,
            int numOwnershipChannels);

  void* placeholderWithShape(int nnXLen,
                             int nnYLen,
                             int numInputChannels,
                             int numInputGlobalChannels,
                             string name);

  void apply(float* userInputBuffer,
             float* userInputGlobalBuffer,
             float* policyOutput,
             float* valueOutput,
             float* ownershipOutput,
             float* miscValuesOutput,
             float* moreMiscValuesOutput);

  int getVersion() { return version; }

private:
  int version;
  void* kataGoGraph;
};
