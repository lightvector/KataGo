#pragma once

#include <string>
#include "desc.h"

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
            const ModelDesc* modelDesc);

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
