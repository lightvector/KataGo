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

void createMetalHandle(int gpuIdx,
                       int nnXLen,
                       int nnYLen,
                       int version,
                       int numInputChannels,
                       int numInputGlobalChannels,
                       int numValueChannels,
                       int numScoreValueChannels,
                       int numOwnershipChannels);

void getMetalHandleOutput(
  float* userInputBuffer,
  float* userInputGlobalBuffer,
  float* policyOutput,
  float* valueOutput,
  float* ownershipOutput,
  float* miscValuesOutput,
  float* moreMiscValuesOutput,
  int gpuIndex);

void testMetalEvaluateConv(int convXSize,
                           int convYSize,
                           int inChannels,
                           int outChannels,
                           int dilationX,
                           int dilationY,
                           int nnXLen,
                           int nnYLen,
                           int batchSize,
                           bool useFP16,
                           bool useNHWC,
                           float* weights,
                           float* input,
                           float* output);
