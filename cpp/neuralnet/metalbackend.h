#pragma once

#include <string>
#include "desc.h"
#include "../core/commontypes.h"

using namespace std;

class MetalDevices {
public:
  MetalDevices();
  ~MetalDevices();
  void printDevices();
};

void createMetalContext(int nnXLen,
                        int nnYLen,
                        enabled_t inputUseFP16Mode,
                        enabled_t inputUseNHWCMode);

void createMetalHandle(int gpuIdxForThisThread,
                       const ModelDesc* desc,
                       int batchSize,
                       int serverThreadIdx);

void getMetalHandleOutput(
  float* userInputBuffer,
  float* userInputGlobalBuffer,
  float* policyOutput,
  float* valueOutput,
  float* ownershipOutput,
  float* miscValuesOutput,
  float* moreMiscValuesOutput,
  int gpuIndex);

void testMetalEvaluateConv(const ConvLayerDesc* desc,
                           int nnXLen,
                           int nnYLen,
                           int batchSize,
                           bool useFP16,
                           bool useNHWC,
                           float* input,
                           float* output);

void testMetalEvaluateBatchNorm(const BatchNormLayerDesc* desc,
                                int nnXLen,
                                int nnYLen,
                                int batchSize,
                                bool useFP16,
                                bool useNHWC,
                                float* input,
                                float* mask,
                                float* output);

void testMetalEvaluateResidualBlock(const ResidualBlockDesc* desc,
                                    int batchSize,
                                    int nnXLen,
                                    int nnYLen,
                                    bool useFP16,
                                    bool useNHWC,
                                    float* input,
                                    float* mask,
                                    float* output);

void testMetalEvaluateGlobalPoolingResidualBlock(const GlobalPoolingResidualBlockDesc* desc,
                                                 int batchSize,
                                                 int nnXLen,
                                                 int nnYLen,
                                                 bool useFP16,
                                                 bool useNHWC,
                                                 float* input,
                                                 float* mask,
                                                 float* output);
