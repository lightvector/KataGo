#pragma once

#ifdef USE_COREML_BACKEND

#include <string>
#include "desc.h"
#include "../core/commontypes.h"
#include "../neuralnet/modelversion.h"
#include "../neuralnet/nneval.h"
#include "../neuralnet/nninputs.h"
#include "../neuralnet/nninterface.h"
#include <KataGoCoreML/KataGoCoreML-swift.h>

using namespace std;
using namespace KataGoCoreML;

namespace CoreMLProcess {

void copyRowData(float* dest, const float* src, size_t numElements);
void convertNCHW(float* rowSpatialInput, int C, int H, int W, bool inputsUseNHWC);
void processRowData(size_t row, ComputeHandle* gpuHandle, InputBuffers* inputBuffers, NNResultBuf** inputBufs);
float policyOptimismCalc(const double policyOptimism, const float p, const float pOpt);
void processOptimism(InputBuffers* inputBuffers, NNOutput* currentOutput, const double policyOptimism, size_t row);

void processPolicy(InputBuffers* inputBuffers,
                   NNOutput* currentOutput,
                   const ComputeHandle* gpuHandle,
                   NNResultBuf* inputBuf,
                   size_t row);

void processValue(const InputBuffers* inputBuffers, NNOutput* currentOutput, const size_t row);

void processOwnership(const InputBuffers* inputBuffers,
                      NNOutput* currentOutput,
                      const ComputeHandle* gpuHandle,
                      const int symmetry,
                      const size_t row);

void
processScoreValues(const InputBuffers* inputBuffers, NNOutput* currentOutput, const int modelVersion, const size_t row);

void processRow(size_t row,
                const ComputeHandle* gpuHandle,
                InputBuffers* inputBuffers,
                NNResultBuf** inputBufs,
                vector<NNOutput*>& outputs);

void getCoreMLOutput(ComputeHandle* gpuHandle,
                     InputBuffers* inputBuffers,
                     int numBatchEltsFilled,
                     NNResultBuf** inputBufs,
                     vector<NNOutput*>& outputs);
};

/**
 * @brief Represents a loaded neural network model.
 * A LoadedModel object contains a ModelDesc object that describes the characteristics of the loaded model.
 * For Core ML backend, we also store the model path for on-demand conversion.
 */
struct LoadedModel {
  /**
   * @brief The description of the loaded model.
   */
  ModelDesc modelDesc;

  /**
   * @brief Path to the original .bin.gz model file for conversion.
   */
  string modelPath;

  /**
   * @brief Construct a new Loaded Model object
   * This constructor loads a machine learning model from a file and sets the modelDesc field.
   * @param fileName The name of the file containing the machine learning model.
   * @param expectedSha256 The expected SHA-256 hash of the model file.
   */
  LoadedModel(const string& fileName, const string& expectedSha256);

  LoadedModel() = delete;
  LoadedModel(const LoadedModel&) = delete;
  LoadedModel& operator=(const LoadedModel&) = delete;
};

/**
 * @brief Context for computing neural network operations using Core ML.
 * Contains global configuration settings for neural network computations.
 */
struct ComputeContext {
  /**
   * @brief Whether to use FP16 mode for computations.
   */
  enabled_t useFP16Mode;

  /**
   * @brief The width of the neural network input.
   */
  int nnXLen;

  /**
   * @brief The height of the neural network input.
   */
  int nnYLen;

  /**
   * @brief Core ML compute context instance from Swift.
   */
  CoreMLComputeContext coremlContext;

  /**
   * @brief Constructs a ComputeContext object.
   * @param nnX The width of the input tensor.
   * @param nnY The height of the input tensor.
   * @param useFP16Mode Whether to use half-precision floating-point (FP16) mode.
   * @param useNHWCMode Whether to use the NHWC format for input tensors.
   */
  ComputeContext(int nnX, int nnY, enabled_t useFP16Mode, enabled_t useNHWCMode);

  ~ComputeContext();
  ComputeContext() = delete;
  ComputeContext(const ComputeContext&) = delete;
  ComputeContext& operator=(const ComputeContext&) = delete;
};

/**
 * @brief A handle for performing neural network computations using Core ML.
 * This struct represents a per-thread handle for computing neural network operations.
 */
struct ComputeHandle {
  /**
   * @brief The x length of the neural network.
   */
  int nnXLen;

  /**
   * @brief The y length of the neural network.
   */
  int nnYLen;

  /**
   * @brief The index of the GPU to use for computation.
   */
  int gpuIndex;

  /**
   * @brief The version of the loaded model.
   */
  int version;

  /**
   * @brief The version of the metadata encoder.
   */
  int metaEncoderVersion;

  /**
   * @brief Whether the input data uses NHWC format.
   */
  bool inputsUseNHWC;

  /**
   * @brief Whether to use 16-bit floating-point precision.
   */
  bool useFP16;

  /**
   * @brief Whether exact neural net length is required (enables mask optimization).
   */
  bool requireExactNNLen;

  /**
   * @brief The hybrid compute handle instance from Swift.
   * This handle dispatches work to both CoreML (CPU+ANE) and MPSGraph (GPU).
   */
  swift::Optional<HybridComputeHandle> hybridHandle;

  /**
   * @brief Construct a new ComputeHandle object.
   * @param context The ComputeContext object to use for computation.
   * @param loadedModel A pointer to the LoadedModel object.
   * @param inputsUseNHWC Whether the input data uses NHWC format.
   * @param gpuIdx The index of the GPU to use.
   * @param serverThreadIdx The index of the server thread.
   * @param requireExactNNLen Whether exact NN length is required.
   * @param maxBatchSize Maximum batch size for dynamic batch support.
   */
  ComputeHandle(
    ComputeContext* context,
    const LoadedModel* loadedModel,
    bool inputsUseNHWC,
    int gpuIdx,
    int serverThreadIdx,
    bool requireExactNNLen,
    int maxBatchSize);

  ~ComputeHandle();
  ComputeHandle() = delete;
  ComputeHandle(const ComputeHandle&) = delete;
  ComputeHandle& operator=(const ComputeHandle&) = delete;
};

/**
 * @brief Input and output buffers for neural network inference.
 */
struct InputBuffers {
  int maxBatchSize;
  size_t policyResultChannels;

  size_t singleSpatialElts;
  size_t singleInputElts;
  size_t singleInputGlobalElts;
  size_t singleInputMetaElts;
  size_t singlePolicyResultElts;
  size_t singlePolicyPassResultElts;
  size_t singlePolicyProbsElts;
  size_t singleValueResultElts;
  size_t singleOwnershipResultElts;
  size_t singleOwnerMapElts;
  size_t singleScoreValuesResultElts;
  size_t singleMaskElts;

  size_t rowSpatialBufferElts;
  size_t userInputBufferElts;
  size_t userInputGlobalBufferElts;
  size_t userInputMetaBufferElts;
  size_t policyResultBufferElts;
  size_t policyPassResultBufferElts;
  size_t policyProbsBufferElts;
  size_t valueResultBufferElts;
  size_t ownershipResultBufferElts;
  size_t ownerMapBufferElts;
  size_t scoreValuesResultBufferElts;
  size_t userInputMaskBufferElts;

  float* rowSpatialBuffer;
  float* userInputBuffer;
  float* userInputGlobalBuffer;
  float* userInputMetaBuffer;
  float* policyResults;
  float* policyPassResults;
  float* policyProbsBuffer;
  float* valueResults;
  float* ownershipResults;
  float* ownerMapBuffer;
  float* scoreValuesResults;
  float* userInputMaskBuffer;

  InputBuffers(const LoadedModel* loadedModel, int maxBatchSz, int nnXLen, int nnYLen);
  ~InputBuffers();
  InputBuffers() = delete;
  InputBuffers(const InputBuffers&) = delete;
  InputBuffers& operator=(const InputBuffers&) = delete;
};

#endif // USE_COREML_BACKEND
