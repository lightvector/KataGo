#pragma once

#include <string>
#include "desc.h"
#include "../core/commontypes.h"
#include "../neuralnet/modelversion.h"
#include "../neuralnet/nneval.h"
#include "../neuralnet/nninputs.h"
#include "../neuralnet/nninterface.h"

using namespace std;

/**
 * @brief Represents a loaded neural network model.
 * A LoadedModel object contains a ModelDesc object that describes the characteristics of the loaded model.
 * The default constructor, copy constructor, and assignment operator are deleted to prevent
 * creation of an uninitialized LoadedModel object, copying of the loaded model, and potential memory leaks.
 */
struct LoadedModel {
  /**
   * @brief The description of the loaded model.
   * The modelDesc field is a ModelDesc object that describes the characteristics of the loaded model.
   */
  ModelDesc modelDesc;

  /**
   * @brief Construct a new Loaded Model object
   * This constructor loads a machine learning model from a file and sets the modelDesc field to the
   * characteristics of the loaded model.
   * @param fileName The name of the file containing the machine learning model.
   * @param expectedSha256 The expected SHA-256 hash of the model file.
   */
  LoadedModel(const string& fileName, const string& expectedSha256) {
    ModelDesc::loadFromFileMaybeGZipped(fileName, modelDesc, expectedSha256);
  }

  /**
   * @brief Delete the default constructor
   * The default constructor is deleted to prevent creation of an uninitialized LoadedModel object.
   */
  LoadedModel() = delete;

  /**
   * @brief Delete the copy constructor
   * The copy constructor is deleted to prevent copying of the loaded model.
   */
  LoadedModel(const LoadedModel&) = delete;

  /**
   * @brief Delete the assignment operator
   * The assignment operator is deleted to prevent copying of the loaded model.
   */
  LoadedModel& operator=(const LoadedModel&) = delete;
};

/**
 * @brief Context for computing neural network operations.
 * A ComputeContext object contains configuration settings for neural network computations, such as
 * whether to use half-precision floating-point (FP16) mode and whether to use the NHWC format for
 * input tensors. The default constructor, copy constructor, and assignment operator are deleted
 * to prevent creation of an uninitialized ComputeContext object, copying of the object, and potential
 * memory leaks.
 */
struct ComputeContext {
  /**
   * @brief Whether to use FP16 mode for computations.
   */
  enabled_t useFP16Mode;

  /**
   * @brief Constructs a ComputeContext object.
   * This constructor creates a ComputeContext object and sets the configuration settings for neural network
   * computations, including whether to use FP16 mode and whether to use the NHWC format for input tensors.
   * @param nnX The width of the input tensor.
   * @param nnY The height of the input tensor.
   * @param useFP16Mode Whether to use half-precision floating-point (FP16) mode for computations.
   * @param useNHWCMode Whether to use the NHWC format for input tensors.
   */
  ComputeContext(int nnX, int nnY, enabled_t useFP16Mode, enabled_t useNHWCMode);

  /**
   * @brief Destroys the ComputeContext object.
   */
  ~ComputeContext();

  /**
   * @brief Deletes the default constructor.
   */
  ComputeContext() = delete;

  /**
   * @brief Deletes the copy constructor.
   */
  ComputeContext(const ComputeContext&) = delete;

  /**
   * @brief Deletes the copy constructor.
   * 
   * @return ComputeContext& 
   */
  ComputeContext& operator=(const ComputeContext&) = delete;
};

/**
 * @brief A handle for performing neural network computations.
 * This struct represents a handle for computing neural network operations. It contains various
 * parameters and settings that determine how the computation is performed.
 */
struct ComputeHandle {
  /**
   * @brief The x length of the neural network computation context.
   */
  int nnXLen;

  /**
   * @brief The y length of the neural network computation context.
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
   * @brief Whether the input data uses NHWC format.
   */
  bool inputsUseNHWC;

  /**
   * @brief Whether to use 16-bit floating-point precision for computation.
   */
  bool useFP16;

  /**
   * @brief Whether to use Metal for computations (as opposed to CoreML).
   */
  bool useMetal;

  /**
   * @brief The x length of the CoreML model.
   */
  int modelXLen = COMPILE_MAX_BOARD_LEN;

  /**
   * @brief The y length of the CoreML model.
   */
  int modelYLen = COMPILE_MAX_BOARD_LEN;

  /**
   * @brief The version of the CoreML model.
   */
  int modelVersion;

  /**
   * @brief The index of the CoreML model.
   */
  int modelIndex;

  /**
   * @brief Construct a new ComputeHandle object.
   * This constructor initializes a new ComputeHandle object with the specified parameters and settings.
   * @param context The ComputeContext object to use for computation.
   * @param loadedModel A pointer to the LoadedModel object containing the neural network model to use.
   * @param inputsUseNHWC Whether the input data uses NHWC format.
   * @param gpuIdx The index of the GPU to use for computation.
   * @param serverThreadIdx The index of the server thread to use for computation.
   */
  ComputeHandle(
    ComputeContext* context,
    const LoadedModel* loadedModel,
    bool inputsUseNHWC,
    int gpuIdx,
    int serverThreadIdx);

  /**
   * @brief Destroy the ComputeHandle object.
   * This destructor frees any resources that were allocated for the ComputeHandle object.
   */
  ~ComputeHandle();

  /**
   * @brief Delete the default constructor.
   */
  ComputeHandle() = delete;

  /**
   * @brief Delete the copy constructor.
   */
  ComputeHandle(const ComputeHandle&) = delete;

  /**
   * @brief Delete the assignment operator.
   */
  ComputeHandle& operator=(const ComputeHandle&) = delete;
};

struct InputBuffers {
  int maxBatchSize;
  size_t policyResultChannels;

  size_t singleSpatialElts;
  size_t singleInputElts;
  size_t singleInputGlobalElts;
  size_t singleNnPolicyResultElts;
  size_t singleModelPolicyResultElts;
  size_t singlePolicyPassResultElts;
  size_t singlePolicyProbsElts;
  size_t singleValueResultElts;
  size_t singleNnOwnershipResultElts;
  size_t singleModelOwnershipResultElts;
  size_t singleOwnerMapElts;
  size_t singleScoreValuesResultElts;
  size_t singleMoreMiscValuesResultElts;

  size_t rowSpatialBufferElts;
  size_t userInputBufferElts;
  size_t userInputGlobalBufferElts;
  size_t policyResultBufferElts;
  size_t policyPassResultBufferElts;
  size_t policyProbsBufferElts;
  size_t valueResultBufferElts;
  size_t ownershipResultBufferElts;
  size_t ownerMapBufferElts;
  size_t scoreValuesResultBufferElts;
  size_t moreMiscValuesResultsBufferElts;

  float* rowSpatialBuffer;
  float* userInputBuffer;
  float* userInputGlobalBuffer;
  float* policyResults;
  float* policyPassResults;
  float* policyProbsBuffer;
  float* valueResults;
  float* ownershipResults;
  float* ownerMapBuffer;
  float* scoreValuesResults;
  float* moreMiscValuesResults;

  InputBuffers(const LoadedModel* loadedModel, int maxBatchSz, int nnXLen, int nnYLen);
  ~InputBuffers();
  InputBuffers() = delete;
  InputBuffers(const InputBuffers&) = delete;
  InputBuffers& operator=(const InputBuffers&) = delete;
};

/// Print the available Metal devices.
void printMetalDevices(void);

/// Create a Metal computing context.
/// - Parameters:
///   - nnXLen: The length of the neural network input in the x dimension.
///   - nnYLen: The length of the neural network input in the y dimension.
///   - inputUseFP16Mode: Whether to use 16-bit floating-point precision or not.
///   - inputUseNHWCMode: Whether to use NHWC mode or not.
void createMetalContext(int nnXLen,
                        int nnYLen,
                        enabled_t inputUseFP16Mode,
                        enabled_t inputUseNHWCMode);

/// Destroy a Metal computing context.
void destroyMetalContext(void);

/// Get the length of the neural network input in the x dimension from Metal computing context
int getMetalContextXLen(void);

/// Get the length of the neural network input in the y dimension from Metal computing context
int getMetalContextYLen(void);

/// Create a Metal computing handle.
/// - Parameters:
///   - gpuIdxForThisThread: A GPU index for this thread.
///   - desc: A model description.
///   - serverThreadIdx: A server thread index.
void createMetalHandle(int gpuIdxForThisThread,
                       const ModelDesc* desc,
                       int serverThreadIdx);

/// Get output from a Metal computing handle.
/// - Parameters:
///   - userInputBuffer: A user input buffer.
///   - userInputGlobalBuffer: A user input global buffer.
///   - policyOutput: A policy output buffer.
///   - policyPassOutput: A policy pass output buffer.
///   - valueOutput: A value output buffer.
///   - ownershipOutput: An ownership output buffer.
///   - scoreValueOutput: A score value output buffer.
///   - gpuIdx: A GPU index.
///   - batchSize: A batch size.
void getMetalHandleOutput(float* userInputBuffer,
                          float* userInputGlobalBuffer,
                          float* policyOutput,
                          float* policyPassOutput,
                          float* valueOutput,
                          float* ownershipOutput,
                          float* scoreValueOutput,
                          int gpuIdx,
                          int batchSize);

/// Test Metal evaluating convolution layer with a given input
/// - Parameters:
///   - desc: A convolution layer description.
///   - nnXLen: A neural network input length in the x dimension.
///   - nnYLen: A neural network input length in the y dimension.
///   - batchSize: A batch size.
///   - input: An input buffer.
///   - output: An output buffer.
void testMetalEvaluateConv(const ConvLayerDesc* desc,
                           int nnXLen,
                           int nnYLen,
                           int batchSize,
                           float* input,
                           float* output);

/// Test Metal evaluating batch normalization layer with a given input
/// - Parameters:
///   - desc: A batch normalization layer description.
///   - nnXLen: A neural network input length in the x dimension.
///   - nnYLen: A neural network input length in the y dimension.
///   - batchSize: A batch size.
///   - input: an input buffer.
///   - mask: a mask buffer.
///   - output: an output buffer.
void testMetalEvaluateBatchNorm(const BatchNormLayerDesc* desc,
                                int nnXLen,
                                int nnYLen,
                                int batchSize,
                                float* input,
                                float* mask,
                                float* output);

/// Test Metal evaluating residual block with a given input
/// - Parameters:
///   - desc: a residual block description.
///   - batchSize: a batch size.
///   - nnXLen: a neural network input length in the x dimension.
///   - nnYLen: a neural network input length in the y dimension.
///   - input: An input buffer.
///   - mask: A mask buffer.
///   - output: An output buffer.
void testMetalEvaluateResidualBlock(const ResidualBlockDesc* desc,
                                    int batchSize,
                                    int nnXLen,
                                    int nnYLen,
                                    float* input,
                                    float* mask,
                                    float* output);

/// Test Metal evaluating global pooling residual block with a given input
/// - Parameters:
///   - desc: A global pooling residual block description.
///   - batchSize: A batch size.
///   - nnXLen: A neural network input length in the x dimension.
///   - nnYLen: A neural network input length in the y dimension.
///   - input: An input buffer.
///   - mask: A mask buffer.
///   - output: An output buffer.
void testMetalEvaluateGlobalPoolingResidualBlock(const GlobalPoolingResidualBlockDesc* desc,
                                                 int batchSize,
                                                 int nnXLen,
                                                 int nnYLen,
                                                 float* input,
                                                 float* mask,
                                                 float* output);
