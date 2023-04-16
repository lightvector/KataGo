#ifdef USE_COREML_BACKEND

#include "../neuralnet/modelversion.h"
#include "../neuralnet/nneval.h"
#include "../neuralnet/nninputs.h"
#include "../neuralnet/nninterface.h"
#include "../neuralnet/metalbackend.h"
#include "../neuralnet/coremlbackend.h"

using namespace std;

//---------------------------------------------------------------------------------------------------------

/**
 * @brief This function initializes the global state of the NeuralNet class upon program startup.
 * This function should be called only once upon program startup. It ensures that the global state
 * of the NeuralNet class is properly initialized, enabling it to function correctly throughout
 * the lifetime of the program.
 * Note that this function does not take any input parameters or return any values.
 */
void NeuralNet::globalInitialize() {
  // Do nothing.
}

/**
 * @brief This function cleans up the global state of the NeuralNet class at program termination.
 * This function should be called once at program termination. It ensures that the global state of
 * the NeuralNet class is properly cleaned up, freeing any resources that were allocated during the
 * lifetime of the program.
 * Note that this function does not take any input parameters or return any values.
 */
void NeuralNet::globalCleanup() {
  // Do nothing.
}

/**
 * @brief Loads a neural network model from a file.
 * This function creates a LoadedModel object by loading a neural network model from a file specified by
 * the `file` parameter and expected SHA-256 hash specified by the `expectedSha256` parameter. The LoadedModel
 * object is returned as a pointer.
 * @param file The name of the file containing the neural network model.
 * @param expectedSha256 The expected SHA-256 hash of the model file.
 * @return A pointer to the LoadedModel object created by loading the model file.
 */
LoadedModel* NeuralNet::loadModelFile(const string& file, const string& expectedSha256) {
  LoadedModel* loadedModel = new LoadedModel(file, expectedSha256);
  return loadedModel;
}

/**
 * @brief Frees memory used by a LoadedModel object.
 * This function deallocates memory used by a LoadedModel object specified by the `loadedModel` parameter.
 * @param loadedModel A pointer to the LoadedModel object to deallocate memory for.
 */
void NeuralNet::freeLoadedModel(LoadedModel* loadedModel) {
  delete loadedModel;
}

/**
 * @brief Gets the name of the loaded model.
 * This function returns the name of the loaded model contained in the LoadedModel object specified
 * by the `loadedModel` parameter.
 * @param loadedModel A pointer to the LoadedModel object to get the model name from.
 * @return The name of the loaded model.
 */
string NeuralNet::getModelName(const LoadedModel* loadedModel) {
  return loadedModel->modelDesc.name;
}

/**
 * @brief Gets the version of the loaded model.
 * This function returns the version of the loaded model contained in the LoadedModel object specified
 * by the `loadedModel` parameter.
 * @param loadedModel A pointer to the LoadedModel object to get the model version from.
 * @return The version of the loaded model.
 */
int NeuralNet::getModelVersion(const LoadedModel* loadedModel) {
  return loadedModel->modelDesc.version;
}

/**
 * @brief Gets the rules supported by the loaded model.
 * This function returns a Rules object that describes the rules supported by the loaded model contained
 * in the LoadedModel object specified by the `loadedModel` parameter. The desired rules are specified by
 * the `desiredRules` parameter. The `supported` output parameter is set to true if the desired rules are
 * supported by the loaded model, and false otherwise.
 * @param loadedModel A pointer to the LoadedModel object to get the supported rules from.
 * @param desiredRules The desired rules to check support for.
 * @param supported Set to true if the desired rules are supported by the loaded model, false otherwise.
 * @return A Rules object that describes the rules supported by the loaded model.
 */
Rules NeuralNet::getSupportedRules(const LoadedModel* loadedModel, const Rules& desiredRules, bool& supported) {
  return loadedModel->modelDesc.getSupportedRules(desiredRules, supported);
}

//------------------------------------------------------------------------------

ComputeContext::ComputeContext(int nnX, int nnY, enabled_t useFP16Mode, enabled_t useNHWCMode) {
  this->useFP16Mode = useFP16Mode;
  createMetalContext(nnX, nnY, useFP16Mode, useNHWCMode);
  createCoreMLContext();
}

ComputeContext::~ComputeContext() {
  destroyMetalContext();
  destroyCoreMLContext();
}

/**
 * @brief Creates a ComputeContext object for computing neural network operations.
 * This function creates a ComputeContext object by setting configuration settings for neural network computations,
 * such as whether to use half-precision floating-point (FP16) mode and whether to use the NHWC format for input
 * tensors. The ComputeContext object is returned as a pointer.
 * @param gpuIdxs (Unused) A vector of GPU indices to use for computations.
 * @param logger (Unused) A pointer to a Logger object to use for logging messages.
 * @param nnXLen The width of the input tensor.
 * @param nnYLen The height of the input tensor.
 * @param openCLTunerFile (Unused) The name of a file containing OpenCL tuning parameters.
 * @param homeDataDirOverride (Unused) A directory to use for storing data.
 * @param openCLReTunePerBoardSize (Unused) Whether to re-tune OpenCL parameters for different board sizes.
 * @param useFP16Mode Whether to use half-precision floating-point (FP16) mode for computations.
 * @param useNHWCMode Whether to use the NHWC format for input tensors.
 * @param loadedModel (Unused) A pointer to a LoadedModel object containing a loaded neural network model.
 * @return A pointer to the ComputeContext object created.
 */
ComputeContext* NeuralNet::createComputeContext(
  const vector<int>& gpuIdxs,
  Logger* logger,
  int nnXLen,
  int nnYLen,
  const string& openCLTunerFile,
  const string& homeDataDirOverride,
  bool openCLReTunePerBoardSize,
  enabled_t useFP16Mode,
  enabled_t useNHWCMode,
  const LoadedModel* loadedModel) {

  (void)gpuIdxs;
  (void)logger;
  (void)openCLTunerFile;
  (void)homeDataDirOverride;
  (void)openCLReTunePerBoardSize;
  (void)loadedModel;

  return new ComputeContext(nnXLen, nnYLen, useFP16Mode, useNHWCMode);
}

/**
 * @brief Frees memory used by a ComputeContext object.
 * This function deallocates memory used by a ComputeContext object specified by the `computeContext` parameter.
 * @param computeContext A pointer to the ComputeContext object to deallocate memory for.
 */
void NeuralNet::freeComputeContext(ComputeContext* computeContext) {
  delete computeContext;
}

//--------------------------------------------------------------

ComputeHandle::ComputeHandle(
  ComputeContext* context,
  const LoadedModel* loadedModel,
  bool inputsUseNHWC,
  int gpuIdx,
  int serverThreadIdx) {
  const ModelDesc* modelDesc = &loadedModel->modelDesc;
  int coreMLStartIndex = 100;

  nnXLen = getMetalContextXLen();
  nnYLen = getMetalContextYLen();
  gpuIndex = gpuIdx;
  version = modelDesc->version;
  this->inputsUseNHWC = inputsUseNHWC;

  /* Use FP16 mode if the model supports it and the user has not explicitly
   * disabled it. */
  useFP16 = (context->useFP16Mode != enabled_t::False);
  useMetal = (gpuIdx < coreMLStartIndex);

  if(useMetal) {
    createMetalHandle(gpuIdx, modelDesc, serverThreadIdx);
  } else {
    // Create a Core ML backend
    modelIndex = createCoreMLBackend(modelXLen, modelYLen, serverThreadIdx, useFP16);
    // Get the model version
    modelVersion = getCoreMLBackendVersion(modelIndex);
  }
}

ComputeHandle::~ComputeHandle() {
  if(!useMetal) {
    // Free the CoreML backend
    freeCoreMLBackend(modelIndex);
  }
}

/**
 * @brief Create a new ComputeHandle object for performing neural network computations.
 * This function creates a new ComputeHandle object for performing neural network computations,
 * using the specified parameters and settings. The object is allocated on the heap using the
 * 'new' operator and returned as a pointer.
 * @param context A pointer to the ComputeContext object to use for computation.
 * @param loadedModel A pointer to the LoadedModel object containing the neural network model to use.
 * @param logger A pointer to the Logger object to use for logging messages.
 * @param maxBatchSize The maximum batch size to use for computation.
 * @param requireExactNNLen Whether the neural network length must match the input data length exactly.
 * @param inputsUseNHWC Whether the input data uses NHWC format.
 * @param gpuIdxForThisThread The index of the GPU to use for computation.
 * @param serverThreadIdx The index of the server thread to use for computation.
 * @return A pointer to the newly-created ComputeHandle object.
 */
ComputeHandle* NeuralNet::createComputeHandle(
  ComputeContext* context,
  const LoadedModel* loadedModel,
  Logger* logger,
  int maxBatchSize,
  bool requireExactNNLen,
  bool inputsUseNHWC,
  int gpuIdxForThisThread,
  int serverThreadIdx) {

  (void)maxBatchSize;
  // Current implementation always tolerates excess nn len
  (void)requireExactNNLen;
  ComputeHandle* handle = new ComputeHandle(context, loadedModel, inputsUseNHWC, gpuIdxForThisThread, serverThreadIdx);

  return handle;
}

/**
 * @brief Free the memory used by a ComputeHandle object.
 * This function frees the memory used by the specified ComputeHandle object, which was
 * previously allocated on the heap using the 'new' operator.
 * @param handle A pointer to the ComputeHandle object to free.
 */
void NeuralNet::freeComputeHandle(ComputeHandle* handle) {
  delete handle;
}

/**
 * @brief Check whether a ComputeHandle object is using 16-bit floating-point precision.
 * This function checks whether the specified ComputeHandle object is using 16-bit floating-point
 * precision for computation, and returns a boolean value indicating the result.
 * @param handle A pointer to the ComputeHandle object to check.
 * @return True if the ComputeHandle object is using 16-bit floating-point precision, false otherwise.
 */
bool NeuralNet::isUsingFP16(const ComputeHandle* handle) {
  return handle->useFP16;
}

//------------------------------------------------------------------------------

/**
 * @brief Print information about the available devices.
 */
void NeuralNet::printDevices() {
  printMetalDevices();
}

//--------------------------------------------------------------

/**
 * @brief Construct a new InputBuffers object for storing input data for neural network computation.
 * This constructor initializes a new InputBuffers object for storing input data for neural network
 * computation, based on the specified parameters and settings.
 * @param loadedModel A pointer to the LoadedModel object containing the neural network model to use.
 * @param maxBatchSz The maximum batch size to use for computation.
 * @param nnXLen The x length of the neural network computation context.
 * @param nnYLen The y length of the neural network computation context.
 */
InputBuffers::InputBuffers(const LoadedModel* loadedModel, int maxBatchSz, int nnXLen, int nnYLen) {
  const ModelDesc& m = loadedModel->modelDesc;

  int modelXLen = COMPILE_MAX_BOARD_LEN;
  int modelYLen = COMPILE_MAX_BOARD_LEN;

  maxBatchSize = maxBatchSz;
  policyResultChannels = 1;
  singleSpatialElts = (size_t)m.numInputChannels * nnXLen * nnYLen;
  singleInputElts = (size_t)m.numInputChannels * modelXLen * modelYLen;
  singleInputGlobalElts = (size_t)m.numInputGlobalChannels;
  singleNnPolicyResultElts = (size_t)(nnXLen * nnYLen);
  singleModelPolicyResultElts = (size_t)((modelXLen * modelYLen) + 1);
  singlePolicyPassResultElts = 1;
  singlePolicyProbsElts = (size_t)((nnXLen * nnYLen) + 1);
  singleValueResultElts = (size_t)m.numValueChannels;
  singleNnOwnershipResultElts = (size_t)m.numOwnershipChannels * nnXLen * nnYLen;
  singleModelOwnershipResultElts = (size_t)m.numOwnershipChannels * modelXLen * modelYLen;
  singleOwnerMapElts = (size_t)m.numOwnershipChannels * nnXLen * nnYLen;
  singleScoreValuesResultElts = 10;
  singleMoreMiscValuesResultElts = 8;

  assert(NNModelVersion::getNumSpatialFeatures(m.version) == m.numInputChannels);
  assert(NNModelVersion::getNumGlobalFeatures(m.version) == m.numInputGlobalChannels);
  assert(singleValueResultElts == 3);

  rowSpatialBufferElts = (size_t)maxBatchSz * singleSpatialElts;
  userInputBufferElts = (size_t)maxBatchSize * singleInputElts;
  userInputGlobalBufferElts = (size_t)maxBatchSize * singleInputGlobalElts;
  policyResultBufferElts = (size_t)maxBatchSize * singleModelPolicyResultElts * policyResultChannels;
  policyPassResultBufferElts = (size_t)maxBatchSize * singlePolicyPassResultElts;
  policyProbsBufferElts = (size_t)maxBatchSize * singlePolicyProbsElts;
  valueResultBufferElts = (size_t)maxBatchSize * singleValueResultElts;
  ownershipResultBufferElts = (size_t)maxBatchSize * singleModelOwnershipResultElts;
  ownerMapBufferElts = (size_t)maxBatchSz * singleOwnerMapElts;
  scoreValuesResultBufferElts = (size_t)maxBatchSize * singleScoreValuesResultElts;
  moreMiscValuesResultsBufferElts = (size_t)maxBatchSz * singleMoreMiscValuesResultElts;

  rowSpatialBuffer = new float[rowSpatialBufferElts];
  userInputBuffer = new float[userInputBufferElts];
  // Zero out the input buffer for arbitrary board sizes
  memset(&userInputBuffer[0], 0, userInputBufferElts * sizeof(userInputBuffer[0]));

  userInputGlobalBuffer = new float[userInputGlobalBufferElts];
  policyResults = new float[policyResultBufferElts];
  policyPassResults = new float[policyPassResultBufferElts];
  policyProbsBuffer = new float[policyProbsBufferElts];
  valueResults = new float[valueResultBufferElts];
  ownershipResults = new float[ownershipResultBufferElts];
  ownerMapBuffer = new float[ownerMapBufferElts];
  scoreValuesResults = new float[scoreValuesResultBufferElts];
  moreMiscValuesResults = new float[moreMiscValuesResultsBufferElts];
}

/**
 * @brief Destroy the InputBuffers object and free all associated memory.
 * This destructor destroys the InputBuffers object and frees all memory associated with it,
 * including all input and output buffers used for neural network computation.
 */
InputBuffers::~InputBuffers() {
  delete[] rowSpatialBuffer;
  delete[] userInputBuffer;
  delete[] userInputGlobalBuffer;
  delete[] policyResults;
  delete[] policyPassResults;
  delete[] policyProbsBuffer;
  delete[] valueResults;
  delete[] ownershipResults;
  delete[] ownerMapBuffer;
  delete[] scoreValuesResults;
  delete[] moreMiscValuesResults;
}

/**
 * @brief Create a new InputBuffers object for storing input data for neural network computation.
 * This function creates a new InputBuffers object for storing input data for neural network computation,
 * using the specified parameters and settings. The object is allocated on the heap using the 'new' operator
 * and returned as a pointer.
 * @param loadedModel A pointer to the LoadedModel object containing the neural network model to use.
 * @param maxBatchSize The maximum batch size to use for computation.
 * @param nnXLen The x length of the neural network computation context.
 * @param nnYLen The y length of the neural network computation context.
 * @return A pointer to the newly-created InputBuffers object.
 */
InputBuffers* NeuralNet::createInputBuffers(const LoadedModel* loadedModel, int maxBatchSize, int nnXLen, int nnYLen) {
  return new InputBuffers(loadedModel, maxBatchSize, nnXLen, nnYLen);
}

/**
 * @brief Free the memory used by an InputBuffers object.
 * This function frees the memory used by the specified InputBuffers object, which was
 * previously allocated on the heap using the 'new' operator.
 * @param inputBuffers A pointer to the InputBuffers object to free.
 */
void NeuralNet::freeInputBuffers(InputBuffers* inputBuffers) {
  delete inputBuffers;
}

//--------------------------------------------------------------

/**
 * @brief Compute the neural network output using Metal API and the specified input data and GPU handle.
 * This function computes the neural network output using the Metal API and the specified input data and ComputeHandle
 * object for GPU acceleration. The computed output is stored in the specified vector of NNOutput pointers.
 * @param gpuHandle A pointer to the ComputeHandle object to use for GPU computation.
 * @param inputBuffers A pointer to the InputBuffers object containing the input data for computation.
 * @param numBatchEltsFilled The number of batch elements filled in the input buffer.
 * @param inputBufs An array of pointers to NNResultBuf objects containing the neural network input data.
 * @param outputs A vector of NNOutput pointers to store the computed output.
 */
static void getMetalOutput(
  ComputeHandle* gpuHandle,
  InputBuffers* inputBuffers,
  int numBatchEltsFilled,
  NNResultBuf** inputBufs,
  vector<NNOutput*>& outputs) {

  int batchSize = numBatchEltsFilled;
  int nnXLen = gpuHandle->nnXLen;
  int nnYLen = gpuHandle->nnYLen;
  int version = gpuHandle->version;
  int numSpatialFeatures = NNModelVersion::getNumSpatialFeatures(version);
  int numGlobalFeatures = NNModelVersion::getNumGlobalFeatures(version);

  assert(batchSize <= inputBuffers->maxBatchSize);
  assert(batchSize > 0);
  assert((numSpatialFeatures * nnXLen * nnYLen) <= inputBuffers->singleInputElts);
  assert(numGlobalFeatures == inputBuffers->singleInputGlobalElts);

  size_t policyResultChannels = inputBuffers->policyResultChannels;
  size_t singleSpatialElts = inputBuffers->singleSpatialElts;
  size_t singleInputGlobalElts = inputBuffers->singleInputGlobalElts;
  size_t singlePolicyResultElts = inputBuffers->singleNnPolicyResultElts;
  size_t singlePolicyPassResultElts = inputBuffers->singlePolicyPassResultElts;
  size_t singleValueResultElts = inputBuffers->singleValueResultElts;
  size_t singleOwnershipResultElts = inputBuffers->singleNnOwnershipResultElts;
  size_t singleScoreValuesResultElts = inputBuffers->singleScoreValuesResultElts;
  size_t singlePolicyProbsElts = inputBuffers->singlePolicyProbsElts;

  assert(policyResultChannels == 1);
  assert(singleValueResultElts == 3);
  assert(singleScoreValuesResultElts >= 6);

  for(size_t row = 0; row < batchSize; row++) {
    float* rowSpatialInput = &inputBuffers->userInputBuffer[singleSpatialElts * row];
    float* rowGlobalInput = &inputBuffers->userInputGlobalBuffer[singleInputGlobalElts * row];
    const float* rowGlobal = inputBufs[row]->rowGlobal;
    const float* rowSpatial = inputBufs[row]->rowSpatial;

    copy(&rowGlobal[0], &rowGlobal[numGlobalFeatures], rowGlobalInput);

    SymmetryHelpers::copyInputsWithSymmetry(
      rowSpatial,
      rowSpatialInput,
      1,
      nnYLen,
      nnXLen,
      numSpatialFeatures,
      gpuHandle->inputsUseNHWC,
      inputBufs[row]->symmetry);
  }

  getMetalHandleOutput(inputBuffers->userInputBuffer,
                       inputBuffers->userInputGlobalBuffer,
                       inputBuffers->policyResults,
                       inputBuffers->policyPassResults,
                       inputBuffers->valueResults,
                       inputBuffers->ownershipResults,
                       inputBuffers->scoreValuesResults,
                       gpuHandle->gpuIndex,
                       batchSize);

  for(size_t row = 0; row < batchSize; row++) {
    NNOutput* output = outputs[row];

    assert(output->nnXLen == nnXLen);
    assert(output->nnYLen == nnYLen);

    float* policyOutputBuf = &inputBuffers->policyResults[row * (singlePolicyResultElts * policyResultChannels)];

    // These are not actually correct, the client does the postprocessing to turn them into
    // policy probabilities and white game outcome probabilities
    // Also we don't fill in the nnHash here either
    SymmetryHelpers::copyOutputsWithSymmetry(
      policyOutputBuf, output->policyProbs, 1, nnYLen, nnXLen, inputBufs[row]->symmetry);

    output->policyProbs[singlePolicyProbsElts - 1] = inputBuffers->policyPassResults[row * singlePolicyPassResultElts];

    const float* valueOutputBuf = &inputBuffers->valueResults[row * singleValueResultElts];

    output->whiteWinProb = valueOutputBuf[0];
    output->whiteLossProb = valueOutputBuf[1];
    output->whiteNoResultProb = valueOutputBuf[2];

    if(output->whiteOwnerMap != NULL) {
      const float* ownershipOutputBuf = &inputBuffers->ownershipResults[row * singleOwnershipResultElts];

      SymmetryHelpers::copyOutputsWithSymmetry(
        ownershipOutputBuf, output->whiteOwnerMap, 1, nnYLen, nnXLen, inputBufs[row]->symmetry);
    }

    const float* scoreValuesOutputBuf = &inputBuffers->scoreValuesResults[row * singleScoreValuesResultElts];

    if(version >= 9) {
      output->whiteScoreMean = scoreValuesOutputBuf[0];
      output->whiteScoreMeanSq = scoreValuesOutputBuf[1];
      output->whiteLead = scoreValuesOutputBuf[2];
      output->varTimeLeft = scoreValuesOutputBuf[3];
      output->shorttermWinlossError = scoreValuesOutputBuf[4];
      output->shorttermScoreError = scoreValuesOutputBuf[5];
    } else if(version >= 8) {
      output->whiteScoreMean = scoreValuesOutputBuf[0];
      output->whiteScoreMeanSq = scoreValuesOutputBuf[1];
      output->whiteLead = scoreValuesOutputBuf[2];
      output->varTimeLeft = scoreValuesOutputBuf[3];
      output->shorttermWinlossError = 0;
      output->shorttermScoreError = 0;
    } else if(version >= 4) {
      output->whiteScoreMean = scoreValuesOutputBuf[0];
      output->whiteScoreMeanSq = scoreValuesOutputBuf[1];
      output->whiteLead = output->whiteScoreMean;
      output->varTimeLeft = 0;
      output->shorttermWinlossError = 0;
      output->shorttermScoreError = 0;
    } else {
      assert(version >= 3);
      output->whiteScoreMean = scoreValuesOutputBuf[0];
      // Version 3 neural nets don't have any second moment output, implicitly already folding it in, so we just use the
      // mean squared
      output->whiteScoreMeanSq = output->whiteScoreMean * output->whiteScoreMean;
      output->whiteLead = output->whiteScoreMean;
      output->varTimeLeft = 0;
      output->shorttermWinlossError = 0;
      output->shorttermScoreError = 0;
    }
  }
}

/**
 * @brief Compute the neural network output using the specified input data and GPU handle.
 * This function computes the neural network output using the specified input data and ComputeHandle object
 * for GPU acceleration. The computed output is stored in the specified vector of NNOutput pointers.
 * @param gpuHandle A pointer to the ComputeHandle object to use for GPU computation.
 * @param inputBuffers A pointer to the InputBuffers object containing the input data for computation.
 * @param numBatchEltsFilled The number of batch elements filled in the input buffer.
 * @param inputBufs An array of pointers to NNResultBuf objects containing the neural network input data.
 * @param outputs A vector of NNOutput pointers to store the computed output.
 */
void NeuralNet::getOutput(
  ComputeHandle* gpuHandle,
  InputBuffers* inputBuffers,
  int numBatchEltsFilled,
  NNResultBuf** inputBufs,
  vector<NNOutput*>& outputs) {

  if (gpuHandle->useMetal) {
    getMetalOutput(gpuHandle, inputBuffers, numBatchEltsFilled, inputBufs, outputs);
  } else {
    getCoreMLOutput(gpuHandle, inputBuffers, numBatchEltsFilled, inputBufs, outputs);
  }
}

/**
 * @brief Evaluate a convolutional layer using Metal API for testing purposes.
 * This function evaluates a convolutional layer using the Metal API for testing purposes.
 * The input buffer and output buffer are specified as vectors of floats, and the result of the computation
 * is stored in the output buffer. The function returns true if the evaluation is implemented.
 * @param desc A pointer to the ConvLayerDesc object describing the convolutional layer to evaluate.
 * @param batchSize The batch size to use for computation.
 * @param nnXLen The x length of the neural network computation context.
 * @param nnYLen The y length of the neural network computation context.
 * @param useFP16 A boolean indicating whether to use half-precision floating point format for computation.
 * @param useNHWC A boolean indicating whether to use NHWC layout for input and output buffers.
 * @param inputBuffer A vector of floats containing the input buffer data.
 * @param outputBuffer A vector of floats to store the computed output.
 * @return true if the convolutional layer evaluation is implemented, false otherwise.
 */
bool NeuralNet::testEvaluateConv(
  const ConvLayerDesc* desc,
  int batchSize,
  int nnXLen,
  int nnYLen,
  bool useFP16,
  bool useNHWC,
  const vector<float>& inputBuffer,
  vector<float>& outputBuffer) {
  return false;
}

// Mask should be in 'NHW' format (no "C" channel).

/**
 * @brief Evaluate a batch normalization layer using Metal API for testing purposes.
 * This function evaluates a batch normalization layer using the Metal API for testing purposes.
 * The input buffer and output buffer are specified as vectors of floats, and the result of the computation
 * is stored in the output buffer. The function returns true if the evaluation is implemented.
 * @param desc A pointer to the BatchNormLayerDesc object describing the batch normalization layer to evaluate.
 * @param batchSize The batch size to use for computation.
 * @param nnXLen The x length of the neural network computation context.
 * @param nnYLen The y length of the neural network computation context.
 * @param useFP16 A boolean indicating whether to use half-precision floating point format for computation.
 * @param useNHWC A boolean indicating whether to use NHWC layout for input and output buffers.
 * @param inputBuffer A vector of floats containing the input buffer data.
 * @param maskBuffer A vector of floats containing the mask buffer data.
 * @param outputBuffer A vector of floats to store the computed output.
 * @return true if the batch normalization layer evaluation is implemented, false otherwise.
 */
bool NeuralNet::testEvaluateBatchNorm(
  const BatchNormLayerDesc* desc,
  int batchSize,
  int nnXLen,
  int nnYLen,
  bool useFP16,
  bool useNHWC,
  const vector<float>& inputBuffer,
  const vector<float>& maskBuffer,
  vector<float>& outputBuffer) {
  return false;
}

/**
 * @brief Evaluate a residual block using Metal API for testing purposes.
 * This function evaluates a residual block using the Metal API for testing purposes.
 * The input buffer and output buffer are specified as vectors of floats, and the result of the computation
 * is stored in the output buffer. The function returns true if the evaluation is implemented.
 * @param desc A pointer to the ResidualBlockDesc object describing the residual block to evaluate.
 * @param batchSize The batch size to use for computation.
 * @param nnXLen The x length of the neural network computation context.
 * @param nnYLen The y length of the neural network computation context.
 * @param useFP16 A boolean indicating whether to use half-precision floating point format for computation.
 * @param useNHWC A boolean indicating whether to use NHWC layout for input and output buffers.
 * @param inputBuffer A vector of floats containing the input buffer data.
 * @param maskBuffer A vector of floats containing the mask buffer data.
 * @param outputBuffer A vector of floats to store the computed output.
 * @return true if the residual block evaluation is implemented, false otherwise.
 */
bool NeuralNet::testEvaluateResidualBlock(
  const ResidualBlockDesc* desc,
  int batchSize,
  int nnXLen,
  int nnYLen,
  bool useFP16,
  bool useNHWC,
  const vector<float>& inputBuffer,
  const vector<float>& maskBuffer,
  vector<float>& outputBuffer) {
  return false;
}

/**
 * @brief Evaluate a global pooling residual block using Metal API for testing purposes.
 * This function evaluates a global pooling residual block using the Metal API for testing purposes.
 * The input buffer and output buffer are specified as vectors of floats, and the result of the computation
 * is stored in the output buffer. The function returns true if the evaluation is implemented.
 * @param desc A pointer to the GlobalPoolingResidualBlockDesc object describing the global pooling residual block to
 * evaluate.
 * @param batchSize The batch size to use for computation.
 * @param nnXLen The x length of the neural network computation context.
 * @param nnYLen The y length of the neural network computation context.
 * @param useFP16 A boolean indicating whether to use half-precision floating point format for computation.
 * @param useNHWC A boolean indicating whether to use NHWC layout for input and output buffers.
 * @param inputBuffer A vector of floats containing the input buffer data.
 * @param maskBuffer A vector of floats containing the mask buffer data.
 * @param outputBuffer A vector of floats to store the computed output.
 * @return true if the global pooling residual block evaluation is implemented, false otherwise.
 */
bool NeuralNet::testEvaluateGlobalPoolingResidualBlock(
  const GlobalPoolingResidualBlockDesc* desc,
  int batchSize,
  int nnXLen,
  int nnYLen,
  bool useFP16,
  bool useNHWC,
  const vector<float>& inputBuffer,
  const vector<float>& maskBuffer,
  vector<float>& outputBuffer) {
  return false;
}

#endif  // USE_COREML_BACKEND
