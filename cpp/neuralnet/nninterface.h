#ifndef NEURALNET_NNINTERFACE_H_
#define NEURALNET_NNINTERFACE_H_

#include "../core/global.h"
#include "../core/commontypes.h"
#include "../core/config_parser.h"
#include "../core/hash.h"
#include "../core/logger.h"
#include "../neuralnet/desc.h"
#include "../neuralnet/nninputs.h"

// Defined in nneval.h
struct NNResultBuf;

// A handle to cross-thread cross-gpu initialization state.
// Create one of these per process, although creating more is fine.
struct ComputeContext;

// A handle to the local compute backend. Not thread-safe, each handle should
// only be used by one thread.
struct ComputeHandle;

// The interface for the input buffers for the neural network. The MCTS code
// uses this interface to pass data into the neural network for computation.
struct InputBuffers;

// A handle to the loaded neural network model.
struct LoadedModel;

// Generic interface to neural net inference.
// There is a single CUDA backend.
namespace NeuralNet {
  // Call globalInitialize() once upon program startup to construct the net.
  void globalInitialize();
  // Call globalCleanup() at program termination.
  void globalCleanup();

  // Print available backend devices
  void printDevices();

  // A short lowercase alphanumeric string identifying any materially-different runtime
  // configuration of this backend, or the empty string if the compile-time backend name
  // already says everything (the common case).
  // Currently only the ONNX backend returns anything: the execution provider selected
  // by the onnxProvider config key (e.g. "openvino", "directml"),
  // Used to disambiguate the version that contribute reports to the data server,
  // so keep results short and stable for a given config.
  std::string getRuntimeBackendDetail(ConfigParser& cfg);

  // How this backend wants nnMaxBatchSize chosen. Setup::initializeNNEvaluator asks the backend
  // rather than deriving one number for everybody, because "bigger batches up to the number of
  // concurrent evaluations" is only right for GPU backends that accept any batch size.
  enum class BatchPolicy {
    // Any batch size is fine and larger ones amortize per-call overhead, so size the batch to the
    // work that can be in flight. Every GPU backend except the fixed-shape cases below.
    Dynamic,
    // The graph is compiled for one exact input shape and recompiles whenever it changes, so the
    // batch size is held constant and short batches are padded up to it (see the ONNX backend's
    // onnxPadBatch). Wants a batch of about half the evals one device sees at a time, so that two
    // batches can be in flight per device without the padding wasting many rows.
    FixedShape,
    // Batching buys nothing: each evaluation runs single-threaded on a CPU core, with parallelism
    // coming from many server threads evaluating at once, so a big batch only costs memory. Wants
    // a small fixed size regardless of concurrency or config. Currently also tells the benchmark
    // that the server threads are the CPU workers, so it resizes and respawns them per tested
    // thread count. A CPU-style backend that manages its own parallelism instead should get a new
    // policy value here rather than reusing this one.
    CpuLocal,
  };
  BatchPolicy getBatchPolicy(ConfigParser& cfg);

  // The number of distinct devices that server threads with these gpu indices will run on, for
  // per-device sizing such as Setup::computeFixedShapeMaxBatchSize. Most backends identify a
  // device by the gpu index alone. The ONNX backend's OpenVINO provider instead selects devices
  // by device-type string, which it reads from the same config keys createComputeContext does.
  int getNumEffectiveDevices(ConfigParser& cfg, const std::vector<int>& gpuIdxByServerThread);

  // Model I/O -----------------------------------------------------------------

  LoadedModel* loadModelFile(const std::string& file, const std::string& expectedSha256);
  void freeLoadedModel(LoadedModel* loadedModel);

  const ModelDesc& getModelDesc(const LoadedModel* loadedModel);

  // Context -------------------------------------------------------------------

  ComputeContext* createComputeContext(
    // The indices of all gpus that this context will be used for.
    // -1 as an entry indicates to select a default
    const std::vector<int>& gpuIdxs,
    Logger* logger,
    int nnXLen,
    int nnYLen,
    const std::string& homeDataDirOverride,
    enabled_t useFP16Mode,
    const LoadedModel* loadedModel,
    // Config that the backend may consult for its own custom options (e.g. OpenCL tuner file, cuDNN
    // SDPA disable). Backends read whatever keys they care about directly off of this.
    ConfigParser& cfg
  );
  // A ComputeContext should NOT be freed until all ComputeHandles created using it have also been freed.
  void freeComputeContext(ComputeContext* computeContext);

  // Compute Handle -----------------------------------------------------------------

  // Any given thread should only ever create one of these at a time.
  // When using the CUDA backend, will mutably set the GPU that this thread is
  // associated with to the specified index. If logger is specified, may output
  // some info messages to it. If requireExactNNLen is true, the backend is
  // allowed to assume that all boards to evaluate will be of size exactly equal
  // to (nnXLen,nnYLen) rather than smaller, and skip any masking operations.
  // gpuIdxForThisThread == -1 indicates to select a default GPU.
  ComputeHandle* createComputeHandle(
    ComputeContext* context,
    const LoadedModel* loadedModel,
    Logger* logger,
    int maxBatchSize,
    bool requireExactNNLen,
    bool inputsUseNHWC,
    int gpuIdxForThisThread,
    int serverThreadIdx
  );
  void freeComputeHandle(ComputeHandle* computeHandle);

  bool isUsingFP16(const ComputeHandle* computeHandle);

  // Set whether the handle is currently being used in a warmup mode, returning the previous value.
  // Currently only used during maybeWarmupComputeHandle to indicate for the CUDA backend that failures should
  // be a bit more lenient: during warmup a failed cudnn SDPA execution falls back to the custom kernel and
  // disables SDPA going forward, whereas outside of warmup such a failure is fatal.
  bool setIsWarmup(const ComputeHandle* computeHandle, bool isWarmup);

  //Input Buffers ---------------------------------------------------------------

  InputBuffers* createInputBuffers(const LoadedModel* loadedModel, int maxBatchSize, int nnXLen, int nnYLen);
  void freeInputBuffers(InputBuffers* buffers);

  // The neural net takes in 2 tensors as input.
  // One of them ("spatial") is 3-dimensional per-batch-element (4-dimensional including the batch dimension N),
  // containing floats for the values of different features (C) across the space of the board (H,W),
  // such as placement of stones and prior move locations.
  // The other ("global") is 1-dimensional per-batch-element containing floats for features that are
  // global to the board state, such as game rules and komi.

  // Perform Neural Net Evals ---------------------------------------------------------

  // Preconditions:
  // buffers inputBufs[nIdx]->{rowSpatial,rowGlobal} have been filled with input data for all values of nIdx in [0,numBatchEltsFilled-1]
  // outputs has length numBatchEltsFilled containing allocated but possibly-uninitialized NNOutput structs.

  // Result: mutably writes the results of the numBatchEltsFilled many parallel neural net evaluations
  // into the NNOutput structs.
  // All outputs are in logits - all final activation functions softmax, tanh, etc. are NOT applied.
  void getOutput(
    ComputeHandle* computeHandle,
    InputBuffers* buffers,
    int numBatchEltsFilled,
    NNResultBuf** inputBufs,
    std::vector<NNOutput*>& outputs
  );


  // FOR TESTING -----------------------------------------------------------------------
  // For all of the below, the input buffers must have exactly the size expected of the input for the operation.
  // If useNHWC, assumes inputBuffer and outputBuffer are NHWC format, else assumes NCHW format.

  // If the operation is implemented for testing, a backend should return true and evaluate the
  // specific operation on the input buffer, resizing the output buffer and writing the result.
  // If it is not implemented, backend should return false.

  bool testEvaluateConv(
    const ConvLayerDesc* desc,
    int batchSize,
    int nnXLen,
    int nnYLen,
    bool useFP16,
    bool useNHWC,
    const std::vector<float>& inputBuffer,
    std::vector<float>& outputBuffer
  );

  // Mask should be in 'NHW' format (no "C" channel).
  bool testEvaluateBatchNorm(
    const BatchNormLayerDesc* desc,
    int batchSize,
    int nnXLen,
    int nnYLen,
    bool useFP16,
    bool useNHWC,
    const std::vector<float>& inputBuffer,
    const std::vector<float>& maskBuffer,
    std::vector<float>& outputBuffer
  );

  bool testEvaluateResidualBlock(
    const ResidualBlockDesc* desc,
    int batchSize,
    int nnXLen,
    int nnYLen,
    bool useFP16,
    bool useNHWC,
    const std::vector<float>& inputBuffer,
    const std::vector<float>& maskBuffer,
    std::vector<float>& outputBuffer
  );

  bool testEvaluateGlobalPoolingResidualBlock(
    const GlobalPoolingResidualBlockDesc* desc,
    int batchSize,
    int nnXLen,
    int nnYLen,
    bool useFP16,
    bool useNHWC,
    const std::vector<float>& inputBuffer,
    const std::vector<float>& maskBuffer,
    std::vector<float>& outputBuffer
  );

}  // namespace NeuralNet


#endif  // NEURALNET_NNINTERFACE_H_
