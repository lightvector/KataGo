#ifndef NEURALNET_NNINTERFACE_H_
#define NEURALNET_NNINTERFACE_H_

#include "../core/global.h"
#include "../core/hash.h"
#include "../core/logger.h"
#include "../neuralnet/nninputs.h"
#include "../neuralnet/desc.h"

// A handle to the local compute backend. Not thread-safe, each handle should
// only be used by one thread.
struct ComputeHandle;

// The interface for the input buffers for the neural network. The MCTS code
// uses this interface to pass data into the neural network for computation.
struct InputBuffers {
  virtual ~InputBuffers();

  // Returns a pointer to a float array of size getBatchLen() in (N)HWC or
  // (N)CHW format that can be filled with board input data.
  virtual float* getBatchInplace(int nIdx) = 0;

  // Returns a pointer to a float array of size getGlobalLen() that can be
  // filled with global input data.
  virtual float* getBatchGlobalInplace(int nIdx) = 0;

  // Returns a pointer to bool array of length 3 to input the board symmetries.
  virtual bool* getSymmetriesInplace() = 0;

  virtual int getBatchLen() const = 0;
  virtual int getGlobalLen() const = 0;
};


// A handle to the loaded neural network model.
struct LoadedModel;

// Generic interface to neural net inference.
// There is a single CUDA backend.

namespace NeuralNet {
  // Call globalInitialize() once upon program startup to construct the net.
  void globalInitialize();
  // Call globalCleanup() at program termination.
  void globalCleanup();

  // Model I/O -----------------------------------------------------------------
  LoadedModel* loadModelFile(const std::string& file, int modelFileIdx);
  void freeLoadedModel(LoadedModel* loadedModel);
  int getModelVersion(const LoadedModel* loadedModel);

  // Any given thread should only ever create one of these at a time.
  // When using the CUDA backend, will mutably set the GPU that this thread is
  // associated with to the specified index. If logger is specified, may output
  // some info messages to it. If requireExactNNLen is true, the backend is
  // allowed to assume that all boards to evaluate will be of size exactly equal
  // to (nnXLen,nnYLen) rather than smaller, and skip any masking operations.
  ComputeHandle* createComputeHandle(const LoadedModel* loadedModel,
                                     Logger* logger,
                                     int maxBatchSize,
                                     int nnXLen,
                                     int nnYLen,
                                     bool requireExactNNLen,
                                     bool inputsUseNHWC,
                                     int cudaGpuIdxForThisThread,
                                     bool useFP16,
                                     bool cudaUseNHWC);
  void freeComputeHandle(ComputeHandle* computeHandle);

  InputBuffers* createInputBuffers(const LoadedModel* loadedModel,
                                   int maxBatchSize,
                                   int nnXLen,
                                   int nnYLen);
  void freeInputBuffers(InputBuffers* buffers);

  void getOutput(ComputeHandle* gpuHandle,
                 InputBuffers* buffers,
                 int numFilledRows,
                 std::vector<NNOutput*>& outputs);

}  // namespace NeuralNet

#endif  // NEURALNET_NNINTERFACE_H_
