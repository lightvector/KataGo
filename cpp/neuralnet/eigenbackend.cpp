/** Eigen3 backend.
 *
 * Only supports float32 computation with NHWC memory layout (at runtime and as input).
 * Does not currently support symmetries.
 */

// CR lpuchallafiore: Add support for symmetries.
// CR lpuchallafiore: Add multi-threading support (see "Evaluating with a Thread Pool" in the Eigen Tensor docs).

#include "../neuralnet/nninterface.h"

#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>
#include <zstr/src/zstr.hpp>

#include "../neuralnet/desc.h"
#include "../neuralnet/modelversion.h"

#define SCALAR float

using namespace std;

using Eigen::Tensor;

// LoadedModel / ModelDesc ---------------------------------------------------------------------------------------------

struct LoadedModel {
  ModelDesc modelDesc;

  LoadedModel(istream& in) { modelDesc = std::move(ModelDesc(in)); }

  LoadedModel() = delete;
  LoadedModel(const LoadedModel&) = delete;
  LoadedModel& operator=(const LoadedModel&) = delete;
};

// Layers --------------------------------------------------------------------------------------------------------------

// Convolution layer with zero-padding and dilation filtering.
struct ConvLayer {
  string name;

  Eigen::array<pair<int, int>, 4> paddings;
  Tensor<SCALAR, 4> kernel;  // outC x fW x fH x inC

  ConvLayer() = delete;
  ConvLayer(const ConvLayer&) = delete;
  ConvLayer& operator=(const ConvLayer&) = delete;

  ConvLayer(const ConvLayerDesc& desc, int maxBatchSize) {
    name = desc.name;
    int convYSize = desc.convYSize;
    int convXSize = desc.convXSize;
    int inChannels = desc.inChannels;
    int outChannels = desc.outChannels;
    int dilationY = desc.dilationY;
    int dilationX = desc.dilationX;
    int paddingX = (convXSize / 2) * dilationX;
    int paddingY = (convYSize / 2) * dilationY;

    assert(convXSize % 2 == 1);
    assert(convYSize % 2 == 1);

    paddings[0] = make_pair(0, 0);                // C
    paddings[1] = make_pair(paddingX, paddingX);  // W
    paddings[2] = make_pair(paddingY, paddingY);  // H
    paddings[3] = make_pair(0, 0);                // N

    bool filterNHWC = dilationY == 1 && dilationX == 1;
    if(filterNHWC) {
      vector<float> weightsTransposed(desc.weights.size());
      for(int y = 0; y < convYSize; y++) {
        for(int x = 0; x < convXSize; x++) {
          for(int ic = 0; ic < inChannels; ic++) {
            for(int oc = 0; oc < outChannels; oc++) {
              weightsTransposed[((oc * convYSize + y) * convXSize + x) * inChannels + ic] =
                desc.weights[((oc * inChannels + ic) * convYSize + y) * convXSize + x];
            }
          }
        }
      }
      // CR lpuchallafiore: double-check this forces a copy to kernel, otherwise weightsTranspose will be deleted and
      // this will not work.
      kernel =
        Eigen::TensorMap<const Tensor<SCALAR, 4>>(&weightsTransposed[0], outChannels, convXSize, convYSize, inChannels);
    } else {
      kernel =
        Eigen::TensorMap<const Tensor<SCALAR, 4>>((float *)&desc.weights[0], outChannels, convXSize, convYSize, inChannels);
    }
  }

  // CR lpuchallafiore: accumulate?
  void apply(const Tensor<SCALAR, 4>& input, Tensor<SCALAR, 4>* output, bool accumulate) const {
    Eigen::array<ptrdiff_t, 2> dims({1, 2});  // Specify second (W) and third (H) dimensions for convolution.
    *output = input.pad(paddings).convolve(kernel, dims);
  }
};

// Model and Buffer I/O ------------------------------------------------------------------------------------------------

struct InputBuffers {
  int maxBatchSize;

  size_t singleInputElts;
  size_t singleInputGlobalElts;

  // Eigen tensors are stored in column-major order, so an NHWC memory layout is given by Tensor<4>(C,W,H,N).
  Tensor<SCALAR, 4> spatialInput;
  Tensor<SCALAR, 4> globalInput;
  bool* symmetriesBuffer;

  InputBuffers(const LoadedModel* loadedModel, int maxBatchSz, int nnXLen, int nnYLen) {
    const ModelDesc& m = loadedModel->modelDesc;

    int xSize = m.version >= 3 ? nnXLen : m.xSizePreV3;
    int ySize = m.version >= 3 ? nnYLen : m.ySizePreV3;

    maxBatchSize = maxBatchSz;
    singleInputElts = m.numInputChannels * xSize * ySize;
    singleInputGlobalElts = m.numInputGlobalChannels;

    assert(NNModelVersion::getNumSpatialFeatures(m.version) == m.numInputChannels);
    assert(NNModelVersion::getNumGlobalFeatures(m.version) == m.numInputGlobalChannels);

    spatialInput = Tensor<SCALAR, 4>(m.numInputChannels, xSize, ySize, maxBatchSize);
    globalInput = Tensor<SCALAR, 4>(1, 1, m.numInputGlobalChannels, maxBatchSize);

    symmetriesBuffer = new bool[NNInputs::NUM_SYMMETRY_BOOLS];
  }

  ~InputBuffers() { delete[] symmetriesBuffer; }

  InputBuffers() = delete;
  InputBuffers(const InputBuffers&) = delete;
  InputBuffers& operator=(const InputBuffers&) = delete;
};

InputBuffers* NeuralNet::createInputBuffers(const LoadedModel* loadedModel, int maxBatchSize, int nnXLen, int nnYLen) {
  return new InputBuffers(loadedModel, maxBatchSize, nnXLen, nnYLen);
}
void NeuralNet::freeInputBuffers(InputBuffers* inputBuffers) {
  delete inputBuffers;
}

float* NeuralNet::getBatchEltSpatialInplace(InputBuffers* inputBuffers, int nIdx) {
  assert(nIdx < inputBuffers->maxBatchSize);
  return inputBuffers->spatialInput.data() + (inputBuffers->singleInputElts * nIdx);
}

float* NeuralNet::getBatchEltGlobalInplace(InputBuffers* inputBuffers, int rowIdx) {
  assert(rowIdx < inputBuffers->maxBatchSize);
  return inputBuffers->globalInput.data() + (inputBuffers->singleInputGlobalElts * rowIdx);
}

int NeuralNet::getBatchEltSpatialLen(const InputBuffers* inputBuffers) {
  return inputBuffers->singleInputElts;
}
int NeuralNet::getBatchEltGlobalLen(const InputBuffers* inputBuffers) {
  return inputBuffers->singleInputGlobalElts;
}

bool* NeuralNet::getSymmetriesInplace(InputBuffers* inputBuffers) {
  return inputBuffers->symmetriesBuffer;
}

LoadedModel* NeuralNet::loadModelFile(const string& file, int modelFileIdx) {
  (void)modelFileIdx;

  try {
    // zstr has a bad property of simply aborting if the file doesn't exist
    // So we try to catch this common error by explicitly testing first if the
    // file exists by trying to open it normally to turn it into a regular C++
    // exception.
    {
      ifstream testIn(file);
      if(!testIn.good())
        throw StringError("File does not exist or could not be opened");
    }
    zstr::ifstream in(file);
    LoadedModel* loadedModel = new LoadedModel(in);
    return loadedModel;
  } catch(const StringError& e) {
    throw StringError("Error parsing model file " + file + ": " + e.what());
  }
}

void NeuralNet::freeLoadedModel(LoadedModel* loadedModel) {
  delete loadedModel;
}

int NeuralNet::getModelVersion(const LoadedModel* loadedModel) {
  return loadedModel->modelDesc.version;
}

// NeuralNet -----------------------------------------------------------------------------------------------------------

void NeuralNet::globalInitialize() {
  // no-op for cpu
}

void NeuralNet::globalCleanup() {
  // no-op for cpu
}

struct ComputeHandle {
  // unique_ptr<Model> model;
  int nnXLen;
  int nnYLen;
  bool requireExactPosLen;
  int policySize;

  ComputeHandle(const LoadedModel& loadedModel, int maxBatchSize, int xLen, int yLen, bool rExactPosLen)
    :  // model(make_unique<Model>(loadedModel.modelDesc, maxBatchSize, xLen, yLen)),
      nnXLen(xLen),
      nnYLen(yLen),
      requireExactPosLen(rExactPosLen),
      policySize(NNPos::getPolicySize(nnXLen, nnYLen)) {}
};

ComputeHandle* NeuralNet::createComputeHandle(
  const LoadedModel* loadedModel,
  Logger* logger,
  int maxBatchSize,
  int nnXLen,
  int nnYLen,
  bool requireExactPosLen,
  bool inputsUseNHWC,
  int cudaGpuIdxForThisThread,
  bool useFP16,
  bool cudaUseNHWC) {
  (void)cudaUseNHWC;      // Always use NHWC
  (void)useFP16;          // Always use FP32
  assert(inputsUseNHWC);  // Only support inputs in NHWC format.
  return new ComputeHandle(*loadedModel, maxBatchSize, nnXLen, nnYLen, requireExactPosLen);
}

void NeuralNet::freeComputeHandle(ComputeHandle* gpuHandle) {
  delete gpuHandle;
}

void NeuralNet::getOutput(
  ComputeHandle* gpuHandle,
  InputBuffers* buffers,
  int numFilledRows,
  vector<NNOutput*>& outputs) {
  assert(false);
}
