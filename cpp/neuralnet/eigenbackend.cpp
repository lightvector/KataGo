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

// Debugging -----------------------------------------------------------------------------------------------------------
void printTensor4Size(const string& name, const Tensor<SCALAR, 4>& t) {
  cout << name << " rank=" << t.NumDimensions << " - ";
  for(int i = 0; i < t.NumDimensions; i++) {
    cout << t.dimension(i) << "x";
  }
  cout << endl;
}

// NHWC
void printTensor4(const string& name, const Tensor<SCALAR, 4>& t) {
  printTensor4Size(name, t);
  for(int n = 0; n < t.dimension(3); n++) {
    cout << "n = " << n << endl;
    for(int h = 0; h < t.dimension(2); h++) {
      for(int w = 0; w < t.dimension(1); w++) {
        for(int c = 0; c < t.dimension(0); c++) {
          cout << t(c, w, h, n) << (c == t.dimension(0) - 1 ? ", " : " ");
        }
      }
      cout << endl;
    }
    cout << endl;
  }
}

// LoadedModel / ModelDesc ---------------------------------------------------------------------------------------------

struct LoadedModel {
  ModelDesc modelDesc;

  LoadedModel(istream& in) { modelDesc = std::move(ModelDesc(in)); }

  LoadedModel() = delete;
  LoadedModel(const LoadedModel&) = delete;
  LoadedModel& operator=(const LoadedModel&) = delete;
};

// Layers --------------------------------------------------------------------------------------------------------------

// Convolution layer with zero-padding.
struct ConvLayer {
  string name;

  Eigen::array<pair<int, int>, 4> paddings;
  Tensor<SCALAR, 4> kernel;
  int inChannels, outChannels;
  int paddingX, paddingY;

  ConvLayer() = delete;
  ConvLayer(const ConvLayer&) = delete;
  ConvLayer& operator=(const ConvLayer&) = delete;

  ConvLayer(const ConvLayerDesc& desc, int maxBatchSize) {
    name = desc.name;
    int convYSize = desc.convYSize;
    int convXSize = desc.convXSize;
    inChannels = desc.inChannels;
    outChannels = desc.outChannels;
    // CR lpuchallafiore: dilation?
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

    // CR-someday lpuchallafiore: optimize NHWC vs NCHW, etc.
    kernel = Eigen::TensorMap<const Tensor<SCALAR, 4>>(
      (float*)&desc.weights[0], convXSize, convYSize, inChannels, outChannels);
  }

  // CR lpuchallafiore: accumulate?
  void apply(const Tensor<SCALAR, 4>& input, Tensor<SCALAR, 4>* output, bool accumulate) const {
    auto padded = input.pad(paddings);

    *output = Tensor<SCALAR, 4>(outChannels, input.dimension(1), input.dimension(2), input.dimension(3));
    for(int n = 0; n < input.dimension(3); n++) {
      auto inN = padded.chip(n, 3);
      for(int oc = 0; oc < outChannels; oc++) {
        Tensor<SCALAR, 2> sum(input.dimension(1), input.dimension(2));
        sum.setZero();

        for(int ic = 0; ic < inChannels; ic++) {
          Eigen::array<ptrdiff_t, 2> dims({0, 1});
          auto kChip = kernel.chip(oc, 3).chip(ic, 2);
          auto inNC = inN.chip(ic, 0);
          sum += inNC.convolve(kChip, dims);
        }

        output->chip(n, 3).chip(oc, 0) = sum;
      }
    }
  }
};

struct BatchNormLayer {
  string name;
  int numChannels;
  float epsilon;
  int xSize;
  int ySize;

  vector<float> mergedScale;
  vector<float> mergedBias;

  BatchNormLayer() = delete;
  BatchNormLayer(const BatchNormLayer&) = delete;
  BatchNormLayer& operator=(const BatchNormLayer&) = delete;

  BatchNormLayer(const BatchNormLayerDesc& desc) {
    name = desc.name;
    numChannels = desc.numChannels;
    epsilon = desc.epsilon;

    mergedScale.resize(numChannels);
    mergedBias.resize(numChannels);
    for(int c = 0; c < numChannels; c++) {
      mergedScale[c] = desc.scale[c] / sqrt(desc.variance[c] + epsilon);
      mergedBias[c] = desc.bias[c] - mergedScale[c] * desc.mean[c];
    }
  }

  ~BatchNormLayer() {}

  // Mask should be in 'NHW' format (no "C" channel).
  void apply(
    bool applyRelu,
    const Tensor<SCALAR, 4>& input,
    const Tensor<SCALAR, 3>& mask,
    Tensor<SCALAR, 4>* output) const {

    *output = Tensor<SCALAR, 4>(input.dimension(0), input.dimension(1), input.dimension(2), input.dimension(3));
    for (int c = 0; c < input.dimension(0); c++) {
      auto inC = input.chip(c, 0);
      auto x = inC * mergedScale[c] + mergedBias[c];
      if (applyRelu) {
        auto z = Tensor<SCALAR, 3>(mask.dimension(0), mask.dimension(1), mask.dimension(2)).setZero();
        output->chip(c, 0) = (mask == 1.f).select(x.cwiseMax(0.f), z);
      } else {
        auto z = Tensor<SCALAR, 3>(mask.dimension(0), mask.dimension(1), mask.dimension(2)).setZero();
        output->chip(c, 0) = (mask == 1.f).select(x, z);
      }
    }
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

Rules NeuralNet::getSupportedRules(const LoadedModel* loadedModel, const Rules& desiredRules, bool& supported) {
  return loadedModel->modelDesc.getSupportedRules(desiredRules, supported);
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

// FOR TESTING ---------------------------------------------------------------------------------------------------------
bool NeuralNet::testEvaluateConv(
  const ConvLayerDesc* desc,
  int batchSize,
  int nnXLen,
  int nnYLen,
  bool useFP16,
  bool useNHWC,
  const std::vector<float>& inputBuffer,
  std::vector<float>& outputBuffer) {
  if(useNHWC && !useFP16) {
    ConvLayer layer(*desc, batchSize);
    Eigen::TensorMap<const Tensor<SCALAR, 4>> inTensor(
      (float*)&inputBuffer[0], desc->inChannels, nnXLen, nnYLen, batchSize);
    Tensor<SCALAR, 4> outTensor;

    layer.apply(inTensor, &outTensor, false);

    outputBuffer.resize(outTensor.size());
    memcpy(&outputBuffer[0], outTensor.data(), sizeof(SCALAR) * outTensor.size());
    return true;
  } else {
    return false;
  }
}

// Mask should be in 'NHW' format (no "C" channel).
bool NeuralNet::testEvaluateBatchNorm(
  const BatchNormLayerDesc* desc,
  int batchSize,
  int nnXLen,
  int nnYLen,
  bool useFP16,
  bool useNHWC,
  const std::vector<float>& inputBuffer,
  const std::vector<float>& maskBuffer,
  std::vector<float>& outputBuffer) {
  if(useNHWC && !useFP16) {
    BatchNormLayer layer(*desc);
    Eigen::TensorMap<const Tensor<SCALAR, 4>> inTensor(
      (float*)&inputBuffer[0], desc->numChannels, nnXLen, nnYLen, batchSize);
    Eigen::TensorMap<const Tensor<SCALAR, 3>> maskTensor((float*)&maskBuffer[0], nnXLen, nnYLen, batchSize);
    Tensor<SCALAR, 4> outTensor;

    layer.apply(false, inTensor, maskTensor, &outTensor);

    outputBuffer.resize(outTensor.size());
    memcpy(&outputBuffer[0], outTensor.data(), sizeof(SCALAR) * outTensor.size());
    return true;
  } else {
    return false;
  }
}

bool NeuralNet::testEvaluateResidualBlock(
  const ResidualBlockDesc* desc,
  int batchSize,
  int nnXLen,
  int nnYLen,
  bool useFP16,
  bool useNHWC,
  const std::vector<float>& inputBuffer,
  const std::vector<float>& maskBuffer,
  std::vector<float>& outputBuffer) {
  return false;
}

bool NeuralNet::testEvaluateGlobalPoolingResidualBlock(
  const GlobalPoolingResidualBlockDesc* desc,
  int batchSize,
  int nnXLen,
  int nnYLen,
  bool useFP16,
  bool useNHWC,
  const std::vector<float>& inputBuffer,
  const std::vector<float>& maskBuffer,
  std::vector<float>& outputBuffer) {
  return false;
}
