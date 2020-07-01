#ifdef USE_EIGEN_BACKEND

/** Eigen3 backend.
 *
 * Only supports float32 computation with NHWC memory layout (at runtime and as input).
 */

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
// static void printTensor4Size(const string& name, const Tensor<SCALAR, 4>& t) {
//   cout << name << " rank=" << t.NumDimensions << " - ";
//   for(int i = 0; i < t.NumDimensions; i++) {
//     cout << t.dimension(i) << "x";
//   }
//   cout << endl;
// }

// // NHWC
// static void printTensor4(const string& name, const Tensor<SCALAR, 4>& t) {
//   printTensor4Size(name, t);
//   for(int n = 0; n < t.dimension(3); n++) {
//     cout << "n = " << n << endl;
//     for(int h = 0; h < t.dimension(2); h++) {
//       for(int w = 0; w < t.dimension(1); w++) {
//         for(int c = 0; c < t.dimension(0); c++) {
//           cout << t(c, w, h, n) << (c == t.dimension(0) - 1 ? ", " : " ");
//         }
//       }
//       cout << endl;
//     }
//     cout << endl;
//   }
// }

//------------------------------------------------------------------------------

struct LoadedModel {
  ModelDesc modelDesc;

  LoadedModel(const string& fileName) {
    ModelDesc::loadFromFileMaybeGZipped(fileName,modelDesc);
  }

  LoadedModel() = delete;
  LoadedModel(const LoadedModel&) = delete;
  LoadedModel& operator=(const LoadedModel&) = delete;
};

LoadedModel* NeuralNet::loadModelFile(const string& file) {
  LoadedModel* loadedModel = new LoadedModel(file);
  return loadedModel;
}

void NeuralNet::freeLoadedModel(LoadedModel* loadedModel) {
  delete loadedModel;
}

string NeuralNet::getModelName(const LoadedModel* loadedModel) {
  return loadedModel->modelDesc.name;
}

int NeuralNet::getModelVersion(const LoadedModel* loadedModel) {
  return loadedModel->modelDesc.version;
}

Rules NeuralNet::getSupportedRules(const LoadedModel* loadedModel, const Rules& desiredRules, bool& supported) {
  return loadedModel->modelDesc.getSupportedRules(desiredRules, supported);
}

//------------------------------------------------------------------------------

struct ComputeContext {
  int nnXLen;
  int nnYLen;
};

ComputeContext* NeuralNet::createComputeContext(
  const std::vector<int>& gpuIdxs,
  Logger* logger,
  int nnXLen,
  int nnYLen,
  const string& openCLTunerFile,
  const string& homeDataDirOverride,
  bool openCLReTunePerBoardSize,
  enabled_t useFP16Mode,
  enabled_t useNHWCMode,
  const LoadedModel* loadedModel
) {
  (void)gpuIdxs;
  (void)logger;
  (void)openCLTunerFile;
  (void)homeDataDirOverride;
  (void)openCLReTunePerBoardSize;
  (void)loadedModel;

  bool useFP16 = useFP16Mode == enabled_t::True ? true : false;
  bool useNHWC = useNHWCMode == enabled_t::False ? false : true;

  if(useFP16)
    throw StringError("Eigen backend: useFP16 = true not supported");
  if(!useNHWC)
    throw StringError("Eigen backend: useNHWC = false not supported");

  ComputeContext* context = new ComputeContext();
  context->nnXLen = nnXLen;
  context->nnYLen = nnYLen;
  return context;
}

void NeuralNet::freeComputeContext(ComputeContext* computeContext) {
  delete computeContext;
}

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

  ConvLayer(const ConvLayerDesc& desc) {
    name = desc.name;
    int convYSize = desc.convYSize;
    int convXSize = desc.convXSize;
    inChannels = desc.inChannels;
    outChannels = desc.outChannels;
    //Currently eigen impl doesn't support dilated convs
    int dilationY = desc.dilationY;
    int dilationX = desc.dilationX;
    paddingX = (convXSize / 2) * dilationX;
    paddingY = (convYSize / 2) * dilationY;

    if(dilationX != 1 || dilationY != 1)
      throw StringError("Eigen backend: Encountered convolution dilation factors other than 1, not supported");

    assert(convXSize % 2 == 1);
    assert(convYSize % 2 == 1);

    paddings[0] = make_pair(0, 0);                // C
    paddings[1] = make_pair(paddingX, paddingX);  // W
    paddings[2] = make_pair(paddingY, paddingY);  // H
    paddings[3] = make_pair(0, 0);                // N

    // CR-someday lpuchallafiore: optimize NHWC vs NCHW, etc.
    kernel = Eigen::TensorMap<const Tensor<const SCALAR, 4>>(
      &desc.weights[0], convXSize, convYSize, inChannels, outChannels);
  }

  void apply(const Tensor<SCALAR, 4>& input, Tensor<SCALAR, 4>& output, bool accumulate) const {
    auto padded = input.pad(paddings);
    assert(output.dimension(0) == outChannels);
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

        if(accumulate)
          output.chip(n, 3).chip(oc, 0) += sum;
        else
          output.chip(n, 3).chip(oc, 0) = sum;
      }
    }
  }
};

//--------------------------------------------------------------

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
    Tensor<SCALAR, 4>& output,
    const Tensor<SCALAR, 3>& mask
  ) const {

    output = Tensor<SCALAR, 4>(input.dimension(0), input.dimension(1), input.dimension(2), input.dimension(3));
    for (int c = 0; c < input.dimension(0); c++) {
      auto inC = input.chip(c, 0);
      auto x = inC * mergedScale[c] + mergedBias[c];
      if (applyRelu) {
        auto z = Tensor<SCALAR, 3>(mask.dimension(0), mask.dimension(1), mask.dimension(2)).setZero();
        output.chip(c, 0) = (mask == 1.f).select(x.cwiseMax(0.f), z);
      } else {
        auto z = Tensor<SCALAR, 3>(mask.dimension(0), mask.dimension(1), mask.dimension(2)).setZero();
        output.chip(c, 0) = (mask == 1.f).select(x, z);
      }
    }
  }
};

//--------------------------------------------------------------

struct ResidualBlock {
  string name;
  BatchNormLayer preBN;
  ConvLayer regularConv;
  BatchNormLayer midBN;
  ConvLayer finalConv;

  ResidualBlock(
    const ResidualBlockDesc& desc
  ): name(desc.name),
     preBN(desc.preBN),
     regularConv(desc.regularConv),
     midBN(desc.midBN),
     finalConv(desc.finalConv)
  {
  }

  ~ResidualBlock() {
  }

  ResidualBlock() = delete;
  ResidualBlock(const ResidualBlock&) = delete;
  ResidualBlock& operator=(const ResidualBlock&) = delete;

  void apply(
    Tensor<SCALAR, 4>& trunk,
    Tensor<SCALAR, 4>& trunkScratch,
    Tensor<SCALAR, 4>& mid,
    Tensor<SCALAR, 4>& midScratch,
    const Tensor<SCALAR, 3>& mask
  ) const {
    preBN.apply(true,trunk,trunkScratch,mask);
    regularConv.apply(trunkScratch,mid,false);
    midBN.apply(true,mid,midScratch,mask);
    finalConv.apply(midScratch,trunk,true);
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

    int xSize = nnXLen;
    int ySize = nnYLen;

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

// float* NeuralNet::getBatchEltSpatialInplace(InputBuffers* inputBuffers, int nIdx) {
//   assert(nIdx < inputBuffers->maxBatchSize);
//   return inputBuffers->spatialInput.data() + (inputBuffers->singleInputElts * nIdx);
// }

// float* NeuralNet::getBatchEltGlobalInplace(InputBuffers* inputBuffers, int rowIdx) {
//   assert(rowIdx < inputBuffers->maxBatchSize);
//   return inputBuffers->globalInput.data() + (inputBuffers->singleInputGlobalElts * rowIdx);
// }

// int NeuralNet::getBatchEltSpatialLen(const InputBuffers* inputBuffers) {
//   return inputBuffers->singleInputElts;
// }
// int NeuralNet::getBatchEltGlobalLen(const InputBuffers* inputBuffers) {
//   return inputBuffers->singleInputGlobalElts;
// }

// bool* NeuralNet::getSymmetriesInplace(InputBuffers* inputBuffers) {
//   return inputBuffers->symmetriesBuffer;
// }


// NeuralNet -----------------------------------------------------------------------------------------------------------

void NeuralNet::globalInitialize() {
  // no-op for cpu
}

void NeuralNet::globalCleanup() {
  // no-op for cpu
}

struct ComputeHandle {
  const ComputeContext* context;
  int maxBatchSize;
  // unique_ptr<Model> model;

  ComputeHandle(const ComputeContext* ctx, const LoadedModel& loadedModel, int maxBSize)
    : context(ctx),
      maxBatchSize(maxBSize)
    // model(make_unique<Model>(loadedModel.modelDesc, maxBatchSize, xLen, yLen)),
  {}
};

ComputeHandle* NeuralNet::createComputeHandle(
  ComputeContext* context,
  const LoadedModel* loadedModel,
  Logger* logger,
  int maxBatchSize,
  bool requireExactNNLen,
  bool inputsUseNHWC,
  int gpuIdxForThisThread
) {
  if(logger != NULL) {
    logger->write("Eigen backend: Model version " + Global::intToString(loadedModel->modelDesc.version));
    logger->write(
      "Eigen backend: Model name: " + loadedModel->modelDesc.name
    );
  }

  (void)requireExactNNLen; //We don't bother with mask optimizations if we know exact sizes right now.
  (void)gpuIdxForThisThread; //Doesn't matter

  if(!inputsUseNHWC)
    throw StringError("Eigen backend: inputsUseNHWC = false unsupported");
  return new ComputeHandle(context, *loadedModel, maxBatchSize);
}

void NeuralNet::freeComputeHandle(ComputeHandle* gpuHandle) {
  delete gpuHandle;
}

void NeuralNet::getOutput(
  ComputeHandle* gpuHandle,
  InputBuffers* inputBuffers,
  int numBatchEltsFilled,
  NNResultBuf** inputBufs,
  int symmetry,
  vector<NNOutput*>& outputs
) {
  (void)gpuHandle;
  (void)inputBuffers;
  (void)numBatchEltsFilled;
  (void)inputBufs;
  (void)symmetry;
  (void)outputs;
  assert(false);
}


void NeuralNet::printDevices() {
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
  std::vector<float>& outputBuffer
) {
  if(!useNHWC || useFP16)
    return false;
  ConvLayer layer(*desc);
  Eigen::TensorMap<const Tensor<const SCALAR, 4>> inTensor(
    &inputBuffer[0], desc->inChannels, nnXLen, nnYLen, batchSize);
  Tensor<SCALAR, 4> outTensor(desc->outChannels, nnXLen, nnYLen, batchSize);

  layer.apply(inTensor, outTensor, false);

  outputBuffer.resize(outTensor.size());
  memcpy(&outputBuffer[0], outTensor.data(), sizeof(SCALAR) * outTensor.size());
  return true;
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
  std::vector<float>& outputBuffer
) {
  if(!useNHWC || useFP16)
    return false;
  BatchNormLayer layer(*desc);
  Eigen::TensorMap<const Tensor<const SCALAR, 4>> inTensor(&inputBuffer[0], desc->numChannels, nnXLen, nnYLen, batchSize);
  Eigen::TensorMap<const Tensor<const SCALAR, 3>> maskTensor(&maskBuffer[0], nnXLen, nnYLen, batchSize);
  Tensor<SCALAR, 4> outTensor(desc->numChannels, nnXLen, nnYLen, batchSize);

  layer.apply(false, inTensor, outTensor, maskTensor);

  outputBuffer.resize(outTensor.size());
  memcpy(&outputBuffer[0], outTensor.data(), sizeof(SCALAR) * outTensor.size());
  return true;
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
  std::vector<float>& outputBuffer
) {
  if(!useNHWC || useFP16)
    return false;
  ResidualBlock block(*desc);
  Eigen::TensorMap<const Tensor<const SCALAR, 4>> inTensor(&inputBuffer[0], desc->preBN.numChannels, nnXLen, nnYLen, batchSize);
  Eigen::TensorMap<const Tensor<const SCALAR, 3>> maskTensor(&maskBuffer[0], nnXLen, nnYLen, batchSize);

  Tensor<SCALAR, 4> trunkTensor(desc->preBN.numChannels, nnXLen, nnYLen, batchSize);
  Tensor<SCALAR, 4> trunkScratchTensor(desc->preBN.numChannels, nnXLen, nnYLen, batchSize);
  Tensor<SCALAR, 4> midTensor(desc->finalConv.inChannels, nnXLen, nnYLen, batchSize);
  Tensor<SCALAR, 4> midScratchTensor(desc->finalConv.inChannels, nnXLen, nnYLen, batchSize);

  trunkTensor = inTensor;

  block.apply(trunkTensor,trunkScratchTensor,midTensor,midScratchTensor,maskTensor);

  outputBuffer.resize(trunkTensor.size());
  memcpy(&outputBuffer[0], trunkTensor.data(), sizeof(SCALAR) * trunkTensor.size());
  return true;
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
  std::vector<float>& outputBuffer
) {
  (void)desc;
  (void)batchSize;
  (void)nnXLen;
  (void)nnYLen;
  (void)useFP16;
  (void)useNHWC;
  (void)inputBuffer;
  (void)maskBuffer;
  (void)outputBuffer;
  return false;
}

#endif  // USE_EIGEN_BACKEND
