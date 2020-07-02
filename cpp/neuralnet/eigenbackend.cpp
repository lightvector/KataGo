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
// #define DEBUG true

template <typename T>
void printTensorShape(const string& name, const T& t) {
  auto d = t.dimensions();
  cout << name << " rank=" << d.size() << " - (";
  for (int i = 0; i < d.size(); i++) {
    cout << d[i] << ",";
  }
  cout << ")" << endl;
}

#if DEBUG
#define DSHAPE(n, x) printTensorShape(n,x)
#define DTENSOR(n, x) cout << n << x << endl
#else
#define DSHAPE(n, x)
#define DTENSOR(n, x)
#endif

// LoadedModel / ModelDesc ---------------------------------------------------------------------------------------------

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

// Helpers --------------------------------------------------------------------------------------------------------------

static void computeMaskSum(const Tensor<SCALAR, 3>& mask, float* maskSum) {
  for (int n = 0; n < mask.dimension(2); n++) {
    float s = 0.f;
    for (int h = 0; h < mask.dimension(1); h++) {
      for (int w = 0; w < mask.dimension(0); w++) {
        s += mask(w, h, n);
      }
    }
    maskSum[n] = s;
  }
}

// in NxHxWxC, bias NxC
static void addNCBiasInplace(Tensor<SCALAR, 4>& in, const Tensor<SCALAR, 2>& bias) {
  assert(in.dimension(0) == bias.dimension(0) && in.dimension(3) == bias.dimension(1));
  for (int n = 0; n < in.dimension(3); n++) {
    for (int h = 0; h < in.dimension(2); h++) {
      for (int w = 0; w < in.dimension(1); w++) {
        for (int c = 0; c < in.dimension(0); c++) {
          in(c,w,h,n) += bias(c,n);
        }
      }
    }
  }
}

static void poolRowsGPool(const Tensor<SCALAR, 4>& in, Tensor<SCALAR, 2>& out, const float* maskSum) {
  for (int n = 0; n < in.dimension(3); n++) {
    for (int c = 0; c < in.dimension(0); c++) {
      float s = 0.f;
      float m = 0.f;
      for (int h = 0; h < in.dimension(2); h++) {
        for (int w = 0; w < in.dimension(1); w++) {
          float x = in(c, w, h, n);
          s += x;
          m = max(m, x);
        }
      }
      float div = maskSum[n];
      float sqrtdiv = sqrt(div);
      float mean = s / div;
      out(c, n) = mean;
      out(c + in.dimension(0), n) = mean * (sqrtdiv - 14.f) * 0.1f;
      out(c + 2*in.dimension(0), n) = m;
    }
  }
}

// // Given input [n,w,h,c] fills output of shape [n,c] with sum over c.
// static void poolRowsSum(const Tensor<SCALAR, 4>& in, Tensor<SCALAR, 2>& out, float scaleSum) {
//   for (int n = 0; n < in.dimension(3); n++) {
//     for (int c = 0; c < in.dimension(0); c++) {
//       float s = 0.f;
//       for (int h = 0; h < in.dimension(2); h++) {
//         for (int w = 0; w < in.dimension(1); w++) {
//           float x = in(c, w, h, n);
//           s += x;
//         }
//       }
//       out(c, n) = s * scaleSum;
//     }
//   }
// }

static void poolRowsValueHead(const Tensor<SCALAR, 4>& in, Tensor<SCALAR, 2>& out, const float* maskSum) {
  for (int n = 0; n < in.dimension(3); n++) {
    for (int c = 0; c < in.dimension(0); c++) {
      float s = 0.f;
      for (int h = 0; h < in.dimension(2); h++) {
        for (int w = 0; w < in.dimension(1); w++) {
          float x = in(c, w, h, n);
          s += x;
        }
      }
      float div = maskSum[n];
      float sqrtdiv = sqrt(div);
      float mean = s / div;
      out(c, n) = mean;
      out(c + in.dimension(0), n) = mean * (sqrtdiv - 14.f) * 0.1f;
      out(c + 2*in.dimension(0), n) = mean * ((sqrtdiv - 14.0f) * (sqrtdiv - 14.0f) * 0.01f - 0.1f);
    }
  }
}

// Layers --------------------------------------------------------------------------------------------------------------

// Convolution layer with zero-padding.
struct ConvLayer {
  string name;

  Eigen::array<pair<int, int>, 4> paddings;
  Tensor<SCALAR, 4> kernel;
  int inChannels, outChannels;

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
    int paddingX = (convXSize / 2) * dilationX;
    int paddingY = (convYSize / 2) * dilationY;

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
      desc.weights.data(), convXSize, convYSize, inChannels, outChannels);
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

  vector<float> mergedScale;
  vector<float> mergedBias;

  BatchNormLayer() = delete;
  BatchNormLayer(const BatchNormLayer&) = delete;
  BatchNormLayer& operator=(const BatchNormLayer&) = delete;

  BatchNormLayer(const BatchNormLayerDesc& desc) {
    name = desc.name;
    int numChannels = desc.numChannels;
    float epsilon = desc.epsilon;

    mergedScale.resize(numChannels);
    mergedBias.resize(numChannels);
    for(int c = 0; c < numChannels; c++) {
      mergedScale[c] = desc.scale[c] / sqrt(desc.variance[c] + epsilon);
      mergedBias[c] = desc.bias[c] - mergedScale[c] * desc.mean[c];
    }
  }

  // Mask should be in 'NHW' format (no "C" channel).
  void apply(
    bool applyRelu,
    const Tensor<SCALAR, 4>& input,
    Tensor<SCALAR, 4>& output,
    const Tensor<SCALAR, 3>& mask
  ) const {

    output = Tensor<SCALAR, 4>(input.dimension(0), input.dimension(1), input.dimension(2), input.dimension(3));
    for(int c = 0; c < input.dimension(0); c++) {
      auto inC = input.chip(c, 0);
      auto x = inC * mergedScale[c] + mergedBias[c];
      auto z = Tensor<SCALAR, 3>(mask.dimension(0), mask.dimension(1), mask.dimension(2)).setZero();
      if(applyRelu)
        output.chip(c, 0) = (mask == 1.f).select(x.cwiseMax(0.f), z);
      else
        output.chip(c, 0) = (mask == 1.f).select(x, z);
    }
  }
};

//--------------------------------------------------------------

struct ActivationLayer {
  string name;

  ActivationLayer() = delete;
  ActivationLayer(const ActivationLayer&) = delete;
  ActivationLayer& operator=(const ActivationLayer&) = delete;

  ActivationLayer(const ActivationLayerDesc& desc) { name = desc.name; }

  template <int N>
  void apply(const Tensor<SCALAR, N>& input, Tensor<SCALAR, N>& output) const { output = input.cwiseMax(0.f); }
};

//--------------------------------------------------------------

struct MatMulLayer {
  string name;
  Tensor<SCALAR, 2> weights;

  MatMulLayer() = delete;
  MatMulLayer(const MatMulLayer&) = delete;
  MatMulLayer& operator=(const MatMulLayer&) = delete;

  MatMulLayer(const MatMulLayerDesc& desc)
    : name(desc.name)
  {
    weights = Tensor<SCALAR, 2>(desc.outChannels, desc.inChannels);
    memcpy(weights.data(), desc.weights.data(), sizeof(SCALAR) * weights.size());
  }

  void apply(const Tensor<SCALAR, 2>& in, Tensor<SCALAR, 2>& out) const {
    Eigen::array<Eigen::IndexPair<int>, 1> product_dims = { Eigen::IndexPair<int>(1, 0) };
    out = weights.contract(in, product_dims);
  }
};

struct MatBiasLayer {
  string name;
  std::vector<float> weights;

  MatBiasLayer() = delete;
  MatBiasLayer(const MatBiasLayer&) = delete;
  MatBiasLayer& operator=(const MatBiasLayer&) = delete;

  MatBiasLayer(const MatBiasLayerDesc& desc)
    : name(desc.name),
      weights(desc.weights) {}

  void apply(Tensor<SCALAR, 2>& mat) const {
    for(int n = 0; n < mat.dimension(1); n++) {
      for(int c = 0; c < mat.dimension(0); c++) {
        mat(c, n) += weights[c];
      }
    }
  }
};

// Blocks
// --------------------------------------------------------------------------------------------------------------

struct ResidualBlockIntf {
  virtual ~ResidualBlockIntf(){}

  virtual void apply(
    Tensor<SCALAR, 4>& trunk,
    Tensor<SCALAR, 4>& trunkScratch,
    Tensor<SCALAR, 4>& regularOut,
    Tensor<SCALAR, 4>& regularScratch,
    Tensor<SCALAR, 4>& midIn,
    Tensor<SCALAR, 4>& midScratch,
    Tensor<SCALAR, 4>& gpoolOut,
    Tensor<SCALAR, 4>& gpoolOut2,
    Tensor<SCALAR, 2>& gpoolConcat,
    Tensor<SCALAR, 2>& gpoolBias,
    const Tensor<SCALAR, 3>& mask,
    const float* maskSum
  ) const = 0;
};

struct ResidualBlock final : public ResidualBlockIntf {
  string name;
  BatchNormLayer preBN;
  ConvLayer regularConv;
  BatchNormLayer midBN;
  ConvLayer finalConv;

  ResidualBlock() = delete;
  ResidualBlock(const ResidualBlock&) = delete;
  ResidualBlock& operator=(const ResidualBlock&) = delete;

  ~ResidualBlock(){}

  ResidualBlock(const ResidualBlockDesc& desc)
    : name(desc.name),
      preBN(desc.preBN),
      regularConv(desc.regularConv),
      midBN(desc.midBN),
      finalConv(desc.finalConv) {}

  void apply(
    Tensor<SCALAR, 4>& trunk,
    Tensor<SCALAR, 4>& trunkScratch,
    Tensor<SCALAR, 4>& regularOut,
    Tensor<SCALAR, 4>& regularScratch,
    Tensor<SCALAR, 4>& midIn,
    Tensor<SCALAR, 4>& midScratch,
    Tensor<SCALAR, 4>& gpoolOut,
    Tensor<SCALAR, 4>& gpoolOut2,
    Tensor<SCALAR, 2>& gpoolConcat,
    Tensor<SCALAR, 2>& gpoolBias,
    const Tensor<SCALAR, 3>& mask,
    const float* maskSum
  ) const override {
    (void)regularOut;
    (void)regularScratch;
    (void)gpoolOut;
    (void)gpoolOut2;
    (void)gpoolConcat;
    (void)gpoolBias;
    (void)maskSum;
    const bool applyBNRelu = true;
    preBN.apply(applyBNRelu, trunk, trunkScratch, mask);
    regularConv.apply(trunkScratch, midIn, false);
    midBN.apply(applyBNRelu, midIn, midScratch, mask);
    finalConv.apply(midScratch, trunk, true);
  }
};

// // Given two tensors with shapes inA: [n, h, w, cA] and inB: [n, h, w, cB]
// // Copy them into a single tensor out: [n, h, w, cA + cB]
// Tensor<SCALAR, 4> concatTensors(const Tensor<SCALAR, 4>& a, const Tensor<SCALAR, 4>& b) {
//   assert(a.dimension(1) == b.dimension(1) && a.dimension(2) == b.dimension(2) && a.dimension(3) == b.dimension(3));
//   Tensor<SCALAR, 4> x = Tensor<SCALAR, 4>(/* C */ a.dimension(0) + b.dimension(0),
//                                           /* W */ a.dimension(1),
//                                           /* H */ a.dimension(2),
//                                           /* N */ a.dimension(3));
//   for (int n = 0; n < a.dimension(3); n++) {
//     for (int h = 0; h < a.dimension(2); h++) {
//       for (int w = 0; w < a.dimension(1); w++) {
//         int c = 0;
//         for (int ca = 0; a.dimension(0); ca++, c++) {
//           x(c,w,h,n) = a(ca,w,h,n);
//         }
//         for (int cb = 0; b.dimension(0); cb++, c++) {
//           x(c,w,h,n) = b(cb,w,h,n);
//         }
//       }
//     }
//   }
//   return x;
// }


struct GlobalPoolingResidualBlock final : public ResidualBlockIntf {
  string name;
  BatchNormLayer preBN;
  ActivationLayer preActivation;
  ConvLayer regularConv;
  ConvLayer gpoolConv;
  BatchNormLayer gpoolBN;
  ActivationLayer gpoolActivation;
  MatMulLayer gpoolToBiasMul;
  BatchNormLayer midBN;
  ActivationLayer midActivation;
  ConvLayer finalConv;

  GlobalPoolingResidualBlock() = delete;
  GlobalPoolingResidualBlock(const GlobalPoolingResidualBlock&) = delete;
  GlobalPoolingResidualBlock& operator=(const GlobalPoolingResidualBlock&) = delete;

  ~GlobalPoolingResidualBlock(){}

  GlobalPoolingResidualBlock(const GlobalPoolingResidualBlockDesc& desc)
    : name(desc.name),
      preBN(desc.preBN),
      preActivation(desc.preActivation),
      regularConv(desc.regularConv),
      gpoolConv(desc.gpoolConv),
      gpoolBN(desc.gpoolBN),
      gpoolActivation(desc.gpoolActivation),
      gpoolToBiasMul(desc.gpoolToBiasMul),
      midBN(desc.midBN),
      midActivation(desc.midActivation),
      finalConv(desc.finalConv) {}

  void apply(
    Tensor<SCALAR, 4>& trunk,
    Tensor<SCALAR, 4>& trunkScratch,
    Tensor<SCALAR, 4>& regularOut,
    Tensor<SCALAR, 4>& regularScratch,
    Tensor<SCALAR, 4>& midIn,
    Tensor<SCALAR, 4>& midScratch,
    Tensor<SCALAR, 4>& gpoolOut,
    Tensor<SCALAR, 4>& gpoolOut2,
    Tensor<SCALAR, 2>& gpoolConcat,
    Tensor<SCALAR, 2>& gpoolBias,
    const Tensor<SCALAR, 3>& mask,
    const float* maskSum
  ) const override {
    (void)midIn;
    (void)midScratch;
    const bool applyBNRelu = true;
    DTENSOR("trunk", trunk);
    DTENSOR("mask", mask);
    preBN.apply(applyBNRelu, trunk, trunkScratch, mask);
    DTENSOR("trunkScratch", trunkScratch);
    regularConv.apply(trunkScratch, regularOut, false);
    DTENSOR("regularOut", regularOut);
    gpoolConv.apply(trunkScratch, gpoolOut, false);
    DTENSOR("gpoolOut", gpoolOut);
    gpoolBN.apply(applyBNRelu, gpoolOut, gpoolOut2, mask);
    DTENSOR("gpoolOut2", gpoolOut2);
    poolRowsGPool(gpoolOut2, gpoolConcat, maskSum);
    gpoolToBiasMul.apply(gpoolConcat, gpoolBias);
    addNCBiasInplace(regularOut, gpoolBias);
    midBN.apply(applyBNRelu, regularOut, regularScratch, mask);
    finalConv.apply(regularScratch, trunk, true);
    DSHAPE("trunk", trunk);
    DSHAPE("trunkScratch", trunkScratch);
    DSHAPE("regularOut", regularOut);
    DSHAPE("gpoolOut", gpoolOut);
    DSHAPE("gpoolOut2", gpoolOut2);
    DSHAPE("gpoolConcat", gpoolConcat);
    DSHAPE("gpoolBias", gpoolBias);
    DSHAPE("mask", mask);
  }
};

struct Trunk {
  string name;
  int version;
  int numBlocks;

  ConvLayer initialConv;
  MatMulLayer initialMatMul;
  vector<pair<int, ResidualBlockIntf*>> blocks;
  BatchNormLayer trunkTipBN;
  ActivationLayer trunkTipActivation;

  Trunk() = delete;
  Trunk(const Trunk&) = delete;
  Trunk& operator=(const Trunk&) = delete;

  Trunk(const TrunkDesc& desc)
    : name(desc.name),
      version(desc.version),
      numBlocks(desc.numBlocks),
      initialConv(desc.initialConv),
      initialMatMul(desc.initialMatMul),
      trunkTipBN(desc.trunkTipBN),
      trunkTipActivation(desc.trunkTipActivation)
  {
    for (int i = 0; i < numBlocks; ++i) {
      if (desc.blocks[i].first == ORDINARY_BLOCK_KIND) {
        ResidualBlockDesc* blockDesc = (ResidualBlockDesc*)desc.blocks[i].second;
        ResidualBlockIntf* block = new ResidualBlock(*blockDesc);
        blocks.push_back(make_pair(ORDINARY_BLOCK_KIND, block));
      }
      else if (desc.blocks[i].first == DILATED_BLOCK_KIND) {
        throw StringError("Eigen backend: Dilated residual blocks are not supported right now");
      }
      else if (desc.blocks[i].first == GLOBAL_POOLING_BLOCK_KIND) {
        GlobalPoolingResidualBlockDesc* blockDesc = (GlobalPoolingResidualBlockDesc*)desc.blocks[i].second;
        GlobalPoolingResidualBlock* block = new GlobalPoolingResidualBlock(*blockDesc);
        blocks.push_back(make_pair(GLOBAL_POOLING_BLOCK_KIND, block));
      }
      else {
        ASSERT_UNREACHABLE;
      }
    }
  }

  virtual ~Trunk() {
    for (auto p : blocks) {
      delete p.second;
    }
  }

  void apply(
    const Tensor<SCALAR, 4>& input,
    const Tensor<SCALAR, 2>& inputGlobal,
    Tensor<SCALAR, 2>& inputMatMulOut,
    Tensor<SCALAR, 4>& trunk,
    Tensor<SCALAR, 4>& trunkScratch,
    Tensor<SCALAR, 4>& regularOut,
    Tensor<SCALAR, 4>& regularScratch,
    Tensor<SCALAR, 4>& midIn,
    Tensor<SCALAR, 4>& midScratch,
    Tensor<SCALAR, 4>& gpoolOut,
    Tensor<SCALAR, 4>& gpoolOut2,
    Tensor<SCALAR, 2>& gpoolConcat,
    Tensor<SCALAR, 2>& gpoolBias,
    const Tensor<SCALAR, 3>& mask,
    const float* maskSum
  ) const {

    initialConv.apply(input, trunkScratch, false);
    initialMatMul.apply(inputGlobal, inputMatMulOut);
    addNCBiasInplace(trunkScratch, inputMatMulOut);

    // apply blocks
    // Flip trunkBuf and trunkScratchBuf so that the result gets accumulated in trunkScratchBuf
    for (auto block : blocks) {
      block.second->apply(
        trunkScratch,
        trunk,
        regularOut,
        regularScratch,
        midIn,
        midScratch,
        gpoolOut,
        gpoolOut2,
        gpoolConcat,
        gpoolBias,
        mask,
        maskSum
      );
    }

    // And now with the final BN port it from trunkScratchBuf to trunkBuf.
    const bool applyBNRelu = true;
    trunkTipBN.apply(applyBNRelu, trunkScratch, trunk, mask);
  }
};

struct PolicyHead {
  string name;
  int version;

  ConvLayer p1Conv;
  ConvLayer g1Conv;
  BatchNormLayer g1BN;
  ActivationLayer g1Activation;
  MatMulLayer gpoolToBiasMul;
  BatchNormLayer p1BN;
  ActivationLayer p1Activation;
  ConvLayer p2Conv;
  MatMulLayer gpoolToPassMul;

  PolicyHead() = delete;
  PolicyHead(const PolicyHead&) = delete;
  PolicyHead& operator=(const PolicyHead&) = delete;

  PolicyHead(const PolicyHeadDesc& desc)
    : name(desc.name),
      version(desc.version),
      p1Conv(desc.p1Conv),
      g1Conv(desc.g1Conv),
      g1BN(desc.g1BN),
      g1Activation(desc.g1Activation),
      gpoolToBiasMul(desc.gpoolToBiasMul),
      p1BN(desc.p1BN),
      p1Activation(desc.p1Activation),
      p2Conv(desc.p2Conv),
      gpoolToPassMul(desc.gpoolToPassMul) {}

  void apply(
    const Tensor<SCALAR, 4>& trunk,
    Tensor<SCALAR, 4>& p1Out,
    Tensor<SCALAR, 4>& p1Out2,
    Tensor<SCALAR, 4>& g1Out,
    Tensor<SCALAR, 4>& g1Out2,
    Tensor<SCALAR, 2>& g1Concat,
    Tensor<SCALAR, 2>& g1Bias,
    Tensor<SCALAR, 4>& p2Out,
    Tensor<SCALAR, 2>& g1Pass,
    const Tensor<SCALAR, 3>& mask,
    const float* maskSum
  ) const {
    const bool applyBNRelu = true;
    p1Conv.apply(trunk, p1Out, false);
    g1Conv.apply(trunk, g1Out, false);
    g1BN.apply(applyBNRelu, g1Out, g1Out2, mask);
    poolRowsGPool(g1Out2, g1Concat, maskSum);
    gpoolToBiasMul.apply(g1Concat, g1Bias);
    addNCBiasInplace(p1Out, g1Bias);
    p1BN.apply(true, p1Out, p1Out2, mask);
    p2Conv.apply(p1Out2, p2Out, false);
    gpoolToPassMul.apply(g1Concat, g1Pass);
  }
};

struct ValueHead {
  string name;
  int version;

  ConvLayer v1Conv;
  BatchNormLayer v1BN;
  ActivationLayer v1Activation;
  MatMulLayer v2Mul;
  MatBiasLayer v2Bias;
  ActivationLayer v2Activation;
  MatMulLayer v3Mul;
  MatBiasLayer v3Bias;
  MatMulLayer sv3Mul;
  MatBiasLayer sv3Bias;
  ConvLayer vOwnershipConv;

  ValueHead() = delete;
  ValueHead(const ValueHead&) = delete;
  ValueHead& operator=(const ValueHead&) = delete;

  ValueHead(const ValueHeadDesc& desc)
    : name(desc.name),
      version(desc.version),
      v1Conv(desc.v1Conv),
      v1BN(desc.v1BN),
      v1Activation(desc.v1Activation),
      v2Mul(desc.v2Mul),
      v2Bias(desc.v2Bias),
      v2Activation(desc.v2Activation),
      v3Mul(desc.v3Mul),
      v3Bias(desc.v3Bias),
      sv3Mul(desc.sv3Mul),
      sv3Bias(desc.sv3Bias),
      vOwnershipConv(desc.vOwnershipConv) {}

  void apply(
    const Tensor<SCALAR, 4>& trunk,
    Tensor<SCALAR, 4>& v1Out,
    Tensor<SCALAR, 4>& v1Out2,
    Tensor<SCALAR, 2>& v1Mean,
    Tensor<SCALAR, 2>& v2Out,
    Tensor<SCALAR, 2>& value,
    Tensor<SCALAR, 2>& scoreValue,
    Tensor<SCALAR, 4>& ownership,
    const Tensor<SCALAR, 4>& mask,
    const float* maskSum
  ) const {
    bool applyBNRelu = true;
    v1Conv.apply(trunk, v1Out, false);
    v1BN.apply(applyBNRelu, v1Out, v1Out2, mask);
    poolRowsValueHead(v1Out2, v1Mean, maskSum);
    v2Mul.apply(v1Mean, v2Out);
    v2Bias.apply(v2Out);
    v2Activation.apply(v2Out, v2Out);
    v3Mul.apply(v2Out, value);
    v3Bias.apply(value);

    sv3Mul.apply(v2Out, scoreValue);
    sv3Bias.apply(scoreValue);

    vOwnershipConv.apply(v1Out2, ownership, false);
  }
};


// Model and Buffer I/O ------------------------------------------------------------------------------------------------

struct Model {
  string name;
  int version;
  int numInputChannels;
  int numInputGlobalChannels;
  int numValueChannels;
  int numScoreValueChannels;
  int numOwnershipChannels;

  Trunk trunk;
  PolicyHead policyHead;
  ValueHead valueHead;

  Model() = delete;
  Model(const Model&) = delete;
  Model& operator=(const Model&) = delete;

  Model(const ModelDesc& desc)
    : name(desc.name), version(desc.version), numInputChannels(desc.numInputChannels),
      numInputGlobalChannels(desc.numInputGlobalChannels),
      numValueChannels(desc.numValueChannels),
      numScoreValueChannels(desc.numScoreValueChannels),
      numOwnershipChannels(desc.numOwnershipChannels),
      trunk(desc.trunk),
      policyHead(desc.policyHead),
      valueHead(desc.valueHead) {}

  void apply(void* input,
             void* inputGlobal) const {
    // TODO: fill mask
    // TODO: apply Trunk
    // TODO: apply PolicyHead
    // TODO: apply ValueHead
  }
};

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
    inputBuffer.data(), desc->inChannels, nnXLen, nnYLen, batchSize);
  Tensor<SCALAR, 4> outTensor(desc->outChannels, nnXLen, nnYLen, batchSize);

  layer.apply(inTensor, outTensor, false);

  outputBuffer.resize(outTensor.size());
  memcpy(outputBuffer.data(), outTensor.data(), sizeof(SCALAR) * outTensor.size());
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
  Eigen::TensorMap<const Tensor<const SCALAR, 4>> inTensor(inputBuffer.data(), desc->numChannels, nnXLen, nnYLen, batchSize);
  Eigen::TensorMap<const Tensor<const SCALAR, 3>> mask(maskBuffer.data(), nnXLen, nnYLen, batchSize);
  Tensor<SCALAR, 4> outTensor(desc->numChannels, nnXLen, nnYLen, batchSize);

  layer.apply(false, inTensor, outTensor, mask);

  outputBuffer.resize(outTensor.size());
  memcpy(outputBuffer.data(), outTensor.data(), sizeof(SCALAR) * outTensor.size());
  return true;
}

// CR lpuchallafiore: test evaluate activation layer.
// CR lpuchallafiore: test evaluate matmul layer.
// CR lpuchallafiore: test evaluate matbias layer.

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
  Eigen::TensorMap<const Tensor<const SCALAR, 4>> inTensor(inputBuffer.data(), desc->preBN.numChannels, nnXLen, nnYLen, batchSize);
  Eigen::TensorMap<const Tensor<const SCALAR, 3>> mask(maskBuffer.data(), nnXLen, nnYLen, batchSize);

  Tensor<SCALAR, 4> trunk(desc->preBN.numChannels, nnXLen, nnYLen, batchSize);
  Tensor<SCALAR, 4> trunkScratch(desc->preBN.numChannels, nnXLen, nnYLen, batchSize);
  Tensor<SCALAR, 4> regularOut;
  Tensor<SCALAR, 4> regularScratch;
  Tensor<SCALAR, 4> mid(desc->finalConv.inChannels, nnXLen, nnYLen, batchSize);
  Tensor<SCALAR, 4> midScratch(desc->finalConv.inChannels, nnXLen, nnYLen, batchSize);
  Tensor<SCALAR, 4> gpoolOut;
  Tensor<SCALAR, 4> gpoolOut2;
  Tensor<SCALAR, 2> gpoolConcat;
  Tensor<SCALAR, 2> gpoolBias;
  float* maskSum = NULL;

  trunk = inTensor;

  block.apply(
    trunk,
    trunkScratch,
    regularOut,
    regularScratch,
    mid,
    midScratch,
    gpoolOut,
    gpoolOut2,
    gpoolConcat,
    gpoolBias,
    mask,
    maskSum
  );

  outputBuffer.resize(trunk.size());
  memcpy(outputBuffer.data(), trunk.data(), sizeof(SCALAR) * trunk.size());
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
  std::vector<float>& outputBuffer) {
  if(!useNHWC || useFP16)
    return false;

  GlobalPoolingResidualBlock block(*desc);

  Eigen::TensorMap<const Tensor<const SCALAR, 4>> inTensor(inputBuffer.data(), desc->preBN.numChannels, nnXLen, nnYLen, batchSize);
  Eigen::TensorMap<const Tensor<const SCALAR, 3>> mask(maskBuffer.data(), nnXLen, nnYLen, batchSize);

  Tensor<SCALAR, 4> trunk(desc->preBN.numChannels, nnXLen, nnYLen, batchSize);
  Tensor<SCALAR, 4> trunkScratch(desc->preBN.numChannels, nnXLen, nnYLen, batchSize);
  Tensor<SCALAR, 4> regularOut(desc->finalConv.inChannels, nnXLen, nnYLen, batchSize);
  Tensor<SCALAR, 4> regularScratch(desc->finalConv.inChannels, nnXLen, nnYLen, batchSize);
  Tensor<SCALAR, 4> mid;
  Tensor<SCALAR, 4> midScratch;
  Tensor<SCALAR, 4> gpoolOut(desc->gpoolConv.outChannels, nnXLen, nnYLen, batchSize);
  Tensor<SCALAR, 4> gpoolOut2(desc->gpoolConv.outChannels, nnXLen, nnYLen, batchSize);
  Tensor<SCALAR, 2> gpoolConcat(desc->gpoolConv.outChannels*3, batchSize);
  Tensor<SCALAR, 2> gpoolBias(desc->gpoolToBiasMul.outChannels, batchSize);

  std::vector<float> maskSum(batchSize);
  computeMaskSum(mask,maskSum.data());

  trunk = inTensor;

  block.apply(
    trunk,
    trunkScratch,
    regularOut,
    regularScratch,
    mid,
    midScratch,
    gpoolOut,
    gpoolOut2,
    gpoolConcat,
    gpoolBias,
    mask,
    maskSum.data()
  );

  outputBuffer.resize(trunk.size());
  memcpy(outputBuffer.data(), trunk.data(), sizeof(SCALAR) * trunk.size());

  return true;
}

#endif  // USE_EIGEN_BACKEND

// CR lpuchallafiore: test evaluate Trunk
// CR lpuchallafiore: test evaluate Policy Head
// CR lpuchallafiore: test evaluate Value Head
