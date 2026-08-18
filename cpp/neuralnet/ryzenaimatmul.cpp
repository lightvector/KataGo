#include "../neuralnet/ryzenaimatmul.h"

#include <algorithm>
#include <chrono>
#include <map>
#include <sstream>

#include "../core/global.h"
#include "../neuralnet/desc.h"
#include "../neuralnet/ryzenaikernel.h"

using namespace std;

namespace RyzenAIMatMul {

namespace {

  // One dense layer's device-resident weights, plus the engine that serves its
  // reduction dim. Engines are shared between every layer with the same K.
  struct Layer {
    RyzenAIKernel::Engine* engine = nullptr;  // not owned
    RyzenAIKernel::Weights* weights = nullptr;
    int K = 0;         // the engine's K, >= inChannels
    int paddedN = 0;   // >= outChannels
    bool usable = false;
    string why;        // set when !usable, for the report
  };

  // Everything tryAttention needs to route one attention block shape's QK^T
  // and P*V GEMMs to the NPU. Unlike a Layer, the B operands (the K and V
  // activations) change every evaluation, so the weights BOs are rewritten in
  // place per dispatch via RyzenAIKernel::rewriteWeights. The engines are the
  // same shared ones the dense layers use.
  //
  // All heads of one batch element go into ONE GEMM per direction, because a
  // dispatch costs ~1 ms regardless of size while the multiply-accumulates are
  // nearly free: A carries the per-head data side by side along the reduction
  // dim and B is block-diagonal, so C comes out with the per-head results side
  // by side along N. Per (batch, block) that is 2 dispatches instead of
  // 2*numHeads.
  struct AttnState {
    // The dims this state was built for; a block with different dims declines.
    int S = 0;
    int numHeads = 0;
    int numKVHeads = 0;
    int qHeadDim = 0;
    int vHeadDim = 0;

    RyzenAIKernel::Engine* engineQK = nullptr;    // not owned (shared)
    RyzenAIKernel::Engine* enginePV = nullptr;    // not owned (shared)
    RyzenAIKernel::Weights* weightsQK = nullptr;  // owned
    RyzenAIKernel::Weights* weightsPV = nullptr;  // owned
    int KQK = 0;        // engine K serving QK^T (>= numHeads*qHeadDim)
    int KPV = 0;        // engine K serving P*V (>= numHeads*S)
    int paddedMQK = 0;  // padM(S) on the QK^T engine
    int paddedMPV = 0;  // padM(S) on the P*V engine
    int paddedNQK = 0;  // padN(numHeads*S): head h's scores at columns [h*S, (h+1)*S)
    int paddedNPV = 0;  // padN(numHeads*vHeadDim)

    // Host scratch, dedicated to attention so that the padding regions can be
    // zeroed once at (re)size time; per dispatch only the real regions are
    // rewritten. (The dense path's hostA/hostC cannot offer that invariant --
    // every dense layer rewrites them to its own shape.)
    vector<uint16_t> hostAQK;  // paddedMQK x KQK; row qi is qBuf[n][qi][:] outright
    vector<uint16_t> hostAPV;  // paddedMPV x KPV; head h's P rows at columns [h*S, (h+1)*S)
    vector<uint16_t> hostBQK;  // KQK x paddedNQK, block-diagonal transposed K
    vector<uint16_t> hostBPV;  // KPV x paddedNPV, block-diagonal V
    vector<float> hostC;       // max(paddedMQK*paddedNQK, paddedMPV*paddedNPV)
    vector<float> scoreRow;    // one softmax working row, S floats

    // The softmax between the two GEMMs, when an op compiled for exactly this
    // (rows, width) = (numHeads*S, S) padded to (64, 32) exists. Null keeps the
    // CPU softmax below. Rows are head-major: row h*S+qi holds head h's scores
    // for query qi, so the scatter into hostAPV is a pure copy.
    RyzenAIKernel::Op* softmaxOp = nullptr;  // owned
    int smRows = 0;    // the op's compiled rows (>= numHeads*S)
    int smWidth = 0;   // the op's compiled width (>= S)
    vector<uint16_t> hostSmIn;   // smRows x smWidth, bfloat16
    vector<uint16_t> hostSmOut;  // same

    // The fused whole-attention op (QK^T + softmax + P*V in one dispatch),
    // when one compiled for exactly (numHeads, S) exists. Buffers are the
    // pre-tiled host packing: attnQ [heads][48][256] bf16 (8x32 chunks in
    // A-tile order, pre-scaled), attnKV [kvHeads][2][12288] bf16 (B-tile
    // order, K then V), attnC [heads][48][256] f32 (C-tile order).
    RyzenAIKernel::Op* attnOp = nullptr;  // owned
    vector<uint16_t> attnQ;
    vector<uint16_t> attnKV;
    vector<float> attnC;

    bool usable = false;
    string why;  // set when !usable, for the report
  };

}  // namespace

namespace {


}  // namespace

struct Accel {
  Options options;
  RyzenAIKernel::Dtype dtype = RyzenAIKernel::Dtype::Auto;

  // K -> engine. Every dense layer with that reduction dim shares one.
  map<int, RyzenAIKernel::Engine*> engines;
  map<int, RyzenAIKernel::EngineInfo> infos;

  // Which K values have artifacts, so a layer can be rejected without paying
  // for a load attempt.
  vector<int> availableK;

  // Keyed by the desc's address, which is stable for the model's lifetime.
  map<const void*, Layer> layers;

  // Fused projections (several layers sharing one input, weights concatenated
  // along N into one uploaded B), keyed by the first desc's address. A desc
  // used as a fusion key must never also be used standalone -- true for
  // q/k/vProj and linear1/linearGate, which only ever appear together.
  map<const void*, Layer> fusedLayers;

  // SwiGLU-epilogue engines (a separate xclbin per K, so a separate hardware
  // context from the plain GEMM engines) and the FFN layers using them,
  // keyed by the linear1 desc's address.
  map<int, RyzenAIKernel::Engine*> swigluEngines;
  map<int, RyzenAIKernel::EngineInfo> swigluInfos;
  vector<int> availableSwigluK;
  map<const void*, Layer> swigluLayers;
  long long numSwiglu = 0;
  long long numSwigluFallback = 0;

  // Scratch, grown on demand: A in bfloat16 padded to (paddedM x K), C in
  // float32 padded to (paddedM x paddedN).
  vector<uint16_t> hostA;
  vector<float> hostC;

  // The convolution input, converted to bfloat16 once per layer. A 3x3 window
  // reads every input point from up to nine output positions, so converting
  // inside the im2col gather did the same float->bf16 arithmetic nine times
  // over; with this the gather is a memcpy per tap.
  vector<uint16_t> hostConvIn;

  long long numAccelerated = 0;
  long long numFallback = 0;

  // Attention (QK^T and P*V) routing state and accounting.
  AttnState attn;
  long long numAttn = 0;
  long long numAttnFallback = 0;
  double secsAttnPack = 0.0;      // A and B host-side builds
  double secsAttnUploadB = 0.0;   // rewriteWeights (memcpy + sync)
  double secsAttnDispatch = 0.0;  // runGemm
  double secsAttnUnpack = 0.0;    // C -> attnOut
  double secsAttnSoftmax = 0.0;        // on-CPU softmax between the two GEMMs
  double secsAttnSoftmaxNpu = 0.0;     // NPU softmax dispatch
  double secsAttnSoftmaxHost = 0.0;    // gather/scatter around the NPU softmax

  // Where the wall clock goes inside tryMatmul, so that the host-side packing
  // can be told apart from the NPU's own time.
  double secsPackA = 0.0;
  double secsGather = 0.0;   // im2col, for convolutions with taps
  double secsDispatch = 0.0;
  double secsUnpackC = 0.0;

  // BatchNorm+Mish fused op state, keyed by (channels, rows-per-dispatch).
  //
  // The row count is baked into each artifact, and which one is cheapest
  // depends on how many rows the call actually has. Measured on b40c768: a
  // dispatch costs ~0.73 ms fixed plus ~0.85 us per row, so a batch of one
  // board (361 rows) wants the 384-row op (~1.06 ms) while a batch of seven
  // (2635 rows) wants the 3072-row one (~3.3 ms in a single dispatch instead
  // of seven 384-row dispatches at ~7.4 ms). Picking per call rather than
  // latching one height is the whole point - latching is what silently cost
  // b40c768 its BN acceleration under batching in the first place.
  // constexpr, so no out-of-class definition is needed (C++17 makes it inline).
  static constexpr int kBnMishHeights[3] = {384, 1536, 3072};

  struct BnMishState {
    RyzenAIKernel::Op* op = nullptr;  // owned
    // Why this width is not accelerated, for the report. A layer that simply
    // is not Mish is not recorded: that is routing, not a failure. A missing
    // artifact is, because it costs real time and is otherwise invisible -
    // the CPU path produces identical numbers, just slowly.
    string why;
    int rowsPad = 0;
    int width = 0;
    bool tried = false;
    vector<uint16_t> hostX;   // rowsPad x width bf16
    vector<uint16_t> hostSB;  // 8 x 2*width bf16 ([scale|bias] per core)
    vector<uint16_t> hostY;   // rowsPad x width bf16
  };
  std::map<std::pair<int,int>, BnMishState> bnmByShape;  // (channels, rows)
  long long numBnMish = 0;
  double secsBnMish = 0.0;

  // One-time weight preparation: fp32 -> bfloat16, padded, uploaded. This is
  // what a persistent cache would replace, so it is measured separately.
  double secsPrepareConvert = 0.0;  // conversion + padding on the host
  double secsPrepareUpload = 0.0;   // memcpy into the BO and sync
  long long prepareBytes = 0;
};

namespace {
  inline double nowSecs() {
    return std::chrono::duration<double>(
      std::chrono::steady_clock::now().time_since_epoch()).count();
  }
}

namespace {

  // Smallest artifact K that can hold this layer's reduction dim, or the
  // single forced one when the caller has asked for a one-context model.
  bool pickK(const Accel* accel, int inChannels, int& kOut) {
    // forceK is a preference, not a requirement: it collapses every layer it
    // can onto one context, and anything needing a larger reduction dim (the
    // block-diagonal P*V, whose K is numHeads*S) still gets its own rather than
    // dropping to the CPU. One extra context beats losing the whole operator.
    if(accel->options.forceK > 0 && accel->options.forceK >= inChannels) {
      kOut = accel->options.forceK;
      return true;
    }
    for(size_t i = 0; i < accel->availableK.size(); i++) {
      if(accel->availableK[i] >= inChannels) {
        kOut = accel->availableK[i];
        return true;
      }
    }
    return false;
  }

  RyzenAIKernel::Engine* engineForK(Accel* accel, int K, RyzenAIKernel::EngineInfo& infoOut) {
    auto it = accel->engines.find(K);
    if(it != accel->engines.end()) {
      infoOut = accel->infos[K];
      return it->second;
    }
    RyzenAIKernel::EngineInfo info;
    string err;
    RyzenAIKernel::Engine* engine = RyzenAIKernel::loadEngine(
      accel->options.artifactDir, accel->options.deviceIdx, accel->dtype, K,
      accel->options.maxCols, info, err);
    if(engine == nullptr)
      return nullptr;
    accel->engines[info.K] = engine;
    accel->infos[info.K] = info;
    infoOut = info;
    return engine;
  }

  // How a layer's source weights are laid out, and therefore how they have to
  // be shuffled into the K x N row-major B the GEMM wants.
  enum class WeightForm {
    InByOut,   // MatMulLayerDesc: w[ic*outC + oc]
    OutByIn,   // 1x1 ConvLayerDesc: w[oc*inC + ic]
    ConvTaps,  // ConvLayerDesc with taps: w[(oc*inC + ic)*ky*kx + dy*kx + dx]
  };

  // First use of a layer: pick an engine, then pad and upload its weights.
  // numTaps is convY*convX (1 for a dense layer), so the reduction dim is
  // numTaps*inChannels.
  Layer& prepareLayer(
    Accel* accel, const void* key, const float* w, int inChannels, int outChannels,
    WeightForm form, int numTaps = 1) {
    Layer& layer = accel->layers[key];
    if(layer.engine != nullptr || layer.why.size() > 0)
      return layer;

    const int reduceDim = numTaps * inChannels;
    int K = 0;
    if(!pickK(accel, reduceDim, K)) {
      layer.why = "no artifact with K >= " + Global::intToString(reduceDim);
      return layer;
    }

    RyzenAIKernel::EngineInfo info;
    RyzenAIKernel::Engine* engine = engineForK(accel, K, info);
    if(engine == nullptr) {
      layer.why = "engine for K=" + Global::intToString(K) + " would not load";
      return layer;
    }

    const int paddedN = RyzenAIKernel::padN(info, outChannels);

    // B must end up K x N row-major, zero-padded on both axes. Row k of B pairs
    // with column k of A, so for a convolution the tap index has to lead: the
    // gather below writes tap t's inChannels values at column t*inChannels.
    const double tConvert = nowSecs();
    vector<uint16_t> b((size_t)info.K * (size_t)paddedN, RyzenAIKernel::floatToBf16(0.0f));
    for(int t = 0; t < numTaps; t++) {
      for(int ic = 0; ic < inChannels; ic++) {
        uint16_t* dst = b.data() + ((size_t)t * inChannels + ic) * paddedN;
        for(int oc = 0; oc < outChannels; oc++) {
          float v;
          switch(form) {
            case WeightForm::InByOut: v = w[(size_t)ic * outChannels + oc]; break;
            case WeightForm::OutByIn: v = w[(size_t)oc * inChannels + ic]; break;
            default: v = w[((size_t)oc * inChannels + ic) * numTaps + t]; break;
          }
          dst[oc] = RyzenAIKernel::floatToBf16(v);
        }
      }
    }

    accel->secsPrepareConvert += nowSecs() - tConvert;
    accel->prepareBytes += (long long)b.size() * 2;

    const double tUpload = nowSecs();
    string err;
    RyzenAIKernel::Weights* weights = RyzenAIKernel::uploadWeights(engine, paddedN, b.data(), err);
    accel->secsPrepareUpload += nowSecs() - tUpload;
    if(weights == nullptr) {
      layer.why = err;
      return layer;
    }

    layer.engine = engine;
    layer.weights = weights;
    layer.K = info.K;
    layer.paddedN = paddedN;
    layer.usable = true;
    return layer;
  }

}  // namespace

Accel* create(const Options& options, string& err) {
  err.clear();

  RyzenAIKernel::Dtype dtype = RyzenAIKernel::Dtype::Auto;
  if(!RyzenAIKernel::parseDtype(options.dtype, dtype)) {
    err = "unrecognized dtype '" + options.dtype + "'";
    return nullptr;
  }

  vector<int> ks = RyzenAIKernel::listGemmK(options.artifactDir);
  if(ks.empty()) {
    err = "no NPU artifacts under " + options.artifactDir;
    return nullptr;
  }

  Accel* accel = new Accel();
  accel->options = options;
  accel->dtype = dtype;
  accel->availableK = ks;  // listGemmK returns them sorted
  accel->availableSwigluK = RyzenAIKernel::listSwigluK(options.artifactDir);  // sorted too
  return accel;
}

void free(Accel* accel) {
  if(accel == nullptr)
    return;
  for(auto& entry : accel->layers)
    RyzenAIKernel::freeWeights(entry.second.weights);
  for(auto& entry : accel->fusedLayers)
    RyzenAIKernel::freeWeights(entry.second.weights);
  for(auto& entry : accel->swigluLayers)
    RyzenAIKernel::freeWeights(entry.second.weights);
  RyzenAIKernel::freeWeights(accel->attn.weightsQK);
  RyzenAIKernel::freeWeights(accel->attn.weightsPV);
  RyzenAIKernel::freeOp(accel->attn.softmaxOp);
  RyzenAIKernel::freeOp(accel->attn.attnOp);
  for(auto& e : accel->bnmByShape)
    RyzenAIKernel::freeOp(e.second.op);
  for(auto& entry : accel->engines)
    RyzenAIKernel::freeEngine(entry.second);
  for(auto& entry : accel->swigluEngines)
    RyzenAIKernel::freeEngine(entry.second);
  delete accel;
}

namespace {

// `in` may be null, meaning the caller has already written the padded bfloat16
// A into accel->hostA itself (the im2col path does).
// Packs A (unless in == nullptr, meaning the caller already wrote the padded
// bfloat16 A into accel->hostA, as the im2col path does) and dispatches,
// leaving the padded float32 result in accel->hostC. On a dispatch failure the
// layer is dropped to the CPU permanently and false is returned, already
// accounted; the caller must then run its own implementation.
bool dispatchPacked(
  Accel* accel, Layer& layer, const float* in, int inC, int numRows) {
  const RyzenAIKernel::EngineInfo& info = RyzenAIKernel::engineInfo(layer.engine);
  const int paddedM = RyzenAIKernel::padM(info, numRows);
  const int K = layer.K;
  const int paddedN = layer.paddedN;

  accel->hostA.resize((size_t)paddedM * (size_t)K);
  accel->hostC.resize((size_t)paddedM * (size_t)paddedN);

  // A: convert to bfloat16, zero-filling the columns past inChannels and the
  // rows past numRows. Those contribute nothing to the real outputs.
  const double tPack = nowSecs();
  if(in != nullptr) {
    const uint16_t zero = RyzenAIKernel::floatToBf16(0.0f);
    for(int r = 0; r < numRows; r++) {
      const float* src = in + (size_t)r * inC;
      uint16_t* dst = accel->hostA.data() + (size_t)r * K;
      RyzenAIKernel::floatToBf16Bulk(src, dst, (size_t)inC);
      for(int c = inC; c < K; c++)
        dst[c] = zero;
    }
    if(paddedM > numRows)
      std::fill(
        accel->hostA.begin() + (size_t)numRows * K,
        accel->hostA.begin() + (size_t)paddedM * K, zero);
  }

  const double tDispatch = nowSecs();
  accel->secsPackA += tDispatch - tPack;
  try {
    RyzenAIKernel::runGemm(
      layer.engine, layer.weights, paddedM, paddedN, accel->hostA.data(), accel->hostC.data());
  }
  catch(const std::exception&) {
    layer.usable = false;
    layer.why = "dispatch failed at run time";
    accel->numFallback++;
    return false;
  }
  accel->secsDispatch += nowSecs() - tDispatch;
  return true;
}

bool runLayer(
  Accel* accel, Layer& layer, float* out, const float* in, int inC, int outC, int numRows,
  bool accumulate) {
  const int paddedN = layer.paddedN;
  if(!dispatchPacked(accel, layer, in, inC, numRows))
    return false;

  const double tUnpack = nowSecs();
  // C: take the real rows and columns out of the padded result.
  for(int r = 0; r < numRows; r++) {
    const float* src = accel->hostC.data() + (size_t)r * paddedN;
    float* dst = out + (size_t)r * outC;
    if(accumulate) {
      for(int c = 0; c < outC; c++)
        dst[c] += src[c];
    }
    else {
      std::memcpy(dst, src, (size_t)outC * sizeof(float));
    }
  }

  accel->secsUnpackC += nowSecs() - tUnpack;
  accel->numAccelerated++;
  return true;
}

}  // namespace

bool tryMatmul(
  Accel* accel, float* out, const float* in, const MatMulLayerDesc& desc, int numRows) {
  if(accel == nullptr || numRows < accel->options.minRows) {
    if(accel != nullptr)
      accel->numFallback++;
    return false;
  }
  Layer& layer = prepareLayer(
    accel, &desc, desc.weights.data(), desc.inChannels, desc.outChannels, WeightForm::InByOut);
  if(!layer.usable) {
    accel->numFallback++;
    return false;
  }
  return runLayer(accel, layer, out, in, desc.inChannels, desc.outChannels, numRows, false);
}

namespace {

  // First use of a fused projection: concatenates the descs' weights along N
  // into one uploaded B. All descs must share the same inChannels (checked by
  // the caller). Column block j holds descs[j]'s outChannels columns, in
  // order, so one dispatch produces every output side by side.
  Layer& prepareFusedLayer(
    Accel* accel, const MatMulLayerDesc* const* descs, int numDescs) {
    Layer& layer = accel->fusedLayers[descs[0]];
    if(layer.engine != nullptr || layer.why.size() > 0)
      return layer;

    const int inC = descs[0]->inChannels;
    int K = 0;
    if(!pickK(accel, inC, K)) {
      layer.why = "no artifact with K >= " + Global::intToString(inC);
      return layer;
    }

    RyzenAIKernel::EngineInfo info;
    RyzenAIKernel::Engine* engine = engineForK(accel, K, info);
    if(engine == nullptr) {
      layer.why = "engine for K=" + Global::intToString(K) + " would not load";
      return layer;
    }

    int totalOutC = 0;
    for(int j = 0; j < numDescs; j++)
      totalOutC += descs[j]->outChannels;
    const int paddedN = RyzenAIKernel::padN(info, totalOutC);

    const double tConvert = nowSecs();
    vector<uint16_t> b((size_t)info.K * (size_t)paddedN, RyzenAIKernel::floatToBf16(0.0f));
    int off = 0;
    for(int j = 0; j < numDescs; j++) {
      const MatMulLayerDesc* d = descs[j];
      const float* w = d->weights.data();
      for(int ic = 0; ic < inC; ic++) {
        uint16_t* dst = b.data() + (size_t)ic * paddedN + off;
        const float* src = w + (size_t)ic * d->outChannels;
        for(int oc = 0; oc < d->outChannels; oc++)
          dst[oc] = RyzenAIKernel::floatToBf16(src[oc]);
      }
      off += d->outChannels;
    }
    accel->secsPrepareConvert += nowSecs() - tConvert;
    accel->prepareBytes += (long long)b.size() * 2;

    const double tUpload = nowSecs();
    string err;
    RyzenAIKernel::Weights* weights = RyzenAIKernel::uploadWeights(engine, paddedN, b.data(), err);
    accel->secsPrepareUpload += nowSecs() - tUpload;
    if(weights == nullptr) {
      layer.why = err;
      return layer;
    }

    layer.engine = engine;
    layer.weights = weights;
    layer.K = info.K;
    layer.paddedN = paddedN;
    layer.usable = true;
    return layer;
  }

}  // namespace

bool tryMatmulMulti(
  Accel* accel, float* const* outs, const float* in,
  const MatMulLayerDesc* const* descs, int numDescs, int numRows) {
  if(accel == nullptr || numDescs < 2 || numRows < accel->options.minRows) {
    if(accel != nullptr)
      accel->numFallback += numDescs;
    return false;
  }
  const int inC = descs[0]->inChannels;
  for(int j = 1; j < numDescs; j++)
    if(descs[j]->inChannels != inC)
      return false;  // not fusible; the caller runs each on its own path

  Layer& layer = prepareFusedLayer(accel, descs, numDescs);
  if(!layer.usable) {
    accel->numFallback += numDescs;
    return false;
  }
  if(!dispatchPacked(accel, layer, in, inC, numRows))
    return false;

  const double tUnpack = nowSecs();
  const int paddedN = layer.paddedN;
  int off = 0;
  for(int j = 0; j < numDescs; j++) {
    const int outC = descs[j]->outChannels;
    float* out = outs[j];
    for(int r = 0; r < numRows; r++)
      std::memcpy(
        out + (size_t)r * outC, accel->hostC.data() + (size_t)r * paddedN + off,
        (size_t)outC * sizeof(float));
    off += outC;
  }

  accel->secsUnpackC += nowSecs() - tUnpack;
  accel->numAccelerated += numDescs;
  return true;
}

namespace {

  // The swiglu-epilogue engines sit on their own hardware contexts, one per K.
  RyzenAIKernel::Engine* swigluEngineForK(
    Accel* accel, int K, RyzenAIKernel::EngineInfo& infoOut) {
    auto it = accel->swigluEngines.find(K);
    if(it != accel->swigluEngines.end()) {
      infoOut = accel->swigluInfos[K];
      return it->second;
    }
    RyzenAIKernel::EngineInfo info;
    string err;
    RyzenAIKernel::Engine* engine = RyzenAIKernel::loadEngineSwiglu(
      accel->options.artifactDir, accel->options.deviceIdx, accel->dtype, K,
      accel->options.maxCols, info, err);
    if(engine == nullptr)
      return nullptr;
    accel->swigluEngines[info.K] = engine;
    accel->swigluInfos[info.K] = info;
    infoOut = info;
    return engine;
  }

  // Which reduction dim the swiglu path can serve this layer on. forceK wins
  // only when a swiglu artifact exists at exactly that K -- substituting a
  // different one would add yet another hardware context, which the forced-K
  // policy exists to eliminate. Without forceK, take the smallest available
  // swiglu K that fits (zero-padding the reduction is nearly free).
  bool pickSwigluK(const Accel* accel, int inC, int& kOut) {
    const vector<int>& ks = accel->availableSwigluK;
    if(accel->options.forceK > 0 && accel->options.forceK >= inC) {
      if(std::binary_search(ks.begin(), ks.end(), accel->options.forceK)) {
        kOut = accel->options.forceK;
        return true;
      }
      return false;
    }
    for(size_t i = 0; i < ks.size(); i++) {
      if(ks[i] >= inC) {
        kOut = ks[i];
        return true;
      }
    }
    return false;
  }

  // First use of an FFN's linear1/linearGate pair: pad and upload the two
  // weight matrices as one B whose columns interleave in groups of 8 --
  // [linear1 ch 0-7, gate ch 0-7, linear1 ch 8-15, ...] -- so that in every
  // core's C tile the even 8-column sub-tiles hold linear1 outputs and the odd
  // ones the matching gates, which is the pairing the on-chip epilogue
  // assumes (see python/ryzenai_kernels/kernels/mm_swiglu_epilogue.cc).
  Layer& prepareSwigluLayer(
    Accel* accel, const MatMulLayerDesc& linear1, const MatMulLayerDesc& linearGate) {
    Layer& layer = accel->swigluLayers[&linear1];
    if(layer.engine != nullptr || layer.why.size() > 0)
      return layer;

    const int inC = linear1.inChannels;
    const int ffnC = linear1.outChannels;
    int K = 0;
    if(!pickSwigluK(accel, inC, K)) {
      layer.why = "no swiglu-epilogue artifact serving K >= " + Global::intToString(inC);
      return layer;
    }

    RyzenAIKernel::EngineInfo info;
    RyzenAIKernel::Engine* engine = swigluEngineForK(accel, K, info);
    if(engine == nullptr) {
      layer.why = "swiglu engine for K=" + Global::intToString(K) + " would not load";
      return layer;
    }

    const int paddedN = RyzenAIKernel::padN(info, 2 * ffnC);

    const double tConvert = nowSecs();
    vector<uint16_t> b((size_t)info.K * (size_t)paddedN, RyzenAIKernel::floatToBf16(0.0f));
    const float* w1 = linear1.weights.data();
    const float* wg = linearGate.weights.data();
    for(int ic = 0; ic < inC; ic++) {
      uint16_t* dst = b.data() + (size_t)ic * paddedN;
      const float* s1 = w1 + (size_t)ic * ffnC;
      const float* sg = wg + (size_t)ic * ffnC;
      for(int c = 0; c < ffnC; c++) {
        const int base = (c >> 3) << 4;
        dst[base + (c & 7)] = RyzenAIKernel::floatToBf16(s1[c]);
        dst[base + 8 + (c & 7)] = RyzenAIKernel::floatToBf16(sg[c]);
      }
    }
    accel->secsPrepareConvert += nowSecs() - tConvert;
    accel->prepareBytes += (long long)b.size() * 2;

    const double tUpload = nowSecs();
    string err;
    RyzenAIKernel::Weights* weights = RyzenAIKernel::uploadWeights(engine, paddedN, b.data(), err);
    accel->secsPrepareUpload += nowSecs() - tUpload;
    if(weights == nullptr) {
      layer.why = err;
      return layer;
    }

    layer.engine = engine;
    layer.weights = weights;
    layer.K = info.K;
    layer.paddedN = paddedN;
    layer.usable = true;
    return layer;
  }

}  // namespace

bool tryMatmulSwiglu(
  Accel* accel, float* out, const float* in,
  const MatMulLayerDesc& linear1, const MatMulLayerDesc& linearGate, int numRows) {
  if(accel == nullptr || numRows < accel->options.minRows) {
    if(accel != nullptr)
      accel->numSwigluFallback++;
    return false;
  }
  // The epilogue pairs adjacent 8-column groups, so the two projections must
  // agree on the shape and the channel count must fill whole groups.
  if(linear1.inChannels != linearGate.inChannels ||
     linear1.outChannels != linearGate.outChannels ||
     linear1.outChannels % 8 != 0) {
    accel->numSwigluFallback++;
    return false;
  }

  Layer& layer = prepareSwigluLayer(accel, linear1, linearGate);
  if(!layer.usable) {
    accel->numSwigluFallback++;
    return false;
  }
  if(!dispatchPacked(accel, layer, in, linear1.inChannels, numRows))
    return false;

  // C row r holds silu(l1)*gate for channel c at column (c>>3)*16 + (c&7);
  // the interleaved gate columns in between are the raw GEMM outputs and are
  // simply never read.
  const double tUnpack = nowSecs();
  const int paddedN = layer.paddedN;
  const int ffnC = linear1.outChannels;
  for(int r = 0; r < numRows; r++) {
    const float* src = accel->hostC.data() + (size_t)r * paddedN;
    float* dst = out + (size_t)r * ffnC;
    for(int c = 0; c < ffnC; c++)
      dst[c] = src[((c >> 3) << 4) + (c & 7)];
  }

  accel->secsUnpackC += nowSecs() - tUnpack;
  accel->numSwiglu++;
  return true;
}

bool tryConv1x1(
  Accel* accel, float* out, const float* in, const ConvLayerDesc& desc, int numRows,
  bool accumulate) {
  if(accel == nullptr || numRows < accel->options.minRows) {
    if(accel != nullptr)
      accel->numFallback++;
    return false;
  }
  if(desc.convXSize != 1 || desc.convYSize != 1)
    return false;  // not a fallback, just not this function's business
  Layer& layer = prepareLayer(
    accel, &desc, desc.weights.data(), desc.inChannels, desc.outChannels, WeightForm::OutByIn);
  if(!layer.usable) {
    accel->numFallback++;
    return false;
  }
  return runLayer(accel, layer, out, in, desc.inChannels, desc.outChannels, numRows, accumulate);
}

bool tryConv(
  Accel* accel, float* out, const float* in, const ConvLayerDesc& desc, int batchSize,
  int nnXLen, int nnYLen, bool accumulate) {
  const int numRows = batchSize * nnXLen * nnYLen;
  if(accel == nullptr || numRows < accel->options.minRows) {
    if(accel != nullptr)
      accel->numFallback++;
    return false;
  }
  const int kx = desc.convXSize;
  const int ky = desc.convYSize;
  if(kx == 1 && ky == 1)
    return tryConv1x1(accel, out, in, desc, numRows, accumulate);
  // Even kernel sizes would not centre; KataGo only ever uses odd ones.
  if(kx <= 0 || ky <= 0 || (kx % 2) == 0 || (ky % 2) == 0)
    return false;

  const int numTaps = ky * kx;
  Layer& layer = prepareLayer(
    accel, &desc, desc.weights.data(), desc.inChannels, desc.outChannels, WeightForm::ConvTaps,
    numTaps);
  if(!layer.usable) {
    accel->numFallback++;
    return false;
  }

  const RyzenAIKernel::EngineInfo& info = RyzenAIKernel::engineInfo(layer.engine);
  const int paddedM = RyzenAIKernel::padM(info, numRows);
  const int K = layer.K;
  const int inC = desc.inChannels;
  const int padX = kx / 2;
  const int padY = ky / 2;
  const int dilX = desc.dilationX;
  const int dilY = desc.dilationY;

  // im2col straight into the A scratch, in bfloat16. Off-board taps and the
  // columns past numTaps*inChannels stay zero, which is exactly what the
  // convolution's zero padding means.
  const double tGather = nowSecs();
  accel->hostA.assign((size_t)paddedM * (size_t)K, RyzenAIKernel::floatToBf16(0.0f));

  // One pass over the input converts it; the gather below then only moves
  // bytes. Same conversion function on the same values, so the bytes handed to
  // the NPU are identical to the element-wise version this replaces.
  const size_t inCount = (size_t)numRows * (size_t)inC;
  if(accel->hostConvIn.size() < inCount)
    accel->hostConvIn.resize(inCount);
  RyzenAIKernel::floatToBf16Bulk(in, accel->hostConvIn.data(), inCount);

  const uint16_t* convIn = accel->hostConvIn.data();
  const size_t tapBytes = (size_t)inC * sizeof(uint16_t);
  for(int n = 0; n < batchSize; n++) {
    for(int y = 0; y < nnYLen; y++) {
      for(int x = 0; x < nnXLen; x++) {
        uint16_t* dstRow =
          accel->hostA.data() + (((size_t)n * nnYLen + y) * nnXLen + x) * K;
        for(int dy = 0; dy < ky; dy++) {
          const int iy = y + (dy - padY) * dilY;
          if(iy < 0 || iy >= nnYLen)
            continue;
          for(int dx = 0; dx < kx; dx++) {
            const int ix = x + (dx - padX) * dilX;
            if(ix < 0 || ix >= nnXLen)
              continue;
            const uint16_t* src = convIn + ((((size_t)n * nnYLen + iy) * nnXLen + ix) * inC);
            uint16_t* dst = dstRow + (size_t)(dy * kx + dx) * inC;
            std::memcpy(dst, src, tapBytes);
          }
        }
      }
    }
  }
  accel->secsGather += nowSecs() - tGather;

  return runLayer(accel, layer, out, nullptr, inC, desc.outChannels, numRows, accumulate);
}

namespace {

  // First attention block of a given shape: pick engines for the two reduction
  // dims (numHeads*qHeadDim for QK^T, numHeads*S for P*V), then allocate and
  // zero-fill the host scratch and the two device-resident B buffers.
  // Attention shapes are fixed for a (model, geometry) pair, so this runs once
  // per Accel.
  void prepareAttention(
    Accel* accel, int S, int numHeads, int numKVHeads, int qHeadDim, int vHeadDim) {
    AttnState& st = accel->attn;
    if(st.usable || st.why.size() > 0)
      return;

    st.S = S;
    st.numHeads = numHeads;
    st.numKVHeads = numKVHeads;
    st.qHeadDim = qHeadDim;
    st.vHeadDim = vHeadDim;

    const int qTot = numHeads * qHeadDim;
    const int headsByS = numHeads * S;
    int kQK = 0;
    int kPV = 0;
    if(!pickK(accel, qTot, kQK)) {
      st.why = "no artifact with K >= qTot " + Global::intToString(qTot);
      return;
    }
    if(!pickK(accel, headsByS, kPV)) {
      st.why = "no artifact with K >= numHeads*S " + Global::intToString(headsByS);
      return;
    }

    RyzenAIKernel::EngineInfo infoQK;
    RyzenAIKernel::EngineInfo infoPV;
    st.engineQK = engineForK(accel, kQK, infoQK);
    if(st.engineQK == nullptr) {
      st.why = "attention engine for K=" + Global::intToString(kQK) + " would not load";
      return;
    }
    st.enginePV = engineForK(accel, kPV, infoPV);
    if(st.enginePV == nullptr) {
      st.why = "attention engine for K=" + Global::intToString(kPV) + " would not load";
      return;
    }

    st.KQK = infoQK.K;
    st.KPV = infoPV.K;
    st.paddedMQK = RyzenAIKernel::padM(infoQK, S);
    st.paddedMPV = RyzenAIKernel::padM(infoPV, S);
    st.paddedNQK = RyzenAIKernel::padN(infoQK, headsByS);
    st.paddedNPV = RyzenAIKernel::padN(infoPV, numHeads * vHeadDim);

    // Zero-filled: the padding regions of every buffer (and the off-diagonal
    // blocks of the two B buffers) stay zero for the state's lifetime, and
    // only the real regions are rewritten per dispatch.
    const uint16_t zero = RyzenAIKernel::floatToBf16(0.0f);
    st.hostAQK.assign((size_t)st.paddedMQK * st.KQK, zero);
    st.hostAPV.assign((size_t)st.paddedMPV * st.KPV, zero);
    st.hostBQK.assign((size_t)st.KQK * st.paddedNQK, zero);
    st.hostBPV.assign((size_t)st.KPV * st.paddedNPV, zero);
    st.hostC.assign(
      std::max((size_t)st.paddedMQK * st.paddedNQK, (size_t)st.paddedMPV * st.paddedNPV), 0.0f);
    st.scoreRow.assign(S, 0.0f);

    string err;
    st.weightsQK = RyzenAIKernel::uploadWeights(st.engineQK, st.paddedNQK, st.hostBQK.data(), err);
    if(st.weightsQK == nullptr) {
      st.why = err;
      return;
    }
    st.weightsPV = RyzenAIKernel::uploadWeights(st.enginePV, st.paddedNPV, st.hostBPV.data(), err);
    if(st.weightsPV == nullptr) {
      st.why = err;
      return;
    }

    // The softmax between the GEMMs goes to the NPU when an op compiled for
    // exactly this shape exists; its absence is ordinary (CPU path stays).
    // Rows pad to a multiple of 64 (8 cores x 8-row chunks) and columns to a
    // multiple of 32 (the kernel's vector width).
    st.smRows = ((numHeads * S + 63) / 64) * 64;
    st.smWidth = ((S + 31) / 32) * 32;
    {
      const string base = accel->options.artifactDir + "/ops/softmax_" +
        Global::intToString(st.smRows) + "x" + Global::intToString(st.smWidth);
      const size_t smBytes = (size_t)st.smRows * (size_t)st.smWidth * sizeof(uint16_t);
      string smErr;
      st.softmaxOp = RyzenAIKernel::loadOp(
        base + ".xclbin", base + ".insts.bin", accel->options.deviceIdx, &smBytes, 1, smBytes,
        smErr);
      if(st.softmaxOp != nullptr) {
        st.hostSmIn.assign(smBytes / sizeof(uint16_t), RyzenAIKernel::floatToBf16(0.0f));
        st.hostSmOut.assign(smBytes / sizeof(uint16_t), 0);
      }
    }

    // The fused attention op (QK^T + softmax + P*V in one dispatch) replaces
    // the whole staged path when present. Its host buffers carry Q/K/V
    // pre-tiled into the mmul tile orders; see
    // python/ryzenai_kernels/kernels/attention_head.cc for the layout contract.
    {
      const string attnBase = accel->options.artifactDir + "/ops/attn_h" +
        Global::intToString(numHeads) + "_s" + Global::intToString(S);
      const size_t qBytes = (size_t)numHeads * st.smWidth * qHeadDim * sizeof(uint16_t);
      // K/V are packed per query head (a GQA group head's data is replicated
      // onto each of its query heads' slots -- the compiled op's taps are
      // per-head).
      const size_t kvBytes = (size_t)numHeads * 2 * st.smWidth * qHeadDim * sizeof(uint16_t);
      const size_t cBytes = (size_t)numHeads * st.smWidth * vHeadDim * sizeof(float);
      const size_t inBytes[2] = {qBytes, kvBytes};
      string attErr;
      st.attnOp = RyzenAIKernel::loadOp(
        attnBase + ".xclbin", attnBase + ".insts.bin", accel->options.deviceIdx,
        inBytes, 2, cBytes, attErr);
      if(st.attnOp != nullptr) {
        st.attnQ.assign(qBytes / sizeof(uint16_t), RyzenAIKernel::floatToBf16(0.0f));
        st.attnKV.assign(kvBytes / sizeof(uint16_t), RyzenAIKernel::floatToBf16(0.0f));
        st.attnC.assign(cBytes / sizeof(float), 0.0f);
      }
    }

    st.usable = true;
  }

}  // namespace

bool tryAttention(
  Accel* accel, float* attnOut,
  const float* qBuf, const float* kBuf, const float* vBuf, const float* mask,
  int batchSize, int S, int numHeads, int numKVHeads, int qHeadDim, int vHeadDim,
  double* softmaxSecsOut) {
  if(softmaxSecsOut != nullptr)
    *softmaxSecsOut = 0.0;
  if(accel == nullptr)
    return false;
  prepareAttention(accel, S, numHeads, numKVHeads, qHeadDim, vHeadDim);
  AttnState& st = accel->attn;
  if(!st.usable || st.S != S || st.numHeads != numHeads || st.numKVHeads != numKVHeads ||
     st.qHeadDim != qHeadDim || st.vHeadDim != vHeadDim) {
    accel->numAttnFallback++;
    return false;
  }

  const int qTot = numHeads * qHeadDim;
  const int kTot = numKVHeads * qHeadDim;
  const int vTot = numKVHeads * vHeadDim;
  const int oTot = numHeads * vHeadDim;
  const int kvGroupSize = numHeads / numKVHeads;
  const float scale = 1.0f / sqrtf((float)qHeadDim);
  const uint16_t zero16 = RyzenAIKernel::floatToBf16(0.0f);
  double softmaxSecs = 0.0;

  // ---- fused path: the whole attention in one dispatch per batch element ---
  // The op computes softmax((q*scale) @ K^T) @ V entirely on-chip. It handles
  // full boards only (masked positions would need the -inf treatment the
  // staged path's gather does), so anything masked falls through below.
  if(st.attnOp != nullptr && qHeadDim == 32 && vHeadDim == 32) {
    bool fullBoard = true;
    for(int i = 0; i < batchSize * S; i++) {
      if(mask[i] == 0.0f) {
        fullBoard = false;
        break;
      }
    }
    if(fullBoard) {
      const int sPad = st.smWidth;  // the op's padded S (384 for 361)
      for(int n = 0; n < batchSize; n++) {
        const float* qN = qBuf + (size_t)n * S * qTot;
        const float* kN = kBuf + (size_t)n * S * kTot;
        const float* vN = vBuf + (size_t)n * S * vTot;
        const double tPack = nowSecs();

        // Each head-row is 32 contiguous floats going to a strided place, so
        // convert the 32 in one call and then move plain uint16s. Doing the
        // conversion inside the scatter meant 32 branchy scalar conversions per
        // row, which is what made packing rival dispatch in the profile.
        float scaled[32];
        uint16_t tmp[32];

        // Q: per-head contiguous 8x32 chunks in A-tile order, pre-scaled.
        for(int h = 0; h < numHeads; h++) {
          uint16_t* qH = st.attnQ.data() + (size_t)h * sPad * 32;
          const float* src = qN + (size_t)h * 32;
          for(int qi = 0; qi < S; qi++, src += qTot) {
            // within-chunk offset = (rowgroup*4 + row)*8 lanes: rows 4-7 of
            // the chunk live 128 elements in, not at (qi&3)*8.
            uint16_t* chunk = qH + (size_t)(qi >> 3) * 256 +
                              (((qi & 7) >> 2) * 128) + ((qi & 3) * 8);
            for(int d = 0; d < 32; d++)
              scaled[d] = src[d] * scale;
            RyzenAIKernel::floatToBf16Bulk(scaled, tmp, 32);
            // d's low three bits stay adjacent, so each group of eight is one
            // contiguous run at stride 32.
            for(int g = 0; g < 4; g++)
              std::memcpy(chunk + (size_t)g * 32, tmp + g * 8, 8 * sizeof(uint16_t));
          }
        }
        // K/V: per head, K then V back to back, each in its B-tile order.
        for(int h = 0; h < numHeads; h++) {
          const int kvh = h / kvGroupSize;
          uint16_t* kvH = st.attnKV.data() + (size_t)h * 2 * sPad * 32;
          const float* kSrc = kN + (size_t)kvh * 32;
          const float* vSrc = vN + (size_t)kvh * 32;
          for(int ki = 0; ki < S; ki++, kSrc += kTot, vSrc += vTot) {
            uint16_t* kDst = kvH + (size_t)(ki >> 3) * 256 + (ki & 7) * 8;
            RyzenAIKernel::floatToBf16Bulk(kSrc, tmp, 32);
            for(int g = 0; g < 4; g++)
              std::memcpy(kDst + (size_t)g * 64, tmp + g * 8, 8 * sizeof(uint16_t));
            // V is transposed relative to K, so this one stays a scatter - but
            // now it moves already-converted values.
            uint16_t* vDst = kvH + (size_t)sPad * 32 + (ki & 7) + (size_t)(ki >> 3) * 64;
            RyzenAIKernel::floatToBf16Bulk(vSrc, tmp, 32);
            for(int dv = 0; dv < 32; dv++)
              vDst[(dv >> 3) * 3072 + (dv & 7) * 8] = tmp[dv];
          }
        }
        accel->secsAttnPack += nowSecs() - tPack;

        const double tDispatch = nowSecs();
        try {
          const void* ins[2] = {st.attnQ.data(), st.attnKV.data()};
          RyzenAIKernel::runOp(st.attnOp, ins, 2, st.attnC.data());
        }
        catch(const std::exception&) {
          st.attnOp = nullptr;  // permanent: staged path from here on
          accel->numAttnFallback++;
          return false;
        }
        accel->secsAttnDispatch += nowSecs() - tDispatch;

        // C: per-head 8x32 chunks in C-tile order -> attnOut rows.
        const double tUnpack = nowSecs();
        for(int h = 0; h < numHeads; h++) {
          const float* cH = st.attnC.data() + (size_t)h * sPad * 32;
          float* dst = attnOut + (size_t)n * S * oTot + (size_t)h * 32;
          for(int qi = 0; qi < S; qi++) {
            const float* chunk = cH + (size_t)(qi >> 3) * 256 +
                                 (((qi & 7) >> 2) * 128) + ((qi & 3) * 8);
            float* out = dst + (size_t)qi * oTot;
            for(int dv = 0; dv < 32; dv++)
              out[dv] = chunk[(dv >> 3) * 32 + (dv & 7)];
          }
        }
        accel->secsAttnUnpack += nowSecs() - tUnpack;
      }
      accel->numAttn++;
      return true;
    }
  }

  for(int n = 0; n < batchSize; n++) {
    const float* maskN = mask + (size_t)n * S;

    // ---- QK^T for every head in one GEMM ----------------------------------
    // A[qi][h*qHeadDim + d] = qBuf[n][qi][h*qHeadDim + d] -- i.e. the raw
    // qBuf row, which already concatenates the heads exactly along the
    // reduction dim.
    // B is block-diagonal: B[h*qHeadDim + d][h*S + ki] = kBuf[n][ki][kvh*qHeadDim+d]
    // (the K matrix of head h, transposed), zero elsewhere, so
    // C[qi][h*S + ki] = <Q_h[qi], K_h[ki]> falls out per head.
    const double tPack = nowSecs();
    for(int qi = 0; qi < S; qi++) {
      const float* src = qBuf + ((size_t)n * S + qi) * qTot;
      uint16_t* dst = st.hostAQK.data() + (size_t)qi * st.KQK;
      for(int c = 0; c < qTot; c++)
        dst[c] = RyzenAIKernel::floatToBf16(src[c]);
    }
    for(int h = 0; h < numHeads; h++) {
      const int kvh = h / kvGroupSize;
      for(int d = 0; d < qHeadDim; d++) {
        uint16_t* dst = st.hostBQK.data() + (size_t)(h * qHeadDim + d) * st.paddedNQK + (size_t)h * S;
        const float* src = kBuf + (size_t)n * S * kTot + (size_t)kvh * qHeadDim + d;
        for(int ki = 0; ki < S; ki++)
          dst[ki] = RyzenAIKernel::floatToBf16(src[(size_t)ki * kTot]);
      }
    }
    const double tUpload = nowSecs();
    accel->secsAttnPack += tUpload - tPack;
    RyzenAIKernel::rewriteWeights(st.weightsQK, st.hostBQK.data());
    const double tDispatch = nowSecs();
    accel->secsAttnUploadB += tDispatch - tUpload;
    try {
      RyzenAIKernel::runGemm(
        st.engineQK, st.weightsQK, st.paddedMQK, st.paddedNQK, st.hostAQK.data(),
        st.hostC.data());
    }
    catch(const std::exception&) {
      // A dispatch failure is not fatal: drop attention to the CPU
      // permanently and let the evaluation finish with correct numbers.
      st.usable = false;
      st.why = "dispatch failed at run time";
      accel->numAttnFallback++;
      return false;
    }
    accel->secsAttnDispatch += nowSecs() - tDispatch;

    // ---- softmax ------------------------------------------------------------
    // NPU when an op for this exact shape loaded, CPU otherwise. Either way
    // the probabilities land in the P*V GEMM's A (bfloat16) at
    // A[qi][h*S + ki], and rows of masked-out queries and columns of
    // masked-out keys are exactly 0, matching the reference semantics.
    if(st.softmaxOp != nullptr) {
      const double tGather = nowSecs();
      const uint16_t negInf = RyzenAIKernel::floatToBf16(-1e30f);
      for(int h = 0; h < numHeads; h++) {
        for(int qi = 0; qi < S; qi++) {
          if(maskN[qi] == 0.0f)
            continue;  // the scatter writes exact zeros for this row
          const float* cRow = st.hostC.data() + (size_t)qi * st.paddedNQK + (size_t)h * S;
          uint16_t* dst = st.hostSmIn.data() + ((size_t)h * S + qi) * st.smWidth;
          for(int ki = 0; ki < S; ki++)
            dst[ki] = maskN[ki] == 0.0f ? negInf : RyzenAIKernel::floatToBf16(cRow[ki] * scale);
          for(int ki = S; ki < st.smWidth; ki++)
            dst[ki] = negInf;
        }
      }
      const double tSmDispatch = nowSecs();
      accel->secsAttnSoftmaxHost += tSmDispatch - tGather;
      try {
        const void* smIns[1] = {st.hostSmIn.data()};
        RyzenAIKernel::runOp(st.softmaxOp, smIns, 1, st.hostSmOut.data());
      }
      catch(const std::exception&) {
        st.usable = false;
        st.why = "softmax dispatch failed at run time";
        accel->numAttnFallback++;
        return false;
      }
      const double tScatter = nowSecs();
      accel->secsAttnSoftmaxNpu += tScatter - tSmDispatch;

      for(int h = 0; h < numHeads; h++) {
        for(int qi = 0; qi < S; qi++) {
          uint16_t* dst = st.hostAPV.data() + (size_t)qi * st.KPV + (size_t)h * S;
          if(maskN[qi] == 0.0f) {
            std::fill(dst, dst + S, zero16);
            continue;
          }
          const uint16_t* src = st.hostSmOut.data() + ((size_t)h * S + qi) * st.smWidth;
          for(int ki = 0; ki < S; ki++)
            dst[ki] = maskN[ki] == 0.0f ? zero16 : src[ki];
        }
      }
      accel->secsAttnSoftmaxHost += nowSecs() - tScatter;
    }
    else {
    // ---- softmax on the CPU, byte-for-byte the reference semantics --------
    // Reads the raw scores out of C and writes the probabilities straight
    // into the P*V GEMM's A (bfloat16), at A[qi][h*S + ki]. Rows of
    // masked-out queries and columns of masked-out keys are exactly 0.
    const double tSoftmax = nowSecs();
    for(int h = 0; h < numHeads; h++) {
      for(int qi = 0; qi < S; qi++) {
        uint16_t* dst = st.hostAPV.data() + (size_t)qi * st.KPV + (size_t)h * S;
        if(maskN[qi] == 0.0f) {
          std::fill(dst, dst + S, zero16);
          continue;
        }
        const float* cRow = st.hostC.data() + (size_t)qi * st.paddedNQK + (size_t)h * S;
        float maxVal = -1e30f;
        for(int ki = 0; ki < S; ki++) {
          if(maskN[ki] == 0.0f)
            continue;
          const float acc = cRow[ki] * scale;
          st.scoreRow[ki] = acc;
          if(acc > maxVal)
            maxVal = acc;
        }
        float sumExp = 0.0f;
        for(int ki = 0; ki < S; ki++) {
          if(maskN[ki] == 0.0f)
            continue;
          float e = expf(st.scoreRow[ki] - maxVal);
          st.scoreRow[ki] = e;
          sumExp += e;
        }
        float invSum = 1.0f / sumExp;
        for(int ki = 0; ki < S; ki++)
          dst[ki] = maskN[ki] == 0.0f ? zero16 : RyzenAIKernel::floatToBf16(st.scoreRow[ki] * invSum);
      }
    }
    softmaxSecs += nowSecs() - tSoftmax;
    }

    // ---- P*V for every head in one GEMM -----------------------------------
    // A is the softmax output written above. B is block-diagonal:
    // B[h*S + ki][h*vHeadDim + dv] = vBuf[n][ki][kvh*vHeadDim + dv], zero
    // elsewhere, so C[qi][h*vHeadDim + dv] = sum_ki P_h[qi][ki] * V_h[ki][dv]
    // -- i.e. one C row is exactly one attnOut row.
    const double tPack2 = nowSecs();
    for(int h = 0; h < numHeads; h++) {
      const int kvh = h / kvGroupSize;
      for(int ki = 0; ki < S; ki++) {
        const float* src = vBuf + ((size_t)n * S + ki) * vTot + (size_t)kvh * vHeadDim;
        uint16_t* dst =
          st.hostBPV.data() + (size_t)(h * S + ki) * st.paddedNPV + (size_t)h * vHeadDim;
        for(int dv = 0; dv < vHeadDim; dv++)
          dst[dv] = RyzenAIKernel::floatToBf16(src[dv]);
      }
    }
    const double tUpload2 = nowSecs();
    accel->secsAttnPack += tUpload2 - tPack2;
    RyzenAIKernel::rewriteWeights(st.weightsPV, st.hostBPV.data());
    const double tDispatch2 = nowSecs();
    accel->secsAttnUploadB += tDispatch2 - tUpload2;
    try {
      RyzenAIKernel::runGemm(
        st.enginePV, st.weightsPV, st.paddedMPV, st.paddedNPV, st.hostAPV.data(),
        st.hostC.data());
    }
    catch(const std::exception&) {
      st.usable = false;
      st.why = "dispatch failed at run time";
      accel->numAttnFallback++;
      return false;
    }
    const double tUnpack = nowSecs();
    accel->secsAttnDispatch += tUnpack - tDispatch2;

    for(int qi = 0; qi < S; qi++) {
      float* dst = attnOut + ((size_t)n * S + qi) * oTot;
      const float* src = st.hostC.data() + (size_t)qi * st.paddedNPV;
      std::memcpy(dst, src, (size_t)oTot * sizeof(float));
    }
    accel->secsAttnUnpack += nowSecs() - tUnpack;
  }

  accel->numAttn++;
  accel->secsAttnSoftmax += softmaxSecs;
  if(softmaxSecsOut != nullptr)
    *softmaxSecsOut = softmaxSecs;
  return true;
}





bool tryBnMish(
  Accel* accel, float* out, const float* in, const BatchNormLayerDesc& bn,
  int activation, int numRows) {
  if(accel == nullptr)
    return false;
  if(activation != ACTIVATION_MISH)
    return false;

  const int C = bn.numChannels;
  // Smallest shipped height that covers this call in one dispatch, or the
  // tallest if none does (which then chunks). Both ends cost: too short pays
  // the fixed dispatch cost repeatedly, too tall pays the per-row cost on rows
  // that are only padding. Choosing per call is the point - latching a height
  // on the first call is what silently cost b40c768 its BN acceleration under
  // batching (CPU norms 0.38 s -> 21.55 s).
  int rows = Accel::kBnMishHeights[2];
  for(int i = 0; i < 3; i++) {
    if(Accel::kBnMishHeights[i] >= numRows) {
      rows = Accel::kBnMishHeights[i];
      break;
    }
  }
  Accel::BnMishState& st = accel->bnmByShape[std::make_pair(C, rows)];
  if(!st.tried) {
    st.tried = true;
    st.width = C;
    st.rowsPad = rows;
    const string base = accel->options.artifactDir + "/ops/bnmish_" +
      Global::intToString(st.rowsPad) + "x" + Global::intToString(st.width);
    const size_t xBytes = (size_t)st.rowsPad * st.width * sizeof(uint16_t);
    const size_t sbBytes = (size_t)8 * 2 * st.width * sizeof(uint16_t);
    const size_t inBytes[2] = {xBytes, sbBytes};
    string err;
    st.op = RyzenAIKernel::loadOp(
      base + ".xclbin", base + ".insts.bin", accel->options.deviceIdx,
      inBytes, 2, xBytes, err);
    if(st.op != nullptr) {
      st.hostX.assign(xBytes / sizeof(uint16_t), RyzenAIKernel::floatToBf16(0.0f));
      st.hostSB.assign(sbBytes / sizeof(uint16_t), 0);
      st.hostY.assign(xBytes / sizeof(uint16_t), 0);
    }
    else {
      st.why = "no " + Global::intToString(st.rowsPad) + "x" +
        Global::intToString(st.width) + " artifact (" + err + ")";
    }
  }
  if(st.op == nullptr)
    return false;

  const int W = st.width;
  // Row blocks are independent: feeding the op 64 rows at a time instead of
  // 384 reproduced the whole raw-nn output byte for byte, partial final block
  // included, so this stride is purely a performance knob.
  const int rowsPerRun = st.rowsPad;
  const double t0 = nowSecs();

  // scale/bias, replicated per core, [scale | bias] per core. Per layer, not
  // per row block, so it is packed once for the whole call.
  for(int core = 0; core < 8; core++) {
    uint16_t* sb = st.hostSB.data() + (size_t)core * 2 * W;
    for(int c = 0; c < W; c++) {
      sb[c] = RyzenAIKernel::floatToBf16(bn.mergedScale[c]);
      sb[W + c] = RyzenAIKernel::floatToBf16(bn.mergedBias[c]);
    }
  }

  for(int base = 0; base < numRows; base += rowsPerRun) {
    const int rows = std::min(rowsPerRun, numRows - base);

    // X: rows of board points in bf16. A short final block leaves whatever the
    // previous block wrote in the tail; those outputs are simply not read back,
    // and Mish is elementwise so they cannot disturb the rows that are.
    for(int r = 0; r < rows; r++) {
      const float* src = in + (size_t)(base + r) * W;
      uint16_t* dst = st.hostX.data() + (size_t)r * W;
      for(int c = 0; c < W; c++)
        dst[c] = RyzenAIKernel::floatToBf16(src[c]);
    }

    try {
      const void* ins[2] = {st.hostX.data(), st.hostSB.data()};
      RyzenAIKernel::runOp(st.op, ins, 2, st.hostY.data());
    }
    catch(const std::exception&) {
      RyzenAIKernel::freeOp(st.op);
      st.op = nullptr;
      // Rows before this block are already written; the caller's contract is
      // that a false return means it must redo the whole layer, which
      // overwrites them.
      return false;
    }

    for(int r = 0; r < rows; r++) {
      float* dst = out + (size_t)(base + r) * W;
      const uint16_t* src = st.hostY.data() + (size_t)r * W;
      for(int c = 0; c < W; c++)
        dst[c] = RyzenAIKernel::bf16ToFloat(src[c]);
    }
    accel->numBnMish++;  // counts dispatches, so batching shows up here
  }

  accel->secsBnMish += nowSecs() - t0;
  return true;
}

string report(const Accel* accel) {  if(accel == nullptr)
    return "RyzenAI dense layers: accelerator not created";

  std::ostringstream out;
  out << "RyzenAI dense layers: " << accel->numAccelerated << " on NPU, " << accel->numFallback
      << " on CPU";
  if(accel->numBnMish > 0) {
    char b[128];
    std::snprintf(
      b, sizeof(b), "\n  bn+mish on NPU: %lld dispatches (%.2f s)", accel->numBnMish,
      accel->secsBnMish);
    out << b;
  }
  // A width with no artifact runs on the CPU at full numerical fidelity, so
  // nothing looks wrong - it is just slow. Name it, or the next model with an
  // unshipped channel count silently loses this acceleration the way batched
  // b40c768 did.
  for(const auto& e : accel->bnmByShape) {
    if(e.second.why.size() > 0)
      out << "\n  bn+mish NOT on NPU for " << e.first.first << " channels x "
          << e.first.second << " rows: " << e.second.why;
  }
  if(accel->numAccelerated > 0) {
    const double total =
      accel->secsPackA + accel->secsGather + accel->secsDispatch + accel->secsUnpackC;
    char buf[320];
    std::snprintf(
      buf, sizeof(buf),
      "\n  %.2f s total: pack A %.2f s (%.0f%%), im2col %.2f s (%.0f%%), "
      "dispatch %.2f s (%.0f%%), unpack C %.2f s (%.0f%%)"
      "\n  %.3f ms per accelerated layer",
      total, accel->secsPackA, 100 * accel->secsPackA / total,
      accel->secsGather, 100 * accel->secsGather / total,
      accel->secsDispatch, 100 * accel->secsDispatch / total,
      accel->secsUnpackC, 100 * accel->secsUnpackC / total,
      1000.0 * total / (double)accel->numAccelerated);
    out << buf;
    std::snprintf(
      buf, sizeof(buf),
      "\n  one-time weight prep: convert+pad %.3f s, upload %.3f s, %.1f MB",
      accel->secsPrepareConvert, accel->secsPrepareUpload,
      (double)accel->prepareBytes / (1024.0 * 1024.0));
    out << buf;
  }

  if(!accel->engines.empty() || !accel->swigluEngines.empty()) {
    out << "\n  engines:";
    for(const auto& entry : accel->infos)
      out << " K=" << entry.first << "(" << entry.second.cols << "col)";
    for(const auto& entry : accel->swigluInfos)
      out << " K=" << entry.first << "(" << entry.second.cols << "col,swiglu)";

    // Split the dispatch cost, which is what decides whether keeping
    // activations resident on the device would pay.
    RyzenAIKernel::Timings t;
    for(const auto& entry : accel->engines) {
      const RyzenAIKernel::Timings& e = RyzenAIKernel::engineTimings(entry.second);
      t.secsUploadA += e.secsUploadA;
      t.secsExecute += e.secsExecute;
      t.secsDownloadC += e.secsDownloadC;
      t.numDispatches += e.numDispatches;
    }
    for(const auto& entry : accel->swigluEngines) {
      const RyzenAIKernel::Timings& e = RyzenAIKernel::engineTimings(entry.second);
      t.secsUploadA += e.secsUploadA;
      t.secsExecute += e.secsExecute;
      t.secsDownloadC += e.secsDownloadC;
      t.numDispatches += e.numDispatches;
    }
    if(t.numDispatches > 0) {
      char buf[256];
      std::snprintf(
        buf, sizeof(buf),
        "\n  inside dispatch: upload A %.2f s, execute %.2f s, download C %.2f s"
        "  (%.3f / %.3f / %.3f ms each, %lld dispatches)",
        t.secsUploadA, t.secsExecute, t.secsDownloadC,
        1000.0 * t.secsUploadA / (double)t.numDispatches,
        1000.0 * t.secsExecute / (double)t.numDispatches,
        1000.0 * t.secsDownloadC / (double)t.numDispatches,
        t.numDispatches);
      out << buf;
    }
  }

  // Attention (QK^T / P*V) routing, reported separately from the dense layers
  // because its per-call cost profile is entirely different.
  if(accel->numAttn > 0 || accel->numAttnFallback > 0) {
    out << "\n  attention blocks: " << accel->numAttn << " on NPU, " << accel->numAttnFallback
        << " on CPU";
    if(accel->numAttn > 0) {
      const double total =
        accel->secsAttnPack + accel->secsAttnUploadB + accel->secsAttnDispatch +
        accel->secsAttnUnpack;
      char buf[448];
      std::snprintf(
        buf, sizeof(buf),
        "\n  %.2f s around the GEMMs: pack %.2f s, upload B %.2f s, dispatch %.2f s, "
        "unpack %.2f s  (%.3f ms per block)",
        total, accel->secsAttnPack, accel->secsAttnUploadB, accel->secsAttnDispatch,
        accel->secsAttnUnpack, 1000.0 * total / (double)accel->numAttn);
      out << buf;
      if(accel->attn.softmaxOp != nullptr)
        std::snprintf(
          buf, sizeof(buf),
          "\n  softmax on NPU: dispatch %.2f s, gather/scatter %.2f s",
          accel->secsAttnSoftmaxNpu, accel->secsAttnSoftmaxHost);
      else
        std::snprintf(
          buf, sizeof(buf), "\n  softmax still on CPU: %.2f s", accel->secsAttnSoftmax);
      out << buf;
    }
    if(!accel->attn.usable && accel->attn.why.size() > 0)
      out << "\n  attention on CPU: " << accel->attn.why;
  }

  // Only the reasons matter, not which layer hit them: a model has at most a
  // handful of distinct failure modes and dozens of layers sharing each.
  if(accel->numSwiglu > 0 || accel->numSwigluFallback > 0)
    out << "\n  swiglu epilogue: " << accel->numSwiglu << " FFN blocks on NPU, "
        << accel->numSwigluFallback << " on CPU";
  std::map<string, int> reasons;
  for(const auto& entry : accel->layers) {
    if(!entry.second.usable && entry.second.why.size() > 0)
      reasons[entry.second.why]++;
  }
  for(const auto& entry : accel->swigluLayers) {
    if(!entry.second.usable && entry.second.why.size() > 0)
      reasons[entry.second.why]++;
  }
  for(const auto& entry : reasons)
    out << "\n  " << entry.second << " layer(s) on CPU: " << entry.first;

  return out.str();
}

}  // namespace RyzenAIMatMul
