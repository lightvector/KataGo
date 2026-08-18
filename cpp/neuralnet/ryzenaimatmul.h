#ifndef NEURALNET_RYZENAI_MATMUL_H_
#define NEURALNET_RYZENAI_MATMUL_H_

// Routes KataGo's dense layers to the NPU, falling back to the caller's CPU
// implementation whenever it cannot help. Also routes the two attention
// matmuls (QK^T and P*V), whose B operands are per-evaluation activations
// rather than resident weights -- see tryAttention below.
//
// KataGo already stores MatMulLayerDesc weights as inChannels x outChannels
// row-major, which is exactly the B operand the GEMM kernel wants, so weights
// need converting to bfloat16 but never transposing. Each layer's B is uploaded
// once and stays resident; only A and C cross the bus per evaluation.
//
// Padding: an xclbin fixes K, and the instruction stream requires M to be a
// multiple of tileM*8 and N a multiple of tileN*cols. Rows and columns beyond
// the real shape are zero-filled and their outputs discarded, so any layer
// shape runs -- at the cost of the wasted multiply-accumulates.

#include <string>
#include <vector>

struct MatMulLayerDesc;
struct ConvLayerDesc;
struct BatchNormLayerDesc;
struct TransformerAttentionDesc;
struct TransformerFFNDesc;

namespace RyzenAIMatMul {

  struct Accel;

  struct Options {
    std::string artifactDir;
    int deviceIdx = -1;         // <0 selects the default device
    std::string dtype = "auto"; // auto | bf16 | bfp16
    int maxCols = 0;            // 0 = as wide as the device allows
    // Dense layers whose row count is below this stay on the CPU. A dispatch
    // costs on the order of 0.2 ms no matter how small the work is, which
    // swamps the batch-row matmuls in the heads (a few dozen rows at most).
    int minRows = 128;
    // Force every layer onto one reduction dim, so the whole model runs from a
    // single xclbin and therefore a single hardware context. Switching contexts
    // measured ~0.46 ms per dispatch, which dwarfs the arithmetic these kernels
    // do, so paying extra zero-padded multiply-accumulates to avoid it can win
    // by a wide margin -- on models whose GEMMs are small. 0 disables it.
    int forceK = 0;
    bool verbose = false;
  };

  // Returns nullptr (and fills err) when the NPU cannot be used at all, which
  // is an ordinary outcome: callers then run entirely on the CPU path.
  Accel* create(const Options& options, std::string& err);
  void free(Accel* accel);

  // out[numRows][outChannels] = in[numRows][inChannels] * weights, matching
  // reference.cpp's matmulRows exactly. Returns false without touching `out`
  // if this layer is not eligible or no artifact covers it, in which case the
  // caller must run its own implementation.
  bool tryMatmul(
    Accel* accel, float* out, const float* in, const MatMulLayerDesc& desc, int numRows);

  // Several projections that read the same input (attention's q/k/vProj, the
  // FFN's linear1/linearGate) fused into one GEMM: the weights are
  // concatenated along N once at upload, so one dispatch replaces numDescs.
  // out[j] receives descs[j]'s columns. Returns false without touching any
  // output if the group is not eligible; the caller runs each layer itself.
  bool tryMatmulMulti(
    Accel* accel, float* const* outs, const float* in,
    const MatMulLayerDesc* const* descs, int numDescs, int numRows);

  // The transformer FFN's SwiGLU: out = silu(in @ linear1) * (in @
  // linearGate), with both projections fused into ONE GEMM dispatch whose
  // epilogue applies the silu and the multiply on chip. Requires a
  // gemm_swiglu_bf16 artifact for the layer's reduction dim (a separate
  // xclbin, hence a separate hardware context) and outChannels % 8 == 0 --
  // the uploaded B interleaves the two weight matrices in groups of 8 columns
  // so each core's C tile holds (linear1, gate) sub-tile pairs. Returns false
  // without touching `out` when unavailable; the caller then runs the two
  // projections and the elementwise SwiGLU itself.
  bool tryMatmulSwiglu(
    Accel* accel, float* out, const float* in,
    const MatMulLayerDesc& linear1, const MatMulLayerDesc& linearGate, int numRows);

  // A 1x1 convolution is a dense layer over board points, so it takes the same
  // path. Declines anything with a larger kernel. ConvLayerDesc stores weights
  // as outChannels x inChannels, the transpose of what the GEMM wants, so those
  // are transposed once at upload rather than per evaluation.
  //
  // accumulate adds into `out` instead of overwriting it, matching
  // reference.cpp's convNHWC.
  bool tryConv1x1(
    Accel* accel, float* out, const float* in, const ConvLayerDesc& desc, int numRows,
    bool accumulate);

  // BatchNorm + Mish fused op (out = mish(scale*x + bias) per channel), for
  // the convolutional model's trunk norms. Only handles Mish and full boards
  // (masked positions would need zeroing the staged path does); anything else
  // returns false and the caller runs its CPU loop. rows = batchSize*S.
  bool tryBnMish(
    Accel* accel, float* out, const float* in, const BatchNormLayerDesc& bn,
    int activation, int numRows);

  // A convolution with a larger kernel becomes the same GEMM once its input
  // patches are gathered into rows: K = convY*convX*inChannels, one row per
  // board point ("implicit GEMM"). The gather costs one pass over the input per
  // tap, which is far cheaper than the direct convolution it replaces.
  //
  // Declines when no artifact reaches K, which for a 3x3 means 9*inChannels.
  bool tryConv(
    Accel* accel, float* out, const float* in, const ConvLayerDesc& desc, int batchSize,
    int nnXLen, int nnYLen, bool accumulate);

  // Routes the two attention matmuls (QK^T and P*V) of one transformer
  // attention block to the NPU: one GEMM per batch element each way, with all
  // heads laid out side by side along A's reduction dim and a block-diagonal
  // B built from the K / V activations, so a block costs 2 dispatches per
  // batch element instead of 2 per head (a dispatch costs ~1 ms regardless of
  // size; the multiply-accumulates are nearly free). The softmax in between
  // goes to the NPU too when an op compiled for exactly this (numHeads*S, S)
  // shape is present under artifactDir/ops; otherwise it runs on the CPU.
  // Either way the semantics are exactly the reference path's (scale on the
  // scores, masked-out query rows produce exact zeros, masked-out key columns
  // excluded -- as -1e30 through the NPU softmax, whose exp underflows to 0).
  //
  // Unlike layer weights, the B operands here (the K and V activations)
  // change every evaluation, so they are re-uploaded per dispatch through
  // RyzenAIKernel::rewriteWeights into BOs that stay resident.
  //
  // qBuf/kBuf/vBuf are [batchSize][S][qTot/kTot/vTot] with the per-head slices
  // at h*qHeadDim and kvh*qHeadDim / kvh*vHeadDim (kvh = h / (numHeads/
  // numKVHeads)); attnOut is [batchSize][S][numHeads*vHeadDim]; mask is
  // [batchSize][S] with 1.0f on-board. All exactly as reference.cpp's
  // applyTransformerAttentionBlock already lays them out.
  //
  // softmaxSecsOut, when non-null, accumulates the wall clock spent in the
  // on-CPU softmax so the caller's profile can keep attributing it.
  //
  // Returns false if any piece is not eligible or a dispatch failed. attnOut
  // may then be partially written, so the caller must run its own
  // implementation for the whole block, which overwrites it fully.
  bool tryAttention(
    Accel* accel, float* attnOut,
    const float* qBuf, const float* kBuf, const float* vBuf, const float* mask,
    int batchSize, int S, int numHeads, int numKVHeads, int qHeadDim, int vHeadDim,
    double* softmaxSecsOut);

  // Per-layer accounting, for logging: how many dense layers were routed to the
  // NPU, how many fell back, and why.
  std::string report(const Accel* accel);

}  // namespace RyzenAIMatMul

#endif  // NEURALNET_RYZENAI_MATMUL_H_
