/*
 * Pure C++ (C++17, standard library only) CPU reference implementation of the
 * KataGo neural network forward pass, for the RyzenAI (AMD NPU) backend.
 * See reference.h for the API and the exact input/output buffer contracts.
 *
 * The math here is a direct, loop-level port of cpp/neuralnet/eigenbackend.cpp
 * (Winograd convolutions there are replaced by plain direct convolutions,
 * which are mathematically identical). Internal layout is NHWC throughout.
 *
 * No heap allocation happens inside forward(): all scratch is bump-allocated
 * from an arena whose exact worst-case size is computed in createWorkspace().
 */

#include "../neuralnet/ryzenaireference.h"

#include <algorithm>
#include <cmath>
#include <map>
#include <memory>
#include <vector>

#include "../neuralnet/activations.h"
#include "../neuralnet/ryzenaimatmul.h"

#include <chrono>
#include <cstdio>

namespace RyzenAIRef {

namespace {

// ---------------------------------------------------------------------------
// Precomputed per-attention-block RoPE cos/sin tables (created once per
// Workspace via TransformerAttentionDesc::computeRopeCosSin).
// ---------------------------------------------------------------------------
struct RopeTables {
  std::vector<float> cosTable;
  std::vector<float> sinTable;
};

// ---------------------------------------------------------------------------
// Bump allocator over a preallocated float arena. No heap traffic at forward
// time; alloc() throws if the compile-time-computed capacity is exceeded
// (which would indicate a bug in the size accounting below).
// ---------------------------------------------------------------------------
struct Arena {
  float* base;
  size_t capacity;
  size_t offset;

  float* alloc(size_t numElts) {
    if(numElts > capacity - offset)
      throw StringError("RyzenAIRef: internal scratch arena overflow");
    float* p = base + offset;
    offset += numElts;
    return p;
  }
  size_t mark() const { return offset; }
  void rewind(size_t m) { offset = m; }
};

// Coarse profile of the parts that are still on the CPU, so that the next
// thing to move onto the NPU is chosen from measurement rather than from
// multiply-accumulate counts (which have already misled once: see
// references/performance.md). Off unless RyzenAIRef::profileEnabled() is set.
struct CpuProfile {
  double attnScores = 0.0;   // CPU-path attention only (QK^T, softmax, P*V loops)
  double softmax = 0.0;      // softmax anywhere: nested in the CPU attention path,
                             // or reported by the NPU attention path
  double norms = 0.0;        // RMSNorm / BatchNorm / activations
  double rope = 0.0;
  double swiglu = 0.0;       // silu(linear1) * gate elementwise
  double residual = 0.0;     // masked residual adds
  double gpool = 0.0;
  double headSmall = 0.0;    // heads' batch-row matmuls and pooling
  bool enabled = false;
};
CpuProfile g_profile;

inline double nowSecs() {
  return std::chrono::duration<double>(
    std::chrono::steady_clock::now().time_since_epoch()).count();
}

// Adds its lifetime to one bucket.
struct ProfileScope {
  double* bucket;
  double start;
  explicit ProfileScope(double* b)
    : bucket(g_profile.enabled ? b : nullptr), start(bucket ? nowSecs() : 0.0) {}
  ~ProfileScope() { if(bucket) *bucket += nowSecs() - start; }
};

// ---------------------------------------------------------------------------
// Per-forward-pass context handed to every block.
// mask is [batchSize][nnYLen*nnXLen] (1.0f on-board, 0.0f padding).
// ---------------------------------------------------------------------------
struct ForwardCtx {
  int batchSize;
  int nnXLen;
  int nnYLen;
  const float* mask;
  const float* maskSum;
  Arena* arena;
  const std::map<const void*, RopeTables>* ropeTables;
  RyzenAIMatMul::Accel* accel;  // may be null: then everything stays on the CPU
};

// ---------------------------------------------------------------------------
// Scalar activation. Mirrors eigenbackend.cpp exactly, including the
// numerically-stable softplus formulation log1p(exp(min(x,20)))+(max(x,20)-20).
// ---------------------------------------------------------------------------
inline float softplusForMish(float x) {
  float lo = x < 20.0f ? x : 20.0f;
  float hi = x > 20.0f ? x : 20.0f;
  return log1pf(expf(lo)) + (hi - 20.0f);
}

inline float applyActivation(int activation, float x) {
  switch(activation) {
    case ACTIVATION_IDENTITY: return x;
    case ACTIVATION_RELU: return x > 0.0f ? x : 0.0f;
    case ACTIVATION_MISH: return x * tanhf(softplusForMish(x));
    case ACTIVATION_SILU: return x / (1.0f + expf(-x));
    // x * tanh(softplus(8x)); not used by the Eigen backend (fp32 only there),
    // but supported here so scale8-transformed descs can also be referenced.
    case ACTIVATION_MISH_SCALE8: return x * tanhf(softplusForMish(8.0f * x));
    default: throw StringError("RyzenAIRef: unsupported activation " + std::to_string(activation));
  }
}

void applyActivationInplace(float* data, size_t numElts, int activation) {
  if(activation == ACTIVATION_IDENTITY)
    return;
  for(size_t i = 0; i < numElts; i++)
    data[i] = applyActivation(activation, data[i]);
}

// ---------------------------------------------------------------------------
// Direct zero-padded cross-correlation, NHWC in and out.
// Weight layout (from ConvLayerDesc): w[((oc*inC + ic)*ky + dy)*kx + dx].
// out[n][y][x][oc] = sum_{ic,dy,dx} in[n][y+(dy-ky/2)*dilY][x+(dx-kx/2)*dilX][ic] * w[...]
// ---------------------------------------------------------------------------
void convNHWC(
  float* out,
  const float* in,
  const ConvLayerDesc& desc,
  int batchSize,
  int nnXLen,
  int nnYLen,
  bool accumulate
) {
  const int inC = desc.inChannels;
  const int outC = desc.outChannels;
  const int kx = desc.convXSize;
  const int ky = desc.convYSize;
  const int padX = kx / 2;
  const int padY = ky / 2;
  const int dilX = desc.dilationX;
  const int dilY = desc.dilationY;
  const float* w = desc.weights.data();
  const size_t kernelPosStride = (size_t)ky * kx; // stride of ic within one oc

  for(int n = 0; n < batchSize; n++) {
    for(int y = 0; y < nnYLen; y++) {
      for(int x = 0; x < nnXLen; x++) {
        size_t outBase = (((size_t)n * nnYLen + y) * nnXLen + x) * outC;
        for(int oc = 0; oc < outC; oc++) {
          const float* wo = w + (size_t)oc * inC * kernelPosStride;
          float acc = 0.0f;
          for(int dy = 0; dy < ky; dy++) {
            int iy = y + (dy - padY) * dilY;
            if(iy < 0 || iy >= nnYLen)
              continue;
            for(int dx = 0; dx < kx; dx++) {
              int ix = x + (dx - padX) * dilX;
              if(ix < 0 || ix >= nnXLen)
                continue;
              size_t inBase = (((size_t)n * nnYLen + iy) * nnXLen + ix) * inC;
              const float* wi = wo + (size_t)dy * kx + dx;
              for(int ic = 0; ic < inC; ic++)
                acc += in[inBase + ic] * wi[(size_t)ic * kernelPosStride];
            }
          }
          if(accumulate)
            out[outBase + oc] += acc;
          else
            out[outBase + oc] = acc;
        }
      }
    }
  }
}

// ---------------------------------------------------------------------------
// BatchNorm (using precomputed mergedScale/mergedBias) + activation, NHWC.
// Positions where mask != 1.0f are zeroed (exactly as eigenbackend's select).
// ---------------------------------------------------------------------------
void batchNormActNHWC(
  float* out,
  const float* in,
  const BatchNormLayerDesc& bn,
  int activation,
  const float* mask,
  int batchSize,
  int nnXLen,
  int nnYLen
) {
  ProfileScope profileScope(&g_profile.norms);
  const int C = bn.numChannels;
  const size_t S = (size_t)nnXLen * nnYLen;
  for(int n = 0; n < batchSize; n++) {
    for(size_t s = 0; s < S; s++) {
      size_t base = ((size_t)n * S + s) * C;
      if(mask[(size_t)n * S + s] == 1.0f) {
        for(int c = 0; c < C; c++)
          out[base + c] = applyActivation(activation, in[base + c] * bn.mergedScale[c] + bn.mergedBias[c]);
      }
      else {
        for(int c = 0; c < C; c++)
          out[base + c] = 0.0f;
      }
    }
  }
}

// ---------------------------------------------------------------------------
// Lightweight per-position RMSNorm used inside transformer blocks
// (weight only, no bias, no activation). Masked positions are zeroed.
// ---------------------------------------------------------------------------
void transformerRMSNormNHWC(
  float* out,
  const float* in,
  const TransformerRMSNormDesc& desc,
  const float* mask,
  int batchSize,
  int nnXLen,
  int nnYLen
) {
  ProfileScope profileScope(&g_profile.norms);
  const int C = desc.numChannels;
  const size_t S = (size_t)nnXLen * nnYLen;
  for(int n = 0; n < batchSize; n++) {
    for(size_t s = 0; s < S; s++) {
      size_t base = ((size_t)n * S + s) * C;
      if(mask[(size_t)n * S + s] == 0.0f) {
        for(int c = 0; c < C; c++)
          out[base + c] = 0.0f;
        continue;
      }
      float sumSq = 0.0f;
      for(int c = 0; c < C; c++) {
        float v = in[base + c];
        sumSq += v * v;
      }
      float rms = 1.0f / sqrtf(sumSq / (float)C + desc.epsilon);
      for(int c = 0; c < C; c++)
        out[base + c] = in[base + c] * rms * desc.weight[c];
    }
  }
}

// ---------------------------------------------------------------------------
// Full RMSNorm for the trunk tip (gamma+beta+activation, spatial or
// non-spatial). Direct port of RMSNormLayer in eigenbackend.cpp.
// Non-spatial: normalize across channels per position.
// Spatial: normalize across channels AND all valid (masked-in) positions.
// ---------------------------------------------------------------------------
void trunkTipRMSNormNHWC(
  float* out,
  const float* in,
  const RMSNormLayerDesc& desc,
  int activation,
  const float* mask,
  int batchSize,
  int nnXLen,
  int nnYLen
) {
  const int C = desc.numChannels;
  const size_t S = (size_t)nnXLen * nnYLen;
  if(!desc.spatial) {
    for(int n = 0; n < batchSize; n++) {
      for(size_t s = 0; s < S; s++) {
        size_t base = ((size_t)n * S + s) * C;
        if(mask[(size_t)n * S + s] == 0.0f) {
          for(int c = 0; c < C; c++)
            out[base + c] = 0.0f;
          continue;
        }
        float sumSq = 0.0f;
        for(int c = 0; c < C; c++) {
          float v = in[base + c];
          sumSq += v * v;
        }
        float rms = 1.0f / sqrtf(sumSq / (float)C + desc.epsilon);
        for(int c = 0; c < C; c++)
          out[base + c] = applyActivation(activation, in[base + c] * rms * desc.gamma[c] + desc.beta[c]);
      }
    }
  }
  else {
    for(int n = 0; n < batchSize; n++) {
      float sumSq = 0.0f;
      size_t count = 0;
      for(size_t s = 0; s < S; s++) {
        if(mask[(size_t)n * S + s] == 0.0f)
          continue;
        size_t base = ((size_t)n * S + s) * C;
        for(int c = 0; c < C; c++) {
          float v = in[base + c];
          sumSq += v * v;
        }
        count++;
      }
      float totalElts = (float)count * (float)C;
      float rms = 1.0f / sqrtf(sumSq / totalElts + desc.epsilon);
      for(size_t s = 0; s < S; s++) {
        size_t base = ((size_t)n * S + s) * C;
        if(mask[(size_t)n * S + s] == 0.0f) {
          for(int c = 0; c < C; c++)
            out[base + c] = 0.0f;
          continue;
        }
        for(int c = 0; c < C; c++)
          out[base + c] = applyActivation(activation, in[base + c] * rms * desc.gamma[c] + desc.beta[c]);
      }
    }
  }
}

// ---------------------------------------------------------------------------
// Dense layer over rows: out[r][oc] = sum_ic in[r][ic] * W[ic*outC + oc].
// Used for all MatMulLayerDesc applications. The "rows" are either batch
// rows ([N][C]) or flattened batch*spatial rows ([N*H*W][C]).
// ---------------------------------------------------------------------------
void matmulRows(float* out, const float* in, const MatMulLayerDesc& desc, int numRows) {
  const int inC = desc.inChannels;
  const int outC = desc.outChannels;
  const float* w = desc.weights.data();
  for(int r = 0; r < numRows; r++) {
    const float* inRow = in + (size_t)r * inC;
    float* outRow = out + (size_t)r * outC;
    for(int oc = 0; oc < outC; oc++)
      outRow[oc] = 0.0f;
    for(int ic = 0; ic < inC; ic++) {
      float iv = inRow[ic];
      const float* wRow = w + (size_t)ic * outC;
      for(int oc = 0; oc < outC; oc++)
        outRow[oc] += iv * wRow[oc];
    }
  }
}

// Every convolution is offered to the NPU: 1x1 directly as a dense layer,
// larger kernels via implicit GEMM (the accelerator gathers the input patches
// itself). Anything it declines falls through to the direct convolution.
void convNHWCMaybeNpu(
  ForwardCtx& ctx,
  float* out,
  const float* in,
  const ConvLayerDesc& desc,
  int batchSize,
  int nnXLen,
  int nnYLen,
  bool accumulate
) {
  if(RyzenAIMatMul::tryConv(ctx.accel, out, in, desc, batchSize, nnXLen, nnYLen, accumulate))
    return;
  convNHWC(out, in, desc, batchSize, nnXLen, nnYLen, accumulate);
}

// Same, but offers the layer to the NPU first. Used only for the trunk's
// board-sized dense layers; the heads' batch-row matmuls are far too small to
// repay a dispatch and call matmulRows directly.
void matmulRowsMaybeNpu(
  ForwardCtx& ctx, float* out, const float* in, const MatMulLayerDesc& desc, int numRows) {
  if(RyzenAIMatMul::tryMatmul(ctx.accel, out, in, desc, numRows))
    return;
  matmulRows(out, in, desc, numRows);
}

// Projections that share an input go to the NPU as one fused GEMM; if the
// accelerator declines, each runs the ordinary maybe-NPU path on its own.
void matmulRowsMultiMaybeNpu(
  ForwardCtx& ctx,
  float* const* outs,
  const float* in,
  const MatMulLayerDesc* const* descs,
  int numDescs,
  int numRows
) {
  if(RyzenAIMatMul::tryMatmulMulti(ctx.accel, outs, in, descs, numDescs, numRows))
    return;
  for(int j = 0; j < numDescs; j++)
    matmulRowsMaybeNpu(ctx, outs[j], in, *descs[j], numRows);
}

// BatchNorm+activation, offered to the NPU first when it's Mish on a full
// board (the fused op handles exactly that; anything else falls through).
void bnActMaybeNpu(
  ForwardCtx& ctx,
  float* out,
  const float* in,
  const BatchNormLayerDesc& bn,
  int activation,
  int batchSize,
  int nnXLen,
  int nnYLen
) {
  const size_t S = (size_t)nnXLen * nnYLen;
  bool fullBoard = true;
  for(int n = 0; n < batchSize; n++)
    if(ctx.maskSum[n] != (float)S) {
      fullBoard = false;
      break;
    }
  if(fullBoard &&
     RyzenAIMatMul::tryBnMish(ctx.accel, out, in, bn, activation, batchSize * (int)S))
    return;
  batchNormActNHWC(out, in, bn, activation, ctx.mask, batchSize, nnXLen, nnYLen);
}

// io[r][c] += bias[c]
void matBiasAddRows(float* io, const MatBiasLayerDesc& desc, int numRows) {
  const int C = desc.numChannels;
  for(int r = 0; r < numRows; r++) {
    float* row = io + (size_t)r * C;
    for(int c = 0; c < C; c++)
      row[c] += desc.weights[c];
  }
}

// io[n][y][x][c] += bias[n][c]   (addNCBiasInplace in eigenbackend.cpp)
void addPerBatchChannelBiasNHWC(
  float* io,
  const float* bias,
  int C,
  int batchSize,
  int nnXLen,
  int nnYLen
) {
  const size_t S = (size_t)nnXLen * nnYLen;
  for(int n = 0; n < batchSize; n++) {
    const float* biasN = bias + (size_t)n * C;
    for(size_t s = 0; s < S; s++) {
      float* row = io + ((size_t)n * S + s) * C;
      for(int c = 0; c < C; c++)
        row[c] += biasN[c];
    }
  }
}

// ---------------------------------------------------------------------------
// Global pooling for gpool residual blocks / policy head.
// in: [N][H][W][C] (must be zero at masked-out positions, which holds because
// this only ever follows a masked BN+activation), out: [N][3*C] with
//   [c]     = mean over valid positions (denominator = maskSum)
//   [C+c]   = mean * (sqrt(maskSum) - 14) * 0.1
//   [2*C+c] = max over valid positions
// Direct port of poolRowsGPool in eigenbackend.cpp.
// ---------------------------------------------------------------------------
void poolRowsGPoolNHWC(
  float* out,
  const float* in,
  const float* mask,
  const float* maskSum,
  int C,
  int batchSize,
  int nnXLen,
  int nnYLen
) {
  const size_t S = (size_t)nnXLen * nnYLen;
  for(int n = 0; n < batchSize; n++) {
    const float* inN = in + (size_t)n * S * C;
    const float* maskN = mask + (size_t)n * S;
    float* outN = out + (size_t)n * 3 * C;
    for(int c = 0; c < C; c++) {
      float s = 0.0f;
      float m = -1.0f;
      for(size_t xy = 0; xy < S; xy++) {
        float x = inN[xy * C + c];
        s += x;
        // Init to -1.0 and +(mask-1.0) makes padded space effectively -1.0,
        // below anything current activations produce (padded inputs are 0).
        float xm = x + (maskN[xy] - 1.0f);
        if(xm > m)
          m = xm;
      }
      float div = maskSum[n];
      float sqrtdiv = sqrtf(div);
      float mean = s / div;
      outN[c] = mean;
      outN[C + c] = mean * (sqrtdiv - 14.0f) * 0.1f;
      outN[2 * C + c] = m;
    }
  }
}

// ---------------------------------------------------------------------------
// Global pooling for the value head. out: [N][3*C] with
//   [c]     = mean
//   [C+c]   = mean * (sqrt(maskSum) - 14) * 0.1
//   [2*C+c] = mean * ((sqrt(maskSum) - 14)^2 * 0.01 - 0.1)
// Direct port of poolRowsValueHead in eigenbackend.cpp.
// ---------------------------------------------------------------------------
void poolRowsValueHeadNHWC(
  float* out,
  const float* in,
  const float* maskSum,
  int C,
  int batchSize,
  int nnXLen,
  int nnYLen
) {
  const size_t S = (size_t)nnXLen * nnYLen;
  for(int n = 0; n < batchSize; n++) {
    const float* inN = in + (size_t)n * S * C;
    float* outN = out + (size_t)n * 3 * C;
    for(int c = 0; c < C; c++) {
      float s = 0.0f;
      for(size_t xy = 0; xy < S; xy++)
        s += inN[xy * C + c];
      float div = maskSum[n];
      float sqrtdiv = sqrtf(div);
      float mean = s / div;
      outN[c] = mean;
      outN[C + c] = mean * (sqrtdiv - 14.0f) * 0.1f;
      outN[2 * C + c] = mean * ((sqrtdiv - 14.0f) * (sqrtdiv - 14.0f) * 0.01f - 0.1f);
    }
  }
}

// ---------------------------------------------------------------------------
// Blocks. `trunk` is the [N][H][W][C] residual stream, updated in place.
// `trunkScratch` is a caller-provided buffer of the same shape used for
// norm outputs / projection results. All other intermediates come from the
// arena and are released (rewound) before the block returns.
// ---------------------------------------------------------------------------
void applyBlockStack(
  const std::vector<std::pair<int, unique_ptr_void>>& blocks,
  int numBlocks,
  ForwardCtx& ctx,
  float* trunk,
  float* trunkScratch
);

// ORDINARY_BLOCK_KIND
void applyResidualBlock(
  const ResidualBlockDesc& desc,
  ForwardCtx& ctx,
  float* trunk,
  float* trunkScratch
) {
  const int N = ctx.batchSize;
  const int W = ctx.nnXLen;
  const int H = ctx.nnYLen;
  const size_t BXY = (size_t)N * W * H;
  const int midC = desc.regularConv.outChannels;

  size_t mark = ctx.arena->mark();
  float* midIn = ctx.arena->alloc(BXY * midC);
  float* midScratch = ctx.arena->alloc(BXY * midC);

  bnActMaybeNpu(ctx, trunkScratch, trunk, desc.preBN, desc.preActivation.activation, N, W, H);
  convNHWCMaybeNpu(ctx, midIn, trunkScratch, desc.regularConv, N, W, H, false);
  bnActMaybeNpu(ctx, midScratch, midIn, desc.midBN, desc.midActivation.activation, N, W, H);
  convNHWCMaybeNpu(ctx, trunk, midScratch, desc.finalConv, N, W, H, true);

  ctx.arena->rewind(mark);
}

// GLOBAL_POOLING_BLOCK_KIND
void applyGlobalPoolingResidualBlock(
  const GlobalPoolingResidualBlockDesc& desc,
  ForwardCtx& ctx,
  float* trunk,
  float* trunkScratch
) {
  const int N = ctx.batchSize;
  const int W = ctx.nnXLen;
  const int H = ctx.nnYLen;
  const size_t BXY = (size_t)N * W * H;
  const int regC = desc.regularConv.outChannels;
  const int gpoolC = desc.gpoolConv.outChannels;

  size_t mark = ctx.arena->mark();
  float* regularOut = ctx.arena->alloc(BXY * regC);
  float* regularScratch = ctx.arena->alloc(BXY * regC);
  float* gpoolOut = ctx.arena->alloc(BXY * gpoolC);
  float* gpoolOut2 = ctx.arena->alloc(BXY * gpoolC);
  float* gpoolConcat = ctx.arena->alloc((size_t)N * 3 * gpoolC);
  float* gpoolBias = ctx.arena->alloc((size_t)N * regC);

  bnActMaybeNpu(ctx, trunkScratch, trunk, desc.preBN, desc.preActivation.activation, N, W, H);
  convNHWCMaybeNpu(ctx, regularOut, trunkScratch, desc.regularConv, N, W, H, false);
  convNHWCMaybeNpu(ctx, gpoolOut, trunkScratch, desc.gpoolConv, N, W, H, false);
  bnActMaybeNpu(ctx, gpoolOut2, gpoolOut, desc.gpoolBN, desc.gpoolActivation.activation, N, W, H);
  poolRowsGPoolNHWC(gpoolConcat, gpoolOut2, ctx.mask, ctx.maskSum, gpoolC, N, W, H);
  matmulRows(gpoolBias, gpoolConcat, desc.gpoolToBiasMul, N);
  addPerBatchChannelBiasNHWC(regularOut, gpoolBias, regC, N, W, H);
  bnActMaybeNpu(ctx, regularScratch, regularOut, desc.midBN, desc.midActivation.activation, N, W, H);
  convNHWCMaybeNpu(ctx, trunk, regularScratch, desc.finalConv, N, W, H, true);

  ctx.arena->rewind(mark);
}

// NESTED_BOTTLENECK_BLOCK_KIND (inner stack may contain any block kind)
void applyNestedBottleneckResidualBlock(
  const NestedBottleneckResidualBlockDesc& desc,
  ForwardCtx& ctx,
  float* trunk,
  float* trunkScratch
) {
  const int N = ctx.batchSize;
  const int W = ctx.nnXLen;
  const int H = ctx.nnYLen;
  const size_t BXY = (size_t)N * W * H;
  const int bottleC = desc.preConv.outChannels;
  size_t mark = ctx.arena->mark();
  float* midIn = ctx.arena->alloc(BXY * bottleC);
  float* midScratch = ctx.arena->alloc(BXY * bottleC);

  bnActMaybeNpu(ctx, trunkScratch, trunk, desc.preBN, desc.preActivation.activation, N, W, H);
  convNHWCMaybeNpu(ctx, midIn, trunkScratch, desc.preConv, N, W, H, false);
  // The inner block stack treats midIn as its residual stream and midScratch
  // as its scratch.
  applyBlockStack(desc.blocks, desc.numBlocks, ctx, midIn, midScratch);
  bnActMaybeNpu(ctx, midScratch, midIn, desc.postBN, desc.postActivation.activation, N, W, H);
  convNHWCMaybeNpu(ctx, trunk, midScratch, desc.postConv, N, W, H, true);

  ctx.arena->rewind(mark);
}

// TRANSFORMER_ATTENTION_BLOCK_KIND
// Multi-head attention with grouped-query KV heads, optional RoPE (learnable
// or fixed), masked softmax. Direct port of TransformerAttentionBlock in
// eigenbackend.cpp.
void applyTransformerAttentionBlock(
  const TransformerAttentionDesc& desc,
  ForwardCtx& ctx,
  float* trunk,
  float* trunkScratch
) {
  const int N = ctx.batchSize;
  const int W = ctx.nnXLen;
  const int H = ctx.nnYLen;
  const int S = W * H;
  const size_t BXY = (size_t)N * S;
  const int inC = desc.qProj.inChannels;
  const int numHeads = desc.numHeads;
  const int numKVHeads = desc.numKVHeads;
  const int qHeadDim = desc.qHeadDim;
  const int vHeadDim = desc.vHeadDim;
  const int qTot = numHeads * qHeadDim;
  const int kTot = numKVHeads * qHeadDim;
  const int vTot = numKVHeads * vHeadDim;
  const int oTot = numHeads * vHeadDim;

  const RopeTables* rope = nullptr;
  if(desc.useRope) {
    std::map<const void*, RopeTables>::const_iterator it = ctx.ropeTables->find((const void*)&desc);
    if(it == ctx.ropeTables->end())
      throw StringError("RyzenAIRef: missing RoPE tables for attention block " + desc.name);
    rope = &it->second;
  }

  size_t mark = ctx.arena->mark();
  float* qBuf = ctx.arena->alloc(BXY * qTot);
  float* kBuf = ctx.arena->alloc(BXY * kTot);
  float* vBuf = ctx.arena->alloc(BXY * vTot);
  float* attnOut = ctx.arena->alloc(BXY * oTot);
  float* scores = ctx.arena->alloc((size_t)S * S);

  // Step 1: pre-norm (per-position RMSNorm), masked.
  transformerRMSNormNHWC(trunkScratch, trunk, desc.preLN, ctx.mask, N, W, H);

  // Step 2: Q/K/V projections; the [N][H][W][C] tensors are treated as
  // [N*S][C] row matrices (C innermost, so rows are contiguous).
  {
    float* qkvOuts[3] = {qBuf, kBuf, vBuf};
    const MatMulLayerDesc* qkvDescs[3] = {&desc.qProj, &desc.kProj, &desc.vProj};
    matmulRowsMultiMaybeNpu(ctx, qkvOuts, trunkScratch, qkvDescs, 3, N * S);
  }

  // Step 3: RoPE on Q and K, rotating channel pairs (2p, 2p+1).
  {
    ProfileScope ropeScope(&g_profile.rope);
  if(desc.useRope) {
    const int ropeNumPairs = qHeadDim / 2;
    for(int which = 0; which < 2; which++) {
      float* data = which == 0 ? qBuf : kBuf;
      const int numBufHeads = which == 0 ? numHeads : numKVHeads;
      const int totalDim = which == 0 ? qTot : kTot;
      for(int n = 0; n < N; n++) {
        for(int h = 0; h < numBufHeads; h++) {
          // For Q heads, map to the corresponding KV head; for K heads, identity.
          const int kvh = h * numKVHeads / numBufHeads;
          for(int xy = 0; xy < S; xy++) {
            size_t rowBase = ((size_t)n * S + xy) * totalDim + (size_t)h * qHeadDim;
            for(int p = 0; p < ropeNumPairs; p++) {
              size_t tableIdx;
              if(desc.learnableRope)
                tableIdx = ((size_t)kvh * ropeNumPairs + p) * S + xy;
              else
                tableIdx = (size_t)p * S + xy;
              float cosVal = rope->cosTable[tableIdx];
              float sinVal = rope->sinTable[tableIdx];
              size_t i0 = rowBase + 2 * p;
              float x0 = data[i0];
              float x1 = data[i0 + 1];
              data[i0] = x0 * cosVal - x1 * sinVal;
              data[i0 + 1] = x0 * sinVal + x1 * cosVal;
            }
          }
        }
      }
    }
  }

  }  // ropeScope

  // Step 4: masked scaled dot-product attention, per (batch, head).
  {
    // Offer the two matmuls (QK^T and P*V) to the NPU first. The softmax
    // between them stays on the CPU either way; whatever the accelerator
    // declines runs the original CPU loops below, byte-for-byte unchanged
    // (and deliberately left at their original indentation).
    //
    // Profiling note: the NPU path's time is accounted inside matmul.cpp
    // (pack/uploadB/dispatch/unpack/softmax), so the attnScores bucket must
    // only wrap the CPU loops -- wrapping the dispatch too would double-count
    // it against engineTimings' execute window.
    double npuSoftmaxSecs = 0.0;
    const bool attnOnNpu = RyzenAIMatMul::tryAttention(
      ctx.accel, attnOut, qBuf, kBuf, vBuf, ctx.mask, N, S,
      numHeads, numKVHeads, qHeadDim, vHeadDim,
      g_profile.enabled ? &npuSoftmaxSecs : nullptr);
    g_profile.softmax += npuSoftmaxSecs;
    if(!attnOnNpu) {
    ProfileScope profileScope(&g_profile.attnScores);
    const float scale = 1.0f / sqrtf((float)qHeadDim);
    const int kvGroupSize = numHeads / numKVHeads;
    for(int n = 0; n < N; n++) {
      const float* maskN = ctx.mask + (size_t)n * S;
      for(int h = 0; h < numHeads; h++) {
        const int kvh = h / kvGroupSize;
        const float* qHead = qBuf + (size_t)n * S * qTot + (size_t)h * qHeadDim;
        const float* kHead = kBuf + (size_t)n * S * kTot + (size_t)kvh * qHeadDim;
        const float* vHead = vBuf + (size_t)n * S * vTot + (size_t)kvh * vHeadDim;
        float* outHead = attnOut + (size_t)n * S * oTot + (size_t)h * vHeadDim;

        // scores[qi*S + ki] = softmax over valid ki of (scale * <Q[qi], K[ki]>).
        // Rows of masked-out queries are exactly 0 (as in eigenbackend).
        for(int qi = 0; qi < S; qi++) {
          float* scoreRow = scores + (size_t)qi * S;
          if(maskN[qi] == 0.0f) {
            for(int ki = 0; ki < S; ki++)
              scoreRow[ki] = 0.0f;
            continue;
          }
          const float* qRow = qHead + (size_t)qi * qTot;
          float maxVal = -1e30f;
          for(int ki = 0; ki < S; ki++) {
            if(maskN[ki] == 0.0f) {
              scoreRow[ki] = 0.0f;
              continue;
            }
            const float* kRow = kHead + (size_t)ki * kTot;
            float acc = 0.0f;
            for(int d = 0; d < qHeadDim; d++)
              acc += qRow[d] * kRow[d];
            acc *= scale;
            scoreRow[ki] = acc;
            if(acc > maxVal)
              maxVal = acc;
          }
          {
            ProfileScope softmaxScope(&g_profile.softmax);
            float sumExp = 0.0f;
            for(int ki = 0; ki < S; ki++) {
              if(maskN[ki] == 0.0f)
                continue;
              float e = expf(scoreRow[ki] - maxVal);
              scoreRow[ki] = e;
              sumExp += e;
            }
            float invSum = 1.0f / sumExp;
            for(int ki = 0; ki < S; ki++) {
              if(maskN[ki] != 0.0f)
                scoreRow[ki] *= invSum;
            }
          }
        }

        // attnOut[qi, h*vHeadDim + dv] = sum_ki scores[qi,ki] * V[ki, kvh*vHeadDim + dv]
        for(int qi = 0; qi < S; qi++) {
          const float* scoreRow = scores + (size_t)qi * S;
          float* outRow = outHead + (size_t)qi * oTot;
          for(int dv = 0; dv < vHeadDim; dv++)
            outRow[dv] = 0.0f;
          for(int ki = 0; ki < S; ki++) {
            float wgt = scoreRow[ki];
            if(wgt == 0.0f)
              continue; // exact: 0-weight terms contribute nothing
            const float* vRow = vHead + (size_t)ki * vTot;
            for(int dv = 0; dv < vHeadDim; dv++)
              outRow[dv] += wgt * vRow[dv];
          }
        }
      }
    }
    }  // !attnOnNpu
  }

  // Step 5: output projection back to trunk channels.
  matmulRowsMaybeNpu(ctx, trunkScratch, attnOut, desc.outProj, N * S);

  // Step 6: residual add, masked (padded positions stay untouched).
  ProfileScope residualScope(&g_profile.residual);
  for(int n = 0; n < N; n++) {
    const float* maskN = ctx.mask + (size_t)n * S;
    for(int xy = 0; xy < S; xy++) {
      float maskVal = maskN[xy];
      size_t base = ((size_t)n * S + xy) * inC;
      for(int c = 0; c < inC; c++)
        trunk[base + c] += trunkScratch[base + c] * maskVal;
    }
  }

  ctx.arena->rewind(mark);
}

// TRANSFORMER_FFN_BLOCK_KIND (SwiGLU form; useSwiGLU is validated at
// workspace creation time, matching the Eigen backend's restriction).
void applyTransformerFFNBlock(
  const TransformerFFNDesc& desc,
  ForwardCtx& ctx,
  float* trunk,
  float* trunkScratch
) {
  const int N = ctx.batchSize;
  const int W = ctx.nnXLen;
  const int H = ctx.nnYLen;
  const int S = W * H;
  const size_t BXY = (size_t)N * S;
  const int C = desc.numChannels;
  const int ffnC = desc.ffnChannels;
    return;

  size_t mark = ctx.arena->mark();
  float* ffnBuf = ctx.arena->alloc(BXY * ffnC);
  float* gateBuf = ctx.arena->alloc(BXY * ffnC);

  // Step 1: pre-norm, masked.
  transformerRMSNormNHWC(trunkScratch, trunk, desc.preLN, ctx.mask, N, W, H);

  // Step 2/3: SwiGLU = silu(linear1(x)) * linearGate(x), applied to all rows.
  // The NPU path fuses both projections into one GEMM whose epilogue applies
  // the silu and the multiply on chip, so ffnBuf comes back holding the
  // finished SwiGLU output and the elementwise loop below is skipped.
  const bool swigluOnNpu = RyzenAIMatMul::tryMatmulSwiglu(
    ctx.accel, ffnBuf, trunkScratch, desc.linear1, desc.linearGate, N * S);
  if(!swigluOnNpu) {
    {
      float* ffnOuts[2] = {ffnBuf, gateBuf};
      const MatMulLayerDesc* ffnDescs[2] = {&desc.linear1, &desc.linearGate};
      matmulRowsMultiMaybeNpu(ctx, ffnOuts, trunkScratch, ffnDescs, 2, N * S);
    }
    ProfileScope swigluScope(&g_profile.swiglu);
    const size_t total = BXY * ffnC;
    for(size_t i = 0; i < total; i++) {
      float a = ffnBuf[i];
      float siluA = a / (1.0f + expf(-a));
      ffnBuf[i] = siluA * gateBuf[i];
    }
  }

  // Step 4: down projection back to trunk channels.
  matmulRowsMaybeNpu(ctx, trunkScratch, ffnBuf, desc.linear2, N * S);

  // Step 5: residual add, masked.
  ProfileScope residualScope(&g_profile.residual);
  for(int n = 0; n < N; n++) {
    const float* maskN = ctx.mask + (size_t)n * S;
    for(int xy = 0; xy < S; xy++) {
      float maskVal = maskN[xy];
      size_t base = ((size_t)n * S + xy) * C;
      for(int c = 0; c < C; c++)
        trunk[base + c] += trunkScratch[base + c] * maskVal;
    }
  }

  ctx.arena->rewind(mark);
}

void applySingleBlock(
  int kind,
  const void* descPtr,
  ForwardCtx& ctx,
  float* trunk,
  float* trunkScratch
) {
  switch(kind) {
    case ORDINARY_BLOCK_KIND:
      applyResidualBlock(*(const ResidualBlockDesc*)descPtr, ctx, trunk, trunkScratch);
      break;
    case GLOBAL_POOLING_BLOCK_KIND:
      applyGlobalPoolingResidualBlock(*(const GlobalPoolingResidualBlockDesc*)descPtr, ctx, trunk, trunkScratch);
      break;
    case NESTED_BOTTLENECK_BLOCK_KIND:
      applyNestedBottleneckResidualBlock(*(const NestedBottleneckResidualBlockDesc*)descPtr, ctx, trunk, trunkScratch);
      break;
    case TRANSFORMER_ATTENTION_BLOCK_KIND:
      applyTransformerAttentionBlock(*(const TransformerAttentionDesc*)descPtr, ctx, trunk, trunkScratch);
      break;
    case TRANSFORMER_FFN_BLOCK_KIND:
      applyTransformerFFNBlock(*(const TransformerFFNDesc*)descPtr, ctx, trunk, trunkScratch);
      break;
    default:
      throw StringError("RyzenAIRef: unknown block kind " + std::to_string(kind));
  }
}

void applyBlockStack(
  const std::vector<std::pair<int, unique_ptr_void>>& blocks,
  int numBlocks,
  ForwardCtx& ctx,
  float* trunk,
  float* trunkScratch
) {
  for(int i = 0; i < numBlocks; i++)
    applySingleBlock(blocks[i].first, blocks[i].second.get(), ctx, trunk, trunkScratch);
}

// ---------------------------------------------------------------------------
// SGF metadata encoder: mul1+bias1+act1, mul2+bias2+act2, mul3.
// input: [N][numInputMetaChannels], output: [N][trunkNumChannels].
// ---------------------------------------------------------------------------
void applySGFMetadataEncoder(
  const SGFMetadataEncoderDesc& desc,
  ForwardCtx& ctx,
  const float* input,
  float* output
) {
  const int N = ctx.batchSize;
  size_t mark = ctx.arena->mark();
  float* internal1 = ctx.arena->alloc((size_t)N * desc.mul1.outChannels);
  float* internal2 = ctx.arena->alloc((size_t)N * desc.mul2.outChannels);

  matmulRows(internal1, input, desc.mul1, N);
  matBiasAddRows(internal1, desc.bias1, N);
  applyActivationInplace(internal1, (size_t)N * desc.mul1.outChannels, desc.act1.activation);
  matmulRows(internal2, internal1, desc.mul2, N);
  matBiasAddRows(internal2, desc.bias2, N);
  applyActivationInplace(internal2, (size_t)N * desc.mul2.outChannels, desc.act2.activation);
  matmulRows(output, internal2, desc.mul3, N);

  ctx.arena->rewind(mark);
}

// ---------------------------------------------------------------------------
// Policy head. trunk: [N][H][W][trunkC] (post tip-norm).
// policy: [N][H][W][numPolicyChannels] logits, policyPass: [N][numPolicyChannels].
// ---------------------------------------------------------------------------
void applyPolicyHead(
  const PolicyHeadDesc& desc,
  ForwardCtx& ctx,
  const float* trunk,
  float* policyPass,
  float* policy
) {
  const int N = ctx.batchSize;
  const int W = ctx.nnXLen;
  const int H = ctx.nnYLen;
  const size_t BXY = (size_t)N * W * H;
  const int p1C = desc.p1Conv.outChannels;
  const int g1C = desc.g1Conv.outChannels;

  size_t mark = ctx.arena->mark();
  float* p1Out = ctx.arena->alloc(BXY * p1C);
  float* p1Out2 = ctx.arena->alloc(BXY * p1C);
  float* g1Out = ctx.arena->alloc(BXY * g1C);
  float* g1Out2 = ctx.arena->alloc(BXY * g1C);
  float* g1Concat = ctx.arena->alloc((size_t)N * 3 * g1C);
  float* g1Bias = ctx.arena->alloc((size_t)N * p1C);
  float* p1Pass = ctx.arena->alloc((size_t)N * p1C);

  convNHWCMaybeNpu(ctx, p1Out, trunk, desc.p1Conv, N, W, H, false);
  convNHWCMaybeNpu(ctx, g1Out, trunk, desc.g1Conv, N, W, H, false);
  batchNormActNHWC(g1Out2, g1Out, desc.g1BN, desc.g1Activation.activation, ctx.mask, N, W, H);
  poolRowsGPoolNHWC(g1Concat, g1Out2, ctx.mask, ctx.maskSum, g1C, N, W, H);
  matmulRows(g1Bias, g1Concat, desc.gpoolToBiasMul, N);
  addPerBatchChannelBiasNHWC(p1Out, g1Bias, p1C, N, W, H);
  batchNormActNHWC(p1Out2, p1Out, desc.p1BN, desc.p1Activation.activation, ctx.mask, N, W, H);
  // Raw logits; intentionally NOT masked after the final conv (as eigenbackend).
  convNHWCMaybeNpu(ctx, policy, p1Out2, desc.p2Conv, N, W, H, false);

  if(desc.modelVersion >= 15) {
    matmulRows(p1Pass, g1Concat, desc.gpoolToPassMul, N);
    matBiasAddRows(p1Pass, desc.gpoolToPassBias, N);
    applyActivationInplace(p1Pass, (size_t)N * p1C, desc.passActivation.activation);
    matmulRows(policyPass, p1Pass, desc.gpoolToPassMul2, N);
  }
  else {
    matmulRows(policyPass, g1Concat, desc.gpoolToPassMul, N);
  }

  ctx.arena->rewind(mark);
}

// ---------------------------------------------------------------------------
// Value head. trunk: [N][H][W][trunkC] (post tip-norm).
// value: [N][numValueChannels] logits, scoreValue: [N][numScoreValueChannels]
// raw, ownership: [N][H][W][numOwnershipChannels] pre-tanh, unmasked.
// ---------------------------------------------------------------------------
void applyValueHead(
  const ValueHeadDesc& desc,
  ForwardCtx& ctx,
  const float* trunk,
  float* value,
  float* scoreValue,
  float* ownership
) {
  const int N = ctx.batchSize;
  const int W = ctx.nnXLen;
  const int H = ctx.nnYLen;
  const size_t BXY = (size_t)N * W * H;
  const int v1C = desc.v1Conv.outChannels;
  const int v2C = desc.v2Mul.outChannels;

  size_t mark = ctx.arena->mark();
  float* v1Out = ctx.arena->alloc(BXY * v1C);
  float* v1Out2 = ctx.arena->alloc(BXY * v1C);
  float* v1Mean = ctx.arena->alloc((size_t)N * 3 * v1C);
  float* v2Out = ctx.arena->alloc((size_t)N * v2C);

  convNHWCMaybeNpu(ctx, v1Out, trunk, desc.v1Conv, N, W, H, false);
  batchNormActNHWC(v1Out2, v1Out, desc.v1BN, desc.v1Activation.activation, ctx.mask, N, W, H);
  poolRowsValueHeadNHWC(v1Mean, v1Out2, ctx.maskSum, v1C, N, W, H);
  matmulRows(v2Out, v1Mean, desc.v2Mul, N);
  matBiasAddRows(v2Out, desc.v2Bias, N);
  applyActivationInplace(v2Out, (size_t)N * v2C, desc.v2Activation.activation);
  matmulRows(value, v2Out, desc.v3Mul, N);
  matBiasAddRows(value, desc.v3Bias, N);
  matmulRows(scoreValue, v2Out, desc.sv3Mul, N);
  matBiasAddRows(scoreValue, desc.sv3Bias, N);
  // Raw conv output; intentionally NOT masked or tanh'd (as eigenbackend).
  convNHWCMaybeNpu(ctx, ownership, v1Out2, desc.vOwnershipConv, N, W, H, false);

  ctx.arena->rewind(mark);
}

// ---------------------------------------------------------------------------
// Worst-case arena sizing (in floats), mirroring exactly the buffers each
// code path above holds simultaneously. batchXY = maxBatchSize*nnXLen*nnYLen.
// ---------------------------------------------------------------------------
size_t arenaEltsForBlockStack(
  const std::vector<std::pair<int, unique_ptr_void>>& blocks,
  size_t batchXY,
  size_t batch,
  size_t seqLen
);

size_t arenaEltsForBlock(
  int kind,
  const void* descPtr,
  size_t batchXY,
  size_t batch,
  size_t seqLen
) {
  switch(kind) {
    case ORDINARY_BLOCK_KIND: {
      const ResidualBlockDesc* d = (const ResidualBlockDesc*)descPtr;
      return 2 * (size_t)d->regularConv.outChannels * batchXY;
    }
    case GLOBAL_POOLING_BLOCK_KIND: {
      const GlobalPoolingResidualBlockDesc* d = (const GlobalPoolingResidualBlockDesc*)descPtr;
      size_t regC = (size_t)d->regularConv.outChannels;
      size_t gpoolC = (size_t)d->gpoolConv.outChannels;
      return (2 * regC + 2 * gpoolC) * batchXY + (3 * gpoolC + regC) * batch;
    }
    case NESTED_BOTTLENECK_BLOCK_KIND: {
      const NestedBottleneckResidualBlockDesc* d = (const NestedBottleneckResidualBlockDesc*)descPtr;
      // Outer midIn/midScratch stay live while the inner stack runs.
      return 2 * (size_t)d->preConv.outChannels * batchXY +
             arenaEltsForBlockStack(d->blocks, batchXY, batch, seqLen);
    }
    case TRANSFORMER_ATTENTION_BLOCK_KIND: {
      const TransformerAttentionDesc* d = (const TransformerAttentionDesc*)descPtr;
      size_t qTot = (size_t)d->numHeads * d->qHeadDim;
      size_t kTot = (size_t)d->numKVHeads * d->qHeadDim;
      size_t vTot = (size_t)d->numKVHeads * d->vHeadDim;
      size_t oTot = (size_t)d->numHeads * d->vHeadDim;
      return (qTot + kTot + vTot + oTot) * batchXY + seqLen * seqLen;
    }
    case TRANSFORMER_FFN_BLOCK_KIND: {
      const TransformerFFNDesc* d = (const TransformerFFNDesc*)descPtr;
      return 2 * (size_t)d->ffnChannels * batchXY;
    }
    default:
      throw StringError("RyzenAIRef: unknown block kind " + std::to_string(kind));
  }
}

size_t arenaEltsForBlockStack(
  const std::vector<std::pair<int, unique_ptr_void>>& blocks,
  size_t batchXY,
  size_t batch,
  size_t seqLen
) {
  size_t maxElts = 0;
  for(size_t i = 0; i < blocks.size(); i++)
    maxElts = std::max(maxElts, arenaEltsForBlock(blocks[i].first, blocks[i].second.get(), batchXY, batch, seqLen));
  return maxElts;
}

// ---------------------------------------------------------------------------
// Model validation + RoPE table precomputation (recurses into nested
// bottleneck blocks). Throws on unsupported features.
// ---------------------------------------------------------------------------
void validateAndCollectBlocks(
  const std::vector<std::pair<int, unique_ptr_void>>& blocks,
  int nnXLen,
  int nnYLen,
  std::map<const void*, RopeTables>& ropeTables
) {
  for(size_t i = 0; i < blocks.size(); i++) {
    int kind = blocks[i].first;
    const void* ptr = blocks[i].second.get();
    if(kind == ORDINARY_BLOCK_KIND || kind == GLOBAL_POOLING_BLOCK_KIND) {
      // nothing special
    }
    else if(kind == NESTED_BOTTLENECK_BLOCK_KIND) {
      const NestedBottleneckResidualBlockDesc* d = (const NestedBottleneckResidualBlockDesc*)ptr;
      validateAndCollectBlocks(d->blocks, nnXLen, nnYLen, ropeTables);
    }
    else if(kind == TRANSFORMER_ATTENTION_BLOCK_KIND) {
      const TransformerAttentionDesc* d = (const TransformerAttentionDesc*)ptr;
      if(d->useRope) {
        RopeTables tables;
        // paddedNNXYLen == nnXLen*nnYLen (no extra padding), as eigenbackend.
        d->computeRopeCosSin(nnXLen, nnYLen, nnXLen * nnYLen, tables.cosTable, tables.sinTable);
        ropeTables[ptr] = std::move(tables);
      }
    }
    else if(kind == TRANSFORMER_FFN_BLOCK_KIND) {
      const TransformerFFNDesc* d = (const TransformerFFNDesc*)ptr;
      if(!d->useSwiGLU)
        throw StringError("RyzenAIRef: non-SwiGLU transformer FFN block '" + d->name + "' is not supported");
    }
    else {
      throw StringError("RyzenAIRef: unknown block kind " + std::to_string(kind));
    }
  }
}

}  // namespace

// ---------------------------------------------------------------------------
// Workspace
// ---------------------------------------------------------------------------
struct Workspace {
  const ModelDesc* model; // not owned; must outlive this Workspace
  int maxBatchSize;
  int nnXLen;
  int nnYLen;

  std::vector<float> mask;     // [maxBatchSize][nnYLen*nnXLen]
  std::vector<float> maskSum;  // [maxBatchSize]
  std::vector<float> trunkA;   // [maxBatchSize][nnYLen*nnXLen][trunkC]
  std::vector<float> trunkB;   // same; scratch / heads' input
  std::vector<float> arenaStorage;
  std::map<const void*, RopeTables> ropeTables;
  RyzenAIMatMul::Accel* accel = nullptr;  // not owned
};

Workspace* createWorkspace(
  const ModelDesc& model,
  int maxBatchSize,
  int nnXLen,
  int nnYLen
) {
  if(maxBatchSize < 1)
    throw StringError("RyzenAIRef: maxBatchSize must be positive");
  if(nnXLen < 1 || nnYLen < 1)
    throw StringError("RyzenAIRef: nnXLen/nnYLen must be positive");

  std::unique_ptr<Workspace> ws(new Workspace());
  ws->model = &model;
  ws->maxBatchSize = maxBatchSize;
  ws->nnXLen = nnXLen;
  ws->nnYLen = nnYLen;

  const TrunkDesc& trunk = model.trunk;
  const size_t S = (size_t)nnXLen * nnYLen;
  const size_t B = (size_t)maxBatchSize;
  const size_t BXY = B * S;
  const size_t trunkC = (size_t)trunk.trunkNumChannels;

  // Validates all blocks (recursively) and precomputes RoPE tables.
  validateAndCollectBlocks(trunk.blocks, nnXLen, nnYLen, ws->ropeTables);

  // Arena sizing, mirroring the live buffers of each top-level phase.
  // Trunk preamble: initial matmul output (+ SGF metadata encoder internals).
  size_t preambleElts = (size_t)trunk.initialMatMul.outChannels * B;
  if(trunk.metaEncoderVersion > 0) {
    const SGFMetadataEncoderDesc& enc = trunk.sgfMetadataEncoder;
    size_t encInternal = (size_t)std::max(enc.mul1.outChannels, enc.mul2.outChannels);
    preambleElts += 2 * encInternal * B;
  }
  size_t trunkElts = std::max(preambleElts, arenaEltsForBlockStack(trunk.blocks, BXY, B, S));

  const PolicyHeadDesc& ph = model.policyHead;
  size_t p1C = (size_t)ph.p1Conv.outChannels;
  size_t g1C = (size_t)ph.g1Conv.outChannels;
  size_t policyElts = (2 * p1C + 2 * g1C) * BXY + (3 * g1C + 2 * p1C) * B;

  const ValueHeadDesc& vh = model.valueHead;
  size_t v1C = (size_t)vh.v1Conv.outChannels;
  size_t v2C = (size_t)vh.v2Mul.outChannels;
  size_t valueElts = 2 * v1C * BXY + (3 * v1C + v2C) * B;

  size_t arenaElts = std::max(trunkElts, std::max(policyElts, valueElts));

  ws->mask.resize(B * S);
  ws->maskSum.resize(B);
  ws->trunkA.resize(BXY * trunkC);
  ws->trunkB.resize(BXY * trunkC);
  ws->arenaStorage.resize(arenaElts);
  return ws.release();
}

void setProfileEnabled(bool enabled) {
  g_profile = CpuProfile();
  g_profile.enabled = enabled;
}

std::string profileReport() {
  if(!g_profile.enabled)
    return "RyzenAI CPU profile: not enabled";
  // attnScores covers only the CPU attention path and includes that path's
  // softmax, so subtract -- but only what the CPU path itself contributed,
  // which is exactly the softmax bucket when attention ran on the CPU and
  // zero when it ran on the NPU (the NPU path's softmax is reported there).
  char buf[320];
  std::snprintf(
    buf, sizeof(buf),
    "RyzenAI CPU profile: attention(CPU) %.2f s, softmax %.2f s, norms %.2f s, "
    "rope %.2f s, swiglu %.2f s, residual %.2f s",
    g_profile.attnScores, g_profile.softmax, g_profile.norms,
    g_profile.rope, g_profile.swiglu, g_profile.residual);
  return std::string(buf);
}

void setMatMulAccel(Workspace* workspace, RyzenAIMatMul::Accel* accel) {
  workspace->accel = accel;
}

void freeWorkspace(Workspace* workspace) {
  delete workspace;
}

void forward(
  Workspace* workspace,
  int batchSize,
  const float* spatialInput,
  const float* globalInput,
  const float* metaInput,
  float* policy,
  float* policyPass,
  float* value,
  float* scoreValue,
  float* ownership
) {
  if(workspace == nullptr)
    throw StringError("RyzenAIRef: forward called with null workspace");
  const ModelDesc& model = *workspace->model;
  if(batchSize < 1 || batchSize > workspace->maxBatchSize)
    throw StringError("RyzenAIRef: batchSize out of range for workspace");
  if(spatialInput == nullptr || globalInput == nullptr)
    throw StringError("RyzenAIRef: spatialInput and globalInput must be non-null");
  const bool hasMeta = model.trunk.metaEncoderVersion > 0;
  if(hasMeta && metaInput == nullptr)
    throw StringError("RyzenAIRef: model has an SGF metadata encoder but metaInput is null");
  if(!hasMeta && metaInput != nullptr)
    throw StringError("RyzenAIRef: model has no SGF metadata encoder but metaInput is non-null");
  if(policy == nullptr || policyPass == nullptr || value == nullptr || scoreValue == nullptr || ownership == nullptr)
    throw StringError("RyzenAIRef: output buffers must be non-null");

  const int N = batchSize;
  const int W = workspace->nnXLen;
  const int H = workspace->nnYLen;
  const int S = W * H;
  const int inC = model.numInputChannels;

  // Mask = channel 0 of the spatial input (1.0f on-board, 0.0f padding),
  // matching eigenbackend's `*mask = input->chip(0,0)` + computeMaskSum.
  float* mask = workspace->mask.data();
  float* maskSum = workspace->maskSum.data();
  for(int n = 0; n < N; n++) {
    float s = 0.0f;
    for(int xy = 0; xy < S; xy++) {
      float mv = spatialInput[((size_t)n * S + xy) * inC];
      mask[(size_t)n * S + xy] = mv;
      s += mv;
    }
    maskSum[n] = s;
  }

  Arena arena;
  arena.base = workspace->arenaStorage.data();
  arena.capacity = workspace->arenaStorage.size();
  arena.offset = 0;

  ForwardCtx ctx;
  ctx.batchSize = N;
  ctx.nnXLen = W;
  ctx.nnYLen = H;
  ctx.mask = mask;
  ctx.maskSum = maskSum;
  ctx.arena = &arena;
  ctx.ropeTables = &workspace->ropeTables;
  ctx.accel = workspace->accel;

  float* trunkA = workspace->trunkA.data(); // residual stream
  float* trunkB = workspace->trunkB.data(); // trunk scratch / heads' input

  const TrunkDesc& trunk = model.trunk;

  // ---- Trunk preamble: initial conv + global projection (+ SGF metadata) ----
  {
    size_t mark = arena.mark();
    const int immC = trunk.initialMatMul.outChannels;
    float* immOut = arena.alloc((size_t)N * immC);
    convNHWCMaybeNpu(ctx, trunkA, spatialInput, trunk.initialConv, N, W, H, false);
    matmulRows(immOut, globalInput, trunk.initialMatMul, N);
    addPerBatchChannelBiasNHWC(trunkA, immOut, immC, N, W, H);
    if(hasMeta) {
      applySGFMetadataEncoder(trunk.sgfMetadataEncoder, ctx, metaInput, immOut);
      addPerBatchChannelBiasNHWC(trunkA, immOut, immC, N, W, H);
    }
    arena.rewind(mark);
  }

  // ---- Trunk blocks (residual stream in trunkA, trunkB as block scratch) ----
  applyBlockStack(trunk.blocks, trunk.numBlocks, ctx, trunkA, trunkB);

  // ---- Trunk tip norm + activation: trunkA -> trunkB ----
  if(trunk.trunkNormKind == TRUNK_NORM_KIND_STANDARD) {
    batchNormActNHWC(trunkB, trunkA, trunk.trunkTipBN, trunk.trunkTipActivation.activation, mask, N, W, H);
  }
  else {
    trunkTipRMSNormNHWC(trunkB, trunkA, trunk.trunkTipRMSNorm, trunk.trunkTipActivation.activation, mask, N, W, H);
  }

  // ---- Heads, reading the normalized trunk output in trunkB ----
  applyPolicyHead(model.policyHead, ctx, trunkB, policyPass, policy);
  applyValueHead(model.valueHead, ctx, trunkB, value, scoreValue, ownership);
}

}  // namespace RyzenAIRef
