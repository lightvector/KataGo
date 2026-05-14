#ifndef NEURALNET_CUDAHELPERS_H_
#define NEURALNET_CUDAHELPERS_H_

#include "../neuralnet/cudaincludes.h"
#include "../neuralnet/activations.h"

//Given two tensors with shapes inA: [n,cA,h,w] and inB: [n,cB,h,w], that are on the GPU
//Copy them into a single tensor out: [n,cA+cB,h,w] that is also allocated on the gpu
void customCudaChannelConcat(const float* inA, const float* inB, float* out, int chwA, int chwB, int n);
void customCudaChannelConcat(const half* inA, const half* inB, half* out, int chwA, int chwB, int n);

//Given a tensor [n,c,hw], extract out channel 0 to [n,hw]
void customCudaChannel0ExtractNCHW(const float* in, float* out, int n, int c, int hw);
void customCudaChannel0ExtractNCHW(const half* in, half* out, int n, int c, int hw);
//Given a tensor [n,hw,c], extract out channel 0 to [n,hw]
void customCudaChannel0ExtractNHWC(const float* in, float* out, int n, int hw, int c);
void customCudaChannel0ExtractNHWC(const half* in, half* out, int n, int hw, int c);

//Given an input tensor and an output buffer of shape [n,c], fill output buffer with sum or max over c.
void customCudaPoolRowsSumNCHW(const float* in, float* out, int nSize, int cSize, int xySize, float scaleSum);
void customCudaPoolRowsSumNHWC(const float* in, float* out, int nSize, int xySize, int cSize, float scaleSum);

//Specialized operations for value head and general global pooling. Same as the other pooling, but fusedly fills
//an output buffer of shape [n,c*3].
void customCudaValueHeadPoolNCHW(const float* in, float* out, int nSize, int cSize, int xySize, const float* maskSum);
void customCudaValueHeadPoolNHWC(const float* in, float* out, int nSize, int xySize, int cSize, const float* maskSum);
void customCudaPoolRowsGPoolNCHW(const float* in, float* out, int nSize, int cSize, int xySize, const float* mask, const float* maskSum);
void customCudaPoolRowsGPoolNHWC(const float* in, float* out, int nSize, int xySize, int cSize, const float* mask, const float* maskSum);
void customCudaPoolRowsGPoolNCHW(const half* in, half* out, int nSize, int cSize, int xySize, const half* mask, const float* maskSum);
void customCudaPoolRowsGPoolNHWC(const half* in, half* out, int nSize, int xySize, int cSize, const half* mask, const float* maskSum);

void customCudaCopyToHalf(const float* in, half* out, int n);
void customCudaCopyFromHalf(const half* in, float* out, int n);

//Given a tensor, add another tensor to it.
void customCudaAddTensorInplace(half* buf, const half* biases, int n);
//Given an input with shape [n,c] and biases of shape [c], add the biases in-place.
void customCudaAddCBiasInplaceNC(float* buf, const float* biases, int n, int c, int activation);
void customCudaAddCBiasInplaceNC(half* buf, const half* biases, int n, int c, int activation);
//Given an input with shape [n,c,xy] and biases of shape [n,c], add the biases in-place.
void customCudaAddNCBiasInplaceNCHW(float *buf, const float* biases, int nSize, int cSize, int xySize);
void customCudaAddNCBiasInplaceNCHW(half *buf, const half* biases, int nSize, int cSize, int xySize);
//Given an input with shape [n,xy,c] and biases of shape [n,c], add the biases in-place.
void customCudaAddNCBiasInplaceNHWC(float *buf, const float* biases, int nSize, int xySize, int cSize);
void customCudaAddNCBiasInplaceNHWC(half *buf, const half* biases, int nSize, int xySize, int cSize);

//Given an input with shape [n,c,xy] and scale and biases of shape [c], multiply by scale and add the biases
//Optionally also apply an activation.
//Optionally also multiply by mask (can be null), with shape [n,xy]
void customCudaApplyCScaleBiasNCHW(const float* in, float* out, const float* scale, const float* biases, const float* mask, int n, int c, int xy, int activation);
void customCudaApplyCScaleBiasNCHW(const half* in, half* out, const half* scale, const half* biases, const half* mask, int n, int c, int xy, int activation);
//Given an input with shape [n,xy,c] and scale and biases of shape [c], multiply by scale and add the biases
//Optionally also apply relu.
//Optionally also multiply by mask (can be null), with shape [n,xy]
void customCudaApplyCScaleBiasNHWC(const float* in, float* out, const float* scale, const float* biases, const float* mask, int n, int xy, int c, int activation);
void customCudaApplyCScaleBiasNHWC(const half* in, half* out, const half* scale, const half* biases, const half* mask, int n, int xy, int c, int activation);

//Apply RoPE (rotary position embeddings) in-place on Q or K buffer.
//buf has shape [totalDim, seqLen*batchSize] (column-major).
//cosTable/sinTable have shape depending on learnable: if learnable, [numKVHeads*numPairs*seqLen], else [numPairs*seqLen].
void customCudaApplyRoPE(
  float* buf, const float* cosTable, const float* sinTable,
  int batchSize, int seqLen, int numBufHeads, int numKVHeads, int qHeadDim, int numPairs, bool learnableRope);
void customCudaApplyRoPE(
  half* buf, const half* cosTable, const half* sinTable,
  int batchSize, int seqLen, int numBufHeads, int numKVHeads, int qHeadDim, int numPairs, bool learnableRope);

//Convert a [batchSize, seqLen] mask (0/1) into a fully-materialized additive attention bias of shape
//[batchSize, seqLen, seqLen] suitable for cuDNN SDPA's [B, 1, S, S] bias input:
//  bias[b, q, k] = (mask[b, k] != 0 ? 0 : -1e4).
//cuDNN doesn't currently have plans for the [B, 1, 1, S] broadcast-over-q variant for our shape,
//so we materialize the full bias. Using -1e4 (well within FP16 max ~65504) avoids -inf-minus-inf
//NaNs in cuDNN's softmax.
void customCudaMaskToAttnBiasFull(const float* mask, float* outBias, int batchSize, int seqLen);
void customCudaMaskToAttnBiasFull(const half* mask, half* outBias, int batchSize, int seqLen);

//FlashAttention-style scaled dot product attention with online softmax.
//Layout (BSHD, matching CUDA backend's Q/K/V buffers from MatMulLayer):
//  Q: [batchSize*seqLen, numHeads*qHeadDim] row-major
//     i.e. element at (n, xy, h, d) = Q[(h*qHeadDim + d) + (n*seqLen + xy)*(numHeads*qHeadDim)]
//  K: [batchSize*seqLen, numKVHeads*qHeadDim] row-major
//  V: [batchSize*seqLen, numKVHeads*vHeadDim] row-major
//  Output: [batchSize*seqLen, numHeads*vHeadDim] row-major (same layout as Q/V).
//  mask: [batchSize, seqLen] (0 means masked).
//No score-matrix materialization; output is computed via online softmax.
void customCudaFlashAttention(
  const float* Q, const float* K, const float* V, const float* mask, float* output,
  int batchSize, int seqLen, int numHeads, int numKVHeads, int qHeadDim, int vHeadDim);
void customCudaFlashAttention(
  const half* Q, const half* K, const half* V, const half* mask, half* output,
  int batchSize, int seqLen, int numHeads, int numKVHeads, int qHeadDim, int vHeadDim);

//SwiGLU: out[i] = SiLU(a[i]) * b[i], where SiLU(x) = x / (1 + exp(-x))
void customCudaSwiGLU(const float* a, const float* b, float* out, int size);
void customCudaSwiGLU(const half* a, const half* b, half* out, int size);

//Masked residual add: trunk[i] += residual[i] * mask[spatial_idx], for NHWC or NCHW layouts.
//mask has shape [n, xy].
void customCudaMaskedResidualAddNCHW(float* trunk, const float* residual, const float* mask, int nSize, int cSize, int xySize);
void customCudaMaskedResidualAddNCHW(half* trunk, const half* residual, const half* mask, int nSize, int cSize, int xySize);
void customCudaMaskedResidualAddNHWC(float* trunk, const float* residual, const float* mask, int nSize, int xySize, int cSize);
void customCudaMaskedResidualAddNHWC(half* trunk, const half* residual, const half* mask, int nSize, int xySize, int cSize);

//RMSNorm with gamma/beta and optional activation. Non-spatial: per-position across channels.
//input/output [n, xy, c] NHWC or [n, c, xy] NCHW. gamma/beta shape [c].
void customCudaRMSNormGammaBetaNHWC(
  const float* in, float* out, const float* gamma, const float* beta, const float* mask,
  int nSize, int xySize, int cSize, float epsilon, int activation);
void customCudaRMSNormGammaBetaNHWC(
  const half* in, half* out, const half* gamma, const half* beta, const half* mask,
  int nSize, int xySize, int cSize, float epsilon, int activation);
void customCudaRMSNormGammaBetaNCHW(
  const float* in, float* out, const float* gamma, const float* beta, const float* mask,
  int nSize, int cSize, int xySize, float epsilon, int activation);
void customCudaRMSNormGammaBetaNCHW(
  const half* in, half* out, const half* gamma, const half* beta, const half* mask,
  int nSize, int cSize, int xySize, float epsilon, int activation);

//Spatial RMSNorm: normalizes over all C*H*W per batch element. gamma/beta shape [c].
//Uses a deterministic multi-block reduction: many blocks per batch element compute partial sums of
//squares, then a reduce pass combines them. sumSqBuf is a pre-allocated float scratch buffer that must
//hold both the per-block partials and the final value: size [nSize * CUDA_SPATIAL_RMSNORM_SUMSQ_STRIDE].
#define CUDA_SPATIAL_RMSNORM_SUMSQ_STRIDE 9   // SPATIAL_RMSNORM_BLOCKS_PER_BATCH (8) partials + 1 final
void customCudaSpatialRMSNormNHWC(
  const float* in, float* out, const float* gamma, const float* beta, const float* mask, const float* maskSum,
  int nSize, int xySize, int cSize, float epsilon, int activation, float* sumSqBuf);
void customCudaSpatialRMSNormNHWC(
  const half* in, half* out, const half* gamma, const half* beta, const half* mask, const float* maskSum,
  int nSize, int xySize, int cSize, float epsilon, int activation, float* sumSqBuf);
void customCudaSpatialRMSNormNCHW(
  const float* in, float* out, const float* gamma, const float* beta, const float* mask, const float* maskSum,
  int nSize, int cSize, int xySize, float epsilon, int activation, float* sumSqBuf);
void customCudaSpatialRMSNormNCHW(
  const half* in, half* out, const half* gamma, const half* beta, const half* mask, const float* maskSum,
  int nSize, int cSize, int xySize, float epsilon, int activation, float* sumSqBuf);


#endif  // NEURALNET_CUDAHELPERS_H_
