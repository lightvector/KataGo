// Declarations for the GPU kernel launchers shared between the CUDA backend (cudahelpers.cu)
// and the ROCm backend (rocmhelpers.hip), whose implementations live in cudaandrocmhelpers.inc.
// Never included directly. It is reached via cudahelpers.h or rocmhelpers.h, which provide the
// vendor headers (for the half type) and activations.h first.

#ifndef NEURALNET_CUDAANDROCMHELPERS_H_
#define NEURALNET_CUDAANDROCMHELPERS_H_

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

//Given a tensor, add another tensor element-wise to it (same shape).
void customCudaAddTensorInplace(float* buf, const float* biases, int n);
void customCudaAddTensorInplace(half* buf, const half* biases, int n);

//Batched transpose between NCHW [n,c,xy] and NHWC [n,xy,c] layouts. Used by the ROCm backend to
//convert the uploaded input tensor to the model's layout before the initial convolution when the
//two differ, since MIOpen (unlike cuDNN) requires all tensors of a convolution to share one
//layout. The kernel bodies are only compiled on ROCm, but the declarations are harmless on CUDA.
void customCudaCopyNCHWtoNHWC(const float* in, float* out, int nSize, int cSize, int xySize);
void customCudaCopyNCHWtoNHWC(const half* in, half* out, int nSize, int cSize, int xySize);
void customCudaCopyNHWCtoNCHW(const float* in, float* out, int nSize, int cSize, int xySize);
void customCudaCopyNHWCtoNCHW(const half* in, half* out, int nSize, int cSize, int xySize);

//Whether the half-precision kernels in this compilation unit have real bodies. Always true for
//CUDA, where half kernel bodies are gated per-arch on __CUDA_ARCH__ and the backend refuses FP16
//at runtime on archs below compute capability 5.3. On ROCm it is false if the build system did
//not define HIP_SUPPORTS_FP16, in which case the backend must not run FP16.
bool customCudaFp16KernelsCompiled();
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

//==============================================================================================
// Transformer support kernels
//==============================================================================================

//Apply rotary position embeddings in-place to a BSHD-laid-out Q or K buffer.
//buf: [totalDim, seqLen*batchSize] column-major (totalDim = numBufHeads*qHeadDim).
//cosTable/sinTable: (numPairs, seqLen) if !learnableRope, else (numKVHeads, numPairs, seqLen).
void customCudaApplyRoPE(
  float* buf, const float* cosTable, const float* sinTable,
  int batchSize, int seqLen, int numBufHeads, int numKVHeads, int qHeadDim, int numPairs, bool learnableRope
);
void customCudaApplyRoPE(
  half* buf, const half* cosTable, const half* sinTable,
  int batchSize, int seqLen, int numBufHeads, int numKVHeads, int qHeadDim, int numPairs, bool learnableRope
);
//Table-free learnable RoPE: recomputes cos/sin in-kernel from per-head frequencies (numKVHeads, numPairs, 2) flattened.
void customCudaApplyRoPELearnableRecompute(
  float* buf, const float* freqs,
  int batchSize, int seqLen, int numBufHeads, int numKVHeads, int qHeadDim, int numPairs, int nnXLen
);
void customCudaApplyRoPELearnableRecompute(
  half* buf, const float* freqs,
  int batchSize, int seqLen, int numBufHeads, int numKVHeads, int qHeadDim, int numPairs, int nnXLen
);

//Scaled dot product attention (online-softmax, tiled). Q/K/V/output are BSHD row-major.
//mask (can be null) is [batchSize, seqLen].
void customCudaFlashAttention(
  const float* Q, const float* K, const float* V, const float* mask, float* output,
  int batchSize, int seqLen, int numHeads, int numKVHeads, int qHeadDim, int vHeadDim
);
void customCudaFlashAttention(
  const half* Q, const half* K, const half* V, const half* mask, half* output,
  int batchSize, int seqLen, int numHeads, int numKVHeads, int qHeadDim, int vHeadDim
);

//SwiGLU: out[i] = SiLU(a[i]) * b[i]
void customCudaSwiGLU(const float* a, const float* b, float* out, int size);
void customCudaSwiGLU(const half* a, const half* b, half* out, int size);

//Convert mask [batchSize, seqLen] (0/1) into a fully-materialized additive attention bias of shape
//[batchSize, seqLen, seqLen]: bias[b,q,k] = (mask[b,k] != 0 ? 0 : -3e4). Used to feed KataGo's
//mask into the fused attention paths - cudnn-frontend graph SDPA on CUDA (as a [B,1,S,S] bias)
//and Composable Kernel (CK) FMHA on ROCm - which (unlike the plain kernel) need the bias pre-materialized rather
//than taking the raw per-position mask. See the kernel comment in cudaandrocmhelpers.inc for
//why -3e4 masks exactly even in FP16.
void customCudaMaskToAttnBiasFull(const float* mask, float* outBias, int batchSize, int seqLen);
void customCudaMaskToAttnBiasFull(const half* mask, half* outBias, int batchSize, int seqLen);

//Masked residual add: trunk[i] += residual[i] * mask[spatial_idx]. mask can be null (treated as all ones).
//NCHW: trunk/residual [n,c,xy], mask [n,xy]. NHWC: trunk/residual [n,xy,c], mask [n,xy].
void customCudaMaskedResidualAddNCHW(float* trunk, const float* residual, const float* mask, int nSize, int cSize, int xySize);
void customCudaMaskedResidualAddNCHW(half* trunk, const half* residual, const half* mask, int nSize, int cSize, int xySize);
void customCudaMaskedResidualAddNHWC(float* trunk, const float* residual, const float* mask, int nSize, int xySize, int cSize);
void customCudaMaskedResidualAddNHWC(half* trunk, const half* residual, const half* mask, int nSize, int xySize, int cSize);

//RMSNorm with gamma/beta/activation, non-spatial mode (for transformer pre-norm and trunk tip).
//NHWC: input/output [n,xy,c], gamma/beta [c], mask [n,xy] (can be null).
void customCudaRMSNormGammaBetaNHWC(
  const float* in, float* out, const float* gamma, const float* beta, const float* mask,
  int nSize, int xySize, int cSize, float epsilon, int activation
);
void customCudaRMSNormGammaBetaNHWC(
  const half* in, half* out, const half* gamma, const half* beta, const half* mask,
  int nSize, int xySize, int cSize, float epsilon, int activation
);
//NCHW: input/output [n,c,xy], gamma/beta [c], mask [n,xy] (can be null).
void customCudaRMSNormGammaBetaNCHW(
  const float* in, float* out, const float* gamma, const float* beta, const float* mask,
  int nSize, int cSize, int xySize, float epsilon, int activation
);
void customCudaRMSNormGammaBetaNCHW(
  const half* in, half* out, const half* gamma, const half* beta, const half* mask,
  int nSize, int cSize, int xySize, float epsilon, int activation
);

//Spatial RMSNorm: normalizes over all C*H*W per batch element (rather than per-position over C).
//sumSqBuf must be a scratch buffer of size nSize * CUDA_SPATIAL_RMSNORM_SUMSQ_STRIDE floats.
#define CUDA_SPATIAL_RMSNORM_SUMSQ_STRIDE 9   // SPATIAL_RMSNORM_BLOCKS_PER_BATCH (8) partials + 1 final
void customCudaSpatialRMSNormNHWC(
  const float* in, float* out, const float* gamma, const float* beta, const float* mask, const float* maskSum,
  int nSize, int xySize, int cSize, float epsilon, int activation, float* sumSqBuf
);
void customCudaSpatialRMSNormNHWC(
  const half* in, half* out, const half* gamma, const half* beta, const half* mask, const float* maskSum,
  int nSize, int xySize, int cSize, float epsilon, int activation, float* sumSqBuf
);
void customCudaSpatialRMSNormNCHW(
  const float* in, float* out, const float* gamma, const float* beta, const float* mask, const float* maskSum,
  int nSize, int cSize, int xySize, float epsilon, int activation, float* sumSqBuf
);
void customCudaSpatialRMSNormNCHW(
  const half* in, half* out, const half* gamma, const half* beta, const half* mask, const float* maskSum,
  int nSize, int cSize, int xySize, float epsilon, int activation, float* sumSqBuf
);

#endif  // NEURALNET_CUDAANDROCMHELPERS_H_
