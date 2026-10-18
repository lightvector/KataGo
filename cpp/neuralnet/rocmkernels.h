// Declarations for ROCm-only GPU kernels (rocmkernels.hip). Unlike the kernels shared with the
// CUDA backend in cudaandrocmhelpers.inc, these are written directly against AMDGPU features -
// v_dot2_f32_f16 and SGPR (scalar-cache) broadcast operands - to make the RDNA archs that have
// no matrix cores (and no CK FMHA kernels) fast, so they have no CUDA counterpart.

#ifndef NEURALNET_ROCMKERNELS_H_
#define NEURALNET_ROCMKERNELS_H_

#include "../neuralnet/rocmincludes.h"

// Whether the device architecture has v_dot2_f32_f16, which every kernel here is built on. Takes
// hipDeviceProp_t::gcnArchName (e.g. "gfx1030" or "gfx90a:sramecc+:xnack-"). The archs without it
// (GCN gfx900-class and RDNA1 gfx1010/gfx1013) get empty kernel bodies, so callers must check this.
bool rocmArchHasDot2(const char* gcnArchName);
// Whether rocmUseDot2Attention = auto enables the dot2 attention kernels on this arch: only RDNA2
// (gfx1030-gfx1036), where they were developed and measured (on gfx1030). Other archs with
// v_dot2_f32_f16 run them only with rocmUseDot2Attention = true.
bool rocmDot2AttentionAutoArch(const char* gcnArchName);

//==============================================================================================
// dot2 scaled dot product attention
//==============================================================================================

// How the attention prep kernel applies rotary position embeddings to Q and K, mirroring the
// customCudaApplyRoPE* launchers in cudaandrocmhelpers.h.
struct RocmAttnRope {
  enum Mode { NONE = 0, TABLE = 1, RECOMPUTE = 2 };
  Mode mode = NONE;
  // TABLE: cos/sin tables, (numPairs, seqLen) if !learnableTable, else (numKVHeads, numPairs, seqLen).
  const half* cosTable = nullptr;
  const half* sinTable = nullptr;
  bool learnableTable = false;
  // RECOMPUTE: per-head frequencies (numKVHeads, numPairs, 2), cos/sin recomputed from board x/y.
  const float* freqs = nullptr;
  int nnXLen = 0;
};

// Whether the dot2 attention kernels support this head configuration.
bool rocmDot2AttentionSupportsShape(int numHeads, int numKVHeads, int qHeadDim, int vHeadDim);

// Scratch sizes for the prep kernel's outputs: the compacted K rows (also the size of the
// key-pair-interleaved V rows), and the per-batch-element position permutation.
size_t rocmDot2AttentionKVBytes(int batchSize, int seqLen, int numKVHeads, int headDim);
size_t rocmDot2AttentionPermBytes(int batchSize, int seqLen);

// Prep: applies RoPE to Q in place (on-board positions only, the only ones attention reads),
// writes RoPE'd K rows compacted over on-board positions into kc and the V rows compacted and
// interleaved in key pairs into vc, and the permutation of positions (on-board first) plus the
// on-board count into perm. Q/K/V are BSHD with per-token row strides qStride/kvStride (halfs).
// mask (can be null) is [batchSize, seqLen].
void rocmDot2AttentionPrep(
  half* q, const half* k, const half* v, const half* mask,
  half* kc, half* vc, short* perm,
  int batchSize, int seqLen, int numHeads, int numKVHeads, int headDim, int qStride, int kvStride,
  const RocmAttnRope& rope, hipStream_t stream
);

// Attention over the prep kernel's outputs. output is BSHD with row stride numHeads*headDim, and
// off-board query rows are written as zeros. Same semantics as customCudaFlashAttention.
// numMultiprocessors and waveSize (hipDeviceProp_t::multiProcessorCount and warpSize) decide
// whether small batches split the keys over more waves.
void rocmDot2Attention(
  const half* q, const half* kc, const half* vc, const short* perm, half* output,
  int batchSize, int seqLen, int numHeads, int numKVHeads, int headDim, int qStride,
  int numMultiprocessors, int waveSize, hipStream_t stream
);

#endif  // NEURALNET_ROCMKERNELS_H_
