// Tensor-core (mma.sync) FlashAttention-style scaled dot product attention for the
// KataGo CUDA backend. Matches the semantics of customCudaFlashAttention in
// cudaandrocmhelpers.inc:
//   - Q/K/V/output are BSHD row-major FP16: element (b, s, h, d) at offset
//     ((b*seqLen + s)*numHeads + h)*headDim + d, with numKVHeads for K/V.
//   - GQA: query head h uses kv head (h * numKVHeads / numHeads).
//   - Softmax scale 1/sqrt(qHeadDim), applied to the FP32 QK^T result.
//   - mask: optional [batchSize, seqLen] FP16, NULL = no mask. mask[b*s+k] == 0 excludes
//     key position k (as if score were -inf). A query row whose own mask is 0, or whose
//     keys are all masked, outputs zeros.
//
// FlashAttention-2 style: online softmax in FP32, K/V tiled through shared memory,
// Q fragments held in registers, one warp per 16 query rows. FP32 accumulation for both
// QK^T and PV, with P (the softmax numerator) rounded to FP16 for the PV mma, as in standard
// FlashAttention implementations.
//
// FP16-accumulate variants of the QK^T and PV mmas (2x issue rate on consumer GPUs) were
// implemented and benchmarked, then removed (see git history): accuracy was fine but the
// whole-net effect was neutral to slightly negative everywhere except H100 at +1-4%, not
// worth the extra code paths and config surface.
//
// Requires sm_75+ at runtime. sm_80+ uses mma.sync.aligned.m16n8k16 f32.f16.f16.f32 with
// cp.async prefetch. sm_75 uses pairs of mma.sync.aligned.m16n8k8 and synchronous copies
// instead. For older archs the kernels compile to empty stubs, so the caller must check
// flashAttentionMmaSupportedOnCurrentDevice() before dispatching here.

#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <math_constants.h>
#include <stdint.h>
#include <math.h>

namespace flashmma {

constexpr int FMMA_BLOCK_Q = 64;   // query rows per block (4 warps x 16 rows)
constexpr int FMMA_BLOCK_KV = 64;  // key/value rows per shared memory tile
constexpr int FMMA_NWARPS = 4;

// c += a * b with a 16x16 row-major f16, b 16x8 col-major f16, c 16x8 f32.
__device__ __forceinline__ void fmmaMma16816(float c[4], const uint32_t a[4], const uint32_t b[2]) {
#if __CUDA_ARCH__ >= 800
  asm volatile(
    "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
    "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
    : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
    : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
#elif __CUDA_ARCH__ >= 750
  asm volatile(
    "mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 "
    "{%0,%1,%2,%3}, {%4,%5}, {%6}, {%0,%1,%2,%3};\n"
    : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
    : "r"(a[0]), "r"(a[1]), "r"(b[0]));
  asm volatile(
    "mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 "
    "{%0,%1,%2,%3}, {%4,%5}, {%6}, {%0,%1,%2,%3};\n"
    : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
    : "r"(a[2]), "r"(a[3]), "r"(b[1]));
#endif
}

__device__ __forceinline__ uint32_t fmmaPackHalf2(float lo, float hi) {
  __half2 h = __float22half2_rn(make_float2(lo, hi));
  return *reinterpret_cast<uint32_t*>(&h);
}

// Fast 2^x (MUFU.EX2). Maps -inf to exactly 0, which the masking below relies on.
__device__ __forceinline__ float fmmaExp2(float x) {
  float r;
  asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(r) : "f"(x));
  return r;
}

// ldmatrix: load four 8x8 f16 tiles from shared memory into mma fragment layout.
// Lane i supplies the address of row i%8 of tile i/8 (all 32 lanes must participate).
__device__ __forceinline__ void fmmaLdmatrixX4(
  uint32_t& r0, uint32_t& r1, uint32_t& r2, uint32_t& r3, const half* rowPtr
) {
#if __CUDA_ARCH__ >= 750
  uint32_t addr = (uint32_t)__cvta_generic_to_shared(rowPtr);
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
    : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(addr));
#endif
}
__device__ __forceinline__ void fmmaLdmatrixX4Trans(
  uint32_t& r0, uint32_t& r1, uint32_t& r2, uint32_t& r3, const half* rowPtr
) {
#if __CUDA_ARCH__ >= 750
  uint32_t addr = (uint32_t)__cvta_generic_to_shared(rowPtr);
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
    : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(addr));
#endif
}

// 16-byte global->shared copy, asynchronous (cp.async) on sm_80+. sm_75 has no cp.async,
// so it uses a plain synchronous copy instead and the commit/wait calls below do nothing.
// With valid == false the destination is zero-filled without touching global memory.
__device__ __forceinline__ void fmmaCpAsync16(void* smemDst, const void* gmemSrc, bool valid) {
#if __CUDA_ARCH__ >= 800
  uint32_t dst = (uint32_t)__cvta_generic_to_shared(smemDst);
  int srcSize = valid ? 16 : 0;
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" :: "r"(dst), "l"(gmemSrc), "r"(srcSize));
#elif __CUDA_ARCH__ >= 750
  uint4 val = make_uint4(0, 0, 0, 0);
  if(valid)
    val = *reinterpret_cast<const uint4*>(gmemSrc);
  *reinterpret_cast<uint4*>(smemDst) = val;
#endif
}
__device__ __forceinline__ void fmmaCpAsyncCommit() {
#if __CUDA_ARCH__ >= 800
  asm volatile("cp.async.commit_group;\n");
#elif __CUDA_ARCH__ >= 750
  // no-op
#endif
}
__device__ __forceinline__ void fmmaCpAsyncWaitAll() {
#if __CUDA_ARCH__ >= 800
  asm volatile("cp.async.wait_group 0;\n");
#elif __CUDA_ARCH__ >= 750
  // no-op
#endif
}

// Initial running row max. Large-magnitude negative rather than -inf so that when a row has
// seen no live keys yet, (rowMax - newMax) == 0 stays finite and alpha == 1. Masked scores
// are -inf, making their exp2 exactly 0, so a fully-masked row keeps sum == 0 and the final
// (sum > 0 ? 1/sum : 0) yields all-zero output, matching the scalar kernel.
#define FMMA_NEG_BIG (-1e30f)

// D = qHeadDim = vHeadDim; 32 or 64.
template<int D>
__global__ void __launch_bounds__(FMMA_NWARPS * 32)
flashAttentionMmaKernel(
  const half* __restrict__ Q, const half* __restrict__ K, const half* __restrict__ V,
  const half* __restrict__ mask, half* __restrict__ out,
  int seqLen, int numHeads, int numKVHeads, float scale, int qStride, int kvStride
) {
#if __CUDA_ARCH__ >= 750
  constexpr int BQ = FMMA_BLOCK_Q;
  constexpr int BKV = FMMA_BLOCK_KV;
  constexpr int NTHREADS = FMMA_NWARPS * 32;
  constexpr int ST = D + 8;      // smem row stride (halfs): keeps 16B store alignment and
                                 // makes all fragment loads bank-conflict-free
  constexpr int VEC = 8;         // halfs per vectorized global load (uint4)
  constexpr int KT = D / 16;     // k-tiles along head dim for QK^T
  constexpr int NT_S = BKV / 8;  // n-tiles of the score matrix (8 kv cols each)
  constexpr int KT_PV = BKV / 16;// k-tiles along kv for PV
  constexpr int NT_O = D / 8;    // n-tiles of the output (8 dim cols each)

  const int tid = threadIdx.x;
  const int warp = tid >> 5;
  const int lane = tid & 31;
  const int gr = lane >> 2;   // row within 8-row fragment group
  const int q4 = lane & 3;    // quad index -> column pairs

  const int qBlockStart = blockIdx.x * BQ;
  const int h = blockIdx.y;
  const int n = blockIdx.z;
  const int kvh = h * numKVHeads / numHeads;

  // Per-token row strides of Q and of K/V are launch parameters so that Q/K/V may be
  // interior slices of a combined [tokens, q+k+v] projection output. K and V share a stride
  // (they always come from the same buffer and Dq == Dv). The output is always packed.
  const int oTotalDim = numHeads * D;

  // Double-buffered K/V/mask tiles for cp.async prefetch. Q is staged through buffer 1's
  // K/V space (same size, only needed until the Q fragments are in registers).
  // 16-byte alignment is required by cp.async, the uint4 stores, and ldmatrix, so enforce it
  // rather than relying on declaration order and ST making it work out.
  __shared__ __align__(16) half kTiles[2][BKV * ST];
  __shared__ __align__(16) half vTiles[2][BKV * ST];
  __shared__ float maskTiles[2][BKV];
  // Per-buffer live-key ballots (one word per 32 keys). A tile whose keys are all masked
  // or out-of-range contributes nothing and is skipped entirely.
  __shared__ int anyLiveParts[2][BKV / 32];
  static_assert(BKV % 32 == 0 && NTHREADS >= BKV, "mask ballot layout");
  static_assert(BQ <= BKV, "Q staging reuses buffer 1's K tile and must fit in it");

  const half* Kbase = K + (size_t)n * seqLen * kvStride + kvh * D;
  const half* Vbase = V + (size_t)n * seqLen * kvStride + kvh * D;
  const half* maskBase = (mask != NULL) ? mask + (size_t)n * seqLen : NULL;

  constexpr int KVVECS = BKV * (D / VEC);
  const int numKvTiles = (seqLen + BKV - 1) / BKV;

  // Async-load a K/V/mask tile into the given buffer. Rows past seqLen zero-fill, and their
  // mask bias is -inf so they are excluded from the softmax regardless.
  auto issueTileLoad = [&](int tile, int buf) {
    const int kvStart = tile * BKV;
    #pragma unroll
    for(int t = tid; t < KVVECS; t += NTHREADS) {
      int row = t / (D / VEC);
      int d0 = (t % (D / VEC)) * VEC;
      int g = kvStart + row;
      bool valid = g < seqLen;
      // With valid == false the zero-fill copy should not dereference the source, but keep the
      // address in range anyway rather than pointing up to a tile past the end of K/V.
      size_t off = (size_t)(valid ? g : 0) * kvStride + d0;
      fmmaCpAsync16(&kTiles[buf][row * ST + d0], Kbase + off, valid);
      fmmaCpAsync16(&vTiles[buf][row * ST + d0], Vbase + off, valid);
    }
    // Additive bias form: 0 for a live key, -inf for a masked or out-of-range key, so
    // scale+mask is a single fma and exp2 maps masked scores to exactly 0. Warps 0 and 1
    // also record live-key ballots for the whole-tile skip.
    if(tid < BKV) {
      int g = kvStart + tid;
      bool live = g < seqLen && (maskBase == NULL || (float)maskBase[g] != 0.0f);
      maskTiles[buf][tid] = live ? 0.0f : -CUDART_INF_F;
      uint32_t ballot = __ballot_sync(0xffffffff, live);
      if((tid & 31) == 0)
        anyLiveParts[buf][tid >> 5] = (int)ballot;
    }
    fmmaCpAsyncCommit();
  };

  // Prologue: start loading kv tile 0, and stage Q into buffer 1's K space (coalesced),
  // zero-filled past seqLen.
  issueTileLoad(0, 0);
  {
    half* qTile = kTiles[1];
    constexpr int QVECS = BQ * (D / VEC);
    #pragma unroll
    for(int t = tid; t < QVECS; t += NTHREADS) {
      int row = t / (D / VEC);
      int d0 = (t % (D / VEC)) * VEC;
      int qPos = qBlockStart + row;
      uint4 val = make_uint4(0, 0, 0, 0);
      if(qPos < seqLen)
        val = *reinterpret_cast<const uint4*>(Q + ((size_t)n * seqLen + qPos) * qStride + h * D + d0);
      *reinterpret_cast<uint4*>(&qTile[row * ST + d0]) = val;
    }
  }
  __syncthreads();

  // Q fragments in registers (A operand of m16n8k16, rows warp*16 .. warp*16+15).
  uint32_t qA[KT][4];
  {
    const half* qTile = kTiles[1];
    const int r0 = warp * 16 + gr;
    #pragma unroll
    for(int kt = 0; kt < KT; kt++) {
      int d0 = kt * 16 + q4 * 2;
      qA[kt][0] = *reinterpret_cast<const uint32_t*>(&qTile[r0 * ST + d0]);
      qA[kt][1] = *reinterpret_cast<const uint32_t*>(&qTile[(r0 + 8) * ST + d0]);
      qA[kt][2] = *reinterpret_cast<const uint32_t*>(&qTile[r0 * ST + d0 + 8]);
      qA[kt][3] = *reinterpret_cast<const uint32_t*>(&qTile[(r0 + 8) * ST + d0 + 8]);
    }
  }

  const float scaleLog2 = scale * 1.4426950408889634f;  // fold log2(e) into the scale

  float sFrag[NT_S][4];      // scores, later reused as softmax numerators p
  float oFrag[NT_O][4];      // output accumulator
  float rowMax[2] = {FMMA_NEG_BIG, FMMA_NEG_BIG};  // rows gr and gr+8 of this warp's 16
  float rowSum[2] = {0.0f, 0.0f};
  #pragma unroll
  for(int i = 0; i < NT_O; i++) {
    oFrag[i][0] = 0.0f; oFrag[i][1] = 0.0f; oFrag[i][2] = 0.0f; oFrag[i][3] = 0.0f;
  }

  for(int tile = 0; tile < numKvTiles; tile++) {
    // Wait for this tile's async loads, then start prefetching the next tile. The barrier
    // also guarantees the buffer being written was fully consumed two iterations ago
    // (and, on the first iteration, that all warps have read their Q fragments).
    fmmaCpAsyncWaitAll();
    __syncthreads();
    if(tile + 1 < numKvTiles)
      issueTileLoad(tile + 1, (tile + 1) & 1);

    // Skip tiles with no live keys: they contribute nothing to max, sum, or output.
    {
      int liveBits = 0;
      #pragma unroll
      for(int i = 0; i < BKV / 32; i++)
        liveBits |= anyLiveParts[tile & 1][i];
      if(liveBits == 0)
        continue;
    }

    const half* kTile = kTiles[tile & 1];
    const half* vTile = vTiles[tile & 1];
    const float* maskTile = maskTiles[tile & 1];

    // S = Q K^T for this warp's 16 rows x BKV cols. B fragments via ldmatrix: per
    // (kt, ntp) one x4 covers score n-tiles 2*ntp and 2*ntp+1 (kv rows ntp*16..+15,
    // head-dim cols kt*16..+15).
    #pragma unroll
    for(int nt = 0; nt < NT_S; nt++) {
      sFrag[nt][0] = 0.0f; sFrag[nt][1] = 0.0f; sFrag[nt][2] = 0.0f; sFrag[nt][3] = 0.0f;
    }
    #pragma unroll
    for(int kt = 0; kt < KT; kt++) {
      #pragma unroll
      for(int ntp = 0; ntp < NT_S / 2; ntp++) {
        uint32_t b[4];
        const half* rp = &kTile[(ntp * 16 + ((lane >> 4) & 1) * 8 + (lane & 7)) * ST
                                + kt * 16 + ((lane >> 3) & 1) * 8];
        fmmaLdmatrixX4(b[0], b[1], b[2], b[3], rp);
        fmmaMma16816(sFrag[2 * ntp], qA[kt], b);
        fmmaMma16816(sFrag[2 * ntp + 1], qA[kt], b + 2);
      }
    }

    // Scale (folded with log2(e) so exp becomes exp2) and add the key mask bias, then
    // find the per-row max of this tile.
    // C fragment layout: c0=(row gr, col 2*q4), c1=(gr, 2*q4+1), c2=(gr+8, 2*q4),
    // c3=(gr+8, 2*q4+1), col offset nt*8 within the tile.
    float pm0[NT_S], pm1[NT_S];
    #pragma unroll
    for(int nt = 0; nt < NT_S; nt++) {
      float b0 = maskTile[nt * 8 + q4 * 2];
      float b1 = maskTile[nt * 8 + q4 * 2 + 1];
      float s0 = fmaf(sFrag[nt][0], scaleLog2, b0);
      float s1 = fmaf(sFrag[nt][1], scaleLog2, b1);
      float s2 = fmaf(sFrag[nt][2], scaleLog2, b0);
      float s3 = fmaf(sFrag[nt][3], scaleLog2, b1);
      sFrag[nt][0] = s0; sFrag[nt][1] = s1; sFrag[nt][2] = s2; sFrag[nt][3] = s3;
      pm0[nt] = fmaxf(s0, s1);
      pm1[nt] = fmaxf(s2, s3);
    }
    // Tree reduction keeps the dependency chain short.
    #pragma unroll
    for(int w = NT_S / 2; w >= 1; w >>= 1) {
      #pragma unroll
      for(int i = 0; i < w; i++) {
        pm0[i] = fmaxf(pm0[i], pm0[i + w]);
        pm1[i] = fmaxf(pm1[i], pm1[i + w]);
      }
    }
    float tileMax0 = pm0[0];
    float tileMax1 = pm1[0];
    // Row stats live across the 4 lanes of a quad (same gr, different q4).
    tileMax0 = fmaxf(tileMax0, __shfl_xor_sync(0xffffffff, tileMax0, 1));
    tileMax0 = fmaxf(tileMax0, __shfl_xor_sync(0xffffffff, tileMax0, 2));
    tileMax1 = fmaxf(tileMax1, __shfl_xor_sync(0xffffffff, tileMax1, 1));
    tileMax1 = fmaxf(tileMax1, __shfl_xor_sync(0xffffffff, tileMax1, 2));

    // rowMax stays >= -1e30 (its initial value) even when the whole tile is masked
    // (tileMax == -inf), so alpha is well-defined and masked scores give
    // exp2(-inf - rowMax) == 0 exactly.
    const float newMax0 = fmaxf(rowMax[0], tileMax0);
    const float newMax1 = fmaxf(rowMax[1], tileMax1);
    const float alpha0 = fmmaExp2(rowMax[0] - newMax0);
    const float alpha1 = fmmaExp2(rowMax[1] - newMax1);
    rowMax[0] = newMax0;
    rowMax[1] = newMax1;

    // p = exp2(s - rowMax). Masked entries are exactly 0 via s == -inf.
    #pragma unroll
    for(int nt = 0; nt < NT_S; nt++) {
      float p0 = fmmaExp2(sFrag[nt][0] - newMax0);
      float p1 = fmmaExp2(sFrag[nt][1] - newMax0);
      float p2 = fmmaExp2(sFrag[nt][2] - newMax1);
      float p3 = fmmaExp2(sFrag[nt][3] - newMax1);
      sFrag[nt][0] = p0; sFrag[nt][1] = p1; sFrag[nt][2] = p2; sFrag[nt][3] = p3;
      pm0[nt] = p0 + p1;
      pm1[nt] = p2 + p3;
    }
    #pragma unroll
    for(int w = NT_S / 2; w >= 1; w >>= 1) {
      #pragma unroll
      for(int i = 0; i < w; i++) {
        pm0[i] += pm0[i + w];
        pm1[i] += pm1[i + w];
      }
    }
    float tsum0 = pm0[0];
    float tsum1 = pm1[0];
    tsum0 += __shfl_xor_sync(0xffffffff, tsum0, 1);
    tsum0 += __shfl_xor_sync(0xffffffff, tsum0, 2);
    tsum1 += __shfl_xor_sync(0xffffffff, tsum1, 1);
    tsum1 += __shfl_xor_sync(0xffffffff, tsum1, 2);
    rowSum[0] = rowSum[0] * alpha0 + tsum0;
    rowSum[1] = rowSum[1] * alpha1 + tsum1;

    // Rescale the running output by alpha.
    #pragma unroll
    for(int nt = 0; nt < NT_O; nt++) {
      oFrag[nt][0] *= alpha0;
      oFrag[nt][1] *= alpha0;
      oFrag[nt][2] *= alpha1;
      oFrag[nt][3] *= alpha1;
    }

    // Convert P to FP16 A fragments. The QK^T C fragment layout maps directly onto the
    // A fragment layout: score tiles 2*kt2 and 2*kt2+1 form the 16x16 A tile for PV k-tile kt2.
    uint32_t pA[KT_PV][4];
    #pragma unroll
    for(int kt2 = 0; kt2 < KT_PV; kt2++) {
      pA[kt2][0] = fmmaPackHalf2(sFrag[2 * kt2][0], sFrag[2 * kt2][1]);
      pA[kt2][1] = fmmaPackHalf2(sFrag[2 * kt2][2], sFrag[2 * kt2][3]);
      pA[kt2][2] = fmmaPackHalf2(sFrag[2 * kt2 + 1][0], sFrag[2 * kt2 + 1][1]);
      pA[kt2][3] = fmmaPackHalf2(sFrag[2 * kt2 + 1][2], sFrag[2 * kt2 + 1][3]);
    }

    // O += P V. B fragments via transposed ldmatrix: per (c, nt) one x4 covers PV
    // k-tiles 2*c and 2*c+1 (kv rows c*32 + lane, output cols nt*8..+7).
    #pragma unroll
    for(int c = 0; c < KT_PV / 2; c++) {
      #pragma unroll
      for(int nt = 0; nt < NT_O; nt++) {
        uint32_t b[4];
        const half* rp = &vTile[(c * 32 + lane) * ST + nt * 8];
        fmmaLdmatrixX4Trans(b[0], b[1], b[2], b[3], rp);
        fmmaMma16816(oFrag[nt], pA[2 * c], b);
        fmmaMma16816(oFrag[nt], pA[2 * c + 1], b + 2);
      }
    }
  }

  // Normalize and write output. Masked query rows (and rows whose keys were all masked,
  // via rowSum == 0) write zeros, matching the scalar kernel.
  {
    const float inv0 = (rowSum[0] > 0.0f) ? (1.0f / rowSum[0]) : 0.0f;
    const float inv1 = (rowSum[1] > 0.0f) ? (1.0f / rowSum[1]) : 0.0f;
    const int r0 = qBlockStart + warp * 16 + gr;
    const int r1 = r0 + 8;
    if(r0 < seqLen) {
      float qm = (mask != NULL) ? (float)mask[(size_t)n * seqLen + r0] : 1.0f;
      float f = (qm != 0.0f) ? inv0 : 0.0f;
      half* o = out + ((size_t)n * seqLen + r0) * oTotalDim + h * D;
      #pragma unroll
      for(int nt = 0; nt < NT_O; nt++) {
        int d0 = nt * 8 + q4 * 2;
        *reinterpret_cast<uint32_t*>(o + d0) = fmmaPackHalf2(oFrag[nt][0] * f, oFrag[nt][1] * f);
      }
    }
    if(r1 < seqLen) {
      float qm = (mask != NULL) ? (float)mask[(size_t)n * seqLen + r1] : 1.0f;
      float f = (qm != 0.0f) ? inv1 : 0.0f;
      half* o = out + ((size_t)n * seqLen + r1) * oTotalDim + h * D;
      #pragma unroll
      for(int nt = 0; nt < NT_O; nt++) {
        int d0 = nt * 8 + q4 * 2;
        *reinterpret_cast<uint32_t*>(o + d0) = fmmaPackHalf2(oFrag[nt][2] * f, oFrag[nt][3] * f);
      }
    }
  }
#endif  // __CUDA_ARCH__ >= 750
}

#undef FMMA_NEG_BIG

}  // namespace flashmma

namespace flashmma {
__global__ void flashAttentionMmaSupportProbeKernel(int* out) {
#if __CUDA_ARCH__ >= 750
  *out = 1;
#else
  (void)out;
#endif
}
}  // namespace flashmma

// Whether the mma kernels as compiled (or JIT-compiled from PTX) for the current device contain
// their sm_75+ bodies. A device property check is not enough: a build whose arch list predates
// sm_75, JIT-run on a newer GPU, would launch the kernels as empty stubs and silently produce
// garbage attention output. Runs a tiny probe kernel on the calling thread's current device, so
// callers should probe once per compute handle (per-handle probing is deliberate: it evaluates
// the right device on machines mixing GPU architectures) rather than on any hot path.
inline bool flashAttentionMmaSupportedOnCurrentDevice() {
  int* devFlag = nullptr;
  if(cudaMalloc(&devFlag, sizeof(int)) != cudaSuccess)
    return false;
  int hostFlag = 0;
  bool ok = cudaMemset(devFlag, 0, sizeof(int)) == cudaSuccess;
  if(ok) {
    flashmma::flashAttentionMmaSupportProbeKernel<<<1,1>>>(devFlag);
    ok = cudaGetLastError() == cudaSuccess;
  }
  if(ok)
    ok = cudaMemcpy(&hostFlag, devFlag, sizeof(int), cudaMemcpyDeviceToHost) == cudaSuccess;
  (void)cudaFree(devFlag);
  return ok && hostFlag == 1;
}

namespace flashmma {
template<int D>
inline void launchFlashAttentionMmaForDim(
  const half* Q, const half* K, const half* V, const half* mask, half* out,
  int batchSize, int seqLen, int numHeads, int numKVHeads, float scale,
  int qStride, int kvStride, cudaStream_t stream
) {
  dim3 grid((seqLen + FMMA_BLOCK_Q - 1) / FMMA_BLOCK_Q, numHeads, batchSize);
  dim3 block(FMMA_NWARPS * 32);
  flashAttentionMmaKernel<D><<<grid, block, 0, stream>>>(Q, K, V, mask, out, seqLen, numHeads, numKVHeads, scale, qStride, kvStride);
}
}  // namespace flashmma

// Whether the mma kernels support this attention shape at all. Callers may rely on this at
// model load time to commit to layouts that assume the mma path will be taken.
inline bool flashAttentionMmaSupportsShape(int numHeads, int numKVHeads, int qHeadDim, int vHeadDim) {
  if(qHeadDim != vHeadDim)
    return false;
  if(qHeadDim != 32 && qHeadDim != 64)
    return false;
  if(numHeads <= 0 || numKVHeads <= 0)
    return false;
  if(numHeads % numKVHeads != 0)
    return false;
  if(numHeads > 65535)
    return false;
  // The kernel's GQA head mapping computes h * numKVHeads in int, so reject head counts that
  // could overflow it, far beyond any real model.
  if((long long)numHeads * numKVHeads > 2147483647LL)
    return false;
  return true;
}

// Returns false if the shape, strides, or pointer alignment is unsupported, in which case the
// caller should fall back to another implementation. The caller must also have verified
// flashAttentionMmaSupportedOnCurrentDevice() once beforehand. qStride/kvStride are the
// per-token row strides of Q and of K/V in halfs (pass numHeads*qHeadDim and
// numKVHeads*qHeadDim for packed BSHD tensors).
inline bool launchFlashAttentionMma(
  const half* Q, const half* K, const half* V, const half* mask, half* out,
  int batchSize, int seqLen, int numHeads, int numKVHeads, int qHeadDim, int vHeadDim,
  int qStride, int kvStride, cudaStream_t stream
) {
  using namespace flashmma;
  if(!flashAttentionMmaSupportsShape(numHeads, numKVHeads, qHeadDim, vHeadDim))
    return false;
  if(batchSize <= 0 || seqLen <= 0 || batchSize > 65535)
    return false;
  if(qStride < numHeads * qHeadDim || kvStride < numKVHeads * qHeadDim)
    return false;
  // Vectorized (16-byte) global loads/stores require aligned base pointers and row strides.
  // The head-dim offsets within rows are multiples of 32 halfs, so these are the only conditions.
  if((((uintptr_t)Q) | ((uintptr_t)K) | ((uintptr_t)V) | ((uintptr_t)out)) & 15)
    return false;
  if(qStride % 8 != 0 || kvStride % 8 != 0)
    return false;

  float scale = 1.0f / sqrtf((float)qHeadDim);
  if(qHeadDim == 32)
    launchFlashAttentionMmaForDim<32>(Q, K, V, mask, out, batchSize, seqLen, numHeads, numKVHeads, scale, qStride, kvStride, stream);
  else
    launchFlashAttentionMmaForDim<64>(Q, K, V, mask, out, batchSize, seqLen, numHeads, numKVHeads, scale, qStride, kvStride, stream);
  return true;
}
