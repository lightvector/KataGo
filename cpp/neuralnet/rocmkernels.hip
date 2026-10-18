// ROCm-only GPU kernels, see rocmkernels.h.

#include "../neuralnet/rocmkernels.h"

#include <stdexcept>
#include <string>

// v_dot2_f32_f16 (__builtin_amdgcn_fdot2) does not exist on these archs. Their device passes
// compile the kernels below to empty bodies, and rocmArchHasDot2 keeps the host from launching
// them there. Keep the two lists in sync.
#if defined(__gfx900__) || defined(__gfx902__) || defined(__gfx909__) || defined(__gfx90c__) \
  || defined(__gfx1010__) || defined(__gfx1013__) || defined(__gfx9_generic__) || defined(__gfx10_1_generic__)
  #define KATAGO_ROCM_DEVICE_HAS_DOT2 0
#else
  #define KATAGO_ROCM_DEVICE_HAS_DOT2 1
#endif

static std::string rkArchBaseName(const char* gcnArchName) {
  std::string arch(gcnArchName);
  size_t colonPos = arch.find(':');
  if(colonPos != std::string::npos)
    arch = arch.substr(0, colonPos);
  return arch;
}

bool rocmArchHasDot2(const char* gcnArchName) {
  const std::string arch = rkArchBaseName(gcnArchName);
  return !(
    arch == "gfx900" || arch == "gfx902" || arch == "gfx909" || arch == "gfx90c" ||
    arch == "gfx1010" || arch == "gfx1013" || arch == "gfx9-generic" || arch == "gfx10-1-generic"
  );
}

bool rocmDot2AttentionAutoArch(const char* gcnArchName) {
  const std::string arch = rkArchBaseName(gcnArchName);
  return arch.size() == 7 && arch.compare(0, 6, "gfx103") == 0 && arch[6] >= '0' && arch[6] <= '6';
}

typedef _Float16 rk_h2 __attribute__((ext_vector_type(2)));

static __device__ __forceinline__ rk_h2 rkU32ToH2(uint32_t u) { return __builtin_bit_cast(rk_h2, u); }
static __device__ __forceinline__ uint32_t rkPackF32ToH2(float a, float b) {
  return __builtin_bit_cast(uint32_t, __builtin_amdgcn_cvt_pkrtz(a, b));
}
static __device__ __forceinline__ float rkDot2(rk_h2 a, rk_h2 b, float c) {
#if KATAGO_ROCM_DEVICE_HAS_DOT2
  return __builtin_amdgcn_fdot2(a, b, c, false);
#else
  (void)a; (void)b;
  return c;
#endif
}

//==============================================================================================
// dot2 scaled dot product attention
//==============================================================================================
//
// RDNA2 has no matrix cores, so attention runs on v_dot2_f32_f16 (two FP16 products accumulated
// in FP32 per lane per cycle). Each lane owns whole query rows, which keeps the online softmax
// entirely lane-local, so every K and V value is a broadcast operand shared by the 32 lanes of a
// wave. Broadcasting through LDS or vector memory costs 32x the bytes in data-path bandwidth and
// tops out well below the dot2 rate, so K and V are instead read through the scalar cache into
// SGPRs and used directly as the scalar operand of dot2, which costs no vector bandwidth at all.
//
// That requires K and V rows contiguous per (batch element, KV head) in the order the keys are
// visited, and V interleaved in key pairs so that one dot2 accumulates two keys into one output
// channel. A prep kernel produces that layout, compacting over on-board positions so that masked
// keys (small boards in a larger buffer) cost nothing, and applies RoPE on the way, replacing the
// separate RoPE kernel.

// Kc/Vc rows per (batch element, KV head): seqLen rounded up to whole key pairs.
static __host__ __device__ inline int rkAttnPaddedSeq(int seqLen) { return (seqLen + 1) & ~1; }

bool rocmDot2AttentionSupportsShape(int numHeads, int numKVHeads, int qHeadDim, int vHeadDim) {
  return
    qHeadDim == vHeadDim && (qHeadDim == 32 || qHeadDim == 64) &&
    numKVHeads > 0 && numHeads % numKVHeads == 0;
}

size_t rocmDot2AttentionKVBytes(int batchSize, int seqLen, int numKVHeads, int headDim) {
  return (size_t)batchSize * numKVHeads * rkAttnPaddedSeq(seqLen) * headDim * sizeof(half);
}
size_t rocmDot2AttentionPermBytes(int batchSize, int seqLen) {
  // Per batch element: seqLen positions followed by the on-board count, padded to 4 bytes.
  return (size_t)batchSize * ((seqLen + 2) & ~1) * sizeof(short);
}
static __host__ __device__ inline int rkPermStride(int seqLen) { return (seqLen + 2) & ~1; }

// Block-wide ordered partition of the positions of batch element n: on-board positions in order
// into perm[0..count), off-board ones into perm[count..seqLen) (in reverse order, which is
// irrelevant to every consumer). Returns count. Requires blockDim.x == NT.
template<int NT>
static __device__ __forceinline__ int rkPartitionPositions(const half* __restrict__ mask, int n, int seqLen, short* perm) {
  __shared__ int waveCounts[NT / 32 + 1];
  const int tid = threadIdx.x;
  const int lane = __lane_id();
  const int wave = tid / warpSize;
  const int numWaves = NT / warpSize;
  int validBase = 0;
  int invalidBase = 0;
  for(int p0 = 0; p0 < seqLen; p0 += NT) {
    int pos = p0 + tid;
    bool inRange = pos < seqLen;
    bool valid = inRange && (mask == NULL || (float)mask[(size_t)n * seqLen + pos] != 0.0f);
    unsigned long long below = (1ull << lane) - 1ull;
    unsigned long long vb = __ballot(valid);
    unsigned long long ib = __ballot(inRange && !valid);
    if(lane == 0)
      waveCounts[wave] = __popcll(vb) | (__popcll(ib) << 16);
    __syncthreads();
    int validOff = validBase, invalidOff = invalidBase;
    for(int w = 0; w < numWaves; w++) {
      int c = waveCounts[w];
      if(w < wave) { validOff += c & 0xffff; invalidOff += c >> 16; }
      validBase += c & 0xffff;
      invalidBase += c >> 16;
    }
    if(valid)
      perm[validOff + __popcll(vb & below)] = (short)pos;
    else if(inRange)
      perm[seqLen - 1 - (invalidOff + __popcll(ib & below))] = (short)pos;
    __syncthreads();
  }
  return validBase;
}

// Grid (numKVHeads, batchSize), block NT. See rocmDot2AttentionPrep.
template<int D, int NT, int ROPE>
__global__ __launch_bounds__(NT)
void rkAttnPrepKernel(
  half* __restrict__ Q, const half* __restrict__ K, const half* __restrict__ V, const half* __restrict__ mask,
  half* __restrict__ Kc, uint32_t* __restrict__ Vc, short* __restrict__ permOut,
  int seqLen, int numHeads, int numKVHeads, int qStride, int kvStride, float kScale,
  const half* __restrict__ cosTable, const half* __restrict__ sinTable, int learnableTable,
  const float* __restrict__ freqs, int nnXLen
) {
#if KATAGO_ROCM_DEVICE_HAS_DOT2
  constexpr int DV4 = D / 8;   // 16-byte chunks per row
  constexpr int NUM_PAIRS = D / 2;
  extern __shared__ short perm[];
  const int kvh = blockIdx.x;
  const int n = blockIdx.y;
  const int count = rkPartitionPositions<NT>(mask, n, seqLen, perm);
  if(kvh == 0) {
    short* dst = permOut + (size_t)n * rkPermStride(seqLen);
    for(int i = threadIdx.x; i < seqLen; i += NT)
      dst[i] = perm[i];
    if(threadIdx.x == 0)
      dst[seqLen] = (short)count;
  }
  if(count == 0)
    return;

  const int SP = rkAttnPaddedSeq(seqLen);
  const size_t rowBase = ((size_t)n * numKVHeads + kvh) * SP;
  const int countP = (count + 1) & ~1;
  const int groupSize = numHeads / numKVHeads;

  // K rows (rotated and pre-scaled by kScale, folded into the one rounding to half that the
  // rotation needs anyway) and, for real slots, the rotation of the Q rows of this KV head's
  // group. The slot past an odd count repeats the last key, so it cannot change the softmax max,
  // and the attention kernel gives it zero weight.
  for(int t = threadIdx.x; t < countP * DV4; t += NT) {
    const int j = t / DV4;
    const int c = t - j * DV4;
    const bool realSlot = j < count;
    const int pos = perm[realSlot ? j : count - 1];
    float cosV[4] = {1.0f, 1.0f, 1.0f, 1.0f};
    float sinV[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    if(ROPE != 0) {
      #pragma unroll
      for(int p = 0; p < 4; p++) {
        const int pairIdx = c * 4 + p;
        if(ROPE == 1) {
          int tableIdx = learnableTable ? (kvh * NUM_PAIRS + pairIdx) * seqLen + pos : pairIdx * seqLen + pos;
          cosV[p] = __half2float(cosTable[tableIdx]);
          sinV[p] = __half2float(sinTable[tableIdx]);
        }
        else {
          int x = pos % nnXLen;
          int y = pos / nnXLen;
          float angle = (float)x * freqs[(kvh * NUM_PAIRS + pairIdx) * 2 + 0] + (float)y * freqs[(kvh * NUM_PAIRS + pairIdx) * 2 + 1];
          sincosf(angle, &sinV[p], &cosV[p]);
        }
      }
    }
    // Same arithmetic as the customCudaApplyRoPE* half kernels: rotate in float, round once.
    auto rotate = [&](uint4 in, float scale) {
      uint32_t w[4] = {in.x, in.y, in.z, in.w};
      #pragma unroll
      for(int p = 0; p < 4; p++) {
        rk_h2 x = rkU32ToH2(w[p]);
        float a = (float)x[0], b = (float)x[1];
        rk_h2 r;
        r[0] = (_Float16)((a * cosV[p] - b * sinV[p]) * scale);
        r[1] = (_Float16)((a * sinV[p] + b * cosV[p]) * scale);
        w[p] = __builtin_bit_cast(uint32_t, r);
      }
      return make_uint4(w[0], w[1], w[2], w[3]);
    };
    const uint4 kv = *(const uint4*)(K + ((size_t)n * seqLen + pos) * kvStride + kvh * D + c * 8);
    *(uint4*)(Kc + (rowBase + j) * D + c * 8) = rotate(kv, kScale);
    if(ROPE != 0 && realSlot) {
      for(int g = 0; g < groupSize; g++) {
        uint4* qp = (uint4*)(Q + ((size_t)n * seqLen + pos) * qStride + (kvh * groupSize + g) * D + c * 8);
        *qp = rotate(*qp, 1.0f);
      }
    }
  }

  // V rows, interleaved in key pairs: word d of pair row jp holds (V[key 2jp][d], V[key 2jp+1][d]).
  // The slot past an odd count gets a zero V row.
  for(int t = threadIdx.x; t < (countP / 2) * DV4; t += NT) {
    const int jp = t / DV4;
    const int c = t - jp * DV4;
    const int key0 = 2 * jp;
    uint4 a = *(const uint4*)(V + ((size_t)n * seqLen + perm[key0]) * kvStride + kvh * D + c * 8);
    uint4 b = make_uint4(0, 0, 0, 0);
    if(key0 + 1 < count)
      b = *(const uint4*)(V + ((size_t)n * seqLen + perm[key0 + 1]) * kvStride + kvh * D + c * 8);
    uint4 lo, hi;
    lo.x = (a.x & 0xffffu) | (b.x << 16); lo.y = (a.x >> 16) | (b.x & 0xffff0000u);
    lo.z = (a.y & 0xffffu) | (b.y << 16); lo.w = (a.y >> 16) | (b.y & 0xffff0000u);
    hi.x = (a.z & 0xffffu) | (b.z << 16); hi.y = (a.z >> 16) | (b.z & 0xffff0000u);
    hi.z = (a.w & 0xffffu) | (b.w << 16); hi.w = (a.w >> 16) | (b.w & 0xffff0000u);
    uint4* dst = (uint4*)(Vc + (rowBase / 2 + jp) * D + c * 8);
    dst[0] = lo;
    dst[1] = hi;
  }
#endif
}

// Grid (ceil(seqLen / (NT*TM)), batchSize*numHeads), block NT*KS. Each lane owns TM query slots of
// the prep kernel's permutation (on-board positions first). Per key pair it computes the two
// scores of each of its rows against K rows held in SGPRs, updates a lazily-rescaled online
// softmax, and accumulates both keys into every output channel with one dot2 per channel against
// the interleaved V pair row, also in SGPRs.
// With KS > 1, the KS groups of NT threads cover the same query slots but split the key pairs into
// contiguous chunks, then merge their partial softmax states through LDS. Small batches launch too
// few waves to hide the latency of this loop, and the split multiplies them.
template<int D, int TM, int NT, int KS>
__global__ __launch_bounds__(NT * KS)
void rkDot2AttnKernel(
  const half* __restrict__ Q, const half* __restrict__ Kc, const uint32_t* __restrict__ Vc,
  const short* __restrict__ permIn, half* __restrict__ out,
  int seqLen, int numHeads, int numKVHeads, int qStride
) {
#if KATAGO_ROCM_DEVICE_HAS_DOT2
  constexpr int DH2 = D / 2;
  constexpr int DV4 = D / 8;
  // NT is a multiple of the wave size, so split is wave-uniform; readfirstlane tells the compiler,
  // which keeps the K/V loads below on the scalar unit.
  const int split = __builtin_amdgcn_readfirstlane(threadIdx.x / NT);
  const int tid = threadIdx.x - split * NT;
  const int bh = blockIdx.y;
  const int n = bh / numHeads;
  const int h = bh - n * numHeads;
  const int kvh = h / (numHeads / numKVHeads);
  const int oStride = numHeads * D;
  const short* perm = permIn + (size_t)n * rkPermStride(seqLen);
  const int count = perm[seqLen];

  const int slotBase = blockIdx.x * (NT * TM);
  int qPos[TM];
  #pragma unroll
  for(int i = 0; i < TM; i++) {
    const int slot = slotBase + i * NT + tid;
    qPos[i] = slot < seqLen ? (int)perm[slot] : -1;
    if(slot >= count && qPos[i] >= 0) {
      // Off-board query: zero output row.
      if(split == 0) {
        uint4* o = (uint4*)(out + ((size_t)n * seqLen + qPos[i]) * oStride + h * D);
        #pragma unroll
        for(int j = 0; j < DV4; j++) o[j] = make_uint4(0, 0, 0, 0);
      }
      qPos[i] = -1;
    }
  }
  // Block-uniform, so no thread skips the merge barrier below.
  if(slotBase >= count)
    return;

  // Q rows. The prep kernel pre-scaled K by softmax scale * log2(e), so dot products with these
  // are directly exp2 arguments.
  rk_h2 q[TM][DH2];
  #pragma unroll
  for(int i = 0; i < TM; i++) {
    const int p = qPos[i] < 0 ? 0 : qPos[i];
    const uint4* src = (const uint4*)(Q + ((size_t)n * seqLen + p) * qStride + h * D);
    #pragma unroll
    for(int j = 0; j < DV4; j++) {
      uint4 v = src[j];
      q[i][j * 4 + 0] = rkU32ToH2(v.x);
      q[i][j * 4 + 1] = rkU32ToH2(v.y);
      q[i][j * 4 + 2] = rkU32ToH2(v.z);
      q[i][j * 4 + 3] = rkU32ToH2(v.w);
    }
  }

  // Online softmax with a lazily moved reference max: scores are kept relative to -negM, and the
  // reference only moves (rescaling acc and l) when the running max leaves [-8, 8], so p <= 256
  // always fits FP16 with full relative precision, and rescaling is rare after the first pairs.
  float negM[TM], mrun[TM], l[TM];
  float acc[TM][D];
  #pragma unroll
  for(int i = 0; i < TM; i++) {
    negM[i] = 0.0f;
    mrun[i] = -INFINITY;
    l[i] = 0.0f;
    #pragma unroll
    for(int d = 0; d < D; d++) acc[i][d] = 0.0f;
  }

  const int SP = rkAttnPaddedSeq(seqLen);
  const uint32_t* __restrict__ kBase = (const uint32_t*)(Kc + ((size_t)n * numKVHeads + kvh) * SP * D);
  const uint32_t* __restrict__ vBase = Vc + ((size_t)n * numKVHeads + kvh) * (SP / 2) * D;
  const int numPairs = (count + 1) >> 1;
  // This group's chunk of the key pairs. Split 0's is never empty, since count >= 1 here.
  const int chunk = (numPairs + KS - 1) / KS;
  const int jpBegin = min(split * chunk, numPairs);
  const int jpEnd = min(jpBegin + chunk, numPairs);

  // K rows of the current key pair. Loaded through the scalar cache (wave-uniform addresses into
  // read-only memory) and refilled with the next pair's rows right after use, so that their load
  // latency overlaps the softmax and PV work.
  uint32_t kr[2 * DH2];
  {
    const uint32_t* k0 = kBase + (size_t)(2 * min(jpBegin, numPairs - 1)) * DH2;
    #pragma unroll
    for(int dd = 0; dd < 2 * DH2; dd++) kr[dd] = k0[dd];
  }
  for(int jp = jpBegin; jp < jpEnd; jp++) {
    uint32_t vr[D];
    const uint32_t* vp = vBase + (size_t)jp * D;
    #pragma unroll
    for(int d = 0; d < D; d++) vr[d] = vp[d];

    float s0[TM], s1[TM];
    #pragma unroll
    for(int i = 0; i < TM; i++) { s0[i] = negM[i]; s1[i] = negM[i]; }
    #pragma unroll
    for(int dd = 0; dd < DH2; dd++) {
      #pragma unroll
      for(int i = 0; i < TM; i++) {
        s0[i] = rkDot2(rkU32ToH2(kr[dd]), q[i][dd], s0[i]);
        s1[i] = rkDot2(rkU32ToH2(kr[DH2 + dd]), q[i][dd], s1[i]);
      }
    }
    {
      const uint32_t* kn = kBase + (size_t)(2 * min(jp + 1, jpEnd - 1)) * DH2;
      #pragma unroll
      for(int dd = 0; dd < 2 * DH2; dd++) kr[dd] = kn[dd];
    }

    bool needRescale = false;
    float mr[TM];
    #pragma unroll
    for(int i = 0; i < TM; i++) {
      mr[i] = fmaxf(mrun[i], fmaxf(s0[i], s1[i]));
      needRescale |= (mr[i] > 8.0f || mr[i] < -8.0f);
    }
    // Wave-uniform branch: a divergent one makes the compiler shuffle the whole accumulator
    // array around the join point, which costs far more than rescaling a few extra rows by 1.
    if(__any(needRescale)) {
      #pragma unroll
      for(int i = 0; i < TM; i++) {
        float shift = (mr[i] > 8.0f || mr[i] < -8.0f) ? mr[i] : 0.0f;
        float alpha = __builtin_amdgcn_exp2f(-shift);
        negM[i] -= shift;
        l[i] *= alpha;
        #pragma unroll
        for(int d = 0; d < D; d++) acc[i][d] *= alpha;
        s0[i] -= shift;
        s1[i] -= shift;
        mr[i] -= shift;
      }
    }
    // The padded slot past an odd count has a zero V row; this also drops it from the denominator.
    const rk_h2 ones = rkU32ToH2((2 * jp + 1 < count) ? 0x3C003C00u : 0x00003C00u);
    #pragma unroll
    for(int i = 0; i < TM; i++) {
      mrun[i] = mr[i];
      rk_h2 ph = rkU32ToH2(rkPackF32ToH2(__builtin_amdgcn_exp2f(s0[i]), __builtin_amdgcn_exp2f(s1[i])));
      l[i] = rkDot2(ones, ph, l[i]);
      #pragma unroll
      for(int d = 0; d < D; d++)
        acc[i][d] = rkDot2(rkU32ToH2(vr[d]), ph, acc[i][d]);
    }
  }

  if constexpr(KS > 1) {
    // Splits 1..KS-1 hand their partial states to split 0 one at a time through one LDS buffer,
    // lane-minor so that the accesses are conflict-free, keeping the LDS per block (and so the
    // occupancy) independent of KS.
    __shared__ float part[TM][D + 2][NT];
    // Split 0's weights are relative to the reference max -negM; incoming ones are rebased onto
    // the larger of the two references.
    for(int s = 1; s < KS; s++) {
      if(split == s) {
        #pragma unroll
        for(int i = 0; i < TM; i++) {
          // An empty chunk contributes nothing, whatever its (initial) reference.
          part[i][0][tid] = l[i] > 0.0f ? -negM[i] : -INFINITY;
          part[i][1][tid] = l[i];
          #pragma unroll
          for(int d = 0; d < D; d++) part[i][2 + d][tid] = acc[i][d];
        }
      }
      __syncthreads();
      if(split == 0) {
        #pragma unroll
        for(int i = 0; i < TM; i++) {
          const float rs = part[i][0][tid];
          const float r = fmaxf(-negM[i], rs);
          const float f = __builtin_amdgcn_exp2f(-negM[i] - r);
          const float fs = __builtin_amdgcn_exp2f(rs - r);
          negM[i] = -r;
          l[i] = l[i] * f + part[i][1][tid] * fs;
          #pragma unroll
          for(int d0 = 0; d0 < D; d0 += 16) {
            // Keeps the compiler from hoisting all the LDS reads at once, whose registers would
            // cut the occupancy of the whole kernel.
            __builtin_amdgcn_sched_barrier(0);
            #pragma unroll
            for(int d = d0; d < d0 + 16; d++) acc[i][d] = acc[i][d] * f + part[i][2 + d][tid] * fs;
          }
        }
      }
      __syncthreads();
    }
    if(split > 0)
      return;
  }

  #pragma unroll
  for(int i = 0; i < TM; i++) {
    if(qPos[i] < 0)
      continue;
    const float inv = 1.0f / l[i];
    uint4* o = (uint4*)(out + ((size_t)n * seqLen + qPos[i]) * oStride + h * D);
    #pragma unroll
    for(int j = 0; j < DV4; j++) {
      uint4 v;
      v.x = rkPackF32ToH2(acc[i][j * 8 + 0] * inv, acc[i][j * 8 + 1] * inv);
      v.y = rkPackF32ToH2(acc[i][j * 8 + 2] * inv, acc[i][j * 8 + 3] * inv);
      v.z = rkPackF32ToH2(acc[i][j * 8 + 4] * inv, acc[i][j * 8 + 5] * inv);
      v.w = rkPackF32ToH2(acc[i][j * 8 + 6] * inv, acc[i][j * 8 + 7] * inv);
      o[j] = v;
    }
  }
#endif
}

void rocmDot2AttentionPrep(
  half* q, const half* k, const half* v, const half* mask,
  half* kc, half* vc, short* perm,
  int batchSize, int seqLen, int numHeads, int numKVHeads, int headDim, int qStride, int kvStride,
  const RocmAttnRope& rope, hipStream_t stream
) {
  if(!rocmDot2AttentionSupportsShape(numHeads, numKVHeads, headDim, headDim))
    throw std::runtime_error("rocmDot2AttentionPrep: unsupported head configuration");
  if(seqLen >= 32767 || batchSize > 65535)
    throw std::runtime_error("rocmDot2AttentionPrep: seqLen or batchSize too large");
  if(qStride % 8 != 0 || kvStride % 8 != 0)
    throw std::runtime_error("rocmDot2AttentionPrep: row strides must be multiples of 8 halfs");
  constexpr int NT = 256;
  dim3 grid(numKVHeads, batchSize);
  size_t shm = ((size_t)seqLen * sizeof(short) + 15) / 16 * 16;
  uint32_t* vcw = (uint32_t*)vc;
  // Softmax scale and the log2(e) of computing exp as exp2, folded into K.
  const float kScale = 1.4426950408889634f / sqrtf((float)headDim);
#define RK_PREP_LAUNCH(DIM, MODE) \
  rkAttnPrepKernel<DIM, NT, MODE><<<grid, NT, shm, stream>>>( \
    q, k, v, mask, kc, vcw, perm, seqLen, numHeads, numKVHeads, qStride, kvStride, kScale, \
    rope.cosTable, rope.sinTable, rope.learnableTable ? 1 : 0, rope.freqs, rope.nnXLen)
  const int mode = (int)rope.mode;
  if(headDim == 32) {
    if(mode == RocmAttnRope::NONE) RK_PREP_LAUNCH(32, 0);
    else if(mode == RocmAttnRope::TABLE) RK_PREP_LAUNCH(32, 1);
    else RK_PREP_LAUNCH(32, 2);
  }
  else {
    if(mode == RocmAttnRope::NONE) RK_PREP_LAUNCH(64, 0);
    else if(mode == RocmAttnRope::TABLE) RK_PREP_LAUNCH(64, 1);
    else RK_PREP_LAUNCH(64, 2);
  }
#undef RK_PREP_LAUNCH
}

// Launches rkDot2AttnKernel with KS key splits, or none.
template<int D, int TM, int NT, int KS>
static void rkLaunchDot2Attention(
  bool splitKeys, const half* q, const half* kc, const uint32_t* vc, const short* perm, half* output,
  int batchSize, int seqLen, int numHeads, int numKVHeads, int qStride, hipStream_t stream
) {
  dim3 grid((seqLen + NT * TM - 1) / (NT * TM), batchSize * numHeads);
  if(splitKeys)
    rkDot2AttnKernel<D, TM, NT, KS><<<grid, NT * KS, 0, stream>>>(
      q, kc, vc, perm, output, seqLen, numHeads, numKVHeads, qStride);
  else
    rkDot2AttnKernel<D, TM, NT, 1><<<grid, NT, 0, stream>>>(
      q, kc, vc, perm, output, seqLen, numHeads, numKVHeads, qStride);
}

void rocmDot2Attention(
  const half* q, const half* kc, const half* vc, const short* perm, half* output,
  int batchSize, int seqLen, int numHeads, int numKVHeads, int headDim, int qStride,
  int numMultiprocessors, int waveSize, hipStream_t stream
) {
  if(!rocmDot2AttentionSupportsShape(numHeads, numKVHeads, headDim, headDim))
    throw std::runtime_error("rocmDot2Attention: unsupported head configuration");
  if((long long)batchSize * numHeads > 65535)
    throw std::runtime_error("rocmDot2Attention: batchSize * numHeads too large");
  const uint32_t* vcw = (const uint32_t*)vc;
  // Measured on gfx1030: head dim 32 prefers two query rows per lane (more scalar-operand reuse at
  // a still high occupancy), head dim 64 one row (its accumulators are twice as large).
  constexpr int TM32 = 2, NT32 = 64, TM64 = 1, NT64 = 128;
  const int TM = headDim == 32 ? TM32 : TM64;
  const int NT = headDim == 32 ? NT32 : NT64;
  // Unsplit waves, against the ~8 per SIMD it takes to hide the latency of the per-key-pair
  // dependency chain, with 4 SIMDs per multiprocessor (a CU on GCN/CDNA, a WGP on RDNA). Also
  // measured on gfx1030: head dim 32 gains from 4 key splits up to about twice that many waves
  // (2.7x faster at batch size 1, 1.25x at 8, even at 48) and loses beyond; head dim 64, whose
  // merge costs it occupancy, only from 2 splits below half.
  const long long waves = (long long)((seqLen + NT * TM - 1) / (NT * TM)) * batchSize * numHeads * (NT / waveSize);
  const long long targetWaves = 32LL * numMultiprocessors;
  if(headDim == 32)
    rkLaunchDot2Attention<32, TM32, NT32, 4>(
      waves <= 2 * targetWaves, q, kc, vcw, perm, output, batchSize, seqLen, numHeads, numKVHeads, qStride, stream);
  else
    rkLaunchDot2Attention<64, TM64, NT64, 2>(
      2 * waves <= targetWaves, q, kc, vcw, perm, output, batchSize, seqLen, numHeads, numKVHeads, qStride, stream);
}
