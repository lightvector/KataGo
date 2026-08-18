// attention_head.cc - fused single-head attention tile kernel for AIE2P
// (Strix/npu2), bf16 in with f32 accumulation.
//
// One AIE core owns one attention head for a whole dispatch. K and V arrive
// as one packed per-head buffer (a core tile has only two input DMA
// channels), pre-tiled by the host into the mmul tile order the stock
// 2x2_mmul template reads; Q streams through in 8-row blocks, likewise
// pre-tiled. Per 8-row query block:
//
//   scores[8 x 384] = Q[8 x 32] @ K^T[32 x 384]   (K col-major B)
//   scores = softmax(scores)                       (pad columns forced -1e30)
//   out[8 x 32]     = P[8 x 384] @ V[384 x 32]    (V row-major B, f32 out)
//
// LAYOUT NOTE (the subtle part of this kernel): the stock matmul template
// reads A and B and writes C in mmul tile order -- (4x8) tiles of A, (8x8)
// tiles of B, (4x8) tiles of C, NOT plain row-major. The host pre-tiles Q/K/V
// and un-tiles the output; the score buffer never leaves the core, so the
// softmax here gathers it out of QK^T's C-tile order and scatters the
// probabilities into P*V's A-tile order. The three index mappings below are
// derived from matmul_vectorized_2x2_mmul's pointer arithmetic in mm.cc and
// verified on hardware by attnbench.
//
// Numerics match the staged path: scores are bf16 before the exp (as they
// are when the standalone softmax op reads them), P*V accumulates in f32.
// Pad key columns are set to (row max - 20) before the softmax -- NOT -inf,
// because this kernel's f32 exp2 wraps huge negative inputs to huge positive
// ones -- and then written as exact 0 into P, so they contribute nothing.
// Pad query rows produce finite garbage the host discards.

#define DIM_M 8
#define DIM_K 32
#define DIM_N 384
#define bf16_bf16_ONLY
#define B_COL_MAJ

#include "aie_kernels/aie2p/mm.cc"
#include "aie_kernels/aie2p/softmax.cc"

#ifndef ATTN_S_REAL
#define ATTN_S_REAL 361
#endif

namespace {

// Per-core scratch in L1: QK^T's scores and P*V's probabilities (same
// elements, different tile order), plus one plain row for the softmax.
// Static storage is private to each core (each core links its own ELF).
alignas(64) bfloat16 g_scores[8 * 384];
bfloat16 g_row[384];

// The C-tile order of QK^T and the A-tile order of P*V are the SAME formula
// for these shapes (verified on hardware), so the softmax reads and writes
// g_scores in place through idx_tile -- no separate probability buffer.
int idx_tile(int qi, int ki) {
  return (qi >> 2) * 1536 + (ki >> 3) * 32 + (qi & 3) * 8 + (ki & 7);
}

} // namespace

extern "C" {

// One 8-row query block of one head: q = 256 bf16 in A-tile order, kv = the
// head's K then V, each 384x32 bf16 in B-tile order (back to back), out =
// 8x32 f32 in C-tile order (host un-tiles on readback).
void attn_block_bf16(bfloat16 *restrict q, bfloat16 *restrict kv,
                     float *restrict out) {
  const bfloat16 *k = kv;
  const bfloat16 *v = kv + (size_t)384 * 32;

  // The 2x2_mmul template accumulates onto C in place (the GEMM design zeroes
  // between calls), so both output buffers must be zeroed first.
  for (int i = 0; i < 8 * 384; i++)
    g_scores[i] = (bfloat16)0.0f;
  for (int i = 0; i < 8 * 32; i++)
    out[i] = 0.0f;

  // scores = Q @ K^T (K's tile order is the col-major-B one).
  matmul_vectorized_2x2_mmul<bfloat16, bfloat16, /*rowA=*/2, /*colA=*/4,
                             /*colB=*/48, /*r=*/4, /*s=*/8, /*t=*/8,
                             /*b_row_maj=*/false, /*c_row_maj=*/true>(
      q, k, g_scores);

  // Per-row softmax. The tiled score layout keeps one row's 8 columns
  // contiguous within each 32-element group, so gather/scatter run as
  // 8-wide vector ops. Pad columns get (row max - 20), NOT -inf: this
  // softmax kernel computes exp2 in f32 and huge negative inputs wrap the
  // exponent field to huge POSITIVE values (measured: -1e30 pads come out as
  // ~+7.7e30). max-20 gives exp2(-28.9) ~ 3e-9, negligible and safe. Pad
  // columns are stored as exact 0 afterwards.
  for (int r = 0; r < 8; r++) {
    const int rbase = (r >> 2) * 1536 + (r & 3) * 8;
    // Full 8-column chunks are gathered as vectors; the ragged tail (S is not
    // a multiple of 8) is gathered element-wise so stale values never leak in.
    aie::vector<float, 8> vmax8 = aie::broadcast<float, 8>(-1e30f);
    constexpr int fullG = ATTN_S_REAL / 8;
    for (int g = 0; g < fullG; g++) {
      aie::vector<bfloat16, 8> el = aie::load_v<8>(g_scores + rbase + g * 32);
      aie::store_v(g_row + g * 8, el);
      aie::vector<float, 8> f = aie::accum<accfloat, 8>(el).to_vector<float>();
      vmax8 = aie::max(vmax8, f);
    }
    float maxVal = aie::reduce_max(vmax8);
    for (int ki = fullG * 8; ki < ATTN_S_REAL; ki++) {
      const bfloat16 el = g_scores[rbase + (ki >> 3) * 32 + (ki & 7)];
      g_row[ki] = el;
      if((float)el > maxVal)
        maxVal = (float)el;
    }
    // Pad columns: row max - 20, NOT -inf (see above about the exp2 wrap).
    // Their softmax output is stored as exact 0 below, and V's pad rows are
    // host-zeroed, so the pads contribute nothing either way.
    for (int ki = ATTN_S_REAL; ki < 384; ki++)
      g_row[ki] = (bfloat16)(maxVal - 20.0f);
    softmax_simple_bf16(g_row, g_row, 384);
    for (int g = 0; g < fullG; g++)
      aie::store_v(g_scores + rbase + g * 32, aie::load_v<8>(g_row + g * 8));
    for (int ki = fullG * 8; ki < ATTN_S_REAL; ki++)
      g_scores[rbase + (ki >> 3) * 32 + (ki & 7)] = g_row[ki];
    for (int g = (ATTN_S_REAL + 7) / 8; g < 48; g++)
      aie::store_v(g_scores + rbase + g * 32, aie::broadcast<bfloat16, 8>(0.0f));
  }

  // out = P @ V, f32. V arrives col-major (its host tiling matches the
  // col-major B read verified by the PV probe; the row-major read was never
  // verified for this shape).
  matmul_vectorized_2x2_mmul<bfloat16, float, /*rowA=*/2, /*colA=*/48,
                             /*colB=*/4, /*r=*/4, /*s=*/8, /*t=*/8,
                             /*b_row_maj=*/false, /*c_row_maj=*/true>(
      g_scores, v, out);
}

} // extern "C"
