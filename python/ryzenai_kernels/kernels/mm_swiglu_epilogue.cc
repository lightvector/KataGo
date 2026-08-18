// mm_swiglu_epilogue.cc -- SwiGLU epilogue for the whole_array bf16 GEMM core.
//
// Runs on the fully accumulated C tile (DIM_M x DIM_N f32) after the K loop,
// before the tile is released to the output fifo. The host uploaded B with the
// linear1 / linearGate columns interleaved in groups of 8:
//
//   B column (c>>3)*16 + (c&7)       = linear1  weight for out channel c
//   B column (c>>3)*16 + 8 + (c&7)   = linearGate weight for out channel c
//
// so inside a core's C tile every even 8-column sub-tile holds linear1 outputs
// and the following odd sub-tile the matching gates. The epilogue computes
// silu(l) * g for each pair and writes it back over the l positions; the gate
// positions keep their raw values and the host simply never reads them.
//
// The C tile's in-memory order is the stock 2x2_mmul template's (r=4, t=8
// sub-tiles): logical element (i, j) of the tile sits at
//
//   offset(i, j) = (i >> 2) * (4 * DIM_N) + (j >> 3) * 32 + (i & 3) * 8 + (j & 7)
//
// (hardware-verified for this template family, see the attention op's probes).
// An (l, g) sub-tile pair is therefore always 32 contiguous f32 apart, and each
// 16-lane chunk within a sub-tile holds whole rows of one pair half.
//
// SiLU follows mm_activation_epilogue.cc: sigmoid built from the tanh SFU as
// 0.5 * (1 + tanh(x/2)), narrowing to bf16 only inside the sigmoid where the
// [0, 1] range is harmless; x, the gate and both multiplies stay f32.

#include <aie_api/aie.hpp>
#include <stdint.h>

#ifndef DIM_M
#define DIM_M 32
#endif
#ifndef DIM_N
#define DIM_N 32
#endif

using namespace aie;

static inline aie::vector<float, 16> swiglu16(aie::vector<float, 16> l,
                                              aie::vector<float, 16> g) {
  const aie::vector<float, 16> halff = aie::broadcast<float, 16>(0.5f);
  const aie::vector<float, 16> onef = aie::broadcast<float, 16>(1.0f);
  aie::vector<float, 16> half_l = aie::mul(l, halff);
  // Only the tanh itself narrows to bf16 (the SFU's output format); the
  // 1 + t and the 0.5x stay f32 so they add no further rounding, and x, the
  // gate and both multiplies stay f32 as in mm_activation_epilogue.cc.
  aie::vector<bfloat16, 16> tanh_half = aie::tanh<bfloat16>(half_l);
  aie::accum<accfloat, 16> tacc;
  tacc.from_vector(tanh_half);
  aie::vector<float, 16> t = tacc.to_vector<float>();
  aie::vector<float, 16> sig = aie::mul(aie::add(t, onef), halff);
  // aie::mul on f32 vectors yields an accumulator, so chain via explicit
  // vector temporaries (as mm_activation_epilogue.cc does).
  aie::vector<float, 16> ls = aie::mul(l, sig);
  return aie::vector<float, 16>(aie::mul(ls, g));
}

extern "C" {

void mm_swiglu_epilogue_f32(float *__restrict c) {
  event0();
  static_assert(DIM_M % 4 == 0 && DIM_N % 16 == 0,
                "sub-tiles are 4x8 and (l, g) pairs span two of them");
  for (int rb = 0; rb < DIM_M / 4; rb++) {
    for (int cg = 0; cg < DIM_N / 8; cg += 2) {
      float *__restrict pl = c + rb * (4 * DIM_N) + cg * 32;
      float *__restrict pg = pl + 32;
      for (int h = 0; h < 2; h++) {  // 32 elements per sub-tile = 2 x 16 lanes
        aie::vector<float, 16> l = aie::load_v<16>(pl + h * 16);
        aie::vector<float, 16> g = aie::load_v<16>(pg + h * 16);
        aie::store_v(pl + h * 16, swiglu16(l, g));
      }
    }
  }
  event1();
}

} // extern "C"
