// bn_mish.cc - fused per-channel BatchNorm scale/bias + Mish activation,
// row-streamed, bf16 in/out, f32 internal, for AIE2P (Strix/npu2).
//
//   out[r][c] = mish(scale[c] * x[r][c] + bias[c])
//
// mish(t) = t * (s^2 + 2s) / (s^2 + 2s + 2) with s = e^t. The division uses
// aie::inv (vector reciprocal). For t > 16 the formula saturates to t
// (mish(16) == 16 to bf16 precision), so the exp argument is clamped at 16
// and the result selected from the identity branch above the clamp. For
// t < -16 the result is ~t*e^t, already below bf16 resolution.
//
// scale/bias are resident per core (the host replicates the vectors across
// the 8 columns). width must be a multiple of 16.

#include <aie_api/aie.hpp>
#include <stdint.h>

namespace {

constexpr float kLog2e = 1.44269504089f;

} // namespace

extern "C" {

void bn_mish_bf16(bfloat16 *restrict input, bfloat16 *restrict sb,
                  bfloat16 *restrict output, const int32_t rows,
                  const int32_t width) {
  // A core tile has only two input DMA channels, so scale and bias arrive as
  // one buffer: [scale (width) | bias (width)].
  const bfloat16 *scale = sb;
  const bfloat16 *bias = sb + width;
  constexpr int V = 16;
  const int chunks = width / V;
  for (int r = 0; r < rows; r++) {
    const bfloat16 *in = input + (size_t)r * width;
    bfloat16 *out = output + (size_t)r * width;
    for (int i = 0; i < chunks; i++) {
      aie::vector<bfloat16, V> xb = aie::load_v<V>(in + i * V);
#ifdef BNM_DUMPMODE
      // 1: copy x; 2: output t=x*s+b; else full mish
      if(BNM_DUMPMODE == 1) { aie::store_v(out + i * V, xb); continue; }
#endif
      aie::vector<bfloat16, V> sb = aie::load_v<V>(scale + i * V);
      aie::vector<bfloat16, V> bb = aie::load_v<V>(bias + i * V);

      aie::vector<float, V> x = aie::accum<accfloat, V>(xb).to_vector<float>();
      aie::vector<float, V> sc = aie::accum<accfloat, V>(sb).to_vector<float>();
      aie::vector<float, V> bi = aie::accum<accfloat, V>(bb).to_vector<float>();

      // t = x*scale + bias. The exp path clamps at +16 (bf16 resolution makes
      // mish(16) == 16); anything above the clamp is the identity branch and
      // is added back as the excess, so no select/mask is needed:
      //   out = mish(min(t,16)) + max(t-16, 0)
      aie::vector<float, V> t = aie::add(aie::mul(x, sc), bi);
      // Clamp for the exp overflow point. aie::min/max on f32 vectors are
      // broken on XDNA2 (they returned garbage -- measured), so the clamp runs
      // in bf16 through the proven from_vector/to_vector conversions. The
      // excess above the clamp is kept exactly: out = mish(min(t,16)) + (t-tc).
      aie::accum<accfloat, V> tacc;
      tacc.from_vector(t);
      aie::vector<bfloat16, V> tcb =
          aie::min(tacc.to_vector<bfloat16>(), aie::broadcast<bfloat16, V>(16.0f));
      aie::accum<accfloat, V> tcacc;
      tcacc.from_vector(tcb);
      aie::vector<float, V> tc = tcacc.to_vector<float>();
      aie::vector<float, V> over = aie::sub(t, tc);

      // s = e^tc = exp2(tc * log2e). XDNA2's exp2 returns bf16 (f32 return is
      // AIE_MLv2-only), which is plenty since the output is bf16 anyway.
      aie::vector<float, V> arg =
          aie::mul(tc, aie::broadcast<float, V>(kLog2e));
      aie::vector<bfloat16, V> sigb = aie::exp2<bfloat16>(arg);
      aie::vector<float, V> sig = aie::accum<accfloat, V>(sigb).to_vector<float>();
      // mish = tc * (s^2+2s)/(s^2+2s+2) = tc * num/(num+2). XDNA2's vector
      // reciprocal only takes bf16, so the divide narrows through bf16 --
      // harmless at bf16 output precision.
      aie::vector<float, V> two = aie::broadcast<float, V>(2.0f);
      aie::vector<float, V> num = aie::add(aie::mul(sig, sig), aie::mul(two, sig));
      aie::vector<float, V> den = aie::add(num, two);
      aie::accum<accfloat, V> den_acc;
      den_acc.from_vector(den);
      aie::vector<bfloat16, V> rinv_b = aie::inv(den_acc.to_vector<bfloat16>());
      aie::accum<accfloat, V> rinv_acc;
      rinv_acc.from_vector(rinv_b);
      aie::vector<float, V> frac = aie::mul(num, rinv_acc.to_vector<float>());
      aie::vector<float, V> mish = aie::mul(tc, frac);
      aie::vector<float, V> res = aie::add(mish, over);
#ifdef BNM_DUMPMODE
      if(BNM_DUMPMODE == 2)
        res = t;  // output the pre-activation
#endif
      aie::accum<accfloat, V> rac;
      rac.from_vector(res);
      aie::store_v(out + i * V, rac.to_vector<bfloat16>());
    }
  }
}

} // extern "C"
