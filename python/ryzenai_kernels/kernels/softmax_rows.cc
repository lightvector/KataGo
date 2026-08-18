// softmax_rows.cc - multi-row bf16 softmax wrapper for AIE2P (Strix/npu2).
//
// Reuses the stock mlir-aie aie2p kernel (per-row softmax, vector width 32,
// aie::exp2 native exp) and loops it over the rows of one fifo element.
// Width must be a multiple of 32 (the design pads 361 -> 384 on the host;
// pad columns are -1e30 so exp() underflows to exactly 0 and the row sum is
// unaffected).
//
// Origin: written for the KataGo RyzenAI backend's attention softmax
// (rows = numHeads * S). Compiled per shape by build_softmax.py.

#include "aie_kernels/aie2p/softmax.cc"

extern "C" {

void softmax_rows_bf16(bfloat16 *restrict input, bfloat16 *restrict output,
                       const int32_t n_rows, const int32_t width) {
  for (int r = 0; r < n_rows; r++) {
    softmax_simple_bf16(input + (size_t)r * width,
                        output + (size_t)r * width, width);
  }
}

} // extern "C"
