#ifndef NEURALNET_RYZENAI_SEQUENCE_H_
#define NEURALNET_RYZENAI_SEQUENCE_H_

// Builds the NPU instruction stream (the TXN control code that an .insts.bin
// file holds) for a GEMM, in plain C++ with no toolchain of any kind. This is
// what lets an arbitrary model run without Python: the .xclbin bakes in only K,
// so M and N -- which vary per layer, per board size and per batch -- are
// resolved here at run time instead of needing a precompiled artifact each.
//
// Pure C++17, standard library only. No XRT headers, no IRON, no file I/O:
// callers hand the result straight to an xrt::bo.
//
// Verified by byte-for-byte comparison against IRON-compiled goldens; see
// python/ryzenai_kernels/INSTS_FORMAT.md for the decoded format and the field formulas.

#include <cstdint>
#include <vector>

namespace RyzenAISequence {

// Header field: which AIE generation the stream targets. npu1 = Phoenix /
// Hawk Point (aie2), npu2 = Strix / Krackan (aie2p).
enum class Arch { NPU1 = 3, NPU2 = 4 };

struct GemmShape {
  int M;      // rows of A and C
  int K;      // reduction dim -- must match the .xclbin, which bakes K in
  int N;      // columns of B and C
  int tileM;  // MMUL micro-kernel tile, from the artifact manifest
  int tileK;
  int tileN;
};

// True if instruction streams can be generated for this column count. The
// per-column descriptor layout comes from IRON's placer and is tabulated in
// sequence_layout.h, so only the tabulated widths are supported.
bool supportsColumns(int cols);

// Builds the stream for `cols` AIE columns. Throws std::invalid_argument if the
// shape is not realizable (see the checks at the top of the implementation) or
// if the column count has no layout.
std::vector<uint32_t> generateSequence(Arch arch, int cols, const GemmShape& shape);
std::vector<uint32_t> generateSequence(
  Arch arch, int cols, int M, int K, int N, int tileM, int tileK, int tileN);

// Convenience wrappers for the single-column case.
std::vector<uint32_t> generateSingleColSequence(Arch arch, const GemmShape& shape);
std::vector<uint32_t> generateSingleColSequence(
  Arch arch, int M, int K, int N, int tileM, int tileK, int tileN);

}  // namespace RyzenAISequence

#endif  // NEURALNET_RYZENAI_SEQUENCE_H_
