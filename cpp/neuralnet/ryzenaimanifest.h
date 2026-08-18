/*
 * ABI contract between the NPU kernel binaries under artifacts/ and this C++
 * host code.
 *
 * The kernel build (python/ryzenai_kernels/build_kernels.py) writes the
 * same facts into artifacts/manifest.json. Both must agree: ABI_VERSION here is
 * checked against the manifest's "manifest_version" at load time so that a
 * mismatched pair fails loudly instead of computing garbage.
 *
 * None of this is shape information. M/K/N live in the per-shape instruction
 * stream, not in the xclbin - see artifacts/README.md.
 */

#ifndef NEURALNET_RYZENAI_MANIFEST_H_
#define NEURALNET_RYZENAI_MANIFEST_H_

#include <cstdint>

namespace RyzenAIManifest {

  // Bump together with "manifest_version" in artifacts/manifest.json whenever
  // anything below changes.
  constexpr int ABI_VERSION = 1;

  // mlir-aie names the xclbin's kernel with this prefix plus a generated
  // suffix, so hosts must prefix-match over xclbin.get_kernels() rather than
  // compare for equality (this mirrors what mlir-aie's own
  // runtime_lib/test_lib/test_utils.cpp does).
  constexpr const char* KERNEL_NAME_PREFIX = "MLIR_AIE";

  // ERT opcode for "start kernel with instruction buffer".
  constexpr uint32_t OPCODE_START_WITH_INSTRUCTIONS = 3;

  // xrt::run argument slots.
  constexpr int ARG_OPCODE = 0;  // scalar, OPCODE_START_WITH_INSTRUCTIONS
  constexpr int ARG_INSTR = 1;   // xrt::bo, cacheable, holds the .insts.bin bytes
  constexpr int ARG_NINSTR = 2;  // scalar, instruction length in 32-bit words
  constexpr int ARG_A = 3;       // xrt::bo, host_only, M*K bfloat16 row-major
  constexpr int ARG_B = 4;       // xrt::bo, host_only, K*N bfloat16 row-major
  constexpr int ARG_C = 5;       // xrt::bo, host_only, M*N float32 row-major

  // MMUL tile geometry the GEMM artifacts are compiled with, as recorded in
  // artifacts/manifest.json. These drive the padding rules: M must be a
  // multiple of GEMM_TILE_M*8 (one pass over 4 double-buffered AIE rows), N a
  // multiple of GEMM_TILE_N*cols, and K a multiple of GEMM_TILE_K.
  //
  // A few early artifacts were compiled at tileN 48 because their N was 384;
  // the grid is tileN 32 throughout, which divides every N the padding rules
  // produce.
  constexpr int GEMM_TILE_M = 32;
  constexpr int GEMM_TILE_K = 64;
  constexpr int GEMM_TILE_N = 32;

}  // namespace RyzenAIManifest

#endif  // NEURALNET_RYZENAI_MANIFEST_H_
