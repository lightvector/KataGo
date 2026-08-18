/*
 * Loading and dispatch of the NPU kernel binaries (xclbin) produced by
 * python/ryzenai_kernels/.
 *
 * Like device.{h,cpp}, this is one of the only places that includes XRT
 * headers; callers deal in plain buffers.
 *
 * An .xclbin bakes in only the reduction dimension K (measured; see
 * python/ryzenai_kernels/INSTS_FORMAT.md), so one binary serves every M and N -- the
 * instruction stream that carries those is generated per dispatch by
 * sequence.cpp. That is why the unit of loading here is a K, not a shape.
 */

#ifndef NEURALNET_RYZENAI_KERNEL_H_
#define NEURALNET_RYZENAI_KERNEL_H_

#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

#include "../neuralnet/ryzenaidevice.h"

namespace RyzenAIKernel {

  // ---- bfloat16 -------------------------------------------------------------
  // bf16 is just the top 16 bits of an IEEE fp32, so conversion is a shift plus
  // round-to-nearest-even. No lookup tables and no library needed.

  // Converts a whole run of floats, eight at a time where the hardware allows.
  // Bit-for-bit what the element-wise version below produces, NaN handling
  // included - it is the same arithmetic, just widened. Worth having as its own
  // function because packing activations is a few percent of every evaluation:
  // the branch on NaN keeps compilers from vectorising the obvious loop.
  void floatToBf16Bulk(const float* src, uint16_t* dst, size_t n);

  inline uint16_t floatToBf16(float f) {
    uint32_t bits;
    static_assert(sizeof(bits) == sizeof(f), "float is not 32 bits");
    std::memcpy(&bits, &f, sizeof(bits));
    // NaN must stay NaN: rounding a quiet NaN's payload away could turn it into
    // an infinity, so force the quiet bit instead of rounding.
    if((bits & 0x7F800000u) == 0x7F800000u && (bits & 0x007FFFFFu) != 0u)
      return (uint16_t)((bits >> 16) | 0x0040u);
    const uint32_t roundingBias = 0x7FFFu + ((bits >> 16) & 1u);
    return (uint16_t)((bits + roundingBias) >> 16);
  }

  inline float bf16ToFloat(uint16_t b) {
    uint32_t bits = (uint32_t)b << 16;
    float f;
    std::memcpy(&f, &bits, sizeof(f));
    return f;
  }

  // ---- numeric format -------------------------------------------------------

  // Bf16 runs the native bfloat16 MMUL. Bfp16 compiles the same bfloat16 kernel
  // against AIE_API_EMULATE_BFLOAT16_MMUL_WITH_BFP16, which doubles the MMUL
  // micro-kernel (measured 1.42x) and is XDNA2-only. The distinction is entirely
  // inside the kernel: host buffers are bfloat16 in / float32 out either way, so
  // only which artifact directory we load from changes.
  enum class Dtype { Auto, Bf16, Bfp16 };

  const char* dtypeName(Dtype dtype);  // "bf16" / "bfp16"

  // Parses a ryzenaiDtype config value. Returns false for anything unrecognized.
  bool parseDtype(const std::string& s, Dtype& out);

  // Resolves Auto against the hardware. See the implementation for why Auto is
  // currently bf16 even where bfp16 is available.
  Dtype resolveDtype(Dtype requested, RyzenAIDevice::Arch arch);

  // ---- engines --------------------------------------------------------------

  // One loaded xclbin, bound to a device, computing C[M,N] = A[M,K] * B[K,N]
  // with bfloat16 inputs and float32 accumulation for any M and N the shape
  // rules below allow. K is fixed by the binary.
  struct Engine;

  struct EngineInfo {
    int K = 0;       // the reduction dim this binary was compiled for
    int cols = 0;    // AIE columns acquired
    int tileM = 0;   // MMUL tile geometry, needed to pad M and N
    int tileK = 0;
    int tileN = 0;
    Dtype dtype = Dtype::Bf16;
  };

  // Smallest M >= want that an instruction stream can express, and likewise for
  // N. Callers zero-pad up to these; the extra rows and columns of C are
  // computed and ignored. K is not paddable here -- pick an engine whose K is
  // >= the layer's K and zero-pad A's columns and B's rows into it.
  int padM(const EngineInfo& info, int want);
  int padN(const EngineInfo& info, int want);

  // Loads the artifact for reduction dim K: an exact match if one exists,
  // otherwise the smallest available K greater than it (the caller then
  // zero-pads). Column counts are tried widest-first down to 1, since a device
  // whose columns are partly claimed by another process should still get NPU
  // acceleration rather than falling back to the CPU. maxCols caps that search;
  // pass 0 for no cap. Wider is not always faster at small M -- see
  // references/performance.md.
  //
  // Returns nullptr and fills err when nothing loads; that is an ordinary
  // outcome (callers use the CPU reference path), not a fatal error.
  Engine* loadEngine(
    const std::string& artifactDir,
    int deviceIdx,
    Dtype dtype,
    int K,
    int maxCols,
    EngineInfo& infoOut,
    std::string& err
  );
  void freeEngine(Engine* engine);
  const EngineInfo& engineInfo(const Engine* engine);

  // Same as loadEngine, but loads the SwiGLU-epilogue variant of the GEMM
  // (gemm_swiglu_bf16_K<K>, in the <variant>_swiglu directories). That binary
  // computes silu(l)*g on chip for B whose columns interleave two weight
  // matrices in groups of 8; see python/ryzenai_kernels/gemm_swiglu_bf16.py.
  // It is a separate xclbin and therefore a separate hardware context.
  Engine* loadEngineSwiglu(
    const std::string& artifactDir,
    int deviceIdx,
    Dtype dtype,
    int K,
    int maxCols,
    EngineInfo& infoOut,
    std::string& err
  );

  // Reduction dims for which a SwiGLU-epilogue artifact exists. Sorted.
  std::vector<int> listSwigluK(const std::string& artifactDir);

  // Device-resident B for one layer. Weights do not change between evaluations,
  // so they are uploaded once and bound per dispatch. N is the padded width.
  struct Weights;
  Weights* uploadWeights(Engine* engine, int paddedN, const uint16_t* B, std::string& err);
  void freeWeights(Weights* weights);

  // Rewrites the contents of an already-uploaded B in place. This is the path
  // for operands that change every dispatch (attention keys/values, as opposed
  // to layer weights): the BO itself does not move, so the xrt::run cached
  // against it in runGemm stays valid and only a memcpy + sync is paid.
  void rewriteWeights(Weights* weights, const uint16_t* B);

  // ---- standalone ops -------------------------------------------------------
  // A non-GEMM kernel (softmax, fused attention, ...) whose instruction stream
  // is precompiled next to the xclbin -- unlike the engines nothing is
  // generated at run time, so the op computes exactly the one shape it was
  // compiled for. Same opcode-3 ABI as the GEMM except that the data args are
  // arg3..arg3+numIns-1 = inputs and arg3+numIns = output. Buffer sizes are
  // fixed at load.
  struct Op;
  Op* loadOp(
    const std::string& xclbinPath, const std::string& instsPath, int deviceIdx,
    const size_t* inBytes, int numIns, size_t bytesC, std::string& err);
  void freeOp(Op* op);
  // memcpy the inputs in, dispatch, memcpy the output out. ins[j] must point
  // at inBytes[j] bytes. Throws StringError on a dispatch failure.
  void runOp(Op* op, const void* const* ins, int numIns, void* C);

  // C[M,N] = A[M,K] * B[K,N], all row-major, with M and N already padded per
  // padM/padN and A's row stride equal to the engine's K. Blocks until the NPU
  // has finished. Throws StringError on a dispatch failure.
  void runGemm(
    Engine* engine, Weights* weights, int paddedM, int paddedN, const uint16_t* A, float* C);

  // Wall clock accumulated inside runGemm, split so that host<->device transfer
  // can be told apart from the NPU's own execution. Deciding whether keeping
  // activations resident on the device is worth it needs exactly this split.
  struct Timings {
    double secsUploadA = 0.0;
    double secsExecute = 0.0;
    double secsDownloadC = 0.0;
    long long numDispatches = 0;
  };
  const Timings& engineTimings(const Engine* engine);

  // Reduction dims for which an artifact exists under artifactDir, parsed from
  // the filenames. Sorted, for reproducible logging.
  std::vector<int> listGemmK(const std::string& artifactDir);

  // Loads every available artifact, runs it against a plain-C++ GEMM on
  // pseudorandom data, and times it. Returns a human-readable report; never
  // throws, so it is safe to call from a logging path.
  std::string selfTest(const std::string& artifactDir, int deviceIdx);

}  // namespace RyzenAIKernel

#endif  // NEURALNET_RYZENAI_KERNEL_H_
