// CUTLASS dual-GEMM + SwiGLU epilogue implementation of CudaFusedFFN (see cudafusedffn.h).
// Built on the DualGemm extension from CUTLASS's example 45 (vendored under
// external/cutlass/examples/45_dual_gemm, included via CMake): both GEMMs share the A operand
// tile pipeline, and the epilogue combines their accumulators as SiLU(D0) * D1 so the
// intermediates never touch global memory.
//
// A single tile configuration (threadblock 128x64x32, warp 64x32x32, 3 stages, FP16
// accumulation) is instantiated: it measured fastest on every architecture tested
// (sm_86 A5000, sm_89 4090, sm_90 H100, sm_120 RTX PRO 6000), 1.4-1.9x the unfused
// cublasHgemm x2 + SwiGLU kernel sequence at KataGo's FFN shapes.

#include "../neuralnet/cudafusedffn.h"

#include <cstdint>
#include <stdexcept>
#include <string>

#include "cutlass/cutlass.h"
#include "cutlass/gemm/gemm.h"
#include "cutlass/epilogue/thread/linear_combination.h"
#include "device/dual_gemm.h"
#include "thread/left_silu_and_mul.h"

namespace {

using ElementT = cutlass::half_t;
static constexpr int kAlignment = 128 / cutlass::sizeof_bits<ElementT>::value;

using EpilogueOutputOp01 = cutlass::epilogue::thread::LinearCombination<
  ElementT, kAlignment, ElementT, ElementT,
  cutlass::epilogue::thread::ScaleType::Nothing>;
using EpilogueOutputOp2 = cutlass::epilogue::thread::LeftSiLUAndMul<
  ElementT, kAlignment, ElementT, ElementT>;

using DualGemm = cutlass::gemm::device::DualGemm<
  ElementT, cutlass::layout::RowMajor,
  ElementT, cutlass::layout::ColumnMajor, cutlass::layout::ColumnMajor,
  ElementT, cutlass::layout::RowMajor,
  ElementT,
  cutlass::arch::OpClassTensorOp,
  cutlass::arch::Sm80,
  cutlass::gemm::GemmShape<128, 64, 32>,
  cutlass::gemm::GemmShape<64, 32, 32>,
  cutlass::gemm::GemmShape<16, 8, 16>,
  EpilogueOutputOp01, EpilogueOutputOp01, EpilogueOutputOp2,
  cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<1>,
  3,      // stages
  false,  // kStoreD0
  false,  // kStoreD1
  false   // kSplitKSerial
>;

DualGemm::Arguments makeArgs(
  const half* A, const half* w1, const half* wGate, half* out, int M, int N, int K
) {
  return DualGemm::Arguments(
    cutlass::gemm::DualGemmMode::kGemm,
    {M, N, K},
    {(ElementT const*)A, K},
    {(ElementT const*)w1, K},
    cutlass::TensorRef<ElementT const, cutlass::layout::RowMajor>(),
    cutlass::TensorRef<ElementT, cutlass::layout::RowMajor>(),
    {(ElementT const*)wGate, K},
    cutlass::TensorRef<ElementT const, cutlass::layout::RowMajor>(),
    cutlass::TensorRef<ElementT, cutlass::layout::RowMajor>(),
    {(ElementT*)out, N},
    EpilogueOutputOp01::Params(),
    EpilogueOutputOp01::Params(),
    EpilogueOutputOp2::Params(),
    1);
}

bool argsAreImplementable(const DualGemm::Arguments& args) {
  if(DualGemm::can_implement(args) != cutlass::Status::kSuccess)
    return false;
  // With split-K off no workspace is required. A nonzero requirement would mean the
  // configuration changed, so treat it as unsupported rather than allocating here.
  if(DualGemm::get_workspace_size(args) != 0)
    return false;
  return true;
}

}  // namespace

namespace CudaFusedFFN {

bool supportedOnCurrentDevice() {
  // Never launch on a pre-sm_80 device: unlike KataGo's own arch-guarded kernels, CUTLASS's
  // below-arch fallback path is not an empty stub but a device-side trap
  // (CUTLASS_NOT_IMPLEMENTED), which would leave a sticky, unclearable error on the context.
  {
    int device = 0;
    int major = 0;
    if(cudaGetDevice(&device) != cudaSuccess)
      return false;
    if(cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device) != cudaSuccess)
      return false;
    if(major < 8)
      return false;
  }
  // Run a real tiny dual GEMM rather than a trivial probe kernel: this exercises the actual
  // kernel launch, including its 48KB+ dynamic shared memory opt-in, which a lighter probe
  // would not. With all-zero inputs the epilogue writes SiLU(0)*0 == 0, so pre-filling the
  // output with a nonzero pattern and checking it became zero verifies the kernel body truly
  // executed (a launch of an empty JIT stub would still report success).
  constexpr int M = 16;
  constexpr int N = 64;
  constexpr int K = 64;
  constexpr size_t numIn = (size_t)M * K + 2 * (size_t)N * K;
  constexpr size_t numOut = (size_t)M * N;
  half* buf = nullptr;
  if(cudaMalloc(&buf, (numIn + numOut) * sizeof(half)) != cudaSuccess)
    return false;
  half* A = buf;
  half* w1 = A + (size_t)M * K;
  half* wGate = w1 + (size_t)N * K;
  half* out = wGate + (size_t)N * K;

  bool ok = cudaMemset(buf, 0, numIn * sizeof(half)) == cudaSuccess;
  ok = ok && cudaMemset(out, 0xFF, numOut * sizeof(half)) == cudaSuccess;
  if(ok) {
    DualGemm::Arguments args = makeArgs(A, w1, wGate, out, M, N, K);
    ok = argsAreImplementable(args);
    if(ok) {
      DualGemm op;
      ok = op.initialize(args, nullptr, nullptr) == cutlass::Status::kSuccess;
      ok = ok && op.run(nullptr) == cutlass::Status::kSuccess;
    }
  }
  half hostOut[numOut];
  ok = ok && cudaMemcpy(hostOut, out, numOut * sizeof(half), cudaMemcpyDeviceToHost) == cudaSuccess;
  if(ok) {
    for(size_t i = 0; i < numOut; i++) {
      if(__half2float(hostOut[i]) != 0.0f) {
        ok = false;
        break;
      }
    }
  }
  // Clear any recoverable (non-sticky) launch error so an unsupported probe cannot leak error
  // state into later, unrelated CUDA calls. A device-side trap would be sticky and unclearable,
  // but the compute capability check above prevents the known trap path.
  (void)cudaGetLastError();
  (void)cudaFree(buf);
  return ok;
}

bool supportsShape(int N, int K) {
  if(N <= 0 || K <= 0)
    return false;
  // can_implement depends on the dimensions and on operand alignment. M does not affect it
  // beyond positivity (A is row-major with lda == K, and the kernel predicates partial M
  // tiles), so a fixed representative M stands in for the runtime batch. The dummy pointers
  // carry the 16-byte alignment that the backend's device allocations guarantee.
  const int M = 128;
  const half* dummyA = reinterpret_cast<const half*>(uintptr_t(256));
  const half* dummyW = reinterpret_cast<const half*>(uintptr_t(256));
  half* dummyOut = reinterpret_cast<half*>(uintptr_t(256));
  return argsAreImplementable(makeArgs(dummyA, dummyW, dummyW, dummyOut, M, N, K));
}

void runSwiGLU(
  const half* A, const half* w1, const half* wGate, half* out,
  int M, int N, int K, cudaStream_t stream
) {
  // supportsShape probed can_implement with 16-byte-aligned dummy pointers, so enforce the same
  // alignment on the real operands rather than leaving it a documentation-only contract.
  if((uintptr_t(A) | uintptr_t(w1) | uintptr_t(wGate) | uintptr_t(out)) & 15)
    throw std::runtime_error("CudaFusedFFN::runSwiGLU: operand pointers must be 16-byte aligned");
  DualGemm::Arguments args = makeArgs(A, w1, wGate, out, M, N, K);
  DualGemm op;
  cutlass::Status status = op.initialize(args, nullptr, stream);
  // run() repeats a cheap cudaFuncSetAttribute driver call on every invocation. Hoisting it
  // would require replicating the vendored kernel-params construction outside DualGemm, which
  // is not worth the coupling for ~1us of host time per launch.
  if(status == cutlass::Status::kSuccess)
    status = op.run(stream);
  if(status != cutlass::Status::kSuccess) {
    // The caller checked shape support at model load, so this is a genuine failure (a CUDA
    // error or an unexpected argument rejection), never a condition to silently fall back on.
    // Note run() ends with cudaGetLastError, so kErrorInternal may also reflect a pending
    // async error from an earlier kernel on this stream rather than this GEMM itself.
    throw std::runtime_error(
      std::string("CUTLASS fused FFN kernel failed (or a prior CUDA error was pending): ") +
      cutlass::cutlassGetStatusString(status) +
      ", M=" + std::to_string(M) + " N=" + std::to_string(N) + " K=" + std::to_string(K));
  }
}

}  // namespace CudaFusedFFN
