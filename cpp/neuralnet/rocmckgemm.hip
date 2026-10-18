// CK DPP GEMM instances, see rocmckgemm.h. Built as its own object only for the gfx103x archs of
// the build (CK's DPP GEMM targets RDNA2 and its headers are slow to compile), listed in
// KATAGO_ROCM_CK_GEMM_ARCHS by CMakeLists.txt.

#include "../neuralnet/rocmckgemm.h"

#include <cstring>
#include <stdexcept>
#include <string>

#include "ck/ck.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/tensor_operation/gpu/device/gemm_specialization.hpp"
#include "ck/tensor_operation/gpu/device/impl/device_gemm_dpp.hpp"
#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"

#ifndef KATAGO_ROCM_CK_GEMM_ARCHS
#error "KATAGO_ROCM_CK_GEMM_ARCHS must list the archs this file is compiled for, separated by ';'"
#endif

namespace {

using F16 = ck::half_t;
using F32 = float;
using Row = ck::tensor_layout::gemm::RowMajor;
using Col = ck::tensor_layout::gemm::ColumnMajor;
template <ck::index_t... Is> using S = ck::Sequence<Is...>;
using PassThrough = ck::tensor_operation::element_wise::PassThrough;
constexpr auto MNPadding = ck::tensor_operation::device::GemmSpecialization::MNPadding;

// a[m, k] * b[n, k] = c[m, n] instances, the subset of CK's device_gemm_dpp_f16_f16_f16_mk_nk_mn
// instance list that measured fastest on gfx1030 for M in [361, 23104] and N, K in [256, 1536],
// with MN padding so that M (batch * board area) need not be a tile multiple.
#define KATAGO_CK_DPP_GEMM(BS, MB, NB, KB, MD, ND, MW, NW, AC1, BC1) \
  ck::tensor_operation::device::DeviceGemmDpp<F16, F16, F16, F32, Row, Col, Row, PassThrough, PassThrough, PassThrough, \
    MNPadding, BS, MB, NB, KB, 8, 8, MD, ND, MW, NW, \
    S<4, AC1, 1>, S<1, 0, 2>, S<1, 0, 2>, 2, 8, 8, true, \
    S<4, BC1, 1>, S<1, 0, 2>, S<1, 0, 2>, 2, 8, 8, true, 5, 1>

using Inst0 = KATAGO_CK_DPP_GEMM(256, 64, 64, 64, 16, 16, 1, 2, 64, 64);
using Inst1 = KATAGO_CK_DPP_GEMM(128, 64, 64, 64, 32, 8, 2, 2, 32, 32);
using Inst2 = KATAGO_CK_DPP_GEMM(128, 32, 64, 64, 16, 16, 1, 2, 16, 32);
using Inst3 = KATAGO_CK_DPP_GEMM(128, 32, 32, 32, 32, 8, 1, 1, 32, 32);
#undef KATAGO_CK_DPP_GEMM

constexpr int NUM_INSTANCES = 4;

template<typename Op>
bool runInstance(bool checkOnly, const half* A, int lda, const half* B, int ldb, half* C, int ldc, int M, int N, int K, hipStream_t stream) {
  auto arg = Op::MakeArgument(
    reinterpret_cast<const F16*>(A), reinterpret_cast<const F16*>(B), reinterpret_cast<F16*>(C),
    M, N, K, lda, ldb, ldc, PassThrough{}, PassThrough{}, PassThrough{});
  // Only the shape check: Op::IsSupportedArgument also queries the device name on every call.
  if(!Op::GridwiseGemm::CheckValidity(arg))
    return false;
  if(!checkOnly)
    Op::MakeInvoker().Run(arg, StreamConfig{stream, false});
  return true;
}

bool runIndexed(int instance, bool checkOnly, const half* A, int lda, const half* B, int ldb, half* C, int ldc, int M, int N, int K, hipStream_t stream) {
  switch(instance) {
    case 0: return runInstance<Inst0>(checkOnly, A, lda, B, ldb, C, ldc, M, N, K, stream);
    case 1: return runInstance<Inst1>(checkOnly, A, lda, B, ldb, C, ldc, M, N, K, stream);
    case 2: return runInstance<Inst2>(checkOnly, A, lda, B, ldb, C, ldc, M, N, K, stream);
    case 3: return runInstance<Inst3>(checkOnly, A, lda, B, ldb, C, ldc, M, N, K, stream);
    default: return false;
  }
}

}  // namespace

bool rocmCkGemmSupportsArch(const char* gcnArchName) {
  std::string arch(gcnArchName);
  size_t colonPos = arch.find(':');
  if(colonPos != std::string::npos)
    arch = arch.substr(0, colonPos);
  std::string list = KATAGO_ROCM_CK_GEMM_ARCHS;
  size_t start = 0;
  while(start <= list.size()) {
    size_t end = list.find(';', start);
    if(end == std::string::npos)
      end = list.size();
    std::string entry = list.substr(start, end - start);
    size_t entryColon = entry.find(':');
    if(entryColon != std::string::npos)
      entry = entry.substr(0, entryColon);
    if(!entry.empty() && entry == arch)
      return true;
    start = end + 1;
  }
  return false;
}

int rocmCkGemmTune(
  const half* A, int lda, const half* B, int ldb, half* C, int ldc, int M, int N, int K, hipStream_t stream,
  float* bestMs
) {
  hipEvent_t start, stop;
  if(hipEventCreate(&start) != hipSuccess)
    throw std::runtime_error("rocmCkGemmTune: hipEventCreate failed");
  if(hipEventCreate(&stop) != hipSuccess) {
    (void)hipEventDestroy(start);
    throw std::runtime_error("rocmCkGemmTune: hipEventCreate failed");
  }
  int best = -1;
  float bestInstanceMs = 0.0f;
  for(int i = 0; i < NUM_INSTANCES; i++) {
    if(!runIndexed(i, true, A, lda, B, ldb, C, ldc, M, N, K, stream))
      continue;
    // One untimed run, then the best of three, so a slow first launch does not decide.
    runIndexed(i, false, A, lda, B, ldb, C, ldc, M, N, K, stream);
    float minMs = 1e30f;
    for(int rep = 0; rep < 3; rep++) {
      (void)hipEventRecord(start, stream);
      runIndexed(i, false, A, lda, B, ldb, C, ldc, M, N, K, stream);
      (void)hipEventRecord(stop, stream);
      (void)hipEventSynchronize(stop);
      float ms = 0.0f;
      (void)hipEventElapsedTime(&ms, start, stop);
      minMs = std::min(minMs, ms);
    }
    if(best < 0 || minMs < bestInstanceMs) {
      best = i;
      bestInstanceMs = minMs;
    }
  }
  (void)hipEventDestroy(start);
  (void)hipEventDestroy(stop);
  if(hipGetLastError() != hipSuccess)
    throw std::runtime_error("rocmCkGemmTune: kernel launch failed");
  *bestMs = bestInstanceMs;
  return best;
}

void rocmCkGemmRun(
  int instance, const half* A, int lda, const half* B, int ldb, half* C, int ldc, int M, int N, int K, hipStream_t stream
) {
  if(!runIndexed(instance, false, A, lda, B, ldb, C, ldc, M, N, K, stream))
    throw std::runtime_error("rocmCkGemmRun: instance does not support this problem");
}
