// CK DPP GEMM instances, see rocmckgemm.h. Built as its own object only for the gfx103x archs of
// the build (CK's DPP GEMM targets RDNA2 and its headers are slow to compile), listed in
// KATAGO_ROCM_CK_GEMM_ARCHS by CMakeLists.txt.

#include "../neuralnet/rocmckgemm.h"

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

// In CK's terms each instance computes c[m][n] = sum_k a[m][k] * b[n][k] with A = the weights
// ([outC][inC], row-major), B = the activations ([tokens][inC], CK's column-major B), and C = the
// output as column-major [outC][tokens], i.e. KataGo's row-major [tokens][outC]. With M along the
// output channels, each thread's 8 consecutive M values of C are contiguous in memory, so C is
// stored (and for Add, read) 16 bytes at a time, which measured 0-17% faster than the transposed
// formulation (M along tokens, C row-major), whose C accesses can only be scalar.
//
// Tile configurations: the subset of CK's device_gemm_dpp_f16_f16_f16_mk_nk_mn instance list that
// measured fastest on gfx1030 for 361 to 23104 tokens and 256 to 1536 channels, with MN padding so
// that the token count (batch * board area) need not be a tile multiple. Instantiated directly as
// the gridwise GEMM that DeviceGemmDpp wraps, because DeviceGemmDpp fixes the C memory operation
// to Set, while accumulating needs Add.
template<ck::InMemoryDataOperationEnum COp, ck::index_t BS, ck::index_t MB, ck::index_t NB, ck::index_t KB,
         ck::index_t MD, ck::index_t ND, ck::index_t MW, ck::index_t NW, ck::index_t AC1, ck::index_t BC1>
struct DppGemm {
  static constexpr ck::index_t BlockSize = BS;
  // As DeviceGemmDpp<F16, F16, F16, F32, Row, Col, Col, PassThrough x3, MNPadding, BS, MB, NB, KB,
  // 8, 8, MD, ND, MW, NW, S<4, AC1, 1>, S<1, 0, 2>, S<1, 0, 2>, 2, 8, 8, true, S<4, BC1, 1>,
  // S<1, 0, 2>, S<1, 0, 2>, 2, 8, 8, true, 4, 8> would instantiate it: C vectorized along M2
  // (dimension 4 of the per-thread C slice, 8 elements).
  using Gridwise = ck::GridwiseGemm_ak0mak1_bk0nbk1_mn_dpp<
    BS, F16, F32, F16, COp, Row, Col, Col, PassThrough, PassThrough, PassThrough, MNPadding,
    MB, NB, KB, MD, ND, 8, 8, MW, NW,
    S<4, AC1, 1>, S<1, 0, 2>, S<1, 0, 2>, 2, 8, 8, false, true,
    S<4, BC1, 1>, S<1, 0, 2>, S<1, 0, 2>, 2, 8, 8, false, true,
    S<0, 2, 4, 1, 3, 5>, 4, 8>;
};

template<ck::InMemoryDataOperationEnum COp>
struct Instances {
  using I0 = DppGemm<COp, 256, 64, 64, 64, 16, 16, 1, 2, 64, 64>;
  using I1 = DppGemm<COp, 128, 64, 64, 64, 32, 8, 2, 2, 32, 32>;
  using I2 = DppGemm<COp, 128, 32, 64, 64, 16, 16, 1, 2, 16, 32>;
  using I3 = DppGemm<COp, 128, 32, 32, 32, 32, 8, 1, 1, 32, 32>;
};

constexpr int NUM_INSTANCES = 4;

template<typename Inst>
bool runInstance(
  bool checkOnly, const half* in, int inStride, const half* w, int wStride, half* out, int outStride,
  int tokens, int outChannels, int inChannels, hipStream_t stream
) {
  using Gridwise = typename Inst::Gridwise;
  const typename Gridwise::Argument arg(
    reinterpret_cast<const F16*>(w), reinterpret_cast<const F16*>(in), reinterpret_cast<F16*>(out),
    outChannels, tokens, inChannels, wStride, inStride, outStride);
  if(!Gridwise::CheckValidity(arg))
    return false;
  if(checkOnly)
    return true;
  // As DeviceGemmDpp's invoker launches it.
  const auto [gdx, gdy, gdz] = Gridwise::CalculateGridSize(outChannels, tokens);
  if(Gridwise::CalculateHasMainKBlockLoop(inChannels))
    ck::kernel_gemm_dpp<Gridwise, true><<<dim3(gdx, gdy, gdz), dim3(Inst::BlockSize), 0, stream>>>(arg);
  else
    ck::kernel_gemm_dpp<Gridwise, false><<<dim3(gdx, gdy, gdz), dim3(Inst::BlockSize), 0, stream>>>(arg);
  return true;
}

template<ck::InMemoryDataOperationEnum COp>
bool runIndexed(
  int instance, bool checkOnly, const half* in, int inStride, const half* w, int wStride, half* out, int outStride,
  int tokens, int outChannels, int inChannels, hipStream_t stream
) {
  using Is = Instances<COp>;
  switch(instance) {
    case 0: return runInstance<typename Is::I0>(checkOnly, in, inStride, w, wStride, out, outStride, tokens, outChannels, inChannels, stream);
    case 1: return runInstance<typename Is::I1>(checkOnly, in, inStride, w, wStride, out, outStride, tokens, outChannels, inChannels, stream);
    case 2: return runInstance<typename Is::I2>(checkOnly, in, inStride, w, wStride, out, outStride, tokens, outChannels, inChannels, stream);
    case 3: return runInstance<typename Is::I3>(checkOnly, in, inStride, w, wStride, out, outStride, tokens, outChannels, inChannels, stream);
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

int rocmCkGemmNumInstances() {
  return NUM_INSTANCES;
}

bool rocmCkGemmSupports(int instance, int tokens, int outChannels, int inChannels, int inStride, int wStride, int outStride) {
  // The vectorized loads and stores need 8-element (16-byte) aligned rows, which CK's shape check
  // does not cover for C. Set and Add instances share their shape constraints.
  if(outChannels % 8 != 0 || inStride % 8 != 0 || wStride % 8 != 0 || outStride % 8 != 0)
    return false;
  return runIndexed<ck::InMemoryDataOperationEnum::Set>(
    instance, true, nullptr, inStride, nullptr, wStride, nullptr, outStride, tokens, outChannels, inChannels, nullptr);
}

void rocmCkGemmRun(
  int instance, bool accumulate,
  const half* in, int inStride, const half* w, int wStride, half* out, int outStride,
  int tokens, int outChannels, int inChannels, hipStream_t stream
) {
  const bool ran = accumulate
    ? runIndexed<ck::InMemoryDataOperationEnum::Add>(
        instance, false, in, inStride, w, wStride, out, outStride, tokens, outChannels, inChannels, stream)
    : runIndexed<ck::InMemoryDataOperationEnum::Set>(
        instance, false, in, inStride, w, wStride, out, outStride, tokens, outChannels, inChannels, stream);
  if(!ran)
    throw std::runtime_error("rocmCkGemmRun: instance does not support this problem");
}
