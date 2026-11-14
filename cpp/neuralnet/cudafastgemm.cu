// See cudafastgemm.h.

#include "../neuralnet/cudafastgemm.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <mutex>
#include <stdexcept>
#include <tuple>
#include <type_traits>
#include <vector>

#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm_universal.h"
#include "cutlass/gemm/threadblock/threadblock_swizzle_streamk.h"
#include "cutlass/epilogue/thread/linear_combination.h"

namespace {

using T = cutlass::half_t;
using RM = cutlass::layout::RowMajor;

// FP16 in/out with FP16 accumulation (the same precision as cublasHgemm), 128-bit accesses.
template<class TB, class WS, int Stages, bool StreamK>
struct Cfg {
  using Swz = typename std::conditional<StreamK, cutlass::gemm::threadblock::ThreadblockSwizzleStreamK,
                                        cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<8>>::type;
  using Gemm = cutlass::gemm::device::GemmUniversal<T, RM, T, RM, T, RM, T,
      cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80, TB, WS, cutlass::gemm::GemmShape<16, 8, 16>,
      cutlass::epilogue::thread::LinearCombination<T, 8, T, T>, Swz, Stages, 8, 8>;
};

// Row-major view of the column-major call: D[n x m] = X[n x k] * Wt[k x m] (+ D).
template<class G>
typename G::Arguments makeArgs(int m, int n, int k, const half* W, const half* X, half* C, bool acc) {
  return typename G::Arguments(
    cutlass::gemm::GemmUniversalMode::kGemm, {n, m, k}, 1,
    {T(1.0f), T(acc ? 1.0f : 0.0f)},
    (const T*)X, (const T*)W, (const T*)C, (T*)C,
    (int64_t)n * k, (int64_t)k * m, (int64_t)n * m, (int64_t)n * m,
    k, m, m, m);
}

template<class G>
size_t workspaceFor(int m, int n, int k, const half* W, const half* X, half* C, bool acc) {
  auto args = makeArgs<G>(m, n, k, W, X, C, acc);
  if(G::can_implement(args) != cutlass::Status::kSuccess)
    return SIZE_MAX;
  return G::get_workspace_size(args);
}

template<class G>
cutlass::Status runG(int m, int n, int k, const half* W, const half* X, half* C, bool acc, void* ws, cudaStream_t s) {
  auto args = makeArgs<G>(m, n, k, W, X, C, acc);
  G op;
  cutlass::Status st = op.initialize(args, ws, s);
  if(st == cutlass::Status::kSuccess)
    st = op.run(s);
  return st;
}

// Each of these won at least one KataGo transformer shape on an RTX 4060 Ti (sm_89). The
// 3-stage 128x128 tile is the most frequent winner: its smaller shared memory footprint fits
// two blocks per SM where the 4-stage one fits only one. Stream-K variants win where the tile
// count leaves a poor last wave.
#define TB(a, b, c) cutlass::gemm::GemmShape<a, b, c>
using G0 = Cfg<TB(128, 128, 32), TB(64, 64, 32), 4, false>::Gemm;
using G1 = Cfg<TB(64, 128, 32), TB(32, 64, 32), 4, false>::Gemm;
using G2 = Cfg<TB(128, 128, 32), TB(64, 64, 32), 3, false>::Gemm;
using G3 = Cfg<TB(64, 128, 32), TB(32, 64, 32), 3, false>::Gemm;
using G4 = Cfg<TB(128, 64, 32), TB(64, 32, 32), 3, false>::Gemm;
using G5 = Cfg<TB(64, 64, 32), TB(32, 32, 32), 6, false>::Gemm;
using G6 = Cfg<TB(128, 128, 32), TB(64, 64, 32), 4, true>::Gemm;
using G7 = Cfg<TB(128, 64, 32), TB(64, 32, 32), 4, true>::Gemm;
using G8 = Cfg<TB(64, 128, 32), TB(32, 64, 32), 4, true>::Gemm;
using G9 = Cfg<TB(128, 128, 64), TB(64, 64, 64), 3, true>::Gemm;
using G10 = Cfg<TB(128, 128, 32), TB(64, 64, 32), 3, true>::Gemm;
using G11 = Cfg<TB(64, 128, 32), TB(32, 64, 32), 3, true>::Gemm;
#undef TB

struct Candidate {
  const char* name;
  size_t (*workspace)(int, int, int, const half*, const half*, half*, bool);
  cutlass::Status (*run)(int, int, int, const half*, const half*, half*, bool, void*, cudaStream_t);
};

const Candidate kCandidates[] = {
  {"128x128x32-s4", workspaceFor<G0>, runG<G0>},
  {"64x128x32-s4", workspaceFor<G1>, runG<G1>},
  {"128x128x32-s3", workspaceFor<G2>, runG<G2>},
  {"64x128x32-s3", workspaceFor<G3>, runG<G3>},
  {"128x64x32-s3", workspaceFor<G4>, runG<G4>},
  {"64x64x32-s6", workspaceFor<G5>, runG<G5>},
  {"128x128x32-s4-sk", workspaceFor<G6>, runG<G6>},
  {"128x64x32-s4-sk", workspaceFor<G7>, runG<G7>},
  {"64x128x32-s4-sk", workspaceFor<G8>, runG<G8>},
  {"128x128x64-s3-sk", workspaceFor<G9>, runG<G9>},
  {"128x128x32-s3-sk", workspaceFor<G10>, runG<G10>},
  {"64x128x32-s3-sk", workspaceFor<G11>, runG<G11>},
};
constexpr int kNumCandidates = (int)(sizeof(kCandidates) / sizeof(kCandidates[0]));
constexpr int kCublas = -1;

// (device, m, n, k, accumulate). Results are shared by all threads using the same device.
typedef std::tuple<int, int, int, int, bool> Key;
std::mutex gMutex;
std::map<Key, int> gChoice;

// Per-thread state. Each NN server thread owns its stream and synchronizes it at the end of every
// forward pass, so a per-thread stream-K workspace is never used by two kernels at once.
// Workspaces are never freed while the thread lives, only superseded: CUDA graphs captured
// earlier keep pointing at the buffer that was current at capture time.
struct ThreadState {
  std::map<Key, int> choice;
  std::map<int, std::pair<void*, size_t>> ws;  // device -> current workspace
  std::vector<void*> retired;
  ~ThreadState() {
    for(auto& kv: ws)
      if(kv.second.first != nullptr) (void)cudaFree(kv.second.first);
    for(void* p: retired)
      (void)cudaFree(p);
  }
};
thread_local ThreadState tState;

void* ensureWorkspace(int device, size_t bytes) {
  auto& cur = tState.ws[device];
  if(bytes <= cur.second)
    return cur.first;
  if(cur.first != nullptr)
    tState.retired.push_back(cur.first);
  size_t alloc = std::max(bytes, (size_t)4 << 20);
  void* p = nullptr;
  if(cudaMalloc(&p, alloc) != cudaSuccess) {
    (void)cudaGetLastError();
    cur = std::make_pair(nullptr, (size_t)0);
    return nullptr;
  }
  cur = std::make_pair(p, alloc);
  return p;
}

void runCublas(cublasHandle_t cublas, int m, int n, int k, const half* W, const half* X, half* C, bool acc) {
  const half alpha = __float2half(1.0f);
  const half beta = __float2half(acc ? 1.0f : 0.0f);
  cublasStatus_t st = cublasHgemm(cublas, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &alpha, W, m, X, k, &beta, C, m);
  if(st != CUBLAS_STATUS_SUCCESS)
    throw std::runtime_error("CudaFastGemm: cublasHgemm failed");
}

// Returns false, having launched nothing, if the CUTLASS candidate cannot run this problem.
bool runCandidate(int choice, int device, cudaStream_t stream, int m, int n, int k, const half* W, const half* X, half* C, bool acc) {
  const Candidate& c = kCandidates[choice];
  size_t wsBytes = c.workspace(m, n, k, W, X, C, acc);
  if(wsBytes == SIZE_MAX)
    return false;
  void* ws = nullptr;
  if(wsBytes > 0) {
    ws = ensureWorkspace(device, wsBytes);
    if(ws == nullptr)
      return false;
  }
  return c.run(m, n, k, W, X, C, acc, ws, stream) == cutlass::Status::kSuccess;
}

float maxAbsDiff(const half* a, const half* b, size_t count, float* maxAbsRef) {
  std::vector<half> ha(count), hb(count);
  cudaMemcpy(ha.data(), a, count * sizeof(half), cudaMemcpyDeviceToHost);
  cudaMemcpy(hb.data(), b, count * sizeof(half), cudaMemcpyDeviceToHost);
  float d = 0.0f, r = 0.0f;
  for(size_t i = 0; i < count; i++) {
    float x = __half2float(ha[i]), y = __half2float(hb[i]);
    if(!std::isfinite(y) || !std::isfinite(x)) {
      if(std::isfinite(x) != std::isfinite(y))
        d = INFINITY;
      continue;
    }
    d = std::max(d, std::fabs(x - y));
    r = std::max(r, std::fabs(x));
  }
  *maxAbsRef = r;
  return d;
}

// Times cuBLAS and every candidate on the real operands (C is never modified) and returns the
// fastest. A candidate is only eligible if its result matches cuBLAS.
int autotune(cublasHandle_t cublas, int device, cudaStream_t stream, int m, int n, int k, const half* W, const half* X, const half* C, bool acc) {
  const size_t count = (size_t)m * n;
  half* ref = nullptr;
  half* test = nullptr;
  if(cudaMalloc(&ref, count * sizeof(half)) != cudaSuccess || cudaMalloc(&test, count * sizeof(half)) != cudaSuccess) {
    (void)cudaGetLastError();
    if(ref != nullptr) cudaFree(ref);
    return kCublas;
  }
  cudaStreamSynchronize(stream);
  if(acc) cudaMemcpyAsync(ref, C, count * sizeof(half), cudaMemcpyDeviceToDevice, stream);
  runCublas(cublas, m, n, k, W, X, ref, acc);
  cudaStreamSynchronize(stream);

  cudaEvent_t e0, e1;
  cudaEventCreate(&e0);
  cudaEventCreate(&e1);
  auto timeChoice = [&](int choice) -> float {
    auto launch = [&]() {
      if(choice == kCublas) { runCublas(cublas, m, n, k, W, X, test, acc); return true; }
      return runCandidate(choice, device, stream, m, n, k, W, X, test, acc);
    };
    const int reps = 10;
    for(int i = 0; i < 3; i++)
      if(!launch()) return INFINITY;
    float best = INFINITY;
    for(int trial = 0; trial < 3; trial++) {
      cudaEventRecord(e0, stream);
      for(int i = 0; i < reps; i++)
        launch();
      cudaEventRecord(e1, stream);
      cudaEventSynchronize(e1);
      float ms = 0.0f;
      cudaEventElapsedTime(&ms, e0, e1);
      best = std::min(best, ms / reps);
    }
    return best;
  };

  int bestChoice = kCublas;
  float bestTime = timeChoice(kCublas);
  const float cublasTime = bestTime;
  for(int c = 0; c < kNumCandidates; c++) {
    if(acc) cudaMemcpyAsync(test, C, count * sizeof(half), cudaMemcpyDeviceToDevice, stream);
    if(!runCandidate(c, device, stream, m, n, k, W, X, test, acc)) { (void)cudaGetLastError(); continue; }
    if(cudaStreamSynchronize(stream) != cudaSuccess) { (void)cudaGetLastError(); continue; }
    float refMax = 0.0f;
    float diff = maxAbsDiff(ref, test, count, &refMax);
    if(!(diff <= 1e-2f * refMax + 1e-3f))
      continue;
    float t = timeChoice(c);
    if(t < bestTime) { bestTime = t; bestChoice = c; }
  }
  cudaEventDestroy(e0);
  cudaEventDestroy(e1);
  cudaFree(ref);
  cudaFree(test);
  (void)cudaGetLastError();
  if(getenv("KATAGO_CUDA_FAST_GEMM_LOG") != nullptr)
    fprintf(stderr, "CudaFastGemm device=%d m=%d n=%d k=%d acc=%d: cublas %.1fus -> %s %.1fus\n", device, m, n, k, (int)acc,
            cublasTime * 1000.0f, bestChoice == kCublas ? "cublas" : kCandidates[bestChoice].name, bestTime * 1000.0f);
  return bestChoice;
}

}  // namespace

namespace CudaFastGemm {

bool enabled() {
  static const bool on = [] { const char* e = getenv("KATAGO_CUDA_DISABLE_FAST_GEMM"); return e == nullptr || atoi(e) == 0; }();
  return on;
}

void hgemm(
  cublasHandle_t cublas, cudaStream_t stream,
  int m, int n, int k,
  const half* W, const half* X, half* C, bool accumulate
) {
  const bool aligned = ((uintptr_t(W) | uintptr_t(X) | uintptr_t(C)) & 15) == 0 && m % 8 == 0 && k % 8 == 0;
  int device = 0;
  if(!aligned || cudaGetDevice(&device) != cudaSuccess) {
    runCublas(cublas, m, n, k, W, X, C, accumulate);
    return;
  }
  Key key(device, m, n, k, accumulate);
  int choice;
  auto it = tState.choice.find(key);
  if(it != tState.choice.end())
    choice = it->second;
  else {
    std::lock_guard<std::mutex> lock(gMutex);
    auto git = gChoice.find(key);
    if(git != gChoice.end())
      choice = git->second;
    else {
      choice = autotune(cublas, device, stream, m, n, k, W, X, C, accumulate);
      gChoice[key] = choice;
    }
    tState.choice[key] = choice;
  }
  if(choice == kCublas || !runCandidate(choice, device, stream, m, n, k, W, X, C, accumulate)) {
    (void)cudaGetLastError();
    runCublas(cublas, m, n, k, W, X, C, accumulate);
  }
}

}  // namespace CudaFastGemm
