#include "../neuralnet/ryzenaikernel.h"

#include <algorithm>
#include <chrono>
#if defined(_M_X64) || defined(__x86_64__) || defined(__SSE2__)
#define RYZENAI_BF16_SSE2 1
#include <emmintrin.h>
#endif
#include <cstdio>
#include <fstream>
#include <map>
#include <memory>
#include <random>
#include <set>

#include "../core/fileutils.h"
#include "../core/global.h"
#include "../neuralnet/ryzenaimanifest.h"
#include "../neuralnet/ryzenaisequence.h"

#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable : 4100 4245 4267 4996)
#endif
#include "xrt/xrt_bo.h"
#include "xrt/xrt_device.h"
#include "xrt/xrt_hw_context.h"
#include "xrt/xrt_kernel.h"
#include "xrt/experimental/xrt_xclbin.h"
#ifdef _MSC_VER
#pragma warning(pop)
#endif

#include "../external/filesystem-1.5.8/include/ghc/filesystem.hpp"

namespace gfs = ghc::filesystem;

using namespace std;

namespace {

  string variantDir(
    const string& artifactDir, RyzenAIKernel::Dtype dtype, RyzenAIDevice::Arch arch, int cols,
    bool swiglu = false) {
    return (gfs::u8path(artifactDir) / RyzenAIKernel::dtypeName(dtype) /
            (string(RyzenAIDevice::archName(arch)) + "_" + Global::intToString(cols) + "col" +
             (swiglu ? "_swiglu" : "")))
      .u8string();
  }

  // Artifact file names carry the shape they were compiled at. Only K matters
  // to the binary, so the K is parsed out and the rest ignored -- which also
  // means the older gemm_bf16_M<M>K<K>N<N> names and a plain gemm_bf16_K<K>
  // both resolve correctly.
  bool parseKFromStem(const string& stem, int& kOut) {
    const size_t at = stem.rfind('K');
    if(at == string::npos)
      return false;
    size_t end = at + 1;
    while(end < stem.size() && stem[end] >= '0' && stem[end] <= '9')
      end++;
    if(end == at + 1)
      return false;
    try {
      kOut = Global::stringToInt(stem.substr(at + 1, end - at - 1));
    }
    catch(const std::exception&) {
      return false;
    }
    return kOut > 0;
  }

  // The grid names artifacts by the only dimension that matters to the binary.
  bool isCanonicalStem(const string& stem, int k) {
    return stem == "gemm_bf16_K" + Global::intToString(k);
  }

  // K -> xclbin path for one variant directory.
  map<int, string> scanVariant(const string& dir) {
    map<int, string> byK;
    std::set<int> canonical;
    std::error_code ec;
    if(!gfs::is_directory(gfs::u8path(dir), ec))
      return byK;
    for(auto& entry : gfs::directory_iterator(gfs::u8path(dir), ec)) {
      if(!entry.is_regular_file(ec))
        continue;
      const gfs::path p = entry.path();
      if(p.extension().u8string() != ".xclbin")
        continue;
      const string stem = p.stem().u8string();
      int k = 0;
      if(!parseKFromStem(stem, k))
        continue;
      // Older artifacts are named for the full shape they happened to be
      // compiled at, which can collide with a grid entry for the same K. Both
      // are correct binaries, but prefer the grid's so the choice does not
      // depend on directory order.
      const bool canon = isCanonicalStem(stem, k);
      if(canon || byK.find(k) == byK.end()) {
        if(canon || canonical.find(k) == canonical.end())
          byK[k] = p.u8string();
        if(canon)
          canonical.insert(k);
      }
    }
    return byK;
  }

  // mlir-aie appends a generated suffix to the kernel name, so match on the
  // prefix over what the xclbin actually declares rather than assuming a
  // literal name.
  string findKernelName(const xrt::xclbin& xclbin, const string& prefix) {
    for(const auto& kernel : xclbin.get_kernels()) {
      string name = kernel.get_name();
      if(name.rfind(prefix, 0) == 0)
        return name;
    }
    return string();
  }

  int roundUpTo(int v, int q) {
    return ((v + q - 1) / q) * q;
  }

}  // namespace

struct RyzenAIKernel::Engine {
  RyzenAIKernel::EngineInfo info;
  RyzenAISequence::Arch seqArch = RyzenAISequence::Arch::NPU2;

  xrt::device device;
  xrt::xclbin xclbin;
  xrt::hw_context context;
  xrt::kernel kernel;

  // Scratch grown on demand. A and C are per-dispatch; the instruction stream
  // changes whenever (M, N) changes, so it is uploaded per dispatch too, but
  // only re-generated when the shape actually differs from the last one.
  xrt::bo boInstr;
  xrt::bo boA;
  xrt::bo boC;
  size_t capInstrBytes = 0;
  size_t capABytes = 0;
  size_t capCBytes = 0;

  vector<uint32_t> instr;
  int lastM = 0;
  int lastN = 0;

  // Bumped whenever a scratch BO is reallocated, so cached runs know their
  // bound arguments are stale.
  uint64_t bufferGeneration = 0;

  RyzenAIKernel::Timings timings;
};

struct RyzenAIKernel::Weights {
  xrt::bo bo;
  int paddedN = 0;
  size_t bytes = 0;  // K * paddedN * sizeof(uint16_t), for rewriteWeights

  // Building an xrt::run and binding six arguments costs about as much as the
  // GEMM itself at these sizes, and the binding only changes when a scratch BO
  // moves, so the run is cached per weight set.
  xrt::run run;
  bool runValid = false;
  uint64_t runGeneration = 0;
  uint32_t lastNumInstr = 0;
};

struct RyzenAIKernel::Op {
  xrt::device device;
  xrt::xclbin xclbin;
  xrt::hw_context context;
  xrt::kernel kernel;
  xrt::bo boInstr;
  std::vector<xrt::bo> boIns;
  xrt::bo boC;
  xrt::run run;
  std::vector<size_t> inBytes;
  size_t bytesC = 0;
};

void RyzenAIKernel::floatToBf16Bulk(const float* src, uint16_t* dst, size_t n) {
  size_t i = 0;
#ifdef RYZENAI_BF16_SSE2
  const __m128i expMask = _mm_set1_epi32(0x7F800000);
  const __m128i mantMask = _mm_set1_epi32(0x007FFFFF);
  const __m128i one = _mm_set1_epi32(1);
  const __m128i bias = _mm_set1_epi32(0x7FFF);
  const __m128i quiet = _mm_set1_epi32(0x0040);
  const __m128i zero = _mm_setzero_si128();
  // Packing 32-bit lanes down to 16 has no unsigned form in SSE2, so shift the
  // values into the signed range first and shift them back afterwards. Every
  // value is in [0, 0xFFFF] by then, so the bias lands them exactly inside
  // int16 and the saturating pack never actually saturates.
  const __m128i packBias = _mm_set1_epi32(0x8000);
  const __m128i unBias = _mm_set1_epi16((short)0x8000);

  auto round4 = [&](const float* p) {
    const __m128i bits = _mm_loadu_si128(reinterpret_cast<const __m128i*>(p));
    const __m128i isExpAllOnes = _mm_cmpeq_epi32(_mm_and_si128(bits, expMask), expMask);
    const __m128i mantIsZero = _mm_cmpeq_epi32(_mm_and_si128(bits, mantMask), zero);
    const __m128i isNan = _mm_andnot_si128(mantIsZero, isExpAllOnes);
    const __m128i lsb = _mm_and_si128(_mm_srli_epi32(bits, 16), one);
    const __m128i rounded =
      _mm_srli_epi32(_mm_add_epi32(_mm_add_epi32(bits, bias), lsb), 16);
    const __m128i nanVal = _mm_or_si128(_mm_srli_epi32(bits, 16), quiet);
    return _mm_or_si128(_mm_and_si128(isNan, nanVal), _mm_andnot_si128(isNan, rounded));
  };

  for(; i + 8 <= n; i += 8) {
    const __m128i lo = _mm_sub_epi32(round4(src + i), packBias);
    const __m128i hi = _mm_sub_epi32(round4(src + i + 4), packBias);
    _mm_storeu_si128(
      reinterpret_cast<__m128i*>(dst + i), _mm_add_epi16(_mm_packs_epi32(lo, hi), unBias));
  }
#endif
  for(; i < n; i++)
    dst[i] = floatToBf16(src[i]);
}

const char* RyzenAIKernel::dtypeName(Dtype dtype) {
  return dtype == Dtype::Bfp16 ? "bfp16" : "bf16";
}

bool RyzenAIKernel::parseDtype(const string& s, Dtype& out) {
  const string lowered = Global::toLower(Global::trim(s));
  if(lowered == "auto") { out = Dtype::Auto; return true; }
  if(lowered == "bf16") { out = Dtype::Bf16; return true; }
  if(lowered == "bfp16") { out = Dtype::Bfp16; return true; }
  return false;
}

RyzenAIKernel::Dtype RyzenAIKernel::resolveDtype(Dtype requested, RyzenAIDevice::Arch arch) {
  if(requested != Dtype::Auto)
    return requested;

  // Auto means BFP16 where the hardware has it, which is XDNA2 only.
  //
  // A single BFP16 GEMM carries roughly 3% relative error against a float32
  // reference, against ~2e-5 for bf16, and that gap is what kept the default on
  // bf16 until a whole network could be measured. It now has been, and the
  // compounding turns out to be mild: over the ~160 chained GEMMs of an
  // evaluation, BFP16 lands within about 1.3-2x of bf16's deviation from the
  // CPU reference rather than anywhere near 1000x (b10c384h6 policy mean
  // 4.1e-4 vs 2.5e-4; b28c512nbt 2.4e-4 vs 1.8e-4), and the policy top-10 set
  // is identical for both formats on both models. The per-GEMM error is input
  // quantisation, not accumulation - the accumulator is fp32 either way - so it
  // does not compound the way a truncated accumulator would.
  //
  // ryzenaiDtype = bf16 forces the more accurate format for anyone who wants
  // it, and is still the only option on XDNA1.
  if(arch == RyzenAIDevice::Arch::NPU2)
    return Dtype::Bfp16;
  return Dtype::Bf16;
}

int RyzenAIKernel::padM(const EngineInfo& info, int want) {
  // One pass over the 4 double-buffered AIE rows consumes tileM*8 rows.
  return roundUpTo(std::max(want, 1), info.tileM * 8);
}

int RyzenAIKernel::padN(const EngineInfo& info, int want) {
  return roundUpTo(std::max(want, 1), info.tileN * info.cols);
}

namespace {

  bool readWholeFile(const string& path, vector<char>& out) {
    try {
      if(!FileUtils::exists(path))
        return false;
      ifstream in;
      FileUtils::open(in, path, ios::in | ios::binary | ios::ate);
      streamsize size = in.tellg();
      if(size < 0)
        return false;
      in.seekg(0, ios::beg);
      out.resize((size_t)size);
      if(size > 0)
        in.read(out.data(), size);
      return in.good() || in.eof();
    }
    catch(const std::exception&) {
      return false;
    }
  }

  // One attempt at one (dtype, arch, cols). Picks the smallest available K that
  // is >= the requested one. With swiglu set, scans the SwiGLU-epilogue
  // variant's directory instead (same naming, gemm_swiglu_bf16_K<K> stems).
  RyzenAIKernel::Engine* tryLoadVariant(
    const string& artifactDir, int deviceIdx, RyzenAIKernel::Dtype dtype,
    RyzenAIDevice::Arch arch, int cols, int K, bool swiglu, string& err
  ) {
    err.clear();
    const map<int, string> byK = scanVariant(variantDir(artifactDir, dtype, arch, cols, swiglu));
    auto it = byK.lower_bound(K);
    if(it == byK.end()) {
      err = "no artifact with reduction dim >= " + Global::intToString(K) + " at " +
            Global::intToString(cols) + " column(s)";
      return nullptr;
    }

    try {
      unique_ptr<RyzenAIKernel::Engine> engine(new RyzenAIKernel::Engine());
      engine->info.K = it->first;
      engine->info.cols = cols;
      engine->info.dtype = dtype;
      // Every artifact in the set is built from the same whole_array design.
      // tileN 48 only appears on shapes whose N is a multiple of 48, and the
      // grid is compiled at tileN 32 throughout; manifest.json records the
      // actual geometry if that ever stops being true.
      engine->info.tileM = RyzenAIManifest::GEMM_TILE_M;
      engine->info.tileK = RyzenAIManifest::GEMM_TILE_K;
      engine->info.tileN = RyzenAIManifest::GEMM_TILE_N;
      engine->seqArch = (arch == RyzenAIDevice::Arch::NPU1) ? RyzenAISequence::Arch::NPU1
                                                            : RyzenAISequence::Arch::NPU2;

      if(!RyzenAISequence::supportsColumns(cols)) {
        err = "no instruction-stream layout for " + Global::intToString(cols) + " column(s)";
        return nullptr;
      }

      engine->device = xrt::device((unsigned int)(deviceIdx < 0 ? 0 : deviceIdx));
      engine->xclbin = xrt::xclbin(it->second);

      const string kernelName = findKernelName(engine->xclbin, RyzenAIManifest::KERNEL_NAME_PREFIX);
      if(kernelName.size() <= 0) {
        err = "no kernel named " + string(RyzenAIManifest::KERNEL_NAME_PREFIX) + "* in " + it->second;
        return nullptr;
      }

      engine->device.register_xclbin(engine->xclbin);
      engine->context = xrt::hw_context(engine->device, engine->xclbin.get_uuid());
      engine->kernel = xrt::kernel(engine->context, kernelName);
      return engine.release();
    }
    catch(const std::exception& e) {
      err = string("XRT failed to load ") + it->second + ": " + e.what();
      return nullptr;
    }
  }

}  // namespace

namespace {

  RyzenAIKernel::Engine* loadEngineImpl(
    const string& artifactDir, int deviceIdx, RyzenAIKernel::Dtype dtype, int K, int maxCols,
    bool swiglu, RyzenAIKernel::EngineInfo& infoOut, string& err
  ) {
    err.clear();

    const RyzenAIDevice::Arch arch = RyzenAIDevice::archOfDevice(deviceIdx);
    if(arch == RyzenAIDevice::Arch::Unknown) {
      err = "NPU architecture could not be determined, so no kernel binary can be selected";
      return nullptr;
    }
    const RyzenAIKernel::Dtype resolved = RyzenAIKernel::resolveDtype(dtype, arch);

    int startCols = RyzenAIDevice::maxColumns(arch);
    if(maxCols > 0)
      startCols = std::min(startCols, maxCols);

    string lastErr;
    for(int cols = startCols; cols >= 1; cols /= 2) {
      string variantErr;
      RyzenAIKernel::Engine* engine =
        tryLoadVariant(artifactDir, deviceIdx, resolved, arch, cols, K, swiglu, variantErr);
      if(engine != nullptr) {
        infoOut = engine->info;
        return engine;
      }
      if(lastErr.size() <= 0)
        lastErr = variantErr;
    }

    err = string("no usable ") + RyzenAIKernel::dtypeName(resolved) +
          (swiglu ? " swiglu-epilogue" : "") + " kernel for K=" + Global::intToString(K) +
          " on " + RyzenAIDevice::archName(arch) + " (" + lastErr + ")";
    return nullptr;
  }

}  // namespace

RyzenAIKernel::Engine* RyzenAIKernel::loadEngine(
  const string& artifactDir, int deviceIdx, Dtype dtype, int K, int maxCols,
  EngineInfo& infoOut, string& err
) {
  return loadEngineImpl(artifactDir, deviceIdx, dtype, K, maxCols, false, infoOut, err);
}

RyzenAIKernel::Engine* RyzenAIKernel::loadEngineSwiglu(
  const string& artifactDir, int deviceIdx, Dtype dtype, int K, int maxCols,
  EngineInfo& infoOut, string& err
) {
  return loadEngineImpl(artifactDir, deviceIdx, dtype, K, maxCols, true, infoOut, err);
}

void RyzenAIKernel::freeEngine(Engine* engine) {
  delete engine;
}

const RyzenAIKernel::EngineInfo& RyzenAIKernel::engineInfo(const Engine* engine) {
  return engine->info;
}

const RyzenAIKernel::Timings& RyzenAIKernel::engineTimings(const Engine* engine) {
  return engine->timings;
}

RyzenAIKernel::Weights* RyzenAIKernel::uploadWeights(
  Engine* engine, int paddedN, const uint16_t* B, string& err) {
  err.clear();
  try {
    unique_ptr<Weights> w(new Weights());
    w->paddedN = paddedN;
    w->bytes = (size_t)engine->info.K * (size_t)paddedN * sizeof(uint16_t);
    w->bo = xrt::bo(
      engine->device, w->bytes, XRT_BO_FLAGS_HOST_ONLY,
      engine->kernel.group_id(RyzenAIManifest::ARG_B));
    std::memcpy(w->bo.map<void*>(), B, w->bytes);
    w->bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    return w.release();
  }
  catch(const std::exception& e) {
    err = string("XRT failed to upload weights: ") + e.what();
    return nullptr;
  }
}

void RyzenAIKernel::freeWeights(Weights* weights) {
  delete weights;
}

void RyzenAIKernel::rewriteWeights(Weights* weights, const uint16_t* B) {
  std::memcpy(weights->bo.map<void*>(), B, weights->bytes);
  weights->bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
}

RyzenAIKernel::Op* RyzenAIKernel::loadOp(
  const string& xclbinPath, const string& instsPath, int deviceIdx,
  const size_t* inBytes, int numIns, size_t bytesC, string& err) {
  err.clear();

  vector<char> insts;
  if(!readWholeFile(instsPath, insts) || insts.empty()) {
    err = "no instruction stream at " + instsPath;
    return nullptr;
  }

  try {
    unique_ptr<Op> op(new Op());
    op->bytesC = bytesC;
    op->device = xrt::device((unsigned int)(deviceIdx < 0 ? 0 : deviceIdx));
    op->xclbin = xrt::xclbin(xclbinPath);

    const string kernelName = findKernelName(op->xclbin, RyzenAIManifest::KERNEL_NAME_PREFIX);
    if(kernelName.empty()) {
      err = "no kernel named " + string(RyzenAIManifest::KERNEL_NAME_PREFIX) + "* in " + xclbinPath;
      return nullptr;
    }

    op->device.register_xclbin(op->xclbin);
    op->context = xrt::hw_context(op->device, op->xclbin.get_uuid());
    op->kernel = xrt::kernel(op->context, kernelName);

    op->boInstr = xrt::bo(
      op->device, insts.size(), XCL_BO_FLAGS_CACHEABLE,
      op->kernel.group_id(RyzenAIManifest::ARG_INSTR));
    op->inBytes.assign(inBytes, inBytes + numIns);
    for(int j = 0; j < numIns; j++)
      op->boIns.emplace_back(
        op->device, inBytes[j], XRT_BO_FLAGS_HOST_ONLY,
        op->kernel.group_id(RyzenAIManifest::ARG_A + j));
    op->boC = xrt::bo(
      op->device, bytesC, XRT_BO_FLAGS_HOST_ONLY,
      op->kernel.group_id(RyzenAIManifest::ARG_A + numIns));
    std::memcpy(op->boInstr.map<void*>(), insts.data(), insts.size());
    op->boInstr.sync(XCL_BO_SYNC_BO_TO_DEVICE);

    // Every argument is fixed for the op's lifetime, so the run is built once.
    op->run = xrt::run(op->kernel);
    op->run.set_arg(RyzenAIManifest::ARG_OPCODE, RyzenAIManifest::OPCODE_START_WITH_INSTRUCTIONS);
    op->run.set_arg(RyzenAIManifest::ARG_INSTR, op->boInstr);
    op->run.set_arg(RyzenAIManifest::ARG_NINSTR, (uint32_t)(insts.size() / sizeof(uint32_t)));
    for(int j = 0; j < numIns; j++)
      op->run.set_arg(RyzenAIManifest::ARG_A + j, op->boIns[j]);
    op->run.set_arg(RyzenAIManifest::ARG_A + numIns, op->boC);
    return op.release();
  }
  catch(const std::exception& e) {
    err = string("XRT failed to load op ") + xclbinPath + ": " + e.what();
    return nullptr;
  }
}

void RyzenAIKernel::freeOp(Op* op) {
  delete op;
}

void RyzenAIKernel::runOp(Op* op, const void* const* ins, int numIns, void* C) {
  try {
    for(int j = 0; j < numIns && j < (int)op->boIns.size(); j++) {
      std::memcpy(op->boIns[j].map<void*>(), ins[j], op->inBytes[j]);
      op->boIns[j].sync(XCL_BO_SYNC_BO_TO_DEVICE);
    }
    op->run.start();
    ert_cmd_state state = op->run.wait();
    if(state != ERT_CMD_STATE_COMPLETED)
      throw StringError(
        "RyzenAI op dispatch did not complete, ERT state " + Global::intToString((int)state));
    op->boC.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    std::memcpy(C, op->boC.map<void*>(), op->bytesC);
  }
  catch(const StringError&) {
    throw;
  }
  catch(const std::exception& e) {
    throw StringError(string("RyzenAI op dispatch failed: ") + e.what());
  }
}

void RyzenAIKernel::runGemm(
  Engine* engine, Weights* weights, int paddedM, int paddedN, const uint16_t* A, float* C) {
  try {
    const EngineInfo& info = engine->info;
    if(paddedN != weights->paddedN)
      throw StringError(
        "RyzenAI GEMM: N=" + Global::intToString(paddedN) + " does not match the uploaded " +
        "weight width " + Global::intToString(weights->paddedN));

    if(paddedM != engine->lastM || paddedN != engine->lastN) {
      engine->instr = RyzenAISequence::generateSequence(
        engine->seqArch, info.cols, paddedM, info.K, paddedN, info.tileM, info.tileK, info.tileN);
      engine->lastM = paddedM;
      engine->lastN = paddedN;
      const size_t instrBytes = engine->instr.size() * sizeof(uint32_t);
      if(instrBytes > engine->capInstrBytes) {
        engine->boInstr = xrt::bo(
          engine->device, instrBytes, XCL_BO_FLAGS_CACHEABLE,
          engine->kernel.group_id(RyzenAIManifest::ARG_INSTR));
        engine->capInstrBytes = instrBytes;
      }
      std::memcpy(engine->boInstr.map<void*>(), engine->instr.data(), instrBytes);
      engine->boInstr.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    }

    const size_t bytesA = (size_t)paddedM * (size_t)info.K * sizeof(uint16_t);
    const size_t bytesC = (size_t)paddedM * (size_t)paddedN * sizeof(float);
    if(bytesA > engine->capABytes) {
      engine->boA = xrt::bo(
        engine->device, bytesA, XRT_BO_FLAGS_HOST_ONLY,
        engine->kernel.group_id(RyzenAIManifest::ARG_A));
      engine->capABytes = bytesA;
      engine->bufferGeneration++;
    }
    if(bytesC > engine->capCBytes) {
      engine->boC = xrt::bo(
        engine->device, bytesC, XRT_BO_FLAGS_HOST_ONLY,
        engine->kernel.group_id(RyzenAIManifest::ARG_C));
      engine->capCBytes = bytesC;
      engine->bufferGeneration++;
    }

    const auto tUpload = std::chrono::steady_clock::now();
    std::memcpy(engine->boA.map<void*>(), A, bytesA);
    engine->boA.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    const auto tExec = std::chrono::steady_clock::now();

    if(!weights->runValid || weights->runGeneration != engine->bufferGeneration) {
      weights->run = xrt::run(engine->kernel);
      weights->run.set_arg(
        RyzenAIManifest::ARG_OPCODE, RyzenAIManifest::OPCODE_START_WITH_INSTRUCTIONS);
      weights->run.set_arg(RyzenAIManifest::ARG_INSTR, engine->boInstr);
      weights->run.set_arg(RyzenAIManifest::ARG_A, engine->boA);
      weights->run.set_arg(RyzenAIManifest::ARG_B, weights->bo);
      weights->run.set_arg(RyzenAIManifest::ARG_C, engine->boC);
      weights->runValid = true;
      weights->runGeneration = engine->bufferGeneration;
      weights->lastNumInstr = 0;
    }
    // The instruction count is the one argument that tracks the shape.
    if(weights->lastNumInstr != (uint32_t)engine->instr.size()) {
      weights->run.set_arg(RyzenAIManifest::ARG_NINSTR, (uint32_t)engine->instr.size());
      weights->lastNumInstr = (uint32_t)engine->instr.size();
    }

    weights->run.start();
    ert_cmd_state state = weights->run.wait();
    const auto tDownload = std::chrono::steady_clock::now();
    if(state != ERT_CMD_STATE_COMPLETED)
      throw StringError(
        "RyzenAI GEMM dispatch did not complete, ERT state " + Global::intToString((int)state));

    engine->boC.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    std::memcpy(C, engine->boC.map<void*>(), bytesC);

    const auto tEnd = std::chrono::steady_clock::now();
    using Secs = std::chrono::duration<double>;
    engine->timings.secsUploadA += Secs(tExec - tUpload).count();
    engine->timings.secsExecute += Secs(tDownload - tExec).count();
    engine->timings.secsDownloadC += Secs(tEnd - tDownload).count();
    engine->timings.numDispatches++;
  }
  catch(const StringError&) {
    throw;
  }
  catch(const std::exception& e) {
    throw StringError(string("RyzenAI GEMM dispatch failed: ") + e.what());
  }
}

vector<int> RyzenAIKernel::listGemmK(const string& artifactDir) {
  std::set<int> ks;
  std::error_code ec;
  const gfs::path root = gfs::u8path(artifactDir);
  if(!gfs::is_directory(root, ec))
    return vector<int>();
  for(auto& entry : gfs::recursive_directory_iterator(root, ec)) {
    if(!entry.is_regular_file(ec))
      continue;
    const gfs::path p = entry.path();
    if(p.extension().u8string() != ".xclbin")
      continue;
    const string stem = p.stem().u8string();
    // The swiglu-epilogue variants share the K-suffixed naming but are a
    // different operator; they must not advertise a plain-GEMM K.
    if(stem.rfind("gemm_swiglu_", 0) == 0)
      continue;
    int k = 0;
    if(parseKFromStem(stem, k))
      ks.insert(k);
  }
  return vector<int>(ks.begin(), ks.end());
}

vector<int> RyzenAIKernel::listSwigluK(const string& artifactDir) {
  std::set<int> ks;
  std::error_code ec;
  const gfs::path root = gfs::u8path(artifactDir);
  if(!gfs::is_directory(root, ec))
    return vector<int>();
  for(auto& entry : gfs::recursive_directory_iterator(root, ec)) {
    if(!entry.is_regular_file(ec))
      continue;
    const gfs::path p = entry.path();
    if(p.extension().u8string() != ".xclbin")
      continue;
    const string stem = p.stem().u8string();
    if(stem.rfind("gemm_swiglu_", 0) != 0)
      continue;
    int k = 0;
    if(parseKFromStem(stem, k))
      ks.insert(k);
  }
  return vector<int>(ks.begin(), ks.end());
}

string RyzenAIKernel::selfTest(const string& artifactDir, int deviceIdx) {
  string out = "RyzenAI NPU self-test:";
  const vector<int> ks = listGemmK(artifactDir);
  if(ks.empty())
    return out + " no artifacts found under " + artifactDir;

  std::mt19937 rng(12345);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

  for(size_t i = 0; i < ks.size(); i++) {
    const int K = ks[i];
    EngineInfo info;
    string err;
    Engine* engine = loadEngine(artifactDir, deviceIdx, Dtype::Auto, K, 0, info, err);
    if(engine == nullptr) {
      out += "\n  K=" + Global::intToString(K) + ": LOAD FAILED (" + err + ")";
      continue;
    }

    try {
      // A board's worth of rows and a typical layer width, padded as a real
      // dispatch would be.
      const int M = padM(info, 361);
      const int N = padN(info, 512);
      vector<uint16_t> A((size_t)M * info.K);
      vector<uint16_t> B((size_t)info.K * N);
      for(size_t j = 0; j < A.size(); j++) A[j] = floatToBf16(dist(rng));
      for(size_t j = 0; j < B.size(); j++) B[j] = floatToBf16(dist(rng));
      vector<float> C((size_t)M * N, 0.0f);

      Weights* w = uploadWeights(engine, N, B.data(), err);
      if(w == nullptr) {
        out += "\n  K=" + Global::intToString(K) + ": " + err;
        freeEngine(engine);
        continue;
      }

      runGemm(engine, w, M, N, A.data(), C.data());

      // Spot-check a sample of C: a wrong descriptor field produces gross
      // garbage, never a subtle drift, so full recomputation buys nothing.
      double maxErr = 0.0;
      std::mt19937 pick(7);
      for(int t = 0; t < 200; t++) {
        const int r = (int)(pick() % (unsigned)M);
        const int c = (int)(pick() % (unsigned)N);
        double acc = 0.0;
        for(int k = 0; k < info.K; k++)
          acc += (double)bf16ToFloat(A[(size_t)r * info.K + k]) *
                 bf16ToFloat(B[(size_t)k * N + c]);
        maxErr = std::max(maxErr, std::fabs(acc - C[(size_t)r * N + c]));
      }

      const int iters = 20;
      auto t0 = std::chrono::steady_clock::now();
      for(int t = 0; t < iters; t++)
        runGemm(engine, w, M, N, A.data(), C.data());
      auto t1 = std::chrono::steady_clock::now();
      const double ms = std::chrono::duration<double, std::milli>(t1 - t0).count() / iters;

      char line[256];
      std::snprintf(
        line, sizeof(line), "\n  K=%-5d %s %dcol  M=%d N=%d  %.3f ms  %.1f GFLOP/s  maxAbsErr=%.3g %s",
        info.K, dtypeName(info.dtype), info.cols, M, N, ms,
        2.0 * M * info.K * N / (ms * 1e6), maxErr, maxErr < 0.5 ? "OK" : "FAILED");
      out += line;

      freeWeights(w);
    }
    catch(const std::exception& e) {
      out += "\n  K=" + Global::intToString(K) + ": THREW (" + e.what() + ")";
    }
    freeEngine(engine);
  }
  return out;
}
