// KataGo GPU neural net backend implementation shared between the CUDA backend
// (cudabackend.cpp, compiled as host C++ against cuDNN/cuBLAS) and the ROCm backend
// (rocmbackend.cpp, compiled by hip clang against MIOpen/hipBLAS). This file is never compiled
// directly. It is #included by exactly one of those two wrapper files per build. The GPU kernels
// the code below launches and the CudaUtils helper functions are shared the same way, in
// cudaandrocmhelpers.inc and cudaandrocmutils.inc.
//
// The wrapper must, before including this file:
//   - Include the backend's vendor headers and error-check header (cudaerrorcheck.h or
//     rocmerrorcheck.h, which define the same macro names CUDA_ERR, CUBLAS_ERR and CUDNN_ERR),
//     plus the backend's kernel-launcher header (cudahelpers.h / rocmhelpers.h) and utils header
//     (cudautils.h / rocmutils.h).
//   - Define exactly one of KATAGO_GPU_CUDA or KATAGO_GPU_HIP (to 1). These gate the regions
//     where the two backends genuinely differ.
//   - Define KATAGO_GPU_BACKEND_NAME to a short string literal ("CUDA" / "ROCm") used in error
//     messages and debug labels that are otherwise identical across the backends.
//   - Provide `using cublas_half_t = ...;`, the element type cublasHgemm/hipblasHgemm expects
//     buffers to be cast to (half / hipblasHalf).
//   - (ROCm only) Include rocmcudanames.h, which maps the CUDA spellings used in shared code
//     (cudaMalloc, cublasSgemm, cudnnTensorDescriptor_t, ...) to their HIP/hipBLAS/MIOpen
//     equivalents. Shared code is written with CUDA spellings, while code inside KATAGO_GPU_HIP
//     regions uses native HIP/MIOpen spellings.
//   - Optionally define KATAGO_CUDA_HAS_SDPA (CUDA) or KATAGO_ROCM_HAS_CK_FMHA /
//     KATAGO_ROCM_CK_FMHA_ARCH_OK (ROCm) to 1 to enable that backend's fused attention path.
//     All safely evaluate as 0 where left undefined.
//
// Editing rules: code outside the KATAGO_GPU_* gates must be valid for both backends. In
// particular it may only use CUDA API spellings that rocmcudanames.h maps (extend that shim
// when adding a new API use - a HIP build error is the reminder), and may only touch fields of
// cudaDeviceProp/hipDeviceProp_t that exist in both (major, minor, name, totalGlobalMem).
// Struct names (CudaHandles, CudnnManager, ...) and the customCuda* kernel names keep their
// historical CUDA-flavored spellings on both platforms, which is unambiguous because only one
// backend is ever compiled into a build.
//
// A call whose error return is deliberately ignored, such as a cleanup-path free, must be written
// as (void)cudaFoo(...). The HIP entry points these map to are declared [[nodiscard]] while their
// CUDA counterparts are not, so a bare call warns on the ROCm build only. Use the cast only where
// ignoring the error is genuinely intended, and CUDA_ERR everywhere else.

#ifdef KATAGO_BACKEND_INC_INCLUDED
#error "cudaandrocmbackend.inc may only be included once"
#endif
#define KATAGO_BACKEND_INC_INCLUDED 1
#if !defined(KATAGO_GPU_CUDA) && !defined(KATAGO_GPU_HIP)
#error "Define KATAGO_GPU_CUDA or KATAGO_GPU_HIP before including cudaandrocmbackend.inc"
#endif
#if defined(KATAGO_GPU_CUDA) && defined(KATAGO_GPU_HIP)
#error "Define only one of KATAGO_GPU_CUDA and KATAGO_GPU_HIP (the gates below are if/else)"
#endif

#include <atomic>
#include <map>
#include <mutex>
#include <tuple>

#include "../neuralnet/modelversion.h"
#include "../neuralnet/nninterface.h"
#include "../neuralnet/nninputs.h"
#include "../neuralnet/sgfmetadata.h"
#include "../neuralnet/nneval.h"
#include "../neuralnet/desc.h"

#include "../core/simpleallocator.h"
#include "../core/test.h"
#ifdef KATAGO_GPU_CUDA
#include "../core/hash.h"
#endif

#include "../external/half-2.2.0/include/half.hpp"

//------------------------
#include "../core/using.h"
//------------------------

using half_t = half_float::half;

//Define this to print out some of the intermediate values of the neural net
//#define DEBUG_INTERMEDIATE_VALUES

void NeuralNet::globalInitialize() {
  //Empty in this backend
}

void NeuralNet::globalCleanup() {
  (void)cudaDeviceReset();
}

#ifdef KATAGO_GPU_CUDA
//---------------------------------------------------------------------------------
// cudnn SDPA support. Graphs + execution plans cached lazily per SDPAGraphKey (batch size, head
// geometry, seqLen, hasMask, dtype - see SDPAGraphKey below).
// Used only when useFP16=true and cudnn supports SDPA at runtime. Otherwise falls back to
// customCudaFlashAttention (see cudaandrocmhelpers.inc).
//
// Tensor layout: BSHD physical, with strides chosen so that the (B,H,S,D)-dim graph view matches
// the existing CUDA backend's Q/K/V/output buffers from MatMulLayer:
//   element at (n, xy, h, d) lives at offset (h*headDim + d) + (n*seqLen + xy) * (numHeads*headDim).
//
// Masking: when a mask is present, we build a fully-materialized additive attention bias of shape
// [B, 1, S, S] from the [B, S] mask: bias[b,q,k] = (mask[b,k] != 0 ? 0 : -3e4). cudnn does not have
// plans for the [B,1,1,S] broadcast pattern that would let us avoid this materialization, but the
// full bias is correct for arbitrary (non-prefix) masks, which we need to support sub-board games.
// The mask is the same across all attention blocks, so the bias is built once per forward
// (ScratchBuffers::attnBiasBuf, flagged by attnBiasBuiltThisForward) and shared by every
// attention block that runs SDPA.
//
// The bias tensor dtype MUST match the io dtype (fp16). An fp32 bias with fp16 Q/K/V passes
// validate/check_support/build_plans on cudnn 9.x but silently misexecutes (nonfinite outputs -
// the fused kernel evidently reinterprets the buffer). Since cudnn adds the (converted) bias to the
// fp32 scores and computes the softmax in fp32, the fp16-range-limited -3e4 constant still masks
// exactly (exp underflow) for any model whose genuine logit spread is below ~3e4.
// See the customCudaMaskToAttnBiasFull kernel comment in cudaandrocmhelpers.inc for details.
//
// When mask is NULL (full-board, requireExactNNLen case), we build a no-bias graph instead, which
// avoids both the extra memory and the bias build kernel.

#if KATAGO_CUDA_HAS_SDPA
struct SDPAPlanForBatchSize {
  std::shared_ptr<cudnn_frontend::graph::Graph> graph;
  int64_t workspaceBytes;
  bool hasMask;  // true if the graph expects a bias variant-pack entry

  // UIDs for the variant pack, fixed at graph build time.
  static constexpr int64_t Q_UID = 1;
  static constexpr int64_t K_UID = 2;
  static constexpr int64_t V_UID = 3;
  static constexpr int64_t O_UID = 4;
  static constexpr int64_t BIAS_UID = 5;
};

// Full discriminating key for an SDPA execution plan. Every field that changes the cudnn graph shape
// must be here: if any attention layer in a future model differs in head count/dim/seqLen, it gets its
// own plan rather than incorrectly reusing another layer's. (batchSize and hasMask vary at runtime.)
struct SDPAGraphKey {
  int numHeads;
  int numKVHeads;
  int qHeadDim;
  int vHeadDim;
  int seqLen;
  int batchSize;
  bool hasMask;
  bool usingFP16;

  bool operator==(const SDPAGraphKey& o) const {
    return
      numHeads == o.numHeads &&
      numKVHeads == o.numKVHeads &&
      qHeadDim == o.qHeadDim &&
      vHeadDim == o.vHeadDim &&
      seqLen == o.seqLen &&
      batchSize == o.batchSize &&
      hasMask == o.hasMask &&
      usingFP16 == o.usingFP16;
  }
};
struct SDPAGraphKeyHash {
  uint64_t operator()(const SDPAGraphKey& k) const noexcept {
    uint64_t acc = (uint64_t)123456789;
    auto mix = [&acc](uint64_t x) {
      acc += x;
      acc += acc << 13;
      acc ^= acc >> 6;
    };
    mix((uint64_t)k.numHeads);
    mix((uint64_t)k.numKVHeads);
    mix((uint64_t)k.qHeadDim);
    mix((uint64_t)k.vHeadDim);
    mix((uint64_t)k.seqLen);
    mix((uint64_t)k.batchSize);
    mix(k.hasMask ? 1 : 0);
    mix(k.usingFP16 ? 1 : 0);
    acc = Hash::basicLCong(acc);
    return (size_t)(acc ^ (acc >> 32));
  }
};

struct SDPAGraphCache {
  std::unordered_map<SDPAGraphKey, std::shared_ptr<SDPAPlanForBatchSize>, SDPAGraphKeyHash> plansByKey;
  bool sdpaSupported;
  string disableReason;

  SDPAGraphCache() :
    plansByKey(),
    sdpaSupported(true),
    disableReason()
  {}

  // Build (or fetch from cache) an execution plan for the given attention shape + batchSize + hasMask.
  // Returns nullptr if SDPA is not supported for this configuration; caller should use fallback.
  // On a build or check_support failure, SDPA is disabled going forward and nullptr is returned
  // (the caller falls back to the custom kernel): such a failure means cudnn cannot handle this
  // shape/config (e.g. cuDNN 8.9.x rejects head dims that 9.x accepts), which is a capability
  // rejection rather than a correctness failure, and it normally surfaces during warmup, which
  // pre-builds plans for every batch size. The disable is deliberately cache-wide rather than
  // per-key: within one model's handle the keys differ in batch size and mask presence (real
  // models use one head config throughout), known rejection causes depend on neither, and
  // disabling wholesale avoids re-attempting doomed plan builds at each new batch size.
  // logger (if non-NULL) reports a disable.
  std::shared_ptr<SDPAPlanForBatchSize> getOrBuildPlan(cudnnHandle_t cudnn, const SDPAGraphKey& key, Logger* logger) {
    if(!sdpaSupported)
      return nullptr;

    // Cuda graphs for SDPA path only well-tested for FP16/BF16; FP32 uses fallback
    if(!key.usingFP16)
      return nullptr;

    auto it = plansByKey.find(key);
    if(it != plansByKey.end())
      return it->second;

    namespace fe = cudnn_frontend;

    // Disable SDPA and report the reason, falling back to the custom kernel (see the doc above).
    auto disable = [&](const string& reason) -> std::shared_ptr<SDPAPlanForBatchSize> {
      sdpaSupported = false;
      disableReason = reason;
      if(logger != NULL)
        logger->write("Cuda backend: disabling cudnn SDPA and falling back to custom attention kernel: " + reason);
      return nullptr;
    };
    auto plan = std::make_shared<SDPAPlanForBatchSize>();
    plan->hasMask = key.hasMask;
    auto graph = std::make_shared<fe::graph::Graph>();

    bool useFP16 = key.usingFP16;

    fe::DataType_t ioType = useFP16 ? fe::DataType_t::HALF : fe::DataType_t::FLOAT;
    graph->set_io_data_type(ioType)
      .set_intermediate_data_type(fe::DataType_t::FLOAT)
      .set_compute_data_type(fe::DataType_t::FLOAT);

    int64_t B = key.batchSize;
    int64_t Hq = key.numHeads;
    int64_t Hkv = key.numKVHeads;
    int64_t S = key.seqLen;
    int64_t Dq = key.qHeadDim;
    int64_t Dv = key.vHeadDim;

    // BSHD physical layout, with logical dim ordering (B, H, S, D):
    // stride for B = S * H_inner * D
    // stride for H = D
    // stride for S = H_inner * D
    // stride for D = 1
    // where H_inner is the number of heads packed for this tensor (numHeads or numKVHeads).
    int64_t qHinner = key.numHeads;
    int64_t kHinner = key.numKVHeads;
    int64_t vHinner = key.numKVHeads;

    auto Q = graph->tensor(
      fe::graph::Tensor_attributes()
      .set_name("Q")
      .set_uid(SDPAPlanForBatchSize::Q_UID)
      .set_dim({B, Hq, S, Dq})
      .set_stride({S * qHinner * Dq, Dq, qHinner * Dq, 1})
    );
    auto K = graph->tensor(
      fe::graph::Tensor_attributes()
      .set_name("K")
      .set_uid(SDPAPlanForBatchSize::K_UID)
      .set_dim({B, Hkv, S, Dq})
      .set_stride({S * kHinner * Dq, Dq, kHinner * Dq, 1})
    );
    auto V = graph->tensor(
      fe::graph::Tensor_attributes()
      .set_name("V")
      .set_uid(SDPAPlanForBatchSize::V_UID)
      .set_dim({B, Hkv, S, Dv})
      .set_stride({S * vHinner * Dv, Dv, vHinner * Dv, 1})
    );

    float scale = 1.0f / std::sqrt((float)key.qHeadDim);
    auto sdpa_options = (
      fe::graph::SDPA_attributes()
      .set_name("sdpa_fwd")
      .set_generate_stats(false)
      .set_attn_scale(scale)
    );

    if(key.hasMask) {
      // Full [B, 1, S, S] additive bias, broadcast over heads only. Per cudnn 9.8 empirical
      // testing the broadcast-over-q variant ([B,1,1,S]) has no supported plans for our shape.
      auto bias = graph->tensor(
        fe::graph::Tensor_attributes()
        .set_name("bias")
        .set_uid(SDPAPlanForBatchSize::BIAS_UID)
        .set_dim({B, 1, S, S})
        .set_stride({S * S, S * S, S, 1})
      );
      sdpa_options.set_bias(bias);
    }

    auto [O, Stats] = graph->sdpa(Q, K, V, sdpa_options);
    (void)Stats;

    // Output O also uses BSHD physical layout (matches what outProj expects).
    int64_t oHinner = key.numHeads;
    O->set_output(true)
      .set_dim({B, Hq, S, Dv})
      .set_stride({S * oHinner * Dv, Dv, oHinner * Dv, 1})
      .set_uid(SDPAPlanForBatchSize::O_UID);

    auto status = graph->validate();
    if(status.is_bad())
      return disable(string("cudnn SDPA graph validate failed: ") + status.get_message());
    status = graph->build_operation_graph(cudnn);
    if(status.is_bad())
      return disable(string("cudnn SDPA build_operation_graph failed: ") + status.get_message());
    status = graph->create_execution_plans({fe::HeurMode_t::A});
    if(status.is_bad())
      return disable(string("cudnn SDPA create_execution_plans failed: ") + status.get_message());
    status = graph->check_support(cudnn);
    if(status.is_bad())
      return disable(string("cudnn SDPA check_support failed: ") + status.get_message());
    status = graph->build_plans(cudnn);
    if(status.is_bad())
      return disable(string("cudnn SDPA build_plans failed: ") + status.get_message());

    int64_t ws = 0;
    status = graph->get_workspace_size(ws);
    if(status.is_bad())
      return disable(string("cudnn SDPA get_workspace_size failed: ") + status.get_message());

    plan->graph = graph;
    plan->workspaceBytes = ws;
    plansByKey[key] = plan;
    return plan;
  }
};
#else
struct SDPAGraphCache {
  SDPAGraphCache() {}
};
#endif
#else  // KATAGO_GPU_HIP
// Whether the CK (Composable Kernel) FMHA fused attention kernels can run on this device at all.
// This must stay in sync with two things: the arch families CK's generated dispatcher
// (fmha_fwd_api.cpp) matches by device-name prefix, and the archs CMakeLists.txt actually
// compiles the CK kernels for (CMAKE_HIP_ARCHITECTURES minus the pre-MFMA gfx9 and RDNA1 archs
// its CK section excludes). The dispatcher matches "gfx9"
// as a bare prefix, so without this check a gfx906 or gfx900 device would dispatch to kernels
// whose code objects were never compiled for it and fail (throw) at launch instead of falling
// back to the plain attention kernel. Archs the dispatcher has no branch for at all (e.g. RDNA2
// gfx103x) safely return -1 from fmha_fwd, but we exclude them here too so we don't pay for a
// doomed attempt (and its attention-bias materialization) before the runtime-disable kicks in.
static bool isCkFmhaSupportedArch(const hipDeviceProp_t& prop) {
  std::string arch(prop.gcnArchName);
  size_t colonPos = arch.find(':');  // e.g. "gfx942:sramecc+:xnack-"
  if(colonPos != std::string::npos)
    arch = arch.substr(0, colonPos);
  return
    arch == "gfx908" || arch == "gfx90a" ||
    arch.compare(0, 5, "gfx94") == 0 || arch.compare(0, 5, "gfx95") == 0 ||
    arch.compare(0, 5, "gfx11") == 0 || arch.compare(0, 5, "gfx12") == 0;
}

// Archs where MIOpen's NHWC FP16 convolutions (MFMA implicit GEMM) are known faster than NCHW,
// so NHWC mode "Auto" should enable NHWC when using FP16 - the analog of the CUDA backend
// enabling NHWC on tensor-core (Volta+) GPUs. Currently the CDNA/MFMA family only, where NHWC
// measured substantially faster on convnets. RDNA (WMMA) is unmeasured rather than known
// slower. Extend this list if NHWC measures faster there too, testable via rocmUseNHWC=true.
static bool isNhwcFp16PreferredArch(const hipDeviceProp_t& prop) {
  std::string arch(prop.gcnArchName);
  size_t colonPos = arch.find(':');
  if(colonPos != std::string::npos)
    arch = arch.substr(0, colonPos);
  return
    arch == "gfx908" || arch == "gfx90a" ||
    arch.compare(0, 5, "gfx94") == 0 || arch.compare(0, 5, "gfx95") == 0;
}
#endif


//---------------------------------------------------------------------------------
//RAII owners so that when a constructor throws partway through (model load can fail at many
//points - bad weights, failed algo queries, out of memory), whatever was already created is
//destroyed instead of leaked.

//Owner for the compute stream held by CudaHandles. Non-blocking so that streams of different
//NN server threads on the same GPU run concurrently and do not synchronize with the legacy
//default stream.
struct OwnedComputeStream {
  cudaStream_t stream;
  OwnedComputeStream() { CUDA_ERR("CudaHandles",cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking)); }
  ~OwnedComputeStream() { (void)cudaStreamDestroy(stream); }
  operator cudaStream_t() const { return stream; }
  OwnedComputeStream(const OwnedComputeStream&) = delete;
  OwnedComputeStream& operator=(const OwnedComputeStream&) = delete;
};

//Owners for the cublas/hipblas and cudnn/miopen library handles held by CudaHandles.
struct OwnedCublasHandle {
  cublasHandle_t handle;
  OwnedCublasHandle() { CUBLAS_ERR("CudaHandles",cublasCreate(&handle)); }
  ~OwnedCublasHandle() { (void)cublasDestroy(handle); }
  operator cublasHandle_t() const { return handle; }
  OwnedCublasHandle(const OwnedCublasHandle&) = delete;
  OwnedCublasHandle& operator=(const OwnedCublasHandle&) = delete;
};
struct OwnedCudnnHandle {
  cudnnHandle_t handle;
  OwnedCudnnHandle() { CUDNN_ERR("CudaHandles",cudnnCreate(&handle)); }
  ~OwnedCudnnHandle() { (void)cudnnDestroy(handle); }
  operator cudnnHandle_t() const { return handle; }
  OwnedCudnnHandle(const OwnedCudnnHandle&) = delete;
  OwnedCudnnHandle& operator=(const OwnedCudnnHandle&) = delete;
};

//Owner for a vendor descriptor object (ConvLayer's filter/convolution descriptors). Starts
//null, and the owning code creates it by passing &desc to the vendor create call.
template<typename T, cudnnStatus_t (*DestroyFn)(T)>
struct OwnedDesc {
  T desc = nullptr;
  OwnedDesc() {}
  ~OwnedDesc() { if(desc != nullptr) (void)DestroyFn(desc); }
  operator T() const { return desc; }
  OwnedDesc(const OwnedDesc&) = delete;
  OwnedDesc& operator=(const OwnedDesc&) = delete;
};

//Owner for a raw device buffer (filled via CudaUtils::mallocAndCopyToDevice and the like).
struct OwnedDeviceBuf {
  void* buf = nullptr;
  OwnedDeviceBuf() {}
  ~OwnedDeviceBuf() { if(buf != nullptr) (void)cudaFree(buf); }
  OwnedDeviceBuf(const OwnedDeviceBuf&) = delete;
  OwnedDeviceBuf& operator=(const OwnedDeviceBuf&) = delete;
};

//Refcounted owner for a device weight buffer that may be shared between the compute handles of
//multiple NN server threads on the same GPU (cudaShareModelWeights). Freed when the last handle
//using it is destroyed.
struct SharedDeviceBuf {
  std::shared_ptr<void> owner;
  void* buf = nullptr;
};

//Discriminates the multiple weight buffers that can be keyed by one descriptor, and guards
//against distinct descriptors aliasing (a struct and its first member share an address).
enum class SharedWeightsKind : int {
  MATMUL = 0,
  CONV_FILTER = 1,
  CONV_1X1_MATMUL = 2,
  QKV_PACKED = 3,
  FFN_PACKED = 4,
  FFN_FUSED_PACKED = 5,
  QKV_COMBINED = 6,
};

//Upload a weight buffer to the device, taking ownership of the allocation even if the upload
//itself throws partway.
static std::shared_ptr<void> uploadOwnedDeviceWeights(const std::function<void(void*&)>& uploadFn) {
  void* buf = nullptr;
  try {
    uploadFn(buf);
  }
  catch(...) {
    if(buf != nullptr)
      (void)cudaFree(buf);
    throw;
  }
  return std::shared_ptr<void>(buf, [](void* p){ if(p != nullptr) (void)cudaFree(p); });
}

//Get a shared read-only weight buffer, uploading it via uploadFn only if no other live handle
//has already uploaded the same weights to the same device at the same precision. shareKey is
//the descriptor address. Contract: the LoadedModel owning the descriptor must outlive every
//compute handle built from it, or a recycled descriptor address could silently alias a
//different model's weights (NNEvaluator upholds this by joining its server threads, which own
//the handles, before freeing its LoadedModel). The registry is process-global keyed by
//(desc, kind, device, fp16). This is sound because descriptor addresses are unique across
//live LoadedModels, layout-affecting context settings are uniform per ComputeContext (and
//descs are private to one context), and layouts chosen per handle (combined QKV, fused FFN)
//are distinguished by kind. With sharing disabled the upload is unconditional and unshared.
static SharedDeviceBuf mallocAndCopyToDeviceMaybeShared(
  const string& name, const std::function<void(void*&)>& uploadFn,
  const void* shareKey, SharedWeightsKind kind, bool useFP16, bool shareEnabled
) {
  SharedDeviceBuf out;
  if(!shareEnabled || shareKey == nullptr) {
    out.owner = uploadOwnedDeviceWeights(uploadFn);
    out.buf = out.owner.get();
    return out;
  }
  struct Key {
    const void* k; int kind; int device; bool fp16;
    bool operator<(const Key& o) const { return std::tie(k,kind,device,fp16) < std::tie(o.k,o.kind,o.device,o.fp16); }
  };
  static std::mutex registryMutex;
  static std::map<Key, std::weak_ptr<void>> registry;
  int device = 0;
  CUDA_ERR(name.c_str(), cudaGetDevice(&device));
  Key key{shareKey, (int)kind, device, useFP16};
  std::lock_guard<std::mutex> lock(registryMutex);
  auto it = registry.find(key);
  if(it != registry.end()) {
    std::shared_ptr<void> p = it->second.lock();
    if(p != nullptr) {
      out.owner = p;
      out.buf = p.get();
      return out;
    }
    registry.erase(it);
  }
  std::shared_ptr<void> p = uploadOwnedDeviceWeights(uploadFn);
  // Sweep entries whose buffers are gone, so the map stays bounded by live weights in
  // long-running processes that load and unload many models. Uploads are rare (model load),
  // so the linear sweep is cheap.
  for(auto iter = registry.begin(); iter != registry.end();) {
    if(iter->second.expired())
      iter = registry.erase(iter);
    else
      ++iter;
  }
  registry[key] = p;
  out.owner = p;
  out.buf = p.get();
  return out;
}

#ifdef KATAGO_GPU_CUDA
struct CudaHandles {
  // All kernels, library calls, and inference-time transfers for this handle are issued on this
  // stream, so that multiple NN server threads on one GPU can overlap. The constructor body
  // binds the cublas and cudnn handles to it.
  OwnedComputeStream stream;
  OwnedCublasHandle cublas;
  OwnedCudnnHandle cudnn;
  const int majorComputeCapability;
  const int minorComputeCapability;
  std::unique_ptr<SDPAGraphCache> sdpaCache;
  // Logger for this handle's server thread; may be NULL. Used to report cudnn SDPA falling back.
  Logger* logger;
  // Set while warming up (see NNEvaluator::maybeWarmupComputeHandle). When true, a failed cudnn SDPA
  // execution is tolerated (fall back to the custom kernel); when false such a failure is fatal.
  bool isWarmup;
  // If true, the cudnn graph SDPA path is skipped entirely and the custom attention kernel is always used.
  bool cudaDisableGraphSDPA;
  // Set once we have logged that cudaDisableGraphSDPA actually suppressed an otherwise-usable SDPA path,
  // so the message is emitted only a single time per handle rather than on every attention block.
  bool loggedGraphSDPADisabled;
  // Controls whether 1x1 NHWC convs run as a cuBLAS GEMM (vs cuDNN). Auto = matmul iff FP16.
  // True/False force the choice regardless of precision.
  enabled_t use1x1MatmulMode;
  // Whether attention blocks try the tensor-core mma flash kernel first (requires sm_80+).
  bool mmaAttentionEnabled;
  // Fuse each transformer block's masked residual add into the next block's pre-norm kernel.
  // Numerically identical to the unfused path, kept as an escape hatch for debugging and
  // benchmarking.
  bool fuseResidualNorm;
  // Run Q/K/V projections as one combined GEMM writing an interleaved [tokens, q|k|v] buffer
  // consumed via strides, instead of a strided-batched GEMM into separate slices. Only takes
  // effect for attention shapes the mma kernel handles. Escape hatch for benchmarking.
  bool combineQKV;
  // CUTLASS fused FFN (dual GEMM + SwiGLU epilogue). Requires a build with USE_CUTLASS_FUSED_FFN,
  // FP16, and sm_80+. Must be set before Model construction: TransformerFFNBlock commits to a
  // weight layout based on it.
  bool fusedFFNEnabled;
  bool loggedUsingFusedFFN;
  bool loggedFusedFFNRejected;
  bool loggedMmaAttentionRejectedShape;
  // Set once the first successful mma attention launch has been logged.
  bool loggedUsingMmaAttention;
  // Set once the first successful cudnn graph SDPA execution has been logged.
  bool loggedUsingGraphSDPA;
  // Set once the first attention block constructed with the combined QKV layout has been logged.
  bool loggedUsingCombinedQKV;
  // Set once the first engaged residual-add-into-pre-norm deferral has been logged.
  bool loggedUsingFusedResidualNorm;
  // Set once the first 1x1 conv taking the GEMM path has been logged.
  bool loggedUsing1x1Matmul;
  // Whether large read-only weight buffers are shared between the handles of multiple NN
  // server threads on the same GPU. Must be set before Model construction.
  bool shareModelWeights;

  CudaHandles(int major, int minor)
    : majorComputeCapability(major),
      minorComputeCapability(minor),
      sdpaCache(std::make_unique<SDPAGraphCache>()),
      logger(NULL),
      isWarmup(false),
      cudaDisableGraphSDPA(false),
      loggedGraphSDPADisabled(false),
      use1x1MatmulMode(enabled_t::Auto),
      mmaAttentionEnabled(false),
      fuseResidualNorm(true),
      combineQKV(true),
      fusedFFNEnabled(false),
      loggedUsingFusedFFN(false),
      loggedFusedFFNRejected(false),
      loggedMmaAttentionRejectedShape(false),
      loggedUsingMmaAttention(false),
      loggedUsingGraphSDPA(false),
      loggedUsingCombinedQKV(false),
      loggedUsingFusedResidualNorm(false),
      loggedUsing1x1Matmul(false),
      shareModelWeights(false)
  {
    CUBLAS_ERR("CudaHandles",cublasSetStream(cublas.handle,stream.stream));
    CUDNN_ERR("CudaHandles",cudnnSetStream(cudnn.handle,stream.stream));
  }

  static CudaHandles* cudaHandlesTesting() {
    const int gpuIdxForThisThread = 0;
    cudaDeviceProp prop;
    CUDA_ERR("cudaHandlesTesting",cudaGetDeviceProperties(&prop,gpuIdxForThisThread));
    return new CudaHandles(prop.major, prop.minor);
  }

  CudaHandles(const CudaHandles&) = delete;
  CudaHandles& operator=(const CudaHandles&) = delete;
};
#else  // KATAGO_GPU_HIP
struct CudaHandles {
  // See the CUDA variant: per-handle compute stream that all work for this handle runs on.
  OwnedComputeStream stream;
  OwnedCublasHandle cublas;
  OwnedCudnnHandle cudnn;
  const int majorComputeCapability;
  const int minorComputeCapability;
  // May be NULL. Not owned. Used for one-time notices (e.g. disabling the fused attention path).
  Logger* logger;
  // Whether this device's architecture can run the CK FMHA fused attention kernels. See
  // isCkFmhaSupportedArch above.
  const bool ckFmhaArchSupported;
  // Disables the optional CK FMHA fused attention path (see KATAGO_ROCM_HAS_CK_FMHA), falling back
  // to the plain attention kernel unconditionally. Off by default.
  const bool disableFusedAttention;
  // Set once if a CK FMHA call fails or reports unsupported at runtime, so subsequent calls skip
  // the attempt (and its attention-bias materialization) entirely. Only touched by the single
  // server thread that owns this handle, hence no synchronization.
  bool fusedAttentionRuntimeDisabled;
  // See the CUDA variant. Must be set before Model construction.
  bool shareModelWeights;
  // See the CUDA variant.
  bool fuseResidualNorm;
  // See the CUDA variant.
  bool loggedUsingFusedResidualNorm;
  // Set once the first successful CK fused attention call has been logged.
  bool loggedUsingCkFmha;
  // Set once the report that disableFusedAttention alone is suppressing the CK path has been logged.
  bool loggedCkFmhaDisabled;
  // See the CUDA variant.
  bool loggedUsing1x1Matmul;
  // Whether 1x1 NHWC convs run as a hipBLAS GEMM instead of a MIOpen conv, already resolved
  // against the GPU architecture, so Auto here means on. Set before Model construction.
  enabled_t use1x1MatmulMode;

  CudaHandles(int major, int minor, Logger* logger_, bool ckFmhaArchSupported_, bool disableFusedAttention_)
    : majorComputeCapability(major),
      minorComputeCapability(minor),
      logger(logger_),
      ckFmhaArchSupported(ckFmhaArchSupported_),
      disableFusedAttention(disableFusedAttention_),
      fusedAttentionRuntimeDisabled(false),
      shareModelWeights(false),
      fuseResidualNorm(true),
      loggedUsingFusedResidualNorm(false),
      loggedUsingCkFmha(false),
      loggedCkFmhaDisabled(false),
      loggedUsing1x1Matmul(false),
      use1x1MatmulMode(enabled_t::Auto)
  {
    CUBLAS_ERR("CudaHandles",cublasSetStream(cublas.handle,stream.stream));
    CUDNN_ERR("CudaHandles",cudnnSetStream(cudnn.handle,stream.stream));
  }

  static CudaHandles* cudaHandlesTesting() {
    const int gpuIdxForThisThread = 0;
    hipDeviceProp_t prop;
    CUDA_ERR("cudaHandlesTesting",hipGetDeviceProperties(&prop,gpuIdxForThisThread));
    return new CudaHandles(prop.major, prop.minor, NULL, isCkFmhaSupportedArch(prop), false);
  }

  CudaHandles(const CudaHandles&) = delete;
  CudaHandles& operator=(const CudaHandles&) = delete;
};
#endif

//---------------------------------------------------------------------------------

//Destructor helpers for ByBatchSize. Descriptor-style entries are opaque pointers and may be
//null if construction threw partway, while non-pointer entries (algo/solution structs) never
//have a destroyFunc, but both overloads must compile for every instantiation.
template<typename U> static bool byBatchSizeEntryNonNull(U* p) { return p != nullptr; }
template<typename U> static bool byBatchSizeEntryNonNull(const U&) { return true; }

template<typename T>
struct ByBatchSize {
  const int maxBatchSize;
  T* data;
  cudnnStatus_t (*destroyFunc)(T);

  ByBatchSize()
    : maxBatchSize(0), data(nullptr), destroyFunc(nullptr)
  {}

  ByBatchSize(
    int maxBatchSize_
  ) : maxBatchSize(maxBatchSize_), data(nullptr), destroyFunc(nullptr) {
    data = new T[maxBatchSize]();  //value-init: descriptor entries start null
  }

  ByBatchSize(const ByBatchSize&) = delete;
  ByBatchSize& operator=(const ByBatchSize&) = delete;

  ~ByBatchSize() {
    if(destroyFunc != nullptr && data != nullptr) {
      for(int batchSize = 1; batchSize <= maxBatchSize; batchSize++) {
        if(byBatchSizeEntryNonNull(data[batchSize-1]))
          (*destroyFunc)(data[batchSize-1]);
      }
    }
    if(data != nullptr) {
      delete[] data;
      data = nullptr;
    }
  }
  T& operator[](int batchSize) {
    return data[batchSize-1];
  }
  const T& operator[](int batchSize) const {
    return data[batchSize-1];
  }
};

template<typename T>
struct ByBatchSizeView {
  int maxBatchSize;
  T* data;

  ByBatchSizeView()
    : maxBatchSize(0), data(nullptr)
  {}

  ByBatchSizeView(const ByBatchSize<T>& toView)
    : maxBatchSize(toView.maxBatchSize), data(toView.data)
  {}
  ByBatchSizeView& operator=(const ByBatchSize<T>& toView) {
    maxBatchSize = toView.maxBatchSize;
    data = toView.data;
    return *this;
  }

  ~ByBatchSizeView() {
  }
  T& operator[](int batchSize) {
    return data[batchSize-1];
  }
  const T& operator[](int batchSize) const {
    return data[batchSize-1];
  }
};

//---------------------------------------------------------------------------------


//channels, useFP16, useNHWC
typedef std::tuple<int, bool, bool> CudnnTensorDesc4DKey;

struct CudnnManager {
  const string name;
  const int maxBatchSize;
  const int nnXLen;
  const int nnYLen;
  std::map<CudnnTensorDesc4DKey, ByBatchSize<cudnnTensorDescriptor_t>*> tensorDesc4DByBatchSizeByKey;

  CudnnManager(string name_, int maxBatchSize_, int nnXLen_, int nnYLen_)
    :name(name_),
     maxBatchSize(maxBatchSize_),
     nnXLen(nnXLen_),
     nnYLen(nnYLen_),
     tensorDesc4DByBatchSizeByKey()
  {
  }

  ~CudnnManager() {
    for(auto& iter: tensorDesc4DByBatchSizeByKey) {
      delete iter.second;
    }
  }

  ByBatchSizeView<cudnnTensorDescriptor_t> getTensorDesc4DByBatchSize(
    int channels, bool useFP16, bool useNHWC
  ) {
    auto iter = tensorDesc4DByBatchSizeByKey.find({channels, useFP16, useNHWC});
    if(iter != tensorDesc4DByBatchSizeByKey.end()) {
      return ByBatchSizeView<cudnnTensorDescriptor_t>(*(iter->second));
    }
    //Hold in a unique_ptr with destroyFunc set before creating anything, so that if descriptor
    //creation throws partway through, the already-created descriptors are destroyed with it.
    auto descs = std::make_unique<ByBatchSize<cudnnTensorDescriptor_t>>(maxBatchSize);
    descs->destroyFunc = cudnnDestroyTensorDescriptor;
    for(int batchSize = 1; batchSize <= maxBatchSize; batchSize++) {
      cudnnTensorDescriptor_t& desc = (*descs)[batchSize];
      CUDNN_ERR(name.c_str(),cudnnCreateTensorDescriptor(&desc));
#ifdef KATAGO_GPU_CUDA
      CUDNN_ERR(name.c_str(),cudnnSetTensor4dDescriptor(
                  desc,
                  (useNHWC ? CUDNN_TENSOR_NHWC : CUDNN_TENSOR_NCHW),
                  (useFP16 ? CUDNN_DATA_HALF : CUDNN_DATA_FLOAT),
                  batchSize,
                  channels,
                  nnYLen,
                  nnXLen
                ));
#else  // KATAGO_GPU_HIP
      if(useNHWC) {
        int lens[4] = {batchSize, channels, nnYLen, nnXLen};
        CUDNN_ERR(name.c_str(),miopenSetNdTensorDescriptorWithLayout(
                    desc,
                    (useFP16 ? miopenHalf : miopenFloat),
                    miopenTensorNHWC,
                    lens,
                    4
                  ));
      }
      else {
        CUDNN_ERR(name.c_str(),miopenSet4dTensorDescriptor(
                    desc,
                    (useFP16 ? miopenHalf : miopenFloat),
                    batchSize,
                    channels,
                    nnYLen,
                    nnXLen
                  ));
      }
#endif
    }
    tensorDesc4DByBatchSizeByKey[{channels, useFP16, useNHWC}] = descs.get();
    return ByBatchSizeView<cudnnTensorDescriptor_t>(*descs.release());
  }
};

//---------------------------------------------------------------------------------

struct ScratchBuffers {

  const size_t batchXYFloatBytes;
  const size_t batchFloatBytes;
  const size_t batchXYBytes;
  const size_t batchBytes;

  SimpleAllocator<void*>* allocator;

  // Not scratch, but convenient to have here
  void* zeroBuf;
  void* oneBuf;

  // Per-forward cache of the [batchSize, seqLen, seqLen] additive attention bias derived from the
  // mask, shared by all attention blocks of one forward pass. Model::apply clears the flag at the
  // start of each forward and the first attention block needing the bias builds it. The buffer
  // itself is allocated lazily once (its size depends only on max batch size) and then reused.
  std::unique_ptr<SizedBuf<void*>> attnBiasBuf;
  bool attnBiasBuiltThisForward = false;

  ScratchBuffers() = delete;
  ScratchBuffers(const ScratchBuffers&) = delete;
  ScratchBuffers& operator=(const ScratchBuffers&) = delete;

  ScratchBuffers(int maxBatchSize, int nnXLen, int nnYLen, bool useFP16)
    : batchXYFloatBytes((size_t)maxBatchSize * nnXLen * nnYLen * sizeof(float)),
      batchFloatBytes((size_t)maxBatchSize * sizeof(float)),
      batchXYBytes((size_t)maxBatchSize * nnXLen * nnYLen * (useFP16 ? sizeof(half_t) : sizeof(float))),
      batchBytes((size_t)maxBatchSize * (useFP16 ? sizeof(half_t) : sizeof(float)))
  {
    std::function<void*(size_t)> allocateFunc = [](size_t size) {
      void* buf;
      CUDA_ERR("ScratchBuffers",cudaMalloc(&buf, size));
      return buf;
    };
    std::function<void(void*)> releaseFunc = [](void* buf) {
      (void)cudaFree(buf);
    };

    //Build into a unique_ptr first (same pattern as ByBatchSize) so that if
    //hostMallocZeroOneBufs throws, the allocator is not leaked.
    auto allocatorOwned = std::make_unique<SimpleAllocator<void*>>(allocateFunc, releaseFunc);

    CudaUtils::hostMallocZeroOneBufs(zeroBuf, oneBuf, useFP16);
    allocator = allocatorOwned.release();
  }
  ~ScratchBuffers() {
    //Before the allocator: ~SizedBuf hands its buffer back to the allocator, and members are
    //destroyed after this body runs.
    attnBiasBuf.reset();
    delete allocator;
    free(zeroBuf);
    free(oneBuf);
  }

  size_t getBufSizeXY(int channels) const {
    return channels * batchXYBytes;
  }
  size_t getBufSizeXYFloat(int channels) const {
    return channels * batchXYFloatBytes;
  }
  size_t getBufSizeFloat(int channels) const {
    return channels * batchFloatBytes;
  }
  size_t getBufSize(int channels) const {
    return channels * batchBytes;
  }

};


//---------------------------------------------------------------------------------

#ifdef KATAGO_GPU_CUDA
struct ConvLayer {
  const string name;
  const int inChannels;
  const int outChannels;
  ByBatchSizeView<cudnnTensorDescriptor_t> inputDescriptors;
  ByBatchSizeView<cudnnTensorDescriptor_t> outputDescriptors;
  OwnedDesc<cudnnFilterDescriptor_t, cudnnDestroyFilterDescriptor> filterDescriptor;
  OwnedDesc<cudnnConvolutionDescriptor_t, cudnnDestroyConvolutionDescriptor> convolutionDescriptor;
#if CUDNN_MAJOR >= 8
  std::unique_ptr<ByBatchSize<cudnnConvolutionFwdAlgoPerf_t>> convolutionAlgorithms; //array of one for each batch size
#else
  std::unique_ptr<ByBatchSize<cudnnConvolutionFwdAlgo_t>> convolutionAlgorithms; //array of one for each batch size
#endif
  SharedDeviceBuf filterBuf;
  // A 1x1 conv is equivalent to a matmul. When use1x1Matmul is set we run it as a cuBLAS GEMM over
  // batch*spatial tokens and build NO cuDNN objects. This is the default for 1x1 NHWC FP16 convs.
  // matmulWeightBuf is [inC, outC] row-major, which cuBLAS reads as column-major [outC, inC].
  bool use1x1Matmul;
  int matmulSpatialSize;
  SharedDeviceBuf matmulWeightBuf;
  bool usingFP16;

  ConvLayer() = delete;
  ConvLayer(const ConvLayer&) = delete;
  ConvLayer& operator=(const ConvLayer&) = delete;

  ConvLayer(
    CudaHandles* cudaHandles,
    CudnnManager* manager,
    const ConvLayerDesc* desc,
    bool useFP16,
    bool useNHWC
  ) : ConvLayer(cudaHandles, manager, desc, useFP16, useNHWC, useNHWC)
  {}

  ConvLayer(
    CudaHandles* cudaHandles,
    CudnnManager* manager,
    const ConvLayerDesc* desc,
    bool useFP16,
    bool useNHWCIn,
    bool useNHWCOut
  ) :
    name(desc->name),
    inChannels(desc->inChannels),
    outChannels(desc->outChannels)
  {
    int convYSize = desc->convYSize;
    int convXSize = desc->convXSize;
    int dilationY = desc->dilationY;
    int dilationX = desc->dilationX;
    int paddingX = (convXSize / 2) * dilationX;
    int paddingY = (convYSize / 2) * dilationY;

    testAssert(convXSize % 2 == 1);
    testAssert(convYSize % 2 == 1);

    usingFP16 = useFP16;

    // A 1x1 conv is a matmul, and cuBLAS is faster than cuDNN's conv in FP16 (tensor cores).
    // Benchmarked as slightly faster on convnets and neutral on transformers.
    // In FP32 there wasn't an improvement, so the default (cudaUse1x1Matmul=Auto) only uses GEMM in FP16.
    // The config flag can force it either way regardless of precision. Supports NHWC only (the GEMM assumes
    // channel-contiguous-per-position layout).
    use1x1Matmul = false;
    if(convXSize == 1 && convYSize == 1 && useNHWCIn && useNHWCOut) {
      enabled_t mode = cudaHandles->use1x1MatmulMode;
      use1x1Matmul = (mode == enabled_t::True) || (mode == enabled_t::Auto && useFP16);
    }
    matmulSpatialSize = use1x1Matmul ? (manager->nnYLen * manager->nnXLen) : 0;

    if(use1x1Matmul) {
      // 1x1 conv weights are [outC, inC]. cuBLAS GEMM wants column-major, i.e. [inC, outC] in row-major notation.
      // So transpose. No cuDNN objects are built.
      vector<float> wT((size_t)inChannels * outChannels);
      for(int oc = 0; oc < outChannels; oc++)
        for(int ic = 0; ic < inChannels; ic++)
          wT[(size_t)oc + (size_t)ic * outChannels] = desc->weights[(size_t)oc * inChannels + ic];
      matmulWeightBuf = mallocAndCopyToDeviceMaybeShared(
        name + ":matmulW",
        [&](void*& b){ CudaUtils::mallocAndCopyToDevice(name + ":matmulW", wT, b, useFP16); },
        desc, SharedWeightsKind::CONV_1X1_MATMUL, useFP16, cudaHandles->shareModelWeights);
      if(!cudaHandles->loggedUsing1x1Matmul) {
        cudaHandles->loggedUsing1x1Matmul = true;
        if(cudaHandles->logger != NULL)
          cudaHandles->logger->write(KATAGO_GPU_BACKEND_NAME " backend: running 1x1 NHWC convolutions as a GEMM");
      }
      return;
    }

    inputDescriptors = manager->getTensorDesc4DByBatchSize(inChannels,useFP16,useNHWCIn);
    outputDescriptors = manager->getTensorDesc4DByBatchSize(outChannels,useFP16,useNHWCOut);
    int maxBatchSize = manager->maxBatchSize;

    bool filterNHWC = useNHWCOut && dilationY == 1 && dilationX == 1;

    CUDNN_ERR(name.c_str(),cudnnCreateFilterDescriptor(&filterDescriptor.desc));
    CUDNN_ERR(name.c_str(),cudnnSetFilter4dDescriptor(
      filterDescriptor,
      (useFP16 ? CUDNN_DATA_HALF : CUDNN_DATA_FLOAT),
      (filterNHWC ? CUDNN_TENSOR_NHWC : CUDNN_TENSOR_NCHW),
      outChannels,
      inChannels,
      convYSize,
      convXSize
    ));

    int yStride = 1;
    int xStride = 1;

    //NVIDIA compute capability 7 is when we first hit Volta architecture, with tensor cores
    //See https://en.wikipedia.org/wiki/CUDA#Version_features_and_specifications
    bool tensorCoresSupported = cudaHandles->majorComputeCapability >= 7;

    CUDNN_ERR(name.c_str(),cudnnCreateConvolutionDescriptor(&convolutionDescriptor.desc));
    CUDNN_ERR(name.c_str(),cudnnSetConvolution2dDescriptor(
      convolutionDescriptor,
      paddingY,
      paddingX,
      yStride,
      xStride,
      dilationY,
      dilationX,
      CUDNN_CROSS_CORRELATION,
      (useFP16 && !tensorCoresSupported) ? CUDNN_DATA_HALF : CUDNN_DATA_FLOAT
    ));
    if(useFP16 && tensorCoresSupported)
      CUDNN_ERR(name.c_str(),cudnnSetConvolutionMathType(convolutionDescriptor, CUDNN_TENSOR_OP_MATH));

#if CUDNN_MAJOR >= 8
    convolutionAlgorithms = std::make_unique<ByBatchSize<cudnnConvolutionFwdAlgoPerf_t>>(maxBatchSize);
#else
    convolutionAlgorithms = std::make_unique<ByBatchSize<cudnnConvolutionFwdAlgo_t>>(maxBatchSize);
#endif

    for(int batchSize = 1; batchSize <= maxBatchSize; batchSize++) {
      if(useFP16 && dilationX <= 1 && dilationY <= 1) {
#if CUDNN_MAJOR >= 8
        (*convolutionAlgorithms)[batchSize].algo = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM;
#else
        (*convolutionAlgorithms)[batchSize] = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM;
#endif
      }
      else {
        const cudnnTensorDescriptor_t& inputDescriptor = inputDescriptors[batchSize];
        const cudnnTensorDescriptor_t& outputDescriptor = outputDescriptors[batchSize];

#if CUDNN_MAJOR >= 8
        int requestedAlgoCount = CUDNN_CONVOLUTION_FWD_ALGO_COUNT;
        int returnedAlgoCount = -1;
        cudnnConvolutionFwdAlgoPerf_t results[2 * CUDNN_CONVOLUTION_FWD_ALGO_COUNT];
        CUDNN_ERR(name.c_str(),cudnnGetConvolutionForwardAlgorithm_v7(
          cudaHandles->cudnn,
          inputDescriptor,
          filterDescriptor,
          convolutionDescriptor,
          outputDescriptor,
          requestedAlgoCount,
          &returnedAlgoCount,
          results
        ));
        if(returnedAlgoCount <= 0)
          throw StringError("cudnnGetConvolutionForwardAlgorithm_v7 returned no algorithms?");
        (*convolutionAlgorithms)[batchSize] = results[0];
#else
        size_t bytesMemoryLimit = 0;
        CUDNN_ERR(name.c_str(),cudnnGetConvolutionForwardAlgorithm(
           cudaHandles->cudnn,
           inputDescriptor,
           filterDescriptor,
           convolutionDescriptor,
           outputDescriptor,
           CUDNN_CONVOLUTION_FWD_PREFER_FASTEST,
           bytesMemoryLimit,
           &((*convolutionAlgorithms)[batchSize])
         ));
#endif
      }
    }

    testAssert(desc->weights.size() == convYSize * convXSize * inChannels * outChannels);

    if(filterNHWC) {
      vector<float> weightsTransposed(desc->weights.size());
      for(int y = 0; y < convYSize; y++) {
        for(int x = 0; x < convXSize; x++) {
          for(int ic = 0; ic < inChannels; ic++) {
            for(int oc = 0; oc < outChannels; oc++) {
              weightsTransposed[((oc*convYSize + y)*convXSize + x)*inChannels + ic] =
                desc->weights[((oc*inChannels + ic)*convYSize + y)*convXSize + x];
            }
          }
        }
      }
      filterBuf = mallocAndCopyToDeviceMaybeShared(
        name,
        [&](void*& b){ CudaUtils::mallocAndCopyToDevice(name,weightsTransposed,b,useFP16); },
        desc, SharedWeightsKind::CONV_FILTER, useFP16, cudaHandles->shareModelWeights);
      (void)cudaDeviceSynchronize();
    }
    else
      filterBuf = mallocAndCopyToDeviceMaybeShared(
        name,
        [&](void*& b){ CudaUtils::mallocAndCopyToDevice(name,desc->weights,b,useFP16); },
        desc, SharedWeightsKind::CONV_FILTER, useFP16, cudaHandles->shareModelWeights);
  }

  size_t requiredWorkspaceBytes(
    CudaHandles* cudaHandles,
    int batchSize
  ) const {
    if(use1x1Matmul)
      return 0;
    size_t workspaceBytes = 0;
#if CUDNN_MAJOR >= 8
    CUDNN_ERR(name.c_str(),cudnnGetConvolutionForwardWorkspaceSize(
      cudaHandles->cudnn,
      inputDescriptors[batchSize],
      filterDescriptor,
      convolutionDescriptor,
      outputDescriptors[batchSize],
      (*convolutionAlgorithms)[batchSize].algo,
      &workspaceBytes
    ));
#else
    CUDNN_ERR(name.c_str(),cudnnGetConvolutionForwardWorkspaceSize(
      cudaHandles->cudnn,
      inputDescriptors[batchSize],
      filterDescriptor,
      convolutionDescriptor,
      outputDescriptors[batchSize],
      (*convolutionAlgorithms)[batchSize],
      &workspaceBytes
    ));
#endif
    return workspaceBytes;
  }

  void apply(
    CudaHandles* cudaHandles,
    ScratchBuffers* scratch,
    int batchSize,
    bool accumulate,
    void* inputBuf,
    void* outputBuf,
    void* workspaceBuf,
    size_t workspaceBytes
  ) const {
    //Unused here, but the ROCm backend's version of this function needs it to emulate accumulate.
    (void)scratch;
    if(use1x1Matmul) {
      // out[outC, tokens] = W[outC, inC] x in[inC, tokens]
      // where tokens = batchSize * spatial. NHWC buffers are [tokens, C] row-major = [C, tokens] column-major
      // matching cuBLAS's expectation. Same as MatMulLayer.
      int tokens = batchSize * matmulSpatialSize;
      if(!usingFP16) {
        const float alpha = 1.0f;
        const float beta = accumulate ? 1.0f : 0.0f;
        CUBLAS_ERR(name.c_str(),cublasSgemm(
          cudaHandles->cublas, CUBLAS_OP_N, CUBLAS_OP_N,
          outChannels, tokens, inChannels,
          &alpha, (const float*)matmulWeightBuf.buf, outChannels,
          (const float*)inputBuf, inChannels,
          &beta, (float*)outputBuf, outChannels));
      }
      else {
        const half alpha = __float2half(1.0f);
        const half beta = __float2half(accumulate ? 1.0f : 0.0f);
        CUBLAS_ERR(name.c_str(),cublasHgemm(
          cudaHandles->cublas, CUBLAS_OP_N, CUBLAS_OP_N,
          outChannels, tokens, inChannels,
          &alpha, (const half*)matmulWeightBuf.buf, outChannels,
          (const half*)inputBuf, inChannels,
          &beta, (half*)outputBuf, outChannels));
      }
      return;
    }
    const float alpha = 1.0f;
    const float beta = accumulate ? 1.0f : 0.0f;
#if CUDNN_MAJOR >= 8
    CUDNN_ERR(name.c_str(),cudnnConvolutionForward(
      cudaHandles->cudnn,
      &alpha,
      inputDescriptors[batchSize],
      inputBuf,
      filterDescriptor,
      filterBuf.buf,
      convolutionDescriptor,
      (*convolutionAlgorithms)[batchSize].algo,
      workspaceBuf,
      workspaceBytes,
      &beta,
      outputDescriptors[batchSize],
      outputBuf
    ));
#else
    CUDNN_ERR(name.c_str(),cudnnConvolutionForward(
      cudaHandles->cudnn,
      &alpha,
      inputDescriptors[batchSize],
      inputBuf,
      filterDescriptor,
      filterBuf.buf,
      convolutionDescriptor,
      (*convolutionAlgorithms)[batchSize],
      workspaceBuf,
      workspaceBytes,
      &beta,
      outputDescriptors[batchSize],
      outputBuf
    ));
#endif
  }

};
#else  // KATAGO_GPU_HIP
struct ConvLayer {
  const string name;
  const int inChannels;
  const int outChannels;
  const int nnXLen;
  const int nnYLen;
  const int maxBatchSize;
  const bool usingFP16;
  ByBatchSizeView<miopenTensorDescriptor_t> inputDescriptors;
  ByBatchSizeView<miopenTensorDescriptor_t> outputDescriptors;
  OwnedDesc<miopenTensorDescriptor_t, miopenDestroyTensorDescriptor> filterDescriptor;
  OwnedDesc<miopenConvolutionDescriptor_t, miopenDestroyConvolutionDescriptor> convolutionDescriptor;
  std::unique_ptr<ByBatchSize<miopenConvSolution_t>> convolutionAlgorithms; //array of one for each batch size
  SharedDeviceBuf filterBuf;
  // A 1x1 conv is a matmul. When use1x1Matmul is set it runs as a hipBLAS GEMM over
  // batch*spatial tokens and no MIOpen objects are built. matmulWeightBuf is [inC, outC]
  // row-major, which hipBLAS reads as column-major [outC, inC].
  bool use1x1Matmul;
  int matmulSpatialSize;
  SharedDeviceBuf matmulWeightBuf;
  // Set when MIOpen's only NHWC solution for this conv is its naive fallback. The conv is then
  // re-expressed as NCHW at runtime, transposing the input and output around it.
  bool useNchwFallback;
  ByBatchSizeView<miopenTensorDescriptor_t> nchwInputDescriptors;
  ByBatchSizeView<miopenTensorDescriptor_t> nchwOutputDescriptors;
  OwnedDesc<miopenTensorDescriptor_t, miopenDestroyTensorDescriptor> nchwFilterDescriptor;
  OwnedDesc<miopenConvolutionDescriptor_t, miopenDestroyConvolutionDescriptor> nchwConvolutionDescriptor;
  std::unique_ptr<ByBatchSize<miopenConvSolution_t>> nchwConvolutionAlgorithms;
  // Unshared: the registry keys on the descriptor address, which filterBuf already uses, and
  // both filters exist at once.
  SharedDeviceBuf nchwFilterBuf;

  ConvLayer() = delete;
  ConvLayer(const ConvLayer&) = delete;
  ConvLayer& operator=(const ConvLayer&) = delete;

  ConvLayer(
    CudaHandles* cudaHandles,
    CudnnManager* manager,
    const ConvLayerDesc* desc,
    bool useFP16,
    bool useNHWC
  ) : ConvLayer(cudaHandles, manager, desc, useFP16, useNHWC, useNHWC)
  {}

  ConvLayer(
    CudaHandles* cudaHandles,
    CudnnManager* manager,
    const ConvLayerDesc* desc,
    bool useFP16,
    bool useNHWCIn,
    bool useNHWCOut
  ) :
    name(desc->name),
    inChannels(desc->inChannels),
    outChannels(desc->outChannels),
    nnXLen(manager->nnXLen),
    nnYLen(manager->nnYLen),
    maxBatchSize(manager->maxBatchSize),
    usingFP16(useFP16)
  {
    int convYSize = desc->convYSize;
    int convXSize = desc->convXSize;
    int dilationY = desc->dilationY;
    int dilationX = desc->dilationX;

    testAssert(convXSize % 2 == 1);
    testAssert(convYSize % 2 == 1);

    use1x1Matmul = false;
    useNchwFallback = false;
    // NHWC only, since the GEMM assumes channels are contiguous per position.
    if(convXSize == 1 && convYSize == 1 && useNHWCIn && useNHWCOut) {
      enabled_t mode = cudaHandles->use1x1MatmulMode;
      use1x1Matmul = (mode == enabled_t::True) || (mode == enabled_t::Auto);
    }
    matmulSpatialSize = use1x1Matmul ? (manager->nnYLen * manager->nnXLen) : 0;

    if(use1x1Matmul) {
      // Desc weights are [outC, inC], but hipBLAS wants column-major, which is [inC, outC] here.
      vector<float> wT((size_t)inChannels * outChannels);
      for(int oc = 0; oc < outChannels; oc++)
        for(int ic = 0; ic < inChannels; ic++)
          wT[(size_t)oc + (size_t)ic * outChannels] = desc->weights[(size_t)oc * inChannels + ic];
      // Keying on desc is safe despite filterBuf using the same key, since this path builds
      // no filter.
      matmulWeightBuf = mallocAndCopyToDeviceMaybeShared(
        name + ":matmulW",
        [&](void*& b){ CudaUtils::mallocAndCopyToDevice(name + ":matmulW",wT,b,useFP16); },
        desc, SharedWeightsKind::CONV_1X1_MATMUL, useFP16, cudaHandles->shareModelWeights);
      if(!cudaHandles->loggedUsing1x1Matmul) {
        cudaHandles->loggedUsing1x1Matmul = true;
        if(cudaHandles->logger != NULL)
          cudaHandles->logger->write(KATAGO_GPU_BACKEND_NAME " backend: running 1x1 NHWC convolutions as a GEMM");
      }
      return;
    }

    inputDescriptors = manager->getTensorDesc4DByBatchSize(inChannels,useFP16,useNHWCIn);
    outputDescriptors = manager->getTensorDesc4DByBatchSize(outChannels,useFP16,useNHWCOut);

    bool filterNHWC = useNHWCOut && dilationY == 1 && dilationX == 1;

    bool anyNaive = buildMiopenConvPath(
      cudaHandles, name, desc, useFP16, filterNHWC, maxBatchSize, desc,
      inputDescriptors, outputDescriptors, filterDescriptor, convolutionDescriptor,
      convolutionAlgorithms, filterBuf);

    // On some architectures MIOpen's NHWC coverage drops all the way to its naive fallback for a
    // layer. Its NCHW coverage is much better, so re-express the conv there instead and pay a
    // transpose on either side.
    if(anyNaive && useNHWCIn && useNHWCOut) {
      string convDescStr = Global::intToString(inChannels) + "->" + Global::intToString(outChannels) +
        " " + Global::intToString(convYSize) + "x" + Global::intToString(convXSize);
      nchwInputDescriptors = manager->getTensorDesc4DByBatchSize(inChannels,useFP16,false);
      nchwOutputDescriptors = manager->getTensorDesc4DByBatchSize(outChannels,useFP16,false);
      bool nchwNaive = buildMiopenConvPath(
        cudaHandles, name + ":nchw", desc, useFP16, false, maxBatchSize, NULL,
        nchwInputDescriptors, nchwOutputDescriptors, nchwFilterDescriptor, nchwConvolutionDescriptor,
        nchwConvolutionAlgorithms, nchwFilterBuf);
      if(!nchwNaive) {
        useNchwFallback = true;
        if(cudaHandles->logger != NULL)
          cudaHandles->logger->write(
            "ROCm backend: MIOpen offers only its naive direct-conv fallback for NHWC conv layer '" +
            name + "' (" + convDescStr + "); re-expressing it as NCHW with per-call transposes instead");
      }
      else if(cudaHandles->logger != NULL)
        cudaHandles->logger->write(
          "ROCm backend: WARNING: MIOpen offers only its naive direct-conv fallback for NHWC conv layer '" +
          name + "' (" + convDescStr + "), and the NCHW alternative is naive too - expect reduced conv performance");
    }
    else if(anyNaive && cudaHandles->logger != NULL)
      cudaHandles->logger->write(
        "ROCm backend: WARNING: MIOpen offers only its naive direct-conv fallback for conv layer '" +
        name + "' - expect reduced conv performance");
  }

  // Builds the MIOpen objects and the filter for one layout choice of this conv, and compiles the
  // chosen solution for every batch size. Returns whether any of those solutions is MIOpen's naive
  // fallback. shareKey may be NULL to keep the filter unshared, which the NCHW call must do: the
  // two filters hold different bytes and the registry keys on the descriptor address.
  static bool buildMiopenConvPath(
    CudaHandles* cudaHandles,
    const string& name,
    const ConvLayerDesc* desc,
    bool useFP16,
    bool filterNHWC,
    int maxBatchSize,
    const void* shareKey,
    ByBatchSizeView<miopenTensorDescriptor_t>& inputDescriptors,
    ByBatchSizeView<miopenTensorDescriptor_t>& outputDescriptors,
    OwnedDesc<miopenTensorDescriptor_t, miopenDestroyTensorDescriptor>& filterDescriptor,
    OwnedDesc<miopenConvolutionDescriptor_t, miopenDestroyConvolutionDescriptor>& convolutionDescriptor,
    std::unique_ptr<ByBatchSize<miopenConvSolution_t>>& convolutionAlgorithms,
    SharedDeviceBuf& filterBuf
  ) {
    int convYSize = desc->convYSize;
    int convXSize = desc->convXSize;
    int dilationY = desc->dilationY;
    int dilationX = desc->dilationX;
    int paddingX = (convXSize / 2) * dilationX;
    int paddingY = (convYSize / 2) * dilationY;
    int inChannels = desc->inChannels;
    int outChannels = desc->outChannels;

    CUDNN_ERR(name.c_str(),miopenCreateTensorDescriptor(&filterDescriptor.desc));
    if(filterNHWC) {
      int lens[4] = {outChannels, inChannels, convYSize, convXSize};
      CUDNN_ERR(name.c_str(),miopenSetNdTensorDescriptorWithLayout(
        filterDescriptor,
        (useFP16 ? miopenHalf : miopenFloat),
        miopenTensorNHWC,
        lens,
        4
      ));
    }
    else {
      CUDNN_ERR(name.c_str(),miopenSet4dTensorDescriptor(
        filterDescriptor,
        (useFP16 ? miopenHalf : miopenFloat),
        outChannels,
        inChannels,
        convYSize,
        convXSize
      ));
    }

    int yStride = 1;
    int xStride = 1;


    CUDNN_ERR(name.c_str(),miopenCreateConvolutionDescriptor(&convolutionDescriptor.desc));
    CUDNN_ERR(name.c_str(),miopenInitConvolutionDescriptor(
      convolutionDescriptor,
      miopenConvolution,
      paddingY,
      paddingX,
      yStride,
      xStride,
      dilationY,
      dilationX
    ));
    if(useFP16) {
      int alt = 1; // Nonzero enables MIOpen's alternate FP16 implementation on MI200-series and newer GPUs.
      CUDNN_ERR(name.c_str(),miopenSetConvolutionAttribute(convolutionDescriptor,MIOPEN_CONVOLUTION_ATTRIB_FP16_ALT_IMPL,alt));
    }

    convolutionAlgorithms = std::make_unique<ByBatchSize<miopenConvSolution_t>>(maxBatchSize);

    bool anyNaive = false;
    for(int batchSize = 1; batchSize <= maxBatchSize; batchSize++) {
      const miopenTensorDescriptor_t& inputDescriptor = inputDescriptors[batchSize];
      const miopenTensorDescriptor_t& outputDescriptor = outputDescriptors[batchSize];
      size_t availableAlgoCount = 0;
      CUDNN_ERR(name.c_str(),miopenConvolutionForwardGetSolutionCount(
        cudaHandles->cudnn,
        filterDescriptor,
        inputDescriptor,
        convolutionDescriptor,
        outputDescriptor,
        &availableAlgoCount
      ));
      if(availableAlgoCount <= 0)
        throw StringError("miopenConvolutionForwardGetSolutionCount returned 0 algorithms?");
      std::vector<miopenConvSolution_t> solutions(availableAlgoCount);
      size_t returnedAlgoCount = 0;
      CUDNN_ERR(name.c_str(),miopenConvolutionForwardGetSolution(
          cudaHandles->cudnn,
          filterDescriptor,
          inputDescriptor,
          convolutionDescriptor,
          outputDescriptor,
          availableAlgoCount,
          &returnedAlgoCount,
          solutions.data()
        ));
      if(returnedAlgoCount <= 0)
        throw StringError("miopenConvolutionForwardGetSolution returned no algorithms?");
      (*convolutionAlgorithms)[batchSize] = solutions[0];
      // MIOpen stamps its naive fallback with a sentinel time of 1000ms so that it sorts last.
      // Do not also test the solver id: ids are not stable across MIOpen versions, and one that
      // names the naive solver in one version names an implicit-GEMM solver in another, which
      // would reroute a well-served layer through the transposes. The algorithm check stops a
      // genuinely slow estimate from a real solver tripping this.
      if(solutions[0].time >= 999.0f && solutions[0].algorithm == miopenConvolutionAlgoDirect)
        anyNaive = true;
      CUDNN_ERR(name.c_str(),miopenConvolutionForwardCompileSolution(
        cudaHandles->cudnn,
        filterDescriptor,
        inputDescriptor,
        convolutionDescriptor,
        outputDescriptor,
        (*convolutionAlgorithms)[batchSize].solution_id
      ));
    }

    testAssert(desc->weights.size() == convYSize * convXSize * inChannels * outChannels);

    if(filterNHWC) {
      vector<float> weightsTransposed(desc->weights.size());
      for(int y = 0; y < convYSize; y++) {
        for(int x = 0; x < convXSize; x++) {
          for(int ic = 0; ic < inChannels; ic++) {
            for(int oc = 0; oc < outChannels; oc++) {
              weightsTransposed[((oc*convYSize + y)*convXSize + x)*inChannels + ic] =
                desc->weights[((oc*inChannels + ic)*convYSize + y)*convXSize + x];
            }
          }
        }
      }
      filterBuf = mallocAndCopyToDeviceMaybeShared(
        name,
        [&](void*& b){ CudaUtils::mallocAndCopyToDevice(name,weightsTransposed,b,useFP16); },
        shareKey, SharedWeightsKind::CONV_FILTER, useFP16, cudaHandles->shareModelWeights);
    }
    else
      filterBuf = mallocAndCopyToDeviceMaybeShared(
        name,
        [&](void*& b){ CudaUtils::mallocAndCopyToDevice(name,desc->weights,b,useFP16); },
        shareKey, SharedWeightsKind::CONV_FILTER, useFP16, cudaHandles->shareModelWeights);

    return anyNaive;
  }

  size_t requiredWorkspaceBytes(
    CudaHandles* cudaHandles,
    int batchSize
  ) const {
    if(use1x1Matmul)
      return 0;
    if(useNchwFallback) {
      size_t workspaceBytes = 0;
      CUDNN_ERR(name.c_str(),miopenConvolutionForwardGetSolutionWorkspaceSize(
        cudaHandles->cudnn,
        nchwFilterDescriptor,
        nchwInputDescriptors[batchSize],
        nchwConvolutionDescriptor,
        nchwOutputDescriptors[batchSize],
        (*nchwConvolutionAlgorithms)[batchSize].solution_id,
        &workspaceBytes
      ));
      return workspaceBytes;
    }
    size_t workspaceBytes = 0;
    CUDNN_ERR(name.c_str(),miopenConvolutionForwardGetSolutionWorkspaceSize(
      cudaHandles->cudnn,
      filterDescriptor,
      inputDescriptors[batchSize],
      convolutionDescriptor,
      outputDescriptors[batchSize],
      (*convolutionAlgorithms)[batchSize].solution_id,
      &workspaceBytes
    ));
    return workspaceBytes;
  }

  void apply(
    CudaHandles* cudaHandles,
    ScratchBuffers* scratch,
    int batchSize,
    bool accumulate,
    void* inputBuf,
    void* outputBuf,
    void* workspaceBuf,
    size_t workspaceBytes
  ) const {
    if(use1x1Matmul) {
      // out[outC, tokens] = W[outC, inC] x in[inC, tokens], tokens = batchSize * spatial. NHWC
      // buffers are [tokens, C] row-major, which is the [C, tokens] column-major hipBLAS wants.
      int tokens = batchSize * matmulSpatialSize;
      if(!usingFP16) {
        const float alpha = 1.0f;
        const float beta = accumulate ? 1.0f : 0.0f;
        CUBLAS_ERR(name.c_str(),cublasSgemm(
          cudaHandles->cublas, CUBLAS_OP_N, CUBLAS_OP_N,
          outChannels, tokens, inChannels,
          &alpha, (const float*)matmulWeightBuf.buf, outChannels,
          (const float*)inputBuf, inChannels,
          &beta, (float*)outputBuf, outChannels));
      }
      else {
#if defined(hipblasVersionMajor) && hipblasVersionMajor >= 2
        // FP32 compute for accuracy, as in MatMulLayer.
        const float alpha = 1.0f;
        const float beta = accumulate ? 1.0f : 0.0f;
        CUBLAS_ERR(name.c_str(),hipblasGemmEx(
          cudaHandles->cublas,
          HIPBLAS_OP_N,
          HIPBLAS_OP_N,
          outChannels,
          tokens,
          inChannels,
          &alpha,
          matmulWeightBuf.buf, HIP_R_16F, outChannels,
          inputBuf, HIP_R_16F, inChannels,
          &beta,
          outputBuf, HIP_R_16F, outChannels,
          HIPBLAS_COMPUTE_32F,
          HIPBLAS_GEMM_DEFAULT
        ));
#else
        // hipBLAS older than 2.0, below the minimum Compiling.md documents.
        const cublas_half_t* alpha = (const cublas_half_t*)scratch->oneBuf;
        const cublas_half_t* betaHalf = (const cublas_half_t*)(accumulate ? scratch->oneBuf : scratch->zeroBuf);
        CUBLAS_ERR(name.c_str(),cublasHgemm(
          cudaHandles->cublas, CUBLAS_OP_N, CUBLAS_OP_N,
          outChannels, tokens, inChannels,
          alpha, (const cublas_half_t*)matmulWeightBuf.buf, outChannels,
          (const cublas_half_t*)inputBuf, inChannels,
          betaHalf, (cublas_half_t*)outputBuf, outChannels));
#endif
      }
      return;
    }
    // miopenConvolutionForwardImmediate does NOT support alpha/beta (unlike cuDNN).
    // When accumulate=true, we need: outputBuf = conv(inputBuf) + outputBuf (residual skip
    // connection). Convolve into a scratch buffer and add that into outputBuf, so that emulating
    // accumulate costs only the add. Convolving in place instead would additionally require saving
    // outputBuf to scratch first, a full copy of the output tensor per accumulating conv.
    // The SizedBuf may be handed back to the allocator's pool before the async ops complete, but
    // any later user of the same allocation enqueues on the same stream, so this is safe - the
    // same pattern every other transient buffer in this file relies on.
    // The buffer is sized by maxBatchSize (getBufSizeXY), NOT the actual batch size: the allocator
    // pools buffers by exact byte size and retains them for the handle's lifetime, so sizing by
    // actual batch size would mint (and keep) a separate allocation for every distinct batch size
    // that ever runs.
    size_t outputElts = (size_t)batchSize * outChannels * nnXLen * nnYLen;
    if(outputElts >= (size_t)2147483647)
      throw StringError(name + ": conv output element count exceeds the 32-bit index limit used by the accumulate kernel");
    std::unique_ptr<SizedBuf<void*>> accumBuf;
    void* convOutputBuf = outputBuf;
    if(accumulate) {
      accumBuf = std::make_unique<SizedBuf<void*>>(scratch->allocator, scratch->getBufSizeXY(outChannels));
      convOutputBuf = accumBuf->buf;
    }

    if(useNchwFallback) {
      // Transpose to NCHW, run the conv, transpose back. Targeting convOutputBuf keeps
      // accumulate to a single add, as on the direct path below.
      int xySize = nnXLen * nnYLen;
      SizedBuf<void*> inNchw(scratch->allocator, scratch->getBufSizeXY(inChannels));
      SizedBuf<void*> outNchw(scratch->allocator, scratch->getBufSizeXY(outChannels));
      if(usingFP16)
        customCudaCopyNHWCtoNCHW((const half*)inputBuf, (half*)inNchw.buf, batchSize, inChannels, xySize,cudaHandles->stream);
      else
        customCudaCopyNHWCtoNCHW((const float*)inputBuf, (float*)inNchw.buf, batchSize, inChannels, xySize,cudaHandles->stream);
      CUDNN_ERR(name.c_str(), miopenConvolutionForwardImmediate(
        cudaHandles->cudnn,
        nchwFilterDescriptor,
        nchwFilterBuf.buf,
        nchwInputDescriptors[batchSize],
        inNchw.buf,
        nchwConvolutionDescriptor,
        nchwOutputDescriptors[batchSize],
        outNchw.buf,
        workspaceBuf,
        workspaceBytes,
        (*nchwConvolutionAlgorithms)[batchSize].solution_id
      ));
      if(usingFP16)
        customCudaCopyNCHWtoNHWC((const half*)outNchw.buf, (half*)convOutputBuf, batchSize, outChannels, xySize,cudaHandles->stream);
      else
        customCudaCopyNCHWtoNHWC((const float*)outNchw.buf, (float*)convOutputBuf, batchSize, outChannels, xySize,cudaHandles->stream);
      if(accumulate) {
        if(usingFP16)
          customCudaAddTensorInplace((half*)outputBuf, (const half*)accumBuf->buf, (int)outputElts,cudaHandles->stream);
        else
          customCudaAddTensorInplace((float*)outputBuf, (const float*)accumBuf->buf, (int)outputElts,cudaHandles->stream);
      }
      CUDA_ERR(name.c_str(), hipPeekAtLastError());
      return;
    }

    CUDNN_ERR(name.c_str(), miopenConvolutionForwardImmediate(
      cudaHandles->cudnn,
      filterDescriptor,
      filterBuf.buf,
      inputDescriptors[batchSize],
      inputBuf,
      convolutionDescriptor,
      outputDescriptors[batchSize],
      convOutputBuf,
      workspaceBuf,
      workspaceBytes,
      (*convolutionAlgorithms)[batchSize].solution_id
    ));

    if(accumulate) {
      if(usingFP16)
        customCudaAddTensorInplace((half*)outputBuf, (const half*)accumBuf->buf, (int)outputElts,cudaHandles->stream);
      else
        customCudaAddTensorInplace((float*)outputBuf, (const float*)accumBuf->buf, (int)outputElts,cudaHandles->stream);
      CUDA_ERR(name.c_str(), hipPeekAtLastError());
    }
  }

};
#endif


//---------------------------------------------------------------------------------

struct BatchNormLayer {
  const string name;
  const int numChannels;
  const float epsilon;
  const int activation;
  const int nnXLen;
  const int nnYLen;

  const bool usingFP16;
  const bool usingNHWC;

  OwnedDeviceBuf mergedScaleBuf;
  OwnedDeviceBuf mergedBiasBuf;

  BatchNormLayer() = delete;
  BatchNormLayer(const BatchNormLayer&) = delete;
  BatchNormLayer& operator=(const BatchNormLayer&) = delete;

  BatchNormLayer(
    CudaHandles* cudaHandles,
    const BatchNormLayerDesc* desc,
    const ActivationLayerDesc* actDesc,
    int nnX,
    int nnY,
    bool useFP16,
    bool useNHWC
  ) :
    name(desc->name),
    numChannels(desc->numChannels),
    epsilon(desc->epsilon),
    activation(actDesc->activation),
    nnXLen(nnX),
    nnYLen(nnY),
    usingFP16(useFP16),
    usingNHWC(useNHWC)
  {
    (void)cudaHandles;

    testAssert(desc->mean.size() == numChannels);
    testAssert(desc->variance.size() == numChannels);
    testAssert(desc->scale.size() == numChannels);
    testAssert(desc->bias.size() == numChannels);
    testAssert(desc->mergedScale.size() == numChannels);
    testAssert(desc->mergedBias.size() == numChannels);
    CudaUtils::mallocAndCopyToDevice(name,desc->mergedScale,mergedScaleBuf.buf,useFP16);
    CudaUtils::mallocAndCopyToDevice(name,desc->mergedBias,mergedBiasBuf.buf,useFP16);
  }
  void apply(
    CudaHandles* cudaHandles,
    int batchSize,
    void* inputBuf,
    const void* maskBuf, //ok to be null
    void* outputBuf
  ) const {
    if(!usingFP16) {
      if(!usingNHWC)
        customCudaApplyCScaleBiasNCHW((const float*)inputBuf,(float*)outputBuf,(const float*)mergedScaleBuf.buf,(const float*)mergedBiasBuf.buf,
                                      (const float*)maskBuf,
                                      batchSize,numChannels,nnXLen*nnYLen,activation,cudaHandles->stream);
      else
        customCudaApplyCScaleBiasNHWC((const float*)inputBuf,(float*)outputBuf,(const float*)mergedScaleBuf.buf,(const float*)mergedBiasBuf.buf,
                                      (const float*)maskBuf,
                                      batchSize,nnXLen*nnYLen,numChannels,activation,cudaHandles->stream);
    }
    else {
      if(!usingNHWC)
        customCudaApplyCScaleBiasNCHW((const half*)inputBuf,(half*)outputBuf,(const half*)mergedScaleBuf.buf,(const half*)mergedBiasBuf.buf,
                                      (const half*)maskBuf,
                                      batchSize,numChannels,nnXLen*nnYLen,activation,cudaHandles->stream);
      else
        customCudaApplyCScaleBiasNHWC((const half*)inputBuf,(half*)outputBuf,(const half*)mergedScaleBuf.buf,(const half*)mergedBiasBuf.buf,
                                      (const half*)maskBuf,
                                      batchSize,nnXLen*nnYLen,numChannels,activation,cudaHandles->stream);
    }
    CUDA_ERR(name.c_str(),cudaPeekAtLastError());
  }

};


//---------------------------------------------------------------------------------

struct MatMulLayer {
  const string name;
  const int inChannels;
  const int outChannels;
  const bool usingFP16;
  SharedDeviceBuf matBuf;

  MatMulLayer() = delete;
  MatMulLayer(const MatMulLayer&) = delete;
  MatMulLayer& operator=(const MatMulLayer&) = delete;

  MatMulLayer(
    CudaHandles* cudaHandles,
    const MatMulLayerDesc* desc,
    bool useFP16
  ) :
    name(desc->name),
    inChannels(desc->inChannels),
    outChannels(desc->outChannels),
    usingFP16(useFP16)
  {
    if(inChannels > 0 && outChannels > 0) {
      testAssert(desc->weights.size() == inChannels * outChannels);
      matBuf = mallocAndCopyToDeviceMaybeShared(
        name,
        [&](void*& b){ CudaUtils::mallocAndCopyToDevice(name,desc->weights,b,useFP16); },
        desc, SharedWeightsKind::MATMUL, useFP16, cudaHandles->shareModelWeights);
    }
  }

  size_t requiredWorkspaceBytes(
    CudaHandles* cudaHandles
  ) const {
    (void)cudaHandles;
    size_t workspaceBytes = 0;
    return workspaceBytes;
  }

  //If accumulate is true, adds the matmul result into outputBuf (GEMM beta=1) instead of
  //overwriting it, letting a residual add fuse into the GEMM epilogue.
  void apply(
    CudaHandles* cudaHandles,
    ScratchBuffers* scratch,
    int batchSize,
    void* inputBuf,
    void* outputBuf,
    void* workspaceBuf,
    size_t workspaceBytes,
    bool accumulate = false
  ) const {
    (void)workspaceBuf;
    (void)workspaceBytes;
    assert(inChannels > 0 && outChannels > 0);

    if(!usingFP16) {
      const float alpha = 1.0f;
      const float beta = accumulate ? 1.0f : 0.0f;
      CUBLAS_ERR(name.c_str(),cublasSgemm(
        cudaHandles->cublas,
        CUBLAS_OP_N,
        CUBLAS_OP_N,
        outChannels,
        batchSize,
        inChannels,
        &alpha,
        (const float*)matBuf.buf,outChannels,
        (const float*)inputBuf,inChannels,
        &beta,
        (float*)outputBuf,outChannels
      ));
    }
    else {
#if defined(KATAGO_GPU_HIP) && defined(hipblasVersionMajor) && hipblasVersionMajor >= 2
      //hipblasHgemm computes in FP16, which costs real accuracy - enough that on transformer nets
      //the ROCm backend's FP16 error exceeded the backend accuracy test limits. hipblasGemmEx with
      //FP16 in/out and FP32 compute fixes that and costs nothing in speed. alpha/beta must match
      //the compute type, so they are plain host floats here rather than the half zeroBuf/oneBuf
      //the CUDA path uses.
      const float alpha = 1.0f;
      const float beta = accumulate ? 1.0f : 0.0f;
      CUBLAS_ERR(name.c_str(),hipblasGemmEx(
        cudaHandles->cublas,
        HIPBLAS_OP_N,
        HIPBLAS_OP_N,
        outChannels,
        batchSize,
        inChannels,
        &alpha,
        matBuf.buf, HIP_R_16F, outChannels,
        inputBuf, HIP_R_16F, inChannels,
        &beta,
        outputBuf, HIP_R_16F, outChannels,
        HIPBLAS_COMPUTE_32F,
        HIPBLAS_GEMM_DEFAULT
      ));
#else
      //CUDA, or hipBLAS older than 2.0 (below ROCm 6.0, itself below the minimum Compiling.md
      //documents). There hipblasGemmEx has no spelling that takes hipblasComputeType_t at all, so
      //keep the FP16-compute path: correct, just less accurate and slower.
      const cublas_half_t* alpha = (const cublas_half_t*)scratch->oneBuf;
      const cublas_half_t* beta = (const cublas_half_t*)(accumulate ? scratch->oneBuf : scratch->zeroBuf);
      CUBLAS_ERR(name.c_str(),cublasHgemm(
        cudaHandles->cublas,
        CUBLAS_OP_N,
        CUBLAS_OP_N,
        outChannels,
        batchSize,
        inChannels,
        alpha,
        (const cublas_half_t*)matBuf.buf,outChannels,
        (const cublas_half_t*)inputBuf,inChannels,
        beta,
        (cublas_half_t*)outputBuf,outChannels
      ));
#endif
    }

  }

};

//---------------------------------------------------------------------------------

//Run `count` GEMMs that share one input (B stride 0) over weight matrices packed contiguously
//in device memory, writing the outputs at a fixed stride. Used to batch the Q/K/V projections
//and the FFN linear+gate projections of transformer blocks into single GEMM calls.
//Precision behavior matches MatMulLayer::apply.
static void applySharedInputStridedMatMuls(
  CudaHandles* cudaHandles,
  ScratchBuffers* scratch,
  bool usingFP16,
  const string& name,
  int outChannels,
  int batchSize,
  int inChannels,
  const void* packedWeights,
  const void* inputBuf,
  void* outputBuf,
  long long outputStrideElts,
  int count
) {
  const long long weightStrideElts = (long long)inChannels * outChannels;
  if(!usingFP16) {
    const float alpha = 1.0f;
    const float beta = 0.0f;
    CUBLAS_ERR(name.c_str(),cublasSgemmStridedBatched(
      cudaHandles->cublas,
      CUBLAS_OP_N,
      CUBLAS_OP_N,
      outChannels,
      batchSize,
      inChannels,
      &alpha,
      (const float*)packedWeights, outChannels, weightStrideElts,
      (const float*)inputBuf, inChannels, 0LL,
      &beta,
      (float*)outputBuf, outChannels, outputStrideElts,
      count
    ));
  }
  else {
#if defined(KATAGO_GPU_HIP) && defined(hipblasVersionMajor) && hipblasVersionMajor >= 2
    //FP32 compute for accuracy, same reasoning as in MatMulLayer::apply.
    const float alpha = 1.0f;
    const float beta = 0.0f;
    CUBLAS_ERR(name.c_str(),hipblasGemmStridedBatchedEx(
      cudaHandles->cublas,
      HIPBLAS_OP_N,
      HIPBLAS_OP_N,
      outChannels,
      batchSize,
      inChannels,
      &alpha,
      packedWeights, HIP_R_16F, outChannels, weightStrideElts,
      inputBuf, HIP_R_16F, inChannels, 0LL,
      &beta,
      outputBuf, HIP_R_16F, outChannels, outputStrideElts,
      count,
      HIPBLAS_COMPUTE_32F,
      HIPBLAS_GEMM_DEFAULT
    ));
#else
    const cublas_half_t* alpha = (const cublas_half_t*)scratch->oneBuf;
    const cublas_half_t* beta = (const cublas_half_t*)scratch->zeroBuf;
    CUBLAS_ERR(name.c_str(),cublasHgemmStridedBatched(
      cudaHandles->cublas,
      CUBLAS_OP_N,
      CUBLAS_OP_N,
      outChannels,
      batchSize,
      inChannels,
      alpha,
      (const cublas_half_t*)packedWeights, outChannels, weightStrideElts,
      (const cublas_half_t*)inputBuf, inChannels, 0LL,
      beta,
      (cublas_half_t*)outputBuf, outChannels, outputStrideElts,
      count
    ));
#endif
  }
}

//---------------------------------------------------------------------------------

struct MatBiasLayer {
  const string name;
  const int numChannels;
  const bool usingFP16;
  const int activation;

  OwnedDeviceBuf biasBuf;

  MatBiasLayer() = delete;
  MatBiasLayer(const MatBiasLayer&) = delete;
  MatBiasLayer& operator=(const MatBiasLayer&) = delete;

  MatBiasLayer(
    CudaHandles* cudaHandles,
    const MatBiasLayerDesc* desc,
    bool useFP16,
    int activation_
  ) :
    name(desc->name),
    numChannels(desc->numChannels),
    usingFP16(useFP16),
    activation(activation_)
  {
    (void)cudaHandles;
    if(numChannels > 0) {
      testAssert(desc->weights.size() == numChannels);
      CudaUtils::mallocAndCopyToDevice(name,desc->weights,biasBuf.buf,useFP16);
    }
  }

  void apply(
    CudaHandles* cudaHandles,
    int batchSize,
    void* matBuf
  ) const {
    assert(numChannels > 0);
    if(!usingFP16) {
      customCudaAddCBiasInplaceNC((float*)matBuf,(const float*)biasBuf.buf,batchSize,numChannels,activation,cudaHandles->stream);
      CUDA_ERR(name.c_str(),cudaPeekAtLastError());
    }
    else {
      customCudaAddCBiasInplaceNC((half*)matBuf,(const half*)biasBuf.buf,batchSize,numChannels,activation,cudaHandles->stream);
      CUDA_ERR(name.c_str(),cudaPeekAtLastError());
    }
  }

};

//---------------------------------------------------------------------------------

struct NormActConv {
  const BatchNormLayer norm;
  const ConvLayer conv;

  const int inChannels;
  const int outChannels;
  const int nnXLen;
  const int nnYLen;
  const bool usingFP16;
  const bool usingNHWC;

  NormActConv() = delete;
  NormActConv(const NormActConv&) = delete;
  NormActConv& operator=(const NormActConv&) = delete;

  NormActConv(
    CudaHandles* cudaHandles,
    CudnnManager* manager,
    const BatchNormLayerDesc* normDesc,
    const ActivationLayerDesc* actDesc,
    const ConvLayerDesc* convDesc,
    int nnX,
    int nnY,
    bool useFP16,
    bool useNHWC
  ): norm(cudaHandles,normDesc,actDesc,nnX,nnY,useFP16,useNHWC),
     conv(cudaHandles,manager,convDesc,useFP16,useNHWC),
     inChannels(norm.numChannels),
     outChannels(conv.outChannels),
     nnXLen(nnX),
     nnYLen(nnY),
     usingFP16(useFP16),
     usingNHWC(useNHWC)
  {
    testAssert(norm.numChannels == conv.inChannels);
  }

  ~NormActConv()
  {}

  size_t requiredWorkspaceBytes(
    CudaHandles* cudaHandles,
    int batchSize
  ) const {
    size_t bytes = 0;
    size_t b;
    b = conv.requiredWorkspaceBytes(cudaHandles,batchSize);
    bytes = std::max(bytes,b);
    return bytes;
  }

  void apply(
    CudaHandles* cudaHandles,
    ScratchBuffers* scratch,
    int batchSize,
    bool accumulate,
    void* inBuf,
    void* inScratchBuf,
    void* outBuf,
    void* maskBuf,
    void* workspaceBuf,
    size_t workspaceBytes
  ) const {
    norm.apply(cudaHandles,batchSize,inBuf,maskBuf,inScratchBuf);
#ifdef DEBUG_INTERMEDIATE_VALUES
    CudaUtils::debugPrint3D(string("AFTER NORM "), inScratchBuf, batchSize, inChannels, nnXLen*nnYLen, usingNHWC, usingFP16);
#endif
    conv.apply(cudaHandles,scratch,batchSize,accumulate,inScratchBuf,outBuf,workspaceBuf,workspaceBytes);
  }

};


//---------------------------------------------------------------------------------

struct ResidualBlock {
  const string name;
  const NormActConv normActConv1;
  const NormActConv normActConv2;

  ResidualBlock() = delete;
  ResidualBlock(const ResidualBlock&) = delete;
  ResidualBlock& operator=(const ResidualBlock&) = delete;

  ResidualBlock(
    CudaHandles* cudaHandles,
    CudnnManager* manager,
    const ResidualBlockDesc* desc,
    int nnX,
    int nnY,
    bool useFP16,
    bool useNHWC
  ): name(desc->name),
     normActConv1(cudaHandles,manager,&desc->preBN,&desc->preActivation,&desc->regularConv,nnX,nnY,useFP16,useNHWC),
     normActConv2(cudaHandles,manager,&desc->midBN,&desc->midActivation,&desc->finalConv,nnX,nnY,useFP16,useNHWC)
  {
  }

  ~ResidualBlock()
  {}

  size_t requiredWorkspaceBytes(
    CudaHandles* cudaHandles,
    int batchSize
  ) const {
    size_t bytes = 0;
    size_t b;
    b = normActConv1.requiredWorkspaceBytes(cudaHandles,batchSize);
    bytes = std::max(bytes,b);
    b = normActConv2.requiredWorkspaceBytes(cudaHandles,batchSize);
    bytes = std::max(bytes,b);
    return bytes;
  }

  void apply(
    CudaHandles* cudaHandles,
    ScratchBuffers* scratch,
    int batchSize,
    void* trunkBuf,
    void* trunkScratchBuf,
    void* maskBuf,
    void* workspaceBuf,
    size_t workspaceBytes
  ) const {
    SizedBuf<void*> midIn(scratch->allocator, scratch->getBufSizeXY(normActConv1.outChannels));
    SizedBuf<void*> midScratch(scratch->allocator, scratch->getBufSizeXY(normActConv1.outChannels));
    normActConv1.apply(cudaHandles,scratch,batchSize,false,trunkBuf,trunkScratchBuf,midIn.buf,maskBuf,workspaceBuf,workspaceBytes);
    normActConv2.apply(cudaHandles,scratch,batchSize,true,midIn.buf,midScratch.buf,trunkBuf,maskBuf,workspaceBuf,workspaceBytes);
  }

};


//----------------------------------------------------------------------------


struct GlobalPoolingResidualBlock {
  const string name;
  const BatchNormLayer preBN;
  const ConvLayer regularConv;
  const ConvLayer gpoolConv;
  const BatchNormLayer gpoolBN;
  const MatMulLayer gpoolToBiasMul;
  const NormActConv normActConv2;

  const int nnXLen;
  const int nnYLen;
  const int regularChannels;
  const int gpoolChannels;
  const bool usingFP16;
  const bool usingNHWC;

  GlobalPoolingResidualBlock() = delete;
  GlobalPoolingResidualBlock(const GlobalPoolingResidualBlock&) = delete;
  GlobalPoolingResidualBlock& operator=(const GlobalPoolingResidualBlock&) = delete;

  GlobalPoolingResidualBlock(
    CudaHandles* cudaHandles,
    CudnnManager* manager,
    const GlobalPoolingResidualBlockDesc* desc,
    int nnX,
    int nnY,
    bool useFP16,
    bool useNHWC
  ): name(desc->name),
     preBN(cudaHandles,&desc->preBN,&desc->preActivation,nnX,nnY,useFP16,useNHWC),
     regularConv(cudaHandles,manager,&desc->regularConv,useFP16,useNHWC),
     gpoolConv(cudaHandles,manager,&desc->gpoolConv,useFP16,useNHWC),
     gpoolBN(cudaHandles,&desc->gpoolBN,&desc->gpoolActivation,nnX,nnY,useFP16,useNHWC),
     gpoolToBiasMul(cudaHandles,&desc->gpoolToBiasMul,useFP16),
     normActConv2(cudaHandles,manager,&desc->midBN,&desc->midActivation,&desc->finalConv,nnX,nnY,useFP16,useNHWC),
     nnXLen(nnX),
     nnYLen(nnY),
     regularChannels(desc->regularConv.outChannels),
     gpoolChannels(desc->gpoolConv.outChannels),
     usingFP16(useFP16),
     usingNHWC(useNHWC)
  {
  }

  ~GlobalPoolingResidualBlock() {
  }

  size_t requiredWorkspaceBytes(
    CudaHandles* cudaHandles,
    int batchSize
  ) const {
    size_t bytes = 0;
    size_t b;
    b = regularConv.requiredWorkspaceBytes(cudaHandles,batchSize);
    bytes = std::max(bytes,b);
    b = gpoolConv.requiredWorkspaceBytes(cudaHandles,batchSize);
    bytes = std::max(bytes,b);
    b = gpoolToBiasMul.requiredWorkspaceBytes(cudaHandles);
    bytes = std::max(bytes,b);
    b = normActConv2.requiredWorkspaceBytes(cudaHandles,batchSize);
    bytes = std::max(bytes,b);
    b = sizeof(float)*batchSize*gpoolChannels*nnXLen*nnYLen;
    bytes = std::max(bytes,b);
    return bytes;
  }

  void apply(
    CudaHandles* cudaHandles,
    ScratchBuffers* scratch,
    int batchSize,
    void* trunkBuf,
    void* trunkScratchBuf,
    void* maskBuf,
    float* maskSumBuf,
    void* workspaceBuf,
    size_t workspaceBytes
  ) const {
    SizedBuf<void*> regularOut(scratch->allocator, scratch->getBufSizeXY(regularChannels));
    SizedBuf<void*> regularScratch(scratch->allocator, scratch->getBufSizeXY(regularChannels));
    SizedBuf<void*> gpoolOut(scratch->allocator, scratch->getBufSizeXY(gpoolChannels));
    SizedBuf<void*> gpoolOut2(scratch->allocator, scratch->getBufSizeXY(gpoolChannels));
    SizedBuf<void*> gpoolConcat(scratch->allocator, scratch->getBufSize(gpoolChannels*3));
    SizedBuf<void*> gpoolBias(scratch->allocator, scratch->getBufSize(regularChannels));

    preBN.apply(cudaHandles,batchSize,trunkBuf,maskBuf,trunkScratchBuf);
    regularConv.apply(cudaHandles,scratch,batchSize,false,trunkScratchBuf,regularOut.buf,workspaceBuf,workspaceBytes);
    gpoolConv.apply(cudaHandles,scratch,batchSize,false,trunkScratchBuf,gpoolOut.buf,workspaceBuf,workspaceBytes);
    gpoolBN.apply(cudaHandles,batchSize,gpoolOut.buf,maskBuf,gpoolOut2.buf);

    if(!usingFP16) {
      if(!usingNHWC)
        customCudaPoolRowsGPoolNCHW((const float*)gpoolOut2.buf,(float*)gpoolConcat.buf,batchSize,gpoolChannels,nnXLen*nnYLen,(const float*)maskBuf,maskSumBuf,cudaHandles->stream);
      else
        customCudaPoolRowsGPoolNHWC((const float*)gpoolOut2.buf,(float*)gpoolConcat.buf,batchSize,nnXLen*nnYLen,gpoolChannels,(const float*)maskBuf,maskSumBuf,cudaHandles->stream);
    }
    else {
      if(!usingNHWC)
        customCudaPoolRowsGPoolNCHW((const half*)gpoolOut2.buf,(half*)gpoolConcat.buf,batchSize,gpoolChannels,nnXLen*nnYLen,(const half*)maskBuf,maskSumBuf,cudaHandles->stream);
      else
        customCudaPoolRowsGPoolNHWC((const half*)gpoolOut2.buf,(half*)gpoolConcat.buf,batchSize,nnXLen*nnYLen,gpoolChannels,(const half*)maskBuf,maskSumBuf,cudaHandles->stream);
    }
    CUDA_ERR(name.c_str(),cudaPeekAtLastError());

    gpoolToBiasMul.apply(cudaHandles,scratch,batchSize,gpoolConcat.buf,gpoolBias.buf,workspaceBuf,workspaceBytes);

    if(!usingFP16) {
      if(!usingNHWC)
        customCudaAddNCBiasInplaceNCHW((float*)regularOut.buf,(const float*)gpoolBias.buf,batchSize,regularChannels,nnXLen*nnYLen,cudaHandles->stream);
      else
        customCudaAddNCBiasInplaceNHWC((float*)regularOut.buf,(const float*)gpoolBias.buf,batchSize,nnXLen*nnYLen,regularChannels,cudaHandles->stream);
    }
    else {
      if(!usingNHWC)
        customCudaAddNCBiasInplaceNCHW((half*)regularOut.buf,(const half*)gpoolBias.buf,batchSize,regularChannels,nnXLen*nnYLen,cudaHandles->stream);
      else
        customCudaAddNCBiasInplaceNHWC((half*)regularOut.buf,(const half*)gpoolBias.buf,batchSize,nnXLen*nnYLen,regularChannels,cudaHandles->stream);
    }
    CUDA_ERR(name.c_str(),cudaPeekAtLastError());

    normActConv2.apply(cudaHandles,scratch,batchSize,true,regularOut.buf,regularScratch.buf,trunkBuf,maskBuf,workspaceBuf,workspaceBytes);
  }

};

//------------------------------------------------------------------------------

struct BlockStack {
  const int numBlocks;
  const int trunkNumChannels;
  const int nnXLen;
  const int nnYLen;
  const bool usingFP16;
  const bool usingNHWC;
  vector<pair<int,unique_ptr_void>> blocks;

  BlockStack() = delete;
  BlockStack(const BlockStack&) = delete;
  BlockStack& operator=(const BlockStack&) = delete;

  BlockStack(
    CudaHandles* cudaHandles,
    CudnnManager* manager,
    int nBlocks,
    int trunkChannels,
    const std::vector<std::pair<int, unique_ptr_void>>& descBlocks,
    int nnX,
    int nnY,
    bool useFP16,
    bool useNHWC
  );
  ~BlockStack();

  size_t requiredWorkspaceBytes(
    CudaHandles* cudaHandles,
    int batchSize
  ) const;

  void apply(
    CudaHandles* cudaHandles,
    ScratchBuffers* scratch,
    int batchSize,
    void* maskBuf,
    float* maskSumBuf,
    void* trunkBuf,
    void* trunkScratchBuf,
    void* workspaceBuf,
    size_t workspaceBytes
  ) const;

};

//------------------------------------------------------------------------------

struct NestedBottleneckResidualBlock {
  const string name;
  const NormActConv normActConv1;
  const BlockStack blocks;
  const NormActConv normActConv2;

  NestedBottleneckResidualBlock() = delete;
  NestedBottleneckResidualBlock(const NestedBottleneckResidualBlock&) = delete;
  NestedBottleneckResidualBlock& operator=(const NestedBottleneckResidualBlock&) = delete;

  NestedBottleneckResidualBlock(
    CudaHandles* cudaHandles,
    CudnnManager* manager,
    const NestedBottleneckResidualBlockDesc* desc,
    int nnX,
    int nnY,
    bool useFP16,
    bool useNHWC
  ): name(desc->name),
     normActConv1(cudaHandles,manager,&desc->preBN,&desc->preActivation,&desc->preConv,nnX,nnY,useFP16,useNHWC),
     blocks(cudaHandles,manager,desc->numBlocks,desc->preConv.outChannels,desc->blocks,nnX,nnY,useFP16,useNHWC),
     normActConv2(cudaHandles,manager,&desc->postBN,&desc->postActivation,&desc->postConv,nnX,nnY,useFP16,useNHWC)
  {
  }

  ~NestedBottleneckResidualBlock()
  {}

  size_t requiredWorkspaceBytes(
    CudaHandles* cudaHandles,
    int batchSize
  ) const {
    size_t bytes = 0;
    size_t b;
    b = normActConv1.requiredWorkspaceBytes(cudaHandles,batchSize);
    bytes = std::max(bytes,b);
    b = blocks.requiredWorkspaceBytes(cudaHandles,batchSize);
    bytes = std::max(bytes,b);
    b = normActConv2.requiredWorkspaceBytes(cudaHandles,batchSize);
    bytes = std::max(bytes,b);
    return bytes;
  }

  void apply(
    CudaHandles* cudaHandles,
    ScratchBuffers* scratch,
    int batchSize,
    void* trunkBuf,
    void* trunkScratchBuf,
    void* maskBuf,
    float* maskSumBuf,
    void* workspaceBuf,
    size_t workspaceBytes
  ) const {
    SizedBuf<void*> mid(scratch->allocator, scratch->getBufSizeXY(normActConv1.outChannels));
    SizedBuf<void*> midScratch(scratch->allocator, scratch->getBufSizeXY(normActConv1.outChannels));
    assert(normActConv1.outChannels == normActConv2.inChannels);
    normActConv1.apply(cudaHandles,scratch,batchSize,false,trunkBuf,trunkScratchBuf,mid.buf,maskBuf,workspaceBuf,workspaceBytes);
    blocks.apply(
      cudaHandles,
      scratch,
      batchSize,
      maskBuf,
      maskSumBuf,
      mid.buf,
      midScratch.buf,
      workspaceBuf,
      workspaceBytes
    );
    normActConv2.apply(cudaHandles,scratch,batchSize,true,mid.buf,midScratch.buf,trunkBuf,maskBuf,workspaceBuf,workspaceBytes);
  }

};

//------------------------------------------------------------------------------

// Lightweight RMSNorm used inside transformer blocks (weight only, no bias, no spatial modes)
struct TransformerRMSNormLayer {
  const string name;
  const int numChannels;
  const float epsilon;
  const bool usingFP16;
  OwnedDeviceBuf weightBuf;
  OwnedDeviceBuf zeroBetaBuf;

  TransformerRMSNormLayer() = delete;
  TransformerRMSNormLayer(const TransformerRMSNormLayer&) = delete;
  TransformerRMSNormLayer& operator=(const TransformerRMSNormLayer&) = delete;

  TransformerRMSNormLayer(
    CudaHandles* cudaHandles,
    const TransformerRMSNormDesc* desc,
    bool useFP16
  ) :
    name(desc->name),
    numChannels(desc->numChannels),
    epsilon(desc->epsilon),
    usingFP16(useFP16)
  {
    (void)cudaHandles;
    testAssert((int)desc->weight.size() == numChannels);
    CudaUtils::mallocAndCopyToDevice(name, desc->weight, weightBuf.buf, useFP16);
    // Allocate a zero buffer for beta (TransformerRMSNorm has no bias)
    vector<float> zeros(numChannels, 0.0f);
    CudaUtils::mallocAndCopyToDevice(name + ":zeroBeta", zeros, zeroBetaBuf.buf, useFP16);
  }

  // Apply RMSNorm on NHWC data [N, XY, C], applying mask [N, XY] to zero padded positions.
  // Uses the RMSNormGammaBeta kernel with gamma=weight, beta=0, no activation.
  // If residualDeltaBuf is non-NULL, first performs inputBuf += residualDeltaBuf * mask in
  // place (a deferred masked residual add) fused into the same kernel. outputBuf may alias
  // residualDeltaBuf in that case.
  void apply(
    CudaHandles* cudaHandles,
    int batchSize,
    int xySize,
    void* inputBuf,
    void* outputBuf,
    const void* maskBuf,
    const void* residualDeltaBuf = NULL
  ) const {
    // RMSNormGammaBetaNHWC with gamma=weight, beta=zero, mask, identity activation.
    if(!usingFP16) {
      if(residualDeltaBuf != NULL)
        customCudaMaskedResidualAddRMSNormNHWC(
          (float*)inputBuf, (const float*)residualDeltaBuf, (float*)outputBuf,
          (const float*)weightBuf.buf, (const float*)zeroBetaBuf.buf,
          (const float*)maskBuf,
          batchSize, xySize, numChannels, epsilon, ACTIVATION_IDENTITY,cudaHandles->stream);
      else
        customCudaRMSNormGammaBetaNHWC(
          (const float*)inputBuf, (float*)outputBuf,
          (const float*)weightBuf.buf, (const float*)zeroBetaBuf.buf,
          (const float*)maskBuf,
          batchSize, xySize, numChannels, epsilon, ACTIVATION_IDENTITY,cudaHandles->stream);
    }
    else {
      if(residualDeltaBuf != NULL)
        customCudaMaskedResidualAddRMSNormNHWC(
          (half*)inputBuf, (const half*)residualDeltaBuf, (half*)outputBuf,
          (const half*)weightBuf.buf, (const half*)zeroBetaBuf.buf,
          (const half*)maskBuf,
          batchSize, xySize, numChannels, epsilon, ACTIVATION_IDENTITY,cudaHandles->stream);
      else
        customCudaRMSNormGammaBetaNHWC(
          (const half*)inputBuf, (half*)outputBuf,
          (const half*)weightBuf.buf, (const half*)zeroBetaBuf.buf,
          (const half*)maskBuf,
          batchSize, xySize, numChannels, epsilon, ACTIVATION_IDENTITY,cudaHandles->stream);
    }
    CUDA_ERR(name.c_str(), cudaPeekAtLastError());
  }
};

//------------------------------------------------------------------------------

struct RMSNormLayer {
  const string name;
  const int numChannels;
  const bool spatial;
  const int activation;
  const float epsilon;
  const int nnXLen;
  const int nnYLen;
  const bool usingFP16;
  const bool usingNHWC;

  OwnedDeviceBuf gammaBuf;
  OwnedDeviceBuf betaBuf;

  RMSNormLayer() = delete;
  RMSNormLayer(const RMSNormLayer&) = delete;
  RMSNormLayer& operator=(const RMSNormLayer&) = delete;

  RMSNormLayer(
    CudaHandles* cudaHandles,
    const RMSNormLayerDesc* desc,
    int act,
    int nnX,
    int nnY,
    bool useFP16,
    bool useNHWC
  ) :
    name(desc->name),
    numChannels(desc->numChannels),
    spatial(desc->spatial),
    activation(act),
    epsilon(desc->epsilon),
    nnXLen(nnX),
    nnYLen(nnY),
    usingFP16(useFP16),
    usingNHWC(useNHWC)
  {
    (void)cudaHandles;
    testAssert((int)desc->gamma.size() == numChannels);
    testAssert((int)desc->beta.size() == numChannels);
    // The device kernels apply only RELU/MISH/SILU explicitly and treat anything else as
    // identity; guard here so an unsupported kind (e.g. MISH_SCALE8, which applyScale8 can
    // produce for non-transformer nets) fails loudly instead of silently skipping activation.
    if(activation != ACTIVATION_IDENTITY && activation != ACTIVATION_RELU &&
       activation != ACTIVATION_MISH && activation != ACTIVATION_SILU)
      throw StringError(name + ": RMSNorm layer unsupported activation: " + Global::intToString(activation));
    CudaUtils::mallocAndCopyToDevice(name, desc->gamma, gammaBuf.buf, useFP16);
    CudaUtils::mallocAndCopyToDevice(name, desc->beta, betaBuf.buf, useFP16);
  }

  void apply(
    CudaHandles* cudaHandles,
    ScratchBuffers* scratch,
    int batchSize,
    void* inputBuf,
    void* outputBuf,
    const void* maskBuf,
    const float* maskSumBuf
  ) const {
    int xySize = nnXLen * nnYLen;
    if(!spatial) {
      if(!usingFP16) {
        if(!usingNHWC)
          customCudaRMSNormGammaBetaNCHW(
            (const float*)inputBuf, (float*)outputBuf, (const float*)gammaBuf.buf, (const float*)betaBuf.buf,
            (const float*)maskBuf, batchSize, numChannels, xySize, epsilon, activation,cudaHandles->stream);
        else
          customCudaRMSNormGammaBetaNHWC(
            (const float*)inputBuf, (float*)outputBuf, (const float*)gammaBuf.buf, (const float*)betaBuf.buf,
            (const float*)maskBuf, batchSize, xySize, numChannels, epsilon, activation,cudaHandles->stream);
      }
      else {
        if(!usingNHWC)
          customCudaRMSNormGammaBetaNCHW(
            (const half*)inputBuf, (half*)outputBuf, (const half*)gammaBuf.buf, (const half*)betaBuf.buf,
            (const half*)maskBuf, batchSize, numChannels, xySize, epsilon, activation,cudaHandles->stream);
        else
          customCudaRMSNormGammaBetaNHWC(
            (const half*)inputBuf, (half*)outputBuf, (const half*)gammaBuf.buf, (const half*)betaBuf.buf,
            (const half*)maskBuf, batchSize, xySize, numChannels, epsilon, activation,cudaHandles->stream);
      }
    }
    else {
      // Allocate temp buffer for spatial reduction from scratch (float regardless of FP16 mode).
      // Holds per-block partial sums plus the final reduced value per batch element; see
      // SPATIAL_RMSNORM_BLOCKS_PER_BATCH in cudaandrocmhelpers.inc (partialStride = that + 1).
      SizedBuf<void*> sumSqBuf(scratch->allocator, scratch->getBufSizeFloat(CUDA_SPATIAL_RMSNORM_SUMSQ_STRIDE));
      if(!usingFP16) {
        if(!usingNHWC)
          customCudaSpatialRMSNormNCHW(
            (const float*)inputBuf, (float*)outputBuf, (const float*)gammaBuf.buf, (const float*)betaBuf.buf,
            (const float*)maskBuf, maskSumBuf, batchSize, numChannels, xySize, epsilon, activation, (float*)sumSqBuf.buf,cudaHandles->stream);
        else
          customCudaSpatialRMSNormNHWC(
            (const float*)inputBuf, (float*)outputBuf, (const float*)gammaBuf.buf, (const float*)betaBuf.buf,
            (const float*)maskBuf, maskSumBuf, batchSize, xySize, numChannels, epsilon, activation, (float*)sumSqBuf.buf,cudaHandles->stream);
      }
      else {
        if(!usingNHWC)
          customCudaSpatialRMSNormNCHW(
            (const half*)inputBuf, (half*)outputBuf, (const half*)gammaBuf.buf, (const half*)betaBuf.buf,
            (const half*)maskBuf, maskSumBuf, batchSize, numChannels, xySize, epsilon, activation, (float*)sumSqBuf.buf,cudaHandles->stream);
        else
          customCudaSpatialRMSNormNHWC(
            (const half*)inputBuf, (half*)outputBuf, (const half*)gammaBuf.buf, (const half*)betaBuf.buf,
            (const half*)maskBuf, maskSumBuf, batchSize, xySize, numChannels, epsilon, activation, (float*)sumSqBuf.buf,cudaHandles->stream);
      }
    }
    CUDA_ERR(name.c_str(), cudaPeekAtLastError());
  }
};

//------------------------------------------------------------------------------

struct TransformerAttentionBlock {
  const string name;
  const int numHeads;
  const int numKVHeads;
  const int qHeadDim;
  const int vHeadDim;
  const bool useRope;
  const bool learnableRope;
  const int inChannels;

  const int nnXLen;
  const int nnYLen;
  const bool usingFP16;
  const bool usingNHWC;

  const TransformerRMSNormLayer preLN;
  const MatMulLayer outProj;

  // When the Q, K, and V projections all share one shape (numKVHeads == numHeads and
  // qHeadDim == vHeadDim), their weights are packed contiguously in qkvPackedWeights and the
  // three projections run as a single strided-batched GEMM sharing the input. Otherwise the
  // three separate projection layers below are constructed and applied individually.
  const bool sameQKVShapes;
  // When the mma attention path will consume Q/K/V (FP16, mma-supported shapes), the three
  // projections instead run as ONE combined GEMM writing an interleaved [tokens, q|k|v]
  // buffer that downstream kernels read via strides. qkvPackedWeights then holds the
  // combined interleaved layout (SharedWeightsKind::QKV_COMBINED) instead of the
  // strided-batched slices.
  const bool useCombinedQKV;
  SharedDeviceBuf qkvPackedWeights;
  std::unique_ptr<MatMulLayer> qProj;
  std::unique_ptr<MatMulLayer> kProj;
  std::unique_ptr<MatMulLayer> vProj;

  // Fixed RoPE: precomputed cos/sin tables on device (NULL for learnable RoPE).
  // Learnable RoPE: per-head frequencies on device (ropeFreqsBuf, FP32), cos/sin recomputed in-kernel.
  OwnedDeviceBuf ropeCosTable;
  OwnedDeviceBuf ropeSinTable;
  OwnedDeviceBuf ropeFreqsBuf;
  int ropeNumPairs;
  int ropeNumKVHeads;

  static bool shouldCombineQKV(CudaHandles* cudaHandles, const TransformerAttentionDesc* desc, bool useFP16) {
#if defined(DEBUG_INTERMEDIATE_VALUES) || !defined(KATAGO_GPU_CUDA)
    // The intermediate-value debug prints assume packed per-tensor layouts, so keep them
    // meaningful by never using the interleaved combined layout in debug-print builds.
    // HIP has no combined-QKV path at all.
    (void)cudaHandles; (void)desc; (void)useFP16;
    return false;
#else
    return
      useFP16 &&
      cudaHandles->mmaAttentionEnabled &&
      cudaHandles->combineQKV &&
      desc->qProj.inChannels == desc->kProj.inChannels &&
      desc->qProj.inChannels == desc->vProj.inChannels &&
      customCudaFlashAttentionMmaSupportsShape(desc->numHeads, desc->numKVHeads, desc->qHeadDim, desc->vHeadDim);
#endif
  }

  TransformerAttentionBlock() = delete;
  TransformerAttentionBlock(const TransformerAttentionBlock&) = delete;
  TransformerAttentionBlock& operator=(const TransformerAttentionBlock&) = delete;

  TransformerAttentionBlock(
    CudaHandles* cudaHandles,
    const TransformerAttentionDesc* desc,
    int nnX,
    int nnY,
    bool useFP16,
    bool useNHWC
  ) :
    name(desc->name),
    numHeads(desc->numHeads),
    numKVHeads(desc->numKVHeads),
    qHeadDim(desc->qHeadDim),
    vHeadDim(desc->vHeadDim),
    useRope(desc->useRope),
    learnableRope(desc->learnableRope),
    inChannels(desc->qProj.inChannels),
    nnXLen(nnX),
    nnYLen(nnY),
    usingFP16(useFP16),
    usingNHWC(useNHWC),
    preLN(cudaHandles, &desc->preLN, useFP16),
    outProj(cudaHandles, &desc->outProj, useFP16),
    sameQKVShapes(
      desc->qProj.inChannels == desc->kProj.inChannels &&
      desc->qProj.inChannels == desc->vProj.inChannels &&
      desc->qProj.outChannels == desc->kProj.outChannels &&
      desc->qProj.outChannels == desc->vProj.outChannels
    ),
    useCombinedQKV(shouldCombineQKV(cudaHandles, desc, useFP16)),
    ropeNumPairs(0),
    ropeNumKVHeads(0)
  {
    if(!useNHWC) {
      throw StringError("Transformer blocks with NCHW layout are not yet supported by the " KATAGO_GPU_BACKEND_NAME " backend");
    }
    // The Q, K, and V slices of the packed projection buffer must be 16-byte aligned for the
    // attention implementations, which holds whenever each projection's total channel count is
    // a multiple of 8. Enforced rather than handled: no real model comes close to violating it
    // (head dims are 32+), and the python side rejects such configs at model creation too.
    if(numHeads * qHeadDim % 8 != 0 || numKVHeads * qHeadDim % 8 != 0 || numKVHeads * vHeadDim % 8 != 0)
      throw StringError(
        name + ": " KATAGO_GPU_BACKEND_NAME
        " backend requires each of the attention Q, K, and V projections to have a total"
        " channel count (numHeads * headDim) that is a multiple of 8, got " +
        Global::intToString(numHeads * qHeadDim) + ", " + Global::intToString(numKVHeads * qHeadDim) +
        ", " + Global::intToString(numKVHeads * vHeadDim));
    if(useCombinedQKV) {
#ifdef KATAGO_GPU_CUDA
      if(!cudaHandles->loggedUsingCombinedQKV) {
        cudaHandles->loggedUsingCombinedQKV = true;
        if(cudaHandles->logger != NULL)
          cudaHandles->logger->write("Cuda backend: using combined QKV projection for attention");
      }
#endif
      // Interleave per input channel: the q, k, then v output weights of each input channel,
      // giving in-major [inChannels, qTotal+kTotal+vTotal] for one combined GEMM.
      const int qTot = numHeads * qHeadDim;
      const int kTot = numKVHeads * qHeadDim;
      const int vTot = numKVHeads * vHeadDim;
      const size_t combinedN = (size_t)qTot + kTot + vTot;
      testAssert(desc->qProj.weights.size() == (size_t)inChannels * qTot);
      testAssert(desc->kProj.weights.size() == (size_t)inChannels * kTot);
      testAssert(desc->vProj.weights.size() == (size_t)inChannels * vTot);
      vector<float> packed((size_t)inChannels * combinedN);
      for(int i = 0; i < inChannels; i++) {
        std::copy(
          desc->qProj.weights.begin() + (size_t)i * qTot, desc->qProj.weights.begin() + (size_t)(i+1) * qTot,
          packed.begin() + i * combinedN);
        std::copy(
          desc->kProj.weights.begin() + (size_t)i * kTot, desc->kProj.weights.begin() + (size_t)(i+1) * kTot,
          packed.begin() + i * combinedN + qTot);
        std::copy(
          desc->vProj.weights.begin() + (size_t)i * vTot, desc->vProj.weights.begin() + (size_t)(i+1) * vTot,
          packed.begin() + i * combinedN + qTot + kTot);
      }
      qkvPackedWeights = mallocAndCopyToDeviceMaybeShared(
        name + ":qkvCombined",
        [&](void*& b){ CudaUtils::mallocAndCopyToDevice(name + ":qkvCombined", packed, b, useFP16); },
        desc, SharedWeightsKind::QKV_COMBINED, useFP16, cudaHandles->shareModelWeights);
    }
    else if(sameQKVShapes) {
      vector<float> packed;
      packed.reserve(desc->qProj.weights.size() * 3);
      packed.insert(packed.end(), desc->qProj.weights.begin(), desc->qProj.weights.end());
      packed.insert(packed.end(), desc->kProj.weights.begin(), desc->kProj.weights.end());
      packed.insert(packed.end(), desc->vProj.weights.begin(), desc->vProj.weights.end());
      qkvPackedWeights = mallocAndCopyToDeviceMaybeShared(
        name + ":qkvPacked",
        [&](void*& b){ CudaUtils::mallocAndCopyToDevice(name + ":qkvPacked", packed, b, useFP16); },
        desc, SharedWeightsKind::QKV_PACKED, useFP16, cudaHandles->shareModelWeights);
    }
    else {
      qProj = std::make_unique<MatMulLayer>(cudaHandles, &desc->qProj, useFP16);
      kProj = std::make_unique<MatMulLayer>(cudaHandles, &desc->kProj, useFP16);
      vProj = std::make_unique<MatMulLayer>(cudaHandles, &desc->vProj, useFP16);
    }
    if(useRope) {
      ropeNumPairs = qHeadDim / 2;
      ropeNumKVHeads = numKVHeads;
      if(learnableRope) {
        // Table-free: upload the tiny per-head frequencies (FP32) and recompute cos/sin in-kernel.
        // Avoids the numKVHeads-times-larger cos/sin table that otherwise spills L2 (see kernel comment).
        testAssert(desc->ropeFreqs.size() == (size_t)(numKVHeads * ropeNumPairs * 2));
        CudaUtils::mallocAndCopyToDevice(name + ":ropeFreqs", desc->ropeFreqs.data(), (int)desc->ropeFreqs.size(), ropeFreqsBuf.buf, false);
      }
      else {
        int seqLen = nnXLen * nnYLen;
        vector<float> cosTableData;
        vector<float> sinTableData;
        desc->computeRopeCosSin(nnXLen, nnYLen, seqLen, cosTableData, sinTableData);
        CudaUtils::mallocAndCopyToDevice(name + ":ropeCos", cosTableData.data(), (int)cosTableData.size(), ropeCosTable.buf, useFP16);
        CudaUtils::mallocAndCopyToDevice(name + ":ropeSin", sinTableData.data(), (int)sinTableData.size(), ropeSinTable.buf, useFP16);
      }
    }
  }

  size_t requiredWorkspaceBytes(
    CudaHandles* cudaHandles,
    int batchSize
  ) const {
    (void)cudaHandles;
    (void)batchSize;
    return 0;
  }

  // pendingTrunkDelta, if non-NULL, carries a deferred masked residual delta for trunkBuf
  // between consecutive transformer blocks (see BlockStack::apply): a non-NULL *pendingTrunkDelta
  // on entry is added into trunkBuf fused with the pre-norm, and on exit this block's own
  // masked residual delta is left pending in *pendingTrunkDelta instead of being applied.
  void apply(
    CudaHandles* cudaHandles,
    ScratchBuffers* scratch,
    int batchSize,
    void* trunkBuf,
    void* trunkScratchBuf,
    void* maskBuf,
    float* maskSumBuf,
    void* workspaceBuf,
    size_t workspaceBytes,
    void** pendingTrunkDelta = NULL
  ) const {
    (void)maskSumBuf;

    int seqLen = nnXLen * nnYLen;
    int qTotalDim = numHeads * qHeadDim;
    int kTotalDim = numKVHeads * qHeadDim;
    int vTotalDim = numKVHeads * vHeadDim;

    // NHWC: trunk is [N, XY, C]. RMSNorm + mask zeroing, with any deferred residual add fused in.
    const void* pendingDeltaIn = (pendingTrunkDelta != NULL) ? *pendingTrunkDelta : NULL;
    preLN.apply(cudaHandles, batchSize, seqLen, trunkBuf, trunkScratchBuf, maskBuf, pendingDeltaIn);
    if(pendingTrunkDelta != NULL)
      *pendingTrunkDelta = NULL;

#ifdef DEBUG_INTERMEDIATE_VALUES
    CudaUtils::debugPrint3D(KATAGO_GPU_BACKEND_NAME " Attn RMSNorm out", trunkScratchBuf, batchSize, inChannels, seqLen, usingNHWC, usingFP16, maskBuf);
#endif

    // Step 2: Q/K/V projections
    // trunkScratchBuf is [N, XY, C] NHWC = [C, N*seqLen] column-major.
    // MatMulLayer expects input as [inChannels, batchSize], which matches.
    int matBatchSize = batchSize * seqLen;

    //Every transient buffer here is sized by max batch size rather than the actual batch size.
    //SimpleAllocator pools by exact byte size and never frees, so sizing by the actual batch size
    //would retain a separate allocation for every distinct batch size ever seen, which costs
    //O(maxBatchSize) times more memory than reusing one max-sized allocation.
    SizedBuf<void*> qkvBuf(scratch->allocator, scratch->getBufSizeXY(qTotalDim + kTotalDim + vTotalDim));
    void* qPtr;
    void* kPtr;
    void* vPtr;
    // Per-token row strides (in elements) of the Q slice and of the K/V slices, consumed by
    // the RoPE and mma attention kernels.
    int qStrideElts;
    int kvStrideElts;
    if(useCombinedQKV) {
      // One combined GEMM writes an interleaved [tokens, q|k|v] buffer, and Q/K/V are interior
      // column slices read via the combined row stride. FP16-only by construction.
      const int combinedN = qTotalDim + kTotalDim + vTotalDim;
      qPtr = qkvBuf.buf;
      kPtr = (char*)qkvBuf.buf + (size_t)qTotalDim * sizeof(half);
      vPtr = (char*)qkvBuf.buf + (size_t)(qTotalDim + kTotalDim) * sizeof(half);
      qStrideElts = combinedN;
      kvStrideElts = combinedN;
      // The mma kernel wants 16-byte-aligned slice pointers, so the interior offsets must be
      // multiples of 8 halfs. True for every real head configuration, so assert.
      testAssert(qTotalDim % 8 == 0 && kTotalDim % 8 == 0);
      applySharedInputStridedMatMuls(
        cudaHandles, scratch, usingFP16, name,
        combinedN, matBatchSize, inChannels,
        qkvPackedWeights.buf, trunkScratchBuf, qkvBuf.buf, 0, 1
      );
    }
    else {
      qPtr = qkvBuf.buf;
      kPtr = (char*)qkvBuf.buf + scratch->getBufSizeXY(qTotalDim);
      vPtr = (char*)qkvBuf.buf + scratch->getBufSizeXY(qTotalDim + kTotalDim);
      qStrideElts = qTotalDim;
      kvStrideElts = kTotalDim;
      // The attention implementations (cudnn SDPA, CK FMHA, the mma kernel) want 16-byte-aligned
      // tensors, which these interior pointers only get if the sub-buffer offsets cooperate.
      // Guaranteed by the constructor's multiple-of-8 projection-width check.
      testAssert(scratch->getBufSizeXY(qTotalDim) % 16 == 0);
      testAssert(scratch->getBufSizeXY(qTotalDim + kTotalDim) % 16 == 0);

      if(sameQKVShapes) {
        const size_t eltBytes = usingFP16 ? sizeof(cublas_half_t) : sizeof(float);
        const long long strideElts = (long long)(scratch->getBufSizeXY(qTotalDim) / eltBytes);
        applySharedInputStridedMatMuls(
          cudaHandles, scratch, usingFP16, name,
          qTotalDim, matBatchSize, inChannels,
          qkvPackedWeights.buf, trunkScratchBuf, qPtr, strideElts, 3
        );
      }
      else {
        qProj->apply(cudaHandles, scratch, matBatchSize, trunkScratchBuf, qPtr, workspaceBuf, workspaceBytes);
        kProj->apply(cudaHandles, scratch, matBatchSize, trunkScratchBuf, kPtr, workspaceBuf, workspaceBytes);
        vProj->apply(cudaHandles, scratch, matBatchSize, trunkScratchBuf, vPtr, workspaceBuf, workspaceBytes);
      }
    }

#ifdef DEBUG_INTERMEDIATE_VALUES
    CudaUtils::debugPrint2D(KATAGO_GPU_BACKEND_NAME " Attn Q", qPtr, matBatchSize, qTotalDim, usingFP16);
#endif

    // Step 3: Apply RoPE to Q and K
    // Q is [qTotalDim, seqLen*batchSize] column-major = [batchSize*seqLen, qTotalDim] row-major
    if(useRope) {
      // The fused Q+K kernels assume each KV head maps to a whole number of query heads.
      // Models always satisfy this, but fall back to separate launches if not.
      bool fuseQK = (numHeads % numKVHeads == 0);
      // The combined interleaved layout is only correct through the stride-aware fused kernels,
      // since the separate-launch fallback hardcodes packed row strides. Currently implied because
      // the combined layout requires an mma-supported shape, which requires divisibility, but
      // assert so that widening mma shape support cannot silently break this.
      testAssert(fuseQK || !useCombinedQKV);
      if(learnableRope) {
        // Table-free recompute path (cos/sin computed in-kernel from ropeFreqsBuf).
        if(!usingFP16) {
          if(fuseQK)
            customCudaApplyRoPEQKLearnableRecompute((float*)qPtr, (float*)kPtr, (const float*)ropeFreqsBuf.buf,
              batchSize, seqLen, numHeads, numKVHeads, qHeadDim, qStrideElts, kvStrideElts, ropeNumPairs, nnXLen,cudaHandles->stream);
          else {
            customCudaApplyRoPELearnableRecompute((float*)qPtr, (const float*)ropeFreqsBuf.buf,
              batchSize, seqLen, numHeads, numKVHeads, qHeadDim, ropeNumPairs, nnXLen,cudaHandles->stream);
            customCudaApplyRoPELearnableRecompute((float*)kPtr, (const float*)ropeFreqsBuf.buf,
              batchSize, seqLen, numKVHeads, numKVHeads, qHeadDim, ropeNumPairs, nnXLen,cudaHandles->stream);
          }
        }
        else {
          if(fuseQK)
            customCudaApplyRoPEQKLearnableRecompute((half*)qPtr, (half*)kPtr, (const float*)ropeFreqsBuf.buf,
              batchSize, seqLen, numHeads, numKVHeads, qHeadDim, qStrideElts, kvStrideElts, ropeNumPairs, nnXLen,cudaHandles->stream);
          else {
            customCudaApplyRoPELearnableRecompute((half*)qPtr, (const float*)ropeFreqsBuf.buf,
              batchSize, seqLen, numHeads, numKVHeads, qHeadDim, ropeNumPairs, nnXLen,cudaHandles->stream);
            customCudaApplyRoPELearnableRecompute((half*)kPtr, (const float*)ropeFreqsBuf.buf,
              batchSize, seqLen, numKVHeads, numKVHeads, qHeadDim, ropeNumPairs, nnXLen,cudaHandles->stream);
          }
        }
      }
      else {
        if(!usingFP16) {
          if(fuseQK)
            customCudaApplyRoPEQK((float*)qPtr, (float*)kPtr, (const float*)ropeCosTable.buf, (const float*)ropeSinTable.buf,
              batchSize, seqLen, numHeads, numKVHeads, qHeadDim, qStrideElts, kvStrideElts, ropeNumPairs, learnableRope,cudaHandles->stream);
          else {
            customCudaApplyRoPE((float*)qPtr, (const float*)ropeCosTable.buf, (const float*)ropeSinTable.buf,
              batchSize, seqLen, numHeads, numKVHeads, qHeadDim, ropeNumPairs, learnableRope,cudaHandles->stream);
            customCudaApplyRoPE((float*)kPtr, (const float*)ropeCosTable.buf, (const float*)ropeSinTable.buf,
              batchSize, seqLen, numKVHeads, numKVHeads, qHeadDim, ropeNumPairs, learnableRope,cudaHandles->stream);
          }
        }
        else {
          if(fuseQK)
            customCudaApplyRoPEQK((half*)qPtr, (half*)kPtr, (const half*)ropeCosTable.buf, (const half*)ropeSinTable.buf,
              batchSize, seqLen, numHeads, numKVHeads, qHeadDim, qStrideElts, kvStrideElts, ropeNumPairs, learnableRope,cudaHandles->stream);
          else {
            customCudaApplyRoPE((half*)qPtr, (const half*)ropeCosTable.buf, (const half*)ropeSinTable.buf,
              batchSize, seqLen, numHeads, numKVHeads, qHeadDim, ropeNumPairs, learnableRope,cudaHandles->stream);
            customCudaApplyRoPE((half*)kPtr, (const half*)ropeCosTable.buf, (const half*)ropeSinTable.buf,
              batchSize, seqLen, numKVHeads, numKVHeads, qHeadDim, ropeNumPairs, learnableRope,cudaHandles->stream);
          }
        }
      }
      CUDA_ERR(name.c_str(), cudaPeekAtLastError());
    }

#ifdef KATAGO_GPU_CUDA
    // Step 4: Scaled dot-product attention.
    // Preference order: our tensor-core mma flash kernel (sm_80+, FP16, head dims 32/64), then
    // cudnn SDPA (FP16 + cudnn >= 8.9.3 + supported GPU), then a scalar online-softmax kernel.
    // All paths consume Q/K/V in BSHD layout and produce attnOut in the same layout as expected by outProj:
    //   attnOut: [numHeads*vHeadDim, seqLen*batchSize] col-major = [batchSize*seqLen, numHeads*vHeadDim] row-major.

    SizedBuf<void*> attnOutBuf(scratch->allocator, scratch->getBufSizeXY(numHeads * vHeadDim));

    bool usedSDPA = false;
    // The mma kernel consumes the raw per-position mask directly, so unlike cudnn SDPA the
    // masked path needs no materialized [B,S,S] attention bias at all.
    if(usingFP16 && cudaHandles->mmaAttentionEnabled) {
      usedSDPA = customCudaFlashAttentionMma(
        (const half*)qPtr, (const half*)kPtr, (const half*)vPtr, (const half*)maskBuf, (half*)attnOutBuf.buf,
        batchSize, seqLen, numHeads, numKVHeads, qHeadDim, vHeadDim,
        qStrideElts, kvStrideElts, cudaHandles->stream);
      if(usedSDPA) {
        CUDA_ERR(name.c_str(), cudaPeekAtLastError());
        if(!cudaHandles->loggedUsingMmaAttention) {
          cudaHandles->loggedUsingMmaAttention = true;
          if(cudaHandles->logger != NULL)
            cudaHandles->logger->write("Cuda backend: using tensor-core mma flash attention kernel");
        }
      }
      else if(!cudaHandles->loggedMmaAttentionRejectedShape) {
        cudaHandles->loggedMmaAttentionRejectedShape = true;
        if(cudaHandles->logger != NULL)
          cudaHandles->logger->write(
            "Cuda backend: mma flash attention kernel rejected this launch (heads " +
            Global::intToString(numHeads) + "/" + Global::intToString(numKVHeads) +
            " head dims " + Global::intToString(qHeadDim) + "/" + Global::intToString(vHeadDim) +
            " batch " + Global::intToString(batchSize) +
            "), unsupported shape or launch limits, using slower fallback attention paths");
      }
    }
    // The fallback attention paths below assume the packed (non-interleaved) Q/K/V layout.
    // useCombinedQKV was gated at construction on the mma kernel supporting this shape, so
    // the only reachable rejection here is a launch limit (batch size over 65535), and anything
    // else means an invariant broke.
    if(!usedSDPA && useCombinedQKV)
      throw StringError(
        name + ": combined QKV layout requires the mma attention kernel, which rejected this launch (batch " +
        Global::intToString(batchSize) + "); a batch size over 65535 is unsupported, anything else is a bug");
#if KATAGO_CUDA_HAS_SDPA
    SDPAGraphCache* sdpaCache = cudaHandles->sdpaCache.get();
    // Report once if cudaDisableGraphSDPA is the only reason we are not taking the cudnn graph SDPA path,
    // i.e. FP16 and a still-enabled cache are available so SDPA would otherwise have been used.
    if(!usedSDPA && usingFP16 && sdpaCache != NULL && sdpaCache->sdpaSupported && cudaHandles->cudaDisableGraphSDPA && !cudaHandles->loggedGraphSDPADisabled) {
      cudaHandles->loggedGraphSDPADisabled = true;
      if(cudaHandles->logger != NULL)
        cudaHandles->logger->write(
          "Cuda backend: cudaDisableGraphSDPA is set, using the custom attention kernel instead of the cudnn graph SDPA path that would otherwise have been used");
    }
    if(!usedSDPA && usingFP16 && sdpaCache != NULL && !cudaHandles->cudaDisableGraphSDPA) {
      bool hasMask = (maskBuf != NULL);
      SDPAGraphKey sdpaKey = {numHeads, numKVHeads, qHeadDim, vHeadDim, seqLen, batchSize, hasMask, usingFP16};
      auto plan = sdpaCache->getOrBuildPlan(cudaHandles->cudnn, sdpaKey, cudaHandles->logger);
      if(plan != nullptr) {
        std::unordered_map<int64_t, void*> variant_pack = {
          {SDPAPlanForBatchSize::Q_UID, qPtr},
          {SDPAPlanForBatchSize::K_UID, kPtr},
          {SDPAPlanForBatchSize::V_UID, vPtr},
          {SDPAPlanForBatchSize::O_UID, attnOutBuf.buf},
        };

        // When a mask is present, use a [B, 1, S, S] additive bias: bias[b,q,k] = (mask[b,k] != 0 ? 0 : -3e4).
        // The bias only depends on the mask, so it is built once per forward pass and shared by
        // all attention blocks (see ScratchBuffers::attnBiasBuf).
        if(hasMask) {
          if(!scratch->attnBiasBuiltThisForward) {
            if(scratch->attnBiasBuf == nullptr)
              scratch->attnBiasBuf = std::make_unique<SizedBuf<void*>>(scratch->allocator, scratch->getBufSizeXY(seqLen));
            customCudaMaskToAttnBiasFull((const half*)maskBuf, (half*)scratch->attnBiasBuf->buf, batchSize, seqLen,cudaHandles->stream);
            CUDA_ERR(name.c_str(),cudaPeekAtLastError());
            scratch->attnBiasBuiltThisForward = true;
          }
          variant_pack[SDPAPlanForBatchSize::BIAS_UID] = scratch->attnBiasBuf->buf;
        }

        // Workspace from cudnn (separate from the conv workspace - different shape and lifetime).
        SizedBuf<void*> sdpaWs(scratch->allocator, (size_t)plan->workspaceBytes);

        auto status = plan->graph->execute(cudaHandles->cudnn, variant_pack, sdpaWs.buf);
        if(status.is_bad()) {
          string reason = string("cudnn SDPA execute failed: ") + status.get_message();
          // During warmup we tolerate this: disable SDPA from here on and fall through to the custom
          // kernel. Outside of warmup a failure here is fatal - the plan was already validated and
          // built, so an execute failure means something is genuinely wrong.
          if(!cudaHandles->isWarmup)
            throw StringError(reason);
          sdpaCache->sdpaSupported = false;
          sdpaCache->disableReason = reason;
          if(cudaHandles->logger != NULL)
            cudaHandles->logger->write("Cuda backend: disabling cudnn SDPA and falling back to custom attention kernel: " + reason);
        }
        else {
          usedSDPA = true;
          if(!cudaHandles->loggedUsingGraphSDPA) {
            cudaHandles->loggedUsingGraphSDPA = true;
            if(cudaHandles->logger != NULL)
              cudaHandles->logger->write("Cuda backend: using cudnn graph SDPA attention");
          }
        }
      }
    }
#endif
#else  // KATAGO_GPU_HIP
    // Step 4: Scaled dot-product attention. When enabled and available, try the CK FMHA fused path
    // first (only supports FP16, matching the CUDA backend's cudnn-frontend SDPA path, which is
    // also FP16-only). Otherwise, or if CK reports the shape/traits as unsupported at runtime,
    // fall back to the plain online-softmax kernel, which is fully adequate at KataGo's sequence
    // lengths.
    SizedBuf<void*> attnOutBuf(scratch->allocator, scratch->getBufSizeXY(numHeads * vHeadDim));

    bool usedSDPA = false;
#if KATAGO_ROCM_HAS_CK_FMHA && KATAGO_ROCM_CK_FMHA_ARCH_OK
    if(usingFP16 && !cudaHandles->disableFusedAttention &&
       cudaHandles->ckFmhaArchSupported && !cudaHandles->fusedAttentionRuntimeDisabled) {
      bool hasMask = (maskBuf != NULL);

      // CK's fused path takes a pre-materialized additive bias rather than a raw per-position
      // mask, so build a [B, S, S] bias broadcast over heads, matching what the CUDA backend does
      // for its cudnn-frontend graph SDPA path. Built once per forward and shared by all
      // attention blocks (see ScratchBuffers::attnBiasBuf).
      if(hasMask && !scratch->attnBiasBuiltThisForward) {
        if(scratch->attnBiasBuf == nullptr)
          scratch->attnBiasBuf = std::make_unique<SizedBuf<void*>>(scratch->allocator, scratch->getBufSizeXY(seqLen));
        customCudaMaskToAttnBiasFull((const half*)maskBuf, (half*)scratch->attnBiasBuf->buf, batchSize, seqLen,cudaHandles->stream);
        CUDA_ERR(name.c_str(), hipPeekAtLastError());
        scratch->attnBiasBuiltThisForward = true;
      }

      fmha_fwd_traits traits;
      traits.hdim_q = qHeadDim;
      traits.hdim_v = vHeadDim;
      traits.data_type = "fp16";
      traits.is_group_mode = false;
      traits.is_v_rowmajor = true;
      traits.has_logits_soft_cap = false;
      traits.mask_type = mask_enum::no_mask;
      traits.bias_type = hasMask ? bias_enum::elementwise_bias : bias_enum::no_bias;
      traits.has_lse = false;
      traits.has_dropout = false;
      traits.qscale_type = quant_scale_enum::no_scale;
      traits.skip_min_seqlen_q = false;
      traits.has_sink = false;

      // Physical layout for Q/K/V/O is [N, S, H, D] (i_perm=false, o_perm=false), V is row-major
      // ([N, S, H, Dv], is_v_rowmajor=true) - all matching the BSHD buffers MatMulLayer produces.
      // Value-initialized so the fields not assigned below (e.g. the batch-mode cu_seqlen
      // pointers) are null/zero. Not memset: args contains a std::variant member, which raw
      // byte-zeroing may not legally overwrite.
      fmha_fwd_args args{};
      args.q_ptr = qPtr;
      args.k_ptr = kPtr;
      args.v_ptr = vPtr;
      args.bias_ptr = hasMask ? scratch->attnBiasBuf->buf : nullptr;
      args.o_ptr = attnOutBuf.buf;
      args.seqlen_q = seqLen;
      args.seqlen_k = seqLen;
      args.batch = batchSize;
      args.max_seqlen_q = seqLen;
      args.hdim_q = qHeadDim;
      args.hdim_v = vHeadDim;
      args.nhead_q = numHeads;
      args.nhead_k = numKVHeads;
      args.scale_s = 1.0f / sqrtf((float)qHeadDim);
      args.logits_soft_cap = 0.0f;
      args.stride_q = (ck_tile::index_t)numHeads * qHeadDim;
      args.stride_k = (ck_tile::index_t)numKVHeads * qHeadDim;
      args.stride_v = (ck_tile::index_t)numKVHeads * vHeadDim;
      args.stride_bias = hasMask ? seqLen : 0;
      args.stride_o = (ck_tile::index_t)numHeads * vHeadDim;
      args.nhead_stride_q = qHeadDim;
      args.nhead_stride_k = qHeadDim;
      args.nhead_stride_v = vHeadDim;
      args.nhead_stride_bias = 0; // broadcast the bias over heads
      args.nhead_stride_o = vHeadDim;
      args.batch_stride_q = (ck_tile::index_t)numHeads * seqLen * qHeadDim;
      args.batch_stride_k = (ck_tile::index_t)numKVHeads * seqLen * qHeadDim;
      args.batch_stride_v = (ck_tile::index_t)numKVHeads * seqLen * vHeadDim;
      args.batch_stride_bias = hasMask ? (ck_tile::index_t)seqLen * seqLen : 0;
      args.batch_stride_o = (ck_tile::index_t)numHeads * seqLen * vHeadDim;
      args.window_size_left = -1;
      args.window_size_right = -1;
      args.mask_type = static_cast<ck_tile::index_t>(mask_enum::no_mask);
      args.min_seqlen_q = 0;
      args.p_drop = 0.0f;
      args.s_randval = false;
      args.drop_seed_offset = std::pair<uint64_t, uint64_t>{0, 0};

      // fmha_fwd returns a negative value if no generated kernel supports this shape/config.
      // ck_tile can also throw (std::runtime_error) if a kernel launch itself fails - e.g. a
      // device that slipped past the arch gating with no compiled code object. Either way,
      // permanently fall back to the plain attention kernel for this handle so we don't pay for
      // the attempt (or the bias materialization above) again on every batch.
      float ckResult = -1.0f;
      string ckFailure;
      // Note on a speedup deliberately left on the table: because a bias is present here, CK's
      // dispatch selects its synchronous QRKSVS pipeline, not the asynchronous QRKSVSAsync one it
      // uses when there is no bias. The async pipeline supports bias perfectly well and is much
      // faster (on gfx942, 195 vs 239 us per call, worth roughly 8-12% whole-net on masked boards)
      // with no accuracy cost, but CK's kernel codegen never emits that combination, so taking it
      // means instantiating the kernel ourselves. That was tried and works, but the async pipeline
      // only compiles for CDNA - its LDS-direct loads do not exist on RDNA - so it would speed up
      // datacenter cards while adding a second place that tracks CK's internal pipeline names.
      // Revisit if CK starts emitting async+bias itself, or gains RDNA support.
      try {
        ckResult = fmha_fwd(traits, args, ck_tile::stream_config{cudaHandles->stream.stream});
      }
      catch(const std::exception& e) {
        ckFailure = e.what();
      }
      if(ckResult >= 0.0f) {
        usedSDPA = true;
        if(!cudaHandles->loggedUsingCkFmha) {
          cudaHandles->loggedUsingCkFmha = true;
          if(cudaHandles->logger != NULL)
            cudaHandles->logger->write("ROCm backend: using CK fused attention");
        }
      }
      else {
        cudaHandles->fusedAttentionRuntimeDisabled = true;
        if(cudaHandles->logger != NULL) {
          cudaHandles->logger->write(
            "ROCm backend: CK fused attention unavailable" +
            (ckFailure.empty() ? string(" (unsupported shape/config)") : (": " + ckFailure)) +
            "; using the plain attention kernel instead");
        }
      }
    }
    // Report once if rocmDisableFusedAttention is the only reason the CK path is not taken, so
    // that the knob can be distinguished from an arch or shape that never supported it anyway.
    if(usingFP16 && cudaHandles->disableFusedAttention && cudaHandles->ckFmhaArchSupported &&
       !cudaHandles->loggedCkFmhaDisabled) {
      cudaHandles->loggedCkFmhaDisabled = true;
      if(cudaHandles->logger != NULL)
        cudaHandles->logger->write(
          "ROCm backend: rocmDisableFusedAttention is set, using the plain attention kernel instead of the CK fused attention path that would otherwise have been used");
    }
#endif
#endif

    if(!usedSDPA) {
      if(!usingFP16) {
        customCudaFlashAttention(
          (const float*)qPtr, (const float*)kPtr, (const float*)vPtr,
          (const float*)maskBuf, (float*)attnOutBuf.buf,
          batchSize, seqLen, numHeads, numKVHeads, qHeadDim, vHeadDim,cudaHandles->stream);
      }
      else {
        customCudaFlashAttention(
          (const half*)qPtr, (const half*)kPtr, (const half*)vPtr,
          (const half*)maskBuf, (half*)attnOutBuf.buf,
          batchSize, seqLen, numHeads, numKVHeads, qHeadDim, vHeadDim,cudaHandles->stream);
      }
      CUDA_ERR(name.c_str(), cudaPeekAtLastError());
    }

    // Step 5 and 6: Output projection and residual addition.
    // attnOutBuf is [numHeads*vHeadDim, seqLen*batchSize] col-major
    // outProj maps to [inChannels, seqLen*batchSize]
    // With no mask the residual add folds into the projection GEMM (beta=1). With a mask the add
    // must stay a separate masked kernel: the attention outputs are nonzero at off-board query
    // positions (the bias masks only keys), so an unmasked beta=1 add would let off-board trunk
    // values grow block over block. Downstream consumers do re-zero off-board positions, but once
    // an off-board value overflows FP16 to infinity, the masking multiply produces inf * 0 = NaN,
    // which then propagates into on-board results. FP32 cannot realistically overflow but takes
    // the same masked path so both precisions behave identically.
    if(maskBuf == NULL) {
      outProj.apply(cudaHandles, scratch, matBatchSize, attnOutBuf.buf, trunkBuf, workspaceBuf, workspaceBytes, true);
      CUDA_ERR(name.c_str(), cudaPeekAtLastError());
    }
    else {
      outProj.apply(cudaHandles, scratch, matBatchSize, attnOutBuf.buf, trunkScratchBuf, workspaceBuf, workspaceBytes);

#ifdef DEBUG_INTERMEDIATE_VALUES
      CudaUtils::debugPrint3D(KATAGO_GPU_BACKEND_NAME " Attn outProj", trunkScratchBuf, batchSize, inChannels, seqLen, usingNHWC, usingFP16, maskBuf);
#endif

      if(pendingTrunkDelta != NULL) {
        // Defer the add so the next block can fuse it into its pre-norm.
        *pendingTrunkDelta = trunkScratchBuf;
      }
      // NHWC: trunk is [N, XY, C], mask is [N, XY]
      else if(!usingFP16) {
        customCudaMaskedResidualAddNHWC((float*)trunkBuf, (const float*)trunkScratchBuf, (const float*)maskBuf, batchSize, seqLen, inChannels,cudaHandles->stream);
      }
      else {
        customCudaMaskedResidualAddNHWC((half*)trunkBuf, (const half*)trunkScratchBuf, (const half*)maskBuf, batchSize, seqLen, inChannels,cudaHandles->stream);
      }
      CUDA_ERR(name.c_str(), cudaPeekAtLastError());
    }

#ifdef DEBUG_INTERMEDIATE_VALUES
    CudaUtils::debugPrint3D(KATAGO_GPU_BACKEND_NAME " Attn residual", trunkBuf, batchSize, inChannels, seqLen, usingNHWC, usingFP16, maskBuf);
#endif
  }
};

//------------------------------------------------------------------------------

struct TransformerFFNBlock {
  const string name;
  const int numChannels;
  const int ffnChannels;
  const bool useSwiGLU;

  const int nnXLen;
  const int nnYLen;
  const bool usingFP16;
  const bool usingNHWC;

  const TransformerRMSNormLayer preLN;
  const MatMulLayer linear2;

  // linear1 and the gate share the same shape and input, so their weights are packed
  // contiguously: in-major for the unfused strided-batched GEMM path, or out-major
  // (transposed) when the CUTLASS fused kernel is used. Whether the fused kernel runs is
  // decided once at construction, since its shape support depends only on dimensions and
  // alignment, so only one layout is ever uploaded and the fused path never falls back at
  // apply time.
  SharedDeviceBuf ffnPackedWeights;
#if defined(KATAGO_GPU_CUDA) && defined(USE_CUTLASS_FUSED_FFN)
  bool usingFusedFFN = false;
#endif

  TransformerFFNBlock() = delete;
  TransformerFFNBlock(const TransformerFFNBlock&) = delete;
  TransformerFFNBlock& operator=(const TransformerFFNBlock&) = delete;

  TransformerFFNBlock(
    CudaHandles* cudaHandles,
    const TransformerFFNDesc* desc,
    int nnX,
    int nnY,
    bool useFP16,
    bool useNHWC
  ) :
    name(desc->name),
    numChannels(desc->numChannels),
    ffnChannels(desc->ffnChannels),
    useSwiGLU(desc->useSwiGLU),
    nnXLen(nnX),
    nnYLen(nnY),
    usingFP16(useFP16),
    usingNHWC(useNHWC),
    preLN(cudaHandles, &desc->preLN, useFP16),
    linear2(cudaHandles, &desc->linear2, useFP16)
  {
    if(!useSwiGLU) {
      throw StringError("Non-SwiGLU transformer FFN is not yet supported in " KATAGO_GPU_BACKEND_NAME " backend");
    }
    if(!useNHWC) {
      throw StringError("Transformer blocks with NCHW layout are not yet supported by the " KATAGO_GPU_BACKEND_NAME " backend");
    }
    testAssert(desc->linearGate.inChannels == desc->linear1.inChannels);
    testAssert(desc->linearGate.outChannels == desc->linear1.outChannels);
    testAssert(desc->linear1.inChannels == numChannels);
    testAssert(desc->linear1.outChannels == ffnChannels);
#if defined(KATAGO_GPU_CUDA) && defined(USE_CUTLASS_FUSED_FFN)
    if(cudaHandles->fusedFFNEnabled && useFP16) {
      usingFusedFFN = CudaFusedFFN::supportsShape(ffnChannels, numChannels);
      if(!usingFusedFFN && !cudaHandles->loggedFusedFFNRejected) {
        cudaHandles->loggedFusedFFNRejected = true;
        if(cudaHandles->logger != NULL)
          cudaHandles->logger->write(
            "Cuda backend: CUTLASS fused FFN kernel does not support this FFN shape (channels " +
            Global::intToString(numChannels) + " -> " + Global::intToString(ffnChannels) +
            "), using the unfused path");
      }
    }
    if(usingFusedFFN) {
      if(!cudaHandles->loggedUsingFusedFFN) {
        cudaHandles->loggedUsingFusedFFN = true;
        if(cudaHandles->logger != NULL)
          cudaHandles->logger->write("Cuda backend: using CUTLASS fused FFN kernel");
      }
      // Desc weights are in-major ([inC][outC] flat, what the cuBLAS path consumes directly),
      // so transpose each to out-major for the fused kernel.
      const int inC = numChannels;
      const int outC = ffnChannels;
      vector<float> packedT((size_t)2 * outC * inC);
      for(int o = 0; o < outC; o++) {
        for(int i = 0; i < inC; i++) {
          packedT[(size_t)o * inC + i] = desc->linear1.weights[(size_t)i * outC + o];
          packedT[(size_t)outC * inC + (size_t)o * inC + i] = desc->linearGate.weights[(size_t)i * outC + o];
        }
      }
      ffnPackedWeights = mallocAndCopyToDeviceMaybeShared(
        name + ":ffnFusedPacked",
        [&](void*& b){ CudaUtils::mallocAndCopyToDevice(name + ":ffnFusedPacked", packedT, b, useFP16); },
        desc, SharedWeightsKind::FFN_FUSED_PACKED, useFP16, cudaHandles->shareModelWeights);
    }
    else
#endif
    {
      vector<float> packed;
      packed.reserve(desc->linear1.weights.size() * 2);
      packed.insert(packed.end(), desc->linear1.weights.begin(), desc->linear1.weights.end());
      packed.insert(packed.end(), desc->linearGate.weights.begin(), desc->linearGate.weights.end());
      ffnPackedWeights = mallocAndCopyToDeviceMaybeShared(
        name + ":ffnPacked",
        [&](void*& b){ CudaUtils::mallocAndCopyToDevice(name + ":ffnPacked", packed, b, useFP16); },
        desc, SharedWeightsKind::FFN_PACKED, useFP16, cudaHandles->shareModelWeights);
    }
  }

  ~TransformerFFNBlock()
  {}

  size_t requiredWorkspaceBytes(
    CudaHandles* cudaHandles,
    int batchSize
  ) const {
    (void)cudaHandles;
    (void)batchSize;
    return 0;
  }

  // pendingTrunkDelta: see TransformerAttentionBlock::apply.
  void apply(
    CudaHandles* cudaHandles,
    ScratchBuffers* scratch,
    int batchSize,
    void* trunkBuf,
    void* trunkScratchBuf,
    void* maskBuf,
    float* maskSumBuf,
    void* workspaceBuf,
    size_t workspaceBytes,
    void** pendingTrunkDelta = NULL
  ) const {
    (void)maskSumBuf;

    int seqLen = nnXLen * nnYLen;
    int matBatchSize = batchSize * seqLen;

    // Step 1: RMSNorm, with any deferred residual add fused in.
    const void* pendingDeltaIn = (pendingTrunkDelta != NULL) ? *pendingTrunkDelta : NULL;
    preLN.apply(cudaHandles, batchSize, seqLen, trunkBuf, trunkScratchBuf, maskBuf, pendingDeltaIn);
    if(pendingTrunkDelta != NULL)
      *pendingTrunkDelta = NULL;

#ifdef DEBUG_INTERMEDIATE_VALUES
    CudaUtils::debugPrint3D(KATAGO_GPU_BACKEND_NAME " FFN RMSNorm out", trunkScratchBuf, batchSize, numChannels, seqLen, usingNHWC, usingFP16, maskBuf);
#endif

    // Steps 2 and 3: linear1 and gate projections, then SwiGLU combining them. When chosen at
    // construction, the CUTLASS fused dual GEMM computes all three in one kernel with no
    // intermediate global memory traffic. Otherwise one strided-batched GEMM over the shared
    // input runs both projections, then a separate SwiGLU kernel.
    //Max-batch-sized. See the scratch-sizing note at the attention q/k/v buffers.
    SizedBuf<void*> ffnBuf(scratch->allocator, scratch->getBufSizeXY(2 * ffnChannels));
    void* gatePtr = (char*)ffnBuf.buf + scratch->getBufSizeXY(ffnChannels);
    bool usedFusedFFN = false;
#if defined(KATAGO_GPU_CUDA) && defined(USE_CUTLASS_FUSED_FFN)
    if(usingFusedFFN) {
      // Same output-element-count limit the unfused SwiGLU path enforces below, for parity
      // and so CUTLASS's int-based problem coordinates stay comfortably in range.
      if((size_t)ffnChannels * (size_t)matBatchSize >= (size_t)2147483647)
        throw StringError(KATAGO_GPU_BACKEND_NAME " fused FFN element count exceeds the 32-bit limit");
      const half* w1 = (const half*)ffnPackedWeights.buf;
      const half* wGate = w1 + (size_t)ffnChannels * numChannels;
      CudaFusedFFN::runSwiGLU(
        (const half*)trunkScratchBuf, w1, wGate, (half*)ffnBuf.buf,
        matBatchSize, ffnChannels, numChannels, cudaHandles->stream);
      CUDA_ERR(name.c_str(), cudaPeekAtLastError());
      usedFusedFFN = true;
    }
#endif
    if(!usedFusedFFN) {
      {
        const size_t eltBytes = usingFP16 ? sizeof(cublas_half_t) : sizeof(float);
        const long long strideElts = (long long)(scratch->getBufSizeXY(ffnChannels) / eltBytes);
        applySharedInputStridedMatMuls(
          cudaHandles, scratch, usingFP16, name,
          ffnChannels, matBatchSize, numChannels,
          ffnPackedWeights.buf, trunkScratchBuf, ffnBuf.buf, strideElts, 2
        );
      }

      if((size_t)ffnChannels * (size_t)matBatchSize >= (size_t)2147483647)
        throw StringError(KATAGO_GPU_BACKEND_NAME " SwiGLU element count exceeds the 32-bit index limit used by the kernel");
      int totalSize = (int)((size_t)ffnChannels * matBatchSize);
      if(!usingFP16) {
        customCudaSwiGLU((const float*)ffnBuf.buf, (const float*)gatePtr, (float*)ffnBuf.buf, totalSize,cudaHandles->stream);
      }
      else {
        customCudaSwiGLU((const half*)ffnBuf.buf, (const half*)gatePtr, (half*)ffnBuf.buf, totalSize,cudaHandles->stream);
      }
      CUDA_ERR(name.c_str(), cudaPeekAtLastError());
    }

#ifdef DEBUG_INTERMEDIATE_VALUES
    CudaUtils::debugPrint2D(KATAGO_GPU_BACKEND_NAME " FFN SwiGLU", ffnBuf.buf, matBatchSize, ffnChannels, usingFP16);
#endif

    // Step 4 and 5: linear2 projection back to trunk channels, and residual addition.
    // Same masking logic as the attention block's output projection: fuse the residual into the
    // GEMM only when there is no mask.
    if(maskBuf == NULL) {
      linear2.apply(cudaHandles, scratch, matBatchSize, ffnBuf.buf, trunkBuf, workspaceBuf, workspaceBytes, true);
      CUDA_ERR(name.c_str(), cudaPeekAtLastError());
    }
    else {
      linear2.apply(cudaHandles, scratch, matBatchSize, ffnBuf.buf, trunkScratchBuf, workspaceBuf, workspaceBytes);

      if(pendingTrunkDelta != NULL) {
        // Defer the add so the next block can fuse it into its pre-norm.
        *pendingTrunkDelta = trunkScratchBuf;
      }
      // trunk += trunkScratch * mask
      else if(!usingFP16) {
        customCudaMaskedResidualAddNHWC((float*)trunkBuf, (const float*)trunkScratchBuf, (const float*)maskBuf, batchSize, seqLen, numChannels,cudaHandles->stream);
      }
      else {
        customCudaMaskedResidualAddNHWC((half*)trunkBuf, (const half*)trunkScratchBuf, (const half*)maskBuf, batchSize, seqLen, numChannels,cudaHandles->stream);
      }
      CUDA_ERR(name.c_str(), cudaPeekAtLastError());
    }

#ifdef DEBUG_INTERMEDIATE_VALUES
    CudaUtils::debugPrint3D(KATAGO_GPU_BACKEND_NAME " FFN residual", trunkBuf, batchSize, numChannels, seqLen, usingNHWC, usingFP16, maskBuf);
#endif
  }
};

//------------------------------------------------------------------------------

BlockStack::BlockStack(
  CudaHandles* cudaHandles,
  CudnnManager* manager,
  int nBlocks,
  int trunkChannels,
  const std::vector<std::pair<int, unique_ptr_void>>& descBlocks,
  int nnX,
  int nnY,
  bool useFP16,
  bool useNHWC
) :
  numBlocks(nBlocks),
  trunkNumChannels(trunkChannels),
  nnXLen(nnX),
  nnYLen(nnY),
  usingFP16(useFP16),
  usingNHWC(useNHWC)
{
  testAssert(numBlocks == descBlocks.size());
  for(int i = 0; i<numBlocks; i++) {
    if(descBlocks[i].first == ORDINARY_BLOCK_KIND) {
      ResidualBlockDesc* blockDesc = (ResidualBlockDesc*)descBlocks[i].second.get();
      unique_ptr_void blockPtr = make_unique_void(
        new ResidualBlock(
          cudaHandles,
          manager,
          blockDesc,
          nnXLen,
          nnYLen,
          useFP16,
          useNHWC
        )
      );
      blocks.emplace_back(ORDINARY_BLOCK_KIND,std::move(blockPtr));
    }
    else if(descBlocks[i].first == GLOBAL_POOLING_BLOCK_KIND) {
      GlobalPoolingResidualBlockDesc* blockDesc = (GlobalPoolingResidualBlockDesc*)descBlocks[i].second.get();
      unique_ptr_void blockPtr = make_unique_void(
        new GlobalPoolingResidualBlock(
          cudaHandles,
          manager,
          blockDesc,
          nnXLen,
          nnYLen,
          useFP16,
          useNHWC
        )
      );
      blocks.emplace_back(GLOBAL_POOLING_BLOCK_KIND,std::move(blockPtr));
    }
    else if(descBlocks[i].first == NESTED_BOTTLENECK_BLOCK_KIND) {
      NestedBottleneckResidualBlockDesc* blockDesc = (NestedBottleneckResidualBlockDesc*)descBlocks[i].second.get();
      unique_ptr_void blockPtr = make_unique_void(
        new NestedBottleneckResidualBlock(
          cudaHandles,
          manager,
          blockDesc,
          nnXLen,
          nnYLen,
          useFP16,
          useNHWC
        )
      );
      blocks.emplace_back(NESTED_BOTTLENECK_BLOCK_KIND,std::move(blockPtr));
    }
    else if(descBlocks[i].first == TRANSFORMER_ATTENTION_BLOCK_KIND) {
      TransformerAttentionDesc* blockDesc = (TransformerAttentionDesc*)descBlocks[i].second.get();
      unique_ptr_void blockPtr = make_unique_void(
        new TransformerAttentionBlock(
          cudaHandles,
          blockDesc,
          nnXLen,
          nnYLen,
          useFP16,
          useNHWC
        )
      );
      blocks.emplace_back(TRANSFORMER_ATTENTION_BLOCK_KIND, std::move(blockPtr));
    }
    else if(descBlocks[i].first == TRANSFORMER_FFN_BLOCK_KIND) {
      TransformerFFNDesc* blockDesc = (TransformerFFNDesc*)descBlocks[i].second.get();
      unique_ptr_void blockPtr = make_unique_void(
        new TransformerFFNBlock(
          cudaHandles,
          blockDesc,
          nnXLen,
          nnYLen,
          useFP16,
          useNHWC
        )
      );
      blocks.emplace_back(TRANSFORMER_FFN_BLOCK_KIND, std::move(blockPtr));
    }
    else {
      ASSERT_UNREACHABLE;
    }
  }
}
BlockStack::~BlockStack() {
}

size_t BlockStack::requiredWorkspaceBytes(
  CudaHandles* cudaHandles,
  int batchSize
) const {
  size_t bytes = 0;
  size_t b;

  for(int i = 0; i<blocks.size(); i++) {
    if(blocks[i].first == ORDINARY_BLOCK_KIND) {
      ResidualBlock* block = (ResidualBlock*)blocks[i].second.get();
      b = block->requiredWorkspaceBytes(cudaHandles,batchSize);
      bytes = std::max(bytes,b);
    }
    else if(blocks[i].first == GLOBAL_POOLING_BLOCK_KIND) {
      GlobalPoolingResidualBlock* block = (GlobalPoolingResidualBlock*)blocks[i].second.get();
      b = block->requiredWorkspaceBytes(cudaHandles,batchSize);
      bytes = std::max(bytes,b);
    }
    else if(blocks[i].first == NESTED_BOTTLENECK_BLOCK_KIND) {
      NestedBottleneckResidualBlock* block = (NestedBottleneckResidualBlock*)blocks[i].second.get();
      b = block->requiredWorkspaceBytes(cudaHandles,batchSize);
      bytes = std::max(bytes,b);
    }
    else if(blocks[i].first == TRANSFORMER_ATTENTION_BLOCK_KIND) {
      TransformerAttentionBlock* block = (TransformerAttentionBlock*)blocks[i].second.get();
      b = block->requiredWorkspaceBytes(cudaHandles, batchSize);
      bytes = std::max(bytes, b);
    }
    else if(blocks[i].first == TRANSFORMER_FFN_BLOCK_KIND) {
      TransformerFFNBlock* block = (TransformerFFNBlock*)blocks[i].second.get();
      b = block->requiredWorkspaceBytes(cudaHandles, batchSize);
      bytes = std::max(bytes, b);
    }
    else {
      ASSERT_UNREACHABLE;
    }
  }
  return bytes;
}

void BlockStack::apply(
  CudaHandles* cudaHandles,
  ScratchBuffers* scratch,
  int batchSize,
  void* maskBuf,
  float* maskSumBuf,
  void* trunkBuf,
  void* trunkScratchBuf,
  void* workspaceBuf,
  size_t workspaceBytes
) const {

  // Consecutive transformer blocks on the masked path defer each block's residual add and fuse
  // it into the next block's pre-norm (one kernel instead of two, one less trunk round trip).
  // pendingDelta points at the deferred delta (in trunkScratchBuf) when one is outstanding,
  // and it must be flushed into trunkBuf before anything other than a transformer block reads trunk.
  void* pendingDelta = NULL;
  auto flushPendingDelta = [&]() {
    if(pendingDelta == NULL)
      return;
    if(!usingFP16)
      customCudaMaskedResidualAddNHWC((float*)trunkBuf, (const float*)pendingDelta, (const float*)maskBuf, batchSize, nnXLen*nnYLen, trunkNumChannels,cudaHandles->stream);
    else
      customCudaMaskedResidualAddNHWC((half*)trunkBuf, (const half*)pendingDelta, (const half*)maskBuf, batchSize, nnXLen*nnYLen, trunkNumChannels,cudaHandles->stream);
    CUDA_ERR("BlockStack::flushPendingDelta", cudaPeekAtLastError());
    pendingDelta = NULL;
  };
  void** pendingDeltaArg = cudaHandles->fuseResidualNorm ? &pendingDelta : NULL;
#ifdef DEBUG_INTERMEDIATE_VALUES
  // Deferral would make the residual debug prints inside the transformer blocks show a stale
  // trunk, so disable it entirely when debug printing is compiled in.
  pendingDeltaArg = NULL;
#endif

  for(int i = 0; i<blocks.size(); i++) {
#ifdef DEBUG_INTERMEDIATE_VALUES
    CudaUtils::debugPrint3D(KATAGO_GPU_BACKEND_NAME " Blockstack block " + Global::intToString(i), trunkBuf, batchSize, trunkNumChannels, nnXLen*nnYLen, usingNHWC, usingFP16, maskBuf);
#endif

    if(blocks[i].first == ORDINARY_BLOCK_KIND) {
      flushPendingDelta();
      ResidualBlock* block = (ResidualBlock*)blocks[i].second.get();
      block->apply(
        cudaHandles,
        scratch,
        batchSize,
        trunkBuf,
        trunkScratchBuf,
        maskBuf,
        workspaceBuf,
        workspaceBytes
      );
    }
    else if(blocks[i].first == GLOBAL_POOLING_BLOCK_KIND) {
      flushPendingDelta();
      GlobalPoolingResidualBlock* block = (GlobalPoolingResidualBlock*)blocks[i].second.get();
      block->apply(
        cudaHandles,
        scratch,
        batchSize,
        trunkBuf,
        trunkScratchBuf,
        maskBuf,
        maskSumBuf,
        workspaceBuf,
        workspaceBytes
      );
    }
    else if(blocks[i].first == NESTED_BOTTLENECK_BLOCK_KIND) {
      flushPendingDelta();
      NestedBottleneckResidualBlock* block = (NestedBottleneckResidualBlock*)blocks[i].second.get();
      block->apply(
        cudaHandles,
        scratch,
        batchSize,
        trunkBuf,
        trunkScratchBuf,
        maskBuf,
        maskSumBuf,
        workspaceBuf,
        workspaceBytes
      );
    }
    else if(blocks[i].first == TRANSFORMER_ATTENTION_BLOCK_KIND) {
      TransformerAttentionBlock* block = (TransformerAttentionBlock*)blocks[i].second.get();
      block->apply(
        cudaHandles,
        scratch,
        batchSize,
        trunkBuf,
        trunkScratchBuf,
        maskBuf,
        maskSumBuf,
        workspaceBuf,
        workspaceBytes,
        pendingDeltaArg
      );
    }
    else if(blocks[i].first == TRANSFORMER_FFN_BLOCK_KIND) {
      TransformerFFNBlock* block = (TransformerFFNBlock*)blocks[i].second.get();
      block->apply(
        cudaHandles,
        scratch,
        batchSize,
        trunkBuf,
        trunkScratchBuf,
        maskBuf,
        maskSumBuf,
        workspaceBuf,
        workspaceBytes,
        pendingDeltaArg
      );
    }
    else {
      ASSERT_UNREACHABLE;
    }

    if(!cudaHandles->loggedUsingFusedResidualNorm && pendingDelta != NULL) {
      cudaHandles->loggedUsingFusedResidualNorm = true;
      if(cudaHandles->logger != NULL)
        cudaHandles->logger->write(KATAGO_GPU_BACKEND_NAME " backend: using fused residual add and pre-norm for transformer blocks");
    }
  }
  flushPendingDelta();
}
//------------------------------------------------------------------------------

struct SGFMetadataEncoder {
  const string name;

  const bool usingFP16;

  const MatMulLayer mul1;
  const MatBiasLayer bias1;
  const MatMulLayer mul2;
  const MatBiasLayer bias2;
  const MatMulLayer mul3;

  SGFMetadataEncoder() = delete;
  SGFMetadataEncoder(const SGFMetadataEncoder&) = delete;
  SGFMetadataEncoder& operator=(const SGFMetadataEncoder&) = delete;

  SGFMetadataEncoder(
    CudaHandles* cudaHandles,
    const SGFMetadataEncoderDesc* desc,
    bool useFP16
  ) :
    name(desc->name),
    usingFP16(useFP16),
    mul1(cudaHandles,&desc->mul1,useFP16),
    bias1(cudaHandles,&desc->bias1,useFP16,desc->act1.activation),
    mul2(cudaHandles,&desc->mul2,useFP16),
    bias2(cudaHandles,&desc->bias2,useFP16,desc->act2.activation),
    mul3(cudaHandles,&desc->mul3,useFP16)
  {
  }

  ~SGFMetadataEncoder()
  {
  }

  size_t requiredWorkspaceBytes(
    CudaHandles* cudaHandles,
    int batchSize
  ) const {
    (void)batchSize;
    size_t bytes = 0;
    size_t b;

    b = mul1.requiredWorkspaceBytes(cudaHandles);
    bytes = std::max(bytes,b);
    b = mul2.requiredWorkspaceBytes(cudaHandles);
    bytes = std::max(bytes,b);
    b = mul3.requiredWorkspaceBytes(cudaHandles);
    bytes = std::max(bytes,b);

    return bytes;
  }

  void apply(
    CudaHandles* cudaHandles,
    ScratchBuffers* scratch,
    int batchSize,
    void* inputBuf,
    void* outputBuf,
    void* workspaceBuf,
    size_t workspaceBytes
  ) const {
    SizedBuf<void*> internalBuf1(scratch->allocator, scratch->getBufSizeFloat(std::max(mul1.outChannels,mul2.outChannels)));
    SizedBuf<void*> internalBuf2(scratch->allocator, scratch->getBufSizeFloat(std::max(mul1.outChannels,mul2.outChannels)));

    mul1.apply(cudaHandles,scratch,batchSize,inputBuf,internalBuf1.buf,workspaceBuf,workspaceBytes);
    bias1.apply(cudaHandles,batchSize,internalBuf1.buf);
    mul2.apply(cudaHandles,scratch,batchSize,internalBuf1.buf,internalBuf2.buf,workspaceBuf,workspaceBytes);
    bias2.apply(cudaHandles,batchSize,internalBuf2.buf);
    mul3.apply(cudaHandles,scratch,batchSize,internalBuf2.buf,outputBuf,workspaceBuf,workspaceBytes);
  }

};


//----------------------------------------------------------------------------

struct Trunk {
  const string name;
  const int modelVersion;
  const int numBlocks;
  const int trunkNumChannels;

  const int nnXLen;
  const int nnYLen;
  const bool usingFP16;
  const bool usingNHWC;
#ifdef KATAGO_GPU_HIP
  // MIOpen (unlike cuDNN) requires the input and output tensors of a convolution to share one
  // layout, so when the layout the CPU wrote the input in differs from the layout the model runs
  // in, the input is converted on-device before the initial convolution (see apply).
  const bool inputsRequireLayoutConversion;
#endif

  std::unique_ptr<ConvLayer> initialConv;
  std::unique_ptr<MatMulLayer> initialMatMul;
  std::unique_ptr<SGFMetadataEncoder> sgfMetadataEncoder;
  const BlockStack blocks;
  const int trunkNormKind;
  std::unique_ptr<BatchNormLayer> trunkTipBN;
  std::unique_ptr<RMSNormLayer> trunkTipRMSNorm;

  Trunk() = delete;
  Trunk(const Trunk&) = delete;
  Trunk& operator=(const Trunk&) = delete;

  Trunk(
    CudaHandles* cudaHandles,
    CudnnManager* manager,
    const TrunkDesc* desc,
    int nnX,
    int nnY,
    bool inputsUseNHWC,
    bool useFP16,
    bool useNHWC
  ) :
    name(desc->name),
    modelVersion(desc->modelVersion),
    numBlocks(desc->numBlocks),
    trunkNumChannels(desc->trunkNumChannels),
    nnXLen(nnX),
    nnYLen(nnY),
    usingFP16(useFP16),
    usingNHWC(useNHWC),
#ifdef KATAGO_GPU_HIP
    inputsRequireLayoutConversion(inputsUseNHWC != useNHWC),
#endif
    blocks(cudaHandles,manager,desc->numBlocks,desc->trunkNumChannels,desc->blocks,nnX,nnY,useFP16,useNHWC),
    trunkNormKind(desc->trunkNormKind)
  {
    int midNumChannels = desc->midNumChannels;
    int regularNumChannels = desc->regularNumChannels;
    int gpoolNumChannels = desc->gpoolNumChannels;

    int maxBatchSize = manager->maxBatchSize;
    CudaUtils::checkBufferSize(maxBatchSize,nnXLen,nnYLen,trunkNumChannels);
    CudaUtils::checkBufferSize(maxBatchSize,nnXLen,nnYLen,midNumChannels);
    CudaUtils::checkBufferSize(maxBatchSize,nnXLen,nnYLen,regularNumChannels);
    CudaUtils::checkBufferSize(maxBatchSize,nnXLen,nnYLen,gpoolNumChannels);

#ifdef KATAGO_GPU_CUDA
    initialConv = std::make_unique<ConvLayer>(cudaHandles,manager,&desc->initialConv,useFP16,inputsUseNHWC,useNHWC);
#else  // KATAGO_GPU_HIP
    // The initial conv's input descriptor uses the model's layout, not the CPU-side input layout:
    // if the two differ, apply() converts the input on-device first (MIOpen cannot do the
    // conversion inside the convolution the way cuDNN does with mixed input/output descriptors).
    initialConv = std::make_unique<ConvLayer>(cudaHandles,manager,&desc->initialConv,useFP16,useNHWC,useNHWC);
#endif
    initialMatMul = std::make_unique<MatMulLayer>(cudaHandles,&desc->initialMatMul,useFP16);
    if(desc->metaEncoderVersion > 0) {
      sgfMetadataEncoder = std::make_unique<SGFMetadataEncoder>(cudaHandles,&desc->sgfMetadataEncoder,useFP16);
      testAssert(sgfMetadataEncoder->mul3.outChannels == initialMatMul->outChannels);
    }

    if(desc->trunkNormKind == TRUNK_NORM_KIND_STANDARD) {
      trunkTipBN = std::make_unique<BatchNormLayer>(cudaHandles,&desc->trunkTipBN,&desc->trunkTipActivation,nnXLen,nnYLen,useFP16,useNHWC);
    }
    else if(desc->trunkNormKind == TRUNK_NORM_KIND_RMSNORM) {
      trunkTipRMSNorm = std::make_unique<RMSNormLayer>(cudaHandles,&desc->trunkTipRMSNorm,desc->trunkTipActivation.activation,nnXLen,nnYLen,useFP16,useNHWC);
    }
    else {
      throw StringError("Unsupported trunk norm kind: " + Global::intToString(desc->trunkNormKind));
    }
    testAssert(desc->blocks.size() == numBlocks);
  }

  ~Trunk()
  {
  }

  size_t requiredWorkspaceBytes(
    CudaHandles* cudaHandles,
    int batchSize
  ) const {
    size_t bytes = 0;
    size_t b;

    b = initialConv->requiredWorkspaceBytes(cudaHandles,batchSize);
    bytes = std::max(bytes,b);

    b = initialMatMul->requiredWorkspaceBytes(cudaHandles);
    bytes = std::max(bytes,b);

    if(sgfMetadataEncoder != nullptr) {
      b = sgfMetadataEncoder->requiredWorkspaceBytes(cudaHandles,batchSize);
      bytes = std::max(bytes,b);
    }

    b = blocks.requiredWorkspaceBytes(cudaHandles,batchSize);
    bytes = std::max(bytes,b);
    return bytes;
  }

  void apply(
    CudaHandles* cudaHandles,
    ScratchBuffers* scratch,
    int batchSize,
    void* inputBuf,
    void* inputGlobalBuf,
    void* inputMetaBuf,
    void* maskBuf,
    float* maskSumBuf,
    void* trunkBuf,
    void* workspaceBuf,
    size_t workspaceBytes
  ) const {

    SizedBuf<void*> trunkScratch(scratch->allocator, scratch->getBufSizeXY(trunkNumChannels));

#ifdef KATAGO_GPU_CUDA
    //Feed the conv into trunkScratch.buf, not trunkBuf
    initialConv->apply(cudaHandles,scratch,batchSize,false,inputBuf,trunkScratch.buf,workspaceBuf,workspaceBytes);
#else  // KATAGO_GPU_HIP
    // Convert the input tensor to the model's layout if the CPU wrote it in the other one.
    // Every other consumer of inputBuf (mask extraction) is layout-independent, so only the
    // initial conv's input needs this. See inputsRequireLayoutConversion.
    SizedBuf<void*> inputConverted(
      scratch->allocator,
      inputsRequireLayoutConversion ? scratch->getBufSizeXY(initialConv->inChannels) : 1);
    void* initialConvInput = inputBuf;
    if(inputsRequireLayoutConversion) {
      int inChannels = initialConv->inChannels;
      int xySize = nnXLen * nnYLen;
      if(usingNHWC) {
        if(!usingFP16)
          customCudaCopyNCHWtoNHWC((const float*)inputBuf,(float*)inputConverted.buf,batchSize,inChannels,xySize,cudaHandles->stream);
        else
          customCudaCopyNCHWtoNHWC((const half*)inputBuf,(half*)inputConverted.buf,batchSize,inChannels,xySize,cudaHandles->stream);
      }
      else {
        if(!usingFP16)
          customCudaCopyNHWCtoNCHW((const float*)inputBuf,(float*)inputConverted.buf,batchSize,inChannels,xySize,cudaHandles->stream);
        else
          customCudaCopyNHWCtoNCHW((const half*)inputBuf,(half*)inputConverted.buf,batchSize,inChannels,xySize,cudaHandles->stream);
      }
      CUDA_ERR(name.c_str(),hipPeekAtLastError());
      initialConvInput = inputConverted.buf;
    }

    //Feed the conv into trunkScratch.buf, not trunkBuf
    initialConv->apply(cudaHandles,scratch,batchSize,false,initialConvInput,trunkScratch.buf,workspaceBuf,workspaceBytes);
#endif

    #ifdef DEBUG_INTERMEDIATE_VALUES
    CudaUtils::debugPrint3D(string("After initial conv"), trunkScratch.buf, batchSize, trunkNumChannels, nnXLen*nnYLen, usingNHWC, usingFP16);
    #endif

    //Feed the matmul into trunkBuf
    initialMatMul->apply(cudaHandles,scratch,batchSize,inputGlobalBuf,trunkBuf,workspaceBuf,workspaceBytes);
    //Then accumulate it into trunkScratch.buf, broadcasting during the process
    if(!usingFP16) {
      if(!usingNHWC)
        customCudaAddNCBiasInplaceNCHW((float*)trunkScratch.buf,(const float*)trunkBuf,batchSize,trunkNumChannels,nnXLen*nnYLen,cudaHandles->stream);
      else
        customCudaAddNCBiasInplaceNHWC((float*)trunkScratch.buf,(const float*)trunkBuf,batchSize,nnXLen*nnYLen,trunkNumChannels,cudaHandles->stream);
    }
    else {
      if(!usingNHWC)
        customCudaAddNCBiasInplaceNCHW((half*)trunkScratch.buf,(const half*)trunkBuf,batchSize,trunkNumChannels,nnXLen*nnYLen,cudaHandles->stream);
      else
        customCudaAddNCBiasInplaceNHWC((half*)trunkScratch.buf,(const half*)trunkBuf,batchSize,nnXLen*nnYLen,trunkNumChannels,cudaHandles->stream);
    }
    CUDA_ERR(name.c_str(),cudaPeekAtLastError());

    if(sgfMetadataEncoder != nullptr) {
      testAssert(inputMetaBuf != NULL);
      //Feed the result into trunkBuf
      sgfMetadataEncoder->apply(cudaHandles,scratch,batchSize,inputMetaBuf,trunkBuf,workspaceBuf,workspaceBytes);
      //Then accumulate it into trunkScratch.buf, broadcasting during the process
      if(!usingFP16) {
        if(!usingNHWC)
          customCudaAddNCBiasInplaceNCHW((float*)trunkScratch.buf,(const float*)trunkBuf,batchSize,trunkNumChannels,nnXLen*nnYLen,cudaHandles->stream);
        else
          customCudaAddNCBiasInplaceNHWC((float*)trunkScratch.buf,(const float*)trunkBuf,batchSize,nnXLen*nnYLen,trunkNumChannels,cudaHandles->stream);
      }
      else {
        if(!usingNHWC)
          customCudaAddNCBiasInplaceNCHW((half*)trunkScratch.buf,(const half*)trunkBuf,batchSize,trunkNumChannels,nnXLen*nnYLen,cudaHandles->stream);
        else
          customCudaAddNCBiasInplaceNHWC((half*)trunkScratch.buf,(const half*)trunkBuf,batchSize,nnXLen*nnYLen,trunkNumChannels,cudaHandles->stream);
      }
      CUDA_ERR(name.c_str(),cudaPeekAtLastError());
    }
    else {
      testAssert(inputMetaBuf == NULL);
    }

    //Flip trunkBuf and trunkScratch.buf so that the result gets accumulated in trunkScratch.buf
    blocks.apply(
      cudaHandles,
      scratch,
      batchSize,
      maskBuf,
      maskSumBuf,
      trunkScratch.buf,
      trunkBuf,
      workspaceBuf,
      workspaceBytes
    );

    //And now with the final norm port it from trunkScratch.buf to trunkBuf.
    if(trunkNormKind == TRUNK_NORM_KIND_STANDARD) {
      trunkTipBN->apply(cudaHandles,batchSize,trunkScratch.buf,maskBuf,trunkBuf);
    }
    else {
      trunkTipRMSNorm->apply(cudaHandles,scratch,batchSize,trunkScratch.buf,trunkBuf,maskBuf,maskSumBuf);
    }

    #ifdef DEBUG_INTERMEDIATE_VALUES
    CudaUtils::debugPrint3D(string("Trunk tip"), trunkBuf, batchSize, trunkNumChannels, nnXLen*nnYLen, usingNHWC, usingFP16);
    #endif
  }

};

//------------------------------------------------------------------------------

static void fillMaskFloatBufAndMaskSumBuf(CudaHandles* cudaHandles, void* maskBuf, float*& maskFloatBuf, float*& maskSumBuf, bool usingFP16, int batchSize, int nnXLen, int nnYLen) {
  if(!usingFP16) {
    maskFloatBuf = (float*)maskBuf;
    customCudaPoolRowsSumNCHW((const float*)maskFloatBuf,maskSumBuf,batchSize,1,nnXLen*nnYLen,1.0,cudaHandles->stream);
    CUDA_ERR("sumMask",cudaPeekAtLastError());
  }
  else {
    customCudaCopyFromHalf((const half*)maskBuf,maskFloatBuf,batchSize*nnXLen*nnYLen,cudaHandles->stream);
    CUDA_ERR("copyMaskFromHalf",cudaPeekAtLastError());
    customCudaPoolRowsSumNCHW((const float*)maskFloatBuf,maskSumBuf,batchSize,1,nnXLen*nnYLen,1.0,cudaHandles->stream);
    CUDA_ERR("sumMask",cudaPeekAtLastError());
  }
}


//------------------------------------------------------------------------------

struct PolicyHead {
  const string name;
  const int modelVersion;
  const int nnXLen;
  const int nnYLen;
  const int p1Channels;
  const int g1Channels;
  const int p2Channels;
  const bool usingFP16;
  const bool usingNHWC;

  const ConvLayer p1Conv;
  const ConvLayer g1Conv;
  const BatchNormLayer g1BN;
  const MatMulLayer gpoolToBiasMul;
  const BatchNormLayer p1BN;
  const ConvLayer p2Conv;
  const MatMulLayer gpoolToPassMul;
  const MatBiasLayer gpoolToPassBias;
  const MatMulLayer gpoolToPassMul2;

  PolicyHead() = delete;
  PolicyHead(const PolicyHead&) = delete;
  PolicyHead& operator=(const PolicyHead&) = delete;

  PolicyHead(
    CudaHandles* cudaHandles,
    CudnnManager* manager,
    const PolicyHeadDesc* desc,
    int nnX,
    int nnY,
    bool useFP16,
    bool useNHWC
  ) :
    name(desc->name),
    modelVersion(desc->modelVersion),
    nnXLen(nnX),
    nnYLen(nnY),
    p1Channels(desc->p1Conv.outChannels),
    g1Channels(desc->g1Conv.outChannels),
    p2Channels(desc->p2Conv.outChannels),
    usingFP16(useFP16),
    usingNHWC(useNHWC),
    p1Conv(cudaHandles,manager,&desc->p1Conv,useFP16,useNHWC),
    g1Conv(cudaHandles,manager,&desc->g1Conv,useFP16,useNHWC),
    g1BN(cudaHandles,&desc->g1BN,&desc->g1Activation,nnX,nnY,useFP16,useNHWC),
    gpoolToBiasMul(cudaHandles,&desc->gpoolToBiasMul,false),
    p1BN(cudaHandles,&desc->p1BN,&desc->p1Activation,nnX,nnY,false,useNHWC),
    p2Conv(cudaHandles,manager,&desc->p2Conv,false,useNHWC),
    gpoolToPassMul(cudaHandles,&desc->gpoolToPassMul,false),
    gpoolToPassBias(cudaHandles,&desc->gpoolToPassBias,false,desc->passActivation.activation),
    gpoolToPassMul2(cudaHandles,&desc->gpoolToPassMul2,false)
  {
  }

  ~PolicyHead()
  {
  }

  size_t requiredWorkspaceBytes(
    CudaHandles* cudaHandles,
    int batchSize
  ) const {
    size_t bytes = 0;
    size_t b;

    b = p1Conv.requiredWorkspaceBytes(cudaHandles,batchSize);
    bytes = std::max(bytes,b);
    b = g1Conv.requiredWorkspaceBytes(cudaHandles,batchSize);
    bytes = std::max(bytes,b);
    b = gpoolToBiasMul.requiredWorkspaceBytes(cudaHandles);
    bytes = std::max(bytes,b);
    b = p2Conv.requiredWorkspaceBytes(cudaHandles,batchSize);
    bytes = std::max(bytes,b);
    b = gpoolToPassMul.requiredWorkspaceBytes(cudaHandles);
    bytes = std::max(bytes,b);
    b = gpoolToPassMul2.requiredWorkspaceBytes(cudaHandles);
    bytes = std::max(bytes,b);
    b = sizeof(float)*batchSize*g1Channels*nnXLen*nnYLen;
    bytes = std::max(bytes,b);

    return bytes;
  }

  void apply(
    CudaHandles* cudaHandles,
    ScratchBuffers* scratch,
    int batchSize,
    void* maskBuf,
    float* maskFloatBuf,
    float* maskSumBuf,
    void* trunkBuf,
    float* policyPassBuf,
    float* policyBuf,
    void* workspaceBuf,
    size_t workspaceBytes
  ) const {

    SizedBuf<void*> p1Out(scratch->allocator, scratch->getBufSizeXYFloat(p1Channels)); //Need to hold floats, not just halfs
    SizedBuf<void*> p1Out2(scratch->allocator, scratch->getBufSizeXYFloat(p1Channels)); //Need to hold floats, not just halfs
    SizedBuf<void*> g1Out(scratch->allocator, scratch->getBufSizeXY(g1Channels));
    SizedBuf<void*> g1Out2(scratch->allocator, scratch->getBufSizeXY(g1Channels));
    SizedBuf<void*> g1Concat(scratch->allocator, scratch->getBufSizeFloat(g1Channels*3));
    SizedBuf<void*> g1Bias(scratch->allocator, scratch->getBufSizeFloat(p1Channels));
    SizedBuf<void*> p1Pass(scratch->allocator, scratch->getBufSizeFloat(p1Channels));

    p1Conv.apply(cudaHandles,scratch,batchSize,false,trunkBuf,p1Out.buf,workspaceBuf,workspaceBytes);
    g1Conv.apply(cudaHandles,scratch,batchSize,false,trunkBuf,g1Out.buf,workspaceBuf,workspaceBytes);
    g1BN.apply(cudaHandles,batchSize,g1Out.buf,maskBuf,g1Out2.buf);

    if(!usingFP16) {
      if(!usingNHWC)
        customCudaPoolRowsGPoolNCHW((const float*)g1Out2.buf,(float*)g1Concat.buf,batchSize,g1Channels,nnXLen*nnYLen,maskFloatBuf,maskSumBuf,cudaHandles->stream);
      else
        customCudaPoolRowsGPoolNHWC((const float*)g1Out2.buf,(float*)g1Concat.buf,batchSize,nnXLen*nnYLen,g1Channels,maskFloatBuf,maskSumBuf,cudaHandles->stream);
      CUDA_ERR(name.c_str(),cudaPeekAtLastError());
    }
    else {
      customCudaCopyFromHalf((const half*)g1Out2.buf,(float*)workspaceBuf,batchSize*g1Channels*nnXLen*nnYLen,cudaHandles->stream);
      CUDA_ERR(name.c_str(),cudaPeekAtLastError());
      if(!usingNHWC)
        customCudaPoolRowsGPoolNCHW((const float*)workspaceBuf,(float*)g1Concat.buf,batchSize,g1Channels,nnXLen*nnYLen,maskFloatBuf,maskSumBuf,cudaHandles->stream);
      else
        customCudaPoolRowsGPoolNHWC((const float*)workspaceBuf,(float*)g1Concat.buf,batchSize,nnXLen*nnYLen,g1Channels,maskFloatBuf,maskSumBuf,cudaHandles->stream);
      CUDA_ERR(name.c_str(),cudaPeekAtLastError());
    }

    gpoolToBiasMul.apply(cudaHandles,scratch,batchSize,g1Concat.buf,g1Bias.buf,workspaceBuf,workspaceBytes);

    #ifdef DEBUG_INTERMEDIATE_VALUES
    CudaUtils::debugPrint3D(string("p1 pre-gpool-sum"), p1Out.buf, batchSize, p1Channels, nnXLen*nnYLen, usingNHWC, usingFP16);
    CudaUtils::debugPrint3D(string("g1 pre-gpool"), g1Out.buf, batchSize, g1Channels, nnXLen*nnYLen, usingNHWC, usingFP16);
    CudaUtils::debugPrint2D(string("g1 pooled"), g1Concat.buf, batchSize, g1Channels*3, false);
    CudaUtils::debugPrint2D(string("g1 biases"), g1Bias.buf, batchSize, p1Channels, false);
    #endif

    float* p1OutBufA;
    float* p1OutBufB;
    if(!usingFP16) {
      p1OutBufA = (float*)p1Out.buf;
      p1OutBufB = (float*)p1Out2.buf;
    }
    else {
      customCudaCopyFromHalf((const half*)p1Out.buf,(float*)p1Out2.buf,batchSize*p1Channels*nnXLen*nnYLen,cudaHandles->stream);
      CUDA_ERR(name.c_str(),cudaPeekAtLastError());
      p1OutBufA = (float*)p1Out2.buf;
      p1OutBufB = (float*)p1Out.buf;
    }

    if(!usingNHWC)
      customCudaAddNCBiasInplaceNCHW(p1OutBufA,(float*)g1Bias.buf,batchSize,p1Channels,nnXLen*nnYLen,cudaHandles->stream);
    else
      customCudaAddNCBiasInplaceNHWC(p1OutBufA,(float*)g1Bias.buf,batchSize,nnXLen*nnYLen,p1Channels,cudaHandles->stream);
    CUDA_ERR(name.c_str(),cudaPeekAtLastError());

    p1BN.apply(cudaHandles,batchSize,p1OutBufA,maskFloatBuf,p1OutBufB);
    p2Conv.apply(cudaHandles,scratch,batchSize,false,p1OutBufB,(float*)policyBuf,workspaceBuf,workspaceBytes);

    if(modelVersion >= 15) {
      gpoolToPassMul.apply(cudaHandles,scratch,batchSize,g1Concat.buf,p1Pass.buf,workspaceBuf,workspaceBytes);
      gpoolToPassBias.apply(cudaHandles,batchSize,p1Pass.buf);
      gpoolToPassMul2.apply(cudaHandles,scratch,batchSize,p1Pass.buf,policyPassBuf,workspaceBuf,workspaceBytes);
    }
    else {
      gpoolToPassMul.apply(cudaHandles,scratch,batchSize,g1Concat.buf,policyPassBuf,workspaceBuf,workspaceBytes);
    }

    #ifdef DEBUG_INTERMEDIATE_VALUES
    CudaUtils::debugPrint3D(string("p1 after-gpool-sum"), p1OutBufA, batchSize, p1Channels, nnXLen*nnYLen, usingNHWC, false);
    CudaUtils::debugPrint2D(string("policypass"), policyPassBuf, batchSize, 1, false);
    CudaUtils::debugPrint3D(string("policy"), policyBuf, batchSize, p2Channels, nnXLen*nnYLen, usingNHWC, false);
    #endif

  }

};

//------------------------------------------------------------------------------

struct ValueHead {
  const string name;
  const int modelVersion;
  const int nnXLen;
  const int nnYLen;
  const int v1Channels;
  const int v2Channels;
  const int valueChannels;
  const int scoreValueChannels;
  const int ownershipChannels;
  const bool usingFP16;
  const bool usingNHWC;

  const ConvLayer v1Conv;
  const BatchNormLayer v1BN;
  const MatMulLayer v2Mul;
  const MatBiasLayer v2Bias;
  const MatMulLayer v3Mul;
  const MatBiasLayer v3Bias;
  const MatMulLayer sv3Mul;
  const MatBiasLayer sv3Bias;
  const ConvLayer vOwnershipConv;

  ValueHead() = delete;
  ValueHead(const ValueHead&) = delete;
  ValueHead& operator=(const ValueHead&) = delete;

  ValueHead(
    CudaHandles* cudaHandles,
    CudnnManager* manager,
    const ValueHeadDesc* desc,
    int nnX,
    int nnY,
    bool useFP16,
    bool useNHWC
  ) :
    name(desc->name),
    modelVersion(desc->modelVersion),
    nnXLen(nnX),
    nnYLen(nnY),
    v1Channels(desc->v1Conv.outChannels),
    v2Channels(desc->v2Mul.outChannels),
    valueChannels(desc->v3Mul.outChannels),
    scoreValueChannels(desc->sv3Mul.outChannels),
    ownershipChannels(desc->vOwnershipConv.outChannels),
    usingFP16(useFP16),
    usingNHWC(useNHWC),
    v1Conv(cudaHandles,manager,&desc->v1Conv,useFP16,useNHWC),
    v1BN(cudaHandles,&desc->v1BN,&desc->v1Activation,nnX,nnY,useFP16,useNHWC),
    v2Mul(cudaHandles,&desc->v2Mul,false),
    v2Bias(cudaHandles,&desc->v2Bias,false,desc->v2Activation.activation),
    v3Mul(cudaHandles,&desc->v3Mul,false),
    v3Bias(cudaHandles,&desc->v3Bias,false,ACTIVATION_IDENTITY),
    sv3Mul(cudaHandles,&desc->sv3Mul,false),
    sv3Bias(cudaHandles,&desc->sv3Bias,false,ACTIVATION_IDENTITY),
    vOwnershipConv(cudaHandles,manager,&desc->vOwnershipConv,useFP16,useNHWC)
  {
  }

  ~ValueHead()
  {
  }

  size_t requiredWorkspaceBytes(
    CudaHandles* cudaHandles,
    int batchSize
  ) const {
    size_t bytes = 0;
    size_t b;

    b = v1Conv.requiredWorkspaceBytes(cudaHandles,batchSize);
    bytes = std::max(bytes,b);
    b = v2Mul.requiredWorkspaceBytes(cudaHandles);
    bytes = std::max(bytes,b);
    b = v3Mul.requiredWorkspaceBytes(cudaHandles);
    bytes = std::max(bytes,b);
    b = sizeof(float)*batchSize*v1Channels*nnXLen*nnYLen;
    bytes = std::max(bytes,b);

    b = sv3Mul.requiredWorkspaceBytes(cudaHandles);
    bytes = std::max(bytes,b);
    b = vOwnershipConv.requiredWorkspaceBytes(cudaHandles,batchSize);
    bytes = std::max(bytes,b);
    b = sizeof(float)*batchSize*ownershipChannels*nnXLen*nnYLen;
    bytes = std::max(bytes,b);

    return bytes;
  }


  void apply(
    CudaHandles* cudaHandles,
    ScratchBuffers* scratch,
    int batchSize,
    void* maskBuf,
    float* maskSumBuf,
    void* trunkBuf,
    float* valueBuf,
    float* scoreValueBuf,
    void* ownershipBuf,
    void* workspaceBuf,
    size_t workspaceBytes
  ) const {
    SizedBuf<void*> v1Out(scratch->allocator, scratch->getBufSizeXY(v1Channels));
    SizedBuf<void*> v1Out2(scratch->allocator, scratch->getBufSizeXY(v1Channels));
    SizedBuf<void*> v1Mean(scratch->allocator, scratch->getBufSizeFloat(v1Channels*3));
    SizedBuf<void*> v2Out(scratch->allocator, scratch->getBufSizeFloat(v2Channels));
    SizedBuf<void*> ownershipScratch(scratch->allocator, scratch->getBufSizeXYFloat(ownershipChannels));

    v1Conv.apply(cudaHandles,scratch,batchSize,false,trunkBuf,v1Out.buf,workspaceBuf,workspaceBytes);
    v1BN.apply(cudaHandles,batchSize,v1Out.buf,maskBuf,v1Out2.buf);

    void* bufToBePooled = v1Out2.buf;
    if(usingFP16) {
      customCudaCopyFromHalf((const half*)v1Out2.buf,(float*)workspaceBuf,batchSize*v1Channels*nnXLen*nnYLen,cudaHandles->stream);
      CUDA_ERR(name.c_str(),cudaPeekAtLastError());
      bufToBePooled = workspaceBuf;
    }

    if(!usingNHWC)
      customCudaValueHeadPoolNCHW((float*)bufToBePooled,(float*)v1Mean.buf,batchSize,v1Channels,nnXLen*nnYLen,maskSumBuf,cudaHandles->stream);
    else
      customCudaValueHeadPoolNHWC((const float*)bufToBePooled,(float*)v1Mean.buf,batchSize,nnXLen*nnYLen,v1Channels,maskSumBuf,cudaHandles->stream);
    CUDA_ERR(name.c_str(),cudaPeekAtLastError());

    v2Mul.apply(cudaHandles,scratch,batchSize,v1Mean.buf,v2Out.buf,workspaceBuf,workspaceBytes);
    v2Bias.apply(cudaHandles,batchSize,v2Out.buf);
    v3Mul.apply(cudaHandles,scratch,batchSize,v2Out.buf,valueBuf,workspaceBuf,workspaceBytes);
    v3Bias.apply(cudaHandles,batchSize,valueBuf);

    sv3Mul.apply(cudaHandles,scratch,batchSize,v2Out.buf,scoreValueBuf,workspaceBuf,workspaceBytes);
    sv3Bias.apply(cudaHandles,batchSize,scoreValueBuf);

    #ifdef DEBUG_INTERMEDIATE_VALUES
    CudaUtils::debugPrint3D(string("v1"), v1Out.buf, batchSize, v1Channels, nnXLen*nnYLen, usingNHWC, usingFP16);
    CudaUtils::debugPrint2D(string("v1 pooled"), v1Mean.buf, batchSize, v1Channels, false);
    CudaUtils::debugPrint2D(string("v2"), v2Out.buf, batchSize, v2Channels, false);
    #endif

    if(!usingFP16) {
      vOwnershipConv.apply(cudaHandles,scratch,batchSize,false,v1Out2.buf,ownershipBuf,workspaceBuf,workspaceBytes);
    }
    else {
      vOwnershipConv.apply(cudaHandles,scratch,batchSize,false,v1Out2.buf,ownershipScratch.buf,workspaceBuf,workspaceBytes);
      customCudaCopyFromHalf((const half*)ownershipScratch.buf,(float*)ownershipBuf,batchSize*ownershipChannels*nnXLen*nnYLen,cudaHandles->stream);
      CUDA_ERR("vOwnership copy",cudaPeekAtLastError());
    }

  }

};

//------------------------------------------------------------------------------

struct Model {
  const string name;
  const int modelVersion;
  const int maxBatchSize;
  const int nnXLen;
  const int nnYLen;
  const int numInputChannels;
  const int numInputGlobalChannels;
  const int numInputMetaChannels;
  const int numPolicyChannels;
  const int numValueChannels;
  const int numScoreValueChannels;
  const int numOwnershipChannels;
  const bool usingFP16;
  const bool usingNHWC;
  const bool inputsUsingNHWC;

  std::unique_ptr<Trunk> trunk;
  std::unique_ptr<PolicyHead> policyHead;
  std::unique_ptr<ValueHead> valueHead;
  std::unique_ptr<CudnnManager> manager;

  Model() = delete;
  Model(const Model&) = delete;
  Model& operator=(const Model&) = delete;

  Model(
    CudaHandles* cudaHandles,
    const ModelDesc* desc,
    int maxBatchSz,
    int nnX,
    int nnY,
    bool inputsUseNHWC,
    bool useFP16,
    bool useNHWC
  ) :
    name(desc->name),
    modelVersion(desc->modelVersion),
    maxBatchSize(maxBatchSz),
    nnXLen(nnX),
    nnYLen(nnY),
    numInputChannels(desc->numInputChannels),
    numInputGlobalChannels(desc->numInputGlobalChannels),
    numInputMetaChannels(desc->numInputMetaChannels),
    numPolicyChannels(desc->numPolicyChannels),
    numValueChannels(desc->numValueChannels),
    numScoreValueChannels(desc->numScoreValueChannels),
    numOwnershipChannels(desc->numOwnershipChannels),
    usingFP16(useFP16),
    usingNHWC(useNHWC),
    inputsUsingNHWC(inputsUseNHWC)
  {
    if(nnXLen > NNPos::MAX_BOARD_LEN)
      throw StringError(Global::strprintf("nnXLen (%d) is greater than NNPos::MAX_BOARD_LEN (%d)",
        nnXLen, NNPos::MAX_BOARD_LEN
      ));
    if(nnYLen > NNPos::MAX_BOARD_LEN)
      throw StringError(Global::strprintf("nnYLen (%d) is greater than NNPos::MAX_BOARD_LEN (%d)",
        nnYLen, NNPos::MAX_BOARD_LEN
      ));

    int numFeatures = NNModelVersion::getNumSpatialFeatures(modelVersion);
    if(numInputChannels != numFeatures)
      throw StringError(Global::strprintf("Neural net numInputChannels (%d) was not the expected number based on version (%d)",
        numInputChannels, numFeatures
      ));
    int numGlobalFeatures = NNModelVersion::getNumGlobalFeatures(modelVersion);
    if(numInputGlobalChannels != numGlobalFeatures)
      throw StringError(Global::strprintf("Neural net numInputGlobalChannels (%d) was not the expected number based on version (%d)",
        numInputGlobalChannels, numGlobalFeatures
      ));
    if(numInputMetaChannels > 0) {
      if(numInputMetaChannels != SGFMetadata::METADATA_INPUT_NUM_CHANNELS)
        throw StringError(Global::strprintf("Neural net numInputMetaChannels (%d) was not the expected number (%d)",
          numInputMetaChannels, SGFMetadata::METADATA_INPUT_NUM_CHANNELS
        ));
    }

    CudaUtils::checkBufferSize(maxBatchSize,nnXLen,nnYLen,numInputChannels);
    CudaUtils::checkBufferSize(maxBatchSize,nnXLen,nnYLen,numInputGlobalChannels);
    CudaUtils::checkBufferSize(maxBatchSize,nnXLen,nnYLen,numInputMetaChannels);
    CudaUtils::checkBufferSize(maxBatchSize,nnXLen,nnYLen,numPolicyChannels);
    CudaUtils::checkBufferSize(maxBatchSize,nnXLen,nnYLen,numValueChannels);
    CudaUtils::checkBufferSize(maxBatchSize,nnXLen,nnYLen,numScoreValueChannels);
    CudaUtils::checkBufferSize(maxBatchSize,nnXLen,nnYLen,numOwnershipChannels);

    manager = std::make_unique<CudnnManager>(name, maxBatchSize, nnXLen, nnYLen);
    trunk = std::make_unique<Trunk>(cudaHandles,manager.get(),&desc->trunk,nnXLen,nnYLen,inputsUseNHWC,useFP16,useNHWC);
    policyHead = std::make_unique<PolicyHead>(cudaHandles,manager.get(),&desc->policyHead,nnXLen,nnYLen,useFP16,useNHWC);
    valueHead = std::make_unique<ValueHead>(cudaHandles,manager.get(),&desc->valueHead,nnXLen,nnYLen,useFP16,useNHWC);
  }

  ~Model()
  {
  }

  size_t requiredWorkspaceBytes(
    CudaHandles* cudaHandles,
    int batchSize
  ) const {
    size_t bytes = 0;
    size_t b;

    b = trunk->requiredWorkspaceBytes(cudaHandles,batchSize);
    bytes = std::max(bytes,b);
    b = policyHead->requiredWorkspaceBytes(cudaHandles,batchSize);
    bytes = std::max(bytes,b);
    b = valueHead->requiredWorkspaceBytes(cudaHandles,batchSize);
    bytes = std::max(bytes,b);

    return bytes;
  }

  void apply(
    CudaHandles* cudaHandles,
    ScratchBuffers* scratch,
    int batchSize,
    bool requireExactNNLen,

    void* inputBuf,
    void* inputGlobalBuf,
    void* inputMetaBuf,

    float* policyPassBuf,
    float* policyBuf,

    float* valueBuf,
    float* scoreValueBuf,
    void* ownershipBuf,

    void* workspaceBuf,
    size_t workspaceBytes
  ) const {
    // The attention bias derived from this forward's mask has not been built yet.
    scratch->attnBiasBuiltThisForward = false;

    SizedBuf<void*> mask(scratch->allocator, scratch->getBufSizeXY(1));
    SizedBuf<void*> maskFloat(scratch->allocator, scratch->getBufSizeXYFloat(1));
    SizedBuf<void*> maskSum(scratch->allocator, scratch->getBufSizeFloat(1));

    void* maskBuf = mask.buf;
    float* maskFloatBuf = (float*)maskFloat.buf;
    float* maskSumBuf = (float*)maskSum.buf;

    if(!usingFP16) {
      if(inputsUsingNHWC)
        customCudaChannel0ExtractNHWC((const float*)inputBuf, (float*)maskBuf, batchSize, nnXLen*nnYLen, numInputChannels,cudaHandles->stream);
      else
        customCudaChannel0ExtractNCHW((const float*)inputBuf, (float*)maskBuf, batchSize, numInputChannels, nnXLen*nnYLen,cudaHandles->stream);
      CUDA_ERR("modelExtractMask",cudaPeekAtLastError());
    }
    else {
      if(inputsUsingNHWC)
        customCudaChannel0ExtractNHWC((const half*)inputBuf, (half*)maskBuf, batchSize, nnXLen*nnYLen, numInputChannels,cudaHandles->stream);
      else
        customCudaChannel0ExtractNCHW((const half*)inputBuf, (half*)maskBuf, batchSize, numInputChannels, nnXLen*nnYLen,cudaHandles->stream);
      CUDA_ERR("modelExtractMask",cudaPeekAtLastError());
    }

    fillMaskFloatBufAndMaskSumBuf(cudaHandles,maskBuf,maskFloatBuf,maskSumBuf,usingFP16,batchSize,nnXLen,nnYLen);

    //Don't do any masking if we know the board is exactly the desired size
    if(requireExactNNLen) {
      //Set to NULL to signal downstream that this buf doesn't need to be used
      maskBuf = NULL;
      maskFloatBuf = NULL;
      //The global pooling structures need this no matter what, for normalizing based on this and its sqrt.
      //maskSumBuf = NULL;
    }

    #ifdef DEBUG_INTERMEDIATE_VALUES
    CudaUtils::debugPrint3D(string("Initial bin features"), inputBuf, batchSize, trunk->initialConv->inChannels, nnXLen*nnYLen, inputsUsingNHWC, usingFP16);
    CudaUtils::debugPrint2D(string("Initial global features"), inputGlobalBuf, batchSize, trunk->initialMatMul->inChannels, usingFP16);
    if(trunk->sgfMetadataEncoder != nullptr) {
      assert(inputMetaBuf != NULL);
      CudaUtils::debugPrint2D(string("Initial meta features"), inputMetaBuf, batchSize, trunk->sgfMetadataEncoder->mul1.inChannels, usingFP16);
    }
    #endif

    SizedBuf<void*> trunkBuf(scratch->allocator, scratch->getBufSizeXY(trunk->trunkNumChannels));

    trunk->apply(
      cudaHandles,
      scratch,
      batchSize,
      inputBuf,
      inputGlobalBuf,
      inputMetaBuf,
      maskBuf,
      maskSumBuf,
      trunkBuf.buf,
      workspaceBuf,
      workspaceBytes
    );
    policyHead->apply(
      cudaHandles,
      scratch,
      batchSize,
      maskBuf,
      maskFloatBuf,
      maskSumBuf,
      trunkBuf.buf,
      policyPassBuf,
      policyBuf,
      workspaceBuf,
      workspaceBytes
    );
    valueHead->apply(
      cudaHandles,
      scratch,
      batchSize,
      maskBuf,
      maskSumBuf,
      trunkBuf.buf,
      valueBuf,
      scoreValueBuf,
      ownershipBuf,
      workspaceBuf,
      workspaceBytes
    );
  }

};


//------------------------------------------------------------------------------

struct LoadedModel {
  ModelDesc modelDesc;

  LoadedModel(const string& fileName, const string& expectedSha256) {
    ModelDesc::loadFromFileMaybeGZipped(fileName,modelDesc,expectedSha256);
    modelDesc.applyScale8ToReduceActivations();
  }

  LoadedModel() = delete;
  LoadedModel(const LoadedModel&) = delete;
  LoadedModel& operator=(const LoadedModel&) = delete;
};

LoadedModel* NeuralNet::loadModelFile(const string& file, const string& expectedSha256) {
  LoadedModel* loadedModel = new LoadedModel(file,expectedSha256);
  return loadedModel;
}

void NeuralNet::freeLoadedModel(LoadedModel* loadedModel) {
  delete loadedModel;
}

const ModelDesc& NeuralNet::getModelDesc(const LoadedModel* loadedModel) {
  return loadedModel->modelDesc;
}

//------------------------------------------------------------------------------

struct Buffers {
  //All of these are device pointers

  float* inputBufFloat;
  void* inputBuf;
  float* inputGlobalBufFloat;
  void* inputGlobalBuf;
  float* inputMetaBufFloat;
  void* inputMetaBuf;
  size_t inputBufBytesFloat;
  size_t inputBufBytes;
  size_t inputGlobalBufBytesFloat;
  size_t inputGlobalBufBytes;
  size_t inputMetaBufBytesFloat;
  size_t inputMetaBufBytes;

  float* policyPassBuf;
  size_t policyPassBufBytes;
  float* policyBuf;
  size_t policyBufBytes;

  float* valueBuf;
  size_t valueBufBytes;
  float* scoreValueBuf;
  size_t scoreValueBufBytes;
  void* ownershipBuf;
  size_t ownershipBufBytes;

  void* workspaceBuf;
  size_t workspaceBytes;

  Buffers() = delete;
  Buffers(const Buffers&) = delete;
  Buffers& operator=(const Buffers&) = delete;

  Buffers(CudaHandles* cudaHandles, const Model& m, const ScratchBuffers& scratch) {
    size_t batchXYFloatBytes = (size_t)scratch.batchXYFloatBytes;
    size_t batchFloatBytes = (size_t)scratch.batchFloatBytes;
    size_t batchXYBytes = (size_t)scratch.batchXYBytes;
    size_t batchBytes = (size_t)scratch.batchBytes;

    inputBufBytesFloat = m.numInputChannels * batchXYFloatBytes;
    inputBufBytes = m.numInputChannels * batchXYBytes;
    inputGlobalBufBytesFloat = m.numInputGlobalChannels * batchFloatBytes;
    inputGlobalBufBytes = m.numInputGlobalChannels * batchBytes;
    inputMetaBufBytesFloat = m.numInputMetaChannels * batchFloatBytes;
    inputMetaBufBytes = m.numInputMetaChannels * batchBytes;

    CUDA_ERR("Buffers",cudaMalloc(reinterpret_cast<void**>(&inputBufFloat), inputBufBytesFloat));
    CUDA_ERR("Buffers",cudaMalloc(&inputBuf, inputBufBytes));
    CUDA_ERR("Buffers",cudaMalloc(reinterpret_cast<void**>(&inputGlobalBufFloat), inputGlobalBufBytesFloat));
    CUDA_ERR("Buffers",cudaMalloc(&inputGlobalBuf, inputGlobalBufBytes));
    if(m.numInputMetaChannels > 0) {
      CUDA_ERR("Buffers",cudaMalloc(reinterpret_cast<void**>(&inputMetaBufFloat), inputMetaBufBytesFloat));
      CUDA_ERR("Buffers",cudaMalloc(&inputMetaBuf, inputMetaBufBytes));
    }
    else {
      inputMetaBufFloat = NULL;
      inputMetaBuf = NULL;
    }

    if(m.modelVersion >= 17)
      testAssert(m.policyHead->p2Channels == 2 || m.policyHead->p2Channels == 4);
    else if(m.modelVersion >= 16)
      testAssert(m.policyHead->p2Channels == 4);
    else if(m.modelVersion >= 12)
      testAssert(m.policyHead->p2Channels == 2);
    else
      testAssert(m.policyHead->p2Channels == 1);

    policyPassBufBytes = m.policyHead->p2Channels * batchFloatBytes;
    CUDA_ERR("Buffers",cudaMalloc(reinterpret_cast<void**>(&policyPassBuf), policyPassBufBytes));
    policyBufBytes = m.policyHead->p2Channels * batchXYFloatBytes;
    CUDA_ERR("Buffers",cudaMalloc(reinterpret_cast<void**>(&policyBuf), policyBufBytes));

    valueBufBytes = m.valueHead->valueChannels * batchFloatBytes;
    CUDA_ERR("Buffers",cudaMalloc(reinterpret_cast<void**>(&valueBuf), valueBufBytes));

    scoreValueBufBytes = m.valueHead->scoreValueChannels * batchFloatBytes;
    CUDA_ERR("Buffers",cudaMalloc(reinterpret_cast<void**>(&scoreValueBuf), scoreValueBufBytes));

    //This buf is used for both an intermdiate fp16 result in fp16 mode, and ALSO the final fp32 output, so always must be fp32-sized
    ownershipBufBytes = m.valueHead->ownershipChannels * batchXYFloatBytes;
    CUDA_ERR("Buffers",cudaMalloc(&ownershipBuf, ownershipBufBytes));

    //In theory the requiredWorkspaceBytes calls could give us values non-monotone in batch size
    //such as if the convolution algorithm changes between batch size 1 and larger.
    //So we call it for all the batch sizes.
    size_t bytes = 0;
    size_t b;
    for(int batchSize = 1; batchSize <= m.maxBatchSize; batchSize++) {
      b = m.requiredWorkspaceBytes(cudaHandles,batchSize);
      bytes = std::max(bytes,b);
    }

    CUDA_ERR("Buffers",cudaMalloc(&workspaceBuf, bytes));
    workspaceBytes = bytes;
  }

  ~Buffers() {
    (void)cudaFree(inputBufFloat);
    (void)cudaFree(inputBuf);
    (void)cudaFree(inputGlobalBufFloat);
    (void)cudaFree(inputGlobalBuf);
    if(inputMetaBufFloat != NULL)
      (void)cudaFree(inputMetaBufFloat);
    if(inputMetaBuf != NULL)
      (void)cudaFree(inputMetaBuf);

    (void)cudaFree(policyPassBuf);
    (void)cudaFree(policyBuf);

    (void)cudaFree(valueBuf);
    (void)cudaFree(scoreValueBuf);
    (void)cudaFree(ownershipBuf);

    (void)cudaFree(workspaceBuf);
  }

};

//------------------------------------------------------------------------------

#ifdef KATAGO_GPU_CUDA
struct ComputeContext {
  int nnXLen;
  int nnYLen;
  enabled_t useFP16Mode;
  enabled_t useNHWCMode;
  // If true, skip the cudnn graph SDPA path entirely and always use the custom attention kernel.
  bool cudaDisableGraphSDPA;
  // Whether 1x1 NHWC convs use the cuBLAS GEMM path. Auto = matmul iff FP16.
  enabled_t use1x1MatmulMode;
  // Tensor-core mma flash attention kernel. Auto = on for compute capability >= 8.0.
  enabled_t useMmaAttentionMode;
  // Share large read-only weight buffers between the handles of NN server threads on the
  // same GPU, roughly halving weight memory for two-thread configs. Off by default because
  // the throughput effect is hardware-dependent and can be slightly negative.
  bool shareModelWeights;
  // Fuse transformer masked residual adds into the next block's pre-norm kernel. Numerically
  // identical either way. On by default, kept as an escape hatch for debugging and benchmarking.
  bool fuseResidualNorm;
  // Combined QKV projection GEMM (see CudaHandles::combineQKV). On by default, escape hatch
  // for benchmarking.
  bool combineQKV;
  // CUTLASS fused FFN (dual GEMM + SwiGLU epilogue). Auto = on when the build has CUTLASS and
  // the GPU and model are eligible (FP16, sm_80+).
  enabled_t useFusedFFNMode;
};
#else  // KATAGO_GPU_HIP
struct ComputeContext {
  int nnXLen;
  int nnYLen;
  enabled_t useFP16Mode;
  enabled_t useNHWCMode;
  bool disableFusedAttention;
  // See the CUDA variant's shareModelWeights.
  bool shareModelWeights;
  // See the CUDA variant.
  bool fuseResidualNorm;
  // See CudaHandles::use1x1MatmulMode.
  enabled_t use1x1MatmulMode;
};
#endif

#ifdef KATAGO_GPU_CUDA
ComputeContext* NeuralNet::createComputeContext(
  const std::vector<int>& gpuIdxs,
  Logger* logger,
  int nnXLen,
  int nnYLen,
  const string& homeDataDirOverride,
  enabled_t useFP16Mode,
  const LoadedModel* loadedModel,
  ConfigParser& cfg
) {
  (void)gpuIdxs;
  (void)logger;
  (void)homeDataDirOverride;
  (void)loadedModel;

  ComputeContext* context = new ComputeContext();
  context->nnXLen = nnXLen;
  context->nnYLen = nnYLen;
  context->useFP16Mode = useFP16Mode;

  // NHWC layout is a CUDA-specific option read directly off of cfg. Auto means "NHWC if FP16" (see below).
  context->useNHWCMode =
    cfg.contains("cudaUseNHWC") ? cfg.getEnabled("cudaUseNHWC") : enabled_t::Auto;
  context->cudaDisableGraphSDPA =
    cfg.contains("cudaDisableGraphSDPA") ? cfg.getBool("cudaDisableGraphSDPA") : false;
  context->use1x1MatmulMode =
    cfg.contains("cudaUse1x1Matmul") ? cfg.getEnabled("cudaUse1x1Matmul") : enabled_t::Auto;
  context->useMmaAttentionMode =
    cfg.contains("cudaUseMmaAttention") ? cfg.getEnabled("cudaUseMmaAttention") : enabled_t::Auto;
  context->shareModelWeights =
    cfg.contains("cudaShareModelWeights") ? cfg.getBool("cudaShareModelWeights") : false;
  context->fuseResidualNorm =
    cfg.contains("cudaFuseResidualNorm") ? cfg.getBool("cudaFuseResidualNorm") : true;
  context->combineQKV =
    cfg.contains("cudaCombineQKV") ? cfg.getBool("cudaCombineQKV") : true;
  context->useFusedFFNMode =
    cfg.contains("cudaUseFusedFFN") ? cfg.getEnabled("cudaUseFusedFFN") : enabled_t::Auto;
  return context;
}
#else  // KATAGO_GPU_HIP
ComputeContext* NeuralNet::createComputeContext(
  const std::vector<int>& gpuIdxs,
  Logger* logger,
  int nnXLen,
  int nnYLen,
  const string& homeDataDirOverride,
  enabled_t useFP16Mode,
  const LoadedModel* loadedModel,
  ConfigParser& cfg
) {
  (void)gpuIdxs;
  (void)logger;
  (void)homeDataDirOverride;
  (void)loadedModel;

  // ROCm-specific NHWC override, read directly off cfg (mirrors cudaUseNHWC in the CUDA backend).
  enabled_t useNHWCMode =
    cfg.contains("rocmUseNHWC") ? cfg.getEnabled("rocmUseNHWC") : enabled_t::Auto;
  // Disables the optional CK FMHA fused attention path (see KATAGO_ROCM_HAS_CK_FMHA), mirroring
  // the CUDA backend's cudaDisableGraphSDPA. Only meaningful for transformer models.
  bool disableFusedAttention =
    cfg.contains("rocmDisableFusedAttention") ? cfg.getBool("rocmDisableFusedAttention") : false;
  // Mirrors the CUDA backend's cudaUse1x1Matmul, but matters more here, since MIOpen's NHWC conv
  // solver coverage is weak on some architectures where hipBLAS is fast.
  enabled_t use1x1MatmulMode =
    cfg.contains("rocmUse1x1Matmul") ? cfg.getEnabled("rocmUse1x1Matmul") : enabled_t::Auto;

  ComputeContext* context = new ComputeContext();
  context->nnXLen = nnXLen;
  context->nnYLen = nnYLen;
  context->useFP16Mode = useFP16Mode;
  context->useNHWCMode = useNHWCMode;
  context->disableFusedAttention = disableFusedAttention;
  context->shareModelWeights =
    cfg.contains("rocmShareModelWeights") ? cfg.getBool("rocmShareModelWeights") : false;
  context->fuseResidualNorm =
    cfg.contains("rocmFuseResidualNorm") ? cfg.getBool("rocmFuseResidualNorm") : true;
  context->use1x1MatmulMode = use1x1MatmulMode;
  return context;
}
#endif

void NeuralNet::freeComputeContext(ComputeContext* computeContext) {
  delete computeContext;
}

//------------------------------------------------------------------------------

struct ComputeHandle {
  std::unique_ptr<CudaHandles> cudaHandles;
  std::unique_ptr<Model> model;
  std::unique_ptr<ScratchBuffers> scratch;
  std::unique_ptr<Buffers> buffers;
  const bool usingFP16;
  const int nnXLen;
  const int nnYLen;
  const bool requireExactNNLen;
  const bool inputsUseNHWC;
  const bool usingNHWC;

#ifdef KATAGO_GPU_CUDA
  ComputeHandle(
    const ComputeContext* context,
    const LoadedModel* loadedModel,
    Logger* logger,
    int majorComputeCapability,
    int minorComputeCapability,
    int maxBatchSize,
    bool requireExactNNLen_,
    bool inputsUseNHWC_,
    bool useFP16,
    bool useNHWC
  ) :
    usingFP16(useFP16),
    nnXLen(context->nnXLen),
    nnYLen(context->nnYLen),
    requireExactNNLen(requireExactNNLen_),
    inputsUseNHWC(inputsUseNHWC_),
    usingNHWC(useNHWC)
  {
    cudaHandles = std::make_unique<CudaHandles>(majorComputeCapability,minorComputeCapability);
    // Must be set before building the model: ConvLayer reads it at construction to pick the 1x1 conv path,
    // and the weight-owning layers read shareModelWeights at construction.
    cudaHandles->logger = logger;
    cudaHandles->cudaDisableGraphSDPA = context->cudaDisableGraphSDPA;
    cudaHandles->use1x1MatmulMode = context->use1x1MatmulMode;
    cudaHandles->shareModelWeights = context->shareModelWeights;
    cudaHandles->fuseResidualNorm = context->fuseResidualNorm;
    cudaHandles->combineQKV = context->combineQKV;
    // Decided before Model construction: TransformerAttentionBlock commits to the combined
    // QKV weight layout at construction when the mma attention path will handle its shapes.
    {
      bool wantMmaAttention =
        context->useMmaAttentionMode == enabled_t::True ||
        (context->useMmaAttentionMode == enabled_t::Auto && majorComputeCapability >= 8);
      cudaHandles->mmaAttentionEnabled = wantMmaAttention && customCudaFlashAttentionMmaSupported();
      if(wantMmaAttention && !cudaHandles->mmaAttentionEnabled) {
        // The probe failing on a compute-capability-8.0+ GPU means the kernel did not survive
        // compilation or JIT for this device (e.g. this binary lacks both SASS and JIT-able PTX
        // for it). Losing the mma path is a large silent slowdown on some GPUs, so be loud.
        if(context->useMmaAttentionMode == enabled_t::True)
          throw StringError("Cuda backend: cudaUseMmaAttention=true but the mma flash attention kernel is not functional on this GPU/driver/build");
        if(logger != NULL)
          logger->write(
            "WARNING: Cuda backend: mma flash attention kernel not functional on this GPU/driver/build, transformer models will fall back to slower attention paths");
      }
    }
#ifdef USE_CUTLASS_FUSED_FFN
    // Decided before Model construction because TransformerFFNBlock commits to a weight layout
    // based on it. The probe runs a real tiny dual GEMM, guarding the same empty-stub JIT
    // hazard as the mma kernel plus any launch-time failure.
    {
      bool wantFusedFFN =
        context->useFusedFFNMode == enabled_t::True ||
        (context->useFusedFFNMode == enabled_t::Auto && majorComputeCapability >= 8);
      bool fusedFFNProbeOk = wantFusedFFN && useFP16 && CudaFusedFFN::supportedOnCurrentDevice();
      cudaHandles->fusedFFNEnabled = fusedFFNProbeOk;
      if(context->useFusedFFNMode == enabled_t::True && !cudaHandles->fusedFFNEnabled)
        throw StringError(
          "Cuda backend: cudaUseFusedFFN=true but the fused FFN kernel is not usable here (requires FP16 and a functional sm_80+ build)");
      // Same loudness policy as the mma attention probe above: losing this kernel on a GPU that
      // should support it is a large silent slowdown, so say so.
      if(wantFusedFFN && useFP16 && !fusedFFNProbeOk && logger != NULL)
        logger->write(
          "WARNING: Cuda backend: CUTLASS fused FFN kernel not functional on this GPU/driver/build, transformer models will use the slower unfused FFN path");
    }
#else
    if(context->useFusedFFNMode == enabled_t::True)
      throw StringError("Cuda backend: cudaUseFusedFFN=true but this build was compiled without CUTLASS (external/cutlass missing at build time)");
#endif
    model = std::make_unique<Model>(
      cudaHandles.get(), &(loadedModel->modelDesc), maxBatchSize,
      nnXLen, nnYLen, inputsUseNHWC, useFP16, useNHWC
    );
    scratch = std::make_unique<ScratchBuffers>(maxBatchSize, nnXLen, nnYLen, useFP16);
    buffers = std::make_unique<Buffers>(cudaHandles.get(), *model, *scratch);

    //Synchronize after creating buffers and copying all the weights, just in case
    CUDA_ERR("ComputeHandle", cudaDeviceSynchronize());
  }
#else  // KATAGO_GPU_HIP
  ComputeHandle(
    const ComputeContext* context,
    const LoadedModel* loadedModel,
    Logger* logger,
    int majorComputeCapability,
    int minorComputeCapability,
    bool ckFmhaArchSupported,
    bool preferMiopen1x1Conv,
    int maxBatchSize,
    bool requireExactNNLen_,
    bool inputsUseNHWC_,
    bool useFP16,
    bool useNHWC
  ) :
    usingFP16(useFP16),
    nnXLen(context->nnXLen),
    nnYLen(context->nnYLen),
    requireExactNNLen(requireExactNNLen_),
    inputsUseNHWC(inputsUseNHWC_),
    usingNHWC(useNHWC)
  {
    cudaHandles = std::make_unique<CudaHandles>(
      majorComputeCapability,minorComputeCapability,logger,ckFmhaArchSupported,context->disableFusedAttention);
    // Must be set before building the model: the weight-owning layers and ConvLayer read them
    // at construction.
    cudaHandles->shareModelWeights = context->shareModelWeights;
    cudaHandles->fuseResidualNorm = context->fuseResidualNorm;
    // Resolved here rather than in ConvLayer, which has no device properties. On CDNA, MIOpen
    // sends NHWC 1x1 convs to the same MFMA implicit-GEMM solvers that make NHWC the preferred
    // layout there, and measured slightly faster than the GEMM. Elsewhere its NHWC coverage is
    // weak and the GEMM is much faster. True and False still force the choice.
    cudaHandles->use1x1MatmulMode =
      (context->use1x1MatmulMode == enabled_t::Auto && preferMiopen1x1Conv)
      ? enabled_t::False : context->use1x1MatmulMode;
    model = std::make_unique<Model>(
      cudaHandles.get(), &(loadedModel->modelDesc), maxBatchSize,
      nnXLen, nnYLen, inputsUseNHWC, useFP16, useNHWC
    );
    scratch = std::make_unique<ScratchBuffers>(maxBatchSize, nnXLen, nnYLen, useFP16);
    buffers = std::make_unique<Buffers>(cudaHandles.get(), *model, *scratch);

    //Synchronize after creating buffers and copying all the weights, just in case
    CUDA_ERR("ComputeHandle", hipDeviceSynchronize());
  }
#endif
  ~ComputeHandle() {
  }

  ComputeHandle() = delete;
  ComputeHandle(const ComputeHandle&) = delete;
  ComputeHandle& operator=(const ComputeHandle&) = delete;
};

#ifdef KATAGO_GPU_CUDA
ComputeHandle* NeuralNet::createComputeHandle(
  ComputeContext* context,
  const LoadedModel* loadedModel,
  Logger* logger,
  int maxBatchSize,
  bool requireExactNNLen,
  bool inputsUseNHWC,
  int gpuIdxForThisThread,
  int serverThreadIdx
) {
  //Use whatever CUDA believes GPU 0 to be.
  if(gpuIdxForThisThread == -1)
    gpuIdxForThisThread = 0;

  CUDA_ERR("createComputeHandle",cudaSetDevice(gpuIdxForThisThread));

  cudaDeviceProp prop;
  CUDA_ERR("createComputeHandle",cudaGetDeviceProperties(&prop,gpuIdxForThisThread));

  bool useFP16 = false;
  bool useNHWC = false;
  //Old GPUs - use FP32 and explicitly fail if FP16 enabled
  if(prop.major < 5 || (prop.major == 5 && prop.minor < 3)) {
    if(context->useFP16Mode == enabled_t::True)
      throw StringError("Cuda device versions below 5.3 do not support useFP16=true");
    if(context->useNHWCMode == enabled_t::True)
      useNHWC = true;
  }
  //In theory these GPUs support FP16, so allow if the user wants.
  else if(prop.major < 6) {
    if(context->useFP16Mode == enabled_t::True)
      useFP16 = true;
    if(context->useNHWCMode == enabled_t::True)
      useNHWC = true;
  }
  //On Pascal architecture, default to using FP16 operations
  //Actually, just use FP32 - there's a risk that on certain cards this might just be a lot worse.
  //A user manually fine-tuning for performance can just enable it themselves if they know how.
  else if(prop.major < 7) {
    if(context->useFP16Mode == enabled_t::True)
      useFP16 = true;
    if(context->useNHWCMode == enabled_t::True)
      useNHWC = true;
  }
  //On Volta and higher, use FP16 and NHWC together because we have tensor cores.
  else {
    if(context->useFP16Mode == enabled_t::True || context->useFP16Mode == enabled_t::Auto)
      useFP16 = true;
    if(context->useNHWCMode == enabled_t::True || (context->useNHWCMode == enabled_t::Auto && useFP16))
      useNHWC = true;
  }

  //The CUDA transformer block implementation only supports NHWC (its channel projections, RoPE, and
  //attention all assume the channel dim is contiguous per position). Unlike convnets, NHWC here is not
  //tied to FP16/tensor-cores - the transformer kernels have FP32 paths too. So for transformer models
  //force NHWC regardless of the FP16/NHWC-mode decision above, otherwise FP32 (or NHWC=false) would hit
  //the "NCHW layout not supported" throw. No effect on convnets.
  if(!useNHWC && loadedModel->modelDesc.trunk.hasAnyTransformerBlocks()) {
    if(context->useNHWCMode == enabled_t::False)
      throw StringError("CUDA backend: transformer models require NHWC, but cudaUseNHWC=false was set");
    useNHWC = true;
  }

  if(logger != NULL) {
    logger->write(
      "Cuda backend thread " + Global::intToString(serverThreadIdx) + ": Found GPU " + string(prop.name)
      + " memory " + Global::uint64ToString(prop.totalGlobalMem)
      + " compute capability major " + Global::intToString(prop.major)
      + " minor " + Global::intToString(prop.minor)
    );
    logger->write(
      "Cuda backend thread " + Global::intToString(serverThreadIdx) + ": Model version " + Global::intToString(loadedModel->modelDesc.modelVersion) +
      " useFP16 = " + Global::boolToString(useFP16) +
      " useNHWC = " + Global::boolToString(useNHWC)
    );
    logger->write(
      "Cuda backend thread " + Global::intToString(serverThreadIdx) + ": Model name: " + loadedModel->modelDesc.name +
      " (" + loadedModel->modelDesc.getShortInfoString() + ")"
    );
  }

  // A persisting-L2 access policy window on the trunk residual buffer was tried here and
  // removed (see git history): it measured -10% to -30% on H100 and slightly negative on
  // 4090/Blackwell convnets, since the L2 set-aside starves everything else.
  ComputeHandle* gpuHandle = new ComputeHandle(
    context,loadedModel,logger,prop.major,prop.minor,maxBatchSize,requireExactNNLen,inputsUseNHWC,useFP16,useNHWC
  );
  return gpuHandle;
}
#else  // KATAGO_GPU_HIP
ComputeHandle* NeuralNet::createComputeHandle(
  ComputeContext* context,
  const LoadedModel* loadedModel,
  Logger* logger,
  int maxBatchSize,
  bool requireExactNNLen,
  bool inputsUseNHWC,
  int gpuIdxForThisThread,
  int serverThreadIdx
) {
  //Use whatever HIP believes GPU 0 to be.
  if(gpuIdxForThisThread == -1)
    gpuIdxForThisThread = 0;

  CUDA_ERR("createComputeHandle",hipSetDevice(gpuIdxForThisThread));

  hipDeviceProp_t prop;
  CUDA_ERR("createComputeHandle",hipGetDeviceProperties(&prop,gpuIdxForThisThread));

  bool useFP16 = false;
  bool useNHWC = false;
  if(context->useFP16Mode == enabled_t::True || context->useFP16Mode == enabled_t::Auto)
    useFP16 = true;
  if(context->useNHWCMode == enabled_t::True)
    useNHWC = true;

  // Every arch in the default CMAKE_HIP_ARCHITECTURES list supports packed FP16, so FP16 "Auto"
  // enables it unconditionally. But if this build's arch list wasn't recognized by the CMake
  // HIP_SUPPORTS_FP16 check, the half-precision kernels compiled to empty stubs - detect that and
  // refuse (or auto-disable) FP16 loudly rather than silently computing garbage.
  if(useFP16 && !customCudaFp16KernelsCompiled()) {
    if(context->useFP16Mode == enabled_t::True)
      throw StringError("rocmUseFP16 = true, but this build was compiled without FP16 kernel support (HIP_SUPPORTS_FP16 was not defined; check the GPU architecture list the build used)");
    useFP16 = false;
    if(logger != NULL)
      logger->write("ROCm backend: this build was compiled without FP16 kernel support; using FP32");
  }

  // Mirror the CUDA backend's tensor-core rule: on archs whose matrix instructions make NHWC
  // convolutions the fast path, NHWC "Auto" turns NHWC on together with FP16. Decided after the
  // FP16-kernel check above so a build that just downgraded to FP32 does not also switch layouts.
  if(context->useNHWCMode == enabled_t::Auto && useFP16 && isNhwcFp16PreferredArch(prop))
    useNHWC = true;

  // The transformer block implementation (attention/RoPE/FFN) only supports NHWC, since its channel
  // projections assume the channel dim is contiguous per spatial position. Force NHWC for transformer
  // models regardless of the useNHWCMode decision above, matching the CUDA backend's behavior.
  if(!useNHWC && loadedModel->modelDesc.trunk.hasAnyTransformerBlocks()) {
    if(context->useNHWCMode == enabled_t::False)
      throw StringError("ROCm backend: transformer models require NHWC, but rocmUseNHWC=false was set");
    useNHWC = true;
  }

  if(logger != NULL) {
    logger->write(
      "ROCm backend thread " + Global::intToString(serverThreadIdx) + ": Found GPU " + string(prop.name)
      + " memory " + Global::uint64ToString(prop.totalGlobalMem)
      + " compute capability major " + Global::intToString(prop.major)
      + " minor " + Global::intToString(prop.minor)
    );
    logger->write(
      "ROCm backend thread " + Global::intToString(serverThreadIdx) + ": Model version " + Global::intToString(loadedModel->modelDesc.modelVersion) +
      " useFP16 = " + Global::boolToString(useFP16) +
      " useNHWC = " + Global::boolToString(useNHWC)
    );
    logger->write(
      "ROCm backend thread " + Global::intToString(serverThreadIdx) + ": Model name: " + loadedModel->modelDesc.name +
      " (" + loadedModel->modelDesc.getShortInfoString() + ")"
    );
    logger->write(
      "MIOpen finding convolution algorithms for GPU " + string(prop.name) + ". This may take a while, please wait......"
    );
  }

  // Reusing the NHWC predicate: the archs that prefer MIOpen's NHWC 1x1 conv over a GEMM are the
  // same ones that prefer NHWC at all.
  ComputeHandle* gpuHandle = new ComputeHandle(
    context,loadedModel,logger,prop.major,prop.minor,isCkFmhaSupportedArch(prop),
    isNhwcFp16PreferredArch(prop),
    maxBatchSize,requireExactNNLen,inputsUseNHWC,useFP16,useNHWC
  );
  return gpuHandle;
}
#endif

void NeuralNet::freeComputeHandle(ComputeHandle* gpuHandle) {
  delete gpuHandle;
}

bool NeuralNet::isUsingFP16(const ComputeHandle* handle) {
  return handle->usingFP16;
}

#ifdef KATAGO_GPU_CUDA
bool NeuralNet::setIsWarmup(const ComputeHandle* handle, bool isWarmup) {
  CudaHandles* cudaHandles = handle->cudaHandles.get();
  bool prev = cudaHandles->isWarmup;
  cudaHandles->isWarmup = isWarmup;
  return prev;
}
#else  // KATAGO_GPU_HIP
bool NeuralNet::setIsWarmup(const ComputeHandle* handle, bool isWarmup) {
  (void)handle;
  (void)isWarmup;
  return false;
}
#endif

//------------------------------------------------------------------------------

void NeuralNet::printDevices() {
  int numDevices = 0;
  //Best-effort listing, so a failure here prints nothing rather than aborting. The errors are
  //checked rather than discarded because prop is left uninitialized on failure, and printing
  //prop.name would then read uninitialized memory.
  if(cudaGetDeviceCount(&numDevices) != cudaSuccess)
    return;
  for(int i = 0; i<numDevices; i++) {
    cudaDeviceProp prop;
    if(cudaGetDeviceProperties(&prop, i) != cudaSuccess)
      continue;
    cout << "Found " KATAGO_GPU_BACKEND_NAME " device " << i << ": " << prop.name << endl;
  }
}

std::string NeuralNet::getRuntimeBackendDetail(ConfigParser& cfg) {
  (void)cfg;
  return std::string();
}


//------------------------------------------------------------------------------

struct InputBuffers {
  int maxBatchSize;

  size_t singleInputElts;
  size_t singleInputBytes;
  size_t singleInputGlobalElts;
  size_t singleInputGlobalBytes;
  size_t singleInputMetaElts;
  size_t singleInputMetaBytes;
  size_t singlePolicyPassResultElts;
  size_t singlePolicyPassResultBytes;
  size_t singlePolicyResultElts;
  size_t singlePolicyResultBytes;
  size_t singleValueResultElts;
  size_t singleValueResultBytes;
  size_t singleScoreValueResultElts;
  size_t singleScoreValueResultBytes;
  size_t singleOwnershipResultElts;
  size_t singleOwnershipResultBytes;

  size_t userInputBufferBytes;
  size_t userInputGlobalBufferBytes;
  size_t userInputMetaBufferBytes;
  size_t policyPassResultBufferBytes;
  size_t policyResultBufferBytes;
  size_t valueResultBufferBytes;
  size_t scoreValueResultBufferBytes;
  size_t ownershipResultBufferBytes;

  float* userInputBuffer; //Host pointer
  float* userInputGlobalBuffer; //Host pointer
  float* userInputMetaBuffer; //Host pointer

  float* policyPassResults; //Host pointer
  float* policyResults; //Host pointer
  float* valueResults; //Host pointer
  float* scoreValueResults; //Host pointer
  float* ownershipResults; //Host pointer

  InputBuffers(const LoadedModel* loadedModel, int maxBatchSz, int nnXLen, int nnYLen) {
    const ModelDesc& m = loadedModel->modelDesc;

    maxBatchSize = maxBatchSz;
    singleInputElts = (size_t)m.numInputChannels * nnXLen * nnYLen;
    singleInputBytes = (size_t)m.numInputChannels * nnXLen * nnYLen * sizeof(float);
    singleInputGlobalElts = (size_t)m.numInputGlobalChannels;
    singleInputGlobalBytes = (size_t)m.numInputGlobalChannels * sizeof(float);
    singleInputMetaElts = (size_t)m.numInputMetaChannels;
    singleInputMetaBytes = (size_t)m.numInputMetaChannels * sizeof(float);
    singlePolicyPassResultElts = (size_t)(m.numPolicyChannels);
    singlePolicyPassResultBytes = (size_t)(m.numPolicyChannels) * sizeof(float);
    singlePolicyResultElts = (size_t)(m.numPolicyChannels * nnXLen * nnYLen);
    singlePolicyResultBytes = (size_t)(m.numPolicyChannels * nnXLen * nnYLen) * sizeof(float);
    singleValueResultElts = (size_t)m.numValueChannels;
    singleValueResultBytes = (size_t)m.numValueChannels * sizeof(float);
    singleScoreValueResultElts = (size_t)m.numScoreValueChannels;
    singleScoreValueResultBytes = (size_t)m.numScoreValueChannels * sizeof(float);
    singleOwnershipResultElts = (size_t)m.numOwnershipChannels * nnXLen * nnYLen;
    singleOwnershipResultBytes = (size_t)m.numOwnershipChannels * nnXLen * nnYLen * sizeof(float);

    testAssert(NNModelVersion::getNumSpatialFeatures(m.modelVersion) == m.numInputChannels);
    testAssert(NNModelVersion::getNumGlobalFeatures(m.modelVersion) == m.numInputGlobalChannels);
    if(m.numInputMetaChannels > 0) {
      testAssert(SGFMetadata::METADATA_INPUT_NUM_CHANNELS == m.numInputMetaChannels);
    }

    userInputBufferBytes = (size_t)m.numInputChannels * maxBatchSize * nnXLen * nnYLen * sizeof(float);
    userInputGlobalBufferBytes = (size_t)m.numInputGlobalChannels * maxBatchSize * sizeof(float);
    userInputMetaBufferBytes = (size_t)m.numInputMetaChannels * maxBatchSize * sizeof(float);
    policyPassResultBufferBytes = (size_t)maxBatchSize * m.numPolicyChannels * sizeof(float);
    policyResultBufferBytes = (size_t)maxBatchSize * m.numPolicyChannels * nnXLen * nnYLen * sizeof(float);
    valueResultBufferBytes = (size_t)maxBatchSize * m.numValueChannels * sizeof(float);
    scoreValueResultBufferBytes = (size_t)maxBatchSize * m.numScoreValueChannels * sizeof(float);
    ownershipResultBufferBytes = (size_t)maxBatchSize * nnXLen * nnYLen * m.numOwnershipChannels * sizeof(float);

    // Pinned host memory, so the async transfers in getOutput are true DMA and overlap between
    // the streams of different NN server threads.
    userInputBuffer = NULL;
    userInputGlobalBuffer = NULL;
    userInputMetaBuffer = NULL;
    policyPassResults = NULL;
    policyResults = NULL;
    valueResults = NULL;
    scoreValueResults = NULL;
    ownershipResults = NULL;
    try {
      userInputBuffer = mallocPinned((size_t)m.numInputChannels * maxBatchSize * nnXLen * nnYLen);
      userInputGlobalBuffer = mallocPinned((size_t)m.numInputGlobalChannels * maxBatchSize);
      if(m.numInputMetaChannels > 0)
        userInputMetaBuffer = mallocPinned((size_t)m.numInputMetaChannels * maxBatchSize);

      policyPassResults = mallocPinned((size_t)maxBatchSize * m.numPolicyChannels);
      policyResults = mallocPinned((size_t)maxBatchSize * m.numPolicyChannels * nnXLen * nnYLen);
      valueResults = mallocPinned((size_t)maxBatchSize * m.numValueChannels);

      scoreValueResults = mallocPinned((size_t)maxBatchSize * m.numScoreValueChannels);
      ownershipResults = mallocPinned((size_t)maxBatchSize * nnXLen * nnYLen * m.numOwnershipChannels);
    }
    catch(...) {
      freeBuffers();
      throw;
    }
  }

  //Tracks each allocation's flavor so freeBuffers releases it with the matching call.
  std::vector<void*> pinnedPtrs;
  std::vector<void*> pageablePtrs;

  float* mallocPinned(size_t numFloats) {
    void* buf = NULL;
    if(cudaHostAlloc(&buf, numFloats * sizeof(float), cudaHostAllocPortable) == cudaSuccess) {
      pinnedPtrs.push_back(buf);
      return (float*)buf;
    }
    //Clear the sticky error so a later cudaPeekAtLastError doesn't see this failure.
    (void)cudaGetLastError();
    //Pinned memory is a limited resource (e.g. many KataGo processes on one machine), so degrade
    //to pageable memory rather than failing. Transfers still work, just without DMA overlap.
    static std::atomic<bool> warned(false);
    if(!warned.exchange(true))
      std::cerr << "WARNING: pinned host memory allocation failed, falling back to pageable memory for NN input/output buffers (GPU transfers may overlap less)" << std::endl;
    buf = malloc(numFloats * sizeof(float));
    if(buf == NULL)
      throw StringError("InputBuffers: out of host memory");
    pageablePtrs.push_back(buf);
    return (float*)buf;
  }

  void freeBuffers() {
    for(void* p: pinnedPtrs)
      (void)cudaFreeHost(p);
    for(void* p: pageablePtrs)
      free(p);
    pinnedPtrs.clear();
    pageablePtrs.clear();
  }

  ~InputBuffers() {
    freeBuffers();
  }

  InputBuffers() = delete;
  InputBuffers(const InputBuffers&) = delete;
  InputBuffers& operator=(const InputBuffers&) = delete;

};

InputBuffers* NeuralNet::createInputBuffers(const LoadedModel* loadedModel, int maxBatchSize, int nnXLen, int nnYLen) {
  return new InputBuffers(loadedModel,maxBatchSize,nnXLen,nnYLen);
}
void NeuralNet::freeInputBuffers(InputBuffers* inputBuffers) {
  delete inputBuffers;
}

//---------------------------------------------------------------------------------------


void NeuralNet::getOutput(
  ComputeHandle* gpuHandle,
  InputBuffers* inputBuffers,
  int numBatchEltsFilled,
  NNResultBuf** inputBufs,
  vector<NNOutput*>& outputs
) {
  //testAssert rather than assert: these guard the sizes of every device memcpy below, so an
  //oversized batch must fail loudly in release builds too rather than overflow GPU buffers.
  testAssert(numBatchEltsFilled <= inputBuffers->maxBatchSize);
  testAssert(numBatchEltsFilled > 0);
  const int batchSize = numBatchEltsFilled;
  const int nnXLen = gpuHandle->nnXLen;
  const int nnYLen = gpuHandle->nnYLen;
  const int modelVersion = gpuHandle->model->modelVersion;

  const int numSpatialFeatures = NNModelVersion::getNumSpatialFeatures(modelVersion);
  const int numGlobalFeatures = NNModelVersion::getNumGlobalFeatures(modelVersion);
  const int numMetaFeatures = inputBuffers->singleInputMetaElts;
  assert(numSpatialFeatures == gpuHandle->model->numInputChannels);
  assert(numSpatialFeatures * nnXLen * nnYLen == inputBuffers->singleInputElts);
  assert(numGlobalFeatures == inputBuffers->singleInputGlobalElts);
  const int numPolicyChannels = gpuHandle->model->numPolicyChannels;

  for(int nIdx = 0; nIdx<batchSize; nIdx++) {
    float* rowSpatialInput = inputBuffers->userInputBuffer + (inputBuffers->singleInputElts * nIdx);
    float* rowGlobalInput = inputBuffers->userInputGlobalBuffer + (inputBuffers->singleInputGlobalElts * nIdx);
    float* rowMetaInput = inputBuffers->userInputMetaBuffer + (inputBuffers->singleInputMetaElts * nIdx);

    const float* rowGlobal = inputBufs[nIdx]->rowGlobalBuf.data();
    const float* rowSpatial = inputBufs[nIdx]->rowSpatialBuf.data();
    const float* rowMeta = inputBufs[nIdx]->rowMetaBuf.data();
    bool hasRowMeta = inputBufs[nIdx]->hasRowMeta;
    std::copy(rowGlobal,rowGlobal+numGlobalFeatures,rowGlobalInput);
    if(numMetaFeatures > 0) {
      testAssert(rowMeta != NULL);
      testAssert(hasRowMeta);
      std::copy(rowMeta,rowMeta+numMetaFeatures,rowMetaInput);
    }
    else {
      testAssert(!hasRowMeta);
    }
    SymmetryHelpers::copyInputsWithSymmetry(rowSpatial, rowSpatialInput, 1, nnYLen, nnXLen, numSpatialFeatures, gpuHandle->inputsUseNHWC, inputBufs[nIdx]->symmetry);
  }

  Buffers* buffers = gpuHandle->buffers.get();
  ScratchBuffers* scratch = gpuHandle->scratch.get();
  CudaHandles* cudaHandles = gpuHandle->cudaHandles.get();
  cudaStream_t stream = cudaHandles->stream;

  if(!gpuHandle->usingFP16) {
    assert(inputBuffers->userInputBufferBytes == buffers->inputBufBytes);
    assert(inputBuffers->userInputGlobalBufferBytes == buffers->inputGlobalBufBytes);
    assert(inputBuffers->userInputMetaBufferBytes == buffers->inputMetaBufBytes);
    assert(inputBuffers->policyPassResultBufferBytes == buffers->policyPassBufBytes);
    assert(inputBuffers->policyResultBufferBytes == buffers->policyBufBytes);
    assert(inputBuffers->valueResultBufferBytes == buffers->valueBufBytes);
    assert(inputBuffers->singleInputBytes == inputBuffers->singleInputElts*4);
    assert(inputBuffers->singleInputGlobalBytes == inputBuffers->singleInputGlobalElts*4);
    assert(inputBuffers->singleInputMetaBytes == inputBuffers->singleInputMetaElts*4);
    assert(inputBuffers->singlePolicyPassResultElts == numPolicyChannels);
    assert(inputBuffers->singlePolicyPassResultBytes == numPolicyChannels * sizeof(float));
    assert(inputBuffers->singlePolicyResultElts == numPolicyChannels*nnXLen*nnYLen);
    assert(inputBuffers->singlePolicyResultBytes == numPolicyChannels*nnXLen*nnYLen * sizeof(float));
    assert(inputBuffers->scoreValueResultBufferBytes == buffers->scoreValueBufBytes);
    assert(inputBuffers->ownershipResultBufferBytes == buffers->ownershipBufBytes);
    assert(inputBuffers->singleOwnershipResultElts == nnXLen*nnYLen);
    assert(inputBuffers->singleOwnershipResultBytes == nnXLen*nnYLen * sizeof(float));

    CUDA_ERR("getOutput",cudaMemcpyAsync(buffers->inputBuf, inputBuffers->userInputBuffer, inputBuffers->singleInputBytes*batchSize, cudaMemcpyHostToDevice, stream));
    CUDA_ERR("getOutput",cudaMemcpyAsync(buffers->inputGlobalBuf, inputBuffers->userInputGlobalBuffer, inputBuffers->singleInputGlobalBytes*batchSize, cudaMemcpyHostToDevice, stream));
    if(numMetaFeatures > 0) {
      CUDA_ERR("getOutput",cudaMemcpyAsync(buffers->inputMetaBuf, inputBuffers->userInputMetaBuffer, inputBuffers->singleInputMetaBytes*batchSize, cudaMemcpyHostToDevice, stream));
    }
  }
  else {
    assert(inputBuffers->userInputBufferBytes == buffers->inputBufBytesFloat);
    assert(inputBuffers->userInputGlobalBufferBytes == buffers->inputGlobalBufBytesFloat);
    assert(inputBuffers->userInputMetaBufferBytes == buffers->inputMetaBufBytesFloat);
    assert(inputBuffers->policyResultBufferBytes == buffers->policyBufBytes);
    assert(inputBuffers->valueResultBufferBytes == buffers->valueBufBytes);
    assert(inputBuffers->userInputBufferBytes == buffers->inputBufBytes*2);
    assert(inputBuffers->userInputGlobalBufferBytes == buffers->inputGlobalBufBytes*2);
    assert(inputBuffers->userInputMetaBufferBytes == buffers->inputMetaBufBytes*2);
    assert(inputBuffers->singleInputBytes == inputBuffers->singleInputElts*4);
    assert(inputBuffers->singleInputGlobalBytes == inputBuffers->singleInputGlobalElts*4);
    assert(inputBuffers->singleInputMetaBytes == inputBuffers->singleInputMetaElts*4);
    assert(inputBuffers->singlePolicyPassResultElts == numPolicyChannels);
    assert(inputBuffers->singlePolicyPassResultBytes == numPolicyChannels * sizeof(float));
    assert(inputBuffers->singlePolicyResultElts == numPolicyChannels*nnXLen*nnYLen);
    assert(inputBuffers->singlePolicyResultBytes == numPolicyChannels*nnXLen*nnYLen * sizeof(float));
    assert(inputBuffers->scoreValueResultBufferBytes == buffers->scoreValueBufBytes);
    assert(inputBuffers->ownershipResultBufferBytes == buffers->ownershipBufBytes);
    assert(inputBuffers->singleOwnershipResultElts == nnXLen*nnYLen);
    assert(inputBuffers->singleOwnershipResultBytes == nnXLen*nnYLen * sizeof(float));

    CUDA_ERR("getOutput",cudaMemcpyAsync(buffers->inputBufFloat, inputBuffers->userInputBuffer, inputBuffers->singleInputBytes*batchSize, cudaMemcpyHostToDevice, stream));
    CUDA_ERR("getOutput",cudaMemcpyAsync(buffers->inputGlobalBufFloat, inputBuffers->userInputGlobalBuffer, inputBuffers->singleInputGlobalBytes*batchSize, cudaMemcpyHostToDevice, stream));
    if(numMetaFeatures > 0) {
      CUDA_ERR("getOutput",cudaMemcpyAsync(buffers->inputMetaBufFloat, inputBuffers->userInputMetaBuffer, inputBuffers->singleInputMetaBytes*batchSize, cudaMemcpyHostToDevice, stream));
    }

    customCudaCopyToHalf((const float*)buffers->inputBufFloat,(half*)buffers->inputBuf,inputBuffers->singleInputElts*batchSize,cudaHandles->stream);
    CUDA_ERR("getOutput",cudaPeekAtLastError());
    customCudaCopyToHalf((const float*)buffers->inputGlobalBufFloat,(half*)buffers->inputGlobalBuf,inputBuffers->singleInputGlobalElts*batchSize,cudaHandles->stream);
    CUDA_ERR("getOutput",cudaPeekAtLastError());
    if(numMetaFeatures > 0) {
      customCudaCopyToHalf((const float*)buffers->inputMetaBufFloat,(half*)buffers->inputMetaBuf,inputBuffers->singleInputMetaElts*batchSize,cudaHandles->stream);
      CUDA_ERR("getOutput",cudaPeekAtLastError());
    }
  }

  gpuHandle->model->apply(
    gpuHandle->cudaHandles.get(),
    scratch,
    batchSize,
    gpuHandle->requireExactNNLen,

    buffers->inputBuf,
    buffers->inputGlobalBuf,
    buffers->inputMetaBuf,

    buffers->policyPassBuf,
    buffers->policyBuf,

    buffers->valueBuf,
    buffers->scoreValueBuf,
    buffers->ownershipBuf,

    buffers->workspaceBuf,
    buffers->workspaceBytes
  );

  CUDA_ERR("getOutput",cudaMemcpyAsync(inputBuffers->policyPassResults, buffers->policyPassBuf, inputBuffers->singlePolicyPassResultBytes*batchSize, cudaMemcpyDeviceToHost, stream));
  CUDA_ERR("getOutput",cudaMemcpyAsync(inputBuffers->policyResults, buffers->policyBuf, inputBuffers->singlePolicyResultBytes*batchSize, cudaMemcpyDeviceToHost, stream));
  CUDA_ERR("getOutput",cudaMemcpyAsync(inputBuffers->valueResults, buffers->valueBuf, inputBuffers->singleValueResultBytes*batchSize, cudaMemcpyDeviceToHost, stream));
  CUDA_ERR("getOutput",cudaMemcpyAsync(inputBuffers->scoreValueResults, buffers->scoreValueBuf, inputBuffers->singleScoreValueResultBytes*batchSize, cudaMemcpyDeviceToHost, stream));
  CUDA_ERR("getOutput",cudaMemcpyAsync(inputBuffers->ownershipResults, buffers->ownershipBuf, inputBuffers->singleOwnershipResultBytes*batchSize, cudaMemcpyDeviceToHost, stream));
  CUDA_ERR("getOutput",cudaStreamSynchronize(stream));

  assert(outputs.size() == batchSize);

  float policyProbsTmp[NNPos::MAX_NN_POLICY_SIZE];

  for(int row = 0; row < batchSize; row++) {
    NNOutput* output = outputs[row];
    assert(output->nnXLen == nnXLen);
    assert(output->nnYLen == nnYLen);
    float policyOptimism = (float)inputBufs[row]->policyOptimism;

    const float* policyPassSrcBuf = inputBuffers->policyPassResults + row * numPolicyChannels;
    const float* policySrcBuf = inputBuffers->policyResults + row * numPolicyChannels * nnXLen * nnYLen;
    float* policyProbs = output->policyProbs;

    // These are in logits, the client does the postprocessing to turn them into
    // policy probabilities and white game outcome probabilities
    // Also we don't fill in the nnHash here either
    // Handle version >= 12 policy optimism
    if(numPolicyChannels == 2 || (numPolicyChannels == 4 && modelVersion >= 16)) {
      if(gpuHandle->usingNHWC) {
        for(int i = 0; i<nnXLen*nnYLen; i++) {
          float p = policySrcBuf[i*numPolicyChannels];
          float pOpt = policySrcBuf[i*numPolicyChannels+1];
          policyProbsTmp[i] = p + (pOpt-p) * policyOptimism;
        }
        SymmetryHelpers::copyOutputsWithSymmetry(policyProbsTmp, policyProbs, 1, nnYLen, nnXLen, inputBufs[row]->symmetry);
        policyProbs[nnXLen*nnYLen] = policyPassSrcBuf[0] + (policyPassSrcBuf[1] - policyPassSrcBuf[0]) * policyOptimism;
      }
      else {
        for(int i = 0; i<nnXLen*nnYLen; i++) {
          float p = policySrcBuf[i];
          float pOpt = policySrcBuf[i+nnXLen*nnYLen];
          policyProbsTmp[i] = p + (pOpt-p) * policyOptimism;
        }
        SymmetryHelpers::copyOutputsWithSymmetry(policyProbsTmp, policyProbs, 1, nnYLen, nnXLen, inputBufs[row]->symmetry);
        policyProbs[nnXLen*nnYLen] = policyPassSrcBuf[0] + (policyPassSrcBuf[1] - policyPassSrcBuf[0]) * policyOptimism;
      }
    }
    else {
      assert(numPolicyChannels == 1);
      SymmetryHelpers::copyOutputsWithSymmetry(policySrcBuf, policyProbs, 1, nnYLen, nnXLen, inputBufs[row]->symmetry);
      policyProbs[nnXLen*nnYLen] = policyPassSrcBuf[0];
    }

    int numValueChannels = gpuHandle->model->numValueChannels;
    assert(numValueChannels == 3);
    output->whiteWinProb = inputBuffers->valueResults[row * numValueChannels];
    output->whiteLossProb = inputBuffers->valueResults[row * numValueChannels + 1];
    output->whiteNoResultProb = inputBuffers->valueResults[row * numValueChannels + 2];

    //As above, these are NOT actually from white's perspective, but rather the player to move.
    //As usual the client does the postprocessing.
    if(output->whiteOwnerMap != NULL) {
      const float* ownershipSrcBuf = inputBuffers->ownershipResults + row * nnXLen * nnYLen;
      assert(gpuHandle->model->numOwnershipChannels == 1);
      SymmetryHelpers::copyOutputsWithSymmetry(ownershipSrcBuf, output->whiteOwnerMap, 1, nnYLen, nnXLen, inputBufs[row]->symmetry);
    }

    if(modelVersion >= 9) {
      int numScoreValueChannels = gpuHandle->model->numScoreValueChannels;
      assert(numScoreValueChannels == 6);
      output->whiteScoreMean = inputBuffers->scoreValueResults[row * numScoreValueChannels];
      output->whiteScoreMeanSq = inputBuffers->scoreValueResults[row * numScoreValueChannels + 1];
      output->whiteLead = inputBuffers->scoreValueResults[row * numScoreValueChannels + 2];
      output->varTimeLeft = inputBuffers->scoreValueResults[row * numScoreValueChannels + 3];
      output->shorttermWinlossError = inputBuffers->scoreValueResults[row * numScoreValueChannels + 4];
      output->shorttermScoreError = inputBuffers->scoreValueResults[row * numScoreValueChannels + 5];
    }
    else if(modelVersion >= 8) {
      int numScoreValueChannels = gpuHandle->model->numScoreValueChannels;
      assert(numScoreValueChannels == 4);
      output->whiteScoreMean = inputBuffers->scoreValueResults[row * numScoreValueChannels];
      output->whiteScoreMeanSq = inputBuffers->scoreValueResults[row * numScoreValueChannels + 1];
      output->whiteLead = inputBuffers->scoreValueResults[row * numScoreValueChannels + 2];
      output->varTimeLeft = inputBuffers->scoreValueResults[row * numScoreValueChannels + 3];
      output->shorttermWinlossError = 0;
      output->shorttermScoreError = 0;
    }
    else if(modelVersion >= 4) {
      int numScoreValueChannels = gpuHandle->model->numScoreValueChannels;
      assert(numScoreValueChannels == 2);
      output->whiteScoreMean = inputBuffers->scoreValueResults[row * numScoreValueChannels];
      output->whiteScoreMeanSq = inputBuffers->scoreValueResults[row * numScoreValueChannels + 1];
      output->whiteLead = output->whiteScoreMean;
      output->varTimeLeft = 0;
      output->shorttermWinlossError = 0;
      output->shorttermScoreError = 0;
    }
    else if(modelVersion >= 3) {
      int numScoreValueChannels = gpuHandle->model->numScoreValueChannels;
      assert(numScoreValueChannels == 1);
      output->whiteScoreMean = inputBuffers->scoreValueResults[row * numScoreValueChannels];
      //Version 3 neural nets don't have any second moment output, implicitly already folding it in, so we just use the mean squared
      output->whiteScoreMeanSq = output->whiteScoreMean * output->whiteScoreMean;
      output->whiteLead = output->whiteScoreMean;
      output->varTimeLeft = 0;
      output->shorttermWinlossError = 0;
      output->shorttermScoreError = 0;
    }
    else {
      ASSERT_UNREACHABLE;
    }
  }

}

//TESTING ----------------------------------------------------------------------------------


bool NeuralNet::testEvaluateConv(
  const ConvLayerDesc* desc,
  int desiredBatchSize,
  int nnXLen,
  int nnYLen,
  bool useFP16,
  bool useNHWC,
  const vector<float>& inputBuffer,
  vector<float>& outputBuffer
) {
  //If this build was compiled without FP16 kernel support, the half kernels are empty stubs.
  testAssert(!useFP16 || customCudaFp16KernelsCompiled());
  (void)cudaDeviceSynchronize();
  CudaHandles* cudaHandles = CudaHandles::cudaHandlesTesting();

  size_t numInputFloats = (size_t)desiredBatchSize * nnXLen * nnYLen * desc->inChannels;
  size_t numOutputFloats = (size_t)desiredBatchSize * nnXLen * nnYLen * desc->outChannels;
  if(numInputFloats != inputBuffer.size())
    throw StringError("testEvaluateConv: unexpected input buffer size");

  void* deviceInput;
  void* deviceOutput;
  CudaUtils::mallocAndCopyToDevice("deviceInput", inputBuffer.data(), numInputFloats, deviceInput, useFP16);
  CudaUtils::mallocOnDevice("deviceOutput", numOutputFloats, deviceOutput, useFP16);

  int maxBatchSize = desiredBatchSize;

  CudnnManager* manager = new CudnnManager("manager",maxBatchSize,nnXLen,nnYLen);
  ConvLayer* convLayer = new ConvLayer(cudaHandles,manager,desc,useFP16,useNHWC);

  size_t workspaceBytes =
    convLayer->requiredWorkspaceBytes(cudaHandles,desiredBatchSize);
  void* deviceWorkspace;
  CUDA_ERR("deviceWorkspace",cudaMalloc(&deviceWorkspace, workspaceBytes));


  ScratchBuffers* scratch = new ScratchBuffers(desiredBatchSize, nnXLen, nnYLen, useFP16);

  bool accumulate = false;
  convLayer->apply(
    cudaHandles,
    scratch,
    desiredBatchSize,
    accumulate,
    deviceInput,
    deviceOutput,
    deviceWorkspace,
    workspaceBytes
  );

  outputBuffer.resize(numOutputFloats);
  CudaUtils::expensiveCopyFromDevice("copyResultsToHost", outputBuffer.data(), numOutputFloats, deviceOutput, useFP16);

  (void)cudaFree(deviceWorkspace);

  delete convLayer;
  delete manager;
  (void)cudaFree(deviceInput);
  (void)cudaFree(deviceOutput);
  delete scratch;
  delete cudaHandles;

  return true;
}


bool NeuralNet::testEvaluateBatchNorm(
  const BatchNormLayerDesc* desc,
  int desiredBatchSize,
  int nnXLen,
  int nnYLen,
  bool useFP16,
  bool useNHWC,
  const vector<float>& inputBuffer,
  const vector<float>& maskBuffer,
  vector<float>& outputBuffer
) {
  //If this build was compiled without FP16 kernel support, the half kernels are empty stubs.
  testAssert(!useFP16 || customCudaFp16KernelsCompiled());
  (void)cudaDeviceSynchronize();
  CudaHandles* cudaHandles = CudaHandles::cudaHandlesTesting();

  size_t numInputFloats = (size_t)desiredBatchSize * nnXLen * nnYLen * desc->numChannels;
  size_t numMaskFloats = (size_t)desiredBatchSize * nnXLen * nnYLen;
  size_t numOutputFloats = (size_t)desiredBatchSize * nnXLen * nnYLen * desc->numChannels;
  if(numInputFloats != inputBuffer.size())
    throw StringError("testEvaluateBatchNorm: unexpected input buffer size");
  if(numMaskFloats != maskBuffer.size())
    throw StringError("testEvaluateBatchNorm: unexpected mask buffer size");

  ActivationLayerDesc actDesc;
  actDesc.activation = ACTIVATION_IDENTITY;

  void* deviceInput;
  void* deviceMask;
  void* deviceOutput;
  CudaUtils::mallocAndCopyToDevice("deviceInput", inputBuffer.data(), numInputFloats, deviceInput, useFP16);
  CudaUtils::mallocAndCopyToDevice("deviceMask", maskBuffer.data(), numMaskFloats, deviceMask, useFP16);
  CudaUtils::mallocOnDevice("deviceOutput", numOutputFloats, deviceOutput, useFP16);

  BatchNormLayer* batchNormLayer = new BatchNormLayer(cudaHandles,desc,&actDesc,nnXLen,nnYLen,useFP16,useNHWC);

  batchNormLayer->apply(
    cudaHandles,
    desiredBatchSize,
    deviceInput,
    deviceMask,
    deviceOutput
  );

  outputBuffer.resize(numOutputFloats);
  CudaUtils::expensiveCopyFromDevice("copyResultsToHost", outputBuffer.data(), numOutputFloats, deviceOutput, useFP16);

  delete batchNormLayer;

  (void)cudaFree(deviceInput);
  (void)cudaFree(deviceMask);
  (void)cudaFree(deviceOutput);
  delete cudaHandles;

  return true;
}


bool NeuralNet::testEvaluateResidualBlock(
  const ResidualBlockDesc* desc,
  int desiredBatchSize,
  int nnXLen,
  int nnYLen,
  bool useFP16,
  bool useNHWC,
  const vector<float>& inputBuffer,
  const vector<float>& maskBuffer,
  vector<float>& outputBuffer
) {
  //If this build was compiled without FP16 kernel support, the half kernels are empty stubs.
  testAssert(!useFP16 || customCudaFp16KernelsCompiled());
  (void)cudaDeviceSynchronize();
  CudaHandles* cudaHandles = CudaHandles::cudaHandlesTesting();

  size_t numInputFloats = (size_t)desiredBatchSize * nnXLen * nnYLen * desc->preBN.numChannels;
  size_t numMaskFloats = (size_t)desiredBatchSize * nnXLen * nnYLen;
  size_t numOutputFloats = (size_t)desiredBatchSize * nnXLen * nnYLen * desc->finalConv.outChannels;
  if(numInputFloats != inputBuffer.size())
    throw StringError("testEvaluateResidualBlock: unexpected input buffer size");
  if(numMaskFloats != maskBuffer.size())
    throw StringError("testEvaluateResidualBlock: unexpected mask buffer size");

  ScratchBuffers* scratch = new ScratchBuffers(desiredBatchSize, nnXLen, nnYLen, useFP16);

  void* deviceInput;
  void* deviceMask;
  void* deviceScratch;
  CudaUtils::mallocAndCopyToDevice("deviceInput", inputBuffer.data(), numInputFloats, deviceInput, useFP16);
  CudaUtils::mallocAndCopyToDevice("deviceMask", maskBuffer.data(), numMaskFloats, deviceMask, useFP16);
  CudaUtils::mallocOnDevice("deviceScratch", numInputFloats, deviceScratch, useFP16);

  int maxBatchSize = desiredBatchSize;

  CudnnManager* manager = new CudnnManager("manager",maxBatchSize,nnXLen,nnYLen);
  ResidualBlock* residualBlock = new ResidualBlock(cudaHandles,manager,desc,nnXLen,nnYLen,useFP16,useNHWC);

  size_t workspaceBytes =
    residualBlock->requiredWorkspaceBytes(cudaHandles,desiredBatchSize);
  void* deviceWorkspace;
  CUDA_ERR("deviceWorkspace",cudaMalloc(&deviceWorkspace, workspaceBytes));

  residualBlock->apply(
    cudaHandles,
    scratch,
    desiredBatchSize,
    deviceInput,
    deviceScratch,
    deviceMask,
    deviceWorkspace,
    workspaceBytes
  );

  outputBuffer.resize(numOutputFloats);
  CudaUtils::expensiveCopyFromDevice("copyResultsToHost", outputBuffer.data(), numOutputFloats, deviceInput, useFP16);

  (void)cudaFree(deviceWorkspace);

  delete residualBlock;
  delete manager;
  (void)cudaFree(deviceInput);
  (void)cudaFree(deviceMask);
  (void)cudaFree(deviceScratch);
  delete scratch;
  delete cudaHandles;

  return true;
}

bool NeuralNet::testEvaluateGlobalPoolingResidualBlock(
  const GlobalPoolingResidualBlockDesc* desc,
  int desiredBatchSize,
  int nnXLen,
  int nnYLen,
  bool useFP16,
  bool useNHWC,
  const vector<float>& inputBuffer,
  const vector<float>& maskBuffer,
  vector<float>& outputBuffer
) {
  //If this build was compiled without FP16 kernel support, the half kernels are empty stubs.
  testAssert(!useFP16 || customCudaFp16KernelsCompiled());
  (void)cudaDeviceSynchronize();
  CudaHandles* cudaHandles = CudaHandles::cudaHandlesTesting();

  size_t numInputFloats = (size_t)desiredBatchSize * nnXLen * nnYLen * desc->preBN.numChannels;
  size_t numMaskFloats = (size_t)desiredBatchSize * nnXLen * nnYLen;
  size_t numMaskSumFloats = (size_t)desiredBatchSize;
  size_t numOutputFloats = (size_t)desiredBatchSize * nnXLen * nnYLen * desc->finalConv.outChannels;

  if(numInputFloats != inputBuffer.size())
    throw StringError("testEvaluateGlobalPoolingResidualBlock: unexpected input buffer size");
  if(numMaskFloats != maskBuffer.size())
    throw StringError("testEvaluateGlobalPoolingResidualBlock: unexpected mask buffer size");

  ScratchBuffers* scratch = new ScratchBuffers(desiredBatchSize, nnXLen, nnYLen, useFP16);

  void* deviceInput;
  void* deviceMask;
  float* deviceMaskFloatOrig;
  float* deviceMaskFloat;
  float* deviceMaskSum;
  void* deviceScratch;

  CudaUtils::mallocAndCopyToDevice("deviceInput", inputBuffer.data(), numInputFloats, deviceInput, useFP16);
  CudaUtils::mallocAndCopyToDevice("deviceMask", maskBuffer.data(), numMaskFloats, deviceMask, useFP16);
  CUDA_ERR("deviceMaskFloat",cudaMalloc(reinterpret_cast<void**>(&deviceMaskFloat), numMaskFloats * sizeof(float)));
  CUDA_ERR("deviceMaskSum",cudaMalloc(reinterpret_cast<void**>(&deviceMaskSum), numMaskSumFloats * sizeof(float)));
  deviceMaskFloatOrig = deviceMaskFloat;
  CudaUtils::mallocOnDevice("deviceScratch", numInputFloats, deviceScratch, useFP16);

  fillMaskFloatBufAndMaskSumBuf(cudaHandles,deviceMask, deviceMaskFloat, deviceMaskSum, useFP16, desiredBatchSize, nnXLen, nnYLen);

  int maxBatchSize = desiredBatchSize;

  CudnnManager* manager = new CudnnManager("manager",maxBatchSize,nnXLen,nnYLen);
  GlobalPoolingResidualBlock* residualBlock = new GlobalPoolingResidualBlock(
    cudaHandles,manager,desc,nnXLen,nnYLen,useFP16,useNHWC
  );

  size_t workspaceBytes =
    residualBlock->requiredWorkspaceBytes(
      cudaHandles,desiredBatchSize
    );

  void* deviceWorkspace;
  CUDA_ERR("deviceWorkspace",cudaMalloc(&deviceWorkspace, workspaceBytes));

  residualBlock->apply(
    cudaHandles,
    scratch,
    desiredBatchSize,
    deviceInput,
    deviceScratch,
    deviceMask,
    deviceMaskSum,
    deviceWorkspace,
    workspaceBytes
  );

  outputBuffer.resize(numOutputFloats);
  CudaUtils::expensiveCopyFromDevice("copyResultsToHost", outputBuffer.data(), numOutputFloats, deviceInput, useFP16);

  (void)cudaFree(deviceWorkspace);

  delete residualBlock;
  delete manager;

  (void)cudaFree(deviceInput);
  (void)cudaFree(deviceMask);
  (void)cudaFree(deviceMaskFloatOrig);
  (void)cudaFree(deviceMaskSum);
  (void)cudaFree(deviceScratch);
  delete scratch;
  delete cudaHandles;

  return true;
}
