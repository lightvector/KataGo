#pragma once

#include <cstdint>
#include <sstream>
#include <string>
#include <vector>

// cudaLibrary_t, cudaKernel_t, cudaJitOption, cudaLibraryOption etc.
// were introduced in CUDA 12.8. On older toolkits (or when using dynamic
// loading where cuda_runtime.h may lack these types), provide typedefs
// from the driver API equivalents (available via cuda.h) so that
// engine headers can declare members without #ifdef clutter.
// The actual runtime API functions are guarded by CUDART_VERSION checks below.
#if !defined(CUDART_VERSION) || CUDART_VERSION < 12080
using cudaLibrary_t     = CUlibrary;
using cudaKernel_t      = CUkernel;
using cudaJitOption     = CUjit_option;
using cudaLibraryOption = CUlibraryOption;
#endif

namespace cudnn_frontend::experimental {

// ============================================================
// Internal types matching generated kernel parameter layouts
// ============================================================

struct AttentionShape_t {
    uint32_t b, q_h, k_h, v_h, s_q, s_kv, d_qk, d_v;
};

struct AttentionDescriptor_t {
    uint32_t b, q_h, k_h, v_h, s_q, s_kv, d_qk, d_v;
    uint16_t q_heads_per_k, q_heads_per_v, min_q_heads_per_kv;
};

struct FastDivisor_t {
    uint32_t val, shr, mul;
};

struct tensor_descriptor {
    static const int MAX_DIMS = 12;
    int64_t num_dims;
    int64_t dims[MAX_DIMS];
    int64_t strides[MAX_DIMS];
};

// ============================================================
// Utility functions
// ============================================================

inline int
div_up(int a, int b) {
    return (a + b - 1) / b;
}

// floor(log2(x)) for x > 0.
inline int
find_log_2_floor(uint32_t x) {
    if (x <= 1) return 0;
    int a = 0;
    while ((1u << (a + 1)) <= x) a++;
    return a;
}

// Compute FastDivisor_t for the kernel's fastDivMod which uses:
//   div = __umulhi(2 * val, mul) >> shr
// This matches cuDNN's find_divisor_v2 (xmma/fast_math.h:118-125).
inline FastDivisor_t
make_fast_divisor(uint32_t divisor) {
    FastDivisor_t d;
    d.val = divisor;

    if (divisor <= 1) {
        // Division by 1: umulhi(2*val, 0x80000000) >> 0 = val, mod = 0
        d.shr = 0;
        d.mul = 0x80000000u;
        return d;
    }

    // find_log_2(2 * divisor, round_up=true)
    uint32_t x2 = 2u * divisor;
    int a       = 0;
    {
        uint32_t tmp = x2;
        while (tmp > 1) {
            tmp >>= 1;
            a++;
        }
    }
    // round up if not a power of 2
    if (x2 & (x2 - 1)) a++;

    uint32_t p = 31 + static_cast<uint32_t>(a);
    d.mul      = static_cast<uint32_t>(((1ULL << p) + static_cast<uint64_t>(x2) - 1) / static_cast<uint64_t>(x2));
    d.shr      = p - 32;
    return d;
}

inline std::vector<std::string>
parse_flags_string(const char* data, size_t len) {
    std::vector<std::string> flags;
    std::string content(data, len);
    std::istringstream stream(content);
    std::string line;
    while (std::getline(stream, line)) {
        size_t start = line.find_first_not_of(" \t\r\n");
        if (start == std::string::npos) continue;
        size_t end = line.find_last_not_of(" \t\r\n");
        line       = line.substr(start, end - start + 1);
        if (!line.empty()) {
            flags.push_back(line);
        }
    }
    return flags;
}

// ============================================================
// Kernel specification (compile-time metadata per kernel variant)
// ============================================================

struct KernelSpec {
    const char* source;
    size_t source_len;
    const char* flags_raw;
    size_t flags_len;
    const char* kernel_name;
    int tile_m, tile_n, tile_k;
    int smem_bytes;
};

// ============================================================
// CUDA runtime API wrappers (using NV_FE_CALL_TO_CUDA)
// ============================================================

namespace detail {

// NV_FE_CALL_TO_CUDA macros reference symbols in cudnn_frontend::detail
// via unqualified lookup. Import them into this namespace.
#if defined NV_CUDNN_FRONTEND_USE_DYNAMIC_LOADING
using cudnn_frontend::detail::CudaLibrary;
using cudnn_frontend::detail::get_cuda_symbol;
#endif

// Re-export CUDA runtime wrappers from the main shim so the engine
// never calls cudaMalloc/cudaFree/etc. directly (required for dynamic loading).
using cudnn_frontend::detail::cuda_get_device;
using cudnn_frontend::detail::cuda_get_device_properties;
using cudnn_frontend::detail::cuda_get_error_string;
using cudnn_frontend::detail::cuda_mem_cpy_async;
using cudnn_frontend::detail::cuda_mem_set_async;

// Write a 32-bit pattern to device memory (async, on stream).
// Uses thread_local storage so the source buffer persists through the async copy.
// Only used with N=1 in practice (writing float 1.0f for FP8 default scale).
// Cannot use NV_FE_CALL_TO_CUDA because cudaMemcpyAsync has more args than this wrapper.
inline cudaError_t
cuda_mem_set_d32_async(void* dstDevice, unsigned int ui, size_t N, cudaStream_t stream) {
    static thread_local unsigned int val;
    val = ui;
#if defined NV_CUDNN_FRONTEND_USE_DYNAMIC_LOADING
    using fn_t = cudaError_t (*)(void*, const void*, size_t, cudaMemcpyKind, cudaStream_t);
    auto _fn = reinterpret_cast<fn_t>(cudnn_frontend::detail::get_cuda_symbol(CudaLibrary::CUDART, "cudaMemcpyAsync"));
    return _fn(dstDevice, &val, N * sizeof(unsigned int), cudaMemcpyHostToDevice, stream);
#else
    return cudaMemcpyAsync(dstDevice, &val, N * sizeof(unsigned int), cudaMemcpyHostToDevice, stream);
#endif
}

// Convert cudaError_t to a descriptive string (e.g., "invalid argument")
inline std::string
cuda_error_to_string(cudaError_t err) {
    const char* str = cuda_get_error_string(err);
    return str ? std::string(str) : ("cudaError=" + std::to_string(static_cast<int>(err)));
}

// cudaGetLastError takes zero arguments — can't use NV_FE_CALL_TO_CUDA
// (variadic macro requires at least one arg). Handle both paths manually.
inline cudaError_t
cuda_get_last_error() {
#if defined NV_CUDNN_FRONTEND_USE_DYNAMIC_LOADING
    using fn_t = cudaError_t (*)();
    auto _fn = reinterpret_cast<fn_t>(cudnn_frontend::detail::get_cuda_symbol(CudaLibrary::CUDART, "cudaGetLastError"));
    return _fn();
#else
    return cudaGetLastError();
#endif
}

// ============================================================
// CUDA 12.8+ runtime API wrappers for library/kernel management.
// These APIs (cudaLibraryLoadData, cudaLibraryGetKernel, cudaLibraryUnload,
// cudaKernelSetAttributeForDevice, cudaGetDriverEntryPointByVersion) require
// CUDART_VERSION >= 12080. On older toolkits, the OSS engine check_support()
// will reject the configuration before these are called.
// ============================================================

#if !defined(NV_CUDNN_FRONTEND_USE_DYNAMIC_LOADING) && defined(CUDART_VERSION) && CUDART_VERSION < 12080

// Stubs that return errors — OSS engines require CUDA 12.8+.
inline cudaError_t
cuda_library_load_data(void*, const void*, void*, void**, unsigned int, void*, void**, unsigned int) {
    return cudaErrorNotSupported;
}
inline cudaError_t
cuda_library_get_kernel(void*, void*, const char*) {
    return cudaErrorNotSupported;
}
inline cudaError_t
cuda_library_unload(void*) {
    return cudaErrorNotSupported;
}
inline cudaError_t
cuda_kernel_set_attribute_for_device(void*, int, int, int) {
    return cudaErrorNotSupported;
}

#else  // CUDART_VERSION >= 12080 or dynamic loading

inline cudaError_t
cuda_library_load_data(cudaLibrary_t* library,
                       const void* code,
                       cudaJitOption* jitOptions,
                       void** jitOptionsValues,
                       unsigned int numJitOptions,
                       cudaLibraryOption* libraryOptions,
                       void** libraryOptionsValues,
                       unsigned int numLibraryOptions) {
    NV_FE_CALL_TO_CUDA(cuda_library_load_data,
                       cudaLibraryLoadData,
                       library,
                       code,
                       jitOptions,
                       jitOptionsValues,
                       numJitOptions,
                       libraryOptions,
                       libraryOptionsValues,
                       numLibraryOptions);
}

inline cudaError_t
cuda_library_get_kernel(cudaKernel_t* pKernel, cudaLibrary_t library, const char* name) {
    NV_FE_CALL_TO_CUDA(cuda_library_get_kernel, cudaLibraryGetKernel, pKernel, library, name);
}

inline cudaError_t
cuda_library_unload(cudaLibrary_t library) {
    NV_FE_CALL_TO_CUDA(cuda_library_unload, cudaLibraryUnload, library);
}

inline cudaError_t
cuda_kernel_set_attribute_for_device(cudaKernel_t kernel, cudaFuncAttribute attrib, int val, int dev) {
    NV_FE_CALL_TO_CUDA(cuda_kernel_set_attribute_for_device, cudaKernelSetAttributeForDevice, kernel, attrib, val, dev);
}

#endif  // CUDART_VERSION check

inline cudaError_t
cuda_launch_kernel(const void* func, dim3 gridDim, dim3 blockDim, void** args, size_t sharedMem, cudaStream_t stream) {
    NV_FE_CALL_TO_CUDA(cuda_launch_kernel, cudaLaunchKernel, func, gridDim, blockDim, args, sharedMem, stream);
}

// cuTensorMapEncodeTiled has no runtime API equivalent.
// Resolve the driver function pointer via cudaGetDriverEntryPointByVersion
// so we never link against libcuda.so directly. Requires CUDA 12.8+.
inline CUresult
cu_tensor_map_encode_tiled(CUtensorMap* tensorMap,
                           CUtensorMapDataType tensorDataType,
                           cuuint32_t tensorRank,
                           void* globalAddress,
                           const cuuint64_t* globalDim,
                           const cuuint64_t* globalStrides,
                           const cuuint32_t* boxDim,
                           const cuuint32_t* elementStrides,
                           CUtensorMapInterleave interleave,
                           CUtensorMapSwizzle swizzle,
                           CUtensorMapL2promotion l2Promotion,
                           CUtensorMapFloatOOBfill oobFill) {
#if !defined(NV_CUDNN_FRONTEND_USE_DYNAMIC_LOADING) && defined(CUDART_VERSION) && CUDART_VERSION < 12080
    (void)tensorMap;
    (void)tensorDataType;
    (void)tensorRank;
    (void)globalAddress;
    (void)globalDim;
    (void)globalStrides;
    (void)boxDim;
    (void)elementStrides;
    (void)interleave;
    (void)swizzle;
    (void)l2Promotion;
    (void)oobFill;
    return CUDA_ERROR_NOT_SUPPORTED;
#else
    using PFN = CUresult(CUDAAPI*)(CUtensorMap*,
                                   CUtensorMapDataType,
                                   cuuint32_t,
                                   void*,
                                   const cuuint64_t*,
                                   const cuuint64_t*,
                                   const cuuint32_t*,
                                   const cuuint32_t*,
                                   CUtensorMapInterleave,
                                   CUtensorMapSwizzle,
                                   CUtensorMapL2promotion,
                                   CUtensorMapFloatOOBfill);
    // Thread-safe static initialization (C++11 guarantees)
    static const PFN pfn = []() -> PFN {
        void* raw_pfn = nullptr;
        cudaDriverEntryPointQueryResult status;
#if defined NV_CUDNN_FRONTEND_USE_DYNAMIC_LOADING
        // In dynamic loading mode, resolve cudaGetDriverEntryPointByVersion via dlsym
        using GetEntryPointFn =
            cudaError_t (*)(const char*, void**, unsigned int, int, cudaDriverEntryPointQueryResult*);
        auto get_entry_point = reinterpret_cast<GetEntryPointFn>(
            cudnn_frontend::detail::get_cuda_symbol(CudaLibrary::CUDART, "cudaGetDriverEntryPointByVersion"));
        cudaError_t err = get_entry_point("cuTensorMapEncodeTiled", &raw_pfn, 12000, cudaEnableDefault, &status);
#else
        cudaError_t err =
            cudaGetDriverEntryPointByVersion("cuTensorMapEncodeTiled", &raw_pfn, 12000, cudaEnableDefault, &status);
#endif
        if (err != cudaSuccess || status != cudaDriverEntryPointSuccess || !raw_pfn) {
            return nullptr;
        }
        return reinterpret_cast<PFN>(raw_pfn);
    }();
    if (!pfn) {
        return CUDA_ERROR_NOT_FOUND;
    }
    return pfn(tensorMap,
               tensorDataType,
               tensorRank,
               globalAddress,
               globalDim,
               globalStrides,
               boxDim,
               elementStrides,
               interleave,
               swizzle,
               l2Promotion,
               oobFill);
#endif  // CUDART_VERSION check
}

}  // namespace detail

// ============================================================
// TMA descriptor creation (4D)
// ============================================================
inline error_t
create_tma_desc_4d(CUtensorMap* desc,
                   void* globalAddress,
                   CUtensorMapDataType dataType,
                   uint32_t dim0,
                   uint32_t dim1,
                   uint32_t dim2,
                   uint32_t dim3,
                   uint64_t stride1_bytes,
                   uint64_t stride2_bytes,
                   uint64_t stride3_bytes,
                   uint32_t boxDim0,
                   uint32_t boxDim1,
                   CUtensorMapSwizzle swizzle) {
    uint64_t globalDims[4]    = {dim0, dim1, dim2, dim3};
    uint64_t globalStrides[3] = {stride1_bytes, stride2_bytes, stride3_bytes};
    uint32_t boxDims[4]       = {boxDim0, boxDim1, 1, 1};
    uint32_t elemStrides[4]   = {1, 1, 1, 1};

    CUresult err_status = detail::cu_tensor_map_encode_tiled(desc,
                                                             dataType,
                                                             4,
                                                             globalAddress,
                                                             globalDims,
                                                             globalStrides,
                                                             boxDims,
                                                             elemStrides,
                                                             CU_TENSOR_MAP_INTERLEAVE_NONE,
                                                             swizzle,
                                                             CU_TENSOR_MAP_L2_PROMOTION_NONE,
                                                             CU_TENSOR_MAP_FLOAT_OOB_FILL_NAN_REQUEST_ZERO_FMA);

    RETURN_CUDNN_FRONTEND_ERROR_IF(
        err_status != CUDA_SUCCESS, error_code_t::CUDA_API_FAILED, "cuTensorMapEncodeTiled failed");
    return {error_code_t::OK, ""};
}

}  // namespace cudnn_frontend::experimental