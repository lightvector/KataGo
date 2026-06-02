/*
 * Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */

#pragma once

#include "oss_engine_interface.h"
#include "../generated/sdpa/sm100/prefill/full_seqlens/d128_fprop_kernel.h"
#include "../generated/sdpa/sm100/prefill/full_seqlens/d64_fprop_kernel.h"

#include <cuda_fp16.h>

// Suppress MSVC C4127 "conditional expression is constant" triggered by
// RETURN_CUDNN_FRONTEND_ERROR_IF macro (which has `if (retval == error_code_t::OK)`)
#if defined(_MSC_VER)
#pragma warning(push)
#pragma warning(disable : 4127)
#endif

#include <algorithm>
#include <cmath>

namespace cudnn_frontend {
namespace experimental {

// ============================================================
// Kernel spec lookup for SM100 SDPA prefill kernels
// ============================================================

/// @brief Look up the SM100 kernel specification for a given head dimension.
/// @param d Head dimension (d_qk). Currently supports 64 and 128.
/// @param sm_version SM version from cudaDeviceProp (e.g. 100, 101).
/// @return Pointer to static KernelSpec, or nullptr if unsupported.
inline const KernelSpec*
lookup_sm100_kernel_spec(int d, int sm_version) {
    // SM100 family: sm_version 100..109
    if (sm_version / 10 != 10) return nullptr;

    static const KernelSpec spec_d128 = {
        generated::sm100_d128_fprop_source,
        generated::sm100_d128_fprop_source_len,
        generated::sm100_d128_fprop_flags,
        generated::sm100_d128_fprop_flags_len,
        "cudnn_generated_oss_sdpa_sm100_flash_fprop_f16_knob_7_128x128x128_4x1x1_cga1x1x1_kernel0_0",
        128,
        128,
        128,
        232448  // smem_bytes for d=128
    };

    static const KernelSpec spec_d64 = {
        generated::sm100_d64_fprop_source,
        generated::sm100_d64_fprop_source_len,
        generated::sm100_d64_fprop_flags,
        generated::sm100_d64_fprop_flags_len,
        "cudnn_generated_oss_sdpa_sm100_flash_fprop_f16_knob_7_128x128x64_4x1x1_cga1x1x1_kernel0_0",
        128,
        128,
        64,
        232448  // smem_bytes for d=64
    };

    switch (d) {
        case 128:
            return &spec_d128;
        case 64:
            return &spec_d64;
        default:
            return nullptr;
    }
}

// ============================================================
// Sm100SdpaPrefillEngine
// ============================================================

/// @brief SM100-specific SDPA prefill engine using NVRTC-compiled kernels.
///
/// This engine targets the SM100 (Blackwell) architecture and uses L2-aware
/// tile scheduling instead of the SM90 persistent-tile-counter approach.
/// The kernel is compiled at runtime via NVRTC from embedded source and
/// launched with TMA descriptors for Q, K, V, and O tensors.
///
/// Usage:
///   1. check_support() -- validates shape + SM version, caches device info.
///   2. build()         -- NVRTC-compiles the kernel and loads the CU module.
///   3. execute()       -- sets up TMA descriptors, packs params, launches.
///
/// Thread safety: instances are NOT thread-safe. Each thread should own its
/// own engine, or external synchronization must be provided.
class Sm100SdpaPrefillEngine : public IOssSdpaEngine {
   public:
    Sm100SdpaPrefillEngine() = default;

    ~Sm100SdpaPrefillEngine() override {
        if (module_) {
            detail::cuda_library_unload(module_);
        }
    }

    // Non-copyable
    Sm100SdpaPrefillEngine(const Sm100SdpaPrefillEngine&) = delete;
    Sm100SdpaPrefillEngine&
    operator=(const Sm100SdpaPrefillEngine&) = delete;

    // Movable
    Sm100SdpaPrefillEngine(Sm100SdpaPrefillEngine&& other) noexcept
        : spec_(other.spec_),
          d_(other.d_),
          sm_version_(other.sm_version_),
          sm_count_(other.sm_count_),
          support_checked_(other.support_checked_),
          module_(other.module_),
          kernelPtr_(other.kernelPtr_),
          cubin_(std::move(other.cubin_)),
          cubinSize_(other.cubinSize_),
          tile_m_(other.tile_m_),
          tile_n_(other.tile_n_),
          tile_k_(other.tile_k_),
          smemBytes_(other.smemBytes_),
          kernelName_(std::move(other.kernelName_)),
          built_(other.built_) {
        other.module_    = nullptr;
        other.kernelPtr_ = nullptr;
        other.built_     = false;
    }

    Sm100SdpaPrefillEngine&
    operator=(Sm100SdpaPrefillEngine&& other) noexcept {
        if (this != &other) {
            if (module_) {
                detail::cuda_library_unload(module_);
            }
            spec_            = other.spec_;
            d_               = other.d_;
            sm_version_      = other.sm_version_;
            sm_count_        = other.sm_count_;
            support_checked_ = other.support_checked_;
            module_          = other.module_;
            kernelPtr_       = other.kernelPtr_;
            cubin_           = std::move(other.cubin_);
            cubinSize_       = other.cubinSize_;
            tile_m_          = other.tile_m_;
            tile_n_          = other.tile_n_;
            tile_k_          = other.tile_k_;
            smemBytes_       = other.smemBytes_;
            kernelName_      = std::move(other.kernelName_);
            built_           = other.built_;
            other.module_    = nullptr;
            other.kernelPtr_ = nullptr;
            other.built_     = false;
        }
        return *this;
    }

    // ---- Phase 1: Support check (spec lookup, cache SM count) ----

    /// @brief Validate that the given attention shape and SM version are supported.
    /// @param shape Attention shape (batch, heads, seqlens, head dim).
    /// @param sm_version SM version (e.g. 100 for SM100).
    /// @return error_t with OK on success, GRAPH_NOT_SUPPORTED if unsupported.
    error_t
    check_support(AttentionShape_t shape, int sm_version) override {
        spec_ = lookup_sm100_kernel_spec(shape.d_qk, sm_version);
        RETURN_CUDNN_FRONTEND_ERROR_IF(
            spec_ == nullptr,
            error_code_t::GRAPH_NOT_SUPPORTED,
            "Unsupported config: d=" + std::to_string(shape.d_qk) + " sm_version=" + std::to_string(sm_version));

        d_          = shape.d_qk;
        sm_version_ = sm_version;
        tile_m_     = spec_->tile_m;
        tile_n_     = spec_->tile_n;
        tile_k_     = spec_->tile_k;
        smemBytes_  = spec_->smem_bytes;
        kernelName_ = spec_->kernel_name;

        // Cache SM count for later use in execute()
        cudaDeviceProp devProp;
        int device_ordinal   = 0;
        cudaError_t cuda_err = detail::cuda_get_device(&device_ordinal);
        RETURN_CUDNN_FRONTEND_ERROR_IF(cuda_err != cudaSuccess, error_code_t::CUDA_API_FAILED, "cudaGetDevice failed");
        cuda_err = detail::cuda_get_device_properties(&devProp, device_ordinal);
        RETURN_CUDNN_FRONTEND_ERROR_IF(
            cuda_err != cudaSuccess, error_code_t::CUDA_API_FAILED, "cudaGetDeviceProperties failed");
        sm_count_ = devProp.multiProcessorCount;

        support_checked_ = true;
        return {error_code_t::OK, ""};
    }

    /// @brief Returns the workspace size in bytes needed by execute().
    /// SM100 does not use a workspace tile counter, but we keep the interface
    /// consistent with SM90 for uniform allocation by the caller.
    static int64_t
    get_workspace_size() {
        return 16;
    }

    // ---- Phase 2: NVRTC compilation + module loading ----

    /// @brief Compile the kernel via NVRTC and load the resulting CU module.
    /// @pre check_support() must have been called successfully.
    /// @return error_t with OK on success.
    error_t
    build() override {
        RETURN_CUDNN_FRONTEND_ERROR_IF(!support_checked_ || spec_ == nullptr,
                                       error_code_t::INVALID_VALUE,
                                       "build() called before check_support()");

        auto status = compile_and_load_kernel(spec_, module_, kernelPtr_, cubin_, cubinSize_, kernelName_);
        if (status.is_good()) {
            built_ = true;
        }
        return status;
    }

    // ---- Phase 3: Set up TMA descriptors, pack params, launch ----

    /// @brief Execute the SM100 SDPA prefill kernel.
    ///
    /// Sets up TMA descriptors for Q/K/V/O, computes L2-aware tile scheduling
    /// parameters, and launches the kernel on the given stream.
    ///
    /// Strides are in ELEMENTS (not bytes), matching PyTorch's .stride() convention.
    /// All stride vectors have 4 elements in (b, h, s, d) order.
    /// For max/sum_exp the 4th element (d-stride) is typically 1 and is unused internally.
    ///
    /// @param batch Batch size (B).
    /// @param heads_q Number of query heads (H_q).
    /// @param heads_kv Number of key/value heads (H_kv).
    /// @param seq_q Query sequence length (S_q).
    /// @param seq_kv Key/value sequence length (S_kv).
    /// @param d Head dimension (d_qk = d_v).
    /// @param d_Q Device pointer to Q tensor (fp16).
    /// @param q_strides Q strides in elements: [b, h, s, d].
    /// @param d_K Device pointer to K tensor (fp16).
    /// @param k_strides K strides in elements: [b, h, s, d].
    /// @param d_V Device pointer to V tensor (fp16).
    /// @param v_strides V strides in elements: [b, h, s, d].
    /// @param d_O Device pointer to O tensor (fp16, output).
    /// @param o_strides O strides in elements: [b, h, s, d].
    /// @param d_max Device pointer to per-row max values (fp32, [B, H_q, S_q]).
    /// @param max_strides Max strides in elements: [b, h, s].
    /// @param d_sum_exp Device pointer to per-row sum-of-exp values (fp32, [B, H_q, S_q]).
    /// @param se_strides Sum-exp strides in elements: [b, h, s].
    /// @param workspace Device workspace pointer (at least 16 bytes, unused by SM100 but kept for API consistency).
    /// @param device Device ordinal for kernel attribute setting.
    /// @param stream CUDA stream for asynchronous execution.
    /// @param user_attn_scale Optional attention scale; defaults to 1/sqrt(D).
    /// @return error_t with OK on success.
    error_t
    execute(int batch,
            int heads_q,
            int heads_kv,
            int seq_q,
            int seq_kv,
            int d,
            void* d_Q,
            std::vector<int64_t> const& q_strides,
            void* d_K,
            std::vector<int64_t> const& k_strides,
            void* d_V,
            std::vector<int64_t> const& v_strides,
            void* d_O,
            std::vector<int64_t> const& o_strides,
            void* d_max,
            std::vector<int64_t> const& max_strides,
            void* d_sum_exp,
            std::vector<int64_t> const& se_strides,
            void* workspace,
            int device,
            cudaStream_t stream,
            std::optional<float> user_attn_scale = std::nullopt) override {
        RETURN_CUDNN_FRONTEND_ERROR_IF(!built_, error_code_t::INVALID_VALUE, "execute() called before build()");
        RETURN_CUDNN_FRONTEND_ERROR_IF(
            d > 128, error_code_t::GRAPH_NOT_SUPPORTED, "SM100 OSS engine requires d <= 128");
        RETURN_CUDNN_FRONTEND_ERROR_IF(
            q_strides.size() < 4 || k_strides.size() < 4 || v_strides.size() < 4 || o_strides.size() < 4,
            error_code_t::INVALID_VALUE,
            "Q/K/V/O strides must have at least 4 elements");
        RETURN_CUDNN_FRONTEND_ERROR_IF(max_strides.size() < 3 || se_strides.size() < 3,
                                       error_code_t::INVALID_VALUE,
                                       "max/sum_exp strides must have at least 3 elements");

        // Suppress unused-parameter warning for workspace (SM100 does not use a tile counter)
        (void)workspace;

        int B      = batch;
        int H_q    = heads_q;
        int H_kv   = heads_kv;
        int S_q    = seq_q;
        int S_kv   = seq_kv;
        int D      = d;
        int TILE_M = tile_m_;
        int TILE_N = tile_n_;

        // ---- Set up AttentionDescriptor ----
        AttentionDescriptor_t attnDesc = {};
        attnDesc.b                     = B;
        attnDesc.q_h                   = H_q;
        attnDesc.k_h                   = H_kv;
        attnDesc.v_h                   = H_kv;
        attnDesc.s_q                   = S_q;
        attnDesc.s_kv                  = S_kv;
        attnDesc.d_qk                  = D;
        attnDesc.d_v                   = D;
        attnDesc.q_heads_per_k         = (H_q >= H_kv && H_kv > 0) ? (uint16_t)(H_q / H_kv) : 1;
        attnDesc.q_heads_per_v         = (H_q >= H_kv && H_kv > 0) ? (uint16_t)(H_q / H_kv) : 1;
        attnDesc.min_q_heads_per_kv    = attnDesc.q_heads_per_k;

        // ---- L2-aware tile scheduling ----
        // SM100 uses an L2-aware swizzle pattern for better cache utilization
        // instead of SM90's persistent tile counter approach.
        constexpr int Tiles_Q      = 2;
        constexpr uint32_t size_l2 = 50u * 1024u * 1024u;

        int ctas_in_q_dim = div_up(S_q, TILE_M * Tiles_Q);
        int num_hb        = B * H_q;

        // Compute the swizzle factor based on how many KV heads fit in L2
        uint32_t one_kv_head_size = static_cast<uint32_t>(S_kv) * (D + D) * sizeof(half);
        int swizzle               = (one_kv_head_size > 0) ? (1 << find_log_2_floor(size_l2 / one_kv_head_size)) : 1;
        swizzle                   = std::max(1, std::min(swizzle, num_hb));

        int num_hb_quotient  = num_hb / swizzle;
        int num_hb_remainder = num_hb % swizzle;
        int num_block        = ctas_in_q_dim;
        int total_blocks     = num_block * num_hb;

        // Fast divisors for tile index decomposition in the kernel
        FastDivisor_t num_head_divmod = make_fast_divisor(static_cast<uint32_t>(H_q));
        FastDivisor_t l2_minor_divmod = make_fast_divisor(static_cast<uint32_t>(swizzle));
        FastDivisor_t l2_major_divmod = make_fast_divisor(static_cast<uint32_t>(swizzle * num_block));
        FastDivisor_t l2_minor_residual_divmod =
            make_fast_divisor(static_cast<uint32_t>(std::max(1, num_hb_remainder)));

        // ---- TMA descriptors (4D) ----
        // Caller strides are in elements; TMA needs bytes.
        uint32_t elem_bytes   = sizeof(half);
        uint32_t elems_per_ld = std::min(64, D);

        // Q TMA: dims = (D, S_q, H_q, B), box = (elems_per_ld, TILE_M)
        // SM100 uses TILE_M (128) directly, unlike SM90 which uses TILE_M * 2.
        CUtensorMap tma_q = {};
        CHECK_CUDNN_FRONTEND_ERROR(create_tma_desc_4d(&tma_q,
                                                      d_Q,
                                                      CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
                                                      D,
                                                      S_q,
                                                      H_q,
                                                      B,
                                                      (uint64_t)q_strides[2] * elem_bytes,
                                                      (uint64_t)q_strides[1] * elem_bytes,
                                                      (uint64_t)q_strides[0] * elem_bytes,
                                                      elems_per_ld,
                                                      TILE_M,
                                                      CU_TENSOR_MAP_SWIZZLE_128B));

        // K TMA: dims = (D, S_kv, H_kv, B), box = (elems_per_ld, TILE_N)
        CUtensorMap tma_k = {};
        CHECK_CUDNN_FRONTEND_ERROR(create_tma_desc_4d(&tma_k,
                                                      d_K,
                                                      CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
                                                      D,
                                                      S_kv,
                                                      H_kv,
                                                      B,
                                                      (uint64_t)k_strides[2] * elem_bytes,
                                                      (uint64_t)k_strides[1] * elem_bytes,
                                                      (uint64_t)k_strides[0] * elem_bytes,
                                                      elems_per_ld,
                                                      TILE_N,
                                                      CU_TENSOR_MAP_SWIZZLE_128B));

        // V TMA: dims = (D, S_kv, H_kv, B), box = (elems_per_ld, TILE_N)
        CUtensorMap tma_v = {};
        CHECK_CUDNN_FRONTEND_ERROR(create_tma_desc_4d(&tma_v,
                                                      d_V,
                                                      CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
                                                      D,
                                                      S_kv,
                                                      H_kv,
                                                      B,
                                                      (uint64_t)v_strides[2] * elem_bytes,
                                                      (uint64_t)v_strides[1] * elem_bytes,
                                                      (uint64_t)v_strides[0] * elem_bytes,
                                                      elems_per_ld,
                                                      TILE_N,
                                                      CU_TENSOR_MAP_SWIZZLE_128B));

        // O TMA: dims = (D, S_q, H_q, B), box = (elems_per_ld, TILE_M)
        // SM100 uses TILE_M (128) for the O box dim, unlike SM90 which uses 16.
        CUtensorMap tma_o = {};
        CHECK_CUDNN_FRONTEND_ERROR(create_tma_desc_4d(&tma_o,
                                                      d_O,
                                                      CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
                                                      D,
                                                      S_q,
                                                      H_q,
                                                      B,
                                                      (uint64_t)o_strides[2] * elem_bytes,
                                                      (uint64_t)o_strides[1] * elem_bytes,
                                                      (uint64_t)o_strides[0] * elem_bytes,
                                                      elems_per_ld,
                                                      TILE_M,
                                                      CU_TENSOR_MAP_SWIZZLE_128B));

        // ---- Scaling ----
        float attn_scale = user_attn_scale.value_or(1.0f / std::sqrt((float)D));
        float tensor_13  = -INFINITY;

        // ---- Tensor descriptors for max and sum_exp (f32, [B, H_q, S_q]) ----
        // Strides provided by caller in elements.
        tensor_descriptor desc_max = {};
        desc_max.num_dims          = 3;
        desc_max.dims[0]           = B;
        desc_max.dims[1]           = H_q;
        desc_max.dims[2]           = S_q;
        desc_max.strides[0]        = max_strides[0];
        desc_max.strides[1]        = max_strides[1];
        desc_max.strides[2]        = max_strides[2];

        tensor_descriptor desc_sum_exp = {};
        desc_sum_exp.num_dims          = 3;
        desc_sum_exp.dims[0]           = B;
        desc_sum_exp.dims[1]           = H_q;
        desc_sum_exp.dims[2]           = S_q;
        desc_sum_exp.strides[0]        = se_strides[0];
        desc_sum_exp.strides[1]        = se_strides[1];
        desc_sum_exp.strides[2]        = se_strides[2];

        // ---- Grid/Block ----
        // SM100 launches total_blocks CTAs with 512 threads each.
        // No persistent scheduling -- the L2 swizzle parameters handle work distribution.
        dim3 grid(total_blocks, 1, 1);
        dim3 block(512, 1, 1);

        // ---- Build kernel params array (18 parameters) ----
        // The parameter order must match the SM100 kernel's expected layout exactly.
        void* kernelParams[] = {
            (void*)&attnDesc,
            (void*)&num_head_divmod,
            (void*)&l2_minor_divmod,
            (void*)&l2_major_divmod,
            (void*)&l2_minor_residual_divmod,
            (void*)&num_hb_quotient,
            (void*)&num_block,
            (void*)&total_blocks,
            (void*)&tma_q,
            (void*)&tma_k,
            (void*)&attn_scale,
            (void*)&tensor_13,
            (void*)&d_max,
            (void*)&desc_max,
            (void*)&d_sum_exp,
            (void*)&desc_sum_exp,
            (void*)&tma_v,
            (void*)&tma_o,
        };

        // ---- Debug logging ----
        if (cudnn_frontend::isLoggingEnabled()) {
            CUDNN_FE_LOG_LABEL_ENDL("SM100 OSS Engine Launch:");
            CUDNN_FE_LOG_LABEL_ENDL("  Grid = (" << grid.x << "," << grid.y << "," << grid.z << ")");
            CUDNN_FE_LOG_LABEL_ENDL("  Block = (" << block.x << "," << block.y << "," << block.z << ")");
            CUDNN_FE_LOG_LABEL_ENDL("  Smem = " << smemBytes_);
            CUDNN_FE_LOG_LABEL_ENDL("  Kernel: " << kernelName_);
            CUDNN_FE_LOG_LABEL_ENDL("  B=" << B << " H_q=" << H_q << " H_kv=" << H_kv << " S_q=" << S_q
                                           << " S_kv=" << S_kv << " D=" << D);
            CUDNN_FE_LOG_LABEL_ENDL("  L2: swizzle=" << swizzle << " num_hb_quotient=" << num_hb_quotient
                                                     << " num_block=" << num_block << " total_blocks=" << total_blocks);
            CUDNN_FE_LOG_LABEL_ENDL("  attn_scale=" << attn_scale);
        }

        // ---- Set shared memory attribute and launch ----
        cudaError_t cuda_err;

        cuda_err = detail::cuda_kernel_set_attribute_for_device(
            kernelPtr_, cudaFuncAttributeMaxDynamicSharedMemorySize, smemBytes_, device);
        RETURN_CUDNN_FRONTEND_ERROR_IF(cuda_err != cudaSuccess,
                                       error_code_t::CUDA_API_FAILED,
                                       "cudaKernelSetAttributeForDevice failed (smem=" + std::to_string(smemBytes_) +
                                           " bytes, error=" + detail::cuda_error_to_string(cuda_err) + ")");

        cuda_err = detail::cuda_launch_kernel((const void*)kernelPtr_, grid, block, kernelParams, smemBytes_, stream);
        cudaError_t last_err = detail::cuda_get_last_error();
        RETURN_CUDNN_FRONTEND_ERROR_IF(cuda_err != cudaSuccess,
                                       error_code_t::CUDA_API_FAILED,
                                       "cudaLaunchKernel failed (grid=" + std::to_string(grid.x) +
                                           ", block=512, smem=" + std::to_string(smemBytes_) +
                                           ", error=" + detail::cuda_error_to_string(cuda_err) + ")");

        RETURN_CUDNN_FRONTEND_ERROR_IF(
            last_err != cudaSuccess,
            error_code_t::CUDA_API_FAILED,
            "cudaGetLastError after kernel launch: " + detail::cuda_error_to_string(last_err));

        return {error_code_t::OK, ""};
    }

   private:
    // State from check_support()
    const KernelSpec* spec_ = nullptr;
    int d_                  = 0;
    int sm_version_         = 0;
    int sm_count_           = 0;
    bool support_checked_   = false;

    // State from build()
    cudaLibrary_t module_   = nullptr;
    cudaKernel_t kernelPtr_ = nullptr;
    std::unique_ptr<char[]> cubin_;
    size_t cubinSize_ = 0;
    int tile_m_ = 0, tile_n_ = 0, tile_k_ = 0;
    int smemBytes_ = 0;
    std::string kernelName_;
    bool built_ = false;
};

}  // namespace experimental
}  // namespace cudnn_frontend

#if defined(_MSC_VER)
#pragma warning(pop)
#endif
