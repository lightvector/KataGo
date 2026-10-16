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
#include "sm100_rms_norm_silu_knobs.h"
// LN headers and kernel source for NVRTC compilation.
#include "../generated/rms_norm_silu/sm100/ln_headers.h"
#include "../generated/rms_norm_silu/sm100/ln_fwd_silu_kernel.h"

#include <cuda_fp16.h>
#include <algorithm>
#include <cmath>
#include <string>
#include <memory>

namespace cudnn_frontend {
namespace experimental {

// ============================================================
// PersistentLnFwdParams — matches the kernel's expected parameter struct.
// Matches the kernel's expected parameter struct layout (PersistentLnFwdParams).
// ============================================================

struct PersistentLnFwdParams {
    int ctas_per_col = 0;
    int rows         = 0;
    int cols         = 0;
    int batchSize    = 1;
    int seqLen       = 1;

    void* x     = nullptr;
    void* mu    = nullptr;
    void* rs    = nullptr;
    void* gamma = nullptr;

    void* workspace = nullptr;
    int* barrier    = nullptr;

    bool isRMSNorm    = false;
    bool noScale      = false;
    bool noBias       = false;
    bool isAdaLN      = false;
    bool isBatchFirst = true;

    // Output
    void* z       = nullptr;
    void* beta    = nullptr;
    float epsilon = 0.f;

    // FP8
    bool fp8_out     = false;
    float* scale     = nullptr;
    float* scale_inv = nullptr;
    float* amax      = nullptr;

    // Block scale (NVFP4) — field order must match CuDNN's ln.h exactly
    void* scale_row = nullptr;
    void* scale_col = nullptr;  // only used in 1d2x2x
    void* z_col     = nullptr;  // only used in 1d2x2x
    void* z_math    = nullptr;  // only used for 1d2x2x colwise kernel
};

// ============================================================
// reduced_divisor — matches the kernel's expected second argument.
// Matches the kernel's expected second argument (reduced_divisor).
// ============================================================

struct reduced_divisor {
    uint32_t mul_coeff   = 0;
    uint32_t shift_coeff = 0;
    int32_t y            = 0;

    reduced_divisor() = default;
    reduced_divisor(int32_t _y) : y(_y) {
        if (_y <= 1) {
            mul_coeff   = 0;
            shift_coeff = 0;
            return;
        }
        // Find shift such that 2^(32+shift) / y fits in 32 bits
        for (shift_coeff = 0; shift_coeff < 32; ++shift_coeff) {
            uint64_t one = 1;
            uint64_t num = (one << (32 + shift_coeff));
            uint64_t den = static_cast<uint64_t>(_y);
            if (num / den < (one << 32)) {
                mul_coeff = static_cast<uint32_t>(num / den + 1);
                break;
            }
        }
    }
};

// ============================================================
// Sm100RmsNormSiluEngine
// ============================================================

class Sm100RmsNormSiluEngine : public IOssNormEngine {
   public:
    Sm100RmsNormSiluEngine() = default;

    ~Sm100RmsNormSiluEngine() override {
        if (module_) {
            detail::cuda_library_unload(module_);
        }
    }

    // Non-copyable
    Sm100RmsNormSiluEngine(const Sm100RmsNormSiluEngine&) = delete;
    Sm100RmsNormSiluEngine&
    operator=(const Sm100RmsNormSiluEngine&) = delete;

    // ---- Phase 1: Support check + knob selection ----
    error_t
    check_support(NormSiluShape_t shape, int sm_version) override {
        RETURN_CUDNN_FRONTEND_ERROR_IF(
            sm_version < 80, error_code_t::GRAPH_NOT_SUPPORTED, "RmsNormSiluEngine requires SM80+");

        // FP8 output requires SM89+ (Ada Lovelace / Hopper)
        RETURN_CUDNN_FRONTEND_ERROR_IF(shape.output_dtype == RmsNormSiluDtype::FP8 && sm_version < 89,
                                       error_code_t::GRAPH_NOT_SUPPORTED,
                                       "FP8 output requires SM89+ (Ada/Hopper)");

        // NVFP4 output requires SM100+ (Blackwell)
        RETURN_CUDNN_FRONTEND_ERROR_IF(shape.output_dtype == RmsNormSiluDtype::NVFP4 && sm_version < 100,
                                       error_code_t::GRAPH_NOT_SUPPORTED,
                                       "NVFP4 output requires SM100+ (Blackwell)");

        sm_version_ = sm_version;

        // Look up knobs: SM100 uses sweep-tuned LUT, other archs use fallback heuristic
        knobs_ = lookup_rms_norm_silu_knobs(shape.C, shape.num_tokens, shape.output_dtype, sm_version);
        RETURN_CUDNN_FRONTEND_ERROR_IF(
            knobs_ == nullptr,
            error_code_t::GRAPH_NOT_SUPPORTED,
            "Unsupported problem size: C=" + std::to_string(shape.C) + " tokens=" + std::to_string(shape.num_tokens));

        C_             = shape.C;
        num_tokens_    = shape.num_tokens;
        output_dtype_  = shape.output_dtype;
        bytes_per_ldg_ = knobs_->bytes_per_ldg;

        // Derived config
        warps_m_         = knobs_->warps_m;
        warps_n_         = 1;  // Always 1
        threads_per_cta_ = warps_m_ * warps_n_ * 32;

        // Estimate CTAS_PER_ROW when the sweep found column splitting optimal.
        // Matches CuDNN's estimate_ctas_per_row() from LayerNorm_common.cpp:
        //   Find largest LDGS per CTA that avoids register spilling (< 64/NUM_ELTS),
        //   then CTAS_PER_ROW = total_ldgs / ldgs_per_cta.
        // kernel_cfg=2 (non-persistent) requires CTAS_PER_ROW=1.
        if (knobs_->split_cols == 4 && knobs_->kernel_cfg != 2) {
            int input_size   = 2;  // bf16
            int num_elts     = bytes_per_ldg_ / input_size;
            int elts_per_ldg = num_elts * warps_n_ * 32;
            if (elts_per_ldg > 0 && shape.C % elts_per_ldg == 0) {
                int ldgs_per_row                 = shape.C / elts_per_ldg;
                int ldgs_to_cause_register_spill = (num_elts > 0) ? (64 / num_elts) : 1;
                ctas_per_row_                    = 1;
                for (int ldgs = std::min(ldgs_per_row, ldgs_to_cause_register_spill - 1); ldgs > 0; ldgs--) {
                    if (ldgs_per_row % ldgs == 0) {
                        ctas_per_row_ = ldgs_per_row / ldgs;
                        break;
                    }
                }
            } else {
                ctas_per_row_ = 1;
            }
        } else {
            ctas_per_row_ = 1;
        }
        is_cooperative_ = (ctas_per_row_ > 1);

        // Cache SM count
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

    // Workspace layout (all offsets 128-byte aligned):
    //   [0]                 rs              : num_tokens * sizeof(float)
    //   [aligned]           fp8_scale       : sizeof(float)
    //   [aligned]           scale_row       : num_tokens * ceil(C/16) bytes (NVFP4 only)
    //   [aligned]           coop_workspace  : ctas_per_col * WARPS_M * CTAS_PER_ROW * sizeof(float2) * 2
    //   [aligned]           coop_barrier    : 2 * ctas_per_col * sizeof(int32_t)
    //   + 128 padding
    int64_t
    get_workspace_size() const override {
        int64_t ws = static_cast<int64_t>(num_tokens_) * sizeof(float);  // rs
        ws         = ((ws + 127) / 128) * 128;
        ws += sizeof(float);  // FP8 default scale
        ws = ((ws + 127) / 128) * 128;
        if (output_dtype_ == RmsNormSiluDtype::NVFP4) {
            ws += static_cast<int64_t>(num_tokens_) * ((C_ + 15) / 16);
            ws = ((ws + 127) / 128) * 128;
        }
        if (is_cooperative_) {
            int ctas_per_col = std::max(1, (num_tokens_ + warps_m_ - 1) / warps_m_);
            // Inter-CTA partial stats: ctas_per_col * WARPS_M * CTAS_PER_ROW * sizeof(float2) * 2
            ws += static_cast<int64_t>(ctas_per_col) * warps_m_ * ctas_per_row_ * sizeof(float) * 2 * 2;
            ws = ((ws + 127) / 128) * 128;
            // Barrier array: 2 * ctas_per_col * sizeof(int32_t)
            ws += 2 * static_cast<int64_t>(ctas_per_col) * sizeof(int32_t);
            ws = ((ws + 127) / 128) * 128;
        }
        ws += 128;  // final alignment
        return ws;
    }

    // Get offset of scale_row within workspace (for reading back in tests).
    // Returns -1 if not NVFP4 output.
    int64_t
    get_scale_row_workspace_offset() const {
        if (output_dtype_ != RmsNormSiluDtype::NVFP4) return -1;
        int64_t off = static_cast<int64_t>(num_tokens_) * sizeof(float);
        off         = ((off + 127) / 128) * 128;
        off += sizeof(float);
        off = ((off + 127) / 128) * 128;
        return off;
    }

    int64_t
    get_scale_row_size_bytes() const {
        if (output_dtype_ != RmsNormSiluDtype::NVFP4) return 0;
        return static_cast<int64_t>(num_tokens_) * ((C_ + 15) / 16);
    }

    // ---- Phase 2: NVRTC compilation ----
    error_t
    build() override {
        RETURN_CUDNN_FRONTEND_ERROR_IF(!support_checked_ || knobs_ == nullptr,
                                       error_code_t::INVALID_VALUE,
                                       "build() called before check_support()");

        // Assemble the full kernel source:
        // 1. Generate constexpr defines from knobs
        // 2. Prepend to headers + kernel body

        // Order: preamble → headers (defines cuda types) → constexpr defines
        // (uses those types) → kernel body (uses everything).
        std::string defines = generate_constexpr_defines();

        // Undefine EWP/RDC to avoid removed cudaCGScopeMultiGrid in cooperative_groups
        std::string preamble = "#undef __CUDACC_EWP__\n#undef __CUDACC_RDC__\n";

        std::string full_source = preamble + std::string(generated::ln_headers_source) + defines +
                                  std::string(generated::ln_fwd_silu_kernel_source);

        // Parse compile flags and override --gpu-architecture for the target SM version
        std::vector<std::string> flags =
            parse_flags_string(generated::ln_fwd_silu_kernel_flags, generated::ln_fwd_silu_kernel_flags_len);

        // Replace the embedded --gpu-architecture with the correct arch for this GPU.
        // The 'a' suffix = architecture-specific (enables SM-specific features like redux.sync).
        // The 'f' suffix = forward-compatible across an SM family (e.g., sm_100f works on SM100-SM103).
        // The kernel source has #if __CUDA_ARCH__ guards for SM-specific instructions.
        {
            std::string target_arch;
            if (sm_version_ == 100)
                target_arch = "sm_100a";  // B200: arch-specific for best codegen
            else if (sm_version_ > 100 && sm_version_ < 110)
                target_arch = "sm_100f";  // GB300 etc: forward-compat
            else if (sm_version_ == 90)
                target_arch = "sm_90a";
            else if (sm_version_ > 90 && sm_version_ < 100)
                target_arch = "sm_90";  // SM92 etc
            else if (sm_version_ >= 89)
                target_arch = "sm_89";
            else if (sm_version_ >= 86)
                target_arch = "sm_86";
            else
                target_arch = "sm_80";

            for (auto& f : flags) {
                if (f.find("--gpu-architecture=") == 0) {
                    f = "--gpu-architecture=" + target_arch;
                    break;
                }
            }
        }

        // Add CUDA Toolkit include paths for NVRTC to resolve #include <cuda_bf16.h> etc.
        std::string cuda_include = "/usr/local/cuda/include";
        if (auto env0 = std::getenv("CUDA_HOME")) {
            cuda_include = std::string(env0) + "/include";
        } else if (auto env1 = std::getenv("CUDA_PATH")) {
            cuda_include = std::string(env1) + "/include";
        }
        flags.push_back("--include-path=" + cuda_include);

        // CCCL headers needed for cuda::ptx:: used by block-scale (NVFP4) code paths.
        // CUDA 13+ ships CCCL under targets/<arch>-linux/include/cccl.
        // Add both x86_64 and aarch64 paths — NVRTC silently ignores non-existent paths.
        std::string cuda_base = cuda_include.substr(0, cuda_include.rfind("/include"));
        for (const char* arch : {"x86_64", "aarch64"}) {
            std::string target_dir = cuda_base + "/targets/" + arch + "-linux/include";
            flags.push_back("--include-path=" + target_dir);
            flags.push_back("--include-path=" + target_dir + "/cccl");
        }

        std::vector<const char*> flag_ptrs;
        for (auto& f : flags) flag_ptrs.push_back(f.c_str());

        // NVRTC compile
        RETURN_CUDNN_FRONTEND_ERROR_IF(
            !detail::nvrtc_is_loaded(), error_code_t::CUDA_API_FAILED, "NVRTC library could not be loaded");

        nvrtcProgram prog;
        nvrtcResult nvrtc_err;

        nvrtc_err = detail::nvrtc_create_program(&prog, full_source.c_str(), "ln_fwd_silu.cu", 0, nullptr, nullptr);
        RETURN_CUDNN_FRONTEND_ERROR_IF(
            nvrtc_err != NVRTC_SUCCESS, error_code_t::CUDA_API_FAILED, "nvrtcCreateProgram failed");

        nvrtcResult compResult = detail::nvrtc_compile_program(prog, (int)flag_ptrs.size(), flag_ptrs.data());
        if (compResult != NVRTC_SUCCESS) {
            size_t logSize = 0;
            detail::nvrtc_get_program_log_size(prog, &logSize);
            std::string log_msg = "NVRTC compilation failed";
            if (logSize > 1) {
                std::vector<char> log(logSize);
                detail::nvrtc_get_program_log(prog, log.data());
                log_msg += ": ";
                log_msg += log.data();
            }
            detail::nvrtc_destroy_program(&prog);
            return {error_code_t::CUDA_API_FAILED, log_msg};
        }

        // Extract CUBIN
        nvrtc_err = detail::nvrtc_get_cubin_size(prog, &cubinSize_);
        RETURN_CUDNN_FRONTEND_ERROR_IF(
            nvrtc_err != NVRTC_SUCCESS, error_code_t::CUDA_API_FAILED, "nvrtcGetCUBINSize failed");

        cubin_    = std::make_unique<char[]>(cubinSize_);
        nvrtc_err = detail::nvrtc_get_cubin(prog, cubin_.get());
        RETURN_CUDNN_FRONTEND_ERROR_IF(
            nvrtc_err != NVRTC_SUCCESS, error_code_t::CUDA_API_FAILED, "nvrtcGetCUBIN failed");

        detail::nvrtc_destroy_program(&prog);

        // Load module + kernel
        cudaError_t cuda_err;
        cuda_err = detail::cuda_library_load_data(&module_, cubin_.get(), nullptr, nullptr, 0, nullptr, nullptr, 0);
        RETURN_CUDNN_FRONTEND_ERROR_IF(cuda_err != cudaSuccess,
                                       error_code_t::CUDA_API_FAILED,
                                       "cudaLibraryLoadData failed: " + detail::cuda_error_to_string(cuda_err));

        cuda_err = detail::cuda_library_get_kernel(&kernelPtr_, module_, "ln_fwd_kernel");
        RETURN_CUDNN_FRONTEND_ERROR_IF(cuda_err != cudaSuccess,
                                       error_code_t::CUDA_API_FAILED,
                                       "cudaLibraryGetKernel failed: " + detail::cuda_error_to_string(cuda_err));

        built_ = true;
        return {error_code_t::OK, ""};
    }

    // ---- Phase 3: Execute ----
    error_t
    execute(void* input,
            void* output,
            void* weight,  // gamma
            void* bias,    // beta (can be nullptr)
            int rows,
            int cols,
            float epsilon,
            void* workspace,
            int device,
            cudaStream_t stream,
            RmsNormSiluExtraParams const& extra = {}) override {
        RETURN_CUDNN_FRONTEND_ERROR_IF(!built_, error_code_t::INVALID_VALUE, "execute() called before build()");

        // Compute grid dimensions
        int ctas_per_col_max = (rows + warps_m_ - 1) / warps_m_;
        int ctas_per_col;
        if (knobs_->kernel_cfg == 2) {
            // Non-persistent mode: launch one CTA group per row (max CTAs).
            // CuDNN overrides ctas_per_col to max when use_non_persistent_mode=true.
            ctas_per_col = ctas_per_col_max;
        } else {
            ctas_per_col = std::min(sm_count_ * static_cast<int>(knobs_->occupancy) / ctas_per_row_, ctas_per_col_max);
        }
        ctas_per_col = std::max(ctas_per_col, 1);

        dim3 grid(ctas_per_row_ * ctas_per_col);
        dim3 block(threads_per_cta_);

        // Pack params
        PersistentLnFwdParams params{};
        params.rows         = rows;
        params.cols         = cols;
        params.ctas_per_col = ctas_per_col;
        params.isRMSNorm    = true;
        params.noScale      = (weight == nullptr);
        params.noBias       = (bias == nullptr);
        params.isBatchFirst = true;
        params.batchSize    = 1;
        params.seqLen       = rows;
        params.epsilon      = epsilon;
        params.x            = input;
        params.z            = output;
        params.gamma        = weight;
        params.beta         = bias;

        // Workspace layout (128-byte aligned):
        //   [0]        rs         : rows * sizeof(float)
        //   [aligned]  fp8_scale  : sizeof(float)
        //   [aligned]  scale_row  : rows * ceil(cols/16) bytes  (NVFP4 only)
        char* ws_ptr = static_cast<char*>(workspace);
        params.rs    = ws_ptr;
        {
            int64_t off = static_cast<int64_t>(rows) * sizeof(float);
            off         = ((off + 127) / 128) * 128;
            ws_ptr      = static_cast<char*>(workspace) + off;
        }

        // FP8 output: set scale pointer
        if (output_dtype_ == RmsNormSiluDtype::FP8) {
            params.fp8_out = true;
            if (extra.fp8_scale) {
                params.scale = static_cast<float*>(extra.fp8_scale);
            } else {
                // Use default scale = 1.0 from workspace (device memory)
                // Write 1.0f (IEEE 754: 0x3F800000) asynchronously on stream
                float* default_scale   = reinterpret_cast<float*>(ws_ptr);
                cudaError_t memset_err = detail::cuda_mem_set_d32_async(default_scale, 0x3F800000u, 1, stream);
                RETURN_CUDNN_FRONTEND_ERROR_IF(memset_err != cudaSuccess,
                                               error_code_t::CUDA_API_FAILED,
                                               "cudaMemcpyAsync for default scale failed");
                params.scale = default_scale;
            }
            params.scale_inv = static_cast<float*>(extra.fp8_scale_inv);  // may be nullptr
            params.amax      = static_cast<float*>(extra.fp8_amax);       // may be nullptr
        }

        // NVFP4 output: set scale_row pointer
        if (output_dtype_ == RmsNormSiluDtype::NVFP4) {
            if (extra.nvfp4_scale_row) {
                params.scale_row = extra.nvfp4_scale_row;
            } else {
                // Auto-allocate from workspace after fp8_scale slot
                int64_t scale_row_off = get_scale_row_workspace_offset();
                params.scale_row      = static_cast<char*>(workspace) + scale_row_off;
            }
        }

        // Multi-CTA cooperative reduction: allocate workspace + barrier from workspace
        if (is_cooperative_) {
            // Compute workspace offset after rs + fp8_scale + scale_row
            int64_t coop_off = static_cast<int64_t>(rows) * sizeof(float);
            coop_off         = ((coop_off + 127) / 128) * 128;
            coop_off += sizeof(float);
            coop_off = ((coop_off + 127) / 128) * 128;
            if (output_dtype_ == RmsNormSiluDtype::NVFP4) {
                coop_off += static_cast<int64_t>(rows) * ((cols + 15) / 16);
                coop_off = ((coop_off + 127) / 128) * 128;
            }

            // Inter-CTA partial stats workspace
            int64_t coop_ws_size =
                static_cast<int64_t>(ctas_per_col) * warps_m_ * ctas_per_row_ * sizeof(float) * 2 * 2;
            params.workspace = static_cast<char*>(workspace) + coop_off;
            coop_off += coop_ws_size;
            coop_off = ((coop_off + 127) / 128) * 128;

            // Barrier array — must be zeroed before each launch
            int64_t barrier_count = 2 * ctas_per_col;
            params.barrier        = reinterpret_cast<int*>(static_cast<char*>(workspace) + coop_off);
            cudaError_t memset_err =
                detail::cuda_mem_set_async(params.barrier, 0, barrier_count * sizeof(int32_t), stream);
            RETURN_CUDNN_FRONTEND_ERROR_IF(
                memset_err != cudaSuccess,
                error_code_t::CUDA_API_FAILED,
                "cudaMemsetAsync failed for cooperative barrier (count=" + std::to_string(barrier_count) +
                    " offset=" + std::to_string(coop_off) + ")");
        }

        // reduced_divisor for batch/seqLen (batchFirst=true, so divide by seqLen=rows)
        reduced_divisor divisor(rows);

        // Kernel args
        void* kernelParams[] = {
            reinterpret_cast<void*>(&params),
            reinterpret_cast<void*>(&divisor),
        };

        // Static shared memory: USE_STATIC_SMEM_VALUE is defined in the NVRTC source,
        // so the kernel allocates smem at compile time. Pass 0 for dynamic smem.
        // The kernel uses compile-time static shared memory (no dynamic smem needed).
        int smem_bytes = 0;

        // Set shared memory attribute
        cudaError_t cuda_err;
        cuda_err = detail::cuda_kernel_set_attribute_for_device(
            kernelPtr_, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes, device);
        RETURN_CUDNN_FRONTEND_ERROR_IF(
            cuda_err != cudaSuccess,
            error_code_t::CUDA_API_FAILED,
            "cudaKernelSetAttributeForDevice failed: " + detail::cuda_error_to_string(cuda_err));

        // Launch
        cuda_err = detail::cuda_launch_kernel((const void*)kernelPtr_, grid, block, kernelParams, smem_bytes, stream);
        RETURN_CUDNN_FRONTEND_ERROR_IF(cuda_err != cudaSuccess,
                                       error_code_t::CUDA_API_FAILED,
                                       "cudaLaunchKernel failed: " + detail::cuda_error_to_string(cuda_err) +
                                           " (grid=" + std::to_string(grid.x) + " block=" + std::to_string(block.x) +
                                           " smem=" + std::to_string(smem_bytes) + ")");

        return {error_code_t::OK, ""};
    }

   private:
    std::string
    generate_constexpr_defines() const {
        std::string s;

        // Type aliases (bf16 input always for VAE)
        s += "\nusing ITYPE = nv_bfloat16;";
        switch (output_dtype_) {
            case RmsNormSiluDtype::BF16:
                s += "\nusing OTYPE = nv_bfloat16;";
                break;
            case RmsNormSiluDtype::FP8:
                s += "\nusing OTYPE = nv_fp8_e4m3;";
                break;
            case RmsNormSiluDtype::NVFP4:
                s += "\nusing OTYPE = nv_fp4_e2m1;";
                break;
        }
        s += "\nusing WTYPE = nv_bfloat16;";
        s += "\nusing CTYPE = float;";
        // NORM_OTYPE must match the norm output type, not compute type.
        // Must match the norm output type, not compute type.
        // For bf16 output: nv_bfloat16.  For FP8/NVFP4: float (block-scale intermediate).
        switch (output_dtype_) {
            case RmsNormSiluDtype::BF16:
                s += "\nusing NORM_OTYPE = nv_bfloat16;";
                break;
            case RmsNormSiluDtype::FP8:
            case RmsNormSiluDtype::NVFP4:
                s += "\nusing NORM_OTYPE = float;";
                break;
        }

        // Kernel config
        s += "\nconstexpr int HIDDEN_SIZE = " + std::to_string(C_) + ";";
        s += "\nconstexpr int BATCH_SIZE = 1;";
        s += "\nconstexpr int CTAS_PER_ROW = " + std::to_string(ctas_per_row_) + ";";
        s += "\nconstexpr int WARPS_M = " + std::to_string(warps_m_) + ";";
        s += "\nconstexpr int WARPS_N = " + std::to_string(warps_n_) + ";";
        s += "\nconstexpr int BYTES_PER_LDG = " + std::to_string(bytes_per_ldg_) + ";";
        s += "\nconstexpr bool isRMSNorm = 1;";
        s += "\nconstexpr bool isAdaLN = 0;";
        s += "\nconstexpr bool isBatchFirst = 1;";
        s += "\nconstexpr bool hasGamma = 1;";
        s += "\nconstexpr bool hasBeta = 0;";
        s += "\nconstexpr bool isZeroCenteredGamma = 0;";
        s += "\nconstexpr bool isZeroCenteredGammaCastBeforeAdd = 0;";

        bool use_smem_gamma     = (knobs_->kernel_cfg == 1);
        bool use_non_persistent = (knobs_->kernel_cfg == 2);
        s += "\nconstexpr bool useSmemGamma = " + std::to_string(use_smem_gamma ? 1 : 0) + ";";
        s += "\nconstexpr bool GAMMA_ON_DEMAND = " + std::to_string((!use_smem_gamma && use_non_persistent) ? 1 : 0) +
             ";";

        bool is_fp8   = (output_dtype_ == RmsNormSiluDtype::FP8);
        bool is_nvfp4 = (output_dtype_ == RmsNormSiluDtype::NVFP4);
        s += "\nconstexpr bool isFP8Out = " + std::to_string(is_fp8 ? 1 : 0) + ";";
        s += "\nconstexpr bool hasScaleInv = 0;";
        s += "\nconstexpr bool hasAmax = 0;";
        s += "\n#define LN_USE_CLUSTER 0";
        s += "\nconstexpr bool USE_CLUSTER = 0;";
        s += "\nconstexpr bool isBlockScaleOut = " + std::to_string(is_nvfp4 ? 1 : 0) + ";";
        s += "\nconstexpr bool isFP4Out = " + std::to_string(is_nvfp4 ? 1 : 0) + ";";
        s += "\nconstexpr bool isBlockScale_1D1X1X = " + std::to_string(is_nvfp4 ? 1 : 0) + ";";
        s += "\nconstexpr bool isBlockScale_1D2X2X = 0;";
        s += "\nconstexpr bool isBlockScale_1D2X2X_Transpose = 0;";
        s += "\nconstexpr bool useBlockScaleColwiseKernel = 0;";
        s += "\nconstexpr int DESIRED_OCCUPANCY = " + std::to_string(knobs_->occupancy) + ";";

        // Ktraits instantiation
        s += "\nusing Ktraits = Kernel_traits<WTYPE, ITYPE, OTYPE, CTYPE, NORM_OTYPE, uint32_t,";
        s += "\n    HIDDEN_SIZE, BATCH_SIZE, CTAS_PER_ROW, WARPS_M, WARPS_N, BYTES_PER_LDG,";
        s += "\n    isRMSNorm, isAdaLN, isBatchFirst, hasGamma, hasBeta, useSmemGamma,";
        s += "\n    USE_CLUSTER, false>;";  // WHOLE_CTA = false

        // Static shared memory: let the kernel template compute the exact struct size.
        // The kernel template computes the exact struct size at compile time.
        // The kernel uses __shared__ char smem[USE_STATIC_SMEM_VALUE] instead of extern __shared__.
        s += "\n#define USE_STATIC_SMEM_VALUE ((int)sizeof(LnFwdShared<Ktraits>))";

        return s;
    }

    // Compute shared memory size matching the kernel's LnFwdShared struct.
    // Every field has at least 1 element and __align__(16).
    //
    // LnFwdShared<Ktraits>:
    //   __align__(16) char smem_stats[SMEM_STATS_ELEMENTS]      — 1 or Stats::SMEM_BYTES
    //   __align__(16) uint64_t smem_bar[SMEM_BAR_ELEMENTS]      — 1 (no cluster)
    //   __align__(16) weight_t smem_gamma[GAMMA_ELEMENTS]       — 1 or LDGS*THREADS_PER_ROW*NUM_ELTS
    //   __align__(16) weight_t smem_beta[BETA_ELEMENTS]         — 1 (no beta)
    //   __align__(16) float smem_mxfp8[SMEM_MXFP8_ELEMENTS]    — 1 (no 1D2X2X)
    int
    compute_smem_bytes() const {
        int elts_per_ldg     = bytes_per_ldg_ / 2;
        int vec_cols         = C_ / elts_per_ldg;
        int threads_per_row  = warps_n_ * 32;
        int vec_cols_per_ldg = ctas_per_row_ * threads_per_row;
        int ldgs             = (vec_cols_per_ldg > 0) ? (vec_cols / vec_cols_per_ldg) : 1;
        int num_elts         = elts_per_ldg;

        auto align16 = [](int bytes) { return ((bytes + 15) / 16) * 16; };

        // smem_stats: min 1 byte → 16 when aligned.
        // When CTAS_PER_ROW=1 && WARPS_N=1, Stats::SMEM_BYTES=0, so element count = max(0,1) = 1.
        int stats_bytes = 1;
        if (ctas_per_row_ > 1 || warps_n_ > 1) {
            stats_bytes = warps_m_ * warps_n_ * sizeof(float) * 2 * 2;
        }
        int smem_stats = align16(stats_bytes);

        // smem_bar: 1 uint64_t (no cluster)
        int smem_bar = align16(static_cast<int>(sizeof(uint64_t)));

        // smem_gamma: kernel_cfg=1 loads gamma to smem
        int gamma_elements = 1;
        if (knobs_->kernel_cfg == 1) {
            gamma_elements = 1 * ldgs * threads_per_row * num_elts;  // BATCH_SIZE=1
        }
        int smem_gamma = align16(gamma_elements * 2);  // weight_t = bf16 = 2 bytes

        // smem_beta: 1 element (hasBeta=false)
        int smem_beta = align16(2);  // 1 * sizeof(bf16)

        // smem_mxfp8: 1 element (not 1D2X2X)
        int smem_mxfp8 = align16(static_cast<int>(sizeof(float)));

        int smem_bytes = smem_stats + smem_bar + smem_gamma + smem_beta + smem_mxfp8;
        smem_bytes     = ((smem_bytes + 127) / 128) * 128;
        return smem_bytes;
    }

    // State from check_support()
    const RmsNormSiluKnobs* knobs_ = nullptr;
    int C_                         = 0;
    int num_tokens_                = 0;
    RmsNormSiluDtype output_dtype_ = RmsNormSiluDtype::BF16;
    int bytes_per_ldg_             = 0;
    int warps_m_                   = 0;
    int warps_n_                   = 0;
    int threads_per_cta_           = 0;
    int ctas_per_row_              = 0;
    int sm_count_                  = 0;
    int sm_version_                = 0;
    bool is_cooperative_           = false;
    bool support_checked_          = false;

    // State from build()
    cudaLibrary_t module_   = nullptr;
    cudaKernel_t kernelPtr_ = nullptr;
    std::unique_ptr<char[]> cubin_;
    size_t cubinSize_ = 0;
    bool built_       = false;
};

}  // namespace experimental
}  // namespace cudnn_frontend
