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

#include "nvrtc_shim.h"
#include "attention_utils.h"
#include "../graph_helpers.h"
#include "../../cudnn_frontend_shim.h"

#include <cuda_runtime.h>

#include <cstdint>
#include <fstream>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace cudnn_frontend {
namespace experimental {

// ============================================================
// IOssSdpaEngine — virtual interface for OSS SDPA engines
// Implemented by Sm90SdpaPrefillEngine and Sm100SdpaPrefillEngine.
// ============================================================

class IOssSdpaEngine {
   public:
    virtual ~IOssSdpaEngine() = default;

    virtual error_t
    check_support(AttentionShape_t shape, int sm_version) = 0;
    virtual error_t
    build() = 0;

    virtual error_t
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
            std::optional<float> user_attn_scale = std::nullopt) = 0;

    static int64_t
    get_workspace_size() {
        return 16;
    }
};

// ============================================================
// IOssNormEngine — virtual interface for OSS norm+activation engines.
// Implemented by Sm100RmsNormSiluEngine (SM100/Blackwell).
// Future implementations: Sm90, Sm110, LayerNorm variants, etc.
// ============================================================

// RmsNormSiluDtype is defined in sm100_rms_norm_silu_knobs.h (self-contained,
// also used by standalone tests without the framework). All arch-specific
// engine implementations share this enum.
//
// Forward-declared here; concrete implementations #include the knobs header.
enum class RmsNormSiluDtype : uint8_t;

// Problem shape for RmsNorm+SiLU (extensible for future fields).
struct NormSiluShape_t {
    int C          = 0;               // hidden dimension (columns)
    int num_tokens = 0;               // number of rows
    RmsNormSiluDtype output_dtype{};  // output data type (bf16, fp8, nvfp4)
};

// Optional parameters for FP8 / NVFP4 output modes.
struct RmsNormSiluExtraParams {
    // FP8 output: quantization scale (float*). If nullptr, uses 1.0 from workspace.
    void* fp8_scale     = nullptr;
    void* fp8_scale_inv = nullptr;  // optional: written by kernel if hasScaleInv
    void* fp8_amax      = nullptr;  // optional: written by kernel if hasAmax

    // NVFP4 (1D1X1X) output: row-wise block-scale factors (float*, written by kernel)
    // Shape: [num_tokens, ceil(C / 16)]
    void* nvfp4_scale_row = nullptr;
};

class IOssNormEngine {
   public:
    virtual ~IOssNormEngine() = default;

    // Phase 1: Validate that (shape, sm_version) is supported.
    virtual error_t
    check_support(NormSiluShape_t shape, int sm_version) = 0;

    // Phase 2: NVRTC compile the kernel for the selected config.
    virtual error_t
    build() = 0;

    // Phase 3: Launch the kernel.
    //   input   — [num_tokens, C] bf16 input tensor
    //   output  — [num_tokens, C] output tensor (bf16, fp8, or nvfp4)
    //   weight  — [C] bf16 gamma weights
    //   bias    — [C] bf16 beta bias (can be nullptr)
    //   rows    — num_tokens
    //   cols    — C (hidden dimension)
    //   epsilon — RMSNorm epsilon (for L2Norm equiv: eps_l2 / C)
    //   extra   — optional FP8/NVFP4 params (can be empty)
    virtual error_t
    execute(void* input,
            void* output,
            void* weight,
            void* bias,
            int rows,
            int cols,
            float epsilon,
            void* workspace,
            int device,
            cudaStream_t stream,
            RmsNormSiluExtraParams const& extra = {}) = 0;

    // Workspace size in bytes. The kernel needs an `rs` (inverse RMS) buffer,
    // plus a float for FP8 default scale if no external scale is provided.
    virtual int64_t
    get_workspace_size() const = 0;
};

// ============================================================
// Shared NVRTC compilation + module loading
// Used by both Sm90 and Sm100 engines to avoid code duplication.
// ============================================================

inline error_t
compile_and_load_kernel(const KernelSpec* spec,
                        cudaLibrary_t& module,
                        cudaKernel_t& kernelPtr,
                        std::unique_ptr<char[]>& cubin,
                        size_t& cubinSize,
                        std::string& kernelName) {
    // Ensure NVRTC library is available
    RETURN_CUDNN_FRONTEND_ERROR_IF(
        !detail::nvrtc_is_loaded(), error_code_t::CUDA_API_FAILED, "NVRTC library could not be loaded");

    // Parse flags from embedded string
    std::vector<std::string> flags = parse_flags_string(spec->flags_raw, spec->flags_len);
    std::vector<const char*> flag_ptrs;
    flag_ptrs.reserve(flags.size());
    for (auto& f : flags) {
        flag_ptrs.push_back(f.c_str());
    }

    // Create NVRTC program from embedded source
    nvrtcProgram prog;
    nvrtcResult nvrtc_err;

    nvrtc_err = detail::nvrtc_create_program(&prog, spec->source, "kernel.cu", 0, nullptr, nullptr);
    RETURN_CUDNN_FRONTEND_ERROR_IF(
        nvrtc_err != NVRTC_SUCCESS, error_code_t::CUDA_API_FAILED, "nvrtcCreateProgram failed");

    // Compile
    nvrtcResult compResult = detail::nvrtc_compile_program(prog, (int)flag_ptrs.size(), flag_ptrs.data());

    if (cudnn_frontend::isLoggingEnabled()) {
        // Write the source to a file for debugging
        std::ofstream ofs("kernel.cu");
        ofs << spec->source;

        // Try to dump PTX (cicc may have succeeded even if ptxas failed)
        size_t ptxSize = 0;
        if (detail::nvrtc_get_ptx_size(prog, &ptxSize) == NVRTC_SUCCESS && ptxSize > 1) {
            std::vector<char> ptx(ptxSize);
            if (detail::nvrtc_get_ptx(prog, ptx.data()) == NVRTC_SUCCESS) {
                std::ofstream ptx_ofs("kernel.ptx");
                ptx_ofs.write(ptx.data(), static_cast<std::streamsize>(ptxSize - 1));
            }
        }
    }

    if (compResult != NVRTC_SUCCESS) {
        // Try to retrieve the compilation log for diagnostics
        size_t logSize = 0;
        detail::nvrtc_get_program_log_size(prog, &logSize);
        std::string log_msg = "NVRTC compilation failed";
        if (logSize > 1) {
            std::vector<char> log(logSize);
            detail::nvrtc_get_program_log(prog, log.data());
            log_msg += ": ";
            log_msg += log.data();
        }

        CUDNN_FE_LOG_LABEL_ENDL("ERROR: NVRTC compilation failed: " << log_msg);
        detail::nvrtc_destroy_program(&prog);
        return {error_code_t::CUDA_API_FAILED, log_msg};
    }

    // Extract CUBIN
    nvrtc_err = detail::nvrtc_get_cubin_size(prog, &cubinSize);
    RETURN_CUDNN_FRONTEND_ERROR_IF(
        nvrtc_err != NVRTC_SUCCESS, error_code_t::CUDA_API_FAILED, "nvrtcGetCUBINSize failed");

    cubin = std::make_unique<char[]>(cubinSize);

    nvrtc_err = detail::nvrtc_get_cubin(prog, cubin.get());
    RETURN_CUDNN_FRONTEND_ERROR_IF(nvrtc_err != NVRTC_SUCCESS, error_code_t::CUDA_API_FAILED, "nvrtcGetCUBIN failed");

    nvrtc_err = detail::nvrtc_destroy_program(&prog);
    RETURN_CUDNN_FRONTEND_ERROR_IF(
        nvrtc_err != NVRTC_SUCCESS, error_code_t::CUDA_API_FAILED, "nvrtcDestroyProgram failed");

    // Load module + extract kernel function
    cudaError_t cuda_err;

    cuda_err = detail::cuda_library_load_data(&module, cubin.get(), nullptr, nullptr, 0, nullptr, nullptr, 0);
    RETURN_CUDNN_FRONTEND_ERROR_IF(cuda_err != cudaSuccess,
                                   error_code_t::CUDA_API_FAILED,
                                   "cudaLibraryLoadData failed (cubin_size=" + std::to_string(cubinSize) +
                                       ", error=" + detail::cuda_error_to_string(cuda_err) + ")");

    kernelName = spec->kernel_name;
    cuda_err   = detail::cuda_library_get_kernel(&kernelPtr, module, kernelName.c_str());
    RETURN_CUDNN_FRONTEND_ERROR_IF(
        cuda_err != cudaSuccess,
        error_code_t::CUDA_API_FAILED,
        "cudaLibraryGetKernel failed: " + detail::cuda_error_to_string(cuda_err) + " (kernel: " + kernelName + ")");

    CUDNN_FE_LOG_LABEL_ENDL("INFO: NVRTC compilation successful, kernel: " << kernelName);
    return {error_code_t::OK, ""};
}

}  // namespace experimental
}  // namespace cudnn_frontend
