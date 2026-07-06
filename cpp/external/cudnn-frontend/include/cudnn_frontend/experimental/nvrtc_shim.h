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

#include <nvrtc.h>
#include <cuda_runtime.h>  // for cudaError_t (needed by cudnn_frontend_shim.h wrappers)

// Pull in HMODULE, dlopen, dlsym, dlclose macros and platform definitions
// from the existing cudnn_frontend shim infrastructure.
#include "../../cudnn_frontend_shim.h"

#if defined NV_CUDNN_FRONTEND_USE_DYNAMIC_LOADING
#include <mutex>
#endif

namespace cudnn_frontend {
namespace experimental {
namespace detail {

#if defined NV_CUDNN_FRONTEND_USE_DYNAMIC_LOADING

/// @brief Attempt to dynamically load the NVRTC shared library.
///
/// Tries a sequence of candidate library names in order, returning
/// the handle for the first one that loads successfully.
///
/// @return A valid library handle on success, or nullptr on failure.
///         This function does NOT throw.
inline HMODULE
load_nvrtc_so() {
#ifdef _WIN32
    constexpr const char *candidates[] = {
        "nvrtc64_131_0.dll",
        "nvrtc64_130_0.dll",
        "nvrtc64_120_0.dll",
    };
#else
    constexpr const char *candidates[] = {
        "libnvrtc.so",
        "libnvrtc.so.13",
        "libnvrtc.so.12",
    };
#endif
    constexpr size_t num_candidates = sizeof(candidates) / sizeof(candidates[0]);

    for (size_t i = 0; i < num_candidates; ++i) {
        // Clear any prior error state
        dlerror();

        HMODULE handle = dlopen(candidates[i], RTLD_NOW);
        if (handle) {
            return handle;
        }
    }

    // All candidates failed -- return nullptr (no throw)
    return nullptr;
}

/// @brief Look up a symbol in the dynamically-loaded NVRTC library.
///
/// The library handle is loaded once (lazily, thread-safe) and cached
/// in a function-local static.  If the library could not be loaded,
/// every call returns nullptr.
///
/// @param function_name  The mangled / C symbol name to resolve.
/// @return A generic pointer to the symbol, or nullptr if the library
///         is not loaded or the symbol is not found.
inline void *
get_nvrtc_symbol(const char *function_name) {
    static std::mutex nvrtc_lib_mutex;
    static HMODULE nvrtc_handle = nullptr;
    static bool load_attempted  = false;

    std::lock_guard<std::mutex> lock(nvrtc_lib_mutex);

    if (!load_attempted) {
        load_attempted = true;
        nvrtc_handle   = load_nvrtc_so();
    }

    if (!nvrtc_handle) {
        return nullptr;
    }

    // Clear any existing error before calling dlsym
    dlerror();

    void *symbol = dlsym(nvrtc_handle, function_name);
    // If dlsym fails we simply return nullptr -- callers decide how to
    // report the error (the macro returns NVRTC_ERROR_INTERNAL_ERROR).
    return symbol;
}

/// @brief Check whether the NVRTC library was successfully loaded.
///
/// Internally triggers a lazy load attempt if one has not been made yet.
///
/// @return true if the library handle is valid, false otherwise.
inline bool
nvrtc_is_loaded() {
    // Trigger the lazy load by requesting any symbol (the name is
    // irrelevant -- we only care about whether the handle is valid).
    // get_nvrtc_symbol will initialise the handle on the first call.
    get_nvrtc_symbol("nvrtcVersion");

    // We cannot directly inspect the static inside get_nvrtc_symbol,
    // so we probe for a symbol that must exist in every NVRTC build.
    return get_nvrtc_symbol("nvrtcVersion") != nullptr;
}

/// @brief Macro for calling an NVRTC function via dynamic loading.
///
/// Looks up @p nvrtc_symbol by name, and if the symbol is nullptr
/// returns @c NVRTC_ERROR_INTERNAL_ERROR.  Otherwise casts the
/// pointer to the wrapper function's own signature and tail-calls it.
#define NV_FE_CALL_TO_NVRTC(function_name, nvrtc_symbol, ...) \
    void *fptr = get_nvrtc_symbol(#nvrtc_symbol);             \
    if (fptr == nullptr) {                                    \
        return NVRTC_ERROR_INTERNAL_ERROR;                    \
    }                                                         \
    return reinterpret_cast<decltype(function_name) *>(fptr)(__VA_ARGS__);

#else  // static linking

/// @brief When statically linked NVRTC is always considered loaded.
inline bool
nvrtc_is_loaded() {
    return true;
}

/// @brief Macro for calling an NVRTC function via static linking.
///
/// Simply forwards the call directly to the NVRTC symbol.
#define NV_FE_CALL_TO_NVRTC(function_name, nvrtc_symbol, ...) return nvrtc_symbol(__VA_ARGS__);

#endif  // NV_CUDNN_FRONTEND_USE_DYNAMIC_LOADING

// ---------------------------------------------------------------------------
// Wrapper functions for the NVRTC API
//
// Each wrapper has the exact same signature as the underlying NVRTC
// function it delegates to.  Under dynamic loading the macro resolves
// the symbol at first call; under static linking the call is direct.
// ---------------------------------------------------------------------------

/// @brief Create an NVRTC program from source.
/// @see nvrtcCreateProgram
inline nvrtcResult
nvrtc_create_program(nvrtcProgram *prog,
                     const char *src,
                     const char *name,
                     int numHeaders,
                     const char *const *headers,
                     const char *const *includeNames) {
    NV_FE_CALL_TO_NVRTC(nvrtc_create_program, nvrtcCreateProgram, prog, src, name, numHeaders, headers, includeNames);
}

/// @brief Destroy an NVRTC program.
/// @see nvrtcDestroyProgram
inline nvrtcResult
nvrtc_destroy_program(nvrtcProgram *prog) {
    NV_FE_CALL_TO_NVRTC(nvrtc_destroy_program, nvrtcDestroyProgram, prog);
}

/// @brief Compile an NVRTC program with the given options.
/// @see nvrtcCompileProgram
inline nvrtcResult
nvrtc_compile_program(nvrtcProgram prog, int numOptions, const char *const *options) {
    NV_FE_CALL_TO_NVRTC(nvrtc_compile_program, nvrtcCompileProgram, prog, numOptions, options);
}

/// @brief Query the size of the compilation log (including the NUL terminator).
/// @see nvrtcGetProgramLogSize
inline nvrtcResult
nvrtc_get_program_log_size(nvrtcProgram prog, size_t *logSizeRet) {
    NV_FE_CALL_TO_NVRTC(nvrtc_get_program_log_size, nvrtcGetProgramLogSize, prog, logSizeRet);
}

/// @brief Retrieve the compilation log.
/// @see nvrtcGetProgramLog
inline nvrtcResult
nvrtc_get_program_log(nvrtcProgram prog, char *log) {
    NV_FE_CALL_TO_NVRTC(nvrtc_get_program_log, nvrtcGetProgramLog, prog, log);
}

/// @brief Query the size of the compiled CUBIN image.
/// @see nvrtcGetCUBINSize
inline nvrtcResult
nvrtc_get_cubin_size(nvrtcProgram prog, size_t *cubinSizeRet) {
    NV_FE_CALL_TO_NVRTC(nvrtc_get_cubin_size, nvrtcGetCUBINSize, prog, cubinSizeRet);
}

/// @brief Retrieve the compiled CUBIN image.
/// @see nvrtcGetCUBIN
inline nvrtcResult
nvrtc_get_cubin(nvrtcProgram prog, char *cubin) {
    NV_FE_CALL_TO_NVRTC(nvrtc_get_cubin, nvrtcGetCUBIN, prog, cubin);
}

/// @brief Query the size of the generated PTX.
/// @see nvrtcGetPTXSize
inline nvrtcResult
nvrtc_get_ptx_size(nvrtcProgram prog, size_t *ptxSizeRet) {
    NV_FE_CALL_TO_NVRTC(nvrtc_get_ptx_size, nvrtcGetPTXSize, prog, ptxSizeRet);
}

/// @brief Retrieve the generated PTX.
/// @see nvrtcGetPTX
inline nvrtcResult
nvrtc_get_ptx(nvrtcProgram prog, char *ptx) {
    NV_FE_CALL_TO_NVRTC(nvrtc_get_ptx, nvrtcGetPTX, prog, ptx);
}

/// @brief Map an nvrtcResult code to a human-readable string.
/// @see nvrtcGetErrorString
inline const char *
nvrtc_get_error_string(nvrtcResult result) {
#if defined NV_CUDNN_FRONTEND_USE_DYNAMIC_LOADING
    void *fptr = get_nvrtc_symbol("nvrtcGetErrorString");
    if (fptr == nullptr) {
        return "NVRTC error (library not loaded)";
    }
    return reinterpret_cast<decltype(nvrtc_get_error_string) *>(fptr)(result);
#else
    return nvrtcGetErrorString(result);
#endif
}

}  // namespace detail
}  // namespace experimental
}  // namespace cudnn_frontend
