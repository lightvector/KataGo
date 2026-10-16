/*
 * Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
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

#include <algorithm>
#include <array>
#include <functional>
#include <memory>
#include <sstream>
#include <utility>

#include "../graph_helpers.h"
#include "backend_descriptor.h"

namespace cudnn_frontend {
namespace graph {
class Graph;
}  // namespace graph
///
/// KernelCache Class
/// Wraps the kernel_cache backend descriptor
/// Wraps backend utility functions for user's convenience
/// Backend accessor functions: size()
/// Contains internal utilities for kernel cache finalization and operation graph attributes
///
class KernelCache : public detail::backend_descriptor {
   public:
    friend class graph::Graph;
    // Uses the default backend constructor so that we can check for initialization error during build()
    KernelCache() : backend_descriptor() {}

    std::string
    describe() const {
        std::stringstream ss;
        ss << "CUDNN_BACKEND_KERNEL_CACHE_DESCRIPTOR : " << std::endl;
        return ss.str();
    }

    bool
    is_finalized() {
        return finalized;
    }

    // Used to check kernel cache status (particularly after initialization)
    error_t
    status() {
        if (get_status() != CUDNN_STATUS_SUCCESS) {
            return {error_code_t::CUDNN_BACKEND_API_FAILED,
                    "CUDNN_BACKEND_KERNEL_CACHE_DESCRIPTOR: Check CUDNN_VERSION >= 9.4"};
        }
        return {};
    }

    error_t
    to_json(std::string &str_json) const {
        str_json.clear();
#if (CUDNN_VERSION >= 91000)
        RETURN_CUDNN_FRONTEND_ERROR_IF(detail::get_backend_version() < 91000,
                                       error_code_t::CUDNN_BACKEND_API_FAILED,
                                       "CUDNN_ATTR_KERNEL_CACHE_JSON_REPRESENTATION is only available starting 9.10.");

        int64_t serializationSize;
        std::vector<char> serialization_buf;
        _CUDNN_CHECK_CUDNN_ERROR(detail::get_attribute(
            get_ptr(), CUDNN_ATTR_KERNEL_CACHE_JSON_REPRESENTATION, CUDNN_TYPE_CHAR, 0, &serializationSize, nullptr));
        serialization_buf.resize(static_cast<size_t>(serializationSize));

        _CUDNN_CHECK_CUDNN_ERROR(detail::get_attribute(get_ptr(),
                                                       CUDNN_ATTR_KERNEL_CACHE_JSON_REPRESENTATION,
                                                       CUDNN_TYPE_CHAR,
                                                       serializationSize,
                                                       &serializationSize,
                                                       serialization_buf.data()));
        std::string json_string(serialization_buf.begin(), serialization_buf.end());
        str_json = std::move(json_string);
        return {};
#else
        (void)str_json;
        return {error_code_t::CUDNN_BACKEND_API_FAILED,
                "CUDNN_ATTR_KERNEL_CACHE_JSON_REPRESENTATION is only available starting 9.10."};
#endif
    }

    error_t
    from_json(const std::string &json_cache) {
#if (CUDNN_VERSION >= 91000)
        RETURN_CUDNN_FRONTEND_ERROR_IF(detail::get_backend_version() < 91000,
                                       error_code_t::CUDNN_BACKEND_API_FAILED,
                                       "CUDNN_ATTR_KERNEL_CACHE_JSON_REPRESENTATION is only available starting 9.10.");

        // Check if the kernel cache is already initialized
        RETURN_CUDNN_FRONTEND_ERROR_IF(
            get_ptr() != nullptr, error_code_t::CUDNN_BACKEND_API_FAILED, "Kernel cache is already initialized.");

        // // Initialize the kernel cache descriptor
        CHECK_CUDNN_FRONTEND_ERROR(initialize(CUDNN_BACKEND_KERNEL_CACHE_DESCRIPTOR));

        std::vector<char> serialization_buf;
        serialization_buf.assign(json_cache.begin(), json_cache.end());
        _CUDNN_CHECK_CUDNN_ERROR(detail::set_attribute(get_ptr(),
                                                       CUDNN_ATTR_KERNEL_CACHE_JSON_REPRESENTATION,
                                                       CUDNN_TYPE_CHAR,
                                                       serialization_buf.size(),
                                                       serialization_buf.data()));
        return {};
#else
        (void)json_cache;
        return {error_code_t::CUDNN_BACKEND_API_FAILED,
                "CUDNN_ATTR_KERNEL_CACHE_JSON_REPRESENTATION is only available starting 9.10."};
#endif
    }

    // Responsible for initializing, setting operation graph attribute, and finalizing kernel cache
    // Check for both compile-time and runtime cuDNN version
    error_t
    build(cudnnBackendDescriptor_t op_graph) {
#if (CUDNN_VERSION >= 90400)
        RETURN_CUDNN_FRONTEND_ERROR_IF(detail::get_backend_version() < 90400,
                                       error_code_t::GRAPH_NOT_SUPPORTED,
                                       "CUDNN_BACKEND_KERNEL_CACHE_DESCRIPTOR is only available starting 9.4.");
        if (get_ptr() == nullptr) {
            CHECK_CUDNN_FRONTEND_ERROR(initialize(CUDNN_BACKEND_KERNEL_CACHE_DESCRIPTOR));
        }
#if (CUDNN_VERSION >= 90500)
        RETURN_CUDNN_FRONTEND_ERROR_IF(detail::get_backend_version() < 90500,
                                       error_code_t::GRAPH_NOT_SUPPORTED,
                                       "CUDNN_ATTR_KERNEL_CACHE_OPERATION_GRAPH is only available starting 9.5.");
        if (op_graph) {
            _CUDNN_CHECK_CUDNN_ERROR(detail::set_attribute(
                get_ptr(), CUDNN_ATTR_KERNEL_CACHE_OPERATION_GRAPH, CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &op_graph));
        }
#else
        (void)op_graph;
#endif
        CHECK_CUDNN_FRONTEND_ERROR(finalize());
        finalized = true;
        return {};
#else
        (void)op_graph;
        return {error_code_t::CUDNN_BACKEND_API_FAILED,
                "CUDNN_BACKEND_KERNEL_CACHE_DESCRIPTOR is only available starting 9.4."};
#endif
    }

   private:
    bool finalized = false;
};
}  // namespace cudnn_frontend