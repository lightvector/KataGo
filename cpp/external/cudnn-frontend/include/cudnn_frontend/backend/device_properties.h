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

#include <algorithm>
#include <array>
#include <functional>
#include <memory>
#include <sstream>
#include <utility>

#include "../graph_helpers.h"
#include "backend_descriptor.h"

namespace cudnn_frontend {
///
/// DeviceProperties Class
/// Wraps the device_properties backend descriptor
/// Wraps backend utility functions for user's convenience
/// Backend accessor functions: size()
/// Contains internal utilities for device properties finalization and operation graph attributes
///
class DeviceProperties : public detail::backend_descriptor {
   public:
    // Uses the default backend constructor so that we can check for initialization error during build()
    DeviceProperties() = default;

    std::string
    describe() const {
        std::stringstream ss;
        ss << "CUDNN_BACKEND_DEVICEPROP_DESCRIPTOR : " << std::endl;
        return ss.str();
    }

    inline DeviceProperties&
    set_device_id(int32_t device_id) {
        this->device_id = device_id;
        return *this;
    }

    inline DeviceProperties&
    set_handle(cudnnHandle_t handle) {
        this->handle = handle;
        return *this;
    }

    // Used to check device properties status (particularly after initialization)
    error_t
    status() const {
        if (get_status() != CUDNN_STATUS_SUCCESS) {
            return {error_code_t::CUDNN_BACKEND_API_FAILED,
                    "CUDNN_BACKEND_DEVICEPROP_DESCRIPTOR: Check CUDNN_VERSION >= 9.8"};
        }
        return {};
    }

    error_t
    serialize(std::vector<uint8_t>& serialization_buf) const {
#if (CUDNN_VERSION >= 90800)
        RETURN_CUDNN_FRONTEND_ERROR_IF(detail::get_backend_version() < 90800,
                                       error_code_t::CUDNN_BACKEND_API_FAILED,
                                       "CUDNN_ATTR_DEVICEPROP_JSON_REPRESENTATION is only available starting 9.8.");

        int64_t serializationSize;
        _CUDNN_CHECK_CUDNN_ERROR(detail::get_attribute(
            get_ptr(), CUDNN_ATTR_DEVICEPROP_JSON_REPRESENTATION, CUDNN_TYPE_CHAR, 0, &serializationSize, nullptr));
        serialization_buf.resize(static_cast<size_t>(serializationSize));

        _CUDNN_CHECK_CUDNN_ERROR(detail::get_attribute(get_ptr(),
                                                       CUDNN_ATTR_DEVICEPROP_JSON_REPRESENTATION,
                                                       CUDNN_TYPE_CHAR,
                                                       serializationSize,
                                                       &serializationSize,
                                                       serialization_buf.data()));
        return {};
#else
        (void)serialization_buf;
        return {error_code_t::CUDNN_BACKEND_API_FAILED,
                "CUDNN_ATTR_DEVICEPROP_JSON_REPRESENTATION is only available starting 9.8."};
#endif
    }

    error_t
    deserialize(const std::vector<uint8_t>& serialized_buf) {
#if (CUDNN_VERSION >= 90800)
        RETURN_CUDNN_FRONTEND_ERROR_IF(detail::get_backend_version() < 90800,
                                       error_code_t::CUDNN_BACKEND_API_FAILED,
                                       "CUDNN_ATTR_DEVICEPROP_JSON_REPRESENTATION is only available starting 9.8.");

        // Check if the device properties is already initialized
        RETURN_CUDNN_FRONTEND_ERROR_IF(
            get_ptr() != nullptr, error_code_t::CUDNN_BACKEND_API_FAILED, "Device properties is already initialized.");

        // Initialize the device properties descriptor
        CHECK_CUDNN_FRONTEND_ERROR(initialize(CUDNN_BACKEND_DEVICEPROP_DESCRIPTOR));

        _CUDNN_CHECK_CUDNN_ERROR(detail::set_attribute(get_ptr(),
                                                       CUDNN_ATTR_DEVICEPROP_JSON_REPRESENTATION,
                                                       CUDNN_TYPE_CHAR,
                                                       serialized_buf.size(),
                                                       serialized_buf.data()));

        CHECK_CUDNN_FRONTEND_ERROR(finalize());
        return {};
#else
        (void)serialized_buf;
        return {error_code_t::CUDNN_BACKEND_API_FAILED,
                "CUDNN_ATTR_DEVICEPROP_JSON_REPRESENTATION is only available starting 9.8."};
#endif
    }

    // Check for both compile-time and runtime cuDNN version
    error_t
    build() {
#if (CUDNN_VERSION >= 90800)
        RETURN_CUDNN_FRONTEND_ERROR_IF(detail::get_backend_version() < 90800,
                                       error_code_t::GRAPH_NOT_SUPPORTED,
                                       "CUDNN_BACKEND_DEVICEPROP_DESCRIPTOR is only available starting 9.8.");
        if (get_ptr() == nullptr) {
            CHECK_CUDNN_FRONTEND_ERROR(initialize(CUDNN_BACKEND_DEVICEPROP_DESCRIPTOR));
        }

        if (handle != nullptr) {
            _CUDNN_CHECK_CUDNN_ERROR(
                detail::set_attribute(get_ptr(), CUDNN_ATTR_DEVICEPROP_HANDLE, CUDNN_TYPE_HANDLE, 1, &handle));
        }

        if (device_id >= 0) {
            _CUDNN_CHECK_CUDNN_ERROR(
                detail::set_attribute(get_ptr(), CUDNN_ATTR_DEVICEPROP_DEVICE_ID, CUDNN_TYPE_INT32, 1, &device_id));
        }

        CHECK_CUDNN_FRONTEND_ERROR(finalize());
        return {};
#else
        return {error_code_t::CUDNN_BACKEND_API_FAILED,
                "CUDNN_BACKEND_DEVICEPROP_DESCRIPTOR is only available starting 9.8."};
#endif
    }

   private:
    cudnnHandle_t handle = nullptr;
    int32_t device_id    = 0;
};
}  // namespace cudnn_frontend