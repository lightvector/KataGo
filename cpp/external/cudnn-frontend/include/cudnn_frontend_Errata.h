/*
 * Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
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

#include "../include/cudnn_frontend_Logging.h"

#include <cstdlib>
#include <fstream>
#pragma once

namespace cudnn_frontend {

// Loads the json handle from the json file
// json file is defined by environment variable
// CUDNN_ERRATA_JSON_FILE. If the environment variable
// is not set the value set in the API is considered.
#ifndef CUDNN_FRONTEND_SKIP_JSON_LIB
[[maybe_unused]] static bool
load_from_config(json &json_handle, const std::string &errata_json) {
    const char *err_json = get_environment("CUDNN_ERRATA_JSON_FILE");
    if (err_json == NULL && errata_json == "") {
        return false;
    }
    if (err_json == NULL) {
        err_json = errata_json.c_str();
    }
    std::ifstream ifs(err_json, std::ifstream::in);
    if (!ifs.is_open() || !ifs.good()) {
        return false;
    }
    ifs >> json_handle;
    return true;
}
#endif

/**
 * @brief Checks the shape of an operation to compare against errata filter height and width for kernel blocking
 *
 * @param op The operation's tensors to check
 * @param shape_format The shape format of the tensor (NCHW vs NHWC)
 * @param tensor_attr The cudnnBackendAttributeName_t of the tensor's shape we want to check
 * @param blocked_height The height we want to filter out
 * @param blocked_width The width we want to filter out
 * @param blocked_channels The channels we want to filter out. Defaults to -1 (not filter out channels)
 * @return true The passed in operation shape matches the blocked shape
 * @return false The passed in operation shape does not match the blocked shape
 */
static bool
check_shape(cudnnBackendDescriptor_t &op,
            const std::string &shape_format,
            cudnnBackendAttributeName_t tensor_attr,
            const std::vector<int64_t> &blocked_shape) {
    // Get backend descriptor to individual tensor to be able to get shape
    ManagedOpaqueDescriptor tensor   = make_shared_backend_pointer(CUDNN_BACKEND_TENSOR_DESCRIPTOR);
    cudnnBackendDescriptor_t tensor_ = tensor->get_backend_descriptor();
    int64_t count                    = 0;
    cudnnStatus_t status = detail::get_attribute(op, tensor_attr, CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &count, &tensor_);
    if (status != CUDNN_STATUS_SUCCESS) {
#ifndef NV_CUDNN_DISABLE_EXCEPTION
        throw cudnnException(std::string("Error getting attribute. cudnn_status: " + to_string(status)).c_str(),
                             status);
#endif
    }

    // Get tensor dims
    std::array<int64_t, 5> tensor_dims;
    status =
        detail::get_attribute(tensor_, CUDNN_ATTR_TENSOR_DIMENSIONS, CUDNN_TYPE_INT64, 5, &count, tensor_dims.data());
    if (status != CUDNN_STATUS_SUCCESS) {
#ifndef NV_CUDNN_DISABLE_EXCEPTION
        throw cudnnException(std::string("Error getting attribute. cudnn_status: " + to_string(status)).c_str(),
                             status);
#endif
    }
    // tensor_dims is 1 indexed
    int64_t first_dim = tensor_dims[1];  // batch size for input/output tensor, output channels for filter tensor
    int64_t blocked_first_dim = blocked_shape[0];

    // Defaults to true becuase -1 means we don't filter that out (Wildcard). If something later blocks, then the
    // comparison will be correct
    bool blocked = (blocked_first_dim != -1) ? (first_dim == blocked_first_dim) : true;

    // Check for shape format to extract the right dimension. Filter shape will always be "NCHW" for convenience.
    int64_t channels         = (shape_format == "NCHW") ? tensor_dims[2] : tensor_dims[4];  // channels
    int64_t blocked_channels = (shape_format == "NCHW") ? blocked_shape[1] : blocked_shape[3];
    blocked                  = (blocked_channels != -1) ? (blocked && channels == blocked_channels) : true;

    int64_t height         = (shape_format == "NCHW") ? tensor_dims[3] : tensor_dims[2];
    int64_t blocked_height = (shape_format == "NCHW") ? blocked_shape[2] : blocked_shape[1];
    blocked                = (blocked_height != -1) ? (blocked && height == blocked_height) : true;

    int64_t width         = (shape_format == "NCHW") ? tensor_dims[4] : tensor_dims[3];
    int64_t blocked_width = (shape_format == "NCHW") ? blocked_shape[3] : blocked_shape[2];
    blocked               = (blocked_width != -1) ? (blocked && width == blocked_width) : true;

    return blocked;
}

#ifndef CUDNN_FRONTEND_SKIP_JSON_LIB
template <typename T>
static bool
check_rule(const json &json_handle, const std::string &executionPlanTag, cudnnHandle_t handle, T fn) {
    std::string operation = json_handle["operation"];
    int64_t engine        = json_handle["engine"];
    uint64_t cudnn_start  = 0;
    uint64_t cudnn_end    = std::numeric_limits<uint64_t>::max();
    if (json_handle.contains("cudnn_version_start")) {
        cudnn_start = json_handle["cudnn_version_start"];
    }
    if (json_handle.contains("cudnn_version_end")) {
        cudnn_end = json_handle["cudnn_version_end"];
    }
    std::string tag_prefix = operation + "_eng" + std::to_string(engine) + "_";
    std::string mod_tag    = executionPlanTag + "_";
    bool blocked           = tag_prefix.size() <= mod_tag.size() &&
                   std::equal(tag_prefix.begin(), tag_prefix.end(), mod_tag.begin()) && CUDNN_VERSION >= cudnn_start &&
                   CUDNN_VERSION < cudnn_end;

    if (blocked && json_handle.contains("knob")) {  // Short circuit if operation and engine do not match
        for (auto &kv : json_handle["knob"]) {
            blocked = blocked && (executionPlanTag.find(kv) != std::string::npos);
        }
    }
    blocked = blocked && fn();
    return blocked;

    CUDNN_FRONTEND_UNUSED(handle);
}

// Overload for check_rule to take in an operation graph for shape filtering
template <typename T>
static bool
check_rule(const json &json_handle,
           const std::string &executionPlanTag,
           cudnnHandle_t handle,
           T fn,
           const OperationGraph &opGraph) {
    std::string operation = json_handle["operation"];
    int64_t engine        = json_handle["engine"];
    uint64_t cudnn_start  = 0;
    uint64_t cudnn_end    = std::numeric_limits<uint64_t>::max();
    if (json_handle.contains("cudnn_version_start")) {
        cudnn_start = json_handle["cudnn_version_start"];
    }
    if (json_handle.contains("cudnn_version_end")) {
        cudnn_end = json_handle["cudnn_version_end"];
    }
    std::string tag_prefix = operation + "_eng" + std::to_string(engine) + "_";
    std::string mod_tag    = executionPlanTag + "_";
    bool blocked           = tag_prefix.size() <= mod_tag.size() &&
                   std::equal(tag_prefix.begin(), tag_prefix.end(), mod_tag.begin()) && CUDNN_VERSION >= cudnn_start &&
                   CUDNN_VERSION < cudnn_end;

    if (blocked && json_handle.contains("knob")) {  // Short circuit if operation and engine do not match
        for (auto &kv : json_handle["knob"]) {
            blocked = blocked && (executionPlanTag.find(kv) != std::string::npos);
        }
    }

    if (blocked &&
        json_handle.contains("input_shape")) {  // Check if user wants to block kernel for specific input shape
        if (!json_handle.contains("shape_format")) {
            std::string message =
                "ERROR: Please set a shape format (e.g. shape_format: \"NCWH\") for errata filters using input/kernel "
                "shape";
#ifndef NV_CUDNN_DISABLE_EXCEPTION
            throw cudnnException(message.c_str(), CUDNN_STATUS_BAD_PARAM);
#else
            CUDNN_FE_LOG(message << std::endl);
            return blocked;
#endif
        }

        std::array<ManagedOpaqueDescriptor, MAX_OPGRAPH_OPS> ops = opGraph.getOps();
        std::array<cudnnBackendDescriptor_t, MAX_OPGRAPH_OPS> ops_;
        for (unsigned int i = 0; i < opGraph.getOpCount(); i++) {
            ops_[i] = ops[i]->get_backend_descriptor();
        }

        std::string shape_format           = json_handle["shape_format"];
        std::vector<int64_t> blocked_shape = json_handle["input_shape"];

        // Forward conv operation
        if (operation == "ConvFwd") {
            blocked = blocked &&
                      check_shape(ops_[0], shape_format, CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_X, blocked_shape);

            // Operation is conv wgrad
        } else if (operation == "ConvBwdFilter") {
            blocked = blocked &&
                      check_shape(ops_[0], shape_format, CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_X, blocked_shape);

            // Operation is conv dgrad
        } else if (operation == "ConvBwdData") {
            blocked = blocked &&
                      check_shape(ops_[0], shape_format, CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_DX, blocked_shape);
        }
    }

    if (blocked &&
        json_handle.contains("filter_shape")) {  // Check if user wants to block kernel for specific filter shape
        std::array<ManagedOpaqueDescriptor, MAX_OPGRAPH_OPS> ops = opGraph.getOps();
        std::array<cudnnBackendDescriptor_t, MAX_OPGRAPH_OPS> ops_;
        for (unsigned int i = 0; i < opGraph.getOpCount(); i++) {
            ops_[i] = ops[i]->get_backend_descriptor();
        }

        std::vector<int64_t> blocked_shape = json_handle["filter_shape"];

        // Forward conv operation
        if (operation == "ConvFwd") {
            // Filter format is always [output channels, input channels, height, width] so we hardcode "NCHW" to match
            // and not repeat code
            blocked =
                blocked && check_shape(ops_[0], "NCHW", CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_W, blocked_shape);

            // Operation is conv wgrad
        } else if (operation == "ConvBwdFilter") {
            blocked =
                blocked && check_shape(ops_[0], "NCHW", CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_DW, blocked_shape);

            // Operation is conv dgrad
        } else if (operation == "ConvBwdData") {
            blocked =
                blocked && check_shape(ops_[0], "NCHW", CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_W, blocked_shape);
        }
    }
    blocked = blocked && fn();
    return blocked;

    CUDNN_FRONTEND_UNUSED(handle);
}

// Takes in an initialzed json handle and checks if it satisfies the
// condition for running it. Returns true if the given executionPlanTag
// is faulty.

template <typename T>
static bool
check_errata(const json &json_handle, const std::string &executionPlanTag, cudnnHandle_t handle, T fn) {
    CUDNN_FE_LOG_LABEL("Verifying " << executionPlanTag);
    for (auto const &rule : json_handle["rules"]) {
        if (check_rule<T>(rule, executionPlanTag, handle, fn)) {
            CUDNN_FE_LOG(". Blocking." << std::endl);
            return true;
        }
    }

    CUDNN_FE_LOG(". Passed." << std::endl);
    return false;
}

// Overload. Takes in an initialzed json handle, an execution plan tag, and a operation graph and checks if it satisfies
// the condition for running it. Returns true if the given executionPlanTag + operation graph is faulty
template <typename T>
static bool
check_errata(const json &json_handle,
             const std::string &executionPlanTag,
             cudnnHandle_t handle,
             const OperationGraph &opGraph,
             T fn) {
    CUDNN_FE_LOG_LABEL("Verifying " << executionPlanTag);
    for (auto const &rule : json_handle["rules"]) {
        if (check_rule<T>(rule, executionPlanTag, handle, fn, opGraph)) {
            CUDNN_FE_LOG(". Blocking." << std::endl);
            return true;
        }
    }

    CUDNN_FE_LOG(". Passed." << std::endl);
    return false;
}
#endif

}  // namespace cudnn_frontend
