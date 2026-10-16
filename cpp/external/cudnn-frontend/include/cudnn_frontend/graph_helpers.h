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

#pragma once

#include <unordered_map>
#include <vector>

#include <iomanip>
#include <unordered_set>
#include <algorithm>
#include <string>
#include <numeric>

namespace cudnn_frontend {

enum class [[nodiscard]] error_code_t {
    OK,
    ATTRIBUTE_NOT_SET,
    SHAPE_DEDUCTION_FAILED,
    INVALID_TENSOR_NAME,
    INVALID_VARIANT_PACK,
    GRAPH_NOT_SUPPORTED,
    GRAPH_EXECUTION_PLAN_CREATION_FAILED,
    GRAPH_EXECUTION_FAILED,
    HEURISTIC_QUERY_FAILED,
    UNSUPPORTED_GRAPH_FORMAT,
    CUDA_API_FAILED,
    CUDNN_BACKEND_API_FAILED,
    INVALID_CUDA_DEVICE,
    HANDLE_ERROR,
    INVALID_VALUE,
    NVRTC_COMPILATION_FAILED
};

typedef struct [[nodiscard]] error_object {
    error_code_t code;
    std::string err_msg;
    error_object() : code(error_code_t::OK), err_msg("") {};
    error_object(error_code_t err, std::string msg) : code(err), err_msg(msg) {};

    error_code_t
    get_code() {
        return code;
    }

    std::string
    get_message() {
        return err_msg;
    }

    bool
    is_good() const {
        return code == error_code_t::OK;
    }

    bool
    is_bad() const {
        return !is_good();
    }

    bool
    operator==(error_code_t compare_code) {
        return code == compare_code;
    }

    bool
    operator!=(error_code_t compare_code) {
        return code != compare_code;
    }

} error_t;

#ifdef WIN32
#define CUDNN_FRONTEND_WHILE_FALSE \
    __pragma(warning(push)) __pragma(warning(disable : 4127)) while (0) __pragma(warning(pop))
#else
#define CUDNN_FRONTEND_WHILE_FALSE while (0)
#endif

#define CHECK_CUDNN_FRONTEND_ERROR(x)                                                          \
    do {                                                                                       \
        if (auto retval = x; retval.is_bad()) {                                                \
            CUDNN_FE_LOG_LABEL_ENDL("ERROR: " << #x << " at " << __FILE__ << ":" << __LINE__); \
            return retval;                                                                     \
        }                                                                                      \
    }                                                                                          \
    CUDNN_FRONTEND_WHILE_FALSE

#define RETURN_CUDNN_FRONTEND_ERROR_IF(cond, retval, message)                                                      \
    do {                                                                                                           \
        if (cond) {                                                                                                \
            if (retval == error_code_t::OK) {                                                                      \
                CUDNN_FE_LOG_LABEL("INFO: ");                                                                      \
            } else {                                                                                               \
                CUDNN_FE_LOG_LABEL("ERROR: ");                                                                     \
            }                                                                                                      \
            CUDNN_FE_LOG(message << ". " << retval << " because (" << #cond ") at " << __FILE__ << ":" << __LINE__ \
                                 << "\n");                                                                         \
            return {retval, message};                                                                              \
        }                                                                                                          \
    }                                                                                                              \
    CUDNN_FRONTEND_WHILE_FALSE

#define _CUDNN_CHECK_CUDNN_ERROR(x)                                                                         \
    do {                                                                                                    \
        if (auto cudnn_retval = x; cudnn_retval != CUDNN_STATUS_SUCCESS) {                                  \
            std::stringstream error_msg;                                                                    \
            error_msg << #x << " failed with message: " << detail::get_last_error_string_()                 \
                      << ", and code: " << detail::get_error_string(cudnn_retval);                          \
            CUDNN_FE_LOG_LABEL_ENDL("ERROR: " << error_msg.str() << " at " << __FILE__ << ":" << __LINE__); \
            return {error_code_t::CUDNN_BACKEND_API_FAILED, error_msg.str()};                               \
        }                                                                                                   \
    }                                                                                                       \
    CUDNN_FRONTEND_WHILE_FALSE

#define _CUDNN_CHECK_CUDA_ERROR(x)                                                                          \
    do {                                                                                                    \
        if (auto cuda_retval = x; cuda_retval != cudaSuccess) {                                             \
            std::stringstream error_msg;                                                                    \
            error_msg << #x << " failed with " << detail::cuda_get_error_string(cuda_retval);               \
            CUDNN_FE_LOG_LABEL_ENDL("ERROR: " << error_msg.str() << " at " << __FILE__ << ":" << __LINE__); \
            return {error_code_t::CUDA_API_FAILED, error_msg.str()};                                        \
        }                                                                                                   \
    }                                                                                                       \
    CUDNN_FRONTEND_WHILE_FALSE

NLOHMANN_JSON_SERIALIZE_ENUM(error_code_t,
                             {
                                 {error_code_t::OK, "OK"},
                                 {error_code_t::ATTRIBUTE_NOT_SET, "ATTRIBUTE_NOT_SET"},
                                 {error_code_t::SHAPE_DEDUCTION_FAILED, "SHAPE_DEDUCTION_FAILED"},
                                 {error_code_t::INVALID_TENSOR_NAME, "INVALID_TENSOR_NAME"},
                                 {error_code_t::INVALID_VARIANT_PACK, "INVALID_VARIANT_PACK"},
                                 {error_code_t::GRAPH_NOT_SUPPORTED, "GRAPH_NOT_SUPPORTED"},
                                 {error_code_t::GRAPH_EXECUTION_PLAN_CREATION_FAILED,
                                  "GRAPH_EXECUTION_PLAN_CREATION_FAILED"},
                                 {error_code_t::GRAPH_EXECUTION_FAILED, "GRAPH_EXECUTION_FAILED"},
                                 {error_code_t::HEURISTIC_QUERY_FAILED, "HEURISTIC_QUERY_FAILED"},
                                 {error_code_t::CUDNN_BACKEND_API_FAILED, "CUDNN_BACKEND_API_FAILED"},
                                 {error_code_t::CUDA_API_FAILED, "CUDA_API_FAILED"},
                                 {error_code_t::INVALID_CUDA_DEVICE, "INVALID_CUDA_DEVICE"},
                                 {error_code_t::UNSUPPORTED_GRAPH_FORMAT, "UNSUPPORTED_GRAPH_FORMAT"},
                                 {error_code_t::HANDLE_ERROR, "HANDLE_ERROR"},
                                 {error_code_t::INVALID_VALUE, "INVALID_VALUE"},
                                 {error_code_t::NVRTC_COMPILATION_FAILED, "NVRTC_COMPILATION_FAILED"},
                             })

static inline std::ostream&
operator<<(std::ostream& os, const error_code_t& mode) {
#ifndef CUDNN_FRONTEND_SKIP_JSON_LIB
    os << json{mode};
#else
    os << int(mode);
#endif
    return os;
}

static inline std::ostream&
operator<<(std::ostream& os, cudnn_frontend::error_object& err) {
    os << err.get_code() << err.get_message();
    return os;
}

static bool
allowAllConfig(cudnnBackendDescriptor_t engine_config) {
    (void)engine_config;
    return false;
}

namespace detail {

inline bool
is_activation_backward_mode(PointwiseMode_t const mode) {
    return ((mode == PointwiseMode_t::RELU_BWD) || (mode == PointwiseMode_t::TANH_BWD) ||
            (mode == PointwiseMode_t::SIGMOID_BWD) || (mode == PointwiseMode_t::ELU_BWD) ||
            (mode == PointwiseMode_t::GELU_BWD) || (mode == PointwiseMode_t::GELU_APPROX_TANH_BWD) ||
            (mode == PointwiseMode_t::SOFTPLUS_BWD) || (mode == PointwiseMode_t::SWISH_BWD));
}

// Creates dense, non-overlapping strides from given dim and stride_order.
// For example, if a is a 4D tensor with dimensions labeled NCHW, then strided(a, (3, 0, 2, 1)) produces
// strides where the C dimension has a corresponding stride of one.
inline std::vector<int64_t>
generate_stride(std::vector<int64_t> const& dim, std::vector<int64_t> const& stride_order) {
    size_t num_dims = dim.size();
    std::vector<int64_t> stride(num_dims);

    // Sort the dimensions according to strides from least to greatest.
    // Example, dim = (2, 3, 4, 5) stride_order = (3, 1, 2, 0)
    // sorted_stride_order = ((0, (3, 5)), (1, (1, 3)), (2, (2, 4)), (3, (0, 2)))
    std::vector<std::pair<int64_t, std::pair<size_t, size_t>>> sorted_stride_order;
    for (size_t i = 0; i < num_dims; ++i) {
        sorted_stride_order.push_back({stride_order[i], {i, dim[i]}});
    }
    std::sort(sorted_stride_order.begin(), sorted_stride_order.end());

    // As dims have now been ordered starting from fastest changing,
    // just fill in strides by iterating linearly over them.
    int64_t product = 1;
    for (size_t i = 0; i < num_dims; ++i) {
        stride[sorted_stride_order[i].second.first] = product;
        product *= sorted_stride_order[i].second.second;
    }

    return stride;
}

// Generate NHWC stride_order
inline std::vector<int64_t>
generate_NHWC_stride_order(int64_t const num_dims) {
    std::vector<int64_t> stride_order(num_dims);

    int64_t order   = 0;
    stride_order[1] = order++;
    for (size_t i = num_dims - 1; i > 1; --i) {
        stride_order[i] = order++;
    }
    stride_order[0] = order;

    return stride_order;
}

// Generate row major stride_order for matrices
// dim = (*, M, N) where * is batch dimsensions
// strides should be (..., N, 1)
inline std::vector<int64_t>
generate_row_major_stride_order(int64_t const num_dims) {
    std::vector<int64_t> stride_order(num_dims);

    int64_t order = num_dims - 1;
    std::generate(stride_order.begin(), stride_order.end(), [&order] { return order--; });

    return stride_order;
}

// Generate column major stride_order for matrices
// dim = (*, M, N)
// strides should be (*, 1, M)
inline std::vector<int64_t>
generate_column_major_stride_order(int64_t const num_dims) {
    std::vector<int64_t> stride_order = generate_row_major_stride_order(num_dims);
    if (num_dims > 2) {
        std::swap(stride_order[num_dims - 1], stride_order[num_dims - 2]);
    }
    return stride_order;
}

/**
 * @brief Computes the common shape with the fewest dimensions that all input shapes can be broadcast to.
 *
 * This function takes a vector of shapes and calculates a common shape that all input shapes
 * can be broadcast to. It follows broadcasting rules similar to those used in NumPy.
 *
 * @param _shapes A vector of vectors, where each inner vector represents a shape.
 *                Each shape is a sequence of dimension sizes.
 * @param[out] common_shape The computed broadcast shape is stored in this vector.
 *                          It will be cleared and resized as necessary.
 *
 * @return error_t An error code indicating the result of the operation
 *
 * @note
 * - Shapes are processed from right to left (last dimension to first).
 * - A dimension of size 1 can be broadcast to any size.
 * - Non-1 dimensions must match exactly for broadcasting.
 * - The resulting shape will have the maximum number of dimensions among all input shapes.
 *
 * @example
 *   std::vector<std::vector<int64_t>> shapes = {{3, 1, 4}, {1, 2, 4}, {2, 4}};
 *   std::vector<int64_t> result;
 *   error_t err = compute_broadcast_shape(shapes, result);
 *   // If err == error_code_t::OK, result will be {3, 2, 4}
 */
inline error_t
compute_broadcast_shape(const std::vector<std::vector<int64_t>>& _shapes, std::vector<int64_t>& common_shape) {
    // Filter out empty shapes
    std::vector<std::vector<int64_t>> shapes;
    std::copy_if(_shapes.begin(), _shapes.end(), std::back_inserter(shapes), [](const std::vector<int64_t>& shape) {
        return !shape.empty();
    });

    // Short-circuits if there are no input shapes
    RETURN_CUDNN_FRONTEND_ERROR_IF(
        shapes.empty(), error_code_t::SHAPE_DEDUCTION_FAILED, "All input shapes provided are empty.");

    // Find the maximum dimension
    int64_t max_dim = std::max_element(shapes.begin(),
                                       shapes.end(),
                                       [](const std::vector<int64_t>& a, const std::vector<int64_t>& b) {
                                           return a.size() < b.size();
                                       })
                          ->size();

    // Initialize common_shape with 1s
    common_shape.assign(max_dim, 1);

    for (const auto& shape : shapes) {
        for (int idx = -1; idx >= -static_cast<int>(shape.size()); --idx) {
            int64_t common_idx = common_shape.size() + idx;
            int64_t shape_idx  = shape.size() + idx;

            if (common_shape[common_idx] == 1) {
                common_shape[common_idx] = shape[shape_idx];
            }

            RETURN_CUDNN_FRONTEND_ERROR_IF((shape[shape_idx] != 1) && (common_shape[common_idx] != shape[shape_idx]),
                                           error_code_t::SHAPE_DEDUCTION_FAILED,
                                           "dimensions mismatch as broadcasting 2 non-one dimension sizes.");
        }
    }

    return {error_code_t::OK, ""};
}
/**
 * @brief Generates a stride order preserving the format of the input tensor.
 *
 * This function derives the exact stride order from the input tensor's strides.
 * It returns the indices of the strides in ascending order of stride values.
 *
 * @param input_stride The stride of the input tensor
 * @param output_dim_size The number of dimensions in the output tensor
 * @return std::vector<int64_t> The generated stride order
 */
inline error_t
generate_stride_order_preserving_format(const std::vector<int64_t>& input_stride,
                                        size_t output_dim_size,
                                        std::vector<int64_t>& stride_order) {
    std::vector<int64_t> indices(input_stride.size());
    std::iota(indices.begin(), indices.end(), 0);

    // Sort indices based on stride values in descending order
    std::sort(indices.begin(), indices.end(), [&input_stride](int64_t i, int64_t j) {
        return input_stride[i] < input_stride[j];
    });

    // Enable this after further debug
    // std::set<int64_t> stride_set(input_stride.begin(), input_stride.end());
    // RETURN_CUDNN_FRONTEND_ERROR_IF((stride_set.size() != input_stride.size()),
    //                                error_code_t::SHAPE_DEDUCTION_FAILED,
    //                                "Have multiple stride with same value. Cant determine stride order");

    // Create the stride order
    stride_order.resize(input_stride.size());
    for (size_t i = 0; i < indices.size(); ++i) {
        stride_order[indices[i]] = i;
    }

    // If output_dim_size is larger, pad with remaining dimensions
    if (output_dim_size > input_stride.size()) {
        size_t start = stride_order.size();
        stride_order.resize(output_dim_size);
        std::iota(stride_order.begin() + start, stride_order.end(), start);
    }

    return {error_code_t::OK, ""};
}

/**
 * @brief Infers the output dimensions for a matrix multiplication operation.
 *
 * This function calculates the output dimensions of a matrix multiplication
 * based on the input dimensions of tensors A and B. It uses compute_broadcast_shape
 * for batch dimensions and ensures the last two dimensions are correct for matrix multiplication.
 *
 * @param a_dim Dimensions of the first input tensor (A).
 * @param b_dim Dimensions of the second input tensor (B).
 * @param output_dim Reference to the vector where the output dimensions will be stored.
 * @return error_t An error code indicating the result of the operation.
 */
inline error_t
generate_matmul_output_dim(const std::vector<int64_t>& a_dim,
                           const std::vector<int64_t>& b_dim,
                           std::vector<int64_t>& output_dim) {
    // Ensure a_dim and b_dim have at least 2 dimensions
    if (a_dim.size() < 2 || b_dim.size() < 2) {
        return {error_code_t::SHAPE_DEDUCTION_FAILED, "Input tensors must have at least 2 dimensions for matmul."};
    }

    // Check if inner dimensions are compatible
    if (a_dim[a_dim.size() - 1] != b_dim[b_dim.size() - 2]) {
        return {error_code_t::SHAPE_DEDUCTION_FAILED,
                "Inner dimensions of input tensors are not compatible for matmul."};
    }

    // Prepare shapes for broadcasting
    std::vector<int64_t> a_batch_dim(a_dim.begin(), a_dim.end() - 2);
    std::vector<int64_t> b_batch_dim(b_dim.begin(), b_dim.end() - 2);

    // Compute broadcast shape for batch dimensions
    std::vector<int64_t> broadcasted_batch;
    CHECK_CUDNN_FRONTEND_ERROR(detail::compute_broadcast_shape({a_batch_dim, b_batch_dim}, broadcasted_batch));

    // Construct final output shape
    output_dim = broadcasted_batch;
    output_dim.push_back(a_dim[a_dim.size() - 2]);  // M from A
    output_dim.push_back(b_dim[b_dim.size() - 1]);  // N from B

    return {error_code_t::OK, ""};
}

inline std::string
to_hex(const void* data, size_t num_elements, size_t elem_size) {
    const auto* bytes = static_cast<const unsigned char*>(data);
    std::stringstream ss;
    ss << "[";
    for (size_t i = 0; i < num_elements; ++i) {
        if (i > 0) ss << ", ";
        ss << "0x" << std::hex << std::uppercase;
        switch (elem_size) {
            case 1:
                ss << static_cast<unsigned>(bytes[i]);
                break;
            case 2:
                ss << *reinterpret_cast<const uint16_t*>(&bytes[i * 2]);
                break;
            case 4:
                ss << *reinterpret_cast<const uint32_t*>(&bytes[i * 4]);
                break;
            case 8:
                ss << *reinterpret_cast<const uint64_t*>(&bytes[i * 8]);
                break;
            default:
                ss << "?";
        }
    }
    ss << "]";
    return ss.str();
}

inline std::string
to_decimal(const void* data, size_t num_elements, size_t elem_size) {
    const auto* bytes = static_cast<const unsigned char*>(data);
    std::stringstream ss;
    ss << "[";
    for (size_t i = 0; i < num_elements; ++i) {
        if (i > 0) ss << ", ";
        switch (elem_size) {
            case 1:
                ss << static_cast<int>(bytes[i]);
                break;
            case 2:
                ss << *reinterpret_cast<const int16_t*>(&bytes[i * 2]);
                break;
            case 4:
                ss << *reinterpret_cast<const int32_t*>(&bytes[i * 4]);
                break;
            case 8:
                ss << *reinterpret_cast<const int64_t*>(&bytes[i * 8]);
                break;
            default:
                ss << "?";
        }
    }
    ss << "]";
    return ss.str();
}

inline std::string
to_base64(const void* data, size_t total_bytes) {
    static const char table[] = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    const auto* bytes         = static_cast<const unsigned char*>(data);
    std::string result;
    result.reserve(((total_bytes + 2) / 3) * 4);
    for (size_t i = 0; i < total_bytes; i += 3) {
        uint32_t n = static_cast<uint32_t>(bytes[i]) << 16;
        if (i + 1 < total_bytes) n |= static_cast<uint32_t>(bytes[i + 1]) << 8;
        if (i + 2 < total_bytes) n |= static_cast<uint32_t>(bytes[i + 2]);
        result.push_back(table[(n >> 18) & 0x3F]);
        result.push_back(table[(n >> 12) & 0x3F]);
        result.push_back((i + 1 < total_bytes) ? table[(n >> 6) & 0x3F] : '=');
        result.push_back((i + 2 < total_bytes) ? table[n & 0x3F] : '=');
    }
    return result;
}

inline error_t
log_dump_tensor_content(int64_t uid,
                        std::string const& name,
                        void* ptr,
                        size_t num_elements,
                        size_t elem_size,
                        char fmt,
                        cudaStream_t stream) {
    if (!isLoggingEnabled()) return {error_code_t::OK, ""};

    size_t total_bytes = num_elements * elem_size;

    cudaPointerAttributes attr;
    _CUDNN_CHECK_CUDA_ERROR(cuda_pointer_get_attributes(&attr, ptr));

    std::vector<unsigned char> host_buf(total_bytes);
    if (attr.type == cudaMemoryTypeDevice || attr.type == cudaMemoryTypeManaged) {
        _CUDNN_CHECK_CUDA_ERROR(cuda_mem_cpy_async(host_buf.data(), ptr, total_bytes, cudaMemcpyDeviceToHost, stream));
        _CUDNN_CHECK_CUDA_ERROR(cuda_stream_synchronize(stream));
    } else {
        std::memcpy(host_buf.data(), ptr, total_bytes);
    }

    std::string data_str;
    switch (fmt) {
        case 'x':
            data_str = to_hex(host_buf.data(), num_elements, elem_size);
            break;
        case 'd':
            data_str = to_decimal(host_buf.data(), num_elements, elem_size);
            break;
        case 'b':
            data_str = to_base64(host_buf.data(), total_bytes);
            break;
        default:
            data_str = to_hex(host_buf.data(), num_elements, elem_size);
    }
    CUDNN_FE_LOG_LABEL_ENDL("Tensor Dump Uid: " << uid << " Name: " << name << " Data: " << data_str);
    return {error_code_t::OK, ""};
}

inline error_t
log_variant_pack_memory_type(int64_t uid, void* ptr) {
    if (!isLoggingEnabled()) return {error_code_t::OK, ""};

    cudaPointerAttributes attributes;
    _CUDNN_CHECK_CUDA_ERROR(cuda_pointer_get_attributes(&attributes, ptr));

    auto memory_type_to_string = [](cudaMemoryType type) {
        switch (type) {
            case cudaMemoryTypeHost:
                return std::string("Host");
            case cudaMemoryTypeDevice:
                return std::string("Device");
            case cudaMemoryTypeManaged:
                return std::string("Managed");
            case cudaMemoryTypeUnregistered:
                return std::string("Unregistered");
            default:
                return "UNKNOWN cudaMemoryType (" + std::to_string(type) + ")";
        }
    };

    auto ptr_to_string = [](void* p) {
        std::stringstream ss;
        ss << "0x" << std::hex << std::setw(sizeof(void*) * 2) << std::setfill('0') << reinterpret_cast<uintptr_t>(p);
        return ss.str();
    };

    // clang-format off
    CUDNN_FE_LOG_LABEL_ENDL("Variant Pack" << std::setw(0) << " Uid: " << std::setw(20) << uid
                                           << std::setw(0) << " MemoryType: " << std::setw(12) << memory_type_to_string(attributes.type)
                                           << std::setw(0) << " Device: " << std::setw(4) << attributes.device
                                           << std::setw(0) << " UnifiedPtr: " << std::setw(20) << ptr_to_string(ptr)
                                           << std::setw(0) << " DevicePtr: " << std::setw(20) << ptr_to_string(attributes.devicePointer)
                                           << std::setw(0) << " HostPtr: " << std::setw(20) << ptr_to_string(attributes.hostPointer));
    // clang-format on
    return {error_code_t::OK, ""};
}

}  // namespace detail

class cudnnGraphNotSupportedException : public std::runtime_error {
   public:
    cudnnGraphNotSupportedException(const char* message) throw() : std::runtime_error(message) {}

    virtual const char*
    what() const throw() {
        return std::runtime_error::what();
    }
};

}  // namespace cudnn_frontend