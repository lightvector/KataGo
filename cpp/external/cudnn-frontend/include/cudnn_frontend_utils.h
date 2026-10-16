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
#include <exception>
#include <optional>
#include <string>
#include <variant>
#include <vector>
#include <iomanip>
#include <sstream>
#include <cmath>
#include <cstdint>

#include <cuda_fp16.h>
#include <cuda_bf16.h>

#ifndef CUDNN_FRONTEND_SKIP_JSON_LIB

#ifndef CUDNN_FRONTEND_SKIP_JSON_LIB
#include "cudnn_frontend/thirdparty/nlohmann/json.hpp"
#endif

using json = nlohmann::json;

template <>
struct nlohmann::adl_serializer<float> {
    static void
    to_json(nlohmann::json& j, const float& f) {
        // Convert float to hexadecimal string
        unsigned int intValue;
        std::memcpy(&intValue, &f, sizeof(float));

        std::stringstream stream;
        stream << std::hex << std::uppercase << std::setw(8) << std::setfill('0') << intValue;
        j = stream.str();
    }

    static void
    from_json(const nlohmann::json& j, float& f) {
        // Read hexadecimal string and convert back to float
        std::string hexValueStr = j.get<std::string>();
        unsigned int hexValue;
        std::stringstream stream(hexValueStr);
        stream >> std::hex >> hexValue;

        std::memcpy(&f, &hexValue, sizeof(float));
    }
};

template <>
struct nlohmann::adl_serializer<half> {
    static void
    to_json(json& j, const half& opt) {
        // No precision loss when converting to float
        j = __half2float(opt);
    }

    static void
    from_json(const json& j, half& opt) {
        opt = __float2half(j.get<float>());
    }
};

template <>
struct nlohmann::adl_serializer<nv_bfloat16> {
    static void
    to_json(json& j, const nv_bfloat16& opt) {
        // No precision loss when converting to float
        j = __bfloat162float(opt);
    }

    static void
    from_json(const json& j, nv_bfloat16& opt) {
        opt = __float2bfloat16(j.get<float>());
    }
};

template <>
struct nlohmann::adl_serializer<std::variant<int64_t, int32_t, half, float, double, nv_bfloat16>> {
    static void
    to_json(nlohmann::json& j, const std::variant<int64_t, int32_t, half, float, double, nv_bfloat16>& data) {
        std::visit([&](const auto& v) { j = {{"index", data.index()}, {"value", v}}; }, data);
    }

    static void
    from_json(const nlohmann::json& j, std::variant<int64_t, int32_t, half, float, double, nv_bfloat16>& data) {
        if (!j.is_object() || !j.contains("index") || !j.contains("value")) {
            return;
        }

        size_t type_index = j.at("index").get<size_t>();
        if (type_index == 0) {
            data = j.at("value").get<int64_t>();
        } else if (type_index == 1) {
            data = j.at("value").get<int32_t>();
        } else if (type_index == 2) {
            data = j.at("value").get<half>();
        } else if (type_index == 3) {
            data = j.at("value").get<float>();
        } else if (type_index == 4) {
            data = j.at("value").get<double>();
        } else if (type_index == 5) {
            data = j.at("value").get<nv_bfloat16>();
        } else {
            return;
        }
    }
};

// Specialization of nlohmann::adl_serializer for std::optional<T>
template <typename T>
struct nlohmann::adl_serializer<std::optional<T>> {
    static void
    to_json(json& j, const std::optional<T>& opt) {
        if (opt.has_value())
            j = *opt;
        else
            j = nullptr;
    }

    static void
    from_json(const json& j, std::optional<T>& opt) {
        if (!j.is_null())
            opt = j.get<T>();
        else
            opt.reset();
    }
};

// Specialization of nlohmann::adl_serializer for std::shared_ptr<T>
template <typename T>
struct nlohmann::adl_serializer<std::shared_ptr<T>> {
    static void
    to_json(json& j, const std::shared_ptr<T>& ptr) {
        if (ptr)
            j = *ptr;
        else
            j = nullptr;
    }

    static void
    from_json(const json& j, std::shared_ptr<T>& ptr) {
        if (!j.is_null())
            ptr = std::make_shared<T>(j.get<T>());
        else
            ptr.reset();
    }
};

// Specialization of nlohmann::adl_serializer for cudnnFraction_t
template <>
struct nlohmann::adl_serializer<cudnnFraction_t> {
    static void
    to_json(json& j, const cudnnFraction_t& fraction) {
        j = fraction.numerator;
    }

    static void
    from_json(const json& j, cudnnFraction_t& fraction) {
        fraction.numerator = j;
    }
};

#else
#define NLOHMANN_JSON_SERIALIZE_ENUM(ENUM_TYPE, ...)
#define NLOHMANN_DEFINE_TYPE_INTRUSIVE(Type, ...)
#endif

#include "cudnn_frontend_shim.h"
#include "cudnn_backend_base.h"
#include "cudnn_frontend_Logging.h"

#ifndef NV_CUDNN_DISABLE_EXCEPTION
#ifdef _MSC_VER
#pragma warning(disable : 4702)  // if exceptions are enabled there are unreachable return statements
#endif
#endif

#define CUDNN_FRONTEND_UNUSED(X) ((void)X)
namespace cudnn_frontend {

/// Detailed feature_vector. Generally the Tensor and Operation properties
using feature_vector_t = std::vector<int64_t>;

class cudnnException : public std::runtime_error {
   public:
    cudnnException(const char* message, cudnnStatus_t status) throw() : std::runtime_error(message) {
        error_status = status;
    }
    virtual const char*
    what() const throw() {
        return std::runtime_error::what();
    }
    cudnnStatus_t
    getCudnnStatus() {
        return error_status;
    }

    cudnnStatus_t error_status;
};

static inline bool
AllowAll(cudnnBackendDescriptor_t engine_config) {
    (void)engine_config;
    return false;
}

static inline std::string
to_string(cudnnStatus_t const status) {
    return detail::get_error_string(status);
}

#ifndef NV_CUDNN_DISABLE_EXCEPTION
[[noreturn]]
#endif
static inline void
set_error_and_throw_exception(BackendDescriptor const* desc, cudnnStatus_t status, const char* message) {

    std::string padded_message = std::string(message) + detail::get_last_error_string_();
    if (desc != nullptr) {
        desc->set_status(status);
        desc->set_error(padded_message.c_str());
    }
#ifndef NV_CUDNN_DISABLE_EXCEPTION
    throw cudnnException(
        std::string(std::string(padded_message) + std::string(" cudnn_status: ") + to_string(status)).c_str(), status);
#endif
}

static inline std::string
to_string(cudnnBackendBehaviorNote_t note) {
    switch (note) {
        case CUDNN_BEHAVIOR_NOTE_RUNTIME_COMPILATION:
            return std::string("CUDNN_BEHAVIOR_NOTE_RUNTIME_COMPILATION");
        case CUDNN_BEHAVIOR_NOTE_REQUIRES_FILTER_INT8x32_REORDER:
            return std::string("CUDNN_BEHAVIOR_NOTE_REQUIRES_FILTER_INT8x32_REORDER");
        case CUDNN_BEHAVIOR_NOTE_REQUIRES_BIAS_INT8x32_REORDER:
            return std::string("CUDNN_BEHAVIOR_NOTE_REQUIRES_BIAS_INT8x32_REORDER");
        case CUDNN_BEHAVIOR_NOTE_TYPE_COUNT:
            return std::string("CUDNN_BEHAVIOR_NOTE_TYPE_COUNT");
            // If none of the above cases hit, its definitely strict nan prop and should raise an error.
#if (CUDNN_VERSION >= 90500)
        case CUDNN_BEHAVIOR_NOTE_SUPPORTS_CUDA_GRAPH_NATIVE_API:
            return std::string("CUDNN_BEHAVIOR_NOTE_SUPPORTS_CUDA_GRAPH_NATIVE_API");
#endif
#if (CUDNN_VERSION >= 91500)
        case CUDNN_BEHAVIOR_NOTE_CUBLASLT_DEPENDENCY:
            return std::string("CUDNN_BEHAVIOR_NOTE_CUBLASLT_DEPENDENCY");
#endif
#ifndef NO_DEFAULT_IN_SWITCH
        default:
            return std::string("UNKNOWN_BEHAVIOR_NOTE");
#endif
    }
    return std::string("INVALID_BEHAVIOR_NOTE");
}

static inline std::string
to_string(cudnnBackendNumericalNote_t note) {
    switch (note) {
        case CUDNN_NUMERICAL_NOTE_TENSOR_CORE:
            return std::string("CUDNN_NUMERICAL_NOTE_TENSOR_CORE");
        case CUDNN_NUMERICAL_NOTE_DOWN_CONVERT_INPUTS:
            return std::string("CUDNN_NUMERICAL_NOTE_DOWN_CONVERT_INPUTS");
        case CUDNN_NUMERICAL_NOTE_REDUCED_PRECISION_REDUCTION:
            return std::string("CUDNN_NUMERICAL_NOTE_REDUCED_PRECISION_REDUCTION");
        case CUDNN_NUMERICAL_NOTE_FFT:
            return std::string("CUDNN_NUMERICAL_NOTE_FFT");
        case CUDNN_NUMERICAL_NOTE_NONDETERMINISTIC:
            return std::string("CUDNN_NUMERICAL_NOTE_NONDETERMINISTIC");
        case CUDNN_NUMERICAL_NOTE_WINOGRAD:
            return std::string("CUDNN_NUMERICAL_NOTE_WINOGRAD");
        case CUDNN_NUMERICAL_NOTE_WINOGRAD_TILE_4x4:
            return std::string("CUDNN_NUMERICAL_NOTE_WINOGRAD_TILE_4x4");
        case CUDNN_NUMERICAL_NOTE_WINOGRAD_TILE_6x6:
            return std::string("CUDNN_NUMERICAL_NOTE_WINOGRAD_TILE_6x6");
        case CUDNN_NUMERICAL_NOTE_WINOGRAD_TILE_13x13:
            return std::string("CUDNN_NUMERICAL_NOTE_WINOGRAD_TILE_13x13");
        case CUDNN_NUMERICAL_NOTE_TYPE_COUNT:
            return std::string("CUDNN_NUMERICAL_NOTE_TYPE_COUNT");

            // If none of the above cases hit, its definitely strict nan prop and should raise an error.
#if (CUDNN_VERSION >= 90100)
        case CUDNN_NUMERICAL_NOTE_STRICT_NAN_PROP:
            return std::string("CUDNN_NUMERICAL_NOTE_STRICT_NAN_PROP");
#endif
#ifndef NO_DEFAULT_IN_SWITCH
        default:
            return std::string("UNKNOWN_NUMERICAL_NOTE");
#endif
    }
    return std::string("INVALID_NUMERICAL_NOTE");
}

#if (CUDNN_VERSION >= 8700)
static inline std::string
to_string(cudnnRngDistribution_t distribution) {
    switch (distribution) {
        case CUDNN_RNG_DISTRIBUTION_BERNOULLI:
            return std::string("CUDNN_RNG_DISTRIBUTION_BERNOULLI");
        case CUDNN_RNG_DISTRIBUTION_UNIFORM:
            return std::string("CUDNN_RNG_DISTRIBUTION_UNIFORM");
        case CUDNN_RNG_DISTRIBUTION_NORMAL:
            return std::string("CUDNN_RNG_DISTRIBUTION_NORMAL");
#ifndef NO_DEFAULT_IN_SWITCH
        default:
            return std::string("UNKNOWN_CUDNN_DISTRIBUTION");
#endif
    }
    return std::string("");
}
#endif

enum class BuildPlanPolicy_t {
    // Builds and stores the "first successful" plan from the list returned by heuristics.
    // heuristics list is traversed sequentially and in decreasing order of potential performance.
    HEURISTICS_CHOICE,
    // Builds and stores all the "successful" plans from the list returned by heuristics.
    ALL,
};

NLOHMANN_JSON_SERIALIZE_ENUM(BuildPlanPolicy_t,
                             {
                                 {BuildPlanPolicy_t::HEURISTICS_CHOICE, "HEURISTICS_CHOICE"},
                                 {BuildPlanPolicy_t::ALL, "ALL"},
                             })

enum class TensorReordering_t {
    NONE,
    INT8x32,
    F16x16,
    F8_128x4,
};

NLOHMANN_JSON_SERIALIZE_ENUM(TensorReordering_t,
                             {
                                 {TensorReordering_t::NONE, "NONE"},
                                 {TensorReordering_t::INT8x32, "INT8x32"},
                                 {TensorReordering_t::F16x16, "F16x16"},
                                 {TensorReordering_t::F8_128x4, "F8_128x4"},
                             })

enum class ResampleMode_t {
    NOT_SET,

    AVGPOOL_EXCLUDE_PADDING,
    AVGPOOL_INCLUDE_PADDING,
    BILINEAR,
    NEAREST,
    MAXPOOL,
};

NLOHMANN_JSON_SERIALIZE_ENUM(ResampleMode_t,
                             {
                                 {ResampleMode_t::NOT_SET, nullptr},
                                 {ResampleMode_t::AVGPOOL_EXCLUDE_PADDING, "AVGPOOL_EXCLUDE_PADDING"},
                                 {ResampleMode_t::AVGPOOL_INCLUDE_PADDING, "AVGPOOL_INCLUDE_PADDING"},
                                 {ResampleMode_t::BILINEAR, "BILINEAR"},
                                 {ResampleMode_t::NEAREST, "NEAREST"},
                                 {ResampleMode_t::MAXPOOL, "MAXPOOL"},
                             })

enum class PaddingMode_t {
    NOT_SET,

    EDGE_VAL_PAD,
    NEG_INF_PAD,
    ZERO_PAD
};

enum class ReshapeMode_t {
    NOT_SET,

    VIEW_ONLY,
    LOGICAL
};

NLOHMANN_JSON_SERIALIZE_ENUM(ReshapeMode_t,
                             {
                                 {ReshapeMode_t::NOT_SET, nullptr},
                                 {ReshapeMode_t::VIEW_ONLY, "VIEW_ONLY"},
                                 {ReshapeMode_t::LOGICAL, "LOGICAL"},
                             })

enum class ConvolutionMode_t {
    NOT_SET,

    CONVOLUTION,
    CROSS_CORRELATION,
};

NLOHMANN_JSON_SERIALIZE_ENUM(ConvolutionMode_t,
                             {
                                 {ConvolutionMode_t::CONVOLUTION, "CONVOLUTION"},
                                 {ConvolutionMode_t::CROSS_CORRELATION, "CROSS_CORRELATION"},
                             })

NLOHMANN_JSON_SERIALIZE_ENUM(PaddingMode_t,
                             {
                                 {PaddingMode_t::NOT_SET, nullptr},
                                 {PaddingMode_t::EDGE_VAL_PAD, "EDGE_VAL_PAD"},
                                 {PaddingMode_t::NEG_INF_PAD, "NEG_INF_PAD"},
                                 {PaddingMode_t::ZERO_PAD, "ZERO_PAD"},
                             })

enum class NormFwdPhase_t {
    NOT_SET,

    INFERENCE,
    TRAINING
};

NLOHMANN_JSON_SERIALIZE_ENUM(NormFwdPhase_t,
                             {
                                 {NormFwdPhase_t::NOT_SET, nullptr},
                                 {NormFwdPhase_t::INFERENCE, "INFERENCE"},
                                 {NormFwdPhase_t::TRAINING, "TRAINING"},
                             })

enum class MoeGroupedMatmulMode_t {
    NOT_SET,

    NONE,
    GATHER,
    SCATTER
};

NLOHMANN_JSON_SERIALIZE_ENUM(MoeGroupedMatmulMode_t,
                             {
                                 {MoeGroupedMatmulMode_t::NOT_SET, nullptr},
                                 {MoeGroupedMatmulMode_t::NONE, "NONE"},
                                 {MoeGroupedMatmulMode_t::GATHER, "GATHER"},
                                 {MoeGroupedMatmulMode_t::SCATTER, "SCATTER"},
                             })

enum class DescriptorType_t {
    NOT_SET,

    POINTWISE_DESCRIPTOR,
    CONVOLUTION_DESCRIPTOR,
    ENGINE_DESCRIPTOR,
    ENGINECFG_DESCRIPTOR,
    ENGINEHEUR_DESCRIPTOR,
    EXECUTION_PLAN_DESCRIPTOR,
    INTERMEDIATE_INFO_DESCRIPTOR,
    KNOB_CHOICE_DESCRIPTOR,
    KNOB_INFO_DESCRIPTOR,
    LAYOUT_INFO_DESCRIPTOR,
    OPERATION_CONVOLUTION_FORWARD_DESCRIPTOR,
    OPERATION_CONVOLUTION_BACKWARD_FILTER_DESCRIPTOR,
    OPERATION_CONVOLUTION_BACKWARD_DATA_DESCRIPTOR,
    OPERATION_POINTWISE_DESCRIPTOR,
    OPERATION_GEN_STATS_DESCRIPTOR,
    OPERATIONGRAPH_DESCRIPTOR,
    VARIANT_PACK_DESCRIPTOR,
    TENSOR_DESCRIPTOR,
    MATMUL_DESCRIPTOR,
    OPERATION_MATMUL_DESCRIPTOR,
    OPERATION_BN_FINALIZE_STATISTICS_DESCRIPTOR,
    REDUCTION_DESCRIPTOR,
    OPERATION_REDUCTION_DESCRIPTOR,
    OPERATION_BN_BWD_WEIGHTS_DESCRIPTOR,
    RESAMPLE_DESCRIPTOR,
    OPERATION_RESAMPLE_FWD_DESCRIPTOR,
    OPERATION_RESAMPLE_BWD_DESCRIPTOR,
    OPERATION_CONCAT_DESCRIPTOR,
    OPERATION_SIGNAL_DESCRIPTOR,
    OPERATION_NORM_FORWARD_DESCRIPTOR,
    OPERATION_NORM_BACKWARD_DESCRIPTOR,
    OPERATION_RESHAPE_DESCRIPTOR,
    RNG_DESCRIPTOR,
    OPERATION_RNG_DESCRIPTOR,
    OPERATION_PAGED_CACHE_LOAD_DESCRIPTOR,
    OPERATION_BLOCK_SCALE_QUANTIZE_DESCRIPTOR,
    OPERATION_BLOCK_SCALE_DEQUANTIZE_DESCRIPTOR,
    OPERATION_CONCATENATE_DESCRIPTOR,
    OPERATION_MOE_GROUPED_MATMUL_DESCRIPTOR,
    OPERATION_MOE_GROUPED_MATMUL_BWD_DESCRIPTOR,
    OPERATION_TRANSPOSE_DESCRIPTOR,
    OPERATION_SLICE_DESCRIPTOR
};

enum class NormMode_t {
    NOT_SET,

    LAYER_NORM,
    INSTANCE_NORM,
    BATCH_NORM,
    GROUP_NORM,
    RMS_NORM,
    ADA_LAYER_NORM,
};

NLOHMANN_JSON_SERIALIZE_ENUM(NormMode_t,
                             {
                                 {NormMode_t::NOT_SET, nullptr},
                                 {NormMode_t::LAYER_NORM, "LAYER_NORM"},
                                 {NormMode_t::INSTANCE_NORM, "INSTANCE_NORM"},
                                 {NormMode_t::BATCH_NORM, "BATCH_NORM"},
                                 {NormMode_t::GROUP_NORM, "GROUP_NORM"},
                                 {NormMode_t::RMS_NORM, "RMS_NORM"},
                                 {NormMode_t::ADA_LAYER_NORM, "ADA_LAYER_NORM"},
                             })

enum class PointwiseMode_t {
    NOT_SET,

    ADD,
    MUL,
    SQRT,
    MAX,
    MIN,
    RELU_FWD,
    TANH_FWD,
    SIGMOID_FWD,
    ELU_FWD,
    GELU_FWD,
    SOFTPLUS_FWD,
    SWISH_FWD,
    RELU_BWD,
    TANH_BWD,
    SIGMOID_BWD,
    ELU_BWD,
    GELU_BWD,
    SOFTPLUS_BWD,
    SWISH_BWD,
    ERF,
    IDENTITY,
    GELU_APPROX_TANH_BWD,
    GELU_APPROX_TANH_FWD,
    GEN_INDEX,
    BINARY_SELECT,
    EXP,
    LOG,
    NEG,
    MOD,
    POW,
    ABS,
    CEIL,
    COS,
    FLOOR,
    RSQRT,
    SIN,
    LOGICAL_NOT,
    TAN,
    SUB,
    ADD_SQUARE,
    DIV,
    CMP_EQ,
    CMP_NEQ,
    CMP_GT,
    CMP_GE,
    CMP_LT,
    CMP_LE,
    LOGICAL_AND,
    LOGICAL_OR,
    RECIPROCAL,
};

NLOHMANN_JSON_SERIALIZE_ENUM(PointwiseMode_t,
                             {
                                 {PointwiseMode_t::NOT_SET, nullptr},
                                 {PointwiseMode_t::ADD, "ADD"},
                                 {PointwiseMode_t::MUL, "MUL"},
                                 {PointwiseMode_t::SQRT, "SQRT"},
                                 {PointwiseMode_t::MAX, "MAX"},
                                 {PointwiseMode_t::MIN, "MIN"},
                                 {PointwiseMode_t::RELU_FWD, "RELU_FWD"},
                                 {PointwiseMode_t::TANH_FWD, "TANH_FWD"},
                                 {PointwiseMode_t::SIGMOID_FWD, "SIGMOID_FWD"},
                                 {PointwiseMode_t::ELU_FWD, "ELU_FWD"},
                                 {PointwiseMode_t::GELU_FWD, "GELU_FWD"},
                                 {PointwiseMode_t::SOFTPLUS_FWD, "SOFTPLUS_FWD"},
                                 {PointwiseMode_t::SWISH_FWD, "SWISH_FWD"},
                                 {PointwiseMode_t::RELU_BWD, "RELU_BWD"},
                                 {PointwiseMode_t::TANH_BWD, "TANH_BWD"},
                                 {PointwiseMode_t::SIGMOID_BWD, "SIGMOID_BWD"},
                                 {PointwiseMode_t::ELU_BWD, "ELU_BWD"},
                                 {PointwiseMode_t::GELU_BWD, "GELU_BWD"},
                                 {PointwiseMode_t::SOFTPLUS_BWD, "SOFTPLUS_BWD"},
                                 {PointwiseMode_t::SWISH_BWD, "SWISH_BWD"},
                                 {PointwiseMode_t::ERF, "ERF"},
                                 {PointwiseMode_t::IDENTITY, "IDENTITY"},
                                 {PointwiseMode_t::GELU_APPROX_TANH_BWD, "GELU_APPROX_TANH_BWD"},
                                 {PointwiseMode_t::GELU_APPROX_TANH_FWD, "GELU_APPROX_TANH_FWD"},
                                 {PointwiseMode_t::GEN_INDEX, "GEN_INDEX"},
                                 {PointwiseMode_t::BINARY_SELECT, "BINARY_SELECT"},
                                 {PointwiseMode_t::EXP, "EXP"},
                                 {PointwiseMode_t::LOG, "LOG"},
                                 {PointwiseMode_t::NEG, "NEG"},
                                 {PointwiseMode_t::MOD, "MOD"},
                                 {PointwiseMode_t::POW, "POW"},
                                 {PointwiseMode_t::ABS, "ABS"},
                                 {PointwiseMode_t::CEIL, "CEIL"},
                                 {PointwiseMode_t::COS, "COS"},
                                 {PointwiseMode_t::FLOOR, "FLOOR"},
                                 {PointwiseMode_t::RSQRT, "RSQRT"},
                                 {PointwiseMode_t::SIN, "SIN"},
                                 {PointwiseMode_t::LOGICAL_NOT, "LOGICAL_NOT"},
                                 {PointwiseMode_t::TAN, "TAN"},
                                 {PointwiseMode_t::SUB, "SUB"},
                                 {PointwiseMode_t::ADD_SQUARE, "ADD_SQUARE"},
                                 {PointwiseMode_t::DIV, "DIV"},
                                 {PointwiseMode_t::CMP_EQ, "CMP_EQ"},
                                 {PointwiseMode_t::CMP_NEQ, "CMP_NEQ"},
                                 {PointwiseMode_t::CMP_GT, "CMP_GT"},
                                 {PointwiseMode_t::CMP_GE, "CMP_GE"},
                                 {PointwiseMode_t::CMP_LT, "CMP_LT"},
                                 {PointwiseMode_t::CMP_LE, "CMP_LE"},
                                 {PointwiseMode_t::LOGICAL_AND, "LOGICAL_AND"},
                                 {PointwiseMode_t::LOGICAL_OR, "LOGICAL_OR"},
                                 {PointwiseMode_t::RECIPROCAL, "RECIPROCAL"},
                             })

enum class HeurMode_t {
    A,
    B,
    FALLBACK,
    OPENSOURCE,
};

NLOHMANN_JSON_SERIALIZE_ENUM(HeurMode_t,
                             {
                                 {HeurMode_t::A, "A"},
                                 {HeurMode_t::B, "B"},
                                 {HeurMode_t::FALLBACK, "FALLBACK"},
                                 {HeurMode_t::OPENSOURCE, "OPENSOURCE"},
                             })

enum class BehaviorNote_t {
    NOT_SET,

    RUNTIME_COMPILATION,
    REQUIRES_FILTER_INT8x32_REORDER,
    REQUIRES_BIAS_INT8x32_REORDER,
    SUPPORTS_CUDA_GRAPH_NATIVE_API,
    CUBLASLT_DEPENDENCY,
};

NLOHMANN_JSON_SERIALIZE_ENUM(BehaviorNote_t,
                             {
                                 {BehaviorNote_t::NOT_SET, "NOT_SET"},
                                 {BehaviorNote_t::RUNTIME_COMPILATION, "RUNTIME_COMPILATION"},
                                 {BehaviorNote_t::REQUIRES_FILTER_INT8x32_REORDER, "REQUIRES_FILTER_INT8x32_REORDER"},
                                 {BehaviorNote_t::REQUIRES_BIAS_INT8x32_REORDER, "REQUIRES_BIAS_INT8x32_REORDER"},
                                 {BehaviorNote_t::SUPPORTS_CUDA_GRAPH_NATIVE_API, "SUPPORTS_CUDA_GRAPH_NATIVE_API"},
                                 {BehaviorNote_t::CUBLASLT_DEPENDENCY, "CUBLASLT_DEPENDENCY"},
                             })

enum class NumericalNote_t {
    NOT_SET,

    TENSOR_CORE,
    DOWN_CONVERT_INPUTS,
    REDUCED_PRECISION_REDUCTION,
    FFT,
    NONDETERMINISTIC,
    WINOGRAD,
    WINOGRAD_TILE_4x4,
    WINOGRAD_TILE_6x6,
    WINOGRAD_TILE_13x13,
    STRICT_NAN_PROP,
};

NLOHMANN_JSON_SERIALIZE_ENUM(NumericalNote_t,
                             {
                                 {NumericalNote_t::NOT_SET, "NOT_SET"},
                                 {NumericalNote_t::TENSOR_CORE, "TENSOR_CORE"},
                                 {NumericalNote_t::DOWN_CONVERT_INPUTS, "DOWN_CONVERT_INPUTS"},
                                 {NumericalNote_t::REDUCED_PRECISION_REDUCTION, "REDUCED_PRECISION_REDUCTION"},
                                 {NumericalNote_t::FFT, "FFT"},
                                 {NumericalNote_t::NONDETERMINISTIC, "NONDETERMINISTIC"},
                                 {NumericalNote_t::WINOGRAD, "WINOGRAD"},
                                 {NumericalNote_t::WINOGRAD_TILE_4x4, "WINOGRAD_TILE_4x4"},
                                 {NumericalNote_t::WINOGRAD_TILE_6x6, "WINOGRAD_TILE_6x6"},
                                 {NumericalNote_t::WINOGRAD_TILE_13x13, "WINOGRAD_TILE_13x13"},
                                 {NumericalNote_t::STRICT_NAN_PROP, "STRICT_NAN_PROP"},
                             })

enum class DataType_t {
    NOT_SET,

    FLOAT,
    DOUBLE,
    HALF,
    INT8,
    INT32,
    INT8x4,
    UINT8,
    UINT8x4,
    INT8x32,
    BFLOAT16,
    INT64,
    BOOLEAN,
    FP8_E4M3,
    FP8_E5M2,
    FAST_FLOAT_FOR_FP8,
    FP8_E8M0,
    FP4_E2M1,
    INT4,
    COMPLEX_FP32,
    COMPLEX_FP64,
};

NLOHMANN_JSON_SERIALIZE_ENUM(DataType_t,
                             {
                                 {DataType_t::NOT_SET, nullptr},
                                 {DataType_t::FLOAT, "FLOAT"},
                                 {DataType_t::DOUBLE, "DOUBLE"},
                                 {DataType_t::HALF, "HALF"},
                                 {DataType_t::INT8, "INT8"},
                                 {DataType_t::INT32, "INT32"},
                                 {DataType_t::INT8x4, "INT8x4"},
                                 {DataType_t::UINT8, "UINT8"},
                                 {DataType_t::UINT8x4, "UINT8x4"},
                                 {DataType_t::INT8x32, "INT8x32"},
                                 {DataType_t::BFLOAT16, "BFLOAT16"},
                                 {DataType_t::INT64, "INT64"},
                                 {DataType_t::BOOLEAN, "BOOLEAN"},
                                 {DataType_t::FP8_E4M3, "FP8_E4M3"},
                                 {DataType_t::FP8_E5M2, "FP8_E5M2"},
                                 {DataType_t::FAST_FLOAT_FOR_FP8, "FAST_FLOAT_FOR_FP8"},
                                 {DataType_t::FP8_E8M0, "FP8_E8M0"},
                                 {DataType_t::FP4_E2M1, "FP4_E2M1"},
                                 {DataType_t::INT4, "INT4"},
                                 {DataType_t::COMPLEX_FP32, "COMPLEX_FP32"},
                                 {DataType_t::COMPLEX_FP64, "COMPLEX_FP64"},

                             })

enum class ReductionMode_t {
    NOT_SET,

    ADD,
    MUL,
    MIN,
    MAX,
    AMAX,
    AVG,
    NORM1,
    NORM2,
    MUL_NO_ZEROS
};

NLOHMANN_JSON_SERIALIZE_ENUM(ReductionMode_t,
                             {
                                 {ReductionMode_t::NOT_SET, nullptr},
                                 {ReductionMode_t::ADD, "ADD"},
                                 {ReductionMode_t::MUL, "MUL"},
                                 {ReductionMode_t::MIN, "MIN"},
                                 {ReductionMode_t::MAX, "MAX"},
                                 {ReductionMode_t::AMAX, "AMAX"},
                                 {ReductionMode_t::AVG, "AVG"},
                                 {ReductionMode_t::NORM1, "NORM1"},
                                 {ReductionMode_t::NORM2, "NORM2"},
                                 {ReductionMode_t::MUL_NO_ZEROS, "MUL_NO_ZEROS"},
                             })

enum class RngDistribution_t {
    NOT_SET,

    BERNOULLI,
    UNIFORM,
    NORMAL,
};

NLOHMANN_JSON_SERIALIZE_ENUM(RngDistribution_t,
                             {
                                 {RngDistribution_t::NOT_SET, nullptr},
                                 {RngDistribution_t::BERNOULLI, "BERNOULLI"},
                                 {RngDistribution_t::UNIFORM, "UNIFORM"},
                                 {RngDistribution_t::NORMAL, "NORMAL"},
                             })

static int64_t
get_pointwise_mode_port_count(PointwiseMode_t const& mode) {
    switch (mode) {
        case PointwiseMode_t::NOT_SET:
            return 0;

        case PointwiseMode_t::ADD:
        case PointwiseMode_t::MUL:
        case PointwiseMode_t::DIV:
        case PointwiseMode_t::ADD_SQUARE:
        case PointwiseMode_t::SUB:
        case PointwiseMode_t::CMP_EQ:
        case PointwiseMode_t::CMP_NEQ:
        case PointwiseMode_t::CMP_GT:
        case PointwiseMode_t::CMP_GE:
        case PointwiseMode_t::CMP_LT:
        case PointwiseMode_t::CMP_LE:
        case PointwiseMode_t::LOGICAL_AND:
        case PointwiseMode_t::LOGICAL_OR:
        case PointwiseMode_t::MIN:
        case PointwiseMode_t::MAX:
        case PointwiseMode_t::MOD:
        case PointwiseMode_t::RELU_BWD:
        case PointwiseMode_t::TANH_BWD:
        case PointwiseMode_t::SIGMOID_BWD:
        case PointwiseMode_t::ELU_BWD:
        case PointwiseMode_t::GELU_BWD:
        case PointwiseMode_t::SOFTPLUS_BWD:
        case PointwiseMode_t::SWISH_BWD:
        case PointwiseMode_t::GELU_APPROX_TANH_BWD:
        case PointwiseMode_t::POW:
            return 3;

        case PointwiseMode_t::SQRT:
        case PointwiseMode_t::RELU_FWD:
        case PointwiseMode_t::TANH_FWD:
        case PointwiseMode_t::SIGMOID_FWD:
        case PointwiseMode_t::ELU_FWD:
        case PointwiseMode_t::GELU_FWD:
        case PointwiseMode_t::SOFTPLUS_FWD:
        case PointwiseMode_t::SWISH_FWD:
        case PointwiseMode_t::EXP:
        case PointwiseMode_t::LOG:
        case PointwiseMode_t::NEG:
        case PointwiseMode_t::ABS:
        case PointwiseMode_t::CEIL:
        case PointwiseMode_t::FLOOR:
        case PointwiseMode_t::COS:
        case PointwiseMode_t::TAN:
        case PointwiseMode_t::SIN:
        case PointwiseMode_t::RSQRT:
        case PointwiseMode_t::LOGICAL_NOT:
        case PointwiseMode_t::GEN_INDEX:
        case PointwiseMode_t::ERF:
        case PointwiseMode_t::GELU_APPROX_TANH_FWD:
        case PointwiseMode_t::IDENTITY:
        case PointwiseMode_t::RECIPROCAL:
            return 2;

        case PointwiseMode_t::BINARY_SELECT:
            return 4;
    }
    return -1;
}

static inline std::ostream&
operator<<(std::ostream& os, const DescriptorType_t& mode) {
    switch (mode) {
        case DescriptorType_t::POINTWISE_DESCRIPTOR:
            os << "POINTWISE_DESCRIPTOR";
            break;
        case DescriptorType_t::CONVOLUTION_DESCRIPTOR:
            os << "CONVOLUTION_DESCRIPTOR";
            break;
        case DescriptorType_t::ENGINE_DESCRIPTOR:
            os << "ENGINE_DESCRIPTOR";
            break;
        case DescriptorType_t::ENGINECFG_DESCRIPTOR:
            os << "ENGINECFG_DESCRIPTOR";
            break;
        case DescriptorType_t::ENGINEHEUR_DESCRIPTOR:
            os << "ENGINEHEUR_DESCRIPTOR";
            break;
        case DescriptorType_t::EXECUTION_PLAN_DESCRIPTOR:
            os << "EXECUTION_PLAN_DESCRIPTOR";
            break;
        case DescriptorType_t::INTERMEDIATE_INFO_DESCRIPTOR:
            os << "INTERMEDIATE_INFO_DESCRIPTOR";
            break;
        case DescriptorType_t::KNOB_CHOICE_DESCRIPTOR:
            os << "KNOB_CHOICE_DESCRIPTOR";
            break;
        case DescriptorType_t::KNOB_INFO_DESCRIPTOR:
            os << "KNOB_INFO_DESCRIPTOR";
            break;
        case DescriptorType_t::LAYOUT_INFO_DESCRIPTOR:
            os << "LAYOUT_INFO_DESCRIPTOR";
            break;
        case DescriptorType_t::OPERATION_CONVOLUTION_FORWARD_DESCRIPTOR:
            os << "OPERATION_CONVOLUTION_FORWARD_DESCRIPTOR";
            break;
        case DescriptorType_t::OPERATION_CONVOLUTION_BACKWARD_FILTER_DESCRIPTOR:
            os << "OPERATION_CONVOLUTION_BACKWARD_FILTER_DESCRIPTOR";
            break;
        case DescriptorType_t::OPERATION_CONVOLUTION_BACKWARD_DATA_DESCRIPTOR:
            os << "OPERATION_CONVOLUTION_BACKWARD_DATA_DESCRIPTOR";
            break;
        case DescriptorType_t::OPERATION_POINTWISE_DESCRIPTOR:
            os << "OPERATION_POINTWISE_DESCRIPTOR";
            break;
        case DescriptorType_t::OPERATION_GEN_STATS_DESCRIPTOR:
            os << "OPERATION_GEN_STATS_DESCRIPTOR";
            break;
        case DescriptorType_t::OPERATIONGRAPH_DESCRIPTOR:
            os << "OPERATIONGRAPH_DESCRIPTOR";
            break;
        case DescriptorType_t::VARIANT_PACK_DESCRIPTOR:
            os << "VARIANT_PACK_DESCRIPTOR";
            break;
        case DescriptorType_t::TENSOR_DESCRIPTOR:
            os << "TENSOR_DESCRIPTOR";
            break;
        case DescriptorType_t::MATMUL_DESCRIPTOR:
            os << "MATMUL_DESCRIPTOR";
            break;
        case DescriptorType_t::OPERATION_MATMUL_DESCRIPTOR:
            os << "OPERATION_MATMUL_DESCRIPTOR";
            break;
        case DescriptorType_t::OPERATION_BN_FINALIZE_STATISTICS_DESCRIPTOR:
            os << "OPERATION_BN_FINALIZE_STATISTICS_DESCRIPTOR";
            break;
        case DescriptorType_t::REDUCTION_DESCRIPTOR:
            os << "REDUCTION_DESCRIPTOR";
            break;
        case DescriptorType_t::OPERATION_REDUCTION_DESCRIPTOR:
            os << "OPERATION_REDUCTION_DESCRIPTOR";
            break;
        case DescriptorType_t::OPERATION_BN_BWD_WEIGHTS_DESCRIPTOR:
            os << "OPERATION_BN_BWD_WEIGHTS_DESCRIPTOR";
            break;
        case DescriptorType_t::RESAMPLE_DESCRIPTOR:
            os << "RESAMPLE_DESCRIPTOR";
            break;
        case DescriptorType_t::OPERATION_RESAMPLE_FWD_DESCRIPTOR:
            os << "OPERATION_RESAMPLE_FWD_DESCRIPTOR";
            break;
        case DescriptorType_t::OPERATION_RESAMPLE_BWD_DESCRIPTOR:
            os << "OPERATION_RESAMPLE_BWD_DESCRIPTOR";
            break;
        case DescriptorType_t::OPERATION_CONCAT_DESCRIPTOR:
            os << "OPERATION_CONCAT_DESCRIPTOR";
            break;
        case DescriptorType_t::OPERATION_SIGNAL_DESCRIPTOR:
            os << "OPERATION_SIGNAL_DESCRIPTOR";
            break;
        case DescriptorType_t::OPERATION_NORM_FORWARD_DESCRIPTOR:
            os << "OPERATION_NORM_FORWARD_DESCRIPTOR";
            break;
        case DescriptorType_t::OPERATION_NORM_BACKWARD_DESCRIPTOR:
            os << "OPERATION_NORM_BACKWARD_DESCRIPTOR";
            break;
        case DescriptorType_t::OPERATION_RESHAPE_DESCRIPTOR:
            os << "OPERATION_RESHAPE_DESCRIPTOR";
            break;
        case DescriptorType_t::RNG_DESCRIPTOR:
            os << "RNG_DESCRIPTOR";
            break;
        case DescriptorType_t::OPERATION_RNG_DESCRIPTOR:
            os << "OPERATION_RNG_DESCRIPTOR";
            break;
        case DescriptorType_t::OPERATION_PAGED_CACHE_LOAD_DESCRIPTOR:
            os << "OPERATION_PAGED_CACHE_LOAD_DESCRIPTOR";
            break;
        case DescriptorType_t::OPERATION_BLOCK_SCALE_QUANTIZE_DESCRIPTOR:
            os << "OPERATION_BLOCK_SCALE_QUANTIZE_DESCRIPTOR";
            break;
        case DescriptorType_t::OPERATION_BLOCK_SCALE_DEQUANTIZE_DESCRIPTOR:
            os << "OPERATION_BLOCK_SCALE_DEQUANTIZE_DESCRIPTOR";
            break;
        case DescriptorType_t::OPERATION_CONCATENATE_DESCRIPTOR:
            os << "OPERATION_CONCATENATE_DESCRIPTOR";
            break;
        case DescriptorType_t::OPERATION_MOE_GROUPED_MATMUL_DESCRIPTOR:
            os << "OPERATION_MOE_GROUPED_MATMUL_DESCRIPTOR";
            break;
        case DescriptorType_t::OPERATION_MOE_GROUPED_MATMUL_BWD_DESCRIPTOR:
            os << "OPERATION_MOE_GROUPED_MATMUL_BWD_DESCRIPTOR";
            break;
        case DescriptorType_t::OPERATION_TRANSPOSE_DESCRIPTOR:
            os << "OPERATION_TRANSPOSE_DESCRIPTOR";
            break;
        case DescriptorType_t::OPERATION_SLICE_DESCRIPTOR:
            os << "OPERATION_SLICE_DESCRIPTOR";
            break;
        case DescriptorType_t::NOT_SET:
            os << "NOT_SET";
            break;
    }
    return os;
}

enum class DiagonalAlignment_t { TOP_LEFT, BOTTOM_RIGHT };
NLOHMANN_JSON_SERIALIZE_ENUM(DiagonalAlignment_t,
                             {
                                 {DiagonalAlignment_t::TOP_LEFT, "TOP_LEFT"},
                                 {DiagonalAlignment_t::BOTTOM_RIGHT, "BOTTOM_RIGHT"},
                             })

enum class AttentionImplementation_t { AUTO, COMPOSITE, UNIFIED };
NLOHMANN_JSON_SERIALIZE_ENUM(AttentionImplementation_t,
                             {
                                 {AttentionImplementation_t::AUTO, "AUTO"},
                                 {AttentionImplementation_t::COMPOSITE, "COMPOSITE"},
                                 {AttentionImplementation_t::UNIFIED, "UNIFIED"},
                             })

namespace detail {

inline size_t
get_data_type_size(DataType_t const data_type) {
    switch (data_type) {
        case DataType_t::FLOAT:
            return sizeof(float);
        case DataType_t::DOUBLE:
            return sizeof(double);
        case DataType_t::HALF:
            return 2;  // 16-bit float
        case DataType_t::INT8:
        case DataType_t::UINT8:
            return 1;
        case DataType_t::INT32:
            return sizeof(int32_t);
        case DataType_t::INT8x4:
        case DataType_t::UINT8x4:
            return 4;
        case DataType_t::INT8x32:
            return 32;
        case DataType_t::BFLOAT16:
            return 2;
        case DataType_t::INT64:
            return sizeof(int64_t);
        case DataType_t::FP8_E4M3:
        case DataType_t::FP8_E5M2:
            return 1;  // 8-bit float
        case DataType_t::NOT_SET:
        case DataType_t::BOOLEAN:
        default:
            return 0;
    }
}

inline std::vector<float>
get_alibi_slope(int64_t const n_heads) {
    std::vector<float> slope;
#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable : 4244)  // this could be ommited with c++17 and contexpr
#endif
    int n = 1 << static_cast<int>(log2(static_cast<double>(n_heads)));
#ifdef _MSC_VER
#pragma warning(pop)
#endif
    for (int i = 0; i < n; i++) {
        slope.push_back((float)(i + 1.0));
    }

    for (int i = 0; i < 2 * (n_heads - n); i += 2) {
        slope.push_back(static_cast<float>(i + 1) * 0.5f);
    }

    for (float& elem : slope) {
        elem *= -8.0f;
        elem /= static_cast<float>(n);
        elem = powf(2.0, elem);
    }

    return slope;
}

static inline cudnnStatus_t
convert_to_cudnn_type(cudnn_frontend::DataType_t const mode, cudnnDataType_t& cudnn_mode) {
    switch (mode) {
        case DataType_t::FLOAT:
            cudnn_mode = CUDNN_DATA_FLOAT;
            return cudnnStatus_t::CUDNN_STATUS_SUCCESS;
        case DataType_t::DOUBLE:
            cudnn_mode = CUDNN_DATA_DOUBLE;
            return cudnnStatus_t::CUDNN_STATUS_SUCCESS;
        case DataType_t::HALF:
            cudnn_mode = CUDNN_DATA_HALF;
            return cudnnStatus_t::CUDNN_STATUS_SUCCESS;
        case DataType_t::INT8:
            cudnn_mode = CUDNN_DATA_INT8;
            return cudnnStatus_t::CUDNN_STATUS_SUCCESS;
        case DataType_t::INT32:
            cudnn_mode = CUDNN_DATA_INT32;
            return cudnnStatus_t::CUDNN_STATUS_SUCCESS;
        case DataType_t::INT8x4:
            cudnn_mode = CUDNN_DATA_INT8x4;
            return cudnnStatus_t::CUDNN_STATUS_SUCCESS;
        case DataType_t::UINT8:
            cudnn_mode = CUDNN_DATA_UINT8;
            return cudnnStatus_t::CUDNN_STATUS_SUCCESS;
        case DataType_t::UINT8x4:
            cudnn_mode = CUDNN_DATA_UINT8x4;
            return cudnnStatus_t::CUDNN_STATUS_SUCCESS;
        case DataType_t::INT8x32:
            cudnn_mode = CUDNN_DATA_INT8x32;
            return cudnnStatus_t::CUDNN_STATUS_SUCCESS;
        case DataType_t::BFLOAT16:
            cudnn_mode = CUDNN_DATA_BFLOAT16;
            return cudnnStatus_t::CUDNN_STATUS_SUCCESS;
        case DataType_t::INT64:
            cudnn_mode = CUDNN_DATA_INT64;
            return cudnnStatus_t::CUDNN_STATUS_SUCCESS;
        case DataType_t::BOOLEAN:
            cudnn_mode = CUDNN_DATA_BOOLEAN;
            return cudnnStatus_t::CUDNN_STATUS_SUCCESS;
        case DataType_t::FP8_E4M3:
#if (CUDNN_VERSION >= 8600)
            NV_CUDNN_FE_DYNAMIC_CHECK_CUDNN_BACKEND_VERSION(8600, cudnnStatus_t::CUDNN_STATUS_INVALID_VALUE);
            cudnn_mode = CUDNN_DATA_FP8_E4M3;
            return cudnnStatus_t::CUDNN_STATUS_SUCCESS;
#else
            return cudnnStatus_t::CUDNN_STATUS_INVALID_VALUE;
#endif
        case DataType_t::FP8_E5M2:
#if (CUDNN_VERSION >= 8600)
            NV_CUDNN_FE_DYNAMIC_CHECK_CUDNN_BACKEND_VERSION(8600, cudnnStatus_t::CUDNN_STATUS_INVALID_VALUE);
            cudnn_mode = CUDNN_DATA_FP8_E5M2;
            return cudnnStatus_t::CUDNN_STATUS_SUCCESS;
#else
            return cudnnStatus_t::CUDNN_STATUS_INVALID_VALUE;
#endif
        case DataType_t::FAST_FLOAT_FOR_FP8:
#if (CUDNN_VERSION >= 8700)
            NV_CUDNN_FE_DYNAMIC_CHECK_CUDNN_BACKEND_VERSION(8700, cudnnStatus_t::CUDNN_STATUS_INVALID_VALUE);
            cudnn_mode = CUDNN_DATA_FAST_FLOAT_FOR_FP8;
            return cudnnStatus_t::CUDNN_STATUS_SUCCESS;
#else
            return cudnnStatus_t::CUDNN_STATUS_INVALID_VALUE;
#endif
        case DataType_t::FP8_E8M0:
#if (CUDNN_VERSION >= 90700)
            NV_CUDNN_FE_DYNAMIC_CHECK_CUDNN_BACKEND_VERSION(90700, cudnnStatus_t::CUDNN_STATUS_INVALID_VALUE);
            cudnn_mode = CUDNN_DATA_FP8_E8M0;
            return cudnnStatus_t::CUDNN_STATUS_SUCCESS;
#else
            return cudnnStatus_t::CUDNN_STATUS_INVALID_VALUE;
#endif
        case DataType_t::FP4_E2M1:
#if (CUDNN_VERSION >= 90700)
            NV_CUDNN_FE_DYNAMIC_CHECK_CUDNN_BACKEND_VERSION(90700, cudnnStatus_t::CUDNN_STATUS_INVALID_VALUE);
            cudnn_mode = CUDNN_DATA_FP4_E2M1;
            return cudnnStatus_t::CUDNN_STATUS_SUCCESS;
#else
            return cudnnStatus_t::CUDNN_STATUS_INVALID_VALUE;
#endif
        case DataType_t::INT4:
#if (CUDNN_VERSION >= 91100)
            NV_CUDNN_FE_DYNAMIC_CHECK_CUDNN_BACKEND_VERSION(91000, cudnnStatus_t::CUDNN_STATUS_INVALID_VALUE);
            cudnn_mode = CUDNN_DATA_INT4;
            return cudnnStatus_t::CUDNN_STATUS_SUCCESS;
#else
            return cudnnStatus_t::CUDNN_STATUS_INVALID_VALUE;
#endif
        case DataType_t::COMPLEX_FP32:
#if (CUDNN_VERSION >= 91400)
            NV_CUDNN_FE_DYNAMIC_CHECK_CUDNN_BACKEND_VERSION(91400, cudnnStatus_t::CUDNN_STATUS_INVALID_VALUE);
            cudnn_mode = CUDNN_DATA_COMPLEX_FP32;
            return cudnnStatus_t::CUDNN_STATUS_SUCCESS;
#else
            return cudnnStatus_t::CUDNN_STATUS_INVALID_VALUE;
#endif
        case DataType_t::COMPLEX_FP64:
#if (CUDNN_VERSION >= 91400)
            NV_CUDNN_FE_DYNAMIC_CHECK_CUDNN_BACKEND_VERSION(91400, cudnnStatus_t::CUDNN_STATUS_INVALID_VALUE);
            cudnn_mode = CUDNN_DATA_COMPLEX_FP64;
            return cudnnStatus_t::CUDNN_STATUS_SUCCESS;
#else
            return cudnnStatus_t::CUDNN_STATUS_INVALID_VALUE;
#endif
#ifndef NO_DEFAULT_IN_SWITCH
        default:
            return cudnnStatus_t::CUDNN_STATUS_INVALID_VALUE;
#endif
    }
    return cudnnStatus_t::CUDNN_STATUS_INVALID_VALUE;
}

static inline cudnnStatus_t
convert_to_cudnn_type(cudnn_frontend::ReductionMode_t const mode, cudnnReduceTensorOp_t& cudnn_mode) {
    switch (mode) {
        case ReductionMode_t::ADD:
            cudnn_mode = CUDNN_REDUCE_TENSOR_ADD;
            return cudnnStatus_t::CUDNN_STATUS_SUCCESS;
        case ReductionMode_t::MUL:
            cudnn_mode = CUDNN_REDUCE_TENSOR_MUL;
            return cudnnStatus_t::CUDNN_STATUS_SUCCESS;
        case ReductionMode_t::MIN:
            cudnn_mode = CUDNN_REDUCE_TENSOR_MIN;
            return cudnnStatus_t::CUDNN_STATUS_SUCCESS;
        case ReductionMode_t::MAX:
            cudnn_mode = CUDNN_REDUCE_TENSOR_MAX;
            return cudnnStatus_t::CUDNN_STATUS_SUCCESS;
        case ReductionMode_t::AMAX:
            cudnn_mode = CUDNN_REDUCE_TENSOR_AMAX;
            return cudnnStatus_t::CUDNN_STATUS_SUCCESS;
        case ReductionMode_t::AVG:
            cudnn_mode = CUDNN_REDUCE_TENSOR_AVG;
            return cudnnStatus_t::CUDNN_STATUS_SUCCESS;
        case ReductionMode_t::NORM1:
            cudnn_mode = CUDNN_REDUCE_TENSOR_NORM1;
            return cudnnStatus_t::CUDNN_STATUS_SUCCESS;
        case ReductionMode_t::NORM2:
            cudnn_mode = CUDNN_REDUCE_TENSOR_NORM2;
            return cudnnStatus_t::CUDNN_STATUS_SUCCESS;
        case ReductionMode_t::MUL_NO_ZEROS:
            cudnn_mode = CUDNN_REDUCE_TENSOR_MUL_NO_ZEROS;
            return cudnnStatus_t::CUDNN_STATUS_SUCCESS;

#ifndef NO_DEFAULT_IN_SWITCH
        default:
            return cudnnStatus_t::CUDNN_STATUS_INVALID_VALUE;
#endif
    }
    return cudnnStatus_t::CUDNN_STATUS_INVALID_VALUE;
}

static inline cudnnStatus_t
convert_to_cudnn_type(cudnn_frontend::PointwiseMode_t const mode, cudnnPointwiseMode_t& cudnn_mode) {
    switch (mode) {
        case PointwiseMode_t::ADD:
            cudnn_mode = CUDNN_POINTWISE_ADD;
            return cudnnStatus_t::CUDNN_STATUS_SUCCESS;
        case PointwiseMode_t::MUL:
            cudnn_mode = CUDNN_POINTWISE_MUL;
            return cudnnStatus_t::CUDNN_STATUS_SUCCESS;
        case PointwiseMode_t::SQRT:
            cudnn_mode = CUDNN_POINTWISE_SQRT;
            return cudnnStatus_t::CUDNN_STATUS_SUCCESS;
        case PointwiseMode_t::MAX:
            cudnn_mode = CUDNN_POINTWISE_MAX;
            return cudnnStatus_t::CUDNN_STATUS_SUCCESS;
        case PointwiseMode_t::MIN:
            cudnn_mode = CUDNN_POINTWISE_MIN;
            return cudnnStatus_t::CUDNN_STATUS_SUCCESS;
        case PointwiseMode_t::RELU_FWD:
            cudnn_mode = CUDNN_POINTWISE_RELU_FWD;
            return cudnnStatus_t::CUDNN_STATUS_SUCCESS;
        case PointwiseMode_t::TANH_FWD:
            cudnn_mode = CUDNN_POINTWISE_TANH_FWD;
            return cudnnStatus_t::CUDNN_STATUS_SUCCESS;
        case PointwiseMode_t::SIGMOID_FWD:
            cudnn_mode = CUDNN_POINTWISE_SIGMOID_FWD;
            return cudnnStatus_t::CUDNN_STATUS_SUCCESS;
        case PointwiseMode_t::ELU_FWD:
            cudnn_mode = CUDNN_POINTWISE_ELU_FWD;
            return cudnnStatus_t::CUDNN_STATUS_SUCCESS;
        case PointwiseMode_t::GELU_FWD:
            cudnn_mode = CUDNN_POINTWISE_GELU_FWD;
            return cudnnStatus_t::CUDNN_STATUS_SUCCESS;
        case PointwiseMode_t::SOFTPLUS_FWD:
            cudnn_mode = CUDNN_POINTWISE_SOFTPLUS_FWD;
            return cudnnStatus_t::CUDNN_STATUS_SUCCESS;
        case PointwiseMode_t::SWISH_FWD:
            cudnn_mode = CUDNN_POINTWISE_SWISH_FWD;
            return cudnnStatus_t::CUDNN_STATUS_SUCCESS;
        case PointwiseMode_t::RELU_BWD:
            cudnn_mode = CUDNN_POINTWISE_RELU_BWD;
            return cudnnStatus_t::CUDNN_STATUS_SUCCESS;
        case PointwiseMode_t::TANH_BWD:
            cudnn_mode = CUDNN_POINTWISE_TANH_BWD;
            return cudnnStatus_t::CUDNN_STATUS_SUCCESS;
        case PointwiseMode_t::SIGMOID_BWD:
            cudnn_mode = CUDNN_POINTWISE_SIGMOID_BWD;
            return cudnnStatus_t::CUDNN_STATUS_SUCCESS;
        case PointwiseMode_t::ELU_BWD:
            cudnn_mode = CUDNN_POINTWISE_ELU_BWD;
            return cudnnStatus_t::CUDNN_STATUS_SUCCESS;
        case PointwiseMode_t::GELU_BWD:
            cudnn_mode = CUDNN_POINTWISE_GELU_BWD;
            return cudnnStatus_t::CUDNN_STATUS_SUCCESS;
        case PointwiseMode_t::SOFTPLUS_BWD:
            cudnn_mode = CUDNN_POINTWISE_SOFTPLUS_BWD;
            return cudnnStatus_t::CUDNN_STATUS_SUCCESS;
        case PointwiseMode_t::SWISH_BWD:
            cudnn_mode = CUDNN_POINTWISE_SWISH_BWD;
            return cudnnStatus_t::CUDNN_STATUS_SUCCESS;
        case PointwiseMode_t::DIV:
            cudnn_mode = CUDNN_POINTWISE_DIV;
            return cudnnStatus_t::CUDNN_STATUS_SUCCESS;
        case PointwiseMode_t::ADD_SQUARE:
            cudnn_mode = CUDNN_POINTWISE_ADD_SQUARE;
            return cudnnStatus_t::CUDNN_STATUS_SUCCESS;
        case PointwiseMode_t::EXP:
            cudnn_mode = CUDNN_POINTWISE_EXP;
            return cudnnStatus_t::CUDNN_STATUS_SUCCESS;
        case PointwiseMode_t::SUB:
            cudnn_mode = CUDNN_POINTWISE_SUB;
            return cudnnStatus_t::CUDNN_STATUS_SUCCESS;
        case PointwiseMode_t::CMP_EQ:
            cudnn_mode = CUDNN_POINTWISE_CMP_EQ;
            return cudnnStatus_t::CUDNN_STATUS_SUCCESS;
        case PointwiseMode_t::CMP_NEQ:
            cudnn_mode = CUDNN_POINTWISE_CMP_NEQ;
            return cudnnStatus_t::CUDNN_STATUS_SUCCESS;
        case PointwiseMode_t::CMP_GT:
            cudnn_mode = CUDNN_POINTWISE_CMP_GT;
            return cudnnStatus_t::CUDNN_STATUS_SUCCESS;
        case PointwiseMode_t::CMP_GE:
            cudnn_mode = CUDNN_POINTWISE_CMP_GE;
            return cudnnStatus_t::CUDNN_STATUS_SUCCESS;
        case PointwiseMode_t::CMP_LT:
            cudnn_mode = CUDNN_POINTWISE_CMP_LT;
            return cudnnStatus_t::CUDNN_STATUS_SUCCESS;
        case PointwiseMode_t::CMP_LE:
            cudnn_mode = CUDNN_POINTWISE_CMP_LE;
            return cudnnStatus_t::CUDNN_STATUS_SUCCESS;
        case PointwiseMode_t::LOGICAL_AND:
            cudnn_mode = CUDNN_POINTWISE_LOGICAL_AND;
            return cudnnStatus_t::CUDNN_STATUS_SUCCESS;
        case PointwiseMode_t::LOGICAL_OR:
            cudnn_mode = CUDNN_POINTWISE_LOGICAL_OR;
            return cudnnStatus_t::CUDNN_STATUS_SUCCESS;
        case PointwiseMode_t::LOGICAL_NOT:
            cudnn_mode = CUDNN_POINTWISE_LOGICAL_NOT;
            return cudnnStatus_t::CUDNN_STATUS_SUCCESS;
        case PointwiseMode_t::LOG:
            cudnn_mode = CUDNN_POINTWISE_LOG;
            return cudnnStatus_t::CUDNN_STATUS_SUCCESS;
        case PointwiseMode_t::NEG:
            cudnn_mode = CUDNN_POINTWISE_NEG;
            return cudnnStatus_t::CUDNN_STATUS_SUCCESS;
        case PointwiseMode_t::MOD:
            cudnn_mode = CUDNN_POINTWISE_MOD;
            return cudnnStatus_t::CUDNN_STATUS_SUCCESS;
        case PointwiseMode_t::POW:
            cudnn_mode = CUDNN_POINTWISE_POW;
            return cudnnStatus_t::CUDNN_STATUS_SUCCESS;
        case PointwiseMode_t::ABS:
            cudnn_mode = CUDNN_POINTWISE_ABS;
            return cudnnStatus_t::CUDNN_STATUS_SUCCESS;
        case PointwiseMode_t::CEIL:
            cudnn_mode = CUDNN_POINTWISE_CEIL;
            return cudnnStatus_t::CUDNN_STATUS_SUCCESS;
        case PointwiseMode_t::COS:
            cudnn_mode = CUDNN_POINTWISE_COS;
            return cudnnStatus_t::CUDNN_STATUS_SUCCESS;
        case PointwiseMode_t::FLOOR:
            cudnn_mode = CUDNN_POINTWISE_FLOOR;
            return cudnnStatus_t::CUDNN_STATUS_SUCCESS;
        case PointwiseMode_t::RSQRT:
            cudnn_mode = CUDNN_POINTWISE_RSQRT;
            return cudnnStatus_t::CUDNN_STATUS_SUCCESS;
        case PointwiseMode_t::SIN:
            cudnn_mode = CUDNN_POINTWISE_SIN;
            return cudnnStatus_t::CUDNN_STATUS_SUCCESS;
        case PointwiseMode_t::TAN:
            cudnn_mode = CUDNN_POINTWISE_TAN;
            return cudnnStatus_t::CUDNN_STATUS_SUCCESS;
        case PointwiseMode_t::GEN_INDEX:
            cudnn_mode = CUDNN_POINTWISE_GEN_INDEX;
            return cudnnStatus_t::CUDNN_STATUS_SUCCESS;
        case PointwiseMode_t::BINARY_SELECT:
            cudnn_mode = CUDNN_POINTWISE_BINARY_SELECT;
            return cudnnStatus_t::CUDNN_STATUS_SUCCESS;
        case PointwiseMode_t::ERF:
            cudnn_mode = CUDNN_POINTWISE_ERF;
            return cudnnStatus_t::CUDNN_STATUS_SUCCESS;
        case PointwiseMode_t::IDENTITY:
            cudnn_mode = CUDNN_POINTWISE_IDENTITY;
            return cudnnStatus_t::CUDNN_STATUS_SUCCESS;
        case PointwiseMode_t::GELU_APPROX_TANH_BWD:
            cudnn_mode = CUDNN_POINTWISE_GELU_APPROX_TANH_BWD;
            return cudnnStatus_t::CUDNN_STATUS_SUCCESS;
        case PointwiseMode_t::GELU_APPROX_TANH_FWD:
            cudnn_mode = CUDNN_POINTWISE_GELU_APPROX_TANH_FWD;
            return cudnnStatus_t::CUDNN_STATUS_SUCCESS;
        case PointwiseMode_t::RECIPROCAL:
#if (CUDNN_VERSION >= 8900)
            NV_CUDNN_FE_DYNAMIC_CHECK_CUDNN_BACKEND_VERSION(8900, cudnnStatus_t::CUDNN_STATUS_INVALID_VALUE);
            cudnn_mode = CUDNN_POINTWISE_RECIPROCAL;
            return cudnnStatus_t::CUDNN_STATUS_SUCCESS;
#else
            return cudnnStatus_t::CUDNN_STATUS_INVALID_VALUE;
#endif

#ifndef NO_DEFAULT_IN_SWITCH
        default:
            return cudnnStatus_t::CUDNN_STATUS_INVALID_VALUE;
#endif
    }
    return cudnnStatus_t::CUDNN_STATUS_INVALID_VALUE;
}

static inline cudnnStatus_t
convert_to_cudnn_type(cudnn_frontend::NumericalNote_t const mode, cudnnBackendNumericalNote_t& cudnn_mode) {
    switch (mode) {
        case NumericalNote_t::TENSOR_CORE:
            cudnn_mode = CUDNN_NUMERICAL_NOTE_TENSOR_CORE;
            return cudnnStatus_t::CUDNN_STATUS_SUCCESS;
        case NumericalNote_t::DOWN_CONVERT_INPUTS:
            cudnn_mode = CUDNN_NUMERICAL_NOTE_DOWN_CONVERT_INPUTS;
            return cudnnStatus_t::CUDNN_STATUS_SUCCESS;
        case NumericalNote_t::REDUCED_PRECISION_REDUCTION:
            cudnn_mode = CUDNN_NUMERICAL_NOTE_REDUCED_PRECISION_REDUCTION;
            return cudnnStatus_t::CUDNN_STATUS_SUCCESS;
        case NumericalNote_t::FFT:
            cudnn_mode = CUDNN_NUMERICAL_NOTE_FFT;
            return cudnnStatus_t::CUDNN_STATUS_SUCCESS;
        case NumericalNote_t::NONDETERMINISTIC:
            cudnn_mode = CUDNN_NUMERICAL_NOTE_NONDETERMINISTIC;
            return cudnnStatus_t::CUDNN_STATUS_SUCCESS;
        case NumericalNote_t::WINOGRAD:
            cudnn_mode = CUDNN_NUMERICAL_NOTE_WINOGRAD;
            return cudnnStatus_t::CUDNN_STATUS_SUCCESS;
        case NumericalNote_t::WINOGRAD_TILE_4x4:
            cudnn_mode = CUDNN_NUMERICAL_NOTE_WINOGRAD_TILE_4x4;
            return cudnnStatus_t::CUDNN_STATUS_SUCCESS;
        case NumericalNote_t::WINOGRAD_TILE_6x6:
            cudnn_mode = CUDNN_NUMERICAL_NOTE_WINOGRAD_TILE_6x6;
            return cudnnStatus_t::CUDNN_STATUS_SUCCESS;
        case NumericalNote_t::WINOGRAD_TILE_13x13:
            cudnn_mode = CUDNN_NUMERICAL_NOTE_WINOGRAD_TILE_13x13;
            return cudnnStatus_t::CUDNN_STATUS_SUCCESS;
        case NumericalNote_t::STRICT_NAN_PROP:
#if (CUDNN_VERSION >= 90100)
            NV_CUDNN_FE_DYNAMIC_CHECK_CUDNN_BACKEND_VERSION(90100, cudnnStatus_t::CUDNN_STATUS_INVALID_VALUE);
            cudnn_mode = CUDNN_NUMERICAL_NOTE_STRICT_NAN_PROP;
            return cudnnStatus_t::CUDNN_STATUS_SUCCESS;
#else
            return cudnnStatus_t::CUDNN_STATUS_INVALID_VALUE;
#endif

#ifndef NO_DEFAULT_IN_SWITCH
        default:
            return cudnnStatus_t::CUDNN_STATUS_INVALID_VALUE;
#endif
    }
    return cudnnStatus_t::CUDNN_STATUS_INVALID_VALUE;
}

static inline cudnnStatus_t
convert_to_cudnn_type(cudnn_frontend::BehaviorNote_t const mode, cudnnBackendBehaviorNote_t& cudnn_mode) {
    switch (mode) {
        case BehaviorNote_t::RUNTIME_COMPILATION:
            cudnn_mode = CUDNN_BEHAVIOR_NOTE_RUNTIME_COMPILATION;
            return cudnnStatus_t::CUDNN_STATUS_SUCCESS;
        case BehaviorNote_t::REQUIRES_FILTER_INT8x32_REORDER:
            cudnn_mode = CUDNN_BEHAVIOR_NOTE_REQUIRES_FILTER_INT8x32_REORDER;
            return cudnnStatus_t::CUDNN_STATUS_SUCCESS;
        case BehaviorNote_t::REQUIRES_BIAS_INT8x32_REORDER:
            cudnn_mode = CUDNN_BEHAVIOR_NOTE_REQUIRES_BIAS_INT8x32_REORDER;
            return cudnnStatus_t::CUDNN_STATUS_SUCCESS;
        case BehaviorNote_t::SUPPORTS_CUDA_GRAPH_NATIVE_API:
#if (CUDNN_VERSION >= 90500)
            NV_CUDNN_FE_DYNAMIC_CHECK_CUDNN_BACKEND_VERSION(90500, cudnnStatus_t::CUDNN_STATUS_INVALID_VALUE);
            cudnn_mode = CUDNN_BEHAVIOR_NOTE_SUPPORTS_CUDA_GRAPH_NATIVE_API;
            return cudnnStatus_t::CUDNN_STATUS_SUCCESS;
#else
            return cudnnStatus_t::CUDNN_STATUS_INVALID_VALUE;
#endif
        case BehaviorNote_t::CUBLASLT_DEPENDENCY:
#if (CUDNN_VERSION >= 91500)
            NV_CUDNN_FE_DYNAMIC_CHECK_CUDNN_BACKEND_VERSION(91500, cudnnStatus_t::CUDNN_STATUS_INVALID_VALUE);
            cudnn_mode = CUDNN_BEHAVIOR_NOTE_CUBLASLT_DEPENDENCY;
            return cudnnStatus_t::CUDNN_STATUS_SUCCESS;
#else
            return cudnnStatus_t::CUDNN_STATUS_INVALID_VALUE;
#endif

#ifndef NO_DEFAULT_IN_SWITCH
        default:
            return cudnnStatus_t::CUDNN_STATUS_INVALID_VALUE;
#endif
    }
    return cudnnStatus_t::CUDNN_STATUS_INVALID_VALUE;
}

static inline cudnn_frontend::BehaviorNote_t
convert_from_cudnn_type(cudnnBackendBehaviorNote_t const cudnn_mode) {
    switch (cudnn_mode) {
        case CUDNN_BEHAVIOR_NOTE_RUNTIME_COMPILATION:
            return BehaviorNote_t::RUNTIME_COMPILATION;
        case CUDNN_BEHAVIOR_NOTE_REQUIRES_FILTER_INT8x32_REORDER:
            return BehaviorNote_t::REQUIRES_FILTER_INT8x32_REORDER;
        case CUDNN_BEHAVIOR_NOTE_REQUIRES_BIAS_INT8x32_REORDER:
            return BehaviorNote_t::REQUIRES_BIAS_INT8x32_REORDER;
#if (CUDNN_VERSION >= 90500)
        case CUDNN_BEHAVIOR_NOTE_SUPPORTS_CUDA_GRAPH_NATIVE_API:
            return BehaviorNote_t::SUPPORTS_CUDA_GRAPH_NATIVE_API;
#endif
#if (CUDNN_VERSION >= 91500)
        case CUDNN_BEHAVIOR_NOTE_CUBLASLT_DEPENDENCY:
            return BehaviorNote_t::CUBLASLT_DEPENDENCY;
#endif

#ifndef NO_DEFAULT_IN_SWITCH
        default:
            return BehaviorNote_t::NOT_SET;
            break;
#endif
    }
    return BehaviorNote_t::NOT_SET;
}

static inline cudnn_frontend::NumericalNote_t
convert_from_cudnn_type(cudnnBackendNumericalNote_t const cudnn_mode) {
    switch (cudnn_mode) {
        case CUDNN_NUMERICAL_NOTE_TENSOR_CORE:
            return NumericalNote_t::TENSOR_CORE;
        case CUDNN_NUMERICAL_NOTE_DOWN_CONVERT_INPUTS:
            return NumericalNote_t::DOWN_CONVERT_INPUTS;
        case CUDNN_NUMERICAL_NOTE_REDUCED_PRECISION_REDUCTION:
            return NumericalNote_t::REDUCED_PRECISION_REDUCTION;
        case CUDNN_NUMERICAL_NOTE_FFT:
            return NumericalNote_t::FFT;
        case CUDNN_NUMERICAL_NOTE_NONDETERMINISTIC:
            return NumericalNote_t::NONDETERMINISTIC;
        case CUDNN_NUMERICAL_NOTE_WINOGRAD:
            return NumericalNote_t::WINOGRAD;
        case CUDNN_NUMERICAL_NOTE_WINOGRAD_TILE_4x4:
            return NumericalNote_t::WINOGRAD_TILE_4x4;
        case CUDNN_NUMERICAL_NOTE_WINOGRAD_TILE_6x6:
            return NumericalNote_t::WINOGRAD_TILE_6x6;
        case CUDNN_NUMERICAL_NOTE_WINOGRAD_TILE_13x13:
            return NumericalNote_t::WINOGRAD_TILE_13x13;
#if (CUDNN_VERSION >= 90100)
        case CUDNN_NUMERICAL_NOTE_STRICT_NAN_PROP:
            return NumericalNote_t::STRICT_NAN_PROP;
#endif

#ifndef NO_DEFAULT_IN_SWITCH
        default:
            return NumericalNote_t::NOT_SET;
            break;
#endif
    }
    return NumericalNote_t::NOT_SET;
}

static inline cudnnStatus_t
convert_to_cudnn_type(cudnn_frontend::DescriptorType_t const mode, cudnnBackendDescriptorType_t& cudnn_mode) {
    switch (mode) {
        case DescriptorType_t::POINTWISE_DESCRIPTOR:
            cudnn_mode = CUDNN_BACKEND_POINTWISE_DESCRIPTOR;
            return cudnnStatus_t::CUDNN_STATUS_SUCCESS;
        case DescriptorType_t::CONVOLUTION_DESCRIPTOR:
            cudnn_mode = CUDNN_BACKEND_CONVOLUTION_DESCRIPTOR;
            return cudnnStatus_t::CUDNN_STATUS_SUCCESS;
        case DescriptorType_t::ENGINE_DESCRIPTOR:
            cudnn_mode = CUDNN_BACKEND_ENGINE_DESCRIPTOR;
            return cudnnStatus_t::CUDNN_STATUS_SUCCESS;
        case DescriptorType_t::ENGINECFG_DESCRIPTOR:
            cudnn_mode = CUDNN_BACKEND_ENGINECFG_DESCRIPTOR;
            return cudnnStatus_t::CUDNN_STATUS_SUCCESS;
        case DescriptorType_t::ENGINEHEUR_DESCRIPTOR:
            cudnn_mode = CUDNN_BACKEND_ENGINEHEUR_DESCRIPTOR;
            return cudnnStatus_t::CUDNN_STATUS_SUCCESS;
        case DescriptorType_t::EXECUTION_PLAN_DESCRIPTOR:
            cudnn_mode = CUDNN_BACKEND_EXECUTION_PLAN_DESCRIPTOR;
            return cudnnStatus_t::CUDNN_STATUS_SUCCESS;
        case DescriptorType_t::INTERMEDIATE_INFO_DESCRIPTOR:
            cudnn_mode = CUDNN_BACKEND_INTERMEDIATE_INFO_DESCRIPTOR;
            return cudnnStatus_t::CUDNN_STATUS_SUCCESS;
        case DescriptorType_t::KNOB_CHOICE_DESCRIPTOR:
            cudnn_mode = CUDNN_BACKEND_KNOB_CHOICE_DESCRIPTOR;
            return cudnnStatus_t::CUDNN_STATUS_SUCCESS;
        case DescriptorType_t::KNOB_INFO_DESCRIPTOR:
            cudnn_mode = CUDNN_BACKEND_KNOB_INFO_DESCRIPTOR;
            return cudnnStatus_t::CUDNN_STATUS_SUCCESS;
        case DescriptorType_t::LAYOUT_INFO_DESCRIPTOR:
            cudnn_mode = CUDNN_BACKEND_LAYOUT_INFO_DESCRIPTOR;
            return cudnnStatus_t::CUDNN_STATUS_SUCCESS;
        case DescriptorType_t::OPERATION_CONVOLUTION_FORWARD_DESCRIPTOR:
            cudnn_mode = CUDNN_BACKEND_OPERATION_CONVOLUTION_FORWARD_DESCRIPTOR;
            return cudnnStatus_t::CUDNN_STATUS_SUCCESS;
        case DescriptorType_t::OPERATION_CONVOLUTION_BACKWARD_FILTER_DESCRIPTOR:
            cudnn_mode = CUDNN_BACKEND_OPERATION_CONVOLUTION_BACKWARD_FILTER_DESCRIPTOR;
            return cudnnStatus_t::CUDNN_STATUS_SUCCESS;
        case DescriptorType_t::OPERATION_CONVOLUTION_BACKWARD_DATA_DESCRIPTOR:
            cudnn_mode = CUDNN_BACKEND_OPERATION_CONVOLUTION_BACKWARD_DATA_DESCRIPTOR;
            return cudnnStatus_t::CUDNN_STATUS_SUCCESS;
        case DescriptorType_t::OPERATION_POINTWISE_DESCRIPTOR:
            cudnn_mode = CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR;
            return cudnnStatus_t::CUDNN_STATUS_SUCCESS;
        case DescriptorType_t::OPERATION_GEN_STATS_DESCRIPTOR:
            cudnn_mode = CUDNN_BACKEND_OPERATION_GEN_STATS_DESCRIPTOR;
            return cudnnStatus_t::CUDNN_STATUS_SUCCESS;
        case DescriptorType_t::OPERATIONGRAPH_DESCRIPTOR:
            cudnn_mode = CUDNN_BACKEND_OPERATIONGRAPH_DESCRIPTOR;
            return cudnnStatus_t::CUDNN_STATUS_SUCCESS;
        case DescriptorType_t::VARIANT_PACK_DESCRIPTOR:
            cudnn_mode = CUDNN_BACKEND_VARIANT_PACK_DESCRIPTOR;
            return cudnnStatus_t::CUDNN_STATUS_SUCCESS;
        case DescriptorType_t::TENSOR_DESCRIPTOR:
            cudnn_mode = CUDNN_BACKEND_TENSOR_DESCRIPTOR;
            return cudnnStatus_t::CUDNN_STATUS_SUCCESS;
        case DescriptorType_t::MATMUL_DESCRIPTOR:
            cudnn_mode = CUDNN_BACKEND_MATMUL_DESCRIPTOR;
            return cudnnStatus_t::CUDNN_STATUS_SUCCESS;
        case DescriptorType_t::OPERATION_MATMUL_DESCRIPTOR:
            cudnn_mode = CUDNN_BACKEND_OPERATION_MATMUL_DESCRIPTOR;
            return cudnnStatus_t::CUDNN_STATUS_SUCCESS;
        case DescriptorType_t::OPERATION_BN_FINALIZE_STATISTICS_DESCRIPTOR:
            cudnn_mode = CUDNN_BACKEND_OPERATION_BN_FINALIZE_STATISTICS_DESCRIPTOR;
            return cudnnStatus_t::CUDNN_STATUS_SUCCESS;
        case DescriptorType_t::REDUCTION_DESCRIPTOR:
            cudnn_mode = CUDNN_BACKEND_REDUCTION_DESCRIPTOR;
            return cudnnStatus_t::CUDNN_STATUS_SUCCESS;
        case DescriptorType_t::OPERATION_REDUCTION_DESCRIPTOR:
            cudnn_mode = CUDNN_BACKEND_OPERATION_REDUCTION_DESCRIPTOR;
            return cudnnStatus_t::CUDNN_STATUS_SUCCESS;
        case DescriptorType_t::OPERATION_BN_BWD_WEIGHTS_DESCRIPTOR:
            cudnn_mode = CUDNN_BACKEND_OPERATION_BN_BWD_WEIGHTS_DESCRIPTOR;
            return cudnnStatus_t::CUDNN_STATUS_SUCCESS;
        case DescriptorType_t::RESAMPLE_DESCRIPTOR:
            cudnn_mode = CUDNN_BACKEND_RESAMPLE_DESCRIPTOR;
            return cudnnStatus_t::CUDNN_STATUS_SUCCESS;
        case DescriptorType_t::OPERATION_RESAMPLE_FWD_DESCRIPTOR:
            cudnn_mode = CUDNN_BACKEND_OPERATION_RESAMPLE_FWD_DESCRIPTOR;
            return cudnnStatus_t::CUDNN_STATUS_SUCCESS;
        case DescriptorType_t::OPERATION_RESAMPLE_BWD_DESCRIPTOR:
#if (CUDNN_VERSION >= 8600)
            NV_CUDNN_FE_DYNAMIC_CHECK_CUDNN_BACKEND_VERSION(8600, cudnnStatus_t::CUDNN_STATUS_INVALID_VALUE);
            cudnn_mode = CUDNN_BACKEND_OPERATION_RESAMPLE_BWD_DESCRIPTOR;
            return cudnnStatus_t::CUDNN_STATUS_SUCCESS;
#else
            return cudnnStatus_t::CUDNN_STATUS_INVALID_VALUE;
#endif
        case DescriptorType_t::OPERATION_CONCAT_DESCRIPTOR:
            cudnn_mode = CUDNN_BACKEND_OPERATION_CONCAT_DESCRIPTOR;
            return cudnnStatus_t::CUDNN_STATUS_SUCCESS;
        case DescriptorType_t::OPERATION_SIGNAL_DESCRIPTOR:
            cudnn_mode = CUDNN_BACKEND_OPERATION_SIGNAL_DESCRIPTOR;
            return cudnnStatus_t::CUDNN_STATUS_SUCCESS;
        case DescriptorType_t::OPERATION_NORM_FORWARD_DESCRIPTOR:
            cudnn_mode = CUDNN_BACKEND_OPERATION_NORM_FORWARD_DESCRIPTOR;
            return cudnnStatus_t::CUDNN_STATUS_SUCCESS;
        case DescriptorType_t::OPERATION_NORM_BACKWARD_DESCRIPTOR:
            cudnn_mode = CUDNN_BACKEND_OPERATION_NORM_BACKWARD_DESCRIPTOR;
            return cudnnStatus_t::CUDNN_STATUS_SUCCESS;
        case DescriptorType_t::OPERATION_RESHAPE_DESCRIPTOR:
#if (CUDNN_VERSION >= 8700)
            NV_CUDNN_FE_DYNAMIC_CHECK_CUDNN_BACKEND_VERSION(8700, cudnnStatus_t::CUDNN_STATUS_INVALID_VALUE);
            cudnn_mode = CUDNN_BACKEND_OPERATION_RESHAPE_DESCRIPTOR;
            return cudnnStatus_t::CUDNN_STATUS_SUCCESS;
#else
            return cudnnStatus_t::CUDNN_STATUS_INVALID_VALUE;
#endif

        case DescriptorType_t::RNG_DESCRIPTOR:
#if (CUDNN_VERSION >= 8700)
            NV_CUDNN_FE_DYNAMIC_CHECK_CUDNN_BACKEND_VERSION(8700, cudnnStatus_t::CUDNN_STATUS_INVALID_VALUE);
            cudnn_mode = CUDNN_BACKEND_RNG_DESCRIPTOR;
            return cudnnStatus_t::CUDNN_STATUS_SUCCESS;
#else
            return cudnnStatus_t::CUDNN_STATUS_INVALID_VALUE;
#endif

        case DescriptorType_t::OPERATION_RNG_DESCRIPTOR:
#if (CUDNN_VERSION >= 8700)
            NV_CUDNN_FE_DYNAMIC_CHECK_CUDNN_BACKEND_VERSION(8700, cudnnStatus_t::CUDNN_STATUS_INVALID_VALUE);
            cudnn_mode = CUDNN_BACKEND_OPERATION_RNG_DESCRIPTOR;
            return cudnnStatus_t::CUDNN_STATUS_SUCCESS;
#else
            return cudnnStatus_t::CUDNN_STATUS_INVALID_VALUE;
#endif

        case DescriptorType_t::OPERATION_PAGED_CACHE_LOAD_DESCRIPTOR:
#if (CUDNN_VERSION >= 90500)
            NV_CUDNN_FE_DYNAMIC_CHECK_CUDNN_BACKEND_VERSION(90500, cudnnStatus_t::CUDNN_STATUS_INVALID_VALUE);
            cudnn_mode = CUDNN_BACKEND_OPERATION_PAGED_CACHE_LOAD_DESCRIPTOR;
            return cudnnStatus_t::CUDNN_STATUS_SUCCESS;
#endif
        case DescriptorType_t::OPERATION_BLOCK_SCALE_QUANTIZE_DESCRIPTOR:
#if (CUDNN_VERSION >= 90700)  // TODO: v9.99 is new feature branch; switch to release branch when ready
            NV_CUDNN_FE_DYNAMIC_CHECK_CUDNN_BACKEND_VERSION(90700, cudnnStatus_t::CUDNN_STATUS_INVALID_VALUE);
            cudnn_mode = CUDNN_BACKEND_OPERATION_BLOCK_SCALE_QUANTIZE_DESCRIPTOR;
            return cudnnStatus_t::CUDNN_STATUS_SUCCESS;
#else
            return cudnnStatus_t::CUDNN_STATUS_INVALID_VALUE;
#endif
        case DescriptorType_t::OPERATION_BLOCK_SCALE_DEQUANTIZE_DESCRIPTOR:
#if (CUDNN_VERSION >= 90700)  // TODO: v9.99 is new feature branch; switch to release branch when ready
            NV_CUDNN_FE_DYNAMIC_CHECK_CUDNN_BACKEND_VERSION(90700, cudnnStatus_t::CUDNN_STATUS_INVALID_VALUE);
            cudnn_mode = CUDNN_BACKEND_OPERATION_BLOCK_SCALE_DEQUANTIZE_DESCRIPTOR;
            return cudnnStatus_t::CUDNN_STATUS_SUCCESS;
#else
            return cudnnStatus_t::CUDNN_STATUS_INVALID_VALUE;
#endif
        case DescriptorType_t::OPERATION_CONCATENATE_DESCRIPTOR:
#if (CUDNN_VERSION >= 90700)
            NV_CUDNN_FE_DYNAMIC_CHECK_CUDNN_BACKEND_VERSION(90700, cudnnStatus_t::CUDNN_STATUS_INVALID_VALUE);
            cudnn_mode = CUDNN_BACKEND_OPERATION_CONCAT_DESCRIPTOR;
            return cudnnStatus_t::CUDNN_STATUS_SUCCESS;
#else
            return cudnnStatus_t::CUDNN_STATUS_INVALID_VALUE;
#endif
        case DescriptorType_t::OPERATION_MOE_GROUPED_MATMUL_DESCRIPTOR:
#if (CUDNN_VERSION >= 91500)
            NV_CUDNN_FE_DYNAMIC_CHECK_CUDNN_BACKEND_VERSION(91500, cudnnStatus_t::CUDNN_STATUS_INVALID_VALUE);
            cudnn_mode = CUDNN_BACKEND_OPERATION_MOE_GROUPED_MATMUL_DESCRIPTOR;
            return cudnnStatus_t::CUDNN_STATUS_SUCCESS;
#else
            return cudnnStatus_t::CUDNN_STATUS_INVALID_VALUE;
#endif
        case DescriptorType_t::OPERATION_MOE_GROUPED_MATMUL_BWD_DESCRIPTOR:
#if (CUDNN_VERSION >= 92200) && (CUDNN_VERSION < 99900)
            NV_CUDNN_FE_DYNAMIC_CHECK_CUDNN_BACKEND_VERSION(92200, cudnnStatus_t::CUDNN_STATUS_INVALID_VALUE);
            cudnn_mode = CUDNN_BACKEND_OPERATION_MOE_GROUPED_MATMUL_BWD_DESCRIPTOR;
            return cudnnStatus_t::CUDNN_STATUS_SUCCESS;
#else
            return cudnnStatus_t::CUDNN_STATUS_INVALID_VALUE;
#endif
        case DescriptorType_t::OPERATION_TRANSPOSE_DESCRIPTOR:
#if (CUDNN_VERSION >= 92200) && (CUDNN_VERSION < 99900)
            NV_CUDNN_FE_DYNAMIC_CHECK_CUDNN_BACKEND_VERSION(92200, cudnnStatus_t::CUDNN_STATUS_INVALID_VALUE);
            cudnn_mode = CUDNN_BACKEND_OPERATION_TRANSPOSE_DESCRIPTOR;
            return cudnnStatus_t::CUDNN_STATUS_SUCCESS;
#else
            return cudnnStatus_t::CUDNN_STATUS_INVALID_VALUE;
#endif
        case DescriptorType_t::OPERATION_SLICE_DESCRIPTOR:
#if (CUDNN_VERSION >= 92200) && (CUDNN_VERSION < 99900)
            NV_CUDNN_FE_DYNAMIC_CHECK_CUDNN_BACKEND_VERSION(92200, cudnnStatus_t::CUDNN_STATUS_INVALID_VALUE);
            cudnn_mode = CUDNN_BACKEND_OPERATION_SLICE_DESCRIPTOR;
            return cudnnStatus_t::CUDNN_STATUS_SUCCESS;
#else
            return cudnnStatus_t::CUDNN_STATUS_INVALID_VALUE;
#endif

#ifndef NO_DEFAULT_IN_SWITCH
        default:
            return cudnnStatus_t::CUDNN_STATUS_INVALID_VALUE;
#endif
    }
    return cudnnStatus_t::CUDNN_STATUS_INVALID_VALUE;
}

static inline cudnnStatus_t
convert_to_cudnn_type(cudnn_frontend::ResampleMode_t const mode, cudnnResampleMode_t& cudnn_mode) {
    switch (mode) {
#if (CUDNN_VERSION >= 8600)
        case cudnn_frontend::ResampleMode_t::AVGPOOL_EXCLUDE_PADDING:
            cudnn_mode = CUDNN_RESAMPLE_AVGPOOL_EXCLUDE_PADDING;
            return cudnnStatus_t::CUDNN_STATUS_SUCCESS;
        case cudnn_frontend::ResampleMode_t::AVGPOOL_INCLUDE_PADDING:
            cudnn_mode = CUDNN_RESAMPLE_AVGPOOL_INCLUDE_PADDING;
            return cudnnStatus_t::CUDNN_STATUS_SUCCESS;
#else
        case cudnn_frontend::ResampleMode_t::AVGPOOL_INCLUDE_PADDING:
            cudnn_mode = CUDNN_RESAMPLE_AVGPOOL;
            return cudnnStatus_t::CUDNN_STATUS_SUCCESS;
#endif
        case cudnn_frontend::ResampleMode_t::BILINEAR:
            cudnn_mode = CUDNN_RESAMPLE_BILINEAR;
            return cudnnStatus_t::CUDNN_STATUS_SUCCESS;
        case cudnn_frontend::ResampleMode_t::NEAREST:
            cudnn_mode = CUDNN_RESAMPLE_NEAREST;
            return cudnnStatus_t::CUDNN_STATUS_SUCCESS;
        case cudnn_frontend::ResampleMode_t::MAXPOOL:
            cudnn_mode = CUDNN_RESAMPLE_MAXPOOL;
            return cudnnStatus_t::CUDNN_STATUS_SUCCESS;
#ifndef NO_DEFAULT_IN_SWITCH
        default:
            return cudnnStatus_t::CUDNN_STATUS_INVALID_VALUE;
#endif
    }
    return cudnnStatus_t::CUDNN_STATUS_INVALID_VALUE;
}

static inline cudnnStatus_t
convert_to_cudnn_type(cudnn_frontend::PaddingMode_t const mode, cudnnPaddingMode_t& cudnn_mode) {
    switch (mode) {
        case cudnn_frontend::PaddingMode_t::ZERO_PAD:
            cudnn_mode = CUDNN_ZERO_PAD;
            return cudnnStatus_t::CUDNN_STATUS_SUCCESS;
        case cudnn_frontend::PaddingMode_t::NEG_INF_PAD:
            cudnn_mode = CUDNN_NEG_INF_PAD;
            return cudnnStatus_t::CUDNN_STATUS_SUCCESS;
        case cudnn_frontend::PaddingMode_t::EDGE_VAL_PAD:
            cudnn_mode = CUDNN_EDGE_VAL_PAD;
            return cudnnStatus_t::CUDNN_STATUS_SUCCESS;
#ifndef NO_DEFAULT_IN_SWITCH
        default:
            return cudnnStatus_t::CUDNN_STATUS_INVALID_VALUE;
#endif
    }
    return cudnnStatus_t::CUDNN_STATUS_INVALID_VALUE;
}

static inline cudnnStatus_t
convert_to_cudnn_type(cudnn_frontend::NormMode_t const mode, cudnnBackendNormMode_t& cudnn_mode) {
    switch (mode) {
        case NormMode_t::LAYER_NORM:
            cudnn_mode = CUDNN_LAYER_NORM;
            return cudnnStatus_t::CUDNN_STATUS_SUCCESS;
        case NormMode_t::INSTANCE_NORM:
            cudnn_mode = CUDNN_INSTANCE_NORM;
            return cudnnStatus_t::CUDNN_STATUS_SUCCESS;
        case NormMode_t::BATCH_NORM:
            cudnn_mode = CUDNN_BATCH_NORM;
            return cudnnStatus_t::CUDNN_STATUS_SUCCESS;
        case NormMode_t::GROUP_NORM:
            cudnn_mode = CUDNN_GROUP_NORM;
            return cudnnStatus_t::CUDNN_STATUS_SUCCESS;

#if (CUDNN_VERSION >= 8906)
        case NormMode_t::RMS_NORM:
            NV_CUDNN_FE_DYNAMIC_CHECK_CUDNN_BACKEND_VERSION(8906, cudnnStatus_t::CUDNN_STATUS_INVALID_VALUE);
            cudnn_mode = CUDNN_RMS_NORM;
            return cudnnStatus_t::CUDNN_STATUS_SUCCESS;
#endif
#if (CUDNN_VERSION >= 90900)
        case NormMode_t::ADA_LAYER_NORM:
            NV_CUDNN_FE_DYNAMIC_CHECK_CUDNN_BACKEND_VERSION(90900, cudnnStatus_t::CUDNN_STATUS_INVALID_VALUE);
            cudnn_mode = CUDNN_ADA_LAYER_NORM;
            return cudnnStatus_t::CUDNN_STATUS_SUCCESS;
#endif
#ifndef NO_DEFAULT_IN_SWITCH
        default:
            return cudnnStatus_t::CUDNN_STATUS_INVALID_VALUE;
#endif
    }
    return cudnnStatus_t::CUDNN_STATUS_INVALID_VALUE;
}

static inline cudnnStatus_t
convert_to_cudnn_type(cudnn_frontend::NormFwdPhase_t const mode, cudnnBackendNormFwdPhase_t& cudnn_mode) {
    switch (mode) {
        case NormFwdPhase_t::INFERENCE:
            cudnn_mode = CUDNN_NORM_FWD_INFERENCE;
            return cudnnStatus_t::CUDNN_STATUS_SUCCESS;
        case NormFwdPhase_t::TRAINING:
            cudnn_mode = CUDNN_NORM_FWD_TRAINING;
            return cudnnStatus_t::CUDNN_STATUS_SUCCESS;

#ifndef NO_DEFAULT_IN_SWITCH
        default:
            return cudnnStatus_t::CUDNN_STATUS_INVALID_VALUE;
#endif
    }
    return cudnnStatus_t::CUDNN_STATUS_INVALID_VALUE;
}

#if (CUDNN_VERSION >= 92200)
static inline cudnnStatus_t
convert_to_cudnn_type(cudnn_frontend::ReshapeMode_t const mode, cudnnBackendReshapeMode_t& cudnn_mode) {
    switch (mode) {
        case ReshapeMode_t::VIEW_ONLY:
            cudnn_mode = CUDNN_RESHAPE_VIEW_ONLY;
            return cudnnStatus_t::CUDNN_STATUS_SUCCESS;
        case ReshapeMode_t::LOGICAL:
            cudnn_mode = CUDNN_RESHAPE_LOGICAL;
            return cudnnStatus_t::CUDNN_STATUS_SUCCESS;

#ifndef NO_DEFAULT_IN_SWITCH
        default:
            return cudnnStatus_t::CUDNN_STATUS_INVALID_VALUE;
#endif
    }
    return cudnnStatus_t::CUDNN_STATUS_INVALID_VALUE;
}
#endif

// To be deprecated. Only exists as setResampleMode(cudnnPaddingMode_t) requires it.
static inline void
convert_from_cudnn_type(cudnnPaddingMode_t const cudnn_mode, cudnn_frontend::PaddingMode_t& mode) {
    mode = cudnn_frontend::PaddingMode_t::NOT_SET;
    switch (cudnn_mode) {
        case CUDNN_EDGE_VAL_PAD:
            mode = cudnn_frontend::PaddingMode_t::EDGE_VAL_PAD;
            break;
        case CUDNN_NEG_INF_PAD:
            mode = cudnn_frontend::PaddingMode_t::NEG_INF_PAD;
            break;
        case CUDNN_ZERO_PAD:
            mode = cudnn_frontend::PaddingMode_t::ZERO_PAD;
            break;
#ifndef NO_DEFAULT_IN_SWITCH
        default:
            break;
#endif
    }
}

static inline cudnn_frontend::ConvolutionMode_t
convert_from_cudnn_type(cudnnConvolutionMode_t const cudnn_mode) {
    switch (cudnn_mode) {
        case CUDNN_CONVOLUTION:
            return cudnn_frontend::ConvolutionMode_t::CONVOLUTION;
        case CUDNN_CROSS_CORRELATION:
            return cudnn_frontend::ConvolutionMode_t::CROSS_CORRELATION;
#ifndef NO_DEFAULT_IN_SWITCH
        default:
            return cudnn_frontend::ConvolutionMode_t::NOT_SET;
#endif
    }
    return cudnn_frontend::ConvolutionMode_t::NOT_SET;
}

static inline cudnnConvolutionMode_t
convert_to_cudnn_type(cudnn_frontend::ConvolutionMode_t const cudnn_mode) {
    switch (cudnn_mode) {
        case cudnn_frontend::ConvolutionMode_t::CONVOLUTION:
            return CUDNN_CONVOLUTION;
        case cudnn_frontend::ConvolutionMode_t::CROSS_CORRELATION:
            return CUDNN_CROSS_CORRELATION;
        case cudnn_frontend::ConvolutionMode_t::NOT_SET:
            return CUDNN_CROSS_CORRELATION;
    }
    return CUDNN_CROSS_CORRELATION;
}
// To be deprecated. Only exists as setResampleMode(cudnnResampleMode_t) requires it.
static inline void
convert_from_cudnn_type(cudnnResampleMode_t const cudnn_mode, cudnn_frontend::ResampleMode_t& mode) {
    mode = cudnn_frontend::ResampleMode_t::NOT_SET;
    switch (cudnn_mode) {
#if (CUDNN_VERSION >= 8600)
        case CUDNN_RESAMPLE_AVGPOOL_EXCLUDE_PADDING:
            mode = cudnn_frontend::ResampleMode_t::AVGPOOL_EXCLUDE_PADDING;
            break;
        case CUDNN_RESAMPLE_AVGPOOL_INCLUDE_PADDING:
            mode = cudnn_frontend::ResampleMode_t::AVGPOOL_INCLUDE_PADDING;
            break;
#else
        case CUDNN_RESAMPLE_AVGPOOL:
            mode = cudnn_frontend::ResampleMode_t::AVGPOOL_INCLUDE_PADDING;
            break;
#endif
        case CUDNN_RESAMPLE_BILINEAR:
            mode = cudnn_frontend::ResampleMode_t::BILINEAR;
            break;
        case CUDNN_RESAMPLE_NEAREST:
            mode = cudnn_frontend::ResampleMode_t::NEAREST;
            break;
        case CUDNN_RESAMPLE_MAXPOOL:
            mode = cudnn_frontend::ResampleMode_t::MAXPOOL;
            break;
#ifndef NO_DEFAULT_IN_SWITCH
        default:
            break;
#endif
    }
}

// To be deprecated. Only exists as setNormalizationMode(cudnnBackendNormMode_t) requires it.
static inline void
convert_from_cudnn_type(cudnnBackendNormMode_t const cudnn_mode, cudnn_frontend::NormMode_t& mode) {
    mode = NormMode_t::NOT_SET;
    switch (cudnn_mode) {
        case CUDNN_LAYER_NORM:
            mode = NormMode_t::LAYER_NORM;
            break;
        case CUDNN_INSTANCE_NORM:
            mode = NormMode_t::INSTANCE_NORM;
            break;
        case CUDNN_BATCH_NORM:
            mode = NormMode_t::BATCH_NORM;
            break;
        case CUDNN_GROUP_NORM:
            mode = NormMode_t::GROUP_NORM;
            break;

#if (CUDNN_VERSION >= 8906)
        case CUDNN_RMS_NORM:
            mode = NormMode_t::RMS_NORM;
            break;
#endif

#ifndef NO_DEFAULT_IN_SWITCH
        default:
            break;
#endif
    }
}

// To be deprecated. Only exists as setNormFwdPhase(cudnnBackendNormFwdPhase_t) requires it.
static inline void
convert_from_cudnn_type(cudnnBackendNormFwdPhase_t const cudnn_mode, cudnn_frontend::NormFwdPhase_t& mode) {
    mode = NormFwdPhase_t::NOT_SET;
    switch (cudnn_mode) {
        case CUDNN_NORM_FWD_INFERENCE:
            mode = NormFwdPhase_t::INFERENCE;
            break;
        case CUDNN_NORM_FWD_TRAINING:
            mode = NormFwdPhase_t::TRAINING;
            break;

#ifndef NO_DEFAULT_IN_SWITCH
        default:
            break;
#endif
    }
}

#if (CUDNN_VERSION >= 92200)
// To be deprecated. Only exists as setReshapeMode(cudnnBackendReshapeMode_t) requires it.
static inline void
convert_from_cudnn_type(cudnnBackendReshapeMode_t const cudnn_mode, cudnn_frontend::ReshapeMode_t& mode) {
    mode = ReshapeMode_t::NOT_SET;
    switch (cudnn_mode) {
        case CUDNN_RESHAPE_VIEW_ONLY:
            mode = ReshapeMode_t::VIEW_ONLY;
            break;
        case CUDNN_RESHAPE_LOGICAL:
            mode = ReshapeMode_t::LOGICAL;
            break;

#ifndef NO_DEFAULT_IN_SWITCH
        default:
            break;
#endif
    }
}
#endif

static inline cudnnStatus_t
convert_to_cudnn_type(cudnn_frontend::TensorReordering_t const mode, cudnnBackendTensorReordering_t& cudnn_mode) {
    switch (mode) {
        case cudnn_frontend::TensorReordering_t::NONE:
            cudnn_mode = CUDNN_TENSOR_REORDERING_NONE;
            return cudnnStatus_t::CUDNN_STATUS_SUCCESS;
        case cudnn_frontend::TensorReordering_t::INT8x32:
            cudnn_mode = CUDNN_TENSOR_REORDERING_INT8x32;
            return cudnnStatus_t::CUDNN_STATUS_SUCCESS;
        case cudnn_frontend::TensorReordering_t::F16x16:
#if CUDNN_VERSION >= 8800
#if defined NV_CUDNN_FRONTEND_USE_DYNAMIC_LOADING
            if (get_backend_version() >= 8800) {
                cudnn_mode = CUDNN_TENSOR_REORDERING_F16x16;
                return cudnnStatus_t::CUDNN_STATUS_SUCCESS;
            } else if (get_backend_version() >= 8700) {
                cudnn_mode = CUDNN_TENSOR_REORDERING_NONE;
                return cudnnStatus_t::CUDNN_STATUS_SUCCESS;
            } else {
                return cudnnStatus_t::CUDNN_STATUS_INVALID_VALUE;
            }
#endif
            cudnn_mode = CUDNN_TENSOR_REORDERING_F16x16;
            return cudnnStatus_t::CUDNN_STATUS_SUCCESS;
#elif CUDNN_VERSION >= 8700
            cudnn_mode = CUDNN_TENSOR_REORDERING_NONE;
            return cudnnStatus_t::CUDNN_STATUS_SUCCESS;
#else
            return cudnnStatus_t::CUDNN_STATUS_INVALID_VALUE;
#endif
        case cudnn_frontend::TensorReordering_t::F8_128x4:
#if CUDNN_VERSION >= 90700
            cudnn_mode = CUDNN_TENSOR_REORDERING_F8_128x4;
            return cudnnStatus_t::CUDNN_STATUS_SUCCESS;
#endif
#ifndef NO_DEFAULT_IN_SWITCH
        default:
            return cudnnStatus_t::CUDNN_STATUS_INVALID_VALUE;
#endif
    }
    return cudnnStatus_t::CUDNN_STATUS_INVALID_VALUE;
}

// To be deprecated. Only exists as setReorderType(cudnnBackendTensorReordering_t) requires it.
static inline void
convert_from_cudnn_type(cudnnBackendTensorReordering_t const cudnn_mode, cudnn_frontend::TensorReordering_t& mode) {
    mode = cudnn_frontend::TensorReordering_t::NONE;
    switch (cudnn_mode) {
        case CUDNN_TENSOR_REORDERING_INT8x32:
            mode = cudnn_frontend::TensorReordering_t::INT8x32;
            break;
#if CUDNN_VERSION >= 8800
        case CUDNN_TENSOR_REORDERING_F16x16:
            mode = cudnn_frontend::TensorReordering_t::F16x16;
            break;
#endif
#if CUDNN_VERSION >= 90700
        case CUDNN_TENSOR_REORDERING_F8_128x4:
            mode = cudnn_frontend::TensorReordering_t::F8_128x4;
            break;
#endif
#ifndef NO_DEFAULT_IN_SWITCH
        default:
            break;
#endif
    }
}

// To be deprecated. Only exists as OperationBuilder_v8(::cudnnBackendDescriptorType_t mode) requires it.
static inline cudnn_frontend::DescriptorType_t
convert_from_cudnn_type(cudnnBackendDescriptorType_t const cudnn_mode) {
    switch (cudnn_mode) {
        case CUDNN_BACKEND_POINTWISE_DESCRIPTOR:
            return DescriptorType_t::POINTWISE_DESCRIPTOR;
        case CUDNN_BACKEND_CONVOLUTION_DESCRIPTOR:
            return DescriptorType_t::CONVOLUTION_DESCRIPTOR;
        case CUDNN_BACKEND_ENGINE_DESCRIPTOR:
            return DescriptorType_t::ENGINE_DESCRIPTOR;
        case CUDNN_BACKEND_ENGINECFG_DESCRIPTOR:
            return DescriptorType_t::ENGINECFG_DESCRIPTOR;
        case CUDNN_BACKEND_ENGINEHEUR_DESCRIPTOR:
            return DescriptorType_t::ENGINEHEUR_DESCRIPTOR;
        case CUDNN_BACKEND_EXECUTION_PLAN_DESCRIPTOR:
            return DescriptorType_t::EXECUTION_PLAN_DESCRIPTOR;
        case CUDNN_BACKEND_INTERMEDIATE_INFO_DESCRIPTOR:
            return DescriptorType_t::INTERMEDIATE_INFO_DESCRIPTOR;
        case CUDNN_BACKEND_KNOB_CHOICE_DESCRIPTOR:
            return DescriptorType_t::KNOB_CHOICE_DESCRIPTOR;
        case CUDNN_BACKEND_KNOB_INFO_DESCRIPTOR:
            return DescriptorType_t::KNOB_INFO_DESCRIPTOR;
        case CUDNN_BACKEND_LAYOUT_INFO_DESCRIPTOR:
            return DescriptorType_t::LAYOUT_INFO_DESCRIPTOR;
        case CUDNN_BACKEND_OPERATION_CONVOLUTION_FORWARD_DESCRIPTOR:
            return DescriptorType_t::OPERATION_CONVOLUTION_FORWARD_DESCRIPTOR;
        case CUDNN_BACKEND_OPERATION_CONVOLUTION_BACKWARD_FILTER_DESCRIPTOR:
            return DescriptorType_t::OPERATION_CONVOLUTION_BACKWARD_FILTER_DESCRIPTOR;
        case CUDNN_BACKEND_OPERATION_CONVOLUTION_BACKWARD_DATA_DESCRIPTOR:
            return DescriptorType_t::OPERATION_CONVOLUTION_BACKWARD_DATA_DESCRIPTOR;
        case CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR:
            return DescriptorType_t::OPERATION_POINTWISE_DESCRIPTOR;
        case CUDNN_BACKEND_OPERATION_GEN_STATS_DESCRIPTOR:
            return DescriptorType_t::OPERATION_GEN_STATS_DESCRIPTOR;
        case CUDNN_BACKEND_OPERATIONGRAPH_DESCRIPTOR:
            return DescriptorType_t::OPERATIONGRAPH_DESCRIPTOR;
        case CUDNN_BACKEND_VARIANT_PACK_DESCRIPTOR:
            return DescriptorType_t::VARIANT_PACK_DESCRIPTOR;
        case CUDNN_BACKEND_TENSOR_DESCRIPTOR:
            return DescriptorType_t::TENSOR_DESCRIPTOR;
        case CUDNN_BACKEND_MATMUL_DESCRIPTOR:
            return DescriptorType_t::MATMUL_DESCRIPTOR;
        case CUDNN_BACKEND_OPERATION_MATMUL_DESCRIPTOR:
            return DescriptorType_t::OPERATION_MATMUL_DESCRIPTOR;
        case CUDNN_BACKEND_OPERATION_BN_FINALIZE_STATISTICS_DESCRIPTOR:
            return DescriptorType_t::OPERATION_BN_FINALIZE_STATISTICS_DESCRIPTOR;
        case CUDNN_BACKEND_REDUCTION_DESCRIPTOR:
            return DescriptorType_t::REDUCTION_DESCRIPTOR;
        case CUDNN_BACKEND_OPERATION_REDUCTION_DESCRIPTOR:
            return DescriptorType_t::OPERATION_REDUCTION_DESCRIPTOR;
        case CUDNN_BACKEND_OPERATION_BN_BWD_WEIGHTS_DESCRIPTOR:
            return DescriptorType_t::OPERATION_BN_BWD_WEIGHTS_DESCRIPTOR;
        case CUDNN_BACKEND_RESAMPLE_DESCRIPTOR:
            return DescriptorType_t::RESAMPLE_DESCRIPTOR;
        case CUDNN_BACKEND_OPERATION_RESAMPLE_FWD_DESCRIPTOR:
            return DescriptorType_t::OPERATION_RESAMPLE_FWD_DESCRIPTOR;
        case CUDNN_BACKEND_OPERATION_CONCAT_DESCRIPTOR:
            return DescriptorType_t::OPERATION_CONCAT_DESCRIPTOR;
        case CUDNN_BACKEND_OPERATION_SIGNAL_DESCRIPTOR:
            return DescriptorType_t::OPERATION_SIGNAL_DESCRIPTOR;
        case CUDNN_BACKEND_OPERATION_NORM_FORWARD_DESCRIPTOR:
            return DescriptorType_t::OPERATION_NORM_FORWARD_DESCRIPTOR;
        case CUDNN_BACKEND_OPERATION_NORM_BACKWARD_DESCRIPTOR:
            return DescriptorType_t::OPERATION_NORM_BACKWARD_DESCRIPTOR;
#if (CUDNN_VERSION >= 8600)
        case CUDNN_BACKEND_OPERATION_RESAMPLE_BWD_DESCRIPTOR:
            return DescriptorType_t::OPERATION_RESAMPLE_BWD_DESCRIPTOR;
#endif
#if (CUDNN_VERSION >= 8700)
        case CUDNN_BACKEND_OPERATION_RESHAPE_DESCRIPTOR:
            return DescriptorType_t::OPERATION_RESHAPE_DESCRIPTOR;
        case CUDNN_BACKEND_RNG_DESCRIPTOR:
            return DescriptorType_t::RNG_DESCRIPTOR;
        case CUDNN_BACKEND_OPERATION_RNG_DESCRIPTOR:
            return DescriptorType_t::OPERATION_RNG_DESCRIPTOR;
#endif

#if (CUDNN_VERSION >= 90500)
        case CUDNN_BACKEND_OPERATION_PAGED_CACHE_LOAD_DESCRIPTOR:
            return DescriptorType_t::OPERATION_PAGED_CACHE_LOAD_DESCRIPTOR;
#endif
#if (CUDNN_VERSION >= 90700)  // TODO: v9.99 is new feature branch; switch to release branch when ready
        case CUDNN_BACKEND_OPERATION_BLOCK_SCALE_QUANTIZE_DESCRIPTOR:
            return DescriptorType_t::OPERATION_BLOCK_SCALE_QUANTIZE_DESCRIPTOR;
        case CUDNN_BACKEND_OPERATION_BLOCK_SCALE_DEQUANTIZE_DESCRIPTOR:
            return DescriptorType_t::OPERATION_BLOCK_SCALE_DEQUANTIZE_DESCRIPTOR;
#endif
#if (CUDNN_VERSION >= 91500)
        case CUDNN_BACKEND_OPERATION_MOE_GROUPED_MATMUL_DESCRIPTOR:
            return DescriptorType_t::OPERATION_MOE_GROUPED_MATMUL_DESCRIPTOR;
#endif
#if (CUDNN_VERSION >= 92200) && (CUDNN_VERSION < 99900)
        case CUDNN_BACKEND_OPERATION_MOE_GROUPED_MATMUL_BWD_DESCRIPTOR:
            return DescriptorType_t::OPERATION_MOE_GROUPED_MATMUL_BWD_DESCRIPTOR;
#endif

#if (CUDNN_VERSION >= 92200) && (CUDNN_VERSION < 99900)
        case CUDNN_BACKEND_OPERATION_TRANSPOSE_DESCRIPTOR:
            return DescriptorType_t::OPERATION_TRANSPOSE_DESCRIPTOR;
        case CUDNN_BACKEND_OPERATION_SLICE_DESCRIPTOR:
            return DescriptorType_t::OPERATION_SLICE_DESCRIPTOR;
#endif

#ifndef NO_DEFAULT_IN_SWITCH
        default:
            return DescriptorType_t::NOT_SET;
            break;
#endif
    }
    return DescriptorType_t::NOT_SET;
}

// To be deprecated. Only exists as setPointwiseMode(cudnnPointwiseMode_t mode) requires it.
static inline cudnn_frontend::PointwiseMode_t
convert_from_cudnn_type(cudnnPointwiseMode_t const cudnn_mode) {
    switch (cudnn_mode) {
        case CUDNN_POINTWISE_ADD:
            return PointwiseMode_t::ADD;
        case CUDNN_POINTWISE_MUL:
            return PointwiseMode_t::MUL;
        case CUDNN_POINTWISE_SQRT:
            return PointwiseMode_t::SQRT;
        case CUDNN_POINTWISE_MAX:
            return PointwiseMode_t::MAX;
        case CUDNN_POINTWISE_MIN:
            return PointwiseMode_t::MIN;
        case CUDNN_POINTWISE_RELU_FWD:
            return PointwiseMode_t::RELU_FWD;
        case CUDNN_POINTWISE_TANH_FWD:
            return PointwiseMode_t::TANH_FWD;
        case CUDNN_POINTWISE_SIGMOID_FWD:
            return PointwiseMode_t::SIGMOID_FWD;
        case CUDNN_POINTWISE_ELU_FWD:
            return PointwiseMode_t::ELU_FWD;
        case CUDNN_POINTWISE_GELU_FWD:
            return PointwiseMode_t::GELU_FWD;
        case CUDNN_POINTWISE_SOFTPLUS_FWD:
            return PointwiseMode_t::SOFTPLUS_FWD;
        case CUDNN_POINTWISE_SWISH_FWD:
            return PointwiseMode_t::SWISH_FWD;
        case CUDNN_POINTWISE_RELU_BWD:
            return PointwiseMode_t::RELU_BWD;
        case CUDNN_POINTWISE_TANH_BWD:
            return PointwiseMode_t::TANH_BWD;
        case CUDNN_POINTWISE_SIGMOID_BWD:
            return PointwiseMode_t::SIGMOID_BWD;
        case CUDNN_POINTWISE_ELU_BWD:
            return PointwiseMode_t::ELU_BWD;
        case CUDNN_POINTWISE_GELU_BWD:
            return PointwiseMode_t::GELU_BWD;
        case CUDNN_POINTWISE_SOFTPLUS_BWD:
            return PointwiseMode_t::SOFTPLUS_BWD;
        case CUDNN_POINTWISE_SWISH_BWD:
            return PointwiseMode_t::SWISH_BWD;
        case CUDNN_POINTWISE_DIV:
            return PointwiseMode_t::DIV;
        case CUDNN_POINTWISE_ADD_SQUARE:
            return PointwiseMode_t::ADD_SQUARE;
        case CUDNN_POINTWISE_EXP:
            return PointwiseMode_t::EXP;
        case CUDNN_POINTWISE_SUB:
            return PointwiseMode_t::SUB;
        case CUDNN_POINTWISE_CMP_EQ:
            return PointwiseMode_t::CMP_EQ;
        case CUDNN_POINTWISE_CMP_NEQ:
            return PointwiseMode_t::CMP_NEQ;
        case CUDNN_POINTWISE_CMP_GT:
            return PointwiseMode_t::CMP_GT;
        case CUDNN_POINTWISE_CMP_GE:
            return PointwiseMode_t::CMP_GE;
        case CUDNN_POINTWISE_CMP_LT:
            return PointwiseMode_t::CMP_LT;
        case CUDNN_POINTWISE_CMP_LE:
            return PointwiseMode_t::CMP_LE;
        case CUDNN_POINTWISE_LOGICAL_AND:
            return PointwiseMode_t::LOGICAL_AND;
        case CUDNN_POINTWISE_LOGICAL_OR:
            return PointwiseMode_t::LOGICAL_OR;
        case CUDNN_POINTWISE_LOGICAL_NOT:
            return PointwiseMode_t::LOGICAL_NOT;
        case CUDNN_POINTWISE_LOG:
            return PointwiseMode_t::LOG;
        case CUDNN_POINTWISE_NEG:
            return PointwiseMode_t::NEG;
        case CUDNN_POINTWISE_MOD:
            return PointwiseMode_t::MOD;
        case CUDNN_POINTWISE_POW:
            return PointwiseMode_t::POW;
        case CUDNN_POINTWISE_ABS:
            return PointwiseMode_t::ABS;
        case CUDNN_POINTWISE_CEIL:
            return PointwiseMode_t::CEIL;
        case CUDNN_POINTWISE_COS:
            return PointwiseMode_t::COS;
        case CUDNN_POINTWISE_FLOOR:
            return PointwiseMode_t::FLOOR;
        case CUDNN_POINTWISE_RSQRT:
            return PointwiseMode_t::RSQRT;
        case CUDNN_POINTWISE_SIN:
            return PointwiseMode_t::SIN;
        case CUDNN_POINTWISE_TAN:
            return PointwiseMode_t::TAN;
        case CUDNN_POINTWISE_GEN_INDEX:
            return PointwiseMode_t::GEN_INDEX;
        case CUDNN_POINTWISE_BINARY_SELECT:
            return PointwiseMode_t::BINARY_SELECT;
        case CUDNN_POINTWISE_ERF:
            return PointwiseMode_t::ERF;
        case CUDNN_POINTWISE_IDENTITY:
            return PointwiseMode_t::IDENTITY;
        case CUDNN_POINTWISE_GELU_APPROX_TANH_BWD:
            return PointwiseMode_t::GELU_APPROX_TANH_BWD;
        case CUDNN_POINTWISE_GELU_APPROX_TANH_FWD:
            return PointwiseMode_t::GELU_APPROX_TANH_FWD;
#if (CUDNN_VERSION >= 8900)
        case CUDNN_POINTWISE_RECIPROCAL:
            return PointwiseMode_t::RECIPROCAL;
#endif

#ifndef NO_DEFAULT_IN_SWITCH
        default:
            return PointwiseMode_t::NOT_SET;
#endif
    }
    return PointwiseMode_t::NOT_SET;
}

// To be deprecated. Only exists as setDataType(cudnnDataType_t mode) requires it.
static inline cudnn_frontend::DataType_t
convert_from_cudnn_type(cudnnDataType_t const cudnn_mode) {
    switch (cudnn_mode) {
        case CUDNN_DATA_FLOAT:
            return DataType_t::FLOAT;
        case CUDNN_DATA_DOUBLE:
            return DataType_t::DOUBLE;
        case CUDNN_DATA_HALF:
            return DataType_t::HALF;
        case CUDNN_DATA_INT8:
            return DataType_t::INT8;
        case CUDNN_DATA_INT32:
            return DataType_t::INT32;
        case CUDNN_DATA_INT8x4:
            return DataType_t::INT8x4;
        case CUDNN_DATA_UINT8:
            return DataType_t::UINT8;
        case CUDNN_DATA_UINT8x4:
            return DataType_t::UINT8x4;
        case CUDNN_DATA_INT8x32:
            return DataType_t::INT8x32;
        case CUDNN_DATA_BFLOAT16:
            return DataType_t::BFLOAT16;
        case CUDNN_DATA_INT64:
            return DataType_t::INT64;
        case CUDNN_DATA_BOOLEAN:
            return DataType_t::BOOLEAN;
#if (CUDNN_VERSION >= 8600)
        case CUDNN_DATA_FP8_E4M3:
            return DataType_t::FP8_E4M3;
        case CUDNN_DATA_FP8_E5M2:
            return DataType_t::FP8_E5M2;
#endif
#if (CUDNN_VERSION >= 8700)
        case CUDNN_DATA_FAST_FLOAT_FOR_FP8:
            return DataType_t::FAST_FLOAT_FOR_FP8;
#endif
#if (CUDNN_VERSION >= 90700)
        case CUDNN_DATA_FP8_E8M0:
            return DataType_t::FP8_E8M0;
#endif
#if (CUDNN_VERSION >= 90700)
        case CUDNN_DATA_FP4_E2M1:
            return DataType_t::FP4_E2M1;
#endif
#if (CUDNN_VERSION >= 91100)
        case CUDNN_DATA_INT4:
            return DataType_t::INT4;
#endif
#if (CUDNN_VERSION >= 91400)
        case CUDNN_DATA_COMPLEX_FP32:
            return DataType_t::COMPLEX_FP32;
        case CUDNN_DATA_COMPLEX_FP64:
            return DataType_t::COMPLEX_FP64;
#endif
#ifndef NO_DEFAULT_IN_SWITCH
        default:
            return DataType_t::NOT_SET;
#endif
    }
    return DataType_t::NOT_SET;
}

static size_t
get_element_size_in_bits(cudnn_frontend::DataType_t datatype) {
    switch (datatype) {
        case DataType_t::INT8x32:
            return 256;
            break;
#if (CUDNN_VERSION >= 91400)
        case DataType_t::COMPLEX_FP64:
            return 128;
            break;
#endif
        case DataType_t::DOUBLE:
        case DataType_t::INT64:
#if (CUDNN_VERSION >= 91400)
        case DataType_t::COMPLEX_FP32:
#endif
            return 64;
            break;
        case DataType_t::FLOAT:
        case DataType_t::INT32:
        case DataType_t::INT8x4:
        case DataType_t::UINT8x4:
            return 32;
            break;
        case DataType_t::HALF:
        case DataType_t::BFLOAT16:
            return 16;
            break;
        case DataType_t::INT8:
        case DataType_t::UINT8:
#if (CUDNN_VERSION >= 8600)
        case DataType_t::FP8_E4M3:
        case DataType_t::FP8_E5M2:
#endif
#if (CUDNN_VERSION >= 8700)
        case DataType_t::FAST_FLOAT_FOR_FP8:
#endif
#if (CUDNN_VERSION >= 90700)
        case DataType_t::FP8_E8M0:
#endif
            return 8;
            break;
#if (CUDNN_VERSION >= 90700)
        case DataType_t::FP4_E2M1:
#if (CUDNN_VERSION >= 91100)
        case DataType_t::INT4:
#endif
            return 4;
#endif
        case DataType_t::BOOLEAN:
            return 1;
            break;
        default:
            return 0;
            break;
    }
}

// To be deprecated. Only exists as setReductionOp(cudnnReduceTensorOp_t mode) requires it.
static inline cudnn_frontend::ReductionMode_t
convert_from_cudnn_type(cudnnReduceTensorOp_t const cudnn_mode) {
    switch (cudnn_mode) {
        case CUDNN_REDUCE_TENSOR_ADD:
            return ReductionMode_t::ADD;
        case CUDNN_REDUCE_TENSOR_MUL:
            return ReductionMode_t::MUL;
        case CUDNN_REDUCE_TENSOR_MIN:
            return ReductionMode_t::MIN;
        case CUDNN_REDUCE_TENSOR_MAX:
            return ReductionMode_t::MAX;
        case CUDNN_REDUCE_TENSOR_AMAX:
            return ReductionMode_t::AMAX;
        case CUDNN_REDUCE_TENSOR_AVG:
            return ReductionMode_t::AVG;
        case CUDNN_REDUCE_TENSOR_NORM1:
            return ReductionMode_t::NORM1;
        case CUDNN_REDUCE_TENSOR_NORM2:
            return ReductionMode_t::NORM2;
        case CUDNN_REDUCE_TENSOR_MUL_NO_ZEROS:
            return ReductionMode_t::MUL_NO_ZEROS;
#ifndef NO_DEFAULT_IN_SWITCH
        default:
            return ReductionMode_t::NOT_SET;
#endif
    }
    return ReductionMode_t::NOT_SET;
}

#if (CUDNN_VERSION >= 8700)
static inline cudnnStatus_t
convert_to_cudnn_type(cudnn_frontend::RngDistribution_t const mode, cudnnRngDistribution_t& cudnn_mode) {
    NV_CUDNN_FE_DYNAMIC_CHECK_CUDNN_BACKEND_VERSION(8700, cudnnStatus_t::CUDNN_STATUS_INVALID_VALUE);

    switch (mode) {
        case RngDistribution_t::BERNOULLI:
            cudnn_mode = CUDNN_RNG_DISTRIBUTION_BERNOULLI;
            return cudnnStatus_t::CUDNN_STATUS_SUCCESS;
        case RngDistribution_t::UNIFORM:
            cudnn_mode = CUDNN_RNG_DISTRIBUTION_UNIFORM;
            return cudnnStatus_t::CUDNN_STATUS_SUCCESS;
        case RngDistribution_t::NORMAL:
            cudnn_mode = CUDNN_RNG_DISTRIBUTION_NORMAL;
            return cudnnStatus_t::CUDNN_STATUS_SUCCESS;

#ifndef NO_DEFAULT_IN_SWITCH
        default:
            return cudnnStatus_t::CUDNN_STATUS_INVALID_VALUE;
#endif
    }
    return cudnnStatus_t::CUDNN_STATUS_INVALID_VALUE;
}

// To be deprecated. Only exists as setRngDistribution(cudnnRngDistribution_t mode) requires it.
static inline cudnn_frontend::RngDistribution_t
convert_from_cudnn_type(cudnnRngDistribution_t const cudnn_mode) {
    switch (cudnn_mode) {
        case CUDNN_RNG_DISTRIBUTION_BERNOULLI:
            return RngDistribution_t::BERNOULLI;
        case CUDNN_RNG_DISTRIBUTION_UNIFORM:
            return RngDistribution_t::UNIFORM;
        case CUDNN_RNG_DISTRIBUTION_NORMAL:
            return RngDistribution_t::NORMAL;
#ifndef NO_DEFAULT_IN_SWITCH
        default:
            return RngDistribution_t::NOT_SET;
#endif
    }
    return RngDistribution_t::NOT_SET;
}
#endif

#if (CUDNN_VERSION >= 91500)
static inline cudnnStatus_t
convert_to_cudnn_type(cudnn_frontend::MoeGroupedMatmulMode_t const mode, cudnnMoeGroupedMatmulMode_t& cudnn_mode) {
    NV_CUDNN_FE_DYNAMIC_CHECK_CUDNN_BACKEND_VERSION(91500, cudnnStatus_t::CUDNN_STATUS_INVALID_VALUE);
    switch (mode) {
        case MoeGroupedMatmulMode_t::NONE:
            cudnn_mode = CUDNN_MOE_GROUPED_MATMUL_MODE_NONE;
            return cudnnStatus_t::CUDNN_STATUS_SUCCESS;
        case MoeGroupedMatmulMode_t::GATHER:
            cudnn_mode = CUDNN_MOE_GROUPED_MATMUL_MODE_GATHER;
            return cudnnStatus_t::CUDNN_STATUS_SUCCESS;
        case MoeGroupedMatmulMode_t::SCATTER:
            cudnn_mode = CUDNN_MOE_GROUPED_MATMUL_MODE_SCATTER;
            return cudnnStatus_t::CUDNN_STATUS_SUCCESS;
#ifndef NO_DEFAULT_IN_SWITCH
        default:
            return cudnnStatus_t::CUDNN_STATUS_INVALID_VALUE;
#endif
    }
    return cudnnStatus_t::CUDNN_STATUS_INVALID_VALUE;
}

static inline cudnn_frontend::MoeGroupedMatmulMode_t
convert_from_cudnn_type(cudnnMoeGroupedMatmulMode_t const cudnn_mode) {
    switch (cudnn_mode) {
        case CUDNN_MOE_GROUPED_MATMUL_MODE_NONE:
            return MoeGroupedMatmulMode_t::NONE;
        case CUDNN_MOE_GROUPED_MATMUL_MODE_GATHER:
            return MoeGroupedMatmulMode_t::GATHER;
        case CUDNN_MOE_GROUPED_MATMUL_MODE_SCATTER:
            return MoeGroupedMatmulMode_t::SCATTER;
#ifndef NO_DEFAULT_IN_SWITCH
        default:
            return MoeGroupedMatmulMode_t::NOT_SET;
#endif
    }
    return MoeGroupedMatmulMode_t::NOT_SET;
}
#endif

std::string static get_engine_tag(ManagedOpaqueDescriptor const config) {
    std::stringstream tag{""};
    ManagedOpaqueDescriptor extractedEngine = make_shared_backend_pointer(CUDNN_BACKEND_ENGINE_DESCRIPTOR);
    auto status                             = extractedEngine->get_status();

    cudnnBackendDescriptor_t extractedEngine_ = extractedEngine->get_backend_descriptor();
    int64_t elemCount                         = 0;
    status                                    = detail::get_attribute(config->get_backend_descriptor(),
                                   CUDNN_ATTR_ENGINECFG_ENGINE,
                                   CUDNN_TYPE_BACKEND_DESCRIPTOR,
                                   1,
                                   &elemCount,
                                   &extractedEngine_);
    if (status != CUDNN_STATUS_SUCCESS) {
        return "INVALID_ENGINE_NAME_CFG";
    }

    int64_t engineId = 0, numKnobs = 0;

    std::array<ManagedOpaqueDescriptor, CUDNN_KNOB_TYPE_COUNTS> extractedKnobs{{nullptr}};
    for (auto& knob : extractedKnobs) {
        knob   = make_shared_backend_pointer(CUDNN_BACKEND_KNOB_CHOICE_DESCRIPTOR);
        status = knob->get_status();
        if (status != CUDNN_STATUS_SUCCESS) {
            return "INVALID_ENGINE_NAME_KNOB";
        }
    }

    std::array<cudnnBackendDescriptor_t, CUDNN_KNOB_TYPE_COUNTS> extractedKnobs_{{nullptr}};
    for (std::uint32_t i = 0; i < extractedKnobs.size(); i++) {
        extractedKnobs_[i] = extractedKnobs[i]->get_backend_descriptor();
    }

    status = detail::get_attribute(
        extractedEngine_, CUDNN_ATTR_ENGINE_GLOBAL_INDEX, CUDNN_TYPE_INT64, 1, &elemCount, &engineId);
    if (status != CUDNN_STATUS_SUCCESS) {
        return "INVALID_ENGINE_NAME_IDX";
    }
    tag << "eng" << engineId;

    status = detail::get_attribute(config->get_backend_descriptor(),
                                   CUDNN_ATTR_ENGINECFG_KNOB_CHOICES,
                                   CUDNN_TYPE_BACKEND_DESCRIPTOR,
                                   CUDNN_KNOB_TYPE_COUNTS,
                                   &numKnobs,
                                   &(extractedKnobs_[0]));
    if (status != CUDNN_STATUS_SUCCESS) {
        return "INVALID_ENGINE_NAME_KNOB_QUERY";
    }
    if (numKnobs > CUDNN_KNOB_TYPE_COUNTS) {
        return "INVALID_ENGINE_NAME_KNOB_COUNT";
    }

    for (size_t idx = 0; idx < static_cast<size_t>(numKnobs); ++idx) {
        const cudnnBackendDescriptor_t& knob = extractedKnobs_[idx];
        cudnnBackendKnobType_t type          = CUDNN_KNOB_TYPE_COUNTS;
        int64_t choice                       = -2;
        status = detail::get_attribute(knob, CUDNN_ATTR_KNOB_CHOICE_KNOB_TYPE, CUDNN_TYPE_KNOB_TYPE, 1, nullptr, &type);
        if (status != CUDNN_STATUS_SUCCESS) {
            return "INVALID_ENGINE_NAME_KNOB_CHOICE_KNOB_TYPE";
        }
        status = detail::get_attribute(knob, CUDNN_ATTR_KNOB_CHOICE_KNOB_VALUE, CUDNN_TYPE_INT64, 1, nullptr, &choice);
        if (status != CUDNN_STATUS_SUCCESS) {
            return "INVALID_ENGINE_NAME_KNOB_CHOICE_KNOB_VALUE";
        }
        tag << "_k" << type << "=" << choice;
    }
    return tag.str();
}

}  // namespace detail

}  // namespace cudnn_frontend
