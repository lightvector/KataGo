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

#include <algorithm>

namespace cudnn_frontend {
namespace graph {
class ReductionNode;
}
}  // namespace cudnn_frontend
#include <array>
#include <functional>
#include <memory>
#include <sstream>
#include <utility>

#include "cudnn_frontend_utils.h"

namespace cudnn_frontend {
///
/// ReductionDesc  Descriptor Class
/// This class tells the properties of the Reduction operation
/// Properties:
///    - compute_type
///    - reduction_mode
///    - is_deterministic
///
/// Use ReductionDesc_v8 to build this class.
/// Describe returns a string describing the Reduction operation
///
class ReductionDesc_v8 : public BackendDescriptor {
   public:
    friend class ReductionDescBuilder_v8;
    friend class cudnn_frontend::graph::ReductionNode;
    std::string
    describe() const override {
        std::stringstream ss;
#ifndef CUDNN_FRONTEND_SKIP_JSON_LIB
        ss << "CUDNN_BACKEND_REDUCTION_DESCRIPTOR :"
           << " Math precision " << json{compute_type} << " Reduction mode " << json{reduction_mode}
           << " Is deterministic " << json(is_deterministic);
#else
        ss << "CUDNN_BACKEND_REDUCTION_DESCRIPTOR :"
           << " Math precision " << (int)compute_type << " Reduction mode " << int(reduction_mode)
           << " Is deterministic " << int(is_deterministic);
#endif
        return ss.str();
    }

    ReductionDesc_v8(ReductionDesc_v8 &&from) = default;
    ReductionDesc_v8 &
    operator=(ReductionDesc_v8 &&from) = default;

    ~ReductionDesc_v8() = default;

   private:
    ReductionDesc_v8()                         = default;
    ReductionDesc_v8(ReductionDesc_v8 const &) = delete;
    ReductionDesc_v8 &
    operator=(ReductionDesc_v8 const &) = delete;

    DataType_t compute_type        = DataType_t::NOT_SET;
    ReductionMode_t reduction_mode = ReductionMode_t::NOT_SET;
    bool is_deterministic          = false;
};

////
/// ReductionDescBuilder_v8 Class
/// Helper class used to build ReductionDesc_v8 class
class ReductionDescBuilder_v8 {
   public:
    /** @defgroup ReductionDescBuilder_v8
     *  Set individual property of ReductionDesc_v8 class
     *  @{
     */
    //! Set Math Precision Data Type for the Reduction Operation
    auto
    setComputeType(DataType_t data_type_) -> ReductionDescBuilder_v8 & {
        m_reductionDesc.compute_type = data_type_;
        return *this;
    }
    auto
    setComputeType(cudnnDataType_t data_type_) -> ReductionDescBuilder_v8 & {
        m_reductionDesc.compute_type = detail::convert_from_cudnn_type(data_type_);
        return *this;
    }
    //! Set redution operator for the Reduction Operation
    auto
    setReductionOp(ReductionMode_t op_) -> ReductionDescBuilder_v8 & {
        m_reductionDesc.reduction_mode = op_;
        return *this;
    }
    auto
    setReductionOp(cudnnReduceTensorOp_t op_) -> ReductionDescBuilder_v8 & {
        m_reductionDesc.reduction_mode = detail::convert_from_cudnn_type(op_);
        return *this;
    }
    /** @} */

    // TODO Deprecate in v1.0
    auto
    setMathPrecision(cudnnDataType_t data_type_) -> ReductionDescBuilder_v8 & {
        return setComputeType(data_type_);
    }
    auto
    setIsDeterministic(bool is_deterministic_) -> ReductionDescBuilder_v8 & {
        m_reductionDesc.is_deterministic = is_deterministic_;
        return *this;
    }

    //! constructs the ReductionDesc_v8 by calling the cudnn API
    //! Throws the appropriate error message
    ReductionDesc_v8 &&
    build() {
        // Create a descriptor. Memory allocation happens here.
        auto status = m_reductionDesc.initialize_managed_backend_pointer(CUDNN_BACKEND_REDUCTION_DESCRIPTOR);
        if (status != CUDNN_STATUS_SUCCESS) {
            set_error_and_throw_exception(
                &m_reductionDesc, status, "CUDNN_BACKEND_REDUCTION_DESCRIPTOR: cudnnCreate Failed");
            return std::move(m_reductionDesc);
        }

        // Once Created lets set the descriptor parameters.
        cudnnDataType_t cudnn_data_type;
        status = detail::convert_to_cudnn_type(m_reductionDesc.compute_type, cudnn_data_type);
        if (status != CUDNN_STATUS_SUCCESS) {
            set_error_and_throw_exception(
                &m_reductionDesc,
                status,
                "CUDNN_BACKEND_REDUCTION_DESCRIPTOR: SetAttribute CUDNN_ATTR_REDUCTION_COMP_TYPE Failed");
            return std::move(m_reductionDesc);
        }
        status = detail::set_attribute(m_reductionDesc.pointer->get_backend_descriptor(),
                                       CUDNN_ATTR_REDUCTION_COMP_TYPE,
                                       CUDNN_TYPE_DATA_TYPE,
                                       1,
                                       &cudnn_data_type);
        if (status != CUDNN_STATUS_SUCCESS) {
            set_error_and_throw_exception(
                &m_reductionDesc,
                status,
                "CUDNN_BACKEND_REDUCTION_DESCRIPTOR: SetAttribute CUDNN_ATTR_REDUCTION_COMP_TYPE Failed");
            return std::move(m_reductionDesc);
        }

        cudnnReduceTensorOp_t cudnn_reduction_mode;
        status = detail::convert_to_cudnn_type(m_reductionDesc.reduction_mode, cudnn_reduction_mode);
        if (status != CUDNN_STATUS_SUCCESS) {
            set_error_and_throw_exception(
                &m_reductionDesc,
                status,
                "CUDNN_BACKEND_REDUCTION_DESCRIPTOR: SetAttribute CUDNN_ATTR_REDUCTION_OPERATOR Failed");
            return std::move(m_reductionDesc);
        }
        status = detail::set_attribute(m_reductionDesc.pointer->get_backend_descriptor(),
                                       CUDNN_ATTR_REDUCTION_OPERATOR,
                                       CUDNN_TYPE_REDUCTION_OPERATOR_TYPE,
                                       1,
                                       &cudnn_reduction_mode);
        if (status != CUDNN_STATUS_SUCCESS) {
            set_error_and_throw_exception(
                &m_reductionDesc,
                status,
                "CUDNN_BACKEND_REDUCTION_DESCRIPTOR: SetAttribute CUDNN_ATTR_REDUCTION_OPERATOR Failed");
            return std::move(m_reductionDesc);
        }

#if (CUDNN_VERSION >= 91100)
        // If backend version is less then 9.11.0, then determinisitc mode is not even supported.
        // But in the default case which exists in current implementations, is_deterministic is false, and should be
        // ignored.
        if (detail::get_backend_version() < 91100) {
            if (m_reductionDesc.is_deterministic) {
                set_error_and_throw_exception(&m_reductionDesc,
                                              CUDNN_STATUS_NOT_SUPPORTED,
                                              "CUDNN_BACKEND_REDUCTION_DESCRIPTOR: DETERMINISTIC mode is not supported "
                                              "in cudnn version < 9.11.0");
                return std::move(m_reductionDesc);
            } else {
                // Do nothing.
            }
        } else {
            status = detail::set_attribute(m_reductionDesc.pointer->get_backend_descriptor(),
                                           CUDNN_ATTR_REDUCTION_IS_DETERMINISTIC,
                                           CUDNN_TYPE_BOOLEAN,
                                           1,
                                           &m_reductionDesc.is_deterministic);
            if (status != CUDNN_STATUS_SUCCESS) {
                set_error_and_throw_exception(
                    &m_reductionDesc,
                    status,
                    "CUDNN_BACKEND_REDUCTION_DESCRIPTOR: SetAttribute CUDNN_ATTR_REDUCTION_IS_DETERMINISTIC Failed");
                return std::move(m_reductionDesc);
            }
        }
#endif

        // Finalizing the descriptor
        status = detail::finalize(m_reductionDesc.pointer->get_backend_descriptor());
        if (status != CUDNN_STATUS_SUCCESS) {
            set_error_and_throw_exception(
                &m_reductionDesc, status, "CUDNN_BACKEND_REDUCTION_DESCRIPTOR: cudnnFinalize Failed");
            return std::move(m_reductionDesc);
        }

        CUDNN_FE_LOG_LABEL_ENDL(m_reductionDesc);
        return std::move(m_reductionDesc);
    }

    explicit ReductionDescBuilder_v8()                       = default;
    ~ReductionDescBuilder_v8()                               = default;
    ReductionDescBuilder_v8(ReductionDescBuilder_v8 &&)      = delete;
    ReductionDescBuilder_v8(ReductionDescBuilder_v8 const &) = delete;
    ReductionDescBuilder_v8 &
    operator=(ReductionDescBuilder_v8 const &) = delete;

   private:
    ReductionDesc_v8 m_reductionDesc;
};

using ReductionDesc        = ReductionDesc_v8;
using ReductionDescBuilder = ReductionDescBuilder_v8;

}  // namespace cudnn_frontend
