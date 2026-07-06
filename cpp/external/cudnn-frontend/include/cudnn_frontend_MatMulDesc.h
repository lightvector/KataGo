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
class MatmulNode;
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
/// MatMulDesc  Descriptor Class
/// This class tells the properties of the MatMul operation
/// Properties:
///    - compute_type
///
/// Use MatMulDesc_v8 to build this class.
/// Describe returns a string describing the MatMul operation
///
class MatMulDesc_v8 : public BackendDescriptor {
   public:
    friend class MatMulDescBuilder_v8;
    friend class cudnn_frontend::graph::MatmulNode;
    std::string
    describe() const override {
        std::stringstream ss;
#ifndef CUDNN_FRONTEND_SKIP_JSON_LIB
        ss << "CUDNN_BACKEND_MATMUL_DESCRIPTOR :" << " Math precision " << json{compute_type};
#else
        ss << "CUDNN_BACKEND_MATMUL_DESCRIPTOR :" << " Math precision " << int(compute_type);
#endif
        return ss.str();
    }

    MatMulDesc_v8(MatMulDesc_v8 &&from) = default;
    MatMulDesc_v8 &
    operator=(MatMulDesc_v8 &&from) = default;

    ~MatMulDesc_v8() = default;

   private:
    MatMulDesc_v8()                      = default;
    MatMulDesc_v8(MatMulDesc_v8 const &) = delete;
    MatMulDesc_v8 &
    operator=(MatMulDesc_v8 const &) = delete;

    DataType_t compute_type = DataType_t::NOT_SET;
    bool isPadded           = false;
    double paddingValue     = 0.0;
};

////
/// MatMulDescBuilder_v8 Class
/// Helper class used to build MatMulDesc_v8 class
class MatMulDescBuilder_v8 {
   public:
    /** @defgroup MatMulDescBuilder_v8
     *  Set individual property of MatMulDesc_v8 class
     *  @{
     */
    //! Set Math Precision Data Type for the Matmul Operation
    auto
    setComputeType(DataType_t data_type_) -> MatMulDescBuilder_v8 & {
        m_matMulDesc.compute_type = data_type_;
        return *this;
    }
    auto
    setComputeType(cudnnDataType_t data_type_) -> MatMulDescBuilder_v8 & {
        m_matMulDesc.compute_type = detail::convert_from_cudnn_type(data_type_);
        return *this;
    }
    /** @} */

    // TODO Deprecate in v1.0
    auto
    setMathPrecision(cudnnDataType_t data_type_) -> MatMulDescBuilder_v8 & {
        return setComputeType(data_type_);
    }

    //! Set padding value for matmul descriptor
    auto
    setPaddingValue(double paddingValue) -> MatMulDescBuilder_v8 & {
        m_matMulDesc.isPadded     = true;
        m_matMulDesc.paddingValue = paddingValue;
        return *this;
    }

    //! constructs the MatMulDesc_v8 by calling the cudnn API
    //! Throws the appropriate error message
    MatMulDesc_v8 &&
    build() {
        // Create a descriptor. Memory allocation happens here.
        auto status = m_matMulDesc.initialize_managed_backend_pointer(CUDNN_BACKEND_MATMUL_DESCRIPTOR);
        if (status != CUDNN_STATUS_SUCCESS) {
            set_error_and_throw_exception(&m_matMulDesc, status, "CUDNN_BACKEND_MATMUL_DESCRIPTOR: cudnnCreate Failed");
            return std::move(m_matMulDesc);
        }

        // Once Created lets set the descriptor parameters.
        cudnnDataType_t cudnn_data_type;
        status = detail::convert_to_cudnn_type(m_matMulDesc.compute_type, cudnn_data_type);
        if (status != CUDNN_STATUS_SUCCESS) {
            set_error_and_throw_exception(
                &m_matMulDesc,
                status,
                "CUDNN_BACKEND_MATMUL_DESCRIPTOR: SetAttribute CUDNN_ATTR_MATMUL_COMP_TYPE Failed");
            return std::move(m_matMulDesc);
        }
        status = detail::set_attribute(m_matMulDesc.pointer->get_backend_descriptor(),
                                       CUDNN_ATTR_MATMUL_COMP_TYPE,
                                       CUDNN_TYPE_DATA_TYPE,
                                       1,
                                       &cudnn_data_type);
        if (status != CUDNN_STATUS_SUCCESS) {
            set_error_and_throw_exception(
                &m_matMulDesc,
                status,
                "CUDNN_BACKEND_MATMUL_DESCRIPTOR: SetAttribute CUDNN_ATTR_MATMUL_COMP_TYPE Failed");
            return std::move(m_matMulDesc);
        }

#if (CUDNN_VERSION >= 8900)
        // Setting padding value if matmul desc is padded
        NV_CUDNN_FE_DYNAMIC_CHECK_BACKEND_DESCRIPTOR(
            8900,
            m_matMulDesc,
            "CUDNN_BACKEND_MATMUL_DESCRIPTOR: SetAttribute CUDNN_ATTR_MATMUL_PADDING_VALUE requires cudnn 8.9.0");
        if (m_matMulDesc.isPadded) {
            status = detail::set_attribute(m_matMulDesc.pointer->get_backend_descriptor(),
                                           CUDNN_ATTR_MATMUL_PADDING_VALUE,
                                           CUDNN_TYPE_DOUBLE,
                                           1,
                                           &m_matMulDesc.paddingValue);
            if (status != CUDNN_STATUS_SUCCESS) {
                set_error_and_throw_exception(
                    &m_matMulDesc,
                    status,
                    "CUDNN_BACKEND_MATMUL_DESCRIPTOR: SetAttribute CUDNN_ATTR_MATMUL_PADDING_VALUE Failed");
                return std::move(m_matMulDesc);
            }
        }
#endif

        // Finalizing the descriptor
        status = detail::finalize(m_matMulDesc.pointer->get_backend_descriptor());
        if (status != CUDNN_STATUS_SUCCESS) {
            set_error_and_throw_exception(
                &m_matMulDesc, status, "CUDNN_BACKEND_MATMUL_DESCRIPTOR: cudnnFinalize Failed");
            return std::move(m_matMulDesc);
        }

        CUDNN_FE_LOG_LABEL_ENDL(m_matMulDesc);
        return std::move(m_matMulDesc);
    }

    explicit MatMulDescBuilder_v8()                    = default;
    ~MatMulDescBuilder_v8()                            = default;
    MatMulDescBuilder_v8(MatMulDescBuilder_v8 &&)      = delete;
    MatMulDescBuilder_v8(MatMulDescBuilder_v8 const &) = delete;
    MatMulDescBuilder_v8 &
    operator=(MatMulDescBuilder_v8 const &) = delete;

   private:
    MatMulDesc_v8 m_matMulDesc;
};
using MatMulDesc        = MatMulDesc_v8;
using MatMulDescBuilder = MatMulDescBuilder_v8;
}  // namespace cudnn_frontend
