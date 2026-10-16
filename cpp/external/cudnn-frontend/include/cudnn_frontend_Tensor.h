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
#include <cstring>
#include <functional>
#include <memory>
#include <sstream>
#include <utility>
#include <vector>

#include "cudnn_frontend_utils.h"

namespace cudnn_frontend {

///
/// Tensor_v8 Class
/// This class tells the properties of the Tensor_v8 on which the operation will be
/// performed
/// Properties:
///    - dataType
///    - alignment
///    - unique identifier
///    - tensor dimensions
///    - tensor strides
///    - isVirtual
///    - isByValue
///
/// Use TensorBuilder_v8 to build this class.
/// Describe returns a string describing the tensor class
///
class Tensor_v8 : public BackendDescriptor {
   public:
    friend class TensorBuilder_v8;
    std::string
    describe() const override {
        std::stringstream ss;
        ss << "               CUDNN_BACKEND_TENSOR_DESCRIPTOR :\n"
#ifndef CUDNN_FRONTEND_SKIP_JSON_LIB
           << "                                   Datatype: " << json{data_type}
#else
           << "                                   Datatype: " << int(data_type)
#endif
           << "\n                                   Id: " << id << "\n                                   Dims " << nDims
           << "\n                                   VectorCount: " << vectorCount
           << "\n                                   vectorDimension " << vectorDimension;
        ss << "\n                                   Dim [ ";
        for (auto i = 0; i < nDims; i++) {
            if (i != 0) {
                ss << ',';
            }
            ss << btensor_dimA[i];
        }
        ss << " ]\n                                   Str [ ";
        for (auto i = 0; i < nDims; i++) {
            if (i != 0) {
                ss << ',';
            }
            ss << btensor_strA[i];
        }
        ss << " ]";
        ss << "\n                                   isVirtual: " << isVirtual
           << "\n                                   isByValue: " << isByValue
           << "\n                                   Alignment: " << alignment;
#ifndef CUDNN_FRONTEND_SKIP_JSON_LIB
        ss << "\n                                   reorder_type: " << json{reorder_type};
#else
        ss << "\n                                   reorder_type: " << int(reorder_type);
#endif
        if (raggedOffset != nullptr) {
            ss << "\n                                   raggedOffset: Enabled UID: " << raggedOffset->getId();
        }
        return ss.str();
    }

    int64_t
    getPackedElementCount() const {
        int64_t count = vectorCount;
        for (auto i = 0; i < nDims; i++) {
            count = count * btensor_dimA[i];
        }
        return count;
    };

    int64_t
    getDimCount() const {
        return nDims;
    }

    int64_t const *
    getDim() const {
        return btensor_dimA;
    }

    int64_t const *
    getStride() const {
        return btensor_strA;
    }

    // TODO: Deprecate in v1.0
    int64_t const *
    getDimArray() const {
        return getDim();
    }

    // TODO: Deprecate in v1.0
    int64_t const *
    getStrideArray() const {
        return getStride();
    }

    int64_t
    getDataType() const {
        return static_cast<int64_t>(data_type);
    }

    int64_t
    getId() const {
        return id;
    }

    int64_t
    getAlignment() const {
        return alignment;
    }

    bool
    isVirtualTensor() const {
        return isVirtual;
    }

    // TODO: Deprecate in v1.0
    int64_t
    getDimensionCount() const {
        return getDimCount();
    }

    Tensor_v8(Tensor_v8 &&from) = default;
    Tensor_v8 &
    operator=(Tensor_v8 &&) = default;

    ~Tensor_v8() = default;

   private:
    Tensor_v8()                  = default;
    Tensor_v8(Tensor_v8 const &) = delete;
    Tensor_v8 &
    operator=(Tensor_v8 const &) = delete;

    DataType_t data_type                    = DataType_t::NOT_SET;  //! Datatype of the elements
    int64_t btensor_dimA[CUDNN_DIM_MAX + 1] = {-1};                 //! n, g, c, d, h, w
    int64_t btensor_strA[CUDNN_DIM_MAX + 1] = {-1};                 //! n, g, c, d, h, w
    int64_t id                              = -1;                   //! Unique id of the tensor
    int64_t alignment                       = -1;                   //! Alignment of the tensor.
    //! Certain engine config expect minimum alignment of 16B
    int64_t nDims           = -1;     //! Number of Dimensions of the tensor
    int64_t vectorDimension = -1;     //! Which dimension of the tensor is vectorized (Generally the c dim)
    int64_t vectorCount     = 1;      //! What is the vectorization count (4 or 32)
    bool isVirtual          = false;  //! Whether it is an intermediate tensor of an op graph
    bool isByValue = false;  //! Whether the tensor is in host memory that needs to be passed to the kernel by value

    bool hasConstValue = false;            //! Whether this tensor has a compile-time constant value
    std::vector<uint8_t> constValueBytes;  //! Raw bytes of the compile-time constant value

    cudnn_frontend::TensorReordering_t reorder_type =
        cudnn_frontend::TensorReordering_t::NONE;  //! Type of reordering in the tensor
    std::shared_ptr<Tensor_v8> raggedOffset;       //! Ragged offsets for ragged tensors
};

///
/// TensorBuilder_v8 Class
/// Helper class used to build Tensor_v8 class
class TensorBuilder_v8 {
   public:
    /** @defgroup TensorBuilder_v8
     *  Set individual property of Tensor_v8 class
     *  @{
     */
    //! Set Datatype for the Tensor_v8
    auto
    setDataType(DataType_t data_type) -> TensorBuilder_v8 & {
        m_tensor.data_type = data_type;
        return *this;
    }
    // To be deprecated in v1.0. Please use setDataType(DataType_t) instead.
    auto
    setDataType(cudnnDataType_t data_type) -> TensorBuilder_v8 & {
        m_tensor.data_type = detail::convert_from_cudnn_type(data_type);
        return *this;
    }
    //! Set Dimensions of the tensor
    auto
    setDim(int64_t ndim, int64_t const *dim) -> TensorBuilder_v8 & {
        std::copy((dim), dim + ndim, m_tensor.btensor_dimA);
        m_tensor.nDims = ndim;
        return *this;
    }
    //! Set Strides of the tensor
    auto
    setStride(int64_t ndim, int64_t const *strides) -> TensorBuilder_v8 & {
        std::copy(strides, strides + ndim, m_tensor.btensor_strA);
        return *this;
    }
    //! Set Unique Id  of the tensor
    auto
    setId(int64_t id_) -> TensorBuilder_v8 & {
        m_tensor.id = id_;
        return *this;
    }
    //! Set Alignment of the tensor
    auto
    setAlignment(int64_t alignment_) -> TensorBuilder_v8 & {
        m_tensor.alignment = alignment_;
        return *this;
    }
    //! Set isVirtual of the tensor
    auto
    setVirtual(bool virtual_ = true) -> TensorBuilder_v8 & {
        m_tensor.isVirtual = virtual_;
        return *this;
    }
    //! Set isByValue of the tensor
    auto
    setByValue(bool isByValue_ = true) -> TensorBuilder_v8 & {
        m_tensor.isByValue = isByValue_;
        return *this;
    }

    //! Set compile-time constant value for this tensor
    template <typename T>
    auto
    setConstValue(T const &value) -> TensorBuilder_v8 & {
        m_tensor.hasConstValue = true;
        m_tensor.constValueBytes.resize(sizeof(T));
        std::memcpy(m_tensor.constValueBytes.data(), &value, sizeof(T));
        m_tensor.isByValue = true;
        return *this;
    }

    auto
    setVectorCountAndDimension(int64_t vectorCount_, int64_t vectorDimension_) -> TensorBuilder_v8 & {
        m_tensor.vectorCount     = vectorCount_;
        m_tensor.vectorDimension = vectorDimension_;
        return *this;
    }

    auto
    setReorderType(cudnn_frontend::TensorReordering_t reordering_type) -> TensorBuilder_v8 & {
        m_tensor.reorder_type = reordering_type;
        return *this;
    }

    // To be deprecated. Please use setReorderType(cudnn_frontend::cudnnBackendTensorReordering_t).
    auto
    setReorderType(cudnnBackendTensorReordering_t reordering_type) -> TensorBuilder_v8 & {
        detail::convert_from_cudnn_type(reordering_type, m_tensor.reorder_type);
        return *this;
    }

    /** @} */

    // TODO: Deprecate in v1.0
    auto
    setStrides(int64_t ndim, int64_t const *strides) -> TensorBuilder_v8 & {
        return setStride(ndim, strides);
    }

    auto
    setRaggedOffset(std::shared_ptr<Tensor_v8> &raggedOffset) -> TensorBuilder_v8 & {
        m_tensor.raggedOffset = raggedOffset;
        return *this;
    }

    // Clone parameters of another tensor. Make sure to still set the UID since UID of two tensors shouldn't be the
    // same.
    auto
    cloneFrom(Tensor_v8 const &from, int64_t newID) -> TensorBuilder_v8 & {
        m_tensor.data_type = from.data_type;
        m_tensor.nDims     = from.nDims;
        m_tensor.id        = newID;
        std::copy(from.getDimArray(), from.getDimArray() + m_tensor.nDims, m_tensor.btensor_dimA);
        std::copy(from.getStrideArray(), from.getStrideArray() + m_tensor.nDims, m_tensor.btensor_strA);
        m_tensor.alignment       = from.alignment;
        m_tensor.isVirtual       = from.isVirtual;
        m_tensor.isByValue       = from.isByValue;
        m_tensor.vectorCount     = from.vectorCount;
        m_tensor.vectorDimension = from.vectorDimension;
        m_tensor.reorder_type    = from.reorder_type;
        return *this;
    }

    //! constructs the Tensor_v8 by calling the cudnn API
    //! Throws the appropriate error message
    Tensor_v8 &&
    build() {
        // Sanity check if non-default fields have been set correctly.
        if (m_tensor.alignment <= 0) {
            set_error_and_throw_exception(
                &m_tensor,
                CUDNN_STATUS_BAD_PARAM,
                "CUDNN_BACKEND_TENSOR_DESCRIPTOR: Check and Set the CUDNN_ATTR_TENSOR_BYTE_ALIGNMENT field");
            return std::move(m_tensor);
        }
        if (m_tensor.id < 0) {
            set_error_and_throw_exception(
                &m_tensor,
                CUDNN_STATUS_BAD_PARAM,
                "CUDNN_BACKEND_TENSOR_DESCRIPTOR: Check and Set the CUDNN_ATTR_TENSOR_UNIQUE_ID as a valid value");
            return std::move(m_tensor);
        }
        if (m_tensor.btensor_strA[0] < 0) {
            set_error_and_throw_exception(
                &m_tensor,
                CUDNN_STATUS_BAD_PARAM,
                "CUDNN_BACKEND_TENSOR_DESCRIPTOR: Check and Set the CUDNN_ATTR_TENSOR_STRIDES Correctly");
            return std::move(m_tensor);
        }
        if (m_tensor.btensor_dimA[0] <= 0) {
            set_error_and_throw_exception(
                &m_tensor,
                CUDNN_STATUS_BAD_PARAM,
                "CUDNN_BACKEND_TENSOR_DESCRIPTOR: Check and Set the CUDNN_ATTR_TENSOR_DIMENSIONS Correctly");
            return std::move(m_tensor);
        }
        if (m_tensor.pointer != nullptr) {
            set_error_and_throw_exception(&m_tensor,
                                          CUDNN_STATUS_BAD_PARAM,
                                          "CUDNN_BACKEND_TENSOR_DESCRIPTOR: Bad tensor created. The tensor already "
                                          "seems to be pointing to something");
            return std::move(m_tensor);
        }

        // Create a descriptor. Memory allocation happens here.
        auto status = m_tensor.initialize_managed_backend_pointer(CUDNN_BACKEND_TENSOR_DESCRIPTOR);
        if (status != CUDNN_STATUS_SUCCESS) {
            set_error_and_throw_exception(
                &m_tensor, status, "CUDNN_BACKEND_TENSOR_DESCRIPTOR: cudnnCreate Descriptor Failed");
            return std::move(m_tensor);
        }

        // Once Created lets set the descriptor parameters.
        cudnnDataType_t cudnn_data_type;
        status = detail::convert_to_cudnn_type(m_tensor.data_type, cudnn_data_type);
        if (status != CUDNN_STATUS_SUCCESS) {
            set_error_and_throw_exception(
                &m_tensor, status, "CUDNN_BACKEND_TENSOR_DESCRIPTOR: SetAttribute CUDNN_ATTR_TENSOR_DATA_TYPE Failed");
            return std::move(m_tensor);
        }
        status = detail::set_attribute(m_tensor.pointer->get_backend_descriptor(),
                                       CUDNN_ATTR_TENSOR_DATA_TYPE,
                                       CUDNN_TYPE_DATA_TYPE,
                                       1,
                                       &cudnn_data_type);
        if (status != CUDNN_STATUS_SUCCESS) {
            set_error_and_throw_exception(
                &m_tensor, status, "CUDNN_BACKEND_TENSOR_DESCRIPTOR: SetAttribute CUDNN_ATTR_TENSOR_DATA_TYPE Failed");
            return std::move(m_tensor);
        }
        status = detail::set_attribute(m_tensor.pointer->get_backend_descriptor(),
                                       CUDNN_ATTR_TENSOR_DIMENSIONS,
                                       CUDNN_TYPE_INT64,
                                       m_tensor.nDims,
                                       m_tensor.btensor_dimA);
        if (status != CUDNN_STATUS_SUCCESS) {
            set_error_and_throw_exception(
                &m_tensor, status, "CUDNN_BACKEND_TENSOR_DESCRIPTOR: SetAttribute CUDNN_ATTR_TENSOR_DIMENSIONS Failed");
            return std::move(m_tensor);
        }
        status = detail::set_attribute(m_tensor.pointer->get_backend_descriptor(),
                                       CUDNN_ATTR_TENSOR_STRIDES,
                                       CUDNN_TYPE_INT64,
                                       m_tensor.nDims,
                                       m_tensor.btensor_strA);
        if (status != CUDNN_STATUS_SUCCESS) {
            set_error_and_throw_exception(
                &m_tensor, status, "CUDNN_BACKEND_TENSOR_DESCRIPTOR: SetAttribute CUDNN_ATTR_TENSOR_STRIDES Failed");
            return std::move(m_tensor);
        }
        status = detail::set_attribute(
            m_tensor.pointer->get_backend_descriptor(), CUDNN_ATTR_TENSOR_UNIQUE_ID, CUDNN_TYPE_INT64, 1, &m_tensor.id);
        if (status != CUDNN_STATUS_SUCCESS) {
            set_error_and_throw_exception(
                &m_tensor, status, "CUDNN_BACKEND_TENSOR_DESCRIPTOR: SetAttribute CUDNN_ATTR_TENSOR_UNIQUE_ID Failed");
            return std::move(m_tensor);
        }
        status = detail::set_attribute(m_tensor.pointer->get_backend_descriptor(),
                                       CUDNN_ATTR_TENSOR_BYTE_ALIGNMENT,
                                       CUDNN_TYPE_INT64,
                                       1,
                                       &m_tensor.alignment);
        if (status != CUDNN_STATUS_SUCCESS) {
            set_error_and_throw_exception(
                &m_tensor,
                status,
                "CUDNN_BACKEND_TENSOR_DESCRIPTOR: SetAttribute CUDNN_ATTR_TENSOR_BYTE_ALIGNMENT Failed");
            return std::move(m_tensor);
        }
        if (m_tensor.isVirtual) {
            status = detail::set_attribute(m_tensor.pointer->get_backend_descriptor(),
                                           CUDNN_ATTR_TENSOR_IS_VIRTUAL,
                                           CUDNN_TYPE_BOOLEAN,
                                           1,
                                           &m_tensor.isVirtual);
            if (status != CUDNN_STATUS_SUCCESS) {
                set_error_and_throw_exception(
                    &m_tensor,
                    status,
                    "CUDNN_BACKEND_TENSOR_DESCRIPTOR: SetAttribute CUDNN_ATTR_TENSOR_BYTE_ALIGNMENT Failed");
                return std::move(m_tensor);
            }
        }
        if (m_tensor.isByValue) {
            status = detail::set_attribute(m_tensor.pointer->get_backend_descriptor(),
                                           CUDNN_ATTR_TENSOR_IS_BY_VALUE,
                                           CUDNN_TYPE_BOOLEAN,
                                           1,
                                           &m_tensor.isByValue);
            if (status != CUDNN_STATUS_SUCCESS) {
                set_error_and_throw_exception(
                    &m_tensor,
                    status,
                    "CUDNN_BACKEND_TENSOR_DESCRIPTOR: SetAttribute CUDNN_ATTR_TENSOR_IS_BY_VALUE Failed");
                return std::move(m_tensor);
            }
        }

        if (m_tensor.vectorCount > 1) {
            status = detail::set_attribute(m_tensor.pointer->get_backend_descriptor(),
                                           CUDNN_ATTR_TENSOR_VECTOR_COUNT,
                                           CUDNN_TYPE_INT64,
                                           1,
                                           &m_tensor.vectorCount);
            if (status != CUDNN_STATUS_SUCCESS) {
                set_error_and_throw_exception(
                    &m_tensor,
                    status,
                    "CUDNN_BACKEND_TENSOR_DESCRIPTOR: SetAttribute CUDNN_ATTR_TENSOR_VECTOR_COUNT Failed");
                return std::move(m_tensor);
            }
        }
        if (m_tensor.vectorDimension >= 0) {
            status = detail::set_attribute(m_tensor.pointer->get_backend_descriptor(),
                                           CUDNN_ATTR_TENSOR_VECTORIZED_DIMENSION,
                                           CUDNN_TYPE_INT64,
                                           1,
                                           &m_tensor.vectorDimension);
            if (status != CUDNN_STATUS_SUCCESS) {
                set_error_and_throw_exception(
                    &m_tensor,
                    status,
                    "CUDNN_BACKEND_TENSOR_DESCRIPTOR: SetAttribute CUDNN_ATTR_TENSOR_VECTORIZED_DIMENSION Failed");
                return std::move(m_tensor);
            }
        }

        // Set ragged offset descriptor
#if (CUDNN_VERSION >= 8900)
        NV_CUDNN_FE_DYNAMIC_CHECK_BACKEND_DESCRIPTOR(8900,
                                                     m_tensor,
                                                     "CUDNN_BACKEND_TENSOR_DESCRIPTOR: SetAttribute "
                                                     "CUDNN_ATTR_TENSOR_RAGGED_OFFSET_DESC requires cudnn version 8.9");
        if (m_tensor.raggedOffset != nullptr) {
            std::vector<cudnnBackendDescriptor_t> backendRaggedOffset;
            backendRaggedOffset.push_back(m_tensor.raggedOffset.get()->pointer->get_backend_descriptor());
            status = detail::set_attribute(m_tensor.pointer->get_backend_descriptor(),
                                           CUDNN_ATTR_TENSOR_RAGGED_OFFSET_DESC,
                                           CUDNN_TYPE_BACKEND_DESCRIPTOR,
                                           static_cast<int64_t>(backendRaggedOffset.size()),
                                           backendRaggedOffset.data());
            if (status != CUDNN_STATUS_SUCCESS) {
                set_error_and_throw_exception(
                    &m_tensor,
                    status,
                    "CUDNN_BACKEND_TENSOR_DESCRIPTOR: SetAttribute CUDNN_ATTR_TENSOR_RAGGED_OFFSET_DESC Failed");
                return std::move(m_tensor);
            }
        }
#endif

        // Set the reorder_type
        if (m_tensor.reorder_type != cudnn_frontend::TensorReordering_t::NONE) {
            cudnnBackendTensorReordering_t cudnn_reordering_type;
            status = detail::convert_to_cudnn_type(m_tensor.reorder_type, cudnn_reordering_type);
            if (status != CUDNN_STATUS_SUCCESS) {
                set_error_and_throw_exception(
                    &m_tensor,
                    status,
                    "CUDNN_BACKEND_TENSOR_DESCRIPTOR: SetAttribute CUDNN_ATTR_TENSOR_REORDERING_MODE Failed");
                return std::move(m_tensor);
            }
            status = detail::set_attribute(m_tensor.pointer->get_backend_descriptor(),
                                           CUDNN_ATTR_TENSOR_REORDERING_MODE,
                                           CUDNN_TYPE_TENSOR_REORDERING_MODE,
                                           1,
                                           &m_tensor.reorder_type);
            if (status != CUDNN_STATUS_SUCCESS) {
                set_error_and_throw_exception(
                    &m_tensor,
                    status,
                    "CUDNN_BACKEND_TENSOR_DESCRIPTOR: SetAttribute CUDNN_ATTR_TENSOR_REORDERING_MODE Failed");
                return std::move(m_tensor);
            }
        }

        // Set compile-time constant value (if present); CUDNN_ATTR_TENSOR_CONSTANT_VALUE is cuDNN 9.22.0+
#if (CUDNN_VERSION >= 92200)
        if (m_tensor.hasConstValue) {
            NV_CUDNN_FE_DYNAMIC_CHECK_BACKEND_DESCRIPTOR(
                92200,
                m_tensor,
                "CUDNN_BACKEND_TENSOR_DESCRIPTOR: SetAttribute "
                "CUDNN_ATTR_TENSOR_CONSTANT_VALUE requires cudnn version 9.22.0");
            void *value_ptr = m_tensor.constValueBytes.data();
            status          = detail::set_attribute(m_tensor.pointer->get_backend_descriptor(),
                                           CUDNN_ATTR_TENSOR_CONSTANT_VALUE,
                                           CUDNN_TYPE_VOID_PTR,
                                           1,
                                           &value_ptr);
            if (status != CUDNN_STATUS_SUCCESS) {
                set_error_and_throw_exception(
                    &m_tensor,
                    status,
                    "CUDNN_BACKEND_TENSOR_DESCRIPTOR: SetAttribute CUDNN_ATTR_TENSOR_CONSTANT_VALUE Failed");
                return std::move(m_tensor);
            }
        }
#else
        if (m_tensor.hasConstValue) {
            set_error_and_throw_exception(
                &m_tensor,
                CUDNN_STATUS_INVALID_VALUE,
                "CUDNN_BACKEND_TENSOR_DESCRIPTOR: CUDNN_ATTR_TENSOR_CONSTANT_VALUE requires cudnn version 9.22.0");
            return std::move(m_tensor);
        }
#endif  // CUDNN_VERSION >= 92200

        // Finalizing the descriptor
        status = detail::finalize(m_tensor.pointer->get_backend_descriptor());
        if (status != CUDNN_STATUS_SUCCESS) {
            set_error_and_throw_exception(&m_tensor, status, "CUDNN_BACKEND_TENSOR_DESCRIPTOR cudnnFinalize failed");
            return std::move(m_tensor);
        }
        CUDNN_FE_LOG_LABEL_ENDL(m_tensor);
        return std::move(m_tensor);
    }

    explicit TensorBuilder_v8()                = default;
    ~TensorBuilder_v8()                        = default;
    TensorBuilder_v8(TensorBuilder_v8 &&)      = delete;
    TensorBuilder_v8(TensorBuilder_v8 const &) = delete;
    TensorBuilder_v8 &
    operator=(TensorBuilder_v8 const &) = delete;

   private:
    Tensor_v8 m_tensor;  //! Tensor built by the TensorBuilder class.
};

using Tensor        = Tensor_v8;
using TensorBuilder = TensorBuilder_v8;

}  // namespace cudnn_frontend
