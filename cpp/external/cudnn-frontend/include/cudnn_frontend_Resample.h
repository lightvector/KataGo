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
#include <array>
#include <functional>
#include <memory>
#include <sstream>
#include <utility>

#include "cudnn_frontend_utils.h"

namespace cudnn_frontend {
namespace graph {
class ResampleNode;
}
}  // namespace cudnn_frontend

namespace cudnn_frontend {

///
/// Resample Descriptor Class
/// This class tells the properties of the Resample operation
/// Properties:
///
/// Use ResampleDescBuilder_v8 to build this class.
/// Describe returns a string describing the Resample operation
///
class ResampleDesc_v8 : public BackendDescriptor {
   public:
    friend class ResampleDescBuilder_v8;
    friend class graph::ResampleNode;
    std::string
    describe() const override {
        std::stringstream ss;
        char sep = ',';
#ifndef CUDNN_FRONTEND_SKIP_JSON_LIB
        ss << "CUDNN_BACKEND_RESAMPLE_DESCRIPTOR: " << "Compute Type: " << json{computeType}
           << ", Resample Mode: " << json{resample_mode} << ", Spatial Dimensions: " << spatialDim
           << ", Nan Propagation: " << std::to_string(nanOpt) << ", Padding Mode: " << json{padding_mode};
#else
        ss << "CUDNN_BACKEND_RESAMPLE_DESCRIPTOR: " << "Compute Type: " << int(computeType)
           << ", Resample Mode: " << int(resample_mode) << ", Spatial Dimensions: " << spatialDim
           << ", Nan Propagation: " << std::to_string(nanOpt) << ", Padding Mode: " << int(padding_mode);
#endif

        ss << ", WindowDim: [";
        for (auto i = 0; i < spatialDim; i++) {
            ss << '(' << windowDim[i].numerator << sep << windowDim[i].denominator << ')' << sep;
        }
        ss << "]";
        ss << ", prePadding: [";
        for (auto i = 0; i < spatialDim; i++) {
            ss << '(' << prePadding[i].numerator << sep << prePadding[i].denominator << ')' << sep;
        }
        ss << "]";
        ss << ", postPadding: [";
        for (auto i = 0; i < spatialDim; i++) {
            ss << '(' << postPadding[i].numerator << sep << postPadding[i].denominator << ')' << sep;
        }
        ss << "]";
        ss << ", stride: [ ";
        for (auto i = 0; i < spatialDim; i++) {
            ss << '(' << stride[i].numerator << sep << stride[i].denominator << ')' << sep;
        }
        ss << "]";
        return ss.str();
    }

    ResampleDesc_v8(ResampleDesc_v8 &&from) = default;
    ResampleDesc_v8 &
    operator=(ResampleDesc_v8 &&) = default;

    ~ResampleDesc_v8() = default;

    /** @defgroup ResampleDescBuilder_v8
     *  Get individual property of ResampleDesc_v8 class
     *  @{
     */

    DataType_t
    getComputeType() const {
        return computeType;
    }

    int64_t
    getSpatialDimCount() const {
        return spatialDim;
    }

    cudnnNanPropagation_t
    getNanOpt() const {
        return nanOpt;
    }

    ResampleMode_t
    getMode() const {
        return resample_mode;
    }

    PaddingMode_t
    getPaddingMode() const {
        return padding_mode;
    }

    cudnnFraction_t const *
    getSpatialStride() const {
        return stride;
    }

    cudnnFraction_t const *
    getPrePadding() const {
        return prePadding;
    }

    cudnnFraction_t const *
    getPostPadding() const {
        return postPadding;
    }

    cudnnFraction_t const *
    getWindowDim() const {
        return windowDim;
    }
    /** @} */

   private:
    ResampleDesc_v8()                        = default;
    ResampleDesc_v8(ResampleDesc_v8 const &) = delete;
    ResampleDesc_v8 &
    operator=(ResampleDesc_v8 const &) = delete;

    // default values for attributes
    DataType_t computeType       = DataType_t::FLOAT;
    cudnnNanPropagation_t nanOpt = CUDNN_PROPAGATE_NAN;
    ResampleMode_t resample_mode = ResampleMode_t::NOT_SET;
    PaddingMode_t padding_mode   = PaddingMode_t::NOT_SET;

    int64_t spatialDim = 0;

    // Shape attributes
    cudnnFraction_t windowDim[CUDNN_DIM_MAX]   = {{0, 1}, {0, 1}};
    cudnnFraction_t prePadding[CUDNN_DIM_MAX]  = {{0, 1}, {0, 1}};
    cudnnFraction_t postPadding[CUDNN_DIM_MAX] = {{0, 1}, {0, 1}};
    cudnnFraction_t stride[CUDNN_DIM_MAX]      = {{0, 1}, {0, 1}};
};

///
/// ResampleDescBuilder_v8 Class
/// Helper class used to build ResampleDesc_v8 class
class ResampleDescBuilder_v8 {
   public:
    /** @defgroup ResampleDescBuilder_v8
     *  Set individual property of ResampleDesc_v8 class
     *  @{
     */
    //! Set compute type for the Resample Descriptor
    auto
    setComputeType(DataType_t data_type) -> ResampleDescBuilder_v8 & {
        m_resampleDesc.computeType = data_type;
        return *this;
    }
    // To be deprecated in v1.0.
    auto
    setComputeType(cudnnDataType_t data_type_) -> ResampleDescBuilder_v8 & {
        m_resampleDesc.computeType = detail::convert_from_cudnn_type(data_type_);
        return *this;
    }

    //! Set nan propagation mode for the Resample Operation
    auto
    setNanPropagation(cudnnNanPropagation_t nanOpt_) -> ResampleDescBuilder_v8 & {
        m_resampleDesc.nanOpt = nanOpt_;
        return *this;
    }

    //! (Overloaded) Set post padding for the Resample Operation with cudnnFraction_t
    auto
    setPostPadding(int64_t count, cudnnFraction_t const *arr) -> ResampleDescBuilder_v8 & {
        // TODO: check the provided array count against the stored spatial dimension count.
        std::copy(arr, arr + count, m_resampleDesc.postPadding);
        return *this;
    }

    //! (Overloaded) Set pre padding for the Resample Operation with cudnnFraction_t
    auto
    setPrePadding(int64_t count, cudnnFraction_t const *arr) -> ResampleDescBuilder_v8 & {
        // TODO: check the provided array count against the stored spatial dimension count.
        std::copy(arr, arr + count, m_resampleDesc.prePadding);
        return *this;
    }

    //! (Overloaded) Set stride for the Resample Operation with cudnnFraction_t
    auto
    setSpatialStride(int64_t count, cudnnFraction_t const *arr) -> ResampleDescBuilder_v8 & {
        // TODO: check the provided array count against the stored spatial dimension count.
        std::copy(arr, arr + count, m_resampleDesc.stride);
        return *this;
    }

    //! Set resample mode for the Resample Operation
    // To be deprecated. Please use setResampleMode(cudnn_frontend::ResampleMode_t).
    auto
    setResampleMode(cudnnResampleMode_t const mode_) -> ResampleDescBuilder_v8 & {
        detail::convert_from_cudnn_type(mode_, m_resampleDesc.resample_mode);
        return *this;
    }

    //! (Overloaded) Set window dim for the Resample Operation with cudnnFraction_t
    auto
    setSpatialDim(int64_t count, cudnnFraction_t const *arr) -> ResampleDescBuilder_v8 & {
        // TODO: check the provided array count against the stored spatial dimension count.
        m_resampleDesc.spatialDim = count;
        std::copy(arr, arr + count, m_resampleDesc.windowDim);
        return *this;
    }

    //! Set padding mode for the Resample Operation
    // To be deprecated. Please use setPaddingMode(cudnn_frontend::PaddingMode_t).
    auto
    setPaddingMode(cudnnPaddingMode_t const padding_mode) -> ResampleDescBuilder_v8 & {
        detail::convert_from_cudnn_type(padding_mode, m_resampleDesc.padding_mode);
        return *this;
    }

    //! Set padding mode for the Resample Operation
    auto
    setPaddingMode(PaddingMode_t const padding_mode) -> ResampleDescBuilder_v8 & {
        m_resampleDesc.padding_mode = padding_mode;
        return *this;
    }

    //! Set resample mode for the Resample Operation
    auto
    setResampleMode(ResampleMode_t const mode) -> ResampleDescBuilder_v8 & {
        m_resampleDesc.resample_mode = mode;
        return *this;
    }

    //! (Overloaded) Set post padding for the Resample Operation with int64_t
    auto
    setPostPadding(int64_t count, int64_t const *arr) -> ResampleDescBuilder_v8 & {
        // TODO: check the provided array count against the stored spatial dimension count.
        for (int i = 0; i < count; i++) {
            m_resampleDesc.postPadding[i].numerator   = arr[i];
            m_resampleDesc.postPadding[i].denominator = 1;
        }
        return *this;
    }

    //! (Overloaded) Set pre padding for the Resample Operation with int64_t
    auto
    setPrePadding(int64_t count, int64_t const *arr) -> ResampleDescBuilder_v8 & {
        // TODO: check the provided array count against the stored spatial dimension count.
        for (int i = 0; i < count; i++) {
            m_resampleDesc.prePadding[i].numerator   = arr[i];
            m_resampleDesc.prePadding[i].denominator = 1;
        }
        return *this;
    }

    //! (Overloaded) Set stride for the Resample Operation with int64_t
    auto
    setSpatialStride(int64_t count, int64_t const *arr) -> ResampleDescBuilder_v8 & {
        // TODO: check the provided array count against the stored spatial dimension count.
        for (int i = 0; i < count; i++) {
            m_resampleDesc.stride[i].numerator   = arr[i];
            m_resampleDesc.stride[i].denominator = 1;
        }
        return *this;
    }

    //! (Overloaded) Set window dim for the Resample Operation with int64_t
    auto
    setSpatialDim(int64_t count, int64_t const *arr) -> ResampleDescBuilder_v8 & {
        // TODO: check the provided array count against the stored spatial dimension count.
        m_resampleDesc.spatialDim = count;
        for (int i = 0; i < count; i++) {
            m_resampleDesc.windowDim[i].numerator   = arr[i];
            m_resampleDesc.windowDim[i].denominator = 1;
        }
        return *this;
    }

    /** @} */

    //! constructs the ResampleDesc_v8 by calling the cudnn API
    //! Throws the appropriate error message
    ResampleDesc_v8 &&
    build() {
        // Sanity check if non-default fields have been set correctly.
        if (m_resampleDesc.spatialDim < 0) {
            set_error_and_throw_exception(&m_resampleDesc,
                                          CUDNN_STATUS_BAD_PARAM,
                                          "CUDNN_BACKEND_RESAMPLE_DESCRIPTOR: Check and Set the spatialDim field");
            return std::move(m_resampleDesc);
        };

        // Create a descriptor. Memory allocation happens here.
        auto status = m_resampleDesc.initialize_managed_backend_pointer(CUDNN_BACKEND_RESAMPLE_DESCRIPTOR);
        if (status != CUDNN_STATUS_SUCCESS) {
            set_error_and_throw_exception(
                &m_resampleDesc, status, "CUDNN_BACKEND_RESAMPLE_DESCRIPTOR: cudnnCreate Failed");
            return std::move(m_resampleDesc);
        }

        // Once Created lets set the descriptor parameters.
        ::cudnnResampleMode_t cudnn_resample_mode;
        status = detail::convert_to_cudnn_type(m_resampleDesc.resample_mode, cudnn_resample_mode);
        if (status != CUDNN_STATUS_SUCCESS) {
            set_error_and_throw_exception(
                &m_resampleDesc,
                status,
                "CUDNN_BACKEND_RESAMPLE_DESCRIPTOR: SetAttribute CUDNN_ATTR_RESAMPLE_MODE Failed");
            return std::move(m_resampleDesc);
        }
        status = detail::set_attribute(m_resampleDesc.pointer->get_backend_descriptor(),
                                       CUDNN_ATTR_RESAMPLE_MODE,
                                       CUDNN_TYPE_RESAMPLE_MODE,
                                       1,
                                       &cudnn_resample_mode);
        if (status != CUDNN_STATUS_SUCCESS) {
            set_error_and_throw_exception(
                &m_resampleDesc,
                status,
                "CUDNN_BACKEND_RESAMPLE_DESCRIPTOR: SetAttribute CUDNN_ATTR_RESAMPLE_MODE Failed");
            return std::move(m_resampleDesc);
        }

        cudnnDataType_t cudnn_data_type;
        status = detail::convert_to_cudnn_type(m_resampleDesc.computeType, cudnn_data_type);
        if (status != CUDNN_STATUS_SUCCESS) {
            set_error_and_throw_exception(
                &m_resampleDesc,
                status,
                "CUDNN_BACKEND_RESAMPLE_DESCRIPTOR: SetAttribute CUDNN_ATTR_RESAMPLE_COMP_TYPE Failed");
            return std::move(m_resampleDesc);
        }
        status = detail::set_attribute(m_resampleDesc.pointer->get_backend_descriptor(),
                                       CUDNN_ATTR_RESAMPLE_COMP_TYPE,
                                       CUDNN_TYPE_DATA_TYPE,
                                       1,
                                       &cudnn_data_type);
        if (status != CUDNN_STATUS_SUCCESS) {
            set_error_and_throw_exception(
                &m_resampleDesc,
                status,
                "CUDNN_BACKEND_RESAMPLE_DESCRIPTOR: SetAttribute CUDNN_ATTR_RESAMPLE_COMP_TYPE Failed");
            return std::move(m_resampleDesc);
        }

        status = detail::set_attribute(m_resampleDesc.pointer->get_backend_descriptor(),
                                       CUDNN_ATTR_RESAMPLE_NAN_PROPAGATION,
                                       CUDNN_TYPE_NAN_PROPOGATION,
                                       1,
                                       &(m_resampleDesc.nanOpt));
        if (status != CUDNN_STATUS_SUCCESS) {
            set_error_and_throw_exception(
                &m_resampleDesc,
                status,
                "CUDNN_BACKEND_RESAMPLE_DESCRIPTOR: SetAttribute CUDNN_ATTR_RESAMPLE_NAN_PROPAGATION Failed");
            return std::move(m_resampleDesc);
        }

        cudnnPaddingMode_t cudnn_padding_mode;
        status = detail::convert_to_cudnn_type(m_resampleDesc.padding_mode, cudnn_padding_mode);
        if (status != CUDNN_STATUS_SUCCESS) {
            set_error_and_throw_exception(
                &m_resampleDesc,
                status,
                "CUDNN_BACKEND_RESAMPLE_DESCRIPTOR: SetAttribute CUDNN_ATTR_RESAMPLE_PADDING_MODE Failed");
            return std::move(m_resampleDesc);
        }
        status = detail::set_attribute(m_resampleDesc.pointer->get_backend_descriptor(),
                                       CUDNN_ATTR_RESAMPLE_PADDING_MODE,
                                       CUDNN_TYPE_PADDING_MODE,
                                       1,
                                       &cudnn_padding_mode);
        if (status != CUDNN_STATUS_SUCCESS) {
            set_error_and_throw_exception(
                &m_resampleDesc,
                status,
                "CUDNN_BACKEND_RESAMPLE_DESCRIPTOR: SetAttribute CUDNN_ATTR_RESAMPLE_PADDING_MODE Failed");
            return std::move(m_resampleDesc);
        }

        status = detail::set_attribute(m_resampleDesc.pointer->get_backend_descriptor(),
                                       CUDNN_ATTR_RESAMPLE_SPATIAL_DIMS,
                                       CUDNN_TYPE_INT64,
                                       1,
                                       &(m_resampleDesc.spatialDim));
        if (status != CUDNN_STATUS_SUCCESS) {
            set_error_and_throw_exception(
                &m_resampleDesc,
                status,
                "CUDNN_BACKEND_RESAMPLE_DESCRIPTOR: SetAttribute CUDNN_ATTR_RESAMPLE_SPATIAL_DIMS Failed");
            return std::move(m_resampleDesc);
        }

        status = detail::set_attribute(m_resampleDesc.pointer->get_backend_descriptor(),
                                       CUDNN_ATTR_RESAMPLE_WINDOW_DIMS,
                                       CUDNN_TYPE_FRACTION,
                                       m_resampleDesc.spatialDim,
                                       m_resampleDesc.windowDim);
        if (status != CUDNN_STATUS_SUCCESS) {
            set_error_and_throw_exception(
                &m_resampleDesc,
                status,
                "CUDNN_BACKEND_RESAMPLE_DESCRIPTOR: SetAttribute CUDNN_ATTR_RESAMPLE_WINDOW_DIMS Failed");
            return std::move(m_resampleDesc);
        }

        status = detail::set_attribute(m_resampleDesc.pointer->get_backend_descriptor(),
                                       CUDNN_ATTR_RESAMPLE_PRE_PADDINGS,
                                       CUDNN_TYPE_FRACTION,
                                       m_resampleDesc.spatialDim,
                                       m_resampleDesc.prePadding);
        if (status != CUDNN_STATUS_SUCCESS) {
            set_error_and_throw_exception(
                &m_resampleDesc,
                status,
                "CUDNN_BACKEND_RESAMPLE_DESCRIPTOR: SetAttribute CUDNN_ATTR_RESAMPLE_PRE_PADDINGS Failed");
            return std::move(m_resampleDesc);
        }

        status = detail::set_attribute(m_resampleDesc.pointer->get_backend_descriptor(),
                                       CUDNN_ATTR_RESAMPLE_POST_PADDINGS,
                                       CUDNN_TYPE_FRACTION,
                                       m_resampleDesc.spatialDim,
                                       m_resampleDesc.postPadding);
        if (status != CUDNN_STATUS_SUCCESS) {
            set_error_and_throw_exception(
                &m_resampleDesc,
                status,
                "CUDNN_BACKEND_RESAMPLE_DESCRIPTOR: SetAttribute CUDNN_ATTR_RESAMPLE_POST_PADDINGS Failed");
            return std::move(m_resampleDesc);
        }

        status = detail::set_attribute(m_resampleDesc.pointer->get_backend_descriptor(),
                                       CUDNN_ATTR_RESAMPLE_STRIDES,
                                       CUDNN_TYPE_FRACTION,
                                       m_resampleDesc.spatialDim,
                                       m_resampleDesc.stride);
        if (status != CUDNN_STATUS_SUCCESS) {
            set_error_and_throw_exception(
                &m_resampleDesc,
                status,
                "CUDNN_BACKEND_RESAMPLE_DESCRIPTOR: SetAttribute CUDNN_ATTR_RESAMPLE_STRIDES Failed");
            return std::move(m_resampleDesc);
        }

        // Finalizing the descriptor
        status = detail::finalize(m_resampleDesc.pointer->get_backend_descriptor());
        if (status != CUDNN_STATUS_SUCCESS) {
            set_error_and_throw_exception(
                &m_resampleDesc, status, "CUDNN_BACKEND_RESAMPLE_DESCRIPTOR: cudnnFinalize Failed");
            return std::move(m_resampleDesc);
        }
        CUDNN_FE_LOG_LABEL_ENDL(m_resampleDesc);
        return std::move(m_resampleDesc);
    }

    explicit ResampleDescBuilder_v8()                      = default;
    ~ResampleDescBuilder_v8()                              = default;
    ResampleDescBuilder_v8(ResampleDescBuilder_v8 &&)      = delete;
    ResampleDescBuilder_v8(ResampleDescBuilder_v8 const &) = delete;
    ResampleDescBuilder_v8 &
    operator=(ResampleDescBuilder_v8 const &) = delete;

   private:
    ResampleDesc_v8 m_resampleDesc;
};
}  // namespace cudnn_frontend
