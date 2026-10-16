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
class PointwiseNode;
class MatmulNode;
class ConvolutionNode;
class DgradNode;
class WgradNode;
class LayerNormNode;
class BatchNormNode;
class BatchnormInferenceNode;
class RMSNormNode;
class DRMSNormNode;
class InstanceNormNode;
class DINNode;
class DLNNode;
class DBNNode;
class DBNWeightNode;
class BatchNormFinalizeNode;
class GenstatsNode;
class ReshapeNode;
class ResampleNode;
class RngNode;
class PagedCacheLoadNode;
}  // namespace graph
}  // namespace cudnn_frontend

#include <array>
#include <cstddef>
#include <functional>
#include <memory>
#include <sstream>
#include <utility>

#include "cudnn_frontend_ConvDesc.h"
#include "cudnn_frontend_PointWiseDesc.h"
#include "cudnn_frontend_MatMulDesc.h"
#include "cudnn_frontend_ReductionDesc.h"
#include "cudnn_frontend_Resample.h"
#include "cudnn_frontend_Rng.h"
#include "cudnn_frontend_Tensor.h"
#include "cudnn_frontend_utils.h"

namespace cudnn_frontend {
///
/// Operation_v8 Class
/// This class has the properties of the operation
/// Properties:
///    - xDesc
///    - yDesc
///    - wdesc
///    - bdesc
///    - tDesc
///    - dydesc
///    - dxdesc
///    - cdesc
///    - amatdesc
///    - bmatdesc
///    - cmatdesc
///    - moverridedesc
///    - noverridedesc
///    - koverridedesc
///    - pwdesc
///    - matmuldesc
///    - reductiondesc
///    - flagdesc
///    - inputDescs
///    - alpha
///    - beta
///    - alpha2
///    - axis
///    - inplaceIndex
///    - mode
///    - value
///
/// Use OperationBuilder_v8 to build this class.
/// Describe returns a string describing the convolution operation
///
class Operation_v8 : public BackendDescriptor {
   public:
    friend class OperationBuilder_v8;
    friend class cudnn_frontend::graph::ReductionNode;
    friend class cudnn_frontend::graph::PointwiseNode;
    friend class cudnn_frontend::graph::MatmulNode;
    friend class cudnn_frontend::graph::ConvolutionNode;
    friend class cudnn_frontend::graph::DgradNode;
    friend class cudnn_frontend::graph::WgradNode;
    friend class cudnn_frontend::graph::LayerNormNode;
    friend class cudnn_frontend::graph::BatchNormNode;
    friend class cudnn_frontend::graph::BatchnormInferenceNode;
    friend class cudnn_frontend::graph::RMSNormNode;
    friend class cudnn_frontend::graph::DRMSNormNode;
    friend class cudnn_frontend::graph::InstanceNormNode;
    friend class cudnn_frontend::graph::DINNode;
    friend class cudnn_frontend::graph::DLNNode;
    friend class cudnn_frontend::graph::DBNNode;
    friend class cudnn_frontend::graph::DBNWeightNode;
    friend class cudnn_frontend::graph::BatchNormFinalizeNode;
    friend class cudnn_frontend::graph::GenstatsNode;
    friend class cudnn_frontend::graph::ReshapeNode;
    friend class cudnn_frontend::graph::ResampleNode;
    friend class cudnn_frontend::graph::RngNode;
    friend class cudnn_frontend::graph::PagedCacheLoadNode;
    std::string
    describe() const override {
        std::stringstream ss;
        ss << "CUDNN_BACKEND_OPERATION :" << " OpMode: " << op_mode;
        ss << std::hex << " X " << xdesc;
        ss << std::hex << " Y " << ydesc;
        ss << std::hex << " W " << wdesc;
        ss << std::hex << " B " << bdesc;
        ss << std::hex << " T " << tdesc;
        ss << std::hex << " DW " << dwdesc;
        ss << std::hex << " DY " << dydesc;
        ss << std::hex << " DX " << dxdesc;
        ss << std::hex << " C " << cdesc;
        ss << std::hex << " A Mtrix " << amatdesc;
        ss << std::hex << " B Mtrix " << bmatdesc;
        ss << std::hex << " C Mtrix " << cmatdesc;
        ss << std::hex << " P " << pwdesc;
        ss << std::hex << " MatMul " << matmuldesc;
        ss << std::hex << " Reduction " << reductiondesc;
        ss << std::dec << " alphabetaType " << alphabetaType;
        ss << " Alpha: " << alpha_s << " " << alpha_d;
        ss << " Alpha2: " << alpha2_s << " " << alpha2_d;
        ss << " Beta: " << beta_s << " " << beta_d;
        return ss.str();
    }

    Operation_v8(Operation_v8 &&from) = default;
    Operation_v8 &
    operator=(Operation_v8 &&from) = default;

    // Will be deprecated. Do Not use
    ManagedOpaqueDescriptor
    getOutputTensor() {
        return (op_mode == DescriptorType_t::OPERATION_MATMUL_DESCRIPTOR) ? cmatdesc : ydesc;
    }

    std::string const &
    getTag() const {
        return operationTag;
    }

    feature_vector_t
    getFeatureVector() const {
        return feature_vector;
    }

    ~Operation_v8() = default;

   private:
    Operation_v8()                     = default;
    Operation_v8(Operation_v8 const &) = delete;
    Operation_v8 &
    operator=(Operation_v8 const &) = delete;

    DescriptorType_t op_mode = DescriptorType_t::NOT_SET;

    ManagedOpaqueDescriptor xdesc              = nullptr;
    ManagedOpaqueDescriptor ydesc              = nullptr;
    ManagedOpaqueDescriptor wdesc              = nullptr;
    ManagedOpaqueDescriptor bdesc              = nullptr;
    ManagedOpaqueDescriptor tdesc              = nullptr;
    ManagedOpaqueDescriptor dydesc             = nullptr;
    ManagedOpaqueDescriptor dxdesc             = nullptr;
    ManagedOpaqueDescriptor dwdesc             = nullptr;
    ManagedOpaqueDescriptor cdesc              = nullptr;
    ManagedOpaqueDescriptor resampledesc       = nullptr;
    ManagedOpaqueDescriptor rngdesc            = nullptr;
    ManagedOpaqueDescriptor amatdesc           = nullptr;
    ManagedOpaqueDescriptor bmatdesc           = nullptr;
    ManagedOpaqueDescriptor cmatdesc           = nullptr;
    ManagedOpaqueDescriptor moverridedesc      = nullptr;
    ManagedOpaqueDescriptor noverridedesc      = nullptr;
    ManagedOpaqueDescriptor koverridedesc      = nullptr;
    ManagedOpaqueDescriptor pwdesc             = nullptr;
    ManagedOpaqueDescriptor matmuldesc         = nullptr;
    ManagedOpaqueDescriptor reductiondesc      = nullptr;
    ManagedOpaqueDescriptor sumdesc            = nullptr;
    ManagedOpaqueDescriptor sqsumdesc          = nullptr;
    ManagedOpaqueDescriptor scaledesc          = nullptr;
    ManagedOpaqueDescriptor biasdesc           = nullptr;
    ManagedOpaqueDescriptor dscaledesc         = nullptr;
    ManagedOpaqueDescriptor dbiasdesc          = nullptr;
    ManagedOpaqueDescriptor eqscaledesc        = nullptr;
    ManagedOpaqueDescriptor eqscaledesc1       = nullptr;
    ManagedOpaqueDescriptor eqbiasdesc         = nullptr;
    ManagedOpaqueDescriptor prevMeandesc       = nullptr;
    ManagedOpaqueDescriptor prevVardesc        = nullptr;
    ManagedOpaqueDescriptor nextMeandesc       = nullptr;
    ManagedOpaqueDescriptor nextVardesc        = nullptr;
    ManagedOpaqueDescriptor savedMeandesc      = nullptr;
    ManagedOpaqueDescriptor savedInVardesc     = nullptr;
    ManagedOpaqueDescriptor accumCountdesc     = nullptr;
    ManagedOpaqueDescriptor epsilondesc        = nullptr;
    ManagedOpaqueDescriptor expDecayFactordesc = nullptr;
    ManagedOpaqueDescriptor idxdesc            = nullptr;
    ManagedOpaqueDescriptor offsetdesc         = nullptr;
    ManagedOpaqueDescriptor seeddesc           = nullptr;
    ManagedOpaqueDescriptor containerdesc      = nullptr;
    ManagedOpaqueDescriptor pageTabledesc      = nullptr;
    ManagedOpaqueDescriptor sequencedesc       = nullptr;
    std::vector<ManagedOpaqueDescriptor> peerStatdescs;

    cudnnBackendAttributeType_t alphabetaType = CUDNN_TYPE_FLOAT;
    cudnnDataType_t compute_type              = CUDNN_DATA_FLOAT;
    cudnnGenStatsMode_t genstats_mode         = CUDNN_GENSTATS_SUM_SQSUM;
    cudnnBnFinalizeStatsMode_t bn_stats_mode  = CUDNN_BN_FINALIZE_STATISTICS_TRAINING;

    NormFwdPhase_t norm_fwd_phase;
    NormMode_t norm_mode;
    ReshapeMode_t reshape_mode = ReshapeMode_t::VIEW_ONLY;

    float alpha_s = 1.0f, beta_s = .0f, alpha2_s = 1.0f;
    double alpha_d = 1.0, beta_d = 0.0, alpha2_d = 1.0;
    int64_t pointwise_port_count        = -1;
    PointwiseMode_t pointwise_mode      = PointwiseMode_t::NOT_SET;
    bool is_pointwise_activation_fwd_op = false;
    bool is_pointwise_identity_op       = false;
    bool is_pointwise_activation_bwd_op = false;
    bool is_pointwise_math_op           = false;
    std::string operationTag;
    feature_vector_t feature_vector;
    int64_t seed = 0;
};

///
/// OperationBuilder_v8 Class
/// Helper class used to build Operation_v8 class

class OperationBuilder_v8 {
   private:
    Operation_v8 m_operation;
    bool is_convolution_op      = false;
    bool is_pointwise_op        = false;
    bool is_matmul_op           = false;
    bool is_reduction_op        = false;
    bool is_genstats_op         = false;
    bool is_bn_finalize_op      = false;
    bool is_resample_fwd_op     = false;
    bool is_resample_bwd_op     = false;
    bool is_norm_forward_op     = false;
    bool is_norm_backward_op    = false;
    bool is_bn_bwd_weight       = false;
    bool is_rng_op              = false;
    bool is_reshape_op          = false;
    bool is_paged_cache_load_op = false;

    using Message_t = const char *;

    int64_t xTensor_dimA[CUDNN_DIM_MAX + 1];
    int64_t xTensor_strA[CUDNN_DIM_MAX + 1];
    int64_t wTensor_dimA[CUDNN_DIM_MAX + 1];
    int64_t wTensor_strA[CUDNN_DIM_MAX + 1];
    int64_t yTensor_dimA[CUDNN_DIM_MAX + 1];
    int64_t yTensor_strA[CUDNN_DIM_MAX + 1];
    int64_t idxTensor_dimA[CUDNN_DIM_MAX + 1];
    int64_t idxTensor_strA[CUDNN_DIM_MAX + 1];

    bool is2D = true;

    int64_t conv_padding[CUDNN_DIM_MAX];
    int64_t conv_dilation[CUDNN_DIM_MAX];
    int64_t conv_stride[CUDNN_DIM_MAX];
    int64_t mode;
    int64_t xType, yType, wType, cType, idxType /* compute_precision */;

    int64_t tensor_dims = 0;

    Operation_v8 &&
    build_reduction_op() {
        m_operation.operationTag = "Reduction";
        auto status              = detail::set_attribute(m_operation.pointer->get_backend_descriptor(),
                                            CUDNN_ATTR_OPERATION_REDUCTION_DESC,
                                            CUDNN_TYPE_BACKEND_DESCRIPTOR,
                                            1,
                                            &(m_operation.reductiondesc->get_backend_descriptor()));

        if (status != CUDNN_STATUS_SUCCESS) {
            set_error_and_throw_exception(
                &m_operation,
                status,
                "CUDNN_BACKEND_OPERATION: SetAttribute CUDNN_ATTR_OPERATION_REDUCTION_DESC Failed");
            return std::move(m_operation);
        }
        status = detail::set_attribute(m_operation.pointer->get_backend_descriptor(),
                                       CUDNN_ATTR_OPERATION_REDUCTION_XDESC,
                                       CUDNN_TYPE_BACKEND_DESCRIPTOR,
                                       1,
                                       &(m_operation.xdesc->get_backend_descriptor()));
        if (status != CUDNN_STATUS_SUCCESS) {
            set_error_and_throw_exception(
                &m_operation,
                status,
                "CUDNN_BACKEND_OPERATION: SetAttribute CUDNN_ATTR_OPERATION_REDUCTION_XDESC Failed");
            return std::move(m_operation);
        }
        status = detail::set_attribute(m_operation.pointer->get_backend_descriptor(),
                                       CUDNN_ATTR_OPERATION_REDUCTION_YDESC,
                                       CUDNN_TYPE_BACKEND_DESCRIPTOR,
                                       1,
                                       &(m_operation.ydesc->get_backend_descriptor()));
        if (status != CUDNN_STATUS_SUCCESS) {
            set_error_and_throw_exception(
                &m_operation,
                status,
                "CUDNN_BACKEND_OPERATION: SetAttribute CUDNN_ATTR_OPERATION_REDUCTION_YDESC Failed");
            return std::move(m_operation);
        }
        status = detail::finalize(m_operation.pointer->get_backend_descriptor());
        if (status != CUDNN_STATUS_SUCCESS) {
            set_error_and_throw_exception(&m_operation, status, "CUDNN_BACKEND_OPERATION: cudnnFinalize Failed");
            return std::move(m_operation);
        }
        return std::move(m_operation);
    }

    Operation_v8 &&
    build_matmul_op() {
        m_operation.operationTag = "Matmul";
        auto status              = CUDNN_STATUS_SUCCESS;
        status                   = detail::set_attribute(m_operation.pointer->get_backend_descriptor(),
                                       CUDNN_ATTR_OPERATION_MATMUL_ADESC,
                                       CUDNN_TYPE_BACKEND_DESCRIPTOR,
                                       1,
                                       &(m_operation.amatdesc->get_backend_descriptor()));
        if (status != CUDNN_STATUS_SUCCESS) {
            set_error_and_throw_exception(
                &m_operation, status, "CUDNN_BACKEND_OPERATION: SetAttribute CUDNN_ATTR_OPERATION_MATMUL_ADESC Failed");
            return std::move(m_operation);
        }
        status = detail::set_attribute(m_operation.pointer->get_backend_descriptor(),
                                       CUDNN_ATTR_OPERATION_MATMUL_BDESC,
                                       CUDNN_TYPE_BACKEND_DESCRIPTOR,
                                       1,
                                       &(m_operation.bmatdesc->get_backend_descriptor()));
        if (status != CUDNN_STATUS_SUCCESS) {
            set_error_and_throw_exception(
                &m_operation, status, "CUDNN_BACKEND_OPERATION: SetAttribute CUDNN_ATTR_OPERATION_MATMUL_BDESC Failed");
            return std::move(m_operation);
        }
        status = detail::set_attribute(m_operation.pointer->get_backend_descriptor(),
                                       CUDNN_ATTR_OPERATION_MATMUL_CDESC,
                                       CUDNN_TYPE_BACKEND_DESCRIPTOR,
                                       1,
                                       &(m_operation.cmatdesc->get_backend_descriptor()));
        if (status != CUDNN_STATUS_SUCCESS) {
            set_error_and_throw_exception(
                &m_operation, status, "CUDNN_BACKEND_OPERATION: SetAttribute CUDNN_ATTR_OPERATION_MATMUL_CDESC Failed");
            return std::move(m_operation);
        }
#if (CUDNN_VERSION >= 8700)
        NV_CUDNN_FE_DYNAMIC_CHECK_BACKEND_DESCRIPTOR(
            8700, m_operation, "CUDNN_BACKEND_OPERATION: M,N,K override Requires cudnn 8.7.0 and above");
        if (m_operation.moverridedesc != nullptr) {
            status = detail::set_attribute(m_operation.pointer->get_backend_descriptor(),
                                           CUDNN_ATTR_OPERATION_MATMUL_GEMM_M_OVERRIDE_DESC,
                                           CUDNN_TYPE_BACKEND_DESCRIPTOR,
                                           1,
                                           &(m_operation.moverridedesc->get_backend_descriptor()));
            if (status != CUDNN_STATUS_SUCCESS) {
                set_error_and_throw_exception(
                    &m_operation,
                    status,
                    "CUDNN_BACKEND_OPERATION: SetAttribute CUDNN_ATTR_OPERATION_MATMUL_GEMM_M_OVERRIDE_DESC Failed");
                return std::move(m_operation);
            }
        }
        if (m_operation.noverridedesc != nullptr) {
            status = detail::set_attribute(m_operation.pointer->get_backend_descriptor(),
                                           CUDNN_ATTR_OPERATION_MATMUL_GEMM_N_OVERRIDE_DESC,
                                           CUDNN_TYPE_BACKEND_DESCRIPTOR,
                                           1,
                                           &(m_operation.noverridedesc->get_backend_descriptor()));
            if (status != CUDNN_STATUS_SUCCESS) {
                set_error_and_throw_exception(
                    &m_operation,
                    status,
                    "CUDNN_BACKEND_OPERATION: SetAttribute CUDNN_ATTR_OPERATION_MATMUL_GEMM_N_OVERRIDE_DESC Failed");
                return std::move(m_operation);
            }
        }
        if (m_operation.koverridedesc != nullptr) {
            status = detail::set_attribute(m_operation.pointer->get_backend_descriptor(),
                                           CUDNN_ATTR_OPERATION_MATMUL_GEMM_K_OVERRIDE_DESC,
                                           CUDNN_TYPE_BACKEND_DESCRIPTOR,
                                           1,
                                           &(m_operation.koverridedesc->get_backend_descriptor()));
            if (status != CUDNN_STATUS_SUCCESS) {
                set_error_and_throw_exception(
                    &m_operation,
                    status,
                    "CUDNN_BACKEND_OPERATION: SetAttribute CUDNN_ATTR_OPERATION_MATMUL_GEMM_K_OVERRIDE_DESC Failed");
                return std::move(m_operation);
            }
        }
#endif
        status = detail::set_attribute(m_operation.pointer->get_backend_descriptor(),
                                       CUDNN_ATTR_OPERATION_MATMUL_DESC,
                                       CUDNN_TYPE_BACKEND_DESCRIPTOR,
                                       1,
                                       &(m_operation.matmuldesc->get_backend_descriptor()));
        if (status != CUDNN_STATUS_SUCCESS) {
            set_error_and_throw_exception(
                &m_operation, status, "CUDNN_BACKEND_OPERATION: SetAttribute CUDNN_ATTR_OPERATION_MATMUL_DESC Failed");
            return std::move(m_operation);
        }
        status = detail::finalize(m_operation.pointer->get_backend_descriptor());
        if (status != CUDNN_STATUS_SUCCESS) {
            set_error_and_throw_exception(&m_operation, status, "CUDNN_BACKEND_OPERATION: cudnnFinalize Failed");
            return std::move(m_operation);
        }
        return std::move(m_operation);
    }

    Operation_v8 &&
    build_pointwise_op() {
        auto status = CUDNN_STATUS_SUCCESS;

#ifndef CUDNN_FRONTEND_SKIP_JSON_LIB
        json j                   = m_operation.pointwise_mode;
        m_operation.operationTag = j;
#else
        m_operation.operationTag = std::to_string((int)m_operation.pointwise_mode);
#endif

        status = detail::set_attribute(m_operation.pointer->get_backend_descriptor(),
                                       CUDNN_ATTR_OPERATION_POINTWISE_PW_DESCRIPTOR,
                                       CUDNN_TYPE_BACKEND_DESCRIPTOR,
                                       1,
                                       &(m_operation.pwdesc->get_backend_descriptor()));
        if (status != CUDNN_STATUS_SUCCESS) {
            set_error_and_throw_exception(
                &m_operation,
                status,
                "CUDNN_BACKEND_OPERATION: SetAttribute CUDNN_ATTR_OPERATION_POINTWISE_PW_DESCRIPTOR Failed");
            return std::move(m_operation);
        }

        status = detail::set_attribute(m_operation.pointer->get_backend_descriptor(),
                                       CUDNN_ATTR_OPERATION_POINTWISE_XDESC,
                                       CUDNN_TYPE_BACKEND_DESCRIPTOR,
                                       1,
                                       &(m_operation.xdesc->get_backend_descriptor()));
        if (status != CUDNN_STATUS_SUCCESS) {
            set_error_and_throw_exception(
                &m_operation,
                status,
                "CUDNN_BACKEND_OPERATION: SetAttribute CUDNN_ATTR_OPERATION_POINTWISE_XDESC Failed");
            return std::move(m_operation);
        }

        if (!m_operation.is_pointwise_activation_bwd_op) {
            status = detail::set_attribute(m_operation.pointer->get_backend_descriptor(),
                                           CUDNN_ATTR_OPERATION_POINTWISE_YDESC,
                                           CUDNN_TYPE_BACKEND_DESCRIPTOR,
                                           1,
                                           &(m_operation.ydesc->get_backend_descriptor()));
            if (status != CUDNN_STATUS_SUCCESS) {
                set_error_and_throw_exception(
                    &m_operation,
                    status,
                    "CUDNN_BACKEND_OPERATION: SetAttribute CUDNN_ATTR_OPERATION_POINTWISE_YDESC Failed");
                return std::move(m_operation);
            }
        } else {
            status = detail::set_attribute(m_operation.pointer->get_backend_descriptor(),
                                           CUDNN_ATTR_OPERATION_POINTWISE_DYDESC,
                                           CUDNN_TYPE_BACKEND_DESCRIPTOR,
                                           1,
                                           &(m_operation.dydesc->get_backend_descriptor()));
            if (status != CUDNN_STATUS_SUCCESS) {
                set_error_and_throw_exception(
                    &m_operation,
                    status,
                    "CUDNN_BACKEND_OPERATION: SetAttribute CUDNN_ATTR_OPERATION_POINTWISE_DYDESC Failed");
                return std::move(m_operation);
            }

            status = detail::set_attribute(m_operation.pointer->get_backend_descriptor(),
                                           CUDNN_ATTR_OPERATION_POINTWISE_DXDESC,
                                           CUDNN_TYPE_BACKEND_DESCRIPTOR,
                                           1,
                                           &(m_operation.dxdesc->get_backend_descriptor()));
            if (status != CUDNN_STATUS_SUCCESS) {
                set_error_and_throw_exception(
                    &m_operation,
                    status,
                    "CUDNN_BACKEND_OPERATION: SetAttribute CUDNN_ATTR_OPERATION_POINTWISE_DXDESC Failed");
                return std::move(m_operation);
            }
        }

        void *alpha  = (m_operation.alphabetaType == CUDNN_TYPE_FLOAT ? static_cast<void *>(&m_operation.alpha_s)
                                                                      : static_cast<void *>(&m_operation.alpha_d));
        void *alpha2 = (m_operation.alphabetaType == CUDNN_TYPE_FLOAT ? static_cast<void *>(&m_operation.alpha2_s)
                                                                      : static_cast<void *>(&m_operation.alpha2_d));
        status       = detail::set_attribute(m_operation.pointer->get_backend_descriptor(),
                                       CUDNN_ATTR_OPERATION_POINTWISE_ALPHA1,
                                       m_operation.alphabetaType,
                                       1,
                                       alpha);
        if (status != CUDNN_STATUS_SUCCESS) {
            set_error_and_throw_exception(
                &m_operation,
                status,
                "CUDNN_BACKEND_OPERATION: SetAttribute CUDNN_ATTR_OPERATION_POINTWISE_ALPHA1 Failed");
            return std::move(m_operation);
        }
        status = detail::set_attribute(m_operation.pointer->get_backend_descriptor(),
                                       CUDNN_ATTR_OPERATION_POINTWISE_ALPHA2,
                                       m_operation.alphabetaType,
                                       1,
                                       alpha2);
        if (status != CUDNN_STATUS_SUCCESS) {
            set_error_and_throw_exception(
                &m_operation,
                status,
                "CUDNN_BACKEND_OPERATION: SetAttribute CUDNN_ATTR_OPERATION_POINTWISE_ALPHA2 Failed");
            return std::move(m_operation);
        }

        if (m_operation.pointwise_port_count >= 3 && !m_operation.is_pointwise_activation_bwd_op) {
            status = detail::set_attribute(m_operation.pointer->get_backend_descriptor(),
                                           CUDNN_ATTR_OPERATION_POINTWISE_BDESC,
                                           CUDNN_TYPE_BACKEND_DESCRIPTOR,
                                           1,
                                           &(m_operation.bdesc->get_backend_descriptor()));
            if (status != CUDNN_STATUS_SUCCESS) {
                set_error_and_throw_exception(
                    &m_operation,
                    status,
                    "CUDNN_BACKEND_OPERATION: SetAttribute CUDNN_ATTR_OPERATION_POINTWISE_BDESC Failed");
                return std::move(m_operation);
            }
        }

        if (m_operation.pointwise_port_count == 4) {
            status = detail::set_attribute(m_operation.pointer->get_backend_descriptor(),
                                           CUDNN_ATTR_OPERATION_POINTWISE_TDESC,
                                           CUDNN_TYPE_BACKEND_DESCRIPTOR,
                                           1,
                                           &(m_operation.tdesc->get_backend_descriptor()));
            if (status != CUDNN_STATUS_SUCCESS) {
                set_error_and_throw_exception(
                    &m_operation,
                    status,
                    "CUDNN_BACKEND_OPERATION: SetAttribute CUDNN_ATTR_OPERATION_POINTWISE_TDESC Failed");
                return std::move(m_operation);
            }
        }

        status = detail::finalize(m_operation.pointer->get_backend_descriptor());
        if (status != CUDNN_STATUS_SUCCESS) {
            set_error_and_throw_exception(&m_operation, status, "CUDNN_BACKEND_OPERATION: cudnnFinalize Failed");
            return std::move(m_operation);
        }
        return std::move(m_operation);
    }

    Operation_v8 &&
    build_conv_backward_data() {
        m_operation.operationTag = "ConvBwdData";

        auto status = CUDNN_STATUS_SUCCESS;

        auto dxdesc_ = m_operation.dxdesc != nullptr ? m_operation.dxdesc : m_operation.xdesc;
        status       = detail::set_attribute(m_operation.pointer->get_backend_descriptor(),
                                       CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_DX,
                                       CUDNN_TYPE_BACKEND_DESCRIPTOR,
                                       1,
                                       &(dxdesc_->get_backend_descriptor()));
        if (status != CUDNN_STATUS_SUCCESS) {
            set_error_and_throw_exception(
                &m_operation,
                status,
                "CUDNN_BACKEND_OPERATION: SetAttribute CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_DX Failed");
            return std::move(m_operation);
        }

        status = detail::set_attribute(m_operation.pointer->get_backend_descriptor(),
                                       CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_W,
                                       CUDNN_TYPE_BACKEND_DESCRIPTOR,
                                       1,
                                       &(m_operation.wdesc->get_backend_descriptor()));
        if (status != CUDNN_STATUS_SUCCESS) {
            set_error_and_throw_exception(
                &m_operation,
                status,
                "CUDNN_BACKEND_OPERATION: SetAttribute CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_W Failed");
            return std::move(m_operation);
        }

        auto dydesc_ = m_operation.dydesc != nullptr ? m_operation.dydesc : m_operation.ydesc;
        status       = detail::set_attribute(m_operation.pointer->get_backend_descriptor(),
                                       CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_DY,
                                       CUDNN_TYPE_BACKEND_DESCRIPTOR,
                                       1,
                                       &(dydesc_->get_backend_descriptor()));
        if (status != CUDNN_STATUS_SUCCESS) {
            set_error_and_throw_exception(
                &m_operation,
                status,
                "CUDNN_BACKEND_OPERATION: SetAttribute CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_DY Failed");
            return std::move(m_operation);
        }

        status = detail::set_attribute(m_operation.pointer->get_backend_descriptor(),
                                       CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_CONV_DESC,
                                       CUDNN_TYPE_BACKEND_DESCRIPTOR,
                                       1,
                                       &(m_operation.cdesc->get_backend_descriptor()));
        if (status != CUDNN_STATUS_SUCCESS) {
            set_error_and_throw_exception(
                &m_operation,
                status,
                "CUDNN_BACKEND_OPERATION: SetAttribute CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_CONV_DESC Failed");
            return std::move(m_operation);
        }

        void *alpha = (m_operation.alphabetaType == CUDNN_TYPE_FLOAT ? static_cast<void *>(&m_operation.alpha_s)
                                                                     : static_cast<void *>(&m_operation.alpha_d));
        void *beta  = (m_operation.alphabetaType == CUDNN_TYPE_FLOAT ? static_cast<void *>(&m_operation.beta_s)
                                                                     : static_cast<void *>(&m_operation.beta_d));
        status      = detail::set_attribute(m_operation.pointer->get_backend_descriptor(),
                                       CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_ALPHA,
                                       m_operation.alphabetaType,
                                       1,
                                       alpha);
        if (status != CUDNN_STATUS_SUCCESS) {
            set_error_and_throw_exception(
                &m_operation,
                status,
                "CUDNN_BACKEND_OPERATION: SetAttribute CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_ALPHA Failed");
            return std::move(m_operation);
        }
        status = detail::set_attribute(m_operation.pointer->get_backend_descriptor(),
                                       CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_BETA,
                                       m_operation.alphabetaType,
                                       1,
                                       beta);
        if (status != CUDNN_STATUS_SUCCESS) {
            set_error_and_throw_exception(
                &m_operation,
                status,
                "CUDNN_BACKEND_OPERATION: SetAttribute CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_BETA Failed");
            return std::move(m_operation);
        }

        status = detail::finalize(m_operation.pointer->get_backend_descriptor());
        if (status != CUDNN_STATUS_SUCCESS) {
            set_error_and_throw_exception(&m_operation, status, "CUDNN_BACKEND_OPERATION: cudnnFinalize Failed");
            return std::move(m_operation);
        }
        CUDNN_FE_LOG_LABEL_ENDL("Extracting the feature vector");
        extract_feature_vector(DescriptorType_t::OPERATION_CONVOLUTION_BACKWARD_DATA_DESCRIPTOR);
        return std::move(m_operation);
    }

    Operation_v8 &&
    build_bn_finalize_op() {
        m_operation.operationTag = "BNFinalize";
        auto status              = CUDNN_STATUS_SUCCESS;

        auto set_attribute = [&status](Operation_v8 &operation,
                                       cudnnBackendAttributeName_t attr,
                                       const char *fail_msg,
                                       void const *ptr,
                                       cudnnBackendAttributeType_t type = CUDNN_TYPE_BACKEND_DESCRIPTOR,
                                       int64_t cnt                      = 1) {
            status = detail::set_attribute(operation.pointer->get_backend_descriptor(), attr, type, cnt, ptr);
            if (status != CUDNN_STATUS_SUCCESS) {
                set_error_and_throw_exception(&operation, status, fail_msg);
            }
        };

        set_attribute(m_operation,
                      CUDNN_ATTR_OPERATION_BN_FINALIZE_STATS_MODE,
                      "CUDNN_BACKEND_OPERATION: SetAttribute CUDNN_ATTR_OPERATION_BN_FINALIZE_STATS_MODE Failed",
                      &(m_operation.bn_stats_mode),
                      CUDNN_TYPE_BN_FINALIZE_STATS_MODE,
                      1);
        if (status != CUDNN_STATUS_SUCCESS) {
            return std::move(m_operation);
        }

        set_attribute(m_operation,
                      CUDNN_ATTR_OPERATION_BN_FINALIZE_MATH_PREC,
                      "CUDNN_BACKEND_OPERATION: SetAttribute CUDNN_ATTR_OPERATION_BN_FINALIZE_MATH_PREC Failed",
                      &(m_operation.compute_type),
                      CUDNN_TYPE_DATA_TYPE,
                      1);
        if (status != CUDNN_STATUS_SUCCESS) {
            return std::move(m_operation);
        }

        if (m_operation.sumdesc) {
            set_attribute(m_operation,
                          CUDNN_ATTR_OPERATION_BN_FINALIZE_Y_SUM_DESC,
                          "CUDNN_BACKEND_OPERATION: SetAttribute CUDNN_ATTR_OPERATION_BN_FINALIZE_Y_SUM_DESC Failed",
                          &(m_operation.sumdesc->get_backend_descriptor()));
            if (status != CUDNN_STATUS_SUCCESS) {
                return std::move(m_operation);
            }
        }

        if (m_operation.sqsumdesc) {
            set_attribute(m_operation,
                          CUDNN_ATTR_OPERATION_BN_FINALIZE_Y_SQ_SUM_DESC,
                          "CUDNN_BACKEND_OPERATION: SetAttribute CUDNN_ATTR_OPERATION_BN_FINALIZE_Y_SQ_SUM_DESC Failed",
                          &(m_operation.sqsumdesc->get_backend_descriptor()));
            if (status != CUDNN_STATUS_SUCCESS) {
                return std::move(m_operation);
            }
        }

        if (m_operation.biasdesc) {
            set_attribute(m_operation,
                          CUDNN_ATTR_OPERATION_BN_FINALIZE_BIAS_DESC,
                          "CUDNN_BACKEND_OPERATION: SetAttribute CUDNN_ATTR_OPERATION_BN_FINALIZE_BIAS_DESC Failed",
                          &(m_operation.biasdesc->get_backend_descriptor()));
            if (status != CUDNN_STATUS_SUCCESS) {
                return std::move(m_operation);
            }
        }

        if (m_operation.scaledesc) {
            set_attribute(m_operation,
                          CUDNN_ATTR_OPERATION_BN_FINALIZE_SCALE_DESC,
                          "CUDNN_BACKEND_OPERATION: SetAttribute CUDNN_ATTR_OPERATION_BN_FINALIZE_SCALE_DESC Failed",
                          &(m_operation.scaledesc->get_backend_descriptor()));
            if (status != CUDNN_STATUS_SUCCESS) {
                return std::move(m_operation);
            }
        }

        if (m_operation.eqscaledesc) {
            set_attribute(m_operation,
                          CUDNN_ATTR_OPERATION_BN_FINALIZE_EQ_SCALE_DESC,
                          "CUDNN_BACKEND_OPERATION: SetAttribute CUDNN_ATTR_OPERATION_BN_FINALIZE_EQ_SCALE_DESC Failed",
                          &(m_operation.eqscaledesc->get_backend_descriptor()));
            if (status != CUDNN_STATUS_SUCCESS) {
                return std::move(m_operation);
            }
        }

        if (m_operation.eqbiasdesc) {
            set_attribute(m_operation,
                          CUDNN_ATTR_OPERATION_BN_FINALIZE_EQ_BIAS_DESC,
                          "CUDNN_BACKEND_OPERATION: SetAttribute CUDNN_ATTR_OPERATION_BN_FINALIZE_EQ_BIAS_DESC Failed",
                          &(m_operation.eqbiasdesc->get_backend_descriptor()));
            if (status != CUDNN_STATUS_SUCCESS) {
                return std::move(m_operation);
            }
        }

        if (m_operation.prevMeandesc) {
            set_attribute(
                m_operation,
                CUDNN_ATTR_OPERATION_BN_FINALIZE_PREV_RUNNING_MEAN_DESC,
                "CUDNN_BACKEND_OPERATION: SetAttribute CUDNN_ATTR_OPERATION_BN_FINALIZE_PREV_RUNNING_MEAN_DESC Failed",
                &(m_operation.prevMeandesc->get_backend_descriptor()));
            if (status != CUDNN_STATUS_SUCCESS) {
                return std::move(m_operation);
            }
        }

        if (m_operation.prevVardesc) {
            set_attribute(
                m_operation,
                CUDNN_ATTR_OPERATION_BN_FINALIZE_PREV_RUNNING_VAR_DESC,
                "CUDNN_BACKEND_OPERATION: SetAttribute CUDNN_ATTR_OPERATION_BN_FINALIZE_PREV_RUNNING_VAR_DESC Failed",
                &(m_operation.prevVardesc->get_backend_descriptor()));
            if (status != CUDNN_STATUS_SUCCESS) {
                return std::move(m_operation);
            }
        }

        if (m_operation.nextMeandesc) {
            set_attribute(m_operation,
                          CUDNN_ATTR_OPERATION_BN_FINALIZE_UPDATED_RUNNING_MEAN_DESC,
                          "CUDNN_BACKEND_OPERATION: SetAttribute "
                          "CUDNN_ATTR_OPERATION_BN_FINALIZE_UPDATED_RUNNING_MEAN_DESC Failed",
                          &(m_operation.nextMeandesc->get_backend_descriptor()));
            if (status != CUDNN_STATUS_SUCCESS) {
                return std::move(m_operation);
            }
        }

        if (m_operation.nextVardesc) {
            set_attribute(m_operation,
                          CUDNN_ATTR_OPERATION_BN_FINALIZE_UPDATED_RUNNING_VAR_DESC,
                          "CUDNN_BACKEND_OPERATION: SetAttribute "
                          "CUDNN_ATTR_OPERATION_BN_FINALIZE_UPDATED_RUNNING_VAR_DESC Failed",
                          &(m_operation.nextVardesc->get_backend_descriptor()));
            if (status != CUDNN_STATUS_SUCCESS) {
                return std::move(m_operation);
            }
        }

        if (m_operation.savedMeandesc) {
            set_attribute(
                m_operation,
                CUDNN_ATTR_OPERATION_BN_FINALIZE_SAVED_MEAN_DESC,
                "CUDNN_BACKEND_OPERATION: SetAttribute CUDNN_ATTR_OPERATION_BN_FINALIZE_SAVED_MEAN_DESC Failed",
                &(m_operation.savedMeandesc->get_backend_descriptor()));
            if (status != CUDNN_STATUS_SUCCESS) {
                return std::move(m_operation);
            }
        }

        if (m_operation.savedInVardesc) {
            set_attribute(
                m_operation,
                CUDNN_ATTR_OPERATION_BN_FINALIZE_SAVED_INV_STD_DESC,
                "CUDNN_BACKEND_OPERATION: SetAttribute CUDNN_ATTR_OPERATION_BN_FINALIZE_SAVED_INV_STD_DESC Failed",
                &(m_operation.savedInVardesc->get_backend_descriptor()));
            if (status != CUDNN_STATUS_SUCCESS) {
                return std::move(m_operation);
            }
        }

        if (m_operation.epsilondesc) {
            set_attribute(m_operation,
                          CUDNN_ATTR_OPERATION_BN_FINALIZE_EPSILON_DESC,
                          "CUDNN_BACKEND_OPERATION: SetAttribute CUDNN_ATTR_OPERATION_BN_FINALIZE_EPSILON_DESC Failed",
                          &(m_operation.epsilondesc->get_backend_descriptor()));
            if (status != CUDNN_STATUS_SUCCESS) {
                return std::move(m_operation);
            }
        }

        if (m_operation.expDecayFactordesc) {
            set_attribute(
                m_operation,
                CUDNN_ATTR_OPERATION_BN_FINALIZE_EXP_AVERATE_FACTOR_DESC,
                "CUDNN_BACKEND_OPERATION: SetAttribute CUDNN_ATTR_OPERATION_BN_FINALIZE_EXP_AVERATE_FACTOR_DESC Failed",
                &(m_operation.expDecayFactordesc->get_backend_descriptor()));
            if (status != CUDNN_STATUS_SUCCESS) {
                return std::move(m_operation);
            }
        }

        if (m_operation.accumCountdesc) {
            set_attribute(
                m_operation,
                CUDNN_ATTR_OPERATION_BN_FINALIZE_ACCUM_COUNT_DESC,
                "CUDNN_BACKEND_OPERATION: SetAttribute CUDNN_ATTR_OPERATION_BN_FINALIZE_ACCUM_COUNT_DESC Failed",
                &(m_operation.accumCountdesc->get_backend_descriptor()));
            if (status != CUDNN_STATUS_SUCCESS) {
                return std::move(m_operation);
            }
        }

        status = detail::finalize(m_operation.pointer->get_backend_descriptor());
        if (status != CUDNN_STATUS_SUCCESS) {
            set_error_and_throw_exception(&m_operation, status, "CUDNN_BACKEND_OPERATION: cudnnFinalize Failed");
            return std::move(m_operation);
        }
        return std::move(m_operation);
    }

    Operation_v8 &&
    build_genstats_op() {
        m_operation.operationTag = "GenStats";
        auto status              = CUDNN_STATUS_SUCCESS;

        status = detail::set_attribute(m_operation.pointer->get_backend_descriptor(),
                                       CUDNN_ATTR_OPERATION_GENSTATS_XDESC,
                                       CUDNN_TYPE_BACKEND_DESCRIPTOR,
                                       1,
                                       &(m_operation.xdesc->get_backend_descriptor()));
        if (status != CUDNN_STATUS_SUCCESS) {
            set_error_and_throw_exception(
                &m_operation,
                status,
                "CUDNN_BACKEND_OPERATION: SetAttribute CUDNN_ATTR_OPERATION_GENSTATS_XDESC Failed");
            return std::move(m_operation);
        }

        status = detail::set_attribute(m_operation.pointer->get_backend_descriptor(),
                                       CUDNN_ATTR_OPERATION_GENSTATS_SUMDESC,
                                       CUDNN_TYPE_BACKEND_DESCRIPTOR,
                                       1,
                                       &(m_operation.sumdesc->get_backend_descriptor()));
        if (status != CUDNN_STATUS_SUCCESS) {
            set_error_and_throw_exception(
                &m_operation,
                status,
                "CUDNN_BACKEND_OPERATION: SetAttribute CUDNN_ATTR_OPERATION_GENSTATS_SUMDESC Failed");
            return std::move(m_operation);
        }

        status = detail::set_attribute(m_operation.pointer->get_backend_descriptor(),
                                       CUDNN_ATTR_OPERATION_GENSTATS_SQSUMDESC,
                                       CUDNN_TYPE_BACKEND_DESCRIPTOR,
                                       1,
                                       &(m_operation.sqsumdesc->get_backend_descriptor()));
        if (status != CUDNN_STATUS_SUCCESS) {
            set_error_and_throw_exception(
                &m_operation,
                status,
                "CUDNN_BACKEND_OPERATION: SetAttribute CUDNN_ATTR_OPERATION_GENSTATS_SQSUMDESC Failed");
            return std::move(m_operation);
        }

        status = detail::set_attribute(m_operation.pointer->get_backend_descriptor(),
                                       CUDNN_ATTR_OPERATION_GENSTATS_MODE,
                                       CUDNN_TYPE_GENSTATS_MODE,
                                       1,
                                       &(m_operation.genstats_mode));
        if (status != CUDNN_STATUS_SUCCESS) {
            set_error_and_throw_exception(
                &m_operation,
                status,
                "CUDNN_BACKEND_OPERATION: SetAttribute CUDNN_ATTR_OPERATION_GENSTATS_MODE Failed");
            return std::move(m_operation);
        }

        status = detail::set_attribute(m_operation.pointer->get_backend_descriptor(),
                                       CUDNN_ATTR_OPERATION_GENSTATS_MATH_PREC,
                                       CUDNN_TYPE_DATA_TYPE,
                                       1,
                                       &(m_operation.compute_type));
        if (status != CUDNN_STATUS_SUCCESS) {
            set_error_and_throw_exception(
                &m_operation,
                status,
                "CUDNN_BACKEND_OPERATION: SetAttribute CUDNN_ATTR_OPERATION_GENSTATS_MATH_PREC Failed");
            return std::move(m_operation);
        }

        status = detail::finalize(m_operation.pointer->get_backend_descriptor());
        if (status != CUDNN_STATUS_SUCCESS) {
            set_error_and_throw_exception(&m_operation, status, "CUDNN_BACKEND_OPERATION: cudnnFinalize Failed");
            return std::move(m_operation);
        }

        return std::move(m_operation);
    }

    Operation_v8 &&
    build_conv_backward_filter() {
        m_operation.operationTag = "ConvBwdFilter";

        auto status = CUDNN_STATUS_SUCCESS;

        status = detail::set_attribute(m_operation.pointer->get_backend_descriptor(),
                                       CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_X,
                                       CUDNN_TYPE_BACKEND_DESCRIPTOR,
                                       1,
                                       &(m_operation.xdesc->get_backend_descriptor()));
        if (status != CUDNN_STATUS_SUCCESS) {
            set_error_and_throw_exception(
                &m_operation,
                status,
                "CUDNN_BACKEND_OPERATION: SetAttribute CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_X Failed");
            return std::move(m_operation);
        }

        auto dwdesc_ = m_operation.dwdesc != nullptr ? m_operation.dwdesc : m_operation.wdesc;
        status       = detail::set_attribute(m_operation.pointer->get_backend_descriptor(),
                                       CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_DW,
                                       CUDNN_TYPE_BACKEND_DESCRIPTOR,
                                       1,
                                       &(dwdesc_->get_backend_descriptor()));
        if (status != CUDNN_STATUS_SUCCESS) {
            set_error_and_throw_exception(
                &m_operation,
                status,
                "CUDNN_BACKEND_OPERATION: SetAttribute CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_DW Failed");
            return std::move(m_operation);
        }

        auto dydesc_ = m_operation.dydesc != nullptr ? m_operation.dydesc : m_operation.ydesc;
        status       = detail::set_attribute(m_operation.pointer->get_backend_descriptor(),
                                       CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_DY,
                                       CUDNN_TYPE_BACKEND_DESCRIPTOR,
                                       1,
                                       &(dydesc_->get_backend_descriptor()));
        if (status != CUDNN_STATUS_SUCCESS) {
            set_error_and_throw_exception(
                &m_operation,
                status,
                "CUDNN_BACKEND_OPERATION: SetAttribute CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_DY Failed");
            return std::move(m_operation);
        }

        status = detail::set_attribute(m_operation.pointer->get_backend_descriptor(),
                                       CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_CONV_DESC,
                                       CUDNN_TYPE_BACKEND_DESCRIPTOR,
                                       1,
                                       &(m_operation.cdesc->get_backend_descriptor()));
        if (status != CUDNN_STATUS_SUCCESS) {
            set_error_and_throw_exception(&m_operation,
                                          status,
                                          "CUDNN_BACKEND_OPERATION: SetAttribute "
                                          "CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_CONV_DESC Failed");
            return std::move(m_operation);
        }
        void *alpha = (m_operation.alphabetaType == CUDNN_TYPE_FLOAT ? static_cast<void *>(&m_operation.alpha_s)
                                                                     : static_cast<void *>(&m_operation.alpha_d));
        void *beta  = (m_operation.alphabetaType == CUDNN_TYPE_FLOAT ? static_cast<void *>(&m_operation.beta_s)
                                                                     : static_cast<void *>(&m_operation.beta_d));
        status      = detail::set_attribute(m_operation.pointer->get_backend_descriptor(),
                                       CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_ALPHA,
                                       m_operation.alphabetaType,
                                       1,
                                       alpha);
        if (status != CUDNN_STATUS_SUCCESS) {
            set_error_and_throw_exception(
                &m_operation,
                status,
                "CUDNN_BACKEND_OPERATION: SetAttribute CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_ALPHA Failed");
            return std::move(m_operation);
        }
        status = detail::set_attribute(m_operation.pointer->get_backend_descriptor(),
                                       CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_BETA,
                                       m_operation.alphabetaType,
                                       1,
                                       beta);
        if (status != CUDNN_STATUS_SUCCESS) {
            set_error_and_throw_exception(
                &m_operation,
                status,
                "CUDNN_BACKEND_OPERATION: SetAttribute CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_BETA Failed");
            return std::move(m_operation);
        }

        status = detail::finalize(m_operation.pointer->get_backend_descriptor());
        if (status != CUDNN_STATUS_SUCCESS) {
            set_error_and_throw_exception(&m_operation, status, "CUDNN_BACKEND_OPERATION: cudnnFinalize Failed");
            return std::move(m_operation);
        }
        CUDNN_FE_LOG_LABEL_ENDL("Extracting the feature vector");
        extract_feature_vector(DescriptorType_t::OPERATION_CONVOLUTION_BACKWARD_FILTER_DESCRIPTOR);
        return std::move(m_operation);
    }

    Operation_v8 &&
    build_norm_forward() {
        m_operation.operationTag = "Norm_Fwd";
        auto status              = CUDNN_STATUS_SUCCESS;

        auto set_attribute = [&status](Operation_v8 &operation,
                                       cudnnBackendAttributeName_t attr,
                                       const char *fail_msg,
                                       void const *ptr,
                                       cudnnBackendAttributeType_t type = CUDNN_TYPE_BACKEND_DESCRIPTOR,
                                       int64_t cnt                      = 1) {
            status = detail::set_attribute(operation.pointer->get_backend_descriptor(), attr, type, cnt, ptr);
            if (status != CUDNN_STATUS_SUCCESS) {
                set_error_and_throw_exception(&operation, status, fail_msg);
            }
        };

        cudnnBackendNormMode_t cudnn_norm_mode;
        status = detail::convert_to_cudnn_type(m_operation.norm_mode, cudnn_norm_mode);
        if (status != CUDNN_STATUS_SUCCESS) {
            set_error_and_throw_exception(
                &m_operation,
                status,
                "CUDNN_BACKEND_OPERATION: SetAttribute CUDNN_ATTR_OPERATION_NORM_FWD_MODE Failed");
            return std::move(m_operation);
        }
        status = detail::set_attribute(m_operation.pointer->get_backend_descriptor(),
                                       CUDNN_ATTR_OPERATION_NORM_FWD_MODE,
                                       CUDNN_TYPE_NORM_MODE,
                                       1,
                                       &cudnn_norm_mode);
        if (status != CUDNN_STATUS_SUCCESS) {
            set_error_and_throw_exception(
                &m_operation,
                status,
                "CUDNN_BACKEND_OPERATION: SetAttribute CUDNN_ATTR_OPERATION_NORM_FWD_MODE Failed");
            return std::move(m_operation);
        }

        cudnnBackendNormFwdPhase_t cudnn_norm_fwd_phase;
        status = detail::convert_to_cudnn_type(m_operation.norm_fwd_phase, cudnn_norm_fwd_phase);
        if (status != CUDNN_STATUS_SUCCESS) {
            set_error_and_throw_exception(
                &m_operation,
                status,
                "CUDNN_BACKEND_OPERATION: SetAttribute CUDNN_ATTR_OPERATION_NORM_FWD_PHASE Failed");
            return std::move(m_operation);
        }
        status = detail::set_attribute(m_operation.pointer->get_backend_descriptor(),
                                       CUDNN_ATTR_OPERATION_NORM_FWD_PHASE,
                                       CUDNN_TYPE_NORM_FWD_PHASE,
                                       1,
                                       &cudnn_norm_fwd_phase);
        if (status != CUDNN_STATUS_SUCCESS) {
            set_error_and_throw_exception(
                &m_operation,
                status,
                "CUDNN_BACKEND_OPERATION: SetAttribute CUDNN_ATTR_OPERATION_NORM_FWD_PHASE Failed");
            return std::move(m_operation);
        }

        set_attribute(m_operation,
                      CUDNN_ATTR_OPERATION_NORM_FWD_XDESC,
                      "CUDNN_BACKEND_OPERATION: SetAttribute CUDNN_ATTR_OPERATION_NORM_FWD_XDESC Failed",
                      &m_operation.xdesc->get_backend_descriptor());
        if (status != CUDNN_STATUS_SUCCESS) {
            return std::move(m_operation);
        }
        if (m_operation.savedMeandesc)
            set_attribute(m_operation,
                          CUDNN_ATTR_OPERATION_NORM_FWD_MEAN_DESC,
                          "CUDNN_BACKEND_OPERATION: SetAttribute CUDNN_ATTR_OPERATION_NORM_FWD_MEAN_DESC Failed",
                          &m_operation.savedMeandesc->get_backend_descriptor());
        if (status != CUDNN_STATUS_SUCCESS) {
            return std::move(m_operation);
        }
        if (m_operation.savedInVardesc)
            set_attribute(
                m_operation,
                CUDNN_ATTR_OPERATION_NORM_FWD_INV_VARIANCE_DESC,
                "CUDNN_BACKEND_OPERATION: SetAttribute CUDNN_ATTR_OPERATION_NORM_FWD_INV_VARIANCE_DESC Failed",
                &m_operation.savedInVardesc->get_backend_descriptor());
        if (status != CUDNN_STATUS_SUCCESS) {
            return std::move(m_operation);
        }
        if (m_operation.scaledesc)
            set_attribute(m_operation,
                          CUDNN_ATTR_OPERATION_NORM_FWD_SCALE_DESC,
                          "CUDNN_BACKEND_OPERATION: SetAttribute CUDNN_ATTR_OPERATION_NORM_FWD_SCALE_DESC Failed",
                          &m_operation.scaledesc->get_backend_descriptor());
        if (status != CUDNN_STATUS_SUCCESS) {
            return std::move(m_operation);
        }
        if (m_operation.biasdesc)
            set_attribute(m_operation,
                          CUDNN_ATTR_OPERATION_NORM_FWD_BIAS_DESC,
                          "CUDNN_BACKEND_OPERATION: SetAttribute CUDNN_ATTR_OPERATION_NORM_FWD_BIAS_DESC Failed",
                          &m_operation.biasdesc->get_backend_descriptor());
        if (status != CUDNN_STATUS_SUCCESS) {
            return std::move(m_operation);
        }
        if (m_operation.epsilondesc)
            set_attribute(m_operation,
                          CUDNN_ATTR_OPERATION_NORM_FWD_EPSILON_DESC,
                          "CUDNN_BACKEND_OPERATION: SetAttribute CUDNN_ATTR_OPERATION_NORM_FWD_EPSILON Failed",
                          &m_operation.epsilondesc->get_backend_descriptor());
        if (status != CUDNN_STATUS_SUCCESS) {
            return std::move(m_operation);
        }
        if (m_operation.expDecayFactordesc)
            set_attribute(m_operation,
                          CUDNN_ATTR_OPERATION_NORM_FWD_EXP_AVG_FACTOR_DESC,
                          "CUDNN_BACKEND_OPERATION: SetAttribute CUDNN_ATTR_OPERATION_NORM_FWD_EXP_AVG_FACTOR Failed",
                          &m_operation.expDecayFactordesc->get_backend_descriptor());
        if (status != CUDNN_STATUS_SUCCESS) {
            return std::move(m_operation);
        }
        if (m_operation.prevMeandesc)
            set_attribute(
                m_operation,
                CUDNN_ATTR_OPERATION_NORM_FWD_INPUT_RUNNING_MEAN_DESC,
                "CUDNN_BACKEND_OPERATION: SetAttribute CUDNN_ATTR_OPERATION_NORM_FWD_INPUT_RUNNING_MEAN_DESC Failed",
                &m_operation.prevMeandesc->get_backend_descriptor());
        if (status != CUDNN_STATUS_SUCCESS) {
            return std::move(m_operation);
        }
        if (m_operation.prevVardesc)
            set_attribute(
                m_operation,
                CUDNN_ATTR_OPERATION_NORM_FWD_INPUT_RUNNING_VAR_DESC,
                "CUDNN_BACKEND_OPERATION: SetAttribute CUDNN_ATTR_OPERATION_NORM_FWD_INPUT_RUNNING_VAR_DESC Failed",
                &m_operation.prevVardesc->get_backend_descriptor());
        if (status != CUDNN_STATUS_SUCCESS) {
            return std::move(m_operation);
        }
        if (m_operation.nextMeandesc)
            set_attribute(
                m_operation,
                CUDNN_ATTR_OPERATION_NORM_FWD_OUTPUT_RUNNING_MEAN_DESC,
                "CUDNN_BACKEND_OPERATION: SetAttribute CUDNN_ATTR_OPERATION_NORM_FWD_OUTPUT_RUNNING_MEAN_DESC Failed",
                &m_operation.nextMeandesc->get_backend_descriptor());
        if (status != CUDNN_STATUS_SUCCESS) {
            return std::move(m_operation);
        }
        if (m_operation.nextVardesc)
            set_attribute(
                m_operation,
                CUDNN_ATTR_OPERATION_NORM_FWD_OUTPUT_RUNNING_VAR_DESC,
                "CUDNN_BACKEND_OPERATION: SetAttribute CUDNN_ATTR_OPERATION_NORM_FWD_OUTPUT_RUNNING_VAR_DESC Failed",
                &m_operation.nextVardesc->get_backend_descriptor());
        if (status != CUDNN_STATUS_SUCCESS) {
            return std::move(m_operation);
        }
        if (m_operation.ydesc)
            set_attribute(m_operation,
                          CUDNN_ATTR_OPERATION_NORM_FWD_YDESC,
                          "CUDNN_BACKEND_OPERATION: SetAttribute CUDNCUDNN_ATTR_OPERATION_NORM_FWD_YDESC Failed",
                          &m_operation.ydesc->get_backend_descriptor());
        if (status != CUDNN_STATUS_SUCCESS) {
            return std::move(m_operation);
        }
        if (m_operation.peerStatdescs.size()) {
            std::vector<cudnnBackendDescriptor_t> backend_peer_stat_descs;
            for (auto &desc : m_operation.peerStatdescs) {
                backend_peer_stat_descs.push_back(desc->get_backend_descriptor());
            }
            set_attribute(
                m_operation,
                CUDNN_ATTR_OPERATION_NORM_FWD_PEER_STAT_DESCS,
                "CUDNN_BACKEND_OPERATION: SetAttribute CUDNCUDNN_ATTR_OPERATION_NORM_FWD_PEER_STAT_DESCS Failed",
                backend_peer_stat_descs.data(),
                CUDNN_TYPE_BACKEND_DESCRIPTOR,
                backend_peer_stat_descs.size());
        }
        if (status != CUDNN_STATUS_SUCCESS) {
            return std::move(m_operation);
        }

        status = detail::finalize(m_operation.pointer->get_backend_descriptor());

        if (status != CUDNN_STATUS_SUCCESS) {
            set_error_and_throw_exception(&m_operation, status, "CUDNN_BACKEND_OPERATION: cudnnFinalize Failed");
            return std::move(m_operation);
        }
        return std::move(m_operation);
    }

    Operation_v8 &&
    build_norm_backward() {
        m_operation.operationTag = "Norm_Bwd";
        auto status              = CUDNN_STATUS_SUCCESS;

        auto set_attribute = [&status](Operation_v8 &operation,
                                       cudnnBackendAttributeName_t attr,
                                       const char *fail_msg,
                                       void const *ptr,
                                       cudnnBackendAttributeType_t type = CUDNN_TYPE_BACKEND_DESCRIPTOR,
                                       int64_t cnt                      = 1) {
            status = detail::set_attribute(operation.pointer->get_backend_descriptor(), attr, type, cnt, ptr);
            if (status != CUDNN_STATUS_SUCCESS) {
                set_error_and_throw_exception(&operation, status, fail_msg);
            }
        };
        cudnnBackendNormMode_t cudnn_norm_mode;
        status = detail::convert_to_cudnn_type(m_operation.norm_mode, cudnn_norm_mode);
        set_attribute(m_operation,
                      CUDNN_ATTR_OPERATION_NORM_BWD_MODE,
                      "CUDNN_BACKEND_OPERATION: SetAttribute CUDNN_ATTR_OPERATION_NORM_BWD_MODE Failed",
                      &cudnn_norm_mode,
                      CUDNN_TYPE_NORM_MODE);
        if (status != CUDNN_STATUS_SUCCESS) {
            return std::move(m_operation);
        }
        if (m_operation.xdesc)
            set_attribute(m_operation,
                          CUDNN_ATTR_OPERATION_NORM_BWD_XDESC,
                          "CUDNN_BACKEND_OPERATION: SetAttribute CUDNN_ATTR_OPERATION_NORM_BWD_XDESC Failed",
                          &m_operation.xdesc->get_backend_descriptor());
        if (status != CUDNN_STATUS_SUCCESS) {
            return std::move(m_operation);
        }
        if (m_operation.savedMeandesc)
            set_attribute(m_operation,
                          CUDNN_ATTR_OPERATION_NORM_BWD_MEAN_DESC,
                          "CUDNN_BACKEND_OPERATION: SetAttribute CUDNN_ATTR_OPERATION_NORM_BWD_MEAN_DESC Failed",
                          &m_operation.savedMeandesc->get_backend_descriptor());
        if (status != CUDNN_STATUS_SUCCESS) {
            return std::move(m_operation);
        }
        if (m_operation.savedInVardesc)
            set_attribute(
                m_operation,
                CUDNN_ATTR_OPERATION_NORM_BWD_INV_VARIANCE_DESC,
                "CUDNN_BACKEND_OPERATION: SetAttribute CUDNN_ATTR_OPERATION_NORM_BWD_INV_VARIANCE_DESC Failed",
                &m_operation.savedInVardesc->get_backend_descriptor());
        if (status != CUDNN_STATUS_SUCCESS) {
            return std::move(m_operation);
        }
        if (m_operation.dydesc)
            set_attribute(m_operation,
                          CUDNN_ATTR_OPERATION_NORM_BWD_DYDESC,
                          "CUDNN_BACKEND_OPERATION: SetAttribute CUDNN_ATTR_OPERATION_NORM_BWD_DYDESC Failed",
                          &m_operation.dydesc->get_backend_descriptor());
        if (status != CUDNN_STATUS_SUCCESS) {
            return std::move(m_operation);
        }
        if (m_operation.scaledesc)
            set_attribute(m_operation,
                          CUDNN_ATTR_OPERATION_NORM_BWD_SCALE_DESC,
                          "CUDNN_BACKEND_OPERATION: SetAttribute CUDNN_ATTR_OPERATION_NORM_BWD_SCALE_DESC Failed",
                          &m_operation.scaledesc->get_backend_descriptor());
        if (status != CUDNN_STATUS_SUCCESS) {
            return std::move(m_operation);
        }
        if (m_operation.dxdesc)
            set_attribute(m_operation,
                          CUDNN_ATTR_OPERATION_NORM_BWD_DXDESC,
                          "CUDNN_BACKEND_OPERATION: SetAttribute CUDNN_ATTR_OPERATION_NORM_BWD_DXDESC Failed",
                          &m_operation.dxdesc->get_backend_descriptor());
        if (status != CUDNN_STATUS_SUCCESS) {
            return std::move(m_operation);
        }
        if (m_operation.dscaledesc)
            set_attribute(m_operation,
                          CUDNN_ATTR_OPERATION_NORM_BWD_DSCALE_DESC,
                          "CUDNN_BACKEND_OPERATION: SetAttribute CUDNN_ATTR_OPERATION_NORM_BWD_DSCALE_DESC Failed",
                          &m_operation.dscaledesc->get_backend_descriptor());
        if (status != CUDNN_STATUS_SUCCESS) {
            return std::move(m_operation);
        }
        if (m_operation.dbiasdesc)
            set_attribute(m_operation,
                          CUDNN_ATTR_OPERATION_NORM_BWD_DBIAS_DESC,
                          "CUDNN_BACKEND_OPERATION: SetAttribute CUDNN_ATTR_OPERATION_NORM_BWD_DBIAS_DESC Failed",
                          &m_operation.dbiasdesc->get_backend_descriptor());
        if (status != CUDNN_STATUS_SUCCESS) {
            return std::move(m_operation);
        }
        if (m_operation.peerStatdescs.size()) {
            std::vector<cudnnBackendDescriptor_t> backend_peer_stat_descs;
            for (auto &desc : m_operation.peerStatdescs) {
                backend_peer_stat_descs.push_back(desc->get_backend_descriptor());
            }
            set_attribute(
                m_operation,
                CUDNN_ATTR_OPERATION_NORM_BWD_PEER_STAT_DESCS,
                "CUDNN_BACKEND_OPERATION: SetAttribute CUDNCUDNN_ATTR_OPERATION_NORM_BWD_PEER_STAT_DESCS Failed",
                backend_peer_stat_descs.data(),
                CUDNN_TYPE_BACKEND_DESCRIPTOR,
                backend_peer_stat_descs.size());
        }
        if (status != CUDNN_STATUS_SUCCESS) {
            return std::move(m_operation);
        }
        if (m_operation.epsilondesc) {
            set_attribute(m_operation,
                          CUDNN_ATTR_OPERATION_NORM_BWD_EPSILON_DESC,
                          "CUDNN_BACKEND_OPERATION: SetAttribute CUDNN_ATTR_OPERATION_NORM_BWD_EPSILON Failed",
                          &m_operation.epsilondesc->get_backend_descriptor());
        }
        if (status != CUDNN_STATUS_SUCCESS) {
            return std::move(m_operation);
        }

        status = detail::finalize(m_operation.pointer->get_backend_descriptor());

        if (status != CUDNN_STATUS_SUCCESS) {
            set_error_and_throw_exception(&m_operation, status, "CUDNN_BACKEND_OPERATION: cudnnFinalize Failed");
            return std::move(m_operation);
        }
        return std::move(m_operation);
    }

    Operation_v8 &&
    build_resample_fwd_operation() {
        m_operation.operationTag = "Resample_fwd";
        auto status              = CUDNN_STATUS_SUCCESS;
        status                   = detail::set_attribute(m_operation.pointer->get_backend_descriptor(),
                                       CUDNN_ATTR_OPERATION_RESAMPLE_FWD_XDESC,
                                       CUDNN_TYPE_BACKEND_DESCRIPTOR,
                                       1,
                                       &(m_operation.xdesc->get_backend_descriptor()));
        if (status != CUDNN_STATUS_SUCCESS) {
            set_error_and_throw_exception(
                &m_operation,
                status,
                "CUDNN_BACKEND_OPERATION: SetAttribute CUDNN_ATTR_OPERATION_RESAMPLE_FWD_XDESC Failed");
            return std::move(m_operation);
        }
        status = detail::set_attribute(m_operation.pointer->get_backend_descriptor(),
                                       CUDNN_ATTR_OPERATION_RESAMPLE_FWD_YDESC,
                                       CUDNN_TYPE_BACKEND_DESCRIPTOR,
                                       1,
                                       &(m_operation.ydesc->get_backend_descriptor()));
        if (status != CUDNN_STATUS_SUCCESS) {
            set_error_and_throw_exception(
                &m_operation,
                status,
                "CUDNN_BACKEND_OPERATION: SetAttribute CUDNN_ATTR_OPERATION_RESAMPLE_FWD_YDESC Failed");
            return std::move(m_operation);
        }
        status = detail::set_attribute(m_operation.pointer->get_backend_descriptor(),
                                       CUDNN_ATTR_OPERATION_RESAMPLE_FWD_ALPHA,
                                       CUDNN_TYPE_DOUBLE,
                                       1,
                                       &(m_operation.alpha_d));
        if (status != CUDNN_STATUS_SUCCESS) {
            set_error_and_throw_exception(
                &m_operation,
                status,
                "CUDNN_BACKEND_OPERATION: SetAttribute CUDNN_ATTR_OPERATION_RESAMPLE_FWD_ALPHA Failed");
            return std::move(m_operation);
        }
        status = detail::set_attribute(m_operation.pointer->get_backend_descriptor(),
                                       CUDNN_ATTR_OPERATION_RESAMPLE_FWD_BETA,
                                       CUDNN_TYPE_DOUBLE,
                                       1,
                                       &(m_operation.beta_d));
        if (status != CUDNN_STATUS_SUCCESS) {
            set_error_and_throw_exception(
                &m_operation,
                status,
                "CUDNN_BACKEND_OPERATION: SetAttribute CUDNN_ATTR_OPERATION_RESAMPLE_FWD_BETA Failed");
            return std::move(m_operation);
        }
        status = detail::set_attribute(m_operation.pointer->get_backend_descriptor(),
                                       CUDNN_ATTR_OPERATION_RESAMPLE_FWD_DESC,
                                       CUDNN_TYPE_BACKEND_DESCRIPTOR,
                                       1,
                                       &(m_operation.resampledesc->get_backend_descriptor()));
        if (status != CUDNN_STATUS_SUCCESS) {
            set_error_and_throw_exception(
                &m_operation,
                status,
                "CUDNN_BACKEND_OPERATION: SetAttribute CUDNN_ATTR_OPERATION_RESAMPLE_FWD_DESC Failed");
            return std::move(m_operation);
        }

        // Maxpooling forward
        if (m_operation.idxdesc != nullptr) {
            status = detail::set_attribute(m_operation.pointer->get_backend_descriptor(),
                                           CUDNN_ATTR_OPERATION_RESAMPLE_FWD_IDXDESC,
                                           CUDNN_TYPE_BACKEND_DESCRIPTOR,
                                           1,
                                           &(m_operation.idxdesc->get_backend_descriptor()));
            if (status != CUDNN_STATUS_SUCCESS) {
                set_error_and_throw_exception(
                    &m_operation,
                    status,
                    "CUDNN_BACKEND_OPERATION: SetAttribute CUDNN_ATTR_OPERATION_RESAMPLE_FWD_IDXDESC Failed");
                return std::move(m_operation);
            }
        }

        status = detail::finalize(m_operation.pointer->get_backend_descriptor());
        if (status != CUDNN_STATUS_SUCCESS) {
            set_error_and_throw_exception(&m_operation, status, "CUDNN_BACKEND_OPERATION: cudnnFinalize Failed");
            return std::move(m_operation);
        }
        return std::move(m_operation);
    }

    Operation_v8 &&
    build_resample_bwd_operation() {
#if (CUDNN_VERSION >= 8600)
        NV_CUDNN_FE_DYNAMIC_CHECK_BACKEND_DESCRIPTOR(
            8600, m_operation, "CUDNN_BACKEND_OPERATION: Resample_bwd requires cudnn 8.6.0");
        m_operation.operationTag = "Resample_bwd";
        auto status              = CUDNN_STATUS_SUCCESS;
        status                   = detail::set_attribute(m_operation.pointer->get_backend_descriptor(),
                                       CUDNN_ATTR_OPERATION_RESAMPLE_BWD_DXDESC,
                                       CUDNN_TYPE_BACKEND_DESCRIPTOR,
                                       1,
                                       &(m_operation.dxdesc->get_backend_descriptor()));
        if (status != CUDNN_STATUS_SUCCESS) {
            set_error_and_throw_exception(
                &m_operation,
                status,
                "CUDNN_BACKEND_OPERATION: SetAttribute CUDNN_ATTR_OPERATION_RESAMPLE_BWD_DXDESC Failed");
            return std::move(m_operation);
        }
#if (CUDNN_VERSION >= 8700)
        if (m_operation.xdesc != nullptr) {
            NV_CUDNN_FE_DYNAMIC_CHECK_BACKEND_DESCRIPTOR(
                8700,
                m_operation,
                "CUDNN_BACKEND_OPERATION: SetAttribute CUDNN_ATTR_OPERATION_RESAMPLE_BWD_XDESC requires cudnn 8.7.0");
            status = detail::set_attribute(m_operation.pointer->get_backend_descriptor(),
                                           CUDNN_ATTR_OPERATION_RESAMPLE_BWD_XDESC,
                                           CUDNN_TYPE_BACKEND_DESCRIPTOR,
                                           1,
                                           &(m_operation.xdesc->get_backend_descriptor()));
            if (status != CUDNN_STATUS_SUCCESS) {
                set_error_and_throw_exception(
                    &m_operation,
                    status,
                    "CUDNN_BACKEND_OPERATION: SetAttribute CUDNN_ATTR_OPERATION_RESAMPLE_BWD_XDESC Failed");
                return std::move(m_operation);
            }
        }
        if (m_operation.ydesc != nullptr) {
            NV_CUDNN_FE_DYNAMIC_CHECK_BACKEND_DESCRIPTOR(
                8700,
                m_operation,
                "CUDNN_BACKEND_OPERATION: SetAttribute CUDNN_ATTR_OPERATION_RESAMPLE_BWD_YDESC requires cudnn 8.7.0");
            status = detail::set_attribute(m_operation.pointer->get_backend_descriptor(),
                                           CUDNN_ATTR_OPERATION_RESAMPLE_BWD_YDESC,
                                           CUDNN_TYPE_BACKEND_DESCRIPTOR,
                                           1,
                                           &(m_operation.ydesc->get_backend_descriptor()));
            if (status != CUDNN_STATUS_SUCCESS) {
                set_error_and_throw_exception(
                    &m_operation,
                    status,
                    "CUDNN_BACKEND_OPERATION: SetAttribute CUDNN_ATTR_OPERATION_RESAMPLE_BWD_YDESC Failed");
                return std::move(m_operation);
            }
        }
#endif
        status = detail::set_attribute(m_operation.pointer->get_backend_descriptor(),
                                       CUDNN_ATTR_OPERATION_RESAMPLE_BWD_DYDESC,
                                       CUDNN_TYPE_BACKEND_DESCRIPTOR,
                                       1,
                                       &(m_operation.dydesc->get_backend_descriptor()));
        if (status != CUDNN_STATUS_SUCCESS) {
            set_error_and_throw_exception(
                &m_operation,
                status,
                "CUDNN_BACKEND_OPERATION: SetAttribute CUDNN_ATTR_OPERATION_RESAMPLE_BWD_DYDESC Failed");
            return std::move(m_operation);
        }
        status = detail::set_attribute(m_operation.pointer->get_backend_descriptor(),
                                       CUDNN_ATTR_OPERATION_RESAMPLE_BWD_ALPHA,
                                       CUDNN_TYPE_DOUBLE,
                                       1,
                                       &(m_operation.alpha_d));
        if (status != CUDNN_STATUS_SUCCESS) {
            set_error_and_throw_exception(
                &m_operation,
                status,
                "CUDNN_BACKEND_OPERATION: SetAttribute CUDNN_ATTR_OPERATION_RESAMPLE_BWD_ALPHA Failed");
            return std::move(m_operation);
        }
        status = detail::set_attribute(m_operation.pointer->get_backend_descriptor(),
                                       CUDNN_ATTR_OPERATION_RESAMPLE_BWD_BETA,
                                       CUDNN_TYPE_DOUBLE,
                                       1,
                                       &(m_operation.beta_d));
        if (status != CUDNN_STATUS_SUCCESS) {
            set_error_and_throw_exception(
                &m_operation,
                status,
                "CUDNN_BACKEND_OPERATION: SetAttribute CUDNN_ATTR_OPERATION_RESAMPLE_BWD_BETA Failed");
            return std::move(m_operation);
        }
        status = detail::set_attribute(m_operation.pointer->get_backend_descriptor(),
                                       CUDNN_ATTR_OPERATION_RESAMPLE_BWD_DESC,
                                       CUDNN_TYPE_BACKEND_DESCRIPTOR,
                                       1,
                                       &(m_operation.resampledesc->get_backend_descriptor()));
        if (status != CUDNN_STATUS_SUCCESS) {
            set_error_and_throw_exception(
                &m_operation,
                status,
                "CUDNN_BACKEND_OPERATION: SetAttribute CUDNN_ATTR_OPERATION_RESAMPLE_BWD_DESC Failed");
            return std::move(m_operation);
        }

        // Maxpooling backward
        if (m_operation.idxdesc != nullptr) {
            status = detail::set_attribute(m_operation.pointer->get_backend_descriptor(),
                                           CUDNN_ATTR_OPERATION_RESAMPLE_BWD_IDXDESC,
                                           CUDNN_TYPE_BACKEND_DESCRIPTOR,
                                           1,
                                           &(m_operation.idxdesc->get_backend_descriptor()));
            if (status != CUDNN_STATUS_SUCCESS) {
                set_error_and_throw_exception(
                    &m_operation,
                    status,
                    "CUDNN_BACKEND_OPERATION: SetAttribute CUDNN_ATTR_OPERATION_RESAMPLE_BWD_IDXDESC Failed");
                return std::move(m_operation);
            }
        }

        status = detail::finalize(m_operation.pointer->get_backend_descriptor());
        if (status != CUDNN_STATUS_SUCCESS) {
            set_error_and_throw_exception(&m_operation, status, "CUDNN_BACKEND_OPERATION: cudnnFinalize Failed");
            return std::move(m_operation);
        }
#else
        set_error_and_throw_exception(&m_operation,
                                      CUDNN_STATUS_NOT_SUPPORTED,
                                      "CUDNN_BACKEND_OPERATION: Resample operation Not supported in this version");
#endif
        return std::move(m_operation);
    }

    Operation_v8 &&
    build_rng_operation() {
#if (CUDNN_VERSION >= 8700)
        NV_CUDNN_FE_DYNAMIC_CHECK_BACKEND_DESCRIPTOR(
            8700, m_operation, "CUDNN_BACKEND_OPERATION: build_rng_operation requires cudnn 8.7.0");
        m_operation.operationTag = "Rng";
        auto status              = CUDNN_STATUS_SUCCESS;
        status                   = detail::set_attribute(m_operation.pointer->get_backend_descriptor(),
                                       CUDNN_ATTR_OPERATION_RNG_YDESC,
                                       CUDNN_TYPE_BACKEND_DESCRIPTOR,
                                       1,
                                       &(m_operation.ydesc->get_backend_descriptor()));
        if (status != CUDNN_STATUS_SUCCESS) {
            set_error_and_throw_exception(
                &m_operation, status, "CUDNN_BACKEND_OPERATION: SetAttribute CUDNN_ATTR_OPERATION_RNG_YDESC Failed");
            return std::move(m_operation);
        }

#if (CUDNN_VERSION >= 8800)
        // seed can be a tensor or an int64
        // if tensor is defined we give it precedence
        if (m_operation.seeddesc) {
            NV_CUDNN_FE_DYNAMIC_CHECK_BACKEND_DESCRIPTOR(
                8800,
                m_operation,
                "CUDNN_BACKEND_OPERATION: SetAttribute CUDNN_ATTR_OPERATION_RNG_SEED requires cudnn 8.8.0");
            status = detail::set_attribute(m_operation.pointer->get_backend_descriptor(),
                                           CUDNN_ATTR_OPERATION_RNG_SEED,
                                           CUDNN_TYPE_BACKEND_DESCRIPTOR,
                                           1,
                                           &(m_operation.seeddesc->get_backend_descriptor()));
            if (status != CUDNN_STATUS_SUCCESS) {
                set_error_and_throw_exception(
                    &m_operation, status, "CUDNN_BACKEND_OPERATION: SetAttribute CUDNN_ATTR_OPERATION_RNG_SEED Failed");
                return std::move(m_operation);
            }
        } else
#endif
        {
            status = detail::set_attribute(m_operation.pointer->get_backend_descriptor(),
                                           CUDNN_ATTR_OPERATION_RNG_SEED,
                                           CUDNN_TYPE_INT64,
                                           1,
                                           &(m_operation.seed));
            if (status != CUDNN_STATUS_SUCCESS) {
                set_error_and_throw_exception(
                    &m_operation, status, "CUDNN_BACKEND_OPERATION: SetAttribute CUDNN_ATTR_OPERATION_RNG_SEED Failed");
                return std::move(m_operation);
            }
        }
        status = detail::set_attribute(m_operation.pointer->get_backend_descriptor(),
                                       CUDNN_ATTR_OPERATION_RNG_DESC,
                                       CUDNN_TYPE_BACKEND_DESCRIPTOR,
                                       1,
                                       &(m_operation.rngdesc->get_backend_descriptor()));
        if (status != CUDNN_STATUS_SUCCESS) {
            set_error_and_throw_exception(
                &m_operation, status, "CUDNN_BACKEND_OPERATION: SetAttribute CUDNN_ATTR_OPERATION_RNG_DESC Failed");
            return std::move(m_operation);
        }

#if (CUDNN_VERSION >= 8800)
        if (m_operation.offsetdesc) {
            NV_CUDNN_FE_DYNAMIC_CHECK_BACKEND_DESCRIPTOR(
                8800,
                m_operation,
                "CUDNN_BACKEND_OPERATION: SetAttribute CUDNN_ATTR_OPERATION_RNG_OFFSET_DESC requires cudnn 8.8.0");
            status = detail::set_attribute(m_operation.pointer->get_backend_descriptor(),
                                           CUDNN_ATTR_OPERATION_RNG_OFFSET_DESC,
                                           CUDNN_TYPE_BACKEND_DESCRIPTOR,
                                           1,
                                           &(m_operation.offsetdesc->get_backend_descriptor()));
            if (status != CUDNN_STATUS_SUCCESS) {
                set_error_and_throw_exception(
                    &m_operation,
                    status,
                    "CUDNN_BACKEND_OPERATION: SetAttribute CUDNN_ATTR_OPERATION_RNG_OFFSET_DESC Failed");
                return std::move(m_operation);
            }
        }
#endif

        status = detail::finalize(m_operation.pointer->get_backend_descriptor());
        if (status != CUDNN_STATUS_SUCCESS) {
            set_error_and_throw_exception(&m_operation, status, "CUDNN_BACKEND_OPERATION: cudnnFinalize Failed");
            return std::move(m_operation);
        }
#else
        set_error_and_throw_exception(&m_operation,
                                      CUDNN_STATUS_NOT_SUPPORTED,
                                      "CUDNN_BACKEND_OPERATION: Rng operation Not supported in this version");
#endif
        return std::move(m_operation);
    }

    Operation_v8 &&
    build_paged_cache_load_op() {
#if (CUDNN_VERSION < 90500)
        set_error_and_throw_exception(
            &m_operation,
            CUDNN_STATUS_NOT_SUPPORTED,
            "CUDNN_BACKEND_OPERATION: paged_cache_load_op operation Not supported in this version");
#else
        NV_CUDNN_FE_DYNAMIC_CHECK_BACKEND_DESCRIPTOR(
            90500, m_operation, "CUDNN_BACKEND_OPERATION: build_paged_cache_load_op requires cudnn 9.5.0");

        // Quick helper lambda to ensure code being DRY
        auto set_tensor_descriptor = [&](auto attr, const std::string &descriptor_name, auto &descriptor) {
            std::string error_msg = "CUDNN_BACKEND_OPERATION: Check and Set " + descriptor_name;
            auto status           = CUDNN_STATUS_SUCCESS;
            if (descriptor != nullptr) {
                status = detail::set_attribute(m_operation.pointer->get_backend_descriptor(),
                                               attr,
                                               CUDNN_TYPE_BACKEND_DESCRIPTOR,
                                               1,
                                               &(descriptor->get_backend_descriptor()));
            } else {
                status = CUDNN_STATUS_BAD_PARAM;
            }

            if (status != CUDNN_STATUS_SUCCESS) {
                set_error_and_throw_exception(&m_operation, status, error_msg.c_str());
            }
            return status;
        };

        if (CUDNN_STATUS_SUCCESS != set_tensor_descriptor(CUDNN_ATTR_OPERATION_PAGED_CACHE_LOAD_CONTAINER_DESC,
                                                          "CUDNN_ATTR_OPERATION_PAGED_CACHE_LOAD_CONTAINER_DESC",
                                                          m_operation.containerdesc)) {
            return std::move(m_operation);
        }

        if (CUDNN_STATUS_SUCCESS != set_tensor_descriptor(CUDNN_ATTR_OPERATION_PAGED_CACHE_LOAD_PAGE_TABLE_DESC,
                                                          "CUDNN_ATTR_OPERATION_PAGED_CACHE_LOAD_PAGE_TABLE_DESC",
                                                          m_operation.pageTabledesc)) {
            return std::move(m_operation);
        }

        if (CUDNN_STATUS_SUCCESS != set_tensor_descriptor(CUDNN_ATTR_OPERATION_PAGED_CACHE_LOAD_SEQUENCE_DESC,
                                                          "CUDNN_ATTR_OPERATION_PAGED_CACHE_SEQUENCE_DESC",
                                                          m_operation.sequencedesc)) {
            return std::move(m_operation);
        }

        if (CUDNN_STATUS_SUCCESS != set_tensor_descriptor(CUDNN_ATTR_OPERATION_PAGED_CACHE_LOAD_YDESC,
                                                          "CUDNN_ATTR_OPERATION_PAGED_CACHE_YDESC",
                                                          m_operation.ydesc)) {
            return std::move(m_operation);
        }

        auto status = detail::finalize(m_operation.pointer->get_backend_descriptor());
        if (status != CUDNN_STATUS_SUCCESS) {
            set_error_and_throw_exception(&m_operation, status, "CUDNN_BACKEND_OPERATION: cudnnFinalize Failed");
            return std::move(m_operation);
        }
#endif
        return std::move(m_operation);
    }

    Operation_v8 &&
    build_reshape_operation() {
#if (CUDNN_VERSION >= 8700)
        NV_CUDNN_FE_DYNAMIC_CHECK_BACKEND_DESCRIPTOR(
            8700, m_operation, "CUDNN_BACKEND_OPERATION: build_reshape_operation requires cudnn 8.7.0");
        m_operation.operationTag = "Reshape";
        auto status              = CUDNN_STATUS_SUCCESS;
        status                   = detail::set_attribute(m_operation.pointer->get_backend_descriptor(),
                                       CUDNN_ATTR_OPERATION_RESHAPE_XDESC,
                                       CUDNN_TYPE_BACKEND_DESCRIPTOR,
                                       1,
                                       &(m_operation.xdesc->get_backend_descriptor()));
        if (status != CUDNN_STATUS_SUCCESS) {
            set_error_and_throw_exception(
                &m_operation,
                status,
                "CUDNN_BACKEND_OPERATION: SetAttribute CUDNN_ATTR_OPERATION_RESHAPE_XDESC Failed");
            return std::move(m_operation);
        }
        status = detail::set_attribute(m_operation.pointer->get_backend_descriptor(),
                                       CUDNN_ATTR_OPERATION_RESHAPE_YDESC,
                                       CUDNN_TYPE_BACKEND_DESCRIPTOR,
                                       1,
                                       &(m_operation.ydesc->get_backend_descriptor()));
        if (status != CUDNN_STATUS_SUCCESS) {
            set_error_and_throw_exception(
                &m_operation,
                status,
                "CUDNN_BACKEND_OPERATION: SetAttribute CUDNN_ATTR_OPERATION_RESHAPE_YDESC Failed");
            return std::move(m_operation);
        }

#if (CUDNN_VERSION >= 92200)
        // Set reshape mode if it's not NOT_SET
        if (m_operation.reshape_mode != ReshapeMode_t::NOT_SET) {
            cudnnBackendReshapeMode_t cudnn_reshape_mode;
            status = detail::convert_to_cudnn_type(m_operation.reshape_mode, cudnn_reshape_mode);
            if (status == CUDNN_STATUS_SUCCESS) {
                status = detail::set_attribute(m_operation.pointer->get_backend_descriptor(),
                                               CUDNN_ATTR_OPERATION_RESHAPE_MODE,
                                               CUDNN_TYPE_RESHAPE_MODE,
                                               1,
                                               &cudnn_reshape_mode);
                if (status != CUDNN_STATUS_SUCCESS) {
                    set_error_and_throw_exception(
                        &m_operation,
                        status,
                        "CUDNN_BACKEND_OPERATION: SetAttribute CUDNN_ATTR_OPERATION_RESHAPE_MODE Failed");
                    return std::move(m_operation);
                }
            }
        }
#endif

        status = detail::finalize(m_operation.pointer->get_backend_descriptor());
        if (status != CUDNN_STATUS_SUCCESS) {
            set_error_and_throw_exception(&m_operation, status, "CUDNN_BACKEND_OPERATION: cudnnFinalize Failed");
            return std::move(m_operation);
        }
#else
        set_error_and_throw_exception(&m_operation,
                                      CUDNN_STATUS_NOT_SUPPORTED,
                                      "CUDNN_BACKEND_OPERATION: Reshape operation Not supported in this version");
#endif
        return std::move(m_operation);
    }

    Operation_v8 &&
    build_bn_bwd_weight_op() {
        m_operation.operationTag = "Dgrad_Drelu_BN_Bwd";
        auto status              = CUDNN_STATUS_SUCCESS;

        status = detail::set_attribute(m_operation.pointer->get_backend_descriptor(),
                                       CUDNN_ATTR_OPERATION_BN_BWD_WEIGHTS_MATH_PREC,
                                       CUDNN_TYPE_DATA_TYPE,
                                       1,
                                       &(m_operation.compute_type));
        if (status != CUDNN_STATUS_SUCCESS) {
            set_error_and_throw_exception(
                &m_operation,
                status,
                "CUDNN_BACKEND_OPERATION: SetAttribute CUDNN_ATTR_OPERATION_BN_BWD_WEIGHTS_MATH_PREC Failed");
            return std::move(m_operation);
        }

        auto set_attribute = [&status](Operation_v8 &operation,
                                       cudnnBackendAttributeName_t attr,
                                       const char *fail_msg,
                                       void const *ptr,
                                       cudnnBackendAttributeType_t type = CUDNN_TYPE_BACKEND_DESCRIPTOR,
                                       int64_t cnt                      = 1) {
            status = detail::set_attribute(operation.pointer->get_backend_descriptor(), attr, type, cnt, ptr);
            if (status != CUDNN_STATUS_SUCCESS) {
                set_error_and_throw_exception(&operation, status, fail_msg);
            }
        };

        if (m_operation.xdesc)
            set_attribute(m_operation,
                          CUDNN_ATTR_OPERATION_BN_BWD_WEIGHTS_X_DESC,
                          "CUDNN_BACKEND_OPERATION: SetAttribute CUDNN_ATTR_OPERATION_BN_BWD_WEIGHTS_X_DESC Failed",
                          &m_operation.xdesc->get_backend_descriptor());
        if (status != CUDNN_STATUS_SUCCESS) {
            return std::move(m_operation);
        }

        if (m_operation.savedMeandesc)
            set_attribute(m_operation,
                          CUDNN_ATTR_OPERATION_BN_BWD_WEIGHTS_MEAN_DESC,
                          "CUDNN_BACKEND_OPERATION: SetAttribute CUDNN_ATTR_OPERATION_BN_BWD_WEIGHTS_MEAN_DESC Failed",
                          &m_operation.savedMeandesc->get_backend_descriptor());
        if (status != CUDNN_STATUS_SUCCESS) {
            return std::move(m_operation);
        }

        if (m_operation.savedInVardesc)
            set_attribute(
                m_operation,
                CUDNN_ATTR_OPERATION_BN_BWD_WEIGHTS_INVSTD_DESC,
                "CUDNN_BACKEND_OPERATION: SetAttribute CUDNN_ATTR_OPERATION_BN_BWD_WEIGHTS_INVSTD_DESC Failed",
                &m_operation.savedInVardesc->get_backend_descriptor());
        if (status != CUDNN_STATUS_SUCCESS) {
            return std::move(m_operation);
        }

        if (m_operation.scaledesc)
            set_attribute(
                m_operation,
                CUDNN_ATTR_OPERATION_BN_BWD_WEIGHTS_BN_SCALE_DESC,
                "CUDNN_BACKEND_OPERATION: SetAttribute CUDNN_ATTR_OPERATION_BN_BWD_WEIGHTS_BN_SCALE_DESC Failed",
                &m_operation.scaledesc->get_backend_descriptor());
        if (status != CUDNN_STATUS_SUCCESS) {
            return std::move(m_operation);
        }

        if (m_operation.dydesc)
            set_attribute(m_operation,
                          CUDNN_ATTR_OPERATION_BN_BWD_WEIGHTS_DY_DESC,
                          "CUDNN_BACKEND_OPERATION: SetAttribute CUDNN_ATTR_OPERATION_BN_BWD_WEIGHTS_DY_DESC Failed",
                          &m_operation.dydesc->get_backend_descriptor());
        if (status != CUDNN_STATUS_SUCCESS) {
            return std::move(m_operation);
        }

        if (m_operation.dscaledesc)
            set_attribute(
                m_operation,
                CUDNN_ATTR_OPERATION_BN_BWD_WEIGHTS_DBN_SCALE_DESC,
                "CUDNN_BACKEND_OPERATION: SetAttribute CUDNN_ATTR_OPERATION_BN_BWD_WEIGHTS_DBN_SCALE_DESC Failed",
                &m_operation.dscaledesc->get_backend_descriptor());
        if (status != CUDNN_STATUS_SUCCESS) {
            return std::move(m_operation);
        }

        if (m_operation.dbiasdesc)
            set_attribute(
                m_operation,
                CUDNN_ATTR_OPERATION_BN_BWD_WEIGHTS_DBN_BIAS_DESC,
                "CUDNN_BACKEND_OPERATION: SetAttribute CUDNN_ATTR_OPERATION_BN_BWD_WEIGHTS_DBN_BIAS_DESC Failed",
                &m_operation.dbiasdesc->get_backend_descriptor());
        if (status != CUDNN_STATUS_SUCCESS) {
            return std::move(m_operation);
        }

        if (m_operation.eqscaledesc)
            set_attribute(
                m_operation,
                CUDNN_ATTR_OPERATION_BN_BWD_WEIGHTS_EQ_DY_SCALE_DESC,
                "CUDNN_BACKEND_OPERATION: SetAttribute CUDNN_ATTR_OPERATION_BN_BWD_WEIGHTS_EQ_DY_SCALE_DESC Failed",
                &m_operation.eqscaledesc->get_backend_descriptor());
        if (status != CUDNN_STATUS_SUCCESS) {
            return std::move(m_operation);
        }

        if (m_operation.eqscaledesc1)
            set_attribute(
                m_operation,
                CUDNN_ATTR_OPERATION_BN_BWD_WEIGHTS_EQ_X_SCALE_DESC,
                "CUDNN_BACKEND_OPERATION: SetAttribute CUDNN_ATTR_OPERATION_BN_BWD_WEIGHTS_EQ_X_SCALE_DESC Failed",
                &m_operation.eqscaledesc1->get_backend_descriptor());
        if (status != CUDNN_STATUS_SUCCESS) {
            return std::move(m_operation);
        }

        if (m_operation.eqbiasdesc)
            set_attribute(m_operation,
                          CUDNN_ATTR_OPERATION_BN_BWD_WEIGHTS_EQ_BIAS,
                          "CUDNN_BACKEND_OPERATION: SetAttribute CUDNN_ATTR_OPERATION_BN_BWD_WEIGHTS_EQ_BIAS Failed",
                          &m_operation.eqbiasdesc->get_backend_descriptor());
        if (status != CUDNN_STATUS_SUCCESS) {
            return std::move(m_operation);
        }
        status = detail::finalize(m_operation.pointer->get_backend_descriptor());
        if (status != CUDNN_STATUS_SUCCESS) {
            set_error_and_throw_exception(&m_operation, status, "CUDNN_BACKEND_OPERATION: cudnnFinalize Failed");
            return std::move(m_operation);
        }
        return std::move(m_operation);
    }

    Operation_v8 &&
    build_conv_forward() {
        m_operation.operationTag = "ConvFwd";

        auto status = CUDNN_STATUS_SUCCESS;

        status = detail::set_attribute(m_operation.pointer->get_backend_descriptor(),
                                       CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_X,
                                       CUDNN_TYPE_BACKEND_DESCRIPTOR,
                                       1,
                                       &(m_operation.xdesc->get_backend_descriptor()));
        if (status != CUDNN_STATUS_SUCCESS) {
            set_error_and_throw_exception(
                &m_operation,
                status,
                "CUDNN_BACKEND_OPERATION: SetAttribute CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_X Failed");
            return std::move(m_operation);
        }
        status = detail::set_attribute(m_operation.pointer->get_backend_descriptor(),
                                       CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_W,
                                       CUDNN_TYPE_BACKEND_DESCRIPTOR,
                                       1,
                                       &(m_operation.wdesc->get_backend_descriptor()));
        if (status != CUDNN_STATUS_SUCCESS) {
            set_error_and_throw_exception(
                &m_operation,
                status,
                "CUDNN_BACKEND_OPERATION: SetAttribute CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_W Failed");
            return std::move(m_operation);
        }
        status = detail::set_attribute(m_operation.pointer->get_backend_descriptor(),
                                       CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_Y,
                                       CUDNN_TYPE_BACKEND_DESCRIPTOR,
                                       1,
                                       &(m_operation.ydesc->get_backend_descriptor()));
        if (status != CUDNN_STATUS_SUCCESS) {
            set_error_and_throw_exception(
                &m_operation,
                status,
                "CUDNN_BACKEND_OPERATION: SetAttribute CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_Y Failed");
            return std::move(m_operation);
        }
        status = detail::set_attribute(m_operation.pointer->get_backend_descriptor(),
                                       CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_CONV_DESC,
                                       CUDNN_TYPE_BACKEND_DESCRIPTOR,
                                       1,
                                       &(m_operation.cdesc->get_backend_descriptor()));
        if (status != CUDNN_STATUS_SUCCESS) {
            set_error_and_throw_exception(
                &m_operation,
                status,
                "CUDNN_BACKEND_OPERATION: SetAttribute CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_CONV_DESC Failed");
            return std::move(m_operation);
        }
        void *alpha = (m_operation.alphabetaType == CUDNN_TYPE_FLOAT ? static_cast<void *>(&m_operation.alpha_s)
                                                                     : static_cast<void *>(&m_operation.alpha_d));
        void *beta  = (m_operation.alphabetaType == CUDNN_TYPE_FLOAT ? static_cast<void *>(&m_operation.beta_s)
                                                                     : static_cast<void *>(&m_operation.beta_d));
        status      = detail::set_attribute(m_operation.pointer->get_backend_descriptor(),
                                       CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_ALPHA,
                                       m_operation.alphabetaType,
                                       1,
                                       alpha);
        if (status != CUDNN_STATUS_SUCCESS) {
            set_error_and_throw_exception(
                &m_operation,
                status,
                "CUDNN_BACKEND_OPERATION: SetAttribute CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_ALPHA Failed");
            return std::move(m_operation);
        }
        status = detail::set_attribute(m_operation.pointer->get_backend_descriptor(),
                                       CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_BETA,
                                       m_operation.alphabetaType,
                                       1,
                                       beta);
        if (status != CUDNN_STATUS_SUCCESS) {
            set_error_and_throw_exception(
                &m_operation,
                status,
                "CUDNN_BACKEND_OPERATION: SetAttribute CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_BETA Failed");
            return std::move(m_operation);
        }
        status = detail::finalize(m_operation.pointer->get_backend_descriptor());
        if (status != CUDNN_STATUS_SUCCESS) {
            set_error_and_throw_exception(&m_operation, status, "CUDNN_BACKEND_OPERATION: cudnnFinalize Failed");
            return std::move(m_operation);
        }

        CUDNN_FE_LOG_LABEL_ENDL("Extracting the feature vector");
        extract_feature_vector(DescriptorType_t::OPERATION_CONVOLUTION_FORWARD_DESCRIPTOR);
        return std::move(m_operation);
    }

    void
    extract_feature_vector(DescriptorType_t op_type) {
        /// Build the feature vector of this operation now.
        m_operation.feature_vector.reserve(50);

        m_operation.feature_vector.push_back(static_cast<int>(op_type));
        for (auto i = 0; i < tensor_dims; i++) {
            m_operation.feature_vector.push_back(xTensor_dimA[i]);  // n, c, (g), d, h , w
        }
        for (auto i = 0; i < tensor_dims; i++) {
            m_operation.feature_vector.push_back(wTensor_dimA[i]);  // n, c, (g), d, h , w
        }
        for (auto i = 0; i < tensor_dims; i++) {
            m_operation.feature_vector.push_back(yTensor_dimA[i]);  // n, c, (g), d, h , w
        }
        const int max_spatial_dim = 3;

        /// Padding
        for (auto i = 0; i < max_spatial_dim; i++) {
            if (i == max_spatial_dim - 1 && is2D) {
                m_operation.feature_vector.push_back(0);
            } else {
                m_operation.feature_vector.push_back(conv_padding[i]);
            }
        }
        /// Dilation
        for (auto i = 0; i < max_spatial_dim; i++) {
            if (i == max_spatial_dim - 1 && is2D) {
                m_operation.feature_vector.push_back(0);
            } else {
                m_operation.feature_vector.push_back(conv_dilation[i]);
            }
        }
        /// Strides
        for (auto i = 0; i < max_spatial_dim; i++) {
            if (i == max_spatial_dim - 1 && is2D) {
                m_operation.feature_vector.push_back(0);
            } else {
                m_operation.feature_vector.push_back(conv_stride[i]);
            }
        }

        m_operation.feature_vector.push_back(xType);
        m_operation.feature_vector.push_back(wType);
        m_operation.feature_vector.push_back(yType);
        m_operation.feature_vector.push_back(cType);
        m_operation.feature_vector.push_back(mode);

        for (auto i = 0; i < tensor_dims; i++) {
            m_operation.feature_vector.push_back(xTensor_strA[i]);  // n, c, (g), d, h , w
        }
        for (auto i = 0; i < tensor_dims; i++) {
            m_operation.feature_vector.push_back(wTensor_strA[i]);  // n, c, (g), d, h , w
        }
        for (auto i = 0; i < tensor_dims; i++) {
            m_operation.feature_vector.push_back(yTensor_strA[i]);  // n, c, (g), d, h , w
        }

        int64_t alpha_as_int;
        int64_t beta_as_int;
        std::memcpy((void *)&alpha_as_int, (void *)(&m_operation.alpha_s), sizeof(int64_t));
        std::memcpy((void *)&beta_as_int, (void *)(&m_operation.beta_s), sizeof(int64_t));

        m_operation.feature_vector.push_back(alpha_as_int);
        m_operation.feature_vector.push_back(beta_as_int);
    }

    cudnnStatus_t
    validate_matmul_op(Message_t &msg) {
        if (m_operation.matmuldesc == nullptr) {
            msg = "CUDNN_BACKEND_OPERATION: Check and Set the CUDNN_ATTR_OPERATION_MATMUL_DESC";
            return CUDNN_STATUS_BAD_PARAM;
        }
        if (m_operation.amatdesc == nullptr) {
            msg = "CUDNN_BACKEND_OPERATION: Check and Set the CUDNN_ATTR_OPERATION_MATMUL_ADESC";
            return CUDNN_STATUS_BAD_PARAM;
        }
        if (m_operation.bmatdesc == nullptr) {
            msg = "CUDNN_BACKEND_OPERATION: Check and Set the CUDNN_ATTR_OPERATION_MATMUL_BDESC";
            return CUDNN_STATUS_BAD_PARAM;
        }
        if (m_operation.cmatdesc == nullptr) {
            msg = "CUDNN_BACKEND_OPERATION: Check and Set the CUDNN_ATTR_OPERATION_MATMUL_CDESC";
            return CUDNN_STATUS_BAD_PARAM;
        }
        return CUDNN_STATUS_SUCCESS;
    }

    cudnnStatus_t
    validate_norm_op(Message_t &msg) {
        cudnnStatus_t status = CUDNN_STATUS_SUCCESS;
        if (m_operation.xdesc == nullptr) {
            msg = "CUDNN_BACKEND_OPERATION: Check and Set the CUDNN_ATTR_OPERATION_NORM.*XDESC";
            return CUDNN_STATUS_BAD_PARAM;
        }

        if (detail::get_backend_version() == 8500) {
            std::array<int64_t, 10> x_dimensions;
            int64_t dim_count;
            status = detail::get_attribute(m_operation.xdesc->get_backend_descriptor(),
                                           CUDNN_ATTR_TENSOR_DIMENSIONS,
                                           CUDNN_TYPE_INT64,
                                           x_dimensions.size(),
                                           &dim_count,
                                           x_dimensions.data());
            if (status != CUDNN_STATUS_SUCCESS) {
                msg = "CUDNN_BACKEND_OPERATION: CUDNN_BACKEND_TENSOR has invalid CUDNN_ATTR_TENSOR_DIMENSIONS";
                return status;
            }

            int64_t N = x_dimensions[0];
            int64_t C = x_dimensions[1];

            if ((N != 1) || ((C % 8) != 0)) {
                msg = "CUDNN_BACKEND_OPERATION: CUDNN_BACKEND_TENSOR has bad CUDNN_ATTR_TENSOR_DIMENSIONS";
                return CUDNN_STATUS_BAD_PARAM;
            }
        }

        return status;
    }

    cudnnStatus_t
    validate_resample_op(Message_t &msg) {
        if (m_operation.op_mode == DescriptorType_t::OPERATION_RESAMPLE_FWD_DESCRIPTOR) {
            if (m_operation.xdesc == nullptr) {
                msg = "CUDNN_BACKEND_OPERATION: Check and Set the CUDNN_ATTR_OPERATION_RESAMPLE.*XDESC";
                return CUDNN_STATUS_BAD_PARAM;
            }
            if (m_operation.ydesc == nullptr) {
                msg = "CUDNN_BACKEND_OPERATION: Check and Set the CUDNN_ATTR_OPERATION_RESAMPLE.*YDESC";
                return CUDNN_STATUS_BAD_PARAM;
            }
        } else if (m_operation.op_mode == DescriptorType_t::OPERATION_RESAMPLE_BWD_DESCRIPTOR) {
            if (m_operation.dxdesc == nullptr) {
                msg = "CUDNN_BACKEND_OPERATION: Check and Set the CUDNN_ATTR_OPERATION_RESAMPLE.*DXDESC";
                return CUDNN_STATUS_BAD_PARAM;
            }
            if (m_operation.dydesc == nullptr) {
                msg = "CUDNN_BACKEND_OPERATION: Check and Set the CUDNN_ATTR_OPERATION_RESAMPLE.*DYDESC";
                return CUDNN_STATUS_BAD_PARAM;
            }
        }

        if (m_operation.resampledesc == nullptr) {
            msg = "CUDNN_BACKEND_OPERATION: Check and Set the CUDNN_ATTR_OPERATION_RESAMPLE.*RESAMPLEDESC";
            return CUDNN_STATUS_BAD_PARAM;
        }

        return CUDNN_STATUS_SUCCESS;
    }

    cudnnStatus_t
    validate_rng_op(Message_t &msg) {
        if (m_operation.ydesc == nullptr) {
            msg = "CUDNN_BACKEND_OPERATION: Check and Set the CUDNN_ATTR_OPERATION_REDUCTION_YDESC";
            return CUDNN_STATUS_BAD_PARAM;
        }

        if (m_operation.rngdesc == nullptr) {
            msg = "CUDNN_BACKEND_OPERATION: Check and Set the CUDNN_ATTR_OPERATION_RNG_DESC";
            return CUDNN_STATUS_BAD_PARAM;
        }

        return CUDNN_STATUS_SUCCESS;
    }

    cudnnStatus_t
    validate_reshape_op(Message_t &msg) {
        if (m_operation.xdesc == nullptr) {
            msg = "CUDNN_BACKEND_OPERATION: Check and Set the CUDNN_ATTR_OPERATION_RESHAPE_XDESC";
            return CUDNN_STATUS_BAD_PARAM;
        }

        if (m_operation.ydesc == nullptr) {
            msg = "CUDNN_BACKEND_OPERATION: Check and Set the CUDNN_ATTR_OPERATION_RESHAPE_YDESC";
            return CUDNN_STATUS_BAD_PARAM;
        }

        return CUDNN_STATUS_SUCCESS;
    }

    cudnnStatus_t
    validate_bn_bwd_weight_op(Message_t &msg) {
        if (m_operation.xdesc == nullptr) {
            msg = "CUDNN_BACKEND_OPERATION: Check and Set the CUDNN_ATTR_OPERATION_BN_BWD_WEIGHTS_X_DESC";
            return CUDNN_STATUS_BAD_PARAM;
        }

        if (m_operation.dydesc == nullptr) {
            msg = "CUDNN_BACKEND_OPERATION: Check and Set the CUDNN_ATTR_OPERATION_BN_BWD_WEIGHTS_DY_DESC";
            return CUDNN_STATUS_BAD_PARAM;
        }

        if (m_operation.savedMeandesc == nullptr) {
            msg = "CUDNN_BACKEND_OPERATION: Check and Set the CUDNN_ATTR_OPERATION_BN_BWD_WEIGHTS_MEAN_DESC";
            return CUDNN_STATUS_BAD_PARAM;
        }

        if (m_operation.savedInVardesc == nullptr) {
            msg = "CUDNN_BACKEND_OPERATION: Check and Set the CUDNN_ATTR_OPERATION_BN_BWD_WEIGHTS_INVSTD_DESC";
            return CUDNN_STATUS_BAD_PARAM;
        }

        return CUDNN_STATUS_SUCCESS;
    }

    cudnnStatus_t
    validate_reduction_op(Message_t &msg) {
        if (m_operation.reductiondesc == nullptr) {
            msg = "CUDNN_BACKEND_OPERATION: Check and Set the CUDNN_ATTR_OPERATION_REDUCTION_DESC";
            return CUDNN_STATUS_BAD_PARAM;
        }
        if (m_operation.xdesc == nullptr) {
            msg = "CUDNN_BACKEND_OPERATION: Check and Set the CUDNN_ATTR_OPERATION_REDUCTION_XDESC";
            return CUDNN_STATUS_BAD_PARAM;
        }
        if (m_operation.ydesc == nullptr) {
            msg = "CUDNN_BACKEND_OPERATION: Check and Set the CUDNN_ATTR_OPERATION_REDUCTION_YDESC";
            return CUDNN_STATUS_BAD_PARAM;
        }
        return CUDNN_STATUS_SUCCESS;
    }

    cudnnStatus_t
    validate_pointwise_op(Message_t &msg) {
        if (m_operation.xdesc == nullptr) {
            msg = "CUDNN_BACKEND_OPERATION: Check and Set the CUDNN_ATTR_OPERATION_POINTWISE_XDESC";
            return CUDNN_STATUS_BAD_PARAM;
        }
        if (m_operation.is_pointwise_math_op) {
            if (m_operation.pointwise_port_count == 3 && m_operation.bdesc == nullptr) {
                msg = "CUDNN_BACKEND_OPERATION: Check and Set the CUDNN_ATTR_OPERATION_POINTWISE_BDESC";
                return CUDNN_STATUS_BAD_PARAM;
            }
            if (m_operation.ydesc == nullptr) {
                msg = "CUDNN_BACKEND_OPERATION: Check and Set the CUDNN_ATTR_OPERATION_POINTWISE_YDESC";
                return CUDNN_STATUS_BAD_PARAM;
            }
        } else if (m_operation.is_pointwise_activation_fwd_op || m_operation.is_pointwise_identity_op) {
            if (m_operation.ydesc == nullptr) {
                msg = "CUDNN_BACKEND_OPERATION: Check and Set the CUDNN_ATTR_OPERATION_POINTWISE_YDESC";
                return CUDNN_STATUS_BAD_PARAM;
            }
        } else if (m_operation.is_pointwise_activation_bwd_op) {
            if (m_operation.dydesc == nullptr) {
                msg = "CUDNN_BACKEND_OPERATION: Check and Set the CUDNN_ATTR_OPERATION_POINTWISE_DYDESC";
                return CUDNN_STATUS_BAD_PARAM;
            }
            if (m_operation.dxdesc == nullptr) {
                msg = "CUDNN_BACKEND_OPERATION: Check and Set the CUDNN_ATTR_OPERATION_POINTWISE_DXDESC";
                return CUDNN_STATUS_BAD_PARAM;
            }
        } else {
            msg = "CUDNN_BACKEND_OPERATION: Unsupported cudnn pointwise mode. Check PointwiseMode_t::*";
            return CUDNN_STATUS_BAD_PARAM;
        }
        return CUDNN_STATUS_SUCCESS;
    }

    cudnnStatus_t
    validate_convolution_op(Message_t &msg) {
        if (m_operation.cdesc == nullptr) {
            msg = "CUDNN_BACKEND_OPERATION: Check and Set the CUDNN_ATTR_OPERATION_CONVOLUTION_*_CONV_DESC";
            return CUDNN_STATUS_BAD_PARAM;
        }
        if (m_operation.op_mode == DescriptorType_t::OPERATION_CONVOLUTION_FORWARD_DESCRIPTOR) {
            if (m_operation.xdesc == nullptr) {
                msg = "CUDNN_BACKEND_OPERATION: Check and Set the CUDNN_ATTR_OPERATION_CONVOLUTION_*_X";
                return CUDNN_STATUS_BAD_PARAM;
            }
            if (m_operation.wdesc == nullptr) {
                msg = "CUDNN_BACKEND_OPERATION: Check and Set the CUDNN_ATTR_OPERATION_CONVOLUTION_*_W";
                return CUDNN_STATUS_BAD_PARAM;
            }
            if (m_operation.ydesc == nullptr) {
                msg = "CUDNN_BACKEND_OPERATION: Check and Set the CUDNN_ATTR_OPERATION_CONVOLUTION_*_Y";
                return CUDNN_STATUS_BAD_PARAM;
            }

        } else if (m_operation.op_mode == DescriptorType_t::OPERATION_CONVOLUTION_BACKWARD_FILTER_DESCRIPTOR) {
            if (m_operation.ydesc != nullptr && m_operation.dydesc != nullptr) {
                msg =
                    "CUDNN_BACKEND_OPERATION: Ambiguous specification. Choose and Set only one of setyDesc() or "
                    "setdyDesc()";
                return CUDNN_STATUS_BAD_PARAM;
            }
            if (m_operation.ydesc == nullptr && m_operation.dydesc == nullptr) {
                msg = "CUDNN_BACKEND_OPERATION: Choose and Set one of setyDesc() or setdyDesc()";
                return CUDNN_STATUS_BAD_PARAM;
            }
            if (m_operation.xdesc == nullptr) {
                msg = "CUDNN_BACKEND_OPERATION: Check and Set the CUDNN_ATTR_OPERATION_CONVOLUTION_*_X";
                return CUDNN_STATUS_BAD_PARAM;
            }
            if (m_operation.wdesc != nullptr && m_operation.dwdesc != nullptr) {
                msg =
                    "CUDNN_BACKEND_OPERATION: Ambiguous specification. Choose and Set only one of setwDesc() or "
                    "setdwDesc()";
                return CUDNN_STATUS_BAD_PARAM;
            }
            if (m_operation.wdesc == nullptr && m_operation.dwdesc == nullptr) {
                msg = "CUDNN_BACKEND_OPERATION: Choose and Set one of setwDesc() or setdwDesc()";
                return CUDNN_STATUS_BAD_PARAM;
            }
        } else if (m_operation.op_mode == DescriptorType_t::OPERATION_CONVOLUTION_BACKWARD_DATA_DESCRIPTOR) {
            if (m_operation.ydesc != nullptr && m_operation.dydesc != nullptr) {
                msg =
                    "CUDNN_BACKEND_OPERATION: Ambiguous specification. Choose and Set only one of setyDesc() or "
                    "setdyDesc()";
                return CUDNN_STATUS_BAD_PARAM;
            }
            if (m_operation.ydesc == nullptr && m_operation.dydesc == nullptr) {
                msg = "CUDNN_BACKEND_OPERATION: Choose and Set one of setyDesc() or setdyDesc()";
                return CUDNN_STATUS_BAD_PARAM;
            }
            if (m_operation.wdesc == nullptr) {
                msg = "CUDNN_BACKEND_OPERATION: Check and Set the CUDNN_ATTR_OPERATION_CONVOLUTION_*_W";
                return CUDNN_STATUS_BAD_PARAM;
            }
            if (m_operation.xdesc != nullptr && m_operation.dxdesc != nullptr) {
                msg =
                    "CUDNN_BACKEND_OPERATION: Ambiguous specification. Choose and Set only one of setxDesc() or "
                    "setdxDesc()";
                return CUDNN_STATUS_BAD_PARAM;
            }
            if (m_operation.xdesc == nullptr && m_operation.dxdesc == nullptr) {
                msg = "CUDNN_BACKEND_OPERATION: Choose and Set one of setxDesc() or setdxDesc()";
                return CUDNN_STATUS_BAD_PARAM;
            }
        } else {
            msg =
                "CUDNN_BACKEND_OPERATION: Unsupported convolution operation. Check and set "
                "CUDNN_BACKEND_OPERATION_CONVOLUTION_*_DESCRIPTOR";
            return CUDNN_STATUS_BAD_PARAM;
        }
        return CUDNN_STATUS_SUCCESS;
    }

    void
    copy_dims_and_strides(const int64_t *from, int64_t *to) const {
        for (auto i = 0; i < CUDNN_DIM_MAX + 1; i++) {
            to[i] = from[i];
        }
    }

   public:
    /** @defgroup OperationBuilder_v8
     *  Set individual property of Operation_v8 class
     *  @{
     */
    /// Will be Deprecated Do not use
    auto
    setxDesc(ManagedOpaqueDescriptor const &raw_tensor) -> OperationBuilder_v8 & {
        m_operation.xdesc = raw_tensor;
        return *this;
    }

    auto
    setxDesc(Tensor_v8 const &tensor) -> OperationBuilder_v8 & {
        m_operation.xdesc = tensor.get_desc();
        copy_dims_and_strides(tensor.getDimArray(), xTensor_dimA);
        copy_dims_and_strides(tensor.getStrideArray(), xTensor_strA);
        tensor_dims = tensor.getDimensionCount();
        xType       = tensor.getDataType();
        return *this;
    }
    auto
    setbDesc(Tensor_v8 const &tensor) -> OperationBuilder_v8 & {
        if (is_pointwise_op == false) {
            set_error_and_throw_exception(
                &m_operation,
                CUDNN_STATUS_BAD_PARAM,
                "CUDNN_BACKEND_OPERATION_*_DESCRIPTOR: Non Pointwise operation does not need bTensor");
        }
        m_operation.bdesc = tensor.get_desc();
        return *this;
    }

    auto
    settDesc(Tensor_v8 const &tensor) -> OperationBuilder_v8 & {
        if (is_pointwise_op == false) {
            set_error_and_throw_exception(
                &m_operation,
                CUDNN_STATUS_BAD_PARAM,
                "CUDNN_BACKEND_OPERATION_*_DESCRIPTOR: Non Pointwise operation does not need tTensor");
        }
        m_operation.tdesc = tensor.get_desc();
        return *this;
    }

    auto
    setyDesc(Tensor_v8 const &tensor) -> OperationBuilder_v8 & {
        m_operation.ydesc = tensor.get_desc();
        copy_dims_and_strides(tensor.getDimArray(), yTensor_dimA);
        copy_dims_and_strides(tensor.getStrideArray(), yTensor_strA);
        yType = tensor.getDataType();
        return *this;
    }
    auto
    setwDesc(Tensor_v8 const &tensor) -> OperationBuilder_v8 & {
        if (is_convolution_op == false) {
            set_error_and_throw_exception(
                &m_operation,
                CUDNN_STATUS_BAD_PARAM,
                "CUDNN_BACKEND_OPERATION_*_DESCRIPTOR: Non Convolution operation does not need wTensor");
        }
        m_operation.wdesc = tensor.get_desc();
        copy_dims_and_strides(tensor.getDimArray(), wTensor_dimA);
        copy_dims_and_strides(tensor.getStrideArray(), wTensor_strA);
        wType = tensor.getDataType();
        return *this;
    }

    /// Will be Deprecated Do not use
    auto
    setdyDesc(ManagedOpaqueDescriptor const &raw_tensor) -> OperationBuilder_v8 & {
        m_operation.dydesc = raw_tensor;
        return *this;
    }
    auto
    setdyDesc(Tensor_v8 const &tensor) -> OperationBuilder_v8 & {
        m_operation.dydesc = tensor.get_desc();
        copy_dims_and_strides(tensor.getDimArray(), yTensor_dimA);
        copy_dims_and_strides(tensor.getStrideArray(), yTensor_strA);
        yType = tensor.getDataType();
        return *this;
    }
    auto
    setdxDesc(Tensor_v8 const &tensor) -> OperationBuilder_v8 & {
        m_operation.dxdesc = tensor.get_desc();
        copy_dims_and_strides(tensor.getDimArray(), xTensor_dimA);
        copy_dims_and_strides(tensor.getStrideArray(), xTensor_strA);
        tensor_dims = tensor.getDimensionCount();
        xType       = tensor.getDataType();
        return *this;
    }
    auto
    setdwDesc(Tensor_v8 const &tensor) -> OperationBuilder_v8 & {
        m_operation.dwdesc = tensor.get_desc();
        copy_dims_and_strides(tensor.getDimArray(), wTensor_dimA);
        copy_dims_and_strides(tensor.getStrideArray(), wTensor_strA);
        wType = tensor.getDataType();
        return *this;
    }
    auto
    setResampleDesc(ResampleDesc_v8 const &resampleDesc) -> OperationBuilder_v8 & {
        if (is_resample_fwd_op == false && is_resample_bwd_op == false) {
            set_error_and_throw_exception(&m_operation,
                                          CUDNN_STATUS_BAD_PARAM,
                                          "RESAMPLE_DESC: Non Resample operation does not need Resample DESCRIPTOR");
        }
        m_operation.resampledesc = resampleDesc.get_desc();
        return *this;
    }

    auto
    setRngDesc(RngDesc_v8 const &rngDesc) -> OperationBuilder_v8 & {
        if (is_rng_op == false) {
            set_error_and_throw_exception(
                &m_operation, CUDNN_STATUS_BAD_PARAM, "RNG_DESC: Non Rng operation does not need Rng DESCRIPTOR");
        }
        m_operation.rngdesc = rngDesc.get_desc();
        return *this;
    }

    auto
    setidxDesc(Tensor_v8 const &tensor) -> OperationBuilder_v8 & {
        m_operation.idxdesc = tensor.get_desc();
        copy_dims_and_strides(tensor.getDimArray(), idxTensor_dimA);
        copy_dims_and_strides(tensor.getStrideArray(), idxTensor_strA);
        idxType = tensor.getDataType();
        return *this;
    }

    auto
    setSeedDesc(Tensor_v8 const &tensor) -> OperationBuilder_v8 & {
        m_operation.seeddesc = tensor.get_desc();
        return *this;
    }

    auto
    setOffsetDesc(Tensor_v8 const &tensor) -> OperationBuilder_v8 & {
        m_operation.offsetdesc = tensor.get_desc();
        return *this;
    }

    auto
    setcDesc(ConvDesc_v8 const &conv) -> OperationBuilder_v8 & {
        if (is_convolution_op == false) {
            set_error_and_throw_exception(
                &m_operation,
                CUDNN_STATUS_BAD_PARAM,
                "CUDNN_BACKEND_OPERATION_*_DESCRIPTOR: Non Convolution operation does not need Convolution DESCRIPTOR");
        }
        m_operation.cdesc = conv.get_desc();
        if (conv.getComputePrecision() == DataType_t::DOUBLE) {
            m_operation.alphabetaType = CUDNN_TYPE_DOUBLE;
        }
        is2D = conv.getDimensionCount() == 2;
        copy_dims_and_strides(conv.getPadding(), conv_padding);
        copy_dims_and_strides(conv.getDilation(), conv_dilation);
        copy_dims_and_strides(conv.getStride(), conv_stride);
        cType = static_cast<int>(conv.getComputePrecision());
        mode  = conv.getMathMode();
        return *this;
    }

    auto
    setcontainerDesc(Tensor_v8 const &tensor) -> OperationBuilder_v8 & {
        m_operation.containerdesc = tensor.get_desc();
        return *this;
    }

    auto
    setpageTableDesc(Tensor_v8 const &tensor) -> OperationBuilder_v8 & {
        m_operation.pageTabledesc = tensor.get_desc();
        return *this;
    }

    auto
    setsequenceDesc(Tensor_v8 const &tensor) -> OperationBuilder_v8 & {
        m_operation.sequencedesc = tensor.get_desc();
        return *this;
    }

    auto
    setNormFwdPhase(NormFwdPhase_t mode) -> OperationBuilder_v8 & {
        m_operation.norm_fwd_phase = mode;
        return *this;
    }

    auto
    setReshapeMode(ReshapeMode_t mode) -> OperationBuilder_v8 & {
        m_operation.reshape_mode = mode;
        return *this;
    }

#if (CUDNN_VERSION >= 92200)
    // To be deprecated. Please use setReshapeMode(cudnn_frontend::ReshapeMode_t mode) instead.
    auto
    setReshapeMode(cudnnBackendReshapeMode_t mode) -> OperationBuilder_v8 & {
        detail::convert_from_cudnn_type(mode, m_operation.reshape_mode);
        return *this;
    }
#endif

    auto
    setNormalizationMode(NormMode_t mode) -> OperationBuilder_v8 & {
        m_operation.norm_mode = mode;
        return *this;
    }

    // To be deprecated. Please use setNormalizationMode(cudnn_frontend::NormMode_t mode) instead.
    auto
    setNormalizationMode(cudnnBackendNormMode_t mode) -> OperationBuilder_v8 & {
        detail::convert_from_cudnn_type(mode, m_operation.norm_mode);
        return *this;
    }

    // To be deprecated. Please use setNormFwdPhase(cudnn_frontend::NormFwdPhase_t mode) instead.
    auto
    setNormFwdPhase(cudnnBackendNormFwdPhase_t mode) -> OperationBuilder_v8 & {
        detail::convert_from_cudnn_type(mode, m_operation.norm_fwd_phase);
        return *this;
    }

    auto
    setBNFinalizeMode(cudnnBnFinalizeStatsMode_t mode) -> OperationBuilder_v8 & {
        m_operation.bn_stats_mode = mode;
        return *this;
    }

    auto
    setAccumCountTensor(Tensor_v8 const &tensor) -> OperationBuilder_v8 & {
        m_operation.accumCountdesc = tensor.get_desc();
        return *this;
    }

    auto
    setEpsilonTensor(Tensor_v8 const &tensor) -> OperationBuilder_v8 & {
        m_operation.epsilondesc = tensor.get_desc();
        return *this;
    }

    auto
    setExpDecayFactorTensor(Tensor_v8 const &tensor) -> OperationBuilder_v8 & {
        m_operation.expDecayFactordesc = tensor.get_desc();
        return *this;
    }

    auto
    addPeerStatTensor(Tensor_v8 const &peer_stat_tensor) -> OperationBuilder_v8 & {
        m_operation.peerStatdescs.push_back(peer_stat_tensor.get_desc());
        return *this;
    }

    auto
    setPeerStatTensor(std::vector<Tensor_v8> const &peer_stat_tensors) -> OperationBuilder_v8 & {
        for (auto &tensor : peer_stat_tensors) {
            m_operation.peerStatdescs.push_back(tensor.get_desc());
        }
        return *this;
    }

    auto
    setPrevRunningMeanAndVar(Tensor_v8 const &mean, Tensor_v8 const &var) -> OperationBuilder_v8 & {
        m_operation.prevMeandesc = mean.get_desc();
        m_operation.prevVardesc  = var.get_desc();
        return *this;
    }

    auto
    setNextRunningMeanAndVar(Tensor_v8 const &mean, Tensor_v8 const &var) -> OperationBuilder_v8 & {
        m_operation.nextMeandesc = mean.get_desc();
        m_operation.nextVardesc  = var.get_desc();
        return *this;
    }

    auto
    setSavedMeanAndInvVar(Tensor_v8 const &mean, Tensor_v8 const &var) -> OperationBuilder_v8 & {
        m_operation.savedMeandesc  = mean.get_desc();
        m_operation.savedInVardesc = var.get_desc();
        return *this;
    }

    auto
    setSavedInvVar(Tensor_v8 const &var) -> OperationBuilder_v8 & {
        m_operation.savedInVardesc = var.get_desc();
        return *this;
    }

    auto
    setScale(Tensor_v8 const &scale_tensor) -> OperationBuilder_v8 & {
        m_operation.scaledesc = scale_tensor.get_desc();
        return *this;
    }

    auto
    setBias(Tensor_v8 const &bias_tensor) -> OperationBuilder_v8 & {
        m_operation.biasdesc = bias_tensor.get_desc();
        return *this;
    }

    auto
    setScaleAndBias(Tensor_v8 const &scale_tensor, Tensor_v8 const &bias_tensor) -> OperationBuilder_v8 & {
        m_operation.scaledesc = scale_tensor.get_desc();
        m_operation.biasdesc  = bias_tensor.get_desc();
        return *this;
    }

    auto
    setDScale(Tensor_v8 const &scale_tensor) -> OperationBuilder_v8 & {
        m_operation.dscaledesc = scale_tensor.get_desc();
        return *this;
    }

    auto
    setDBias(Tensor_v8 const &bias_tensor) -> OperationBuilder_v8 & {
        m_operation.dbiasdesc = bias_tensor.get_desc();
        return *this;
    }

    auto
    setDScaleAndDBias(Tensor_v8 const &scale_tensor, Tensor_v8 const &bias_tensor) -> OperationBuilder_v8 & {
        m_operation.dscaledesc = scale_tensor.get_desc();
        m_operation.dbiasdesc  = bias_tensor.get_desc();
        return *this;
    }

    auto
    setEqScalesAndBias(Tensor_v8 const &eq_scale_tensor1,
                       Tensor_v8 const &eq_scale_tensor2,
                       Tensor_v8 const &eq_bias_tensor) -> OperationBuilder_v8 & {
        m_operation.eqscaledesc  = eq_scale_tensor1.get_desc();
        m_operation.eqscaledesc1 = eq_scale_tensor2.get_desc();
        m_operation.eqbiasdesc   = eq_bias_tensor.get_desc();
        return *this;
    }

    auto
    setEqScaleAndBias(Tensor_v8 const &eq_scale_tensor, Tensor_v8 const &eq_bias_tensor) -> OperationBuilder_v8 & {
        m_operation.eqscaledesc = eq_scale_tensor.get_desc();
        m_operation.eqbiasdesc  = eq_bias_tensor.get_desc();
        return *this;
    }

    auto
    setSumDesc(Tensor_v8 const &tensor) -> OperationBuilder_v8 & {
        m_operation.sumdesc = tensor.get_desc();
        return *this;
    }

    auto
    setSqSumDesc(Tensor_v8 const &tensor) -> OperationBuilder_v8 & {
        m_operation.sqsumdesc = tensor.get_desc();
        return *this;
    }

    auto
    setaMatDesc(ManagedOpaqueDescriptor const &raw_tensor) -> OperationBuilder_v8 & {
        m_operation.amatdesc = raw_tensor;
        return *this;
    }
    auto
    setaMatDesc(Tensor_v8 const &tensor) -> OperationBuilder_v8 & {
        if (is_matmul_op == false) {
            set_error_and_throw_exception(
                &m_operation,
                CUDNN_STATUS_BAD_PARAM,
                "CUDNN_BACKEND_OPERATION_*_DESCRIPTOR: Non Matmul operation does not need a Matrix Tensor");
        }
        m_operation.amatdesc = tensor.get_desc();
        return *this;
    }
    auto
    setbMatDesc(Tensor_v8 const &tensor) -> OperationBuilder_v8 & {
        if (is_matmul_op == false) {
            set_error_and_throw_exception(
                &m_operation,
                CUDNN_STATUS_BAD_PARAM,
                "CUDNN_BACKEND_OPERATION_*_DESCRIPTOR: Non Matmul operation does not need b Matrix Tensor");
        }
        m_operation.bmatdesc = tensor.get_desc();
        return *this;
    }
    auto
    setcMatDesc(Tensor_v8 const &tensor) -> OperationBuilder_v8 & {
        if (is_matmul_op == false) {
            set_error_and_throw_exception(
                &m_operation,
                CUDNN_STATUS_BAD_PARAM,
                "CUDNN_BACKEND_OPERATION_*_DESCRIPTOR: Non Matmul operation does not need c Matrix Tensor");
        }
        m_operation.cmatdesc = tensor.get_desc();
        return *this;
    }
    auto
    setmOverrideDesc(Tensor_v8 const &tensor) -> OperationBuilder_v8 & {
        if (is_matmul_op == false) {
            set_error_and_throw_exception(
                &m_operation,
                CUDNN_STATUS_BAD_PARAM,
                "CUDNN_BACKEND_OPERATION_*_DESCRIPTOR: Non Matmul operation does not need mOverride Tensor");
        }
        m_operation.moverridedesc = tensor.get_desc();
        return *this;
    }
    auto
    setnOverrideDesc(Tensor_v8 const &tensor) -> OperationBuilder_v8 & {
        if (is_matmul_op == false) {
            set_error_and_throw_exception(
                &m_operation,
                CUDNN_STATUS_BAD_PARAM,
                "CUDNN_BACKEND_OPERATION_*_DESCRIPTOR: Non Matmul operation does not need nOverride Tensor");
        }
        m_operation.noverridedesc = tensor.get_desc();
        return *this;
    }
    auto
    setkOverrideDesc(Tensor_v8 const &tensor) -> OperationBuilder_v8 & {
        if (is_matmul_op == false) {
            set_error_and_throw_exception(
                &m_operation,
                CUDNN_STATUS_BAD_PARAM,
                "CUDNN_BACKEND_OPERATION_*_DESCRIPTOR: Non Matmul operation does not need kOverride Tensor");
        }
        m_operation.koverridedesc = tensor.get_desc();
        return *this;
    }
    auto
    setmatmulDesc(MatMulDesc_v8 const &matmulDesc) -> OperationBuilder_v8 & {
        if (is_matmul_op == false) {
            set_error_and_throw_exception(
                &m_operation,
                CUDNN_STATUS_BAD_PARAM,
                "CUDNN_BACKEND_OPERATION_*_DESCRIPTOR: Non Matmul operation does not need MATMUL DESCRIPTOR");
        }
        m_operation.matmuldesc = matmulDesc.get_desc();
        return *this;
    }
    auto
    setreductionDesc(ReductionDesc_v8 const &reductionDesc) -> OperationBuilder_v8 & {
        if (is_reduction_op == false) {
            set_error_and_throw_exception(
                &m_operation,
                CUDNN_STATUS_BAD_PARAM,
                "CUDNN_BACKEND_OPERATION_*_DESCRIPTOR: Non Reduction operation does not need REDUCTION DESCRIPTOR");
        }
        m_operation.reductiondesc = reductionDesc.get_desc();
        return *this;
    }
    auto
    setpwDesc(PointWiseDesc_v8 const &pointWiseDesc) -> OperationBuilder_v8 & {
        if (is_pointwise_op == false) {
            set_error_and_throw_exception(
                &m_operation,
                CUDNN_STATUS_BAD_PARAM,
                "CUDNN_BACKEND_OPERATION_*_DESCRIPTOR: Non Pointwise operation does not need POINTWISE DESCRIPTOR");
        }
        m_operation.pwdesc               = pointWiseDesc.get_desc();
        m_operation.pointwise_port_count = pointWiseDesc.getPortCount();
        m_operation.pointwise_mode       = pointWiseDesc.getPointWiseMode();

        m_operation.is_pointwise_math_op = ((m_operation.pointwise_mode == PointwiseMode_t::ADD) ||
                                            (m_operation.pointwise_mode == PointwiseMode_t::MUL) ||
                                            (m_operation.pointwise_mode == PointwiseMode_t::DIV) ||
                                            (m_operation.pointwise_mode == PointwiseMode_t::SUB) ||
                                            (m_operation.pointwise_mode == PointwiseMode_t::ADD_SQUARE) ||
                                            (m_operation.pointwise_mode == PointwiseMode_t::RSQRT) ||
                                            (m_operation.pointwise_mode == PointwiseMode_t::SIN) ||
                                            (m_operation.pointwise_mode == PointwiseMode_t::COS) ||
                                            (m_operation.pointwise_mode == PointwiseMode_t::TAN) ||
                                            (m_operation.pointwise_mode == PointwiseMode_t::LOGICAL_OR) ||
                                            (m_operation.pointwise_mode == PointwiseMode_t::LOGICAL_AND) ||
                                            (m_operation.pointwise_mode == PointwiseMode_t::LOGICAL_NOT) ||
                                            (m_operation.pointwise_mode == PointwiseMode_t::CMP_EQ) ||
                                            (m_operation.pointwise_mode == PointwiseMode_t::CMP_NEQ) ||
                                            (m_operation.pointwise_mode == PointwiseMode_t::CMP_GT) ||
                                            (m_operation.pointwise_mode == PointwiseMode_t::CMP_GE) ||
                                            (m_operation.pointwise_mode == PointwiseMode_t::CMP_LT) ||
                                            (m_operation.pointwise_mode == PointwiseMode_t::CMP_LE) ||
                                            (m_operation.pointwise_mode == PointwiseMode_t::LOG) ||
                                            (m_operation.pointwise_mode == PointwiseMode_t::NEG) ||
                                            (m_operation.pointwise_mode == PointwiseMode_t::MOD) ||
                                            (m_operation.pointwise_mode == PointwiseMode_t::POW) ||
                                            (m_operation.pointwise_mode == PointwiseMode_t::ABS) ||
                                            (m_operation.pointwise_mode == PointwiseMode_t::CEIL) ||
                                            (m_operation.pointwise_mode == PointwiseMode_t::FLOOR) ||
                                            (m_operation.pointwise_mode == PointwiseMode_t::GEN_INDEX) ||
                                            (m_operation.pointwise_mode == PointwiseMode_t::BINARY_SELECT) ||
                                            (m_operation.pointwise_mode == PointwiseMode_t::ERF) ||
                                            (m_operation.pointwise_mode == PointwiseMode_t::RECIPROCAL) ||
                                            (m_operation.pointwise_mode == PointwiseMode_t::MIN) ||
                                            (m_operation.pointwise_mode == PointwiseMode_t::MAX) ||
                                            (m_operation.pointwise_mode == PointwiseMode_t::SQRT));

        m_operation.is_pointwise_identity_op = (m_operation.pointwise_mode == PointwiseMode_t::IDENTITY);

        m_operation.is_pointwise_activation_fwd_op =
            ((m_operation.pointwise_mode == PointwiseMode_t::RELU_FWD) ||
             (m_operation.pointwise_mode == PointwiseMode_t::TANH_FWD) ||
             (m_operation.pointwise_mode == PointwiseMode_t::SIGMOID_FWD) ||
             (m_operation.pointwise_mode == PointwiseMode_t::ELU_FWD) ||
             (m_operation.pointwise_mode == PointwiseMode_t::GELU_FWD) ||
             (m_operation.pointwise_mode == PointwiseMode_t::GELU_APPROX_TANH_FWD) ||
             (m_operation.pointwise_mode == PointwiseMode_t::SOFTPLUS_FWD) ||
             (m_operation.pointwise_mode == PointwiseMode_t::EXP) ||
             (m_operation.pointwise_mode == PointwiseMode_t::SWISH_FWD));

        m_operation.is_pointwise_activation_bwd_op =
            ((m_operation.pointwise_mode == PointwiseMode_t::RELU_BWD) ||
             (m_operation.pointwise_mode == PointwiseMode_t::TANH_BWD) ||
             (m_operation.pointwise_mode == PointwiseMode_t::SIGMOID_BWD) ||
             (m_operation.pointwise_mode == PointwiseMode_t::ELU_BWD) ||
             (m_operation.pointwise_mode == PointwiseMode_t::GELU_BWD) ||
             (m_operation.pointwise_mode == PointwiseMode_t::GELU_APPROX_TANH_BWD) ||
             (m_operation.pointwise_mode == PointwiseMode_t::SOFTPLUS_BWD) ||
             (m_operation.pointwise_mode == PointwiseMode_t::SWISH_BWD));

        return *this;
    }

    auto
    setAlpha(float alpha) -> OperationBuilder_v8 & {
        m_operation.alpha_d = static_cast<double>(alpha);
        m_operation.alpha_s = alpha;
        return *this;
    }
    auto
    setAlpha(double alpha) -> OperationBuilder_v8 & {
        m_operation.alpha_s = static_cast<float>(alpha);
        m_operation.alpha_d = alpha;
        return *this;
    }
    auto
    setAlpha2(float alpha) -> OperationBuilder_v8 & {
        m_operation.alpha2_d = static_cast<double>(alpha);
        m_operation.alpha2_s = alpha;
        return *this;
    }
    auto
    setAlpha2(double alpha) -> OperationBuilder_v8 & {
        m_operation.alpha2_s = static_cast<float>(alpha);
        m_operation.alpha2_d = alpha;
        return *this;
    }
    auto
    setBeta(float beta) -> OperationBuilder_v8 & {
        m_operation.beta_d = static_cast<double>(beta);
        m_operation.beta_s = beta;
        return *this;
    }
    auto
    setBeta(double beta) -> OperationBuilder_v8 & {
        m_operation.beta_s = static_cast<float>(beta);
        m_operation.beta_d = beta;
        return *this;
    }

    auto
    setSeed(int64_t seed) -> OperationBuilder_v8 & {
        m_operation.seed = seed;
        return *this;
    }

    auto
    setComputeType(cudnnDataType_t dtype) -> OperationBuilder_v8 & {
        m_operation.compute_type = dtype;
        return *this;
    }

    auto
    setMathPrecision(cudnnDataType_t dtype) -> OperationBuilder_v8 & {
        return setComputeType(dtype);
    }

    auto
    setGenStatsMode(cudnnGenStatsMode_t type) -> OperationBuilder_v8 & {
        m_operation.genstats_mode = type;
        return *this;
    }

    OperationBuilder_v8(DescriptorType_t mode) {
        m_operation.op_mode = mode;
        is_convolution_op =
            ((m_operation.op_mode == DescriptorType_t::OPERATION_CONVOLUTION_FORWARD_DESCRIPTOR) ||
             (m_operation.op_mode == DescriptorType_t::OPERATION_CONVOLUTION_BACKWARD_FILTER_DESCRIPTOR) ||
             (m_operation.op_mode == DescriptorType_t::OPERATION_CONVOLUTION_BACKWARD_DATA_DESCRIPTOR));

        is_pointwise_op        = (m_operation.op_mode == DescriptorType_t::OPERATION_POINTWISE_DESCRIPTOR);
        is_matmul_op           = (m_operation.op_mode == DescriptorType_t::OPERATION_MATMUL_DESCRIPTOR);
        is_reduction_op        = (m_operation.op_mode == DescriptorType_t::OPERATION_REDUCTION_DESCRIPTOR);
        is_genstats_op         = (m_operation.op_mode == DescriptorType_t::OPERATION_GEN_STATS_DESCRIPTOR);
        is_bn_finalize_op      = (m_operation.op_mode == DescriptorType_t::OPERATION_BN_FINALIZE_STATISTICS_DESCRIPTOR);
        is_bn_bwd_weight       = (m_operation.op_mode == DescriptorType_t::OPERATION_BN_BWD_WEIGHTS_DESCRIPTOR);
        is_resample_fwd_op     = (m_operation.op_mode == DescriptorType_t::OPERATION_RESAMPLE_FWD_DESCRIPTOR);
        is_norm_forward_op     = (m_operation.op_mode == DescriptorType_t::OPERATION_NORM_FORWARD_DESCRIPTOR);
        is_norm_backward_op    = (m_operation.op_mode == DescriptorType_t::OPERATION_NORM_BACKWARD_DESCRIPTOR);
        is_resample_bwd_op     = (m_operation.op_mode == DescriptorType_t::OPERATION_RESAMPLE_BWD_DESCRIPTOR);
        is_rng_op              = (m_operation.op_mode == DescriptorType_t::OPERATION_RNG_DESCRIPTOR);
        is_reshape_op          = (m_operation.op_mode == DescriptorType_t::OPERATION_RESHAPE_DESCRIPTOR);
        is_paged_cache_load_op = (m_operation.op_mode == DescriptorType_t::OPERATION_PAGED_CACHE_LOAD_DESCRIPTOR);
    }

    // This constructor which takes in cudnn C backend enum for cudnnBackendDescriptorType_t will be deprecated,
    // in favour of OperationBuilder_v8(cudnn_frontend::DescriptorType_t)
    OperationBuilder_v8(cudnnBackendDescriptorType_t mode)
        : OperationBuilder_v8(detail::convert_from_cudnn_type(mode)) {}

    /** @} */

    //! constructs the backend Operation_v8 by calling the cudnn API
    //! Throws the appropriate error message
    Operation_v8 &&
    build() {
        if (m_operation.status != CUDNN_STATUS_SUCCESS) {
            set_error_and_throw_exception(
                &m_operation, m_operation.status, "CUDNN_BACKEND_OPERATION: Operation not initialized properly");
            return std::move(m_operation);
        }

        Message_t msg         = nullptr;
        cudnnStatus_t status_ = CUDNN_STATUS_SUCCESS;
        if (is_convolution_op) {
            status_ = validate_convolution_op(msg);
        } else if (is_pointwise_op) {
            status_ = validate_pointwise_op(msg);
        } else if (is_matmul_op) {
            status_ = validate_matmul_op(msg);
        } else if (is_reduction_op) {
            status_ = validate_reduction_op(msg);
        } else if (is_genstats_op) {
            status_ = CUDNN_STATUS_SUCCESS;
        } else if (is_bn_finalize_op) {
            status_ = CUDNN_STATUS_SUCCESS;
        } else if (is_bn_bwd_weight) {
            status_ = validate_bn_bwd_weight_op(msg);
        } else if (is_resample_fwd_op) {
            status_ = validate_resample_op(msg);
        } else if (is_resample_bwd_op) {
            status_ = validate_resample_op(msg);
        } else if (is_rng_op) {
            status_ = validate_rng_op(msg);
        } else if (is_norm_forward_op || is_norm_backward_op) {
            status_ = validate_norm_op(msg);
        } else if (is_reshape_op) {
            status_ = validate_reshape_op(msg);
        } else if (is_paged_cache_load_op) {
            status_ = CUDNN_STATUS_SUCCESS;
        } else {
            status_ = CUDNN_STATUS_BAD_PARAM;
            msg =
                "CUDNN_BACKEND_OPERATION_DESCRIPTOR: Unsupported cudnn backend descriptor type. Check and set "
                "CUDNN_BACKEND_OPERATION_*_DESCRIPTOR";
        }
        if (status_ != CUDNN_STATUS_SUCCESS) {
            set_error_and_throw_exception(&m_operation, status_, msg);
            return std::move(m_operation);
        }

        // Create the descriptor.
        cudnnBackendDescriptorType_t cudnn_backend_descriptor_type;
        auto status = detail::convert_to_cudnn_type(m_operation.op_mode, cudnn_backend_descriptor_type);
        if (status != CUDNN_STATUS_SUCCESS) {
            std::stringstream ss;
            ss << "CUDNN_BACKEND_OPERATION: unable to identify backend operation for " << m_operation.op_mode;
            set_error_and_throw_exception(&m_operation, status, (ss.str()).c_str());
            return std::move(m_operation);
        }
        status = m_operation.initialize_managed_backend_pointer(cudnn_backend_descriptor_type);
        if (status != CUDNN_STATUS_SUCCESS) {
            set_error_and_throw_exception(&m_operation, status, "CUDNN_BACKEND_OPERATION: cudnnCreate Failed");
            return std::move(m_operation);
        }

        if (m_operation.op_mode == DescriptorType_t::OPERATION_CONVOLUTION_FORWARD_DESCRIPTOR) {
            return build_conv_forward();
        } else if (m_operation.op_mode == DescriptorType_t::OPERATION_CONVOLUTION_BACKWARD_FILTER_DESCRIPTOR) {
            return build_conv_backward_filter();
        } else if (m_operation.op_mode == DescriptorType_t::OPERATION_CONVOLUTION_BACKWARD_DATA_DESCRIPTOR) {
            return build_conv_backward_data();
        } else if (m_operation.op_mode == DescriptorType_t::OPERATION_POINTWISE_DESCRIPTOR) {
            return build_pointwise_op();
        } else if (m_operation.op_mode == DescriptorType_t::OPERATION_MATMUL_DESCRIPTOR) {
            return build_matmul_op();
        } else if (m_operation.op_mode == DescriptorType_t::OPERATION_REDUCTION_DESCRIPTOR) {
            return build_reduction_op();
        } else if (m_operation.op_mode == DescriptorType_t::OPERATION_GEN_STATS_DESCRIPTOR) {
            return build_genstats_op();
        } else if (m_operation.op_mode == DescriptorType_t::OPERATION_BN_FINALIZE_STATISTICS_DESCRIPTOR) {
            return build_bn_finalize_op();
        } else if (m_operation.op_mode == DescriptorType_t::OPERATION_BN_BWD_WEIGHTS_DESCRIPTOR) {
            return build_bn_bwd_weight_op();
        } else if (m_operation.op_mode == DescriptorType_t::OPERATION_RESAMPLE_FWD_DESCRIPTOR) {
            return build_resample_fwd_operation();
        } else if (m_operation.op_mode == DescriptorType_t::OPERATION_NORM_FORWARD_DESCRIPTOR) {
            return build_norm_forward();
        } else if (m_operation.op_mode == DescriptorType_t::OPERATION_NORM_BACKWARD_DESCRIPTOR) {
            return build_norm_backward();
        } else if (m_operation.op_mode == DescriptorType_t::OPERATION_RESAMPLE_BWD_DESCRIPTOR) {
            return build_resample_bwd_operation();
        } else if (m_operation.op_mode == DescriptorType_t::OPERATION_RNG_DESCRIPTOR) {
            return build_rng_operation();
        } else if (m_operation.op_mode == DescriptorType_t::OPERATION_PAGED_CACHE_LOAD_DESCRIPTOR) {
            return build_paged_cache_load_op();
        } else if (m_operation.op_mode == DescriptorType_t::OPERATION_RESHAPE_DESCRIPTOR) {
            return build_reshape_operation();
        } else {
            set_error_and_throw_exception(
                &m_operation, status, "CUDNN_BACKEND_OPERATION: unimplemented operation in frontend");
        }
        CUDNN_FE_LOG_LABEL_ENDL(m_operation);
        return std::move(m_operation);
    }
};

using Operation        = Operation_v8;
using OperationBuilder = OperationBuilder_v8;
}  // namespace cudnn_frontend
