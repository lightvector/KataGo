#pragma once

#include "../graph_helpers.h"
#include "../node_interface.h"

namespace cudnn_frontend {

namespace graph {
class RMSNormNode : public NodeCRTP<RMSNormNode> {
   public:
    Rmsnorm_attributes attributes;

    RMSNormNode(Rmsnorm_attributes&& attributes_, detail::Context const& context)
        : NodeCRTP(context), attributes(std::move(attributes_)) {}

    Type
    getType() override final {
        return Type::RMSNORM;
    }

    error_t
    infer_properties_node() override final {
        CUDNN_FE_LOG_LABEL_ENDL("INFO: Inferencing properties for rmsnorm node " << attributes.name);

        attributes.fill_from_context(context);

        auto X = attributes.inputs[Rmsnorm_attributes::input_names::X];
        auto Y = attributes.outputs[Rmsnorm_attributes::output_names::Y];

        // Only infer dims and strides if user did not set them
        if (Y->get_dim().empty()) {
            Y->set_dim(X->get_dim());
        }
        if (Y->get_stride().empty()) {
            Y->set_stride(X->get_stride());
        }

        if (attributes.forward_phase == NormFwdPhase_t::TRAINING) {
            auto inv_var = attributes.outputs[Rmsnorm_attributes::output_names::INV_VARIANCE];
            // Only infer dims and strides if user did not set them
            if (inv_var->get_dim().empty()) {
                auto inv_var_dim = X->get_dim();
                auto scale       = attributes.inputs[Rmsnorm_attributes::input_names::SCALE];
                if (scale->get_dim().empty()) {
                    // mean inv_var dim is n,1,1,1
                    for (size_t i = 1; i < inv_var_dim.size(); i++) {
                        inv_var_dim[i] = 1;
                    }
                } else {
                    for (size_t i = 0; i < inv_var_dim.size(); i++) {
                        if (scale->get_dim()[i] != 1) {
                            inv_var_dim[i] = 1;
                        }
                    }
                }
                inv_var->set_dim(inv_var_dim);
            }
            if (inv_var->get_stride().empty()) {
                auto const& inv_var_dim = inv_var->get_dim();
                std::vector<int64_t> stride_order;
                CHECK_CUDNN_FRONTEND_ERROR(
                    detail::generate_stride_order_preserving_format(X->get_stride(), inv_var_dim.size(), stride_order));
                inv_var->set_stride(detail::generate_stride(inv_var_dim, stride_order));
            }
        }
        return {error_code_t::OK, ""};
    }

    error_t
    pre_validate_node() const override final {
        CUDNN_FE_LOG_LABEL_ENDL("INFO: Validating RMSNormNode " << attributes.name);

        // Norm forward phase should be set
        RETURN_CUDNN_FRONTEND_ERROR_IF(attributes.forward_phase == NormFwdPhase_t::NOT_SET,
                                       error_code_t::ATTRIBUTE_NOT_SET,
                                       "Forward phase not set of rmsnorm node.");

        return {error_code_t::OK, ""};
    }

    error_t
    create_cudnn_operations(
        std::unordered_set<Tensor_attributes::uid_t>& uids_involved_in_operations,
        std::vector<std::shared_ptr<cudnn_frontend::Operation>>& operations,
        managed_backend_descriptor_t& raw_operations,
        std::unordered_map<int64_t, std::shared_ptr<cudnn_frontend::Tensor>>& tensors) const override final {
        CUDNN_FRONTEND_UNUSED(raw_operations);
        CUDNN_FE_LOG_LABEL("INFO: Building RMSNormNode operations " << attributes.name << " ");

        // Create operation by directly calling cuDNN backend API
        Operation_v8 rmsnorm_operation;

        _CUDNN_CHECK_CUDNN_ERROR(
            rmsnorm_operation.initialize_managed_backend_pointer(CUDNN_BACKEND_OPERATION_NORM_FORWARD_DESCRIPTOR));

        // Set norm mode to RMS_NORM
        cudnnBackendNormMode_t cudnn_norm_mode;

        _CUDNN_CHECK_CUDNN_ERROR(detail::convert_to_cudnn_type(NormMode_t::RMS_NORM, cudnn_norm_mode));

        _CUDNN_CHECK_CUDNN_ERROR(detail::set_attribute(rmsnorm_operation.get_raw_desc(),
                                                       CUDNN_ATTR_OPERATION_NORM_FWD_MODE,
                                                       CUDNN_TYPE_NORM_MODE,
                                                       1,
                                                       &cudnn_norm_mode));

        // Set forward phase
        cudnnBackendNormFwdPhase_t cudnn_norm_fwd_phase;

        _CUDNN_CHECK_CUDNN_ERROR(detail::convert_to_cudnn_type(attributes.forward_phase, cudnn_norm_fwd_phase));

        _CUDNN_CHECK_CUDNN_ERROR(detail::set_attribute(rmsnorm_operation.get_raw_desc(),
                                                       CUDNN_ATTR_OPERATION_NORM_FWD_PHASE,
                                                       CUDNN_TYPE_NORM_FWD_PHASE,
                                                       1,
                                                       &cudnn_norm_fwd_phase));

        // Set input tensor X
        CUDNN_FE_VALIDATE_AND_ASSIGN_INPUT_TENSOR(X, Rmsnorm_attributes::input_names::X);
        auto x_desc = tensors.at(X->second->get_uid())->get_raw_desc();

        _CUDNN_CHECK_CUDNN_ERROR(detail::set_attribute(rmsnorm_operation.get_raw_desc(),
                                                       CUDNN_ATTR_OPERATION_NORM_FWD_XDESC,
                                                       CUDNN_TYPE_BACKEND_DESCRIPTOR,
                                                       1,
                                                       &x_desc));

        // Set scale tensor
        CUDNN_FE_VALIDATE_AND_ASSIGN_INPUT_TENSOR(SCALE, Rmsnorm_attributes::input_names::SCALE);
        auto scale_desc = tensors.at(SCALE->second->get_uid())->get_raw_desc();

        _CUDNN_CHECK_CUDNN_ERROR(detail::set_attribute(rmsnorm_operation.get_raw_desc(),
                                                       CUDNN_ATTR_OPERATION_NORM_FWD_SCALE_DESC,
                                                       CUDNN_TYPE_BACKEND_DESCRIPTOR,
                                                       1,
                                                       &scale_desc));

        // Set epsilon tensor
        CUDNN_FE_VALIDATE_AND_ASSIGN_INPUT_TENSOR(EPSILON, Rmsnorm_attributes::input_names::EPSILON);
        auto epsilon_desc = tensors.at(EPSILON->second->get_uid())->get_raw_desc();

        _CUDNN_CHECK_CUDNN_ERROR(detail::set_attribute(rmsnorm_operation.get_raw_desc(),
                                                       CUDNN_ATTR_OPERATION_NORM_FWD_EPSILON_DESC,
                                                       CUDNN_TYPE_BACKEND_DESCRIPTOR,
                                                       1,
                                                       &epsilon_desc));

        // Set output tensor Y
        CUDNN_FE_VALIDATE_AND_ASSIGN_OUTPUT_TENSOR(Y, Rmsnorm_attributes::output_names::Y);
        auto y_desc = tensors.at(Y->second->get_uid())->get_raw_desc();

        _CUDNN_CHECK_CUDNN_ERROR(detail::set_attribute(rmsnorm_operation.get_raw_desc(),
                                                       CUDNN_ATTR_OPERATION_NORM_FWD_YDESC,
                                                       CUDNN_TYPE_BACKEND_DESCRIPTOR,
                                                       1,
                                                       &y_desc));

        // Set inv_variance for training phase
        if (attributes.forward_phase == NormFwdPhase_t::TRAINING) {
            CUDNN_FE_VALIDATE_AND_ASSIGN_OUTPUT_TENSOR(INV_VARIANCE, Rmsnorm_attributes::output_names::INV_VARIANCE);
            auto inv_var_desc = tensors.at(INV_VARIANCE->second->get_uid())->get_raw_desc();

            _CUDNN_CHECK_CUDNN_ERROR(detail::set_attribute(rmsnorm_operation.get_raw_desc(),
                                                           CUDNN_ATTR_OPERATION_NORM_FWD_INV_VARIANCE_DESC,
                                                           CUDNN_TYPE_BACKEND_DESCRIPTOR,
                                                           1,
                                                           &inv_var_desc));
        }

        // Set optional bias tensor
        auto BIAS = attributes.inputs.find(Rmsnorm_attributes::input_names::BIAS);
        if ((BIAS != attributes.inputs.end()) && (BIAS->second != nullptr)) {
            auto bias_desc = tensors.at(BIAS->second->get_uid())->get_raw_desc();

            _CUDNN_CHECK_CUDNN_ERROR(detail::set_attribute(rmsnorm_operation.get_raw_desc(),
                                                           CUDNN_ATTR_OPERATION_NORM_FWD_BIAS_DESC,
                                                           CUDNN_TYPE_BACKEND_DESCRIPTOR,
                                                           1,
                                                           &bias_desc));
        }

        _CUDNN_CHECK_CUDNN_ERROR(detail::finalize(rmsnorm_operation.get_raw_desc()));

        operations.push_back(std::make_shared<Operation_v8>(std::move(rmsnorm_operation)));

        auto const& non_virtual_uids = attributes.get_non_virtual_uids();
        uids_involved_in_operations.insert(non_virtual_uids.begin(), non_virtual_uids.end());
        return {error_code_t::OK, ""};
    }

#ifndef CUDNN_FRONTEND_SKIP_JSON_LIB
    virtual void
    serialize(json& j) const override final {
        j = attributes;
        j.update(R"( {"tag": "RMS_NORM"})"_json);
    }
#endif
};

class DRMSNormNode : public NodeCRTP<DRMSNormNode> {
   public:
    Rmsnorm_backward_attributes attributes;

    DRMSNormNode(Rmsnorm_backward_attributes&& attributes_, detail::Context const& context)
        : NodeCRTP(context), attributes(std::move(attributes_)) {}

    Type
    getType() override final {
        return Type::DRMSNorm;
    }

    error_t
    pre_validate_node() const override final {
        CUDNN_FE_LOG_LABEL_ENDL("INFO: Validating DRMSNormNode node " << attributes.name);

        RETURN_CUDNN_FRONTEND_ERROR_IF(attributes.use_dbias.has_value() == false,
                                       error_code_t::ATTRIBUTE_NOT_SET,
                                       "DRMSNormNode node needs has_bias(bool) to be called.");

        return {error_code_t::OK, ""};
    }

    error_t
    infer_properties_node() override final {
        CUDNN_FE_LOG_LABEL_ENDL("INFO: Inferencing properties for DRMSNorm node " << attributes.name);

        attributes.fill_from_context(context);

        // TODO: Only inferencing from X works today.
        auto X                  = attributes.inputs[Rmsnorm_backward_attributes::input_names::X];
        auto const x_tensor_dim = X->get_dim();

        auto DY            = attributes.inputs[Rmsnorm_backward_attributes::input_names::DY];
        auto dy_tensor_dim = DY->get_dim();

        // Only infer dims and strides if user did not set them
        if (dy_tensor_dim.empty()) {
            dy_tensor_dim.resize(x_tensor_dim.size());
            DY->set_dim(x_tensor_dim);
        }
        if (DY->get_stride().empty()) {
            auto const& DY_dim = DY->get_dim();
            // Default to NHWC
            auto const& stride_order = detail::generate_NHWC_stride_order(DY_dim.size());
            DY->set_stride(detail::generate_stride(DY_dim, stride_order));
        }

        auto DX            = attributes.outputs[Rmsnorm_backward_attributes::output_names::DX];
        auto dx_tensor_dim = DX->get_dim();
        // Only infer dims and strides if user did not set them
        if (dx_tensor_dim.empty()) {
            dx_tensor_dim.resize(x_tensor_dim.size());
            DX->set_dim(x_tensor_dim);
        }
        if (DX->get_stride().empty()) {
            auto const& DX_dim = DX->get_dim();
            // Default to NHWC
            auto const& stride_order = detail::generate_NHWC_stride_order(DX_dim.size());
            DX->set_stride(detail::generate_stride(DX_dim, stride_order));
        }

        auto scale_bias_dim = X->get_dim();
        scale_bias_dim[0]   = 1;

        // Set channel length tensors
        auto infer_scale_bias_tensors = [&scale_bias_dim](std::shared_ptr<Tensor_attributes>& T) {
            auto tensor_dim = T->get_dim();
            // Only infer dims and strides if user did not set them
            if (tensor_dim.empty()) {
                T->set_dim(scale_bias_dim);
            }
            if (T->get_stride().empty()) {
                auto const& T_dim = T->get_dim();
                // Default to NHWC
                auto const& stride_order = detail::generate_NHWC_stride_order(T_dim.size());
                T->set_stride(detail::generate_stride(T_dim, stride_order));
            }
        };

        infer_scale_bias_tensors(attributes.outputs[Rmsnorm_backward_attributes::output_names::DSCALE]);
        if (attributes.use_dbias.value()) {
            infer_scale_bias_tensors(attributes.outputs[Rmsnorm_backward_attributes::output_names::DBIAS]);
        }

        return {error_code_t::OK, ""};
    }

    error_t
    create_cudnn_operations(
        std::unordered_set<Tensor_attributes::uid_t>& uids_involved_in_operations,
        std::vector<std::shared_ptr<cudnn_frontend::Operation>>& operations,
        managed_backend_descriptor_t& raw_operations,
        std::unordered_map<int64_t, std::shared_ptr<cudnn_frontend::Tensor>>& tensors) const override final {
        CUDNN_FRONTEND_UNUSED(raw_operations);
        CUDNN_FE_LOG_LABEL("INFO: Building DRMSNormNode operations " << attributes.name << " ");

        // Create operation by directly calling cuDNN backend API
        Operation_v8 drmsnorm_operation;

        _CUDNN_CHECK_CUDNN_ERROR(
            drmsnorm_operation.initialize_managed_backend_pointer(CUDNN_BACKEND_OPERATION_NORM_BACKWARD_DESCRIPTOR));

        // Set norm mode to RMS_NORM
        cudnnBackendNormMode_t cudnn_norm_mode;

        _CUDNN_CHECK_CUDNN_ERROR(detail::convert_to_cudnn_type(NormMode_t::RMS_NORM, cudnn_norm_mode));

        _CUDNN_CHECK_CUDNN_ERROR(detail::set_attribute(drmsnorm_operation.get_raw_desc(),
                                                       CUDNN_ATTR_OPERATION_NORM_BWD_MODE,
                                                       CUDNN_TYPE_NORM_MODE,
                                                       1,
                                                       &cudnn_norm_mode));

        // Set input tensor X
        CUDNN_FE_VALIDATE_AND_ASSIGN_INPUT_TENSOR(X, Rmsnorm_backward_attributes::input_names::X);
        auto x_desc = tensors.at(X->second->get_uid())->get_raw_desc();

        _CUDNN_CHECK_CUDNN_ERROR(detail::set_attribute(drmsnorm_operation.get_raw_desc(),
                                                       CUDNN_ATTR_OPERATION_NORM_BWD_XDESC,
                                                       CUDNN_TYPE_BACKEND_DESCRIPTOR,
                                                       1,
                                                       &x_desc));

        // Set DY tensor
        CUDNN_FE_VALIDATE_AND_ASSIGN_INPUT_TENSOR(DY, Rmsnorm_backward_attributes::input_names::DY);
        auto dy_desc = tensors.at(DY->second->get_uid())->get_raw_desc();

        _CUDNN_CHECK_CUDNN_ERROR(detail::set_attribute(drmsnorm_operation.get_raw_desc(),
                                                       CUDNN_ATTR_OPERATION_NORM_BWD_DYDESC,
                                                       CUDNN_TYPE_BACKEND_DESCRIPTOR,
                                                       1,
                                                       &dy_desc));

        // Set scale tensor
        CUDNN_FE_VALIDATE_AND_ASSIGN_INPUT_TENSOR(SCALE, Rmsnorm_backward_attributes::input_names::SCALE);
        auto scale_desc = tensors.at(SCALE->second->get_uid())->get_raw_desc();

        _CUDNN_CHECK_CUDNN_ERROR(detail::set_attribute(drmsnorm_operation.get_raw_desc(),
                                                       CUDNN_ATTR_OPERATION_NORM_BWD_SCALE_DESC,
                                                       CUDNN_TYPE_BACKEND_DESCRIPTOR,
                                                       1,
                                                       &scale_desc));

        // Set inv_variance tensor
        CUDNN_FE_VALIDATE_AND_ASSIGN_INPUT_TENSOR(INV_VARIANCE, Rmsnorm_backward_attributes::input_names::INV_VARIANCE);
        auto inv_var_desc = tensors.at(INV_VARIANCE->second->get_uid())->get_raw_desc();

        _CUDNN_CHECK_CUDNN_ERROR(detail::set_attribute(drmsnorm_operation.get_raw_desc(),
                                                       CUDNN_ATTR_OPERATION_NORM_BWD_INV_VARIANCE_DESC,
                                                       CUDNN_TYPE_BACKEND_DESCRIPTOR,
                                                       1,
                                                       &inv_var_desc));

        // Set DSCALE output tensor
        CUDNN_FE_VALIDATE_AND_ASSIGN_OUTPUT_TENSOR(DSCALE, Rmsnorm_backward_attributes::output_names::DSCALE);
        auto dscale_desc = tensors.at(DSCALE->second->get_uid())->get_raw_desc();

        _CUDNN_CHECK_CUDNN_ERROR(detail::set_attribute(drmsnorm_operation.get_raw_desc(),
                                                       CUDNN_ATTR_OPERATION_NORM_BWD_DSCALE_DESC,
                                                       CUDNN_TYPE_BACKEND_DESCRIPTOR,
                                                       1,
                                                       &dscale_desc));

        // Set optional DBIAS output tensor
        if (attributes.use_dbias.value()) {
            CUDNN_FE_VALIDATE_AND_ASSIGN_OUTPUT_TENSOR(DBIAS, Rmsnorm_backward_attributes::output_names::DBIAS);
            auto dbias_desc = tensors.at(DBIAS->second->get_uid())->get_raw_desc();

            _CUDNN_CHECK_CUDNN_ERROR(detail::set_attribute(drmsnorm_operation.get_raw_desc(),
                                                           CUDNN_ATTR_OPERATION_NORM_BWD_DBIAS_DESC,
                                                           CUDNN_TYPE_BACKEND_DESCRIPTOR,
                                                           1,
                                                           &dbias_desc));
        }

        // Set DX output tensor
        CUDNN_FE_VALIDATE_AND_ASSIGN_OUTPUT_TENSOR(DX, Rmsnorm_backward_attributes::output_names::DX);
        auto dx_desc = tensors.at(DX->second->get_uid())->get_raw_desc();

        _CUDNN_CHECK_CUDNN_ERROR(detail::set_attribute(drmsnorm_operation.get_raw_desc(),
                                                       CUDNN_ATTR_OPERATION_NORM_BWD_DXDESC,
                                                       CUDNN_TYPE_BACKEND_DESCRIPTOR,
                                                       1,
                                                       &dx_desc));

        _CUDNN_CHECK_CUDNN_ERROR(detail::finalize(drmsnorm_operation.get_raw_desc()));

        operations.push_back(std::make_shared<Operation_v8>(std::move(drmsnorm_operation)));

        auto const& non_virtual_uids = attributes.get_non_virtual_uids();
        uids_involved_in_operations.insert(non_virtual_uids.begin(), non_virtual_uids.end());
        return {error_code_t::OK, ""};
    }

#ifndef CUDNN_FRONTEND_SKIP_JSON_LIB
    virtual void
    serialize(json& j) const override final {
        j = attributes;
        j.update(R"( {"tag": "RMS_NORM_BPROP"})"_json);
    }
#endif
};

}  // namespace graph

}  // namespace cudnn_frontend