#pragma once

#include "../graph_helpers.h"
#include "../node_interface.h"

namespace cudnn_frontend {

namespace graph {
class InstanceNormNode : public NodeCRTP<InstanceNormNode> {
   public:
    Instancenorm_attributes attributes;

    InstanceNormNode(Instancenorm_attributes&& attributes_, detail::Context const& context)
        : NodeCRTP(context), attributes(std::move(attributes_)) {}

    Type
    getType() override final {
        return Type::INSTANCENORM;
    }

    error_t
    infer_properties_node() override final {
        CUDNN_FE_LOG_LABEL_ENDL("INFO: Inferencing properties for instancenorm node " << attributes.name);

        attributes.fill_from_context(context);

        auto X = attributes.inputs[Instancenorm_attributes::input_names::X];
        auto Y = attributes.outputs[Instancenorm_attributes::output_names::Y];

        // Only infer dims and strides if user did not set them
        if (Y->get_dim().empty()) {
            Y->set_dim(X->get_dim());
        }
        if (Y->get_stride().empty()) {
            Y->set_stride(X->get_stride());
        }

        // mean inv_var dim is n,c,1,1
        auto stats_dim = X->get_dim();
        for (size_t i = 2; i < stats_dim.size(); i++) {
            stats_dim[i] = 1;
        }

        if (attributes.forward_phase == NormFwdPhase_t::TRAINING) {
            auto mean = attributes.outputs[Instancenorm_attributes::output_names::MEAN];
            if (mean->get_dim().empty()) {
                mean->set_dim(stats_dim);
            }
            if (mean->get_stride().empty()) {
                auto const& mean_dim = mean->get_dim();
                // Default to NHWC
                auto const& stride_order = detail::generate_NHWC_stride_order(mean_dim.size());
                mean->set_stride(detail::generate_stride(mean_dim, stride_order));
            }

            auto inv_var = attributes.outputs[Instancenorm_attributes::output_names::INV_VARIANCE];
            if (inv_var->get_dim().empty()) {
                inv_var->set_dim(stats_dim);
            }
            if (inv_var->get_stride().empty()) {
                auto const& inv_var_dim = inv_var->get_dim();
                // Default to NHWC
                auto const& stride_order = detail::generate_NHWC_stride_order(inv_var_dim.size());
                inv_var->set_stride(detail::generate_stride(inv_var_dim, stride_order));
            }
        }
        return {error_code_t::OK, ""};
    }

    error_t
    pre_validate_node() const override final {
        CUDNN_FE_LOG_LABEL_ENDL("INFO: Validating InstanceNormNode " << attributes.name);

        // Norm forward phase should be set
        RETURN_CUDNN_FRONTEND_ERROR_IF(attributes.forward_phase == NormFwdPhase_t::NOT_SET,
                                       error_code_t::ATTRIBUTE_NOT_SET,
                                       "Forward phase not set of instancenorm node.");

        return {error_code_t::OK, ""};
    }

    error_t
    create_cudnn_operations(
        std::unordered_set<Tensor_attributes::uid_t>& uids_involved_in_operations,
        std::vector<std::shared_ptr<cudnn_frontend::Operation>>& operations,
        managed_backend_descriptor_t& raw_operations,
        std::unordered_map<int64_t, std::shared_ptr<cudnn_frontend::Tensor>>& tensors) const override final {
        CUDNN_FRONTEND_UNUSED(raw_operations);
        CUDNN_FE_LOG_LABEL("INFO: Building InstanceNormNode operations " << attributes.name << " ");

        // Create operation by directly calling cuDNN backend API
        Operation_v8 instancenorm_operation;

        _CUDNN_CHECK_CUDNN_ERROR(
            instancenorm_operation.initialize_managed_backend_pointer(CUDNN_BACKEND_OPERATION_NORM_FORWARD_DESCRIPTOR));

        // Set norm mode to INSTANCE_NORM
        cudnnBackendNormMode_t cudnn_norm_mode;

        _CUDNN_CHECK_CUDNN_ERROR(detail::convert_to_cudnn_type(NormMode_t::INSTANCE_NORM, cudnn_norm_mode));

        _CUDNN_CHECK_CUDNN_ERROR(detail::set_attribute(instancenorm_operation.get_raw_desc(),
                                                       CUDNN_ATTR_OPERATION_NORM_FWD_MODE,
                                                       CUDNN_TYPE_NORM_MODE,
                                                       1,
                                                       &cudnn_norm_mode));

        // Set forward phase
        cudnnBackendNormFwdPhase_t cudnn_norm_fwd_phase;

        _CUDNN_CHECK_CUDNN_ERROR(detail::convert_to_cudnn_type(attributes.forward_phase, cudnn_norm_fwd_phase));

        _CUDNN_CHECK_CUDNN_ERROR(detail::set_attribute(instancenorm_operation.get_raw_desc(),
                                                       CUDNN_ATTR_OPERATION_NORM_FWD_PHASE,
                                                       CUDNN_TYPE_NORM_FWD_PHASE,
                                                       1,
                                                       &cudnn_norm_fwd_phase));

        // Set input tensor X
        CUDNN_FE_VALIDATE_AND_ASSIGN_INPUT_TENSOR(X, Instancenorm_attributes::input_names::X);
        auto x_desc = tensors.at(X->second->get_uid())->get_raw_desc();

        _CUDNN_CHECK_CUDNN_ERROR(detail::set_attribute(instancenorm_operation.get_raw_desc(),
                                                       CUDNN_ATTR_OPERATION_NORM_FWD_XDESC,
                                                       CUDNN_TYPE_BACKEND_DESCRIPTOR,
                                                       1,
                                                       &x_desc));

        // Set scale and bias tensors
        CUDNN_FE_VALIDATE_AND_ASSIGN_INPUT_TENSOR(SCALE, Instancenorm_attributes::input_names::SCALE);
        auto scale_desc = tensors.at(SCALE->second->get_uid())->get_raw_desc();

        _CUDNN_CHECK_CUDNN_ERROR(detail::set_attribute(instancenorm_operation.get_raw_desc(),
                                                       CUDNN_ATTR_OPERATION_NORM_FWD_SCALE_DESC,
                                                       CUDNN_TYPE_BACKEND_DESCRIPTOR,
                                                       1,
                                                       &scale_desc));

        CUDNN_FE_VALIDATE_AND_ASSIGN_INPUT_TENSOR(BIAS, Instancenorm_attributes::input_names::BIAS);
        auto bias_desc = tensors.at(BIAS->second->get_uid())->get_raw_desc();

        _CUDNN_CHECK_CUDNN_ERROR(detail::set_attribute(instancenorm_operation.get_raw_desc(),
                                                       CUDNN_ATTR_OPERATION_NORM_FWD_BIAS_DESC,
                                                       CUDNN_TYPE_BACKEND_DESCRIPTOR,
                                                       1,
                                                       &bias_desc));

        // Set epsilon tensor
        CUDNN_FE_VALIDATE_AND_ASSIGN_INPUT_TENSOR(EPSILON, Instancenorm_attributes::input_names::EPSILON);
        auto epsilon_desc = tensors.at(EPSILON->second->get_uid())->get_raw_desc();

        _CUDNN_CHECK_CUDNN_ERROR(detail::set_attribute(instancenorm_operation.get_raw_desc(),
                                                       CUDNN_ATTR_OPERATION_NORM_FWD_EPSILON_DESC,
                                                       CUDNN_TYPE_BACKEND_DESCRIPTOR,
                                                       1,
                                                       &epsilon_desc));

        // Set output tensor Y
        CUDNN_FE_VALIDATE_AND_ASSIGN_OUTPUT_TENSOR(Y, Instancenorm_attributes::output_names::Y);
        auto y_desc = tensors.at(Y->second->get_uid())->get_raw_desc();

        _CUDNN_CHECK_CUDNN_ERROR(detail::set_attribute(instancenorm_operation.get_raw_desc(),
                                                       CUDNN_ATTR_OPERATION_NORM_FWD_YDESC,
                                                       CUDNN_TYPE_BACKEND_DESCRIPTOR,
                                                       1,
                                                       &y_desc));

        // Set mean and inv_variance for training phase
        if (attributes.forward_phase == NormFwdPhase_t::TRAINING) {
            CUDNN_FE_VALIDATE_AND_ASSIGN_OUTPUT_TENSOR(MEAN, Instancenorm_attributes::output_names::MEAN);
            auto mean_desc = tensors.at(MEAN->second->get_uid())->get_raw_desc();

            _CUDNN_CHECK_CUDNN_ERROR(detail::set_attribute(instancenorm_operation.get_raw_desc(),
                                                           CUDNN_ATTR_OPERATION_NORM_FWD_MEAN_DESC,
                                                           CUDNN_TYPE_BACKEND_DESCRIPTOR,
                                                           1,
                                                           &mean_desc));

            CUDNN_FE_VALIDATE_AND_ASSIGN_OUTPUT_TENSOR(INV_VARIANCE,
                                                       Instancenorm_attributes::output_names::INV_VARIANCE);
            auto inv_var_desc = tensors.at(INV_VARIANCE->second->get_uid())->get_raw_desc();

            _CUDNN_CHECK_CUDNN_ERROR(detail::set_attribute(instancenorm_operation.get_raw_desc(),
                                                           CUDNN_ATTR_OPERATION_NORM_FWD_INV_VARIANCE_DESC,
                                                           CUDNN_TYPE_BACKEND_DESCRIPTOR,
                                                           1,
                                                           &inv_var_desc));
        }

        _CUDNN_CHECK_CUDNN_ERROR(detail::finalize(instancenorm_operation.get_raw_desc()));

        operations.push_back(std::make_shared<Operation_v8>(std::move(instancenorm_operation)));

        auto const& non_virtual_uids = attributes.get_non_virtual_uids();
        uids_involved_in_operations.insert(non_virtual_uids.begin(), non_virtual_uids.end());
        return {error_code_t::OK, ""};
    }

#ifndef CUDNN_FRONTEND_SKIP_JSON_LIB
    virtual void
    serialize(json& j) const override final {
        j = attributes;
        j.update(R"( {"tag": "INSTANCE_NORM"})"_json);
    }
#endif
};

class DINNode : public NodeCRTP<DINNode> {
   public:
    Instancenorm_backward_attributes attributes;

    DINNode(Instancenorm_backward_attributes&& attributes_, detail::Context const& context)
        : NodeCRTP(context), attributes(std::move(attributes_)) {}

    Type
    getType() override final {
        return Type::DIN;
    }

    error_t
    infer_properties_node() override final {
        CUDNN_FE_LOG_LABEL_ENDL("INFO: Inferencing properties for DIN node " << attributes.name);

        attributes.fill_from_context(context);

        // TODO: Only inferencing from X works today.
        auto X                  = attributes.inputs[Instancenorm_backward_attributes::input_names::X];
        auto const x_tensor_dim = X->get_dim();

        auto DY            = attributes.inputs[Instancenorm_backward_attributes::input_names::DY];
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

        auto DX            = attributes.outputs[Instancenorm_backward_attributes::output_names::DX];
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

        // scale_bias   dim is 1,c,1,1
        // mean inv_var dim is n,c,1,1
        auto scale_bias_dim = X->get_dim();
        for (size_t i = 0; i < scale_bias_dim.size(); i++) {
            if (i != 1) {
                scale_bias_dim[i] = 1;
            }
        }

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

        infer_scale_bias_tensors(attributes.outputs[Instancenorm_backward_attributes::output_names::DSCALE]);
        infer_scale_bias_tensors(attributes.outputs[Instancenorm_backward_attributes::output_names::DBIAS]);

        return {error_code_t::OK, ""};
    }

    error_t
    create_cudnn_operations(
        std::unordered_set<Tensor_attributes::uid_t>& uids_involved_in_operations,
        std::vector<std::shared_ptr<cudnn_frontend::Operation>>& operations,
        managed_backend_descriptor_t& raw_operations,
        std::unordered_map<int64_t, std::shared_ptr<cudnn_frontend::Tensor>>& tensors) const override final {
        CUDNN_FRONTEND_UNUSED(raw_operations);
        CUDNN_FE_LOG_LABEL("INFO: Building DINNode operations " << attributes.name << " ");

        // Create operation by directly calling cuDNN backend API
        Operation_v8 din_operation;

        _CUDNN_CHECK_CUDNN_ERROR(
            din_operation.initialize_managed_backend_pointer(CUDNN_BACKEND_OPERATION_NORM_BACKWARD_DESCRIPTOR));

        // Set norm mode to INSTANCE_NORM
        cudnnBackendNormMode_t cudnn_norm_mode;

        _CUDNN_CHECK_CUDNN_ERROR(detail::convert_to_cudnn_type(NormMode_t::INSTANCE_NORM, cudnn_norm_mode));

        _CUDNN_CHECK_CUDNN_ERROR(detail::set_attribute(din_operation.get_raw_desc(),
                                                       CUDNN_ATTR_OPERATION_NORM_BWD_MODE,
                                                       CUDNN_TYPE_NORM_MODE,
                                                       1,
                                                       &cudnn_norm_mode));

        // Set input tensor X
        CUDNN_FE_VALIDATE_AND_ASSIGN_INPUT_TENSOR(X, Instancenorm_backward_attributes::input_names::X);
        auto x_desc = tensors.at(X->second->get_uid())->get_raw_desc();

        _CUDNN_CHECK_CUDNN_ERROR(detail::set_attribute(din_operation.get_raw_desc(),
                                                       CUDNN_ATTR_OPERATION_NORM_BWD_XDESC,
                                                       CUDNN_TYPE_BACKEND_DESCRIPTOR,
                                                       1,
                                                       &x_desc));

        // Set DY tensor
        CUDNN_FE_VALIDATE_AND_ASSIGN_INPUT_TENSOR(DY, Instancenorm_backward_attributes::input_names::DY);
        auto dy_desc = tensors.at(DY->second->get_uid())->get_raw_desc();

        _CUDNN_CHECK_CUDNN_ERROR(detail::set_attribute(din_operation.get_raw_desc(),
                                                       CUDNN_ATTR_OPERATION_NORM_BWD_DYDESC,
                                                       CUDNN_TYPE_BACKEND_DESCRIPTOR,
                                                       1,
                                                       &dy_desc));

        // Set scale tensor
        CUDNN_FE_VALIDATE_AND_ASSIGN_INPUT_TENSOR(SCALE, Instancenorm_backward_attributes::input_names::SCALE);
        auto scale_desc = tensors.at(SCALE->second->get_uid())->get_raw_desc();

        _CUDNN_CHECK_CUDNN_ERROR(detail::set_attribute(din_operation.get_raw_desc(),
                                                       CUDNN_ATTR_OPERATION_NORM_BWD_SCALE_DESC,
                                                       CUDNN_TYPE_BACKEND_DESCRIPTOR,
                                                       1,
                                                       &scale_desc));

        // Set mean and inv_variance tensors
        CUDNN_FE_VALIDATE_AND_ASSIGN_INPUT_TENSOR(MEAN, Instancenorm_backward_attributes::input_names::MEAN);
        auto mean_desc = tensors.at(MEAN->second->get_uid())->get_raw_desc();

        _CUDNN_CHECK_CUDNN_ERROR(detail::set_attribute(din_operation.get_raw_desc(),
                                                       CUDNN_ATTR_OPERATION_NORM_BWD_MEAN_DESC,
                                                       CUDNN_TYPE_BACKEND_DESCRIPTOR,
                                                       1,
                                                       &mean_desc));

        CUDNN_FE_VALIDATE_AND_ASSIGN_INPUT_TENSOR(INV_VARIANCE,
                                                  Instancenorm_backward_attributes::input_names::INV_VARIANCE);
        auto inv_var_desc = tensors.at(INV_VARIANCE->second->get_uid())->get_raw_desc();

        _CUDNN_CHECK_CUDNN_ERROR(detail::set_attribute(din_operation.get_raw_desc(),
                                                       CUDNN_ATTR_OPERATION_NORM_BWD_INV_VARIANCE_DESC,
                                                       CUDNN_TYPE_BACKEND_DESCRIPTOR,
                                                       1,
                                                       &inv_var_desc));

        // Set DSCALE and DBIAS output tensors
        CUDNN_FE_VALIDATE_AND_ASSIGN_OUTPUT_TENSOR(DSCALE, Instancenorm_backward_attributes::output_names::DSCALE);
        auto dscale_desc = tensors.at(DSCALE->second->get_uid())->get_raw_desc();

        _CUDNN_CHECK_CUDNN_ERROR(detail::set_attribute(din_operation.get_raw_desc(),
                                                       CUDNN_ATTR_OPERATION_NORM_BWD_DSCALE_DESC,
                                                       CUDNN_TYPE_BACKEND_DESCRIPTOR,
                                                       1,
                                                       &dscale_desc));

        CUDNN_FE_VALIDATE_AND_ASSIGN_OUTPUT_TENSOR(DBIAS, Instancenorm_backward_attributes::output_names::DBIAS);
        auto dbias_desc = tensors.at(DBIAS->second->get_uid())->get_raw_desc();

        _CUDNN_CHECK_CUDNN_ERROR(detail::set_attribute(din_operation.get_raw_desc(),
                                                       CUDNN_ATTR_OPERATION_NORM_BWD_DBIAS_DESC,
                                                       CUDNN_TYPE_BACKEND_DESCRIPTOR,
                                                       1,
                                                       &dbias_desc));

        // Set DX output tensor
        CUDNN_FE_VALIDATE_AND_ASSIGN_OUTPUT_TENSOR(DX, Instancenorm_backward_attributes::output_names::DX);
        auto dx_desc = tensors.at(DX->second->get_uid())->get_raw_desc();

        _CUDNN_CHECK_CUDNN_ERROR(detail::set_attribute(din_operation.get_raw_desc(),
                                                       CUDNN_ATTR_OPERATION_NORM_BWD_DXDESC,
                                                       CUDNN_TYPE_BACKEND_DESCRIPTOR,
                                                       1,
                                                       &dx_desc));

        _CUDNN_CHECK_CUDNN_ERROR(detail::finalize(din_operation.get_raw_desc()));

        operations.push_back(std::make_shared<Operation_v8>(std::move(din_operation)));

        auto const& non_virtual_uids = attributes.get_non_virtual_uids();
        uids_involved_in_operations.insert(non_virtual_uids.begin(), non_virtual_uids.end());
        return {error_code_t::OK, ""};
    }

#ifndef CUDNN_FRONTEND_SKIP_JSON_LIB
    virtual void
    serialize(json& j) const override final {
        j = attributes;
        j.update(R"( {"tag": "INSTANCE_NORM_BPROP"})"_json);
    }
#endif
};

}  // namespace graph

}  // namespace cudnn_frontend