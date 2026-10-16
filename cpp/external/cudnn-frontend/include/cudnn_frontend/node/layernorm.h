#pragma once

#include "../graph_helpers.h"
#include "../node_interface.h"

namespace cudnn_frontend {

namespace graph {
class LayerNormNode : public NodeCRTP<LayerNormNode> {
   public:
    Layernorm_attributes attributes;

    LayerNormNode(Layernorm_attributes&& attributes_, detail::Context const& context)
        : NodeCRTP(context), attributes(std::move(attributes_)) {}

    Type
    getType() override final {
        return Type::LAYERNORM;
    }

    error_t
    infer_properties_node() override final {
        CUDNN_FE_LOG_LABEL_ENDL("INFO: Inferencing properties for layernorm node " << attributes.name);

        attributes.fill_from_context(context);

        auto X = attributes.inputs[Layernorm_attributes::input_names::X];
        auto Y = attributes.outputs[Layernorm_attributes::output_names::Y];

        // Only infer dims and strides if user did not set them
        if (Y->get_dim().empty()) {
            Y->set_dim(X->get_dim());
        }
        if (Y->get_stride().empty()) {
            Y->set_stride(X->get_stride());
        }

        // scale_bias dim is 1,c,h,w
        auto scale_bias_dim = X->get_dim();
        scale_bias_dim[0]   = 1;

        auto scale = attributes.inputs[Layernorm_attributes::input_names::SCALE];
        // Only infer dims and strides if user did not set them
        if (scale->get_dim().empty()) {
            scale->set_dim(scale_bias_dim);
        }
        if (scale->get_stride().empty()) {
            auto const& scale_dim = scale->get_dim();
            std::vector<int64_t> stride_order;
            CHECK_CUDNN_FRONTEND_ERROR(
                detail::generate_stride_order_preserving_format(X->get_stride(), scale_dim.size(), stride_order));
            scale->set_stride(detail::generate_stride(scale_dim, stride_order));
        }

        auto bias = attributes.inputs[Layernorm_attributes::input_names::BIAS];
        // Only infer dims and strides if user did not set them
        if (bias->get_dim().empty()) {
            bias->set_dim(scale_bias_dim);
        }
        if (bias->get_stride().empty()) {
            auto const& bias_dim = bias->get_dim();
            std::vector<int64_t> stride_order;
            CHECK_CUDNN_FRONTEND_ERROR(
                detail::generate_stride_order_preserving_format(X->get_stride(), bias_dim.size(), stride_order));
            bias->set_stride(detail::generate_stride(bias_dim, stride_order));
        }

        if (attributes.forward_phase == NormFwdPhase_t::TRAINING) {
            // stats dim is x where scale == 1 else 1
            auto stats_dim = X->get_dim();
            for (size_t i = 0; i < stats_dim.size(); i++) {
                if (scale->get_dim()[i] != 1) {
                    stats_dim[i] = 1;
                }
            }

            auto mean = attributes.outputs[Layernorm_attributes::output_names::MEAN];
            // Only infer dims and strides if user did not set them
            if (mean->get_dim().empty()) {
                mean->set_dim(stats_dim);
            }
            if (mean->get_stride().empty()) {
                auto const& mean_dim = mean->get_dim();
                std::vector<int64_t> stride_order;
                CHECK_CUDNN_FRONTEND_ERROR(
                    detail::generate_stride_order_preserving_format(X->get_stride(), mean_dim.size(), stride_order));
                mean->set_stride(detail::generate_stride(mean_dim, stride_order));
            }

            auto inv_var = attributes.outputs[Layernorm_attributes::output_names::INV_VARIANCE];
            // Only infer dims and strides if user did not set them
            if (inv_var->get_dim().empty()) {
                inv_var->set_dim(stats_dim);
            }
            if (inv_var->get_stride().empty()) {
                auto const& inv_var_dim = inv_var->get_dim();
                std::vector<int64_t> stride_order;
                CHECK_CUDNN_FRONTEND_ERROR(
                    detail::generate_stride_order_preserving_format(X->get_stride(), inv_var_dim.size(), stride_order));
                inv_var->set_stride(detail::generate_stride(inv_var_dim, stride_order));
            }
        }

        // Set scalar tensors
        std::vector<int64_t> ones(X->get_dim().size(), 1);
        auto infer_scalar_tensors = [&ones](std::shared_ptr<Tensor_attributes>& T) {
            // Only infer dims and strides if user did not set them
            if (T->get_dim().empty()) {
                T->set_dim(ones);
            }
            if (T->get_stride().empty()) {
                T->set_stride(ones);
            }
        };
        infer_scalar_tensors(attributes.inputs[Layernorm_attributes::input_names::EPSILON]);

        return {error_code_t::OK, ""};
    }

    error_t
    pre_validate_node() const override final {
        CUDNN_FE_LOG_LABEL_ENDL("INFO: " << "Validating LayerNormNode " << attributes.name);

        // Norm forward phase should be set
        RETURN_CUDNN_FRONTEND_ERROR_IF(attributes.forward_phase == NormFwdPhase_t::NOT_SET,
                                       error_code_t::ATTRIBUTE_NOT_SET,
                                       "Forward phase not set of layernorm node.");

        return {error_code_t::OK, ""};
    }

    error_t
    create_cudnn_operations(
        std::unordered_set<Tensor_attributes::uid_t>& uids_involved_in_operations,
        std::vector<std::shared_ptr<cudnn_frontend::Operation>>& operations,
        managed_backend_descriptor_t& raw_operations,
        std::unordered_map<int64_t, std::shared_ptr<cudnn_frontend::Tensor>>& tensors) const override final {
        CUDNN_FRONTEND_UNUSED(raw_operations);
        CUDNN_FE_LOG_LABEL("INFO: " << "Building LayerNormNode operations " << attributes.name << " ");

        // Create operation by directly calling cuDNN backend API
        Operation_v8 layernorm_operation;

        _CUDNN_CHECK_CUDNN_ERROR(
            layernorm_operation.initialize_managed_backend_pointer(CUDNN_BACKEND_OPERATION_NORM_FORWARD_DESCRIPTOR));

        // Set norm mode
        cudnnBackendNormMode_t cudnn_norm_mode;

        _CUDNN_CHECK_CUDNN_ERROR(detail::convert_to_cudnn_type(NormMode_t::LAYER_NORM, cudnn_norm_mode));

        _CUDNN_CHECK_CUDNN_ERROR(detail::set_attribute(layernorm_operation.get_raw_desc(),
                                                       CUDNN_ATTR_OPERATION_NORM_FWD_MODE,
                                                       CUDNN_TYPE_NORM_MODE,
                                                       1,
                                                       &cudnn_norm_mode));

        // Set forward phase
        cudnnBackendNormFwdPhase_t cudnn_norm_fwd_phase;

        _CUDNN_CHECK_CUDNN_ERROR(detail::convert_to_cudnn_type(attributes.forward_phase, cudnn_norm_fwd_phase));

        _CUDNN_CHECK_CUDNN_ERROR(detail::set_attribute(layernorm_operation.get_raw_desc(),
                                                       CUDNN_ATTR_OPERATION_NORM_FWD_PHASE,
                                                       CUDNN_TYPE_NORM_FWD_PHASE,
                                                       1,
                                                       &cudnn_norm_fwd_phase));

        // Set input tensor X
        CUDNN_FE_VALIDATE_AND_ASSIGN_INPUT_TENSOR(X, Layernorm_attributes::input_names::X);
        auto x_desc = tensors.at(X->second->get_uid())->get_raw_desc();

        _CUDNN_CHECK_CUDNN_ERROR(detail::set_attribute(layernorm_operation.get_raw_desc(),
                                                       CUDNN_ATTR_OPERATION_NORM_FWD_XDESC,
                                                       CUDNN_TYPE_BACKEND_DESCRIPTOR,
                                                       1,
                                                       &x_desc));

        // Set scale and bias tensors
        CUDNN_FE_VALIDATE_AND_ASSIGN_INPUT_TENSOR(SCALE, Layernorm_attributes::input_names::SCALE);
        auto scale_desc = tensors.at(SCALE->second->get_uid())->get_raw_desc();

        _CUDNN_CHECK_CUDNN_ERROR(detail::set_attribute(layernorm_operation.get_raw_desc(),
                                                       CUDNN_ATTR_OPERATION_NORM_FWD_SCALE_DESC,
                                                       CUDNN_TYPE_BACKEND_DESCRIPTOR,
                                                       1,
                                                       &scale_desc));

        CUDNN_FE_VALIDATE_AND_ASSIGN_INPUT_TENSOR(BIAS, Layernorm_attributes::input_names::BIAS);
        auto bias_desc = tensors.at(BIAS->second->get_uid())->get_raw_desc();

        _CUDNN_CHECK_CUDNN_ERROR(detail::set_attribute(layernorm_operation.get_raw_desc(),
                                                       CUDNN_ATTR_OPERATION_NORM_FWD_BIAS_DESC,
                                                       CUDNN_TYPE_BACKEND_DESCRIPTOR,
                                                       1,
                                                       &bias_desc));

        // Set epsilon tensor
        CUDNN_FE_VALIDATE_AND_ASSIGN_INPUT_TENSOR(EPSILON, Layernorm_attributes::input_names::EPSILON);
        auto epsilon_desc = tensors.at(EPSILON->second->get_uid())->get_raw_desc();

        _CUDNN_CHECK_CUDNN_ERROR(detail::set_attribute(layernorm_operation.get_raw_desc(),
                                                       CUDNN_ATTR_OPERATION_NORM_FWD_EPSILON_DESC,
                                                       CUDNN_TYPE_BACKEND_DESCRIPTOR,
                                                       1,
                                                       &epsilon_desc));

        // Set output tensor Y
        CUDNN_FE_VALIDATE_AND_ASSIGN_OUTPUT_TENSOR(Y, Layernorm_attributes::output_names::Y);
        auto y_desc = tensors.at(Y->second->get_uid())->get_raw_desc();

        _CUDNN_CHECK_CUDNN_ERROR(detail::set_attribute(layernorm_operation.get_raw_desc(),
                                                       CUDNN_ATTR_OPERATION_NORM_FWD_YDESC,
                                                       CUDNN_TYPE_BACKEND_DESCRIPTOR,
                                                       1,
                                                       &y_desc));

        // Set mean and inv_variance for training phase
        if (attributes.forward_phase == NormFwdPhase_t::TRAINING) {
            CUDNN_FE_VALIDATE_AND_ASSIGN_OUTPUT_TENSOR(MEAN, Layernorm_attributes::output_names::MEAN);
            auto mean_desc = tensors.at(MEAN->second->get_uid())->get_raw_desc();

            _CUDNN_CHECK_CUDNN_ERROR(detail::set_attribute(layernorm_operation.get_raw_desc(),
                                                           CUDNN_ATTR_OPERATION_NORM_FWD_MEAN_DESC,
                                                           CUDNN_TYPE_BACKEND_DESCRIPTOR,
                                                           1,
                                                           &mean_desc));

            CUDNN_FE_VALIDATE_AND_ASSIGN_OUTPUT_TENSOR(INV_VARIANCE, Layernorm_attributes::output_names::INV_VARIANCE);
            auto inv_var_desc = tensors.at(INV_VARIANCE->second->get_uid())->get_raw_desc();

            _CUDNN_CHECK_CUDNN_ERROR(detail::set_attribute(layernorm_operation.get_raw_desc(),
                                                           CUDNN_ATTR_OPERATION_NORM_FWD_INV_VARIANCE_DESC,
                                                           CUDNN_TYPE_BACKEND_DESCRIPTOR,
                                                           1,
                                                           &inv_var_desc));
        }

        _CUDNN_CHECK_CUDNN_ERROR(detail::finalize(layernorm_operation.get_raw_desc()));

        operations.push_back(std::make_shared<Operation_v8>(std::move(layernorm_operation)));

        auto const& non_virtual_uids = attributes.get_non_virtual_uids();
        uids_involved_in_operations.insert(non_virtual_uids.begin(), non_virtual_uids.end());
        return {error_code_t::OK, ""};
    }

#ifndef CUDNN_FRONTEND_SKIP_JSON_LIB
    virtual void
    serialize(json& j) const override final {
        j = attributes;
        j.update(R"( {"tag": "LAYER_NORM"})"_json);
    }
#endif
};

}  // namespace graph

}  // namespace cudnn_frontend