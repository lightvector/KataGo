#pragma once

#include "../../cudnn_frontend_Heuristics.h"
#include "../../cudnn_frontend_Logging.h"

#include "../graph_helpers.h"
#include "../node_interface.h"

namespace cudnn_frontend {

namespace graph {
class AdaLayerNormNode : public NodeCRTP<AdaLayerNormNode> {
   public:
    AdaLayernorm_attributes attributes;

    AdaLayerNormNode(AdaLayernorm_attributes&& attributes_, detail::Context const& context)
        : NodeCRTP(context), attributes(std::move(attributes_)) {}

    Type
    getType() override final {
        return Type::ADALAYERNORM;
    }

    error_t
    infer_properties_node() override final {
        CUDNN_FE_LOG_LABEL_ENDL("INFO: Inferencing properties for adalayernorm node " << attributes.name);

        attributes.fill_from_context(context);

        auto X = attributes.inputs[AdaLayernorm_attributes::input_names::X];
        auto Y = attributes.outputs[AdaLayernorm_attributes::output_names::Y];

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

        auto scale = attributes.inputs[AdaLayernorm_attributes::input_names::SCALE];
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

        auto bias = attributes.inputs[AdaLayernorm_attributes::input_names::BIAS];
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
            for (size_t i = 1; i < stats_dim.size(); i++) {
                if (scale->get_dim()[i] != 1) {
                    stats_dim[i] = 1;
                }
            }

            auto mean = attributes.outputs[AdaLayernorm_attributes::output_names::MEAN];
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

            auto inv_var = attributes.outputs[AdaLayernorm_attributes::output_names::INV_VARIANCE];
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
        infer_scalar_tensors(attributes.inputs[AdaLayernorm_attributes::input_names::EPSILON]);

        return {error_code_t::OK, ""};
    }

    error_t
    pre_validate_node() const override final {
        CUDNN_FE_LOG_LABEL_ENDL("INFO: " << "Validating AdaLayerNormNode " << attributes.name);
        // Norm forward phase should be set
        RETURN_CUDNN_FRONTEND_ERROR_IF(attributes.forward_phase == NormFwdPhase_t::NOT_SET,
                                       error_code_t::ATTRIBUTE_NOT_SET,
                                       "Forward phase not set of adalayernorm node.");

        return {error_code_t::OK, ""};
    }

    error_t
    create_cudnn_operations(
        std::unordered_set<Tensor_attributes::uid_t>& uids_involved_in_operations,
        std::vector<std::shared_ptr<cudnn_frontend::Operation>>& operations,
        managed_backend_descriptor_t& raw_operations,
        std::unordered_map<int64_t, std::shared_ptr<cudnn_frontend::Tensor>>& tensors) const override final {
        getLogger() << "[cudnn_frontend] INFO: " << "Building AdaLayernorm operations " << attributes.name << std::endl;

        auto cudnn_ver_error = error_t{error_code_t::GRAPH_NOT_SUPPORTED, "AdaLN fwd requires cuDNN v9.9.0"};
#if (CUDNN_VERSION >= 90900)
        NV_CUDNN_FE_DYNAMIC_CHECK_CUDNN_BACKEND_VERSION(90900, cudnn_ver_error);
        CUDNN_FRONTEND_UNUSED(operations);
        auto adalayernorm_operation =
            make_shared_backend_pointer((cudnnBackendDescriptorType_t)CUDNN_BACKEND_OPERATION_NORM_FORWARD_DESCRIPTOR);

        cudnnBackendNormMode_t cudnn_norm_mode;
        _CUDNN_CHECK_CUDNN_ERROR(detail::convert_to_cudnn_type(NormMode_t::ADA_LAYER_NORM, cudnn_norm_mode));
        _CUDNN_CHECK_CUDNN_ERROR(detail::set_attribute(adalayernorm_operation->get_backend_descriptor(),
                                                       CUDNN_ATTR_OPERATION_NORM_FWD_MODE,
                                                       CUDNN_TYPE_NORM_MODE,
                                                       1,
                                                       &cudnn_norm_mode));

        cudnnBackendNormFwdPhase_t cudnn_norm_fwd_phase;
        _CUDNN_CHECK_CUDNN_ERROR(detail::convert_to_cudnn_type(attributes.forward_phase, cudnn_norm_fwd_phase));

        _CUDNN_CHECK_CUDNN_ERROR(detail::set_attribute(adalayernorm_operation->get_backend_descriptor(),
                                                       CUDNN_ATTR_OPERATION_NORM_FWD_PHASE,
                                                       CUDNN_TYPE_NORM_FWD_PHASE,
                                                       1,
                                                       &cudnn_norm_fwd_phase));

        auto X         = attributes.inputs.find(AdaLayernorm_attributes::input_names::X)->second;
        auto backend_x = tensors[X->get_uid()]->get_desc()->get_backend_descriptor();
        _CUDNN_CHECK_CUDNN_ERROR(detail::set_attribute(adalayernorm_operation->get_backend_descriptor(),
                                                       CUDNN_ATTR_OPERATION_NORM_FWD_XDESC,
                                                       CUDNN_TYPE_BACKEND_DESCRIPTOR,
                                                       1,
                                                       &backend_x));

        auto Scale         = attributes.inputs.find(AdaLayernorm_attributes::input_names::SCALE)->second;
        auto backend_scale = tensors[Scale->get_uid()]->get_desc()->get_backend_descriptor();
        _CUDNN_CHECK_CUDNN_ERROR(detail::set_attribute(adalayernorm_operation->get_backend_descriptor(),
                                                       CUDNN_ATTR_OPERATION_NORM_FWD_SCALE_DESC,
                                                       CUDNN_TYPE_BACKEND_DESCRIPTOR,
                                                       1,
                                                       &backend_scale));

        auto Bias_iter = attributes.inputs.find(AdaLayernorm_attributes::input_names::BIAS);
        if (Bias_iter != attributes.inputs.end() && Bias_iter->second->get_is_virtual() == false) {
            auto backend_bias = tensors[Bias_iter->second->get_uid()]->get_desc()->get_backend_descriptor();
            _CUDNN_CHECK_CUDNN_ERROR(detail::set_attribute(adalayernorm_operation->get_backend_descriptor(),
                                                           CUDNN_ATTR_OPERATION_NORM_FWD_BIAS_DESC,
                                                           CUDNN_TYPE_BACKEND_DESCRIPTOR,
                                                           1,
                                                           &backend_bias));
        }

        auto Epsilon         = attributes.inputs.find(AdaLayernorm_attributes::input_names::EPSILON)->second;
        auto backend_epsilon = tensors[Epsilon->get_uid()]->get_desc()->get_backend_descriptor();
        _CUDNN_CHECK_CUDNN_ERROR(detail::set_attribute(adalayernorm_operation->get_backend_descriptor(),
                                                       CUDNN_ATTR_OPERATION_NORM_FWD_EPSILON_DESC,
                                                       CUDNN_TYPE_BACKEND_DESCRIPTOR,
                                                       1,
                                                       &backend_epsilon));

        auto Y         = attributes.outputs.find(AdaLayernorm_attributes::output_names::Y)->second;
        auto backend_y = tensors[Y->get_uid()]->get_desc()->get_backend_descriptor();
        _CUDNN_CHECK_CUDNN_ERROR(detail::set_attribute(adalayernorm_operation->get_backend_descriptor(),
                                                       CUDNN_ATTR_OPERATION_NORM_FWD_YDESC,
                                                       CUDNN_TYPE_BACKEND_DESCRIPTOR,
                                                       1,
                                                       &backend_y));

        if (attributes.forward_phase == NormFwdPhase_t::TRAINING) {
            auto Mean         = attributes.outputs.find(AdaLayernorm_attributes::output_names::MEAN)->second;
            auto backend_mean = tensors[Mean->get_uid()]->get_desc()->get_backend_descriptor();
            _CUDNN_CHECK_CUDNN_ERROR(detail::set_attribute(adalayernorm_operation->get_backend_descriptor(),
                                                           CUDNN_ATTR_OPERATION_NORM_FWD_MEAN_DESC,
                                                           CUDNN_TYPE_BACKEND_DESCRIPTOR,
                                                           1,
                                                           &backend_mean));

            auto Inv_variance = attributes.outputs.find(AdaLayernorm_attributes::output_names::INV_VARIANCE)->second;
            auto backend_inv_variance = tensors[Inv_variance->get_uid()]->get_desc()->get_backend_descriptor();
            _CUDNN_CHECK_CUDNN_ERROR(detail::set_attribute(adalayernorm_operation->get_backend_descriptor(),
                                                           CUDNN_ATTR_OPERATION_NORM_FWD_INV_VARIANCE_DESC,
                                                           CUDNN_TYPE_BACKEND_DESCRIPTOR,
                                                           1,
                                                           &backend_inv_variance));
        }

        _CUDNN_CHECK_CUDNN_ERROR(detail::finalize(adalayernorm_operation->get_backend_descriptor()));

        raw_operations.push_back(adalayernorm_operation);

        auto const& non_virtual_uids = attributes.get_non_virtual_uids();
        uids_involved_in_operations.insert(non_virtual_uids.begin(), non_virtual_uids.end());
        return {error_code_t::OK, ""};
#else
        CUDNN_FRONTEND_UNUSED(uids_involved_in_operations);
        CUDNN_FRONTEND_UNUSED(operations);
        CUDNN_FRONTEND_UNUSED(raw_operations);
        CUDNN_FRONTEND_UNUSED(tensors);
        return cudnn_ver_error;
#endif
    }

#ifndef CUDNN_FRONTEND_SKIP_JSON_LIB
    virtual void
    serialize(json& j) const override final {
        j = attributes;
        j.update(R"( {"tag": "ADA_LAYER_NORM"})"_json);
    }
#endif
};

/*******/

class DAdaLayerNormNode : public NodeCRTP<DAdaLayerNormNode> {
   public:
    AdaLayernorm_backward_attributes attributes;

    DAdaLayerNormNode(AdaLayernorm_backward_attributes&& attributes_, detail::Context const& context)
        : NodeCRTP(context), attributes(std::move(attributes_)) {}

    Type
    getType() override final {
        return Type::DADALAYERNORM;
    }

    error_t
    infer_properties_node() override final {
        CUDNN_FE_LOG_LABEL_ENDL("INFO: Inferencing properties for DAdaLayerNorm node " << attributes.name);

        attributes.fill_from_context(context);

        // TODO: Only inferencing from X works today.
        auto X                  = attributes.inputs[AdaLayernorm_backward_attributes::input_names::X];
        auto const x_tensor_dim = X->get_dim();

        auto DY            = attributes.inputs[AdaLayernorm_backward_attributes::input_names::DY];
        auto dy_tensor_dim = DY->get_dim();

        // Only infer dims and strides if user did not set them
        if (dy_tensor_dim.empty()) {
            dy_tensor_dim.resize(x_tensor_dim.size());
            DY->set_dim(x_tensor_dim);
        }
        if (DY->get_stride().empty()) {
            auto const& DY_dim = DY->get_dim();
            // Default to NCHW
            auto const& stride_order = detail::generate_row_major_stride_order(DY_dim.size());
            DY->set_stride(detail::generate_stride(DY_dim, stride_order));
        }

        auto DX            = attributes.outputs[AdaLayernorm_backward_attributes::output_names::DX];
        auto dx_tensor_dim = DX->get_dim();
        // Only infer dims and strides if user did not set them
        if (dx_tensor_dim.empty()) {
            dx_tensor_dim.resize(x_tensor_dim.size());
            DX->set_dim(x_tensor_dim);
        }
        if (DX->get_stride().empty()) {
            auto const& DX_dim = DX->get_dim();
            // Default to NCHW
            auto const& stride_order = detail::generate_row_major_stride_order(DX_dim.size());
            DX->set_stride(detail::generate_stride(DX_dim, stride_order));
        }

        auto SCALE          = attributes.inputs[AdaLayernorm_backward_attributes::input_names::SCALE];
        auto scale_bias_dim = SCALE->get_dim();

        // Set channel length tensors
        auto infer_scale_bias_tensors = [&scale_bias_dim](std::shared_ptr<Tensor_attributes>& T) {
            auto tensor_dim = T->get_dim();
            // Only infer dims and strides if user did not set them
            if (tensor_dim.empty()) {
                T->set_dim(scale_bias_dim);
            }
            if (T->get_stride().empty()) {
                auto const& T_dim = T->get_dim();
                // Default to NCHW
                auto const& stride_order = detail::generate_row_major_stride_order(T_dim.size());
                T->set_stride(detail::generate_stride(T_dim, stride_order));
            }
        };

        infer_scale_bias_tensors(attributes.outputs[AdaLayernorm_backward_attributes::output_names::DSCALE]);
        auto DBIAS = attributes.outputs.at(AdaLayernorm_backward_attributes::output_names::DBIAS);
        if (DBIAS->get_is_virtual() == false) {
            infer_scale_bias_tensors(DBIAS);
        }

        return {error_code_t::OK, ""};
    }

    error_t
    pre_validate_node() const override final {
        CUDNN_FE_LOG_LABEL_ENDL("INFO: Validating DAdaLayerNormNode node " << attributes.name);

        return {error_code_t::OK, ""};
    }

    error_t
    create_cudnn_operations(
        std::unordered_set<Tensor_attributes::uid_t>& uids_involved_in_operations,
        std::vector<std::shared_ptr<cudnn_frontend::Operation>>& operations,
        managed_backend_descriptor_t& raw_operations,
        std::unordered_map<int64_t, std::shared_ptr<cudnn_frontend::Tensor>>& tensors) const override final {
        getLogger() << "[cudnn_frontend] INFO: " << "Building DAdaLayerNormNode operations " << attributes.name
                    << std::endl;
        auto cudnn_ver_error = error_t{error_code_t::GRAPH_NOT_SUPPORTED, "AdaLN bwd requires cuDNN v9.9.0"};
#if (CUDNN_VERSION >= 90900)
        NV_CUDNN_FE_DYNAMIC_CHECK_CUDNN_BACKEND_VERSION(90900, cudnn_ver_error);
        CUDNN_FRONTEND_UNUSED(operations);
        auto adalayernorm_operation =
            make_shared_backend_pointer((cudnnBackendDescriptorType_t)CUDNN_BACKEND_OPERATION_NORM_BACKWARD_DESCRIPTOR);

        cudnnBackendNormMode_t cudnn_norm_mode;
        _CUDNN_CHECK_CUDNN_ERROR(detail::convert_to_cudnn_type(NormMode_t::ADA_LAYER_NORM, cudnn_norm_mode));
        _CUDNN_CHECK_CUDNN_ERROR(detail::set_attribute(adalayernorm_operation->get_backend_descriptor(),
                                                       CUDNN_ATTR_OPERATION_NORM_BWD_MODE,
                                                       CUDNN_TYPE_NORM_MODE,
                                                       1,
                                                       &cudnn_norm_mode));

        auto X         = attributes.inputs.find(AdaLayernorm_backward_attributes::input_names::X)->second;
        auto backend_x = tensors[X->get_uid()]->get_desc()->get_backend_descriptor();
        _CUDNN_CHECK_CUDNN_ERROR(detail::set_attribute(adalayernorm_operation->get_backend_descriptor(),
                                                       CUDNN_ATTR_OPERATION_NORM_BWD_XDESC,
                                                       CUDNN_TYPE_BACKEND_DESCRIPTOR,
                                                       1,
                                                       &backend_x));

        auto Mean         = attributes.inputs.find(AdaLayernorm_backward_attributes::input_names::MEAN)->second;
        auto backend_mean = tensors[Mean->get_uid()]->get_desc()->get_backend_descriptor();
        _CUDNN_CHECK_CUDNN_ERROR(detail::set_attribute(adalayernorm_operation->get_backend_descriptor(),
                                                       CUDNN_ATTR_OPERATION_NORM_BWD_MEAN_DESC,
                                                       CUDNN_TYPE_BACKEND_DESCRIPTOR,
                                                       1,
                                                       &backend_mean));

        auto Inv_variance = attributes.inputs.find(AdaLayernorm_backward_attributes::input_names::INV_VARIANCE)->second;
        auto backend_inv_variance = tensors[Inv_variance->get_uid()]->get_desc()->get_backend_descriptor();
        _CUDNN_CHECK_CUDNN_ERROR(detail::set_attribute(adalayernorm_operation->get_backend_descriptor(),
                                                       CUDNN_ATTR_OPERATION_NORM_BWD_INV_VARIANCE_DESC,
                                                       CUDNN_TYPE_BACKEND_DESCRIPTOR,
                                                       1,
                                                       &backend_inv_variance));

        auto Dy         = attributes.inputs.find(AdaLayernorm_backward_attributes::input_names::DY)->second;
        auto backend_dy = tensors[Dy->get_uid()]->get_desc()->get_backend_descriptor();
        _CUDNN_CHECK_CUDNN_ERROR(detail::set_attribute(adalayernorm_operation->get_backend_descriptor(),
                                                       CUDNN_ATTR_OPERATION_NORM_BWD_DYDESC,
                                                       CUDNN_TYPE_BACKEND_DESCRIPTOR,
                                                       1,
                                                       &backend_dy));

        auto Scale         = attributes.inputs.find(AdaLayernorm_backward_attributes::input_names::SCALE)->second;
        auto backend_scale = tensors[Scale->get_uid()]->get_desc()->get_backend_descriptor();
        _CUDNN_CHECK_CUDNN_ERROR(detail::set_attribute(adalayernorm_operation->get_backend_descriptor(),
                                                       CUDNN_ATTR_OPERATION_NORM_BWD_SCALE_DESC,
                                                       CUDNN_TYPE_BACKEND_DESCRIPTOR,
                                                       1,
                                                       &backend_scale));

        auto Dx         = attributes.outputs.find(AdaLayernorm_backward_attributes::output_names::DX)->second;
        auto backend_dx = tensors[Dx->get_uid()]->get_desc()->get_backend_descriptor();
        _CUDNN_CHECK_CUDNN_ERROR(detail::set_attribute(adalayernorm_operation->get_backend_descriptor(),
                                                       CUDNN_ATTR_OPERATION_NORM_BWD_DXDESC,
                                                       CUDNN_TYPE_BACKEND_DESCRIPTOR,
                                                       1,
                                                       &backend_dx));

        auto Dscale         = attributes.outputs.find(AdaLayernorm_backward_attributes::output_names::DSCALE)->second;
        auto backend_dscale = tensors[Dscale->get_uid()]->get_desc()->get_backend_descriptor();
        _CUDNN_CHECK_CUDNN_ERROR(detail::set_attribute(adalayernorm_operation->get_backend_descriptor(),
                                                       CUDNN_ATTR_OPERATION_NORM_BWD_DSCALE_DESC,
                                                       CUDNN_TYPE_BACKEND_DESCRIPTOR,
                                                       1,
                                                       &backend_dscale));

        auto Dbias_iter = attributes.outputs.find(AdaLayernorm_backward_attributes::output_names::DBIAS);
        if (Dbias_iter != attributes.outputs.end() && Dbias_iter->second->get_is_virtual() == false) {
            auto backend_dbias = tensors[Dbias_iter->second->get_uid()]->get_desc()->get_backend_descriptor();
            _CUDNN_CHECK_CUDNN_ERROR(detail::set_attribute(adalayernorm_operation->get_backend_descriptor(),
                                                           CUDNN_ATTR_OPERATION_NORM_BWD_DBIAS_DESC,
                                                           CUDNN_TYPE_BACKEND_DESCRIPTOR,
                                                           1,
                                                           &backend_dbias));
        }

        _CUDNN_CHECK_CUDNN_ERROR(detail::finalize(adalayernorm_operation->get_backend_descriptor()));

        raw_operations.push_back(adalayernorm_operation);

        auto const& non_virtual_uids = attributes.get_non_virtual_uids();
        uids_involved_in_operations.insert(non_virtual_uids.begin(), non_virtual_uids.end());
        return {error_code_t::OK, ""};
#else
        CUDNN_FRONTEND_UNUSED(uids_involved_in_operations);
        CUDNN_FRONTEND_UNUSED(operations);
        CUDNN_FRONTEND_UNUSED(raw_operations);
        CUDNN_FRONTEND_UNUSED(tensors);
        return cudnn_ver_error;
#endif
    }

#ifndef CUDNN_FRONTEND_SKIP_JSON_LIB
    virtual void
    serialize(json& j) const override final {
        j = attributes;
        j.update(R"( {"tag": "ADA_LAYER_NORM_BPROP"})"_json);
    }
#endif
};

}  // namespace graph

}  // namespace cudnn_frontend