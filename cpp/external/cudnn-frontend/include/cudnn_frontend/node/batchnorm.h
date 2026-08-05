#pragma once

#include "../graph_helpers.h"
#include "../node_interface.h"

namespace cudnn_frontend {

namespace graph {
class BatchNormNode : public NodeCRTP<BatchNormNode> {
   public:
    Batchnorm_attributes attributes;

    BatchNormNode(Batchnorm_attributes&& attributes_, detail::Context const& context)
        : NodeCRTP(context), attributes(std::move(attributes_)) {}

    Type
    getType() override final {
        return Type::BATCHNORM;
    }

    error_t
    infer_properties_node() override final {
        CUDNN_FE_LOG_LABEL_ENDL("INFO: Inferencing properties for batchnorm node " << attributes.name);

        attributes.fill_from_context(context);

        auto X = attributes.inputs[Batchnorm_attributes::input_names::X];
        auto Y = attributes.outputs[Batchnorm_attributes::output_names::Y];
        // Only infer dims and strides if user did not set them
        if (Y->get_dim().empty()) {
            Y->set_dim(X->get_dim());
        }
        if (Y->get_stride().empty()) {
            Y->set_stride(X->get_stride());
        }

        // Set channel length tensors
        auto const x_tensor_dim        = X->get_dim();
        auto infer_per_channel_tensors = [&x_tensor_dim](std::shared_ptr<Tensor_attributes>& T) {
            auto tensor_dim = T->get_dim();
            // Only infer dims and strides if user did not set them
            if (tensor_dim.empty()) {
                tensor_dim.resize(x_tensor_dim.size(), 1);
                tensor_dim[1] = x_tensor_dim[1];
                T->set_dim(tensor_dim);
            }
            if (T->get_stride().empty()) {
                auto const& T_dim = T->get_dim();
                // Default to NHWC
                auto const& stride_order = detail::generate_NHWC_stride_order(T_dim.size());
                T->set_stride(detail::generate_stride(T_dim, stride_order));
            }
        };
        infer_per_channel_tensors(attributes.outputs[Batchnorm_attributes::output_names::MEAN]);
        infer_per_channel_tensors(attributes.outputs[Batchnorm_attributes::output_names::INV_VARIANCE]);

        auto has_running_stats = attributes.inputs[Batchnorm_attributes::input_names::PREV_RUNNING_MEAN] ||
                                 attributes.inputs[Batchnorm_attributes::input_names::PREV_RUNNING_VAR];

        if (has_running_stats) {
            infer_per_channel_tensors(attributes.outputs[Batchnorm_attributes::output_names::NEXT_RUNNING_MEAN]);
            infer_per_channel_tensors(attributes.outputs[Batchnorm_attributes::output_names::NEXT_RUNNING_VAR]);
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
        CUDNN_FE_LOG_LABEL("INFO: Building BatchNormNode operations " << attributes.name << " ");

        // Create operation by directly calling cuDNN backend API
        Operation_v8 batchnorm_operation;

        _CUDNN_CHECK_CUDNN_ERROR(
            batchnorm_operation.initialize_managed_backend_pointer(CUDNN_BACKEND_OPERATION_NORM_FORWARD_DESCRIPTOR));

        // Set norm mode to BATCH_NORM
        cudnnBackendNormMode_t cudnn_norm_mode;

        _CUDNN_CHECK_CUDNN_ERROR(detail::convert_to_cudnn_type(NormMode_t::BATCH_NORM, cudnn_norm_mode));

        _CUDNN_CHECK_CUDNN_ERROR(detail::set_attribute(batchnorm_operation.get_raw_desc(),
                                                       CUDNN_ATTR_OPERATION_NORM_FWD_MODE,
                                                       CUDNN_TYPE_NORM_MODE,
                                                       1,
                                                       &cudnn_norm_mode));

        // Set forward phase to TRAINING
        cudnnBackendNormFwdPhase_t cudnn_norm_fwd_phase;

        _CUDNN_CHECK_CUDNN_ERROR(detail::convert_to_cudnn_type(NormFwdPhase_t::TRAINING, cudnn_norm_fwd_phase));

        _CUDNN_CHECK_CUDNN_ERROR(detail::set_attribute(batchnorm_operation.get_raw_desc(),
                                                       CUDNN_ATTR_OPERATION_NORM_FWD_PHASE,
                                                       CUDNN_TYPE_NORM_FWD_PHASE,
                                                       1,
                                                       &cudnn_norm_fwd_phase));

        // Set input tensor X
        CUDNN_FE_VALIDATE_AND_ASSIGN_INPUT_TENSOR(X, Batchnorm_attributes::input_names::X);
        auto x_desc = tensors.at(X->second->get_uid())->get_raw_desc();

        _CUDNN_CHECK_CUDNN_ERROR(detail::set_attribute(batchnorm_operation.get_raw_desc(),
                                                       CUDNN_ATTR_OPERATION_NORM_FWD_XDESC,
                                                       CUDNN_TYPE_BACKEND_DESCRIPTOR,
                                                       1,
                                                       &x_desc));

        // Set saved mean and inv_variance
        CUDNN_FE_VALIDATE_AND_ASSIGN_OUTPUT_TENSOR(MEAN, Batchnorm_attributes::output_names::MEAN);
        auto mean_desc = tensors.at(MEAN->second->get_uid())->get_raw_desc();

        _CUDNN_CHECK_CUDNN_ERROR(detail::set_attribute(batchnorm_operation.get_raw_desc(),
                                                       CUDNN_ATTR_OPERATION_NORM_FWD_MEAN_DESC,
                                                       CUDNN_TYPE_BACKEND_DESCRIPTOR,
                                                       1,
                                                       &mean_desc));

        CUDNN_FE_VALIDATE_AND_ASSIGN_OUTPUT_TENSOR(INV_VARIANCE, Batchnorm_attributes::output_names::INV_VARIANCE);
        auto inv_var_desc = tensors.at(INV_VARIANCE->second->get_uid())->get_raw_desc();

        _CUDNN_CHECK_CUDNN_ERROR(detail::set_attribute(batchnorm_operation.get_raw_desc(),
                                                       CUDNN_ATTR_OPERATION_NORM_FWD_INV_VARIANCE_DESC,
                                                       CUDNN_TYPE_BACKEND_DESCRIPTOR,
                                                       1,
                                                       &inv_var_desc));

        // Set scale and bias tensors
        CUDNN_FE_VALIDATE_AND_ASSIGN_INPUT_TENSOR(SCALE, Batchnorm_attributes::input_names::SCALE);
        auto scale_desc = tensors.at(SCALE->second->get_uid())->get_raw_desc();

        _CUDNN_CHECK_CUDNN_ERROR(detail::set_attribute(batchnorm_operation.get_raw_desc(),
                                                       CUDNN_ATTR_OPERATION_NORM_FWD_SCALE_DESC,
                                                       CUDNN_TYPE_BACKEND_DESCRIPTOR,
                                                       1,
                                                       &scale_desc));

        CUDNN_FE_VALIDATE_AND_ASSIGN_INPUT_TENSOR(BIAS, Batchnorm_attributes::input_names::BIAS);
        auto bias_desc = tensors.at(BIAS->second->get_uid())->get_raw_desc();

        _CUDNN_CHECK_CUDNN_ERROR(detail::set_attribute(batchnorm_operation.get_raw_desc(),
                                                       CUDNN_ATTR_OPERATION_NORM_FWD_BIAS_DESC,
                                                       CUDNN_TYPE_BACKEND_DESCRIPTOR,
                                                       1,
                                                       &bias_desc));

        // Set epsilon tensor
        CUDNN_FE_VALIDATE_AND_ASSIGN_INPUT_TENSOR(EPSILON, Batchnorm_attributes::input_names::EPSILON);
        auto epsilon_desc = tensors.at(EPSILON->second->get_uid())->get_raw_desc();

        _CUDNN_CHECK_CUDNN_ERROR(detail::set_attribute(batchnorm_operation.get_raw_desc(),
                                                       CUDNN_ATTR_OPERATION_NORM_FWD_EPSILON_DESC,
                                                       CUDNN_TYPE_BACKEND_DESCRIPTOR,
                                                       1,
                                                       &epsilon_desc));

        // Check for running stats
        bool has_running_stats = true;
        auto it                = attributes.inputs.find(Batchnorm_attributes::input_names::PREV_RUNNING_MEAN);
        if (it == attributes.inputs.end() || it->second == nullptr) {
            has_running_stats = false;
        }

        if (has_running_stats) {
            // Set momentum (exp decay factor)
            CUDNN_FE_VALIDATE_AND_ASSIGN_INPUT_TENSOR(MOMENTUM, Batchnorm_attributes::input_names::MOMENTUM);
            auto momentum_desc = tensors.at(MOMENTUM->second->get_uid())->get_raw_desc();

            _CUDNN_CHECK_CUDNN_ERROR(detail::set_attribute(batchnorm_operation.get_raw_desc(),
                                                           CUDNN_ATTR_OPERATION_NORM_FWD_EXP_AVG_FACTOR_DESC,
                                                           CUDNN_TYPE_BACKEND_DESCRIPTOR,
                                                           1,
                                                           &momentum_desc));

            // Set prev running mean and var
            CUDNN_FE_VALIDATE_AND_ASSIGN_INPUT_TENSOR(PREV_RUNNING_MEAN,
                                                      Batchnorm_attributes::input_names::PREV_RUNNING_MEAN);
            auto prev_mean_desc = tensors.at(PREV_RUNNING_MEAN->second->get_uid())->get_raw_desc();

            _CUDNN_CHECK_CUDNN_ERROR(detail::set_attribute(batchnorm_operation.get_raw_desc(),
                                                           CUDNN_ATTR_OPERATION_NORM_FWD_INPUT_RUNNING_MEAN_DESC,
                                                           CUDNN_TYPE_BACKEND_DESCRIPTOR,
                                                           1,
                                                           &prev_mean_desc));

            CUDNN_FE_VALIDATE_AND_ASSIGN_INPUT_TENSOR(PREV_RUNNING_VAR,
                                                      Batchnorm_attributes::input_names::PREV_RUNNING_VAR);
            auto prev_var_desc = tensors.at(PREV_RUNNING_VAR->second->get_uid())->get_raw_desc();

            _CUDNN_CHECK_CUDNN_ERROR(detail::set_attribute(batchnorm_operation.get_raw_desc(),
                                                           CUDNN_ATTR_OPERATION_NORM_FWD_INPUT_RUNNING_VAR_DESC,
                                                           CUDNN_TYPE_BACKEND_DESCRIPTOR,
                                                           1,
                                                           &prev_var_desc));

            // Set next running mean and var
            CUDNN_FE_VALIDATE_AND_ASSIGN_OUTPUT_TENSOR(NEXT_RUNNING_MEAN,
                                                       Batchnorm_attributes::output_names::NEXT_RUNNING_MEAN);
            auto next_mean_desc = tensors.at(NEXT_RUNNING_MEAN->second->get_uid())->get_raw_desc();

            _CUDNN_CHECK_CUDNN_ERROR(detail::set_attribute(batchnorm_operation.get_raw_desc(),
                                                           CUDNN_ATTR_OPERATION_NORM_FWD_OUTPUT_RUNNING_MEAN_DESC,
                                                           CUDNN_TYPE_BACKEND_DESCRIPTOR,
                                                           1,
                                                           &next_mean_desc));

            CUDNN_FE_VALIDATE_AND_ASSIGN_OUTPUT_TENSOR(NEXT_RUNNING_VAR,
                                                       Batchnorm_attributes::output_names::NEXT_RUNNING_VAR);
            auto next_var_desc = tensors.at(NEXT_RUNNING_VAR->second->get_uid())->get_raw_desc();

            _CUDNN_CHECK_CUDNN_ERROR(detail::set_attribute(batchnorm_operation.get_raw_desc(),
                                                           CUDNN_ATTR_OPERATION_NORM_FWD_OUTPUT_RUNNING_VAR_DESC,
                                                           CUDNN_TYPE_BACKEND_DESCRIPTOR,
                                                           1,
                                                           &next_var_desc));
        }

        // Set output tensor Y
        CUDNN_FE_VALIDATE_AND_ASSIGN_OUTPUT_TENSOR(Y, Batchnorm_attributes::output_names::Y);
        auto y_desc = tensors.at(Y->second->get_uid())->get_raw_desc();

        _CUDNN_CHECK_CUDNN_ERROR(detail::set_attribute(batchnorm_operation.get_raw_desc(),
                                                       CUDNN_ATTR_OPERATION_NORM_FWD_YDESC,
                                                       CUDNN_TYPE_BACKEND_DESCRIPTOR,
                                                       1,
                                                       &y_desc));

        // Set peer stat tensors if any
        if (!attributes.peer_stats.empty()) {
            std::vector<cudnnBackendDescriptor_t> peer_stat_descs;
            for (auto const& peer_stat : attributes.peer_stats) {
                peer_stat_descs.push_back(tensors.at(peer_stat->get_uid())->get_raw_desc());
            }

            _CUDNN_CHECK_CUDNN_ERROR(detail::set_attribute(batchnorm_operation.get_raw_desc(),
                                                           CUDNN_ATTR_OPERATION_NORM_FWD_PEER_STAT_DESCS,
                                                           CUDNN_TYPE_BACKEND_DESCRIPTOR,
                                                           peer_stat_descs.size(),
                                                           peer_stat_descs.data()));
        }

        _CUDNN_CHECK_CUDNN_ERROR(detail::finalize(batchnorm_operation.get_raw_desc()));

        operations.push_back(std::make_shared<Operation_v8>(std::move(batchnorm_operation)));

        auto const& non_virtual_uids = attributes.get_non_virtual_uids();
        uids_involved_in_operations.insert(non_virtual_uids.begin(), non_virtual_uids.end());
        return {error_code_t::OK, ""};
    }

#ifndef CUDNN_FRONTEND_SKIP_JSON_LIB
    virtual void
    serialize(json& j) const override final {
        j = attributes;
        j.update(R"( {"tag": "BATCHNORM"})"_json);
    }
#endif
};

}  // namespace graph

}  // namespace cudnn_frontend