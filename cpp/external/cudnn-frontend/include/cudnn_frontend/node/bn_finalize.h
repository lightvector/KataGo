#pragma once

#include "../graph_helpers.h"
#include "../node_interface.h"

namespace cudnn_frontend {

namespace graph {

class BatchNormFinalizeNode : public NodeCRTP<BatchNormFinalizeNode> {
   public:
    BN_finalize_attributes attributes;

    BatchNormFinalizeNode(BN_finalize_attributes&& attributes_, detail::Context const& context)
        : NodeCRTP(context), attributes(std::move(attributes_)) {}

    Type
    getType() override final {
        return Type::BN_FINALIZE;
    }

    error_t
    infer_properties_node() override final {
        CUDNN_FE_LOG_LABEL_ENDL("INFO:Inferencing properties for batchnorm finalize node " << attributes.name);

        attributes.fill_from_context(context);

        auto SUM                  = attributes.inputs[BN_finalize_attributes::input_names::SUM];
        auto const sum_tensor_dim = SUM->get_dim();

        // Set channel length tensors
        auto infer_per_channel_tensors = [&sum_tensor_dim](std::shared_ptr<Tensor_attributes>& T) {
            auto tensor_dim = T->get_dim();
            // Only infer dims and strides if user did not set them
            if (tensor_dim.empty()) {
                tensor_dim = sum_tensor_dim;
                T->set_dim(tensor_dim);
            }
            if (T->get_stride().empty()) {
                auto const& T_dim = T->get_dim();
                // Default to NHWC
                auto const& stride_order = detail::generate_NHWC_stride_order(T_dim.size());
                T->set_stride(detail::generate_stride(T_dim, stride_order));
            }
        };
        infer_per_channel_tensors(attributes.outputs[BN_finalize_attributes::output_names::EQ_BIAS]);
        infer_per_channel_tensors(attributes.outputs[BN_finalize_attributes::output_names::EQ_SCALE]);
        infer_per_channel_tensors(attributes.outputs[BN_finalize_attributes::output_names::MEAN]);
        infer_per_channel_tensors(attributes.outputs[BN_finalize_attributes::output_names::INV_VARIANCE]);
        infer_per_channel_tensors(attributes.outputs[BN_finalize_attributes::output_names::NEXT_RUNNING_MEAN]);
        infer_per_channel_tensors(attributes.outputs[BN_finalize_attributes::output_names::NEXT_RUNNING_VAR]);

        return {error_code_t::OK, ""};
    }

    error_t
    create_cudnn_operations(
        std::unordered_set<Tensor_attributes::uid_t>& uids_involved_in_operations,
        std::vector<std::shared_ptr<cudnn_frontend::Operation>>& operations,
        managed_backend_descriptor_t& raw_operations,
        std::unordered_map<int64_t, std::shared_ptr<cudnn_frontend::Tensor>>& tensors) const override final {
        CUDNN_FRONTEND_UNUSED(raw_operations);
        CUDNN_FE_LOG_LABEL("INFO: " << "Building BatchNormFinalizeNode operations " << attributes.name << " ");

        // Create operation by directly calling cuDNN backend API
        Operation_v8 bn_finalize_operation;

        _CUDNN_CHECK_CUDNN_ERROR(bn_finalize_operation.initialize_managed_backend_pointer(
            CUDNN_BACKEND_OPERATION_BN_FINALIZE_STATISTICS_DESCRIPTOR));

        // Set BN finalize mode
        cudnnBnFinalizeStatsMode_t bn_finalize_mode = CUDNN_BN_FINALIZE_STATISTICS_TRAINING;

        _CUDNN_CHECK_CUDNN_ERROR(detail::set_attribute(bn_finalize_operation.get_raw_desc(),
                                                       CUDNN_ATTR_OPERATION_BN_FINALIZE_STATS_MODE,
                                                       CUDNN_TYPE_BN_FINALIZE_STATS_MODE,
                                                       1,
                                                       &bn_finalize_mode));

        // Set compute type (math precision)
        cudnnDataType_t compute_type = CUDNN_DATA_FLOAT;

        _CUDNN_CHECK_CUDNN_ERROR(detail::set_attribute(bn_finalize_operation.get_raw_desc(),
                                                       CUDNN_ATTR_OPERATION_BN_FINALIZE_MATH_PREC,
                                                       CUDNN_TYPE_DATA_TYPE,
                                                       1,
                                                       &compute_type));

        // Set SUM input tensor
        CUDNN_FE_VALIDATE_AND_ASSIGN_INPUT_TENSOR(SUM, BN_finalize_attributes::input_names::SUM);
        auto sum_desc = tensors.at(SUM->second->get_uid())->get_raw_desc();

        _CUDNN_CHECK_CUDNN_ERROR(detail::set_attribute(bn_finalize_operation.get_raw_desc(),
                                                       CUDNN_ATTR_OPERATION_BN_FINALIZE_Y_SUM_DESC,
                                                       CUDNN_TYPE_BACKEND_DESCRIPTOR,
                                                       1,
                                                       &sum_desc));

        // Set SQ_SUM input tensor
        CUDNN_FE_VALIDATE_AND_ASSIGN_INPUT_TENSOR(SQ_SUM, BN_finalize_attributes::input_names::SQ_SUM);
        auto sq_sum_desc = tensors.at(SQ_SUM->second->get_uid())->get_raw_desc();

        _CUDNN_CHECK_CUDNN_ERROR(detail::set_attribute(bn_finalize_operation.get_raw_desc(),
                                                       CUDNN_ATTR_OPERATION_BN_FINALIZE_Y_SQ_SUM_DESC,
                                                       CUDNN_TYPE_BACKEND_DESCRIPTOR,
                                                       1,
                                                       &sq_sum_desc));

        // Set SCALE input tensor
        CUDNN_FE_VALIDATE_AND_ASSIGN_INPUT_TENSOR(SCALE, BN_finalize_attributes::input_names::SCALE);
        auto scale_desc = tensors.at(SCALE->second->get_uid())->get_raw_desc();

        _CUDNN_CHECK_CUDNN_ERROR(detail::set_attribute(bn_finalize_operation.get_raw_desc(),
                                                       CUDNN_ATTR_OPERATION_BN_FINALIZE_SCALE_DESC,
                                                       CUDNN_TYPE_BACKEND_DESCRIPTOR,
                                                       1,
                                                       &scale_desc));

        // Set BIAS input tensor
        CUDNN_FE_VALIDATE_AND_ASSIGN_INPUT_TENSOR(BIAS, BN_finalize_attributes::input_names::BIAS);
        auto bias_desc = tensors.at(BIAS->second->get_uid())->get_raw_desc();

        _CUDNN_CHECK_CUDNN_ERROR(detail::set_attribute(bn_finalize_operation.get_raw_desc(),
                                                       CUDNN_ATTR_OPERATION_BN_FINALIZE_BIAS_DESC,
                                                       CUDNN_TYPE_BACKEND_DESCRIPTOR,
                                                       1,
                                                       &bias_desc));

        // Set EQ_SCALE output tensor
        CUDNN_FE_VALIDATE_AND_ASSIGN_OUTPUT_TENSOR(EQ_SCALE, BN_finalize_attributes::output_names::EQ_SCALE);
        auto eq_scale_desc = tensors.at(EQ_SCALE->second->get_uid())->get_raw_desc();

        _CUDNN_CHECK_CUDNN_ERROR(detail::set_attribute(bn_finalize_operation.get_raw_desc(),
                                                       CUDNN_ATTR_OPERATION_BN_FINALIZE_EQ_SCALE_DESC,
                                                       CUDNN_TYPE_BACKEND_DESCRIPTOR,
                                                       1,
                                                       &eq_scale_desc));

        // Set EQ_BIAS output tensor
        CUDNN_FE_VALIDATE_AND_ASSIGN_OUTPUT_TENSOR(EQ_BIAS, BN_finalize_attributes::output_names::EQ_BIAS);
        auto eq_bias_desc = tensors.at(EQ_BIAS->second->get_uid())->get_raw_desc();

        _CUDNN_CHECK_CUDNN_ERROR(detail::set_attribute(bn_finalize_operation.get_raw_desc(),
                                                       CUDNN_ATTR_OPERATION_BN_FINALIZE_EQ_BIAS_DESC,
                                                       CUDNN_TYPE_BACKEND_DESCRIPTOR,
                                                       1,
                                                       &eq_bias_desc));

        // Set PREV_RUNNING_MEAN input tensor
        CUDNN_FE_VALIDATE_AND_ASSIGN_INPUT_TENSOR(PREV_RUNNING_MEAN,
                                                  BN_finalize_attributes::input_names::PREV_RUNNING_MEAN);
        auto prev_running_mean_desc = tensors.at(PREV_RUNNING_MEAN->second->get_uid())->get_raw_desc();

        _CUDNN_CHECK_CUDNN_ERROR(detail::set_attribute(bn_finalize_operation.get_raw_desc(),
                                                       CUDNN_ATTR_OPERATION_BN_FINALIZE_PREV_RUNNING_MEAN_DESC,
                                                       CUDNN_TYPE_BACKEND_DESCRIPTOR,
                                                       1,
                                                       &prev_running_mean_desc));

        // Set PREV_RUNNING_VAR input tensor
        CUDNN_FE_VALIDATE_AND_ASSIGN_INPUT_TENSOR(PREV_RUNNING_VAR,
                                                  BN_finalize_attributes::input_names::PREV_RUNNING_VAR);
        auto prev_running_var_desc = tensors.at(PREV_RUNNING_VAR->second->get_uid())->get_raw_desc();

        _CUDNN_CHECK_CUDNN_ERROR(detail::set_attribute(bn_finalize_operation.get_raw_desc(),
                                                       CUDNN_ATTR_OPERATION_BN_FINALIZE_PREV_RUNNING_VAR_DESC,
                                                       CUDNN_TYPE_BACKEND_DESCRIPTOR,
                                                       1,
                                                       &prev_running_var_desc));

        // Set NEXT_RUNNING_MEAN output tensor
        CUDNN_FE_VALIDATE_AND_ASSIGN_OUTPUT_TENSOR(NEXT_RUNNING_MEAN,
                                                   BN_finalize_attributes::output_names::NEXT_RUNNING_MEAN);
        auto next_running_mean_desc = tensors.at(NEXT_RUNNING_MEAN->second->get_uid())->get_raw_desc();

        _CUDNN_CHECK_CUDNN_ERROR(detail::set_attribute(bn_finalize_operation.get_raw_desc(),
                                                       CUDNN_ATTR_OPERATION_BN_FINALIZE_UPDATED_RUNNING_MEAN_DESC,
                                                       CUDNN_TYPE_BACKEND_DESCRIPTOR,
                                                       1,
                                                       &next_running_mean_desc));

        // Set NEXT_RUNNING_VAR output tensor
        CUDNN_FE_VALIDATE_AND_ASSIGN_OUTPUT_TENSOR(NEXT_RUNNING_VAR,
                                                   BN_finalize_attributes::output_names::NEXT_RUNNING_VAR);
        auto next_running_var_desc = tensors.at(NEXT_RUNNING_VAR->second->get_uid())->get_raw_desc();

        _CUDNN_CHECK_CUDNN_ERROR(detail::set_attribute(bn_finalize_operation.get_raw_desc(),
                                                       CUDNN_ATTR_OPERATION_BN_FINALIZE_UPDATED_RUNNING_VAR_DESC,
                                                       CUDNN_TYPE_BACKEND_DESCRIPTOR,
                                                       1,
                                                       &next_running_var_desc));

        // Set MEAN output tensor (saved mean)
        CUDNN_FE_VALIDATE_AND_ASSIGN_OUTPUT_TENSOR(MEAN, BN_finalize_attributes::output_names::MEAN);
        auto mean_desc = tensors.at(MEAN->second->get_uid())->get_raw_desc();

        _CUDNN_CHECK_CUDNN_ERROR(detail::set_attribute(bn_finalize_operation.get_raw_desc(),
                                                       CUDNN_ATTR_OPERATION_BN_FINALIZE_SAVED_MEAN_DESC,
                                                       CUDNN_TYPE_BACKEND_DESCRIPTOR,
                                                       1,
                                                       &mean_desc));

        // Set INV_VARIANCE output tensor (saved inv std)
        CUDNN_FE_VALIDATE_AND_ASSIGN_OUTPUT_TENSOR(INV_VARIANCE, BN_finalize_attributes::output_names::INV_VARIANCE);
        auto inv_var_desc = tensors.at(INV_VARIANCE->second->get_uid())->get_raw_desc();

        _CUDNN_CHECK_CUDNN_ERROR(detail::set_attribute(bn_finalize_operation.get_raw_desc(),
                                                       CUDNN_ATTR_OPERATION_BN_FINALIZE_SAVED_INV_STD_DESC,
                                                       CUDNN_TYPE_BACKEND_DESCRIPTOR,
                                                       1,
                                                       &inv_var_desc));

        // Set EPSILON tensor
        CUDNN_FE_VALIDATE_AND_ASSIGN_INPUT_TENSOR(EPSILON, BN_finalize_attributes::input_names::EPSILON);
        auto epsilon_desc = tensors.at(EPSILON->second->get_uid())->get_raw_desc();

        _CUDNN_CHECK_CUDNN_ERROR(detail::set_attribute(bn_finalize_operation.get_raw_desc(),
                                                       CUDNN_ATTR_OPERATION_BN_FINALIZE_EPSILON_DESC,
                                                       CUDNN_TYPE_BACKEND_DESCRIPTOR,
                                                       1,
                                                       &epsilon_desc));

        // Set MOMENTUM tensor (exp average factor)
        CUDNN_FE_VALIDATE_AND_ASSIGN_INPUT_TENSOR(MOMENTUM, BN_finalize_attributes::input_names::MOMENTUM);
        auto momentum_desc = tensors.at(MOMENTUM->second->get_uid())->get_raw_desc();

        _CUDNN_CHECK_CUDNN_ERROR(detail::set_attribute(bn_finalize_operation.get_raw_desc(),
                                                       CUDNN_ATTR_OPERATION_BN_FINALIZE_EXP_AVERATE_FACTOR_DESC,
                                                       CUDNN_TYPE_BACKEND_DESCRIPTOR,
                                                       1,
                                                       &momentum_desc));

        // Set ACCUM_COUNT tensor
        CUDNN_FE_VALIDATE_AND_ASSIGN_INPUT_TENSOR(ACCUM_COUNT, BN_finalize_attributes::input_names::ACCUM_COUNT);
        auto accum_count_desc = tensors.at(ACCUM_COUNT->second->get_uid())->get_raw_desc();

        _CUDNN_CHECK_CUDNN_ERROR(detail::set_attribute(bn_finalize_operation.get_raw_desc(),
                                                       CUDNN_ATTR_OPERATION_BN_FINALIZE_ACCUM_COUNT_DESC,
                                                       CUDNN_TYPE_BACKEND_DESCRIPTOR,
                                                       1,
                                                       &accum_count_desc));

        _CUDNN_CHECK_CUDNN_ERROR(detail::finalize(bn_finalize_operation.get_raw_desc()));

        operations.push_back(std::make_shared<Operation_v8>(std::move(bn_finalize_operation)));

        auto const& non_virtual_uids = attributes.get_non_virtual_uids();
        uids_involved_in_operations.insert(non_virtual_uids.begin(), non_virtual_uids.end());
        return {error_code_t::OK, ""};
    }

#ifndef CUDNN_FRONTEND_SKIP_JSON_LIB
    virtual void
    serialize(json& j) const override final {
        j = attributes;
        j.update(R"( {"tag": "BN_FINALIZE"})"_json);
    }
#endif
};

}  // namespace graph

}  // namespace cudnn_frontend