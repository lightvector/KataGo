#pragma once

#include "../graph_helpers.h"
#include "../node_interface.h"

namespace cudnn_frontend {

namespace graph {
class BatchnormInferenceNode : public NodeCRTP<BatchnormInferenceNode> {
   public:
    Batchnorm_inference_attributes attributes;

    BatchnormInferenceNode(Batchnorm_inference_attributes&& attributes_, detail::Context const& context)
        : NodeCRTP(context), attributes(std::move(attributes_)) {}

    Type
    getType() override final {
        return Type::BATCHNORM_INFERENCE;
    }

    error_t
    infer_properties_node() override final {
        CUDNN_FE_LOG_LABEL_ENDL("INFO: Inferencing properties for batchnorm inference node " << attributes.name);

        attributes.fill_from_context(context);

        auto X = attributes.inputs[Batchnorm_inference_attributes::input_names::X];
        auto Y = attributes.outputs[Batchnorm_inference_attributes::output_names::Y];
        // Only infer dims and strides if user did not set them
        if (Y->get_dim().empty()) {
            Y->set_dim(X->get_dim());
        }
        if (Y->get_stride().empty()) {
            Y->set_stride(X->get_stride());
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
        CUDNN_FE_LOG_LABEL("INFO: Building BatchnormInferenceNode operations " << attributes.name << " ");

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

        // Set forward phase to INFERENCE
        cudnnBackendNormFwdPhase_t cudnn_norm_fwd_phase;

        _CUDNN_CHECK_CUDNN_ERROR(detail::convert_to_cudnn_type(NormFwdPhase_t::INFERENCE, cudnn_norm_fwd_phase));

        _CUDNN_CHECK_CUDNN_ERROR(detail::set_attribute(batchnorm_operation.get_raw_desc(),
                                                       CUDNN_ATTR_OPERATION_NORM_FWD_PHASE,
                                                       CUDNN_TYPE_NORM_FWD_PHASE,
                                                       1,
                                                       &cudnn_norm_fwd_phase));

        // Set input tensor X
        CUDNN_FE_VALIDATE_AND_ASSIGN_INPUT_TENSOR(X, Batchnorm_inference_attributes::input_names::X);
        auto x_desc = tensors.at(X->second->get_uid())->get_raw_desc();

        _CUDNN_CHECK_CUDNN_ERROR(detail::set_attribute(batchnorm_operation.get_raw_desc(),
                                                       CUDNN_ATTR_OPERATION_NORM_FWD_XDESC,
                                                       CUDNN_TYPE_BACKEND_DESCRIPTOR,
                                                       1,
                                                       &x_desc));

        // Set mean and inv_variance (as inputs for inference)
        CUDNN_FE_VALIDATE_AND_ASSIGN_INPUT_TENSOR(MEAN, Batchnorm_inference_attributes::input_names::MEAN);
        auto mean_desc = tensors.at(MEAN->second->get_uid())->get_raw_desc();

        _CUDNN_CHECK_CUDNN_ERROR(detail::set_attribute(batchnorm_operation.get_raw_desc(),
                                                       CUDNN_ATTR_OPERATION_NORM_FWD_MEAN_DESC,
                                                       CUDNN_TYPE_BACKEND_DESCRIPTOR,
                                                       1,
                                                       &mean_desc));

        CUDNN_FE_VALIDATE_AND_ASSIGN_INPUT_TENSOR(INV_VARIANCE,
                                                  Batchnorm_inference_attributes::input_names::INV_VARIANCE);
        auto inv_var_desc = tensors.at(INV_VARIANCE->second->get_uid())->get_raw_desc();

        _CUDNN_CHECK_CUDNN_ERROR(detail::set_attribute(batchnorm_operation.get_raw_desc(),
                                                       CUDNN_ATTR_OPERATION_NORM_FWD_INV_VARIANCE_DESC,
                                                       CUDNN_TYPE_BACKEND_DESCRIPTOR,
                                                       1,
                                                       &inv_var_desc));

        // Set scale and bias tensors
        CUDNN_FE_VALIDATE_AND_ASSIGN_INPUT_TENSOR(SCALE, Batchnorm_inference_attributes::input_names::SCALE);
        auto scale_desc = tensors.at(SCALE->second->get_uid())->get_raw_desc();

        _CUDNN_CHECK_CUDNN_ERROR(detail::set_attribute(batchnorm_operation.get_raw_desc(),
                                                       CUDNN_ATTR_OPERATION_NORM_FWD_SCALE_DESC,
                                                       CUDNN_TYPE_BACKEND_DESCRIPTOR,
                                                       1,
                                                       &scale_desc));

        CUDNN_FE_VALIDATE_AND_ASSIGN_INPUT_TENSOR(BIAS, Batchnorm_inference_attributes::input_names::BIAS);
        auto bias_desc = tensors.at(BIAS->second->get_uid())->get_raw_desc();

        _CUDNN_CHECK_CUDNN_ERROR(detail::set_attribute(batchnorm_operation.get_raw_desc(),
                                                       CUDNN_ATTR_OPERATION_NORM_FWD_BIAS_DESC,
                                                       CUDNN_TYPE_BACKEND_DESCRIPTOR,
                                                       1,
                                                       &bias_desc));

        // Set output tensor Y
        CUDNN_FE_VALIDATE_AND_ASSIGN_OUTPUT_TENSOR(Y, Batchnorm_inference_attributes::output_names::Y);
        auto y_desc = tensors.at(Y->second->get_uid())->get_raw_desc();

        _CUDNN_CHECK_CUDNN_ERROR(detail::set_attribute(batchnorm_operation.get_raw_desc(),
                                                       CUDNN_ATTR_OPERATION_NORM_FWD_YDESC,
                                                       CUDNN_TYPE_BACKEND_DESCRIPTOR,
                                                       1,
                                                       &y_desc));

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
        j.update(R"( {"tag": "BATCHNORM_INFERENCE"})"_json);
    }
#endif
};

}  // namespace graph

}  // namespace cudnn_frontend