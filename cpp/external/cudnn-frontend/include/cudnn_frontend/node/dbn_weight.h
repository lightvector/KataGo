#pragma once

#include "../graph_helpers.h"
#include "../node_interface.h"

namespace cudnn_frontend {

namespace graph {

class DBNWeightNode : public NodeCRTP<DBNWeightNode> {
   public:
    DBN_weight_attributes attributes;

    DBNWeightNode(DBN_weight_attributes&& attributes_, detail::Context const& context)
        : NodeCRTP(context), attributes(std::move(attributes_)) {}

    Type
    getType() override final {
        return Type::DBN_WEIGHT;
    }

    error_t
    infer_properties_node() override final {
        CUDNN_FE_LOG_LABEL_ENDL("INFO: Inferencing properties for batchnorm finalize node " << attributes.name);

        attributes.fill_from_context(context);

        // TODO: Only inferencing from DY works today.
        auto DY                  = attributes.inputs[DBN_weight_attributes::input_names::DY];
        auto const dy_tensor_dim = DY->get_dim();

        auto X            = attributes.inputs[DBN_weight_attributes::input_names::X];
        auto x_tensor_dim = X->get_dim();
        // Only infer dims and strides if user did not set them
        if (x_tensor_dim.empty()) {
            x_tensor_dim.resize(dy_tensor_dim.size());
            X->set_dim(dy_tensor_dim);
        }
        if (X->get_stride().empty()) {
            auto const& X_dim = X->get_dim();
            // Default to NHWC
            auto const& stride_order = detail::generate_NHWC_stride_order(X_dim.size());
            X->set_stride(detail::generate_stride(X_dim, stride_order));
        }

        // Set channel length tensors
        auto infer_per_channel_tensors = [&dy_tensor_dim](std::shared_ptr<Tensor_attributes> const& T) {
            auto tensor_dim = T->get_dim();
            // Only infer dims and strides if user did not set them
            if (T->get_dim().empty()) {
                tensor_dim.resize(dy_tensor_dim.size(), 1);
                tensor_dim[1] = dy_tensor_dim[1];
                T->set_dim(tensor_dim);
            }
            if (T->get_stride().empty()) {
                auto const& T_dim = T->get_dim();
                // Default to NHWC
                auto const& stride_order = detail::generate_NHWC_stride_order(T_dim.size());
                T->set_stride(detail::generate_stride(T_dim, stride_order));
            }
        };
        infer_per_channel_tensors(attributes.outputs[DBN_weight_attributes::output_names::DBIAS]);
        infer_per_channel_tensors(attributes.outputs[DBN_weight_attributes::output_names::DSCALE]);
        infer_per_channel_tensors(attributes.outputs[DBN_weight_attributes::output_names::EQ_BIAS]);
        infer_per_channel_tensors(attributes.outputs[DBN_weight_attributes::output_names::EQ_SCALE_DY]);
        infer_per_channel_tensors(attributes.outputs[DBN_weight_attributes::output_names::EQ_SCALE_X]);

        return {error_code_t::OK, ""};
    }

    error_t
    create_cudnn_operations(
        std::unordered_set<Tensor_attributes::uid_t>& uids_involved_in_operations,
        std::vector<std::shared_ptr<cudnn_frontend::Operation>>& operations,
        managed_backend_descriptor_t& raw_operations,
        std::unordered_map<int64_t, std::shared_ptr<cudnn_frontend::Tensor>>& tensors) const override final {
        CUDNN_FRONTEND_UNUSED(raw_operations);
        CUDNN_FE_LOG_LABEL("INFO: " << "Building DBNWeightNode operations " << attributes.name << " ");

        // Create operation by directly calling cuDNN backend API
        Operation_v8 bn_bwd_weight_operation;

        _CUDNN_CHECK_CUDNN_ERROR(bn_bwd_weight_operation.initialize_managed_backend_pointer(
            CUDNN_BACKEND_OPERATION_BN_BWD_WEIGHTS_DESCRIPTOR));

        // Set compute type (math precision)
        cudnnDataType_t compute_type = CUDNN_DATA_FLOAT;

        _CUDNN_CHECK_CUDNN_ERROR(detail::set_attribute(bn_bwd_weight_operation.get_raw_desc(),
                                                       CUDNN_ATTR_OPERATION_BN_BWD_WEIGHTS_MATH_PREC,
                                                       CUDNN_TYPE_DATA_TYPE,
                                                       1,
                                                       &compute_type));

        // Set input tensor X
        CUDNN_FE_VALIDATE_AND_ASSIGN_INPUT_TENSOR(X, DBN_weight_attributes::input_names::X);
        auto x_desc = tensors.at(X->second->get_uid())->get_raw_desc();

        _CUDNN_CHECK_CUDNN_ERROR(detail::set_attribute(bn_bwd_weight_operation.get_raw_desc(),
                                                       CUDNN_ATTR_OPERATION_BN_BWD_WEIGHTS_X_DESC,
                                                       CUDNN_TYPE_BACKEND_DESCRIPTOR,
                                                       1,
                                                       &x_desc));

        // Set DY tensor
        CUDNN_FE_VALIDATE_AND_ASSIGN_INPUT_TENSOR(DY, DBN_weight_attributes::input_names::DY);
        auto dy_desc = tensors.at(DY->second->get_uid())->get_raw_desc();

        _CUDNN_CHECK_CUDNN_ERROR(detail::set_attribute(bn_bwd_weight_operation.get_raw_desc(),
                                                       CUDNN_ATTR_OPERATION_BN_BWD_WEIGHTS_DY_DESC,
                                                       CUDNN_TYPE_BACKEND_DESCRIPTOR,
                                                       1,
                                                       &dy_desc));

        // Set mean tensor
        CUDNN_FE_VALIDATE_AND_ASSIGN_INPUT_TENSOR(MEAN, DBN_weight_attributes::input_names::MEAN);
        auto mean_desc = tensors.at(MEAN->second->get_uid())->get_raw_desc();

        _CUDNN_CHECK_CUDNN_ERROR(detail::set_attribute(bn_bwd_weight_operation.get_raw_desc(),
                                                       CUDNN_ATTR_OPERATION_BN_BWD_WEIGHTS_MEAN_DESC,
                                                       CUDNN_TYPE_BACKEND_DESCRIPTOR,
                                                       1,
                                                       &mean_desc));

        // Set inv_variance tensor
        CUDNN_FE_VALIDATE_AND_ASSIGN_INPUT_TENSOR(INV_VARIANCE, DBN_weight_attributes::input_names::INV_VARIANCE);
        auto inv_var_desc = tensors.at(INV_VARIANCE->second->get_uid())->get_raw_desc();

        _CUDNN_CHECK_CUDNN_ERROR(detail::set_attribute(bn_bwd_weight_operation.get_raw_desc(),
                                                       CUDNN_ATTR_OPERATION_BN_BWD_WEIGHTS_INVSTD_DESC,
                                                       CUDNN_TYPE_BACKEND_DESCRIPTOR,
                                                       1,
                                                       &inv_var_desc));

        // Set scale tensor
        CUDNN_FE_VALIDATE_AND_ASSIGN_INPUT_TENSOR(SCALE, DBN_weight_attributes::input_names::SCALE);
        auto scale_desc = tensors.at(SCALE->second->get_uid())->get_raw_desc();

        _CUDNN_CHECK_CUDNN_ERROR(detail::set_attribute(bn_bwd_weight_operation.get_raw_desc(),
                                                       CUDNN_ATTR_OPERATION_BN_BWD_WEIGHTS_BN_SCALE_DESC,
                                                       CUDNN_TYPE_BACKEND_DESCRIPTOR,
                                                       1,
                                                       &scale_desc));

        // Set DSCALE output tensor
        CUDNN_FE_VALIDATE_AND_ASSIGN_OUTPUT_TENSOR(DSCALE, DBN_weight_attributes::output_names::DSCALE);
        auto dscale_desc = tensors.at(DSCALE->second->get_uid())->get_raw_desc();

        _CUDNN_CHECK_CUDNN_ERROR(detail::set_attribute(bn_bwd_weight_operation.get_raw_desc(),
                                                       CUDNN_ATTR_OPERATION_BN_BWD_WEIGHTS_DBN_SCALE_DESC,
                                                       CUDNN_TYPE_BACKEND_DESCRIPTOR,
                                                       1,
                                                       &dscale_desc));

        // Set DBIAS output tensor
        CUDNN_FE_VALIDATE_AND_ASSIGN_OUTPUT_TENSOR(DBIAS, DBN_weight_attributes::output_names::DBIAS);
        auto dbias_desc = tensors.at(DBIAS->second->get_uid())->get_raw_desc();

        _CUDNN_CHECK_CUDNN_ERROR(detail::set_attribute(bn_bwd_weight_operation.get_raw_desc(),
                                                       CUDNN_ATTR_OPERATION_BN_BWD_WEIGHTS_DBN_BIAS_DESC,
                                                       CUDNN_TYPE_BACKEND_DESCRIPTOR,
                                                       1,
                                                       &dbias_desc));

        // Set EQ_SCALE_DY output tensor
        CUDNN_FE_VALIDATE_AND_ASSIGN_OUTPUT_TENSOR(EQ_SCALE_DY, DBN_weight_attributes::output_names::EQ_SCALE_DY);
        auto eq_scale_dy_desc = tensors.at(EQ_SCALE_DY->second->get_uid())->get_raw_desc();

        _CUDNN_CHECK_CUDNN_ERROR(detail::set_attribute(bn_bwd_weight_operation.get_raw_desc(),
                                                       CUDNN_ATTR_OPERATION_BN_BWD_WEIGHTS_EQ_DY_SCALE_DESC,
                                                       CUDNN_TYPE_BACKEND_DESCRIPTOR,
                                                       1,
                                                       &eq_scale_dy_desc));

        // Set EQ_SCALE_X output tensor
        CUDNN_FE_VALIDATE_AND_ASSIGN_OUTPUT_TENSOR(EQ_SCALE_X, DBN_weight_attributes::output_names::EQ_SCALE_X);
        auto eq_scale_x_desc = tensors.at(EQ_SCALE_X->second->get_uid())->get_raw_desc();

        _CUDNN_CHECK_CUDNN_ERROR(detail::set_attribute(bn_bwd_weight_operation.get_raw_desc(),
                                                       CUDNN_ATTR_OPERATION_BN_BWD_WEIGHTS_EQ_X_SCALE_DESC,
                                                       CUDNN_TYPE_BACKEND_DESCRIPTOR,
                                                       1,
                                                       &eq_scale_x_desc));

        // Set EQ_BIAS output tensor
        CUDNN_FE_VALIDATE_AND_ASSIGN_OUTPUT_TENSOR(EQ_BIAS, DBN_weight_attributes::output_names::EQ_BIAS);
        auto eq_bias_desc = tensors.at(EQ_BIAS->second->get_uid())->get_raw_desc();

        _CUDNN_CHECK_CUDNN_ERROR(detail::set_attribute(bn_bwd_weight_operation.get_raw_desc(),
                                                       CUDNN_ATTR_OPERATION_BN_BWD_WEIGHTS_EQ_BIAS,
                                                       CUDNN_TYPE_BACKEND_DESCRIPTOR,
                                                       1,
                                                       &eq_bias_desc));

        _CUDNN_CHECK_CUDNN_ERROR(detail::finalize(bn_bwd_weight_operation.get_raw_desc()));

        operations.push_back(std::make_shared<Operation_v8>(std::move(bn_bwd_weight_operation)));

        auto const& non_virtual_uids = attributes.get_non_virtual_uids();
        uids_involved_in_operations.insert(non_virtual_uids.begin(), non_virtual_uids.end());
        return {error_code_t::OK, ""};
    }

#ifndef CUDNN_FRONTEND_SKIP_JSON_LIB
    virtual void
    serialize(json& j) const override final {
        j = attributes;
        j.update(R"( {"tag": "DBN_WEIGHT"})"_json);
    }
#endif
};

}  // namespace graph

}  // namespace cudnn_frontend