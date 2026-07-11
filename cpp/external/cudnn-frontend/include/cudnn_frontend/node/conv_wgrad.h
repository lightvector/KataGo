#pragma once

#include "../graph_helpers.h"
#include "../node_interface.h"

namespace cudnn_frontend::graph {

class WgradNode : public NodeCRTP<WgradNode> {
   public:
    Conv_wgrad_attributes attributes;

    WgradNode(Conv_wgrad_attributes&& attributes_, detail::Context const& context)
        : NodeCRTP(context), attributes(std::move(attributes_)) {}

    Type
    getType() override final {
        return Type::WGRAD;
    }

    error_t
    pre_validate_node() const override final {
        CUDNN_FE_LOG_LABEL_ENDL("INFO: Validating Node Type::WGRAD " << attributes.name);

        RETURN_CUDNN_FRONTEND_ERROR_IF(
            attributes.get_pre_padding().empty(), error_code_t::ATTRIBUTE_NOT_SET, "Pre padding not set.");
        RETURN_CUDNN_FRONTEND_ERROR_IF(
            attributes.get_post_padding().empty(), error_code_t::ATTRIBUTE_NOT_SET, "Post padding not set.");
        RETURN_CUDNN_FRONTEND_ERROR_IF(
            attributes.get_stride().empty(), error_code_t::ATTRIBUTE_NOT_SET, "Conv strides not set.");
        RETURN_CUDNN_FRONTEND_ERROR_IF(
            attributes.get_dilation().empty(), error_code_t::ATTRIBUTE_NOT_SET, "Conv dilation not set.");

        return {error_code_t::OK, ""};
    }

    error_t
    infer_properties_node() override final {
        CUDNN_FE_LOG_LABEL_ENDL("INFO:     Inferrencing properties for conv node " << attributes.name);

        attributes.fill_from_context(context);

        // TODO: Only inferrencing from (X, DY) -> DW works today.
        auto X  = attributes.inputs[Conv_wgrad_attributes::input_names::X];
        auto DW = attributes.outputs[Conv_wgrad_attributes::output_names::DW];
        auto DY = attributes.inputs[Conv_wgrad_attributes::input_names::DY];

        auto const x_tensor_dim  = X->get_dim();
        auto const dy_tensor_dim = DY->get_dim();
        auto dw_tensor_dim       = DW->get_dim();

        // No dim inferencing as inverse mapping from DY, X to DX is not unique.
        // Only infer strides if user did not set them
        if (DW->get_stride().empty()) {
            auto const& DW_dim = DW->get_dim();
            // Default to NHWC
            auto const& stride_order = detail::generate_NHWC_stride_order(DW_dim.size());
            DW->set_stride(detail::generate_stride(DW_dim, stride_order));
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
        CUDNN_FE_LOG_LABEL("INFO: Building WgradNode operations " << attributes.name << " ");

        // Create wgrad descriptor by directly calling cuDNN backend API
        ConvDesc_v8 wgrad_descriptor;
        int64_t const spatial_dim_count = attributes.get_pre_padding().size();

        _CUDNN_CHECK_CUDNN_ERROR(
            wgrad_descriptor.initialize_managed_backend_pointer(CUDNN_BACKEND_CONVOLUTION_DESCRIPTOR));

        cudnnDataType_t cudnn_data_type;

        _CUDNN_CHECK_CUDNN_ERROR(detail::convert_to_cudnn_type(attributes.compute_data_type, cudnn_data_type));

        _CUDNN_CHECK_CUDNN_ERROR(detail::set_attribute(wgrad_descriptor.get_raw_desc(),
                                                       CUDNN_ATTR_CONVOLUTION_COMP_TYPE,
                                                       CUDNN_TYPE_DATA_TYPE,
                                                       1,
                                                       &cudnn_data_type));

        cudnnConvolutionMode_t mode = detail::convert_to_cudnn_type(attributes.math_mode);

        _CUDNN_CHECK_CUDNN_ERROR(detail::set_attribute(
            wgrad_descriptor.get_raw_desc(), CUDNN_ATTR_CONVOLUTION_CONV_MODE, CUDNN_TYPE_CONVOLUTION_MODE, 1, &mode));

        _CUDNN_CHECK_CUDNN_ERROR(detail::set_attribute(wgrad_descriptor.get_raw_desc(),
                                                       CUDNN_ATTR_CONVOLUTION_SPATIAL_DIMS,
                                                       CUDNN_TYPE_INT64,
                                                       1,
                                                       &spatial_dim_count));

        _CUDNN_CHECK_CUDNN_ERROR(detail::set_attribute(wgrad_descriptor.get_raw_desc(),
                                                       CUDNN_ATTR_CONVOLUTION_PRE_PADDINGS,
                                                       CUDNN_TYPE_INT64,
                                                       spatial_dim_count,
                                                       attributes.get_pre_padding().data()));

        _CUDNN_CHECK_CUDNN_ERROR(detail::set_attribute(wgrad_descriptor.get_raw_desc(),
                                                       CUDNN_ATTR_CONVOLUTION_POST_PADDINGS,
                                                       CUDNN_TYPE_INT64,
                                                       spatial_dim_count,
                                                       attributes.get_post_padding().data()));

        _CUDNN_CHECK_CUDNN_ERROR(detail::set_attribute(wgrad_descriptor.get_raw_desc(),
                                                       CUDNN_ATTR_CONVOLUTION_DILATIONS,
                                                       CUDNN_TYPE_INT64,
                                                       spatial_dim_count,
                                                       attributes.get_dilation().data()));

        _CUDNN_CHECK_CUDNN_ERROR(detail::set_attribute(wgrad_descriptor.get_raw_desc(),
                                                       CUDNN_ATTR_CONVOLUTION_FILTER_STRIDES,
                                                       CUDNN_TYPE_INT64,
                                                       spatial_dim_count,
                                                       attributes.get_stride().data()));

        _CUDNN_CHECK_CUDNN_ERROR(detail::finalize(wgrad_descriptor.get_raw_desc()));
        CUDNN_FE_LOG_LABEL_ENDL(wgrad_descriptor);

        // Create operation by directly calling cuDNN backend API
        Operation_v8 wgrad_operation;

        _CUDNN_CHECK_CUDNN_ERROR(wgrad_operation.initialize_managed_backend_pointer(
            CUDNN_BACKEND_OPERATION_CONVOLUTION_BACKWARD_FILTER_DESCRIPTOR));

        CUDNN_FE_VALIDATE_AND_ASSIGN_INPUT_TENSOR(X, Conv_wgrad_attributes::input_names::X);
        auto x_desc = tensors.at(X->second->get_uid())->get_raw_desc();

        _CUDNN_CHECK_CUDNN_ERROR(detail::set_attribute(wgrad_operation.get_raw_desc(),
                                                       CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_X,
                                                       CUDNN_TYPE_BACKEND_DESCRIPTOR,
                                                       1,
                                                       &x_desc));

        CUDNN_FE_VALIDATE_AND_ASSIGN_INPUT_TENSOR(DY, Conv_wgrad_attributes::input_names::DY);
        auto dy_desc = tensors.at(DY->second->get_uid())->get_raw_desc();

        _CUDNN_CHECK_CUDNN_ERROR(detail::set_attribute(wgrad_operation.get_raw_desc(),
                                                       CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_DY,
                                                       CUDNN_TYPE_BACKEND_DESCRIPTOR,
                                                       1,
                                                       &dy_desc));

        CUDNN_FE_VALIDATE_AND_ASSIGN_OUTPUT_TENSOR(DW, Conv_wgrad_attributes::output_names::DW);
        auto dw_desc = tensors.at(DW->second->get_uid())->get_raw_desc();

        _CUDNN_CHECK_CUDNN_ERROR(detail::set_attribute(wgrad_operation.get_raw_desc(),
                                                       CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_DW,
                                                       CUDNN_TYPE_BACKEND_DESCRIPTOR,
                                                       1,
                                                       &dw_desc));

        auto conv_desc_ptr = wgrad_descriptor.get_raw_desc();

        _CUDNN_CHECK_CUDNN_ERROR(detail::set_attribute(wgrad_operation.get_raw_desc(),
                                                       CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_CONV_DESC,
                                                       CUDNN_TYPE_BACKEND_DESCRIPTOR,
                                                       1,
                                                       &conv_desc_ptr));

        float alpha = 1.0f;
        float beta  = 0.0f;

        _CUDNN_CHECK_CUDNN_ERROR(detail::set_attribute(wgrad_operation.get_raw_desc(),
                                                       CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_ALPHA,
                                                       CUDNN_TYPE_FLOAT,
                                                       1,
                                                       &alpha));

        _CUDNN_CHECK_CUDNN_ERROR(detail::set_attribute(wgrad_operation.get_raw_desc(),
                                                       CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_BETA,
                                                       CUDNN_TYPE_FLOAT,
                                                       1,
                                                       &beta));

        _CUDNN_CHECK_CUDNN_ERROR(detail::finalize(wgrad_operation.get_raw_desc()));

        operations.push_back(std::make_shared<Operation_v8>(std::move(wgrad_operation)));

        auto const& non_virtual_uids = attributes.get_non_virtual_uids();
        uids_involved_in_operations.insert(non_virtual_uids.begin(), non_virtual_uids.end());
        return {error_code_t::OK, ""};
    }

#ifndef CUDNN_FRONTEND_SKIP_JSON_LIB
    virtual void
    serialize(json& j) const override final {
        j = attributes;
        j.update(R"( {"tag": "CONV_WGRAD"})"_json);
    }
#endif
};

}  // namespace cudnn_frontend::graph