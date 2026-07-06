#pragma once

#include "../graph_helpers.h"
#include "../node_interface.h"

namespace cudnn_frontend::graph {
class ConvolutionNode : public NodeCRTP<ConvolutionNode> {
   public:
    Conv_fprop_attributes attributes;

    ConvolutionNode(Conv_fprop_attributes&& attributes_, detail::Context const& context)
        : NodeCRTP(context), attributes(std::move(attributes_)) {}

    Type
    getType() override final {
        return Type::CONVOLUTION;
    }

    error_t
    pre_validate_node() const override final {
        CUDNN_FE_LOG_LABEL_ENDL("INFO: Validating Node Type::CONVOLUTION " << attributes.name);

        RETURN_CUDNN_FRONTEND_ERROR_IF(
            attributes.get_pre_padding().empty(), error_code_t::ATTRIBUTE_NOT_SET, "Pre padding not set.");
        RETURN_CUDNN_FRONTEND_ERROR_IF(
            attributes.get_post_padding().empty(), error_code_t::ATTRIBUTE_NOT_SET, "Post padding not set.");
        RETURN_CUDNN_FRONTEND_ERROR_IF(
            attributes.get_stride().empty(), error_code_t::ATTRIBUTE_NOT_SET, "Conv strides not set.");
        RETURN_CUDNN_FRONTEND_ERROR_IF(
            attributes.get_dilation().empty(), error_code_t::ATTRIBUTE_NOT_SET, "Conv dilation not set.");

        // Implicit GEMM kernels compute B_offset as int32. When the product of all filter
        // dimensions (K * C * spatial...) exceeds INT32_MAX the offset overflows, causing IMA or
        // silent NaN. Conservative upper bound — exact check requires kernel tiling params not
        // available at graph-validate time. Applies to 2D (4D filter) and 3D (5D filter) fprop.
        auto W_it = attributes.inputs.find(Conv_fprop_attributes::input_names::W);
        if (W_it != attributes.inputs.end() && W_it->second) {
            auto const& w_dim = W_it->second->get_dim();
            if (w_dim.size() == 4 || w_dim.size() == 5) {
                int64_t filter_elements = 1;
                for (auto d : w_dim) filter_elements *= d;
                RETURN_CUDNN_FRONTEND_ERROR_IF(
                    filter_elements > static_cast<int64_t>(std::numeric_limits<int32_t>::max()),
                    error_code_t::GRAPH_NOT_SUPPORTED,
                    "Conv filter total elements exceed INT32_MAX.");
            }
        }

        return {error_code_t::OK, ""};
    }

    error_t
    infer_properties_node() override final {
        CUDNN_FE_LOG_LABEL_ENDL("INFO:     Inferrencing properties for conv node " << attributes.name);

        attributes.fill_from_context(context);

        // TODO: Only inferrencing from (X, W) -> Y works today.
        auto& X = attributes.inputs.find(Conv_fprop_attributes::input_names::X)->second;
        auto& W = attributes.inputs.find(Conv_fprop_attributes::input_names::W)->second;
        auto& Y = attributes.outputs.find(Conv_fprop_attributes::output_names::Y)->second;

        auto const x_tensor_dim = X->get_dim();
        auto const w_tensor_dim = W->get_dim();
        auto y_tensor_dim       = Y->get_dim();

        // Only infer dims and strides if user did not set them
        if (y_tensor_dim.empty()) {
            y_tensor_dim.resize(x_tensor_dim.size());
            auto const& pre_padding  = attributes.get_pre_padding();
            auto const& post_padding = attributes.get_post_padding();
            auto const& stride       = attributes.get_stride();
            auto const& dilation     = attributes.get_dilation();
            // N
            y_tensor_dim[0] = x_tensor_dim[0];
            // PQ
            for (size_t dim = 2; dim < x_tensor_dim.size(); ++dim) {
                y_tensor_dim[dim] = 1 + (x_tensor_dim[dim] - dilation[dim - 2] * (w_tensor_dim[dim] - 1) - 1 +
                                         pre_padding[dim - 2] + post_padding[dim - 2]) /
                                            stride[dim - 2];
            }
            // K
            y_tensor_dim[1] = w_tensor_dim[0];
            Y->set_dim(y_tensor_dim);
        }
        if (Y->get_stride().empty()) {
            auto const& Y_dim = Y->get_dim();
            // Default to NHWC
            auto const& stride_order = detail::generate_NHWC_stride_order(Y_dim.size());
            Y->set_stride(detail::generate_stride(Y_dim, stride_order));
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
        CUDNN_FE_LOG_LABEL("INFO: Building ConvolutionNode operations " << attributes.name << " ");

        // Create convolution descriptor by directly calling cuDNN backend API
        ConvDesc_v8 convolution_descriptor;
        int64_t const spatial_dim_count = attributes.get_pre_padding().size();

        _CUDNN_CHECK_CUDNN_ERROR(
            convolution_descriptor.initialize_managed_backend_pointer(CUDNN_BACKEND_CONVOLUTION_DESCRIPTOR));

        // Set compute type
        cudnnDataType_t cudnn_data_type;

        _CUDNN_CHECK_CUDNN_ERROR(detail::convert_to_cudnn_type(attributes.compute_data_type, cudnn_data_type));

        _CUDNN_CHECK_CUDNN_ERROR(detail::set_attribute(convolution_descriptor.get_raw_desc(),
                                                       CUDNN_ATTR_CONVOLUTION_COMP_TYPE,
                                                       CUDNN_TYPE_DATA_TYPE,
                                                       1,
                                                       &cudnn_data_type));

        // Set convolution mode
        cudnnConvolutionMode_t mode = detail::convert_to_cudnn_type(attributes.math_mode);

        _CUDNN_CHECK_CUDNN_ERROR(detail::set_attribute(convolution_descriptor.get_raw_desc(),
                                                       CUDNN_ATTR_CONVOLUTION_CONV_MODE,
                                                       CUDNN_TYPE_CONVOLUTION_MODE,
                                                       1,
                                                       &mode));

        // Set spatial dimensions

        _CUDNN_CHECK_CUDNN_ERROR(detail::set_attribute(convolution_descriptor.get_raw_desc(),
                                                       CUDNN_ATTR_CONVOLUTION_SPATIAL_DIMS,
                                                       CUDNN_TYPE_INT64,
                                                       1,
                                                       &spatial_dim_count));

        // Set pre-padding

        _CUDNN_CHECK_CUDNN_ERROR(detail::set_attribute(convolution_descriptor.get_raw_desc(),
                                                       CUDNN_ATTR_CONVOLUTION_PRE_PADDINGS,
                                                       CUDNN_TYPE_INT64,
                                                       spatial_dim_count,
                                                       attributes.get_pre_padding().data()));

        // Set post-padding

        _CUDNN_CHECK_CUDNN_ERROR(detail::set_attribute(convolution_descriptor.get_raw_desc(),
                                                       CUDNN_ATTR_CONVOLUTION_POST_PADDINGS,
                                                       CUDNN_TYPE_INT64,
                                                       spatial_dim_count,
                                                       attributes.get_post_padding().data()));

        // Set dilation

        _CUDNN_CHECK_CUDNN_ERROR(detail::set_attribute(convolution_descriptor.get_raw_desc(),
                                                       CUDNN_ATTR_CONVOLUTION_DILATIONS,
                                                       CUDNN_TYPE_INT64,
                                                       spatial_dim_count,
                                                       attributes.get_dilation().data()));

        // Set strides

        _CUDNN_CHECK_CUDNN_ERROR(detail::set_attribute(convolution_descriptor.get_raw_desc(),
                                                       CUDNN_ATTR_CONVOLUTION_FILTER_STRIDES,
                                                       CUDNN_TYPE_INT64,
                                                       spatial_dim_count,
                                                       attributes.get_stride().data()));

        _CUDNN_CHECK_CUDNN_ERROR(detail::finalize(convolution_descriptor.get_raw_desc()));
        CUDNN_FE_LOG_LABEL_ENDL(convolution_descriptor);

        // Create operation by directly calling cuDNN backend API
        Operation_v8 convolution_operation;

        _CUDNN_CHECK_CUDNN_ERROR(convolution_operation.initialize_managed_backend_pointer(
            CUDNN_BACKEND_OPERATION_CONVOLUTION_FORWARD_DESCRIPTOR));

        // Set input tensor X
        CUDNN_FE_VALIDATE_AND_ASSIGN_INPUT_TENSOR(X, Conv_fprop_attributes::input_names::X);
        auto x_desc = tensors.at(X->second->get_uid())->get_raw_desc();

        _CUDNN_CHECK_CUDNN_ERROR(detail::set_attribute(convolution_operation.get_raw_desc(),
                                                       CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_X,
                                                       CUDNN_TYPE_BACKEND_DESCRIPTOR,
                                                       1,
                                                       &x_desc));

        // Set weight tensor W
        CUDNN_FE_VALIDATE_AND_ASSIGN_INPUT_TENSOR(W, Conv_fprop_attributes::input_names::W);
        auto w_desc = tensors.at(W->second->get_uid())->get_raw_desc();

        _CUDNN_CHECK_CUDNN_ERROR(detail::set_attribute(convolution_operation.get_raw_desc(),
                                                       CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_W,
                                                       CUDNN_TYPE_BACKEND_DESCRIPTOR,
                                                       1,
                                                       &w_desc));

        // Set output tensor Y
        CUDNN_FE_VALIDATE_AND_ASSIGN_OUTPUT_TENSOR(Y, Conv_fprop_attributes::output_names::Y);
        auto y_desc = tensors.at(Y->second->get_uid())->get_raw_desc();

        _CUDNN_CHECK_CUDNN_ERROR(detail::set_attribute(convolution_operation.get_raw_desc(),
                                                       CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_Y,
                                                       CUDNN_TYPE_BACKEND_DESCRIPTOR,
                                                       1,
                                                       &y_desc));

        // Set convolution descriptor
        auto conv_desc_ptr = convolution_descriptor.get_raw_desc();

        _CUDNN_CHECK_CUDNN_ERROR(detail::set_attribute(convolution_operation.get_raw_desc(),
                                                       CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_CONV_DESC,
                                                       CUDNN_TYPE_BACKEND_DESCRIPTOR,
                                                       1,
                                                       &conv_desc_ptr));

        // Set alpha and beta
        float alpha = 1.0f;
        float beta  = 0.0f;

        _CUDNN_CHECK_CUDNN_ERROR(detail::set_attribute(convolution_operation.get_raw_desc(),
                                                       CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_ALPHA,
                                                       CUDNN_TYPE_FLOAT,
                                                       1,
                                                       &alpha));

        _CUDNN_CHECK_CUDNN_ERROR(detail::set_attribute(convolution_operation.get_raw_desc(),
                                                       CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_BETA,
                                                       CUDNN_TYPE_FLOAT,
                                                       1,
                                                       &beta));

        _CUDNN_CHECK_CUDNN_ERROR(detail::finalize(convolution_operation.get_raw_desc()));

        operations.push_back(std::make_shared<Operation_v8>(std::move(convolution_operation)));

        auto const& non_virtual_uids = attributes.get_non_virtual_uids();
        uids_involved_in_operations.insert(non_virtual_uids.begin(), non_virtual_uids.end());
        return {error_code_t::OK, ""};
    }

#ifndef CUDNN_FRONTEND_SKIP_JSON_LIB
    virtual void
    serialize(json& j) const override final {
        j = attributes;
        j.update(R"({"tag": "CONV_FPROP"})"_json);
    }
#endif
};

}  // namespace cudnn_frontend::graph