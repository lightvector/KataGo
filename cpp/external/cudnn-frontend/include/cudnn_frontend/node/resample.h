#pragma once

#include "../graph_helpers.h"
#include "../node_interface.h"

namespace cudnn_frontend::graph {

class ResampleNode : public NodeCRTP<ResampleNode> {
   public:
    Resample_attributes attributes;

    ResampleNode(Resample_attributes&& attributes_, detail::Context const& context)
        : NodeCRTP(context), attributes(std::move(attributes_)) {}

    Type
    getType() override final {
        return Type::RESAMPLE;
    }

    error_t
    pre_validate_node() const override final {
        CUDNN_FE_LOG_LABEL_ENDL("INFO: " << "Validating ResampleNode " << attributes.name);

        RETURN_CUDNN_FRONTEND_ERROR_IF(attributes.generate_index.has_value() == false,
                                       error_code_t::ATTRIBUTE_NOT_SET,
                                       "generate_index attribute not set");

        if (attributes.generate_index.value() == true && attributes.resample_mode == ResampleMode_t::MAXPOOL) {
            CUDNN_FE_VALIDATE_OUTPUT_TENSOR(Resample_attributes::output_names::Index);
        }

        // Make sure that the mode can be lowered to BE
        cudnnResampleMode_t dummy;
        RETURN_CUDNN_FRONTEND_ERROR_IF(
            detail::convert_to_cudnn_type(attributes.resample_mode, dummy) != CUDNN_STATUS_SUCCESS,
            error_code_t::ATTRIBUTE_NOT_SET,
            "Invalid resample mode.");

        return {error_code_t::OK, ""};
    }

    error_t
    infer_properties_node() override final {
        CUDNN_FE_LOG_LABEL_ENDL("INFO:     Inferrencing properties for resample node " << attributes.name);

        auto y_tensor = attributes.outputs[Resample_attributes::output_names::Y];
        auto x_tensor = attributes.inputs[Resample_attributes::input_names::X];

        attributes.fill_from_context(context);

        // If user does not set shape and layout of the output tensor,
        // Get it from node attributes
        if (y_tensor->get_dim().empty()) {
            auto const x_dim = x_tensor->get_dim();
            auto y_dim       = y_tensor->get_dim();
            y_dim            = x_dim;

            // 2 cause first two dimensions are batch and channels
            for (auto dim = 2u; dim < x_dim.size(); ++dim) {
                auto spatial_dim = dim - 2u;
                y_dim[dim] =
                    1 + (x_dim[dim] + attributes.pre_padding[spatial_dim].numerator +
                         attributes.post_padding[spatial_dim].numerator - attributes.window[spatial_dim].numerator) /
                            attributes.stride[spatial_dim].numerator;
            }

            y_tensor->set_dim(y_dim);
        }

        // If layout is not set, generate the strides from layout
        if (y_tensor->get_stride().empty()) {
            auto const& y_dim = y_tensor->get_dim();
            // Default to NHWC
            auto const& stride_order = detail::generate_NHWC_stride_order(y_dim.size());
            y_tensor->set_stride(detail::generate_stride(y_dim, stride_order));
        }

        if (attributes.outputs[Resample_attributes::output_names::Index]) {
            auto index_tensor = attributes.outputs[Resample_attributes::output_names::Index];
            index_tensor->set_dim(y_tensor->get_dim());

            // If layout is not set, generate the strides from layout
            if (index_tensor->get_stride().empty()) {
                auto const& index_dim = index_tensor->get_dim();
                // Default to NHWC
                auto const& stride_order = detail::generate_NHWC_stride_order(index_dim.size());
                index_tensor->set_stride(detail::generate_stride(index_dim, stride_order));
            }
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
        CUDNN_FE_LOG_LABEL("INFO: " << "Building ResampleNode operations " << attributes.name << " ");

        auto number_of_spatial_dim = static_cast<int64_t>(attributes.window.size());

        // Create resample descriptor by directly calling cuDNN backend API
        ResampleDesc_v8 resample_descriptor;

        _CUDNN_CHECK_CUDNN_ERROR(
            resample_descriptor.initialize_managed_backend_pointer(CUDNN_BACKEND_RESAMPLE_DESCRIPTOR));

        // Set resample mode
        cudnnResampleMode_t cudnn_resample_mode;

        _CUDNN_CHECK_CUDNN_ERROR(detail::convert_to_cudnn_type(attributes.resample_mode, cudnn_resample_mode));

        _CUDNN_CHECK_CUDNN_ERROR(detail::set_attribute(resample_descriptor.get_raw_desc(),
                                                       CUDNN_ATTR_RESAMPLE_MODE,
                                                       CUDNN_TYPE_RESAMPLE_MODE,
                                                       1,
                                                       &cudnn_resample_mode));

        // Set compute type
        cudnnDataType_t cudnn_data_type;

        _CUDNN_CHECK_CUDNN_ERROR(detail::convert_to_cudnn_type(attributes.compute_data_type, cudnn_data_type));

        _CUDNN_CHECK_CUDNN_ERROR(detail::set_attribute(resample_descriptor.get_raw_desc(),
                                                       CUDNN_ATTR_RESAMPLE_COMP_TYPE,
                                                       CUDNN_TYPE_DATA_TYPE,
                                                       1,
                                                       &cudnn_data_type));

        // Set nan propagation
        cudnnNanPropagation_t nan_opt = CUDNN_PROPAGATE_NAN;

        _CUDNN_CHECK_CUDNN_ERROR(detail::set_attribute(resample_descriptor.get_raw_desc(),
                                                       CUDNN_ATTR_RESAMPLE_NAN_PROPAGATION,
                                                       CUDNN_TYPE_NAN_PROPOGATION,
                                                       1,
                                                       &nan_opt));

        // Set padding mode
        cudnnPaddingMode_t cudnn_padding_mode;

        _CUDNN_CHECK_CUDNN_ERROR(detail::convert_to_cudnn_type(attributes.padding_mode, cudnn_padding_mode));

        _CUDNN_CHECK_CUDNN_ERROR(detail::set_attribute(resample_descriptor.get_raw_desc(),
                                                       CUDNN_ATTR_RESAMPLE_PADDING_MODE,
                                                       CUDNN_TYPE_PADDING_MODE,
                                                       1,
                                                       &cudnn_padding_mode));

        // Set spatial dimensions

        _CUDNN_CHECK_CUDNN_ERROR(detail::set_attribute(resample_descriptor.get_raw_desc(),
                                                       CUDNN_ATTR_RESAMPLE_SPATIAL_DIMS,
                                                       CUDNN_TYPE_INT64,
                                                       1,
                                                       &number_of_spatial_dim));

        // Set window dimensions

        _CUDNN_CHECK_CUDNN_ERROR(detail::set_attribute(resample_descriptor.get_raw_desc(),
                                                       CUDNN_ATTR_RESAMPLE_WINDOW_DIMS,
                                                       CUDNN_TYPE_FRACTION,
                                                       number_of_spatial_dim,
                                                       attributes.window.data()));

        // Set pre padding

        _CUDNN_CHECK_CUDNN_ERROR(detail::set_attribute(resample_descriptor.get_raw_desc(),
                                                       CUDNN_ATTR_RESAMPLE_PRE_PADDINGS,
                                                       CUDNN_TYPE_FRACTION,
                                                       number_of_spatial_dim,
                                                       attributes.pre_padding.data()));

        // Set post padding

        _CUDNN_CHECK_CUDNN_ERROR(detail::set_attribute(resample_descriptor.get_raw_desc(),
                                                       CUDNN_ATTR_RESAMPLE_POST_PADDINGS,
                                                       CUDNN_TYPE_FRACTION,
                                                       number_of_spatial_dim,
                                                       attributes.post_padding.data()));

        // Set strides

        _CUDNN_CHECK_CUDNN_ERROR(detail::set_attribute(resample_descriptor.get_raw_desc(),
                                                       CUDNN_ATTR_RESAMPLE_STRIDES,
                                                       CUDNN_TYPE_FRACTION,
                                                       number_of_spatial_dim,
                                                       attributes.stride.data()));

        // Finalize the descriptor

        _CUDNN_CHECK_CUDNN_ERROR(detail::finalize(resample_descriptor.get_raw_desc()));
        CUDNN_FE_LOG_LABEL_ENDL(resample_descriptor);

        // Create operation by directly calling cuDNN backend API
        Operation_v8 resample_operation;

        _CUDNN_CHECK_CUDNN_ERROR(
            resample_operation.initialize_managed_backend_pointer(CUDNN_BACKEND_OPERATION_RESAMPLE_FWD_DESCRIPTOR));

        // Set input tensor X
        CUDNN_FE_VALIDATE_AND_ASSIGN_INPUT_TENSOR(X, Resample_attributes::input_names::X);
        auto x_desc = tensors.at(X->second->get_uid())->get_raw_desc();

        _CUDNN_CHECK_CUDNN_ERROR(detail::set_attribute(resample_operation.get_raw_desc(),
                                                       CUDNN_ATTR_OPERATION_RESAMPLE_FWD_XDESC,
                                                       CUDNN_TYPE_BACKEND_DESCRIPTOR,
                                                       1,
                                                       &x_desc));

        // Set output tensor Y
        CUDNN_FE_VALIDATE_AND_ASSIGN_OUTPUT_TENSOR(Y, Resample_attributes::output_names::Y);
        auto y_desc = tensors.at(Y->second->get_uid())->get_raw_desc();

        _CUDNN_CHECK_CUDNN_ERROR(detail::set_attribute(resample_operation.get_raw_desc(),
                                                       CUDNN_ATTR_OPERATION_RESAMPLE_FWD_YDESC,
                                                       CUDNN_TYPE_BACKEND_DESCRIPTOR,
                                                       1,
                                                       &y_desc));

        // Set alpha and beta
        double alpha = 1.0;
        double beta  = 0.0;

        _CUDNN_CHECK_CUDNN_ERROR(detail::set_attribute(
            resample_operation.get_raw_desc(), CUDNN_ATTR_OPERATION_RESAMPLE_FWD_ALPHA, CUDNN_TYPE_DOUBLE, 1, &alpha));

        _CUDNN_CHECK_CUDNN_ERROR(detail::set_attribute(
            resample_operation.get_raw_desc(), CUDNN_ATTR_OPERATION_RESAMPLE_FWD_BETA, CUDNN_TYPE_DOUBLE, 1, &beta));

        // Set resample descriptor
        auto resample_raw_desc = resample_descriptor.get_raw_desc();

        _CUDNN_CHECK_CUDNN_ERROR(detail::set_attribute(resample_operation.get_raw_desc(),
                                                       CUDNN_ATTR_OPERATION_RESAMPLE_FWD_DESC,
                                                       CUDNN_TYPE_BACKEND_DESCRIPTOR,
                                                       1,
                                                       &resample_raw_desc));

        // Set index tensor if available
        auto index = attributes.outputs.find(Resample_attributes::output_names::Index);
        if ((index != attributes.outputs.end()) && (index->second != nullptr)) {
            auto idx_desc = tensors.at(index->second->get_uid())->get_raw_desc();

            _CUDNN_CHECK_CUDNN_ERROR(detail::set_attribute(resample_operation.get_raw_desc(),
                                                           CUDNN_ATTR_OPERATION_RESAMPLE_FWD_IDXDESC,
                                                           CUDNN_TYPE_BACKEND_DESCRIPTOR,
                                                           1,
                                                           &idx_desc));
        }

        _CUDNN_CHECK_CUDNN_ERROR(detail::finalize(resample_operation.get_raw_desc()));

        operations.push_back(std::make_shared<Operation_v8>(std::move(resample_operation)));

        auto const& non_virtual_uids = attributes.get_non_virtual_uids();
        uids_involved_in_operations.insert(non_virtual_uids.begin(), non_virtual_uids.end());
        return {error_code_t::OK, ""};
    }

#ifndef CUDNN_FRONTEND_SKIP_JSON_LIB
    virtual void
    serialize(json& j) const override final {
        j = attributes;
        j.update(R"( {"tag": "RESAMPLE"})"_json);
    }
#endif
};

inline std::array<std::shared_ptr<Tensor_attributes>, 2>
INode::resample(std::shared_ptr<Tensor_attributes> input, Resample_attributes attributes) {
    if (attributes.name.empty()) {
        attributes.name += std::to_string(sub_nodes.size());
    }
    attributes.inputs[Resample_attributes::input_names::X] = input;
    auto Y = attributes.outputs[Resample_attributes::output_names::Y] = output_tensor(attributes.name + "::Y");
    std::shared_ptr<Tensor_attributes> Index                          = nullptr;
    if (attributes.generate_index.has_value() && attributes.generate_index.value() == true &&
        attributes.resample_mode == ResampleMode_t::MAXPOOL) {
        Index = attributes.outputs[Resample_attributes::output_names::Index] =
            output_tensor(attributes.name + "::Index");
    }

    sub_nodes.emplace_back(std::make_unique<ResampleNode>(std::move(attributes), context));
    return {Y, Index};
}

}  // namespace cudnn_frontend::graph