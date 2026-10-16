#pragma once

#include "../../cudnn_frontend_Logging.h"

#include "../graph_helpers.h"
#include "../node_interface.h"

namespace cudnn_frontend::graph {

class TransposeNode : public NodeCRTP<TransposeNode> {
   public:
    Transpose_attributes attributes;

    TransposeNode(Transpose_attributes&& attributes_, detail::Context const& context)
        : NodeCRTP(context), attributes(std::move(attributes_)) {}

    Type
    getType() override final {
        return Type::TRANSPOSE;
    }

    error_t
    infer_properties_node() override final {
        getLogger() << "[cudnn_frontend] INFO:     Inferrencing properties for transpose node " << attributes.name
                    << std::endl;

        attributes.fill_from_context(context);

        auto const& X_tensor = attributes.inputs[Transpose_attributes::input_names::X];
        auto Y_tensor        = attributes.outputs[Transpose_attributes::output_names::Y];

        // Get input properties
        auto const& input_dim       = X_tensor->get_dim();
        auto const& input_stride    = X_tensor->get_stride();
        auto const& input_data_type = X_tensor->get_data_type();
        auto const& permutation     = attributes.get_permutation();

        // Validate permutation
        RETURN_CUDNN_FRONTEND_ERROR_IF(
            permutation.empty(), error_code_t::ATTRIBUTE_NOT_SET, "Permutation must be set for transpose operation.");

        RETURN_CUDNN_FRONTEND_ERROR_IF(permutation.size() != input_dim.size(),
                                       error_code_t::INVALID_VALUE,
                                       "Permutation size must match input tensor dimensionality.");

        // Check that permutation is a valid permutation (contains each index 0 to n-1 exactly once)
        std::vector<bool> seen(permutation.size(), false);
        for (auto idx : permutation) {
            RETURN_CUDNN_FRONTEND_ERROR_IF(idx < 0 || idx >= static_cast<int64_t>(permutation.size()),
                                           error_code_t::INVALID_VALUE,
                                           "Permutation indices must be in range [0, n-1].");
            RETURN_CUDNN_FRONTEND_ERROR_IF(
                seen[idx], error_code_t::INVALID_VALUE, "Permutation indices must be unique.");
            seen[idx] = true;
        }

        // Infer output dimensions by permuting input dimensions
        std::vector<int64_t> output_dim(input_dim.size());
        for (size_t i = 0; i < permutation.size(); ++i) {
            output_dim[i] = input_dim[permutation[i]];
        }

        // Infer output strides by permuting input strides
        std::vector<int64_t> output_stride(input_stride.size());
        for (size_t i = 0; i < permutation.size(); ++i) {
            output_stride[i] = input_stride[permutation[i]];
        }

        // Set output tensor properties
        if (Y_tensor->get_dim().empty()) {
            Y_tensor->set_dim(output_dim);
        }
        if (Y_tensor->get_stride().empty()) {
            Y_tensor->set_stride(output_stride);
        }
        if (Y_tensor->get_data_type() == DataType_t::NOT_SET) {
            Y_tensor->set_data_type(input_data_type);
        }

        RETURN_CUDNN_FRONTEND_ERROR_IF(Y_tensor->get_data_type() != input_data_type,
                                       error_code_t::INVALID_VALUE,
                                       "Output and input tensor data types must match for transpose operation.");

        return {error_code_t::OK, ""};
    }

    error_t
    create_cudnn_operations(
        std::unordered_set<Tensor_attributes::uid_t>& uids_involved_in_operations,
        std::vector<std::shared_ptr<cudnn_frontend::Operation>>& operations,
        managed_backend_descriptor_t& raw_operations,
        std::unordered_map<int64_t, std::shared_ptr<cudnn_frontend::Tensor>>& tensors) const override final {
        getLogger() << "[cudnn_frontend] INFO: " << "Building TransposeNode operations " << attributes.name
                    << std::endl;

#if (CUDNN_VERSION >= 92200)
        // cuDNN >= 9.22.0: Use native backend transpose operation
        auto cudnn_ver_error = error_t{error_code_t::GRAPH_NOT_SUPPORTED, "Transpose requires cuDNN v9.22.0"};
        NV_CUDNN_FE_DYNAMIC_CHECK_CUDNN_BACKEND_VERSION(92200, cudnn_ver_error);
        CUDNN_FRONTEND_UNUSED(operations);

        auto transpose_operation = make_shared_backend_pointer(CUDNN_BACKEND_OPERATION_TRANSPOSE_DESCRIPTOR);

        // Set input tensor
        auto X         = attributes.inputs.at(Transpose_attributes::input_names::X);
        auto backend_x = tensors[X->get_uid()]->get_desc()->get_backend_descriptor();
        _CUDNN_CHECK_CUDNN_ERROR(detail::set_attribute(transpose_operation->get_backend_descriptor(),
                                                       CUDNN_ATTR_OPERATION_TRANSPOSE_XDESC,
                                                       CUDNN_TYPE_BACKEND_DESCRIPTOR,
                                                       1,
                                                       &backend_x));

        // Set output tensor
        auto Y         = attributes.outputs.at(Transpose_attributes::output_names::Y);
        auto backend_y = tensors[Y->get_uid()]->get_desc()->get_backend_descriptor();
        _CUDNN_CHECK_CUDNN_ERROR(detail::set_attribute(transpose_operation->get_backend_descriptor(),
                                                       CUDNN_ATTR_OPERATION_TRANSPOSE_YDESC,
                                                       CUDNN_TYPE_BACKEND_DESCRIPTOR,
                                                       1,
                                                       &backend_y));

        // Set permutation
        auto permutation = attributes.get_permutation();
        _CUDNN_CHECK_CUDNN_ERROR(detail::set_attribute(transpose_operation->get_backend_descriptor(),
                                                       CUDNN_ATTR_OPERATION_TRANSPOSE_PERMUTATION,
                                                       CUDNN_TYPE_INT64,
                                                       permutation.size(),
                                                       permutation.data()));

        _CUDNN_CHECK_CUDNN_ERROR(detail::finalize(transpose_operation->get_backend_descriptor()));

        raw_operations.push_back(transpose_operation);

        auto const& non_virtual_uids = attributes.get_non_virtual_uids();
        uids_involved_in_operations.insert(non_virtual_uids.begin(), non_virtual_uids.end());
#else
        CUDNN_FRONTEND_UNUSED(operations);
        CUDNN_FRONTEND_UNUSED(raw_operations);
        CUDNN_FRONTEND_UNUSED(tensors);
        CUDNN_FRONTEND_UNUSED(uids_involved_in_operations);
        return {error_code_t::GRAPH_NOT_SUPPORTED, "Transpose operation requires cuDNN version >= 9.22.0"};
#endif
        return {error_code_t::OK, ""};
    }

#ifndef CUDNN_FRONTEND_SKIP_JSON_LIB
    virtual void
    serialize(json& j) const override final {
        j = attributes;
        j.update(R"( {"tag": "TRANSPOSE"})"_json);
    }
#endif
};

inline std::shared_ptr<Tensor_attributes>
INode::transpose(std::shared_ptr<Tensor_attributes> input, Transpose_attributes attributes) {
    attributes.inputs[Transpose_attributes::input_names::X] = input;
    auto Y = attributes.outputs[Transpose_attributes::output_names::Y] = output_tensor(attributes.name + "::Y");

    sub_nodes.emplace_back(std::make_unique<TransposeNode>(std::move(attributes), context));
    return Y;
}

}  // namespace cudnn_frontend::graph
