#pragma once

#include "../graph_helpers.h"
#include "../node_interface.h"

namespace cudnn_frontend::graph {

class ReductionNode : public NodeCRTP<ReductionNode> {
   public:
    Reduction_attributes attributes;

    ReductionNode(Reduction_attributes&& attributes_, detail::Context const& context)
        : NodeCRTP(context), attributes(std::move(attributes_)) {}

    Type
    getType() override final {
        return Type::REDUCTION;
    }

    error_t
    pre_validate_node() const override final {
        CUDNN_FE_LOG_LABEL_ENDL("INFO: Validating ReductionNode " << attributes.name);

        if (attributes.get_is_deterministic() && detail::get_backend_version() < 91100) {
            return {error_code_t::GRAPH_NOT_SUPPORTED, "DETERMINISTIC mode is not supported in cudnn version < 9.11.0"};
        }
        return {error_code_t::OK, ""};
    }

    error_t
    infer_properties_node() override final {
        CUDNN_FE_LOG_LABEL_ENDL("INFO:     Inferrencing properties for reduction node " << attributes.name);

        attributes.fill_from_context(context);

        // Only inferrencing from IN_0 to OUT_0 works today.
        auto x_tensor = attributes.inputs[Reduction_attributes::input_names::X];
        auto y_tensor = attributes.outputs[Reduction_attributes::output_names::Y];

        auto const& x_tensor_dim = x_tensor->get_dim();
        auto y_tensor_dim        = y_tensor->get_dim();
        // Only infer dims and strides if user did not set them
        if (y_tensor_dim.empty()) {
            y_tensor->set_dim(x_tensor_dim);
        }
        if (y_tensor->get_stride().empty()) {
            auto const& y_dim = y_tensor->get_dim();
            // Default to NHWC
            auto const& stride_order = detail::generate_NHWC_stride_order(y_dim.size());
            y_tensor->set_stride(detail::generate_stride(y_dim, stride_order));
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
        CUDNN_FE_LOG_LABEL("INFO: " << "Building ReductionNode operations " << attributes.name << " ");

        // Create reduction descriptor by directly calling cuDNN backend API
        ReductionDesc_v8 reduction_descriptor;

        // 1. Create the backend descriptor

        _CUDNN_CHECK_CUDNN_ERROR(
            reduction_descriptor.initialize_managed_backend_pointer(CUDNN_BACKEND_REDUCTION_DESCRIPTOR));

        // 2. Set compute type attribute
        cudnnDataType_t cudnn_data_type;

        _CUDNN_CHECK_CUDNN_ERROR(detail::convert_to_cudnn_type(attributes.compute_data_type, cudnn_data_type));

        _CUDNN_CHECK_CUDNN_ERROR(detail::set_attribute(reduction_descriptor.get_raw_desc(),
                                                       CUDNN_ATTR_REDUCTION_COMP_TYPE,
                                                       CUDNN_TYPE_DATA_TYPE,
                                                       1,
                                                       &cudnn_data_type));

        // 3. Set reduction operator attribute
        cudnnReduceTensorOp_t cudnn_reduction_mode;

        _CUDNN_CHECK_CUDNN_ERROR(detail::convert_to_cudnn_type(attributes.get_mode().value(), cudnn_reduction_mode));

        _CUDNN_CHECK_CUDNN_ERROR(detail::set_attribute(reduction_descriptor.get_raw_desc(),
                                                       CUDNN_ATTR_REDUCTION_OPERATOR,
                                                       CUDNN_TYPE_REDUCTION_OPERATOR_TYPE,
                                                       1,
                                                       &cudnn_reduction_mode));

        // 4. Set deterministic mode if supported
#if (CUDNN_VERSION >= 91100)
        if (detail::get_backend_version() >= 91100) {
            bool is_deterministic = attributes.get_is_deterministic();

            _CUDNN_CHECK_CUDNN_ERROR(detail::set_attribute(reduction_descriptor.get_raw_desc(),
                                                           CUDNN_ATTR_REDUCTION_IS_DETERMINISTIC,
                                                           CUDNN_TYPE_BOOLEAN,
                                                           1,
                                                           &is_deterministic));
        }
#endif

        // 5. Finalize the descriptor

        _CUDNN_CHECK_CUDNN_ERROR(detail::finalize(reduction_descriptor.get_raw_desc()));
        CUDNN_FE_LOG_LABEL_ENDL(reduction_descriptor);

        // Create operation by directly calling cuDNN backend API
        Operation_v8 reduction_operation;

        // Validate input tensors are set
        CUDNN_FE_VALIDATE_AND_ASSIGN_INPUT_TENSOR(X, Reduction_attributes::input_names::X);
        CUDNN_FE_VALIDATE_AND_ASSIGN_OUTPUT_TENSOR(Y, Reduction_attributes::output_names::Y);

        // 1. Create the backend operation descriptor

        _CUDNN_CHECK_CUDNN_ERROR(
            reduction_operation.initialize_managed_backend_pointer(CUDNN_BACKEND_OPERATION_REDUCTION_DESCRIPTOR));

        // 2. Set the reduction descriptor attribute
        auto reduction_desc_ptr = reduction_descriptor.get_raw_desc();

        _CUDNN_CHECK_CUDNN_ERROR(detail::set_attribute(reduction_operation.get_raw_desc(),
                                                       CUDNN_ATTR_OPERATION_REDUCTION_DESC,
                                                       CUDNN_TYPE_BACKEND_DESCRIPTOR,
                                                       1,
                                                       &reduction_desc_ptr));

        // 3. Set the input tensor (X) descriptor attribute
        auto x_backend_desc = tensors.at(X->second->get_uid())->get_raw_desc();

        _CUDNN_CHECK_CUDNN_ERROR(detail::set_attribute(reduction_operation.get_raw_desc(),
                                                       CUDNN_ATTR_OPERATION_REDUCTION_XDESC,
                                                       CUDNN_TYPE_BACKEND_DESCRIPTOR,
                                                       1,
                                                       &x_backend_desc));

        // 4. Set the output tensor (Y) descriptor attribute
        auto y_backend_desc = tensors.at(Y->second->get_uid())->get_raw_desc();

        _CUDNN_CHECK_CUDNN_ERROR(detail::set_attribute(reduction_operation.get_raw_desc(),
                                                       CUDNN_ATTR_OPERATION_REDUCTION_YDESC,
                                                       CUDNN_TYPE_BACKEND_DESCRIPTOR,
                                                       1,
                                                       &y_backend_desc));

        // 5. Finalize the operation descriptor

        _CUDNN_CHECK_CUDNN_ERROR(detail::finalize(reduction_operation.get_raw_desc()));

        operations.push_back(std::make_shared<Operation_v8>(std::move(reduction_operation)));

        auto const& non_virtual_uids = attributes.get_non_virtual_uids();
        uids_involved_in_operations.insert(non_virtual_uids.begin(), non_virtual_uids.end());
        return {error_code_t::OK, ""};
    }

#ifndef CUDNN_FRONTEND_SKIP_JSON_LIB
    virtual void
    serialize(json& j) const override final {
        j = attributes;
        j.update(R"({"tag": "REDUCTION"})"_json);
    }
#endif
};

inline void
INode::reduction(std::shared_ptr<Tensor_attributes> a,
                 Reduction_attributes attributes,
                 std::shared_ptr<Tensor_attributes> c) {
    attributes.inputs[Reduction_attributes::input_names::X]   = a;
    attributes.outputs[Reduction_attributes::output_names::Y] = c;
    sub_nodes.emplace_back(std::make_unique<ReductionNode>(std::move(attributes), context));
}

inline std::shared_ptr<Tensor_attributes>
INode::reduction(std::shared_ptr<Tensor_attributes> input, Reduction_attributes attributes) {
    attributes.inputs[Reduction_attributes::input_names::X] = input;
    auto Y = attributes.outputs[Reduction_attributes::output_names::Y] = output_tensor(attributes.name + "::Y");

    sub_nodes.emplace_back(std::make_unique<ReductionNode>(std::move(attributes), context));
    return Y;
}
}  // namespace cudnn_frontend::graph
