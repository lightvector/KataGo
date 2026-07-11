#pragma once

#include "../graph_helpers.h"
#include "../node_interface.h"

namespace cudnn_frontend::graph {

class ReshapeNode : public NodeCRTP<ReshapeNode> {
   public:
    Reshape_attributes attributes;

    ReshapeNode(Reshape_attributes&& attributes_, detail::Context const& context)
        : NodeCRTP(context), attributes(std::move(attributes_)) {}

    Type
    getType() override final {
        return Type::RESHAPE;
    }

    error_t
    infer_properties_node() override final {
        CUDNN_FE_LOG_LABEL_ENDL("INFO:     Inferrencing properties for reshape node " << attributes.name);

        auto y_tensor = attributes.outputs[Reshape_attributes::output_names::Y];

        attributes.fill_from_context(context);

        // If user does not set shape and layout of the output tensor,
        // Get it from node attributes
        // If layout is not set, generate the strides from layout

        if (y_tensor->get_dim().empty() && attributes.get_dim().size()) {
            y_tensor->set_dim(attributes.dim);
        }

        // Graph producers may represent scalar reshape outputs
        // as rank-0 tensors (dim={}).  Without promotion, stride inference below
        // would call generate_NHWC_stride_order(0), which is UB, and the cuDNN
        // backend rejects rank-0 descriptors regardless.  Promote to canonical
        // rank-1 length-1 -- value-preserving because a scalar has volume 1 --
        // while keeping the normalization node-local so downstream broadcast/shape
        // inference remains intact.
        if (y_tensor->get_dim().empty()) {
            y_tensor->set_dim({1});
            if (y_tensor->get_stride().empty()) {
                y_tensor->set_stride({1});
            }
        }

        if (y_tensor->get_stride().empty()) {
            if (attributes.get_stride().size()) {
                y_tensor->set_stride(attributes.get_stride());
            } else {
                auto const& y_dim = y_tensor->get_dim();
                // Default to NHWC for multi-axis tensors. generate_NHWC_stride_order
                // indexes stride_order[1] and assumes num_dims >= 2; use row-major
                // for scalars (0-D) and vectors (1-D).
                const int64_t rank = static_cast<int64_t>(y_dim.size());
                std::vector<int64_t> const stride_order =
                    rank < 2 ? detail::generate_row_major_stride_order(rank) : detail::generate_NHWC_stride_order(rank);
                y_tensor->set_stride(detail::generate_stride(y_dim, stride_order));
            }
        }

        if (y_tensor->get_dim().empty() || y_tensor->get_stride().empty()) {
            return {error_code_t::SHAPE_DEDUCTION_FAILED, "Reshape node output shape deduction failed"};
        }

        CUDNN_FE_VALIDATE_AND_ASSIGN_INPUT_TENSOR(X, Reshape_attributes::input_names::X);
        auto const& input_data_type = X->second->get_data_type();
        if (y_tensor->get_data_type() == DataType_t::NOT_SET) {
            y_tensor->set_data_type(input_data_type);
        } else if (attributes.get_reshape_mode() == ReshapeMode_t::LOGICAL) {
            // Lexicographic reshape preserves element type; reject inconsistent metadata.
            // VIEW_ONLY paths (e.g. SDPA backward) may set Y dtype after reshape to match a consumer.
            RETURN_CUDNN_FRONTEND_ERROR_IF(
                y_tensor->get_data_type() != input_data_type,
                error_code_t::INVALID_VALUE,
                "Output and input tensor data types must match for LOGICAL reshape operation.");
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
        CUDNN_FE_LOG_LABEL("INFO: " << "Building ReshapeNode operations " << attributes.name << " ");

        // Create operation by directly calling cuDNN backend API
        Operation_v8 reshape_operation;

        _CUDNN_CHECK_CUDNN_ERROR(
            reshape_operation.initialize_managed_backend_pointer(CUDNN_BACKEND_OPERATION_RESHAPE_DESCRIPTOR));

        // Set input tensor X
        CUDNN_FE_VALIDATE_AND_ASSIGN_INPUT_TENSOR(X, Reshape_attributes::input_names::X);
        auto x_desc = tensors.at(X->second->get_uid())->get_raw_desc();

        _CUDNN_CHECK_CUDNN_ERROR(detail::set_attribute(reshape_operation.get_raw_desc(),
                                                       CUDNN_ATTR_OPERATION_RESHAPE_XDESC,
                                                       CUDNN_TYPE_BACKEND_DESCRIPTOR,
                                                       1,
                                                       &x_desc));
#if (CUDNN_VERSION >= 92200)
        // Set reshape mode
        cudnnBackendReshapeMode_t cudnn_reshape_mode;
        _CUDNN_CHECK_CUDNN_ERROR(detail::convert_to_cudnn_type(attributes.get_reshape_mode(), cudnn_reshape_mode));
        _CUDNN_CHECK_CUDNN_ERROR(detail::set_attribute(reshape_operation.get_raw_desc(),
                                                       CUDNN_ATTR_OPERATION_RESHAPE_MODE,
                                                       CUDNN_TYPE_RESHAPE_MODE,
                                                       1,
                                                       &cudnn_reshape_mode));
#endif
        // Set output tensor Y
        CUDNN_FE_VALIDATE_AND_ASSIGN_OUTPUT_TENSOR(Y, Reshape_attributes::output_names::Y);
        auto y_desc = tensors.at(Y->second->get_uid())->get_raw_desc();

        _CUDNN_CHECK_CUDNN_ERROR(detail::set_attribute(reshape_operation.get_raw_desc(),
                                                       CUDNN_ATTR_OPERATION_RESHAPE_YDESC,
                                                       CUDNN_TYPE_BACKEND_DESCRIPTOR,
                                                       1,
                                                       &y_desc));

        _CUDNN_CHECK_CUDNN_ERROR(detail::finalize(reshape_operation.get_raw_desc()));

        operations.push_back(std::make_shared<Operation_v8>(std::move(reshape_operation)));

        auto const& non_virtual_uids = attributes.get_non_virtual_uids();
        uids_involved_in_operations.insert(non_virtual_uids.begin(), non_virtual_uids.end());
        return {error_code_t::OK, ""};
    }

#ifndef CUDNN_FRONTEND_SKIP_JSON_LIB
    virtual void
    serialize(json& j) const override final {
        j = attributes;
        j.update(R"( {"tag": "RESHAPE"})"_json);
    }
#endif
};

inline std::shared_ptr<Tensor_attributes>
INode::reshape(std::shared_ptr<Tensor_attributes> input, Reshape_attributes attributes) {
    attributes.inputs[Reshape_attributes::input_names::X] = input;
    auto Y = attributes.outputs[Reshape_attributes::output_names::Y] = output_tensor(attributes.name + "::Y");

    sub_nodes.emplace_back(std::make_unique<ReshapeNode>(std::move(attributes), context));
    return Y;
}

}  // namespace cudnn_frontend::graph