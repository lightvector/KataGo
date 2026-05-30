#pragma once

#include "../graph_helpers.h"
#include "../node_interface.h"

namespace cudnn_frontend::graph {

class MoeGroupedMatmulNode : public NodeCRTP<MoeGroupedMatmulNode> {
   public:
    Moe_grouped_matmul_attributes attributes;

    MoeGroupedMatmulNode(Moe_grouped_matmul_attributes&& attributes_, detail::Context const& context)
        : NodeCRTP(context), attributes(std::move(attributes_)) {}

    Type
    getType() override final {
        return Type::MOE_GROUPED_MATMUL;
    }

    error_t
    pre_validate_node() const override final {
        CUDNN_FE_LOG_LABEL_ENDL("INFO: Validating MoeGroupedMatmulNode " << attributes.name);

        auto const token_it = attributes.inputs.find(Moe_grouped_matmul_attributes::input_names::Token);
        RETURN_CUDNN_FRONTEND_ERROR_IF(token_it == attributes.inputs.end() || token_it->second == nullptr,
                                       error_code_t::ATTRIBUTE_NOT_SET,
                                       "MoeGroupedMatmul input Token not set.");

        auto const weight_it = attributes.inputs.find(Moe_grouped_matmul_attributes::input_names::Weight);
        RETURN_CUDNN_FRONTEND_ERROR_IF(weight_it == attributes.inputs.end() || weight_it->second == nullptr,
                                       error_code_t::ATTRIBUTE_NOT_SET,
                                       "MoeGroupedMatmul input Weight not set.");

        auto const first_token_offset_it =
            attributes.inputs.find(Moe_grouped_matmul_attributes::input_names::FirstTokenOffset);
        RETURN_CUDNN_FRONTEND_ERROR_IF(
            first_token_offset_it == attributes.inputs.end() || first_token_offset_it->second == nullptr,
            error_code_t::ATTRIBUTE_NOT_SET,
            "MoeGroupedMatmul input FirstTokenOffset not set.");

        auto const output_it = attributes.outputs.find(Moe_grouped_matmul_attributes::output_names::Output);
        RETURN_CUDNN_FRONTEND_ERROR_IF(output_it == attributes.outputs.end() || output_it->second == nullptr,
                                       error_code_t::ATTRIBUTE_NOT_SET,
                                       "MoeGroupedMatmul output Output not set.");

        return {error_code_t::OK, ""};
    }

    error_t
    infer_properties_node() override final {
        CUDNN_FE_LOG_LABEL_ENDL("INFO:     Inferrencing properties for moe grouped matmul node " << attributes.name);

        attributes.fill_from_context(context);

        auto token_tensor       = attributes.inputs[Moe_grouped_matmul_attributes::input_names::Token];
        auto weight_tensor      = attributes.inputs[Moe_grouped_matmul_attributes::input_names::Weight];
        auto token_index_tensor = attributes.inputs[Moe_grouped_matmul_attributes::input_names::TokenIndex];
        auto output_tensor      = attributes.outputs[Moe_grouped_matmul_attributes::output_names::Output];

        auto const token_tensor_dim  = token_tensor->get_dim();
        auto const weight_tensor_dim = weight_tensor->get_dim();
        auto output_tensor_dim       = output_tensor->get_dim();

        if (output_tensor_dim.empty()) {
            output_tensor_dim.resize(3);
            output_tensor_dim[0] = 1;
            output_tensor_dim[2] = weight_tensor_dim[2];
            if (attributes.mode == MoeGroupedMatmulMode_t::GATHER) {
                output_tensor_dim[1] = token_index_tensor->get_dim()[1];
            } else {
                output_tensor_dim[1] = token_tensor_dim[1];
            }
            output_tensor_dim.resize(3);

            output_tensor->set_dim(output_tensor_dim);
        }

        if (output_tensor->get_stride().empty()) {
            auto const& output_dim   = output_tensor->get_dim();
            auto const& stride_order = detail::generate_row_major_stride_order(output_dim.size());
            output_tensor->set_stride(detail::generate_stride(output_dim, stride_order));
        }

        return {error_code_t::OK, ""};
    }

    error_t
    create_cudnn_operations(
        std::unordered_set<Tensor_attributes::uid_t>& uids_involved_in_operations,
        std::vector<std::shared_ptr<cudnn_frontend::Operation>>& operations,
        managed_backend_descriptor_t& raw_operations,
        std::unordered_map<int64_t, std::shared_ptr<cudnn_frontend::Tensor>>& tensors) const override final {
        getLogger() << "[cudnn_frontend] INFO: "
                    << "Building MoeGroupedMatmulNode operations " << attributes.name << std::endl;
        auto cudnn_ver_error = error_t{error_code_t::GRAPH_NOT_SUPPORTED, "Moe grouped matmul requires cuDNN v9.15.0"};

#if (CUDNN_VERSION >= 91500)
        NV_CUDNN_FE_DYNAMIC_CHECK_CUDNN_BACKEND_VERSION(91500, cudnn_ver_error);
        CUDNN_FRONTEND_UNUSED(operations);

        auto moe_grouped_matmul_operation =
            make_shared_backend_pointer(CUDNN_BACKEND_OPERATION_MOE_GROUPED_MATMUL_DESCRIPTOR);

        cudnnMoeGroupedMatmulMode_t moe_grouped_matmul_mode;
        _CUDNN_CHECK_CUDNN_ERROR(detail::convert_to_cudnn_type(attributes.mode, moe_grouped_matmul_mode));
        _CUDNN_CHECK_CUDNN_ERROR(detail::set_attribute(moe_grouped_matmul_operation->get_backend_descriptor(),
                                                       CUDNN_ATTR_OPERATION_MOE_GROUPED_MATMUL_MODE,
                                                       CUDNN_TYPE_MOE_GROUPED_MATMUL_MODE,
                                                       1,
                                                       &moe_grouped_matmul_mode));

        cudnnDataType_t cudnn_data_type;
        _CUDNN_CHECK_CUDNN_ERROR(detail::convert_to_cudnn_type(attributes.compute_data_type, cudnn_data_type));
        _CUDNN_CHECK_CUDNN_ERROR(detail::set_attribute(moe_grouped_matmul_operation->get_backend_descriptor(),
                                                       CUDNN_ATTR_OPERATION_MOE_GROUPED_MATMUL_MATH_PREC,
                                                       CUDNN_TYPE_DATA_TYPE,
                                                       1,
                                                       &cudnn_data_type));

        auto token         = attributes.inputs.find(Moe_grouped_matmul_attributes::input_names::Token)->second;
        auto backend_token = tensors[token->get_uid()]->get_desc()->get_backend_descriptor();
        _CUDNN_CHECK_CUDNN_ERROR(detail::set_attribute(moe_grouped_matmul_operation->get_backend_descriptor(),
                                                       CUDNN_ATTR_OPERATION_MOE_GROUPED_MATMUL_TOKEN_DESC,
                                                       CUDNN_TYPE_BACKEND_DESCRIPTOR,
                                                       1,
                                                       &backend_token));

        auto weight         = attributes.inputs.find(Moe_grouped_matmul_attributes::input_names::Weight)->second;
        auto backend_weight = tensors[weight->get_uid()]->get_desc()->get_backend_descriptor();
        _CUDNN_CHECK_CUDNN_ERROR(detail::set_attribute(moe_grouped_matmul_operation->get_backend_descriptor(),
                                                       CUDNN_ATTR_OPERATION_MOE_GROUPED_MATMUL_WEIGHT_DESC,
                                                       CUDNN_TYPE_BACKEND_DESCRIPTOR,
                                                       1,
                                                       &backend_weight));

        auto first_token_offset =
            attributes.inputs.find(Moe_grouped_matmul_attributes::input_names::FirstTokenOffset)->second;
        auto backend_first_token_offset = tensors[first_token_offset->get_uid()]->get_desc()->get_backend_descriptor();
        _CUDNN_CHECK_CUDNN_ERROR(detail::set_attribute(moe_grouped_matmul_operation->get_backend_descriptor(),
                                                       CUDNN_ATTR_OPERATION_MOE_GROUPED_MATMUL_FIRST_TOKEN_OFFSET_DESC,
                                                       CUDNN_TYPE_BACKEND_DESCRIPTOR,
                                                       1,
                                                       &backend_first_token_offset));

        auto output         = attributes.outputs.find(Moe_grouped_matmul_attributes::output_names::Output)->second;
        auto backend_output = tensors[output->get_uid()]->get_desc()->get_backend_descriptor();
        _CUDNN_CHECK_CUDNN_ERROR(detail::set_attribute(moe_grouped_matmul_operation->get_backend_descriptor(),
                                                       CUDNN_ATTR_OPERATION_MOE_GROUPED_MATMUL_OUTPUT_DESC,
                                                       CUDNN_TYPE_BACKEND_DESCRIPTOR,
                                                       1,
                                                       &backend_output));

        if (attributes.mode == MoeGroupedMatmulMode_t::GATHER || attributes.mode == MoeGroupedMatmulMode_t::SCATTER) {
            auto token_index = attributes.inputs.find(Moe_grouped_matmul_attributes::input_names::TokenIndex)->second;
            auto backend_token_index = tensors[token_index->get_uid()]->get_desc()->get_backend_descriptor();
            _CUDNN_CHECK_CUDNN_ERROR(detail::set_attribute(moe_grouped_matmul_operation->get_backend_descriptor(),
                                                           CUDNN_ATTR_OPERATION_MOE_GROUPED_MATMUL_TOKEN_INDEX_DESC,
                                                           CUDNN_TYPE_BACKEND_DESCRIPTOR,
                                                           1,
                                                           &backend_token_index));
        }

        if (attributes.mode == MoeGroupedMatmulMode_t::SCATTER) {
            auto token_ks         = attributes.inputs.find(Moe_grouped_matmul_attributes::input_names::TokenKs)->second;
            auto backend_token_ks = tensors[token_ks->get_uid()]->get_desc()->get_backend_descriptor();
            _CUDNN_CHECK_CUDNN_ERROR(detail::set_attribute(moe_grouped_matmul_operation->get_backend_descriptor(),
                                                           CUDNN_ATTR_OPERATION_MOE_GROUPED_MATMUL_TOKEN_KS_DESC,
                                                           CUDNN_TYPE_BACKEND_DESCRIPTOR,
                                                           1,
                                                           &backend_token_ks));

            _CUDNN_CHECK_CUDNN_ERROR(detail::set_attribute(moe_grouped_matmul_operation->get_backend_descriptor(),
                                                           CUDNN_ATTR_OPERATION_MOE_GROUPED_MATMUL_TOP_K,
                                                           CUDNN_TYPE_INT32,
                                                           1,
                                                           &(attributes.top_k)));
        }

        _CUDNN_CHECK_CUDNN_ERROR(detail::finalize(moe_grouped_matmul_operation->get_backend_descriptor()));

        raw_operations.push_back(moe_grouped_matmul_operation);

        auto const& non_virtual_uids = attributes.get_non_virtual_uids();
        uids_involved_in_operations.insert(non_virtual_uids.begin(), non_virtual_uids.end());
        return {error_code_t::OK, ""};
#else
        CUDNN_FRONTEND_UNUSED(uids_involved_in_operations);
        CUDNN_FRONTEND_UNUSED(operations);
        CUDNN_FRONTEND_UNUSED(raw_operations);
        CUDNN_FRONTEND_UNUSED(tensors);
        return cudnn_ver_error;
#endif
    }

#ifndef CUDNN_FRONTEND_SKIP_JSON_LIB
    virtual void
    serialize(json& j) const override final {
        j = attributes;
        j.update(R"( {"tag": "MOE_GROUPED_MATMUL"})"_json);
    }
#endif
};

inline void
INode::moe_grouped_matmul(std::shared_ptr<Tensor_attributes> token,
                          std::shared_ptr<Tensor_attributes> weight,
                          std::shared_ptr<Tensor_attributes> first_token_offset,
                          std::shared_ptr<Tensor_attributes> token_index,
                          std::shared_ptr<Tensor_attributes> token_ks,
                          Moe_grouped_matmul_attributes attributes,
                          std::shared_ptr<Tensor_attributes> output) {
    attributes.inputs[Moe_grouped_matmul_attributes::input_names::Token]            = token;
    attributes.inputs[Moe_grouped_matmul_attributes::input_names::Weight]           = weight;
    attributes.inputs[Moe_grouped_matmul_attributes::input_names::FirstTokenOffset] = first_token_offset;
    attributes.inputs[Moe_grouped_matmul_attributes::input_names::TokenIndex]       = token_index;
    attributes.inputs[Moe_grouped_matmul_attributes::input_names::TokenKs]          = token_ks;
    attributes.outputs[Moe_grouped_matmul_attributes::output_names::Output]         = output;
    sub_nodes.emplace_back(std::make_unique<MoeGroupedMatmulNode>(std::move(attributes), context));
}

}  // namespace cudnn_frontend::graph