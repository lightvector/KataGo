#pragma once

#include "../graph_helpers.h"
#include "../node_interface.h"

namespace cudnn_frontend::graph {

class MoeGroupedMatmulBwdNode : public NodeCRTP<MoeGroupedMatmulBwdNode> {
   public:
    Moe_grouped_matmul_bwd_attributes attributes;

    MoeGroupedMatmulBwdNode(Moe_grouped_matmul_bwd_attributes&& attributes_, detail::Context const& context)
        : NodeCRTP(context), attributes(std::move(attributes_)) {}

    Type
    getType() override final {
        return Type::MOE_GROUPED_MATMUL_BWD;
    }

    error_t
    pre_validate_node() const override final {
        CUDNN_FE_LOG_LABEL_ENDL("INFO: Validating MoeGroupedMatmulBwdNode " << attributes.name);

        auto const doutput_it = attributes.inputs.find(Moe_grouped_matmul_bwd_attributes::input_names::DOutput);
        RETURN_CUDNN_FRONTEND_ERROR_IF(doutput_it == attributes.inputs.end() || doutput_it->second == nullptr,
                                       error_code_t::ATTRIBUTE_NOT_SET,
                                       "MoeGroupedMatmulBwd input DOutput not set.");

        auto const token_it = attributes.inputs.find(Moe_grouped_matmul_bwd_attributes::input_names::Token);
        RETURN_CUDNN_FRONTEND_ERROR_IF(token_it == attributes.inputs.end() || token_it->second == nullptr,
                                       error_code_t::ATTRIBUTE_NOT_SET,
                                       "MoeGroupedMatmulBwd input Token not set.");

        auto const first_token_offset_it =
            attributes.inputs.find(Moe_grouped_matmul_bwd_attributes::input_names::FirstTokenOffset);
        RETURN_CUDNN_FRONTEND_ERROR_IF(
            first_token_offset_it == attributes.inputs.end() || first_token_offset_it->second == nullptr,
            error_code_t::ATTRIBUTE_NOT_SET,
            "MoeGroupedMatmulBwd input FirstTokenOffset not set.");

        auto const dweight_it = attributes.outputs.find(Moe_grouped_matmul_bwd_attributes::output_names::DWeight);
        RETURN_CUDNN_FRONTEND_ERROR_IF(dweight_it == attributes.outputs.end() || dweight_it->second == nullptr,
                                       error_code_t::ATTRIBUTE_NOT_SET,
                                       "MoeGroupedMatmulBwd output DWeight not set.");

        return {error_code_t::OK, ""};
    }

    error_t
    infer_properties_node() override final {
        CUDNN_FE_LOG_LABEL_ENDL("INFO:     Inferrencing properties for moe grouped matmul bwd node "
                                << attributes.name);

        attributes.fill_from_context(context);

        auto doutput_tensor     = attributes.inputs[Moe_grouped_matmul_bwd_attributes::input_names::DOutput];
        auto token_tensor       = attributes.inputs[Moe_grouped_matmul_bwd_attributes::input_names::Token];
        auto token_index_tensor = attributes.inputs[Moe_grouped_matmul_bwd_attributes::input_names::FirstTokenOffset];
        auto dweight_tensor     = attributes.outputs[Moe_grouped_matmul_bwd_attributes::output_names::DWeight];

        auto const doutput_tensor_dim     = doutput_tensor->get_dim();
        auto const token_tensor_dim       = token_tensor->get_dim();
        auto const token_index_tensor_dim = token_index_tensor->get_dim();
        auto dweight_tensor_dim           = dweight_tensor->get_dim();

        if (dweight_tensor_dim.empty()) {
            dweight_tensor_dim.resize(3);
            dweight_tensor_dim[0] = token_index_tensor_dim[0];
            dweight_tensor_dim[1] = token_tensor_dim[2];
            dweight_tensor_dim[2] = doutput_tensor_dim[2];
            dweight_tensor->set_dim(dweight_tensor_dim);
        }

        if (dweight_tensor->get_stride().empty()) {
            auto const& dweight_dim  = dweight_tensor->get_dim();
            auto const& stride_order = detail::generate_column_major_stride_order(dweight_dim.size());
            dweight_tensor->set_stride(detail::generate_stride(dweight_dim, stride_order));
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
                    << "Building MoeGroupedMatmulBwdNode operations " << attributes.name << std::endl;
        auto cudnn_ver_error =
            error_t{error_code_t::GRAPH_NOT_SUPPORTED, "Moe grouped matmul bwd requires cuDNN v9.22.0"};

#if (CUDNN_VERSION >= 92200) && (CUDNN_VERSION < 99900)
        NV_CUDNN_FE_DYNAMIC_CHECK_CUDNN_BACKEND_VERSION(92200, cudnn_ver_error);
        CUDNN_FRONTEND_UNUSED(operations);

        auto moe_grouped_matmul_bwd_operation =
            make_shared_backend_pointer(CUDNN_BACKEND_OPERATION_MOE_GROUPED_MATMUL_BWD_DESCRIPTOR);

        cudnnDataType_t cudnn_data_type;
        _CUDNN_CHECK_CUDNN_ERROR(detail::convert_to_cudnn_type(attributes.compute_data_type, cudnn_data_type));
        _CUDNN_CHECK_CUDNN_ERROR(detail::set_attribute(moe_grouped_matmul_bwd_operation->get_backend_descriptor(),
                                                       CUDNN_ATTR_OPERATION_MOE_GROUPED_MATMUL_BWD_MATH_PREC,
                                                       CUDNN_TYPE_DATA_TYPE,
                                                       1,
                                                       &cudnn_data_type));

        auto token         = attributes.inputs.find(Moe_grouped_matmul_bwd_attributes::input_names::Token)->second;
        auto backend_token = tensors[token->get_uid()]->get_desc()->get_backend_descriptor();
        _CUDNN_CHECK_CUDNN_ERROR(detail::set_attribute(moe_grouped_matmul_bwd_operation->get_backend_descriptor(),
                                                       CUDNN_ATTR_OPERATION_MOE_GROUPED_MATMUL_BWD_TOKEN_DESC,
                                                       CUDNN_TYPE_BACKEND_DESCRIPTOR,
                                                       1,
                                                       &backend_token));

        auto dweight = attributes.outputs.find(Moe_grouped_matmul_bwd_attributes::output_names::DWeight)->second;
        auto backend_dweight = tensors[dweight->get_uid()]->get_desc()->get_backend_descriptor();
        _CUDNN_CHECK_CUDNN_ERROR(detail::set_attribute(moe_grouped_matmul_bwd_operation->get_backend_descriptor(),
                                                       CUDNN_ATTR_OPERATION_MOE_GROUPED_MATMUL_BWD_DWEIGHT_DESC,
                                                       CUDNN_TYPE_BACKEND_DESCRIPTOR,
                                                       1,
                                                       &backend_dweight));

        auto first_token_offset =
            attributes.inputs.find(Moe_grouped_matmul_bwd_attributes::input_names::FirstTokenOffset)->second;
        auto backend_first_token_offset = tensors[first_token_offset->get_uid()]->get_desc()->get_backend_descriptor();
        _CUDNN_CHECK_CUDNN_ERROR(
            detail::set_attribute(moe_grouped_matmul_bwd_operation->get_backend_descriptor(),
                                  CUDNN_ATTR_OPERATION_MOE_GROUPED_MATMUL_BWD_FIRST_TOKEN_OFFSET_DESC,
                                  CUDNN_TYPE_BACKEND_DESCRIPTOR,
                                  1,
                                  &backend_first_token_offset));

        auto doutput         = attributes.inputs.find(Moe_grouped_matmul_bwd_attributes::input_names::DOutput)->second;
        auto backend_doutput = tensors[doutput->get_uid()]->get_desc()->get_backend_descriptor();
        _CUDNN_CHECK_CUDNN_ERROR(detail::set_attribute(moe_grouped_matmul_bwd_operation->get_backend_descriptor(),
                                                       CUDNN_ATTR_OPERATION_MOE_GROUPED_MATMUL_BWD_DOUTPUT_DESC,
                                                       CUDNN_TYPE_BACKEND_DESCRIPTOR,
                                                       1,
                                                       &backend_doutput));

        _CUDNN_CHECK_CUDNN_ERROR(detail::finalize(moe_grouped_matmul_bwd_operation->get_backend_descriptor()));

        raw_operations.push_back(moe_grouped_matmul_bwd_operation);

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
        j.update(R"( {"tag": "MOE_GROUPED_MATMUL_BWD"})"_json);
    }
#endif
};

inline void
INode::moe_grouped_matmul_bwd(std::shared_ptr<Tensor_attributes> doutput,
                              std::shared_ptr<Tensor_attributes> token,
                              std::shared_ptr<Tensor_attributes> first_token_offset,
                              Moe_grouped_matmul_bwd_attributes attributes,
                              std::shared_ptr<Tensor_attributes> dweight) {
    attributes.inputs[Moe_grouped_matmul_bwd_attributes::input_names::DOutput]          = doutput;
    attributes.inputs[Moe_grouped_matmul_bwd_attributes::input_names::Token]            = token;
    attributes.inputs[Moe_grouped_matmul_bwd_attributes::input_names::FirstTokenOffset] = first_token_offset;
    attributes.outputs[Moe_grouped_matmul_bwd_attributes::output_names::DWeight]        = dweight;
    sub_nodes.emplace_back(std::make_unique<MoeGroupedMatmulBwdNode>(std::move(attributes), context));
}

}  // namespace cudnn_frontend::graph