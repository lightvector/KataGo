#pragma once

#include "../graph_helpers.h"
#include "../node_interface.h"

namespace cudnn_frontend {

namespace graph {
class RoPENode : public NodeCRTP<RoPENode> {
   public:
    RoPE_attributes attributes;

    RoPENode(RoPE_attributes&& attributes_, detail::Context const& context)
        : NodeCRTP(context), attributes(std::move(attributes_)) {}

    Type
    getType() override final {
        return Type::ROPE;
    }

    error_t
    infer_properties_node() override final {
        CUDNN_FE_LOG_LABEL_ENDL("INFO: Inferencing properties for RoPE node " << attributes.name);

        attributes.fill_from_context(context);

        auto INPUT  = attributes.inputs[RoPE_attributes::input_names::INPUT];
        auto OUTPUT = attributes.outputs[RoPE_attributes::output_names::OUTPUT];

        if (OUTPUT->get_dim().empty()) {
            OUTPUT->set_dim(INPUT->get_dim());
        }
        if (OUTPUT->get_stride().empty()) {
            OUTPUT->set_stride(INPUT->get_stride());
        }

        return {error_code_t::OK, ""};
    }

    error_t
    pre_validate_node() const override final {
        CUDNN_FE_LOG_LABEL_ENDL("INFO: Validating RoPENode " << attributes.name);

        RETURN_CUDNN_FRONTEND_ERROR_IF(
            attributes.inputs.find(RoPE_attributes::input_names::INPUT) == attributes.inputs.end(),
            error_code_t::ATTRIBUTE_NOT_SET,
            "RoPE node requires INPUT tensor.");

        RETURN_CUDNN_FRONTEND_ERROR_IF(
            attributes.inputs.find(RoPE_attributes::input_names::FREQS) == attributes.inputs.end(),
            error_code_t::ATTRIBUTE_NOT_SET,
            "RoPE node requires FREQS tensor.");

        return {error_code_t::OK, ""};
    }

    error_t
    create_cudnn_operations(
        std::unordered_set<Tensor_attributes::uid_t>& uids_involved_in_operations,
        std::vector<std::shared_ptr<cudnn_frontend::Operation>>& operations,
        managed_backend_descriptor_t& raw_operations,
        std::unordered_map<int64_t, std::shared_ptr<cudnn_frontend::Tensor>>& tensors) const override final {
        CUDNN_FRONTEND_UNUSED(operations);
        CUDNN_FE_LOG_LABEL("INFO: Building RoPENode operations " << attributes.name << " ");

#if (CUDNN_VERSION >= 92400)
        // Compile- and run-time checks: RoPE op was added in cuDNN 9.24.
        RETURN_CUDNN_FRONTEND_ERROR_IF(
            detail::get_backend_version() < 92400,
            error_code_t::GRAPH_NOT_SUPPORTED,
            "CUDNN_BACKEND_OPERATION_ROPE_FWD_DESCRIPTOR is only available starting cuDNN 9.24.");

        auto rope_operation =
            make_shared_backend_pointer((cudnnBackendDescriptorType_t)CUDNN_BACKEND_OPERATION_ROPE_FWD_DESCRIPTOR);

        // Set input X
        auto X         = attributes.inputs.find(RoPE_attributes::input_names::INPUT)->second;
        auto backend_x = tensors[X->get_uid()]->get_desc()->get_backend_descriptor();
        _CUDNN_CHECK_CUDNN_ERROR(detail::set_attribute(rope_operation->get_backend_descriptor(),
                                                       CUDNN_ATTR_OPERATION_ROPE_FWD_XDESC,
                                                       CUDNN_TYPE_BACKEND_DESCRIPTOR,
                                                       1,
                                                       &backend_x));

        // Set freqs
        auto FREQS         = attributes.inputs.find(RoPE_attributes::input_names::FREQS)->second;
        auto backend_freqs = tensors[FREQS->get_uid()]->get_desc()->get_backend_descriptor();
        _CUDNN_CHECK_CUDNN_ERROR(detail::set_attribute(rope_operation->get_backend_descriptor(),
                                                       CUDNN_ATTR_OPERATION_ROPE_FWD_FREQSDESC,
                                                       CUDNN_TYPE_BACKEND_DESCRIPTOR,
                                                       1,
                                                       &backend_freqs));

        // Set output Y
        auto Y         = attributes.outputs.find(RoPE_attributes::output_names::OUTPUT)->second;
        auto backend_y = tensors[Y->get_uid()]->get_desc()->get_backend_descriptor();
        _CUDNN_CHECK_CUDNN_ERROR(detail::set_attribute(rope_operation->get_backend_descriptor(),
                                                       CUDNN_ATTR_OPERATION_ROPE_FWD_YDESC,
                                                       CUDNN_TYPE_BACKEND_DESCRIPTOR,
                                                       1,
                                                       &backend_y));

        // Set output_scale (host fp32 scalar; default 1.0)
        _CUDNN_CHECK_CUDNN_ERROR(detail::set_attribute(rope_operation->get_backend_descriptor(),
                                                       CUDNN_ATTR_OPERATION_ROPE_FWD_OUTPUT_SCALE,
                                                       CUDNN_TYPE_FLOAT,
                                                       1,
                                                       &attributes.output_scale));

        // Set rope_dim (int64; 0 = full rotation = input's head_dim)
        _CUDNN_CHECK_CUDNN_ERROR(detail::set_attribute(rope_operation->get_backend_descriptor(),
                                                       CUDNN_ATTR_OPERATION_ROPE_FWD_ROPE_DIM,
                                                       CUDNN_TYPE_INT64,
                                                       1,
                                                       &attributes.rope_dim));

        // Finalize
        _CUDNN_CHECK_CUDNN_ERROR(detail::finalize(rope_operation->get_backend_descriptor()));

        auto const& non_virtual_uids = attributes.get_non_virtual_uids();
        uids_involved_in_operations.insert(non_virtual_uids.begin(), non_virtual_uids.end());

        raw_operations.push_back(rope_operation);

        return {error_code_t::OK, ""};
#else
        CUDNN_FRONTEND_UNUSED(uids_involved_in_operations);
        CUDNN_FRONTEND_UNUSED(raw_operations);
        CUDNN_FRONTEND_UNUSED(tensors);
        return {error_code_t::GRAPH_NOT_SUPPORTED,
                "CUDNN_BACKEND_OPERATION_ROPE_FWD_DESCRIPTOR is only available starting cuDNN 9.24."};
#endif
    }

#ifndef CUDNN_FRONTEND_SKIP_JSON_LIB
    virtual void
    serialize(json& j) const override final {
        j = attributes;
        j.update(R"({"tag": "ROPE"})"_json);
    }
#endif
};

}  // namespace graph

}  // namespace cudnn_frontend
