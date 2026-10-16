#pragma once

#include "../graph_helpers.h"
#include "../node_interface.h"

namespace cudnn_frontend {

namespace graph {
class RoPEBackwardNode : public NodeCRTP<RoPEBackwardNode> {
   public:
    RoPE_backward_attributes attributes;

    RoPEBackwardNode(RoPE_backward_attributes&& attributes_, detail::Context const& context)
        : NodeCRTP(context), attributes(std::move(attributes_)) {}

    Type
    getType() override final {
        return Type::ROPE_BWD;
    }

    error_t
    infer_properties_node() override final {
        CUDNN_FE_LOG_LABEL_ENDL("INFO: Inferencing properties for RoPE backward node " << attributes.name);

        attributes.fill_from_context(context);

        auto DY = attributes.inputs[RoPE_backward_attributes::input_names::DY];
        auto DX = attributes.outputs[RoPE_backward_attributes::output_names::DX];

        if (DX->get_dim().empty()) {
            DX->set_dim(DY->get_dim());
        }
        if (DX->get_stride().empty()) {
            DX->set_stride(DY->get_stride());
        }

        return {error_code_t::OK, ""};
    }

    error_t
    pre_validate_node() const override final {
        CUDNN_FE_LOG_LABEL_ENDL("INFO: Validating RoPEBackwardNode " << attributes.name);

        RETURN_CUDNN_FRONTEND_ERROR_IF(
            attributes.inputs.find(RoPE_backward_attributes::input_names::DY) == attributes.inputs.end(),
            error_code_t::ATTRIBUTE_NOT_SET,
            "RoPE backward node requires DY (input gradient) tensor.");

        RETURN_CUDNN_FRONTEND_ERROR_IF(
            attributes.inputs.find(RoPE_backward_attributes::input_names::FREQS) == attributes.inputs.end(),
            error_code_t::ATTRIBUTE_NOT_SET,
            "RoPE backward node requires FREQS tensor.");

        return {error_code_t::OK, ""};
    }

    error_t
    create_cudnn_operations(
        std::unordered_set<Tensor_attributes::uid_t>& uids_involved_in_operations,
        std::vector<std::shared_ptr<cudnn_frontend::Operation>>& operations,
        managed_backend_descriptor_t& raw_operations,
        std::unordered_map<int64_t, std::shared_ptr<cudnn_frontend::Tensor>>& tensors) const override final {
        CUDNN_FRONTEND_UNUSED(operations);
        CUDNN_FE_LOG_LABEL("INFO: Building RoPEBackwardNode operations " << attributes.name << " ");

#if (CUDNN_VERSION >= 92400)
        // Compile- and run-time checks: RoPE bwd op was added in cuDNN 9.24.
        RETURN_CUDNN_FRONTEND_ERROR_IF(
            detail::get_backend_version() < 92400,
            error_code_t::GRAPH_NOT_SUPPORTED,
            "CUDNN_BACKEND_OPERATION_ROPE_BWD_DESCRIPTOR is only available starting cuDNN 9.24.");

        auto rope_bwd_operation =
            make_shared_backend_pointer((cudnnBackendDescriptorType_t)CUDNN_BACKEND_OPERATION_ROPE_BWD_DESCRIPTOR);

        // Set DY (input gradient)
        auto DY         = attributes.inputs.find(RoPE_backward_attributes::input_names::DY)->second;
        auto backend_dy = tensors[DY->get_uid()]->get_desc()->get_backend_descriptor();
        _CUDNN_CHECK_CUDNN_ERROR(detail::set_attribute(rope_bwd_operation->get_backend_descriptor(),
                                                       CUDNN_ATTR_OPERATION_ROPE_BWD_DYDESC,
                                                       CUDNN_TYPE_BACKEND_DESCRIPTOR,
                                                       1,
                                                       &backend_dy));

        // Set freqs
        auto FREQS         = attributes.inputs.find(RoPE_backward_attributes::input_names::FREQS)->second;
        auto backend_freqs = tensors[FREQS->get_uid()]->get_desc()->get_backend_descriptor();
        _CUDNN_CHECK_CUDNN_ERROR(detail::set_attribute(rope_bwd_operation->get_backend_descriptor(),
                                                       CUDNN_ATTR_OPERATION_ROPE_BWD_FREQSDESC,
                                                       CUDNN_TYPE_BACKEND_DESCRIPTOR,
                                                       1,
                                                       &backend_freqs));

        // Set DX (output gradient)
        auto DX         = attributes.outputs.find(RoPE_backward_attributes::output_names::DX)->second;
        auto backend_dx = tensors[DX->get_uid()]->get_desc()->get_backend_descriptor();
        _CUDNN_CHECK_CUDNN_ERROR(detail::set_attribute(rope_bwd_operation->get_backend_descriptor(),
                                                       CUDNN_ATTR_OPERATION_ROPE_BWD_DXDESC,
                                                       CUDNN_TYPE_BACKEND_DESCRIPTOR,
                                                       1,
                                                       &backend_dx));

        _CUDNN_CHECK_CUDNN_ERROR(detail::set_attribute(rope_bwd_operation->get_backend_descriptor(),
                                                       CUDNN_ATTR_OPERATION_ROPE_BWD_OUTPUT_SCALE,
                                                       CUDNN_TYPE_FLOAT,
                                                       1,
                                                       &attributes.output_scale));

        _CUDNN_CHECK_CUDNN_ERROR(detail::set_attribute(rope_bwd_operation->get_backend_descriptor(),
                                                       CUDNN_ATTR_OPERATION_ROPE_BWD_ROPE_DIM,
                                                       CUDNN_TYPE_INT64,
                                                       1,
                                                       &attributes.rope_dim));

        _CUDNN_CHECK_CUDNN_ERROR(detail::finalize(rope_bwd_operation->get_backend_descriptor()));

        auto const& non_virtual_uids = attributes.get_non_virtual_uids();
        uids_involved_in_operations.insert(non_virtual_uids.begin(), non_virtual_uids.end());

        raw_operations.push_back(rope_bwd_operation);

        return {error_code_t::OK, ""};
#else
        CUDNN_FRONTEND_UNUSED(uids_involved_in_operations);
        CUDNN_FRONTEND_UNUSED(raw_operations);
        CUDNN_FRONTEND_UNUSED(tensors);
        return {error_code_t::GRAPH_NOT_SUPPORTED,
                "CUDNN_BACKEND_OPERATION_ROPE_BWD_DESCRIPTOR is only available starting cuDNN 9.24."};
#endif
    }

#ifndef CUDNN_FRONTEND_SKIP_JSON_LIB
    virtual void
    serialize(json& j) const override final {
        j = attributes;
        j.update(R"({"tag": "ROPE_BWD"})"_json);
    }
#endif
};

}  // namespace graph

}  // namespace cudnn_frontend
