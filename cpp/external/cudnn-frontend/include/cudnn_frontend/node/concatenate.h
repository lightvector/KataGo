#pragma once

#include "../../cudnn_frontend_Logging.h"
#include "../../cudnn_frontend_shim.h"

#include "../graph_helpers.h"
#include "../node_interface.h"
#include <string>

namespace cudnn_frontend {

namespace graph {

class ConcatenateNode : public NodeCRTP<ConcatenateNode> {
   public:
    Concatenate_attributes attributes;

    ConcatenateNode(Concatenate_attributes&& attributes_, detail::Context const& context)
        : NodeCRTP(context), attributes(std::move(attributes_)) {}

    Type
    getType() override final {
        return Type::CONCATENATE;
    }

    error_t
    pre_validate_node() const override final {
        getLogger() << "[cudnn_frontend] INFO: " << "Validating ConcatenateNode " << attributes.name << std::endl;

        RETURN_CUDNN_FRONTEND_ERROR_IF(!attributes.axis.has_value(), error_code_t::ATTRIBUTE_NOT_SET, "Axis not set\n");

        auto X = attributes.inputs;

        RETURN_CUDNN_FRONTEND_ERROR_IF(
            (X.size() == 0), error_code_t::INVALID_VALUE, "Input size of the concatenate node cannot be zero\n");

        return {error_code_t::OK, ""};
    }

    error_t
    infer_properties_node() override final {
        getLogger() << "[cudnn_frontend] INFO: Inferring properties for ConcatenateNode " << attributes.name
                    << std::endl;

        attributes.fill_from_context(context);

        auto Y = attributes.outputs[Concatenate_attributes::output_names::Y];

        // Infer dims and strides only if user did not set them
        int64_t dim_sum = 0;
        for (const auto& input : attributes.inputs) {
            dim_sum += input->get_dim()[attributes.axis.value()];
        }

        auto X                        = attributes.inputs[0];
        auto dims                     = X->get_dim();
        dims[attributes.axis.value()] = dim_sum;

        if (Y->get_dim().empty()) {
            Y->set_dim(dims);
            Y->set_dim(dims);
        }

        if (Y->get_stride().empty()) {
            std::vector<int64_t> stride_order;
            CHECK_CUDNN_FRONTEND_ERROR(
                detail::generate_stride_order_preserving_format(X->get_stride(), dims.size(), stride_order));
            Y->set_stride(detail::generate_stride(dims, stride_order));
        }

        return {error_code_t::OK, ""};
    }

    error_t
    create_cudnn_operations(
        std::unordered_set<Tensor_attributes::uid_t>& uids_involved_in_operations,
        std::vector<std::shared_ptr<cudnn_frontend::Operation>>& operations,
        managed_backend_descriptor_t& raw_operations,
        std::unordered_map<int64_t, std::shared_ptr<cudnn_frontend::Tensor>>& tensors) const override final {
        getLogger() << "[cudnn_frontend] INFO: " << "Building ConcatenateNode operations " << attributes.name
                    << std::endl;
        auto cudnn_ver_error = error_t{error_code_t::GRAPH_NOT_SUPPORTED, "Concatenate requires cuDNN v9.7.0"};

#if (CUDNN_VERSION >= 90700)
        NV_CUDNN_FE_DYNAMIC_CHECK_CUDNN_BACKEND_VERSION(90700, cudnn_ver_error);
        CUDNN_FRONTEND_UNUSED(operations);
        auto concatenate_operation = make_shared_backend_pointer(CUDNN_BACKEND_OPERATION_CONCAT_DESCRIPTOR);

        std::vector<void*> backend_x(attributes.inputs.size());
        size_t index = 0;
        for (const auto& input : attributes.inputs) {
            backend_x[index] = tensors[input->get_uid()]->get_desc()->get_backend_descriptor();
            index++;
        }
        _CUDNN_CHECK_CUDNN_ERROR(detail::set_attribute(concatenate_operation->get_backend_descriptor(),
                                                       CUDNN_ATTR_OPERATION_CONCAT_INPUT_DESCS,
                                                       CUDNN_TYPE_BACKEND_DESCRIPTOR,
                                                       attributes.inputs.size(),
                                                       backend_x.data()));

        auto Y         = attributes.outputs.find(Concatenate_attributes::output_names::Y)->second;
        auto backend_y = tensors[Y->get_uid()]->get_desc()->get_backend_descriptor();
        _CUDNN_CHECK_CUDNN_ERROR(detail::set_attribute(concatenate_operation->get_backend_descriptor(),
                                                       CUDNN_ATTR_OPERATION_CONCAT_OUTPUT_DESC,
                                                       CUDNN_TYPE_BACKEND_DESCRIPTOR,
                                                       1,
                                                       &backend_y));

        int64_t axis = attributes.axis.value();
        _CUDNN_CHECK_CUDNN_ERROR(detail::set_attribute(concatenate_operation->get_backend_descriptor(),
                                                       CUDNN_ATTR_OPERATION_CONCAT_AXIS,
                                                       CUDNN_TYPE_INT64,
                                                       1,
                                                       &axis));

        if (attributes.in_place_index.has_value()) {
            int64_t in_place_index = attributes.in_place_index.value();
            _CUDNN_CHECK_CUDNN_ERROR(detail::set_attribute(concatenate_operation->get_backend_descriptor(),
                                                           CUDNN_ATTR_OPERATION_CONCAT_INPLACE_INDEX,
                                                           CUDNN_TYPE_INT64,
                                                           1,
                                                           &in_place_index));
        }

        _CUDNN_CHECK_CUDNN_ERROR(detail::finalize(concatenate_operation->get_backend_descriptor()));

        raw_operations.push_back(concatenate_operation);

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
        j.update(R"( {"tag": "CONCATENATE"})"_json);
    }
#endif
};

inline void
INode::concatenate(std::vector<std::shared_ptr<Tensor_attributes>> x,
                   Concatenate_attributes attributes,
                   std::shared_ptr<Tensor_attributes> y) {
    for (auto& element : x) {
        attributes.inputs.push_back(element);
    }
    attributes.outputs[Concatenate_attributes::output_names::Y] = y;
    sub_nodes.emplace_back(std::make_unique<ConcatenateNode>(std::move(attributes), context));
}

}  // namespace graph

}  // namespace cudnn_frontend
