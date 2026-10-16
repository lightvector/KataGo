#pragma once

#include "../../cudnn_frontend_Logging.h"
#include "../../cudnn_frontend_shim.h"

#include "../graph_helpers.h"
#include "../node_interface.h"

namespace cudnn_frontend {

namespace graph {

class BlockScaleQuantizeNode : public NodeCRTP<BlockScaleQuantizeNode> {
   public:
    Block_scale_quantize_attributes attributes;

    BlockScaleQuantizeNode(Block_scale_quantize_attributes&& attributes_, detail::Context const& context)
        : NodeCRTP(context), attributes(std::move(attributes_)) {}

    Type
    getType() override final {
        return Type::BLOCK_SCALE_QUANTIZE;
    }

    error_t
    pre_validate_node() const override final {
        getLogger() << "[cudnn_frontend] INFO: " << "Validating BlockScaleQuantizeNode " << attributes.name
                    << std::endl;

        RETURN_CUDNN_FRONTEND_ERROR_IF(
            !attributes.block_size.has_value(), error_code_t::ATTRIBUTE_NOT_SET, "Block size not set.");

        return {error_code_t::OK, ""};
    }

    error_t
    infer_properties_node() override final {
        getLogger() << "[cudnn_frontend] INFO: Inferencing properties for BlockScaleQuantizeNode " << attributes.name
                    << std::endl;

        attributes.fill_from_context(context);

        auto X     = attributes.inputs[Block_scale_quantize_attributes::input_names::X];
        auto Y     = attributes.outputs[Block_scale_quantize_attributes::output_names::Y];
        auto scale = attributes.outputs[Block_scale_quantize_attributes::output_names::scale];

        // Block scale quantize requires the block scale axis to be packed
        auto infer_strides_transposed = [&X](std::shared_ptr<Tensor_attributes>& T,
                                             std::optional<int64_t> const& axis) {
            auto const& dim      = T->get_dim();
            auto const& X_dim    = X->get_dim();
            auto const& X_stride = X->get_stride();

            std::vector<int64_t> indices(X_stride.size());
            std::iota(indices.begin(), indices.end(), 0);
            // Sort indices based on stride values in descending order
            std::sort(indices.begin(), indices.end(), [&X_dim, &X_stride](int64_t i, int64_t j) {
                // Prioritize singleton dimensions
                if (X_stride[i] == X_stride[j]) {
                    return (X_dim[i] == 1) || (X_dim[j] != 1);
                }
                return X_stride[i] < X_stride[j];
            });
            if (axis) {
                // Rotate left until the axis is the packed dim
                std::rotate(indices.begin(), std::find(indices.begin(), indices.end(), axis.value()), indices.end());
            }
            std::vector<int64_t> stride_order(X_stride.size());
            for (size_t i = 0; i < indices.size(); ++i) {
                stride_order[indices[i]] = i;
            }
            T->set_stride(detail::generate_stride(dim, stride_order));
        };

        // Only infer dims and strides if user did not set them
        if (Y->get_dim().empty()) {
            Y->set_dim(X->get_dim());
        }
        if (Y->get_stride().empty()) {
            if (attributes.transpose) {
                infer_strides_transposed(Y, attributes.axis);
            } else {
                Y->set_stride(X->get_stride());
            }
        }

        // Only infer dims and strides if user did not set them
        if (scale->get_dim().empty()) {
            auto scale_dim = X->get_dim();
            if (attributes.axis) {
                scale_dim[attributes.axis.value()] /= attributes.block_size.value();
            } else {
                scale_dim.back() /= attributes.block_size.value();
            }
            scale->set_dim(scale_dim);
        }
        if (scale->get_stride().empty()) {
            if (attributes.transpose) {
                infer_strides_transposed(scale, attributes.axis);
            } else {
                auto const& scale_dim = scale->get_dim();
                std::vector<int64_t> stride_order;
                CHECK_CUDNN_FRONTEND_ERROR(
                    detail::generate_stride_order_preserving_format(X->get_stride(), scale_dim.size(), stride_order));
                scale->set_stride(detail::generate_stride(scale_dim, stride_order));
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
        getLogger() << "[cudnn_frontend] INFO: " << "Building BlockScaleQuantizeNode operations " << attributes.name
                    << std::endl;
        auto cudnn_ver_error = error_t{error_code_t::GRAPH_NOT_SUPPORTED, "Block scale quantize requires cuDNN v9.7.0"};

#if (CUDNN_VERSION >= 90700)  // TODO: v9.99 is new feature branch; switch to release branch when ready
        NV_CUDNN_FE_DYNAMIC_CHECK_CUDNN_BACKEND_VERSION(90700, cudnn_ver_error);
        CUDNN_FRONTEND_UNUSED(operations);
        auto block_scale_quantize_operation = make_shared_backend_pointer(
            (cudnnBackendDescriptorType_t)CUDNN_BACKEND_OPERATION_BLOCK_SCALE_QUANTIZE_DESCRIPTOR);

        auto X         = attributes.inputs.find(Block_scale_quantize_attributes::input_names::X)->second;
        auto backend_x = tensors[X->get_uid()]->get_desc()->get_backend_descriptor();
        _CUDNN_CHECK_CUDNN_ERROR(detail::set_attribute(block_scale_quantize_operation->get_backend_descriptor(),
                                                       CUDNN_ATTR_OPERATION_BLOCK_SCALE_QUANTIZE_XDESC,
                                                       CUDNN_TYPE_BACKEND_DESCRIPTOR,
                                                       1,
                                                       &backend_x));

        auto Y         = attributes.outputs.find(Block_scale_quantize_attributes::output_names::Y)->second;
        auto backend_y = tensors[Y->get_uid()]->get_desc()->get_backend_descriptor();
        _CUDNN_CHECK_CUDNN_ERROR(detail::set_attribute(block_scale_quantize_operation->get_backend_descriptor(),
                                                       CUDNN_ATTR_OPERATION_BLOCK_SCALE_QUANTIZE_YDESC,
                                                       CUDNN_TYPE_BACKEND_DESCRIPTOR,
                                                       1,
                                                       &backend_y));

        auto scale         = attributes.outputs.find(Block_scale_quantize_attributes::output_names::scale)->second;
        auto backend_scale = tensors[scale->get_uid()]->get_desc()->get_backend_descriptor();
        _CUDNN_CHECK_CUDNN_ERROR(detail::set_attribute(block_scale_quantize_operation->get_backend_descriptor(),
                                                       CUDNN_ATTR_OPERATION_BLOCK_SCALE_QUANTIZE_SCALE_DESC,
                                                       CUDNN_TYPE_BACKEND_DESCRIPTOR,
                                                       1,
                                                       &backend_scale));

        cudnnDataType_t cudnn_data_type;
        _CUDNN_CHECK_CUDNN_ERROR(detail::convert_to_cudnn_type(attributes.compute_data_type, cudnn_data_type));
        _CUDNN_CHECK_CUDNN_ERROR(detail::set_attribute(block_scale_quantize_operation->get_backend_descriptor(),
                                                       CUDNN_ATTR_OPERATION_BLOCK_SCALE_QUANTIZE_MATH_PREC,
                                                       CUDNN_TYPE_DATA_TYPE,
                                                       1,
                                                       &cudnn_data_type));

        int32_t block_size = attributes.block_size.value();
        _CUDNN_CHECK_CUDNN_ERROR(detail::set_attribute(block_scale_quantize_operation->get_backend_descriptor(),
                                                       CUDNN_ATTR_OPERATION_BLOCK_SCALE_QUANTIZE_BLOCK_SIZE,
                                                       CUDNN_TYPE_INT32,
                                                       1,
                                                       &block_size));

        _CUDNN_CHECK_CUDNN_ERROR(detail::finalize(block_scale_quantize_operation->get_backend_descriptor()));

        raw_operations.push_back(block_scale_quantize_operation);

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
        j.update(R"( {"tag": "BLOCK_SCALE_QUANTIZE"})"_json);
    }
#endif
};

inline void
INode::block_scale_quantize(std::shared_ptr<Tensor_attributes> x,
                            Block_scale_quantize_attributes attributes,
                            std::shared_ptr<Tensor_attributes> y,
                            std::shared_ptr<Tensor_attributes> scale) {
    attributes.inputs[Block_scale_quantize_attributes::input_names::X]       = x;
    attributes.outputs[Block_scale_quantize_attributes::output_names::Y]     = y;
    attributes.outputs[Block_scale_quantize_attributes::output_names::scale] = scale;
    sub_nodes.emplace_back(std::make_unique<BlockScaleQuantizeNode>(std::move(attributes), context));
}

}  // namespace graph

}  // namespace cudnn_frontend