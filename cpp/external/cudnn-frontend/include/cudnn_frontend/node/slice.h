#pragma once

#include "../graph_helpers.h"
#include "../node_interface.h"

namespace cudnn_frontend::graph {

class SliceNode : public NodeCRTP<SliceNode> {
   public:
    Slice_attributes attributes;

    SliceNode(Slice_attributes&& attributes_, detail::Context const& context)
        : NodeCRTP(context), attributes(std::move(attributes_)) {}

    Type
    getType() override final {
        return Type::SLICE;
    }

    error_t
    infer_properties_node() override final {
        getLogger() << "[cudnn_frontend] INFO:     Inferrencing properties for slice node " << attributes.name
                    << std::endl;

        attributes.fill_from_context(context);

        for (size_t i = 0; i < attributes.slice_strides.size(); ++i) {
            RETURN_CUDNN_FRONTEND_ERROR_IF(
                attributes.slice_strides[i] <= 0,
                error_code_t::INVALID_VALUE,
                "Slice slice_strides[" + std::to_string(i) + "] must be strictly positive (got " +
                    std::to_string(attributes.slice_strides[i]) +
                    "). Non-positive strides break output dimension calculation and can cause division by zero.");
        }

        auto output     = attributes.outputs.at(Slice_attributes::output_names::Y);
        auto output_dim = output->get_dim();

        if (output_dim.empty()) {
            for (size_t i = 0; i < attributes.slices.size(); ++i) {
                int64_t start  = attributes.slices[i].first;
                int64_t limit  = attributes.slices[i].second;
                int64_t stride = (!attributes.slice_strides.empty() && i < attributes.slice_strides.size())
                                     ? attributes.slice_strides[i]
                                     : 1;
                // Output dimension = ceil((limit - start) / stride)
                int64_t dim = (limit - start + stride - 1) / stride;
                output_dim.push_back(dim);
            }
            output->set_dim(output_dim);
        }

        auto const input            = attributes.inputs.at(Slice_attributes::input_names::X);
        auto const input_data_type  = input->get_data_type();
        auto const output_data_type = output->get_data_type();
        if (output_data_type == DataType_t::NOT_SET) {
            output->set_data_type(input_data_type);
        } else {
            RETURN_CUDNN_FRONTEND_ERROR_IF(output_data_type != input_data_type,
                                           error_code_t::INVALID_VALUE,
                                           "output and input tensor data types should match for slice operation.");
        }

        auto const input_stride = input->get_stride();
        if (output->get_stride().empty()) {
            // When slice strides > 1, output strides need to be multiplied accordingly
            std::vector<int64_t> output_stride;
            for (size_t i = 0; i < input_stride.size(); ++i) {
                int64_t stride = (!attributes.slice_strides.empty() && i < attributes.slice_strides.size())
                                     ? attributes.slice_strides[i]
                                     : 1;
                output_stride.push_back(input_stride[i] * stride);
            }
            output->set_stride(output_stride);
        }

        return {error_code_t::OK, ""};
    }

    error_t
    create_cudnn_tensors_node(std::unordered_map<int64_t, std::shared_ptr<cudnn_frontend::Tensor>>& tensors,
                              int64_t& potential_uid,
                              std::unordered_set<int64_t> const& used_uids) const override final {
        getLogger() << "[cudnn_frontend] INFO: Creating cudnn tensors for SliceNode " << attributes.name << std::endl;

        auto const input  = attributes.inputs.at(Slice_attributes::input_names::X);
        auto const output = attributes.outputs.at(Slice_attributes::output_names::Y);

        if (detail::get_backend_version() >= 92200 && detail::get_compiled_version() >= 92200) {
            CHECK_CUDNN_FRONTEND_ERROR(detail::create_cudnn_tensor(input, tensors, potential_uid, used_uids));
            CHECK_CUDNN_FRONTEND_ERROR(detail::create_cudnn_tensor(output, tensors, potential_uid, used_uids));
            return {error_code_t::OK, ""};
        }

        if (input->has_uid() == false) {
            detail::assign_uid(input.get(), potential_uid, used_uids);
        }
        output->set_is_virtual(false);
        CHECK_CUDNN_FRONTEND_ERROR(detail::create_cudnn_tensor(output, tensors, potential_uid, used_uids));

        return {error_code_t::OK, ""};
    }

    error_t
    create_cudnn_operations(
        std::unordered_set<Tensor_attributes::uid_t>& uids_involved_in_operations,
        std::vector<std::shared_ptr<cudnn_frontend::Operation>>& operations,
        managed_backend_descriptor_t& raw_operations,
        std::unordered_map<int64_t, std::shared_ptr<cudnn_frontend::Tensor>>& tensors) const override final {
        getLogger() << "[cudnn_frontend] INFO: " << "Building SliceNode operations " << attributes.name << std::endl;

#if (CUDNN_VERSION >= 92200)
        // cuDNN >= 9.22.0: Use native backend slice operation
        auto cudnn_ver_error = error_t{error_code_t::GRAPH_NOT_SUPPORTED, "Slice backend operation requires 9.22.0"};
        NV_CUDNN_FE_DYNAMIC_CHECK_CUDNN_BACKEND_VERSION(92200, cudnn_ver_error);

        if (detail::get_backend_version() >= 92200) {
            CUDNN_FRONTEND_UNUSED(operations);

            auto slice_operation = make_shared_backend_pointer(CUDNN_BACKEND_OPERATION_SLICE_DESCRIPTOR);

            // Set input tensor
            auto X         = attributes.inputs.at(Slice_attributes::input_names::X);
            auto backend_x = tensors[X->get_uid()]->get_desc()->get_backend_descriptor();
            _CUDNN_CHECK_CUDNN_ERROR(detail::set_attribute(slice_operation->get_backend_descriptor(),
                                                           CUDNN_ATTR_OPERATION_SLICE_XDESC,
                                                           CUDNN_TYPE_BACKEND_DESCRIPTOR,
                                                           1,
                                                           &backend_x));

            // Set output tensor
            auto Y         = attributes.outputs.at(Slice_attributes::output_names::Y);
            auto backend_y = tensors[Y->get_uid()]->get_desc()->get_backend_descriptor();
            _CUDNN_CHECK_CUDNN_ERROR(detail::set_attribute(slice_operation->get_backend_descriptor(),
                                                           CUDNN_ATTR_OPERATION_SLICE_YDESC,
                                                           CUDNN_TYPE_BACKEND_DESCRIPTOR,
                                                           1,
                                                           &backend_y));

            // Extract start and limit indices from slices
            std::vector<int64_t> start_indices;
            std::vector<int64_t> limit_indices;

            for (const auto& slice : attributes.slices) {
                start_indices.push_back(slice.first);
                limit_indices.push_back(slice.second);
            }

            // Per-dimension strides: use user slice_strides[i] when set, else 1 (preserves partial configuration)
            std::vector<int64_t> strides(attributes.slices.size());
            for (size_t i = 0; i < strides.size(); ++i) {
                strides[i] = (i < attributes.slice_strides.size()) ? attributes.slice_strides[i] : 1;
            }

            _CUDNN_CHECK_CUDNN_ERROR(detail::set_attribute(slice_operation->get_backend_descriptor(),
                                                           CUDNN_ATTR_OPERATION_SLICE_START_INDICES,
                                                           CUDNN_TYPE_INT64,
                                                           start_indices.size(),
                                                           start_indices.data()));

            _CUDNN_CHECK_CUDNN_ERROR(detail::set_attribute(slice_operation->get_backend_descriptor(),
                                                           CUDNN_ATTR_OPERATION_SLICE_LIMIT_INDICES,
                                                           CUDNN_TYPE_INT64,
                                                           limit_indices.size(),
                                                           limit_indices.data()));

            _CUDNN_CHECK_CUDNN_ERROR(detail::set_attribute(slice_operation->get_backend_descriptor(),
                                                           CUDNN_ATTR_OPERATION_SLICE_STRIDES,
                                                           CUDNN_TYPE_INT64,
                                                           strides.size(),
                                                           strides.data()));

            _CUDNN_CHECK_CUDNN_ERROR(detail::finalize(slice_operation->get_backend_descriptor()));

            raw_operations.push_back(slice_operation);

            auto const& non_virtual_uids = attributes.get_non_virtual_uids();
            uids_involved_in_operations.insert(non_virtual_uids.begin(), non_virtual_uids.end());
            return {error_code_t::OK, ""};
        }
#endif
        CUDNN_FRONTEND_UNUSED(operations);
        CUDNN_FRONTEND_UNUSED(raw_operations);
        CUDNN_FRONTEND_UNUSED(tensors);

        getLogger() << "[cudnn_frontend] INFO: " << "Using pointer arithmetic fallback for slice on cuDNN < 9.22.0"
                    << std::endl;

        auto const output = attributes.outputs.at(Slice_attributes::output_names::Y);
        if (output && output->get_is_virtual() == false) {
            uids_involved_in_operations.insert(output->get_uid());
            if (auto ragged_offset = output->get_ragged_offset()) {
                uids_involved_in_operations.insert(ragged_offset->get_uid());
            }
        }
        return {error_code_t::OK, ""};
    }

    error_t
    collect_variant_pack_replacements_node(
        std::unordered_map<Tensor_attributes::uid_t, std::pair<Tensor_attributes::uid_t, int64_t>>&
            variant_pack_replacements) const override final {
        if (detail::get_backend_version() >= 92200 && detail::get_compiled_version() >= 92200) {
            CUDNN_FRONTEND_UNUSED(variant_pack_replacements);
            return {error_code_t::OK, ""};
        }
        auto const input  = attributes.inputs.at(Slice_attributes::input_names::X);
        auto const output = attributes.outputs.at(Slice_attributes::output_names::Y);

        variant_pack_replacements[input->get_uid()] = {output->get_uid(), attributes.get_offset()};

        return {error_code_t::OK, ""};
    };

#ifndef CUDNN_FRONTEND_SKIP_JSON_LIB
    virtual void
    serialize(json& j) const override final {
        j = attributes;
        j.update(R"( {"tag": "SLICE"})"_json);
    }
#endif
};

}  // namespace cudnn_frontend::graph