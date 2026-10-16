#pragma once

#include "../graph_helpers.h"
#include "../node_interface.h"

#include "pointwise.h"
#include "reduction.h"

namespace cudnn_frontend::graph {

class PagedCacheLoadNode : public NodeCRTP<PagedCacheLoadNode> {
   public:
    PagedCacheLoad_attributes attributes;

    PagedCacheLoadNode(PagedCacheLoad_attributes&& attributes_, detail::Context const& context)
        : NodeCRTP(context), attributes(std::move(attributes_)) {}

    Type
    getType() override final {
        return Type::PAGED_CACHE_LOAD;
    }

    error_t
    create_cudnn_operations(
        std::unordered_set<Tensor_attributes::uid_t>& uids_involved_in_operations,
        std::vector<std::shared_ptr<cudnn_frontend::Operation>>& operations,
        managed_backend_descriptor_t& raw_operations,
        std::unordered_map<int64_t, std::shared_ptr<cudnn_frontend::Tensor>>& tensors) const override final {
        CUDNN_FRONTEND_UNUSED(raw_operations);
        CUDNN_FE_LOG_LABEL("INFO: " << "Building PagedCacheLoadNode operations " << attributes.name << " ");
        auto cudnn_ver_error = error_t{error_code_t::GRAPH_NOT_SUPPORTED, "Paged cache load requires cuDNN v9.5.0"};

#if (CUDNN_VERSION >= 90500)
        NV_CUDNN_FE_DYNAMIC_CHECK_CUDNN_BACKEND_VERSION(90500, cudnn_ver_error);

        // Create operation by directly calling cuDNN backend API
        Operation_v8 paged_cache_load_operation;

        _CUDNN_CHECK_CUDNN_ERROR(paged_cache_load_operation.initialize_managed_backend_pointer(
            CUDNN_BACKEND_OPERATION_PAGED_CACHE_LOAD_DESCRIPTOR));

        // Set container tensor
        CUDNN_FE_VALIDATE_AND_ASSIGN_INPUT_TENSOR(container, PagedCacheLoad_attributes::input_names::container);
        auto container_desc = tensors.at(container->second->get_uid())->get_raw_desc();
        _CUDNN_CHECK_CUDNN_ERROR(detail::set_attribute(paged_cache_load_operation.get_raw_desc(),
                                                       CUDNN_ATTR_OPERATION_PAGED_CACHE_LOAD_CONTAINER_DESC,
                                                       CUDNN_TYPE_BACKEND_DESCRIPTOR,
                                                       1,
                                                       &container_desc));

        // Set page table tensor
        CUDNN_FE_VALIDATE_AND_ASSIGN_INPUT_TENSOR(pageTable, PagedCacheLoad_attributes::input_names::pageTable);
        auto page_table_desc = tensors.at(pageTable->second->get_uid())->get_raw_desc();
        _CUDNN_CHECK_CUDNN_ERROR(detail::set_attribute(paged_cache_load_operation.get_raw_desc(),
                                                       CUDNN_ATTR_OPERATION_PAGED_CACHE_LOAD_PAGE_TABLE_DESC,
                                                       CUDNN_TYPE_BACKEND_DESCRIPTOR,
                                                       1,
                                                       &page_table_desc));

        // Set sequence length tensor
        CUDNN_FE_VALIDATE_AND_ASSIGN_INPUT_TENSOR(seqLen, PagedCacheLoad_attributes::input_names::seqLen);
        auto seq_len_desc = tensors.at(seqLen->second->get_uid())->get_raw_desc();
        _CUDNN_CHECK_CUDNN_ERROR(detail::set_attribute(paged_cache_load_operation.get_raw_desc(),
                                                       CUDNN_ATTR_OPERATION_PAGED_CACHE_LOAD_SEQUENCE_DESC,
                                                       CUDNN_TYPE_BACKEND_DESCRIPTOR,
                                                       1,
                                                       &seq_len_desc));

        // Set output tensor Y
        CUDNN_FE_VALIDATE_AND_ASSIGN_OUTPUT_TENSOR(yOut, PagedCacheLoad_attributes::output_names::yOut);
        auto y_desc = tensors.at(yOut->second->get_uid())->get_raw_desc();

        _CUDNN_CHECK_CUDNN_ERROR(detail::set_attribute(paged_cache_load_operation.get_raw_desc(),
                                                       CUDNN_ATTR_OPERATION_PAGED_CACHE_LOAD_YDESC,
                                                       CUDNN_TYPE_BACKEND_DESCRIPTOR,
                                                       1,
                                                       &y_desc));

        _CUDNN_CHECK_CUDNN_ERROR(detail::finalize(paged_cache_load_operation.get_raw_desc()));

        operations.push_back(std::make_shared<Operation_v8>(std::move(paged_cache_load_operation)));

        auto const& non_virtual_uids = attributes.get_non_virtual_uids();
        uids_involved_in_operations.insert(non_virtual_uids.begin(), non_virtual_uids.end());

        return {error_code_t::OK, ""};
#else
        CUDNN_FRONTEND_UNUSED(uids_involved_in_operations);
        CUDNN_FRONTEND_UNUSED(operations);
        CUDNN_FRONTEND_UNUSED(tensors);
        return cudnn_ver_error;
#endif
    }

    error_t
    pre_validate_node() const override final {
        CUDNN_FE_LOG_LABEL_ENDL("INFO: Validating PagedCacheLoadNode " << attributes.name);

        RETURN_CUDNN_FRONTEND_ERROR_IF(detail::get_backend_version() < 90500 || detail::get_compiled_version() < 90500,
                                       error_code_t::CUDNN_BACKEND_API_FAILED,
                                       "The cuDNN backend version must be at least 9.5.0 at compile time and runtime "
                                       "in order to use PagedCacheLoadNode.");

        auto const yOut_dims       = attributes.outputs.at(PagedCacheLoad_attributes::output_names::yOut)->get_dim();
        auto const yOut_strides    = attributes.outputs.at(PagedCacheLoad_attributes::output_names::yOut)->get_stride();
        auto const container_dims  = attributes.inputs.at(PagedCacheLoad_attributes::input_names::container)->get_dim();
        auto const blockTable_dims = attributes.inputs.at(PagedCacheLoad_attributes::input_names::pageTable)->get_dim();

        // In the backend, the k-cache is passed as K^T and has dims [B,H,D,S], while v-cache has dims [B,H,S,D]
        // Use the strides to distinguish.
        auto yIsTransposed = yOut_strides[2] == 1;
        auto s_kv          = !yIsTransposed ? yOut_dims[2] : yOut_dims[3];

        auto block_size       = container_dims[2];
        auto block_table_size = blockTable_dims[2];
        bool is_block_table_packed =
            attributes.inputs.at(PagedCacheLoad_attributes::input_names::pageTable)->get_ragged_offset() != nullptr;

        RETURN_CUDNN_FRONTEND_ERROR_IF(
            !is_block_table_packed && (s_kv + (block_size - 1)) / block_size != block_table_size,
            error_code_t::INVALID_VALUE,
            "Paged cache load: block table size must equal ceil(s_kv/block_size), except when using packed block "
            "tables");

        return {error_code_t::OK, ""};
    }

    error_t
    infer_properties_node() override final {
        return {error_code_t::OK, ""};
    }

#ifndef CUDNN_FRONTEND_SKIP_JSON_LIB
    virtual void
    serialize(json& j) const override final {
        j = attributes;
    }
#endif
};

inline void
INode::paged_cache_load(std::shared_ptr<Tensor_attributes> container,
                        std::shared_ptr<Tensor_attributes> seqLen,
                        std::shared_ptr<Tensor_attributes> pageTable,
                        PagedCacheLoad_attributes attributes,
                        std::shared_ptr<Tensor_attributes> yOut) {
    attributes.inputs[PagedCacheLoad_attributes::input_names::container] = std::move(container);
    attributes.inputs[PagedCacheLoad_attributes::input_names::seqLen]    = std::move(seqLen);
    attributes.inputs[PagedCacheLoad_attributes::input_names::pageTable] = std::move(pageTable);
    attributes.outputs[PagedCacheLoad_attributes::output_names::yOut]    = std::move(yOut);
    sub_nodes.emplace_back(std::make_unique<PagedCacheLoadNode>(std::move(attributes), context));
}
}  // namespace cudnn_frontend::graph