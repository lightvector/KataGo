
#pragma once

#include <cstdlib>
#include <unordered_set>

#include "../../cudnn_frontend_Heuristics.h"
#include "../../cudnn_frontend_Logging.h"

#include "../graph_helpers.h"
#include "../node_interface.h"

#include "diagonal_band_mask.h"
#include "matmul.h"
#include "pointwise.h"
#include "reduction.h"
#include "rng.h"
#include "softmax.h"
#include "paged_cache_load.h"
#include "block_scale_dequantize.h"
#include "sdpa_support_surface.h"

namespace cudnn_frontend::graph {

namespace attn::score_modifiers {

// clang-format off
inline float get_negative_inf_value();

inline std::shared_ptr<Tensor_attributes> causal_mask(
    std::shared_ptr<Graph> graph,
    std::shared_ptr<Tensor_attributes> attention_score
);

inline std::shared_ptr<Tensor_attributes> bias(
    std::shared_ptr<Graph> graph,
    std::shared_ptr<Tensor_attributes> attention_score,
    std::shared_ptr<Tensor_attributes> bias_tensor
);

inline std::shared_ptr<Tensor_attributes> causal_mask_bottom_right(
    std::shared_ptr<Graph> graph,
    std::shared_ptr<Tensor_attributes> attention_score,
    std::shared_ptr<Tensor_attributes> seq_len_q,
    std::shared_ptr<Tensor_attributes> seq_len_kv
);

inline std::shared_ptr<Tensor_attributes> padding_mask(
    std::shared_ptr<Graph> graph,
    std::shared_ptr<Tensor_attributes> attention_score,
    std::shared_ptr<Tensor_attributes> seq_len_kv,
    std::shared_ptr<Tensor_attributes> seq_len_q
);

inline std::shared_ptr<Tensor_attributes> sliding_window_mask(
    std::shared_ptr<Graph> graph,
    std::shared_ptr<Tensor_attributes> attention_score,
    DiagonalAlignment_t diagonal_alignment,
    std::optional<int64_t> left_window,
    std::optional<int64_t> right_window,
    int64_t s_q,
    int64_t s_kv,
    std::shared_ptr<Tensor_attributes> s_q_ptr,
    std::shared_ptr<Tensor_attributes> s_kv_ptr
);

inline std::shared_ptr<Tensor_attributes> alibi_mask(
    std::shared_ptr<Graph> graph,
    std::shared_ptr<Tensor_attributes> attention_score,
    std::shared_ptr<Tensor_attributes>& alibi_slopes,
    int64_t h_q,
    int64_t& alibi_slopes_size
);

inline error_t build_operation_subgraph(std::shared_ptr<Graph> graph);
// clang-format on

}  // namespace attn::score_modifiers

template <typename DerivedT>
class SDPANodeBase : public NodeCRTP<DerivedT> {
   protected:
    using input_names  = SDPA_attributes::input_names;
    using output_names = SDPA_attributes::output_names;

    std::shared_ptr<Tensor_attributes> rng_output;
    std::shared_ptr<Tensor_attributes> alibi_slopes;
    int64_t alibi_slopes_size = 0;

   public:
    SDPA_attributes attributes;

    SDPANodeBase(SDPA_attributes&& attributes_, detail::Context const& context)
        : NodeCRTP<DerivedT>(context), attributes(std::move(attributes_)) {}

    bool
    is_paged_v() const {
        auto page_table_v_it = attributes.inputs.find(input_names::Page_table_V);
        return ((page_table_v_it) != attributes.inputs.end() && page_table_v_it->second != nullptr);
    }

    bool
    is_paged_k() const {
        auto page_table_k_it = attributes.inputs.find(input_names::Page_table_K);
        return ((page_table_k_it) != attributes.inputs.end() && page_table_k_it->second != nullptr);
    }

    bool
    has_seq_len_q() const {
        auto seq_len_Q_it = attributes.inputs.find(SDPA_attributes::input_names::SEQ_LEN_Q);
        return ((seq_len_Q_it) != attributes.inputs.end() && seq_len_Q_it->second != nullptr);
    }

    bool
    has_seq_len_kv() const {
        auto seq_len_KV_it = attributes.inputs.find(SDPA_attributes::input_names::SEQ_LEN_KV);
        return ((seq_len_KV_it) != attributes.inputs.end() && seq_len_KV_it->second != nullptr);
    }

    // Helper function to detect MXFP8 (microscaling FP8) mode
    // MXFP8 uses block-wise scale factors with E8M0 data type and F8_128x4 reordering
    // When detected, we use block_scale_dequantize before matmuls instead of pointwise descale after
    bool
    is_mxfp8_scaling() const {
        auto descale_q_it = attributes.inputs.find(input_names::Descale_Q);
        if (descale_q_it == attributes.inputs.end() || descale_q_it->second == nullptr) {
            return false;
        }
        auto const& descale_q = descale_q_it->second;
        return (descale_q->get_data_type() == DataType_t::FP8_E8M0 &&
                descale_q->get_reordering_type() == TensorReordering_t::F8_128x4);
    }

    // Helper function to infer KV sequence length
    // Note that it cannot be run as part of infer_properties_node as
    // this is being used in pre_validate_node
    int64_t
    infer_s_kv() const {
        int64_t s_kv = -1;

        auto get_input_dim = [this](const SDPA_attributes::input_names& input_name) {
            auto const input_it = attributes.inputs.find(input_name);
            if (input_it != attributes.inputs.end()) {
                return input_it->second->get_dim();
            } else {
                return std::vector<int64_t>({-1, -1, -1, -1});
            }
        };

        auto const& k_dim = get_input_dim(input_names::K);
        auto const& v_dim = get_input_dim(input_names::V);

        // If s_kv was set explicitly, use that
        if (attributes.max_seq_len_kv.has_value()) {
            s_kv = attributes.max_seq_len_kv.value();
        }
        // When one of K or V cache are paged, s_kv can be extracted directly
        else if (!is_paged_k()) {
            s_kv = k_dim[2];

        } else if (!is_paged_v()) {
            s_kv = v_dim[2];
        } else {
            CUDNN_FE_LOG_LABEL_ENDL(
                "WARNING: maximum kv sequence length is being inferred. To set it explicitly, please use  "
                "\"set_paged_attention_max_seq_len_kv\"");

            auto bias_it = attributes.inputs.find(input_names::Bias);
            auto rng_it  = attributes.outputs.find(output_names::RNG_DUMP);

            // If there is a bias, extract it from there
            if (bias_it != attributes.inputs.end() && bias_it->second != nullptr) {
                s_kv = get_input_dim(input_names::Bias)[3];
                // If there is an rng_dump output, extract it from there
            } else if (rng_it != attributes.outputs.end() && rng_it->second != nullptr) {
                s_kv = rng_it->second->get_dim()[3];
                // When both caches are paged, and the above failed, we need to infer s_kv from the page table and
                // container
            } else {
                // [b, 1, ceil(s_kv/block_size), 1]
                auto page_table_dim_k = get_input_dim(input_names::Page_table_K);
                // [b, h_k, block_size, d_k]
                auto const container_dim_k = get_input_dim(input_names::K);
                int64_t s_k                = page_table_dim_k[2] * container_dim_k[2];

                // [b, 1, ceil(s_kv/block_size), 1]
                auto page_table_dim_v = get_input_dim(input_names::Page_table_V);
                // [b, h_v, block_size, d_v]
                auto const container_dim_v = get_input_dim(input_names::V);
                int64_t s_v                = page_table_dim_v[2] * container_dim_v[2];

                s_kv = std::min(s_k, s_v);
            }
        }

        return s_kv;
    }

    error_t
    pre_validate_node() const override final {
        CUDNN_FE_LOG_LABEL_ENDL("INFO: Validating SDPANode " << attributes.name);

        // check that Q, K, V, O tensors has been assigned
        // check that dim and strides has been assigned and last stride is 1
#define CUDNN_FE_SDPA_VALIDATE_DIM_STRIDE(port, port_map)                                                       \
    {                                                                                                           \
        std::shared_ptr<Tensor_attributes> tensor_ptr = port_map.at(port);                                      \
        RETURN_CUDNN_FRONTEND_ERROR_IF(tensor_ptr->get_dim().size() != 4,                                       \
                                       error_code_t::ATTRIBUTE_NOT_SET,                                         \
                                       "The dim for " + std::string(#port) + " is invalid");                    \
        RETURN_CUDNN_FRONTEND_ERROR_IF(tensor_ptr->get_stride().size() != 4,                                    \
                                       error_code_t::ATTRIBUTE_NOT_SET,                                         \
                                       "The stride for " + std::string(#port) + " is invalid");                 \
        RETURN_CUDNN_FRONTEND_ERROR_IF(                                                                         \
            tensor_ptr->get_stride()[3] != 1,                                                                   \
            error_code_t::GRAPH_NOT_SUPPORTED,                                                                  \
            "The stride for the last dimension corresponding to the embedding size per head should be 1 for " + \
                std::string(#port));                                                                            \
    }

        CUDNN_FE_SDPA_VALIDATE_DIM_STRIDE(input_names::Q, attributes.inputs);
        CUDNN_FE_SDPA_VALIDATE_DIM_STRIDE(input_names::K, attributes.inputs);
        CUDNN_FE_SDPA_VALIDATE_DIM_STRIDE(input_names::V, attributes.inputs);
        CUDNN_FE_SDPA_VALIDATE_DIM_STRIDE(output_names::O, attributes.outputs);

        if (attributes.generate_stats.value_or(false) == true) {
            CUDNN_FE_VALIDATE_OUTPUT_TENSOR(output_names::Stats);
        }

        // If max is requested, validate that the output tensor is present
        if (attributes.outputs.find(output_names::Max) != attributes.outputs.end() &&
            attributes.outputs.at(output_names::Max) != nullptr) {
            CUDNN_FE_VALIDATE_OUTPUT_TENSOR(output_names::Max);
        }

        // If sum_exp is requested, validate that the output tensor is present
        if (attributes.outputs.find(output_names::Sum_exp) != attributes.outputs.end() &&
            attributes.outputs.at(output_names::Sum_exp) != nullptr) {
            CUDNN_FE_VALIDATE_OUTPUT_TENSOR(output_names::Sum_exp);
        }

#undef CUDNN_FE_SDPA_VALIDATE_DIM_STRIDE

        // validate backend limitations for the operation
        auto validation_result =
            attributes.validate_sdpa_support_surface(this->context, infer_s_kv(), is_paged_k(), is_paged_v());
        if (validation_result.is_good() == false) {
            return validation_result;
        }

        // return NOT_SET if sink_token present with 9.12 and below
        RETURN_CUDNN_FRONTEND_ERROR_IF(detail::get_backend_version() < 91300 &&
                                           attributes.inputs.find(input_names::SINK_TOKEN) != attributes.inputs.end(),
                                       error_code_t::ATTRIBUTE_NOT_SET,
                                       "SDPA with sink_token is not supported before 9.13.");

        // Validate MXFP8 scale factors if present
        if (is_mxfp8_scaling()) {
            // MXFP8 requires cuDNN 9.21.0 or later
            RETURN_CUDNN_FRONTEND_ERROR_IF(detail::get_backend_version() < 92100,
                                           error_code_t::GRAPH_NOT_SUPPORTED,
                                           "MXFP8 SDPA requires cuDNN 9.21.0 or later");

            // Get device SM version from context
            CHECK_CUDNN_FRONTEND_ERROR(this->context.populate_sm_version_from_device());
            int32_t const sm_version = this->context.get_sm_version();
            int32_t const prop_major = sm_version / 10;

            RETURN_CUDNN_FRONTEND_ERROR_IF(10 != prop_major,
                                           error_code_t::GRAPH_NOT_SUPPORTED,
                                           "MXFP8 SDPA is only supported on Blackwell Data Center architectures.");

            auto const& q_dim  = attributes.inputs.at(input_names::Q)->get_dim();
            auto const& k_dim  = attributes.inputs.at(input_names::K)->get_dim();
            auto const& v_dim  = attributes.inputs.at(input_names::V)->get_dim();
            int64_t const b    = q_dim[0];
            int64_t const h_q  = q_dim[1];
            int64_t const s_q  = q_dim[2];
            int64_t const d    = q_dim[3];
            int64_t const h_k  = k_dim[1];
            int64_t const h_v  = v_dim[1];
            int64_t const s_kv = infer_s_kv();

            // MXFP8 block size is fixed at 32
            constexpr int64_t block_size = 32;
            int64_t const d_scale        = (d + block_size - 1) / block_size;
            int64_t const s_scale        = (s_kv + block_size - 1) / block_size;

            // Validate Descale_Q
            auto const& descale_q = attributes.inputs.at(input_names::Descale_Q);
            RETURN_CUDNN_FRONTEND_ERROR_IF(descale_q->get_data_type() != DataType_t::FP8_E8M0,
                                           error_code_t::ATTRIBUTE_NOT_SET,
                                           "MXFP8 SDPA requires Descale_Q to have FP8_E8M0 data type");
            RETURN_CUDNN_FRONTEND_ERROR_IF(descale_q->get_reordering_type() != TensorReordering_t::F8_128x4,
                                           error_code_t::ATTRIBUTE_NOT_SET,
                                           "MXFP8 SDPA requires Descale_Q to have F8_128x4 reordering");
            RETURN_CUDNN_FRONTEND_ERROR_IF(
                descale_q->get_stride()[3] != 1,
                error_code_t::GRAPH_NOT_SUPPORTED,
                "MXFP8 SDPA requires Descale_Q to have contiguous d_scale dimension (stride[3] == 1)");

            // Validate Descale_K
            auto const& descale_k = attributes.inputs.at(input_names::Descale_K);
            RETURN_CUDNN_FRONTEND_ERROR_IF(descale_k->get_data_type() != DataType_t::FP8_E8M0,
                                           error_code_t::ATTRIBUTE_NOT_SET,
                                           "MXFP8 SDPA requires Descale_K to have FP8_E8M0 data type");
            RETURN_CUDNN_FRONTEND_ERROR_IF(descale_k->get_reordering_type() != TensorReordering_t::F8_128x4,
                                           error_code_t::ATTRIBUTE_NOT_SET,
                                           "MXFP8 SDPA requires Descale_K to have F8_128x4 reordering");
            RETURN_CUDNN_FRONTEND_ERROR_IF(
                descale_k->get_stride()[3] != 1,
                error_code_t::GRAPH_NOT_SUPPORTED,
                "MXFP8 SDPA requires Descale_K to have contiguous d_scale dimension (stride[3] == 1)");

            // Validate Descale_V
            auto const& descale_v = attributes.inputs.at(input_names::Descale_V);
            RETURN_CUDNN_FRONTEND_ERROR_IF(descale_v->get_data_type() != DataType_t::FP8_E8M0,
                                           error_code_t::ATTRIBUTE_NOT_SET,
                                           "MXFP8 SDPA requires Descale_V to have FP8_E8M0 data type");
            RETURN_CUDNN_FRONTEND_ERROR_IF(descale_v->get_reordering_type() != TensorReordering_t::F8_128x4,
                                           error_code_t::ATTRIBUTE_NOT_SET,
                                           "MXFP8 SDPA requires Descale_V to have F8_128x4 reordering");
            // // SF_V scales along s dimension (not d), so s_scale must be contiguous
            // RETURN_CUDNN_FRONTEND_ERROR_IF(
            //     descale_v->get_stride()[2] != 1,
            //     error_code_t::GRAPH_NOT_SUPPORTED,
            //     "MXFP8 SDPA requires Descale_V to have contiguous s_scale dimension (stride[2] == 1)");

            // Validate dimension consistency for SF_Q: [b, h_q, s_q_padded, d_scale_padded]
            auto const& sf_q_dim = descale_q->get_dim();
            RETURN_CUDNN_FRONTEND_ERROR_IF(sf_q_dim[0] != b || sf_q_dim[1] != h_q,
                                           error_code_t::ATTRIBUTE_NOT_SET,
                                           "MXFP8 SDPA: Descale_Q batch/head dimensions must match Q");
            RETURN_CUDNN_FRONTEND_ERROR_IF(
                sf_q_dim[3] < d_scale,
                error_code_t::ATTRIBUTE_NOT_SET,
                "MXFP8 SDPA: Descale_Q d_scale dimension too small (expected >= " + std::to_string(d_scale) + ")");

            // Validate dimension consistency for SF_K: [b, h_k, s_kv_padded, d_scale_padded]
            auto const& sf_k_dim = descale_k->get_dim();
            RETURN_CUDNN_FRONTEND_ERROR_IF(sf_k_dim[0] != b || sf_k_dim[1] != h_k,
                                           error_code_t::ATTRIBUTE_NOT_SET,
                                           "MXFP8 SDPA: Descale_K batch/head dimensions must match K");
            RETURN_CUDNN_FRONTEND_ERROR_IF(
                sf_k_dim[3] < d_scale,
                error_code_t::ATTRIBUTE_NOT_SET,
                "MXFP8 SDPA: Descale_K d_scale dimension too small (expected >= " + std::to_string(d_scale) + ")");

            // Validate dimension consistency for SF_V: [b, h_v, s_scale_padded, d_padded]
            auto const& sf_v_dim = descale_v->get_dim();
            RETURN_CUDNN_FRONTEND_ERROR_IF(sf_v_dim[0] != b || sf_v_dim[1] != h_v,
                                           error_code_t::ATTRIBUTE_NOT_SET,
                                           "MXFP8 SDPA: Descale_V batch/head dimensions must match V");
            RETURN_CUDNN_FRONTEND_ERROR_IF(
                sf_v_dim[2] < s_scale,
                error_code_t::ATTRIBUTE_NOT_SET,
                "MXFP8 SDPA: Descale_V s_scale dimension too small (expected >= " + std::to_string(s_scale) + ")");

            // Log MXFP8 configuration for debugging
            getLogger() << "[cudnn_frontend] INFO: MXFP8 SDPA configuration validated:" << std::endl
                        << "  Q dims: [" << b << ", " << h_q << ", " << s_q << ", " << d << "]" << std::endl
                        << "  SF_Q dims: [" << sf_q_dim[0] << ", " << sf_q_dim[1] << ", " << sf_q_dim[2] << ", "
                        << sf_q_dim[3] << "]" << std::endl
                        << "  SF_K dims: [" << sf_k_dim[0] << ", " << sf_k_dim[1] << ", " << sf_k_dim[2] << ", "
                        << sf_k_dim[3] << "]" << std::endl
                        << "  SF_V dims: [" << sf_v_dim[0] << ", " << sf_v_dim[1] << ", " << sf_v_dim[2] << ", "
                        << sf_v_dim[3] << "]" << std::endl
                        << "  Expected d_scale: " << d_scale << ", s_scale: " << s_scale << std::endl;
        }

        return {error_code_t::OK, ""};
    }

    error_t
    infer_properties_node() override final {
        if (attributes.generate_stats.value_or(false)) {
            auto stats     = attributes.outputs.at(output_names::Stats);
            auto stats_dim = stats->get_dim();

            if (stats_dim.empty()) {
                // Fill properties of virtual tensors
                auto const& p_dim = attributes.inputs[input_names::Q]->get_dim();
                auto b            = p_dim[0];
                auto h            = p_dim[1];
                auto s_q          = p_dim[2];
                stats->set_dim({b, h, s_q, 1}).set_stride({h * s_q, s_q, 1, 1});
            }
        }

        if (attributes.outputs[output_names::Max] != nullptr) {
            auto max = attributes.outputs.at(output_names::Max);

            if (max->get_dim().empty()) {
                // Fill properties of virtual tensors
                auto const& p_dim = attributes.inputs[input_names::Q]->get_dim();
                auto b            = p_dim[0];
                auto h            = p_dim[1];
                auto s_q          = p_dim[2];
                max->set_dim({b, h, s_q, 1}).set_stride({h * s_q, s_q, 1, 1});
            }
        }

        if (attributes.outputs[output_names::Sum_exp] != nullptr) {
            auto sum_exp = attributes.outputs.at(output_names::Sum_exp);

            if (sum_exp->get_dim().empty()) {
                // Fill properties of virtual tensors
                auto const& p_dim = attributes.inputs[input_names::Q]->get_dim();
                auto b            = p_dim[0];
                auto h            = p_dim[1];
                auto s_q          = p_dim[2];
                sum_exp->set_dim({b, h, s_q, 1}).set_stride({h * s_q, s_q, 1, 1});
            }
        }
        return {error_code_t::OK, ""};
    }

    error_t
    post_validate_node() const override final {
#define CUDNN_FE_VALIDATE_STRIDE(port, port_map)                                                                \
    {                                                                                                           \
        auto const& t = port_map.find(port);                                                                    \
        RETURN_CUDNN_FRONTEND_ERROR_IF(                                                                         \
            t->second->get_stride().back() != 1,                                                                \
            error_code_t::GRAPH_NOT_SUPPORTED,                                                                  \
            "The stride for the last dimension corresponding to the embedding size per head should be 1 for " + \
                std::string(#port));                                                                            \
    }

        CUDNN_FE_VALIDATE_STRIDE(output_names::O, attributes.outputs);

#undef CUDNN_FE_VALIDATE_STRIDE

        return {error_code_t::OK, ""};
    }

    virtual int64_t
    get_fe_workspace_size_node() const override final {
        int64_t size = 0;

        // align alibi slopes memory to 16 bytes
        size += ((alibi_slopes_size + 15) / 16 * 16);

        return size;
    }

    virtual error_t
    collect_tensors_in_workspace_node(
        std::unordered_map<Tensor_attributes::uid_t, std::tuple<int64_t, int64_t, std::vector<float>>>&
            workspace_modifications,
        int64_t& offset) const override final {
        if (attributes.alibi_mask) {
            CUDNN_FE_VALIDATE_AND_ASSIGN_INPUT_TENSOR(Q, input_names::Q);
            int64_t const h_q     = Q->second->get_dim()[1];
            auto alibi_slopes_vec = detail::get_alibi_slope(h_q);
            workspace_modifications.emplace(alibi_slopes->get_uid(), std::make_tuple(0, offset, alibi_slopes_vec));
            int64_t alibi_slopes_size_padded = ((alibi_slopes_size + 15) / 16 * 16);
            offset                           = offset + alibi_slopes_size_padded;
        }
        return {error_code_t::OK, ""};
    }

    error_t
    collect_tensors_to_dump_node(
        std::vector<std::pair<std::shared_ptr<Tensor_attributes>, char>>& tensors_to_dump) const override final {
        std::unordered_set<Tensor_attributes::uid_t> seen_uids;
        auto add_tensor = [&tensors_to_dump, &seen_uids](std::shared_ptr<Tensor_attributes> const& tensor) {
            if (tensor == nullptr) {
                return;
            }
            if (seen_uids.insert(tensor->get_uid()).second) {
                tensors_to_dump.emplace_back(tensor, 'd');
            }
        };

        auto const seq_len_q_it = attributes.inputs.find(input_names::SEQ_LEN_Q);
        if (seq_len_q_it != attributes.inputs.end()) {
            add_tensor(seq_len_q_it->second);
        }

        auto const seq_len_kv_it = attributes.inputs.find(input_names::SEQ_LEN_KV);
        if (seq_len_kv_it != attributes.inputs.end()) {
            add_tensor(seq_len_kv_it->second);
        }

        for (auto const& tensor : {attributes.inputs.at(input_names::Q),
                                   attributes.inputs.at(input_names::K),
                                   attributes.inputs.at(input_names::V),
                                   attributes.outputs.at(output_names::O)}) {
            if (tensor != nullptr) {
                add_tensor(tensor->get_ragged_offset());
            }
        }

        auto const stats_it = attributes.outputs.find(output_names::Stats);
        if (stats_it != attributes.outputs.end() && stats_it->second != nullptr) {
            add_tensor(stats_it->second->get_ragged_offset());
        }

        auto const max_it = attributes.outputs.find(output_names::Max);
        if (max_it != attributes.outputs.end() && max_it->second != nullptr) {
            add_tensor(max_it->second->get_ragged_offset());
        }

        auto const sum_exp_it = attributes.outputs.find(output_names::Sum_exp);
        if (sum_exp_it != attributes.outputs.end() && sum_exp_it->second != nullptr) {
            add_tensor(sum_exp_it->second->get_ragged_offset());
        }

        return {error_code_t::OK, ""};
    }

#ifndef CUDNN_FRONTEND_SKIP_JSON_LIB
    virtual void
    serialize(json& j) const override final {
        j               = attributes;
        j["is_mxfp8"]   = is_mxfp8_scaling();
        j["unfuse_fma"] = attributes.unfuse_fma;
        if (auto const rescale_threshold = get_rescale_threshold_from_env(); rescale_threshold.has_value()) {
            j["rescale_threshold"] = rescale_threshold.value();
        }
        if (is_mxfp8_scaling()) {
            j.update(R"({"tag": "SDPA_MXFP8_FWD"})"_json);
        } else if (attributes.mma_core_mode == DataType_t::FP8_E4M3 ||
                   attributes.mma_core_mode == DataType_t::FP8_E5M2) {
            j.update(R"({"tag": "SDPA_FP8_FWD"})"_json);
        } else {
            j.update(R"({"tag": "SDPA"})"_json);
        }
    }
#endif
};

class CompositeSDPANode : public SDPANodeBase<CompositeSDPANode> {
   public:
    CompositeSDPANode(SDPA_attributes&& attributes_, detail::Context const& context)
        : SDPANodeBase(std::move(attributes_), context) {}

    Type
    getType() override final {
        return Type::COMPOSITE;
    }

    error_t
    expand_node() override final {
        CUDNN_FE_LOG_LABEL_ENDL("INFO:     Inferrencing properties for CompositeSDPANode node " << attributes.name);

        // DO NOT REMOVE
        // input data type is needed for:
        // - aType of bmm2
        // - dropout scale in pre 8.9.3
        attributes.fill_from_context(this->context);

        // Gather dim to fill properties of virtual tensors
        auto const& q_dim = attributes.inputs[input_names::Q]->get_dim();
        auto b            = q_dim[0];
        auto h_q          = q_dim[1];
        auto s_q          = q_dim[2];
        auto d_qk         = q_dim[3];
        auto const& k_dim = attributes.inputs[input_names::K]->get_dim();
        auto h_k          = k_dim[1];
        auto const& v_dim = attributes.inputs[input_names::V]->get_dim();
        auto h_v          = v_dim[1];
        auto d_v          = v_dim[3];
        // Infer s_kv
        int64_t s_kv = infer_s_kv();

        // Check if using MXFP8 (microscaling FP8) with block-wise scale factors
        // Need to check this early because MXFP8 dequantization must happen before K transpose
        bool const use_mxfp8 = is_mxfp8_scaling();

        std::shared_ptr<Tensor_attributes> k_cache;
        if (!is_paged_k()) {
            if (use_mxfp8) {
                // MXFP8: Transpose K and SF_K first, then dequantize
                // The backend expects both K and its scale factor in transposed layout

                // Step 1: Transpose K -> KT by swapping dims and strides
                // K = {b, h_k, s_kv, d_qk} -> KT = {b, h_k, d_qk, s_kv}
                std::vector<int64_t> kt_dim    = attributes.inputs[input_names::K]->get_dim();
                std::vector<int64_t> kt_stride = attributes.inputs[input_names::K]->get_stride();
                std::swap(kt_dim[2], kt_dim[3]);
                std::swap(kt_stride[2], kt_stride[3]);
                attributes.inputs[input_names::K]->set_dim(kt_dim);
                attributes.inputs[input_names::K]->set_stride(kt_stride);

                // Step 2: Transpose SF_K similarly
                // SF_K = {b, h_k, s_kv_scale, d_qk_scale} -> {b, h_k, d_qk_scale, s_kv_scale}
                auto& sf_k                       = attributes.inputs[input_names::Descale_K];
                std::vector<int64_t> sf_k_dim    = sf_k->get_dim();
                std::vector<int64_t> sf_k_stride = sf_k->get_stride();
                std::swap(sf_k_dim[2], sf_k_dim[3]);
                std::swap(sf_k_stride[2], sf_k_stride[3]);
                sf_k->set_dim(sf_k_dim);
                sf_k->set_stride(sf_k_stride);

                // Step 3: Dequantize transposed K with transposed SF_K
                auto dequant_k_attrs = Block_scale_dequantize_attributes()
                                           .set_name("dequant_k")
                                           .set_block_size({1, 32});  // Standard MXFP8 block size

                auto k_dequant = std::make_shared<Tensor_attributes>();
                k_dequant->set_is_virtual(true);
                k_dequant->set_dim(kt_dim);
                k_dequant->set_stride(kt_stride);

                block_scale_dequantize(attributes.inputs[input_names::K], sf_k, dequant_k_attrs, k_dequant);

                k_cache = k_dequant;
            } else {
                // Non-MXFP8: map K->KT directly
                // cuDNN frontend API attention requires Q, K, V where
                // Q = {b, h_q, s_q, d_qk}
                // K = {b, h_k, s_kv, d_qk}
                // V = {b, h_v, s_kv, d_v}
                // but cuDNN backend API attention requires Q, KT, V
                // Q = {b, h_q, s_q, d_qk}
                // KT = {b, h_k, d_qk, s_kv}
                // V = {b, h_v, s_kv, d_v}
                // So the code below maps the K->KT
                std::vector<int64_t> temp_vec;

                temp_vec = attributes.inputs[input_names::K]->get_dim();
                std::swap(temp_vec[2], temp_vec[3]);
                attributes.inputs[input_names::K]->set_dim(temp_vec);

                temp_vec = attributes.inputs[input_names::K]->get_stride();
                std::swap(temp_vec[2], temp_vec[3]);
                attributes.inputs[input_names::K]->set_stride(temp_vec);

                k_cache = attributes.inputs[input_names::K];
            }
        } else {
            // Create a paged cache load operation
            auto paged_cache_load_attributes_k = PagedCacheLoad_attributes().set_name("paged_k_cache_operation");
            // Need to create virtual tensor descriptor for yOut here as it cannot be inferred
            // K-cache has BHDS layout
            k_cache = std::make_shared<Tensor_attributes>();
            k_cache->set_is_virtual(true);
            k_cache->set_dim({b, h_k, d_qk, s_kv});
            k_cache->set_stride({d_qk * s_kv * h_k, d_qk * s_kv, 1, d_qk});
            k_cache->set_data_type(attributes.inputs[input_names::K]->get_data_type());
            paged_cache_load(attributes.inputs[input_names::K],
                             attributes.inputs[input_names::SEQ_LEN_KV],
                             attributes.inputs[input_names::Page_table_K],
                             paged_cache_load_attributes_k,
                             k_cache);
        }

        // This tensor tracks the main chain of data flow
        std::shared_ptr<Tensor_attributes> last_output;

        // Prepare Q and K for first matmul
        // For MXFP8: Q is dequantized here, K was dequantized above before transpose
        // For regular FP8: use raw tensors, descale applied after matmul
        std::shared_ptr<Tensor_attributes> q_for_bmm1;
        std::shared_ptr<Tensor_attributes> k_for_bmm1;

        if (use_mxfp8) {
            // MXFP8: Dequantize Q using block scale factors
            auto dequant_q_attrs = Block_scale_dequantize_attributes()
                                       .set_name("dequant_q")
                                       .set_block_size({1, 32});  // Standard MXFP8 block size
            auto q_dequant = std::make_shared<Tensor_attributes>();
            q_dequant->set_is_virtual(true);
            q_dequant->set_dim(attributes.inputs[input_names::Q]->get_dim());
            q_dequant->set_stride(attributes.inputs[input_names::Q]->get_stride());
            block_scale_dequantize(attributes.inputs[input_names::Q],
                                   attributes.inputs[input_names::Descale_Q],
                                   dequant_q_attrs,
                                   q_dequant);
            q_for_bmm1 = q_dequant;

            // K was already dequantized and transposed above
            k_for_bmm1 = k_cache;
        } else {
            q_for_bmm1 = attributes.inputs[input_names::Q];
            k_for_bmm1 = k_cache;
        }

        //// Q * K
        auto bmm1_attributes = Matmul_attributes()
                                   .set_name("bmm1")
                                   .set_m_override(attributes.inputs[input_names::SEQ_LEN_Q])
                                   .set_n_override(attributes.inputs[input_names::SEQ_LEN_KV]);

        if (attributes.padding_mask) {
            bmm1_attributes.set_padding(0.0);
        }

        auto const& bmm1_output = matmul(q_for_bmm1, k_for_bmm1, bmm1_attributes);
        // Setting dim and strides as pointwise op wont have knowledge of how to do it for mha.
        bmm1_output->set_dim({b, h_q, s_q, s_kv}).set_stride({h_q * s_q * s_kv, s_q * s_kv, s_kv, 1});
        last_output = bmm1_output;

        //// Optional Attn scale
        // In case user provided a scalar value, do a fused scalar.
        if (attributes.attn_scale_value.has_value()) {
            attributes.inputs[input_names::Attn_scale] =
                std::make_shared<Tensor_attributes>(attributes.attn_scale_value.value());
        }

        // If attn scale present, add a pointwise mul node
        if (attributes.inputs[input_names::Attn_scale]) {
            Pointwise_attributes scale_attributes;
            scale_attributes.set_name("attn_scale").set_mode(PointwiseMode_t::MUL);
            auto const& attn_scale_output =
                pointwise(last_output, attributes.inputs[input_names::Attn_scale], scale_attributes);
            last_output = attn_scale_output;
        }

        // Descale Q (only for non-MXFP8 per-tensor scaling)
        // For MXFP8, descaling was done via block_scale_dequantize before matmul
        if (!use_mxfp8 && attributes.inputs.find(input_names::Descale_Q) != attributes.inputs.end() &&
            attributes.inputs.at(input_names::Descale_Q) != nullptr) {
            auto descale_q_attributes = Pointwise_attributes().set_mode(PointwiseMode_t::MUL).set_name("descale_q");
            last_output = pointwise(last_output, attributes.inputs.at(input_names::Descale_Q), descale_q_attributes);
        }

        // Descale K (only for non-MXFP8 per-tensor scaling)
        // For MXFP8, descaling was done via block_scale_dequantize before matmul
        if (!use_mxfp8 && attributes.inputs.find(input_names::Descale_K) != attributes.inputs.end() &&
            attributes.inputs.at(input_names::Descale_K) != nullptr) {
            auto descale_k_attributes = Pointwise_attributes().set_mode(PointwiseMode_t::MUL).set_name("descale_k");
            last_output = pointwise(last_output, attributes.inputs.at(input_names::Descale_K), descale_k_attributes);
        }

        if (attributes.attention_score_modifier != nullptr) {
            auto graph_                  = std::make_shared<Graph>();
            std::shared_ptr<INode> node_ = std::static_pointer_cast<INode>(graph_);
            node_->context               = this->context;
            last_output                  = attributes.attention_score_modifier(graph_, last_output);
            sub_nodes.emplace_back(node_);
        }

        // Optional bias
        if (attributes.inputs.find(input_names::Bias) != attributes.inputs.end() &&
            attributes.inputs[input_names::Bias]) {
            auto graph_                  = std::make_shared<Graph>();
            std::shared_ptr<INode> node_ = std::static_pointer_cast<INode>(graph_);
            node_->context               = this->context;
            last_output = attn::score_modifiers::bias(graph_, last_output, attributes.inputs[input_names::Bias]);
            sub_nodes.emplace_back(node_);
        }

        if (attributes.alibi_mask) {
            auto graph_                  = std::make_shared<Graph>();
            std::shared_ptr<INode> node_ = std::static_pointer_cast<INode>(graph_);
            node_->context               = this->context;
            last_output = attn::score_modifiers::alibi_mask(graph_, last_output, alibi_slopes, h_q, alibi_slopes_size);
            sub_nodes.emplace_back(node_);
        }

        // There are two cases of applying padding mask
        // 1. when actual seq_len is less than or equal to max_seq_len
        if (attributes.padding_mask) {
            auto graph_                  = std::make_shared<Graph>();
            std::shared_ptr<INode> node_ = std::static_pointer_cast<INode>(graph_);
            node_->context               = this->context;
            last_output                  = attn::score_modifiers::padding_mask(graph_,
                                                              last_output,
                                                              attributes.inputs[input_names::SEQ_LEN_KV],
                                                              attributes.inputs[input_names::SEQ_LEN_Q]);
            sub_nodes.emplace_back(node_);
        }

        // 2. (bug in cudnn backend) no padding with max_seq_len%64!=0
        if ((s_kv % 64 != 0) && (!(attributes.padding_mask)) && (detail::get_backend_version() < 90000)) {
            auto col_index_attributes =
                Pointwise_attributes().set_name("gen_col_index").set_mode(PointwiseMode_t::GEN_INDEX).set_axis(3);
            auto col_index_output = pointwise(last_output, col_index_attributes);
            // scalar seq_kv only needs to be passed in case there in no padding mask and seq_kv is not multiple of 64.
            // Also future versions of cudnn will not need it, hence tensor is pre-fixed.
            auto scalar_max_seq_kv = std::make_shared<Tensor_attributes>(static_cast<int32_t>(s_kv));

            auto col_less_seq_kv_attributes =
                Pointwise_attributes().set_name("col_less_seq_kv").set_mode(PointwiseMode_t::CMP_LT);
            auto col_less_seq_kv_output = pointwise(col_index_output, scalar_max_seq_kv, col_less_seq_kv_attributes);

            // Lower attributes to binary select attributes
            auto negative_inf_padding =
                std::make_shared<Tensor_attributes>(attn::score_modifiers::get_negative_inf_value());
            auto binary_select_attributes =
                Pointwise_attributes().set_name("binary_select").set_mode(PointwiseMode_t::BINARY_SELECT);
            auto padding_mask_output =
                pointwise(last_output, negative_inf_padding, col_less_seq_kv_output, binary_select_attributes);
            last_output = padding_mask_output;
        }

        // Apply (bottom-right) causal masking (with right bound) and/or set the left bound
        if (attributes.left_bound.has_value() || attributes.right_bound.has_value()) {
            auto graph_                  = std::make_shared<Graph>();
            std::shared_ptr<INode> node_ = std::static_pointer_cast<INode>(graph_);
            node_->context               = this->context;

            auto s_kv_ptr = attributes.inputs.find(input_names::SEQ_LEN_KV) != attributes.inputs.end()
                                ? attributes.inputs[input_names::SEQ_LEN_KV]
                                : nullptr;
            auto s_q_ptr  = attributes.inputs.find(input_names::SEQ_LEN_Q) != attributes.inputs.end()
                                ? attributes.inputs[input_names::SEQ_LEN_Q]
                                : nullptr;

            last_output = attn::score_modifiers::sliding_window_mask(graph_,
                                                                     last_output,
                                                                     attributes.diagonal_alignment,
                                                                     attributes.left_bound,
                                                                     attributes.right_bound,
                                                                     s_q,
                                                                     s_kv,
                                                                     s_q_ptr,
                                                                     s_kv_ptr);
            sub_nodes.emplace_back(node_);
        }

        // Lower attributes to softmax attributes
        auto softmax_output = std::make_shared<Tensor_attributes>();
        softmax_output->set_is_virtual(true);

        auto softmax_attributes = Softmax_attributes().set_name("softmax");
        // Set sink for softmax if user has provided a sink tensor
        if (attributes.inputs.find(input_names::SINK_TOKEN) != attributes.inputs.end()) {
            softmax_attributes.set_sink(attributes.inputs[input_names::SINK_TOKEN]);
        }
        // Special non-functional-style call. Needed because output already created and provided to user.
        softmax(last_output,
                softmax_attributes,
                softmax_output,
                attributes.outputs[output_names::Stats],
                attributes.outputs[output_names::Max],
                attributes.outputs[output_names::Sum_exp]);
        last_output = softmax_output;

        // Two cases for training: dropout present or not
        bool dropout_present         = false;
        auto const& dropout_mask     = attributes.inputs.find(input_names::Dropout_mask);
        bool const is_dropout_custom = (dropout_mask != attributes.inputs.end()) && (dropout_mask->second != nullptr);
        if (attributes.dropout_probability.has_value()) {
            dropout_present = true;
            // Special case: Skip dropout when 0.0 probability. Only do for 8.9.3 and up as rng isn't optional earlier.
            if (detail::get_backend_version() > 8902 && attributes.dropout_probability.value() == 0.0) {
                dropout_present = false;
            }
        } else if (is_dropout_custom) {
            dropout_present = true;
        }

        if (dropout_present) {
            if (is_dropout_custom) {
                auto dropout_scale_attributes =
                    Pointwise_attributes().set_name("dropout_scale_mul").set_mode(PointwiseMode_t::MUL);
                auto const& dropout_scale_output =
                    pointwise(last_output, attributes.inputs[input_names::Dropout_scale], dropout_scale_attributes);

                auto mask_attributes =
                    Pointwise_attributes().set_name("dropout_mask_mul").set_mode(PointwiseMode_t::MUL);
                auto const& dropout_mask_output =
                    pointwise(dropout_scale_output, dropout_mask->second, mask_attributes);
                last_output = dropout_mask_output;
            } else {
                if (attributes.outputs[output_names::RNG_DUMP] != nullptr) {
                    rng_output = attributes.outputs[output_names::RNG_DUMP];
                    rng(attributes.inputs[input_names::Seed],
                        attributes.inputs[input_names::Offset],
                        Rng_attributes()
                            .set_name("rng")
                            .set_distribution(RngDistribution_t::BERNOULLI)
                            .set_bernoulli_probability(1.0 - attributes.dropout_probability.value()),
                        rng_output);
                } else {
                    rng_output = rng(attributes.inputs[input_names::Seed],
                                     attributes.inputs[input_names::Offset],
                                     Rng_attributes()
                                         .set_name("rng")
                                         .set_distribution(RngDistribution_t::BERNOULLI)
                                         .set_bernoulli_probability(1.0 - attributes.dropout_probability.value()));
                    rng_output
                        // Hard coding dim and strides as rng output can no inputs to infer it from.
                        ->set_dim({b, h_q, s_q, s_kv})
                        .set_stride({h_q * s_q * s_kv, s_q * s_kv, s_kv, 1});
                }

                auto mask_attributes =
                    Pointwise_attributes().set_name("dropout_mask_mul").set_mode(PointwiseMode_t::MUL);
                auto const& dropout_mask_output = pointwise(last_output, rng_output, mask_attributes);
                last_output                     = dropout_mask_output;

                std::shared_ptr<cudnn_frontend::graph::Tensor_attributes> dropout_scale = nullptr;

                if (detail::get_backend_version() < 8903) {
                    half dropout_scale_value = __float2half(1.0f / (1.0f - attributes.dropout_probability.value()));
                    dropout_scale            = std::make_shared<Tensor_attributes>(dropout_scale_value);
                } else {
                    float dropout_scale_value = (1.0f / (1.0f - attributes.dropout_probability.value()));
                    dropout_scale             = std::make_shared<Tensor_attributes>(dropout_scale_value);
                }

                auto dropout_scale_attributes =
                    Pointwise_attributes().set_name("dropout_scale").set_mode(PointwiseMode_t::MUL);
                auto const& dropout_scale_output = pointwise(last_output, dropout_scale, dropout_scale_attributes);
                last_output                      = dropout_scale_output;
            }
        }

        // Amax S
        if (attributes.outputs.find(output_names::Amax_S) != attributes.outputs.end() &&
            attributes.outputs.at(output_names::Amax_S) != nullptr) {
            auto amax_attributes = Reduction_attributes().set_name("amax_s").set_mode(ReductionMode_t::AMAX);
            // Special non-functional-style call. Needed because output already created and provided to user.
            reduction(last_output, amax_attributes, attributes.outputs.at(output_names::Amax_S));
        }

        // Scale S
        if (attributes.inputs.find(input_names::Scale_S) != attributes.inputs.end() &&
            attributes.inputs.at(input_names::Scale_S) != nullptr) {
            auto scale_s_attributes = Pointwise_attributes().set_name("scale_s").set_mode(PointwiseMode_t::MUL);
            last_output = pointwise(last_output, attributes.inputs.at(input_names::Scale_S), scale_s_attributes);
        }

        // Lower attributes to bmm2 attributes
        // Requirement by cudnn backend to take in bmm2 aType as i/o type.
        last_output->set_data_type(attributes.inputs[input_names::Q]->get_data_type());

        auto const& seq_len_q  = attributes.inputs[input_names::SEQ_LEN_Q];
        auto const& seq_len_kv = attributes.inputs[input_names::SEQ_LEN_KV];
        // auto const& V          = attributes.inputs[input_names::V];
        auto const& O = attributes.outputs[output_names::O];

        std::shared_ptr<Tensor_attributes> v_cache;

        if (!is_paged_v()) {
            v_cache = attributes.inputs[input_names::V];
        } else {
            auto paged_cache_load_attributes_v = PagedCacheLoad_attributes().set_name("paged_v_cache_operation");
            v_cache                            = std::make_shared<Tensor_attributes>();
            v_cache->set_dim({b, h_v, s_kv, d_v})
                .set_stride({d_v * s_kv * h_v, d_v * s_kv, d_v, 1})
                .set_data_type(attributes.inputs[input_names::V]->get_data_type());
            v_cache->set_is_virtual(true);
            paged_cache_load(attributes.inputs[input_names::V],
                             attributes.inputs[input_names::SEQ_LEN_KV],
                             attributes.inputs[input_names::Page_table_V],
                             paged_cache_load_attributes_v,
                             v_cache);
        }

        //// S * V
        if (attributes.mma_core_mode == DataType_t::HALF) {
            auto bmm2_attributes =
                Matmul_attributes().set_name("bmm2").set_m_override(seq_len_q).set_k_override(seq_len_kv);
            // Special non-functional-style call. Needed because output already created and provided to user.
            matmul(last_output, v_cache, bmm2_attributes, O);
        } else if (attributes.mma_core_mode == DataType_t::FP8_E4M3 ||
                   attributes.mma_core_mode == DataType_t::FP8_E5M2) {
            if (use_mxfp8) {
                // MXFP8: Dequantize V using block scale factors
                // SF_V should be provided with dims [b, h, s_scale, d] where s is scaled
                // (different from SF_Q/SF_K which have d scaled)
                // No transformation needed - use SF_V as provided

                // Dequantize V with SF_V
                auto dequant_v_attrs = Block_scale_dequantize_attributes()
                                           .set_name("dequant_v")
                                           .set_block_size({1, 32});  // Standard MXFP8 block size
                auto v_dequant = std::make_shared<Tensor_attributes>();
                v_dequant->set_is_virtual(true);
                v_dequant->set_dim(v_cache->get_dim());
                v_dequant->set_stride(v_cache->get_stride());
                block_scale_dequantize(v_cache, attributes.inputs[input_names::Descale_V], dequant_v_attrs, v_dequant);

                // Use regular matmul with dequantized inputs
                auto bmm2_attributes =
                    Matmul_attributes().set_name("bmm2").set_m_override(seq_len_q).set_k_override(seq_len_kv);
                // Special non-functional-style call. Needed because output already created and provided to user.
                matmul(last_output, v_dequant, bmm2_attributes, O);

                // Compute Amax_O for MXFP8
                auto const& amax_o = attributes.outputs.at(output_names::Amax_O);
                if (amax_o != nullptr) {
                    auto amax_attributes = Reduction_attributes().set_name("amax_o").set_mode(ReductionMode_t::AMAX);
                    // Special non-functional-style call. Needed because output already created and provided to user.
                    reduction(O, amax_attributes, amax_o);
                }
            } else {
                // Regular per-tensor FP8 scaling
                auto const& descale_s = attributes.inputs.at(input_names::Descale_S);
                auto const& descale_v = attributes.inputs.at(input_names::Descale_V);
                auto const& scale_o   = attributes.inputs.at(input_names::Scale_O);
                auto const& amax_o    = attributes.outputs.at(output_names::Amax_O);

                auto bmm2_attributes =
                    Matmul_fp8_attributes().set_name("bmm2").set_m_override(seq_len_q).set_k_override(seq_len_kv);
                // Special non-functional-style call. Needed because output already created and provided to user.
                matmul_fp8(last_output, v_cache, descale_s, descale_v, scale_o, bmm2_attributes, O, amax_o);
            }
        } else {
            RETURN_CUDNN_FRONTEND_ERROR_IF(true, error_code_t::GRAPH_NOT_SUPPORTED, "Unsupported MMA core mode");
        }

        return {error_code_t::OK, ""};
    }
};

class CompositeSDPABackwardNode : public NodeCRTP<CompositeSDPABackwardNode> {
    using input_names  = SDPA_backward_attributes::input_names;
    using output_names = SDPA_backward_attributes::output_names;

   private:
    // non-virtual node gpu tensors
    std::shared_ptr<Tensor_attributes> dQ_accum;
    int64_t dQ_accum_size = 0;
    std::shared_ptr<Tensor_attributes> dK_fullhead;
    int64_t dK_fullhead_size = 0;
    std::shared_ptr<Tensor_attributes> dV_fullhead;
    int64_t dV_fullhead_size = 0;
    std::shared_ptr<Tensor_attributes> softmax_sum;
    int64_t softmax_sum_size = 0;
    std::shared_ptr<Tensor_attributes> alibi_slopes;
    int64_t alibi_slopes_size = 0;

    mutable bool has_workaround_padding_mask         = false;  // Will be edited in pre_validate_node()
    mutable int32_t s_q_for_workaround_padding_mask  = 0;      // Will be edited in pre_validate_node()
    mutable int32_t s_kv_for_workaround_padding_mask = 0;      // Will be edited in pre_validate_node()
    mutable std::shared_ptr<Tensor_attributes>
        workaround_padding_mask_seq_len_q;  // Will be edited in pre_validate_node()
    mutable std::shared_ptr<Tensor_attributes>
        workaround_padding_mask_seq_len_kv;                                  // Will be edited in pre_validate_node()
    mutable int64_t batch_size_for_workaround_padding_mask         = 0;      // Will be edited in pre_validate_node()
    mutable bool is_deterministic_algorithm_supported_on_blackwell = false;  // Will be edited in pre_validate_node()
    mutable bool is_d256_on_blackwell                              = false;  // Will be edited in pre_validate_node()

   public:
    mutable SDPA_backward_attributes attributes;  // Will be edited in pre_validate_node() for workaround padding mask

    CompositeSDPABackwardNode(SDPA_backward_attributes&& attributes_, detail::Context const& context)
        : NodeCRTP(context), attributes(std::move(attributes_)) {}

    Type
    getType() override final {
        return Type::COMPOSITE;
    }

    error_t
    pre_validate_node() const override final {
        CUDNN_FE_LOG_LABEL_ENDL("INFO: Validating CompositeSDPABackwardNode" << attributes.name);

        // check that Q, K, V, O, stats, dO, dQ, dK, dV tensors has been assigned
        // check that dim and strides has been assigned and last stride is 1
#define CUDNN_FE_SDPA_VALIDATE_DIM_STRIDE(port, port_map)                                                       \
    {                                                                                                           \
        std::shared_ptr<Tensor_attributes> tensor_ptr = port_map.at(port);                                      \
        RETURN_CUDNN_FRONTEND_ERROR_IF(tensor_ptr->get_dim().size() != 4,                                       \
                                       error_code_t::ATTRIBUTE_NOT_SET,                                         \
                                       "The dim for " + std::string(#port) + " is invalid");                    \
        RETURN_CUDNN_FRONTEND_ERROR_IF(tensor_ptr->get_stride().size() != 4,                                    \
                                       error_code_t::ATTRIBUTE_NOT_SET,                                         \
                                       "The stride for " + std::string(#port) + " is invalid");                 \
        RETURN_CUDNN_FRONTEND_ERROR_IF(                                                                         \
            tensor_ptr->get_stride()[3] != 1,                                                                   \
            error_code_t::GRAPH_NOT_SUPPORTED,                                                                  \
            "The stride for the last dimension corresponding to the embedding size per head should be 1 for " + \
                std::string(#port));                                                                            \
    }

        CUDNN_FE_SDPA_VALIDATE_DIM_STRIDE(input_names::Q, attributes.inputs);
        CUDNN_FE_SDPA_VALIDATE_DIM_STRIDE(input_names::K, attributes.inputs);
        CUDNN_FE_SDPA_VALIDATE_DIM_STRIDE(input_names::V, attributes.inputs);
        CUDNN_FE_SDPA_VALIDATE_DIM_STRIDE(input_names::O, attributes.inputs);
        CUDNN_FE_SDPA_VALIDATE_DIM_STRIDE(input_names::Stats, attributes.inputs);
        CUDNN_FE_SDPA_VALIDATE_DIM_STRIDE(input_names::dO, attributes.inputs);
        CUDNN_FE_SDPA_VALIDATE_DIM_STRIDE(output_names::dQ, attributes.outputs);
        CUDNN_FE_SDPA_VALIDATE_DIM_STRIDE(output_names::dK, attributes.outputs);
        CUDNN_FE_SDPA_VALIDATE_DIM_STRIDE(output_names::dV, attributes.outputs);

#undef CUDNN_FE_SDPA_VALIDATE_DIM_STRIDE

        // validate backend limitations for the operation
        // clang-format off
        int64_t s_q  = attributes.inputs.at(input_names::Q)->get_dim()[2];
        int64_t s_kv = attributes.inputs.at(input_names::V)->get_dim()[2];
        int64_t h_q  = attributes.inputs.at(input_names::Q)->get_dim()[1];
        int64_t h_k  = attributes.inputs.at(input_names::K)->get_dim()[1];
        int64_t h_v  = attributes.inputs.at(input_names::V)->get_dim()[1];
        int64_t d_qk = attributes.inputs.at(input_names::Q)->get_dim()[3];
        int64_t d_v  = attributes.inputs.at(input_names::V)->get_dim()[3];

        bool const is_ragged = attributes.inputs.at(input_names::Q)->get_ragged_offset() ||
                               attributes.inputs.at(input_names::K)->get_ragged_offset() ||
                               attributes.inputs.at(input_names::V)->get_ragged_offset() ||
                               attributes.inputs.at(input_names::O)->get_ragged_offset();

        auto const& bias_mask = attributes.inputs.find(input_names::Bias);
        bool const is_bias   = (bias_mask != attributes.inputs.end() && bias_mask->second != nullptr);
        auto const& dbias_mask = attributes.outputs.find(output_names::dBias);
        bool const is_dbias   = (dbias_mask != attributes.outputs.end() && dbias_mask->second != nullptr);

        auto const& dropout_mask     = attributes.inputs.find(input_names::Dropout_mask);
        bool const is_dropout_custom = (dropout_mask != attributes.inputs.end()) && (dropout_mask->second != nullptr);
        bool const is_dropout        = attributes.dropout_probability.has_value() || is_dropout_custom;

        auto const& rng_tensor = attributes.outputs.find(output_names::RNG_DUMP);
        bool const is_rng   = (rng_tensor != attributes.outputs.end() && rng_tensor->second != nullptr);

        // validation TODO:
        //    - validate stats has valid dims
        //    - validate Q and dQ have the same dims

        // Stop s_q = S_kv = 1 from running
        RETURN_CUDNN_FRONTEND_ERROR_IF(s_q == 1 && s_kv == 1,
                                       error_code_t::GRAPH_NOT_SUPPORTED,
                                       "s_q = s_kv = 1 is not supported.");

        // Bug workarounds for known problematic versions (TE constraint)
        RETURN_CUDNN_FRONTEND_ERROR_IF(
            detail::get_backend_version() == 91000 || detail::get_backend_version() == 91001,
            error_code_t::GRAPH_NOT_SUPPORTED,
            "SDPA FP16/BF16 backward is not supported on cuDNN 9.10.0/9.10.1 due to known bugs. "
            "Please consider upgrading to 9.10.2 or newer.");

        // 9.14.0 sliding window bug: non-causal + s_kv > 1024 + sliding window
        RETURN_CUDNN_FRONTEND_ERROR_IF(
            detail::get_backend_version() == 91400 && s_kv > 1024 && attributes.left_bound.has_value() &&
                !attributes.has_causal_like_masking(),
            error_code_t::GRAPH_NOT_SUPPORTED,
            "cuDNN 9.14.0 has a known bug with non-causal + s_kv > 1024 + sliding window attention. "
            "Please consider upgrading to 9.14.1 or newer.");

        CHECK_CUDNN_FRONTEND_ERROR(context.populate_sm_version_from_device());
        int32_t const sm_version = context.get_sm_version();
        int32_t const prop_major = sm_version / 10;

        if (prop_major == 9) { 
            // validate basic dimension requirements

            if ((detail::get_backend_version() >= 91100) && (detail::get_backend_version() < 91300)) {
                
                if ((128 < d_qk) && (d_qk <= 192) && (64 < d_v) && (d_v <= 128)) {

                    // DeepSeek case, 9.11 only supports 192 hidden dim
                        RETURN_CUDNN_FRONTEND_ERROR_IF( (d_v != 128) && (d_qk != 192),
                                                error_code_t::GRAPH_NOT_SUPPORTED,
                                                "Num hidden_dim d_v should be equal to 128 if d_qk is 192");
                }
            }

            RETURN_CUDNN_FRONTEND_ERROR_IF((d_qk > 256) || (d_qk % 8 != 0) || (d_v > 256) || (d_v % 8 != 0),
                        error_code_t::GRAPH_NOT_SUPPORTED,
                        "Num hidden_dim should be less than or equal to 256 and hidden_dim should be multiple of 8");

        } else if (prop_major == 10 && detail::get_backend_version() >= 91100) {
            // validate basic dimension requirements
            if (d_qk == 192) { // special case for 192 hidden dim
                RETURN_CUDNN_FRONTEND_ERROR_IF( (d_v != 128),
                                        error_code_t::GRAPH_NOT_SUPPORTED,
                                        "Num hidden_dim d_v should be equal to 128 if d_qk is 192");
            } else if (detail::get_backend_version() >= 92300 && d_qk == 256 && d_v == 256) {
                is_d256_on_blackwell = true;
                attributes.is_deterministic_algorithm = true;
            } else {
                RETURN_CUDNN_FRONTEND_ERROR_IF((d_qk > 128) || (d_qk % 8 != 0) || (d_v > 128) || (d_v % 8 != 0),
                                            error_code_t::GRAPH_NOT_SUPPORTED,
                                            "Num hidden_dim should be less than or equal to 128 and hidden_dim should be multiple of 8 when d_qk != d_v");
            }
        } else {
            // validate basic dimension requirements
            RETURN_CUDNN_FRONTEND_ERROR_IF((d_qk > 128) || (d_qk % 8 != 0) || (d_v > 128) || (d_v % 8 != 0),
                                        error_code_t::GRAPH_NOT_SUPPORTED,
                                        "Num hidden_dim should be less than or equal to 128 and hidden_dim should be multiple of 8");
        }

        RETURN_CUDNN_FRONTEND_ERROR_IF((attributes.attention_score_modifier != nullptr) &&
                    (attributes.alibi_mask || attributes.padding_mask || attributes.has_causal_like_masking() ||
                     attributes.left_bound.has_value()), error_code_t::GRAPH_NOT_SUPPORTED,"Attention score mod enabled and hence other subgraphs are disabled.");

        RETURN_CUDNN_FRONTEND_ERROR_IF((h_q % h_k != 0) || (h_q % h_v != 0),
                                       error_code_t::GRAPH_NOT_SUPPORTED,
                                       "For group-query attention, number of heads for key and query must be a factor of number of heads for query");

        // validate options for attn_scale
        auto const& attn_scale    = attributes.inputs.find(input_names::Attn_scale);
        bool const has_attn_scale = (attn_scale != attributes.inputs.end()) && (attn_scale->second != nullptr);
        RETURN_CUDNN_FRONTEND_ERROR_IF(has_attn_scale && attributes.attn_scale_value.has_value(),
                                       error_code_t::ATTRIBUTE_NOT_SET,
                                       "attn_scale with tensor and value cannot be set at the same time.");

        // validate alibi requirements
        RETURN_CUDNN_FRONTEND_ERROR_IF(attributes.alibi_mask && !(attributes.right_bound.has_value() && attributes.right_bound.value() == 0),
                        error_code_t::GRAPH_NOT_SUPPORTED,
                        "When alibi mask is used, diagonal_band_right_bound needs to be set to 0.");

        // validate options for bias mask
        RETURN_CUDNN_FRONTEND_ERROR_IF(is_bias && (bias_mask->second->get_data_type() == DataType_t::BOOLEAN),
                                        error_code_t::GRAPH_NOT_SUPPORTED,
                                        "Bias mask data type cannot be boolean");

        if (s_kv % 128 != 0 && attributes.padding_mask == false && is_ragged == false && detail::get_backend_version() <= 91500) {
            CUDNN_FE_LOG_LABEL_ENDL("INFO: Workaround padding mask is enabled for s_q % 128 != 0 and use_padding_mask == false and is_ragged == false");
            has_workaround_padding_mask = true;
            batch_size_for_workaround_padding_mask = attributes.inputs.at(input_names::Q)->get_dim()[0];
            s_q_for_workaround_padding_mask = s_q;
            s_kv_for_workaround_padding_mask = s_kv;
            workaround_padding_mask_seq_len_q = std::make_shared<Tensor_attributes>();
            workaround_padding_mask_seq_len_q->set_name("workaround_padding_mask_seq_len_q").set_dim({batch_size_for_workaround_padding_mask,1,1,1}).set_stride({1,1,1,1}).set_data_type(DataType_t::INT32);
            workaround_padding_mask_seq_len_kv = std::make_shared<Tensor_attributes>();
            workaround_padding_mask_seq_len_kv->set_name("workaround_padding_mask_seq_len_kv").set_dim({batch_size_for_workaround_padding_mask,1,1,1}).set_stride({1,1,1,1}).set_data_type(DataType_t::INT32);
            attributes.set_padding_mask(true);
            attributes.set_seq_len_q(workaround_padding_mask_seq_len_q).set_seq_len_kv(workaround_padding_mask_seq_len_kv);
        }

        // validate options for padding mask
        auto const& seq_len_q     = attributes.inputs.find(input_names::SEQ_LEN_Q);
        bool const has_seq_len_q  = (seq_len_q != attributes.inputs.end()) && (seq_len_q->second != nullptr);
        auto const& seq_len_kv    = attributes.inputs.find(input_names::SEQ_LEN_KV);
        bool const has_seq_len_kv = (seq_len_kv != attributes.inputs.end()) && (seq_len_kv->second != nullptr);
        RETURN_CUDNN_FRONTEND_ERROR_IF(attributes.padding_mask && (!has_seq_len_q || !has_seq_len_kv),
                                       error_code_t::ATTRIBUTE_NOT_SET,
                                       "Padding mask requires seq_len_q and seq_len_kv to be set.");
        RETURN_CUDNN_FRONTEND_ERROR_IF((!attributes.padding_mask && !attributes.attention_score_modifier) && (has_seq_len_q || has_seq_len_kv),
                                       error_code_t::ATTRIBUTE_NOT_SET,
                                       "seq_len_q and seq_len_kv needs to be set only if padding mask is enabled.");

        // validate options for max_total_seq_len
        RETURN_CUDNN_FRONTEND_ERROR_IF((attributes.max_total_seq_len_q.has_value() || attributes.max_total_seq_len_kv.has_value()) && !is_ragged,
                                       error_code_t::GRAPH_NOT_SUPPORTED,
                                       "max_total_seq_len_q is only supported with packed layout");

        // validate options for bottom right causal mask
        RETURN_CUDNN_FRONTEND_ERROR_IF(attributes.has_causal_mask_bottom_right() && (!attributes.padding_mask) && s_q > s_kv,
                                       error_code_t::GRAPH_NOT_SUPPORTED,
                                       "Bottom right causal mask does not support max_s_q > max_s_kv. Please virtually slice the Q tensor and pass it as max_s_q == max_s_kv");

        RETURN_CUDNN_FRONTEND_ERROR_IF(attributes.has_causal_mask_bottom_right() && (is_bias || attributes.alibi_mask || (is_ragged && !attributes.padding_mask) || is_dropout),
                                       error_code_t::GRAPH_NOT_SUPPORTED,
                                       "Bottom right causal mask is only supported with is_bias=False, is_alibi=False, is_dropout=False. Further is_ragged==True is only allowed when padding_mask=True.");

        RETURN_CUDNN_FRONTEND_ERROR_IF(attributes.has_causal_mask_bottom_right() && (detail::get_backend_version() < 90600) && ((s_q % 64 != 0) || (s_kv % 64 != 0)),
                                       error_code_t::GRAPH_NOT_SUPPORTED,
                                       "Bottom right causal mask is only supported with s_q multiple of 64, and s_kv multiple of 64, for cudnn version below 9.6.0");

        // validate options for sliding window length
        RETURN_CUDNN_FRONTEND_ERROR_IF(attributes.left_bound.has_value() && attributes.left_bound.value() <= 0,
                                       error_code_t::INVALID_VALUE,
                                       "Left bound (Sliding window length) should be greater than or equals to zero when set.");

        RETURN_CUDNN_FRONTEND_ERROR_IF(attributes.left_bound.has_value() && (s_q * attributes.left_bound.value() == s_kv * attributes.left_bound.value()) && (detail::get_backend_version() <= 90900) && (prop_major == 9) && attributes.has_causal_mask_bottom_right(),
                                       error_code_t::GRAPH_NOT_SUPPORTED,
                                       "On Hopper architecture, this specific combination of s_q, s_kv, and left_bound + right_bound + bottom right diagonal alignment is not supported for backend version 9.9 or below");

        RETURN_CUDNN_FRONTEND_ERROR_IF(attributes.left_bound.has_value() && (!attributes.padding_mask) && s_q > s_kv,
                                       error_code_t::GRAPH_NOT_SUPPORTED,
                                       "Sliding window attention is only supported with max_s_q <= max_s_kv.");

        if ((detail::get_backend_version() >= 91002)) {
             RETURN_CUDNN_FRONTEND_ERROR_IF((attributes.left_bound.has_value() || attributes.right_bound.has_value()) && ((is_ragged && !attributes.padding_mask)),
                                        error_code_t::GRAPH_NOT_SUPPORTED,
                                        "Left and right bounds with is_ragged==True is only allowed when padding_mask=True. And the diagonal alignment must be set.");
        } else {
            RETURN_CUDNN_FRONTEND_ERROR_IF(attributes.left_bound.has_value() && (! attributes.has_causal_like_masking() || is_dropout || is_bias || (is_ragged && !attributes.padding_mask)),
                                        error_code_t::GRAPH_NOT_SUPPORTED,
                                        "Left and right bounds are only supported with is_dropout=False, is_bias=False. Further is_ragged==True is only allowed when padding_mask=True. Lastly the diagonal alignment must be set.");
        }
        
        RETURN_CUDNN_FRONTEND_ERROR_IF(attributes.right_bound.has_value() && attributes.right_bound.value() < 0,
                                       error_code_t::INVALID_VALUE,
                                       "Right bound needs to be larger than or equal to zero");

        // validate options for dropout mask
        RETURN_CUDNN_FRONTEND_ERROR_IF(attributes.dropout_probability.has_value() && is_dropout_custom,
                                       error_code_t::ATTRIBUTE_NOT_SET,
                                       "Using both, custom dropout mask and internal-mask generation using dropout probability, is ill-formed.");

        RETURN_CUDNN_FRONTEND_ERROR_IF(attributes.dropout_probability.has_value() && attributes.dropout_probability.value() == 1.0,
                                       error_code_t::ATTRIBUTE_NOT_SET,
                                       "Dropout probability cannot be 1 as corresponding scale wont be well formed.");

        // validate options for deterministic algorithm
        if(attributes.is_deterministic_algorithm && (prop_major == 10)) {
            RETURN_CUDNN_FRONTEND_ERROR_IF( (detail::get_backend_version() < 91800),
                                        error_code_t::GRAPH_NOT_SUPPORTED,
                                        "Deterministic algorithm is not supported on blackwell architecture with cudnn version below 9.18.0");

            // dbias bias rng/dropout alibi
            RETURN_CUDNN_FRONTEND_ERROR_IF(is_dbias || is_rng || is_dropout || attributes.alibi_mask,
                                        error_code_t::GRAPH_NOT_SUPPORTED,
                                        "Deterministic algorithm is not supported on blackwell architecture when dbias, rng/dropout, alibi is enabled");

            is_deterministic_algorithm_supported_on_blackwell = true;
        }

        if(detail::get_backend_version() >= 91801) {
            RETURN_CUDNN_FRONTEND_ERROR_IF(is_ragged && (8 == prop_major || 12 == prop_major) && attributes.is_deterministic_algorithm,
                                        error_code_t::GRAPH_NOT_SUPPORTED,
                                        "Deterministic algorithm is not supported for bprop thd on SM8X and SM12X GPUs");

	    RETURN_CUDNN_FRONTEND_ERROR_IF(is_ragged && (8 == prop_major || 12 == prop_major) && attributes.inputs[input_names::Stats]->get_ragged_offset(),
                                        error_code_t::GRAPH_NOT_SUPPORTED,
                                        "Packed/ragged LSE is not supported for bprop thd on SM8X and SM12X GPUs");
	}

        // version specific validation
        RETURN_CUDNN_FRONTEND_ERROR_IF(detail::get_backend_version() < 90500 && is_dbias && attributes.padding_mask,
                                       error_code_t::GRAPH_NOT_SUPPORTED,
                                       "For cuDNN version below 9.5.0, dBias with variable sequence lengths is not supported");

        RETURN_CUDNN_FRONTEND_ERROR_IF(detail::get_backend_version() < 90500 && is_dbias && ((s_q % 64 != 0) || (s_kv % 64 != 0)),
                                       error_code_t::GRAPH_NOT_SUPPORTED,
                                       "For cuDNN version below 9.5.0, dBias not support s_q/s_kv which aren't multiple of 64");

        RETURN_CUDNN_FRONTEND_ERROR_IF(detail::get_backend_version() < 90600 && is_ragged && ((h_q != h_k) || (h_q != h_v)),
                                       error_code_t::GRAPH_NOT_SUPPORTED,
                                       "For cuDNN version below 9.6.0, group-query attention with raggged offset is not supported");

        // TODO add version check once fixed
        RETURN_CUDNN_FRONTEND_ERROR_IF(prop_major == 10 && is_ragged && is_dbias,
                                       error_code_t::GRAPH_NOT_SUPPORTED,
                                       "dbias with ragged is not supported for SM Major version 10");

        // validate that datatype is set for the graph
        RETURN_CUDNN_FRONTEND_ERROR_IF(this->context.get_intermediate_data_type() == DataType_t::NOT_SET,
                                       error_code_t::ATTRIBUTE_NOT_SET,
                                       "Intermediate tensor data type needs to be set as internal tensors require it.");
        // If dsink is set, sink also needs to be set
        RETURN_CUDNN_FRONTEND_ERROR_IF(attributes.outputs.find(output_names::DSINK_TOKEN) != attributes.outputs.end() && attributes.inputs.find(input_names::SINK_TOKEN) == attributes.inputs.end(),
                                       error_code_t::ATTRIBUTE_NOT_SET,
                                       "If dsink is set, sink also needs to be set.");
        // clang-format on

        return {error_code_t::OK, ""};
    }

    error_t
    infer_properties_node() override final {
        // clang-format off
        if (detail::get_backend_version() < 90600 && (attributes.max_total_seq_len_q.has_value() || attributes.max_total_seq_len_kv.has_value())) {
            CUDNN_FE_LOG_LABEL_ENDL("WARNING: sdpa_backward.attributes.max_total_seq_len has been set, but cuDNN version is below 9.6.0 does not support max_total_seq_len_q. The workspace memory size required to execute this graph may be unexpectedly large");
            attributes.max_total_seq_len_q.reset();
            attributes.max_total_seq_len_kv.reset();
        }

        // TODO add version check once fixed
        int64_t d_qk = attributes.inputs.at(input_names::Q)->get_dim()[3];
        int64_t d_v  = attributes.inputs.at(input_names::V)->get_dim()[3];
        if ((attributes.max_total_seq_len_q.has_value() || attributes.max_total_seq_len_kv.has_value()) && (d_qk % 16 != 0 || d_v % 16 != 0)) {
            CUDNN_FE_LOG_LABEL_ENDL("WARNING: sdpa_backward.attributes.max_total_seq_len has been set, but d is not a multiple of 16 has a known functional issue. The workspace memory size required to execute this graph may be unexpectedly large");
            attributes.max_total_seq_len_q.reset();
            attributes.max_total_seq_len_kv.reset();
        }


        if(detail::get_backend_version() >= 91801) {
            int32_t const prop_major = context.get_sm_version() / 10;
            if(8 == prop_major || 12 == prop_major) {
		if(attributes.max_total_seq_len_q.has_value() || attributes.max_total_seq_len_kv.has_value()) {
                    attributes.max_total_seq_len_q.reset();
                    attributes.max_total_seq_len_kv.reset();
		    CUDNN_FE_LOG_LABEL_ENDL("WARNING: sdpa_backward.attributes.max_total_seq_len has been set, but ampere style kernels have a known functional issue. The workspace memory size required to execute this graph may be unexpectedly large");
            
		}
	    }
        }
        // clang-format on

        return {error_code_t::OK, ""};
    }

    error_t
    expand_node() override final {
        CUDNN_FE_LOG_LABEL_ENDL("INFO:     Inferrencing properties for CompositeSDPABackwardNode " << attributes.name);

        attributes.fill_from_context(context);

        // Gather dim to fill properties of virtual tensors
        auto const& q_dim = attributes.inputs[input_names::Q]->get_dim();
        auto b            = q_dim[0];
        auto h_q          = q_dim[1];
        auto s_q          = q_dim[2];
        auto d_qk         = q_dim[3];
        auto const& k_dim = attributes.inputs[input_names::K]->get_dim();
        auto h_k          = k_dim[1];
        auto s_kv         = k_dim[2];
        auto const& v_dim = attributes.inputs[input_names::V]->get_dim();
        auto h_v          = v_dim[1];
        auto d_v          = v_dim[3];

        // cuDNN frontend API attention requires Q, K, V where
        // Q = {b, h_q, s_q, d_qk}
        // K = {b, h_k, s_kv, d_qk}
        // V = {b, h_v, s_kv, d_v}
        // but cuDNN backend API attention requires Q, KT, VT
        // Q = {b, h_q, s_q, d_qk}
        // KT = {b, h_k, d_qk, s_kv}
        // VT = {b, h_v, d_v, s_kv}
        // So the code below maps the K->KT and V->VT
        std::vector<int64_t> temp_vec;

        temp_vec = attributes.inputs[input_names::K]->get_dim();
        std::swap(temp_vec[2], temp_vec[3]);
        attributes.inputs[input_names::K]->set_dim(temp_vec);

        temp_vec = attributes.inputs[input_names::K]->get_stride();
        std::swap(temp_vec[2], temp_vec[3]);
        attributes.inputs[input_names::K]->set_stride(temp_vec);

        temp_vec = attributes.inputs[input_names::V]->get_dim();
        std::swap(temp_vec[2], temp_vec[3]);
        attributes.inputs[input_names::V]->set_dim(temp_vec);

        temp_vec = attributes.inputs[input_names::V]->get_stride();
        std::swap(temp_vec[2], temp_vec[3]);
        attributes.inputs[input_names::V]->set_stride(temp_vec);

        std::shared_ptr<Tensor_attributes> last_output, exp_s_output, dS_output, rng_output;

        // --------------Initialize and create tensors before creating nodes--------------------
        // one_tensor is needed for non-dropout graphs
        // one_tensor is passed by the node
        auto one_tensor = std::make_shared<Tensor_attributes>(1.0f);

        if (attributes.attn_scale_value.has_value()) {
            attributes.inputs[input_names::Attn_scale] =
                std::make_shared<Tensor_attributes>(attributes.attn_scale_value.value());
        }

        // if dropout_mask is used, then the user passes scale and scale_inverse
        bool is_dropout_prob = (attributes.dropout_probability.has_value());
        bool is_dropout_mask = (attributes.inputs[input_names::Dropout_mask] != nullptr);
        if (is_dropout_prob) {
            float dropout_scale_value     = 1.0f / (1.0f - attributes.dropout_probability.value());
            float dropout_scale_inv_value = (1.0f - attributes.dropout_probability.value());

            attributes.inputs[input_names::Dropout_scale] = std::make_shared<Tensor_attributes>(dropout_scale_value);
            attributes.inputs[input_names::Dropout_scale_inv] =
                std::make_shared<Tensor_attributes>(dropout_scale_inv_value);
        }

        // ---------------------input tensor workarounds---------------------------

        bool use_dp_workspace = false;

        int32_t const prop_major = context.get_sm_version() / 10;

        if (detail::get_backend_version() >= 8905 && detail::get_backend_version() < 90000) {
            // workspace optimization is enabled by default when:
            //   8.9.5 <= cudnn version < 9.0.0
            //   device >= hopper
            //   batch * num_heads * seq_len_q * seq_len_kv * 2 <= dP workspace limit
            //
            // This following environment variable allows you to control the dP workspace limit.
            // From cuDNN version 9.0.0, this option is obsolete will be ignored.
            // CUDNN_FRONTEND_ATTN_DP_WORKSPACE_LIMIT=unset  - enable workspace opt. until the default 256MB limit.
            // CUDNN_FRONTEND_ATTN_DP_WORKSPACE_LIMIT=-1     - always enable workspace opt.
            // CUDNN_FRONTEND_ATTN_DP_WORKSPACE_LIMIT=0      - always disable workspace opt.
            // CUDNN_FRONTEND_ATTN_DP_WORKSPACE_LIMIT=n      - enable workspace opt. until the n byte limit

            // hopper or above
            if (prop_major >= 9) {
                // default upper limit for workspace 256MB
                int64_t max_dp_workspace_bytes = 256 * 1024 * 1024;

                // allow setting the upper limit with envvars
                char* env_dp_workspace_limit_char = std::getenv("CUDNN_FRONTEND_ATTN_DP_WORKSPACE_LIMIT");
                if (env_dp_workspace_limit_char) {
                    char* end_ptr          = nullptr;
                    max_dp_workspace_bytes = std::strtoll(env_dp_workspace_limit_char, &end_ptr, 10);

                    if (*end_ptr != '\0') {
                        RETURN_CUDNN_FRONTEND_ERROR_IF(true,
                                                       error_code_t::ATTRIBUTE_NOT_SET,
                                                       "Invalid argument for CUDNN_FRONTEND_ATTN_DP_WORKSPACE_LIMIT "
                                                       "(int64_t; in bytes)");
                    }
                }

                int64_t workspace_s_q               = ((s_q + 64 - 1) / 64) * 64;
                int64_t workspace_s_kv              = ((s_kv + 64 - 1) / 64) * 64;
                int64_t required_dp_workspace_bytes = b * h_q * workspace_s_q * workspace_s_kv * 2;

                if (max_dp_workspace_bytes == -1) {
                    use_dp_workspace = true;
                } else if (max_dp_workspace_bytes == 0) {
                    use_dp_workspace = false;
                } else {
                    use_dp_workspace = (required_dp_workspace_bytes <= max_dp_workspace_bytes);
                }
            }
        }

        // Force dP workspace implementation if:
        //  - dBias is enabled (dBias is only supported on workspace implementation)
        //  - the user force requests deterministic algorithm on hopper
        if (attributes.outputs[output_names::dBias] || attributes.is_deterministic_algorithm) {
            use_dp_workspace = true;
        }

        // --------------RNG node--------------------

        if (is_dropout_prob) {
            if (attributes.outputs[output_names::RNG_DUMP] != nullptr) {
                rng_output = attributes.outputs[output_names::RNG_DUMP];
                rng(attributes.inputs[input_names::Seed],
                    attributes.inputs[input_names::Offset],
                    Rng_attributes()
                        .set_name("rng")
                        .set_distribution(RngDistribution_t::BERNOULLI)
                        .set_bernoulli_probability(1.0f - attributes.dropout_probability.value()),
                    rng_output);
            } else {
                rng_output = rng(attributes.inputs[input_names::Seed],
                                 attributes.inputs[input_names::Offset],
                                 Rng_attributes()
                                     .set_name("rng")
                                     .set_distribution(RngDistribution_t::BERNOULLI)
                                     .set_bernoulli_probability(1.0f - attributes.dropout_probability.value()));
                rng_output->set_dim({b, h_q, s_q, s_kv}).set_stride({h_q * s_q * s_kv, s_q * s_kv, s_kv, 1});
            }
        } else if (is_dropout_mask) {
            rng_output = attributes.inputs[input_names::Dropout_mask];
        }

        // --------------"dO * o => softmax_sum" chain--------------------

        // last_output = dO * O
        last_output = pointwise(attributes.inputs[input_names::dO],
                                attributes.inputs[input_names::O],
                                Pointwise_attributes().set_name("mul_dO_O").set_mode(PointwiseMode_t::MUL));
        last_output->set_dim({b, h_q, s_q, d_v}).set_stride({h_q * s_q * d_v, s_q * d_v, h_q * d_v, 1});

        // last_output = reduce(last_output, "b hq sq dv -> b hq sq 1")
        last_output =
            reduction(last_output, Reduction_attributes().set_name("reduce_dO_o").set_mode(ReductionMode_t::ADD));
        last_output->set_dim({b, h_q, s_q, 1}).set_stride({h_q * s_q, s_q, 1, 1});

        if (attributes.outputs.find(output_names::DSINK_TOKEN) != attributes.outputs.end()) {
            // sub_sink = sink - stats
            auto sub_sink = pointwise(attributes.inputs[input_names::SINK_TOKEN],
                                      attributes.inputs[input_names::Stats],
                                      Pointwise_attributes().set_name("sub_sink").set_mode(PointwiseMode_t::SUB));

            // exp_sink = exp(sub_sink)
            auto exp_sink =
                pointwise(sub_sink, Pointwise_attributes().set_name("exp_sink").set_mode(PointwiseMode_t::EXP));

            // per_token_grad = exp_sink * last_output
            auto per_token_grad =
                pointwise(exp_sink,
                          last_output,
                          Pointwise_attributes().set_name("mul_exp_sink_last_output").set_mode(PointwiseMode_t::MUL));

            // dSink = redduce(per_token_grad)
            reduction(per_token_grad,
                      Reduction_attributes().set_name("reduce_per_token_grad").set_mode(ReductionMode_t::ADD),
                      attributes.outputs[output_names::DSINK_TOKEN]);
        }

        // softmax_sum = last_output * dropout_scale
        last_output = pointwise(last_output,
                                attributes.inputs[input_names::Dropout_scale_inv]
                                    ? attributes.inputs[input_names::Dropout_scale_inv]
                                    : one_tensor,
                                Pointwise_attributes().set_name("scale_dropout_inv").set_mode(PointwiseMode_t::MUL));
        last_output->set_dim({b, h_q, s_q, 1}).set_stride({h_q * s_q, s_q, 1, 1});

        softmax_sum = last_output;
        softmax_sum->set_is_virtual(false);
        softmax_sum->set_dim({b, h_q, s_q, 1});
        softmax_sum->set_data_type(DataType_t::FLOAT);

        if (attributes.inputs[input_names::Stats]->get_ragged_offset() && attributes.max_total_seq_len_q.has_value()) {
            // sized TH1 softmax_sum
            softmax_sum->set_stride(attributes.inputs[input_names::Stats]->get_stride());
            softmax_sum->set_ragged_offset(attributes.inputs[input_names::Stats]->get_ragged_offset());
            softmax_sum_size = attributes.max_total_seq_len_q.value() *
                               (attributes.inputs[input_names::Stats]->get_stride())[2] * sizeof(float);
        } else {
            // sized BHS1 softmax_sum
            softmax_sum->set_stride({h_q * s_q, s_q, 1, 1});
            softmax_sum_size = b * h_q * s_q * 1 * sizeof(float);
        }

        // --------------"Q @ KT => exp_softmax => dV" chain--------------------

        // s = einsum(q, k, "b hq sq dqk, b (hk g) skv dqk -> b hq sq skv", g=hq//hk)
        last_output = matmul(attributes.inputs[input_names::Q],
                             attributes.inputs[input_names::K],
                             Matmul_attributes()
                                 .set_name("matmul_Q_KT")
                                 .set_m_override(attributes.inputs[input_names::SEQ_LEN_Q])
                                 .set_n_override(attributes.inputs[input_names::SEQ_LEN_KV]));
        last_output->set_dim({b, h_q, s_q, s_kv}).set_stride({h_q * s_q * s_kv, s_q * s_kv, s_kv, 1});

        // last_output = last_output * attention_scale
        if (attributes.inputs[input_names::Attn_scale]) {
            last_output = pointwise(last_output,
                                    attributes.inputs[input_names::Attn_scale],
                                    Pointwise_attributes().set_name("mul_s_attn_scale").set_mode(PointwiseMode_t::MUL));
        }

        if (attributes.attention_score_modifier != nullptr) {
            auto graph_                  = std::make_shared<Graph>();
            std::shared_ptr<INode> node_ = std::static_pointer_cast<INode>(graph_);
            node_->context               = context;
            last_output                  = attributes.attention_score_modifier(graph_, last_output);
            sub_nodes.emplace_back(node_);
        }

        // (optional) last_output = last_output + bias
        if (attributes.inputs.find(input_names::Bias) != attributes.inputs.end() &&
            attributes.inputs[input_names::Bias]) {
            auto graph_                  = std::make_shared<Graph>();
            std::shared_ptr<INode> node_ = std::static_pointer_cast<INode>(graph_);
            node_->context               = context;
            last_output = attn::score_modifiers::bias(graph_, last_output, attributes.inputs[input_names::Bias]);
            sub_nodes.emplace_back(node_);
        }

        // (optional) last_output = last_output + alibi_mask
        if (attributes.alibi_mask) {
            auto graph_                  = std::make_shared<Graph>();
            std::shared_ptr<INode> node_ = std::static_pointer_cast<INode>(graph_);
            node_->context               = context;
            last_output = attn::score_modifiers::alibi_mask(graph_, last_output, alibi_slopes, h_q, alibi_slopes_size);
            sub_nodes.emplace_back(node_);
        }

        // (optional) Apply padding mask
        if (attributes.padding_mask) {
            auto graph_                  = std::make_shared<Graph>();
            std::shared_ptr<INode> node_ = std::static_pointer_cast<INode>(graph_);
            node_->context               = context;
            last_output                  = attn::score_modifiers::padding_mask(graph_,
                                                              last_output,
                                                              attributes.inputs[input_names::SEQ_LEN_KV],
                                                              attributes.inputs[input_names::SEQ_LEN_Q]);
            sub_nodes.emplace_back(node_);
        }

        // last_output = last_output - stats
        last_output = pointwise(last_output,
                                attributes.inputs[input_names::Stats],
                                Pointwise_attributes().set_name("sub_s_m").set_mode(PointwiseMode_t::SUB));

        // Explicitly put the padding value again after the stats have been loaded
        if (attributes.padding_mask && detail::get_backend_version() >= 90000 &&
            detail::get_backend_version() < 91000) {
            auto row_idx_output = pointwise(last_output,
                                            Pointwise_attributes()
                                                .set_name("gen_row_idx_2nd_padding")
                                                .set_mode(PointwiseMode_t::GEN_INDEX)
                                                .set_axis(2)
                                                .set_compute_data_type(DataType_t::INT32));
            row_idx_output->set_data_type(DataType_t::INT32);

            auto col_idx_output = pointwise(last_output,
                                            Pointwise_attributes()
                                                .set_name("gen_col_idx_2nd_padding")
                                                .set_mode(PointwiseMode_t::GEN_INDEX)
                                                .set_axis(3)
                                                .set_compute_data_type(DataType_t::INT32));
            col_idx_output->set_data_type(DataType_t::INT32);

            auto row_mask_output = pointwise(row_idx_output,
                                             attributes.inputs[input_names::SEQ_LEN_Q],
                                             Pointwise_attributes()
                                                 .set_name("lt_row_sq_2nd_padding")
                                                 .set_mode(PointwiseMode_t::CMP_LT)
                                                 .set_compute_data_type(DataType_t::BOOLEAN));
            row_mask_output->set_data_type(DataType_t::BOOLEAN);

            auto col_mask_output = pointwise(col_idx_output,
                                             attributes.inputs[input_names::SEQ_LEN_KV],
                                             Pointwise_attributes()
                                                 .set_name("lt_col_skv_2nd_padding")
                                                 .set_mode(PointwiseMode_t::CMP_LT)
                                                 .set_compute_data_type(DataType_t::BOOLEAN));
            col_mask_output->set_data_type(DataType_t::BOOLEAN);

            auto padding_mask_output = pointwise(row_mask_output,
                                                 col_mask_output,
                                                 Pointwise_attributes()
                                                     .set_name("and_row_col_2nd_padding")
                                                     .set_mode(PointwiseMode_t::LOGICAL_AND)
                                                     .set_compute_data_type(DataType_t::BOOLEAN));
            padding_mask_output->set_data_type(DataType_t::BOOLEAN);
            auto negative_inf_padding =
                std::make_shared<Tensor_attributes>(attn::score_modifiers::get_negative_inf_value());

            last_output = pointwise(
                last_output,
                negative_inf_padding,
                padding_mask_output,
                Pointwise_attributes().set_name("select_2nd_padding").set_mode(PointwiseMode_t::BINARY_SELECT));
        }

        // Apply (bottom-right) causal masking (with right bound) and/or set the left bound
        if (attributes.left_bound.has_value() || attributes.right_bound.has_value()) {
            auto graph_                  = std::make_shared<Graph>();
            std::shared_ptr<INode> node_ = std::static_pointer_cast<INode>(graph_);
            node_->context               = context;

            auto s_kv_ptr = attributes.inputs.find(input_names::SEQ_LEN_KV) != attributes.inputs.end()
                                ? attributes.inputs[input_names::SEQ_LEN_KV]
                                : nullptr;
            auto s_q_ptr  = attributes.inputs.find(input_names::SEQ_LEN_Q) != attributes.inputs.end()
                                ? attributes.inputs[input_names::SEQ_LEN_Q]
                                : nullptr;

            last_output = attn::score_modifiers::sliding_window_mask(graph_,
                                                                     last_output,
                                                                     attributes.diagonal_alignment,
                                                                     attributes.left_bound,
                                                                     attributes.right_bound,
                                                                     s_q,
                                                                     s_kv,
                                                                     s_q_ptr,
                                                                     s_kv_ptr);
            sub_nodes.emplace_back(std::move(node_));
        }

        // last_output = exp(last_output)
        last_output = pointwise(last_output, Pointwise_attributes().set_name("exp_s").set_mode(PointwiseMode_t::EXP));

        exp_s_output = last_output;

        // (optional) last_output = last_output * dropout rng_output
        if (is_dropout_prob || is_dropout_mask) {
            last_output =
                pointwise(last_output,
                          rng_output,
                          Pointwise_attributes().set_name("mul_p_dropout_mask").set_mode(PointwiseMode_t::MUL));
        }

        // (optional) last_output = last_output * dropout_scale
        if (attributes.inputs[input_names::Dropout_scale]) {
            last_output =
                pointwise(last_output,
                          attributes.inputs[input_names::Dropout_scale],
                          Pointwise_attributes().set_name("mul_p_dropout_scale").set_mode(PointwiseMode_t::MUL));
        }

        // dV = einsum(p, dO, "b hq sq skv", "b hq sq dv -> b hq skv dv")
        // if GQA, then dV = reduce(dV, "b (hv g) skv dv -> b hv skv dv", g=hq//hv)
        // as reshape + matmul
        last_output = reshape(last_output, Reshape_attributes().set_name("reshape_p"));
        last_output->set_dim({b, h_q, s_kv, s_q}).set_stride({h_q * s_q * s_kv, s_q * s_kv, 1, s_kv});
        last_output->set_data_type(attributes.inputs[input_names::Q]->get_data_type());

        if (h_q == h_v) {
            // for MHA
            matmul(last_output,
                   attributes.inputs[input_names::dO],
                   Matmul_attributes()
                       .set_name("matmul_pT_dO")
                       .set_m_override(attributes.inputs[input_names::SEQ_LEN_KV])
                       .set_k_override(attributes.inputs[input_names::SEQ_LEN_Q]),
                   attributes.outputs[output_names::dV]);
        } else {
            // for GQA and MQA
            dV_fullhead = matmul(last_output,
                                 attributes.inputs[input_names::dO],
                                 Matmul_attributes()
                                     .set_name("matmul_pT_dO")
                                     .set_m_override(attributes.inputs[input_names::SEQ_LEN_KV])
                                     .set_k_override(attributes.inputs[input_names::SEQ_LEN_Q]));

            dV_fullhead->set_dim({b, h_q, s_kv, d_v});
            dV_fullhead->set_data_type(attributes.inputs[input_names::Q]->get_data_type());

            if (attributes.outputs[output_names::dV]->get_ragged_offset() &&
                attributes.max_total_seq_len_kv.has_value()) {
                // map dV strides to dV_fullhead strides
                std::vector<int64_t> dV_fullhead_stride = attributes.outputs[output_names::dV]->get_stride();
                dV_fullhead_stride[2]                   = dV_fullhead_stride[2] * (h_q / h_v);  // sequence stride
                dV_fullhead_stride[0]                   = dV_fullhead_stride[0] * (h_q / h_v);  // batch stride
                dV_fullhead->set_stride(dV_fullhead_stride);
                // map dV ragged offset to dV_fullhead ragged offset with implicit multiplier
                // implicit multiplier = h_q / h_v
                dV_fullhead->set_ragged_offset(attributes.outputs[output_names::dV]->get_ragged_offset());
                // non virtual dV full head
                dV_fullhead->set_is_virtual(false);
                dV_fullhead_size = attributes.max_total_seq_len_kv.value() * dV_fullhead_stride[2] * sizeof(float);
            } else {
                // sized BHSD dQ_accum
                dV_fullhead->set_stride({h_q * s_kv * d_v, s_kv * d_v, d_v, 1});
            }

            reduction(dV_fullhead,
                      Reduction_attributes().set_name("red_dV_head").set_mode(ReductionMode_t::ADD),
                      attributes.outputs[output_names::dV]);
        }

        // --------------"dO @ VT => dS_output => dK" chain--------------------

        // dP = einsum(dO, v, "b hq sq dv, b (hv g) skv dv -> b hq sq skv", g=hq//hv)
        last_output = matmul(attributes.inputs[input_names::dO],
                             attributes.inputs[input_names::V],
                             Matmul_attributes()
                                 .set_name("matmul_dO_VT")
                                 .set_m_override(attributes.inputs[input_names::SEQ_LEN_Q])
                                 .set_n_override(attributes.inputs[input_names::SEQ_LEN_KV]));
        last_output->set_dim({b, h_q, s_q, s_kv}).set_stride({h_q * s_q * s_kv, s_q * s_kv, s_kv, 1});

        // last_output = last_output(dP) * mask
        if (is_dropout_prob || is_dropout_mask) {
            last_output = pointwise(last_output,
                                    rng_output,
                                    Pointwise_attributes().set_name("dP_dropout_mask").set_mode(PointwiseMode_t::MUL));
        }

        // last_output = last_output - softmax_sum
        last_output = pointwise(last_output,
                                softmax_sum,
                                Pointwise_attributes().set_name("sub_dP_softmax_sum").set_mode(PointwiseMode_t::SUB));

        // last_output = last_output * exp_s_output
        last_output = pointwise(
            last_output, exp_s_output, Pointwise_attributes().set_name("mul_dP_exp_s").set_mode(PointwiseMode_t::MUL));

        // (optional) last_output = last_output * dropout_scale
        if (attributes.inputs[input_names::Dropout_scale]) {
            last_output =
                pointwise(last_output,
                          attributes.inputs[input_names::Dropout_scale],
                          Pointwise_attributes().set_name("mul_dS_dropout_scale").set_mode(PointwiseMode_t::MUL));
        }

        if (attributes.outputs[output_names::dBias]) {
            reduction(last_output,
                      Reduction_attributes().set_name("red_dP_dBias").set_mode(ReductionMode_t::ADD),
                      attributes.outputs[output_names::dBias]);
        }

        // apply the bprop of attention score modifier
        if (attributes.attention_score_modifier_bprop != nullptr) {
            auto graph_                  = std::make_shared<Graph>();
            std::shared_ptr<INode> node_ = std::static_pointer_cast<INode>(graph_);
            node_->context               = context;
            last_output                  = attributes.attention_score_modifier_bprop(graph_, last_output);
            sub_nodes.emplace_back(node_);
        }

        // (optional) last_output = last_output * bmm_scale
        if (attributes.inputs[input_names::Attn_scale]) {
            last_output =
                pointwise(last_output,
                          attributes.inputs[input_names::Attn_scale],
                          Pointwise_attributes().set_name("mul_dS_attn_scale").set_mode(PointwiseMode_t::MUL));
        }

        dS_output = last_output;

        // dK = einsum(dS, Q, "b hq sq skv", "b hq sq dqk -> b hq skv dqk")
        // if GQA, then dK = reduce(dK, "b (hk g) skv dqk -> b hk skv dqk", hq//hk)
        // as reshape + matmul
        last_output = reshape(last_output, Reshape_attributes().set_name("reshape_dS"));
        last_output->set_dim({b, h_q, s_kv, s_q}).set_stride({h_q * s_q * s_kv, s_q * s_kv, 1, s_kv});
        last_output->set_data_type(attributes.inputs[input_names::Q]->get_data_type());

        if (h_q == h_k) {
            // for MHA
            matmul(last_output,
                   attributes.inputs[input_names::Q],
                   Matmul_attributes()
                       .set_name("matmul_dST_Q")
                       .set_m_override(attributes.inputs[input_names::SEQ_LEN_KV])
                       .set_k_override(attributes.inputs[input_names::SEQ_LEN_Q]),
                   attributes.outputs[output_names::dK]);
        } else {
            // for GQA and MQA
            dK_fullhead = matmul(last_output,
                                 attributes.inputs[input_names::Q],
                                 Matmul_attributes()
                                     .set_name("matmul_dST_Q")
                                     .set_m_override(attributes.inputs[input_names::SEQ_LEN_KV])
                                     .set_k_override(attributes.inputs[input_names::SEQ_LEN_Q]));

            dK_fullhead->set_dim({b, h_q, s_kv, d_qk});
            dK_fullhead->set_data_type(attributes.inputs[input_names::Q]->get_data_type());

            if (attributes.outputs[output_names::dK]->get_ragged_offset() &&
                attributes.max_total_seq_len_kv.has_value()) {
                // sized THD dK_full_heads
                // map dK strides to dK_fullhead strides
                std::vector<int64_t> dK_fullhead_stride = attributes.outputs[output_names::dK]->get_stride();
                dK_fullhead_stride[0]                   = dK_fullhead_stride[0] * (h_q / h_k);  // batch stride
                dK_fullhead_stride[2]                   = dK_fullhead_stride[2] * (h_q / h_k);  // sequence stride
                dK_fullhead->set_stride(dK_fullhead_stride);
                // map dK ragged offset to dK_fullhead ragged offset with implicit multiplier
                // implicit multiplier = h_q / h_k
                dK_fullhead->set_ragged_offset(attributes.outputs[output_names::dK]->get_ragged_offset());
                // non virtual dK full head
                dK_fullhead->set_is_virtual(false);
                dK_fullhead_size = attributes.max_total_seq_len_kv.value() * dK_fullhead_stride[2] * sizeof(float);
            } else {
                // sized BHSD dQ_accum
                dK_fullhead->set_stride({h_q * s_kv * d_qk, s_kv * d_qk, d_qk, 1});
            }

            reduction(dK_fullhead,
                      Reduction_attributes().set_name("red_dK_head").set_mode(ReductionMode_t::ADD),
                      attributes.outputs[output_names::dK]);
        }

        // --------------"dp_scaled @ K => dQ" chain--------------------

        auto const& kt_dim    = attributes.inputs[input_names::K]->get_dim();
        auto const& kt_stride = attributes.inputs[input_names::K]->get_stride();

        // dQ = einsum(dS, K, "b hq sq skv, b (hk g) skv dqk -> b hq sq dqk", g=hq//hk)
        // as reshape + matmul
        last_output = reshape(attributes.inputs[input_names::K], Reshape_attributes().set_name("reshape_k"));
        last_output->set_dim({kt_dim[0], kt_dim[1], kt_dim[3], kt_dim[2]})
            .set_stride({kt_stride[0], kt_stride[1], kt_stride[3], kt_stride[2]});

        if (attributes.inputs[input_names::K]->get_ragged_offset() != nullptr) {
            last_output->set_ragged_offset(attributes.inputs[input_names::K]->get_ragged_offset());
        }

        if (!use_dp_workspace) {
            dQ_accum = std::make_shared<Tensor_attributes>();
            dQ_accum->set_is_virtual(false);
            dQ_accum->set_dim({b, h_q, s_q, d_qk});
            dQ_accum->set_data_type(DataType_t::FLOAT);

            if (attributes.outputs[output_names::dQ]->get_ragged_offset() &&
                attributes.max_total_seq_len_q.has_value()) {
                // sized THD dQ_accum
                dQ_accum->set_stride(attributes.outputs[output_names::dQ]->get_stride());
                dQ_accum->set_ragged_offset(attributes.outputs[output_names::dQ]->get_ragged_offset());
                dQ_accum_size = attributes.max_total_seq_len_q.value() *
                                (attributes.outputs[output_names::dQ]->get_stride())[2] * sizeof(float);
            } else {
                // sized BHSD dQ_accum
                dQ_accum->set_stride({h_q * s_q * d_qk, s_q * d_qk, d_qk, 1});
                dQ_accum_size = b * h_q * s_q * d_qk * sizeof(float);
            }

            matmul(dS_output,
                   last_output,
                   Matmul_attributes()
                       .set_name("matmul_dS_K")
                       .set_m_override(attributes.inputs[input_names::SEQ_LEN_Q])
                       .set_k_override(attributes.inputs[input_names::SEQ_LEN_KV]),
                   dQ_accum);

            pointwise(dQ_accum,
                      Pointwise_attributes().set_name("identity_dQ").set_mode(PointwiseMode_t::IDENTITY),
                      attributes.outputs[output_names::dQ]);
        } else {
            matmul(dS_output,
                   last_output,
                   Matmul_attributes()
                       .set_name("matmul_dS_K")
                       .set_m_override(attributes.inputs[input_names::SEQ_LEN_Q])
                       .set_k_override(attributes.inputs[input_names::SEQ_LEN_KV]),
                   attributes.outputs[output_names::dQ]);
        }

        return {error_code_t::OK, ""};
    }

    std::pair<int64_t, std::unordered_map<KnobType_t, int64_t>>
    override_heuristics_query() const {
        int32_t const sm_version = context.get_sm_version();
        bool const use_new_knobs = detail::get_backend_version() >= 92300;
        // {128,128} bprop: tileM=3, tileN=2, kernelCfg=2(bprop warp), streamK=0, cgaM=0
        if (sm_version > 103 && (is_deterministic_algorithm_supported_on_blackwell)) {
            if (use_new_knobs) {
                return {17,
                        {{KnobType_t::TILE_M, 3},
                         {KnobType_t::TILE_N, 2},
                         {KnobType_t::KERNEL_CFG, 2},
                         {KnobType_t::STREAM_K, 0},
                         {KnobType_t::TILE_CGA_M, 0},
                         {KnobType_t::STAGES, is_d256_on_blackwell ? 3 : 2}}};
            } else {
                return {17, {{KnobType_t::KERNEL_CFG, 31}, {KnobType_t::STAGES, is_d256_on_blackwell ? 3 : 2}}};
            }
        } else if (is_deterministic_algorithm_supported_on_blackwell) {
            if (use_new_knobs) {
                return {5,
                        {{KnobType_t::TILE_M, 3},
                         {KnobType_t::TILE_N, 2},
                         {KnobType_t::KERNEL_CFG, 2},
                         {KnobType_t::STREAM_K, 0},
                         {KnobType_t::TILE_CGA_M, 0},
                         {KnobType_t::STAGES, is_d256_on_blackwell ? 3 : 2}}};
            } else {
                return {5, {{KnobType_t::KERNEL_CFG, 31}, {KnobType_t::STAGES, is_d256_on_blackwell ? 3 : 2}}};
            }
        } else {
            return {-1, {}};
        }
    }

    virtual int64_t
    get_fe_workspace_size_node() const override final {
        int64_t size = 0;

        size += ((alibi_slopes_size + 15) / 16 * 16);  // align alibi slopes memory to 16 bytes
        size += dQ_accum_size;
        size += dK_fullhead_size;
        size += dV_fullhead_size;
        size += softmax_sum_size;

        if (has_workaround_padding_mask) {
            size += batch_size_for_workaround_padding_mask * sizeof(int32_t) * 2;
        }

        return size;
    }

    virtual error_t
    collect_tensors_in_workspace_node(
        std::unordered_map<Tensor_attributes::uid_t, std::tuple<int64_t, int64_t, std::vector<float>>>&
            workspace_modifications,
        int64_t& offset) const override final {
        if (attributes.alibi_mask) {
            CUDNN_FE_VALIDATE_AND_ASSIGN_INPUT_TENSOR(Q, input_names::Q);
            int64_t const h_q     = Q->second->get_dim()[1];
            auto alibi_slopes_vec = detail::get_alibi_slope(h_q);
            workspace_modifications.emplace(alibi_slopes->get_uid(), std::make_tuple(0, offset, alibi_slopes_vec));
            int64_t alibi_slopes_size_padded = ((alibi_slopes_size + 15) / 16 * 16);
            offset                           = offset + alibi_slopes_size_padded;
        }

        if (dQ_accum && !dQ_accum->get_is_virtual()) {
            if (detail::get_backend_version() < 90600) {
                // prior to cuDNN 9.6.0, dQ_accum needed to be memset by frontend
                workspace_modifications.emplace(dQ_accum->get_uid(),
                                                std::make_tuple(1, offset, std::vector<float>{(float)dQ_accum_size}));
            } else {
                workspace_modifications.emplace(dQ_accum->get_uid(), std::make_tuple(2, offset, std::vector<float>()));
            }
            offset = offset + dQ_accum_size;
        }

        if (dK_fullhead && !dK_fullhead->get_is_virtual()) {
            workspace_modifications.emplace(dK_fullhead->get_uid(), std::make_tuple(2, offset, std::vector<float>()));
            offset = offset + dK_fullhead_size;
        }

        if (dV_fullhead && !dV_fullhead->get_is_virtual()) {
            workspace_modifications.emplace(dV_fullhead->get_uid(), std::make_tuple(2, offset, std::vector<float>()));
            offset = offset + dV_fullhead_size;
        }

        if (softmax_sum && !softmax_sum->get_is_virtual()) {
            workspace_modifications.emplace(softmax_sum->get_uid(), std::make_tuple(2, offset, std::vector<float>()));
            offset = offset + softmax_sum_size;
        }

        if (has_workaround_padding_mask) {
            CUDNN_FE_LOG_LABEL_ENDL("INFO: Collecting workaround padding mask tensors with batch size "
                                    << batch_size_for_workaround_padding_mask << " with UIDs "
                                    << workaround_padding_mask_seq_len_q->get_uid() << " and "
                                    << workaround_padding_mask_seq_len_kv->get_uid());
            std::vector<int32_t> workaround_padding_mask_seq_len_q_vec(batch_size_for_workaround_padding_mask,
                                                                       s_q_for_workaround_padding_mask);
            std::vector<int32_t> workaround_padding_mask_seq_len_kv_vec(batch_size_for_workaround_padding_mask,
                                                                        s_kv_for_workaround_padding_mask);

            // reinterpret_cast the int32_t vector data to float vector for workspace_modifications
            std::vector<float> workaround_padding_mask_seq_len_q_vec_float(
                reinterpret_cast<float*>(workaround_padding_mask_seq_len_q_vec.data()),
                reinterpret_cast<float*>(workaround_padding_mask_seq_len_q_vec.data()) +
                    batch_size_for_workaround_padding_mask);
            std::vector<float> workaround_padding_mask_seq_len_kv_vec_float(
                reinterpret_cast<float*>(workaround_padding_mask_seq_len_kv_vec.data()),
                reinterpret_cast<float*>(workaround_padding_mask_seq_len_kv_vec.data()) +
                    batch_size_for_workaround_padding_mask);

            workspace_modifications.emplace(workaround_padding_mask_seq_len_q->get_uid(),
                                            std::make_tuple(0, offset, workaround_padding_mask_seq_len_q_vec_float));
            offset = offset + batch_size_for_workaround_padding_mask * sizeof(float);
            workspace_modifications.emplace(workaround_padding_mask_seq_len_kv->get_uid(),
                                            std::make_tuple(0, offset, workaround_padding_mask_seq_len_kv_vec_float));
            offset = offset + batch_size_for_workaround_padding_mask * sizeof(float);
        }

        return {error_code_t::OK, ""};
    }

#ifndef CUDNN_FRONTEND_SKIP_JSON_LIB
    virtual void
    serialize(json& j) const override final {
        j = attributes;
        j.update(R"({"tag": "SDPA_BWD"})"_json);
    }
#endif
};

class UnifiedSDPANode : public SDPANodeBase<UnifiedSDPANode> {
   public:
    UnifiedSDPANode(SDPA_attributes&& attributes_, detail::Context const& context)
        : SDPANodeBase(std::move(attributes_), context) {}

    Type
    getType() override final {
        return Type::SDPA;
    }

    error_t
    expand_node() override final {
        CUDNN_FE_LOG_LABEL_ENDL("INFO:     Inferrencing properties for UnifiedSDPANode node  " << attributes.name);

        // DO NOT REMOVE
        // input data type is needed for:
        // - aType of bmm2
        // - dropout scale in pre 8.9.3
        attributes.fill_from_context(this->context);

        //// Optional Attn scale
        // In case user provided a scalar value, do a fused scalar.
        if (attributes.attn_scale_value.has_value()) {
            attributes.inputs[input_names::Attn_scale] =
                std::make_shared<Tensor_attributes>(attributes.attn_scale_value.value());
        }

        // Optional attention score modifier
        if (attributes.attention_score_modifier != nullptr) {
            if (!subgraph) init_subgraph();
            subgraph_output = attributes.attention_score_modifier(subgraph, subgraph_output);
        }

        // Optional bias
        if (attributes.inputs.find(input_names::Bias) != attributes.inputs.end() &&
            attributes.inputs[input_names::Bias]) {
            if (!subgraph) init_subgraph();
            subgraph_output =
                attn::score_modifiers::bias(subgraph, subgraph_output, attributes.inputs[input_names::Bias]);
        }

        // Optional alibi mask
        if (attributes.alibi_mask) {
            if (!subgraph) init_subgraph();
            auto h_q = attributes.inputs[input_names::Q]->get_dim()[1];
            subgraph_output =
                attn::score_modifiers::alibi_mask(subgraph, subgraph_output, alibi_slopes, h_q, alibi_slopes_size);
        }

        // Optional casual masking
        if (attributes.left_bound.has_value() || attributes.right_bound.has_value()) {
            if (!subgraph) init_subgraph();

            auto s_q      = attributes.inputs[input_names::Q]->get_dim()[2];
            auto s_kv     = infer_s_kv();
            auto s_kv_ptr = attributes.inputs.find(input_names::SEQ_LEN_KV) != attributes.inputs.end()
                                ? attributes.inputs[input_names::SEQ_LEN_KV]
                                : nullptr;
            auto s_q_ptr  = attributes.inputs.find(input_names::SEQ_LEN_Q) != attributes.inputs.end()
                                ? attributes.inputs[input_names::SEQ_LEN_Q]
                                : nullptr;

            subgraph_output = attn::score_modifiers::sliding_window_mask(subgraph,
                                                                         subgraph_output,
                                                                         attributes.diagonal_alignment,
                                                                         attributes.left_bound,
                                                                         attributes.right_bound,
                                                                         s_q,
                                                                         s_kv,
                                                                         s_q_ptr,
                                                                         s_kv_ptr);
        }

        if (subgraph) {
            subgraph_output->set_name(attributes.name + "::subgraph_output");
            auto subgraph_node = std::static_pointer_cast<INode>(subgraph);
            sub_nodes.emplace_back(subgraph_node);
        }

        // For BE >= 9.21, express softmax as a UnifiedSoftmaxNode whose backend descriptor
        // will be set via CUDNN_ATTR_OPERATION_SDPA_FWD_SOFTMAX_DESC.
        // For older versions, stats (if present) is set directly via CUDNN_ATTR_OPERATION_SDPA_FWD_STATSDESC.
        auto effective_cudnn_ver = std::min(detail::get_compiled_version(), detail::get_backend_version());

        auto has_output = [&](auto name) {
            auto it = attributes.outputs.find(name);
            return it != attributes.outputs.end() && it->second != nullptr;
        };
        bool has_softmax_features = has_output(output_names::Stats) || has_output(output_names::Max) ||
                                    has_output(output_names::Sum_exp) || attributes.has_sink_token();

        if (effective_cudnn_ver >= 92100 && has_softmax_features) {
            auto b    = attributes.inputs[input_names::Q]->get_dim()[0];
            auto h_q  = attributes.inputs[input_names::Q]->get_dim()[1];
            auto s_q  = attributes.inputs[input_names::Q]->get_dim()[2];
            auto s_kv = infer_s_kv();

            auto softmax_p = std::make_shared<Tensor_attributes>();
            softmax_p->set_is_virtual(true);
            softmax_p->set_name(attributes.name + "::softmax_P");
            softmax_p->set_dim({b, h_q, s_q, s_kv});
            softmax_p->set_stride({h_q * s_q * s_kv, s_q * s_kv, s_kv, 1});

            auto softmax_s = std::make_shared<Tensor_attributes>();
            softmax_s->set_is_virtual(true);
            softmax_s->set_name(attributes.name + "::softmax_S");
            softmax_s->set_dim({b, h_q, s_q, s_kv});
            softmax_s->set_stride({h_q * s_q * s_kv, s_q * s_kv, s_kv, 1});

            auto softmax_attrs = Softmax_attributes().set_name(attributes.name + "::softmax");
            softmax_attrs.inputs[Softmax_attributes::input_names::P]   = softmax_p;
            softmax_attrs.outputs[Softmax_attributes::output_names::S] = softmax_s;

            if (attributes.has_sink_token()) {
                softmax_attrs.set_sink(attributes.inputs[input_names::SINK_TOKEN]);
            }
            if (has_output(output_names::Stats)) {
                softmax_attrs.outputs[Softmax_attributes::output_names::Stats] =
                    attributes.outputs[output_names::Stats];
            }
            if (has_output(output_names::Max)) {
                softmax_attrs.outputs[Softmax_attributes::output_names::Max] = attributes.outputs[output_names::Max];
            }
            if (has_output(output_names::Sum_exp)) {
                softmax_attrs.outputs[Softmax_attributes::output_names::Sum_exp] =
                    attributes.outputs[output_names::Sum_exp];
            }

            softmax_node = std::make_shared<UnifiedSoftmaxNode>(std::move(softmax_attrs), context);
            sub_nodes.emplace_back(softmax_node);
        }

        return {error_code_t::OK, ""};
    }

    error_t
    create_cudnn_operations(
        std::unordered_set<Tensor_attributes::uid_t>& uids_involved_in_operations,
        std::vector<std::shared_ptr<cudnn_frontend::Operation>>& operations,
        managed_backend_descriptor_t& raw_operations,
        std::unordered_map<int64_t, std::shared_ptr<cudnn_frontend::Tensor>>& tensors) const override final {
        CUDNN_FRONTEND_UNUSED(operations);
        CUDNN_FE_LOG_LABEL("INFO: " << "Building UnifiedSDPANode operations " << attributes.name << " ");
        auto cudnn_ver_error = error_t{error_code_t::GRAPH_NOT_SUPPORTED, "Unified SDPA node requires cuDNN 9.13.1"};

#if (CUDNN_VERSION >= 91301)
        NV_CUDNN_FE_DYNAMIC_CHECK_CUDNN_BACKEND_VERSION(91301, cudnn_ver_error);
        auto unified_sdpa_operation =
            make_shared_backend_pointer((cudnnBackendDescriptorType_t)CUDNN_BACKEND_OPERATION_SDPA_FWD_DESCRIPTOR);

        auto Q         = attributes.inputs.find(SDPA_attributes::input_names::Q)->second;
        auto backend_q = tensors[Q->get_uid()]->get_desc()->get_backend_descriptor();
        _CUDNN_CHECK_CUDNN_ERROR(detail::set_attribute(unified_sdpa_operation->get_backend_descriptor(),
                                                       CUDNN_ATTR_OPERATION_SDPA_FWD_QDESC,
                                                       CUDNN_TYPE_BACKEND_DESCRIPTOR,
                                                       1,
                                                       &backend_q));

        auto K         = attributes.inputs.find(SDPA_attributes::input_names::K)->second;
        auto backend_k = tensors[K->get_uid()]->get_desc()->get_backend_descriptor();
        _CUDNN_CHECK_CUDNN_ERROR(detail::set_attribute(unified_sdpa_operation->get_backend_descriptor(),
                                                       CUDNN_ATTR_OPERATION_SDPA_FWD_KDESC,
                                                       CUDNN_TYPE_BACKEND_DESCRIPTOR,
                                                       1,
                                                       &backend_k));

        auto V         = attributes.inputs.find(SDPA_attributes::input_names::V)->second;
        auto backend_v = tensors[V->get_uid()]->get_desc()->get_backend_descriptor();
        _CUDNN_CHECK_CUDNN_ERROR(detail::set_attribute(unified_sdpa_operation->get_backend_descriptor(),
                                                       CUDNN_ATTR_OPERATION_SDPA_FWD_VDESC,
                                                       CUDNN_TYPE_BACKEND_DESCRIPTOR,
                                                       1,
                                                       &backend_v));

        auto O         = attributes.outputs.find(SDPA_attributes::output_names::O)->second;
        auto backend_o = tensors[O->get_uid()]->get_desc()->get_backend_descriptor();
        _CUDNN_CHECK_CUDNN_ERROR(detail::set_attribute(unified_sdpa_operation->get_backend_descriptor(),
                                                       CUDNN_ATTR_OPERATION_SDPA_FWD_ODESC,
                                                       CUDNN_TYPE_BACKEND_DESCRIPTOR,
                                                       1,
                                                       &backend_o));

        if (softmax_node) {
            auto softmax_ver_error =
                error_t{error_code_t::GRAPH_NOT_SUPPORTED, "SOFTMAX_DESC in unified SDPA node requires cuDNN 9.21.0"};
#if (CUDNN_VERSION >= 92100)
            NV_CUDNN_FE_DYNAMIC_CHECK_CUDNN_BACKEND_VERSION(92100, softmax_ver_error);

            managed_backend_descriptor_t softmax_raw_ops;
            std::unordered_set<Tensor_attributes::uid_t> softmax_uids;
            std::vector<std::shared_ptr<cudnn_frontend::Operation>> softmax_ops;
            CHECK_CUDNN_FRONTEND_ERROR(
                softmax_node->create_cudnn_operations(softmax_uids, softmax_ops, softmax_raw_ops, tensors));

            auto backend_softmax = softmax_raw_ops[0]->get_backend_descriptor();
            _CUDNN_CHECK_CUDNN_ERROR(detail::set_attribute(unified_sdpa_operation->get_backend_descriptor(),
                                                           CUDNN_ATTR_OPERATION_SDPA_FWD_SOFTMAX_DESC,
                                                           CUDNN_TYPE_BACKEND_DESCRIPTOR,
                                                           1,
                                                           &backend_softmax));

            uids_involved_in_operations.insert(softmax_uids.begin(), softmax_uids.end());
#else
            return softmax_ver_error;
#endif
        } else {
            auto stats_it = attributes.outputs.find(SDPA_attributes::output_names::Stats);
            if (stats_it != attributes.outputs.end() && stats_it->second) {
                auto backend_stats = tensors[stats_it->second->get_uid()]->get_desc()->get_backend_descriptor();
                _CUDNN_CHECK_CUDNN_ERROR(detail::set_attribute(unified_sdpa_operation->get_backend_descriptor(),
                                                               CUDNN_ATTR_OPERATION_SDPA_FWD_STATSDESC,
                                                               CUDNN_TYPE_BACKEND_DESCRIPTOR,
                                                               1,
                                                               &backend_stats));
            }
        }

        auto attn_scale_it = attributes.inputs.find(SDPA_attributes::input_names::Attn_scale);
        if (attn_scale_it != attributes.inputs.end()) {
            auto backend_scale = tensors[attn_scale_it->second->get_uid()]->get_desc()->get_backend_descriptor();
            _CUDNN_CHECK_CUDNN_ERROR(detail::set_attribute(unified_sdpa_operation->get_backend_descriptor(),
                                                           CUDNN_ATTR_OPERATION_SDPA_FWD_SCALEDESC,
                                                           CUDNN_TYPE_BACKEND_DESCRIPTOR,
                                                           1,
                                                           &backend_scale));
        }

        auto block_mask_it = attributes.inputs.find(SDPA_attributes::input_names::Block_mask);
        if (block_mask_it != attributes.inputs.end() && block_mask_it->second != nullptr) {
            auto block_mask_cudnn_ver_error =
                error_t{error_code_t::GRAPH_NOT_SUPPORTED, "Block mask in unified SDPA node requires cuDNN 9.14.0"};
#if CUDNN_VERSION >= 91400
            NV_CUDNN_FE_DYNAMIC_CHECK_CUDNN_BACKEND_VERSION(91400, block_mask_cudnn_ver_error);
            auto backend_block_mask = tensors[block_mask_it->second->get_uid()]->get_desc()->get_backend_descriptor();
            _CUDNN_CHECK_CUDNN_ERROR(detail::set_attribute(unified_sdpa_operation->get_backend_descriptor(),
                                                           CUDNN_ATTR_OPERATION_SDPA_FWD_BLOCK_MASK_DESC,
                                                           CUDNN_TYPE_BACKEND_DESCRIPTOR,
                                                           1,
                                                           &backend_block_mask));
#else
            return block_mask_cudnn_ver_error;
#endif
        }

        // Paged attention attributes
        if (is_paged_k() || is_paged_v() || has_seq_len_q() || has_seq_len_kv()) {
            auto paged_cudnn_ver_error = error_t{error_code_t::GRAPH_NOT_SUPPORTED,
                                                 "Paged attention in unified SDPA node requires cuDNN 9.15.0"};
#if (CUDNN_VERSION >= 91500)
            NV_CUDNN_FE_DYNAMIC_CHECK_CUDNN_BACKEND_VERSION(91500, paged_cudnn_ver_error);

            if (is_paged_k()) {
                auto page_table_K         = attributes.inputs.find(SDPA_attributes::input_names::Page_table_K)->second;
                auto backend_page_table_K = tensors[page_table_K->get_uid()]->get_desc()->get_backend_descriptor();
                _CUDNN_CHECK_CUDNN_ERROR(detail::set_attribute(unified_sdpa_operation->get_backend_descriptor(),
                                                               CUDNN_ATTR_OPERATION_SDPA_FWD_PAGE_TABLE_KDESC,
                                                               CUDNN_TYPE_BACKEND_DESCRIPTOR,
                                                               1,
                                                               &backend_page_table_K));
            }

            if (is_paged_v()) {
                auto page_table_V         = attributes.inputs.find(SDPA_attributes::input_names::Page_table_V)->second;
                auto backend_page_table_V = tensors[page_table_V->get_uid()]->get_desc()->get_backend_descriptor();
                _CUDNN_CHECK_CUDNN_ERROR(detail::set_attribute(unified_sdpa_operation->get_backend_descriptor(),
                                                               CUDNN_ATTR_OPERATION_SDPA_FWD_PAGE_TABLE_VDESC,
                                                               CUDNN_TYPE_BACKEND_DESCRIPTOR,
                                                               1,
                                                               &backend_page_table_V));
            }

            if (has_seq_len_q()) {
                auto seq_len_Q         = attributes.inputs.find(SDPA_attributes::input_names::SEQ_LEN_Q)->second;
                auto backend_seq_len_Q = tensors[seq_len_Q->get_uid()]->get_desc()->get_backend_descriptor();
                _CUDNN_CHECK_CUDNN_ERROR(detail::set_attribute(unified_sdpa_operation->get_backend_descriptor(),
                                                               CUDNN_ATTR_OPERATION_SDPA_FWD_SEQ_LEN_QDESC,
                                                               CUDNN_TYPE_BACKEND_DESCRIPTOR,
                                                               1,
                                                               &backend_seq_len_Q));
            }

            if (has_seq_len_kv()) {
                auto seq_len_KV         = attributes.inputs.find(SDPA_attributes::input_names::SEQ_LEN_KV)->second;
                auto backend_seq_len_KV = tensors[seq_len_KV->get_uid()]->get_desc()->get_backend_descriptor();
                _CUDNN_CHECK_CUDNN_ERROR(detail::set_attribute(unified_sdpa_operation->get_backend_descriptor(),
                                                               CUDNN_ATTR_OPERATION_SDPA_FWD_SEQ_LEN_KVDESC,
                                                               CUDNN_TYPE_BACKEND_DESCRIPTOR,
                                                               1,
                                                               &backend_seq_len_KV));
            }

            // Ignore attributes.max_seq_len_kv, because unified engine doesn't need it (it's harmless if set).

            // Ignore attributes.padding_mask, because unified engine already applies an implicit padding mask
            // if seq_len_Q and seq_len_KV are both provided. We already checked in
            // `SDPA_attributes::validate_sdpa_support_surface()` that padding_mask must be true if and
            // only if seq_len_Q and seq_len_KV are both set, so we don't need to check it here.
#else
            return paged_cudnn_ver_error;
#endif
        }

        // Subgraph attributes
        if (subgraph) {
            auto subgraph_cudnn_ver_error = error_t{error_code_t::GRAPH_NOT_SUPPORTED,
                                                    "Subgraph attributes in unified SDPA node requires cuDNN 9.21.0"};
#if (CUDNN_VERSION >= 92100)
            NV_CUDNN_FE_DYNAMIC_CHECK_CUDNN_BACKEND_VERSION(92100, subgraph_cudnn_ver_error);

            CHECK_CUDNN_FRONTEND_ERROR(attn::score_modifiers::build_operation_subgraph(subgraph));
            auto subgraph_cudnn   = std::static_pointer_cast<ICudnn>(subgraph);
            auto backend_subgraph = subgraph_cudnn->get_operation_graph()->get_raw_desc();
            _CUDNN_CHECK_CUDNN_ERROR(detail::set_attribute(unified_sdpa_operation->get_backend_descriptor(),
                                                           CUDNN_ATTR_OPERATION_SDPA_FWD_SUBGRAPH,
                                                           CUDNN_TYPE_BACKEND_DESCRIPTOR,
                                                           1,
                                                           &backend_subgraph));

            auto subgraph_input_uid = subgraph_input->get_uid();
            _CUDNN_CHECK_CUDNN_ERROR(detail::set_attribute(unified_sdpa_operation->get_backend_descriptor(),
                                                           CUDNN_ATTR_OPERATION_SDPA_FWD_SUBGRAPH_INPUT_UID,
                                                           CUDNN_TYPE_INT64,
                                                           1,
                                                           &subgraph_input_uid));

            auto subgraph_output_uid = subgraph_output->get_uid();
            _CUDNN_CHECK_CUDNN_ERROR(detail::set_attribute(unified_sdpa_operation->get_backend_descriptor(),
                                                           CUDNN_ATTR_OPERATION_SDPA_FWD_SUBGRAPH_OUTPUT_UID,
                                                           CUDNN_TYPE_INT64,
                                                           1,
                                                           &subgraph_output_uid));

            // Add non-virtual uids from subgraph to uids_involved_in_operations
            uids_involved_in_operations.insert(subgraph_cudnn->variant_pack_uids.begin(),
                                               subgraph_cudnn->variant_pack_uids.end());
#else
            return subgraph_cudnn_ver_error;
#endif
        }

        // Set unfuse_fma attribute (for SM100: use __fmul_rn + __fadd_rn instead of ffma2 in softmax)
        if (attributes.unfuse_fma) {
            auto unfuse_fma_cudnn_ver_error =
                error_t{error_code_t::GRAPH_NOT_SUPPORTED, "Unfuse FMA in unified SDPA node requires cuDNN 9.21.0"};
#if CUDNN_VERSION >= 92100
            NV_CUDNN_FE_DYNAMIC_CHECK_CUDNN_BACKEND_VERSION(92100, unfuse_fma_cudnn_ver_error);
            bool unfuse_fma_value = true;
            _CUDNN_CHECK_CUDNN_ERROR(detail::set_attribute(unified_sdpa_operation->get_backend_descriptor(),
                                                           CUDNN_ATTR_OPERATION_SDPA_FWD_UNFUSE_FMA,
                                                           CUDNN_TYPE_BOOLEAN,
                                                           1,
                                                           &unfuse_fma_value));
#else
            return unfuse_fma_cudnn_ver_error;
#endif
        }

        // Dropout attributes
        if (attributes.dropout_probability.has_value() && attributes.dropout_probability.value() != 0.0f) {
            auto dropout_cudnn_ver_error =
                error_t{error_code_t::GRAPH_NOT_SUPPORTED, "Dropout in unified SDPA node requires cuDNN 9.21.0"};
#if (CUDNN_VERSION >= 92100)
            NV_CUDNN_FE_DYNAMIC_CHECK_CUDNN_BACKEND_VERSION(92100, dropout_cudnn_ver_error);

            float dropout_prob = attributes.dropout_probability.value();
            _CUDNN_CHECK_CUDNN_ERROR(detail::set_attribute(unified_sdpa_operation->get_backend_descriptor(),
                                                           CUDNN_ATTR_OPERATION_SDPA_FWD_DROPOUT_PROBABILITY,
                                                           CUDNN_TYPE_FLOAT,
                                                           1,
                                                           &dropout_prob));

            auto seed_it = attributes.inputs.find(SDPA_attributes::input_names::Seed);
            if (seed_it != attributes.inputs.end() && seed_it->second != nullptr) {
                auto backend_seed = tensors[seed_it->second->get_uid()]->get_desc()->get_backend_descriptor();
                _CUDNN_CHECK_CUDNN_ERROR(detail::set_attribute(unified_sdpa_operation->get_backend_descriptor(),
                                                               CUDNN_ATTR_OPERATION_SDPA_FWD_DROPOUT_SEED_DESC,
                                                               CUDNN_TYPE_BACKEND_DESCRIPTOR,
                                                               1,
                                                               &backend_seed));
            }

            auto offset_it = attributes.inputs.find(SDPA_attributes::input_names::Offset);
            if (offset_it != attributes.inputs.end() && offset_it->second != nullptr) {
                auto backend_offset = tensors[offset_it->second->get_uid()]->get_desc()->get_backend_descriptor();
                _CUDNN_CHECK_CUDNN_ERROR(detail::set_attribute(unified_sdpa_operation->get_backend_descriptor(),
                                                               CUDNN_ATTR_OPERATION_SDPA_FWD_DROPOUT_OFFSET_DESC,
                                                               CUDNN_TYPE_BACKEND_DESCRIPTOR,
                                                               1,
                                                               &backend_offset));
            }

            auto rng_dump_it = attributes.outputs.find(SDPA_attributes::output_names::RNG_DUMP);
            if (rng_dump_it != attributes.outputs.end() && rng_dump_it->second != nullptr) {
                auto backend_rng_dump = tensors[rng_dump_it->second->get_uid()]->get_desc()->get_backend_descriptor();
                _CUDNN_CHECK_CUDNN_ERROR(detail::set_attribute(unified_sdpa_operation->get_backend_descriptor(),
                                                               CUDNN_ATTR_OPERATION_SDPA_FWD_DROPOUT_RNG_DUMP_DESC,
                                                               CUDNN_TYPE_BACKEND_DESCRIPTOR,
                                                               1,
                                                               &backend_rng_dump));
            }
#else
            return dropout_cudnn_ver_error;
#endif
        }

        _CUDNN_CHECK_CUDNN_ERROR(detail::finalize(unified_sdpa_operation->get_backend_descriptor()));

        raw_operations.push_back(unified_sdpa_operation);

        auto const& non_virtual_uids = attributes.get_non_virtual_uids();
        uids_involved_in_operations.insert(non_virtual_uids.begin(), non_virtual_uids.end());
        return {error_code_t::OK, ""};
#else
        CUDNN_FRONTEND_UNUSED(uids_involved_in_operations);
        CUDNN_FRONTEND_UNUSED(raw_operations);
        CUDNN_FRONTEND_UNUSED(tensors);
        return cudnn_ver_error;
#endif  // CUDNN_VERSION >= 91301
    }

   protected:
    std::shared_ptr<Graph> subgraph;  // Pre-softmax subgraph
    std::shared_ptr<Tensor_attributes> subgraph_input;
    std::shared_ptr<Tensor_attributes> subgraph_output;
    std::shared_ptr<UnifiedSoftmaxNode> softmax_node;  // Softmax descriptor for BE >= 9.21

    void
    init_subgraph() {
        subgraph               = std::make_shared<Graph>();
        auto subgraph_node     = std::static_pointer_cast<INode>(subgraph);
        subgraph_node->context = context;
        subgraph_input         = std::make_shared<Tensor_attributes>();
        subgraph_input->set_is_virtual(true);
        subgraph_input->set_name(attributes.name + "::subgraph_input");
        auto b    = attributes.inputs[input_names::Q]->get_dim()[0];
        auto h_q  = attributes.inputs[input_names::Q]->get_dim()[1];
        auto s_q  = attributes.inputs[input_names::Q]->get_dim()[2];
        auto s_kv = infer_s_kv();
        subgraph_input->set_dim({b, h_q, s_q, s_kv});
        subgraph_input->set_stride({h_q * s_q * s_kv, s_q * s_kv, s_kv, 1});
        subgraph_output = subgraph_input;
    }
};

}  // namespace cudnn_frontend::graph
