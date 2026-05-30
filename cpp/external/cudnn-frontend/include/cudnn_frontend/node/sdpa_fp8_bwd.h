#pragma once

#include "../../cudnn_frontend_Heuristics.h"
#include "../../cudnn_frontend_Logging.h"

#include "../graph_helpers.h"
#include "../node_interface.h"

#include "diagonal_band_mask.h"
#include "matmul_fp8.h"
#include "pointwise.h"
#include "reduction.h"
#include "softmax.h"
#include "block_scale_dequantize.h"

namespace cudnn_frontend::graph {

class SDPAFP8BackwardNode : public NodeCRTP<SDPAFP8BackwardNode> {
    using input_names  = SDPA_fp8_backward_attributes::input_names;
    using output_names = SDPA_fp8_backward_attributes::output_names;

   private:
    mutable bool is_deterministic_algorithm_supported_on_blackwell = false;  // Will be edited in pre_validate_node()

   public:
    mutable SDPA_fp8_backward_attributes
        attributes;  // mutable to allow auto-routing to deterministic in pre_validate_node()

    SDPAFP8BackwardNode(SDPA_fp8_backward_attributes&& attributes_, detail::Context const& context)
        : NodeCRTP(context), attributes(std::move(attributes_)) {}

    Type
    getType() override final {
        return Type::COMPOSITE;
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

    error_t
    pre_validate_node() const override final {
        CUDNN_FE_LOG_LABEL_ENDL("INFO: Validating SDPAFP8BackwardNode " << attributes.name);

        RETURN_CUDNN_FRONTEND_ERROR_IF(detail::get_backend_version() < 90100,
                                       error_code_t::GRAPH_NOT_SUPPORTED,
                                       "sdpa fp8 backward operation is only supported starting cudnn 9.1.0. Please "
                                       "consider upgrading your current version.");

        // Bug workaround for 9.10.0 (TE constraint)
        RETURN_CUDNN_FRONTEND_ERROR_IF(
            detail::get_backend_version() == 91000,
            error_code_t::GRAPH_NOT_SUPPORTED,
            "sdpa fp8 backward operation is not supported on cudnn 9.10.0 due to known bugs. "
            "Please consider upgrading to 9.10.2 or newer.");

        CHECK_CUDNN_FRONTEND_ERROR(context.populate_sm_version_from_device());
        int32_t const sm_version = context.get_sm_version();
        int32_t const prop_major = sm_version / 10;
        RETURN_CUDNN_FRONTEND_ERROR_IF(
            prop_major < 9,
            error_code_t::GRAPH_NOT_SUPPORTED,
            "sdpa fp8 forward operation is only supported on Hopper architecture and newer. Please "
            "consider using a newer architecture.");

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
        int64_t s_kv = attributes.inputs.at(input_names::K)->get_dim()[2];
        int64_t h_q  = attributes.inputs.at(input_names::Q)->get_dim()[1];
        int64_t h_k  = attributes.inputs.at(input_names::K)->get_dim()[1];
        int64_t h_v  = attributes.inputs.at(input_names::V)->get_dim()[1];
        int64_t d_qk = attributes.inputs.at(input_names::Q)->get_dim()[3];
        int64_t d_v  = attributes.inputs.at(input_names::V)->get_dim()[3];

        auto const& dq_tensor = attributes.outputs.at(output_names::dQ);
        auto const& dq_data_type = dq_tensor->get_data_type();
        auto const& dk_tensor = attributes.outputs.at(output_names::dK);
        auto const& dk_data_type = dk_tensor->get_data_type();
        auto const& dv_tensor = attributes.outputs.at(output_names::dV);
        auto const& dv_data_type = dv_tensor->get_data_type();

        auto const& bias_mask = attributes.inputs.find(input_names::Bias);
        bool const is_bias    = (bias_mask != attributes.inputs.end() && bias_mask->second != nullptr);

        auto const& dropout_mask     = attributes.inputs.find(input_names::Dropout_mask);
        bool const is_dropout_custom = (dropout_mask != attributes.inputs.end()) && (dropout_mask->second != nullptr);
        bool const is_dropout        = attributes.dropout_probability.has_value();
        bool const is_ragged         =
            attributes.inputs.at(input_names::Q)->get_ragged_offset() ||
            attributes.inputs.at(input_names::K)->get_ragged_offset() ||
            attributes.inputs.at(input_names::V)->get_ragged_offset() ||
            attributes.inputs.at(input_names::O)->get_ragged_offset() ||
            attributes.inputs.at(input_names::Stats)->get_ragged_offset() ||
            attributes.inputs.at(input_names::dO)->get_ragged_offset() ||
            attributes.outputs.at(output_names::dQ)->get_ragged_offset() ||
            attributes.outputs.at(output_names::dK)->get_ragged_offset() ||
            attributes.outputs.at(output_names::dV)->get_ragged_offset();

        // validation TODO:
        //    - validate stats has valid dims

        RETURN_CUDNN_FRONTEND_ERROR_IF((prop_major == 9) && is_ragged,
                                       error_code_t::GRAPH_NOT_SUPPORTED,
                                       "sdpa fp8 backward with THD is not supported on Hopper architecture.");

        // validate basic dimension requirements
        if(prop_major >= 10) {
            RETURN_CUDNN_FRONTEND_ERROR_IF(((d_qk > 128) || (d_qk % 16 != 0)) && !(d_qk == 192 && d_v == 128),
                                            error_code_t::GRAPH_NOT_SUPPORTED,
                                            "hidden_dim d_qk shoud be less than or equal to 128 and hidden_dim d_qk should be multiple of 16 unless d_qk == 192 and d_v == 128");

            RETURN_CUDNN_FRONTEND_ERROR_IF(((d_v > 128) || (d_v % 16 != 0)),
                                            error_code_t::GRAPH_NOT_SUPPORTED,
                                            "hidden_dim d_v shoud be less than or equal to 128 and hidden_dim d_v should be multiple of 16");
        }
        else {
            RETURN_CUDNN_FRONTEND_ERROR_IF((d_qk != 128) || (d_qk % 16 != 0) || (d_v != 128) || (d_v % 16 != 0),
                                error_code_t::GRAPH_NOT_SUPPORTED,
                                "hidden_dim shoud be equal to 128 and hidden_dim should be multiple of 16");
        }
        RETURN_CUDNN_FRONTEND_ERROR_IF((h_q % h_k != 0) || (h_q % h_v != 0),
                                       error_code_t::GRAPH_NOT_SUPPORTED,
                                       "For group-query attention, number of heads for key and query must be a factor of number of heads for query");

        // validate options for attn_scale
        auto const& attn_scale    = attributes.inputs.find(input_names::Attn_scale);
        bool const has_attn_scale = (attn_scale != attributes.inputs.end()) && (attn_scale->second != nullptr);
        RETURN_CUDNN_FRONTEND_ERROR_IF(has_attn_scale && attributes.attn_scale_value.has_value(),
                                       error_code_t::ATTRIBUTE_NOT_SET,
                                       "attn_scale with tensor and value cannot be set at the same time.");

        // validate options for bias mask
        RETURN_CUDNN_FRONTEND_ERROR_IF(is_bias && (bias_mask->second->get_data_type() == DataType_t::BOOLEAN),
                                       error_code_t::GRAPH_NOT_SUPPORTED,
                                       "Bias mask data type cannot be boolean");

        // validate options for padding mask
        auto const& seq_len_q     = attributes.inputs.find(input_names::SEQ_LEN_Q);
        bool const has_seq_len_q  = (seq_len_q != attributes.inputs.end()) && (seq_len_q->second != nullptr);
        auto const& seq_len_kv    = attributes.inputs.find(input_names::SEQ_LEN_KV);
        bool const has_seq_len_kv = (seq_len_kv != attributes.inputs.end()) && (seq_len_kv->second != nullptr);
        RETURN_CUDNN_FRONTEND_ERROR_IF(attributes.padding_mask && (!has_seq_len_q || !has_seq_len_kv),
                                       error_code_t::ATTRIBUTE_NOT_SET,
                                       "Padding mask requires seq_len_q and seq_len_kv to be set.");
        RETURN_CUDNN_FRONTEND_ERROR_IF((!attributes.padding_mask) && (has_seq_len_q || has_seq_len_kv),
                                       error_code_t::ATTRIBUTE_NOT_SET,
                                       "seq_len_q and seq_len_kv needs to be set only if padding mask is enabled.");

        // validate options for dropout mask
        RETURN_CUDNN_FRONTEND_ERROR_IF(
            attributes.dropout_probability.has_value() && is_dropout_custom,
            error_code_t::ATTRIBUTE_NOT_SET,
            "Using both, custom dropout mask and internal-mask generation using dropout probability, is ill-formed.");

        RETURN_CUDNN_FRONTEND_ERROR_IF(
            attributes.dropout_probability.has_value() && attributes.dropout_probability.value() == 1.0,
            error_code_t::ATTRIBUTE_NOT_SET,
            "Dropout probability cannot be 1 as corresponding scale wont be well formed.");

        // Validate options for causal_mask_bottom_right
        RETURN_CUDNN_FRONTEND_ERROR_IF(attributes.has_causal_mask_bottom_right() && detail::get_backend_version() < 90700,
            error_code_t::GRAPH_NOT_SUPPORTED,
            "For cuDNN version below 9.7.0, bottom right causal masking is not supported.");

        RETURN_CUDNN_FRONTEND_ERROR_IF(attributes.has_causal_mask_bottom_right() && prop_major < 10, 
            error_code_t::GRAPH_NOT_SUPPORTED,
            "sdpa fp8 forward operation is only supported on Blackwell architecture and newer. Please "
            "consider using a newer architecture.");

        RETURN_CUDNN_FRONTEND_ERROR_IF(attributes.has_causal_mask_bottom_right() && s_q > s_kv,
            error_code_t::GRAPH_NOT_SUPPORTED,
            "Bottom right causal mask does not support s_q > s_kv. Please virtually slice the Q tensor and pass it as s_q == s_kv");

        RETURN_CUDNN_FRONTEND_ERROR_IF(attributes.has_causal_mask_bottom_right() && (is_bias || is_dropout),
            error_code_t::GRAPH_NOT_SUPPORTED,
            "Bottom right causal mask is only supported with is_bias=False, is_dropout=False.");

        RETURN_CUDNN_FRONTEND_ERROR_IF(attributes.has_causal_mask_bottom_right() && ((s_q % 64 != 0) || (s_kv % 64 != 0)),
                error_code_t::GRAPH_NOT_SUPPORTED,
                "Bottom right causal mask is only supported with s_q multiple of 64, and s_kv multiple of 64");

        RETURN_CUDNN_FRONTEND_ERROR_IF(
            attributes.left_bound.has_value() && (!attributes.padding_mask) && s_q > s_kv,
            error_code_t::GRAPH_NOT_SUPPORTED,
            "Sliding window attention is only supported with max_s_q <= max_s_kv.");

        // validate that datatype is set for the graph
        RETURN_CUDNN_FRONTEND_ERROR_IF(context.get_intermediate_data_type() == DataType_t::NOT_SET,
                                       error_code_t::ATTRIBUTE_NOT_SET,
                                       "Intermediate tensor data type needs to be set as internal tensors require it.");

        // Auto-route to deterministic algorithm for cases where non-deterministic is not supported on Blackwell:
        // 1. MXFP8 backward - requires deterministic (STAGES=2)
        // 2. FP8 backward with d_qk <= 192 and d_v <= 128 - requires deterministic
        bool const is_mxfp8 = is_mxfp8_scaling();
        bool const is_supported_dims = (d_qk <= 192) && (d_v <= 128);
        if ((prop_major == 10) && !attributes.is_deterministic_algorithm) {
            if (is_mxfp8) {
                getLogger() << "[cudnn_frontend] INFO: MXFP8 SDPA backward detected on Blackwell - "
                            << "auto-routing to deterministic algorithm (non-deterministic not supported for MXFP8)"
                            << std::endl;
                attributes.is_deterministic_algorithm = true;
            } else if (is_supported_dims) {
                getLogger() << "[cudnn_frontend] INFO: FP8 SDPA backward with d_qk=" << d_qk << ", d_v=" << d_v
                            << " detected on Blackwell - auto-routing to deterministic algorithm"
                            << std::endl;
                attributes.is_deterministic_algorithm = true;
            }
        }

        // validate options for deterministic algorithm
        if (attributes.is_deterministic_algorithm && (prop_major == 10)) {
            RETURN_CUDNN_FRONTEND_ERROR_IF((detail::get_backend_version() < 91900),
                                           error_code_t::GRAPH_NOT_SUPPORTED,
                                           "FP8 deterministic algorithm (required for MXFP8 and d_qk=192/d_v=128) is not supported on Blackwell with cuDNN version below 9.19.0");

            RETURN_CUDNN_FRONTEND_ERROR_IF(is_dropout,
                                           error_code_t::GRAPH_NOT_SUPPORTED,
                                           "FP8 deterministic algorithm (required for MXFP8 and d_qk=192/d_v=128) is not supported on Blackwell when dropout is enabled");

            is_deterministic_algorithm_supported_on_blackwell = true;
        }

        // if output data type is half or bfloat16 for any of dq, dk, dv, and version is below 9.13 or is not blackwell, return NOT_SUPPORTED
        RETURN_CUDNN_FRONTEND_ERROR_IF(
            (dq_data_type == DataType_t::HALF || dq_data_type == DataType_t::BFLOAT16 ||
             dk_data_type == DataType_t::HALF || dk_data_type == DataType_t::BFLOAT16 ||
             dv_data_type == DataType_t::HALF || dv_data_type == DataType_t::BFLOAT16) &&
                (detail::get_backend_version() < 91300 || prop_major < 10),
            error_code_t::GRAPH_NOT_SUPPORTED,
            "sdpa fp8 backward with HALF/BFLOAT16 output is only supported on Blackwell architecture "
            "with cuDNN version 9.13.0 and newer.");

        // Validate MXFP8 scale factors if present
        if (is_mxfp8_scaling()) {
            // MXFP8 requires cuDNN 9.21.0 or later
            RETURN_CUDNN_FRONTEND_ERROR_IF(detail::get_backend_version() < 92100,
                                           error_code_t::GRAPH_NOT_SUPPORTED,
                                           "MXFP8 SDPA requires cuDNN 9.21.0 or later");
            int64_t const b = attributes.inputs.at(input_names::Q)->get_dim()[0];

            // MXFP8 block size is fixed at 32
            constexpr int64_t block_size = 32;
            int64_t const d_qk_scale     = (d_qk + block_size - 1) / block_size;
            int64_t const d_v_scale      = (d_v + block_size - 1) / block_size;
            int64_t const s_q_scale      = (s_q + block_size - 1) / block_size;
            int64_t const s_kv_scale     = (s_kv + block_size - 1) / block_size;

            // Validate Descale_Q
            auto const& descale_q = attributes.inputs.at(input_names::Descale_Q);
            RETURN_CUDNN_FRONTEND_ERROR_IF(descale_q->get_data_type() != DataType_t::FP8_E8M0,
                                           error_code_t::ATTRIBUTE_NOT_SET,
                                           "MXFP8 SDPA requires Descale_Q to have FP8_E8M0 data type");
            RETURN_CUDNN_FRONTEND_ERROR_IF(descale_q->get_reordering_type() != TensorReordering_t::F8_128x4,
                                           error_code_t::ATTRIBUTE_NOT_SET,
                                           "MXFP8 SDPA requires Descale_Q to have F8_128x4 reordering");
            
            // Validate Descale_Q_T
            auto const& descale_q_T = attributes.inputs.at(input_names::Descale_Q_T);
            RETURN_CUDNN_FRONTEND_ERROR_IF(descale_q_T->get_data_type() != DataType_t::FP8_E8M0,
                                           error_code_t::ATTRIBUTE_NOT_SET,
                                           "MXFP8 SDPA requires Descale_Q_T to have FP8_E8M0 data type");
            RETURN_CUDNN_FRONTEND_ERROR_IF(descale_q_T->get_reordering_type() != TensorReordering_t::F8_128x4,
                                           error_code_t::ATTRIBUTE_NOT_SET,
                                           "MXFP8 SDPA requires Descale_Q_T to have F8_128x4 reordering");

            // Validate Descale_K
            auto const& descale_k = attributes.inputs.at(input_names::Descale_K);
            RETURN_CUDNN_FRONTEND_ERROR_IF(descale_k->get_data_type() != DataType_t::FP8_E8M0,
                                           error_code_t::ATTRIBUTE_NOT_SET,
                                           "MXFP8 SDPA requires Descale_K to have FP8_E8M0 data type");
            RETURN_CUDNN_FRONTEND_ERROR_IF(descale_k->get_reordering_type() != TensorReordering_t::F8_128x4,
                                           error_code_t::ATTRIBUTE_NOT_SET,
                                           "MXFP8 SDPA requires Descale_K to have F8_128x4 reordering");
            
            // Validate Descale_K_T
            auto const& descale_k_T = attributes.inputs.at(input_names::Descale_K_T);
            RETURN_CUDNN_FRONTEND_ERROR_IF(descale_k_T->get_data_type() != DataType_t::FP8_E8M0,
                                           error_code_t::ATTRIBUTE_NOT_SET,
                                           "MXFP8 SDPA requires Descale_K_T to have FP8_E8M0 data type");
            RETURN_CUDNN_FRONTEND_ERROR_IF(descale_k_T->get_reordering_type() != TensorReordering_t::F8_128x4,
                                           error_code_t::ATTRIBUTE_NOT_SET,
                                           "MXFP8 SDPA requires Descale_K_T to have F8_128x4 reordering");

            // Validate Descale_V
            auto const& descale_v = attributes.inputs.at(input_names::Descale_V);
            RETURN_CUDNN_FRONTEND_ERROR_IF(descale_v->get_data_type() != DataType_t::FP8_E8M0,
                                           error_code_t::ATTRIBUTE_NOT_SET,
                                           "MXFP8 SDPA requires Descale_V to have FP8_E8M0 data type");
            RETURN_CUDNN_FRONTEND_ERROR_IF(descale_v->get_reordering_type() != TensorReordering_t::F8_128x4,
                                           error_code_t::ATTRIBUTE_NOT_SET,
                                           "MXFP8 SDPA requires Descale_V to have F8_128x4 reordering");
            
            // Validate Descale_dO
            auto const& descale_do = attributes.inputs.at(input_names::Descale_dO);
            RETURN_CUDNN_FRONTEND_ERROR_IF(descale_do->get_data_type() != DataType_t::FP8_E8M0,
                                           error_code_t::ATTRIBUTE_NOT_SET,
                                           "MXFP8 SDPA requires Descale_dO to have FP8_E8M0 data type");
            RETURN_CUDNN_FRONTEND_ERROR_IF(descale_do->get_reordering_type() != TensorReordering_t::F8_128x4,
                                           error_code_t::ATTRIBUTE_NOT_SET,
                                           "MXFP8 SDPA requires Descale_dO to have F8_128x4 reordering");
            
            // Validate Descale_dO_T
            auto const& descale_do_T = attributes.inputs.at(input_names::Descale_dO_T);
            RETURN_CUDNN_FRONTEND_ERROR_IF(descale_do_T->get_data_type() != DataType_t::FP8_E8M0,
                                           error_code_t::ATTRIBUTE_NOT_SET,
                                           "MXFP8 SDPA requires Descale_dO_T to have FP8_E8M0 data type");
            RETURN_CUDNN_FRONTEND_ERROR_IF(descale_do_T->get_reordering_type() != TensorReordering_t::F8_128x4,
                                           error_code_t::ATTRIBUTE_NOT_SET,
                                           "MXFP8 SDPA requires Descale_dO_T to have F8_128x4 reordering");

            // Validate dimension consistency for SF_Q: [b, h_q, s_q_padded, d_scale_padded]
            auto const& sf_q_dim = descale_q->get_dim();
            RETURN_CUDNN_FRONTEND_ERROR_IF(sf_q_dim[0] != b || sf_q_dim[1] != h_q,
                                           error_code_t::ATTRIBUTE_NOT_SET,
                                           "MXFP8 SDPA: Descale_Q batch/head dimensions must match Q");
            RETURN_CUDNN_FRONTEND_ERROR_IF(
                sf_q_dim[3] < d_qk_scale,
                error_code_t::ATTRIBUTE_NOT_SET,
                "MXFP8 SDPA: Descale_Q d_scale dimension too small (expected >= " + std::to_string(d_qk_scale) + ")");
            
            // Validate dimension consistency for SF_Q_T: [b, h_q, s_q_scale_padded, d_padded]
            auto const& sf_q_T_dim = descale_q_T->get_dim();
            RETURN_CUDNN_FRONTEND_ERROR_IF(sf_q_T_dim[0] != b || sf_q_T_dim[1] != h_q,
                                           error_code_t::ATTRIBUTE_NOT_SET,
                                           "MXFP8 SDPA: Descale_Q_T batch/head dimensions must match Q");
            RETURN_CUDNN_FRONTEND_ERROR_IF(
                sf_q_T_dim[2] < s_q_scale,
                error_code_t::ATTRIBUTE_NOT_SET,
                "MXFP8 SDPA: Descale_Q_T s_scale dimension too small (expected >= " + std::to_string(s_q_scale) + ")");

            // Validate dimension consistency for SF_K: [b, h_k, s_kv_padded, d_qk_scale_padded]
            auto const& sf_k_dim = descale_k->get_dim();
            RETURN_CUDNN_FRONTEND_ERROR_IF(sf_k_dim[0] != b || sf_k_dim[1] != h_k,
                                           error_code_t::ATTRIBUTE_NOT_SET,
                                           "MXFP8 SDPA: Descale_K batch/head dimensions must match K");
            RETURN_CUDNN_FRONTEND_ERROR_IF(
                sf_k_dim[3] < d_qk_scale,
                error_code_t::ATTRIBUTE_NOT_SET,
                "MXFP8 SDPA: Descale_K d_scale dimension too small (expected >= " + std::to_string(d_qk_scale) + ")");
            
            // Validate dimension consistency for SF_K_T: [b, h_k, s_kv_scale_padded, d_qk_padded]
            auto const& sf_k_T_dim = descale_k_T->get_dim();
            RETURN_CUDNN_FRONTEND_ERROR_IF(sf_k_T_dim[0] != b || sf_k_T_dim[1] != h_k,
                                           error_code_t::ATTRIBUTE_NOT_SET,
                                           "MXFP8 SDPA: Descale_K_T batch/head dimensions must match K");
            RETURN_CUDNN_FRONTEND_ERROR_IF(
                sf_k_T_dim[2] < s_kv_scale,
                error_code_t::ATTRIBUTE_NOT_SET,
                "MXFP8 SDPA: Descale_K_T s_scale dimension too small (expected >= " + std::to_string(s_kv_scale) + ")");

            // Validate dimension consistency for SF_V: [b, h_v, s_kv_scale_padded, d_v_padded]
            auto const& sf_v_dim = descale_v->get_dim();
            RETURN_CUDNN_FRONTEND_ERROR_IF(sf_v_dim[0] != b || sf_v_dim[1] != h_v,
                                           error_code_t::ATTRIBUTE_NOT_SET,
                                           "MXFP8 SDPA: Descale_V batch/head dimensions must match V");
            RETURN_CUDNN_FRONTEND_ERROR_IF(
                sf_v_dim[2] < s_kv_scale,
                error_code_t::ATTRIBUTE_NOT_SET,
                "MXFP8 SDPA: Descale_V s_scale dimension too small (expected >= " + std::to_string(s_kv_scale) + ")");
            
            // Validate dimension consistency for SF_dO: [b, h_q, s_q_padded, d_v_scale_padded]
            auto const& sf_do_dim = descale_do->get_dim();
            RETURN_CUDNN_FRONTEND_ERROR_IF(sf_do_dim[0] != b || sf_do_dim[1] != h_q,
                                           error_code_t::ATTRIBUTE_NOT_SET,
                                           "MXFP8 SDPA: Descale_dO batch/head dimensions must match dO");
            RETURN_CUDNN_FRONTEND_ERROR_IF(
                sf_do_dim[3] < d_v_scale,
                error_code_t::ATTRIBUTE_NOT_SET,
                "MXFP8 SDPA: Descale_dO d_scale dimension too small (expected >= " + std::to_string(d_v_scale) + ")");
            
            // Validate dimension consistency for SF_dO_T: [b, h_q, s_q_scale_padded, d_scale_padded]
            auto const& sf_do_T_dim = descale_do_T->get_dim();
            RETURN_CUDNN_FRONTEND_ERROR_IF(sf_do_T_dim[0] != b || sf_do_T_dim[1] != h_q,
                                           error_code_t::ATTRIBUTE_NOT_SET,
                                           "MXFP8 SDPA: Descale_dO_T batch/head dimensions must match dO");
            RETURN_CUDNN_FRONTEND_ERROR_IF(
                sf_do_T_dim[2] < s_q_scale,
                error_code_t::ATTRIBUTE_NOT_SET,
                "MXFP8 SDPA: Descale_dO_T s_scale dimension too small (expected >= " + std::to_string(s_q_scale) + ")");

            // Log MXFP8 configuration for debugging
            getLogger() << "[cudnn_frontend] INFO: MXFP8 SDPA configuration validated:" << std::endl
                        << "  SF_Q dims: [" << sf_q_dim[0] << ", " << sf_q_dim[1] << ", " << sf_q_dim[2] << ", "
                        << sf_q_dim[3] << "]" << std::endl
                        << "  SF_Q_T dims: [" << sf_q_T_dim[0] << ", " << sf_q_T_dim[1] << ", " << sf_q_T_dim[2] << ", "
                        << sf_q_T_dim[3] << "]" << std::endl
                        << "  SF_K dims: [" << sf_k_dim[0] << ", " << sf_k_dim[1] << ", " << sf_k_dim[2] << ", "
                        << sf_k_dim[3] << "]" << std::endl
                        << "  SF_K_T dims: [" << sf_k_T_dim[0] << ", " << sf_k_T_dim[1] << ", " << sf_k_T_dim[2] << ", "
                        << sf_k_T_dim[3] << "]" << std::endl
                        << "  SF_V dims: [" << sf_v_dim[0] << ", " << sf_v_dim[1] << ", " << sf_v_dim[2] << ", "
                        << sf_v_dim[3] << "]" << std::endl
                        << "  SF_dO dims: [" << sf_do_dim[0] << ", " << sf_do_dim[1] << ", " << sf_do_dim[2] << ", "
                        << sf_do_dim[3] << "]" << std::endl
                        << "  SF_dO_T dims: [" << sf_do_T_dim[0] << ", " << sf_do_T_dim[1] << ", " << sf_do_T_dim[2] << ", "
                        << sf_do_T_dim[3] << "]" << std::endl
                        << "  Expected d_qk_scale: " << d_qk_scale << ", d_v_scale: " << d_v_scale
                        << ", s_q_scale: " << s_q_scale << ", s_kv_scale: " << s_kv_scale << std::endl;
        }

        // validate options for sink token
        auto const& sink_token     = attributes.inputs.find(input_names::SINK_TOKEN);
        bool const has_sink_token  = (sink_token != attributes.inputs.end()) && (sink_token->second != nullptr);
        auto const& dsink_token    = attributes.outputs.find(output_names::DSINK_TOKEN);
        bool const has_dsink_token = (dsink_token != attributes.outputs.end()) && (dsink_token->second != nullptr);
        RETURN_CUDNN_FRONTEND_ERROR_IF(has_dsink_token && !has_sink_token,
                                       error_code_t::ATTRIBUTE_NOT_SET,
                                       "If dSink_token output is requested, sink_token input must also be set.");

        return {error_code_t::OK, ""};
    }

    error_t
    infer_properties_node() override final {
        return {error_code_t::OK, ""};
    }

    error_t
    expand_node() override final {
        CUDNN_FE_LOG_LABEL_ENDL("INFO:     Inferrencing properties for Scaled_dot_product_flash_attention node  "
                                << attributes.name);

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

        bool const use_mxfp8 = is_mxfp8_scaling();
        std::shared_ptr<Tensor_attributes> eff_Q, eff_K, eff_V, eff_dO;

        // cuDNN frontend API attention requires Q, K, V where
        // Q = {b, h_q, s_q, d_qk}
        // K = {b, h_k, s_kv, d_qk}
        // V = {b, h_v, s_kv, d_v}
        // but cuDNN backend API attention requires Q, KT, VT
        // Q = {b, h_q, s_q, d_qk}
        // KT = {b, h_k, d_qk, s_kv}
        // VT = {b, h_v, d_v, s_kv}
        // So the code below maps the K->KT and V->VT
        if (use_mxfp8) {
            // MXFP8 2-kernel bprop: Each tensor is dequantized separately for each matmul it participates in.
            // Q, K, dO each need two dequantizations with different scale factors.
            // O and dO are used directly (as f16) for the dO*O computation without dequantization.

            // K for BMM1 (Q@K.T): Transpose K and SF_K, then dequantize
            std::vector<int64_t> kt_dim    = attributes.inputs[input_names::K]->get_dim();
            std::vector<int64_t> kt_stride = attributes.inputs[input_names::K]->get_stride();
            std::swap(kt_dim[2], kt_dim[3]);
            std::swap(kt_stride[2], kt_stride[3]);
            attributes.inputs[input_names::K]->set_dim(kt_dim);
            attributes.inputs[input_names::K]->set_stride(kt_stride);

            auto& sf_k                       = attributes.inputs[input_names::Descale_K];
            std::vector<int64_t> sf_k_dim    = sf_k->get_dim();
            std::vector<int64_t> sf_k_stride = sf_k->get_stride();
            std::swap(sf_k_dim[2], sf_k_dim[3]);
            std::swap(sf_k_stride[2], sf_k_stride[3]);
            sf_k->set_dim(sf_k_dim);
            sf_k->set_stride(sf_k_stride);

            auto dequant_k_dqk_attrs = Block_scale_dequantize_attributes()
                                           .set_name("DQ_K_dqk")
                                           .set_block_size({1, 32});
            auto k_dqk_dequant = std::make_shared<Tensor_attributes>();
            k_dqk_dequant->set_is_virtual(true);
            k_dqk_dequant->set_dim(kt_dim);
            k_dqk_dequant->set_stride(kt_stride);
            block_scale_dequantize(attributes.inputs[input_names::K], sf_k, dequant_k_dqk_attrs, k_dqk_dequant);
            eff_K = k_dqk_dequant;

            // V for dO@V.T: Transpose V and SF_V, then dequantize
            std::vector<int64_t> vt_dim    = attributes.inputs[input_names::V]->get_dim();
            std::vector<int64_t> vt_stride = attributes.inputs[input_names::V]->get_stride();
            std::swap(vt_dim[2], vt_dim[3]);
            std::swap(vt_stride[2], vt_stride[3]);
            attributes.inputs[input_names::V]->set_dim(vt_dim);
            attributes.inputs[input_names::V]->set_stride(vt_stride);

            auto& sf_v                       = attributes.inputs[input_names::Descale_V];
            std::vector<int64_t> sf_v_dim    = sf_v->get_dim();
            std::vector<int64_t> sf_v_stride = sf_v->get_stride();
            std::swap(sf_v_dim[2], sf_v_dim[3]);
            std::swap(sf_v_stride[2], sf_v_stride[3]);
            sf_v->set_dim(sf_v_dim);
            sf_v->set_stride(sf_v_stride);

            auto dequant_v_dv_attrs = Block_scale_dequantize_attributes()
                                          .set_name("DQ_V_dv")
                                          .set_block_size({1, 32});
            auto v_dv_dequant = std::make_shared<Tensor_attributes>();
            v_dv_dequant->set_is_virtual(true);
            v_dv_dequant->set_dim(vt_dim);
            v_dv_dequant->set_stride(vt_stride);
            block_scale_dequantize(attributes.inputs[input_names::V], sf_v, dequant_v_dv_attrs, v_dv_dequant);
            eff_V = v_dv_dequant;

            // Q for BMM1 (Q@K.T): Dequantize Q with Descale_Q
            auto dequant_q_dqk_attrs = Block_scale_dequantize_attributes()
                                           .set_name("DQ_Q_dqk")
                                           .set_block_size({1, 32});
            auto q_dqk_dequant = std::make_shared<Tensor_attributes>();
            q_dqk_dequant->set_is_virtual(true);
            q_dqk_dequant->set_dim(attributes.inputs[input_names::Q]->get_dim());
            q_dqk_dequant->set_stride(attributes.inputs[input_names::Q]->get_stride());
            block_scale_dequantize(attributes.inputs[input_names::Q],
                                   attributes.inputs[input_names::Descale_Q],
                                   dequant_q_dqk_attrs,
                                   q_dqk_dequant);
            eff_Q = q_dqk_dequant;

            // dO for dO@V.T: Dequantize dO with Descale_dO
            auto dequant_dO_dv_attrs = Block_scale_dequantize_attributes()
                                           .set_name("DQ_dO_dv")
                                           .set_block_size({1, 32});
            auto dO_dv_dequant = std::make_shared<Tensor_attributes>();
            dO_dv_dequant->set_is_virtual(true);
            dO_dv_dequant->set_dim(attributes.inputs[input_names::dO]->get_dim());
            dO_dv_dequant->set_stride(attributes.inputs[input_names::dO]->get_stride());
            block_scale_dequantize(attributes.inputs[input_names::dO],
                                   attributes.inputs[input_names::Descale_dO],
                                   dequant_dO_dv_attrs,
                                   dO_dv_dequant);
            eff_dO = dO_dv_dequant;

            // Note: O and dO for the dO*O computation are used as raw f16 inputs (no MXFP8 dequant).
            // The separate dequantizations for dO_seq, K_seq, Q_seq are done inline at their usage sites.
        } else {
            // Non-MXFP8
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

            eff_Q  = attributes.inputs[input_names::Q];
            eff_K  = attributes.inputs[input_names::K];
            eff_V  = attributes.inputs[input_names::V];
            eff_dO = attributes.inputs[input_names::dO];
        }

        std::shared_ptr<Tensor_attributes> rng_output;

        auto mul_attributes = Pointwise_attributes().set_mode(PointwiseMode_t::MUL);

        // if dropout_prob is used, then the node passes scale and scale inverse
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

        // --------------RNG node--------------------

        if (is_dropout_prob) {
                rng_output = rng(attributes.inputs[input_names::Seed],
                                 attributes.inputs[input_names::Offset],
                                 Rng_attributes()
                                     .set_name("rng")
                                     .set_distribution(RngDistribution_t::BERNOULLI)
                                     .set_bernoulli_probability(1.0f - attributes.dropout_probability.value()));
                rng_output->set_dim({b, h_q, s_q, s_kv}).set_stride({h_q * s_q * s_kv, s_q * s_kv, s_kv, 1});
        } else if (is_dropout_mask) {
            rng_output = attributes.inputs[input_names::Dropout_mask];
        }

        //// dO * O
        // For MXFP8: dO_f16 and O (f16) are used directly (no block dequantization needed)
        // For non-MXFP8 (17-arg API): dO and O (FP8) are used (per-tensor descale applied after reduction)
        mul_attributes.set_name("mul_dO_O");
        auto last_output = use_mxfp8
                              ? pointwise(attributes.inputs[input_names::dO_f16], attributes.inputs[input_names::O], mul_attributes)
                              : pointwise(attributes.inputs[input_names::dO], attributes.inputs[input_names::O], mul_attributes);

        // reduce(dO)
        last_output =
            reduction(last_output, Reduction_attributes().set_name("reduce_dO").set_mode(ReductionMode_t::ADD));
        last_output->set_dim({b, h_q, s_q, 1}).set_stride({h_q * s_q, s_q, 1, 1});

        // Descale dO (only for non-MXFP8 per-tensor scaling)
        if (!use_mxfp8) {
            mul_attributes.set_name("descale_dO");
            last_output = pointwise(last_output, attributes.inputs.at(input_names::Descale_dO), mul_attributes);
            last_output->set_dim({b, h_q, s_q, 1}).set_stride({h_q * s_q, s_q, 1, 1});
        }

        // Descale O (only for non-MXFP8 per-tensor scaling)
        if (!use_mxfp8) {
            mul_attributes.set_name("descale_O");
            last_output = pointwise(last_output, attributes.inputs.at(input_names::Descale_O), mul_attributes);
        }

        // dSink_token computation: dSink = sum(exp(sink - stats) * D) over batch and sequence
        // Note: The backend kernel applies the negative sign internally.
        auto const& sink_token_it  = attributes.inputs.find(input_names::SINK_TOKEN);
        auto const& dsink_token_it = attributes.outputs.find(output_names::DSINK_TOKEN);
        if (dsink_token_it != attributes.outputs.end() && dsink_token_it->second != nullptr &&
            sink_token_it != attributes.inputs.end() && sink_token_it->second != nullptr) {
            // sub_sink = sink - stats
            auto sub_sink = pointwise(sink_token_it->second,
                                      attributes.inputs[input_names::Stats],
                                      Pointwise_attributes().set_name("sub_sink_stats").set_mode(PointwiseMode_t::SUB));
            sub_sink->set_dim({b, h_q, s_q, 1}).set_stride({h_q * s_q, s_q, 1, 1});

            // exp_sink = exp(sub_sink)
            auto exp_sink =
                pointwise(sub_sink, Pointwise_attributes().set_name("exp_sink").set_mode(PointwiseMode_t::EXP));
            exp_sink->set_dim({b, h_q, s_q, 1}).set_stride({h_q * s_q, s_q, 1, 1});

            // per_token_grad = exp_sink * last_output (D)
            auto per_token_grad = pointwise(
                exp_sink, last_output, Pointwise_attributes().set_name("mul_exp_sink_D").set_mode(PointwiseMode_t::MUL));
            per_token_grad->set_dim({b, h_q, s_q, 1}).set_stride({h_q * s_q, s_q, 1, 1});

            // dSink = reduce(per_token_grad) over batch and sequence dimensions
            reduction(per_token_grad,
                      Reduction_attributes().set_name("reduce_dSink").set_mode(ReductionMode_t::ADD),
                      dsink_token_it->second);
        }

        // softmax_sum = last_output * dropout_scale
        // Note: When sink_token is enabled and dropout is NOT present, we still need a MUL operation
        // in the softmax path so the backend pattern matcher (fortAttentionBackwardRuntimeFusionEngine)
        // can correctly distinguish between the dSink chain (MUL→REDUCE) and the softmax path (MUL→SUB).
        // Without this, the fallback at lines 3786-3796 incorrectly identifies dSink MUL as the softmax path.
        bool const has_sink_token = (sink_token_it != attributes.inputs.end() && sink_token_it->second != nullptr);
        if (attributes.inputs[input_names::Dropout_scale_inv]) {
            last_output = pointwise(last_output,
                                    attributes.inputs[input_names::Dropout_scale_inv],
                                    Pointwise_attributes().set_name("scale_dropout_inv").set_mode(PointwiseMode_t::MUL));
        } else if (has_sink_token) {
            // Add identity MUL (multiply by 1) to create MUL→SUB pattern for pattern matcher
            auto one_tensor = std::make_shared<Tensor_attributes>(1.0f);
            last_output     = pointwise(last_output,
                                    one_tensor,
                                    Pointwise_attributes().set_name("identity_for_sink").set_mode(PointwiseMode_t::MUL));
            last_output->set_dim({b, h_q, s_q, 1}).set_stride({h_q * s_q, s_q, 1, 1});
        }
        auto softmax_sum = last_output;

        //// Q * K
        auto bmm_Q_K_attributes = Matmul_attributes().set_name("bmm_Q_K")
                                 .set_m_override(attributes.inputs[input_names::SEQ_LEN_Q])
                                 .set_n_override(attributes.inputs[input_names::SEQ_LEN_KV]);
        auto last_dV = matmul(eff_Q, eff_K, bmm_Q_K_attributes);

        //// Optional Attn scale
        // In case user provided a scalar value, do a fused scalar.
        if (attributes.attn_scale_value.has_value()) {
            attributes.inputs[input_names::Attn_scale] =
                std::make_shared<Tensor_attributes>(attributes.attn_scale_value.value());
        }

        // If attn scale present, add a pointwise mul node
        if (auto attn_scale_it = attributes.inputs.find(input_names::Attn_scale); attn_scale_it != attributes.inputs.end()) {
            mul_attributes.set_name("attn_scale");
            last_dV = pointwise(last_dV, attn_scale_it->second, mul_attributes);
        }

        // Descale Q (only for non-MXFP8 per-tensor scaling)
        if (!use_mxfp8) {
            mul_attributes.set_name("descale_q");
            last_dV = pointwise(last_dV, attributes.inputs.at(input_names::Descale_Q), mul_attributes);
        }

        // Descale K (only for non-MXFP8 per-tensor scaling)
        if (!use_mxfp8) {
            mul_attributes.set_name("descale_k");
            last_dV = pointwise(last_dV, attributes.inputs.at(input_names::Descale_K), mul_attributes);
        }

        // (optional) last_dV = last_dV + bias
        if (auto bias_it = attributes.inputs.find(input_names::Bias); bias_it != attributes.inputs.end()) {
            last_dV = pointwise(last_dV,
                                    bias_it->second,
                                    Pointwise_attributes().set_name("add_bias").set_mode(PointwiseMode_t::ADD));
        }

        // (optional) Apply padding mask
        if (attributes.padding_mask) {
            auto row_idx_output = pointwise(last_dV,
                                            Pointwise_attributes()
                                                .set_name("gen_row_idx_padding")
                                                .set_mode(PointwiseMode_t::GEN_INDEX)
                                                .set_axis(2)
                                                .set_compute_data_type(DataType_t::INT32));
            row_idx_output->set_data_type(DataType_t::INT32);

            auto col_idx_output = pointwise(last_dV,
                                            Pointwise_attributes()
                                                .set_name("gen_col_idx_padding")
                                                .set_mode(PointwiseMode_t::GEN_INDEX)
                                                .set_axis(3)
                                                .set_compute_data_type(DataType_t::INT32));
            col_idx_output->set_data_type(DataType_t::INT32);

            auto row_mask_output = pointwise(row_idx_output,
                                             attributes.inputs[input_names::SEQ_LEN_Q],
                                             Pointwise_attributes()
                                                 .set_name("lt_row_sq_padding")
                                                 .set_mode(PointwiseMode_t::CMP_LT)
                                                 .set_compute_data_type(DataType_t::BOOLEAN));
            row_mask_output->set_data_type(DataType_t::BOOLEAN);

            auto col_mask_output = pointwise(col_idx_output,
                                             attributes.inputs[input_names::SEQ_LEN_KV],
                                             Pointwise_attributes()
                                                 .set_name("lt_col_skv_padding")
                                                 .set_mode(PointwiseMode_t::CMP_LT)
                                                 .set_compute_data_type(DataType_t::BOOLEAN));
            col_mask_output->set_data_type(DataType_t::BOOLEAN);

            auto padding_mask_output = pointwise(row_mask_output,
                                                 col_mask_output,
                                                 Pointwise_attributes()
                                                     .set_name("and_row_col_padding")
                                                     .set_mode(PointwiseMode_t::LOGICAL_AND)
                                                     .set_compute_data_type(DataType_t::BOOLEAN));
            padding_mask_output->set_data_type(DataType_t::BOOLEAN);

            // Use a smaller value of neg infinity so that the softmax stats for rows that are fully padded dont
            // go towards NaNs/Infs when multipled by the numerous scale/descale
            auto negative_inf_padding = std::make_shared<Tensor_attributes>(attn::score_modifiers::get_negative_inf_value());

            last_dV =
                pointwise(last_dV,
                          negative_inf_padding,
                          padding_mask_output,
                          Pointwise_attributes().set_name("select_padding").set_mode(PointwiseMode_t::BINARY_SELECT));
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

            last_dV = attn::score_modifiers::sliding_window_mask(graph_,
                                                                 last_dV,
                                                                 attributes.diagonal_alignment,
                                                                 attributes.left_bound,
                                                                 attributes.right_bound,
                                                                 s_q,
                                                                 s_kv,
                                                                 s_q_ptr,
                                                                 s_kv_ptr);
            sub_nodes.emplace_back(std::move(node_));
        }

        //// Apply Softmax
        // last_dV = last_dV - stats
        last_dV = pointwise(last_dV,
                            attributes.inputs[input_names::Stats],
                            Pointwise_attributes().set_name("sub_dV_Stats").set_mode(PointwiseMode_t::SUB));

        // last_dV = exp(last_dV)
        last_dV    = pointwise(last_dV, Pointwise_attributes().set_name("exp_dV").set_mode(PointwiseMode_t::EXP));
        auto exp_S = last_dV;

        // (optional) last_dV = last_dV * dropout rng_output
        if (is_dropout_prob || is_dropout_mask) {
            last_dV =
                pointwise(last_dV,
                          rng_output,
                          Pointwise_attributes().set_name("mul_p_dropout_mask").set_mode(PointwiseMode_t::MUL));
        }

        // (optional) last_dV = last_dV * dropout_scale
        if (attributes.inputs[input_names::Dropout_scale]) {
            last_dV =
                pointwise(last_dV,
                          attributes.inputs[input_names::Dropout_scale],
                          Pointwise_attributes().set_name("mul_dS_dropout_scale").set_mode(PointwiseMode_t::MUL));
        }

        // Scale S (only for non-MXFP8 per-tensor scaling; MXFP8 keeps P in fp32)
        if (!use_mxfp8) {
            mul_attributes.set_name("scale_S");
            last_dV = pointwise(last_dV, attributes.inputs.at(input_names::Scale_S), mul_attributes);
            last_dV->set_data_type(attributes.inputs.at(input_names::Q)->get_data_type());
        }

        // Reshape P
        last_dV = reshape(last_dV, Reshape_attributes().set_name("S_transpose"));
        last_dV->set_name("S_T").set_dim({b, h_q, s_kv, s_q}).set_stride({h_q * s_q * s_kv, s_q * s_kv, 1, s_kv});
        if (!use_mxfp8) {
            last_dV->set_data_type(attributes.inputs[input_names::Q]->get_data_type());
        }

        //// P.T @ dO -> dV
        if (use_mxfp8) {
            // MXFP8: Dequantize dO_T with Descale_dO_T for P.T@dO
            auto dequant_dO_seq_attrs = Block_scale_dequantize_attributes()
                                            .set_name("DQ_dO_seq")
                                            .set_block_size({1, 32});
            auto dO_seq_dequant = std::make_shared<Tensor_attributes>();
            dO_seq_dequant->set_is_virtual(true);
            dO_seq_dequant->set_dim(attributes.inputs[input_names::dO_T]->get_dim());
            dO_seq_dequant->set_stride(attributes.inputs[input_names::dO_T]->get_stride());
            block_scale_dequantize(attributes.inputs[input_names::dO_T],
                                   attributes.inputs[input_names::Descale_dO_T],
                                   dequant_dO_seq_attrs,
                                   dO_seq_dequant);

            auto bmm_S_T_dO_attributes = Matmul_attributes().set_name("bmm_S_T_dO")
                                             .set_m_override(attributes.inputs[input_names::SEQ_LEN_KV])
                                             .set_k_override(attributes.inputs[input_names::SEQ_LEN_Q]);
            if (h_q == h_v) {
                matmul(last_dV, dO_seq_dequant, bmm_S_T_dO_attributes, attributes.outputs[output_names::dV]);
            } else {
                auto dV_fullhead = matmul(last_dV, dO_seq_dequant, bmm_S_T_dO_attributes);
                dV_fullhead->set_dim({b, h_q, s_kv, d_v});
                dV_fullhead->set_stride({h_q * s_kv * d_v, s_kv * d_v, d_v, 1});
                dV_fullhead->set_data_type(DataType_t::FLOAT);
                reduction(dV_fullhead,
                          Reduction_attributes().set_name("red_dV_head").set_mode(ReductionMode_t::ADD),
                          attributes.outputs[output_names::dV]);
            }

            auto const& amax_dv = attributes.outputs.at(output_names::Amax_dV);
            if (amax_dv != nullptr) {
                auto amax_dv_attributes =
                    Reduction_attributes().set_name("amax_dV").set_mode(ReductionMode_t::AMAX);
                reduction(attributes.outputs[output_names::dV], amax_dv_attributes, amax_dv);
            }
        } else {
            matmul_fp8(last_dV,
                       attributes.inputs[input_names::dO],
                       attributes.inputs[input_names::Descale_S],
                       attributes.inputs[input_names::Descale_dO],
                       attributes.inputs[input_names::Scale_dV],
                       Matmul_fp8_attributes().set_name("bmm_S_T_dO")
                           .set_m_override(attributes.inputs[input_names::SEQ_LEN_KV])
                           .set_k_override(attributes.inputs[input_names::SEQ_LEN_Q]),
                       attributes.outputs[output_names::dV],
                       attributes.outputs[output_names::Amax_dV]);
        }

        //// dO * V_T
        auto bmm_dO_V_T_attributes = Matmul_attributes().set_name("bmm_dO_V_T")
                                 .set_m_override(attributes.inputs[input_names::SEQ_LEN_Q])
                                 .set_n_override(attributes.inputs[input_names::SEQ_LEN_KV]);
        last_output = matmul(eff_dO, eff_V, bmm_dO_V_T_attributes);

        // Descale dO (only for non-MXFP8 per-tensor scaling)
        if (!use_mxfp8) {
            mul_attributes.set_name("descale_dO");
            last_output = pointwise(last_output, attributes.inputs.at(input_names::Descale_dO), mul_attributes);
        }

        // Descale V (only for non-MXFP8 per-tensor scaling)
        if (!use_mxfp8) {
            mul_attributes.set_name("descale_V");
            last_output = pointwise(last_output, attributes.inputs.at(input_names::Descale_V), mul_attributes);
        }

        // dP = last_output - softmax_sum
        auto dP = pointwise(last_output,
                            softmax_sum,
                            Pointwise_attributes().set_name("sub_dP_softmax_sum").set_mode(PointwiseMode_t::SUB));

        // dP = dP * exp_S
        mul_attributes.set_name("mul_dP_exp_S");
        dP = pointwise(dP, exp_S, mul_attributes);

        // (optional) dP = dP * dropout_scale
        if (attributes.inputs[input_names::Dropout_scale]) {
            dP =
                pointwise(dP,
                          attributes.inputs[input_names::Dropout_scale],
                          Pointwise_attributes().set_name("mul_dS_dropout_scale").set_mode(PointwiseMode_t::MUL));
        }

        // if (attributes.outputs[output_names::dBias]) {
        //     reduction(dP,
        //               Reduction_attributes().set_name("red_dP_dBias").set_mode(ReductionMode_t::ADD),
        //               attributes.outputs[output_names::dBias]);
        // }

        // (optional) dP = dP * attn_scale
        if (auto attn_scale_it = attributes.inputs.find(input_names::Attn_scale); attn_scale_it != attributes.inputs.end()) {
            mul_attributes.set_name("mul_dS_attn_scale");
            dP = pointwise(dP, attn_scale_it->second, mul_attributes);
        }

        // Amax dP and Scale dP (only for non-MXFP8 per-tensor scaling; MXFP8 keeps dS in fp32)
        if (!use_mxfp8) {
            auto amax_attributes = Reduction_attributes().set_name("amax_dP").set_mode(ReductionMode_t::AMAX);
            reduction(dP, amax_attributes, attributes.outputs.at(output_names::Amax_dP));

            mul_attributes.set_name("scale_dP");
            dP = pointwise(dP, attributes.inputs.at(input_names::Scale_dP), mul_attributes);
            dP->set_data_type(attributes.inputs.at(input_names::dO)->get_data_type());
        }

        //// dS @ K -> dQ
        if (use_mxfp8) {
            // MXFP8: Dequantize K_T with Descale_K_T (seq-dimension scaling) for dS@K -> dQ
            auto dequant_K_seq_attrs = Block_scale_dequantize_attributes()
                                           .set_name("DQ_K_seq")
                                           .set_block_size({1, 32});
            auto K_seq_dequant = std::make_shared<Tensor_attributes>();
            K_seq_dequant->set_is_virtual(true);
            K_seq_dequant->set_dim(attributes.inputs[input_names::K_T]->get_dim());
            K_seq_dequant->set_stride(attributes.inputs[input_names::K_T]->get_stride());

            int64_t s_kv_scale_padded = ((s_kv + 127) / 128) * 4;
            auto& sf_k_T = attributes.inputs[input_names::Descale_K_T];
            sf_k_T->set_stride({h_k * s_kv_scale_padded * 128, s_kv_scale_padded * 128, b * h_k * s_kv_scale_padded * 128, 1});

            block_scale_dequantize(attributes.inputs[input_names::K_T],
                                   attributes.inputs[input_names::Descale_K_T],
                                   dequant_K_seq_attrs,
                                   K_seq_dequant);

            auto bmm_dS_K_attributes = Matmul_attributes().set_name("bmm_dS_K")
                       .set_m_override(attributes.inputs[input_names::SEQ_LEN_Q])
                       .set_k_override(attributes.inputs[input_names::SEQ_LEN_KV]);
            matmul(dP, K_seq_dequant, bmm_dS_K_attributes, attributes.outputs[output_names::dQ]);

            auto const& amax_dq = attributes.outputs.at(output_names::Amax_dQ);
            if (amax_dq != nullptr) {
                auto amax_dq_attributes =
                    Reduction_attributes().set_name("amax_dQ").set_mode(ReductionMode_t::AMAX);
                reduction(attributes.outputs[output_names::dQ], amax_dq_attributes, amax_dq);
            }
        } else {
            auto const& kt_dim    = attributes.inputs[input_names::K]->get_dim();
            auto const& kt_stride = attributes.inputs[input_names::K]->get_stride();

            auto K = reshape(attributes.inputs[input_names::K], Reshape_attributes().set_name("reshape_K"));
            K->set_dim({kt_dim[0], kt_dim[1], kt_dim[3], kt_dim[2]})
                .set_stride({kt_stride[0], kt_stride[1], kt_stride[3], kt_stride[2]});

            auto bmm_dP_K_attributes = Matmul_fp8_attributes().set_name("bmm_dP_K")
                       .set_m_override(attributes.inputs[input_names::SEQ_LEN_Q])
                       .set_k_override(attributes.inputs[input_names::SEQ_LEN_KV]);
            matmul_fp8(dP,
                       K,
                       attributes.inputs[input_names::Descale_dP],
                       attributes.inputs[input_names::Descale_K],
                       attributes.inputs[input_names::Scale_dQ],
                       bmm_dP_K_attributes,
                       attributes.outputs[output_names::dQ],
                       attributes.outputs[output_names::Amax_dQ]);
        }

        //// dS.T * Q (transpose dS -> dS_reshape)
        auto dP_T_attributes = Reshape_attributes().set_name("dP_T");
        auto dP_T            = reshape(dP, dP_T_attributes);
        if (!use_mxfp8) {
            dP_T->set_data_type(attributes.inputs.at(input_names::dO)->get_data_type());
        }
        dP_T->set_name("dP_T").set_dim({b, h_q, s_kv, s_q}).set_stride({h_q * s_q * s_kv, s_q * s_kv, 1, s_kv});

        if (use_mxfp8) {
            // MXFP8: Dequantize Q_T with Descale_Q_T (seq-dimension scaling) for dS.T@Q -> dK
            auto dequant_Q_seq_attrs = Block_scale_dequantize_attributes()
                                           .set_name("DQ_Q_seq")
                                           .set_block_size({1, 32});
            auto Q_seq_dequant = std::make_shared<Tensor_attributes>();
            Q_seq_dequant->set_is_virtual(true);
            Q_seq_dequant->set_dim(attributes.inputs[input_names::Q_T]->get_dim());
            Q_seq_dequant->set_stride(attributes.inputs[input_names::Q_T]->get_stride());

            int64_t s_q_scale_padded = ((s_q + 127) / 128) * 4;
            auto& sf_q_T = attributes.inputs[input_names::Descale_Q_T];
            sf_q_T->set_stride({h_q * s_q_scale_padded * 128, s_q_scale_padded * 128, b * h_q * s_q_scale_padded * 128, 1});

            block_scale_dequantize(attributes.inputs[input_names::Q_T],
                                   attributes.inputs[input_names::Descale_Q_T],
                                   dequant_Q_seq_attrs,
                                   Q_seq_dequant);

            auto bmm_dS_T_Q_attributes = Matmul_attributes().set_name("bmm_dS_T_Q")
                           .set_m_override(attributes.inputs[input_names::SEQ_LEN_KV])
                           .set_k_override(attributes.inputs[input_names::SEQ_LEN_Q]);
            if (h_q == h_k) {
                matmul(dP_T, Q_seq_dequant, bmm_dS_T_Q_attributes, attributes.outputs[output_names::dK]);
            } else {
                auto dK_fullhead = matmul(dP_T, Q_seq_dequant, bmm_dS_T_Q_attributes);
                dK_fullhead->set_dim({b, h_q, s_kv, d_qk});
                dK_fullhead->set_stride({h_q * s_kv * d_qk, s_kv * d_qk, d_qk, 1});
                dK_fullhead->set_data_type(DataType_t::FLOAT);
                reduction(dK_fullhead,
                          Reduction_attributes().set_name("red_dK_head").set_mode(ReductionMode_t::ADD),
                          attributes.outputs[output_names::dK]);
            }

            auto const& amax_dk = attributes.outputs.at(output_names::Amax_dK);
            if (amax_dk != nullptr) {
                auto amax_dk_attributes =
                    Reduction_attributes().set_name("amax_dK").set_mode(ReductionMode_t::AMAX);
                reduction(attributes.outputs[output_names::dK], amax_dk_attributes, amax_dk);
            }
        } else {
            auto bmm_dP_T_Q_attributes = Matmul_fp8_attributes().set_name("bmm_dP_T_Q")
                           .set_m_override(attributes.inputs[input_names::SEQ_LEN_KV])
                           .set_k_override(attributes.inputs[input_names::SEQ_LEN_Q]);
            matmul_fp8(dP_T,
                       attributes.inputs[input_names::Q],
                       attributes.inputs[input_names::Descale_dP],
                       attributes.inputs[input_names::Descale_Q],
                       attributes.inputs[input_names::Scale_dK],
                       bmm_dP_T_Q_attributes,
                       attributes.outputs[output_names::dK],
                       attributes.outputs[output_names::Amax_dK]);
        }

        return {error_code_t::OK, ""};
    }

    std::pair<int64_t, std::unordered_map<KnobType_t, int64_t>>
    override_heuristics_query() const {
        int32_t const sm_version = context.get_sm_version();
        bool const use_new_knobs = detail::get_backend_version() >= 92300;
        // {128,128} bprop: tileM=3, tileN=2, kernelCfg=2(bprop warp), streamK=0, cgaM=0
        if (sm_version > 103 && is_deterministic_algorithm_supported_on_blackwell) {
            if (use_new_knobs) {
                return {17,
                        {{KnobType_t::TILE_M, 3},
                         {KnobType_t::TILE_N, 2},
                         {KnobType_t::KERNEL_CFG, 2},
                         {KnobType_t::STREAM_K, 0},
                         {KnobType_t::TILE_CGA_M, 0},
                         {KnobType_t::STAGES, 2}}};
            } else {
                return {17, {{KnobType_t::KERNEL_CFG, 31}, {KnobType_t::STAGES, 2}}};
            }
        } else if (is_deterministic_algorithm_supported_on_blackwell) {
            if (use_new_knobs) {
                return {5,
                        {{KnobType_t::TILE_M, 3},
                         {KnobType_t::TILE_N, 2},
                         {KnobType_t::KERNEL_CFG, 2},
                         {KnobType_t::STREAM_K, 0},
                         {KnobType_t::TILE_CGA_M, 0},
                         {KnobType_t::STAGES, 2}}};
            } else {
                return {5, {{KnobType_t::KERNEL_CFG, 31}, {KnobType_t::STAGES, 2}}};
            }
        } else {
            return {-1, {}};
        }
    }

#ifndef CUDNN_FRONTEND_SKIP_JSON_LIB
    virtual void
    serialize(json& j) const override final {
        j = attributes;
        j["is_mxfp8"] = is_mxfp8_scaling();
        if (auto const rescale_threshold = get_rescale_threshold_from_env(); rescale_threshold.has_value()) {
            j["rescale_threshold"] = rescale_threshold.value();
        }
        if (is_mxfp8_scaling()) {
            j.update(R"({"tag": "SDPA_MXFP8_BWD"})"_json);
        } else {
            j.update(R"({"tag": "SDPA_FP8_BWD"})"_json);
        }
    }
#endif
};

}  // namespace cudnn_frontend::graph
