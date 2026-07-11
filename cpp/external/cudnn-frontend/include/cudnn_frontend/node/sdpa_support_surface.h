#pragma once

#include <cstdlib>
#include <unordered_set>

#include "../../cudnn_frontend_Heuristics.h"
#include "../../cudnn_frontend_Logging.h"
#include "../graph_helpers.h"
#include "../node_interface.h"

namespace cudnn_frontend::graph {

inline error_t
SDPA_attributes::validate_sdpa_support_surface(const detail::Context& context,
                                               int64_t s_kv,
                                               bool is_paged_k,
                                               bool is_paged_v) const {
    // Extract dimensions from tensors
    int64_t s_q = inputs.at(SDPA_attributes::input_names::Q)->get_dim()[2];
    // s_kv is passed in from the caller
    int64_t h_q  = inputs.at(SDPA_attributes::input_names::Q)->get_dim()[1];
    int64_t h_k  = inputs.at(SDPA_attributes::input_names::K)->get_dim()[1];
    int64_t h_v  = inputs.at(SDPA_attributes::input_names::V)->get_dim()[1];
    int64_t d_qk = inputs.at(SDPA_attributes::input_names::Q)->get_dim()[3];
    int64_t d_v  = inputs.at(SDPA_attributes::input_names::V)->get_dim()[3];

    bool const is_ragged = inputs.at(SDPA_attributes::input_names::Q)->get_ragged_offset() ||
                           inputs.at(SDPA_attributes::input_names::K)->get_ragged_offset() ||
                           inputs.at(SDPA_attributes::input_names::V)->get_ragged_offset() ||
                           outputs.at(SDPA_attributes::output_names::O)->get_ragged_offset();

    auto const& output_tensor    = outputs.at(SDPA_attributes::output_names::O);
    auto const& output_data_type = output_tensor->get_data_type();

    auto const& bias_mask = inputs.find(SDPA_attributes::input_names::Bias);
    bool const is_bias    = (bias_mask != inputs.end() && bias_mask->second != nullptr);

    auto const& dropout_mask     = inputs.find(SDPA_attributes::input_names::Dropout_mask);
    bool const is_dropout_custom = (dropout_mask != inputs.end()) && (dropout_mask->second != nullptr);
    bool const is_dropout        = dropout_probability.has_value() || is_dropout_custom;

    bool const is_paged = is_paged_k || is_paged_v;

    auto const& rng_tensor = outputs.find(SDPA_attributes::output_names::RNG_DUMP);
    bool const is_rng      = (rng_tensor != outputs.end() && rng_tensor->second != nullptr);

    bool const max_seq_kv_explicit = max_seq_len_kv.has_value();

    auto const& attn_scale    = inputs.find(SDPA_attributes::input_names::Attn_scale);
    bool const has_attn_scale = (attn_scale != inputs.end()) && (attn_scale->second != nullptr);

    auto const& seq_len_q     = inputs.find(SDPA_attributes::input_names::SEQ_LEN_Q);
    bool const has_seq_len_q  = (seq_len_q != inputs.end()) && (seq_len_q->second != nullptr);
    auto const& seq_len_kv    = inputs.find(SDPA_attributes::input_names::SEQ_LEN_KV);
    bool const has_seq_len_kv = (seq_len_kv != inputs.end()) && (seq_len_kv->second != nullptr);

    // validation TODO:
    //    - validate stats has valid dims

    // Get device SM version from context
    CHECK_CUDNN_FRONTEND_ERROR(context.populate_sm_version_from_device());
    int32_t const sm_version = context.get_sm_version();
    int32_t const prop_major = sm_version / 10;

    // Common FP16 and FP8 validation
    // validate basic dimension requirements
    RETURN_CUDNN_FRONTEND_ERROR_IF(
        (h_q % h_k != 0) || (h_q % h_v != 0),
        error_code_t::GRAPH_NOT_SUPPORTED,
        "For group-query attention, number of heads for key and query must be a factor of number of heads for query");

    // validate options for attn_scale
    RETURN_CUDNN_FRONTEND_ERROR_IF(has_attn_scale && attn_scale_value.has_value(),
                                   error_code_t::ATTRIBUTE_NOT_SET,
                                   "attn_scale with tensor and value cannot be set at the same time.");

    // validate options for bias mask
    RETURN_CUDNN_FRONTEND_ERROR_IF(is_bias && (bias_mask->second->get_data_type() == DataType_t::BOOLEAN),
                                   error_code_t::GRAPH_NOT_SUPPORTED,
                                   "Bias mask data type cannot be boolean");

    // validate options for padding mask
    RETURN_CUDNN_FRONTEND_ERROR_IF(padding_mask && (!has_seq_len_q || !has_seq_len_kv),
                                   error_code_t::ATTRIBUTE_NOT_SET,
                                   "Padding mask requires seq_len_q and seq_len_kv to be set.");
    RETURN_CUDNN_FRONTEND_ERROR_IF((!padding_mask && !attention_score_modifier) && (has_seq_len_q || has_seq_len_kv),
                                   error_code_t::ATTRIBUTE_NOT_SET,
                                   "seq_len_q and seq_len_kv needs to be set only if padding mask is enabled.");

    RETURN_CUDNN_FRONTEND_ERROR_IF(is_ragged && ((padding_mask == false) && (attention_score_modifier == nullptr)),
                                   error_code_t::GRAPH_NOT_SUPPORTED,
                                   "Ragged offsets are only supported with padding mask.");

    // validate options for dropout mask
    RETURN_CUDNN_FRONTEND_ERROR_IF(
        dropout_probability.has_value() && is_dropout_custom,
        error_code_t::ATTRIBUTE_NOT_SET,
        "Using both, custom dropout mask and internal-mask generation using dropout probability, is ill-formed.");

    RETURN_CUDNN_FRONTEND_ERROR_IF(dropout_probability.has_value() && dropout_probability.value() == 1.0,
                                   error_code_t::ATTRIBUTE_NOT_SET,
                                   "Dropout probability cannot be 1 as corresponding scale wont be well formed.");

    // validate options for causal mask and bottom right causal mask
    RETURN_CUDNN_FRONTEND_ERROR_IF(
        has_causal_mask_bottom_right() && (!padding_mask) && s_q > s_kv,
        error_code_t::GRAPH_NOT_SUPPORTED,
        "Bottom right causal mask does not support max_s_q > max_s_kv. Please virtually slice the Q tensor and pass it "
        "as max_s_q == max_s_kv");

    RETURN_CUDNN_FRONTEND_ERROR_IF(
        has_causal_mask_bottom_right() && (is_bias || alibi_mask || is_dropout),
        error_code_t::GRAPH_NOT_SUPPORTED,
        "Bottom right causal mask is only supported with is_bias=False, is_alibi=False, is_dropout=False.");

    RETURN_CUDNN_FRONTEND_ERROR_IF(has_causal_mask_bottom_right() && (detail::get_backend_version() < 90600) &&
                                       ((s_q % 64 != 0) || (s_kv % 64 != 0)),
                                   error_code_t::GRAPH_NOT_SUPPORTED,
                                   "Bottom right causal mask is only supported with s_q multiple of 64, and s_kv "
                                   "multiple of 64, for cudnn version below 9.6.0");

    RETURN_CUDNN_FRONTEND_ERROR_IF(detail::get_backend_version() < 90600 && left_bound.has_value() &&
                                       has_causal_mask_bottom_right() && s_q != s_kv,
                                   error_code_t::GRAPH_NOT_SUPPORTED,
                                   "Sliding window attention with bottom right causal mask requires s_q == s_kv "
                                   "for cuDNN version below 9.6.0");

    // validate that datatype is set for the graph
    RETURN_CUDNN_FRONTEND_ERROR_IF(context.get_intermediate_data_type() == DataType_t::NOT_SET,
                                   error_code_t::ATTRIBUTE_NOT_SET,
                                   "Intermediate tensor data type needs to be set as internal tensors require it.");

    if (mma_core_mode == DataType_t::FP8_E4M3 || mma_core_mode == DataType_t::FP8_E5M2) {
        // FP8 specific validation

        RETURN_CUDNN_FRONTEND_ERROR_IF((prop_major == 12) && is_ragged,
                                       error_code_t::GRAPH_NOT_SUPPORTED,
                                       "sdpa fp8 with THD not supported for sm120 yet.");

        // version specific validation
        RETURN_CUDNN_FRONTEND_ERROR_IF(detail::get_backend_version() < 90100,
                                       error_code_t::GRAPH_NOT_SUPPORTED,
                                       "sdpa fp8 forward operation is only supported starting cudnn 9.1.0. Please "
                                       "consider upgrading your current version.");
        RETURN_CUDNN_FRONTEND_ERROR_IF(detail::get_backend_version() == 91000,
                                       error_code_t::GRAPH_NOT_SUPPORTED,
                                       "sdpa fp8 forward operation is not supported on cudnn 9.10.0. Please "
                                       "consider upgrading your current version.");
        RETURN_CUDNN_FRONTEND_ERROR_IF(
            prop_major < 9,
            error_code_t::GRAPH_NOT_SUPPORTED,
            "sdpa fp8 forward operation is only supported on Hopper architecture and newer. Please "
            "consider using a newer architecture.");

        // FP8 does not support bias (TE constraint)
        RETURN_CUDNN_FRONTEND_ERROR_IF(is_bias, error_code_t::GRAPH_NOT_SUPPORTED, "SDPA FP8 does not support bias");

        // validate basic dimension requirements
        // d_qk=192 with d_v=128 is only supported starting from cuDNN 9.19
        bool const d192_v128_supported = (detail::get_backend_version() >= 91900);
        if (prop_major >= 10) {
            RETURN_CUDNN_FRONTEND_ERROR_IF(
                ((d_qk > 128) || (d_qk % 16 != 0)) && !(d192_v128_supported && d_qk == 192 && d_v == 128),
                error_code_t::GRAPH_NOT_SUPPORTED,
                "hidden_dim d_qk should be less than or equal to 128 and hidden_dim d_qk "
                "should be multiple of 16 unless d_qk == 192 and d_v == 128 (requires cuDNN 9.19+)");
            RETURN_CUDNN_FRONTEND_ERROR_IF(
                ((d_v > 128) || (d_v % 16 != 0)),
                error_code_t::GRAPH_NOT_SUPPORTED,
                "hidden_dim d_v should be less than or equal to 128 and hidden_dim d_v should be multiple of 16");
        } else {
            RETURN_CUDNN_FRONTEND_ERROR_IF(
                (d_qk > 256) || (d_qk % 16 != 0) || (d_v > 256) || (d_v % 16 != 0),
                error_code_t::GRAPH_NOT_SUPPORTED,
                "hidden_dim shoud be less than or equal to 256 and hidden_dim should be multiple of 16");
        }

        // Validate options for causal_mask_bottom_right
        RETURN_CUDNN_FRONTEND_ERROR_IF(has_causal_mask_bottom_right() && detail::get_backend_version() < 90700,
                                       error_code_t::GRAPH_NOT_SUPPORTED,
                                       "For cuDNN version below 9.7.0, bottom right causal masking is not supported.");

        RETURN_CUDNN_FRONTEND_ERROR_IF(
            has_causal_mask_bottom_right() && prop_major < 10,
            error_code_t::GRAPH_NOT_SUPPORTED,
            "sdpa fp8 forward operation is only supported on Blackwell architecture and newer. Please "
            "consider using a newer architecture.");

        // if output data type is half or bfloat16, and version is below 9.13 or is not blackwell, return NOT_SUPPORTED
        RETURN_CUDNN_FRONTEND_ERROR_IF(
            (output_data_type == DataType_t::HALF || output_data_type == DataType_t::BFLOAT16) &&
                (detail::get_backend_version() < 91300 || prop_major < 10),
            error_code_t::GRAPH_NOT_SUPPORTED,
            "sdpa fp8 forward with HALF/BFLOAT16 output is only supported on Blackwell architecture "
            "with cuDNN version 9.13.0 and newer.");
    } else if (mma_core_mode == DataType_t::HALF) {
        // FP16 specific validation

        // Bug workarounds for known problematic versions
        RETURN_CUDNN_FRONTEND_ERROR_IF(
            detail::get_backend_version() == 91000 || detail::get_backend_version() == 91001,
            error_code_t::GRAPH_NOT_SUPPORTED,
            "SDPA FP16/BF16 forward is not supported on cuDNN 9.10.0/9.10.1 due to known bugs. "
            "Please consider upgrading to 9.10.2 or newer.");

        // 9.14.0 sliding window bug: non-causal + s_kv > 1024 + sliding window
        RETURN_CUDNN_FRONTEND_ERROR_IF(
            detail::get_backend_version() == 91400 && s_kv > 1024 && left_bound.has_value() &&
                !has_causal_like_masking(),
            error_code_t::GRAPH_NOT_SUPPORTED,
            "cuDNN 9.14.0 has a known bug with non-causal + s_kv > 1024 + sliding window attention. "
            "Please consider upgrading to 9.14.1 or newer.");

        RETURN_CUDNN_FRONTEND_ERROR_IF(
            (attention_score_modifier != nullptr) &&
                (alibi_mask || has_causal_like_masking() || padding_mask || left_bound.has_value()),
            error_code_t::GRAPH_NOT_SUPPORTED,
            "Attention score mod enabled and hence other subgraphs are disabled.");

        // validate basic dimension requirements
        RETURN_CUDNN_FRONTEND_ERROR_IF(
            (d_qk % 8 != 0) || (d_v % 8 != 0), error_code_t::GRAPH_NOT_SUPPORTED, "hidden_dim should be multiple of 8");

        // validate alibi requirements
        RETURN_CUDNN_FRONTEND_ERROR_IF(alibi_mask && !(right_bound.has_value() && right_bound.value() == 0),
                                       error_code_t::GRAPH_NOT_SUPPORTED,
                                       "When alibi mask is used, diagonal_band_right_bound needs to be set to 0.");

        RETURN_CUDNN_FRONTEND_ERROR_IF(
            left_bound.has_value() && left_bound.value() <= 0 && detail::get_backend_version() < 91000,
            error_code_t::INVALID_VALUE,
            "Left bound (Sliding window length) should be greater than zero when set.");

        RETURN_CUDNN_FRONTEND_ERROR_IF(left_bound.has_value() && (!padding_mask) && s_q > s_kv,
                                       error_code_t::GRAPH_NOT_SUPPORTED,
                                       "Sliding window attention is only supported with max_s_q <= max_s_kv.");

        RETURN_CUDNN_FRONTEND_ERROR_IF(
            left_bound.has_value() && (s_q * left_bound.value() == s_kv * left_bound.value()) &&
                (detail::get_backend_version() <= 90900) && (prop_major == 9) && has_causal_mask_bottom_right(),
            error_code_t::GRAPH_NOT_SUPPORTED,
            "On Hopper architecture, this specific combination of s_q, s_kv, and left_bound + right_bound + bottom "
            "right diagonal alignment is not supported for backend version 9.9 or below");

        if ((detail::get_backend_version() < 91002)) {
            RETURN_CUDNN_FRONTEND_ERROR_IF(
                left_bound.has_value() && (!has_causal_like_masking() || is_dropout || is_bias),
                error_code_t::GRAPH_NOT_SUPPORTED,
                "Left and right bounds are only supported with is_dropout=False, is_bias=False. And the diagonal "
                "alignment must be set.");
        }

        RETURN_CUDNN_FRONTEND_ERROR_IF(right_bound.has_value() && right_bound.value() < 0,
                                       error_code_t::INVALID_VALUE,
                                       "Right bound needs to be larger than or equal to zero");

        // Validate options for s_q == 1
        const bool is_decode_only = (s_q == 1);
        RETURN_CUDNN_FRONTEND_ERROR_IF(is_decode_only && (prop_major == 10) && (d_qk > 128 || d_v > 128) &&
                                           (detail::get_backend_version() <= 90900),
                                       error_code_t::GRAPH_NOT_SUPPORTED,
                                       "decode only mode, i.e. s_q == 1 not supported for blackwell architecture with "
                                       "d_qk or d_v > 128 for backend version 9.9 or below");

        RETURN_CUDNN_FRONTEND_ERROR_IF(
            is_decode_only && (detail::get_backend_version() <= 90900) && (right_bound.has_value()),
            error_code_t::GRAPH_NOT_SUPPORTED,
            "decode only mode, i.e. s_q == 1, not supported with masking (right_bound is set) for backend version 9.9 "
            "or below");

        RETURN_CUDNN_FRONTEND_ERROR_IF(is_decode_only && has_sink_token(),
                                       error_code_t::GRAPH_NOT_SUPPORTED,
                                       "decode only mode, i.e. s_q == 1, not supported with sink_token");

        // validate options for paged attention
        RETURN_CUDNN_FRONTEND_ERROR_IF(
            is_paged && (d_qk > 128 || d_v > 128) && detail::get_backend_version() <= 90900,
            error_code_t::GRAPH_NOT_SUPPORTED,
            "Paged attention only supported with d_qk and d_v <= 128 for backend version 9.9 or below");

        RETURN_CUDNN_FRONTEND_ERROR_IF(is_paged && is_ragged && detail::get_backend_version() < 90700,
                                       error_code_t::GRAPH_NOT_SUPPORTED,
                                       "Paged caches are not supported in combination with ragged offsets.");

        RETURN_CUDNN_FRONTEND_ERROR_IF(is_paged && (!has_seq_len_q || !has_seq_len_kv),
                                       error_code_t::GRAPH_NOT_SUPPORTED,
                                       "Paged caches can only be used in combination with padding mask and variable "
                                       "sequence lengths for both Q and KV.");

        RETURN_CUDNN_FRONTEND_ERROR_IF(
            !is_paged && max_seq_kv_explicit,
            error_code_t::GRAPH_NOT_SUPPORTED,
            "When not using paged attention, there is no need to explicitly set max kv sequence length.");

        if (max_seq_kv_explicit) {
            auto max_seq_kv = max_seq_len_kv.value();

            RETURN_CUDNN_FRONTEND_ERROR_IF(is_bias && (bias_mask->second->get_dim()[3] != max_seq_kv),
                                           error_code_t::GRAPH_NOT_SUPPORTED,
                                           "Value set through set_paged_attention_max_seq_len_kv is incompatible with "
                                           "the sequence length of the bias");

            RETURN_CUDNN_FRONTEND_ERROR_IF(is_rng && rng_tensor->second->get_dim()[3] != max_seq_kv,
                                           error_code_t::GRAPH_NOT_SUPPORTED,
                                           "Value set through set_paged_attention_max_seq_len_kv is incompatible with "
                                           "the sequence length of the RNG_DUMP");
        }

        // Additional validation for paged attention with packed page tables
        RETURN_CUDNN_FRONTEND_ERROR_IF(
            ((is_paged_k && inputs.at(SDPA_attributes::input_names::Page_table_K)->get_ragged_offset()) ||
             (is_paged_v && inputs.at(SDPA_attributes::input_names::Page_table_V)->get_ragged_offset())) &&
                detail::get_backend_version() < 91002,
            error_code_t::GRAPH_NOT_SUPPORTED,
            "Paged attention with packed page tables only supported with cudnn version 9.10.2 and above");

        // SM version check for SDPA
        RETURN_CUDNN_FRONTEND_ERROR_IF(prop_major < 8,
                                       error_code_t::GRAPH_NOT_SUPPORTED,
                                       "SDPA FP16/BF16 requires SM80 (Ampere) or newer architecture");

        // version specific validation by architecture
        if (prop_major == 8) {
            RETURN_CUDNN_FRONTEND_ERROR_IF(
                detail::get_backend_version() <= 90900 && ((d_qk > 128) || (d_v > 128)),
                error_code_t::GRAPH_NOT_SUPPORTED,
                "head_dim should be less than or equal to 128 for backend version 9.9 or below on ampere architecture");
        }
        if (prop_major == 9) {
            RETURN_CUDNN_FRONTEND_ERROR_IF(
                detail::get_backend_version() <= 90900 && ((d_qk > 256) || (d_v > 256)),
                error_code_t::GRAPH_NOT_SUPPORTED,
                "head_dim should be less than or equal to 256 for backend version 9.9 or below on hopper architecture");
        }
        if (prop_major == 10) {
            RETURN_CUDNN_FRONTEND_ERROR_IF((detail::get_backend_version() < 90900) && ((d_qk > 128) || (d_v > 128)),
                                           error_code_t::GRAPH_NOT_SUPPORTED,
                                           "head_dim should be less than or equal to 128 for backend version 9.8 or "
                                           "below on blackwell architecture");
        }

        RETURN_CUDNN_FRONTEND_ERROR_IF(detail::get_backend_version() < 90500 && is_paged,
                                       error_code_t::GRAPH_NOT_SUPPORTED,
                                       "For cuDNN version below 9.5.0, paged caches are not supported");

        if (is_ragged) {
            RETURN_CUDNN_FRONTEND_ERROR_IF((context.get_sm_version() > 0 && context.get_sm_version() < 90 &&
                                            detail::get_backend_version() < 91801),
                                           error_code_t::GRAPH_NOT_SUPPORTED,
                                           "THD (ragged offset) is only supported in Hopper and above : " +
                                               std::to_string(context.get_sm_version()));
        }
    } else {
        RETURN_CUDNN_FRONTEND_ERROR_IF(true, error_code_t::GRAPH_NOT_SUPPORTED, "Unsupported mma core mode");
    }

    // Check whether the selected implementation supports the requested features.
    CHECK_CUDNN_FRONTEND_ERROR(verify_sdpa_support_surface_for_implementation(context, implementation));

    return {error_code_t::OK, ""};
}

// Verify that the underlying implementation supports all the features in these attributes.
// Unlike `validate_sdpa_support_surface()`, this may be called before validation, so:
//   * don't assume any particular keys already exist in `inputs` or `outputs`
//   * don't assume any tensor dims or strides are already set
// We return error codes directly instead of using `RETURN_CUDNN_FRONTEND_ERROR_IF`
// to avoid unneeded logging when this function is being called in a non-error-generating
// situation (e.g. during auto-select of SDPA implementation).
inline error_t
SDPA_attributes::verify_sdpa_support_surface_for_implementation(const detail::Context& context,
                                                                AttentionImplementation_t impl) const {
    switch (impl) {
        case AttentionImplementation_t::AUTO:
            // This function should not be called with AUTO.
            return {error_code_t::INVALID_VALUE,
                    "Can't call verify_sdpa_support_surface_for_implementation with impl=AUTO"};
        case AttentionImplementation_t::COMPOSITE:
            for (const auto& [key, value] : inputs) {
                RETURN_CUDNN_FRONTEND_ERROR_IF(key == input_names::Block_mask && value != nullptr,
                                               error_code_t::GRAPH_NOT_SUPPORTED,
                                               "Composite SDPA node doesn't support Block_mask input");
            }
            break;
        case AttentionImplementation_t::UNIFIED: {
            auto effective_cudnn_ver = std::min(detail::get_backend_version(), detail::get_compiled_version());
            RETURN_CUDNN_FRONTEND_ERROR_IF(effective_cudnn_ver < 91301,
                                           error_code_t::GRAPH_NOT_SUPPORTED,
                                           "Unified SDPA node requires cuDNN 9.13.1");

            RETURN_CUDNN_FRONTEND_ERROR_IF(context.get_dynamic_shape_enabled(),
                                           error_code_t::GRAPH_NOT_SUPPORTED,
                                           "Unified SDPA node doesn't yet support dynamic shape");

            // TODO: Provide smarter error messages that provide the required cuDNN version for each input.
            std::unordered_set<SDPA_attributes::input_names> allowed_input_names{
                input_names::Q, input_names::K, input_names::V, input_names::Attn_scale};
            std::string allowed_input_msg =
                "Unified SDPA node doesn't yet support inputs other than Q, K, V, Attn_scale";

            if (effective_cudnn_ver >= 91400) {
                allowed_input_names.insert({input_names::Block_mask});
                allowed_input_msg += ", Block_mask";
            }

            if (effective_cudnn_ver >= 91500) {
                allowed_input_names.insert({input_names::Page_table_K,
                                            input_names::Page_table_V,
                                            input_names::SEQ_LEN_Q,
                                            input_names::SEQ_LEN_KV});
                allowed_input_msg += ", Page_table_K, Page_table_V, SEQ_LEN_Q, SEQ_LEN_KV";
            }

            if (effective_cudnn_ver >= 92100) {
                // NOTE: For unified engine, we support dropout via Seed and Offset only.
                // Custom dropout mask (via Dropout_mask, Dropout_scale) is not supported.
                allowed_input_names.insert(
                    {input_names::Bias, input_names::Seed, input_names::Offset, input_names::SINK_TOKEN});
                allowed_input_msg += ", Bias, Seed, Offset, SINK_TOKEN";
            }

            for (const auto& [key, value] : inputs) {
                if (allowed_input_names.find(key) == allowed_input_names.end() && value != nullptr) {
                    return {error_code_t::GRAPH_NOT_SUPPORTED, allowed_input_msg};
                }
            }

            std::unordered_set<SDPA_attributes::output_names> allowed_output_names{output_names::O,
                                                                                   output_names::Stats};
            std::string allowed_output_msg = "Unified SDPA node doesn't yet support outputs other than O, Stats";

            if (effective_cudnn_ver >= 92100) {
                allowed_output_names.insert({output_names::RNG_DUMP, output_names::Max, output_names::Sum_exp});
                allowed_output_msg += ", RNG_DUMP, Max, Sum_exp";
            }

            for (const auto& [key, value] : outputs) {
                if (allowed_output_names.find(key) == allowed_output_names.end() && value != nullptr) {
                    return {error_code_t::GRAPH_NOT_SUPPORTED, allowed_output_msg};
                }
            }

            if (alibi_mask && effective_cudnn_ver < 92100) {
                return {error_code_t::GRAPH_NOT_SUPPORTED,
                        "Alibi mask for unified SDPA node requires cuDNN 9.21.0 or above"};
            }

            if (padding_mask && effective_cudnn_ver < 91500) {
                return {error_code_t::GRAPH_NOT_SUPPORTED,
                        "Padding mask for unified SDPA node requires cuDNN 9.15.0 or above"};
            }

            if ((left_bound.has_value() || right_bound.has_value()) && effective_cudnn_ver < 92100) {
                return {error_code_t::GRAPH_NOT_SUPPORTED,
                        "Left bound or right bound for unified SDPA node requires cuDNN 9.21.0 or above"};
            }

            if (diagonal_alignment != DiagonalAlignment_t::TOP_LEFT && effective_cudnn_ver < 92100) {
                return {error_code_t::GRAPH_NOT_SUPPORTED,
                        "Diagonal alignment for unified SDPA node requires cuDNN 9.21.0 or above"};
            }

            if (dropout_probability.has_value() && effective_cudnn_ver < 92200) {
                return {error_code_t::GRAPH_NOT_SUPPORTED, "Dropout for unified SDPA node requires cuDNN 9.22.0"};
            }

            if (dropout_probability.has_value() && generate_stats.value_or(false)) {
                return {error_code_t::GRAPH_NOT_SUPPORTED,
                        "Dropout for unified SDPA node with generated stats is not supported"};
            }

            // Unified engine in cuDNN < 9.15 can't meaningfully support max sequence length,
            // while versions >= 9.15 "support" it by ignoring it (unified engine doesn't need it).
            if (max_seq_len_kv.has_value() && effective_cudnn_ver < 91500) {
                return {error_code_t::GRAPH_NOT_SUPPORTED,
                        "Max sequence length for unified SDPA node cannot be set in cuDNN < 9.15.0"};
            }

            if (attention_score_modifier != nullptr && effective_cudnn_ver < 92100) {
                return {error_code_t::GRAPH_NOT_SUPPORTED,
                        "Attention score modifier for unified SDPA node requires cuDNN 9.21.0 or above"};
            }

            if (mma_core_mode != DataType_t::HALF) {
                return {error_code_t::GRAPH_NOT_SUPPORTED,
                        "Unified SDPA node doesn't yet support a data type other than fp16/bf16"};
            }

            if ((compute_data_type != DataType_t::NOT_SET && compute_data_type != DataType_t::FLOAT) ||
                context.get_compute_data_type() != DataType_t::FLOAT) {
                return {error_code_t::GRAPH_NOT_SUPPORTED,
                        "Unified SDPA node doesn't yet support compute data type other than float"};
            }
        } break;
    }

    return {error_code_t::OK, ""};
}

}  // namespace cudnn_frontend::graph
