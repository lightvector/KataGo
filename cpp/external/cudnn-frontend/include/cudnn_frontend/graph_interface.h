#pragma once

#include <atomic>
#include <mutex>
#include <unordered_map>
#include <stdexcept>
#include <string>

#include "../cudnn_frontend_version.h"
#include "node/batchnorm.h"
#include "node/batchnorm_inference.h"
#include "node/bn_finalize.h"
#include "node/conv_fprop.h"
#include "node/conv_dgrad.h"
#include "node/conv_wgrad.h"
#include "node/dbn.h"
#include "node/dln.h"
#include "node/dbn_weight.h"
#include "node/genstats.h"
#include "node/layernorm.h"
#include "node/adaptive_layernorm.h"
#include "node/instancenorm.h"
#include "node/rmsnorm.h"
#include "node/rope.h"
#include "node/rope_backward.h"
#include "node/resample.h"
#include "node/reshape.h"
#include "node/slice.h"
#include "node/transpose.h"
// #include "node/scaled_dot_product_attention.h"
#include "node/scaled_dot_product_flash_attention.h"
#include "node/sdpa_fp8_bwd.h"
#include "node/block_scale_quantize.h"
#include "node/block_scale_dequantize.h"
#include "node/concatenate.h"
#include "node/moe_grouped_matmul.h"
#include "node/moe_grouped_matmul_bwd.h"

#include "backend/backend_descriptor.h"
#include "plans.h"
#include "knobs.h"
#include "graph_helpers.h"
#include "backend/kernel_cache.h"

namespace cudnn_frontend::graph {

class Graph : public ICudnn, public INode {
   private:
    std::unordered_set<std::shared_ptr<Tensor_attributes>> full_graph_inputs;
    std::unordered_set<Tensor_attributes::uid_t> used_uids;
    int64_t fe_workspace_size = 0;
    uint64_t graph_uid;

    std::unordered_set<std::shared_ptr<Tensor_attributes>> deserialized_tensor_properties;
    std::unordered_map<uid_t, pass_by_values_t> deserialized_pass_by_value;
    std::unordered_map<uid_t, std::tuple<int64_t, int64_t, std::vector<float>>> deserialized_workspace_modifications;

    // Cached values computed during build/deserialize, used during execute to avoid repeated collection.
    // These are mutable because execute() is const but needs non-const access for pointer extraction.
    mutable std::unordered_map<uid_t, pass_by_values_t> cached_pass_by_value;
    mutable std::unordered_map<uid_t, std::tuple<int64_t, int64_t, std::vector<float>>> cached_workspace_modifications;

    // char: 'x'=hex, 'd'=decimal, 'b'=base64
    std::vector<std::pair<std::shared_ptr<Tensor_attributes>, char>> tensors_to_dump;

    error_t
    get_pre_assigned_uids(std::unordered_set<Tensor_attributes::uid_t> &used_uids) {
        for (auto const &input : full_graph_inputs) {
            if (input->has_uid()) {
                auto uid  = input->get_uid();
                auto iter = used_uids.find(uid);
                RETURN_CUDNN_FRONTEND_ERROR_IF(iter != used_uids.end(),
                                               error_code_t::INVALID_VALUE,
                                               "uid " + std::to_string(uid) + " for tensor named " + input->get_name() +
                                                   " has been already assigned to another tensor.");
                used_uids.insert(uid);
            }
        }
        for (auto const &output : full_graph_outputs) {
            if (output->has_uid()) {
                auto uid  = output->get_uid();
                auto iter = used_uids.find(uid);
                RETURN_CUDNN_FRONTEND_ERROR_IF(iter != used_uids.end(),
                                               error_code_t::INVALID_VALUE,
                                               "uid " + std::to_string(uid) + " for tensor named " +
                                                   output->get_name() +
                                                   " has been already assigned to another tensor.");
                used_uids.insert(uid);
            }
        }

        return {error_code_t::OK, ""};
    }

    error_t
    log_tensors_to_dump_(cudnnHandle_t handle,
                         std::unordered_map<int64_t, void *> const &tensor_uid_to_pointer_map) const {
        if (!isLoggingTensorDumpEnabled()) {
            return {error_code_t::OK, ""};
        }

        for (auto const &[uid, ptr] : tensor_uid_to_pointer_map) {
            CHECK_CUDNN_FRONTEND_ERROR(detail::log_variant_pack_memory_type(uid, ptr));
        }

        cudaStream_t stream;
        _CUDNN_CHECK_CUDNN_ERROR(detail::get_stream(handle, &stream));
        for (auto const &[tensor, fmt] : tensors_to_dump) {
            auto it = tensor_uid_to_pointer_map.find(tensor->get_uid());
            if (it != tensor_uid_to_pointer_map.end()) {
                auto const &dims    = tensor->get_dim();
                size_t num_elements = 1;
                for (auto d : dims) num_elements *= static_cast<size_t>(d);
                size_t elem_size = detail::get_data_type_size(tensor->get_data_type());
                CHECK_CUDNN_FRONTEND_ERROR(detail::log_dump_tensor_content(
                    it->first, tensor->get_name(), it->second, num_elements, elem_size, fmt, stream));
            }
        }

        return {error_code_t::OK, ""};
    }

    error_t
    pre_validate_node() const override final {
        RETURN_CUDNN_FRONTEND_ERROR_IF(
            (context.get_dynamic_shape_enabled() || kernel_cache != nullptr) && detail::get_backend_version() < 90400,
            error_code_t::GRAPH_NOT_SUPPORTED,
            "Dynamic shapes or kernel caching enabled, but cuDNN version < 9.4!");
        RETURN_CUDNN_FRONTEND_ERROR_IF(((context.get_dynamic_shape_enabled() == false) && (kernel_cache != nullptr)),
                                       error_code_t::GRAPH_NOT_SUPPORTED,
                                       "Kernel caching enabled but dynamic shapes is disabled");
        if (detail::get_backend_version() != detail::get_compiled_version()) {
            CUDNN_FE_LOG_LABEL_ENDL("INFO: The cuDNN version used at compilation ("
                                    << detail::get_compiled_version() << ") and the one used at runtime ("
                                    << detail::get_backend_version() << ") differ.");
        }
        return {error_code_t::OK, ""};
    }

    error_t
    infer_properties_node() override final {
        return {error_code_t::OK, ""};
    }

    error_t
    post_validate_node() const override final {
        return {error_code_t::OK, ""};
    }

    virtual error_t
    collect_pass_by_value_tensors_node(
        std::unordered_map<uid_t, pass_by_values_t> &pass_by_values) const override final {
        for (auto [uid, value] : deserialized_pass_by_value) {
            pass_by_values.emplace(uid, value);
        }
        return {error_code_t::OK, ""};
    }

    virtual error_t
    collect_tensors_in_workspace_node(
        std::unordered_map<Tensor_attributes::uid_t, std::tuple<int64_t, int64_t, std::vector<float>>>
            &worskspace_modifications,
        int64_t &) const override {
        for (auto [uid, value] : deserialized_workspace_modifications) {
            worskspace_modifications.emplace(uid, value);
        }
        return {error_code_t::OK, ""};
    }

    virtual error_t
    create_cudnn_tensors_node(std::unordered_map<int64_t, std::shared_ptr<cudnn_frontend::Tensor>> &,
                              int64_t &,
                              std::unordered_set<int64_t> const &) const override final {
        return {error_code_t::OK, ""};
    }

    error_t
    extend_tensor_map_with_workspace_tensors_(
        std::unordered_map<int64_t, void *> &tensor_to_pointer_map,
        void *workspace,
        std::unordered_map<uid_t, std::tuple<int64_t, int64_t, std::vector<float>>> const &worskspace_modifications)
        const {
        for (auto const &[uid, data] : worskspace_modifications) {
            tensor_to_pointer_map.emplace(uid, static_cast<char *>(workspace) + std::get<1>(data));
        }
        return {error_code_t::OK, ""};
    }

    error_t
    extend_tensor_map_with_pass_by_value_tensors_(
        std::unordered_map<int64_t, void *> &tensor_to_pointer_map,
        std::unordered_map<uid_t, pass_by_values_t> &tensor_to_pass_by_value) const {
        for (auto &[uid, value] : tensor_to_pass_by_value) {
            if (half *half_value_ptr = std::get_if<half>(&value)) {
                tensor_to_pointer_map.emplace(uid, half_value_ptr);
            } else if (nv_bfloat16 *nv_bfloat16_value_ptr = std::get_if<nv_bfloat16>(&value)) {
                tensor_to_pointer_map.emplace(uid, nv_bfloat16_value_ptr);
            } else if (int32_t *int32_t_value_ptr = std::get_if<int32_t>(&value)) {
                tensor_to_pointer_map.emplace(uid, int32_t_value_ptr);
            } else if (int64_t *int64_t_value_ptr = std::get_if<int64_t>(&value)) {
                tensor_to_pointer_map.emplace(uid, int64_t_value_ptr);
            } else if (float *float_value_ptr = std::get_if<float>(&value)) {
                tensor_to_pointer_map.emplace(uid, float_value_ptr);
            } else if (double *double_value_ptr = std::get_if<double>(&value)) {
                tensor_to_pointer_map.emplace(uid, double_value_ptr);
            } else {
                RETURN_CUDNN_FRONTEND_ERROR_IF(
                    true, error_code_t::INVALID_VARIANT_PACK, "Unexpected type for pass by value tensor.");
            }
        }
        return {error_code_t::OK, ""};
    }

    error_t
    make_variant_pack_replacements(
        std::unordered_map<int64_t, void *> &tensor_to_pointer_map,
        std::unordered_map<Tensor_attributes::uid_t, std::pair<Tensor_attributes::uid_t, int64_t>> replacements) const {
        for (auto &[from_uid, value] : replacements) {
            const auto &[to_uid, start_offset] = value;

            // Check if from_uid exists in the map
            auto it = tensor_to_pointer_map.find(from_uid);
            RETURN_CUDNN_FRONTEND_ERROR_IF(it == tensor_to_pointer_map.end(),
                                           error_code_t::INVALID_VARIANT_PACK,
                                           "Variant pack expected uid " + std::to_string(from_uid) + " but not found.");

            // Perform pointer arithmetic
            tensor_to_pointer_map[to_uid] = static_cast<void *>(static_cast<char *>(it->second) + start_offset);
        }
        return {error_code_t::OK, ""};
    }

    int64_t
    get_max_cudnn_workspace_size() const {
        return get_max_cudnn_workspace_size_node();
    }

    // Key: uid to replace in variant pack
    // Value: uid to replace with, start offset to add to pointer
    std::unordered_map<Tensor_attributes::uid_t, std::pair<Tensor_attributes::uid_t, int64_t>>
        variant_pack_replacements;

    error_t
    run_auxiliary_kernels(
        cudnnHandle_t handle,
        void *fe_workspace,
        std::unordered_map<uid_t, std::tuple<int64_t, int64_t, std::vector<float>>> &workspace_modifications) const {
        cudaStream_t stream;
        _CUDNN_CHECK_CUDNN_ERROR(detail::get_stream(handle, &stream));
        char *workspace = static_cast<char *>(fe_workspace);

        for (auto [uid, data] : workspace_modifications) {
            (void)uid;
            if (std::get<0>(data) == 0) {
                auto &vec_data = std::get<2>(data);
                _CUDNN_CHECK_CUDA_ERROR(detail::cuda_mem_cpy_async(workspace + std::get<1>(data),
                                                                   vec_data.data(),
                                                                   vec_data.size() * sizeof(float),
                                                                   cudaMemcpyHostToDevice,
                                                                   stream));
            } else if (std::get<0>(data) == 1) {
                int64_t memset_size = (int64_t)std::get<2>(data)[0];
                _CUDNN_CHECK_CUDA_ERROR(
                    detail::cuda_mem_set_async(workspace + std::get<1>(data), 0, memset_size, stream));
            }
        }
        return {error_code_t::OK, ""};
    }

    size_t
    key(bool remove_shape) {
#ifndef CUDNN_FRONTEND_SKIP_JSON_LIB
        json j;
        serialize(j);
        j.erase("graph_uid");
        if (remove_shape) {
            for (auto &tensor : j["tensors"]) {
                tensor["dim"].clear();
                tensor["stride"].clear();
            }
        }
        return std::hash<json>{}(j);
#else
        CUDNN_FRONTEND_UNUSED(remove_shape);
        return 1;
#endif
    }

    // Private unified sdpa method - internal implementation for both FP16 and FP8 modes
    inline SDPA_attributes::SDPA_outputs
    sdpa_internal(std::shared_ptr<Tensor_attributes> q,
                  std::shared_ptr<Tensor_attributes> k,
                  std::shared_ptr<Tensor_attributes> v,
                  SDPA_attributes &&attributes) {
        // Set inputs
        attributes.inputs[SDPA_attributes::input_names::Q] = q;
        attributes.inputs[SDPA_attributes::input_names::K] = k;
        attributes.inputs[SDPA_attributes::input_names::V] = v;

        // Make required output tensors
        SDPA_attributes::SDPA_outputs sdpa_outputs;

        sdpa_outputs.O = attributes.outputs[SDPA_attributes::output_names::O] = output_tensor(attributes.name + "::O");

        if (attributes.generate_stats == true) {
            sdpa_outputs.Stats = attributes.outputs[SDPA_attributes::output_names::Stats] =
                output_tensor(attributes.name + "::Stats");
        }

        // Dropout mask dump (created conditionally based on dropout parameters)
        if (attributes.outputs.find(SDPA_attributes::output_names::RNG_DUMP) != attributes.outputs.end() &&
            attributes.outputs.at(SDPA_attributes::output_names::RNG_DUMP) != nullptr) {
            sdpa_outputs.RNG_DUMP = attributes.outputs[SDPA_attributes::output_names::RNG_DUMP];
        }

        // FP8-specific outputs (created conditionally based on FP8 scaling parameters)
        if (attributes.inputs.find(SDPA_attributes::input_names::Descale_S) != attributes.inputs.end() &&
            attributes.inputs.at(SDPA_attributes::input_names::Descale_S) != nullptr) {
            sdpa_outputs.Amax_S = attributes.outputs[SDPA_attributes::output_names::Amax_S] =
                output_tensor(attributes.name + "::Amax_S");
        }
        if (attributes.mma_core_mode == DataType_t::FP8_E4M3 || attributes.mma_core_mode == DataType_t::FP8_E5M2) {
            sdpa_outputs.Amax_O = attributes.outputs[SDPA_attributes::output_names::Amax_O] =
                output_tensor(attributes.name + "::Amax_O");
        }

        if (attributes.implementation == AttentionImplementation_t::AUTO) {
            // Sets attributes.implementation to a supporting implementation,
            // or leaves as AUTO if none found
            attributes._auto_select_implementation(context);
        }

        switch (attributes.implementation) {
            case AttentionImplementation_t::AUTO:
                throw std::runtime_error("No suitable implementation for given SDPA_attributes");
                break;
            case AttentionImplementation_t::COMPOSITE:
                sub_nodes.emplace_back(std::make_unique<CompositeSDPANode>(std::move(attributes), context));
                break;
            case AttentionImplementation_t::UNIFIED:
                sub_nodes.emplace_back(std::make_unique<UnifiedSDPANode>(std::move(attributes), context));
                break;
        }

        return sdpa_outputs;
    }

    // Register an OSS NVRTC engine for SDPA by extracting tensor metadata from the SDPA node's attributes
    error_t
    register_oss_engine_() {
        // Find the SDPA node in the graph's sub_nodes via dynamic_cast
        SDPA_attributes const *sdpa_attrs = nullptr;
        for (auto const &sub_node : sub_nodes) {
            if (auto *composite = dynamic_cast<CompositeSDPANode *>(sub_node.get())) {
                sdpa_attrs = &composite->attributes;
                break;
            }
            if (auto *unified = dynamic_cast<UnifiedSDPANode *>(sub_node.get())) {
                sdpa_attrs = &unified->attributes;
                break;
            }
        }

        RETURN_CUDNN_FRONTEND_ERROR_IF(
            sdpa_attrs == nullptr, error_code_t::GRAPH_NOT_SUPPORTED, "No SDPA node found for OPENSOURCE engine");

        graph::Execution_plan_list::OssSdpaEngineContext ctx;

        // Q tensor
        auto q_it = sdpa_attrs->inputs.find(SDPA_attributes::input_names::Q);
        RETURN_CUDNN_FRONTEND_ERROR_IF(q_it == sdpa_attrs->inputs.end() || !q_it->second,
                                       error_code_t::GRAPH_NOT_SUPPORTED,
                                       "Q tensor not found in SDPA node");
        auto const &q = q_it->second;
        ctx.q_uid     = q->get_uid();
        ctx.q_stride  = q->get_stride();
        ctx.batch     = q->get_dim()[0];
        ctx.heads_q   = q->get_dim()[1];
        ctx.seq_q     = q->get_dim()[2];
        ctx.d         = q->get_dim()[3];

        // K tensor — CompositeSDPANode transposes K in-place (swaps dims[2]/dims[3]
        // and strides[2]/strides[3]) for the Q*K^T GEMM.  Detect and undo.
        auto k_it = sdpa_attrs->inputs.find(SDPA_attributes::input_names::K);
        RETURN_CUDNN_FRONTEND_ERROR_IF(k_it == sdpa_attrs->inputs.end() || !k_it->second,
                                       error_code_t::GRAPH_NOT_SUPPORTED,
                                       "K tensor not found in SDPA node");
        auto const &k         = k_it->second;
        auto const &k_dims    = k->get_dim();
        auto const &k_strides = k->get_stride();
        ctx.k_uid             = k->get_uid();
        ctx.heads_kv          = k_dims[1];
        if (k_dims[2] < k_dims[3]) {
            // dims currently (B, H_kv, D, S_kv) — swapped
            ctx.seq_kv   = k_dims[3];
            ctx.k_stride = {k_strides[0], k_strides[1], k_strides[3], k_strides[2]};
        } else {
            ctx.seq_kv   = k_dims[2];
            ctx.k_stride = k_strides;
        }

        // V tensor
        auto v_it = sdpa_attrs->inputs.find(SDPA_attributes::input_names::V);
        RETURN_CUDNN_FRONTEND_ERROR_IF(v_it == sdpa_attrs->inputs.end() || !v_it->second,
                                       error_code_t::GRAPH_NOT_SUPPORTED,
                                       "V tensor not found in SDPA node");
        ctx.v_uid    = v_it->second->get_uid();
        ctx.v_stride = v_it->second->get_stride();

        // O tensor
        auto o_it = sdpa_attrs->outputs.find(SDPA_attributes::output_names::O);
        RETURN_CUDNN_FRONTEND_ERROR_IF(o_it == sdpa_attrs->outputs.end() || !o_it->second,
                                       error_code_t::GRAPH_NOT_SUPPORTED,
                                       "O tensor not found in SDPA node");
        ctx.o_uid    = o_it->second->get_uid();
        ctx.o_stride = o_it->second->get_stride();

        // Max tensor (optional — only present if user called set_logit_max)
        auto max_it = sdpa_attrs->outputs.find(SDPA_attributes::output_names::Max);
        if (max_it != sdpa_attrs->outputs.end() && max_it->second) {
            ctx.max_uid    = max_it->second->get_uid();
            ctx.max_stride = max_it->second->get_stride();
        }

        // Sum_exp tensor (optional — only present if user called set_score_sum_exp)
        auto se_it = sdpa_attrs->outputs.find(SDPA_attributes::output_names::Sum_exp);
        if (se_it != sdpa_attrs->outputs.end() && se_it->second) {
            ctx.sum_exp_uid    = se_it->second->get_uid();
            ctx.sum_exp_stride = se_it->second->get_stride();
        }

        // Attention scale (use user-provided value if set, otherwise engine computes 1/sqrt(d))
        if (sdpa_attrs->attn_scale_value.has_value()) {
            ctx.attn_scale = sdpa_attrs->attn_scale_value.value();
        }

        RETURN_CUDNN_FRONTEND_ERROR_IF(ctx.q_uid == -1 || ctx.k_uid == -1 || ctx.v_uid == -1 || ctx.o_uid == -1,
                                       error_code_t::GRAPH_NOT_SUPPORTED,
                                       "Could not find Q/K/V/O tensors in SDPA node for OPENSOURCE engine");

        // Detect SM version and instantiate the appropriate OSS engine
        int oss_device_ordinal = 0;
        experimental::detail::cuda_get_device(&oss_device_ordinal);
        cudaDeviceProp oss_dev_prop;
        experimental::detail::cuda_get_device_properties(&oss_dev_prop, oss_device_ordinal);
        int oss_sm = oss_dev_prop.major * 10 + oss_dev_prop.minor;

        std::shared_ptr<experimental::IOssSdpaEngine> engine;
        if (oss_sm / 10 == 10) {
            engine = std::make_shared<experimental::Sm100SdpaPrefillEngine>();
        } else {
            engine = std::make_shared<experimental::Sm90SdpaPrefillEngine>();
        }
        plans.set_oss_sdpa_engine(engine);
        plans.set_oss_sdpa_engine_context(std::move(ctx));

        return {error_code_t::OK, ""};
    }

    // Register an OSS NVRTC engine for RmsNorm+SiLU by detecting the fusion pattern:
    //   RMSNormNode(X, SCALE) → Y → PointwiseNode(SWISH_FWD) → Z
    // Extracts tensor metadata and instantiates the appropriate arch-specific engine.
    error_t
    register_oss_rms_norm_silu_engine_() {
        // Scan sub_nodes for the RMSNorm → SiLU pattern
        Rmsnorm_attributes const *rmsnorm_attrs = nullptr;
        Pointwise_attributes const *swish_attrs = nullptr;
        std::shared_ptr<Tensor_attributes> swish_output;

        for (size_t i = 0; i + 1 < sub_nodes.size(); ++i) {
            auto *rmsnorm_node = dynamic_cast<RMSNormNode *>(sub_nodes[i].get());
            if (!rmsnorm_node) continue;

            auto *pointwise_node = dynamic_cast<PointwiseNode *>(sub_nodes[i + 1].get());
            if (!pointwise_node) continue;

            if (pointwise_node->attributes.get_mode() != PointwiseMode_t::SWISH_FWD) continue;

            // OSS engine only supports SiLU (Swish with beta=1.0).
            // Reject if beta is explicitly set to a non-1.0 value.
            auto beta = pointwise_node->attributes.get_swish_beta().value_or(1.0f);
            if (beta != 1.0f) continue;

            // Verify the RMSNorm output feeds into the pointwise input
            auto rmsnorm_y_it = rmsnorm_node->attributes.outputs.find(Rmsnorm_attributes::output_names::Y);
            if (rmsnorm_y_it == rmsnorm_node->attributes.outputs.end() || !rmsnorm_y_it->second) continue;

            auto swish_in0_it = pointwise_node->attributes.inputs.find(Pointwise_attributes::input_names::IN_0);
            if (swish_in0_it == pointwise_node->attributes.inputs.end() || !swish_in0_it->second) continue;

            // Check that RMSNorm output Y == SiLU input IN_0 (same tensor)
            if (rmsnorm_y_it->second->get_uid() != swish_in0_it->second->get_uid()) continue;

            // Pattern matched!
            rmsnorm_attrs = &rmsnorm_node->attributes;
            swish_attrs   = &pointwise_node->attributes;
            swish_output  = pointwise_node->attributes.outputs.at(Pointwise_attributes::output_names::OUT_0);
            break;
        }

        RETURN_CUDNN_FRONTEND_ERROR_IF(rmsnorm_attrs == nullptr || swish_attrs == nullptr,
                                       error_code_t::GRAPH_NOT_SUPPORTED,
                                       "No RMSNorm → SiLU fusion pattern found for OPENSOURCE engine");

        // Only inference is supported (no mean/inv_variance output for training)
        RETURN_CUDNN_FRONTEND_ERROR_IF(rmsnorm_attrs->forward_phase != NormFwdPhase_t::INFERENCE,
                                       error_code_t::GRAPH_NOT_SUPPORTED,
                                       "OSS RmsNorm+SiLU engine only supports INFERENCE phase");

        // Extract tensor metadata
        graph::Execution_plan_list::OssRmsNormSiluContext ctx;

        // X input tensor
        auto x_it = rmsnorm_attrs->inputs.find(Rmsnorm_attributes::input_names::X);
        RETURN_CUDNN_FRONTEND_ERROR_IF(x_it == rmsnorm_attrs->inputs.end() || !x_it->second,
                                       error_code_t::GRAPH_NOT_SUPPORTED,
                                       "X tensor not found in RMSNorm node");

        // Input must be bf16 (the kernel always uses ITYPE = nv_bfloat16)
        auto x_dtype = x_it->second->get_data_type();
        RETURN_CUDNN_FRONTEND_ERROR_IF(x_dtype != DataType_t::BFLOAT16,
                                       error_code_t::GRAPH_NOT_SUPPORTED,
                                       "OSS RmsNorm+SiLU engine requires BFLOAT16 input");

        ctx.x_uid  = x_it->second->get_uid();
        auto x_dim = x_it->second->get_dim();
        // X shape in NCHW: [N, C, H, W] where norm operates per-row (N) across columns (C*H*W)
        // num_tokens = N (first dim), C = product of remaining dims (C*H*W)
        RETURN_CUDNN_FRONTEND_ERROR_IF(
            x_dim.size() < 2, error_code_t::GRAPH_NOT_SUPPORTED, "X tensor must have at least 2 dimensions");
        ctx.num_tokens = x_dim[0];
        ctx.C          = 1;
        for (size_t d = 1; d < x_dim.size(); ++d) {
            ctx.C *= x_dim[d];
        }
        RETURN_CUDNN_FRONTEND_ERROR_IF(
            ctx.num_tokens <= 0 || ctx.C <= 0,
            error_code_t::GRAPH_NOT_SUPPORTED,
            "Invalid problem dimensions: num_tokens=" + std::to_string(ctx.num_tokens) + " C=" + std::to_string(ctx.C));

        // SCALE (gamma) tensor — must be bf16 (kernel uses WTYPE = nv_bfloat16)
        auto scale_it = rmsnorm_attrs->inputs.find(Rmsnorm_attributes::input_names::SCALE);
        RETURN_CUDNN_FRONTEND_ERROR_IF(scale_it == rmsnorm_attrs->inputs.end() || !scale_it->second,
                                       error_code_t::GRAPH_NOT_SUPPORTED,
                                       "SCALE tensor not found in RMSNorm node");
        RETURN_CUDNN_FRONTEND_ERROR_IF(scale_it->second->get_data_type() != DataType_t::BFLOAT16,
                                       error_code_t::GRAPH_NOT_SUPPORTED,
                                       "OSS RmsNorm+SiLU engine requires BFLOAT16 scale weights");
        ctx.scale_uid = scale_it->second->get_uid();

        // BIAS (beta) tensor — optional
        auto bias_it = rmsnorm_attrs->inputs.find(Rmsnorm_attributes::input_names::BIAS);
        if (bias_it != rmsnorm_attrs->inputs.end() && bias_it->second) {
            ctx.bias_uid = bias_it->second->get_uid();
        }

        // EPSILON tensor
        auto eps_it = rmsnorm_attrs->inputs.find(Rmsnorm_attributes::input_names::EPSILON);
        RETURN_CUDNN_FRONTEND_ERROR_IF(eps_it == rmsnorm_attrs->inputs.end() || !eps_it->second,
                                       error_code_t::GRAPH_NOT_SUPPORTED,
                                       "EPSILON tensor not found in RMSNorm node");
        ctx.epsilon_uid = eps_it->second->get_uid();

        // Output Z tensor (output of SiLU, not RMSNorm)
        RETURN_CUDNN_FRONTEND_ERROR_IF(
            !swish_output, error_code_t::GRAPH_NOT_SUPPORTED, "Output tensor not found in SiLU node");
        ctx.y_uid = swish_output->get_uid();

        // Determine output dtype from the output tensor
        auto out_dtype = swish_output->get_data_type();
        if (out_dtype == DataType_t::BFLOAT16) {
            ctx.output_dtype = experimental::RmsNormSiluDtype::BF16;
        } else if (out_dtype == DataType_t::FP8_E4M3) {
            ctx.output_dtype = experimental::RmsNormSiluDtype::FP8;
        } else if (out_dtype == DataType_t::FP4_E2M1) {
            ctx.output_dtype = experimental::RmsNormSiluDtype::NVFP4;
        } else {
            return {error_code_t::GRAPH_NOT_SUPPORTED,
                    "OSS RmsNorm+SiLU engine supports BFLOAT16, FP8_E4M3, or FP4_E2M1 output"};
        }

        // Detect SM version and instantiate the appropriate OSS engine
        int oss_device_ordinal = 0;
        experimental::detail::cuda_get_device(&oss_device_ordinal);
        cudaDeviceProp oss_dev_prop;
        experimental::detail::cuda_get_device_properties(&oss_dev_prop, oss_device_ordinal);
        int oss_sm = oss_dev_prop.major * 10 + oss_dev_prop.minor;

        std::shared_ptr<experimental::IOssNormEngine> engine;
        if (oss_sm >= 80) {
            engine = std::make_shared<experimental::Sm100RmsNormSiluEngine>();
        } else {
            return {error_code_t::GRAPH_NOT_SUPPORTED,
                    "RmsNorm+SiLU OSS engine requires SM80+, got SM" + std::to_string(oss_sm)};
        }

        plans.set_oss_rms_norm_silu_engine(engine);
        plans.set_oss_rms_norm_silu_context(std::move(ctx));

        return {error_code_t::OK, ""};
    }

   public:
    Graph() : INode(detail::Context{}) {
        static std::atomic<uint64_t> next_graph_uid{1};
        graph_uid = next_graph_uid.fetch_add(1, std::memory_order_relaxed);
    }

    error_t
    update_cuda_graph(cudnnHandle_t handle,
                      std::unordered_map<std::shared_ptr<Tensor_attributes>, void *> &tensor_to_pointer_map,
                      void *workspace,
                      cudaGraph_t cudnn_cuda_graph) {
        // First get all the uids from the map
        std::unordered_map<Tensor_attributes::uid_t, void *> tensor_uid_to_pointer_map;
        tensor_uid_to_pointer_map.reserve(tensor_to_pointer_map.size());
        for (auto const &[tensor, pointer] : tensor_to_pointer_map) {
            tensor_uid_to_pointer_map.emplace(tensor->get_uid(), pointer);
        }

        return update_cuda_graph(handle, tensor_uid_to_pointer_map, workspace, cudnn_cuda_graph);
    }

    error_t
    update_cuda_graph(cudnnHandle_t handle,
                      std::unordered_map<Tensor_attributes::uid_t, void *> &uid_to_device_ptrs,
                      void *workspace,
                      cudaGraph_t cudnn_cuda_graph) {
        // Initializes this cudnn graph
        RETURN_CUDNN_FRONTEND_ERROR_IF(
            cudnn_cuda_graph == nullptr, error_code_t::INVALID_VALUE, "cudnn_cuda_graph should not be a nullptr");

        size_t num_root_nodes;
        _CUDNN_CHECK_CUDA_ERROR(detail::cuda_graph_get_root_nodes(cudnn_cuda_graph, nullptr, &num_root_nodes));
        RETURN_CUDNN_FRONTEND_ERROR_IF(
            num_root_nodes != 1, error_code_t::INVALID_VALUE, "cudnn_cuda_graph should have exactly 1 root node.");

        cudaGraphNode_t current_node = nullptr;
        _CUDNN_CHECK_CUDA_ERROR(detail::cuda_graph_get_root_nodes(cudnn_cuda_graph, &current_node, &num_root_nodes));

        ///////////////////////////////////////
        //// PASS BY VALUE TENSOR HANDLING ////
        ///////////////////////////////////////
        // Add pass_by_value data pointers to uid_to_pointer map.
        // Using cached values to avoid repeated tree traversal overhead.
        // cuda graph will keep a copy of the kernel parameters, meaning that at the time of
        // launching the cuda_graph executable, cached values being deallocated does not affect these cpu values.
        // No cuda graph nodes are required for handling fe owned pass by value tensors.
        CHECK_CUDNN_FRONTEND_ERROR(
            extend_tensor_map_with_pass_by_value_tensors_(uid_to_device_ptrs, cached_pass_by_value));

        ////////////////////////////
        //// WORKSPACE HANDLING ////
        ////////////////////////////
        // Using cached workspace modifications to avoid repeated tree traversal.
        for (auto const &[uid, data] : cached_workspace_modifications) {
            const auto &[operation_type, offset, vec_data] = data;
            uid_to_device_ptrs[uid]                        = static_cast<char *>(workspace) + offset;

            // 0 means memcpy
            if (operation_type == 0) {
                _CUDNN_CHECK_CUDA_ERROR(
                    detail::cuda_graph_add_memcpy_node_set_params_1D(current_node,
                                                                     static_cast<char *>(workspace) + offset,
                                                                     vec_data.data(),
                                                                     vec_data.size() * sizeof(float),
                                                                     cudaMemcpyHostToDevice));
            }
            // 1 means memset
            else if (operation_type == 1) {
                // offset from workspace
                void *device_ptr    = static_cast<char *>(workspace) + offset;
                int64_t memset_size = static_cast<int64_t>(vec_data[0]);

                cudaMemsetParams params;
                params.dst         = device_ptr;
                params.elementSize = sizeof(char);
                params.value       = 0x0;
                params.width       = memset_size;
                params.height      = 1;  // 1D memset currently
                params.pitch       = 0;  // unused

                _CUDNN_CHECK_CUDA_ERROR(detail::cuda_graph_add_memset_node_set_params(current_node, &params));
            }
            // Other values do not correspond to CUDA graph nodes
            else {
                continue;
            }

            size_t num_dependent_nodes;
            _CUDNN_CHECK_CUDA_ERROR(
                detail::cuda_graph_node_get_dependent_nodes(current_node, nullptr, &num_dependent_nodes));
            RETURN_CUDNN_FRONTEND_ERROR_IF(
                num_dependent_nodes != 1,
                error_code_t::INVALID_VALUE,
                "Each node of cudnn_cuda_graph before the backend graph node should have exactly 1 dependent node.");
            _CUDNN_CHECK_CUDA_ERROR(
                detail::cuda_graph_node_get_dependent_nodes(current_node, &current_node, &num_dependent_nodes));
        }

        // Make sure device pointer is provided for all uids expected for this plan
        std::vector<void *> device_ptrs;
        std::vector<uid_t> uids;

        device_ptrs.reserve(variant_pack_uids.size());
        uids.reserve(variant_pack_uids.size());

        for (auto const &uid : variant_pack_uids) {
            auto search = uid_to_device_ptrs.find(uid);
            RETURN_CUDNN_FRONTEND_ERROR_IF(search == uid_to_device_ptrs.end(),
                                           error_code_t::INVALID_VARIANT_PACK,
                                           "Uid " + std::to_string(uid) + " does not exist in variant pack.");
            device_ptrs.push_back(search->second);
            uids.push_back(uid);
        }

        ///////////////////
        //// BE GRAPH ////
        ///////////////////
        cudaGraph_t backend_cuda_graph;
        _CUDNN_CHECK_CUDA_ERROR(detail::cuda_graph_child_graph_node_get_graph(current_node, &backend_cuda_graph));

        detail::backend_descriptor variant_pack_descriptor(CUDNN_BACKEND_VARIANT_PACK_DESCRIPTOR);
        RETURN_CUDNN_FRONTEND_ERROR_IF(variant_pack_descriptor.get_status() != CUDNN_STATUS_SUCCESS,
                                       error_code_t::CUDNN_BACKEND_API_FAILED,
                                       "Failed to create variant pack's backend descriptor.");

        // offset workspace by the already used fe graph workspace
        // this is where cudnn backend can start using workspace for its execution plans
        void *cudnn_workspace = static_cast<char *>(workspace) + fe_workspace_size;
        CHECK_CUDNN_FRONTEND_ERROR(create_variant_pack(variant_pack_descriptor, device_ptrs, uids, cudnn_workspace));

        int64_t candidate = plans.candidate;
        CHECK_CUDNN_FRONTEND_ERROR(plans.is_plan_index_executable(candidate));
        _CUDNN_CHECK_CUDNN_ERROR(detail::update_cuda_graph(handle,
                                                           plans.execution_plans[candidate]->get_raw_desc(),
                                                           variant_pack_descriptor.get_ptr(),
                                                           backend_cuda_graph));

        // There should be nothing after the backend graph
        size_t num_dependent_nodes;
        _CUDNN_CHECK_CUDA_ERROR(
            detail::cuda_graph_node_get_dependent_nodes(current_node, nullptr, &num_dependent_nodes));
        RETURN_CUDNN_FRONTEND_ERROR_IF(num_dependent_nodes != 0,
                                       error_code_t::INVALID_VALUE,
                                       "cudnn_cuda_graph should have no graph nodes after the backend graph node.");

        return {error_code_t::OK, ""};
    }

    error_t
    populate_cuda_graph(cudnnHandle_t handle,
                        std::unordered_map<std::shared_ptr<Tensor_attributes>, void *> &tensor_to_pointer_map,
                        void *workspace,
                        cudaGraph_t cudnn_cuda_graph) {
        // First get all the uids from the map
        std::unordered_map<Tensor_attributes::uid_t, void *> tensor_uid_to_pointer_map;
        tensor_uid_to_pointer_map.reserve(tensor_to_pointer_map.size());
        for (auto const &[tensor, pointer] : tensor_to_pointer_map) {
            tensor_uid_to_pointer_map.emplace(tensor->get_uid(), pointer);
        }

        return populate_cuda_graph(handle, tensor_uid_to_pointer_map, workspace, cudnn_cuda_graph);
    }

    error_t
    populate_cuda_graph(cudnnHandle_t handle,
                        std::unordered_map<Tensor_attributes::uid_t, void *> &uid_to_device_ptrs,
                        void *workspace,
                        cudaGraph_t cudnn_cuda_graph) {
        // Check if the cuda graph is empty
        size_t numNodes = 0;
        _CUDNN_CHECK_CUDA_ERROR(detail::cuda_graph_get_nodes(cudnn_cuda_graph, nullptr, &numNodes));
        RETURN_CUDNN_FRONTEND_ERROR_IF(numNodes != 0,
                                       error_code_t::INVALID_VALUE,
                                       "cuda graph provided to populate is not empty. cuDNN requires it to be empty "
                                       "for the corresponding update APIs to work correctly.");

        // This function makes linear cuda graphs. And that makes it easy to walk
        // the graph when updating it.
        // So just keeping track of the last node in the cuda graph is sufficient.
        cudaGraphNode_t last_node = nullptr;

        ///////////////////////////////////////
        //// PASS BY VALUE TENSOR HANDLING ////
        ///////////////////////////////////////
        // Add pass_by_value data pointers to uid_to_pointer map.
        // Using cached values to avoid repeated tree traversal overhead.
        // cuda graph will keep a copy of the kernel parameters, meaning that at the time of
        // launching the cuda_graph executable, cached values being deallocated does not affect these cpu values.
        // No cuda graph nodes are required for handling fe owned pass by value tensors.
        CHECK_CUDNN_FRONTEND_ERROR(
            extend_tensor_map_with_pass_by_value_tensors_(uid_to_device_ptrs, cached_pass_by_value));

        /////////////////////////////////
        //// WORKSPACE HANDLING ////
        /////////////////////////////////
        // Using cached workspace modifications to avoid repeated tree traversal.
        for (auto const &[uid, data] : cached_workspace_modifications) {
            const auto &[operation_type, offset, vec_data] = data;
            uid_to_device_ptrs[uid]                        = static_cast<char *>(workspace) + offset;

            cudaGraphNode_t node = nullptr;

            // 0 means memcpy
            if (operation_type == 0) {
                _CUDNN_CHECK_CUDA_ERROR(detail::cuda_graph_add_memcpy_node_1D(&node,
                                                                              cudnn_cuda_graph,
                                                                              &last_node,
                                                                              last_node != nullptr,
                                                                              static_cast<char *>(workspace) + offset,
                                                                              vec_data.data(),
                                                                              vec_data.size() * sizeof(float),
                                                                              cudaMemcpyHostToDevice));
            }
            // 1 means memset
            else if (operation_type == 1) {
                // offset from workspace
                void *device_ptr    = static_cast<char *>(workspace) + offset;
                int64_t memset_size = static_cast<int64_t>(vec_data[0]);

                cudaMemsetParams params;
                params.dst         = device_ptr;
                params.elementSize = sizeof(char);
                params.value       = 0x0;
                params.width       = memset_size;
                params.height      = 1;  // 1D memset currently
                params.pitch       = 0;  // unused

                _CUDNN_CHECK_CUDA_ERROR(detail::cuda_graph_add_memset_node(
                    &node, cudnn_cuda_graph, &last_node, last_node != nullptr, &params));
            }
            // Other values do not correspond to CUDA graph nodes
            else {
                continue;
            }

            last_node = node;
        }

        //////////////
        // BE graph //
        //////////////

        // Get the BE's cuda graph

        // Make sure device pointer is provided for all uids expected for this plan
        std::vector<void *> device_ptrs;
        device_ptrs.reserve(variant_pack_uids.size());
        std::vector<uid_t> uids;
        uids.reserve(variant_pack_uids.size());
        for (auto const &uid : variant_pack_uids) {
            auto search = uid_to_device_ptrs.find(uid);
            RETURN_CUDNN_FRONTEND_ERROR_IF(search == uid_to_device_ptrs.end(),
                                           error_code_t::INVALID_VARIANT_PACK,
                                           "Uid " + std::to_string(uid) + " does not exist in variant pack.");
            device_ptrs.push_back(search->second);
            uids.push_back(uid);
        }

        // Create the variant pack to pass to backend
        detail::backend_descriptor variant_pack_descriptor(CUDNN_BACKEND_VARIANT_PACK_DESCRIPTOR);
        RETURN_CUDNN_FRONTEND_ERROR_IF(variant_pack_descriptor.get_status() != CUDNN_STATUS_SUCCESS,
                                       error_code_t::CUDNN_BACKEND_API_FAILED,
                                       "Failed to create variant pack's backend descriptor.");

        // offset workspace by the already used fe graph workspace
        // this is where cudnn backend can start using workspace for its execution plans
        void *cudnn_workspace = static_cast<char *>(workspace) + fe_workspace_size;
        CHECK_CUDNN_FRONTEND_ERROR(create_variant_pack(variant_pack_descriptor, device_ptrs, uids, cudnn_workspace));

        // Get the plan candidate. It only makes to sense to make cuda graph after execution plan has been built.
        // And in that case the candidate would have been set.
        int64_t candidate = plans.candidate;
        CHECK_CUDNN_FRONTEND_ERROR(plans.is_plan_index_executable(candidate));

        // Finally get the backend cuda graph.
        cudaGraph_t backend_cuda_graph;
        // Initialize the cudnn cuda graph.
        // The responsibility to destroy is on the user.
        _CUDNN_CHECK_CUDA_ERROR(detail::cuda_graph_create(&backend_cuda_graph, 0));

        _CUDNN_CHECK_CUDNN_ERROR(detail::populate_cuda_graph(handle,
                                                             plans.execution_plans[candidate]->get_raw_desc(),
                                                             variant_pack_descriptor.get_ptr(),
                                                             backend_cuda_graph));

        // Clone BE graph into a graph_node
        // This same call also places the newly created into FE's graph
        // TODO: BE graph is at the end, so put in appropriate dependencies
        cudaGraphNode_t backend_cuda_graph_node;
        detail::cuda_graph_add_child_graph_node(
            &backend_cuda_graph_node, cudnn_cuda_graph, &last_node, last_node != nullptr, backend_cuda_graph);

        // Destroy the BE graph as it now has been cloned into a node
        // It was initialized by internals of backend, but the responsibility to destroy it is on FE.
        _CUDNN_CHECK_CUDA_ERROR(detail::cuda_graph_destroy(backend_cuda_graph));

        return {error_code_t::OK, ""};
    }

    error_t
    validate() {
        CUDNN_FE_LOG_BANNER("  VALIDATING GRAPH  ");
        CUDNN_FE_LOG(*this << std::endl;);

        // First validate all inputs that the user set.
        for (auto const &input : full_graph_inputs) {
            CHECK_CUDNN_FRONTEND_ERROR(input->validate());
        }

        // Validate the nodes, which in turn also infers missing tensor attributes.
        CHECK_CUDNN_FRONTEND_ERROR(validate_subtree());
        // Validate all outputs, which should now have everything set to be lowered to backend.
        for (auto const &output : full_graph_outputs) {
            CHECK_CUDNN_FRONTEND_ERROR(output->validate());
        }

        // Get all the pre assigned uids
        CHECK_CUDNN_FRONTEND_ERROR(get_pre_assigned_uids(used_uids));
        // Clear state
        used_uids.clear();

        CUDNN_FE_LOG_BANNER("  VALIDATED ALL OK  ");

        return {error_code_t::OK, ""};
    }

    // overload for deviceless AoT compilation
    error_t
    build_operation_graph() {
        CUDNN_FE_LOG_BANNER("  BUILD OP GRAPH WITHOUT HANDLE  ");

        if (device_properties == nullptr) {
            return {error_code_t::ATTRIBUTE_NOT_SET, "Device properties are not set."};
        }
        CUDNN_FE_LOG_BANNER("  BUILT OP GRAPH WITHOUT HANDLE  ");
        return build_operation_graph(nullptr);
    }

    error_t
    build_operation_graph(cudnnHandle_t handle) {
        CUDNN_FE_LOG_BANNER("  BUILD OP GRAPH  ");

        CUDNN_FE_LOG_BANNER("  1/4 INFER PROPERTIES OF NODES  ");

        // expand composite nodes
        CHECK_CUDNN_FRONTEND_ERROR(expand_subtree());

        // Get all the pre assigned uids
        CHECK_CUDNN_FRONTEND_ERROR(get_pre_assigned_uids(used_uids));

        CUDNN_FE_LOG_BANNER("  2/4 CREATE TENSORS  ");

        Tensor_attributes::uid_t start_uid = 1;
        CHECK_CUDNN_FRONTEND_ERROR(create_cudnn_tensors_subtree(uid_to_tensors, start_uid, used_uids));
        tensors_to_dump.clear();
        CHECK_CUDNN_FRONTEND_ERROR(collect_tensors_to_dump_subtree(tensors_to_dump));

        CUDNN_FE_LOG_BANNER("  3/4 CREATE OPERATIONS  ");
        // INode keeps track of all uids that an operation graph uses.
        // This helps to return errors to user during execution, without relying on backend to do so.
        // Also, as uid in a variant pack have to be unique, keep a set of them.
        CHECK_CUDNN_FRONTEND_ERROR(
            create_cudnn_operations(variant_pack_uids, operations, raw_operations, uid_to_tensors));

        // Collect variant pack modifiers when lowering to backend.
        // The collected map is used everytime when execute is called.
        CHECK_CUDNN_FRONTEND_ERROR(collect_variant_pack_replacements_subtree(variant_pack_replacements));

        fe_workspace_size = get_fe_workspace_size_subtree();

        // Cache pass_by_value tensors and workspace modifications for fast execution.
        // These are collected once here and reused in every execute() call to avoid
        // repeated tree traversal and map allocation overhead.
        CHECK_CUDNN_FRONTEND_ERROR(collect_pass_by_value_tensors_subtree(cached_pass_by_value));
        {
            int64_t temp_offset = 0;
            CHECK_CUDNN_FRONTEND_ERROR(
                collect_tensors_in_workspace_subtree(cached_workspace_modifications, temp_offset));
        }

        CUDNN_FE_LOG_BANNER("  4/4 LOWERING TO BACKEND OPERATION GRAPH  ");

        // The method here fuses all operations. There will be 1 operation graph in total.
        CHECK_CUDNN_FRONTEND_ERROR(create_cudnn_operation_graph(handle));

        if (context.get_dynamic_shape_enabled() && kernel_cache && !kernel_cache->is_finalized()) {
            CUDNN_FE_LOG_BANNER("  BUILD KERNEL CACHE  ");
            CHECK_CUDNN_FRONTEND_ERROR(kernel_cache->build(operation_graph->get_raw_desc()));
        }

        CUDNN_FE_LOG_BANNER("  BUILD OP GRAPH ALL OK === ");

        return {error_code_t::OK, ""};
    }

    error_t
    get_plan_name(std::string &name) const {
        return get_plan_name_at_index(plans.candidate, name);
    }

    error_t
    get_plan_name_at_index(int64_t plan_index, std::string &name) const {
        auto ret_val = plans.get_name_at_index(plan_index, name);
        CUDNN_FE_LOG_LABEL_ENDL("INFO: get_plan_name_at_index(" << plan_index << ") is " + name);
        return ret_val;
    }

    error_t
    get_workspace_size(int64_t &cudnn_workspace_size) const {
        return get_workspace_size_plan_at_index(plans.candidate, cudnn_workspace_size);
    }

    error_t
    get_workspace_size(cudnnHandle_t handle,
                       int64_t &cudnn_workspace_size,
                       std::vector<int64_t> const &override_uids,
                       std::vector<std::vector<int64_t>> const &override_shapes,
                       std::vector<std::vector<int64_t>> const &override_strides) const {
        return get_workspace_size_plan_at_index(
            handle, plans.candidate, cudnn_workspace_size, override_uids, override_shapes, override_strides);
    }

    error_t
    get_workspace_size_plan_at_index(int64_t plan_index, int64_t &cudnn_workspace_size) const {
        // OSS SDPA engine workspace: 16 bytes for tile_id_counter
        if (plan_index == graph::Execution_plan_list::OSS_SDPA_ENGINE_CANDIDATE) {
            cudnn_workspace_size = fe_workspace_size + experimental::Sm90SdpaPrefillEngine::get_workspace_size();
            CUDNN_FE_LOG_LABEL_ENDL("INFO: get_workspace_size() is " << cudnn_workspace_size << " (OSS SDPA engine)");
            return {error_code_t::OK, ""};
        }

        // OSS RmsNorm+SiLU engine workspace
        if (plan_index == graph::Execution_plan_list::OSS_RMS_NORM_SILU_ENGINE_CANDIDATE) {
            cudnn_workspace_size = fe_workspace_size + plans.get_oss_rms_norm_silu_workspace_size();
            CUDNN_FE_LOG_LABEL_ENDL("INFO: get_workspace_size() is " << cudnn_workspace_size
                                                                     << " (OSS RmsNorm+SiLU engine)");
            return {error_code_t::OK, ""};
        }

        // There are two workspaces:
        // - cudnn execution plan workspace
        // - FE node workspace (example: alibiSlope for fmha)
        int64_t cudnn_ws = 0;
        CHECK_CUDNN_FRONTEND_ERROR(get_cudnn_workspace_size_node(plan_index, cudnn_ws));
        cudnn_workspace_size = cudnn_ws + fe_workspace_size;
        CUDNN_FE_LOG_LABEL_ENDL("INFO: get_workspace_size() is " << cudnn_workspace_size);
        return {error_code_t::OK, ""};
    }

    error_t
    get_workspace_size_plan_at_index(cudnnHandle_t handle,
                                     int64_t plan_index,
                                     int64_t &cudnn_workspace_size,
                                     std::vector<int64_t> const &override_uids,
                                     std::vector<std::vector<int64_t>> const &override_shapes,
                                     std::vector<std::vector<int64_t>> const &override_strides) const {
        RETURN_CUDNN_FRONTEND_ERROR_IF(override_uids.size() != override_shapes.size(),
                                       error_code_t::INVALID_VALUE,
                                       "override_uids and override_shapes must have the same size.");
        RETURN_CUDNN_FRONTEND_ERROR_IF(override_uids.size() != override_strides.size(),
                                       error_code_t::INVALID_VALUE,
                                       "override_uids and override_strides must have the same size.");

        if (override_uids.empty()) {
            return get_workspace_size_plan_at_index(plan_index, cudnn_workspace_size);
        }

        // OSS engines are not backed by cuDNN execution plans, so their workspace stays frontend-owned.
        if (plan_index == graph::Execution_plan_list::OSS_SDPA_ENGINE_CANDIDATE ||
            plan_index == graph::Execution_plan_list::OSS_RMS_NORM_SILU_ENGINE_CANDIDATE) {
            return get_workspace_size_plan_at_index(plan_index, cudnn_workspace_size);
        }

        auto cudnn_ver_error = error_t{error_code_t::GRAPH_NOT_SUPPORTED,
                                       "Runtime workspace query with override shapes requires cuDNN v9.23.0"};
        NV_CUDNN_FE_DYNAMIC_CHECK_CUDNN_BACKEND_VERSION(92300, cudnn_ver_error);

#if (CUDNN_VERSION < 92300) || (CUDNN_VERSION >= 99900)
        return cudnn_ver_error;
#endif

        CHECK_CUDNN_FRONTEND_ERROR(plans.is_plan_index_executable(plan_index));

        detail::backend_descriptor variant_pack_descriptor(CUDNN_BACKEND_VARIANT_PACK_DESCRIPTOR);
        RETURN_CUDNN_FRONTEND_ERROR_IF(variant_pack_descriptor.get_status() != CUDNN_STATUS_SUCCESS,
                                       error_code_t::CUDNN_BACKEND_API_FAILED,
                                       "Failed to create variant pack's backend descriptor.");

#if (CUDNN_VERSION >= 92100)
        _CUDNN_CHECK_CUDNN_ERROR(detail::set_attribute(variant_pack_descriptor.get_ptr(),
                                                       CUDNN_ATTR_VARIANT_PACK_OVERRIDE_UNIQUE_IDS,
                                                       CUDNN_TYPE_INT64,
                                                       override_uids.size(),
                                                       override_uids.data()));

        _CUDNN_CHECK_CUDNN_ERROR(detail::set_attribute(variant_pack_descriptor.get_ptr(),
                                                       CUDNN_ATTR_VARIANT_PACK_OVERRIDE_SHAPES,
                                                       CUDNN_TYPE_VOID_PTR,
                                                       1,
                                                       (void *)&override_shapes));

        _CUDNN_CHECK_CUDNN_ERROR(detail::set_attribute(variant_pack_descriptor.get_ptr(),
                                                       CUDNN_ATTR_VARIANT_PACK_OVERRIDE_STRIDES,
                                                       CUDNN_TYPE_VOID_PTR,
                                                       1,
                                                       (void *)&override_strides));
#else
        CUDNN_FRONTEND_UNUSED(override_uids);
        CUDNN_FRONTEND_UNUSED(override_shapes);
        CUDNN_FRONTEND_UNUSED(override_strides);
#endif

        _CUDNN_CHECK_CUDNN_ERROR(detail::finalize(variant_pack_descriptor.get_ptr()));

        size_t cudnn_ws = 0;
        _CUDNN_CHECK_CUDNN_ERROR(detail::get_execution_plan_workspace_size(
            handle, plans.execution_plans[plan_index]->get_raw_desc(), variant_pack_descriptor.get_ptr(), &cudnn_ws));
        cudnn_workspace_size = cudnn_ws + fe_workspace_size;
        CUDNN_FE_LOG_LABEL_ENDL("INFO: get_workspace_size() is " << cudnn_workspace_size
                                                                 << " (runtime override shape)");
        return {error_code_t::OK, ""};
    }

    int64_t
    get_workspace_size() const {
        return get_workspace_size_plan_at_index(plans.candidate);
    }

    int64_t
    get_workspace_size(cudnnHandle_t handle,
                       std::vector<int64_t> const &override_uids,
                       std::vector<std::vector<int64_t>> const &override_shapes,
                       std::vector<std::vector<int64_t>> const &override_strides) const {
        int64_t cudnn_workspace = 0;
        auto status = get_workspace_size(handle, cudnn_workspace, override_uids, override_shapes, override_strides);
        if (status.is_bad()) {
            CUDNN_FE_LOG_LABEL_ENDL("ERROR: Querying workspace failed.");
        }
        return cudnn_workspace;
    }

    int64_t
    get_workspace_size_plan_at_index(int64_t plan_index) const {
        int64_t cudnn_workspace = 0;
        auto status             = get_workspace_size_plan_at_index(plan_index, cudnn_workspace);
        if (status.is_bad()) {
            CUDNN_FE_LOG_LABEL_ENDL("ERROR: Querying workspace failed.");
        }
        return cudnn_workspace;
    }

    int64_t
    get_workspace_size_plan_at_index(cudnnHandle_t handle,
                                     int64_t plan_index,
                                     std::vector<int64_t> const &override_uids,
                                     std::vector<std::vector<int64_t>> const &override_shapes,
                                     std::vector<std::vector<int64_t>> const &override_strides) const {
        int64_t cudnn_workspace = 0;
        auto status             = get_workspace_size_plan_at_index(
            handle, plan_index, cudnn_workspace, override_uids, override_shapes, override_strides);
        if (status.is_bad()) {
            CUDNN_FE_LOG_LABEL_ENDL("ERROR: Querying workspace failed.");
        }
        return cudnn_workspace;
    }

    int64_t
    get_autotune_workspace_size() const {
        // There are two workspaces:
        // - cudnn execution plan workspace
        // - FE node workspace (example: alibiSlope for fmha)
        return fe_workspace_size + get_max_cudnn_workspace_size();
    }

    error_t
    autotune(cudnnHandle_t handle,
             std::unordered_map<int64_t, void *> &tensor_uid_to_pointer_map,
             void *workspace,
             void *user_impl = nullptr) {
        (void)user_impl;  // reserved for future use

        const int maxIterCount = 100;
        const float threshold  = 0.95f;

        auto plan_cmp = [](std::shared_ptr<ExecutionPlan> a, std::shared_ptr<ExecutionPlan> b) {
            return a->getExecutionTime() < b->getExecutionTime();
        };
        std::multiset<std::shared_ptr<ExecutionPlan>, decltype(plan_cmp)> timed_plans(plan_cmp);

        cudaEvent_t start, stop;
        detail::cuda_event_create(&start);
        detail::cuda_event_create(&stop);
        detail::cuda_device_synchronize();

        cudaStream_t stream = nullptr;
        detail::get_stream(handle, &stream);

        uint64_t successful_plan_count = 0;
        for (int64_t i = 0; i < static_cast<int64_t>(plans.execution_plans.size()); i++) {
            if (plans.execution_plans[i] == nullptr) continue;

            // Warm-up run
            auto warmup_status = execute_plan_at_index(handle, tensor_uid_to_pointer_map, workspace, i);
            if (warmup_status.is_bad()) {
                CUDNN_FE_LOG_LABEL_ENDL("WARN: Plan " << i << " failed warmup, skipping.");
                continue;
            }
            successful_plan_count++;
            detail::cuda_device_synchronize();

            float min_time_ms = std::numeric_limits<float>::max();
            for (int iter = 0; iter < maxIterCount; iter++) {
                detail::cuda_event_record(start, stream);
                auto iter_status = execute_plan_at_index(handle, tensor_uid_to_pointer_map, workspace, i);
                detail::cuda_event_record(stop, stream);
                detail::cuda_event_synchronize(stop);

                if (iter_status.is_bad()) {
                    CUDNN_FE_LOG_LABEL_ENDL("WARN: Plan " << i << " failed at iter " << iter << ", skipping time.");
                    continue;
                }

                float time_ms = 0.0f;
                detail::cuda_event_elapsed_time(&time_ms, start, stop);
                float new_min = std::min(min_time_ms, time_ms);
                if (time_ms / min_time_ms < threshold) {
                    min_time_ms = new_min;
                } else {
                    break;
                }
            }

            CUDNN_FE_LOG_LABEL_ENDL("Plan " << plans.execution_plans[i]->getTag() << " took " << std::setw(10)
                                            << min_time_ms);
            plans.execution_plans[i]->setExecutionTime(min_time_ms);
            timed_plans.insert(plans.execution_plans[i]);
        }

        // Re-order plans by measured time, winner at index 0
        plans.execution_plans.clear();
        for (auto sorted_plan : timed_plans) {
            plans.execution_plans.push_back(sorted_plan);
        }
        plans.candidate = 0;

        // Re-prepare OSS slot indices to match the new plan ordering
        apply_oss_slot_indices_to_plans();

        detail::cuda_event_destroy(start);
        detail::cuda_event_destroy(stop);

        CUDNN_FE_LOG_LABEL_ENDL("Autotuned " << successful_plan_count << " plans.");
        return {error_code_t::OK, ""};
    }

    error_t
    autotune(cudnnHandle_t handle,
             std::unordered_map<std::shared_ptr<Tensor_attributes>, void *> &tensor_to_pointer_map,
             void *workspace,
             void *user_impl = nullptr) {
        std::unordered_map<int64_t, void *> uid_map;
        for (auto const &[tensor, pointer] : tensor_to_pointer_map) {
            uid_map.emplace(tensor->get_uid(), pointer);
        }
        return autotune(handle, uid_map, workspace, user_impl);
    }

    // -----------------------------------------------------------------------
    // Execute overloads.
    // All roads lead to execute_plan_at_index(handle, sorted_ptrs, n, ws, plan_index).
    // -----------------------------------------------------------------------

    // --- Convenience wrappers: convert key types, use default plan index ---

    error_t
    execute(cudnnHandle_t handle,
            std::unordered_map<std::shared_ptr<Tensor_attributes>, void *> &tensor_to_pointer_map,
            void *workspace) const {
        std::unordered_map<int64_t, void *> uid_map;
        for (auto const &[tensor, pointer] : tensor_to_pointer_map) {
            uid_map.emplace(tensor->get_uid(), pointer);
        }
        return execute(handle, uid_map, workspace);
    }

    error_t
    execute(cudnnHandle_t handle,
            std::unordered_map<int64_t, void *> &tensor_uid_to_pointer_map,
            void *workspace) const {
        return execute_plan_at_index(handle, tensor_uid_to_pointer_map, workspace, plans.candidate);
    }

    error_t
    execute(cudnnHandle_t handle,
            std::unordered_map<int64_t, void *> &tensor_uid_to_pointer_map,
            void *workspace,
            std::vector<int64_t> const &override_uids,
            std::vector<std::vector<int64_t>> const &override_shapes,
            std::vector<std::vector<int64_t>> const &override_strides) const {
        return execute_plan_at_index(handle,
                                     tensor_uid_to_pointer_map,
                                     workspace,
                                     plans.candidate,
                                     override_uids,
                                     override_shapes,
                                     override_strides);
    }

    error_t
    execute(cudnnHandle_t handle, void **sorted_user_ptrs, int n_user, void *workspace) const {
        return execute_plan_at_index(handle, sorted_user_ptrs, n_user, workspace, plans.candidate);
    }

    // --- execute_plan_at_index: convert key types ---

    error_t
    execute_plan_at_index(cudnnHandle_t handle,
                          std::unordered_map<std::shared_ptr<Tensor_attributes>, void *> &tensor_to_pointer_map,
                          void *workspace,
                          int64_t plan_index) const {
        std::unordered_map<int64_t, void *> uid_map;
        for (auto const &[tensor, pointer] : tensor_to_pointer_map) {
            uid_map.emplace(tensor->get_uid(), pointer);
        }
        return execute_plan_at_index(handle, uid_map, workspace, plan_index);
    }

    // uid map → extract sorted ptrs, delegate to the sorted_ptrs implementation
    error_t
    execute_plan_at_index(cudnnHandle_t handle,
                          std::unordered_map<int64_t, void *> &tensor_uid_to_pointer_map,
                          void *workspace,
                          int64_t plan_index,
                          std::vector<int64_t> const &override_uids                 = {},
                          std::vector<std::vector<int64_t>> const &override_shapes  = {},
                          std::vector<std::vector<int64_t>> const &override_strides = {}) const {
        if (!varpack_prep_state->prepared.load(std::memory_order_acquire)) {
            CHECK_CUDNN_FRONTEND_ERROR(const_cast<Graph *>(this)->prepare_variant_pack_template());
        }
        const int n_user        = (int)varpack_template.user_slots.size();
        constexpr int STACK_MAX = 32;
        void *stack_ptrs[STACK_MAX];
        std::vector<void *> heap_ptrs;
        void **user_ptrs = (n_user <= STACK_MAX) ? stack_ptrs : (heap_ptrs.resize(n_user), heap_ptrs.data());
        for (int i = 0; i < n_user; i++) {
            int64_t uid = varpack_template.all_uids[varpack_template.user_slots[i]];
            auto it     = tensor_uid_to_pointer_map.find(uid);
            RETURN_CUDNN_FRONTEND_ERROR_IF(it == tensor_uid_to_pointer_map.end(),
                                           error_code_t::INVALID_VARIANT_PACK,
                                           "Uid " + std::to_string(uid) + " not found in variant pack.");
            user_ptrs[i] = it->second;
        }
        return execute_plan_at_index(
            handle, user_ptrs, n_user, workspace, plan_index, override_uids, override_shapes, override_strides);
    }

    // --- THE implementation: sorted pointer array + plan index + optional overrides ---
    // Thread-safe: copies the pre-built template to stack, patches user pointers, dispatches.
    error_t
    execute_plan_at_index(cudnnHandle_t handle,
                          void **sorted_user_ptrs,
                          int n_user,
                          void *workspace,
                          int64_t plan_index,
                          std::vector<int64_t> const &override_uids                 = {},
                          std::vector<std::vector<int64_t>> const &override_shapes  = {},
                          std::vector<std::vector<int64_t>> const &override_strides = {}) const {
        // Lazy init: prepare template if not done (e.g. deserialized graphs, build_plan_at_index)
        if (!varpack_prep_state->prepared.load(std::memory_order_acquire)) {
            CHECK_CUDNN_FRONTEND_ERROR(const_cast<Graph *>(this)->prepare_variant_pack_template());
        }

        CHECK_CUDNN_FRONTEND_ERROR(plans.is_plan_index_executable(plan_index));

        // Validate n_user matches expected user slot count
        RETURN_CUDNN_FRONTEND_ERROR_IF(n_user != static_cast<int>(varpack_template.user_slots.size()),
                                       error_code_t::INVALID_VARIANT_PACK,
                                       "n_user (" + std::to_string(n_user) +
                                           ") does not match expected user slot count (" +
                                           std::to_string(varpack_template.user_slots.size()) + ").");
        RETURN_CUDNN_FRONTEND_ERROR_IF(
            n_user > 0 && sorted_user_ptrs == nullptr, error_code_t::INVALID_VARIANT_PACK, "sorted_user_ptrs is null.");

        // Thread-safe: copy template to local stack, then patch.
        const int N             = (int)varpack_template.all_uids.size();
        constexpr int STACK_MAX = 32;
        void *stack_ptrs[STACK_MAX];
        std::vector<void *> heap_ptrs;
        void **ptrs;
        if (N <= STACK_MAX) {
            std::memcpy(stack_ptrs, varpack_template.template_ptrs.data(), N * sizeof(void *));
            ptrs = stack_ptrs;
        } else {
            heap_ptrs = varpack_template.template_ptrs;
            ptrs      = heap_ptrs.data();
        }

        // 1. Patch user pointers
        for (int i = 0; i < n_user; i++) {
            ptrs[varpack_template.user_slots[i]] = sorted_user_ptrs[i];
        }

        // 2. Update workspace-relative pointers (pre-computed slot indices)
        for (auto const &[slot, offset] : varpack_template.workspace_slots) {
            ptrs[slot] = static_cast<char *>(workspace) + offset;
        }

        // 3. Re-apply replacements (currently only Slice nodes)
        for (auto const &[dst_slot, src_info] : varpack_template.replacement_slots) {
            ptrs[dst_slot] = static_cast<char *>(ptrs[src_info.first]) + src_info.second;
        }

        // 4. Run auxiliary kernels (e.g. SDPA reduction accumulator init)
        CHECK_CUDNN_FRONTEND_ERROR(run_auxiliary_kernels(handle, workspace, cached_workspace_modifications));

        CUDNN_FE_LOG_LABEL_ENDL("INFO: Executing graph_uid " << graph_uid);

        // 5. Dispatch
        void *engine_workspace = static_cast<char *>(workspace) + fe_workspace_size;

        if (plan_index == graph::Execution_plan_list::OSS_SDPA_ENGINE_CANDIDATE) {
            cudaStream_t stream = nullptr;
            _CUDNN_CHECK_CUDNN_ERROR(detail::get_stream(handle, &stream));
            int device_ordinal = 0;
            detail::cuda_get_device(&device_ordinal);
            if (override_uids.empty()) {
                CHECK_CUDNN_FRONTEND_ERROR(
                    plans.execute_oss_sdpa_engine(ptrs, engine_workspace, device_ordinal, stream));
            } else {
                CHECK_CUDNN_FRONTEND_ERROR(plans.execute_oss_sdpa_engine(
                    ptrs, engine_workspace, device_ordinal, stream, override_uids, override_shapes, override_strides));
            }
            return {error_code_t::OK, ""};
        }

        if (plan_index == graph::Execution_plan_list::OSS_RMS_NORM_SILU_ENGINE_CANDIDATE) {
            cudaStream_t stream = nullptr;
            _CUDNN_CHECK_CUDNN_ERROR(detail::get_stream(handle, &stream));
            int device_ordinal = 0;
            detail::cuda_get_device(&device_ordinal);
            CHECK_CUDNN_FRONTEND_ERROR(
                plans.execute_oss_rms_norm_silu_engine(ptrs, engine_workspace, device_ordinal, stream));
            return {error_code_t::OK, ""};
        }

        // Backend path
        if (override_uids.empty()) {
            CHECK_CUDNN_FRONTEND_ERROR(detail::execute(
                handle, plans.execution_plans[plan_index].get(), ptrs, varpack_template.all_uids, engine_workspace));
        } else {
            CHECK_CUDNN_FRONTEND_ERROR(detail::execute(handle,
                                                       plans.execution_plans[plan_index].get(),
                                                       ptrs,
                                                       varpack_template.all_uids,
                                                       engine_workspace,
                                                       override_uids,
                                                       override_shapes,
                                                       override_strides));
        }
        return {error_code_t::OK, ""};
    }

    error_t
    warmup(cudnnHandle_t handle) {
        cudaStream_t fake_stream;

        cudaStream_t original_stream;

        _CUDNN_CHECK_CUDNN_ERROR(detail::get_stream(handle, &original_stream));

        CUDNN_FE_LOG_BANNER("WARMUP (BEGIN FAKE GRAPH CAPTURE) ");

        if (original_stream == nullptr) {
            _CUDNN_CHECK_CUDA_ERROR(detail::cuda_stream_create(&fake_stream));
            _CUDNN_CHECK_CUDNN_ERROR(detail::set_stream(handle, fake_stream));
        } else {
            fake_stream = original_stream;
        }

        cudaGraph_t graph_obj;

        cudaStreamCaptureStatus capture_status;

        _CUDNN_CHECK_CUDA_ERROR(detail::cuda_stream_is_capturing(fake_stream, &capture_status));

        CUDNN_FE_LOG_LABEL_ENDL("INFO: capture_status "
                                << capture_status << " original_stream "
                                << ((original_stream == nullptr) ? "DEFAULT (NULL) Stream" : "NON-DEFAULT Stream"));

        if (capture_status != cudaStreamCaptureStatusNone) {
            CUDNN_FE_LOG_LABEL_ENDL("INFO: cuda graph capture active, aborting warmup");
            return {error_code_t::OK, "cuda graph capture active, aborting warmup"};
        }

        _CUDNN_CHECK_CUDA_ERROR(detail::cuda_graph_begin_capture(fake_stream, cudaStreamCaptureModeRelaxed));

        std::unordered_map<int64_t, void *> tensor_uid_to_pointer_map;

        void *tmp_pointer = reinterpret_cast<void *>(0x7f0000000000llu);

        _CUDNN_CHECK_CUDA_ERROR(detail::cuda_malloc((void **)&tmp_pointer, 1024 * 1024));

        float tmp_double  = 1.0f;
        void *cpu_pointer = reinterpret_cast<void *>(&tmp_double);

        for (auto const &tensor : deserialized_tensor_properties) {
            if (tensor->get_is_virtual() == false) {
                if (tensor->get_is_pass_by_value() == false) {
                    tensor_uid_to_pointer_map.emplace(tensor->get_uid(), tmp_pointer);
                } else {
                    tensor_uid_to_pointer_map.emplace(tensor->get_uid(), cpu_pointer);
                }
            }
        }

        CUDNN_FE_LOG_LABEL_ENDL("INFO: full_graph_inputs: " << full_graph_inputs.size() << " elements");
        for (auto const &tensor : full_graph_inputs) {
            CUDNN_FE_LOG_LABEL_ENDL("\tuid: " << tensor->get_uid()
                                              << ", is_pass_by_value = " << tensor->get_is_pass_by_value());
            if (tensor->get_is_pass_by_value() == false) {
                tensor_uid_to_pointer_map.emplace(tensor->get_uid(), tmp_pointer);
            } else {
                tensor_uid_to_pointer_map.emplace(tensor->get_uid(), cpu_pointer);
            }
        }
        CUDNN_FE_LOG_LABEL_ENDL("INFO: full_graph_outputs: " << full_graph_outputs.size() << " elements");
        for (auto const &tensor : full_graph_outputs) {
            CUDNN_FE_LOG_LABEL_ENDL("\tuid: " << tensor->get_uid());
            tensor_uid_to_pointer_map.emplace(tensor->get_uid(), tmp_pointer);
        }

        CHECK_CUDNN_FRONTEND_ERROR(
            extend_tensor_map_with_pass_by_value_tensors_(tensor_uid_to_pointer_map, deserialized_pass_by_value));

        auto cudnn_status = execute(handle, tensor_uid_to_pointer_map, tmp_pointer);
        (void)cudnn_status;  // No need to check bad executes

        _CUDNN_CHECK_CUDA_ERROR(detail::cuda_graph_end_capture(fake_stream, &graph_obj));

        _CUDNN_CHECK_CUDA_ERROR(detail::cuda_graph_destroy(graph_obj));

        _CUDNN_CHECK_CUDA_ERROR(detail::cuda_free(tmp_pointer));

        _CUDNN_CHECK_CUDNN_ERROR(detail::set_stream(handle, original_stream));

        if (original_stream == nullptr) {
            _CUDNN_CHECK_CUDA_ERROR(detail::cuda_stream_destroy(fake_stream));
        }

        CUDNN_FE_LOG_BANNER("WARMUP (END FAKE GRAPH CAPTURE) ");

        return {error_code_t::OK, ""};
    }

    error_t
    serialize(std::vector<uint8_t> &data) const {
        CUDNN_FE_LOG_BANNER(" SERIALIZE PLAN  ");
#ifndef CUDNN_FRONTEND_SKIP_JSON_LIB
        json j;
        serialize(j);

        auto const candidate = plans.candidate;
        auto execution_plan  = plans.execution_plans[candidate];
        if (execution_plan != nullptr) {
            auto serialized_plan    = execution_plan->getJsonRepresentation();
            j["cudnn_backend_data"] = serialized_plan;
            j["variant_pack_uids"]  = variant_pack_uids;
        }

        j["behavior_notes"] = plans.behavior_notes;

        std::unordered_map<uid_t, pass_by_values_t> tensor_to_pass_by_value;
        CHECK_CUDNN_FRONTEND_ERROR(collect_pass_by_value_tensors_subtree(tensor_to_pass_by_value));

        // Convert pass_by_values to JSON (unordered_map with numeric keys needs manual conversion)
        json pass_by_values_json = json::object();
        for (const auto &[uid, variant_value] : tensor_to_pass_by_value) {
            json variant_json;
            variant_json                             = variant_value;
            pass_by_values_json[std::to_string(uid)] = variant_json;
        }
        j["pass_by_values"] = pass_by_values_json;

        std::unordered_map<uid_t, std::tuple<int64_t, int64_t, std::vector<float>>> workspace_modifications;
        int64_t workspace_offset = 0;
        CHECK_CUDNN_FRONTEND_ERROR(collect_tensors_in_workspace_subtree(workspace_modifications, workspace_offset));

        // Convert workspace_modifications to JSON (nlohmann::json doesn't support std::tuple directly)
        json workspace_modifications_json = json::object();
        for (const auto &[uid, tuple_value] : workspace_modifications) {
            json tuple_json = json::array();
            tuple_json.push_back(std::get<0>(tuple_value));
            tuple_json.push_back(std::get<1>(tuple_value));
            tuple_json.push_back(std::get<2>(tuple_value));
            workspace_modifications_json[std::to_string(uid)] = tuple_json;
        }
        j["workspace_modifications"] = workspace_modifications_json;

        j["variant_pack_replacements"] = variant_pack_replacements;

        j["fe_workspace_size"] = fe_workspace_size;

        std::vector<std::pair<uid_t, char>> tensors_to_dump_uids;
        for (auto const &[tensor, fmt] : tensors_to_dump) {
            tensors_to_dump_uids.emplace_back(tensor->get_uid(), fmt);
        }
        j["tensors_to_dump"] = tensors_to_dump_uids;

        data = json::to_ubjson(j);
        CUDNN_FE_LOG_BANNER(" SERIALIZE PLAN (ALL OK) ");
        return {error_code_t::OK, ""};
#else
        CUDNN_FRONTEND_UNUSED(data);
        return {error_code_t::GRAPH_NOT_SUPPORTED, "unavailable when compiled with CUDNN_FRONTEND_SKIP_JSON_LIB"};
#endif
    }

    error_t
    deserialize(cudnnHandle_t handle, std::vector<uint8_t> const &data) {
        CUDNN_FE_LOG_BANNER(" DESERIALIZE PLAN WITH HANDLE  ");

#ifndef CUDNN_FRONTEND_SKIP_JSON_LIB
        json j = json::from_ubjson(data);

        if (j.contains("graph_uid") && !j["graph_uid"].is_null()) {
            graph_uid = j["graph_uid"].get<uint64_t>();
        }

        if (j.contains("tensors")) {
            auto tensor_map = j["tensors"].get<std::unordered_map<std::string, json>>();
            for (const auto &tensor_info : tensor_map) {
                auto tensor_attributes = std::make_shared<Tensor_attributes>();
                from_json(tensor_info.second, *tensor_attributes);
                deserialized_tensor_properties.insert(tensor_attributes);
            }
        }

        auto serialized_plan = j["cudnn_backend_data"];

        CHECK_CUDNN_FRONTEND_ERROR(plans.build_plans(handle, serialized_plan));

        plans.behavior_notes = j["behavior_notes"].get<std::vector<std::vector<BehaviorNote_t>>>();

        variant_pack_uids = j["variant_pack_uids"].get<std::unordered_set<graph::Tensor_attributes::uid_t>>();

        // Deserialize pass_by_values from JSON
        if (j.contains("pass_by_values")) {
            auto pass_by_values_json = j["pass_by_values"];
            for (auto it = pass_by_values_json.begin(); it != pass_by_values_json.end(); ++it) {
                uid_t uid                       = std::stoll(it.key());
                pass_by_values_t value          = it.value().get<pass_by_values_t>();
                deserialized_pass_by_value[uid] = value;
            }
        }

        // Deserialize workspace_modifications from JSON
        if (j.contains("workspace_modifications")) {
            auto workspace_modifications_json = j["workspace_modifications"];
            for (auto it = workspace_modifications_json.begin(); it != workspace_modifications_json.end(); ++it) {
                uid_t uid                                 = std::stoll(it.key());
                auto tuple_json                           = it.value();
                auto tuple_value                          = std::make_tuple(tuple_json[0].get<int64_t>(),
                                                   tuple_json[1].get<int64_t>(),
                                                   tuple_json[2].get<std::vector<float>>());
                deserialized_workspace_modifications[uid] = tuple_value;
            }
        }

        variant_pack_replacements = j["variant_pack_replacements"];

        fe_workspace_size = j["fe_workspace_size"];

        // Initialize the execution caches from deserialized data
        cached_pass_by_value           = deserialized_pass_by_value;
        cached_workspace_modifications = deserialized_workspace_modifications;

        // Eager prep, matching what build_plans() does for fresh-build graphs.
        CHECK_CUDNN_FRONTEND_ERROR(prepare_variant_pack_template());

        if (j.contains("tensors_to_dump")) {
            auto dump_uids = j["tensors_to_dump"].get<std::vector<std::pair<uid_t, char>>>();
            for (auto const &[uid, fmt] : dump_uids) {
                for (auto const &tensor : deserialized_tensor_properties) {
                    if (tensor->get_uid() == uid) {
                        tensors_to_dump.emplace_back(tensor, fmt);
                        break;
                    }
                }
            }
        }

        CHECK_CUDNN_FRONTEND_ERROR(warmup(handle));

        CUDNN_FE_LOG_BANNER(" DESERIALIZE PLAN WITH HANDLE (ALL OK) ");

        return {error_code_t::OK, ""};
#else
        CUDNN_FRONTEND_UNUSED(handle);
        CUDNN_FRONTEND_UNUSED(data);
        return {error_code_t::GRAPH_NOT_SUPPORTED, "unavailable when compiled with CUDNN_FRONTEND_SKIP_JSON_LIB"};
#endif
    }

    Type
    getType() override {
        return Type::COMPOSITE;
    }

    Graph &
    set_intermediate_data_type(DataType_t type);
    Graph &
    set_io_data_type(DataType_t type);
    Graph &
    set_compute_data_type(DataType_t type);
    Graph &
    set_dynamic_shape_enabled(bool is_enabled);
    Graph &
    set_override_shape_enabled(bool is_enabled);
    Graph &
    set_sm_count(int32_t type);
    Graph &
    set_sm_version(int32_t version);
    Graph &
    set_kernel_cache(std::shared_ptr<KernelCache> cache);
    Graph &
    set_device_properties(std::shared_ptr<const DeviceProperties> device_prop);

    Graph &
    set_name(std::string const &name) {
        context.set_name(name);
        return *this;
    }

    error_t
    query_tensor_attributes_of_uid(int64_t const uid, Tensor_attributes &tensor) const;

    std::shared_ptr<Tensor_attributes>
    tensor(Tensor_attributes const &tensor);

    // Overloaded tensor() methods for compile-time constants
    std::shared_ptr<Tensor_attributes>
    tensor(float const &scalar, ScalarType scalar_type);

    std::shared_ptr<Tensor_attributes>
    tensor(half const &scalar, ScalarType scalar_type);

    std::shared_ptr<Tensor_attributes>
    tensor(nv_bfloat16 const &scalar, ScalarType scalar_type);

    std::shared_ptr<Tensor_attributes>
    tensor(int32_t const &scalar, ScalarType scalar_type);

    std::shared_ptr<Tensor_attributes>
    tensor(int64_t const &scalar, ScalarType scalar_type);

    std::shared_ptr<Tensor_attributes>
    tensor(double const &scalar, ScalarType scalar_type);

    std::shared_ptr<Tensor_attributes>
    tensor_like(std::shared_ptr<Tensor_attributes> const &tensor, std::string const &name = std::string{});

    std::array<std::shared_ptr<Tensor_attributes>, 3> layernorm(std::shared_ptr<Tensor_attributes>,
                                                                std::shared_ptr<Tensor_attributes>,
                                                                std::shared_ptr<Tensor_attributes>,
                                                                Layernorm_attributes);

    std::array<std::shared_ptr<Tensor_attributes>, 3> adalayernorm(std::shared_ptr<Tensor_attributes>,
                                                                   std::shared_ptr<Tensor_attributes>,
                                                                   std::shared_ptr<Tensor_attributes>,
                                                                   AdaLayernorm_attributes);

    std::array<std::shared_ptr<Tensor_attributes>, 3> instancenorm(std::shared_ptr<Tensor_attributes>,
                                                                   std::shared_ptr<Tensor_attributes>,
                                                                   std::shared_ptr<Tensor_attributes>,
                                                                   Instancenorm_attributes);

    std::array<std::shared_ptr<Tensor_attributes>, 5> batchnorm(std::shared_ptr<Tensor_attributes>,
                                                                std::shared_ptr<Tensor_attributes>,
                                                                std::shared_ptr<Tensor_attributes>,
                                                                Batchnorm_attributes);

    std::shared_ptr<Tensor_attributes> batchnorm_inference(std::shared_ptr<Tensor_attributes>,
                                                           std::shared_ptr<Tensor_attributes>,
                                                           std::shared_ptr<Tensor_attributes>,
                                                           std::shared_ptr<Tensor_attributes>,
                                                           std::shared_ptr<Tensor_attributes>,
                                                           Batchnorm_inference_attributes);

    std::array<std::shared_ptr<Tensor_attributes>, 6> bn_finalize(std::shared_ptr<Tensor_attributes>,
                                                                  std::shared_ptr<Tensor_attributes>,
                                                                  std::shared_ptr<Tensor_attributes>,
                                                                  std::shared_ptr<Tensor_attributes>,
                                                                  std::shared_ptr<Tensor_attributes>,
                                                                  std::shared_ptr<Tensor_attributes>,
                                                                  BN_finalize_attributes);

    std::shared_ptr<Tensor_attributes> conv_fprop(std::shared_ptr<Tensor_attributes>,
                                                  std::shared_ptr<Tensor_attributes>,
                                                  Conv_fprop_attributes);

    std::shared_ptr<Tensor_attributes> conv_dgrad(std::shared_ptr<Tensor_attributes>,
                                                  std::shared_ptr<Tensor_attributes>,
                                                  Conv_dgrad_attributes);

    std::shared_ptr<Tensor_attributes> conv_wgrad(std::shared_ptr<Tensor_attributes>,
                                                  std::shared_ptr<Tensor_attributes>,
                                                  Conv_wgrad_attributes);

    std::array<std::shared_ptr<Tensor_attributes>, 5> dbn_weight(std::shared_ptr<Tensor_attributes>,
                                                                 std::shared_ptr<Tensor_attributes>,
                                                                 std::shared_ptr<Tensor_attributes>,
                                                                 std::shared_ptr<Tensor_attributes>,
                                                                 std::shared_ptr<Tensor_attributes>,
                                                                 DBN_weight_attributes);

    std::array<std::shared_ptr<Tensor_attributes>, 3> batchnorm_backward(std::shared_ptr<Tensor_attributes>,
                                                                         std::shared_ptr<Tensor_attributes>,
                                                                         std::shared_ptr<Tensor_attributes>,
                                                                         Batchnorm_backward_attributes);

    std::array<std::shared_ptr<Tensor_attributes>, 3> layernorm_backward(std::shared_ptr<Tensor_attributes>,
                                                                         std::shared_ptr<Tensor_attributes>,
                                                                         std::shared_ptr<Tensor_attributes>,
                                                                         Layernorm_backward_attributes);

    std::array<std::shared_ptr<Tensor_attributes>, 3> adalayernorm_backward(std::shared_ptr<Tensor_attributes>,
                                                                            std::shared_ptr<Tensor_attributes>,
                                                                            std::shared_ptr<Tensor_attributes>,
                                                                            AdaLayernorm_backward_attributes);

    std::array<std::shared_ptr<Tensor_attributes>, 3> instancenorm_backward(std::shared_ptr<Tensor_attributes>,
                                                                            std::shared_ptr<Tensor_attributes>,
                                                                            std::shared_ptr<Tensor_attributes>,
                                                                            Instancenorm_backward_attributes);
    std::array<std::shared_ptr<Tensor_attributes>, 2> genstats(std::shared_ptr<Tensor_attributes>, Genstats_attributes);

    std::array<std::shared_ptr<Tensor_attributes>, 2> rmsnorm(std::shared_ptr<Tensor_attributes>,
                                                              std::shared_ptr<Tensor_attributes>,
                                                              Rmsnorm_attributes);

    std::shared_ptr<Tensor_attributes> rope(std::shared_ptr<Tensor_attributes>,
                                            std::shared_ptr<Tensor_attributes>,
                                            RoPE_attributes);

    std::shared_ptr<Tensor_attributes> rope_backward(std::shared_ptr<Tensor_attributes>,
                                                     std::shared_ptr<Tensor_attributes>,
                                                     RoPE_backward_attributes);

    std::array<std::shared_ptr<Tensor_attributes>, 3> rmsnorm_backward(std::shared_ptr<Tensor_attributes>,
                                                                       std::shared_ptr<Tensor_attributes>,
                                                                       std::shared_ptr<Tensor_attributes>,
                                                                       std::shared_ptr<Tensor_attributes>,
                                                                       Rmsnorm_backward_attributes);

    std::array<std::shared_ptr<Tensor_attributes>, 2> sdpa(std::shared_ptr<Tensor_attributes>,
                                                           std::shared_ptr<Tensor_attributes>,
                                                           std::shared_ptr<Tensor_attributes>,
                                                           SDPA_attributes);

    // FP8 version
    std::array<std::shared_ptr<Tensor_attributes>, 4> sdpa_fp8(std::shared_ptr<Tensor_attributes>,
                                                               std::shared_ptr<Tensor_attributes>,
                                                               std::shared_ptr<Tensor_attributes>,
                                                               std::shared_ptr<Tensor_attributes>,
                                                               std::shared_ptr<Tensor_attributes>,
                                                               std::shared_ptr<Tensor_attributes>,
                                                               std::shared_ptr<Tensor_attributes>,
                                                               std::shared_ptr<Tensor_attributes>,
                                                               std::shared_ptr<Tensor_attributes>,
                                                               SDPA_fp8_attributes);

    // MXFP8 version
    std::array<std::shared_ptr<Tensor_attributes>, 3> sdpa_fp8(std::shared_ptr<Tensor_attributes>,
                                                               std::shared_ptr<Tensor_attributes>,
                                                               std::shared_ptr<Tensor_attributes>,
                                                               std::shared_ptr<Tensor_attributes>,
                                                               std::shared_ptr<Tensor_attributes>,
                                                               std::shared_ptr<Tensor_attributes>,
                                                               SDPA_fp8_attributes);

    inline std::array<std::shared_ptr<Tensor_attributes>, 7> sdpa_fp8_backward(std::shared_ptr<Tensor_attributes>,
                                                                               std::shared_ptr<Tensor_attributes>,
                                                                               std::shared_ptr<Tensor_attributes>,
                                                                               std::shared_ptr<Tensor_attributes>,
                                                                               std::shared_ptr<Tensor_attributes>,
                                                                               std::shared_ptr<Tensor_attributes>,
                                                                               std::shared_ptr<Tensor_attributes>,
                                                                               std::shared_ptr<Tensor_attributes>,
                                                                               std::shared_ptr<Tensor_attributes>,
                                                                               std::shared_ptr<Tensor_attributes>,
                                                                               std::shared_ptr<Tensor_attributes>,
                                                                               std::shared_ptr<Tensor_attributes>,
                                                                               std::shared_ptr<Tensor_attributes>,
                                                                               std::shared_ptr<Tensor_attributes>,
                                                                               std::shared_ptr<Tensor_attributes>,
                                                                               std::shared_ptr<Tensor_attributes>,
                                                                               std::shared_ptr<Tensor_attributes>,
                                                                               std::shared_ptr<Tensor_attributes>,
                                                                               SDPA_fp8_backward_attributes);

    // MXFP8 version
    std::array<std::shared_ptr<Tensor_attributes>, 6> sdpa_fp8_backward(std::shared_ptr<Tensor_attributes>,
                                                                        std::shared_ptr<Tensor_attributes>,
                                                                        std::shared_ptr<Tensor_attributes>,
                                                                        std::shared_ptr<Tensor_attributes>,
                                                                        std::shared_ptr<Tensor_attributes>,
                                                                        std::shared_ptr<Tensor_attributes>,
                                                                        std::shared_ptr<Tensor_attributes>,
                                                                        std::shared_ptr<Tensor_attributes>,
                                                                        std::shared_ptr<Tensor_attributes>,
                                                                        std::shared_ptr<Tensor_attributes>,
                                                                        std::shared_ptr<Tensor_attributes>,
                                                                        std::shared_ptr<Tensor_attributes>,
                                                                        std::shared_ptr<Tensor_attributes>,
                                                                        std::shared_ptr<Tensor_attributes>,
                                                                        std::shared_ptr<Tensor_attributes>,
                                                                        std::shared_ptr<Tensor_attributes>,
                                                                        std::shared_ptr<Tensor_attributes>,
                                                                        SDPA_fp8_backward_attributes);

    std::array<std::shared_ptr<Tensor_attributes>, 3> sdpa_backward(std::shared_ptr<Tensor_attributes>,
                                                                    std::shared_ptr<Tensor_attributes>,
                                                                    std::shared_ptr<Tensor_attributes>,
                                                                    std::shared_ptr<Tensor_attributes>,
                                                                    std::shared_ptr<Tensor_attributes>,
                                                                    std::shared_ptr<Tensor_attributes>,
                                                                    SDPA_backward_attributes);

    std::shared_ptr<Tensor_attributes> slice(std::shared_ptr<Tensor_attributes>, Slice_attributes);

    std::shared_ptr<Tensor_attributes> transpose(std::shared_ptr<Tensor_attributes>, Transpose_attributes);

    std::array<std::shared_ptr<Tensor_attributes>, 2> block_scale_quantize(std::shared_ptr<Tensor_attributes>,
                                                                           Block_scale_quantize_attributes);

    std::shared_ptr<Tensor_attributes> block_scale_dequantize(std::shared_ptr<Tensor_attributes>,
                                                              std::shared_ptr<Tensor_attributes>,
                                                              Block_scale_dequantize_attributes);

    std::shared_ptr<Tensor_attributes> concatenate(std::vector<std::shared_ptr<Tensor_attributes>>,
                                                   Concatenate_attributes);

    std::shared_ptr<Tensor_attributes> moe_grouped_matmul(std::shared_ptr<Tensor_attributes>,
                                                          std::shared_ptr<Tensor_attributes>,
                                                          std::shared_ptr<Tensor_attributes>,
                                                          std::shared_ptr<Tensor_attributes>,
                                                          std::shared_ptr<Tensor_attributes>,
                                                          Moe_grouped_matmul_attributes);

    std::shared_ptr<Tensor_attributes> moe_grouped_matmul_bwd(std::shared_ptr<Tensor_attributes>,
                                                              std::shared_ptr<Tensor_attributes>,
                                                              std::shared_ptr<Tensor_attributes>,
                                                              Moe_grouped_matmul_bwd_attributes);

    [[deprecated]] std::array<std::shared_ptr<Tensor_attributes>, 2>
    scaled_dot_product_flash_attention(std::shared_ptr<Tensor_attributes> q,
                                       std::shared_ptr<Tensor_attributes> k,
                                       std::shared_ptr<Tensor_attributes> v,
                                       SDPA_attributes attributes) {
        return sdpa(q, k, v, attributes);
    }
    [[deprecated]] std::array<std::shared_ptr<Tensor_attributes>, 3>
    scaled_dot_product_flash_attention_backward(std::shared_ptr<Tensor_attributes> q,
                                                std::shared_ptr<Tensor_attributes> k,
                                                std::shared_ptr<Tensor_attributes> v,
                                                std::shared_ptr<Tensor_attributes> o,
                                                std::shared_ptr<Tensor_attributes> dO,
                                                std::shared_ptr<Tensor_attributes> stats,
                                                SDPA_backward_attributes attributes) {
        return sdpa_backward(q, k, v, o, dO, stats, attributes);
    }

    error_t
    create_execution_plans(std::vector<HeurMode_t> const &mode);

    error_t
    create_execution_plan(int64_t const engine_id, std::unordered_map<KnobType_t, int64_t> const &knobs);

    int64_t
    get_execution_plan_count() const;

    inline error_t
    get_engine_count(int64_t &count);

    inline error_t
    get_knobs_for_engine(int64_t const engine, std::vector<Knob> &);

    error_t
    check_support(cudnnHandle_t h) {
        // handle not required anymore
        // TODO: remove this function in next release
        (void)h;
        return check_support();
    }

    // overload for deviceless AoT compilation
    error_t
    check_support() {
        // Check OSS engine first if registered

        CHECK_CUDNN_FRONTEND_ERROR(context.populate_sm_version_from_device());
        auto sm_version = context.get_sm_version();

        // Check OSS SDPA engine
        if (plans.has_oss_sdpa_engine()) {
            auto oss_status = plans.check_oss_sdpa_engine_support(sm_version);
            if (oss_status.is_good()) {
                return {error_code_t::OK, ""};
            }
            // Fall through to check other engines
        }

        // Check OSS RmsNorm+SiLU engine
        if (plans.has_oss_rms_norm_silu_engine()) {
            auto oss_status = plans.check_oss_rms_norm_silu_support(sm_version);
            if (oss_status.is_good()) {
                return {error_code_t::OK, ""};
            }
            // Fall through to check cuDNN plans
        }

        CHECK_CUDNN_FRONTEND_ERROR(plans.check_support());
        return {error_code_t::OK, ""};
    }

    // TODO: remove this function in next release
    error_t
    build(cudnnHandle_t const &handle,
          std::vector<HeurMode_t> const &mode,
          BuildPlanPolicy_t const policy     = BuildPlanPolicy_t::HEURISTICS_CHOICE,
          bool const do_multithreaded_builds = false);

    // overload for deviceless AoT compilation
    error_t
    build(std::vector<HeurMode_t> const &mode,
          BuildPlanPolicy_t const policy     = BuildPlanPolicy_t::HEURISTICS_CHOICE,
          bool const do_multithreaded_builds = false);

    // -----------------------------------------------------------------------
    // Variant pack template: pre-indexed flat array built at build_plans() time.
    // Eliminates per-call map operations in execute().
    // -----------------------------------------------------------------------

    struct VariantPackTemplate {
        std::vector<int64_t> all_uids;      // all UIDs in sorted order (user + internal)
        std::vector<void *> template_ptrs;  // pre-filled: pass_by_value entries, nullptr for user/workspace slots
        std::vector<int> user_slots;        // indices into template_ptrs that the user fills (sorted UID order)

        // Pre-computed slot indices for per-call updates (avoid linear scans at execute time)
        std::vector<std::pair<int, int64_t>> workspace_slots;  // (slot, byte_offset)
        std::vector<std::pair<int, std::pair<int, int64_t>>>
            replacement_slots;  // (dst_slot, (src_slot, offset)) — only used by Slice nodes
    };

    VariantPackTemplate varpack_template;

    // Per-Graph sync state for lazy varpack-template prep. Wrapped in a box so
    // Graph stays default-copy/move-constructible (samples build Graphs into
    // std::tuple by copy).
    struct VarpackPrepState {
        std::atomic<bool> prepared{false};
        std::mutex mu;
    };
    struct VarpackPrepStateBox {
        std::unique_ptr<VarpackPrepState> ptr                = std::make_unique<VarpackPrepState>();
        VarpackPrepStateBox()                                = default;
        VarpackPrepStateBox(VarpackPrepStateBox &&) noexcept = default;
        VarpackPrepStateBox &
        operator=(VarpackPrepStateBox &&) noexcept = default;
        VarpackPrepStateBox(VarpackPrepStateBox const &other) : ptr(std::make_unique<VarpackPrepState>()) {
            if (other.ptr) {
                ptr->prepared.store(other.ptr->prepared.load(std::memory_order_acquire), std::memory_order_release);
            }
        }
        VarpackPrepStateBox &
        operator=(VarpackPrepStateBox const &other) {
            if (this != &other) {
                VarpackPrepStateBox tmp(other);
                ptr.swap(tmp.ptr);
            }
            return *this;
        }
        VarpackPrepState *
        operator->() const {
            return ptr.get();
        }
    };
    mutable VarpackPrepStateBox varpack_prep_state;

    // Prepares the variant pack template. Called automatically at the end of build_plans().
    error_t
    prepare_variant_pack_template() {
        std::lock_guard<std::mutex> lk(varpack_prep_state->mu);
        if (varpack_prep_state->prepared.load(std::memory_order_relaxed)) {
            return {error_code_t::OK, ""};
        }

        VariantPackTemplate t;

        // 1. Start with variant_pack_uids + any replacement source UIDs not already included.
        //    Replacement sources (e.g. slice input on cuDNN < 9.22 pointer-arithmetic fallback) may not
        //    be in variant_pack_uids; we still need a slot for the source pointer when replacements apply.
        t.all_uids.assign(variant_pack_uids.begin(), variant_pack_uids.end());
        for (auto const &[from_uid, value] : variant_pack_replacements) {
            if (variant_pack_uids.find(from_uid) == variant_pack_uids.end()) {
                t.all_uids.push_back(from_uid);
            }
        }
        std::sort(t.all_uids.begin(), t.all_uids.end());
        t.template_ptrs.assign(t.all_uids.size(), nullptr);

        // 2. Build UID → slot index
        std::unordered_map<int64_t, int> uid_to_slot;
        for (int i = 0; i < (int)t.all_uids.size(); i++) {
            uid_to_slot[t.all_uids[i]] = i;
        }

        // 3. Pre-fill pass_by_value entries (scalars like epsilon, alpha, beta)
        std::unordered_map<int64_t, void *> pbv_ptrs;
        CHECK_CUDNN_FRONTEND_ERROR(extend_tensor_map_with_pass_by_value_tensors_(pbv_ptrs, cached_pass_by_value));
        for (auto const &[uid, ptr] : pbv_ptrs) {
            auto it = uid_to_slot.find(uid);
            if (it != uid_to_slot.end()) {
                t.template_ptrs[it->second] = ptr;
            } else {
                int slot = (int)t.all_uids.size();
                t.all_uids.push_back(uid);
                t.template_ptrs.push_back(ptr);
                uid_to_slot[uid] = slot;
            }
        }

        // 4. Register workspace entries (slots allocated, pointers filled at execute time)
        for (auto const &[uid, data] : cached_workspace_modifications) {
            auto it = uid_to_slot.find(uid);
            if (it == uid_to_slot.end()) {
                int slot = (int)t.all_uids.size();
                t.all_uids.push_back(uid);
                t.template_ptrs.push_back(nullptr);  // filled at execute time
                uid_to_slot[uid] = slot;
            }
        }

        // 5. Identify user-fillable slots: nullptr entries that are NOT workspace/replacement targets
        std::unordered_set<int64_t> workspace_uids;
        for (auto const &[uid, data] : cached_workspace_modifications) {
            workspace_uids.insert(uid);
        }
        std::unordered_set<int64_t> replacement_dst_uids;
        for (auto const &[from_uid, value] : variant_pack_replacements) {
            (void)from_uid;
            replacement_dst_uids.insert(value.first);  // value.first = to_uid (the destination)
        }
        for (int i = 0; i < (int)t.template_ptrs.size(); i++) {
            int64_t uid = t.all_uids[i];
            if (t.template_ptrs[i] == nullptr && workspace_uids.find(uid) == workspace_uids.end() &&
                replacement_dst_uids.find(uid) == replacement_dst_uids.end()) {
                t.user_slots.push_back(i);
            }
        }

        // 6. Pre-compute workspace slot indices
        for (auto const &[uid, data] : cached_workspace_modifications) {
            const auto &[operation_type, offset, vec_data] = data;
            auto it                                        = uid_to_slot.find(uid);
            if (it != uid_to_slot.end()) {
                t.workspace_slots.emplace_back(it->second, offset);
            }
        }

        // 7. Pre-compute replacement slot indices
        // variant_pack_replacements: key = from_uid (source), value = {to_uid (destination), byte_offset}
        // Meaning: ptr[to_uid] = ptr[from_uid] + byte_offset (e.g. legacy slice fallback on cuDNN < 9.22)
        for (auto const &[from_uid, value] : variant_pack_replacements) {
            const auto &[to_uid, byte_offset] = value;
            auto it_src                       = uid_to_slot.find(from_uid);
            auto it_dst                       = uid_to_slot.find(to_uid);
            // Replacement: dst_ptr = src_ptr + byte_offset
            if (it_src != uid_to_slot.end() && it_dst != uid_to_slot.end()) {
                t.replacement_slots.emplace_back(it_dst->second, std::make_pair(it_src->second, byte_offset));
            }
        }

        varpack_template = std::move(t);
        // Apply OSS slot indices before the release-store so a reader that
        // observes prepared=true also sees the slot writes.
        apply_oss_slot_indices_to_plans();
        varpack_prep_state->prepared.store(true, std::memory_order_release);

        return {error_code_t::OK, ""};
    }

    // Depends on plans.candidate; kept separate so autotune can re-apply without
    // rebuilding the (candidate-independent) variant pack template.
    void
    apply_oss_slot_indices_to_plans() {
        plans.set_oss_slot_indices([this](int64_t uid) -> int {
            if (uid < 0) return -1;
            for (size_t i = 0; i < varpack_template.all_uids.size(); ++i) {
                if (varpack_template.all_uids[i] == uid) return static_cast<int>(i);
            }
            return -1;
        });
    }

    std::vector<int64_t>
    get_variant_pack_uids_sorted() const {
        std::vector<int64_t> user_uids;
        for (int slot : varpack_template.user_slots) {
            user_uids.push_back(varpack_template.all_uids[slot]);
        }
        return user_uids;
    }

    error_t
    build_plans(cudnnHandle_t const &handle,
                BuildPlanPolicy_t const policy     = BuildPlanPolicy_t::HEURISTICS_CHOICE,
                bool const do_multithreaded_builds = false) {
        // handle not required anymore
        // TODO: remove this function in next release
        (void)handle;
        return build_plans(policy, do_multithreaded_builds);
    }

    // overload for deviceless AoT compilation
    error_t
    build_plans(BuildPlanPolicy_t const policy     = BuildPlanPolicy_t::HEURISTICS_CHOICE,
                bool const do_multithreaded_builds = false);

    error_t
    build_plan_at_index(cudnnHandle_t const &handle, int64_t index) {
        // handle not required anymore
        // TODO: remove this function in next release
        (void)handle;
        return build_plan_at_index(index);
    }

    // overload for deviceless AoT compilation
    error_t
    build_plan_at_index(int64_t index);

    Graph &
    deselect_workspace_greater_than(int64_t const workspace) {
        plans.set_max_workspace_allowed(workspace);
        return *this;
    }

    Graph &
    deselect_shared_mem_greater_than(int64_t const workspace) {
        plans.set_max_shared_mem_allowed(workspace);
        return *this;
    }

    Graph &
    deselect_engines(std::vector<std::string> const &engine_names) {
        plans.set_barred_names(engine_names);
        return *this;
    }

    Graph &
    select_behavior_notes(std::vector<BehaviorNote_t> const &notes) {
        auto status = plans.filter_behavior_notes(notes, true);
        if (status.is_bad()) {
            CUDNN_FE_LOG(status.get_message() << std::endl);
        }
        return *this;
    }

    Graph &
    select_numeric_notes(std::vector<NumericalNote_t> const &notes) {
        auto status = plans.filter_numeric_notes(notes, true);
        if (status.is_bad()) {
            CUDNN_FE_LOG(status.get_message() << std::endl);
        }
        return *this;
    }

    Graph &
    deselect_behavior_notes(std::vector<BehaviorNote_t> const &notes) {
        auto status = plans.filter_behavior_notes(notes, false);
        if (status.is_bad()) {
            CUDNN_FE_LOG(status.get_message() << std::endl);
        }
        return *this;
    }

    Graph &
    deselect_numeric_notes(std::vector<NumericalNote_t> const &notes) {
        auto status = plans.filter_numeric_notes(notes, false);
        if (status.is_bad()) {
            CUDNN_FE_LOG(status.get_message() << std::endl);
        }
        return *this;
    }

    error_t
    get_behavior_notes_for_plan_at_index(int64_t const index, std::vector<BehaviorNote_t> &notes) const;

    error_t
    get_behavior_notes(std::vector<BehaviorNote_t> &notes) const;

#ifndef CUDNN_FRONTEND_SKIP_JSON_LIB
    virtual void
    serialize(json &j) const override final {
        // Different from serialization of other INodes.
        // Go over each subnode and serialize them.
        json full_json;

        full_json["context"]["name"]                      = context.get_name();
        full_json["context"]["compute_data_type"]         = context.get_compute_data_type();
        full_json["context"]["intermediate_data_type"]    = context.get_intermediate_data_type();
        full_json["context"]["io_data_type"]              = context.get_io_data_type();
        full_json["context"]["sm_count"]                  = context.get_target_sm_count();
        full_json["context"]["is_dynamic_shape_enabled"]  = context.get_dynamic_shape_enabled();
        full_json["context"]["is_override_shape_enabled"] = context.get_override_shape_enabled();
        full_json["graph_uid"]                            = graph_uid;

        full_json.update(R"( {"tag": "GRAPH"})"_json);
        full_json["nodes"];
        for (auto const &sub_node : sub_nodes) {
            json j_sub_node;
            sub_node->serialize(j_sub_node);
            full_json["nodes"].push_back(j_sub_node);
        }

        j["context"]   = full_json["context"];
        j["graph_uid"] = full_json["graph_uid"];

        j["json_version"]           = "1.0";
        j["cudnn_backend_version"]  = detail::get_backend_version_string();
        j["cudnn_frontend_version"] = CUDNN_FRONTEND_VERSION;
        j["nodes"];
        j["tensors"];
        std::unordered_set<std::string> tensors;
        for (const auto &sub_node : full_json["nodes"]) {
            // Create a short version of the node
            auto short_node       = sub_node;
            short_node["inputs"]  = {};
            short_node["outputs"] = {};

            auto node_name = sub_node["tag"].get<std::string>();
            auto i         = 0;
            // Process node inputs
            for (const auto &input : sub_node["inputs"]) {
                std::string port_name;
                json tensor_info;

                if (node_name == "CONCATENATE") {
                    // Extract port_name and tensor_name
                    port_name   = std::to_string(i);
                    tensor_info = input;
                    i++;
                } else {
                    // Extract port_name and tensor_name
                    port_name   = input[0].get<std::string>();
                    tensor_info = input[1];
                }

                if (tensor_info.is_null()) {
                    continue;
                }

                // Determine the key to use for this tensor
                std::string tensor_key;
                json tensor_ref;
                bool uid_assigned = tensor_info.contains("uid_assigned") && tensor_info["uid_assigned"].get<bool>();

                if (uid_assigned && tensor_info.contains("uid") && tensor_info["uid"].is_number_integer()) {
                    // Use numeric UID if it was explicitly assigned
                    int64_t tensor_uid = tensor_info["uid"].get<int64_t>();
                    tensor_key         = std::to_string(tensor_uid);
                    tensor_ref         = json(tensor_uid);
                } else if (tensor_info.contains("name")) {
                    // Fall back to tensor name if UID not assigned
                    tensor_key = tensor_info["name"].get<std::string>();
                    tensor_ref = tensor_key;
                } else {
                    continue;
                }

                // Update short_node inputs
                short_node["inputs"][port_name] = tensor_ref;

                // Check if the tensor is already in the tensors map
                if (tensors.find(tensor_key) == tensors.end()) {
                    // If not, add it to the j["tensors"]
                    j["tensors"][tensor_key] = tensor_info;
                }
            }

            // Process node outputs
            for (const auto &output : sub_node["outputs"]) {
                // Extract port_name and tensor_name
                auto port_name   = output[0].get<std::string>();
                auto tensor_info = output[1];

                if (tensor_info.is_null()) {
                    continue;
                }

                // Determine the key to use for this tensor
                std::string tensor_key;
                json tensor_ref;
                bool uid_assigned = tensor_info.contains("uid_assigned") && tensor_info["uid_assigned"].get<bool>();

                if (uid_assigned && tensor_info.contains("uid") && tensor_info["uid"].is_number_integer()) {
                    // Use numeric UID if it was explicitly assigned
                    int64_t tensor_uid = tensor_info["uid"].get<int64_t>();
                    tensor_key         = std::to_string(tensor_uid);
                    tensor_ref         = json(tensor_uid);
                } else if (tensor_info.contains("name")) {
                    // Fall back to tensor name if UID not assigned
                    tensor_key = tensor_info["name"].get<std::string>();
                    tensor_ref = tensor_key;
                } else {
                    continue;
                }

                // Update short_node outputs
                short_node["outputs"][port_name] = tensor_ref;

                // Check if the tensor is already in the tensors map
                if (tensors.find(tensor_key) == tensors.end()) {
                    // If not, add it to the j["tensors"]
                    j["tensors"][tensor_key] = tensor_info;
                }
            }

            // Add the short_node to j["nodes"]
            j["nodes"].push_back(short_node);
        }
    };
#endif

    size_t
    key() override final {
        return key(context.get_dynamic_shape_enabled());
    }

    // TODO: temparorily placed in graphs class. This function needs to be a free standing function.
#ifndef CUDNN_FRONTEND_SKIP_JSON_LIB
    error_t
    deserialize(const json &j) {
        if (j.contains("context")) {
            const auto &j_context = j["context"];
            if (j_context.contains("compute_data_type") && !j_context["compute_data_type"].is_null()) {
                context.set_compute_data_type(j_context["compute_data_type"].get<DataType_t>());
            }
            if (j_context.contains("intermediate_data_type") && !j_context["intermediate_data_type"].is_null()) {
                context.set_intermediate_data_type(j_context["intermediate_data_type"].get<DataType_t>());
            }
            if (j_context.contains("io_data_type") && !j_context["io_data_type"].is_null()) {
                context.set_io_data_type(j_context["io_data_type"].get<DataType_t>());
            }
            if (j_context.contains("name") && !j_context["name"].is_null()) {
                context.set_name(j_context["name"].get<std::string>());
            }
            if (j_context.contains("sm_count") && !j_context["sm_count"].is_null()) {
                context.set_target_sm_count(j_context["sm_count"].get<int32_t>());
            }
            if (j_context.contains("is_dynamic_shape_enabled") && !j_context["is_dynamic_shape_enabled"].is_null()) {
                context.set_dynamic_shape_enabled(j_context["is_dynamic_shape_enabled"].get<bool>());
            }
            if (j_context.contains("is_override_shape_enabled") && !j_context["is_override_shape_enabled"].is_null()) {
                context.set_override_shape_enabled(j_context["is_override_shape_enabled"].get<bool>());
            }
        }

        if (j.contains("graph_uid") && !j["graph_uid"].is_null()) {
            graph_uid = j["graph_uid"].get<uint64_t>();
        }

        std::map<std::string, std::shared_ptr<Tensor_attributes>> created_tensors;
        // Iterate through each sub-node in the full JSON
        if (j.contains("nodes") && j["nodes"].is_array()) {
            for (auto j_sub_node : j["nodes"]) {
                // Create a JSON object for inputs
                json inputs;

                // Iterate through each input of the sub-node
                if (j_sub_node.contains("inputs") && j_sub_node["inputs"].is_object()) {
                    for (auto &[port_name, tensor_ref] : j_sub_node["inputs"].items()) {
                        // Convert tensor reference (either numeric UID or string name) to string key
                        std::string tensor_key = tensor_ref.is_number_integer()
                                                     ? std::to_string(tensor_ref.get<int64_t>())
                                                     : tensor_ref.get<std::string>();

                        if (j.contains("tensors") && j["tensors"].contains(tensor_key)) {
                            // Add the input to the inputs JSON object
                            inputs.push_back({port_name, j["tensors"][tensor_key]});
                        }
                    }
                }

                // Create a JSON object for outputs
                json outputs;

                // Iterate through each output of the sub-node
                if (j_sub_node.contains("outputs") && j_sub_node["outputs"].is_object()) {
                    for (auto &[port_name, tensor_ref] : j_sub_node["outputs"].items()) {
                        // Convert tensor reference (either numeric UID or string name) to string key
                        std::string tensor_key = tensor_ref.is_number_integer()
                                                     ? std::to_string(tensor_ref.get<int64_t>())
                                                     : tensor_ref.get<std::string>();

                        if (j.contains("tensors") && j["tensors"].contains(tensor_key)) {
                            // Add the output to the outputs JSON object
                            outputs.push_back({port_name, j["tensors"][tensor_key]});
                        }
                    }
                }

                // Replace the original inputs and outputs of the sub-node with the new JSON objects
                j_sub_node["inputs"]  = inputs;
                j_sub_node["outputs"] = outputs;

                auto check_if_pre_created_tensor = [&created_tensors](std::shared_ptr<Tensor_attributes> t) {
                    if (t == nullptr) {
                        return t;
                    }

                    if (created_tensors.find(t->get_name()) == created_tensors.end()) {
                        created_tensors.insert({t->get_name(), t});
                        return t;
                    } else {
                        return created_tensors[t->get_name()];
                    }
                };

#define CHECK_TENSORS(attributes)                                      \
    for (const auto &[key, tensor] : attributes.inputs) {              \
        attributes.inputs[key] = check_if_pre_created_tensor(tensor);  \
    }                                                                  \
    for (const auto &[key, tensor] : attributes.outputs) {             \
        attributes.outputs[key] = check_if_pre_created_tensor(tensor); \
    }

#define FILL_GLOBAL_IO_TENSOR_MAP(attributes)                              \
    for (auto input_name_to_attr_pair : attributes.inputs) {               \
        if (input_name_to_attr_pair.second != nullptr &&                   \
            (input_name_to_attr_pair.second->get_is_virtual() == false)) { \
            full_graph_inputs.emplace(input_name_to_attr_pair.second);     \
        }                                                                  \
    }                                                                      \
    for (auto output_name_to_attr_pair : attributes.outputs) {             \
        if (output_name_to_attr_pair.second != nullptr) {                  \
            full_graph_outputs.emplace(output_name_to_attr_pair.second);   \
        }                                                                  \
    }
                if (j_sub_node.contains("tag") && j_sub_node["tag"].is_string()) {
                    auto tag = j_sub_node["tag"].get<std::string>();
                    if (tag == "CONV_FPROP") {
                        auto conv_fprop_attributes = j_sub_node.get<Conv_fprop_attributes>();
                        CHECK_TENSORS(conv_fprop_attributes);
                        FILL_GLOBAL_IO_TENSOR_MAP(conv_fprop_attributes);
                        sub_nodes.emplace_back(
                            std::make_unique<ConvolutionNode>(std::move(conv_fprop_attributes), context));
                    } else if (tag == "POINTWISE") {
                        auto pointwise_attributes = j_sub_node.get<Pointwise_attributes>();
                        CHECK_TENSORS(pointwise_attributes);
                        FILL_GLOBAL_IO_TENSOR_MAP(pointwise_attributes);
                        sub_nodes.emplace_back(
                            std::make_unique<PointwiseNode>(std::move(pointwise_attributes), context));
                    } else if (tag == "REDUCTION") {
                        auto reduction_attributes = j_sub_node.get<Reduction_attributes>();
                        CHECK_TENSORS(reduction_attributes);
                        FILL_GLOBAL_IO_TENSOR_MAP(reduction_attributes);
                        sub_nodes.emplace_back(
                            std::make_unique<ReductionNode>(std::move(reduction_attributes), context));
                    } else if (tag == "SDPA") {
                        auto sdpa_attributes = j_sub_node.get<SDPA_attributes>();
                        CHECK_TENSORS(sdpa_attributes);
                        FILL_GLOBAL_IO_TENSOR_MAP(sdpa_attributes);
                        switch (sdpa_attributes.implementation) {
                            case AttentionImplementation_t::AUTO:
                                return {error_code_t::INVALID_VALUE,
                                        "Implementation cannot be AUTO in serialized form"};
                            case AttentionImplementation_t::COMPOSITE:
                                sub_nodes.emplace_back(
                                    std::make_unique<CompositeSDPANode>(std::move(sdpa_attributes), context));
                                break;
                            case AttentionImplementation_t::UNIFIED:
                                sub_nodes.emplace_back(
                                    std::make_unique<UnifiedSDPANode>(std::move(sdpa_attributes), context));
                        }
                    } else if (tag == "SDPA_BWD") {
                        auto sdpa_bwd_attributes = j_sub_node.get<SDPA_backward_attributes>();
                        CHECK_TENSORS(sdpa_bwd_attributes);
                        FILL_GLOBAL_IO_TENSOR_MAP(sdpa_bwd_attributes);
                        sub_nodes.emplace_back(
                            std::make_unique<CompositeSDPABackwardNode>(std::move(sdpa_bwd_attributes), context));
                    } else if (tag == "MATMUL") {
                        auto matmul_attributes = j_sub_node.get<Matmul_attributes>();
                        CHECK_TENSORS(matmul_attributes);
                        FILL_GLOBAL_IO_TENSOR_MAP(matmul_attributes);
                        sub_nodes.emplace_back(std::make_unique<MatmulNode>(std::move(matmul_attributes), context));
                    } else if (tag == "SLICE") {
                        auto slice_attributes = j_sub_node.get<Slice_attributes>();
                        CHECK_TENSORS(slice_attributes);
                        FILL_GLOBAL_IO_TENSOR_MAP(slice_attributes);
                        sub_nodes.emplace_back(std::make_unique<SliceNode>(std::move(slice_attributes), context));
                    } else if (tag == "TRANSPOSE") {
                        auto transpose_attributes = j_sub_node.get<Transpose_attributes>();
                        CHECK_TENSORS(transpose_attributes);
                        FILL_GLOBAL_IO_TENSOR_MAP(transpose_attributes);
                        sub_nodes.emplace_back(
                            std::make_unique<TransposeNode>(std::move(transpose_attributes), context));
                    } else if (tag == "RESAMPLE") {
                        auto resample_attributes = j_sub_node.get<Resample_attributes>();
                        CHECK_TENSORS(resample_attributes);
                        FILL_GLOBAL_IO_TENSOR_MAP(resample_attributes);
                        sub_nodes.emplace_back(std::make_unique<ResampleNode>(std::move(resample_attributes), context));
                    } else if (tag == "CONV_DGRAD") {
                        auto dgrad_attributes = j_sub_node.get<Conv_dgrad_attributes>();
                        CHECK_TENSORS(dgrad_attributes);
                        FILL_GLOBAL_IO_TENSOR_MAP(dgrad_attributes);
                        sub_nodes.emplace_back(std::make_unique<DgradNode>(std::move(dgrad_attributes), context));
                    } else if (tag == "CONV_WGRAD") {
                        auto wgrad_attributes = j_sub_node.get<Conv_wgrad_attributes>();
                        CHECK_TENSORS(wgrad_attributes);
                        FILL_GLOBAL_IO_TENSOR_MAP(wgrad_attributes);
                        sub_nodes.emplace_back(std::make_unique<WgradNode>(std::move(wgrad_attributes), context));
                    } else if (tag == "MOE_GROUPED_MATMUL") {
                        auto moe_grouped_matmul_attributes = j_sub_node.get<Moe_grouped_matmul_attributes>();
                        CHECK_TENSORS(moe_grouped_matmul_attributes);
                        FILL_GLOBAL_IO_TENSOR_MAP(moe_grouped_matmul_attributes);
                        sub_nodes.emplace_back(
                            std::make_unique<MoeGroupedMatmulNode>(std::move(moe_grouped_matmul_attributes), context));
                    }
                }
#undef CHECK_TENSORS
            }
        }

        return {error_code_t::OK, ""};
    }
#endif

    std::string
    print(void) const {
#ifndef CUDNN_FRONTEND_SKIP_JSON_LIB
        std::stringstream ss;
        json j = *this;
        ss << j;
        return ss.str();
#else
        return "print is unavailable when compiled with CUDNN_FRONTEND_SKIP_JSON_LIB";
#endif
    }
};

inline error_t
Graph::get_behavior_notes_for_plan_at_index(int64_t const index, std::vector<BehaviorNote_t> &notes) const {
    CHECK_CUDNN_FRONTEND_ERROR(plans.get_behavior_notes_at_index(index, notes));
    return {error_code_t::OK, ""};
}

inline error_t
Graph::get_behavior_notes(std::vector<BehaviorNote_t> &notes) const {
    int64_t const candidate = plans.candidate;
    RETURN_CUDNN_FRONTEND_ERROR_IF(
        candidate == -1,
        error_code_t::INVALID_VALUE,
        "No candiate plan set for the graph. You can set one by building a plan, which in turn sets the "
        "candidate internally. Do note that you also query behaviour notes for a created-but-not-built plan by using "
        "get_behavior_notes_for_plan_at_index API.");

    CHECK_CUDNN_FRONTEND_ERROR(get_behavior_notes_for_plan_at_index(candidate, notes));
    return {error_code_t::OK, ""};
}

inline int64_t
Graph::get_execution_plan_count() const {
    return plans.execution_plans.size();
}

inline error_t
Graph::get_engine_count(int64_t &count) {
    _CUDNN_CHECK_CUDNN_ERROR(detail::get_attribute(operation_graph->get_raw_desc(),
                                                   CUDNN_ATTR_OPERATIONGRAPH_ENGINE_GLOBAL_COUNT,
                                                   CUDNN_TYPE_INT64,
                                                   1,
                                                   nullptr,
                                                   &count));

    return {error_code_t::OK, ""};
}

inline error_t
Graph::get_knobs_for_engine(int64_t const engine, std::vector<Knob> &knobs) {
    CHECK_CUDNN_FRONTEND_ERROR(detail::query_knobs(engine, operation_graph->get_raw_desc(), knobs));

    return {error_code_t::OK, ""};
}

inline error_t
Graph::create_execution_plans(std::vector<HeurMode_t> const &mode) {
    CUDNN_FE_LOG_BANNER("  CREATE EXECUTION PLANS  (HEURISTICS QUERY)  ");

    // CHECK IF NEED TO OVERRIDE HEURISTICS QUERY
    for (auto &sub_node : sub_nodes) {
        if (auto [engine_id, user_knobs] = sub_node->override_heuristics_query(); engine_id != -1) {
#ifndef CUDNN_FRONTEND_SKIP_JSON_LIB
            CUDNN_FE_LOG_LABEL_ENDL("INFO: Overriding heuristics query with engine ID "
                                    << engine_id << " and user knobs " << nlohmann::json(user_knobs).dump());
#else
            CUDNN_FE_LOG_LABEL_ENDL("INFO: Overriding heuristics query with engine ID "
                                    << engine_id << " and user knobs " << static_cast<int>(user_knobs.size()));
#endif
            CHECK_CUDNN_FRONTEND_ERROR(create_execution_plan(engine_id, user_knobs));
            return {error_code_t::OK, ""};
        }
    }

    // Separate OPENSOURCE from cuDNN modes
    bool has_opensource = false;
    std::vector<HeurMode_t> cudnn_modes;
    for (auto const &m : mode) {
        if (m == HeurMode_t::OPENSOURCE) {
            has_opensource = true;
        } else {
            cudnn_modes.push_back(m);
        }
    }

    // Register OSS engines if OPENSOURCE mode requested
    if (has_opensource) {
        // Try SDPA OSS engine
        auto oss_sdpa_status = register_oss_engine_();
        if (oss_sdpa_status.is_good()) {
            CUDNN_FE_LOG_LABEL_ENDL("INFO: Registered OSS SDPA prefill engine");
        }

        // Try RmsNorm+SiLU OSS engine
        auto oss_norm_status = register_oss_rms_norm_silu_engine_();
        if (oss_norm_status.is_good()) {
            CUDNN_FE_LOG_LABEL_ENDL("INFO: Registered OSS RmsNorm+SiLU engine");
        }

        if (oss_sdpa_status.is_bad() && oss_norm_status.is_bad()) {
            CUDNN_FE_LOG_LABEL_ENDL("WARN: No OSS engine matched the graph pattern");
        }
    }

    // Query cuDNN heuristics for non-OPENSOURCE modes
    if (!cudnn_modes.empty()) {
        EngineConfigList op_graph_to_configs;
        CHECK_CUDNN_FRONTEND_ERROR(detail::query_cudnn_heuristics_impl(
            operation_graph, op_graph_to_configs, cudnn_modes, context.get_target_sm_count(), device_properties));

        CUDNN_FE_LOG_LABEL_ENDL("INFO: Extracting engine configs.");

        plans.set_tag(operation_graph->getTag());
        plans.enqueue_engine_configs(op_graph_to_configs);
        plans.set_kernel_cache(kernel_cache);

        CUDNN_FE_LOG_LABEL_ENDL("INFO: Querying engine config properties.");
        CHECK_CUDNN_FRONTEND_ERROR(plans.query_properties());
    }

    CUDNN_FE_LOG_BANNER("  HEURISTICS QUERY ALL OK  ");
    return {error_code_t::OK, ""};
}

inline error_t
Graph::create_execution_plan(int64_t const engine_id, std::unordered_map<KnobType_t, int64_t> const &user_knobs) {
    // first create the engine
    // this just uses the global engine id and operation graph
    CUDNN_FE_LOG_BANNER("  CREATE EXECUTION PLAN  for engine id " << engine_id << "  ");
    detail::backend_descriptor engine(CUDNN_BACKEND_ENGINE_DESCRIPTOR);
    RETURN_CUDNN_FRONTEND_ERROR_IF(engine.get_status() != CUDNN_STATUS_SUCCESS,
                                   error_code_t::CUDNN_BACKEND_API_FAILED,
                                   "Failed to create engine's backend descriptor.");
    CHECK_CUDNN_FRONTEND_ERROR(
        detail::create_engine(engine, engine_id, operation_graph->get_raw_desc(), device_properties));

    // Create an array of knob choices
    std::vector<detail::backend_descriptor> knob_choices;
    CHECK_CUDNN_FRONTEND_ERROR(detail::set_knob_choices(user_knobs, knob_choices));

    auto engine_config = make_shared_backend_pointer((cudnnBackendDescriptorType_t)CUDNN_BACKEND_ENGINECFG_DESCRIPTOR);
    CHECK_CUDNN_FRONTEND_ERROR(detail::create_engine_config(engine_config, engine, knob_choices));
    plans.enqueue_engine_configs({engine_config});
    CHECK_CUDNN_FRONTEND_ERROR(plans.query_properties());

    CUDNN_FE_LOG_BANNER("  CREATE EXECUTION PLAN ALL OK  ");

    return {error_code_t::OK, ""};
}

inline error_t
Graph::build_plan_at_index(int64_t plan_index) {
    CHECK_CUDNN_FRONTEND_ERROR(plans.build_plan_at_index(plan_index));
    return {error_code_t::OK, ""};
}

inline error_t
Graph::build_plans(BuildPlanPolicy_t const policy, bool const do_multithreaded_builds) {
#ifndef CUDNN_FRONTEND_SKIP_JSON_LIB
    CUDNN_FE_LOG_BANNER("  BUILD PLANS  for policy " << nlohmann::json(policy).dump() << "  ");
#else
    CUDNN_FE_LOG_BANNER("  BUILD PLANS  for policy " << static_cast<int>(policy) << "  ");
#endif

    // Build OSS SDPA engine if it passed check_support
    if (plans.has_oss_sdpa_engine()) {
        auto oss_status = plans.build_oss_sdpa_engine();
        if (oss_status.is_good()) {
            CUDNN_FE_LOG_LABEL_ENDL("INFO: OSS SDPA engine built successfully (NVRTC compilation done)");
            if (policy == BuildPlanPolicy_t::HEURISTICS_CHOICE) {
                CUDNN_FE_LOG_BANNER("  BUILD PLANS ALL OK (OSS SDPA engine)  ");
                return {error_code_t::OK, ""};
            }
        } else {
            CUDNN_FE_LOG_LABEL_ENDL("WARN: OSS SDPA engine build failed: " << oss_status.get_message());
        }
    }

    // Build OSS RmsNorm+SiLU engine if it passed check_support
    if (plans.has_oss_rms_norm_silu_engine()) {
        auto oss_status = plans.build_oss_rms_norm_silu_engine();
        if (oss_status.is_good()) {
            CUDNN_FE_LOG_LABEL_ENDL("INFO: OSS RmsNorm+SiLU engine built successfully (NVRTC compilation done)");
            if (policy == BuildPlanPolicy_t::HEURISTICS_CHOICE) {
                CUDNN_FE_LOG_BANNER("  BUILD PLANS ALL OK (OSS RmsNorm+SiLU engine)  ");
                return {error_code_t::OK, ""};
            }
        } else {
            CUDNN_FE_LOG_LABEL_ENDL("WARN: OSS RmsNorm+SiLU engine build failed: " << oss_status.get_message());
        }
    }

    CHECK_CUDNN_FRONTEND_ERROR(plans.build_plans(policy, do_multithreaded_builds));

    // Prepare the variant pack template for fast execution.
    // This pre-computes slot indices so execute() can skip map operations.
    CHECK_CUDNN_FRONTEND_ERROR(prepare_variant_pack_template());

    CUDNN_FE_LOG_BANNER("  BUILD PLANS ALL OK  ");
    return {error_code_t::OK, ""};
}

inline error_t
Graph::build(cudnnHandle_t const &handle,
             std::vector<HeurMode_t> const &modes,
             BuildPlanPolicy_t const policy,
             bool const do_multithreaded_builds) {
#ifndef CUDNN_FRONTEND_SKIP_JSON_LIB
    CUDNN_FE_LOG_BANNER(" BUILD with handle " << nlohmann::json(policy).dump());
#else
    CUDNN_FE_LOG_BANNER(" BUILD with handle " << static_cast<int>(policy) << "  ");
#endif
    CHECK_CUDNN_FRONTEND_ERROR(this->validate());
    CHECK_CUDNN_FRONTEND_ERROR(this->build_operation_graph(handle));
    CHECK_CUDNN_FRONTEND_ERROR(this->create_execution_plans(modes));
    CHECK_CUDNN_FRONTEND_ERROR(this->check_support());
    CHECK_CUDNN_FRONTEND_ERROR(this->build_plans(policy, do_multithreaded_builds));
    CUDNN_FE_LOG_BANNER("  BUILD ALL OK (with handle) ");
    return {error_code_t::OK, ""};
}

inline error_t
Graph::build(std::vector<HeurMode_t> const &modes, BuildPlanPolicy_t const policy, bool const do_multithreaded_builds) {
#ifndef CUDNN_FRONTEND_SKIP_JSON_LIB
    CUDNN_FE_LOG_BANNER("  BUILD PLANS  without handle " << nlohmann::json(policy).dump() << "  ");
#else
    CUDNN_FE_LOG_BANNER("  BUILD PLANS  without handle " << static_cast<int>(policy) << "  ");
#endif
    CHECK_CUDNN_FRONTEND_ERROR(this->validate());
    CHECK_CUDNN_FRONTEND_ERROR(this->build_operation_graph());
    CHECK_CUDNN_FRONTEND_ERROR(this->create_execution_plans(modes));
    CHECK_CUDNN_FRONTEND_ERROR(this->check_support());
    CHECK_CUDNN_FRONTEND_ERROR(this->build_plans(policy, do_multithreaded_builds));
    CUDNN_FE_LOG_BANNER("  BUILD PLANS ALL OK (no handle) ");
    return {error_code_t::OK, ""};
}

inline Graph &
Graph::set_intermediate_data_type(DataType_t const type) {
    context.set_intermediate_data_type(type);
    return *this;
}

inline Graph &
Graph::set_io_data_type(DataType_t const type) {
    context.set_io_data_type(type);
    return *this;
}

inline Graph &
Graph::set_compute_data_type(DataType_t const type) {
    context.set_compute_data_type(type);
    return *this;
}

inline Graph &
Graph::set_dynamic_shape_enabled(bool is_enabled) {
    context.set_dynamic_shape_enabled(is_enabled);
    this->is_dynamic_shape_enabled = is_enabled;
    return *this;
}

inline Graph &
Graph::set_override_shape_enabled(bool is_enabled) {
    context.set_override_shape_enabled(is_enabled);
    this->is_override_shape_enabled = is_enabled;
    return *this;
}

inline Graph &
Graph::set_kernel_cache(std::shared_ptr<KernelCache> cache) {
    kernel_cache = cache;
    return *this;
}

inline Graph &
Graph::set_device_properties(std::shared_ptr<const DeviceProperties> device_prop) {
    device_properties = device_prop;
    return *this;
}

inline Graph &
Graph::set_sm_count(int32_t count) {
    context.set_target_sm_count(count);
    return *this;
}

inline Graph &
Graph::set_sm_version(int32_t version) {
    context.set_sm_version(version);
    return *this;
}

inline std::shared_ptr<Tensor_attributes>
Graph::tensor(Tensor_attributes const &tensor) {
    auto tensor_ptr = std::make_shared<Tensor_attributes>(tensor);
    full_graph_inputs.emplace(tensor_ptr);
    return tensor_ptr;
}

// Overloaded tensor() methods for compile-time constants
inline std::shared_ptr<Tensor_attributes>
Graph::tensor(float const &scalar, ScalarType scalar_type) {
    auto tensor_ptr = std::make_shared<Tensor_attributes>(scalar, scalar_type);
    full_graph_inputs.emplace(tensor_ptr);
    return tensor_ptr;
}

inline std::shared_ptr<Tensor_attributes>
Graph::tensor(half const &scalar, ScalarType scalar_type) {
    auto tensor_ptr = std::make_shared<Tensor_attributes>(scalar, scalar_type);
    full_graph_inputs.emplace(tensor_ptr);
    return tensor_ptr;
}

inline std::shared_ptr<Tensor_attributes>
Graph::tensor(nv_bfloat16 const &scalar, ScalarType scalar_type) {
    auto tensor_ptr = std::make_shared<Tensor_attributes>(scalar, scalar_type);
    full_graph_inputs.emplace(tensor_ptr);
    return tensor_ptr;
}

inline std::shared_ptr<Tensor_attributes>
Graph::tensor(int32_t const &scalar, ScalarType scalar_type) {
    auto tensor_ptr = std::make_shared<Tensor_attributes>(scalar, scalar_type);
    full_graph_inputs.emplace(tensor_ptr);
    return tensor_ptr;
}

inline std::shared_ptr<Tensor_attributes>
Graph::tensor(int64_t const &scalar, ScalarType scalar_type) {
    auto tensor_ptr = std::make_shared<Tensor_attributes>(scalar, scalar_type);
    full_graph_inputs.emplace(tensor_ptr);
    return tensor_ptr;
}

inline std::shared_ptr<Tensor_attributes>
Graph::tensor(double const &scalar, ScalarType scalar_type) {
    auto tensor_ptr = std::make_shared<Tensor_attributes>(scalar, scalar_type);
    full_graph_inputs.emplace(tensor_ptr);
    return tensor_ptr;
}

inline error_t
Graph::query_tensor_attributes_of_uid(int64_t const uid, Tensor_attributes &tensor) const {
    for (auto const &o_tensor : full_graph_outputs) {
        if (uid == o_tensor->get_uid()) {
            tensor = *o_tensor;
            return {error_code_t::OK, ""};
        }
    }

    for (auto const &i_tensor : full_graph_inputs) {
        if (uid == i_tensor->get_uid()) {
            tensor = *i_tensor;
            return {error_code_t::OK, ""};
        }
    }

    for (auto const &d_tensor : deserialized_tensor_properties) {
        if (uid == d_tensor->get_uid()) {
            tensor = *d_tensor;
            return {error_code_t::OK, ""};
        }
    }

    return {error_code_t::INVALID_VALUE, "No matching tensor for this UID"};
}

// tensor_like is meant to create "useable" copies of a tensor.
// By usable, it means not copying over the uids, as uids are FE-level(internal) detail.
// It also means not copying over names, which are user-level(external) detail. But user is given option to provide a
// new name.
inline std::shared_ptr<Tensor_attributes>
Graph::tensor_like(std::shared_ptr<Tensor_attributes> const &tensor, std::string const &name) {
    auto tensor_ptr = std::make_shared<Tensor_attributes>(*tensor);

    // reset the uid of the cloned tensor
    // uids are not meant to be copied by tensor_like
    // When lowering to cudnn backend, both tensors involved here will get unique uids.
    tensor_ptr->clear_uid();

    // reset the name too. Defaults to empty string.
    tensor_ptr->set_name(name);
    full_graph_inputs.emplace(tensor_ptr);

    return tensor_ptr;
}

inline std::array<std::shared_ptr<Tensor_attributes>, 6>
Graph::bn_finalize(std::shared_ptr<Tensor_attributes> sum,
                   std::shared_ptr<Tensor_attributes> sq_sum,
                   std::shared_ptr<Tensor_attributes> scale,
                   std::shared_ptr<Tensor_attributes> bias,
                   std::shared_ptr<Tensor_attributes> epsilon,
                   std::shared_ptr<Tensor_attributes> accum_count,
                   BN_finalize_attributes attributes) {
    // Set outputs
    auto EQ_SCALE = attributes.outputs[BN_finalize_attributes::output_names::EQ_SCALE] =
        output_tensor(attributes.name + "::EQ_SCALE");
    auto EQ_BIAS = attributes.outputs[BN_finalize_attributes::output_names::EQ_BIAS] =
        output_tensor(attributes.name + "::EQ_BIAS");
    auto MEAN = attributes.outputs[BN_finalize_attributes::output_names::MEAN] =
        output_tensor(attributes.name + "::MEAN");
    auto INV_VARIANCE = attributes.outputs[BN_finalize_attributes::output_names::INV_VARIANCE] =
        output_tensor(attributes.name + "::INV_VARIANCE");
    std::shared_ptr<Tensor_attributes> NEXT_RUNNING_MEAN = nullptr;
    std::shared_ptr<Tensor_attributes> NEXT_RUNNING_VAR  = nullptr;
    if (attributes.inputs[BN_finalize_attributes::input_names::PREV_RUNNING_MEAN] &&
        attributes.inputs[BN_finalize_attributes::input_names::PREV_RUNNING_VAR] &&
        attributes.inputs[BN_finalize_attributes::input_names::MOMENTUM]) {
        NEXT_RUNNING_MEAN = output_tensor(attributes.name + "::NEXT_RUNNING_MEAN");
        NEXT_RUNNING_VAR  = output_tensor(attributes.name + "::NEXT_RUNNING_VAR");
    }
    attributes.outputs[BN_finalize_attributes::output_names::NEXT_RUNNING_MEAN] = NEXT_RUNNING_MEAN;
    attributes.outputs[BN_finalize_attributes::output_names::NEXT_RUNNING_VAR]  = NEXT_RUNNING_VAR;

    // Set inputs
    attributes.inputs[BN_finalize_attributes::input_names::SUM]         = sum;
    attributes.inputs[BN_finalize_attributes::input_names::SQ_SUM]      = sq_sum;
    attributes.inputs[BN_finalize_attributes::input_names::SCALE]       = scale;
    attributes.inputs[BN_finalize_attributes::input_names::BIAS]        = bias;
    attributes.inputs[BN_finalize_attributes::input_names::EPSILON]     = epsilon;
    attributes.inputs[BN_finalize_attributes::input_names::ACCUM_COUNT] = accum_count;

    sub_nodes.emplace_back(std::make_unique<BatchNormFinalizeNode>(std::move(attributes), context));

    return {EQ_SCALE, EQ_BIAS, MEAN, INV_VARIANCE, NEXT_RUNNING_MEAN, NEXT_RUNNING_VAR};
}

inline std::array<std::shared_ptr<Tensor_attributes>, 3>
Graph::layernorm(std::shared_ptr<Tensor_attributes> x,
                 std::shared_ptr<Tensor_attributes> scale,
                 std::shared_ptr<Tensor_attributes> bias,
                 Layernorm_attributes attributes) {
    // Set outputs
    auto Y = attributes.outputs[Layernorm_attributes::output_names::Y] = output_tensor(attributes.name + "::Y");
    std::shared_ptr<Tensor_attributes> MEAN                            = nullptr;
    std::shared_ptr<Tensor_attributes> INV_VARIANCE                    = nullptr;
    if (attributes.forward_phase == NormFwdPhase_t::TRAINING) {
        MEAN = attributes.outputs[Layernorm_attributes::output_names::MEAN] = output_tensor(attributes.name + "::MEAN");
        INV_VARIANCE = attributes.outputs[Layernorm_attributes::output_names::INV_VARIANCE] =
            output_tensor(attributes.name + "::INV_VARIANCE");
    }
    // Set inputs
    attributes.inputs[Layernorm_attributes::input_names::X]     = x;
    attributes.inputs[Layernorm_attributes::input_names::SCALE] = scale;
    attributes.inputs[Layernorm_attributes::input_names::BIAS]  = bias;

    sub_nodes.emplace_back(std::make_unique<LayerNormNode>(std::move(attributes), context));

    return {Y, MEAN, INV_VARIANCE};
}

inline std::array<std::shared_ptr<Tensor_attributes>, 3>
Graph::adalayernorm(std::shared_ptr<Tensor_attributes> x,
                    std::shared_ptr<Tensor_attributes> scale,
                    std::shared_ptr<Tensor_attributes> bias,
                    AdaLayernorm_attributes attributes) {
    // Set outputs
    auto Y = attributes.outputs[AdaLayernorm_attributes::output_names::Y] = output_tensor(attributes.name + "::Y");
    std::shared_ptr<Tensor_attributes> MEAN                               = nullptr;
    std::shared_ptr<Tensor_attributes> INV_VARIANCE                       = nullptr;
    if (attributes.forward_phase == NormFwdPhase_t::TRAINING) {
        MEAN = attributes.outputs[AdaLayernorm_attributes::output_names::MEAN] =
            output_tensor(attributes.name + "::MEAN");
        INV_VARIANCE = attributes.outputs[AdaLayernorm_attributes::output_names::INV_VARIANCE] =
            output_tensor(attributes.name + "::INV_VARIANCE");
    }
    // Set inputs
    attributes.inputs[AdaLayernorm_attributes::input_names::X]     = x;
    attributes.inputs[AdaLayernorm_attributes::input_names::SCALE] = scale;
    attributes.inputs[AdaLayernorm_attributes::input_names::BIAS]  = bias;

    sub_nodes.emplace_back(std::make_unique<AdaLayerNormNode>(std::move(attributes), context));

    return {std::move(Y), std::move(MEAN), std::move(INV_VARIANCE)};
}

inline std::array<std::shared_ptr<Tensor_attributes>, 3>
Graph::instancenorm(std::shared_ptr<Tensor_attributes> x,
                    std::shared_ptr<Tensor_attributes> scale,
                    std::shared_ptr<Tensor_attributes> bias,
                    Instancenorm_attributes attributes) {
    // Set outputs
    auto Y = attributes.outputs[Instancenorm_attributes::output_names::Y] = output_tensor(attributes.name + "::Y");
    std::shared_ptr<Tensor_attributes> MEAN                               = nullptr;
    std::shared_ptr<Tensor_attributes> INV_VARIANCE                       = nullptr;
    if (attributes.forward_phase == NormFwdPhase_t::TRAINING) {
        MEAN = attributes.outputs[Instancenorm_attributes::output_names::MEAN] =
            output_tensor(attributes.name + "::MEAN");
        INV_VARIANCE = attributes.outputs[Instancenorm_attributes::output_names::INV_VARIANCE] =
            output_tensor(attributes.name + "::INV_VARIANCE");
    }
    // Set inputs
    attributes.inputs[Instancenorm_attributes::input_names::X]     = x;
    attributes.inputs[Instancenorm_attributes::input_names::SCALE] = scale;
    attributes.inputs[Instancenorm_attributes::input_names::BIAS]  = bias;

    sub_nodes.emplace_back(std::make_unique<InstanceNormNode>(std::move(attributes), context));

    return {Y, MEAN, INV_VARIANCE};
}

inline std::array<std::shared_ptr<Tensor_attributes>, 5>
Graph::batchnorm(std::shared_ptr<Tensor_attributes> x,
                 std::shared_ptr<Tensor_attributes> scale,
                 std::shared_ptr<Tensor_attributes> bias,
                 Batchnorm_attributes attributes) {
    // Set outputs
    auto Y = attributes.outputs[Batchnorm_attributes::output_names::Y] = output_tensor(attributes.name + "::Y");
    auto MEAN = attributes.outputs[Batchnorm_attributes::output_names::MEAN] =
        output_tensor(attributes.name + "::MEAN");
    auto INV_VARIANCE = attributes.outputs[Batchnorm_attributes::output_names::INV_VARIANCE] =
        output_tensor(attributes.name + "::INV_VARIANCE");
    std::shared_ptr<Tensor_attributes> NEXT_RUNNING_MEAN = nullptr;
    std::shared_ptr<Tensor_attributes> NEXT_RUNNING_VAR  = nullptr;
    if (attributes.inputs[Batchnorm_attributes::input_names::PREV_RUNNING_MEAN] &&
        attributes.inputs[Batchnorm_attributes::input_names::PREV_RUNNING_VAR] &&
        attributes.inputs[Batchnorm_attributes::input_names::MOMENTUM]) {
        NEXT_RUNNING_MEAN = output_tensor(attributes.name + "::NEXT_RUNNING_MEAN");
        NEXT_RUNNING_VAR  = output_tensor(attributes.name + "::NEXT_RUNNING_VAR");
    }
    attributes.outputs[Batchnorm_attributes::output_names::NEXT_RUNNING_MEAN] = NEXT_RUNNING_MEAN;
    attributes.outputs[Batchnorm_attributes::output_names::NEXT_RUNNING_VAR]  = NEXT_RUNNING_VAR;

    // Set inputs
    attributes.inputs[Batchnorm_attributes::input_names::X]     = x;
    attributes.inputs[Batchnorm_attributes::input_names::SCALE] = scale;
    attributes.inputs[Batchnorm_attributes::input_names::BIAS]  = bias;

    sub_nodes.emplace_back(std::make_unique<BatchNormNode>(std::move(attributes), context));

    return {Y, MEAN, INV_VARIANCE, NEXT_RUNNING_MEAN, NEXT_RUNNING_VAR};
}

inline std::shared_ptr<Tensor_attributes>
Graph::batchnorm_inference(std::shared_ptr<Tensor_attributes> x,
                           std::shared_ptr<Tensor_attributes> mean,
                           std::shared_ptr<Tensor_attributes> inv_variance,
                           std::shared_ptr<Tensor_attributes> scale,
                           std::shared_ptr<Tensor_attributes> bias,
                           Batchnorm_inference_attributes attributes) {
    // Set outputs
    auto Y = attributes.outputs[Batchnorm_inference_attributes::output_names::Y] =
        output_tensor(attributes.name + "::Y");

    // Set inputs
    attributes.inputs[Batchnorm_inference_attributes::input_names::X]            = x;
    attributes.inputs[Batchnorm_inference_attributes::input_names::MEAN]         = mean;
    attributes.inputs[Batchnorm_inference_attributes::input_names::INV_VARIANCE] = inv_variance;
    attributes.inputs[Batchnorm_inference_attributes::input_names::SCALE]        = scale;
    attributes.inputs[Batchnorm_inference_attributes::input_names::BIAS]         = bias;

    sub_nodes.emplace_back(std::make_unique<BatchnormInferenceNode>(std::move(attributes), context));

    return Y;
}

inline std::array<std::shared_ptr<Tensor_attributes>, 3>
Graph::batchnorm_backward(std::shared_ptr<Tensor_attributes> dy,
                          std::shared_ptr<Tensor_attributes> x,
                          std::shared_ptr<Tensor_attributes> scale,
                          Batchnorm_backward_attributes attributes) {
    // Set outputs
    auto DX = attributes.outputs[Batchnorm_backward_attributes::output_names::DX] =
        output_tensor(attributes.name + "::DX");
    auto DSCALE = attributes.outputs[Batchnorm_backward_attributes::output_names::DSCALE] =
        output_tensor(attributes.name + "::DSCALE");
    auto DBIAS = attributes.outputs[Batchnorm_backward_attributes::output_names::DBIAS] =
        output_tensor(attributes.name + "::DBIAS");

    // Set inputs
    attributes.inputs[Batchnorm_backward_attributes::input_names::DY]    = dy;
    attributes.inputs[Batchnorm_backward_attributes::input_names::X]     = x;
    attributes.inputs[Batchnorm_backward_attributes::input_names::SCALE] = scale;

    sub_nodes.emplace_back(std::make_unique<DBNNode>(std::move(attributes), context));

    return {DX, DSCALE, DBIAS};
}

inline std::array<std::shared_ptr<Tensor_attributes>, 3>
Graph::instancenorm_backward(std::shared_ptr<Tensor_attributes> dy,
                             std::shared_ptr<Tensor_attributes> x,
                             std::shared_ptr<Tensor_attributes> scale,
                             Instancenorm_backward_attributes attributes) {
    // Set outputs
    auto DX = attributes.outputs[Instancenorm_backward_attributes::output_names::DX] =
        output_tensor(attributes.name + "::DX");
    auto DSCALE = attributes.outputs[Instancenorm_backward_attributes::output_names::DSCALE] =
        output_tensor(attributes.name + "::DSCALE");
    auto DBIAS = attributes.outputs[Instancenorm_backward_attributes::output_names::DBIAS] =
        output_tensor(attributes.name + "::DBIAS");

    // Set inputs
    attributes.inputs[Instancenorm_backward_attributes::input_names::DY]    = dy;
    attributes.inputs[Instancenorm_backward_attributes::input_names::X]     = x;
    attributes.inputs[Instancenorm_backward_attributes::input_names::SCALE] = scale;

    sub_nodes.emplace_back(std::make_unique<DINNode>(std::move(attributes), context));

    return {DX, DSCALE, DBIAS};
}

inline std::array<std::shared_ptr<Tensor_attributes>, 3>
Graph::layernorm_backward(std::shared_ptr<Tensor_attributes> dy,
                          std::shared_ptr<Tensor_attributes> x,
                          std::shared_ptr<Tensor_attributes> scale,
                          Layernorm_backward_attributes attributes) {
    // Set outputs
    auto DX = attributes.outputs[Layernorm_backward_attributes::output_names::DX] =
        output_tensor(attributes.name + "::DX");
    auto DSCALE = attributes.outputs[Layernorm_backward_attributes::output_names::DSCALE] =
        output_tensor(attributes.name + "::DSCALE");
    auto DBIAS = attributes.outputs[Layernorm_backward_attributes::output_names::DBIAS] =
        output_tensor(attributes.name + "::DBIAS");

    // Set inputs
    attributes.inputs[Layernorm_backward_attributes::input_names::DY]    = dy;
    attributes.inputs[Layernorm_backward_attributes::input_names::X]     = x;
    attributes.inputs[Layernorm_backward_attributes::input_names::SCALE] = scale;

    sub_nodes.emplace_back(std::make_unique<DLNNode>(std::move(attributes), context));

    return {DX, DSCALE, DBIAS};
}

inline std::array<std::shared_ptr<Tensor_attributes>, 3>
Graph::adalayernorm_backward(std::shared_ptr<Tensor_attributes> dy,
                             std::shared_ptr<Tensor_attributes> x,
                             std::shared_ptr<Tensor_attributes> scale,
                             AdaLayernorm_backward_attributes attributes) {
    // Set outputs
    auto DX = attributes.outputs[AdaLayernorm_backward_attributes::output_names::DX] =
        output_tensor(attributes.name + "::DX");
    auto DSCALE = attributes.outputs[AdaLayernorm_backward_attributes::output_names::DSCALE] =
        output_tensor(attributes.name + "::DSCALE");
    auto DBIAS = attributes.outputs[AdaLayernorm_backward_attributes::output_names::DBIAS] =
        output_tensor(attributes.name + "::DBIAS");
    // Set inputs
    attributes.inputs[AdaLayernorm_backward_attributes::input_names::DY]    = dy;
    attributes.inputs[AdaLayernorm_backward_attributes::input_names::X]     = x;
    attributes.inputs[AdaLayernorm_backward_attributes::input_names::SCALE] = scale;

    sub_nodes.emplace_back(std::make_unique<DAdaLayerNormNode>(std::move(attributes), context));

    return {std::move(DX), std::move(DSCALE), std::move(DBIAS)};
}

inline std::shared_ptr<Tensor_attributes>
Graph::conv_fprop(std::shared_ptr<Tensor_attributes> x,
                  std::shared_ptr<Tensor_attributes> w,
                  Conv_fprop_attributes attributes) {
    // Make required output tensors
    if (attributes.name.empty()) {
        attributes.name += std::to_string(sub_nodes.size());
    }
    auto Y                                                     = output_tensor(attributes.name + "::Y");
    attributes.outputs[Conv_fprop_attributes::output_names::Y] = Y;

    // Set inputs
    attributes.inputs[Conv_fprop_attributes::input_names::X] = x;
    attributes.inputs[Conv_fprop_attributes::input_names::W] = w;

    sub_nodes.emplace_back(std::make_unique<ConvolutionNode>(std::move(attributes), context));

    return Y;
}

inline std::array<std::shared_ptr<Tensor_attributes>, 5>
Graph::dbn_weight(std::shared_ptr<Tensor_attributes> dy,
                  std::shared_ptr<Tensor_attributes> x,
                  std::shared_ptr<Tensor_attributes> mean,
                  std::shared_ptr<Tensor_attributes> inv_variance,
                  std::shared_ptr<Tensor_attributes> scale,
                  DBN_weight_attributes attributes) {
    if (attributes.name.empty()) {
        attributes.name += std::to_string(sub_nodes.size());
    }
    // Make required output tensors
    auto DBIAS = attributes.outputs[DBN_weight_attributes::output_names::DBIAS] =
        output_tensor(attributes.name + "::DBIAS");
    auto DSCALE = attributes.outputs[DBN_weight_attributes::output_names::DSCALE] =
        output_tensor(attributes.name + "::DSCALE");
    auto EQ_BIAS = attributes.outputs[DBN_weight_attributes::output_names::EQ_BIAS] =
        output_tensor(attributes.name + "::EQ_BIAS");
    auto EQ_SCALE_DY = attributes.outputs[DBN_weight_attributes::output_names::EQ_SCALE_DY] =
        output_tensor(attributes.name + "::EQ_SCALE_DY");
    auto EQ_SCALE_X = attributes.outputs[DBN_weight_attributes::output_names::EQ_SCALE_X] =
        output_tensor(attributes.name + "::EQ_SCALE_X");

    // Set inputs
    attributes.inputs[DBN_weight_attributes::input_names::DY]           = dy;
    attributes.inputs[DBN_weight_attributes::input_names::X]            = x;
    attributes.inputs[DBN_weight_attributes::input_names::SCALE]        = scale;
    attributes.inputs[DBN_weight_attributes::input_names::MEAN]         = mean;
    attributes.inputs[DBN_weight_attributes::input_names::INV_VARIANCE] = inv_variance;

    sub_nodes.emplace_back(std::make_unique<DBNWeightNode>(std::move(attributes), context));

    return {DSCALE, DBIAS, EQ_SCALE_DY, EQ_SCALE_X, EQ_BIAS};
}

inline std::shared_ptr<Tensor_attributes>
Graph::conv_dgrad(std::shared_ptr<Tensor_attributes> dy,
                  std::shared_ptr<Tensor_attributes> w,
                  Conv_dgrad_attributes attributes) {
    // Make required output tensors
    auto DX = attributes.outputs[Conv_dgrad_attributes::output_names::DX] = output_tensor(attributes.name + "::DX");

    // Set inputs
    attributes.inputs[Conv_dgrad_attributes::input_names::DY] = dy;
    attributes.inputs[Conv_dgrad_attributes::input_names::W]  = w;

    sub_nodes.emplace_back(std::make_unique<DgradNode>(std::move(attributes), context));

    return DX;
}

inline std::array<std::shared_ptr<Tensor_attributes>, 2>
Graph::genstats(std::shared_ptr<Tensor_attributes> x, Genstats_attributes attributes) {
    // Set outputs
    auto SUM = attributes.outputs[Genstats_attributes::output_names::SUM] =
        output_tensor(attributes.name + "_sum_output");
    auto SQ_SUM = attributes.outputs[Genstats_attributes::output_names::SQ_SUM] =
        output_tensor(attributes.name + "_sq_sum_output");

    // Set inputs
    attributes.inputs[Genstats_attributes::input_names::X] = x;

    sub_nodes.emplace_back(std::make_unique<GenstatsNode>(std::move(attributes), context));

    return {SUM, SQ_SUM};
}

inline std::shared_ptr<Tensor_attributes>
Graph::conv_wgrad(std::shared_ptr<Tensor_attributes> dy,
                  std::shared_ptr<Tensor_attributes> x,
                  Conv_wgrad_attributes attributes) {
    // Make required output tensors
    auto DW = attributes.outputs[Conv_wgrad_attributes::output_names::DW] = output_tensor(attributes.name + "::DW");

    // Set inputs
    attributes.inputs[Conv_wgrad_attributes::input_names::X]  = x;
    attributes.inputs[Conv_wgrad_attributes::input_names::DY] = dy;

    sub_nodes.emplace_back(std::make_unique<WgradNode>(std::move(attributes), context));

    return DW;
}

inline std::array<std::shared_ptr<Tensor_attributes>, 2>
Graph::rmsnorm(std::shared_ptr<Tensor_attributes> x,
               std::shared_ptr<Tensor_attributes> scale,
               Rmsnorm_attributes attributes) {
    // Set outputs
    auto Y = attributes.outputs[Rmsnorm_attributes::output_names::Y] = output_tensor(attributes.name + "::Y");
    std::shared_ptr<Tensor_attributes> INV_VARIANCE                  = nullptr;
    if (attributes.forward_phase == NormFwdPhase_t::TRAINING) {
        INV_VARIANCE = attributes.outputs[Rmsnorm_attributes::output_names::INV_VARIANCE] =
            output_tensor(attributes.name + "::INV_VARIANCE");
    }
    // Set inputs
    attributes.inputs[Rmsnorm_attributes::input_names::X]     = x;
    attributes.inputs[Rmsnorm_attributes::input_names::SCALE] = scale;

    sub_nodes.emplace_back(std::make_unique<RMSNormNode>(std::move(attributes), context));

    return {Y, INV_VARIANCE};
}

inline std::shared_ptr<Tensor_attributes>
Graph::rope(std::shared_ptr<Tensor_attributes> input,
            std::shared_ptr<Tensor_attributes> freqs,
            RoPE_attributes attributes) {
    // RoPE writes to a user-bound buffer (no longer a virtual workspace tensor),
    // because the rotated Q/K need to be saved across fwd→bwd for the bwd to consume them
    // as inputs to SDPA bwd.
    auto OUTPUT = attributes.outputs[RoPE_attributes::output_names::OUTPUT] =
        output_tensor(attributes.name + "::OUTPUT");
    OUTPUT->set_is_virtual(false);

    // Set inputs
    attributes.inputs[RoPE_attributes::input_names::INPUT] = input;
    attributes.inputs[RoPE_attributes::input_names::FREQS] = freqs;

    sub_nodes.emplace_back(std::make_unique<RoPENode>(std::move(attributes), context));

    return OUTPUT;
}

inline std::shared_ptr<Tensor_attributes>
Graph::rope_backward(std::shared_ptr<Tensor_attributes> dy,
                     std::shared_ptr<Tensor_attributes> freqs,
                     RoPE_backward_attributes attributes) {
    // dX is a real (user-bound) tensor by default — symmetric to fwd RoPE's Y output.
    auto DX = attributes.outputs[RoPE_backward_attributes::output_names::DX] = output_tensor(attributes.name + "::DX");
    DX->set_is_virtual(false);

    attributes.inputs[RoPE_backward_attributes::input_names::DY]    = dy;
    attributes.inputs[RoPE_backward_attributes::input_names::FREQS] = freqs;

    sub_nodes.emplace_back(std::make_unique<RoPEBackwardNode>(std::move(attributes), context));

    return DX;
}

inline std::array<std::shared_ptr<Tensor_attributes>, 3>
Graph::rmsnorm_backward(std::shared_ptr<Tensor_attributes> dy,
                        std::shared_ptr<Tensor_attributes> x,
                        std::shared_ptr<Tensor_attributes> scale,
                        std::shared_ptr<Tensor_attributes> inv_variance,
                        Rmsnorm_backward_attributes attributes) {
    // Set outputs
    auto DX = attributes.outputs[Rmsnorm_backward_attributes::output_names::DX] =
        output_tensor(attributes.name + "::DX");
    auto DScale = attributes.outputs[Rmsnorm_backward_attributes::output_names::DSCALE] =
        output_tensor(attributes.name + "::Dscale");
    std::shared_ptr<Tensor_attributes> DBias = nullptr;
    if (attributes.use_dbias.value_or(true)) {
        DBias = attributes.outputs[Rmsnorm_backward_attributes::output_names::DBIAS] =
            output_tensor(attributes.name + "::Dbias");
    }

    // Set inputs
    attributes.inputs[Rmsnorm_backward_attributes::input_names::DY]           = dy;
    attributes.inputs[Rmsnorm_backward_attributes::input_names::X]            = x;
    attributes.inputs[Rmsnorm_backward_attributes::input_names::SCALE]        = scale;
    attributes.inputs[Rmsnorm_backward_attributes::input_names::INV_VARIANCE] = inv_variance;

    sub_nodes.emplace_back(std::make_unique<DRMSNormNode>(std::move(attributes), context));

    return {DX, DScale, DBias};
}

inline std::array<std::shared_ptr<Tensor_attributes>, 2>
Graph::sdpa(std::shared_ptr<Tensor_attributes> q,
            std::shared_ptr<Tensor_attributes> k,
            std::shared_ptr<Tensor_attributes> v,
            SDPA_attributes attributes) {
    if (attributes.mma_core_mode == DataType_t::NOT_SET) {
        attributes._set_mma_core_mode(DataType_t::HALF);
    }

    // Call internal implementation and return only the O and Stats outputs for backward compatibility
    auto internal_result = sdpa_internal(q, k, v, std::move(attributes));
    return {internal_result.O, internal_result.Stats};
}

inline std::array<std::shared_ptr<Tensor_attributes>, 4>
Graph::sdpa_fp8(std::shared_ptr<Tensor_attributes> q,
                std::shared_ptr<Tensor_attributes> k,
                std::shared_ptr<Tensor_attributes> v,
                std::shared_ptr<Tensor_attributes> descale_q,
                std::shared_ptr<Tensor_attributes> descale_k,
                std::shared_ptr<Tensor_attributes> descale_v,
                std::shared_ptr<Tensor_attributes> descale_s,
                std::shared_ptr<Tensor_attributes> scale_s,
                std::shared_ptr<Tensor_attributes> scale_o,
                SDPA_fp8_attributes attributes) {
    if (attributes.mma_core_mode == DataType_t::NOT_SET) {
        attributes._set_mma_core_mode(DataType_t::FP8_E4M3);
    }

    // Set FP8 scaling inputs
    attributes.inputs[SDPA_fp8_attributes::input_names::Descale_Q] = descale_q;
    attributes.inputs[SDPA_fp8_attributes::input_names::Descale_K] = descale_k;
    attributes.inputs[SDPA_fp8_attributes::input_names::Descale_V] = descale_v;
    attributes.inputs[SDPA_fp8_attributes::input_names::Descale_S] = descale_s;
    attributes.inputs[SDPA_fp8_attributes::input_names::Scale_S]   = scale_s;
    attributes.inputs[SDPA_fp8_attributes::input_names::Scale_O]   = scale_o;

    // Call internal implementation and return {Output, Stats, Amax_S, Amax_O} as array for backward compatibility
    auto internal_result = sdpa_internal(q, k, v, std::move(attributes));
    return {internal_result.O, internal_result.Stats, internal_result.Amax_S, internal_result.Amax_O};
}

inline std::array<std::shared_ptr<Tensor_attributes>, 3>
Graph::sdpa_fp8(std::shared_ptr<Tensor_attributes> q,
                std::shared_ptr<Tensor_attributes> k,
                std::shared_ptr<Tensor_attributes> v,
                std::shared_ptr<Tensor_attributes> descale_q,
                std::shared_ptr<Tensor_attributes> descale_k,
                std::shared_ptr<Tensor_attributes> descale_v,
                SDPA_fp8_attributes attributes) {
    if (attributes.mma_core_mode == DataType_t::NOT_SET) {
        attributes._set_mma_core_mode(DataType_t::FP8_E4M3);
    }

    // Set FP8 scaling inputs
    attributes.inputs[SDPA_fp8_attributes::input_names::Descale_Q] = descale_q;
    attributes.inputs[SDPA_fp8_attributes::input_names::Descale_K] = descale_k;
    attributes.inputs[SDPA_fp8_attributes::input_names::Descale_V] = descale_v;

    // Call internal implementation and return {Output, Stats, Amax_O} as array for backward compatibility
    auto internal_result = sdpa_internal(q, k, v, std::move(attributes));
    return {internal_result.O, internal_result.Stats, internal_result.Amax_O};
}

inline std::array<std::shared_ptr<Tensor_attributes>, 7>
Graph::sdpa_fp8_backward(std::shared_ptr<Tensor_attributes> q,
                         std::shared_ptr<Tensor_attributes> k,
                         std::shared_ptr<Tensor_attributes> v,
                         std::shared_ptr<Tensor_attributes> o,
                         std::shared_ptr<Tensor_attributes> dO,
                         std::shared_ptr<Tensor_attributes> Stats,
                         std::shared_ptr<Tensor_attributes> descale_q,
                         std::shared_ptr<Tensor_attributes> descale_k,
                         std::shared_ptr<Tensor_attributes> descale_v,
                         std::shared_ptr<Tensor_attributes> descale_o,
                         std::shared_ptr<Tensor_attributes> descale_do,
                         std::shared_ptr<Tensor_attributes> descale_s,
                         std::shared_ptr<Tensor_attributes> descale_dp,
                         std::shared_ptr<Tensor_attributes> scale_s,
                         std::shared_ptr<Tensor_attributes> scale_dq,
                         std::shared_ptr<Tensor_attributes> scale_dk,
                         std::shared_ptr<Tensor_attributes> scale_dv,
                         std::shared_ptr<Tensor_attributes> scale_dp,
                         SDPA_fp8_backward_attributes attributes) {
    // Make required output tensors
    auto dQ = attributes.outputs[SDPA_fp8_backward_attributes::output_names::dQ] =
        output_tensor(attributes.name + "::dQ");
    auto dK = attributes.outputs[SDPA_fp8_backward_attributes::output_names::dK] =
        output_tensor(attributes.name + "::dK");
    auto dV = attributes.outputs[SDPA_fp8_backward_attributes::output_names::dV] =
        output_tensor(attributes.name + "::dV");
    auto Amax_dQ = attributes.outputs[SDPA_fp8_backward_attributes::output_names::Amax_dQ] =
        output_tensor(attributes.name + "::Amax_dQ");
    auto Amax_dK = attributes.outputs[SDPA_fp8_backward_attributes::output_names::Amax_dK] =
        output_tensor(attributes.name + "::Amax_dK");
    auto Amax_dV = attributes.outputs[SDPA_fp8_backward_attributes::output_names::Amax_dV] =
        output_tensor(attributes.name + "::Amax_dV");
    auto Amax_dP = attributes.outputs[SDPA_fp8_backward_attributes::output_names::Amax_dP] =
        output_tensor(attributes.name + "::Amax_dP");

    // Set inputs
    attributes.inputs[SDPA_fp8_backward_attributes::input_names::Q]     = q;
    attributes.inputs[SDPA_fp8_backward_attributes::input_names::K]     = k;
    attributes.inputs[SDPA_fp8_backward_attributes::input_names::V]     = v;
    attributes.inputs[SDPA_fp8_backward_attributes::input_names::O]     = o;
    attributes.inputs[SDPA_fp8_backward_attributes::input_names::Stats] = Stats;
    attributes.inputs[SDPA_fp8_backward_attributes::input_names::dO]    = dO;

    attributes.inputs[SDPA_fp8_backward_attributes::input_names::Descale_Q]  = descale_q;
    attributes.inputs[SDPA_fp8_backward_attributes::input_names::Descale_K]  = descale_k;
    attributes.inputs[SDPA_fp8_backward_attributes::input_names::Descale_V]  = descale_v;
    attributes.inputs[SDPA_fp8_backward_attributes::input_names::Descale_S]  = descale_s;
    attributes.inputs[SDPA_fp8_backward_attributes::input_names::Descale_O]  = descale_o;
    attributes.inputs[SDPA_fp8_backward_attributes::input_names::Descale_dO] = descale_do;
    attributes.inputs[SDPA_fp8_backward_attributes::input_names::Descale_dP] = descale_dp;

    attributes.inputs[SDPA_fp8_backward_attributes::input_names::Scale_dQ] = scale_dq;
    attributes.inputs[SDPA_fp8_backward_attributes::input_names::Scale_dK] = scale_dk;
    attributes.inputs[SDPA_fp8_backward_attributes::input_names::Scale_dV] = scale_dv;
    attributes.inputs[SDPA_fp8_backward_attributes::input_names::Scale_S]  = scale_s;
    attributes.inputs[SDPA_fp8_backward_attributes::input_names::Scale_dP] = scale_dp;

    sub_nodes.emplace_back(std::make_unique<SDPAFP8BackwardNode>(std::move(attributes), context));

    return {dQ, dK, dV, Amax_dQ, Amax_dK, Amax_dV, Amax_dP};
}

inline std::array<std::shared_ptr<Tensor_attributes>, 6>
Graph::sdpa_fp8_backward(std::shared_ptr<Tensor_attributes> q,
                         std::shared_ptr<Tensor_attributes> q_T,
                         std::shared_ptr<Tensor_attributes> k,
                         std::shared_ptr<Tensor_attributes> k_T,
                         std::shared_ptr<Tensor_attributes> v,
                         std::shared_ptr<Tensor_attributes> o_f16,
                         std::shared_ptr<Tensor_attributes> dO_f16,
                         std::shared_ptr<Tensor_attributes> dO,
                         std::shared_ptr<Tensor_attributes> dO_T,
                         std::shared_ptr<Tensor_attributes> Stats,
                         std::shared_ptr<Tensor_attributes> descale_q,
                         std::shared_ptr<Tensor_attributes> descale_q_T,
                         std::shared_ptr<Tensor_attributes> descale_k,
                         std::shared_ptr<Tensor_attributes> descale_k_T,
                         std::shared_ptr<Tensor_attributes> descale_v,
                         std::shared_ptr<Tensor_attributes> descale_dO,
                         std::shared_ptr<Tensor_attributes> descale_dO_T,
                         SDPA_fp8_backward_attributes attributes) {
    // Make required output tensors
    auto dQ = attributes.outputs[SDPA_fp8_backward_attributes::output_names::dQ] =
        output_tensor(attributes.name + "::dQ");
    auto dK = attributes.outputs[SDPA_fp8_backward_attributes::output_names::dK] =
        output_tensor(attributes.name + "::dK");
    auto dV = attributes.outputs[SDPA_fp8_backward_attributes::output_names::dV] =
        output_tensor(attributes.name + "::dV");
    auto Amax_dQ = attributes.outputs[SDPA_fp8_backward_attributes::output_names::Amax_dQ] =
        output_tensor(attributes.name + "::Amax_dQ");
    auto Amax_dK = attributes.outputs[SDPA_fp8_backward_attributes::output_names::Amax_dK] =
        output_tensor(attributes.name + "::Amax_dK");
    auto Amax_dV = attributes.outputs[SDPA_fp8_backward_attributes::output_names::Amax_dV] =
        output_tensor(attributes.name + "::Amax_dV");

    // Set inputs
    attributes.inputs[SDPA_fp8_backward_attributes::input_names::Q]      = q;
    attributes.inputs[SDPA_fp8_backward_attributes::input_names::K]      = k;
    attributes.inputs[SDPA_fp8_backward_attributes::input_names::V]      = v;
    attributes.inputs[SDPA_fp8_backward_attributes::input_names::Q_T]    = q_T;
    attributes.inputs[SDPA_fp8_backward_attributes::input_names::K_T]    = k_T;
    attributes.inputs[SDPA_fp8_backward_attributes::input_names::O]      = o_f16;
    attributes.inputs[SDPA_fp8_backward_attributes::input_names::dO_f16] = dO_f16;
    attributes.inputs[SDPA_fp8_backward_attributes::input_names::dO_T]   = dO_T;
    attributes.inputs[SDPA_fp8_backward_attributes::input_names::Stats]  = Stats;
    attributes.inputs[SDPA_fp8_backward_attributes::input_names::dO]     = dO;

    attributes.inputs[SDPA_fp8_backward_attributes::input_names::Descale_Q]    = descale_q;
    attributes.inputs[SDPA_fp8_backward_attributes::input_names::Descale_Q_T]  = descale_q_T;
    attributes.inputs[SDPA_fp8_backward_attributes::input_names::Descale_K]    = descale_k;
    attributes.inputs[SDPA_fp8_backward_attributes::input_names::Descale_K_T]  = descale_k_T;
    attributes.inputs[SDPA_fp8_backward_attributes::input_names::Descale_V]    = descale_v;
    attributes.inputs[SDPA_fp8_backward_attributes::input_names::Descale_dO]   = descale_dO;
    attributes.inputs[SDPA_fp8_backward_attributes::input_names::Descale_dO_T] = descale_dO_T;

    sub_nodes.emplace_back(std::make_unique<SDPAFP8BackwardNode>(std::move(attributes), context));

    return {dQ, dK, dV, Amax_dQ, Amax_dK, Amax_dV};
}

inline std::array<std::shared_ptr<Tensor_attributes>, 3>
Graph::sdpa_backward(std::shared_ptr<Tensor_attributes> q,
                     std::shared_ptr<Tensor_attributes> k,
                     std::shared_ptr<Tensor_attributes> v,
                     std::shared_ptr<Tensor_attributes> o,
                     std::shared_ptr<Tensor_attributes> dO,
                     std::shared_ptr<Tensor_attributes> stats,
                     SDPA_backward_attributes attributes) {
    // Set inputs
    attributes.inputs[SDPA_backward_attributes::input_names::Q]     = q;
    attributes.inputs[SDPA_backward_attributes::input_names::K]     = k;
    attributes.inputs[SDPA_backward_attributes::input_names::V]     = v;
    attributes.inputs[SDPA_backward_attributes::input_names::O]     = o;
    attributes.inputs[SDPA_backward_attributes::input_names::dO]    = dO;
    attributes.inputs[SDPA_backward_attributes::input_names::Stats] = stats;

    // Make required output tensors
    auto dQ = attributes.outputs[SDPA_backward_attributes::output_names::dQ] = output_tensor(attributes.name + "::dQ");
    auto dK = attributes.outputs[SDPA_backward_attributes::output_names::dK] = output_tensor(attributes.name + "::dK");
    auto dV = attributes.outputs[SDPA_backward_attributes::output_names::dV] = output_tensor(attributes.name + "::dV");

    sub_nodes.emplace_back(std::make_unique<CompositeSDPABackwardNode>(std::move(attributes), context));

    return {dQ, dK, dV};
}

inline std::shared_ptr<Tensor_attributes>
Graph::slice(std::shared_ptr<Tensor_attributes> input, Slice_attributes attributes) {
    attributes.inputs[Slice_attributes::input_names::X] = input;
    auto Y = attributes.outputs[Slice_attributes::output_names::Y] = output_tensor(attributes.name + "::Y");

    sub_nodes.emplace_back(std::make_unique<SliceNode>(std::move(attributes), context));
    return Y;
}

inline std::shared_ptr<Tensor_attributes>
Graph::transpose(std::shared_ptr<Tensor_attributes> input, Transpose_attributes attributes) {
    attributes.inputs[Transpose_attributes::input_names::X] = input;
    auto Y = attributes.outputs[Transpose_attributes::output_names::Y] = output_tensor(attributes.name + "::Y");

    sub_nodes.emplace_back(std::make_unique<TransposeNode>(std::move(attributes), context));
    return Y;
}

inline std::array<std::shared_ptr<Tensor_attributes>, 2>
Graph::block_scale_quantize(std::shared_ptr<Tensor_attributes> x, Block_scale_quantize_attributes attributes) {
    // Set outputs
    auto Y = attributes.outputs[Block_scale_quantize_attributes::output_names::Y] =
        output_tensor(attributes.name + "::Y");
    auto scale = attributes.outputs[Block_scale_quantize_attributes::output_names::scale] =
        output_tensor(attributes.name + "::scale");

    // Set inputs
    attributes.inputs[Block_scale_quantize_attributes::input_names::X] = x;

    sub_nodes.emplace_back(std::make_unique<BlockScaleQuantizeNode>(std::move(attributes), context));

    return {Y, scale};
}

inline std::shared_ptr<Tensor_attributes>
Graph::block_scale_dequantize(std::shared_ptr<Tensor_attributes> x,
                              std::shared_ptr<Tensor_attributes> scale,
                              Block_scale_dequantize_attributes attributes) {
    // Set outputs
    auto Y = attributes.outputs[Block_scale_dequantize_attributes::output_names::Y] =
        output_tensor(attributes.name + "::Y");

    // Set inputs
    attributes.inputs[Block_scale_dequantize_attributes::input_names::X]     = x;
    attributes.inputs[Block_scale_dequantize_attributes::input_names::scale] = scale;

    sub_nodes.emplace_back(std::make_unique<BlockScaleDequantizeNode>(std::move(attributes), context));

    return Y;
}

inline std::shared_ptr<Tensor_attributes>
Graph::concatenate(std::vector<std::shared_ptr<Tensor_attributes>> x, Concatenate_attributes attributes) {
    if (attributes.name.empty()) {
        attributes.name += std::to_string(sub_nodes.size());
    }

    // Set outputs
    auto Y = attributes.outputs[Concatenate_attributes::output_names::Y] = output_tensor(attributes.name + "::Y");

    // Set inputs
    for (auto &element : x) {
        attributes.inputs.push_back(element);
    }

    sub_nodes.emplace_back(std::make_unique<ConcatenateNode>(std::move(attributes), context));

    return Y;
}

inline std::shared_ptr<Tensor_attributes>
Graph::moe_grouped_matmul(std::shared_ptr<Tensor_attributes> token,
                          std::shared_ptr<Tensor_attributes> weight,
                          std::shared_ptr<Tensor_attributes> first_token_offset,
                          std::shared_ptr<Tensor_attributes> token_index,
                          std::shared_ptr<Tensor_attributes> token_ks,
                          Moe_grouped_matmul_attributes attributes) {
    if (attributes.name.empty()) {
        attributes.name += std::to_string(sub_nodes.size());
    }

    auto output = attributes.outputs[Moe_grouped_matmul_attributes::output_names::Output] =
        output_tensor(attributes.name + "::Output");

    attributes.inputs[Moe_grouped_matmul_attributes::input_names::Token]            = token;
    attributes.inputs[Moe_grouped_matmul_attributes::input_names::Weight]           = weight;
    attributes.inputs[Moe_grouped_matmul_attributes::input_names::FirstTokenOffset] = first_token_offset;
    attributes.inputs[Moe_grouped_matmul_attributes::input_names::TokenIndex]       = token_index;
    attributes.inputs[Moe_grouped_matmul_attributes::input_names::TokenKs]          = token_ks;

    sub_nodes.emplace_back(std::make_unique<MoeGroupedMatmulNode>(std::move(attributes), context));

    return output;
}

inline std::shared_ptr<Tensor_attributes>
Graph::moe_grouped_matmul_bwd(std::shared_ptr<Tensor_attributes> doutput,
                              std::shared_ptr<Tensor_attributes> token,
                              std::shared_ptr<Tensor_attributes> first_token_offset,
                              Moe_grouped_matmul_bwd_attributes attributes) {
    if (attributes.name.empty()) {
        attributes.name += std::to_string(sub_nodes.size());
    }

    auto dweight = attributes.outputs[Moe_grouped_matmul_bwd_attributes::output_names::DWeight] =
        output_tensor(attributes.name + "::DWeight");

    attributes.inputs[Moe_grouped_matmul_bwd_attributes::input_names::DOutput]          = doutput;
    attributes.inputs[Moe_grouped_matmul_bwd_attributes::input_names::Token]            = token;
    attributes.inputs[Moe_grouped_matmul_bwd_attributes::input_names::FirstTokenOffset] = first_token_offset;

    sub_nodes.emplace_back(std::make_unique<MoeGroupedMatmulBwdNode>(std::move(attributes), context));

    return dweight;
}

static inline std::ostream &
operator<<(std::ostream &os, Graph const &graph) {
    os << graph.print();
    return os;
}

}  // namespace cudnn_frontend::graph
