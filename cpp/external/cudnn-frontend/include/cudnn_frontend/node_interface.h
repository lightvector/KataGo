#pragma once

#include <memory>
#include <vector>
#include <unordered_map>
#include <variant>
#include <limits>

#include <cuda_fp16.h>
#include <cuda_bf16.h>

#include "../cudnn_frontend_Tensor.h"
#include "../cudnn_frontend_Operation.h"
#include "../cudnn_frontend_OperationGraph.h"
#include "../cudnn_frontend_ExecutionPlan.h"
#include "../cudnn_frontend_VariantPack.h"
#include "../cudnn_frontend_shim.h"

#include "cudnn_interface.h"

#include "graph_properties.h"

namespace cudnn_frontend {

namespace graph {

class BatchNormNode;
class DBNNode;
class ConcatenateNode;
class MatmulNode;
class MatmulFP8Node;
class PointwiseNode;
class ReductionNode;
class ResampleNode;
class ReshapeNode;
class RngNode;
class CompositeSoftmaxNode;
class UnifiedSoftmaxNode;
class MoeGroupedMatmulNode;
class UnifiedDiagonalBandMaskNode;
class TransposeNode;
class SliceNode;

// Interface for all nodes to follow.
class INode {
   public:
    // A closed set of types that are allowed to be passed by value today
    using pass_by_values_t = Tensor_attributes::pass_by_values_t;

    detail::Context context;

   protected:
    // Will eventually be moved to Graph class
    std::unordered_set<std::shared_ptr<Tensor_attributes>> full_graph_outputs;
    std::shared_ptr<Tensor_attributes>
    output_tensor(std::string const& name) {
        auto tensor = std::make_shared<Tensor_attributes>();
        tensor->set_name(name).set_is_virtual(true);
        full_graph_outputs.insert(tensor);
        return tensor;
    }

   private:
    virtual error_t
    pre_validate_node() const {
        return {error_code_t::OK, ""};
    };

    virtual error_t
    infer_properties_node() = 0;

    virtual error_t
    expand_node() {
        return {error_code_t::OK, ""};
    };

    virtual error_t
    post_validate_node() const {
        return {error_code_t::OK, ""};
    };

    virtual int64_t
    get_fe_workspace_size_node() const {
        return 0;
    }

    virtual error_t
    collect_pass_by_value_tensors_node(std::unordered_map<Tensor_attributes::uid_t, pass_by_values_t>&) const {
        return {error_code_t::OK, ""};
    };

    virtual error_t
    collect_variant_pack_replacements_node(
        std::unordered_map<Tensor_attributes::uid_t, std::pair<Tensor_attributes::uid_t, int64_t>>&) const {
        return {error_code_t::OK, ""};
    };

    virtual error_t
    collect_tensors_to_dump_node(std::vector<std::pair<std::shared_ptr<Tensor_attributes>, char>>&) const {
        return {error_code_t::OK, ""};
    };

    virtual error_t
    create_cudnn_tensors_node(
        std::unordered_map<int64_t, std::shared_ptr<cudnn_frontend::Tensor>>& uid_to_backend_tensors,
        int64_t& potential_uid,
        std::unordered_set<int64_t> const& used_uids) const = 0;

    virtual error_t
    collect_tensors_in_workspace_node(
        std::unordered_map<Tensor_attributes::uid_t, std::tuple<int64_t, int64_t, std::vector<float>>>&,
        int64_t&) const {
        return {error_code_t::OK, ""};
    }

   protected:
    // Type of each node. Nodes can either be a composite (value COMPOSITE) or
    // one of the other primitive types. Primitives types are nothing but
    // cudnn operations.
    enum class Type {
        COMPOSITE,
        BATCHNORM,
        BATCHNORM_INFERENCE,
        BN_FINALIZE,
        CONVOLUTION,
        DBN,
        DBN_WEIGHT,
        DLN,
        DIN,
        DGRAD,
        DRMSNorm,
        GENSTATS,
        LAYERNORM,
        INSTANCENORM,
        MATMUL,
        POINTWISE,
        REDUCTION,
        RESAMPLE,
        RESHAPE,
        RMSNORM,
        RNG,
        SLICE,
        TRANSPOSE,
        WGRAD,
        PAGED_CACHE_LOAD,
        BLOCK_SCALE_QUANTIZE,
        BLOCK_SCALE_DEQUANTIZE,
        CONCATENATE,
        ADALAYERNORM,
        DADALAYERNORM,
        SDPA,
        ROPE,
        ROPE_BWD,
        MOE_GROUPED_MATMUL,
        MOE_GROUPED_MATMUL_BWD,
        DIAGONAL_BAND_MASK,
        SOFTMAX,
    };
    Type tag;

    inline void
    matmul(std::shared_ptr<Tensor_attributes> a,
           std::shared_ptr<Tensor_attributes> b,
           Matmul_attributes attributes,
           std::shared_ptr<Tensor_attributes> c);

    void
    matmul_fp8(std::shared_ptr<Tensor_attributes> a,
               std::shared_ptr<Tensor_attributes> b,
               std::shared_ptr<Tensor_attributes> descale_a,
               std::shared_ptr<Tensor_attributes> descale_b,
               std::shared_ptr<Tensor_attributes> scale_c,
               Matmul_fp8_attributes attributes,
               std::shared_ptr<Tensor_attributes> c,
               std::shared_ptr<Tensor_attributes> amax_c);

    void
    softmax(std::shared_ptr<Tensor_attributes> p,
            Softmax_attributes attributes,
            std::shared_ptr<Tensor_attributes> s,
            std::shared_ptr<Tensor_attributes> stats,
            std::shared_ptr<Tensor_attributes> max,
            std::shared_ptr<Tensor_attributes> sum_exp);

    void
    softmax(std::shared_ptr<Tensor_attributes> p,
            Softmax_attributes attributes,
            std::shared_ptr<Tensor_attributes> s,
            std::shared_ptr<Tensor_attributes> m,
            std::shared_ptr<Tensor_attributes> zinv);

    void
    pointwise(std::shared_ptr<Tensor_attributes> a,
              Pointwise_attributes attributes,
              std::shared_ptr<Tensor_attributes> c);

    void
    pointwise(std::shared_ptr<Tensor_attributes> a,
              std::shared_ptr<Tensor_attributes> b,
              Pointwise_attributes attributes,
              std::shared_ptr<Tensor_attributes> c);

    void
    pointwise(std::shared_ptr<Tensor_attributes> a,
              std::shared_ptr<Tensor_attributes> b,
              std::shared_ptr<Tensor_attributes> c,
              Pointwise_attributes attributes,
              std::shared_ptr<Tensor_attributes> d);

    void
    reduction(std::shared_ptr<Tensor_attributes> a,
              Reduction_attributes attributes,
              std::shared_ptr<Tensor_attributes> c);

    void
    rng(std::shared_ptr<Tensor_attributes> seed,
        std::shared_ptr<Tensor_attributes> offset,
        Rng_attributes attributes,
        std::shared_ptr<Tensor_attributes> y);

    void
    paged_cache_load(std::shared_ptr<Tensor_attributes> container,
                     std::shared_ptr<Tensor_attributes> seqLen,
                     std::shared_ptr<Tensor_attributes> pageTable,
                     PagedCacheLoad_attributes attributes,
                     std::shared_ptr<Tensor_attributes> yOut);

    void
    block_scale_quantize(std::shared_ptr<Tensor_attributes> x,
                         Block_scale_quantize_attributes attributes,
                         std::shared_ptr<Tensor_attributes> y,
                         std::shared_ptr<Tensor_attributes> scale);

    void
    block_scale_dequantize(std::shared_ptr<Tensor_attributes> x,
                           std::shared_ptr<Tensor_attributes> scale,
                           Block_scale_dequantize_attributes attributes,
                           std::shared_ptr<Tensor_attributes> y);

    void
    concatenate(std::vector<std::shared_ptr<Tensor_attributes>> x,
                Concatenate_attributes attributes,
                std::shared_ptr<Tensor_attributes> y);

    void
    moe_grouped_matmul(std::shared_ptr<Tensor_attributes> token,
                       std::shared_ptr<Tensor_attributes> weight,
                       std::shared_ptr<Tensor_attributes> first_token_offset,
                       std::shared_ptr<Tensor_attributes> token_index,
                       std::shared_ptr<Tensor_attributes> token_ks,
                       Moe_grouped_matmul_attributes attributes,
                       std::shared_ptr<Tensor_attributes> output);

    void
    moe_grouped_matmul_bwd(std::shared_ptr<Tensor_attributes> doutput,
                           std::shared_ptr<Tensor_attributes> token,
                           std::shared_ptr<Tensor_attributes> first_token_offset,
                           Moe_grouped_matmul_bwd_attributes attributes,
                           std::shared_ptr<Tensor_attributes> dweight);

    void
    transpose(std::shared_ptr<Tensor_attributes> input,
              Transpose_attributes attributes,
              std::shared_ptr<Tensor_attributes> output);

    void
    slice(std::shared_ptr<Tensor_attributes> input,
          Slice_attributes attributes,
          std::shared_ptr<Tensor_attributes> output);
    error_t
    validate_subtree() {
        // pre validate to catch errors early
        // Otherwise code reability decreases in expand_and_infer
        CHECK_CUDNN_FRONTEND_ERROR(pre_validate_node());
        CHECK_CUDNN_FRONTEND_ERROR(infer_properties_node());
        for (auto const& sub_node : sub_nodes) {
            CHECK_CUDNN_FRONTEND_ERROR(sub_node->validate_subtree());
        }
        CHECK_CUDNN_FRONTEND_ERROR(post_validate_node());
        return {error_code_t::OK, ""};
    }

    error_t
    expand_subtree() {
        // Validate self
        CHECK_CUDNN_FRONTEND_ERROR(pre_validate_node());
        CHECK_CUDNN_FRONTEND_ERROR(infer_properties_node());
        CHECK_CUDNN_FRONTEND_ERROR(expand_node());
        for (auto const& sub_node : sub_nodes) {
            CHECK_CUDNN_FRONTEND_ERROR(sub_node->expand_subtree());
        }
        CHECK_CUDNN_FRONTEND_ERROR(post_validate_node());
        return {error_code_t::OK, ""};
    }

    // Creates cudnn tensors for each node (and its sub nodes)
    error_t
    create_cudnn_tensors_subtree(
        std::unordered_map<int64_t, std::shared_ptr<cudnn_frontend::Tensor>>& uid_to_backend_tensors,
        int64_t& potential_uid,
        std::unordered_set<int64_t> const& used_uids) const {
        CHECK_CUDNN_FRONTEND_ERROR(create_cudnn_tensors_node(uid_to_backend_tensors, potential_uid, used_uids));
        for (auto const& sub_node : sub_nodes) {
            CHECK_CUDNN_FRONTEND_ERROR(
                sub_node->create_cudnn_tensors_subtree(uid_to_backend_tensors, potential_uid, used_uids));
        }
        return {error_code_t::OK, ""};
    }

    error_t
    collect_pass_by_value_tensors_subtree(
        std::unordered_map<Tensor_attributes::uid_t, pass_by_values_t>& tensor_to_pass_by_value) const {
        CHECK_CUDNN_FRONTEND_ERROR(collect_pass_by_value_tensors_node(tensor_to_pass_by_value));
        for (auto const& sub_node : sub_nodes) {
            CHECK_CUDNN_FRONTEND_ERROR(sub_node->collect_pass_by_value_tensors_subtree(tensor_to_pass_by_value));
        }
        return {error_code_t::OK, ""};
    }

    error_t
    collect_tensors_in_workspace_subtree(
        std::unordered_map<Tensor_attributes::uid_t, std::tuple<int64_t, int64_t, std::vector<float>>>&
            worskspace_modifications,
        int64_t& offset) const {
        CHECK_CUDNN_FRONTEND_ERROR(collect_tensors_in_workspace_node(worskspace_modifications, offset));
        offset = get_fe_workspace_size_node();
        for (auto const& sub_node : sub_nodes) {
            CHECK_CUDNN_FRONTEND_ERROR(
                sub_node->collect_tensors_in_workspace_subtree(worskspace_modifications, offset));
            offset += sub_node->get_fe_workspace_size_node();
        }
        return {error_code_t::OK, ""};
    }

    error_t
    collect_variant_pack_replacements_subtree(
        std::unordered_map<Tensor_attributes::uid_t, std::pair<Tensor_attributes::uid_t, int64_t>>& replacements)
        const {
        CHECK_CUDNN_FRONTEND_ERROR(collect_variant_pack_replacements_node(replacements));
        for (auto const& sub_node : sub_nodes) {
            CHECK_CUDNN_FRONTEND_ERROR(sub_node->collect_variant_pack_replacements_subtree(replacements));
        }
        return {error_code_t::OK, ""};
    }

    error_t
    collect_tensors_to_dump_subtree(
        std::vector<std::pair<std::shared_ptr<Tensor_attributes>, char>>& tensors_to_dump) const {
        CHECK_CUDNN_FRONTEND_ERROR(collect_tensors_to_dump_node(tensors_to_dump));
        for (auto const& sub_node : sub_nodes) {
            CHECK_CUDNN_FRONTEND_ERROR(sub_node->collect_tensors_to_dump_subtree(tensors_to_dump));
        }
        return {error_code_t::OK, ""};
    }

    int64_t
    get_fe_workspace_size_subtree() const {
        int64_t fe_workspace_size = get_fe_workspace_size_node();
        for (auto const& sub_node : sub_nodes) {
            fe_workspace_size += sub_node->get_fe_workspace_size_subtree();
        }
        return fe_workspace_size;
    }

    // Creates cudnn operation for each node (and its sub nodes)
    // Only INode that map to a primitive cudnn operation need to specialize.
    virtual error_t
    create_cudnn_operations(
        std::unordered_set<Tensor_attributes::uid_t>& uids_involved_in_operation,
        std::vector<std::shared_ptr<cudnn_frontend::Operation>>& backend_operations,
        managed_backend_descriptor_t& raw_operations,
        std::unordered_map<int64_t, std::shared_ptr<cudnn_frontend::Tensor>>& uid_to_backend_tensors) const {
        for (auto const& sub_node : sub_nodes) {
            CHECK_CUDNN_FRONTEND_ERROR(sub_node->create_cudnn_operations(
                uids_involved_in_operation, backend_operations, raw_operations, uid_to_backend_tensors));
        }
        return {error_code_t::OK, ""};
    }

    // An implicitly topological-sorted vector of sub nodes.
    // The sorted order is a side effect of functional API.
    std::vector<std::shared_ptr<INode>> sub_nodes;

   public:
    virtual Type
    getType() = 0;

    virtual std::pair<int64_t, std::unordered_map<KnobType_t, int64_t>>
    override_heuristics_query() const {
        return {-1, {}};
    }

    std::shared_ptr<Tensor_attributes> matmul(std::shared_ptr<Tensor_attributes>,
                                              std::shared_ptr<Tensor_attributes>,
                                              Matmul_attributes);

    std::shared_ptr<Tensor_attributes> pointwise(std::shared_ptr<Tensor_attributes>, Pointwise_attributes);
    std::shared_ptr<Tensor_attributes> pointwise(std::shared_ptr<Tensor_attributes>,
                                                 std::shared_ptr<Tensor_attributes>,
                                                 Pointwise_attributes);
    std::shared_ptr<Tensor_attributes> pointwise(std::shared_ptr<Tensor_attributes>,
                                                 std::shared_ptr<Tensor_attributes>,
                                                 std::shared_ptr<Tensor_attributes>,
                                                 Pointwise_attributes);

    std::shared_ptr<Tensor_attributes> reduction(std::shared_ptr<Tensor_attributes>, Reduction_attributes);
    std::array<std::shared_ptr<Tensor_attributes>, 2> resample(std::shared_ptr<Tensor_attributes>, Resample_attributes);
    std::shared_ptr<Tensor_attributes> reshape(std::shared_ptr<Tensor_attributes>, Reshape_attributes);
    std::shared_ptr<Tensor_attributes> transpose(std::shared_ptr<Tensor_attributes>, Transpose_attributes);
    std::shared_ptr<Tensor_attributes> slice(std::shared_ptr<Tensor_attributes>, Slice_attributes);

    std::shared_ptr<Tensor_attributes> rng(std::shared_ptr<Tensor_attributes>,
                                           std::shared_ptr<Tensor_attributes>,
                                           Rng_attributes);

    std::shared_ptr<Tensor_attributes>
    diagonal_band_mask(std::shared_ptr<Tensor_attributes> x,
                       std::shared_ptr<Tensor_attributes> b,
                       std::shared_ptr<Tensor_attributes> seq_len_q,
                       std::shared_ptr<Tensor_attributes> seq_len_kv,
                       std::shared_ptr<Tensor_attributes> left_bound,
                       std::shared_ptr<Tensor_attributes> shift_right_bound,
                       DiagonalBandMask_attributes attributes);

    INode(detail::Context const& context) : context(context) {}

    // Make sure each node implements a public serialize function
#ifndef CUDNN_FRONTEND_SKIP_JSON_LIB
    virtual void
    serialize(json& j) const = 0;
#endif

    virtual size_t
    key() {
#ifndef CUDNN_FRONTEND_SKIP_JSON_LIB
        json j;
        serialize(j);
        return std::hash<json>{}(j);
#else
        return 1;
#endif
    }

    virtual ~INode() = default;
};

#ifndef CUDNN_FRONTEND_SKIP_JSON_LIB
[[maybe_unused]] static void
to_json(json& j, const INode& p) {
    p.serialize(j);
}
#endif

template <typename DerivedT>
class NodeCRTP : public INode {
    DerivedT&
    self() {
        return *static_cast<DerivedT*>(this);
    }
    DerivedT const&
    self() const {
        return *static_cast<DerivedT const*>(this);
    }

    error_t
    collect_pass_by_value_tensors_node(
        std::unordered_map<Tensor_attributes::uid_t, pass_by_values_t>& tensor_to_pass_by_value) const override final {
        CHECK_CUDNN_FRONTEND_ERROR(self().attributes.fill_pass_by_value(tensor_to_pass_by_value));

        return {error_code_t::OK, ""};
    }

    error_t
    create_cudnn_tensors_node(std::unordered_map<int64_t, std::shared_ptr<cudnn_frontend::Tensor>>& tensors,
                              int64_t& potential_uid,
                              std::unordered_set<int64_t> const& used_uids) const override {
        CUDNN_FE_LOG_LABEL_ENDL("INFO: Creating cudnn tensors for node named '" << self().attributes.name << "':");

        if constexpr (std::is_same_v<DerivedT, ConcatenateNode>) {
            for (auto const& tensor : self().attributes.inputs) {
                if (tensor) {
                    CHECK_CUDNN_FRONTEND_ERROR(detail::create_cudnn_tensor(tensor, tensors, potential_uid, used_uids));
                }
            }
        } else {
            for (auto const& [name, tensor] : self().attributes.inputs) {
                (void)name;
                if (tensor) {
                    CHECK_CUDNN_FRONTEND_ERROR(detail::create_cudnn_tensor(tensor, tensors, potential_uid, used_uids));
                }
            }
        }

        for (auto const& [name, tensor] : self().attributes.outputs) {
            (void)name;
            if (tensor) {
                CHECK_CUDNN_FRONTEND_ERROR(detail::create_cudnn_tensor(tensor, tensors, potential_uid, used_uids));
            }
        }

        // Handle special case of BN where peer_stats is also an input
        if constexpr (std::is_same_v<DerivedT, DBNNode> || std::is_same_v<DerivedT, BatchNormNode>) {
            // Special case in BN where peer stats is also an input but is not present in inputs map
            for (auto const& tensor : self().attributes.peer_stats) {
                if (tensor) {
                    CHECK_CUDNN_FRONTEND_ERROR(detail::create_cudnn_tensor(tensor, tensors, potential_uid, used_uids));
                }
            }
        }

        return {error_code_t::OK, ""};
    }

   protected:
    using INode::INode;
};

#define CUDNN_FE_VALIDATE_TENSOR_(port, map_)                                                      \
    {                                                                                              \
        auto t           = map_.find(port);                                                        \
        bool const has_t = (t != map_.end()) && (t->second != nullptr);                            \
        RETURN_CUDNN_FRONTEND_ERROR_IF(                                                            \
            !has_t, error_code_t::ATTRIBUTE_NOT_SET, std::string("Tensor ") + #port + " not set"); \
    }

#define CUDNN_FE_VALIDATE_AND_ASSIGN_TENSOR_(tensor, port, map_)                                   \
    auto tensor = map_.find(port);                                                                 \
    {                                                                                              \
        bool const has_t = (tensor != map_.end()) && (tensor->second != nullptr);                  \
        RETURN_CUDNN_FRONTEND_ERROR_IF(                                                            \
            !has_t, error_code_t::ATTRIBUTE_NOT_SET, std::string("Tensor ") + #port + " not set"); \
    }

#define CUDNN_FE_VALIDATE_INPUT_TENSOR(port) CUDNN_FE_VALIDATE_TENSOR_(port, attributes.inputs)

#define CUDNN_FE_VALIDATE_OUTPUT_TENSOR(port) CUDNN_FE_VALIDATE_TENSOR_(port, attributes.outputs)

#define CUDNN_FE_VALIDATE_AND_ASSIGN_INPUT_TENSOR(tensor, port) \
    CUDNN_FE_VALIDATE_AND_ASSIGN_TENSOR_(tensor, port, attributes.inputs)

#define CUDNN_FE_VALIDATE_AND_ASSIGN_OUTPUT_TENSOR(tensor, port) \
    CUDNN_FE_VALIDATE_AND_ASSIGN_TENSOR_(tensor, port, attributes.outputs)

}  // namespace graph

}  // namespace cudnn_frontend
