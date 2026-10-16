#pragma once

#include "../../cudnn_frontend_Heuristics.h"
#include "../../cudnn_frontend_Logging.h"

#include "../graph_helpers.h"
#include "../node_interface.h"

#include "pointwise.h"
#include "reduction.h"

namespace cudnn_frontend::graph {

// Base class for nodes that represent a softmax operation.
// This is either implemented as a composite of pointwise/reduction operations (CompositeSoftmaxNode)
// or as a single unified backend operation (UnifiedSoftmaxNode).
template <typename DerivedT>
class SoftmaxNodeBase : public NodeCRTP<DerivedT> {
   protected:
    using input_names  = Softmax_attributes::input_names;
    using output_names = Softmax_attributes::output_names;

   public:
    Softmax_attributes attributes;

    SoftmaxNodeBase(Softmax_attributes&& attributes_, detail::Context const& context)
        : NodeCRTP<DerivedT>(context), attributes(std::move(attributes_)) {}

    error_t
    infer_properties_node() override final {
        CUDNN_FE_LOG_LABEL_ENDL("INFO:     Inferrencing properties for Softmax node " << attributes.name);

        attributes.fill_from_context(this->context);

        auto input_tensor  = attributes.inputs.at(Softmax_attributes::input_names::P);
        auto output_tensor = attributes.outputs.at(Softmax_attributes::output_names::S);

        // Only infer dims and strides if user did not set them
        if (output_tensor->get_dim().empty()) {
            output_tensor->set_dim(input_tensor->get_dim());
        }

        if (output_tensor->get_stride().empty()) {
            output_tensor->set_stride(input_tensor->get_stride());
        }

        return {error_code_t::OK, ""};
    }

   protected:
    bool
    has_sink() const {
        auto sink_it = attributes.inputs.find(Softmax_attributes::input_names::SINK);
        return ((sink_it) != attributes.inputs.end() && sink_it->second != nullptr);
    }

    bool
    has_stats() const {
        auto stats_it = attributes.outputs.find(Softmax_attributes::output_names::Stats);
        return ((stats_it) != attributes.outputs.end() && stats_it->second != nullptr);
    }

    bool
    has_max() const {
        auto max_it = attributes.outputs.find(Softmax_attributes::output_names::Max);
        return ((max_it) != attributes.outputs.end() && max_it->second != nullptr);
    }

    bool
    has_sum_exp() const {
        auto sum_exp_it = attributes.outputs.find(Softmax_attributes::output_names::Sum_exp);
        return ((sum_exp_it) != attributes.outputs.end() && sum_exp_it->second != nullptr);
    }

#ifndef CUDNN_FRONTEND_SKIP_JSON_LIB
    void
    serialize(json& j) const override final {
        j = attributes;
    }
#endif
};

// Fallback implementation of softmax node that represents the operation as a series of pointwise and reduction
// operations. This is used for cuDNN versions before v9.21.0.
// Only certain combinations of outputs are allowed.
class CompositeSoftmaxNode : public SoftmaxNodeBase<CompositeSoftmaxNode> {
   public:
    CompositeSoftmaxNode(Softmax_attributes&& attributes_, detail::Context const& context)
        : SoftmaxNodeBase(std::move(attributes_), context) {}

    Type
    getType() override final {
        return Type::COMPOSITE;
    }

    error_t
    pre_validate_node() const override final {
        CUDNN_FE_LOG_LABEL_ENDL("INFO: Validating CompositeSoftmaxNode " << attributes.name);

        RETURN_CUDNN_FRONTEND_ERROR_IF(
            !((!has_stats() && !has_max() && !has_sum_exp()) ||  //
              (has_stats() && !has_max() && !has_sum_exp()) ||   //
              (!has_stats() && has_max() && has_sum_exp())),
            error_code_t::INVALID_VALUE,
            "CompositeSoftmaxNode can only output certain combinations of stats, max and sum_exp: "
            "stats only, max and sum_exp only, or none of the above.");

        return {error_code_t::OK, ""};
    }

    error_t
    expand_node() override final {
        CUDNN_FE_LOG_LABEL_ENDL("INFO:     Inferrencing properties for CompositeSoftmaxNode " << attributes.name);

        attributes.fill_from_context(context);

        // Fill properties of virtual tensors
        auto const p_dim = attributes.inputs[Softmax_attributes::input_names::P]->get_dim();
        auto b           = p_dim[0];
        auto h           = p_dim[1];
        auto s_q         = p_dim[2];

        std::shared_ptr<Tensor_attributes> max_output;
        if (has_max()) {
            max_output = attributes.outputs[Softmax_attributes::output_names::Max];
        } else {
            max_output = std::make_shared<Tensor_attributes>();
            max_output->set_is_virtual(true);
        }
        //////////////// TODO //////////////////////////
        // Check Stride (Before setting dimension?)
        if (max_output->get_dim().empty()) {
            max_output->set_dim({b, h, s_q, 1});
        }
        if (max_output->get_stride().empty()) {
            max_output->set_stride({h * s_q, s_q, 1, 1});
        }

        auto max_attributes = Reduction_attributes().set_name("Max").set_mode(ReductionMode_t::MAX);
        // If sink tensor is present, we also need to take a pointwise max with sink
        if (has_sink()) {
            auto s_max = reduction(attributes.inputs[Softmax_attributes::input_names::P], max_attributes);
            s_max->set_name("s_max");

            auto sink_tensor     = attributes.inputs[Softmax_attributes::input_names::SINK];
            auto sink_attributes = Pointwise_attributes().set_name("max_sink").set_mode(PointwiseMode_t::MAX);
            pointwise(s_max, sink_tensor, sink_attributes, max_output);
        } else {
            // Special non-functional-style call. Needed because output already created and provided to user.
            reduction(attributes.inputs[Softmax_attributes::input_names::P], max_attributes, max_output);
        }

        auto sub_attributes = Pointwise_attributes().set_name("sub").set_mode(PointwiseMode_t::SUB);
        auto const& sub_output =
            pointwise(attributes.inputs[Softmax_attributes::input_names::P], max_output, sub_attributes);
        sub_output->set_name("sub_M");

        auto exp_attributes    = Pointwise_attributes().set_name("exp").set_mode(PointwiseMode_t::EXP);
        auto const& exp_output = pointwise(sub_output, exp_attributes);
        exp_output->set_name("exp_sub_M");

        std::shared_ptr<Tensor_attributes> sum_output;
        if (has_sum_exp()) {
            sum_output = attributes.outputs[Softmax_attributes::output_names::Sum_exp];
        } else {
            sum_output = std::make_shared<Tensor_attributes>();
            sum_output->set_is_virtual(true);
        }
        sum_output->set_name("SumExp");
        if (sum_output->get_dim().empty()) {
            sum_output->set_dim({b, h, s_q, 1});
        }
        if (sum_output->get_stride().empty()) {
            sum_output->set_stride({h * s_q, s_q, 1, 1});
        }
        auto sum_attributes = Reduction_attributes().set_name("sum").set_mode(ReductionMode_t::ADD);
        // If sink tensor is present, also subtract it and take its exp
        if (has_sink()) {
            auto sink_tensor = attributes.inputs[Softmax_attributes::input_names::SINK];
            auto sub_sink    = pointwise(sink_tensor, max_output, sub_attributes);
            sub_sink->set_name("sub_sink");

            auto exp_sink = pointwise(sub_sink, exp_attributes);
            exp_sink->set_name("exp_sink");

            auto temp_sum = reduction(exp_output, sum_attributes);
            temp_sum->set_name("SumExp_elements").set_dim({b, h, s_q, 1}).set_stride({h * s_q, s_q, 1, 1});

            auto add_attributes = Pointwise_attributes().set_name("add_sink").set_mode(PointwiseMode_t::ADD);
            pointwise(temp_sum, exp_sink, add_attributes, sum_output);
        } else {
            reduction(exp_output, sum_attributes, sum_output);
        }

        // Set to virtual when:
        // - softmax stats in not requested
        // - max and sum_exp are not requested
        if (!has_stats() && !has_max() && !has_sum_exp()) {
            auto softmax_stats = std::make_shared<Tensor_attributes>();
            softmax_stats->set_is_virtual(true);
            attributes.outputs[Softmax_attributes::output_names::Stats] = softmax_stats;
        }

        if (has_stats()) {
            auto log_attributes    = Pointwise_attributes().set_name("log").set_mode(PointwiseMode_t::LOG);
            auto const& log_output = pointwise(sum_output, log_attributes);
            log_output->set_dim({b, h, s_q, 1}).set_stride({h * s_q, s_q, 1, 1});

            auto add_attributes = Pointwise_attributes().set_name("add").set_mode(PointwiseMode_t::ADD);
            // Special non-functional-style call. Needed because output already created and provided to user.
            pointwise(
                max_output, log_output, add_attributes, attributes.outputs[Softmax_attributes::output_names::Stats]);
        }

        auto div_attributes = Pointwise_attributes().set_name("div").set_mode(PointwiseMode_t::DIV);
        // Special non-functional-style call. Needed because output already created and provided to user.
        pointwise(exp_output, sum_output, div_attributes, attributes.outputs[Softmax_attributes::output_names::S]);

        return {error_code_t::OK, ""};
    }
};

// Newer implementation of softmax node that represents the operation as a single backend operation.
// This is used for cuDNN versions v9.21.0 and later.
class UnifiedSoftmaxNode : public SoftmaxNodeBase<UnifiedSoftmaxNode> {
   public:
    UnifiedSoftmaxNode(Softmax_attributes&& attributes_, detail::Context const& context)
        : SoftmaxNodeBase(std::move(attributes_), context) {}

    Type
    getType() override final {
        return Type::SOFTMAX;
    }

    error_t
    pre_validate_node() const override final {
        CUDNN_FE_LOG_LABEL_ENDL("INFO: Validating UnifiedSoftmaxNode " << attributes.name);

        return {error_code_t::OK, ""};
    }

    error_t
    create_cudnn_operations(
        std::unordered_set<Tensor_attributes::uid_t>& uids_involved_in_operations,
        std::vector<std::shared_ptr<cudnn_frontend::Operation>>& operations,
        managed_backend_descriptor_t& raw_operations,
        std::unordered_map<int64_t, std::shared_ptr<cudnn_frontend::Tensor>>& tensors) const override final {
        getLogger() << "[cudnn_frontend] INFO: " << "Building UnifiedSoftmaxNode operations " << attributes.name
                    << std::endl;
        auto cudnn_ver_error = error_t{error_code_t::GRAPH_NOT_SUPPORTED, "UnifiedSoftmaxNode requires cuDNN v9.21.0"};

#if (CUDNN_VERSION >= 92100)
        NV_CUDNN_FE_DYNAMIC_CHECK_CUDNN_BACKEND_VERSION(92100, cudnn_ver_error);
        CUDNN_FRONTEND_UNUSED(operations);
        auto softmax_operation =
            make_shared_backend_pointer((cudnnBackendDescriptorType_t)CUDNN_BACKEND_OPERATION_SOFTMAX_DESCRIPTOR);

        // Set input tensor P
        auto P         = attributes.inputs.find(Softmax_attributes::input_names::P)->second;
        auto backend_p = tensors[P->get_uid()]->get_desc()->get_backend_descriptor();
        _CUDNN_CHECK_CUDNN_ERROR(detail::set_attribute(softmax_operation->get_backend_descriptor(),
                                                       CUDNN_ATTR_OPERATION_SOFTMAX_XDESC,
                                                       CUDNN_TYPE_BACKEND_DESCRIPTOR,
                                                       1,
                                                       &backend_p));

        // Set output tensor S
        auto S         = attributes.outputs.find(Softmax_attributes::output_names::S)->second;
        auto backend_s = tensors[S->get_uid()]->get_desc()->get_backend_descriptor();
        _CUDNN_CHECK_CUDNN_ERROR(detail::set_attribute(softmax_operation->get_backend_descriptor(),
                                                       CUDNN_ATTR_OPERATION_SOFTMAX_YDESC,
                                                       CUDNN_TYPE_BACKEND_DESCRIPTOR,
                                                       1,
                                                       &backend_s));

        // Set optional SINK tensor if present
        if (has_sink()) {
            auto SINK         = attributes.inputs.find(Softmax_attributes::input_names::SINK)->second;
            auto backend_sink = tensors[SINK->get_uid()]->get_desc()->get_backend_descriptor();
            _CUDNN_CHECK_CUDNN_ERROR(detail::set_attribute(softmax_operation->get_backend_descriptor(),
                                                           CUDNN_ATTR_OPERATION_SOFTMAX_SINK_DESC,
                                                           CUDNN_TYPE_BACKEND_DESCRIPTOR,
                                                           1,
                                                           &backend_sink));
        }

        // Set optional Stats output tensor if present
        if (has_stats()) {
            auto Stats         = attributes.outputs.find(Softmax_attributes::output_names::Stats)->second;
            auto backend_stats = tensors[Stats->get_uid()]->get_desc()->get_backend_descriptor();
            _CUDNN_CHECK_CUDNN_ERROR(detail::set_attribute(softmax_operation->get_backend_descriptor(),
                                                           CUDNN_ATTR_OPERATION_SOFTMAX_STATS_DESC,
                                                           CUDNN_TYPE_BACKEND_DESCRIPTOR,
                                                           1,
                                                           &backend_stats));
        }

        // Set optional Max output tensor if present
        if (has_max()) {
            auto Max         = attributes.outputs.find(Softmax_attributes::output_names::Max)->second;
            auto backend_max = tensors[Max->get_uid()]->get_desc()->get_backend_descriptor();
            _CUDNN_CHECK_CUDNN_ERROR(detail::set_attribute(softmax_operation->get_backend_descriptor(),
                                                           CUDNN_ATTR_OPERATION_SOFTMAX_MAX_DESC,
                                                           CUDNN_TYPE_BACKEND_DESCRIPTOR,
                                                           1,
                                                           &backend_max));
        }

        // Set optional Sum_exp output tensor if present
        if (has_sum_exp()) {
            auto Sum_exp         = attributes.outputs.find(Softmax_attributes::output_names::Sum_exp)->second;
            auto backend_sum_exp = tensors[Sum_exp->get_uid()]->get_desc()->get_backend_descriptor();
            _CUDNN_CHECK_CUDNN_ERROR(detail::set_attribute(softmax_operation->get_backend_descriptor(),
                                                           CUDNN_ATTR_OPERATION_SOFTMAX_SUM_EXP_DESC,
                                                           CUDNN_TYPE_BACKEND_DESCRIPTOR,
                                                           1,
                                                           &backend_sum_exp));
        }

        _CUDNN_CHECK_CUDNN_ERROR(detail::finalize(softmax_operation->get_backend_descriptor()));

        raw_operations.push_back(softmax_operation);

        auto const& non_virtual_uids = attributes.get_non_virtual_uids();
        uids_involved_in_operations.insert(non_virtual_uids.begin(), non_virtual_uids.end());
        return {error_code_t::OK, ""};
#else
        CUDNN_FRONTEND_UNUSED(uids_involved_in_operations);
        CUDNN_FRONTEND_UNUSED(operations);
        CUDNN_FRONTEND_UNUSED(raw_operations);
        CUDNN_FRONTEND_UNUSED(tensors);
        return cudnn_ver_error;
#endif  // CUDNN_VERSION >= 92100
    }
};

// Factory function implementation for INode::softmax
// This function will create the appropriate node based on the cuDNN version.
inline void
INode::softmax(std::shared_ptr<Tensor_attributes> p,
               Softmax_attributes attributes,
               std::shared_ptr<Tensor_attributes> s,
               std::shared_ptr<Tensor_attributes> stats,
               std::shared_ptr<Tensor_attributes> max,
               std::shared_ptr<Tensor_attributes> sum_exp) {
    attributes.inputs[Softmax_attributes::input_names::P]   = p;
    attributes.outputs[Softmax_attributes::output_names::S] = s;
    if (stats) {
        attributes.outputs[Softmax_attributes::output_names::Stats] = stats;
    }
    if (max) {
        attributes.outputs[Softmax_attributes::output_names::Max] = max;
    }
    if (sum_exp) {
        attributes.outputs[Softmax_attributes::output_names::Sum_exp] = sum_exp;
    }

    // Newer versions of cuDNN can represent the softmax as a single operation.
    if (std::min(detail::get_compiled_version(), detail::get_backend_version()) >= 92100) {
        sub_nodes.emplace_back(std::make_unique<UnifiedSoftmaxNode>(std::move(attributes), context));
    } else {
        sub_nodes.emplace_back(std::make_unique<CompositeSoftmaxNode>(std::move(attributes), context));
    }
}

}  // namespace cudnn_frontend::graph
