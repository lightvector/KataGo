#pragma once

#include "../graph_helpers.h"
#include "../node_interface.h"

namespace cudnn_frontend::graph {

class RngNode : public NodeCRTP<RngNode> {
   public:
    Rng_attributes attributes;

    RngNode(Rng_attributes&& attributes_, detail::Context const& context)
        : NodeCRTP(context), attributes(std::move(attributes_)) {}

    Type
    getType() override final {
        return Type::RNG;
    }

    error_t
    infer_properties_node() override final {
        CUDNN_FE_LOG_LABEL_ENDL("INFO:     Inferrencing properties for rng node " << attributes.name);

        auto y_tensor = attributes.outputs[Rng_attributes::output_names::Y];

        attributes.fill_from_context(context);

        // If user does not set shape and layout of the generated tensor,
        // Get it from node attributes
        // If layout is not set, generate the strides from layout

        if (y_tensor->get_dim().empty() && attributes.get_dim().size()) {
            y_tensor->set_dim(attributes.dim);
        }

        if (y_tensor->get_stride().empty()) {
            if (attributes.get_stride().size()) {
                y_tensor->set_stride(attributes.get_stride());
            } else {
                auto const& y_dim = y_tensor->get_dim();
                // Default to NHWC
                auto const& stride_order = detail::generate_NHWC_stride_order(y_dim.size());
                y_tensor->set_stride(detail::generate_stride(y_dim, stride_order));
            }
        }

        if (y_tensor->get_dim().empty() || y_tensor->get_stride().empty()) {
            return {error_code_t::SHAPE_DEDUCTION_FAILED, "RNG node output shape deduction failed"};
        }

        return {error_code_t::OK, ""};
    }

    error_t
    create_cudnn_operations(
        std::unordered_set<Tensor_attributes::uid_t>& uids_involved_in_operations,
        std::vector<std::shared_ptr<cudnn_frontend::Operation>>& operations,
        managed_backend_descriptor_t& raw_operations,
        std::unordered_map<int64_t, std::shared_ptr<cudnn_frontend::Tensor>>& tensors) const override final {
        CUDNN_FRONTEND_UNUSED(raw_operations);
        CUDNN_FE_LOG_LABEL("INFO: " << "Building RngNode operations " << attributes.name << " ");

        RETURN_CUDNN_FRONTEND_ERROR_IF(attributes.get_distribution() != RngDistribution_t::BERNOULLI,
                                       error_code_t::ATTRIBUTE_NOT_SET,
                                       "no other distribution except bernoulli supported.");

        // Create RNG descriptor by directly calling cuDNN backend API
        RngDesc_v8 rng_descriptor;

        _CUDNN_CHECK_CUDNN_ERROR(rng_descriptor.initialize_managed_backend_pointer(CUDNN_BACKEND_RNG_DESCRIPTOR));

        // Set distribution type
        cudnnRngDistribution_t cudnn_rng_distribution;

        _CUDNN_CHECK_CUDNN_ERROR(detail::convert_to_cudnn_type(attributes.get_distribution(), cudnn_rng_distribution));

        _CUDNN_CHECK_CUDNN_ERROR(detail::set_attribute(rng_descriptor.get_raw_desc(),
                                                       CUDNN_ATTR_RNG_DISTRIBUTION,
                                                       CUDNN_TYPE_RNG_DISTRIBUTION,
                                                       1,
                                                       &cudnn_rng_distribution));

        // Set Bernoulli distribution probability
        double bernoulli_prob = attributes.get_bernoulli_probability().value();

        _CUDNN_CHECK_CUDNN_ERROR(detail::set_attribute(rng_descriptor.get_raw_desc(),
                                                       CUDNN_ATTR_RNG_BERNOULLI_DIST_PROBABILITY,
                                                       CUDNN_TYPE_DOUBLE,
                                                       1,
                                                       &bernoulli_prob));

        // Finalize the descriptor

        _CUDNN_CHECK_CUDNN_ERROR(detail::finalize(rng_descriptor.get_raw_desc()));
        CUDNN_FE_LOG_LABEL_ENDL(rng_descriptor);

        // Create operation by directly calling cuDNN backend API
        Operation_v8 rng_operation;

        _CUDNN_CHECK_CUDNN_ERROR(
            rng_operation.initialize_managed_backend_pointer(CUDNN_BACKEND_OPERATION_RNG_DESCRIPTOR));

        // Set output tensor Y
        CUDNN_FE_VALIDATE_AND_ASSIGN_OUTPUT_TENSOR(Y, Rng_attributes::output_names::Y);
        auto y_desc = tensors.at(Y->second->get_uid())->get_raw_desc();

        _CUDNN_CHECK_CUDNN_ERROR(detail::set_attribute(
            rng_operation.get_raw_desc(), CUDNN_ATTR_OPERATION_RNG_YDESC, CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &y_desc));

        // Set RNG descriptor
        auto rng_raw_desc = rng_descriptor.get_raw_desc();

        _CUDNN_CHECK_CUDNN_ERROR(detail::set_attribute(rng_operation.get_raw_desc(),
                                                       CUDNN_ATTR_OPERATION_RNG_DESC,
                                                       CUDNN_TYPE_BACKEND_DESCRIPTOR,
                                                       1,
                                                       &rng_raw_desc));

        if (attributes.seed.has_value()) {
            // Set seed as int64_t value
            int64_t seed_value = attributes.get_seed().value();

            _CUDNN_CHECK_CUDNN_ERROR(detail::set_attribute(
                rng_operation.get_raw_desc(), CUDNN_ATTR_OPERATION_RNG_SEED, CUDNN_TYPE_INT64, 1, &seed_value));
        } else {
            // Set seed tensor descriptor
            CUDNN_FE_VALIDATE_AND_ASSIGN_INPUT_TENSOR(Seed, Rng_attributes::input_names::Seed);
            auto seed_desc = tensors.at(Seed->second->get_uid())->get_raw_desc();

            _CUDNN_CHECK_CUDNN_ERROR(detail::set_attribute(rng_operation.get_raw_desc(),
                                                           CUDNN_ATTR_OPERATION_RNG_SEED,
                                                           CUDNN_TYPE_BACKEND_DESCRIPTOR,
                                                           1,
                                                           &seed_desc));

            // Set offset tensor descriptor
            CUDNN_FE_VALIDATE_AND_ASSIGN_INPUT_TENSOR(Offset, Rng_attributes::input_names::Offset);
            auto offset_desc = tensors.at(Offset->second->get_uid())->get_raw_desc();

            _CUDNN_CHECK_CUDNN_ERROR(detail::set_attribute(rng_operation.get_raw_desc(),
                                                           CUDNN_ATTR_OPERATION_RNG_OFFSET_DESC,
                                                           CUDNN_TYPE_BACKEND_DESCRIPTOR,
                                                           1,
                                                           &offset_desc));
        }

        _CUDNN_CHECK_CUDNN_ERROR(detail::finalize(rng_operation.get_raw_desc()));

        operations.push_back(std::make_shared<Operation_v8>(std::move(rng_operation)));

        auto const& non_virtual_uids = attributes.get_non_virtual_uids();
        uids_involved_in_operations.insert(non_virtual_uids.begin(), non_virtual_uids.end());
        return {error_code_t::OK, ""};
    }

#ifndef CUDNN_FRONTEND_SKIP_JSON_LIB
    virtual void
    serialize(json& j) const override final {
        j = attributes;
        j.update(R"( {"tag": "RNG"})"_json);
    }
#endif
};

inline void
INode::rng(std::shared_ptr<Tensor_attributes> seed,
           std::shared_ptr<Tensor_attributes> offset,
           Rng_attributes attributes,
           std::shared_ptr<Tensor_attributes> y) {
    attributes.inputs[Rng_attributes::input_names::Seed]   = seed;
    attributes.inputs[Rng_attributes::input_names::Offset] = offset;
    attributes.outputs[Rng_attributes::output_names::Y]    = y;
    sub_nodes.emplace_back(std::make_unique<RngNode>(std::move(attributes), context));
}

inline std::shared_ptr<Tensor_attributes>
INode::rng(std::shared_ptr<Tensor_attributes> seed,
           std::shared_ptr<Tensor_attributes> offset,
           Rng_attributes attributes) {
    attributes.inputs[Rng_attributes::input_names::Seed]   = seed;
    attributes.inputs[Rng_attributes::input_names::Offset] = offset;
    auto Y = attributes.outputs[Rng_attributes::output_names::Y] = output_tensor(attributes.name + "::Y");

    sub_nodes.emplace_back(std::make_unique<RngNode>(std::move(attributes), context));
    return Y;
}
}  // namespace cudnn_frontend::graph