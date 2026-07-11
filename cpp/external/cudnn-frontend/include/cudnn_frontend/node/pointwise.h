#pragma once

#include "../graph_helpers.h"
#include "../node_interface.h"

namespace cudnn_frontend::graph {

class PointwiseNode : public NodeCRTP<PointwiseNode> {
   public:
    Pointwise_attributes attributes;

    PointwiseNode(Pointwise_attributes&& attributes_, detail::Context const& context)
        : NodeCRTP(context), attributes(std::move(attributes_)) {}

    Type
    getType() override final {
        return Type::POINTWISE;
    }

    error_t
    infer_properties_node() override final {
        CUDNN_FE_LOG_LABEL_ENDL("INFO:     Inferrencing properties for pointwise node " << attributes.name);

        attributes.fill_from_context(context);

        auto out_0_tensor = attributes.outputs.at(Pointwise_attributes::output_names::OUT_0);

        auto output_dim = out_0_tensor->get_dim();
        // Only infer dims and strides if user did not set them
        if (output_dim.empty()) {
            std::vector<std::vector<int64_t>> input_shapes;
            for (const auto& [input_name, input_tensor] : attributes.inputs) {
                if (!input_tensor) {
                    continue;
                }
                input_shapes.push_back(input_tensor->get_dim());
            }

            CHECK_CUDNN_FRONTEND_ERROR(detail::compute_broadcast_shape(input_shapes, output_dim));
            out_0_tensor->set_dim(output_dim);
        }

        if (out_0_tensor->get_stride().empty()) {
            for (const auto& [input_name, input_tensor] : attributes.inputs) {
                if (input_tensor == nullptr) {
                    continue;
                }
                if (input_tensor->get_dim() == out_0_tensor->get_dim()) {
                    CUDNN_FE_LOG_LABEL_ENDL("INFO:" << "        " << out_0_tensor->get_name()
                                                    << " stride computed from " << input_tensor->get_name());
                    out_0_tensor->set_stride(input_tensor->get_stride());
                    break;
                }
            }
            if (out_0_tensor->get_stride().empty() && out_0_tensor->get_is_virtual()) {
                // If the tensor is virtual the strides are immaterial
                auto input_stride = attributes.inputs.at(Pointwise_attributes::input_names::IN_0)->get_stride();
                std::vector<int64_t> stride_order;
                CHECK_CUDNN_FRONTEND_ERROR(
                    detail::generate_stride_order_preserving_format(input_stride, output_dim.size(), stride_order));
                out_0_tensor->set_stride(detail::generate_stride(output_dim, stride_order));
            }
            RETURN_CUDNN_FRONTEND_ERROR_IF(out_0_tensor->get_stride().empty(),
                                           error_code_t::SHAPE_DEDUCTION_FAILED,
                                           "Pointwise output strides could not be computed");
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
        CUDNN_FE_LOG_LABEL("INFO: " << "Building PointwiseNode operations " << attributes.name << " ");

        // Create pointwise descriptor by directly calling cuDNN backend API
        PointWiseDesc_v8 pointwise_descriptor;

        _CUDNN_CHECK_CUDNN_ERROR(
            pointwise_descriptor.initialize_managed_backend_pointer(CUDNN_BACKEND_POINTWISE_DESCRIPTOR));

        // Set pointwise mode
        cudnnPointwiseMode_t cudnn_pointwise_mode;

        _CUDNN_CHECK_CUDNN_ERROR(detail::convert_to_cudnn_type(attributes.mode, cudnn_pointwise_mode));

        _CUDNN_CHECK_CUDNN_ERROR(detail::set_attribute(pointwise_descriptor.get_raw_desc(),
                                                       CUDNN_ATTR_POINTWISE_MODE,
                                                       CUDNN_TYPE_POINTWISE_MODE,
                                                       1,
                                                       &cudnn_pointwise_mode));

        // Set compute type
        cudnnDataType_t cudnn_data_type;

        _CUDNN_CHECK_CUDNN_ERROR(detail::convert_to_cudnn_type(attributes.compute_data_type, cudnn_data_type));

        _CUDNN_CHECK_CUDNN_ERROR(detail::set_attribute(pointwise_descriptor.get_raw_desc(),
                                                       CUDNN_ATTR_POINTWISE_MATH_PREC,
                                                       CUDNN_TYPE_DATA_TYPE,
                                                       1,
                                                       &cudnn_data_type));

        // Set mode-specific attributes
        if (attributes.mode == PointwiseMode_t::RELU_FWD || attributes.mode == PointwiseMode_t::RELU_BWD) {
            cudnnNanPropagation_t nan_propagation = CUDNN_PROPAGATE_NAN;

            _CUDNN_CHECK_CUDNN_ERROR(detail::set_attribute(pointwise_descriptor.get_raw_desc(),
                                                           CUDNN_ATTR_POINTWISE_NAN_PROPAGATION,
                                                           CUDNN_TYPE_NAN_PROPOGATION,
                                                           1,
                                                           &nan_propagation));

            double lower_clip = attributes.relu_lower_clip.value_or(0.0);

            _CUDNN_CHECK_CUDNN_ERROR(detail::set_attribute(pointwise_descriptor.get_raw_desc(),
                                                           CUDNN_ATTR_POINTWISE_RELU_LOWER_CLIP,
                                                           CUDNN_TYPE_DOUBLE,
                                                           1,
                                                           &lower_clip));

            double upper_clip = attributes.relu_upper_clip.value_or(std::numeric_limits<double>::max());
            if (attributes.compute_data_type == DataType_t::FLOAT) {
                upper_clip = std::min<double>(upper_clip, std::numeric_limits<float>::max());
            }

            _CUDNN_CHECK_CUDNN_ERROR(detail::set_attribute(pointwise_descriptor.get_raw_desc(),
                                                           CUDNN_ATTR_POINTWISE_RELU_UPPER_CLIP,
                                                           CUDNN_TYPE_DOUBLE,
                                                           1,
                                                           &upper_clip));

            double lower_clip_slope = attributes.relu_lower_clip_slope.value_or(0.0);

            _CUDNN_CHECK_CUDNN_ERROR(detail::set_attribute(pointwise_descriptor.get_raw_desc(),
                                                           CUDNN_ATTR_POINTWISE_RELU_LOWER_CLIP_SLOPE,
                                                           CUDNN_TYPE_DOUBLE,
                                                           1,
                                                           &lower_clip_slope));
        } else if (attributes.mode == PointwiseMode_t::ELU_FWD || attributes.mode == PointwiseMode_t::ELU_BWD) {
            double elu_alpha = attributes.elu_alpha.value_or(1.0);

            _CUDNN_CHECK_CUDNN_ERROR(detail::set_attribute(
                pointwise_descriptor.get_raw_desc(), CUDNN_ATTR_POINTWISE_ELU_ALPHA, CUDNN_TYPE_DOUBLE, 1, &elu_alpha));
        } else if (attributes.mode == PointwiseMode_t::SOFTPLUS_FWD ||
                   attributes.mode == PointwiseMode_t::SOFTPLUS_BWD) {
            double softplus_beta = attributes.softplus_beta.value_or(1.0);

            _CUDNN_CHECK_CUDNN_ERROR(detail::set_attribute(pointwise_descriptor.get_raw_desc(),
                                                           CUDNN_ATTR_POINTWISE_SOFTPLUS_BETA,
                                                           CUDNN_TYPE_DOUBLE,
                                                           1,
                                                           &softplus_beta));
        } else if (attributes.mode == PointwiseMode_t::SWISH_FWD || attributes.mode == PointwiseMode_t::SWISH_BWD) {
            double swish_beta = attributes.swish_beta.value_or(1.0);

            _CUDNN_CHECK_CUDNN_ERROR(detail::set_attribute(pointwise_descriptor.get_raw_desc(),
                                                           CUDNN_ATTR_POINTWISE_SWISH_BETA,
                                                           CUDNN_TYPE_DOUBLE,
                                                           1,
                                                           &swish_beta));
        } else if (attributes.mode == PointwiseMode_t::GEN_INDEX) {
            int64_t axis = attributes.get_axis().value_or(-1);

            _CUDNN_CHECK_CUDNN_ERROR(detail::set_attribute(
                pointwise_descriptor.get_raw_desc(), CUDNN_ATTR_POINTWISE_AXIS, CUDNN_TYPE_INT64, 1, &axis));
        }

        _CUDNN_CHECK_CUDNN_ERROR(detail::finalize(pointwise_descriptor.get_raw_desc()));
        CUDNN_FE_LOG_LABEL_ENDL(pointwise_descriptor);

        // Create operation by directly calling cuDNN backend API
        Operation_v8 pointwise_operation;

        _CUDNN_CHECK_CUDNN_ERROR(
            pointwise_operation.initialize_managed_backend_pointer(CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR));

        // Set the pointwise descriptor
        auto pw_desc_ptr = pointwise_descriptor.get_raw_desc();

        _CUDNN_CHECK_CUDNN_ERROR(detail::set_attribute(pointwise_operation.get_raw_desc(),
                                                       CUDNN_ATTR_OPERATION_POINTWISE_PW_DESCRIPTOR,
                                                       CUDNN_TYPE_BACKEND_DESCRIPTOR,
                                                       1,
                                                       &pw_desc_ptr));

        auto const port_count        = get_pointwise_mode_port_count(attributes.mode);
        bool const is_activation_bwd = detail::is_activation_backward_mode(attributes.mode);

        if (is_activation_bwd) {
            // Backward mode: IN_0 is dy, IN_1 is x, OUT_0 is dx
            CUDNN_FE_VALIDATE_AND_ASSIGN_INPUT_TENSOR(IN_0, Pointwise_attributes::input_names::IN_0);
            auto dy_desc = tensors.at(IN_0->second->get_uid())->get_raw_desc();

            CUDNN_FE_VALIDATE_AND_ASSIGN_INPUT_TENSOR(IN_1, Pointwise_attributes::input_names::IN_1);
            auto x_desc = tensors.at(IN_1->second->get_uid())->get_raw_desc();

            _CUDNN_CHECK_CUDNN_ERROR(detail::set_attribute(pointwise_operation.get_raw_desc(),
                                                           CUDNN_ATTR_OPERATION_POINTWISE_XDESC,
                                                           CUDNN_TYPE_BACKEND_DESCRIPTOR,
                                                           1,
                                                           &x_desc));

            _CUDNN_CHECK_CUDNN_ERROR(detail::set_attribute(pointwise_operation.get_raw_desc(),
                                                           CUDNN_ATTR_OPERATION_POINTWISE_DYDESC,
                                                           CUDNN_TYPE_BACKEND_DESCRIPTOR,
                                                           1,
                                                           &dy_desc));

            CUDNN_FE_VALIDATE_AND_ASSIGN_OUTPUT_TENSOR(OUT_0, Pointwise_attributes::output_names::OUT_0);
            auto dx_desc = tensors.at(OUT_0->second->get_uid())->get_raw_desc();

            _CUDNN_CHECK_CUDNN_ERROR(detail::set_attribute(pointwise_operation.get_raw_desc(),
                                                           CUDNN_ATTR_OPERATION_POINTWISE_DXDESC,
                                                           CUDNN_TYPE_BACKEND_DESCRIPTOR,
                                                           1,
                                                           &dx_desc));
        } else {
            // Forward mode
            CUDNN_FE_VALIDATE_AND_ASSIGN_INPUT_TENSOR(IN_0, Pointwise_attributes::input_names::IN_0);
            auto x_desc = tensors.at(IN_0->second->get_uid())->get_raw_desc();

            _CUDNN_CHECK_CUDNN_ERROR(detail::set_attribute(pointwise_operation.get_raw_desc(),
                                                           CUDNN_ATTR_OPERATION_POINTWISE_XDESC,
                                                           CUDNN_TYPE_BACKEND_DESCRIPTOR,
                                                           1,
                                                           &x_desc));

            CUDNN_FE_VALIDATE_AND_ASSIGN_OUTPUT_TENSOR(OUT_0, Pointwise_attributes::output_names::OUT_0);
            auto y_desc = tensors.at(OUT_0->second->get_uid())->get_raw_desc();

            _CUDNN_CHECK_CUDNN_ERROR(detail::set_attribute(pointwise_operation.get_raw_desc(),
                                                           CUDNN_ATTR_OPERATION_POINTWISE_YDESC,
                                                           CUDNN_TYPE_BACKEND_DESCRIPTOR,
                                                           1,
                                                           &y_desc));

            if (port_count >= 3) {
                CUDNN_FE_VALIDATE_AND_ASSIGN_INPUT_TENSOR(IN_1, Pointwise_attributes::input_names::IN_1);
                auto b_desc = tensors.at(IN_1->second->get_uid())->get_raw_desc();

                _CUDNN_CHECK_CUDNN_ERROR(detail::set_attribute(pointwise_operation.get_raw_desc(),
                                                               CUDNN_ATTR_OPERATION_POINTWISE_BDESC,
                                                               CUDNN_TYPE_BACKEND_DESCRIPTOR,
                                                               1,
                                                               &b_desc));
            }

            if (port_count >= 4) {
                CUDNN_FE_VALIDATE_AND_ASSIGN_INPUT_TENSOR(IN_2, Pointwise_attributes::input_names::IN_2);
                auto t_desc = tensors.at(IN_2->second->get_uid())->get_raw_desc();

                _CUDNN_CHECK_CUDNN_ERROR(detail::set_attribute(pointwise_operation.get_raw_desc(),
                                                               CUDNN_ATTR_OPERATION_POINTWISE_TDESC,
                                                               CUDNN_TYPE_BACKEND_DESCRIPTOR,
                                                               1,
                                                               &t_desc));
            }
        }

        // Set alpha scaling factors (always set to 1.0)
        float alpha1 = 1.0f;
        float alpha2 = 1.0f;

        _CUDNN_CHECK_CUDNN_ERROR(detail::set_attribute(
            pointwise_operation.get_raw_desc(), CUDNN_ATTR_OPERATION_POINTWISE_ALPHA1, CUDNN_TYPE_FLOAT, 1, &alpha1));

        _CUDNN_CHECK_CUDNN_ERROR(detail::set_attribute(
            pointwise_operation.get_raw_desc(), CUDNN_ATTR_OPERATION_POINTWISE_ALPHA2, CUDNN_TYPE_FLOAT, 1, &alpha2));

        _CUDNN_CHECK_CUDNN_ERROR(detail::finalize(pointwise_operation.get_raw_desc()));

        operations.push_back(std::make_shared<Operation_v8>(std::move(pointwise_operation)));

        auto const& non_virtual_uids = attributes.get_non_virtual_uids();
        uids_involved_in_operations.insert(non_virtual_uids.begin(), non_virtual_uids.end());
        return {error_code_t::OK, ""};
    }

#ifndef CUDNN_FRONTEND_SKIP_JSON_LIB
    virtual void
    serialize(json& j) const override final {
        j = attributes;
        j.update(R"({"tag": "POINTWISE"})"_json);
    }
#endif
};

inline void
INode::pointwise(std::shared_ptr<Tensor_attributes> a,
                 Pointwise_attributes attributes,
                 std::shared_ptr<Tensor_attributes> c) {
    attributes.inputs[Pointwise_attributes::input_names::IN_0]    = a;
    attributes.outputs[Pointwise_attributes::output_names::OUT_0] = c;
    sub_nodes.emplace_back(std::make_unique<PointwiseNode>(std::move(attributes), context));
}

inline void
INode::pointwise(std::shared_ptr<Tensor_attributes> a,
                 std::shared_ptr<Tensor_attributes> b,
                 Pointwise_attributes attributes,
                 std::shared_ptr<Tensor_attributes> c) {
    attributes.inputs[Pointwise_attributes::input_names::IN_0]    = a;
    attributes.inputs[Pointwise_attributes::input_names::IN_1]    = b;
    attributes.outputs[Pointwise_attributes::output_names::OUT_0] = c;
    sub_nodes.emplace_back(std::make_unique<PointwiseNode>(std::move(attributes), context));
}

inline void
INode::pointwise(std::shared_ptr<Tensor_attributes> a,
                 std::shared_ptr<Tensor_attributes> b,
                 std::shared_ptr<Tensor_attributes> c,
                 Pointwise_attributes attributes,
                 std::shared_ptr<Tensor_attributes> d) {
    attributes.inputs[Pointwise_attributes::input_names::IN_0]    = a;
    attributes.inputs[Pointwise_attributes::input_names::IN_1]    = b;
    attributes.inputs[Pointwise_attributes::input_names::IN_2]    = c;
    attributes.outputs[Pointwise_attributes::output_names::OUT_0] = d;
    sub_nodes.emplace_back(std::make_unique<PointwiseNode>(std::move(attributes), context));
}

inline std::shared_ptr<Tensor_attributes>
INode::pointwise(std::shared_ptr<Tensor_attributes> a, Pointwise_attributes attributes) {
    if (attributes.name.empty()) {
        attributes.name += std::to_string(sub_nodes.size());
    }
    attributes.inputs[Pointwise_attributes::input_names::IN_0] = a;
    if (a->get_name().empty()) {
        a->set_name(attributes.name + "::IN_0");
    };
    auto OUT_0 = attributes.outputs[Pointwise_attributes::output_names::OUT_0] =
        output_tensor(attributes.name + "::OUT_0");

    sub_nodes.emplace_back(std::make_unique<PointwiseNode>(std::move(attributes), context));
    return OUT_0;
}

inline std::shared_ptr<Tensor_attributes>
INode::pointwise(std::shared_ptr<Tensor_attributes> a,
                 std::shared_ptr<Tensor_attributes> b,
                 Pointwise_attributes attributes) {
    if (attributes.name.empty()) {
        attributes.name += std::to_string(sub_nodes.size());
    }
    attributes.inputs[Pointwise_attributes::input_names::IN_0] = a;
    attributes.inputs[Pointwise_attributes::input_names::IN_1] = b;
    if (a->get_name().empty()) {
        a->set_name(attributes.name + "::IN_0");
    };
    if (b->get_name().empty()) {
        b->set_name(attributes.name + "::IN_1");
    };
    auto OUT_0 = attributes.outputs[Pointwise_attributes::output_names::OUT_0] =
        output_tensor(attributes.name + "::OUT_0");

    sub_nodes.emplace_back(std::make_unique<PointwiseNode>(std::move(attributes), context));
    return OUT_0;
}

inline std::shared_ptr<Tensor_attributes>
INode::pointwise(std::shared_ptr<Tensor_attributes> a,
                 std::shared_ptr<Tensor_attributes> b,
                 std::shared_ptr<Tensor_attributes> c,
                 Pointwise_attributes attributes) {
    if (attributes.name.empty()) {
        attributes.name += std::to_string(sub_nodes.size());
    }
    attributes.inputs[Pointwise_attributes::input_names::IN_0] = a;
    attributes.inputs[Pointwise_attributes::input_names::IN_1] = b;
    attributes.inputs[Pointwise_attributes::input_names::IN_2] = c;
    if (a->get_name().empty()) {
        a->set_name(attributes.name + "::IN_0");
    };
    if (b->get_name().empty()) {
        b->set_name(attributes.name + "::IN_1");
    };
    if (c->get_name().empty()) {
        c->set_name(attributes.name + "::IN_2");
    };
    auto OUT_0 = attributes.outputs[Pointwise_attributes::output_names::OUT_0] =
        output_tensor(attributes.name + "::OUT_0");

    sub_nodes.emplace_back(std::make_unique<PointwiseNode>(std::move(attributes), context));
    return OUT_0;
}
}  // namespace cudnn_frontend::graph