#pragma once

#include "../graph_helpers.h"
#include "../node_interface.h"

namespace cudnn_frontend::graph {

class MatmulFP8Node : public NodeCRTP<MatmulFP8Node> {
   public:
    Matmul_fp8_attributes attributes;

    MatmulFP8Node(Matmul_fp8_attributes&& attributes_, detail::Context const& context)
        : NodeCRTP(context), attributes(std::move(attributes_)) {}

    Type
    getType() override final {
        return Type::MATMUL;
    }

    error_t
    infer_properties_node() override final {
        CUDNN_FE_LOG_LABEL_ENDL("INFO:     Inferrencing properties for matmul fp8 node " << attributes.name);

        attributes.fill_from_context(context);

        auto const& a_dim = attributes.inputs.at(Matmul_fp8_attributes::input_names::A)->get_dim();
        auto const& b_dim = attributes.inputs.at(Matmul_fp8_attributes::input_names::B)->get_dim();
        auto const& c_dim = attributes.outputs.at(Matmul_fp8_attributes::output_names::C)->get_dim();

        std::shared_ptr<Tensor_attributes> last_output;

        // Matmul

        auto matmul_attributes = Matmul_attributes();
        matmul_attributes.clone_fp8_attributes(attributes);
        matmul_attributes.set_name("matmul");

        last_output = matmul(attributes.inputs.at(Matmul_fp8_attributes::input_names::A),
                             attributes.inputs.at(Matmul_fp8_attributes::input_names::B),
                             matmul_attributes);

        // Reduction if GQA for head dimension
        if (a_dim.size() == 4 && b_dim.size() == 4 && c_dim.size() == 4 && a_dim[1] == b_dim[1] &&
            a_dim[1] != c_dim[1] && (a_dim[1] % c_dim[1] == 0)) {
            auto gqa_attributes = Reduction_attributes().set_name("gqa_c").set_mode(ReductionMode_t::ADD);
            last_output         = reduction(last_output, gqa_attributes);
            last_output->set_dim(c_dim);
        }

        //// Scale Descales
        auto mul_attributes = Pointwise_attributes().set_mode(PointwiseMode_t::MUL);
        // Descale A
        mul_attributes.set_name("descale_a");
        last_output =
            pointwise(last_output, attributes.inputs.at(Matmul_fp8_attributes::input_names::Descale_A), mul_attributes);

        // Descale B
        mul_attributes.set_name("descale_b");
        last_output =
            pointwise(last_output, attributes.inputs.at(Matmul_fp8_attributes::input_names::Descale_B), mul_attributes);

        // Scale C
        mul_attributes.set_name("scale_c");
        // Special non-functional-style call. Needed because output already created and provided to user.
        pointwise(last_output,
                  attributes.inputs.at(Matmul_fp8_attributes::input_names::Scale_C),
                  mul_attributes,
                  attributes.outputs.at(Matmul_fp8_attributes::output_names::C));

        // Amax C
        auto amax_attributes = Reduction_attributes().set_name("amax_c").set_mode(ReductionMode_t::AMAX);
        // Special non-functional-style call. Needed because output already created and provided to user.
        reduction(last_output, amax_attributes, attributes.outputs.at(Matmul_fp8_attributes::output_names::Amax_C));

        return {error_code_t::OK, ""};
    }

#ifndef CUDNN_FRONTEND_SKIP_JSON_LIB
    virtual void
    serialize(json& j) const override final {
        j = attributes;
        j.update(R"( {"tag": "MATMUL_FP8"})"_json);
    }
#endif
};
inline void
INode::matmul_fp8(std::shared_ptr<Tensor_attributes> a,
                  std::shared_ptr<Tensor_attributes> b,
                  std::shared_ptr<Tensor_attributes> descale_a,
                  std::shared_ptr<Tensor_attributes> descale_b,
                  std::shared_ptr<Tensor_attributes> scale_c,
                  Matmul_fp8_attributes attributes,
                  std::shared_ptr<Tensor_attributes> c,
                  std::shared_ptr<Tensor_attributes> amax_c) {
    attributes.inputs[Matmul_fp8_attributes::input_names::A]         = a;
    attributes.inputs[Matmul_fp8_attributes::input_names::B]         = b;
    attributes.inputs[Matmul_fp8_attributes::input_names::Descale_A] = descale_a;
    attributes.inputs[Matmul_fp8_attributes::input_names::Descale_B] = descale_b;
    attributes.inputs[Matmul_fp8_attributes::input_names::Scale_C]   = scale_c;
    attributes.outputs[Matmul_fp8_attributes::output_names::C]       = c;
    attributes.outputs[Matmul_fp8_attributes::output_names::Amax_C]  = amax_c;
    sub_nodes.emplace_back(std::make_unique<MatmulFP8Node>(std::move(attributes), context));
}
}  // namespace cudnn_frontend::graph