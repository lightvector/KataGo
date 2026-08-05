#pragma once

#include "../graph_helpers.h"
#include "../node_interface.h"

#include "pointwise.h"
#include "reduction.h"

namespace cudnn_frontend::graph {

// Base class for nodes that represent a diagonal band mask operation
// (which is either a left-bound or a right-bound mask, but not both at the same time).
template <typename DerivedT>
class DiagonalBandMaskNodeBase : public NodeCRTP<DerivedT> {
   protected:
    using input_names  = DiagonalBandMask_attributes::input_names;
    using output_names = DiagonalBandMask_attributes::output_names;

   public:
    DiagonalBandMask_attributes attributes;

    DiagonalBandMaskNodeBase(DiagonalBandMask_attributes&& attributes_, detail::Context const& context)
        : NodeCRTP<DerivedT>(context), attributes(std::move(attributes_)) {}

    error_t
    pre_validate_node() const override final {
        CUDNN_FE_LOG_LABEL_ENDL("INFO: Validating DiagonalBandMaskNode " << attributes.name);

        RETURN_CUDNN_FRONTEND_ERROR_IF(has_left_bound() && has_shift_right_bound(),
                                       error_code_t::INVALID_VALUE,
                                       "DiagonalBandMaskNode cannot have both left_bound and shift_right_bound");

        return {error_code_t::OK, ""};
    }

    error_t
    infer_properties_node() override final {
        CUDNN_FE_LOG_LABEL_ENDL("INFO:     Inferrencing properties for DiagonalBandMask node " << attributes.name);

        attributes.fill_from_context(this->context);

        auto input_tensor  = attributes.inputs.at(DiagonalBandMask_attributes::input_names::X);
        auto output_tensor = attributes.outputs.at(DiagonalBandMask_attributes::output_names::Y);

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
    has_seq_len_q() const {
        auto seq_len_Q_it = attributes.inputs.find(DiagonalBandMask_attributes::input_names::SEQ_LEN_Q);
        return ((seq_len_Q_it) != attributes.inputs.end() && seq_len_Q_it->second != nullptr);
    }

    bool
    has_seq_len_kv() const {
        auto seq_len_KV_it = attributes.inputs.find(DiagonalBandMask_attributes::input_names::SEQ_LEN_KV);
        return ((seq_len_KV_it) != attributes.inputs.end() && seq_len_KV_it->second != nullptr);
    }

    bool
    has_left_bound() const {
        auto left_bound_it = attributes.inputs.find(DiagonalBandMask_attributes::input_names::LeftBound);
        return ((left_bound_it) != attributes.inputs.end() && left_bound_it->second != nullptr);
    }

    bool
    has_right_bound() const {
        return !has_left_bound();
    }

    bool
    has_shift_right_bound() const {
        auto shift_right_bound_it = attributes.inputs.find(DiagonalBandMask_attributes::input_names::ShiftRightBound);
        return ((shift_right_bound_it) != attributes.inputs.end() && shift_right_bound_it->second != nullptr);
    }

#ifndef CUDNN_FRONTEND_SKIP_JSON_LIB
    virtual void
    serialize(json& j) const override final {
        j = attributes;
    }
#endif
};

// Fallback implementation of diagonal band mask node that represents the mask as a series of pointwise operations.
// This is used for cuDNN versions before v9.21.0.
// Note: the individual masks can be constructed in different ways as well that yield functionally
// correct results. However, for performance reasons in the cuDNN backend they are organized as they are. Be
// cautious of performance when editing.
class CompositeDiagonalBandMaskNode : public DiagonalBandMaskNodeBase<CompositeDiagonalBandMaskNode> {
   public:
    CompositeDiagonalBandMaskNode(DiagonalBandMask_attributes&& attributes_, detail::Context const& context)
        : DiagonalBandMaskNodeBase(std::move(attributes_), context) {}

    Type
    getType() override final {
        return Type::COMPOSITE;
    }

    error_t
    expand_node() override final {
        CUDNN_FE_LOG_LABEL_ENDL("INFO:     Inferrencing properties for CompositeDiagonalBandMaskNode "
                                << attributes.name);

        attributes.fill_from_context(context);

        auto attention_score    = attributes.inputs[DiagonalBandMask_attributes::input_names::X];
        auto negative_inf_value = attributes.inputs[DiagonalBandMask_attributes::input_names::B];

        auto row_index = pointwise(
            attention_score,
            Pointwise_attributes().set_name("gen_row_index").set_mode(PointwiseMode_t::GEN_INDEX).set_axis(2));
        row_index->set_data_type(DataType_t::INT32);
        auto col_index = pointwise(
            attention_score,
            Pointwise_attributes().set_name("gen_col_index").set_mode(PointwiseMode_t::GEN_INDEX).set_axis(3));
        col_index->set_data_type(DataType_t::INT32);

        std::shared_ptr<Tensor_attributes> swa_comparison_output;

        if (has_right_bound()) {
            std::string row_index_name = "row";

            if (has_seq_len_kv()) {
                auto seq_len_kv = attributes.inputs[DiagonalBandMask_attributes::input_names::SEQ_LEN_KV];
                row_index       = pointwise(row_index,
                                      seq_len_kv,
                                      Pointwise_attributes()
                                          .set_name(row_index_name += "+skv")
                                          .set_mode(PointwiseMode_t::ADD)
                                          .set_compute_data_type(DataType_t::INT32)
                                          .set_axis(3));
                row_index->set_data_type(DataType_t::INT32);
            }

            if (has_seq_len_q()) {
                auto seq_len_q = attributes.inputs[DiagonalBandMask_attributes::input_names::SEQ_LEN_Q];
                row_index      = pointwise(row_index,
                                      seq_len_q,
                                      Pointwise_attributes()
                                          .set_name(row_index_name += "-sq")
                                          .set_mode(PointwiseMode_t::SUB)
                                          .set_compute_data_type(DataType_t::INT32)
                                          .set_axis(3));
                row_index->set_data_type(DataType_t::INT32);
            }

            if (has_shift_right_bound()) {
                auto shift_right_bound = attributes.inputs[DiagonalBandMask_attributes::input_names::ShiftRightBound];
                // Use shift_right_bound's data type
                row_index = pointwise(row_index,
                                      shift_right_bound,
                                      Pointwise_attributes()
                                          .set_name(row_index_name += "+window")
                                          .set_mode(PointwiseMode_t::ADD)
                                          .set_compute_data_type(shift_right_bound->get_data_type())
                                          .set_axis(3));
                row_index->set_data_type(shift_right_bound->get_data_type());
            }

            swa_comparison_output = pointwise(row_index,
                                              col_index,
                                              Pointwise_attributes()
                                                  .set_name(row_index_name += ">col")
                                                  .set_mode(attributes.comparison_mode)
                                                  .set_compute_data_type(DataType_t::BOOLEAN));
            swa_comparison_output->set_data_type(DataType_t::BOOLEAN);
        }

        // Set the left bound
        else if (has_left_bound()) {
            auto left_bound            = attributes.inputs[DiagonalBandMask_attributes::input_names::LeftBound];
            std::string col_index_name = "col";

            // Use left_bound's data type
            col_index = pointwise(col_index,
                                  left_bound,
                                  Pointwise_attributes()
                                      .set_name(col_index_name += "+window")
                                      .set_mode(PointwiseMode_t::ADD)
                                      .set_compute_data_type(left_bound->get_data_type())
                                      .set_axis(3));
            col_index->set_data_type(left_bound->get_data_type());

            if (has_seq_len_kv()) {
                auto seq_len_kv = attributes.inputs[DiagonalBandMask_attributes::input_names::SEQ_LEN_KV];
                col_index       = pointwise(col_index,
                                      seq_len_kv,
                                      Pointwise_attributes()
                                          .set_name(col_index_name += "-skv")
                                          .set_mode(PointwiseMode_t::SUB)
                                          .set_compute_data_type(DataType_t::INT32)
                                          .set_axis(3));
                col_index->set_data_type(DataType_t::INT32);
            }

            if (has_seq_len_q()) {
                auto seq_len_q = attributes.inputs[DiagonalBandMask_attributes::input_names::SEQ_LEN_Q];
                col_index      = pointwise(col_index,
                                      seq_len_q,
                                      Pointwise_attributes()
                                          .set_name(col_index_name += "+sq")
                                          .set_mode(PointwiseMode_t::ADD)
                                          .set_compute_data_type(DataType_t::INT32)
                                          .set_axis(3));
                col_index->set_data_type(DataType_t::INT32);
            }

            swa_comparison_output = pointwise(col_index,
                                              row_index,
                                              Pointwise_attributes()
                                                  .set_name(col_index_name += ">row")
                                                  .set_mode(attributes.comparison_mode)
                                                  .set_compute_data_type(DataType_t::BOOLEAN));
            swa_comparison_output->set_data_type(DataType_t::BOOLEAN);
        } else {
            // This should not be reached (since one of has_left_bound() or has_right_bound() is always true by
            // construction), just here for completeness.
            return {error_code_t::INVALID_VALUE,
                    "CompositeDiagonalBandMaskNode must have a left bound or a right bound"};
        }

        // Special non-functional-style call. Needed because output already created and provided to user.
        pointwise(attention_score,
                  negative_inf_value,
                  swa_comparison_output,
                  Pointwise_attributes().set_name("binary_select").set_mode(PointwiseMode_t::BINARY_SELECT),
                  attributes.outputs[DiagonalBandMask_attributes::output_names::Y]);

        return {error_code_t::OK, ""};
    }
};

// Newer implementation of diagonal band mask node that represents the mask as a single operation.
// This is used for cuDNN versions v9.21.0 and later.
class UnifiedDiagonalBandMaskNode : public DiagonalBandMaskNodeBase<UnifiedDiagonalBandMaskNode> {
   public:
    UnifiedDiagonalBandMaskNode(DiagonalBandMask_attributes&& attributes_, detail::Context const& context)
        : DiagonalBandMaskNodeBase(std::move(attributes_), context) {}

    Type
    getType() override final {
        return Type::DIAGONAL_BAND_MASK;
    }

    error_t
    create_cudnn_operations(
        std::unordered_set<Tensor_attributes::uid_t>& uids_involved_in_operations,
        std::vector<std::shared_ptr<cudnn_frontend::Operation>>& operations,
        managed_backend_descriptor_t& raw_operations,
        std::unordered_map<int64_t, std::shared_ptr<cudnn_frontend::Tensor>>& tensors) const override final {
        getLogger() << "[cudnn_frontend] INFO: " << "Building UnifiedDiagonalBandMaskNode operations "
                    << attributes.name << std::endl;
        auto cudnn_ver_error =
            error_t{error_code_t::GRAPH_NOT_SUPPORTED, "UnifiedDiagonalBandMaskNode requires cuDNN v9.21.0"};

#if (CUDNN_VERSION >= 92100)
        NV_CUDNN_FE_DYNAMIC_CHECK_CUDNN_BACKEND_VERSION(92100, cudnn_ver_error);
        CUDNN_FRONTEND_UNUSED(operations);
        auto diagonal_band_mask_operation = make_shared_backend_pointer(
            (cudnnBackendDescriptorType_t)CUDNN_BACKEND_OPERATION_DIAGONAL_BAND_MASK_DESCRIPTOR);

        auto X         = attributes.inputs.find(DiagonalBandMask_attributes::input_names::X)->second;
        auto backend_x = tensors[X->get_uid()]->get_desc()->get_backend_descriptor();
        _CUDNN_CHECK_CUDNN_ERROR(detail::set_attribute(diagonal_band_mask_operation->get_backend_descriptor(),
                                                       CUDNN_ATTR_OPERATION_DIAGONAL_BAND_MASK_XDESC,
                                                       CUDNN_TYPE_BACKEND_DESCRIPTOR,
                                                       1,
                                                       &backend_x));

        auto B         = attributes.inputs.find(DiagonalBandMask_attributes::input_names::B)->second;
        auto backend_b = tensors[B->get_uid()]->get_desc()->get_backend_descriptor();
        _CUDNN_CHECK_CUDNN_ERROR(detail::set_attribute(diagonal_band_mask_operation->get_backend_descriptor(),
                                                       CUDNN_ATTR_OPERATION_DIAGONAL_BAND_MASK_BDESC,
                                                       CUDNN_TYPE_BACKEND_DESCRIPTOR,
                                                       1,
                                                       &backend_b));

        if (has_seq_len_q()) {
            auto seq_len_Q = attributes.inputs.find(DiagonalBandMask_attributes::input_names::SEQ_LEN_Q)->second;
            auto backend_seq_len_Q = tensors[seq_len_Q->get_uid()]->get_desc()->get_backend_descriptor();
            _CUDNN_CHECK_CUDNN_ERROR(detail::set_attribute(diagonal_band_mask_operation->get_backend_descriptor(),
                                                           CUDNN_ATTR_OPERATION_DIAGONAL_BAND_MASK_SEQ_LEN_QDESC,
                                                           CUDNN_TYPE_BACKEND_DESCRIPTOR,
                                                           1,
                                                           &backend_seq_len_Q));
        }

        if (has_seq_len_kv()) {
            auto seq_len_KV = attributes.inputs.find(DiagonalBandMask_attributes::input_names::SEQ_LEN_KV)->second;
            auto backend_seq_len_KV = tensors[seq_len_KV->get_uid()]->get_desc()->get_backend_descriptor();
            _CUDNN_CHECK_CUDNN_ERROR(detail::set_attribute(diagonal_band_mask_operation->get_backend_descriptor(),
                                                           CUDNN_ATTR_OPERATION_DIAGONAL_BAND_MASK_SEQ_LEN_KVDESC,
                                                           CUDNN_TYPE_BACKEND_DESCRIPTOR,
                                                           1,
                                                           &backend_seq_len_KV));
        }

        if (has_left_bound()) {
            auto left_bound = attributes.inputs.find(DiagonalBandMask_attributes::input_names::LeftBound)->second;
            auto backend_left_bound = tensors[left_bound->get_uid()]->get_desc()->get_backend_descriptor();
            _CUDNN_CHECK_CUDNN_ERROR(detail::set_attribute(diagonal_band_mask_operation->get_backend_descriptor(),
                                                           CUDNN_ATTR_OPERATION_DIAGONAL_BAND_MASK_LEFT_BOUND_DESC,
                                                           CUDNN_TYPE_BACKEND_DESCRIPTOR,
                                                           1,
                                                           &backend_left_bound));
        }

        if (has_shift_right_bound()) {
            auto shift_right_bound =
                attributes.inputs.find(DiagonalBandMask_attributes::input_names::ShiftRightBound)->second;
            auto backend_shift_right_bound =
                tensors[shift_right_bound->get_uid()]->get_desc()->get_backend_descriptor();
            _CUDNN_CHECK_CUDNN_ERROR(
                detail::set_attribute(diagonal_band_mask_operation->get_backend_descriptor(),
                                      CUDNN_ATTR_OPERATION_DIAGONAL_BAND_MASK_SHIFT_RIGHT_BOUND_DESC,
                                      CUDNN_TYPE_BACKEND_DESCRIPTOR,
                                      1,
                                      &backend_shift_right_bound));
        }

        auto Y         = attributes.outputs.find(DiagonalBandMask_attributes::output_names::Y)->second;
        auto backend_y = tensors[Y->get_uid()]->get_desc()->get_backend_descriptor();
        _CUDNN_CHECK_CUDNN_ERROR(detail::set_attribute(diagonal_band_mask_operation->get_backend_descriptor(),
                                                       CUDNN_ATTR_OPERATION_DIAGONAL_BAND_MASK_YDESC,
                                                       CUDNN_TYPE_BACKEND_DESCRIPTOR,
                                                       1,
                                                       &backend_y));

        cudnnPointwiseMode_t cudnn_pointwise_mode;
        _CUDNN_CHECK_CUDNN_ERROR(detail::convert_to_cudnn_type(attributes.comparison_mode, cudnn_pointwise_mode));
        _CUDNN_CHECK_CUDNN_ERROR(detail::set_attribute(diagonal_band_mask_operation->get_backend_descriptor(),
                                                       CUDNN_ATTR_OPERATION_DIAGONAL_BAND_MASK_COMPARISON_MODE,
                                                       CUDNN_TYPE_POINTWISE_MODE,
                                                       1,
                                                       &cudnn_pointwise_mode));

        _CUDNN_CHECK_CUDNN_ERROR(detail::finalize(diagonal_band_mask_operation->get_backend_descriptor()));

        raw_operations.push_back(diagonal_band_mask_operation);

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

// Factory function to create a diagonal band mask node.
// This function will create the appropriate node based on the cuDNN version.
inline std::shared_ptr<Tensor_attributes>
INode::diagonal_band_mask(std::shared_ptr<Tensor_attributes> x,
                          std::shared_ptr<Tensor_attributes> b,
                          std::shared_ptr<Tensor_attributes> seq_len_q,
                          std::shared_ptr<Tensor_attributes> seq_len_kv,
                          std::shared_ptr<Tensor_attributes> left_bound,
                          std::shared_ptr<Tensor_attributes> shift_right_bound,
                          DiagonalBandMask_attributes attributes) {
    attributes.inputs[DiagonalBandMask_attributes::input_names::X] = x;
    attributes.inputs[DiagonalBandMask_attributes::input_names::B] = b;
    if (seq_len_q) {
        attributes.inputs[DiagonalBandMask_attributes::input_names::SEQ_LEN_Q] = seq_len_q;
    }
    if (seq_len_kv) {
        attributes.inputs[DiagonalBandMask_attributes::input_names::SEQ_LEN_KV] = seq_len_kv;
    }
    if (left_bound) {
        attributes.inputs[DiagonalBandMask_attributes::input_names::LeftBound] = left_bound;
    }
    if (shift_right_bound) {
        attributes.inputs[DiagonalBandMask_attributes::input_names::ShiftRightBound] = shift_right_bound;
    }
    auto y = attributes.outputs[DiagonalBandMask_attributes::output_names::Y] = output_tensor(attributes.name + "::Y");

    // Newer versions of cuDNN can represent the diagonal band mask as a single operation.
    if (std::min(detail::get_compiled_version(), detail::get_backend_version()) >= 92100) {
        sub_nodes.emplace_back(std::make_unique<UnifiedDiagonalBandMaskNode>(std::move(attributes), context));
    } else {
        sub_nodes.emplace_back(std::make_unique<CompositeDiagonalBandMaskNode>(std::move(attributes), context));
    }
    return y;
}

}  // namespace cudnn_frontend::graph