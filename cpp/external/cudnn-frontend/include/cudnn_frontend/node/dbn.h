#pragma once

#include "../graph_helpers.h"
#include "../node_interface.h"

namespace cudnn_frontend {

namespace graph {

class DBNNode : public NodeCRTP<DBNNode> {
   public:
    Batchnorm_backward_attributes attributes;

    DBNNode(Batchnorm_backward_attributes&& attributes_, detail::Context const& context)
        : NodeCRTP(context), attributes(std::move(attributes_)) {}

    Type
    getType() override final {
        return Type::DBN;
    }

    error_t
    infer_properties_node() override final {
        CUDNN_FE_LOG_LABEL_ENDL("INFO: Inferencing properties for DBN node " << attributes.name);

        attributes.fill_from_context(context);

        // TODO: Only inferencing from X works today.
        auto X                  = attributes.inputs[Batchnorm_backward_attributes::input_names::X];
        auto const x_tensor_dim = X->get_dim();

        auto DX            = attributes.outputs[Batchnorm_backward_attributes::output_names::DX];
        auto dx_tensor_dim = DX->get_dim();
        // Only infer dims and strides if user did not set them
        if (dx_tensor_dim.empty()) {
            dx_tensor_dim.resize(x_tensor_dim.size());
            DX->set_dim(x_tensor_dim);
        }
        if (DX->get_stride().empty()) {
            auto const& DX_dim = DX->get_dim();
            // Default to NHWC
            auto const& stride_order = detail::generate_NHWC_stride_order(DX_dim.size());
            DX->set_stride(detail::generate_stride(DX_dim, stride_order));
        }

        // Set channel length tensors
        auto infer_per_channel_tensors = [&x_tensor_dim](std::shared_ptr<Tensor_attributes>& T) {
            auto tensor_dim = T->get_dim();
            // Only infer dims and strides if user did not set them
            if (tensor_dim.empty()) {
                tensor_dim.resize(x_tensor_dim.size(), 1);
                tensor_dim[1] = x_tensor_dim[1];
                T->set_dim(tensor_dim);
            }
            if (T->get_stride().empty()) {
                auto const& T_dim = T->get_dim();
                // Default to NHWC
                auto const& stride_order = detail::generate_NHWC_stride_order(T_dim.size());
                T->set_stride(detail::generate_stride(T_dim, stride_order));
            }
        };
        infer_per_channel_tensors(attributes.outputs[Batchnorm_backward_attributes::output_names::DSCALE]);
        infer_per_channel_tensors(attributes.outputs[Batchnorm_backward_attributes::output_names::DBIAS]);
        return {error_code_t::OK, ""};
    }

    error_t
    create_cudnn_operations(
        std::unordered_set<Tensor_attributes::uid_t>& uids_involved_in_operations,
        std::vector<std::shared_ptr<cudnn_frontend::Operation>>& operations,
        managed_backend_descriptor_t& raw_operations,
        std::unordered_map<int64_t, std::shared_ptr<cudnn_frontend::Tensor>>& tensors) const override final {
        CUDNN_FRONTEND_UNUSED(raw_operations);
        CUDNN_FE_LOG_LABEL("INFO: " << "Building DBNNode operations " << attributes.name << " ");

        // Create operation by directly calling cuDNN backend API
        Operation_v8 dbn_operation;

        _CUDNN_CHECK_CUDNN_ERROR(
            dbn_operation.initialize_managed_backend_pointer(CUDNN_BACKEND_OPERATION_NORM_BACKWARD_DESCRIPTOR));

        // Set norm mode to BATCH_NORM
        cudnnBackendNormMode_t cudnn_norm_mode;

        _CUDNN_CHECK_CUDNN_ERROR(detail::convert_to_cudnn_type(NormMode_t::BATCH_NORM, cudnn_norm_mode));

        _CUDNN_CHECK_CUDNN_ERROR(detail::set_attribute(dbn_operation.get_raw_desc(),
                                                       CUDNN_ATTR_OPERATION_NORM_BWD_MODE,
                                                       CUDNN_TYPE_NORM_MODE,
                                                       1,
                                                       &cudnn_norm_mode));

        // Set input tensor X
        CUDNN_FE_VALIDATE_AND_ASSIGN_INPUT_TENSOR(X, Batchnorm_backward_attributes::input_names::X);
        auto x_desc = tensors.at(X->second->get_uid())->get_raw_desc();

        _CUDNN_CHECK_CUDNN_ERROR(detail::set_attribute(dbn_operation.get_raw_desc(),
                                                       CUDNN_ATTR_OPERATION_NORM_BWD_XDESC,
                                                       CUDNN_TYPE_BACKEND_DESCRIPTOR,
                                                       1,
                                                       &x_desc));

        // Set DY tensor
        CUDNN_FE_VALIDATE_AND_ASSIGN_INPUT_TENSOR(DY, Batchnorm_backward_attributes::input_names::DY);
        auto dy_desc = tensors.at(DY->second->get_uid())->get_raw_desc();

        _CUDNN_CHECK_CUDNN_ERROR(detail::set_attribute(dbn_operation.get_raw_desc(),
                                                       CUDNN_ATTR_OPERATION_NORM_BWD_DYDESC,
                                                       CUDNN_TYPE_BACKEND_DESCRIPTOR,
                                                       1,
                                                       &dy_desc));

        // Set scale tensor
        CUDNN_FE_VALIDATE_AND_ASSIGN_INPUT_TENSOR(SCALE, Batchnorm_backward_attributes::input_names::SCALE);
        auto scale_desc = tensors.at(SCALE->second->get_uid())->get_raw_desc();

        _CUDNN_CHECK_CUDNN_ERROR(detail::set_attribute(dbn_operation.get_raw_desc(),
                                                       CUDNN_ATTR_OPERATION_NORM_BWD_SCALE_DESC,
                                                       CUDNN_TYPE_BACKEND_DESCRIPTOR,
                                                       1,
                                                       &scale_desc));

        // Set mean and inv_variance tensors
        CUDNN_FE_VALIDATE_AND_ASSIGN_INPUT_TENSOR(MEAN, Batchnorm_backward_attributes::input_names::MEAN);
        auto mean_desc = tensors.at(MEAN->second->get_uid())->get_raw_desc();

        _CUDNN_CHECK_CUDNN_ERROR(detail::set_attribute(dbn_operation.get_raw_desc(),
                                                       CUDNN_ATTR_OPERATION_NORM_BWD_MEAN_DESC,
                                                       CUDNN_TYPE_BACKEND_DESCRIPTOR,
                                                       1,
                                                       &mean_desc));

        CUDNN_FE_VALIDATE_AND_ASSIGN_INPUT_TENSOR(INV_VARIANCE,
                                                  Batchnorm_backward_attributes::input_names::INV_VARIANCE);
        auto inv_var_desc = tensors.at(INV_VARIANCE->second->get_uid())->get_raw_desc();

        _CUDNN_CHECK_CUDNN_ERROR(detail::set_attribute(dbn_operation.get_raw_desc(),
                                                       CUDNN_ATTR_OPERATION_NORM_BWD_INV_VARIANCE_DESC,
                                                       CUDNN_TYPE_BACKEND_DESCRIPTOR,
                                                       1,
                                                       &inv_var_desc));

        // Set DSCALE and DBIAS output tensors
        CUDNN_FE_VALIDATE_AND_ASSIGN_OUTPUT_TENSOR(DSCALE, Batchnorm_backward_attributes::output_names::DSCALE);
        auto dscale_desc = tensors.at(DSCALE->second->get_uid())->get_raw_desc();

        _CUDNN_CHECK_CUDNN_ERROR(detail::set_attribute(dbn_operation.get_raw_desc(),
                                                       CUDNN_ATTR_OPERATION_NORM_BWD_DSCALE_DESC,
                                                       CUDNN_TYPE_BACKEND_DESCRIPTOR,
                                                       1,
                                                       &dscale_desc));

        CUDNN_FE_VALIDATE_AND_ASSIGN_OUTPUT_TENSOR(DBIAS, Batchnorm_backward_attributes::output_names::DBIAS);
        auto dbias_desc = tensors.at(DBIAS->second->get_uid())->get_raw_desc();

        _CUDNN_CHECK_CUDNN_ERROR(detail::set_attribute(dbn_operation.get_raw_desc(),
                                                       CUDNN_ATTR_OPERATION_NORM_BWD_DBIAS_DESC,
                                                       CUDNN_TYPE_BACKEND_DESCRIPTOR,
                                                       1,
                                                       &dbias_desc));

        // Set DX output tensor
        CUDNN_FE_VALIDATE_AND_ASSIGN_OUTPUT_TENSOR(DX, Batchnorm_backward_attributes::output_names::DX);
        auto dx_desc = tensors.at(DX->second->get_uid())->get_raw_desc();

        _CUDNN_CHECK_CUDNN_ERROR(detail::set_attribute(dbn_operation.get_raw_desc(),
                                                       CUDNN_ATTR_OPERATION_NORM_BWD_DXDESC,
                                                       CUDNN_TYPE_BACKEND_DESCRIPTOR,
                                                       1,
                                                       &dx_desc));

        // Set peer stat tensors if any
        if (!attributes.peer_stats.empty()) {
            std::vector<cudnnBackendDescriptor_t> peer_stat_descs;
            for (auto const& peer_stat : attributes.peer_stats) {
                peer_stat_descs.push_back(tensors.at(peer_stat->get_uid())->get_raw_desc());
            }

            _CUDNN_CHECK_CUDNN_ERROR(detail::set_attribute(dbn_operation.get_raw_desc(),
                                                           CUDNN_ATTR_OPERATION_NORM_BWD_PEER_STAT_DESCS,
                                                           CUDNN_TYPE_BACKEND_DESCRIPTOR,
                                                           peer_stat_descs.size(),
                                                           peer_stat_descs.data()));
        }

        _CUDNN_CHECK_CUDNN_ERROR(detail::finalize(dbn_operation.get_raw_desc()));

        operations.push_back(std::make_shared<Operation_v8>(std::move(dbn_operation)));

        auto const& non_virtual_uids = attributes.get_non_virtual_uids();
        uids_involved_in_operations.insert(non_virtual_uids.begin(), non_virtual_uids.end());
        return {error_code_t::OK, ""};
    }

#ifndef CUDNN_FRONTEND_SKIP_JSON_LIB
    virtual void
    serialize(json& j) const override final {
        j = attributes;
        j.update(R"( {"tag": "DBN"})"_json);
    }
#endif
};

}  // namespace graph

}  // namespace cudnn_frontend