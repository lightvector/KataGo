#pragma once

#include "../cudnn_frontend_shim.h"
#include "../cudnn_frontend_utils.h"

namespace cudnn_frontend::detail {

class Context {
    DataType_t compute_data_type      = DataType_t::NOT_SET;
    DataType_t intermediate_data_type = DataType_t::NOT_SET;
    DataType_t io_data_type           = DataType_t::NOT_SET;
    int32_t target_sm_count           = -1;
    mutable int32_t target_sm_version = -1;
    bool is_dynamic_shape_enabled     = false;
    bool is_override_shape_enabled    = false;

    std::string name = "";

   public:
    Context&
    set_intermediate_data_type(DataType_t const type) {
        intermediate_data_type = type;
        return *this;
    }

    Context&
    set_io_data_type(DataType_t const type) {
        io_data_type = type;
        return *this;
    }

    Context&
    set_compute_data_type(DataType_t const type) {
        compute_data_type = type;
        return *this;
    }

    DataType_t
    get_io_data_type() const {
        return io_data_type;
    }

    DataType_t
    get_intermediate_data_type() const {
        return intermediate_data_type;
    }

    DataType_t
    get_compute_data_type() const {
        return compute_data_type;
    }

    Context&
    set_name(std::string const& name_) {
        name = name_;
        return *this;
    }

    std::string
    get_name() const {
        return name;
    }

    Context&
    set_target_sm_count(int32_t count) {
        target_sm_count = count;
        return *this;
    }

    Context&
    set_sm_version(int32_t version) {
        target_sm_version = version;
        return *this;
    }

    Context&
    set_dynamic_shape_enabled(bool is_enabled) {
        is_dynamic_shape_enabled = is_enabled;
        return *this;
    }

    Context&
    set_override_shape_enabled(bool is_enabled) {
        is_override_shape_enabled = is_enabled;
        return *this;
    }

    bool
    get_dynamic_shape_enabled() const {
        return is_dynamic_shape_enabled;
    }

    bool
    get_override_shape_enabled() const {
        return is_override_shape_enabled;
    }

    int32_t
    get_target_sm_count() const {
        return target_sm_count;
    }

    int32_t
    get_sm_version() const {
        return target_sm_version;
    }

    error_t
    populate_sm_version_from_device() const {
        if (target_sm_version > 0) {
            // Already set by user or previous call
            return {error_code_t::OK, ""};
        }
        cudaDeviceProp prop;
        int device;
        _CUDNN_CHECK_CUDA_ERROR(cuda_get_device(&device));
        _CUDNN_CHECK_CUDA_ERROR(cuda_get_device_properties(&prop, device));
        target_sm_version = prop.major * 10 + prop.minor;
        CUDNN_FE_LOG_LABEL_ENDL("INFO: Populated SM version from device: " << device << " " << target_sm_version);
        return {error_code_t::OK, ""};
    }

    Context&
    fill_missing_properties(Context const& global_context) {
        if (get_compute_data_type() == DataType_t::NOT_SET) {
            set_compute_data_type(global_context.get_compute_data_type());
        }
        if (get_intermediate_data_type() == DataType_t::NOT_SET) {
            set_intermediate_data_type(global_context.get_intermediate_data_type());
        }
        if (get_io_data_type() == DataType_t::NOT_SET) {
            set_io_data_type(global_context.get_io_data_type());
        }
        return *this;
    }
};

}  // namespace cudnn_frontend::detail