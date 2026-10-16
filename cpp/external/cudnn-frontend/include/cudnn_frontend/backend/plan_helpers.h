#pragma once

#include <vector>

#include "cudnn.h"

#include "backend_descriptor.h"
#include "../knobs.h"

namespace cudnn_frontend::detail {
/**
 * @brief Creates a CUDNN backend variant pack descriptor.
 *
 * This function creates a `backend_descriptor` object representing a CUDNN backend variant pack
 * descriptor. The variant pack descriptor is configured with the provided device pointers, unique
 * IDs, and a workspace pointer.
 *
 * @param[out] variant_pack The created `backend_descriptor` object representing the variant pack.
 * @param device_ptrs A vector of device pointers to be associated with the variant pack.
 * @param uids A vector of unique IDs to be associated with the variant pack.
 * @param workspace_ptr A pointer to the workspace memory to be associated with the variant pack.
 * @return `error_t` A tuple containing the error code and an optional error message.
 *         The error code is `error_code_t::OK` on success, or an appropriate error code on failure.
 */
inline error_t
get_workspace_size(ManagedOpaqueDescriptor& engine_config, int64_t& workspace) {
#if CUDNN_VERSION >= 90200
    _CUDNN_CHECK_CUDNN_ERROR(detail::get_attribute(engine_config->get_backend_descriptor(),
                                                   CUDNN_ATTR_ENGINECFG_WORKSPACE_SIZE,
                                                   CUDNN_TYPE_INT64,
                                                   1,
                                                   nullptr,
                                                   &workspace));
    return {error_code_t::OK, ""};
#else
    (void)engine_config;
    (void)workspace;
    return {error_code_t::CUDNN_BACKEND_API_FAILED,
            "CUDNN_ATTR_ENGINECFG_WORKSPACE_SIZE is only available starting 9.2."};
#endif
}

inline error_t
get_shared_memory_size(ManagedOpaqueDescriptor& engine_config, int32_t& shared_memory_size) {
#if CUDNN_VERSION >= 90200
    _CUDNN_CHECK_CUDNN_ERROR(detail::get_attribute(engine_config->get_backend_descriptor(),
                                                   CUDNN_ATTR_ENGINECFG_SHARED_MEMORY_USED,
                                                   CUDNN_TYPE_INT32,
                                                   1,
                                                   nullptr,
                                                   &shared_memory_size));
    return {error_code_t::OK, ""};
#else
    (void)engine_config;
    (void)shared_memory_size;
    return {error_code_t::CUDNN_BACKEND_API_FAILED,
            "CUDNN_ATTR_ENGINECFG_SHARED_MEMORY_USED is only available starting 9.2."};
#endif
}

inline error_t
create_engine(backend_descriptor& engine,
              int64_t const engine_id,
              cudnnBackendDescriptor_t op_graph,
              std::shared_ptr<const DeviceProperties> device_properties = nullptr) {
    _CUDNN_CHECK_CUDNN_ERROR(detail::set_attribute(
        engine.get_ptr(), CUDNN_ATTR_ENGINE_OPERATION_GRAPH, CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &op_graph));

    // Validate before setting
    int64_t count;
    _CUDNN_CHECK_CUDNN_ERROR(detail::get_attribute(
        op_graph, CUDNN_ATTR_OPERATIONGRAPH_ENGINE_GLOBAL_COUNT, CUDNN_TYPE_INT64, 1, nullptr, &count));
    RETURN_CUDNN_FRONTEND_ERROR_IF(
        engine_id >= count || engine_id < 0, error_code_t::INVALID_VALUE, "Invalid engine id.");

    _CUDNN_CHECK_CUDNN_ERROR(
        detail::set_attribute(engine.get_ptr(), CUDNN_ATTR_ENGINE_GLOBAL_INDEX, CUDNN_TYPE_INT64, 1, &engine_id));

    if (device_properties != nullptr) {
#if (CUDNN_VERSION >= 90800)
        _CUDNN_CHECK_CUDNN_ERROR(detail::set_attribute(engine.get_ptr(),
                                                       CUDNN_ATTR_ENGINE_DEVICEPROP,
                                                       CUDNN_TYPE_BACKEND_DESCRIPTOR,
                                                       1,
                                                       &device_properties->get_ptr()));
#endif
    }

    _CUDNN_CHECK_CUDNN_ERROR(detail::finalize(engine.get_ptr()));

    return {error_code_t::OK, ""};
}

inline error_t
query_knobs(int64_t const engine_id, cudnnBackendDescriptor_t op_graph, std::vector<Knob>& knobs) {
    detail::backend_descriptor engine(CUDNN_BACKEND_ENGINE_DESCRIPTOR);
    RETURN_CUDNN_FRONTEND_ERROR_IF(engine.get_status() != CUDNN_STATUS_SUCCESS,
                                   error_code_t::CUDNN_BACKEND_API_FAILED,
                                   "Failed to create engine's backend descriptor.");
    CHECK_CUDNN_FRONTEND_ERROR(detail::create_engine(engine, engine_id, op_graph));

    // Initialize a backend descriptor for each knob type
    // The size of the array should be CUDNN_KNOB_TYPE_COUNTS, as currently we dont know how many knobs the engine will
    // support
    std::array<backend_descriptor, CUDNN_KNOB_TYPE_COUNTS> frontend_knobs;
    for (size_t i = 0; i < CUDNN_KNOB_TYPE_COUNTS; i++) {
        backend_descriptor frontend_knob(CUDNN_BACKEND_KNOB_INFO_DESCRIPTOR);
        RETURN_CUDNN_FRONTEND_ERROR_IF(frontend_knob.get_status() != CUDNN_STATUS_SUCCESS,
                                       error_code_t::CUDNN_BACKEND_API_FAILED,
                                       "Failed to create knob's backend descriptor.");
        frontend_knobs[i] = std::move(frontend_knob);
    }

    // Create an auxillary array to hold the raw knob descriptors
    std::array<cudnnBackendDescriptor_t, CUDNN_KNOB_TYPE_COUNTS> backend_knobs;
    for (size_t i = 0; i < CUDNN_KNOB_TYPE_COUNTS; i++) {
        backend_knobs[i] = frontend_knobs[i].get_ptr();
    }

    // This is the actual number of knobs that is supported by the engine
    int64_t knobs_size;
    _CUDNN_CHECK_CUDNN_ERROR(detail::get_attribute(engine.get_ptr(),
                                                   CUDNN_ATTR_ENGINE_KNOB_INFO,
                                                   CUDNN_TYPE_BACKEND_DESCRIPTOR,
                                                   CUDNN_KNOB_TYPE_COUNTS,
                                                   &knobs_size,
                                                   backend_knobs.data()));

    for (int64_t i = 0; i < knobs_size; i++) {
        cudnnBackendKnobType_t type;
        int64_t elemCount;
        _CUDNN_CHECK_CUDNN_ERROR(detail::get_attribute(
            frontend_knobs[i].get_ptr(), CUDNN_ATTR_KNOB_INFO_TYPE, CUDNN_TYPE_KNOB_TYPE, 1, &elemCount, &type));

        int64_t maxValue;
        _CUDNN_CHECK_CUDNN_ERROR(detail::get_attribute(frontend_knobs[i].get_ptr(),
                                                       CUDNN_ATTR_KNOB_INFO_MAXIMUM_VALUE,
                                                       CUDNN_TYPE_INT64,
                                                       1,
                                                       &elemCount,
                                                       &maxValue));

        int64_t minValue;
        _CUDNN_CHECK_CUDNN_ERROR(detail::get_attribute(frontend_knobs[i].get_ptr(),
                                                       CUDNN_ATTR_KNOB_INFO_MINIMUM_VALUE,
                                                       CUDNN_TYPE_INT64,
                                                       1,
                                                       &elemCount,
                                                       &minValue));

        int64_t stride;
        _CUDNN_CHECK_CUDNN_ERROR(detail::get_attribute(
            frontend_knobs[i].get_ptr(), CUDNN_ATTR_KNOB_INFO_STRIDE, CUDNN_TYPE_INT64, 1, &elemCount, &stride));

        auto frontend_knob_type = convert_from_backend_knob_type(type);
        knobs.emplace_back(frontend_knob_type, maxValue, minValue, stride);
    }

    return {error_code_t::OK, ""};
}

inline error_t
set_knob_choices(std::unordered_map<KnobType_t, int64_t> const& user_choices,
                 std::vector<detail::backend_descriptor>& knob_choices) {
    for (auto const& [type, choice] : user_choices) {
        backend_descriptor knob_choice(CUDNN_BACKEND_KNOB_CHOICE_DESCRIPTOR);
        RETURN_CUDNN_FRONTEND_ERROR_IF(knob_choice.get_status() != CUDNN_STATUS_SUCCESS,
                                       error_code_t::CUDNN_BACKEND_API_FAILED,
                                       "Failed to create knob_choice's backend descriptor.");

        cudnnBackendKnobType_t backend_type;
        _CUDNN_CHECK_CUDNN_ERROR(convert_to_backend_knob_type(type, backend_type));
        _CUDNN_CHECK_CUDNN_ERROR(detail::set_attribute(
            knob_choice.get_ptr(), CUDNN_ATTR_KNOB_CHOICE_KNOB_TYPE, CUDNN_TYPE_KNOB_TYPE, 1, &backend_type));
        _CUDNN_CHECK_CUDNN_ERROR(detail::set_attribute(
            knob_choice.get_ptr(), CUDNN_ATTR_KNOB_CHOICE_KNOB_VALUE, CUDNN_TYPE_INT64, 1, &choice));

        _CUDNN_CHECK_CUDNN_ERROR(detail::finalize(knob_choice.get_ptr()));

        knob_choices.push_back(std::move(knob_choice));
    }

    return {error_code_t::OK, ""};
}

inline error_t
create_engine_config(ManagedOpaqueDescriptor& engine_config,
                     backend_descriptor& engine,
                     std::vector<detail::backend_descriptor>& knob_choices) {
    _CUDNN_CHECK_CUDNN_ERROR(detail::set_attribute(engine_config->get_backend_descriptor(),
                                                   CUDNN_ATTR_ENGINECFG_ENGINE,
                                                   CUDNN_TYPE_BACKEND_DESCRIPTOR,
                                                   1,
                                                   &(engine.get_ptr())));

    std::vector<cudnnBackendDescriptor_t> backend_knob_choices(CUDNN_KNOB_TYPE_COUNTS);
    for (size_t i = 0; i < knob_choices.size(); i++) {
        backend_knob_choices[i] = knob_choices[i].get_ptr();
    }
    _CUDNN_CHECK_CUDNN_ERROR(detail::set_attribute(engine_config->get_backend_descriptor(),
                                                   CUDNN_ATTR_ENGINECFG_KNOB_CHOICES,
                                                   CUDNN_TYPE_BACKEND_DESCRIPTOR,
                                                   knob_choices.size(),
                                                   backend_knob_choices.data()));

    // Finalizing the descriptor
    _CUDNN_CHECK_CUDNN_ERROR(detail::finalize(engine_config->get_backend_descriptor()));

    return {error_code_t::OK, ""};
}

}  // namespace cudnn_frontend::detail
