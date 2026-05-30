#pragma once

#include <vector>

#include "cudnn.h"

#include "backend_descriptor.h"

namespace cudnn_frontend::detail {
/**
 * @brief Creates a CUDNN backend variant pack descriptor.
 *
 * This function creates a `backend_descriptor` object representing a CUDNN backend variant pack
 * descriptor. The variant pack descriptor is configured with the provided device pointers, unique
 * IDs, and a workspace pointer.
 *
 * @param[out] variant_pack The created `backend_descriptor` object representing the variant pack.
 * @param device_ptrs A pointer to an array of device pointers to be associated with the variant pack.
 * @param uids A vector of unique IDs to be associated with the variant pack.
 * @param workspace_ptr A pointer to the workspace memory to be associated with the variant pack.
 * @return `error_t` A tuple containing the error code and an optional error message.
 *         The error code is `error_code_t::OK` on success, or an appropriate error code on failure.
 */
inline error_t
create_variant_pack(backend_descriptor& variant_pack,
                    void* const* device_ptrs,
                    std::vector<int64_t> const& uids,
                    void* workspace_ptr) {
    _CUDNN_CHECK_CUDNN_ERROR(detail::set_attribute(
        variant_pack.get_ptr(), CUDNN_ATTR_VARIANT_PACK_WORKSPACE, CUDNN_TYPE_VOID_PTR, 1, &workspace_ptr));

    _CUDNN_CHECK_CUDNN_ERROR(detail::set_attribute(
        variant_pack.get_ptr(), CUDNN_ATTR_VARIANT_PACK_DATA_POINTERS, CUDNN_TYPE_VOID_PTR, uids.size(), device_ptrs));

    _CUDNN_CHECK_CUDNN_ERROR(detail::set_attribute(
        variant_pack.get_ptr(), CUDNN_ATTR_VARIANT_PACK_UNIQUE_IDS, CUDNN_TYPE_INT64, uids.size(), uids.data()));

    _CUDNN_CHECK_CUDNN_ERROR(detail::finalize(variant_pack.get_ptr()));

    return {error_code_t::OK, ""};
}

inline error_t
create_variant_pack(backend_descriptor& variant_pack,
                    std::vector<void*>& device_ptrs,
                    std::vector<int64_t> const& uids,
                    void* workspace_ptr) {
    RETURN_CUDNN_FRONTEND_ERROR_IF(device_ptrs.size() != uids.size(),
                                   error_code_t::INVALID_VARIANT_PACK,
                                   "device_ptrs and uids must have the same length.");
    return create_variant_pack(variant_pack, device_ptrs.data(), uids, workspace_ptr);
}

inline error_t
create_variant_pack(backend_descriptor& variant_pack,
                    void* const* device_ptrs,
                    std::vector<int64_t> const& uids,
                    void* workspace_ptr,
                    std::vector<int64_t> const& override_uids,
                    std::vector<std::vector<int64_t>> const& override_shapes,
                    std::vector<std::vector<int64_t>> const& override_strides) {
    auto cudnn_ver_error = error_t{error_code_t::GRAPH_NOT_SUPPORTED, "Dynamic shapes requires cuDNN v9.18.0"};

    NV_CUDNN_FE_DYNAMIC_CHECK_CUDNN_BACKEND_VERSION(91800, cudnn_ver_error);

    CUDNN_FRONTEND_UNUSED(override_uids);
    CUDNN_FRONTEND_UNUSED(override_shapes);
    CUDNN_FRONTEND_UNUSED(override_strides);

    _CUDNN_CHECK_CUDNN_ERROR(detail::set_attribute(
        variant_pack.get_ptr(), CUDNN_ATTR_VARIANT_PACK_WORKSPACE, CUDNN_TYPE_VOID_PTR, 1, &workspace_ptr));

    _CUDNN_CHECK_CUDNN_ERROR(detail::set_attribute(
        variant_pack.get_ptr(), CUDNN_ATTR_VARIANT_PACK_DATA_POINTERS, CUDNN_TYPE_VOID_PTR, uids.size(), device_ptrs));

    _CUDNN_CHECK_CUDNN_ERROR(detail::set_attribute(
        variant_pack.get_ptr(), CUDNN_ATTR_VARIANT_PACK_UNIQUE_IDS, CUDNN_TYPE_INT64, uids.size(), uids.data()));

#if (CUDNN_VERSION >= 91800)
    _CUDNN_CHECK_CUDNN_ERROR(detail::set_attribute(variant_pack.get_ptr(),
                                                   CUDNN_ATTR_VARIANT_PACK_OVERRIDE_UNIQUE_IDS,
                                                   CUDNN_TYPE_INT64,
                                                   override_uids.size(),
                                                   override_uids.data()));

    _CUDNN_CHECK_CUDNN_ERROR(detail::set_attribute(variant_pack.get_ptr(),
                                                   CUDNN_ATTR_VARIANT_PACK_OVERRIDE_SHAPES,
                                                   CUDNN_TYPE_VOID_PTR,
                                                   1,
                                                   (void*)&override_shapes));

    _CUDNN_CHECK_CUDNN_ERROR(detail::set_attribute(variant_pack.get_ptr(),
                                                   CUDNN_ATTR_VARIANT_PACK_OVERRIDE_STRIDES,
                                                   CUDNN_TYPE_VOID_PTR,
                                                   1,
                                                   (void*)&override_strides));
#endif

    _CUDNN_CHECK_CUDNN_ERROR(detail::finalize(variant_pack.get_ptr()));

    return {error_code_t::OK, ""};
}

inline error_t
create_variant_pack(backend_descriptor& variant_pack,
                    std::vector<void*>& device_ptrs,
                    std::vector<int64_t> const& uids,
                    void* workspace_ptr,
                    std::vector<int64_t> const& override_uids,
                    std::vector<std::vector<int64_t>> const& override_shapes,
                    std::vector<std::vector<int64_t>> const& override_strides) {
    RETURN_CUDNN_FRONTEND_ERROR_IF(device_ptrs.size() != uids.size(),
                                   error_code_t::INVALID_VARIANT_PACK,
                                   "device_ptrs and uids must have the same length.");
    return create_variant_pack(
        variant_pack, device_ptrs.data(), uids, workspace_ptr, override_uids, override_shapes, override_strides);
}

}  // namespace cudnn_frontend::detail
