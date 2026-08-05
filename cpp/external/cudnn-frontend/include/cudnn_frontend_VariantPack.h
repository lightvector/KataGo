/*
 * Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */

#pragma once

#include <algorithm>
#include <array>
#include <functional>
#include <memory>
#include <set>
#include <sstream>
#include <utility>

#include "cudnn_frontend_utils.h"

namespace cudnn_frontend {

///
/// VariantPack_v8 Class
/// This class tells the Configuration of the Engine in terms of the knob
/// choices
/// Properties:
///    - num knobs
///    - Choice
///    - Engine
///
/// Use VariantPackBuilder_v8 to build this class.
/// Describe returns a string describing the tensor class
///
class VariantPack_v8 : public BackendDescriptor {
   public:
    friend class VariantPackBuilder_v8;
    std::string
    describe() const override {
        std::stringstream ss;
        ss << "CUDNN_BACKEND_VARIANT_PACK_DESCRIPTOR :" << " has " << num_ptrs << " data pointers";
        return ss.str();
    }

    VariantPack_v8(VariantPack_v8 &&from) = default;
    VariantPack_v8 &
    operator=(VariantPack_v8 &&from) = default;

    ~VariantPack_v8() = default;

   private:
    VariantPack_v8()                       = default;
    VariantPack_v8(VariantPack_v8 const &) = delete;
    VariantPack_v8 &
    operator=(VariantPack_v8 const &) = delete;

    void *workspace = nullptr;
    std::vector<void *> data_pointers;
    std::vector<int64_t> uid;
    int64_t num_ptrs = -1;
};

///
/// VariantPackBuilder_v8 Class
/// Helper class used to build VariantPack_v8 class
class VariantPackBuilder_v8 {
   public:
    /** @defgroup VariantPackBuilder_v8
     *  Set individual property of VariantPack_v8 class
     *  @{
     */
    //! Set dataPointers for the VariantPack_v8
    auto
    setDataPointers(int64_t num_ptr, void **ptrs) -> VariantPackBuilder_v8 & {
        m_variant_pack.data_pointers.reserve(static_cast<size_t>(num_ptr));
        std::copy(ptrs, ptrs + num_ptr, std::back_inserter(m_variant_pack.data_pointers));
        m_variant_pack.num_ptrs = num_ptr;
        return *this;
    }
    //! Set Uids for the VariantPack_v8
    auto
    setUids(int64_t num_uids, const int64_t *uid) -> VariantPackBuilder_v8 & {
        return setUids(num_uids, const_cast<int64_t *>(uid));
    }

    auto
    setUids(int64_t num_uids, int64_t *uid) -> VariantPackBuilder_v8 & {
        m_variant_pack.uid.reserve(static_cast<size_t>(num_uids));
        std::copy(uid, uid + num_uids, std::back_inserter(m_variant_pack.uid));
        return *this;
    }
    //! Initialize a set of pairs containing uid and data pointer.
    auto
    setDataPointers(std::set<std::pair<uint64_t, void *>> const &data_pointers) -> VariantPackBuilder_v8 & {
        m_variant_pack.num_ptrs = data_pointers.size();
        m_variant_pack.uid.reserve(static_cast<size_t>(m_variant_pack.num_ptrs));
        m_variant_pack.data_pointers.reserve(static_cast<size_t>(m_variant_pack.num_ptrs));
        for (auto &data_pointer : data_pointers) {
            m_variant_pack.uid.push_back(data_pointer.first);
            m_variant_pack.data_pointers.push_back(data_pointer.second);
        }
        return *this;
    }
    //! Set Workspace
    auto
    setWorkspacePointer(void *ws) -> VariantPackBuilder_v8 & {
        m_variant_pack.workspace = ws;
        return *this;
    }
    /** @} */

    //! constructs the Engine Config by calling the cudnn API
    //! Throws the appropriate error message
    VariantPack_v8 &&
    build() {
        // Create a descriptor. Memory allocation happens here.
        auto status = m_variant_pack.initialize_managed_backend_pointer(CUDNN_BACKEND_VARIANT_PACK_DESCRIPTOR);
        if (status != CUDNN_STATUS_SUCCESS) {
            set_error_and_throw_exception(
                &m_variant_pack, status, "CUDNN_BACKEND_VARIANT_PACK_DESCRIPTOR: cudnnCreate Failed");
            return std::move(m_variant_pack);
        }

        status = detail::set_attribute(m_variant_pack.pointer->get_backend_descriptor(),
                                       CUDNN_ATTR_VARIANT_PACK_DATA_POINTERS,
                                       CUDNN_TYPE_VOID_PTR,
                                       m_variant_pack.num_ptrs,
                                       m_variant_pack.data_pointers.data());
        if (status != CUDNN_STATUS_SUCCESS) {
            set_error_and_throw_exception(
                &m_variant_pack,
                status,
                "CUDNN_BACKEND_VARIANT_PACK_DESCRIPTOR: SetAttribute CUDNN_ATTR_VARIANT_PACK_DATA_POINTERS Failed");
            return std::move(m_variant_pack);
        }

        status = detail::set_attribute(m_variant_pack.pointer->get_backend_descriptor(),
                                       CUDNN_ATTR_VARIANT_PACK_UNIQUE_IDS,
                                       CUDNN_TYPE_INT64,
                                       m_variant_pack.num_ptrs,
                                       m_variant_pack.uid.data());
        if (status != CUDNN_STATUS_SUCCESS) {
            set_error_and_throw_exception(
                &m_variant_pack,
                status,
                "CUDNN_BACKEND_VARIANT_PACK_DESCRIPTOR: SetAttribute CUDNN_ATTR_VARIANT_PACK_UNIQUE_IDS Failed");
            return std::move(m_variant_pack);
        }

        status = detail::set_attribute(m_variant_pack.pointer->get_backend_descriptor(),
                                       CUDNN_ATTR_VARIANT_PACK_WORKSPACE,
                                       CUDNN_TYPE_VOID_PTR,
                                       1,
                                       &m_variant_pack.workspace);
        if (status != CUDNN_STATUS_SUCCESS) {
            set_error_and_throw_exception(
                &m_variant_pack,
                status,
                "CUDNN_BACKEND_VARIANT_PACK_DESCRIPTOR: SetAttribute CUDNN_ATTR_VARIANT_PACK_WORKSPACE Failed");
            return std::move(m_variant_pack);
        }

        // Finalizing the descriptor
        status = detail::finalize(m_variant_pack.pointer->get_backend_descriptor());
        if (status != CUDNN_STATUS_SUCCESS) {
            set_error_and_throw_exception(
                &m_variant_pack, status, "CUDNN_BACKEND_VARIANT_PACK_DESCRIPTOR: cudnnFinalize Failed");
            return std::move(m_variant_pack);
        }
        CUDNN_FE_LOG_LABEL_ENDL(m_variant_pack);
        return std::move(m_variant_pack);
    }

    explicit VariantPackBuilder_v8()                     = default;
    ~VariantPackBuilder_v8()                             = default;
    VariantPackBuilder_v8(VariantPackBuilder_v8 &&)      = delete;
    VariantPackBuilder_v8(VariantPackBuilder_v8 const &) = delete;
    VariantPackBuilder_v8 &
    operator=(VariantPackBuilder_v8 const &) = delete;

   private:
    VariantPack_v8 m_variant_pack;
};

using VariantPack        = VariantPack_v8;
using VariantPackBuilder = VariantPackBuilder_v8;

}  // namespace cudnn_frontend
