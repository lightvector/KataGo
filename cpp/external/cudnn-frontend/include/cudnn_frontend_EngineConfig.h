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
#include <sstream>
#include <utility>

#include "cudnn_frontend_Engine.h"
#include "cudnn_frontend_utils.h"

namespace cudnn_frontend {
///
/// EngineConfig_v8 Class
/// This class tells the Configuration of the Engine_v8 in terms of the knob
/// choices
/// Properties:
///    - num knobs
///    - Choice
///    - Engine_v8
///
/// Use EngineConfigBuilder_v8 to build this class.
/// Describe returns a string describing the tensor class
///
class EngineConfig_v8 : public BackendDescriptor {
   public:
    friend class EngineConfigBuilder_v8;

    std::string
    describe() const override {
        std::stringstream ss;
        ss << "CUDNN_BACKEND_ENGINECFG_DESCRIPTOR :";
        ss << " Number of knobs: " << numKnobs;
        return ss.str();
    }

    EngineConfig_v8 &
    operator=(EngineConfig_v8 &&from) = default;

    EngineConfig_v8(EngineConfig_v8 &&from) = default;

    ~EngineConfig_v8() = default;

    std::string const &
    getTag() const {
        return opGraphTag;
    }

   private:
    EngineConfig_v8() : BackendDescriptor() {
        cudnnStatus_t status;
        for (size_t i = 0; i < bChoices.size(); i++) {
            bChoices[i] = make_shared_backend_pointer(CUDNN_BACKEND_KNOB_CHOICE_DESCRIPTOR);
            if (bChoices[i]->is_good() == false) {
                status = bChoices[i]->get_status();
                set_error_and_throw_exception(
                    this,
                    status,
                    "CUDNN_BACKEND_ENGINECFG_DESCRIPTOR: CUDNN_BACKEND_KNOB_CHOICE_DESCRIPTOR cudnnCreate Failed");
                break;
            }
        }
    }
    EngineConfig_v8(EngineConfig_v8 const &) = delete;
    EngineConfig_v8 &
    operator=(EngineConfig_v8 const &) = delete;

    ManagedOpaqueDescriptor engine = nullptr;
    int64_t numKnobs               = 0;
    std::string opGraphTag;
    bool set_knobs_attr                                                  = false;
    std::array<ManagedOpaqueDescriptor, CUDNN_KNOB_TYPE_COUNTS> bChoices = {};  //!< Opaque pointer to the backend knobs
};

///
/// EngineConfigBuilder_v8 Class
/// Helper class used to build EngineConfig_v8 class
class EngineConfigBuilder_v8 {
   public:
    /** @defgroup EngineConfigBuilder_v8
     *  Set individual property of EngineConfig_v8 class
     *  @{
     */
    //! Set engine for the EngineConfig_v8
    auto
    setEngine(Engine_v8 const &engine_) -> EngineConfigBuilder_v8 & {
        m_engine_config.engine     = engine_.get_desc();
        m_engine_config.opGraphTag = engine_.getTag();
        auto &knobs                = engine_.getFinalizedKnobs();
        m_engine_config.numKnobs   = knobs.size();

        m_engine_config.set_knobs_attr = engine_.knobs_set();

        for (std::uint32_t i = 0; i < knobs.size(); i++) {
            cudnnStatus_t status;
            cudnnBackendKnobType_t type = knobs[i].getKnobType();
            int64_t value               = knobs[i].getChoice();
            status                      = detail::set_attribute(m_engine_config.bChoices[i]->get_backend_descriptor(),
                                           CUDNN_ATTR_KNOB_CHOICE_KNOB_TYPE,
                                           CUDNN_TYPE_KNOB_TYPE,
                                           1,
                                           &type);
            if (status != CUDNN_STATUS_SUCCESS) {
                set_error_and_throw_exception(&m_engine_config,
                                              status,
                                              "CUDNN_BACKEND_ENGINECFG_DESCRIPTOR: "
                                              "CUDNN_BACKEND_KNOB_CHOICE_DESCRIPTOR SetAttribute "
                                              "CUDNN_ATTR_KNOB_CHOICE_KNOB_TYPE Failed");
            }
            status = detail::set_attribute(m_engine_config.bChoices[i]->get_backend_descriptor(),
                                           CUDNN_ATTR_KNOB_CHOICE_KNOB_VALUE,
                                           CUDNN_TYPE_INT64,
                                           1,
                                           &value);
            if (status != CUDNN_STATUS_SUCCESS) {
                set_error_and_throw_exception(&m_engine_config,
                                              status,
                                              "CUDNN_BACKEND_ENGINECFG_DESCRIPTOR: "
                                              "CUDNN_BACKEND_KNOB_CHOICE_DESCRIPTOR SetAttribute "
                                              "CUDNN_ATTR_KNOB_CHOICE_KNOB_VALUE Failed");
            }
            status = detail::finalize(m_engine_config.bChoices[i]->get_backend_descriptor());
            if (status != CUDNN_STATUS_SUCCESS) {
                set_error_and_throw_exception(
                    &m_engine_config,
                    status,
                    "CUDNN_BACKEND_ENGINECFG_DESCRIPTOR: CUDNN_BACKEND_KNOB_CHOICE_DESCRIPTOR cudnnFinalize Failed");
            }
        }

        return *this;
    }
    /** @} */

    //! constructs the Engine_v8 Config by calling the cudnn API
    //! Throws the appropriate error message
    EngineConfig_v8 &&
    build() {
        if (m_engine_config.status != CUDNN_STATUS_SUCCESS) {
            set_error_and_throw_exception(&m_engine_config,
                                          m_engine_config.status,
                                          "CUDNN_BACKEND_ENGINECFG_DESCRIPTOR: is not created properly");
            return std::move(m_engine_config);
        }
        if (m_engine_config.engine == nullptr) {
            set_error_and_throw_exception(
                &m_engine_config,
                CUDNN_STATUS_BAD_PARAM,
                "CUDNN_BACKEND_ENGINECFG_DESCRIPTOR: Check and Set the CUDNN_ATTR_ENGINECFG_ENGINE.");
            return std::move(m_engine_config);
        }
        // Create a descriptor. Memory allocation happens here.
        auto status = m_engine_config.initialize_managed_backend_pointer(CUDNN_BACKEND_ENGINECFG_DESCRIPTOR);
        if (status != CUDNN_STATUS_SUCCESS) {
            set_error_and_throw_exception(
                &m_engine_config, status, "CUDNN_BACKEND_ENGINECFG_DESCRIPTOR: cudnnCreate Failed");
            return std::move(m_engine_config);
        }

        status = detail::set_attribute(m_engine_config.pointer->get_backend_descriptor(),
                                       CUDNN_ATTR_ENGINECFG_ENGINE,
                                       CUDNN_TYPE_BACKEND_DESCRIPTOR,
                                       1,
                                       &(m_engine_config.engine->get_backend_descriptor()));
        if (status != CUDNN_STATUS_SUCCESS) {
            set_error_and_throw_exception(
                &m_engine_config,
                status,
                "CUDNN_BACKEND_ENGINECFG_DESCRIPTOR: SetAttribute CUDNN_ATTR_ENGINECFG_ENGINE Failed");
            return std::move(m_engine_config);
        }

        if (m_engine_config.set_knobs_attr && m_engine_config.numKnobs > 0) {
            std::array<cudnnBackendDescriptor_t, CUDNN_KNOB_TYPE_COUNTS> bChoices_;
            for (auto i = 0; i < m_engine_config.numKnobs; i++) {
                bChoices_[i] = m_engine_config.bChoices[i]->get_backend_descriptor();
            }
            status = detail::set_attribute(m_engine_config.pointer->get_backend_descriptor(),
                                           CUDNN_ATTR_ENGINECFG_KNOB_CHOICES,
                                           CUDNN_TYPE_BACKEND_DESCRIPTOR,
                                           m_engine_config.numKnobs,
                                           bChoices_.data());
            if (status != CUDNN_STATUS_SUCCESS) {
                set_error_and_throw_exception(
                    &m_engine_config,
                    status,
                    "CUDNN_BACKEND_ENGINECFG_DESCRIPTOR: SetAttribute CUDNN_ATTR_ENGINECFG_KNOB_CHOICES Failed");
                return std::move(m_engine_config);
            }
        }

        // Finalizing the descriptor
        status = detail::finalize(m_engine_config.pointer->get_backend_descriptor());
        if (status != CUDNN_STATUS_SUCCESS) {
            set_error_and_throw_exception(
                &m_engine_config, status, "CUDNN_BACKEND_ENGINECFG_DESCRIPTOR: cudnnFinalize Failed");
            return std::move(m_engine_config);
        }
        CUDNN_FE_LOG_LABEL_ENDL(m_engine_config);
        return std::move(m_engine_config);
    }

    explicit EngineConfigBuilder_v8()                      = default;
    ~EngineConfigBuilder_v8()                              = default;
    EngineConfigBuilder_v8(EngineConfigBuilder_v8 &&)      = delete;
    EngineConfigBuilder_v8(EngineConfigBuilder_v8 const &) = delete;
    EngineConfigBuilder_v8 &
    operator=(EngineConfigBuilder_v8 const &) = delete;

   private:
    EngineConfig_v8 m_engine_config;
};

///
/// EngineConfigList class
/// This is a RAII type class that holds naked
/// EngineConfig backendDescriptor.
/// The purpose of this class is to provide an
/// easy interface to store the EngineConfigs generated
/// from various source and apply a filter.

using EngineConfigList = std::vector<ManagedOpaqueDescriptor>;
}  // namespace cudnn_frontend
