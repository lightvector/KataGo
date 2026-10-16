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
#include <vector>

#include "cudnn_frontend_OperationGraph.h"
#include "cudnn_frontend_utils.h"

namespace cudnn_frontend {

///
/// Engine_v8 Class
/// This class tells the properties of the Engine_v8 on which performs the
/// operation requested
/// Properties:
///    - Index
///    - OperationGraph_v8
///
/// Use EngineBuilder_v8 to build this class.
/// Describe returns a string describing the tensor class
///
class Engine_v8 : public BackendDescriptor {
   private:
    Engine_v8()                  = default;
    Engine_v8(Engine_v8 const &) = delete;
    Engine_v8 &
    operator=(Engine_v8 const &) = delete;

    /// Internal class which controls the different knobs for a given engine
    /// Has min-max and stride as the options.
    /// User has the option to set the required value as a choice.
    class Knob {
       public:
        Knob(cudnnBackendKnobType_t type_, int64_t max, int64_t min, int64_t stride_)
            : knobType(type_), maxValue(max), minValue(min), stride(stride_) {}

        std::string
        describe() const {
            std::stringstream ss;
            ss << "Knob:" << knobType;
            ss << " Min: " << minValue;
            ss << " Max: " << maxValue;
            ss << " Stride: " << stride;
            return ss.str();
        }

        void
        setChoice(uint64_t val_) {
            choice = val_;
        }

        int64_t
        getChoice() const {
            return choice;
        }

        cudnnBackendKnobType_t
        getKnobType() const {
            return knobType;
        }

        int64_t
        getMinValue() const {
            return minValue;
        }

        int64_t
        getMaxValue() const {
            return minValue;
        }

        int64_t
        getStride() const {
            return stride;
        }

       private:
        cudnnBackendKnobType_t knobType = CUDNN_KNOB_TYPE_COUNTS;
        int64_t maxValue = 0, minValue = 0, stride = 0;  //!< min, max and stride of the knob value
        int64_t choice = -1;                             //!< Choice set by the user
    };

    ManagedOpaqueDescriptor opGraph = nullptr;
    int64_t idx                     = -1;  //!< Global Index of the engine for the given operationGraph.
    int64_t numKnobs                = 0;   //!< Count of the backend knobs in the engine
    std::array<ManagedOpaqueDescriptor, CUDNN_KNOB_TYPE_COUNTS> bKnobs = {};  //!< Opaque pointer to the backend knobs
    std::vector<Knob> knobs;
    std::string opGraphTag;

    //! Called from the constructor builds the internal knobs vector
    void
    buildKnobs() {
        cudnnStatus_t status;
        for (auto i = 0; i < numKnobs; i++) {
            auto bKnob = bKnobs[i]->get_backend_descriptor();
            cudnnBackendKnobType_t type;
            int64_t maxValue, minValue, stride, elemCount;
            status =
                detail::get_attribute(bKnob, CUDNN_ATTR_KNOB_INFO_TYPE, CUDNN_TYPE_KNOB_TYPE, 1, &elemCount, &type);
            if (status != CUDNN_STATUS_SUCCESS) {
                set_error_and_throw_exception(this,
                                              status,
                                              "CUDNN_BACKEND_ENGINE_DESCRIPTOR: CUDNN_BACKEND_KNOB_INFO_DESCRIPTOR "
                                              "GetAttribute CUDNN_ATTR_KNOB_INFO_TYPE failed");
            }
            status = detail::get_attribute(
                bKnob, CUDNN_ATTR_KNOB_INFO_MAXIMUM_VALUE, CUDNN_TYPE_INT64, 1, &elemCount, &maxValue);
            if (status != CUDNN_STATUS_SUCCESS) {
                set_error_and_throw_exception(this,
                                              status,
                                              "CUDNN_BACKEND_ENGINE_DESCRIPTOR: CUDNN_BACKEND_KNOB_INFO_DESCRIPTOR "
                                              "GetAttribute CUDNN_ATTR_KNOB_INFO_MAXIMUM_VALUE Failed");
            }
            status = detail::get_attribute(
                bKnob, CUDNN_ATTR_KNOB_INFO_MINIMUM_VALUE, CUDNN_TYPE_INT64, 1, &elemCount, &minValue);
            if (status != CUDNN_STATUS_SUCCESS) {
                set_error_and_throw_exception(this,
                                              status,
                                              "CUDNN_BACKEND_ENGINE_DESCRIPTOR: CUDNN_BACKEND_KNOB_INFO_DESCRIPTOR "
                                              "GetAttribute CUDNN_ATTR_KNOB_INFO_MINIMUM_VALUE Failed");
            }
            status =
                detail::get_attribute(bKnob, CUDNN_ATTR_KNOB_INFO_STRIDE, CUDNN_TYPE_INT64, 1, &elemCount, &stride);
            if (status != CUDNN_STATUS_SUCCESS) {
                set_error_and_throw_exception(this,
                                              status,
                                              "CUDNN_BACKEND_ENGINE_DESCRIPTOR: CUDNN_BACKEND_KNOB_INFO_DESCRIPTOR "
                                              "GetAttribute CUDNN_ATTR_KNOB_INFO_STRIDE Failed");
            }
            knobs.emplace_back(Knob(type, maxValue, minValue, stride));
        }
    }

   public:
    friend class EngineBuilder_v8;
    std::string
    describe() const override {
        std::stringstream ss;
        ss << "CUDNN_BACKEND_ENGINE_DESCRIPTOR :";
        ss << " ID: " << idx;
        ss << " Has " << numKnobs << " knobs";
        return ss.str();
    }
    Engine_v8(Engine_v8 &&from) = default;

    Engine_v8 &
    operator=(Engine_v8 &&) = default;
    ~Engine_v8()            = default;

    std::string const &
    getTag() const {
        return opGraphTag;
    }

    //! Returns a vector of knobs to the user for modification
    std::vector<Knob> &
    getSupportedKnobs() {
        return knobs;
    }

    //! Returns a final vector of knobs. Used in EngineConfigBuilder
    std::vector<Knob> const &
    getFinalizedKnobs() const {
        return knobs;
    }

    bool
    knobs_set() const {
        bool is_knob_set = false;
        for (auto i = 0; i < numKnobs; i++) {
            if (knobs[i].getChoice() != -1) {
                is_knob_set = true;
                break;
            }
        }
        return is_knob_set;
    }
};

///
/// EngineBuilder_v8 Class
/// Helper class used to build Engine_v8 class
class EngineBuilder_v8 {
   public:
    /** @defgroup EngineBuilder_v8
     *  Set individual property of Engine_v8 class
     *  @{
     */
    //! Set operationGraph for the engine
    auto
    setOperationGraph(OperationGraph_v8 const &opGraph_) -> EngineBuilder_v8 & {
        m_engine.opGraph    = opGraph_.get_desc();
        m_engine.opGraphTag = opGraph_.getTag();
        return *this;
    }

    //! Set operationGraph for the engine
    auto
    setOperationGraph(ManagedOpaqueDescriptor desc_) -> EngineBuilder_v8 & {
        m_engine.opGraph = desc_;
        return *this;
    }
    //! Set engine index for the engine
    auto
    setGlobalEngineIdx(int64_t idx_) -> EngineBuilder_v8 & {
        m_engine.idx = idx_;
        return *this;
    }
    /** @} */

    //! constructs the Engine_v8 by calling the cudnn API
    //! Throws the appropriate error message
    Engine_v8 &&
    build() {
        if (m_engine.idx < 0) {
            set_error_and_throw_exception(
                &m_engine,
                CUDNN_STATUS_BAD_PARAM,
                "CUDNN_BACKEND_ENGINE_DESCRIPTOR: Check and Set the CUDNN_ATTR_ENGINE_GLOBAL_INDEX to valid value");
            return std::move(m_engine);
        }
        if (m_engine.opGraph == nullptr) {
            set_error_and_throw_exception(
                &m_engine,
                CUDNN_STATUS_BAD_PARAM,
                "CUDNN_BACKEND_ENGINE_DESCRIPTOR: Check and Set CUDNN_ATTR_ENGINE_OPERATION_GRAPH to valid value");
            return std::move(m_engine);
        }

        // Create a descriptor. Memory allocation happens here.
        auto status = m_engine.initialize_managed_backend_pointer(CUDNN_BACKEND_ENGINE_DESCRIPTOR);
        if (status != CUDNN_STATUS_SUCCESS) {
            set_error_and_throw_exception(
                &m_engine, status, "CUDNN_BACKEND_ENGINE_DESCRIPTOR: cudnnCreate Descriptor Failed");
            return std::move(m_engine);
        }

        status = detail::set_attribute(m_engine.pointer->get_backend_descriptor(),
                                       CUDNN_ATTR_ENGINE_OPERATION_GRAPH,
                                       CUDNN_TYPE_BACKEND_DESCRIPTOR,
                                       1,
                                       &(m_engine.opGraph->get_backend_descriptor()));
        if (status != CUDNN_STATUS_SUCCESS) {
            set_error_and_throw_exception(
                &m_engine,
                status,
                "CUDNN_BACKEND_ENGINE_DESCRIPTOR: SetAttribute CUDNN_ATTR_ENGINE_OPERATION_GRAPH Failed");
            return std::move(m_engine);
        }

        status = detail::set_attribute(m_engine.pointer->get_backend_descriptor(),
                                       CUDNN_ATTR_ENGINE_GLOBAL_INDEX,
                                       CUDNN_TYPE_INT64,
                                       1,
                                       &m_engine.idx);
        if (status != CUDNN_STATUS_SUCCESS) {
            set_error_and_throw_exception(
                &m_engine,
                status,
                "CUDNN_BACKEND_ENGINE_DESCRIPTOR: SetAttribute CUDNN_ATTR_ENGINE_GLOBAL_INDEX Failed");
            return std::move(m_engine);
        }

        // Finalizing the descriptor
        status = detail::finalize(m_engine.pointer->get_backend_descriptor());
        if (status != CUDNN_STATUS_SUCCESS) {
            set_error_and_throw_exception(&m_engine, status, "CUDNN_BACKEND_ENGINE_DESCRIPTOR: cudnnFinalize Failed");
            return std::move(m_engine);
        }

        for (size_t i = 0; i < m_engine.bKnobs.size(); i++) {
            m_engine.bKnobs[i] = make_shared_backend_pointer(CUDNN_BACKEND_KNOB_INFO_DESCRIPTOR);
            if (m_engine.bKnobs[i]->is_good() == false) {
                status = m_engine.bKnobs[i]->get_status();
                set_error_and_throw_exception(
                    &m_engine,
                    status,
                    "CUDNN_BACKEND_ENGINE_DESCRIPTOR: CUDNN_BACKEND_KNOB_INFO_DESCRIPTOR cudnnCreate Failed");
            }
        }

        std::array<cudnnBackendDescriptor_t, CUDNN_KNOB_TYPE_COUNTS> bKnobs_ =
            {};  //!< Opaque pointer to the backend knobs
        for (std::uint32_t i = 0; i < m_engine.bKnobs.size(); i++) {
            bKnobs_[i] = m_engine.bKnobs[i]->get_backend_descriptor();
        }
        status = detail::get_attribute(m_engine.pointer->get_backend_descriptor(),
                                       CUDNN_ATTR_ENGINE_KNOB_INFO,
                                       CUDNN_TYPE_BACKEND_DESCRIPTOR,
                                       CUDNN_KNOB_TYPE_COUNTS,
                                       &m_engine.numKnobs,
                                       bKnobs_.data());
        if (status != CUDNN_STATUS_SUCCESS) {
            set_error_and_throw_exception(
                &m_engine,
                status,
                "CUDNN_BACKEND_ENGINE_DESCRIPTOR: GetAttribute CUDNN_ATTR_ENGINE_KNOB_INFO Query Failed");
        }
        m_engine.buildKnobs();
        CUDNN_FE_LOG_LABEL_ENDL(m_engine);
        return std::move(m_engine);
    }

    explicit EngineBuilder_v8()                = default;
    ~EngineBuilder_v8()                        = default;
    EngineBuilder_v8(EngineBuilder_v8 &&)      = delete;
    EngineBuilder_v8(EngineBuilder_v8 const &) = delete;
    EngineBuilder_v8 &
    operator=(EngineBuilder_v8 const &) = delete;

   private:
    Engine_v8 m_engine;
};
}  // namespace cudnn_frontend
