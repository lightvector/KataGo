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

#include <vector>
#include <mutex>

#include "cudnn_frontend_OperationGraph.h"
#include "cudnn_frontend_EngineConfig.h"
#include "cudnn_frontend_utils.h"
#include "cudnn_frontend_Filters.h"
#include "cudnn_frontend/backend/device_properties.h"

#if defined(_MSC_VER)
#pragma warning(push)
#pragma warning(disable : 4127)
#endif
namespace cudnn_frontend {
///
/// Engine Heuristic Class
/// This class helps determine the engine from the operation graph
/// based on the heuristics
/// Properties:
///    - heuristic mode
///    - operation graph
///
/// Use EngineHeuristicsBuilder_v8 to build this class.
/// Describe returns a string describing the EngineHeuristics_v8 class
///
class EngineHeuristics_v8 : public BackendDescriptor {
   public:
    friend class EngineHeuristicsBuilder_v8;
    std::string
    describe() const override {
        std::stringstream ss;
        ss << "CUDNN_BACKEND_ENGINEHEUR_DESCRIPTOR :";
        return ss.str();
    }

    EngineHeuristics_v8(EngineHeuristics_v8 &&from) = default;
    EngineHeuristics_v8 &
    operator=(EngineHeuristics_v8 &&from) = default;

    ~EngineHeuristics_v8() = default;

    /** @defgroup EngineHeuristicsQuery
     *  Query individual property of EngineHeuristics_v8 class
     *  @{
     */
    //! Query the total count of the engines for the Operation Set
    auto
    getEngineConfig(int64_t count = 1) -> std::vector<ManagedOpaqueDescriptor> & {
        cudnnStatus_t status;
        for (auto i = 0u; i < count; ++i) {
            ManagedOpaqueDescriptor engConfig = nullptr;
            engConfig                         = make_shared_backend_pointer(CUDNN_BACKEND_ENGINECFG_DESCRIPTOR);
            if (engConfig->is_good() == false) {
                set_error_and_throw_exception(
                    this,
                    engConfig->get_status(),
                    "CUDNN_BACKEND_ENGINEHEUR_DESCRIPTOR: CUDNN_BACKEND_ENGINECFG_DESCRIPTOR cudnnCreate Failed");
                return m_heuristic_results;
            };
            m_heuristic_results.emplace_back(engConfig);
        }
        std::vector<cudnnBackendDescriptor_t> heuristic_results_;
        for (std::uint32_t i = 0; i < m_heuristic_results.size(); i++) {
            heuristic_results_.emplace_back(m_heuristic_results[i]->get_backend_descriptor());
        }
        int64_t result = -1;
        status         = detail::get_attribute(pointer->get_backend_descriptor(),
                                       CUDNN_ATTR_ENGINEHEUR_RESULTS,
                                       CUDNN_TYPE_BACKEND_DESCRIPTOR,
                                       count,
                                       &result,
                                       heuristic_results_.data());

        if (status != CUDNN_STATUS_SUCCESS) {
            set_error_and_throw_exception(
                this, status, "CUDNN_BACKEND_ENGINEHEUR_DESCRIPTOR: GetAttribute CUDNN_ATTR_ENGINEHEUR_RESULTS Failed");
        };
        m_heuristic_results.resize(result);
        return m_heuristic_results;
    }

    //! Query the total count of the engine config for the Operation Set
    auto
    getEngineConfigCount(void) const -> int64_t {
        cudnnStatus_t status;
        int64_t count = -1;
        status        = detail::get_attribute(pointer->get_backend_descriptor(),
                                       CUDNN_ATTR_ENGINEHEUR_RESULTS,
                                       CUDNN_TYPE_BACKEND_DESCRIPTOR,
                                       0,
                                       &count,
                                       nullptr);
        if (status != CUDNN_STATUS_SUCCESS) {
            set_error_and_throw_exception(
                this,
                status,
                "CUDNN_BACKEND_ENGINEHEUR_DESCRIPTOR: GetAttribute CUDNN_ATTR_ENGINEHEUR_RESULTS Count Failed");
        };
        return count;
    }
    /** @} */

   private:
    EngineHeuristics_v8()                            = default;
    EngineHeuristics_v8(EngineHeuristics_v8 const &) = delete;
    EngineHeuristics_v8 &
    operator=(EngineHeuristics_v8 const &) = delete;

    cudnnBackendHeurMode_t mode                               = CUDNN_HEUR_MODE_INSTANT;
    ManagedOpaqueDescriptor opGraph                           = nullptr;
    std::shared_ptr<const DeviceProperties> device_properties = nullptr;
    std::vector<ManagedOpaqueDescriptor> m_heuristic_results;  //! storage of heuristic results
    std::string opGraphTag;
    int32_t target_sm_count = -1;

    static std::mutex &
    get_heur_b_mutex() {
        static std::mutex heur_b_mutex;
        return heur_b_mutex;
    }
};

///
/// EngineHeuristicsBuilder_v8 Class
/// Helper class used to build EngineHeuristics_v8 class
class EngineHeuristicsBuilder_v8 {
   public:
    /** @defgroup EngineHeuristicsBuilder_v8
     *  Set individual property of EngineHeuristics_v8 class
     *  @{
     */
    //! Set operationGraph for the engine (opGraph is not destroyed)
    auto
    setOperationGraph(OperationGraph_v8 &opGraph_) -> EngineHeuristicsBuilder_v8 & {
        m_heuristics.opGraph    = opGraph_.get_desc();
        m_heuristics.opGraphTag = opGraph_.getTag();
        return *this;
    }
    auto
    setOperationGraph(ManagedOpaqueDescriptor opGraph, std::string tag) -> EngineHeuristicsBuilder_v8 & {
        m_heuristics.opGraph    = opGraph;
        m_heuristics.opGraphTag = tag;
        return *this;
    }

    auto
    setDeviceProperties(std::shared_ptr<const DeviceProperties> device_properties) -> EngineHeuristicsBuilder_v8 & {
        m_heuristics.device_properties = device_properties;
        return *this;
    }

    auto
    setHeurMode(cudnnBackendHeurMode_t mode_) -> EngineHeuristicsBuilder_v8 & {
        m_heuristics.mode = mode_;
        return *this;
    }

    auto
    setSMCount(int32_t sm_count_) -> EngineHeuristicsBuilder_v8 & {
        m_heuristics.target_sm_count = sm_count_;
        return *this;
    }
    /** @} */

    //! constructs the EngineHeuristics_v8 by calling the cudnn API
    //! Throws the appropriate error message
    EngineHeuristics_v8 &&
    build() {
        if (m_heuristics.opGraph == nullptr) {
            set_error_and_throw_exception(&m_heuristics,
                                          CUDNN_STATUS_BAD_PARAM,
                                          "CUDNN_BACKEND_ENGINEHEUR_DESCRIPTOR: Check and Set the "
                                          "CUDNN_ATTR_ENGINEHEUR_OPERATION_GRAPH field for heuristic");
            return std::move(m_heuristics);
        };

        // Create a descriptor. Memory allocation happens here.
        auto status = m_heuristics.initialize_managed_backend_pointer(CUDNN_BACKEND_ENGINEHEUR_DESCRIPTOR);
        if (status != CUDNN_STATUS_SUCCESS) {
            set_error_and_throw_exception(
                &m_heuristics, status, "CUDNN_BACKEND_ENGINEHEUR_DESCRIPTOR: cudnnCreate Failed");
            return std::move(m_heuristics);
        };

        status = detail::set_attribute(m_heuristics.pointer->get_backend_descriptor(),
                                       CUDNN_ATTR_ENGINEHEUR_OPERATION_GRAPH,
                                       CUDNN_TYPE_BACKEND_DESCRIPTOR,
                                       1,
                                       &(m_heuristics.opGraph->get_backend_descriptor()));
        if (status != CUDNN_STATUS_SUCCESS) {
            set_error_and_throw_exception(
                &m_heuristics,
                status,
                "CUDNN_BACKEND_ENGINEHEUR_DESCRIPTOR: SetAttribute  CUDNN_ATTR_ENGINEHEUR_OPERATION_GRAPH Failed");
            return std::move(m_heuristics);
        };

        status = detail::set_attribute(m_heuristics.pointer->get_backend_descriptor(),
                                       CUDNN_ATTR_ENGINEHEUR_MODE,
                                       CUDNN_TYPE_HEUR_MODE,
                                       1,
                                       &m_heuristics.mode);
        if (status != CUDNN_STATUS_SUCCESS) {
            set_error_and_throw_exception(
                &m_heuristics,
                status,
                "CUDNN_BACKEND_ENGINEHEUR_DESCRIPTOR: SetAttribute CUDNN_ATTR_ENGINEHEUR_MODE Failed");
            return std::move(m_heuristics);
        };

        if (m_heuristics.device_properties != nullptr) {
#if (CUDNN_VERSION >= 90800)
            status = detail::set_attribute(m_heuristics.pointer->get_backend_descriptor(),
                                           CUDNN_ATTR_ENGINEHEUR_DEVICEPROP,
                                           CUDNN_TYPE_BACKEND_DESCRIPTOR,
                                           1,
                                           &m_heuristics.device_properties->get_ptr());
            if (status != CUDNN_STATUS_SUCCESS) {
                set_error_and_throw_exception(&m_heuristics,
                                              status,
                                              "CUDNN_BACKEND_ENGINEHEUR_DESCRIPTOR: SetAttribute "
                                              "CUDNN_ATTR_ENGINEHEUR_DEVICEPROP Failed");
                return std::move(m_heuristics);
            }
#endif
        }

#if (CUDNN_VERSION >= 8905)
        if (m_heuristics.target_sm_count >= 0) {
            NV_CUDNN_FE_DYNAMIC_CHECK_BACKEND_DESCRIPTOR(8905,
                                                         m_heuristics,
                                                         "CUDNN_BACKEND_ENGINEHEUR_DESCRIPTOR: SetAttribute "
                                                         "CUDNN_ATTR_ENGINEHEUR_SM_COUNT_TARGET requires cudnn "
                                                         "version 8.9.5");
            status = detail::set_attribute(m_heuristics.pointer->get_backend_descriptor(),
                                           CUDNN_ATTR_ENGINEHEUR_SM_COUNT_TARGET,
                                           CUDNN_TYPE_INT32,
                                           1,
                                           &m_heuristics.target_sm_count);
            if (status != CUDNN_STATUS_SUCCESS) {
                set_error_and_throw_exception(
                    &m_heuristics,
                    status,
                    "CUDNN_BACKEND_ENGINEHEUR_DESCRIPTOR: SetAttribute CUDNN_ATTR_ENGINEHEUR_SM_COUNT_TARGET Failed");
                return std::move(m_heuristics);
            };
        }
#endif

        if (m_heuristics.mode == CUDNN_HEUR_MODE_B) {
            EngineHeuristics_v8::get_heur_b_mutex().lock();
        }

        // Finalizing the descriptor
        status = detail::finalize(m_heuristics.pointer->get_backend_descriptor());

        if (m_heuristics.mode == CUDNN_HEUR_MODE_B) {
            EngineHeuristics_v8::get_heur_b_mutex().unlock();
        }

        if (status != CUDNN_STATUS_SUCCESS) {
            set_error_and_throw_exception(
                &m_heuristics, status, "CUDNN_BACKEND_ENGINEHEUR_DESCRIPTOR: cudnn Finalize failed");
            return std::move(m_heuristics);
        };

        CUDNN_FE_LOG_LABEL_ENDL(m_heuristics);
        return std::move(m_heuristics);
    }

    explicit EngineHeuristicsBuilder_v8()                          = default;
    ~EngineHeuristicsBuilder_v8()                                  = default;
    EngineHeuristicsBuilder_v8(EngineHeuristicsBuilder_v8 &&)      = delete;
    EngineHeuristicsBuilder_v8(EngineHeuristicsBuilder_v8 const &) = delete;
    EngineHeuristicsBuilder_v8 &
    operator=(EngineHeuristicsBuilder_v8 const &) = delete;

   private:
    EngineHeuristics_v8 m_heuristics;
};

template <std::size_t SIZE>
EngineConfigList
get_heuristics_list(std::array<cudnnBackendHeurMode_t, SIZE> modes,
                    OperationGraph_v8 &opGraph,
                    std::function<bool(cudnnBackendDescriptor_t)> filter_fn) {
    CUDNN_FRONTEND_UNUSED(modes);
    EngineConfigList filtered_configs;

    for (auto mode : modes) {
        if (mode == CUDNN_HEUR_MODES_COUNT) {
            continue;
        }
        auto heuristics = EngineHeuristicsBuilder_v8().setOperationGraph(opGraph).setHeurMode(mode).build();
        CUDNN_FE_LOG_LABEL_ENDL("Heuristic Mode " << mode << " has " << heuristics.getEngineConfigCount()
                                                  << " configurations.");
        auto &engine_config = heuristics.getEngineConfig(heuristics.getEngineConfigCount());
        cudnn_frontend::filter(engine_config, filtered_configs, filter_fn);
    }
    return filtered_configs;
}

#ifndef NV_CUDNN_DISABLE_EXCEPTION
#define NV_CUDNN_FE_TRY() try {
#else
#define NV_CUDNN_FE_TRY()
#endif

#ifndef NV_CUDNN_DISABLE_EXCEPTION
#define NV_CUDNN_FE_CATCH(...)                   \
    }                                            \
    catch (cudnn_frontend::cudnnException & e) { \
        __VA_ARGS__;                             \
    }
#else
#define NV_CUDNN_FE_CATCH(...)
#endif

#define NV_CUDNN_RETURN_IF_ERROR(heuristics)                   \
    do {                                                       \
        if (heuristics.get_status() != CUDNN_STATUS_SUCCESS) { \
            return heuristics.get_status();                    \
        }                                                      \
    } while (0);

#define NV_CUDNN_SET_STATUS_BREAK_OR_CONTINUE(status, is_last_status) \
    if (is_last_status || status != CUDNN_STATUS_SUCCESS) {           \
        statuses.push_back(status);                                   \
        if (status == CUDNN_STATUS_SUCCESS || evaluate_all) {         \
            continue;                                                 \
        }                                                             \
        break;                                                        \
    }

static inline cudnnStatus_t
get_heuristics_list_impl(cudnnBackendHeurMode_t heur_mode,
                         OperationGraph_v8 &opGraph,
                         std::function<bool(cudnnBackendDescriptor_t)> filter_fn,
                         int32_t sm_count,
                         EngineConfigList &filtered_configs,
                         std::shared_ptr<const DeviceProperties> device_properties = nullptr) {
    auto heuristics = EngineHeuristicsBuilder_v8()
                          .setDeviceProperties(device_properties)
                          .setOperationGraph(opGraph)
                          .setHeurMode(heur_mode)
                          .setSMCount(sm_count)
                          .build();
    NV_CUDNN_RETURN_IF_ERROR(heuristics);
    auto num_config = heuristics.getEngineConfigCount();
    NV_CUDNN_RETURN_IF_ERROR(heuristics);
    CUDNN_FE_LOG_LABEL_ENDL("Heuristic query for mode " << heur_mode << " has " << num_config << " configurations.");
    auto &engine_config = heuristics.getEngineConfig(num_config);
    NV_CUDNN_RETURN_IF_ERROR(heuristics);
    CUDNN_FE_LOG_LABEL_ENDL("Backend heuristics recommendation count: " << engine_config.size());
    cudnn_frontend::filter(engine_config, filtered_configs, filter_fn);
    return CUDNN_STATUS_SUCCESS;
}

static inline std::vector<cudnnStatus_t>
get_heuristics_list(std::vector<std::string> const &modes,
                    OperationGraph_v8 &opGraph,
                    std::function<bool(cudnnBackendDescriptor_t)> filter_fn,
                    EngineConfigList &filtered_configs,
                    bool evaluate_all                                         = false,
                    int32_t sm_count                                          = -1,
                    std::shared_ptr<const DeviceProperties> device_properties = nullptr) {
    std::vector<cudnnStatus_t> statuses;

    // Try building the heuristics for each mode
    // if fails push the status in the list of statuses
    for (auto &mode : modes) {
        if (mode.find("heuristics_instant") != std::string::npos ||
            mode.find("heuristics_mode_a") != std::string::npos) {
            auto heur_mode = CUDNN_HEUR_MODE_A;
            NV_CUDNN_FE_TRY();
            auto status_l =
                get_heuristics_list_impl(heur_mode, opGraph, filter_fn, sm_count, filtered_configs, device_properties);
            NV_CUDNN_SET_STATUS_BREAK_OR_CONTINUE(status_l, true);
            NV_CUDNN_FE_CATCH(NV_CUDNN_SET_STATUS_BREAK_OR_CONTINUE(e.getCudnnStatus(), true));

        } else if (mode.find("heuristics_fallback") != std::string::npos) {
            NV_CUDNN_FE_TRY();
            auto status_l = get_heuristics_list_impl(
                CUDNN_HEUR_MODE_FALLBACK, opGraph, filter_fn, sm_count, filtered_configs, device_properties);
            NV_CUDNN_SET_STATUS_BREAK_OR_CONTINUE(status_l, true);
            NV_CUDNN_FE_CATCH(NV_CUDNN_SET_STATUS_BREAK_OR_CONTINUE(e.getCudnnStatus(), true));
        } else if (mode.find("heuristics_mode_b") != std::string::npos) {
            auto heur_mode = CUDNN_HEUR_MODE_B;
            NV_CUDNN_FE_TRY();
            auto status_l =
                get_heuristics_list_impl(heur_mode, opGraph, filter_fn, sm_count, filtered_configs, device_properties);

            // Between cudnn version 8.3 and 8.6, when heur_mode_b heuristics did not succeed,
            // there was no fallback to the instant mode. We are here manually adding instant mode
            // to the heur_mode_b to alleviate this issue.
#if (CUDNN_VERSION >= 8300) && (CUDNN_VERSION < 8600)
            if (status_l != CUDNN_STATUS_SUCCESS) {
                status_l = get_heuristics_list_impl(
                    CUDNN_HEUR_MODE_INSTANT, opGraph, filter_fn, sm_count, filtered_configs, device_properties);
            }
#endif
            NV_CUDNN_SET_STATUS_BREAK_OR_CONTINUE(status_l, true);
#if (CUDNN_VERSION >= 8300) && (CUDNN_VERSION < 8600)
#ifndef NV_CUDNN_DISABLE_EXCEPTION
        }
        catch (cudnn_frontend::cudnnException &) {
            NV_CUDNN_FE_TRY();
            auto status_ =
                get_heuristics_list_impl(heur_mode, opGraph, filter_fn, sm_count, filtered_configs, device_properties);
            statuses.push_back(status_);
            NV_CUDNN_FE_CATCH(NV_CUDNN_SET_STATUS_BREAK_OR_CONTINUE(e.getCudnnStatus(), true));
        }
#endif
#else
            NV_CUDNN_FE_CATCH(NV_CUDNN_SET_STATUS_BREAK_OR_CONTINUE(e.getCudnnStatus(), true));
#endif
    }
}
return statuses;
}

static inline std::vector<cudnnStatus_t>
get_heuristics_list(std::vector<cudnn_frontend::HeurMode_t> const &modes,
                    OperationGraph_v8 &opGraph,
                    std::function<bool(cudnnBackendDescriptor_t)> filter_fn,
                    EngineConfigList &filtered_configs,
                    bool evaluate_all                                         = false,
                    int32_t sm_count                                          = -1,
                    std::shared_ptr<const DeviceProperties> device_properties = nullptr) {
    std::unordered_map<HeurMode_t, std::string> mode_to_string = {
        {HeurMode_t::A, "heuristics_mode_a"},
        {HeurMode_t::B, "heuristics_mode_b"},
        {HeurMode_t::FALLBACK, "heuristics_fallback"},
    };

    std::vector<std::string> string_modes(modes.size());
    std::transform(modes.begin(), modes.end(), string_modes.begin(), [&mode_to_string](const auto &mode) {
        return mode_to_string.at(mode);
    });

    return get_heuristics_list(
        string_modes, opGraph, filter_fn, filtered_configs, evaluate_all, sm_count, device_properties);
}

template <std::size_t SIZE>
std::vector<cudnnStatus_t>
get_heuristics_list(std::array<std::string, SIZE> modes,
                    OperationGraph_v8 &opGraph,
                    std::function<bool(cudnnBackendDescriptor_t)> filter_fn,
                    EngineConfigList &filtered_configs,
                    bool evaluate_all = false) {
    std::vector<std::string> modes_vector(modes.begin(), modes.end());
    return get_heuristics_list(modes_vector, opGraph, filter_fn, filtered_configs, evaluate_all);
}

#undef NV_CUDNN_FE_TRY
#undef NV_CUDNN_FE_CATCH
#undef NV_CUDNN_RETURN_IF_ERROR
#undef NV_CUDNN_SET_STATUS_BREAK_OR_CONTINUE

using EngineHeuristicsBuilde = EngineHeuristicsBuilder_v8;
using EngineHeuristics       = EngineHeuristics_v8;
}  // namespace cudnn_frontend

#if defined(_MSC_VER)
#pragma warning(pop)
#endif
