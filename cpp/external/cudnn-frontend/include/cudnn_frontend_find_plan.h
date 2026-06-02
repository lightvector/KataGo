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

#include <iomanip>
#include <set>
#include "cudnn_frontend_EngineConfigGenerator.h"

namespace cudnn_frontend {

/// Sorts the execution plans by their run time.
/// The run time of plan may not trivial and hence we
/// run it multiple times till we get a stable value.
/// We have an additional dry-run which helps stabilize the
/// time further.
template <CudnnFindSamplingTechnique samplingTechnique>
auto
time_sorted_plan(cudnnHandle_t handle,
                 executionPlans_t plans,
                 VariantPack const &variantPack,
                 uint64_t max_plans_to_evaluate = 1000) -> executionPlans_t {
    executionPlans_t time_sorted_plans;

    auto plan_cmp = [](const ExecutionPlan &a, const ExecutionPlan &b) {
        return a.getExecutionTime() < b.getExecutionTime();
    };
    std::set<std::reference_wrapper<ExecutionPlan>, decltype(plan_cmp)> timed_execution_plans(plan_cmp);

    const int maxIterCount         = (samplingTechnique == CudnnFindSamplingTechnique::CUDNN_FIND_SAMPLE_ONCE) ? 1
                                     : (samplingTechnique == CudnnFindSamplingTechnique::CUDNN_FIND_SAMPLE_MEDIAN_OF_THREE)
                                         ? 3
                                         : 100;
    const float threshhold         = 0.95f;
    uint64_t successful_plan_count = 0;
    cudaEvent_t start, stop;
    detail::cuda_event_create(&start);
    detail::cuda_event_create(&stop);
    detail::cuda_device_synchronize();

    cudaStream_t stream = nullptr;
    detail::get_stream(handle, &stream);

    for (auto &plan : plans) {
        float time_ms       = 0.0f;
        float final_time_ms = 0.0f;
        float min_time_ms   = std::numeric_limits<float>::max();

        // Warm-up run
        auto warmup_status = detail::execute(handle, plan.get_raw_desc(), variantPack.get_raw_desc());
        if (warmup_status != CUDNN_STATUS_SUCCESS) {
            CUDNN_FE_LOG_LABEL_ENDL("Plan " << plan.getTag() << " failed with " << to_string(warmup_status));
            continue;
        }
        successful_plan_count++;
        detail::cuda_device_synchronize();

        float time_run_ms[3] = {0.0f, 0.0f, 0.0f};
        for (int i = 0; i < maxIterCount; i++) {
            detail::cuda_event_record(start, stream);

            detail::execute(handle, plan.get_raw_desc(), variantPack.get_raw_desc());

            detail::cuda_event_record(stop, stream);
            detail::cuda_event_synchronize(stop);
            detail::cuda_event_elapsed_time(&time_ms, start, stop);

            if constexpr (samplingTechnique == CudnnFindSamplingTechnique::CUDNN_FIND_SAMPLE_TILL_STABLE) {
                final_time_ms = std::min(min_time_ms, time_ms);
                if (time_ms / min_time_ms < threshhold) {
                    min_time_ms = final_time_ms;
                } else {
                    break;
                }
            } else if constexpr (samplingTechnique == CudnnFindSamplingTechnique::CUDNN_FIND_SAMPLE_MEDIAN_OF_THREE) {
                time_run_ms[i] = time_ms;
                if (i == maxIterCount - 1) {
                    auto min_of_first_two = std::min(time_run_ms[0], time_run_ms[1]);
                    auto max_of_first_two = std::max(time_run_ms[0], time_run_ms[1]);
                    final_time_ms         = std::max(min_of_first_two, std::min(max_of_first_two, time_run_ms[2]));
                }
            } else {
                final_time_ms = time_ms;
            }
        }
        CUDNN_FE_LOG_LABEL_ENDL("Plan " << plan.getTag() << " took " << std::setw(10) << final_time_ms);
        plan.setExecutionTime(final_time_ms);
        timed_execution_plans.insert(plan);
        if (successful_plan_count >= max_plans_to_evaluate) {
            CUDNN_FE_LOG_LABEL_ENDL("Successfully profiled " << max_plans_to_evaluate << "plans.");
            break;
        }
    }

    for (ExecutionPlan &plan : timed_execution_plans) {
        time_sorted_plans.emplace_back(std::move(plan));
    }

    detail::cuda_event_destroy(start);
    detail::cuda_event_destroy(stop);

    CUDNN_FE_LOG_LABEL_ENDL("Auto-tuning returns " << time_sorted_plans.size() << " plans.");

    return time_sorted_plans;
}

template <CudnnFindSamplingTechnique samplingTechnique>
auto
EngineConfigGenerator::cudnnFindPlan(cudnnHandle_t handle,
                                     cudnn_frontend::OperationGraph &opGraph,
                                     cudnn_frontend::VariantPack const &variantPack) -> executionPlans_t {
    /// Creating a set of execution plans that are supported.
    executionPlans_t plans = cudnnGetPlan(handle, opGraph);
    return time_sorted_plan<samplingTechnique>(handle, std::move(plans), variantPack);
}

template <CudnnFindSamplingTechnique samplingTechnique>
auto
EngineConfigGenerator::cudnnFindPlan(cudnnHandle_t handle,
                                     cudnn_frontend::OperationGraph &opGraph,
                                     cudnn_frontend::VariantPack const &variantPack,
                                     Predicate pred) -> executionPlans_t {
    /// Creating a set of execution plans that are supported.
    executionPlans_t plans = cudnnGetPlan(handle, opGraph, pred);
    return time_sorted_plan<samplingTechnique>(handle, std::move(plans), variantPack);
}

template <CudnnFindSamplingTechnique samplingTechnique>
auto
EngineConfigGenerator::cudnnFindPlanAndCache(cudnnHandle_t handle,
                                             cudnn_frontend::OperationGraph &opGraph,
                                             cudnn_frontend::VariantPack const &variantPack,
                                             cudnn_frontend::ExecutionPlanCache &cache,
                                             Predicate pred) -> cudnn_frontend::ExecutionPlan {
    /// Creating a set of execution plans that are supported.
    auto sorted_plans = cudnnFindPlan<samplingTechnique>(handle, opGraph, variantPack, pred);
    /// Check if the fastest plan is stable enough to be added to the plan cache
    if (cache.is_fastest_plan_stable(opGraph, sorted_plans.front().getTag())) {
        cache.add_plan_to_cache(opGraph, sorted_plans.front());
    }
    return sorted_plans.front();
}

}  // namespace cudnn_frontend
