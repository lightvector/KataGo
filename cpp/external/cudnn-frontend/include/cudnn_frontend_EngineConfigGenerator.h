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

#include <cudnn_frontend.h>

namespace cudnn_frontend {

/// Variety of renames.
using executionPlans_t = std::vector<cudnn_frontend::ExecutionPlan>;
using Predicate        = std::function<bool(cudnn_frontend::ExecutionPlan const &plan)>;
using GeneratorSource  = std::function<cudnn_frontend::EngineConfigList(cudnn_frontend::OperationGraph &)>;

enum class CudnnFindSamplingTechnique {
    CUDNN_FIND_SAMPLE_ONCE,             //!< Sample once quick but may have unstable values
    CUDNN_FIND_SAMPLE_MEDIAN_OF_THREE,  //!< Sample 3 times and take median.
    CUDNN_FIND_SAMPLE_TILL_STABLE       //!< Sample multiple times till stable.
};

/// EngineConfigGenerator class
/// Contains a vector of methods that generate a vector of backend descriptor
/// that can be used to create a plan for the method.
class EngineConfigGenerator {
   private:
    std::vector<GeneratorSource> engine_config_generators;

   public:
    /// Constructor that takes int a array of function pointers that will be called later.
    /// in the generate_engine_config function.
    EngineConfigGenerator(int const sourceSize, GeneratorSource const *sources) {
        for (int i = 0; i < sourceSize; i++) {
            engine_config_generators.push_back(sources[i]);
        }
    };

    /// Calls the vector of engine_config_generators one by one and concatenates the generated
    /// engine together into a single list.
    auto
    generate_engine_config(cudnn_frontend::OperationGraph &opGraph) -> cudnn_frontend::EngineConfigList {
        cudnn_frontend::EngineConfigList engine_configs;
        for (auto fn : engine_config_generators) {
            cudnn_frontend::EngineConfigList new_engine_config = fn(opGraph);
            CUDNN_FE_LOG_LABEL_ENDL("Called engine config generator and produced " << new_engine_config.size()
                                                                                   << " configs.");
            std::copy(new_engine_config.begin(), new_engine_config.end(), std::back_inserter(engine_configs));
            new_engine_config.clear();
        }
        return engine_configs;
    }

    /// Returns the concatenated plan in the order of heuristic results.
    auto
    cudnnGetPlan(cudnnHandle_t handle, cudnn_frontend::OperationGraph &opGraph, Predicate pred, size_t max_plans = 1000)
        -> executionPlans_t;
    auto
    cudnnGetPlan(cudnnHandle_t handle, cudnn_frontend::OperationGraph &opGraph, size_t max_plans = 1000)
        -> executionPlans_t;

    /// Reruns the concatenated plans and measures the execution time following which
    /// a sorted order of executionPlans are return to the user.
    template <CudnnFindSamplingTechnique samplingTechnique>
    auto
    cudnnFindPlan(cudnnHandle_t handle,
                  cudnn_frontend::OperationGraph &opGraph,
                  cudnn_frontend::VariantPack const &variantPack,
                  Predicate pred) -> executionPlans_t;

    template <CudnnFindSamplingTechnique samplingTechnique>
    auto
    cudnnFindPlan(cudnnHandle_t handle,
                  cudnn_frontend::OperationGraph &opGraph,
                  cudnn_frontend::VariantPack const &variantPack) -> executionPlans_t;

    template <CudnnFindSamplingTechnique samplingTechnique>
    auto
    cudnnFindPlanAndCache(
        cudnnHandle_t handle,
        cudnn_frontend::OperationGraph &opGraph,
        cudnn_frontend::VariantPack const &variantPack,
        cudnn_frontend::ExecutionPlanCache &cache,
        Predicate pred = [](const cudnn_frontend::ExecutionPlan &) { return false; }) -> cudnn_frontend::ExecutionPlan;
};

/// Filter out the execution plan based on the prerequisite conditions.
/// Goes through vector of execution plans and if the predicate returns
/// not to block (false), it is inserted into the filtered plans.
static auto
filter(Predicate pred, executionPlans_t &plans) -> executionPlans_t {
    executionPlans_t filtered_plans;
    for (auto &plan : plans) {
        CUDNN_FE_LOG_LABEL("Filtered ");
        if (!pred(plan)) {
            CUDNN_FE_LOG("and Added ");
            filtered_plans.emplace_back(std::move(plan));
        }
        if (filtered_plans.size()) {
            CUDNN_FE_LOG(filtered_plans.back().getTag() << std::endl);
        }
    }
    CUDNN_FE_LOG_LABEL_ENDL("Filtered plans count " << filtered_plans.size());
    return filtered_plans;
}
}  // namespace cudnn_frontend
