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

#include <tuple>
#include <unordered_map>
#include <map>
#include <memory>
#include <mutex>

#include "cudnn_frontend_OperationGraph.h"
#include "cudnn_frontend_ExecutionPlan.h"

/// Execution Plan Caching:
/// Goal is to auto-tune once and then save the best auto-tuned result for a problem for later use.
/// For every unique Operation Graph (denoted by a string) we have a set of plans identified by a feature vector.
/// The feature vector could be Tensor dimension/data_type and so on.
/// Multiple operation Graph can share a feature vector type but may have different Execution Plan(s).

/// The v1 cache has the following format.
/// It is the reponsibility of the user to query the correct cache for the given device/operation graph combination.
/***
 *    device_id_0 Operation_Graph0 (conv_fprop)
 *    -------------------------------------------------------------------------------
 *    | Feature_vector_type0_val0   |  Plan A0   |
 *    | Feature_vector_type0_val1   |  Plan B0   |
 *    ===============================================================================
 *
 *    device_id_0 Operation_Graph1 (dgrad)
 *    -------------------------------------------------------------------------------
 *    | Feature_vector_type1_val0   |  Plan A1   |
 *    | Feature_vector_type1_val1   |  Plan B1   |
 *    ===============================================================================
 *
 *    device_id_0 Operation_Graph2 (wgrad)
 *    -------------------------------------------------------------------------------
 *    | Feature_vector_type2_val0   |  Plan B2   |
 *    ===============================================================================
 *
 *    device_id_1 Operation_Graph0 (conv_fprop)
 *    -------------------------------------------------------------------------------
 *    | Feature_vector_type0_val0   |  Plan A0   |
 *    | Feature_vector_type0_val1   |  Plan B0   |
 *    ===============================================================================
 *
 *    device_id_1 Operation_Graph1 (dgrad)
 *    -------------------------------------------------------------------------------
 *    | Feature_vector_type1_val0   |  Plan A1   |
 *    | Feature_vector_type1_val1   |  Plan B1   |
 *    ===============================================================================
 *
 *    device_id_1 Operation_Graph2 (wgrad)
 *    -------------------------------------------------------------------------------
 *    | Feature_vector_type2_val0   |  Plan B2   |
 *    ===============================================================================
 */

namespace cudnn_frontend {

/// Plan Cache structure for the above table
class ExecutionPlanCache_v1 {
   protected:
    struct compare {
        bool
        operator()(const feature_vector_t &fv1, const feature_vector_t &fv2) const {
            return fv1 < fv2;
        }
    };

    std::string name = "plan_cache_[unnamed]";

    /// String to map of feature_vector to execution plan
    /// For a given FeatureVector of type T according to the Operation Graph, we get the plan.
    using FeatureVectorToPlanMap = std::map<cudnn_frontend::feature_vector_t,
                                            cudnn_frontend::ExecutionPlan,
                                            cudnn_frontend::ExecutionPlanCache_v1::compare>;
    FeatureVectorToPlanMap cache;

    mutable std::mutex cache_mutex;

   public:
    virtual bool
    is_fastest_plan_stable(const cudnn_frontend::OperationGraph &op_graph, const std::string &tag) {
        CUDNN_FRONTEND_UNUSED(op_graph);
        CUDNN_FRONTEND_UNUSED(tag);
        return true;
    }

    void
    add_plan_to_cache(const cudnn_frontend::OperationGraph &op_graph, const cudnn_frontend::ExecutionPlan &plan) {
        std::lock_guard<std::mutex> guard(cache_mutex);
        cache.insert(std::make_pair(op_graph.getFeatureVector(), plan));
        CUDNN_FE_LOG_LABEL_ENDL("Added to " << name << " " << op_graph.getTag());
    }

    ExecutionPlanCache_v1(const char *name_) { name = name_; }

    const std::string &
    get_name() const {
        return name;
    }

    // Plan is the output here.
    bool
    get_plan_from_cache(const cudnn_frontend::OperationGraph &op_graph,
                        const cudnn_frontend::ExecutionPlan *&plan) const {
        {
            std::lock_guard<std::mutex> guard(cache_mutex);
            auto it = cache.find(op_graph.getFeatureVector());

            if (it == cache.end()) {
                CUDNN_FE_LOG_LABEL_ENDL("Cached Plan Not Found in " << name);
                return false;
            }
            plan = &(it->second);
        }
        CUDNN_FE_LOG_LABEL_ENDL("Cached Plan Found in " << name);
        return true;
    }

    virtual ~ExecutionPlanCache_v1() = default;
};

class ExecutionPlanCache_v2 : public ExecutionPlanCache_v1 {
    using SaturationTracker = std::map<std::pair<cudnn_frontend::feature_vector_t, std::string>, int32_t>;
    SaturationTracker tracker;

    int32_t saturationCount = 1;

   public:
    virtual bool
    is_fastest_plan_stable(const cudnn_frontend::OperationGraph &op_graph, const std::string &tag) {
        if (saturationCount == 1) {
            return true;
        }  // Special case. Always add to the cache.

        // If plan cache is already created for the op_graph no need to update.
        // Ideally, one will auto-tune only if the plan cache has no plan for the op_graph.
        cudnn_frontend::ExecutionPlan const *plan = nullptr;
        if (get_plan_from_cache(op_graph, plan)) {
            CUDNN_FE_LOG_LABEL_ENDL("SaturationTracker " << name << " " << op_graph.getTag() << " " << tag
                                                         << " plan already in cache.");
            return false;
        }

        // Lock the cache and increase the count till we saturate
        std::lock_guard<std::mutex> guard(cache_mutex);
        auto cnt = tracker[std::make_pair(op_graph.getFeatureVector(), tag)] += 1;
        CUDNN_FE_LOG_LABEL_ENDL("SaturationTracker " << name << " " << op_graph.getTag() << " " << tag << " " << cnt);
        return cnt >= saturationCount;
    }

    void
    set_saturation_count(int32_t count) {
        saturationCount = count;
    }

    ExecutionPlanCache_v2(const char *name_) : ExecutionPlanCache_v1(name_) {}

    virtual ~ExecutionPlanCache_v2() = default;
};

using ExecutionPlanCache = ExecutionPlanCache_v2;

}  // namespace cudnn_frontend
