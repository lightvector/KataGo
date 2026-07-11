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

#include "cudnn_frontend_utils.h"

namespace cudnn_frontend {
namespace graph {
class RngNode;
}
}  // namespace cudnn_frontend

namespace cudnn_frontend {

///
/// Rng Descriptor Class
/// This class tells the properties of the Rng operation
/// Properties:
///
/// Use RngDescBuilder_v8 to build this class.
/// Describe returns a string describing the Rng operation
///
class RngDesc_v8 : public BackendDescriptor {
   public:
    friend class RngDescBuilder_v8;
    friend class graph::RngNode;
    std::string
    describe() const override {
        std::stringstream ss;
#if (CUDNN_VERSION >= 8700)
#ifndef CUDNN_FRONTEND_SKIP_JSON_LIB
        ss << "CUDNN_BACKEND_RNG_DESCRIPTOR: " << "Distribution Type: " << json{distribution}
#else
        ss << "CUDNN_BACKEND_RNG_DESCRIPTOR: " << "Distribution Type: " << int(distribution)
#endif
           << ", Normal Distribution Mean: " << normal_dist_mean
           << ", Normal Distribution Standard Deviation: " << normal_dist_std_dev
           << ", Uniform Distribution Maximum: " << uniform_dist_max
           << ", Uniform Distribution Minimum: " << uniform_dist_min
           << ", Bernoulli Distribution Probability: " << bernoulli_dist_probability;
#endif
        return ss.str();
    }

    RngDesc_v8(RngDesc_v8 &&from) = default;
    RngDesc_v8 &
    operator=(RngDesc_v8 &&) = default;

    ~RngDesc_v8() = default;

    /** @defgroup RngDescBuilder_v8
     *  Get individual property of RngDesc_v8 class
     *  @{
     */

    double
    getNormalDistMean() const {
        return normal_dist_mean;
    }

    double
    getNormalDistStdDev() const {
        return normal_dist_std_dev;
    }

    double
    getUniformDistMax() const {
        return uniform_dist_max;
    }

    double
    getUniformDistMin() const {
        return normal_dist_std_dev;
    }

    double
    getBernoulliDistProbability() const {
        return bernoulli_dist_probability;
    }

    RngDistribution_t
    getDistribution() const {
        return distribution;
    }

    /** @} */

   private:
    RngDesc_v8()                   = default;
    RngDesc_v8(RngDesc_v8 const &) = delete;
    RngDesc_v8 &
    operator=(RngDesc_v8 const &) = delete;

    // default values for attributes
    double normal_dist_mean           = -1;
    double normal_dist_std_dev        = -1;
    double uniform_dist_max           = -1;
    double uniform_dist_min           = -1;
    double bernoulli_dist_probability = -1;

    RngDistribution_t distribution = RngDistribution_t::NOT_SET;
};

///
/// RngDescBuilder_v8 Class
/// Helper class used to build RngDesc_v8 class
class RngDescBuilder_v8 {
   public:
    /** @defgroup RngDescBuilder_v8
     *  Set individual property of RngDesc_v8 class
     *  @{
     */

    //! Set Rng distribution for the Rng Operation
    auto
    setRngDistribution(RngDistribution_t distribution_) -> RngDescBuilder_v8 & {
        m_RngDesc.distribution = distribution_;
        return *this;
    }

#if (CUDNN_VERSION >= 8700)
    //! Set Rng distribution for the Rng Operation
    auto
    setRngDistribution(cudnnRngDistribution_t distribution_) -> RngDescBuilder_v8 & {
        m_RngDesc.distribution = detail::convert_from_cudnn_type(distribution_);
        return *this;
    }

#endif

    //! Set normal distribution params (mean and std dev) for the Rng Operation
    auto
    setNormalDistParams(double normal_dist_mean_, double normal_dist_std_dev_) -> RngDescBuilder_v8 & {
        m_RngDesc.normal_dist_mean    = normal_dist_mean_;
        m_RngDesc.normal_dist_std_dev = normal_dist_std_dev_;
        return *this;
    }

    //! Set normal distribution mean for the Rng Operation
    auto
    setNormalDistMean(double normal_dist_mean_) -> RngDescBuilder_v8 & {
        m_RngDesc.normal_dist_mean = normal_dist_mean_;
        return *this;
    }

    //! Set normal distribution std dev for the Rng Operation
    auto
    setNormalDistStdDev(double normal_dist_std_dev_) -> RngDescBuilder_v8 & {
        m_RngDesc.normal_dist_std_dev = normal_dist_std_dev_;
        return *this;
    }

    //! Set uniform distribution params (min and max) for the Rng Operation
    auto
    setUniformDistParams(double uniform_dist_max_, double uniform_dist_min_) -> RngDescBuilder_v8 & {
        m_RngDesc.uniform_dist_max = uniform_dist_max_;
        m_RngDesc.uniform_dist_min = uniform_dist_min_;
        return *this;
    }

    //! Set uniform distribution max for the Rng Operation
    auto
    setUniformDistMax(double uniform_dist_max_) -> RngDescBuilder_v8 & {
        m_RngDesc.uniform_dist_max = uniform_dist_max_;
        return *this;
    }

    //! Set uniform distribution min for the Rng Operation
    auto
    setUniformDistMin(double uniform_dist_min_) -> RngDescBuilder_v8 & {
        m_RngDesc.uniform_dist_min = uniform_dist_min_;
        return *this;
    }

    //! Set bernoulli distribution probability for the Rng Operation
    auto
    setBernoulliDistProbability(double bernoulli_dist_probability_) -> RngDescBuilder_v8 & {
        m_RngDesc.bernoulli_dist_probability = bernoulli_dist_probability_;
        return *this;
    }

    /** @} */

    //! constructs the RngDesc_v8 by calling the cudnn API
    //! Throws the appropriate error message
    RngDesc_v8 &&
    build() {
#if (CUDNN_VERSION >= 8700)
        // Create a descriptor. Memory allocation happens here.
        NV_CUDNN_FE_DYNAMIC_CHECK_BACKEND_DESCRIPTOR(
            8700, m_RngDesc, "CUDNN_BACKEND_RNG_DESCRIPTOR: Requires cudnn 8.7.0");

        auto status = m_RngDesc.initialize_managed_backend_pointer(CUDNN_BACKEND_RNG_DESCRIPTOR);
        if (status != CUDNN_STATUS_SUCCESS) {
            set_error_and_throw_exception(&m_RngDesc, status, "CUDNN_BACKEND_RNG_DESCRIPTOR: cudnnCreate Failed");
            return std::move(m_RngDesc);
        }

        // Once Created lets set the descriptor parameters.
        cudnnRngDistribution_t cudnn_rng_distribution;
        status = detail::convert_to_cudnn_type(m_RngDesc.distribution, cudnn_rng_distribution);
        if (status != CUDNN_STATUS_SUCCESS) {
            set_error_and_throw_exception(
                &m_RngDesc, status, "CUDNN_BACKEND_RNG_DESCRIPTOR: SetAttribute CUDNN_ATTR_RNG_DISTRIBUTION Failed");
            return std::move(m_RngDesc);
        }

        status = detail::set_attribute(m_RngDesc.pointer->get_backend_descriptor(),
                                       CUDNN_ATTR_RNG_DISTRIBUTION,
                                       CUDNN_TYPE_RNG_DISTRIBUTION,
                                       1,
                                       &cudnn_rng_distribution);
        if (status != CUDNN_STATUS_SUCCESS) {
            set_error_and_throw_exception(
                &m_RngDesc, status, "CUDNN_BACKEND_RNG_DESCRIPTOR: SetAttribute CUDNN_ATTR_RNG_DISTRIBUTION Failed");
            return std::move(m_RngDesc);
        }

        status = detail::set_attribute(m_RngDesc.pointer->get_backend_descriptor(),
                                       CUDNN_ATTR_RNG_NORMAL_DIST_MEAN,
                                       CUDNN_TYPE_DOUBLE,
                                       1,
                                       &(m_RngDesc.normal_dist_mean));
        if (status != CUDNN_STATUS_SUCCESS) {
            set_error_and_throw_exception(
                &m_RngDesc,
                status,
                "CUDNN_BACKEND_RNG_DESCRIPTOR: SetAttribute CUDNN_ATTR_RNG_NORMAL_DIST_MEAN Failed");
            return std::move(m_RngDesc);
        }

        status = detail::set_attribute(m_RngDesc.pointer->get_backend_descriptor(),
                                       CUDNN_ATTR_RNG_NORMAL_DIST_STANDARD_DEVIATION,
                                       CUDNN_TYPE_DOUBLE,
                                       1,
                                       &(m_RngDesc.normal_dist_std_dev));
        if (status != CUDNN_STATUS_SUCCESS) {
            set_error_and_throw_exception(
                &m_RngDesc,
                status,
                "CUDNN_BACKEND_RNG_DESCRIPTOR: SetAttribute CUDNN_ATTR_RNG_NORMAL_DIST_STANDARD_DEVIATION Failed");
            return std::move(m_RngDesc);
        }

        status = detail::set_attribute(m_RngDesc.pointer->get_backend_descriptor(),
                                       CUDNN_ATTR_RNG_UNIFORM_DIST_MAXIMUM,
                                       CUDNN_TYPE_DOUBLE,
                                       1,
                                       &(m_RngDesc.uniform_dist_max));
        if (status != CUDNN_STATUS_SUCCESS) {
            set_error_and_throw_exception(
                &m_RngDesc,
                status,
                "CUDNN_BACKEND_RNG_DESCRIPTOR: SetAttribute CUDNN_ATTR_RNG_UNIFORM_DIST_MAXIMUM Failed");
            return std::move(m_RngDesc);
        }

        status = detail::set_attribute(m_RngDesc.pointer->get_backend_descriptor(),
                                       CUDNN_ATTR_RNG_UNIFORM_DIST_MINIMUM,
                                       CUDNN_TYPE_DOUBLE,
                                       1,
                                       &(m_RngDesc.uniform_dist_min));
        if (status != CUDNN_STATUS_SUCCESS) {
            set_error_and_throw_exception(
                &m_RngDesc,
                status,
                "CUDNN_BACKEND_RNG_DESCRIPTOR: SetAttribute CUDNN_ATTR_RNG_UNIFORM_DIST_MINIMUM Failed");
            return std::move(m_RngDesc);
        }

        status = detail::set_attribute(m_RngDesc.pointer->get_backend_descriptor(),
                                       CUDNN_ATTR_RNG_BERNOULLI_DIST_PROBABILITY,
                                       CUDNN_TYPE_DOUBLE,
                                       1,
                                       &(m_RngDesc.bernoulli_dist_probability));
        if (status != CUDNN_STATUS_SUCCESS) {
            set_error_and_throw_exception(
                &m_RngDesc,
                status,
                "CUDNN_BACKEND_RNG_DESCRIPTOR: SetAttribute CUDNN_ATTR_RNG_BERNOULLI_DIST_PROBABILITY Failed");
            return std::move(m_RngDesc);
        }

        // Finalizing the descriptor
        status = detail::finalize(m_RngDesc.pointer->get_backend_descriptor());
        if (status != CUDNN_STATUS_SUCCESS) {
            set_error_and_throw_exception(&m_RngDesc, status, "CUDNN_BACKEND_RNG_DESCRIPTOR: cudnnFinalize Failed");
            return std::move(m_RngDesc);
        }
        CUDNN_FE_LOG_LABEL_ENDL(m_RngDesc);
        return std::move(m_RngDesc);
#else
        set_error_and_throw_exception(
            &m_RngDesc, CUDNN_STATUS_NOT_SUPPORTED, "CUDNN_RNG_DESCRIPTOR: Rng only supported in cuDNN v8.7 or later");
        return std::move(m_RngDesc);
#endif
    }

    explicit RngDescBuilder_v8()                 = default;
    ~RngDescBuilder_v8()                         = default;
    RngDescBuilder_v8(RngDescBuilder_v8 &&)      = delete;
    RngDescBuilder_v8(RngDescBuilder_v8 const &) = delete;
    RngDescBuilder_v8 &
    operator=(RngDescBuilder_v8 const &) = delete;

   private:
    RngDesc_v8 m_RngDesc;
};

using RngDesc        = RngDesc_v8;
using RngDescBuilder = RngDescBuilder_v8;
}  // namespace cudnn_frontend
