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

// Suppress MSVC warning C4756 (overflow in constant arithmetic) that occurs
// in MSVC's <optional> header with certain compiler versions
#ifdef _MSC_VER
#pragma warning(disable : 4756)
#endif

/*! \mainpage CUDNN FRONTEND API
 *
 * \section Introduction
 *
 * The cuDNN Frontend API is a C++ header-only library that demonstrates how to use the cuDNN C backend API. The cuDNN C
 * backend API is documented in the cuDNN developer guide.
 *
 * \section Why use Frontend API
 *
 * Consider the following code snippet which showcases cudnnBackendTensor creation using the backend API and its
 * equivalent front-end API code. Many among the backend constructs follow similar pattern.
 *
 *  ~~~~~~~~~~~~~~~{.cpp}
 *
 *  ===========================================================================================
 *  auto check_status = [](cudnnStatus_t status) { assert (status == CUDNN_STATUS_SUCCESS); };
 *  ===========================================================================================
 *  // Backend code for Tensor Creation.
 *  cudnnBackendDescriptor_t tensor;
 *
 *  check_status (cudnnBackendCreateDescriptor(CUDNN_BACKEND_TENSOR_DESCRIPTOR, &tensor));
 *
 *  check_status (cudnnBackendSetAttribute(tensor,
 *                                         CUDNN_ATTR_TENSOR_DATA_TYPE,
 *                                         CUDNN_TYPE_DATA_TYPE,
 *                                         1,
 *                                         &data_type));
 *  check_status (cudnnBackendSetAttribute(tensor,
 *                                         CUDNN_ATTR_TENSOR_DIMENSIONS,
 *                                         CUDNN_TYPE_INT64,
 *                                         tensor_dim.size(),
 *                                         tensor_dim.data()));
 *  check_status (cudnnBackendSetAttribute(tensor,
 *                                         CUDNN_ATTR_TENSOR_STRIDES,
 *                                         CUDNN_TYPE_INT64,
 *                                         tensor_str.size(),
 *                                         tensor_str.data()));
 *  check_status (cudnnBackendSetAttribute(tensor,
 *                                         CUDNN_ATTR_TENSOR_UNIQUE_ID,
 *                                         CUDNN_TYPE_INT64,
 *                                         1,
 *                                         &id));
 *  check_status (cudnnBackendSetAttribute(tensor,
 *                                         CUDNN_ATTR_TENSOR_BYTE_ALIGNMENT,
 *                                         CUDNN_TYPE_INT64,
 *                                         1,
 *                                         &alignment));
 *  check_status (cudnnBackendFinalize(tensor));
 *
 *  check_status (cudnnBackendDestroyDescriptor(tensor));
 *  ===========================================================================================
 *  // FrontEnd equivalent code.
 *  auto tensor =  cudnn_frontend::TensorBuilder()
 *                     .setDim(tensor_dim.size(), tensor_dim.data())
 *                     .setStrides(tensor_str.size(), tensor_str.data())
 *                     .setId(id)
 *                     .setAlignment(alignment)
 *                     .setDataType(data_type)
 *                     .build();
 *  check_status(tensor.get_status());
 *  ===========================================================================================
 *
 *  ~~~~~~~~~~~~~~~
 *
 *  Frontend API serves two major purpose as a companion to the backend API.
 *  - Functional additions:
 *      - Support for auto-tuning. (cudnnGet and cudnnFind)
 *      - Errata filters.
 *  - Programmatic ease:
 *      - Easy memory management for the cudnnBackendDescriptor_t (RAII based classes).
 *      - Error handling with optional exception support. Better error messages.
 *      - Fewer lines of code (5-10x reduction in LOC).
 *      - Simpler samples on how to use the new API.
 */

#include <cudnn.h>

#include "cudnn_frontend_ConvDesc.h"
#include "cudnn_frontend_Heuristics.h"
#include "cudnn_frontend_Engine.h"
#include "cudnn_frontend_EngineConfig.h"
#include "cudnn_frontend_EngineFallbackList.h"
#include "cudnn_frontend_Errata.h"
#include "cudnn_frontend_ExecutionPlan.h"
#include "cudnn_frontend_Filters.h"
#include "cudnn_frontend_Operation.h"
#include "cudnn_frontend_OperationGraph.h"
#include "cudnn_frontend_Tensor.h"
#include "cudnn_frontend_VariantPack.h"
#include "cudnn_frontend_PointWiseDesc.h"
#include "cudnn_frontend_MatMulDesc.h"
#include "cudnn_frontend_Logging.h"
#include "cudnn_frontend_Reorder_Tensor.h"
#include "cudnn_frontend_ExecutionPlanCache.h"
#include "cudnn_frontend_utils.h"

#include "cudnn_frontend_Resample.h"

#include "cudnn_frontend/graph_interface.h"
#include "cudnn_frontend/utils/serialize.h"
#include "cudnn_frontend/backend/kernel_cache.h"
#include "cudnn_frontend/utils/attn_score_modifiers.h"
#include "cudnn_frontend/backend/device_properties.h"

#include "cudnn_frontend_version.h"

namespace cudnn_frontend {
using ConvDesc                  = ConvDesc_v8;
using ConvDescBuilder           = ConvDescBuilder_v8;
using ReductionDesc             = ReductionDesc_v8;
using ReductionDescBuilder      = ReductionDescBuilder_v8;
using EngineHeuristicsBuilder   = EngineHeuristicsBuilder_v8;
using EngineHeuristics          = EngineHeuristics_v8;
using EngineBuilder             = EngineBuilder_v8;
using Engine                    = Engine_v8;
using EngineConfig              = EngineConfig_v8;
using EngineConfigBuilder       = EngineConfigBuilder_v8;
using EngineFallbackList        = EngineFallbackList_v8;
using EngineFallbackListBuilder = EngineFallbackListBuilder_v8;
using ResampleDesc              = ResampleDesc_v8;
using ResampleDescBuilder       = ResampleDescBuilder_v8;
}  // namespace cudnn_frontend
