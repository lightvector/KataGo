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

namespace cudnn_frontend {

// If filter_fn returns true
// The engine config will be filtered out and will
// not be part of the to list.
static void
filter(EngineConfigList &from, EngineConfigList &to, std::function<bool(cudnnBackendDescriptor_t)> filter_fn) {
    auto p = std::stable_partition(from.begin(), from.end(), [filter_fn](ManagedOpaqueDescriptor &p) {
        return filter_fn(const_cast<cudnnBackendDescriptor_t>(p->get_backend_descriptor()));
    });
    // range insert with move
    to.insert(to.end(), std::make_move_iterator(p), std::make_move_iterator(from.end()));
    // erase the moved-from elements.
    from.erase(p, from.end());
}

template <cudnnBackendNumericalNote_t NUMERIC_NOTE>
bool
hasNumericalNote(cudnnBackendDescriptor_t engine_config) {
    bool hasNumerics                 = false;
    auto status                      = CUDNN_STATUS_SUCCESS;
    ManagedOpaqueDescriptor engine   = make_shared_backend_pointer(CUDNN_BACKEND_ENGINE_DESCRIPTOR);
    cudnnBackendDescriptor_t engine_ = engine->get_backend_descriptor();
    int64_t engine_count             = -1;
    status                           = detail::get_attribute(
        engine_config, CUDNN_ATTR_ENGINECFG_ENGINE, CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &engine_count, &engine_);
    if (status == CUDNN_STATUS_SUCCESS) {
        cudnnBackendNumericalNote_t notes[CUDNN_NUMERICAL_NOTE_TYPE_COUNT];
        std::fill_n(notes, CUDNN_NUMERICAL_NOTE_TYPE_COUNT, CUDNN_NUMERICAL_NOTE_TYPE_COUNT);
        int64_t elem_count = 0;
        detail::get_attribute(engine->get_backend_descriptor(),
                              CUDNN_ATTR_ENGINE_NUMERICAL_NOTE,
                              CUDNN_TYPE_NUMERICAL_NOTE,
                              CUDNN_NUMERICAL_NOTE_TYPE_COUNT,
                              &elem_count,
                              notes);
        if (std::any_of(
                notes, notes + elem_count, [](cudnnBackendNumericalNote_t note) { return note == NUMERIC_NOTE; })) {
            hasNumerics = true;
        }
    }
    return hasNumerics;
}

template <cudnnBackendBehaviorNote_t BEHAVIOR_NOTE>
bool
hasBehaviorNote(cudnnBackendDescriptor_t engine_config) {
    bool hasBehavior                 = false;
    auto status                      = CUDNN_STATUS_SUCCESS;
    ManagedOpaqueDescriptor engine   = make_shared_backend_pointer(CUDNN_BACKEND_ENGINE_DESCRIPTOR);
    cudnnBackendDescriptor_t engine_ = engine->get_backend_descriptor();
    int64_t engine_count             = -1;
    status                           = detail::get_attribute(
        engine_config, CUDNN_ATTR_ENGINECFG_ENGINE, CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &engine_count, &engine_);
    if (status == CUDNN_STATUS_SUCCESS) {
        cudnnBackendBehaviorNote_t notes[CUDNN_BEHAVIOR_NOTE_TYPE_COUNT];
        std::fill_n(notes, CUDNN_BEHAVIOR_NOTE_TYPE_COUNT, CUDNN_BEHAVIOR_NOTE_TYPE_COUNT);
        int64_t elem_count = 0;
        detail::get_attribute(engine->get_backend_descriptor(),
                              CUDNN_ATTR_ENGINE_BEHAVIOR_NOTE,
                              CUDNN_TYPE_BEHAVIOR_NOTE,
                              CUDNN_BEHAVIOR_NOTE_TYPE_COUNT,
                              &elem_count,
                              notes);
        if (std::any_of(
                notes, notes + elem_count, [](cudnnBackendBehaviorNote_t note) { return note == BEHAVIOR_NOTE; })) {
            hasBehavior = true;
        }
    }
    return hasBehavior;
}
}  // namespace cudnn_frontend
