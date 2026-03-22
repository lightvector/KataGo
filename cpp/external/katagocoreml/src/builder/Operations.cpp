// katagocoreml - Standalone C++ KataGo to Core ML Converter
// Copyright (c) 2025, Chin-Chang Yang

#include "Operations.hpp"

namespace katagocoreml {

KataGoOps::KataGoOps(int board_x_size, int board_y_size, bool optimize_identity_mask)
    : m_board_x_size(board_x_size)
    , m_board_y_size(board_y_size)
    , m_optimize_identity_mask(optimize_identity_mask)
    , m_mask_constants(board_x_size, board_y_size)
    , m_op_counter(0) {}

std::string KataGoOps::registerWeight(const std::string& name,
                                       const std::vector<float>& data,
                                       const std::vector<int64_t>& shape) {
    WeightEntry entry;
    entry.name = name;
    entry.data = data;
    entry.shape = shape;
    entry.blob_offset = 0;  // Will be set during serialization
    m_weights.push_back(std::move(entry));
    return name;
}

std::string KataGoOps::genOpName(const std::string& prefix) {
    return prefix + "_" + std::to_string(m_op_counter++);
}

}  // namespace katagocoreml
