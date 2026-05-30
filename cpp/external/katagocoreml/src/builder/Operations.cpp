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
                                       const std::vector<int64_t>& shape,
                                       bool is_fp32) {
    WeightEntry entry;
    entry.name = name;
    entry.data = data.data();
    entry.count = data.size();
    entry.shape = shape;
    entry.blob_offset = 0;  // Will be set during serialization
    entry.is_fp32 = is_fp32;
    m_weights.push_back(std::move(entry));
    return name;
}

std::string KataGoOps::registerOwnedWeight(const std::string& name,
                                            std::vector<float>&& data,
                                            const std::vector<int64_t>& shape) {
    m_owned.push_back(std::move(data));
    const std::vector<float>& stored = m_owned.back();
    WeightEntry entry;
    entry.name = name;
    entry.data = stored.data();
    entry.count = stored.size();
    entry.shape = shape;
    entry.blob_offset = 0;
    m_weights.push_back(std::move(entry));
    return name;
}

std::string KataGoOps::genOpName(const std::string& prefix) {
    return prefix + "_" + std::to_string(m_op_counter++);
}

}  // namespace katagocoreml
