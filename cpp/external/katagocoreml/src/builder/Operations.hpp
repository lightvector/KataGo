// katagocoreml - Standalone C++ KataGo to Core ML Converter
// Copyright (c) 2025, Chin-Chang Yang

#pragma once

#include "../types/KataGoTypes.hpp"
#include <cmath>
#include <string>
#include <vector>

namespace katagocoreml {

/// Weight entry for blob file storage
struct WeightEntry {
    std::string name;
    std::vector<float> data;
    std::vector<int64_t> shape;
    uint64_t blob_offset = 0;  // Set during serialization
};

/// Precomputed constants for identity mask optimization
struct MaskConstants {
    float mask_sum = 361.0f;  // 19 * 19
    float mask_sum_reciprocal = 1.0f / 361.0f;
    float mask_sum_sqrt_s14_m01 = 0.5f;  // (sqrt(361) - 14) * 0.1
    float mask_sum_sqrt_s14_m01_sq_s01 = 0.15f;  // (0.5^2) - 0.1

    MaskConstants() = default;

    MaskConstants(int board_x_size, int board_y_size) {
        mask_sum = static_cast<float>(board_x_size * board_y_size);
        mask_sum_reciprocal = 1.0f / mask_sum;
        float sqrt_mask_sum = std::sqrt(mask_sum);
        mask_sum_sqrt_s14_m01 = (sqrt_mask_sum - 14.0f) * 0.1f;
        float sq = mask_sum_sqrt_s14_m01 * mask_sum_sqrt_s14_m01;
        mask_sum_sqrt_s14_m01_sq_s01 = sq - 0.1f;
    }
};

/// KataGo operation builder for MIL program construction
/// This class builds the structure needed for MIL program generation
class KataGoOps {
public:
    KataGoOps(int board_x_size, int board_y_size, bool optimize_identity_mask);

    /// Get the board dimensions
    int getBoardXSize() const { return m_board_x_size; }
    int getBoardYSize() const { return m_board_y_size; }
    bool isOptimizeIdentityMask() const { return m_optimize_identity_mask; }

    /// Get precomputed mask constants
    const MaskConstants& getMaskConstants() const { return m_mask_constants; }

    /// Register a weight tensor and return its reference name
    std::string registerWeight(const std::string& name,
                               const std::vector<float>& data,
                               const std::vector<int64_t>& shape);

    /// Get all registered weights
    const std::vector<WeightEntry>& getWeights() const { return m_weights; }

    /// Clear all registered weights
    void clearWeights() { m_weights.clear(); }

    /// Generate unique operation name
    std::string genOpName(const std::string& prefix);

private:
    int m_board_x_size;
    int m_board_y_size;
    bool m_optimize_identity_mask;
    MaskConstants m_mask_constants;
    std::vector<WeightEntry> m_weights;
    int m_op_counter = 0;
};

}  // namespace katagocoreml
