// katagocoreml - Standalone C++ KataGo to Core ML Converter
// Copyright (c) 2025, Chin-Chang Yang

#pragma once

#include "../types/KataGoTypes.hpp"
#include <cmath>
#include <deque>
#include <string>
#include <vector>

namespace katagocoreml {

/// Weight entry for blob file storage. `data`/`count` are a NON-OWNING view into
/// the live KataGoModelDesc (or into KataGoOps::m_owned for derived tensors).
struct WeightEntry {
    std::string name;
    const float* data = nullptr;
    size_t count = 0;
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

    /// Register a weight that lives in the model (stored as a non-owning view).
    std::string registerWeight(const std::string& name,
                               const std::vector<float>& data,
                               const std::vector<int64_t>& shape);

    /// Register a derived/temporary weight; KataGoOps takes ownership so the
    /// view stays valid through serialization.
    std::string registerOwnedWeight(const std::string& name,
                                    std::vector<float>&& data,
                                    const std::vector<int64_t>& shape);

    /// Get all registered weights (mutable; serialization sets blob_offset)
    std::vector<WeightEntry>& getWeightsMutable() { return m_weights; }

    /// Clear all registered weights (and their owned backing buffers)
    void clearWeights() { m_weights.clear(); m_owned.clear(); }

    /// Generate unique operation name
    std::string genOpName(const std::string& prefix);

private:
    int m_board_x_size;
    int m_board_y_size;
    bool m_optimize_identity_mask;
    MaskConstants m_mask_constants;
    std::vector<WeightEntry> m_weights;
    std::deque<std::vector<float>> m_owned;
    int m_op_counter = 0;
};

}  // namespace katagocoreml
