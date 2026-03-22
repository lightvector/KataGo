// katagocoreml - Standalone C++ KataGo to Core ML Converter
// Copyright (c) 2025, Chin-Chang Yang

#pragma once

#include <string>

namespace katagocoreml {

/// Conversion options for KataGo to Core ML conversion
struct ConversionOptions {
    /// Board width (default: 19)
    int board_x_size = 19;

    /// Board height (default: 19)
    int board_y_size = 19;

    /// Optimize for full board (skip mask operations)
    /// Provides ~6.5% inference speedup but requires all positions valid
    bool optimize_identity_mask = false;

    /// Compute precision: "FLOAT32" or "FLOAT16"
    std::string compute_precision = "FLOAT32";

    /// Use FLOAT16 for model inputs/outputs (instead of FLOAT32)
    /// Only effective when compute_precision="FLOAT16"
    /// When true with compute_precision="FLOAT16", creates pure FP16 model
    /// When false (default), uses FP32 I/O with FP16 internal computation
    /// Has no effect when compute_precision="FLOAT32"
    bool use_fp16_io = false;

    /// Core ML specification version (default: 6 for iOS 15+)
    int specification_version = 6;

    /// KataGo model version (set internally during conversion)
    int model_version = 0;

    /// Metadata encoder version (0 = no encoder, >0 = has encoder)
    int meta_encoder_version = 0;

    /// Number of metadata input channels (192 for human SL networks)
    int num_input_meta_channels = 0;

    /// Number of spatial input channels (set internally, typically 22)
    int num_input_channels = 0;

    /// Number of global input channels (set internally, typically 19)
    int num_input_global_channels = 0;

    /// Minimum batch size for inference (must be >= 1)
    /// Default: 1 (single sample inference)
    int min_batch_size = 1;

    /// Maximum batch size for inference
    /// If equal to min_batch_size, uses fixed batch size
    /// If greater than min_batch_size, enables dynamic batch support
    /// If <= 0, allows unlimited batch size (unbounded)
    /// Default: 1 (fixed single batch, backward compatible)
    int max_batch_size = 1;

    /// Author name (who ran the converter) - optional, set via CLI
    std::string author;

    /// License for the model - optional, set via CLI
    /// Typical values: "MIT", "CC0", "BSD-3-Clause"
    std::string license;

    /// Source KataGo model filename (set internally)
    std::string source_filename;

    /// Number of residual blocks (set internally)
    int num_blocks = 0;

    /// Trunk channel width (set internally)
    int trunk_channels = 0;

    /// Model name from KataGo binary (set internally)
    std::string model_name;

    /// Check if dynamic batch is enabled
    /// Dynamic batch allows variable batch sizes at runtime
    bool isDynamicBatch() const {
        return min_batch_size != max_batch_size || max_batch_size <= 0;
    }
};

/// Information about a KataGo model (without full conversion)
struct ModelInfo {
    /// Model name from file header
    std::string name;

    /// KataGo model version (8-16)
    int version = 0;

    /// Number of spatial input channels (typically 22)
    int num_input_channels = 0;

    /// Number of global input channels (typically 19)
    int num_input_global_channels = 0;

    /// Number of residual blocks
    int num_blocks = 0;

    /// Trunk channel width
    int trunk_channels = 0;

    /// Whether model has SGF metadata encoder (human SL networks)
    bool has_metadata_encoder = false;

    /// Number of policy output channels (1, 2, or 4 depending on version)
    int num_policy_channels = 0;

    /// Number of score value channels (4 or 6 depending on version)
    int num_score_value_channels = 0;
};

}  // namespace katagocoreml
