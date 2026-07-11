// katagocoreml - Standalone C++ KataGo to Core ML Converter
// Copyright (c) 2025, Chin-Chang Yang

#include "katagocoreml/KataGoConverter.hpp"
#include "parser/KataGoParser.hpp"
#include "builder/MILBuilder.hpp"
#include "serializer/CoreMLSerializer.hpp"
#include <stdexcept>
#include <filesystem>

namespace katagocoreml {

void KataGoConverter::convert(const std::string& input_path,
                               const std::string& output_path,
                               const ConversionOptions& options) {
    // Validate board sizes
    if (options.board_x_size < 2 || options.board_x_size > 37) {
        throw std::invalid_argument("board_x_size must be in range [2, 37]");
    }
    if (options.board_y_size < 2 || options.board_y_size > 37) {
        throw std::invalid_argument("board_y_size must be in range [2, 37]");
    }

    // Validate batch sizes
    if (options.min_batch_size < 1) {
        throw std::invalid_argument("min_batch_size must be at least 1");
    }
    if (options.max_batch_size > 0 && options.max_batch_size < options.min_batch_size) {
        throw std::invalid_argument("max_batch_size must be >= min_batch_size or <= 0 for unlimited");
    }

    // Parse KataGo model
    KataGoParser parser(input_path);
    KataGoModelDesc model = parser.parse();

    // Determine if using FP16 precision
    bool use_fp16 = (options.compute_precision == "FLOAT16");

    // Validate configuration: use_fp16_io requires FP16 compute
    if (options.use_fp16_io && !use_fp16) {
        throw std::invalid_argument("use_fp16_io requires compute_precision=\"FLOAT16\"");
    }

    // Build MIL program
    MILBuilder builder(model,
                       options.board_x_size,
                       options.board_y_size,
                       options.optimize_identity_mask,
                       use_fp16,
                       options.min_batch_size,
                       options.max_batch_size,
                       options.use_fp16_io);
    auto program = builder.build();

    // Get weights from builder
    auto weights = builder.getWeights();
    std::vector<WeightEntry> weights_copy(weights.begin(), weights.end());

    // Update options with model metadata for serialization
    ConversionOptions final_options = options;
    final_options.model_version = model.model_version;
    final_options.meta_encoder_version = model.meta_encoder_version;
    final_options.num_input_meta_channels = model.num_input_meta_channels;
    final_options.num_input_channels = model.num_input_channels;
    final_options.num_input_global_channels = model.num_input_global_channels;

    // Add model architecture info for metadata
    final_options.num_blocks = model.trunk.num_blocks;
    final_options.trunk_channels = model.trunk.trunk_num_channels;
    final_options.model_name = model.name;

    // Extract filename from input path
    if (final_options.source_filename.empty()) {
        std::filesystem::path p(input_path);
        final_options.source_filename = p.filename().string();
    }

    // FLOAT16 I/O requires specification version >= 7 (iOS 16+)
    if (final_options.use_fp16_io && final_options.specification_version < 7) {
        final_options.specification_version = 7;
    }

    // Serialize to .mlpackage
    CoreMLSerializer serializer(final_options.specification_version);
    serializer.serialize(program.get(), weights_copy, output_path, final_options);
}

ModelInfo KataGoConverter::getModelInfo(const std::string& input_path) {
    KataGoParser parser(input_path);
    KataGoModelDesc model = parser.parse();

    ModelInfo info;
    info.name = model.name;
    info.version = model.model_version;
    info.num_input_channels = model.num_input_channels;
    info.num_input_global_channels = model.num_input_global_channels;
    info.num_blocks = model.trunk.num_blocks;
    info.trunk_channels = model.trunk.trunk_num_channels;
    info.has_metadata_encoder = model.meta_encoder_version > 0;
    info.num_policy_channels = model.num_policy_channels;
    info.num_score_value_channels = model.num_score_value_channels;

    return info;
}

}  // namespace katagocoreml
