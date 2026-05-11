// katagocoreml - Standalone C++ KataGo to Core ML Converter
// Copyright (c) 2025, Chin-Chang Yang

#pragma once

#include <memory>
#include <optional>
#include <string>
#include <variant>
#include <vector>

namespace katagocoreml {

// ============================================================================
// Activation Types
// ============================================================================

/// Activation function types used in KataGo models
enum class ActivationType : int {
    Identity = 0,
    ReLU = 1,
    Mish = 2
    // MISH_SCALE8 = 12 is internal optimization, treated as Mish
};

// ============================================================================
// Block Kind Constants
// ============================================================================

/// Block kind constants (matching KataGo's desc.h)
constexpr int ORDINARY_BLOCK_KIND = 0;
constexpr int GLOBAL_POOLING_BLOCK_KIND = 2;
constexpr int NESTED_BOTTLENECK_BLOCK_KIND = 3;

// ============================================================================
// Layer Descriptors
// ============================================================================

/// Convolutional layer descriptor
struct ConvLayerDesc {
    std::string name;
    int conv_y_size = 0;
    int conv_x_size = 0;
    int in_channels = 0;
    int out_channels = 0;
    int dilation_y = 1;
    int dilation_x = 1;
    std::vector<float> weights;  // Shape: [out_channels, in_channels, y, x] (OIHW)

    /// Get weight shape as vector
    std::vector<int64_t> getWeightShape() const {
        return {out_channels, in_channels, conv_y_size, conv_x_size};
    }
};

/// Batch normalization layer descriptor
/// KataGo pre-computes merged scale and bias for efficiency:
///   merged_scale = scale / sqrt(variance + epsilon)
///   merged_bias = bias - mean * merged_scale
struct BatchNormLayerDesc {
    std::string name;
    int num_channels = 0;
    float epsilon = 1e-5f;
    bool has_scale = true;
    bool has_bias = true;
    std::vector<float> mean;
    std::vector<float> variance;
    std::vector<float> scale;
    std::vector<float> bias;
    std::vector<float> merged_scale;  // Pre-computed
    std::vector<float> merged_bias;   // Pre-computed
};

/// Activation layer descriptor
struct ActivationLayerDesc {
    std::string name;
    ActivationType activation_type = ActivationType::ReLU;
};

/// Matrix multiplication (fully connected) layer descriptor
/// Computes: output = input @ weights
struct MatMulLayerDesc {
    std::string name;
    int in_channels = 0;
    int out_channels = 0;
    std::vector<float> weights;  // Shape: [in_channels, out_channels]

    std::vector<int64_t> getWeightShape() const {
        return {in_channels, out_channels};
    }
};

/// Bias addition layer descriptor
/// Computes: output = input + bias
struct MatBiasLayerDesc {
    std::string name;
    int num_channels = 0;
    std::vector<float> weights;  // Shape: [num_channels]
};

// ============================================================================
// Block Descriptors
// ============================================================================

/// Forward declarations for recursive block types
struct ResidualBlockDesc;
struct GlobalPoolingResidualBlockDesc;
struct NestedBottleneckResidualBlockDesc;

/// Block descriptor variant
using BlockDesc = std::variant<
    ResidualBlockDesc,
    GlobalPoolingResidualBlockDesc,
    NestedBottleneckResidualBlockDesc
>;

/// Block with its kind
struct BlockEntry {
    int block_kind = ORDINARY_BLOCK_KIND;
    std::shared_ptr<BlockDesc> block;
};

/// Standard residual block descriptor
/// Architecture:
///   input -> preBN -> preActivation -> regularConv ->
///            midBN -> midActivation -> finalConv -> + input
struct ResidualBlockDesc {
    std::string name;
    BatchNormLayerDesc pre_bn;
    ActivationLayerDesc pre_activation;
    ConvLayerDesc regular_conv;
    BatchNormLayerDesc mid_bn;
    ActivationLayerDesc mid_activation;
    ConvLayerDesc final_conv;
};

/// Global pooling residual block descriptor
/// Similar to ResidualBlock but includes a global pooling path
struct GlobalPoolingResidualBlockDesc {
    std::string name;
    int model_version = 0;
    BatchNormLayerDesc pre_bn;
    ActivationLayerDesc pre_activation;
    ConvLayerDesc regular_conv;
    ConvLayerDesc gpool_conv;
    BatchNormLayerDesc gpool_bn;
    ActivationLayerDesc gpool_activation;
    MatMulLayerDesc gpool_to_bias_mul;
    BatchNormLayerDesc mid_bn;
    ActivationLayerDesc mid_activation;
    ConvLayerDesc final_conv;
};

/// Nested bottleneck residual block descriptor
/// A bottleneck block that can contain other blocks inside it
struct NestedBottleneckResidualBlockDesc {
    std::string name;
    int num_blocks = 0;
    BatchNormLayerDesc pre_bn;
    ActivationLayerDesc pre_activation;
    ConvLayerDesc pre_conv;
    std::vector<BlockEntry> blocks;
    BatchNormLayerDesc post_bn;
    ActivationLayerDesc post_activation;
    ConvLayerDesc post_conv;
};

// ============================================================================
// SGF Metadata Encoder (v15+)
// ============================================================================

/// SGF metadata encoder descriptor (model version >= 15)
/// Encodes game metadata through a 3-layer MLP
struct SGFMetadataEncoderDesc {
    std::string name;
    int meta_encoder_version = 0;
    int num_input_meta_channels = 0;
    MatMulLayerDesc mul1;
    MatBiasLayerDesc bias1;
    ActivationLayerDesc act1;
    MatMulLayerDesc mul2;
    MatBiasLayerDesc bias2;
    ActivationLayerDesc act2;
    MatMulLayerDesc mul3;
};

// ============================================================================
// Network Component Descriptors
// ============================================================================

/// Trunk (backbone) network descriptor
struct TrunkDesc {
    std::string name;
    int model_version = 0;
    int num_blocks = 0;
    int trunk_num_channels = 0;
    int mid_num_channels = 0;
    int regular_num_channels = 0;
    int gpool_num_channels = 0;
    int meta_encoder_version = 0;
    ConvLayerDesc initial_conv;
    MatMulLayerDesc initial_matmul;
    std::optional<SGFMetadataEncoderDesc> sgf_metadata_encoder;
    std::vector<BlockEntry> blocks;
    BatchNormLayerDesc trunk_tip_bn;
    ActivationLayerDesc trunk_tip_activation;
};

/// Policy head descriptor
struct PolicyHeadDesc {
    std::string name;
    int model_version = 0;
    int policy_out_channels = 0;
    ConvLayerDesc p1_conv;
    ConvLayerDesc g1_conv;
    BatchNormLayerDesc g1_bn;
    ActivationLayerDesc g1_activation;
    MatMulLayerDesc gpool_to_bias_mul;
    BatchNormLayerDesc p1_bn;
    ActivationLayerDesc p1_activation;
    ConvLayerDesc p2_conv;
    MatMulLayerDesc gpool_to_pass_mul;
    std::optional<MatBiasLayerDesc> gpool_to_pass_bias;      // v15+
    std::optional<ActivationLayerDesc> pass_activation;      // v15+
    std::optional<MatMulLayerDesc> gpool_to_pass_mul2;       // v15+
};

/// Value head descriptor
struct ValueHeadDesc {
    std::string name;
    int model_version = 0;
    ConvLayerDesc v1_conv;
    BatchNormLayerDesc v1_bn;
    ActivationLayerDesc v1_activation;
    MatMulLayerDesc v2_mul;
    MatBiasLayerDesc v2_bias;
    ActivationLayerDesc v2_activation;
    MatMulLayerDesc v3_mul;
    MatBiasLayerDesc v3_bias;
    MatMulLayerDesc sv3_mul;
    MatBiasLayerDesc sv3_bias;
    ConvLayerDesc v_ownership_conv;
};

// ============================================================================
// Post-Processing Parameters
// ============================================================================

/// Post-processing parameters for model outputs (v13+)
struct ModelPostProcessParams {
    float td_score_multiplier = 20.0f;
    float score_mean_multiplier = 20.0f;
    float score_stdev_multiplier = 20.0f;
    float lead_multiplier = 20.0f;
    float variance_time_multiplier = 40.0f;
    float shortterm_value_error_multiplier = 0.25f;
    float shortterm_score_error_multiplier = 30.0f;
    float output_scale_multiplier = 1.0f;
};

// ============================================================================
// Complete Model Descriptor
// ============================================================================

/// Complete KataGo model descriptor
struct KataGoModelDesc {
    std::string name;
    std::string sha256;
    int model_version = 0;
    int num_input_channels = 0;
    int num_input_global_channels = 0;
    int num_input_meta_channels = 0;
    int num_policy_channels = 0;
    int num_value_channels = 3;  // Always 3: win/loss/noresult
    int num_score_value_channels = 0;
    int num_ownership_channels = 1;  // Always 1
    int meta_encoder_version = 0;
    ModelPostProcessParams post_process_params;
    TrunkDesc trunk;
    PolicyHeadDesc policy_head;
    ValueHeadDesc value_head;

    /// Get number of policy channels based on model version
    static int getPolicyChannels(int version) {
        if (version >= 16) return 4;
        if (version >= 12) return 2;
        return 1;
    }

    /// Get number of score value channels based on model version
    static int getScoreValueChannels(int version) {
        if (version >= 9) return 6;
        return 4;
    }
};

}  // namespace katagocoreml
