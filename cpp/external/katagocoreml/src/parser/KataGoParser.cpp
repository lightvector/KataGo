// katagocoreml - Standalone C++ KataGo to Core ML Converter
// Copyright (c) 2025, Chin-Chang Yang

#include "KataGoParser.hpp"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <zlib.h>

namespace katagocoreml {

// ============================================================================
// Constructor
// ============================================================================

KataGoParser::KataGoParser(const std::string& model_path)
    : m_model_path(model_path) {}

// ============================================================================
// Version Support
// ============================================================================

bool KataGoParser::isVersionSupported(int version) {
    for (int v : SUPPORTED_VERSIONS) {
        if (v == version) return true;
    }
    return false;
}

// ============================================================================
// File Loading
// ============================================================================

void KataGoParser::loadFile() {
    // Check if gzip compressed
    bool is_gzip = false;
    if (m_model_path.size() >= 3) {
        std::string ext = m_model_path.substr(m_model_path.size() - 3);
        is_gzip = (ext == ".gz");
    }

    if (is_gzip) {
        // Read gzipped file
        gzFile gz = gzopen(m_model_path.c_str(), "rb");
        if (!gz) {
            throw std::runtime_error("Cannot open gzip file: " + m_model_path);
        }

        // Read in chunks
        m_buffer.clear();
        std::vector<uint8_t> chunk(1024 * 1024);  // 1MB chunks
        int bytes_read;
        while ((bytes_read = gzread(gz, chunk.data(), static_cast<unsigned>(chunk.size()))) > 0) {
            m_buffer.insert(m_buffer.end(), chunk.begin(), chunk.begin() + bytes_read);
        }

        if (bytes_read < 0) {
            int errnum;
            const char* errmsg = gzerror(gz, &errnum);
            gzclose(gz);
            throw std::runtime_error("Error reading gzip file: " + std::string(errmsg));
        }

        gzclose(gz);
    } else {
        // Read regular file
        std::ifstream file(m_model_path, std::ios::binary | std::ios::ate);
        if (!file) {
            throw std::runtime_error("Cannot open file: " + m_model_path);
        }

        std::streamsize size = file.tellg();
        file.seekg(0, std::ios::beg);

        m_buffer.resize(static_cast<size_t>(size));
        if (!file.read(reinterpret_cast<char*>(m_buffer.data()), size)) {
            throw std::runtime_error("Error reading file: " + m_model_path);
        }
    }
}

// ============================================================================
// Main Parse Function
// ============================================================================

KataGoModelDesc KataGoParser::parse() {
    loadFile();
    m_pos = 0;

    // Detect if binary format (check for @BIN@ marker)
    const std::string bin_marker = "@BIN@";
    auto it = std::search(m_buffer.begin(), m_buffer.end(),
                          bin_marker.begin(), bin_marker.end());
    m_binary_floats = (it != m_buffer.end());

    return parseModel();
}

// ============================================================================
// Low-Level Reading Functions
// ============================================================================

void KataGoParser::skipWhitespace() {
    while (m_pos < m_buffer.size()) {
        char c = static_cast<char>(m_buffer[m_pos]);
        if (c != ' ' && c != '\t' && c != '\n' && c != '\r') {
            break;
        }
        m_pos++;
    }
}

void KataGoParser::readUntilWhitespace(std::string& out) {
    out.clear();
    while (m_pos < m_buffer.size()) {
        char c = static_cast<char>(m_buffer[m_pos]);
        if (c == ' ' || c == '\t' || c == '\n' || c == '\r') {
            break;
        }
        out += c;
        m_pos++;
    }
}

std::string KataGoParser::readString() {
    skipWhitespace();
    std::string token;
    readUntilWhitespace(token);
    return token;
}

int KataGoParser::readInt() {
    std::string token = readString();
    return std::stoi(token);
}

float KataGoParser::readFloat() {
    std::string token = readString();
    return std::stof(token);
}

bool KataGoParser::readBool() {
    return readInt() != 0;
}

std::vector<float> KataGoParser::readFloats(size_t count, const std::string& name) {
    std::vector<float> floats(count);

    if (!m_binary_floats) {
        // Text format
        for (size_t i = 0; i < count; i++) {
            floats[i] = readFloat();
        }
    } else {
        // Binary format - find @BIN@ marker
        while (m_pos < m_buffer.size()) {
            if (m_buffer[m_pos] == '@') {
                break;
            }
            m_pos++;
        }

        // Check for @BIN@ header
        if (m_pos + 5 > m_buffer.size() ||
            std::memcmp(&m_buffer[m_pos], "@BIN@", 5) != 0) {
            throw std::runtime_error(name + ": expected @BIN@ marker for binary float block");
        }
        m_pos += 5;

        // Read binary floats (little-endian)
        size_t num_bytes = count * 4;
        if (m_pos + num_bytes > m_buffer.size()) {
            throw std::runtime_error(name + ": not enough bytes for " + std::to_string(count) + " floats");
        }

        // Copy as little-endian float32
        std::memcpy(floats.data(), &m_buffer[m_pos], num_bytes);
        m_pos += num_bytes;
    }

    return floats;
}

// ============================================================================
// Layer Parsing Functions
// ============================================================================

ConvLayerDesc KataGoParser::parseConvLayer() {
    ConvLayerDesc layer;
    layer.name = readString();
    layer.conv_y_size = readInt();
    layer.conv_x_size = readInt();
    layer.in_channels = readInt();
    layer.out_channels = readInt();
    layer.dilation_y = readInt();
    layer.dilation_x = readInt();

    // Read weights in file order: [y, x, ic, oc]
    size_t num_weights = static_cast<size_t>(layer.conv_y_size) * layer.conv_x_size *
                         layer.in_channels * layer.out_channels;
    std::vector<float> weights_flat = readFloats(num_weights, layer.name);

    // Transpose from [y, x, ic, oc] to [oc, ic, y, x]
    layer.weights.resize(num_weights);
    int y_size = layer.conv_y_size;
    int x_size = layer.conv_x_size;
    int ic = layer.in_channels;
    int oc = layer.out_channels;

    for (int out_c = 0; out_c < oc; out_c++) {
        for (int in_c = 0; in_c < ic; in_c++) {
            for (int y = 0; y < y_size; y++) {
                for (int x = 0; x < x_size; x++) {
                    // Source index: [y, x, ic, oc]
                    size_t src_idx = static_cast<size_t>(y) * x_size * ic * oc +
                                     x * ic * oc +
                                     in_c * oc +
                                     out_c;
                    // Dest index: [oc, ic, y, x]
                    size_t dst_idx = static_cast<size_t>(out_c) * ic * y_size * x_size +
                                     in_c * y_size * x_size +
                                     y * x_size +
                                     x;
                    layer.weights[dst_idx] = weights_flat[src_idx];
                }
            }
        }
    }

    return layer;
}

BatchNormLayerDesc KataGoParser::parseBatchNormLayer() {
    BatchNormLayerDesc layer;
    layer.name = readString();
    layer.num_channels = readInt();
    layer.epsilon = readFloat();
    layer.has_scale = readBool();
    layer.has_bias = readBool();

    layer.mean = readFloats(layer.num_channels, layer.name + "/mean");
    layer.variance = readFloats(layer.num_channels, layer.name + "/variance");

    if (layer.has_scale) {
        layer.scale = readFloats(layer.num_channels, layer.name + "/scale");
    } else {
        layer.scale.resize(layer.num_channels, 1.0f);
    }

    if (layer.has_bias) {
        layer.bias = readFloats(layer.num_channels, layer.name + "/bias");
    } else {
        layer.bias.resize(layer.num_channels, 0.0f);
    }

    // Compute merged scale and bias
    layer.merged_scale.resize(layer.num_channels);
    layer.merged_bias.resize(layer.num_channels);
    for (int i = 0; i < layer.num_channels; i++) {
        layer.merged_scale[i] = layer.scale[i] / std::sqrt(layer.variance[i] + layer.epsilon);
        layer.merged_bias[i] = layer.bias[i] - layer.merged_scale[i] * layer.mean[i];
    }

    return layer;
}

ActivationLayerDesc KataGoParser::parseActivationLayer(int model_version) {
    ActivationLayerDesc layer;
    layer.name = readString();

    if (model_version >= 11) {
        std::string activation_str = readString();
        if (activation_str == "ACTIVATION_IDENTITY") {
            layer.activation_type = ActivationType::Identity;
        } else if (activation_str == "ACTIVATION_RELU") {
            layer.activation_type = ActivationType::ReLU;
        } else if (activation_str == "ACTIVATION_MISH") {
            layer.activation_type = ActivationType::Mish;
        } else {
            throw std::runtime_error("Unknown activation type: " + activation_str);
        }
    } else {
        // Pre-v11 models only have ReLU
        layer.activation_type = ActivationType::ReLU;
    }

    return layer;
}

MatMulLayerDesc KataGoParser::parseMatMulLayer() {
    MatMulLayerDesc layer;
    layer.name = readString();
    layer.in_channels = readInt();
    layer.out_channels = readInt();

    // Weights in [ic, oc] order
    size_t num_weights = static_cast<size_t>(layer.in_channels) * layer.out_channels;
    layer.weights = readFloats(num_weights, layer.name);

    return layer;
}

MatBiasLayerDesc KataGoParser::parseMatBiasLayer() {
    MatBiasLayerDesc layer;
    layer.name = readString();
    layer.num_channels = readInt();
    layer.weights = readFloats(layer.num_channels, layer.name);

    return layer;
}

// ============================================================================
// Block Parsing Functions
// ============================================================================

ResidualBlockDesc KataGoParser::parseResidualBlock(int model_version) {
    ResidualBlockDesc block;
    block.name = readString();
    block.pre_bn = parseBatchNormLayer();
    block.pre_activation = parseActivationLayer(model_version);
    block.regular_conv = parseConvLayer();
    block.mid_bn = parseBatchNormLayer();
    block.mid_activation = parseActivationLayer(model_version);
    block.final_conv = parseConvLayer();

    return block;
}

GlobalPoolingResidualBlockDesc KataGoParser::parseGlobalPoolingResidualBlock(int model_version) {
    GlobalPoolingResidualBlockDesc block;
    block.name = readString();
    block.model_version = model_version;
    block.pre_bn = parseBatchNormLayer();
    block.pre_activation = parseActivationLayer(model_version);
    block.regular_conv = parseConvLayer();
    block.gpool_conv = parseConvLayer();
    block.gpool_bn = parseBatchNormLayer();
    block.gpool_activation = parseActivationLayer(model_version);
    block.gpool_to_bias_mul = parseMatMulLayer();
    block.mid_bn = parseBatchNormLayer();
    block.mid_activation = parseActivationLayer(model_version);
    block.final_conv = parseConvLayer();

    return block;
}

NestedBottleneckResidualBlockDesc KataGoParser::parseNestedBottleneckBlock(int model_version, int trunk_num_channels) {
    NestedBottleneckResidualBlockDesc block;
    block.name = readString();
    block.num_blocks = readInt();

    block.pre_bn = parseBatchNormLayer();
    block.pre_activation = parseActivationLayer(model_version);
    block.pre_conv = parseConvLayer();

    block.blocks = parseBlockStack(model_version, block.num_blocks, block.pre_conv.out_channels);

    block.post_bn = parseBatchNormLayer();
    block.post_activation = parseActivationLayer(model_version);
    block.post_conv = parseConvLayer();

    return block;
}

std::vector<BlockEntry> KataGoParser::parseBlockStack(int model_version, int num_blocks, int trunk_num_channels) {
    std::vector<BlockEntry> blocks;
    blocks.reserve(num_blocks);

    for (int i = 0; i < num_blocks; i++) {
        std::string block_kind_name = readString();
        BlockEntry entry;

        if (block_kind_name == "ordinary_block") {
            entry.block_kind = ORDINARY_BLOCK_KIND;
            entry.block = std::make_shared<BlockDesc>(parseResidualBlock(model_version));
        } else if (block_kind_name == "gpool_block") {
            entry.block_kind = GLOBAL_POOLING_BLOCK_KIND;
            entry.block = std::make_shared<BlockDesc>(parseGlobalPoolingResidualBlock(model_version));
        } else if (block_kind_name == "nested_bottleneck_block") {
            entry.block_kind = NESTED_BOTTLENECK_BLOCK_KIND;
            entry.block = std::make_shared<BlockDesc>(parseNestedBottleneckBlock(model_version, trunk_num_channels));
        } else {
            throw std::runtime_error("Unknown block kind: " + block_kind_name);
        }

        blocks.push_back(std::move(entry));
    }

    return blocks;
}

// ============================================================================
// Component Parsing Functions
// ============================================================================

SGFMetadataEncoderDesc KataGoParser::parseSGFMetadataEncoder(int model_version, int meta_encoder_version) {
    SGFMetadataEncoderDesc encoder;
    encoder.name = readString();
    encoder.meta_encoder_version = meta_encoder_version;
    encoder.num_input_meta_channels = readInt();

    encoder.mul1 = parseMatMulLayer();
    encoder.bias1 = parseMatBiasLayer();
    encoder.act1 = parseActivationLayer(model_version);
    encoder.mul2 = parseMatMulLayer();
    encoder.bias2 = parseMatBiasLayer();
    encoder.act2 = parseActivationLayer(model_version);
    encoder.mul3 = parseMatMulLayer();

    return encoder;
}

TrunkDesc KataGoParser::parseTrunk(int model_version, int meta_encoder_version) {
    TrunkDesc trunk;
    trunk.name = readString();
    trunk.model_version = model_version;
    trunk.meta_encoder_version = meta_encoder_version;
    trunk.num_blocks = readInt();
    trunk.trunk_num_channels = readInt();
    trunk.mid_num_channels = readInt();
    trunk.regular_num_channels = readInt();
    readInt();  // dilatedNumChannels (unused)
    trunk.gpool_num_channels = readInt();

    // Version >= 15 has 6 unused int parameters
    if (model_version >= 15) {
        for (int i = 0; i < 6; i++) {
            readInt();
        }
    }

    trunk.initial_conv = parseConvLayer();
    trunk.initial_matmul = parseMatMulLayer();

    // Parse SGF metadata encoder if present
    if (meta_encoder_version > 0) {
        trunk.sgf_metadata_encoder = parseSGFMetadataEncoder(model_version, meta_encoder_version);
    }

    // Parse residual blocks
    trunk.blocks = parseBlockStack(model_version, trunk.num_blocks, trunk.trunk_num_channels);

    trunk.trunk_tip_bn = parseBatchNormLayer();
    trunk.trunk_tip_activation = parseActivationLayer(model_version);

    return trunk;
}

PolicyHeadDesc KataGoParser::parsePolicyHead(int model_version) {
    PolicyHeadDesc head;
    head.name = readString();
    head.model_version = model_version;

    head.p1_conv = parseConvLayer();
    head.g1_conv = parseConvLayer();
    head.g1_bn = parseBatchNormLayer();
    head.g1_activation = parseActivationLayer(model_version);
    head.gpool_to_bias_mul = parseMatMulLayer();
    head.p1_bn = parseBatchNormLayer();
    head.p1_activation = parseActivationLayer(model_version);
    head.p2_conv = parseConvLayer();
    head.gpool_to_pass_mul = parseMatMulLayer();

    // Version >= 15 has additional pass move layers
    if (model_version >= 15) {
        head.gpool_to_pass_bias = parseMatBiasLayer();
        head.pass_activation = parseActivationLayer(model_version);
        head.gpool_to_pass_mul2 = parseMatMulLayer();
    }

    // Determine policy output channels based on version
    if (model_version >= 16) {
        head.policy_out_channels = 4;
    } else if (model_version >= 12) {
        head.policy_out_channels = 2;
    } else {
        head.policy_out_channels = 1;
    }

    return head;
}

ValueHeadDesc KataGoParser::parseValueHead(int model_version) {
    ValueHeadDesc head;
    head.name = readString();
    head.model_version = model_version;

    head.v1_conv = parseConvLayer();
    head.v1_bn = parseBatchNormLayer();
    head.v1_activation = parseActivationLayer(model_version);
    head.v2_mul = parseMatMulLayer();
    head.v2_bias = parseMatBiasLayer();
    head.v2_activation = parseActivationLayer(model_version);
    head.v3_mul = parseMatMulLayer();
    head.v3_bias = parseMatBiasLayer();
    head.sv3_mul = parseMatMulLayer();
    head.sv3_bias = parseMatBiasLayer();
    head.v_ownership_conv = parseConvLayer();

    return head;
}

// ============================================================================
// Main Model Parsing
// ============================================================================

KataGoModelDesc KataGoParser::parseModel() {
    KataGoModelDesc model;

    // Read header
    model.name = readString();
    model.model_version = readInt();

    if (!isVersionSupported(model.model_version)) {
        throw std::runtime_error(
            "Only KataGo model versions 8-16 are supported, got version " +
            std::to_string(model.model_version));
    }

    model.num_input_channels = readInt();
    model.num_input_global_channels = readInt();

    // Parse post-process params (version >= 13)
    if (model.model_version >= 13) {
        model.post_process_params.td_score_multiplier = readFloat();
        model.post_process_params.score_mean_multiplier = readFloat();
        model.post_process_params.score_stdev_multiplier = readFloat();
        model.post_process_params.lead_multiplier = readFloat();
        model.post_process_params.variance_time_multiplier = readFloat();
        model.post_process_params.shortterm_value_error_multiplier = readFloat();
        model.post_process_params.shortterm_score_error_multiplier = readFloat();
    }

    // Parse meta encoder version (version >= 15)
    model.meta_encoder_version = 0;
    model.num_input_meta_channels = 0;
    if (model.model_version >= 15) {
        model.meta_encoder_version = readInt();
        // Read unused params
        for (int i = 0; i < 7; i++) {
            readInt();
        }

        if (model.meta_encoder_version > 0) {
            model.num_input_meta_channels = 192;  // SGFMetadata::METADATA_INPUT_NUM_CHANNELS
        }
    }

    // Parse trunk, policy head, value head
    model.trunk = parseTrunk(model.model_version, model.meta_encoder_version);
    model.policy_head = parsePolicyHead(model.model_version);
    model.value_head = parseValueHead(model.model_version);

    // Determine output channel counts
    model.num_policy_channels = model.policy_head.policy_out_channels;
    model.num_value_channels = 3;  // win, loss, noresult

    if (model.model_version >= 9) {
        model.num_score_value_channels = 6;
    } else if (model.model_version >= 8) {
        model.num_score_value_channels = 4;
    } else {
        model.num_score_value_channels = 1;
    }

    model.num_ownership_channels = 1;

    return model;
}

}  // namespace katagocoreml
