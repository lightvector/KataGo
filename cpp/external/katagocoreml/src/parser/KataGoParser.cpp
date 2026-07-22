// katagocoreml - Standalone C++ KataGo to Core ML Converter
// Copyright (c) 2025, Chin-Chang Yang

#include "KataGoParser.hpp"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <zlib.h>

namespace katagocoreml {

namespace {
// Cross-field dimension consistency check, mirroring desc.cpp's pattern of named errors.
// Takes int64_t so callers can pass products like numChannels*3 without signed-int overflow UB.
void checkDimsEqual(const std::string& name, const char* aDesc, int64_t a, const char* bDesc, int64_t b) {
    if (a != b) {
        throw std::runtime_error(name + ": " + aDesc + " (" + std::to_string(a) + ") != " +
                                 bDesc + " (" + std::to_string(b) + ")");
    }
}
}  // namespace

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
    try {
        return std::stoi(token);
    } catch (const std::exception&) {
        throw std::runtime_error(
            "model file parse error: expected integer, got \"" + token + "\"" +
            (token.empty() ? " (unexpected end of file?)" : ""));
    }
}

float KataGoParser::readFloat() {
    std::string token = readString();
    try {
        return std::stof(token);
    } catch (const std::exception&) {
        throw std::runtime_error(
            "model file parse error: expected float, got \"" + token + "\"" +
            (token.empty() ? " (unexpected end of file?)" : ""));
    }
}

bool KataGoParser::readBool() {
    return readInt() != 0;
}

std::vector<float> KataGoParser::readFloats(size_t count, const std::string& name) {
    // Bound count by the remaining file size BEFORE allocating: text floats need at least
    // 2 bytes each and binary exactly 4, so a count exceeding the remaining byte count is
    // impossible. Without this, a crafted file (gzip compresses multi-GB of zeros to a few
    // MB) could trigger an enormous zero-initializing allocation before any read fails.
    if (count > m_buffer.size() - m_pos) {
        throw std::runtime_error(name + ": not enough bytes for " + std::to_string(count) + " floats");
    }
    std::vector<float> floats(count);

    if (!m_binary_floats) {
        // Text format
        for (size_t i = 0; i < count; i++) {
            floats[i] = readFloat();
        }
    } else {
        // Binary format - find @BIN@ marker. Bound the scan like desc.cpp (which allows at
        // most 100 chars before the marker): an unbounded scan would silently skip arbitrary
        // junk between weight blocks, masking corruption or resyncing past a real error.
        size_t scan_start = m_pos;
        while (m_pos < m_buffer.size()) {
            if (m_buffer[m_pos] == '@') {
                break;
            }
            if (m_pos - scan_start >= 100) {
                throw std::runtime_error(name + ": could not find @BIN@ marker near expected position."
                                         " Invalid model - perhaps a .txt model with a binary header, or corrupted data?");
            }
            m_pos++;
        }

        // Check for @BIN@ header
        if (m_pos + 5 > m_buffer.size() ||
            std::memcmp(&m_buffer[m_pos], "@BIN@", 5) != 0) {
            throw std::runtime_error(name + ": expected @BIN@ marker for binary float block");
        }
        m_pos += 5;

        // Read binary floats (little-endian). Compare count against the remaining bytes
        // directly so a huge count cannot overflow the num_bytes computation and slip
        // past this bounds check.
        size_t remaining = m_buffer.size() - m_pos;
        if (count > remaining / 4) {
            throw std::runtime_error(name + ": not enough bytes for " + std::to_string(count) + " floats");
        }
        size_t num_bytes = count * 4;

        // Copy as little-endian float32
        std::memcpy(floats.data(), &m_buffer[m_pos], num_bytes);
        m_pos += num_bytes;
    }

    // Reject NaN/Inf weights: corrupted or otherwise invalid models would
    // otherwise produce a structurally-valid CoreML file that yields garbage at
    // inference time. Matches the CHECKFINITE behavior of cpp/neuralnet/desc.cpp.
    for (size_t i = 0; i < floats.size(); i++) {
        if (!std::isfinite(floats[i])) {
            throw std::runtime_error(name + ": NaN or infinite neural net weight or parameter");
        }
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

    if (layer.dilation_y < 1 || layer.dilation_x < 1) {
        throw std::runtime_error(layer.name + ": dilation must be >= 1, got y=" +
                                 std::to_string(layer.dilation_y) + " x=" +
                                 std::to_string(layer.dilation_x));
    }
    if (layer.conv_y_size < 1 || layer.conv_x_size < 1 ||
        layer.in_channels < 1 || layer.out_channels < 1) {
        throw std::runtime_error(layer.name + ": invalid conv dimensions");
    }
    // KataGo training only emits odd filter sizes (1, 3, 5). An even size would
    // make "same"-padding ambiguous. Match master desc.cpp's validation.
    if ((layer.conv_y_size % 2) == 0 || (layer.conv_x_size % 2) == 0) {
        throw std::runtime_error(layer.name + ": convolution filter sizes must be odd, got y=" +
                                 std::to_string(layer.conv_y_size) + " x=" +
                                 std::to_string(layer.conv_x_size));
    }

    // Read weights in file order: [y, x, ic, oc]. Multiply with an overflow check: four
    // attacker-controlled 31-bit ints can wrap a 64-bit product, and a wrapped-small
    // num_weights would pass the readFloats bounds check while the transpose loops below
    // still iterate the full unwrapped ranges, indexing far out of bounds.
    auto checkedMul = [&layer](size_t a, size_t b) {
        if (b != 0 && a > std::numeric_limits<size_t>::max() / b) {
            throw std::runtime_error(layer.name + ": conv weight count overflows");
        }
        return a * b;
    };
    size_t num_weights = checkedMul(
        checkedMul(
            checkedMul(static_cast<size_t>(layer.conv_y_size), static_cast<size_t>(layer.conv_x_size)),
            static_cast<size_t>(layer.in_channels)),
        static_cast<size_t>(layer.out_channels));
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

    if (layer.num_channels < 1) {
        throw std::runtime_error(layer.name + ": batchnorm numChannels must be >= 1, got " +
                                 std::to_string(layer.num_channels));
    }
    // epsilon == 0 with variance == 0 would divide by zero in the merged-scale
    // formula below and produce Inf weights. Written NaN-safe and finite-checked
    // (stof accepts "nan"/"inf" tokens, and NaN passes a `<= 0` comparison).
    if (!(layer.epsilon > 0.0f) || !std::isfinite(layer.epsilon)) {
        throw std::runtime_error(layer.name + ": batchnorm epsilon must be positive and finite, got " +
                                 std::to_string(layer.epsilon));
    }

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
        } else if (activation_str == "ACTIVATION_SILU") {
            layer.activation_type = ActivationType::Silu;
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

    // Matches desc.cpp; a negative count would wrap the size_t product below.
    if (layer.in_channels < 1 || layer.out_channels < 1) {
        throw std::runtime_error(layer.name + ": matmul channels must be positive, got in=" +
                                 std::to_string(layer.in_channels) + " out=" +
                                 std::to_string(layer.out_channels));
    }

    // Weights in [ic, oc] order
    size_t num_weights = static_cast<size_t>(layer.in_channels) * layer.out_channels;
    layer.weights = readFloats(num_weights, layer.name);

    return layer;
}

MatBiasLayerDesc KataGoParser::parseMatBiasLayer() {
    MatBiasLayerDesc layer;
    layer.name = readString();
    layer.num_channels = readInt();
    if (layer.num_channels < 1) {
        throw std::runtime_error(layer.name + ": matbias numChannels must be >= 1, got " +
                                 std::to_string(layer.num_channels));
    }
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

    // Internal cross-checks, matching desc.cpp.
    checkDimsEqual(block.name, "preBN.numChannels", block.pre_bn.num_channels,
                   "regularConv.inChannels", block.regular_conv.in_channels);
    checkDimsEqual(block.name, "midBN.numChannels", block.mid_bn.num_channels,
                   "regularConv.outChannels", block.regular_conv.out_channels);
    checkDimsEqual(block.name, "midBN.numChannels", block.mid_bn.num_channels,
                   "finalConv.inChannels", block.final_conv.in_channels);

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

    // Internal cross-checks, matching desc.cpp.
    checkDimsEqual(block.name, "preBN.numChannels", block.pre_bn.num_channels,
                   "regularConv.inChannels", block.regular_conv.in_channels);
    checkDimsEqual(block.name, "preBN.numChannels", block.pre_bn.num_channels,
                   "gpoolConv.inChannels", block.gpool_conv.in_channels);
    checkDimsEqual(block.name, "gpoolBN.numChannels", block.gpool_bn.num_channels,
                   "gpoolConv.outChannels", block.gpool_conv.out_channels);
    checkDimsEqual(block.name, "gpoolBN.numChannels * 3", (int64_t)block.gpool_bn.num_channels * 3,
                   "gpoolToBiasMul.inChannels", block.gpool_to_bias_mul.in_channels);
    checkDimsEqual(block.name, "midBN.numChannels", block.mid_bn.num_channels,
                   "regularConv.outChannels", block.regular_conv.out_channels);
    checkDimsEqual(block.name, "midBN.numChannels", block.mid_bn.num_channels,
                   "gpoolToBiasMul.outChannels", block.gpool_to_bias_mul.out_channels);
    checkDimsEqual(block.name, "midBN.numChannels", block.mid_bn.num_channels,
                   "finalConv.inChannels", block.final_conv.in_channels);

    return block;
}

NestedBottleneckResidualBlockDesc KataGoParser::parseNestedBottleneckBlock(int model_version, int trunk_num_channels) {
    NestedBottleneckResidualBlockDesc block;
    block.name = readString();
    block.num_blocks = readInt();
    if (block.num_blocks < 1) {
        throw std::runtime_error(block.name + ": nested bottleneck res block num blocks must be positive, got " +
                                 std::to_string(block.num_blocks));
    }

    block.pre_bn = parseBatchNormLayer();
    block.pre_activation = parseActivationLayer(model_version);
    block.pre_conv = parseConvLayer();

    block.blocks = parseBlockStack(model_version, block.num_blocks, block.pre_conv.out_channels);

    block.post_bn = parseBatchNormLayer();
    block.post_activation = parseActivationLayer(model_version);
    block.post_conv = parseConvLayer();

    // Internal cross-checks, matching desc.cpp.
    checkDimsEqual(block.name, "preBN.numChannels", block.pre_bn.num_channels,
                   "preConv.inChannels", block.pre_conv.in_channels);
    checkDimsEqual(block.name, "postBN.numChannels", block.post_bn.num_channels,
                   "preConv.outChannels", block.pre_conv.out_channels);
    checkDimsEqual(block.name, "postBN.numChannels", block.post_bn.num_channels,
                   "postConv.inChannels", block.post_conv.in_channels);

    return block;
}

// Validate that a block's input/output channel counts match the surrounding
// trunk. A mismatch would either fail loudly inside CoreML's compiler or, worse,
// produce a graph whose shapes accidentally line up but compute nonsense.
// Match master desc.cpp's parseResidualBlockStack() consistency checks.
static void checkBlockChannels(const std::string& block_name, const std::string& kind,
                               int pre_bn_channels, int output_channels,
                               int trunk_num_channels) {
    if (pre_bn_channels != trunk_num_channels) {
        throw std::runtime_error(block_name + " (" + kind + "): preBN.numChannels (" +
                                 std::to_string(pre_bn_channels) + ") != trunkNumChannels (" +
                                 std::to_string(trunk_num_channels) + ")");
    }
    if (output_channels != trunk_num_channels) {
        throw std::runtime_error(block_name + " (" + kind + "): block output channels (" +
                                 std::to_string(output_channels) + ") != trunkNumChannels (" +
                                 std::to_string(trunk_num_channels) + ")");
    }
}

TransformerRMSNormDesc KataGoParser::parseTransformerRMSNorm() {
    TransformerRMSNormDesc layer;
    layer.name = readString();
    layer.num_channels = readInt();
    layer.epsilon = readFloat();
    if (layer.num_channels < 1) {
        throw std::runtime_error(layer.name + ": transformer rmsnorm numChannels must be >= 1");
    }
    // Matches desc.cpp; epsilon <= 0 makes rsqrt(0+0)=inf at masked-off positions and NaN
    // then propagates silently through the whole net. Written NaN-safe (!(x > 0) catches NaN).
    if (!(layer.epsilon > 0.0f) || layer.epsilon > 1.0f) {
        throw std::runtime_error(layer.name + ": transformer rmsnorm epsilon is not positive or is too large");
    }
    layer.weight = readFloats(layer.num_channels, layer.name + "/weight");
    return layer;
}

RMSNormLayerDesc KataGoParser::parseRMSNormLayer() {
    RMSNormLayerDesc layer;
    layer.name = readString();
    layer.num_channels = readInt();
    layer.epsilon = readFloat();
    layer.spatial = (readInt() != 0);
    layer.cgroup_size = readInt();
    if (layer.num_channels < 1) {
        throw std::runtime_error(layer.name + ": rmsnorm numChannels must be >= 1");
    }
    if (layer.cgroup_size != 0) {
        throw std::runtime_error(layer.name + ": grouped spatial RMSNorm is not supported");
    }
    // Matches desc.cpp (see parseTransformerRMSNorm for rationale; NaN-safe form).
    if (!(layer.epsilon > 0.0f) || layer.epsilon > 1.0f) {
        throw std::runtime_error(layer.name + ": rmsnorm epsilon is not positive or is too large");
    }
    layer.gamma = readFloats(layer.num_channels, layer.name + "/gamma");
    layer.beta = readFloats(layer.num_channels, layer.name + "/beta");
    return layer;
}

TransformerAttentionBlockDesc KataGoParser::parseTransformerAttentionBlock(int model_version) {
    TransformerAttentionBlockDesc block;
    block.name = readString();
    block.num_heads = readInt();
    block.num_kv_heads = readInt();
    block.q_head_dim = readInt();
    block.v_head_dim = readInt();
    block.use_rope = (readInt() != 0);
    block.learnable_rope = (readInt() != 0);

    if (block.num_heads < 1 || block.num_kv_heads < 1 || (block.num_heads % block.num_kv_heads != 0)) {
        throw std::runtime_error(block.name + ": invalid numHeads/numKVHeads");
    }
    if (block.q_head_dim < 1 || block.v_head_dim < 1) {
        throw std::runtime_error(block.name + ": head dims must be positive");
    }
    if (block.use_rope && (block.q_head_dim % 2 != 0)) {
        throw std::runtime_error(block.name + ": qHeadDim must be even when RoPE is used");
    }

    block.pre_ln = parseTransformerRMSNorm();
    block.q_proj = parseMatMulLayer();
    block.k_proj = parseMatMulLayer();
    block.v_proj = parseMatMulLayer();
    block.out_proj = parseMatMulLayer();

    // Cross-check projection dims against the header, matching desc.cpp. The MIL builder
    // slices out_proj.weights assuming in_channels == numHeads*vHeadDim, so without the
    // out_proj check a malformed file causes a heap out-of-bounds read in the converter;
    // the others would only surface as cryptic CoreML shape errors.
    if (block.q_proj.out_channels != (int64_t)block.num_heads * block.q_head_dim) {
        throw std::runtime_error(block.name + ": qProj.outChannels (" + std::to_string(block.q_proj.out_channels) +
                                 ") != numHeads*qHeadDim (" + std::to_string((int64_t)block.num_heads * block.q_head_dim) + ")");
    }
    if (block.k_proj.out_channels != (int64_t)block.num_kv_heads * block.q_head_dim) {
        throw std::runtime_error(block.name + ": kProj.outChannels (" + std::to_string(block.k_proj.out_channels) +
                                 ") != numKVHeads*qHeadDim (" + std::to_string((int64_t)block.num_kv_heads * block.q_head_dim) + ")");
    }
    if (block.v_proj.out_channels != (int64_t)block.num_kv_heads * block.v_head_dim) {
        throw std::runtime_error(block.name + ": vProj.outChannels (" + std::to_string(block.v_proj.out_channels) +
                                 ") != numKVHeads*vHeadDim (" + std::to_string((int64_t)block.num_kv_heads * block.v_head_dim) + ")");
    }
    if (block.out_proj.in_channels != (int64_t)block.num_heads * block.v_head_dim) {
        throw std::runtime_error(block.name + ": outProj.inChannels (" + std::to_string(block.out_proj.in_channels) +
                                 ") != numHeads*vHeadDim (" + std::to_string((int64_t)block.num_heads * block.v_head_dim) + ")");
    }

    if (block.use_rope) {
        if (block.learnable_rope) {
            readString();  // ropeFreqs name
            block.rope_num_kv_heads = readInt();
            block.rope_num_pairs = readInt();
            int rope_dim2 = readInt();
            if (block.rope_num_kv_heads != block.num_kv_heads ||
                block.rope_num_pairs != block.q_head_dim / 2 || rope_dim2 != 2) {
                throw std::runtime_error(block.name + ": invalid learnable rope header");
            }
            block.rope_freqs = readFloats(
                static_cast<size_t>(block.rope_num_kv_heads) * block.rope_num_pairs * 2,
                block.name + "/rope_freqs");
        } else {
            readString();  // ropeTheta name
            block.rope_theta = readFloat();
            // Matches desc.cpp; theta <= 0 (or NaN/inf, which readFloat accepts for scalars)
            // would silently bake NaN/garbage RoPE tables into a structurally valid model.
            if (!(block.rope_theta > 0.0f) || !std::isfinite(block.rope_theta)) {
                throw std::runtime_error(block.name + ": rope theta must be positive and finite");
            }
        }
    }
    return block;
}

TransformerFFNBlockDesc KataGoParser::parseTransformerFFNBlock(int model_version) {
    TransformerFFNBlockDesc block;
    block.name = readString();
    block.num_channels = readInt();
    block.ffn_channels = readInt();
    block.use_swiglu = (readInt() != 0);
    if (block.num_channels < 1 || block.ffn_channels < 1) {
        throw std::runtime_error(block.name + ": transformer ffn channels must be positive");
    }
    block.pre_ln = parseTransformerRMSNorm();
    block.linear1 = parseMatMulLayer();
    if (block.use_swiglu) {
        block.linear_gate = parseMatMulLayer();
    }
    block.linear2 = parseMatMulLayer();

    // Cross-check layer dims against the header, matching desc.cpp; mismatches would
    // otherwise surface as cryptic CoreML shape errors at compile/load time.
    if (block.linear1.in_channels != block.num_channels || block.linear1.out_channels != block.ffn_channels) {
        throw std::runtime_error(block.name + ": linear1 dims (" + std::to_string(block.linear1.in_channels) +
                                 "->" + std::to_string(block.linear1.out_channels) + ") do not match numChannels->ffnChannels");
    }
    if (block.use_swiglu &&
        (block.linear_gate.in_channels != block.num_channels || block.linear_gate.out_channels != block.ffn_channels)) {
        throw std::runtime_error(block.name + ": linearGate dims (" + std::to_string(block.linear_gate.in_channels) +
                                 "->" + std::to_string(block.linear_gate.out_channels) + ") do not match numChannels->ffnChannels");
    }
    if (block.linear2.in_channels != block.ffn_channels || block.linear2.out_channels != block.num_channels) {
        throw std::runtime_error(block.name + ": linear2 dims (" + std::to_string(block.linear2.in_channels) +
                                 "->" + std::to_string(block.linear2.out_channels) + ") do not match ffnChannels->numChannels");
    }
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
            auto desc = parseResidualBlock(model_version);
            checkBlockChannels(desc.name, "ordinary_block",
                               desc.pre_bn.num_channels,
                               desc.final_conv.out_channels, trunk_num_channels);
            entry.block = std::make_shared<BlockDesc>(std::move(desc));
        } else if (block_kind_name == "gpool_block") {
            entry.block_kind = GLOBAL_POOLING_BLOCK_KIND;
            auto desc = parseGlobalPoolingResidualBlock(model_version);
            checkBlockChannels(desc.name, "gpool_block",
                               desc.pre_bn.num_channels,
                               desc.final_conv.out_channels, trunk_num_channels);
            entry.block = std::make_shared<BlockDesc>(std::move(desc));
        } else if (block_kind_name == "nested_bottleneck_block") {
            entry.block_kind = NESTED_BOTTLENECK_BLOCK_KIND;
            auto desc = parseNestedBottleneckBlock(model_version, trunk_num_channels);
            checkBlockChannels(desc.name, "nested_bottleneck_block",
                               desc.pre_bn.num_channels,
                               desc.post_conv.out_channels, trunk_num_channels);
            entry.block = std::make_shared<BlockDesc>(std::move(desc));
        } else if (block_kind_name == "transformer_attention_block") {
            entry.block_kind = TRANSFORMER_ATTENTION_BLOCK_KIND;
            auto desc = parseTransformerAttentionBlock(model_version);
            // Matches desc.cpp's trunk-channel cross-checks for transformer blocks.
            if (desc.q_proj.in_channels != trunk_num_channels) {
                throw std::runtime_error(desc.name + ": qProj.inChannels (" + std::to_string(desc.q_proj.in_channels) +
                                         ") != trunkNumChannels (" + std::to_string(trunk_num_channels) + ")");
            }
            if (desc.out_proj.out_channels != trunk_num_channels) {
                throw std::runtime_error(desc.name + ": outProj.outChannels (" + std::to_string(desc.out_proj.out_channels) +
                                         ") != trunkNumChannels (" + std::to_string(trunk_num_channels) + ")");
            }
            entry.block = std::make_shared<BlockDesc>(std::move(desc));
        } else if (block_kind_name == "transformer_ffn_block") {
            entry.block_kind = TRANSFORMER_FFN_BLOCK_KIND;
            auto desc = parseTransformerFFNBlock(model_version);
            if (desc.num_channels != trunk_num_channels) {
                throw std::runtime_error(desc.name + ": numChannels (" + std::to_string(desc.num_channels) +
                                         ") != trunkNumChannels (" + std::to_string(trunk_num_channels) + ")");
            }
            entry.block = std::make_shared<BlockDesc>(std::move(desc));
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

    // desc.cpp validates the declared count against NNModelVersion::getNumInputMetaChannels;
    // for the only supported metaEncoderVersion (1) that is 192, which parseModel also hardcodes.
    if (meta_encoder_version == 1 && encoder.num_input_meta_channels != 192) {
        throw std::runtime_error(encoder.name + ": numInputMetaChannels (" +
                                 std::to_string(encoder.num_input_meta_channels) +
                                 ") != 192 expected for metaEncoderVersion 1");
    }

    encoder.mul1 = parseMatMulLayer();
    encoder.bias1 = parseMatBiasLayer();
    encoder.act1 = parseActivationLayer(model_version);
    encoder.mul2 = parseMatMulLayer();
    encoder.bias2 = parseMatBiasLayer();
    encoder.act2 = parseActivationLayer(model_version);
    encoder.mul3 = parseMatMulLayer();

    // Internal cross-checks, matching desc.cpp.
    checkDimsEqual(encoder.name, "mul1.outChannels", encoder.mul1.out_channels,
                   "bias1.numChannels", encoder.bias1.num_channels);
    checkDimsEqual(encoder.name, "mul2.inChannels", encoder.mul2.in_channels,
                   "mul1.outChannels", encoder.mul1.out_channels);
    checkDimsEqual(encoder.name, "mul2.outChannels", encoder.mul2.out_channels,
                   "bias2.numChannels", encoder.bias2.num_channels);
    checkDimsEqual(encoder.name, "mul3.inChannels", encoder.mul3.in_channels,
                   "mul2.outChannels", encoder.mul2.out_channels);

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

    if (trunk.num_blocks < 1) {
        throw std::runtime_error(trunk.name + ": trunk numBlocks must be >= 1, got " +
                                 std::to_string(trunk.num_blocks));
    }
    if (trunk.trunk_num_channels <= 0 || trunk.mid_num_channels <= 0 ||
        trunk.regular_num_channels <= 0 || trunk.gpool_num_channels <= 0) {
        throw std::runtime_error(trunk.name + ": all trunk channel counts must be positive (trunk=" +
                                 std::to_string(trunk.trunk_num_channels) + ", mid=" +
                                 std::to_string(trunk.mid_num_channels) + ", regular=" +
                                 std::to_string(trunk.regular_num_channels) + ", gpool=" +
                                 std::to_string(trunk.gpool_num_channels) + ")");
    }

    // Version >= 15 writes the trunk norm kind followed by 5 unused int parameters.
    // Unlike upstream's CoreML parser (which rejects any non-standard norm), this fork
    // implements RMSNorm, so we capture the kind here instead of throwing. The 5 trailing
    // ints are reserved and still expected to be zero.
    if (model_version >= 15) {
        trunk.trunk_norm_kind = readInt();
        if (trunk.trunk_norm_kind != TRUNK_NORM_KIND_STANDARD &&
            trunk.trunk_norm_kind != TRUNK_NORM_KIND_RMSNORM) {
            throw std::runtime_error(trunk.name + ": unknown/unsupported trunk norm kind " +
                                     std::to_string(trunk.trunk_norm_kind));
        }
        for (int i = 0; i < 5; i++) {
            int unused = readInt();
            if (unused != 0) {
                throw std::runtime_error(trunk.name + ": unknown/unsupported trunk option " +
                                         std::string(1, static_cast<char>('B' + i)) + ": " +
                                         std::to_string(unused));
            }
        }
    }

    trunk.initial_conv = parseConvLayer();
    if (trunk.initial_conv.out_channels != trunk.trunk_num_channels) {
        throw std::runtime_error(trunk.name + ": initialConv.outChannels (" +
                                 std::to_string(trunk.initial_conv.out_channels) +
                                 ") != trunkNumChannels (" +
                                 std::to_string(trunk.trunk_num_channels) + ")");
    }
    trunk.initial_matmul = parseMatMulLayer();
    if (trunk.initial_matmul.out_channels != trunk.trunk_num_channels) {
        throw std::runtime_error(trunk.name + ": initialMatMul.outChannels (" +
                                 std::to_string(trunk.initial_matmul.out_channels) +
                                 ") != trunkNumChannels (" +
                                 std::to_string(trunk.trunk_num_channels) + ")");
    }

    // Parse SGF metadata encoder if present
    if (meta_encoder_version > 0) {
        auto enc = parseSGFMetadataEncoder(model_version, meta_encoder_version);
        if (enc.mul1.in_channels != enc.num_input_meta_channels) {
            throw std::runtime_error(enc.name + ": sgfMetadataEncoder.mul1.inChannels (" +
                                     std::to_string(enc.mul1.in_channels) +
                                     ") != numInputMetaChannels (" +
                                     std::to_string(enc.num_input_meta_channels) + ")");
        }
        if (enc.mul3.out_channels != trunk.trunk_num_channels) {
            throw std::runtime_error(enc.name + ": sgfMetadataEncoder.mul3.outChannels (" +
                                     std::to_string(enc.mul3.out_channels) +
                                     ") != trunkNumChannels (" +
                                     std::to_string(trunk.trunk_num_channels) + ")");
        }
        trunk.sgf_metadata_encoder = std::move(enc);
    }

    // Parse residual blocks
    trunk.blocks = parseBlockStack(model_version, trunk.num_blocks, trunk.trunk_num_channels);

    if (trunk.trunk_norm_kind == TRUNK_NORM_KIND_STANDARD) {
        trunk.trunk_tip_bn = parseBatchNormLayer();
        if (trunk.trunk_tip_bn.num_channels != trunk.trunk_num_channels) {
            throw std::runtime_error(trunk.name + ": trunkTipBN.numChannels (" +
                                     std::to_string(trunk.trunk_tip_bn.num_channels) +
                                     ") != trunkNumChannels (" +
                                     std::to_string(trunk.trunk_num_channels) + ")");
        }
    } else {
        trunk.trunk_tip_rms_norm = parseRMSNormLayer();
        if (trunk.trunk_tip_rms_norm.num_channels != trunk.trunk_num_channels) {
            throw std::runtime_error(trunk.name + ": trunkTipRMSNorm.numChannels (" +
                                     std::to_string(trunk.trunk_tip_rms_norm.num_channels) +
                                     ") != trunkNumChannels (" +
                                     std::to_string(trunk.trunk_num_channels) + ")");
        }
    }
    trunk.trunk_tip_activation = parseActivationLayer(model_version);

    return trunk;
}

PolicyHeadDesc KataGoParser::parsePolicyHead(int model_version) {
    PolicyHeadDesc head;
    head.name = readString();
    head.model_version = model_version;

    // Version >= 17 writes policyOutChannels (2 or 4) explicitly, followed by 3 unused int parameters.
    // For version < 17 the channel count is implied by the version (handled below).
    int policy_out_channels_v17 = 0;
    if (model_version >= 17) {
        policy_out_channels_v17 = readInt();
        if (policy_out_channels_v17 != 2 && policy_out_channels_v17 != 4) {
            throw std::runtime_error(head.name + ": invalid policyOutChannels " +
                                     std::to_string(policy_out_channels_v17) +
                                     " (expected 2 or 4)");
        }
        for (int i = 0; i < 3; i++) {
            int unused = readInt();
            if (unused != 0) {
                throw std::runtime_error(head.name + ": unknown/unsupported policy option " +
                                         std::string(1, static_cast<char>('A' + i)) + ": " +
                                         std::to_string(unused));
            }
        }
    }

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
    if (model_version >= 17) {
        head.policy_out_channels = policy_out_channels_v17;  // read explicitly above
    } else if (model_version >= 16) {
        head.policy_out_channels = 4;
    } else if (model_version >= 12) {
        head.policy_out_channels = 2;
    } else {
        head.policy_out_channels = 1;
    }

    // Internal cross-checks, matching desc.cpp. In particular p2Conv.outChannels must equal
    // policy_out_channels: parseModel reports num_policy_channels from the latter while the MIL
    // graph emits p2Conv's actual width, so a mismatch would mean the consumer reads the policy
    // buffer with the wrong channel count (silent wrong output) rather than a load error.
    checkDimsEqual(head.name, "p1Conv.outChannels", head.p1_conv.out_channels,
                   "p1BN.numChannels", head.p1_bn.num_channels);
    checkDimsEqual(head.name, "g1Conv.outChannels", head.g1_conv.out_channels,
                   "g1BN.numChannels", head.g1_bn.num_channels);
    checkDimsEqual(head.name, "gpoolToBiasMul.inChannels", head.gpool_to_bias_mul.in_channels,
                   "g1BN.numChannels * 3", (int64_t)head.g1_bn.num_channels * 3);
    checkDimsEqual(head.name, "gpoolToBiasMul.outChannels", head.gpool_to_bias_mul.out_channels,
                   "p1BN.numChannels", head.p1_bn.num_channels);
    checkDimsEqual(head.name, "p2Conv.inChannels", head.p2_conv.in_channels,
                   "p1BN.numChannels", head.p1_bn.num_channels);
    checkDimsEqual(head.name, "gpoolToPassMul.inChannels", head.gpool_to_pass_mul.in_channels,
                   "g1BN.numChannels * 3", (int64_t)head.g1_bn.num_channels * 3);
    checkDimsEqual(head.name, "p2Conv.outChannels", head.p2_conv.out_channels,
                   "policyOutChannels", head.policy_out_channels);
    if (model_version >= 15) {
        checkDimsEqual(head.name, "gpoolToPassMul.outChannels", head.gpool_to_pass_mul.out_channels,
                       "gpoolToPassBias.numChannels", head.gpool_to_pass_bias->num_channels);
        checkDimsEqual(head.name, "gpoolToPassMul.outChannels", head.gpool_to_pass_mul.out_channels,
                       "gpoolToPassMul2.inChannels", head.gpool_to_pass_mul2->in_channels);
        checkDimsEqual(head.name, "gpoolToPassMul.outChannels", head.gpool_to_pass_mul.out_channels,
                       "p1Conv.outChannels", head.p1_conv.out_channels);
        checkDimsEqual(head.name, "gpoolToPassMul2.outChannels", head.gpool_to_pass_mul2->out_channels,
                       "policyOutChannels", head.policy_out_channels);
    } else {
        checkDimsEqual(head.name, "gpoolToPassMul.outChannels", head.gpool_to_pass_mul.out_channels,
                       "policyOutChannels", head.policy_out_channels);
    }

    return head;
}

ValueHeadDesc KataGoParser::parseValueHead(int model_version) {
    ValueHeadDesc head;
    head.name = readString();
    head.model_version = model_version;

    // Version >= 17 writes 3 unused int parameters reserved for future features.
    if (model_version >= 17) {
        for (int i = 0; i < 3; i++) {
            int unused = readInt();
            if (unused != 0) {
                throw std::runtime_error(head.name + ": unknown/unsupported value option " +
                                         std::string(1, static_cast<char>('A' + i)) + ": " +
                                         std::to_string(unused));
            }
        }
    }

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

    // Internal cross-checks, matching desc.cpp (only v8+ branches: older versions are
    // rejected by isVersionSupported). v3Mul/sv3Mul/vOwnershipConv widths must match the
    // counts parseModel hardcodes, since the MIL graph emits the actual layer widths.
    checkDimsEqual(head.name, "v1Conv.outChannels", head.v1_conv.out_channels,
                   "v1BN.numChannels", head.v1_bn.num_channels);
    checkDimsEqual(head.name, "v2Mul.inChannels", head.v2_mul.in_channels,
                   "v1BN.numChannels * 3", (int64_t)head.v1_bn.num_channels * 3);
    checkDimsEqual(head.name, "v2Mul.outChannels", head.v2_mul.out_channels,
                   "v2Bias.numChannels", head.v2_bias.num_channels);
    checkDimsEqual(head.name, "v2Mul.outChannels", head.v2_mul.out_channels,
                   "v3Mul.inChannels", head.v3_mul.in_channels);
    checkDimsEqual(head.name, "v3Mul.outChannels", head.v3_mul.out_channels, "3", 3);
    checkDimsEqual(head.name, "v3Bias.numChannels", head.v3_bias.num_channels, "3", 3);
    checkDimsEqual(head.name, "sv3Mul.inChannels", head.sv3_mul.in_channels,
                   "v2Mul.outChannels", head.v2_mul.out_channels);
    const int expected_sv = (model_version >= 9) ? 6 : 4;
    checkDimsEqual(head.name, "sv3Mul.outChannels", head.sv3_mul.out_channels,
                   "expected scoreValueChannels", expected_sv);
    checkDimsEqual(head.name, "sv3Bias.numChannels", head.sv3_bias.num_channels,
                   "expected scoreValueChannels", expected_sv);
    checkDimsEqual(head.name, "vOwnershipConv.inChannels", head.v_ownership_conv.in_channels,
                   "v1Conv.outChannels", head.v1_conv.out_channels);
    checkDimsEqual(head.name, "vOwnershipConv.outChannels", head.v_ownership_conv.out_channels, "1", 1);

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
        // Note: the main engine (desc.cpp) accepts versions 3+, so a pre-v8 model that loads
        // fine on other backends (including Metal's MPSGraph GPU path) still lands here on the
        // CoreML/ANE path; say so rather than leaving the user to guess.
        throw std::runtime_error(
            "Only KataGo model versions 8-17 are supported by the CoreML/ANE converter, got version " +
            std::to_string(model.model_version) +
            (model.model_version < 8
                 ? ". This older model can still be used with the GPU (MPSGraph) path or other backends."
                 : ". You may need a newer KataGo version."));
    }

    model.num_input_channels = readInt();
    model.num_input_global_channels = readInt();
    if (model.num_input_channels <= 0) {
        throw std::runtime_error(model.name + ": numInputChannels must be > 0, got " +
                                 std::to_string(model.num_input_channels));
    }
    if (model.num_input_global_channels <= 0) {
        throw std::runtime_error(model.name + ": numInputGlobalChannels must be > 0, got " +
                                 std::to_string(model.num_input_global_channels));
    }

    // Parse post-process params (version >= 13).
    // Match master desc.cpp: each multiplier must be positive.
    if (model.model_version >= 13) {
        auto& p = model.post_process_params;
        p.td_score_multiplier = readFloat();
        p.score_mean_multiplier = readFloat();
        p.score_stdev_multiplier = readFloat();
        p.lead_multiplier = readFloat();
        p.variance_time_multiplier = readFloat();
        p.shortterm_value_error_multiplier = readFloat();
        p.shortterm_score_error_multiplier = readFloat();
        // NaN-safe and finite-checked: these multipliers scale engine outputs, so a NaN/Inf
        // here (which stof will happily parse) would silently poison every evaluation.
        auto checkPositive = [&](const char* field, float v) {
            if (!(v > 0.0f) || !std::isfinite(v)) {
                throw std::runtime_error(model.name + ": postProcessParams." + field +
                                         " must be positive and finite, got " + std::to_string(v));
            }
        };
        checkPositive("tdScoreMultiplier", p.td_score_multiplier);
        checkPositive("scoreMeanMultiplier", p.score_mean_multiplier);
        checkPositive("scoreStdevMultiplier", p.score_stdev_multiplier);
        checkPositive("leadMultiplier", p.lead_multiplier);
        checkPositive("varianceTimeMultiplier", p.variance_time_multiplier);
        checkPositive("shorttermValueErrorMultiplier", p.shortterm_value_error_multiplier);
        checkPositive("shorttermScoreErrorMultiplier", p.shortterm_score_error_multiplier);
    }

    // Parse meta encoder version (version >= 15)
    model.meta_encoder_version = 0;
    model.num_input_meta_channels = 0;
    if (model.model_version >= 15) {
        model.meta_encoder_version = readInt();
        // Forward-compat gate: master rejects values outside [0, 1] with a
        // "you may need a newer KataGo version" message. Without this, a
        // future model with metaEncoderVersion=2 would parse silently and
        // get hardcoded the wrong num_input_meta_channels below.
        if (model.meta_encoder_version < 0) {
            throw std::runtime_error(model.name + ": metaEncoderVersion unexpected value: " +
                                     std::to_string(model.meta_encoder_version));
        }
        if (model.meta_encoder_version > 1) {
            throw std::runtime_error(model.name + ": metaEncoderVersion " +
                                     std::to_string(model.meta_encoder_version) +
                                     " not implemented; you may need a newer KataGo version");
        }
        // Read reserved params (model options B..H). desc.cpp rejects any nonzero value so
        // future format features fail loudly instead of silently producing wrong output;
        // mirror that here (the trunk/policy/value reserved ints below are already checked).
        for (int i = 0; i < 7; i++) {
            int unused = readInt();
            if (unused != 0) {
                throw std::runtime_error(model.name + ": unknown/unsupported model option " +
                                         std::string(1, static_cast<char>('B' + i)) + ": " +
                                         std::to_string(unused));
            }
        }

        if (model.meta_encoder_version > 0) {
            model.num_input_meta_channels = 192;  // SGFMetadata::METADATA_INPUT_NUM_CHANNELS
        }
    }

    // Parse trunk, policy head, value head
    model.trunk = parseTrunk(model.model_version, model.meta_encoder_version);
    model.policy_head = parsePolicyHead(model.model_version);
    model.value_head = parseValueHead(model.model_version);

    // Top-level input-channel cross-checks against the trunk's first layers.
    // Match master desc.cpp: catches mismatches between header-declared input
    // channel counts and what the trunk actually expects.
    if (model.num_input_channels != model.trunk.initial_conv.in_channels) {
        throw std::runtime_error(model.name + ": numInputChannels (" +
                                 std::to_string(model.num_input_channels) +
                                 ") != trunk.initialConv.inChannels (" +
                                 std::to_string(model.trunk.initial_conv.in_channels) + ")");
    }
    if (model.num_input_global_channels != model.trunk.initial_matmul.in_channels) {
        throw std::runtime_error(model.name + ": numInputGlobalChannels (" +
                                 std::to_string(model.num_input_global_channels) +
                                 ") != trunk.initialMatMul.inChannels (" +
                                 std::to_string(model.trunk.initial_matmul.in_channels) + ")");
    }

    // Head-input-vs-trunk cross-checks, matching desc.cpp.
    checkDimsEqual(model.name, "policyHead.p1Conv.inChannels", model.policy_head.p1_conv.in_channels,
                   "trunkNumChannels", model.trunk.trunk_num_channels);
    checkDimsEqual(model.name, "policyHead.g1Conv.inChannels", model.policy_head.g1_conv.in_channels,
                   "trunkNumChannels", model.trunk.trunk_num_channels);
    checkDimsEqual(model.name, "valueHead.v1Conv.inChannels", model.value_head.v1_conv.in_channels,
                   "trunkNumChannels", model.trunk.trunk_num_channels);

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
