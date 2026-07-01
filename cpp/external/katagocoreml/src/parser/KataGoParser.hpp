// katagocoreml - Standalone C++ KataGo to Core ML Converter
// Copyright (c) 2025, Chin-Chang Yang

#pragma once

#include "../types/KataGoTypes.hpp"
#include <array>
#include <memory>
#include <string>
#include <type_traits>
#include <vector>
#include <zlib.h>

namespace katagocoreml {

/// Parser for KataGo neural network model files.
/// Supports versions 8-17 models in binary format (.bin, .bin.gz).
class KataGoParser {
public:
    /// Supported KataGo model versions
    static constexpr std::array<int, 10> SUPPORTED_VERSIONS = {8, 9, 10, 11, 12, 13, 14, 15, 16, 17};

    /// Constructor
    /// @param model_path Path to the KataGo model file (.bin or .bin.gz)
    explicit KataGoParser(const std::string& model_path);

    /// Parse the model file and return a structured model description
    /// @return KataGoModelDesc containing all model parameters
    /// @throws std::runtime_error if the file cannot be read or parsed
    KataGoModelDesc parse();

    /// Check if a version is supported
    static bool isVersionSupported(int version);

private:
    std::string m_model_path;
    // Custom-deleter unique_ptr owns the gzFile so it closes on every exit path
    // (normal return, exception, or bad_alloc) without manual try/catch.
    struct GzCloser {
        void operator()(gzFile f) const noexcept { if(f) gzclose(f); }
    };
    using GzHandle = std::unique_ptr<std::remove_pointer_t<gzFile>, GzCloser>;
    GzHandle m_gz;
    std::vector<uint8_t> m_refill;   // bounded refill buffer (~1 MB)
    size_t m_refillPos = 0;          // read cursor within m_refill
    size_t m_refillLen = 0;          // valid bytes in m_refill
    bool m_binary_floats = true;
    bool m_formatDetected = false;

    // Stream primitives
    bool refill();                   // returns false at EOF
    int  peekByte();                 // -1 at EOF
    void readExact(uint8_t* dst, size_t n, const std::string& name);

    // Low-level reading functions
    void readUntilWhitespace(std::string& out);
    void skipWhitespace();
    std::string readString();
    int readInt();
    float readFloat();
    bool readBool();
    std::vector<float> readFloats(size_t count, const std::string& name);

    // Layer parsing functions
    ConvLayerDesc parseConvLayer();
    BatchNormLayerDesc parseBatchNormLayer();
    ActivationLayerDesc parseActivationLayer(int model_version);
    MatMulLayerDesc parseMatMulLayer();
    MatBiasLayerDesc parseMatBiasLayer();
    TransformerRMSNormDesc parseTransformerRMSNorm();
    RMSNormLayerDesc parseRMSNormLayer();

    // Block parsing functions
    ResidualBlockDesc parseResidualBlock(int model_version);
    GlobalPoolingResidualBlockDesc parseGlobalPoolingResidualBlock(int model_version);
    NestedBottleneckResidualBlockDesc parseNestedBottleneckBlock(int model_version, int trunk_num_channels);
    TransformerAttentionBlockDesc parseTransformerAttentionBlock(int model_version, int trunk_num_channels);
    TransformerFFNBlockDesc parseTransformerFFNBlock(int model_version, int trunk_num_channels);
    std::vector<BlockEntry> parseBlockStack(int model_version, int num_blocks, int trunk_num_channels);

    // Component parsing functions
    SGFMetadataEncoderDesc parseSGFMetadataEncoder(int model_version, int meta_encoder_version);
    TrunkDesc parseTrunk(int model_version, int meta_encoder_version);
    PolicyHeadDesc parsePolicyHead(int model_version);
    ValueHeadDesc parseValueHead(int model_version);

    // Main model parsing
    KataGoModelDesc parseModel();
};

}  // namespace katagocoreml
