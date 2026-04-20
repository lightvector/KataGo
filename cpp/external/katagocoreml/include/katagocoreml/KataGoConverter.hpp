// katagocoreml - Standalone C++ KataGo to Core ML Converter
// Copyright (c) 2025, Chin-Chang Yang

#pragma once

#include "katagocoreml/Options.hpp"
#include <string>

namespace katagocoreml {

/// Main converter class for KataGo to Core ML conversion
class KataGoConverter {
public:
    /// Supported KataGo model versions
    static constexpr int MIN_SUPPORTED_VERSION = 8;
    static constexpr int MAX_SUPPORTED_VERSION = 16;

    /// Convert KataGo model file to Core ML mlpackage
    ///
    /// @param input_path Path to .bin or .bin.gz KataGo model file
    /// @param output_path Path for output .mlpackage directory
    /// @param options Conversion options
    /// @throws std::runtime_error on conversion failure
    static void convert(
        const std::string& input_path,
        const std::string& output_path,
        const ConversionOptions& options = ConversionOptions{}
    );

    /// Get model information without full conversion
    ///
    /// @param input_path Path to .bin or .bin.gz KataGo model file
    /// @return ModelInfo structure with model metadata
    /// @throws std::runtime_error if file cannot be parsed
    static ModelInfo getModelInfo(const std::string& input_path);

    /// Check if a model version is supported
    ///
    /// @param version KataGo model version number
    /// @return true if version is supported
    static bool isVersionSupported(int version) {
        return version >= MIN_SUPPORTED_VERSION && version <= MAX_SUPPORTED_VERSION;
    }

    /// Get library version string
    static std::string getVersion() {
        return "1.1.0";
    }
};

}  // namespace katagocoreml
