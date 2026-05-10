// katagocoreml - Standalone C++ KataGo to Core ML Converter
// Copyright (c) 2025, Chin-Chang Yang

#pragma once

#include "../builder/MILBuilder.hpp"
#include "katagocoreml/Options.hpp"
#include "Model.pb.h"
#include <memory>
#include <string>

namespace katagocoreml {

/// Serializes MIL program to Core ML .mlpackage format
class CoreMLSerializer {
public:
    /// Constructor
    /// @param spec_version Core ML specification version (default: 6 for iOS 15+)
    explicit CoreMLSerializer(int spec_version = 6);

    /// Serialize MIL program to .mlpackage
    /// @param program The MIL program protobuf
    /// @param weights Weight entries for blob serialization
    /// @param output_path Path for .mlpackage directory
    /// @param options Conversion options for metadata
    void serialize(CoreML::Specification::MILSpec::Program* program,
                   std::vector<WeightEntry>& weights,
                   const std::string& output_path,
                   const ConversionOptions& options);

private:
    int m_spec_version;

    /// Create the top-level Model protobuf wrapping the MIL program
    std::unique_ptr<CoreML::Specification::Model> createModelSpec(
        CoreML::Specification::MILSpec::Program* program,
        const ConversionOptions& options);

    /// Write weight blob file
    void writeWeightBlob(const std::string& weights_dir,
                         std::vector<WeightEntry>& weights,
                         bool use_fp16);

    /// Create .mlpackage directory structure
    void createPackage(const std::string& output_path,
                       CoreML::Specification::Model* model,
                       const std::string& weights_dir);

    /// Update blob offsets in MIL program after weights are serialized
    void updateBlobOffsets(CoreML::Specification::MILSpec::Program* program,
                          const std::vector<WeightEntry>& weights);
};

}  // namespace katagocoreml
