// katagocoreml - Standalone C++ KataGo to Core ML Converter
// Copyright (c) 2025, Chin-Chang Yang

#pragma once

#include "../builder/Operations.hpp"
#include <string>
#include <vector>

namespace katagocoreml {

/// Serializes model weights to MIL blob storage format
class WeightSerializer {
public:
    /// Write weights to blob file
    /// @param weights Vector of weight entries to serialize
    /// @param blob_path Path to output blob file
    /// @param use_fp16 If true, convert weights to FLOAT16
    /// @return Total bytes written
    static size_t serialize(std::vector<WeightEntry>& weights,
                            const std::string& blob_path,
                            bool use_fp16 = false);
};

}  // namespace katagocoreml
