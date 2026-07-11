// katagocoreml - Standalone C++ KataGo to Core ML Converter
// Copyright (c) 2025, Chin-Chang Yang

#include "WeightSerializer.hpp"
#include "MILBlob/Blob/StorageWriter.hpp"
#include "MILBlob/Fp16.hpp"
#include "MILBlob/Util/Span.hpp"

namespace katagocoreml {

size_t WeightSerializer::serialize(std::vector<WeightEntry>& weights,
                                   const std::string& blob_path,
                                   bool use_fp16) {
    MILBlob::Blob::StorageWriter writer(blob_path, true);
    size_t total_bytes = 0;

    for (auto& entry : weights) {
        if (use_fp16) {
            // Convert FP32 weights to FP16
            std::vector<MILBlob::Fp16> fp16_data(entry.data.size());
            for (size_t i = 0; i < entry.data.size(); ++i) {
                fp16_data[i] = MILBlob::Fp16::FromFloat(entry.data[i]);
            }
            MILBlob::Util::Span<const MILBlob::Fp16> span(fp16_data.data(), fp16_data.size());
            entry.blob_offset = writer.WriteData(span);
            total_bytes += entry.data.size() * sizeof(MILBlob::Fp16);
        } else {
            // Write FP32 weights
            MILBlob::Util::Span<const float> span(entry.data.data(), entry.data.size());
            entry.blob_offset = writer.WriteData(span);
            total_bytes += entry.data.size() * sizeof(float);
        }
    }

    return total_bytes;
}

}  // namespace katagocoreml
