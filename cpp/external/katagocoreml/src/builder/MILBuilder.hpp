// katagocoreml - Standalone C++ KataGo to Core ML Converter
// Copyright (c) 2025, Chin-Chang Yang

#pragma once

#include "../types/KataGoTypes.hpp"
#include "Operations.hpp"
#include "MIL.pb.h"
#include <memory>
#include <string>
#include <vector>

namespace katagocoreml {

/// Builder for constructing MIL programs from KataGo models.
/// Converts a parsed KataGo model description into a MIL protobuf program.
class MILBuilder {
public:
    MILBuilder(const KataGoModelDesc& model,
               int board_x_size,
               int board_y_size,
               bool optimize_identity_mask,
               bool use_fp16 = false,
               int min_batch_size = 1,
               int max_batch_size = 1,
               bool use_fp16_io = false);

    /// Build and return the MIL program protobuf
    /// @return Unique pointer to MIL Program protobuf
    std::unique_ptr<CoreML::Specification::MILSpec::Program> build();

    /// Get weight entries for blob serialization
    const std::vector<WeightEntry>& getWeights() const { return m_ops.getWeights(); }

    /// Get board dimensions
    int getBoardXSize() const { return m_board_x_size; }
    int getBoardYSize() const { return m_board_y_size; }

    /// Effective precision flags after construction. The constructor may demote a narrow
    /// transformer trunk to full FP32 (see the precision-tier comment in the .cpp); callers
    /// serializing the model must use these, not the requested options, so the declared
    /// Model-description I/O dtype and metadata match the actual MIL program dtypes.
    bool getEffectiveUseFp16() const { return m_use_fp16; }
    bool getEffectiveUseFp16IO() const { return m_use_fp16_io; }

    /// True if converting a model with this trunk width / transformer-ness would be demoted to
    /// fully-FP32 compute even when FP16 is requested (see the precision-tier comment in the
    /// .cpp). Exposed so callers can report effective precision without performing a conversion.
    static bool wouldBuildFullyFp32(int trunk_num_channels, bool has_transformer_blocks) {
        return has_transformer_blocks && trunk_num_channels < FULL_FP32_MAX_TRUNK_CHANNELS;
    }

private:
    const KataGoModelDesc& m_model;
    int m_board_x_size;
    int m_board_y_size;
    bool m_optimize_identity_mask;
    bool m_use_fp16;
    bool m_use_fp16_io;
    // FP32-in-FP16-mode escalations all run off the FP16-only ANE, so they apply ONLY to transformer
    // trunks (attention widens activation range, overflowing FP16 conv/matmul/pooling accumulation).
    // Plain convnets run pure FP16 on the ANE -- the long-standing pre-tier path, verified to pass
    // testgpuerror (b18c384nbt) and ~2.3x faster than forcing their per-block global pooling to FP32.
    // For transformers: trunks below FULL_FP32_MAX_TRUNK_CHANNELS build fully FP32; at/above it they
    // use non-spatial FP32 (matmuls + pooling) plus conv FP32 (>= CONV_FP32_MIN_TRUNK_CHANNELS).
    // The two constants are currently equal, so there is no partial-tier band with FP16 convs; they
    // are kept separate in case the boundaries ever need to diverge again. See the constructor
    // comment in the .cpp for the measurements behind these values. RMSNorm reductions: FP32 when
    // m_use_fp16.
    static constexpr int CONV_FP32_MIN_TRUNK_CHANNELS = 320;   // transformer convs run FP32 at/above this width
    static constexpr int FULL_FP32_MAX_TRUNK_CHANNELS = 320;   // transformer trunks below this build fully FP32
    // Escalate the attention scores/softmax/attn@V chain to FP32 in the partial tier. Measured on
    // an M2 (b11c768, testgpuerror -quick, baseline 308s): fixes the worst observed fp16 margins
    // (b10c384 rect leadError max 0.73x of limit -> 0.35x; ~10x better policyKLDiv) but costs
    // ~1.7-2.3x eval time, because ANE<->CPU boundary transitions dominate, not FLOPs. Off by
    // default: all current models pass with margin without it. If accuracy is ever preferred over
    // ANE throughput, enable this (softmax-only escalation is NOT a valid cheaper substitute; it
    // measured slower AND less accurate - see buildTransformerAttentionBlock).
    static constexpr bool ATTENTION_CORE_FP32 = false;
    bool m_nonspatial_fp32 = false;  // = m_use_fp16 && hasTransformer (matmuls + global pooling)
    bool m_conv_fp32 = false;        // = m_use_fp16 && hasTransformer && trunk_channels >= CONV_FP32_MIN_...
    int m_min_batch_size;
    int m_max_batch_size;
    CoreML::Specification::MILSpec::DataType m_weight_dtype;
    KataGoOps m_ops;

    // Batch size helpers
    bool isDynamicBatch() const {
        return m_min_batch_size != m_max_batch_size || m_max_batch_size <= 0;
    }
    void setBatchDimension(CoreML::Specification::MILSpec::TensorType* tensor_type);

    // Tensor output helpers with batch dimension support
    void setTensorOutput4D(CoreML::Specification::MILSpec::Operation* op,
                           const std::string& name,
                           int channels, int height, int width);
    void setTensorOutput2D(CoreML::Specification::MILSpec::Operation* op,
                           const std::string& name,
                           int channels);
    void setTensorOutputPooled4D(CoreML::Specification::MILSpec::Operation* op,
                                  const std::string& name,
                                  int channels);
    void setTensorOutputMask4D(CoreML::Specification::MILSpec::Operation* op,
                                const std::string& name);
    void setTensorOutputMaskSpatial4D(CoreML::Specification::MILSpec::Operation* op,
                                       const std::string& name,
                                       int height, int width);

    // Operation name counter for unique names
    int m_var_counter = 0;
    std::string genVarName(const std::string& prefix);

    // MIL program construction helpers
    void addConstOp(CoreML::Specification::MILSpec::Block* block,
                    const std::string& name,
                    const std::vector<float>& data,
                    const std::vector<int64_t>& shape);

    void addIntArrayConstOp(CoreML::Specification::MILSpec::Block* block,
                            const std::string& name,
                            const std::vector<int32_t>& values);

    void addBoolScalarConstOp(CoreML::Specification::MILSpec::Block* block,
                              const std::string& name,
                              bool value);

    void addFloatScalarConstOp(CoreML::Specification::MILSpec::Block* block,
                               const std::string& name,
                               float value);

    void addIntScalarConstOp(CoreML::Specification::MILSpec::Block* block,
                             const std::string& name,
                             int32_t value);

    void addCastOp(CoreML::Specification::MILSpec::Block* block,
                   const std::string& input,
                   const std::string& output,
                   const std::string& dtype,
                   const std::vector<int64_t>& shape);

    // Cast to a tensor with FULLY-specified dims (no forced batch dim like addCastOp). Use for
    // weight tensors (fixed [in,out] dims) when running an otherwise-FP16 op in FP32. Returns the
    // new tensor name. dims use -1 for an unknown/batch dim, >=0 for a constant dim.
    std::string castFixed(CoreML::Specification::MILSpec::Block* block,
                          const std::string& input,
                          const std::string& dtype,
                          const std::vector<int64_t>& dims);

    // Emit global pooling, running it in FP32 when m_nonspatial_fp32 (cast input/mask up, pool,
    // cast the pooled features back to FP16). valueVariant selects the value-head pooling variant.
    void addGlobalPoolingFp32(CoreML::Specification::MILSpec::Block* block,
                              const std::string& input,
                              const std::string& mask,
                              int channels,
                              const std::string& output,
                              bool valueVariant);

    void addConvOp(CoreML::Specification::MILSpec::Block* block,
                   const std::string& input,
                   const ConvLayerDesc& layer,
                   const std::string& output);

    void addBatchNormActivationOps(CoreML::Specification::MILSpec::Block* block,
                                   const std::string& input,
                                   const BatchNormLayerDesc& bn,
                                   const ActivationLayerDesc& act,
                                   const std::string& mask,
                                   const std::string& output);

    void addMishOps(CoreML::Specification::MILSpec::Block* block,
                    const std::string& input,
                    const std::string& output,
                    int rank,
                    int channels);

    void addSiluOps(CoreML::Specification::MILSpec::Block* block,
                    const std::string& input,
                    const std::string& output,
                    int rank,
                    int channels);

    // Generic output-shape setter: dims with -1 entries become unknown/dynamic dimensions.
    void setShape(CoreML::Specification::MILSpec::Operation* op,
                  const std::string& name,
                  const std::vector<int64_t>& dims);

    // Lightweight transformer RMSNorm (weight only, per-position over channels). NCHW in/out.
    std::string addTransformerRMSNorm(CoreML::Specification::MILSpec::Block* block,
                                      const std::string& input,
                                      const TransformerRMSNormDesc& desc,
                                      const std::string& mask,
                                      const std::string& prefix);

    // Full RMSNorm at trunk tip: gamma/beta, spatial or per-position, fused activation. NCHW in/out.
    std::string addTrunkRMSNorm(CoreML::Specification::MILSpec::Block* block,
                                const std::string& input,
                                const RMSNormLayerDesc& desc,
                                const ActivationLayerDesc& act,
                                const std::string& mask,
                                const std::string& prefix);

    std::string buildTransformerAttentionBlock(CoreML::Specification::MILSpec::Block* block,
                                               const std::string& input,
                                               const TransformerAttentionBlockDesc& block_desc,
                                               const std::string& mask,
                                               const std::string& prefix);

    std::string buildTransformerFFNBlock(CoreML::Specification::MILSpec::Block* block,
                                         const std::string& input,
                                         const TransformerFFNBlockDesc& block_desc,
                                         const std::string& mask,
                                         const std::string& prefix);

    void addGlobalPoolingOps(CoreML::Specification::MILSpec::Block* block,
                             const std::string& input,
                             const std::string& mask,
                             int channels,
                             const std::string& output);

    void addGlobalPoolingValueOps(CoreML::Specification::MILSpec::Block* block,
                                  const std::string& input,
                                  const std::string& mask,
                                  int channels,
                                  const std::string& output);

    void addMatMulOp(CoreML::Specification::MILSpec::Block* block,
                     const std::string& input,
                     const MatMulLayerDesc& layer,
                     const std::string& output);

    void addMatBiasOp(CoreML::Specification::MILSpec::Block* block,
                      const std::string& input,
                      const MatBiasLayerDesc& layer,
                      const std::string& output);

    void addLinearOp(CoreML::Specification::MILSpec::Block* block,
                     const std::string& input,
                     const MatMulLayerDesc& matmul,
                     const MatBiasLayerDesc& bias,
                     const std::string& output);

    // Network component builders
    std::string buildTrunk(CoreML::Specification::MILSpec::Block* block,
                           const std::string& spatial_input,
                           const std::string& global_input,
                           const std::string& mask,
                           const std::string* meta_input);

    std::string buildResidualBlock(CoreML::Specification::MILSpec::Block* block,
                                   const std::string& input,
                                   const ResidualBlockDesc& block_desc,
                                   const std::string& mask,
                                   const std::string& prefix);

    std::string buildGlobalPoolingResidualBlock(CoreML::Specification::MILSpec::Block* block,
                                                 const std::string& input,
                                                 const GlobalPoolingResidualBlockDesc& block_desc,
                                                 const std::string& mask,
                                                 const std::string& prefix);

    std::string buildNestedBottleneckBlock(CoreML::Specification::MILSpec::Block* block,
                                            const std::string& input,
                                            const NestedBottleneckResidualBlockDesc& block_desc,
                                            const std::string& mask,
                                            const std::string& prefix);

    void buildPolicyHead(CoreML::Specification::MILSpec::Block* block,
                         const std::string& trunk_out,
                         const std::string& mask,
                         std::string& policy_out,
                         std::string& pass_out);

    void buildValueHead(CoreML::Specification::MILSpec::Block* block,
                        const std::string& trunk_out,
                        const std::string& mask,
                        std::string& value_out,
                        std::string& ownership_out,
                        std::string& score_value_out);

    std::string buildSGFMetadataEncoder(CoreML::Specification::MILSpec::Block* block,
                                        const std::string& meta_input,
                                        const SGFMetadataEncoderDesc& encoder);
};

}  // namespace katagocoreml
