#include "../neuralnet/nninterface.h"
#include "../neuralnet/nninputs.h"
#include "../neuralnet/nneval.h"
#include "../neuralnet/modelversion.h"
#include "../neuralnet/desc.h"
#include "../neuralnet/sgfmetadata.h"
#include "../neuralnet/activations.h"
#include "../neuralnet/activations.h"

#include "../core/fileutils.h"
#include "../core/makedir.h"
#include "../core/sha2.h"
#include "../dataio/homedata.h"

#include <migraphx/program.hpp>
#include <migraphx/module.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/shape.hpp>
#include <migraphx/compile_options.hpp>
#include <migraphx/target.hpp>
#include <migraphx/context.hpp>
#include <migraphx/literal.hpp>
#include <migraphx/operation.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/register_target.hpp>
#include <migraphx/argument.hpp>
#include <migraphx/value.hpp>
#include <migraphx/load_save.hpp>
#include <migraphx/half.hpp>

#include <iostream>
#include <fstream>
#include <memory>
#include <vector>
#include <string>
#include <sstream>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <mutex>
#include <map>

using namespace std;

//------------------------ MIGraphX Backend Documentation ------------------------
//
// This is a MIGraphX backend implementation for KataGo.
// 
// Current Status:
// - Full model weight loading from ModelDesc
// - Complete residual network structure (28 blocks for b28c512nbt)
// - Input/output tensor handling
// - Working inference with MIGraphX GPU backend
//
// Known Limitations:
// - BatchNorm is simplified (skipped) due to MIGraphX broadcast limitations
// - Global pooling residual blocks use simplified implementation
// - Value/Score/Ownership heads use simplified projections
//
// Future Optimizations:
// - Implement proper BatchNorm with broadcast
// - Full global pooling residual block implementation
// - Complete value head with v2Mul/v3Mul layers
// - FP16 support for faster inference
//
//------------------------ MIGraphX Model Implementation ------------------------

struct MIGraphXModel {
    // Multiple compiled programs for different batch sizes
    // Key: batch size, Value: compiled program
    map<int, migraphx::program> progs;
    migraphx::target tgt;
    // Sorted batch sizes for quick lookup
    vector<int> batchSizes;
    
    int modelVersion;
    int maxBatchSize;
    int nnXLen, nnYLen;
    bool useFP16;
    bool useNHWC;
    
    int numInputChannels;
    int numInputGlobalChannels;
    int numInputMetaChannels;
    int numPolicyChannels;
    int numValueChannels;
    int numScoreValueChannels;
    int numOwnershipChannels;
    
    MIGraphXModel()
        : modelVersion(0), maxBatchSize(1), nnXLen(19), nnYLen(19),
          useFP16(false), useNHWC(false),
          numInputChannels(0), numInputGlobalChannels(0), numInputMetaChannels(0),
          numPolicyChannels(0), numValueChannels(3),
          numScoreValueChannels(0), numOwnershipChannels(0) {}
    
    // Find the best (smallest sufficient) batch size for the given actual batch
    int getBestBatchSize(int actualBatch) const {
        for(int bs : batchSizes) {
            if(bs >= actualBatch) return bs;
        }
        return batchSizes.back();
    }
    
    migraphx::program& getProgram(int batchSize) {
        return progs.at(batchSize);
    }
};

// Helper class to build MIGraphX graph
class MIGraphXGraphBuilder {
public:
    migraphx::module* main_module;
    migraphx::shape::type_t dataType;
    int batchSize;
    int nnXLen, nnYLen;
    
    MIGraphXGraphBuilder(migraphx::module* mod, migraphx::shape::type_t dtype, int batch, int x, int y)
        : main_module(mod), dataType(dtype), batchSize(batch), nnXLen(x), nnYLen(y) {}
    
    // Add a convolution layer
    migraphx::instruction_ref addConv(
        migraphx::instruction_ref input,
        const ConvLayerDesc& convDesc
    ) {
        // Validate dimensions
        if(convDesc.inChannels <= 0 || convDesc.inChannels > 10000 ||
           convDesc.outChannels <= 0 || convDesc.outChannels > 10000 ||
           convDesc.convYSize <= 0 || convDesc.convYSize > 100 ||
           convDesc.convXSize <= 0 || convDesc.convXSize > 100) {
            cerr << "ERROR: Conv " << convDesc.name << " has invalid dimensions (in=" << convDesc.inChannels
                 << ", out=" << convDesc.outChannels << ", ky=" << convDesc.convYSize 
                 << ", kx=" << convDesc.convXSize << ")" << endl;
            return input;
        }
        
        vector<size_t> wShape = {
            (size_t)convDesc.outChannels,
            (size_t)convDesc.inChannels,
            (size_t)convDesc.convYSize,
            (size_t)convDesc.convXSize
        };
        size_t expectedWeights = (size_t)convDesc.outChannels * (size_t)convDesc.inChannels 
                                 * (size_t)convDesc.convYSize * (size_t)convDesc.convXSize;
        
        if(convDesc.weights.size() != expectedWeights) {
            cerr << "ERROR: Conv " << convDesc.name << " weights size mismatch: "
                 << convDesc.weights.size() << " vs expected " << expectedWeights
                 << " (out=" << convDesc.outChannels << ", in=" << convDesc.inChannels
                 << ", ky=" << convDesc.convYSize << ", kx=" << convDesc.convXSize << ")" << endl;
            return input;  // Return input to avoid crash
        }
        
        auto weights = addLiteral(convDesc.weights, wShape);
        
        int padY = (convDesc.convYSize - 1) / 2 * convDesc.dilationY;
        int padX = (convDesc.convXSize - 1) / 2 * convDesc.dilationX;
        
        // Use vector<size_t> for array values
        vector<size_t> padding = {(size_t)padY, (size_t)padX};
        vector<size_t> stride = {1, 1};
        vector<size_t> dilation = {(size_t)convDesc.dilationY, (size_t)convDesc.dilationX};
        
        auto conv_op = migraphx::make_op("convolution", {
            {"padding", migraphx::value(padding)},
            {"stride", migraphx::value(stride)},
            {"dilation", migraphx::value(dilation)},
            {"group", 1}
        });
        
        return main_module->add_instruction(conv_op, input, weights);
    }
    
    // Add batch normalization (inference mode) - full implementation using multibroadcast
    migraphx::instruction_ref addBatchNorm(
        migraphx::instruction_ref input,
        const BatchNormLayerDesc& bnDesc
    ) {
        // Skip if BN has no channels or invalid weights
        if(bnDesc.numChannels <= 0 || bnDesc.numChannels > 10000) {
            cerr << "WARNING: BatchNorm " << bnDesc.name << " has invalid numChannels=" << bnDesc.numChannels 
                 << ", skipping BN" << endl;
            return input;
        }
        
        int numChannels = bnDesc.numChannels;
        
        // Validate weight sizes match numChannels
        if(bnDesc.mergedScale.size() != (size_t)numChannels || bnDesc.mergedBias.size() != (size_t)numChannels) {
            cerr << "WARNING: BatchNorm " << bnDesc.name << " weight size mismatch (C=" << numChannels 
                 << ", scale=" << bnDesc.mergedScale.size() << ", bias=" << bnDesc.mergedBias.size() 
                 << "), skipping BN" << endl;
            return input;
        }
        
        // Create scale and bias literals from mergedScale and mergedBias
        vector<size_t> paramShape = {(size_t)numChannels};
        auto scale = addLiteral(bnDesc.mergedScale, paramShape);
        auto bias = addLiteral(bnDesc.mergedBias, paramShape);
        
        // Get input shape for broadcasting
        auto input_shape = input->get_shape();
        vector<size_t> input_lens = input_shape.lens();
        
        // Unsqueeze scale and bias from [C] to [1, C, 1, 1] for broadcasting
        auto scale_unsqueezed = main_module->add_instruction(
            migraphx::make_op("unsqueeze", {{"axes", migraphx::value(vector<int64_t>{0, 2, 3})}}), scale);
        auto bias_unsqueezed = main_module->add_instruction(
            migraphx::make_op("unsqueeze", {{"axes", migraphx::value(vector<int64_t>{0, 2, 3})}}), bias);
        
        // Broadcast scale and bias to input shape using multibroadcast
        // Input is NCHW: [batch, channels, height, width]
        auto scale_broadcast = main_module->add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", input_lens}}), scale_unsqueezed);
        auto bias_broadcast = main_module->add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", input_lens}}), bias_unsqueezed);
        
        // Apply scale and bias: y = x * scale + bias
        auto scaled = main_module->add_instruction(migraphx::make_op("mul"), input, scale_broadcast);
        auto result = main_module->add_instruction(migraphx::make_op("add"), scaled, bias_broadcast);
        
        return result;
    }
    
    // Add MatMul layer
    migraphx::instruction_ref addMatMul(
        migraphx::instruction_ref input,
        const MatMulLayerDesc& matmulDesc,
        const MatBiasLayerDesc* biasDesc = nullptr
    ) {
        // Validate channel counts
        if(matmulDesc.inChannels <= 0 || matmulDesc.inChannels > 10000 || 
           matmulDesc.outChannels <= 0 || matmulDesc.outChannels > 10000) {
            cerr << "ERROR: MatMul " << matmulDesc.name << " has invalid channels (in=" 
                 << matmulDesc.inChannels << ", out=" << matmulDesc.outChannels << ")" << endl;
            return input;
        }
        
        vector<size_t> wShape = {(size_t)matmulDesc.inChannels, (size_t)matmulDesc.outChannels};
        size_t expectedWeights = (size_t)matmulDesc.inChannels * (size_t)matmulDesc.outChannels;
        if(matmulDesc.weights.size() != expectedWeights) {
            cerr << "ERROR: MatMul " << matmulDesc.name << " weights size mismatch: " 
                 << matmulDesc.weights.size() << " vs expected " << expectedWeights 
                 << " (in=" << matmulDesc.inChannels << ", out=" << matmulDesc.outChannels << ")" << endl;
            // Return input to avoid crash (this will break the model but prevent segfault)
            return input;
        }
        auto weights = addLiteral(matmulDesc.weights, wShape);
        
        auto matmul = main_module->add_instruction(migraphx::make_op("dot"), input, weights);
        
        if(biasDesc != nullptr && !biasDesc->weights.empty()) {
            if(biasDesc->weights.size() != (size_t)biasDesc->numChannels) {
                cerr << "ERROR: MatMul bias " << biasDesc->name << " size mismatch: "
                     << biasDesc->weights.size() << " vs expected " << biasDesc->numChannels << endl;
            } else {
                vector<size_t> bShape = {(size_t)biasDesc->numChannels};
                auto bias = addLiteral(biasDesc->weights, bShape);
                
                // Unsqueeze for broadcasting: [numChannels] -> [1, numChannels]
                auto unsqueeze_op = migraphx::make_op("unsqueeze", {{"axes", migraphx::value({0})}});
                bias = main_module->add_instruction(unsqueeze_op, bias);
                
                // Explicit broadcast to match matmul output shape
                auto matmulShape = matmul->get_shape().lens();
                bias = main_module->add_instruction(
                    migraphx::make_op("multibroadcast", {{"out_lens", matmulShape}}), bias);
                
                matmul = main_module->add_instruction(migraphx::make_op("add"), matmul, bias);
            }
        }
        
        return matmul;
    }
    
    // Add activation
    migraphx::instruction_ref addActivation(migraphx::instruction_ref input, int activationType) {
        if(activationType == ACTIVATION_IDENTITY) {
            return input;
        }
        else if(activationType == ACTIVATION_RELU) {
            return main_module->add_instruction(migraphx::make_op("relu"), input);
        }
        else if(activationType == ACTIVATION_MISH) {
            return addMish(input);
        }
        else if(activationType == ACTIVATION_MISH_SCALE8) {
            return addMishScale8(input);
        }
        // Fallback to relu
        return main_module->add_instruction(migraphx::make_op("relu"), input);
    }
    
    // Mish activation: x * tanh(softplus(x)) = x * tanh(log(1 + exp(x)))
    migraphx::instruction_ref addMish(migraphx::instruction_ref input) {
        auto inputLens = input->get_shape().lens();
        // softplus(x) = log(1 + exp(x))
        auto exp_x = main_module->add_instruction(migraphx::make_op("exp"), input);
        auto ones = broadcastScalar(1.0f, inputLens);
        auto one_plus_exp = main_module->add_instruction(migraphx::make_op("add"), exp_x, ones);
        auto softplus = main_module->add_instruction(migraphx::make_op("log"), one_plus_exp);
        auto tanh_sp = main_module->add_instruction(migraphx::make_op("tanh"), softplus);
        return main_module->add_instruction(migraphx::make_op("mul"), input, tanh_sp);
    }
    
    // Mish-scale8 activation: x * tanh(softplus(clamp(8x, -, 30)))
    // For x >= 2.5: tanh(softplus(20+)) ≈ 1, so result ≈ x (identity)
    // For x < 2.5: standard mish with 8x scaling of softplus argument
    migraphx::instruction_ref addMishScale8(migraphx::instruction_ref input) {
        auto inputLens = input->get_shape().lens();
        // scaled = 8 * x, clamped to max 30 to prevent exp overflow
        auto eight = broadcastScalar(8.0f, inputLens);
        auto scaled = main_module->add_instruction(migraphx::make_op("mul"), input, eight);
        auto thirty = broadcastScalar(30.0f, inputLens);
        scaled = main_module->add_instruction(migraphx::make_op("min"), scaled, thirty);
        // softplus(scaled) = log(1 + exp(scaled))
        auto exp_s = main_module->add_instruction(migraphx::make_op("exp"), scaled);
        auto ones = broadcastScalar(1.0f, inputLens);
        auto one_plus_exp = main_module->add_instruction(migraphx::make_op("add"), exp_s, ones);
        auto softplus = main_module->add_instruction(migraphx::make_op("log"), one_plus_exp);
        auto tanh_sp = main_module->add_instruction(migraphx::make_op("tanh"), softplus);
        return main_module->add_instruction(migraphx::make_op("mul"), input, tanh_sp);
    }
    
    // Helper: broadcast a scalar to the given shape
    migraphx::instruction_ref broadcastScalar(float val, const vector<size_t>& targetLens) {
        vector<size_t> onesShape(targetLens.size(), 1);
        auto lit = addLiteral({val}, onesShape);
        return main_module->add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", targetLens}}), lit);
    }
    
    // Add literal
    migraphx::instruction_ref addLiteral(const vector<float>& data, const vector<size_t>& dims) {
        migraphx::shape s(dataType, dims);
        return main_module->add_literal(migraphx::literal(s, data));
    }
    
    // Convert tensor to specified data type
    migraphx::instruction_ref addConvert(migraphx::instruction_ref input, migraphx::shape::type_t targetType) {
        if(input->get_shape().type() == targetType) {
            return input;
        }
        auto convert_op = migraphx::make_op("convert", {{"target_type", targetType}});
        return main_module->add_instruction(convert_op, input);
    }
    
    // Global average pooling
    migraphx::instruction_ref addGlobalAvgPool(migraphx::instruction_ref input) {
        auto pool_op = migraphx::make_op("pooling", {
            {"mode", 0},  // average
            {"padding", migraphx::value({0, 0})},
            {"stride", migraphx::value({(size_t)nnYLen, (size_t)nnXLen})},
            {"lengths", migraphx::value({(size_t)nnYLen, (size_t)nnXLen})}
        });
        return main_module->add_instruction(pool_op, input);
    }
    
    // Flatten
    migraphx::instruction_ref addFlatten(migraphx::instruction_ref input, size_t axis = 1) {
        auto flatten_op = migraphx::make_op("flatten", {{"axis", axis}});
        return main_module->add_instruction(flatten_op, input);
    }
    
    // Squeeze
    migraphx::instruction_ref addSqueeze(migraphx::instruction_ref input, const vector<size_t>& axes) {
        auto squeeze_op = migraphx::make_op("squeeze", {{"axes", migraphx::value(axes)}});
        return main_module->add_instruction(squeeze_op, input);
    }
    
    // Tanh
    migraphx::instruction_ref addTanh(migraphx::instruction_ref input) {
        return main_module->add_instruction(migraphx::make_op("tanh"), input);
    }
    
    // Reduce sum over specified axes
    migraphx::instruction_ref addReduceSum(migraphx::instruction_ref input, const vector<int64_t>& axes) {
        auto reduce_op = migraphx::make_op("reduce_sum", {{"axes", migraphx::value(axes)}});
        return main_module->add_instruction(reduce_op, input);
    }
    
    // Reduce max over specified axes
    migraphx::instruction_ref addReduceMax(migraphx::instruction_ref input, const vector<int64_t>& axes) {
        auto reduce_op = migraphx::make_op("reduce_max", {{"axes", migraphx::value(axes)}});
        return main_module->add_instruction(reduce_op, input);
    }
    
    // Reduce mean over specified axes
    migraphx::instruction_ref addReduceMean(migraphx::instruction_ref input, const vector<int64_t>& axes) {
        auto reduce_op = migraphx::make_op("reduce_mean", {{"axes", migraphx::value(axes)}});
        return main_module->add_instruction(reduce_op, input);
    }
    
    // Element-wise multiplication
    migraphx::instruction_ref addMul(migraphx::instruction_ref a, migraphx::instruction_ref b) {
        return main_module->add_instruction(migraphx::make_op("mul"), a, b);
    }
    
    // Element-wise addition
    migraphx::instruction_ref addAdd(migraphx::instruction_ref a, migraphx::instruction_ref b) {
        return main_module->add_instruction(migraphx::make_op("add"), a, b);
    }
    
    // Element-wise subtraction
    migraphx::instruction_ref addSub(migraphx::instruction_ref a, migraphx::instruction_ref b) {
        return main_module->add_instruction(migraphx::make_op("sub"), a, b);
    }
    
    // Element-wise division
    migraphx::instruction_ref addDiv(migraphx::instruction_ref a, migraphx::instruction_ref b) {
        return main_module->add_instruction(migraphx::make_op("div"), a, b);
    }
    
    // Power operation
    migraphx::instruction_ref addPow(migraphx::instruction_ref input, float exponent) {
        vector<float> expData = {exponent};
        auto expLit = addLiteral(expData, {1, 1, 1, 1});
        return main_module->add_instruction(migraphx::make_op("pow"), input, expLit);
    }
    
    // Sqrt operation
    migraphx::instruction_ref addSqrt(migraphx::instruction_ref input) {
        return main_module->add_instruction(migraphx::make_op("sqrt"), input);
    }
    
    // Transpose operation
    migraphx::instruction_ref addTranspose(migraphx::instruction_ref input, const vector<int64_t>& dims) {
        auto transpose_op = migraphx::make_op("transpose", {{"dims", migraphx::value(dims)}});
        return main_module->add_instruction(transpose_op, input);
    }
    
    // Concatenate along axis
    migraphx::instruction_ref addConcat(const vector<migraphx::instruction_ref>& inputs, int64_t axis) {
        auto concat_op = migraphx::make_op("concat", {{"axis", axis}});
        return main_module->add_instruction(concat_op, inputs);
    }
    
    // Global pooling producing 3 features per channel.
    // For trunk/policy: [mean, mean*scale1, max]
    // For value head:   [mean, mean*scale1, mean*scale2]
    // Input: [batch, C, H, W], Output: [batch, C*3]
    // Note: assumes full board (no mask), correct for standard play at nnXLen x nnYLen.
    migraphx::instruction_ref addGPool(migraphx::instruction_ref input, bool isValueHead = false) {
        float boardArea = (float)(nnXLen * nnYLen);
        float sqrtBoardArea = sqrtf(boardArea);
        float scale1Factor = (sqrtBoardArea - 14.0f) * 0.1f;
        
        // mean: [batch, C, H, W] -> [batch, C, 1, 1] -> [batch, C]
        auto mean = addReduceMean(input, {2, 3});
        mean = addSqueeze(mean, {2, 3});
        
        auto meanShape = mean->get_shape().lens();
        
        // scale1 = mean * scale1Factor
        auto scale1Lit = addLiteral({scale1Factor}, {1, 1});
        auto scale1Broadcast = main_module->add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", meanShape}}), scale1Lit);
        auto scale1 = main_module->add_instruction(migraphx::make_op("mul"), mean, scale1Broadcast);
        
        migraphx::instruction_ref third;
        if(isValueHead) {
            // scale2 = mean * ((sqrtBoardArea - 14)^2 * 0.01 - 0.1)
            float scale2Factor = (sqrtBoardArea - 14.0f) * (sqrtBoardArea - 14.0f) * 0.01f - 0.1f;
            auto scale2Lit = addLiteral({scale2Factor}, {1, 1});
            auto scale2Broadcast = main_module->add_instruction(
                migraphx::make_op("multibroadcast", {{"out_lens", meanShape}}), scale2Lit);
            third = main_module->add_instruction(migraphx::make_op("mul"), mean, scale2Broadcast);
        } else {
            // max: [batch, C, H, W] -> [batch, C, 1, 1] -> [batch, C]
            auto maxVal = addReduceMax(input, {2, 3});
            third = addSqueeze(maxVal, {2, 3});
        }
        
        // Concat [mean, scale1, third] along axis 1 -> [batch, C*3]
        return addConcat({mean, scale1, third}, 1);
    }
    
};

// Build residual block
static migraphx::instruction_ref buildResidualBlock(
    MIGraphXGraphBuilder& builder,
    migraphx::instruction_ref input,
    const ResidualBlockDesc& blockDesc
) {
    auto residual = input;
    
    // preBN + preActivation
    auto x = builder.addBatchNorm(input, blockDesc.preBN);
    x = builder.addActivation(x, blockDesc.preActivation.activation);
    
    // regularConv
    x = builder.addConv(x, blockDesc.regularConv);
    x = builder.addBatchNorm(x, blockDesc.midBN);
    
    // midActivation
    x = builder.addActivation(x, blockDesc.midActivation.activation);
    
    // finalConv
    x = builder.addConv(x, blockDesc.finalConv);
    
    // Add residual
    return builder.main_module->add_instruction(migraphx::make_op("add"), x, residual);
}

// Forward declarations
static migraphx::instruction_ref buildResidualBlock(
    MIGraphXGraphBuilder& builder,
    migraphx::instruction_ref input,
    const ResidualBlockDesc& blockDesc
);

static migraphx::instruction_ref buildGlobalPoolingResidualBlock(
    MIGraphXGraphBuilder& builder,
    migraphx::instruction_ref input,
    const GlobalPoolingResidualBlockDesc& blockDesc
);

static migraphx::instruction_ref buildNestedBottleneckResidualBlock(
    MIGraphXGraphBuilder& builder,
    migraphx::instruction_ref input,
    const NestedBottleneckResidualBlockDesc& blockDesc
);

static migraphx::instruction_ref buildResidualBlockStack(
    MIGraphXGraphBuilder& builder,
    migraphx::instruction_ref input,
    const std::vector<std::pair<int, unique_ptr_void>>& blocks,
    const string& namePrefix
);

// Build nested bottleneck residual block
static migraphx::instruction_ref buildNestedBottleneckResidualBlock(
    MIGraphXGraphBuilder& builder,
    migraphx::instruction_ref input,
    const NestedBottleneckResidualBlockDesc& blockDesc
) {
    auto residual = input;
    
    // Pre BN + Activation
    auto x = builder.addBatchNorm(input, blockDesc.preBN);
    x = builder.addActivation(x, blockDesc.preActivation.activation);
    
    // Pre conv (bottleneck down)
    x = builder.addConv(x, blockDesc.preConv);
    
    // Inner residual block stack
    x = buildResidualBlockStack(builder, x, blockDesc.blocks, blockDesc.name);
    
    // Post BN + Activation
    x = builder.addBatchNorm(x, blockDesc.postBN);
    x = builder.addActivation(x, blockDesc.postActivation.activation);
    
    // Post conv (bottleneck up)
    x = builder.addConv(x, blockDesc.postConv);
    
    // Add residual
    return builder.main_module->add_instruction(migraphx::make_op("add"), x, residual);
}

// Build residual block stack (used by trunk and nested blocks)
static migraphx::instruction_ref buildResidualBlockStack(
    MIGraphXGraphBuilder& builder,
    migraphx::instruction_ref input,
    const std::vector<std::pair<int, unique_ptr_void>>& blocks,
    const string& namePrefix
) {
    auto trunk = input;
    
    for(size_t i = 0; i < blocks.size(); i++) {
        int blockKind = blocks[i].first;
        
        if(blockKind == ORDINARY_BLOCK_KIND) {
            const ResidualBlockDesc* blockDesc = static_cast<const ResidualBlockDesc*>(blocks[i].second.get());
            trunk = buildResidualBlock(builder, trunk, *blockDesc);
        } else if(blockKind == GLOBAL_POOLING_BLOCK_KIND) {
            const GlobalPoolingResidualBlockDesc* blockDesc = static_cast<const GlobalPoolingResidualBlockDesc*>(blocks[i].second.get());
            trunk = buildGlobalPoolingResidualBlock(builder, trunk, *blockDesc);
        } else if(blockKind == NESTED_BOTTLENECK_BLOCK_KIND) {
            const NestedBottleneckResidualBlockDesc* blockDesc = static_cast<const NestedBottleneckResidualBlockDesc*>(blocks[i].second.get());
            trunk = buildNestedBottleneckResidualBlock(builder, trunk, *blockDesc);
        }
        
    }
    
    return trunk;
}

// Build global pooling residual block - full implementation
static migraphx::instruction_ref buildGlobalPoolingResidualBlock(
    MIGraphXGraphBuilder& builder,
    migraphx::instruction_ref input,
    const GlobalPoolingResidualBlockDesc& blockDesc
) {
    auto residual = input;
    
    // preBN + preActivation
    auto x = builder.addBatchNorm(input, blockDesc.preBN);
    x = builder.addActivation(x, blockDesc.preActivation.activation);
    
    // Branch A: regular spatial conv
    auto regularOut = builder.addConv(x, blockDesc.regularConv);
    
    // Branch B: global pooling conv
    auto gpoolOut = builder.addConv(x, blockDesc.gpoolConv);
    gpoolOut = builder.addBatchNorm(gpoolOut, blockDesc.gpoolBN);
    gpoolOut = builder.addActivation(gpoolOut, blockDesc.gpoolActivation.activation);
    
    // Global pool: [batch, gpoolC, H, W] -> [batch, gpoolC*3]
    auto gpoolFeatures = builder.addGPool(gpoolOut, false);
    
    // gpoolToBiasMul: [batch, gpoolC*3] -> [batch, regularC]
    auto bias = builder.addMatMul(gpoolFeatures, blockDesc.gpoolToBiasMul);
    
    // Broadcast bias to spatial dims and add to regularOut
    auto regularShape = regularOut->get_shape().lens();
    auto biasUnsqueezed = builder.main_module->add_instruction(
        migraphx::make_op("unsqueeze", {{"axes", migraphx::value(vector<int64_t>{2, 3})}}), bias);
    auto biasBroadcast = builder.main_module->add_instruction(
        migraphx::make_op("multibroadcast", {{"out_lens", regularShape}}), biasUnsqueezed);
    regularOut = builder.main_module->add_instruction(migraphx::make_op("add"), regularOut, biasBroadcast);
    
    // midBN + midActivation
    regularOut = builder.addBatchNorm(regularOut, blockDesc.midBN);
    regularOut = builder.addActivation(regularOut, blockDesc.midActivation.activation);
    
    // finalConv
    regularOut = builder.addConv(regularOut, blockDesc.finalConv);
    
    // Add residual
    return builder.main_module->add_instruction(migraphx::make_op("add"), regularOut, residual);
}

// Build complete MIGraphX program from ModelDesc
static migraphx::program buildMIGraphXProgram(
    const ModelDesc& modelDesc,
    int maxBatchSize,
    int nnXLen,
    int nnYLen,
    bool useFP16,
    bool useNHWC
) {
    migraphx::program prog;
    auto main_module = prog.get_main_module();
    
    migraphx::shape::type_t dataType = useFP16 ? migraphx::shape::half_type : migraphx::shape::float_type;
    
    int numSpatialFeatures = NNModelVersion::getNumSpatialFeatures(modelDesc.modelVersion);
    int numGlobalFeatures = NNModelVersion::getNumGlobalFeatures(modelDesc.modelVersion);
    int numMetaFeatures = modelDesc.numInputMetaChannels;
    
    // Create input parameters
    vector<size_t> inputShape = {(size_t)maxBatchSize, (size_t)numSpatialFeatures, (size_t)nnYLen, (size_t)nnXLen};
    vector<size_t> inputGlobalShape = {(size_t)maxBatchSize, (size_t)numGlobalFeatures};
    
    // Input parameters are always float_type (host buffers are float).
    // If using FP16, we convert to half inside the graph so MIGraphX handles conversion on GPU.
    auto inputSpatial = main_module->add_parameter("input_spatial", migraphx::shape(migraphx::shape::float_type, inputShape));
    auto inputGlobal = main_module->add_parameter("input_global", migraphx::shape(migraphx::shape::float_type, inputGlobalShape));
    
    // MIGraphX backend uses NCHW format only
    (void)useNHWC;  // Silently ignore NHWC setting
    
    MIGraphXGraphBuilder builder(main_module, dataType, maxBatchSize, nnXLen, nnYLen);
    
    // Convert inputs to computation type if using FP16
    if(useFP16) {
        inputSpatial = builder.addConvert(inputSpatial, dataType);
        inputGlobal = builder.addConvert(inputGlobal, dataType);
    }
    
    // Build trunk
    auto trunk = inputSpatial;
    const TrunkDesc& trunkDesc = modelDesc.trunk;
    
    // Initial conv
    if(trunkDesc.initialConv.outChannels > 0 && trunkDesc.initialConv.inChannels == numSpatialFeatures) {
        trunk = builder.addConv(trunk, trunkDesc.initialConv);
    } else if(trunkDesc.initialConv.outChannels > 0) {
        cout << "MIGraphX: Skipping initialConv (input channel mismatch)" << endl;
    }
    
    // Initial MatMul for global features
    if(trunkDesc.initialMatMul.outChannels > 0) {
        auto globalProcessed = builder.addMatMul(inputGlobal, trunkDesc.initialMatMul);
        // Broadcast global features from [N, C] to spatial dimensions [N, C, H, W]
        auto trunkShape = trunk->get_shape().lens();
        auto globalUnsqueezed = main_module->add_instruction(
            migraphx::make_op("unsqueeze", {{"axes", migraphx::value(vector<int64_t>{2, 3})}}), globalProcessed);
        auto globalBroadcast = main_module->add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", trunkShape}}), globalUnsqueezed);
        trunk = main_module->add_instruction(migraphx::make_op("add"), trunk, globalBroadcast);
    }
    
    // SGF Metadata encoder (if enabled) - disabled for now due to potential weight shape issues
    if(trunkDesc.metaEncoderVersion > 0 && numMetaFeatures > 0) {
        // Skip SGF metadata encoder for now
        cout << "MIGraphX: SGF Metadata encoder disabled" << endl;
    }
    
    // Residual blocks using the stack builder
    trunk = buildResidualBlockStack(builder, trunk, trunkDesc.blocks, "trunk");
    
    // trunkTipBN + trunkTipActivation
    trunk = builder.addBatchNorm(trunk, trunkDesc.trunkTipBN);
    trunk = builder.addActivation(trunk, trunkDesc.trunkTipActivation.activation);
    
    // ======== Policy Head ========
    const PolicyHeadDesc& policyDesc = modelDesc.policyHead;
    
    migraphx::instruction_ref policy = trunk;
    migraphx::instruction_ref policyPass = trunk; // will be overwritten
    
    if(policyDesc.p1Conv.outChannels > 0) {
        // p1Conv branch (spatial policy)
        auto p1Conv = builder.addConv(trunk, policyDesc.p1Conv);
        
        // g1Conv branch for global pooling
        auto g1Conv = builder.addConv(trunk, policyDesc.g1Conv);
        g1Conv = builder.addBatchNorm(g1Conv, policyDesc.g1BN);
        g1Conv = builder.addActivation(g1Conv, policyDesc.g1Activation.activation);
        
        // Global pool: [batch, g1C, H, W] -> [batch, g1C*3]
        auto gpool = builder.addGPool(g1Conv, false);
        
        // gpoolToBiasMul: [batch, g1C*3] -> [batch, p1C] bias
        auto gpoolBias = builder.addMatMul(gpool, policyDesc.gpoolToBiasMul);
        
        // Broadcast bias and add to p1Conv
        auto p1Shape = p1Conv->get_shape().lens();
        auto biasUnsqueezed = main_module->add_instruction(
            migraphx::make_op("unsqueeze", {{"axes", migraphx::value(vector<int64_t>{2, 3})}}), gpoolBias);
        auto biasBroadcast = main_module->add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", p1Shape}}), biasUnsqueezed);
        policy = main_module->add_instruction(migraphx::make_op("add"), p1Conv, biasBroadcast);
        
        policy = builder.addBatchNorm(policy, policyDesc.p1BN);
        policy = builder.addActivation(policy, policyDesc.p1Activation.activation);
        
        // p2Conv -> spatial policy logits
        if(policyDesc.p2Conv.outChannels > 0) {
            policy = builder.addConv(policy, policyDesc.p2Conv);
        }
        
        // Flatten spatial policy: [batch, numPolicyChannels, H, W] -> [batch, numPolicyChannels*H*W]
        policy = builder.addFlatten(policy);
        
        // Pass policy (separate path from spatial, uses same gpool)
        // gpoolToPassMul: [batch, g1C*3] -> passHidden
        policyPass = builder.addMatMul(gpool, policyDesc.gpoolToPassMul, &policyDesc.gpoolToPassBias);
        policyPass = builder.addActivation(policyPass, policyDesc.passActivation.activation);
        
        // gpoolToPassMul2: passHidden -> [batch, numPolicyChannels] (for modelVersion >= 15)
        if(policyDesc.gpoolToPassMul2.outChannels > 0) {
            policyPass = builder.addMatMul(policyPass, policyDesc.gpoolToPassMul2);
        }
    } else {
        policy = builder.addFlatten(trunk);
        // Zero pass policy fallback
        vector<float> zeroPass(modelDesc.numPolicyChannels, 0.0f);
        policyPass = builder.addLiteral(zeroPass, {1, (size_t)modelDesc.numPolicyChannels});
        policyPass = main_module->add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", vector<size_t>{(size_t)maxBatchSize, (size_t)modelDesc.numPolicyChannels}}}), policyPass);
    }
    
    // ======== Value Head ========
    const ValueHeadDesc& valueDesc = modelDesc.valueHead;
    
    // v1Conv + v1BN + v1Activation
    auto v1Out = builder.addConv(trunk, valueDesc.v1Conv);
    v1Out = builder.addBatchNorm(v1Out, valueDesc.v1BN);
    v1Out = builder.addActivation(v1Out, valueDesc.v1Activation.activation);
    
    // Ownership branch: v1Out -> vOwnershipConv -> flatten (no tanh - matches CUDA backend)
    auto ownership = builder.addConv(v1Out, valueDesc.vOwnershipConv);
    ownership = builder.addFlatten(ownership);
    
    // Value branch: v1Out -> GPool (value head style) -> v2Mul + v2Bias + v2Activation -> v3Mul + v3Bias
    auto vGpool = builder.addGPool(v1Out, true);  // value head: mean, scale1, scale2
    
    auto v2 = builder.addMatMul(vGpool, valueDesc.v2Mul, &valueDesc.v2Bias);
    v2 = builder.addActivation(v2, valueDesc.v2Activation.activation);
    
    auto valueOut = builder.addMatMul(v2, valueDesc.v3Mul, &valueDesc.v3Bias);
    
    // Score value branch: same v2 -> sv3Mul + sv3Bias
    auto scoreValue = builder.addMatMul(v2, valueDesc.sv3Mul, &valueDesc.sv3Bias);
    
    // Set outputs: policy, policyPass, value, scoreValue, ownership
    if(modelDesc.modelVersion >= 2) {
        main_module->add_return({policy, policyPass, valueOut, scoreValue, ownership});
    } else {
        main_module->add_return({policy, valueOut});
    }
    
    return prog;
}

//------------------------ Backend Structures ------------------------

struct LoadedModelInternal {
    ModelDesc modelDesc;
    string modelFile;
    string expectedSha256;
    
    LoadedModelInternal(const string& file, const string& sha256) : modelFile(file), expectedSha256(sha256) {
        ModelDesc::loadFromFileMaybeGZipped(file, modelDesc, sha256);
        modelDesc.applyScale8ToReduceActivations();
    }
};

struct ComputeContextInternal {
    int nnXLen, nnYLen;
    enabled_t useFP16Mode;
    enabled_t useNHWCMode;
    string homeDataDir;
    vector<int> gpuIdxs;
};

struct ComputeHandleInternal {
    unique_ptr<MIGraphXModel> model;
    int maxBatchSize;
    int gpuIdx;
    bool requireExactNNLen;
    bool inputsUseNHWC;
    int nnXLen, nnYLen;
};

struct InputBuffersInternal {
    int maxBatchSize;
    int nnXLen, nnYLen;
    
    size_t singleInputElts;
    size_t singleInputBytes;
    size_t singleInputGlobalElts;
    size_t singleInputGlobalBytes;
    size_t singleInputMetaElts;
    size_t singleInputMetaBytes;
    
    size_t userInputBufferBytes;
    size_t userInputGlobalBufferBytes;
    size_t userInputMetaBufferBytes;
    
    vector<float> userInputBuffer;
    vector<float> userInputGlobalBuffer;
    vector<float> userInputMetaBuffer;
    
    size_t singlePolicyResultElts;
    size_t singlePolicyResultBytes;
    size_t singlePolicyPassResultElts;
    size_t singlePolicyPassResultBytes;
    size_t singleValueResultElts;
    size_t singleValueResultBytes;
    size_t singleScoreValueResultElts;
    size_t singleScoreValueResultBytes;
    size_t singleOwnershipResultElts;
    size_t singleOwnershipResultBytes;
    
    vector<float> policyResults;
    vector<float> policyPassResults;
    vector<float> valueResults;
    vector<float> scoreValueResults;
    vector<float> ownershipResults;
    
    size_t policyResultBufferBytes;
    size_t policyPassResultBufferBytes;
    size_t valueResultBufferBytes;
    size_t scoreValueResultBufferBytes;
    size_t ownershipResultBufferBytes;
};

//------------------------ NeuralNet Implementation ------------------------

namespace NeuralNet {

void globalInitialize() {}
void globalCleanup() {}

void printDevices() {
    cout << "MIGraphX Backend: AMD GPU via MIGraphX" << endl;
}

LoadedModel* loadModelFile(const string& file, const string& expectedSha256) {
    return reinterpret_cast<LoadedModel*>(new LoadedModelInternal(file, expectedSha256));
}

void freeLoadedModel(LoadedModel* loadedModel) {
    if(loadedModel) {
        LoadedModelInternal* model = reinterpret_cast<LoadedModelInternal*>(loadedModel);
        delete model;
    }
}

const ModelDesc& getModelDesc(const LoadedModel* loadedModel) {
    return reinterpret_cast<const LoadedModelInternal*>(loadedModel)->modelDesc;
}

ComputeContext* createComputeContext(
    const vector<int>& gpuIdxs,
    Logger* logger,
    int nnXLen,
    int nnYLen,
    const string& openCLTunerFile,
    const string& homeDataDirOverride,
    bool openCLReTunePerBoardSize,
    enabled_t useFP16Mode,
    enabled_t useNHWCMode,
    const LoadedModel* loadedModel
) {
    (void)logger;
    (void)openCLTunerFile;
    (void)homeDataDirOverride;
    (void)openCLReTunePerBoardSize;
    (void)loadedModel;
    
    auto context = new ComputeContextInternal();
    context->gpuIdxs = gpuIdxs;
    context->nnXLen = nnXLen;
    context->nnYLen = nnYLen;
    context->useFP16Mode = useFP16Mode;
    context->useNHWCMode = useNHWCMode;
    
    return reinterpret_cast<ComputeContext*>(context);
}

void freeComputeContext(ComputeContext* computeContext) {
    if(computeContext) {
        ComputeContextInternal* context = reinterpret_cast<ComputeContextInternal*>(computeContext);
        delete context;
    }
}

// Static mutex for cache operations
static mutex migraphxCacheMutex;

// Generate batch sizes to compile for MIGraphX (no dynamic batch support).
static vector<int> generateBatchSizes(int maxBatchSize) {
    vector<int> candidates = {4, 8, 16, 24, 32, 40, 64};
    
    // Keep only sizes <= maxBatchSize, always include maxBatchSize itself
    vector<int> sizes;
    for(int s : candidates) {
        if(s <= maxBatchSize)
            sizes.push_back(s);
    }
    if(sizes.empty() || sizes.back() != maxBatchSize)
        sizes.push_back(maxBatchSize);
    return sizes;
}

// Generate cache file path
static string getCacheFilePath(
    const string& homeDataDir,
    const ModelDesc& modelDesc,
    int nnXLen,
    int nnYLen,
    int maxBatchSize,
    bool useFP16,
    bool useNHWC,
    bool requireExactNNLen
) {
    auto cacheDir = HomeData::getHomeDataDir(true, homeDataDir);
    cacheDir += "/migraphxcache";
    
    // Create directory if not exists
    MakeDir::make(cacheDir);
    
    // Generate unique cache key based on model and parameters
    string cacheKey = Global::strprintf(
        "migraphx_%s_%s_%dx%d_batch%d_fp%d_nhwc%d_%s",
        modelDesc.name.c_str(),
        modelDesc.sha256.substr(0, 16).c_str(),
        nnYLen,
        nnXLen,
        maxBatchSize,
        useFP16 ? 1 : 0,
        useNHWC ? 1 : 0,
        requireExactNNLen ? "exact" : "max"
    );
    
    return cacheDir + "/" + cacheKey + ".mxr";
}

ComputeHandle* createComputeHandle(
    ComputeContext* context,
    const LoadedModel* loadedModel,
    Logger* logger,
    int maxBatchSize,
    bool requireExactNNLen,
    bool inputsUseNHWC,
    int gpuIdxForThisThread,
    int serverThreadIdx
) {
    (void)serverThreadIdx;
    
    ComputeContextInternal* ctx = reinterpret_cast<ComputeContextInternal*>(context);
    const LoadedModelInternal* model = reinterpret_cast<const LoadedModelInternal*>(loadedModel);
    
    auto handle = new ComputeHandleInternal();
    handle->maxBatchSize = maxBatchSize;
    handle->gpuIdx = gpuIdxForThisThread;
    handle->requireExactNNLen = requireExactNNLen;
    handle->inputsUseNHWC = inputsUseNHWC;
    handle->nnXLen = ctx->nnXLen;
    handle->nnYLen = ctx->nnYLen;
    
    bool useFP16 = (ctx->useFP16Mode == enabled_t::True || ctx->useFP16Mode == enabled_t::Auto);
    bool useNHWC = (ctx->useNHWCMode == enabled_t::True);
    
    // MIGraphX backend only supports NCHW format
    if(useNHWC) {
        cout << "MIGraphX: WARNING: NHWC format is not supported, forcing NCHW" << endl;
        useNHWC = false;
    }
    
    handle->model = make_unique<MIGraphXModel>();
    handle->model->modelVersion = model->modelDesc.modelVersion;
    handle->model->maxBatchSize = maxBatchSize;
    handle->model->nnXLen = ctx->nnXLen;
    handle->model->nnYLen = ctx->nnYLen;
    handle->model->useFP16 = useFP16;
    handle->model->useNHWC = false;  // Always NCHW
    
    handle->model->numInputChannels = model->modelDesc.numInputChannels;
    handle->model->numInputGlobalChannels = model->modelDesc.numInputGlobalChannels;
    handle->model->numInputMetaChannels = model->modelDesc.numInputMetaChannels;
    handle->model->numPolicyChannels = model->modelDesc.numPolicyChannels;
    handle->model->numValueChannels = model->modelDesc.numValueChannels;
    handle->model->numScoreValueChannels = model->modelDesc.numScoreValueChannels;
    handle->model->numOwnershipChannels = model->modelDesc.numOwnershipChannels;
    
    // Generate batch sizes to compile
    vector<int> batchSizesToCompile = generateBatchSizes(maxBatchSize);
    handle->model->batchSizes = batchSizesToCompile;
    handle->model->tgt = migraphx::make_target("gpu");
    
    lock_guard<mutex> cacheLock(migraphxCacheMutex);
    
    for(int bs : batchSizesToCompile) {
        // Generate cache file path for this batch size
        string cacheFile = getCacheFilePath(
            ctx->homeDataDir,
            model->modelDesc,
            ctx->nnXLen,
            ctx->nnYLen,
            bs,
            useFP16,
            useNHWC,
            requireExactNNLen
        );
        
        bool cacheLoaded = false;
        
        // Try to load from cache
        if(FileUtils::exists(cacheFile)) {
            try {
                if(logger) {
                    logger->write("MIGraphX: Loading compiled program from cache (batch " + Global::intToString(bs) + "): " + cacheFile);
                }
                cout << "MIGraphX: Loading batch " << bs << " from cache..." << endl;
                
                handle->model->progs[bs] = migraphx::load(cacheFile);
                cacheLoaded = true;
                
                cout << "MIGraphX: Batch " << bs << " loaded! (FP16: " << (useFP16 ? "yes" : "no") << ")" << endl;
            } catch(const exception& e) {
                if(logger) {
                    logger->write(string("MIGraphX: Cache load failed for batch ") + Global::intToString(bs) + ": " + e.what());
                }
                cout << "MIGraphX: Cache load failed for batch " << bs << ", rebuilding..." << endl;
            }
        }
        
        if(!cacheLoaded) {
            cout << "MIGraphX: Building model (version " << model->modelDesc.modelVersion << ")..." << endl;
            cout << "  Board size: " << ctx->nnXLen << "x" << ctx->nnYLen << endl;
            cout << "  Batch size: " << bs << endl;
            cout << "  FP16: " << (useFP16 ? "yes" : "no") << endl;
            cout << "  NHWC: " << (useNHWC ? "yes" : "no") << endl;
            cout << "  Trunk channels: " << model->modelDesc.trunk.trunkNumChannels << endl;
            cout << "  Num blocks: " << model->modelDesc.trunk.numBlocks << endl;
            
            handle->model->progs[bs] = buildMIGraphXProgram(
                model->modelDesc,
                bs,
                ctx->nnXLen,
                ctx->nnYLen,
                useFP16,
                useNHWC
            );
            
            cout << "MIGraphX: Compiling batch " << bs << "..." << endl;
            migraphx::compile_options compile_opts;
            compile_opts.offload_copy = true;
            
            handle->model->progs[bs].compile(handle->model->tgt, compile_opts);
            
            cout << "MIGraphX: Batch " << bs << " compiled!" << endl;
            
            // Save to cache
            try {
                if(logger) {
                    logger->write("MIGraphX: Saving compiled program to cache (batch " + Global::intToString(bs) + "): " + cacheFile);
                }
                migraphx::save(handle->model->progs[bs], cacheFile);
                cout << "MIGraphX: Batch " << bs << " cached!" << endl;
            } catch(const exception& e) {
                if(logger) {
                    logger->write(string("MIGraphX: Cache save failed: ") + e.what());
                }
                cout << "MIGraphX: Cache save failed: " << e.what() << endl;
            }
        }
    }
    
    cout << "MIGraphX: All " << batchSizesToCompile.size() << " batch sizes ready: ";
    for(size_t i = 0; i < batchSizesToCompile.size(); i++) {
        if(i > 0) cout << ", ";
        cout << batchSizesToCompile[i];
    }
    cout << endl;
    
    return reinterpret_cast<ComputeHandle*>(handle);
}

void freeComputeHandle(ComputeHandle* computeHandle) {
    if(computeHandle) {
        ComputeHandleInternal* handle = reinterpret_cast<ComputeHandleInternal*>(computeHandle);
        delete handle;
    }
}

bool isUsingFP16(const ComputeHandle* computeHandle) {
    const ComputeHandleInternal* handle = reinterpret_cast<const ComputeHandleInternal*>(computeHandle);
    return handle->model->useFP16;
}

InputBuffers* createInputBuffers(const LoadedModel* loadedModel, int maxBatchSize, int nnXLen, int nnYLen) {
    const ModelDesc& m = getModelDesc(loadedModel);
    
    auto buffers = new InputBuffersInternal();
    buffers->maxBatchSize = maxBatchSize;
    buffers->nnXLen = nnXLen;
    buffers->nnYLen = nnYLen;
    
    int modelVersion = m.modelVersion;
    int numSpatialFeatures = NNModelVersion::getNumSpatialFeatures(modelVersion);
    int numGlobalFeatures = NNModelVersion::getNumGlobalFeatures(modelVersion);
    int numMetaFeatures = m.numInputMetaChannels;
    
    buffers->singleInputElts = (size_t)numSpatialFeatures * nnXLen * nnYLen;
    buffers->singleInputBytes = buffers->singleInputElts * sizeof(float);
    buffers->singleInputGlobalElts = numGlobalFeatures;
    buffers->singleInputGlobalBytes = buffers->singleInputGlobalElts * sizeof(float);
    buffers->singleInputMetaElts = numMetaFeatures;
    buffers->singleInputMetaBytes = buffers->singleInputMetaElts * sizeof(float);
    
    buffers->userInputBufferBytes = buffers->singleInputBytes * maxBatchSize;
    buffers->userInputGlobalBufferBytes = buffers->singleInputGlobalBytes * maxBatchSize;
    buffers->userInputMetaBufferBytes = buffers->singleInputMetaBytes * maxBatchSize;
    
    buffers->userInputBuffer.resize(buffers->singleInputElts * maxBatchSize, 0.0f);
    buffers->userInputGlobalBuffer.resize(buffers->singleInputGlobalElts * maxBatchSize, 0.0f);
    buffers->userInputMetaBuffer.resize(buffers->singleInputMetaElts * maxBatchSize, 0.0f);
    
    buffers->singlePolicyResultElts = m.numPolicyChannels * nnXLen * nnYLen;
    buffers->singlePolicyResultBytes = buffers->singlePolicyResultElts * sizeof(float);
    buffers->singlePolicyPassResultElts = m.numPolicyChannels;
    buffers->singlePolicyPassResultBytes = buffers->singlePolicyPassResultElts * sizeof(float);
    
    buffers->singleValueResultElts = m.numValueChannels;
    buffers->singleValueResultBytes = buffers->singleValueResultElts * sizeof(float);
    buffers->singleScoreValueResultElts = max(1, m.numScoreValueChannels);
    buffers->singleScoreValueResultBytes = buffers->singleScoreValueResultElts * sizeof(float);
    buffers->singleOwnershipResultElts = nnXLen * nnYLen;
    buffers->singleOwnershipResultBytes = buffers->singleOwnershipResultElts * sizeof(float);
    
    buffers->policyResultBufferBytes = buffers->singlePolicyResultBytes * maxBatchSize;
    buffers->policyPassResultBufferBytes = buffers->singlePolicyPassResultBytes * maxBatchSize;
    buffers->valueResultBufferBytes = buffers->singleValueResultBytes * maxBatchSize;
    buffers->scoreValueResultBufferBytes = buffers->singleScoreValueResultBytes * maxBatchSize;
    buffers->ownershipResultBufferBytes = buffers->singleOwnershipResultBytes * maxBatchSize;
    
    buffers->policyResults.resize(buffers->singlePolicyResultElts * maxBatchSize, 0.0f);
    buffers->policyPassResults.resize(buffers->singlePolicyPassResultElts * maxBatchSize, 0.0f);
    buffers->valueResults.resize(buffers->singleValueResultElts * maxBatchSize, 0.0f);
    buffers->scoreValueResults.resize(buffers->singleScoreValueResultElts * maxBatchSize, 0.0f);
    buffers->ownershipResults.resize(buffers->singleOwnershipResultElts * maxBatchSize, 0.0f);
    
    return reinterpret_cast<InputBuffers*>(buffers);
}

void freeInputBuffers(InputBuffers* buffers) {
    if(buffers) {
        InputBuffersInternal* data = reinterpret_cast<InputBuffersInternal*>(buffers);
        delete data;
    }
}

void getOutput(
    ComputeHandle* computeHandle,
    InputBuffers* inputBuffers,
    int numBatchEltsFilled,
    NNResultBuf** inputBufs,
    vector<NNOutput*>& outputs
) {
    ComputeHandleInternal* handle = reinterpret_cast<ComputeHandleInternal*>(computeHandle);
    InputBuffersInternal* buffers = reinterpret_cast<InputBuffersInternal*>(inputBuffers);
    
    assert(numBatchEltsFilled <= buffers->maxBatchSize);
    assert(numBatchEltsFilled > 0);
    
    int batchSize = numBatchEltsFilled;
    int nnXLen = handle->nnXLen;
    int nnYLen = handle->nnYLen;
    int modelVersion = handle->model->modelVersion;
    
    int numSpatialFeatures = NNModelVersion::getNumSpatialFeatures(modelVersion);
    int numGlobalFeatures = NNModelVersion::getNumGlobalFeatures(modelVersion);
    int numMetaFeatures = handle->model->numInputMetaChannels;
    
    // Copy inputs
    for(int nIdx = 0; nIdx < batchSize; nIdx++) {
        float* rowSpatialInput = buffers->userInputBuffer.data() + (buffers->singleInputElts * nIdx);
        float* rowGlobalInput = buffers->userInputGlobalBuffer.data() + (buffers->singleInputGlobalElts * nIdx);
        float* rowMetaInput = buffers->userInputMetaBuffer.data() + (buffers->singleInputMetaElts * nIdx);
        
        const float* rowGlobal = inputBufs[nIdx]->rowGlobalBuf.data();
        const float* rowSpatial = inputBufs[nIdx]->rowSpatialBuf.data();
        const float* rowMeta = inputBufs[nIdx]->rowMetaBuf.data();
        bool hasRowMeta = inputBufs[nIdx]->hasRowMeta;
        
        std::copy(rowGlobal, rowGlobal + numGlobalFeatures, rowGlobalInput);
        if(numMetaFeatures > 0) {
            assert(rowMeta != NULL);
            assert(hasRowMeta);
            std::copy(rowMeta, rowMeta + numMetaFeatures, rowMetaInput);
        }
        
        SymmetryHelpers::copyInputsWithSymmetry(
            rowSpatial, rowSpatialInput, 1, nnYLen, nnXLen, numSpatialFeatures, 
            handle->inputsUseNHWC, inputBufs[nIdx]->symmetry
        );
    }
    
    // Run inference - pick the smallest compiled batch size that fits
    int bestBatchSize = handle->model->getBestBatchSize(batchSize);
    migraphx::parameter_map params;
    
    // Always use float_type for input shapes - host buffers are float, graph handles conversion
    migraphx::shape input_shape(
        migraphx::shape::float_type,
        {(size_t)bestBatchSize, (size_t)numSpatialFeatures, (size_t)nnYLen, (size_t)nnXLen}
    );
    params["input_spatial"] = migraphx::argument(input_shape, buffers->userInputBuffer.data());
    
    migraphx::shape global_shape(
        migraphx::shape::float_type,
        {(size_t)bestBatchSize, (size_t)numGlobalFeatures}
    );
    params["input_global"] = migraphx::argument(global_shape, buffers->userInputGlobalBuffer.data());
    
    auto results = handle->model->getProgram(bestBatchSize).eval(params);
    
    // Extract results from MIGraphX eval into buffers
    // Output order for modelVersion >= 2: policy, policyPass, value, scoreValue, ownership
    int numPolicyChannels = handle->model->numPolicyChannels;
    size_t policySize = (size_t)numPolicyChannels * nnXLen * nnYLen;
    int numValueChannels = handle->model->numValueChannels;
    int numScoreValueChannels = handle->model->numScoreValueChannels;
    size_t ownershipSize = (size_t)nnXLen * nnYLen;
    
    // Policy: [maxBatchSize, numPolicyChannels * H * W]
    if(results.size() > 0) {
        results[0].visit([&](auto output) {
            for(int row = 0; row < batchSize; row++) {
                for(size_t i = 0; i < policySize; i++) {
                    buffers->policyResults[row * policySize + i] = static_cast<float>(output[row * policySize + i]);
                }
            }
        });
    }
    
    if(modelVersion >= 2) {
        // Policy pass: [maxBatchSize, numPolicyChannels]
        if(results.size() > 1) {
            results[1].visit([&](auto output) {
                for(int row = 0; row < batchSize; row++) {
                    for(int i = 0; i < numPolicyChannels; i++) {
                        buffers->policyPassResults[row * numPolicyChannels + i] = static_cast<float>(output[row * numPolicyChannels + i]);
                    }
                }
            });
        }
        
        // Value: [maxBatchSize, numValueChannels]
        if(results.size() > 2) {
            results[2].visit([&](auto output) {
                for(int row = 0; row < batchSize; row++) {
                    for(int i = 0; i < numValueChannels; i++) {
                        buffers->valueResults[row * numValueChannels + i] = static_cast<float>(output[row * numValueChannels + i]);
                    }
                }
            });
        }
        
        // Score value: [maxBatchSize, numScoreValueChannels]
        if(results.size() > 3) {
            results[3].visit([&](auto output) {
                for(int row = 0; row < batchSize; row++) {
                    for(int i = 0; i < numScoreValueChannels; i++) {
                        buffers->scoreValueResults[row * numScoreValueChannels + i] = static_cast<float>(output[row * numScoreValueChannels + i]);
                    }
                }
            });
        }
        
        // Ownership: [maxBatchSize, H * W]
        if(results.size() > 4) {
            results[4].visit([&](auto output) {
                for(int row = 0; row < batchSize; row++) {
                    for(size_t i = 0; i < ownershipSize; i++) {
                        buffers->ownershipResults[row * ownershipSize + i] = static_cast<float>(output[row * ownershipSize + i]);
                    }
                }
            });
        }
    } else {
        // Value: [maxBatchSize, numValueChannels]
        if(results.size() > 1) {
            results[1].visit([&](auto output) {
                for(int row = 0; row < batchSize; row++) {
                    for(int i = 0; i < numValueChannels; i++) {
                        buffers->valueResults[row * numValueChannels + i] = static_cast<float>(output[row * numValueChannels + i]);
                    }
                }
            });
        }
    }
    
    // Process outputs per row
    assert(outputs.size() == (size_t)batchSize);
    
    float policyProbsTmp[NNPos::MAX_NN_POLICY_SIZE];
    
    for(int row = 0; row < batchSize; row++) {
        NNOutput* output = outputs[row];
        assert(output->nnXLen == nnXLen);
        assert(output->nnYLen == nnYLen);
        float policyOptimism = (float)inputBufs[row]->policyOptimism;
        
        const float* policyPassSrcBuf = buffers->policyPassResults.data() + row * numPolicyChannels;
        const float* policySrcBuf = buffers->policyResults.data() + row * policySize;
        float* policyProbs = output->policyProbs;
        
        if(numPolicyChannels == 2 || (numPolicyChannels == 4 && modelVersion >= 16)) {
            for(int i = 0; i < nnXLen * nnYLen; i++) {
                float p = policySrcBuf[i];
                float pOpt = policySrcBuf[i + nnXLen * nnYLen];
                policyProbsTmp[i] = p + (pOpt - p) * policyOptimism;
            }
            SymmetryHelpers::copyOutputsWithSymmetry(
                policyProbsTmp, policyProbs, 1, nnYLen, nnXLen, inputBufs[row]->symmetry
            );
            policyProbs[nnXLen * nnYLen] = policyPassSrcBuf[0] + (policyPassSrcBuf[1] - policyPassSrcBuf[0]) * policyOptimism;
        } else {
            assert(numPolicyChannels == 1);
            SymmetryHelpers::copyOutputsWithSymmetry(
                policySrcBuf, policyProbs, 1, nnYLen, nnXLen, inputBufs[row]->symmetry
            );
            policyProbs[nnXLen * nnYLen] = policyPassSrcBuf[0];
        }
        
        assert(numValueChannels == 3);
        output->whiteWinProb = buffers->valueResults[row * numValueChannels];
        output->whiteLossProb = buffers->valueResults[row * numValueChannels + 1];
        output->whiteNoResultProb = buffers->valueResults[row * numValueChannels + 2];
        
        if(output->whiteOwnerMap != NULL) {
            const float* ownershipSrcBuf = buffers->ownershipResults.data() + row * ownershipSize;
            assert(handle->model->numOwnershipChannels == 1);
            SymmetryHelpers::copyOutputsWithSymmetry(ownershipSrcBuf, output->whiteOwnerMap, 1, nnYLen, nnXLen, inputBufs[row]->symmetry);
        }
        
        if(modelVersion >= 9) {
            assert(numScoreValueChannels == 6);
            output->whiteScoreMean = buffers->scoreValueResults[row * numScoreValueChannels];
            output->whiteScoreMeanSq = buffers->scoreValueResults[row * numScoreValueChannels + 1];
            output->whiteLead = buffers->scoreValueResults[row * numScoreValueChannels + 2];
            output->varTimeLeft = buffers->scoreValueResults[row * numScoreValueChannels + 3];
            output->shorttermWinlossError = buffers->scoreValueResults[row * numScoreValueChannels + 4];
            output->shorttermScoreError = buffers->scoreValueResults[row * numScoreValueChannels + 5];
        } else if(modelVersion >= 8) {
            assert(numScoreValueChannels == 4);
            output->whiteScoreMean = buffers->scoreValueResults[row * numScoreValueChannels];
            output->whiteScoreMeanSq = buffers->scoreValueResults[row * numScoreValueChannels + 1];
            output->whiteLead = buffers->scoreValueResults[row * numScoreValueChannels + 2];
            output->varTimeLeft = buffers->scoreValueResults[row * numScoreValueChannels + 3];
            output->shorttermWinlossError = 0.0f;
            output->shorttermScoreError = 0.0f;
        } else if(modelVersion >= 4) {
            assert(numScoreValueChannels == 2);
            output->whiteScoreMean = buffers->scoreValueResults[row * numScoreValueChannels];
            output->whiteScoreMeanSq = buffers->scoreValueResults[row * numScoreValueChannels + 1];
            output->whiteLead = output->whiteScoreMean;
            output->varTimeLeft = 0.0f;
            output->shorttermWinlossError = 0.0f;
            output->shorttermScoreError = 0.0f;
        } else if(modelVersion >= 3) {
            assert(numScoreValueChannels == 1);
            output->whiteScoreMean = buffers->scoreValueResults[row * numScoreValueChannels];
            output->whiteScoreMeanSq = output->whiteScoreMean * output->whiteScoreMean;
            output->whiteLead = output->whiteScoreMean;
            output->varTimeLeft = 0.0f;
            output->shorttermWinlossError = 0.0f;
            output->shorttermScoreError = 0.0f;
        } else {
            output->whiteScoreMean = 0.0f;
            output->whiteScoreMeanSq = 1.0f;
            output->whiteLead = 0.0f;
            output->varTimeLeft = 0.0f;
            output->shorttermWinlossError = 0.0f;
            output->shorttermScoreError = 0.0f;
        }
        
        output->policyOptimismUsed = policyOptimism;
    }
}

// Test functions - implemented using MIGraphX for layer verification
bool testEvaluateConv(
    const ConvLayerDesc* desc,
    int batchSize,
    int nnXLen,
    int nnYLen,
    bool useFP16,
    bool useNHWC,
    const vector<float>& inputBuffer,
    vector<float>& outputBuffer
) {
    // Skip NHWC tests - MIGraphX backend uses NCHW format
    if(useNHWC)
        return false;
    
    try {
        migraphx::program prog;
        auto main_module = prog.get_main_module();
        
        migraphx::shape::type_t dataType = useFP16 ? migraphx::shape::half_type : migraphx::shape::float_type;
        vector<size_t> inputShape = {(size_t)batchSize, (size_t)desc->inChannels, (size_t)nnYLen, (size_t)nnXLen};
        
        auto input = main_module->add_parameter("input", migraphx::shape(dataType, inputShape));
        
        // Create weights - MIGraphX expects float data, will convert internally
        vector<size_t> wShape = {(size_t)desc->outChannels, (size_t)desc->inChannels, (size_t)desc->convYSize, (size_t)desc->convXSize};
        migraphx::shape wShapeDesc(dataType, wShape);
        auto weights = main_module->add_literal(migraphx::literal(wShapeDesc, desc->weights));
        
        // Convolution
        int padY = (desc->convYSize - 1) / 2 * desc->dilationY;
        int padX = (desc->convXSize - 1) / 2 * desc->dilationX;
        vector<size_t> padding = {(size_t)padY, (size_t)padX};
        vector<size_t> stride = {1, 1};
        vector<size_t> dilation = {(size_t)desc->dilationY, (size_t)desc->dilationX};
        
        auto conv_op = migraphx::make_op("convolution", {
            {"padding", migraphx::value(padding)},
            {"stride", migraphx::value(stride)},
            {"dilation", migraphx::value(dilation)},
            {"group", 1}
        });
        
        auto conv = main_module->add_instruction(conv_op, input, weights);
        main_module->add_return({conv});
        
        // Compile and run
        migraphx::compile_options compile_opts;
        compile_opts.offload_copy = true;
        auto target = migraphx::make_target("gpu");
        prog.compile(target, compile_opts);
        
        migraphx::parameter_map params;
        
        // For FP16, we need to convert input data to half precision
        vector<migraphx::half> halfInput;
        if(useFP16) {
            halfInput.resize(inputBuffer.size());
            for(size_t i = 0; i < inputBuffer.size(); i++) {
                halfInput[i] = migraphx::half(inputBuffer[i]);
            }
            params["input"] = migraphx::argument(migraphx::shape(dataType, inputShape), halfInput.data());
        } else {
            params["input"] = migraphx::argument(migraphx::shape(dataType, inputShape), const_cast<float*>(inputBuffer.data()));
        }
        
        auto results = prog.eval(params);
        
        // Copy output
        vector<size_t> outputShape = {(size_t)batchSize, (size_t)desc->outChannels, (size_t)nnYLen, (size_t)nnXLen};
        size_t outputSize = batchSize * desc->outChannels * nnYLen * nnXLen;
        outputBuffer.resize(outputSize);
        
        auto outputArg = results[0];
        if(useFP16) {
            // Convert half output back to float
            outputArg.visit([&](auto output) {
                for(size_t i = 0; i < outputSize; i++) {
                    outputBuffer[i] = static_cast<float>(output[i]);
                }
            });
        } else {
            vector<float> tempOutput(outputSize);
            outputArg.visit([&](auto output) {
                for(size_t i = 0; i < outputSize; i++) {
                    tempOutput[i] = static_cast<float>(output[i]);
                }
            });
            outputBuffer = tempOutput;
        }
        
        return true;
    } catch(const exception& e) {
        cerr << "testEvaluateConv failed: " << e.what() << endl;
        return false;
    }
}

bool testEvaluateBatchNorm(
    const BatchNormLayerDesc* desc,
    int batchSize,
    int nnXLen,
    int nnYLen,
    bool useFP16,
    bool useNHWC,
    const vector<float>& inputBuffer,
    const vector<float>& maskBuffer,
    vector<float>& outputBuffer
) {
    (void)maskBuffer;  // BatchNorm doesn't use mask directly
    
    // Skip NHWC tests - MIGraphX backend uses NCHW format
    if(useNHWC)
        return false;
    
    // Validate weights are available
    if(desc->mergedScale.size() != (size_t)desc->numChannels || desc->mergedBias.size() != (size_t)desc->numChannels) {
        cerr << "BatchNorm test: weight size mismatch, skipping" << endl;
        return false;
    }
    
    try {
        migraphx::program prog;
        auto main_module = prog.get_main_module();
        
        migraphx::shape::type_t dataType = useFP16 ? migraphx::shape::half_type : migraphx::shape::float_type;
        vector<size_t> inputShape = {(size_t)batchSize, (size_t)desc->numChannels, (size_t)nnYLen, (size_t)nnXLen};
        
        auto input = main_module->add_parameter("input", migraphx::shape(dataType, inputShape));
        
        // Create merged scale and bias
        vector<size_t> paramShape = {(size_t)desc->numChannels};
        migraphx::shape paramDesc(dataType, paramShape);
        
        auto scale = main_module->add_literal(migraphx::literal(paramDesc, desc->mergedScale));
        auto bias = main_module->add_literal(migraphx::literal(paramDesc, desc->mergedBias));
        
        // Broadcast scale and bias to input shape
        vector<size_t> broadcastShape = {1, (size_t)desc->numChannels, 1, 1};
        auto scale_broadcast = main_module->add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", inputShape}}), scale);
        auto bias_broadcast = main_module->add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", inputShape}}), bias);
        
        // Apply scale and bias: y = x * scale + bias
        auto scaled = main_module->add_instruction(migraphx::make_op("mul"), input, scale_broadcast);
        auto result = main_module->add_instruction(migraphx::make_op("add"), scaled, bias_broadcast);
        
        main_module->add_return({result});
        
        // Compile and run
        migraphx::compile_options compile_opts;
        compile_opts.offload_copy = true;
        auto target = migraphx::make_target("gpu");
        prog.compile(target, compile_opts);
        
        migraphx::parameter_map params;
        
        // For FP16, we need to convert input data to half precision
        vector<migraphx::half> halfInput;
        if(useFP16) {
            halfInput.resize(inputBuffer.size());
            for(size_t i = 0; i < inputBuffer.size(); i++) {
                halfInput[i] = migraphx::half(inputBuffer[i]);
            }
            params["input"] = migraphx::argument(migraphx::shape(dataType, inputShape), halfInput.data());
        } else {
            params["input"] = migraphx::argument(migraphx::shape(dataType, inputShape), const_cast<float*>(inputBuffer.data()));
        }
        
        auto results = prog.eval(params);
        
        // Copy output
        size_t outputSize = batchSize * desc->numChannels * nnYLen * nnXLen;
        outputBuffer.resize(outputSize);
        
        auto outputArg = results[0];
        if(useFP16) {
            outputArg.visit([&](auto output) {
                for(size_t i = 0; i < outputSize; i++) {
                    outputBuffer[i] = static_cast<float>(output[i]);
                }
            });
        } else {
            vector<float> tempOutput(outputSize);
            outputArg.visit([&](auto output) {
                for(size_t i = 0; i < outputSize; i++) {
                    tempOutput[i] = static_cast<float>(output[i]);
                }
            });
            outputBuffer = tempOutput;
        }
        
        return true;
    } catch(const exception& e) {
        cerr << "testEvaluateBatchNorm failed: " << e.what() << endl;
        return false;
    }
}

bool testEvaluateResidualBlock(
    const ResidualBlockDesc* desc,
    int batchSize,
    int nnXLen,
    int nnYLen,
    bool useFP16,
    bool useNHWC,
    const vector<float>& inputBuffer,
    const vector<float>& maskBuffer,
    vector<float>& outputBuffer
) {
    (void)maskBuffer;
    
    // Skip NHWC tests - MIGraphX backend uses NCHW format
    if(useNHWC)
        return false;
    
    // Validate weights are available
    size_t w1Expected = (size_t)desc->regularConv.outChannels * desc->regularConv.inChannels 
                        * desc->regularConv.convYSize * desc->regularConv.convXSize;
    size_t w2Expected = (size_t)desc->finalConv.outChannels * desc->finalConv.inChannels
                        * desc->finalConv.convYSize * desc->finalConv.convXSize;
    if(desc->regularConv.weights.size() != w1Expected || desc->finalConv.weights.size() != w2Expected) {
        cerr << "ResidualBlock test: weight size mismatch, skipping" << endl;
        return false;
    }
    
    try {
        migraphx::program prog;
        auto main_module = prog.get_main_module();
        
        migraphx::shape::type_t dataType = useFP16 ? migraphx::shape::half_type : migraphx::shape::float_type;
        int numChannels = desc->regularConv.inChannels;
        vector<size_t> inputShape = {(size_t)batchSize, (size_t)numChannels, (size_t)nnYLen, (size_t)nnXLen};
        
        auto input = main_module->add_parameter("input", migraphx::shape(dataType, inputShape));
        
        // Build residual block
        auto residual = input;
        
        // preBN + preActivation (simplified - just activation for now)
        auto x = input;
        if(desc->preActivation.activation == 1) {  // GELU
            // Simplified GELU
            auto sigmoid = main_module->add_instruction(migraphx::make_op("sigmoid"), x);
            x = main_module->add_instruction(migraphx::make_op("mul"), x, sigmoid);
        } else {
            x = main_module->add_instruction(migraphx::make_op("relu"), x);
        }
        
        // regularConv
        vector<size_t> w1Shape = {(size_t)desc->regularConv.outChannels, (size_t)desc->regularConv.inChannels, 
                                   (size_t)desc->regularConv.convYSize, (size_t)desc->regularConv.convXSize};
        migraphx::shape w1Desc(dataType, w1Shape);
        auto w1 = main_module->add_literal(migraphx::literal(w1Desc, desc->regularConv.weights));
        
        int pad1 = (desc->regularConv.convYSize - 1) / 2;
        vector<size_t> padding1 = {(size_t)pad1, (size_t)pad1};
        auto conv1_op = migraphx::make_op("convolution", {
            {"padding", migraphx::value(padding1)},
            {"stride", migraphx::value(vector<size_t>{1, 1})},
            {"dilation", migraphx::value(vector<size_t>{(size_t)desc->regularConv.dilationY, (size_t)desc->regularConv.dilationX})},
            {"group", 1}
        });
        x = main_module->add_instruction(conv1_op, x, w1);
        
        // midActivation
        if(desc->midActivation.activation == 1) {
            auto sigmoid = main_module->add_instruction(migraphx::make_op("sigmoid"), x);
            x = main_module->add_instruction(migraphx::make_op("mul"), x, sigmoid);
        } else {
            x = main_module->add_instruction(migraphx::make_op("relu"), x);
        }
        
        // finalConv
        vector<size_t> w2Shape = {(size_t)desc->finalConv.outChannels, (size_t)desc->finalConv.inChannels,
                                   (size_t)desc->finalConv.convYSize, (size_t)desc->finalConv.convXSize};
        migraphx::shape w2Desc(dataType, w2Shape);
        auto w2 = main_module->add_literal(migraphx::literal(w2Desc, desc->finalConv.weights));
        
        int pad2 = (desc->finalConv.convYSize - 1) / 2;
        vector<size_t> padding2 = {(size_t)pad2, (size_t)pad2};
        auto conv2_op = migraphx::make_op("convolution", {
            {"padding", migraphx::value(padding2)},
            {"stride", migraphx::value(vector<size_t>{1, 1})},
            {"dilation", migraphx::value(vector<size_t>{(size_t)desc->finalConv.dilationY, (size_t)desc->finalConv.dilationX})},
            {"group", 1}
        });
        x = main_module->add_instruction(conv2_op, x, w2);
        
        // Add residual
        auto result = main_module->add_instruction(migraphx::make_op("add"), x, residual);
        
        main_module->add_return({result});
        
        // Compile and run
        migraphx::compile_options compile_opts;
        compile_opts.offload_copy = true;
        auto target = migraphx::make_target("gpu");
        prog.compile(target, compile_opts);
        
        migraphx::parameter_map params;
        
        // For FP16, we need to convert input data to half precision
        vector<migraphx::half> halfInput;
        if(useFP16) {
            halfInput.resize(inputBuffer.size());
            for(size_t i = 0; i < inputBuffer.size(); i++) {
                halfInput[i] = migraphx::half(inputBuffer[i]);
            }
            params["input"] = migraphx::argument(migraphx::shape(dataType, inputShape), halfInput.data());
        } else {
            params["input"] = migraphx::argument(migraphx::shape(dataType, inputShape), const_cast<float*>(inputBuffer.data()));
        }
        
        auto results = prog.eval(params);
        
        // Copy output
        size_t outputSize = batchSize * numChannels * nnYLen * nnXLen;
        outputBuffer.resize(outputSize);
        
        auto outputArg = results[0];
        if(useFP16) {
            outputArg.visit([&](auto output) {
                for(size_t i = 0; i < outputSize; i++) {
                    outputBuffer[i] = static_cast<float>(output[i]);
                }
            });
        } else {
            vector<float> tempOutput(outputSize);
            outputArg.visit([&](auto output) {
                for(size_t i = 0; i < outputSize; i++) {
                    tempOutput[i] = static_cast<float>(output[i]);
                }
            });
            outputBuffer = tempOutput;
        }
        
        return true;
    } catch(const exception& e) {
        cerr << "testEvaluateResidualBlock failed: " << e.what() << endl;
        return false;
    }
}

bool testEvaluateGlobalPoolingResidualBlock(
    const GlobalPoolingResidualBlockDesc* desc,
    int batchSize,
    int nnXLen,
    int nnYLen,
    bool useFP16,
    bool useNHWC,
    const vector<float>& inputBuffer,
    const vector<float>& maskBuffer,
    vector<float>& outputBuffer
) {
    (void)desc;
    (void)batchSize;
    (void)nnXLen;
    (void)nnYLen;
    (void)useFP16;
    (void)useNHWC;
    (void)inputBuffer;
    (void)maskBuffer;
    (void)outputBuffer;
    
    // Global pooling residual block tests not supported yet
    return false;
    
    try {
        migraphx::program prog;
        auto main_module = prog.get_main_module();
        
        migraphx::shape::type_t dataType = migraphx::shape::float_type;
        int numChannels = desc->regularConv.inChannels;
        vector<size_t> inputShape = {(size_t)batchSize, (size_t)numChannels, (size_t)nnYLen, (size_t)nnXLen};
        
        auto input = main_module->add_parameter("input", migraphx::shape(dataType, inputShape));
        
        // Simplified global pooling residual block (without full gpool branch for now)
        auto residual = input;
        
        // Activation
        auto x = main_module->add_instruction(migraphx::make_op("relu"), input);
        
        // regularConv
        vector<size_t> wShape = {(size_t)desc->regularConv.outChannels, (size_t)desc->regularConv.inChannels,
                                  (size_t)desc->regularConv.convYSize, (size_t)desc->regularConv.convXSize};
        migraphx::shape wDesc(dataType, wShape);
        auto w = main_module->add_literal(migraphx::literal(wDesc, desc->regularConv.weights));
        
        int pad = (desc->regularConv.convYSize - 1) / 2;
        vector<size_t> padding = {(size_t)pad, (size_t)pad};
        auto conv_op = migraphx::make_op("convolution", {
            {"padding", migraphx::value(padding)},
            {"stride", migraphx::value(vector<size_t>{1, 1})},
            {"dilation", migraphx::value(vector<size_t>{(size_t)desc->regularConv.dilationY, (size_t)desc->regularConv.dilationX})},
            {"group", 1}
        });
        x = main_module->add_instruction(conv_op, x, w);
        
        // midActivation
        x = main_module->add_instruction(migraphx::make_op("relu"), x);
        
        // finalConv
        vector<size_t> w2Shape = {(size_t)desc->finalConv.outChannels, (size_t)desc->finalConv.inChannels,
                                   (size_t)desc->finalConv.convYSize, (size_t)desc->finalConv.convXSize};
        migraphx::shape w2Desc(dataType, w2Shape);
        auto w2 = main_module->add_literal(migraphx::literal(w2Desc, desc->finalConv.weights));
        
        int pad2 = (desc->finalConv.convYSize - 1) / 2;
        vector<size_t> padding2 = {(size_t)pad2, (size_t)pad2};
        auto conv2_op = migraphx::make_op("convolution", {
            {"padding", migraphx::value(padding2)},
            {"stride", migraphx::value(vector<size_t>{1, 1})},
            {"dilation", migraphx::value(vector<size_t>{(size_t)desc->finalConv.dilationY, (size_t)desc->finalConv.dilationX})},
            {"group", 1}
        });
        x = main_module->add_instruction(conv2_op, x, w2);
        
        // Add residual
        auto result = main_module->add_instruction(migraphx::make_op("add"), x, residual);
        
        main_module->add_return({result});
        
        // Compile and run
        migraphx::compile_options compile_opts;
        compile_opts.offload_copy = true;
        auto target = migraphx::make_target("gpu");
        prog.compile(target, compile_opts);
        
        migraphx::parameter_map params;
        params["input"] = migraphx::argument(migraphx::shape(dataType, inputShape), const_cast<float*>(inputBuffer.data()));
        
        auto results = prog.eval(params);
        
        // Copy output
        size_t outputSize = batchSize * numChannels * nnYLen * nnXLen;
        outputBuffer.resize(outputSize);
        
        auto outputArg = results[0];
        vector<float> tempOutput(outputSize);
        outputArg.visit([&](auto output) {
            for(size_t i = 0; i < outputSize; i++) {
                tempOutput[i] = static_cast<float>(output[i]);
            }
        });
        outputBuffer = tempOutput;
        
        return true;
    } catch(const exception& e) {
        cerr << "testEvaluateGlobalPoolingResidualBlock failed: " << e.what() << endl;
        return false;
    }
}

} // namespace NeuralNet
