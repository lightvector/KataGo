#include "../neuralnet/nninterface.h"
#include "../neuralnet/nninputs.h"
#include "../neuralnet/nneval.h"
#include "../neuralnet/modelversion.h"
#include "../neuralnet/desc.h"
#include "../neuralnet/sgfmetadata.h"
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

#include <hip/hip_runtime.h>

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
// - Complete residual network structure (ordinary, global-pooling, nested-bottleneck blocks)
// - Full BatchNorm support via multibroadcast
// - FP16 support (configurable via useFP16Mode)
// - Input/output tensor handling
// - Working inference with MIGraphX GPU backend
// - Disk cache for compiled programs (keyed by model hash, board size, batch size, FP16, MIGraphX version)
//
// Known Limitations:
// - No dynamic batch size support (multiple static programs compiled per batch size)
// - Global pooling assumes full board (requireExactNNLen required for non-full boards)
// - SGF metadata encoder not supported
//
//------------------------ MIGraphX Model Implementation ------------------------

static constexpr int MAX_CHANNELS_SANITY_CHECK = 10000;

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
        // Validate dimensions: KataGo only uses odd-sized kernels (1x1, 3x3, 5x5)
        if(convDesc.inChannels <= 0 || convDesc.inChannels > MAX_CHANNELS_SANITY_CHECK ||
           convDesc.outChannels <= 0 || convDesc.outChannels > MAX_CHANNELS_SANITY_CHECK ||
           convDesc.convYSize <= 0 || convDesc.convYSize > 9 ||
           convDesc.convXSize <= 0 || convDesc.convXSize > 9)
            throw StringError(
                "Conv " + convDesc.name + " has invalid dimensions (in=" + Global::intToString(convDesc.inChannels) +
                ", out=" + Global::intToString(convDesc.outChannels) +
                ", ky=" + Global::intToString(convDesc.convYSize) +
                ", kx=" + Global::intToString(convDesc.convXSize) + ")"
            );
        if(convDesc.convYSize % 2 == 0 || convDesc.convXSize % 2 == 0)
            throw StringError(
                "Conv " + convDesc.name + " has even kernel size (ky=" + Global::intToString(convDesc.convYSize) +
                ", kx=" + Global::intToString(convDesc.convXSize) +
                "); only odd kernel sizes are supported (SAME padding is undefined for even kernels)"
            );

        vector<size_t> wShape = {
            (size_t)convDesc.outChannels,
            (size_t)convDesc.inChannels,
            (size_t)convDesc.convYSize,
            (size_t)convDesc.convXSize
        };
        size_t expectedWeights = (size_t)convDesc.outChannels * (size_t)convDesc.inChannels
                                 * (size_t)convDesc.convYSize * (size_t)convDesc.convXSize;

        if(convDesc.weights.size() != expectedWeights)
            throw StringError(
                "Conv " + convDesc.name + " weights size mismatch: " +
                Global::uint64ToString(convDesc.weights.size()) + " vs expected " + Global::uint64ToString(expectedWeights) +
                " (out=" + Global::intToString(convDesc.outChannels) +
                ", in=" + Global::intToString(convDesc.inChannels) +
                ", ky=" + Global::intToString(convDesc.convYSize) +
                ", kx=" + Global::intToString(convDesc.convXSize) + ")"
            );
        
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
        if(bnDesc.numChannels <= 0 || bnDesc.numChannels > MAX_CHANNELS_SANITY_CHECK)
            throw StringError(
                "BatchNorm " + bnDesc.name + " has invalid numChannels=" + Global::intToString(bnDesc.numChannels)
            );

        int numChannels = bnDesc.numChannels;

        if(bnDesc.mergedScale.size() != (size_t)numChannels || bnDesc.mergedBias.size() != (size_t)numChannels)
            throw StringError(
                "BatchNorm " + bnDesc.name + " weight size mismatch (C=" + Global::intToString(numChannels) +
                ", scale=" + Global::uint64ToString(bnDesc.mergedScale.size()) +
                ", bias=" + Global::uint64ToString(bnDesc.mergedBias.size()) + ")"
            );
        
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
        if(matmulDesc.inChannels <= 0 || matmulDesc.inChannels > MAX_CHANNELS_SANITY_CHECK ||
           matmulDesc.outChannels <= 0 || matmulDesc.outChannels > MAX_CHANNELS_SANITY_CHECK)
            throw StringError(
                "MatMul " + matmulDesc.name + " has invalid channels (in=" + Global::intToString(matmulDesc.inChannels) +
                ", out=" + Global::intToString(matmulDesc.outChannels) + ")"
            );

        vector<size_t> wShape = {(size_t)matmulDesc.inChannels, (size_t)matmulDesc.outChannels};
        size_t expectedWeights = (size_t)matmulDesc.inChannels * (size_t)matmulDesc.outChannels;
        if(matmulDesc.weights.size() != expectedWeights)
            throw StringError(
                "MatMul " + matmulDesc.name + " weights size mismatch: " +
                Global::uint64ToString(matmulDesc.weights.size()) + " vs expected " + Global::uint64ToString(expectedWeights) +
                " (in=" + Global::intToString(matmulDesc.inChannels) +
                ", out=" + Global::intToString(matmulDesc.outChannels) + ")"
            );
        auto weights = addLiteral(matmulDesc.weights, wShape);
        
        auto matmul = main_module->add_instruction(migraphx::make_op("dot"), input, weights);
        
        if(biasDesc != nullptr && !biasDesc->weights.empty()) {
            if(biasDesc->weights.size() != (size_t)biasDesc->numChannels)
                throw StringError(
                    "MIGraphX: MatMul bias " + biasDesc->name + " size mismatch: " +
                    Global::uint64ToString(biasDesc->weights.size()) + " vs expected " +
                    Global::intToString(biasDesc->numChannels)
                );
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
    
    // Mish activation: x * tanh(softplus(x))
    // Uses numerically stable softplus: max(x,0) + log1p(exp(-|x|))
    // This avoids exp overflow for large positive x (since -|x| <= 0 so exp(-|x|) <= 1)
    migraphx::instruction_ref addMish(migraphx::instruction_ref input) {
        auto inputLens = input->get_shape().lens();
        // softplus(x) = max(x,0) + log(1 + exp(-|x|))  — numerically stable for all x
        auto abs_x = main_module->add_instruction(migraphx::make_op("abs"), input);
        auto neg_abs_x = main_module->add_instruction(migraphx::make_op("neg"), abs_x);
        auto exp_neg_abs = main_module->add_instruction(migraphx::make_op("exp"), neg_abs_x);
        auto ones = broadcastScalar(1.0f, inputLens);
        auto one_plus_exp_neg_abs = main_module->add_instruction(migraphx::make_op("add"), exp_neg_abs, ones);
        auto log_part = main_module->add_instruction(migraphx::make_op("log"), one_plus_exp_neg_abs);
        auto relu_x = main_module->add_instruction(migraphx::make_op("relu"), input);
        auto softplus = main_module->add_instruction(migraphx::make_op("add"), relu_x, log_part);
        auto tanh_sp = main_module->add_instruction(migraphx::make_op("tanh"), softplus);
        return main_module->add_instruction(migraphx::make_op("mul"), input, tanh_sp);
    }

    // Mish-scale8 activation: x * tanh(softplus(8x))
    // Uses numerically stable softplus: max(8x,0) + log1p(exp(-|8x|))
    // Safe for both FP32 and FP16 since exp argument is always <= 0.
    migraphx::instruction_ref addMishScale8(migraphx::instruction_ref input) {
        auto inputLens = input->get_shape().lens();
        auto eight = broadcastScalar(8.0f, inputLens);
        auto scaled = main_module->add_instruction(migraphx::make_op("mul"), input, eight);
        // softplus(scaled) = max(scaled,0) + log(1 + exp(-|scaled|))  — numerically stable
        auto abs_scaled = main_module->add_instruction(migraphx::make_op("abs"), scaled);
        auto neg_abs_scaled = main_module->add_instruction(migraphx::make_op("neg"), abs_scaled);
        auto exp_neg_abs = main_module->add_instruction(migraphx::make_op("exp"), neg_abs_scaled);
        auto ones = broadcastScalar(1.0f, inputLens);
        auto one_plus_exp = main_module->add_instruction(migraphx::make_op("add"), exp_neg_abs, ones);
        auto log_part = main_module->add_instruction(migraphx::make_op("log"), one_plus_exp);
        auto relu_scaled = main_module->add_instruction(migraphx::make_op("relu"), scaled);
        auto softplus = main_module->add_instruction(migraphx::make_op("add"), relu_scaled, log_part);
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
    
    if(useNHWC)
        throw StringError("MIGraphX backend: useNHWC = false required, NHWC format is not supported");
    
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
    if(trunkDesc.initialConv.inChannels != numSpatialFeatures)
        throw StringError(
            "MIGraphX: initialConv input channels mismatch: expected " + Global::intToString(numSpatialFeatures) +
            " but got " + Global::intToString(trunkDesc.initialConv.inChannels)
        );
    trunk = builder.addConv(trunk, trunkDesc.initialConv);

    // Initial MatMul for global features
    {
        if(trunkDesc.initialMatMul.inChannels != numGlobalFeatures)
            throw StringError(
                "MIGraphX: initialMatMul input channels mismatch: expected " +
                Global::intToString(numGlobalFeatures) + " but got " +
                Global::intToString(trunkDesc.initialMatMul.inChannels)
            );
        auto globalProcessed = builder.addMatMul(inputGlobal, trunkDesc.initialMatMul);
        auto trunkShape = trunk->get_shape().lens();
        auto globalUnsqueezed = main_module->add_instruction(
            migraphx::make_op("unsqueeze", {{"axes", migraphx::value(vector<int64_t>{2, 3})}}), globalProcessed);
        auto globalBroadcast = main_module->add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", trunkShape}}), globalUnsqueezed);
        trunk = main_module->add_instruction(migraphx::make_op("add"), trunk, globalBroadcast);
    }

    // SGF Metadata encoder is not supported
    if(trunkDesc.metaEncoderVersion > 0 && numMetaFeatures > 0)
        throw StringError(
            "MIGraphX backend does not support SGF metadata encoder (metaEncoderVersion=" +
            Global::intToString(trunkDesc.metaEncoderVersion) + ")"
        );
    
    // Residual blocks using the stack builder
    trunk = buildResidualBlockStack(builder, trunk, trunkDesc.blocks, "trunk");
    
    // trunkTipBN + trunkTipActivation
    trunk = builder.addBatchNorm(trunk, trunkDesc.trunkTipBN);
    trunk = builder.addActivation(trunk, trunkDesc.trunkTipActivation.activation);
    
    // ======== Policy Head ========
    const PolicyHeadDesc& policyDesc = modelDesc.policyHead;
    
    if(policyDesc.p1Conv.outChannels <= 0)
        throw StringError("MIGraphX: policy head p1Conv has no output channels");

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
    auto policy = main_module->add_instruction(migraphx::make_op("add"), p1Conv, biasBroadcast);

    policy = builder.addBatchNorm(policy, policyDesc.p1BN);
    policy = builder.addActivation(policy, policyDesc.p1Activation.activation);
    policy = builder.addConv(policy, policyDesc.p2Conv);
    policy = builder.addFlatten(policy);

    // Pass policy path
    auto policyPass = builder.addMatMul(gpool, policyDesc.gpoolToPassMul, &policyDesc.gpoolToPassBias);
    policyPass = builder.addActivation(policyPass, policyDesc.passActivation.activation);
    if(policyDesc.gpoolToPassMul2.outChannels > 0)
        policyPass = builder.addMatMul(policyPass, policyDesc.gpoolToPassMul2);
    
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
    
    main_module->add_return({policy, policyPass, valueOut, scoreValue, ownership});
    
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

// Generate cache file path. Returns empty string if caching should be disabled
// (e.g. MIGraphX version macros are not available, in which case different MIGraphX
// installs would collide on the same key and risk loading incompatible binaries).
static string getCacheFilePath(
    const string& homeDataDir,
    const ModelDesc& modelDesc,
    int nnXLen,
    int nnYLen,
    int maxBatchSize,
    bool useFP16,
    bool useNHWC,
    bool requireExactNNLen,
    int gpuIdx,
    Logger* logger
) {
    (void)useNHWC;

    // Cache key includes MIGraphX version to invalidate when the compiler changes.
    // If the version macros are missing, we cannot safely key the cache; skip it.
#if defined(MIGRAPHX_VERSION_MAJOR) && defined(MIGRAPHX_VERSION_MINOR) && defined(MIGRAPHX_VERSION_PATCH)
    string migraphxVersionStr = Global::strprintf("%d_%d_%d", MIGRAPHX_VERSION_MAJOR, MIGRAPHX_VERSION_MINOR, MIGRAPHX_VERSION_PATCH);
#else
    if(logger)
        logger->write("MIGraphX: version macros (MIGRAPHX_VERSION_MAJOR/MINOR/PATCH) not defined; compiled-program cache disabled");
    return "";
#endif

    // Include GPU architecture (e.g. gfx1100) in the key so that cached binaries
    // built for one architecture are not loaded onto an incompatible one.
    hipDeviceProp_t props;
    hipError_t err = hipGetDeviceProperties(&props, gpuIdx);
    if(err != hipSuccess) {
        if(logger)
            logger->write(
                "MIGraphX: hipGetDeviceProperties failed for GPU " + Global::intToString(gpuIdx) +
                " (" + string(hipGetErrorString(err)) + "); compiled-program cache disabled"
            );
        return "";
    }
    string archName = props.gcnArchName;

    auto cacheDir = HomeData::getHomeDataDir(true, homeDataDir);
    cacheDir += "/migraphxcache";

    // Create directory if not exists
    MakeDir::make(cacheDir);

    string cacheKey = Global::strprintf(
        "migraphx%s_%s_%s_%s_%dx%d_batch%d_fp%d_%s",
        migraphxVersionStr.c_str(),
        archName.c_str(),
        modelDesc.name.c_str(),
        modelDesc.sha256.substr(0, 16).c_str(),
        nnYLen,
        nnXLen,
        maxBatchSize,
        useFP16 ? 1 : 0,
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

    if(useNHWC)
        throw StringError("MIGraphX backend: useNHWC = false required, NHWC format is not supported");
    if(inputsUseNHWC)
        throw StringError("MIGraphX backend: inputsUseNHWC = false required, NHWC format is not supported");

    handle->model = make_unique<MIGraphXModel>();
    handle->model->modelVersion = model->modelDesc.modelVersion;
    handle->model->maxBatchSize = maxBatchSize;
    handle->model->nnXLen = ctx->nnXLen;
    handle->model->nnYLen = ctx->nnYLen;
    handle->model->useFP16 = useFP16;
    handle->model->useNHWC = false;

    handle->model->numInputChannels = model->modelDesc.numInputChannels;
    handle->model->numInputGlobalChannels = model->modelDesc.numInputGlobalChannels;
    handle->model->numInputMetaChannels = model->modelDesc.numInputMetaChannels;
    handle->model->numPolicyChannels = model->modelDesc.numPolicyChannels;
    handle->model->numValueChannels = model->modelDesc.numValueChannels;
    handle->model->numScoreValueChannels = model->modelDesc.numScoreValueChannels;
    handle->model->numOwnershipChannels = model->modelDesc.numOwnershipChannels;

    vector<int> batchSizesToCompile = generateBatchSizes(maxBatchSize);
    handle->model->batchSizes = batchSizesToCompile;
    handle->model->tgt = migraphx::make_target("gpu");

    lock_guard<mutex> cacheLock(migraphxCacheMutex);

    for(int bs : batchSizesToCompile) {
        string cacheFile = getCacheFilePath(
            ctx->homeDataDir,
            model->modelDesc,
            ctx->nnXLen,
            ctx->nnYLen,
            bs,
            useFP16,
            useNHWC,
            requireExactNNLen,
            gpuIdxForThisThread,
            logger
        );

        bool cacheLoaded = false;

        if(!cacheFile.empty() && FileUtils::exists(cacheFile)) {
            try {
                if(logger)
                    logger->write("MIGraphX: Loading compiled program from cache (batch " + Global::intToString(bs) + "): " + cacheFile);
                handle->model->progs[bs] = migraphx::load(cacheFile);
                cacheLoaded = true;
                if(logger)
                    logger->write("MIGraphX: Batch " + Global::intToString(bs) + " loaded from cache (FP16: " + string(useFP16 ? "yes" : "no") + ")");
            } catch(const exception& e) {
                if(logger)
                    logger->write("MIGraphX: Cache load failed for batch " + Global::intToString(bs) + ": " + e.what() + " — rebuilding");
            }
        }

        if(!cacheLoaded) {
            if(logger) {
                logger->write(
                    "MIGraphX: Building model (version " + Global::intToString(model->modelDesc.modelVersion) + ")"
                    " board=" + Global::intToString(ctx->nnXLen) + "x" + Global::intToString(ctx->nnYLen) +
                    " batch=" + Global::intToString(bs) +
                    " fp16=" + string(useFP16 ? "yes" : "no") +
                    " trunk_ch=" + Global::intToString(model->modelDesc.trunk.trunkNumChannels) +
                    " blocks=" + Global::intToString(model->modelDesc.trunk.numBlocks)
                );
            }

            handle->model->progs[bs] = buildMIGraphXProgram(
                model->modelDesc,
                bs,
                ctx->nnXLen,
                ctx->nnYLen,
                useFP16,
                useNHWC
            );

            if(logger)
                logger->write("MIGraphX: Compiling batch " + Global::intToString(bs) + "...");
            migraphx::compile_options compile_opts;
            compile_opts.offload_copy = true;
            handle->model->progs[bs].compile(handle->model->tgt, compile_opts);
            if(logger)
                logger->write("MIGraphX: Batch " + Global::intToString(bs) + " compiled");

            // Save to cache using a temp file + atomic rename to avoid corruption from concurrent writes
            if(!cacheFile.empty()) {
                try {
                    string tmpFile = cacheFile + ".tmp";
                    migraphx::save(handle->model->progs[bs], tmpFile);
                    if(std::rename(tmpFile.c_str(), cacheFile.c_str()) != 0)
                        throw StringError("rename failed");
                    if(logger)
                        logger->write("MIGraphX: Saved compiled program to cache: " + cacheFile);
                } catch(const exception& e) {
                    if(logger)
                        logger->write("MIGraphX: Cache save failed (non-fatal): " + string(e.what()));
                }
            }
        }
    }

    if(logger) {
        string batchList;
        for(size_t i = 0; i < batchSizesToCompile.size(); i++) {
            if(i > 0) batchList += ", ";
            batchList += Global::intToString(batchSizesToCompile[i]);
        }
        logger->write("MIGraphX: All " + Global::uint64ToString(batchSizesToCompile.size()) + " batch sizes ready: " + batchList);
    }

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
    
    // Output order: policy[0], policyPass[1], value[2], scoreValue[3], ownership[4]
    assert(results.size() >= 5);
    results[1].visit([&](auto output) {
        for(int row = 0; row < batchSize; row++) {
            for(int i = 0; i < numPolicyChannels; i++) {
                buffers->policyPassResults[row * numPolicyChannels + i] = static_cast<float>(output[row * numPolicyChannels + i]);
            }
        }
    });
    results[2].visit([&](auto output) {
        for(int row = 0; row < batchSize; row++) {
            for(int i = 0; i < numValueChannels; i++) {
                buffers->valueResults[row * numValueChannels + i] = static_cast<float>(output[row * numValueChannels + i]);
            }
        }
    });
    results[3].visit([&](auto output) {
        for(int row = 0; row < batchSize; row++) {
            for(int i = 0; i < numScoreValueChannels; i++) {
                buffers->scoreValueResults[row * numScoreValueChannels + i] = static_cast<float>(output[row * numScoreValueChannels + i]);
            }
        }
    });
    results[4].visit([&](auto output) {
        for(int row = 0; row < batchSize; row++) {
            for(size_t i = 0; i < ownershipSize; i++) {
                buffers->ownershipResults[row * ownershipSize + i] = static_cast<float>(output[row * ownershipSize + i]);
            }
        }
    });
    
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
            ASSERT_UNREACHABLE;
        }
        
        output->policyOptimismUsed = policyOptimism;
    }
}

// Test functions - implemented using MIGraphX for layer verification.
// These exercise the SAME graph-construction code paths used by the production
// inference path (MIGraphXGraphBuilder, buildResidualBlock, ...), so that a passing
// test gives meaningful coverage of what actually runs at inference time.
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

        MIGraphXGraphBuilder builder(main_module, dataType, batchSize, nnXLen, nnYLen);
        auto conv = builder.addConv(input, *desc);
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

    try {
        migraphx::program prog;
        auto main_module = prog.get_main_module();

        migraphx::shape::type_t dataType = useFP16 ? migraphx::shape::half_type : migraphx::shape::float_type;
        vector<size_t> inputShape = {(size_t)batchSize, (size_t)desc->numChannels, (size_t)nnYLen, (size_t)nnXLen};

        auto input = main_module->add_parameter("input", migraphx::shape(dataType, inputShape));

        MIGraphXGraphBuilder builder(main_module, dataType, batchSize, nnXLen, nnYLen);
        auto result = builder.addBatchNorm(input, *desc);

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

    try {
        migraphx::program prog;
        auto main_module = prog.get_main_module();

        migraphx::shape::type_t dataType = useFP16 ? migraphx::shape::half_type : migraphx::shape::float_type;
        int numChannels = desc->regularConv.inChannels;
        vector<size_t> inputShape = {(size_t)batchSize, (size_t)numChannels, (size_t)nnYLen, (size_t)nnXLen};

        auto input = main_module->add_parameter("input", migraphx::shape(dataType, inputShape));

        // Build the residual block using the exact same code path used at inference.
        MIGraphXGraphBuilder builder(main_module, dataType, batchSize, nnXLen, nnYLen);
        auto result = buildResidualBlock(builder, input, *desc);

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
    (void)maskBuffer;

    // Skip NHWC tests - MIGraphX backend uses NCHW format
    if(useNHWC)
        return false;

    try {
        migraphx::program prog;
        auto main_module = prog.get_main_module();

        migraphx::shape::type_t dataType = useFP16 ? migraphx::shape::half_type : migraphx::shape::float_type;
        int numChannels = desc->regularConv.inChannels;
        vector<size_t> inputShape = {(size_t)batchSize, (size_t)numChannels, (size_t)nnYLen, (size_t)nnXLen};

        auto input = main_module->add_parameter("input", migraphx::shape(dataType, inputShape));

        // Build the global pooling residual block using the same code path as inference.
        MIGraphXGraphBuilder builder(main_module, dataType, batchSize, nnXLen, nnYLen);
        auto result = buildGlobalPoolingResidualBlock(builder, input, *desc);

        main_module->add_return({result});

        // Compile and run
        migraphx::compile_options compile_opts;
        compile_opts.offload_copy = true;
        auto target = migraphx::make_target("gpu");
        prog.compile(target, compile_opts);

        migraphx::parameter_map params;

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
        outputArg.visit([&](auto output) {
            for(size_t i = 0; i < outputSize; i++) {
                outputBuffer[i] = static_cast<float>(output[i]);
            }
        });

        return true;
    } catch(const exception& e) {
        cerr << "testEvaluateGlobalPoolingResidualBlock failed: " << e.what() << endl;
        return false;
    }
}

} // namespace NeuralNet
