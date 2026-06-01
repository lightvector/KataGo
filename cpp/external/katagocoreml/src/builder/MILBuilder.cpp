// katagocoreml - Standalone C++ KataGo to Core ML Converter
// Copyright (c) 2025, Chin-Chang Yang

#include "MILBuilder.hpp"
#include "MILBlob/Fp16.hpp"
#include <stdexcept>
#include <cmath>

// Include generated protobuf headers
#include "MIL.pb.h"

namespace katagocoreml {

namespace {
// RAII: set a dtype slot to FLOAT32 for the current scope and restore it on exit. Used to emit a
// sub-region of ops in FP32 inside an otherwise-FP16 model.
struct ScopedFp32 {
    CoreML::Specification::MILSpec::DataType& slot;
    CoreML::Specification::MILSpec::DataType saved;
    explicit ScopedFp32(CoreML::Specification::MILSpec::DataType& s)
        : slot(s), saved(s) { s = CoreML::Specification::MILSpec::DataType::FLOAT32; }
    ~ScopedFp32() { slot = saved; }
    ScopedFp32(const ScopedFp32&) = delete;
    ScopedFp32& operator=(const ScopedFp32&) = delete;
};
}  // namespace

MILBuilder::MILBuilder(const KataGoModelDesc& model,
                       int board_x_size,
                       int board_y_size,
                       bool optimize_identity_mask,
                       bool use_fp16,
                       int min_batch_size,
                       int max_batch_size,
                       bool use_fp16_io)
    : m_model(model)
    , m_board_x_size(board_x_size)
    , m_board_y_size(board_y_size)
    , m_optimize_identity_mask(optimize_identity_mask)
    , m_use_fp16(use_fp16)
    , m_use_fp16_io(use_fp16_io)
    , m_min_batch_size(min_batch_size)
    , m_max_batch_size(max_batch_size)
    , m_weight_dtype(use_fp16
          ? CoreML::Specification::MILSpec::DataType::FLOAT16
          : CoreML::Specification::MILSpec::DataType::FLOAT32)
    , m_ops(board_x_size, board_y_size, optimize_identity_mask)
    , m_var_counter(0) {
    // Precision tiers in FP16 mode (the ANE accumulates FP16 in FP16; FP32 ops run off the FP16-only
    // ANE). NARROW transformer trunks are unreliable on the FP16 ANE: their policy/value metrics sit
    // right on the testgpuerror thresholds and no partial-FP32 config passes all board sizes (partial
    // FP32 leaves a noisy FP16 spatial stream). So build narrow trunks FULLY in FP32 (off-ANE, but
    // cheap since narrow models are small; correct because it equals the FP32 reference). Weights are
    // stored FP32 via per-weight serialization. Wider trunks use partial FP32: non-spatial (matmuls +
    // pooling) always FP32; convs FP32 only for very wide trunks (kept on the ANE for narrower ones).
    const int trunkChannels = model.trunk.trunk_num_channels;
    const bool full_fp32 = use_fp16 && trunkChannels < FULL_FP32_MAX_TRUNK_CHANNELS;
    if (full_fp32) {
        m_use_fp16 = false;
        m_use_fp16_io = false;
        m_weight_dtype = CoreML::Specification::MILSpec::DataType::FLOAT32;
    }
    m_nonspatial_fp32 = m_use_fp16;
    m_conv_fp32 = m_use_fp16 && trunkChannels >= CONV_FP32_MIN_TRUNK_CHANNELS;
}

void MILBuilder::setBatchDimension(CoreML::Specification::MILSpec::TensorType* tensor_type) {
    auto* dim = tensor_type->add_dimensions();
    if (m_min_batch_size == m_max_batch_size && m_max_batch_size > 0) {
        // Fixed batch size
        dim->mutable_constant()->set_size(m_min_batch_size);
    } else {
        // Dynamic batch size - use UnknownDimension
        dim->mutable_unknown()->set_variadic(false);
    }
}

std::string MILBuilder::genVarName(const std::string& prefix) {
    return prefix + "_" + std::to_string(m_var_counter++);
}

std::unique_ptr<CoreML::Specification::MILSpec::Program> MILBuilder::build() {
    auto program = std::make_unique<CoreML::Specification::MILSpec::Program>();
    program->set_version(1);

    // Create main function
    auto& functions = *program->mutable_functions();
    auto& main_func = functions["main"];
    main_func.set_opset("CoreML5");

    // Create main block
    auto& blocks = *main_func.mutable_block_specializations();
    auto& main_block = blocks["CoreML5"];

    // Define inputs
    // spatial_input: [batch, num_input_ch, board_y, board_x]
    auto* spatial_input = main_func.add_inputs();
    spatial_input->set_name("spatial_input");
    auto* spatial_type = spatial_input->mutable_type()->mutable_tensortype();
    spatial_type->set_datatype(m_use_fp16 && m_use_fp16_io
        ? CoreML::Specification::MILSpec::DataType::FLOAT16
        : CoreML::Specification::MILSpec::DataType::FLOAT32);
    spatial_type->set_rank(4);
    setBatchDimension(spatial_type);
    spatial_type->add_dimensions()->mutable_constant()->set_size(m_model.num_input_channels);
    spatial_type->add_dimensions()->mutable_constant()->set_size(m_board_y_size);
    spatial_type->add_dimensions()->mutable_constant()->set_size(m_board_x_size);

    // global_input: [batch, num_global_ch]
    auto* global_input = main_func.add_inputs();
    global_input->set_name("global_input");
    auto* global_type = global_input->mutable_type()->mutable_tensortype();
    global_type->set_datatype(m_use_fp16 && m_use_fp16_io
        ? CoreML::Specification::MILSpec::DataType::FLOAT16
        : CoreML::Specification::MILSpec::DataType::FLOAT32);
    global_type->set_rank(2);
    setBatchDimension(global_type);
    global_type->add_dimensions()->mutable_constant()->set_size(m_model.num_input_global_channels);

    // input_mask: [batch, 1, board_y, board_x]
    auto* mask_input = main_func.add_inputs();
    mask_input->set_name("input_mask");
    auto* mask_type = mask_input->mutable_type()->mutable_tensortype();
    mask_type->set_datatype(m_use_fp16 && m_use_fp16_io
        ? CoreML::Specification::MILSpec::DataType::FLOAT16
        : CoreML::Specification::MILSpec::DataType::FLOAT32);
    mask_type->set_rank(4);
    setBatchDimension(mask_type);
    mask_type->add_dimensions()->mutable_constant()->set_size(1);
    mask_type->add_dimensions()->mutable_constant()->set_size(m_board_y_size);
    mask_type->add_dimensions()->mutable_constant()->set_size(m_board_x_size);

    // Optional meta_input for human SL networks
    std::string meta_input_name;
    if (m_model.meta_encoder_version > 0 && m_model.num_input_meta_channels > 0) {
        auto* meta_input = main_func.add_inputs();
        meta_input->set_name("meta_input");
        auto* meta_type = meta_input->mutable_type()->mutable_tensortype();
        meta_type->set_datatype(m_use_fp16 && m_use_fp16_io
            ? CoreML::Specification::MILSpec::DataType::FLOAT16
            : CoreML::Specification::MILSpec::DataType::FLOAT32);
        meta_type->set_rank(2);
        setBatchDimension(meta_type);
        meta_type->add_dimensions()->mutable_constant()->set_size(m_model.num_input_meta_channels);
        meta_input_name = "meta_input";
    }

    // For FP16 mode with FP32 I/O, add cast operations after inputs
    std::string spatial_name = "spatial_input";
    std::string global_name = "global_input";
    std::string mask_name = "input_mask";
    std::string meta_name = meta_input_name;

    if (m_use_fp16 && !m_use_fp16_io) {
        // Cast spatial_input: [1, num_input_ch, H, W] fp32 -> fp16
        addCastOp(&main_block, "spatial_input", "spatial_input_cast_fp16", "fp16",
                  {1, m_model.num_input_channels, m_board_y_size, m_board_x_size});
        spatial_name = "spatial_input_cast_fp16";

        // Cast global_input: [1, num_global_ch] fp32 -> fp16
        addCastOp(&main_block, "global_input", "global_input_cast_fp16", "fp16",
                  {1, m_model.num_input_global_channels});
        global_name = "global_input_cast_fp16";

        // Cast input_mask: [1, 1, H, W] fp32 -> fp16
        addCastOp(&main_block, "input_mask", "input_mask_cast_fp16", "fp16",
                  {1, 1, m_board_y_size, m_board_x_size});
        mask_name = "input_mask_cast_fp16";

        // Cast meta_input if present
        if (!meta_input_name.empty()) {
            addCastOp(&main_block, "meta_input", "meta_input_cast_fp16", "fp16",
                      {1, m_model.num_input_meta_channels});
            meta_name = "meta_input_cast_fp16";
        }
    }

    // Build the network
    const std::string* meta_ptr = meta_name.empty() ? nullptr : &meta_name;
    std::string trunk_out = buildTrunk(&main_block, spatial_name, global_name, mask_name, meta_ptr);

    // Build heads
    std::string policy_out, pass_out;
    buildPolicyHead(&main_block, trunk_out, mask_name, policy_out, pass_out);

    std::string value_out, ownership_out, score_value_out;
    buildValueHead(&main_block, trunk_out, mask_name, value_out, ownership_out, score_value_out);

    // For FP16 mode with FP32 I/O, add cast operations to convert outputs back to FP32
    std::string final_policy_out = policy_out;
    std::string final_pass_out = pass_out;
    std::string final_value_out = value_out;
    std::string final_ownership_out = ownership_out;
    std::string final_score_value_out = score_value_out;

    if (m_use_fp16 && !m_use_fp16_io) {
        const auto& ph = m_model.policy_head;
        const auto& vh = m_model.value_head;

        // Cast policy_p2_conv: [1, p2_out_channels, H, W] fp16 -> fp32
        final_policy_out = "policy_p2_conv";
        addCastOp(&main_block, policy_out, final_policy_out, "fp32",
                  {1, ph.p2_conv.out_channels, m_board_y_size, m_board_x_size});

        // Cast pass output: [1, 2] fp16 -> fp32
        final_pass_out = "policy_pass";  // Python uses policy_pass for all versions
        int pass_out_channels = ph.gpool_to_pass_mul2.has_value()
            ? ph.gpool_to_pass_mul2->out_channels
            : ph.gpool_to_pass_mul.out_channels;
        addCastOp(&main_block, pass_out, final_pass_out, "fp32",
                  {1, pass_out_channels});

        // Cast value_v3_bias: [1, 3] fp16 -> fp32
        final_value_out = "value_v3_bias";
        addCastOp(&main_block, value_out, final_value_out, "fp32",
                  {1, vh.v3_mul.out_channels});

        // Cast ownership: [1, 1, H, W] fp16 -> fp32
        final_ownership_out = "value_ownership_conv";
        addCastOp(&main_block, ownership_out, final_ownership_out, "fp32",
                  {1, vh.v_ownership_conv.out_channels, m_board_y_size, m_board_x_size});

        // Cast score_value: [1, num_score_value_channels] fp16 -> fp32
        final_score_value_out = "value_sv3_bias";
        addCastOp(&main_block, score_value_out, final_score_value_out, "fp32",
                  {1, vh.sv3_mul.out_channels});
    }

    // Set block outputs
    main_block.add_outputs(final_policy_out);
    main_block.add_outputs(final_pass_out);
    main_block.add_outputs(final_value_out);
    main_block.add_outputs(final_ownership_out);
    main_block.add_outputs(final_score_value_out);

    return program;
}

// ============================================================================
// MIL Operation Helpers
// ============================================================================

void MILBuilder::addConstOp(CoreML::Specification::MILSpec::Block* block,
                            const std::string& name,
                            const std::vector<float>& data,
                            const std::vector<int64_t>& shape) {
    // Register weight for blob storage. Mark FP32 storage when this const is declared FP32 (e.g.
    // inside an FP32 sub-region of an otherwise-FP16 model) so storage matches the declared type.
    m_ops.registerWeight(name, data, shape,
                         m_weight_dtype == CoreML::Specification::MILSpec::DataType::FLOAT32);

    // Add const operation
    auto* op = block->add_operations();
    op->set_type("const");

    // "name" attribute (matching Python structure)
    auto& name_attr = (*op->mutable_attributes())["name"];
    name_attr.mutable_type()->mutable_tensortype()->set_datatype(
        CoreML::Specification::MILSpec::DataType::STRING);
    name_attr.mutable_immediatevalue()->mutable_tensor()->mutable_strings()->add_values(name);

    // "val" attribute with type and blob reference
    auto& val_attr = (*op->mutable_attributes())["val"];
    auto* val_type = val_attr.mutable_type()->mutable_tensortype();
    val_type->set_datatype(m_weight_dtype);
    val_type->set_rank(static_cast<int64_t>(shape.size()));
    for (int64_t dim : shape) {
        val_type->add_dimensions()->mutable_constant()->set_size(dim);
    }
    auto* blob_val = val_attr.mutable_blobfilevalue();
    blob_val->set_filename("@model_path/weights/weight.bin");
    // Offset will be set during serialization

    // Set output
    auto* output = op->add_outputs();
    output->set_name(name);
    auto* output_type = output->mutable_type()->mutable_tensortype();
    output_type->set_datatype(m_weight_dtype);
    output_type->set_rank(static_cast<int64_t>(shape.size()));
    for (int64_t dim : shape) {
        output_type->add_dimensions()->mutable_constant()->set_size(dim);
    }
}

// Helper: Add INT32 array const op (for axes, shape)
void MILBuilder::addIntArrayConstOp(CoreML::Specification::MILSpec::Block* block,
                                     const std::string& name,
                                     const std::vector<int32_t>& values) {
    auto* op = block->add_operations();
    op->set_type("const");

    // "name" attribute
    auto& name_attr = (*op->mutable_attributes())["name"];
    name_attr.mutable_type()->mutable_tensortype()->set_datatype(
        CoreML::Specification::MILSpec::DataType::STRING);
    name_attr.mutable_immediatevalue()->mutable_tensor()->mutable_strings()->add_values(name);

    // "val" attribute with INT32 type
    auto& val_attr = (*op->mutable_attributes())["val"];
    auto* val_type = val_attr.mutable_type()->mutable_tensortype();
    val_type->set_datatype(CoreML::Specification::MILSpec::DataType::INT32);
    val_type->set_rank(1);
    val_type->add_dimensions()->mutable_constant()->set_size(static_cast<int64_t>(values.size()));
    auto* ints = val_attr.mutable_immediatevalue()->mutable_tensor()->mutable_ints();
    for (int32_t v : values) {
        ints->add_values(v);
    }

    // Output
    auto* output = op->add_outputs();
    output->set_name(name);
    auto* out_type = output->mutable_type()->mutable_tensortype();
    out_type->set_datatype(CoreML::Specification::MILSpec::DataType::INT32);
    out_type->set_rank(1);
    out_type->add_dimensions()->mutable_constant()->set_size(static_cast<int64_t>(values.size()));
}

// Helper: Add BOOL scalar const op (for keep_dims)
void MILBuilder::addBoolScalarConstOp(CoreML::Specification::MILSpec::Block* block,
                                       const std::string& name,
                                       bool value) {
    auto* op = block->add_operations();
    op->set_type("const");

    // "name" attribute
    auto& name_attr = (*op->mutable_attributes())["name"];
    name_attr.mutable_type()->mutable_tensortype()->set_datatype(
        CoreML::Specification::MILSpec::DataType::STRING);
    name_attr.mutable_immediatevalue()->mutable_tensor()->mutable_strings()->add_values(name);

    // "val" attribute with BOOL type (rank 0 = scalar)
    auto& val_attr = (*op->mutable_attributes())["val"];
    auto* val_type = val_attr.mutable_type()->mutable_tensortype();
    val_type->set_datatype(CoreML::Specification::MILSpec::DataType::BOOL);
    val_type->set_rank(0);
    val_attr.mutable_immediatevalue()->mutable_tensor()->mutable_bools()->add_values(value);

    // Output
    auto* output = op->add_outputs();
    output->set_name(name);
    auto* out_type = output->mutable_type()->mutable_tensortype();
    out_type->set_datatype(CoreML::Specification::MILSpec::DataType::BOOL);
    out_type->set_rank(0);
}

// Helper: Add FLOAT32 scalar const op (for y values in sub/mul)
void MILBuilder::addFloatScalarConstOp(CoreML::Specification::MILSpec::Block* block,
                                        const std::string& name,
                                        float value) {
    auto* op = block->add_operations();
    op->set_type("const");

    // "name" attribute
    auto& name_attr = (*op->mutable_attributes())["name"];
    name_attr.mutable_type()->mutable_tensortype()->set_datatype(
        CoreML::Specification::MILSpec::DataType::STRING);
    name_attr.mutable_immediatevalue()->mutable_tensor()->mutable_strings()->add_values(name);

    // "val" attribute with appropriate dtype (rank 0 = scalar)
    auto& val_attr = (*op->mutable_attributes())["val"];
    auto* val_type = val_attr.mutable_type()->mutable_tensortype();
    val_type->set_datatype(m_weight_dtype);
    val_type->set_rank(0);

    // Key the storage format off the DECLARED dtype (m_weight_dtype), not the global m_use_fp16:
    // a temporarily-flipped FP32 sub-region (m_weight_dtype=FLOAT32 while m_use_fp16 stays true)
    // must store FP32 floats, or CoreML rejects the model ("storage and type have different number
    // of elements"). For all non-flipped calls m_weight_dtype tracks m_use_fp16, so this is a no-op.
    if (m_weight_dtype == CoreML::Specification::MILSpec::DataType::FLOAT16) {
        // For FP16, use bytes storage with FP16 representation
        MILBlob::Fp16 fp16_val = MILBlob::Fp16::FromFloat(value);
        std::string bytes_data(reinterpret_cast<const char*>(&fp16_val.bytes), sizeof(fp16_val.bytes));
        val_attr.mutable_immediatevalue()->mutable_tensor()->mutable_bytes()->set_values(bytes_data);
    } else {
        // For FP32, use floats storage
        val_attr.mutable_immediatevalue()->mutable_tensor()->mutable_floats()->add_values(value);
    }

    // Output
    auto* output = op->add_outputs();
    output->set_name(name);
    auto* out_type = output->mutable_type()->mutable_tensortype();
    out_type->set_datatype(m_weight_dtype);
    out_type->set_rank(0);
}

// Helper: Add INT32 scalar const op (for concat axis)
void MILBuilder::addIntScalarConstOp(CoreML::Specification::MILSpec::Block* block,
                                      const std::string& name,
                                      int32_t value) {
    auto* op = block->add_operations();
    op->set_type("const");

    // "name" attribute
    auto& name_attr = (*op->mutable_attributes())["name"];
    name_attr.mutable_type()->mutable_tensortype()->set_datatype(
        CoreML::Specification::MILSpec::DataType::STRING);
    name_attr.mutable_immediatevalue()->mutable_tensor()->mutable_strings()->add_values(name);

    // "val" attribute with INT32 type (rank 0 = scalar)
    auto& val_attr = (*op->mutable_attributes())["val"];
    auto* val_type = val_attr.mutable_type()->mutable_tensortype();
    val_type->set_datatype(CoreML::Specification::MILSpec::DataType::INT32);
    val_type->set_rank(0);
    val_attr.mutable_immediatevalue()->mutable_tensor()->mutable_ints()->add_values(value);

    // Output
    auto* output = op->add_outputs();
    output->set_name(name);
    auto* out_type = output->mutable_type()->mutable_tensortype();
    out_type->set_datatype(CoreML::Specification::MILSpec::DataType::INT32);
    out_type->set_rank(0);
}

// Helper: Add cast operation for dtype conversion
void MILBuilder::addCastOp(CoreML::Specification::MILSpec::Block* block,
                           const std::string& input,
                           const std::string& output,
                           const std::string& dtype,
                           const std::vector<int64_t>& shape) {
    // Create dtype const (STRING type)
    std::string dtype_name = output + "_dtype_0";
    {
        auto* op = block->add_operations();
        op->set_type("const");

        auto& name_attr = (*op->mutable_attributes())["name"];
        name_attr.mutable_type()->mutable_tensortype()->set_datatype(
            CoreML::Specification::MILSpec::DataType::STRING);
        name_attr.mutable_immediatevalue()->mutable_tensor()->mutable_strings()->add_values(dtype_name);

        auto& val_attr = (*op->mutable_attributes())["val"];
        val_attr.mutable_type()->mutable_tensortype()->set_datatype(
            CoreML::Specification::MILSpec::DataType::STRING);
        val_attr.mutable_immediatevalue()->mutable_tensor()->mutable_strings()->add_values(dtype);

        auto* out = op->add_outputs();
        out->set_name(dtype_name);
        out->mutable_type()->mutable_tensortype()->set_datatype(
            CoreML::Specification::MILSpec::DataType::STRING);
    }

    // Create cast operation
    auto* op = block->add_operations();
    op->set_type("cast");

    auto& inputs = *op->mutable_inputs();
    inputs["x"].add_arguments()->set_name(input);
    inputs["dtype"].add_arguments()->set_name(dtype_name);

    // Set output with target dtype
    auto* out = op->add_outputs();
    out->set_name(output);
    auto* tt = out->mutable_type()->mutable_tensortype();
    tt->set_datatype(dtype == "fp16"
        ? CoreML::Specification::MILSpec::DataType::FLOAT16
        : CoreML::Specification::MILSpec::DataType::FLOAT32);
    tt->set_rank(static_cast<int64_t>(shape.size()));
    // First dimension is batch - use setBatchDimension
    setBatchDimension(tt);
    // Remaining dimensions are constant
    for (size_t i = 1; i < shape.size(); i++) {
        tt->add_dimensions()->mutable_constant()->set_size(shape[i]);
    }
}

std::string MILBuilder::castFixed(CoreML::Specification::MILSpec::Block* block,
                                  const std::string& input,
                                  const std::string& dtype,
                                  const std::vector<int64_t>& dims) {
    std::string out = genVarName(input + "_cast");
    std::string dtName = out + "_dt";
    {
        auto* op = block->add_operations();
        op->set_type("const");
        auto& na = (*op->mutable_attributes())["name"];
        na.mutable_type()->mutable_tensortype()->set_datatype(CoreML::Specification::MILSpec::DataType::STRING);
        na.mutable_immediatevalue()->mutable_tensor()->mutable_strings()->add_values(dtName);
        auto& va = (*op->mutable_attributes())["val"];
        va.mutable_type()->mutable_tensortype()->set_datatype(CoreML::Specification::MILSpec::DataType::STRING);
        va.mutable_immediatevalue()->mutable_tensor()->mutable_strings()->add_values(dtype);
        auto* o = op->add_outputs();
        o->set_name(dtName);
        o->mutable_type()->mutable_tensortype()->set_datatype(CoreML::Specification::MILSpec::DataType::STRING);
    }
    auto* op = block->add_operations();
    op->set_type("cast");
    (*op->mutable_inputs())["x"].add_arguments()->set_name(input);
    (*op->mutable_inputs())["dtype"].add_arguments()->set_name(dtName);
    auto* o = op->add_outputs();
    o->set_name(out);
    auto* tt = o->mutable_type()->mutable_tensortype();
    tt->set_datatype(dtype == "fp32" ? CoreML::Specification::MILSpec::DataType::FLOAT32
                                     : CoreML::Specification::MILSpec::DataType::FLOAT16);
    tt->set_rank(static_cast<int64_t>(dims.size()));
    for (int64_t d : dims) {
        if (d < 0) tt->add_dimensions()->mutable_unknown()->set_variadic(false);
        else tt->add_dimensions()->mutable_constant()->set_size(d);
    }
    return out;
}

void MILBuilder::addGlobalPoolingFp32(CoreML::Specification::MILSpec::Block* block,
                                      const std::string& input,
                                      const std::string& mask,
                                      int channels,
                                      const std::string& output,
                                      bool valueVariant) {
    auto pool = [&](const std::string& in, const std::string& msk, const std::string& out) {
        if (valueVariant) addGlobalPoolingValueOps(block, in, msk, channels, out);
        else              addGlobalPoolingOps(block, in, msk, channels, out);
    };
    // Non-spatial per KataGo's FP16 convention -> FP32 (the FP16 spatial sum over H*W loses too much
    // precision at larger board sizes). No addConstOp in the pooling, so flipping m_weight_dtype is
    // safe. Cast input/mask up, pool in FP32, cast the [N, channels*3] features back to FP16.
    if (!m_nonspatial_fp32) {
        pool(input, mask, output);
        return;
    }
    std::string in32 = castFixed(block, input, "fp32", {-1, channels, m_board_y_size, m_board_x_size});
    std::string mask32 = m_optimize_identity_mask
        ? mask
        : castFixed(block, mask, "fp32", {-1, 1, m_board_y_size, m_board_x_size});
    std::string out32 = genVarName(output + "_f32");
    { ScopedFp32 g(m_weight_dtype); pool(in32, mask32, out32); }
    addCastOp(block, out32, output, "fp16", {-1, channels * 3});
}

void MILBuilder::addConvOp(CoreML::Specification::MILSpec::Block* block,
                           const std::string& input,
                           const ConvLayerDesc& layer,
                           const std::string& output) {
    // Create const operations for all parameters (matching Python structure)
    std::string weight_name = output + "_weight_0";
    std::string pad_type_name = output + "_pad_type_0";
    std::string dilations_name = output + "_dilations_0";
    std::string strides_name = output + "_strides_0";
    std::string groups_name = output + "_groups_0";
    std::string pad_name = output + "_pad_0";

    // Add weight constant
    addConstOp(block, weight_name, layer.weights, layer.getWeightShape());

    // Add pad_type constant ("same") - STRING type
    {
        auto* const_op = block->add_operations();
        const_op->set_type("const");
        // "name" attribute
        auto& name_attr = (*const_op->mutable_attributes())["name"];
        name_attr.mutable_type()->mutable_tensortype()->set_datatype(
            CoreML::Specification::MILSpec::DataType::STRING);
        name_attr.mutable_immediatevalue()->mutable_tensor()->mutable_strings()->add_values(pad_type_name);
        // "val" attribute with type
        auto& val = (*const_op->mutable_attributes())["val"];
        val.mutable_type()->mutable_tensortype()->set_datatype(
            CoreML::Specification::MILSpec::DataType::STRING);
        val.mutable_immediatevalue()->mutable_tensor()->mutable_strings()->add_values("same");
        // Output
        auto* out = const_op->add_outputs();
        out->set_name(pad_type_name);
        out->mutable_type()->mutable_tensortype()->set_datatype(
            CoreML::Specification::MILSpec::DataType::STRING);
    }

    // Add dilations constant - INT32 type, shape [2]
    {
        auto* const_op = block->add_operations();
        const_op->set_type("const");
        // "name" attribute
        auto& name_attr = (*const_op->mutable_attributes())["name"];
        name_attr.mutable_type()->mutable_tensortype()->set_datatype(
            CoreML::Specification::MILSpec::DataType::STRING);
        name_attr.mutable_immediatevalue()->mutable_tensor()->mutable_strings()->add_values(dilations_name);
        // "val" attribute with type
        auto& val = (*const_op->mutable_attributes())["val"];
        auto* val_type = val.mutable_type()->mutable_tensortype();
        val_type->set_datatype(CoreML::Specification::MILSpec::DataType::INT32);
        val_type->set_rank(1);
        val_type->add_dimensions()->mutable_constant()->set_size(2);
        auto* int_vals = val.mutable_immediatevalue()->mutable_tensor()->mutable_ints();
        int_vals->add_values(layer.dilation_y);
        int_vals->add_values(layer.dilation_x);
        // Output
        auto* out = const_op->add_outputs();
        out->set_name(dilations_name);
        auto* tt = out->mutable_type()->mutable_tensortype();
        tt->set_datatype(CoreML::Specification::MILSpec::DataType::INT32);
        tt->set_rank(1);
        tt->add_dimensions()->mutable_constant()->set_size(2);
    }

    // Add strides constant - INT32 type, shape [2]
    {
        auto* const_op = block->add_operations();
        const_op->set_type("const");
        // "name" attribute
        auto& name_attr = (*const_op->mutable_attributes())["name"];
        name_attr.mutable_type()->mutable_tensortype()->set_datatype(
            CoreML::Specification::MILSpec::DataType::STRING);
        name_attr.mutable_immediatevalue()->mutable_tensor()->mutable_strings()->add_values(strides_name);
        // "val" attribute with type
        auto& val = (*const_op->mutable_attributes())["val"];
        auto* val_type = val.mutable_type()->mutable_tensortype();
        val_type->set_datatype(CoreML::Specification::MILSpec::DataType::INT32);
        val_type->set_rank(1);
        val_type->add_dimensions()->mutable_constant()->set_size(2);
        auto* int_vals = val.mutable_immediatevalue()->mutable_tensor()->mutable_ints();
        int_vals->add_values(1);
        int_vals->add_values(1);
        // Output
        auto* out = const_op->add_outputs();
        out->set_name(strides_name);
        auto* tt = out->mutable_type()->mutable_tensortype();
        tt->set_datatype(CoreML::Specification::MILSpec::DataType::INT32);
        tt->set_rank(1);
        tt->add_dimensions()->mutable_constant()->set_size(2);
    }

    // Add groups constant (always 1 for standard convolution) - INT32 scalar type
    {
        auto* const_op = block->add_operations();
        const_op->set_type("const");
        // "name" attribute
        auto& name_attr = (*const_op->mutable_attributes())["name"];
        name_attr.mutable_type()->mutable_tensortype()->set_datatype(
            CoreML::Specification::MILSpec::DataType::STRING);
        name_attr.mutable_immediatevalue()->mutable_tensor()->mutable_strings()->add_values(groups_name);
        // "val" attribute with type (scalar)
        auto& val = (*const_op->mutable_attributes())["val"];
        val.mutable_type()->mutable_tensortype()->set_datatype(
            CoreML::Specification::MILSpec::DataType::INT32);
        val.mutable_immediatevalue()->mutable_tensor()->mutable_ints()->add_values(1);
        // Output
        auto* out = const_op->add_outputs();
        out->set_name(groups_name);
        out->mutable_type()->mutable_tensortype()->set_datatype(
            CoreML::Specification::MILSpec::DataType::INT32);
    }

    // Add pad constant [0, 0, 0, 0] for "same" padding - INT32 type, shape [4]
    {
        auto* const_op = block->add_operations();
        const_op->set_type("const");
        // "name" attribute
        auto& name_attr = (*const_op->mutable_attributes())["name"];
        name_attr.mutable_type()->mutable_tensortype()->set_datatype(
            CoreML::Specification::MILSpec::DataType::STRING);
        name_attr.mutable_immediatevalue()->mutable_tensor()->mutable_strings()->add_values(pad_name);
        // "val" attribute with type
        auto& val = (*const_op->mutable_attributes())["val"];
        auto* val_type = val.mutable_type()->mutable_tensortype();
        val_type->set_datatype(CoreML::Specification::MILSpec::DataType::INT32);
        val_type->set_rank(1);
        val_type->add_dimensions()->mutable_constant()->set_size(4);
        auto* int_vals = val.mutable_immediatevalue()->mutable_tensor()->mutable_ints();
        int_vals->add_values(0);
        int_vals->add_values(0);
        int_vals->add_values(0);
        int_vals->add_values(0);
        // Output
        auto* out = const_op->add_outputs();
        out->set_name(pad_name);
        auto* tt = out->mutable_type()->mutable_tensortype();
        tt->set_datatype(CoreML::Specification::MILSpec::DataType::INT32);
        tt->set_rank(1);
        tt->add_dimensions()->mutable_constant()->set_size(4);
    }

    // Channel-gated FP32 convs. The ANE accumulates FP16 convs in FP16, which loses too much
    // precision for WIDE trunks and fails testgpuerror at large board sizes (validated: 384ch
    // fails, <=256ch is fine FP16-on-ANE). For wide trunks (>= threshold) run convs in FP32 (weights
    // cast up at runtime, stored fp16). FP32 convs can't run on the fp16-only ANE, so only the wide
    // models that actually need it pay that off-ANE cost; narrow models keep convs on the ANE.
    const bool convFp32 = m_conv_fp32;
    std::string convX = input, convW = weight_name, convOut = output;
    auto savedConvDtype = m_weight_dtype;
    if (convFp32) {
        convX = castFixed(block, input, "fp32", {-1, layer.in_channels, m_board_y_size, m_board_x_size});
        convW = castFixed(block, weight_name, "fp32", layer.getWeightShape());
        m_weight_dtype = CoreML::Specification::MILSpec::DataType::FLOAT32;
        convOut = output + "_cf32";
    }

    // Add conv operation referencing all const parameters
    auto* op = block->add_operations();
    op->set_type("conv");

    // Inputs - reference const operations
    auto& inputs = *op->mutable_inputs();
    inputs["dilations"].add_arguments()->set_name(dilations_name);
    inputs["groups"].add_arguments()->set_name(groups_name);
    inputs["pad"].add_arguments()->set_name(pad_name);
    inputs["pad_type"].add_arguments()->set_name(pad_type_name);
    inputs["strides"].add_arguments()->set_name(strides_name);
    inputs["weight"].add_arguments()->set_name(convW);
    inputs["x"].add_arguments()->set_name(convX);

    // Output with dimensions [batch, out_channels, height, width]
    auto* out = op->add_outputs();
    out->set_name(convOut);
    auto* out_type = out->mutable_type()->mutable_tensortype();
    out_type->set_datatype(m_weight_dtype);
    out_type->set_rank(4);
    setBatchDimension(out_type);
    out_type->add_dimensions()->mutable_constant()->set_size(layer.out_channels);
    out_type->add_dimensions()->mutable_constant()->set_size(m_board_y_size);
    out_type->add_dimensions()->mutable_constant()->set_size(m_board_x_size);

    if (convFp32) {
        m_weight_dtype = savedConvDtype;
        addCastOp(block, convOut, output, "fp16", {-1, layer.out_channels, m_board_y_size, m_board_x_size});
    }
}

// Helper: Set output tensor type with 4D shape [batch, C, H, W]
void MILBuilder::setTensorOutput4D(CoreML::Specification::MILSpec::Operation* op,
                                    const std::string& name,
                                    int channels, int height, int width) {
    auto* out = op->add_outputs();
    out->set_name(name);
    auto* tt = out->mutable_type()->mutable_tensortype();
    tt->set_datatype(m_weight_dtype);
    tt->set_rank(4);
    setBatchDimension(tt);
    tt->add_dimensions()->mutable_constant()->set_size(channels);
    tt->add_dimensions()->mutable_constant()->set_size(height);
    tt->add_dimensions()->mutable_constant()->set_size(width);
}

// Helper: Set output tensor type with 2D shape [batch, C]
void MILBuilder::setTensorOutput2D(CoreML::Specification::MILSpec::Operation* op,
                                    const std::string& name,
                                    int channels) {
    auto* out = op->add_outputs();
    out->set_name(name);
    auto* tt = out->mutable_type()->mutable_tensortype();
    tt->set_datatype(m_weight_dtype);
    tt->set_rank(2);
    setBatchDimension(tt);
    tt->add_dimensions()->mutable_constant()->set_size(channels);
}

// Helper: Set output tensor type with 4D shape [batch, C, 1, 1] for pooled results
void MILBuilder::setTensorOutputPooled4D(CoreML::Specification::MILSpec::Operation* op,
                                          const std::string& name,
                                          int channels) {
    auto* out = op->add_outputs();
    out->set_name(name);
    auto* tt = out->mutable_type()->mutable_tensortype();
    tt->set_datatype(m_weight_dtype);
    tt->set_rank(4);
    setBatchDimension(tt);
    tt->add_dimensions()->mutable_constant()->set_size(channels);
    tt->add_dimensions()->mutable_constant()->set_size(1);
    tt->add_dimensions()->mutable_constant()->set_size(1);
}

// Helper: Set output tensor type with 4D shape [batch, 1, 1, 1] (for mask operations)
void MILBuilder::setTensorOutputMask4D(CoreML::Specification::MILSpec::Operation* op,
                                        const std::string& name) {
    auto* out = op->add_outputs();
    out->set_name(name);
    auto* tt = out->mutable_type()->mutable_tensortype();
    tt->set_datatype(m_weight_dtype);
    tt->set_rank(4);
    setBatchDimension(tt);
    tt->add_dimensions()->mutable_constant()->set_size(1);
    tt->add_dimensions()->mutable_constant()->set_size(1);
    tt->add_dimensions()->mutable_constant()->set_size(1);
}

// Helper: Set output tensor type with 4D shape [batch, 1, H, W] (for mask spatial operations)
void MILBuilder::setTensorOutputMaskSpatial4D(CoreML::Specification::MILSpec::Operation* op,
                                               const std::string& name,
                                               int height, int width) {
    auto* out = op->add_outputs();
    out->set_name(name);
    auto* tt = out->mutable_type()->mutable_tensortype();
    tt->set_datatype(m_weight_dtype);
    tt->set_rank(4);
    setBatchDimension(tt);
    tt->add_dimensions()->mutable_constant()->set_size(1);
    tt->add_dimensions()->mutable_constant()->set_size(height);
    tt->add_dimensions()->mutable_constant()->set_size(width);
}

void MILBuilder::addBatchNormActivationOps(CoreML::Specification::MILSpec::Block* block,
                                           const std::string& input,
                                           const BatchNormLayerDesc& bn,
                                           const ActivationLayerDesc& act,
                                           const std::string& mask,
                                           const std::string& output) {
    // BN: x * scale + bias
    std::string scale_name = output + "_bn_scale";
    std::string bias_name = output + "_bn_bias";

    // Reshape scale/bias to [1, C, 1, 1]
    std::vector<int64_t> bn_shape = {1, static_cast<int64_t>(bn.num_channels), 1, 1};
    addConstOp(block, scale_name, bn.merged_scale, bn_shape);
    addConstOp(block, bias_name, bn.merged_bias, bn_shape);

    // Mul: x * scale
    std::string scaled_name = output + "_scaled";
    {
        auto* op = block->add_operations();
        op->set_type("mul");
        auto& inputs = *op->mutable_inputs();
        inputs["x"].add_arguments()->set_name(input);
        inputs["y"].add_arguments()->set_name(scale_name);
        setTensorOutput4D(op, scaled_name, bn.num_channels, m_board_y_size, m_board_x_size);
    }

    // Add: scaled + bias
    std::string biased_name = output + "_biased";
    {
        auto* op = block->add_operations();
        op->set_type("add");
        auto& inputs = *op->mutable_inputs();
        inputs["x"].add_arguments()->set_name(scaled_name);
        inputs["y"].add_arguments()->set_name(bias_name);
        setTensorOutput4D(op, biased_name, bn.num_channels, m_board_y_size, m_board_x_size);
    }

    std::string bn_output = biased_name;

    // Apply mask if not optimizing
    if (!m_optimize_identity_mask) {
        std::string masked_name = output + "_masked";
        auto* op = block->add_operations();
        op->set_type("mul");
        auto& inputs = *op->mutable_inputs();
        inputs["x"].add_arguments()->set_name(bn_output);
        inputs["y"].add_arguments()->set_name(mask);
        setTensorOutput4D(op, masked_name, bn.num_channels, m_board_y_size, m_board_x_size);
        bn_output = masked_name;
    }

    // Activation
    if (act.activation_type == ActivationType::Identity) {
        // Identity: just rename
        // In MIL we need to copy
        auto* op = block->add_operations();
        op->set_type("identity");
        auto& inputs = *op->mutable_inputs();
        inputs["x"].add_arguments()->set_name(bn_output);
        setTensorOutput4D(op, output, bn.num_channels, m_board_y_size, m_board_x_size);
    } else if (act.activation_type == ActivationType::ReLU) {
        auto* op = block->add_operations();
        op->set_type("relu");
        auto& inputs = *op->mutable_inputs();
        inputs["x"].add_arguments()->set_name(bn_output);
        setTensorOutput4D(op, output, bn.num_channels, m_board_y_size, m_board_x_size);
    } else if (act.activation_type == ActivationType::Mish) {
        addMishOps(block, bn_output, output, 4, bn.num_channels);
    } else if (act.activation_type == ActivationType::Silu) {
        addSiluOps(block, bn_output, output, 4, bn.num_channels);
    }
}

void MILBuilder::addSiluOps(CoreML::Specification::MILSpec::Block* block,
                            const std::string& input,
                            const std::string& output,
                            int rank,
                            int channels) {
    // SiLU / Swish: x * sigmoid(x)
    auto setOutputType = [this, rank, channels](CoreML::Specification::MILSpec::Operation* op, const std::string& name) {
        auto* out = op->add_outputs();
        out->set_name(name);
        auto* out_type = out->mutable_type()->mutable_tensortype();
        out_type->set_datatype(m_weight_dtype);
        out_type->set_rank(rank);
        setBatchDimension(out_type);
        out_type->add_dimensions()->mutable_constant()->set_size(channels);
        if (rank == 4) {
            out_type->add_dimensions()->mutable_constant()->set_size(m_board_y_size);
            out_type->add_dimensions()->mutable_constant()->set_size(m_board_x_size);
        }
    };

    std::string sig = output + "_sigmoid";
    {
        auto* op = block->add_operations();
        op->set_type("sigmoid");
        auto& inputs = *op->mutable_inputs();
        inputs["x"].add_arguments()->set_name(input);
        setOutputType(op, sig);
    }
    {
        auto* op = block->add_operations();
        op->set_type("mul");
        auto& inputs = *op->mutable_inputs();
        inputs["x"].add_arguments()->set_name(input);
        inputs["y"].add_arguments()->set_name(sig);
        setOutputType(op, output);
    }
}

void MILBuilder::setShape(CoreML::Specification::MILSpec::Operation* op,
                          const std::string& name,
                          const std::vector<int64_t>& dims) {
    auto* out = op->add_outputs();
    out->set_name(name);
    auto* t = out->mutable_type()->mutable_tensortype();
    t->set_datatype(m_weight_dtype);
    t->set_rank(static_cast<int64_t>(dims.size()));
    for (int64_t d : dims) {
        auto* dim = t->add_dimensions();
        if (d < 0)
            dim->mutable_unknown()->set_variadic(false);
        else
            dim->mutable_constant()->set_size(d);
    }
}

void MILBuilder::addMishOps(CoreML::Specification::MILSpec::Block* block,
                            const std::string& input,
                            const std::string& output,
                            int rank,
                            int channels) {
    // Mish: x / (1 + 2 / (e * (e + 2)))
    // e = exp(x)
    //
    // rank and channels are used to set output type info:
    // - rank=4: spatial tensors [1, C, H, W] (uses m_board_y_size, m_board_x_size)
    // - rank=2: vector tensors [1, C]

    auto setOutputType = [this, rank, channels](CoreML::Specification::MILSpec::Operation* op, const std::string& name) {
        auto* out = op->add_outputs();
        out->set_name(name);
        auto* out_type = out->mutable_type()->mutable_tensortype();
        out_type->set_datatype(m_weight_dtype);
        out_type->set_rank(rank);
        setBatchDimension(out_type);
        out_type->add_dimensions()->mutable_constant()->set_size(channels);
        if (rank == 4) {
            out_type->add_dimensions()->mutable_constant()->set_size(m_board_y_size);
            out_type->add_dimensions()->mutable_constant()->set_size(m_board_x_size);
        }
    };

    std::string e = output + "_exp";
    std::string ep2 = output + "_ep2";
    std::string emep2 = output + "_emep2";
    std::string tdemep2 = output + "_tdemep2";
    std::string optdemep2 = output + "_optdemep2";

    // Create scalar constants for Mish computation
    std::string const_one = output + "_const_1";
    std::string const_two = output + "_const_2";
    addFloatScalarConstOp(block, const_one, 1.0f);
    addFloatScalarConstOp(block, const_two, 2.0f);

    // e = exp(x)
    {
        auto* op = block->add_operations();
        op->set_type("exp");
        auto& inputs = *op->mutable_inputs();
        inputs["x"].add_arguments()->set_name(input);
        setOutputType(op, e);
    }

    // ep2 = e + 2
    {
        auto* op = block->add_operations();
        op->set_type("add");
        auto& inputs = *op->mutable_inputs();
        inputs["x"].add_arguments()->set_name(e);
        inputs["y"].add_arguments()->set_name(const_two);
        setOutputType(op, ep2);
    }

    // emep2 = e * ep2
    {
        auto* op = block->add_operations();
        op->set_type("mul");
        auto& inputs = *op->mutable_inputs();
        inputs["x"].add_arguments()->set_name(e);
        inputs["y"].add_arguments()->set_name(ep2);
        setOutputType(op, emep2);
    }

    // tdemep2 = 2 / emep2
    {
        auto* op = block->add_operations();
        op->set_type("real_div");
        auto& inputs = *op->mutable_inputs();
        inputs["x"].add_arguments()->set_name(const_two);
        inputs["y"].add_arguments()->set_name(emep2);
        setOutputType(op, tdemep2);
    }

    // optdemep2 = 1 + tdemep2
    {
        auto* op = block->add_operations();
        op->set_type("add");
        auto& inputs = *op->mutable_inputs();
        inputs["x"].add_arguments()->set_name(const_one);
        inputs["y"].add_arguments()->set_name(tdemep2);
        setOutputType(op, optdemep2);
    }

    // output = x / optdemep2
    {
        auto* op = block->add_operations();
        op->set_type("real_div");
        auto& inputs = *op->mutable_inputs();
        inputs["x"].add_arguments()->set_name(input);
        inputs["y"].add_arguments()->set_name(optdemep2);
        setOutputType(op, output);
    }
}

void MILBuilder::addMatMulOp(CoreML::Specification::MILSpec::Block* block,
                             const std::string& input,
                             const MatMulLayerDesc& layer,
                             const std::string& output) {
    // Create const operations for all parameters (matching Python structure)
    std::string weight_name = output + "_y_0";
    std::string transpose_x_name = output + "_transpose_x_0";
    std::string transpose_y_name = output + "_transpose_y_0";

    // Add weight constant
    addConstOp(block, weight_name, layer.weights, layer.getWeightShape());

    // Add transpose_x constant (false) - BOOL type
    {
        auto* const_op = block->add_operations();
        const_op->set_type("const");
        // "name" attribute
        auto& name_attr = (*const_op->mutable_attributes())["name"];
        name_attr.mutable_type()->mutable_tensortype()->set_datatype(
            CoreML::Specification::MILSpec::DataType::STRING);
        name_attr.mutable_immediatevalue()->mutable_tensor()->mutable_strings()->add_values(transpose_x_name);
        // "val" attribute with type
        auto& val = (*const_op->mutable_attributes())["val"];
        val.mutable_type()->mutable_tensortype()->set_datatype(
            CoreML::Specification::MILSpec::DataType::BOOL);
        val.mutable_immediatevalue()->mutable_tensor()->mutable_bools()->add_values(false);
        // Output
        auto* out = const_op->add_outputs();
        out->set_name(transpose_x_name);
        out->mutable_type()->mutable_tensortype()->set_datatype(
            CoreML::Specification::MILSpec::DataType::BOOL);
    }

    // Add transpose_y constant (false) - BOOL type
    {
        auto* const_op = block->add_operations();
        const_op->set_type("const");
        // "name" attribute
        auto& name_attr = (*const_op->mutable_attributes())["name"];
        name_attr.mutable_type()->mutable_tensortype()->set_datatype(
            CoreML::Specification::MILSpec::DataType::STRING);
        name_attr.mutable_immediatevalue()->mutable_tensor()->mutable_strings()->add_values(transpose_y_name);
        // "val" attribute with type
        auto& val = (*const_op->mutable_attributes())["val"];
        val.mutable_type()->mutable_tensortype()->set_datatype(
            CoreML::Specification::MILSpec::DataType::BOOL);
        val.mutable_immediatevalue()->mutable_tensor()->mutable_bools()->add_values(false);
        // Output
        auto* out = const_op->add_outputs();
        out->set_name(transpose_y_name);
        out->mutable_type()->mutable_tensortype()->set_datatype(
            CoreML::Specification::MILSpec::DataType::BOOL);
    }

    // Non-spatial matmul in FP32 (KataGo FP16 convention; weights cast up at runtime, stored fp16).
    std::string mmIn = input, mmW = weight_name, mmOut = output;
    auto savedMmDtype = m_weight_dtype;
    if (m_nonspatial_fp32) {
        mmIn = castFixed(block, input, "fp32", {-1, layer.in_channels});
        mmW = castFixed(block, weight_name, "fp32", layer.getWeightShape());
        m_weight_dtype = CoreML::Specification::MILSpec::DataType::FLOAT32;
        mmOut = output + "_mmf32";
    }

    // Add matmul operation
    auto* op = block->add_operations();
    op->set_type("matmul");
    auto& inputs = *op->mutable_inputs();
    inputs["transpose_x"].add_arguments()->set_name(transpose_x_name);
    inputs["transpose_y"].add_arguments()->set_name(transpose_y_name);
    inputs["x"].add_arguments()->set_name(mmIn);
    inputs["y"].add_arguments()->set_name(mmW);

    // Output with 2D shape [batch, out_channels]
    auto* out = op->add_outputs();
    out->set_name(mmOut);
    auto* out_type = out->mutable_type()->mutable_tensortype();
    out_type->set_datatype(m_weight_dtype);
    out_type->set_rank(2);
    setBatchDimension(out_type);
    out_type->add_dimensions()->mutable_constant()->set_size(layer.out_channels);

    if (m_nonspatial_fp32) {
        m_weight_dtype = savedMmDtype;
        addCastOp(block, mmOut, output, "fp16", {-1, layer.out_channels});
    }
}

void MILBuilder::addMatBiasOp(CoreML::Specification::MILSpec::Block* block,
                              const std::string& input,
                              const MatBiasLayerDesc& layer,
                              const std::string& output) {
    // Add bias constant
    std::string bias_name = output + "_bias";
    std::vector<int64_t> shape = {static_cast<int64_t>(layer.num_channels)};
    addConstOp(block, bias_name, layer.weights, shape);

    // Add add operation
    auto* op = block->add_operations();
    op->set_type("add");
    auto& inputs = *op->mutable_inputs();
    inputs["x"].add_arguments()->set_name(input);
    inputs["y"].add_arguments()->set_name(bias_name);

    // Output with 2D shape [batch, num_channels] (same as matmul output)
    auto* out = op->add_outputs();
    out->set_name(output);
    auto* out_type = out->mutable_type()->mutable_tensortype();
    out_type->set_datatype(m_weight_dtype);
    out_type->set_rank(2);
    setBatchDimension(out_type);
    out_type->add_dimensions()->mutable_constant()->set_size(layer.num_channels);
}

void MILBuilder::addLinearOp(CoreML::Specification::MILSpec::Block* block,
                             const std::string& input,
                             const MatMulLayerDesc& matmul,
                             const MatBiasLayerDesc& bias,
                             const std::string& output) {
    // Create const operations for weight and bias (matching Python's linear op structure)
    // Core ML linear expects weights in [out_channels, in_channels] format
    // KataGo matmul stores weights in [in_channels, out_channels] format
    // We need to transpose the weights to match Python's fuse_matmul_weight_bias pass
    std::string weight_name = output + "_weight_0";
    std::string bias_name = output + "_bias_0";

    // Transpose weights from [in_channels, out_channels] to [out_channels, in_channels]
    const int in_ch = matmul.in_channels;
    const int out_ch = matmul.out_channels;
    std::vector<float> transposed_weights(matmul.weights.size());
    for (int i = 0; i < in_ch; ++i) {
        for (int j = 0; j < out_ch; ++j) {
            // Original: weights[i * out_ch + j] (row-major [in_ch, out_ch])
            // Transposed: weights[j * in_ch + i] (row-major [out_ch, in_ch])
            transposed_weights[j * in_ch + i] = matmul.weights[i * out_ch + j];
        }
    }

    // Add transposed weight constant with shape [out_channels, in_channels]
    std::vector<int64_t> transposed_shape = {static_cast<int64_t>(out_ch), static_cast<int64_t>(in_ch)};
    addConstOp(block, weight_name, transposed_weights, transposed_shape);

    // Add bias constant
    std::vector<int64_t> bias_shape = {static_cast<int64_t>(bias.num_channels)};
    addConstOp(block, bias_name, bias.weights, bias_shape);

    // NOTE: the MIL `linear` op requires const weight/bias, so the runtime-cast-to-FP32 trick can't
    // be applied here (unlike `matmul`). Value-head linear stays FP16; if a model ever needs it in
    // FP32, rewrite as matmul+add (matmul accepts cast inputs).
    auto* op = block->add_operations();
    op->set_type("linear");
    auto& inputs = *op->mutable_inputs();
    inputs["x"].add_arguments()->set_name(input);
    inputs["weight"].add_arguments()->set_name(weight_name);
    inputs["bias"].add_arguments()->set_name(bias_name);

    // Output with 2D shape [batch, out_channels]
    auto* out = op->add_outputs();
    out->set_name(output);
    auto* out_type = out->mutable_type()->mutable_tensortype();
    out_type->set_datatype(m_weight_dtype);
    out_type->set_rank(2);
    setBatchDimension(out_type);
    out_type->add_dimensions()->mutable_constant()->set_size(matmul.out_channels);
}

void MILBuilder::addGlobalPoolingOps(CoreML::Specification::MILSpec::Block* block,
                                     const std::string& input,
                                     const std::string& mask,
                                     int channels,
                                     const std::string& output) {
    // KataGo global pooling produces: [mean, mean_scaled, max]
    // mean_scaled = mean * (sqrt(count) - 14) * 0.1

    if (m_optimize_identity_mask) {
        // Optimized path: use precomputed constants
        const auto& mc = m_ops.getMaskConstants();

        // Mean pooling: sum / count
        std::string sum_name = output + "_sum";
        std::string sum_axes = sum_name + "_axes_0";
        std::string sum_keep_dims = sum_name + "_keep_dims_0";
        addIntArrayConstOp(block, sum_axes, {2, 3});
        addBoolScalarConstOp(block, sum_keep_dims, true);
        {
            auto* op = block->add_operations();
            op->set_type("reduce_sum");
            auto& inputs = *op->mutable_inputs();
            inputs["x"].add_arguments()->set_name(input);
            inputs["axes"].add_arguments()->set_name(sum_axes);
            inputs["keep_dims"].add_arguments()->set_name(sum_keep_dims);
            setTensorOutputPooled4D(op, sum_name, channels);
        }

        std::string mean_name = output + "_mean";
        std::string mean_y = mean_name + "_y_0";
        addFloatScalarConstOp(block, mean_y, mc.mask_sum_reciprocal);
        {
            auto* op = block->add_operations();
            op->set_type("mul");
            auto& inputs = *op->mutable_inputs();
            inputs["x"].add_arguments()->set_name(sum_name);
            inputs["y"].add_arguments()->set_name(mean_y);
            setTensorOutputPooled4D(op, mean_name, channels);
        }

        // Max pooling
        std::string max_name = output + "_max";
        std::string max_axes = max_name + "_axes_0";
        std::string max_keep_dims = max_name + "_keep_dims_0";
        addIntArrayConstOp(block, max_axes, {2, 3});
        addBoolScalarConstOp(block, max_keep_dims, true);
        {
            auto* op = block->add_operations();
            op->set_type("reduce_max");
            auto& inputs = *op->mutable_inputs();
            inputs["x"].add_arguments()->set_name(input);
            inputs["axes"].add_arguments()->set_name(max_axes);
            inputs["keep_dims"].add_arguments()->set_name(max_keep_dims);
            setTensorOutputPooled4D(op, max_name, channels);
        }

        // Mean scaled = mean * constant
        std::string mean_scaled_name = output + "_mean_scaled";
        std::string mean_scaled_y = mean_scaled_name + "_y_0";
        addFloatScalarConstOp(block, mean_scaled_y, mc.mask_sum_sqrt_s14_m01);
        {
            auto* op = block->add_operations();
            op->set_type("mul");
            auto& inputs = *op->mutable_inputs();
            inputs["x"].add_arguments()->set_name(mean_name);
            inputs["y"].add_arguments()->set_name(mean_scaled_y);
            setTensorOutputPooled4D(op, mean_scaled_name, channels);
        }

        // Squeeze spatial dimensions: [N, C, 1, 1] -> [N, C]
        std::string mean_flat = output + "_mean_flat";
        std::string mean_flat_axes = mean_flat + "_axes_0";
        addIntArrayConstOp(block, mean_flat_axes, {2, 3});
        {
            auto* op = block->add_operations();
            op->set_type("squeeze");
            auto& inputs = *op->mutable_inputs();
            inputs["x"].add_arguments()->set_name(mean_name);
            inputs["axes"].add_arguments()->set_name(mean_flat_axes);
            setTensorOutput2D(op, mean_flat, channels);
        }

        std::string mean_scaled_flat = output + "_mean_scaled_flat";
        std::string mean_scaled_flat_axes = mean_scaled_flat + "_axes_0";
        addIntArrayConstOp(block, mean_scaled_flat_axes, {2, 3});
        {
            auto* op = block->add_operations();
            op->set_type("squeeze");
            auto& inputs = *op->mutable_inputs();
            inputs["x"].add_arguments()->set_name(mean_scaled_name);
            inputs["axes"].add_arguments()->set_name(mean_scaled_flat_axes);
            setTensorOutput2D(op, mean_scaled_flat, channels);
        }

        std::string max_flat = output + "_max_flat";
        std::string max_flat_axes = max_flat + "_axes_0";
        addIntArrayConstOp(block, max_flat_axes, {2, 3});
        {
            auto* op = block->add_operations();
            op->set_type("squeeze");
            auto& inputs = *op->mutable_inputs();
            inputs["x"].add_arguments()->set_name(max_name);
            inputs["axes"].add_arguments()->set_name(max_flat_axes);
            setTensorOutput2D(op, max_flat, channels);
        }

        // Concatenate: [mean, mean_scaled, max]
        std::string concat_axis = output + "_concat_axis_0";
        std::string concat_interleave = output + "_concat_interleave_0";
        addIntScalarConstOp(block, concat_axis, 1);
        addBoolScalarConstOp(block, concat_interleave, false);
        {
            auto* op = block->add_operations();
            op->set_type("concat");
            auto& inputs = *op->mutable_inputs();
            inputs["values"].add_arguments()->set_name(mean_flat);
            inputs["values"].add_arguments()->set_name(mean_scaled_flat);
            inputs["values"].add_arguments()->set_name(max_flat);
            inputs["axis"].add_arguments()->set_name(concat_axis);
            inputs["interleave"].add_arguments()->set_name(concat_interleave);
            setTensorOutput2D(op, output, channels * 3);
        }
    } else {
        // Full path with mask operations
        // Count valid positions (mask is [1, 1, H, W], output is [1, 1, 1, 1])
        std::string mask_sum_name = output + "_mask_sum";
        std::string mask_sum_axes = mask_sum_name + "_axes_0";
        std::string mask_sum_keep_dims = mask_sum_name + "_keep_dims_0";
        addIntArrayConstOp(block, mask_sum_axes, {2, 3});
        addBoolScalarConstOp(block, mask_sum_keep_dims, true);
        {
            auto* op = block->add_operations();
            op->set_type("reduce_sum");
            auto& inputs = *op->mutable_inputs();
            inputs["x"].add_arguments()->set_name(mask);
            inputs["axes"].add_arguments()->set_name(mask_sum_axes);
            inputs["keep_dims"].add_arguments()->set_name(mask_sum_keep_dims);
            setTensorOutputMask4D(op, mask_sum_name);
        }

        // Masked input: [1, C, H, W] * [1, 1, H, W] -> [1, C, H, W]
        std::string masked_name = output + "_masked";
        {
            auto* op = block->add_operations();
            op->set_type("mul");
            auto& inputs = *op->mutable_inputs();
            inputs["x"].add_arguments()->set_name(input);
            inputs["y"].add_arguments()->set_name(mask);
            setTensorOutput4D(op, masked_name, channels, m_board_y_size, m_board_x_size);
        }

        // Sum masked values: [1, C, H, W] -> [1, C, 1, 1]
        std::string sum_name = output + "_sum";
        std::string sum_axes = sum_name + "_axes_0";
        std::string sum_keep_dims = sum_name + "_keep_dims_0";
        addIntArrayConstOp(block, sum_axes, {2, 3});
        addBoolScalarConstOp(block, sum_keep_dims, true);
        {
            auto* op = block->add_operations();
            op->set_type("reduce_sum");
            auto& inputs = *op->mutable_inputs();
            inputs["x"].add_arguments()->set_name(masked_name);
            inputs["axes"].add_arguments()->set_name(sum_axes);
            inputs["keep_dims"].add_arguments()->set_name(sum_keep_dims);
            setTensorOutputPooled4D(op, sum_name, channels);
        }

        // Mean = sum / count: [1, C, 1, 1] / [1, 1, 1, 1] -> [1, C, 1, 1]
        std::string mean_name = output + "_mean";
        {
            auto* op = block->add_operations();
            op->set_type("real_div");
            auto& inputs = *op->mutable_inputs();
            inputs["x"].add_arguments()->set_name(sum_name);
            inputs["y"].add_arguments()->set_name(mask_sum_name);
            setTensorOutputPooled4D(op, mean_name, channels);
        }

        // Max pooling (with mask adjustment)
        // mask_minus_one: [1, 1, H, W] - scalar -> [1, 1, H, W]
        std::string mask_minus_one = output + "_mask_minus_one";
        std::string mask_minus_one_y = mask_minus_one + "_y_0";
        addFloatScalarConstOp(block, mask_minus_one_y, 1.0f);
        {
            auto* op = block->add_operations();
            op->set_type("sub");
            auto& inputs = *op->mutable_inputs();
            inputs["x"].add_arguments()->set_name(mask);
            inputs["y"].add_arguments()->set_name(mask_minus_one_y);
            setTensorOutputMaskSpatial4D(op, mask_minus_one, m_board_y_size, m_board_x_size);
        }

        // x_for_max: [1, C, H, W] + [1, 1, H, W] -> [1, C, H, W]
        std::string x_for_max = output + "_x_for_max";
        {
            auto* op = block->add_operations();
            op->set_type("add");
            auto& inputs = *op->mutable_inputs();
            inputs["x"].add_arguments()->set_name(masked_name);
            inputs["y"].add_arguments()->set_name(mask_minus_one);
            setTensorOutput4D(op, x_for_max, channels, m_board_y_size, m_board_x_size);
        }

        // max: [1, C, H, W] -> [1, C, 1, 1]
        std::string max_name = output + "_max";
        std::string max_axes = max_name + "_axes_0";
        std::string max_keep_dims = max_name + "_keep_dims_0";
        addIntArrayConstOp(block, max_axes, {2, 3});
        addBoolScalarConstOp(block, max_keep_dims, true);
        {
            auto* op = block->add_operations();
            op->set_type("reduce_max");
            auto& inputs = *op->mutable_inputs();
            inputs["x"].add_arguments()->set_name(x_for_max);
            inputs["axes"].add_arguments()->set_name(max_axes);
            inputs["keep_dims"].add_arguments()->set_name(max_keep_dims);
            setTensorOutputPooled4D(op, max_name, channels);
        }

        // Mean scaled = mean * (sqrt(count) - 14) * 0.1
        // sqrt_mask_sum: [1, 1, 1, 1] -> [1, 1, 1, 1]
        std::string sqrt_mask_sum = output + "_sqrt_mask_sum";
        {
            auto* op = block->add_operations();
            op->set_type("sqrt");
            auto& inputs = *op->mutable_inputs();
            inputs["x"].add_arguments()->set_name(mask_sum_name);
            setTensorOutputMask4D(op, sqrt_mask_sum);
        }

        // sqrt_m14: [1, 1, 1, 1] - scalar -> [1, 1, 1, 1]
        std::string sqrt_m14 = output + "_sqrt_m14";
        std::string sqrt_m14_y = sqrt_m14 + "_y_0";
        addFloatScalarConstOp(block, sqrt_m14_y, 14.0f);
        {
            auto* op = block->add_operations();
            op->set_type("sub");
            auto& inputs = *op->mutable_inputs();
            inputs["x"].add_arguments()->set_name(sqrt_mask_sum);
            inputs["y"].add_arguments()->set_name(sqrt_m14_y);
            setTensorOutputMask4D(op, sqrt_m14);
        }

        // scaled_factor: [1, 1, 1, 1] * scalar -> [1, 1, 1, 1]
        std::string scaled_factor = output + "_scaled_factor";
        std::string scaled_factor_y = scaled_factor + "_y_0";
        addFloatScalarConstOp(block, scaled_factor_y, 0.1f);
        {
            auto* op = block->add_operations();
            op->set_type("mul");
            auto& inputs = *op->mutable_inputs();
            inputs["x"].add_arguments()->set_name(sqrt_m14);
            inputs["y"].add_arguments()->set_name(scaled_factor_y);
            setTensorOutputMask4D(op, scaled_factor);
        }

        // mean_scaled: [1, C, 1, 1] * [1, 1, 1, 1] -> [1, C, 1, 1]
        std::string mean_scaled = output + "_mean_scaled";
        {
            auto* op = block->add_operations();
            op->set_type("mul");
            auto& inputs = *op->mutable_inputs();
            inputs["x"].add_arguments()->set_name(mean_name);
            inputs["y"].add_arguments()->set_name(scaled_factor);
            setTensorOutputPooled4D(op, mean_scaled, channels);
        }

        // Squeeze spatial dimensions: [1, C, 1, 1] -> [1, C]
        std::string mean_flat = output + "_mean_flat";
        std::string mean_flat_axes = mean_flat + "_axes_0";
        addIntArrayConstOp(block, mean_flat_axes, {2, 3});
        {
            auto* op = block->add_operations();
            op->set_type("squeeze");
            auto& inputs = *op->mutable_inputs();
            inputs["x"].add_arguments()->set_name(mean_name);
            inputs["axes"].add_arguments()->set_name(mean_flat_axes);
            setTensorOutput2D(op, mean_flat, channels);
        }

        std::string mean_scaled_flat = output + "_mean_scaled_flat";
        std::string mean_scaled_flat_axes = mean_scaled_flat + "_axes_0";
        addIntArrayConstOp(block, mean_scaled_flat_axes, {2, 3});
        {
            auto* op = block->add_operations();
            op->set_type("squeeze");
            auto& inputs = *op->mutable_inputs();
            inputs["x"].add_arguments()->set_name(mean_scaled);
            inputs["axes"].add_arguments()->set_name(mean_scaled_flat_axes);
            setTensorOutput2D(op, mean_scaled_flat, channels);
        }

        std::string max_flat = output + "_max_flat";
        std::string max_flat_axes = max_flat + "_axes_0";
        addIntArrayConstOp(block, max_flat_axes, {2, 3});
        {
            auto* op = block->add_operations();
            op->set_type("squeeze");
            auto& inputs = *op->mutable_inputs();
            inputs["x"].add_arguments()->set_name(max_name);
            inputs["axes"].add_arguments()->set_name(max_flat_axes);
            setTensorOutput2D(op, max_flat, channels);
        }

        // Concatenate: [mean, mean_scaled, max] -> [1, 3*C]
        std::string concat_axis = output + "_concat_axis_0";
        std::string concat_interleave = output + "_concat_interleave_0";
        addIntScalarConstOp(block, concat_axis, 1);
        addBoolScalarConstOp(block, concat_interleave, false);
        {
            auto* op = block->add_operations();
            op->set_type("concat");
            auto& inputs = *op->mutable_inputs();
            inputs["values"].add_arguments()->set_name(mean_flat);
            inputs["values"].add_arguments()->set_name(mean_scaled_flat);
            inputs["values"].add_arguments()->set_name(max_flat);
            inputs["axis"].add_arguments()->set_name(concat_axis);
            inputs["interleave"].add_arguments()->set_name(concat_interleave);
            setTensorOutput2D(op, output, channels * 3);
        }
    }
}

void MILBuilder::addGlobalPoolingValueOps(CoreML::Specification::MILSpec::Block* block,
                                          const std::string& input,
                                          const std::string& mask,
                                          int channels,
                                          const std::string& output) {
    // KataGo value head global pooling produces: [mean, mean_scaled, mean_f3]
    // mean_scaled = mean * (sqrt(count) - 14) * 0.1
    // mean_f3 = mean * ((sqrt(count) - 14)^2 * 0.01 - 0.1)

    if (m_optimize_identity_mask) {
        // Optimized path: use precomputed constants
        const auto& mc = m_ops.getMaskConstants();

        // Mean pooling: sum / count -> [1, C, 1, 1]
        std::string sum_name = output + "_sum";
        std::string sum_axes = sum_name + "_axes_0";
        std::string sum_keep_dims = sum_name + "_keep_dims_0";
        addIntArrayConstOp(block, sum_axes, {2, 3});
        addBoolScalarConstOp(block, sum_keep_dims, true);
        {
            auto* op = block->add_operations();
            op->set_type("reduce_sum");
            auto& inputs = *op->mutable_inputs();
            inputs["x"].add_arguments()->set_name(input);
            inputs["axes"].add_arguments()->set_name(sum_axes);
            inputs["keep_dims"].add_arguments()->set_name(sum_keep_dims);
            setTensorOutputPooled4D(op, sum_name, channels);
        }

        std::string mean_name = output + "_mean";
        std::string mean_y = mean_name + "_y_0";
        addFloatScalarConstOp(block, mean_y, mc.mask_sum_reciprocal);
        {
            auto* op = block->add_operations();
            op->set_type("mul");
            auto& inputs = *op->mutable_inputs();
            inputs["x"].add_arguments()->set_name(sum_name);
            inputs["y"].add_arguments()->set_name(mean_y);
            setTensorOutputPooled4D(op, mean_name, channels);
        }

        // Mean scaled = mean * constant -> [1, C, 1, 1]
        std::string mean_scaled_name = output + "_mean_scaled";
        std::string mean_scaled_y = mean_scaled_name + "_y_0";
        addFloatScalarConstOp(block, mean_scaled_y, mc.mask_sum_sqrt_s14_m01);
        {
            auto* op = block->add_operations();
            op->set_type("mul");
            auto& inputs = *op->mutable_inputs();
            inputs["x"].add_arguments()->set_name(mean_name);
            inputs["y"].add_arguments()->set_name(mean_scaled_y);
            setTensorOutputPooled4D(op, mean_scaled_name, channels);
        }

        // Mean feature 3 = mean * constant -> [1, C, 1, 1]
        std::string mean_f3_name = output + "_mean_f3";
        std::string mean_f3_y = mean_f3_name + "_y_0";
        addFloatScalarConstOp(block, mean_f3_y, mc.mask_sum_sqrt_s14_m01_sq_s01);
        {
            auto* op = block->add_operations();
            op->set_type("mul");
            auto& inputs = *op->mutable_inputs();
            inputs["x"].add_arguments()->set_name(mean_name);
            inputs["y"].add_arguments()->set_name(mean_f3_y);
            setTensorOutputPooled4D(op, mean_f3_name, channels);
        }

        // Squeeze spatial dimensions: [N, C, 1, 1] -> [N, C]
        std::string mean_flat = output + "_mean_flat";
        std::string mean_flat_axes = mean_flat + "_axes_0";
        addIntArrayConstOp(block, mean_flat_axes, {2, 3});
        {
            auto* op = block->add_operations();
            op->set_type("squeeze");
            auto& inputs = *op->mutable_inputs();
            inputs["x"].add_arguments()->set_name(mean_name);
            inputs["axes"].add_arguments()->set_name(mean_flat_axes);
            setTensorOutput2D(op, mean_flat, channels);
        }

        std::string mean_scaled_flat = output + "_mean_scaled_flat";
        std::string mean_scaled_flat_axes = mean_scaled_flat + "_axes_0";
        addIntArrayConstOp(block, mean_scaled_flat_axes, {2, 3});
        {
            auto* op = block->add_operations();
            op->set_type("squeeze");
            auto& inputs = *op->mutable_inputs();
            inputs["x"].add_arguments()->set_name(mean_scaled_name);
            inputs["axes"].add_arguments()->set_name(mean_scaled_flat_axes);
            setTensorOutput2D(op, mean_scaled_flat, channels);
        }

        std::string mean_f3_flat = output + "_mean_f3_flat";
        std::string mean_f3_flat_axes = mean_f3_flat + "_axes_0";
        addIntArrayConstOp(block, mean_f3_flat_axes, {2, 3});
        {
            auto* op = block->add_operations();
            op->set_type("squeeze");
            auto& inputs = *op->mutable_inputs();
            inputs["x"].add_arguments()->set_name(mean_f3_name);
            inputs["axes"].add_arguments()->set_name(mean_f3_flat_axes);
            setTensorOutput2D(op, mean_f3_flat, channels);
        }

        // Concatenate: [mean, mean_scaled, mean_f3] -> [1, 3*C]
        std::string concat_axis = output + "_concat_axis_0";
        std::string concat_interleave = output + "_concat_interleave_0";
        addIntScalarConstOp(block, concat_axis, 1);
        addBoolScalarConstOp(block, concat_interleave, false);
        {
            auto* op = block->add_operations();
            op->set_type("concat");
            auto& inputs = *op->mutable_inputs();
            inputs["values"].add_arguments()->set_name(mean_flat);
            inputs["values"].add_arguments()->set_name(mean_scaled_flat);
            inputs["values"].add_arguments()->set_name(mean_f3_flat);
            inputs["axis"].add_arguments()->set_name(concat_axis);
            inputs["interleave"].add_arguments()->set_name(concat_interleave);
            setTensorOutput2D(op, output, channels * 3);
        }
    } else {
        // Full path with mask operations
        // Count valid positions: [1, 1, H, W] -> [1, 1, 1, 1]
        std::string mask_sum_name = output + "_mask_sum";
        std::string mask_sum_axes = mask_sum_name + "_axes_0";
        std::string mask_sum_keep_dims = mask_sum_name + "_keep_dims_0";
        addIntArrayConstOp(block, mask_sum_axes, {2, 3});
        addBoolScalarConstOp(block, mask_sum_keep_dims, true);
        {
            auto* op = block->add_operations();
            op->set_type("reduce_sum");
            auto& inputs = *op->mutable_inputs();
            inputs["x"].add_arguments()->set_name(mask);
            inputs["axes"].add_arguments()->set_name(mask_sum_axes);
            inputs["keep_dims"].add_arguments()->set_name(mask_sum_keep_dims);
            setTensorOutputMask4D(op, mask_sum_name);
        }

        // Masked input: [1, C, H, W] * [1, 1, H, W] -> [1, C, H, W]
        std::string masked_name = output + "_masked";
        {
            auto* op = block->add_operations();
            op->set_type("mul");
            auto& inputs = *op->mutable_inputs();
            inputs["x"].add_arguments()->set_name(input);
            inputs["y"].add_arguments()->set_name(mask);
            setTensorOutput4D(op, masked_name, channels, m_board_y_size, m_board_x_size);
        }

        // Sum masked values: [1, C, H, W] -> [1, C, 1, 1]
        std::string sum_name = output + "_sum";
        std::string sum_axes = sum_name + "_axes_0";
        std::string sum_keep_dims = sum_name + "_keep_dims_0";
        addIntArrayConstOp(block, sum_axes, {2, 3});
        addBoolScalarConstOp(block, sum_keep_dims, true);
        {
            auto* op = block->add_operations();
            op->set_type("reduce_sum");
            auto& inputs = *op->mutable_inputs();
            inputs["x"].add_arguments()->set_name(masked_name);
            inputs["axes"].add_arguments()->set_name(sum_axes);
            inputs["keep_dims"].add_arguments()->set_name(sum_keep_dims);
            setTensorOutputPooled4D(op, sum_name, channels);
        }

        // Mean = sum / count: [1, C, 1, 1] / [1, 1, 1, 1] -> [1, C, 1, 1]
        std::string mean_name = output + "_mean";
        {
            auto* op = block->add_operations();
            op->set_type("real_div");
            auto& inputs = *op->mutable_inputs();
            inputs["x"].add_arguments()->set_name(sum_name);
            inputs["y"].add_arguments()->set_name(mask_sum_name);
            setTensorOutputPooled4D(op, mean_name, channels);
        }

        // Compute (sqrt(count) - 14): [1, 1, 1, 1] -> [1, 1, 1, 1]
        std::string sqrt_mask_sum = output + "_sqrt_mask_sum";
        {
            auto* op = block->add_operations();
            op->set_type("sqrt");
            auto& inputs = *op->mutable_inputs();
            inputs["x"].add_arguments()->set_name(mask_sum_name);
            setTensorOutputMask4D(op, sqrt_mask_sum);
        }

        std::string sqrt_m14 = output + "_sqrt_m14";
        std::string sqrt_m14_y = sqrt_m14 + "_y_0";
        addFloatScalarConstOp(block, sqrt_m14_y, 14.0f);
        {
            auto* op = block->add_operations();
            op->set_type("sub");
            auto& inputs = *op->mutable_inputs();
            inputs["x"].add_arguments()->set_name(sqrt_mask_sum);
            inputs["y"].add_arguments()->set_name(sqrt_m14_y);
            setTensorOutputMask4D(op, sqrt_m14);
        }

        // Feature 2: Mean * (sqrt(count) - 14) * 0.1
        // scaled_factor: [1, 1, 1, 1] * scalar -> [1, 1, 1, 1]
        std::string scaled_factor = output + "_scaled_factor";
        std::string scaled_factor_y = scaled_factor + "_y_0";
        addFloatScalarConstOp(block, scaled_factor_y, 0.1f);
        {
            auto* op = block->add_operations();
            op->set_type("mul");
            auto& inputs = *op->mutable_inputs();
            inputs["x"].add_arguments()->set_name(sqrt_m14);
            inputs["y"].add_arguments()->set_name(scaled_factor_y);
            setTensorOutputMask4D(op, scaled_factor);
        }

        // mean_scaled: [1, C, 1, 1] * [1, 1, 1, 1] -> [1, C, 1, 1]
        std::string mean_scaled = output + "_mean_scaled";
        {
            auto* op = block->add_operations();
            op->set_type("mul");
            auto& inputs = *op->mutable_inputs();
            inputs["x"].add_arguments()->set_name(mean_name);
            inputs["y"].add_arguments()->set_name(scaled_factor);
            setTensorOutputPooled4D(op, mean_scaled, channels);
        }

        // Feature 3: Mean * ((sqrt(count) - 14)^2 * 0.01 - 0.1)
        // sqrt_m14_sq: [1, 1, 1, 1] * [1, 1, 1, 1] -> [1, 1, 1, 1]
        std::string sqrt_m14_sq = output + "_sqrt_m14_sq";
        {
            auto* op = block->add_operations();
            op->set_type("mul");
            auto& inputs = *op->mutable_inputs();
            inputs["x"].add_arguments()->set_name(sqrt_m14);
            inputs["y"].add_arguments()->set_name(sqrt_m14);
            setTensorOutputMask4D(op, sqrt_m14_sq);
        }

        // sq_01: [1, 1, 1, 1] * scalar -> [1, 1, 1, 1]
        std::string sq_01 = output + "_sq_01";
        std::string sq_01_y = sq_01 + "_y_0";
        addFloatScalarConstOp(block, sq_01_y, 0.01f);
        {
            auto* op = block->add_operations();
            op->set_type("mul");
            auto& inputs = *op->mutable_inputs();
            inputs["x"].add_arguments()->set_name(sqrt_m14_sq);
            inputs["y"].add_arguments()->set_name(sq_01_y);
            setTensorOutputMask4D(op, sq_01);
        }

        // f3_factor: [1, 1, 1, 1] - scalar -> [1, 1, 1, 1]
        std::string f3_factor = output + "_f3_factor";
        std::string f3_factor_y = f3_factor + "_y_0";
        addFloatScalarConstOp(block, f3_factor_y, 0.1f);
        {
            auto* op = block->add_operations();
            op->set_type("sub");
            auto& inputs = *op->mutable_inputs();
            inputs["x"].add_arguments()->set_name(sq_01);
            inputs["y"].add_arguments()->set_name(f3_factor_y);
            setTensorOutputMask4D(op, f3_factor);
        }

        // mean_f3: [1, C, 1, 1] * [1, 1, 1, 1] -> [1, C, 1, 1]
        std::string mean_f3 = output + "_mean_f3";
        {
            auto* op = block->add_operations();
            op->set_type("mul");
            auto& inputs = *op->mutable_inputs();
            inputs["x"].add_arguments()->set_name(mean_name);
            inputs["y"].add_arguments()->set_name(f3_factor);
            setTensorOutputPooled4D(op, mean_f3, channels);
        }

        // Squeeze spatial dimensions: [1, C, 1, 1] -> [1, C]
        std::string mean_flat = output + "_mean_flat";
        std::string mean_flat_axes = mean_flat + "_axes_0";
        addIntArrayConstOp(block, mean_flat_axes, {2, 3});
        {
            auto* op = block->add_operations();
            op->set_type("squeeze");
            auto& inputs = *op->mutable_inputs();
            inputs["x"].add_arguments()->set_name(mean_name);
            inputs["axes"].add_arguments()->set_name(mean_flat_axes);
            setTensorOutput2D(op, mean_flat, channels);
        }

        std::string mean_scaled_flat = output + "_mean_scaled_flat";
        std::string mean_scaled_flat_axes = mean_scaled_flat + "_axes_0";
        addIntArrayConstOp(block, mean_scaled_flat_axes, {2, 3});
        {
            auto* op = block->add_operations();
            op->set_type("squeeze");
            auto& inputs = *op->mutable_inputs();
            inputs["x"].add_arguments()->set_name(mean_scaled);
            inputs["axes"].add_arguments()->set_name(mean_scaled_flat_axes);
            setTensorOutput2D(op, mean_scaled_flat, channels);
        }

        std::string mean_f3_flat = output + "_mean_f3_flat";
        std::string mean_f3_flat_axes = mean_f3_flat + "_axes_0";
        addIntArrayConstOp(block, mean_f3_flat_axes, {2, 3});
        {
            auto* op = block->add_operations();
            op->set_type("squeeze");
            auto& inputs = *op->mutable_inputs();
            inputs["x"].add_arguments()->set_name(mean_f3);
            inputs["axes"].add_arguments()->set_name(mean_f3_flat_axes);
            setTensorOutput2D(op, mean_f3_flat, channels);
        }

        // Concatenate: [mean, mean_scaled, mean_f3] -> [1, 3*C]
        std::string concat_axis = output + "_concat_axis_0";
        std::string concat_interleave = output + "_concat_interleave_0";
        addIntScalarConstOp(block, concat_axis, 1);
        addBoolScalarConstOp(block, concat_interleave, false);
        {
            auto* op = block->add_operations();
            op->set_type("concat");
            auto& inputs = *op->mutable_inputs();
            inputs["values"].add_arguments()->set_name(mean_flat);
            inputs["values"].add_arguments()->set_name(mean_scaled_flat);
            inputs["values"].add_arguments()->set_name(mean_f3_flat);
            inputs["axis"].add_arguments()->set_name(concat_axis);
            inputs["interleave"].add_arguments()->set_name(concat_interleave);
            setTensorOutput2D(op, output, channels * 3);
        }
    }
}

// ============================================================================
// Network Component Builders
// ============================================================================

// ---------------------------------------------------------------------------
// Transformer blocks (MIL). Layout is NCHW [B, C, H, W]; spatial positions
// (H*W, ordered y*W+x) are treated as the attention sequence. RoPE is applied
// via a fixed pair-rotation matmul plus host-precomputed cos/sin tables, which
// keeps every tensor rank <= 4 (ANE-friendly).
// ---------------------------------------------------------------------------

std::string MILBuilder::addTransformerRMSNorm(CoreML::Specification::MILSpec::Block* block,
                                              const std::string& input,
                                              const TransformerRMSNormDesc& desc,
                                              const std::string& mask,
                                              const std::string& prefix) {
    const int C = desc.num_channels;
    const int H = m_board_y_size, W = m_board_x_size;
    auto emit2 = [&](const std::string& type, const std::string& x, const std::string& y,
                     const std::string& out, const std::vector<int64_t>& dims) {
        auto* op = block->add_operations();
        op->set_type(type);
        (*op->mutable_inputs())["x"].add_arguments()->set_name(x);
        (*op->mutable_inputs())["y"].add_arguments()->set_name(y);
        setShape(op, out, dims);
    };

    // RMSNorm reduction core: square -> mean over channels -> rsqrt. In FP16 mode compute this
    // core in FP32 (cast input up, flip the working dtype so the core's op outputs + eps scalar are
    // FP32, then cast 1/rms back down). The FP16 channel reduction loses too much precision on the
    // ANE; only this core is FP32 - the scaling/weight/mask below stay FP16. No addConstOp lives in
    // the flipped window, so weight serialization is unaffected.
    auto savedDtype = m_weight_dtype;
    std::string sqSrc = input;
    if (m_use_fp16) {
        sqSrc = genVarName(prefix + "_in32");
        addCastOp(block, input, sqSrc, "fp32", {-1, C, H, W});
        m_weight_dtype = CoreML::Specification::MILSpec::DataType::FLOAT32;
    }
    std::string sq = genVarName(prefix + "_sq");
    emit2("mul", sqSrc, sqSrc, sq, {-1, C, H, W});
    // meanSq = reduce_mean(sq, axes=[1]) over channels. reduce_mean (not reduce_sum) is used so
    // the accumulator stays ~O(activation^2) instead of summing hundreds of channels, which can
    // overflow FP16 (and the FP16 accumulation on ANE) for large activations.
    std::string meanSq = genVarName(prefix + "_meansq");
    {
        std::string axesName = meanSq + "_axes";
        std::string keepName = meanSq + "_keep";
        addIntArrayConstOp(block, axesName, {1});
        addBoolScalarConstOp(block, keepName, true);
        auto* op = block->add_operations();
        op->set_type("reduce_mean");
        (*op->mutable_inputs())["x"].add_arguments()->set_name(sq);
        (*op->mutable_inputs())["axes"].add_arguments()->set_name(axesName);
        (*op->mutable_inputs())["keep_dims"].add_arguments()->set_name(keepName);
        setShape(op, meanSq, {-1, 1, H, W});
    }
    // MIL rsqrt computes 1/sqrt(x + epsilon); supply epsilon directly.
    std::string epsName = prefix + "_eps";
    addFloatScalarConstOp(block, epsName, desc.epsilon);
    std::string invCore = genVarName(prefix + "_inv");
    {
        auto* op = block->add_operations();
        op->set_type("rsqrt");
        (*op->mutable_inputs())["x"].add_arguments()->set_name(meanSq);
        (*op->mutable_inputs())["epsilon"].add_arguments()->set_name(epsName);
        setShape(op, invCore, {-1, 1, H, W});
    }
    std::string inv = invCore;
    if (m_use_fp16) {
        m_weight_dtype = savedDtype;
        inv = genVarName(prefix + "_inv16");
        addCastOp(block, invCore, inv, "fp16", {-1, 1, H, W});
    }
    std::string normalized = genVarName(prefix + "_norm");
    emit2("mul", input, inv, normalized, {-1, C, H, W});
    std::string weightName = prefix + "_weight";
    addConstOp(block, weightName, desc.weight, {1, static_cast<int64_t>(C), 1, 1});
    std::string scaled = genVarName(prefix + "_scaled");
    emit2("mul", normalized, weightName, scaled, {-1, C, H, W});
    std::string out = genVarName(prefix + "_out");
    emit2("mul", scaled, mask, out, {-1, C, H, W});
    return out;
}

std::string MILBuilder::addTrunkRMSNorm(CoreML::Specification::MILSpec::Block* block,
                                        const std::string& input,
                                        const RMSNormLayerDesc& desc,
                                        const ActivationLayerDesc& act,
                                        const std::string& mask,
                                        const std::string& prefix) {
    const int C = desc.num_channels;
    const int H = m_board_y_size, W = m_board_x_size;
    auto emit2 = [&](const std::string& type, const std::string& x, const std::string& y,
                     const std::string& out, const std::vector<int64_t>& dims) {
        auto* op = block->add_operations();
        op->set_type(type);
        (*op->mutable_inputs())["x"].add_arguments()->set_name(x);
        (*op->mutable_inputs())["y"].add_arguments()->set_name(y);
        setShape(op, out, dims);
    };
    auto reduceSum = [&](const std::string& x, const std::string& out, const std::vector<int32_t>& axes,
                         const std::vector<int64_t>& dims) {
        std::string axesName = out + "_axes";
        std::string keepName = out + "_keep";
        addIntArrayConstOp(block, axesName, axes);
        addBoolScalarConstOp(block, keepName, true);
        auto* op = block->add_operations();
        op->set_type("reduce_sum");
        (*op->mutable_inputs())["x"].add_arguments()->set_name(x);
        (*op->mutable_inputs())["axes"].add_arguments()->set_name(axesName);
        (*op->mutable_inputs())["keep_dims"].add_arguments()->set_name(keepName);
        setShape(op, out, dims);
    };

    // Variance core (mask -> square -> reduce -> rsqrt) in FP32 when in FP16 mode. The trunk-tip
    // norm in particular reduces over many elements and loses too much precision in FP16 on the
    // ANE; compute the core in FP32 and cast 1/rms back to FP16. Only the core is FP32 - gamma/beta,
    // the activation and the final mask below stay FP16. No addConstOp lives in the flipped window.
    auto savedDtype = m_weight_dtype;
    std::string tinput = input;
    std::string tmask = mask;
    if (m_use_fp16) {
        tinput = genVarName(prefix + "_in32");
        addCastOp(block, input, tinput, "fp32", {-1, C, H, W});
        tmask = genVarName(prefix + "_mask32");
        addCastOp(block, mask, tmask, "fp32", {-1, 1, H, W});
        m_weight_dtype = CoreML::Specification::MILSpec::DataType::FLOAT32;
    }
    std::string masked = genVarName(prefix + "_premask");
    emit2("mul", tinput, tmask, masked, {-1, C, H, W});
    std::string sq = genVarName(prefix + "_sq");
    emit2("mul", masked, masked, sq, {-1, C, H, W});

    std::string meanSq;
    std::vector<int64_t> denomDims;
    if (desc.spatial) {
        // Mean of squares over valid positions and channels. A reduce_sum over C*H*W elements
        // overflows FP16 (e.g. trunk tip with large activations on ANE -> inf -> rsqrt 0 ->
        // collapse). Instead take reduce_mean over all of C,H,W (masked positions are zero) and
        // rescale by totalPositions/validCount to restrict the mean to valid positions.
        std::string meanAll = genVarName(prefix + "_meanall");
        {
            std::string axesName = meanAll + "_axes", keepName = meanAll + "_keep";
            addIntArrayConstOp(block, axesName, {1, 2, 3});
            addBoolScalarConstOp(block, keepName, true);
            auto* op = block->add_operations();
            op->set_type("reduce_mean");
            (*op->mutable_inputs())["x"].add_arguments()->set_name(sq);
            (*op->mutable_inputs())["axes"].add_arguments()->set_name(axesName);
            (*op->mutable_inputs())["keep_dims"].add_arguments()->set_name(keepName);
            setShape(op, meanAll, {-1, 1, 1, 1});
        }
        std::string count = genVarName(prefix + "_count");
        reduceSum(tmask, count, {1, 2, 3}, {-1, 1, 1, 1});  // valid positions (<= H*W, no overflow)
        std::string totalPosName = prefix + "_totalpos";
        addFloatScalarConstOp(block, totalPosName, static_cast<float>(H * W));
        std::string scaleF = genVarName(prefix + "_scalef");
        emit2("real_div", totalPosName, count, scaleF, {-1, 1, 1, 1});  // totalPos / validCount
        meanSq = genVarName(prefix + "_meansq");
        emit2("mul", meanAll, scaleF, meanSq, {-1, 1, 1, 1});
        denomDims = {-1, 1, 1, 1};
    } else {
        meanSq = genVarName(prefix + "_meansq");
        std::string axesName = meanSq + "_axes";
        std::string keepName = meanSq + "_keep";
        addIntArrayConstOp(block, axesName, {1});
        addBoolScalarConstOp(block, keepName, true);
        auto* op = block->add_operations();
        op->set_type("reduce_mean");
        (*op->mutable_inputs())["x"].add_arguments()->set_name(sq);
        (*op->mutable_inputs())["axes"].add_arguments()->set_name(axesName);
        (*op->mutable_inputs())["keep_dims"].add_arguments()->set_name(keepName);
        setShape(op, meanSq, {-1, 1, H, W});
        denomDims = {-1, 1, H, W};
    }

    // MIL rsqrt computes 1/sqrt(x + epsilon); supply epsilon directly.
    std::string epsName = prefix + "_eps";
    addFloatScalarConstOp(block, epsName, desc.epsilon);
    std::string invCore = genVarName(prefix + "_inv");
    {
        auto* op = block->add_operations();
        op->set_type("rsqrt");
        (*op->mutable_inputs())["x"].add_arguments()->set_name(meanSq);
        (*op->mutable_inputs())["epsilon"].add_arguments()->set_name(epsName);
        setShape(op, invCore, denomDims);
    }
    std::string inv = invCore;
    if (m_use_fp16) {
        m_weight_dtype = savedDtype;
        inv = genVarName(prefix + "_inv16");
        addCastOp(block, invCore, inv, "fp16", denomDims);
    }
    std::string normalized = genVarName(prefix + "_norm");
    emit2("mul", input, inv, normalized, {-1, C, H, W});
    std::string gammaName = prefix + "_gamma";
    std::string betaName = prefix + "_beta";
    addConstOp(block, gammaName, desc.gamma, {1, static_cast<int64_t>(C), 1, 1});
    addConstOp(block, betaName, desc.beta, {1, static_cast<int64_t>(C), 1, 1});
    std::string scaled = genVarName(prefix + "_scaled");
    emit2("mul", normalized, gammaName, scaled, {-1, C, H, W});
    std::string biased = genVarName(prefix + "_biased");
    emit2("add", scaled, betaName, biased, {-1, C, H, W});

    std::string activated;
    if (act.activation_type == ActivationType::Silu) {
        activated = genVarName(prefix + "_act");
        addSiluOps(block, biased, activated, 4, C);
    } else if (act.activation_type == ActivationType::Mish) {
        activated = genVarName(prefix + "_act");
        addMishOps(block, biased, activated, 4, C);
    } else if (act.activation_type == ActivationType::ReLU) {
        activated = genVarName(prefix + "_act");
        auto* op = block->add_operations();
        op->set_type("relu");
        (*op->mutable_inputs())["x"].add_arguments()->set_name(biased);
        setShape(op, activated, {-1, C, H, W});
    } else {
        activated = biased;
    }
    std::string out = genVarName(prefix + "_out");
    emit2("mul", activated, mask, out, {-1, C, H, W});
    return out;
}

std::string MILBuilder::buildTransformerAttentionBlock(CoreML::Specification::MILSpec::Block* block,
                                                       const std::string& input,
                                                       const TransformerAttentionBlockDesc& desc,
                                                       const std::string& mask,
                                                       const std::string& prefix) {
    const int C = desc.q_proj.in_channels;
    const int H = m_board_y_size, W = m_board_x_size;
    const int seq = H * W;
    const int numHeads = desc.num_heads, numKVHeads = desc.num_kv_heads;
    const int qHeadDim = desc.q_head_dim, vHeadDim = desc.v_head_dim;
    const int qTotal = numHeads * qHeadDim, kTotal = numKVHeads * qHeadDim, vTotal = numKVHeads * vHeadDim;

    auto reshape = [&](const std::string& in, const std::string& out, const std::vector<int32_t>& shapeVals,
                       const std::vector<int64_t>& dims) {
        std::string shapeName = out + "_shape";
        addIntArrayConstOp(block, shapeName, shapeVals);
        auto* op = block->add_operations();
        op->set_type("reshape");
        (*op->mutable_inputs())["x"].add_arguments()->set_name(in);
        (*op->mutable_inputs())["shape"].add_arguments()->set_name(shapeName);
        setShape(op, out, dims);
    };
    auto transpose = [&](const std::string& in, const std::string& out, const std::vector<int32_t>& perm,
                         const std::vector<int64_t>& dims) {
        std::string permName = out + "_perm";
        addIntArrayConstOp(block, permName, perm);
        auto* op = block->add_operations();
        op->set_type("transpose");
        (*op->mutable_inputs())["x"].add_arguments()->set_name(in);
        (*op->mutable_inputs())["perm"].add_arguments()->set_name(permName);
        setShape(op, out, dims);
    };
    auto matmul = [&](const std::string& x, const std::string& y, const std::string& out,
                      const std::vector<int64_t>& dims, bool transX, bool transY) {
        std::string txName = out + "_tx", tyName = out + "_ty";
        addBoolScalarConstOp(block, txName, transX);
        addBoolScalarConstOp(block, tyName, transY);
        auto* op = block->add_operations();
        op->set_type("matmul");
        (*op->mutable_inputs())["x"].add_arguments()->set_name(x);
        (*op->mutable_inputs())["y"].add_arguments()->set_name(y);
        (*op->mutable_inputs())["transpose_x"].add_arguments()->set_name(txName);
        (*op->mutable_inputs())["transpose_y"].add_arguments()->set_name(tyName);
        setShape(op, out, dims);
    };
    auto binary = [&](const std::string& type, const std::string& x, const std::string& y,
                      const std::string& out, const std::vector<int64_t>& dims) {
        auto* op = block->add_operations();
        op->set_type(type);
        (*op->mutable_inputs())["x"].add_arguments()->set_name(x);
        (*op->mutable_inputs())["y"].add_arguments()->set_name(y);
        setShape(op, out, dims);
    };

    std::string normed = addTransformerRMSNorm(block, input, desc.pre_ln, mask, prefix + "_ln");
    std::string nhwc = genVarName(prefix + "_nhwc");
    transpose(normed, nhwc, {0, 2, 3, 1}, {-1, H, W, C});
    std::string x2d = genVarName(prefix + "_x2d");
    reshape(nhwc, x2d, {-1, C}, {-1, C});
    // Q/K/V projection matmuls in FP32 (non-spatial, per KataGo's FP16 convention): they reduce over
    // C channels and the ANE's FP16 accumulation loses too much precision for wide models. Weights
    // stay fp16-stored (cast up at runtime); output cast back to FP16 for the FP16 head reshapes.
    auto proj = [&](const MatMulLayerDesc& w, const std::string& nm, int total) {
        std::string wName = nm + "_w";
        addConstOp(block, wName, w.weights, w.getWeightShape());
        std::string out = genVarName(nm);
        if (m_nonspatial_fp32) {
            std::string x32 = castFixed(block, x2d, "fp32", {-1, C});
            std::string w32 = castFixed(block, wName, "fp32", w.getWeightShape());
            auto sd = m_weight_dtype;
            m_weight_dtype = CoreML::Specification::MILSpec::DataType::FLOAT32;
            std::string o32 = genVarName(nm + "_f32");
            matmul(x32, w32, o32, {-1, total}, false, false);
            m_weight_dtype = sd;
            out = castFixed(block, o32, "fp16", {-1, total});
        } else {
            matmul(x2d, wName, out, {-1, total}, false, false);
        }
        return out;
    };
    std::string q2d = proj(desc.q_proj, prefix + "_q", qTotal);
    std::string k2d = proj(desc.k_proj, prefix + "_k", kTotal);
    std::string v2d = proj(desc.v_proj, prefix + "_v", vTotal);
    auto toHeads = [&](const std::string& in2d, const std::string& nm, int nh, int hd) {
        std::string r = genVarName(nm + "_r");
        reshape(in2d, r, {-1, seq, nh, hd}, {-1, seq, nh, hd});
        std::string t = genVarName(nm + "_t");
        transpose(r, t, {0, 2, 1, 3}, {-1, nh, seq, hd});
        return t;
    };
    std::string qh = toHeads(q2d, prefix + "_qh", numHeads, qHeadDim);
    std::string kh = toHeads(k2d, prefix + "_kh", numKVHeads, qHeadDim);
    std::string vh = toHeads(v2d, prefix + "_vh", numKVHeads, vHeadDim);

    if (desc.use_rope) {
        const int numPairs = qHeadDim / 2;
        const int numPairsPerDim = numPairs / 2;
        const int dimHalf = qHeadDim / 2;
        auto applyRope = [&](const std::string& x, int nh, const std::string& tag) {
            std::vector<float> cosFull(static_cast<size_t>(nh) * seq * qHeadDim, 0.0f);
            std::vector<float> sinFull(static_cast<size_t>(nh) * seq * qHeadDim, 0.0f);
            for (int h = 0; h < nh; h++) {
                int kvh = (h * numKVHeads) / nh;
                for (int xy = 0; xy < seq; xy++) {
                    int y = xy / W;
                    int x = xy % W;
                    for (int p = 0; p < numPairs; p++) {
                        float angle = 0.0f;
                        if (desc.learnable_rope) {
                            float fx = desc.rope_freqs[(kvh * numPairs + p) * 2 + 0];
                            float fy = desc.rope_freqs[(kvh * numPairs + p) * 2 + 1];
                            angle = static_cast<float>(x) * fx + static_cast<float>(y) * fy;
                        } else {
                            if (p < numPairsPerDim) {
                                float freq = 1.0f / std::pow(desc.rope_theta, static_cast<float>(2 * p) / dimHalf);
                                angle = static_cast<float>(y) * freq;
                            } else {
                                int pAdj = p - numPairsPerDim;
                                float freq = 1.0f / std::pow(desc.rope_theta, static_cast<float>(2 * pAdj) / dimHalf);
                                angle = static_cast<float>(x) * freq;
                            }
                        }
                        float c = std::cos(angle), s = std::sin(angle);
                        size_t base = (static_cast<size_t>(h) * seq + xy) * qHeadDim + 2 * p;
                        cosFull[base] = c; cosFull[base + 1] = c;
                        sinFull[base] = s; sinFull[base + 1] = s;
                    }
                }
            }
            std::vector<float> R(static_cast<size_t>(qHeadDim) * qHeadDim, 0.0f);
            for (int p = 0; p < numPairs; p++) {
                R[(2 * p) * qHeadDim + (2 * p + 1)] = 1.0f;
                R[(2 * p + 1) * qHeadDim + (2 * p)] = -1.0f;
            }
            std::string cosName = prefix + "_" + tag + "_cos";
            std::string sinName = prefix + "_" + tag + "_sin";
            std::string rName = prefix + "_" + tag + "_R";
            addConstOp(block, cosName, cosFull, {1, nh, seq, qHeadDim});
            addConstOp(block, sinName, sinFull, {1, nh, seq, qHeadDim});
            // Rank-4 [1,1,qd,qd] so matmul batch dims broadcast cleanly against [B,nh,seq,qd].
            addConstOp(block, rName, R, {1, 1, qHeadDim, qHeadDim});
            std::string rotated = genVarName(prefix + "_" + tag + "_rot");
            matmul(x, rName, rotated, {-1, nh, seq, qHeadDim}, false, false);
            std::string xc = genVarName(prefix + "_" + tag + "_xc");
            binary("mul", x, cosName, xc, {-1, nh, seq, qHeadDim});
            std::string rs = genVarName(prefix + "_" + tag + "_rs");
            binary("mul", rotated, sinName, rs, {-1, nh, seq, qHeadDim});
            std::string out = genVarName(prefix + "_" + tag + "_rope");
            binary("add", xc, rs, out, {-1, nh, seq, qHeadDim});
            return out;
        };
        qh = applyRope(qh, numHeads, "q");
        kh = applyRope(kh, numKVHeads, "k");
    }

    // GQA: when numKVHeads < numHeads, repeat each KV head groupSize times along the head
    // axis (axis 1) so query head h consumes kv head (h / groupSize). RoPE has already been
    // applied above to the unexpanded kh (kh = applyRope(kh, numKVHeads, "k")), mirroring the
    // GPU path (metallayers.swift repeatKVHeads runs AFTER applyRope). We slice each KV head
    // and concat its copies consecutively, so the resulting head index is kv*groupSize + g;
    // query head h then maps to kv = h/groupSize == (h*numKVHeads)/numHeads (exact divisor,
    // the same formula the qh RoPE table uses) == Eigen's kvh = h/kvGroupSize. slice_by_size +
    // concat (not reshape+broadcast) avoids the dynamic -1 batch broadcast pitfall, same as the
    // GPU code. The repeat is required so the scores (qh@kh^T) and attnOut (attn@vh) matmuls see
    // matching [B,numHeads,...] batch dims instead of numHeads vs numKVHeads (no broadcast).
    if (numKVHeads != numHeads) {
        const int groupSize = numHeads / numKVHeads;
        auto repeatKVHeads = [&](const std::string& x, const std::string& tag, int headDim) {
            std::vector<std::string> parts;
            parts.reserve(static_cast<size_t>(numKVHeads) * groupSize);
            for (int kv = 0; kv < numKVHeads; kv++) {
                for (int g = 0; g < groupSize; g++) {
                    std::string part = genVarName(prefix + "_" + tag + "_slc");
                    std::string beginName = part + "_begin", sizeName = part + "_size";
                    addIntArrayConstOp(block, beginName, {0, kv, 0, 0});
                    addIntArrayConstOp(block, sizeName, {-1, 1, seq, headDim});
                    auto* sop = block->add_operations();
                    sop->set_type("slice_by_size");
                    (*sop->mutable_inputs())["x"].add_arguments()->set_name(x);
                    (*sop->mutable_inputs())["begin"].add_arguments()->set_name(beginName);
                    (*sop->mutable_inputs())["size"].add_arguments()->set_name(sizeName);
                    setShape(sop, part, {-1, 1, seq, headDim});
                    parts.push_back(part);
                }
            }
            std::string out = genVarName(prefix + "_" + tag + "_exp");
            std::string axisName = out + "_axis", interleaveName = out + "_interleave";
            addIntScalarConstOp(block, axisName, 1);
            addBoolScalarConstOp(block, interleaveName, false);
            auto* cop = block->add_operations();
            cop->set_type("concat");
            auto& cin = *cop->mutable_inputs();
            for (const std::string& part : parts)
                cin["values"].add_arguments()->set_name(part);
            cin["axis"].add_arguments()->set_name(axisName);
            cin["interleave"].add_arguments()->set_name(interleaveName);
            setShape(cop, out, {-1, numHeads, seq, headDim});
            return out;
        };
        kh = repeatKVHeads(kh, "khrep", qHeadDim);
        vh = repeatKVHeads(vh, "vhrep", vHeadDim);
    }

    std::string scores = genVarName(prefix + "_scores");
    matmul(qh, kh, scores, {-1, numHeads, seq, seq}, false, true);
    std::string scaleName = prefix + "_scale";
    addFloatScalarConstOp(block, scaleName, 1.0f / std::sqrt(static_cast<float>(qHeadDim)));
    std::string scaled = genVarName(prefix + "_sc");
    binary("mul", scores, scaleName, scaled, {-1, numHeads, seq, seq});

    // mask [B,1,H,W] -> [B,1,1,seq] directly (contiguous reshape; H,W already trailing so the
    // row-major flatten gives seq index xy=y*W+x). No transpose -> avoids the reshape-after-
    // transpose issue, and is also correct for non-full boards.
    std::string maskSeq = genVarName(prefix + "_mseq");
    reshape(mask, maskSeq, {-1, 1, 1, seq}, {-1, 1, 1, seq});
    std::string oneName = prefix + "_one";
    addFloatScalarConstOp(block, oneName, 1.0f);
    std::string mm1 = genVarName(prefix + "_mm1");
    binary("sub", maskSeq, oneName, mm1, {-1, 1, 1, seq});
    // Use an FP16-safe magnitude: 1e9 overflows FP16 to +inf, and for valid keys
    // (maskSeq-1 == 0) the product 0 * inf becomes NaN, poisoning the whole softmax.
    // 1e4 is well within FP16 range and exp(score - 1e4) still underflows to 0.
    std::string bigName = prefix + "_big";
    addFloatScalarConstOp(block, bigName, 1.0e4f);
    std::string keyBias = genVarName(prefix + "_kb");
    binary("mul", mm1, bigName, keyBias, {-1, 1, 1, seq});
    std::string scoresMasked = genVarName(prefix + "_scm");
    binary("add", scaled, keyBias, scoresMasked, {-1, numHeads, seq, seq});

    std::string attn = genVarName(prefix + "_attn");
    {
        std::string axisName = attn + "_axis";
        addIntScalarConstOp(block, axisName, 3);
        auto* op = block->add_operations();
        op->set_type("softmax");
        (*op->mutable_inputs())["x"].add_arguments()->set_name(scoresMasked);
        (*op->mutable_inputs())["axis"].add_arguments()->set_name(axisName);
        setShape(op, attn, {-1, numHeads, seq, seq});
    }

    std::string attnOut = genVarName(prefix + "_ao");
    matmul(attn, vh, attnOut, {-1, numHeads, seq, vHeadDim}, false, false);

    // Output projection, done per-head to avoid reshape-after-transpose: CoreML's reshape
    // ignores an immediately-preceding transpose, so merging [head,dim]->channels after a
    // transpose scrambles the data. Instead slice each head from attnOut (head is the
    // contiguous axis 1), reshape (leading-merge only), matmul its weight slice, and sum.
    //   out[b,s,c] = sum_h sum_d attnOut[b,h,s,d] * outProj.weights[(h*vHeadDim+d)*outC + c]
    const int outC = desc.out_proj.out_channels;
    std::string proj2d;
    for (int h = 0; h < numHeads; h++) {
        std::string aoh = genVarName(prefix + "_aoh");
        {
            std::string beginName = aoh + "_begin", sizeName = aoh + "_size";
            addIntArrayConstOp(block, beginName, {0, h, 0, 0});
            addIntArrayConstOp(block, sizeName, {-1, 1, seq, vHeadDim});
            auto* op = block->add_operations();
            op->set_type("slice_by_size");
            (*op->mutable_inputs())["x"].add_arguments()->set_name(attnOut);
            (*op->mutable_inputs())["begin"].add_arguments()->set_name(beginName);
            (*op->mutable_inputs())["size"].add_arguments()->set_name(sizeName);
            setShape(op, aoh, {-1, 1, seq, vHeadDim});
        }
        std::string aoh2d = genVarName(prefix + "_aoh2d");
        reshape(aoh, aoh2d, {-1, vHeadDim}, {-1, vHeadDim});  // [B*seq, vHeadDim]
        std::string wh = prefix + "_ow" + std::to_string(h);
        std::vector<float> whData(static_cast<size_t>(vHeadDim) * outC);
        for (int d = 0; d < vHeadDim; d++)
            for (int c = 0; c < outC; c++)
                whData[d * outC + c] = desc.out_proj.weights[static_cast<size_t>(h * vHeadDim + d) * outC + c];
        addConstOp(block, wh, whData, {vHeadDim, outC});
        std::string contrib = genVarName(prefix + "_contrib");
        matmul(aoh2d, wh, contrib, {-1, outC}, false, false);
        if (h == 0) {
            proj2d = contrib;
        } else {
            std::string acc = genVarName(prefix + "_acc");
            binary("add", proj2d, contrib, acc, {-1, outC});
            proj2d = acc;
        }
    }
    std::string projNHWC = genVarName(prefix + "_pnhwc");
    reshape(proj2d, projNHWC, {-1, H, W, C}, {-1, H, W, C});
    std::string projNCHW = genVarName(prefix + "_pnchw");
    transpose(projNHWC, projNCHW, {0, 3, 1, 2}, {-1, C, H, W});
    std::string maskedOut = genVarName(prefix + "_masked");
    binary("mul", projNCHW, mask, maskedOut, {-1, C, H, W});
    std::string out = genVarName(prefix + "_out");
    binary("add", input, maskedOut, out, {-1, C, H, W});
    return out;
}

std::string MILBuilder::buildTransformerFFNBlock(CoreML::Specification::MILSpec::Block* block,
                                                 const std::string& input,
                                                 const TransformerFFNBlockDesc& desc,
                                                 const std::string& mask,
                                                 const std::string& prefix) {
    const int C = desc.num_channels;
    const int ffn = desc.ffn_channels;
    const int H = m_board_y_size, W = m_board_x_size;

    if (!desc.use_swiglu) {
        throw std::runtime_error(desc.name + ": non-SwiGLU transformer FFN not supported in CoreML backend");
    }

    auto reshape = [&](const std::string& in, const std::string& out, const std::vector<int32_t>& shapeVals,
                       const std::vector<int64_t>& dims) {
        std::string shapeName = out + "_shape";
        addIntArrayConstOp(block, shapeName, shapeVals);
        auto* op = block->add_operations();
        op->set_type("reshape");
        (*op->mutable_inputs())["x"].add_arguments()->set_name(in);
        (*op->mutable_inputs())["shape"].add_arguments()->set_name(shapeName);
        setShape(op, out, dims);
    };
    auto transpose = [&](const std::string& in, const std::string& out, const std::vector<int32_t>& perm,
                         const std::vector<int64_t>& dims) {
        std::string permName = out + "_perm";
        addIntArrayConstOp(block, permName, perm);
        auto* op = block->add_operations();
        op->set_type("transpose");
        (*op->mutable_inputs())["x"].add_arguments()->set_name(in);
        (*op->mutable_inputs())["perm"].add_arguments()->set_name(permName);
        setShape(op, out, dims);
    };
    auto matmul = [&](const std::string& x, const std::string& y, const std::string& out,
                      const std::vector<int64_t>& dims) {
        std::string txName = out + "_tx", tyName = out + "_ty";
        addBoolScalarConstOp(block, txName, false);
        addBoolScalarConstOp(block, tyName, false);
        auto* op = block->add_operations();
        op->set_type("matmul");
        (*op->mutable_inputs())["x"].add_arguments()->set_name(x);
        (*op->mutable_inputs())["y"].add_arguments()->set_name(y);
        (*op->mutable_inputs())["transpose_x"].add_arguments()->set_name(txName);
        (*op->mutable_inputs())["transpose_y"].add_arguments()->set_name(tyName);
        setShape(op, out, dims);
    };
    auto binary = [&](const std::string& type, const std::string& x, const std::string& y,
                      const std::string& out, const std::vector<int64_t>& dims) {
        auto* op = block->add_operations();
        op->set_type(type);
        (*op->mutable_inputs())["x"].add_arguments()->set_name(x);
        (*op->mutable_inputs())["y"].add_arguments()->set_name(y);
        setShape(op, out, dims);
    };

    std::string normed = addTransformerRMSNorm(block, input, desc.pre_ln, mask, prefix + "_ln");
    std::string nhwc = genVarName(prefix + "_nhwc");
    transpose(normed, nhwc, {0, 2, 3, 1}, {-1, H, W, C});
    std::string x2d = genVarName(prefix + "_x2d");
    reshape(nhwc, x2d, {-1, C}, {-1, C});

    // FFN matmuls in FP32 (weights cast up at runtime, stored fp16) — KataGo's FP16 convention is
    // spatial(convs)=FP16, non-spatial(matmuls)=FP32 (see openclbackend.cpp). The ANE accumulates
    // FP16 matmuls in FP16, which loses too much precision over C/ffn; run them in FP32 instead.
    std::string w1 = prefix + "_w1";
    addConstOp(block, w1, desc.linear1.weights, desc.linear1.getWeightShape());
    std::string wg = prefix + "_wg";
    addConstOp(block, wg, desc.linear_gate.weights, desc.linear_gate.getWeightShape());
    std::string w2 = prefix + "_w2";
    addConstOp(block, w2, desc.linear2.weights, desc.linear2.getWeightShape());

    auto savedDtype = m_weight_dtype;
    std::string mx2d = x2d, mw1 = w1, mwg = wg, mw2 = w2;
    if (m_nonspatial_fp32) {
        mx2d = castFixed(block, x2d, "fp32", {-1, C});
        mw1 = castFixed(block, w1, "fp32", desc.linear1.getWeightShape());
        mwg = castFixed(block, wg, "fp32", desc.linear_gate.getWeightShape());
        mw2 = castFixed(block, w2, "fp32", desc.linear2.getWeightShape());
        m_weight_dtype = CoreML::Specification::MILSpec::DataType::FLOAT32;
    }
    std::string a = genVarName(prefix + "_a");
    matmul(mx2d, mw1, a, {-1, ffn});
    std::string g = genVarName(prefix + "_g");
    matmul(mx2d, mwg, g, {-1, ffn});

    std::string sig = genVarName(prefix + "_sig");
    {
        auto* op = block->add_operations();
        op->set_type("sigmoid");
        (*op->mutable_inputs())["x"].add_arguments()->set_name(a);
        setShape(op, sig, {-1, ffn});
    }
    std::string siluA = genVarName(prefix + "_silu");
    binary("mul", a, sig, siluA, {-1, ffn});
    std::string h = genVarName(prefix + "_h");
    binary("mul", siluA, g, h, {-1, ffn});

    std::string oCore = genVarName(prefix + "_o");
    matmul(h, mw2, oCore, {-1, C});
    std::string o = oCore;
    if (m_nonspatial_fp32) {
        m_weight_dtype = savedDtype;
        o = castFixed(block, oCore, "fp16", {-1, C});
    }

    std::string oNHWC = genVarName(prefix + "_onhwc");
    reshape(o, oNHWC, {-1, H, W, C}, {-1, H, W, C});
    std::string oNCHW = genVarName(prefix + "_onchw");
    transpose(oNHWC, oNCHW, {0, 3, 1, 2}, {-1, C, H, W});
    std::string maskedOut = genVarName(prefix + "_masked");
    binary("mul", oNCHW, mask, maskedOut, {-1, C, H, W});
    std::string out = genVarName(prefix + "_out");
    binary("add", input, maskedOut, out, {-1, C, H, W});
    return out;
}

std::string MILBuilder::buildTrunk(CoreML::Specification::MILSpec::Block* block,
                                   const std::string& spatial_input,
                                   const std::string& global_input,
                                   const std::string& mask,
                                   const std::string* meta_input) {
    const auto& trunk = m_model.trunk;

    // Initial conv
    std::string x = genVarName("trunk_init_conv");
    addConvOp(block, spatial_input, trunk.initial_conv, x);

    // Global projection
    std::string global_bias = genVarName("trunk_global_proj");
    addMatMulOp(block, global_input, trunk.initial_matmul, global_bias);

    // Reshape global bias to [batch, C, 1, 1]
    // Create shape const first (matching Python structure)
    std::string global_bias_reshaped = genVarName("trunk_global_reshape");
    std::string reshape_shape_name = global_bias_reshaped + "_shape_0";
    // Use -1 for batch to infer from input, explicit channel count
    addIntArrayConstOp(block, reshape_shape_name, {-1, static_cast<int32_t>(trunk.initial_conv.out_channels), 1, 1});
    {
        auto* op = block->add_operations();
        op->set_type("reshape");
        // "name" attribute
        auto& name_attr = (*op->mutable_attributes())["name"];
        name_attr.mutable_type()->mutable_tensortype()->set_datatype(
            CoreML::Specification::MILSpec::DataType::STRING);
        name_attr.mutable_immediatevalue()->mutable_tensor()->mutable_strings()->add_values(global_bias_reshaped);
        // Inputs
        auto& inputs = *op->mutable_inputs();
        inputs["x"].add_arguments()->set_name(global_bias);
        inputs["shape"].add_arguments()->set_name(reshape_shape_name);
        // Output with dimensions [batch, C, 1, 1]
        setTensorOutputPooled4D(op, global_bias_reshaped, trunk.initial_conv.out_channels);
    }

    // Add global bias
    std::string x_with_global = genVarName("trunk_add_global");
    {
        auto* op = block->add_operations();
        op->set_type("add");
        auto& inputs = *op->mutable_inputs();
        inputs["x"].add_arguments()->set_name(x);
        inputs["y"].add_arguments()->set_name(global_bias_reshaped);
        // Output with 4D shape [batch, C, H, W]
        setTensorOutput4D(op, x_with_global, trunk.trunk_num_channels, m_board_y_size, m_board_x_size);
    }
    x = x_with_global;

    // Add metadata bias if present
    if (trunk.sgf_metadata_encoder.has_value() && meta_input != nullptr) {
        std::string meta_bias = buildSGFMetadataEncoder(block, *meta_input, *trunk.sgf_metadata_encoder);

        // Reshape meta bias
        std::string meta_bias_reshaped = genVarName("trunk_meta_reshape");
        std::string meta_bias_shape_name = meta_bias_reshaped + "_shape_0";
        // Use -1 for batch to infer from input, explicit channel count
        addIntArrayConstOp(block, meta_bias_shape_name, {-1, static_cast<int32_t>(trunk.trunk_num_channels), 1, 1});
        {
            auto* op = block->add_operations();
            op->set_type("reshape");
            auto& inputs = *op->mutable_inputs();
            inputs["x"].add_arguments()->set_name(meta_bias);
            inputs["shape"].add_arguments()->set_name(meta_bias_shape_name);
            // Output with 4D shape [batch, C, 1, 1]
            setTensorOutputPooled4D(op, meta_bias_reshaped, trunk.trunk_num_channels);
        }

        // Add meta bias
        std::string x_with_meta = genVarName("trunk_add_meta");
        {
            auto* op = block->add_operations();
            op->set_type("add");
            auto& inputs = *op->mutable_inputs();
            inputs["x"].add_arguments()->set_name(x);
            inputs["y"].add_arguments()->set_name(meta_bias_reshaped);
            // Output with 4D shape [batch, C, H, W]
            setTensorOutput4D(op, x_with_meta, trunk.trunk_num_channels, m_board_y_size, m_board_x_size);
        }
        x = x_with_meta;
    }

    // Apply initial mask
    std::string x_masked = genVarName("trunk_init_mask");
    {
        auto* op = block->add_operations();
        op->set_type("mul");
        auto& inputs = *op->mutable_inputs();
        inputs["x"].add_arguments()->set_name(x);
        inputs["y"].add_arguments()->set_name(mask);
        // Output with 4D shape [batch, C, H, W]
        setTensorOutput4D(op, x_masked, trunk.trunk_num_channels, m_board_y_size, m_board_x_size);
    }
    x = x_masked;

    // Process residual blocks
    for (size_t i = 0; i < trunk.blocks.size(); i++) {
        const auto& entry = trunk.blocks[i];
        std::string prefix = "trunk_block_" + std::to_string(i);

        if (entry.block_kind == ORDINARY_BLOCK_KIND) {
            const auto& block_desc = std::get<ResidualBlockDesc>(*entry.block);
            x = buildResidualBlock(block, x, block_desc, mask, prefix);
        } else if (entry.block_kind == GLOBAL_POOLING_BLOCK_KIND) {
            const auto& block_desc = std::get<GlobalPoolingResidualBlockDesc>(*entry.block);
            x = buildGlobalPoolingResidualBlock(block, x, block_desc, mask, prefix);
        } else if (entry.block_kind == NESTED_BOTTLENECK_BLOCK_KIND) {
            const auto& block_desc = std::get<NestedBottleneckResidualBlockDesc>(*entry.block);
            x = buildNestedBottleneckBlock(block, x, block_desc, mask, prefix);
        } else if (entry.block_kind == TRANSFORMER_ATTENTION_BLOCK_KIND) {
            const auto& block_desc = std::get<TransformerAttentionBlockDesc>(*entry.block);
            x = buildTransformerAttentionBlock(block, x, block_desc, mask, prefix);
        } else if (entry.block_kind == TRANSFORMER_FFN_BLOCK_KIND) {
            const auto& block_desc = std::get<TransformerFFNBlockDesc>(*entry.block);
            x = buildTransformerFFNBlock(block, x, block_desc, mask, prefix);
        }
    }

    // Trunk tip
    std::string trunk_out;
    if (trunk.trunk_norm_kind == TRUNK_NORM_KIND_STANDARD) {
        trunk_out = genVarName("trunk_tip");
        addBatchNormActivationOps(block, x, trunk.trunk_tip_bn, trunk.trunk_tip_activation, mask, trunk_out);
    } else {
        trunk_out = addTrunkRMSNorm(block, x, trunk.trunk_tip_rms_norm, trunk.trunk_tip_activation, mask, "trunk_tip_rms");
    }

    return trunk_out;
}

std::string MILBuilder::buildResidualBlock(CoreML::Specification::MILSpec::Block* block,
                                           const std::string& input,
                                           const ResidualBlockDesc& block_desc,
                                           const std::string& mask,
                                           const std::string& prefix) {
    // Pre BN + activation
    std::string pre_out = genVarName(prefix + "_pre");
    addBatchNormActivationOps(block, input, block_desc.pre_bn, block_desc.pre_activation, mask, pre_out);

    // First conv
    std::string conv1_out = genVarName(prefix + "_conv1");
    addConvOp(block, pre_out, block_desc.regular_conv, conv1_out);

    // Mid BN + activation
    std::string mid_out = genVarName(prefix + "_mid");
    addBatchNormActivationOps(block, conv1_out, block_desc.mid_bn, block_desc.mid_activation, mask, mid_out);

    // Second conv
    std::string conv2_out = genVarName(prefix + "_conv2");
    addConvOp(block, mid_out, block_desc.final_conv, conv2_out);

    // Residual add
    std::string output = genVarName(prefix + "_residual");
    {
        auto* op = block->add_operations();
        op->set_type("add");
        auto& inputs = *op->mutable_inputs();
        inputs["x"].add_arguments()->set_name(conv2_out);
        inputs["y"].add_arguments()->set_name(input);
        // Set proper 4D output type [1, C, H, W]
        setTensorOutput4D(op, output, block_desc.final_conv.out_channels, m_board_y_size, m_board_x_size);
    }

    return output;
}

std::string MILBuilder::buildGlobalPoolingResidualBlock(CoreML::Specification::MILSpec::Block* block,
                                                         const std::string& input,
                                                         const GlobalPoolingResidualBlockDesc& block_desc,
                                                         const std::string& mask,
                                                         const std::string& prefix) {
    // Pre BN + activation
    std::string pre_out = genVarName(prefix + "_pre");
    addBatchNormActivationOps(block, input, block_desc.pre_bn, block_desc.pre_activation, mask, pre_out);

    // Regular conv
    std::string regular_out = genVarName(prefix + "_regular");
    addConvOp(block, pre_out, block_desc.regular_conv, regular_out);

    // Gpool conv
    std::string gpool_conv_out = genVarName(prefix + "_gpool_conv");
    addConvOp(block, pre_out, block_desc.gpool_conv, gpool_conv_out);

    // Gpool BN + activation
    std::string gpool_bn_out = genVarName(prefix + "_gpool_bn");
    addBatchNormActivationOps(block, gpool_conv_out, block_desc.gpool_bn, block_desc.gpool_activation, mask, gpool_bn_out);

    // Global pooling (FP32 when m_nonspatial_fp32 -- see addGlobalPoolingFp32). Feeds a bias back
    // into the whole trunk, so the FP16 spatial sum must not lose precision for wide trunks.
    std::string gpool_features = genVarName(prefix + "_gpool_features");
    addGlobalPoolingFp32(block, gpool_bn_out, mask, block_desc.gpool_conv.out_channels, gpool_features,
                         /*valueVariant=*/false);

    // Project to bias
    std::string gpool_bias = genVarName(prefix + "_gpool_bias");
    addMatMulOp(block, gpool_features, block_desc.gpool_to_bias_mul, gpool_bias);

    // Reshape bias
    std::string gpool_bias_reshaped = genVarName(prefix + "_gpool_bias_reshape");
    std::string gpool_bias_reshape_shape = gpool_bias_reshaped + "_shape_0";
    // Use -1 for batch to infer from input, explicit channel count
    addIntArrayConstOp(block, gpool_bias_reshape_shape, {-1, static_cast<int32_t>(block_desc.regular_conv.out_channels), 1, 1});
    {
        auto* op = block->add_operations();
        op->set_type("reshape");
        auto& inputs = *op->mutable_inputs();
        inputs["x"].add_arguments()->set_name(gpool_bias);
        inputs["shape"].add_arguments()->set_name(gpool_bias_reshape_shape);
        // Output is [batch, regular_conv.out_channels, 1, 1]
        setTensorOutputPooled4D(op, gpool_bias_reshaped, block_desc.regular_conv.out_channels);
    }

    // Add bias to regular path
    std::string combined = genVarName(prefix + "_combined");
    {
        auto* op = block->add_operations();
        op->set_type("add");
        auto& inputs = *op->mutable_inputs();
        inputs["x"].add_arguments()->set_name(regular_out);
        inputs["y"].add_arguments()->set_name(gpool_bias_reshaped);
        // Output is [1, regular_conv.out_channels, H, W]
        setTensorOutput4D(op, combined, block_desc.regular_conv.out_channels, m_board_y_size, m_board_x_size);
    }

    // Mid BN + activation
    std::string mid_out = genVarName(prefix + "_mid");
    addBatchNormActivationOps(block, combined, block_desc.mid_bn, block_desc.mid_activation, mask, mid_out);

    // Final conv
    std::string final_conv_out = genVarName(prefix + "_final");
    addConvOp(block, mid_out, block_desc.final_conv, final_conv_out);

    // Residual add
    std::string output = genVarName(prefix + "_residual");
    {
        auto* op = block->add_operations();
        op->set_type("add");
        auto& inputs = *op->mutable_inputs();
        inputs["x"].add_arguments()->set_name(final_conv_out);
        inputs["y"].add_arguments()->set_name(input);
        // Set proper 4D output type [1, C, H, W]
        setTensorOutput4D(op, output, block_desc.final_conv.out_channels, m_board_y_size, m_board_x_size);
    }

    return output;
}

std::string MILBuilder::buildNestedBottleneckBlock(CoreML::Specification::MILSpec::Block* block,
                                                    const std::string& input,
                                                    const NestedBottleneckResidualBlockDesc& block_desc,
                                                    const std::string& mask,
                                                    const std::string& prefix) {
    // Pre BN + activation
    std::string pre_out = genVarName(prefix + "_pre");
    addBatchNormActivationOps(block, input, block_desc.pre_bn, block_desc.pre_activation, mask, pre_out);

    // Pre conv (bottleneck reduction)
    std::string pre_conv_out = genVarName(prefix + "_pre_conv");
    addConvOp(block, pre_out, block_desc.pre_conv, pre_conv_out);

    std::string x = pre_conv_out;

    // Process nested blocks
    for (size_t i = 0; i < block_desc.blocks.size(); i++) {
        const auto& entry = block_desc.blocks[i];
        std::string nested_prefix = prefix + "_nested_" + std::to_string(i);

        if (entry.block_kind == ORDINARY_BLOCK_KIND) {
            const auto& nested = std::get<ResidualBlockDesc>(*entry.block);
            x = buildResidualBlock(block, x, nested, mask, nested_prefix);
        } else if (entry.block_kind == GLOBAL_POOLING_BLOCK_KIND) {
            const auto& nested = std::get<GlobalPoolingResidualBlockDesc>(*entry.block);
            x = buildGlobalPoolingResidualBlock(block, x, nested, mask, nested_prefix);
        } else if (entry.block_kind == TRANSFORMER_ATTENTION_BLOCK_KIND) {
            const auto& nested = std::get<TransformerAttentionBlockDesc>(*entry.block);
            x = buildTransformerAttentionBlock(block, x, nested, mask, nested_prefix);
        } else if (entry.block_kind == TRANSFORMER_FFN_BLOCK_KIND) {
            const auto& nested = std::get<TransformerFFNBlockDesc>(*entry.block);
            x = buildTransformerFFNBlock(block, x, nested, mask, nested_prefix);
        }
    }

    // Post BN + activation
    std::string post_out = genVarName(prefix + "_post");
    addBatchNormActivationOps(block, x, block_desc.post_bn, block_desc.post_activation, mask, post_out);

    // Post conv (bottleneck expansion)
    std::string post_conv_out = genVarName(prefix + "_post_conv");
    addConvOp(block, post_out, block_desc.post_conv, post_conv_out);

    // Residual add
    std::string output = genVarName(prefix + "_residual");
    {
        auto* op = block->add_operations();
        op->set_type("add");
        auto& inputs = *op->mutable_inputs();
        inputs["x"].add_arguments()->set_name(post_conv_out);
        inputs["y"].add_arguments()->set_name(input);
        // Set proper 4D output type [1, C, H, W]
        setTensorOutput4D(op, output, block_desc.post_conv.out_channels, m_board_y_size, m_board_x_size);
    }

    return output;
}

void MILBuilder::buildPolicyHead(CoreML::Specification::MILSpec::Block* block,
                                 const std::string& trunk_out,
                                 const std::string& mask,
                                 std::string& policy_out,
                                 std::string& pass_out) {
    const auto& ph = m_model.policy_head;

    // P1 conv
    std::string p1 = genVarName("policy_p1");
    addConvOp(block, trunk_out, ph.p1_conv, p1);

    // G1 conv + BN + activation
    std::string g1_conv = genVarName("policy_g1_conv");
    addConvOp(block, trunk_out, ph.g1_conv, g1_conv);

    std::string g1 = genVarName("policy_g1");
    addBatchNormActivationOps(block, g1_conv, ph.g1_bn, ph.g1_activation, mask, g1);

    // Global pooling on G1 (FP32 when m_nonspatial_fp32; feeds the policy bias / policyKLDiv).
    std::string g1_pooled = genVarName("policy_g1_pool");
    addGlobalPoolingFp32(block, g1, mask, ph.g1_conv.out_channels, g1_pooled, /*valueVariant=*/false);

    // Project to spatial bias
    std::string gpool_bias = genVarName("policy_gpool_bias");
    addMatMulOp(block, g1_pooled, ph.gpool_to_bias_mul, gpool_bias);

    // Reshape bias
    std::string gpool_bias_reshaped = genVarName("policy_gpool_bias_reshape");
    std::string policy_gpool_reshape_shape = gpool_bias_reshaped + "_shape_0";
    // Use -1 for batch to infer from input, explicit channel count
    addIntArrayConstOp(block, policy_gpool_reshape_shape, {-1, static_cast<int32_t>(ph.p1_conv.out_channels), 1, 1});
    {
        auto* op = block->add_operations();
        op->set_type("reshape");
        auto& inputs = *op->mutable_inputs();
        inputs["x"].add_arguments()->set_name(gpool_bias);
        inputs["shape"].add_arguments()->set_name(policy_gpool_reshape_shape);
        // Output is [batch, p1_conv.out_channels, 1, 1]
        setTensorOutputPooled4D(op, gpool_bias_reshaped, ph.p1_conv.out_channels);
    }

    // Add bias to P1
    std::string p1_biased = genVarName("policy_p1_biased");
    {
        auto* op = block->add_operations();
        op->set_type("add");
        auto& inputs = *op->mutable_inputs();
        inputs["x"].add_arguments()->set_name(p1);
        inputs["y"].add_arguments()->set_name(gpool_bias_reshaped);
        // Output is [1, p1_conv.out_channels, H, W]
        setTensorOutput4D(op, p1_biased, ph.p1_conv.out_channels, m_board_y_size, m_board_x_size);
    }

    // P1 BN + activation
    std::string p1_activated = genVarName("policy_p1_act");
    addBatchNormActivationOps(block, p1_biased, ph.p1_bn, ph.p1_activation, mask, p1_activated);

    // P2 conv -> policy output
    // Mixed precision uses _fp16 suffix for this intermediate op; cast ops later rename to base name
    policy_out = (m_use_fp16 && !m_use_fp16_io) ? "policy_p2_conv_fp16" : "policy_p2_conv";
    addConvOp(block, p1_activated, ph.p2_conv, policy_out);

    // Pass move
    if (ph.gpool_to_pass_mul2.has_value()) {
        // v15+: two-layer pass (first layer fused matmul+bias -> linear)
        std::string pass_biased = genVarName("policy_pass_biased");
        addLinearOp(block, g1_pooled, ph.gpool_to_pass_mul, *ph.gpool_to_pass_bias, pass_biased);

        // Activation
        std::string pass_activated = genVarName("policy_pass_act");
        if (ph.pass_activation->activation_type == ActivationType::ReLU) {
            auto* op = block->add_operations();
            op->set_type("relu");
            auto& inputs = *op->mutable_inputs();
            inputs["x"].add_arguments()->set_name(pass_biased);
            setTensorOutput2D(op, pass_activated, ph.gpool_to_pass_mul.out_channels);
        } else if (ph.pass_activation->activation_type == ActivationType::Mish) {
            addMishOps(block, pass_biased, pass_activated, 2, ph.gpool_to_pass_mul.out_channels);
        } else if (ph.pass_activation->activation_type == ActivationType::Silu) {
            addSiluOps(block, pass_biased, pass_activated, 2, ph.gpool_to_pass_mul.out_channels);
        } else {
            pass_activated = pass_biased;
        }

        // Mixed precision: _fp16 intermediate, cast ops rename to base name
        pass_out = (m_use_fp16 && !m_use_fp16_io) ? "policy_pass_fp16" : "policy_pass";
        addMatMulOp(block, pass_activated, *ph.gpool_to_pass_mul2, pass_out);
    } else {
        // Pre-v15: single layer pass
        // Mixed precision: _fp16 intermediate, cast ops rename to base name (pre-v15)
        pass_out = (m_use_fp16 && !m_use_fp16_io) ? "policy_pass_fp16" : "policy_pass";
        addMatMulOp(block, g1_pooled, ph.gpool_to_pass_mul, pass_out);
    }
}

void MILBuilder::buildValueHead(CoreML::Specification::MILSpec::Block* block,
                                const std::string& trunk_out,
                                const std::string& mask,
                                std::string& value_out,
                                std::string& ownership_out,
                                std::string& score_value_out) {
    const auto& vh = m_model.value_head;

    // V1 conv + BN + activation
    std::string v1_conv = genVarName("value_v1_conv");
    addConvOp(block, trunk_out, vh.v1_conv, v1_conv);

    std::string v1 = genVarName("value_v1");
    addBatchNormActivationOps(block, v1_conv, vh.v1_bn, vh.v1_activation, mask, v1);

    // Global pooling (value head version; FP32 when m_nonspatial_fp32).
    std::string v1_pooled = genVarName("value_v1_pool");
    addGlobalPoolingFp32(block, v1, mask, vh.v1_conv.out_channels, v1_pooled, /*valueVariant=*/true);

    // V2: linear + activation (fused matmul+bias -> linear)
    std::string v2_bias = genVarName("value_v2_bias");
    addLinearOp(block, v1_pooled, vh.v2_mul, vh.v2_bias, v2_bias);

    std::string v2 = genVarName("value_v2");
    if (vh.v2_activation.activation_type == ActivationType::ReLU) {
        auto* op = block->add_operations();
        op->set_type("relu");
        auto& inputs = *op->mutable_inputs();
        inputs["x"].add_arguments()->set_name(v2_bias);
        setTensorOutput2D(op, v2, vh.v2_mul.out_channels);
    } else if (vh.v2_activation.activation_type == ActivationType::Mish) {
        addMishOps(block, v2_bias, v2, 2, vh.v2_mul.out_channels);
    } else if (vh.v2_activation.activation_type == ActivationType::Silu) {
        addSiluOps(block, v2_bias, v2, 2, vh.v2_mul.out_channels);
    } else {
        v2 = v2_bias;
    }

    // V3: linear -> value output (fused matmul+bias -> linear)
    // Mixed precision: _fp16 intermediate, cast ops rename to base name
    value_out = (m_use_fp16 && !m_use_fp16_io) ? "value_v3_bias_fp16" : "value_v3_bias";
    addLinearOp(block, v2, vh.v3_mul, vh.v3_bias, value_out);

    // SV3: linear -> score value output (fused matmul+bias -> linear)
    // Mixed precision: _fp16 intermediate, cast ops rename to base name
    score_value_out = (m_use_fp16 && !m_use_fp16_io) ? "value_sv3_bias_fp16" : "value_sv3_bias";
    addLinearOp(block, v2, vh.sv3_mul, vh.sv3_bias, score_value_out);

    // Ownership conv
    // Mixed precision: _fp16 intermediate, cast ops rename to base name
    ownership_out = (m_use_fp16 && !m_use_fp16_io) ? "value_ownership_conv_fp16" : "value_ownership_conv";
    addConvOp(block, v1, vh.v_ownership_conv, ownership_out);
}

std::string MILBuilder::buildSGFMetadataEncoder(CoreML::Specification::MILSpec::Block* block,
                                                const std::string& meta_input,
                                                const SGFMetadataEncoderDesc& encoder) {
    // Layer 1 (fused matmul+bias -> linear)
    std::string bias1 = genVarName("meta_bias1");
    addLinearOp(block, meta_input, encoder.mul1, encoder.bias1, bias1);

    std::string act1 = genVarName("meta_act1");
    if (encoder.act1.activation_type == ActivationType::ReLU) {
        auto* op = block->add_operations();
        op->set_type("relu");
        auto& inputs = *op->mutable_inputs();
        inputs["x"].add_arguments()->set_name(bias1);
        setTensorOutput2D(op, act1, encoder.mul1.out_channels);
    } else if (encoder.act1.activation_type == ActivationType::Mish) {
        addMishOps(block, bias1, act1, 2, encoder.mul1.out_channels);
    } else if (encoder.act1.activation_type == ActivationType::Silu) {
        addSiluOps(block, bias1, act1, 2, encoder.mul1.out_channels);
    } else {
        // Identity activation - create identity op to preserve type information
        auto* op = block->add_operations();
        op->set_type("identity");
        auto& inputs = *op->mutable_inputs();
        inputs["x"].add_arguments()->set_name(bias1);
        setTensorOutput2D(op, act1, encoder.mul1.out_channels);
    }

    // Layer 2 (fused matmul+bias -> linear)
    std::string bias2 = genVarName("meta_bias2");
    addLinearOp(block, act1, encoder.mul2, encoder.bias2, bias2);

    std::string act2 = genVarName("meta_act2");
    if (encoder.act2.activation_type == ActivationType::ReLU) {
        auto* op = block->add_operations();
        op->set_type("relu");
        auto& inputs = *op->mutable_inputs();
        inputs["x"].add_arguments()->set_name(bias2);
        setTensorOutput2D(op, act2, encoder.mul2.out_channels);
    } else if (encoder.act2.activation_type == ActivationType::Mish) {
        addMishOps(block, bias2, act2, 2, encoder.mul2.out_channels);
    } else if (encoder.act2.activation_type == ActivationType::Silu) {
        addSiluOps(block, bias2, act2, 2, encoder.mul2.out_channels);
    } else {
        // Identity activation - create identity op to preserve type information
        auto* op = block->add_operations();
        op->set_type("identity");
        auto& inputs = *op->mutable_inputs();
        inputs["x"].add_arguments()->set_name(bias2);
        setTensorOutput2D(op, act2, encoder.mul2.out_channels);
    }

    // Layer 3 (output)
    std::string output = genVarName("meta_output");
    addMatMulOp(block, act2, encoder.mul3, output);

    return output;
}

}  // namespace katagocoreml
