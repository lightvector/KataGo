// katagocoreml - Standalone C++ KataGo to Core ML Converter
// Copyright (c) 2025, Chin-Chang Yang

#include "MILBuilder.hpp"
#include "MILBlob/Fp16.hpp"
#include <stdexcept>

// Include generated protobuf headers
#include "MIL.pb.h"

namespace katagocoreml {

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
    , m_var_counter(0) {}

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
    // Register weight for blob storage
    m_ops.registerWeight(name, data, shape);

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

    if (m_use_fp16) {
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
    inputs["weight"].add_arguments()->set_name(weight_name);
    inputs["x"].add_arguments()->set_name(input);

    // Output with dimensions [batch, out_channels, height, width]
    auto* out = op->add_outputs();
    out->set_name(output);
    auto* out_type = out->mutable_type()->mutable_tensortype();
    out_type->set_datatype(m_weight_dtype);
    out_type->set_rank(4);
    setBatchDimension(out_type);
    out_type->add_dimensions()->mutable_constant()->set_size(layer.out_channels);
    out_type->add_dimensions()->mutable_constant()->set_size(m_board_y_size);
    out_type->add_dimensions()->mutable_constant()->set_size(m_board_x_size);
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

    // Add matmul operation
    auto* op = block->add_operations();
    op->set_type("matmul");
    auto& inputs = *op->mutable_inputs();
    inputs["transpose_x"].add_arguments()->set_name(transpose_x_name);
    inputs["transpose_y"].add_arguments()->set_name(transpose_y_name);
    inputs["x"].add_arguments()->set_name(input);
    inputs["y"].add_arguments()->set_name(weight_name);

    // Output with 2D shape [batch, out_channels]
    auto* out = op->add_outputs();
    out->set_name(output);
    auto* out_type = out->mutable_type()->mutable_tensortype();
    out_type->set_datatype(m_weight_dtype);
    out_type->set_rank(2);
    setBatchDimension(out_type);
    out_type->add_dimensions()->mutable_constant()->set_size(layer.out_channels);
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

    // Add linear operation
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
        }
    }

    // Trunk tip
    std::string trunk_out = genVarName("trunk_tip");
    addBatchNormActivationOps(block, x, trunk.trunk_tip_bn, trunk.trunk_tip_activation, mask, trunk_out);

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

    // Global pooling
    std::string gpool_features = genVarName(prefix + "_gpool_features");
    addGlobalPoolingOps(block, gpool_bn_out, mask, block_desc.gpool_conv.out_channels, gpool_features);

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

    // Global pooling on G1
    std::string g1_pooled = genVarName("policy_g1_pool");
    addGlobalPoolingOps(block, g1, mask, ph.g1_conv.out_channels, g1_pooled);

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

    // Global pooling (value head version)
    std::string v1_pooled = genVarName("value_v1_pool");
    addGlobalPoolingValueOps(block, v1, mask, vh.v1_conv.out_channels, v1_pooled);

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
