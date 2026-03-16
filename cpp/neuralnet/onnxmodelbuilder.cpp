// Builds an ONNX computational graph from a KataGo ModelDesc.
// Uses the ONNX protobuf API (onnx-ml.pb.h) to construct a ModelProto
// that can be loaded directly by ONNX Runtime.

#include "../neuralnet/onnxmodelbuilder.h"
#include "../neuralnet/activations.h"
#include "../core/global.h"

#include <onnx/onnx-ml.pb.h>

#include <string>
#include <vector>

using namespace std;

static string uniqueName(int& nameCounter, const string& prefix) {
  return prefix + "_" + to_string(nameCounter++);
}

// =====================================================================
// Helper: Add a float tensor initializer to the graph
// =====================================================================
static string addInitializer(
  onnx::GraphProto* graph,
  const string& name,
  const vector<int64_t>& shape,
  const float* data,
  size_t numElements
) {
  onnx::TensorProto* tensor = graph->add_initializer();
  tensor->set_name(name);
  tensor->set_data_type(onnx::TensorProto_DataType_FLOAT);
  for(int64_t d : shape)
    tensor->add_dims(d);
  tensor->set_raw_data(data, numElements * sizeof(float));
  return name;
}

static string addInitializer(
  onnx::GraphProto* graph,
  const string& name,
  const vector<int64_t>& shape,
  const vector<float>& data
) {
  return addInitializer(graph, name, shape, data.data(), data.size());
}

// Add a scalar float constant
static string addScalarInitializer(onnx::GraphProto* graph, const string& name, float value) {
  return addInitializer(graph, name, {}, &value, 1);
}

// Add a 1D int64 constant tensor
static string addInt64Initializer(
  onnx::GraphProto* graph,
  const string& name,
  const vector<int64_t>& data
) {
  onnx::TensorProto* tensor = graph->add_initializer();
  tensor->set_name(name);
  tensor->set_data_type(onnx::TensorProto_DataType_INT64);
  tensor->add_dims((int64_t)data.size());
  tensor->set_raw_data(data.data(), data.size() * sizeof(int64_t));
  return name;
}

// =====================================================================
// Helper: Add ONNX graph node
// =====================================================================

// Generic node with n inputs, 1 output
static onnx::NodeProto* addNode(
  onnx::GraphProto* graph,
  const string& opType,
  const vector<string>& inputs,
  const string& outputName
) {
  onnx::NodeProto* node = graph->add_node();
  node->set_op_type(opType);
  for(const auto& inp : inputs)
    node->add_input(inp);
  node->add_output(outputName);
  return node;
}

// Add an attribute (int) to a node
static void setAttrInt(onnx::NodeProto* node, const string& attrName, int64_t value) {
  onnx::AttributeProto* attr = node->add_attribute();
  attr->set_name(attrName);
  attr->set_type(onnx::AttributeProto_AttributeType_INT);
  attr->set_i(value);
}

// Add an attribute (ints) to a node
static void setAttrInts(onnx::NodeProto* node, const string& attrName, const vector<int64_t>& values) {
  onnx::AttributeProto* attr = node->add_attribute();
  attr->set_name(attrName);
  attr->set_type(onnx::AttributeProto_AttributeType_INTS);
  for(int64_t v : values)
    attr->add_ints(v);
}

// =====================================================================
// Convolution: Conv with zero-padding
// =====================================================================
static string addConvNode(
  onnx::GraphProto* graph,
  int& nameCounter,
  const string& input,
  const ConvLayerDesc& desc,
  const string& prefix
) {
  string weightsName = addInitializer(
    graph, prefix + "/w",
    {desc.outChannels, desc.inChannels, desc.convYSize, desc.convXSize},
    desc.weights
  );

  int padY = desc.convYSize / 2;
  int padX = desc.convXSize / 2;
  string output = uniqueName(nameCounter, prefix + "/out");

  onnx::NodeProto* convNode = addNode(graph, "Conv", {input, weightsName}, output);
  setAttrInts(convNode, "kernel_shape", {desc.convYSize, desc.convXSize});
  setAttrInts(convNode, "pads", {padY, padX, padY, padX});
  setAttrInts(convNode, "dilations", {desc.dilationY, desc.dilationX});
  setAttrInts(convNode, "strides", {1, 1});

  return output;
}

// =====================================================================
// Merged Batch Norm: output = input * mergedScale + mergedBias
// Applied channel-wise, broadcasting over [N, C, H, W]
// =====================================================================
static string addMergedBNNode(
  onnx::GraphProto* graph,
  int& nameCounter,
  const string& input,
  const BatchNormLayerDesc& desc,
  const string& prefix
) {
  int C = desc.numChannels;
  string scaleName = addInitializer(graph, prefix + "/scale", {C, 1, 1}, desc.mergedScale);
  string biasName = addInitializer(graph, prefix + "/bias", {C, 1, 1}, desc.mergedBias);

  string scaled = uniqueName(nameCounter, prefix + "/scaled");
  addNode(graph, "Mul", {input, scaleName}, scaled);

  string output = uniqueName(nameCounter, prefix + "/bn_out");
  addNode(graph, "Add", {scaled, biasName}, output);

  return output;
}

// =====================================================================
// Activation: ReLU, Mish (softplus->tanh->mul), or Identity
// =====================================================================
static string addActivationNode(
  onnx::GraphProto* graph,
  int& nameCounter,
  const string& input,
  int activationType,
  const string& prefix
) {
  if(activationType == ACTIVATION_RELU) {
    string output = uniqueName(nameCounter, prefix + "/relu");
    addNode(graph, "Relu", {input}, output);
    return output;
  } else if(activationType == ACTIVATION_MISH) {
    // Mish = x * tanh(softplus(x)) = x * tanh(ln(1 + exp(x)))
    string sp = uniqueName(nameCounter, prefix + "/softplus");
    addNode(graph, "Softplus", {input}, sp);

    string th = uniqueName(nameCounter, prefix + "/tanh");
    addNode(graph, "Tanh", {sp}, th);

    string output = uniqueName(nameCounter, prefix + "/mish");
    addNode(graph, "Mul", {input, th}, output);
    return output;
  } else {
    // ACTIVATION_IDENTITY -- pass through
    return input;
  }
}

// =====================================================================
// BN + Activation + Mask multiply
// output = activation(input * scale + bias) * mask
// =====================================================================
static string addBNActivationMask(
  onnx::GraphProto* graph,
  int& nameCounter,
  const string& input,
  const BatchNormLayerDesc& bnDesc,
  const ActivationLayerDesc& actDesc,
  const string& mask,
  const string& prefix
) {
  string bn = addMergedBNNode(graph, nameCounter, input, bnDesc, prefix + "/bn");
  string act = addActivationNode(graph, nameCounter, bn, actDesc.activation, prefix + "/act");
  string output = uniqueName(nameCounter, prefix + "/masked");
  addNode(graph, "Mul", {act, mask}, output);
  return output;
}

// =====================================================================
// MatMul: output = input @ W
// W is [inC, outC]
// =====================================================================
static string addMatMulNode(
  onnx::GraphProto* graph,
  int& nameCounter,
  const string& input,
  const MatMulLayerDesc& desc,
  const string& prefix
) {
  string weightsName = addInitializer(graph, prefix + "/w", {desc.inChannels, desc.outChannels}, desc.weights);
  string output = uniqueName(nameCounter, prefix + "/matmul");
  addNode(graph, "MatMul", {input, weightsName}, output);
  return output;
}

// =====================================================================
// Bias addition: output = input + bias
// bias is [C], broadcast over [N, C] or [N, C, H, W]
// =====================================================================
static string addBiasNode(
  onnx::GraphProto* graph,
  int& nameCounter,
  const string& input,
  const MatBiasLayerDesc& desc,
  const string& prefix
) {
  string biasName = addInitializer(graph, prefix + "/b", {desc.numChannels}, desc.weights);
  string output = uniqueName(nameCounter, prefix + "/biased");
  addNode(graph, "Add", {input, biasName}, output);
  return output;
}

// =====================================================================
// KataGPool: Global pooling producing 3 values per channel
// Pool 1: mean = ReduceSum(x * mask, [2,3]) / maskSum
// Pool 2: mean * (sqrt(maskSum) - 14.0) * 0.1
// Pool 3: ReduceMax(x + (mask - 1.0), [2,3])
// Output: [N, 3*C]
// =====================================================================
static string addGlobalPool(
  onnx::GraphProto* graph,
  int& nameCounter,
  const string& input,
  const string& mask,
  const string& maskSumHW,
  const string& prefix
) {
  // x_masked = input * mask  (already masked, but let's be safe)
  string xMasked = uniqueName(nameCounter, prefix + "/gpool_xm");
  addNode(graph, "Mul", {input, mask}, xMasked);

  // sum = ReduceSum(xMasked, axes=[2,3])
  string axesName = addInt64Initializer(graph, uniqueName(nameCounter, prefix + "/axes23"), {2, 3});
  string sumOut = uniqueName(nameCounter, prefix + "/gpool_sum");
  onnx::NodeProto* sumNode = addNode(graph, "ReduceSum", {xMasked, axesName}, sumOut);
  setAttrInt(sumNode, "keepdims", 0);

  // mean = sum / maskSumFlat
  // maskSumHW is [N,1,1,1], we need [N,1] for division
  string maskSumFlat = uniqueName(nameCounter, prefix + "/gpool_msf");
  string reshapeShape = addInt64Initializer(graph, uniqueName(nameCounter, prefix + "/shape_n1"), {0, 1});
  addNode(graph, "Reshape", {maskSumHW, reshapeShape}, maskSumFlat);

  string mean = uniqueName(nameCounter, prefix + "/gpool_mean");
  addNode(graph, "Div", {sumOut, maskSumFlat}, mean);

  // sqrtMaskSum = sqrt(maskSumFlat)
  string sqrtMs = uniqueName(nameCounter, prefix + "/gpool_sqrt");
  addNode(graph, "Sqrt", {maskSumFlat}, sqrtMs);

  // sqrtMs - 14.0
  string const14 = addScalarInitializer(graph, uniqueName(nameCounter, prefix + "/c14"), 14.0f);
  string sqrtMsSub = uniqueName(nameCounter, prefix + "/gpool_sqrtsub");
  addNode(graph, "Sub", {sqrtMs, const14}, sqrtMsSub);

  // * 0.1
  string const01 = addScalarInitializer(graph, uniqueName(nameCounter, prefix + "/c01"), 0.1f);
  string scaledSqrt = uniqueName(nameCounter, prefix + "/gpool_ssm");
  addNode(graph, "Mul", {sqrtMsSub, const01}, scaledSqrt);

  // pool2 = mean * scaledSqrt
  string pool2 = uniqueName(nameCounter, prefix + "/gpool_p2");
  addNode(graph, "Mul", {mean, scaledSqrt}, pool2);

  // Pool3: max over (x + mask - 1)
  string constNeg1 = addScalarInitializer(graph, uniqueName(nameCounter, prefix + "/cn1"), -1.0f);
  string maskBias = uniqueName(nameCounter, prefix + "/gpool_mb");
  addNode(graph, "Add", {mask, constNeg1}, maskBias);

  string xShifted = uniqueName(nameCounter, prefix + "/gpool_xs");
  addNode(graph, "Add", {input, maskBias}, xShifted);

  // ReduceMax over [2,3]
  string axesName2 = addInt64Initializer(graph, uniqueName(nameCounter, prefix + "/axes23b"), {2, 3});
  string pool3 = uniqueName(nameCounter, prefix + "/gpool_max");
  onnx::NodeProto* maxNode = addNode(graph, "ReduceMax", {xShifted, axesName2}, pool3);
  setAttrInt(maxNode, "keepdims", 0);

  // Concat [mean, pool2, pool3] along axis=1
  string output = uniqueName(nameCounter, prefix + "/gpool_out");
  onnx::NodeProto* concatNode = addNode(graph, "Concat", {mean, pool2, pool3}, output);
  setAttrInt(concatNode, "axis", 1);

  return output;
}

// =====================================================================
// KataValueHeadGPool: Different third pool from KataGPool
// Pool 3: mean * ((sqrt(maskSum) - 14.0)^2 * 0.01 - 0.1)
// =====================================================================
static string addValueHeadGPool(
  onnx::GraphProto* graph,
  int& nameCounter,
  const string& input,
  const string& mask,
  const string& maskSumHW,
  const string& prefix
) {
  // x for value head already has activation applied
  // sum = ReduceSum(input * mask, [2,3])
  string xMasked = uniqueName(nameCounter, prefix + "/vgpool_xm");
  addNode(graph, "Mul", {input, mask}, xMasked);

  string axesName = addInt64Initializer(graph, uniqueName(nameCounter, prefix + "/axes23"), {2, 3});
  string sumOut = uniqueName(nameCounter, prefix + "/vgpool_sum");
  onnx::NodeProto* sumNode = addNode(graph, "ReduceSum", {xMasked, axesName}, sumOut);
  setAttrInt(sumNode, "keepdims", 0);

  // mean
  string maskSumFlat = uniqueName(nameCounter, prefix + "/vgpool_msf");
  string reshapeShape = addInt64Initializer(graph, uniqueName(nameCounter, prefix + "/shape_n1"), {0, 1});
  addNode(graph, "Reshape", {maskSumHW, reshapeShape}, maskSumFlat);

  string mean = uniqueName(nameCounter, prefix + "/vgpool_mean");
  addNode(graph, "Div", {sumOut, maskSumFlat}, mean);

  // sqrt(maskSum)
  string sqrtMs = uniqueName(nameCounter, prefix + "/vgpool_sqrt");
  addNode(graph, "Sqrt", {maskSumFlat}, sqrtMs);

  // (sqrt(maskSum) - 14.0)
  string const14 = addScalarInitializer(graph, uniqueName(nameCounter, prefix + "/c14"), 14.0f);
  string sqrtMsSub = uniqueName(nameCounter, prefix + "/vgpool_ss");
  addNode(graph, "Sub", {sqrtMs, const14}, sqrtMsSub);

  // pool2 = mean * (sqrtMsSub) * 0.1
  string const01 = addScalarInitializer(graph, uniqueName(nameCounter, prefix + "/c01"), 0.1f);
  string scaledSqrt = uniqueName(nameCounter, prefix + "/vgpool_ssm");
  addNode(graph, "Mul", {sqrtMsSub, const01}, scaledSqrt);
  string pool2 = uniqueName(nameCounter, prefix + "/vgpool_p2");
  addNode(graph, "Mul", {mean, scaledSqrt}, pool2);

  // pool3 = mean * ((sqrtMsSub)^2 * 0.01 - 0.1)
  string sqrtMsSubSq = uniqueName(nameCounter, prefix + "/vgpool_sq");
  addNode(graph, "Mul", {sqrtMsSub, sqrtMsSub}, sqrtMsSubSq);

  string constP01 = addScalarInitializer(graph, uniqueName(nameCounter, prefix + "/cp01"), 0.01f);
  string sqScaled = uniqueName(nameCounter, prefix + "/vgpool_sqs");
  addNode(graph, "Mul", {sqrtMsSubSq, constP01}, sqScaled);

  string constN01 = addScalarInitializer(graph, uniqueName(nameCounter, prefix + "/cn01"), -0.1f);
  string sqShifted = uniqueName(nameCounter, prefix + "/vgpool_sqsh");
  addNode(graph, "Add", {sqScaled, constN01}, sqShifted);

  string pool3 = uniqueName(nameCounter, prefix + "/vgpool_p3");
  addNode(graph, "Mul", {mean, sqShifted}, pool3);

  // Concat [mean, pool2, pool3] along axis=1
  string output = uniqueName(nameCounter, prefix + "/vgpool_out");
  onnx::NodeProto* concatNode = addNode(graph, "Concat", {mean, pool2, pool3}, output);
  setAttrInt(concatNode, "axis", 1);

  return output;
}

// =====================================================================
// Residual Block: BN->Act->Conv->BN->Act->Conv + skip
// =====================================================================
static string addResidualBlock(
  onnx::GraphProto* graph,
  int& nameCounter,
  const string& input,
  const string& mask,
  const ResidualBlockDesc& desc,
  const string& prefix
) {
  string pre = addBNActivationMask(graph, nameCounter, input, desc.preBN, desc.preActivation, mask, prefix + "/pre");
  string mid = addConvNode(graph, nameCounter, pre, desc.regularConv, prefix + "/conv1");
  string midAct = addBNActivationMask(graph, nameCounter, mid, desc.midBN, desc.midActivation, mask, prefix + "/mid");
  string final_ = addConvNode(graph, nameCounter, midAct, desc.finalConv, prefix + "/conv2");

  // Residual add
  string output = uniqueName(nameCounter, prefix + "/resadd");
  addNode(graph, "Add", {input, final_}, output);
  return output;
}

// =====================================================================
// Global Pooling Residual Block
// =====================================================================
static string addGPoolResidualBlock(
  onnx::GraphProto* graph,
  int& nameCounter,
  const string& input,
  const string& mask,
  const string& maskSumHW,
  const GlobalPoolingResidualBlockDesc& desc,
  const string& prefix
) {
  string pre = addBNActivationMask(graph, nameCounter, input, desc.preBN, desc.preActivation, mask, prefix + "/pre");

  // Regular path
  string regOut = addConvNode(graph, nameCounter, pre, desc.regularConv, prefix + "/reg");

  // Global pooling path
  string gpoolConvOut = addConvNode(graph, nameCounter, pre, desc.gpoolConv, prefix + "/gconv");
  string gpoolBNAct = addBNActivationMask(graph, nameCounter, gpoolConvOut, desc.gpoolBN, desc.gpoolActivation, mask, prefix + "/gbn");
  string gpoolResult = addGlobalPool(graph, nameCounter, gpoolBNAct, mask, maskSumHW, prefix + "/gpool");

  // gpoolToBiasMul: [N, 3*gpoolC] -> [N, regC]
  string gpoolBias = addMatMulNode(graph, nameCounter, gpoolResult, desc.gpoolToBiasMul, prefix + "/g2b");

  // Reshape bias to [N, C, 1, 1] for broadcasting
  string biasShape = addInt64Initializer(graph, uniqueName(nameCounter, prefix + "/shape_nc11"), {0, -1, 1, 1});
  string gpoolBiasReshaped = uniqueName(nameCounter, prefix + "/gbr");
  addNode(graph, "Reshape", {gpoolBias, biasShape}, gpoolBiasReshaped);

  // Add bias to regular conv output
  string regPlusBias = uniqueName(nameCounter, prefix + "/rpb");
  addNode(graph, "Add", {regOut, gpoolBiasReshaped}, regPlusBias);

  // Second half: BN->Act->Conv
  string midAct = addBNActivationMask(graph, nameCounter, regPlusBias, desc.midBN, desc.midActivation, mask, prefix + "/mid");
  string final_ = addConvNode(graph, nameCounter, midAct, desc.finalConv, prefix + "/conv2");

  // Residual add
  string output = uniqueName(nameCounter, prefix + "/resadd");
  addNode(graph, "Add", {input, final_}, output);
  return output;
}

// =====================================================================
// Nested Bottleneck Residual Block
// Pre: BN->Act->Mask->1x1Conv (c_main->c_mid)
// Inner: sequence of ordinary/gpool/nested_bottleneck sub-blocks at c_mid
// Post: BN->Act->Mask->1x1Conv (c_mid->c_main) + residual add
// =====================================================================
static string addNestedBottleneckResidualBlock(
  onnx::GraphProto* graph,
  int& nameCounter,
  const string& input,
  const string& mask,
  const string& maskSumHW,
  const NestedBottleneckResidualBlockDesc& desc,
  const string& prefix
) {
  // Pre: BN -> Act -> Mask -> 1x1 Conv (c_main -> c_mid)
  string pre = addBNActivationMask(graph, nameCounter, input, desc.preBN, desc.preActivation, mask, prefix + "/pre");
  string midOut = addConvNode(graph, nameCounter, pre, desc.preConv, prefix + "/preconv");

  // Inner sub-blocks at c_mid channels
  for(int i = 0; i < desc.numBlocks; i++) {
    int kind = desc.blocks[i].first;
    string sub = prefix + "/sub" + to_string(i);
    if(kind == ORDINARY_BLOCK_KIND) {
      midOut = addResidualBlock(graph, nameCounter, midOut, mask,
        *((const ResidualBlockDesc*)desc.blocks[i].second.get()), sub);
    } else if(kind == GLOBAL_POOLING_BLOCK_KIND) {
      midOut = addGPoolResidualBlock(graph, nameCounter, midOut, mask, maskSumHW,
        *((const GlobalPoolingResidualBlockDesc*)desc.blocks[i].second.get()), sub);
    } else if(kind == NESTED_BOTTLENECK_BLOCK_KIND) {
      midOut = addNestedBottleneckResidualBlock(graph, nameCounter, midOut, mask, maskSumHW,
        *((const NestedBottleneckResidualBlockDesc*)desc.blocks[i].second.get()), sub);
    } else {
      throw StringError("ONNX backend: unknown sub-block kind " + to_string(kind));
    }
  }

  // Post: BN -> Act -> Mask -> 1x1 Conv (c_mid -> c_main)
  string post = addBNActivationMask(graph, nameCounter, midOut, desc.postBN, desc.postActivation, mask, prefix + "/post");
  string postOut = addConvNode(graph, nameCounter, post, desc.postConv, prefix + "/postconv");

  // Residual add: input + postOut
  string output = uniqueName(nameCounter, prefix + "/resadd");
  addNode(graph, "Add", {input, postOut}, output);
  return output;
}

// =====================================================================
// Add ValueInfo for graph input/output
// =====================================================================
static void addGraphInput(
  onnx::GraphProto* graph,
  const string& name,
  const vector<int64_t>& shape
) {
  onnx::ValueInfoProto* input = graph->add_input();
  input->set_name(name);
  onnx::TypeProto* type = input->mutable_type();
  onnx::TypeProto_Tensor* tensorType = type->mutable_tensor_type();
  tensorType->set_elem_type(onnx::TensorProto_DataType_FLOAT);
  onnx::TensorShapeProto* shapeProto = tensorType->mutable_shape();
  for(int64_t d : shape) {
    auto* dim = shapeProto->add_dim();
    if(d < 0)
      dim->set_dim_param("N");
    else
      dim->set_dim_value(d);
  }
}

static void addGraphOutput(
  onnx::GraphProto* graph,
  const string& name,
  const vector<int64_t>& shape
) {
  onnx::ValueInfoProto* output = graph->add_output();
  output->set_name(name);
  onnx::TypeProto* type = output->mutable_type();
  onnx::TypeProto_Tensor* tensorType = type->mutable_tensor_type();
  tensorType->set_elem_type(onnx::TensorProto_DataType_FLOAT);
  onnx::TensorShapeProto* shapeProto = tensorType->mutable_shape();
  for(int64_t d : shape) {
    auto* dim = shapeProto->add_dim();
    if(d < 0)
      dim->set_dim_param("N");
    else
      dim->set_dim_value(d);
  }
}

// =====================================================================
// Main: Build the full ONNX model from ModelDesc
// =====================================================================
string OnnxModelBuilder::buildOnnxModel(const ModelDesc& modelDesc, int nnXLen, int nnYLen) {
  int nameCounter = 0;

  const int modelVersion = modelDesc.modelVersion;
  const int numInputChannels = modelDesc.numInputChannels;
  const int numInputGlobalChannels = modelDesc.numInputGlobalChannels;
  const int numPolicyChannels = modelDesc.numPolicyChannels;
  const int numValueChannels = modelDesc.numValueChannels;
  const int numScoreValueChannels = modelDesc.numScoreValueChannels;
  const int numOwnershipChannels = modelDesc.numOwnershipChannels;

  const TrunkDesc& trunk = modelDesc.trunk;
  const PolicyHeadDesc& policyHead = modelDesc.policyHead;
  const ValueHeadDesc& valueHead = modelDesc.valueHead;

  onnx::ModelProto model;
  model.set_ir_version(8);
  model.set_producer_name("KataGo");
  model.set_domain("ai.katago");

  auto* opset = model.add_opset_import();
  opset->set_domain("");
  opset->set_version(18);

  onnx::GraphProto* graph = model.mutable_graph();
  graph->set_name("katago");

  // ------------------------------------------------------------------
  // Graph Inputs
  // ------------------------------------------------------------------
  addGraphInput(graph, "input_spatial", {-1, numInputChannels, nnYLen, nnXLen});
  addGraphInput(graph, "input_global", {-1, numInputGlobalChannels});
  if(modelDesc.numInputMetaChannels > 0) {
    addGraphInput(graph, "input_meta", {-1, modelDesc.numInputMetaChannels});
  }

  // ------------------------------------------------------------------
  // Derive mask and maskSumHW from input_spatial.
  // Channel 0 of the spatial input is the "on board" indicator: 1.0 for
  // positions on the board, 0.0 for off-board padding.  This is Feature 0
  // set by fillRowV3/V4/V5/V6/V7 in nninputs.cpp and holds across all
  // supported input versions (V3-V7).
  //
  // mask = input_spatial[:, 0:1, :, :]  -> [N, 1, H, W]
  // maskSumHW = ReduceSum(mask, [2, 3], keepdims=true) -> [N, 1, 1, 1]
  // ------------------------------------------------------------------

  // Slice channel 0 to get mask
  string sliceStarts = addInt64Initializer(graph, "mask_starts", {0});
  string sliceEnds = addInt64Initializer(graph, "mask_ends", {1});
  string sliceAxes = addInt64Initializer(graph, "mask_axes", {1});
  string mask = uniqueName(nameCounter, "mask");
  addNode(graph, "Slice", {"input_spatial", sliceStarts, sliceEnds, sliceAxes}, mask);

  // maskSumHW
  string sumAxes = addInt64Initializer(graph, "mask_sum_axes", {2, 3});
  string maskSumHW = uniqueName(nameCounter, "maskSumHW");
  onnx::NodeProto* maskSumNode = addNode(graph, "ReduceSum", {mask, sumAxes}, maskSumHW);
  setAttrInt(maskSumNode, "keepdims", 1);

  // ------------------------------------------------------------------
  // Trunk: Initial conv + matmul bias
  // ------------------------------------------------------------------
  string trunkOut = addConvNode(graph, nameCounter, "input_spatial", trunk.initialConv, "trunk/init_conv");

  // initialMatMul: global features -> [N, trunkNumChannels]
  string globalBias = addMatMulNode(graph, nameCounter, "input_global", trunk.initialMatMul, "trunk/init_matmul");

  // Reshape to [N, C, 1, 1] for broadcasting
  string biasShape = addInt64Initializer(graph, "trunk_bias_shape", {0, -1, 1, 1});
  string globalBiasReshaped = uniqueName(nameCounter, "trunk/gbr");
  addNode(graph, "Reshape", {globalBias, biasShape}, globalBiasReshaped);

  // Add global bias to conv output
  string trunkCombined = uniqueName(nameCounter, "trunk/combined");
  addNode(graph, "Add", {trunkOut, globalBiasReshaped}, trunkCombined);
  trunkOut = trunkCombined;

  // ------------------------------------------------------------------
  // Trunk: Metadata encoder (SGF metadata -> trunk bias)
  // ------------------------------------------------------------------
  if(trunk.metaEncoderVersion > 0) {
    const SGFMetadataEncoderDesc& enc = trunk.sgfMetadataEncoder;
    string metaOut = addMatMulNode(graph, nameCounter, "input_meta", enc.mul1, "trunk/meta_mul1");
    metaOut = addBiasNode(graph, nameCounter, metaOut, enc.bias1, "trunk/meta_b1");
    metaOut = addActivationNode(graph, nameCounter, metaOut, enc.act1.activation, "trunk/meta_a1");
    metaOut = addMatMulNode(graph, nameCounter, metaOut, enc.mul2, "trunk/meta_mul2");
    metaOut = addBiasNode(graph, nameCounter, metaOut, enc.bias2, "trunk/meta_b2");
    metaOut = addActivationNode(graph, nameCounter, metaOut, enc.act2.activation, "trunk/meta_a2");
    metaOut = addMatMulNode(graph, nameCounter, metaOut, enc.mul3, "trunk/meta_mul3");

    // Reshape to [N, C, 1, 1] for spatial broadcasting
    string metaBiasShape = addInt64Initializer(graph, "trunk_meta_bias_shape", {0, -1, 1, 1});
    string metaBiasReshaped = uniqueName(nameCounter, "trunk/mbr");
    addNode(graph, "Reshape", {metaOut, metaBiasShape}, metaBiasReshaped);

    // Add to trunk
    string trunkWithMeta = uniqueName(nameCounter, "trunk/with_meta");
    addNode(graph, "Add", {trunkOut, metaBiasReshaped}, trunkWithMeta);
    trunkOut = trunkWithMeta;
  }

  // ------------------------------------------------------------------
  // Trunk: Residual blocks
  // ------------------------------------------------------------------
  for(int i = 0; i < trunk.numBlocks; i++) {
    int blockKind = trunk.blocks[i].first;
    string blockPrefix = "trunk/block" + to_string(i);

    if(blockKind == ORDINARY_BLOCK_KIND) {
      const ResidualBlockDesc& blockDesc = *((const ResidualBlockDesc*)trunk.blocks[i].second.get());
      trunkOut = addResidualBlock(graph, nameCounter, trunkOut, mask, blockDesc, blockPrefix);
    } else if(blockKind == GLOBAL_POOLING_BLOCK_KIND) {
      const GlobalPoolingResidualBlockDesc& blockDesc = *((const GlobalPoolingResidualBlockDesc*)trunk.blocks[i].second.get());
      trunkOut = addGPoolResidualBlock(graph, nameCounter, trunkOut, mask, maskSumHW, blockDesc, blockPrefix);
    } else if(blockKind == NESTED_BOTTLENECK_BLOCK_KIND) {
      const NestedBottleneckResidualBlockDesc& blockDesc = *((const NestedBottleneckResidualBlockDesc*)trunk.blocks[i].second.get());
      trunkOut = addNestedBottleneckResidualBlock(graph, nameCounter, trunkOut, mask, maskSumHW, blockDesc, blockPrefix);
    } else {
      throw StringError("ONNX backend: unknown block kind " + to_string(blockKind));
    }
  }

  // Trunk tip: BN + activation + mask
  trunkOut = addBNActivationMask(graph, nameCounter, trunkOut, trunk.trunkTipBN, trunk.trunkTipActivation, mask, "trunk/tip");

  // ------------------------------------------------------------------
  // Policy Head
  // ------------------------------------------------------------------

  // p1Conv: spatial path
  string p1Out = addConvNode(graph, nameCounter, trunkOut, policyHead.p1Conv, "policy/p1conv");

  // g1Conv: global pooling path
  string g1Out = addConvNode(graph, nameCounter, trunkOut, policyHead.g1Conv, "policy/g1conv");
  string g1BNAct = addBNActivationMask(graph, nameCounter, g1Out, policyHead.g1BN, policyHead.g1Activation, mask, "policy/g1bn");
  string g1Pool = addGlobalPool(graph, nameCounter, g1BNAct, mask, maskSumHW, "policy/g1pool");

  // gpoolToBiasMul: [N, 3*g1C] -> [N, p1C]
  string policyBias = addMatMulNode(graph, nameCounter, g1Pool, policyHead.gpoolToBiasMul, "policy/g2b");

  // Reshape to [N, C, 1, 1]
  string pBiasShape = addInt64Initializer(graph, uniqueName(nameCounter, "policy/bias_shape"), {0, -1, 1, 1});
  string policyBiasReshaped = uniqueName(nameCounter, "policy/pbr");
  addNode(graph, "Reshape", {policyBias, pBiasShape}, policyBiasReshaped);

  // Add bias to p1
  string p1PlusBias = uniqueName(nameCounter, "policy/p1pb");
  addNode(graph, "Add", {p1Out, policyBiasReshaped}, p1PlusBias);

  // p1BN + activation + mask
  string p1BNAct = addBNActivationMask(graph, nameCounter, p1PlusBias, policyHead.p1BN, policyHead.p1Activation, mask, "policy/p1bn");

  // p2Conv: [N, p1C, H, W] -> [N, policyChannels, H, W]
  string p2Out = addConvNode(graph, nameCounter, p1BNAct, policyHead.p2Conv, "policy/p2conv");

  // Reshape to [N, policyChannels, H*W]
  string pSpatialShape = addInt64Initializer(graph, uniqueName(nameCounter, "policy/spat_shape"), {0, numPolicyChannels, -1});
  string policySpatial = uniqueName(nameCounter, "policy/spatial");
  addNode(graph, "Reshape", {p2Out, pSpatialShape}, policySpatial);

  // Pass move: gpoolToPassMul
  string passOut;
  if(modelVersion >= 15) {
    // gpoolToPassMul -> bias -> activation -> gpoolToPassMul2
    string passMul1 = addMatMulNode(graph, nameCounter, g1Pool, policyHead.gpoolToPassMul, "policy/pass_mul1");
    string passBiased = addBiasNode(graph, nameCounter, passMul1, policyHead.gpoolToPassBias, "policy/pass_bias");
    string passAct = addActivationNode(graph, nameCounter, passBiased, policyHead.passActivation.activation, "policy/pass_act");
    passOut = addMatMulNode(graph, nameCounter, passAct, policyHead.gpoolToPassMul2, "policy/pass_mul2");
  } else {
    passOut = addMatMulNode(graph, nameCounter, g1Pool, policyHead.gpoolToPassMul, "policy/pass_mul");
  }

  // Reshape pass to [N, policyChannels, 1]
  string passShape = addInt64Initializer(graph, uniqueName(nameCounter, "policy/pass_shape"), {0, numPolicyChannels, 1});
  string passReshaped = uniqueName(nameCounter, "policy/pass_r");
  addNode(graph, "Reshape", {passOut, passShape}, passReshaped);

  // Concat spatial + pass -> out_policy [N, policyChannels, H*W+1]
  onnx::NodeProto* policyConcatNode = addNode(graph, "Concat", {policySpatial, passReshaped}, "out_policy");
  setAttrInt(policyConcatNode, "axis", 2);

  // ------------------------------------------------------------------
  // Value Head
  // ------------------------------------------------------------------

  // v1Conv
  string v1Out = addConvNode(graph, nameCounter, trunkOut, valueHead.v1Conv, "value/v1conv");

  // v1BN + activation + mask
  string v1BNAct = addBNActivationMask(graph, nameCounter, v1Out, valueHead.v1BN, valueHead.v1Activation, mask, "value/v1bn");

  // Value head global pooling
  string v1Pool = addValueHeadGPool(graph, nameCounter, v1BNAct, mask, maskSumHW, "value/vpool");

  // v2Mul + v2Bias + v2Activation
  string v2Out = addMatMulNode(graph, nameCounter, v1Pool, valueHead.v2Mul, "value/v2mul");
  string v2Biased = addBiasNode(graph, nameCounter, v2Out, valueHead.v2Bias, "value/v2bias");
  string v2Act = addActivationNode(graph, nameCounter, v2Biased, valueHead.v2Activation.activation, "value/v2act");

  // v3Mul + v3Bias -> out_value [N, 3]
  string v3Out = addMatMulNode(graph, nameCounter, v2Act, valueHead.v3Mul, "value/v3mul");
  string v3Biased = addBiasNode(graph, nameCounter, v3Out, valueHead.v3Bias, "value/v3bias");
  addNode(graph, "Identity", {v3Biased}, "out_value");

  // sv3Mul + sv3Bias -> out_miscvalue [N, numScoreValueChannels]
  string sv3Out = addMatMulNode(graph, nameCounter, v2Act, valueHead.sv3Mul, "value/sv3mul");
  string sv3Biased = addBiasNode(graph, nameCounter, sv3Out, valueHead.sv3Bias, "value/sv3bias");
  addNode(graph, "Identity", {sv3Biased}, "out_miscvalue");

  // vOwnershipConv -> out_ownership [N, 1, H, W]
  string ownOut = addConvNode(graph, nameCounter, v1BNAct, valueHead.vOwnershipConv, "value/own_conv");
  addNode(graph, "Identity", {ownOut}, "out_ownership");

  // ------------------------------------------------------------------
  // Graph Outputs
  // ------------------------------------------------------------------
  int policyResultLen = nnXLen * nnYLen + 1;
  addGraphOutput(graph, "out_policy", {-1, numPolicyChannels, policyResultLen});
  addGraphOutput(graph, "out_value", {-1, numValueChannels});
  addGraphOutput(graph, "out_miscvalue", {-1, numScoreValueChannels});
  addGraphOutput(graph, "out_ownership", {-1, numOwnershipChannels, nnYLen, nnXLen});

  // ------------------------------------------------------------------
  // Serialize to string
  // ------------------------------------------------------------------
  string serialized;
  if(!model.SerializeToString(&serialized))
    throw StringError("ONNX backend: failed to serialize ONNX model to protobuf");

  return serialized;
}
