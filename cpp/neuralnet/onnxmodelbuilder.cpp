#include "../neuralnet/onnxmodelbuilder.h"

#include <cmath>

#include "../core/global.h"
#include "../core/test.h"
#include "../neuralnet/activations.h"
#include "../neuralnet/modelversion.h"
#include "../neuralnet/nninputs.h"

#include "onnx.pb.h"

using namespace std;

namespace {

// Builder that accumulates ONNX nodes and initializers into a single GraphProto, handing back
// tensor names as it goes. Tensors are float32; the trunk runs NCHW by
// default, or channel-last NHWC when transformerNHWC is set (see buildBlockStack / buildConv).
struct Builder {
  onnx::GraphProto* graph;
  int nnXLen;
  int nnYLen;
  bool requireExactNNLen;
  bool transformerNHWC;  // run the trunk block stack channel-last (NHWC) vs NCHW

  // Mask-derived feature tensor names (computed once, reused by every gpool). Empty if exact.
  string maskSumName;    // [N,1,1,1] sum of mask over H,W  (only when !exact)
  string maskMeanName;   // [N,1,1,1] mean of mask over the H*W buffer = maskSum/(nnX*nnY)  (only when !exact)
  string maskScaleName;  // [*,1,1,1] (width*0.1 - 1.4)
  string maskQuadName;   // [*,1,1,1] ((width-14)^2*0.01 - 0.1)

  // Additive attention-mask bias [N,1,1,S] (0 on-board, large negative off-board); empty if exact
  // (no masking needed). Computed once and reused by every attention block.
  string maskBiasName;

  // Collected node names for FP32-forcing regions (see OnnxModelBuilder::Result).
  vector<string> trunkTipAndHeadNodeNames;
  vector<string> rmsNormNodeNames;

  // Record all node names added to the graph since `fromNodeIndex` into `dest`. Lets a region
  // builder tag every op it emitted without each helper needing to know the region.
  void recordNodesSince(int fromNodeIndex, vector<string>& dest) {
    for(int i = fromNodeIndex; i < graph->node_size(); i++)
      dest.push_back(graph->node(i).name());
  }

  int uniqueCounter = 0;
  string uniq(const string& base) {
    return base + "/" + Global::intToString(uniqueCounter++);
  }

  // ---- Initializer helpers ----

  // Add a float initializer with the given dims, returning its tensor name.
  string addInitializer(const string& name, const vector<int64_t>& dims, const float* data, size_t count) {
    onnx::TensorProto* init = graph->add_initializer();
    init->set_name(name);
    init->set_data_type(onnx::TensorProto::FLOAT);
    for(int64_t d : dims)
      init->add_dims(d);
    init->mutable_float_data()->Reserve((int)count);
    for(size_t i = 0; i < count; i++)
      init->add_float_data(data[i]);
    return name;
  }
  string addScalarInitializer(const string& name, float value) {
    return addInitializer(name, {1, 1, 1, 1}, &value, 1);
  }
  // Int64 1-D initializer, used for the `axes` input of Reduce* ops (opset 13/18+ take axes as an input).
  string addInt64Initializer(const string& name, const vector<int64_t>& values) {
    onnx::TensorProto* init = graph->add_initializer();
    init->set_name(name);
    init->set_data_type(onnx::TensorProto::INT64);
    init->add_dims((int64_t)values.size());
    for(int64_t v : values)
      init->add_int64_data(v);
    return name;
  }
  // Shared HW-axes initializer ([2,3]) for spatial reductions. Created lazily.
  string hwAxesName;
  string getHWAxes() {
    if(hwAxesName.empty())
      hwAxesName = addInt64Initializer("reduce_axes_hw", {2, 3});
    return hwAxesName;
  }

  // ---- Node helpers ----

  // Generic single-output node. Returns the output tensor name.
  string addNode(
    const string& opType,
    const vector<string>& inputs,
    const string& outName,
    const string& nodeName) {
    onnx::NodeProto* node = graph->add_node();
    node->set_op_type(opType);
    node->set_name(nodeName);
    for(const string& in : inputs)
      node->add_input(in);
    node->add_output(outName);
    return outName;
  }

  onnx::AttributeProto* addAttr(onnx::NodeProto* node, const string& name) {
    onnx::AttributeProto* attr = node->add_attribute();
    attr->set_name(name);
    return attr;
  }
  // Find the last node we added (for attaching attributes).
  onnx::NodeProto* lastNode() {
    return graph->mutable_node(graph->node_size() - 1);
  }

  string elementwise(const string& op, const string& a, const string& b, const string& name) {
    return addNode(op, {a, b}, uniq(name), name);
  }

  // ---- Layer builders ----

  // Conv with explicit symmetric SAME padding. ConvLayerDesc stores weights already in
  // [outC,inC,kY,kX] order (desc.cpp converts from the file's y,x,ic,oc order at load), which is
  // exactly ONNX's layout, so no transpose is needed here. KataGo kernels are odd-sized, so SAME
  // padding is exactly dilation*(k-1)/2 on each side; we emit explicit pads rather than auto_pad
  // (the parser handles explicit pads more robustly).
  //
  // When useNHWC is set, the input/output are channel-last [N,H,W,C]. A 1x1 conv is just a channel
  // projection and is emitted as an NHWC MatMul (no layout change). A spatial conv (k>1) genuinely
  // needs NCHW (ONNX Conv is NCHW-only), so we locally bubble NHWC->NCHW, do the conv, and convert
  // back to NHWC.
  string buildConv(const string& input, const ConvLayerDesc& desc, bool useNHWC) {
    int kY = desc.convYSize, kX = desc.convXSize;
    int inC = desc.inChannels, outC = desc.outChannels;
    testAssert((int)desc.weights.size() == kY * kX * inC * outC);

    if(useNHWC && kY == 1 && kX == 1) {
      // 1x1 conv == channel projection. Weights are [outC,inC,1,1] (== [outC,inC]); ONNX MatMul on
      // [N,H,W,inC] wants [inC,outC], so transpose outC,inC -> inC,outC.
      vector<float> w((size_t)inC * outC);
      for(int oc = 0; oc < outC; oc++)
        for(int ic = 0; ic < inC; ic++)
          w[(size_t)ic * outC + oc] = desc.weights[(size_t)oc * inC + ic];
      string wName = addInitializer(desc.name + ".Wnhwc", {inC, outC}, w.data(), w.size());
      return addNode("MatMul", {input, wName}, uniq(desc.name + "/nhwc"), desc.name + "/nhwc");
    }

    string convIn = useNHWC ? nhwcToNchw(input, desc.name) : input;

    int padY = desc.dilationY * (kY - 1) / 2;
    int padX = desc.dilationX * (kX - 1) / 2;

    string wName = addInitializer(desc.name + ".W", {outC, inC, kY, kX}, desc.weights.data(), desc.weights.size());

    string out = uniq(desc.name);
    addNode("Conv", {convIn, wName}, out, desc.name);
    onnx::NodeProto* node = lastNode();
    { onnx::AttributeProto* a = addAttr(node, "dilations"); a->set_type(onnx::AttributeProto::INTS); a->add_ints(desc.dilationY); a->add_ints(desc.dilationX); }
    { onnx::AttributeProto* a = addAttr(node, "kernel_shape"); a->set_type(onnx::AttributeProto::INTS); a->add_ints(kY); a->add_ints(kX); }
    { onnx::AttributeProto* a = addAttr(node, "pads"); a->set_type(onnx::AttributeProto::INTS); a->add_ints(padY); a->add_ints(padX); a->add_ints(padY); a->add_ints(padX); }
    { onnx::AttributeProto* a = addAttr(node, "strides"); a->set_type(onnx::AttributeProto::INTS); a->add_ints(1); a->add_ints(1); }

    return useNHWC ? nchwToNhwc(out, desc.name) : out;
  }

  // MatMul over channels, applied to an NC11 (or NCHW with the channel being the feature dim) tensor.
  // Implement as a 1x1 conv with weights transposed CK -> KC
  // ONNX Conv weight [outC,inC,1,1].
  string buildMatMul(const string& input, const MatMulLayerDesc& desc) {
    int inC = desc.inChannels, outC = desc.outChannels;
    testAssert((int)desc.weights.size() == inC * outC);
    vector<float> w((size_t)outC * inC);
    for(int ic = 0; ic < inC; ic++)
      for(int oc = 0; oc < outC; oc++)
        w[(size_t)oc * inC + ic] = desc.weights[(size_t)ic * outC + oc];  // CK -> KC
    string wName = addInitializer(desc.name + ".W", {outC, inC, 1, 1}, w.data(), w.size());

    string out = uniq(desc.name);
    addNode("Conv", {input, wName}, out, desc.name);
    onnx::NodeProto* node = lastNode();
    { onnx::AttributeProto* a = addAttr(node, "kernel_shape"); a->set_type(onnx::AttributeProto::INTS); a->add_ints(1); a->add_ints(1); }
    return out;
  }

  // ---- NHWC helpers (transformerNHWC=true path) ----
  // The NHWC transformer path runs transformer-block internals channel-last [N,H,W,C], with one
  // NCHW<->NHWC conversion per run of consecutive transformer blocks (see buildBlockStack). In NHWC,
  // channels are last, so 1x1 convs / channel matmuls / RMSNorm / SwiGLU are natural and the head
  // reshape is [N,H,W,C]->[N,S,heads,dim] (a free reshape, C last). The transformerNHWC=false path
  // instead keeps each block NCHW (buildTransformer{Attention,FFN}Block) and does not use these.
  string nchwToNhwc(const string& input, const string& nameBase) {
    return transpose(input, {0, 2, 3, 1}, nameBase + "/tonhwc");  // [N,C,H,W] -> [N,H,W,C]
  }
  string nhwcToNchw(const string& input, const string& nameBase) {
    return transpose(input, {0, 3, 1, 2}, nameBase + "/tonchw");  // [N,H,W,C] -> [N,C,H,W]
  }

  // Channel projection (1x1 conv / matmul) on an NHWC tensor [N,H,W,inC] -> [N,H,W,outC].
  // MatMulLayerDesc.weights is [inC,outC] (file order) = exactly ONNX MatMul's [inC,outC]: no transpose.
  string projMatMulNhwc(const string& input, const MatMulLayerDesc& desc) {
    int inC = desc.inChannels, outC = desc.outChannels;
    testAssert((int)desc.weights.size() == inC * outC);
    string wName = addInitializer(desc.name + ".Wnhwc", {inC, outC}, desc.weights.data(), desc.weights.size());
    return addNode("MatMul", {input, wName}, uniq(desc.name + "/nhwc"), desc.name + "/nhwc");
  }

  // Per-channel bias add: input + bias broadcast over [1,C,1,1].
  string buildMatBias(const string& input, const MatBiasLayerDesc& desc) {
    int C = desc.numChannels;
    testAssert((int)desc.weights.size() == C);
    string bName = addInitializer(desc.name + ".b", {1, C, 1, 1}, desc.weights.data(), C);
    return addNode("Add", {input, bName}, uniq(desc.name), desc.name);
  }

  // BatchNorm as per-channel affine: out = input * mergedScale + mergedBias, broadcast over the
  // channel dim. NCHW broadcasts over [1,C,1,1]; NHWC ([N,H,W,C]) over [1,1,1,C].
  string buildBatchNorm(const string& input, const BatchNormLayerDesc& desc, bool useNHWC) {
    int C = desc.numChannels;
    testAssert((int)desc.mergedScale.size() == C);
    testAssert((int)desc.mergedBias.size() == C);
    vector<int64_t> dims = useNHWC ? vector<int64_t>{1, 1, 1, C} : vector<int64_t>{1, C, 1, 1};
    string suffix = useNHWC ? ".nhwc" : "";
    string sName = addInitializer(desc.name + ".scale" + suffix, dims, desc.mergedScale.data(), C);
    string bName = addInitializer(desc.name + ".bias" + suffix, dims, desc.mergedBias.data(), C);
    string scaled = addNode("Mul", {input, sName}, uniq(desc.name + "/scale"), desc.name + "/scale");
    return addNode("Add", {scaled, bName}, uniq(desc.name + "/bias"), desc.name + "/bias");
  }

  string buildActivation(const string& input, const ActivationLayerDesc& desc) {
    int act = desc.activation;
    if(act == ACTIVATION_IDENTITY) {
      return addNode("Identity", {input}, uniq(desc.name), desc.name);
    }
    else if(act == ACTIVATION_RELU) {
      return addNode("Relu", {input}, uniq(desc.name), desc.name);
    }
    else if(act == ACTIVATION_SILU) {
      // silu(x) = x * sigmoid(x)
      string sig = addNode("Sigmoid", {input}, uniq(desc.name + "/sigmoid"), desc.name + "/sigmoid");
      return addNode("Mul", {input, sig}, uniq(desc.name), desc.name);
    }
    else if(act == ACTIVATION_MISH || act == ACTIVATION_MISH_SCALE8) {
      // mish(x)         = x * tanh(softplus(x))           = x * tanh(log(1+exp(x)))
      // mish_scale8(x)  = x * tanh(softplus_{beta=8}(x))  = x * tanh(log(1+exp(8x)))
      // The SCALE8 variant is the runtime applyScale8ToReduceActivations() transform that keeps
      // FP16 activations small. ONNX Softplus has no beta, so for SCALE8 we
      // scale the input by 8 before Softplus and do NOT scale the result.
      string spIn = input;
      if(act == ACTIVATION_MISH_SCALE8) {
        string bName = addScalarInitializer(uniq(desc.name + "/beta8"), 8.0f);
        spIn = addNode("Mul", {input, bName}, uniq(desc.name + "/beta8mul"), desc.name + "/beta8mul");
      }
      string sp = addNode("Softplus", {spIn}, uniq(desc.name + "/softplus"), desc.name + "/softplus");
      string th = addNode("Tanh", {sp}, uniq(desc.name + "/tanh"), desc.name + "/tanh");
      return addNode("Mul", {input, th}, uniq(desc.name), desc.name);
    }
    else {
      throw StringError("OnnxModelBuilder: unsupported activation " + Global::intToString(act));
    }
  }

  // Multiply by mask (broadcast [N,1,H,W]). No-op when requireExactNNLen.
  string applyMask(const string& input, const string& maskName, const string& nameBase) {
    if(requireExactNNLen)
      return input;
    return addNode("Mul", {input, maskName}, uniq(nameBase + "/mask"), nameBase + "/mask");
  }
  // Mask multiply that dispatches by layout: NHWC uses the [N,H,W,1] mask, NCHW the given maskName.
  string applyMaskLayout(const string& input, const string& maskName, const string& nameBase, bool useNHWC) {
    return useNHWC ? applyMaskNhwc(input, nameBase) : applyMask(input, maskName, nameBase);
  }

  // Reduce over H,W keeping dims, with axes passed as an input tensor (opset 13/18+ requirement).
  string reduceHW(const string& op, const string& input, const string& outName, const string& nodeName) {
    addNode(op, {input, getHWAxes()}, outName, nodeName);
    onnx::NodeProto* node = lastNode();
    { onnx::AttributeProto* a = addAttr(node, "keepdims"); a->set_type(onnx::AttributeProto::INT); a->set_i(1); }
    return outName;
  }

  // Reduce mean over H,W keeping dims. For !exact, uses sum/maskSum to mean only over on-board cells.
  string reduceMeanHW(const string& input, const string& nameBase) {
    if(requireExactNNLen) {
      return reduceHW("ReduceMean", input, uniq(nameBase + "/gpmean"), nameBase + "/gpmean");
    }
    else {
      string sum = reduceHW("ReduceSum", input, uniq(nameBase + "/gpsum"), nameBase + "/gpsum");
      return addNode("Div", {sum, maskSumName}, uniq(nameBase + "/gpmean"), nameBase + "/gpmean");
    }
  }

  string reduceMaxHW(const string& input, const string& nameBase) {
    return reduceHW("ReduceMax", input, uniq(nameBase + "/gpmax"), nameBase + "/gpmax");
  }

  // Global pooling: concat(mean, mean*maskScale, third), where third is mean*maskQuad for the value
  // head, else masked max. Result is [N, 3C, 1, 1].
  string applyGPool(const string& input, const string& maskName, const string& nameBase, bool isValueHead) {
    string mean = reduceMeanHW(input, nameBase);
    string meanScale = elementwise("Mul", mean, maskScaleName, nameBase + "/gpmeanscale");

    string third;
    if(isValueHead) {
      third = elementwise("Mul", mean, maskQuadName, nameBase + "/gpmeanquad");
    }
    else if(!requireExactNNLen) {
      // off-board cells become 0 after masking; shift by -1 so max ignores them (activations > -1).
      string shifted = addNode("Add", {input, maskMinusOne(maskName, nameBase)}, uniq(nameBase + "/gpmaskadd"), nameBase + "/gpmaskadd");
      third = reduceMaxHW(shifted, nameBase);
    }
    else {
      third = reduceMaxHW(input, nameBase);
    }

    string out = uniq(nameBase + "/gpconcat");
    addNode("Concat", {mean, meanScale, third}, out, nameBase + "/gpconcat");
    onnx::NodeProto* node = lastNode();
    { onnx::AttributeProto* a = addAttr(node, "axis"); a->set_type(onnx::AttributeProto::INT); a->set_i(1); }
    return out;
  }

  // mask - 1, broadcast. On-board (mask=1) -> 0, off-board (mask=0) -> -1, so adding it to a tensor
  // shifts off-board cells down by 1 (matching ModelParser's gpmaskshift) while leaving on-board
  // cells unchanged; the subsequent max then ignores off-board cells (activations are >= 0 > -1).
  string maskMinusOne(const string& maskName, const string& nameBase) {
    string c = addScalarInitializer(uniq(nameBase + "/negone"), -1.0f);
    return addNode("Add", {maskName, c}, uniq(nameBase + "/gpmaskshift"), nameBase + "/gpmaskshift");
  }

  // ---- Residual block builders ----
  // useNHWC: input and output are channel-last [N,H,W,C], and the block's internals run NHWC. The
  // elementwise BN/activation/mask ops and 1x1 convs are layout-free; spatial convs (k>1) bubble to
  // NCHW internally (see buildConv). The residual Add is over the matching layout.

  string buildResidualBlock(const string& input, const ResidualBlockDesc& rb, bool useNHWC) {
    string x = buildBatchNorm(input, rb.preBN, useNHWC);
    x = buildActivation(x, rb.preActivation);
    x = applyMaskLayout(x, "InputMask", rb.name + "/pre", useNHWC);
    x = buildConv(x, rb.regularConv, useNHWC);
    x = buildBatchNorm(x, rb.midBN, useNHWC);
    x = buildActivation(x, rb.midActivation);
    x = applyMaskLayout(x, "InputMask", rb.name + "/mid", useNHWC);
    x = buildConv(x, rb.finalConv, useNHWC);
    return elementwise("Add", input, x, rb.name);
  }

  // Global pooling block. The pooling itself (cross-spatial reductions, concat, matmul-to-bias) is
  // only implemented NCHW, so when useNHWC we bubble the whole interior to NCHW: convert the input
  // once on entry and convert the result back to NHWC on exit. The residual Add stays in NHWC.
  string buildGlobalPoolingResidualBlock(const string& input, const GlobalPoolingResidualBlockDesc& gb, bool useNHWC) {
    string in = useNHWC ? nhwcToNchw(input, gb.name + "/in") : input;
    string pre = buildBatchNorm(in, gb.preBN, false);
    pre = buildActivation(pre, gb.preActivation);
    pre = applyMask(pre, "InputMask", gb.name + "/pre");
    string regular = buildConv(pre, gb.regularConv, false);
    string gp = buildConv(pre, gb.gpoolConv, false);
    gp = buildBatchNorm(gp, gb.gpoolBN, false);
    gp = buildActivation(gp, gb.gpoolActivation);
    gp = applyMask(gp, "InputMask", gb.name + "/gpool");
    string pooled = applyGPool(gp, "InputMask", gb.name, false);
    string bias = buildMatMul(pooled, gb.gpoolToBiasMul);
    string biased = elementwise("Add", regular, bias, gb.name + "/gpbias");
    string mid = buildBatchNorm(biased, gb.midBN, false);
    mid = buildActivation(mid, gb.midActivation);
    mid = applyMask(mid, "InputMask", gb.name + "/mid");
    mid = buildConv(mid, gb.finalConv, false);
    string outNchw = elementwise("Add", in, mid, gb.name);
    return useNHWC ? nchwToNhwc(outNchw, gb.name + "/out") : outNchw;
  }

  string buildNestedBottleneckResidualBlock(const string& input, const NestedBottleneckResidualBlockDesc& nb, bool useNHWC) {
    string pre = buildBatchNorm(input, nb.preBN, useNHWC);
    pre = buildActivation(pre, nb.preActivation);
    pre = applyMaskLayout(pre, "InputMask", nb.name + "/pre", useNHWC);
    pre = buildConv(pre, nb.preConv, useNHWC);
    string stack = buildBlockStack(pre, nb.blocks, useNHWC);
    string post = buildBatchNorm(stack, nb.postBN, useNHWC);
    post = buildActivation(post, nb.postActivation);
    post = applyMaskLayout(post, "InputMask", nb.name + "/post", useNHWC);
    post = buildConv(post, nb.postConv, useNHWC);
    return elementwise("Add", input, post, nb.name);
  }

  // ---- Transformer building blocks ----

  // Reshape helper: emit a Reshape with a constant int64 shape initializer. A dim of 0 means "copy
  // from input" and -1 means "infer", per ONNX Reshape semantics.
  string reshape(const string& input, const vector<int64_t>& shape, const string& nameBase) {
    string shapeName = addInt64Initializer(uniq(nameBase + "/shape"), shape);
    return addNode("Reshape", {input, shapeName}, uniq(nameBase + "/reshape"), nameBase + "/reshape");
  }
  string transpose(const string& input, const vector<int64_t>& perm, const string& nameBase) {
    string out = uniq(nameBase + "/transpose");
    addNode("Transpose", {input}, out, nameBase + "/transpose");
    onnx::NodeProto* node = lastNode();
    onnx::AttributeProto* a = addAttr(node, "perm");
    a->set_type(onnx::AttributeProto::INTS);
    for(int64_t p : perm) a->add_ints(p);
    return out;
  }

  // Transformer-internal RMSNorm: per spatial position, normalize across channels (axis=1),
  // scale by per-channel weight, then mask. Input/output NCHW [N,C,H,W].
  string transformerRMSNorm(const string& input, const TransformerRMSNormDesc& desc, const string& maskName) {
    int C = desc.numChannels;
    testAssert((int)desc.weight.size() == C);
    int rmsStart = graph->node_size();
    // meanSq over channels (axis 1), keepdims -> [N,1,H,W]
    string sq = addNode("Mul", {input, input}, uniq(desc.name + "/sq"), desc.name + "/sq");
    string meanSq = addNode("ReduceMean", {sq, addInt64Initializer(uniq(desc.name + "/axC"), {1})},
                            uniq(desc.name + "/meansq"), desc.name + "/meansq");
    { onnx::NodeProto* n = lastNode(); onnx::AttributeProto* a = addAttr(n, "keepdims"); a->set_type(onnx::AttributeProto::INT); a->set_i(1); }
    string epsName = addScalarInitializer(uniq(desc.name + "/eps"), desc.epsilon);
    string denom = addNode("Add", {meanSq, epsName}, uniq(desc.name + "/denom"), desc.name + "/denom");
    string rms = addNode("Sqrt", {denom}, uniq(desc.name + "/rmsstd"), desc.name + "/rmsstd");
    // Only the square->reduce->sqrt above is precision-sensitive (it sums over many elements); the
    // division/scale/mask below are elementwise and FP16-safe, so we record only the former for
    // FP32-forcing and leave the output in the graph's working precision.
    recordNodesSince(rmsStart, rmsNormNodeNames);
    string normed = addNode("Div", {input, rms}, uniq(desc.name + "/normed"), desc.name + "/normed");  // broadcast [N,C,H,W]/[N,1,H,W]
    string wName = addInitializer(desc.name + ".weight", {1, C, 1, 1}, desc.weight.data(), C);
    string scaled = addNode("Mul", {normed, wName}, uniq(desc.name + "/scaled"), desc.name + "/scaled");
    return applyMask(scaled, maskName, desc.name);
  }

  // NHWC mask [N,H,W,1], built lazily from InputMask [N,1,H,W]. Empty if requireExactNNLen.
  string nhwcMaskName;
  string getNhwcMask() {
    if(requireExactNNLen) return "";
    if(nhwcMaskName.empty())
      nhwcMaskName = transpose("InputMask", {0, 2, 3, 1}, "InputMask/nhwc");  // [N,1,H,W] -> [N,H,W,1]
    return nhwcMaskName;
  }
  string applyMaskNhwc(const string& input, const string& nameBase) {
    if(requireExactNNLen) return input;
    return addNode("Mul", {input, getNhwcMask()}, uniq(nameBase + "/mask"), nameBase + "/mask");
  }

  // Transformer RMSNorm in NHWC [N,H,W,C]: normalize over channels (last axis), scale, mask.
  string transformerRMSNormNhwc(const string& input, const TransformerRMSNormDesc& desc) {
    int C = desc.numChannels;
    testAssert((int)desc.weight.size() == C);
    int rmsStart = graph->node_size();
    string sq = addNode("Mul", {input, input}, uniq(desc.name + "/sq"), desc.name + "/sq");
    string meanSq = addNode("ReduceMean", {sq, addInt64Initializer(uniq(desc.name + "/axC"), {3})},  // C is axis 3 of [N,H,W,C]
                            uniq(desc.name + "/meansq"), desc.name + "/meansq");
    { onnx::NodeProto* n = lastNode(); onnx::AttributeProto* a = addAttr(n, "keepdims"); a->set_type(onnx::AttributeProto::INT); a->set_i(1); }
    string epsName = addScalarInitializer(uniq(desc.name + "/eps"), desc.epsilon);
    string denom = addNode("Add", {meanSq, epsName}, uniq(desc.name + "/denom"), desc.name + "/denom");
    string rms = addNode("Sqrt", {denom}, uniq(desc.name + "/rmsstd"), desc.name + "/rmsstd");
    recordNodesSince(rmsStart, rmsNormNodeNames);  // pin square->reduce->sqrt to FP32
    string normed = addNode("Div", {input, rms}, uniq(desc.name + "/normed"), desc.name + "/normed");  // [N,H,W,C]/[N,H,W,1]
    string wName = addInitializer(desc.name + ".weightnhwc", {1, 1, 1, C}, desc.weight.data(), C);
    string scaled = addNode("Mul", {normed, wName}, uniq(desc.name + "/scaled"), desc.name + "/scaled");
    return applyMaskNhwc(scaled, desc.name);
  }

  // Channel-projection matmul implemented as 1x1 conv (same as the convnet matmul, CK->KC).
  string projConv(const string& input, const MatMulLayerDesc& desc) {
    return buildMatMul(input, desc);
  }

  // Full trunk-tip RMSNorm: gamma/beta, optional spatial mode, optional activation, then mask.
  // Input/output NCHW [N,C,H,W]. Mirrors RMSNormLayer in eigenbackend.cpp.
  string buildTrunkTipRMSNorm(const string& input, const RMSNormLayerDesc& desc, int activation, const string& maskName) {
    int C = desc.numChannels;
    testAssert((int)desc.gamma.size() == C);
    testAssert((int)desc.beta.size() == C);
    int rmsStart = graph->node_size();

    bool spatial = desc.spatial;
    string meanSq;
    if(!spatial) {
      // Per-position mean of squares over channels (axis 1) -> [N,1,H,W]
      string sq = addNode("Mul", {input, input}, uniq(desc.name + "/sq"), desc.name + "/sq");
      meanSq = addNode("ReduceMean", {sq, addInt64Initializer(uniq(desc.name + "/axC"), {1})},
                       uniq(desc.name + "/meansq"), desc.name + "/meansq");
      { onnx::NodeProto* n = lastNode(); onnx::AttributeProto* a = addAttr(n, "keepdims"); a->set_type(onnx::AttributeProto::INT); a->set_i(1); }
    }
    else {
      // RMS over channels AND on-board spatial positions, per batch element -> [N,1,1,1].
      // The mean-of-squares is a reduction over C*H*W (tens of thousands of elements). A naive
      // sum-of-squares can exceed the FP16 max (65504) on a full 19x19 board (e.g. ~138000), overflowing
      // to inf and corrupting the whole trunk tip. Although these nodes are tagged FP32 (see the
      // FP32-forcing in trtbackend.cpp) and the tag is applied successfully, TensorRT disregards it for
      // the reduction: it fuses square+reduce into a Myelin kernel that accumulates in FP16 internally
      // and overflows regardless of the per-layer precision/output constraint. So we always reduce with
      // ReduceMean (the running mean stays O(1) and never overflows, independent of whether the pin is
      // honored) rather than ReduceSum-then-divide.
      //   exact:  meanSq = ReduceMean_{C,H,W}(x^2)                          (whole buffer is on-board)
      //   masked: meanSq = ReduceMean_{C,H,W}(x^2 * mask) / maskMean        (recover the on-board mean;
      //           maskMean = on-board fraction of the buffer, so dividing by it cancels the off-board
      //           zeros that ReduceMean averaged over). maskMean is itself FP32-pinned.
      string sq = addNode("Mul", {input, input}, uniq(desc.name + "/sq"), desc.name + "/sq");
      string reduceInput = sq;
      if(!requireExactNNLen)
        reduceInput = addNode("Mul", {sq, maskName}, uniq(desc.name + "/sqmask"), desc.name + "/sqmask");
      string meanFull = addNode("ReduceMean", {reduceInput, addInt64Initializer(uniq(desc.name + "/axCHW"), {1, 2, 3})},
                                uniq(desc.name + "/meansqfull"), desc.name + "/meansqfull");
      { onnx::NodeProto* n = lastNode(); onnx::AttributeProto* a = addAttr(n, "keepdims"); a->set_type(onnx::AttributeProto::INT); a->set_i(1); }
      if(requireExactNNLen)
        meanSq = meanFull;
      else
        meanSq = addNode("Div", {meanFull, maskMeanName}, uniq(desc.name + "/meansq"), desc.name + "/meansq");
    }

    string epsName = addScalarInitializer(uniq(desc.name + "/eps"), desc.epsilon);
    string denom = addNode("Add", {meanSq, epsName}, uniq(desc.name + "/denom"), desc.name + "/denom");
    string rms = addNode("Sqrt", {denom}, uniq(desc.name + "/rmsstd"), desc.name + "/rmsstd");
    // Only the square->reduce->sqrt above is precision-sensitive; record just that for FP32-forcing
    // and leave the elementwise normalize/affine/activation/mask below in the working precision.
    recordNodesSince(rmsStart, rmsNormNodeNames);
    string normed = addNode("Div", {input, rms}, uniq(desc.name + "/normed"), desc.name + "/normed");
    string gName = addInitializer(desc.name + ".gamma", {1, C, 1, 1}, desc.gamma.data(), C);
    string bName = addInitializer(desc.name + ".beta", {1, C, 1, 1}, desc.beta.data(), C);
    string scaled = addNode("Mul", {normed, gName}, uniq(desc.name + "/scaled"), desc.name + "/scaled");
    string affine = addNode("Add", {scaled, bName}, uniq(desc.name + "/affine"), desc.name + "/affine");
    string activated = affine;
    if(activation == ACTIVATION_SILU) {
      string sig = addNode("Sigmoid", {affine}, uniq(desc.name + "/silu/sig"), desc.name + "/silu/sig");
      activated = addNode("Mul", {affine, sig}, uniq(desc.name + "/silu"), desc.name + "/silu");
    }
    else if(activation != ACTIVATION_IDENTITY) {
      throw StringError("OnnxModelBuilder: trunk-tip RMSNorm activation must be identity or silu");
    }
    return applyMask(activated, maskName, desc.name);
  }

  // SwiGLU FFN block, NHWC-core: input/output both NHWC [N,H,W,C].
  string buildTransformerFFNBlockNhwc(const string& inNhwc, const TransformerFFNDesc& ffn) {
    if(!ffn.useSwiGLU)
      throw StringError("OnnxModelBuilder: non-SwiGLU transformer FFN not supported");
    string xn = transformerRMSNormNhwc(inNhwc, ffn.preLN);        // [N,H,W,C]
    string a = projMatMulNhwc(xn, ffn.linear1);     // [N,H,W,ffnC]
    string g = projMatMulNhwc(xn, ffn.linearGate);  // [N,H,W,ffnC]
    string sig = addNode("Sigmoid", {a}, uniq(ffn.name + "/silu/sig"), ffn.name + "/silu/sig");
    string silu = addNode("Mul", {a, sig}, uniq(ffn.name + "/silu"), ffn.name + "/silu");
    string gated = addNode("Mul", {silu, g}, uniq(ffn.name + "/swiglu"), ffn.name + "/swiglu");
    string outNhwc = projMatMulNhwc(gated, ffn.linear2);  // [N,H,W,C]
    string maskedNhwc = applyMaskNhwc(outNhwc, ffn.name + "/out");
    return addNode("Add", {inNhwc, maskedNhwc}, uniq(ffn.name), ffn.name);  // residual, NHWC
  }

  // SwiGLU FFN block, genuine NCHW (per-block): the trtTransformerNHWC=false path. All ops NCHW.
  string buildTransformerFFNBlock(const string& input, const TransformerFFNDesc& ffn, const string& maskName) {
    if(!ffn.useSwiGLU)
      throw StringError("OnnxModelBuilder: non-SwiGLU transformer FFN not supported");
    string xn = transformerRMSNorm(input, ffn.preLN, maskName);
    string a = projConv(xn, ffn.linear1);     // [N, ffnC, H, W]
    string g = projConv(xn, ffn.linearGate);  // [N, ffnC, H, W]
    // SwiGLU: SiLU(a) * g
    string sig = addNode("Sigmoid", {a}, uniq(ffn.name + "/silu/sig"), ffn.name + "/silu/sig");
    string silu = addNode("Mul", {a, sig}, uniq(ffn.name + "/silu"), ffn.name + "/silu");
    string gated = addNode("Mul", {silu, g}, uniq(ffn.name + "/swiglu"), ffn.name + "/swiglu");
    string out = projConv(gated, ffn.linear2); // [N, C, H, W]
    string masked = applyMask(out, maskName, ffn.name + "/out");
    return addNode("Add", {input, masked}, uniq(ffn.name), ffn.name);
  }

  // Apply RoPE to a tensor shaped [N, heads, hd, S] using baked per-(head,pair,position) cos/sin.
  // KataGo uses interleaved pairs: channel 2p and 2p+1 form pair p. The rotation is
  //   y[2p]   = x[2p]*cos - x[2p+1]*sin
  //   y[2p+1] = x[2p]*sin + x[2p+1]*cos
  // Written as y = x*cosFull + swap(x)*sinSigned, where swap(x) exchanges the two channels of each
  // pair (swap(x)[2p]=x[2p+1], swap(x)[2p+1]=x[2p]) and sinSigned folds in the sign
  // (sinSigned[2p]=-sin, sinSigned[2p+1]=+sin). swap is one Gather along the channel (hd) axis with a
  // fixed index permutation; this avoids a Split/Neg/Concat + reshape form which performs worse
  string applyRope(
    const string& input,  // [N, heads, hd, S]
    int heads, int hd, int seqLen, int ropeNumPairs,
    const vector<float>& cosTable, const vector<float>& sinTable,
    bool learnableRope, int numKVHeads, int numHeads,
    const string& nameBase) {
    // cosFull[h,c,s] / sinSigned[h,c,s] as [1,heads,hd,S] constants (batch-broadcast).
    // Table layout: learnable -> (kvHead, pair, S); fixed -> (pair, S). For Q heads under learnable
    // GQA, head h maps to kv head h*numKVHeads/heads
    vector<float> cosFull((size_t)heads * hd * seqLen);
    vector<float> sinSigned((size_t)heads * hd * seqLen);
    for(int h = 0; h < heads; h++) {
      for(int p = 0; p < ropeNumPairs; p++) {
        for(int s = 0; s < seqLen; s++) {
          int tableIdx;
          if(learnableRope) {
            int kvh = h * numKVHeads / heads;
            tableIdx = (kvh * ropeNumPairs + p) * seqLen + s;
          } else {
            tableIdx = p * seqLen + s;
          }
          float c = cosTable[tableIdx];
          float sn = sinTable[tableIdx];
          size_t i0 = ((size_t)h * hd + (2 * p + 0)) * seqLen + s;
          size_t i1 = ((size_t)h * hd + (2 * p + 1)) * seqLen + s;
          cosFull[i0] = c;   cosFull[i1] = c;
          sinSigned[i0] = -sn;  sinSigned[i1] = sn;  // sign folded in (swap is unsigned)
        }
      }
    }
    string cosName = addInitializer(nameBase + "/ropecos", {1, heads, hd, seqLen}, cosFull.data(), cosFull.size());
    string sinName = addInitializer(nameBase + "/ropesinsigned", {1, heads, hd, seqLen}, sinSigned.data(), sinSigned.size());

    // swap(x): Gather along the hd axis (2) with the fixed pair-swap permutation [1,0,3,2,5,4,...].
    vector<int64_t> swapIdx(hd);
    for(int p = 0; p < ropeNumPairs; p++) { swapIdx[2 * p] = 2 * p + 1; swapIdx[2 * p + 1] = 2 * p; }
    string idxName = addInt64Initializer(nameBase + "/ropeswapidx", swapIdx);
    string xSwap = uniq(nameBase + "/rope_swap");
    addNode("Gather", {input, idxName}, xSwap, nameBase + "/rope_swap");
    { onnx::NodeProto* n = lastNode(); onnx::AttributeProto* a = addAttr(n, "axis"); a->set_type(onnx::AttributeProto::INT); a->set_i(2); }

    string term1 = addNode("Mul", {input, cosName}, uniq(nameBase + "/rope_t1"), nameBase + "/rope_t1");
    string term2 = addNode("Mul", {xSwap, sinName}, uniq(nameBase + "/rope_t2"), nameBase + "/rope_t2");
    return addNode("Add", {term1, term2}, uniq(nameBase + "/rope_out"), nameBase + "/rope_out");
  }

  // Grouped-query attention: repeat each of the nKV key/value heads nRep=nH/nKV times along the head
  // axis so K/V go from [N,nKV,...] to [N,nH,...], matching every query head to its kv head
  // (kvh = h / nRep). Implemented as a Gather on the head axis with index vector [0,0,..,1,1,..]. The
  // repeat is done AFTER RoPE (which is applied at the native nKV heads). No-op when nH == nKV.
  string expandKVHeads(const string& input, int nKV, int nH, int headAxis, const string& nameBase) {
    if(nH == nKV)
      return input;
    int nRep = nH / nKV;
    vector<int64_t> idx((size_t)nH);
    for(int h = 0; h < nH; h++)
      idx[h] = h / nRep;
    string idxName = addInt64Initializer(uniq(nameBase + "/gqaidx"), idx);
    string out = uniq(nameBase + "/gqa");
    addNode("Gather", {input, idxName}, out, nameBase + "/gqa");
    { onnx::NodeProto* n = lastNode(); onnx::AttributeProto* a = addAttr(n, "axis"); a->set_type(onnx::AttributeProto::INT); a->set_i(headAxis); }
    return out;
  }

  // Transformer self-attention block, NHWC-core: input/output both NHWC [N,H,W,C]. Entire block runs
  // channel-last; head ops use [N,heads,S,dim]. maskBiasName is the [N,1,1,S] additive softmax bias.
  string buildTransformerAttentionBlockNhwc(
    const string& inNhwc, const TransformerAttentionDesc& att,
    int nnXLen, int nnYLen, const string& maskBiasName) {
    int seqLen = nnXLen * nnYLen;
    int nH = att.numHeads, nKV = att.numKVHeads, hd = att.qHeadDim, vhd = att.vHeadDim;

    string xnNhwc = transformerRMSNormNhwc(inNhwc, att.preLN);            // [N,H,W,C]
    string qNhwc = projMatMulNhwc(xnNhwc, att.qProj);  // [N,H,W,nH*hd]
    string kNhwc = projMatMulNhwc(xnNhwc, att.kProj);  // [N,H,W,nKV*hd]
    string vNhwc = projMatMulNhwc(xnNhwc, att.vProj);  // [N,H,W,nKV*vhd]

    // [N,H,W,heads*dim] -> [N,S,heads,dim] (free reshape, dim last) -> [N,heads,S,dim]
    string qh = transpose(reshape(qNhwc, {-1, seqLen, nH, hd}, att.name + "/q"), {0, 2, 1, 3}, att.name + "/qh");
    string kh = transpose(reshape(kNhwc, {-1, seqLen, nKV, hd}, att.name + "/k"), {0, 2, 1, 3}, att.name + "/kh");
    string vh = transpose(reshape(vNhwc, {-1, seqLen, nKV, vhd}, att.name + "/v"), {0, 2, 1, 3}, att.name + "/vh");

    if(att.useRope) {
      int ropeNumPairs = hd / 2;
      vector<float> cosTable, sinTable;
      att.computeRopeCosSin(nnXLen, nnYLen, seqLen, cosTable, sinTable);
      qh = applyRopeHeadLast(qh, nH, hd, seqLen, ropeNumPairs, cosTable, sinTable, att.learnableRope, nKV, nH, att.name + "/qrope");
      kh = applyRopeHeadLast(kh, nKV, hd, seqLen, ropeNumPairs, cosTable, sinTable, att.learnableRope, nKV, nH, att.name + "/krope");
    }

    // GQA: expand K/V from nKV to nH heads (head axis is 1 of [N,heads,S,dim]) before the matmuls.
    kh = expandKVHeads(kh, nKV, nH, 1, att.name + "/k");
    vh = expandKVHeads(vh, nKV, nH, 1, att.name + "/v");

    // scores[N,heads,S,S] = qh[N,heads,S,hd] @ kh^T[N,heads,hd,S]
    string khT = transpose(kh, {0, 1, 3, 2}, att.name + "/khT");  // [N,heads,hd,S]
    string scores = addNode("MatMul", {qh, khT}, uniq(att.name + "/scores"), att.name + "/scores");
    string scaleName = addScalarInitializer(uniq(att.name + "/scale"), 1.0f / sqrtf((float)hd));
    string scoresScaled = addNode("Mul", {scores, scaleName}, uniq(att.name + "/scoresscaled"), att.name + "/scoresscaled");
    string scoresMasked = scoresScaled;
    if(!maskBiasName.empty())
      scoresMasked = addNode("Add", {scoresScaled, maskBiasName}, uniq(att.name + "/scoresmasked"), att.name + "/scoresmasked");
    string probs = uniq(att.name + "/probs");
    addNode("Softmax", {scoresMasked}, probs, att.name + "/softmax");
    { onnx::NodeProto* n = lastNode(); onnx::AttributeProto* a = addAttr(n, "axis"); a->set_type(onnx::AttributeProto::INT); a->set_i(3); }

    // attn[N,heads,S,vhd] = probs[N,heads,S,S] @ vh[N,heads,S,vhd]
    string attnSV = addNode("MatMul", {probs, vh}, uniq(att.name + "/sv"), att.name + "/sv");
    // [N,heads,S,vhd] -> [N,S,heads,vhd] -> NHWC [N,H,W,nH*vhd]
    string attnSeqHeads = transpose(attnSV, {0, 2, 1, 3}, att.name + "/svT");  // [N,S,heads,vhd]
    string attnNhwc = reshape(attnSeqHeads, {-1, nnYLen, nnXLen, nH * vhd}, att.name + "/attnnhwc");  // [N,H,W,nH*vhd]

    string outNhwc = projMatMulNhwc(attnNhwc, att.outProj);  // [N,H,W,C]
    string maskedNhwc = applyMaskNhwc(outNhwc, att.name + "/out");
    return addNode("Add", {inNhwc, maskedNhwc}, uniq(att.name), att.name);  // residual, NHWC
  }

  // Transformer self-attention block, genuine NCHW (per-block): the trtTransformerNHWC=false path.
  // Head ops use [N,heads,dim,S]; RoPE via applyRope (hd-axis-2). All NCHW. maskBiasName is the
  // [N,1,1,S] additive softmax bias.
  string buildTransformerAttentionBlock(
    const string& input, const TransformerAttentionDesc& att,
    int nnXLen, int nnYLen, const string& maskName, const string& maskBiasName) {
    int seqLen = nnXLen * nnYLen;
    int nH = att.numHeads, nKV = att.numKVHeads, hd = att.qHeadDim, vhd = att.vHeadDim;

    string xn = transformerRMSNorm(input, att.preLN, maskName);
    string q = projConv(xn, att.qProj);  // [N, nH*hd, H, W]
    string k = projConv(xn, att.kProj);  // [N, nKV*hd, H, W]
    string v = projConv(xn, att.vProj);  // [N, nKV*vhd, H, W]

    // Reshape to [N, heads, dim, S]  (channels are head-major: head h occupies [h*dim, (h+1)*dim))
    string qh = reshape(q, {0, nH, hd, seqLen}, att.name + "/q");
    string kh = reshape(k, {0, nKV, hd, seqLen}, att.name + "/k");
    string vh = reshape(v, {0, nKV, vhd, seqLen}, att.name + "/v");

    if(att.useRope) {
      int ropeNumPairs = hd / 2;
      vector<float> cosTable, sinTable;
      att.computeRopeCosSin(nnXLen, nnYLen, seqLen, cosTable, sinTable);
      qh = applyRope(qh, nH, hd, seqLen, ropeNumPairs, cosTable, sinTable, att.learnableRope, nKV, nH, att.name + "/qrope");
      kh = applyRope(kh, nKV, hd, seqLen, ropeNumPairs, cosTable, sinTable, att.learnableRope, nKV, nH, att.name + "/krope");
    }

    // GQA: expand K/V from nKV to nH heads (head axis is 1 of [N,heads,dim,S]) before the matmuls.
    kh = expandKVHeads(kh, nKV, nH, 1, att.name + "/k");
    vh = expandKVHeads(vh, nKV, nH, 1, att.name + "/v");

    // scores = (Q^T K) * scale: qh/kh are [N,heads,dim,S]; transpose qh to [N,heads,S,dim], matmul kh.
    string qT = transpose(qh, {0, 1, 3, 2}, att.name + "/qT");  // [N,heads,S,dim]
    string scores = addNode("MatMul", {qT, kh}, uniq(att.name + "/scores"), att.name + "/scores");  // [N,heads,S,S]
    string scaleName = addScalarInitializer(uniq(att.name + "/scale"), 1.0f / sqrtf((float)hd));
    string scoresScaled = addNode("Mul", {scores, scaleName}, uniq(att.name + "/scoresscaled"), att.name + "/scoresscaled");
    string scoresMasked = scoresScaled;
    if(!maskBiasName.empty())
      scoresMasked = addNode("Add", {scoresScaled, maskBiasName}, uniq(att.name + "/scoresmasked"), att.name + "/scoresmasked");
    string probs = uniq(att.name + "/probs");
    addNode("Softmax", {scoresMasked}, probs, att.name + "/softmax");
    { onnx::NodeProto* n = lastNode(); onnx::AttributeProto* a = addAttr(n, "axis"); a->set_type(onnx::AttributeProto::INT); a->set_i(3); }

    // out = probs @ V^T_per_head: vh [N,heads,vhd,S] -> [N,heads,S,vhd]; probs[N,h,S,S] @ that.
    string vT = transpose(vh, {0, 1, 3, 2}, att.name + "/vT");  // [N,heads,S,vhd]
    string attnSV = addNode("MatMul", {probs, vT}, uniq(att.name + "/sv"), att.name + "/sv");  // [N,heads,S,vhd]
    string attnT = transpose(attnSV, {0, 1, 3, 2}, att.name + "/svT");  // [N,heads,vhd,S]
    string attnNCHW = reshape(attnT, {0, nH * vhd, nnYLen, nnXLen}, att.name + "/attnnchw");

    string out = projConv(attnNCHW, att.outProj);  // [N, C, H, W]
    string masked = applyMask(out, maskName, att.name + "/out");
    return addNode("Add", {input, masked}, uniq(att.name), att.name);
  }

  // RoPE on [N,heads,S,hd] (hd last). y = x*cosFull + swap(x)*sinSigned; swap = Gather on hd axis (3).
  string applyRopeHeadLast(
    const string& input, int heads, int hd, int seqLen, int ropeNumPairs,
    const vector<float>& cosTable, const vector<float>& sinTable,
    bool learnableRope, int numKVHeads, int /*numHeads*/, const string& nameBase) {
    vector<float> cosFull((size_t)heads * seqLen * hd);
    vector<float> sinSigned((size_t)heads * seqLen * hd);
    for(int h = 0; h < heads; h++)
      for(int s = 0; s < seqLen; s++)
        for(int p = 0; p < ropeNumPairs; p++) {
          int tableIdx = learnableRope ? ((h * numKVHeads / heads) * ropeNumPairs + p) * seqLen + s
                                       : p * seqLen + s;
          float c = cosTable[tableIdx], sn = sinTable[tableIdx];
          size_t i0 = ((size_t)h * seqLen + s) * hd + (2 * p + 0);
          size_t i1 = ((size_t)h * seqLen + s) * hd + (2 * p + 1);
          cosFull[i0] = c;     cosFull[i1] = c;
          sinSigned[i0] = -sn; sinSigned[i1] = sn;
        }
    string cosName = addInitializer(nameBase + "/ropecos", {1, heads, seqLen, hd}, cosFull.data(), cosFull.size());
    string sinName = addInitializer(nameBase + "/ropesinsigned", {1, heads, seqLen, hd}, sinSigned.data(), sinSigned.size());
    vector<int64_t> swapIdx(hd);
    for(int p = 0; p < ropeNumPairs; p++) { swapIdx[2 * p] = 2 * p + 1; swapIdx[2 * p + 1] = 2 * p; }
    string idxName = addInt64Initializer(nameBase + "/ropeswapidx", swapIdx);
    string xSwap = uniq(nameBase + "/rope_swap");
    addNode("Gather", {input, idxName}, xSwap, nameBase + "/rope_swap");
    { onnx::NodeProto* n = lastNode(); onnx::AttributeProto* a = addAttr(n, "axis"); a->set_type(onnx::AttributeProto::INT); a->set_i(3); }
    string term1 = addNode("Mul", {input, cosName}, uniq(nameBase + "/rope_t1"), nameBase + "/rope_t1");
    string term2 = addNode("Mul", {xSwap, sinName}, uniq(nameBase + "/rope_t2"), nameBase + "/rope_t2");
    return addNode("Add", {term1, term2}, uniq(nameBase + "/rope_out"), nameBase + "/rope_out");
  }

  // Build a stack of residual blocks (used by the trunk and recursively by nested bottleneck blocks).
  // useNHWC: the input and output of this stack are channel-last [N,H,W,C]
  // The caller is responsible for any NCHW<->NHWC conversion before passing in. The stack itself never mixes
  // layouts internally - a nested bottleneck block-stack inherits its parent's layout directly.
  string buildBlockStack(const string& input, const std::vector<std::pair<int, unique_ptr_void>>& blocks, bool useNHWC) {
    string cur = input;
    for(size_t i = 0; i < blocks.size(); i++) {
      int kind = blocks[i].first;
      void* bp = blocks[i].second.get();
      if(kind == TRANSFORMER_ATTENTION_BLOCK_KIND) {
        const TransformerAttentionDesc& att = *static_cast<const TransformerAttentionDesc*>(bp);
        cur = useNHWC ? buildTransformerAttentionBlockNhwc(cur, att, nnXLen, nnYLen, maskBiasName)
                      : buildTransformerAttentionBlock(cur, att, nnXLen, nnYLen, "InputMask", maskBiasName);
      }
      else if(kind == TRANSFORMER_FFN_BLOCK_KIND) {
        const TransformerFFNDesc& ffn = *static_cast<const TransformerFFNDesc*>(bp);
        cur = useNHWC ? buildTransformerFFNBlockNhwc(cur, ffn)
                      : buildTransformerFFNBlock(cur, ffn, "InputMask");
      }
      else if(kind == ORDINARY_BLOCK_KIND) {
        cur = buildResidualBlock(cur, *static_cast<const ResidualBlockDesc*>(bp), useNHWC);
      }
      else if(kind == GLOBAL_POOLING_BLOCK_KIND) {
        cur = buildGlobalPoolingResidualBlock(cur, *static_cast<const GlobalPoolingResidualBlockDesc*>(bp), useNHWC);
      }
      else if(kind == NESTED_BOTTLENECK_BLOCK_KIND) {
        cur = buildNestedBottleneckResidualBlock(cur, *static_cast<const NestedBottleneckResidualBlockDesc*>(bp), useNHWC);
      }
      else {
        throw StringError("OnnxModelBuilder: unexpected block kind " + Global::intToString(kind));
      }
    }
    return cur;
  }
};

}  // namespace

namespace OnnxModelBuilder {

Result build(
  const ModelDesc& desc,
  int nnXLen,
  int nnYLen,
  bool requireExactNNLen,
  bool transformerNHWC,
  Logger* logger
) {
  if(desc.metaEncoderVersion > 0)
    throw StringError("OnnxModelBuilder: SGF metadata encoder not yet supported");

  if(logger != NULL)
    logger->write("Building internal onnx model, requireExactNNLen=" + Global::boolToString(requireExactNNLen) + " transformerNHWC=" + Global::boolToString(transformerNHWC));

  int numInputChannels = desc.numInputChannels;
  int numInputGlobalChannels = desc.numInputGlobalChannels;

  onnx::ModelProto model;
  model.set_ir_version(onnx::IR_VERSION_2023_5_5);
  model.set_producer_name("katago");
  {
    onnx::OperatorSetIdProto* opset = model.add_opset_import();
    opset->set_domain("");
    opset->set_version(20);
  }
  auto addMeta = [&](const string& k, const string& v) {
    onnx::StringStringEntryProto* m = model.add_metadata_props();
    m->set_key(k);
    m->set_value(v);
  };
  addMeta("name", desc.name);
  addMeta("modelVersion", Global::intToString(desc.modelVersion));

  onnx::GraphProto* graph = model.mutable_graph();
  graph->set_name(desc.name.empty() ? "katago" : desc.name);

  Builder b;
  b.graph = graph;
  b.nnXLen = nnXLen;
  b.nnYLen = nnYLen;
  b.requireExactNNLen = requireExactNNLen;
  b.transformerNHWC = transformerNHWC;

  // ---- Inputs (NCHW, dynamic batch) ----
  auto addInput = [&](const string& name, int channels) {
    onnx::ValueInfoProto* vi = graph->add_input();
    vi->set_name(name);
    onnx::TypeProto::Tensor* t = vi->mutable_type()->mutable_tensor_type();
    t->set_elem_type(onnx::TensorProto::FLOAT);
    onnx::TensorShapeProto* shape = t->mutable_shape();
    shape->add_dim()->set_dim_param("batch");
    shape->add_dim()->set_dim_value(channels);
    shape->add_dim()->set_dim_value(nnYLen);
    shape->add_dim()->set_dim_value(nnXLen);
  };
  auto addInputNC11 = [&](const string& name, int channels) {
    onnx::ValueInfoProto* vi = graph->add_input();
    vi->set_name(name);
    onnx::TypeProto::Tensor* t = vi->mutable_type()->mutable_tensor_type();
    t->set_elem_type(onnx::TensorProto::FLOAT);
    onnx::TensorShapeProto* shape = t->mutable_shape();
    shape->add_dim()->set_dim_param("batch");
    shape->add_dim()->set_dim_value(channels);
    shape->add_dim()->set_dim_value(1);
    shape->add_dim()->set_dim_value(1);
  };
  addInput("InputMask", 1);
  addInput("InputSpatial", numInputChannels);
  addInputNC11("InputGlobal", numInputGlobalChannels);

  // ---- Mask-derived features ----
  if(!requireExactNNLen) {
    // maskSum = sum over H,W of mask -> [N,1,1,1]
    b.maskSumName = b.reduceHW("ReduceSum", "InputMask", b.uniq("InputMask/sum"), "InputMask/sum");
    // maskMean = mean of mask over the H*W buffer = on-board fraction of the buffer. Computed directly
    // as a ReduceMean over the mask (rather than maskSum/(H*W)). Used to recover a masked mean-of-squares
    // from a full-buffer ReduceMean in the spatial RMSNorm without ever materializing a large sum. It is
    // consumed only inside the FP32-pinned RMSNorm region, so pin it to FP32 too.
    b.maskMeanName = b.reduceHW("ReduceMean", "InputMask", b.uniq("InputMask/mean"), "InputMask/mean");
    b.rmsNormNodeNames.push_back("InputMask/mean");
    string width = b.addNode("Sqrt", {b.maskSumName}, b.uniq("InputMask/width"), "InputMask/width");
    // scale = width*0.1 - 1.4
    {
      string s = b.addScalarInitializer(b.uniq("InputMask/scaleW"), 0.1f);
      string sh = b.addScalarInitializer(b.uniq("InputMask/scaleB"), -1.4f);
      string m = b.addNode("Mul", {width, s}, b.uniq("InputMask/scalemul"), "InputMask/scalemul");
      b.maskScaleName = b.addNode("Add", {m, sh}, b.uniq("InputMask/scale"), "InputMask/scale");
    }
    // quad = (width-14)^2 * 0.01 - 0.1
    {
      string c14 = b.addScalarInitializer(b.uniq("InputMask/c14"), -14.0f);
      string shifted = b.addNode("Add", {width, c14}, b.uniq("InputMask/centershift"), "InputMask/centershift");
      string sq = b.addNode("Mul", {shifted, shifted}, b.uniq("InputMask/centersquare"), "InputMask/centersquare");
      string s = b.addScalarInitializer(b.uniq("InputMask/quadW"), 0.01f);
      string sh = b.addScalarInitializer(b.uniq("InputMask/quadB"), -0.1f);
      string m = b.addNode("Mul", {sq, s}, b.uniq("InputMask/quadmul"), "InputMask/quadmul");
      b.maskQuadName = b.addNode("Add", {m, sh}, b.uniq("InputMask/quad"), "InputMask/quad");
    }
  }
  else {
    float width = sqrtf((float)nnXLen * (float)nnYLen);
    b.maskScaleName = b.addScalarInitializer("InputMask/scale", width * 0.1f - 1.4f);
    b.maskQuadName = b.addScalarInitializer("InputMask/quad", (width - 14.0f) * (width - 14.0f) * 0.01f - 0.1f);
  }

  // ---- Attention mask bias [N,1,1,S]: 0 on-board, large-negative off-board, for attention softmax.
  // Only needed when masking (variable board) and there are transformer attention blocks.
  if(!requireExactNNLen && desc.trunk.hasAnyTransformerBlocks()) {
    // (mask - 1) * BIG : on-board (1) -> 0, off-board (0) -> -BIG
    string one = b.addScalarInitializer(b.uniq("InputMask/biasone"), -1.0f);
    string mShift = b.addNode("Add", {"InputMask", one}, b.uniq("InputMask/biasshift"), "InputMask/biasshift");
    string big = b.addScalarInitializer(b.uniq("InputMask/biasbig"), 1.0e9f);
    string biasNCHW = b.addNode("Mul", {mShift, big}, b.uniq("InputMask/biasnchw"), "InputMask/biasnchw");  // [N,1,H,W]
    // reshape [N,1,H,W] -> [N,1,1,S] so it broadcasts over the key axis of [N,heads,S(query),S(key)]
    b.maskBiasName = b.reshape(biasNCHW, {0, 1, 1, nnXLen * nnYLen}, "InputMask/bias");
  }

  // ================= Trunk =================
  const TrunkDesc& trunk = desc.trunk;
  string initialConv = b.buildConv("InputSpatial", trunk.initialConv, false);
  string initialMatMul = b.buildMatMul("InputGlobal", trunk.initialMatMul);
  string cur = b.elementwise("Add", initialConv, initialMatMul, trunk.name + "/initbias");

  // When transformerNHWC, run the entire trunk block stack channel-last: one NCHW->NHWC conversion
  // here and one NHWC->NCHW conversion before the trunk tip. Every block (convnet/gpool/nbt/
  // transformer, including nested block-stacks) runs NHWC; only spatial convs and gpool interiors
  // bubble locally to NCHW. Models without transformer blocks ignore transformerNHWC (it would just
  // add a pair of trunk-wide transposes for no benefit), so we only enable it when there actually
  // are transformer blocks.
  // It's possible TensorRT could itself get clever and optimize away these kinds of layout differences
  // on its own, but implementing them directly in the onnx output gives an interesting knob to play with
  // in case it doesn't do a perfect job of that.
  bool trunkNHWC = b.transformerNHWC && trunk.hasAnyTransformerBlocks();
  if(trunkNHWC)
    cur = b.nchwToNhwc(cur, trunk.name + "/trunk");
  cur = b.buildBlockStack(cur, trunk.blocks, trunkNHWC);
  if(trunkNHWC)
    cur = b.nhwcToNchw(cur, trunk.name + "/trunk");

  // Everything from the trunk tip through the heads is a candidate for FP32-forcing (it's where the
  // numerically-sensitive normalizations and the small final projections live, and it's cheap
  // relative to the trunk). Snapshot the node index so we can record this whole region.
  int trunkTipAndHeadStart = b.graph->node_size();

  // Trunk tip: either standard BatchNorm+activation, or RMSNorm (transformer/modern models).
  string trunkTip;
  if(trunk.trunkNormKind == TRUNK_NORM_KIND_STANDARD) {
    trunkTip = b.buildBatchNorm(cur, trunk.trunkTipBN, false);
    trunkTip = b.buildActivation(trunkTip, trunk.trunkTipActivation);
    trunkTip = b.applyMask(trunkTip, "InputMask", trunk.name + "/tip");
  }
  else {
    // RMSNorm trunk tip; activation is folded into the RMSNorm (and masking applied inside).
    trunkTip = b.buildTrunkTipRMSNorm(cur, trunk.trunkTipRMSNorm, trunk.trunkTipActivation.activation, "InputMask");
  }

  // ================= Policy head =================
  {
    const PolicyHeadDesc& ph = desc.policyHead;
    string p1 = b.buildConv(trunkTip, ph.p1Conv, false);
    string g1 = b.buildConv(trunkTip, ph.g1Conv, false);
    g1 = b.buildBatchNorm(g1, ph.g1BN, false);
    g1 = b.buildActivation(g1, ph.g1Activation);
    g1 = b.applyMask(g1, "InputMask", ph.name + "/g1");
    string gpool = b.applyGPool(g1, "InputMask", ph.name + "/g", false);
    string gbias = b.buildMatMul(gpool, ph.gpoolToBiasMul);
    string p1b = b.elementwise("Add", p1, gbias, ph.name + "/gpbias");
    p1b = b.buildBatchNorm(p1b, ph.p1BN, false);
    p1b = b.buildActivation(p1b, ph.p1Activation);
    p1b = b.applyMask(p1b, "InputMask", ph.name + "/p1");
    string p2 = b.buildConv(p1b, ph.p2Conv, false);

    string pass;
    if(desc.modelVersion >= 15) {
      string pm = b.buildMatMul(gpool, ph.gpoolToPassMul);
      pm = b.buildMatBias(pm, ph.gpoolToPassBias);
      pm = b.buildActivation(pm, ph.passActivation);
      pass = b.buildMatMul(pm, ph.gpoolToPassMul2);
    }
    else {
      pass = b.buildMatMul(gpool, ph.gpoolToPassMul);
    }

    // Outputs
    auto markOutput = [&](const string& tensorName, const string& outName, int channels, bool spatial) {
      // Rename via Identity so the graph output has the exact expected name.
      onnx::NodeProto* node = graph->add_node();
      node->set_op_type("Identity");
      node->set_name(outName + "/out");
      node->add_input(tensorName);
      node->add_output(outName);
      onnx::ValueInfoProto* vi = graph->add_output();
      vi->set_name(outName);
      onnx::TypeProto::Tensor* t = vi->mutable_type()->mutable_tensor_type();
      t->set_elem_type(onnx::TensorProto::FLOAT);
      onnx::TensorShapeProto* shape = t->mutable_shape();
      shape->add_dim()->set_dim_param("batch");
      shape->add_dim()->set_dim_value(channels);
      shape->add_dim()->set_dim_value(spatial ? nnYLen : 1);
      shape->add_dim()->set_dim_value(spatial ? nnXLen : 1);
    };
    markOutput(pass, "OutputPolicyPass", ph.policyOutChannels, false);
    markOutput(p2, "OutputPolicy", ph.policyOutChannels, true);
  }

  // ================= Value head =================
  {
    const ValueHeadDesc& vh = desc.valueHead;
    string v1 = b.buildConv(trunkTip, vh.v1Conv, false);
    v1 = b.buildBatchNorm(v1, vh.v1BN, false);
    v1 = b.buildActivation(v1, vh.v1Activation);
    string v1Masked = b.applyMask(v1, "InputMask", vh.name + "/v1");
    string gpool = b.applyGPool(v1Masked, "InputMask", vh.name + "/v", true);
    string v2 = b.buildMatMul(gpool, vh.v2Mul);
    v2 = b.buildMatBias(v2, vh.v2Bias);
    v2 = b.buildActivation(v2, vh.v2Activation);
    string v3 = b.buildMatMul(v2, vh.v3Mul);
    v3 = b.buildMatBias(v3, vh.v3Bias);
    string sv3 = b.buildMatMul(v2, vh.sv3Mul);
    sv3 = b.buildMatBias(sv3, vh.sv3Bias);
    string ownership = b.buildConv(v1Masked, vh.vOwnershipConv, false);

    auto markOutput = [&](const string& tensorName, const string& outName, int channels, bool spatial) {
      onnx::NodeProto* node = graph->add_node();
      node->set_op_type("Identity");
      node->set_name(outName + "/out");
      node->add_input(tensorName);
      node->add_output(outName);
      onnx::ValueInfoProto* vi = graph->add_output();
      vi->set_name(outName);
      onnx::TypeProto::Tensor* t = vi->mutable_type()->mutable_tensor_type();
      t->set_elem_type(onnx::TensorProto::FLOAT);
      onnx::TensorShapeProto* shape = t->mutable_shape();
      shape->add_dim()->set_dim_param("batch");
      shape->add_dim()->set_dim_value(channels);
      shape->add_dim()->set_dim_value(spatial ? nnYLen : 1);
      shape->add_dim()->set_dim_value(spatial ? nnXLen : 1);
    };
    markOutput(v3, "OutputValue", desc.numValueChannels, false);
    markOutput(sv3, "OutputScoreValue", desc.numScoreValueChannels, false);
    markOutput(ownership, "OutputOwnership", desc.numOwnershipChannels, true);
  }

  b.recordNodesSince(trunkTipAndHeadStart, b.trunkTipAndHeadNodeNames);

  // DEBUG (kept commented out): expose every internal node output as an extra FP32 graph output so the
  // backend can dump per-layer activations for FP16-vs-FP32 *numerical* divergence analysis. This is
  // complementary to the trtDumpDebugPlanToDir engine dump (which shows fusion structure and boundary
  // dtypes but NOT the internal accumulation precision of a Myelin-fused kernel): this dumps the actual
  // values, which is how the trunk-tip spatial-RMSNorm FP16 sum-of-squares overflow was found. Pairs
  // with ComputeHandle::maybeDumpDebugActivations and the KATAGO_TEST_ONLY_POS/PER_POS hooks in
  // testnnevalcanary.cpp. Uncomment all three (and #include <set>/<cstdlib>) to re-enable.
  // if(std::getenv("KATAGO_TRT_DEBUG_ALL_OUTPUTS") != nullptr) {
  //   std::set<string> alreadyOutput;
  //   for(int i = 0; i < graph->output_size(); i++)
  //     alreadyOutput.insert(graph->output(i).name());
  //   int nNodes = graph->node_size();
  //   for(int i = 0; i < nNodes; i++) {
  //     const onnx::NodeProto& node = graph->node(i);
  //     if(node.output_size() < 1 || node.name().empty())
  //       continue;
  //     const string& tensorName = node.output(0);
  //     if(alreadyOutput.count(tensorName))
  //       continue;
  //     // Identity rename so the graph output has a stable, node-derived name.
  //     string outName = "DBG__" + node.name();
  //     onnx::NodeProto* idn = graph->add_node();
  //     idn->set_op_type("Identity");
  //     idn->set_name(outName + "/id");
  //     idn->add_input(tensorName);
  //     idn->add_output(outName);
  //     onnx::ValueInfoProto* vi = graph->add_output();
  //     vi->set_name(outName);
  //     vi->mutable_type()->mutable_tensor_type()->set_elem_type(onnx::TensorProto::FLOAT);
  //   }
  //   if(logger != NULL)
  //     logger->write("OnnxModelBuilder: DEBUG exposed all internal node outputs as graph outputs");
  // }

  OnnxModelBuilder::Result result;
  if(!model.SerializeToString(&result.serializedModel))
    throw StringError("OnnxModelBuilder: failed to serialize ModelProto");
  result.trunkTipAndHeadNodeNames = std::move(b.trunkTipAndHeadNodeNames);
  result.rmsNormNodeNames = std::move(b.rmsNormNodeNames);
  return result;
}

}  // namespace OnnxModelBuilder
