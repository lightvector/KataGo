#ifndef NEURALNET_ONNXMODELBUILDER_H_
#define NEURALNET_ONNXMODELBUILDER_H_

#include <string>
#include <vector>

#include "../neuralnet/desc.h"
#include "../core/logger.h"

// Emits an ONNX ModelProto (serialized to bytes) describing a KataGo model, given its ModelDesc
// and the runtime board dimensions. The serialized bytes are intended to be handed to TensorRT's
// nvonnxparser, which builds the engine.
//
// The emitted graph reproduces the same tensor semantics as the hand-assembled ModelParser in
// trtbackend.cpp: NCHW float32 tensors, inputs named InputMask / InputSpatial / InputGlobal /
// InputMeta, and RAW-head outputs named OutputPolicyPass / OutputPolicy / OutputValue /
// OutputScoreValue / OutputOwnership. Post-processing is intentionally left to the C++ getOutput
// code, exactly as for the .bin.gz ModelParser path, so both paths share one decode path.
//
// Weights are baked into the ModelProto as initializers, so the serialized bytes are fully
// self-contained.
namespace OnnxModelBuilder {
  struct Result {
    std::string serializedModel;  // the serialized ONNX ModelProto

    // ONNX node names (== the resulting TensorRT layer names) for regions that may need to be forced
    // to FP32 for numerical safety. The TensorRT backend matches engine layers against these and
    // calls setPrecision(kFLOAT) on them. Used to avoid FP16 precision loss without depending on
    // TensorRT not fusing a numerically-equivalent FP16 path back in.
    std::vector<std::string> trunkTipAndHeadNodeNames;  // trunk-tip norm + policy head + value head
    std::vector<std::string> rmsNormNodeNames;          // every RMSNorm (transformer + trunk-tip) op
  };

  // Build a serialized ONNX ModelProto for the given model.
  // Inputs are always declared in the fixed order InputSpatial, InputGlobal, InputMask. This order
  // is required by the OpenVINO execution provider under ONNX Runtime (its name->index map is built
  // from declaration order while the runtime feeds the EP input ports in the order it expects; the
  // default InputMask-first order misroutes the mask tensor at runtime). See onnxmodelbuilder.cpp
  // for the full rationale.
  //
  // emitFusedMishOp: if true, KataGo's Mish activation (x * tanh(softplus(x))) is emitted as a
  // single native ONNX `Mish` node (opset 18+) instead of the decomposed Softplus+Tanh+Mul used
  // by default. The decomposition is what TensorRT's nvonnxparser and the live ONNX/WinML runtime
  // backends have actually been tested against, so this defaults to false and only exportonnx (the
  // offline FP32 export consumed by external quantizers, e.g. for AMD VitisAI/NPU) opts in: AMD's
  // Ryzen AI DPU compiler recognizes the fused `Mish` op for XINT8 quantization but not the
  // decomposed Softplus primitive, so without this the exported graph's Conv/activation layers
  // fail to fuse into NPU-executable subgraphs and silently fall back to CPU end to end.
  Result build(
    const ModelDesc& desc,
    int nnXLen,
    int nnYLen,
    bool requireExactNNLen,
    bool transformerNHWC,
    Logger* logger,
    bool emitFusedMishOp
  );
}

#endif  // NEURALNET_ONNXMODELBUILDER_H_
