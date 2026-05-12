#ifndef NEURALNET_ONNXMODELBUILDER_H_
#define NEURALNET_ONNXMODELBUILDER_H_

#include <string>
#include "../neuralnet/desc.h"

namespace OnnxModelBuilder {
  // Builds a serialized ONNX ModelProto from a KataGo ModelDesc.
  // The model is constructed for a fixed spatial size of nnXLen x nnYLen.
  // Returns the protobuf-serialized bytes, ready for Ort::Session creation.
  std::string buildOnnxModel(const ModelDesc& modelDesc, int nnXLen, int nnYLen);

  // Test-only: build a minimal ONNX model wrapping a single layer/block, using
  // the SAME node-construction helpers used by buildOnnxModel. The ONNX backend's
  // testEvaluate* functions use these so that layer tests exercise the actual
  // production graph-construction code paths instead of a parallel reimplementation.
  //
  // Single input "input" / single output "output", float32, batch dim fixed.
  // BatchNorm / ResidualBlock / GlobalPoolingResidualBlock additionally take a
  // float "mask" input of shape [N, 1, H, W] (1 on-board, 0 off-board).
  std::string buildSingleConvModel(
    const ConvLayerDesc& desc, int batchSize, int nnXLen, int nnYLen);
  std::string buildSingleBatchNormModel(
    const BatchNormLayerDesc& desc, int batchSize, int nnXLen, int nnYLen);
  std::string buildSingleResidualBlockModel(
    const ResidualBlockDesc& desc, int batchSize, int nnXLen, int nnYLen);
  std::string buildSingleGlobalPoolingResidualBlockModel(
    const GlobalPoolingResidualBlockDesc& desc, int batchSize, int nnXLen, int nnYLen);
}

#endif // NEURALNET_ONNXMODELBUILDER_H_
