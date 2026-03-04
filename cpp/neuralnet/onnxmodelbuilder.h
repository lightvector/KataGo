#ifndef NEURALNET_ONNXMODELBUILDER_H_
#define NEURALNET_ONNXMODELBUILDER_H_

#include <string>
#include "../neuralnet/desc.h"

namespace OnnxModelBuilder {
  // Builds a serialized ONNX ModelProto from a KataGo ModelDesc.
  // The model is constructed for a fixed spatial size of nnXLen x nnYLen.
  // Returns the protobuf-serialized bytes, ready for Ort::Session creation.
  std::string buildOnnxModel(const ModelDesc& modelDesc, int nnXLen, int nnYLen);
}

#endif // NEURALNET_ONNXMODELBUILDER_H_
