#ifndef NEURALNET_ONNXMODELBUILDER_H_
#define NEURALNET_ONNXMODELBUILDER_H_

#include <string>
#include <vector>

#include "../neuralnet/desc.h"
#include "../core/logger.h"

// Emits an ONNX ModelProto (serialized to bytes) describing a KataGo model, given its ModelDesc
// and the runtime board dimensions. The serialized bytes are intended to be handed to TensorRT's
// nvonnxparser or to ONNX Runtime, which build the engine/session.
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

  // Build settings that get baked into an emitted graph and cannot be changed afterwards. A graph is
  // only valid for the board size and masking mode it was emitted with, so build() records these in
  // the model's metadata and load() checks them again.
  struct BuildParams {
    int nnXLen;
    int nnYLen;
    // If true, the graph assumes every position fills the whole nnXLen x nnYLen buffer and skips all
    // masking. Such a graph produces wrong results for smaller boards.
    bool requireExactNNLen;
    // Run the trunk block stack channel-last. Only meaningful for models with transformer blocks;
    // build() normalizes it to false for any other model, including in the recorded metadata.
    bool transformerNHWC;
    // Whether ModelDesc::applyScale8ToReduceActivations() was applied to the weights before emitting.
    // The compensation for it lives in postProcessParams.outputScaleMultiplier, which is recorded
    // already transformed, so this is only reported, never re-applied.
    bool scale8Applied;

    BuildParams();
  };

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
  // Inputs are always declared in the order InputSpatial, InputGlobal, InputMeta (when the
  // model has a metadata encoder), InputMask. ONNX Runtime's OpenVINO execution provider
  // requires this exact order and misroutes tensors at runtime otherwise. See the input
  // declarations in onnxmodelbuilder.cpp for the full rationale.
  Result build(
    const ModelDesc& desc,
    const BuildParams& buildParams,
    Logger* logger
  );

  // ---- Reading a .onnx model file ----

  // A KataGo model read from a .onnx file rather than from a .bin.gz. The graph carries the weights
  // but none of KataGo's scalar model parameters, such as the model version, the channel counts and
  // the score post-processing multipliers; those travel in the ModelProto's metadata_props under
  // "katago." keys, which build() writes and load() reads. Files without that block are rejected,
  // since nothing else can supply the parameters.
  //
  // docs/ONNX_Model_Files.md documents the block and the graph IO contract for third parties: a
  // conforming .onnx from any source loads, not only dumponnx output.
  struct LoadResult {
    std::string serializedModel;  // raw bytes as read from the file, to be handed to TRT / ORT as-is
    BuildParams buildParams;      // the settings the graph was emitted with
    int metadataVersion;
    std::string sourceSha256;     // sha256 of the .bin.gz this was emitted from, empty if unrecorded

    // FP32-pinning node name lists, same meaning as in Result. Recorded in the metadata because the
    // TensorRT backend needs them for models it did not emit itself in this process.
    std::vector<std::string> trunkTipAndHeadNodeNames;
    std::vector<std::string> rmsNormNodeNames;

    // True if the graph declares an input that no node consumes anywhere but last. ONNX Runtime's
    // OpenVINO execution provider mis-binds every input after such a one; see build()'s input
    // declarations.
    bool danglingInputNotDeclaredLast;

    LoadResult();
  };

  // True if fileName names a raw ONNX model file (.onnx or .onnx.gz) rather than a KataGo .bin.gz.
  bool isOnnxFileName(const std::string& fileName);

  // Read a .onnx (or .onnx.gz) file, validate it, and fill descBuf with the model parameters
  // recorded in its metadata. Verifies the sha256 of the file contents against expectedSha256 if
  // that is nonempty, and sets descBuf.sha256 to the file's own sha256.
  // Throws StringError with an explanatory message if the file is not parseable as ONNX, carries no
  // KataGo metadata block, states a metadata version this build cannot read, or has a graph whose
  // input/output signature disagrees with the metadata.
  //
  // descBuf is filled with scalar parameters only: trunk/policyHead/valueHead stay empty and
  // descBuf.archSummary carries the recorded architecture summary in their place.
  LoadResult load(
    const std::string& fileName,
    const std::string& expectedSha256,
    ModelDesc& descBuf,
    Logger* logger
  );

  // Check that a graph loaded by load() can be run at the board size and masking mode the backend is
  // about to use, throwing StringError explaining the mismatch if not. Both are baked into the graph
  // when it is built, so a mismatch cannot be fixed at runtime.
  void checkRuntimeParams(
    const LoadResult& loadResult,
    const std::string& modelFileName,
    int nnXLen,
    int nnYLen,
    bool requireExactNNLen
  );
}

#endif  // NEURALNET_ONNXMODELBUILDER_H_
