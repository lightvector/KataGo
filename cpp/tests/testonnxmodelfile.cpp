#include "../tests/tests.h"

// Unit tests for reading .onnx model files (OnnxModelBuilder::load and friends), which only exist
// in builds with a backend that consumes ONNX graphs. See docs/ONNX_Model_Files.md for the file
// format these tests exercise.

#if defined(USE_TENSORRT_BACKEND) || defined(USE_ONNX_BACKEND)

#include <fstream>
#include <functional>

#include "../core/fileutils.h"
#include "../core/makedir.h"
#include "../neuralnet/desc.h"
#include "../neuralnet/modelversion.h"
#include "../neuralnet/nninputs.h"
#include "../neuralnet/onnxmodelbuilder.h"

#include "onnx.pb.h"

using namespace std;

namespace {

// Metadata for a plausible model of the given version, matching the requirements in
// docs/ONNX_Model_Files.md: the postProcess and metaEncoder keys are included exactly at the
// versions where they are required.
map<string, string> makeMeta(int modelVersion, int numPolicyChannels) {
  map<string, string> meta;
  meta["katago.metadataVersion"] = "1";
  meta["katago.name"] = "onnxfiletest";
  meta["katago.modelVersion"] = Global::intToString(modelVersion);
  meta["katago.numInputChannels"] = Global::intToString(NNModelVersion::getNumSpatialFeatures(modelVersion));
  meta["katago.numInputGlobalChannels"] = Global::intToString(NNModelVersion::getNumGlobalFeatures(modelVersion));
  meta["katago.numInputMetaChannels"] = "0";
  meta["katago.numPolicyChannels"] = Global::intToString(numPolicyChannels);
  meta["katago.numValueChannels"] = "3";
  int numScoreValueChannels = modelVersion >= 9 ? 6 : modelVersion >= 8 ? 4 : modelVersion >= 4 ? 2 : 1;
  meta["katago.numScoreValueChannels"] = Global::intToString(numScoreValueChannels);
  meta["katago.numOwnershipChannels"] = "1";
  if(modelVersion >= 15) {
    meta["katago.metaEncoderVersion"] = "0";
    meta["katago.preferPassAliveUnderSuicideRules"] = "false";
    meta["katago.preferExcludeTerritoryAdjacentToAtari"] = "false";
  }
  if(modelVersion >= 13) {
    meta["katago.postProcess.tdScoreMultiplier"] = "20";
    meta["katago.postProcess.scoreMeanMultiplier"] = "20";
    meta["katago.postProcess.scoreStdevMultiplier"] = "20";
    meta["katago.postProcess.leadMultiplier"] = "20";
    meta["katago.postProcess.varianceTimeMultiplier"] = "40";
    meta["katago.postProcess.shorttermValueErrorMultiplier"] = "0.25";
    meta["katago.postProcess.shorttermScoreErrorMultiplier"] = "150";
  }
  meta["katago.build.nnXLen"] = "19";
  meta["katago.build.nnYLen"] = "19";
  meta["katago.build.requireExactNNLen"] = "false";
  return meta;
}

void setTensor(onnx::ValueInfoProto* vi, const string& name, int channels, int h, int w) {
  vi->set_name(name);
  onnx::TypeProto::Tensor* t = vi->mutable_type()->mutable_tensor_type();
  t->set_elem_type(onnx::TensorProto::FLOAT);
  onnx::TensorShapeProto* shape = t->mutable_shape();
  shape->add_dim()->set_dim_param("batch");
  shape->add_dim()->set_dim_value(channels);
  shape->add_dim()->set_dim_value(h);
  shape->add_dim()->set_dim_value(w);
}

// A structurally-conforming model for the given metadata: the right graph inputs and outputs, with
// every input except the final InputMask consumed by a node, as the input-declaration-order rule requires.
onnx::ModelProto makeModel(const map<string, string>& meta) {
  onnx::ModelProto model;
  model.set_ir_version(onnx::IR_VERSION_2023_5_5);
  onnx::OperatorSetIdProto* opset = model.add_opset_import();
  opset->set_domain("");
  opset->set_version(20);
  for(const auto& kv: meta) {
    onnx::StringStringEntryProto* p = model.add_metadata_props();
    p->set_key(kv.first);
    p->set_value(kv.second);
  }
  const int nnXLen = Global::stringToInt(meta.at("katago.build.nnXLen"));
  const int nnYLen = Global::stringToInt(meta.at("katago.build.nnYLen"));
  const int spatialCh = Global::stringToInt(meta.at("katago.numInputChannels"));
  const int globalCh = Global::stringToInt(meta.at("katago.numInputGlobalChannels"));
  const int metaCh = Global::stringToInt(meta.at("katago.numInputMetaChannels"));
  const int policyCh = Global::stringToInt(meta.at("katago.numPolicyChannels"));
  const int valueCh = Global::stringToInt(meta.at("katago.numValueChannels"));
  const int scoreCh = Global::stringToInt(meta.at("katago.numScoreValueChannels"));
  const int ownershipCh = Global::stringToInt(meta.at("katago.numOwnershipChannels"));

  onnx::GraphProto* graph = model.mutable_graph();
  graph->set_name("onnxfiletest");
  setTensor(graph->add_input(), "InputSpatial", spatialCh, nnYLen, nnXLen);
  setTensor(graph->add_input(), "InputGlobal", globalCh, 1, 1);
  if(metaCh > 0)
    setTensor(graph->add_input(), "InputMeta", metaCh, 1, 1);
  setTensor(graph->add_input(), "InputMask", 1, nnYLen, nnXLen);
  setTensor(graph->add_output(), "OutputPolicyPass", policyCh, 1, 1);
  setTensor(graph->add_output(), "OutputPolicy", policyCh, nnYLen, nnXLen);
  setTensor(graph->add_output(), "OutputValue", valueCh, 1, 1);
  setTensor(graph->add_output(), "OutputScoreValue", scoreCh, 1, 1);
  setTensor(graph->add_output(), "OutputOwnership", ownershipCh, nnYLen, nnXLen);

  for(int i = 0; i < graph->input_size(); i++) {
    const string& name = graph->input(i).name();
    if(name == "InputMask")
      continue;
    onnx::NodeProto* node = graph->add_node();
    node->set_op_type("Identity");
    node->set_name(name + "/id");
    node->add_input(name);
    node->add_output(name + "/idout");
  }
  return model;
}

void setMeta(onnx::ModelProto& model, const string& key, const string& value) {
  for(int i = 0; i < model.metadata_props_size(); i++) {
    if(model.metadata_props(i).key() == key) {
      model.mutable_metadata_props(i)->set_value(value);
      return;
    }
  }
  onnx::StringStringEntryProto* p = model.add_metadata_props();
  p->set_key(key);
  p->set_value(value);
}

void removeMeta(onnx::ModelProto& model, const string& key) {
  for(int i = 0; i < model.metadata_props_size(); i++) {
    if(model.metadata_props(i).key() == key) {
      model.mutable_metadata_props()->DeleteSubrange(i, 1);
      return;
    }
  }
  testAssert(false);
}

onnx::ValueInfoProto* findIO(onnx::ModelProto& model, bool isInput, const string& name) {
  onnx::GraphProto* graph = model.mutable_graph();
  const int n = isInput ? graph->input_size() : graph->output_size();
  for(int i = 0; i < n; i++) {
    onnx::ValueInfoProto* vi = isInput ? graph->mutable_input(i) : graph->mutable_output(i);
    if(vi->name() == name)
      return vi;
  }
  testAssert(false);
  return NULL;
}

void writeModelFile(const onnx::ModelProto& model, const string& path) {
  string bytes;
  testAssert(model.SerializeToString(&bytes));
  ofstream out;
  FileUtils::open(out, path, ios::out | ios::binary);
  out.write(bytes.data(), (streamsize)bytes.size());
  out.close();
  testAssert(!out.fail());
}

void expectError(const std::function<void()>& f, const string& substring, const string& label) {
  try {
    f();
  }
  catch(const StringError& e) {
    if(string(e.what()).find(substring) == string::npos)
      throw StringError("Test '" + label + "': error did not contain '" + substring + "', got: " + e.what());
    return;
  }
  throw StringError("Test '" + label + "': expected an error containing '" + substring + "'");
}

OnnxModelBuilder::LoadResult writeAndLoad(const onnx::ModelProto& model, const string& path, ModelDesc& descBuf) {
  writeModelFile(model, path);
  return OnnxModelBuilder::load(path, "", descBuf, NULL);
}

void expectLoadError(const onnx::ModelProto& model, const string& path, const string& substring, const string& label) {
  expectError(
    [&]() {
      ModelDesc desc;
      writeAndLoad(model, path, desc);
    },
    substring, label);
}

}  // namespace

void Tests::runOnnxModelFileTests(const string& scratchDir, const string& modelFile) {
  cout << "Running onnx model file tests" << endl;
  MakeDir::make(scratchDir);
  const string path = scratchDir + "/onnxmodelfiletest.onnx";

  // ---- isOnnxFileName ----
  testAssert(OnnxModelBuilder::isOnnxFileName("foo.onnx"));
  testAssert(OnnxModelBuilder::isOnnxFileName("FOO.ONNX"));
  testAssert(OnnxModelBuilder::isOnnxFileName("foo.onnx.gz"));
  testAssert(!OnnxModelBuilder::isOnnxFileName("foo.bin.gz"));
  testAssert(!OnnxModelBuilder::isOnnxFileName("foo.gz"));
  testAssert(!OnnxModelBuilder::isOnnxFileName("foo.onnxx"));
  testAssert(!OnnxModelBuilder::isOnnxFileName("fooonnx"));

  // ---- checkRuntimeParams ----
  {
    OnnxModelBuilder::LoadResult lr;
    lr.buildParams.nnXLen = 19;
    lr.buildParams.nnYLen = 19;
    lr.buildParams.requireExactNNLen = false;
    OnnxModelBuilder::checkRuntimeParams(lr, "m.onnx", 19, 19, false);
    // A masked graph in an exact-size run is merely conservative, so it is allowed.
    OnnxModelBuilder::checkRuntimeParams(lr, "m.onnx", 19, 19, true);
    expectError(
      [&]() { OnnxModelBuilder::checkRuntimeParams(lr, "m.onnx", 13, 13, false); },
      "was emitted for a 19x19 board buffer", "board size mismatch");
    lr.buildParams.requireExactNNLen = true;
    OnnxModelBuilder::checkRuntimeParams(lr, "m.onnx", 19, 19, true);
    expectError(
      [&]() { OnnxModelBuilder::checkRuntimeParams(lr, "m.onnx", 19, 19, false); },
      "requireExactNNLen", "exact graph in masked run");
  }

  // ---- Happy path, model version 15 ----
  {
    onnx::ModelProto model = makeModel(makeMeta(15, 2));
    ModelDesc desc;
    OnnxModelBuilder::LoadResult lr = writeAndLoad(model, path, desc);
    testAssert(desc.name == "onnxfiletest");
    testAssert(desc.modelVersion == 15);
    testAssert(desc.numInputChannels == NNModelVersion::getNumSpatialFeatures(15));
    testAssert(desc.numInputGlobalChannels == NNModelVersion::getNumGlobalFeatures(15));
    testAssert(desc.numInputMetaChannels == 0);
    testAssert(desc.numPolicyChannels == 2);
    testAssert(desc.numValueChannels == 3);
    testAssert(desc.numScoreValueChannels == 6);
    testAssert(desc.numOwnershipChannels == 1);
    testAssert(desc.metaEncoderVersion == 0);
    testAssert(desc.preferPassAliveUnderSuicideRules == false);
    testAssert(desc.preferExcludeTerritoryAdjacentToAtari == false);
    testAssert(desc.postProcessParams.tdScoreMultiplier == 20.0);
    testAssert(desc.postProcessParams.varianceTimeMultiplier == 40.0);
    testAssert(desc.postProcessParams.shorttermValueErrorMultiplier == 0.25);
    testAssert(desc.postProcessParams.shorttermScoreErrorMultiplier == 150.0);
    testAssert(desc.postProcessParams.outputScaleMultiplier == 1.0f);
    // Arch summary is present-but-unknown: the getters report the recorded zeros rather than
    // walking the (empty) layer structure.
    testAssert(desc.archSummary.present);
    testAssert(desc.getTrunkSpatialConvDepth() == 0.0);
    testAssert(desc.getNumParameters() == 0);
    testAssert(!desc.hasAnyTransformerBlocks());
    testAssert(!desc.hasAnyNestedBottleneckBlocks());
    testAssert(desc.sha256.size() == 64);
    testAssert(lr.metadataVersion == 1);
    testAssert(lr.sourceSha256 == "");
    testAssert(lr.buildParams.nnXLen == 19 && lr.buildParams.nnYLen == 19);
    testAssert(!lr.buildParams.requireExactNNLen);
    testAssert(!lr.buildParams.transformerNHWC);
    testAssert(!lr.buildParams.scale8Applied);
    testAssert(lr.trunkTipAndHeadNodeNames.empty());
    testAssert(lr.rmsNormNodeNames.empty());
    testAssert(!lr.danglingInputNotDeclaredLast);

    // The reported sha256 is the file's own hash, so verifying against it succeeds.
    ModelDesc desc2;
    OnnxModelBuilder::load(path, desc.sha256, desc2, NULL);
    expectError(
      [&]() {
        ModelDesc d;
        OnnxModelBuilder::load(path, string(64, 'a'), d, NULL);
      },
      "does not match the expected sha256", "wrong expected sha256");
  }

  // ---- Optional keys ----
  {
    onnx::ModelProto model = makeModel(makeMeta(15, 2));
    setMeta(model, "katago.info.sourceSha256", "abc123");
    setMeta(model, "katago.postProcess.outputScaleMultiplier", "8");
    setMeta(model, "katago.build.requireExactNNLen", "true");
    setMeta(model, "katago.build.transformerNHWC", "true");
    setMeta(model, "katago.build.scale8Applied", "true");
    setMeta(model, "katago.info.arch.trunkSpatialConvDepth", "14.5");
    setMeta(model, "katago.info.arch.numParameters", "123456789012");
    setMeta(model, "katago.info.arch.hasAnyTransformerBlocks", "true");
    setMeta(model, "katago.info.arch.hasAnyNestedBottleneckBlocks", "true");
    setMeta(model, "katago.fp32Nodes.trunkTipAndHead", "node/a\nnode/b");
    setMeta(model, "katago.fp32Nodes.rmsNorm", "\nnode/c\n");
    // Only the must-understand namespace is policed: a reporting key from a newer writer and a key
    // belonging to some other tool both have to load.
    setMeta(model, "katago.info.someReportingKeyFromTheFuture", "ignored");
    setMeta(model, "some.other.tools.key", "ignored");
    ModelDesc desc;
    OnnxModelBuilder::LoadResult lr = writeAndLoad(model, path, desc);
    testAssert(lr.sourceSha256 == "abc123");
    testAssert(desc.postProcessParams.outputScaleMultiplier == 8.0f);
    testAssert(lr.buildParams.requireExactNNLen);
    testAssert(lr.buildParams.transformerNHWC);
    testAssert(lr.buildParams.scale8Applied);
    testAssert(desc.getTrunkSpatialConvDepth() == 14.5);
    testAssert(desc.getNumParameters() == 123456789012LL);
    testAssert(desc.hasAnyTransformerBlocks());
    testAssert(desc.hasAnyNestedBottleneckBlocks());
    testAssert(lr.trunkTipAndHeadNodeNames == (vector<string>{"node/a", "node/b"}));
    testAssert(lr.rmsNormNodeNames == (vector<string>{"node/c"}));
  }

  // ---- Old model version: version-gated keys become optional and default ----
  {
    onnx::ModelProto model = makeModel(makeMeta(8, 1));
    ModelDesc desc;
    writeAndLoad(model, path, desc);
    testAssert(desc.modelVersion == 8);
    testAssert(desc.numPolicyChannels == 1);
    testAssert(desc.numScoreValueChannels == 4);
    testAssert(desc.metaEncoderVersion == 0);
    testAssert(!desc.preferPassAliveUnderSuicideRules);
    testAssert(!desc.preferExcludeTerritoryAdjacentToAtari);
    const ModelPostProcessParams dflt;
    testAssert(desc.postProcessParams.tdScoreMultiplier == dflt.tdScoreMultiplier);
    testAssert(desc.postProcessParams.leadMultiplier == dflt.leadMultiplier);
    testAssert(desc.postProcessParams.shorttermValueErrorMultiplier == dflt.shorttermValueErrorMultiplier);
    testAssert(desc.postProcessParams.shorttermScoreErrorMultiplier == dflt.shorttermScoreErrorMultiplier);
    testAssert(desc.postProcessParams.outputScaleMultiplier == dflt.outputScaleMultiplier);
  }

  // ---- 4 policy channels are allowed from model version 16 on ----
  {
    onnx::ModelProto model = makeModel(makeMeta(16, 4));
    ModelDesc desc;
    writeAndLoad(model, path, desc);
    testAssert(desc.numPolicyChannels == 4);
  }

  // ---- HumanSL-style model with an InputMeta input ----
  {
    map<string, string> meta = makeMeta(15, 2);
    meta["katago.metaEncoderVersion"] = "1";
    meta["katago.numInputMetaChannels"] = Global::intToString(NNModelVersion::getNumInputMetaChannels(1));
    onnx::ModelProto model = makeModel(meta);
    ModelDesc desc;
    OnnxModelBuilder::LoadResult lr = writeAndLoad(model, path, desc);
    testAssert(desc.metaEncoderVersion == 1);
    testAssert(desc.numInputMetaChannels == NNModelVersion::getNumInputMetaChannels(1));
    testAssert(!lr.danglingInputNotDeclaredLast);
  }

  // ---- The rules-related model options reach the ModelDesc when set ----
  {
    map<string, string> meta = makeMeta(15, 2);
    meta["katago.preferPassAliveUnderSuicideRules"] = "true";
    meta["katago.preferExcludeTerritoryAdjacentToAtari"] = "true";
    onnx::ModelProto model = makeModel(meta);
    ModelDesc desc;
    writeAndLoad(model, path, desc);
    testAssert(desc.preferPassAliveUnderSuicideRules);
    testAssert(desc.preferExcludeTerritoryAdjacentToAtari);
  }

  // ---- Metadata error paths ----
  const onnx::ModelProto base = makeModel(makeMeta(15, 2));
  auto withMeta = [&](const string& key, const string& value) {
    onnx::ModelProto m = base;
    setMeta(m, key, value);
    return m;
  };
  auto withoutMeta = [&](const string& key) {
    onnx::ModelProto m = base;
    removeMeta(m, key);
    return m;
  };
  expectLoadError(withoutMeta("katago.metadataVersion"), path, "carries no KataGo metadata", "metadata block missing");
  expectLoadError(withMeta("katago.metadataVersion", "99"), path, "understands up to version", "metadata version too new");
  expectLoadError(withMeta("katago.metadataVersion", "0"), path, "no longer reads", "metadata version too old");
  expectLoadError(withMeta("katago.metadataVersion", "banana"), path, "is not an integer", "metadata version not an int");
  expectLoadError(withoutMeta("katago.numValueChannels"), path, "missing required key", "required key missing");
  expectLoadError(withMeta("katago.numValueChannels", "4"), path, "requires 3", "wrong channel count");
  expectLoadError(withMeta("katago.modelVersion", "2"), path, "no longer supported", "model version too old");
  expectLoadError(
    withMeta("katago.modelVersion", Global::intToString(NNModelVersion::latestModelVersionImplemented + 1)),
    path, "requires a newer KataGo", "model version too new");
  expectLoadError(withMeta("katago.name", ""), path, "name is empty", "empty name");
  expectLoadError(withMeta("katago.name", "bad/name"), path, "alphanumeric", "name with bad chars");
  expectLoadError(withMeta("katago.name", string(97, 'x')), path, "too long", "name too long");
  expectLoadError(withMeta("katago.numPolicyChannels", "4"), path, "not supported for model version", "4 policy channels before v16");
  expectLoadError(withMeta("katago.postProcess.leadMultiplier", "0"), path, "positive finite", "zero multiplier");
  expectLoadError(withMeta("katago.postProcess.leadMultiplier", "-5"), path, "positive finite", "negative multiplier");
  expectLoadError(withoutMeta("katago.postProcess.leadMultiplier"), path, "missing required key", "postProcess required at v13+");
  expectLoadError(withMeta("katago.metaEncoderVersion", "2"), path, "not implemented", "unknown metaEncoderVersion");
  expectLoadError(withMeta("katago.numInputMetaChannels", "5"), path, "numInputMetaChannels", "meta channels without encoder");
  expectLoadError(withMeta("katago.build.nnXLen", "1"), path, "outside the supported range", "board size too small");
  expectLoadError(
    withMeta("katago.build.nnXLen", Global::intToString(NNPos::MAX_BOARD_LEN + 1)),
    path, "outside the supported range", "board size too large");
  expectLoadError(withoutMeta("katago.build.nnXLen"), path, "missing required key", "build params missing");
  expectLoadError(withMeta("katago.preferPassAliveUnderSuicideRules", "maybe"), path, "is not a boolean", "bad boolean");
  expectLoadError(
    withoutMeta("katago.preferExcludeTerritoryAdjacentToAtari"),
    path, "missing required key", "rules model option missing at v15+");
  expectLoadError(
    withMeta("katago.someKeyFromTheFuture", "surprise"),
    path, "does not know: katago.someKeyFromTheFuture", "unknown must-understand key");
  // A misspelled key is an unknown key, which is the whole point of policing the namespace: the
  // required-key check alone would only report the real key as missing.
  expectLoadError(
    withMeta("katago.preferPassAliveUnderSuicideRule", "true"),
    path, "does not know: katago.preferPassAliveUnderSuicideRule", "misspelled key");

  // ---- Graph/metadata mismatch error paths ----
  {
    onnx::ModelProto m = base;
    testAssert(m.graph().input(m.graph().input_size() - 1).name() == "InputMask");
    m.mutable_graph()->mutable_input()->RemoveLast();
    expectLoadError(m, path, "no input named 'InputMask'", "missing graph input");
  }
  {
    onnx::ModelProto m = base;
    setTensor(m.mutable_graph()->add_input(), "Bogus", 3, 1, 1);
    expectLoadError(m, path, "unexpected input", "extra graph input");
  }
  {
    onnx::ModelProto m = base;
    setTensor(m.mutable_graph()->add_output(), "OutputBogus", 3, 1, 1);
    expectLoadError(m, path, "unexpected output", "extra graph output");
  }
  {
    onnx::ModelProto m = base;
    findIO(m, false, "OutputValue")->mutable_type()->mutable_tensor_type()->set_elem_type(onnx::TensorProto::INT32);
    expectLoadError(m, path, "binds float32", "wrong element type");
  }
  {
    onnx::ModelProto m = base;
    onnx::TensorShapeProto::Dimension* dim =
      findIO(m, true, "InputSpatial")->mutable_type()->mutable_tensor_type()->mutable_shape()->mutable_dim(0);
    dim->clear_dim_param();
    dim->set_dim_value(1);
    expectLoadError(m, path, "fixed batch dimension", "fixed batch");
  }
  {
    onnx::ModelProto m = base;
    findIO(m, false, "OutputPolicy")->mutable_type()->mutable_tensor_type()->mutable_shape()->mutable_dim(2)->set_dim_value(13);
    expectLoadError(m, path, "height dimension 13", "wrong spatial size");
  }
  {
    onnx::ModelProto m = base;
    findIO(m, true, "InputGlobal")->mutable_type()->mutable_tensor_type()->mutable_shape()->mutable_dim()->RemoveLast();
    expectLoadError(m, path, "expected rank 4", "wrong rank");
  }
  {
    onnx::ModelProto m = base;
    onnx::TensorShapeProto::Dimension* dim =
      findIO(m, true, "InputSpatial")->mutable_type()->mutable_tensor_type()->mutable_shape()->mutable_dim(1);
    dim->clear_dim_value();
    dim->set_dim_param("c");
    expectLoadError(m, path, "symbolic channel", "symbolic channel dim");
  }
  {
    onnx::ModelProto m = base;
    onnx::ValueInfoProto maskCopy = *findIO(m, true, "InputMask");
    *m.mutable_graph()->add_input() = maskCopy;
    expectLoadError(m, path, "more than once", "duplicate input");
  }

  // ---- Input declaration order and the dangling-input flag ----
  {
    // InputMask declared first: it is consumed by nothing and sits ahead of consumed inputs, the
    // exact hazard for the OpenVINO execution provider.
    onnx::ModelProto m = base;
    m.mutable_graph()->mutable_input()->SwapElements(0, m.graph().input_size() - 1);
    ModelDesc desc;
    OnnxModelBuilder::LoadResult lr = writeAndLoad(m, path, desc);
    testAssert(lr.danglingInputNotDeclaredLast);
  }
  {
    // Initializers redundantly declared as graph inputs (a legacy ONNX convention) are weights, not
    // IO: they must not count as unexpected inputs.
    onnx::ModelProto m = base;
    onnx::TensorProto* init = m.mutable_graph()->add_initializer();
    init->set_name("SomeWeight");
    init->set_data_type(onnx::TensorProto::FLOAT);
    init->add_dims(1);
    init->add_float_data(1.0f);
    setTensor(m.mutable_graph()->add_input(), "SomeWeight", 1, 1, 1);
    // Keep InputMask last and consume the weight, so the declaration-order rule is still satisfied.
    m.mutable_graph()->mutable_input()->SwapElements(m.graph().input_size() - 2, m.graph().input_size() - 1);
    onnx::NodeProto* node = m.mutable_graph()->add_node();
    node->set_op_type("Identity");
    node->set_name("SomeWeight/id");
    node->add_input("SomeWeight");
    node->add_output("SomeWeight/idout");
    ModelDesc desc;
    OnnxModelBuilder::LoadResult lr = writeAndLoad(m, path, desc);
    testAssert(!lr.danglingInputNotDeclaredLast);
  }

  // ---- Not an ONNX file at all ----
  {
    ofstream out;
    FileUtils::open(out, path, ios::out | ios::binary);
    out << "this is not an onnx file";
    out.close();
    expectError(
      [&]() {
        ModelDesc desc;
        OnnxModelBuilder::load(path, "", desc, NULL);
      },
      "could not be parsed as an ONNX ModelProto", "garbage file");
  }

  // ---- Round trip through the real emitter ----
  if(modelFile != "") {
    cout << "Running onnx dump/load round trip on " << modelFile << endl;
    ModelDesc srcDesc;
    ModelDesc::loadFromFileMaybeGZipped(modelFile, srcDesc, "");
    OnnxModelBuilder::BuildParams buildParams;
    buildParams.nnXLen = 19;
    buildParams.nnYLen = 19;
    buildParams.requireExactNNLen = false;
    buildParams.transformerNHWC = true;
    buildParams.scale8Applied = srcDesc.applyScale8ToReduceActivations();
    OnnxModelBuilder::Result result = OnnxModelBuilder::build(srcDesc, buildParams, NULL);
    {
      ofstream out;
      FileUtils::open(out, path, ios::out | ios::binary);
      out.write(result.serializedModel.data(), (streamsize)result.serializedModel.size());
      out.close();
      testAssert(!out.fail());
    }
    ModelDesc desc;
    OnnxModelBuilder::LoadResult lr = OnnxModelBuilder::load(path, "", desc, NULL);
    testAssert(desc.name == srcDesc.name);
    testAssert(desc.modelVersion == srcDesc.modelVersion);
    testAssert(desc.numInputChannels == srcDesc.numInputChannels);
    testAssert(desc.numInputGlobalChannels == srcDesc.numInputGlobalChannels);
    testAssert(desc.numInputMetaChannels == srcDesc.numInputMetaChannels);
    testAssert(desc.numPolicyChannels == srcDesc.numPolicyChannels);
    testAssert(desc.numValueChannels == srcDesc.numValueChannels);
    testAssert(desc.numScoreValueChannels == srcDesc.numScoreValueChannels);
    testAssert(desc.numOwnershipChannels == srcDesc.numOwnershipChannels);
    testAssert(desc.metaEncoderVersion == srcDesc.metaEncoderVersion);
    testAssert(desc.preferPassAliveUnderSuicideRules == srcDesc.preferPassAliveUnderSuicideRules);
    testAssert(desc.preferExcludeTerritoryAdjacentToAtari == srcDesc.preferExcludeTerritoryAdjacentToAtari);
    const ModelPostProcessParams& a = desc.postProcessParams;
    const ModelPostProcessParams& b = srcDesc.postProcessParams;
    testAssert(a.tdScoreMultiplier == b.tdScoreMultiplier);
    testAssert(a.scoreMeanMultiplier == b.scoreMeanMultiplier);
    testAssert(a.scoreStdevMultiplier == b.scoreStdevMultiplier);
    testAssert(a.leadMultiplier == b.leadMultiplier);
    testAssert(a.varianceTimeMultiplier == b.varianceTimeMultiplier);
    testAssert(a.shorttermValueErrorMultiplier == b.shorttermValueErrorMultiplier);
    testAssert(a.shorttermScoreErrorMultiplier == b.shorttermScoreErrorMultiplier);
    testAssert(a.outputScaleMultiplier == b.outputScaleMultiplier);
    testAssert(lr.metadataVersion >= 1);
    testAssert(lr.sourceSha256 == srcDesc.sha256);
    testAssert(lr.buildParams.nnXLen == buildParams.nnXLen && lr.buildParams.nnYLen == buildParams.nnYLen);
    testAssert(lr.buildParams.requireExactNNLen == buildParams.requireExactNNLen);
    // build() normalizes NHWC to false for models with no transformer blocks.
    testAssert(lr.buildParams.transformerNHWC == srcDesc.hasAnyTransformerBlocks());
    testAssert(lr.buildParams.scale8Applied == buildParams.scale8Applied);
    testAssert(lr.trunkTipAndHeadNodeNames == result.trunkTipAndHeadNodeNames);
    testAssert(lr.rmsNormNodeNames == result.rmsNormNodeNames);
    testAssert(!lr.danglingInputNotDeclaredLast);
    // The arch summary recorded in the metadata reproduces what walking the layer structure gives.
    testAssert(desc.getTrunkSpatialConvDepth() == srcDesc.getTrunkSpatialConvDepth());
    testAssert(desc.getNumParameters() == srcDesc.getNumParameters());
    testAssert(desc.hasAnyTransformerBlocks() == srcDesc.hasAnyTransformerBlocks());
    testAssert(desc.hasAnyNestedBottleneckBlocks() == srcDesc.hasAnyNestedBottleneckBlocks());
  }

  cout << "Onnx model file tests passed" << endl;
}

#else  // no TensorRT or ONNX backend

void Tests::runOnnxModelFileTests(const std::string& scratchDir, const std::string& modelFile) {
  (void)scratchDir;
  (void)modelFile;
  throw StringError(
    "runonnxmodelfiletests requires a build with the TensorRT or ONNX backend, since those are the "
    "backends that read .onnx model files.");
}

#endif
