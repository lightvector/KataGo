#include "../core/global.h"
#include "../core/fileutils.h"
#include "../core/logger.h"
#include "../command/commandline.h"
#include "../main.h"

#if defined(USE_TENSORRT_BACKEND) || defined(USE_ONNX_BACKEND)
#include "../neuralnet/desc.h"
#include "../neuralnet/nninputs.h"
#include "../neuralnet/onnxmodelbuilder.h"

#include <fstream>
#endif

using namespace std;

// Writes out the ONNX graph that the TensorRT and ONNX backends build internally from a .bin.gz
// model, for inspection with external ONNX tooling and for feeding back to those backends as a model
// file in its own right. See docs/ONNX_Model_Files.md.
//
// The board size and the masking mode are baked into the graph and checked when it is loaded, so a
// dump is only usable at the settings it was made with. The defaults here match a normal 19x19 run.
int MainCmds::dumponnx(const vector<string>& args) {
#if !defined(USE_TENSORRT_BACKEND) && !defined(USE_ONNX_BACKEND)
  (void)args;
  cerr << "dumponnx is only available in builds with the TensorRT or ONNX backend, since those are "
       << "the backends that build ONNX graphs. Compile with -DUSE_BACKEND=TENSORRT or -DUSE_BACKEND=ONNX."
       << endl;
  return 1;
#else
  string modelFile;
  string outputFile;
  int nnXLen;
  int nnYLen;
  bool requireExactNNLen;
  bool transformerNHWC;
  bool skipScale8;
  try {
    KataGoCommandLine cmd("Dump the ONNX graph that KataGo builds for a model.");
    cmd.addModelFileArg();

    TCLAP::ValueArg<string> outputFileArg(
      "", "out", "Path of the .onnx file to write.", true, string(), "FILE");
    TCLAP::ValueArg<int> nnXLenArg(
      "", "nn-x-len", "Board width the graph is built for (default 19).", false, 19, "LEN");
    TCLAP::ValueArg<int> nnYLenArg(
      "", "nn-y-len", "Board height the graph is built for (default 19).", false, 19, "LEN");
    TCLAP::SwitchArg requireExactNNLenArg(
      "", "require-exact-nnlen",
      "Build a graph with no board masking, only correct if every position fills the whole buffer. "
      "Matches requireMaxBoardSize = true in a config.");
    // A string arg rather than ValueArg<bool>: TCLAP's bool parsing accepts only 0/1, not the
    // true/false used everywhere in KataGo configs.
    TCLAP::ValueArg<string> transformerNHWCArg(
      "", "transformer-nhwc",
      "Run transformer trunk blocks channel-last (default true). No effect on models without "
      "transformer blocks.", false, "true", "BOOL");
    TCLAP::SwitchArg skipScale8Arg(
      "", "skip-scale8",
      "Skip the 1/8 activation rescaling that keeps convnet activations inside the FP16 range. "
      "Matches onnxSkipScale8 = true in a config; the TensorRT backend always applies it.");
    cmd.add(outputFileArg);
    cmd.add(nnXLenArg);
    cmd.add(nnYLenArg);
    cmd.add(requireExactNNLenArg);
    cmd.add(transformerNHWCArg);
    cmd.add(skipScale8Arg);
    cmd.parseArgs(args);

    modelFile = cmd.getModelFile();
    outputFile = outputFileArg.getValue();
    nnXLen = nnXLenArg.getValue();
    nnYLen = nnYLenArg.getValue();
    requireExactNNLen = requireExactNNLenArg.getValue();
    skipScale8 = skipScale8Arg.getValue();

    const string& nhwcStr = transformerNHWCArg.getValue();
    if(nhwcStr == "0")
      transformerNHWC = false;
    else if(nhwcStr == "1")
      transformerNHWC = true;
    else if(!Global::tryStringToBool(nhwcStr, transformerNHWC)) {
      cerr << "-transformer-nhwc must be true or false, got: " << nhwcStr << endl;
      return 1;
    }
  }
  catch(TCLAP::ArgException& e) {
    cerr << "Error: " << e.error() << " for argument " << e.argId() << endl;
    return 1;
  }

  if(nnXLen < 2 || nnXLen > NNPos::MAX_BOARD_LEN || nnYLen < 2 || nnYLen > NNPos::MAX_BOARD_LEN) {
    cerr << "Board size must be between 2 and " << NNPos::MAX_BOARD_LEN << endl;
    return 1;
  }
  if(OnnxModelBuilder::isOnnxFileName(modelFile)) {
    cerr << "-model must be a KataGo model file (.bin.gz), not an .onnx file. There is nothing to "
         << "build from an .onnx file - it already is the graph." << endl;
    return 1;
  }
  if(!Global::isSuffix(Global::toLower(outputFile), ".onnx")) {
    cerr << "-out must end in .onnx (the backends identify ONNX model files by that suffix; .onnx.gz "
         << "also works, but gzip the file yourself after dumping)." << endl;
    return 1;
  }

  const bool logToStdout = true;
  const bool logToStderr = false;
  const bool logTime = false;
  Logger logger(nullptr, logToStdout, logToStderr, logTime);

  ModelDesc modelDesc;
  ModelDesc::loadFromFileMaybeGZipped(modelFile, modelDesc, "");
  logger.write(
    "Loaded model " + modelDesc.name + " (" + modelDesc.getShortInfoString() + ") from " + modelFile);

  OnnxModelBuilder::BuildParams buildParams;
  buildParams.nnXLen = nnXLen;
  buildParams.nnYLen = nnYLen;
  buildParams.requireExactNNLen = requireExactNNLen;
  buildParams.transformerNHWC = transformerNHWC;
  // The backends apply this to the weights before emitting, so do the same here.
  buildParams.scale8Applied = skipScale8 ? false : modelDesc.applyScale8ToReduceActivations();
  if(skipScale8)
    logger.write("Skipping the scale8 activation rescaling (-skip-scale8)");
  else if(!buildParams.scale8Applied)
    logger.write("Model is not eligible for the scale8 activation rescaling; emitting without it");

  OnnxModelBuilder::Result result = OnnxModelBuilder::build(modelDesc, buildParams, &logger);

  {
    ofstream out;
    FileUtils::open(out, outputFile, ios::out | ios::binary);
    out.write(result.serializedModel.data(), (streamsize)result.serializedModel.size());
    out.close();
    if(out.fail()) {
      cerr << "Failed writing " << outputFile << endl;
      return 1;
    }
  }

  logger.write(Global::strprintf(
    "Wrote %s (%s bytes) for a %dx%d board buffer, requireExactNNLen=%s, transformerNHWC=%s, scale8Applied=%s",
    outputFile.c_str(),
    Global::uint64ToString(result.serializedModel.size()).c_str(),
    nnXLen, nnYLen,
    Global::boolToString(requireExactNNLen).c_str(),
    // build() ignores this for models with no transformer blocks; report what it actually used.
    Global::boolToString(transformerNHWC && modelDesc.hasAnyTransformerBlocks()).c_str(),
    Global::boolToString(buildParams.scale8Applied).c_str()));
  logger.write(
    "This file can be given to -model in place of the .bin.gz on the TensorRT and ONNX backends, at "
    "these same settings.");
  return 0;
#endif
}
