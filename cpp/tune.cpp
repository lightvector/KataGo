#include "core/global.h"
#include "core/config_parser.h"
#include "core/timer.h"
#include "core/makedir.h"
#include "main.h"

#ifdef USE_OPENCL_BACKEND
#include "neuralnet/opencltuner.h"
#endif

using namespace std;

#define TCLAP_NAMESTARTSTRING "-" //Use single dashes for all flags
#include <tclap/CmdLine.h>

int MainCmds::tuner(int argc, const char* const* argv) {
#ifndef USE_OPENCL_BACKEND
  cout << "Currently this command only does anything for the OpenCL version of KataGo" << endl;
  (void)argc;
  (void)argv;
  return 0;
#else

  string modelFile;
  string outputFile;
  int gpuIdx;
  int nnXLen;
  int nnYLen;
  int batchSize;
  bool full;
  try {
    TCLAP::CmdLine cmd("Perform GPU tuning", ' ', Version::getKataGoVersionForHelp(),true);
    TCLAP::ValueArg<string> modelFileArg("","model","Neural net model file to use",true,string(),"FILE");
    TCLAP::ValueArg<string> outputFileArg("","output","Filename to output tuning configration to",false,string(),"FILE");
    TCLAP::ValueArg<int> gpuIdxArg("","gpuIdx","GPU/device number to use",false,0,"FILE");
    TCLAP::ValueArg<int> nnXLenArg("","xSize","Width of board to tune for",false,OpenCLTuner::DEFAULT_X_SIZE,"FILE");
    TCLAP::ValueArg<int> nnYLenArg("","ySize","Height of board to tune for",false,OpenCLTuner::DEFAULT_Y_SIZE,"FILE");
    TCLAP::ValueArg<int> batchSizeArg("","batchSize","Batch size to tune for",false,OpenCLTuner::DEFAULT_BATCH_SIZE,"FILE");
    TCLAP::SwitchArg fullArg("","full","Test more possible configurations");
    cmd.add(modelFileArg);
    cmd.add(outputFileArg);
    cmd.add(gpuIdxArg);
    cmd.add(nnXLenArg);
    cmd.add(nnYLenArg);
    cmd.add(batchSizeArg);
    cmd.add(fullArg);
    cmd.parse(argc,argv);
    modelFile = modelFileArg.getValue();
    outputFile = outputFileArg.getValue();
    gpuIdx = gpuIdxArg.getValue();
    nnXLen = nnXLenArg.getValue();
    nnYLen = nnYLenArg.getValue();
    batchSize = batchSizeArg.getValue();
    full = fullArg.getValue();
  }
  catch (TCLAP::ArgException &e) {
    cerr << "Error: " << e.error() << " for argument " << e.argId() << endl;
    return 1;
  }

  ModelDesc modelDesc;
  ModelDesc::loadFromFileMaybeGZipped(modelFile, modelDesc);

  if(outputFile == "") {
    string dir = OpenCLTuner::defaultDirectory(true);
    outputFile = dir + "/" + OpenCLTuner::defaultFileName(gpuIdx, nnXLen, nnYLen, &modelDesc);
  }

  OpenCLTuneParams initialParams;
  try {
    OpenCLTuneParams loadedParams = OpenCLTuneParams::load(outputFile);
    initialParams = loadedParams;
    cout << "Starting from existing parameters in: " + outputFile << endl;
  }
  catch(const StringError& e) {
    cout << "File does not alrady exist or unable to parse parameters in: " + outputFile << endl;
    cout << "Starting fresh tuning, saving results to " << outputFile << endl;
  }

  Logger logger;
  logger.setLogToStdout(true);
  logger.write("Tuner starting...");

  auto handleBestSoFar = [&outputFile](const OpenCLTuneParams& bestSoFar) {
    OpenCLTuneParams::save(outputFile, bestSoFar);
  };

  OpenCLTuner::tune(
    initialParams,
    gpuIdx,
    &logger,
    batchSize,
    nnXLen,
    nnYLen,
    &modelDesc,
    full,
    cout,
    std::function<void(const OpenCLTuneParams&)>(handleBestSoFar)
  );

  cout << "Done, results saved to " << outputFile << endl;
  return 0;

#endif
}
