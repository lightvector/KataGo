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
  string gpuIdxsStr;
  vector<int> gpuIdxs;
  int nnXLen;
  int nnYLen;
  int batchSize;
  int winograd3x3TileSize;
  bool full;
  try {
    TCLAP::CmdLine cmd("Perform GPU tuning", ' ', Version::getKataGoVersionForHelp(),true);
    TCLAP::ValueArg<string> modelFileArg("","model","Neural net model file to use",true,string(),"FILE");
    TCLAP::ValueArg<string> outputFileArg("","output","Filename to output tuning configration to",false,string(),"FILE");
    TCLAP::ValueArg<string> gpuIdxsArg("","gpus","Specific GPU/device number(s) to tune, comma-separated (default all)",false,string(),"GPUS");
    TCLAP::ValueArg<int> nnXLenArg("","xsize","Width of board to tune for",false,OpenCLTuner::DEFAULT_X_SIZE,"INT");
    TCLAP::ValueArg<int> nnYLenArg("","ysize","Height of board to tune for",false,OpenCLTuner::DEFAULT_Y_SIZE,"INT");
    TCLAP::ValueArg<int> batchSizeArg("","batchsize","Batch size to tune for",false,OpenCLTuner::DEFAULT_BATCH_SIZE,"INT");
    TCLAP::ValueArg<int> winograd3x3TileSizeArg("","winograd3x3tilesize","Batch size to tune for",false,OpenCLTuner::DEFAULT_WINOGRAD_3X3_TILE_SIZE,"INT");
    TCLAP::SwitchArg fullArg("","full","Test more possible configurations");
    cmd.add(modelFileArg);
    cmd.add(outputFileArg);
    cmd.add(gpuIdxsArg);
    cmd.add(nnXLenArg);
    cmd.add(nnYLenArg);
    cmd.add(batchSizeArg);
    cmd.add(fullArg);
    cmd.parse(argc,argv);
    modelFile = modelFileArg.getValue();
    outputFile = outputFileArg.getValue();
    gpuIdxsStr = gpuIdxsArg.getValue();
    nnXLen = nnXLenArg.getValue();
    nnYLen = nnYLenArg.getValue();
    batchSize = batchSizeArg.getValue();
    winograd3x3TileSize = winograd3x3TileSizeArg.getValue();
    full = fullArg.getValue();

    if(gpuIdxsStr.size() > 0) {
      vector<string> pieces = Global::split(gpuIdxsStr,',');
      int parsed;
      for(int i = 0; i<pieces.size(); i++) {
        bool suc = Global::tryStringToInt(Global::trim(pieces[i]),parsed);
        if(!suc) {
          cerr << "Error: Could not parse -gpus as a comma-separated integer list: " << parsed << endl;
          return 1;
        }
        if(parsed < 0) {
          cerr << "Error: Provided negative value for -gpus: " << parsed << endl;
          return 1;
        }
        if(contains(gpuIdxs,parsed)) {
          cerr << "Error: Provided duplicate value for -gpus: " << parsed << endl;
          return 1;
        }
        gpuIdxs.push_back(parsed);
      }
    }
  }
  catch (TCLAP::ArgException &e) {
    cerr << "Error: " << e.error() << " for argument " << e.argId() << endl;
    return 1;
  }

  Logger logger;
  logger.setLogToStdout(true);

  logger.write("Loading model...");
  ModelDesc modelDesc;
  ModelDesc::loadFromFileMaybeGZipped(modelFile, modelDesc);

  logger.write("Querying system devices...");
  vector<DeviceInfo> allDeviceInfos = DeviceInfo::getAllDeviceInfosOnSystem(&logger);

  //If none provided, by default tune everything
  if(gpuIdxs.size() == 0) {
    for(int i = 0; i<allDeviceInfos.size(); i++) {
      gpuIdxs.push_back(allDeviceInfos[i].gpuIdx);
    }
  }

  logger.write("Tuner starting...");

  //Avoid tuning a gpu with the same name more than once
  set<string> alreadyTunedNames;

  for(int i = 0; i < gpuIdxs.size(); i++) {
    int gpuIdx = gpuIdxs[i];

    bool enableProfiling = true;
    DevicesContext devicesContext(allDeviceInfos, {gpuIdx}, &logger, enableProfiling);

    cout << "==============================================================================" << endl;
    const InitializedDevice& device = devicesContext.findGpuExn(gpuIdx);
    if(contains(alreadyTunedNames, device.info.name)) {
      cout << "Skipping tuning " << gpuIdx << " due to same name as an earlier tuned GPU: " << device.info.name << endl;
      continue;
    }
    alreadyTunedNames.insert(device.info.name);
    cout << "Tuning device " << gpuIdx << ": " << device.info.name << endl;

    if(outputFile == "") {
      string dir = OpenCLTuner::defaultDirectory(true);
      outputFile = dir + "/" + OpenCLTuner::defaultFileName(device.info.name, nnXLen, nnYLen, &modelDesc);
    }

    OpenCLTuneParams initialParams;
    try {
      OpenCLTuneParams loadedParams = OpenCLTuneParams::load(outputFile);
      initialParams = loadedParams;
      cout << "Starting from existing parameters in: " + outputFile << endl;
    }
    catch(const StringError& e) {
      (void)e;
      cout << "File does not alrady exist or unable to parse parameters in: " + outputFile << endl;
      cout << "Starting fresh tuning, saving results to " << outputFile << endl;
    }

    OpenCLTuneParams results;
    auto handleBestSoFar = [&results](const OpenCLTuneParams& bestSoFar) {
      results = bestSoFar;
    };
    OpenCLTuner::tune(
      initialParams,
      devicesContext,
      gpuIdx,
      batchSize,
      nnXLen,
      nnYLen,
      &modelDesc,
      full,
      winograd3x3TileSize,
      cout,
      std::function<void(const OpenCLTuneParams&)>(handleBestSoFar)
    );

    OpenCLTuneParams::save(outputFile, results);
    cout << "Done, results saved to " << outputFile << endl;
  }

  return 0;

#endif
}
