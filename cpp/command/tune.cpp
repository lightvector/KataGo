#include "../core/global.h"
#include "../core/config_parser.h"
#include "../core/commontypes.h"
#include "../core/timer.h"
#include "../core/makedir.h"
#include "../command/commandline.h"
#include "../main.h"

#ifdef USE_OPENCL_BACKEND
#include "../program/setup.h"
#include "../neuralnet/opencltuner.h"
#endif

#ifdef USE_MLX_BACKEND
#include "../program/setup.h"
#include "../neuralnet/desc.h"
#include "../neuralnet/mlxwinotuner.h"
#endif

using namespace std;

int MainCmds::tuner(const vector<string>& args) {
#if defined(USE_OPENCL_BACKEND)

  ConfigParser cfg;
  string modelFile;
  string outputFileFromArg;
  string gpuIdxsStr;
  vector<int> gpuIdxs;
  int nnXLen;
  int nnYLen;
  string testFP16Str;
  string testFP16StorageStr;
  string testFP16ComputeStr;
  string testFP16TensorCoresStr;
  enabled_t testFP16Mode;
  enabled_t testFP16StorageMode;
  enabled_t testFP16ComputeMode;
  enabled_t testFP16TensorCoresMode;
  int batchSize;
  int winograd3x3TileSize;
  bool full;
  bool verboseErrors;
  bool verboseTuner;
  try {
    KataGoCommandLine cmd("Perform GPU tuning for OpenCL.");
    cmd.addConfigFileArg(KataGoCommandLine::defaultGtpConfigFileName(),"gtp_example.cfg");
    cmd.addModelFileArg();

    TCLAP::ValueArg<string> outputFileArg("","output","Filename to output tuning configration to",false,string(),"FILE");
    TCLAP::ValueArg<string> gpuIdxsArg("","gpus","Specific GPU/device number(s) to tune, comma-separated (default all)",false,string(),"GPUS");
    TCLAP::ValueArg<int> nnXLenArg("","xsize","Width of board to tune for",false,OpenCLTuner::DEFAULT_X_SIZE,"INT");
    TCLAP::ValueArg<int> nnYLenArg("","ysize","Height of board to tune for",false,OpenCLTuner::DEFAULT_Y_SIZE,"INT");
    TCLAP::ValueArg<string> testFP16Arg("","testFP16","Test FP16? true|false|auto (default auto)",false,"auto","BOOL_OR_AUTO");
    TCLAP::ValueArg<string> testFP16StorageArg("","testFP16Storage","Test FP16 storage? true|false|auto (default auto)",false,"auto","BOOL_OR_AUTO");
    TCLAP::ValueArg<string> testFP16ComputeArg("","testFP16Compute","Test FP16 compute? true|false|auto (default auto)",false,"auto","BOOL_OR_AUTO");
    TCLAP::ValueArg<string> testFP16TensorCoresArg("","testFP16TensorCores","Test FP16 tensor cores? true|false|auto (default auto)",false,"auto","BOOL_OR_AUTO");
    TCLAP::ValueArg<int> batchSizeArg("","batchsize","Batch size to tune for",false,OpenCLTuner::DEFAULT_BATCH_SIZE,"INT");
    TCLAP::ValueArg<int> winograd3x3TileSizeArg("","winograd3x3tilesize","Batch size to tune for",false,OpenCLTuner::DEFAULT_WINOGRAD_3X3_TILE_SIZE,"INT");
    TCLAP::SwitchArg fullArg("","full","Test more possible configurations");
    TCLAP::SwitchArg verboseErrorsArg("","verboseErrors","Verbosely print out errors for configurations that fail");
    TCLAP::SwitchArg verboseTunerArg("","verboseTuner","Verbosely print out tuner results even if they don't improve the best");

    cmd.setShortUsageArgLimit();
    cmd.addOverrideConfigArg();

    cmd.add(outputFileArg);
    cmd.add(gpuIdxsArg);
    cmd.add(nnXLenArg);
    cmd.add(nnYLenArg);
    cmd.add(testFP16Arg);
    cmd.add(testFP16StorageArg);
    cmd.add(testFP16ComputeArg);
    cmd.add(testFP16TensorCoresArg);
    cmd.add(batchSizeArg);
    cmd.add(fullArg);
    cmd.add(verboseErrorsArg);
    cmd.add(verboseTunerArg);
    cmd.parseArgs(args);

    modelFile = cmd.getModelFile();
    outputFileFromArg = outputFileArg.getValue();
    gpuIdxsStr = gpuIdxsArg.getValue();
    nnXLen = nnXLenArg.getValue();
    nnYLen = nnYLenArg.getValue();
    testFP16Str = testFP16Arg.getValue();
    testFP16StorageStr = testFP16StorageArg.getValue();
    testFP16ComputeStr = testFP16ComputeArg.getValue();
    testFP16TensorCoresStr = testFP16TensorCoresArg.getValue();
    batchSize = batchSizeArg.getValue();
    winograd3x3TileSize = winograd3x3TileSizeArg.getValue();
    full = fullArg.getValue();
    verboseErrors = verboseErrorsArg.getValue();
    verboseTuner = verboseTunerArg.getValue();

    if(!enabled_t::tryParse(testFP16Str,testFP16Mode)) {
      cerr << "Error: Could not parse -testFP16 as bool or auto: " << testFP16Str << endl;
      return 1;
    }
    if(!enabled_t::tryParse(testFP16StorageStr,testFP16StorageMode)) {
      cerr << "Error: Could not parse -testFP16Storage as bool or auto: " << testFP16StorageStr << endl;
      return 1;
    }
    if(!enabled_t::tryParse(testFP16ComputeStr,testFP16ComputeMode)) {
      cerr << "Error: Could not parse -testFP16Compute as bool or auto: " << testFP16ComputeStr << endl;
      return 1;
    }
    if(!enabled_t::tryParse(testFP16TensorCoresStr,testFP16TensorCoresMode)) {
      cerr << "Error: Could not parse -testFP16TensorCores as bool or auto: " << testFP16TensorCoresStr << endl;
      return 1;
    }

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

    cmd.getConfigAllowEmpty(cfg);
  }
  catch (TCLAP::ArgException &e) {
    cerr << "Error: " << e.error() << " for argument " << e.argId() << endl;
    return 1;
  }

  string homeDataDirOverride = Setup::loadHomeDataDirOverride(cfg);

  const bool logToStdoutDefault = true;
  Logger logger(&cfg, logToStdoutDefault);

  logger.write("Loading model...");
  ModelDesc modelDesc;
  string expectedSha256 = "";
  ModelDesc::loadFromFileMaybeGZipped(modelFile, modelDesc, expectedSha256);
  OpenCLTuner::ModelInfoForTuning modelInfo = OpenCLTuner::ModelInfoForTuning::ofDesc(&modelDesc);

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
    const InitializedDevice* device = devicesContext.findGpuExn(gpuIdx);
    if(contains(alreadyTunedNames, device->info.name)) {
      cout << "Skipping tuning " << gpuIdx << " due to same name as an earlier tuned GPU: " << device->info.name << endl;
      continue;
    }
    alreadyTunedNames.insert(device->info.name);
    cout << "Tuning device " << gpuIdx << ": " << device->info.name << endl;

    string outputFile;
    if(outputFileFromArg == "") {
      string dir = OpenCLTuner::defaultDirectory(true,homeDataDirOverride);
      outputFile = dir + "/" + OpenCLTuner::defaultFileName(device->info.name, nnXLen, nnYLen, modelInfo);
    }
    else {
      outputFile = outputFileFromArg;
    }

    OpenCLTuneParams initialParams;
    try {
      OpenCLTuneParams loadedParams = OpenCLTuneParams::load(outputFile);
      initialParams = loadedParams;
      cout << "Starting from existing parameters in: " + outputFile << endl;
    }
    catch(const StringError& e) {
      (void)e;
      cout << "File does not already exist or unable to parse parameters in: " + outputFile << endl;
      cout << "Starting fresh tuning, saving results to " << outputFile << endl;
    }

    OpenCLTuneParams results;
    OpenCLTuner::tune(
      initialParams,
      allDeviceInfos,
      devicesContext,
      gpuIdx,
      batchSize,
      nnXLen,
      nnYLen,
      testFP16Mode,
      testFP16StorageMode,
      testFP16ComputeMode,
      testFP16TensorCoresMode,
      modelInfo,
      full,
      winograd3x3TileSize,
      &logger,
      cout,
      verboseErrors,
      verboseTuner,
      results
    );

    OpenCLTuneParams::save(outputFile, results);
    cout << "Done, results saved to " << outputFile << endl;
  }

  return 0;

#elif defined(USE_MLX_BACKEND)

  // MLX (Apple GPU) tuner: searches the Winograd input/output transform grids
  // and writes the winning parameters to the same cache the backend reads at
  // model load. This is the deliberate "command tune" path; the auto-tune that
  // runs during normal model load stays coarse/fast. The OpenCL-only FP16
  // sub-knobs (storage/compute/tensorcores) have no MLX analog and are omitted.
  ConfigParser cfg;
  string modelFile;
  string outputFileFromArg;
  int nnXLen;
  int nnYLen;
  int batchSize;
  string testFP16Str;
  enabled_t testFP16Mode;
  bool full;
  try {
    KataGoCommandLine cmd("Perform Winograd transform tuning for the MLX (Apple GPU) backend.");
    cmd.addConfigFileArg(KataGoCommandLine::defaultGtpConfigFileName(),"gtp_example.cfg");
    cmd.addModelFileArg();

    TCLAP::ValueArg<string> outputFileArg("","output","Filename to output tuning configuration to (default: shared MLX tuner cache)",false,string(),"FILE");
    TCLAP::ValueArg<int> nnXLenArg("","xsize","Width of board to tune for",false,19,"INT");
    TCLAP::ValueArg<int> nnYLenArg("","ysize","Height of board to tune for",false,19,"INT");
    TCLAP::ValueArg<int> batchSizeArg("","batchsize","Batch size to tune for",false,8,"INT");
    TCLAP::ValueArg<string> testFP16Arg("","testFP16","Tune for FP16? true|false|auto (default auto = engine default, FP16)",false,"auto","BOOL_OR_AUTO");
    TCLAP::SwitchArg fullArg("","full","Sweep the wide candidate grid instead of the default coarse one");

    cmd.setShortUsageArgLimit();
    cmd.addOverrideConfigArg();

    cmd.add(outputFileArg);
    cmd.add(nnXLenArg);
    cmd.add(nnYLenArg);
    cmd.add(batchSizeArg);
    cmd.add(testFP16Arg);
    cmd.add(fullArg);
    cmd.parseArgs(args);

    modelFile = cmd.getModelFile();
    outputFileFromArg = outputFileArg.getValue();
    nnXLen = nnXLenArg.getValue();
    nnYLen = nnYLenArg.getValue();
    batchSize = batchSizeArg.getValue();
    testFP16Str = testFP16Arg.getValue();
    full = fullArg.getValue();

    if(!enabled_t::tryParse(testFP16Str,testFP16Mode)) {
      cerr << "Error: Could not parse -testFP16 as bool or auto: " << testFP16Str << endl;
      return 1;
    }

    cmd.getConfigAllowEmpty(cfg);
  }
  catch (TCLAP::ArgException &e) {
    cerr << "Error: " << e.error() << " for argument " << e.argId() << endl;
    return 1;
  }

  // The MLX GPU path runs FP16 unless explicitly disabled (useFP16Mode != False
  // in mlxbackend.cpp), so 'auto' tunes for FP16 - the precision the engine will
  // actually use, and the precision the cache filename is keyed on.
  const bool useFP16 = (testFP16Mode != enabled_t::False);

  string homeDataDirOverride = Setup::loadHomeDataDirOverride(cfg);

  const bool logToStdoutDefault = true;
  Logger logger(&cfg, logToStdoutDefault);

  logger.write("Loading model...");
  ModelDesc modelDesc;
  string expectedSha256 = "";
  ModelDesc::loadFromFileMaybeGZipped(modelFile, modelDesc, expectedSha256);

  // Same shape diagnostic the backend logs at load, so the tuned cache can be
  // correlated with the model's 3x3 conv shape mix.
  logger.write(MLXWinogradTuner::formatConv3x3Distribution(modelDesc));

  MLXWinogradTuner::ModelInfoForTuning modelInfo;
  modelInfo.trunkNumChannels = modelDesc.trunk.trunkNumChannels;
  modelInfo.modelVersion = modelDesc.modelVersion;
  {
    auto histograms = MLXWinogradTuner::buildConv3x3Histograms(modelDesc);
    modelInfo.conv3x3InputHistogram = std::move(histograms.first);
    modelInfo.conv3x3OutputHistogram = std::move(histograms.second);
  }

  // Chip-specific cache key, shared with the backend's model-load path so the
  // command writes exactly the file the backend reads; part of the filename key.
  const string gpuName = MLXWinogradTuner::detectGpuName();

  string outputFile;
  if(outputFileFromArg == "") {
    string dir = MLXWinogradTuner::defaultDirectory(true,homeDataDirOverride);
    outputFile = dir + "/" + MLXWinogradTuner::defaultFileName(
      gpuName, nnXLen, nnYLen, modelInfo.trunkNumChannels, modelInfo.modelVersion, useFP16);
  }
  else {
    outputFile = outputFileFromArg;
  }

  logger.write(string("MLX Winograd tuner starting (") + (full ? "full" : "coarse") +
               " sweep, " + (useFP16 ? "FP16" : "FP32") + ", batch " + Global::intToString(batchSize) + ")...");

  // reTune=true: a command tune always re-runs the search and overwrites the
  // cache, rather than short-circuiting on an existing file.
  MLXWinogradTuner::loadOrAutoTune(
    outputFile,
    homeDataDirOverride,
    gpuName,
    nnXLen,
    nnYLen,
    batchSize,
    modelInfo,
    &logger,
    /*full=*/full,
    /*reTune=*/true,
    /*useFP16=*/useFP16,
    /*seedOverride=*/nullptr
  );

  cout << "Done, results saved to " << outputFile << endl;

  return 0;

#else
  cout << "Currently this command only does anything for the OpenCL and MLX versions of KataGo" << endl;
  (void)args;
  return 0;
#endif
}
