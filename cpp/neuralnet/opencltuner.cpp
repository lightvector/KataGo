#ifdef USE_OPENCL_BACKEND

#include "../neuralnet/openclhelpers.h"
#include "../neuralnet/opencltuner.h"
#include "../neuralnet/openclkernels.h"
#include "../core/rand.h"
#include "../core/makedir.h"
#include "../dataio/homedata.h"

#include <cstring>

using namespace std;
using namespace OpenCLHelpers;

static map<string,int> readDescKeyValues(const string& fileName, const string& desc) {
  istringstream kvIn(desc);
  string kvChunk;
  map<string,int> keyValues;
  while(getline(kvIn,kvChunk,' '))
  {
    if(kvChunk.length() <= 0) continue;
    size_t equalsPos = kvChunk.find_first_of('=');
    if(equalsPos == string::npos) continue;
    string leftChunk = Global::trim(kvChunk.substr(0,equalsPos));
    string rightChunk = Global::trim(kvChunk.substr(equalsPos+1));
    if(leftChunk.length() == 0)
      throw IOError("OpenCLTuner readDescKeyValues: key value pair without key in: " + desc + " in file " + fileName);
    if(rightChunk.length() == 0)
      throw IOError("OpenCLTuner readDescKeyValues: key value pair without value in: " + desc + " in file " + fileName);
    if(keyValues.find(leftChunk) != keyValues.end())
      throw IOError("OpenCLTuner readDescKeyValues: duplicate key: " + leftChunk);
    int value;
    bool suc = Global::tryStringToInt(rightChunk, value);
    if(!suc)
      throw IOError("OpenCLTuner readDescKeyValues: could not parse value for key " + leftChunk + " in file " + fileName);

    keyValues[leftChunk] = value;
  }
  return keyValues;
}

static bool isMultipleOf(int x, int y) {
  return x % y == 0;
}

static int getInt(const map<string,int> map, const string& key, int defaultValue) {
  if(!contains(map,key))
    return defaultValue;
  return map_get(map,key);
}

string OpenCLTuneParams::XGemmDirectParams::desc() const {
  string s;
  s += "WGD=" + Global::intToString(WGD);
  s += " MDIMCD=" + Global::intToString(MDIMCD);
  s += " NDIMCD=" + Global::intToString(NDIMCD);
  s += " MDIMAD=" + Global::intToString(MDIMAD);
  s += " NDIMBD=" + Global::intToString(NDIMBD);
  s += " KWID=" + Global::intToString(KWID);
  s += " VWMD=" + Global::intToString(VWMD);
  s += " VWND=" + Global::intToString(VWND);
  s += " PADA=" + Global::intToString(PADA);
  s += " PADB=" + Global::intToString(PADB);
  return s;
}
string OpenCLTuneParams::XGemmDirectParams::compileOptions() const {
  string s;
  s += "-DWGD=" + Global::intToString(WGD);
  s += " -DMDIMCD=" + Global::intToString(MDIMCD);
  s += " -DNDIMCD=" + Global::intToString(NDIMCD);
  s += " -DMDIMAD=" + Global::intToString(MDIMAD);
  s += " -DNDIMBD=" + Global::intToString(NDIMBD);
  s += " -DKWID=" + Global::intToString(KWID);
  s += " -DVWMD=" + Global::intToString(VWMD);
  s += " -DVWND=" + Global::intToString(VWND);
  s += " -DPADA=" + Global::intToString(PADA);
  s += " -DPADB=" + Global::intToString(PADB);
  return s;
}
void OpenCLTuneParams::XGemmDirectParams::fillFromDesc(const string& fileName, const string& desc) {
  map<string,int> kvs = readDescKeyValues(fileName, desc);
  WGD = getInt(kvs,"WGD",WGD);
  MDIMCD = getInt(kvs,"MDIMCD",MDIMCD);
  NDIMCD = getInt(kvs,"NDIMCD",NDIMCD);
  MDIMAD = getInt(kvs,"MDIMAD",MDIMAD);
  NDIMBD = getInt(kvs,"NDIMBD",NDIMBD);
  KWID = getInt(kvs,"KWID",KWID);
  VWMD = getInt(kvs,"VWMD",VWMD);
  VWND = getInt(kvs,"VWND",VWND);
  PADA = getInt(kvs,"PADA",PADA);
  PADB = getInt(kvs,"PADB",PADB);
}
bool OpenCLTuneParams::XGemmDirectParams::isValid() const {
  if(WGD <= 0) return false;
  if(MDIMCD <= 0) return false;
  if(NDIMCD <= 0) return false;
  if(MDIMAD <= 0) return false;
  if(NDIMBD <= 0) return false;
  if(KWID <= 0) return false;
  if(VWMD <= 0) return false;
  if(VWND <= 0) return false;
  if(PADA < 0) return false;
  if(PADB < 0) return false;
  if(!isMultipleOf(WGD,KWID)) return false;
  if(!isMultipleOf(WGD,MDIMCD*VWMD)) return false;
  if(!isMultipleOf(WGD,NDIMCD*VWND)) return false;
  if(!isMultipleOf(WGD,MDIMAD*VWMD)) return false;
  if(!isMultipleOf(WGD,NDIMBD*VWND)) return false;
  if(!isMultipleOf(WGD,MDIMCD*NDIMCD/MDIMAD)) return false;
  if(!isMultipleOf(WGD,MDIMCD*NDIMCD/NDIMBD)) return false;
  return true;
}


string OpenCLTuneParams::Conv3x3Params::desc() const {
  string s;
  s += "INTILE_XSIZE=" + Global::intToString(INTILE_XSIZE);
  s += " INTILE_YSIZE=" + Global::intToString(INTILE_YSIZE);
  s += " OUTTILE_XSIZE=" + Global::intToString(OUTTILE_XSIZE);
  s += " OUTTILE_YSIZE=" + Global::intToString(OUTTILE_YSIZE);
  return s;
}
string OpenCLTuneParams::Conv3x3Params::compileOptions() const {
  string s;
  s += "-DINTILE_XSIZE=" + Global::intToString(INTILE_XSIZE);
  s += " -DINTILE_YSIZE=" + Global::intToString(INTILE_YSIZE);
  s += " -DOUTTILE_XSIZE=" + Global::intToString(OUTTILE_XSIZE);
  s += " -DOUTTILE_YSIZE=" + Global::intToString(OUTTILE_YSIZE);
  s += " -DCONV_XSIZE=3 -DCONV_YSIZE=3 -DINTILE_XOFFSET=(-1) -DINTILE_YOFFSET=(-1)";
  return s;
}
void OpenCLTuneParams::Conv3x3Params::fillFromDesc(const string& fileName, const string& desc) {
  map<string,int> kvs = readDescKeyValues(fileName, desc);
  INTILE_XSIZE = getInt(kvs,"INTILE_XSIZE",INTILE_XSIZE);
  INTILE_YSIZE = getInt(kvs,"INTILE_YSIZE",INTILE_YSIZE);
  OUTTILE_XSIZE = getInt(kvs,"OUTTILE_XSIZE",OUTTILE_XSIZE);
  OUTTILE_YSIZE = getInt(kvs,"OUTTILE_YSIZE",OUTTILE_YSIZE);
}
bool OpenCLTuneParams::Conv3x3Params::isValid() const {
  //Currently, the only supported winograd tile sizes
  if(INTILE_XSIZE == 4 && OUTTILE_XSIZE == 2 && INTILE_YSIZE == 4 && OUTTILE_YSIZE == 2)
    return true;
  return false;
}

bool OpenCLTuneParams::isValid() const {
  return xGemmDirect.isValid() && conv3x3.isValid();
}

bool OpenCLTuneParams::operator==(const OpenCLTuneParams& other) const {
  if(this == &other)
    return true;
  return std::memcmp(this,&other,sizeof(OpenCLTuneParams)) == 0;
}


void OpenCLTuneParams::save(const string& filename, const OpenCLTuneParams& config) {
  ofstream out(filename);
  if(out.fail())
    throw IOError("Could not create file: " + filename);
  out << "VERSION=1" << "\n";
  out << "#xGemmDirect" << "\n";
  out << config.xGemmDirect.desc() << "\n";
  out << "#conv3x3" << "\n";
  out << config.conv3x3.desc() << "\n";
  out.flush();
  out.close();
}

OpenCLTuneParams OpenCLTuneParams::load(const string& filename) {
  vector<string> lines = Global::readFileLines(filename, '\n');
  vector<string> filteredLines;
  for(size_t i = 0; i<lines.size(); i++) {
    string line = Global::stripComments(lines[i]);
    line = Global::trim(line);
    if(line.length() > 0)
      filteredLines.push_back(line);
  }
  if(filteredLines.size() <= 0)
    throw IOError("OpenCLTuneParams::load: no params in file " + filename);
  if(filteredLines[0] != "VERSION=1")
    throw IOError("OpenCLTuneParams::load: expected first line to be VERSION=1 in " + filename);
  if(filteredLines.size() != 3)
    throw IOError("OpenCLTuneParams::load: unexpected number of parameter lines in file " + filename);

  OpenCLTuneParams config;
  config.xGemmDirect.fillFromDesc(filename,filteredLines[1]);
  config.conv3x3.fillFromDesc(filename,filteredLines[2]);
  return config;
}



static cl_mem randomReadOnlyBuffer(const char* seed, cl_context context, int numFloats, double scale) {
  vector<float> buf(numFloats);
  Rand rand(seed);
  for(int i = 0; i<numFloats; i++)
    buf[i] = rand.nextDouble(scale);
  return createReadOnlyBuffer(context,buf);
}


template<typename T>
static void addConfigs(
  vector<OpenCLTuneParams>& configs,
  std::function<void(OpenCLTuneParams&, T value)> apply,
  const vector<T>& values
) {
  vector<OpenCLTuneParams> newCfgs;
  for(int i = 0; i<values.size(); i++) {
    for(int j = 0; j<configs.size(); j++) {
      OpenCLTuneParams cfg = configs[j];
      apply(cfg,values[i]);
      newCfgs.push_back(cfg);
    }
  }
  configs = newCfgs;
}

static void filterConfigs(
  vector<OpenCLTuneParams>& configs,
  std::function<bool(const OpenCLTuneParams&)> isValid
) {
  vector<OpenCLTuneParams> newCfgs;
  for(int j = 0; j<configs.size(); j++) {
    if(isValid(configs[j]))
      newCfgs.push_back(configs[j]);
  }
  configs = newCfgs;
}


#define SETTER(field) std::function<void(OpenCLTuneParams&, int value)>([](OpenCLTuneParams& p, int value){ p.field = value; })
#define ISVALID(field) std::function<bool(const OpenCLTuneParams&)>([](const OpenCLTuneParams& p){ return p.field.isValid(); })

void OpenCLTuner::tune(
  const OpenCLTuneParams& initialConfig,
  int gpuIdx,
  Logger* logger,
  int batchSize,
  int nnXLen,
  int nnYLen,
  const ModelDesc* model,
  bool full,
  ostream& out,
  std::function<void(const OpenCLTuneParams&)> handleBestSoFar
) {
  bool enableProfiling = true;
  DevicesContext devicesContext({gpuIdx}, logger, enableProfiling);
  const cl_context& context = devicesContext.context;
  const vector<cl_device_id>& deviceIdsToUse = devicesContext.deviceIdsToUse;

  OpenCLTuneParams untunedConfig = OpenCLTuneParams();
  OpenCLTuneParams currentConfig = initialConfig;

  //Tune xGemmDirect
  {
    out << "Tuning xGemmDirect for convolutions" << endl;

    vector<OpenCLTuneParams> configs;
    configs.push_back(initialConfig);
    if(full) {
      addConfigs(configs,SETTER(xGemmDirect.WGD),{8,16,32,64});
      addConfigs(configs,SETTER(xGemmDirect.MDIMCD),{8,16,32});
      addConfigs(configs,SETTER(xGemmDirect.NDIMCD),{8,16,32});
      addConfigs(configs,SETTER(xGemmDirect.MDIMAD),{8,16,32});
      addConfigs(configs,SETTER(xGemmDirect.NDIMBD),{8,16,32});
      addConfigs(configs,SETTER(xGemmDirect.KWID),{2,8,16});
      addConfigs(configs,SETTER(xGemmDirect.VWMD),{1,2,4,8});
      addConfigs(configs,SETTER(xGemmDirect.VWND),{1,2,4,8});
      // addConfigs(configs,SETTER(xGemmDirect.PADA),{1,0});
      // addConfigs(configs,SETTER(xGemmDirect.PADB),{1,0});
    }
    else {
      addConfigs(configs,SETTER(xGemmDirect.WGD),{8,16,32});
      addConfigs(configs,SETTER(xGemmDirect.MDIMCD),{8,16,32});
      addConfigs(configs,SETTER(xGemmDirect.NDIMCD),{8,16,32});
      addConfigs(configs,SETTER(xGemmDirect.MDIMAD),{8,16,32});
      addConfigs(configs,SETTER(xGemmDirect.NDIMBD),{8,16,32});
      addConfigs(configs,SETTER(xGemmDirect.KWID),{2,8});
      addConfigs(configs,SETTER(xGemmDirect.VWMD),{2,4});
      addConfigs(configs,SETTER(xGemmDirect.VWND),{2,4});
      // addConfigs(configs,SETTER(xGemmDirect.PADA),{1,0});
      // addConfigs(configs,SETTER(xGemmDirect.PADB),{1,0});
    }

    filterConfigs(configs,ISVALID(xGemmDirect));

    Rand rand;
    for(int i = configs.size()-1; i > 0; i--) {
      int j = rand.nextUInt(i+1);
      std::swap(configs[i],configs[j]);
    }

    //Make sure we have the vanilla default config
    //Make sure we have the initial config
    configs.insert(configs.begin(),untunedConfig);
    configs.insert(configs.begin(),initialConfig);

    out << "Testing " << configs.size() << " different configs" << endl;

    auto test = [&](OpenCLTuneParams& cfg, double& ret) {
      cl_int err;
      cl_program program = compileProgram("xgemmDirectProgram", context, deviceIdsToUse, OpenCLKernels::xgemmDirect, cfg.xGemmDirect.compileOptions());
      cl_kernel kernel = clCreateKernel(program, "XgemmDirectBatchedTT", &err);
      CHECK_ERR(err);

      int numTilesX = (nnXLen + cfg.conv3x3.OUTTILE_XSIZE - 1) / cfg.conv3x3.OUTTILE_XSIZE;
      int numTilesY = (nnYLen + cfg.conv3x3.OUTTILE_YSIZE - 1) / cfg.conv3x3.OUTTILE_YSIZE;
      int numTilesTotal = batchSize * numTilesX * numTilesY;

      int inTileXSize = cfg.conv3x3.INTILE_XSIZE;
      int inTileYSize = cfg.conv3x3.INTILE_YSIZE;
      int inTileXYSize = inTileXSize * inTileYSize;

      int maxChannels = model->maxConvChannels(3,3);
      maxChannels = std::max(model->trunk.trunkNumChannels,maxChannels);
      maxChannels = std::max(model->trunk.midNumChannels,maxChannels);
      maxChannels = std::max(model->trunk.regularNumChannels,maxChannels);
      maxChannels = std::max(model->trunk.gpoolNumChannels,maxChannels);

      int ioNumFloats = numTilesTotal * maxChannels * inTileXYSize;
      int filterNumFloats = maxChannels * maxChannels * inTileXYSize;
      cl_mem input = randomReadOnlyBuffer("tune3x3Input", context, ioNumFloats, 1.0);
      cl_mem filter = randomReadOnlyBuffer("tune3x3Filter", context, filterNumFloats, 1.0 / sqrt(maxChannels * 3 * 3));
      cl_mem output = createReadWriteBuffer(context, ioNumFloats);

      bool bad = false;
      double weightCounted = 0;
      double weightedTimeTaken = 0;

      //TODO need a reference implementation to compare error against!
      const int reps = 6;
      for(int i = 0; i<reps; i++) {
        int inChannels;
        int outChannels;
        double weight;
        switch(i) {
        //Weight 0 on first kernel call to warm up
        case 0: inChannels = model->trunk.trunkNumChannels; outChannels = model->trunk.midNumChannels; weight = 0; break;
        case 1: inChannels = model->trunk.trunkNumChannels; outChannels = model->trunk.midNumChannels; weight = 1; break;
        case 2: inChannels = model->trunk.midNumChannels; outChannels = model->trunk.trunkNumChannels; weight = 1; break;
        case 3: inChannels = model->trunk.trunkNumChannels; outChannels = model->trunk.regularNumChannels; weight = 0.2; break;
        case 4: inChannels = model->trunk.trunkNumChannels; outChannels = model->trunk.gpoolNumChannels; weight = 0.2; break;
        case 5: inChannels = maxChannels; outChannels = maxChannels; weight = 1; break;
        default: ASSERT_UNREACHABLE; break;
        }

        cl_event event;
        err = doBatchedXGemm_KM_KN_MN(
          kernel,
          devicesContext.commandQueues[0],
          cfg,
          outChannels, numTilesTotal, inChannels,
          filter, input, output,
          inTileXYSize,
          &event
        );
        if(err != 0) {
          bad = true;
          break;
        }

        err = clWaitForEvents(1, &event);
        CHECK_ERR(err);

        cl_ulong time_start, time_end;
        err = clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(time_start), &time_start, NULL); CHECK_ERR(err);
        err = clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(time_end), &time_end, NULL); CHECK_ERR(err);

        weightedTimeTaken += (time_end - time_start) * 1e-9 * weight;
        weightCounted += weight;

        clReleaseEvent(event);
      }

      clReleaseMemObject(input);
      clReleaseMemObject(filter);
      clReleaseMemObject(output);

      clReleaseKernel(kernel);
      clReleaseProgram(program);

      if(bad)
        return false;

      double kernelsPerSecond = weightCounted / weightedTimeTaken;
      ret = kernelsPerSecond;
      return true;
    };

    double bestKernelsPerSecond = 0.0;
    int numTested = 0;
    int numTestedRunnable = 0;
    int lastBestIdx = 0;
    for(int i = 0; i<configs.size(); i++) {
      double kernelsPerSecond;
      bool suc = test(configs[i],kernelsPerSecond);
      numTested++;
      if(suc) {
        numTestedRunnable++;
        if(kernelsPerSecond > bestKernelsPerSecond) {
          bestKernelsPerSecond = kernelsPerSecond;
          currentConfig = configs[i];
          out << "Tuning " << i << "/"  << configs.size() << " Calls/sec " << bestKernelsPerSecond << " " << currentConfig.xGemmDirect.desc() << endl;
          handleBestSoFar(currentConfig);
          lastBestIdx = i;
        }
      }
      if(i % 20 == 0 && i >= lastBestIdx+20)
        out << "Tuning " << i << "/" << configs.size() << " ..." << endl;
    }

  }

}

string OpenCLTuner::defaultDirectory(bool makeDir) {
  string dir = HomeData::getHomeDataDir(true);
  dir += "/opencltuning";
  if(makeDir)
    MakeDir::make(dir);
  return dir;
}

string OpenCLTuner::defaultFileName(int gpuIdx, int nnXLen, int nnYLen, const ModelDesc* model) {
  return Global::strprintf("tune_gpu%d_x%d_y%d_c%d_mv%d.txt", gpuIdx, nnXLen, nnYLen, model->trunk.trunkNumChannels,model->version);
}

OpenCLTuneParams OpenCLTuner::loadOrAutoTune(
  string openCLTunerFile,
  int gpuIdx,
  Logger* logger,
  int nnXLen,
  int nnYLen,
  const ModelDesc* model,
  bool full
) {
  if(openCLTunerFile != "") {
    if(logger != NULL)
      logger->write("Loading tuning parameters from: " + openCLTunerFile);
    return OpenCLTuneParams::load(openCLTunerFile);
  }

  string dir = OpenCLTuner::defaultDirectory(true);
  openCLTunerFile = dir + "/" + OpenCLTuner::defaultFileName(gpuIdx, nnXLen, nnYLen, model);

  try {
    OpenCLTuneParams loadedParams = OpenCLTuneParams::load(openCLTunerFile);
    if(!loadedParams.isValid())
      throw StringError("Loaded parmameters were not valid");
    if(logger != NULL)
      logger->write("Loaded tuning parameters from: " + openCLTunerFile);
    return loadedParams;
  }
  catch(const StringError& e) {
    if(logger != NULL) {
      logger->write("No existing tuning parameters found or parseable or valid at: " + openCLTunerFile);
      logger->write("Performing autotuning");
    }
    OpenCLTuneParams results;
    auto handleBestSoFar = [&results](const OpenCLTuneParams& bestSoFar) {
      results = bestSoFar;
    };

    OpenCLTuneParams initialParams;
    int batchSize = OpenCLTuner::DEFAULT_BATCH_SIZE;
    OpenCLTuner::tune(
      initialParams,
      gpuIdx,
      logger,
      batchSize,
      nnXLen,
      nnYLen,
      model,
      full,
      cerr,
      std::function<void(const OpenCLTuneParams&)>(handleBestSoFar)
    );

    OpenCLTuneParams::save(openCLTunerFile, results);
    if(logger != NULL)
      logger->write("Done tuning, saved results to " + openCLTunerFile);
    return results;
  };

}

#endif
