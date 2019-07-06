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
  s += " transLocalSize0=" + Global::intToString(transLocalSize0);
  s += " transLocalSize1=" + Global::intToString(transLocalSize1);
  s += " transLocalSize2=" + Global::intToString(transLocalSize2);
  s += " untransLocalSize0=" + Global::intToString(untransLocalSize0);
  s += " untransLocalSize1=" + Global::intToString(untransLocalSize1);
  s += " untransLocalSize2=" + Global::intToString(untransLocalSize2);
  return s;
}
string OpenCLTuneParams::Conv3x3Params::transDesc() const {
  string s;
  s += " transLocalSize0=" + Global::intToString(transLocalSize0);
  s += " transLocalSize1=" + Global::intToString(transLocalSize1);
  s += " transLocalSize2=" + Global::intToString(transLocalSize2);
  return s;
}
string OpenCLTuneParams::Conv3x3Params::untransDesc() const {
  string s;
  s += " untransLocalSize0=" + Global::intToString(untransLocalSize0);
  s += " untransLocalSize1=" + Global::intToString(untransLocalSize1);
  s += " untransLocalSize2=" + Global::intToString(untransLocalSize2);
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
  transLocalSize0 = getInt(kvs,"transLocalSize0",transLocalSize0);
  transLocalSize1 = getInt(kvs,"transLocalSize1",transLocalSize1);
  transLocalSize2 = getInt(kvs,"transLocalSize2",transLocalSize2);
  untransLocalSize0 = getInt(kvs,"untransLocalSize0",untransLocalSize0);
  untransLocalSize1 = getInt(kvs,"untransLocalSize1",untransLocalSize1);
  untransLocalSize2 = getInt(kvs,"untransLocalSize2",untransLocalSize2);
}
bool OpenCLTuneParams::Conv3x3Params::isValid() const {
  if(transLocalSize0 <= 0) return false;
  if(transLocalSize1 <= 0) return false;
  if(transLocalSize2 <= 0) return false;
  if(untransLocalSize0 <= 0) return false;
  if(untransLocalSize1 <= 0) return false;
  if(untransLocalSize2 <= 0) return false;

  if(transLocalSize0 * transLocalSize1 * transLocalSize2 > 1024) return false;
  if(untransLocalSize0 * untransLocalSize1 * untransLocalSize2 > 1024) return false;

  //Currently, the only supported winograd tile sizes
  if(INTILE_XSIZE == 4 && OUTTILE_XSIZE == 2 && INTILE_YSIZE == 4 && OUTTILE_YSIZE == 2)
    return true;
  return false;
}

string OpenCLTuneParams::GPoolParams::desc() const {
  string s;
  s += "XYSTRIDE=" + Global::intToString(XYSTRIDE);
  s += " CHANNELSTRIDE=" + Global::intToString(CHANNELSTRIDE);
  s += " BATCHSTRIDE=" + Global::intToString(BATCHSTRIDE);
  return s;
}
string OpenCLTuneParams::GPoolParams::compileOptions() const {
  string s;
  s += "-DXYSTRIDE=" + Global::intToString(XYSTRIDE);
  s += " -DCHANNELSTRIDE=" + Global::intToString(CHANNELSTRIDE);
  s += " -DBATCHSTRIDE=" + Global::intToString(BATCHSTRIDE);
  s += " -DLOCALSIZE_TOTAL=" + Global::intToString(XYSTRIDE * CHANNELSTRIDE * BATCHSTRIDE);
  return s;
}
void OpenCLTuneParams::GPoolParams::fillFromDesc(const string& fileName, const string& desc) {
  map<string,int> kvs = readDescKeyValues(fileName, desc);
  XYSTRIDE = getInt(kvs,"XYSTRIDE",XYSTRIDE);
  CHANNELSTRIDE = getInt(kvs,"CHANNELSTRIDE",CHANNELSTRIDE);
  BATCHSTRIDE = getInt(kvs,"BATCHSTRIDE",BATCHSTRIDE);
}
bool OpenCLTuneParams::GPoolParams::isValid() const {
  if(XYSTRIDE <= 0) return false;
  if(CHANNELSTRIDE <= 0) return false;
  if(BATCHSTRIDE <= 0) return false;

  //Must be power of 2
  if((XYSTRIDE & (XYSTRIDE-1)) != 0) return false;

  if(XYSTRIDE * CHANNELSTRIDE * BATCHSTRIDE > 1024) return false;

  return true;
}

string OpenCLTuneParams::TransposeParams::desc() const {
  string s;
  s += "TILEDIM=" + Global::intToString(TILEDIM);
  s += " TILESTRIDE=" + Global::intToString(TILESTRIDE);
  s += " NCSTRIDE=" + Global::intToString(NCSTRIDE);
  return s;
}
string OpenCLTuneParams::TransposeParams::compileOptions() const {
  string s;
  s += "-DTILEDIM=" + Global::intToString(TILEDIM);
  s += " -DTILESTRIDE=" + Global::intToString(TILESTRIDE);
  s += " -DLOCALSIZE=" + Global::intToString(TILEDIM * (TILEDIM+1) * NCSTRIDE);
  return s;
}
void OpenCLTuneParams::TransposeParams::fillFromDesc(const string& fileName, const string& desc) {
  map<string,int> kvs = readDescKeyValues(fileName, desc);
  TILEDIM = getInt(kvs,"TILEDIM",TILEDIM);
  TILESTRIDE = getInt(kvs,"TILESTRIDE",TILESTRIDE);
  NCSTRIDE = getInt(kvs,"NCSTRIDE",NCSTRIDE);
}
bool OpenCLTuneParams::TransposeParams::isValid() const {
  if(TILEDIM <= 0) return false;
  if(TILESTRIDE <= 0) return false;
  if(NCSTRIDE <= 0) return false;

  if(!isMultipleOf(TILEDIM,TILESTRIDE)) return false;
  if(TILEDIM * TILESTRIDE * NCSTRIDE > 1024) return false;

  //Currently, the kernel actually doesn't support other values
  if(NCSTRIDE != 1)
    return false;

  return true;
}


bool OpenCLTuneParams::isValid() const {
  return xGemmDirect.isValid() && conv3x3.isValid() && gPool.isValid() && transpose.isValid();
}

bool OpenCLTuneParams::operator==(const OpenCLTuneParams& other) const {
  if(this == &other)
    return true;
  return std::memcmp(this,&other,sizeof(OpenCLTuneParams)) == 0;
}


static const char* TUNEPARAMS_VERSION_LINE = "VERSION=4";
void OpenCLTuneParams::save(const string& filename, const OpenCLTuneParams& config) {
  ofstream out(filename);
  if(out.fail())
    throw IOError("Could not create file: " + filename);
  out << TUNEPARAMS_VERSION_LINE << "\n";
  out << "#xGemmDirect" << "\n";
  out << config.xGemmDirect.desc() << "\n";
  out << "#conv3x3" << "\n";
  out << config.conv3x3.desc() << "\n";
  out << "#gPool" << "\n";
  out << config.gPool.desc() << "\n";
  out << "#transpose" << "\n";
  out << config.transpose.desc() << "\n";
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
  if(filteredLines[0] != TUNEPARAMS_VERSION_LINE)
    throw IOError("OpenCLTuneParams::load: expected first line to be " + string(TUNEPARAMS_VERSION_LINE) + " in " + filename);

  if(filteredLines.size() != 5)
    throw IOError("OpenCLTuneParams::load: unexpected number of parameter lines in file " + filename);

  OpenCLTuneParams config;
  config.xGemmDirect.fillFromDesc(filename,filteredLines[1]);
  config.conv3x3.fillFromDesc(filename,filteredLines[2]);
  config.gPool.fillFromDesc(filename,filteredLines[3]);
  config.transpose.fillFromDesc(filename,filteredLines[4]);
  return config;
}

static cl_mem constantReadOnlyBuffer(cl_context context, int numFloats, float constant) {
  vector<float> buf(numFloats);
  for(int i = 0; i<numFloats; i++)
    buf[i] = constant;
  return createReadOnlyBuffer(context,buf);
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

static void shuffleConfigs(
  vector<OpenCLTuneParams>& configs
) {
  Rand rand;
  for(int i = configs.size()-1; i > 0; i--) {
    int j = rand.nextUInt(i+1);
    std::swap(configs[i],configs[j]);
  }
}

struct OpenCLTuneAccums {
  bool bad = false;
  cl_int badErr = 0;
  double weightCounted = 0;
  double weightedTimeTaken = 0;

  void countResultAndFreeEvent(cl_int err, cl_event event, double weight) {
    if(err != 0) {
      if(!bad) {
        bad = true;
        badErr = err;
      }
      return;
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

};

static void testAllConfigs(
  const vector<OpenCLTuneParams>& configsToTest,
  OpenCLTuneParams& currentConfig,
  OpenCLTuneParams referenceConfig,
  ostream& out,
  std::function<string(const OpenCLTuneParams&)> getDesc,
  std::function<OpenCLTuneAccums(const OpenCLTuneParams& cfg, vector<float>& ret)> testConfig,
  std::function<void(const OpenCLTuneParams&)> handleBestSoFar
) {
  vector<OpenCLTuneParams> configs = configsToTest;

  //Insert the reference configuration first
  configs.insert(configs.begin(),referenceConfig);

  double bestKernelsPerSecond = 0.0;
  int lastBestIdx = 0;
  bool anythingGoodYet = false;
  int numTested = 0;
  int numTestedRunnable = 0;

  vector<float> referenceRet;
  vector<float> ret;

  out << "Testing " << configs.size() << " different configs" << endl;
  for(int i = 0; i<configs.size(); i++) {
    OpenCLTuneAccums accums = testConfig(configs[i],ret);

    numTested++;
    if(accums.bad) {
      if(i == 0)
        out << "WARNING: Reference implementation failed: " << getErrorMessage(accums.badErr) << endl;
    }
    else {
      if(!anythingGoodYet) {
        //Just use the first thing that worked as the reference
        //Unless something has gone really weird, this should be the reference implementation
        referenceRet = ret;
        anythingGoodYet = true;
      }

      numTestedRunnable++;

      double squerr = 0.0;
      if(referenceRet.size() != ret.size())
        squerr = std::numeric_limits<double>::infinity();
      else {
        for(int j = 0; j<referenceRet.size(); j++) {
          if(!isfinite(referenceRet[j]) || !isfinite(ret[j]))
            squerr = std::numeric_limits<double>::infinity();
          else {
            double diff = (double)referenceRet[j] - (double)ret[j];
            squerr += diff * diff;
          }
        }
      }

      double kernelsPerSecond = accums.weightCounted / accums.weightedTimeTaken;

      if(kernelsPerSecond > bestKernelsPerSecond) {
        bestKernelsPerSecond = kernelsPerSecond;
        currentConfig = configs[i];
        out << "Tuning " << i << "/"  << configs.size()
            << (i == 0 ? " (reference)" : "")
            << " Calls/sec " << bestKernelsPerSecond
            << " L2Error " << squerr
            << " " << getDesc(currentConfig) << endl;
        handleBestSoFar(currentConfig);
        lastBestIdx = i;
      }
    }
    if(i % 20 == 0 && i >= lastBestIdx+10)
      out << "Tuning " << i << "/" << configs.size() << " ..." << endl;
  }
  if(!anythingGoodYet)
    out << "ERROR: Could not find any configuration that worked" << endl;
}

#define SETTER(field) std::function<void(OpenCLTuneParams&, int value)>([](OpenCLTuneParams& p, int value){ p.field = value; })
#define ISVALID(field) std::function<bool(const OpenCLTuneParams&)>([](const OpenCLTuneParams& p){ return p.field.isValid(); })

void OpenCLTuner::tune(
  const OpenCLTuneParams& initialConfig,
  DevicesContext& devicesContext,
  int gpuIdx,
  int batchSize,
  int nnXLen,
  int nnYLen,
  const ModelDesc* model,
  bool full,
  ostream& out,
  std::function<void(const OpenCLTuneParams&)> handleBestSoFar
) {
  const InitializedDevice& device = devicesContext.findGpuExn(gpuIdx);
  const cl_context& context = devicesContext.context;
  cl_command_queue commandQueue = device.commandQueue;
  const vector<cl_device_id>& deviceIdsToUse = { device.info.deviceId };

  const OpenCLTuneParams untunedConfig = OpenCLTuneParams();
  OpenCLTuneParams currentConfig = initialConfig;
  if(!currentConfig.isValid()) {
    out << "Loaded a config but the config was invalid, starting tuning from basic config" << endl;
    currentConfig = untunedConfig;
  }

  //=======================================================================================
  //Tune convolution transform
  {
    out << "------------------------------------------------------" << endl;
    out << "Tuning winograd transform for convolutions" << endl;

    vector<OpenCLTuneParams> configs;
    configs.push_back(currentConfig);
    if(full) {
      addConfigs(configs,SETTER(conv3x3.transLocalSize0),{1,2,4,8,16,32,64});
      addConfigs(configs,SETTER(conv3x3.transLocalSize1),{1,2,4,8,16,32,64});
      addConfigs(configs,SETTER(conv3x3.transLocalSize2),{1,2,4,8,16,32,64});
    }
    else {
      addConfigs(configs,SETTER(conv3x3.transLocalSize0),{1,2,4,8,16,32});
      addConfigs(configs,SETTER(conv3x3.transLocalSize1),{1,2,4,8,16,32});
      addConfigs(configs,SETTER(conv3x3.transLocalSize2),{1,2,4,8,16,32});
    }

    filterConfigs(configs,ISVALID(conv3x3));
    shuffleConfigs(configs);
    configs.insert(configs.begin(),currentConfig);

    OpenCLTuneParams referenceConfig = currentConfig;
    referenceConfig.conv3x3.transLocalSize0 = untunedConfig.conv3x3.transLocalSize0;
    referenceConfig.conv3x3.transLocalSize1 = untunedConfig.conv3x3.transLocalSize1;
    referenceConfig.conv3x3.transLocalSize2 = untunedConfig.conv3x3.transLocalSize2;

    auto getDesc = [](const OpenCLTuneParams& cfg) { return cfg.conv3x3.transDesc(); };

    auto test = [&](const OpenCLTuneParams& cfg, vector<float>& ret) {
      cl_int err;
      cl_program program = compileProgram(
        "winogradConv3x3NCHWProgram", context, deviceIdsToUse, OpenCLKernels::winogradConvNCHW,
        cfg.conv3x3.compileOptions()
      );
      cl_kernel kernel = clCreateKernel(program, "transform", &err);
      CHECK_ERR(err);

      int numTilesX = (nnXLen + cfg.conv3x3.OUTTILE_XSIZE - 1) / cfg.conv3x3.OUTTILE_XSIZE;
      int numTilesY = (nnYLen + cfg.conv3x3.OUTTILE_YSIZE - 1) / cfg.conv3x3.OUTTILE_YSIZE;
      int numTilesTotal = batchSize * numTilesX * numTilesY;

      int inTileXSize = cfg.conv3x3.INTILE_XSIZE;
      int inTileYSize = cfg.conv3x3.INTILE_YSIZE;

      int maxChannels = model->maxConvChannels(3,3);
      maxChannels = std::max(model->trunk.trunkNumChannels,maxChannels);
      maxChannels = std::max(model->trunk.midNumChannels,maxChannels);
      maxChannels = std::max(model->trunk.regularNumChannels,maxChannels);
      maxChannels = std::max(model->trunk.gpoolNumChannels,maxChannels);

      int inputNumFloats = batchSize * nnXLen * nnYLen * maxChannels;
      int outputNumFloats = numTilesTotal * maxChannels * inTileXSize * inTileYSize;
      cl_mem input = randomReadOnlyBuffer("tune3x3TransInput", context, inputNumFloats, 1.0);
      cl_mem output = createReadWriteBuffer(context, outputNumFloats);

      OpenCLTuneAccums accums;
      const int reps = 10;
      for(int i = 0; i<reps; i++) {
        int inChannels;
        double weight;
        switch(i) {
        //Weight 0 on first kernel call to warm up
        case 0: inChannels = model->trunk.trunkNumChannels; weight = 0; break;
        case 1: inChannels = model->trunk.trunkNumChannels; weight = 1; break;
        case 2: inChannels = model->trunk.midNumChannels; weight = 1; break;
        case 3: inChannels = maxChannels; weight = 1; break;
        case 4: inChannels = model->trunk.trunkNumChannels; weight = 1; break;
        case 5: inChannels = model->trunk.midNumChannels; weight = 1; break;
        case 6: inChannels = maxChannels; weight = 1; break;
        case 7: inChannels = model->trunk.trunkNumChannels; weight = 1; break;
        case 8: inChannels = model->trunk.midNumChannels; weight = 1; break;
        case 9: inChannels = maxChannels; weight = 1; break;
        default: ASSERT_UNREACHABLE; break;
        }

        cl_event event;
        err = doWinogradTransform(
          kernel,
          commandQueue,
          cfg,
          input,output,
          batchSize,nnXLen,nnYLen,
          numTilesX,numTilesY,
          inChannels,
          &event
        );

        accums.countResultAndFreeEvent(err,event,weight);
        if(accums.bad)
          break;
      }

      if(accums.bad)
        ret.assign(outputNumFloats,0.0);
      else
        blockingReadBuffer(commandQueue, output, outputNumFloats, ret);

      clReleaseMemObject(input);
      clReleaseMemObject(output);

      clReleaseKernel(kernel);
      clReleaseProgram(program);

      return accums;
    };

    testAllConfigs(
      configs,
      currentConfig,
      referenceConfig,
      out,
      std::function<string(const OpenCLTuneParams& cfg)>(getDesc),
      std::function<OpenCLTuneAccums(const OpenCLTuneParams& cfg, vector<float>& ret)>(test),
      handleBestSoFar
    );

  }

  //=======================================================================================
  //Tune convolution untransform
  {
    out << "------------------------------------------------------" << endl;
    out << "Tuning winograd untransform for convolutions" << endl;

    vector<OpenCLTuneParams> configs;
    configs.push_back(currentConfig);
    if(full) {
      addConfigs(configs,SETTER(conv3x3.untransLocalSize0),{1,2,4,8,16,32,64});
      addConfigs(configs,SETTER(conv3x3.untransLocalSize1),{1,2,4,8,16,32,64});
      addConfigs(configs,SETTER(conv3x3.untransLocalSize2),{1,2,4,8,16,32,64});
    }
    else {
      addConfigs(configs,SETTER(conv3x3.untransLocalSize0),{1,2,4,8,16,32});
      addConfigs(configs,SETTER(conv3x3.untransLocalSize1),{1,2,4,8,16,32});
      addConfigs(configs,SETTER(conv3x3.untransLocalSize2),{1,2,4,8,16,32});
    }

    filterConfigs(configs,ISVALID(conv3x3));
    shuffleConfigs(configs);
    configs.insert(configs.begin(),currentConfig);

    OpenCLTuneParams referenceConfig = currentConfig;
    referenceConfig.conv3x3.untransLocalSize0 = untunedConfig.conv3x3.untransLocalSize0;
    referenceConfig.conv3x3.untransLocalSize1 = untunedConfig.conv3x3.untransLocalSize1;
    referenceConfig.conv3x3.untransLocalSize2 = untunedConfig.conv3x3.untransLocalSize2;

    auto getDesc = [](const OpenCLTuneParams& cfg) { return cfg.conv3x3.untransDesc(); };

    auto test = [&](const OpenCLTuneParams& cfg, vector<float>& ret) {
      cl_int err;
      cl_program program = compileProgram(
        "winogradConv3x3NCHWProgram", context, deviceIdsToUse, OpenCLKernels::winogradConvNCHW,
        cfg.conv3x3.compileOptions()
      );
      cl_kernel kernel = clCreateKernel(program, "untransform", &err);
      CHECK_ERR(err);

      int numTilesX = (nnXLen + cfg.conv3x3.OUTTILE_XSIZE - 1) / cfg.conv3x3.OUTTILE_XSIZE;
      int numTilesY = (nnYLen + cfg.conv3x3.OUTTILE_YSIZE - 1) / cfg.conv3x3.OUTTILE_YSIZE;
      int numTilesTotal = batchSize * numTilesX * numTilesY;

      int inTileXSize = cfg.conv3x3.INTILE_XSIZE;
      int inTileYSize = cfg.conv3x3.INTILE_YSIZE;

      int maxChannels = model->maxConvChannels(3,3);
      maxChannels = std::max(model->trunk.trunkNumChannels,maxChannels);
      maxChannels = std::max(model->trunk.midNumChannels,maxChannels);
      maxChannels = std::max(model->trunk.regularNumChannels,maxChannels);
      maxChannels = std::max(model->trunk.gpoolNumChannels,maxChannels);

      int inputNumFloats = numTilesTotal * maxChannels * inTileXSize * inTileYSize;
      int outputNumFloats = batchSize * nnXLen * nnYLen * maxChannels;
      cl_mem input = randomReadOnlyBuffer("tune3x3UntransInput", context, inputNumFloats, 1.0);
      cl_mem output = createReadWriteBuffer(context, outputNumFloats);

      OpenCLTuneAccums accums;
      const int reps = 10;
      for(int i = 0; i<reps; i++) {
        int outChannels;
        double weight;
        switch(i) {
        //Weight 0 on first kernel call to warm up
        case 0: outChannels = model->trunk.trunkNumChannels; weight = 0; break;
        case 1: outChannels = model->trunk.trunkNumChannels; weight = 1; break;
        case 2: outChannels = model->trunk.midNumChannels; weight = 1; break;
        case 3: outChannels = maxChannels; weight = 1; break;
        case 4: outChannels = model->trunk.trunkNumChannels; weight = 1; break;
        case 5: outChannels = model->trunk.midNumChannels; weight = 1; break;
        case 6: outChannels = maxChannels; weight = 1; break;
        case 7: outChannels = model->trunk.trunkNumChannels; weight = 1; break;
        case 8: outChannels = model->trunk.midNumChannels; weight = 1; break;
        case 9: outChannels = maxChannels; weight = 1; break;
        default: ASSERT_UNREACHABLE; break;
        }

        cl_event event;
        err = doWinogradUntransform(
          kernel,
          commandQueue,
          cfg,
          input,output,
          batchSize,nnXLen,nnYLen,
          numTilesX,numTilesY,
          outChannels,
          &event
        );

        accums.countResultAndFreeEvent(err,event,weight);
        if(accums.bad)
          break;
      }

      if(accums.bad)
        ret.assign(outputNumFloats,0.0);
      else
        blockingReadBuffer(commandQueue, output, outputNumFloats, ret);

      clReleaseMemObject(input);
      clReleaseMemObject(output);

      clReleaseKernel(kernel);
      clReleaseProgram(program);

      return accums;
    };

    testAllConfigs(
      configs,
      currentConfig,
      referenceConfig,
      out,
      std::function<string(const OpenCLTuneParams& cfg)>(getDesc),
      std::function<OpenCLTuneAccums(const OpenCLTuneParams& cfg, vector<float>& ret)>(test),
      handleBestSoFar
    );

  }

  //=======================================================================================
  //Tune global pooling strides
  {
    out << "------------------------------------------------------" << endl;
    out << "Tuning global pooling strides" << endl;

    vector<OpenCLTuneParams> configs;
    configs.push_back(currentConfig);

    auto powersOfTwoUpTo = [](int n) {
      vector<int> vec;
      for(int i = 1; i <= n; i *= 2)
        vec.push_back(i);
      return vec;
    };

    int numChannels = model->trunk.gpoolNumChannels;
    if(full) {
      addConfigs(configs,SETTER(gPool.XYSTRIDE),{1,2,4,8,16,32,64});
      addConfigs(configs,SETTER(gPool.CHANNELSTRIDE),powersOfTwoUpTo(std::min(64,numChannels)));
      addConfigs(configs,SETTER(gPool.BATCHSTRIDE),powersOfTwoUpTo(std::min(4,batchSize)));
    }
    else {
      addConfigs(configs,SETTER(gPool.XYSTRIDE),{1,2,4,8,16,32});
      addConfigs(configs,SETTER(gPool.CHANNELSTRIDE),powersOfTwoUpTo(std::min(32,numChannels)));
      addConfigs(configs,SETTER(gPool.BATCHSTRIDE),powersOfTwoUpTo(std::min(4,batchSize)));
    }

    filterConfigs(configs,ISVALID(gPool));
    shuffleConfigs(configs);
    configs.insert(configs.begin(),currentConfig);

    OpenCLTuneParams referenceConfig = currentConfig;
    referenceConfig.gPool.XYSTRIDE = untunedConfig.gPool.XYSTRIDE;
    referenceConfig.gPool.CHANNELSTRIDE = untunedConfig.gPool.CHANNELSTRIDE;
    referenceConfig.gPool.BATCHSTRIDE = untunedConfig.gPool.BATCHSTRIDE;

    auto getDesc = [](const OpenCLTuneParams& cfg) { return cfg.gPool.desc(); };

    auto test = [&](const OpenCLTuneParams& cfg, vector<float>& ret) {
      cl_int err;
      cl_program program = compileProgram(
        "gPoolChannelsNCHWProgram", context, deviceIdsToUse, OpenCLKernels::gPoolChannelsNCHW,
        cfg.gPool.compileOptions()
      );
      cl_kernel kernel = clCreateKernel(program, "gPoolChannelsNCHW", &err);
      CHECK_ERR(err);

      int inputNumFloats = batchSize * nnXLen * nnYLen * numChannels;
      int outputNumFloats = batchSize * numChannels * 3;

      cl_mem input = randomReadOnlyBuffer("tuneGPoolInput", context, inputNumFloats, 1.0);
      cl_mem maskSum = constantReadOnlyBuffer(context, batchSize, (float)(nnXLen*nnYLen));
      cl_mem output = createReadWriteBuffer(context, outputNumFloats);

      OpenCLTuneAccums accums;
      const int reps = 20;
      for(int i = 0; i<reps; i++) {
        double weight;
        switch(i) {
        //Weight 0 on first kernel call to warm up
        case 0: weight = 0; break;
        default: weight = 1; break;
        }

        cl_event event;
        err = performGPool(
          kernel,
          commandQueue,
          cfg,
          batchSize, numChannels, nnXLen*nnYLen,
          input,output,maskSum,
          &event
        );

        accums.countResultAndFreeEvent(err,event,weight);
        if(accums.bad)
          break;
      }

      if(accums.bad)
        ret.assign(outputNumFloats,0.0);
      else
        blockingReadBuffer(commandQueue, output, outputNumFloats, ret);

      clReleaseMemObject(input);
      clReleaseMemObject(maskSum);
      clReleaseMemObject(output);

      clReleaseKernel(kernel);
      clReleaseProgram(program);

      return accums;
    };

    testAllConfigs(
      configs,
      currentConfig,
      referenceConfig,
      out,
      std::function<string(const OpenCLTuneParams& cfg)>(getDesc),
      std::function<OpenCLTuneAccums(const OpenCLTuneParams& cfg, vector<float>& ret)>(test),
      handleBestSoFar
    );

  }

  //=======================================================================================
  //Tune transpose strides
  {
    out << "------------------------------------------------------" << endl;
    out << "Tuning transpose strides" << endl;

    vector<OpenCLTuneParams> configs;
    configs.push_back(currentConfig);

    int numChannels = model->numInputChannels;
    if(full) {
      addConfigs(configs,SETTER(transpose.TILEDIM),{1,2,4,8,16,32,64});
      addConfigs(configs,SETTER(transpose.TILESTRIDE),{1,2,4,8,16,32,64});
      addConfigs(configs,SETTER(transpose.NCSTRIDE),{1});
    }
    else {
      addConfigs(configs,SETTER(transpose.TILEDIM),{1,2,4,8,16,32});
      addConfigs(configs,SETTER(transpose.TILESTRIDE),{1,2,4,8,16,32});
      addConfigs(configs,SETTER(transpose.NCSTRIDE),{1});
    }

    filterConfigs(configs,ISVALID(transpose));
    shuffleConfigs(configs);
    configs.insert(configs.begin(),currentConfig);

    OpenCLTuneParams referenceConfig = currentConfig;
    referenceConfig.transpose.TILEDIM = untunedConfig.transpose.TILEDIM;
    referenceConfig.transpose.TILESTRIDE = untunedConfig.transpose.TILESTRIDE;
    referenceConfig.transpose.NCSTRIDE = untunedConfig.transpose.NCSTRIDE;

    auto getDesc = [](const OpenCLTuneParams& cfg) { return cfg.transpose.desc(); };

    auto test = [&](const OpenCLTuneParams& cfg, vector<float>& ret) {
      cl_int err;
      cl_program program = compileProgram(
        "transposeNCHWProgram", context, deviceIdsToUse, OpenCLKernels::transposeNCHW,
        cfg.transpose.compileOptions()
      );
      cl_kernel kernel = clCreateKernel(program, "transposeNCHW", &err);
      CHECK_ERR(err);

      int numFloats = batchSize * nnXLen * nnYLen * numChannels;
      int outputNumFloats = numFloats;

      cl_mem input = randomReadOnlyBuffer("tuneTransposeInput", context, numFloats, 1.0);
      cl_mem output = createReadWriteBuffer(context, numFloats);

      OpenCLTuneAccums accums;
      const int reps = 15;
      for(int i = 0; i<reps; i++) {
        double weight;
        switch(i) {
        //Weight 0 on first kernel call to warm up
        case 0: weight = 0; break;
        default: weight = 1; break;
        }

        cl_event event;
        err = transposeNCHW(
          kernel,
          commandQueue,
          cfg,
          batchSize, numChannels, nnXLen, nnYLen,
          input, output,
          &event
        );

        accums.countResultAndFreeEvent(err,event,weight);
        if(accums.bad)
          break;
      }

      if(accums.bad)
        ret.assign(outputNumFloats,0.0);
      else
        blockingReadBuffer(commandQueue, output, outputNumFloats, ret);

      clReleaseMemObject(input);
      clReleaseMemObject(output);

      clReleaseKernel(kernel);
      clReleaseProgram(program);

      return accums;
    };

    testAllConfigs(
      configs,
      currentConfig,
      referenceConfig,
      out,
      std::function<string(const OpenCLTuneParams& cfg)>(getDesc),
      std::function<OpenCLTuneAccums(const OpenCLTuneParams& cfg, vector<float>& ret)>(test),
      handleBestSoFar
    );

  }


  //=======================================================================================
  //Tune xGemmDirect
  {
    out << "------------------------------------------------------" << endl;
    out << "Tuning xGemmDirect for convolutions" << endl;

    vector<OpenCLTuneParams> configs;
    configs.push_back(currentConfig);
    if(full) {
      addConfigs(configs,SETTER(xGemmDirect.WGD),{8,16,32,64});
      addConfigs(configs,SETTER(xGemmDirect.MDIMCD),{8,16,32});
      addConfigs(configs,SETTER(xGemmDirect.NDIMCD),{8,16,32});
      addConfigs(configs,SETTER(xGemmDirect.MDIMAD),{8,16,32});
      addConfigs(configs,SETTER(xGemmDirect.NDIMBD),{8,16,32});
      addConfigs(configs,SETTER(xGemmDirect.KWID),{2,8,16});
      addConfigs(configs,SETTER(xGemmDirect.VWMD),{1,2,4,8});
      addConfigs(configs,SETTER(xGemmDirect.VWND),{1,2,4,8});
      addConfigs(configs,SETTER(xGemmDirect.PADA),{1});
      addConfigs(configs,SETTER(xGemmDirect.PADB),{1});
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
      addConfigs(configs,SETTER(xGemmDirect.PADA),{1});
      addConfigs(configs,SETTER(xGemmDirect.PADB),{1});
    }

    filterConfigs(configs,ISVALID(xGemmDirect));
    shuffleConfigs(configs);

    OpenCLTuneParams referenceConfig = currentConfig;
    referenceConfig.xGemmDirect.WGD = untunedConfig.xGemmDirect.WGD;
    referenceConfig.xGemmDirect.MDIMCD = untunedConfig.xGemmDirect.MDIMCD;
    referenceConfig.xGemmDirect.NDIMCD = untunedConfig.xGemmDirect.NDIMCD;
    referenceConfig.xGemmDirect.MDIMAD = untunedConfig.xGemmDirect.MDIMAD;
    referenceConfig.xGemmDirect.NDIMBD = untunedConfig.xGemmDirect.NDIMBD;
    referenceConfig.xGemmDirect.KWID = untunedConfig.xGemmDirect.KWID;
    referenceConfig.xGemmDirect.VWMD = untunedConfig.xGemmDirect.VWMD;
    referenceConfig.xGemmDirect.VWND = untunedConfig.xGemmDirect.VWND;
    referenceConfig.xGemmDirect.PADA = untunedConfig.xGemmDirect.PADA;
    referenceConfig.xGemmDirect.PADB = untunedConfig.xGemmDirect.PADB;
    OpenCLTuneParams slightlyTunedConfig = referenceConfig;
    slightlyTunedConfig.xGemmDirect.MDIMCD = 8;
    slightlyTunedConfig.xGemmDirect.NDIMCD = 8;
    slightlyTunedConfig.xGemmDirect.MDIMAD = 8;
    slightlyTunedConfig.xGemmDirect.NDIMBD = 8;
    OpenCLTuneParams slightlyTunedConfig2 = slightlyTunedConfig;
    slightlyTunedConfig2.xGemmDirect.WGD = 16;

    configs.insert(configs.begin(),slightlyTunedConfig2);
    configs.insert(configs.begin(),slightlyTunedConfig);
    configs.insert(configs.begin(),currentConfig);

    auto getDesc = [](const OpenCLTuneParams& cfg) { return cfg.xGemmDirect.desc(); };

    auto test = [&](const OpenCLTuneParams& cfg, vector<float>& ret) {
      cl_int err;
      cl_program program = compileProgram("xgemmDirectProgram", context, deviceIdsToUse, OpenCLKernels::xgemmDirect, cfg.xGemmDirect.compileOptions());
      cl_kernel kernel = clCreateKernel(program, "XgemmDirectBatchedNN", &err);
      CHECK_ERR(err);

      int numTilesX = (nnXLen + cfg.conv3x3.OUTTILE_XSIZE - 1) / cfg.conv3x3.OUTTILE_XSIZE;
      int numTilesY = (nnYLen + cfg.conv3x3.OUTTILE_YSIZE - 1) / cfg.conv3x3.OUTTILE_YSIZE;
      int numTilesTotal = batchSize * numTilesX * numTilesY;

      int inTileXSize = cfg.conv3x3.INTILE_XSIZE;
      int inTileYSize = cfg.conv3x3.INTILE_YSIZE;

      int maxChannels = model->maxConvChannels(3,3);
      maxChannels = std::max(model->trunk.trunkNumChannels,maxChannels);
      maxChannels = std::max(model->trunk.midNumChannels,maxChannels);
      maxChannels = std::max(model->trunk.regularNumChannels,maxChannels);
      maxChannels = std::max(model->trunk.gpoolNumChannels,maxChannels);

      int ioNumFloats = numTilesTotal * maxChannels * inTileXSize * inTileYSize;
      int filterNumFloats = maxChannels * maxChannels * inTileXSize * inTileYSize;
      cl_mem input = randomReadOnlyBuffer("tuneXGemm3x3Input", context, ioNumFloats, 1.0);
      cl_mem filter = randomReadOnlyBuffer("tuneXGemm3x3Filter", context, filterNumFloats, 1.0 / sqrt(maxChannels * 3 * 3));
      cl_mem output = createReadWriteBuffer(context, ioNumFloats);

      OpenCLTuneAccums accums;
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
          commandQueue,
          cfg,
          outChannels, numTilesTotal, inChannels,
          filter, input, output,
          inTileXSize * inTileYSize,
          &event
        );

        accums.countResultAndFreeEvent(err,event,weight);
        if(accums.bad)
          break;
      }

      if(accums.bad)
        ret.assign(ioNumFloats,0.0);
      else
        blockingReadBuffer(commandQueue, output, ioNumFloats, ret);

      clReleaseMemObject(input);
      clReleaseMemObject(filter);
      clReleaseMemObject(output);

      clReleaseKernel(kernel);
      clReleaseProgram(program);

      return accums;
    };

    testAllConfigs(
      configs,
      currentConfig,
      referenceConfig,
      out,
      std::function<string(const OpenCLTuneParams& cfg)>(getDesc),
      std::function<OpenCLTuneAccums(const OpenCLTuneParams& cfg, vector<float>& ret)>(test),
      handleBestSoFar
    );
  }

  out << "Done tuning" << endl;
  out << "------------------------------------------------------" << endl;

}

string OpenCLTuner::defaultDirectory(bool makeDir) {
  string dir = HomeData::getHomeDataDir(true);
  dir += "/opencltuning";
  if(makeDir)
    MakeDir::make(dir);
  return dir;
}

string OpenCLTuner::defaultFileName(const string& gpuName, int nnXLen, int nnYLen, const ModelDesc* model) {
  string gpuNameForFile;
  for(int i = 0; i<gpuName.length(); i++) {
    char c = gpuName[i];
    if(contains("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789", c))
      gpuNameForFile += c;
  }
  return Global::strprintf("tune_gpu%s_x%d_y%d_c%d_mv%d.txt", gpuNameForFile.c_str(), nnXLen, nnYLen, model->trunk.trunkNumChannels,model->version);
}

OpenCLTuneParams OpenCLTuner::loadOrAutoTune(
  string openCLTunerFile,
  const string& gpuName,
  int gpuIdxForTuning,
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
  openCLTunerFile = dir + "/" + OpenCLTuner::defaultFileName(gpuName, nnXLen, nnYLen, model);

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

    vector<DeviceInfo> allDeviceInfos = DeviceInfo::getAllDeviceInfosOnSystem(logger);
    if(gpuIdxForTuning < 0 || gpuIdxForTuning >= allDeviceInfos.size())
      throw StringError("Requested gpuIdxForTuning for autotuning was not a valid device: " + Global::intToString(gpuIdxForTuning));
    if(allDeviceInfos[gpuIdxForTuning].name != gpuName)
      throw StringError(
        "Requested gpuIdxForTuning for autotuning expected a device with name " +
        gpuName + " but found a device with name " + allDeviceInfos[gpuIdxForTuning].name
      );


    bool enableProfiling = true;
    DevicesContext devicesContext(allDeviceInfos, {gpuIdxForTuning}, enableProfiling);

    OpenCLTuneParams initialParams;
    int batchSize = OpenCLTuner::DEFAULT_BATCH_SIZE;
    OpenCLTuner::tune(
      initialParams,
      devicesContext,
      gpuIdxForTuning,
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
