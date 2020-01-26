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

string OpenCLTuneParams::XGemmParams::desc() const {
  string s;
  s += "MWG=" + Global::intToString(MWG);
  s += " NWG=" + Global::intToString(NWG);
  s += " KWG=" + Global::intToString(KWG);
  s += " MDIMC=" + Global::intToString(MDIMC);
  s += " NDIMC=" + Global::intToString(NDIMC);
  s += " MDIMA=" + Global::intToString(MDIMA);
  s += " NDIMB=" + Global::intToString(NDIMB);
  s += " KWI=" + Global::intToString(KWI);
  s += " VWM=" + Global::intToString(VWM);
  s += " VWN=" + Global::intToString(VWN);
  s += " STRM=" + Global::intToString(STRM);
  s += " STRN=" + Global::intToString(STRN);
  s += " SA=" + Global::intToString(SA);
  s += " SB=" + Global::intToString(SB);
  return s;
}
string OpenCLTuneParams::XGemmParams::compileOptions() const {
  string s;
  s += "-DMWG=" + Global::intToString(MWG);
  s += " -DNWG=" + Global::intToString(NWG);
  s += " -DKWG=" + Global::intToString(KWG);
  s += " -DMDIMC=" + Global::intToString(MDIMC);
  s += " -DNDIMC=" + Global::intToString(NDIMC);
  s += " -DMDIMA=" + Global::intToString(MDIMA);
  s += " -DNDIMB=" + Global::intToString(NDIMB);
  s += " -DKWI=" + Global::intToString(KWI);
  s += " -DVWM=" + Global::intToString(VWM);
  s += " -DVWN=" + Global::intToString(VWN);
  s += " -DSTRM=" + Global::intToString(STRM);
  s += " -DSTRN=" + Global::intToString(STRN);
  s += " -DSA=" + Global::intToString(SA);
  s += " -DSB=" + Global::intToString(SB);
  return s;
}
void OpenCLTuneParams::XGemmParams::fillFromDesc(const string& fileName, const string& desc) {
  map<string,int> kvs = readDescKeyValues(fileName, desc);
  MWG = getInt(kvs,"MWG",MWG);
  NWG = getInt(kvs,"NWG",NWG);
  KWG = getInt(kvs,"KWG",KWG);
  MDIMC = getInt(kvs,"MDIMC",MDIMC);
  NDIMC = getInt(kvs,"NDIMC",NDIMC);
  MDIMA = getInt(kvs,"MDIMA",MDIMA);
  NDIMB = getInt(kvs,"NDIMB",NDIMB);
  KWI = getInt(kvs,"KWI",KWI);
  VWM = getInt(kvs,"VWM",VWM);
  VWN = getInt(kvs,"VWN",VWN);
  STRM = getInt(kvs,"STRM",STRM);
  STRN = getInt(kvs,"STRN",STRN);
  SA = getInt(kvs,"SA",SA);
  SB = getInt(kvs,"SB",SB);
}
bool OpenCLTuneParams::XGemmParams::isValid() const {
  if(MWG <= 0) return false;
  if(NWG <= 0) return false;
  if(KWG <= 0) return false;
  if(MDIMC <= 0) return false;
  if(NDIMC <= 0) return false;
  if(MDIMA <= 0) return false;
  if(NDIMB <= 0) return false;
  if(KWI <= 0) return false;
  if(VWM <= 0) return false;
  if(VWN <= 0) return false;
  if(STRM < 0 || STRM > 1) return false;
  if(STRN < 0 || STRN > 1) return false;
  if(SA < 0 || SA > 1) return false;
  if(SB < 0 || SB > 1) return false;
  if(!isMultipleOf(KWG,KWI)) return false;
  if(!isMultipleOf(MWG,MDIMC*VWM)) return false;
  if(!isMultipleOf(NWG,NDIMC*VWN)) return false;
  if(!isMultipleOf(MWG,MDIMA*VWM)) return false;
  if(!isMultipleOf(NWG,NDIMB*VWN)) return false;
  if(!isMultipleOf(KWG,VWM)) return false;
  if(!isMultipleOf(KWG,MDIMC*NDIMC/MDIMA)) return false;
  if(!isMultipleOf(KWG,MDIMC*NDIMC/NDIMB)) return false;
  return true;
}
bool OpenCLTuneParams::XGemmParams::isSimple() const {
  if(MDIMC != MDIMA) return false;
  if(NDIMC != NDIMB) return false;
  if(SA != SB) return false;
  if(VWM != VWN) return false;
  if(MWG != NWG) return false;
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
  s += " untransLocalSize0=" + Global::intToString(untransLocalSize0);
  s += " untransLocalSize1=" + Global::intToString(untransLocalSize1);
  s += " untransLocalSize2=" + Global::intToString(untransLocalSize2);
  return s;
}
string OpenCLTuneParams::Conv3x3Params::transDesc() const {
  string s;
  s += " transLocalSize0=" + Global::intToString(transLocalSize0);
  s += " transLocalSize1=" + Global::intToString(transLocalSize1);
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
  untransLocalSize0 = getInt(kvs,"untransLocalSize0",untransLocalSize0);
  untransLocalSize1 = getInt(kvs,"untransLocalSize1",untransLocalSize1);
  untransLocalSize2 = getInt(kvs,"untransLocalSize2",untransLocalSize2);
}
bool OpenCLTuneParams::Conv3x3Params::isValid() const {
  if(transLocalSize0 <= 0) return false;
  if(transLocalSize1 <= 0) return false;
  if(untransLocalSize0 <= 0) return false;
  if(untransLocalSize1 <= 0) return false;
  if(untransLocalSize2 <= 0) return false;

  if(transLocalSize0 * transLocalSize1 > 1024) return false;
  if(untransLocalSize0 * untransLocalSize1 * untransLocalSize2 > 1024) return false;

  //Currently, the only supported winograd tile sizes
  if(INTILE_XSIZE == 4 && OUTTILE_XSIZE == 2 && INTILE_YSIZE == 4 && OUTTILE_YSIZE == 2)
    return true;
  if(INTILE_XSIZE == 6 && OUTTILE_XSIZE == 4 && INTILE_YSIZE == 6 && OUTTILE_YSIZE == 4)
    return true;
  return false;
}


string OpenCLTuneParams::Conv5x5Params::desc() const {
  string s;
  s += "INTILE_XSIZE=" + Global::intToString(INTILE_XSIZE);
  s += " INTILE_YSIZE=" + Global::intToString(INTILE_YSIZE);
  s += " OUTTILE_XSIZE=" + Global::intToString(OUTTILE_XSIZE);
  s += " OUTTILE_YSIZE=" + Global::intToString(OUTTILE_YSIZE);
  s += " transLocalSize0=" + Global::intToString(transLocalSize0);
  s += " transLocalSize1=" + Global::intToString(transLocalSize1);
  s += " untransLocalSize0=" + Global::intToString(untransLocalSize0);
  s += " untransLocalSize1=" + Global::intToString(untransLocalSize1);
  s += " untransLocalSize2=" + Global::intToString(untransLocalSize2);
  return s;
}
string OpenCLTuneParams::Conv5x5Params::transDesc() const {
  string s;
  s += " transLocalSize0=" + Global::intToString(transLocalSize0);
  s += " transLocalSize1=" + Global::intToString(transLocalSize1);
  return s;
}
string OpenCLTuneParams::Conv5x5Params::untransDesc() const {
  string s;
  s += " untransLocalSize0=" + Global::intToString(untransLocalSize0);
  s += " untransLocalSize1=" + Global::intToString(untransLocalSize1);
  s += " untransLocalSize2=" + Global::intToString(untransLocalSize2);
  return s;
}
string OpenCLTuneParams::Conv5x5Params::compileOptions() const {
  string s;
  s += "-DINTILE_XSIZE=" + Global::intToString(INTILE_XSIZE);
  s += " -DINTILE_YSIZE=" + Global::intToString(INTILE_YSIZE);
  s += " -DOUTTILE_XSIZE=" + Global::intToString(OUTTILE_XSIZE);
  s += " -DOUTTILE_YSIZE=" + Global::intToString(OUTTILE_YSIZE);
  s += " -DCONV_XSIZE=5 -DCONV_YSIZE=5 -DINTILE_XOFFSET=(-2) -DINTILE_YOFFSET=(-2)";
  return s;
}
void OpenCLTuneParams::Conv5x5Params::fillFromDesc(const string& fileName, const string& desc) {
  map<string,int> kvs = readDescKeyValues(fileName, desc);
  INTILE_XSIZE = getInt(kvs,"INTILE_XSIZE",INTILE_XSIZE);
  INTILE_YSIZE = getInt(kvs,"INTILE_YSIZE",INTILE_YSIZE);
  OUTTILE_XSIZE = getInt(kvs,"OUTTILE_XSIZE",OUTTILE_XSIZE);
  OUTTILE_YSIZE = getInt(kvs,"OUTTILE_YSIZE",OUTTILE_YSIZE);
  transLocalSize0 = getInt(kvs,"transLocalSize0",transLocalSize0);
  transLocalSize1 = getInt(kvs,"transLocalSize1",transLocalSize1);
  untransLocalSize0 = getInt(kvs,"untransLocalSize0",untransLocalSize0);
  untransLocalSize1 = getInt(kvs,"untransLocalSize1",untransLocalSize1);
  untransLocalSize2 = getInt(kvs,"untransLocalSize2",untransLocalSize2);
}
bool OpenCLTuneParams::Conv5x5Params::isValid() const {
  if(transLocalSize0 <= 0) return false;
  if(transLocalSize1 <= 0) return false;
  if(untransLocalSize0 <= 0) return false;
  if(untransLocalSize1 <= 0) return false;
  if(untransLocalSize2 <= 0) return false;

  if(transLocalSize0 * transLocalSize1 > 1024) return false;
  if(untransLocalSize0 * untransLocalSize1 * untransLocalSize2 > 1024) return false;

  //Currently, the only supported winograd tile sizes
  if(INTILE_XSIZE == 6 && OUTTILE_XSIZE == 2 && INTILE_YSIZE == 6 && OUTTILE_YSIZE == 2)
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
  return xGemmDirect.isValid() && xGemm.isValid() && conv3x3.isValid() && conv5x5.isValid() && gPool.isValid() && transpose.isValid();
}

bool OpenCLTuneParams::operator==(const OpenCLTuneParams& other) const {
  if(this == &other)
    return true;
  return std::memcmp(this,&other,sizeof(OpenCLTuneParams)) == 0;
}


static const char* TUNEPARAMS_VERSION_LINE = "VERSION=6";
void OpenCLTuneParams::save(const string& filename, const OpenCLTuneParams& config) {
  ofstream out(filename);
  if(out.fail())
    throw IOError("Could not create file: " + filename);
  out << TUNEPARAMS_VERSION_LINE << "\n";
  out << "#xGemmDirect" << "\n";
  out << config.xGemmDirect.desc() << "\n";
  out << "#xGemm" << "\n";
  out << config.xGemm.desc() << "\n";
  out << "#conv3x3" << "\n";
  out << config.conv3x3.desc() << "\n";
  out << "#conv5x5" << "\n";
  out << config.conv5x5.desc() << "\n";
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

  if(filteredLines.size() != 7)
    throw IOError("OpenCLTuneParams::load: unexpected number of parameter lines in file " + filename);

  OpenCLTuneParams config;
  config.xGemmDirect.fillFromDesc(filename,filteredLines[1]);
  config.xGemm.fillFromDesc(filename,filteredLines[2]);
  config.conv3x3.fillFromDesc(filename,filteredLines[3]);
  config.conv5x5.fillFromDesc(filename,filteredLines[4]);
  config.gPool.fillFromDesc(filename,filteredLines[5]);
  config.transpose.fillFromDesc(filename,filteredLines[6]);
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
    buf[i] = (float)rand.nextDouble(scale);
  return createReadOnlyBuffer(context,buf);
}
static cl_mem randomReadOnly3dPaddedBuffer(
  const char* seed, cl_context context,
  int batchSize, int ySize, int ySizePadded, int xSize, int xSizePadded,
  double scale
) {
  vector<float> buf((size_t)batchSize*ySizePadded*xSizePadded);
  Rand rand(seed);
  size_t i = 0;
  for(int n = 0; n<batchSize; n++) {
    for(int y = 0; y<ySizePadded; y++) {
      for(int x = 0; x<xSizePadded; x++) {
        if(y < ySize && x < xSize)
          buf[i++] = (float)rand.nextDouble(scale);
        else
          buf[i++] = 0.0f;
      }
    }
  }
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
#define ISSIMPLE(field) std::function<bool(const OpenCLTuneParams&)>([](const OpenCLTuneParams& p){ return p.field.isSimple(); })

void OpenCLTuner::tune(
  const OpenCLTuneParams& initialConfig,
  DevicesContext& devicesContext,
  int gpuIdx,
  int batchSize,
  int nnXLen,
  int nnYLen,
  const ModelDesc* model,
  bool full,
  int winograd3x3TileSize,
  ostream& out,
  std::function<void(const OpenCLTuneParams&)> handleBestSoFar
) {
  const InitializedDevice& device = devicesContext.findGpuExn(gpuIdx);
  const cl_context& context = devicesContext.context;
  cl_command_queue commandQueue = device.commandQueue;
  const vector<cl_device_id>& deviceIdsToUse = { device.info.deviceId };

  OpenCLTuneParams untunedConfig = OpenCLTuneParams();
  OpenCLTuneParams currentConfig = initialConfig;

  if(winograd3x3TileSize == 2) {
    out << "Setting winograd3x3TileSize = 2" << endl;
    untunedConfig.conv3x3.INTILE_XSIZE = 4;
    untunedConfig.conv3x3.INTILE_YSIZE = 4;
    untunedConfig.conv3x3.OUTTILE_XSIZE = 2;
    untunedConfig.conv3x3.OUTTILE_YSIZE = 2;
    currentConfig.conv3x3.INTILE_XSIZE = 4;
    currentConfig.conv3x3.INTILE_YSIZE = 4;
    currentConfig.conv3x3.OUTTILE_XSIZE = 2;
    currentConfig.conv3x3.OUTTILE_YSIZE = 2;
  }
  else if(winograd3x3TileSize == 4) {
    out << "Setting winograd3x3TileSize = 4" << endl;
    untunedConfig.conv3x3.INTILE_XSIZE = 6;
    untunedConfig.conv3x3.INTILE_YSIZE = 6;
    untunedConfig.conv3x3.OUTTILE_XSIZE = 4;
    untunedConfig.conv3x3.OUTTILE_YSIZE = 4;
    currentConfig.conv3x3.INTILE_XSIZE = 6;
    currentConfig.conv3x3.INTILE_YSIZE = 6;
    currentConfig.conv3x3.OUTTILE_XSIZE = 4;
    currentConfig.conv3x3.OUTTILE_YSIZE = 4;
  }

  if(!currentConfig.isValid()) {
    out << "Loaded a config but the config was invalid, starting tuning from basic config" << endl;
    currentConfig = untunedConfig;
  }

  //=======================================================================================
  //Tune xGemmDirect
  {
    out << "------------------------------------------------------" << endl;
    out << "Tuning xGemmDirect for 1x1 convolutions and matrix mult" << endl;

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
      OpenCLTuneAccums accums;

      cl_int err;
      cl_program program;
      bool compileSuc = tryCompileProgram(
        "xgemmDirectProgram", context, deviceIdsToUse, OpenCLKernels::xgemmDirect,
        cfg.xGemmDirect.compileOptions(), program
      );
      if(!compileSuc) { accums.bad = true; accums.badErr = CL_BUILD_PROGRAM_FAILURE; return accums; }
      cl_kernel kernel = clCreateKernel(program, "XgemmDirectBatchedNN", &err);
      if(err != 0) { accums.bad = true; accums.badErr = err; return accums; }

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
      cl_mem input = randomReadOnlyBuffer("tuneXGemmDirect3x3Input", context, ioNumFloats, 1.0);
      cl_mem filter = randomReadOnlyBuffer("tuneXGemmDirect3x3Filter", context, filterNumFloats, 1.0 / sqrt(maxChannels * 3 * 3));
      cl_mem output = createReadWriteBuffer(context, ioNumFloats);

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
        err = doBatchedXGemmDirect_KM_KN_NM(
          kernel,
          commandQueue,
          cfg,
          numTilesTotal, outChannels, inChannels,
          input, filter, output,
          inTileXYSize,
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

  //=======================================================================================
  //Tune xGemm
  {
    out << "------------------------------------------------------" << endl;
    out << "Tuning xGemm for convolutions" << endl;

    vector<OpenCLTuneParams> configs;
    configs.push_back(currentConfig);
    if(full) {
      addConfigs(configs,SETTER(xGemm.MWG),{8,16,32,64,128});
      addConfigs(configs,SETTER(xGemm.NWG),{8,16,32,64,128});
      addConfigs(configs,SETTER(xGemm.KWG),{8,16,32});
      addConfigs(configs,SETTER(xGemm.MDIMC),{8,16,32});
      addConfigs(configs,SETTER(xGemm.NDIMC),{8,16,32});
      addConfigs(configs,SETTER(xGemm.MDIMA),{8,16,32});
      addConfigs(configs,SETTER(xGemm.NDIMB),{8,16,32});
      addConfigs(configs,SETTER(xGemm.KWI),{2,8});
      addConfigs(configs,SETTER(xGemm.VWM),{1,2,4,8});
      addConfigs(configs,SETTER(xGemm.VWN),{1,2,4,8});
      addConfigs(configs,SETTER(xGemm.STRM),{0});
      addConfigs(configs,SETTER(xGemm.STRN),{0});
      addConfigs(configs,SETTER(xGemm.SA),{0,1});
      addConfigs(configs,SETTER(xGemm.SB),{0,1});
      filterConfigs(configs,ISVALID(xGemm));
    }
    else {
      addConfigs(configs,SETTER(xGemm.MWG),{16,32,64});
      addConfigs(configs,SETTER(xGemm.NWG),{16,32,64});
      addConfigs(configs,SETTER(xGemm.KWG),{16,32});
      addConfigs(configs,SETTER(xGemm.MDIMC),{8,16,32});
      addConfigs(configs,SETTER(xGemm.NDIMC),{8,16,32});
      addConfigs(configs,SETTER(xGemm.MDIMA),{8,16,32});
      addConfigs(configs,SETTER(xGemm.NDIMB),{8,16,32});
      addConfigs(configs,SETTER(xGemm.KWI),{2});
      addConfigs(configs,SETTER(xGemm.VWM),{2,4});
      addConfigs(configs,SETTER(xGemm.VWN),{2,4});
      addConfigs(configs,SETTER(xGemm.STRM),{0});
      addConfigs(configs,SETTER(xGemm.STRN),{0});
      addConfigs(configs,SETTER(xGemm.SA),{0,1});
      addConfigs(configs,SETTER(xGemm.SB),{0,1});
      filterConfigs(configs,ISVALID(xGemm));
      filterConfigs(configs,ISSIMPLE(xGemm));
    }

    shuffleConfigs(configs);

    OpenCLTuneParams referenceConfig = currentConfig;
    referenceConfig.xGemm.MWG = untunedConfig.xGemm.MWG;
    referenceConfig.xGemm.NWG = untunedConfig.xGemm.NWG;
    referenceConfig.xGemm.KWG = untunedConfig.xGemm.KWG;
    referenceConfig.xGemm.MDIMC = untunedConfig.xGemm.MDIMC;
    referenceConfig.xGemm.NDIMC = untunedConfig.xGemm.NDIMC;
    referenceConfig.xGemm.MDIMA = untunedConfig.xGemm.MDIMA;
    referenceConfig.xGemm.NDIMB = untunedConfig.xGemm.NDIMB;
    referenceConfig.xGemm.KWI = untunedConfig.xGemm.KWI;
    referenceConfig.xGemm.VWM = untunedConfig.xGemm.VWM;
    referenceConfig.xGemm.VWN = untunedConfig.xGemm.VWN;
    referenceConfig.xGemm.STRM = untunedConfig.xGemm.STRM;
    referenceConfig.xGemm.STRN = untunedConfig.xGemm.STRN;
    referenceConfig.xGemm.SA = untunedConfig.xGemm.SA;
    referenceConfig.xGemm.SB = untunedConfig.xGemm.SB;

    OpenCLTuneParams slightlyTunedConfig = referenceConfig;
    slightlyTunedConfig.xGemm.MDIMC = 8;
    slightlyTunedConfig.xGemm.NDIMC = 8;
    slightlyTunedConfig.xGemm.MDIMA = 8;
    slightlyTunedConfig.xGemm.NDIMB = 8;
    OpenCLTuneParams slightlyTunedConfig2 = slightlyTunedConfig;
    slightlyTunedConfig2.xGemm.MWG = 16;
    slightlyTunedConfig2.xGemm.NWG = 16;
    slightlyTunedConfig2.xGemm.KWG = 16;

    configs.insert(configs.begin(),slightlyTunedConfig2);
    configs.insert(configs.begin(),slightlyTunedConfig);
    configs.insert(configs.begin(),currentConfig);

    auto getDesc = [](const OpenCLTuneParams& cfg) { return cfg.xGemm.desc(); };

    auto test = [&](const OpenCLTuneParams& cfg, vector<float>& ret) {
      OpenCLTuneAccums accums;

      cl_int err;
      cl_program program;
      bool compileSuc = tryCompileProgram(
        "xgemmProgram", context, deviceIdsToUse, OpenCLKernels::xgemm,
        cfg.xGemm.compileOptions(), program
      );
      if(!compileSuc) { accums.bad = true; accums.badErr = CL_BUILD_PROGRAM_FAILURE; return accums; }
      cl_kernel kernel = clCreateKernel(program, "XgemmBatched", &err);
      if(err != 0) { accums.bad = true; accums.badErr = err; return accums; }

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

      int numTilesTotalPadded = roundUpToMultiple(numTilesTotal,cfg.xGemm.MWG);
      int maxChannelsPadded = roundUpToMultiple(maxChannels,std::max(cfg.xGemm.NWG,cfg.xGemm.KWG));

      int ioNumFloats = numTilesTotalPadded * maxChannelsPadded * inTileXYSize;
      cl_mem input = randomReadOnly3dPaddedBuffer(
        "tuneXGemm3x3Input", context, inTileXYSize, maxChannels, maxChannelsPadded, numTilesTotal, numTilesTotalPadded, 1.0);
      cl_mem filter = randomReadOnly3dPaddedBuffer(
        "tuneXGemm3x3Filter", context, inTileXYSize, maxChannels, maxChannelsPadded, maxChannels, maxChannelsPadded, 1.0 / sqrt(maxChannels * 3 * 3));
      cl_mem output = createReadWriteBuffer(context, ioNumFloats);

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

        int outChannelsPadded = roundUpToMultiple(outChannels, cfg.xGemm.NWG);
        int inChannelsPadded = roundUpToMultiple(inChannels, cfg.xGemm.KWG);

        cl_event event;
        err = doBatchedXGemm_KM_KN_NM(
          kernel,
          commandQueue,
          cfg,
          numTilesTotalPadded, outChannelsPadded, inChannelsPadded,
          input, filter, output,
          inTileXYSize,
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

      //Compact ret down to only what we were supposed to get, without padding
      {
        int i = 0;
        for(int n = 0; n<inTileXYSize; n++) {
          for(int y = 0; y<maxChannels; y++) {
            for(int x = 0; x<numTilesTotal; x++) {
              ret[i++] = ret[x + numTilesTotalPadded * (y + maxChannelsPadded * n)];
            }
          }
        }
        ret.resize(inTileXYSize * maxChannels * numTilesTotal);
      }

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

  //=======================================================================================
  //Tune convolution transform
  {
    out << "------------------------------------------------------" << endl;
    out << "Tuning winograd transform for convolutions" << endl;

    vector<OpenCLTuneParams> configs;
    configs.push_back(currentConfig);
    if(full) {
      addConfigs(configs,SETTER(conv3x3.transLocalSize0),{1,2,4,8,16,32,64,128});
      addConfigs(configs,SETTER(conv3x3.transLocalSize1),{1,2,4,8,16,32,64});
    }
    else {
      addConfigs(configs,SETTER(conv3x3.transLocalSize0),{1,2,4,8,16,32,64,128});
      addConfigs(configs,SETTER(conv3x3.transLocalSize1),{1,2,4,8,16,32});
    }

    filterConfigs(configs,ISVALID(conv3x3));
    shuffleConfigs(configs);
    configs.insert(configs.begin(),currentConfig);

    OpenCLTuneParams referenceConfig = currentConfig;
    referenceConfig.conv3x3.transLocalSize0 = untunedConfig.conv3x3.transLocalSize0;
    referenceConfig.conv3x3.transLocalSize1 = untunedConfig.conv3x3.transLocalSize1;

    auto getDesc = [](const OpenCLTuneParams& cfg) { return cfg.conv3x3.transDesc(); };

    auto test = [&](const OpenCLTuneParams& cfg, vector<float>& ret) {
      OpenCLTuneAccums accums;

      cl_int err;
      cl_program program;
      bool compileSuc = tryCompileProgram(
        "winogradConv3x3NCHWProgram", context, deviceIdsToUse, OpenCLKernels::winogradConvNCHW,
        cfg.conv3x3.compileOptions(), program
      );
      if(!compileSuc) { accums.bad = true; accums.badErr = CL_BUILD_PROGRAM_FAILURE; return accums; }
      cl_kernel kernel = clCreateKernel(program, "transform", &err);
      if(err != 0) { accums.bad = true; accums.badErr = err; return accums; }

      int convSize = 3;
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
      int outputNumFloats = roundUpToMultiple(numTilesTotal,cfg.xGemm.MWG) * roundUpToMultiple(maxChannels,cfg.xGemm.KWG) * inTileXSize * inTileYSize;
      cl_mem input = randomReadOnlyBuffer("tune3x3TransInput", context, inputNumFloats, 1.0);
      cl_mem output = createReadWriteBuffer(context, outputNumFloats);

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
          nnXLen,nnYLen,
          batchSize,numTilesX,numTilesY,cfg.xGemm.MWG,
          inChannels,cfg.xGemm.KWG,
          convSize,
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
      OpenCLTuneAccums accums;

      cl_int err;
      cl_program program;
      bool compileSuc = tryCompileProgram(
        "winogradConv3x3NCHWProgram", context, deviceIdsToUse, OpenCLKernels::winogradConvNCHW,
        cfg.conv3x3.compileOptions(), program
      );
      if(!compileSuc) { accums.bad = true; accums.badErr = CL_BUILD_PROGRAM_FAILURE; return accums; }
      cl_kernel kernel = clCreateKernel(program, "untransform", &err);
      if(err != 0) { accums.bad = true; accums.badErr = err; return accums; }

      int convSize = 3;
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

      int inputNumFloats = roundUpToMultiple(numTilesTotal,cfg.xGemm.MWG) * roundUpToMultiple(maxChannels,cfg.xGemm.NWG) * inTileXSize * inTileYSize;
      int outputNumFloats = batchSize * nnXLen * nnYLen * maxChannels;
      cl_mem input = randomReadOnlyBuffer("tune3x3UntransInput", context, inputNumFloats, 1.0);
      cl_mem output = createReadWriteBuffer(context, outputNumFloats);

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
          nnXLen,nnYLen,
          batchSize,numTilesX,numTilesY,cfg.xGemm.MWG,
          outChannels,cfg.xGemm.NWG,
          convSize,
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
      OpenCLTuneAccums accums;

      cl_int err;
      cl_program program;
      bool compileSuc = tryCompileProgram(
        "gPoolChannelsNCHWProgram", context, deviceIdsToUse, OpenCLKernels::gPoolChannelsNCHW,
        cfg.gPool.compileOptions(), program
      );
      if(!compileSuc) { accums.bad = true; accums.badErr = CL_BUILD_PROGRAM_FAILURE; return accums; }
      cl_kernel kernel = clCreateKernel(program, "gPoolChannelsNCHW", &err);
      if(err != 0) { accums.bad = true; accums.badErr = err; return accums; }

      int inputNumFloats = batchSize * nnXLen * nnYLen * numChannels;
      int outputNumFloats = batchSize * numChannels * 3;

      cl_mem input = randomReadOnlyBuffer("tuneGPoolInput", context, inputNumFloats, 1.0);
      cl_mem maskSum = constantReadOnlyBuffer(context, batchSize, (float)(nnXLen*nnYLen));
      cl_mem output = createReadWriteBuffer(context, outputNumFloats);

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
      OpenCLTuneAccums accums;

      cl_int err;
      cl_program program;
      bool compileSuc = tryCompileProgram(
        "transposeNCHWProgram", context, deviceIdsToUse, OpenCLKernels::transposeNCHW,
        cfg.transpose.compileOptions(), program
      );
      if(!compileSuc) { accums.bad = true; accums.badErr = CL_BUILD_PROGRAM_FAILURE; return accums; }
      cl_kernel kernel = clCreateKernel(program, "transposeNCHW", &err);
      if(err != 0) { accums.bad = true; accums.badErr = err; return accums; }

      int numFloats = batchSize * nnXLen * nnYLen * numChannels;
      int outputNumFloats = numFloats;

      cl_mem input = randomReadOnlyBuffer("tuneTransposeInput", context, numFloats, 1.0);
      cl_mem output = createReadWriteBuffer(context, numFloats);

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


  //Copy 5x5 conv parameters over from 3x3 conv parameters
  currentConfig.conv5x5.transLocalSize0 = currentConfig.conv3x3.transLocalSize0;
  currentConfig.conv5x5.transLocalSize1 = currentConfig.conv3x3.transLocalSize1;
  currentConfig.conv5x5.untransLocalSize0 = currentConfig.conv3x3.untransLocalSize0;
  currentConfig.conv5x5.untransLocalSize1 = currentConfig.conv3x3.untransLocalSize1;
  currentConfig.conv5x5.untransLocalSize2 = currentConfig.conv3x3.untransLocalSize2;
  handleBestSoFar(currentConfig);

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

static OpenCLTuneParams loadFromTunerFile(const string& fileName, Logger* logger) {
  OpenCLTuneParams loadedParams = OpenCLTuneParams::load(fileName);
  if(!loadedParams.isValid())
    throw StringError("Loaded parmameters in " + fileName + " were not valid!");
  if(logger != NULL) {
    string message = "Loaded tuning parameters from: " + fileName;
    logger->write(message);
    if(!logger->isLoggingToStdout() && !logger->isLoggingToStderr())
      cerr << message << endl;
  }
  return loadedParams;
}

OpenCLTuneParams OpenCLTuner::loadOrAutoTune(
  string openCLTunerFile,
  const string& gpuName,
  int gpuIdxForTuning,
  Logger* logger,
  bool openCLReTunePerBoardSize,
  int nnXLen,
  int nnYLen,
  const ModelDesc* model,
  bool full
) {
  if(openCLTunerFile != "") {
    return loadFromTunerFile(openCLTunerFile,logger);
  }

  string dir = OpenCLTuner::defaultDirectory(true);
  openCLTunerFile = dir + "/" + OpenCLTuner::defaultFileName(gpuName, nnXLen, nnYLen, model);

  //Try loading the config for the proper size
  try {
    OpenCLTuneParams loadedParams = loadFromTunerFile(openCLTunerFile,logger);
    return loadedParams;
  }
  catch(const StringError& e) {
    (void)e;
  };

  //If not re-tuning per board size, then check if the tune config for the full size is there
  //And set the nnXLen and nnYLen we'll use for tuning to the full size
  if(!openCLReTunePerBoardSize) {
    nnXLen = NNPos::MAX_BOARD_LEN;
    nnYLen = NNPos::MAX_BOARD_LEN;
    openCLTunerFile = dir + "/" + OpenCLTuner::defaultFileName(gpuName, nnXLen, nnYLen, model);
    try {
      OpenCLTuneParams loadedParams = loadFromTunerFile(openCLTunerFile,logger);
      return loadedParams;
    }
    catch(const StringError& e) {
      (void)e;
    };
  }

  //No configs found at all, so now autotune
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
  DevicesContext devicesContext(allDeviceInfos, {gpuIdxForTuning}, logger, enableProfiling);

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
    DEFAULT_WINOGRAD_3X3_TILE_SIZE,
    cerr,
    std::function<void(const OpenCLTuneParams&)>(handleBestSoFar)
  );

  OpenCLTuneParams::save(openCLTunerFile, results);
  if(logger != NULL)
    logger->write("Done tuning, saved results to " + openCLTunerFile);
  return results;

}

#endif
