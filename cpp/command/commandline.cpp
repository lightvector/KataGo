#include "../command/commandline.h"

#include "../core/os.h"
#include "../dataio/homedata.h"
#include "../program/setup.h"
#include "../main.h"

#include <boost/filesystem.hpp>
namespace bfs = boost::filesystem;

using namespace std;

//--------------------------------------------------------------------------------------

static bool doesPathExist(const string& path) {
  try {
    bfs::path bfsPath(path);
    return bfs::exists(bfsPath);
  }
  catch(const bfs::filesystem_error&) {
    return false;
  }
}

static string getDefaultConfigPathForHelp(const string& defaultConfigFileName) {
  return HomeData::getDefaultFilesDirForHelpMessage() + "/" + defaultConfigFileName;
}
static vector<string> getDefaultConfigPaths(const string& defaultConfigFileName) {
  vector<string> v = HomeData::getDefaultFilesDirs();
  for(int i = 0; i<v.size(); i++) {
    v[i] = v[i] + "/" + defaultConfigFileName;
  }
  return v;
}

static string getDefaultModelPathForHelp() {
  return HomeData::getDefaultFilesDirForHelpMessage() + "/" + "default_model.bin.gz";
}

static vector<string> getDefaultModelPaths() {
  vector<string> dirs = HomeData::getDefaultFilesDirs();
  vector<string> ret;
  for(int i = 0; i<dirs.size(); i++) {
    ret.push_back(dirs[i] + "/" + "default_model.bin.gz");
    ret.push_back(dirs[i] + "/" + "default_model.txt.gz");
  }
  return ret;
}

//--------------------------------------------------------------------------------------

//This is basically TCLAP's StdOutput but some of the methods reimplemented to do a few nicer things.
class KataHelpOutput : public TCLAP::StdOutput
{
  int numBuiltInArgs;
  int shortUsageArgLimit;

  public:

  KataHelpOutput(int numBuiltIn, int shortUsageLimit)
    :TCLAP::StdOutput(),
    numBuiltInArgs(numBuiltIn),
    shortUsageArgLimit(shortUsageLimit)
  {}

  virtual ~KataHelpOutput() {}


  void setShortUsageArgLimit(int n) {
    shortUsageArgLimit = n;
  }

  virtual void usage(TCLAP::CmdLineInterface& _cmd )
  {
    string message = _cmd.getMessage();
    cout << endl << "DESCRIPTION: " << endl << endl;
    spacePrint(cout, message, 75, 3, 0);
    cout << endl << "USAGE: " << endl << endl;
    _shortUsage( _cmd, cout );
    cout << endl << endl << "Where: " << endl << endl;
    _longUsage( _cmd, cout );
    cout << endl;
  }

  virtual void _shortUsage(TCLAP::CmdLineInterface& _cmd, ostream& os) const
  {
    using namespace TCLAP;
    list<Arg*> argList = _cmd.getArgList();
    vector<Arg*> argVec = vector<Arg*>(argList.begin(),argList.end());
    string progName = _cmd.getProgramName();
    XorHandler xorHandler = _cmd.getXorHandler();
    vector<vector<Arg*>> xorList = xorHandler.getXorList();

    string s = progName + " ";

    // first the xor
    for(int i = 0; static_cast<unsigned int>(i) < xorList.size(); i++)
    {
      s += " {";
      for(ArgVectorIterator it = xorList[i].begin(); it != xorList[i].end(); it++)
        s += (*it)->shortID() + "|";

      s[s.length()-1] = '}';
    }

    // TCLAP adds arguments in reverse order for some reason. So we iterate in reverse for help output.
    // Also we limit based on shortUsageArgLimit.
    int lowerLimit = shortUsageArgLimit < 0 ? 0 : std::max(0, (int)argVec.size() - numBuiltInArgs - 1 - shortUsageArgLimit + 1);
    for(int i = argVec.size() - numBuiltInArgs - 1; i >= lowerLimit; i--) {
      if(!xorHandler.contains(argVec[i]))
        s += " " + argVec[i]->shortID();
    }

    if(lowerLimit > 0)
      s += " [...other flags...]";

    // if the program name is too long, then adjust the second line offset
    int secondLineOffset = static_cast<int>(progName.length()) + 2;
    if(secondLineOffset > 75/2)
      secondLineOffset = static_cast<int>(75/2);

    spacePrint(os, s, 75, 3, secondLineOffset);
  }

  virtual void _longUsage(TCLAP::CmdLineInterface& _cmd, ostream& os) const
  {
    using namespace TCLAP;
    list<Arg*> argList = _cmd.getArgList();
    vector<Arg*> argVec = vector<Arg*>(argList.begin(),argList.end());
    XorHandler xorHandler = _cmd.getXorHandler();
    vector<vector<Arg*>> xorList = xorHandler.getXorList();

    // first the xor
    for(int i = 0; static_cast<unsigned int>(i) < xorList.size(); i++)
    {
      for(ArgVectorIterator it = xorList[i].begin(); it != xorList[i].end(); it++)
      {
        spacePrint(os, (*it)->longID(), 75, 3, 3);
        spacePrint(os, (*it)->getDescription(), 75, 5, 0);

        if(it+1 != xorList[i].end())
          spacePrint(os, "-- OR --", 75, 9, 0);
      }
      os << endl << endl;
    }

    // TCLAP adds arguments in reverse order for some reason. So we iterate in reverse for help output.
    // Also we limit based on shortUsageArgLimit.
    for(int i = argVec.size() - numBuiltInArgs - 1; i >= 0; i--) {
      if(!xorHandler.contains(argVec[i])) {
        spacePrint(os, argVec[i]->longID(), 75, 3, 3);
        spacePrint(os, argVec[i]->getDescription(), 75, 5, 0);
        os << endl;
      }
    }

    //Now also show the default args.
    for(int i = argVec.size() - numBuiltInArgs; i < argVec.size(); i++) {
      if(!xorHandler.contains(argVec[i])) {
        spacePrint(os, argVec[i]->longID(), 75, 3, 3);
        spacePrint(os, argVec[i]->getDescription(), 75, 5, 0);
        os << endl;
      }
    }

    os << endl;
  }


};


//--------------------------------------------------------------------------------------


KataGoCommandLine::KataGoCommandLine(const string& message)
  :TCLAP::CmdLine(message, ' ', Version::getKataGoVersionFullInfo(),true),
  modelFileArg(NULL),
  configFileArg(NULL),
  overrideConfigArg(NULL),
  defaultConfigFileName(),
  numBuiltInArgs(_argList.size()),
  helpOutput(NULL)
{
  helpOutput = new KataHelpOutput(numBuiltInArgs, -1);
  setOutput(helpOutput);
}

KataGoCommandLine::~KataGoCommandLine() {
  delete modelFileArg;
  delete configFileArg;
  delete overrideConfigArg;
  delete helpOutput;
}


string KataGoCommandLine::defaultGtpConfigFileName() {
  return "default_gtp.cfg";
}

void KataGoCommandLine::setShortUsageArgLimit() {
  helpOutput->setShortUsageArgLimit(_argList.size() - numBuiltInArgs);
}

void KataGoCommandLine::addModelFileArg() {
  assert(modelFileArg == NULL);
  string helpDesc = "Neural net model file. Defaults to: " + getDefaultModelPathForHelp();
  bool required = false;
  //We don't apply a default directly here, but rather in getModelFile() since there is more than one path we
  //need to check. Also it's more robust if we don't attempt any filesystem access (which could fail)
  //before we've even constructed the command arguments and help.
  string defaultPath = "";
  modelFileArg = new TCLAP::ValueArg<string>("","model",helpDesc,required,defaultPath,"FILE");
  this->add(*modelFileArg);
}

//Empty string indicates no default
void KataGoCommandLine::addConfigFileArg(const string& defaultCfgFileName, const string& exampleConfigFile) {
  assert(configFileArg == NULL);
  defaultConfigFileName = defaultCfgFileName;

  string helpDesc = "Config file to use";
  bool required = true;
  if(!exampleConfigFile.empty())
    helpDesc += " (see " + exampleConfigFile + " or configs/" + exampleConfigFile + ")";
  helpDesc += ".";
  if(!defaultConfigFileName.empty()) {
    helpDesc += " Defaults to: " + getDefaultConfigPathForHelp(defaultConfigFileName);
    required = false;
  }
  //We don't apply the default directly here, but rather in getConfig(). It's more robust if we don't attempt any
  //filesystem access (which could fail) before we've even constructed the command arguments and help.
  string defaultPath = "";
  configFileArg = new TCLAP::ValueArg<string>("","config",helpDesc,required,defaultPath,"FILE");
  this->add(*configFileArg);
}

void KataGoCommandLine::addOverrideConfigArg() {
  assert(overrideConfigArg == NULL);
  overrideConfigArg = new TCLAP::ValueArg<string>(
    "","override-config","Override config parameters. Format: \"key=value, key=value,...\"",false,string(),"KEYVALUEPAIRS"
  );
  this->add(*overrideConfigArg);
}


string KataGoCommandLine::getModelFile() const {
  assert(modelFileArg != NULL);
  string modelFile = modelFileArg->getValue();
  if(modelFile.empty()) {
    string pathForErrMsg;
    try {
      vector<string> paths = getDefaultModelPaths();
      if(paths.size() > 0)
        pathForErrMsg = paths[0];
      for(const string& path: paths)
        if(doesPathExist(path))
          return path;
    }
    catch(const StringError& err) {
      throw StringError(string("'-model MODELFILENAME.bin.gz was not provided but encountered error searching for default: ") + err.what());
    }
    if(pathForErrMsg == "")
      pathForErrMsg = getDefaultModelPathForHelp();
    throw StringError("-model MODELFILENAME.bin.gz was not specified to tell KataGo where to find the neural net model, and default was not found at " + pathForErrMsg);
  }
  return modelFile;
}

bool KataGoCommandLine::modelFileIsDefault() const {
  return modelFileArg->getValue().empty();
}

string KataGoCommandLine::getConfigFile() const {
  assert(configFileArg != NULL);
  string configFile = configFileArg->getValue();
  if(configFile.empty() && !defaultConfigFileName.empty()) {
    string pathForErrMsg;
    try {
      vector<string> paths = getDefaultConfigPaths(defaultConfigFileName);
      if(paths.size() > 0)
        pathForErrMsg = paths[0];
      for(const string& path: paths)
        if(doesPathExist(path))
          return path;
    }
    catch(const StringError& err) {
      throw StringError(string("'-config CONFIG_FILE_NAME.cfg was not provided but encountered error searching for default: ") + err.what());
    }
    if(pathForErrMsg == "")
      pathForErrMsg = getDefaultConfigPathForHelp(defaultConfigFileName);
    throw StringError("-config CONFIG_FILE_NAME.cfg was not specified to tell KataGo where to find the config, and default was not found at " + pathForErrMsg);
  }
  return configFile;
}

void KataGoCommandLine::maybeApplyOverrideConfigArg(ConfigParser& cfg) const {
  if(overrideConfigArg != NULL) {
    string overrideConfig = overrideConfigArg->getValue();
    if(overrideConfig != "") {
      map<string,string> newkvs = ConfigParser::parseCommaSeparated(overrideConfig);
      //HACK to avoid a common possible conflict - if we specify some of the rules options on one side, the other side should be erased.
      vector<pair<set<string>,set<string>>> mutexKeySets = Setup::getMutexKeySets();
      cfg.overrideKeys(newkvs,mutexKeySets);
    }
  }
}

//cfg must be uninitialized, this will initialize it based on user-provided arguments
void KataGoCommandLine::getConfig(ConfigParser& cfg) const {
  string configFile = getConfigFile();
  cfg.initialize(configFile);
  maybeApplyOverrideConfigArg(cfg);
}

void KataGoCommandLine::getConfigAllowEmpty(ConfigParser& cfg) const {
  if(configFileArg->getValue().empty() && defaultConfigFileName.empty()) {
    cfg.initialize(std::map<string,string>());
    maybeApplyOverrideConfigArg(cfg);
  }
  else {
    getConfig(cfg);
  }
}
