#include "commandline.h"

#include "core/os.h"
#include "dataio/homedata.h"
#include "program/setup.h"
#include "main.h"

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
static string getDefaultConfigPath(const string& defaultConfigFileName) {
  return HomeData::getDefaultFilesDir() + "/" + defaultConfigFileName;
}

static string getDefaultModelPathForHelp() {
  return HomeData::getDefaultFilesDirForHelpMessage() + "/" + "default_model.bin.gz";
}

static string getDefaultBinModelPath() {
  return HomeData::getDefaultFilesDir() + "/" + "default_model.bin.gz";
}
static string getDefaultTxtModelPath() {
  return HomeData::getDefaultFilesDir() + "/" + "default_model.txt.gz";
}

//--------------------------------------------------------------------------------------


KataGoCommandLine::KataGoCommandLine(const string& message)
  :TCLAP::CmdLine(message, ' ', Version::getKataGoVersionForHelp(),true),
  modelFileArg(NULL),
  configFileArg(NULL),
  overrideConfigArg(NULL),
  defaultConfigFileName()
{}

KataGoCommandLine::~KataGoCommandLine() {
  delete modelFileArg;
  delete configFileArg;
  delete overrideConfigArg;
}


string KataGoCommandLine::defaultGtpConfigFileName() {
  return "default_gtp.cfg";
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
    string defaultBinModelPath;
    string defaultTxtModelPath;
    try {
      defaultBinModelPath = getDefaultBinModelPath();
      if(doesPathExist(defaultBinModelPath))
        return defaultBinModelPath;
      defaultTxtModelPath = getDefaultTxtModelPath();
      if(doesPathExist(defaultTxtModelPath))
        return defaultTxtModelPath;
    }
    catch(const StringError& err) {
      throw StringError(string("'-model MODELFILENAME.bin.gz was not provided but encountered error searching for default: ") + err.what());
    }
    throw StringError("-model MODELFILENAME.bin.gz was not specified to tell KataGo where to find the neural net model, and default was not found at " + defaultBinModelPath);
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
    string defaultConfigPath;
    try {
      defaultConfigPath = getDefaultConfigPath(defaultConfigFileName);
      if(doesPathExist(defaultConfigPath))
        return defaultConfigPath;
    }
    catch(const StringError& err) {
      throw StringError(string("'-config CONFIG_FILE_NAME.cfg was not provided but encountered error searching for default: ") + err.what());
    }
    throw StringError("-config CONFIG_FILE_NAME.cfg was not specified to tell KataGo where to find the config, and default was not found at " + defaultConfigPath);
  }
  return configFile;
}

//cfg must be uninitialized, this will initialize it based on user-provided arguments
void KataGoCommandLine::getConfig(ConfigParser& cfg) const {
  string configFile = getConfigFile();
  cfg.initialize(configFile);

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
