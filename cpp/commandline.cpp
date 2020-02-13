#include "commandline.h"

#include "core/os.h"
#include "program/setup.h"
#include "main.h"

#if defined(OS_IS_UNIX_OR_APPLE)
  #include <wordexp.h>
#elif defined(OS_IS_WINDOWS)
  // TODO whatever windows needs to expand the path to the home directory
#else
  #error Unknown operating system!
#endif

#include <boost/filesystem.hpp>
namespace bfs = boost::filesystem;

using namespace std;

//--------------------------------------------------------------------------------------

static bfs::path getHomeDirectory() {
  bfs::path homeDirectory;
#if defined(OS_IS_WINDOWS)
  #error FIXME needs implementing
  // TODO I have no clue how windows handles this
  // possibly using ExpandEnvironmentString, see https://stackoverflow.com/questions/1902681/expand-file-names-that-have-environment-variables-in-their-path
#elif defined(OS_IS_UNIX_OR_APPLE)
  wordexp_t expandedPath;
  wordexp("~", &expandedPath, 0);
  homeDirectory = expandedPath.we_wordv[0];
  wordfree(&expandedPath);
#else
  #error Unknown operating system!
#endif
  return homeDirectory;
}

static string defaultPathIfItExists(bfs::path standardFileName) {
  bfs::path homeDirectory = getHomeDirectory();
  bfs::path standardModelPath = homeDirectory / ".katago" / standardFileName;
  if(bfs::exists(standardModelPath)) {
    return standardModelPath.native();
  }

  // no default file found
  return string();
}

static std::string findDefaultConfigPath(const string& defaultConfigFileName) {
  return defaultPathIfItExists(defaultConfigFileName);
}

static std::string findDefaultModelPath() {
  string s = defaultPathIfItExists("default_model.bin.gz");
  if(s == string()) {
    s = defaultPathIfItExists("default_model.txt.gz");
  }
  return s;
}

//--------------------------------------------------------------------------------------


KataGoCommandLine::KataGoCommandLine(const string& message)
  :TCLAP::CmdLine(message, ' ', Version::getKataGoVersionForHelp(),true),
  modelFileArg(NULL),
  configFileArg(NULL),
  overrideConfigArg(NULL)
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
  string helpDesc = "Neural net model file.";
  string defaultModelPath = findDefaultModelPath();
  bool required = true;
  if(defaultModelPath != string()) {
    helpDesc += " Defaults to: " + defaultModelPath;
    required = false;
  }
  modelFileArg = new TCLAP::ValueArg<string>("","model",helpDesc,required,defaultModelPath,"FILE");
  this->add(*modelFileArg);
}

//Empty string indicates no default
void KataGoCommandLine::addConfigFileArg(const string& defaultConfigFileName, const string& exampleConfigFile) {
  assert(configFileArg == NULL);
  string helpDesc = "Config file to use";
  string defaultConfigPath = defaultConfigFileName != string() ? findDefaultConfigPath(defaultConfigFileName) : string();
  bool required = true;
  if(exampleConfigFile != string())
    helpDesc += " (see " + exampleConfigFile + " or configs/" + exampleConfigFile + ")";
  helpDesc += ".";
  if(defaultConfigPath != string()) {
    helpDesc += " Defaults to: " + defaultConfigPath;
    required = false;
  }
  configFileArg = new TCLAP::ValueArg<string>("","config",helpDesc,required,defaultConfigPath,"FILE");
  this->add(*configFileArg);
}

void KataGoCommandLine::addOverrideConfigArg() {
  assert(overrideConfigArg == NULL);
  overrideConfigArg = new TCLAP::ValueArg<string>(
    "","override-config","Override config parameters. Format: \"key=value, key=value,...\"",false,string(),"KEYVALUEPAIRS"
  );
  this->add(*overrideConfigArg);
}


string KataGoCommandLine::getModelFile() {
  assert(modelFileArg != NULL);
  return modelFileArg->getValue();
}

//cfg must be uninitialized, this will initialize it based on user-provided arguments
void KataGoCommandLine::getConfig(ConfigParser& cfg) {
  assert(configFileArg != NULL);
  cfg.initialize(configFileArg->getValue());

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
