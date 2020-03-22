#include "commandline.h"
#include "program/setup.h"
#include "main.h"

using namespace std;

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

//Empty string indicates no default
void KataGoCommandLine::addModelFileArg(const string& defaultModelPath) {
  assert(modelFileArg == NULL);
  string helpDesc = "Neural net model file.";
  bool required = true;
  if(defaultModelPath != string()) {
    helpDesc += " Defaults to: " + defaultModelPath;
    required = false;
  }
  modelFileArg = new TCLAP::ValueArg<string>("","model",helpDesc,required,defaultModelPath,"FILE");
  this->add(*modelFileArg);
}

//Empty string indicates no default
void KataGoCommandLine::addConfigFileArg(const string& defaultConfigPath, const string& exampleConfigFile) {
  assert(configFileArg == NULL);
  string helpDesc = "Config file to use";
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
