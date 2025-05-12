#ifndef COMMANDLINE_H_
#define COMMANDLINE_H_

#include "../core/config_parser.h"

#define TCLAP_NAMESTARTSTRING "-" //Use single dashes for all flags
#include <tclap/CmdLine.h>

class KataHelpOutput;
class Logger;

class KataGoCommandLine : public TCLAP::CmdLine
{
  TCLAP::ValueArg<std::string>* modelFileArg;
  TCLAP::ValueArg<std::string>* coreMLModelFileArg;
  TCLAP::ValueArg<std::string>* humanModelFileArg;
  TCLAP::ValueArg<std::string>* humanCoreMLModelFileArg;
  TCLAP::MultiArg<std::string>* configFileArg;
  TCLAP::MultiArg<std::string>* overrideConfigArg;
  std::string defaultConfigFileName;
  int numBuiltInArgs;
  KataHelpOutput* helpOutput;

  public:
  KataGoCommandLine(const std::string& message);
  ~KataGoCommandLine();

  static std::string defaultGtpConfigFileName();

  void parseArgs(const std::vector<std::string>& args);

  //Args added AFTER calling this will only show up in the long help output, and not the short usage line.
  void setShortUsageArgLimit();

  void addModelFileArg();
  void addCoreMLModelFileArg();
  void addHumanModelFileArg();
  void addHumanCoreMLModelFileArg();
  //Empty string indicates no default or no example
  void addConfigFileArg(const std::string& defaultConfigFileName, const std::string& exampleConfigFile);
  void addConfigFileArg(const std::string& defaultConfigFileName, const std::string& exampleConfigFile, bool required);
  void addOverrideConfigArg();

  void logOverrides(Logger& logger) const;

  std::string getModelFile() const;
  std::string getCoreMLModelFile() const;
  bool modelFileIsDefault() const;

  std::string getHumanModelFile() const;
  std::string getHumanCoreMLModelFile() const;

  //cfg must be uninitialized, this will initialize it based on user-provided arguments
  void getConfig(ConfigParser& cfg) const;
  void getConfigAllowEmpty(ConfigParser& cfg) const;

 private:
  std::vector<std::string> getConfigFiles() const;
  void maybeApplyOverrideConfigArg(ConfigParser& cfg) const;
};

#endif //COMMANDLINE_H_
