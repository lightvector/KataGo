#ifndef COMMANDLINE_H_
#define COMMANDLINE_H_

#include "core/config_parser.h"

#define TCLAP_NAMESTARTSTRING "-" //Use single dashes for all flags
#include <tclap/CmdLine.h>

class KataGoCommandLine : public TCLAP::CmdLine
{
  TCLAP::ValueArg<std::string>* modelFileArg;
  TCLAP::ValueArg<std::string>* configFileArg;
  TCLAP::ValueArg<std::string>* overrideConfigArg;

  public:
  KataGoCommandLine(const std::string& message);
  ~KataGoCommandLine();

  static std::string defaultGtpConfigFileName();

  void addModelFileArg();
  //Empty string indicates no default or no example
  void addConfigFileArg(const std::string& defaultConfigFileName, const std::string& exampleConfigFile);
  void addOverrideConfigArg();

  std::string getModelFile();
  //cfg must be uninitialized, this will initialize it based on user-provided arguments
  void getConfig(ConfigParser& cfg);
};

#endif //COMMANDLINE_H_
