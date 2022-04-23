#include "../tests/tests.h"

#include "../core/config_parser.h"
#include "../core/mainargs.h"
#include "../command/commandline.h"

using namespace std;
using namespace TestCommon;

void Tests::runConfigTests(const vector<string>& args) {
  ConfigParser cfg(true);

  KataGoCommandLine cmd("Run KataGo configuration file(s) unit-tests.");
  try {
    cmd.addConfigFileArg("data/test.cfg","data/analysis_example.cfg");
    cmd.addOverrideConfigArg();

    cmd.parseArgs(args);

    cmd.getConfig(cfg);
  }
  catch (TCLAP::ArgException &e) {
    cerr << "Error: " << e.error() << " for argument " << e.argId() << endl;
  }
}
