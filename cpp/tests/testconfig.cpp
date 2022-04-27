#include "../tests/tests.h"

#include "../core/config_parser.h"
#include "../core/mainargs.h"
#include "../command/commandline.h"

using namespace std;
using namespace TestCommon;

void Tests::runConfigTests(const vector<string>& args) {

  if (args.size() > 1) {
    // interactive test with passing command-line arguments and printing output
    KataGoCommandLine cmd("Run KataGo configuration file(s) unit-tests.");
    try {
      ConfigParser cfg(false);

      cmd.addConfigFileArg("data/test.cfg","data/analysis_example.cfg");
      cmd.addOverrideConfigArg();

      cmd.parseArgs(args);

      cmd.getConfig(cfg);

      Logger logger(&cfg, true);
    } catch (TCLAP::ArgException &e) {
      cerr << "Error: " << e.error() << " for argument " << e.argId() << endl;
      throw StringError("Wrong command-line parameters");
    }
    return;
  }


  // unit-tests
  {
    ConfigParser cfg("data/analysis_example.cfg");
    if(cfg.getInt("nnMaxBatchSize") != 64)
      throw StringError("nnMaxBatchSize param reading error from data/analysis_example.cfg");
    cout << "Config reading param OK\n";
  }

  {
    try {
      ConfigParser cfg("data/test-duplicate.cfg");
      throw StringError("Duplicate param logDir should trigger a error in data/test-duplicate.cfg");
    } catch (const IOError&) {
      // expected behaviour, do nothing here
      cout << "Config duplicate param error triggering OK\n";
    }
  }

  {
    ConfigParser cfg("data/test-duplicate.cfg", true);
    if(cfg.getString("logDir") != "more_logs")
      throw StringError("logDir param overriding in the same file error in data/test-duplicate.cfg");
    cout << "Config param overriding in the same file OK\n";
  }

  {
    try {
      ConfigParser cfg("test.cfg", false, false);
      throw StringError("Overriden param should trigger a error "
                        "when key overriding is disabled in data/test.cfg");
    } catch (const IOError&) {
      // expected behaviour, do nothing here
      cout << "Config overriding error triggering OK\n";
    }
  }

  {
    ConfigParser cfg("data/test.cfg");
    if(!cfg.contains("reportAnalysisWinratesAs"))
      throw StringError("Config reading error from included file in a subdirectory "
                        "(data/folded/analysis_example.cfg) in data/test.cfg");
    if(cfg.getInt("maxVisits") != 1000)
      throw StringError("Config value (maxVisits) overriding error from "
                        "data/test1.cfg in data/test.cfg");
    if(cfg.getString("logDir") != "more_logs")
      throw StringError("logDir param overriding error in data/test.cfg");
    if(cfg.getInt("nnMaxBatchSize") != 100500)
      throw StringError("nnMaxBatchSize param overriding error in data/test.cfg");
    cout << "Config overriding test OK\n";
  }

  // multiple config files from command line
  {
    vector<string> testArgs = {
      "runconfigtests",
      "-config",
      "data/analysis_example.cfg",
      "-config",
      "data/test2.cfg"
    };
    ConfigParser cfg;
    KataGoCommandLine cmd("Run KataGo configuration file(s) unit-tests.");
    try {
      cmd.addConfigFileArg("data/test.cfg","data/analysis_example.cfg");
      cmd.addOverrideConfigArg();

      cmd.parseArgs(testArgs);

      cmd.getConfig(cfg);

    } catch (TCLAP::ArgException &e) {
      cerr << "Error: " << e.error() << " for argument " << e.argId() << endl;
      throw StringError("Wrong command-line parameters");
    }

    if(cfg.getInt("nnMaxBatchSize") != 100)
      throw StringError("nnMaxBatchSize param overriding error while reading "
                        "multiple configs from command line "
                        "(data/analysis_example.cfg and data/test2.cfg)");

    cout << "Config overriding from command line OK\n";
  }
}
