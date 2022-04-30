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
      Global::fatalError("Wrong command-line parameters");
    }
    return;
  }

  std::string dataPath("cpp/tests/data/");

  cout << "Running config tests" << endl;
  // unit-tests
  {
    ConfigParser cfg(dataPath + "analysis_example.cfg");
    if(cfg.getInt("nnMaxBatchSize") != 64)
      Global::fatalError("nnMaxBatchSize param reading error from data/analysis_example.cfg");
    cout << "Config reading param OK\n";
  }

  {
    try {
      ConfigParser cfg(dataPath + "test-duplicate.cfg");
      Global::fatalError("Duplicate param logDir should trigger a error in data/test-duplicate.cfg");
    } catch (const ConfigParsingError&) {
      // expected behaviour, do nothing here
      cout << "Config duplicate param error triggering OK\n";
    }
  }

  {
    ConfigParser cfg(dataPath + "test-duplicate.cfg", true);
    if(cfg.getString("logDir") != "more_logs")
      Global::fatalError("logDir param overriding in the same file error in data/test-duplicate.cfg");
    cout << "Config param overriding in the same file OK\n";
  }

  {
    try {
      ConfigParser cfg(dataPath + "test.cfg", false, false);
      Global::fatalError("Overriden param should trigger a error "
                         "when key overriding is disabled in data/test.cfg");
    } catch (const ConfigParsingError&) {
      // expected behaviour, do nothing here
      cout << "Config overriding error triggering OK\n";
    }
  }

  {
    ConfigParser cfg(dataPath + "test.cfg");
    if(!cfg.contains("reportAnalysisWinratesAs"))
      Global::fatalError("Config reading error from included file in a subdirectory "
                         "(data/folded/analysis_example.cfg) in data/test.cfg");
    if(cfg.getInt("maxVisits") != 1000)
      Global::fatalError("Config value (maxVisits) overriding error from "
                         "data/test1.cfg in data/test.cfg");
    if(cfg.getString("logDir") != "more_logs")
      Global::fatalError("logDir param overriding error in data/test.cfg");
    if(cfg.getInt("nnMaxBatchSize") != 100500)
      Global::fatalError("nnMaxBatchSize param overriding error in data/test.cfg");
    cout << "Config overriding test OK\n";
  }

  // circular dependency test
  {
    try {
      ConfigParser cfg(dataPath + "test-circular0.cfg");
      Global::fatalError("Config circular inclusion should trigger a error "
                         "in data/test-circular0.cfg");
    } catch (const ConfigParsingError&) {
      // expected behaviour, do nothing here
      cout << "Config circular inclusion error triggering OK\n";
    }
  }

  // config from parent dir inclusion test
  {
    ConfigParser cfg(dataPath + "folded/test-parent.cfg");
    if(cfg.getString("param") != "value")
      Global::fatalError("Config reading error from "
                         "data/folded/test-parent.cfg");
    if(cfg.getString("logDir") != "more_logs")
      Global::fatalError("logDir param reading error in data/test.cfg");
    cout << "Config inclusion from parent dir OK\n";
  }

  // multiple config files from command line
  {
    vector<string> testArgs = {
      "runconfigtests",
      "-config",
      dataPath + "analysis_example.cfg",
      "-config",
      dataPath + "test2.cfg"
    };
    ConfigParser cfg;
    KataGoCommandLine cmd("Run KataGo configuration file(s) unit-tests.");
    try {
      cmd.addConfigFileArg("", dataPath + "analysis_example.cfg");
      cmd.addOverrideConfigArg();

      cmd.parseArgs(testArgs);

      cmd.getConfig(cfg);

    } catch (TCLAP::ArgException &e) {
      cerr << "Error: " << e.error() << " for argument " << e.argId() << endl;
      Global::fatalError("Wrong command-line parameters");
    }

    if (!cfg.contains("logDir"))
      Global::fatalError("logDir param reading error from analysis_example.cfg "
                         "while reading multiple configs from command line "
                         "(data/analysis_example.cfg and data/test2.cfg)");

    if(cfg.getInt("nnMaxBatchSize") != 100)
      Global::fatalError("nnMaxBatchSize param overriding error while reading "
                         "multiple configs from command line "
                         "(data/analysis_example.cfg and data/test2.cfg)");

    cout << "Config overriding from command line OK\n";
  }
}
