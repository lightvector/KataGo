#include "../tests/tests.h"

#include "../core/config_parser.h"
#include "../core/fileutils.h"
#include "../core/mainargs.h"
#include "../core/test.h"
#include "../command/commandline.h"

using namespace std;
using namespace TestCommon;

void Tests::runInlineConfigTests() {
  {
    string s = R"%%(
)%%";
    istringstream in(s);
    ConfigParser cfg(in);
    string expected = "";
    testAssert(expected == cfg.getAllKeyVals());
  }
  {
    string s = R"%%(
a1 = k2
#comment
 #comment
  #= == == ayay
  #a = b
  b1 = c5
_c_ = 43
d_= 5
e=6
f =7
abc =    def
bcd    =  g#foo
c-de =  g  #"test's"=== =
_a = "quoted"
_b= "quoted "  #hmm##
 _c =" quoted "
_d =" some # symbols \" yay " # later comment
 _e  = "\"\"\\"  # comment
# _f  = "\"\"\\"  # comment
key =  with spaces
quotes =  i'm a value " with " quotes! # hmmm"!
 test=back\slashes don't \escape \\here\
 test2=back\slashes don't \escape \\here\#comment
)%%";
    istringstream in(s);
    ConfigParser cfg(in);
    string expected =
      "_a = quoted" "\n"
      "_b = quoted " "\n"
      "_c =  quoted " "\n"
      "_c_ = 43" "\n"
      "_d =  some # symbols \" yay " "\n"
      "_e = \"\"\\" "\n"
      "a1 = k2" "\n"
      "abc = def" "\n"
      "b1 = c5" "\n"
      "bcd = g" "\n"
      "c-de = g" "\n"
      "d_ = 5" "\n"
      "e = 6" "\n"
      "f = 7" "\n"
      "key = with spaces" "\n"
      "quotes = i'm a value \" with \" quotes!" "\n"
      "test = back\\slashes don't \\escape \\\\here\\" "\n"
      "test2 = back\\slashes don't \\escape \\\\here\\" "\n"
      ;
    testAssert(expected == cfg.getAllKeyVals());
  }

  auto isCfgFail = [](const string& s) {
    istringstream in(s);
    bool failed = false;
    try {
      ConfigParser cfg(in);
    }
    catch(const StringError&) {
      failed = true;
    }
    return failed;
  };

  {
    string s = R"%%(
abc
)%%";
    testAssert(isCfgFail(s));
  }
  {
    string s = R"%%(
abc =
)%%";
    testAssert(isCfgFail(s));
  }
  {
    string s = R"%%(
abc = # comment
)%%";
    testAssert(isCfgFail(s));
  }
  {
    string s = R"%%(
abc = ""
)%%";
    testAssert(isCfgFail(s));
  }
  {
    string s = R"%%(
abc = ""def
)%%";
    testAssert(isCfgFail(s));
  }
  {
    string s = R"%%(
abc = "data"def
)%%";
    testAssert(isCfgFail(s));
  }
  {
    string s = R"%%(
abc = "data" def
)%%";
    testAssert(isCfgFail(s));
  }
  {
    string s = R"%%(
abc = "data"# def
)%%";
    testAssert(!isCfgFail(s));
  }
  {
    string s = R"%%(
abc = "data" #def
)%%";
    testAssert(!isCfgFail(s));
  }
  {
    string s = R"%%(
 =
)%%";
    testAssert(isCfgFail(s));
  }
  {
    string s = R"%%(
=
)%%";
    testAssert(isCfgFail(s));
  }
  {
    string s = R"%%(
= # foo
)%%";
    testAssert(isCfgFail(s));
  }
  {
    string s = R"%%(
"abc" = def
)%%";
    testAssert(isCfgFail(s));
  }
  {
    string s = R"%%(
a!b = def
)%%";
    testAssert(isCfgFail(s));
  }
  {
    string s = R"%%(
a#b = def
)%%";
    testAssert(isCfgFail(s));
  }
  {
    string s = R"%%(
a$b = def
)%%";
    testAssert(isCfgFail(s));
  }
  {
    string s = R"%%(
a%b = def
)%%";
    testAssert(isCfgFail(s));
  }
  {
    string s = R"%%(
a@b = def
)%%";
    testAssert(isCfgFail(s));
  }
  {
    string s = R"%%(
#ab = def
)%%";
    testAssert(!isCfgFail(s));
  }
  {
    string s = R"%%(
0ab = def
)%%";
    testAssert(!isCfgFail(s));
  }
  {
    string s = R"%%(
!ab = def
)%%";
    testAssert(isCfgFail(s));
  }
  {
    string s = R"%%(
a-x = c-y
)%%";
    testAssert(!isCfgFail(s));
  }
  {
    string s = R"%%(
notrailing = newline is okay)%%";
    testAssert(!isCfgFail(s));
  }
}

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

      const bool logToStdoutDefault = true;
      Logger logger(&cfg, logToStdoutDefault);
    } catch (TCLAP::ArgException &e) {
      cerr << "Error: " << e.error() << " for argument " << e.argId() << endl;
      Global::fatalError("Wrong command-line parameters");
    }
    return;
  }

  std::string dataPath("cpp/tests/data/configs/");
  if(!FileUtils::exists("cpp/tests/data/configs/"))
    dataPath = "tests/data/configs/";

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

    if(!cfg.contains("logDir"))
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

void Tests::runParseAllConfigsTest() {
  std::vector<std::string> collected;
  FileUtils::collectFiles("./configs/", [](const std::string& s) {return Global::isSuffix(s,".cfg");}, collected);
  for(const string& cfgPath: collected) {
    if(cfgPath.find(string("ringmaster")) != string::npos)
      continue;
    ConfigParser cfg(cfgPath);
    if(!cfg.contains("password")) {
      cout << "======================================================" << endl;
      cout << cfgPath << endl;
      cout << cfg.getAllKeyVals() << endl;
    }
  }
}
