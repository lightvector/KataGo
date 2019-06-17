#include "main.h"
#include "program/gitinfo.h"

using namespace std;

static void printHelp(int argc, const char* argv[]) {
  cout << endl;
  if(argc >= 1)
    cout << "Usage: " << argv[0] << " SUBCOMMAND ";
  else
    cout << "Usage: " << "./main" << " SUBCOMMAND ";
  cout << endl;

  cout << R"%%(
---Common subcommands------------------

gtp : Runs GTP engine that can be plugged into any standard Go GUI for play/analysis.
match : Run self-play match games based on a config, more efficient than gtp due to batching.
evalsgf : Utility/debug tool, analyze a single position of a game from an SGF file.
version : Print version and exit.

---Selfplay training subcommands---------

selfplay : Play selfplay games and generate training data.
gatekeeper : Poll directory for new nets and match them against the latest net so far.

---Testing/debugging subcommands-------------

runtests : Test important board algorithms and datastructures
runnnlayertests : Test a few subcomponents of the current neural net backend

runnnontinyboardtest : Run neural net on a tiny board and dump result to stdout

runoutputtests : Run a bunch of things and dump details to stdout
runsearchtests : Run a bunch of things using a neural net and dump details to stdout
runsearchtestsv3 : Run a bunch more things using a neural net and dump details to stdout
runselfplayinittests : Run some tests involving selfplay training init using a neural net and dump details to stdout

---Dev/experimental subcommands-------------
demoplay
lzcost
matchauto
sandbox
)%%" << endl;
}

int main(int argc, const char* argv[]) {
  if(argc < 2) {
    printHelp(argc,argv);
    return 0;
  }
  string cmdArg = string(argv[1]);
  if(cmdArg == "-h" || cmdArg == "--h" || cmdArg == "-help" || cmdArg == "--help" || cmdArg == "help") {
    printHelp(argc,argv);
    return 0;
  }

  if(cmdArg == "evalsgf")
    return MainCmds::evalsgf(argc-1,&argv[1]);
  else if(cmdArg == "gatekeeper")
    return MainCmds::gatekeeper(argc-1,&argv[1]);
  else if(cmdArg == "gtp")
    return MainCmds::gtp(argc-1,&argv[1]);
  else if(cmdArg == "match")
    return MainCmds::match(argc-1,&argv[1]);
  else if(cmdArg == "matchauto")
    return MainCmds::matchauto(argc-1,&argv[1]);
  else if(cmdArg == "selfplay")
    return MainCmds::selfplay(argc-1,&argv[1]);
  else if(cmdArg == "runtests")
    return MainCmds::runtests(argc-1,&argv[1]);
  else if(cmdArg == "runnnlayertests")
    return MainCmds::runnnlayertests(argc-1,&argv[1]);
  else if(cmdArg == "runnnontinyboardtest")
    return MainCmds::runnnontinyboardtest(argc-1,&argv[1]);
  else if(cmdArg == "runoutputtests")
    return MainCmds::runoutputtests(argc-1,&argv[1]);
  else if(cmdArg == "runsearchtests")
    return MainCmds::runsearchtests(argc-1,&argv[1]);
  else if(cmdArg == "runsearchtestsv3")
    return MainCmds::runsearchtestsv3(argc-1,&argv[1]);
  else if(cmdArg == "runselfplayinittests")
    return MainCmds::runselfplayinittests(argc-1,&argv[1]);
  else if(cmdArg == "lzcost")
    return MainCmds::lzcost(argc-1,&argv[1]);
  else if(cmdArg == "demoplay")
    return MainCmds::demoplay(argc-1,&argv[1]);
  else if(cmdArg == "sandbox")
    return MainCmds::sandbox();
  else if(cmdArg == "version") {
    cout << Version::getKataGoVersionForHelp() << endl;
    cout << "Git revision: " << Version::getGitRevision() << endl;
    cout << "Compile Time: " << __DATE__ << " " << __TIME__ << endl;
    #if defined(USE_CUDA_BACKEND)
    cout << "Using CUDA backend" << endl;
    #elif defined(USE_OPENCL_BACKEND)
    cout << "Using OpenCL backend" << endl;
    #else
    cout << "Using dummy backend" << endl;
    #endif
    return 0;
  }
  else {
    cout << "Unknown subcommand: " << cmdArg << endl;
    printHelp(argc,argv);
    return 1;
  }
  return 0;
}

string Version::getKataGoVersion() {
  return string("1.1");
}

string Version::getKataGoVersionForHelp() {
  return string("KataGo v1.1");
}

string Version::getGitRevision() {
  return string(GIT_REVISION);
}
