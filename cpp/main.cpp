#include "main.h"

#include "core/os.h"

#ifdef NO_GIT_REVISION
#define GIT_REVISION "<omitted>"
#else
#include "program/gitinfo.h"
#endif

using namespace std;

static void printHelp(int argc, const char* argv[]) {
  cout << endl;
  if(argc >= 1)
    cout << "Usage: " << argv[0] << " SUBCOMMAND ";
  else
    cout << "Usage: " << "./katago" << " SUBCOMMAND ";
  cout << endl;

  cout << R"%%(
---Common subcommands------------------

gtp : Runs GTP engine that can be plugged into any standard Go GUI for play/analysis.
match : Run self-play match games based on a config, more efficient than gtp due to batching.
version : Print version and exit.

analysis : Runs an engine designed to analyze entire games in parallel.
benchmark : Test speed with different numbers of search threads.
tuner : (OpenCL only) Run tuning to find and optimize parameters that work on your GPU.

---Selfplay training subcommands---------

selfplay : Play selfplay games and generate training data.
gatekeeper : Poll directory for new nets and match them against the latest net so far.

---Testing/debugging subcommands-------------
evalsgf : Utility/debug tool, analyze a single position of a game from an SGF file.

runtests : Test important board algorithms and datastructures
runnnlayertests : Test a few subcomponents of the current neural net backend

runnnontinyboardtest : Run neural net on a tiny board and dump result to stdout
runnnsymmetriestest : Run neural net on a hardcoded rectangle board and dump symmetries result

runoutputtests : Run a bunch of things and dump details to stdout
runsearchtests : Run a bunch of things using a neural net and dump details to stdout
runsearchtestsv3 : Run a bunch more things using a neural net and dump details to stdout
runselfplayinittests : Run some tests involving selfplay training init using a neural net and dump details to stdout
runsekitrainwritetests : Run some tests involving seki train output

---Dev/experimental subcommands-------------
nnerror
demoplay
lzcost
matchauto
sandbox
)%%" << endl;
}

static int handleSubcommand(const string& subcommand, int argc, const char* argv[]) {
  if(subcommand == "analysis")
    return MainCmds::analysis(argc-1,&argv[1]);
  if(subcommand == "benchmark")
    return MainCmds::benchmark(argc-1,&argv[1]);
  if(subcommand == "evalsgf")
    return MainCmds::evalsgf(argc-1,&argv[1]);
  else if(subcommand == "gatekeeper")
    return MainCmds::gatekeeper(argc-1,&argv[1]);
  else if(subcommand == "gtp")
    return MainCmds::gtp(argc-1,&argv[1]);
  else if(subcommand == "tuner")
    return MainCmds::tuner(argc-1,&argv[1]);
  else if(subcommand == "match")
    return MainCmds::match(argc-1,&argv[1]);
  else if(subcommand == "matchauto")
    return MainCmds::matchauto(argc-1,&argv[1]);
  else if(subcommand == "selfplay")
    return MainCmds::selfplay(argc-1,&argv[1]);
  else if(subcommand == "runtests")
    return MainCmds::runtests(argc-1,&argv[1]);
  else if(subcommand == "runnnlayertests")
    return MainCmds::runnnlayertests(argc-1,&argv[1]);
  else if(subcommand == "runnnontinyboardtest")
    return MainCmds::runnnontinyboardtest(argc-1,&argv[1]);
  else if(subcommand == "runnnsymmetriestest")
    return MainCmds::runnnsymmetriestest(argc-1,&argv[1]);
  else if(subcommand == "runoutputtests")
    return MainCmds::runoutputtests(argc-1,&argv[1]);
  else if(subcommand == "runsearchtests")
    return MainCmds::runsearchtests(argc-1,&argv[1]);
  else if(subcommand == "runsearchtestsv3")
    return MainCmds::runsearchtestsv3(argc-1,&argv[1]);
  else if(subcommand == "runselfplayinittests")
    return MainCmds::runselfplayinittests(argc-1,&argv[1]);
  else if(subcommand == "runsekitrainwritetests")
    return MainCmds::runsekitrainwritetests(argc-1,&argv[1]);
  else if(subcommand == "runnnonmanyposestest")
    return MainCmds::runnnonmanyposestest(argc-1,&argv[1]);
  else if(subcommand == "lzcost")
    return MainCmds::lzcost(argc-1,&argv[1]);
  else if(subcommand == "demoplay")
    return MainCmds::demoplay(argc-1,&argv[1]);
  else if(subcommand == "sandbox")
    return MainCmds::sandbox();
  else if(subcommand == "version") {
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
    cout << "Unknown subcommand: " << subcommand << endl;
    printHelp(argc,argv);
    return 1;
  }
  return 0;
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

#if defined(OS_IS_WINDOWS)
  //On windows, uncaught exceptions reaching toplevel don't normally get printed out,
  //so explicitly catch everything and print
  int result;
  try {
    result = handleSubcommand(cmdArg, argc, argv);
  }
  catch(std::exception& e) {
    cout << "Uncaught exception: " << e.what() << endl;
    return 1;
  }
  catch(...) {
    cout << "Uncaught exception that is not a std::exception... exiting due to unknown error" << endl;
    return 1;
  }
  return result;
#else
  return handleSubcommand(cmdArg, argc, argv);
#endif
}


string Version::getKataGoVersion() {
  return string("1.2");
}

string Version::getKataGoVersionForHelp() {
  return string("KataGo v1.2");
}

string Version::getGitRevision() {
  return string(GIT_REVISION);
}
