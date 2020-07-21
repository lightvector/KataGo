#include "main.h"

#include "core/os.h"

#ifdef NO_GIT_REVISION
#define GIT_REVISION "<omitted>"
#else
#include "program/gitinfo.h"
#endif

#include <sstream>

using namespace std;

static void printHelp(int argc, const char* argv[]) {
  cout << endl;
  if(argc >= 1)
    cout << "Usage: " << argv[0] << " SUBCOMMAND ";
  else
    cout << "Usage: " << "./katago" << " SUBCOMMAND ";
  cout << endl;

  cout << R"%%(
  This version is a minified katago which only includes:
    analysis : Runs an engine designed to analyze entire games in parallel.
    version : Show version info.
)%%" << endl;
}

static int handleSubcommand(const string& subcommand, int argc, const char* argv[]) {
  if(subcommand == "analysis")
    return MainCmds::analysis(argc-1,&argv[1]);
  else if(subcommand == "version") {
    cout << Version::getKataGoVersionFullInfo() << std::flush;
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
  return string("1.4.5+kt1.3.3");
}

string Version::getKataGoVersionForHelp() {
  return string("KataGo v1.4.5+kt1.3.3 (minified)");
}

string Version::getKataGoVersionFullInfo() {
  ostringstream out;
  out << Version::getKataGoVersionForHelp() << endl;
  out << "Git revision: " << Version::getGitRevision() << endl;
  out << "Compile Time: " << __DATE__ << " " << __TIME__ << endl;
#if defined(USE_CUDA_BACKEND)
  out << "Using CUDA backend" << endl;
#elif defined(USE_OPENCL_BACKEND)
  out << "Using OpenCL backend" << endl;
#else
  out << "Using dummy backend" << endl;
#endif
  return out.str();
}

string Version::getGitRevision() {
  return string(GIT_REVISION);
}
