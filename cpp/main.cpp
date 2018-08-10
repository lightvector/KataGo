#include "main.h"

using namespace std;

static void printHelp() {
  cout << "Available subcommands:" << endl;
  cout << "evalSgf" << endl;
  cout << "gtp" << endl;
  cout << "match" << endl;
  cout << "runTests" << endl;
  cout << "sandbox" << endl;
}

int main(int argc, const char* argv[]) {
  if(argc < 2) {
    printHelp();
    return 0;
  }
  string cmdArg = string(argv[1]);
  if(cmdArg == "-h" || cmdArg == "--h" || cmdArg == "-help" || cmdArg == "--help" || cmdArg == "help") {
    printHelp();
    return 0;
  }

  if(cmdArg == "evalSgf")
    return MainCmds::evalSgf(argc-1,&argv[1]);
  else if(cmdArg == "gtp")
    return MainCmds::gtp(argc-1,&argv[1]);
  else if(cmdArg == "match")
    return MainCmds::match(argc-1,&argv[1]);
  else if(cmdArg == "runTests")
    return MainCmds::runTests(argc-1,&argv[1]);
  else if(cmdArg == "sandbox")
    return MainCmds::sandbox();
  else {
    cout << "Unknown subcommand: " << cmdArg << endl;
    printHelp();
    return 1;
  }
  return 0;
}
