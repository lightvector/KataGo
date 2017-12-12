#include "core/global.h"
#include "core/rand.h"
#include "fastboard.h"
#include "sgf.h"

#define TCLAP_NAMESTARTSTRING "-" //Use single dashes for all flags
#include <tclap/CmdLine.h>

int main(int argc, const char* argv[]) {
  vector<string> gamesDirs;

  try {
    TCLAP::CmdLine cmd("Sgf->HDF5 data writer", ' ', "1.0");
    TCLAP::MultiArg<string> gamesdirArg("","gamesdir","Directory of sgf files",true,"DIR");
    cmd.add(gamesdirArg);
    cmd.parse(argc,argv);
    gamesDirs = gamesdirArg.getValue();
  }
  catch (TCLAP::ArgException &e) {
    cerr << "Error: " << e.error() << " for argument " << e.argId() << std::endl;
    return 1;
  }

  // XorShift1024Mult::test();
  // PCG32::test();
  // Rand::test();
  FastBoard::initHash();

  const string suffix = ".sgf";
  auto filter = [&suffix](const string& name) {
    return Global::isSuffix(name,suffix);
  };

  vector<string> files;
  for(int i = 0; i<gamesDirs.size(); i++)
    Global::collectFiles(gamesDirs[i], filter, files);
  cout << "Found " << files.size() << " sgf files!" << endl;

  vector<Sgf*> sgfs = Sgf::loadFiles(files);

  // FastBoard board;
  // Rand rand("foo");
  // for(int i = 0; i<10000000; i++) {
  //   int x = rand.nextUInt(19);
  //   int y = rand.nextUInt(19);
  //   Player p = (rand.nextUInt(2) == 0 ? P_BLACK : P_WHITE);
  //   Loc loc = Location::getLoc(x,y,19);
  //   board.playMove(loc,p);
  // }
  // cout << board << endl;
  return 0;
}
