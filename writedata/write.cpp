#include "core/global.h"
#include "core/rand.h"
#include "fastboard.h"
#include "sgf.h"

#define TCLAP_NAMESTARTSTRING "-" //Use single dashes for all flags
#include <tclap/CmdLine.h>

static void processSgf(Sgf* sgf, vector<Move>& placementsBuf, vector<Move>& movesBuf) {
  int bSize;
  try {
    bSize = sgf->getBSize();

    //Apply some filters
    if(bSize != 19)
      return;

    sgf->getPlacements(placementsBuf,bSize);
    sgf->getMoves(movesBuf,bSize);
  }
  catch(const IOError &e) {
    cout << "Skipping sgf file: " << sgf->fileName << ": " << e.message << endl;
  }

  FastBoard board(bSize);
  for(int j = 0; j<placementsBuf.size(); j++) {
    Move m = placementsBuf[j];
    bool suc = board.setStone(m.loc,m.pla);
    if(!suc) {
      cout << sgf->fileName << endl;
      cout << ("Illegal stone placement " + Global::intToString(j)) << endl;
      cout << board << endl;
      return;
    }
  }

  //If there are multiple black moves in a row, then make them all right now.
  //Sometimes sgfs break the standard and do handicap setup in this way.
  int j = 0;
  if(movesBuf.size() > 1 && movesBuf[0].pla == P_BLACK && movesBuf[1].pla == P_BLACK) {
    for(; j<movesBuf.size(); j++) {
      Move m = movesBuf[j];
      if(m.pla != P_BLACK)
        break;
      bool suc = board.playMove(m.loc,m.pla);
      if(!suc) {
        cout << sgf->fileName << endl;
        cout << ("Illegal move! " + Global::intToString(j)) << endl;
        cout << board << endl;
      }
    }
  }

  Player prevPla = C_EMPTY;
  for(; j<movesBuf.size(); j++) {
    Move m = movesBuf[j];

    //Forbid consecutive moves by the same player
    if(m.pla == prevPla) {
      cout << sgf->fileName << endl;
      cout << ("Multiple moves in a row by same player at " + Global::intToString(j)) << endl;
      cout << board << endl;
    }

    bool suc = board.playMove(m.loc,m.pla);
    if(!suc) {
      cout << sgf->fileName << endl;
      cout << ("Illegal move! " + Global::intToString(j)) << endl;
      cout << board << endl;
    }

    prevPla = m.pla;
  }
}



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
  vector<Move> placementsBuf;
  vector<Move> movesBuf;
  for(int i = 0; i<sgfs.size(); i++) {
    Sgf* sgf = sgfs[i];
    processSgf(sgf, placementsBuf, movesBuf);
  }

  for(int i = 0; i<sgfs.size(); i++) {
    delete sgfs[i];
  }

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
