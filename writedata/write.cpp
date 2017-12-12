#include "core/global.h"
#include "core/rand.h"
#include "fastboard.h"

int main(int argc, const char* argv[]) {
  // XorShift1024Mult::test();
  // PCG32::test();
  // Rand::test();
  FastBoard::initHash();

  FastBoard board;
  Rand rand("foo");
  for(int i = 0; i<10000000; i++) {
    int x = rand.nextUInt(19);
    int y = rand.nextUInt(19);
    Player p = (rand.nextUInt(2) == 0 ? P_BLACK : P_WHITE);
    Loc loc = Location::getLoc(x,y,19);
    board.playMove(loc,p);
  }
  cout << board << endl;
  return 0;
}
