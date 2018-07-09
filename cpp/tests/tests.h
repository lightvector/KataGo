#ifndef TESTS_H
#define TESTS_H

#include <sstream>
#include "core/global.h"
#include "core/rand.h"
#include "game/board.h"
#include "game/rules.h"
#include "game/boardhistory.h"

namespace Tests {
  //testmisc.cpp
  void runRandHashTests();

  //testboardbasic.cpp
  void runBoardBasicTests();
  void runBoardUndoTest();
  void runBoardStressTest();

  //testboardarea.cpp
  void runBoardAreaTests();

  //testrules.cpp
  void runRulesTests();
}


//A version of assert that's always defined, regardless of NDEBUG
#define testAssert(EX) (void)((EX) || (TestCommon::testAssertFailed(#EX, __FILE__, __LINE__),0))

namespace TestCommon {
  inline void testAssertFailed(const char *msg, const char *file, int line) {
    Global::fatalError(string("Failed test assert: ") + string(msg) + "\n" + string("file: ") + string(file) + "\n" + string("line: ") + Global::intToString(line));
  }

  inline void expect(const char* name, ostringstream& out, const string& expected) {
    if(Global::trim(out.str()) != Global::trim(expected)) {
      cout << "Expect test failure!" << endl;
      cout << name << endl;
      cout << "Expected===============================================================" << endl;
      cout << expected << endl;
      cout << "Got====================================================================:" << endl;
      cout << out.str() << endl;
      exit(1);
    }
  }

  inline bool boardColorsEqual(const Board& b1, const Board& b2) {
    for(int i = 0; i<Board::MAX_ARR_SIZE; i++)
      if(b1.colors[i] != b2.colors[i])
        return false;
    return true;
  }

  inline Board parseBoard(int xSize, int ySize, const string& s) {
    Board board(xSize,ySize,false);
    vector<string> lines = Global::split(Global::trim(s),'\n');
    assert(lines.size() == ySize);
    for(int y = 0; y<ySize; y++) {
      assert(lines[y].length() == xSize);
      for(int x = 0; x<xSize; x++) {
        char c = lines[y][x];
        Loc loc = Location::getLoc(x,y,board.x_size);
        if(c == '.' || c == ' ' || c == '*' || c == ',' || c == '`')
          continue;
        else if(c == 'o' || c == 'O')
          board.setStone(loc,P_WHITE);
        else if(c == 'x' || c == 'X')
          board.setStone(loc,P_BLACK);
        else
          assert(false);
      }
    }
    return board;
  }

}

#endif
