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
  void runBoardIOTests();
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

  inline bool boardsSeemEqual(const Board& b1, const Board& b2) {
    for(int i = 0; i<Board::MAX_ARR_SIZE; i++)
      if(b1.colors[i] != b2.colors[i])
        return false;
    if(b1.numBlackCaptures != b2.numBlackCaptures)
      return false;
    if(b1.numWhiteCaptures != b2.numWhiteCaptures)
      return false;
    return true;
  }

}

#endif
