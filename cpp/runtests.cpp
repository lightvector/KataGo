
#include <sstream>
#include "core/global.h"
#include "core/rand.h"
#include "core/fancymath.h"
#include "game/board.h"
#include "game/rules.h"
#include "game/boardhistory.h"
#include "tests/tests.h"
#include "main.h"

int MainCmds::runTests(int argc, const char* const* argv) {
  (void)argc;
  (void)argv;
  testAssert(sizeof(size_t) == 8);
  Board::initHash();

  Rand::runTests();
  FancyMath::runTests();

  Tests::runBoardIOTests();
  Tests::runBoardBasicTests();
  Tests::runBoardAreaTests();

  Tests::runRulesTests();
  Tests::runScoreTests();

  Tests::runBoardUndoTest();
  Tests::runNNInputsTests();
  Tests::runBoardStressTest();

  cout << "All tests passed" << endl;
  return 0;
}

int MainCmds::runSearchTests(int argc, const char* const* argv) {
  Board::initHash();

  if(argc != 2) {
    cerr << "Must supply exactly one argument - the model file to use" << endl;
    return 1;
  }
  Tests::runSearchTests(string(argv[1]));
  return 0;
}
