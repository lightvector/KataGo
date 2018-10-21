
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
  Tests::runNNInputsV2Tests();
  Tests::runNNInputsV3Tests();
  Tests::runAutoSearchTests();
  Tests::runBoardStressTest();

  cout << "All tests passed" << endl;
  return 0;
}

int MainCmds::runSearchTests(int argc, const char* const* argv) {
  Board::initHash();

  if(argc != 5) {
    cerr << "Must supply exactly four arguments: MODEL_FILE INPUTSNHWC CUDANHWC SYMMETRY " << endl;
    return 1;
  }
  Tests::runSearchTests(
    string(argv[1]),
    Global::stringToBool(argv[2]),
    Global::stringToBool(argv[3]),
    Global::stringToInt(argv[4])
  );
  return 0;
}
