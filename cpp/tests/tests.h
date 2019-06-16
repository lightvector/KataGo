#ifndef TESTS_H
#define TESTS_H

#include <sstream>

#include "../core/global.h"
#include "../core/rand.h"
#include "../core/test.h"
#include "../game/board.h"
#include "../game/rules.h"
#include "../game/boardhistory.h"

namespace Tests {
  //testboardbasic.cpp
  void runBoardIOTests();
  void runBoardBasicTests();
  void runBoardUndoTest();
  void runBoardStressTest();

  //testboardarea.cpp
  void runBoardAreaTests();

  //testrules.cpp
  void runRulesTests();

  //testscore.cpp
  void runScoreTests();

  //testsgf.cpp
  void runSgfTests();

  //testnninputs.cpp
  void runNNInputsV3V4Tests();

  //testsearch.cpp
  void runNNLessSearchTests();
  void runSearchTests(const std::string& modelFile, bool inputsNHWC, bool cudaNHWC, int symmetry, bool useFP16);
  void runSearchTestsV3(const std::string& modelFile, bool inputsNHWC, bool cudaNHWC, int symmetry, bool useFP16);
  void runNNOnTinyBoard(const std::string& modelFile, bool inputsNHWC, bool cudaNHWC, int symmetry, bool useFP16);

  //testtime.cpp
  void runTimeControlsTests();

  //testtrainingwrite.cpp
  void runTrainingWriteTests();
  void runSelfplayInitTestsWithNN(const std::string& modelFile);

  //testnn.cpp
  void runNNLayerTests();
}

namespace TestCommon {

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
