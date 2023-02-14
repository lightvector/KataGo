#ifndef TESTSEARCHCOMMON_H
#define TESTSEARCHCOMMON_H

#include "../core/logger.h"
#include "../game/boardhistory.h"
#include "../neuralnet/nneval.h"
#include "../search/asyncbot.h"

namespace TestSearchCommon {

  std::string getSearchRandSeed();

  struct TestSearchOptions {
    int numMovesInARow;
    bool printRootPolicy;
    bool printOwnership;
    bool printEndingScoreValueBonus;
    bool printPlaySelectionValues;
    bool printRootValues;
    bool printPrunedRootValues;
    bool noClearBot;
    bool noClearCache;
    bool printMore;
    bool printMoreMoreMore;
    bool printAfterBegun;
    bool ignorePosition;
    bool printPostOrderNodeCount;
    Loc rootHintLoc;
    TestSearchOptions();
  };

  void printPolicyValueOwnership(const Board& board, const NNResultBuf& buf);

  void printBasicStuffAfterSearch(const Board& board, const BoardHistory& hist, const Search* search, PrintTreeOptions options);

  void runBotOnPosition(AsyncBot* bot, Board board, Player nextPla, BoardHistory hist, TestSearchOptions opts);

  void runBotOnSgf(AsyncBot* bot, const std::string& sgfStr, const Rules& defaultRules, int turnIdx, float overrideKomi, TestSearchOptions opts);

  NNEvaluator* startNNEval(
    const std::string& modelFile, Logger& logger, const std::string& seed, int nnXLen, int nnYLen,
    int defaultSymmetry, bool inputsUseNHWC, bool useNHWC, bool useFP16, bool debugSkipNeuralNet,
    bool requireExactNNLen
  );

  void verifyTreePostOrder(Search* search, int onlyRequireAtLeast);
}

#endif
