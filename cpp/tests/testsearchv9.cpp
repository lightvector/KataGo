#include "../tests/tests.h"

#include <algorithm>
#include <iterator>
#include <iomanip>

#include "../core/fileutils.h"
#include "../dataio/sgf.h"
#include "../neuralnet/nninputs.h"
#include "../search/asyncbot.h"
#include "../program/playutils.h"
#include "../tests/testsearchcommon.h"

using namespace std;
using namespace TestCommon;
using namespace TestSearchCommon;


static void runV9Positions(NNEvaluator* nnEval, Logger& logger)
{
  {
    SearchParams params = SearchParams::forTestsV2();
    params.maxVisits = 100;

    AsyncBot* bot = new AsyncBot(params, nnEval, &logger, getSearchRandSeed());
    Rules rules = Rules::parseRules("Japanese");
    TestSearchOptions opts;
    opts.printPlaySelectionValues = true;

    {
      cout << "Flying dagger fight with V2 params and much variety ==========================================================================" << endl;
      cout << endl;

      string sgfStr = "(;GM[1]FF[4]CA[UTF-8]AP[CGoban:3]ST[2]RU[Japanese]SZ[19]KM[6.50]PW[White]PB[Black];B[pd];W[dd];B[pp];W[dp];B[cq];W[dq];B[cp];W[cn];B[bn];W[bm];B[co];W[dn];B[do];W[eo];B[ep];W[er];B[fp];W[fo];B[go];W[gn];B[gq];W[cr];B[bo];W[cl];B[br];W[gr];B[hr];W[bs];B[bq];W[fr];B[hn];W[ho];B[gp];W[ir];B[hs];W[ap];B[fn];W[en];B[gm];W[nc];B[dl];W[dk];B[em];W[el];B[al];W[bk];B[am];W[ao])";

      for(int i = 15; i<47; i++) {
        params.useNonBuggyLcb = i % 3 != 0;
        params.rootNoiseEnabled = i % 5 == 0;
        params.lcbStdevs = (i % 7 == 0) ? 2.5 : 5.0;
        params.minVisitPropForLCB = (i % 7 == 0) ? 0.03 : 0.15;
        params.useLcbForSelection = (i % 7 == 1) ? false : true;
        params.playoutDoublingAdvantage = (i % 11 == 0) ? 0.75 : 0.0;
        bot->setParams(params);
        runBotOnSgf(bot, sgfStr, rules, i, 6.5, opts);
      }
      cout << endl << endl;
    }

    {
      cout << "3-r pincer fight with V2 params and much variety ==========================================================================" << endl;
      cout << endl;

      string sgfStr = "(;GM[1]FF[4]CA[UTF-8]AP[CGoban:3]ST[2]RU[Japanese]SZ[19]KM[6.50]PW[White]PB[Black];B[pd];W[dd];B[pp];W[cp];B[eq];W[hq];B[do];W[dp];B[ep];W[eo];B[dn];W[en];B[bo];W[bp];B[dm];W[em];B[dl];W[dq];B[gp];W[hp];B[er];W[go];B[cr];W[dr];B[ds];W[br];B[bs];W[aq];B[gr];W[hr])";

      for(int i = 11; i<30; i++) {
        params.useNonBuggyLcb = i % 3 != 0;
        params.rootNoiseEnabled = i % 5 == 0;
        params.lcbStdevs = (i % 7 < 4) ? 2.5 : 5.0;
        params.minVisitPropForLCB = (i % 7 < 4) ? 0.03 : 0.15;
        params.useLcbForSelection = (i % 7 > 5) ? false : true;
        params.playoutDoublingAdvantage = (i % 11 < 3) ? 0.75 : 0.0;
        bot->setParams(params);
        runBotOnSgf(bot, sgfStr, rules, i, 6.5, opts);
      }
      cout << endl << endl;
    }

    delete bot;
  }

  {
    cout << "Pruned root values test ==========================================================================" << endl;
    cout << endl;

    Board board = Board::parseBoard(13,13,R"%%(
.xoxo.o......
ooox.oxox....
ooxxxxxo.x...
oxxoooxo.....
ooxo.ooxx.x..
ooox.........
xoxx..o.x....
xx...x.o..x..
..x..........
.......o.xx..
..ox..o...oo.
.ox.xxxoo....
.............
)%%");
    Player nextPla = P_WHITE;
    Rules rules = Rules::parseRules("Chinese");
    BoardHistory hist(board,nextPla,rules,0);

    {
      SearchParams params = SearchParams::forTestsV2();
      params.maxVisits = 600;
      params.rootFpuReductionMax = 0;
      AsyncBot* bot = new AsyncBot(params, nnEval, &logger, getSearchRandSeed());
      TestSearchOptions opts;
      opts.printPlaySelectionValues = true;
      opts.printRootValues = true;
      opts.printPrunedRootValues = true;
      runBotOnPosition(bot, board, nextPla, hist, opts);
      delete bot;
    }
    {
      SearchParams params = SearchParams::forTestsV2();
      params.maxVisits = 600;
      params.rootNoiseEnabled = true;
      params.rootFpuReductionMax = 0;
      AsyncBot* bot = new AsyncBot(params, nnEval, &logger, getSearchRandSeed());
      TestSearchOptions opts;
      opts.printPlaySelectionValues = true;
      opts.printRootValues = true;
      opts.printPrunedRootValues = true;
      runBotOnPosition(bot, board, nextPla, hist, opts);
      delete bot;
    }

  }

  {
    cout << "Conservative passing with pass as hint loc ==========================================================================" << endl;
    cout << endl;

    Board board = Board::parseBoard(9,9,R"%%(
.ox.xo.xx
oxx.xooo.
.xo.x.xoo
xxo...xxx
xxxx.....
ooox..xx.
.xox.xxoo
o.ox..oox
.oxx..ox.
)%%");
    Player nextPla = P_WHITE;

    {
      Rules rules = Rules::parseRules("Chinese");
      BoardHistory hist(board,nextPla,rules,0);

      SearchParams params = SearchParams::forTestsV2();
      params.maxVisits = 75;
      params.rootFpuReductionMax = 0;
      params.conservativePass = false;
      AsyncBot* bot = new AsyncBot(params, nnEval, &logger, getSearchRandSeed());
      TestSearchOptions opts;
      opts.numMovesInARow = 25;
      opts.rootHintLoc = Board::PASS_LOC;
      cout << "Conservative pass false" << endl;
      runBotOnPosition(bot, board, nextPla, hist, opts);
      delete bot;
    }

    {
      Rules rules = Rules::parseRules("Chinese");
      BoardHistory hist(board,nextPla,rules,0);

      SearchParams params = SearchParams::forTestsV2();
      params.maxVisits = 75;
      params.rootFpuReductionMax = 0;
      params.conservativePass = true;
      AsyncBot* bot = new AsyncBot(params, nnEval, &logger, getSearchRandSeed());
      TestSearchOptions opts;
      opts.numMovesInARow = 25;
      opts.rootHintLoc = Board::PASS_LOC;
      cout << "Conservative pass true" << endl;
      runBotOnPosition(bot, board, nextPla, hist, opts);
      delete bot;
    }

    {
      Rules rules = Rules::parseRules("Japanese");
      BoardHistory hist(board,nextPla,rules,0);

      SearchParams params = SearchParams::forTestsV2();
      params.maxVisits = 75;
      params.rootFpuReductionMax = 0;
      params.conservativePass = false;
      AsyncBot* bot = new AsyncBot(params, nnEval, &logger, getSearchRandSeed());
      TestSearchOptions opts;
      opts.numMovesInARow = 25;
      opts.rootHintLoc = Board::PASS_LOC;
      cout << "Conservative pass false jp rules" << endl;
      runBotOnPosition(bot, board, nextPla, hist, opts);
      delete bot;
    }

    {
      Rules rules = Rules::parseRules("Japanese");
      BoardHistory hist(board,nextPla,rules,0);

      SearchParams params = SearchParams::forTestsV2();
      params.maxVisits = 75;
      params.rootFpuReductionMax = 0;
      params.conservativePass = true;
      AsyncBot* bot = new AsyncBot(params, nnEval, &logger, getSearchRandSeed());
      TestSearchOptions opts;
      opts.numMovesInARow = 25;
      opts.rootHintLoc = Board::PASS_LOC;
      cout << "Conservative pass true jp rules" << endl;
      runBotOnPosition(bot, board, nextPla, hist, opts);
      delete bot;
    }

  }

  {
    cout << "Phase change truncating ladder history in spight rules ==========================================================================" << endl;
    cout << endl;

    auto printLadderFeaturesV7 = [](const Board& board, const BoardHistory& hist, bool conservativePass) {
      int nnXLen = 7;
      int nnYLen = 7;
      bool inputsUseNHWC = false;
      float* rowBin = new float[NNInputs::NUM_FEATURES_SPATIAL_V7 * nnXLen * nnYLen];
      float* rowGlobal = new float[NNInputs::NUM_FEATURES_GLOBAL_V7];

      MiscNNInputParams nnInputParams;
      nnInputParams.drawEquivalentWinsForWhite = 0.5;
      nnInputParams.conservativePassAndIsRoot = conservativePass;
      NNInputs::fillRowV7(board,hist,hist.presumedNextMovePla,nnInputParams,nnXLen,nnYLen,inputsUseNHWC,rowBin,rowGlobal);

      cout << "Ladder feature 14" << endl;
      for(int y = 0; y<nnYLen; y++) {
        for(int x = 0; x<nnXLen; x++) {
          cout << rowBin[14 * nnXLen * nnYLen + y * nnXLen + x] << " ";
        }
        cout << endl;
      }
      cout << "Ladder feature 15" << endl;
      for(int y = 0; y<nnYLen; y++) {
        for(int x = 0; x<nnXLen; x++) {
          cout << rowBin[15 * nnXLen * nnYLen + y * nnXLen + x] << " ";
        }
        cout << endl;
      }
      cout << "Ladder feature 16" << endl;
      for(int y = 0; y<nnYLen; y++) {
        for(int x = 0; x<nnXLen; x++) {
          cout << rowBin[16 * nnXLen * nnYLen + y * nnXLen + x] << " ";
        }
        cout << endl;
      }
      delete[] rowBin;
      delete[] rowGlobal;
    };

    Board board = Board::parseBoard(7,7,R"%%(
....xo.
....xo.
....xo.
....xoo
.xxxxo.
.xoooox
.xo.xx.
)%%");
    Player nextPla = P_BLACK;

    {
      Rules rules = Rules::parseRules("Japanese");
      rules.koRule = Rules::KO_SPIGHT;
      BoardHistory hist(board,nextPla,rules,0);
      hist.makeBoardMoveAssumeLegal(board,Location::ofString("G3",board),P_BLACK,NULL);
      hist.makeBoardMoveAssumeLegal(board,Location::ofString("G1",board),P_WHITE,NULL);
      hist.makeBoardMoveAssumeLegal(board,Location::ofString("G2",board),P_BLACK,NULL);
      hist.makeBoardMoveAssumeLegal(board,Location::ofString("pass",board),P_WHITE,NULL);
      hist.makeBoardMoveAssumeLegal(board,Location::ofString("G3",board),P_BLACK,NULL);
      hist.makeBoardMoveAssumeLegal(board,Location::ofString("G1",board),P_WHITE,NULL);
      hist.makeBoardMoveAssumeLegal(board,Location::ofString("G2",board),P_BLACK,NULL);
      hist.makeBoardMoveAssumeLegal(board,Location::ofString("pass",board),P_WHITE,NULL);

      SearchParams params = SearchParams::forTestsV2();
      params.maxVisits = 100;
      params.rootFpuReductionMax = 0;
      params.conservativePass = false;
      AsyncBot* bot = new AsyncBot(params, nnEval, &logger, getSearchRandSeed());
      TestSearchOptions opts;
      cout << "Conservative pass false" << endl;
      runBotOnPosition(bot, board, nextPla, hist, opts);
      printLadderFeaturesV7(board, hist, false);
      delete bot;
    }
    {
      Rules rules = Rules::parseRules("Japanese");
      rules.koRule = Rules::KO_SPIGHT;
      BoardHistory hist(board,nextPla,rules,0);
      hist.makeBoardMoveAssumeLegal(board,Location::ofString("G3",board),P_BLACK,NULL);
      hist.makeBoardMoveAssumeLegal(board,Location::ofString("G1",board),P_WHITE,NULL);
      hist.makeBoardMoveAssumeLegal(board,Location::ofString("G2",board),P_BLACK,NULL);
      hist.makeBoardMoveAssumeLegal(board,Location::ofString("pass",board),P_WHITE,NULL);
      hist.makeBoardMoveAssumeLegal(board,Location::ofString("G3",board),P_BLACK,NULL);
      hist.makeBoardMoveAssumeLegal(board,Location::ofString("G1",board),P_WHITE,NULL);
      hist.makeBoardMoveAssumeLegal(board,Location::ofString("G2",board),P_BLACK,NULL);
      hist.makeBoardMoveAssumeLegal(board,Location::ofString("pass",board),P_WHITE,NULL);

      SearchParams params = SearchParams::forTestsV2();
      params.maxVisits = 100;
      params.rootFpuReductionMax = 0;
      params.conservativePass = true;
      AsyncBot* bot = new AsyncBot(params, nnEval, &logger, getSearchRandSeed());
      TestSearchOptions opts;
      cout << "Conservative pass false" << endl;
      runBotOnPosition(bot, board, nextPla, hist, opts);
      printLadderFeaturesV7(board, hist, true);
      delete bot;
    }
  }

  {
    cout << "Passing details ==========================================================================" << endl;
    cout << endl;

    Board board = Board::parseBoard(8,8,R"%%(
..ooo...
x.ox.xx.
ooox.x..
.xx..xxo
..xxxoo.
xxxooo..
xoo.....
.o.o....
)%%");
    Player nextPla = P_BLACK;

    {
      cout << "Area scoring no friendly pass ok" << endl;
      Rules rules = Rules::parseRules("Chinese");
      rules.friendlyPassOk = false;
      BoardHistory hist(board,nextPla,rules,0);

      SearchParams params = SearchParams::forTestsV2();
      params.maxVisits = 50;
      params.rootFpuReductionMax = 0;
      AsyncBot* bot = new AsyncBot(params, nnEval, &logger, getSearchRandSeed());
      TestSearchOptions opts;
      opts.rootHintLoc = Board::PASS_LOC;
      opts.printMore = true;
      runBotOnPosition(bot, board, nextPla, hist, opts);
      delete bot;
    }

    {
      cout << "Area scoring yes friendly pass ok" << endl;
      Rules rules = Rules::parseRules("Chinese");
      rules.friendlyPassOk = true;
      BoardHistory hist(board,nextPla,rules,0);

      SearchParams params = SearchParams::forTestsV2();
      params.maxVisits = 50;
      params.rootFpuReductionMax = 0;
      AsyncBot* bot = new AsyncBot(params, nnEval, &logger, getSearchRandSeed());
      TestSearchOptions opts;
      opts.rootHintLoc = Board::PASS_LOC;
      opts.printMore = true;
      runBotOnPosition(bot, board, nextPla, hist, opts);
      delete bot;
    }

    {
      cout << "Area scoring no friendly pass ok but pass hacks" << endl;
      Rules rules = Rules::parseRules("Chinese");
      rules.friendlyPassOk = false;
      BoardHistory hist(board,nextPla,rules,0);

      SearchParams params = SearchParams::forTestsV2();
      params.maxVisits = 50;
      params.rootFpuReductionMax = 0;
      params.enablePassingHacks = true;
      AsyncBot* bot = new AsyncBot(params, nnEval, &logger, getSearchRandSeed());
      TestSearchOptions opts;
      opts.rootHintLoc = Board::PASS_LOC;
      opts.printMore = true;
      runBotOnPosition(bot, board, nextPla, hist, opts);
      delete bot;
    }

  }

}

void Tests::runSearchTestsV9(const string& modelFile, bool inputsNHWC, bool useNHWC, bool useFP16) {
  TestCommon::overrideForBackends(inputsNHWC, useNHWC);
  cout << "Running v9 search tests" << endl;
  NeuralNet::globalInitialize();

  const bool logToStdout = true;
  const bool logToStderr = false;
  const bool logTime = false;
  Logger logger(nullptr, logToStdout, logToStderr, logTime);

  int symmetry = 4;
  NNEvaluator* nnEval = startNNEval(modelFile,logger,"",NNPos::MAX_BOARD_LEN,NNPos::MAX_BOARD_LEN,symmetry,inputsNHWC,useNHWC,useFP16,false,false);
  runV9Positions(nnEval, logger);
  delete nnEval;

  NeuralNet::globalCleanup();
  cout << "Done" << endl;
}


