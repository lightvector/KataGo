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
}

void Tests::runSearchTestsV9(const string& modelFile, bool inputsNHWC, bool useNHWC, bool useFP16) {
  TestCommon::overrideForBackends(inputsNHWC, useNHWC);
  cout << "Running v9 search tests" << endl;
  NeuralNet::globalInitialize();

  const bool logToStdOut = true;
  const bool logToStdErr = false;
  const bool logTime = false;
  Logger logger(nullptr, logToStdOut, logToStdErr, logTime);

  int symmetry = 4;
  NNEvaluator* nnEval = startNNEval(modelFile,logger,"",NNPos::MAX_BOARD_LEN,NNPos::MAX_BOARD_LEN,symmetry,inputsNHWC,useNHWC,useFP16,false,false);
  runV9Positions(nnEval, logger);
  delete nnEval;

  NeuralNet::globalCleanup();
  cout << "Done" << endl;
}


