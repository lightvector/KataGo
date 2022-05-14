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

static void runV8TestsSize9(NNEvaluator* nnEval, NNEvaluator* nnEval9, NNEvaluator* nnEval9Exact, Logger& logger)
{
  {
    cout << "TEST EXACT (NO MASKING) VS MASKED 9x9 ==========================================================================" << endl;
    string sgfStr = "(;FF[4]GM[1]SZ[9]HA[0]KM[7]RU[stonescoring];B[ef];W[ed];B[ge])";
    CompactSgf* sgf = CompactSgf::parse(sgfStr);

    Board board;
    Player nextPla;
    BoardHistory hist;
    Rules initialRules = sgf->getRulesOrFail();
    sgf->setupBoardAndHistAssumeLegal(initialRules, board, nextPla, hist, 3);

    SearchParams params = SearchParams::forTestsV1();
    params.maxVisits = 200;
    AsyncBot* botA = new AsyncBot(params, nnEval, &logger, "test exact again");
    AsyncBot* botB = new AsyncBot(params, nnEval9, &logger, "test exact again");
    AsyncBot* botC = new AsyncBot(params, nnEval9Exact, &logger, "test exact again");

    TestSearchOptions opts;
    cout << "BASIC" << endl;
    runBotOnPosition(botA,board,nextPla,hist,opts);
    cout << "BASIC9" << endl;
    runBotOnPosition(botB,board,nextPla,hist,opts);
    cout << "EXACT" << endl;
    runBotOnPosition(botC,board,nextPla,hist,opts);

    cout << endl << endl;

    delete botA;
    delete botB;
    delete botC;
    delete sgf;
  }
}

static void runV8TestsRandomSym(NNEvaluator* nnEval, NNEvaluator* nnEval19Exact, Logger& logger)
{
  {
    cout << "TEST EXACT (NO MASKING) VS MASKED ==========================================================================" << endl;

    string sgfStr = "(;GM[1]FF[4]CA[UTF-8]RU[Japanese]SZ[19]KM[6.5];B[dd];W[qd];B[pq];W[dp];B[oc];W[pe];B[fq];W[jp];B[ph];W[cf];B[ck])";
    CompactSgf* sgf = CompactSgf::parse(sgfStr);

    Board board;
    Player nextPla;
    BoardHistory hist;
    Rules initialRules = sgf->getRulesOrFail();
    sgf->setupBoardAndHistAssumeLegal(initialRules, board, nextPla, hist, 11);

    SearchParams params = SearchParams::forTestsV1();
    params.maxVisits = 200;
    AsyncBot* botA = new AsyncBot(params, nnEval, &logger, "test exact");
    AsyncBot* botB = new AsyncBot(params, nnEval19Exact, &logger, "test exact");

    TestSearchOptions opts;
    cout << "BASIC" << endl;
    runBotOnPosition(botA,board,nextPla,hist,opts);
    cout << "EXACT" << endl;
    runBotOnPosition(botB,board,nextPla,hist,opts);

    cout << endl << endl;

    delete botA;
    delete botB;
    delete sgf;
  }

  {
    cout << "TEST SYMMETRY AVGING ==========================================================================" << endl;
    //Reset random seeds for nnEval
    nnEval->killServerThreads();
    nnEval->spawnServerThreads();
    nnEval19Exact->killServerThreads();
    nnEval19Exact->spawnServerThreads();

    string sgfStr = "(;GM[1]FF[4]CA[UTF-8]RU[Japanese]SZ[19]KM[6.5];B[dd];W[qd];B[od];W[pq];B[dq];W[do];B[eo];W[oe])";
    CompactSgf* sgf = CompactSgf::parse(sgfStr);

    Board board;
    Player nextPla;
    BoardHistory hist;
    Rules initialRules = sgf->getRulesOrFail();
    sgf->setupBoardAndHistAssumeLegal(initialRules, board, nextPla, hist, 8);

    SearchParams params = SearchParams::forTestsV1();
    params.rootNumSymmetriesToSample = 8;
    params.maxVisits = 200;
    AsyncBot* botA = new AsyncBot(params, nnEval, &logger, "test exact");
    AsyncBot* botB = new AsyncBot(params, nnEval19Exact, &logger, "test exact");

    TestSearchOptions opts;
    cout << "BASIC" << endl;
    runBotOnPosition(botA,board,nextPla,hist,opts);
    cout << "EXACT" << endl;
    runBotOnPosition(botB,board,nextPla,hist,opts);

    cout << endl << endl;

    delete botA;
    delete botB;
    delete sgf;

    //Reset random seeds for nnEval
    nnEval->killServerThreads();
    nnEval->spawnServerThreads();
    nnEval19Exact->killServerThreads();
    nnEval19Exact->spawnServerThreads();
  }

  {
    cout << "TEST NN TEMPERATURE ==========================================================================" << endl;
    string sgfStr = "(;GM[1]FF[4]CA[UTF-8]RU[AGA]SZ[19]KM[7.0];B[dd];W[pd];B[dp];W[pp];B[qc];W[qd];B[pc];W[nc];B[nb])";
    CompactSgf* sgf = CompactSgf::parse(sgfStr);

    Board board;
    Player nextPla;
    BoardHistory hist;
    Rules initialRules = sgf->getRulesOrFail();
    sgf->setupBoardAndHistAssumeLegal(initialRules, board, nextPla, hist, 8);

    SearchParams paramsA = SearchParams::forTestsV1();
    SearchParams paramsB = SearchParams::forTestsV1();
    SearchParams paramsC = SearchParams::forTestsV1();
    paramsA.maxVisits = 200;
    paramsB.maxVisits = 200;
    paramsC.maxVisits = 200;
    paramsB.nnPolicyTemperature = 1.5;
    paramsC.nnPolicyTemperature = 0.5;
    AsyncBot* botA = new AsyncBot(paramsA, nnEval, &logger, "test exact");
    AsyncBot* botB = new AsyncBot(paramsB, nnEval, &logger, "test exact");
    AsyncBot* botC = new AsyncBot(paramsC, nnEval, &logger, "test exact");
    AsyncBot* botA2 = new AsyncBot(paramsA, nnEval19Exact, &logger, "test exact");
    AsyncBot* botB2 = new AsyncBot(paramsB, nnEval19Exact, &logger, "test exact");
    AsyncBot* botC2 = new AsyncBot(paramsC, nnEval19Exact, &logger, "test exact");

    TestSearchOptions opts;
    opts.printMore = true;
    nnEval->clearCache();
    nnEval->clearStats();
    opts.noClearCache = true;
    cout << "BASELINE" << endl;
    //Reset random seeds for nnEval
    nnEval->killServerThreads();
    nnEval->spawnServerThreads();
    nnEval19Exact->killServerThreads();
    nnEval19Exact->spawnServerThreads();
    runBotOnPosition(botA,board,nextPla,hist,opts);
    runBotOnPosition(botA2,board,nextPla,hist,opts);

    cout << "TEMP 1.5" << endl;
    //Reset random seeds for nnEval
    nnEval->killServerThreads();
    nnEval->spawnServerThreads();
    nnEval19Exact->killServerThreads();
    nnEval19Exact->spawnServerThreads();
    runBotOnPosition(botB,board,nextPla,hist,opts);
    runBotOnPosition(botB2,board,nextPla,hist,opts);

    cout << "TEMP 0.5" << endl;
    //Reset random seeds for nnEval
    nnEval->killServerThreads();
    nnEval->spawnServerThreads();
    nnEval19Exact->killServerThreads();
    nnEval19Exact->spawnServerThreads();
    runBotOnPosition(botC,board,nextPla,hist,opts);
    runBotOnPosition(botC2,board,nextPla,hist,opts);
    cout << endl << endl;

    delete botA;
    delete botB;
    delete botC;
    delete botA2;
    delete botB2;
    delete botC2;
    delete sgf;
  }
}

static void runV8Tests(NNEvaluator* nnEval, Logger& logger)
{
  {
    cout << "===================================================================" << endl;
    cout << "Testing PDA + pondering, p200 v400" << endl;
    cout << "===================================================================" << endl;

    Board board = Board::parseBoard(13,13,R"%%(
.............
.............
.............
.........x...
.............
.............
.............
.............
.............
..o......x...
.............
.............
.............
)%%");

    const Player startPla = P_WHITE;
    const Rules rules = Rules::getTrompTaylorish();
    const BoardHistory hist(board,startPla,rules,0);
    SearchParams baseParams = SearchParams::forTestsV1();
    baseParams.maxVisits = 400;
    baseParams.maxVisitsPondering = 600;
    baseParams.maxPlayouts = 200;
    baseParams.maxPlayoutsPondering = 300;

    nnEval->clearCache(); nnEval->clearStats();

    auto printSearchResults = [](const Search* search) {
      cout << search->rootBoard << endl;
      cout << "Root visits: " << search->getRootVisits() << "\n";
      cout << "Last search playouts: " << search->lastSearchNumPlayouts << "\n";
      cout << "NN rows: " << search->nnEvaluator->numRowsProcessed() << endl;
      cout << "NN batches: " << search->nnEvaluator->numBatchesProcessed() << endl;
      cout << "NN avg batch size: " << search->nnEvaluator->averageProcessedBatchSize() << endl;
      if(search->searchParams.playoutDoublingAdvantage != 0)
        cout << "PlayoutDoublingAdvantage: " << (
          search->getRootPla() == getOpp(search->getPlayoutDoublingAdvantagePla()) ?
          -search->searchParams.playoutDoublingAdvantage : search->searchParams.playoutDoublingAdvantage) << endl;
      cout << "PV: ";
      search->printPV(cout, search->rootNode, 25);
      cout << "\n";
      cout << "Tree:\n";
      PrintTreeOptions options;
      options = options.maxDepth(1);
      search->printTree(cout, search->rootNode, options, P_WHITE);
    };

    {
      SearchParams params = baseParams;
      params.playoutDoublingAdvantage = 1.5;
      cout << "Basic search with PDA 1.5, no player" << endl;

      Search* search = new Search(params, nnEval, &logger, "autoSearchRandSeed3");
      Player nextPla;

      search->setPosition(startPla,board,hist); nextPla = startPla;

      Loc moveLoc = search->runWholeSearchAndGetMove(nextPla); printSearchResults(search);

      cout << "Search next player - should clear tree and flip PDA" << endl;
      search->makeMove(moveLoc,nextPla); nextPla = getOpp(nextPla);
      moveLoc = search->runWholeSearchAndGetMove(nextPla); printSearchResults(search);

      cout << "Search next player - should clear tree and flip PDA" << endl;
      search->makeMove(moveLoc,nextPla); nextPla = getOpp(nextPla);
      search->runWholeSearch(nextPla); printSearchResults(search);

      delete search;
      nnEval->clearCache(); nnEval->clearStats();
    }

    {
      SearchParams params = baseParams;
      params.playoutDoublingAdvantage = 1.5;
      params.playoutDoublingAdvantagePla = P_BLACK;
      cout << "Basic search with PDA 1.5, force black" << endl;

      Search* search = new Search(params, nnEval, &logger, "autoSearchRandSeed3");
      Player nextPla;

      search->setPosition(startPla,board,hist); nextPla = startPla;

      Loc moveLoc = search->runWholeSearchAndGetMove(nextPla); printSearchResults(search);

      cout << "Search next player - should preserve tree" << endl;
      search->makeMove(moveLoc,nextPla); nextPla = getOpp(nextPla);
      moveLoc = search->runWholeSearchAndGetMove(nextPla); printSearchResults(search);

      cout << "Search next player - should preserve tree" << endl;
      search->makeMove(moveLoc,nextPla); nextPla = getOpp(nextPla);
      moveLoc = search->runWholeSearchAndGetMove(nextPla); printSearchResults(search);

      cout << "Search next player PONDERING - should preserve tree" << endl;
      search->makeMove(moveLoc,nextPla); nextPla = getOpp(nextPla);
      bool pondering = true;
      moveLoc = search->runWholeSearchAndGetMove(nextPla,pondering); printSearchResults(search);

      cout << "Search next player - should preserve tree" << endl;
      search->makeMove(moveLoc,nextPla); nextPla = getOpp(nextPla);
      search->runWholeSearch(nextPla); printSearchResults(search);

      delete search;
      nnEval->clearCache(); nnEval->clearStats();
    }

    {
      SearchParams params = baseParams;
      params.playoutDoublingAdvantage = 1.5;
      params.playoutDoublingAdvantagePla = P_WHITE;
      cout << "Basic search with PDA 1.5, force white" << endl;

      Search* search = new Search(params, nnEval, &logger, "autoSearchRandSeed3");
      Player nextPla;

      search->setPosition(startPla,board,hist); nextPla = startPla;

      Loc moveLoc = search->runWholeSearchAndGetMove(nextPla); printSearchResults(search);

      cout << "Search next player - should preserve tree" << endl;
      search->makeMove(moveLoc,nextPla); nextPla = getOpp(nextPla);
      moveLoc = search->runWholeSearchAndGetMove(nextPla); printSearchResults(search);

      cout << "Search next player - should preserve tree" << endl;
      search->makeMove(moveLoc,nextPla); nextPla = getOpp(nextPla);
      moveLoc = search->runWholeSearchAndGetMove(nextPla); printSearchResults(search);

      cout << "Search next player PONDERING - should preserve tree" << endl;
      search->makeMove(moveLoc,nextPla); nextPla = getOpp(nextPla);
      bool pondering = true;
      moveLoc = search->runWholeSearchAndGetMove(nextPla,pondering); printSearchResults(search);

      cout << "Search next player PONDERING - an extra time, should preserve tree" << endl;
      search->makeMove(moveLoc,nextPla); nextPla = getOpp(nextPla);
      pondering = true;
      moveLoc = search->runWholeSearchAndGetMove(nextPla,pondering); printSearchResults(search);

      cout << "Search next player - should preserve tree" << endl;
      search->makeMove(moveLoc,nextPla); nextPla = getOpp(nextPla);
      search->runWholeSearch(nextPla); printSearchResults(search);

      delete search;
      nnEval->clearCache(); nnEval->clearStats();
    }

    {
      SearchParams params = baseParams;
      params.playoutDoublingAdvantage = 1.5;
      cout << "Basic search with PDA 1.5, no player" << endl;

      Search* search = new Search(params, nnEval, &logger, "autoSearchRandSeed3");
      Player nextPla;

      search->setPosition(startPla,board,hist); nextPla = startPla;

      Loc moveLoc = search->runWholeSearchAndGetMove(nextPla); printSearchResults(search);

      cout << "Search next player PONDERING - should keep prior tree and PDA" << endl;
      search->makeMove(moveLoc,nextPla); nextPla = getOpp(nextPla);
      bool pondering = true;
      moveLoc = search->runWholeSearchAndGetMove(nextPla,pondering); printSearchResults(search);

      cout << "Search next player - should keep tree from ponder" << endl;
      search->makeMove(moveLoc,nextPla); nextPla = getOpp(nextPla);
      search->runWholeSearch(nextPla); printSearchResults(search);

      delete search;
      nnEval->clearCache(); nnEval->clearStats();
    }

    {
      SearchParams params = baseParams;
      params.playoutDoublingAdvantage = 1.5;
      cout << "Basic search with PDA 1.5, no player" << endl;

      Search* search = new Search(params, nnEval, &logger, "autoSearchRandSeed3");
      Player nextPla;

      search->setPosition(startPla,board,hist); nextPla = startPla;

      Loc moveLoc = search->runWholeSearchAndGetMove(nextPla); printSearchResults(search);

      cout << "Search next player PONDERING - should keep prior tree and PDA" << endl;
      search->makeMove(moveLoc,nextPla); nextPla = getOpp(nextPla);
      bool pondering = true;
      moveLoc = search->runWholeSearchAndGetMove(nextPla,pondering); printSearchResults(search);

      cout << "Search next player PONDERING an extra time - should keep prior tree and PDA" << endl;
      search->makeMove(moveLoc,nextPla); nextPla = getOpp(nextPla);
      pondering = true;
      moveLoc = search->runWholeSearchAndGetMove(nextPla,pondering); printSearchResults(search);

      cout << "Search next player - now should lose the tree and PDA, because the player it is for is different" << endl;
      search->makeMove(moveLoc,nextPla); nextPla = getOpp(nextPla);
      search->runWholeSearch(nextPla); printSearchResults(search);

      delete search;
      nnEval->clearCache(); nnEval->clearStats();
    }

    {
      SearchParams params = baseParams;
      params.playoutDoublingAdvantage = 1.5;
      cout << "Basic search with PDA 1.5, no player" << endl;

      Search* search = new Search(params, nnEval, &logger, "autoSearchRandSeed3");
      Player nextPla;

      search->setPosition(startPla,board,hist); nextPla = startPla;

      Loc moveLoc = search->runWholeSearchAndGetMove(nextPla); printSearchResults(search);

      cout << "Search next player PONDERING - should keep prior tree and PDA" << endl;
      search->makeMove(moveLoc,nextPla); nextPla = getOpp(nextPla);
      bool pondering = true;
      moveLoc = search->runWholeSearchAndGetMove(nextPla,pondering); printSearchResults(search);

      cout << "Search next player PONDERING - should still keep prior tree and PDA" << endl;
      search->makeMove(moveLoc,nextPla); nextPla = getOpp(nextPla);
      pondering = true;
      search->runWholeSearch(nextPla,pondering); printSearchResults(search);

      cout << "Without making a move - convert ponder to regular search, should still keep tree and PDA" << endl;
      search->runWholeSearch(nextPla); printSearchResults(search);

      nnEval->clearCache(); nnEval->clearStats();

      cout << "Set position to original, search PONDERING" << endl;
      search->setPosition(startPla,board,hist); nextPla = startPla;
      pondering = true;
      search->runWholeSearch(nextPla,pondering); printSearchResults(search);

      cout << "Without making a move, convert to regular search, should not keep tree" << endl;
      cout << "and should not benefit from cache, since search would guess the opponent as 'our' side" << endl;
      moveLoc = search->runWholeSearchAndGetMove(nextPla); printSearchResults(search);

      cout << "But should be fine thereafter. Make two moves and continue" << endl;
      search->makeMove(moveLoc,nextPla); nextPla = getOpp(nextPla);
      search->makeMove(Location::ofString("D4",board),nextPla); nextPla = getOpp(nextPla);
      search->runWholeSearch(nextPla); printSearchResults(search);

      nnEval->clearCache(); nnEval->clearStats();

      cout << "Set position to original, search PONDERING" << endl;
      search->setPosition(startPla,board,hist); nextPla = startPla;
      pondering = true;
      moveLoc = search->runWholeSearchAndGetMove(nextPla,pondering); printSearchResults(search);

      cout << "Play that move and real search on the next position, should keep tree because correct guess of side" << endl;
      search->makeMove(moveLoc,nextPla); nextPla = getOpp(nextPla);
      search->runWholeSearch(nextPla); printSearchResults(search);

      delete search;
      nnEval->clearCache(); nnEval->clearStats();
    }

  }


  {
    //Reset random seeds for nnEval
    nnEval->killServerThreads();
    nnEval->spawnServerThreads();

    Board board = Board::parseBoard(19,19,R"%%(
...................
............o.oxx..
...x..........ooxo.
...........o..xxo..
..x........oxxxoo.x
..........xxo.oxxo.
............o.o....
..............oxx..
...................
...............x...
..o................
...................
...................
...................
...................
..o.o.........ooo..
.........x...xoxx..
............x.xoo..
.............x.....
)%%");

    Player nextPla = P_BLACK;
    Rules rules = Rules::parseRules("Chinese");
    BoardHistory hist(board,nextPla,rules,0);

    SearchParams params = SearchParams::forTestsV1();
    params.maxVisits = 400;
    params.rootNoiseEnabled = true;
    params.rootPolicyTemperature = 1.2;
    params.rootPolicyTemperatureEarly = 1.2;
    params.rootNumSymmetriesToSample = 2;

    TestSearchOptions opts;
    TestSearchOptions optsContinue;
    optsContinue.ignorePosition = true;

    {
      cout << "===================================================================" << endl;
      cout << "Test real hintloc T16 (with symmetry sampling)" << endl;
      cout << "===================================================================" << endl;
      AsyncBot* bot = new AsyncBot(params, nnEval, &logger, "hintloc");
      bot->setRootHintLoc(Location::ofString("T16",board));
      runBotOnPosition(bot, board, nextPla, hist, opts);
      delete bot;
    }
    {
      cout << "===================================================================" << endl;
      cout << "Test bad hintloc O18 (with symmetry sampling)" << endl;
      cout << "===================================================================" << endl;
      AsyncBot* bot = new AsyncBot(params, nnEval, &logger, "hintloc");
      bot->setRootHintLoc(Location::ofString("O18",board));
      runBotOnPosition(bot, board, nextPla, hist, opts);
      delete bot;
    }

    //Reset random seeds for nnEval
    nnEval->killServerThreads();
    nnEval->spawnServerThreads();
  }

  {
    cout << "AntiMirror white ==========================================================================" << endl;

    string sgfStr = "(;KM[7.5]SZ[19];B[pd];W[dp];B[pp];W[dd];B[cc];W[qq];B[dc];W[pq];B[op];W[ed];B[qp];W[cd];B[ec];W[oq];B[nq];W[fc];B[mp];W[gd];B[rp];W[bd];B[fq];W[nc];B[pi];W[dk];B[fe];W[no];B[cq];W[qc];B[pc];W[dq];B[cp];W[qd];B[do];W[pe];B[oe];W[eo];B[en];W[of];B[nf];W[fn];B[fd];W[np];B[mo];W[ge];B[fo];W[ne];B[od];W[ep];B[gn];W[mf];B[ng];W[fm];B[dn];W[pf];B[ff];W[nn];B[nd];W[fp];B[go];W[me];B[mr];W[gb];B[md];W[gp];B[gm];W[mg];B[mh];W[gl];B[hp];W[ld];B[lc];W[hq];B[fl];W[nh];B[gq];W[mc];B[pb];W[dr];B[hr];W[lb];B[kc];W[iq];B[ir];W[kb];B[hc];W[lq];B[og];W[em];B[hl];W[lh];B[mi];W[gk];B[le];W[ho];B[in];W[kf];B[jq];W[jc];B[pg];W[dm];B[kd];W[ip];B[mq];W[gc];B[bn];W[rf];B[cm];W[qg];B[qh];W[cl];B[rg];W[bm];B[cn];W[qf];B[li];W[hk];B[il];W[kh];B[rh];W[bl];B[bo];W[re];B[ik];W[ki];B[kj];W[ij];B[jj])";
    CompactSgf* sgf = CompactSgf::parse(sgfStr);

    {
      Board board;
      Player nextPla;
      BoardHistory hist;
      Rules initialRules = sgf->getRulesOrFailAllowUnspecified(Rules::getTrompTaylorish());
      sgf->setupBoardAndHistAssumeLegal(initialRules, board, nextPla, hist, 24);
      SearchParams params = SearchParams::forTestsV1();
      params.maxVisits = 200;
      params.antiMirror = true;
      AsyncBot* bot = new AsyncBot(params, nnEval, &logger, "antimirrortest");

      TestSearchOptions opts;
      runBotOnPosition(bot,board,nextPla,hist,opts);
      delete bot;
    }
    {
      Board board;
      Player nextPla;
      BoardHistory hist;
      Rules initialRules = sgf->getRulesOrFailAllowUnspecified(Rules::getTrompTaylorish());
      sgf->setupBoardAndHistAssumeLegal(initialRules, board, nextPla, hist, 32);
      SearchParams params = SearchParams::forTestsV1();
      params.maxVisits = 200;
      params.antiMirror = true;
      AsyncBot* bot = new AsyncBot(params, nnEval, &logger, "antimirrortest");

      TestSearchOptions opts;
      runBotOnPosition(bot,board,nextPla,hist,opts);
      delete bot;
    }
    {
      Board board;
      Player nextPla;
      BoardHistory hist;
      Rules initialRules = sgf->getRulesOrFailAllowUnspecified(Rules::getTrompTaylorish());
      sgf->setupBoardAndHistAssumeLegal(initialRules, board, nextPla, hist, 124);
      SearchParams params = SearchParams::forTestsV1();
      params.maxVisits = 200;
      params.antiMirror = true;
      AsyncBot* bot = new AsyncBot(params, nnEval, &logger, "antimirrortest");

      TestSearchOptions opts;
      runBotOnPosition(bot,board,nextPla,hist,opts);
      delete bot;
    }

    cout << endl << endl;

    delete sgf;
  }

  {
    cout << "AntiMirror black negkomi ==========================================================================" << endl;

    string sgfStr = "(;SZ[19]KM[-3.50];B[jj];W[pd];B[dp];W[dd];B[pp];W[cn];B[qf];W[nq];B[fc];W[qn];B[cf];W[df];B[pn];W[pm];B[dg];W[po];B[de];W[fd];B[np];W[mp];B[gd];W[ed];B[op];W[on];B[ef];W[gc];B[mq];W[lq];B[hc];W[jk];B[ji];W[ik];B[ki];W[ij];B[kj];W[gb];B[mr];W[ic];B[kq];W[pf];B[dn];W[do];B[pe];W[qe];B[co];W[lp];B[hd];W[eo];B[oe];W[qg];B[cm];W[bn];B[rf];W[qd];B[cp];W[hb];B[lr];W[bm];B[rg];W[of];B[en];W[fn];B[nf];W[qh];B[cl];W[ck];B[qi];W[ne];B[fo];W[ep];B[od];W[ng];B[fm];W[gn];B[mf];W[mg];B[gm];W[lf];B[hn];W[rh];B[bl];W[me];B[go];W[ii];B[kk];W[kl])";
    CompactSgf* sgf = CompactSgf::parse(sgfStr);

    {
      Board board;
      Player nextPla;
      BoardHistory hist;
      Rules initialRules = sgf->getRulesOrFailAllowUnspecified(Rules::getTrompTaylorish());
      sgf->setupBoardAndHistAssumeLegal(initialRules, board, nextPla, hist, 29);
      SearchParams params = SearchParams::forTestsV1();
      params.maxVisits = 200;
      params.antiMirror = true;
      AsyncBot* bot = new AsyncBot(params, nnEval, &logger, "antimirrortest");

      TestSearchOptions opts;
      runBotOnPosition(bot,board,nextPla,hist,opts);
      delete bot;
    }
    {
      Board board;
      Player nextPla;
      BoardHistory hist;
      Rules initialRules = sgf->getRulesOrFailAllowUnspecified(Rules::getTrompTaylorish());
      sgf->setupBoardAndHistAssumeLegal(initialRules, board, nextPla, hist, 83);
      SearchParams params = SearchParams::forTestsV1();
      params.maxVisits = 200;
      params.antiMirror = true;
      AsyncBot* bot = new AsyncBot(params, nnEval, &logger, "antimirrortest");

      TestSearchOptions opts;
      runBotOnPosition(bot,board,nextPla,hist,opts);
      delete bot;
    }

    cout << endl << endl;

    delete sgf;
  }

}

static void runMoreV8Tests(NNEvaluator* nnEval, Logger& logger)
{
  {
    cout << "TEST VALUE BIAS ==========================================================================" << endl;

    string sgfStr = "(;GM[1]FF[4]CA[UTF-8]RU[Japanese]SZ[9]KM[0];B[dc];W[ef];B[df];W[de];B[dg];W[eg];B[eh];W[fh];B[ee])";
    CompactSgf* sgf = CompactSgf::parse(sgfStr);

    Board board;
    Player nextPla;
    BoardHistory hist;
    Rules initialRules = sgf->getRulesOrFail();
    sgf->setupBoardAndHistAssumeLegal(initialRules, board, nextPla, hist, 8);

    SearchParams params = SearchParams::forTestsV1();
    params.maxVisits = 20;
    params.chosenMoveTemperature = 0;
    AsyncBot* botA = new AsyncBot(params, nnEval, &logger, "valuebias test");
    params.subtreeValueBiasFactor = 0.5;
    AsyncBot* botB = new AsyncBot(params, nnEval, &logger, "valuebias test");
    params.maxVisits = 300;
    AsyncBot* botC = new AsyncBot(params, nnEval, &logger, "valuebias test");

    TestSearchOptions opts;
    opts.printMoreMoreMore = true;
    opts.numMovesInARow = 3;
    opts.printAfterBegun = true;
    cout << "BASE" << endl;
    runBotOnPosition(botA,board,nextPla,hist,opts);
    cout << "VALUE BIAS 0.5" << endl;
    runBotOnPosition(botB,board,nextPla,hist,opts);

    opts.printMoreMoreMore = false;
    runBotOnPosition(botC,board,nextPla,hist,opts);

    cout << endl << endl;

    delete botA;
    delete botB;
    delete botC;
    delete sgf;
  }

  {
    Board board = Board::parseBoard(11,11,R"%%(
.o.xox.x.o.
xxxxoxxooox
oo.ooxxxxxx
.oo.ox.xooo
oooooxxo.o.
xxxoooxxooo
.x.xoxxxxxx
xxxooox.xx.
oooo.oxx.xx
oxxxooxoooo
.x.o.oxo.x.
)%%");

    SearchParams params = SearchParams::forTestsV1();
    params.maxVisits = 300;
    params.fpuReductionMax = 0.0;
    params.rootFpuReductionMax = 0.0;
    params.rootEndingBonusPoints = 0.0;
    params.rootPolicyTemperature = 1.5;
    params.rootPolicyTemperatureEarly = 1.5;
    SearchParams params2 = params;
    params2.rootEndingBonusPoints = 0.5;

    TestSearchOptions opts;
    opts.printEndingScoreValueBonus = true;

    {
      cout << "===================================================================" << endl;
      cout << "Ending bonus points in area scoring with selfatari moves, white first" << endl;
      cout << "===================================================================" << endl;

      Player nextPla = P_WHITE;
      Rules rules = Rules::parseRules("Chinese");
      BoardHistory hist(board,nextPla,rules,0);
      {
        cout << "Without root ending bonus pts===================" << endl;
        cout << endl;
        AsyncBot* bot = new AsyncBot(params, nnEval, &logger, "async bot ending bonus points seed");
        runBotOnPosition(bot, board, nextPla, hist, opts);
        delete bot;
      }
      {
        cout << "With root ending bonus pts===================" << endl;
        cout << endl;
        AsyncBot* bot = new AsyncBot(params2, nnEval, &logger, "async bot ending bonus points seed");
        runBotOnPosition(bot, board, nextPla, hist, opts);
        delete bot;
      }
    }

    {
      cout << "===================================================================" << endl;
      cout << "Ending bonus points in area scoring with selfatari moves, white first, button" << endl;
      cout << "===================================================================" << endl;

      Player nextPla = P_WHITE;
      Rules rules = Rules::parseRules("Chinese");
      rules.hasButton = true;
      BoardHistory hist(board,nextPla,rules,0);
      {
        cout << "Without root ending bonus pts===================" << endl;
        cout << endl;
        AsyncBot* bot = new AsyncBot(params, nnEval, &logger, "async bot ending bonus points seed");
        runBotOnPosition(bot, board, nextPla, hist, opts);
        delete bot;
      }
      {
        cout << "With root ending bonus pts===================" << endl;
        cout << endl;
        AsyncBot* bot = new AsyncBot(params2, nnEval, &logger, "async bot ending bonus points seed");
        runBotOnPosition(bot, board, nextPla, hist, opts);
        delete bot;
      }
    }

    {
      cout << "===================================================================" << endl;
      cout << "Ending bonus points in area scoring with selfatari moves, black first" << endl;
      cout << "===================================================================" << endl;

      Player nextPla = P_BLACK;
      Rules rules = Rules::parseRules("Chinese");
      BoardHistory hist(board,nextPla,rules,0);
      {
        cout << "Without root ending bonus pts===================" << endl;
        cout << endl;
        AsyncBot* bot = new AsyncBot(params, nnEval, &logger, "async bot ending bonus points seed");
        runBotOnPosition(bot, board, nextPla, hist, opts);
        delete bot;
      }
      {
        cout << "With root ending bonus pts===================" << endl;
        cout << endl;
        AsyncBot* bot = new AsyncBot(params2, nnEval, &logger, "async bot ending bonus points seed");
        runBotOnPosition(bot, board, nextPla, hist, opts);
        delete bot;
      }
    }

    {
      cout << "===================================================================" << endl;
      cout << "Ending bonus points in area scoring with selfatari moves, black first, button" << endl;
      cout << "===================================================================" << endl;

      Player nextPla = P_BLACK;
      Rules rules = Rules::parseRules("Chinese");
      rules.hasButton = true;
      BoardHistory hist(board,nextPla,rules,0);
      {
        cout << "Without root ending bonus pts===================" << endl;
        cout << endl;
        AsyncBot* bot = new AsyncBot(params, nnEval, &logger, "async bot ending bonus points seed");
        runBotOnPosition(bot, board, nextPla, hist, opts);
        delete bot;
      }
      {
        cout << "With root ending bonus pts===================" << endl;
        cout << endl;
        AsyncBot* bot = new AsyncBot(params2, nnEval, &logger, "async bot ending bonus points seed");
        runBotOnPosition(bot, board, nextPla, hist, opts);
        delete bot;
      }
    }

    {
      cout << "===================================================================" << endl;
      cout << "Ending bonus points in territory scoring with selfatari moves, black first" << endl;
      cout << "===================================================================" << endl;

      Player nextPla = P_BLACK;
      Rules rules = Rules::parseRules("Japanese");
      BoardHistory hist(board,nextPla,rules,0);
      {
        cout << "Without root ending bonus pts===================" << endl;
        cout << endl;
        AsyncBot* bot = new AsyncBot(params, nnEval, &logger, "async bot ending bonus points seed");
        runBotOnPosition(bot, board, nextPla, hist, opts);
        delete bot;
      }
      {
        cout << "With root ending bonus pts===================" << endl;
        cout << endl;
        AsyncBot* bot = new AsyncBot(params2, nnEval, &logger, "async bot ending bonus points seed");
        runBotOnPosition(bot, board, nextPla, hist, opts);
        delete bot;
      }
    }


    {
      cout << "===================================================================" << endl;
      cout << "Ending bonus points in territory scoring with selfatari moves, black first, encore 2" << endl;
      cout << "===================================================================" << endl;

      Player nextPla = P_BLACK;
      Rules rules = Rules::parseRules("Japanese");
      BoardHistory hist(board,nextPla,rules,2);
      {
        cout << "Without root ending bonus pts===================" << endl;
        cout << endl;
        AsyncBot* bot = new AsyncBot(params, nnEval, &logger, "async bot ending bonus points seed");
        runBotOnPosition(bot, board, nextPla, hist, opts);
        delete bot;
      }
      {
        cout << "With root ending bonus pts===================" << endl;
        cout << endl;
        AsyncBot* bot = new AsyncBot(params2, nnEval, &logger, "async bot ending bonus points seed");
        runBotOnPosition(bot, board, nextPla, hist, opts);
        delete bot;
      }
    }
  }


  {
    cout << "TEST Ending bonus points ==========================================================================" << endl;

    Board board = Board::parseBoard(11,11,R"%%(
.x..ox.oxo.
xxxooxxox.o
oooox.xoxxx
xxooo.ooooo
x.xoooo..x.
...oxxxox.x
o.oox.xooxo
ooox..xoooo
xxxx.xxoxxx
.....xoox.o
.....xo.xo.
)%%");

    SearchParams params = SearchParams::forTestsV1();
    params.maxVisits = 300;
    params.fpuReductionMax = 0.0;
    params.rootFpuReductionMax = 0.0;
    params.rootEndingBonusPoints = 0.0;
    params.rootPolicyTemperature = 1.5;
    params.rootPolicyTemperatureEarly = 1.5;
    SearchParams params2 = params;
    params2.rootEndingBonusPoints = 0.5;

    TestSearchOptions opts;
    opts.printEndingScoreValueBonus = true;

    {
      cout << "===================================================================" << endl;
      cout << "Ending bonus points in area scoring with selfatari moves, one more fancy position, white first" << endl;
      cout << "===================================================================" << endl;

      Player nextPla = P_WHITE;
      Rules rules = Rules::parseRules("Chinese");
      BoardHistory hist(board,nextPla,rules,0);
      {
        AsyncBot* bot = new AsyncBot(params2, nnEval, &logger, "async bot ending bonus points seed");
        runBotOnPosition(bot, board, nextPla, hist, opts);
        delete bot;
      }
    }
  }

  {
    cout << "TEST futileVisitsThreshold ==========================================================================" << endl;

    Player nextPla = P_BLACK;
    Rules rules = Rules::getTrompTaylorish();
    Board board = Board::parseBoard(9,9,R"%%(
.........
.........
.ox..xo..
.........
..o...x..
.........
..ox..ox.
.........
.........
)%%");
    BoardHistory hist(board,nextPla,rules,0);

    SearchParams params = SearchParams::forTestsV1();
    params.maxVisits = 400;
    AsyncBot* botA = new AsyncBot(params, nnEval, &logger, "futileVisitsThreshold test");
    params.futileVisitsThreshold = 0.15;
    AsyncBot* botB = new AsyncBot(params, nnEval, &logger, "futileVisitsThreshold test");
    params.futileVisitsThreshold = 0.4;
    AsyncBot* botC = new AsyncBot(params, nnEval, &logger, "futileVisitsThreshold test");

    TestSearchOptions opts;
    cout << "BASE" << endl;
    runBotOnPosition(botA,board,nextPla,hist,opts);
    cout << "futileVisitsThreshold 0.15" << endl;
    runBotOnPosition(botB,board,nextPla,hist,opts);
    cout << "futileVisitsThreshold 0.4" << endl;
    runBotOnPosition(botC,board,nextPla,hist,opts);
    cout << endl << endl;

    delete botA;
    delete botB;
    delete botC;
  }

  {
    cout << "TEST futileVisitsThreshold with playouts ==========================================================================" << endl;

    Player nextPla = P_BLACK;
    Rules rules = Rules::getTrompTaylorish();
    Board board = Board::parseBoard(9,9,R"%%(
.........
.........
.ox..xo..
.........
..o...x..
.........
..ox..ox.
.........
.........
)%%");
    BoardHistory hist(board,nextPla,rules,0);

    SearchParams params = SearchParams::forTestsV1();
    params.maxVisits = 10000;
    params.maxPlayouts = 200;
    AsyncBot* botA = new AsyncBot(params, nnEval, &logger, "futileVisitsThreshold test");
    params.futileVisitsThreshold = 0.15;
    AsyncBot* botB = new AsyncBot(params, nnEval, &logger, "futileVisitsThreshold test");
    params.futileVisitsThreshold = 0.4;
    AsyncBot* botC = new AsyncBot(params, nnEval, &logger, "futileVisitsThreshold test");

    TestSearchOptions opts;
    cout << "BASE" << endl;
    runBotOnPosition(botA,board,nextPla,hist,opts);
    cout << "futileVisitsThreshold 0.15" << endl;
    runBotOnPosition(botB,board,nextPla,hist,opts);
    cout << "futileVisitsThreshold 0.4" << endl;
    runBotOnPosition(botC,board,nextPla,hist,opts);
    cout << endl << endl;

    delete botA;
    delete botB;
    delete botC;
  }

  {
    cout << "TEST Hintlocs ==========================================================================" << endl;

    //Reset random seeds for nnEval
    nnEval->killServerThreads();
    nnEval->spawnServerThreads();

    Board board = Board::parseBoard(19,19,R"%%(
...................
...................
................o..
...x...........x...
...................
...................
...................
...................
...................
...................
...................
...................
...................
...o............x..
...................
.............oo.x..
...o.........oxx...
...................
...................
)%%");

    Player nextPla = P_BLACK;
    Rules rules = Rules::parseRules("Chinese");
    BoardHistory hist(board,nextPla,rules,0);

    SearchParams paramsFast = SearchParams::forTestsV1();
    paramsFast.maxVisits = 5;
    SearchParams paramsSlow = SearchParams::forTestsV1();
    paramsSlow.maxVisits = 200;
    SearchParams paramsFastNoised = paramsFast;
    paramsFastNoised.rootNoiseEnabled = true;
    SearchParams paramsSlowNoised = paramsSlow;
    paramsSlowNoised.rootNoiseEnabled = true;
    //Note - symmetry sampling here will randomize the root even though the nnEval is set to not randomize
    //because the root symmetry sampling is done in the search
    testAssert(nnEval->getDoRandomize() == false);
    SearchParams paramsFastSym = paramsFastNoised;
    paramsFastSym.rootNumSymmetriesToSample = 4;
    SearchParams paramsSlowSym = paramsSlowNoised;
    paramsSlowSym.rootNumSymmetriesToSample = 4;

    SearchParams paramsSlowSymNoNoise = paramsSlowSym;
    paramsSlowSymNoNoise.rootNoiseEnabled = false;

    TestSearchOptions opts;
    opts.noClearBot = true;
    TestSearchOptions optsContinue;
    optsContinue.noClearBot = true;
    optsContinue.ignorePosition = true;

    {
      cout << "===================================================================" << endl;
      cout << "Test hintloc C1" << endl;
      cout << "===================================================================" << endl;
      AsyncBot* bot = new AsyncBot(paramsSlow, nnEval, &logger, "hintloc");
      bot->setRootHintLoc(Location::ofString("C1",board));
      runBotOnPosition(bot, board, nextPla, hist, opts);
      delete bot;
    }
    {
      cout << "===================================================================" << endl;
      cout << "Test hintloc C1, again" << endl;
      cout << "===================================================================" << endl;
      AsyncBot* bot = new AsyncBot(paramsSlow, nnEval, &logger, "hintloc");
      bot->setRootHintLoc(Location::ofString("C1",board));
      runBotOnPosition(bot, board, nextPla, hist, opts);
      delete bot;
    }
    {
      cout << "===================================================================" << endl;
      cout << "Test hintloc C1 after attempting same-turn tree reuse" << endl;
      cout << "===================================================================" << endl;
      AsyncBot* bot = new AsyncBot(paramsFast, nnEval, &logger, "hintloc");
      runBotOnPosition(bot, board, nextPla, hist, opts);
      bot->setParamsNoClearing(paramsSlow);
      bot->setRootHintLoc(Location::ofString("C1",board)); //This should actually clear the tree even though we didn't say to do so!
      runBotOnPosition(bot, board, nextPla, hist, optsContinue);
      delete bot;
    }
    {
      cout << "===================================================================" << endl;
      cout << "Test hintloc C1 dirichlet noise" << endl;
      cout << "===================================================================" << endl;
      AsyncBot* bot = new AsyncBot(paramsSlowNoised, nnEval, &logger, "hintloc");
      bot->setRootHintLoc(Location::ofString("C1",board));
      runBotOnPosition(bot, board, nextPla, hist, opts);
      delete bot;
    }
    {
      cout << "===================================================================" << endl;
      cout << "Test hintloc C1 dirichlet noise after attempting same-turn tree reuse" << endl;
      cout << "===================================================================" << endl;
      AsyncBot* bot = new AsyncBot(paramsFastNoised, nnEval, &logger, "hintloc");
      runBotOnPosition(bot, board, nextPla, hist, opts);
      bot->setParamsNoClearing(paramsSlowNoised);
      bot->setRootHintLoc(Location::ofString("C1",board)); //This should actually clear the tree even though we didn't say to do so!
      runBotOnPosition(bot, board, nextPla, hist, optsContinue);
      delete bot;
    }
    {
      cout << "===================================================================" << endl;
      cout << "Test hintloc C1 dirichlet noise and symmetry sampling" << endl;
      cout << "===================================================================" << endl;
      AsyncBot* bot = new AsyncBot(paramsSlowSym, nnEval, &logger, "hintloc");
      bot->setRootHintLoc(Location::ofString("C1",board));
      runBotOnPosition(bot, board, nextPla, hist, opts);
      delete bot;
    }
    {
      cout << "===================================================================" << endl;
      cout << "Test hintloc C1 dirichlet noise and symmetry sampling, again (same due to matching search seed)" << endl;
      cout << "===================================================================" << endl;
      AsyncBot* bot = new AsyncBot(paramsSlowSym, nnEval, &logger, "hintloc");
      bot->setRootHintLoc(Location::ofString("C1",board));
      runBotOnPosition(bot, board, nextPla, hist, opts);
      delete bot;
    }
    {
      cout << "===================================================================" << endl;
      cout << "Test hintloc C1 dirichlet noise and symmetry sampling after attempting same-turn tree reuse" << endl;
      cout << "===================================================================" << endl;
      AsyncBot* bot = new AsyncBot(paramsFastSym, nnEval, &logger, "hintloc");
      runBotOnPosition(bot, board, nextPla, hist, opts);
      bot->setParamsNoClearing(paramsSlowSym);
      bot->setRootHintLoc(Location::ofString("C1",board)); //This should actually clear the tree even though we didn't say to do so!
      runBotOnPosition(bot, board, nextPla, hist, optsContinue);
      delete bot;
    }
    {
      cout << "===================================================================" << endl;
      cout << "Test hintloc C1 symmetry sampling alone, new seed" << endl;
      cout << "===================================================================" << endl;
      AsyncBot* bot = new AsyncBot(paramsSlowSymNoNoise, nnEval, &logger, "abc");
      bot->setRootHintLoc(Location::ofString("C1",board));
      runBotOnPosition(bot, board, nextPla, hist, opts);
      delete bot;
    }
    {
      cout << "===================================================================" << endl;
      cout << "Test hintloc C1 symmetry sampling alone, new seed 2" << endl;
      cout << "===================================================================" << endl;
      AsyncBot* bot = new AsyncBot(paramsSlowSymNoNoise, nnEval, &logger, "abc2");
      bot->setRootHintLoc(Location::ofString("C1",board));
      runBotOnPosition(bot, board, nextPla, hist, opts);
      delete bot;
    }

    //Reset random seeds for nnEval
    nnEval->killServerThreads();
    nnEval->spawnServerThreads();
  }

  {
    cout << "Mix pruning, subtracting, dirichlet noise, value weight exponent =========================================" << endl;

    Board board = Board::parseBoard(19,19,R"%%(
...................
...................
.....x.............
...x............x..
...................
..o................
...................
...................
...................
...................
...................
...................
...................
................x..
...................
..o.........o.o.x..
.....o.......oxx...
...................
...................
)%%");

    Player nextPla = P_WHITE;
    Rules rules = Rules::parseRules("Chinese");
    BoardHistory hist(board,nextPla,rules,0);

    SearchParams paramsBase = SearchParams::forTestsV1();
    paramsBase.maxVisits = 500;
    TestSearchOptions opts;

    {
      cout << "===================================================================" << endl;
      cout << "Base" << endl;
      cout << "===================================================================" << endl;
      SearchParams params = paramsBase;
      AsyncBot* bot = new AsyncBot(params, nnEval, &logger, "mix");
      runBotOnPosition(bot, board, nextPla, hist, opts);
      delete bot;
    }
    {
      cout << "===================================================================" << endl;
      cout << "Dirichlet noise" << endl;
      cout << "===================================================================" << endl;
      SearchParams params = paramsBase;
      params.rootNoiseEnabled = true;
      AsyncBot* bot = new AsyncBot(params, nnEval, &logger, "mix");
      runBotOnPosition(bot, board, nextPla, hist, opts);
      delete bot;
    }
    {
      cout << "===================================================================" << endl;
      cout << "Dirichlet noise Prune 10 sub 7" << endl;
      cout << "===================================================================" << endl;
      SearchParams params = paramsBase;
      params.rootNoiseEnabled = true;
      params.chosenMovePrune = 10;
      params.chosenMoveSubtract = 7;
      AsyncBot* bot = new AsyncBot(params, nnEval, &logger, "mix");
      runBotOnPosition(bot, board, nextPla, hist, opts);
      delete bot;
    }
    {
      cout << "===================================================================" << endl;
      cout << "Dirichlet noise Value weight exponent 0.8" << endl;
      cout << "===================================================================" << endl;
      SearchParams params = paramsBase;
      params.rootNoiseEnabled = true;
      params.valueWeightExponent = 0.8;
      AsyncBot* bot = new AsyncBot(params, nnEval, &logger, "mix");
      runBotOnPosition(bot, board, nextPla, hist, opts);
      delete bot;
    }
    {
      cout << "===================================================================" << endl;
      cout << "Dirichlet noise Value weight exponent 0.8 prune 12 sub 5" << endl;
      cout << "===================================================================" << endl;
      SearchParams params = paramsBase;
      params.rootNoiseEnabled = true;
      params.chosenMovePrune = 12;
      params.chosenMoveSubtract = 5;
      params.valueWeightExponent = 0.8;
      AsyncBot* bot = new AsyncBot(params, nnEval, &logger, "mix");
      runBotOnPosition(bot, board, nextPla, hist, opts);
      delete bot;
    }
    {
      cout << "===================================================================" << endl;
      cout << "Dirichlet noise Value weight exponent 0.8 more visits" << endl;
      cout << "===================================================================" << endl;
      SearchParams params = paramsBase;
      params.maxVisits = 2500;
      params.rootNoiseEnabled = true;
      params.valueWeightExponent = 0.8;
      AsyncBot* bot = new AsyncBot(params, nnEval, &logger, "mix");
      runBotOnPosition(bot, board, nextPla, hist, opts);
      delete bot;
    }
    {
      cout << "===================================================================" << endl;
      cout << "Dirichlet noise Value weight exponent 0.8 prune 12 sub 5 more visits" << endl;
      cout << "===================================================================" << endl;
      SearchParams params = paramsBase;
      params.maxVisits = 2500;
      params.rootNoiseEnabled = true;
      params.chosenMovePrune = 12;
      params.chosenMoveSubtract = 5;
      params.valueWeightExponent = 0.8;
      AsyncBot* bot = new AsyncBot(params, nnEval, &logger, "mix");
      runBotOnPosition(bot, board, nextPla, hist, opts);
      delete bot;
    }
    {
      cout << "===================================================================" << endl;
      cout << "Dirichlet noise Value weight exponent 0.0 more visits" << endl;
      cout << "===================================================================" << endl;
      SearchParams params = paramsBase;
      params.maxVisits = 2500;
      params.rootNoiseEnabled = true;
      params.valueWeightExponent = 0.0;
      AsyncBot* bot = new AsyncBot(params, nnEval, &logger, "mix");
      runBotOnPosition(bot, board, nextPla, hist, opts);
      delete bot;
    }
    {
      cout << "===================================================================" << endl;
      cout << "Dirichlet noise Value weight exponent 0.0 prune 7 sub 3 more visits" << endl;
      cout << "===================================================================" << endl;
      SearchParams params = paramsBase;
      params.maxVisits = 2500;
      params.rootNoiseEnabled = true;
      params.chosenMovePrune = 12;
      params.chosenMoveSubtract = 5;
      params.valueWeightExponent = 0.0;
      AsyncBot* bot = new AsyncBot(params, nnEval, &logger, "mix");
      runBotOnPosition(bot, board, nextPla, hist, opts);
      delete bot;
    }
  }

  {
    cout << "Fill dame before pass =========================================" << endl;

    Board board = Board::parseBoard(9,10,R"%%(
.....xoo.
.o...xo.o
.xx..xoox
x....xxxx
oxx...x..
oooxxox..
...ooxxxx
..o.xoox.
...ooo.ox
.......o.
)%%");

    SearchParams paramsBase = SearchParams::forTestsV1();
    paramsBase.maxVisits = 600;
    TestSearchOptions opts;

    {
      Player nextPla = P_WHITE;
      Rules rules = Rules::parseRules("Japanese");
      BoardHistory hist(board,nextPla,rules,0);

      cout << "===================================================================" << endl;
      cout << "Base, white to play" << endl;
      cout << "===================================================================" << endl;
      SearchParams params = paramsBase;
      AsyncBot* bot = new AsyncBot(params, nnEval, &logger, "fill dame before pass");
      runBotOnPosition(bot, board, nextPla, hist, opts);
      delete bot;
    }

    {
      Player nextPla = P_WHITE;
      Rules rules = Rules::parseRules("Japanese");
      BoardHistory hist(board,nextPla,rules,0);

      cout << "===================================================================" << endl;
      cout << "Fill dame before pass, white to play" << endl;
      cout << "===================================================================" << endl;
      SearchParams params = paramsBase;
      params.fillDameBeforePass = true;
      AsyncBot* bot = new AsyncBot(params, nnEval, &logger, "fill dame before pass");
      runBotOnPosition(bot, board, nextPla, hist, opts);
      delete bot;
    }

    {
      Player nextPla = P_BLACK;
      Rules rules = Rules::parseRules("Japanese");
      BoardHistory hist(board,nextPla,rules,0);

      cout << "===================================================================" << endl;
      cout << "Base, black to play" << endl;
      cout << "===================================================================" << endl;
      SearchParams params = paramsBase;
      AsyncBot* bot = new AsyncBot(params, nnEval, &logger, "fill dame before pass");
      runBotOnPosition(bot, board, nextPla, hist, opts);
      delete bot;
    }

    {
      Player nextPla = P_BLACK;
      Rules rules = Rules::parseRules("Japanese");
      BoardHistory hist(board,nextPla,rules,0);

      cout << "===================================================================" << endl;
      cout << "Fill dame before pass, black to play" << endl;
      cout << "===================================================================" << endl;
      SearchParams params = paramsBase;
      params.fillDameBeforePass = true;
      AsyncBot* bot = new AsyncBot(params, nnEval, &logger, "fill dame before pass");
      runBotOnPosition(bot, board, nextPla, hist, opts);
      delete bot;
    }
  }


  {
    cout << "Conservative pass =========================================" << endl;

    Board board = Board::parseBoard(9,9,R"%%(
ox.x.xx..
.x.x.x.xx
xxxx.xxx.
.x..x..xx
xxxxxxxxo
xoooxoooo
xo..o.ox.
oooo.o.oo
...ooooo.
)%%");

    SearchParams paramsBase = SearchParams::forTestsV1();
    paramsBase.maxVisits = 600;
    TestSearchOptions opts;
    opts.noClearBot = true;

    {
      Player nextPla = P_WHITE;
      Rules rules = Rules::parseRules("Chinese");
      rules.komi = 14;
      BoardHistory hist(board,nextPla,rules,0);

      cout << "===================================================================" << endl;
      cout << "White to play" << endl;
      cout << "===================================================================" << endl;
      SearchParams params = paramsBase;
      AsyncBot* bot = new AsyncBot(params, nnEval, &logger, "conservative pass");
      runBotOnPosition(bot, board, nextPla, hist, opts);

      Loc moveLoc = bot->getSearchStopAndWait()->getChosenMoveLoc();
      Loc moveLoc2 = PlayUtils::maybeCleanupBeforePass(enabled_t::True, enabled_t::False, nextPla, moveLoc, bot);
      cout << "Move loc: " << Location::toString(moveLoc,board) << endl;
      cout << "Conservative pass: " << Location::toString(moveLoc2,board) << endl;
      delete bot;
    }

    {
      Player nextPla = P_BLACK;
      Rules rules = Rules::parseRules("Chinese");
      BoardHistory hist(board,nextPla,rules,0);

      cout << "===================================================================" << endl;
      cout << "White to play" << endl;
      cout << "===================================================================" << endl;
      SearchParams params = paramsBase;
      AsyncBot* bot = new AsyncBot(params, nnEval, &logger, "conservative pass");
      runBotOnPosition(bot, board, nextPla, hist, opts);

      Loc moveLoc = bot->getSearchStopAndWait()->getChosenMoveLoc();
      Loc moveLoc2 = PlayUtils::maybeCleanupBeforePass(enabled_t::True, enabled_t::False, nextPla, moveLoc, bot);
      cout << "Move loc: " << Location::toString(moveLoc,board) << endl;
      cout << "Conservative pass: " << Location::toString(moveLoc2,board) << endl;
      delete bot;
    }

  }

  {
    cout << "Basic graph search 7x7 fight =========================================" << endl;

    Board board = Board::parseBoard(7,7,R"%%(
.....o.
.....ox
..ooox.
.xoxxx.
.xxo.x.
..xooox
.......
)%%");
    Player nextPla = P_WHITE;

    SearchParams paramsBase = SearchParams::forTestsV1();
    paramsBase.maxVisits = 1000;
    paramsBase.useGraphSearch = true;
    TestSearchOptions opts;
    opts.numMovesInARow = 3;
    opts.printPostOrderNodeCount = true;

    {
      Rules rules = Rules::parseRules("Japanese");
      rules.komi = 8;
      BoardHistory hist(board,nextPla,rules,0);

      cout << "===================================================================" << endl;
      cout << "White to play" << endl;
      cout << "===================================================================" << endl;
      SearchParams params = paramsBase;
      AsyncBot* bot = new AsyncBot(params, nnEval, &logger, "conservative pass");
      runBotOnPosition(bot, board, nextPla, hist, opts);
      delete bot;
    }
  }

  {
    cout << "Friendly pass =========================================" << endl;

    SearchParams paramsBase = SearchParams::forTestsV1();
    paramsBase.maxVisits = 600;
    TestSearchOptions opts;
    opts.noClearBot = true;

    {
      Board board = Board::parseBoard(9,9,R"%%(
.........
....x.x..
.x.xox...
...x..xxx
xxxxxxooo
xooooo...
o....xo..
.o...o.o.
.........
)%%");


      Player nextPla = P_BLACK;
      Rules rules = Rules::parseRules("Chinese");
      rules.komi = 4;
      BoardHistory hist(board,nextPla,rules,0);
      hist.makeBoardMoveAssumeLegal(board,Board::PASS_LOC,nextPla,NULL);
      nextPla = P_WHITE;

      cout << "===================================================================" << endl;
      cout << "White to play" << endl;
      cout << "===================================================================" << endl;
      SearchParams params = paramsBase;
      AsyncBot* bot = new AsyncBot(params, nnEval, &logger, "friendly pass");
      runBotOnPosition(bot, board, nextPla, hist, opts);

      Loc moveLoc = bot->getSearchStopAndWait()->getChosenMoveLoc();
      Loc moveLoc2 = PlayUtils::maybeFriendlyPass(enabled_t::False, enabled_t::True, nextPla, moveLoc, bot->getSearchStopAndWait(),50);
      cout << "Move loc: " << Location::toString(moveLoc,board) << endl;
      cout << "Friendly pass: " << Location::toString(moveLoc2,board) << endl;
      delete bot;
    }

    {
      Board board = Board::parseBoard(9,9,R"%%(
.........
....x.x..
.x.xox...
......xxx
xxxxxxooo
xooooo...
o....xo..
.o..oo.o.
.........
)%%");

      Player nextPla = P_WHITE;
      Rules rules = Rules::parseRules("Chinese");
      rules.komi = 7;
      BoardHistory hist(board,nextPla,rules,0);
      hist.makeBoardMoveAssumeLegal(board,Board::PASS_LOC,nextPla,NULL);
      nextPla = P_BLACK;

      cout << "===================================================================" << endl;
      cout << "Black to play" << endl;
      cout << "===================================================================" << endl;
      SearchParams params = paramsBase;
      AsyncBot* bot = new AsyncBot(params, nnEval, &logger, "friendly pass");
      runBotOnPosition(bot, board, nextPla, hist, opts);

      Loc moveLoc = bot->getSearchStopAndWait()->getChosenMoveLoc();
      Loc moveLoc2 = PlayUtils::maybeFriendlyPass(enabled_t::False, enabled_t::True, nextPla, moveLoc, bot->getSearchStopAndWait(),50);
      cout << "Move loc: " << Location::toString(moveLoc,board) << endl;
      cout << "Friendly pass: " << Location::toString(moveLoc2,board) << endl;
      delete bot;
    }

  }

  {
    cout << "Multithreaded tree updating =========================================" << endl;

    Board boardBase = Board::parseBoard(19,19,R"%%(
...................
...................
...................
...x...........x...
...................
...................
...................
...................
...................
...................
...................
...................
..x.xo.............
...xo..............
.o.xo..............
..ooxo.........o...
.oxxxo.............
.....xo............
...................
)%%");

    Player nextPlaBase = P_BLACK;
    Rules rules = Rules::parseRules("Japanese");
    rules.komi = 6.5;
    BoardHistory histBase(boardBase,nextPlaBase,rules,0);

    SearchParams paramsBase = SearchParams::forTestsV1();
    paramsBase.maxVisits = 1000;


    auto runTest = [&](int64_t numVisits, int numThreads, bool subtreeValueBias, bool graphSearch) {
      Board board = boardBase;
      Player nextPla = nextPlaBase;
      BoardHistory hist = histBase;

      SearchParams params = paramsBase;
      params.maxVisits = numVisits;
      if(subtreeValueBias) {
        params.subtreeValueBiasFactor = 0.35;
        params.subtreeValueBiasWeightExponent = 0.8;
        params.subtreeValueBiasFreeProp = 0.0;
      }
      if(graphSearch)
        params.useGraphSearch = true;

      nnEval->clearCache(); nnEval->clearStats();
      Search* search = new Search(params, nnEval, &logger, "multithreaded tree updating");
      search->setPosition(nextPla,board,hist);
      search->runWholeSearch(nextPla);

      params.numThreads = numThreads;
      search->setParamsNoClearing(params);

      PrintTreeOptions options;
      options = options.maxDepth(1);
      printBasicStuffAfterSearch(board,hist,search,options);

      Loc moveLoc = Location::ofString("E2",board);
      search->makeMove(moveLoc,nextPla);
      hist.makeBoardMoveAssumeLegal(board,moveLoc,nextPla,NULL);
      nextPla = getOpp(nextPla);

      cout << "Just after move" << endl;
      search->printTree(cout, search->rootNode, options, P_WHITE);
      bool pondering = false;
      search->beginSearch(pondering);
      cout << "Just after begin search" << endl;
      if(subtreeValueBias)
        cout << "Skipping since exact values are nondeterministic due to subtree value bias float update order" << endl;
      else
        search->printTree(cout, search->rootNode, options, P_WHITE);

      delete search;
      nnEval->clearCache(); nnEval->clearStats();
    };

    cout << "===================================================================" << endl;
    cout << "Baseline 1k visits" << endl;
    cout << "===================================================================" << endl;
    runTest(1000, 1, false, false);

    cout << "===================================================================" << endl;
    cout << "Baseline 1k visits, graph search" << endl;
    cout << "===================================================================" << endl;
    runTest(1000, 1, false, true);

    cout << "===================================================================" << endl;
    cout << "4 threads for search updates, 1k visits" << endl;
    cout << "===================================================================" << endl;
    runTest(1000, 4, false, false);

    cout << "===================================================================" << endl;
    cout << "4 threads for search updates, subtree value bias stability (but 0 free prop), 1k visits" << endl;
    cout << "===================================================================" << endl;
    runTest(1000, 4, true, false);

    cout << "===================================================================" << endl;
    cout << "Baseline 8k visits" << endl;
    cout << "===================================================================" << endl;
    runTest(8000, 1, false, false);

    cout << "===================================================================" << endl;
    cout << "4 threads for search updates, 8k visits" << endl;
    cout << "===================================================================" << endl;
    runTest(8000, 4, false, false);

    cout << "===================================================================" << endl;
    cout << "30 threads for search updates, 8k visits" << endl;
    cout << "===================================================================" << endl;
    runTest(8000, 30, false, false);

    cout << "===================================================================" << endl;
    cout << "30 threads for search updates, 8k visits, graph search" << endl;
    cout << "===================================================================" << endl;
    runTest(8000, 30, false, true);

    cout << "===================================================================" << endl;
    cout << "30 threads for search updates, subtree value bias stability (but 0 free prop), 8k visits" << endl;
    cout << "===================================================================" << endl;
    runTest(8000, 4, true, false);
  }

  {
    cout << "Pattern bonus =========================================" << endl;

    Board board = Board::parseBoard(19,19,R"%%(
   A B C D E F G H J K L M N O P Q R S T
19 . . . . . . . . . . . . . . . . . . .
18 . . . . . . . . . . . . . . . . . . .
17 . . . . . . . . . . . . . . . . . . .
16 . . . X . . . . . . . . . . . X . . .
15 . . . . . . . . . . . . . . . . . . .
14 . . . . . . . . . . . . . . . . . . .
13 . . . . . . . . . . . . . . . . . . .
12 . . . . . . . . . . . . . . . . . . .
11 . . . . . . . . . . . . . . . . . . .
10 . . . . . . . . . . . . . . . . . . .
 9 . . . . . . . . . . . . . . . . . . .
 8 . . . . . . . . . . . . . . . . . . .
 7 . . . . . . . . . . . . . . . . . . .
 6 . . . . . . . . . . . . . . . . . . .
 5 . . . . . . . . . . . . . . . . . . .
 4 . . . O . . . . . . . . . . . O . . .
 3 . . . . . . . . . . . . . . . . . . .
 2 . . . . . . . . . . . . . . . . . . .
 1 . . . . . . . . . . . . . . . . . . .
)%%");

    Player nextPla = P_BLACK;
    Rules rules = Rules::parseRules("Japanese");
    rules.komi = 6.5;
    BoardHistory hist(board,nextPla,rules,0);

    SearchParams params0 = SearchParams::forTestsV1();
    params0.maxVisits = 1000;
    SearchParams params1 = params0;
    params1.avoidRepeatedPatternUtility = 0.2;

    TestSearchOptions opts;
    opts.noClearBot = true;

    hist.makeBoardMoveAssumeLegal(board,Location::ofString("R3",board),nextPla,NULL); nextPla = getOpp(nextPla);
    hist.makeBoardMoveAssumeLegal(board,Location::ofString("Q3",board),nextPla,NULL); nextPla = getOpp(nextPla);
    hist.makeBoardMoveAssumeLegal(board,Location::ofString("R4",board),nextPla,NULL); nextPla = getOpp(nextPla);
    hist.makeBoardMoveAssumeLegal(board,Location::ofString("Q5",board),nextPla,NULL); nextPla = getOpp(nextPla);
    hist.makeBoardMoveAssumeLegal(board,Location::ofString("S6",board),nextPla,NULL); nextPla = getOpp(nextPla);
    hist.makeBoardMoveAssumeLegal(board,Location::ofString("F17",board),nextPla,NULL); nextPla = getOpp(nextPla);
    hist.makeBoardMoveAssumeLegal(board,Location::ofString("C14",board),nextPla,NULL); nextPla = getOpp(nextPla);

    AsyncBot* bot0 = new AsyncBot(params0, nnEval, &logger, "pattern bonus");
    AsyncBot* bot1 = new AsyncBot(params1, nnEval, &logger, "pattern bonus");
    AsyncBot* bot2 = new AsyncBot(params1, nnEval, &logger, "pattern bonus");
    cout << "Bot0: baseline" << endl;
    runBotOnPosition(bot0, board, nextPla, hist, opts);
    cout << "Bot1: avoid repeat 0.2" << endl;
    runBotOnPosition(bot1, board, nextPla, hist, opts);
    cout << "Bot2: avoid repeat 0.2 but will only play one side" << endl;
    runBotOnPosition(bot2, board, nextPla, hist, opts);

    //We manually make the bots play through the next moves
    opts.ignorePosition = true;
    hist.makeBoardMoveAssumeLegal(board,Location::ofString("D18",board),nextPla,NULL);
    bot0->makeMove(Location::ofString("D18",board),nextPla);
    bot1->makeMove(Location::ofString("D18",board),nextPla);
    bot2->makeMove(Location::ofString("D18",board),nextPla);
    nextPla = getOpp(nextPla);

    cout << "Bot0: baseline" << endl;
    runBotOnPosition(bot0, board, nextPla, hist, opts);
    cout << "Bot1: avoid repeat 0.2" << endl;
    runBotOnPosition(bot1, board, nextPla, hist, opts);
    cout << "Bot2: avoid repeat 0.2 but will only play one side, skipping" << endl;
    //runBotOnPosition(bot2, board, nextPla, hist, opts);

    hist.makeBoardMoveAssumeLegal(board,Location::ofString("M4",board),nextPla,NULL);
    bot0->makeMove(Location::ofString("M4",board),nextPla);
    bot1->makeMove(Location::ofString("M4",board),nextPla);
    bot2->makeMove(Location::ofString("M4",board),nextPla);
    nextPla = getOpp(nextPla);

    cout << "Bot0: baseline" << endl;
    runBotOnPosition(bot0, board, nextPla, hist, opts);
    cout << "Bot1: avoid repeat 0.2" << endl;
    runBotOnPosition(bot1, board, nextPla, hist, opts);
    cout << "Bot2: avoid repeat 0.2 but will only played one side" << endl;
    runBotOnPosition(bot2, board, nextPla, hist, opts);

    delete bot0;
    delete bot1;
    delete bot2;
  }

  {
    cout << "Pattern bonus does not care about ko =========================================" << endl;

    Board board = Board::parseBoard(19,19,R"%%(
   A B C D E F G H J K L M N O P Q R S T
19 . . . . . . . . . . . . . . . . . . .
18 . . . . . . . . . . . . . . O X O . .
17 . . . . . . . . . . . . . . X O . O .
16 . . . X . . . . . . . . . . . X O X .
15 . . . . . . . . . . . . . . . . . . .
14 . . X . . . . . . . . . . . . . . . .
13 . . . . . . . . . . . . . . . . . . .
12 . . . . . . . . . . . . . . . . . . .
11 . . . . . . . . . . . . . . . . . . .
10 . . . . . . . . . . . . . . . . . . .
 9 . . . . . . . . . . . . . . . . . . .
 8 . . . . . . . . . . . . . . . . . . .
 7 . O . . . . . . . . . . . . . . . . .
 6 . . O . . . . . . . . . . . . . . . .
 5 . . . X . . . . . . . . . . . . . . .
 4 . . O X . . . . . . . . . . . O . . .
 3 . . O X . . X . . . . . . . . . . . .
 2 . . . . . . . . . . . . . . . . . . .
 1 . . . . . . . . . . . . . . . . . . .
)%%");

    Player nextPla = P_BLACK;
    Rules rules = Rules::parseRules("Japanese");
    rules.komi = 6.5;
    BoardHistory hist(board,nextPla,rules,0);

    SearchParams params0 = SearchParams::forTestsV1();
    params0.maxVisits = 1000;
    SearchParams params1 = params0;
    params1.avoidRepeatedPatternUtility = 0.2;

    TestSearchOptions opts;
    opts.noClearBot = true;

    hist.makeBoardMoveAssumeLegal(board,Location::ofString("R17",board),nextPla,NULL); nextPla = getOpp(nextPla);
    hist.makeBoardMoveAssumeLegal(board,Location::ofString("D15",board),nextPla,NULL); nextPla = getOpp(nextPla);
    hist.makeBoardMoveAssumeLegal(board,Location::ofString("C15",board),nextPla,NULL); nextPla = getOpp(nextPla);
    hist.makeBoardMoveAssumeLegal(board,Location::ofString("Q17",board),nextPla,NULL); nextPla = getOpp(nextPla);
    hist.makeBoardMoveAssumeLegal(board,Location::ofString("C5",board),nextPla,NULL); nextPla = getOpp(nextPla);
    hist.makeBoardMoveAssumeLegal(board,Location::ofString("B5",board),nextPla,NULL); nextPla = getOpp(nextPla);

    AsyncBot* bot0 = new AsyncBot(params0, nnEval, &logger, "pattern bonus");
    AsyncBot* bot1 = new AsyncBot(params1, nnEval, &logger, "pattern bonus");
    cout << "Bot0: baseline" << endl;
    runBotOnPosition(bot0, board, nextPla, hist, opts);
    cout << "Bot1: avoid repeat 0.2" << endl;
    runBotOnPosition(bot1, board, nextPla, hist, opts);

    delete bot0;
    delete bot1;
  }

  {
    cout << "Pattern bonus does not multi-count shapes =========================================" << endl;

    Board board = Board::parseBoard(19,19,R"%%(
   A B C D E F G H J K L M N O P Q R S T
19 . . . . . . . . . . . . . . . . . . .
18 . . . . . . . . . . . . . . . . . . .
17 . . . . . . . . . . . . . . . . . . .
16 . . . X . . . . . . . . . . . X . . .
15 . . . . . . . . . . . . . . . . . . .
14 . . . . . . . . . . . . . . . . . . .
13 . . . . . . . . . . . . . . . . . . .
12 . . . . . . . . . . . . . . . . . . .
11 . . . . . . . . . . . . . . . . . . .
10 . . . . . . . . . . . . . . . . . . .
 9 . . . . . . . . . . . . . . . . . . .
 8 . . . . . . . . . . . . . . . . . . .
 7 . . . . . . . . . . . . . . . . . . .
 6 . . . . . . . . . . . . . . . . . . .
 5 . . . . . . . . . . . . . . . . . . .
 4 . . . X . . . . . . . . . . . X . . .
 3 . . . . . . . . . . . . . . . . . . .
 2 . . . . . . . . . . . . . . . . . . .
 1 . . . . . . . . . . . . . . . . . . .
)%%");

    Player nextPla = P_WHITE;
    Rules rules = Rules::parseRules("Japanese");
    rules.komi = 25.5;
    BoardHistory hist(board,nextPla,rules,0);

    SearchParams params0 = SearchParams::forTestsV1();
    params0.maxVisits = 1000;
    params0.avoidRepeatedPatternUtility = 0.2;

    TestSearchOptions opts;
    opts.noClearBot = false;
    AsyncBot* bot0 = new AsyncBot(params0, nnEval, &logger, "pattern bonus");
    runBotOnPosition(bot0, board, nextPla, hist, opts);

    hist.makeBoardMoveAssumeLegal(board,Location::ofString("R14",board),nextPla,NULL);
    bot0->makeMove(Location::ofString("R14",board),nextPla);
    nextPla = getOpp(nextPla);
    hist.makeBoardMoveAssumeLegal(board,Location::ofString("O17",board),nextPla,NULL);
    bot0->makeMove(Location::ofString("O17",board),nextPla);
    nextPla = getOpp(nextPla);

    runBotOnPosition(bot0, board, nextPla, hist, opts);

    hist.makeBoardMoveAssumeLegal(board,Location::ofString("F17",board),nextPla,NULL);
    bot0->makeMove(Location::ofString("F17",board),nextPla);
    nextPla = getOpp(nextPla);
    hist.makeBoardMoveAssumeLegal(board,Location::ofString("C14",board),nextPla,NULL);
    bot0->makeMove(Location::ofString("C14",board),nextPla);
    nextPla = getOpp(nextPla);

    runBotOnPosition(bot0, board, nextPla, hist, opts);

    hist.makeBoardMoveAssumeLegal(board,Location::ofString("C6",board),nextPla,NULL);
    bot0->makeMove(Location::ofString("C6",board),nextPla);
    nextPla = getOpp(nextPla);
    hist.makeBoardMoveAssumeLegal(board,Location::ofString("F3",board),nextPla,NULL);
    bot0->makeMove(Location::ofString("F3",board),nextPla);
    nextPla = getOpp(nextPla);

    runBotOnPosition(bot0, board, nextPla, hist, opts);

    delete bot0;
  }
}

static void runMoreV8Tests2(NNEvaluator* nnEval, Logger& logger)
{
  {
    cout << "TEST ownership endgame ==========================================================================" << endl;

    Player nextPla = P_WHITE;
    Rules rules = Rules::getTrompTaylorish();
    Board board = Board::parseBoard(7,9,R"%%(
x.ooo.x
xxxxxxx
oooooxx
.o..oo.
ooooooo
.oxxxxx
ooox..o
oxxxxxx
xx.....
)%%");
    BoardHistory hist(board,nextPla,rules,0);

    SearchParams params = SearchParams::forTestsV1();
    params.maxVisits = 100;
    params.futileVisitsThreshold = 0.4;
    AsyncBot* bot = new AsyncBot(params, nnEval, &logger, "Endgame ownership test");

    TestSearchOptions opts;
    opts.printOwnership = true;
    runBotOnPosition(bot,board,nextPla,hist,opts);
    delete bot;
  }
}

static void runMoreV8TestsRandomizedNNEvals(NNEvaluator* nnEval, Logger& logger)
{
  {
    cout << "TEST sampled symmetries ==========================================================================" << endl;
    Board board = Board::parseBoard(15,15,R"%%(
...............
...............
...x.....x.....
............o..
...o...........
............x..
...............
...x........o..
...............
....o..........
...............
...o........x..
.....x....o....
...............
...............
)%%");

    Player nextPla = P_BLACK;
    Rules rules = Rules::parseRules("AGA");
    BoardHistory hist(board,nextPla,rules,0);

    SearchParams params = SearchParams::forTestsV1();
    params.rootNumSymmetriesToSample = 8;
    params.maxVisits = 1;

    TestSearchOptions opts;
    opts.printRootPolicy = true;
    {
      cout << "===================================================================" << endl;
      cout << "Repeatedly run bot with 8 root symmetries sampled, should be deterministic (except for scoreUtility due to dynamic score centering)" << endl;
      cout << "===================================================================" << endl;
      AsyncBot* bot = new AsyncBot(params, nnEval, &logger, "sample");
      runBotOnPosition(bot, board, nextPla, hist, opts);
      delete bot;
    }
    {
      cout << "===================================================================" << endl;
      cout << "Repeatedly run bot with 8 root symmetries sampled, should be deterministic (except for scoreUtility due to dynamic score centering)" << endl;
      cout << "===================================================================" << endl;
      AsyncBot* bot = new AsyncBot(params, nnEval, &logger, "sample2");
      runBotOnPosition(bot, board, nextPla, hist, opts);
      delete bot;
    }
    {
      cout << "===================================================================" << endl;
      cout << "Repeatedly run bot with 8 root symmetries sampled, should be deterministic (except for scoreUtility due to dynamic score centering)" << endl;
      cout << "===================================================================" << endl;
      AsyncBot* bot = new AsyncBot(params, nnEval, &logger, "sample3");
      runBotOnPosition(bot, board, nextPla, hist, opts);
      delete bot;
    }

    {
      cout << "===================================================================" << endl;
      cout << "Repeatedly run bot with 2 root symmetries sampled" << endl;
      cout << "===================================================================" << endl;
      SearchParams params2 = params;
      params2.rootNumSymmetriesToSample = 2;
      AsyncBot* bot = new AsyncBot(params2, nnEval, &logger, "two root syms");
      bot->setPosition(nextPla, board, hist);
      set<double> policySamples;
      set<double> wlSamples;

      float policyProbs[NNPos::MAX_NN_POLICY_SIZE];
      for(int i = 0; i<500; i++) {
        bot->genMoveSynchronous(nextPla,TimeControls());
        bot->getSearch()->getPolicy(policyProbs);
        policySamples.insert(policyProbs[NNPos::xyToPos(2,4,bot->getSearch()->nnXLen)]);
        wlSamples.insert(bot->getSearch()->getRootValuesRequireSuccess().winLossValue);
        bot->clearSearch();
      }
      delete bot;
      int i;
      i = 0;
      cout << "Policy samples" << endl;
      for(double x: policySamples)
        cout << (i++) << " " << x << endl;
      i = 0;
      cout << "WL samples" << endl;
      for(double x: wlSamples)
        cout << (i++) << " " << x << endl;
    }

  }

}

static void runV8SearchMultithreadTest(NNEvaluator* nnEval, Logger& logger)
{
  {
    cout << "Multithreaded search test =========================================" << endl;

    Board board = Board::parseBoard(19,19,R"%%(
...................
...................
...................
...x...........x...
...................
...................
...................
...................
...................
...................
...................
...................
...................
...................
...................
..oo...........o...
..xxo..............
...................
...................
)%%");

    Player nextPla = P_BLACK;
    Rules rules = Rules::parseRules("Japanese");
    rules.komi = 8.5;
    BoardHistory hist(board,nextPla,rules,0);

    SearchParams params = SearchParams::forTestsV1();
    params.maxVisits = 16000;
    params.subtreeValueBiasFactor = 0.35;
    params.subtreeValueBiasWeightExponent = 0.8;
    params.subtreeValueBiasFreeProp = 0.8;
    params.chosenMoveTemperature = 0;
    params.chosenMoveTemperatureEarly = 0;
    params.useNoisePruning = true;
    params.numThreads = 40;

    Loc moveLoc;

    Search* search = new Search(params, nnEval, &logger, "multithreaded test");
    search->setPosition(nextPla,board,hist);
    search->runWholeSearch(nextPla);
    moveLoc = search->getChosenMoveLoc();
    cout << "Chosen move: " << Location::toString(moveLoc,board) << endl;
    cout << "Winloss near 0.05: " << (std::fabs(search->getRootValuesRequireSuccess().winLossValue - (0.05)) < 0.05) << endl;
    cout << "Lead near 1: " << (std::fabs(search->getRootValuesRequireSuccess().lead - (1.0)) < 1.0) << endl;

    // PrintTreeOptions options;
    // options = options.maxDepth(1);
    // printBasicStuffAfterSearch(board,hist,search,options);

    search->makeMove(moveLoc,nextPla);
    hist.makeBoardMoveAssumeLegal(board,moveLoc,nextPla,NULL);
    nextPla = getOpp(nextPla);
    search->runWholeSearch(nextPla);
    moveLoc = search->getChosenMoveLoc();
    cout << "Chosen move: " << Location::toString(moveLoc,board) << endl;
    cout << "Winloss near 0.05: " << (std::fabs(search->getRootValuesRequireSuccess().winLossValue - (0.05)) < 0.05) << endl;
    cout << "Lead near 1: " << (std::fabs(search->getRootValuesRequireSuccess().lead - (1.0)) < 1.0) << endl;

    // PrintTreeOptions options;
    // options = options.maxDepth(1);
    // printBasicStuffAfterSearch(board,hist,search,options);

    search->makeMove(moveLoc,nextPla);
    hist.makeBoardMoveAssumeLegal(board,moveLoc,nextPla,NULL);
    nextPla = getOpp(nextPla);
    search->runWholeSearch(nextPla);
    moveLoc = search->getChosenMoveLoc();
    cout << "Chosen move: " << Location::toString(moveLoc,board) << endl;
    cout << "Winloss near 0.05: " << (std::fabs(search->getRootValuesRequireSuccess().winLossValue - (0.05)) < 0.05) << endl;
    cout << "Lead near 1: " << (std::fabs(search->getRootValuesRequireSuccess().lead - (1.0)) < 1.0) << endl;

    // PrintTreeOptions options;
    // options = options.maxDepth(1);
    // printBasicStuffAfterSearch(board,hist,search,options);

    //Enumerate the tree and make sure every node is indeed hit exactly once in postorder.
    TestSearchCommon::verifyTreePostOrder(search,16000);

    //With 16000 visits per move and three searches, we very likely have about that many nn evals (this shouldn't be a heavily transposing position)
    //and despite doing 3x such searches, we should have caching and tree reuse keep it not much more than that.
    int64_t numRowsProcessed = nnEval->numRowsProcessed();
    cout << "numRowsProcessed as expected: " << (numRowsProcessed > 14000 && numRowsProcessed < 28000) << endl;

    delete search;
    nnEval->clearCache(); nnEval->clearStats();
  }

  {
    cout << "Multithreaded graph search test =========================================" << endl;

    Board board = Board::parseBoard(19,19,R"%%(
...................
....x.x......xxoo..
...xox.....x.o.xo..
..x.o..........xo..
..xo...........xx..
..oo............o..
...................
...................
..o................
...................
...................
...................
................x..
...............o...
.o.............ox..
o.oo............x..
xoxxoo........o.x..
.x.................
...................
)%%");

    Player nextPla = P_WHITE;
    Rules rules = Rules::parseRules("Chinese");
    rules.komi = 6.5;
    BoardHistory hist(board,nextPla,rules,0);

    SearchParams params = SearchParams::forTestsV1();
    params.maxVisits = 8000;
    params.useGraphSearch = true;
    params.subtreeValueBiasFactor = 0.35;
    params.subtreeValueBiasWeightExponent = 0.8;
    params.subtreeValueBiasFreeProp = 0.8;
    params.chosenMoveTemperature = 0;
    params.chosenMoveTemperatureEarly = 0;
    params.useNoisePruning = true;
    params.numThreads = 40;

    Loc moveLoc;

    Search* search = new Search(params, nnEval, &logger, "multithreaded test");
    search->setPosition(nextPla,board,hist);
    search->runWholeSearch(nextPla);
    moveLoc = search->getChosenMoveLoc();
    cout << "Chosen move: " << Location::toString(moveLoc,board) << endl;
    cout << "Winloss near 0.63: " << (std::fabs(search->getRootValuesRequireSuccess().winLossValue - (0.63)) < 0.1) << endl;
    cout << "Lead near 15.5: " << (std::fabs(search->getRootValuesRequireSuccess().lead - (15.8)) < 1.8) << endl;

    // PrintTreeOptions options;
    // options = options.maxDepth(1);
    // printBasicStuffAfterSearch(board,hist,search,options);

    //Enumerate the tree and make sure every node is indeed hit exactly once in postorder.
    TestSearchCommon::verifyTreePostOrder(search,5000);

    search->makeMove(moveLoc,nextPla);
    hist.makeBoardMoveAssumeLegal(board,moveLoc,nextPla,NULL);
    nextPla = getOpp(nextPla);
    search->runWholeSearch(nextPla);
    moveLoc = search->getChosenMoveLoc();
    cout << "Chosen move: " << Location::toString(moveLoc,board) << endl;
    cout << "Winloss near 0.63: " << (std::fabs(search->getRootValuesRequireSuccess().winLossValue - (0.63)) < 0.1) << endl;
    cout << "Lead near 15.5: " << (std::fabs(search->getRootValuesRequireSuccess().lead - (15.8)) < 1.8) << endl;

    // PrintTreeOptions options;
    // options = options.maxDepth(1);
    // printBasicStuffAfterSearch(board,hist,search,options);

    //Enumerate the tree and make sure every node is indeed hit exactly once in postorder.
    TestSearchCommon::verifyTreePostOrder(search,5000);

    //With 8000 visits per move and 2 searches, we very likely have about that many nn evals
    int64_t numRowsProcessed = nnEval->numRowsProcessed();
    cout << "numRowsProcessed as expected: " << (numRowsProcessed > 5000 && numRowsProcessed < 13000) << endl;

    delete search;
    nnEval->clearCache(); nnEval->clearStats();
  }
}


void Tests::runSearchTestsV8(const string& modelFile, bool inputsNHWC, bool useNHWC, bool useFP16) {
  TestCommon::overrideForBackends(inputsNHWC, useNHWC);
  cout << "Running search tests introduced after v8 nets" << endl;
  NeuralNet::globalInitialize();

  const bool logToStdOut = true;
  const bool logToStdErr = false;
  const bool logTime = false;
  Logger logger(nullptr, logToStdOut, logToStdErr, logTime);

  NNEvaluator* nnEval;
  NNEvaluator* nnEval9;
  NNEvaluator* nnEval19Exact;
  NNEvaluator* nnEval9Exact;

  nnEval = startNNEval(
    modelFile,logger,"v8seed",19,19,1,inputsNHWC,useNHWC,useFP16,false,false);
  nnEval9 = startNNEval(
    modelFile,logger,"v8seed",9,9,1,inputsNHWC,useNHWC,useFP16,false,false);
  nnEval9Exact = startNNEval(
    modelFile,logger,"v8seed",9,9,1,inputsNHWC,useNHWC,useFP16,false,true);
  runV8TestsSize9(nnEval,nnEval9,nnEval9Exact,logger);
  delete nnEval;
  delete nnEval9;
  delete nnEval9Exact;
  nnEval = NULL;
  nnEval9 = NULL;
  nnEval9Exact = NULL;

  nnEval = startNNEval(
    modelFile,logger,"v8seed",19,19,-1,inputsNHWC,useNHWC,useFP16,false,false);
  nnEval19Exact = startNNEval(
    modelFile,logger,"v8seed",19,19,-1,inputsNHWC,useNHWC,useFP16,false,true);
  runV8TestsRandomSym(nnEval,nnEval19Exact,logger);
  delete nnEval;
  delete nnEval19Exact;
  nnEval = NULL;
  nnEval19Exact = NULL;

  nnEval = startNNEval(
    modelFile,logger,"v8seed",19,19,2,inputsNHWC,useNHWC,useFP16,false,false);
  runV8Tests(nnEval,logger);
  delete nnEval;
  nnEval = NULL;

  nnEval = startNNEval(
    modelFile,logger,"v8seed",19,19,5,inputsNHWC,useNHWC,useFP16,false,false);
  runMoreV8Tests(nnEval,logger);
  delete nnEval;
  nnEval = NULL;

  nnEval = startNNEval(
    modelFile,logger,"v8seed",19,19,5,inputsNHWC,useNHWC,useFP16,false,false);
  runMoreV8Tests2(nnEval,logger);
  delete nnEval;
  nnEval = NULL;

  nnEval = startNNEval(
    modelFile,logger,"v8seed",19,19,-1,inputsNHWC,useNHWC,useFP16,false,false);
  runMoreV8TestsRandomizedNNEvals(nnEval,logger);
  delete nnEval;
  nnEval = NULL;

  nnEval = startNNEval(
    modelFile,logger,"v8seed",19,19,-1,inputsNHWC,useNHWC,useFP16,false,false);
  runV8SearchMultithreadTest(nnEval,logger);
  // Suppress some nondeterministc messages about number of batches
  logger.setDisabled(true);
  delete nnEval;
  nnEval = NULL;

  NeuralNet::globalCleanup();
  cout << "Done" << endl;
}
