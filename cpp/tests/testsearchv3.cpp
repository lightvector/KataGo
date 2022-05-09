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

static void runOwnershipAndMisc(NNEvaluator* nnEval, NNEvaluator* nnEval11, NNEvaluator* nnEvalPTemp, Logger& logger)
{
  {
    cout << "GAME 5 ==========================================================================" << endl;
    cout << "(A simple opening to test neural net outputs including ownership map)" << endl;

    string sgfStr = "(;FF[4]CA[UTF-8]KM[7.5];B[pp];W[pc];B[cd];W[dq];B[ed];W[pe];B[co];W[cp];B[do];W[fq];B[ck];W[qn];B[qo];W[pn];B[np];W[qj];B[jc];W[lc];B[je];W[lq];B[mq];W[lp];B[ek];W[qq];B[pq];W[ro];B[rp];W[qp];B[po];W[rq];B[rn];W[sp];B[rm];W[ql];B[on];W[om];B[nn];W[nm];B[mn];W[ip];B[mm])";
    CompactSgf* sgf = CompactSgf::parse(sgfStr);

    Board board;
    Player nextPla;
    BoardHistory hist;
    Rules initialRules = sgf->getRulesOrFailAllowUnspecified(Rules::getTrompTaylorish());
    sgf->setupBoardAndHistAssumeLegal(initialRules, board, nextPla, hist, 40);

    MiscNNInputParams nnInputParams;
    NNResultBuf buf;
    bool skipCache = true;
    bool includeOwnerMap = true;
    nnEval->evaluate(board,hist,nextPla,nnInputParams,buf,skipCache,includeOwnerMap);

    printPolicyValueOwnership(board,buf);

    nnEval->clearCache();
    nnEval->clearStats();
    cout << endl << endl;

    cout << "With root temperature===================" << endl;
    nnInputParams.nnPolicyTemperature = 1.5f;
    nnEvalPTemp->evaluate(board,hist,nextPla,nnInputParams,buf,skipCache,includeOwnerMap);

    printPolicyValueOwnership(board,buf);

    nnEvalPTemp->clearCache();
    nnEvalPTemp->clearStats();
    cout << endl << endl;

    SearchParams params;
    params.maxVisits = 200;
    AsyncBot* bot = new AsyncBot(params, nnEval, &logger, "seed");
    TestSearchOptions opts;
    opts.printOwnership = true;
    runBotOnSgf(bot, sgfStr, initialRules, 40, 7.5, opts);
    cout << endl << endl;

    delete sgf;
  }

  {
    cout << "GAME 6 ==========================================================================" << endl;
    cout << "(A simple smaller game, also testing invariance under nnlen)" << endl;

    string sgfStr = "(;FF[4]CA[UTF-8]SZ[11]KM[7.5];B[ci];W[ic];B[ih];W[hi];B[ii];W[ij];B[jj];W[gj];B[ik];W[di];B[hh];W[ch];B[dc];W[cc];B[cb];W[cd];B[eb];W[dd];B[ed];W[ee];B[fd];W[bb];B[ba];W[ab];B[gb];W[je];B[ib];W[jb];B[jc];W[jd];B[hc];W[id];B[dh];W[cg];B[dj];W[ei];B[bi];W[ia];B[hb];W[fg];B[hj];W[eh];B[ej])";
    CompactSgf* sgf = CompactSgf::parse(sgfStr);

    Board board;
    Player nextPla;
    BoardHistory hist;
    Rules initialRules = sgf->getRulesOrFailAllowUnspecified(Rules::getTrompTaylorish());
    sgf->setupBoardAndHistAssumeLegal(initialRules, board, nextPla, hist, 43);

    MiscNNInputParams nnInputParams;
    NNResultBuf buf;
    bool skipCache = true;
    bool includeOwnerMap = true;
    nnEval->evaluate(board,hist,nextPla,nnInputParams,buf,skipCache,includeOwnerMap);
    printPolicyValueOwnership(board,buf);

    cout << "NNLen 11" << endl;
    NNResultBuf buf11;
    nnEval11->evaluate(board,hist,nextPla,nnInputParams,buf11,skipCache,includeOwnerMap);
    testAssert(buf11.result->nnXLen == 11);
    testAssert(buf11.result->nnYLen == 11);
    printPolicyValueOwnership(board,buf11);

    nnEval->clearCache();
    nnEval->clearStats();
    nnEval11->clearCache();
    nnEval11->clearStats();
    delete sgf;
    cout << endl << endl;
  }

  {
    cout << "GAME 7 ==========================================================================" << endl;
    cout << "(Simple extension of game 6 to test root ending bonus points)" << endl;

    SearchParams params;
    params.maxVisits = 500;
    params.fpuReductionMax = 0.0;
    params.rootFpuReductionMax = 0.0;
    AsyncBot* bot = new AsyncBot(params, nnEval, &logger, getSearchRandSeed());
    Rules rules = Rules::getTrompTaylorish();
    TestSearchOptions opts;
    opts.printEndingScoreValueBonus = true;

    string sgfStr = "(;FF[4]CA[UTF-8]SZ[11]KM[7.5];B[ci];W[ic];B[ih];W[hi];B[ii];W[ij];B[jj];W[gj];B[ik];W[di];B[hh];W[ch];B[dc];W[cc];B[cb];W[cd];B[eb];W[dd];B[ed];W[ee];B[fd];W[bb];B[ba];W[ab];B[gb];W[je];B[ib];W[jb];B[jc];W[jd];B[hc];W[id];B[dh];W[cg];B[dj];W[ei];B[bi];W[ia];B[hb];W[fg];B[hj];W[eh];B[ej];W[fj];B[bh];W[bg];B[fe];W[ef];B[jf];W[kc];B[ke];W[ja];B[if];W[fi];B[gg];W[ek];B[ck];W[bj];B[aj];W[bk];B[ah];W[ag];B[cj];W[he];B[hf];W[hd];B[ff];W[kd];B[kf];W[ha];B[gd];W[ga];B[fa];W[gi];B[hk];W[gh];B[ca];W[gk];B[aa];W[bc];B[ge];W[ig];B[fc];W[ka];B[da];W[jg];B[de];W[ce];B[ak];W[ie];B[dk];W[fk];B[hg];W[dg];B[jh];W[ad])";

    cout << "With root ending bonus pts===================" << endl;
    cout << endl;
    SearchParams params2 = params;
    params2.rootEndingBonusPoints = 0.5;
    bot->setParams(params2);
    runBotOnSgf(bot, sgfStr, rules, 88, 7.5, opts);
    cout << endl << endl;

    cout << "With root ending bonus pts one step later===================" << endl;
    cout << endl;
    bot->setParams(params2);
    runBotOnSgf(bot, sgfStr, rules, 89, 7.5, opts);

    cout << "Without root ending bonus pts later later===================" << endl;
    cout << endl;
    bot->setParams(params);
    runBotOnSgf(bot, sgfStr, rules, 96, 7.5, opts);

    cout << "With root ending bonus pts later later===================" << endl;
    cout << endl;
    bot->setParams(params2);
    runBotOnSgf(bot, sgfStr, rules, 96, 7.5, opts);

    delete bot;
  }

  {
    cout << "GAME 8 ==========================================================================" << endl;
    cout << "(Alternate variation of game 7 to test root ending bonus points in territory scoring)" << endl;

    SearchParams params;
    params.maxVisits = 500;
    params.fpuReductionMax = 0.0;
    params.rootFpuReductionMax = 0.0;
    AsyncBot* bot = new AsyncBot(params, nnEval, &logger, getSearchRandSeed());
    Rules rules = Rules::getSimpleTerritory();
    TestSearchOptions opts;
    opts.printEndingScoreValueBonus = true;

    string sgfStr = "(;FF[4]CA[UTF-8]SZ[11]KM[7.5];B[ci];W[ic];B[ih];W[hi];B[ii];W[ij];B[jj];W[gj];B[ik];W[di];B[hh];W[ch];B[dc];W[cc];B[cb];W[cd];B[eb];W[dd];B[ed];W[ee];B[fd];W[bb];B[ba];W[ab];B[gb];W[je];B[ib];W[jb];B[jc];W[jd];B[hc];W[id];B[dh];W[cg];B[dj];W[ei];B[bi];W[ia];B[hb];W[fg];B[hj];W[eh];B[ej];W[fj];B[bh];W[bg];B[fe];W[ef];B[jf];W[kc];B[ke];W[ja];B[if];W[fi];B[gg];W[ek];B[ck];W[bj];B[aj];W[bk];B[ah];W[ag];B[cj];W[he];B[hf];W[hd];B[ff];W[kd];B[kf];W[ha];B[gd];W[ga];B[fa];W[gi];B[hk];W[gh];B[ca];W[gk];B[aa];W[bc];B[ge];W[ig];B[fc];W[ka];B[da];W[jg];B[de];W[ce];B[ak];W[hg];B[gf])";

    cout << "Without root ending bonus pts===================" << endl;
    cout << endl;
    bot->setParams(params);
    runBotOnSgf(bot, sgfStr, rules, 91, 7.5, opts);

    cout << "With root ending bonus pts===================" << endl;
    cout << endl;
    SearchParams params2 = params;
    params2.rootEndingBonusPoints = 0.5;
    bot->setParams(params2);
    runBotOnSgf(bot, sgfStr, rules, 91, 7.5, opts);
    cout << endl << endl;

    delete bot;
  }

  {
    cout << "GAME 9 ==========================================================================" << endl;
    cout << "(A game to visualize root noise)" << endl;
    cout << endl;

    SearchParams params;
    params.maxVisits = 1;
    AsyncBot* bot = new AsyncBot(params, nnEval, &logger, getSearchRandSeed());
    Rules rules = Rules::getSimpleTerritory();
    TestSearchOptions opts;
    opts.printRootPolicy = true;

    string sgfStr = "(;FF[4]CA[UTF-8]SZ[15]KM[7.5];B[lm];W[lc];B[dm];W[dc];B[me];W[md];B[le];W[jc];B[lk];W[ck];B[cl];W[dk];B[fm];W[de];B[dg];W[fk];B[fg];W[hk];B[hm];W[hg];B[ci];W[cf];B[cg];W[mh];B[kh];W[mj];B[lj];W[mk];B[ml];W[nl];B[nm];W[nk];B[hi];W[gh];B[gi];W[fh];B[fi];W[eh];B[ei];W[dh];B[di];W[ch];B[bh];W[eg];B[bg];W[kg];B[jg];W[kf];B[lh];W[mg];B[lg];W[lf];B[mf];W[ke];B[kd];W[je];B[ld];W[jd];B[mc];W[nd];B[kc];W[lb];B[kb];W[mb];B[ne];W[nc];B[ng];W[nh];B[if];W[jh];B[ih];W[ji];B[li];W[ig];B[ij];W[og];B[ef];W[ff];B[ee];W[df];B[fe];W[gf];B[ec];W[db];B[dd];W[cd];B[ed];W[bf];B[eb];W[he];B[gd];W[hc];B[gc];W[hd];B[gb];W[hb];B[fc];W[fa];B[ea];W[bc];B[ga];W[jj];B[ik];W[jk];B[il];W[jl];B[jm];W[gg];B[ge];W[kl];B[km];W[bl];B[ll];W[cn])";

    runBotOnSgf(bot, sgfStr, rules, 114, 6.5, opts);

    cout << "With noise===================" << endl;
    cout << "Adding root noise to the search" << endl;
    cout << endl;

    SearchParams testParams = params;
    testParams.rootNoiseEnabled = true;
    bot->setParams(testParams);
    runBotOnSgf(bot, sgfStr, rules, 114, 6.5, opts);
    bot->setParams(params);
    cout << endl << endl;

    delete bot;
  }

  {
    cout << "GAME 10 ==========================================================================" << endl;
    cout << "(Tricky endgame seki invasion, testing LCB and dynamic utility recompute)" << endl;
    cout << endl;

    SearchParams params;
    params.maxVisits = 280;
    params.staticScoreUtilityFactor = 0.2;
    params.dynamicScoreUtilityFactor = 0.3;
    params.useLcbForSelection = true;
    AsyncBot* bot = new AsyncBot(params, nnEval, &logger, getSearchRandSeed());
    Rules rules = Rules::getTrompTaylorish();
    TestSearchOptions opts;


    string sgfStr = "(;GM[1]FF[4]CA[UTF-8]SZ[19]HA[6]KM[0.5]AB[dc][oc][qd][ce][qo][pq];W[cp];B[ep];W[eq];B[fq];W[dq];B[fp];W[dn];B[jq];W[jp];B[ip];W[kq];B[iq];W[kp];B[fm];W[io];B[ho];W[in];B[en];W[dm];B[hn];W[oq];B[op];W[pr];B[pp];W[or];B[qr];W[mq];B[mo];W[qj];B[ql];W[qe];B[rd];W[qg];B[pe];W[ic];B[gc];W[lc];B[ch];W[cj];B[eh];W[ec];B[eb];W[dd];B[ed];W[cc];B[fc];W[db];B[cd];W[ec];B[de];W[dc];B[gb];W[ea];B[fb];W[bb];B[bd];W[ca];B[bc];W[ab];B[ee];W[nc];B[nd];W[ob];B[nb];W[mc];B[pb];W[od];B[pc];W[ne];B[md];W[le];B[oe];W[rl];B[rm];W[rk];B[qm];W[ie];B[me];W[mf];B[nf];W[ld];B[pd];W[ge];B[hd];W[he];B[fd];W[mg];B[id];W[jd];B[hh];W[bi];B[bh];W[ln];B[im];W[jm];B[jl];W[km];B[lo];W[ko];B[il];W[ek];B[dp];W[cq];B[do];W[co];B[fj];W[jh];B[ig];W[jg];B[nm];W[re];B[se];W[rf];B[pj];W[pi];B[oj];W[qk];B[oi];W[ph];B[mb];W[pk];B[ol];W[ok];B[nk];W[nj];B[mj];W[ni];B[mi];W[nh];B[mk];W[er];B[lb];W[kb];B[fr];W[fk];B[ff];W[di];B[ci];W[bj];B[ei];W[dj];B[dh];W[sf];B[jr];W[kr];B[sd];W[qs];B[rr];W[gl];B[gm];W[ib];B[ks];W[ls];B[js];W[np];B[no];W[pl];B[pm];W[if];B[mp];W[mr];B[nq];W[nr];B[gg];W[rs];B[og];W[oh];B[mn];W[ll];B[lh];W[ih];B[hg];W[ml];B[nl];W[gj];B[kl];W[lk];B[gi];W[ej];B[fi];W[hl];B[hj];W[lg];B[gk];W[fl];B[hk];W[em];B[hm];W[sm];B[sn];W[sl];B[sp];W[la];B[kj];W[pf];B[of];W[ii];B[lj];W[lm];B[kh];W[kg];B[fa];W[da];B[jj];W[fs];B[gs];W[es];B[ha];W[ia];B[ij];W[ah];B[ag];W[ai];B[pg];W[qf];B[lp];W[lq];B[hb];W[kk];B[jk];W[ac];B[ad];W[ji];B[ki];W[ka];B[oa];W[ma];B[na];W[sr];B[sq];W[ps];B[ss];W[np];B[sr];W[nq];B[mh];W[ng];B[fe];W[jn];B[mm];W[gr];B[hs];W[fn];B[eo];W[hr];B[is];W[gp];B[go];W[gq];B[hp];W[fo];B[])";

    opts.noClearBot = true;
    runBotOnSgf(bot, sgfStr, rules, 234, 0.5, opts);

    //Try to check that search tree is idempotent under simply rebeginning the search
    Search* search = bot->getSearchStopAndWait();
    PrintTreeOptions options;
    options = options.maxDepth(1);
    cout << "Beginning search again and then reprinting, should be same" << endl;
    search->beginSearch(false);
    search->printTree(cout, search->rootNode, options, P_WHITE);
    cout << "Making a move O3, should still be same" << endl;
    bot->makeMove(Location::ofString("O3",19,19), P_WHITE);
    search->printTree(cout, search->rootNode, options, P_WHITE);
    cout << "Beginning search again and then reprinting, now score utils should change a little" << endl;
    search->beginSearch(false);
    search->printTree(cout, search->rootNode, options, P_WHITE);

    delete bot;
  }


  {
    cout << "GAME 11 ==========================================================================" << endl;
    cout << "(Non-square board)" << endl;
    cout << endl;

    Rules rules = Rules::getTrompTaylorish();
    Player nextPla = P_BLACK;
    Board boardA = Board::parseBoard(7,11,R"%%(
.......
.......
..x.o..
.......
.......
...xo..
.......
..xx...
..oox..
....o..
.......
)%%");
    BoardHistory histA(boardA,nextPla,rules,0);

    Board boardB = Board::parseBoard(11,7,R"%%(
...........
...........
..x.o.ox...
.......ox..
.......ox..
...........
...........
)%%");
    BoardHistory histB(boardB,nextPla,rules,0);

    SearchParams params;
    params.maxVisits = 200;
    params.dynamicScoreUtilityFactor = 0.25;

    AsyncBot* botA = new AsyncBot(params, nnEval, &logger, getSearchRandSeed());
    AsyncBot* botB = new AsyncBot(params, nnEval, &logger, getSearchRandSeed());

    TestSearchOptions opts;
    runBotOnPosition(botA,boardA,nextPla,histA,opts);
    runBotOnPosition(botB,boardB,nextPla,histB,opts);
    delete botA;
    delete botB;

    cout << endl;
    cout << "NNLen 11" << endl;
    cout << endl;
    AsyncBot* botA11 = new AsyncBot(params, nnEval11, &logger, getSearchRandSeed());
    AsyncBot* botB11 = new AsyncBot(params, nnEval11, &logger, getSearchRandSeed());
    runBotOnPosition(botA11,boardA,nextPla,histA,opts);
    runBotOnPosition(botB11,boardB,nextPla,histB,opts);

    delete botA11;
    delete botB11;
  }


  {
    cout << "GAME 12 ==========================================================================" << endl;
    cout << "(MultiStoneSuicide rules)" << endl;
    cout << endl;

    string seed = getSearchRandSeed();
    for(int i = 0; i <= 1; i++) {
      Rules rules = Rules::getTrompTaylorish();
      rules.komi = 0.5;
      if(i == 1)
        rules.multiStoneSuicideLegal = false;
      cout << rules << endl;

      Player nextPla = P_WHITE;
      Board board = Board::parseBoard(9,9,R"%%(
..ox..xx.
.ooxxxx.x
o..o..oxo
.oooooooo
.xxxxxxxx
....x.x..
.x.x.x.oo
....x.oox
......ox.
)%%");
      BoardHistory hist(board,nextPla,rules,0);
      hist.makeBoardMoveAssumeLegal(board,Location::ofString("H8",board),nextPla,NULL);
      nextPla = P_BLACK;

      SearchParams params;
      params.maxVisits = 200;

      AsyncBot* bot = new AsyncBot(params, nnEval, &logger, seed);

      TestSearchOptions opts;
      runBotOnPosition(bot,board,nextPla,hist,opts);
      delete bot;
    }
  }

  {
    cout << "GAME 13 ==========================================================================" << endl;
    cout << "(Conservative pass)" << endl;
    cout << endl;

    string seed = "abc";
    Rules rules = Rules::getTrompTaylorish();
    rules.komi = 0;

    Player nextPla = P_BLACK;
    Board board = Board::parseBoard(9,9,R"%%(
.........
..x...x..
.........
xxxxxxxx.
ooooooooo
...o.o.o.
xx.o.o.o.
.xxo.o.o.
..xo.o.o.
)%%");
    BoardHistory hist(board,nextPla,rules,0);
    hist.makeBoardMoveAssumeLegal(board,Board::PASS_LOC,nextPla,NULL);
    nextPla = P_WHITE;

    {
      cout << "conservativePass=false" << endl;
      SearchParams params;
      params.maxVisits = 80;
      params.rootFpuReductionMax = 0.0;
      params.rootPolicyTemperature = 1.5;
      params.rootPolicyTemperatureEarly = 1.5;
      params.rootNoiseEnabled = true;
      AsyncBot* bot = new AsyncBot(params, nnEval, &logger, seed);
      TestSearchOptions opts;
      runBotOnPosition(bot,board,nextPla,hist,opts);
      delete bot;
    }

    {
      cout << "conservativePass=true" << endl;
      SearchParams params;
      params.maxVisits = 80;
      params.conservativePass = true;
      params.rootFpuReductionMax = 0.0;
      params.rootPolicyTemperature = 1.5;
      params.rootPolicyTemperatureEarly = 1.5;
      params.rootNoiseEnabled = true;
      AsyncBot* bot = new AsyncBot(params, nnEval, &logger, seed);
      TestSearchOptions opts;
      runBotOnPosition(bot,board,nextPla,hist,opts);
      delete bot;
    }
  }

  {
    cout << "GAME 14 ==========================================================================" << endl;
    cout << "Root noise and temperature across moves" << endl;
    cout << endl;

    string seed = getSearchRandSeed();
    Rules rules = Rules::getTrompTaylorish();
    rules.komi = 5.5;
    TestSearchOptions opts;
    opts.noClearBot = true;

    Player nextPla = P_WHITE;
    Board board = Board::parseBoard(9,9,R"%%(
.........
.........
....o....
..x......
....x.x..
..xo.....
.....o...
.........
.........
)%%");
    BoardHistory hist(board,nextPla,rules,0);

    SearchParams params;
    params.maxVisits = 200;
    params.rootPolicyTemperature = 2.5;
    params.rootPolicyTemperatureEarly = 2.5;
    params.rootNoiseEnabled = true;
    AsyncBot* bot = new AsyncBot(params, nnEval, &logger, seed);
    bot->setAlwaysIncludeOwnerMap(true);

    runBotOnPosition(bot,board,nextPla,hist,opts);
    hist.makeBoardMoveAssumeLegal(board,Location::ofString("D5",board),nextPla,NULL);
    bot->getSearch()->printTree(cout, bot->getSearch()->rootNode, PrintTreeOptions().onlyBranch(board,"D5"), P_WHITE);
    bot->makeMove(Location::ofString("D5",board),nextPla);
    nextPla = getOpp(nextPla);
    opts.ignorePosition = true;
    runBotOnPosition(bot,board,nextPla,hist,opts);

    delete bot;
  }

  {
    cout << "GAME 15 ==========================================================================" << endl;
    cout << "Japanese rules endgame, one dame" << endl;
    cout << endl;

    string seed = getSearchRandSeed();
    Rules rules = Rules::parseRules("Japanese");
    rules.komi = 6;
    TestSearchOptions opts;
    opts.noClearBot = true;

    Player nextPla = P_BLACK;
    Board board = Board::parseBoard(9,7,R"%%(
.........
ooooo.o..
oxxxox...
xx..xoooo
..xx.x.xo
.oox.xxxx
..x...ox.
)%%");
    board.numWhiteCaptures = 3;
    BoardHistory hist(board,nextPla,rules,0);

    MiscNNInputParams nnInputParams;
    NNResultBuf buf;
    bool skipCache = true;
    bool includeOwnerMap = true;
    nnEval->evaluate(board,hist,nextPla,nnInputParams,buf,skipCache,includeOwnerMap);
    printPolicyValueOwnership(board,buf);

    SearchParams params;
    params.maxVisits = 200;
    AsyncBot* bot = new AsyncBot(params, nnEval, &logger, seed);

    runBotOnPosition(bot,board,nextPla,hist,opts);
    bot->getSearch()->printTree(cout, bot->getSearch()->rootNode, PrintTreeOptions().onlyBranch(board,"G3"), P_BLACK);
    delete bot;
  }

  {
    cout << "GAME 16 ==========================================================================" << endl;
    cout << "Chinese rules endgame, one dame" << endl;
    cout << endl;

    string seed = getSearchRandSeed();
    Rules rules = Rules::parseRules("Chinese");
    rules.komi = 6;
    TestSearchOptions opts;
    opts.noClearBot = true;

    Player nextPla = P_BLACK;
    Board board = Board::parseBoard(9,7,R"%%(
.........
ooooo.o..
oxxxox...
xx..xoooo
..xx.x.xo
.oox.xxxx
..x...ox.
)%%");
    board.numWhiteCaptures = 3;
    BoardHistory hist(board,nextPla,rules,0);

    MiscNNInputParams nnInputParams;
    NNResultBuf buf;
    bool skipCache = true;
    bool includeOwnerMap = true;
    nnEval->evaluate(board,hist,nextPla,nnInputParams,buf,skipCache,includeOwnerMap);
    printPolicyValueOwnership(board,buf);

    SearchParams params;
    params.maxVisits = 200;
    AsyncBot* bot = new AsyncBot(params, nnEval, &logger, seed);

    runBotOnPosition(bot,board,nextPla,hist,opts);
    bot->getSearch()->printTree(cout, bot->getSearch()->rootNode, PrintTreeOptions().onlyBranch(board,"G3"), P_BLACK);
    delete bot;
  }
}

void Tests::runSearchTestsV3(const string& modelFile, bool inputsNHWC, bool useNHWC, int symmetry, bool useFP16) {
  TestCommon::overrideForBackends(inputsNHWC, useNHWC);
  cout << "Running search tests specifically for v3 or later nets" << endl;
  NeuralNet::globalInitialize();

  const bool logToStdOut = true;
  const bool logToStdErr = false;
  const bool logTime = false;
  Logger logger(nullptr, logToStdOut, logToStdErr, logTime);

  NNEvaluator* nnEval = startNNEval(modelFile,logger,"",NNPos::MAX_BOARD_LEN,NNPos::MAX_BOARD_LEN,symmetry,inputsNHWC,useNHWC,useFP16,false,false);
  NNEvaluator* nnEval11 = startNNEval(modelFile,logger,"",11,11,symmetry,inputsNHWC,useNHWC,useFP16,false,false);
  NNEvaluator* nnEvalPTemp = startNNEval(modelFile,logger,"",NNPos::MAX_BOARD_LEN,NNPos::MAX_BOARD_LEN,symmetry,inputsNHWC,useNHWC,useFP16,false,false);
  runOwnershipAndMisc(nnEval,nnEval11,nnEvalPTemp,logger);
  delete nnEval;
  delete nnEval11;
  delete nnEvalPTemp;

  NeuralNet::globalCleanup();
  cout << "Done" << endl;
}
