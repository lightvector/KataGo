#include "../tests/tests.h"

#include <algorithm>
#include <iterator>
#include <iomanip>

#include "../dataio/sgf.h"
#include "../neuralnet/nninputs.h"
#include "../search/asyncbot.h"
#include "../program/playutils.h"

using namespace std;
using namespace TestCommon;

static string getSearchRandSeed() {
  static int seedCounter = 0;
  return string("testSearchSeed") + Global::intToString(seedCounter++);
}

struct TestSearchOptions {
  int numMovesInARow;
  bool printRootPolicy;
  bool printEndingScoreValueBonus;
  bool printPlaySelectionValues;
  bool noClearBot;
  bool noClearCache;
  bool printMore;
  bool printMoreMoreMore;
  bool printAfterBegun;
  bool ignorePosition;
  TestSearchOptions()
    :numMovesInARow(1),
     printRootPolicy(false),
     printEndingScoreValueBonus(false),
     printPlaySelectionValues(false),
     noClearBot(false),
     noClearCache(false),
     printMore(false),
     printMoreMoreMore(false),
     printAfterBegun(false),
     ignorePosition(false)
  {}
};

static void printPolicyValueOwnership(const Board& board, const NNResultBuf& buf) {
  cout << board << endl;
  cout << endl;
  buf.result->debugPrint(cout,board);
}

static void runBotOnPosition(AsyncBot* bot, Board board, Player nextPla, BoardHistory hist, TestSearchOptions opts) {

  if(!opts.ignorePosition)
    bot->setPosition(nextPla,board,hist);

  PrintTreeOptions options;
  options = options.maxDepth(1);
  if(opts.printMoreMoreMore)
    options = options.maxDepth(20);
  else if(opts.printMore)
    options = options.minVisitsPropToExpand(0.1).maxDepth(2);

  for(int i = 0; i<opts.numMovesInARow; i++) {

    Loc move;
    if(opts.printAfterBegun) {
      cout << "Just after begun" << endl;
      std::function<void()> onSearchBegun = [&]() {
        const Search* search = bot->getSearch();
        search->printTree(cout, search->rootNode, options, P_WHITE);
      };
      move = bot->genMoveSynchronous(nextPla,TimeControls(),1.0,onSearchBegun);
    }
    else {
      move = bot->genMoveSynchronous(nextPla,TimeControls());
    }
    const Search* search = bot->getSearch();

    Board::printBoard(cout, board, Board::NULL_LOC, &(hist.moveHistory));

    cout << "Root visits: " << search->getRootVisits() << "\n";
    cout << "NN rows: " << search->nnEvaluator->numRowsProcessed() << endl;
    cout << "NN batches: " << search->nnEvaluator->numBatchesProcessed() << endl;
    cout << "NN avg batch size: " << search->nnEvaluator->averageProcessedBatchSize() << endl;
    cout << "PV: ";
    search->printPV(cout, search->rootNode, 25);
    cout << "\n";
    cout << "Tree:\n";

    search->printTree(cout, search->rootNode, options, P_WHITE);

    if(opts.printRootPolicy) {
      search->printRootPolicyMap(cout);
    }
    if(opts.printEndingScoreValueBonus) {
      search->printRootOwnershipMap(cout, P_WHITE);
      search->printRootEndingScoreValueBonus(cout);
    }
    if(opts.printPlaySelectionValues) {
      cout << "Play selection values" << endl;
      double scaleMaxToAtLeast = 10.0;
      vector<Loc> locsBuf;
      vector<double> playSelectionValuesBuf;
      bool success = search->getPlaySelectionValues(locsBuf,playSelectionValuesBuf,scaleMaxToAtLeast);
      testAssert(success);
      for(int j = 0; j<locsBuf.size(); j++) {
        cout << Location::toString(locsBuf[j],board) << " " << playSelectionValuesBuf[j] << endl;
      }
    }

    if(i < opts.numMovesInARow-1) {
      bot->makeMove(move, nextPla);
      hist.makeBoardMoveAssumeLegal(board,move,nextPla,NULL);
      cout << "Just after move" << endl;
      search->printTree(cout, search->rootNode, options, P_WHITE);
      nextPla = getOpp(nextPla);
    }
  }

  const Search* search = bot->getSearch();
  if(!opts.noClearCache) {
    search->nnEvaluator->clearCache();
    search->nnEvaluator->clearStats();
  }
  if(!opts.noClearBot)
    bot->clearSearch();
}

static void runBotOnSgf(AsyncBot* bot, const string& sgfStr, const Rules& defaultRules, int turnIdx, float overrideKomi, TestSearchOptions opts) {
  CompactSgf* sgf = CompactSgf::parse(sgfStr);

  Board board;
  Player nextPla;
  BoardHistory hist;
  Rules initialRules = sgf->getRulesOrFailAllowUnspecified(defaultRules);
  sgf->setupBoardAndHistAssumeLegal(initialRules, board, nextPla, hist, turnIdx);
  hist.setKomi(overrideKomi);
  runBotOnPosition(bot,board,nextPla,hist,opts);
  delete sgf;
}

static NNEvaluator* startNNEval(
  const string& modelFile, Logger& logger, const string& seed, int nnXLen, int nnYLen,
  int defaultSymmetry, bool inputsUseNHWC, bool useNHWC, bool useFP16, bool debugSkipNeuralNet,
  bool requireExactNNLen
) {
  vector<int> gpuIdxByServerThread = {0};
  int maxBatchSize = 16;
  int nnCacheSizePowerOfTwo = 16;
  int nnMutexPoolSizePowerOfTwo = 12;
  int maxConcurrentEvals = 1024;
  //bool debugSkipNeuralNet = false;
  bool openCLReTunePerBoardSize = false;
  const string& modelName = modelFile;
  const string openCLTunerFile = "";
  const string homeDataDirOverride = "";
  int numNNServerThreadsPerModel = 1;
  bool nnRandomize = false;
  string nnRandSeed = "runSearchTestsRandSeed"+seed;

  if(defaultSymmetry == -1) {
    nnRandomize = true;
    defaultSymmetry = 0;
  }

  string expectedSha256 = "";
  NNEvaluator* nnEval = new NNEvaluator(
    modelName,
    modelFile,
    expectedSha256,
    &logger,
    maxBatchSize,
    maxConcurrentEvals,
    nnXLen,
    nnYLen,
    requireExactNNLen,
    inputsUseNHWC,
    nnCacheSizePowerOfTwo,
    nnMutexPoolSizePowerOfTwo,
    debugSkipNeuralNet,
    openCLTunerFile,
    homeDataDirOverride,
    openCLReTunePerBoardSize,
    useFP16 ? enabled_t::True : enabled_t::False,
    useNHWC ? enabled_t::True : enabled_t::False,
    numNNServerThreadsPerModel,
    gpuIdxByServerThread,
    nnRandSeed,
    nnRandomize,
    defaultSymmetry
  );

  nnEval->spawnServerThreads();

  //Hack to get more consistent ordering of log messages spawned by nnEval threads with other output.
  std::this_thread::sleep_for(std::chrono::duration<double>(0.1));
  return nnEval;
}

static void runBasicPositions(NNEvaluator* nnEval, Logger& logger)
{
  {
    SearchParams params;
    params.maxVisits = 200;
    AsyncBot* bot = new AsyncBot(params, nnEval, &logger, getSearchRandSeed());
    Rules rules = Rules::getTrompTaylorish();
    TestSearchOptions opts;

    {
      cout << "GAME 1 ==========================================================================" << endl;
      cout << "(An ordinary pro game)" << endl;
      cout << endl;

      string sgfStr = "(;SZ[19]FF[3]PW[An Seong-chun]WR[6d]PB[Chen Yaoye]BR[9d]DT[2016-07-02]KM[7.5]RU[Chinese]RE[B+R];B[qd];W[dc];B[pq];W[dp];B[nc];W[po];B[qo];W[qn];B[qp];W[pm];B[nq];W[qi];B[qg];W[oi];B[cn];W[ck];B[fp];W[co];B[dn];W[eo];B[cq];W[dq];B[bo];W[cp];B[bp];W[bq];B[fn];W[bm];B[bn];W[fo];B[go];W[cr];B[en];W[gn];B[ho];W[gm];B[er];W[dr];B[ek];W[di];B[in];W[gk];B[cl];W[dk];B[ej];W[dl];B[el];W[gi];B[fi];W[ch];B[gh];W[hi];B[hh];W[ii];B[eh];W[df];B[ih];W[ji];B[kg];W[fg];B[ff];W[gf];B[eg];W[ef];B[fe];W[ge];B[fd];W[gg];B[fh];W[gd];B[cg];W[dg];B[dh];W[bg];B[bh];W[cf];B[ci];W[qc];B[pc];W[mp];B[on];W[mn];B[om];W[iq];B[pn];W[ol];B[qm];W[pl];B[rn];W[gq];B[kn];W[jo];B[ko];W[jp];B[jn];W[li];B[mo];W[pb];B[rc];W[oc];B[qb];W[od];B[cg];W[pd];B[dd];W[fc];B[ec];W[eb];B[ed];W[cd];B[fb];W[gc];B[db];W[cc];B[ea];W[gb];B[cb];W[bb];B[be];W[ce];B[bf];W[bd];B[ag];W[ca];B[jc];W[qe];B[ep];W[do];B[gp];W[fr];B[qc];W[nb];B[ib];W[je];B[re];W[kd];B[ba];W[aa];B[lc];W[ha];B[ld];W[le];B[me];W[mb];B[ie];W[id];B[kc];W[if];B[lf];W[ke];B[nd];W[of];B[jh];W[qf];B[rf];W[pg];B[mh];W[mq];B[mi];W[mj];B[hl];W[kh];B[jf];W[gl];B[lo];W[np];B[nr];W[kq];B[no];W[he];B[mf];W[rg];B[kk];W[jk];B[kj];W[ki];B[kl];W[lj];B[qk];W[ml];B[pa];W[ob];B[hb];W[ga];B[op];W[mr];B[ms];W[ls];B[ns];W[lq];B[pj];W[oj];B[ng];W[qh];B[eq];W[es];B[rj];W[im];B[jj];W[ik];B[jl];W[il];B[hn];W[hm];B[nm];W[mm];B[nl];W[nk];B[sf];W[ri];B[ql];W[ok];B[qj];W[lb];B[hq];W[hr];B[hp])";


      runBotOnSgf(bot, sgfStr, rules, 20, 7.5, opts);
      runBotOnSgf(bot, sgfStr, rules, 40, 7.5, opts);
      runBotOnSgf(bot, sgfStr, rules, 61, 7.5, opts);
      runBotOnSgf(bot, sgfStr, rules, 82, 7.5, opts);
      runBotOnSgf(bot, sgfStr, rules, 103, 7.5, opts);
      cout << endl << endl;
    }

    {
      cout << "GAME 2 ==========================================================================" << endl;
      cout << "(Another ordinary pro game)" << endl;
      cout << endl;

      string sgfStr = "(;SZ[19]FF[3]PW[Go Seigen]WR[9d]PB[Takagawa Shukaku]BR[8d]DT[1957-09-26]KM[0]RE[W+R];B[qd];W[dc];B[pp];W[cp];B[eq];W[oc];B[ce];W[dh];B[fe];W[gc];B[do];W[co];B[dn];W[cm];B[jq];W[qn];B[pn];W[pm];B[on];W[qq];B[qo];W[or];B[mr];W[mq];B[nr];W[oq];B[lq];W[qm];B[rp];W[rq];B[qg];W[mp];B[lp];W[mo];B[om];W[pk];B[kn];W[mm];B[ok];W[pj];B[mk];W[op];B[dm];W[cl];B[dl];W[dk];B[ek];W[ll];B[cn];W[bn];B[bo];W[bm];B[cq];W[bp];B[oj];W[ph];B[qh];W[oi];B[qi];W[pi];B[mi];W[of];B[ki];W[qc];B[rc];W[qe];B[re];W[pd];B[rd];W[de];B[df];W[cd];B[ee];W[dd];B[fg];W[hd];B[jl];W[dj];B[bf];W[fj];B[hg];W[dp];B[ep];W[jk];B[il];W[fk];B[ie];W[he];B[hf];W[gm];B[ke];W[fo];B[eo];W[in];B[ho];W[hn];B[fn];W[gn];B[go];W[io];B[ip];W[jp];B[hq];W[qf];B[rf];W[qb];B[ik];W[lr];B[id];W[kr];B[jr];W[bq];B[ib];W[hb];B[cr];W[rj];B[rb];W[kk];B[ij];W[ic];B[jc];W[jb];B[hc];W[iq];B[ir];W[ic];B[kq];W[kc];B[hc];W[nj];B[nk];W[ic];B[oe];W[jd];B[pe];W[pf];B[od];W[pc];B[md];W[mc];B[me];W[ld];B[ng];W[ri];B[rh];W[pg];B[fl];W[je];B[kg];W[be];B[cf];W[bh];B[bd];W[bc];B[ae];W[kl];B[rn];W[mj];B[lj];W[ni];B[lk];W[mh];B[li];W[mg];B[mf];W[nh];B[jf];W[qj];B[sh];W[rm];B[km];W[if];B[ig];W[dq];B[dr];W[br];B[ci];W[gi];B[ei];W[ej];B[di];W[gl];B[bi];W[cj];B[sq];W[sr];B[so];W[sp];B[fc];W[fb];B[sq];W[lo];B[rr];W[sp];B[ec];W[eb];B[sq];W[ko];B[jn];W[sp];B[nc];W[nb];B[sq];W[nd];B[jo];W[sp];B[qr];W[pq];B[sq];W[ns];B[ks];W[sp];B[bk];W[bj];B[sq];W[ol];B[nl];W[sp];B[aj];W[ck];B[sq];W[nq];B[ls];W[sp];B[gk];W[qp];B[po];W[ro];B[gj];W[eh];B[rp];W[fi];B[sq];W[pl];B[nm];W[sp];B[ch];W[ro];B[dg];W[sn];B[ne];W[er];B[fr];W[cs];B[es];W[fh];B[bb];W[cb];B[ac];W[ba];B[cc];W[el];B[fm];W[bc])";

      runBotOnSgf(bot, sgfStr, rules, 23, 0, opts);
      runBotOnSgf(bot, sgfStr, rules, 38, 0, opts);
      runBotOnSgf(bot, sgfStr, rules, 65, 0, opts);
      runBotOnSgf(bot, sgfStr, rules, 80, 0, opts);
      runBotOnSgf(bot, sgfStr, rules, 115, 0, opts);
      cout << endl << endl;
    }

    {
      cout << "GAME 3 ==========================================================================" << endl;
      cout << "Extremely close botvbot game" << endl;
      cout << endl;

      string sgfStr = "(;FF[4]GM[1]SZ[19]PB[v49-140-400v-fp16]PW[v49-140-400v-fp16-fpu25]HA[0]KM[7.5]RU[koPOSITIONALscoreAREAsui1]RE[W+0.5];B[qd];W[dp];B[cq];W[dq];B[cp];W[co];B[bo];W[bn];B[cn];W[do];B[bm];W[bp];B[an];W[bq];B[cd];W[qp];B[oq];W[pn];B[nd];W[ec];B[df];W[hc];B[jc];W[cb];B[lq];W[ch];B[cj];W[eh];B[gd];W[gc];B[fd];W[hd];B[gf];W[cl];B[dn];W[el];B[eo];W[fp];B[ej];W[bl];B[bk];W[al];B[cr];W[br];B[fi];W[gl];B[gn];W[gp];B[dk];W[dl];B[fm];W[fl];B[ho];W[iq];B[ip];W[jq];B[jp];W[hp];B[in];W[fh];B[gh];W[gg];B[gi];W[hg];B[fg];W[hf];B[eg];W[il];B[ii];W[kl];B[lo];W[jj];B[ql];W[pq];B[op];W[rm];B[ji];W[ki];B[kh];W[li];B[ij];W[gm];B[dc];W[eb];B[fn];W[jk];B[lk];W[ln];B[ll];W[km];B[mn];W[ko];B[kp];W[mo];B[lp];W[mm];B[nn];W[lm];B[on];W[nk];B[qn];W[qm];B[po];W[rn];B[pm];W[ed];B[cf];W[ni];B[rq];W[rp];B[pr];W[qq];B[qr];W[rr];B[rs];W[sr];B[lc];W[rd];B[rc];W[re];B[pd];W[rb];B[qc];W[qg];B[oj];W[ok];B[pk];W[sc];B[qb];W[bc];B[qh];W[ph];B[qi];W[og];B[kr];W[ff];B[ef];W[fe];B[bd];W[rg];B[oi];W[nh];B[pf];W[pg];B[is];W[hr];B[hs];W[gs];B[gr];W[js];B[fs];W[ir];B[ep];W[eq];B[fq];W[cs];B[er];W[dr];B[fo];W[gq];B[go];W[fr];B[ie];W[he];B[fq];W[hq];B[ib];W[bj];B[bi];W[aj];B[ai];W[ak];B[ci];W[rl];B[ee];W[ge];B[rk];W[ol];B[pl];W[mf];B[nf];W[of];B[ne];W[mg];B[qf];W[rf];B[hb];W[ad];B[jr];W[gs];B[hs];W[es];B[ds];W[fr];B[cc];W[bb];B[fq];W[es];B[ae];W[ac];B[ds];W[fr];B[lh];W[mh];B[fq];W[es];B[qo];W[ro];B[ds];W[fr];B[nj];W[mj];B[fq];W[es];B[bf];W[pi];B[pj];W[ri];B[qj];W[rh];B[gb];W[fb];B[ra];W[sb];B[le];W[me];B[md];W[rj];B[jf];W[sk];B[ks];W[fr];B[lf];W[if];B[id];W[ic];B[je];W[em];B[en];W[ck];B[ga];W[ek];B[dj];W[is];B[kq];W[gs];B[nm];W[nl];B[hk];W[im];B[jn];W[kn];B[gk];W[fk];B[db];W[da];B[fa];W[ea];B[jm];W[jl];B[ih];W[ig];B[jg];W[lg];B[kg];W[oe];B[od];W[dd];B[fj];W[ce];B[be];W[de];B[hh];W[jo];B[io];W[om];B[ap];W[aq];B[ao];W[qk];B[pn];W[am];B[bn];W[pp];B[rk];W[qe];B[pe];W[qk];B[sd];W[se];B[rk];W[qa];B[pa];W[qk];B[hl];W[hm];B[rk];W[jb];B[kb];W[qk];B[qs];W[rk];B[or];W[oh];B[ss];W[sq];B[ng];W[ik];B[hn];W[dm];B[cm];W[sd];B[qa];W[sa];B[kj];W[mk];B[ml];W[mi];B[ja];W[lj];B[dh];W[kk];B[ei];W[fc];B[bh];W[];B[cg];W[];B[no];W[];B[mp];W[];B[])";

      runBotOnSgf(bot, sgfStr, rules, 191, 7.5, opts);
      runBotOnSgf(bot, sgfStr, rules, 197, 7.5, opts);
      runBotOnSgf(bot, sgfStr, rules, 330, 7.5, opts);
      runBotOnSgf(bot, sgfStr, rules, 330, 7.0, opts);

      cout << endl;
      cout << "Jigo and drawUtility===================" << endl;
      cout << "(Game almost over, just a little cleanup)" << endl;
      SearchParams testParams = params;
      testParams.drawEquivalentWinsForWhite = 0.7;
      cout << "testParams.drawEquivalentWinsForWhite = 0.7" << endl;
      cout << endl;

      bot->setParams(testParams);
      cout << "Komi 7.5 (white wins by 0.5)" << endl;
      runBotOnSgf(bot, sgfStr, rules, 330, 7.5, opts);
      cout << endl;

      cout << "Komi 7.0 (draw)" << endl;
      runBotOnSgf(bot, sgfStr, rules, 330, 7.0, opts);
      bot->setParams(params);

      cout << endl;
      cout << "Consecutive searches playouts and visits===================" << endl;
      cout << "Doing three consecutive searches by visits" << endl;
      cout << endl;
      TestSearchOptions opts2 = opts;
      opts2.numMovesInARow = 3;
      opts2.printAfterBegun = true;
      runBotOnSgf(bot, sgfStr, rules, 85, 7.5, opts2);
      cout << endl;

      cout << "Doing three consecutive searches by playouts (limit 200)" << endl;
      cout << endl;
      testParams = params;
      testParams.maxPlayouts = 200;
      testParams.maxVisits = 10000;
      bot->setParams(testParams);
      runBotOnSgf(bot, sgfStr, rules, 85, 7.5, opts2);
      bot->setParams(params);
      cout << endl << endl;
    }

    {
      cout << "GAME 4 ==========================================================================" << endl;
      cout << "(A pro game)" << endl;
      cout << endl;

      string sgfStr = "(;SZ[19]FF[3]PW[Gu Li]WR[9d]PB[Ke Jie]BR[9d]DT[2015-07-19]KM[7.5]RU[Chinese]RE[B+R];B[qe];W[dd];B[op];W[dp];B[fc];W[cf];B[jd];W[pc];B[nc];W[nd];B[mc];W[pe];B[pf];W[qd];B[re];W[oe];B[rd];W[rc];B[qb];W[qc];B[qj];W[md];B[ld];W[lc];B[lb];W[kc];B[kb];W[qp];B[qq];W[rq];B[pq];W[ro];B[oc];W[pb];B[le];W[kq];B[fq];W[eq];B[fp];W[dn];B[iq];W[ko];B[io];W[mq];B[pm];W[qn];B[mo];W[oo];B[lp];W[np];B[no];W[oq];B[pp];W[po];B[or];W[lq];B[nq];W[mp];B[mr];W[mm];B[cc];W[dc];B[db];W[eb];B[cb];W[ec];B[be];W[bf];B[fb];W[fa];B[ce];W[de];B[df];W[ae];B[cd];W[fd];B[ad];W[ch];B[ef];W[hc];B[ib];W[gf];B[eh];W[cj];B[ga];W[ea];B[gb];W[hb];B[ha];W[ee];B[ff];W[fe];B[he];W[ge];B[gh];W[ie];B[cl];W[dk];B[cq];W[cp];B[er];W[dr];B[dq];W[ep];B[cr];W[fo];B[go];W[fn];B[fk];W[dl];B[ln];W[jn];B[lm];W[lo];B[ml];W[fr];B[ds];W[hq];B[gr];W[hr];B[ir];W[gp];B[fs];W[ho];B[in];W[im];B[nm];W[jl];B[ii];W[ig];B[of];W[nf];B[mf];W[ng];B[od];W[ne];B[pd];W[qg];B[rg];W[ph];B[qi];W[lh];B[jp];W[kp];B[hm];W[hn];B[qf];W[kj];B[gm];W[gn];B[ik];W[rb];B[jk];W[oj];B[il];W[jm];B[jr];W[lr];B[ms];W[ip];B[kk];W[jo];B[kg];W[ol];B[nk];W[ok];B[om];W[mj];B[sc];W[qk];B[rk];W[ql];B[sb];W[lk];B[kl];W[af];B[rl];W[gc];B[ia];W[id];B[jc];W[rm];B[ab];W[fl];B[gk];W[bq];B[mg];W[ll];B[km];W[nh];B[nl];W[sl];B[rj];W[rh];B[qh];W[pg];B[sf];W[qm];B[br];W[bp];B[di];W[bi];B[ek];W[dj];B[ej])";

      runBotOnSgf(bot, sgfStr, rules, 44, 7.5, opts);

      cout << "With noise===================" << endl;
      cout << "Adding root noise to the search" << endl;
      cout << endl;

      SearchParams testParams = params;
      testParams.rootNoiseEnabled = true;
      testParams.rootFpuReductionMax = 0.0;
      bot->setParams(testParams);
      runBotOnSgf(bot, sgfStr, rules, 44, 7.5, opts);
      bot->setParams(params);
      cout << endl << endl;

      cout << "With root temperature===================" << endl;
      cout << "Adding root policy temperature 1.5 to the search" << endl;
      cout << endl;

      SearchParams testParams2 = params;
      testParams2.rootPolicyTemperature = 1.5;
      testParams2.rootPolicyTemperatureEarly = 1.5;
      bot->setParams(testParams2);
      runBotOnSgf(bot, sgfStr, rules, 44, 7.5, opts);
      bot->setParams(params);
      cout << endl << endl;

      cout << "With noise and rootDesiredPerChildVisits===================" << endl;
      cout << "Root desired child visits factor 1" << endl;
      cout << endl;

      TestSearchOptions opts2 = opts;
      opts2.printPlaySelectionValues = true;

      SearchParams testParams3 = params;
      testParams3.rootNoiseEnabled = true;
      testParams3.maxVisits = 400;
      testParams3.rootFpuReductionMax = 0.0;
      testParams3.rootDesiredPerChildVisitsCoeff = 1.0;
      bot->setParams(testParams3);
      runBotOnSgf(bot, sgfStr, rules, 44, 7.5, opts2);
      bot->setParams(params);
      cout << endl << endl;

      cout << "With noise and rootDesiredPerChildVisits===================" << endl;
      cout << "Root desired child visits factor 9" << endl;
      cout << endl;

      SearchParams testParams4 = params;
      testParams4.rootNoiseEnabled = true;
      testParams4.maxVisits = 400;
      testParams4.rootFpuReductionMax = 0.0;
      testParams4.rootDesiredPerChildVisitsCoeff = 9.0;
      bot->setParams(testParams4);
      runBotOnSgf(bot, sgfStr, rules, 44, 7.5, opts2);
      bot->setParams(params);
      cout << endl << endl;
    }

    delete bot;
  }
}

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
}

static void runV8Tests(NNEvaluator* nnEval, NNEvaluator* nnEval19Exact, Logger& logger)
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

    TestSearchOptions opts;
    opts.printMore = true;
    nnEval->clearCache();
    nnEval->clearStats();
    opts.noClearCache = true;
    cout << "BASELINE" << endl;
    runBotOnPosition(botA,board,nextPla,hist,opts);
    cout << "TEMP 1.5" << endl;
    runBotOnPosition(botB,board,nextPla,hist,opts);
    cout << "TEMP 0.5" << endl;
    runBotOnPosition(botC,board,nextPla,hist,opts);
    cout << endl << endl;

    delete botA;
    delete botB;
    delete botC;
    delete sgf;
  }


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

      Search* search = new Search(params, nnEval, "autoSearchRandSeed3");
      Player nextPla;

      search->setPosition(startPla,board,hist); nextPla = startPla;

      Loc moveLoc = search->runWholeSearchAndGetMove(nextPla,logger); printSearchResults(search);

      cout << "Search next player - should clear tree and flip PDA" << endl;
      search->makeMove(moveLoc,nextPla); nextPla = getOpp(nextPla);
      moveLoc = search->runWholeSearchAndGetMove(nextPla,logger); printSearchResults(search);

      cout << "Search next player - should clear tree and flip PDA" << endl;
      search->makeMove(moveLoc,nextPla); nextPla = getOpp(nextPla);
      search->runWholeSearch(nextPla,logger); printSearchResults(search);

      delete search;
      nnEval->clearCache(); nnEval->clearStats();
    }

    {
      SearchParams params = baseParams;
      params.playoutDoublingAdvantage = 1.5;
      params.playoutDoublingAdvantagePla = P_BLACK;
      cout << "Basic search with PDA 1.5, force black" << endl;

      Search* search = new Search(params, nnEval, "autoSearchRandSeed3");
      Player nextPla;

      search->setPosition(startPla,board,hist); nextPla = startPla;

      Loc moveLoc = search->runWholeSearchAndGetMove(nextPla,logger); printSearchResults(search);

      cout << "Search next player - should preserve tree" << endl;
      search->makeMove(moveLoc,nextPla); nextPla = getOpp(nextPla);
      moveLoc = search->runWholeSearchAndGetMove(nextPla,logger); printSearchResults(search);

      cout << "Search next player - should preserve tree" << endl;
      search->makeMove(moveLoc,nextPla); nextPla = getOpp(nextPla);
      moveLoc = search->runWholeSearchAndGetMove(nextPla,logger); printSearchResults(search);

      cout << "Search next player PONDERING - should preserve tree" << endl;
      search->makeMove(moveLoc,nextPla); nextPla = getOpp(nextPla);
      bool pondering = true;
      moveLoc = search->runWholeSearchAndGetMove(nextPla,logger,pondering); printSearchResults(search);

      cout << "Search next player - should preserve tree" << endl;
      search->makeMove(moveLoc,nextPla); nextPla = getOpp(nextPla);
      search->runWholeSearch(nextPla,logger); printSearchResults(search);

      delete search;
      nnEval->clearCache(); nnEval->clearStats();
    }

    {
      SearchParams params = baseParams;
      params.playoutDoublingAdvantage = 1.5;
      params.playoutDoublingAdvantagePla = P_WHITE;
      cout << "Basic search with PDA 1.5, force white" << endl;

      Search* search = new Search(params, nnEval, "autoSearchRandSeed3");
      Player nextPla;

      search->setPosition(startPla,board,hist); nextPla = startPla;

      Loc moveLoc = search->runWholeSearchAndGetMove(nextPla,logger); printSearchResults(search);

      cout << "Search next player - should preserve tree" << endl;
      search->makeMove(moveLoc,nextPla); nextPla = getOpp(nextPla);
      moveLoc = search->runWholeSearchAndGetMove(nextPla,logger); printSearchResults(search);

      cout << "Search next player - should preserve tree" << endl;
      search->makeMove(moveLoc,nextPla); nextPla = getOpp(nextPla);
      moveLoc = search->runWholeSearchAndGetMove(nextPla,logger); printSearchResults(search);

      cout << "Search next player PONDERING - should preserve tree" << endl;
      search->makeMove(moveLoc,nextPla); nextPla = getOpp(nextPla);
      bool pondering = true;
      moveLoc = search->runWholeSearchAndGetMove(nextPla,logger,pondering); printSearchResults(search);

      cout << "Search next player PONDERING - an extra time, should preserve tree" << endl;
      search->makeMove(moveLoc,nextPla); nextPla = getOpp(nextPla);
      pondering = true;
      moveLoc = search->runWholeSearchAndGetMove(nextPla,logger,pondering); printSearchResults(search);

      cout << "Search next player - should preserve tree" << endl;
      search->makeMove(moveLoc,nextPla); nextPla = getOpp(nextPla);
      search->runWholeSearch(nextPla,logger); printSearchResults(search);

      delete search;
      nnEval->clearCache(); nnEval->clearStats();
    }

    {
      SearchParams params = baseParams;
      params.playoutDoublingAdvantage = 1.5;
      cout << "Basic search with PDA 1.5, no player" << endl;

      Search* search = new Search(params, nnEval, "autoSearchRandSeed3");
      Player nextPla;

      search->setPosition(startPla,board,hist); nextPla = startPla;

      Loc moveLoc = search->runWholeSearchAndGetMove(nextPla,logger); printSearchResults(search);

      cout << "Search next player PONDERING - should keep prior tree and PDA" << endl;
      search->makeMove(moveLoc,nextPla); nextPla = getOpp(nextPla);
      bool pondering = true;
      moveLoc = search->runWholeSearchAndGetMove(nextPla,logger,pondering); printSearchResults(search);

      cout << "Search next player - should keep tree from ponder" << endl;
      search->makeMove(moveLoc,nextPla); nextPla = getOpp(nextPla);
      search->runWholeSearch(nextPla,logger); printSearchResults(search);

      delete search;
      nnEval->clearCache(); nnEval->clearStats();
    }

    {
      SearchParams params = baseParams;
      params.playoutDoublingAdvantage = 1.5;
      cout << "Basic search with PDA 1.5, no player" << endl;

      Search* search = new Search(params, nnEval, "autoSearchRandSeed3");
      Player nextPla;

      search->setPosition(startPla,board,hist); nextPla = startPla;

      Loc moveLoc = search->runWholeSearchAndGetMove(nextPla,logger); printSearchResults(search);

      cout << "Search next player PONDERING - should keep prior tree and PDA" << endl;
      search->makeMove(moveLoc,nextPla); nextPla = getOpp(nextPla);
      bool pondering = true;
      moveLoc = search->runWholeSearchAndGetMove(nextPla,logger,pondering); printSearchResults(search);

      cout << "Search next player PONDERING an extra time - should keep prior tree and PDA" << endl;
      search->makeMove(moveLoc,nextPla); nextPla = getOpp(nextPla);
      pondering = true;
      moveLoc = search->runWholeSearchAndGetMove(nextPla,logger,pondering); printSearchResults(search);

      cout << "Search next player - now should lose the tree and PDA, because the player it is for is different" << endl;
      search->makeMove(moveLoc,nextPla); nextPla = getOpp(nextPla);
      search->runWholeSearch(nextPla,logger); printSearchResults(search);

      delete search;
      nnEval->clearCache(); nnEval->clearStats();
    }

    {
      SearchParams params = baseParams;
      params.playoutDoublingAdvantage = 1.5;
      cout << "Basic search with PDA 1.5, no player" << endl;

      Search* search = new Search(params, nnEval, "autoSearchRandSeed3");
      Player nextPla;

      search->setPosition(startPla,board,hist); nextPla = startPla;

      Loc moveLoc = search->runWholeSearchAndGetMove(nextPla,logger); printSearchResults(search);

      cout << "Search next player PONDERING - should keep prior tree and PDA" << endl;
      search->makeMove(moveLoc,nextPla); nextPla = getOpp(nextPla);
      bool pondering = true;
      moveLoc = search->runWholeSearchAndGetMove(nextPla,logger,pondering); printSearchResults(search);

      cout << "Search next player PONDERING - should still keep prior tree and PDA" << endl;
      search->makeMove(moveLoc,nextPla); nextPla = getOpp(nextPla);
      pondering = true;
      search->runWholeSearch(nextPla,logger,pondering); printSearchResults(search);

      cout << "Without making a move - convert ponder to regular search, should still keep tree and PDA" << endl;
      search->runWholeSearch(nextPla,logger); printSearchResults(search);

      nnEval->clearCache(); nnEval->clearStats();

      cout << "Set position to original, search PONDERING" << endl;
      search->setPosition(startPla,board,hist); nextPla = startPla;
      pondering = true;
      search->runWholeSearch(nextPla,logger,pondering); printSearchResults(search);

      cout << "Without making a move, convert to regular search, should not keep tree" << endl;
      cout << "and should not benefit from cache, since search would guess the opponent as 'our' side" << endl;
      moveLoc = search->runWholeSearchAndGetMove(nextPla,logger); printSearchResults(search);

      cout << "But should be fine thereafter. Make two moves and continue" << endl;
      search->makeMove(moveLoc,nextPla); nextPla = getOpp(nextPla);
      search->makeMove(Location::ofString("D4",board),nextPla); nextPla = getOpp(nextPla);
      search->runWholeSearch(nextPla,logger); printSearchResults(search);

      nnEval->clearCache(); nnEval->clearStats();

      cout << "Set position to original, search PONDERING" << endl;
      search->setPosition(startPla,board,hist); nextPla = startPla;
      pondering = true;
      moveLoc = search->runWholeSearchAndGetMove(nextPla,logger,pondering); printSearchResults(search);

      cout << "Play that move and real search on the next position, should keep tree because correct guess of side" << endl;
      search->makeMove(moveLoc,nextPla); nextPla = getOpp(nextPla);
      search->runWholeSearch(nextPla,logger); printSearchResults(search);

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
      Loc moveLoc2 = PlayUtils::maybeFriendlyPass(enabled_t::False, enabled_t::True, nextPla, moveLoc, bot->getSearchStopAndWait(),50, logger);
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
      Loc moveLoc2 = PlayUtils::maybeFriendlyPass(enabled_t::False, enabled_t::True, nextPla, moveLoc, bot->getSearchStopAndWait(),50, logger);
      cout << "Move loc: " << Location::toString(moveLoc,board) << endl;
      cout << "Friendly pass: " << Location::toString(moveLoc2,board) << endl;
      delete bot;
    }

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

void Tests::runSearchTests(const string& modelFile, bool inputsNHWC, bool useNHWC, int symmetry, bool useFP16) {
  TestCommon::overrideForOpenCL(inputsNHWC, useNHWC);
  cout << "Running search tests" << endl;
  NeuralNet::globalInitialize();

  Logger logger;
  logger.setLogToStdout(true);
  logger.setLogTime(false);

  NNEvaluator* nnEval = startNNEval(modelFile,logger,"",NNPos::MAX_BOARD_LEN,NNPos::MAX_BOARD_LEN,symmetry,inputsNHWC,useNHWC,useFP16,false,false);
  runBasicPositions(nnEval, logger);
  delete nnEval;

  NeuralNet::globalCleanup();
}

void Tests::runSearchTestsV3(const string& modelFile, bool inputsNHWC, bool useNHWC, int symmetry, bool useFP16) {
  TestCommon::overrideForOpenCL(inputsNHWC, useNHWC);
  cout << "Running search tests specifically for v3 or later nets" << endl;
  NeuralNet::globalInitialize();

  Logger logger;
  logger.setLogToStdout(true);
  logger.setLogTime(false);

  NNEvaluator* nnEval = startNNEval(modelFile,logger,"",NNPos::MAX_BOARD_LEN,NNPos::MAX_BOARD_LEN,symmetry,inputsNHWC,useNHWC,useFP16,false,false);
  NNEvaluator* nnEval11 = startNNEval(modelFile,logger,"",11,11,symmetry,inputsNHWC,useNHWC,useFP16,false,false);
  NNEvaluator* nnEvalPTemp = startNNEval(modelFile,logger,"",NNPos::MAX_BOARD_LEN,NNPos::MAX_BOARD_LEN,symmetry,inputsNHWC,useNHWC,useFP16,false,false);
  runOwnershipAndMisc(nnEval,nnEval11,nnEvalPTemp,logger);
  delete nnEval;
  delete nnEval11;
  delete nnEvalPTemp;

  NeuralNet::globalCleanup();
}

void Tests::runSearchTestsV8(const string& modelFile, bool inputsNHWC, bool useNHWC, bool useFP16) {
  TestCommon::overrideForOpenCL(inputsNHWC, useNHWC);
  cout << "Running search tests introduced after v8 nets" << endl;
  NeuralNet::globalInitialize();

  Logger logger;
  logger.setLogToStdout(true);
  logger.setLogTime(false);

  NNEvaluator* nnEval = startNNEval(
    modelFile,logger,"v8seed",19,19,-1,inputsNHWC,useNHWC,useFP16,false,false);
  NNEvaluator* nnEval19Exact = startNNEval(
    modelFile,logger,"v8seed",19,19,-1,inputsNHWC,useNHWC,useFP16,false,true);
  runV8Tests(nnEval,nnEval19Exact,logger);
  delete nnEval;
  delete nnEval19Exact;
  nnEval = NULL;
  nnEval19Exact = NULL;

  nnEval = startNNEval(
    modelFile,logger,"v8seed",19,19,5,inputsNHWC,useNHWC,useFP16,false,false);
  runMoreV8Tests(nnEval,logger);
  delete nnEval;

  nnEval = startNNEval(
    modelFile,logger,"v8seed",19,19,-1,inputsNHWC,useNHWC,useFP16,false,false);
  runMoreV8TestsRandomizedNNEvals(nnEval,logger);

  delete nnEval;
  NeuralNet::globalCleanup();
}



void Tests::runNNLessSearchTests() {
  cout << "Running neuralnetless search tests" << endl;
  NeuralNet::globalInitialize();

  //Placeholder, doesn't actually do anything since we have debugSkipNeuralNet = true
  string modelFile = "/dev/null";

  Logger logger;
  logger.setLogToStdout(false);
  logger.setLogTime(false);
  logger.addOStream(cout);

  {
    cout << "===================================================================" << endl;
    cout << "Basic search with debugSkipNeuralNet and chosen move randomization" << endl;
    cout << "===================================================================" << endl;

    NNEvaluator* nnEval = startNNEval(modelFile,logger,"",NNPos::MAX_BOARD_LEN,NNPos::MAX_BOARD_LEN,0,true,false,false,true,false);
    SearchParams params;
    params.maxVisits = 100;
    Search* search = new Search(params, nnEval, "autoSearchRandSeed");
    Rules rules = Rules::getTrompTaylorish();
    TestSearchOptions opts;

    Board board = Board::parseBoard(9,9,R"%%(
.........
.........
..x..o...
.........
..x...o..
...o.....
..o.x.x..
.........
.........
)%%");
    Player nextPla = P_BLACK;
    BoardHistory hist(board,nextPla,rules,0);

    search->setPosition(nextPla,board,hist);
    search->runWholeSearch(nextPla,logger);

    PrintTreeOptions options;
    options = options.maxDepth(1);
    search->printTree(cout, search->rootNode, options, P_WHITE);

    auto sampleChosenMoves = [&]() {
      std::map<Loc,int> moveLocsAndCounts;
      for(int i = 0; i<10000; i++) {
        Loc loc = search->getChosenMoveLoc();
        moveLocsAndCounts[loc] += 1;
      }
      vector<pair<Loc,int>> moveLocsAndCountsSorted;
      std::copy(moveLocsAndCounts.begin(),moveLocsAndCounts.end(),std::back_inserter(moveLocsAndCountsSorted));
      std::sort(moveLocsAndCountsSorted.begin(), moveLocsAndCountsSorted.end(), [](pair<Loc,int> a, pair<Loc,int> b) { return a.second > b.second; });

      for(int i = 0; i<moveLocsAndCountsSorted.size(); i++) {
        cout << Location::toString(moveLocsAndCountsSorted[i].first,board) << " " << moveLocsAndCountsSorted[i].second << endl;
      }
    };

    cout << "Chosen moves at temperature 0" << endl;
    sampleChosenMoves();

    {
      cout << "Chosen moves at temperature 1 but early temperature 0, when it's perfectly early" << endl;
      search->searchParams.chosenMoveTemperature = 1.0;
      search->searchParams.chosenMoveTemperatureEarly = 0.0;
      sampleChosenMoves();
    }

    {
      cout << "Chosen moves at temperature 1" << endl;
      search->searchParams.chosenMoveTemperature = 1.0;
      search->searchParams.chosenMoveTemperatureEarly = 1.0;
      sampleChosenMoves();
    }

    {
      cout << "Chosen moves at some intermediate temperature" << endl;
      //Ugly hack to artifically fill history. Breaks all sorts of invariants, but should work to
      //make the search htink there's some history to choose an intermediate temperature
      for(int i = 0; i<16; i++)
        search->rootHistory.moveHistory.push_back(Move(Board::NULL_LOC,P_BLACK));

      search->searchParams.chosenMoveTemperature = 1.0;
      search->searchParams.chosenMoveTemperatureEarly = 0.0;
      search->searchParams.chosenMoveTemperatureHalflife = 16.0 * 19.0/9.0;
      sampleChosenMoves();
    }

    delete search;
    delete nnEval;
    cout << endl;
  }

  {
    cout << "===================================================================" << endl;
    cout << "Testing preservation of search tree across moves" << endl;
    cout << "===================================================================" << endl;

    NNEvaluator* nnEval = startNNEval(modelFile,logger,"",NNPos::MAX_BOARD_LEN,NNPos::MAX_BOARD_LEN,0,true,false,false,true,false);
    SearchParams params;
    params.maxVisits = 50;
    Search* search = new Search(params, nnEval, "autoSearchRandSeed");
    Rules rules = Rules::getTrompTaylorish();
    TestSearchOptions opts;

    Board board = Board::parseBoard(7,7,R"%%(
..xx...
xxxxxxx
.xx..xx
.xxoooo
xxxo...
ooooooo
...o...
)%%");
    Player nextPla = P_BLACK;
    BoardHistory hist(board,nextPla,rules,0);

    {
      //--------------------------------------
      cout << "First perform a basic search." << endl;

      search->setPosition(nextPla,board,hist);
      search->runWholeSearch(nextPla,logger);

      //In theory nothing requires this, but it would be kind of crazy if this were false
      testAssert(search->rootNode->iterateAndCountChildren() > 1);
      int childrenCapacity;
      const SearchChildPointer* children = search->rootNode->getChildren(childrenCapacity);
      testAssert(childrenCapacity > 1);
      testAssert(children[1].getIfAllocated() != NULL);
      Loc locToDescend = children[1].getIfAllocated()->prevMoveLoc;

      PrintTreeOptions options;
      options = options.maxDepth(1);
      cout << search->rootBoard << endl;
      search->printTree(cout, search->rootNode, options, P_WHITE);
      search->printTree(cout, search->rootNode, options.onlyBranch(board,Location::toString(locToDescend,board)), P_WHITE);

      cout << endl;

      //--------------------------------------
      cout << "Next, make a move, and with no search, print the tree." << endl;

      search->makeMove(locToDescend,nextPla);
      nextPla = getOpp(nextPla);

      cout << search->rootBoard << endl;
      search->printTree(cout, search->rootNode, options, P_WHITE);
      cout << endl;

      //--------------------------------------
      cout << "Then continue the search to complete 50 visits." << endl;

      search->runWholeSearch(nextPla,logger);
      search->printTree(cout, search->rootNode, options, P_WHITE);
      cout << endl;
    }

    delete search;
    delete nnEval;

    cout << endl;
  }


  {
    cout << "===================================================================" << endl;
    cout << "Testing pruning of search tree across moves due to root restrictions" << endl;
    cout << "===================================================================" << endl;

    Board board = Board::parseBoard(7,7,R"%%(
..xx...
xx.xxxx
x.xx.xx
.xxoooo
xxxo..x
ooooooo
o..oo.x
)%%");
    Player nextPla = P_BLACK;
    Rules rules = Rules::getTrompTaylorish();
    BoardHistory hist(board,nextPla,rules,0);
    hist.makeBoardMoveAssumeLegal(board,Location::ofString("B5",board),nextPla,NULL);
    nextPla = getOpp(nextPla);
    hist.makeBoardMoveAssumeLegal(board,Location::ofString("pass",board),nextPla,NULL);
    nextPla = getOpp(nextPla);
    hist.makeBoardMoveAssumeLegal(board,Location::ofString("C6",board),nextPla,NULL);
    nextPla = getOpp(nextPla);
    hist.makeBoardMoveAssumeLegal(board,Location::ofString("pass",board),nextPla,NULL);
    nextPla = getOpp(nextPla);
    hist.makeBoardMoveAssumeLegal(board,Location::ofString("G7",board),nextPla,NULL);
    nextPla = getOpp(nextPla);
    hist.makeBoardMoveAssumeLegal(board,Location::ofString("pass",board),nextPla,NULL);
    nextPla = getOpp(nextPla);
    hist.makeBoardMoveAssumeLegal(board,Location::ofString("F3",board),nextPla,NULL);
    nextPla = getOpp(nextPla);
    hist.makeBoardMoveAssumeLegal(board,Location::ofString("pass",board),nextPla,NULL);
    nextPla = getOpp(nextPla);

    auto hasSuicideRootMoves = [](const Search* search) {
      int childrenCapacity;
      const SearchChildPointer* children = search->rootNode->getChildren(childrenCapacity);
      for(int i = 0; i<childrenCapacity; i++) {
        const SearchNode* child = children[i].getIfAllocated();
        if(child == NULL)
          break;
        if(search->rootBoard.isSuicide(child->prevMoveLoc,search->rootPla))
          return true;
      }
      return false;
    };
    auto hasPassAliveRootMoves = [](const Search* search) {
      int childrenCapacity;
      const SearchChildPointer* children = search->rootNode->getChildren(childrenCapacity);
      for(int i = 0; i<childrenCapacity; i++) {
        const SearchNode* child = children[i].getIfAllocated();
        if(child == NULL)
          break;
        if(search->rootSafeArea[child->prevMoveLoc] != C_EMPTY)
          return true;
      }
      return false;
    };


    {
      cout << "First with no pruning" << endl;
      NNEvaluator* nnEval = startNNEval(modelFile,logger,"seed1",NNPos::MAX_BOARD_LEN,NNPos::MAX_BOARD_LEN,0,true,false,false,true,false);
      SearchParams params;
      params.maxVisits = 400;
      Search* search = new Search(params, nnEval, "autoSearchRandSeed3");
      TestSearchOptions opts;

      search->setPosition(nextPla,board,hist);
      search->runWholeSearch(nextPla,logger);
      PrintTreeOptions options;
      options = options.maxDepth(1);
      cout << search->rootBoard << endl;
      search->printTree(cout, search->rootNode, options, P_WHITE);

      testAssert(hasSuicideRootMoves(search));

      delete search;
      delete nnEval;

      cout << endl;
    }

    {
      cout << "Next, with rootPruneUselessMoves" << endl;
      NNEvaluator* nnEval = startNNEval(modelFile,logger,"seed1",NNPos::MAX_BOARD_LEN,NNPos::MAX_BOARD_LEN,0,true,false,false,true,false);
      SearchParams params;
      params.maxVisits = 400;
      params.rootPruneUselessMoves = true;
      Search* search = new Search(params, nnEval, "autoSearchRandSeed3");
      TestSearchOptions opts;

      search->setPosition(nextPla,board,hist);
      search->runWholeSearch(nextPla,logger);
      PrintTreeOptions options;
      options = options.maxDepth(1);
      cout << search->rootBoard << endl;
      search->printTree(cout, search->rootNode, options, P_WHITE);

      testAssert(!hasSuicideRootMoves(search));

      delete search;
      delete nnEval;

      cout << endl;
    }

    cout << "Progress the game, having black fill space while white passes..." << endl;
    hist.makeBoardMoveAssumeLegal(board,Location::ofString("A7",board),nextPla,NULL);
    nextPla = getOpp(nextPla);
    hist.makeBoardMoveAssumeLegal(board,Location::ofString("pass",board),nextPla,NULL);
    nextPla = getOpp(nextPla);
    hist.makeBoardMoveAssumeLegal(board,Location::ofString("E7",board),nextPla,NULL);
    nextPla = getOpp(nextPla);
    hist.makeBoardMoveAssumeLegal(board,Location::ofString("pass",board),nextPla,NULL);
    nextPla = getOpp(nextPla);
    hist.makeBoardMoveAssumeLegal(board,Location::ofString("F7",board),nextPla,NULL);
    nextPla = getOpp(nextPla);

    {
      cout << "Searching on the opponent, the move before" << endl;
      NNEvaluator* nnEval = startNNEval(modelFile,logger,"seed1b",NNPos::MAX_BOARD_LEN,NNPos::MAX_BOARD_LEN,0,true,false,false,true,false);
      SearchParams params;
      params.maxVisits = 400;
      params.rootPruneUselessMoves = true;
      Search* search = new Search(params, nnEval, "autoSearchRandSeed3");
      TestSearchOptions opts;

      search->setPosition(nextPla,board,hist);
      search->runWholeSearch(nextPla,logger);
      PrintTreeOptions options;
      options = options.maxDepth(1);
      cout << search->rootBoard << endl;
      search->printTree(cout, search->rootNode, options, P_WHITE);
      search->printTree(cout, search->rootNode, options.onlyBranch(board,"pass"), P_WHITE);

      cout << endl;

      cout << "Now play forward the pass. The tree should still have useless suicides and also other moves in it" << endl;
      search->makeMove(Board::PASS_LOC,nextPla);
      testAssert(hasSuicideRootMoves(search));
      testAssert(hasPassAliveRootMoves(search));

      cout << search->rootBoard << endl;
      search->printTree(cout, search->rootNode, options, P_WHITE);

      cout << endl;

      cout << "But the moment we begin a search, it should no longer." << endl;
      search->beginSearch(false);
      testAssert(!hasSuicideRootMoves(search));
      testAssert(!hasPassAliveRootMoves(search));

      cout << search->rootBoard << endl;
      search->printTree(cout, search->rootNode, options, P_WHITE);

      cout << endl;

      cout << "Continue searching a bit more" << endl;
      search->runWholeSearch(getOpp(nextPla),logger);

      cout << search->rootBoard << endl;
      search->printTree(cout, search->rootNode, options, P_WHITE);

      delete search;
      delete nnEval;
      cout << endl;
    }

  }

  {
    cout << "===================================================================" << endl;
    cout << "Testing search tree update near terminal positions" << endl;
    cout << "===================================================================" << endl;

    Board board = Board::parseBoard(7,7,R"%%(
x.xx.xx
xxx.xxx
xxxxxxx
xxxxxxx
ooooooo
ooooooo
o..o.oo
)%%");

    Player nextPla = P_WHITE;
    Rules rules = Rules::getTrompTaylorish();
    rules.multiStoneSuicideLegal = false;
    BoardHistory hist(board,nextPla,rules,0);

    {
      cout << "First with no pruning" << endl;
      NNEvaluator* nnEval = startNNEval(modelFile,logger,"seed1",NNPos::MAX_BOARD_LEN,NNPos::MAX_BOARD_LEN,0,true,false,false,true,false);
      SearchParams params;
      params.maxVisits = 400;
      params.dynamicScoreUtilityFactor = 0.5;
      params.useLcbForSelection = true;

      Search* search = new Search(params, nnEval, "autoSearchRandSeed3");
      TestSearchOptions opts;

      search->setPosition(nextPla,board,hist);
      search->runWholeSearch(nextPla,logger);
      PrintTreeOptions options;
      options = options.maxDepth(1);
      options = options.printSqs(true);
      cout << search->rootBoard << endl;
      search->printTree(cout, search->rootNode, options, P_WHITE);

      cout << "Begin search is idempotent?" << endl;
      search->beginSearch(false);
      search->printTree(cout, search->rootNode, options, P_WHITE);
      search->makeMove(Location::ofString("B1",board),nextPla);
      search->printTree(cout, search->rootNode, options, P_WHITE);
      search->beginSearch(false);
      search->printTree(cout, search->rootNode, options, P_WHITE);

      delete search;
      delete nnEval;

      cout << endl;
    }
  }

  {
    cout << "===================================================================" << endl;
    cout << "Non-square board search" << endl;
    cout << "===================================================================" << endl;

    NNEvaluator* nnEval = startNNEval(modelFile,logger,"",7,17,0,true,false,false,true,false);
    SearchParams params;
    params.maxVisits = 100;
    Search* search = new Search(params, nnEval, "autoSearchRandSeed");
    Rules rules = Rules::getTrompTaylorish();
    TestSearchOptions opts;

    Board board = Board::parseBoard(7,17,R"%%(
.......
.......
..x.o..
.......
...o...
.......
.......
.......
.......
.......
...x...
.......
.......
..xx...
..oox..
....o..
.......
)%%");
    Player nextPla = P_BLACK;
    BoardHistory hist(board,nextPla,rules,0);

    search->setPosition(nextPla,board,hist);
    search->runWholeSearch(nextPla,logger);

    cout << search->rootBoard << endl;

    PrintTreeOptions options;
    options = options.maxDepth(1);
    search->printTree(cout, search->rootNode, options, P_WHITE);

    delete search;
    delete nnEval;
    cout << endl;
  }


  {
    cout << "===================================================================" << endl;
    cout << "Visualize dirichlet noise" << endl;
    cout << "===================================================================" << endl;

    SearchParams params;
    params.rootNoiseEnabled = true;
    Rand rand("noiseVisualize");

    auto run = [&](int xSize, int ySize) {
      Board board(xSize,ySize);
      int nnXLen = 19;
      int nnYLen = 19;
      float sum = 0.0;
      int counter = 0;

      float origPolicyProbs[NNPos::MAX_NN_POLICY_SIZE];
      float policyProbs[NNPos::MAX_NN_POLICY_SIZE];
      std::fill(policyProbs,policyProbs+NNPos::MAX_NN_POLICY_SIZE,-1.0f);
      {
        for(int y = 0; y<board.y_size; y++) {
          for(int x = 0; x<board.x_size; x++) {
            int pos = NNPos::xyToPos(x,y,nnXLen);
            policyProbs[pos] = (float)pow(0.9,counter++);
            sum += policyProbs[pos];
          }
        }
        int pos = NNPos::locToPos(Board::PASS_LOC,board.x_size,nnXLen,nnYLen);
        policyProbs[pos] = (float)pow(0.9,counter++);
        sum += policyProbs[pos];

        for(int i = 0; i<NNPos::MAX_NN_POLICY_SIZE; i++) {
          if(policyProbs[i] >= 0.0)
            policyProbs[i] /= sum;
        }
      }

      std::copy(policyProbs,policyProbs+NNPos::MAX_NN_POLICY_SIZE,origPolicyProbs);
      Search::addDirichletNoise(params, rand, NNPos::MAX_NN_POLICY_SIZE, policyProbs);

      {
        for(int y = 0; y<board.y_size; y++) {
          for(int x = 0; x<board.x_size; x++) {
            int pos = NNPos::xyToPos(x,y,nnXLen);
            cout << Global::strprintf("%+6.2f ", 100.0*(policyProbs[pos] - origPolicyProbs[pos]));
          }
          cout << endl;
        }
        int pos = NNPos::locToPos(Board::PASS_LOC,board.x_size,nnXLen,nnYLen);
        cout << Global::strprintf("%+6.2f ", 100.0*(policyProbs[pos] - origPolicyProbs[pos]));
        cout << endl;
      }
    };

    run(19,19);
    run(11,7);
  }

  {
    cout << "===================================================================" << endl;
    cout << "Search tolerates moving past game end" << endl;
    cout << "===================================================================" << endl;

    NNEvaluator* nnEval = startNNEval(modelFile,logger,"",7,7,0,true,false,false,true,false);
    SearchParams params;
    params.maxVisits = 200;
    Search* search = new Search(params, nnEval, "autoSearchRandSeed");
    Search* search2 = new Search(params, nnEval, "autoSearchRandSeed");
    Search* search3 = new Search(params, nnEval, "autoSearchRandSeed");
    Rules rules = Rules::getTrompTaylorish();
    TestSearchOptions opts;
    PrintTreeOptions options;
    options = options.maxDepth(1);

    Board board = Board::parseBoard(7,7,R"%%(
.x.xo.o
xxxoooo
xxxxoo.
x.xo.oo
xxxoooo
xxxxooo
.xxxooo
)%%");
    Player nextPla = P_WHITE;
    BoardHistory hist(board,nextPla,rules,0);

    search->setPosition(nextPla,board,hist);
    search2->setPosition(nextPla,board,hist);
    search3->setPosition(nextPla,board,hist);

    search->makeMove(Location::ofString("C7",board),nextPla);
    search2->makeMove(Location::ofString("C7",board),nextPla);
    hist.makeBoardMoveAssumeLegal(board,Location::ofString("C7",board),nextPla,NULL);
    nextPla = getOpp(nextPla);
    search3->setPosition(nextPla,board,hist);

    search->makeMove(Location::ofString("pass",board),nextPla);
    search2->makeMove(Location::ofString("pass",board),nextPla);
    hist.makeBoardMoveAssumeLegal(board,Location::ofString("pass",board),nextPla,NULL);
    nextPla = getOpp(nextPla);
    search3->setPosition(nextPla,board,hist);
    board.checkConsistency();

    search2->runWholeSearch(nextPla,logger);

    search->makeMove(Location::ofString("pass",board),nextPla);
    search2->makeMove(Location::ofString("pass",board),nextPla);
    board.checkConsistency();
    hist.makeBoardMoveAssumeLegal(board,Location::ofString("pass",board),nextPla,NULL);
    nextPla = getOpp(nextPla);
    search3->setPosition(nextPla,board,hist);

    assert(hist.isGameFinished);

    search->runWholeSearch(nextPla,logger);
    search2->runWholeSearch(nextPla,logger);
    search3->runWholeSearch(nextPla,logger);

    hist.printDebugInfo(cout,board);
    cout << "Search made move after gameover" << endl;
    search->printTree(cout, search->rootNode, options, P_WHITE);
    cout << "Search made move (carrying tree over) after gameover" << endl;
    search2->printTree(cout, search2->rootNode, options, P_WHITE);
    cout << "Position was set after gameover" << endl;
    search3->printTree(cout, search3->rootNode, options, P_WHITE);

    cout << "Recapturing ko after two passes and supposed game over (violates superko)" << endl;
    search->makeMove(Location::ofString("D7",board),nextPla);
    search2->makeMove(Location::ofString("D7",board),nextPla);
    hist.makeBoardMoveAssumeLegal(board,Location::ofString("D7",board),nextPla,NULL);
    nextPla = getOpp(nextPla);
    search3->setPosition(nextPla,board,hist);

    search->runWholeSearch(nextPla,logger);
    search2->runWholeSearch(nextPla,logger);
    search3->runWholeSearch(nextPla,logger);

    hist.printDebugInfo(cout,board);
    cout << "Search made move" << endl;
    search->printTree(cout, search->rootNode, options, P_WHITE);
    cout << "Search made move (carrying tree over)" << endl;
    search2->printTree(cout, search2->rootNode, options, P_WHITE);
    cout << "Position was set" << endl;
    search3->printTree(cout, search3->rootNode, options, P_WHITE);

    delete search;
    delete search2;
    delete search3;
    delete nnEval;
    cout << endl;
  }

  {
    cout << "===================================================================" << endl;
    cout << "Integrity of value bias, mem safety and updates" << endl;
    cout << "===================================================================" << endl;

    NNEvaluator* nnEval = startNNEval(modelFile,logger,"",7,7,0,true,false,false,true,false);
    SearchParams params;
    params.maxVisits = 500;
    params.subtreeValueBiasFactor = 0.5;
    params.chosenMoveTemperature = 0;
    Search* search = new Search(params, nnEval, "autoSearchRandSeed");
    Rules rules = Rules::getTrompTaylorish();
    Board board = Board::parseBoard(7,7,R"%%(
x.xxxx.
xxxooxx
xxxxox.
xxx.oxx
ooxoooo
o.oo.oo
.oooooo
)%%");
    Player nextPla = P_BLACK;
    BoardHistory hist(board,nextPla,rules,0);

    PrintTreeOptions options;
    options = options.maxDepth(1);

    search->setPosition(nextPla,board,hist);

    search->runWholeSearch(nextPla,logger);
    cout << search->rootBoard << endl;
    search->printTree(cout, search->rootNode, options, P_WHITE);

    cout << "Making move" << endl;
    search->makeMove(search->getChosenMoveLoc(),nextPla);
    cout << search->rootBoard << endl;
    search->printTree(cout, search->rootNode, options, P_WHITE);

    cout << "Searching again" << endl;
    search->runWholeSearch(nextPla,logger);
    cout << search->rootBoard << endl;
    search->printTree(cout, search->rootNode, options, P_WHITE);

    cout << "Making move" << endl;
    search->makeMove(search->getChosenMoveLoc(),nextPla);
    cout << search->rootBoard << endl;
    search->printTree(cout, search->rootNode, options, P_WHITE);

    cout << "Searching again" << endl;
    search->runWholeSearch(nextPla,logger);
    cout << search->rootBoard << endl;
    search->printTree(cout, search->rootNode, options, P_WHITE);

    cout << "Making move" << endl;
    search->makeMove(search->getChosenMoveLoc(),nextPla);
    cout << search->rootBoard << endl;
    search->printTree(cout, search->rootNode, options, P_WHITE);


    delete search;
    delete nnEval;
    cout << endl;
  }

  {
    cout << "===================================================================" << endl;
    cout << "Analysis json" << endl;
    cout << "===================================================================" << endl;

    NNEvaluator* nnEval = startNNEval(modelFile,logger,"",9,9,0,true,false,false,true,false);
    SearchParams params;
    params.maxVisits = 10;
    params.subtreeValueBiasFactor = 0.5;
    params.chosenMoveTemperature = 0;
    Search* search = new Search(params, nnEval, "autoSearchRandSeed");
    search->setAlwaysIncludeOwnerMap(true);
    Rules rules = Rules::getTrompTaylorish();
    Board board = Board::parseBoard(7,7,R"%%(
.......
.......
.......
.......
.......
.......
.......
)%%");
    Player nextPla = P_BLACK;
    BoardHistory hist(board,nextPla,rules,0);

    PrintTreeOptions options;
    options = options.maxDepth(1);

    search->setPosition(nextPla,board,hist);
    search->runWholeSearch(nextPla,logger);
    search->printTree(cout, search->rootNode, options, P_WHITE);
    nlohmann::json json;
    Player perspective = P_WHITE;
    int analysisPVLen = 2;
    int ownershipMinVisits = 1;
    bool preventEncore = true;
    bool includePolicy = true;
    bool includeOwnership = true;
    bool includeMovesOwnership = false;
    bool includePVVisits = true;
    bool suc = search->getAnalysisJson(
      perspective, board, hist, analysisPVLen, ownershipMinVisits, preventEncore,
      includePolicy, includeOwnership, includeMovesOwnership, includePVVisits,
      json
    );
    testAssert(suc);
    cout << json << endl;

    delete search;
    delete nnEval;
  }

  {
    cout << "===================================================================" << endl;
    cout << "Analysis json 2" << endl;
    cout << "===================================================================" << endl;

    NNEvaluator* nnEval = startNNEval(modelFile,logger,"",9,9,0,true,false,false,true,false);
    SearchParams params;
    params.maxVisits = 10;
    params.subtreeValueBiasFactor = 0.5;
    params.chosenMoveTemperature = 0;
    Search* search = new Search(params, nnEval, "autoSearchRandSeed");
    search->setAlwaysIncludeOwnerMap(false);
    Rules rules = Rules::getTrompTaylorish();
    Board board = Board::parseBoard(9,6,R"%%(
.........
ooooooooo
oooxxxooo
..xxxxx..
xxx...xxx
xxxxxxxxx
)%%");
    Player nextPla = P_BLACK;
    BoardHistory hist(board,nextPla,rules,0);

    PrintTreeOptions options;
    options = options.maxDepth(1);

    search->setPosition(nextPla,board,hist);
    search->runWholeSearch(nextPla,logger);
    search->printTree(cout, search->rootNode, options, P_WHITE);
    nlohmann::json json;
    Player perspective = P_WHITE;
    int analysisPVLen = 2;
    int ownershipMinVisits = 1;
    bool preventEncore = true;
    bool includePolicy = true;
    bool includeOwnership = false;
    bool includeMovesOwnership = false;
    bool includePVVisits = false;
    bool suc = search->getAnalysisJson(
      perspective, board, hist, analysisPVLen, ownershipMinVisits, preventEncore,
      includePolicy, includeOwnership, includeMovesOwnership, includePVVisits,
      json
    );
    testAssert(suc);
    cout << json << endl;

    delete search;
    delete nnEval;
  }

  NeuralNet::globalCleanup();
}

void Tests::runNNOnTinyBoard(const string& modelFile, bool inputsNHWC, bool useNHWC, int symmetry, bool useFP16) {
  TestCommon::overrideForOpenCL(inputsNHWC, useNHWC);
  NeuralNet::globalInitialize();

  Board board = Board::parseBoard(5,5,R"%%(
.....
...x.
..o..
.xxo.
.....
)%%");

  Player nextPla = P_WHITE;
  Rules rules = Rules::getTrompTaylorish();
  BoardHistory hist(board,nextPla,rules,0);

  Logger logger;
  logger.setLogToStdout(true);
  logger.setLogTime(false);

  NNEvaluator* nnEval = startNNEval(modelFile,logger,"",6,6,symmetry,inputsNHWC,useNHWC,useFP16,false,false);

  MiscNNInputParams nnInputParams;
  NNResultBuf buf;
  bool skipCache = true;
  bool includeOwnerMap = true;
  nnEval->evaluate(board,hist,nextPla,nnInputParams,buf,skipCache,includeOwnerMap);

  printPolicyValueOwnership(board,buf);
  cout << endl << endl;

  delete nnEval;
  NeuralNet::globalCleanup();
}

void Tests::runNNSymmetries(const string& modelFile, bool inputsNHWC, bool useNHWC, bool useFP16) {
  TestCommon::overrideForOpenCL(inputsNHWC, useNHWC);
  NeuralNet::globalInitialize();

  Board board = Board::parseBoard(9,13,R"%%(
.........
.........
..x.o....
......o..
..x......
.........
......o..
.........
..x......
....x....
...xoo...
.........
.........
)%%");

  Player nextPla = P_BLACK;
  Rules rules = Rules::getTrompTaylorish();
  BoardHistory hist(board,nextPla,rules,0);

  Logger logger;
  logger.setLogToStdout(true);
  logger.setLogTime(false);

  NNEvaluator* nnEval = startNNEval(modelFile,logger,"",13,13,0,inputsNHWC,useNHWC,useFP16,false,false);
  for(int symmetry = 0; symmetry<8; symmetry++) {
    nnEval->setDoRandomize(false);
    nnEval->setDefaultSymmetry(symmetry);
    nnEval->clearCache();
    nnEval->clearStats();

    MiscNNInputParams nnInputParams;
    NNResultBuf buf;
    bool skipCache = true;
    bool includeOwnerMap = true;
    nnEval->evaluate(board,hist,nextPla,nnInputParams,buf,skipCache,includeOwnerMap);

    printPolicyValueOwnership(board,buf);
    cout << endl << endl;

  }
  delete nnEval;
  NeuralNet::globalCleanup();
}


void Tests::runNNOnManyPoses(const string& modelFile, bool inputsNHWC, bool useNHWC, int symmetry, bool useFP16, const string& comparisonFile) {
  TestCommon::overrideForOpenCL(inputsNHWC, useNHWC);
  NeuralNet::globalInitialize();

  string sgfStr = "(;SZ[19]FF[3]PW[Go Seigen]WR[9d]PB[Takagawa Shukaku]BR[8d]DT[1957-09-26]KM[0]RE[W+R];B[qd];W[dc];B[pp];W[cp];B[eq];W[oc];B[ce];W[dh];B[fe];W[gc];B[do];W[co];B[dn];W[cm];B[jq];W[qn];B[pn];W[pm];B[on];W[qq];B[qo];W[or];B[mr];W[mq];B[nr];W[oq];B[lq];W[qm];B[rp];W[rq];B[qg];W[mp];B[lp];W[mo];B[om];W[pk];B[kn];W[mm];B[ok];W[pj];B[mk];W[op];B[dm];W[cl];B[dl];W[dk];B[ek];W[ll];B[cn];W[bn];B[bo];W[bm];B[cq];W[bp];B[oj];W[ph];B[qh];W[oi];B[qi];W[pi];B[mi];W[of];B[ki];W[qc];B[rc];W[qe];B[re];W[pd];B[rd];W[de];B[df];W[cd];B[ee];W[dd];B[fg];W[hd];B[jl];W[dj];B[bf];W[fj];B[hg];W[dp];B[ep];W[jk];B[il];W[fk];B[ie];W[he];B[hf];W[gm];B[ke];W[fo];B[eo];W[in];B[ho];W[hn];B[fn];W[gn];B[go];W[io];B[ip];W[jp];B[hq];W[qf];B[rf];W[qb];B[ik];W[lr];B[id];W[kr];B[jr];W[bq];B[ib];W[hb];B[cr];W[rj];B[rb];W[kk];B[ij];W[ic];B[jc];W[jb];B[hc];W[iq];B[ir];W[ic];B[kq];W[kc];B[hc];W[nj];B[nk];W[ic];B[oe];W[jd];B[pe];W[pf];B[od];W[pc];B[md];W[mc];B[me];W[ld];B[ng];W[ri];B[rh];W[pg];B[fl];W[je];B[kg];W[be];B[cf];W[bh];B[bd];W[bc];B[ae];W[kl];B[rn];W[mj];B[lj];W[ni];B[lk];W[mh];B[li];W[mg];B[mf];W[nh];B[jf];W[qj];B[sh];W[rm];B[km];W[if];B[ig];W[dq];B[dr];W[br];B[ci];W[gi];B[ei];W[ej];B[di];W[gl];B[bi];W[cj];B[sq];W[sr];B[so];W[sp];B[fc];W[fb];B[sq];W[lo];B[rr];W[sp];B[ec];W[eb];B[sq];W[ko];B[jn];W[sp];B[nc];W[nb];B[sq];W[nd];B[jo];W[sp];B[qr];W[pq];B[sq];W[ns];B[ks];W[sp];B[bk];W[bj];B[sq];W[ol];B[nl];W[sp];B[aj];W[ck];B[sq];W[nq];B[ls];W[sp];B[gk];W[qp];B[po];W[ro];B[gj];W[eh];B[rp];W[fi];B[sq];W[pl];B[nm];W[sp];B[ch];W[ro];B[dg];W[sn];B[ne];W[er];B[fr];W[cs];B[es];W[fh];B[bb];W[cb];B[ac];W[ba];B[cc];W[el];B[fm];W[bc])";

  CompactSgf* sgf = CompactSgf::parse(sgfStr);

  Logger logger;
  logger.setLogToStdout(false);
  logger.setLogToStderr(true);
  logger.setLogTime(false);

  int nnXLen = 19;
  int nnYLen = 19;
  NNEvaluator* nnEval = startNNEval(modelFile,logger,"",nnXLen,nnYLen,symmetry,inputsNHWC,useNHWC,useFP16,false,false);
  MiscNNInputParams nnInputParams;
  NNResultBuf buf;
  bool skipCache = true;
  bool includeOwnerMap = true;

  vector<float> winProbs;
  vector<float> scoreMeans;
  vector<float> policyProbs;

  for(int turnIdx = 0; turnIdx<sgf->moves.size(); turnIdx++) {
    Board board;
    Player nextPla;
    BoardHistory hist;
    Rules initialRules = sgf->getRulesOrFailAllowUnspecified(Rules());
    sgf->setupBoardAndHistAssumeLegal(initialRules, board, nextPla, hist, turnIdx);
    nnEval->evaluate(board,hist,nextPla,nnInputParams,buf,skipCache,includeOwnerMap);

    winProbs.push_back(buf.result->whiteWinProb);
    scoreMeans.push_back(buf.result->whiteScoreMean);
    for(int y = 0; y<board.y_size; y++) {
      for(int x = 0; x<board.x_size; x++) {
        int pos = NNPos::xyToPos(x,y,nnEval->getNNXLen());
        policyProbs.push_back(buf.result->policyProbs[pos]);
      }
    }
  }

  if(comparisonFile == "") {
    cout << std::setprecision(17);
    cout << std::fixed;
    for(int i = 0; i<winProbs.size(); i++)
      cout << winProbs[i] << endl;
    for(int i = 0; i<scoreMeans.size(); i++)
      cout << scoreMeans[i] << endl;
    for(int i = 0; i<policyProbs.size(); i++)
      cout << policyProbs[i] << endl;
  }
  else {
    ifstream in(comparisonFile);
    double d;
    double winProbSquerr = 0.0;
    for(int i = 0; i<winProbs.size(); i++)
    { in >> d; winProbSquerr += (d - winProbs[i]) * (d - winProbs[i]); }
    double scoreMeanSquerr = 0.0;
    for(int i = 0; i<scoreMeans.size(); i++)
    { in >> d; scoreMeanSquerr += (d - scoreMeans[i]) * (d - scoreMeans[i]); }
    double policyProbSquerr = 0.0;
    for(int i = 0; i<policyProbs.size(); i++)
    { in >> d; policyProbSquerr += (d - policyProbs[i]) * (d - policyProbs[i]); }
    cout << "winProbSquerr " << winProbSquerr << endl;
    cout << "scoreMeanSquerr " << scoreMeanSquerr << endl;
    cout << "policyProbSquerr " << policyProbSquerr << endl;
  }

  delete nnEval;
  delete sgf;
  NeuralNet::globalCleanup();

}

STRUCT_NAMED_TRIPLE(Board,board,BoardHistory,hist,Player,nextPla,NNBatchingTestItem);

void Tests::runNNBatchingTest(const string& modelFile, bool inputsNHWC, bool useNHWC, bool useFP16) {
  TestCommon::overrideForOpenCL(inputsNHWC, useNHWC);
  Logger logger;
  logger.setLogToStdout(false);
  logger.setLogToStderr(true);
  logger.setLogTime(false);

  const int nnXLen = 19;
  const int nnYLen = 19;
  int symmetry = -1;
  NNEvaluator* nnEval = startNNEval(modelFile,logger,"",nnXLen,nnYLen,symmetry,inputsNHWC,useNHWC,useFP16,false,false);
  nnEval->setDoRandomize(false);

  string sgf19x19 = "(;FF[4]GM[1]SZ[19]HA[0]KM[7]RU[koPOSITIONALscoreAREAtaxALLsui0button1]RE[W+R];B[pd];W[dp];B[pp];W[dd];B[fc];W[id];B[fq];W[nc];B[cn];W[dn];B[dm];W[en];B[em];W[co];B[bo];W[bn];B[cm];W[bp];B[fn];W[ec];B[fd];W[df];B[kd];W[ne];B[pf];W[if];B[kf];W[le];B[ke];W[gg];B[oc];W[mb];B[jc];W[ic];B[eb];W[db];B[ib];W[hb];B[jb];W[gb];B[ng];W[mg];B[mf];W[og];B[md];W[nd];B[nf];W[of];B[oe];W[od];B[pe];W[pc];B[qc];W[pb];B[oh];W[qj];B[he];W[ie];B[fb];W[ge];B[de];W[ee];B[qb];W[ob];B[ce];W[ed];B[cc];W[cd];B[bd];W[bc];B[ef];W[eg];B[bb];W[cb];B[hd];W[hc];B[cf];W[dg];B[hf];W[hg];B[ac];W[pg];)";
  string sgf19x10 = "(;FF[4]GM[1]SZ[19:10]HA[0]KM[6]RU[koPOSITIONALscoreAREAtaxNONEsui0]RE[W+2];B[dg];W[cd];B[pg];W[pd];B[ec];W[bg];B[cg];W[bh];B[nc];W[de];B[cf];W[di];B[bf];W[eh];B[eg];W[fd];B[dc];W[cc];B[gb];W[he];B[ee];W[ed];B[ci];W[dd];B[dh];W[hc];B[hh];W[jc];B[kc];W[kb];B[jd];W[lc];B[kd];W[ic];B[oe];W[ld];B[re];W[pe];B[pf];W[od];B[nd];W[ob];B[le];W[rd];B[kf];W[oi];B[ph];W[kh];B[ji];W[mh];B[ki];W[kg];B[jf];W[qf];B[rf];W[qe];B[qg];W[pi];B[qc];W[qb];B[qi];W[mf];B[me];W[nf];B[ng];W[mg];B[li];W[rh];B[rg];W[ri];B[nh];W[ne];B[of];W[ig];B[lh];W[qh];B[ni];W[hg];B[sh];W[ih];B[lg];W[fh];B[rb];W[nb];B[rc];W[qj];B[si];W[oj];B[hi];W[fg];B[rj];W[sj];B[ff];W[gf];B[rj];W[mc];B[md];W[sj];B[cb];W[bb];B[rj];W[ei];B[bi];W[sj];B[eb];W[ca];B[rj];W[be];B[af];W[sj];B[da];W[ba];B[rj];W[ai];B[cj];W[sj];B[hb];W[ib];B[rj];W[pb];B[qi];W[qd];B[sd];W[ii];B[ij];W[ra];B[id];W[gi];B[gj];W[gh];B[fj];W[hj];B[hi];W[hd];B[ce];W[bd];B[if];W[ie];B[je];W[ae];B[ef];W[hf];B[fe];W[ge];B[hj];W[df];B[ah];W[hh];B[sa];W[qa];B[sc];W[sb];B[bc];W[ac];B[sa];W[ag];B[ch];W[sb];B[lb];W[mb];B[sa];W[se];B[sf];W[sb];B[jb];W[ja];B[sa];W[pc];B[se];W[sb];B[fa];W[fc];B[sa];W[dj];B[ah];W[sb];B[fb];W[ha];B[sa];W[ag];B[bg];W[sb];B[jg];W[jh];B[sa];W[lf];B[mi];W[sb];B[jj];W[sa];B[ej];W[fi];B[oc];W[ia];B[lf];W[la];B[nj];W[bc];B[sg];W[db];B[pj];W[ea];B[qj];W[da];B[aj];W[gc];B[oh];W[ga];B[];W[])";

  constexpr int numThreads = 10;
  vector<NNBatchingTestItem> items;

  auto appendSgfPoses = [&](string sgfStr) {
    Rand rand("runNNBatchingTest");
    CompactSgf* sgf = CompactSgf::parse(sgfStr);
    for(int turnIdx = 0; turnIdx<sgf->moves.size(); turnIdx++) {
      Board board;
      Player nextPla;
      BoardHistory hist;
      Rules initialRules;
      initialRules.koRule = rand.nextBool(0.5) ? Rules::KO_SIMPLE : rand.nextBool(0.5) ? Rules::KO_POSITIONAL : Rules::KO_SITUATIONAL;
      initialRules.scoringRule = rand.nextBool(0.5) ? Rules::SCORING_AREA : Rules::SCORING_TERRITORY;
      initialRules.taxRule = rand.nextBool(0.5) ? Rules::TAX_NONE : rand.nextBool(0.5) ? Rules::TAX_SEKI : Rules::TAX_ALL;
      initialRules.multiStoneSuicideLegal = rand.nextBool(0.5);
      initialRules.hasButton = initialRules.scoringRule == Rules::SCORING_AREA && rand.nextBool(0.5);
      initialRules.whiteHandicapBonusRule = rand.nextBool(0.5) ? Rules::WHB_ZERO : rand.nextBool(0.5) ? Rules::WHB_N : Rules::WHB_N_MINUS_ONE;
      initialRules.komi = 7.5f + rand.nextInt(-10,10) * 0.5f;
      sgf->setupBoardAndHistAssumeLegal(initialRules, board, nextPla, hist, turnIdx);
      items.push_back(NNBatchingTestItem(board,hist,nextPla));
    }
    delete sgf;
  };
  appendSgfPoses(sgf19x19);
  appendSgfPoses(sgf19x10);

  vector<double> policyResults(items.size());
  vector<double> valueResults(items.size());
  vector<double> scoreResults(items.size());
  vector<double> ownershipResults(items.size());

  auto runEvals = [&](int threadIdx) {
    Rand rand("runNNBatchingTest" + Global::intToString(threadIdx));
    for(size_t i = threadIdx; i < items.size(); i += numThreads) {
      //Get some more thread interleaving
      if(rand.nextBool(0.2))
        std::this_thread::yield();
      if(rand.nextBool(0.2))
        std::this_thread::sleep_for(std::chrono::duration<double>(0.001));
      const NNBatchingTestItem& item = items[i];
      MiscNNInputParams nnInputParams;
      nnInputParams.drawEquivalentWinsForWhite = rand.nextDouble();
      nnInputParams.conservativePass = rand.nextBool(0.5);
      nnInputParams.playoutDoublingAdvantage = rand.nextDouble(-1.0,1.0);
      nnInputParams.symmetry = rand.nextInt(0,7);

      NNResultBuf buf;
      bool skipCache = true;
      bool includeOwnerMap = true;
      Board board = item.board;
      nnEval->evaluate(board,item.hist,item.nextPla,nnInputParams,buf,skipCache,includeOwnerMap);

      NNOutput& nnOutput = *(buf.result);
      // nnOutput.debugPrint(cout,board);
      valueResults[i] = nnOutput.whiteWinProb - nnOutput.whiteLossProb;
      scoreResults[i] = nnOutput.whiteScoreMean + nnOutput.whiteLead;

      double maxPolicy = 0.0;
      for(int y = 0; y<board.y_size; y++) {
        for(int x = 0; x<board.x_size; x++) {
          int pos = NNPos::xyToPos(x,y,nnXLen);
          double ownership = nnOutput.whiteOwnerMap[pos];
          double policy = nnOutput.policyProbs[pos];
          ownershipResults[i] += abs(ownership);
          if(policy >= 0 && policy > maxPolicy)
            maxPolicy = policy;
        }
      }
      policyResults[i] += maxPolicy;

    }
  };

  for(int threadIdx = 0; threadIdx<numThreads; threadIdx++)
    runEvals(threadIdx);
  vector<double> policyResultsSingleThreaded = policyResults;
  vector<double> valueResultsSingleThreaded = valueResults;
  vector<double> scoreResultsSingleThreaded = scoreResults;
  vector<double> ownershipResultsSingleThreaded = ownershipResults;
  std::fill(policyResults.begin(), policyResults.end(), 0.0);
  std::fill(valueResults.begin(), valueResults.end(), 0.0);
  std::fill(scoreResults.begin(), scoreResults.end(), 0.0);
  std::fill(ownershipResults.begin(), ownershipResults.end(), 0.0);

  vector<std::thread> testThreads;
  for(int threadIdx = 0; threadIdx<numThreads; threadIdx++)
    testThreads.push_back(std::thread(runEvals,threadIdx));
  for(int threadIdx = 0; threadIdx<numThreads; threadIdx++)
    testThreads[threadIdx].join();

  for(size_t i = 0; i<items.size(); i++) {
    // cout << "P " << policyResults[i]-policyResultsSingleThreaded[i] << endl;
    // cout << "V " << valueResults[i]-valueResultsSingleThreaded[i] << endl;
    // cout << "S " << scoreResults[i]-scoreResultsSingleThreaded[i] << endl;
    // cout << "O " << ownershipResults[i]-ownershipResultsSingleThreaded[i] << endl;
    testAssert(abs(policyResults[i]-policyResultsSingleThreaded[i]) < 0.008);
    testAssert(abs(valueResults[i]-valueResultsSingleThreaded[i]) < 0.015);
    testAssert(abs(scoreResults[i]-scoreResultsSingleThreaded[i]) < 0.15);
    testAssert(abs(ownershipResults[i]-ownershipResultsSingleThreaded[i]) < 0.1);
  }

  delete nnEval;
  cout << "Done" << endl;
}

