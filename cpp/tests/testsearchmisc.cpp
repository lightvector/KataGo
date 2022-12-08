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

void Tests::runNNOnTinyBoard(const string& modelFile, bool inputsNHWC, bool useNHWC, int symmetry, bool useFP16) {
  TestCommon::overrideForBackends(inputsNHWC, useNHWC);
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

  const bool logToStdout = true;
  const bool logToStderr = false;
  const bool logTime = false;
  Logger logger(nullptr, logToStdout, logToStderr, logTime);

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
  TestCommon::overrideForBackends(inputsNHWC, useNHWC);
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

  const bool logToStdout = true;
  const bool logToStderr = false;
  const bool logTime = false;
  Logger logger(nullptr, logToStdout, logToStderr, logTime);

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
  TestCommon::overrideForBackends(inputsNHWC, useNHWC);
  NeuralNet::globalInitialize();

  string sgfStr = "(;SZ[19]FF[3]PW[Go Seigen]WR[9d]PB[Takagawa Shukaku]BR[8d]DT[1957-09-26]KM[0]RE[W+R];B[qd];W[dc];B[pp];W[cp];B[eq];W[oc];B[ce];W[dh];B[fe];W[gc];B[do];W[co];B[dn];W[cm];B[jq];W[qn];B[pn];W[pm];B[on];W[qq];B[qo];W[or];B[mr];W[mq];B[nr];W[oq];B[lq];W[qm];B[rp];W[rq];B[qg];W[mp];B[lp];W[mo];B[om];W[pk];B[kn];W[mm];B[ok];W[pj];B[mk];W[op];B[dm];W[cl];B[dl];W[dk];B[ek];W[ll];B[cn];W[bn];B[bo];W[bm];B[cq];W[bp];B[oj];W[ph];B[qh];W[oi];B[qi];W[pi];B[mi];W[of];B[ki];W[qc];B[rc];W[qe];B[re];W[pd];B[rd];W[de];B[df];W[cd];B[ee];W[dd];B[fg];W[hd];B[jl];W[dj];B[bf];W[fj];B[hg];W[dp];B[ep];W[jk];B[il];W[fk];B[ie];W[he];B[hf];W[gm];B[ke];W[fo];B[eo];W[in];B[ho];W[hn];B[fn];W[gn];B[go];W[io];B[ip];W[jp];B[hq];W[qf];B[rf];W[qb];B[ik];W[lr];B[id];W[kr];B[jr];W[bq];B[ib];W[hb];B[cr];W[rj];B[rb];W[kk];B[ij];W[ic];B[jc];W[jb];B[hc];W[iq];B[ir];W[ic];B[kq];W[kc];B[hc];W[nj];B[nk];W[ic];B[oe];W[jd];B[pe];W[pf];B[od];W[pc];B[md];W[mc];B[me];W[ld];B[ng];W[ri];B[rh];W[pg];B[fl];W[je];B[kg];W[be];B[cf];W[bh];B[bd];W[bc];B[ae];W[kl];B[rn];W[mj];B[lj];W[ni];B[lk];W[mh];B[li];W[mg];B[mf];W[nh];B[jf];W[qj];B[sh];W[rm];B[km];W[if];B[ig];W[dq];B[dr];W[br];B[ci];W[gi];B[ei];W[ej];B[di];W[gl];B[bi];W[cj];B[sq];W[sr];B[so];W[sp];B[fc];W[fb];B[sq];W[lo];B[rr];W[sp];B[ec];W[eb];B[sq];W[ko];B[jn];W[sp];B[nc];W[nb];B[sq];W[nd];B[jo];W[sp];B[qr];W[pq];B[sq];W[ns];B[ks];W[sp];B[bk];W[bj];B[sq];W[ol];B[nl];W[sp];B[aj];W[ck];B[sq];W[nq];B[ls];W[sp];B[gk];W[qp];B[po];W[ro];B[gj];W[eh];B[rp];W[fi];B[sq];W[pl];B[nm];W[sp];B[ch];W[ro];B[dg];W[sn];B[ne];W[er];B[fr];W[cs];B[es];W[fh];B[bb];W[cb];B[ac];W[ba];B[cc];W[el];B[fm];W[bc])";

  CompactSgf* sgf = CompactSgf::parse(sgfStr);

  const bool logToStdout = false;
  const bool logToStderr = true;
  const bool logTime = false;
  Logger logger(nullptr, logToStdout, logToStderr, logTime);

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
    ifstream in;
    FileUtils::open(in,comparisonFile);
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
  TestCommon::overrideForBackends(inputsNHWC, useNHWC);

  const bool logToStdout = false;
  const bool logToStderr = true;
  const bool logTime = false;
  Logger logger(nullptr, logToStdout, logToStderr, logTime);

  const int nnXLen = 19;
  const int nnYLen = 19;
  int symmetry = -1;
  NNEvaluator* nnEval = startNNEval(modelFile,logger,"",nnXLen,nnYLen,symmetry,inputsNHWC,useNHWC,useFP16,false,false);
  nnEval->setDoRandomize(false);

  string sgf19x19 = "(;FF[4]GM[1]SZ[19]HA[0]KM[7]RU[koPOSITIONALscoreAREAtaxALLsui0button1]RE[W+R];B[pd];W[dp];B[pp];W[dd];B[fc];W[id];B[fq];W[nc];B[cn];W[dn];B[dm];W[en];B[em];W[co];B[bo];W[bn];B[cm];W[bp];B[fn];W[ec];B[fd];W[df];B[kd];W[ne];B[pf];W[if];B[kf];W[le];B[ke];W[gg];B[oc];W[mb];B[jc];W[ic];B[eb];W[db];B[ib];W[hb];B[jb];W[gb];B[ng];W[mg];B[mf];W[og];B[md];W[nd];B[nf];W[of];B[oe];W[od];B[pe];W[pc];B[qc];W[pb];B[oh];W[qj];B[he];W[ie];B[fb];W[ge];B[de];W[ee];B[qb];W[ob];B[ce];W[ed];B[cc];W[cd];B[bd];W[bc];B[ef];W[eg];B[bb];W[cb];B[hd];W[hc];B[cf];W[dg];B[hf];W[hg];B[ac];W[pg];)";
  string sgf19x10 = "(;FF[4]GM[1]SZ[19:10]HA[0]KM[6]RU[koPOSITIONALscoreAREAtaxNONEsui0]RE[W+2];B[dg];W[cd];B[pg];W[pd];B[ec];W[bg];B[cg];W[bh];B[nc];W[de];B[cf];W[di];B[bf];W[eh];B[eg];W[fd];B[dc];W[cc];B[gb];W[he];B[ee];W[ed];B[ci];W[dd];B[dh];W[hc];B[hh];W[jc];B[kc];W[kb];B[jd];W[lc];B[kd];W[ic];B[oe];W[ld];B[re];W[pe];B[pf];W[od];B[nd];W[ob];B[le];W[rd];B[kf];W[oi];B[ph];W[kh];B[ji];W[mh];B[ki];W[kg];B[jf];W[qf];B[rf];W[qe];B[qg];W[pi];B[qc];W[qb];B[qi];W[mf];B[me];W[nf];B[ng];W[mg];B[li];W[rh];B[rg];W[ri];B[nh];W[ne];B[of];W[ig];B[lh];W[qh];B[ni];W[hg];B[sh];W[ih];B[lg];W[fh];B[rb];W[nb];B[rc];W[qj];B[si];W[oj];B[hi];W[fg];B[rj];W[sj];B[ff];W[gf];B[rj];W[mc];B[md];W[sj];B[cb];W[bb];B[rj];W[ei];B[bi];W[sj];B[eb];W[ca];B[rj];W[be];B[af];W[sj];B[da];W[ba];B[rj];W[ai];B[cj];W[sj];B[hb];W[ib];B[rj];W[pb];B[qi];W[qd];B[sd];W[ii];B[ij];W[ra];B[id];W[gi];B[gj];W[gh];B[fj];W[hj];B[hi];W[hd];B[ce];W[bd];B[if];W[ie];B[je];W[ae];B[ef];W[hf];B[fe];W[ge];B[hj];W[df];B[ah];W[hh];B[sa];W[qa];B[sc];W[sb];B[bc];W[ac];B[sa];W[ag];B[ch];W[sb];B[lb];W[mb];B[sa];W[se];B[sf];W[sb];B[jb];W[ja];B[sa];W[pc];B[se];W[sb];B[fa];W[fc];B[sa];W[dj];B[ah];W[sb];B[fb];W[ha];B[sa];W[ag];B[bg];W[sb];B[jg];W[jh];B[sa];W[lf];B[mi];W[sb];B[jj];W[sa];B[ej];W[fi];B[oc];W[ia];B[lf];W[la];B[nj];W[bc];B[sg];W[db];B[pj];W[ea];B[qj];W[da];B[aj];W[gc];B[oh];W[ga];B[];W[])";

  constexpr int numThreads = 30;
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
  appendSgfPoses(TestCommon::getBenchmarkSGFData(13));
  appendSgfPoses(TestCommon::getBenchmarkSGFData(7));
  appendSgfPoses(sgf19x10);
  appendSgfPoses(TestCommon::getBenchmarkSGFData(8));
  appendSgfPoses(TestCommon::getBenchmarkSGFData(14));
  appendSgfPoses(sgf19x19);
  appendSgfPoses(sgf19x10);
  appendSgfPoses(TestCommon::getBenchmarkSGFData(17));

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
          ownershipResults[i] += std::fabs(ownership);
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
    testAssert(std::fabs(policyResults[i]-policyResultsSingleThreaded[i]) < 0.008);
    testAssert(std::fabs(valueResults[i]-valueResultsSingleThreaded[i]) < 0.015);
    testAssert(std::fabs(scoreResults[i]-scoreResultsSingleThreaded[i]) < 0.15);
    testAssert(std::fabs(ownershipResults[i]-ownershipResultsSingleThreaded[i]) < 0.1);
  }

  delete nnEval;
  cout << "Done" << endl;
}

