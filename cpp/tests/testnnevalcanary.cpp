#include "../tests/tests.h"

#include "../neuralnet/nneval.h"
#include "../dataio/sgf.h"

//------------------------
#include "../core/using.h"
//------------------------

void Tests::runCanaryTests(NNEvaluator* nnEval, int symmetry, bool print) {
  {
    string sgfStr = "(;GM[1]FF[4]CA[UTF-8]AP[CGoban:3]ST[2]RU[Chinese]SZ[19]KM[7]PW[White]PB[Black];B[pd];W[pp];B[dd];W[dp];B[qn];W[nq];B[cq];W[dq];B[cp];W[do];B[bn];W[cc];B[cd];W[dc];B[ec];W[eb];B[fb];W[fc];B[ed];W[gb];B[db];W[fa];B[cb];W[qo];B[pn];W[nc];B[qj];W[qc];B[qd];W[pc];B[od];W[nd];B[ne];W[me];B[mf];W[nf])";
    CompactSgf* sgf = CompactSgf::parse(sgfStr);

    Board board;
    Player nextPla;
    BoardHistory hist;
    Rules initialRules = sgf->getRulesOrFail();
    int turnIdx = 18;
    sgf->setupBoardAndHistAssumeLegal(initialRules, board, nextPla, hist, turnIdx);

    MiscNNInputParams nnInputParams;
    NNResultBuf buf;
    bool skipCache = true;
    bool includeOwnerMap = true;
    nnInputParams.symmetry = symmetry;
    nnEval->evaluate(board,hist,nextPla,nnInputParams,buf,skipCache,includeOwnerMap);

    if(print) {
      cout << board << endl;
      cout << endl;
      buf.result->debugPrint(cout,board);
    }

    testAssert(buf.result->policyProbs[buf.result->getPos(Location::ofString("E16",board),board)] >= 0.95);
    testAssert(buf.result->whiteWinProb > 0.30);
    testAssert(buf.result->whiteWinProb < 0.70);
    testAssert(buf.result->whiteLead > -2.5);
    testAssert(buf.result->whiteLead < 2.5);

    delete sgf;
  }

  {
    string sgfStr = "(;GM[1]FF[4]CA[UTF-8]AP[CGoban:3]ST[2]RU[Chinese]SZ[19]KM[7]PW[White]PB[Black];B[pd];W[pp];B[dd];W[dp];B[qn];W[nq];B[cq];W[dq];B[cp];W[do];B[bn];W[cc];B[cd];W[dc];B[ec];W[eb];B[fb];W[fc];B[ed];W[gb];B[db];W[fa];B[cb];W[qo];B[pn];W[nc];B[qj];W[qc];B[qd];W[pc];B[od];W[nd];B[ne];W[me];B[mf];W[nf])";
    CompactSgf* sgf = CompactSgf::parse(sgfStr);

    Board board;
    Player nextPla;
    BoardHistory hist;
    Rules initialRules = sgf->getRulesOrFail();
    int turnIdx = 36;
    sgf->setupBoardAndHistAssumeLegal(initialRules, board, nextPla, hist, turnIdx);

    MiscNNInputParams nnInputParams;
    NNResultBuf buf;
    bool skipCache = true;
    bool includeOwnerMap = true;
    nnInputParams.symmetry = symmetry;
    nnEval->evaluate(board,hist,nextPla,nnInputParams,buf,skipCache,includeOwnerMap);

    if(print) {
      cout << board << endl;
      cout << endl;
      buf.result->debugPrint(cout,board);
    }

    testAssert(buf.result->policyProbs[buf.result->getPos(Location::ofString("P15",board),board)] >= 0.80);
    testAssert(buf.result->whiteWinProb > 0.30);
    testAssert(buf.result->whiteWinProb < 0.70);
    testAssert(buf.result->whiteLead > -2.5);
    testAssert(buf.result->whiteLead < 2.5);

    delete sgf;
  }
  {
    string sgfStr = "(;GM[1]FF[4]CA[UTF-8]AP[CGoban:3]ST[2]RU[Chinese]SZ[19]KM[7]PW[White]PB[Black];B[qd];W[dd];B[pp];W[dp];B[cf];W[fc];B[nd];W[nq];B[cq];W[dq];B[cp];W[cn];B[co];W[do];B[bn];W[cm];B[bm];W[cl];B[qn];W[pq];B[qq];W[qr];B[oq])";
    CompactSgf* sgf = CompactSgf::parse(sgfStr);

    Board board;
    Player nextPla;
    BoardHistory hist;
    Rules initialRules = sgf->getRulesOrFail();
    int turnIdx = 23;
    sgf->setupBoardAndHistAssumeLegal(initialRules, board, nextPla, hist, turnIdx);

    MiscNNInputParams nnInputParams;
    NNResultBuf buf;
    bool skipCache = true;
    bool includeOwnerMap = true;
    nnInputParams.symmetry = symmetry;
    nnEval->evaluate(board,hist,nextPla,nnInputParams,buf,skipCache,includeOwnerMap);

    if(print) {
      cout << board << endl;
      cout << endl;
      buf.result->debugPrint(cout,board);
    }

    testAssert(buf.result->policyProbs[buf.result->getPos(Location::ofString("Q2",board),board)] >= 0.95);
    testAssert(buf.result->whiteWinProb > 0.30);
    testAssert(buf.result->whiteWinProb < 0.70);
    testAssert(buf.result->whiteLead > -2.5);
    testAssert(buf.result->whiteLead < 2.5);

    delete sgf;
  }

  {
    string sgfStr = "(;GM[1]FF[4]CA[UTF-8]AP[CGoban:3]ST[2]RU[Chinese]SZ[19]KM[7]PW[White]PB[Black];B[qd];W[dd];B[pp];W[dp];B[cf];W[fc];B[nd];W[nq];B[cq];W[dq];B[cp];W[cn];B[co];W[do];B[bn];W[cm];B[bm];W[cl];B[qn];W[pq];B[qq];W[qr];B[oq])";
    CompactSgf* sgf = CompactSgf::parse(sgfStr);

    Board board;
    Player nextPla;
    BoardHistory hist;
    Rules initialRules = sgf->getRulesOrFail();
    int turnIdx = 23;
    sgf->setupBoardAndHistAssumeLegal(initialRules, board, nextPla, hist, turnIdx);
    hist.setKomi(-7);

    MiscNNInputParams nnInputParams;
    NNResultBuf buf;
    bool skipCache = true;
    bool includeOwnerMap = true;
    nnInputParams.symmetry = symmetry;
    nnEval->evaluate(board,hist,nextPla,nnInputParams,buf,skipCache,includeOwnerMap);

    if(print) {
      cout << board << endl;
      cout << endl;
      buf.result->debugPrint(cout,board);
    }

    testAssert(buf.result->whiteWinProb < 0.1);
    testAssert(buf.result->whiteLead < -5.0);

    delete sgf;
  }

  {
    string sgfStr = "(;GM[1]FF[4]CA[UTF-8]AP[CGoban:3]ST[2]RU[Chinese]SZ[19]KM[7]PW[White]PB[Black];B[qd];W[dd];B[pp];W[dp];B[cf];W[fc];B[nd];W[nq];B[cq];W[dq];B[cp];W[cn];B[co];W[do];B[bn];W[cm];B[bm];W[cl];B[qn];W[pq];B[qq];W[qr];B[oq])";
    CompactSgf* sgf = CompactSgf::parse(sgfStr);

    Board board;
    Player nextPla;
    BoardHistory hist;
    Rules initialRules = sgf->getRulesOrFail();
    int turnIdx = 23;
    sgf->setupBoardAndHistAssumeLegal(initialRules, board, nextPla, hist, turnIdx);
    hist.setKomi(21);

    MiscNNInputParams nnInputParams;
    NNResultBuf buf;
    bool skipCache = true;
    bool includeOwnerMap = true;
    nnInputParams.symmetry = symmetry;
    nnEval->evaluate(board,hist,nextPla,nnInputParams,buf,skipCache,includeOwnerMap);

    if(print) {
      cout << board << endl;
      cout << endl;
      buf.result->debugPrint(cout,board);
    }

    testAssert(buf.result->whiteWinProb > 0.9);
    testAssert(buf.result->whiteLead > 5.0);

    delete sgf;
  }
}
