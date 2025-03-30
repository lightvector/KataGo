#include "../tests/tests.h"

#include "../core/fileutils.h"
#include "../neuralnet/nneval.h"
#include "../dataio/sgf.h"

#include "../external/nlohmann_json/json.hpp"

//------------------------
#include "../core/using.h"
//------------------------
using json = nlohmann::json;

void Tests::runCanaryTests(NNEvaluator* nnEval, int symmetry, bool print) {
  // Be lenient on smaller models or special runs
  double policyLenience = 0;
  double winrateLenience = 0;
  double leadLenience = 0;
  double scoreLenience = 0;

  const string& internalModelName = nnEval->getInternalModelName();
  if(print) {
    cout << "nnEval->getTrunkSpatialConvDepth() " << nnEval->getTrunkSpatialConvDepth() << endl;
    cout << "internalModelName " << internalModelName << endl;
  }

  if(nnEval->getTrunkSpatialConvDepth() <= 21) {
    policyLenience = 0.05;
    winrateLenience = 0.12;
    leadLenience = 2.0;
    scoreLenience = 3.0;
  }
  else if(nnEval->getTrunkSpatialConvDepth() <= 31 ||
          Global::isPrefix(internalModelName,"rect15") ||
          Global::isPrefix(internalModelName,"special")
  ) {
    policyLenience = 0.15;
    winrateLenience = 0.05;
    leadLenience = 1.0;
    scoreLenience = 1.5;
  }

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
    testAssert(buf.result->policyProbs[buf.result->getPos(Location::ofString("Pass",board),board)] <= 0.005);
    testAssert(buf.result->whiteWinProb > 0.30 - winrateLenience);
    testAssert(buf.result->whiteWinProb < 0.70 + winrateLenience);
    testAssert(buf.result->whiteNoResultProb < 0.03);
    testAssert(buf.result->whiteLead > -2.5 - leadLenience);
    testAssert(buf.result->whiteLead < 2.5 + leadLenience);
    testAssert(buf.result->whiteScoreMean > -3.5 - scoreLenience);
    testAssert(buf.result->whiteScoreMean < 3.5 + scoreLenience);

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

    testAssert(buf.result->policyProbs[buf.result->getPos(Location::ofString("P15",board),board)] >= 0.85 - policyLenience);
    testAssert(buf.result->policyProbs[buf.result->getPos(Location::ofString("Pass",board),board)] <= 0.005);
    testAssert(buf.result->whiteWinProb > 0.30 - winrateLenience);
    testAssert(buf.result->whiteWinProb < 0.70 + winrateLenience);
    testAssert(buf.result->whiteNoResultProb < 0.03);
    testAssert(buf.result->whiteLead > -2.5 - leadLenience);
    testAssert(buf.result->whiteLead < 2.5 + leadLenience);
    testAssert(buf.result->whiteScoreMean > -3.5 - scoreLenience);
    testAssert(buf.result->whiteScoreMean < 3.5 + scoreLenience);

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
    testAssert(buf.result->policyProbs[buf.result->getPos(Location::ofString("Pass",board),board)] <= 0.005);
    testAssert(buf.result->whiteWinProb > 0.30 - winrateLenience);
    testAssert(buf.result->whiteWinProb < 0.70 + winrateLenience);
    testAssert(buf.result->whiteNoResultProb < 0.03);
    testAssert(buf.result->whiteLead > -2.5 - leadLenience);
    testAssert(buf.result->whiteLead < 2.5 + leadLenience);
    testAssert(buf.result->whiteScoreMean > -3.5 - scoreLenience);
    testAssert(buf.result->whiteScoreMean < 3.5 + scoreLenience);

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

    testAssert(buf.result->whiteWinProb < 0.1 + winrateLenience);
    testAssert(buf.result->whiteLead < -9.0 + leadLenience);
    testAssert(buf.result->whiteLead > -19.0 - leadLenience);
    testAssert(buf.result->whiteScoreMean < -8.0 + scoreLenience);
    testAssert(buf.result->whiteScoreMean > -22.0 - scoreLenience);

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

    testAssert(buf.result->whiteWinProb > 0.9 - winrateLenience);
    testAssert(buf.result->whiteLead > 9.0 - leadLenience);
    testAssert(buf.result->whiteLead < 19.0 + leadLenience);
    testAssert(buf.result->whiteScoreMean > 8.0 - scoreLenience);
    testAssert(buf.result->whiteScoreMean < 22.0 + scoreLenience);

    delete sgf;
  }

  {
    string sgfStr = "(;FF[4]GM[1]CA[UTF-8]RU[Japanese]KM[6]SZ[16:11];B[md];W[nh];B[dh];W[cd];B[lh];W[li];B[ki])";
    CompactSgf* sgf = CompactSgf::parse(sgfStr);

    Board board;
    Player nextPla;
    BoardHistory hist;
    Rules initialRules = sgf->getRulesOrFail();
    int turnIdx = 7;
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

    testAssert(buf.result->policyProbs[buf.result->getPos(Location::ofString("N3",board),board)] >= 0.80 - policyLenience);
    testAssert(buf.result->whiteWinProb > 0.20 - winrateLenience);
    testAssert(buf.result->whiteWinProb < 0.80 + winrateLenience);
    testAssert(buf.result->whiteNoResultProb < 0.06);
    testAssert(buf.result->whiteLead > -2.5 - leadLenience);
    testAssert(buf.result->whiteLead < 2.5 + leadLenience);
    testAssert(buf.result->whiteScoreMean > -3.5 - scoreLenience);
    testAssert(buf.result->whiteScoreMean < 3.5 + scoreLenience);

    delete sgf;
  }
}

struct GpuErrorStats {
  std::vector<double> winrateError;
  std::vector<double> leadError;
  std::vector<double> scoreMeanError;
  std::vector<double> scoreStdevError;
  std::vector<double> topPolicyDiff;
  std::vector<double> policyKLDiv;
  std::vector<double> shorttermWinlossErrorError;
  std::vector<double> shorttermScoreErrorError;
  std::vector<double> ownershipError;
  void appendStats(const std::shared_ptr<NNOutput>& base, const std::shared_ptr<NNOutput>& other) {
    winrateError.push_back(
      std::abs(0.5*(base->whiteWinProb - base->whiteLossProb) - 0.5*(other->whiteWinProb - other->whiteLossProb))
      + std::abs(base->whiteNoResultProb - other->whiteNoResultProb)
    );
    leadError.push_back(std::abs(base->whiteLead - other->whiteLead));
    scoreMeanError.push_back(std::abs(base->whiteScoreMean - other->whiteScoreMean));
    scoreStdevError.push_back(
      std::abs(
        sqrt(std::max(0.0, (double)base->whiteScoreMeanSq - base->whiteScoreMean*base->whiteScoreMean)) -
        sqrt(std::max(0.0, (double)other->whiteScoreMeanSq - other->whiteScoreMean*other->whiteScoreMean))
      )
    );

    int topPolicyIdx = 0;
    double topPolicyProb = -1;
    for(int i = 0; i<NNPos::MAX_NN_POLICY_SIZE; i++) {
      if(base->policyProbs[i] > topPolicyProb) {
        topPolicyIdx = i;
        topPolicyProb = base->policyProbs[i];
      }
    }
    topPolicyDiff.push_back(std::abs(topPolicyProb - other->policyProbs[topPolicyIdx]));

    double klDivSum = 0;
    for(int i = 0; i<NNPos::MAX_NN_POLICY_SIZE; i++) {
      if(base->policyProbs[i] > 1e-30) {
        klDivSum += base->policyProbs[i] * (log(base->policyProbs[i]) - log(other->policyProbs[i]));
      }
    }
    policyKLDiv.push_back(klDivSum);

    //A metric indicating the "typical" error in the winloss value or the score that the net expects, relative to the
    //short-term future MCTS value.

    shorttermWinlossErrorError.push_back(std::abs(base->shorttermWinlossError - other->shorttermWinlossError));
    shorttermScoreErrorError.push_back(std::abs(base->shorttermScoreError - other->shorttermScoreError));

    testAssert(base->whiteOwnerMap != NULL);
    testAssert(other->whiteOwnerMap != NULL);
    testAssert(base->nnXLen == other->nnXLen);
    testAssert(base->nnYLen == other->nnYLen);
    for(int y = 0; y<base->nnYLen; y++) {
      for(int x = 0; x<base->nnXLen; x++) {
        int pos = NNPos::xyToPos(x,y,base->nnXLen);
        ownershipError.push_back(std::abs(base->whiteOwnerMap[pos] - other->whiteOwnerMap[pos]));
      }
    }
  }

  double getAverage(std::vector<double>& vec) {
    double sum = 0;
    for(const double& x: vec)
      sum += x;
    return sum / vec.size();
  }

  double get90Percentile(std::vector<double>& sortedVec) {
    return sortedVec[(sortedVec.size()-1) * 9 / 10];
  }

  double get99Percentile(std::vector<double>& sortedVec) {
    return sortedVec[(sortedVec.size()-1) * 99 / 100];
  }
  double getMaxPercentile(std::vector<double>& sortedVec) {
    return sortedVec[sortedVec.size()-1];
  }

  void sortErrors() {
    std::sort(winrateError.begin(),winrateError.end());
    std::sort(leadError.begin(),leadError.end());
    std::sort(scoreMeanError.begin(),scoreMeanError.end());
    std::sort(scoreStdevError.begin(),scoreStdevError.end());
    std::sort(topPolicyDiff.begin(),topPolicyDiff.end());
    std::sort(policyKLDiv.begin(),policyKLDiv.end());
    std::sort(shorttermWinlossErrorError.begin(),shorttermWinlossErrorError.end());
    std::sort(shorttermScoreErrorError.begin(),shorttermScoreErrorError.end());
    std::sort(ownershipError.begin(),ownershipError.end());
  }

  bool checkStats99(double wr, double score, double tpd, double pkld) {
    sortErrors();
    return (
      100*get99Percentile(winrateError) <= wr &&
      get99Percentile(leadError) <= score &&
      get99Percentile(scoreMeanError) <= score &&
      get99Percentile(scoreStdevError) <= score*0.6 &&
      100*get99Percentile(topPolicyDiff) <= tpd &&
      get99Percentile(policyKLDiv) <= pkld &&
      100*get99Percentile(shorttermWinlossErrorError) <= wr*1.8 &&
      get99Percentile(shorttermScoreErrorError) <= score*0.75 &&
      100*get99Percentile(ownershipError) <= wr*1.75
    );
  }

  bool checkStatsMax(double wr, double score, double tpd, double pkld) {
    sortErrors();
    return (
      100*getMaxPercentile(winrateError) <= wr &&
      getMaxPercentile(leadError) <= score &&
      getMaxPercentile(scoreMeanError) <= score &&
      getMaxPercentile(scoreStdevError) <= score*0.6 &&
      100*getMaxPercentile(topPolicyDiff) <= tpd &&
      getMaxPercentile(policyKLDiv) <= pkld &&
      100*getMaxPercentile(shorttermWinlossErrorError) <= wr*1.8 &&
      getMaxPercentile(shorttermScoreErrorError) <= score*0.75 &&
      100*getMaxPercentile(ownershipError) <= wr*4.0 // more lenient since ownership maxes over more stuff
    );
  }


  void reportStats(const string& name, Logger& logger) {
    sortErrors();
    auto rpad = [](const string& s, int n) {
      if(s.size() < n)
        return s + std::string(n - s.size(),' ');
      return s;
    };

    logger.write(
      rpad(name + " winrateError:   ", 60) +
      Global::strprintf(
        " %7.5f%%  %7.5f%%  %7.5f%%  %7.5f%%",
        100*getAverage(winrateError), 100*get90Percentile(winrateError), 100*get99Percentile(winrateError), 100*getMaxPercentile(winrateError)
      )
    );
    logger.write(
      rpad(name + " leadError:      ", 60) +
      Global::strprintf(
        " %7.5f   %7.5f   %7.5f   %7.5f",
        getAverage(leadError), get90Percentile(leadError), get99Percentile(leadError), getMaxPercentile(leadError))
    );
    logger.write(
      rpad(name + " scoreMeanError: ", 60) +
      Global::strprintf(
        " %7.5f   %7.5f   %7.5f   %7.5f",
        getAverage(scoreMeanError), get90Percentile(scoreMeanError), get99Percentile(scoreMeanError), getMaxPercentile(scoreMeanError))
    );
    logger.write(
      rpad(name + " scoreStdevError:", 60) +
      Global::strprintf(
        " %7.5f   %7.5f   %7.5f   %7.5f",
        getAverage(scoreStdevError), get90Percentile(scoreStdevError), get99Percentile(scoreStdevError), getMaxPercentile(scoreStdevError))
    );
    logger.write(
      rpad(name + " topPolicyDelta: ", 60) +
      Global::strprintf(
        " %7.5f%%  %7.5f%%  %7.5f%%  %7.5f%%",
        100*getAverage(topPolicyDiff), 100*get90Percentile(topPolicyDiff), 100*get99Percentile(topPolicyDiff), 100*getMaxPercentile(topPolicyDiff))
    );
    logger.write(
      rpad(name + " policyKLDiv:    ", 60) +
      Global::strprintf(
        " %8.6f  %8.6f  %8.6f  %8.6f",
        getAverage(policyKLDiv), get90Percentile(policyKLDiv), get99Percentile(policyKLDiv), getMaxPercentile(policyKLDiv))
    );
    logger.write(
      rpad(name + " stWLErrorError:", 60) +
      Global::strprintf(
        " %7.5fc  %7.5fc  %7.5fc  %7.5fc",
        100*getAverage(shorttermWinlossErrorError), 100*get90Percentile(shorttermWinlossErrorError), 100*get99Percentile(shorttermWinlossErrorError), 100*getMaxPercentile(shorttermWinlossErrorError))
    );
    logger.write(
      rpad(name + " stScErrorError:", 60) +
      Global::strprintf(
        " %7.5f   %7.5f   %7.5f   %7.5f",
        getAverage(shorttermScoreErrorError), get90Percentile(shorttermScoreErrorError), get99Percentile(shorttermScoreErrorError), getMaxPercentile(shorttermScoreErrorError))
    );
    logger.write(
      rpad(name + " ownershipError:", 60) +
      Global::strprintf(
        " %7.5fc  %7.5fc  %7.5fc  %7.5fc",
        100*getAverage(ownershipError), 100*get90Percentile(ownershipError), 100*get99Percentile(ownershipError), 100*getMaxPercentile(ownershipError))
    );
  }
};

static std::string nnOutputToJson(const std::shared_ptr<NNOutput>& nnOutput) {
  json ret;
  ret["nnHash"] = nnOutput->nnHash.toString();
  ret["whiteWinProb"] = nnOutput->whiteWinProb;
  ret["whiteLossProb"] = nnOutput->whiteLossProb;
  ret["whiteNoResultProb"] = nnOutput->whiteNoResultProb;
  ret["whiteScoreMean"] = nnOutput->whiteScoreMean;
  ret["whiteScoreMeanSq"] = nnOutput->whiteScoreMeanSq;
  ret["whiteLead"] = nnOutput->whiteLead;
  ret["varTimeLeft"] = nnOutput->varTimeLeft;
  ret["shorttermWinlossError"] = nnOutput->shorttermWinlossError;
  ret["shorttermScoreError"] = nnOutput->shorttermScoreError;
  ret["policyProbs"] = std::vector<float>(&(nnOutput->policyProbs[0]), &(nnOutput->policyProbs[0]) + NNPos::MAX_NN_POLICY_SIZE);
  ret["policyOptimismUsed"] = nnOutput->policyOptimismUsed;
  ret["nnXLen"] = nnOutput->nnXLen;
  ret["nnYLen"] = nnOutput->nnYLen;
  testAssert(nnOutput->whiteOwnerMap != NULL);
  ret["whiteOwnerMap"] = std::vector<float>(nnOutput->whiteOwnerMap, nnOutput->whiteOwnerMap + nnOutput->nnXLen*nnOutput->nnYLen);
  return std::string(ret.dump());
}

static std::shared_ptr<NNOutput> nnOutputOfJson(const std::string& s) {
  std::shared_ptr<NNOutput> nnOutput = std::make_shared<NNOutput>();
  json input = json::parse(s);
  nnOutput->nnHash = Hash128::ofString(input["nnHash"].get<string>());
  nnOutput->whiteWinProb = input["whiteWinProb"].get<float>();
  nnOutput->whiteLossProb = input["whiteLossProb"].get<float>();
  nnOutput->whiteNoResultProb = input["whiteNoResultProb"].get<float>();
  nnOutput->whiteScoreMean = input["whiteScoreMean"].get<float>();
  nnOutput->whiteScoreMeanSq = input["whiteScoreMeanSq"].get<float>();
  nnOutput->whiteLead = input["whiteLead"].get<float>();
  nnOutput->varTimeLeft = input["varTimeLeft"].get<float>();
  nnOutput->shorttermWinlossError = input["shorttermWinlossError"].get<float>();
  nnOutput->shorttermScoreError = input["shorttermScoreError"].get<float>();
  std::vector<float> policyProbs = input["policyProbs"].get<std::vector<float>>();
  testAssert(policyProbs.size() == NNPos::MAX_NN_POLICY_SIZE);
  std::copy(policyProbs.begin(),policyProbs.end(),nnOutput->policyProbs);
  nnOutput->policyOptimismUsed = input["policyOptimismUsed"].get<float>();
  nnOutput->nnXLen = input["nnXLen"].get<int>();
  nnOutput->nnYLen = input["nnYLen"].get<int>();
  testAssert(nnOutput->nnXLen >= 2 && nnOutput->nnXLen <= NNPos::MAX_BOARD_LEN);
  testAssert(nnOutput->nnYLen >= 2 && nnOutput->nnYLen <= NNPos::MAX_BOARD_LEN);
  std::vector<float> whiteOwnerMap = input["whiteOwnerMap"].get<std::vector<float>>();
  testAssert(whiteOwnerMap.size() == nnOutput->nnXLen*nnOutput->nnYLen);
  nnOutput->whiteOwnerMap = new float[nnOutput->nnXLen*nnOutput->nnYLen];
  std::copy(whiteOwnerMap.begin(),whiteOwnerMap.end(),nnOutput->whiteOwnerMap);
  nnOutput->noisedPolicyProbs = nullptr;
  return nnOutput;
}

static void saveReferenceValuesToFile(const std::vector<std::shared_ptr<NNOutput>>& referenceValues, const string& referenceFileName, Logger& logger, bool verbose) {
  testAssert(referenceFileName != "");
  std::ofstream outFile;
  FileUtils::open(outFile,referenceFileName);
  if(!outFile)
    throw StringError("Unable to save reference values to: " + referenceFileName);

  for(const std::shared_ptr<NNOutput>& nnOutput : referenceValues) {
    testAssert(nnOutput != nullptr);
    outFile << nnOutputToJson(nnOutput) << "\n";
  }
  if(verbose)
    logger.write("Saved reference values for " + Global::uint64ToString((uint64_t)referenceValues.size()) + " positions to: " + referenceFileName);

  outFile.close();
}

static void loadReferenceValuesFromFile(std::vector<std::shared_ptr<NNOutput>>& referenceValues, const string& referenceFileName, Logger& logger, bool verbose) {
  testAssert(referenceFileName != "");
  referenceValues.clear();
  std::vector<std::string> lines = FileUtils::readFileLines(referenceFileName,'\n');

  for(const string& line: lines) {
    if(Global::trim(line) != "") {
      referenceValues.push_back(nnOutputOfJson(line));
    }
  }
  if(verbose)
    logger.write("Loaded reference values for " + Global::uint64ToString((uint64_t)referenceValues.size()) + " positions from: " + referenceFileName);
}

bool Tests::runBackendErrorTest(
  NNEvaluator* nnEval,
  NNEvaluator* nnEval32,
  Logger& logger,
  const string& boardSizeDataset,
  int maxBatchSizeCap,
  bool verbose,
  bool quickTest,
  bool& fp32BatchSuccessBuf,
  const string& referenceFileName
) {

  int maxBatchSize = nnEval->getCurrentBatchSize();
  if(maxBatchSize != nnEval32->getCurrentBatchSize())
    throw StringError("Inconsistent max batch size for fp16 test");
  if(maxBatchSizeCap > 0)
    maxBatchSize = std::min(maxBatchSize,maxBatchSizeCap);
  if(maxBatchSize <= 0)
    throw StringError("Invalid max batch size for fp16 test");

  Rand filterRand("Tests::runFP16Test filter rand");
  auto loadHists = [&](const std::vector<string>& sgfStrs) {
    std::vector<BoardHistory> hists;
    for(const string& sgfStr: sgfStrs) {
      Sgf* sgf = Sgf::parse(sgfStr);
      std::set<Hash128> uniqueHashes;
      const bool hashComments = false;
      const bool hashParent = false;
      const bool flipIfPassOrWFirst = false;
      const bool allowGameOver = false;
      sgf->iterAllUniquePositions(
        uniqueHashes,
        hashComments,
        hashParent,
        flipIfPassOrWFirst,
        allowGameOver,
        NULL,
        [&](Sgf::PositionSample& sample, const BoardHistory& hist, const string& comments) {
          (void)sample;
          (void)comments;
          if(!quickTest || filterRand.nextBool(0.3))
            hists.push_back(hist);
        }
      );
      delete sgf;
    }
    return hists;
  };

  std::vector<BoardHistory> hists;
  if(boardSizeDataset == "9")
    hists = loadHists(TestCommon::getMultiGameSize9Data());
  else if(boardSizeDataset == "13")
    hists = loadHists(TestCommon::getMultiGameSize13Data());
  else if(boardSizeDataset == "19")
    hists = loadHists(TestCommon::getMultiGameSize19Data());
  else if(boardSizeDataset == "10x14")
    hists = loadHists(TestCommon::getMultiGameSize10x14Data());
  else if(boardSizeDataset == "rectangle")
    hists = loadHists(TestCommon::getMultiGameRectangleData());
  else
    throw StringError("Unknown dataset to test gpu error on: " + boardSizeDataset);

  auto evalBoard = [&](NNEvaluator* nnE, const BoardHistory& hist) {
    Board board = hist.getRecentBoard(0);
    MiscNNInputParams nnInputParams;
    nnInputParams.symmetry = (int)(BoardHistory::getSituationRulesAndKoHash(board,hist,hist.presumedNextMovePla,0.5).hash0 & 7);
    NNResultBuf buf;
    bool skipCache = true;
    bool includeOwnerMap = true;
    SGFMetadata sgfMeta = SGFMetadata::getProfile("preaz_5k");
    nnE->evaluate(board,hist,hist.presumedNextMovePla,&sgfMeta,nnInputParams,buf,skipCache,includeOwnerMap);
    return buf.result;
  };

  std::vector<std::shared_ptr<NNOutput>> referenceValues;
  bool loadedReferenceValuesFromFile = false;
#ifndef USE_EIGEN_BACKEND
  if(referenceFileName != "") {
    loadReferenceValuesFromFile(referenceValues, referenceFileName, logger, verbose);
    loadedReferenceValuesFromFile = true;
  }
#endif
  (void)loadReferenceValuesFromFile;

  std::vector<std::shared_ptr<NNOutput>> fp32;
  std::vector<std::shared_ptr<NNOutput>> fp32Batched(hists.size());
  std::vector<std::shared_ptr<NNOutput>> current;
  std::vector<std::shared_ptr<NNOutput>> currentBatched(hists.size());

  if(verbose)
    logger.write("Beginning evaluations! These may take a long time on pure CPU, or on a weak GPU, but on a decent GPU shouldn't take too long.");

  if(verbose)
    logger.write("Running evaluations in fp32");
  for(const BoardHistory& hist: hists)
    fp32.push_back(evalBoard(nnEval32,hist));

  Rand rand;

  if(maxBatchSize <= 1)
    fp32Batched = fp32;
  else {
    if(verbose)
      logger.write("Running batched evaluations in fp32");
    auto runThread = [&](int threadIdx) {
      for(size_t i = threadIdx; i<hists.size(); i += maxBatchSize)
        fp32Batched[i] = evalBoard(nnEval32,hists[i]);
    };

    std::vector<uint32_t> permutation(maxBatchSize);
    rand.fillShuffledUIntRange(maxBatchSize, permutation.data());
    vector<std::thread> threads;
    for(int i = 0; i<maxBatchSize; i++)
      threads.push_back(std::thread(runThread,permutation[i]));
    for(int i = 0; i<maxBatchSize; i++)
      threads[i].join();
  }

  if(nnEval32 != nnEval) {
    if(verbose)
      logger.write("Running evaluations using current config");
    for(const BoardHistory& hist: hists)
      current.push_back(evalBoard(nnEval,hist));

    if(maxBatchSize <= 1)
      currentBatched = current;
    else {
      if(verbose)
        logger.write("Running batched evaluations using current config");
      auto runThread = [&](int threadIdx) {
        for(size_t i = threadIdx; i<hists.size(); i += maxBatchSize)
          currentBatched[i] = evalBoard(nnEval,hists[i]);
      };
      std::vector<uint32_t> permutation(maxBatchSize);
      rand.fillShuffledUIntRange(maxBatchSize, permutation.data());
      vector<std::thread> threads;
      for(int i = 0; i<maxBatchSize; i++)
        threads.push_back(std::thread(runThread,permutation[i]));
      for(int i = 0; i<maxBatchSize; i++)
        threads[i].join();
    }
  }

  if(loadedReferenceValuesFromFile) {
    if(referenceValues.size() != fp32.size())
      throw StringError(
        "Number of reference values loaded from file does not match number of positions "
        + Global::uint64ToString(referenceValues.size()) + " " + Global::uint64ToString(fp32.size()));
  }
  else {
    logger.write("Using unbatched fp32 as the reference values");
    referenceValues = fp32;
  }

  if(verbose) {
    logger.write("Computed stats on " + Global::uint64ToString((uint64_t)referenceValues.size()) + " positions");
    logger.write("Reporting the average, 90%, 99%, and max abs error between the following configurations: ");
  }

  auto computeStats = [&](const string& name, const std::vector<std::shared_ptr<NNOutput>>& candidateValues, GpuErrorStats& stats) {
    for(size_t i = 0; i<referenceValues.size(); i++)
      stats.appendStats(referenceValues[i], candidateValues[i]);
    if(verbose)
      stats.reportStats(name, logger);
  };

  fp32BatchSuccessBuf = true;
  bool success = true;

  {
    GpuErrorStats stats;
    computeStats("fp32 error vs reference", fp32, stats);
    fp32BatchSuccessBuf = fp32BatchSuccessBuf && stats.checkStats99( 0.45, 0.225, 0.45, 0.0006);
    fp32BatchSuccessBuf = fp32BatchSuccessBuf && stats.checkStatsMax(1.35, 0.900, 1.35, 0.0012);
  }

  {
    GpuErrorStats stats;
    computeStats("batched fp32 error vs reference", fp32Batched, stats);
    fp32BatchSuccessBuf = fp32BatchSuccessBuf && stats.checkStats99( 0.45, 0.225, 0.45, 0.0006);
    fp32BatchSuccessBuf = fp32BatchSuccessBuf && stats.checkStatsMax(1.35, 0.900, 1.35, 0.0012);
  }

  if(nnEval32 != nnEval) {
    {
      GpuErrorStats stats;
      computeStats("current cfg error vs reference", current, stats);
      success = success && stats.checkStats99( 2.0, 1.00, 2.50, 0.0020);
      success = success && stats.checkStatsMax(5.0, 3.00, 6.00, 0.0040);
    }
    {
      GpuErrorStats stats;
      computeStats("batched current cfg error vs reference", currentBatched, stats);
      success = success && stats.checkStats99( 2.0, 1.00, 2.50, 0.0020);
      success = success && stats.checkStatsMax(5.0, 3.00, 6.00, 0.0040);
    }
  }

#ifdef USE_EIGEN_BACKEND
  if(referenceFileName != "")
    saveReferenceValuesToFile(referenceValues, referenceFileName, logger, verbose);
#endif
  (void)saveReferenceValuesToFile;

  return success && fp32BatchSuccessBuf;

}
