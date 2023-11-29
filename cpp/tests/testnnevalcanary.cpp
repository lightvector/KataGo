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

struct GpuErrorStats {
  std::vector<double> winrateError;
  std::vector<double> scoreError;
  std::vector<double> topPolicyDiff;
  std::vector<double> policyKLDiv;
  void appendStats(const std::shared_ptr<NNOutput>& base, const std::shared_ptr<NNOutput>& other) {
    winrateError.push_back(std::abs(0.5*(base->whiteWinProb - base->whiteLossProb) - 0.5*(other->whiteWinProb - other->whiteLossProb)));
    scoreError.push_back(std::abs(base->whiteLead - other->whiteLead));

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
  };

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

  bool checkStats99(double wr, double score, double tpd, double pkld) {
    std::sort(winrateError.begin(),winrateError.end());
    std::sort(scoreError.begin(),scoreError.end());
    std::sort(topPolicyDiff.begin(),topPolicyDiff.end());
    std::sort(policyKLDiv.begin(),policyKLDiv.end());
    return (
      100*get99Percentile(winrateError) <= wr &&
      get99Percentile(scoreError) <= score &&
      100*get99Percentile(topPolicyDiff) <= tpd &&
      get99Percentile(policyKLDiv) <= pkld
    );
  }

  bool checkStatsMax(double wr, double score, double tpd, double pkld) {
    std::sort(winrateError.begin(),winrateError.end());
    std::sort(scoreError.begin(),scoreError.end());
    std::sort(topPolicyDiff.begin(),topPolicyDiff.end());
    std::sort(policyKLDiv.begin(),policyKLDiv.end());
    return (
      100*getMaxPercentile(winrateError) <= wr &&
      getMaxPercentile(scoreError) <= score &&
      100*getMaxPercentile(topPolicyDiff) <= tpd &&
      getMaxPercentile(policyKLDiv) <= pkld
    );
  }


  void reportStats(const string& name, Logger& logger) {
    std::sort(winrateError.begin(),winrateError.end());
    std::sort(scoreError.begin(),scoreError.end());
    std::sort(topPolicyDiff.begin(),topPolicyDiff.end());
    std::sort(policyKLDiv.begin(),policyKLDiv.end());

    logger.write(
      name + " winrateError:  " +
      Global::strprintf(
        "%7.5f%% %7.5f%% %7.5f%% %7.5f%%",
        100*getAverage(winrateError), 100*get90Percentile(winrateError), 100*get99Percentile(winrateError), 100*getMaxPercentile(winrateError)
      )
    );
    logger.write(
      name + " scoreError:    " +
      Global::strprintf(
        " %7.5f  %7.5f  %7.5f  %7.5f",
        getAverage(scoreError), get90Percentile(scoreError), get99Percentile(scoreError), getMaxPercentile(scoreError))
    );
    logger.write(
      name + " topPolicyDelta: " +
      Global::strprintf(
        "%7.5f%% %7.5f%% %7.5f%% %7.5f%%",
        100*getAverage(topPolicyDiff), 100*get90Percentile(topPolicyDiff), 100*get99Percentile(topPolicyDiff), 100*getMaxPercentile(topPolicyDiff))
    );
    logger.write(
      name + " policyKLDiv:   " +
      Global::strprintf(
        "%8.6f %8.6f %8.6f %8.6f",
        getAverage(policyKLDiv), get90Percentile(policyKLDiv), get99Percentile(policyKLDiv), getMaxPercentile(policyKLDiv))
    );
  }
};

void saveBaseToFile(const std::vector<std::shared_ptr<NNOutput>>& base, const string& baseFileName, Logger& logger, bool verbose) {
  assert(baseFileName != "");
  std::ofstream outFile(baseFileName, std::ios::binary);

  if (!outFile)
    throw StringError("Unable to save base to: " + baseFileName);

  size_t size = base.size();
  outFile.write(reinterpret_cast<const char*>(&size), sizeof(size));

  for (const auto& nnOutputPtr : base) {
    if (nnOutputPtr) {
      outFile.write(reinterpret_cast<const char*>(&nnOutputPtr->nnHash), sizeof(nnOutputPtr->nnHash));
      outFile.write(reinterpret_cast<const char*>(&nnOutputPtr->whiteWinProb), sizeof(nnOutputPtr->whiteWinProb));
      outFile.write(reinterpret_cast<const char*>(&nnOutputPtr->whiteLossProb), sizeof(nnOutputPtr->whiteLossProb));
      outFile.write(reinterpret_cast<const char*>(&nnOutputPtr->whiteNoResultProb), sizeof(nnOutputPtr->whiteNoResultProb));
      outFile.write(reinterpret_cast<const char*>(&nnOutputPtr->whiteScoreMean), sizeof(nnOutputPtr->whiteScoreMean));
      outFile.write(reinterpret_cast<const char*>(&nnOutputPtr->whiteScoreMeanSq), sizeof(nnOutputPtr->whiteScoreMeanSq));
      outFile.write(reinterpret_cast<const char*>(&nnOutputPtr->whiteLead), sizeof(nnOutputPtr->whiteLead));
      outFile.write(reinterpret_cast<const char*>(&nnOutputPtr->varTimeLeft), sizeof(nnOutputPtr->varTimeLeft));
      outFile.write(reinterpret_cast<const char*>(&nnOutputPtr->shorttermWinlossError), sizeof(nnOutputPtr->shorttermWinlossError));
      outFile.write(reinterpret_cast<const char*>(&nnOutputPtr->shorttermScoreError), sizeof(nnOutputPtr->shorttermScoreError));
      outFile.write(reinterpret_cast<const char*>(nnOutputPtr->policyProbs), sizeof(float) * NNPos::MAX_NN_POLICY_SIZE);
      outFile.write(reinterpret_cast<const char*>(&nnOutputPtr->nnXLen), sizeof(nnOutputPtr->nnXLen));
      outFile.write(reinterpret_cast<const char*>(&nnOutputPtr->nnYLen), sizeof(nnOutputPtr->nnYLen));
    }
  }

  if (verbose)
    logger.write("Saved " + Global::uint64ToString((uint64_t)base.size()) + " positions to: " + baseFileName);

  outFile.close();
}

void loadBaseFromFile(std::vector<std::shared_ptr<NNOutput>>& base, const string& baseFileName, Logger& logger, bool verbose) {
  assert(baseFileName != "");
  std::ifstream inFile(baseFileName, std::ios::binary);

  if (!inFile)
    throw StringError("Unable to load: " + baseFileName);

  size_t size;
  inFile.read(reinterpret_cast<char*>(&size), sizeof(size));
  base.resize(size);

  for (size_t i = 0; i < size; ++i) {
    base[i] = std::make_shared<NNOutput>();

    inFile.read(reinterpret_cast<char*>(&base[i]->nnHash), sizeof(base[i]->nnHash));
    inFile.read(reinterpret_cast<char*>(&base[i]->whiteWinProb), sizeof(base[i]->whiteWinProb));
    inFile.read(reinterpret_cast<char*>(&base[i]->whiteLossProb), sizeof(base[i]->whiteLossProb));
    inFile.read(reinterpret_cast<char*>(&base[i]->whiteNoResultProb), sizeof(base[i]->whiteNoResultProb));
    inFile.read(reinterpret_cast<char*>(&base[i]->whiteScoreMean), sizeof(base[i]->whiteScoreMean));
    inFile.read(reinterpret_cast<char*>(&base[i]->whiteScoreMeanSq), sizeof(base[i]->whiteScoreMeanSq));
    inFile.read(reinterpret_cast<char*>(&base[i]->whiteLead), sizeof(base[i]->whiteLead));
    inFile.read(reinterpret_cast<char*>(&base[i]->varTimeLeft), sizeof(base[i]->varTimeLeft));
    inFile.read(reinterpret_cast<char*>(&base[i]->shorttermWinlossError), sizeof(base[i]->shorttermWinlossError));
    inFile.read(reinterpret_cast<char*>(&base[i]->shorttermScoreError), sizeof(base[i]->shorttermScoreError));
    inFile.read(reinterpret_cast<char*>(&base[i]->policyProbs), sizeof(float) * NNPos::MAX_NN_POLICY_SIZE);
    inFile.read(reinterpret_cast<char*>(&base[i]->nnXLen), sizeof(base[i]->nnXLen));
    inFile.read(reinterpret_cast<char*>(&base[i]->nnYLen), sizeof(base[i]->nnYLen));

    base[i]->whiteOwnerMap = nullptr;
    base[i]->noisedPolicyProbs = nullptr;
  }

  if (verbose)
    logger.write("Loaded " + Global::uint64ToString((uint64_t)base.size()) + " positions from: " + baseFileName);

  inFile.close();
}

bool Tests::runFP16Test(NNEvaluator* nnEval, NNEvaluator* nnEval32, Logger& logger, int boardSize, int maxBatchSizeCap, bool verbose, bool quickTest, bool& fp32BatchSuccessBuf, const string& baseFileName) {

  int maxBatchSize = nnEval->getMaxBatchSize();
  if(maxBatchSize != nnEval32->getMaxBatchSize())
    throw StringError("Inconsistent max batch size for fp16 test");
  if(maxBatchSizeCap > 0)
    maxBatchSize = std::min(maxBatchSize,maxBatchSizeCap);
  if(maxBatchSize <= 0)
    throw StringError("Invalid max batch size for fp16 test");

#ifdef USE_EIGEN_BACKEND
  if (baseFileName == "")
    return true;
#endif

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
  if(boardSize == 9)
    hists = loadHists(TestCommon::getMultiGameSize9Data());
  if(boardSize == 13)
    hists = loadHists(TestCommon::getMultiGameSize13Data());
  if(boardSize == 19)
    hists = loadHists(TestCommon::getMultiGameSize19Data());

  auto evalBoard = [&](NNEvaluator* nnE, const BoardHistory& hist) {
    Board board = hist.getRecentBoard(0);
    MiscNNInputParams nnInputParams;
    nnInputParams.symmetry = (int)(BoardHistory::getSituationRulesAndKoHash(board,hist,hist.presumedNextMovePla,0.5).hash0 & 7);
    NNResultBuf buf;
    bool skipCache = true;
    bool includeOwnerMap = true;
    nnE->evaluate(board,hist,hist.presumedNextMovePla,nnInputParams,buf,skipCache,includeOwnerMap);
    return buf.result;
  };

  {
    if(verbose)
      logger.write("Running evaluations in fp32");
    std::vector<std::shared_ptr<NNOutput>> base;

    bool loadedBaseFromFile = false;

#ifndef USE_EIGEN_BACKEND
    if (baseFileName != "") {
      loadBaseFromFile(base, baseFileName, logger, verbose);
      loadedBaseFromFile = true;
    }
#endif

    if (!loadedBaseFromFile)
      for(const BoardHistory& hist: hists)
        base.push_back(evalBoard(nnEval32,hist));

#ifdef USE_EIGEN_BACKEND
    assert(baseFileName != "");
    saveBaseToFile(base, baseFileName, logger, verbose);
#endif

    std::vector<std::shared_ptr<NNOutput>> batched(hists.size());
    std::vector<std::shared_ptr<NNOutput>> current;
    std::vector<std::shared_ptr<NNOutput>> cbatched(hists.size());

    if(maxBatchSize <= 1)
      batched = base;
    else {
      if(verbose)
        logger.write("Running batched evaluations in fp32");
      auto runThread = [&](int threadIdx) {
        for(size_t i = threadIdx; i<hists.size(); i += maxBatchSize)
          batched[i] = evalBoard(nnEval32,hists[i]);
      };
      vector<std::thread> threads;
      for(int i = 0; i<maxBatchSize; i++)
        threads.push_back(std::thread(runThread,i));
      for(int i = 0; i<maxBatchSize; i++)
        threads[i].join();
    }

    if(nnEval32 != nnEval) {
      if(verbose)
        logger.write("Running evaluations using current config");
      for(const BoardHistory& hist: hists) current.push_back(evalBoard(nnEval,hist));

      if(maxBatchSize <= 1)
        cbatched = current;
      else {
        if(verbose)
          logger.write("Running batched evaluations using current config");
        auto runThread = [&](int threadIdx) {
          for(size_t i = threadIdx; i<hists.size(); i += maxBatchSize)
            cbatched[i] = evalBoard(nnEval,hists[i]);
        };
        vector<std::thread> threads;
        for(int i = 0; i<maxBatchSize; i++)
          threads.push_back(std::thread(runThread,i));
        for(int i = 0; i<maxBatchSize; i++)
          threads[i].join();
      }
    }

    if(verbose) {
      logger.write("Computed stats on " + Global::uint64ToString((uint64_t)base.size()) + " positions");
      logger.write("Reporting the average, 90%, 99%, and max abs error between the following configurations: ");
    }

    fp32BatchSuccessBuf = true;
    bool success = true;
    {
      GpuErrorStats stats;
      for(size_t i = 0; i<base.size(); i++)
        stats.appendStats(base[i], batched[i]);
      if(verbose)
        stats.reportStats("batched fp32 - fp32", logger);
      fp32BatchSuccessBuf = fp32BatchSuccessBuf && stats.checkStats99( 0.45, 0.225, 0.45, 0.0006);
      fp32BatchSuccessBuf = fp32BatchSuccessBuf && stats.checkStatsMax(1.35, 0.900, 1.35, 0.0012);
    }
    if(nnEval32 != nnEval) {
      {
        GpuErrorStats stats;
        for(size_t i = 0; i<base.size(); i++)
          stats.appendStats(base[i], current[i]);
        if(verbose)
          stats.reportStats("current - fp32", logger);
        success = success && stats.checkStats99( 2.0, 1.00, 2.50, 0.0020);
        success = success && stats.checkStatsMax(5.0, 3.00, 6.00, 0.0040);
      }
      {
        GpuErrorStats stats;
        for(size_t i = 0; i<base.size(); i++)
          stats.appendStats(base[i], cbatched[i]);
        if(verbose)
          stats.reportStats("batched current - fp32", logger);
        success = success && stats.checkStats99( 2.0, 1.00, 2.50, 0.0020);
        success = success && stats.checkStatsMax(5.0, 3.00, 6.00, 0.0040);
      }
    }

    return success;
  }
}
