#include "../core/global.h"
#include "../core/config_parser.h"
#include "../core/fileutils.h"
#include "../dataio/sgf.h"
#include "../search/asyncbot.h"
#include "../program/setup.h"
#include "../program/playutils.h"
#include "../tests/tests.h"
#include "../command/commandline.h"
#include "../main.h"

#include <chrono>
#include <map>
#include <sstream>
#include <fstream>

using namespace std;

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

  void reportStats(const string& name, Logger& logger) {
    std::sort(winrateError.begin(),winrateError.end());
    std::sort(scoreError.begin(),scoreError.end());
    std::sort(topPolicyDiff.begin(),topPolicyDiff.end());
    std::sort(policyKLDiv.begin(),policyKLDiv.end());

    logger.write(
      name + " winrateError:  " +
      Global::strprintf("%7.5f%% %7.5f%% %7.5f%%", 100*getAverage(winrateError), 100*get90Percentile(winrateError), 100*get99Percentile(winrateError))
    );
    logger.write(
      name + " scoreError:    " +
      Global::strprintf(" %7.5f  %7.5f  %7.5f", getAverage(scoreError), get90Percentile(scoreError), get99Percentile(scoreError))
    );
    logger.write(
      name + " topPolicyDiff: " +
      Global::strprintf("%7.5f%% %7.5f%% %7.5f%%", getAverage(topPolicyDiff), get90Percentile(topPolicyDiff), get99Percentile(topPolicyDiff))
    );
    logger.write(
      name + " policyKLDiv:   " +
      Global::strprintf("%8.6f %8.6f %8.6f", getAverage(policyKLDiv), get90Percentile(policyKLDiv), get99Percentile(policyKLDiv))
    );
  }
};

int MainCmds::testgpuerror(const vector<string>& args) {
  Board::initHash();
  ScoreValue::initTables();
  Rand seedRand;

  ConfigParser cfg;
  string modelFile;
  int boardSize;
  try {
    KataGoCommandLine cmd("Benchmark with gtp config to test speed with different numbers of threads.");
    cmd.addConfigFileArg(KataGoCommandLine::defaultGtpConfigFileName(),"gtp_example.cfg");
    cmd.addModelFileArg();
    TCLAP::ValueArg<int> boardSizeArg("","boardsize", "Size of board to benchmark on (9,13,19), default 19", false, 19, "SIZE");
    cmd.add(boardSizeArg);

    cmd.setShortUsageArgLimit();
    cmd.addOverrideConfigArg();

    cmd.parseArgs(args);

    modelFile = cmd.getModelFile();
    boardSize = boardSizeArg.getValue();
    cmd.getConfig(cfg);

    if(boardSize != 19 && boardSize != 13 && boardSize != 9)
      throw StringError("Board size to test: invalid value " + Global::intToString(boardSize));
  }
  catch (TCLAP::ArgException &e) {
    cerr << "Error: " << e.error() << " for argument " << e.argId() << endl;
    return 1;
  }

  const bool logToStdoutDefault = true;
  const bool logToStderrDefault = false;
  const bool logTimeDefault = false;
  Logger logger(NULL, logToStdoutDefault, logToStderrDefault, logTimeDefault);
  logger.write("Testing average errors between different GPU configurations...");

  auto loadHists = [&](const std::vector<string>& sgfStrs) {
    std::vector<BoardHistory> hists;
    for(const string& sgfStr: sgfStrs) {
      Sgf* sgf = Sgf::parse(sgfStr);
      std::set<Hash128> uniqueHashes;
      const bool hashComments = false;
      const bool hashParent = false;
      const bool flipIfPassOrWFirst = false;
      sgf->iterAllUniquePositions(
        uniqueHashes,
        hashComments,
        hashParent,
        flipIfPassOrWFirst,
        NULL,
        [&](Sgf::PositionSample& sample, const BoardHistory& hist, const string& comments) {
          (void)sample;
          (void)comments;
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

  const string expectedSha256 = "";
  int maxBatchSize;
  if(cfg.contains("nnMaxBatchSize")) {
    maxBatchSize = cfg.getInt("nnMaxBatchSize", 1, 65536);
    logger.write("For batch test, using batch size from nnMaxBatchSize in config: " + Global::intToString(maxBatchSize));
  }
  else if(cfg.contains("numSearchThreads")) {
    maxBatchSize = cfg.getInt("numSearchThreads", 1, 65536);
    logger.write("For batch test, using batch size from numSearchThreads in config: " + Global::intToString(maxBatchSize));
  }
  else {
    maxBatchSize = 16;
    logger.write("For batch test, using default batch size 16");
  }
  const int maxConcurrentEvals = maxBatchSize * 2 + 16;
  const int expectedConcurrentEvals = maxBatchSize * 2 + 16;
  const bool defaultRequireExactNNLen = false;

  logger.write("Initializing nneval using current config...");
  NNEvaluator* nnEval = Setup::initializeNNEvaluator(
    modelFile,modelFile,expectedSha256,cfg,logger,seedRand,maxConcurrentEvals,expectedConcurrentEvals,
    boardSize,boardSize,maxBatchSize,defaultRequireExactNNLen,
    Setup::SETUP_FOR_BENCHMARK
  );

  logger.write("Initializing nneval in fp32...");
  ConfigParser cfgFp32(cfg);
  for(const string& prefix: Setup::getBackendPrefixes()) {
    cfgFp32.overrideKey(prefix + "UseFP16-0", "false");
    cfgFp32.overrideKey(prefix + "UseFP16", "false");
  }
  NNEvaluator* nnEval32 = Setup::initializeNNEvaluator(
    modelFile,modelFile,expectedSha256,cfgFp32,logger,seedRand,maxConcurrentEvals,expectedConcurrentEvals,
    boardSize,boardSize,maxBatchSize,defaultRequireExactNNLen,
    Setup::SETUP_FOR_BENCHMARK
  );

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
    logger.write("Running evaluations in fp32");
    std::vector<std::shared_ptr<NNOutput>> base;
    for(const BoardHistory& hist: hists)
      base.push_back(evalBoard(nnEval32,hist));

    logger.write("Running batched evaluations in fp32");
    std::vector<std::shared_ptr<NNOutput>> batched(hists.size());
    {
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

    logger.write("Running evaluations using current config");
    std::vector<std::shared_ptr<NNOutput>> current;
    for(const BoardHistory& hist: hists) current.push_back(evalBoard(nnEval,hist));

    logger.write("Running batched evaluations using current config");
    std::vector<std::shared_ptr<NNOutput>> cbatched(hists.size());
    {
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

    logger.write("Computed stats on " + Global::uint64ToString((uint64_t)base.size()) + " positions");

    {
      GpuErrorStats stats;
      for(size_t i = 0; i<base.size(); i++) stats.appendStats(base[i], batched[i]);
      stats.reportStats("batched fp32 - fp32", logger);
    }
    {
      GpuErrorStats stats;
      for(size_t i = 0; i<base.size(); i++) stats.appendStats(base[i], current[i]);
      stats.reportStats("current - fp32", logger);
    }
    {
      GpuErrorStats stats;
      for(size_t i = 0; i<base.size(); i++) stats.appendStats(base[i], cbatched[i]);
      stats.reportStats("batched current - fp32", logger);
    }

  }

  delete nnEval32;
  delete nnEval;
  NeuralNet::globalCleanup();
  ScoreValue::freeTables();

  return 0;
}
