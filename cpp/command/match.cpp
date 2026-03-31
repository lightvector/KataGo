#include "../core/global.h"
#include "../core/fileutils.h"
#include "../core/makedir.h"
#include "../core/config_parser.h"
#include "../core/timer.h"
#include "../dataio/sgf.h"
#include "../search/asyncbot.h"
#include "../search/patternbonustable.h"
#include "../program/setup.h"
#include "../program/play.h"
#include "../command/commandline.h"
#include "../main.h"

#include <array>
#include <cmath>
#include <csignal>

using namespace std;


static std::atomic<bool> sigReceived(false);
static std::atomic<bool> shouldStop(false);
static void signalHandler(int signal)
{
  if(signal == SIGINT || signal == SIGTERM) {
    sigReceived.store(true);
    shouldStop.store(true);
  }
}

// ===== Match statistics helpers =====

// Wilson score 95% two-tailed confidence interval (draws counted as 0.5 wins)
static void wilsonCI95(double wins, double n, double& lo, double& hi) {
  const double z = 1.96;
  double p = wins / n;
  double denom = 1.0 + z*z/n;
  double center = (p + z*z/(2*n)) / denom;
  double margin = z * sqrt(p*(1-p)/n + z*z/(4*n*n)) / denom;
  lo = center - margin;
  hi = center + margin;
}

// One-tailed p-value: P(experiment winrate <= 0.5 | data), using normal approximation
static double oneTailedPValue(double wins, double n) {
  if(n <= 0) return 0.5;
  double z = (wins - 0.5*n) / (0.5*sqrt(n));
  return 0.5 * erfc(z / sqrt(2.0));
}

// Bradley-Terry MLE Elo (global, all-bot ranking)
// pairStats: {nameA,nameB} -> {winsA, winsB, draws}  nameA < nameB lexicographically
static void computeBradleyTerryElo(
  const vector<string>& botNames,
  const map<pair<string,string>, array<int64_t,3>>& pairStats,
  vector<double>& outElo,
  vector<double>& outStderr
) {
  int N = (int)botNames.size();
  const double ELO_PER_STRENGTH = 400.0 * log10(exp(1.0)); // ~173.7

  map<string,int> nameIdx;
  for(int i = 0; i < N; i++) nameIdx[botNames[i]] = i;

  // w[i][j] = effective wins of i vs j (draws count 0.5)
  vector<vector<double>> w(N, vector<double>(N, 0.0));
  for(auto& kv : pairStats) {
    auto itA = nameIdx.find(kv.first.first);
    auto itB = nameIdx.find(kv.first.second);
    if(itA == nameIdx.end() || itB == nameIdx.end()) continue;
    int a = itA->second, b = itB->second;
    w[a][b] += kv.second[0] + 0.5 * kv.second[2];
    w[b][a] += kv.second[1] + 0.5 * kv.second[2];
  }

  // theta[0] = 0 (reference, first bot), optimize theta[1..N-1]
  vector<double> theta(N, 0.0);
  int M = N - 1;

  if(M > 0) {
    for(int iter = 0; iter < 200; iter++) {
      vector<double> grad(M, 0.0);
      vector<vector<double>> H(M, vector<double>(M, 0.0));
      for(int i = 0; i < N; i++) {
        for(int j = i+1; j < N; j++) {
          double nij = w[i][j] + w[j][i];
          if(nij <= 0.0) continue;
          double sigma = 1.0 / (1.0 + exp(theta[j] - theta[i]));
          double fish = nij * sigma * (1.0 - sigma);
          double gij = w[i][j] - nij * sigma;
          if(i > 0) { grad[i-1] += gij; H[i-1][i-1] -= fish; }
          if(j > 0) { grad[j-1] -= gij; H[j-1][j-1] -= fish; }
          if(i > 0 && j > 0) { H[i-1][j-1] += fish; H[j-1][i-1] += fish; }
        }
      }
      // Solve H*delta = -grad via Gaussian elimination
      vector<vector<double>> aug(M, vector<double>(M+1, 0.0));
      for(int r = 0; r < M; r++) {
        for(int c = 0; c < M; c++) aug[r][c] = H[r][c];
        aug[r][M] = -grad[r];
      }
      for(int col = 0; col < M; col++) {
        int piv = col;
        for(int r = col+1; r < M; r++)
          if(fabs(aug[r][col]) > fabs(aug[piv][col])) piv = r;
        swap(aug[col], aug[piv]);
        if(fabs(aug[col][col]) < 1e-12) continue;
        double inv = 1.0 / aug[col][col];
        for(int r = col+1; r < M; r++) {
          double f = aug[r][col] * inv;
          for(int c = col; c <= M; c++) aug[r][c] -= f * aug[col][c];
        }
      }
      vector<double> delta(M, 0.0);
      for(int r = M-1; r >= 0; r--) {
        double s = aug[r][M];
        for(int c = r+1; c < M; c++) s -= aug[r][c] * delta[c];
        if(fabs(aug[r][r]) > 1e-12) delta[r] = s / aug[r][r];
      }
      double maxDelta = 0.0;
      for(int r = 0; r < M; r++) {
        theta[r+1] += delta[r];
        maxDelta = max(maxDelta, fabs(delta[r]));
      }
      if(maxDelta < 1e-6) break;
    }
  }

  // Convert log-strength to Elo relative to bot 0
  outElo.resize(N);
  outStderr.resize(N, 0.0);
  for(int i = 0; i < N; i++)
    outElo[i] = (theta[i] - theta[0]) * ELO_PER_STRENGTH;

  // Fisher information diagonal -> stderr
  for(int i = 1; i < N; i++) {
    double fish = 0.0;
    for(int j = 0; j < N; j++) {
      if(j == i) continue;
      double nij = w[i][j] + w[j][i];
      if(nij <= 0.0) continue;
      double sigma = 1.0 / (1.0 + exp(theta[j] - theta[i]));
      fish += nij * sigma * (1.0 - sigma);
    }
    if(fish > 0.0) outStderr[i] = ELO_PER_STRENGTH / sqrt(fish);
  }
}

int MainCmds::match(const vector<string>& args) {
  Board::initHash();
  ScoreValue::initTables();
  Rand seedRand;

  ConfigParser cfg;
  string logFile;
  string sgfOutputDir;
  try {
    KataGoCommandLine cmd("Play different nets against each other with different search settings in a match or tournament.");
    cmd.addConfigFileArg("","match_example.cfg");

    TCLAP::ValueArg<string> logFileArg("","log-file","Log file to output to",false,string(),"FILE");
    TCLAP::ValueArg<string> sgfOutputDirArg("","sgf-output-dir","Dir to output sgf files",false,string(),"DIR");

    cmd.add(logFileArg);
    cmd.add(sgfOutputDirArg);

    cmd.setShortUsageArgLimit();
    cmd.addOverrideConfigArg();

    cmd.parseArgs(args);

    logFile = logFileArg.getValue();
    sgfOutputDir = sgfOutputDirArg.getValue();

    cmd.getConfig(cfg);
  }
  catch (TCLAP::ArgException &e) {
    cerr << "Error: " << e.error() << " for argument " << e.argId() << endl;
    return 1;
  }

  Logger logger(&cfg);
  logger.addFile(logFile);

  logger.write("Match Engine starting...");
  logger.write(string("Git revision: ") + Version::getGitRevision());

  //Load per-bot search config, first, which also tells us how many bots we're running
  vector<SearchParams> paramss = Setup::loadParams(cfg,Setup::SETUP_FOR_MATCH);
  assert(paramss.size() > 0);
  int numBots = (int)paramss.size();

  //Figure out all pairs of bots that will be playing.
  std::vector<std::pair<int,int>> matchupsPerRound;
  {
    //Load a filter on what bots we actually want to run. By default, include everything.
    vector<bool> includeBot(numBots);
    if(cfg.contains("includeBots")) {
      vector<int> includeBotIdxs = cfg.getInts("includeBots",0,Setup::MAX_BOT_PARAMS_FROM_CFG);
      for(int i = 0; i<numBots; i++) {
        if(contains(includeBotIdxs,i))
          includeBot[i] = true;
      }
    }
    else {
      for(int i = 0; i<numBots; i++) {
        includeBot[i] = true;
      }
    }

    std::vector<int> secondaryBotIdxs;
    if(cfg.contains("secondaryBots"))
      secondaryBotIdxs = cfg.getInts("secondaryBots",0,Setup::MAX_BOT_PARAMS_FROM_CFG);
    for(int i = 0; i<secondaryBotIdxs.size(); i++)
      assert(secondaryBotIdxs[i] >= 0 && secondaryBotIdxs[i] < numBots);

    for(int i = 0; i<numBots; i++) {
      if(!includeBot[i])
        continue;
      for(int j = 0; j<numBots; j++) {
        if(!includeBot[j])
          continue;
        if(i < j && !(contains(secondaryBotIdxs,i) && contains(secondaryBotIdxs,j))) {
          matchupsPerRound.emplace_back(i,j);
          matchupsPerRound.emplace_back(j,i);
        }
      }
    }

    if(cfg.contains("extraPairs")) {
      std::vector<std::pair<int,int>> pairs = cfg.getNonNegativeIntDashedPairs("extraPairs",0,numBots-1);
      for(const std::pair<int,int>& pair: pairs) {
        int p0 = pair.first;
        int p1 = pair.second;
        if(cfg.contains("extraPairsAreOneSidedBW") && cfg.getBool("extraPairsAreOneSidedBW")) {
          matchupsPerRound.emplace_back(p0,p1);
        }
        else {
          matchupsPerRound.emplace_back(p0,p1);
          matchupsPerRound.emplace_back(p1,p0);
        }
      }
    }
  }

  //Load the names of the bots and which model each bot is using
  vector<string> nnModelFilesByBot(numBots);
  vector<string> botNames(numBots);
  for(int i = 0; i<numBots; i++) {
    string idxStr = Global::intToString(i);

    if(cfg.contains("botName"+idxStr))
      botNames[i] = cfg.getString("botName"+idxStr);
    else if(numBots == 1)
      botNames[i] = cfg.getString("botName");
    else
      throw StringError("If more than one bot, must specify botName0, botName1,... individually");

    if(cfg.contains("nnModelFile"+idxStr))
      nnModelFilesByBot[i] = cfg.getString("nnModelFile"+idxStr);
    else
      nnModelFilesByBot[i] = cfg.getString("nnModelFile");
  }

  vector<bool> botIsUsed(numBots);
  for(const std::pair<int,int>& pair : matchupsPerRound) {
    botIsUsed[pair.first] = true;
    botIsUsed[pair.second] = true;
  }

  //Dedup and load each necessary model exactly once
  vector<string> nnModelFiles;
  vector<int> whichNNModel(numBots);
  for(int i = 0; i<numBots; i++) {
    if(!botIsUsed[i])
      continue;

    const string& desiredFile = nnModelFilesByBot[i];
    int alreadyFoundIdx = -1;
    for(int j = 0; j<nnModelFiles.size(); j++) {
      if(nnModelFiles[j] == desiredFile) {
        alreadyFoundIdx = j;
        break;
      }
    }
    if(alreadyFoundIdx != -1)
      whichNNModel[i] = alreadyFoundIdx;
    else {
      whichNNModel[i] = (int)nnModelFiles.size();
      nnModelFiles.push_back(desiredFile);
    }
  }

  //Load match runner settings
  int numGameThreads = cfg.getInt("numGameThreads",1,16384);
  const string gameSeedBase = Global::uint64ToHexString(seedRand.nextUInt64());

  //Work out an upper bound on how many concurrent nneval requests we could end up making.
  int expectedConcurrentEvals;
  {
    //Work out the max threads any one bot uses
    int maxBotThreads = 0;
    for(int i = 0; i<numBots; i++)
      if(paramss[i].numThreads > maxBotThreads)
        maxBotThreads = paramss[i].numThreads;
    //Mutiply by the number of concurrent games we could have
    expectedConcurrentEvals = maxBotThreads * numGameThreads;
  }

  //Initialize object for randomizing game settings and running games
  PlaySettings playSettings = PlaySettings::loadForMatch(cfg);
  GameRunner* gameRunner = new GameRunner(cfg, playSettings, logger);
  const int minBoardXSizeUsed = gameRunner->getGameInitializer()->getMinBoardXSize();
  const int minBoardYSizeUsed = gameRunner->getGameInitializer()->getMinBoardYSize();
  const int maxBoardXSizeUsed = gameRunner->getGameInitializer()->getMaxBoardXSize();
  const int maxBoardYSizeUsed = gameRunner->getGameInitializer()->getMaxBoardYSize();

  //Initialize neural net inference engine globals, and load models
  Setup::initializeSession(cfg);
  const vector<string>& nnModelNames = nnModelFiles;
  const int defaultMaxBatchSize = -1;
  const bool defaultRequireExactNNLen = minBoardXSizeUsed == maxBoardXSizeUsed && minBoardYSizeUsed == maxBoardYSizeUsed;
  const bool disableFP16 = false;
  const vector<string> expectedSha256s;
  vector<NNEvaluator*> nnEvals = Setup::initializeNNEvaluators(
    nnModelNames,nnModelFiles,expectedSha256s,cfg,logger,seedRand,expectedConcurrentEvals,
    maxBoardXSizeUsed,maxBoardYSizeUsed,defaultMaxBatchSize,defaultRequireExactNNLen,disableFP16,
    Setup::SETUP_FOR_MATCH
  );
  logger.write("Loaded neural net");

  vector<NNEvaluator*> nnEvalsByBot(numBots);
  for(int i = 0; i<numBots; i++) {
    if(!botIsUsed[i])
      continue;
    nnEvalsByBot[i] = nnEvals[whichNNModel[i]];
  }

  std::vector<std::unique_ptr<PatternBonusTable>> patternBonusTables = Setup::loadAvoidSgfPatternBonusTables(cfg,logger);
  assert(patternBonusTables.size() == numBots);

  //Initialize object for randomly pairing bots
  int64_t numGamesTotal = cfg.getInt64("numGamesTotal",1,((int64_t)1) << 62);
  MatchPairer* matchPairer = new MatchPairer(cfg,numBots,botNames,nnEvalsByBot,paramss,matchupsPerRound,numGamesTotal);

  //Check for unused config keys
  cfg.warnUnusedKeys(cerr,&logger);
  for(int i = 0; i<numBots; i++) {
    if(!botIsUsed[i])
      continue;
    Setup::maybeWarnHumanSLParams(paramss[i],nnEvalsByBot[i],NULL,cerr,&logger);
  }

  //Done loading!
  //------------------------------------------------------------------------------------
  logger.write("Loaded all config stuff, starting matches");
  if(!logger.isLoggingToStdout())
    cout << "Loaded all config stuff, starting matches" << endl;

  if(sgfOutputDir != string())
    MakeDir::make(sgfOutputDir);

  if(!std::atomic_is_lock_free(&shouldStop))
    throw StringError("shouldStop is not lock free, signal-quitting mechanism for terminating matches will NOT work!");
  std::signal(SIGINT, signalHandler);
  std::signal(SIGTERM, signalHandler);


  std::mutex statsMutex;
  int64_t gameCount = 0;
  std::map<string,double> timeUsedByBotMap;
  std::map<string,double> movesByBotMap;
  map<pair<string,string>, array<int64_t,3>> pairStats;
  // key: {nameA, nameB} with nameA < nameB lexicographically
  // value: {winsA, winsB, draws}

  auto runMatchLoop = [
    &gameRunner,&matchPairer,&sgfOutputDir,&logger,&gameSeedBase,&patternBonusTables,
    &statsMutex, &gameCount, &timeUsedByBotMap, &movesByBotMap, &pairStats
  ](
    uint64_t threadHash
  ) {
    ofstream* sgfOut = NULL;
    if(sgfOutputDir.length() > 0) {
      sgfOut = new ofstream();
      FileUtils::open(*sgfOut, sgfOutputDir + "/" + Global::uint64ToHexString(threadHash) + ".sgfs");
    }
    auto shouldStopFunc = []() noexcept {
      return shouldStop.load();
    };
    WaitableFlag* shouldPause = nullptr;

    Rand thisLoopSeedRand;
    while(true) {
      if(shouldStop.load())
        break;

      FinishedGameData* gameData = NULL;

      MatchPairer::BotSpec botSpecB;
      MatchPairer::BotSpec botSpecW;
      if(matchPairer->getMatchup(botSpecB, botSpecW, logger)) {
        string seed = gameSeedBase + ":" + Global::uint64ToHexString(thisLoopSeedRand.nextUInt64());
        std::function<void(const MatchPairer::BotSpec&, Search*)> afterInitialization = [&patternBonusTables](const MatchPairer::BotSpec& spec, Search* search) {
          assert(spec.botIdx < patternBonusTables.size());
          search->setCopyOfExternalPatternBonusTable(patternBonusTables[spec.botIdx]);
        };
        gameData = gameRunner->runGame(
          seed, botSpecB, botSpecW, NULL, NULL, logger,
          shouldStopFunc, shouldPause, nullptr, afterInitialization, nullptr
        );
      }

      bool shouldContinue = gameData != NULL;
      if(gameData != NULL) {
        if(sgfOut != NULL) {
          WriteSgf::writeSgf(*sgfOut,gameData->bName,gameData->wName,gameData->endHist,gameData,false,true);
          (*sgfOut) << endl;
        }

        {
          std::lock_guard<std::mutex> lock(statsMutex);
          gameCount += 1;
          timeUsedByBotMap[gameData->bName] += gameData->bTimeUsed;
          timeUsedByBotMap[gameData->wName] += gameData->wTimeUsed;
          movesByBotMap[gameData->bName] += (double)gameData->bMoveCount;
          movesByBotMap[gameData->wName] += (double)gameData->wMoveCount;

          // Update pairwise W/L/D stats
          {
            const string& bName = gameData->bName;
            const string& wName = gameData->wName;
            Player winner = gameData->endHist.winner;
            bool aIsBlack = (bName < wName);
            const string& nameA = aIsBlack ? bName : wName;
            const string& nameB = aIsBlack ? wName : bName;
            auto& ps = pairStats[{nameA, nameB}];
            if(winner == P_BLACK)      { if(aIsBlack) ps[0]++; else ps[1]++; }
            else if(winner == P_WHITE) { if(aIsBlack) ps[1]++; else ps[0]++; }
            else                       { ps[2]++; }
          }

          int64_t x = gameCount;
          while(x % 2 == 0 && x > 1) x /= 2;
          if(x == 1 || x == 3 || x == 5) {
            for(auto& pair : timeUsedByBotMap) {
              logger.write(
                "Avg move time used by " + pair.first + " " +
                Global::doubleToString(pair.second / movesByBotMap[pair.first]) + " " +
                Global::doubleToString(movesByBotMap[pair.first]) + " moves"
              );
            }
          }
        }

        delete gameData;
      }

      if(shouldStop.load())
        break;
      if(!shouldContinue)
        break;
    }
    if(sgfOut != NULL) {
      sgfOut->close();
      delete sgfOut;
    }
    logger.write("Match loop thread terminating");
  };
  auto runMatchLoopProtected = [&logger,&runMatchLoop](uint64_t threadHash) {
    Logger::logThreadUncaught("match loop", &logger, [&](){ runMatchLoop(threadHash); });
  };


  Rand hashRand;
  vector<std::thread> threads;
  threads.reserve(numGameThreads);
  for(int i = 0; i<numGameThreads; i++) {
    threads.emplace_back(runMatchLoopProtected, hashRand.nextUInt64());
  }
  for(int i = 0; i<threads.size(); i++)
    threads[i].join();

  // ===== Final match statistics =====
  if(!pairStats.empty()) {
    vector<string> activeBots;
    {
      set<string> seen;
      for(auto& kv : pairStats) {
        seen.insert(kv.first.first);
        seen.insert(kv.first.second);
      }
      activeBots.assign(seen.begin(), seen.end());
    }

    vector<double> elo, eloStderr;
    computeBradleyTerryElo(activeBots, pairStats, elo, eloStderr);

    logger.write("");
    logger.write("=== match Results ===");
    logger.write("Global Elo (Bradley-Terry MLE, reference=" + activeBots[0] + "):");
    for(int i = 0; i < (int)activeBots.size(); i++) {
      string sign = (elo[i] >= 0) ? "+" : "";
      string line = "  " + activeBots[i] + " : " +
        sign + Global::strprintf("%.1f", elo[i]) + " +/- " + Global::strprintf("%.1f", eloStderr[i]);
      if(i == 0) line += "  (reference)";
      logger.write(line);
    }
    logger.write("");
    logger.write("Pairwise summary:");
    for(auto& kv : pairStats) {
      int64_t wA = kv.second[0], wB = kv.second[1], d = kv.second[2];
      int64_t total = wA + wB + d;
      if(total == 0) continue;
      double wins = wA + 0.5 * d;
      double lo, hi;
      wilsonCI95(wins, (double)total, lo, hi);
      double pval = oneTailedPValue(wins, (double)total);
      string sig = (pval < 0.05) ? " *" : "";
      logger.write(
        "  " + kv.first.first + " vs " + kv.first.second +
        " : Games=" + Global::int64ToString(total) +
        " W=" + Global::int64ToString(wA) +
        " L=" + Global::int64ToString(wB) +
        " D=" + Global::int64ToString(d) +
        " | " + kv.first.first + " winrate=" + Global::strprintf("%.3f", wins/total) +
        " [95% CI: " + Global::strprintf("%.3f", lo) + ", " + Global::strprintf("%.3f", hi) + "]" +
        " | p=" + Global::strprintf("%.4f", pval) + sig
      );
    }
    logger.write("");
  }

  delete matchPairer;
  delete gameRunner;

  nnEvalsByBot.clear();
  for(int i = 0; i<nnEvals.size(); i++) {
    if(nnEvals[i] != NULL) {
      logger.write(nnEvals[i]->getModelFileName());
      logger.write("NN rows: " + Global::int64ToString(nnEvals[i]->numRowsProcessed()));
      logger.write("NN batches: " + Global::int64ToString(nnEvals[i]->numBatchesProcessed()));
      logger.write("NN avg batch size: " + Global::doubleToString(nnEvals[i]->averageProcessedBatchSize()));
      delete nnEvals[i];
    }
  }
  NeuralNet::globalCleanup();
  ScoreValue::freeTables();

  if(sigReceived.load())
    logger.write("Exited cleanly after signal");
  logger.write("All cleaned up, quitting");
  return 0;
}
