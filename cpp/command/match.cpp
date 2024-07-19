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
          matchupsPerRound.push_back(make_pair(i,j));
          matchupsPerRound.push_back(make_pair(j,i));
        }
      }
    }

    if(cfg.contains("extraPairs")) {
      std::vector<std::pair<int,int>> pairs = cfg.getNonNegativeIntDashedPairs("extraPairs",0,numBots-1);
      for(const std::pair<int,int>& pair: pairs) {
        int p0 = pair.first;
        int p1 = pair.second;
        if(cfg.contains("extraPairsAreOneSidedBW") && cfg.getBool("extraPairsAreOneSidedBW")) {
          matchupsPerRound.push_back(std::make_pair(p0,p1));
        }
        else {
          matchupsPerRound.push_back(std::make_pair(p0,p1));
          matchupsPerRound.push_back(std::make_pair(p1,p0));
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

  auto runMatchLoop = [
    &gameRunner,&matchPairer,&sgfOutputDir,&logger,&gameSeedBase,&patternBonusTables,
    &statsMutex, &gameCount, &timeUsedByBotMap, &movesByBotMap
  ](
    uint64_t threadHash
  ) {
    ofstream* sgfOut = NULL;
    if(sgfOutputDir.length() > 0) {
      sgfOut = new ofstream();
      FileUtils::open(*sgfOut, sgfOutputDir + "/" + Global::uint64ToHexString(threadHash) + ".sgfs");
    }
    auto shouldStopFunc = []() {
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
  for(int i = 0; i<numGameThreads; i++) {
    threads.push_back(std::thread(runMatchLoopProtected, hashRand.nextUInt64()));
  }
  for(int i = 0; i<threads.size(); i++)
    threads[i].join();

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
