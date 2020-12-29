#include "../core/global.h"
#include "../core/makedir.h"
#include "../core/config_parser.h"
#include "../core/timer.h"
#include "../dataio/sgf.h"
#include "../search/asyncbot.h"
#include "../program/setup.h"
#include "../program/play.h"
#include "../command/commandline.h"
#include "../main.h"

#include <csignal>

using namespace std;


static std::atomic<bool> sigReceived(false);
static void signalHandler(int signal)
{
  if(signal == SIGINT || signal == SIGTERM)
    sigReceived.store(true);
}

int MainCmds::match(int argc, const char* const* argv) {
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

    cmd.parse(argc,argv);

    logFile = logFileArg.getValue();
    sgfOutputDir = sgfOutputDirArg.getValue();

    cmd.getConfig(cfg);
  }
  catch (TCLAP::ArgException &e) {
    cerr << "Error: " << e.error() << " for argument " << e.argId() << endl;
    return 1;
  }

  Logger logger;
  logger.addFile(logFile);
  bool logToStdout = cfg.getBool("logToStdout");
  logger.setLogToStdout(logToStdout);

  logger.write("Match Engine starting...");
  logger.write(string("Git revision: ") + Version::getGitRevision());

  //Load per-bot search config, first, which also tells us how many bots we're running
  vector<SearchParams> paramss = Setup::loadParams(cfg);
  assert(paramss.size() > 0);
  int numBots = paramss.size();

  //Load a filter on what bots we actually want to run
  vector<bool> excludeBot(numBots);
  if(cfg.contains("includeBots")) {
    vector<int> includeBots = cfg.getInts("includeBots",0,4096);
    for(int i = 0; i<numBots; i++) {
      if(!contains(includeBots,i))
        excludeBot[i] = true;
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

  //Dedup and load each necessary model exactly once
  vector<string> nnModelFiles;
  vector<int> whichNNModel(numBots);
  for(int i = 0; i<numBots; i++) {
    if(excludeBot[i])
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
      whichNNModel[i] = nnModelFiles.size();
      nnModelFiles.push_back(desiredFile);
    }
  }

  //Load match runner settings
  int numGameThreads = cfg.getInt("numGameThreads",1,16384);
  const string gameSeedBase = Global::uint64ToHexString(seedRand.nextUInt64());

  //Work out an upper bound on how many concurrent nneval requests we could end up making.
  int maxConcurrentEvals;
  int expectedConcurrentEvals;
  {
    //Work out the max threads any one bot uses
    int maxBotThreads = 0;
    for(int i = 0; i<numBots; i++)
      if(paramss[i].numThreads > maxBotThreads)
        maxBotThreads = paramss[i].numThreads;
    //Mutiply by the number of concurrent games we could have
    expectedConcurrentEvals = maxBotThreads * numGameThreads;
    //Multiply by 2 and add some buffer, just so we have plenty of headroom.
    maxConcurrentEvals = expectedConcurrentEvals * 2 + 16;
  }

  //Initialize object for randomizing game settings and running games
  PlaySettings playSettings = PlaySettings::loadForMatch(cfg);
  GameRunner* gameRunner = new GameRunner(cfg, playSettings, logger);
  int maxBoardSizeUsed = 0;
  {
    vector<int> allowedBSizes = gameRunner->getGameInitializer()->getAllowedBSizes();
    for(size_t i = 0; i<allowedBSizes.size(); i++) {
      if(maxBoardSizeUsed < allowedBSizes[i])
        maxBoardSizeUsed = allowedBSizes[i];
    }
    if(maxBoardSizeUsed <= 0)
      maxBoardSizeUsed = NNPos::MAX_BOARD_LEN;
    if(maxBoardSizeUsed > NNPos::MAX_BOARD_LEN)
      throw StringError(
        "Max board size used is greater than the largest size supported by the neural net: "
        + Global::intToString(maxBoardSizeUsed) + " > " + Global::intToString(NNPos::MAX_BOARD_LEN)
      );
    logger.write("Initializing neural net buffer to be size " + Global::intToString(maxBoardSizeUsed) + " since that's the largest board size tested");
  }

  //Initialize neural net inference engine globals, and load models
  Setup::initializeSession(cfg);
  const vector<string>& nnModelNames = nnModelFiles;
  int defaultMaxBatchSize = -1;
  vector<NNEvaluator*> nnEvals = Setup::initializeNNEvaluators(
    nnModelNames,nnModelFiles,cfg,logger,seedRand,maxConcurrentEvals,expectedConcurrentEvals,
    maxBoardSizeUsed,maxBoardSizeUsed,defaultMaxBatchSize,
    Setup::SETUP_FOR_MATCH
  );
  logger.write("Loaded neural net");

  vector<NNEvaluator*> nnEvalsByBot(numBots);
  for(int i = 0; i<numBots; i++) {
    if(excludeBot[i])
      continue;
    nnEvalsByBot[i] = nnEvals[whichNNModel[i]];
  }

  //Initialize object for randomly pairing bots
  bool forSelfPlay = false;
  bool forGateKeeper = false;
  MatchPairer* matchPairer = new MatchPairer(cfg,numBots,botNames,nnEvalsByBot,paramss,forSelfPlay,forGateKeeper,excludeBot);

  //Check for unused config keys
  cfg.warnUnusedKeys(cerr,&logger);

  //Done loading!
  //------------------------------------------------------------------------------------
  logger.write("Loaded all config stuff, starting matches");
  if(!logToStdout)
    cout << "Loaded all config stuff, starting matches" << endl;

  if(sgfOutputDir != string())
    MakeDir::make(sgfOutputDir);

  if(!std::atomic_is_lock_free(&sigReceived))
    throw StringError("sigReceived is not lock free, signal-quitting mechanism for terminating matches will NOT work!");
  std::signal(SIGINT, signalHandler);
  std::signal(SIGTERM, signalHandler);

  auto runMatchLoop = [
    &gameRunner,&matchPairer,&sgfOutputDir,&logger,&gameSeedBase
  ](
    uint64_t threadHash
  ) {
    ofstream* sgfOut = sgfOutputDir.length() > 0 ? (new ofstream(sgfOutputDir + "/" + Global::uint64ToHexString(threadHash) + ".sgfs")) : NULL;
    vector<std::atomic<bool>*> stopConditions = {&sigReceived};

    Rand thisLoopSeedRand;
    while(true) {
      if(sigReceived.load())
        break;

      FinishedGameData* gameData = NULL;

      MatchPairer::BotSpec botSpecB;
      MatchPairer::BotSpec botSpecW;
      if(matchPairer->getMatchup(botSpecB, botSpecW, logger)) {
        string seed = gameSeedBase + ":" + Global::uint64ToHexString(thisLoopSeedRand.nextUInt64());
        gameData = gameRunner->runGame(
          seed, botSpecB, botSpecW, NULL, NULL, logger,
          stopConditions, nullptr, nullptr, false
        );
      }

      bool shouldContinue = gameData != NULL;
      if(gameData != NULL) {
        if(sgfOut != NULL) {
          WriteSgf::writeSgf(*sgfOut,gameData->bName,gameData->wName,gameData->endHist,gameData,false,true);
          (*sgfOut) << endl;
        }
        delete gameData;
      }

      if(sigReceived.load())
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
