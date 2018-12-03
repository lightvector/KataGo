#include "core/global.h"
#include "core/makedir.h"
#include "core/config_parser.h"
#include "core/timer.h"
#include "dataio/sgf.h"
#include "search/asyncbot.h"
#include "program/setup.h"
#include "program/play.h"
#include "program/gitinfo.h"
#include "main.h"

using namespace std;

#define TCLAP_NAMESTARTSTRING "-" //Use single dashes for all flags
#include <tclap/CmdLine.h>

#include <csignal>
static std::atomic<bool> sigReceived(false);
static void signalHandler(int signal)
{
  if(signal == SIGINT || signal == SIGTERM)
    sigReceived.store(true);
}

int MainCmds::match(int argc, const char* const* argv) {
  Board::initHash();
  Rand seedRand;

  string configFile;
  string logFile;
  string sgfOutputDir;
  try {
    TCLAP::CmdLine cmd("Play different nets against each other with different search settings", ' ', "1.0",true);
    TCLAP::ValueArg<string> configFileArg("","config-file","Config file to use (see configs/match_example.cfg)",true,string(),"FILE");
    TCLAP::ValueArg<string> logFileArg("","log-file","Log file to output to",true,string(),"FILE");
    TCLAP::ValueArg<string> sgfOutputDirArg("","sgf-output-dir","Dir to output sgf files",false,string(),"DIR");
    cmd.add(configFileArg);
    cmd.add(logFileArg);
    cmd.add(sgfOutputDirArg);
    cmd.parse(argc,argv);
    configFile = configFileArg.getValue();
    logFile = logFileArg.getValue();
    sgfOutputDir = sgfOutputDirArg.getValue();
  }
  catch (TCLAP::ArgException &e) {
    cerr << "Error: " << e.error() << " for argument " << e.argId() << endl;
    return 1;
  }
  ConfigParser cfg(configFile);

  Logger logger;
  logger.addFile(logFile);
  bool logToStdout = cfg.getBool("logToStdout");
  logger.setLogToStdout(logToStdout);

  logger.write("Match Engine starting...");
  logger.write(string("Git revision: ") + GIT_REVISION);

  //Load per-bot search config, first, which also tells us how many bots we're running
  vector<SearchParams> paramss = Setup::loadParams(cfg);
  assert(paramss.size() > 0);
  int numBots = paramss.size();

  //Load the names of the bots and which model each bot is using
  vector<string> nnModelFilesByBot;
  vector<string> botNames;
  for(int i = 0; i<numBots; i++) {
    string idxStr = Global::intToString(i);

    if(cfg.contains("botName"+idxStr))
      botNames.push_back(cfg.getString("botName"+idxStr));
    else if(numBots == 1)
      botNames.push_back(cfg.getString("botName"));
    else
      throw StringError("If more than one bot, must specify botName0, botName1,... individually");

    if(cfg.contains("nnModelFile"+idxStr))
      nnModelFilesByBot.push_back(cfg.getString("nnModelFile"+idxStr));
    else
      nnModelFilesByBot.push_back(cfg.getString("nnModelFile"));
  }

  //Dedup and load each necessary model exactly once
  vector<string> nnModelFiles;
  vector<int> whichNNModel;
  for(int i = 0; i<numBots; i++) {
    const string& desiredFile = nnModelFilesByBot[i];
    int alreadyFoundIdx = -1;
    for(int j = 0; j<nnModelFiles.size(); j++) {
      if(nnModelFiles[j] == desiredFile) {
        alreadyFoundIdx = j;
        break;
      }
    }
    if(alreadyFoundIdx != -1)
      whichNNModel.push_back(alreadyFoundIdx);
    else {
      whichNNModel.push_back(nnModelFiles.size());
      nnModelFiles.push_back(desiredFile);
    }
  }

  //Load match runner settings
  int numGameThreads = cfg.getInt("numGameThreads",1,16384);

  string searchRandSeedBase = Global::uint64ToHexString(seedRand.nextUInt64());

  //Work out an upper bound on how many concurrent nneval requests we could end up making.
  int maxConcurrentEvals;
  {
    //Work out the max threads any one bot uses
    int maxBotThreads = 0;
    for(int i = 0; i<numBots; i++)
      if(paramss[i].numThreads > maxBotThreads)
        maxBotThreads = paramss[i].numThreads;
    //Mutiply by the number of concurrent games we could have
    maxConcurrentEvals = maxBotThreads * numGameThreads;
    //Multiply by 2 and add some buffer, just so we have plenty of headroom.
    maxConcurrentEvals = maxConcurrentEvals * 2 + 16;
  }

  //Initialize neural net inference engine globals, and load models
  Setup::initializeSession(cfg);
  vector<NNEvaluator*> nnEvals = Setup::initializeNNEvaluators(nnModelFiles,cfg,logger,seedRand,maxConcurrentEvals,false);
  logger.write("Loaded neural net");

  vector<NNEvaluator*> nnEvalsByBot;
  for(int i = 0; i<numBots; i++)
    nnEvalsByBot.push_back(nnEvals[whichNNModel[i]]);


  //Initialize object for randomly pairing bots
  bool forSelfPlay = false;
  bool forGateKeeper = false;
  MatchPairer* matchPairer = new MatchPairer(cfg,numBots,botNames,nnEvalsByBot,paramss,forSelfPlay,forGateKeeper);

  //Initialize object for randomizing game settings and running games
  FancyModes fancyModes;
  GameRunner* gameRunner = new GameRunner(cfg, searchRandSeedBase, forSelfPlay, fancyModes);

  //Check for unused config keys
  {
    vector<string> unusedKeys = cfg.unusedKeys();
    for(size_t i = 0; i<unusedKeys.size(); i++) {
      string msg = "WARNING: Unused key '" + unusedKeys[i] + "' in " + configFile;
      logger.write(msg);
      cerr << msg << endl;
    }
  }

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
    &gameRunner,&matchPairer,&sgfOutputDir,&logger
  ](
    uint64_t threadHash
  ) {
    ofstream* sgfOut = sgfOutputDir.length() > 0 ? (new ofstream(sgfOutputDir + "/" + Global::uint64ToHexString(threadHash) + ".sgfs")) : NULL;
    vector<std::atomic<bool>*> stopConditions = {&sigReceived};

    while(true) {
      if(sigReceived.load())
        break;

      int dataPosLen = 19; //Doesn't matter, we don't actually write training data

      std::function<void(const FinishedGameData&)> writeSgf = [&sgfOut](const FinishedGameData& gameData) {
        if(sgfOut != NULL) {
          WriteSgf::writeSgf(*sgfOut,gameData.bName,gameData.wName,gameData.startHist.rules,gameData.preStartBoard,gameData.endHist,NULL);
          (*sgfOut) << endl;
        }
      };

      bool shouldContinue = gameRunner->runGame(matchPairer, logger, dataPosLen, NULL, &writeSgf, stopConditions);

      if(sigReceived.load())
        break;
      if(!shouldContinue)
        break;
    }
    if(sgfOut != NULL) {
      sgfOut->close();
      delete sgfOut;
    }
  };

  Rand hashRand;
  vector<std::thread> threads;
  for(int i = 0; i<numGameThreads; i++) {
    threads.push_back(std::thread(runMatchLoop, hashRand.nextUInt64()));
  }
  for(int i = 0; i<numGameThreads; i++)
    threads[i].join();

  delete matchPairer;
  delete gameRunner;

  nnEvalsByBot.clear();
  for(int i = 0; i<nnEvals.size(); i++) {
    delete nnEvals[i];
  }
  NeuralNet::globalCleanup();

  if(sigReceived.load())
    logger.write("Exited cleanly after signal");
  logger.write("All cleaned up, quitting");
  return 0;
}

