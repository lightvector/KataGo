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
  bool logSearchInfo = cfg.getBool("logSearchInfo");
  bool logMoves = cfg.getBool("logMoves");
  int64_t logGamesEvery = cfg.getInt64("logGamesEvery",1,1000000);

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

  //Load bots that should not play one another
  vector<int> secondaryBots;
  if(cfg.contains("secondaryBots"))
    secondaryBots = cfg.getInts("secondaryBots",0,4096);

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

  //Initialize neural net inference engine globals, and load models
  Setup::initializeSession(cfg);
  vector<NNEvaluator*> nnEvals = Setup::initializeNNEvaluators(nnModelFiles,cfg,logger,seedRand);
  logger.write("Loaded neural net");

  //Initialize object for randomizing game settings
  GameInitializer* gameInit = new GameInitializer(cfg);

  //Load match runner settings
  int numGameThreads = cfg.getInt("numGameThreads",1,16384);
  int64_t numGamesTotal = cfg.getInt64("numGamesTotal",1,((int64_t)1) << 62);
  int maxMovesPerGame = cfg.getInt("maxMovesPerGame",1,1 << 30);

  string searchRandSeedBase = Global::uint64ToHexString(seedRand.nextUInt64());

  //Misc other settings
  bool clearBotAfterSearch = cfg.contains("clearBotAfterSearch") ? cfg.getBool("clearBotAfterSearch") : false;

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

  auto runMatchGame = [&paramss,&nnEvals,&whichNNModel,&logger,logSearchInfo,logMoves,maxMovesPerGame,&searchRandSeedBase,clearBotAfterSearch](
    int64_t gameIdx, int botIdxB, int botIdxW, Board& board, Player pla, BoardHistory& hist, int numExtraBlack
  ) {
    string searchRandSeed = searchRandSeedBase + ":" + Global::int64ToString(gameIdx);
    AsyncBot* botB;
    AsyncBot* botW;
    bool clearBotAfterSearchThisGame = clearBotAfterSearch;
    if(botIdxB == botIdxW) {
      AsyncBot* bot = new AsyncBot(paramss[botIdxB], nnEvals[whichNNModel[botIdxB]], &logger, searchRandSeed);
      botB = bot;
      botW = bot;
      //To avoid interactions between the two bots since they're the same
      clearBotAfterSearchThisGame = true;
    }
    else {
      botB = new AsyncBot(paramss[botIdxB], nnEvals[whichNNModel[botIdxB]], &logger, searchRandSeed+"B");
      botW = new AsyncBot(paramss[botIdxW], nnEvals[whichNNModel[botIdxW]], &logger, searchRandSeed+"W");
    }
    bool doEndGameIfAllPassAlive = true;
    Play::runGame(
      board,pla,hist,numExtraBlack,botB,botW,
      doEndGameIfAllPassAlive,clearBotAfterSearchThisGame,
      logger,logSearchInfo,logMoves,
      maxMovesPerGame,sigReceived
    );
    delete botB;
    delete botW;
  };

  mutex matchSetupMutex;
  int64_t numGamesStartedSoFar = 0;
  MatchPairer matchPairer(numBots,secondaryBots);

  auto runMatchLoop = [
    &botNames,&gameInit,&runMatchGame,&matchSetupMutex,
    numGamesTotal,&numGamesStartedSoFar,&matchPairer,&sgfOutputDir,&logger,logGamesEvery,
    &nnModelFiles,&nnEvals
  ](
    uint64_t threadHash
  ) {
    unique_lock<std::mutex> lock(matchSetupMutex,std::defer_lock);
    ofstream* sgfOut = sgfOutputDir.length() > 0 ? (new ofstream(sgfOutputDir + "/" + Global::uint64ToHexString(threadHash) + ".sgfs")) : NULL;

    while(true) {
      lock.lock();
      if(numGamesStartedSoFar >= numGamesTotal)
        break;
      if(sigReceived.load())
        break;
      int64_t gameIdx = numGamesStartedSoFar;
      numGamesStartedSoFar += 1;

      if(numGamesStartedSoFar % logGamesEvery == 0)
        logger.write("Started " + Global::int64ToString(numGamesStartedSoFar) + " games");
      int logNNEvery = logGamesEvery > 100 ? logGamesEvery : 100;
      if(numGamesStartedSoFar % logNNEvery == 0) {
        for(int i = 0; i<nnModelFiles.size(); i++) {
          logger.write(nnModelFiles[i]);
          logger.write("NN rows: " + Global::int64ToString(nnEvals[i]->numRowsProcessed()));
          logger.write("NN batches: " + Global::int64ToString(nnEvals[i]->numBatchesProcessed()));
          logger.write("NN avg batch size: " + Global::doubleToString(nnEvals[i]->averageProcessedBatchSize()));
        }
      }

      pair<int,int> matchup = matchPairer.getMatchup();
      int botIdxB = matchup.first;
      int botIdxW = matchup.second;

      lock.unlock();

      Board board; Player pla; BoardHistory hist; int numExtraBlack;
      gameInit->createGame(board,pla,hist,numExtraBlack);
      Board initialBoard = board;
      Rules initialRules = hist.rules;

      runMatchGame(gameIdx,botIdxB,botIdxW,board,pla,hist,numExtraBlack);

      if(sigReceived.load())
        break;

      if(sgfOut != NULL) {
        string bName = botNames[botIdxB];
        string wName = botNames[botIdxW];
        WriteSgf::writeSgf(*sgfOut,bName,wName,initialRules,initialBoard,hist);
        (*sgfOut) << endl;
      }
    }
    if(sgfOut != NULL)
      sgfOut->close();
  };

  Rand hashRand;
  vector<std::thread> threads;
  for(int i = 0; i<numGameThreads; i++) {
    threads.push_back(std::thread(runMatchLoop, hashRand.nextUInt64()));
  }
  for(int i = 0; i<numGameThreads; i++)
    threads[i].join();

  for(int i = 0; i<nnEvals.size(); i++) {
    delete nnEvals[i];
  }
  NeuralNet::globalCleanup();

  if(sigReceived.load())
    logger.write("Exited cleanly after signal");
  logger.write("All cleaned up, quitting");
  return 0;
}

