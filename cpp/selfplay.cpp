#include "core/global.h"
#include "core/makedir.h"
#include "core/config_parser.h"
#include "core/timer.h"
#include "core/threadsafequeue.h"
#include "dataio/sgf.h"
#include "dataio/trainingwrite.h"
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

//-----------------------------------------------------------------------------------------

class GameRunner {
  SearchParams params;
  bool logSearchInfo;
  bool logMoves;
  int maxMovesPerGame;
  string searchRandSeedBase;
  MatchPairer* matchPairer;
  GameInitializer* gameInit;

public:
  GameRunner(ConfigParser& cfg, const string& sRandSeedBase)
    :params(),logSearchInfo(),logMoves(),maxMovesPerGame(),
     searchRandSeedBase(sRandSeedBase),matchPairer(NULL),gameInit(NULL)
  {
    //TODO we should dynamically randomize the no result and draw utilities, and provide them as inputs to the net?
    vector<SearchParams> paramss = Setup::loadParams(cfg);
    if(paramss.size() != 1)
      throw StringError("Can only specify one set of search parameters for self-play");
    params = paramss[0];

    logSearchInfo = cfg.getBool("logSearchInfo");
    logMoves = cfg.getBool("logMoves");
    maxMovesPerGame = cfg.getInt("maxMovesPerGame",1,1 << 30);

    //Mostly the matchpairer for the logging and game counting
    bool forSelfPlay = true;
    matchPairer = new MatchPairer(cfg,forSelfPlay);

    //Initialize object for randomizing game settings
    gameInit = new GameInitializer(cfg);
  }

  ~GameRunner() {
    delete matchPairer;
    delete gameInit;
  }

  bool runGameAndWriteData(const string& playerName, NNEvaluator* nnEval, Logger& logger, ofstream* sgfOut) {
    int64_t gameIdx;
    bool shouldContinue = matchPairer->getMatchup(gameIdx, logger, nnEval, NULL);
    if(!shouldContinue)
      return false;

    Board board; Player pla; BoardHistory hist; int numExtraBlack;
    gameInit->createGame(board,pla,hist,numExtraBlack);
    Board initialBoard = board;
    Rules initialRules = hist.rules;

    string searchRandSeed = searchRandSeedBase + ":" + Global::int64ToString(gameIdx);
    Rand gameRand(searchRandSeed + ":" + "forGameRand");

    //Avoid interactions between the two bots and make sure root noise is effective on each new search
    bool clearBotAfterSearchThisGame = true;
    //In 2% of games, don't autoterminate the game upon all pass alive, to just provide a tiny bit of training data on positions that occur
    //as both players must wrap things up manually, because within the search we don't autoterminate games, meaning that the NN will get
    //called on positions that occur after the game would have been autoterminated.
    bool doEndGameIfAllPassAlive = gameRand.nextBool(0.98);

    AsyncBot* bot = new AsyncBot(params, nnEval, &logger, searchRandSeed);
    Play::runGame(
      board,pla,hist,numExtraBlack,bot,bot,
      doEndGameIfAllPassAlive,clearBotAfterSearchThisGame,
      logger,logSearchInfo,logMoves,
      maxMovesPerGame,sigReceived
    );
    delete bot;

    if(sgfOut != NULL) {
      WriteSgf::writeSgf(*sgfOut,playerName,playerName,initialRules,initialBoard,hist);
      (*sgfOut) << endl;
    }

    return true;
  }

};


//-----------------------------------------------------------------------------------------



int MainCmds::selfPlay(int argc, const char* const* argv) {
  Board::initHash();
  Rand seedRand;

  string configFile;
  string logFile;
  string modelFile;
  string sgfOutputDir;
  string trainDataOutputDir;
  try {
    TCLAP::CmdLine cmd("Generate training data via self play", ' ', "1.0",true);
    TCLAP::ValueArg<string> configFileArg("","config-file","Config file to use",true,string(),"FILE");
    TCLAP::ValueArg<string> logFileArg("","log-file","Log file to output to",true,string(),"FILE");
    //TODO do this instead
    //TCLAP::ValueArg<string> modelsDirArg("","models-dir","Dir to poll and load models from",true,string(),"DIR");
    TCLAP::ValueArg<string> modelFileArg("","model-file","Neural net model file to use",true,string(),"FILE");
    TCLAP::ValueArg<string> sgfOutputDirArg("","sgf-output-dir","Dir to output sgf files",true,string(),"DIR");
    TCLAP::ValueArg<string> trainDataOutputDirArg("","train-data-output-dir","Dir to output training data",true,string(),"DIR");
    cmd.add(configFileArg);
    cmd.add(logFileArg);
    //cmd.add(modelsDirArg);
    cmd.add(modelFileArg);
    cmd.add(sgfOutputDirArg);
    cmd.add(trainDataOutputDirArg);
    cmd.parse(argc,argv);
    configFile = configFileArg.getValue();
    logFile = logFileArg.getValue();
    modelFile = modelFileArg.getValue();
    sgfOutputDir = sgfOutputDirArg.getValue();
    trainDataOutputDir = trainDataOutputDirArg.getValue();
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

  logger.write("Self Play Engine starting...");
  logger.write(string("Git revision: ") + GIT_REVISION);

  NNEvaluator* nnEval;
  {
    Setup::initializeSession(cfg);
    vector<NNEvaluator*> nnEvals = Setup::initializeNNEvaluators({modelFile},cfg,logger,seedRand);
    assert(nnEvals.size() == 1);
    nnEval = nnEvals[0];
  }
  logger.write("Loaded neural net");

  //Load runner settings
  int numGameThreads = cfg.getInt("numGameThreads",1,16384);
  string searchRandSeedBase = Global::uint64ToHexString(seedRand.nextUInt64());

  GameRunner* gameRunner = new GameRunner(cfg, searchRandSeedBase);

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
  logger.write("Loaded all config stuff, starting self play");
  if(!logToStdout)
    cout << "Loaded all config stuff, starting self play" << endl;

  //TODO write to subdirs once we have proper polling for new nn models
  if(sgfOutputDir != string())
    MakeDir::make(sgfOutputDir);

  if(!std::atomic_is_lock_free(&sigReceived))
    throw StringError("sigReceived is not lock free, signal-quitting mechanism for terminating matches will NOT work!");
  std::signal(SIGINT, signalHandler);
  std::signal(SIGTERM, signalHandler);


  // const int maxDataQueueSize = 5000;
  // ThreadSafeQueue<FinishedGameData*> dataToWrite(maxDataQueueSize);
  // auto writeDataLoop = [&dataToWrite]() {
  //   while(true) {
  //     FinishedGameData* data = dataToWrite.waitPop();
  //     if(data == NULL)
  //       break;



  //     delete data;
  //   }
  //   logger.write("Data write loop terminating");
  // };


// class TrainingDataWriter {
//  public:
//   TrainingDataWriter(const string& outputDir, int inputsVersion, int maxRowsPerFile, int posLen);
//   ~TrainingDataWriter();

//   void writeGame(const FinishedGameData& data);
//   void close();

//  private:
//   string outputDir;
//   int inputsVersion;
//   Rand rand;
//   TrainingWriteBuffers* writeBuffers;

//   void writeAndClearIfFull();

// };


  auto runMatchLoop = [
    &gameRunner,
    &sgfOutputDir,&logger,
    &nnEval
  ](
    uint64_t threadHash
  ) {
    //TODO once we have polling this needs to go to a subdir, one for each new net
    ofstream* sgfOut = sgfOutputDir.length() > 0 ? (new ofstream(sgfOutputDir + "/" + Global::uint64ToHexString(threadHash) + ".sgfs")) : NULL;

    while(true) {
      if(sigReceived.load())
        break;

      //TODO this needs to adapt once we have polling for the net
      //TODO this also needs to adapt a bit and make sure to be threadsafe when we dynamcally change the nneval
      string playerName = "bot";
      bool shouldContinue = gameRunner->runGameAndWriteData(playerName, nnEval, logger, sgfOut);
      if(!shouldContinue)
        break;

      if(sigReceived.load())
        break;
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

  delete nnEval;
  NeuralNet::globalCleanup();

  if(sigReceived.load())
    logger.write("Exited cleanly after signal");
  logger.write("All cleaned up, quitting");
  return 0;
}


