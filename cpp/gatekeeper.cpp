#include "core/global.h"
#include "core/datetime.h"
#include "core/makedir.h"
#include "core/config_parser.h"
#include "core/timer.h"
#include "core/threadsafequeue.h"
#include "dataio/sgf.h"
#include "dataio/trainingwrite.h"
#include "dataio/loadmodel.h"
#include "search/asyncbot.h"
#include "program/setup.h"
#include "program/play.h"
#include "main.h"

#include <sstream>

#define TCLAP_NAMESTARTSTRING "-" //Use single dashes for all flags
#include <tclap/CmdLine.h>

#include <cstdio>
#include <chrono>
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

//-----------------------------------------------------------------------------------------


//Wraps together a neural net and handles for outputting training data for it.
//There should be one of these active for each gatekeeping match we run, and one active thread
//looping and actually performing the data output
//DOES take ownership of the NNEvaluators
namespace {
  struct NetAndStuff {
    string modelNameBaseline;
    string modelNameCandidate;
    NNEvaluator* nnEvalBaseline;
    NNEvaluator* nnEvalCandidate;
    MatchPairer* matchPairer;

    string testModelDir;

    ThreadSafeQueue<FinishedGameData*> finishedGameQueue;
    int numGameThreads;
    bool isDraining;

    double drawEquivalentWinsForWhite;
    double noResultUtilityForWhite;

    int numGamesTallied;
    double numBaselineWinPoints;
    double numCandidateWinPoints;

    ofstream* sgfOut;

    std::atomic<bool> terminated;

  public:
    NetAndStuff(ConfigParser& cfg, const string& nameB, const string& nameC, const string& tModelDir, NNEvaluator* nevalB, NNEvaluator* nevalC, ofstream* sOut)
      :modelNameBaseline(nameB),
       modelNameCandidate(nameC),
       nnEvalBaseline(nevalB),
       nnEvalCandidate(nevalC),
       matchPairer(NULL),
       testModelDir(tModelDir),
       finishedGameQueue(),
       numGameThreads(0),
       isDraining(false),
       drawEquivalentWinsForWhite(0.5),
       noResultUtilityForWhite(0.0),
       numGamesTallied(0),
       numBaselineWinPoints(0.0),
       numCandidateWinPoints(0.0),
       sgfOut(sOut),
       terminated(false)
    {
      SearchParams baseParams = Setup::loadSingleParams(cfg);

      drawEquivalentWinsForWhite = baseParams.drawEquivalentWinsForWhite;
      noResultUtilityForWhite = baseParams.noResultUtilityForWhite;

      //Initialize object for randomly pairing bots. Actually since this is only selfplay, this only
      //ever gives is the trivial self-pairing, but we use it also for keeping the game count and some logging.
      bool forSelfPlay = false;
      bool forGateKeeper = true;
      matchPairer = new MatchPairer(
        cfg, 2, {modelNameBaseline,modelNameCandidate}, {nnEvalBaseline,nnEvalCandidate}, {baseParams, baseParams}, forSelfPlay, forGateKeeper
      );
    }

    ~NetAndStuff() {
      delete matchPairer;
      delete nnEvalCandidate;
      delete nnEvalBaseline;
      if(sgfOut != NULL)
        delete sgfOut;
    }

    void runWriteDataLoop(Logger& logger) {
      while(true) {
        FinishedGameData* data;
        bool suc = finishedGameQueue.waitPop(data);
        if(!suc || data == NULL)
          break;

        double whitePoints;
        double blackPoints;
        if(data->endHist.isGameFinished && data->endHist.isNoResult) {
          whitePoints = drawEquivalentWinsForWhite;
          blackPoints = 1.0 - whitePoints;
          logger.write("Game " + Global::intToString(numGamesTallied) + ": noresult");
        }
        else {
          BoardHistory hist(data->endHist);
          Board endBoard = hist.getRecentBoard(0);
          //Force game end just in caseif we crossed a move limit
          if(!hist.isGameFinished)
            hist.endAndScoreGameNow(endBoard);

          ostringstream oresult;
          WriteSgf::printGameResult(oresult,hist);
          if(hist.winner == P_BLACK) {
            whitePoints = 0.0;
            blackPoints = 1.0;
            logger.write("Game " + Global::intToString(numGamesTallied) + ": winner black " + data->bName + " " + oresult.str());
          }
          else if(hist.winner == P_WHITE) {
            whitePoints = 1.0;
            blackPoints = 0.0;
            logger.write("Game " + Global::intToString(numGamesTallied) + ": winner white " + data->wName + " " + oresult.str());
          }
          else {
            whitePoints = 0.5 * noResultUtilityForWhite + 0.5;
            blackPoints = 1.0 - whitePoints;
            logger.write("Game " + Global::intToString(numGamesTallied) + ": draw " + oresult.str());
          }
        }

        numGamesTallied++;
        numBaselineWinPoints += (data->bIdx == 0) ? blackPoints : whitePoints;
        numCandidateWinPoints += (data->bIdx == 1) ? blackPoints : whitePoints;

        if(sgfOut != NULL) {
          assert(data->startHist.moveHistory.size() <= data->endHist.moveHistory.size());
          WriteSgf::writeSgf(*sgfOut,data->bName,data->wName,data->endHist,data,false);
          (*sgfOut) << endl;
        }
        delete data;

        //Terminate games if one side has won enough to guarantee the victory.
        int64_t numGamesRemaining = matchPairer->getNumGamesTotalToGenerate() - numGamesTallied;
        assert(numGamesRemaining >= 0);
        if(numGamesRemaining > 0) {
          if(numCandidateWinPoints >= (numBaselineWinPoints + numGamesRemaining)) {
            logger.write("Candidate has already won enough games, terminating remaning games");
            terminated.store(true);
          }
          else if(numBaselineWinPoints > numCandidateWinPoints + numGamesRemaining + 1e-10) {
            logger.write("Candidate has already lost too many games, terminating remaning games");
            terminated.store(true);
          }
        }

      }

      if(sgfOut != NULL)
        sgfOut->close();
    }

    //NOT threadsafe - needs to be externally synchronized
    //Game threads beginning a game using this net call this
    void registerGameThread() {
      assert(!isDraining);
      numGameThreads++;
    }

    //NOT threadsafe - needs to be externally synchronized
    //Game threads finishing a game using this net call this
    void unregisterGameThread() {
      numGameThreads--;
    }

    //NOT threadsafe - needs to be externally synchronized
    //Mark that we should start draining this net and not starting new games with it
    void markAsDraining() {
      if(!isDraining) {
        isDraining = true;
        finishedGameQueue.setReadOnly();
      }
    }

  };
}


//-----------------------------------------------------------------------------------------


int MainCmds::gatekeeper(int argc, const char* const* argv) {
  Board::initHash();
  ScoreValue::initTables();
  Rand seedRand;

  string configFile;
  string testModelsDir;
  string acceptedModelsDir;
  string rejectedModelsDir;
  string sgfOutputDir;
  bool noAutoRejectOldModels;
  bool quitIfNoNetsToTest;
  try {
    TCLAP::CmdLine cmd("Test neural nets to see if they should be accepted", ' ', Version::getKataGoVersionForHelp(),true);
    TCLAP::ValueArg<string> configFileArg("","config-file","Config file to use",true,string(),"FILE");
    TCLAP::ValueArg<string> testModelsDirArg("","test-models-dir","Dir to poll and load models from",true,string(),"DIR");
    TCLAP::ValueArg<string> sgfOutputDirArg("","sgf-output-dir","Dir to output sgf files",true,string(),"DIR");
    TCLAP::ValueArg<string> acceptedModelsDirArg("","accepted-models-dir","Dir to write good models to",true,string(),"DIR");
    TCLAP::ValueArg<string> rejectedModelsDirArg("","rejected-models-dir","Dir to write bad models to",true,string(),"DIR");
    TCLAP::SwitchArg noAutoRejectOldModelsArg("","no-autoreject-old-models","Test older models than the latest accepted model");
    TCLAP::SwitchArg quitIfNoNetsToTestArg("","quit-if-no-nets-to-test","Terminate instead of waiting for a new net to test");
    cmd.add(configFileArg);
    cmd.add(testModelsDirArg);
    cmd.add(sgfOutputDirArg);
    cmd.add(acceptedModelsDirArg);
    cmd.add(rejectedModelsDirArg);
    cmd.add(noAutoRejectOldModelsArg);
    cmd.add(quitIfNoNetsToTestArg);
    cmd.parse(argc,argv);
    configFile = configFileArg.getValue();
    testModelsDir = testModelsDirArg.getValue();
    sgfOutputDir = sgfOutputDirArg.getValue();
    acceptedModelsDir = acceptedModelsDirArg.getValue();
    rejectedModelsDir = rejectedModelsDirArg.getValue();
    noAutoRejectOldModels = noAutoRejectOldModelsArg.getValue();
    quitIfNoNetsToTest = quitIfNoNetsToTestArg.getValue();

    auto checkDirNonEmpty = [](const char* flag, const string& s) {
      if(s.length() <= 0)
        throw StringError("Empty directory specified for " + string(flag));
    };
    checkDirNonEmpty("test-models-dir",testModelsDir);
    checkDirNonEmpty("sgf-output-dir",sgfOutputDir);
    checkDirNonEmpty("accepted-models-dir",acceptedModelsDir);
    checkDirNonEmpty("rejected-models-dir",rejectedModelsDir);
  }
  catch (TCLAP::ArgException &e) {
    cerr << "Error: " << e.error() << " for argument " << e.argId() << endl;
    return 1;
  }
  ConfigParser cfg(configFile);

  MakeDir::make(testModelsDir);
  MakeDir::make(acceptedModelsDir);
  MakeDir::make(rejectedModelsDir);
  MakeDir::make(sgfOutputDir);

  Logger logger;
  //Log to random file name to better support starting/stopping as well as multiple parallel runs
  logger.addFile(sgfOutputDir + "/log" + DateTime::getCompactDateTimeString() + "-" + Global::uint64ToHexString(seedRand.nextUInt64()) + ".log");
  bool logToStdout = cfg.getBool("logToStdout");
  logger.setLogToStdout(logToStdout);

  logger.write("Gatekeeper Engine starting...");
  logger.write(string("Git revision: ") + Version::getGitRevision());

  //Load runner settings
  const int numGameThreads = cfg.getInt("numGameThreads",1,16384);
  const string searchRandSeedBase = Global::uint64ToHexString(seedRand.nextUInt64());

  FancyModes fancyModes;
  fancyModes.allowResignation = cfg.getBool("allowResignation");
  fancyModes.resignThreshold = cfg.getDouble("resignThreshold",-1.0,0.0); //Threshold on [-1,1], regardless of winLossUtilityFactor
  fancyModes.resignConsecTurns = cfg.getInt("resignConsecTurns",1,100);
  fancyModes.compensateKomiVisits = cfg.contains("compensateKomiVisits") ? cfg.getInt("compensateKomiVisits",1,10000) : 100;

  GameRunner* gameRunner = new GameRunner(cfg, searchRandSeedBase, fancyModes, logger);

  Setup::initializeSession(cfg);

  //Done loading!
  //------------------------------------------------------------------------------------
  logger.write("Loaded all config stuff, watching for new neural nets in " + testModelsDir);
  if(!logToStdout)
    cout << "Loaded all config stuff, watching for new neural nets in " + testModelsDir << endl;

  if(!std::atomic_is_lock_free(&shouldStop))
    throw StringError("shouldStop is not lock free, signal-quitting mechanism for terminating matches will NOT work!");
  std::signal(SIGINT, signalHandler);
  std::signal(SIGTERM, signalHandler);

  std::mutex netAndStuffMutex;
  NetAndStuff* netAndStuff = NULL;
  bool netAndStuffDataIsWritten = false;
  std::condition_variable waitNetAndStuffDataIsWritten;

  //Looping thread for writing data for a single net
  auto dataWriteLoop = [&netAndStuffMutex,&netAndStuff,&netAndStuffDataIsWritten,&waitNetAndStuffDataIsWritten,&logger]() {
    string modelNameBaseline = netAndStuff->modelNameBaseline;
    string modelNameCandidate = netAndStuff->modelNameCandidate;
    logger.write("Data write loop starting for neural net: " + modelNameBaseline + " vs " + modelNameCandidate);
    netAndStuff->runWriteDataLoop(logger);
    logger.write("Data write loop finishing for neural net: " + modelNameBaseline + " vs " + modelNameCandidate);

    std::unique_lock<std::mutex> lock(netAndStuffMutex);
    netAndStuffDataIsWritten = true;
    waitNetAndStuffDataIsWritten.notify_all();

    lock.unlock();
    logger.write("Data write loop cleaned up and terminating for " + modelNameBaseline + " vs " + modelNameCandidate);
  };

  auto loadLatestNeuralNet =
    [&testModelsDir,&rejectedModelsDir,&acceptedModelsDir,&sgfOutputDir,&logger,&cfg,numGameThreads,noAutoRejectOldModels]() -> NetAndStuff* {
    Rand rand;

    string testModelName;
    string testModelFile;
    string testModelDir;
    time_t testModelTime;
    bool foundModel = LoadModel::findLatestModel(testModelsDir, logger, testModelName, testModelFile, testModelDir, testModelTime);

    //No new neural nets yet
    if(!foundModel || testModelFile == "/dev/null")
      return NULL;

    logger.write("Found new candidate neural net " + testModelName);

    string acceptedModelName;
    string acceptedModelFile;
    string acceptedModelDir;
    time_t acceptedModelTime;
    foundModel = LoadModel::findLatestModel(acceptedModelsDir, logger, acceptedModelName, acceptedModelFile, acceptedModelDir, acceptedModelTime);
    if(!foundModel) {
      logger.write("Error: No accepted model found in " + acceptedModelsDir);
      return NULL;
    }

    if(acceptedModelTime > testModelTime && !noAutoRejectOldModels) {
      string renameDest = rejectedModelsDir + "/" + testModelName;
      logger.write("Rejecting " + testModelDir + " automatically since older than best accepted model");
      logger.write("Moving " + testModelDir + " to " + renameDest);
      std::rename(testModelDir.c_str(),renameDest.c_str());
      return NULL;
    }

    // * 2 + 16 just in case to have plenty of room
    int maxConcurrentEvals = cfg.getInt("numSearchThreads") * numGameThreads * 2 + 16;
    int defaultMaxBatchSize = -1;

    NNEvaluator* testNNEval = Setup::initializeNNEvaluator(
      testModelName,testModelFile,cfg,logger,rand,maxConcurrentEvals,
      NNPos::MAX_BOARD_LEN,NNPos::MAX_BOARD_LEN,defaultMaxBatchSize,
      Setup::SETUP_FOR_OTHER
    );
    logger.write("Loaded candidate neural net " + testModelName + " from: " + testModelFile);

    NNEvaluator* acceptedNNEval = Setup::initializeNNEvaluator(
      acceptedModelName,acceptedModelFile,cfg,logger,rand,maxConcurrentEvals,
      NNPos::MAX_BOARD_LEN,NNPos::MAX_BOARD_LEN,defaultMaxBatchSize,
      Setup::SETUP_FOR_OTHER
    );
    logger.write("Loaded accepted neural net " + acceptedModelName + " from: " + acceptedModelFile);

    string sgfOutputDirThisModel = sgfOutputDir + "/" + testModelName;
    MakeDir::make(sgfOutputDirThisModel);
    {
      ofstream out(sgfOutputDirThisModel + "/" + "gatekeeper-" + Global::uint64ToHexString(rand.nextUInt64()) + ".cfg");
      out << cfg.getContents();
      out.close();
    }

    ofstream* sgfOut = sgfOutputDirThisModel.length() > 0 ? (new ofstream(sgfOutputDirThisModel + "/" + Global::uint64ToHexString(rand.nextUInt64()) + ".sgfs")) : NULL;
    NetAndStuff* newNet = new NetAndStuff(cfg, acceptedModelName, testModelName, testModelDir, acceptedNNEval, testNNEval, sgfOut);

    //Check for unused config keys
    cfg.warnUnusedKeys(cerr,&logger);

    return newNet;
  };

  auto gameLoop = [
    &gameRunner,
    &logger,
    &netAndStuffMutex,&netAndStuff
  ](int threadIdx) {
    std::unique_lock<std::mutex> lock(netAndStuffMutex);
    netAndStuff->registerGameThread();
    logger.write("Game loop thread " + Global::intToString(threadIdx) + " starting game testing candidate: " + netAndStuff->modelNameCandidate);

    vector<std::atomic<bool>*> stopConditions = {&shouldStop,&(netAndStuff->terminated)};

    while(true) {
      if(shouldStop.load() || netAndStuff->terminated.load())
        break;

      lock.unlock();

      FinishedGameData* gameData = NULL;

      int64_t gameIdx;
      MatchPairer::BotSpec botSpecB;
      MatchPairer::BotSpec botSpecW;
      if(netAndStuff->matchPairer->getMatchup(gameIdx, botSpecB, botSpecW, logger)) {
        gameData = gameRunner->runGame(
          gameIdx, botSpecB, botSpecW, NULL, logger,
          stopConditions, NULL
        );
      }

      bool shouldContinue = gameData != NULL;
      if(gameData != NULL)
        netAndStuff->finishedGameQueue.waitPush(gameData);

      lock.lock();

      if(!shouldContinue)
        break;
    }

    netAndStuff->unregisterGameThread();

    lock.unlock();
    logger.write("Game loop thread " + Global::intToString(threadIdx) + " terminating");
  };

  //Looping polling for new neural nets and loading them in
  while(true) {
    if(shouldStop.load())
      break;

    assert(netAndStuff == NULL);
    netAndStuff = loadLatestNeuralNet();

    if(netAndStuff == NULL) {
      if(quitIfNoNetsToTest) {
        shouldStop.store(true);
      }
      else {
        for(int i = 0; i<4; i++) {
          std::this_thread::sleep_for(std::chrono::seconds(1));
          if(shouldStop.load())
            break;
        }
      }
      continue;
    }

    //Check again if we should be stopping, after loading the new net, and quit more quickly.
    if(shouldStop.load()) {
      delete netAndStuff;
      netAndStuff = NULL;
      break;
    }

    //Otherwise, we're not stopped yet, so let's proceeed. Initialize stuff...
    netAndStuffDataIsWritten = false;

    //And spawn off all the threads
    std::thread newThread(dataWriteLoop);
    newThread.detach();
    vector<std::thread> threads;
    for(int i = 0; i<numGameThreads; i++) {
      threads.push_back(std::thread(gameLoop,i));
    }

    //Wait for all game threads to stop
    for(int i = 0; i<threads.size(); i++)
      threads[i].join();

    //Wait for the data to all be written
    {
      std::unique_lock<std::mutex> lock(netAndStuffMutex);

      //Mark as draining so the data write thread will quit
      netAndStuff->markAsDraining();

      while(!netAndStuffDataIsWritten) {
        waitNetAndStuffDataIsWritten.wait(lock);
      }
    }

    //Don't do anything if the reason we quit was due to signal
    if(shouldStop.load()) {
      delete netAndStuff;
      netAndStuff = NULL;
      break;
    }

    //Candidate wins ties
    if(netAndStuff->numBaselineWinPoints > netAndStuff->numCandidateWinPoints + 1e-10) {
      logger.write(
        Global::strprintf(
          "Candidate lost match, score %.3f to %.3f in %d games, rejecting candidate %s",
          netAndStuff->numCandidateWinPoints,
          netAndStuff->numBaselineWinPoints,
          netAndStuff->numGamesTallied,
          netAndStuff->modelNameCandidate.c_str()
        )
      );

      string renameDest = rejectedModelsDir + "/" + netAndStuff->modelNameCandidate;
      logger.write("Moving " + netAndStuff->testModelDir + " to " + renameDest);
      std::rename(netAndStuff->testModelDir.c_str(),renameDest.c_str());
    }
    else {
      logger.write(
        Global::strprintf(
          "Candidate won match, score %.3f to %.3f in %d games, accepting candidate %s",
          netAndStuff->numCandidateWinPoints,
          netAndStuff->numBaselineWinPoints,
          netAndStuff->numGamesTallied,
          netAndStuff->modelNameCandidate.c_str()
        )
      );

      string renameDest = acceptedModelsDir + "/" + netAndStuff->modelNameCandidate;
      logger.write("Moving " + netAndStuff->testModelDir + " to " + renameDest);
      std::rename(netAndStuff->testModelDir.c_str(),renameDest.c_str());
    }

    //Clean up
    delete netAndStuff;
    netAndStuff = NULL;
    //Loop again after a short while
    std::this_thread::sleep_for(std::chrono::seconds(2));
  }

  //Delete and clean up everything else
  NeuralNet::globalCleanup();
  delete gameRunner;
  ScoreValue::freeTables();

  if(sigReceived.load())
    logger.write("Exited cleanly after signal");
  logger.write("All cleaned up, quitting");
  return 0;
}
