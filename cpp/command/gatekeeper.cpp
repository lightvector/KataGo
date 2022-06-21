#include "../core/global.h"
#include "../core/datetime.h"
#include "../core/fileutils.h"
#include "../core/makedir.h"
#include "../core/config_parser.h"
#include "../core/timer.h"
#include "../core/threadsafequeue.h"
#include "../dataio/sgf.h"
#include "../dataio/trainingwrite.h"
#include "../dataio/loadmodel.h"
#include "../search/asyncbot.h"
#include "../program/setup.h"
#include "../program/play.h"
#include "../command/commandline.h"
#include "../main.h"

#include <sstream>

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

    string testModelFile;
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
    NetAndStuff(
      ConfigParser& cfg,
      const string& nameB,
      const string& nameC,
      const string& tModelFile,
      const string& tModelDir,
      NNEvaluator* nevalB,
      NNEvaluator* nevalC,
      ofstream* sOut
    )
      :modelNameBaseline(nameB),
       modelNameCandidate(nameC),
       nnEvalBaseline(nevalB),
       nnEvalCandidate(nevalC),
       matchPairer(NULL),
       testModelFile(tModelFile),
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
      SearchParams baseParams = Setup::loadSingleParams(cfg,Setup::SETUP_FOR_OTHER);

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
          WriteSgf::writeSgf(*sgfOut,data->bName,data->wName,data->endHist,data,false,true);
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

static void moveModel(const string& modelName, const string& modelFile, const string& modelDir, const string& testModelsDir, const string& intoDir, Logger& logger) {
  // Was the rejected model rooted in the testModels dir itself?
  if(FileUtils::weaklyCanonical(modelDir) == FileUtils::weaklyCanonical(testModelsDir)) {
    string renameDest = intoDir + "/" + modelName;
    logger.write("Moving " + modelFile + " to " + renameDest);
    FileUtils::rename(modelFile,renameDest);
  }
  // Or was it contained in a subdirectory
  else if(Global::isPrefix(FileUtils::weaklyCanonical(modelDir), FileUtils::weaklyCanonical(testModelsDir))) {
    string renameDest = intoDir + "/" + modelName;
    logger.write("Moving " + modelDir + " to " + renameDest);
    FileUtils::rename(modelDir,renameDest);
  }
  else {
    throw StringError("Model " + modelDir + " does not appear to be a subdir of " + testModelsDir + " can't figure out where how to move it to accept or reject it");
  }
}


//-----------------------------------------------------------------------------------------


int MainCmds::gatekeeper(const vector<string>& args) {
  Board::initHash();
  ScoreValue::initTables();
  Rand seedRand;

  ConfigParser cfg;
  string testModelsDir;
  string acceptedModelsDir;
  string rejectedModelsDir;
  string sgfOutputDir;
  string selfplayDir;
  bool noAutoRejectOldModels;
  bool quitIfNoNetsToTest;
  try {
    KataGoCommandLine cmd("Test neural nets to see if they should be accepted for self-play training data generation.");
    cmd.addConfigFileArg("","");

    TCLAP::ValueArg<string> testModelsDirArg("","test-models-dir","Dir to poll and load models from",true,string(),"DIR");
    TCLAP::ValueArg<string> sgfOutputDirArg("","sgf-output-dir","Dir to output sgf files",true,string(),"DIR");
    TCLAP::ValueArg<string> acceptedModelsDirArg("","accepted-models-dir","Dir to write good models to",true,string(),"DIR");
    TCLAP::ValueArg<string> rejectedModelsDirArg("","rejected-models-dir","Dir to write bad models to",true,string(),"DIR");
    TCLAP::ValueArg<string> selfplayDirArg("","selfplay-dir","Dir where selfplay data will be produced if a model passes",false,string(),"DIR");
    TCLAP::SwitchArg noAutoRejectOldModelsArg("","no-autoreject-old-models","Test older models than the latest accepted model");
    TCLAP::SwitchArg quitIfNoNetsToTestArg("","quit-if-no-nets-to-test","Terminate instead of waiting for a new net to test");
    cmd.add(testModelsDirArg);
    cmd.add(sgfOutputDirArg);
    cmd.add(acceptedModelsDirArg);
    cmd.add(rejectedModelsDirArg);
    cmd.add(selfplayDirArg);
    cmd.setShortUsageArgLimit();
    cmd.add(noAutoRejectOldModelsArg);
    cmd.add(quitIfNoNetsToTestArg);
    cmd.parseArgs(args);

    testModelsDir = testModelsDirArg.getValue();
    sgfOutputDir = sgfOutputDirArg.getValue();
    acceptedModelsDir = acceptedModelsDirArg.getValue();
    rejectedModelsDir = rejectedModelsDirArg.getValue();
    selfplayDir = selfplayDirArg.getValue();
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

    //Tolerate this argument being optional
    //checkDirNonEmpty("selfplay-dir",selfplayDir);

    cmd.getConfig(cfg);
  }
  catch (TCLAP::ArgException &e) {
    cerr << "Error: " << e.error() << " for argument " << e.argId() << endl;
    return 1;
  }

  MakeDir::make(testModelsDir);
  MakeDir::make(acceptedModelsDir);
  MakeDir::make(rejectedModelsDir);
  MakeDir::make(sgfOutputDir);
  if(selfplayDir != "")
    MakeDir::make(selfplayDir);

  Logger logger(&cfg);
  //Log to random file name to better support starting/stopping as well as multiple parallel runs
  logger.addFile(sgfOutputDir + "/log" + DateTime::getCompactDateTimeString() + "-" + Global::uint64ToHexString(seedRand.nextUInt64()) + ".log");

  logger.write("Gatekeeper Engine starting...");
  logger.write(string("Git revision: ") + Version::getGitRevision());

  //Load runner settings
  const int numGameThreads = cfg.getInt("numGameThreads",1,16384);
  const string gameSeedBase = Global::uint64ToHexString(seedRand.nextUInt64());

  PlaySettings playSettings = PlaySettings::loadForGatekeeper(cfg);
  GameRunner* gameRunner = new GameRunner(cfg, playSettings, logger);
  const int minBoardXSizeUsed = gameRunner->getGameInitializer()->getMinBoardXSize();
  const int minBoardYSizeUsed = gameRunner->getGameInitializer()->getMinBoardYSize();
  const int maxBoardXSizeUsed = gameRunner->getGameInitializer()->getMaxBoardXSize();
  const int maxBoardYSizeUsed = gameRunner->getGameInitializer()->getMaxBoardYSize();

  Setup::initializeSession(cfg);

  //Done loading!
  //------------------------------------------------------------------------------------
  logger.write("Loaded all config stuff, watching for new neural nets in " + testModelsDir);
  if(!logger.isLoggingToStdout())
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
  auto dataWriteLoopProtected = [&logger,&dataWriteLoop]() {
    Logger::logThreadUncaught("data write loop", &logger, dataWriteLoop);
  };

  auto loadLatestNeuralNet =
    [&testModelsDir,&rejectedModelsDir,&acceptedModelsDir,&sgfOutputDir,&logger,&cfg,numGameThreads,noAutoRejectOldModels,
     minBoardXSizeUsed,maxBoardXSizeUsed,minBoardYSizeUsed,maxBoardYSizeUsed]() -> NetAndStuff* {
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
      logger.write("Rejecting " + testModelName + " automatically since older than best accepted model");
      moveModel(testModelName, testModelFile, testModelDir, testModelsDir, rejectedModelsDir, logger);
      return NULL;
    }

    // * 2 + 16 just in case to have plenty of room
    const int maxConcurrentEvals = cfg.getInt("numSearchThreads") * numGameThreads * 2 + 16;
    const int expectedConcurrentEvals = cfg.getInt("numSearchThreads") * numGameThreads;
    const int defaultMaxBatchSize = -1;
    const bool defaultRequireExactNNLen = minBoardXSizeUsed == maxBoardXSizeUsed && minBoardYSizeUsed == maxBoardYSizeUsed;
    const string expectedSha256 = "";

    NNEvaluator* testNNEval = Setup::initializeNNEvaluator(
      testModelName,testModelFile,expectedSha256,cfg,logger,rand,maxConcurrentEvals,expectedConcurrentEvals,
      maxBoardXSizeUsed,maxBoardYSizeUsed,defaultMaxBatchSize,defaultRequireExactNNLen,
      Setup::SETUP_FOR_OTHER
    );
    logger.write("Loaded candidate neural net " + testModelName + " from: " + testModelFile);

    NNEvaluator* acceptedNNEval = Setup::initializeNNEvaluator(
      acceptedModelName,acceptedModelFile,expectedSha256,cfg,logger,rand,maxConcurrentEvals,expectedConcurrentEvals,
      maxBoardXSizeUsed,maxBoardYSizeUsed,defaultMaxBatchSize,defaultRequireExactNNLen,
      Setup::SETUP_FOR_OTHER
    );
    logger.write("Loaded accepted neural net " + acceptedModelName + " from: " + acceptedModelFile);

    string sgfOutputDirThisModel = sgfOutputDir + "/" + testModelName;
    MakeDir::make(sgfOutputDirThisModel);
    {
      ofstream out;
      FileUtils::open(out, sgfOutputDirThisModel + "/" + "gatekeeper-" + Global::uint64ToHexString(rand.nextUInt64()) + ".cfg");
      out << cfg.getContents();
      out.close();
    }

    ofstream* sgfOut = NULL;
    if(sgfOutputDirThisModel.length() > 0) {
      sgfOut = new ofstream();
      FileUtils::open(*sgfOut, sgfOutputDirThisModel + "/" + Global::uint64ToHexString(rand.nextUInt64()) + ".sgfs");
    }
    NetAndStuff* newNet = new NetAndStuff(
      cfg,
      acceptedModelName,
      testModelName,
      testModelFile,
      testModelDir,
      acceptedNNEval,
      testNNEval,
      sgfOut
    );

    //Check for unused config keys
    cfg.warnUnusedKeys(cerr,&logger);

    return newNet;
  };

  auto gameLoop = [
    &gameRunner,
    &logger,
    &netAndStuffMutex,
    &netAndStuff,
    &gameSeedBase
  ](int threadIdx) {
    std::unique_lock<std::mutex> lock(netAndStuffMutex);
    netAndStuff->registerGameThread();
    logger.write("Game loop thread " + Global::intToString(threadIdx) + " starting game testing candidate: " + netAndStuff->modelNameCandidate);

    auto shouldStopFunc = [&netAndStuff]() {
      return shouldStop.load() || netAndStuff->terminated.load();
    };
    WaitableFlag* shouldPause = nullptr;

    Rand thisLoopSeedRand;
    while(true) {
      if(shouldStopFunc())
        break;

      lock.unlock();

      FinishedGameData* gameData = NULL;

      MatchPairer::BotSpec botSpecB;
      MatchPairer::BotSpec botSpecW;
      if(netAndStuff->matchPairer->getMatchup(botSpecB, botSpecW, logger)) {
        string seed = gameSeedBase + ":" + Global::uint64ToHexString(thisLoopSeedRand.nextUInt64());
        gameData = gameRunner->runGame(
          seed, botSpecB, botSpecW, NULL, NULL, logger,
          shouldStopFunc, shouldPause, nullptr, nullptr, nullptr
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
  auto gameLoopProtected = [&logger,&gameLoop](int threadIdx) {
    Logger::logThreadUncaught("game loop", &logger, [&](){ gameLoop(threadIdx); });
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
    std::thread newThread(dataWriteLoopProtected);
    newThread.detach();
    vector<std::thread> threads;
    for(int i = 0; i<numGameThreads; i++) {
      threads.push_back(std::thread(gameLoopProtected,i));
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

      moveModel(
        netAndStuff->modelNameCandidate,
        netAndStuff->testModelFile,
        netAndStuff->testModelDir,
        testModelsDir,
        rejectedModelsDir,
        logger
      );
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

      //Make a bunch of the directories that selfplay will need so that there isn't a race on the selfplay
      //machines to concurrently make it, since sometimes concurrent making of the same directory can corrupt
      //a filesystem
      if(selfplayDir != "") {
        MakeDir::make(selfplayDir + "/" + netAndStuff->modelNameCandidate);
        std::this_thread::sleep_for(std::chrono::seconds(1));
        MakeDir::make(selfplayDir + "/" + netAndStuff->modelNameCandidate + "/" + "sgfs");
        MakeDir::make(selfplayDir + "/" + netAndStuff->modelNameCandidate + "/" + "tdata");
        MakeDir::make(selfplayDir + "/" + netAndStuff->modelNameCandidate + "/" + "vadata");
      }
      std::this_thread::sleep_for(std::chrono::seconds(2));

      moveModel(
        netAndStuff->modelNameCandidate,
        netAndStuff->testModelFile,
        netAndStuff->testModelDir,
        testModelsDir,
        acceptedModelsDir,
        logger
      );
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
