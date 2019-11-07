#include "core/global.h"
#include "core/datetime.h"
#include "core/makedir.h"
#include "core/config_parser.h"
#include "core/timer.h"
#include "core/threadsafequeue.h"
#include "dataio/sgf.h"
#include "dataio/trainingwrite.h"
#include "dataio/loadmodel.h"
#include "neuralnet/modelversion.h"
#include "search/asyncbot.h"
#include "program/setup.h"
#include "program/play.h"
#include "main.h"

#define TCLAP_NAMESTARTSTRING "-" //Use single dashes for all flags
#include <tclap/CmdLine.h>

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
//There should be one of these active per currently-loaded neural net, and one active thread
//looping and actually performing the data output
//DOES take ownership of the NNEvaluators
namespace {
  struct NetAndStuff {
    string modelName;
    NNEvaluator* nnEval;
    MatchPairer* matchPairer;
    double validationProp;

    int maxDataQueueSize;
    ThreadSafeQueue<FinishedGameData*> finishedGameQueue;
    int numGameThreads;
    bool isDraining;

    std::condition_variable noGameThreadsLeftVar;

    TrainingDataWriter* tdataWriter;
    TrainingDataWriter* vdataWriter;
    ofstream* sgfOut;
    Rand rand;

  public:
    NetAndStuff(
      ConfigParser& cfg, const string& name, NNEvaluator* neval, int maxDQueueSize,
      TrainingDataWriter* tdWriter, TrainingDataWriter* vdWriter, ofstream* sOut, double vProp
    )
      :modelName(name),
       nnEval(neval),
       matchPairer(NULL),
       validationProp(vProp),
       maxDataQueueSize(maxDQueueSize),
       finishedGameQueue(maxDQueueSize),
       numGameThreads(0),isDraining(false),
       tdataWriter(tdWriter),
       vdataWriter(vdWriter),
       sgfOut(sOut),
       rand()
    {
      SearchParams baseParams = Setup::loadSingleParams(cfg);

      //Initialize object for randomly pairing bots. Actually since this is only selfplay, this only
      //ever gives is the trivial self-pairing, but we use it also for keeping the game count and some logging.
      bool forSelfPlay = true;
      bool forGateKeeper = false;
      matchPairer = new MatchPairer(cfg, 1, {modelName}, {nnEval}, {baseParams}, forSelfPlay, forGateKeeper);
    }

    ~NetAndStuff() {
      delete matchPairer;
      delete nnEval;
      delete tdataWriter;
      delete vdataWriter;
      if(sgfOut != NULL)
        delete sgfOut;
    }

    void runWriteDataLoop(Logger& logger) {
      while(true) {
        size_t size = finishedGameQueue.size();
        if(size > maxDataQueueSize / 2)
          logger.write(Global::strprintf("WARNING: Struggling to keep up writing data, %d games enqueued out of %d max",size,maxDataQueueSize));

        FinishedGameData* data;
        bool suc = finishedGameQueue.waitPop(data);
        if(!suc)
          break;

        assert(data != NULL);

        if(rand.nextBool(validationProp))
          vdataWriter->writeGame(*data);
        else
          tdataWriter->writeGame(*data);

        if(sgfOut != NULL) {
          assert(data->startHist.moveHistory.size() <= data->endHist.moveHistory.size());
          WriteSgf::writeSgf(*sgfOut,data->bName,data->wName,data->endHist,data);
          (*sgfOut) << endl;
        }
        delete data;
      }

      tdataWriter->flushIfNonempty();
      vdataWriter->flushIfNonempty();
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
      if(numGameThreads <= 0)
        noGameThreadsLeftVar.notify_all();
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



int MainCmds::selfplay(int argc, const char* const* argv) {
  Board::initHash();
  ScoreValue::initTables();
  Rand seedRand;

  string configFile;
  string modelsDir;
  string outputDir;
  try {
    TCLAP::CmdLine cmd("Generate training data via self play", ' ', Version::getKataGoVersionForHelp(),true);
    TCLAP::ValueArg<string> configFileArg("","config-file","Config file to use",true,string(),"FILE");
    TCLAP::ValueArg<string> modelsDirArg("","models-dir","Dir to poll and load models from",true,string(),"DIR");
    TCLAP::ValueArg<string> outputDirArg("","output-dir","Dir to output files",true,string(),"DIR");
    cmd.add(configFileArg);
    cmd.add(modelsDirArg);
    cmd.add(outputDirArg);
    cmd.parse(argc,argv);
    configFile = configFileArg.getValue();
    modelsDir = modelsDirArg.getValue();
    outputDir = outputDirArg.getValue();

    auto checkDirNonEmpty = [](const char* flag, const string& s) {
      if(s.length() <= 0)
        throw StringError("Empty directory specified for " + string(flag));
    };
    checkDirNonEmpty("models-dir",modelsDir);
    checkDirNonEmpty("output-dir",outputDir);
  }
  catch (TCLAP::ArgException &e) {
    cerr << "Error: " << e.error() << " for argument " << e.argId() << endl;
    return 1;
  }
  ConfigParser cfg(configFile);

  MakeDir::make(outputDir);
  MakeDir::make(modelsDir);

  Logger logger;
  //Log to random file name to better support starting/stopping as well as multiple parallel runs
  logger.addFile(outputDir + "/log" + DateTime::getCompactDateTimeString() + "-" + Global::uint64ToHexString(seedRand.nextUInt64()) + ".log");
  bool logToStdout = cfg.getBool("logToStdout");
  logger.setLogToStdout(logToStdout);

  logger.write("Self Play Engine starting...");
  logger.write(string("Git revision: ") + Version::getGitRevision());

  //Load runner settings
  const int numGameThreads = cfg.getInt("numGameThreads",1,16384);
  const string searchRandSeedBase = Global::uint64ToHexString(seedRand.nextUInt64());

  //Width and height of the board to use when writing data, typically 19
  const int dataBoardLen = cfg.getInt("dataBoardLen",9,37);
  const int inputsVersion =
    cfg.contains("inputsVersion") ?
    cfg.getInt("inputsVersion",0,10000) :
    NNModelVersion::getInputsVersion(NNModelVersion::defaultModelVersion);
  //Max number of games that we will allow to be queued up and not written out
  const int maxDataQueueSize = cfg.getInt("maxDataQueueSize",1,1000000);
  const int maxRowsPerTrainFile = cfg.getInt("maxRowsPerTrainFile",1,100000000);
  const int maxRowsPerValFile = cfg.getInt("maxRowsPerValFile",1,100000000);
  const double firstFileRandMinProp = cfg.getDouble("firstFileRandMinProp",0.0,1.0);

  const double validationProp = cfg.getDouble("validationProp",0.0,0.5);

  const bool switchNetsMidGame = cfg.getBool("switchNetsMidGame");

  //Initialize object for randomizing game settings and running games
  FancyModes fancyModes;
  fancyModes.initGamesWithPolicy = cfg.getBool("initGamesWithPolicy");
  fancyModes.forkSidePositionProb = cfg.getDouble("forkSidePositionProb",0.0,1.0);

  fancyModes.noCompensateKomiProb = cfg.getDouble("noCompensateKomiProb",0.0,1.0);
  fancyModes.compensateKomiVisits = 20;

  fancyModes.earlyForkGameProb = cfg.getDouble("earlyForkGameProb",0.0,0.5);
  fancyModes.earlyForkGameExpectedMoveProp = cfg.getDouble("earlyForkGameExpectedMoveProp",0.0,1.0);
  fancyModes.earlyForkGameMinChoices = cfg.getInt("earlyForkGameMinChoices",1,30);
  fancyModes.earlyForkGameMaxChoices = cfg.getInt("earlyForkGameMaxChoices",1,30);
  fancyModes.cheapSearchProb = cfg.getDouble("cheapSearchProb",0.0,1.0);
  fancyModes.cheapSearchVisits = cfg.getInt("cheapSearchVisits",1,10000000);
  fancyModes.cheapSearchTargetWeight = cfg.getFloat("cheapSearchTargetWeight",0.0f,1.0f);
  fancyModes.reduceVisits = cfg.getBool("reduceVisits");
  fancyModes.reduceVisitsThreshold = cfg.getDouble("reduceVisitsThreshold",0.0,0.999999);
  fancyModes.reduceVisitsThresholdLookback = cfg.getInt("reduceVisitsThresholdLookback",0,1000);
  fancyModes.reducedVisitsMin = cfg.getInt("reducedVisitsMin",1,10000000);
  fancyModes.reducedVisitsWeight = cfg.getFloat("reducedVisitsWeight",0.0f,1.0f);
  fancyModes.policySurpriseDataWeight = cfg.getDouble("policySurpriseDataWeight",0.0f,1.0f);
  fancyModes.recordTreePositions = cfg.getBool("recordTreePositions");
  fancyModes.recordTreeThreshold = cfg.getInt("recordTreeThreshold",1,100000000);
  fancyModes.recordTreeTargetWeight = cfg.getFloat("recordTreeTargetWeight",0.0f,1.0f);
  fancyModes.forSelfPlay = true;
  fancyModes.dataXLen = dataBoardLen;
  fancyModes.dataYLen = dataBoardLen;
  GameRunner* gameRunner = new GameRunner(cfg, searchRandSeedBase, fancyModes);

  Setup::initializeSession(cfg);

  //Done loading!
  //------------------------------------------------------------------------------------
  logger.write("Loaded all config stuff, starting self play");
  if(!logToStdout)
    cout << "Loaded all config stuff, starting self play" << endl;

  if(!std::atomic_is_lock_free(&shouldStop))
    throw StringError("shouldStop is not lock free, signal-quitting mechanism for terminating matches will NOT work!");
  std::signal(SIGINT, signalHandler);
  std::signal(SIGTERM, signalHandler);

  std::mutex netAndStuffsMutex;
  vector<NetAndStuff*> netAndStuffs;
  int numDataWriteLoopsActive = 0;
  std::condition_variable dataWriteLoopsAreDone;

  //Looping thread for writing data for a single net
  auto dataWriteLoop = [&netAndStuffsMutex,&netAndStuffs,&numDataWriteLoopsActive,&dataWriteLoopsAreDone,&logger](NetAndStuff* netAndStuff) {
    logger.write("Data write loop starting for neural net: " + netAndStuff->modelName);
    netAndStuff->runWriteDataLoop(logger);
    logger.write("Data write loop finishing for neural net: " + netAndStuff->modelName);

    std::unique_lock<std::mutex> lock(netAndStuffsMutex);

    //Wait for all threads to be completely done with it
    while(netAndStuff->numGameThreads > 0)
      netAndStuff->noGameThreadsLeftVar.wait(lock);

    //Find where our netAndStuff is and remove it
    string name = netAndStuff->modelName;
    bool found = false;
    for(int i = 0; i<netAndStuffs.size(); i++) {
      if(netAndStuffs[i] == netAndStuff) {
        netAndStuffs.erase(netAndStuffs.begin()+i);
        found = true;
        break;
      }
    }
    assert(found);
    (void)found; //Avoid warning when asserts are disabled
    lock.unlock();

    //Do logging and cleanup while unlocked, so that our freeing and stopping of this neural net doesn't
    //block anyone else
    logger.write(netAndStuff->nnEval->getModelFileName());
    logger.write("NN rows: " + Global::int64ToString(netAndStuff->nnEval->numRowsProcessed()));
    logger.write("NN batches: " + Global::int64ToString(netAndStuff->nnEval->numBatchesProcessed()));
    logger.write("NN avg batch size: " + Global::doubleToString(netAndStuff->nnEval->averageProcessedBatchSize()));

    assert(netAndStuff->numGameThreads == 0);
    assert(netAndStuff->isDraining);
    delete netAndStuff;

    logger.write("Data write loop cleaned up and terminating for " + name);

    //Check back in and notify that we're done once done cleaning up.
    lock.lock();
    numDataWriteLoopsActive--;
    assert(numDataWriteLoopsActive >= 0);
    if(numDataWriteLoopsActive == 0) {
      assert(netAndStuffs.size() == 0);
      dataWriteLoopsAreDone.notify_all();
    }
    lock.unlock();
  };

  auto loadLatestNeuralNet =
    [inputsVersion,maxDataQueueSize,maxRowsPerTrainFile,maxRowsPerValFile,firstFileRandMinProp,dataBoardLen,
     &modelsDir,&outputDir,&logger,&cfg,validationProp,numGameThreads](const string* lastNetName) -> NetAndStuff* {

    string modelName;
    string modelFile;
    string modelDir;
    time_t modelTime;
    bool foundModel = LoadModel::findLatestModel(modelsDir, logger, modelName, modelFile, modelDir, modelTime);

    //No new neural nets yet
    if(!foundModel || (lastNetName != NULL && *lastNetName == modelName))
      return NULL;

    logger.write("Found new neural net " + modelName);

    // * 2 + 16 just in case to have plenty of room
    int maxConcurrentEvals = cfg.getInt("numSearchThreads") * numGameThreads * 2 + 16;

    Rand rand;
    NNEvaluator* nnEval = Setup::initializeNNEvaluator(
      modelName,modelFile,cfg,logger,rand,maxConcurrentEvals,NNPos::MAX_BOARD_LEN,NNPos::MAX_BOARD_LEN
    );
    logger.write("Loaded latest neural net " + modelName + " from: " + modelFile);

    string modelOutputDir = outputDir + "/" + modelName;
    string sgfOutputDir = modelOutputDir + "/sgfs";
    string tdataOutputDir = modelOutputDir + "/tdata";
    string vdataOutputDir = modelOutputDir + "/vdata";

    //Try repeatedly to make directories, in case the filesystem is unhappy with us as we try to make the same dirs as another process.
    //Wait a random amount of time in between each failure.
    int maxTries = 5;
    for(int i = 0; i<maxTries; i++) {
      bool success = false;
      try {
        MakeDir::make(modelOutputDir);
        MakeDir::make(sgfOutputDir);
        MakeDir::make(tdataOutputDir);
        MakeDir::make(vdataOutputDir);
        success = true;
      }
      catch(const StringError& e) {
        logger.write(string("WARNING, error making directories, trying again shortly: ") + e.what());
        success = false;
      }

      if(success)
        break;
      else {
        if(i == maxTries-1) {
          logger.write("ERROR: Could not make selfplay model directories, is something wrong with the filesystem?");
          //Just give up and wait for the next model.
          return NULL;
        }
        double sleepTime = 10.0 + rand.nextDouble() * 30.0;
        std::this_thread::sleep_for(std::chrono::duration<double>(sleepTime));
        continue;
      }
    }

    {
      ofstream out(modelOutputDir + "/" + "selfplay-" + Global::uint64ToHexString(rand.nextUInt64()) + ".cfg");
      out << cfg.getContents();
      out.close();
    }

    //Note that this inputsVersion passed here is NOT necessarily the same as the one used in the neural net self play, it
    //simply controls the input feature version for the written data
    TrainingDataWriter* tdataWriter = new TrainingDataWriter(
      tdataOutputDir, inputsVersion, maxRowsPerTrainFile, firstFileRandMinProp, dataBoardLen, dataBoardLen, Global::uint64ToHexString(rand.nextUInt64()));
    TrainingDataWriter* vdataWriter = new TrainingDataWriter(
      vdataOutputDir, inputsVersion, maxRowsPerValFile, firstFileRandMinProp, dataBoardLen, dataBoardLen, Global::uint64ToHexString(rand.nextUInt64()));
    ofstream* sgfOut = sgfOutputDir.length() > 0 ? (new ofstream(sgfOutputDir + "/" + Global::uint64ToHexString(rand.nextUInt64()) + ".sgfs")) : NULL;
    NetAndStuff* newNet = new NetAndStuff(cfg, modelName, nnEval, maxDataQueueSize, tdataWriter, vdataWriter, sgfOut, validationProp);
    return newNet;
  };

  //Initialize the initial neural net
  {
    NetAndStuff* newNet = loadLatestNeuralNet(NULL);
    assert(newNet != NULL);

    std::unique_lock<std::mutex> lock(netAndStuffsMutex);
    netAndStuffs.push_back(newNet);
    numDataWriteLoopsActive++;
    std::thread newThread(dataWriteLoop,newNet);
    newThread.detach();
  }

  //Check for unused config keys
  cfg.warnUnusedKeys(cerr,&logger);

  auto gameLoop = [
    &gameRunner,
    &logger,
    &netAndStuffsMutex,&netAndStuffs,
    switchNetsMidGame,
    &fancyModes
  ](int threadIdx) {
    vector<std::atomic<bool>*> stopConditions = {&shouldStop};

    std::unique_lock<std::mutex> lock(netAndStuffsMutex);
    string prevModelName;
    const InitialPosition* nextInitialPosition = NULL;
    while(true) {
      if(shouldStop.load())
        break;

      assert(netAndStuffs.size() > 0);
      NetAndStuff* netAndStuff = netAndStuffs[netAndStuffs.size()-1];
      netAndStuff->registerGameThread();

      lock.unlock();

      if(prevModelName != netAndStuff->modelName) {
        prevModelName = netAndStuff->modelName;
        logger.write("Game loop thread " + Global::intToString(threadIdx) + " starting game on new neural net: " + prevModelName);
      }

      //Callback that runGame will call periodically to ask us if we have a new neural net
      std::function<NNEvaluator*()> checkForNewNNEval = [&netAndStuff,&netAndStuffs,&lock,&prevModelName,&logger,&threadIdx]() -> NNEvaluator* {
        lock.lock();
        assert(netAndStuffs.size() > 0);
        NetAndStuff* newNetAndStuff = netAndStuffs[netAndStuffs.size()-1];
        //Do nothing if there is no new net... or if there IS a new net but actually that net is draining for whatever reason
        //(in the latter case, most likely everyone is quitting out right now, and we will notice and quit soon too)
        if(newNetAndStuff == netAndStuff || newNetAndStuff->isDraining) {
          lock.unlock();
          return NULL;
        }
        netAndStuff->unregisterGameThread();
        newNetAndStuff->registerGameThread();
        lock.unlock();

        netAndStuff = newNetAndStuff;
        prevModelName = netAndStuff->modelName;
        logger.write("Game loop thread " + Global::intToString(threadIdx) + " changing midgame to new neural net: " + prevModelName);
        return netAndStuff->nnEval;
      };

      FinishedGameData* gameData = NULL;

      const InitialPosition* initialPosition = nextInitialPosition;
      nextInitialPosition = NULL;

      int64_t gameIdx;
      MatchPairer::BotSpec botSpecB;
      MatchPairer::BotSpec botSpecW;
      if(netAndStuff->matchPairer->getMatchup(gameIdx, botSpecB, botSpecW, logger)) {
        gameData = gameRunner->runGame(
          gameIdx, botSpecB, botSpecW, initialPosition, &nextInitialPosition, logger,
          stopConditions,
          (switchNetsMidGame ? &checkForNewNNEval : NULL)
        );
      }

      if(initialPosition != NULL)
        delete initialPosition;

      bool shouldContinue = gameData != NULL;
      //Note that if we've gotten a newNNEval, we're actually pushing the game on to the new one's queue, rather than the old one!
      if(gameData != NULL)
        netAndStuff->finishedGameQueue.waitPush(gameData);

      lock.lock();

      netAndStuff->unregisterGameThread();

      if(!shouldContinue)
        break;
    }

    lock.unlock();

    if(nextInitialPosition != NULL)
      delete nextInitialPosition;

    logger.write("Game loop thread " + Global::intToString(threadIdx) + " terminating");
  };

  //Looping thread for polling for new neural nets and loading them in
  std::condition_variable modelLoadSleepVar;
  auto modelLoadLoop = [&netAndStuffsMutex,&netAndStuffs,&numDataWriteLoopsActive,&modelLoadSleepVar,&logger,&dataWriteLoop,&loadLatestNeuralNet]() {
    logger.write("Model loading loop thread starting");

    string lastNetName;
    std::unique_lock<std::mutex> lock(netAndStuffsMutex);
    while(true) {
      if(shouldStop.load())
        break;
      if(netAndStuffs.size() <= 0) {
        logger.write("Model loop thread UNEXPECTEDLY found 0 netAndStuffs... terminating now..?");
        break;
      }

      lastNetName = netAndStuffs[netAndStuffs.size()-1]->modelName;

      lock.unlock();

      NetAndStuff* newNet = loadLatestNeuralNet(&lastNetName);

      lock.lock();

      //Check again if we should be stopping, after loading the new net, and quit more quickly.
      if(shouldStop.load()) {
        lock.unlock();
        delete newNet;
        lock.lock();
        break;
      }

      //Otherwise, we're not stopped yet, so stick a new net on to things.
      if(newNet != NULL) {
        logger.write("Model loading loop thread loaded new neural net " + newNet->modelName);
        netAndStuffs.push_back(newNet);
        for(int i = 0; i<netAndStuffs.size()-1; i++) {
          netAndStuffs[i]->markAsDraining();
        }
        numDataWriteLoopsActive++;
        std::thread newThread(dataWriteLoop,newNet);
        newThread.detach();
      }

      //Sleep for a while and then re-poll
      modelLoadSleepVar.wait_for(lock, std::chrono::seconds(20), [](){return shouldStop.load();});
    }

    //As part of cleanup, anything remaining, mark them as draining so that if they also have
    //no more game threads, they all quit out.
    for(int i = 0; i<netAndStuffs.size(); i++)
      netAndStuffs[i]->markAsDraining();

    lock.unlock();
    logger.write("Model loading loop thread terminating");
  };


  vector<std::thread> threads;
  for(int i = 0; i<numGameThreads; i++) {
    threads.push_back(std::thread(gameLoop,i));
  }
  std::thread modelLoadLoopThread(modelLoadLoop);

  //Wait for all game threads to stop
  for(int i = 0; i<threads.size(); i++)
    threads[i].join();

  //Wake up the model loading thread rather than waiting up to 60s for it to wake up on its own, and
  //wait for it to die.
  {
    //Lock so that we don't race where we notify the loading thread to wake when it's still in
    //its own critical section but not yet slept
    std::lock_guard<std::mutex> lock(netAndStuffsMutex);
    //If by now somehow shouldStop is not true, set it to be true since all game threads are toast
    shouldStop.store(true);
    modelLoadSleepVar.notify_all();
  }
  modelLoadLoopThread.join();

  //At this point, nothing else except possibly data write loops are running, so there can't
  //be anything that will spawn any *more* data writing loops.

  //Wait for all data write loops to finish cleaning up.
  {
    std::unique_lock<std::mutex> lock(netAndStuffsMutex);
    while(numDataWriteLoopsActive > 0) {
      dataWriteLoopsAreDone.wait(lock);
    }
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
