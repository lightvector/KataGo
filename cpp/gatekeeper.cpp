#include "core/global.h"
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
#include "program/gitinfo.h"
#include "main.h"

using namespace std;

#define TCLAP_NAMESTARTSTRING "-" //Use single dashes for all flags
#include <tclap/CmdLine.h>

#include <boost/filesystem.hpp>

#include <chrono>

#include <csignal>
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
namespace {
  struct NetAndStuff {
    string modelName;
    NNEvaluator* nnEvalCandidate;
    NNEvaluator* nnEvalBaseline;

    ThreadSafeQueue<FinishedGameData*> finishedGameQueue;
    int numGameThreads;
    bool isDraining;

    ofstream* sgfOut;

  public:
    NetAndStuff(const string& name, NNEvaluator* nevalC, NNEvaluator* nevalB, ofstream* sOut)
      :modelName(name),
       nnEvalCandidate(nevalC),
       nnEvalBaseline(nevalB),
       finishedGameQueue(),
       numGameThreads(0),
       isDraining(false),
       sgfOut(sOut)
    {}

    ~NetAndStuff() {
      delete nnEvalCandidate;
      delete nnEvalBaseline;
      if(sgfOut != NULL)
        delete sgfOut;
    }

    void runWriteDataLoop() {
      while(true) {
        FinishedGameData* data = finishedGameQueue.waitPop();
        if(data == NULL)
          break;

        if(sgfOut != NULL) {
          assert(data->startHist.moveHistory.size() <= data->endHist.moveHistory.size());
          WriteSgf::writeSgf(*sgfOut,modelName,modelName,data->startHist.rules,data->preStartBoard,data->endHist,data);
          (*sgfOut) << endl;
        }
        delete data;
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
      if(isDraining && numGameThreads <= 0)
        finishedGameQueue.forcePush(NULL); //forcePush so as not to block
    }

    //NOT threadsafe - needs to be externally synchronized
    //Mark that we should start draining this net and not starting new games with it
    void markAsDraining() {
      if(!isDraining) {
        isDraining = true;
        if(numGameThreads <= 0)
          finishedGameQueue.forcePush(NULL); //forcePush so as not to block
      }
    }

  };
}


//-----------------------------------------------------------------------------------------


int MainCmds::gatekeeper(int argc, const char* const* argv) {
  Board::initHash();
  Rand seedRand;

  string configFile;
  int inputsVersion;
  string testModelsDir;
  string acceptedModelsDir;
  string rejectedModelsDir;
  string sgfOutputDir;
  try {
    TCLAP::CmdLine cmd("Generate training data via self play", ' ', "1.0",true);
    TCLAP::ValueArg<string> configFileArg("","config-file","Config file to use",true,string(),"FILE");
    TCLAP::ValueArg<int>    inputsVersionArg("","inputs-version","Version of neural net input features to use for data",true,0,"INT");
    TCLAP::ValueArg<string> testModelsDirArg("","test-models-dir","Dir to poll and load models from",true,string(),"DIR");
    TCLAP::ValueArg<string> sgfOutputDirArg("","sgf-output-dir","Dir to output sgf files",true,string(),"DIR");
    TCLAP::ValueArg<string> acceptedModelsDirArg("","accepted-models-dir","Dir to write good models to",true,string(),"DIR");
    TCLAP::ValueArg<string> rejectedModelsDirArg("","rejected-models-dir","Dir to write bad models to",true,string(),"DIR");
    cmd.add(configFileArg);
    cmd.add(inputsVersionArg);
    cmd.add(testModelsDirArg);
    cmd.add(sgfOutputDirArg);
    cmd.add(acceptedModelsDirArg);
    cmd.add(rejectedModelsDirArg);
    cmd.parse(argc,argv);
    configFile = configFileArg.getValue();
    inputsVersion = inputsVersionArg.getValue();
    testModelsDir = testModelsDirArg.getValue();
    sgfOutputDir = sgfOutputDirArg.getValue();
    acceptedModelsDir = acceptedModelsDirArg.getValue();
    rejectedModelsDir = rejectedModelsDirArg.getValue();
  }
  catch (TCLAP::ArgException &e) {
    cerr << "Error: " << e.error() << " for argument " << e.argId() << endl;
    return 1;
  }
  ConfigParser cfg(configFile);

  MakeDir::make(rejectedModelsDir);
  MakeDir::make(sgfOutputDir);

  Logger logger;
  //Log to random file name to better support starting/stopping as well as multiple parallel runs
  logger.addFile(sgfOutputDir + "/log" + Global::uint64ToHexString(seedRand.nextUInt64()) + ".log");
  bool logToStdout = cfg.getBool("logToStdout");
  logger.setLogToStdout(logToStdout);

  logger.write("Gatekeeper Engine starting...");
  logger.write(string("Git revision: ") + GIT_REVISION);

  //Load runner settings
  const int numGameThreads = cfg.getInt("numGameThreads",1,16384);
  const string searchRandSeedBase = Global::uint64ToHexString(seedRand.nextUInt64());

  //Shouldn't matter actually, we need it because we're sharing code with selfPlay, but for our purposes
  //all this needs to be is no smaller than the actual board sizes tested on.
  const int dataPosLen = cfg.getInt("dataPosLen",9,37);

  GameRunner* gameRunner = new GameRunner(cfg, searchRandSeedBase);

  Setup::initializeSession(cfg);

  //Done loading!
  //------------------------------------------------------------------------------------
  logger.write("Loaded all config stuff, starting play");
  if(!logToStdout)
    cout << "Loaded all config stuff, starting play" << endl;

  if(!std::atomic_is_lock_free(&shouldStop))
    throw StringError("shouldStop is not lock free, signal-quitting mechanism for terminating matches will NOT work!");
  std::signal(SIGINT, signalHandler);
  std::signal(SIGTERM, signalHandler);

  std::mutex netAndStuffsMutex;
  vector<NetAndStuff*> netAndStuffs;
  std::condition_variable netAndStuffsIsEmpty;

  //Looping thread for writing data for a single net
  auto dataWriteLoop = [&netAndStuffsMutex,&netAndStuffs,&netAndStuffsIsEmpty,&logger](NetAndStuff* netAndStuff) {
    logger.write("Data write loop starting for neural net: " + netAndStuff->modelName);
    netAndStuff->runWriteDataLoop();
    logger.write("Data write loop finishing for neural net: " + netAndStuff->modelName);

    std::unique_lock<std::mutex> lock(netAndStuffsMutex);

    //Find where our netAndStuff is and remove it
    string name = netAndStuff->modelName;
    bool found = false;
    for(int i = 0; i<netAndStuffs.size(); i++) {
      if(netAndStuffs[i] == netAndStuff) {
        netAndStuffs.erase(netAndStuffs.begin()+i);
        assert(netAndStuff->numGameThreads == 0);
        assert(netAndStuff->isDraining);
        delete netAndStuff;
        found = true;
        break;
      }
    }
    assert(found);
    if(netAndStuffs.size() == 0)
      netAndStuffsIsEmpty.notify_all();

    lock.unlock();
    logger.write("Data write loop cleaned up and terminating for " + name);
  };

  auto loadLatestNeuralNet =
    [inputsVersion,&testModelsDir,&acceptedModelsDir,&sgfOutputDir,&logger,&cfg](const string* lastNetName) -> NetAndStuff* {
    Rand rand;

    string testModelName;
    string testModelFile;
    string testModelDir;
    bool foundModel = LoadModel::findLatestModel(testModelsDir, logger, testModelName, testModelFile, testModelDir);

    //No new neural nets yet
    if(!foundModel || (lastNetName != NULL && *lastNetName == testModelName))
      return NULL;

    logger.write("Found new candidate neural net " + testModelName);

    bool debugSkipNeuralNetDefaultTest = (testModelFile == "/dev/null");

    NNEvaluator* testNNEval;
    {
      vector<NNEvaluator*> nnEvals = Setup::initializeNNEvaluators({testModelFile},cfg,logger,rand,debugSkipNeuralNetDefaultTest);
      assert(nnEvals.size() == 1);
      logger.write("Loaded candidate neural net " + testModelName + " from: " + testModelFile);
      testNNEval = nnEvals[0];
    }

    string acceptedModelName;
    string acceptedModelFile;
    string acceptedModelDir;
    foundModel = LoadModel::findLatestModel(acceptedModelsDir, logger, acceptedModelName, acceptedModelFile, acceptedModelDir);
    if(!foundModel)
      logger.write("Error: No accepted model found in " + acceptedModelsDir);

    bool debugSkipNeuralNetDefaultAccepted = (acceptedModelFile == "/dev/null");

    NNEvaluator* acceptedNNEval;
    {
      vector<NNEvaluator*> nnEvals = Setup::initializeNNEvaluators({acceptedModelFile},cfg,logger,rand,debugSkipNeuralNetDefaultAccepted);
      assert(nnEvals.size() == 1);
      logger.write("Loaded accepted neural net " + acceptedModelName + " from: " + acceptedModelFile);
      acceptedNNEval = nnEvals[0];
    }

    string sgfOutputDirThisModel = sgfOutputDir + "/" + testModelName;
    assert(sgfOutputDir != string());

    MakeDir::make(sgfOutputDirThisModel);

    //Note that this inputsVersion passed here is NOT necessarily the same as the one used in the neural net self play, it
    //simply controls the input feature version for the written data
    ofstream* sgfOut = sgfOutputDirThisModel.length() > 0 ? (new ofstream(sgfOutputDirThisModel + "/" + Global::uint64ToHexString(rand.nextUInt64()) + ".sgfs")) : NULL;
    NetAndStuff* newNet = new NetAndStuff(testModelName, testNNEval, acceptedNNEval, sgfOut);

    //Check for unused config keys
    {
      vector<string> unusedKeys = cfg.unusedKeys();
      for(size_t i = 0; i<unusedKeys.size(); i++) {
        string msg = "WARNING: Unused key '" + unusedKeys[i] + "' in " + cfg.getFileName();
        logger.write(msg);
        cerr << msg << endl;
      }
    }

    return newNet;
  };

  auto gameLoop = [
    &gameRunner,
    &logger,
    &netAndStuffsMutex,&netAndStuffs,
    dataPosLen
  ](int threadIdx) {
    std::unique_lock<std::mutex> lock(netAndStuffsMutex);
    string prevModelName;
    while(true) {
      if(shouldStop.load())
        break;

      assert(netAndStuffs.size() > 0);
      NetAndStuff* netAndStuff = netAndStuffs[netAndStuffs.size()-1];
      netAndStuff->registerGameThread();

      lock.unlock();

      if(prevModelName != netAndStuff->modelName) {
        prevModelName = netAndStuff->modelName;
        logger.write("Game loop thread " + Global::intToString(threadIdx) + " starting game on new candidate neural net: " + prevModelName);
      }

      //TODO randomize between who is black and white? Right now it's fixed. Or does the gameinitializer already take care of that? How did we do that for Match?
      bool shouldContinue = gameRunner->runGameAndEnqueueData(
        netAndStuff->nnEvalCandidate, netAndStuff->nnEvalBaseline, logger,
        dataPosLen, netAndStuff->finishedGameQueue,
        shouldStop
      );

      lock.lock();

      netAndStuff->unregisterGameThread();

      if(!shouldContinue)
        break;
    }

    lock.unlock();
    logger.write("Game loop thread " + Global::intToString(threadIdx) + " terminating");
  };

  //TODO continue from here


}
