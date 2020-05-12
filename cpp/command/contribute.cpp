#include "core/global.h"
#include "core/config_parser.h"
#include "core/datetime.h"
#include "core/timer.h"
#include "core/makedir.h"
#include "dataio/loadmodel.h"
#include "neuralnet/modelversion.h"
#include "search/asyncbot.h"
#include "program/play.h"
#include "program/setup.h"
#include "program/selfplaymanager.h"
#include "commandline.h"
#include "main.h"

#ifndef BUILD_DISTRIBUTED

int MainCmds::contribute(int argc, const char* const* argv) {
  (void)argc;
  (void)argv;
  std::cout << "This version of KataGo was NOT compiled with support for distributed training." << std::endl;
  std::cout << "Compile with -DBUILD_DISTRIBUTED=1 in CMake, and/or see notes at https://github.com/lightvector/KataGo#compiling-katago" << std::endl;
  return 0;
}

#else

#include "distributed/client.h"

#include <sstream>
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

static const string defaultBaseDir = "katago_contribute";
static const int defaultMaxSimultaneousGames = 16;
static const int defaultUnloadUnusedModelsAfter = 60 * 60;
static const int defaultDeleteUnusedModelsAfter = 6 * 60 * 60;

namespace {
  struct GameTask {
    Client::Task task;
    SelfplayManager* manager;
    NNEvaluator* nnEvalBlack;
    NNEvaluator* nnEvalWhite;
  };
}

static void runAndUploadSingleGame(
  Client::Connection* connection, GameTask gameTask, int64_t gameIdx,
  Logger& logger, const string& seed, ForkData* forkData, string sgfsDir, Rand& rand
) {
  vector<std::atomic<bool>*> stopConditions = {&shouldStop};

  istringstream taskCfgIn(gameTask.task.config);
  ConfigParser taskCfg(taskCfgIn);

  NNEvaluator* nnEvalBlack = gameTask.nnEvalBlack;
  NNEvaluator* nnEvalWhite = gameTask.nnEvalWhite;

  SearchParams baseParams;
  PlaySettings playSettings;
  try {
    baseParams = Setup::loadSingleParams(taskCfg);
    playSettings = PlaySettings::loadForSelfplay(taskCfg);
  }
  catch(StringError& e) {
    cerr << "Error parsing task config" << endl;
    cerr << e.what() << endl;
    throw;
  }

  MatchPairer::BotSpec botSpecB;
  MatchPairer::BotSpec botSpecW;
  botSpecB.botIdx = 0;
  botSpecB.botName = nnEvalBlack->getModelName();
  botSpecB.nnEval = nnEvalBlack;
  botSpecB.baseParams = baseParams;
  if(nnEvalWhite == nnEvalBlack)
    botSpecW = botSpecB;
  else {
    botSpecW.botIdx = 1;
    botSpecW.botName = nnEvalWhite->getModelName();
    botSpecW.nnEval = nnEvalWhite;
    botSpecW.baseParams = baseParams;
  }

  GameRunner* gameRunner = new GameRunner(taskCfg, playSettings, logger);

  //Check for unused config keys
  taskCfg.warnUnusedKeys(cerr,&logger);

  FinishedGameData* gameData = gameRunner->runGame(
    seed, botSpecB, botSpecW, forkData, logger,
    stopConditions, NULL
  );

  if(gameData != NULL) {
    string sgfOutputDir;
    if(gameTask.task.isRatingGame)
      sgfOutputDir = sgfsDir + "/" + gameTask.task.taskGroup;
    else
      sgfOutputDir = sgfsDir + "/" + nnEvalBlack->getModelName();
    string sgfFile = sgfOutputDir + "/" + Global::uint64ToHexString(rand.nextUInt64()) + ".sgf";

    ofstream out(sgfFile);
    WriteSgf::writeSgf(out,gameData->bName,gameData->wName,gameData->endHist,gameData,false);
    out.close();

    const bool retryOnFailure = true;
    if(gameTask.task.doWriteTrainingData) {
      gameTask.manager->withDataWriters(
        nnEvalBlack,
        [gameData,&gameTask,gameIdx,&sgfFile,&connection,&logger](TrainingDataWriter* tdataWriter, TrainingDataWriter* vdataWriter, std::ofstream* sgfOut) {
          (void)vdataWriter;
          (void)sgfOut;
          tdataWriter->writeGame(*gameData);
          string resultingFilename;
          bool producedFile = tdataWriter->flushIfNonempty(resultingFilename);
          //It's possible we'll have zero data if the game started in a nearly finished position and cheap search never
          //gave us a real turn of search, in which case just ignore that game.
          if(producedFile) {
            bool suc = connection->uploadTrainingGameAndData(gameTask.task,gameData,sgfFile,resultingFilename,retryOnFailure,shouldStop);
            if(suc)
              logger.write("Finished game " + Global::int64ToString(gameIdx)  + " (training), uploaded sgf " + sgfFile + " and training data " + resultingFilename);
          }
          else {
            logger.write("Finished game " + Global::int64ToString(gameIdx) + " (training), skipping uploading sgf " + sgfFile + " since it's an empty game");
          }
        });
    }
    else {
      bool suc = connection->uploadRatingGame(gameTask.task,gameData,sgfFile,retryOnFailure,shouldStop);
      if(suc)
        logger.write("Finished game " + Global::int64ToString(gameIdx) + " (rating), uploaded sgf " + sgfFile);
    }
  }

  delete gameData;
  delete gameRunner;
}


int MainCmds::contribute(int argc, const char* const* argv) {
  Board::initHash();
  ScoreValue::initTables();
  Rand seedRand;

  string baseDir;
  int maxSimultaneousGames;
  int unloadUnusedModelsAfter;
  int deleteUnusedModelsAfter;
  string userConfigFile;
  string overrideUserConfig;
  try {
    KataGoCommandLine cmd("Run KataGo to generate training data for distributed training");
    TCLAP::ValueArg<string> baseDirArg(
      "","base-dir","Directory to download models, write game results, etc. (default ./katago_contribute)",
      false,defaultBaseDir,"DIR"
    );
    TCLAP::ValueArg<int> maxSimultaneousGamesArg(
      "","max-simultaneous-games","Number of games to play simultaneously (default "+ Global::intToString(defaultMaxSimultaneousGames)+")",
      false,defaultMaxSimultaneousGames,"NGAMES"
    );
    TCLAP::ValueArg<int> unloadUnusedModelsAfterArg(
      "","unload-unused-models-after","After a model is unused in memory for this many seconds, unload it (default "+ Global::intToString(defaultUnloadUnusedModelsAfter)+")",
      false,defaultUnloadUnusedModelsAfter,"SECONDS"
    );
    TCLAP::ValueArg<int> deleteUnusedModelsAfterArg(
      "","delete-unused-models-after","After a model is unused for this many seconds, delete it from disk (default "+ Global::intToString(defaultDeleteUnusedModelsAfter)+")",
      false,defaultDeleteUnusedModelsAfter,"SECONDS"
    );
    TCLAP::ValueArg<string> userConfigFileArg("","config","Config file to use for server connection and/or GPU settings",false,string(),"FILE");
    TCLAP::ValueArg<string> overrideUserConfigArg("","override-config","Override config parameters. Format: \"key=value, key=value,...\"",false,string(),"KEYVALUEPAIRS");
    cmd.add(baseDirArg);
    cmd.add(maxSimultaneousGamesArg);
    cmd.add(unloadUnusedModelsAfterArg);
    cmd.add(deleteUnusedModelsAfterArg);
    cmd.add(userConfigFileArg);
    cmd.add(overrideUserConfigArg);
    cmd.parse(argc,argv);
    baseDir = baseDirArg.getValue();
    maxSimultaneousGames = maxSimultaneousGamesArg.getValue();
    unloadUnusedModelsAfter = unloadUnusedModelsAfterArg.getValue();
    deleteUnusedModelsAfter = deleteUnusedModelsAfterArg.getValue();
    userConfigFile = userConfigFileArg.getValue();
    overrideUserConfig = overrideUserConfigArg.getValue();

    if(maxSimultaneousGames <= 0 || maxSimultaneousGames > 100000)
      throw StringError("-max-simultaneous-games: invalid value");
    if(unloadUnusedModelsAfter < 0 || unloadUnusedModelsAfter > 1000000000)
      throw StringError("-unload-unused-models-after: invalid value");
    if(deleteUnusedModelsAfter < 0 || deleteUnusedModelsAfter > 1000000000)
      throw StringError("-delete-unused-models-after: invalid value");
  }
  catch (TCLAP::ArgException &e) {
    cerr << "Error: " << e.error() << " for argument " << e.argId() << endl;
    return 1;
  }

  ConfigParser* userCfg;
  if(userConfigFile == "") {
    istringstream userCfgIn("");
    userCfg = new ConfigParser(userCfgIn);
  }
  else {
    userCfg = new ConfigParser(userConfigFile);
  }
  if(overrideUserConfig != "") {
    map<string,string> newkvs = ConfigParser::parseCommaSeparated(overrideUserConfig);
    //HACK to avoid a common possible conflict - if we specify some of the rules options on one side, the other side should be erased.
    vector<pair<set<string>,set<string>>> mutexKeySets = Setup::getMutexKeySets();
    userCfg->overrideKeys(newkvs,mutexKeySets);
  }

  if(!std::atomic_is_lock_free(&shouldStop))
    throw StringError("shouldStop is not lock free, signal-quitting mechanism for terminating matches will NOT work!");
  std::signal(SIGINT, signalHandler);
  std::signal(SIGTERM, signalHandler);

  Logger logger;
  logger.setLogToStdout(true);

  logger.write("Distributed Self Play Engine starting...");
  logger.write(Version::getKataGoVersionForHelp());
  logger.write(string("Git revision: ") + Version::getGitRevision());

  string serverUrl = userCfg->getString("serverUrl");
  string username = userCfg->getString("username");
  string password = userCfg->getString("password");

  //Connect to server and get global parameters for the run.
  Client::Connection* connection = new Client::Connection(serverUrl,username,password,&logger);
  const Client::RunParameters runParams = connection->getRunParameters();

  MakeDir::make(baseDir);
  baseDir = baseDir + "/" + runParams.runName;
  MakeDir::make(baseDir);

  const string modelsDir = baseDir + "/models";
  const string sgfsDir = baseDir + "/sgfs";
  const string tdataDir = baseDir + "/tdata";
  const string logsDir = baseDir + "/logs";

  MakeDir::make(modelsDir);
  MakeDir::make(sgfsDir);
  MakeDir::make(tdataDir);
  MakeDir::make(logsDir);

  //Log to random file name to better support starting/stopping as well as multiple parallel runs
  logger.addFile(logsDir + "/log" + DateTime::getCompactDateTimeString() + "-" + Global::uint64ToHexString(seedRand.nextUInt64()) + ".log");

  //Don't write "validation" data for distributed self-play. If the server-side wants to split out some data as "validation" for training
  //then that can be done server-side.
  const double validationProp = 0.0;
  //If we ever get more than this many games behind on writing data, something is weird.
  const int maxDataQueueSize = maxSimultaneousGames * 4;
  const int logGamesEvery = 1;

  const string gameSeedBase = Global::uint64ToHexString(seedRand.nextUInt64());

  Setup::initializeSession(*userCfg);

  //Shared across all game threads
  ThreadSafeQueue<GameTask> gameTaskQueue(1);
  ForkData* forkData = new ForkData();
  std::atomic<int64_t> numGamesStarted(0);

  auto runGameLoop = [&logger,forkData,&gameSeedBase,&gameTaskQueue,&numGamesStarted,&sgfsDir,&connection]() {
    Rand thisLoopSeedRand;
    while(true) {
      GameTask gameTask;
      bool success = gameTaskQueue.waitPop(gameTask);
      if(!success)
        break;
      if(!shouldStop.load()) {
        string seed = gameSeedBase + ":" + Global::uint64ToHexString(thisLoopSeedRand.nextUInt64());
        int64_t gameIdx = numGamesStarted.fetch_add(1,std::memory_order_acq_rel);
        logger.write(
          "Started game " + Global::int64ToString(gameIdx) + " (" + (
            gameTask.nnEvalBlack == gameTask.nnEvalWhite ?
            gameTask.nnEvalBlack->getModelName() :
            (gameTask.nnEvalBlack->getModelName() + " vs " + gameTask.nnEvalWhite->getModelName())
          ) + ")"
        );
        runAndUploadSingleGame(connection,gameTask,gameIdx,logger,seed,forkData,sgfsDir,thisLoopSeedRand);
      }
      gameTask.manager->release(gameTask.nnEvalBlack);
      gameTask.manager->release(gameTask.nnEvalWhite);
    }
  };

  auto loadNeuralNetIntoManager =
    [&runParams,&tdataDir,&sgfsDir,&logger,&userCfg,maxSimultaneousGames](
      SelfplayManager* manager, const string& modelName, const string& modelFile
    ) {

    if(manager->hasModel(modelName))
      return;

    logger.write("Found new neural net " + modelName);

    int maxConcurrentEvals = runParams.maxSearchThreadsAllowed * maxSimultaneousGames * 2 + 16;
    int defaultMaxBatchSize = maxSimultaneousGames;

    //Unlike local self-play, which waits to accumulate a fixed number of rows before writing, distributed selfplay writes
    //training data game by game. So we set a buffer size here large enough to always hold all the rows of a game.
    //These values should be vastly more than enough, yet still not use too much memory.
    double firstFileRandMinProp = 1.0;
    int maxRowsPerTrainFile = 20000;

    Rand rand;
    NNEvaluator* nnEval = Setup::initializeNNEvaluator(
      modelName,modelFile,*userCfg,logger,rand,maxConcurrentEvals,
      NNPos::MAX_BOARD_LEN,NNPos::MAX_BOARD_LEN,defaultMaxBatchSize,
      Setup::SETUP_FOR_DISTRIBUTED
    );
    assert(!nnEval->isNeuralNetLess() || modelFile == "/dev/null");
    logger.write("Loaded latest neural net " + modelName + " from: " + modelFile);

    string sgfOutputDir = sgfsDir + "/" + modelName;
    string tdataOutputDir = tdataDir + "/" + modelName;
    MakeDir::make(sgfOutputDir);
    MakeDir::make(tdataOutputDir);

    //Note that this inputsVersion passed here is NOT necessarily the same as the one used in the neural net self play, it
    //simply controls the input feature version for the written data
    const int inputsVersion = runParams.inputsVersion;
    const int dataBoardLen = runParams.dataBoardLen;
    TrainingDataWriter* tdataWriter = new TrainingDataWriter(
      tdataOutputDir, inputsVersion, maxRowsPerTrainFile, firstFileRandMinProp, dataBoardLen, dataBoardLen, Global::uint64ToHexString(rand.nextUInt64()));
    TrainingDataWriter* vdataWriter = NULL;
    ofstream* sgfOut = NULL;

    logger.write("Loaded new neural net " + nnEval->getModelName());
    manager->loadModelNoDataWritingLoop(nnEval, tdataWriter, vdataWriter, sgfOut);
  };

  //For distributed selfplay, we have a single thread primarily in charge of the manager, so we turn this off
  //to ensure there is no asynchronous removal of models.
  bool autoCleanupAllButLatestIfUnused = false;
  SelfplayManager* manager = new SelfplayManager(validationProp, maxDataQueueSize, &logger, logGamesEvery, autoCleanupAllButLatestIfUnused);

  //Start game loop threads! Yay!
  vector<std::thread> gameThreads;
  for(int i = 0; i<maxSimultaneousGames; i++) {
    gameThreads.push_back(std::thread(runGameLoop));
  }

  //Loop acquiring tasks and feeding them to game threads
  bool anyTaskSuccessfullyParsedYet = false;
  while(true) {
    std::this_thread::sleep_for(std::chrono::duration<double>(1.0));
    if(shouldStop.load())
      break;
    bool retryOnFailure = anyTaskSuccessfullyParsedYet;
    Client::Task task;
    bool suc = connection->getNextTask(task,baseDir,retryOnFailure,shouldStop);
    if(!suc)
      continue;

    if(task.runName != runParams.runName) {
      throw StringError(
        "This self-play client was started with the run \"" + task.runName +
        "\" but the server now appears to be hosting a new run \"" + runParams.runName +
        "\", you may need to re-start this client."
      );
    }

    if(task.isRatingGame) {
      string sgfOutputDir = sgfsDir + "/" + task.taskGroup;
      MakeDir::make(sgfOutputDir);
    }

    bool suc1 = connection->downloadModelIfNotPresent(task.modelBlack,modelsDir,retryOnFailure,shouldStop);
    bool suc2 = connection->downloadModelIfNotPresent(task.modelWhite,modelsDir,retryOnFailure,shouldStop);
    if(shouldStop.load())
      break;
    if(!suc1 || !suc2)
      continue;

    //Update model file modified times as a way to track which ones we've used recently or not
    string modelFileBlack = Client::Connection::getModelPath(task.modelBlack,modelsDir);
    string modelFileWhite = Client::Connection::getModelPath(task.modelWhite,modelsDir);
    if(!task.modelBlack.isRandom) {
      LoadModel::setLastModifiedTimeToNow(modelFileBlack,logger);
    }
    if(!task.modelWhite.isRandom && task.modelWhite.name != task.modelBlack.name) {
      LoadModel::setLastModifiedTimeToNow(modelFileWhite,logger);
    }

    loadNeuralNetIntoManager(manager,task.modelBlack.name,modelFileBlack);
    loadNeuralNetIntoManager(manager,task.modelWhite.name,modelFileWhite);
    if(shouldStop.load())
      break;

    //Game loop threads are responsible for releasing when done.
    NNEvaluator* nnEvalBlack = manager->acquireModel(task.modelBlack.name);
    NNEvaluator* nnEvalWhite = manager->acquireModel(task.modelWhite.name);

    manager->cleanupUnusedModelsOlderThan(unloadUnusedModelsAfter);
    time_t modelFileAgeLimit = time(NULL) - deleteUnusedModelsAfter;
    LoadModel::deleteModelsOlderThan(modelsDir,logger,modelFileAgeLimit);

    GameTask gameTask;
    gameTask.task = task;
    gameTask.manager = manager;
    gameTask.nnEvalBlack = nnEvalBlack;
    gameTask.nnEvalWhite = nnEvalWhite;

    anyTaskSuccessfullyParsedYet = true;
    suc = gameTaskQueue.waitPush(gameTask);
    (void)suc;
    assert(suc);
  }

  //This should trigger game threads to quit
  gameTaskQueue.setReadOnly();

  //Wait for all game threads to stop
  for(int i = 0; i<gameThreads.size(); i++)
    gameThreads[i].join();

  //At this point, nothing else except possibly data write loops are running, within the selfplay manager.
  delete manager;

  //Now we can close the connection since all data is written
  delete connection;

  //Delete and clean up everything else
  NeuralNet::globalCleanup();
  delete forkData;
  ScoreValue::freeTables();
  delete userCfg;

  if(sigReceived.load())
    logger.write("Exited cleanly after signal");
  logger.write("All cleaned up, quitting");
  return 0;
}

#endif //BUILD_DISTRIBUTED
