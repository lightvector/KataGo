#include "../core/global.h"
#include "../core/config_parser.h"
#include "../core/datetime.h"
#include "../core/timer.h"
#include "../core/makedir.h"
#include "../core/os.h"
#include "../core/prioritymutex.h"
#include "../dataio/loadmodel.h"
#include "../dataio/homedata.h"
#include "../external/nlohmann_json/json.hpp"
#include "../neuralnet/modelversion.h"
#include "../search/asyncbot.h"
#include "../program/play.h"
#include "../program/setup.h"
#include "../program/selfplaymanager.h"
#include "../tests/tinymodel.h"
#include "../command/commandline.h"
#include "../main.h"

#ifndef BUILD_DISTRIBUTED

int MainCmds::contribute(int argc, const char* const* argv) {
  (void)argc;
  (void)argv;
  std::cout << "This version of KataGo was NOT compiled with support for distributed training." << std::endl;
  std::cout << "Compile with -DBUILD_DISTRIBUTED=1 in CMake, and/or see notes at https://github.com/lightvector/KataGo#compiling-katago" << std::endl;
  return 0;
}

#else

#include "../distributed/client.h"

#ifdef OS_IS_WINDOWS
#include <stdio.h>
#include <fileapi.h>
#endif

#include <sstream>
#include <chrono>
#include <csignal>
using json = nlohmann::json;

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

// Some OSes, like windows, don't have SIGPIPE
#ifdef SIGPIPE
static std::atomic<int> sigPipeReceivedCount(0);
static void sigPipeHandler(int signal)
{
  if(signal == SIGPIPE) {
    sigPipeReceivedCount.fetch_add(1);
  }
}

static void sigPipeHandlerDoNothing(int signal)
{
  (void)signal;
}
#endif

//-----------------------------------------------------------------------------------------

static const string defaultBaseDir = "katago_contribute";
static const double defaultDeleteUnusedModelsAfterDays = 30;

namespace {
  struct GameTask {
    Client::Task task;
    int repIdx; //0 to taskRepFactor-1
    SelfplayManager* blackManager;
    SelfplayManager* whiteManager;
    NNEvaluator* nnEvalBlack;
    NNEvaluator* nnEvalWhite;
  };
}

static void runAndUploadSingleGame(
  Client::Connection* connection, GameTask gameTask, int64_t gameIdx,
  Logger& logger, const string& seed, ForkData* forkData, string sgfsDir, Rand& rand,
  std::atomic<int64_t>& numMovesPlayed,
  std::unique_ptr<ostream>& outputEachMove, std::function<void()> flushOutputEachMove,
  bool logGamesAsJson, bool alwaysIncludeOwnership
) {
  if(gameTask.task.isRatingGame) {
    logger.write(
      "Starting game " + Global::int64ToString(gameIdx) + " (rating) (" + (
        (gameTask.nnEvalBlack->getModelName() + " vs " + gameTask.nnEvalWhite->getModelName())
      ) + ")"
    );
  }
  else {
    logger.write(
      "Starting game " + Global::int64ToString(gameIdx) + " (training) (" + (
        gameTask.nnEvalBlack == gameTask.nnEvalWhite ?
        gameTask.nnEvalBlack->getModelName() :
        (gameTask.nnEvalBlack->getModelName() + " vs " + gameTask.nnEvalWhite->getModelName())
      ) + ")"
    );
  }

  vector<std::atomic<bool>*> stopConditions = {&shouldStop};

  istringstream taskCfgIn(gameTask.task.config);
  ConfigParser taskCfg(taskCfgIn);

  NNEvaluator* nnEvalBlack = gameTask.nnEvalBlack;
  NNEvaluator* nnEvalWhite = gameTask.nnEvalWhite;

  SearchParams baseParams;
  PlaySettings playSettings;
  try {
    baseParams = Setup::loadSingleParams(taskCfg,Setup::SETUP_FOR_DISTRIBUTED);
    if(gameTask.task.isRatingGame)
      playSettings = PlaySettings::loadForGatekeeper(taskCfg);
    else
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

  //Make sure not to fork games in the middle for rating games!
  if(gameTask.task.isRatingGame)
    forkData = NULL;

  const string gameIdString = Global::uint64ToHexString(rand.nextUInt64());

  std::function<void(const Board&, const BoardHistory&, Player, Loc, const std::vector<double>&, const std::vector<double>&, const std::vector<double>&, const Search*)>
    onEachMove = [&numMovesPlayed, &outputEachMove, &flushOutputEachMove, &logGamesAsJson, &alwaysIncludeOwnership, &gameIdString, &botSpecB, &botSpecW](
      const Board& board, const BoardHistory& hist, Player pla, Loc moveLoc,
      const std::vector<double>& winLossHist, const std::vector<double>& leadHist, const std::vector<double>& scoreStdevHist, const Search* search) {
    numMovesPlayed.fetch_add(1,std::memory_order_relaxed);
    if(outputEachMove != nullptr) {
      ostringstream out;
      Board::printBoard(out, board, moveLoc, &(hist.moveHistory));
      if(botSpecB.botName == botSpecW.botName) {
        out << "Network: " << botSpecB.botName << "\n";
      }
      else {
        out << "Match: " << botSpecB.botName << " (black) vs " << botSpecW.botName << " (white)" << "\n";
      }
      out << "Rules: " << hist.rules.toJsonString() << "\n";
      out << "Player: " << PlayerIO::playerToString(pla) << "\n";
      out << "Move: " << Location::toString(moveLoc,board) << "\n";
      out << "Num Visits: " << search->getRootVisits() << "\n";
      if(winLossHist.size() > 0)
        out << "Black Winrate: " << 100.0*(0.5*(1.0 - winLossHist[winLossHist.size()-1])) << "%\n";
      if(leadHist.size() > 0)
        out << "Black Lead: " << -leadHist[leadHist.size()-1] << "\n";
      (void)scoreStdevHist;
      (void)search;
      out << "\n";
      (*outputEachMove) << out.str() << std::flush;
      if(flushOutputEachMove)
        flushOutputEachMove();
    }

    if(logGamesAsJson and hist.encorePhase == 0) { // If anyone wants to support encorePhase > 0 note passForKo is a thing
      int analysisPVLen = 15;
      const Player perspective = P_BLACK;
      bool preventEncore = true;
      static constexpr int ownershipMinVisits = 3;

      // output format is a mix between an analysis query and response
      json ret;
      // unique to this output
      ret["gameId"] = gameIdString;
      ret["move"] = json::array({PlayerIO::playerToStringShort(pla), Location::toString(moveLoc, board)});
      ret["blackPlayer"] = botSpecB.botName;
      ret["whitePlayer"] = botSpecW.botName;

      // Usual query fields
      ret["rules"] = hist.rules.toJson();
      ret["boardXSize"] = board.x_size;
      ret["boardYSize"] = board.y_size;

      json moves = json::array();
      for(auto move: hist.moveHistory) {
        moves.push_back(json::array({PlayerIO::playerToStringShort(move.pla), Location::toString(move.loc, board)}));
      }
      ret["moves"] = moves;

      json initialStones = json::array();
      const Board& initialBoard = hist.initialBoard;
      for(int y = 0; y < initialBoard.y_size; y++) {
        for(int x = 0; x < initialBoard.x_size; x++) {
          Loc loc = Location::getLoc(x, y, initialBoard.x_size);
          Player locOwner = initialBoard.colors[loc];
          if(locOwner != C_EMPTY)
            initialStones.push_back(json::array({PlayerIO::playerToStringShort(locOwner), Location::toString(loc, initialBoard)}));
        }
      }
      ret["initialStones"] = initialStones;
      ret["initialPlayer"] = PlayerIO::playerToStringShort(hist.initialPla);
      ret["initialTurnNumber"] = hist.initialTurnNumber;

      // Usual analysis response fields
      ret["turnNumber"] = hist.moveHistory.size();
      search->getAnalysisJson(perspective,board,hist,analysisPVLen,ownershipMinVisits,preventEncore,true,alwaysIncludeOwnership,false,false,ret);
      std::cout << ret.dump() + "\n" << std::flush; // no endl due to race conditions
    }

  };

  const Sgf::PositionSample* posSample = gameTask.repIdx < gameTask.task.startPoses.size() ? &(gameTask.task.startPoses[gameTask.repIdx]) : NULL;
  FinishedGameData* gameData = gameRunner->runGame(
    seed, botSpecB, botSpecW, forkData, posSample,
    logger, stopConditions, nullptr, onEachMove, alwaysIncludeOwnership
  );

  if(gameData != NULL) {
    string sgfOutputDir;
    if(gameTask.task.isRatingGame)
      sgfOutputDir = sgfsDir + "/" + gameTask.task.taskGroup;
    else
      sgfOutputDir = sgfsDir + "/" + nnEvalBlack->getModelName();
    string sgfFile = sgfOutputDir + "/" + gameIdString + ".sgf";

    ofstream out(sgfFile);
    WriteSgf::writeSgf(out,gameData->bName,gameData->wName,gameData->endHist,gameData,false,true);
    out.close();
    if(outputEachMove != nullptr) {
      (*outputEachMove) << "Game finished, sgf is " << sgfFile << endl;
      if(flushOutputEachMove)
        flushOutputEachMove();
    }

    static constexpr bool retryOnFailure = true;
    if(gameTask.task.doWriteTrainingData) {
      gameTask.blackManager->withDataWriters(
        nnEvalBlack,
        [gameData,&gameTask,gameIdx,&sgfFile,&connection,&logger](TrainingDataWriter* tdataWriter, TrainingDataWriter* vdataWriter, std::ofstream* sgfOut) {
          (void)vdataWriter;
          (void)sgfOut;
          assert(tdataWriter->isEmpty());
          tdataWriter->writeGame(*gameData);
          string resultingFilename;
          int64_t numDataRows = tdataWriter->numRowsInBuffer();
          bool producedFile = tdataWriter->flushIfNonempty(resultingFilename);
          //It's possible we'll have zero data if the game started in a nearly finished position and cheap search never
          //gave us a real turn of search, in which case just ignore that game.
          if(producedFile) {
            bool suc = false;
            try {
              suc = connection->uploadTrainingGameAndData(gameTask.task,gameData,sgfFile,resultingFilename,numDataRows,retryOnFailure,shouldStop);
            }
            catch(StringError& e) {
              logger.write(string("Giving up uploading training game and data due to error:\n") + e.what());
              suc = false;
            }
            if(suc)
              logger.write(
                "Finished game " + Global::int64ToString(gameIdx)  + " (training), uploaded sgf " + sgfFile + " and training data " + resultingFilename
                + " (" + Global::int64ToString(numDataRows) + " rows)"
              );
          }
          else {
            logger.write("Finished game " + Global::int64ToString(gameIdx) + " (training), skipping uploading sgf " + sgfFile + " since it's an empty game");
          }
        });
    }
    else {
      bool suc = false;
      try {
        suc = connection->uploadRatingGame(gameTask.task,gameData,sgfFile,retryOnFailure,shouldStop);
      }
      catch(StringError& e) {
        logger.write(string("Giving up uploading rating game due to error:\n") + e.what());
        suc = false;
      }
      if(suc)
        logger.write("Finished game " + Global::int64ToString(gameIdx) + " (rating), uploaded sgf " + sgfFile);
    }
  }
  else {
    logger.write("Terminating game " + Global::int64ToString(gameIdx));
  }

  delete gameData;
  delete gameRunner;
}


int MainCmds::contribute(int argc, const char* const* argv) {
  Board::initHash();
  ScoreValue::initTables();
  Rand seedRand;

  string baseDir;
  double deleteUnusedModelsAfterDays;
  string userConfigFile;
  string overrideUserConfig;
  string caCertsFile;
  try {
    KataGoCommandLine cmd("Run KataGo to generate training data for distributed training");
    TCLAP::ValueArg<string> baseDirArg(
      "","base-dir","Directory to download models, write game results, etc. (default ./katago_contribute)",
      false,defaultBaseDir,"DIR"
    );
    TCLAP::ValueArg<double> deleteUnusedModelsAfterDaysArg(
      "","delete-unused-models-after","After a model is unused for this many days, delete it from disk (default "+ Global::doubleToString(defaultDeleteUnusedModelsAfterDays)+")",
      false,defaultDeleteUnusedModelsAfterDays,"DAYS"
    );
    TCLAP::ValueArg<string> userConfigFileArg("","config","Config file to use for server connection and/or GPU settings",false,string(),"FILE");
    TCLAP::ValueArg<string> overrideUserConfigArg("","override-config","Override config parameters. Format: \"key=value, key=value,...\"",false,string(),"KEYVALUEPAIRS");
    TCLAP::ValueArg<string> caCertsFileArg("","cacerts","CA certificates file for SSL (cacerts.pem, ca-bundle.crt)",false,string(),"FILE");
    cmd.add(baseDirArg);
    cmd.add(deleteUnusedModelsAfterDaysArg);
    cmd.add(userConfigFileArg);
    cmd.add(overrideUserConfigArg);
    cmd.add(caCertsFileArg);
    cmd.parse(argc,argv);
    baseDir = baseDirArg.getValue();
    deleteUnusedModelsAfterDays = deleteUnusedModelsAfterDaysArg.getValue();
    userConfigFile = userConfigFileArg.getValue();
    overrideUserConfig = overrideUserConfigArg.getValue();
    caCertsFile = caCertsFileArg.getValue();

    if(!std::isfinite(deleteUnusedModelsAfterDays) || deleteUnusedModelsAfterDays < 0 || deleteUnusedModelsAfterDays > 20000)
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

  if(caCertsFile == "") {
    vector<string> defaultFilesDirs = HomeData::getDefaultFilesDirs();
    vector<string> cacertSearchDirs = defaultFilesDirs;

    //Also look for some system locations
#ifdef OS_IS_UNIX_OR_APPLE
    cacertSearchDirs.push_back("/etc/ssl");
    cacertSearchDirs.push_back("/etc/ssl/certs");
    cacertSearchDirs.push_back("/etc/pki/ca-trust/extracted/pem");
    cacertSearchDirs.push_back("/etc/pki/tls");
    cacertSearchDirs.push_back("/etc/pki/tls/certs");
    cacertSearchDirs.push_back("/etc/certs");
#endif

    vector<string> possiblePaths;
    for(const string& dir: cacertSearchDirs) {
      possiblePaths.push_back(dir + "/cacert.pem");
      possiblePaths.push_back(dir + "/cacert.crt");
      possiblePaths.push_back(dir + "/ca-bundle.pem");
      possiblePaths.push_back(dir + "/ca-bundle.crt");
      possiblePaths.push_back(dir + "/ca-certificates.crt");
      possiblePaths.push_back(dir + "/cert.pem");
      possiblePaths.push_back(dir + "/tls-ca-bundle.pem");
    }
    //In case someone's trying to run katago right out of the compiled github repo
    for(const string& dir: defaultFilesDirs) {
      possiblePaths.push_back(dir + "/external/mozilla-cacerts/cacert.pem");
    }

    bool foundCaCerts = false;
    for(const string& path: possiblePaths) {
      std::ifstream infile(path);
      bool pathExists = infile.good();
      if(pathExists) {
        foundCaCerts = true;
        caCertsFile = path;
        break;
      }
    }
    if(!foundCaCerts) {
      throw StringError(
        "Could not find CA certs (cacert.pem or ca-bundle.crt) at default location " +
        HomeData::getDefaultFilesDirForHelpMessage() +
        " or other default locations, please specify where this file is via '-cacerts' command " +
        " line argument and/or download them from https://curl.haxx.se/docs/caextract.html"
      );
    }
  }
  else {
    if(caCertsFile != "/dev/null") {
      std::ifstream infile(caCertsFile);
      bool pathExists = infile.good();
      if(!pathExists) {
        throw StringError("cacerts file was not found or could not be opened: " + caCertsFile);
      }
    }
  }

  Logger logger;
  logger.setLogToStdout(true);

  logger.write("Distributed Self Play Engine starting...");

  string serverUrl = userCfg->getString("serverUrl");
  string username = userCfg->getString("username");
  string password = userCfg->getString("password");
  string proxyHost = "";
  int proxyPort = 0;
  if(userCfg->contains("proxyHost")) {
    proxyHost = userCfg->getString("proxyHost");
    proxyPort = userCfg->getInt("proxyPort");
  }

  int maxSimultaneousGames;
  if(!userCfg->contains("maxSimultaneousGames")) {
    logger.write("maxSimultaneousGames was NOT specified in config, defaulting to 16");
    maxSimultaneousGames = 16;
  }
  else {
    maxSimultaneousGames = userCfg->getInt("maxSimultaneousGames", 1, 4000);
  }
  bool onlyPlayRatingMatches = false;
  if(userCfg->contains("onlyPlayRatingMatches")) {
    onlyPlayRatingMatches = userCfg->getBool("onlyPlayRatingMatches");
    logger.write("Setting onlyPlayRatingMatches to " + Global::boolToString(onlyPlayRatingMatches));
  }

  int maxRatingMatches;
  if(onlyPlayRatingMatches) {
    maxRatingMatches = 100000000;
  }
  else if(!userCfg->contains("maxRatingMatches")) {
    maxRatingMatches = 1;
  }
  else {
    maxRatingMatches = userCfg->getInt("maxRatingMatches", 0, 100000);
    logger.write("Setting maxRatingMatches to " + Global::intToString(maxRatingMatches));
  }
  bool disablePredownloadLoop = false;
  if(userCfg->contains("disablePredownloadLoop")) {
    disablePredownloadLoop = userCfg->getBool("disablePredownloadLoop");
    logger.write("Setting disablePredownloadLoop to " + Global::boolToString(disablePredownloadLoop));
  }
  string modelDownloadMirrorBaseUrl;
  bool mirrorUseProxy = true;
  if(userCfg->contains("modelDownloadMirrorBaseUrl")) {
    modelDownloadMirrorBaseUrl = userCfg->getString("modelDownloadMirrorBaseUrl");
    logger.write("Setting modelDownloadMirrorBaseUrl to " + modelDownloadMirrorBaseUrl);
    if(userCfg->contains("mirrorUseProxy")) {
      mirrorUseProxy = userCfg->getBool("mirrorUseProxy");
      logger.write("Setting mirrorUseProxy to " + Global::boolToString(mirrorUseProxy));
    }
  }

  //Play selfplay games and rating games in chunks of this many at a time. Each server query
  //gets fanned out into this many games. Having this value be larger helps ensure batching for
  //rating games (since we will have multiple games sharing the same network) while also reducing
  //query load on the server. It shouldn't be too large though, so as to remain responsive to the
  //changes in the next best network to selfplay or rate from the server.
  int taskRepFactor;
  if(!userCfg->contains("taskRepFactor")) {
    taskRepFactor = 4;
  }
  else {
    taskRepFactor = userCfg->getInt("taskRepFactor", 2, 16);
  }

  const double reportPerformanceEvery = userCfg->contains("reportPerformanceEvery") ? userCfg->getDouble("reportPerformanceEvery", 1, 21600) : 120;
  const bool watchOngoingGameInFile = userCfg->contains("watchOngoingGameInFile") ? userCfg->getBool("watchOngoingGameInFile") : false;
  string watchOngoingGameInFileName = userCfg->contains("watchOngoingGameInFileName") ? userCfg->getString("watchOngoingGameInFileName") : "";
  const bool logGamesAsJson = userCfg->contains("logGamesAsJson") ? userCfg->getBool("logGamesAsJson") : false;
  const bool alwaysIncludeOwnership = userCfg->contains("includeOwnership") ? userCfg->getBool("includeOwnership") : false;
  if(watchOngoingGameInFileName == "")
    watchOngoingGameInFileName = "watchgame.txt";

  //Connect to server and get global parameters for the run.
  Client::Connection* connection = new Client::Connection(
    serverUrl,username,password,caCertsFile,
    proxyHost,proxyPort,
    modelDownloadMirrorBaseUrl,
    mirrorUseProxy,
    &logger
  );
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

  //Write out versions now that the logger is all set up
  logger.write(Version::getKataGoVersionForHelp());
  logger.write(string("Git revision: ") + Version::getGitRevision());

  TinyModelTest::runTinyModelTest(baseDir, logger, *userCfg);

  //Set up signal handlers
  if(!std::atomic_is_lock_free(&shouldStop))
    throw StringError("shouldStop is not lock free, signal-quitting mechanism for terminating matches will NOT work!");
  std::signal(SIGINT, signalHandler);
  std::signal(SIGTERM, signalHandler);

#ifdef SIGPIPE
  //We want to make sure sigpipe doesn't kill us, since sigpipe is hard to avoid with network connections if internet is flickery
  if(!std::atomic_is_lock_free(&sigPipeReceivedCount)) {
    logger.write("sigPipeReceivedCount is not lock free, we will just ignore sigpipe outright");
    std::signal(SIGPIPE, sigPipeHandlerDoNothing);
  }
  else {
    std::signal(SIGPIPE, sigPipeHandler);
  }
#endif

  const int maxSimultaneousRatingGamesPossible = std::min(taskRepFactor * maxRatingMatches, maxSimultaneousGames);

  //Don't write "validation" data for distributed self-play. If the server-side wants to split out some data as "validation" for training
  //then that can be done server-side.
  const double validationProp = 0.0;
  //If we ever get more than this many games behind on writing data, something is weird.
  const int maxSelfplayDataQueueSize = maxSimultaneousGames * 4;
  const int maxRatingDataQueueSize = maxSimultaneousRatingGamesPossible * 4;
  const int logGamesEvery = 1;

  const string gameSeedBase = Global::uint64ToHexString(seedRand.nextUInt64());

  Setup::initializeSession(*userCfg);

  //-----------------------------------------------------------------------------------------------------------------

  //Shared across all game threads
  ThreadSafeQueue<GameTask> gameTaskQueue(1);
  ForkData* forkData = new ForkData();
  std::atomic<int64_t> numGamesStarted(0);
  std::atomic<int64_t> numRatingGamesActive(0);
  std::atomic<int64_t> numMovesPlayed(0);

  auto runGameLoop = [
    &logger,forkData,&gameSeedBase,&gameTaskQueue,&numGamesStarted,&sgfsDir,&connection,
    &numRatingGamesActive,&numMovesPlayed,&watchOngoingGameInFile,&watchOngoingGameInFileName,
    &logGamesAsJson, &alwaysIncludeOwnership
  ] (
    int gameLoopThreadIdx
  ) {
    std::unique_ptr<std::ostream> outputEachMove = nullptr;
    std::function<void()> flushOutputEachMove = nullptr;
    if(gameLoopThreadIdx == 0 && watchOngoingGameInFile) {
#ifdef OS_IS_WINDOWS
      FILE* file = NULL;
      fopen_s(&file, watchOngoingGameInFileName.c_str(), "a");
      if(file == NULL)
        throw StringError("Could not open file: " + watchOngoingGameInFileName);
      outputEachMove = std::make_unique<std::ofstream>(file);
      flushOutputEachMove = [file]() {
        FlushFileBuffers((HANDLE) _get_osfhandle(_fileno(file)));
      };
#else
      outputEachMove = std::make_unique<std::ofstream>(watchOngoingGameInFileName.c_str(), ofstream::app);
#endif
    }

    Rand thisLoopSeedRand;
    while(true) {
      GameTask gameTask;
      bool success = gameTaskQueue.waitPop(gameTask);
      if(!success)
        break;
      if(!shouldStop.load()) {
        string seed = gameSeedBase + ":" + Global::uint64ToHexString(thisLoopSeedRand.nextUInt64());
        int64_t gameIdx = numGamesStarted.fetch_add(1,std::memory_order_acq_rel);
        runAndUploadSingleGame(connection,gameTask,gameIdx,logger,seed,forkData,sgfsDir,thisLoopSeedRand,numMovesPlayed,outputEachMove,flushOutputEachMove,logGamesAsJson,alwaysIncludeOwnership);
      }
      gameTask.blackManager->release(gameTask.nnEvalBlack);
      gameTask.whiteManager->release(gameTask.nnEvalWhite);
      gameTask.blackManager->clearUnusedModelCaches();
      if(gameTask.whiteManager != gameTask.blackManager)
        gameTask.whiteManager->clearUnusedModelCaches();

      if(gameTask.task.isRatingGame)
        numRatingGamesActive.fetch_add(-1,std::memory_order_acq_rel);
    }
  };
  auto runGameLoopProtected = [&logger,&runGameLoop](int gameLoopThreadIdx) {
    Logger::logThreadUncaught("game loop", &logger, [&](){ runGameLoop(gameLoopThreadIdx); });
  };

  //-----------------------------------------------------------------------------------------------------------------

  bool userCfgWarnedYet = false;

  auto loadNeuralNetIntoManager =
    [&runParams,&tdataDir,&sgfsDir,&logger,&userCfg,maxSimultaneousGames,maxSimultaneousRatingGamesPossible,&userCfgWarnedYet](
      SelfplayManager* manager, const Client::ModelInfo modelInfo, const string& modelFile, bool isRatingManager
    ) {
    const string& modelName = modelInfo.name;
    if(manager->hasModel(modelName))
      return;

    logger.write("Found new neural net " + modelName);

    //At load time, check the sha256 again to make sure we have the right thing.
    try {
      modelInfo.failIfSha256Mismatch(modelFile);
    }
    catch(const StringError& e) {
      (void)e;
      //If it's wrong, fail (it means someone modified the file on disk, or there was harddrive corruption, or something, since that file
      //must have been valid at download time), but also rename the file out of the way so that if we restart the program, the next try
      //will do a fresh download.
      string newName = modelFile + ".invalid";
      logger.write("Model file modified or corrupted on disk, sha256 no longer matches? Moving it to " + newName + " and failing.");
      std::rename(modelFile.c_str(),newName.c_str());
      throw;
    }

    int maxSimultaneousGamesThisNet = isRatingManager ? maxSimultaneousRatingGamesPossible : maxSimultaneousGames;
    int maxConcurrentEvals = runParams.maxSearchThreadsAllowed * maxSimultaneousGamesThisNet * 2 + 16;
    int expectedConcurrentEvals = runParams.maxSearchThreadsAllowed * maxSimultaneousGamesThisNet;
    int defaultMaxBatchSize = maxSimultaneousGamesThisNet;

    //Unlike local self-play, which waits to accumulate a fixed number of rows before writing, distributed selfplay writes
    //training data game by game. So we set a buffer size here large enough to always hold all the rows of a game.
    //These values should be vastly more than enough, yet still not use too much memory.
    double firstFileRandMinProp = 1.0;
    int maxRowsPerTrainFile = 20000;

    Rand rand;
    NNEvaluator* nnEval = Setup::initializeNNEvaluator(
      modelName,modelFile,modelInfo.sha256,*userCfg,logger,rand,maxConcurrentEvals,expectedConcurrentEvals,
      NNPos::MAX_BOARD_LEN,NNPos::MAX_BOARD_LEN,defaultMaxBatchSize,
      Setup::SETUP_FOR_DISTRIBUTED
    );
    assert(!nnEval->isNeuralNetLess() || modelFile == "/dev/null");
    logger.write("Loaded latest neural net " + modelName + " from: " + modelFile);

    if(!userCfgWarnedYet) {
      userCfgWarnedYet = true;
      userCfg->warnUnusedKeys(cerr,&logger);
    }

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

  //-----------------------------------------------------------------------------------------------------------------

  //For distributed selfplay, we have a single thread primarily in charge of the manager, so we turn this off
  //to ensure there is no asynchronous removal of models.
  bool autoCleanupAllButLatestIfUnused = false;
  SelfplayManager* selfplayManager = new SelfplayManager(validationProp, maxSelfplayDataQueueSize, &logger, logGamesEvery, autoCleanupAllButLatestIfUnused);
  SelfplayManager* ratingManager = new SelfplayManager(validationProp, maxRatingDataQueueSize, &logger, logGamesEvery, autoCleanupAllButLatestIfUnused);

  //Start game loop threads! Yay!
  //Just start based on selfplay games, rating games will poke in as needed
  vector<std::thread> gameThreads;
  for(int i = 0; i<maxSimultaneousGames; i++) {
    gameThreads.push_back(std::thread(runGameLoopProtected,i));
  }

  //-----------------------------------------------------------------------------------------------------------------

  ClockTimer timer;
  double lastPerformanceTime = timer.getSeconds();
  int64_t lastPerformanceNumMoves = numMovesPlayed.load(std::memory_order_relaxed);
  int64_t lastPerformanceNumNNEvals = (int64_t)(selfplayManager->getTotalNumRowsProcessed() + ratingManager->getTotalNumRowsProcessed());
  auto maybePrintPerformanceUnsynchronized = [&]() {
    double now = timer.getSeconds();
    //At most every minute, report performance
    if(now >= lastPerformanceTime + reportPerformanceEvery) {
      int64_t newNumMoves = numMovesPlayed.load(std::memory_order_relaxed);
      int64_t newNumNNEvals = (int64_t)(selfplayManager->getTotalNumRowsProcessed() + ratingManager->getTotalNumRowsProcessed());

      double timeDiff = now - lastPerformanceTime;
      double movesDiff = (double)(newNumMoves - lastPerformanceNumMoves);
      double numNNEvalsDiff = (double)(newNumNNEvals - lastPerformanceNumNNEvals);

      logger.write(
        Global::strprintf(
          "Performance: in the last %.1f seconds, played %.0f moves (%.1f/sec) and %.0f nn evals (%f/sec)",
          timeDiff, movesDiff, movesDiff/timeDiff, numNNEvalsDiff, numNNEvalsDiff/timeDiff
        )
      );

      lastPerformanceTime = now;
      lastPerformanceNumMoves = newNumMoves;
      lastPerformanceNumNNEvals = newNumNNEvals;
    }
  };

  //-----------------------------------------------------------------------------------------------------------------

  //Loop at a random wide interval downloading the latest net if we're going to need it.
  //Randomize a bit so that the server sees download requests as being well-spaced out.
  auto preDownloadLoop = [&]() {
    if(disablePredownloadLoop)
      return;
    Rand preDownloadLoopRand;
    while(true) {
      if(shouldStop.load())
        return;
      logger.write("Maybe predownloading model...");
      connection->maybeDownloadNewestModel(modelsDir,shouldStop);
      //20 to 25 minutes
      double sleepTimeTotal = preDownloadLoopRand.nextDouble(1200,1500);
      constexpr double stopPollFrequency = 5.0;
      while(sleepTimeTotal > 0.0) {
        double sleepTime = std::min(sleepTimeTotal, stopPollFrequency);
        if(shouldStop.load())
          return;
        std::this_thread::sleep_for(std::chrono::duration<double>(sleepTime));
        sleepTimeTotal -= stopPollFrequency;
      }
    }
  };
  auto preDownloadLoopProtected = [&logger,&preDownloadLoop]() {
    Logger::logThreadUncaught("pre download loop", &logger, preDownloadLoop);
  };
  std::thread preDownloadThread(preDownloadLoopProtected);

  //-----------------------------------------------------------------------------------------------------------------

  PriorityMutex taskLoopMutex;
  double lastTaskQueryTime = timer.getSeconds();
  bool anyTaskSuccessfullyParsedYet = false;
  constexpr double taskLoopSleepTime = 1.0;

  //Multiple of these may be running!
  //Loop acquiring tasks and feeding them to game threads
  auto taskLoop = [&]() {
    Rand taskRand;
    while(true) {
      if(shouldStop.load())
        break;
      std::this_thread::sleep_for(std::chrono::duration<double>(taskRand.nextDouble(taskLoopSleepTime,taskLoopSleepTime*2)));
      PriorityLock taskLock(taskLoopMutex);
      taskLock.lockLowPriority();

      maybePrintPerformanceUnsynchronized();
      if(shouldStop.load())
        break;

      //Make sure that among multiple task loops, that we can't loop or query too fast.
      {
        double now = timer.getSeconds();
        if(now < lastTaskQueryTime + taskLoopSleepTime)
          continue;
      }

#ifdef SIGPIPE
      while(sigPipeReceivedCount.load() > 0) {
        sigPipeReceivedCount.fetch_add(-1);
        logger.write("Note: SIGPIPE received at some point, it's possible this is from bad internet rather than a broke shell pipe, so ignoring rather than killing the program.");
      }
#endif

      //Make sure we register if rating games are done so that we can know if we can accept more.
      ratingManager->cleanupUnusedModelsOlderThan(0);

      bool retryOnFailure = anyTaskSuccessfullyParsedYet;
      //Only allow rating tasks when we can handle a whole new chunk of games
      bool allowRatingTask = (
        maxRatingMatches > 0 &&
        numRatingGamesActive.load(std::memory_order_acquire) <= (maxRatingMatches - 1) * taskRepFactor &&
        (int64_t)ratingManager->numModels() <= maxRatingMatches * 2 - 2
      );
      bool allowSelfplayTask = !onlyPlayRatingMatches;

      Client::Task task;
      bool suc = connection->getNextTask(task,baseDir,retryOnFailure,allowSelfplayTask,allowRatingTask,taskRepFactor,shouldStop);
      if(!suc)
        continue;
      lastTaskQueryTime = timer.getSeconds();

      if(task.runName != runParams.runName) {
        throw StringError(
          "This self-play client was started with the run \"" + task.runName +
          "\" but the server now appears to be hosting a new run \"" + runParams.runName +
          "\", you may need to re-start this client."
        );
      }

      logger.write(
        "Number of nets loaded: selfplay " + Global::uint64ToString(selfplayManager->numModels())
        + " rating " + Global::uint64ToString(ratingManager->numModels())
      );

      if(task.isRatingGame) {
        string sgfOutputDir = sgfsDir + "/" + task.taskGroup;
        MakeDir::make(sgfOutputDir);
      }

      //Attempt model file download!
      {
        bool givingUpOnTask = false;
        //This loop is so that if somehow a model gets removed in between our download and re-picking up the lock,
        //we'll download it again, so long as the download itself didn't seem to have any errors.
        while(
          !connection->isModelPresent(task.modelBlack,modelsDir) ||
          !connection->isModelPresent(task.modelWhite,modelsDir)
        ) {
          //Drop the lock while we download, so that other task loops can proceed
          taskLock.unlock();
          bool suc1;
          bool suc2;
          try {
            suc1 = connection->downloadModelIfNotPresent(task.modelBlack,modelsDir,shouldStop);
            suc2 = connection->downloadModelIfNotPresent(task.modelWhite,modelsDir,shouldStop);
          }
          catch(StringError& e) {
            logger.write(string("Giving up on task due to downloading model error:\n") + e.what());
            suc1 = false;
            suc2 = false;
          }
          //Pick up the lock again after download
          taskLock.lockHighPriority();

          if(shouldStop.load())
            break;
          //If the download itself had errors, we give up
          if(!suc1 || !suc2) {
            givingUpOnTask = true;
            break;
          }

          //No apparent errors, hit the while loop condition again to make sure we have the models
          continue;
        }
        if(shouldStop.load())
          break;
        if(givingUpOnTask)
          continue;
      }

      //Update model file modified times as a way to track which ones we've used recently or not
      string modelFileBlack = Client::Connection::getModelPath(task.modelBlack,modelsDir);
      string modelFileWhite = Client::Connection::getModelPath(task.modelWhite,modelsDir);
      if(!task.modelBlack.isRandom) {
        LoadModel::setLastModifiedTimeToNow(modelFileBlack,logger);
      }
      if(!task.modelWhite.isRandom && task.modelWhite.name != task.modelBlack.name) {
        LoadModel::setLastModifiedTimeToNow(modelFileWhite,logger);
      }

      //For selfplay, unload after 20 seconds, so that if we're playing only one game at a time,
      //we don't repeatedly load and unload - we leave time to get the next task which will probably use the same model.
      //For rating, just always unload, we're often not going to be playing the same model the next time around, or not right away.
      selfplayManager->cleanupUnusedModelsOlderThan(20);
      ratingManager->cleanupUnusedModelsOlderThan(0);

      SelfplayManager* blackManager;
      SelfplayManager* whiteManager;

      //If we happen to be rating the same net as for selfplay, then just load it from the selfplay manager
      if(task.isRatingGame) {
        if(selfplayManager->hasModel(task.modelBlack.name))
          blackManager = selfplayManager;
        else
          blackManager = ratingManager;
        if(selfplayManager->hasModel(task.modelWhite.name))
          whiteManager = selfplayManager;
        else
          whiteManager = ratingManager;
      }
      else {
        blackManager = selfplayManager;
        whiteManager = selfplayManager;
      }

      loadNeuralNetIntoManager(blackManager,task.modelBlack,modelFileBlack,task.isRatingGame);
      loadNeuralNetIntoManager(whiteManager,task.modelWhite,modelFileWhite,task.isRatingGame);
      if(shouldStop.load())
        break;

      //Clean up old models, after we've definitely loaded what we needed
      time_t modelFileAgeLimit = time(NULL) - (time_t)(deleteUnusedModelsAfterDays * 86400);
      LoadModel::deleteModelsOlderThan(modelsDir,logger,modelFileAgeLimit);

      for(int rep = 0; rep < taskRepFactor; rep++) {
        //Game loop threads are responsible for releasing when done.
        NNEvaluator* nnEvalBlack = blackManager->acquireModel(task.modelBlack.name);
        NNEvaluator* nnEvalWhite = whiteManager->acquireModel(task.modelWhite.name);

        //Randomly swap black and white per each game in the rep
        GameTask gameTask;
        gameTask.task = task;
        gameTask.repIdx = rep;

        if(taskRand.nextBool(0.5)) {
          gameTask.blackManager = blackManager;
          gameTask.whiteManager = whiteManager;
          gameTask.nnEvalBlack = nnEvalBlack;
          gameTask.nnEvalWhite = nnEvalWhite;
        }
        else {
          //Swap everything
          gameTask.blackManager = whiteManager;
          gameTask.whiteManager = blackManager;
          gameTask.nnEvalBlack = nnEvalWhite;
          gameTask.nnEvalWhite = nnEvalBlack;
          //Also swap the model within the task, which is used for data writing
          gameTask.task.modelBlack = task.modelWhite;
          gameTask.task.modelWhite = task.modelBlack;
        }

        if(task.isRatingGame)
          numRatingGamesActive.fetch_add(1,std::memory_order_acq_rel);
        suc = gameTaskQueue.waitPush(gameTask);
        (void)suc;
        assert(suc);
      }

      anyTaskSuccessfullyParsedYet = true;
    }
  };
  auto taskLoopProtected = [&logger,&taskLoop]() {
    Logger::logThreadUncaught("task loop", &logger, taskLoop);
  };

  int numTaskLoopThreads = 4;
  vector<std::thread> taskLoopThreads;
  for(int i = 0; i<numTaskLoopThreads; i++) {
    taskLoopThreads.push_back(std::thread(taskLoopProtected));
  }
  //Wait for all task loop threads to stop
  for(int i = 0; i<taskLoopThreads.size(); i++)
    taskLoopThreads[i].join();

  logger.write("Beginning shutdown");

  //This should trigger game threads to quit
  gameTaskQueue.setReadOnly();

  //Make sure we have a true in here
  shouldStop.store(true);

  //Wait for download thread to stop
  preDownloadThread.join();

  //Wait for all game threads to stop
  for(int i = 0; i<gameThreads.size(); i++)
    gameThreads[i].join();

  //At this point, nothing else except possibly data write loops are running, within the selfplay manager.
  delete selfplayManager;
  delete ratingManager;

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
