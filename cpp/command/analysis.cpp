#include "../core/global.h"
#include "../core/config_parser.h"
#include "../core/timer.h"
#include "../core/datetime.h"
#include "../core/makedir.h"
#include "../search/asyncbot.h"
#include "../search/patternbonustable.h"
#include "../program/setup.h"
#include "../program/playutils.h"
#include "../program/play.h"
#include "../command/commandline.h"
#include "../main.h"

#include "../external/nlohmann_json/json.hpp"

using namespace std;
using json = nlohmann::json;

struct AnalyzeRequest {
  int64_t internalId;
  string id;
  int turnNumber;
  int64_t priority;

  Board board;
  BoardHistory hist;
  Player nextPla;

  SearchParams params;
  Player perspective;
  int analysisPVLen;
  bool includeOwnership;
  bool includeOwnershipStdev;
  bool includeMovesOwnership;
  bool includeMovesOwnershipStdev;
  bool includePolicy;
  bool includePVVisits;

  bool reportDuringSearch;
  double reportDuringSearchEvery;
  double firstReportDuringSearchAfter;

  vector<int> avoidMoveUntilByLocBlack;
  vector<int> avoidMoveUntilByLocWhite;

  //Starts with STATUS_IN_QUEUE.
  //Thread that grabs it from queue it changes it to STATUS_POPPED
  //Once search is fully started thread sticks in its own thread index
  //At any point it may change to STATUS_TERMINATED.
  //If it ever gets to STATUS_POPPED or later, then the analysis thread is reponsible for writing the result, else the api thread is
  static constexpr int STATUS_IN_QUEUE = -1;
  static constexpr int STATUS_POPPED = -2;
  static constexpr int STATUS_TERMINATED = -3;
  std::atomic<int> status;
};


int MainCmds::analysis(const vector<string>& args) {
  Board::initHash();
  ScoreValue::initTables();
  Rand seedRand;

  ConfigParser cfg;
  string modelFile;
  string humanModelFile;
  bool numAnalysisThreadsCmdlineSpecified;
  int numAnalysisThreadsCmdline;
  bool quitWithoutWaiting;

  KataGoCommandLine cmd("Run KataGo parallel JSON-based analysis engine.");
  try {
    cmd.addConfigFileArg("","analysis_example.cfg");
    cmd.addModelFileArg();
    cmd.addHumanModelFileArg();
    cmd.setShortUsageArgLimit();
    cmd.addOverrideConfigArg();

    TCLAP::ValueArg<int> numAnalysisThreadsArg("","analysis-threads","Analyze up to this many positions in parallel. Equivalent to numAnalysisThreads in the config.",false,0,"THREADS");
    TCLAP::SwitchArg quitWithoutWaitingArg("","quit-without-waiting","When stdin is closed, quit quickly without waiting for queued tasks");
    cmd.add(numAnalysisThreadsArg);
    cmd.add(quitWithoutWaitingArg);
    cmd.parseArgs(args);

    modelFile = cmd.getModelFile();
    humanModelFile = cmd.getHumanModelFile();
    numAnalysisThreadsCmdlineSpecified = numAnalysisThreadsArg.isSet();
    numAnalysisThreadsCmdline = numAnalysisThreadsArg.getValue();
    quitWithoutWaiting = quitWithoutWaitingArg.getValue();

    cmd.getConfig(cfg);
  }
  catch (TCLAP::ArgException &e) {
    cerr << "Error: " << e.error() << " for argument " << e.argId() << endl;
    return 1;
  }
  cfg.applyAlias("numSearchThreadsPerAnalysisThread", "numSearchThreads");

  if(cfg.contains("numAnalysisThreads") && numAnalysisThreadsCmdlineSpecified)
    throw StringError("When specifying numAnalysisThreads in the config (" + cfg.getFileName() + "), it is redundant and disallowed to also specify it via -analysis-threads");

  const int numAnalysisThreads = numAnalysisThreadsCmdlineSpecified ? numAnalysisThreadsCmdline : cfg.getInt("numAnalysisThreads",1,16384);
  if(numAnalysisThreads <= 0 || numAnalysisThreads > 16384)
    throw StringError("Invalid value for numAnalysisThreads: " + Global::intToString(numAnalysisThreads));

  const bool forDeterministicTesting =
    cfg.contains("forDeterministicTesting") ? cfg.getBool("forDeterministicTesting") : false;
  if(forDeterministicTesting)
    seedRand.init("forDeterministicTesting");

  const bool logToStdoutDefault = false;
  const bool logToStderrDefault = true;
  Logger logger(&cfg, logToStdoutDefault, logToStderrDefault);
  const bool logToStderr = logger.isLoggingToStderr();

  logger.write("Analysis Engine starting...");
  logger.write(Version::getKataGoVersionForHelp());
  if(!logToStderr) {
    cerr << Version::getKataGoVersionForHelp() << endl;
  }

  const bool logAllRequests = cfg.contains("logAllRequests") ? cfg.getBool("logAllRequests") : false;
  const bool logAllResponses = cfg.contains("logAllResponses") ? cfg.getBool("logAllResponses") : false;
  const bool logErrorsAndWarnings = cfg.contains("logErrorsAndWarnings") ? cfg.getBool("logErrorsAndWarnings") : true;
  const bool logSearchInfo = cfg.contains("logSearchInfo") ? cfg.getBool("logSearchInfo") : false;

  auto loadParams = [&humanModelFile](ConfigParser& config, SearchParams& params, Player& perspective, Player defaultPerspective) {
    bool hasHumanModel = humanModelFile != "";
    params = Setup::loadSingleParams(config,Setup::SETUP_FOR_ANALYSIS,hasHumanModel);
    perspective = Setup::parseReportAnalysisWinrates(config,defaultPerspective);
    //Set a default for conservativePass that differs from matches or selfplay
    if(!config.contains("conservativePass"))
      params.conservativePass = true;
  };

  SearchParams defaultParams;
  Player defaultPerspective;
  loadParams(cfg, defaultParams, defaultPerspective, C_EMPTY);

  std::unique_ptr<PatternBonusTable> patternBonusTable = nullptr;
  {
    std::vector<std::unique_ptr<PatternBonusTable>> tables = Setup::loadAvoidSgfPatternBonusTables(cfg,logger);
    assert(tables.size() == 1);
    patternBonusTable = std::move(tables[0]);
  }

  const int analysisPVLen = cfg.contains("analysisPVLen") ? cfg.getInt("analysisPVLen",1,100) : 15;
  const bool assumeMultipleStartingBlackMovesAreHandicap =
    cfg.contains("assumeMultipleStartingBlackMovesAreHandicap") ? cfg.getBool("assumeMultipleStartingBlackMovesAreHandicap") : true;
  const bool preventEncore = cfg.contains("preventCleanupPhase") ? cfg.getBool("preventCleanupPhase") : true;

  NNEvaluator* nnEval = NULL;
  NNEvaluator* humanEval = NULL;
  {
    Setup::initializeSession(cfg);
    const int expectedConcurrentEvals = numAnalysisThreads * defaultParams.numThreads;
    const bool defaultRequireExactNNLen = false;
    const int defaultMaxBatchSize = -1;
    const bool disableFP16 = false;
    const string expectedSha256 = "";
    nnEval = Setup::initializeNNEvaluator(
      modelFile,modelFile,expectedSha256,cfg,logger,seedRand,expectedConcurrentEvals,
      NNPos::MAX_BOARD_LEN,NNPos::MAX_BOARD_LEN,defaultMaxBatchSize,defaultRequireExactNNLen,disableFP16,
      Setup::SETUP_FOR_ANALYSIS
    );
    if(humanModelFile != "") {
      humanEval = Setup::initializeNNEvaluator(
        humanModelFile,humanModelFile,expectedSha256,cfg,logger,seedRand,expectedConcurrentEvals,
        NNPos::MAX_BOARD_LEN,NNPos::MAX_BOARD_LEN,defaultMaxBatchSize,defaultRequireExactNNLen,disableFP16,
        Setup::SETUP_FOR_ANALYSIS
      );
      if(!humanEval->requiresSGFMetadata()) {
        string warning;
        warning += "WARNING: Human model was not trained from SGF metadata to vary by rank! Did you pass the wrong model for -human-model?\n";
        logger.write(warning);
        if(!logToStderr)
          cerr << warning << endl;
      }
    }
  }

#ifndef USE_EIGEN_BACKEND
  {
    int nnMaxBatchSizeTotal = nnEval->getNumGpus() * nnEval->getMaxBatchSize();
    int numThreadsTotal = defaultParams.numThreads * numAnalysisThreads;
    if(nnMaxBatchSizeTotal * 1.5 <= numThreadsTotal) {
      logger.write(
        Global::strprintf(
          "Note: nnMaxBatchSize * number of GPUs (%d) is smaller than numSearchThreads * numAnalysisThreads (%d)",
          nnMaxBatchSizeTotal, numThreadsTotal
        )
      );
      logger.write("The number of simultaneous threads that might query the GPU could be larger than the batch size that the GPU will handle at once.");
      logger.write("It may improve performance to increase nnMaxBatchSize, unless you are constrained on GPU memory.");
    }
  }
#endif

  //Check for unused config keys
  cfg.warnUnusedKeys(cerr,&logger);

  logger.write("Loaded config "+ cfg.getFileName());
  logger.write("Loaded model "+ modelFile);
  cmd.logOverrides(logger);

  if(humanModelFile != "" && !cfg.contains("humanSLProfile")) {
    logger.write("WARNING: Provided -human-model but humanSLProfile was not set in the config. The human SL model will not be used until it is set.");
    if(!logger.isLoggingToStderr())
      cerr << "WARNING: Provided -human-model but humanSLProfile was not set in the config. The human SL model will not be used until it is set." << endl;
  }

  ThreadSafeQueue<string*> toWriteQueue;
  auto writeLoop = [&toWriteQueue,&logAllResponses,&logger]() {
    while(true) {
      string* message;
      bool suc = toWriteQueue.waitPop(message);
      if(!suc)
        break;
      cout << *message << endl;
      if(logAllResponses)
        logger.write("Response: " + *message);
      delete message;
    }
  };

  auto pushToWrite = [&toWriteQueue](string* s) {
    bool suc = toWriteQueue.forcePush(s);
    if(!suc)
      delete s;
  };

  ThreadSafePriorityQueue<std::pair<int64_t,int64_t>, AnalyzeRequest*> toAnalyzeQueue;
  int64_t numRequestsSoFar = 0; // Used as tie breaker for requests with same priority
  int64_t internalIdCounter = 0; // Counter for internalId on requests.

  //Open requests, keyed by internalId, mutexed by the mutex
  std::mutex openRequestsMutex;
  std::map<int64_t, AnalyzeRequest*> openRequests;

  auto reportError = [&pushToWrite,&logger,&logErrorsAndWarnings](const string& s) {
    json ret;
    ret["error"] = s;
    pushToWrite(new string(ret.dump()));
    if(logErrorsAndWarnings)
      logger.write("Error: " + ret.dump());
  };
  auto reportErrorForId = [&pushToWrite,&logger,&logErrorsAndWarnings](const string& id, const string& field, const string& s) {
    json ret;
    ret["id"] = id;
    ret["field"] = field;
    ret["error"] = s;
    pushToWrite(new string(ret.dump()));
    if(logErrorsAndWarnings)
      logger.write("Error: " + ret.dump());
  };
  auto reportWarningForId = [&pushToWrite,&logger,&logErrorsAndWarnings](const string& id, const string& field, const string& s) {
    json ret;
    ret["id"] = id;
    ret["field"] = field;
    ret["warning"] = s;
    pushToWrite(new string(ret.dump()));
    if(logErrorsAndWarnings)
      logger.write("Warning: " + ret.dump());
  };

  //Report analysis for which we don't actually have results. This is used when something is user-terminated before being actually
  //analyzed properly. Only used outside of search too
  auto reportNoAnalysis = [&pushToWrite](const AnalyzeRequest* request) {
    json ret;
    ret["id"] = request->id;
    ret["turnNumber"] = request->turnNumber;
    ret["isDuringSearch"] = false;
    ret["noResults"] = true;
    pushToWrite(new string(ret.dump()));
  };

  //Returns false if no analysis was reportable due to there being no root node or search results.
  auto reportAnalysis = [&preventEncore,&pushToWrite](const AnalyzeRequest* request, const Search* search, bool isDuringSearch) {
    json ret;
    ret["id"] = request->id;
    ret["turnNumber"] = request->turnNumber;
    ret["isDuringSearch"] = isDuringSearch;

    bool success = search->getAnalysisJson(
      request->perspective,
      request->analysisPVLen, preventEncore, request->includePolicy,
      request->includeOwnership,request->includeOwnershipStdev,
      request->includeMovesOwnership,request->includeMovesOwnershipStdev,
      request->includePVVisits,
      ret
    );

    if(success)
      pushToWrite(new string(ret.dump()));
    return success;
  };

  auto analysisLoop = [
    &logger,&toAnalyzeQueue,&reportAnalysis,&reportNoAnalysis,&logSearchInfo,&nnEval,&openRequestsMutex,&openRequests
  ](AsyncBot* bot, int threadIdx) {
    while(true) {
      std::pair<std::pair<int64_t,int64_t>,AnalyzeRequest*> analysisItem;
      bool suc = toAnalyzeQueue.waitPop(analysisItem);
      if(!suc)
        break;
      AnalyzeRequest* request = analysisItem.second;
      int expected = AnalyzeRequest::STATUS_IN_QUEUE;
      //If it's already terminated, then there's nothing for us to do
      if(!request->status.compare_exchange_strong(expected, AnalyzeRequest::STATUS_POPPED, std::memory_order_acq_rel)) {
        assert(expected == AnalyzeRequest::STATUS_TERMINATED);
      }
      //Else, the request is live and we marked it as popped
      else {
        bot->setPosition(request->nextPla,request->board,request->hist);
        bot->setAlwaysIncludeOwnerMap(request->includeOwnership || request->includeOwnershipStdev || request->includeMovesOwnership || request->includeMovesOwnershipStdev);
        bot->setParams(request->params);
        bot->setAvoidMoveUntilByLoc(request->avoidMoveUntilByLocBlack,request->avoidMoveUntilByLocWhite);

        Player pla = request->nextPla;
        double searchFactor = 1.0;

        //Handle termination between the time we pop and the search starts
        std::function<void()> onSearchBegun = [&request,&bot,&threadIdx]() {
          //Try to record that we're handling this request and indicate that the search is started by this thread
          int expected2 = AnalyzeRequest::STATUS_POPPED;
          //If it was terminated, then stop our search
          if(!request->status.compare_exchange_strong(expected2, threadIdx, std::memory_order_acq_rel)) {
            assert(expected2 == AnalyzeRequest::STATUS_TERMINATED);
            bot->stopWithoutWait();
          }
        };

        if(request->reportDuringSearch) {
          std::function<void(const Search* search)> callback = [&request,&reportAnalysis](const Search* search) {
            const bool isDuringSearch = true;
            reportAnalysis(request,search,isDuringSearch);
          };
          bot->genMoveSynchronousAnalyze(
            pla, TimeControls(), searchFactor,
            request->reportDuringSearchEvery, request->firstReportDuringSearchAfter,
            callback, onSearchBegun
          );
        }
        else {
          bot->genMoveSynchronous(pla, TimeControls(), searchFactor, onSearchBegun);
        }

        if(logSearchInfo) {
          ostringstream sout;
          PlayUtils::printGenmoveLog(sout,bot,nnEval,Board::NULL_LOC,NAN,request->perspective,false);
          logger.write(sout.str());
        }

        {
          const bool isDuringSearch = false;
          const Search* search = bot->getSearch();
          bool analysisWritten = reportAnalysis(request,search,isDuringSearch);
          //If the search didn't have any root or root neural net output, it must have been interrupted and we must be quitting imminently
          if(!analysisWritten) {
            //If the reason we stopped was because we noticed a terminate, then we will write out a dummy response even if we didn't have
            //enough info to generate a real one, to fulfill a promise in the API docs that we always write something.
            if(request->status.load(std::memory_order_acquire) == AnalyzeRequest::STATUS_TERMINATED)
              reportNoAnalysis(request);
            //Otherwise, this case is only possible if we're just shutting down
            else
              logger.write("Note: Search quitting due to no visits - this is normal and possible when shutting down but a bug under any other situation.");
          }
        }
      }

      //Free up bot resources in case it's a while before we do more search
      bot->clearSearch();

      //This request is no longer open
      {
        std::lock_guard<std::mutex> lock(openRequestsMutex);
        openRequests.erase(request->internalId);
      }
      delete request;
    }
  };
  auto analysisLoopProtected = [&logger,&analysisLoop](AsyncBot* bot, int threadIdx) {
    Logger::logThreadUncaught("analysis loop", &logger, [&](){ analysisLoop(bot, threadIdx); });
  };

  vector<std::thread> threads;
  std::thread write_thread = std::thread(writeLoop);
  vector<AsyncBot*> bots;
  for(int threadIdx = 0; threadIdx<numAnalysisThreads; threadIdx++) {
    string searchRandSeed = Global::uint64ToHexString(seedRand.nextUInt64()) + Global::uint64ToHexString(seedRand.nextUInt64());
    AsyncBot* bot = new AsyncBot(defaultParams, nnEval, humanEval, &logger, searchRandSeed);
    bot->setCopyOfExternalPatternBonusTable(patternBonusTable);
    threads.push_back(std::thread(analysisLoopProtected,bot,threadIdx));
    bots.push_back(bot);
  }

  logger.write("Analyzing up to " + Global::intToString(numAnalysisThreads) + " positions at a time in parallel");
  logger.write("Started, ready to begin handling requests");
  if(!logToStderr) {
    cerr << "Started, ready to begin handling requests" << endl;
  }

  auto terminateRequest = [&bots,&reportNoAnalysis](AnalyzeRequest* request) {
    //Firstly, flag the request as terminated
    int prevStatus = request->status.exchange(AnalyzeRequest::STATUS_TERMINATED,std::memory_order_acq_rel);
    //Already terminated? Nothing to do.
    if(prevStatus == AnalyzeRequest::STATUS_TERMINATED)
    {}
    //No thread claimed it, so it's up to us to write the result
    else if(prevStatus == AnalyzeRequest::STATUS_IN_QUEUE) {
      reportNoAnalysis(request);
    }
    //A thread popped it. That thread will notice that it's terminated once it tries to put its thread idx in, so we need not do anything.
    else if(prevStatus == AnalyzeRequest::STATUS_POPPED)
    {}
    //A thread started searching it and put its thread idx in
    else {
      assert(prevStatus >= 0);
      //We've already set the above status to terminated so when the thread terminates due to our killing it below, it will see this.
      //Or else the thread has already done so, in which case it's already properly written a result, also fine.
      int threadIdx = prevStatus;
      //Terminate it by thread index
      bots[threadIdx]->stopWithoutWait();
    }
  };

  auto requestLoop = [&]() {
    string line;
    json input;
    while(getline(cin,line)) {
      line = Global::trim(line);
      if(line.length() == 0)
        continue;

      if(logAllRequests)
        logger.write("Request: " + line);

      try {
        input = json::parse(line);
      }
      catch(nlohmann::detail::exception& e) {
        reportError(e.what() + string(" - could not parse input line as json request: ") + line);
        continue;
      }

      if(!input.is_object()) {
        reportError("Request line was valid json but was not an object, ignoring: " + input.dump());
        continue;
      }

      if(input.find("id") == input.end() || !input["id"].is_string()) {
        reportError("Request must have a string \"id\" field");
        continue;
      }

      AnalyzeRequest rbase;
      rbase.id = input["id"].get<string>();

      //Special actions
      if(input.find("action") != input.end() && input["action"].is_string()) {
        string action = input["action"].get<string>();
        if(action == "query_version") {
          input["version"] = Version::getKataGoVersion();
          input["git_hash"] = Version::getGitRevision();
          pushToWrite(new string(input.dump()));
        }
        else if(action == "clear_cache") {
          //This should be thread-safe.
          nnEval->clearCache();
          if(humanEval != NULL)
            humanEval->clearCache();
          pushToWrite(new string(input.dump()));
        }
        else if(action == "terminate") {

          bool terminateIdFound = false;
          string terminateId;
          if(input.find("terminateId") != input.end() && input["terminateId"].is_string()) {
            terminateId = input["terminateId"].get<string>();
            terminateIdFound = true;
          }
          if(!terminateIdFound) {
            reportErrorForId(rbase.id, "terminateId", "Requests for a terminate action must have a string \"terminateId\" field");
            continue;
          }

          bool hasTurnNumbers = false;
          vector<int> turnNumbers;
          if(input.find("turnNumbers") != input.end()) {
            try {
              turnNumbers = input["turnNumbers"].get<vector<int> >();
              hasTurnNumbers = true;
            }
            catch(nlohmann::detail::exception&) {
              reportErrorForId(rbase.id, "turnNumbers", "If provided, must be an array of integers indicating turns to terminate");
              continue;
            }
          }

          {
            std::lock_guard<std::mutex> lock(openRequestsMutex);
            std::set<int> turnNumbersSet(turnNumbers.begin(),turnNumbers.end());
            for(auto it = openRequests.begin(); it != openRequests.end(); ++it) {
              AnalyzeRequest* request = it->second;
              if(request->id == terminateId && (!hasTurnNumbers || (turnNumbersSet.find(request->turnNumber) != turnNumbersSet.end())))
                terminateRequest(request);
            }
          }
          pushToWrite(new string(input.dump()));
        }
        else if(action == "terminate_all") {
          bool hasTurnNumbers = false;
          vector<int> turnNumbers;
          if(input.find("turnNumbers") != input.end()) {
            try {
              turnNumbers = input["turnNumbers"].get<vector<int> >();
              hasTurnNumbers = true;
            }
            catch(nlohmann::detail::exception&) {
              reportErrorForId(rbase.id, "turnNumbers", "If provided, must be an array of integers indicating turns to terminate");
              continue;
            }
          }

          {
            std::lock_guard<std::mutex> lock(openRequestsMutex);
            std::set<int> turnNumbersSet(turnNumbers.begin(),turnNumbers.end());
            for(auto it = openRequests.begin(); it != openRequests.end(); ++it) {
              AnalyzeRequest* request = it->second;
              if(!hasTurnNumbers || (turnNumbersSet.find(request->turnNumber) != turnNumbersSet.end()))
                terminateRequest(request);
            }
          }
          pushToWrite(new string(input.dump()));
        }
        else {
          reportError("'action' field must be 'query_version' or 'terminate' or 'terminate_all'");
        }

        continue;
      }

      //Defaults
      rbase.params = defaultParams;
      rbase.perspective = defaultPerspective;
      rbase.analysisPVLen = analysisPVLen;
      rbase.includeOwnership = false;
      rbase.includeOwnershipStdev = false;
      rbase.includeMovesOwnership = false;
      rbase.includeMovesOwnershipStdev = false;
      rbase.includePolicy = false;
      rbase.includePVVisits = false;
      rbase.reportDuringSearch = false;
      rbase.reportDuringSearchEvery = 1e30;
      rbase.firstReportDuringSearchAfter = 1e30;
      rbase.priority = 0;
      rbase.avoidMoveUntilByLocBlack.clear();
      rbase.avoidMoveUntilByLocWhite.clear();

      auto parseInteger = [&rbase,&reportErrorForId](const json& dict, const char* field, int64_t& buf, int64_t min, int64_t max, const char* errorMessage) {
        try {
          if(!dict[field].is_number_integer()) {
            reportErrorForId(rbase.id, field, errorMessage);
            return false;
          }
          int64_t x = dict[field].get<int64_t>();
          if(x < min || x > max) {
            reportErrorForId(rbase.id, field, errorMessage);
            return false;
          }
          buf = x;
          return true;
        }
        catch(nlohmann::detail::exception& e) {
          (void)e;
          reportErrorForId(rbase.id, field, errorMessage);
          return false;
        }
      };

      auto parseDouble = [&rbase,&reportErrorForId](const json& dict, const char* field, double& buf, double min, double max, const char* errorMessage) {
        try {
          if(!dict[field].is_number()) {
            reportErrorForId(rbase.id, field, errorMessage);
            return false;
          }
          double x = dict[field].get<double>();
          if(!isfinite(x) || x < min || x > max) {
            reportErrorForId(rbase.id, field, errorMessage);
            return false;
          }
          buf = x;
          return true;
        }
        catch(nlohmann::detail::exception& e) {
          (void)e;
          reportErrorForId(rbase.id, field, errorMessage);
          return false;
        }
      };

      auto parseBoolean = [&rbase,&reportErrorForId](const json& dict, const char* field, bool& buf, const char* errorMessage) {
        try {
          if(!dict[field].is_boolean()) {
            reportErrorForId(rbase.id, field, errorMessage);
            return false;
          }
          buf = dict[field].get<bool>();
          return true;
        }
        catch(nlohmann::detail::exception& e) {
          (void)e;
          reportErrorForId(rbase.id, field, errorMessage);
          return false;
        }
      };

      auto parsePlayer = [&rbase,&reportErrorForId](const json& dict, const char* field, Player& buf) {
        buf = C_EMPTY;
        try {
          string s = dict[field].get<string>();
          PlayerIO::tryParsePlayer(s,buf);
        }
        catch(nlohmann::detail::exception&) {}
        if(buf != P_BLACK && buf != P_WHITE) {
          reportErrorForId(rbase.id, field, "Must be \"b\" or \"w\"");
          return false;
        }
        return true;
      };

      int boardXSize;
      int boardYSize;
      {
        int64_t xBuf;
        int64_t yBuf;
        static const string boardSizeError = string("Must provide an integer from 2 to ") + Global::intToString(Board::MAX_LEN);
        if(input.find("boardXSize") == input.end()) {
          reportErrorForId(rbase.id, "boardXSize", boardSizeError.c_str());
          continue;
        }
        if(input.find("boardYSize") == input.end()) {
          reportErrorForId(rbase.id, "boardYSize", boardSizeError.c_str());
          continue;
        }
        if(!parseInteger(input, "boardXSize", xBuf, 2, Board::MAX_LEN, boardSizeError.c_str())) {
          continue;
        }
        if(!parseInteger(input, "boardYSize", yBuf, 2, Board::MAX_LEN, boardSizeError.c_str())) {
          continue;
        }
        boardXSize = (int)xBuf;
        boardYSize = (int)yBuf;
      }

      auto parseBoardLocs = [boardXSize,boardYSize,&rbase,&reportErrorForId](const json& dict, const char* field, vector<Loc>& buf, bool allowPass) {
        buf.clear();
        if(!dict[field].is_array()) {
          reportErrorForId(rbase.id, field, "Must be an array of GTP board vertices");
          return false;
        }
        for(auto& elt : dict[field]) {
          string s;
          try {
            s = elt.get<string>();
          }
          catch(nlohmann::detail::exception& e) {
            (void)e;
            reportErrorForId(rbase.id, field, "Must be an array of GTP board vertices");
            return false;
          }

          Loc loc;
          if(!Location::tryOfString(s, boardXSize, boardYSize, loc) ||
             (!allowPass && loc == Board::PASS_LOC) ||
             (loc == Board::NULL_LOC)) {
            reportErrorForId(rbase.id, field, "Could not parse board location: " + s);
            return false;
          }
          buf.push_back(loc);
        }
        return true;
      };

      auto parseBoardMoves = [boardXSize,boardYSize,&rbase,&reportErrorForId](const json& dict, const char* field, vector<Move>& buf, bool allowPass) {
        buf.clear();
        if(!dict[field].is_array()) {
          reportErrorForId(rbase.id, field, "Must be an array of pairs of the form: [\"b\" or \"w\", GTP board vertex]");
          return false;
        }
        for(auto& elt : dict[field]) {
          if(!elt.is_array() || elt.size() != 2) {
            reportErrorForId(rbase.id, field, "Must be an array of pairs of the form: [\"b\" or \"w\", GTP board vertex]");
            return false;
          }

          string s0;
          string s1;
          try {
            s0 = elt[0].get<string>();
            s1 = elt[1].get<string>();
          }
          catch(nlohmann::detail::exception& e) {
            (void)e;
            reportErrorForId(rbase.id, field, "Must be an array of pairs of the form: [\"b\" or \"w\", GTP board vertex]");
            return false;
          }

          Player pla;
          if(!PlayerIO::tryParsePlayer(s0,pla)) {
            reportErrorForId(rbase.id, field, "Could not parse player: " + s0);
            return false;
          }

          Loc loc;
          if(!Location::tryOfString(s1, boardXSize, boardYSize, loc) ||
             (!allowPass && loc == Board::PASS_LOC) ||
             (loc == Board::NULL_LOC)) {
            reportErrorForId(rbase.id, field, "Could not parse board location: " + s1);
            return false;
          }
          buf.push_back(Move(loc,pla));
        }
        return true;
      };

      vector<Move> placements;
      if(input.find("initialStones") != input.end()) {
        if(!parseBoardMoves(input, "initialStones", placements, false))
          continue;
      }
      vector<Move> moveHistory;
      if(input.find("moves") != input.end()) {
        if(!parseBoardMoves(input, "moves", moveHistory, true))
          continue;
      }
      else {
        reportErrorForId(rbase.id, "moves", "Must specify an array of [player,location] pairs");
        continue;
      }
      Player initialPlayer = C_EMPTY;
      if(input.find("initialPlayer") != input.end()) {
        bool suc = parsePlayer(input, "initialPlayer", initialPlayer);
        if(!suc)
          continue;
      }

      vector<bool> shouldAnalyze(moveHistory.size()+1,false);
      if(input.find("analyzeTurns") != input.end()) {
        vector<int> analyzeTurns;
        try {
          analyzeTurns = input["analyzeTurns"].get<vector<int> >();
        }
        catch(nlohmann::detail::exception&) {
          reportErrorForId(rbase.id, "analyzeTurns", "Must specify an array of integers indicating turns to analyze");
          continue;
        }

        bool failed = false;
        for(int i = 0; i<analyzeTurns.size(); i++) {
          int turnNumber = analyzeTurns[i];
          if(turnNumber < 0 || turnNumber >= shouldAnalyze.size()) {
            reportErrorForId(rbase.id, "analyzeTurns", "Invalid turn number: " + Global::intToString(turnNumber));
            failed = true;
            break;
          }
          shouldAnalyze[turnNumber] = true;
        }
        if(failed)
          continue;
      }
      else {
        shouldAnalyze[shouldAnalyze.size()-1] = true;
      }

      std::map<int,int64_t> priorities;
      if(input.find("priorities") != input.end()) {
        vector<int64_t> prioritiesVec;
        try {
          prioritiesVec = input["priorities"].get<vector<int64_t> >();
        }
        catch(nlohmann::detail::exception&) {
          reportErrorForId(rbase.id, "priorities", "Must specify an array of integers indicating priorities");
          continue;
        }
        if(input.find("analyzeTurns") == input.end()) {
          reportErrorForId(rbase.id, "priorities", "Can only specify when also specifying analyzeTurns");
          continue;
        }
        vector<int> analyzeTurns = input["analyzeTurns"].get<vector<int> >();
        if(prioritiesVec.size() != analyzeTurns.size()) {
          reportErrorForId(rbase.id, "priorities", "Must be of matching length to analyzeTurns");
          continue;
        }

        bool failed = false;
        for(int i = 0; i<prioritiesVec.size(); i++) {
          int64_t priority = prioritiesVec[i];
          if(priority < -0x3FFFffffFFFFffffLL || priority > 0x3FFFffffFFFFffffLL) {
            reportErrorForId(rbase.id, "priorities", "Invalid priority: " + Global::int64ToString(priority));
            failed = true;
            break;
          }
          priorities[analyzeTurns[i]] = priority;
        }
        if(failed) {
          priorities.clear();
          continue;
        }
      }


      Rules rules;
      if(input.find("rules") != input.end()) {
        if(input["rules"].is_string()) {
          string s = input["rules"].get<string>();
          if(!Rules::tryParseRules(s,rules)) {
            reportErrorForId(rbase.id, "rules", "Could not parse rules: " + s);
            continue;
          }
        }
        else if(input["rules"].is_object()) {
          string s = input["rules"].dump();
          if(!Rules::tryParseRules(s,rules)) {
            reportErrorForId(rbase.id, "rules", "Could not parse rules: " + s);
            continue;
          }
        }
        else {
          reportErrorForId(rbase.id, "rules", "Must specify rules string, such as \"chinese\" or \"tromp-taylor\", or a JSON object with detailed rules parameters.");
          continue;
        }
      }
      else {
        reportErrorForId(rbase.id, "rules", "Must specify rules string, such as \"chinese\" or \"tromp-taylor\", or a JSON object with detailed rules parameters.");
        continue;
      }

      if(input.find("komi") != input.end()) {
        double komi;
        static_assert(Rules::MIN_USER_KOMI == -150.0f, "");
        static_assert(Rules::MAX_USER_KOMI == 150.0f, "");
        const char* msg = "Must be a integer or half-integer from -150.0 to 150.0";
        bool suc = parseDouble(input, "komi", komi, Rules::MIN_USER_KOMI, Rules::MAX_USER_KOMI, msg);
        if(!suc)
          continue;
        rules.komi = (float)komi;
        if(!Rules::komiIsIntOrHalfInt(rules.komi)) {
          reportErrorForId(rbase.id, "rules", msg);
          continue;
        }
      }

      if(input.find("whiteHandicapBonus") != input.end()) {
        if(!input["whiteHandicapBonus"].is_string()) {
          reportErrorForId(rbase.id, "whiteHandicapBonus", "Must be a string");
          continue;
        }
        string s = input["whiteHandicapBonus"].get<string>();
        try {
          int whiteHandicapBonusRule = Rules::parseWhiteHandicapBonusRule(s);
          rules.whiteHandicapBonusRule = whiteHandicapBonusRule;
        }
        catch(const StringError& err) {
          reportErrorForId(rbase.id, "whiteHandicapBonus", err.what());
          continue;
        }
      }

      if(input.find("overrideSettings") != input.end()) {
        json settings = input["overrideSettings"];
        if(!settings.is_object()) {
          reportErrorForId(rbase.id, "overrideSettings", "Must be an object");
          continue;
        }
        std::map<string,string> overrideSettings;
        for(auto it = settings.begin(); it != settings.end(); ++it) {
          overrideSettings[it.key()] = it.value().is_string() ? it.value().get<string>(): it.value().dump(); // always convert to string
        }

        // Reload settings to allow overrides
        if(!overrideSettings.empty()) {
          try {
            ConfigParser localCfg(cfg);
            //Ignore any unused keys in the ORIGINAL config
            localCfg.markAllKeysUsedWithPrefix("");
            localCfg.overrideKeys(overrideSettings);
            loadParams(localCfg, rbase.params, rbase.perspective, defaultPerspective);
            SearchParams::failIfParamsDifferOnUnchangeableParameter(defaultParams,rbase.params);
            //Soft failure on unused override keys newly present in the config
            vector<string> unusedKeys = localCfg.unusedKeys();
            if(unusedKeys.size() > 0) {
              reportWarningForId(rbase.id, "overrideSettings", string("Unknown config params: ") + Global::concat(unusedKeys,","));
            }
          }
          catch(const StringError& exception) {
            reportErrorForId(rbase.id, "overrideSettings", string("Could not set settings: ") + exception.what());
            continue;
          }
        }
      }

      if(input.find("maxVisits") != input.end()) {
        bool suc = parseInteger(input, "maxVisits", rbase.params.maxVisits, 1, (int64_t)1 << 50, "Must be an integer from 1 to 2^50");
        if(!suc)
          continue;
      }

      if(input.find("analysisPVLen") != input.end()) {
        int64_t buf;
        bool suc = parseInteger(input, "analysisPVLen", buf, 1, 1000, "Must be an integer from 1 to 1000");
        if(!suc)
          continue;
        rbase.analysisPVLen = (int)buf;
      }

      if(input.find("rootFpuReductionMax") != input.end()) {
        bool suc = parseDouble(input, "rootFpuReductionMax", rbase.params.rootFpuReductionMax, 0.0, 2.0, "Must be a number from 0.0 to 2.0");
        if(!suc)
          continue;
      }
      if(input.find("rootPolicyTemperature") != input.end()) {
        bool suc = parseDouble(input, "rootPolicyTemperature", rbase.params.rootPolicyTemperature, 0.01, 100.0, "Must be a number from 0.01 to 100.0");
        if(!suc)
          continue;
        rbase.params.rootPolicyTemperatureEarly = rbase.params.rootPolicyTemperature;
      }
      if(input.find("includeMovesOwnership") != input.end()) {
        bool suc = parseBoolean(input, "includeMovesOwnership", rbase.includeMovesOwnership, "Must be a boolean");
        if(!suc)
          continue;
      }
      if(input.find("includeMovesOwnershipStdev") != input.end()) {
        bool suc = parseBoolean(input, "includeMovesOwnershipStdev", rbase.includeMovesOwnershipStdev, "Must be a boolean");
        if(!suc)
          continue;
      }
      if(input.find("includeOwnership") != input.end()) {
        bool suc = parseBoolean(input, "includeOwnership", rbase.includeOwnership, "Must be a boolean");
        if(!suc)
          continue;
      }
      if(input.find("includeOwnershipStdev") != input.end()) {
        bool suc = parseBoolean(input, "includeOwnershipStdev", rbase.includeOwnershipStdev, "Must be a boolean");
        if(!suc)
          continue;
      }
      if(input.find("includePolicy") != input.end()) {
        bool suc = parseBoolean(input, "includePolicy", rbase.includePolicy, "Must be a boolean");
        if(!suc)
          continue;
      }
      if(input.find("includePVVisits") != input.end()) {
        bool suc = parseBoolean(input, "includePVVisits", rbase.includePVVisits, "Must be a boolean");
        if(!suc)
          continue;
      }
      if(input.find("reportDuringSearchEvery") != input.end()) {
        bool suc = parseDouble(input, "reportDuringSearchEvery", rbase.reportDuringSearchEvery, 0.001, 1000000.0, "Must be number of seconds from 0.001 to 1000000.0");
        if(!suc)
          continue;
        rbase.reportDuringSearch = true;
        rbase.firstReportDuringSearchAfter = rbase.reportDuringSearchEvery;
      }
      if(input.find("firstReportDuringSearchAfter") != input.end()) {
        bool suc = parseDouble(input, "firstReportDuringSearchAfter", rbase.firstReportDuringSearchAfter, 0.001, 1000000.0, "Must be number of seconds from 0.001 to 1000000.0");
        if(!suc)
          continue;
        rbase.reportDuringSearch = true;
      }
      if(input.find("priority") != input.end()) {
        if(input.find("priorities") != input.end()) {
          reportErrorForId(rbase.id, "priority", "Cannot specify both priority and priorities");
          continue;
        }
        int64_t buf;
        bool suc = parseInteger(input, "priority", buf, -0x3FFFffffFFFFffffLL,0x3FFFffffFFFFffffLL, "Must be a number between -2^62 and 2^62");
        if(!suc)
          continue;
        rbase.priority = buf;
      }

      bool hasAllowMoves = input.find("allowMoves") != input.end();
      bool hasAvoidMoves = input.find("avoidMoves") != input.end();
      if(hasAllowMoves || hasAvoidMoves) {
        if(hasAllowMoves && hasAvoidMoves) {
          reportErrorForId(rbase.id, "allowMoves", string("Cannot specify both allowMoves and avoidMoves"));
          continue;
        }
        string field = hasAllowMoves ? "allowMoves" : "avoidMoves";
        json& avoidParamsList = input[field];
        if(!avoidParamsList.is_array()) {
          reportErrorForId(rbase.id, field, string("Must be a list of dicts with subfields 'player', 'moves', 'untilDepth'"));
          continue;
        }
        if(hasAllowMoves && avoidParamsList.size() > 1) {
          reportErrorForId(rbase.id, field, string("Currently allowMoves only allows one entry"));
          continue;
        }

        bool failed = false;
        for(size_t i = 0; i<avoidParamsList.size(); i++) {
          json& avoidParams = avoidParamsList[i];
          if(avoidParams.find("moves") == avoidParams.end() ||
             avoidParams.find("untilDepth") == avoidParams.end() ||
             avoidParams.find("player") == avoidParams.end()) {
            reportErrorForId(rbase.id, field, string("Must be a list of dicts with subfields 'player', 'moves', 'untilDepth'"));
            failed = true;
            break;
          }

          Player avoidPla;
          vector<Loc> parsedLocs;
          int64_t untilDepth;
          bool suc;
          suc = parsePlayer(avoidParams, "player", avoidPla);
          if(!suc) { failed = true; break; }
          suc = parseBoardLocs(avoidParams, "moves", parsedLocs, true);
          if(!suc) { failed = true; break; }
          suc = parseInteger(avoidParams, "untilDepth", untilDepth, 1, 1000000000, "Must be a positive integer");
          if(!suc) { failed = true; break; }

          vector<int>& avoidMoveUntilByLoc = avoidPla == P_BLACK ? rbase.avoidMoveUntilByLocBlack : rbase.avoidMoveUntilByLocWhite;
          avoidMoveUntilByLoc.resize(Board::MAX_ARR_SIZE);
          if(hasAllowMoves) {
            std::fill(avoidMoveUntilByLoc.begin(),avoidMoveUntilByLoc.end(),(int)untilDepth);
            for(Loc loc: parsedLocs) {
              avoidMoveUntilByLoc[loc] = 0;
            }
          }
          else {
            for(Loc loc: parsedLocs) {
              avoidMoveUntilByLoc[loc] = (int)untilDepth;
            }
          }
        }
        if(failed)
          continue;
      }


      Board board(boardXSize,boardYSize);
      for(int i = 0; i<placements.size(); i++) {
        board.setStone(placements[i].loc,placements[i].pla);
      }

      if(initialPlayer == C_EMPTY) {
        if(moveHistory.size() > 0)
          initialPlayer = moveHistory[0].pla;
        else
          initialPlayer = BoardHistory::numHandicapStonesOnBoard(board) > 0 ? P_WHITE : P_BLACK;
      }

      bool rulesWereSupported;
      Rules supportedRules = nnEval->getSupportedRules(rules,rulesWereSupported);
      if(!rulesWereSupported) {
        ostringstream out;
        out << "Rules " << rules << " not supported by neural net, using " << supportedRules << " instead";
        reportWarningForId(rbase.id, "rules", out.str());
        rules = supportedRules;
      }

      Player nextPla = initialPlayer;
      BoardHistory hist(board,nextPla,rules,0);
      hist.setAssumeMultipleStartingBlackMovesAreHandicap(assumeMultipleStartingBlackMovesAreHandicap);

      //Build and enqueue requests
      vector<AnalyzeRequest*> newRequests;
      bool foundIllegalMove =  false;
      for(int turnNumber = 0; turnNumber <= moveHistory.size(); turnNumber++) {
        if(shouldAnalyze[turnNumber]) {
          int64_t priority = rbase.priority;
          if(priorities.size() > 0) {
            assert(priorities.size() > newRequests.size());
            assert(priorities.find(turnNumber) != priorities.end());
            priority = priorities[turnNumber];
          }

          AnalyzeRequest* newRequest = new AnalyzeRequest();
          newRequest->internalId = internalIdCounter++;
          newRequest->id = rbase.id;
          newRequest->turnNumber = turnNumber;
          newRequest->board = board;
          newRequest->hist = hist;
          newRequest->nextPla = nextPla;
          newRequest->params = rbase.params;
          newRequest->perspective = rbase.perspective;
          newRequest->analysisPVLen = rbase.analysisPVLen;
          newRequest->includeOwnership = rbase.includeOwnership;
          newRequest->includeOwnershipStdev = rbase.includeOwnershipStdev;
          newRequest->includeMovesOwnership = rbase.includeMovesOwnership;
          newRequest->includeMovesOwnershipStdev = rbase.includeMovesOwnershipStdev;
          newRequest->includePolicy = rbase.includePolicy;
          newRequest->includePVVisits = rbase.includePVVisits;
          newRequest->reportDuringSearch = rbase.reportDuringSearch;
          newRequest->reportDuringSearchEvery = rbase.reportDuringSearchEvery;
          newRequest->firstReportDuringSearchAfter = rbase.firstReportDuringSearchAfter;
          newRequest->priority = priority;
          newRequest->avoidMoveUntilByLocBlack = rbase.avoidMoveUntilByLocBlack;
          newRequest->avoidMoveUntilByLocWhite = rbase.avoidMoveUntilByLocWhite;
          newRequest->status.store(AnalyzeRequest::STATUS_IN_QUEUE,std::memory_order_release);
          newRequests.push_back(newRequest);
        }
        if(turnNumber >= moveHistory.size())
          break;

        Player movePla = moveHistory[turnNumber].pla;
        Loc moveLoc = moveHistory[turnNumber].loc;
        if(movePla != nextPla) {
          board.clearSimpleKoLoc();
          hist.clear(board,movePla,rules,hist.encorePhase);
          hist.setAssumeMultipleStartingBlackMovesAreHandicap(assumeMultipleStartingBlackMovesAreHandicap);
        }

        bool suc = hist.makeBoardMoveTolerant(board,moveLoc,movePla,preventEncore);
        if(!suc) {
          reportErrorForId(rbase.id, "moves", "Illegal move " + Global::intToString(turnNumber) + ": " + Location::toString(moveLoc,board));
          foundIllegalMove = true;
          break;
        }
        nextPla = getOpp(movePla);
      }

      if(foundIllegalMove) {
        for(int i = 0; i<newRequests.size(); i++)
          delete newRequests[i];
        newRequests.clear();
        continue;
      }

      //Add all requests to open requests
      {
        std::lock_guard<std::mutex> lock(openRequestsMutex);
        for(int i = 0; i<newRequests.size(); i++) {
          openRequests[newRequests[i]->internalId] = newRequests[i];
        }
      }
      //Push into queue for processing
      for(int i = 0; i<newRequests.size(); i++) {
        //Compare first by user-provided priority, and next breaks ties by preferring earlier requests.
        std::pair<int64_t,int64_t> priorityKey = std::make_pair(newRequests[i]->priority, -numRequestsSoFar);
        bool suc = toAnalyzeQueue.forcePush( std::make_pair(priorityKey, newRequests[i]) );
        assert(suc);
        (void)suc;
        numRequestsSoFar++;
      }
      newRequests.clear();
    }
  };

  //If request loop raises an exception, we need to log here BEFORE destructing main context, because in some cases
  //gameThreads[i].join() will abort without useful exception due to thread not being joinable,
  //hiding the real exception.
  Logger::logThreadUncaught("request loop", &logger, requestLoop);

  if(quitWithoutWaiting) {
    //Making this readOnly will halt futher output that isn't already queued and signal the write loop thread to terminate.
    toWriteQueue.setReadOnly();
    //Making this readOnly should signal the analysis loop threads to terminate once they have nothing left.
    toAnalyzeQueue.setReadOnly();
    //Interrupt any searches going on to help the analysis threads realize to terminate faster.
    for(int i = 0; i<bots.size(); i++)
      bots[i]->stopWithoutWait();
    for(int i = 0; i<bots.size(); i++)
      bots[i]->setKilled();
    for(int i = 0; i<threads.size(); i++)
      threads[i].join();
    write_thread.join();
  }
  else {
    //Making this readOnly should signal the analysis loop threads to terminate once they have nothing left.
    toAnalyzeQueue.setReadOnly();
    //Wait patiently for everything to finish
    for(int i = 0; i<threads.size(); i++)
      threads[i].join();
    //Signal the write loop thread to terminate
    toWriteQueue.setReadOnly();
    write_thread.join();
  }

  for(int i = 0; i<bots.size(); i++)
    delete bots[i];

  logger.write(nnEval->getModelFileName());
  logger.write("NN rows: " + Global::int64ToString(nnEval->numRowsProcessed()));
  logger.write("NN batches: " + Global::int64ToString(nnEval->numBatchesProcessed()));
  logger.write("NN avg batch size: " + Global::doubleToString(nnEval->averageProcessedBatchSize()));
  if(humanEval != NULL) {
    logger.write(humanEval->getModelFileName());
    logger.write("NN rows: " + Global::int64ToString(humanEval->numRowsProcessed()));
    logger.write("NN batches: " + Global::int64ToString(humanEval->numBatchesProcessed()));
    logger.write("NN avg batch size: " + Global::doubleToString(humanEval->averageProcessedBatchSize()));
  }
  delete nnEval;
  delete humanEval;
  NeuralNet::globalCleanup();
  ScoreValue::freeTables();
  logger.write("All cleaned up, quitting");
  return 0;
}
