#include "../core/global.h"
#include "../core/config_parser.h"
#include "../core/timer.h"
#include "../core/datetime.h"
#include "../core/makedir.h"
#include "../search/asyncbot.h"
#include "../program/setup.h"
#include "../program/playutils.h"
#include "../program/play.h"
#include "../command/commandline.h"
#include "../main.h"

#include "../external/nlohmann_json/json.hpp"

using namespace std;
using json = nlohmann::json;

struct AnalyzeRequest {
  string id;
  int turnNumber;
  int priority;

  Board board;
  BoardHistory hist;
  Player nextPla;

  SearchParams params;
  Player perspective;
  int analysisPVLen;
  bool includeOwnership;
  bool includePolicy;
  bool includePVVisits;

  vector<int> avoidMoveUntilByLocBlack;
  vector<int> avoidMoveUntilByLocWhite;
};

int MainCmds::analysis(int argc, const char* const* argv) {
  Board::initHash();
  ScoreValue::initTables();
  Rand seedRand;

  ConfigParser cfg;
  string modelFile;
  int numAnalysisThreads;
  bool quitWithoutWaiting;

  try {
    KataGoCommandLine cmd("Run KataGo parallel JSON-based analysis engine.");
    cmd.addConfigFileArg("","analysis_example.cfg");
    cmd.addModelFileArg();
    cmd.setShortUsageArgLimit();
    cmd.addOverrideConfigArg();

    TCLAP::ValueArg<int> numAnalysisThreadsArg("","analysis-threads","Analysis up to this many positions in parallel",true,0,"THREADS");
    TCLAP::SwitchArg quitWithoutWaitingArg("","quit-without-waiting","When stdin is closed, quit quickly without waiting for queued tasks");
    cmd.add(numAnalysisThreadsArg);
    cmd.add(quitWithoutWaitingArg);
    cmd.parse(argc,argv);

    modelFile = cmd.getModelFile();
    numAnalysisThreads = numAnalysisThreadsArg.getValue();
    quitWithoutWaiting = quitWithoutWaitingArg.getValue();

    if(numAnalysisThreads <= 0 || numAnalysisThreads >= 16384)
      throw StringError("Invalid value for numAnalysisThreads");

    cmd.getConfig(cfg);
  }
  catch (TCLAP::ArgException &e) {
    cerr << "Error: " << e.error() << " for argument " << e.argId() << endl;
    return 1;
  }

  Logger logger;
  if(cfg.contains("logFile") && cfg.contains("logDir"))
    throw StringError("Cannot specify both logFile and logDir in config");
  else if(cfg.contains("logFile"))
    logger.addFile(cfg.getString("logFile"));
  else if(cfg.contains("logDir")) {
    MakeDir::make(cfg.getString("logDir"));
    Rand rand;
    logger.addFile(cfg.getString("logDir") + "/" + DateTime::getCompactDateTimeString() + "-" + Global::uint32ToHexString(rand.nextUInt()) + ".log");
  }

  const bool logToStderr = cfg.contains("logToStderr") ? cfg.getBool("logToStderr") : true;
  if(logToStderr)
    logger.setLogToStderr(true);

  logger.write("Analysis Engine starting...");
  logger.write(Version::getKataGoVersionForHelp());
  if(!logToStderr) {
    cerr << Version::getKataGoVersionForHelp() << endl;
  }

  const bool logAllRequests = cfg.contains("logAllRequests") ? cfg.getBool("logAllRequests") : false;
  const bool logAllResponses = cfg.contains("logAllResponses") ? cfg.getBool("logAllResponses") : false;
  const bool logSearchInfo = cfg.contains("logSearchInfo") ? cfg.getBool("logSearchInfo") : false;

  auto loadParams = [](ConfigParser& config, SearchParams& params, Player& perspective, Player defaultPerspective) {
    params = Setup::loadSingleParams(config);
    perspective = Setup::parseReportAnalysisWinrates(config,defaultPerspective);
    //Set a default for conservativePass that differs from matches or selfplay
    if(!config.contains("conservativePass") && !config.contains("conservativePass0"))
      params.conservativePass = true;
  };

  SearchParams defaultParams;
  Player defaultPerspective;
  loadParams(cfg, defaultParams, defaultPerspective, C_EMPTY);

  const int analysisPVLen = cfg.contains("analysisPVLen") ? cfg.getInt("analysisPVLen",1,100) : 15;
  const bool assumeMultipleStartingBlackMovesAreHandicap =
    cfg.contains("assumeMultipleStartingBlackMovesAreHandicap") ? cfg.getBool("assumeMultipleStartingBlackMovesAreHandicap") : true;
  const bool preventEncore = cfg.contains("preventCleanupPhase") ? cfg.getBool("preventCleanupPhase") : true;

  NNEvaluator* nnEval;
  {
    Setup::initializeSession(cfg);
    int maxConcurrentEvals = numAnalysisThreads * defaultParams.numThreads * 2 + 16; // * 2 + 16 just to give plenty of headroom
    int defaultMaxBatchSize = -1;
    nnEval = Setup::initializeNNEvaluator(
      modelFile,modelFile,cfg,logger,seedRand,maxConcurrentEvals,
      NNPos::MAX_BOARD_LEN,NNPos::MAX_BOARD_LEN,defaultMaxBatchSize,
      Setup::SETUP_FOR_ANALYSIS
    );
  }

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

  //Check for unused config keys
  cfg.warnUnusedKeys(cerr,&logger);

  logger.write("Loaded config "+ cfg.getFileName());
  logger.write("Loaded model "+ modelFile);

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

  ThreadSafePriorityQueue<std::pair<int,int64_t>, AnalyzeRequest*> toAnalyzeQueue;
  int64_t numRequestsSoFar = 0; // used as tie breaker for requests with same priority

  auto reportError = [&pushToWrite](const string& s) {
    json ret;
    ret["error"] = s;
    pushToWrite(new string(ret.dump()));
  };
  auto reportErrorForId = [&pushToWrite](const string& id, const string& field, const string& s) {
    json ret;
    ret["id"] = id;
    ret["field"] = field;
    ret["error"] = s;
    pushToWrite(new string(ret.dump()));
  };
  auto reportWarningForId = [&pushToWrite](const string& id, const string& field, const string& s) {
    json ret;
    ret["id"] = id;
    ret["field"] = field;
    ret["warning"] = s;
    pushToWrite(new string(ret.dump()));
  };

  auto analysisLoop = [
    &logger,&toAnalyzeQueue,&toWriteQueue,&preventEncore,&pushToWrite,&reportError,&logSearchInfo,&nnEval
  ](AsyncBot* bot) {
    while(true) {
      std::pair<std::pair<int,int64_t>,AnalyzeRequest*> analysisItem;
      bool suc = toAnalyzeQueue.waitPop(analysisItem);
      if(!suc)
        break;
      AnalyzeRequest* request = analysisItem.second;

      bot->setPosition(request->nextPla,request->board,request->hist);
      bot->setAlwaysIncludeOwnerMap(request->includeOwnership);
      bot->setParams(request->params);
      bot->setAvoidMoveUntilByLoc(request->avoidMoveUntilByLocBlack,request->avoidMoveUntilByLocWhite);

      Player pla = request->nextPla;
      bot->genMoveSynchronous(pla, TimeControls());
      if(logSearchInfo) {
        ostringstream sout;
        PlayUtils::printGenmoveLog(sout,bot,nnEval,Board::NULL_LOC,NAN,request->perspective);
        logger.write(sout.str());
      }

      json ret;
      ret["id"] = request->id;
      ret["turnNumber"] = request->turnNumber;

      int minMoves = 0;
      vector<AnalysisData> buf;

      const Search* search = bot->getSearch();
      search->getAnalysisData(buf,minMoves,false,request->analysisPVLen);

      const Player perspective = request->perspective;

      // Stats for all the individual moves
      json moveInfos = json::array();
      for(int i = 0; i<buf.size(); i++) {
        const AnalysisData& data = buf[i];
        double winrate = 0.5 * (1.0 + data.winLossValue);
        double utility = data.utility;
        double lcb = PlayUtils::getHackedLCBForWinrate(search,data,pla);
        double utilityLcb = data.lcb;
        double scoreMean = data.scoreMean;
        double lead = data.lead;
        if(perspective == P_BLACK || (perspective != P_BLACK && perspective != P_WHITE && pla == P_BLACK)) {
          winrate = 1.0-winrate;
          lcb = 1.0 - lcb;
          utility = -utility;
          scoreMean = -scoreMean;
          lead = -lead;
          utilityLcb = -utilityLcb;
        }
        json moveInfo;
        moveInfo["move"] = Location::toString(data.move,request->board);
        moveInfo["visits"] = data.numVisits;
        moveInfo["utility"] = utility;
        moveInfo["winrate"] = winrate;
        moveInfo["scoreMean"] = lead;
        moveInfo["scoreSelfplay"] = scoreMean;
        moveInfo["scoreLead"] = lead;
        moveInfo["scoreStdev"] = data.scoreStdev;
        moveInfo["prior"] = data.policyPrior;
        moveInfo["lcb"] = lcb;
        moveInfo["utilityLcb"] = utilityLcb;
        moveInfo["order"] = data.order;

        json pv = json::array();
        int pvLen = (preventEncore && data.pvContainsPass()) ? data.getPVLenUpToPhaseEnd(request->board,request->hist,request->nextPla) : (int)data.pv.size();
        for(int j = 0; j<pvLen; j++)
          pv.push_back(Location::toString(data.pv[j],request->board));
        moveInfo["pv"] = pv;

        if(request->includePVVisits) {
          assert(data.pvVisits.size() >= pvLen);
          json pvVisits = json::array();
          for(int j = 0; j<pvLen; j++)
            pvVisits.push_back(data.pvVisits[j]);
          moveInfo["pvVisits"] = pvVisits;
        }
        moveInfos.push_back(moveInfo);
      }
      ret["moveInfos"] = moveInfos;

      //If the search didn't have any root or root neural net output, it must have been interrupted and we must be quitting imminently
      if(search->rootNode == NULL || search->rootNode->nnOutput == nullptr) {
        logger.write("Note: Search quitting due to no visits - this is normal and possible when shutting down but a bug under any other situation.");
        delete request;
        continue;
      }

      // Stats for root position
      {
        ReportedSearchValues rootVals;
        search->getRootValues(rootVals);
        Player rootPla = getOpp(request->nextPla);

        double winrate = 0.5 * (1.0 + rootVals.winLossValue);
        double scoreMean = rootVals.expectedScore;
        double lead = rootVals.lead;
        double utility = rootVals.utility;

        if(perspective == P_BLACK || (perspective != P_BLACK && perspective != P_WHITE && rootPla == P_BLACK)) {
          winrate = 1.0-winrate;
          scoreMean = -scoreMean;
          lead = -lead;
          utility = -utility;
        }

        json rootInfo;
        rootInfo["visits"] = search->rootNode->stats.visits; // not in ReportedSearchValues
        rootInfo["winrate"] = winrate;
        rootInfo["scoreSelfplay"] = scoreMean;
        rootInfo["scoreLead"] = lead;
        rootInfo["scoreStdev"] = rootVals.expectedScoreStdev;
        rootInfo["utility"] = utility;
        ret["rootInfo"] = rootInfo;
      }

      // Raw policy prior
      if(request->includePolicy) {
        float policyProbs[NNPos::MAX_NN_POLICY_SIZE];
        std::copy(search->rootNode->nnOutput->policyProbs, search->rootNode->nnOutput->policyProbs+NNPos::MAX_NN_POLICY_SIZE, policyProbs);
        json policy = json::array();
        int nnXLen = bot->getSearch()->nnXLen, nnYLen = bot->getSearch()->nnYLen;
        const Board& board = request->board;
        for(int y = 0; y<board.y_size; y++) {
          for(int x = 0; x<board.x_size; x++) {
            int pos = NNPos::xyToPos(x,y,nnXLen);
            policy.push_back(policyProbs[pos]);
          }
        }

        int passPos = NNPos::locToPos(Board::PASS_LOC, board.x_size, nnXLen, nnYLen);
        policy.push_back(policyProbs[passPos]);
        ret["policy"] = policy;
      }
      // Average tree ownership
      if(request->includeOwnership) {
        static constexpr int ownershipMinVisits = 3;
        vector<double> ownership = search->getAverageTreeOwnership(ownershipMinVisits);

        json ownerships = json::array();
        const Board& board = request->board;
        int nnXLen = bot->getSearch()->nnXLen;
        for(int y = 0; y<board.y_size; y++) {
          for(int x = 0; x<board.x_size; x++) {
            int pos = NNPos::xyToPos(x,y,nnXLen);
            double o;
            if(perspective == P_BLACK || (perspective != P_BLACK && perspective != P_WHITE && pla == P_BLACK))
              o = -ownership[pos];
            else
              o = ownership[pos];
            ownerships.push_back(o);
          }
        }
        ret["ownership"] = ownerships;
      }
      pushToWrite(new string(ret.dump()));

      //Free up bot resources in case it's a while before we do more search
      bot->clearSearch();
      delete request;
    }
  };

  vector<std::thread> threads;
  std::thread write_thread = std::thread(writeLoop);
  vector<AsyncBot*> bots;
  for(int i = 0; i<numAnalysisThreads; i++) {
    string searchRandSeed = Global::uint64ToHexString(seedRand.nextUInt64()) + Global::uint64ToHexString(seedRand.nextUInt64());
    AsyncBot* bot = new AsyncBot(defaultParams, nnEval, &logger, searchRandSeed);
    threads.push_back(std::thread(analysisLoop,bot));
    bots.push_back(bot);
  }

  logger.write("Analyzing up to " + Global::intToString(numAnalysisThreads) + " positions at at time in parallel");
  logger.write("Started, ready to begin handling requests");
  if(!logToStderr) {
    cerr << "Started, ready to begin handling requests" << endl;
  }

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

    AnalyzeRequest rbase;
    if(input.find("id") == input.end() || !input["id"].is_string()) {
      reportError("Request must have a string \"id\" field");
      continue;
    }
    rbase.id = input["id"].get<string>();

    //Defaults
    rbase.params = defaultParams;
    rbase.perspective = defaultPerspective;
    rbase.analysisPVLen = analysisPVLen;
    rbase.includeOwnership = false;
    rbase.includePolicy = false;
    rbase.includePVVisits = false;
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
        reportErrorForId(rbase.id, "boardXSize", boardSizeError.c_str());
        continue;
      }
      if(!parseInteger(input, "boardYSize", yBuf, 2, Board::MAX_LEN, boardSizeError.c_str())) {
        reportErrorForId(rbase.id, "boardYSize", boardSizeError.c_str());
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
          //Hard failure on unused override keys newly present in the config
          vector<string> unusedKeys = localCfg.unusedKeys();
          if(unusedKeys.size() > 0) {
            reportErrorForId(rbase.id, "overrideSettings", string("Unknown config params: ") + Global::concat(unusedKeys,","));
            continue;
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
    if(input.find("includeOwnership") != input.end()) {
      bool suc = parseBoolean(input, "includeOwnership", rbase.includeOwnership, "Must be a boolean");
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
    if(input.find("priority") != input.end()) {
      int64_t buf;
      bool suc = parseInteger(input, "priority", buf, -0x7FFFFFFF,0x7FFFFFFF, "Must be a number from -2,147,483,647 to 2,147,483,647");
      if(!suc)
        continue;
      rbase.priority = (int)buf;
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
        AnalyzeRequest* newRequest = new AnalyzeRequest();
        newRequest->id = rbase.id;
        newRequest->turnNumber = turnNumber;
        newRequest->board = board;
        newRequest->hist = hist;
        newRequest->nextPla = nextPla;
        newRequest->params = rbase.params;
        newRequest->perspective = rbase.perspective;
        newRequest->analysisPVLen = rbase.analysisPVLen;
        newRequest->includeOwnership = rbase.includeOwnership;
        newRequest->includePolicy = rbase.includePolicy;
        newRequest->includePVVisits = rbase.includePVVisits;
        newRequest->priority = rbase.priority;
        newRequest->avoidMoveUntilByLocBlack = rbase.avoidMoveUntilByLocBlack;
        newRequest->avoidMoveUntilByLocWhite = rbase.avoidMoveUntilByLocWhite;
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

    for(int i = 0; i<newRequests.size(); i++) {
      //Compare first by user-provided priority, and next breaks ties by preferring earlier requests.
      std::pair<int,int64_t> priorityKey = std::make_pair(newRequests[i]->priority, -numRequestsSoFar);
      bool suc = toAnalyzeQueue.forcePush( std::make_pair(priorityKey, newRequests[i]) );
      assert(suc);
      (void)suc;
      numRequestsSoFar++;
    }
    newRequests.clear();
  }

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
  delete nnEval;
  NeuralNet::globalCleanup();
  ScoreValue::freeTables();
  logger.write("All cleaned up, quitting");
  return 0;
}
