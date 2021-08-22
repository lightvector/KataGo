#include "../core/global.h"
#include "../core/config_parser.h"
#include "../core/timer.h"
#include "../core/datetime.h"
#include "../core/makedir.h"
#include "../dataio/sgf.h"
#include "../search/asyncbot.h"
#include "../program/setup.h"
#include "../program/playutils.h"
#include "../program/play.h"
#include "../command/commandline.h"
#include "../main.h"

using namespace std;

static const vector<string> knownCommands = {
  //Basic GTP commands
  "protocol_version",
  "name",
  "version",
  "known_command",
  "list_commands",
  "quit",

  //GTP extension - specify "boardsize X:Y" or "boardsize X Y" for non-square sizes
  //rectangular_boardsize is an alias for boardsize, intended to make it more evident that we have such support
  "boardsize",
  "rectangular_boardsize",

  "clear_board",
  "set_position",
  "komi",
  //GTP extension - get KataGo's current komi setting
  "get_komi",
  "play",
  "undo",

  //GTP extension - specify rules
  "kata-get-rules",
  "kata-set-rule",
  "kata-set-rules",

  //Get or change a few limited params dynamically
  "kata-get-param",
  "kata-set-param",
  "kata-list-params",
  "kgs-rules",

  "genmove",
  "genmove_debug", //Prints additional info to stderr
  "search_debug", //Prints additional info to stderr, doesn't actually make the move

  //Clears neural net cached evaluations and bot search tree, allows fresh randomization
  "clear_cache",

  "showboard",
  "fixed_handicap",
  "place_free_handicap",
  "set_free_handicap",

  "time_settings",
  "kgs-time_settings",
  "time_left",
  //KataGo extensions for time settings
  "kata-list_time_settings",
  "kata-time_settings",

  "final_score",
  "final_status_list",

  "loadsgf",
  "printsgf",

  //GTP extensions for board analysis
  // "genmove_analyze",
  "lz-genmove_analyze",
  "kata-genmove_analyze",
  // "analyze",
  "lz-analyze",
  "kata-analyze",

  //Display raw neural net evaluations
  "kata-raw-nn",

  //Misc other stuff
  "cputime",
  "gomill-cpu_time",

  //Some debug commands
  "kata-debug-print-tc",

  //Stop any ongoing ponder or analyze
  "stop",
};

static bool tryParseLoc(const string& s, const Board& b, Loc& loc) {
  return Location::tryOfString(s,b,loc);
}

static bool timeIsValid(const double& time) {
  if(isnan(time) || time < 0.0 || time > TimeControls::MAX_USER_INPUT_TIME)
    return false;
  return true;
}
static bool timeIsValidAllowNegative(const double& time) {
  if(isnan(time) || time < -TimeControls::MAX_USER_INPUT_TIME || time > TimeControls::MAX_USER_INPUT_TIME)
    return false;
  return true;
}

static double parseTime(const vector<string>& args, int argIdx, const string& description) {
  double time = 0.0;
  if(args.size() <= argIdx || !Global::tryStringToDouble(args[argIdx],time))
    throw StringError("Expected float for " + description + " as argument " + Global::intToString(argIdx));
  if(!timeIsValid(time))
    throw StringError(description + " is an invalid value: " + args[argIdx]);
  return time;
}
static double parseTimeAllowNegative(const vector<string>& args, int argIdx, const string& description) {
  double time = 0.0;
  if(args.size() <= argIdx || !Global::tryStringToDouble(args[argIdx],time))
    throw StringError("Expected float for " + description + " as argument " + Global::intToString(argIdx));
  if(!timeIsValidAllowNegative(time))
    throw StringError(description + " is an invalid value: " + args[argIdx]);
  return time;
}
static int parseByoYomiStones(const vector<string>& args, int argIdx) {
  int byoYomiStones = 0;
  if(args.size() <= argIdx || !Global::tryStringToInt(args[argIdx],byoYomiStones))
    throw StringError("Expected int for byo-yomi overtime stones as argument " + Global::intToString(argIdx));
  if(byoYomiStones < 0 || byoYomiStones > 1000000)
    throw StringError("byo-yomi overtime stones is an invalid value: " + args[argIdx]);
  return byoYomiStones;
}
static int parseByoYomiPeriods(const vector<string>& args, int argIdx) {
  int byoYomiPeriods = 0;
  if(args.size() <= argIdx || !Global::tryStringToInt(args[argIdx],byoYomiPeriods))
    throw StringError("Expected int for byo-yomi overtime periods as argument " + Global::intToString(argIdx));
  if(byoYomiPeriods < 0 || byoYomiPeriods > 1000000)
    throw StringError("byo-yomi overtime periods is an invalid value: " + args[argIdx]);
  return byoYomiPeriods;
}

//Assumes that stones are worth 15 points area and 14 points territory, and that 7 komi is fair
static double initialBlackAdvantage(const BoardHistory& hist) {
  BoardHistory histCopy = hist;
  histCopy.setAssumeMultipleStartingBlackMovesAreHandicap(true);
  int handicapStones = histCopy.computeNumHandicapStones();
  if(handicapStones <= 1)
    return 7.0 - hist.rules.komi;

  //Subtract one since white gets the first move afterward
  int extraBlackStones = handicapStones - 1;
  double stoneValue = hist.rules.scoringRule == Rules::SCORING_AREA ? 15.0 : 14.0;
  double whiteHandicapBonus = 0.0;
  if(hist.rules.whiteHandicapBonusRule == Rules::WHB_N)
    whiteHandicapBonus += handicapStones;
  else if(hist.rules.whiteHandicapBonusRule == Rules::WHB_N_MINUS_ONE)
    whiteHandicapBonus += handicapStones-1;

  return stoneValue * extraBlackStones + (7.0 - hist.rules.komi - whiteHandicapBonus);
}

static double getBoardSizeScaling(const Board& board) {
  return pow(19.0 * 19.0 / (double)(board.x_size * board.y_size), 0.75);
}
static double getPointsThresholdForHandicapGame(double boardSizeScaling) {
  return std::max(4.0 / boardSizeScaling, 2.0);
}

static bool noWhiteStonesOnBoard(const Board& board) {
  for(int y = 0; y < board.y_size; y++) {
    for(int x = 0; x < board.x_size; x++) {
      Loc loc = Location::getLoc(x,y,board.x_size);
      if(board.colors[loc] == P_WHITE)
        return false;
    }
  }
  return true;
}

static void updateDynamicPDAHelper(
  const Board& board, const BoardHistory& hist,
  const double dynamicPlayoutDoublingAdvantageCapPerOppLead,
  const vector<double>& recentWinLossValues,
  double& desiredDynamicPDAForWhite
) {
  (void)board;
  if(dynamicPlayoutDoublingAdvantageCapPerOppLead <= 0.0) {
    desiredDynamicPDAForWhite = 0.0;
  }
  else {
    double boardSizeScaling = getBoardSizeScaling(board);
    double pdaScalingStartPoints = getPointsThresholdForHandicapGame(boardSizeScaling);
    double initialBlackAdvantageInPoints = initialBlackAdvantage(hist);
    Player disadvantagedPla = initialBlackAdvantageInPoints >= 0 ? P_WHITE : P_BLACK;
    double initialAdvantageInPoints = abs(initialBlackAdvantageInPoints);
    if(initialAdvantageInPoints < pdaScalingStartPoints || board.x_size <= 7 || board.y_size <= 7) {
      desiredDynamicPDAForWhite = 0.0;
    }
    else {
      double desiredDynamicPDAForDisadvantagedPla =
        (disadvantagedPla == P_WHITE) ? desiredDynamicPDAForWhite : -desiredDynamicPDAForWhite;

      //What increment to adjust desiredPDA at.
      //Power of 2 to avoid any rounding issues.
      const double increment = 0.125;

      //Hard cap of 2.75 in this parameter, since more extreme values start to reach into values without good training.
      //Scale mildly with board size - small board a given point lead counts as "more".
      double pdaCap = std::min(
        2.75,
        dynamicPlayoutDoublingAdvantageCapPerOppLead *
        (initialAdvantageInPoints - pdaScalingStartPoints) * boardSizeScaling
      );
      pdaCap = round(pdaCap / increment) * increment;

      //No history, or literally no white stones on board? Then this is a new game or a newly set position
      if(recentWinLossValues.size() <= 0 || noWhiteStonesOnBoard(board)) {
        //Just use the cap
        desiredDynamicPDAForDisadvantagedPla = pdaCap;
      }
      else {
        double winLossValue = recentWinLossValues[recentWinLossValues.size()-1];
        //Convert to perspective of disadvantagedPla
        if(disadvantagedPla == P_BLACK)
          winLossValue = -winLossValue;

        //Keep winLossValue between 5% and 25%, subject to available caps.
        if(winLossValue < -0.9)
          desiredDynamicPDAForDisadvantagedPla = desiredDynamicPDAForDisadvantagedPla + 0.125;
        else if(winLossValue > -0.5)
          desiredDynamicPDAForDisadvantagedPla = desiredDynamicPDAForDisadvantagedPla - 0.125;

        desiredDynamicPDAForDisadvantagedPla = std::max(desiredDynamicPDAForDisadvantagedPla, 0.0);
        desiredDynamicPDAForDisadvantagedPla = std::min(desiredDynamicPDAForDisadvantagedPla, pdaCap);
      }

      desiredDynamicPDAForWhite = (disadvantagedPla == P_WHITE) ? desiredDynamicPDAForDisadvantagedPla : -desiredDynamicPDAForDisadvantagedPla;
    }
  }
}

static bool shouldResign(
  const Board& board,
  const BoardHistory& hist,
  Player pla,
  const vector<double>& recentWinLossValues,
  double lead,
  const double resignThreshold,
  const int resignConsecTurns,
  const double resignMinScoreDifference
) {
  double initialBlackAdvantageInPoints = initialBlackAdvantage(hist);

  int minTurnForResignation = 0;
  double noResignationWhenWhiteScoreAbove = board.x_size * board.y_size;
  if(initialBlackAdvantageInPoints > 0.9 && pla == P_WHITE) {
    //Play at least some moves no matter what
    minTurnForResignation = 1 + board.x_size * board.y_size / 5;

    //In a handicap game, also only resign if the lead difference is well behind schedule assuming
    //that we're supposed to catch up over many moves.
    double numTurnsToCatchUp = 0.60 * board.x_size * board.y_size - minTurnForResignation;
    double numTurnsSpent = (double)(hist.moveHistory.size()) - minTurnForResignation;
    if(numTurnsToCatchUp <= 1.0)
      numTurnsToCatchUp = 1.0;
    if(numTurnsSpent <= 0.0)
      numTurnsSpent = 0.0;
    if(numTurnsSpent > numTurnsToCatchUp)
      numTurnsSpent = numTurnsToCatchUp;

    double resignScore = -initialBlackAdvantageInPoints * ((numTurnsToCatchUp - numTurnsSpent) / numTurnsToCatchUp);
    resignScore -= 5.0; //Always require at least a 5 point buffer
    resignScore -= initialBlackAdvantageInPoints * 0.15; //And also require a 15% of the initial handicap

    noResignationWhenWhiteScoreAbove = resignScore;
  }

  if(hist.moveHistory.size() < minTurnForResignation)
    return false;
  if(pla == P_WHITE && lead > noResignationWhenWhiteScoreAbove)
    return false;
  if(resignConsecTurns > recentWinLossValues.size())
    return false;
  //Don't resign close games.
  if((pla == P_WHITE && lead > -resignMinScoreDifference) || (pla == P_BLACK && lead < resignMinScoreDifference))
    return false;

  for(int i = 0; i<resignConsecTurns; i++) {
    double winLossValue = recentWinLossValues[recentWinLossValues.size()-1-i];
    Player resignPlayerThisTurn = C_EMPTY;
    if(winLossValue < resignThreshold)
      resignPlayerThisTurn = P_WHITE;
    else if(winLossValue > -resignThreshold)
      resignPlayerThisTurn = P_BLACK;

    if(resignPlayerThisTurn != pla)
      return false;
  }

  return true;
}

struct GTPEngine {
  GTPEngine(const GTPEngine&) = delete;
  GTPEngine& operator=(const GTPEngine&) = delete;

  const string nnModelFile;
  const bool assumeMultipleStartingBlackMovesAreHandicap;
  const int analysisPVLen;
  const bool preventEncore;

  const double dynamicPlayoutDoublingAdvantageCapPerOppLead;
  double staticPlayoutDoublingAdvantage;
  bool staticPDATakesPrecedence;
  double normalAvoidRepeatedPatternUtility;
  double handicapAvoidRepeatedPatternUtility;

  double genmoveWideRootNoise;
  double analysisWideRootNoise;
  bool genmoveAntiMirror;
  bool analysisAntiMirror;

  NNEvaluator* nnEval;
  AsyncBot* bot;
  Rules currentRules; //Should always be the same as the rules in bot, if bot is not NULL.

  //Stores the params we want to be using during genmoves or analysis
  SearchParams params;

  TimeControls bTimeControls;
  TimeControls wTimeControls;

  //This move history doesn't get cleared upon consecutive moves by the same side, and is used
  //for undo, whereas the one in search does.
  Board initialBoard;
  Player initialPla;
  vector<Move> moveHistory;

  vector<double> recentWinLossValues;
  double lastSearchFactor;
  double desiredDynamicPDAForWhite;
  bool avoidMYTDaggerHack;
  std::unique_ptr<PatternBonusTable> patternBonusTable;

  Player perspective;

  double genmoveTimeSum;

  GTPEngine(
    const string& modelFile, SearchParams initialParams, Rules initialRules,
    bool assumeMultiBlackHandicap, bool prevtEncore,
    double dynamicPDACapPerOppLead, double staticPDA, bool staticPDAPrecedence,
    double normAvoidRepeatedPatternUtility, double hcapAvoidRepeatedPatternUtility,
    bool avoidDagger,
    double genmoveWRN, double analysisWRN,
    bool genmoveAntiMir, bool analysisAntiMir,
    Player persp, int pvLen,
    std::unique_ptr<PatternBonusTable>&& pbTable
  )
    :nnModelFile(modelFile),
     assumeMultipleStartingBlackMovesAreHandicap(assumeMultiBlackHandicap),
     analysisPVLen(pvLen),
     preventEncore(prevtEncore),
     dynamicPlayoutDoublingAdvantageCapPerOppLead(dynamicPDACapPerOppLead),
     staticPlayoutDoublingAdvantage(staticPDA),
     staticPDATakesPrecedence(staticPDAPrecedence),
     normalAvoidRepeatedPatternUtility(normAvoidRepeatedPatternUtility),
     handicapAvoidRepeatedPatternUtility(hcapAvoidRepeatedPatternUtility),
     genmoveWideRootNoise(genmoveWRN),
     analysisWideRootNoise(analysisWRN),
     genmoveAntiMirror(genmoveAntiMir),
     analysisAntiMirror(analysisAntiMir),
     nnEval(NULL),
     bot(NULL),
     currentRules(initialRules),
     params(initialParams),
     bTimeControls(),
     wTimeControls(),
     initialBoard(),
     initialPla(P_BLACK),
     moveHistory(),
     recentWinLossValues(),
     lastSearchFactor(1.0),
     desiredDynamicPDAForWhite(0.0),
     avoidMYTDaggerHack(avoidDagger),
     patternBonusTable(std::move(pbTable)),
     perspective(persp),
     genmoveTimeSum(0.0)
  {
  }

  ~GTPEngine() {
    stopAndWait();
    delete bot;
    delete nnEval;
  }

  void stopAndWait() {
    bot->stopAndWait();
  }

  Rules getCurrentRules() {
    return currentRules;
  }

  void clearStatsForNewGame() {
    //Currently nothing
  }

  //Specify -1 for the sizes for a default
  void setOrResetBoardSize(ConfigParser& cfg, Logger& logger, Rand& seedRand, int boardXSize, int boardYSize) {
    if(nnEval != NULL && boardXSize == nnEval->getNNXLen() && boardYSize == nnEval->getNNYLen())
      return;
    if(nnEval != NULL) {
      assert(bot != NULL);
      bot->stopAndWait();
      delete bot;
      delete nnEval;
      bot = NULL;
      nnEval = NULL;
      logger.write("Cleaned up old neural net and bot");
    }

    bool wasDefault = false;
    if(boardXSize == -1 || boardYSize == -1) {
      boardXSize = Board::DEFAULT_LEN;
      boardYSize = Board::DEFAULT_LEN;
      wasDefault = true;
    }

    int maxConcurrentEvals = params.numThreads * 2 + 16; // * 2 + 16 just to give plenty of headroom
    int expectedConcurrentEvals = params.numThreads;
    int defaultMaxBatchSize = std::max(8,((params.numThreads+3)/4)*4);
    string expectedSha256 = "";
    nnEval = Setup::initializeNNEvaluator(
      nnModelFile,nnModelFile,expectedSha256,cfg,logger,seedRand,maxConcurrentEvals,expectedConcurrentEvals,
      boardXSize,boardYSize,defaultMaxBatchSize,
      Setup::SETUP_FOR_GTP
    );
    logger.write("Loaded neural net with nnXLen " + Global::intToString(nnEval->getNNXLen()) + " nnYLen " + Global::intToString(nnEval->getNNYLen()));

    {
      bool rulesWereSupported;
      nnEval->getSupportedRules(currentRules,rulesWereSupported);
      if(!rulesWereSupported) {
        throw StringError("Rules " + currentRules.toJsonStringNoKomi() + " from config file " + cfg.getFileName() + " are NOT supported by neural net");
      }
    }

    //On default setup, also override board size to whatever the neural net was initialized with
    //So that if the net was initalized smaller, we don't fail with a big board
    if(wasDefault) {
      boardXSize = nnEval->getNNXLen();
      boardYSize = nnEval->getNNYLen();
    }

    string searchRandSeed;
    if(cfg.contains("searchRandSeed"))
      searchRandSeed = cfg.getString("searchRandSeed");
    else
      searchRandSeed = Global::uint64ToString(seedRand.nextUInt64());

    bot = new AsyncBot(params, nnEval, &logger, searchRandSeed);
    bot->setCopyOfExternalPatternBonusTable(patternBonusTable);

    Board board(boardXSize,boardYSize);
    Player pla = P_BLACK;
    BoardHistory hist(board,pla,currentRules,0);
    vector<Move> newMoveHistory;
    setPositionAndRules(pla,board,hist,board,pla,newMoveHistory);
    clearStatsForNewGame();
  }

  void setPositionAndRules(Player pla, const Board& board, const BoardHistory& h, const Board& newInitialBoard, Player newInitialPla, const vector<Move> newMoveHistory) {
    BoardHistory hist(h);
    //Ensure we always have this value correct
    hist.setAssumeMultipleStartingBlackMovesAreHandicap(assumeMultipleStartingBlackMovesAreHandicap);

    currentRules = hist.rules;
    bot->setPosition(pla,board,hist);
    initialBoard = newInitialBoard;
    initialPla = newInitialPla;
    moveHistory = newMoveHistory;
    recentWinLossValues.clear();
    updateDynamicPDA();
  }

  void clearBoard() {
    assert(bot->getRootHist().rules == currentRules);
    int newXSize = bot->getRootBoard().x_size;
    int newYSize = bot->getRootBoard().y_size;
    Board board(newXSize,newYSize);
    Player pla = P_BLACK;
    BoardHistory hist(board,pla,currentRules,0);
    vector<Move> newMoveHistory;
    setPositionAndRules(pla,board,hist,board,pla,newMoveHistory);
    clearStatsForNewGame();
  }

  bool setPosition(const vector<Move>& initialStones) {
    assert(bot->getRootHist().rules == currentRules);
    int newXSize = bot->getRootBoard().x_size;
    int newYSize = bot->getRootBoard().y_size;
    Board board(newXSize,newYSize);
    for(int i = 0; i<initialStones.size(); i++) {
      if(!board.isOnBoard(initialStones[i].loc) || board.colors[initialStones[i].loc] != C_EMPTY) {
        return false;
      }
      bool suc = board.setStone(initialStones[i].loc, initialStones[i].pla);
      if(!suc) {
        return false;
      }
    }

    //Make sure nothing died along the way
    for(int i = 0; i<initialStones.size(); i++) {
      if(board.colors[initialStones[i].loc] != initialStones[i].pla) {
        return false;
      }
    }
    Player pla = P_BLACK;
    BoardHistory hist(board,pla,currentRules,0);
    hist.setInitialTurnNumber(board.numStonesOnBoard()); //Heuristic to guess at what turn this is
    vector<Move> newMoveHistory;
    setPositionAndRules(pla,board,hist,board,pla,newMoveHistory);
    clearStatsForNewGame();
    return true;
  }

  void updateKomiIfNew(float newKomi) {
    bot->setKomiIfNew(newKomi);
    currentRules.komi = newKomi;
  }

  void setStaticPlayoutDoublingAdvantage(double d) {
    staticPlayoutDoublingAdvantage = d;
    staticPDATakesPrecedence = true;
  }
  void setAnalysisWideRootNoise(double x) {
    analysisWideRootNoise = x;
  }
  void setRootPolicyTemperature(double x) {
    params.rootPolicyTemperature = x;
    bot->setParams(params);
    bot->clearSearch();
  }
  void setNumSearchThreads(int numThreads) {
    params.numThreads = numThreads;
    bot->setParams(params);
    bot->clearSearch();
  }

  void updateDynamicPDA() {
    updateDynamicPDAHelper(
      bot->getRootBoard(),bot->getRootHist(),
      dynamicPlayoutDoublingAdvantageCapPerOppLead,
      recentWinLossValues,
      desiredDynamicPDAForWhite
    );
  }

  bool play(Loc loc, Player pla) {
    assert(bot->getRootHist().rules == currentRules);
    bool suc = bot->makeMove(loc,pla,preventEncore);
    if(suc)
      moveHistory.push_back(Move(loc,pla));
    return suc;
  }

  bool undo() {
    if(moveHistory.size() <= 0)
      return false;
    assert(bot->getRootHist().rules == currentRules);

    vector<Move> moveHistoryCopy = moveHistory;

    Board undoneBoard = initialBoard;
    BoardHistory undoneHist(undoneBoard,initialPla,currentRules,0);
    undoneHist.setInitialTurnNumber(bot->getRootHist().initialTurnNumber);
    vector<Move> emptyMoveHistory;
    setPositionAndRules(initialPla,undoneBoard,undoneHist,initialBoard,initialPla,emptyMoveHistory);

    for(int i = 0; i<moveHistoryCopy.size()-1; i++) {
      Loc moveLoc = moveHistoryCopy[i].loc;
      Player movePla = moveHistoryCopy[i].pla;
      bool suc = play(moveLoc,movePla);
      assert(suc);
      (void)suc; //Avoid warning when asserts are off
    }
    return true;
  }

  bool setRulesNotIncludingKomi(Rules newRules, string& error) {
    assert(nnEval != NULL);
    assert(bot->getRootHist().rules == currentRules);
    newRules.komi = currentRules.komi;

    bool rulesWereSupported;
    nnEval->getSupportedRules(newRules,rulesWereSupported);
    if(!rulesWereSupported) {
      error = "Rules " + newRules.toJsonStringNoKomi() + " are not supported by this neural net version";
      return false;
    }

    vector<Move> moveHistoryCopy = moveHistory;

    Board board = initialBoard;
    BoardHistory hist(board,initialPla,newRules,0);
    hist.setInitialTurnNumber(bot->getRootHist().initialTurnNumber);
    vector<Move> emptyMoveHistory;
    setPositionAndRules(initialPla,board,hist,initialBoard,initialPla,emptyMoveHistory);

    for(int i = 0; i<moveHistoryCopy.size(); i++) {
      Loc moveLoc = moveHistoryCopy[i].loc;
      Player movePla = moveHistoryCopy[i].pla;
      bool suc = play(moveLoc,movePla);

      //Because internally we use a highly tolerant test, we don't expect this to actually trigger
      //even if a rules change did make some earlier moves illegal. But this check simply futureproofs
      //things in case we ever do
      if(!suc) {
        error = "Could not make the rules change, some earlier moves in the game would now become illegal.";
        return false;
      }
    }
    return true;
  }

  void ponder() {
    bot->ponder(lastSearchFactor);
  }

  struct AnalyzeArgs {
    bool analyzing = false;
    bool lz = false;
    bool kata = false;
    int minMoves = 0;
    int maxMoves = 10000000;
    bool showOwnership = false;
    bool showPVVisits = false;
    double secondsPerReport = TimeControls::UNLIMITED_TIME_DEFAULT;
    vector<int> avoidMoveUntilByLocBlack;
    vector<int> avoidMoveUntilByLocWhite;
  };

  void filterZeroVisitMoves(const AnalyzeArgs& args, vector<AnalysisData> buf) {
    //Avoid printing moves that have 0 visits, unless we need them
    //These should already be sorted so that 0-visit moves only appear at the end.
    int keptMoves = 0;
    for(int i = 0; i<buf.size(); i++) {
      if(buf[i].numVisits > 0 || keptMoves < args.minMoves)
        buf[keptMoves++] = buf[i];
    }
    buf.resize(keptMoves);
  }

  std::function<void(const Search* search)> getAnalyzeCallback(Player pla, AnalyzeArgs args) {
    std::function<void(const Search* search)> callback;
    //lz-analyze
    if(args.lz && !args.kata) {
      //Avoid capturing anything by reference except [this], since this will potentially be used
      //asynchronously and called after we return
      callback = [args,pla,this](const Search* search) {
        vector<AnalysisData> buf;
        bool duplicateForSymmetries = true;
        search->getAnalysisData(buf,args.minMoves,false,analysisPVLen,duplicateForSymmetries);
        filterZeroVisitMoves(args,buf);
        if(buf.size() > args.maxMoves)
          buf.resize(args.maxMoves);
        if(buf.size() <= 0)
          return;

        const Board board = search->getRootBoard();
        for(int i = 0; i<buf.size(); i++) {
          if(i > 0)
            cout << " ";
          const AnalysisData& data = buf[i];
          double winrate = 0.5 * (1.0 + data.winLossValue);
          double lcb = PlayUtils::getHackedLCBForWinrate(search,data,pla);
          if(perspective == P_BLACK || (perspective != P_BLACK && perspective != P_WHITE && pla == P_BLACK)) {
            winrate = 1.0-winrate;
            lcb = 1.0 - lcb;
          }
          cout << "info";
          cout << " move " << Location::toString(data.move,board);
          cout << " visits " << data.numVisits;
          cout << " winrate " << round(winrate * 10000.0);
          cout << " prior " << round(data.policyPrior * 10000.0);
          cout << " lcb " << round(lcb * 10000.0);
          cout << " order " << data.order;
          cout << " pv ";
          if(preventEncore && data.pvContainsPass())
            data.writePVUpToPhaseEnd(cout,board,search->getRootHist(),search->getRootPla());
          else
            data.writePV(cout,board);
          if(args.showPVVisits) {
            cout << " pvVisits ";
            if(preventEncore && data.pvContainsPass())
              data.writePVVisitsUpToPhaseEnd(cout,board,search->getRootHist(),search->getRootPla());
            else
              data.writePVVisits(cout);
          }
        }
        cout << endl;
      };
    }
    //kata-analyze, analyze (sabaki)
    else {
      callback = [args,pla,this](const Search* search) {
        vector<AnalysisData> buf;
        bool duplicateForSymmetries = true;
        search->getAnalysisData(buf,args.minMoves,false,analysisPVLen,duplicateForSymmetries);
        filterZeroVisitMoves(args,buf);
        if(buf.size() > args.maxMoves)
          buf.resize(args.maxMoves);
        if(buf.size() <= 0)
          return;

        vector<double> ownership;
        if(args.showOwnership) {
          static constexpr int64_t ownershipMinVisits = 3;
          ownership = search->getAverageTreeOwnership(ownershipMinVisits);
        }

        ostringstream out;
        if(!args.kata) {
          //Hack for sabaki - ensure always showing decimal point. Also causes output to be more verbose with trailing zeros,
          //unfortunately, despite doing not improving the precision of the values.
          out << std::showpoint;
        }

        const Board board = search->getRootBoard();
        for(int i = 0; i<buf.size(); i++) {
          if(i > 0)
            out << " ";
          const AnalysisData& data = buf[i];
          double winrate = 0.5 * (1.0 + data.winLossValue);
          double utility = data.utility;
          //We still hack the LCB for consistency with LZ-analyze
          double lcb = PlayUtils::getHackedLCBForWinrate(search,data,pla);
          ///But now we also offer the proper LCB that KataGo actually uses.
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
          out << "info";
          out << " move " << Location::toString(data.move,board);
          out << " visits " << data.numVisits;
          out << " utility " << utility;
          out << " winrate " << winrate;
          out << " scoreMean " << lead;
          out << " scoreStdev " << data.scoreStdev;
          out << " scoreLead " << lead;
          out << " scoreSelfplay " << scoreMean;
          out << " prior " << data.policyPrior;
          out << " lcb " << lcb;
          out << " utilityLcb " << utilityLcb;
          if(data.isSymmetryOf != Board::NULL_LOC)
            out << " isSymmetryOf " << Location::toString(data.isSymmetryOf,board);
          out << " order " << data.order;
          out << " pv ";
          if(preventEncore && data.pvContainsPass())
            data.writePVUpToPhaseEnd(out,board,search->getRootHist(),search->getRootPla());
          else
            data.writePV(out,board);
          if(args.showPVVisits) {
            out << " pvVisits ";
            if(preventEncore && data.pvContainsPass())
              data.writePVVisitsUpToPhaseEnd(out,board,search->getRootHist(),search->getRootPla());
            else
              data.writePVVisits(out);
          }
        }

        if(args.showOwnership) {
          out << " ";

          out << "ownership";
          int nnXLen = search->nnXLen;
          for(int y = 0; y<board.y_size; y++) {
            for(int x = 0; x<board.x_size; x++) {
              int pos = NNPos::xyToPos(x,y,nnXLen);
              if(perspective == P_BLACK || (perspective != P_BLACK && perspective != P_WHITE && pla == P_BLACK))
                out << " " << -ownership[pos];
              else
                out << " " << ownership[pos];
            }
          }
        }

        cout << out.str() << endl;
      };
    }
    return callback;
  }

  void genMove(
    Player pla,
    Logger& logger, double searchFactorWhenWinningThreshold, double searchFactorWhenWinning,
    enabled_t cleanupBeforePass, enabled_t friendlyPass, bool ogsChatToStderr,
    bool allowResignation, double resignThreshold, int resignConsecTurns, double resignMinScoreDifference,
    bool logSearchInfo, bool debug, bool playChosenMove,
    string& response, bool& responseIsError, bool& maybeStartPondering,
    AnalyzeArgs args
  ) {
    ClockTimer timer;

    response = "";
    responseIsError = false;
    maybeStartPondering = false;

    nnEval->clearStats();
    TimeControls tc = pla == P_BLACK ? bTimeControls : wTimeControls;

    //Update dynamic PDA given whatever the most recent values are, if we're using dynamic
    updateDynamicPDA();
    //Make sure we have the right parameters, in case someone ran analysis in the meantime.
    if(staticPDATakesPrecedence) {
      if(params.playoutDoublingAdvantage != staticPlayoutDoublingAdvantage) {
        params.playoutDoublingAdvantage = staticPlayoutDoublingAdvantage;
        bot->setParams(params);
      }
    }
    else {
      double desiredDynamicPDA =
        (params.playoutDoublingAdvantagePla == P_WHITE) ? desiredDynamicPDAForWhite :
        (params.playoutDoublingAdvantagePla == P_BLACK) ? -desiredDynamicPDAForWhite :
        (params.playoutDoublingAdvantagePla == C_EMPTY && pla == P_WHITE) ? desiredDynamicPDAForWhite :
        (params.playoutDoublingAdvantagePla == C_EMPTY && pla == P_BLACK) ? -desiredDynamicPDAForWhite :
        (assert(false),0.0);

      if(params.playoutDoublingAdvantage != desiredDynamicPDA) {
        params.playoutDoublingAdvantage = desiredDynamicPDA;
        bot->setParams(params);
      }
    }
    Player avoidMYTDaggerHackPla = avoidMYTDaggerHack ? pla : C_EMPTY;
    if(params.avoidMYTDaggerHackPla != avoidMYTDaggerHackPla) {
      params.avoidMYTDaggerHackPla = avoidMYTDaggerHackPla;
      bot->setParams(params);
    }
    if(params.wideRootNoise != genmoveWideRootNoise) {
      params.wideRootNoise = genmoveWideRootNoise;
      bot->setParams(params);
    }
    if(params.antiMirror != genmoveAntiMirror) {
      params.antiMirror = genmoveAntiMirror;
      bot->setParams(params);
    }

    {
      double avoidRepeatedPatternUtility = normalAvoidRepeatedPatternUtility;
      if(!args.analyzing) {
        double initialOppAdvantage = initialBlackAdvantage(bot->getRootHist()) * (pla == P_WHITE ? 1 : -1);
        if(initialOppAdvantage > getPointsThresholdForHandicapGame(getBoardSizeScaling(bot->getRootBoard())))
          avoidRepeatedPatternUtility = handicapAvoidRepeatedPatternUtility;
      }
      if(params.avoidRepeatedPatternUtility != avoidRepeatedPatternUtility) {
        params.avoidRepeatedPatternUtility = avoidRepeatedPatternUtility;
        bot->setParams(params);
      }
    }

    //Play faster when winning
    double searchFactor = PlayUtils::getSearchFactor(searchFactorWhenWinningThreshold,searchFactorWhenWinning,params,recentWinLossValues,pla);
    lastSearchFactor = searchFactor;

    Loc moveLoc;
    bot->setAvoidMoveUntilByLoc(args.avoidMoveUntilByLocBlack,args.avoidMoveUntilByLocWhite);
    if(args.analyzing) {
      std::function<void(const Search* search)> callback = getAnalyzeCallback(pla,args);
      if(args.showOwnership)
        bot->setAlwaysIncludeOwnerMap(true);
      else
        bot->setAlwaysIncludeOwnerMap(false);
      moveLoc = bot->genMoveSynchronousAnalyze(pla, tc, searchFactor, args.secondsPerReport, callback);
      //Make sure callback happens at least once
      callback(bot->getSearch());
    }
    else {
      moveLoc = bot->genMoveSynchronous(pla,tc,searchFactor);
    }

    bool isLegal = bot->isLegalStrict(moveLoc,pla);
    if(moveLoc == Board::NULL_LOC || !isLegal) {
      responseIsError = true;
      response = "genmove returned null location or illegal move";
      ostringstream sout;
      sout << "genmove null location or illegal move!?!" << "\n";
      sout << bot->getRootBoard() << "\n";
      sout << "Pla: " << PlayerIO::playerToString(pla) << "\n";
      sout << "MoveLoc: " << Location::toString(moveLoc,bot->getRootBoard()) << "\n";
      logger.write(sout.str());
      genmoveTimeSum += timer.getSeconds();
      return;
    }

    ReportedSearchValues values;
    double winLossValue;
    double lead;
    {
      values = bot->getSearch()->getRootValuesRequireSuccess();
      winLossValue = values.winLossValue;
      lead = values.lead;
    }

    //Record data for resignation or adjusting handicap behavior ------------------------
    recentWinLossValues.push_back(winLossValue);

    //Decide whether we should resign---------------------
    bool resigned = allowResignation && shouldResign(
      bot->getRootBoard(),bot->getRootHist(),pla,recentWinLossValues,lead,
      resignThreshold,resignConsecTurns,resignMinScoreDifference
    );


    //Snapshot the time NOW - all meaningful play-related computation time is done, the rest is just
    //output of various things.
    double timeTaken = timer.getSeconds();
    genmoveTimeSum += timeTaken;

    //Chatting and logging ----------------------------

    if(ogsChatToStderr) {
      int64_t visits = bot->getSearch()->getRootVisits();
      double winrate = 0.5 * (1.0 + (values.winValue - values.lossValue));
      double leadForPrinting = lead;
      //Print winrate from desired perspective
      if(perspective == P_BLACK || (perspective != P_BLACK && perspective != P_WHITE && pla == P_BLACK)) {
        winrate = 1.0 - winrate;
        leadForPrinting = -leadForPrinting;
      }
      cerr << "CHAT:"
           << "Visits " << visits
           << " Winrate " << Global::strprintf("%.2f%%", winrate * 100.0)
           << " ScoreLead " << Global::strprintf("%.1f", leadForPrinting)
           << " ScoreStdev " << Global::strprintf("%.1f", values.expectedScoreStdev);
      if(params.playoutDoublingAdvantage != 0.0) {
        cerr << Global::strprintf(
          " (PDA %.2f)",
          bot->getSearch()->getRootPla() == getOpp(params.playoutDoublingAdvantagePla) ?
          -params.playoutDoublingAdvantage : params.playoutDoublingAdvantage);
      }
      cerr << " PV ";
      bot->getSearch()->printPVForMove(cerr,bot->getSearch()->rootNode, moveLoc, analysisPVLen);
      cerr << endl;
    }

    if(logSearchInfo) {
      ostringstream sout;
      PlayUtils::printGenmoveLog(sout,bot,nnEval,moveLoc,timeTaken,perspective);
      logger.write(sout.str());
    }
    if(debug) {
      PlayUtils::printGenmoveLog(cerr,bot,nnEval,moveLoc,timeTaken,perspective);
    }

    //Hacks--------------------------------------------------
    //At least one of these hacks will use the bot to search stuff and clears its tree, so we apply them AFTER
    //all relevant logging and stuff.

    //Implement friendly pass - in area scoring rules other than tromp-taylor, maybe pass once there are no points
    //left to gain.
    int64_t numVisitsForFriendlyPass = 8 + std::min((int64_t)1000, std::min(params.maxVisits, params.maxPlayouts) / 10);
    moveLoc = PlayUtils::maybeFriendlyPass(cleanupBeforePass, friendlyPass, pla, moveLoc, bot->getSearchStopAndWait(), numVisitsForFriendlyPass);

    //Implement cleanupBeforePass hack - if the bot wants to pass, instead cleanup if there is something to clean
    //and we are in a ruleset where this is necessary or the user has configured it.
    moveLoc = PlayUtils::maybeCleanupBeforePass(cleanupBeforePass, friendlyPass, pla, moveLoc, bot);

    //Actual reporting of chosen move---------------------
    if(resigned)
      response = "resign";
    else
      response = Location::toString(moveLoc,bot->getRootBoard());

    if(!resigned && moveLoc != Board::NULL_LOC && isLegal && playChosenMove) {
      bool suc = bot->makeMove(moveLoc,pla,preventEncore);
      if(suc)
        moveHistory.push_back(Move(moveLoc,pla));
      assert(suc);
      (void)suc; //Avoid warning when asserts are off

      maybeStartPondering = true;
    }

    if(args.analyzing) {
      response = "play " + response;
    }

    return;
  }

  void clearCache() {
    bot->clearSearch();
    nnEval->clearCache();
  }

  void placeFixedHandicap(int n, string& response, bool& responseIsError) {
    int xSize = bot->getRootBoard().x_size;
    int ySize = bot->getRootBoard().y_size;
    Board board(xSize,ySize);
    try {
      PlayUtils::placeFixedHandicap(board,n);
    }
    catch(const StringError& e) {
      responseIsError = true;
      response = string(e.what()) + ", try place_free_handicap";
      return;
    }
    assert(bot->getRootHist().rules == currentRules);

    Player pla = P_BLACK;
    BoardHistory hist(board,pla,currentRules,0);

    //Also switch the initial player, expecting white should be next.
    hist.clear(board,P_WHITE,currentRules,0);
    hist.setAssumeMultipleStartingBlackMovesAreHandicap(assumeMultipleStartingBlackMovesAreHandicap);
    hist.setInitialTurnNumber(board.numStonesOnBoard()); //Should give more accurate temperaure and time control behavior
    pla = P_WHITE;

    response = "";
    for(int y = 0; y<board.y_size; y++) {
      for(int x = 0; x<board.x_size; x++) {
        Loc loc = Location::getLoc(x,y,board.x_size);
        if(board.colors[loc] != C_EMPTY) {
          response += " " + Location::toString(loc,board);
        }
      }
    }
    response = Global::trim(response);
    (void)responseIsError;

    vector<Move> newMoveHistory;
    setPositionAndRules(pla,board,hist,board,pla,newMoveHistory);
    clearStatsForNewGame();
  }

  void placeFreeHandicap(int n, string& response, bool& responseIsError, Rand& rand) {
    stopAndWait();

    //If asked to place more, we just go ahead and only place up to 30, or a quarter of the board
    int xSize = bot->getRootBoard().x_size;
    int ySize = bot->getRootBoard().y_size;
    int maxHandicap = xSize*ySize / 4;
    if(maxHandicap > 30)
      maxHandicap = 30;
    if(n > maxHandicap)
      n = maxHandicap;

    assert(bot->getRootHist().rules == currentRules);

    Board board(xSize,ySize);
    Player pla = P_BLACK;
    BoardHistory hist(board,pla,currentRules,0);
    double extraBlackTemperature = 0.25;
    PlayUtils::playExtraBlack(bot->getSearchStopAndWait(), n, board, hist, extraBlackTemperature, rand);
    //Also switch the initial player, expecting white should be next.
    hist.clear(board,P_WHITE,currentRules,0);
    hist.setAssumeMultipleStartingBlackMovesAreHandicap(assumeMultipleStartingBlackMovesAreHandicap);
    hist.setInitialTurnNumber(board.numStonesOnBoard()); //Should give more accurate temperaure and time control behavior
    pla = P_WHITE;

    response = "";
    for(int y = 0; y<board.y_size; y++) {
      for(int x = 0; x<board.x_size; x++) {
        Loc loc = Location::getLoc(x,y,board.x_size);
        if(board.colors[loc] != C_EMPTY) {
          response += " " + Location::toString(loc,board);
        }
      }
    }
    response = Global::trim(response);
    (void)responseIsError;

    vector<Move> newMoveHistory;
    setPositionAndRules(pla,board,hist,board,pla,newMoveHistory);
    clearStatsForNewGame();
  }

  void analyze(Player pla, AnalyzeArgs args) {
    assert(args.analyzing);
    //Analysis should ALWAYS be with the static value to prevent random hard-to-predict changes
    //for users.
    if(params.playoutDoublingAdvantage != staticPlayoutDoublingAdvantage) {
      params.playoutDoublingAdvantage = staticPlayoutDoublingAdvantage;
      bot->setParams(params);
    }
    if(params.avoidMYTDaggerHackPla != C_EMPTY) {
      params.avoidMYTDaggerHackPla = C_EMPTY;
      bot->setParams(params);
    }
    //Also wide root, if desired
    if(params.wideRootNoise != analysisWideRootNoise) {
      params.wideRootNoise = analysisWideRootNoise;
      bot->setParams(params);
    }
    if(params.antiMirror != analysisAntiMirror) {
      params.antiMirror = analysisAntiMirror;
      bot->setParams(params);
    }

    std::function<void(const Search* search)> callback = getAnalyzeCallback(pla,args);
    bot->setAvoidMoveUntilByLoc(args.avoidMoveUntilByLocBlack,args.avoidMoveUntilByLocWhite);
    if(args.showOwnership)
      bot->setAlwaysIncludeOwnerMap(true);
    else
      bot->setAlwaysIncludeOwnerMap(false);

    double searchFactor = 1e40; //go basically forever
    bot->analyzeAsync(pla, searchFactor, args.secondsPerReport, callback);
  }

  void computeAnticipatedWinnerAndScore(Player& winner, double& finalWhiteMinusBlackScore) {
    stopAndWait();

    //No playoutDoublingAdvantage to avoid bias
    //Also never assume the game will end abruptly due to pass
    {
      SearchParams tmpParams = params;
      tmpParams.playoutDoublingAdvantage = 0.0;
      tmpParams.conservativePass = true;
      bot->setParams(tmpParams);
    }

    //Make absolutely sure we can restore the bot's old state
    const Player oldPla = bot->getRootPla();
    const Board oldBoard = bot->getRootBoard();
    const BoardHistory oldHist = bot->getRootHist();

    Board board = bot->getRootBoard();
    BoardHistory hist = bot->getRootHist();
    Player pla = bot->getRootPla();

    //Tromp-taylorish scoring, or finished territory game scoring (including noresult)
    if(hist.isGameFinished && (
         (hist.rules.scoringRule == Rules::SCORING_AREA && !hist.rules.friendlyPassOk) ||
         (hist.rules.scoringRule == Rules::SCORING_TERRITORY)
       )
    ) {
      //For GTP purposes, we treat noResult as a draw since there is no provision for anything else.
      winner = hist.winner;
      finalWhiteMinusBlackScore = hist.finalWhiteMinusBlackScore;
    }
    //Human-friendly score or incomplete game score estimation
    else {
      int64_t numVisits = std::max(50, params.numThreads * 10);
      //Try computing the lead for white
      double lead = PlayUtils::computeLead(bot->getSearchStopAndWait(),NULL,board,hist,pla,numVisits,OtherGameProperties());

      //Round lead to nearest integer or half-integer
      if(hist.rules.gameResultWillBeInteger())
        lead = round(lead);
      else
        lead = round(lead+0.5)-0.5;

      finalWhiteMinusBlackScore = lead;
      winner = lead > 0 ? P_WHITE : lead < 0 ? P_BLACK : C_EMPTY;
    }

    //Restore
    bot->setPosition(oldPla,oldBoard,oldHist);
    bot->setParams(params);
  }

  vector<bool> computeAnticipatedStatuses() {
    stopAndWait();

    //Make absolutely sure we can restore the bot's old state
    const Player oldPla = bot->getRootPla();
    const Board oldBoard = bot->getRootBoard();
    const BoardHistory oldHist = bot->getRootHist();

    Board board = bot->getRootBoard();
    BoardHistory hist = bot->getRootHist();
    Player pla = bot->getRootPla();

    int64_t numVisits = std::max(100, params.numThreads * 20);
    vector<bool> isAlive;
    //Tromp-taylorish statuses, or finished territory game statuses (including noresult)
    if(hist.isGameFinished && (
         (hist.rules.scoringRule == Rules::SCORING_AREA && !hist.rules.friendlyPassOk) ||
         (hist.rules.scoringRule == Rules::SCORING_TERRITORY)
       )
    )
      isAlive = PlayUtils::computeAnticipatedStatusesSimple(board,hist);
    //Human-friendly statuses or incomplete game status estimation
    else {
      vector<double> ownershipsBuf;
      isAlive = PlayUtils::computeAnticipatedStatusesWithOwnership(bot->getSearchStopAndWait(),board,hist,pla,numVisits,ownershipsBuf);
    }

    //Restore
    bot->setPosition(oldPla,oldBoard,oldHist);

    return isAlive;
  }

  string rawNN(int whichSymmetry) {
    if(nnEval == NULL)
      return "";
    ostringstream out;

    for(int symmetry = 0; symmetry < SymmetryHelpers::NUM_SYMMETRIES; symmetry++) {
      if(whichSymmetry == NNInputs::SYMMETRY_ALL || whichSymmetry == symmetry) {
        Board board = bot->getRootBoard();
        BoardHistory hist = bot->getRootHist();
        Player nextPla = bot->getRootPla();

        MiscNNInputParams nnInputParams;
        nnInputParams.playoutDoublingAdvantage =
          (params.playoutDoublingAdvantagePla == C_EMPTY || params.playoutDoublingAdvantagePla == nextPla) ?
          staticPlayoutDoublingAdvantage : -staticPlayoutDoublingAdvantage;
        nnInputParams.symmetry = symmetry;
        NNResultBuf buf;
        bool skipCache = true;
        bool includeOwnerMap = true;
        nnEval->evaluate(board,hist,nextPla,nnInputParams,buf,skipCache,includeOwnerMap);

        NNOutput* nnOutput = buf.result.get();
        out << "symmetry " << symmetry << endl;
        out << "whiteWin " << Global::strprintf("%.6f",nnOutput->whiteWinProb) << endl;
        out << "whiteLoss " << Global::strprintf("%.6f",nnOutput->whiteLossProb) << endl;
        out << "noResult " << Global::strprintf("%.6f",nnOutput->whiteNoResultProb) << endl;
        out << "whiteLead " << Global::strprintf("%.3f",nnOutput->whiteLead) << endl;
        out << "whiteScoreSelfplay " << Global::strprintf("%.3f",nnOutput->whiteScoreMean) << endl;
        out << "whiteScoreSelfplaySq " << Global::strprintf("%.3f",nnOutput->whiteScoreMeanSq) << endl;
        out << "varTimeLeft " << Global::strprintf("%.3f",nnOutput->varTimeLeft) << endl;
        out << "shorttermWinlossError " << Global::strprintf("%.3f",nnOutput->shorttermWinlossError) << endl;
        out << "shorttermScoreError " << Global::strprintf("%.3f",nnOutput->shorttermScoreError) << endl;

        out << "policy" << endl;
        for(int y = 0; y<board.y_size; y++) {
          for(int x = 0; x<board.x_size; x++) {
            int pos = NNPos::xyToPos(x,y,nnOutput->nnXLen);
            float prob = nnOutput->policyProbs[pos];
            if(prob < 0)
              out << "    NAN ";
            else
              out << Global::strprintf("%8.6f ", prob);
          }
          out << endl;
        }
        out << "policyPass ";
        {
          int pos = NNPos::locToPos(Board::PASS_LOC,board.x_size,nnOutput->nnXLen,nnOutput->nnYLen);
          float prob = nnOutput->policyProbs[pos];
          if(prob < 0)
            out << "    NAN "; // Probably shouldn't ever happen for pass unles the rules change, but we handle it anyways
          else
            out << Global::strprintf("%8.6f ", prob);
          out << endl;
        }

        out << "whiteOwnership" << endl;
        for(int y = 0; y<board.y_size; y++) {
          for(int x = 0; x<board.x_size; x++) {
            int pos = NNPos::xyToPos(x,y,nnOutput->nnXLen);
            float whiteOwn = nnOutput->whiteOwnerMap[pos];
            out << Global::strprintf("%9.7f ", whiteOwn);
          }
          out << endl;
        }
        out << endl;
      }
    }

    return Global::trim(out.str());
  }

  SearchParams getParams() {
    return params;
  }

  void setParams(SearchParams p) {
    params = p;
    bot->setParams(params);
  }
};


//User should pre-fill pla with a default value, as it will not get filled in if the parsed command doesn't specify
static GTPEngine::AnalyzeArgs parseAnalyzeCommand(
  const string& command,
  const vector<string>& pieces,
  Player& pla,
  bool& parseFailed,
  GTPEngine* engine
) {
  int numArgsParsed = 0;

  bool isLZ = (command == "lz-analyze" || command == "lz-genmove_analyze");
  bool isKata = (command == "kata-analyze" || command == "kata-genmove_analyze");
  double lzAnalyzeInterval = TimeControls::UNLIMITED_TIME_DEFAULT;
  int minMoves = 0;
  int maxMoves = 10000000;
  bool showOwnership = false;
  bool showPVVisits = false;
  vector<int> avoidMoveUntilByLocBlack;
  vector<int> avoidMoveUntilByLocWhite;
  bool gotAvoidMovesBlack = false;
  bool gotAllowMovesBlack = false;
  bool gotAvoidMovesWhite = false;
  bool gotAllowMovesWhite = false;

  parseFailed = false;

  //Format:
  //lz-analyze [optional player] [optional interval float] <keys and values>
  //Keys and values consists of zero or more of:

  //interval <float interval in centiseconds>
  //avoid <player> <comma-separated moves> <until movenum>
  //minmoves <int min number of moves to show>
  //maxmoves <int max number of moves to show>
  //ownership <bool whether to show ownership or not>
  //pvVisits <bool whether to show pvVisits or not>

  //Parse optional player
  if(pieces.size() > numArgsParsed && PlayerIO::tryParsePlayer(pieces[numArgsParsed],pla))
    numArgsParsed += 1;

  //Parse optional interval float
  if(pieces.size() > numArgsParsed &&
     Global::tryStringToDouble(pieces[numArgsParsed],lzAnalyzeInterval) &&
     !isnan(lzAnalyzeInterval) && lzAnalyzeInterval >= 0 && lzAnalyzeInterval < TimeControls::MAX_USER_INPUT_TIME)
    numArgsParsed += 1;

  //Now loop and handle all key value pairs
  while(pieces.size() > numArgsParsed) {
    const string& key = pieces[numArgsParsed];
    numArgsParsed += 1;
    //Make sure we have a value. If not, then we fail.
    if(pieces.size() <= numArgsParsed) {
      parseFailed = true;
      break;
    }

    const string& value = pieces[numArgsParsed];
    numArgsParsed += 1;

    if(key == "interval" && Global::tryStringToDouble(value,lzAnalyzeInterval) &&
       !isnan(lzAnalyzeInterval) && lzAnalyzeInterval >= 0 && lzAnalyzeInterval < TimeControls::MAX_USER_INPUT_TIME) {
      continue;
    }
    else if(key == "avoid" || key == "allow") {
      //Parse two more arguments
      if(pieces.size() < numArgsParsed+2) {
        parseFailed = true;
        break;
      }
      const string& movesStr = pieces[numArgsParsed];
      numArgsParsed += 1;
      const string& untilDepthStr = pieces[numArgsParsed];
      numArgsParsed += 1;

      int untilDepth = -1;
      if(!Global::tryStringToInt(untilDepthStr,untilDepth) || untilDepth < 1) {
        parseFailed = true;
        break;
      }
      Player avoidPla = C_EMPTY;
      if(!PlayerIO::tryParsePlayer(value,avoidPla)) {
        parseFailed = true;
        break;
      }
      vector<Loc> parsedLocs;
      vector<string> locPieces = Global::split(movesStr,',');
      for(size_t i = 0; i<locPieces.size(); i++) {
        string s = Global::trim(locPieces[i]);
        if(s.size() <= 0)
          continue;
        Loc loc;
        if(!tryParseLoc(s,engine->bot->getRootBoard(),loc)) {
          parseFailed = true;
          break;
        }
        parsedLocs.push_back(loc);
      }
      if(parseFailed)
        break;

      //Make sure the same analyze command can't specify both avoid and allow, and allow at most one allow.
      vector<int>& avoidMoveUntilByLoc = avoidPla == P_BLACK ? avoidMoveUntilByLocBlack : avoidMoveUntilByLocWhite;
      bool& gotAvoidMoves = avoidPla == P_BLACK ? gotAvoidMovesBlack : gotAvoidMovesWhite;
      bool& gotAllowMoves = avoidPla == P_BLACK ? gotAllowMovesBlack : gotAllowMovesWhite;
      if((key == "allow" && gotAvoidMoves) || (key == "allow" && gotAllowMoves) || (key == "avoid" && gotAllowMoves)) {
        parseFailed = true;
        break;
      }
      avoidMoveUntilByLoc.resize(Board::MAX_ARR_SIZE);
      if(key == "allow") {
        std::fill(avoidMoveUntilByLoc.begin(),avoidMoveUntilByLoc.end(),untilDepth);
        for(Loc loc: parsedLocs) {
          avoidMoveUntilByLoc[loc] = 0;
        }
      }
      else {
        for(Loc loc: parsedLocs) {
          avoidMoveUntilByLoc[loc] = untilDepth;
        }
      }
      gotAvoidMoves |= (key == "avoid");
      gotAllowMoves |= (key == "allow");

      continue;
    }
    else if(key == "minmoves" && Global::tryStringToInt(value,minMoves) &&
            minMoves >= 0 && minMoves < 1000000000) {
      continue;
    }
    else if(key == "maxmoves" && Global::tryStringToInt(value,maxMoves) &&
            maxMoves >= 0 && maxMoves < 1000000000) {
      continue;
    }
    else if(isKata && key == "ownership" && Global::tryStringToBool(value,showOwnership)) {
      continue;
    }
    else if(isKata && key == "pvVisits" && Global::tryStringToBool(value,showPVVisits)) {
      continue;
    }

    parseFailed = true;
    break;
  }

  GTPEngine::AnalyzeArgs args = GTPEngine::AnalyzeArgs();
  args.analyzing = true;
  args.lz = isLZ;
  args.kata = isKata;
  //Convert from centiseconds to seconds
  args.secondsPerReport = lzAnalyzeInterval * 0.01;
  args.minMoves = minMoves;
  args.maxMoves = maxMoves;
  args.showOwnership = showOwnership;
  args.showPVVisits = showPVVisits;
  args.avoidMoveUntilByLocBlack = avoidMoveUntilByLocBlack;
  args.avoidMoveUntilByLocWhite = avoidMoveUntilByLocWhite;
  return args;
}


int MainCmds::gtp(const vector<string>& args) {
  Board::initHash();
  ScoreValue::initTables();
  Rand seedRand;

  ConfigParser cfg;
  string nnModelFile;
  string overrideVersion;
  KataGoCommandLine cmd("Run KataGo main GTP engine for playing games or casual analysis.");
  try {
    cmd.addConfigFileArg(KataGoCommandLine::defaultGtpConfigFileName(),"gtp_example.cfg");
    cmd.addModelFileArg();
    cmd.setShortUsageArgLimit();
    cmd.addOverrideConfigArg();

    TCLAP::ValueArg<string> overrideVersionArg("","override-version","Force KataGo to say a certain value in response to gtp version command",false,string(),"VERSION");
    cmd.add(overrideVersionArg);
    cmd.parseArgs(args);
    nnModelFile = cmd.getModelFile();
    overrideVersion = overrideVersionArg.getValue();

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

  const bool logAllGTPCommunication = cfg.getBool("logAllGTPCommunication");
  const bool logSearchInfo = cfg.getBool("logSearchInfo");
  bool loggingToStderr = false;

  const bool logTimeStamp = cfg.contains("logTimeStamp") ? cfg.getBool("logTimeStamp") : true;
  if(!logTimeStamp)
    logger.setLogTime(false);

  bool startupPrintMessageToStderr = true;
  if(cfg.contains("startupPrintMessageToStderr"))
    startupPrintMessageToStderr = cfg.getBool("startupPrintMessageToStderr");

  if(cfg.contains("logToStderr") && cfg.getBool("logToStderr")) {
    loggingToStderr = true;
    logger.setLogToStderr(true);
  }

  logger.write("GTP Engine starting...");
  logger.write(Version::getKataGoVersionForHelp());
  //Also check loggingToStderr so that we don't duplicate the message from the log file
  if(startupPrintMessageToStderr && !loggingToStderr) {
    cerr << Version::getKataGoVersionForHelp() << endl;
  }

  //Defaults to 7.5 komi, gtp will generally override this
  const bool loadKomiFromCfg = false;
  Rules initialRules = Setup::loadSingleRules(cfg,loadKomiFromCfg);
  logger.write("Using " + initialRules.toStringNoKomiMaybeNice() + " rules initially, unless GTP/GUI overrides this");
  if(startupPrintMessageToStderr && !loggingToStderr) {
    cerr << "Using " + initialRules.toStringNoKomiMaybeNice() + " rules initially, unless GTP/GUI overrides this" << endl;
  }
  bool isForcingKomi = false;
  float forcedKomi = 0;
  if(cfg.contains("ignoreGTPAndForceKomi")) {
    isForcingKomi = true;
    forcedKomi = cfg.getFloat("ignoreGTPAndForceKomi", Rules::MIN_USER_KOMI, Rules::MAX_USER_KOMI);
    initialRules.komi = forcedKomi;
  }

  SearchParams initialParams = Setup::loadSingleParams(cfg,Setup::SETUP_FOR_GTP);
  logger.write("Using " + Global::intToString(initialParams.numThreads) + " CPU thread(s) for search");
  //Set a default for conservativePass that differs from matches or selfplay
  if(!cfg.contains("conservativePass") && !cfg.contains("conservativePass0"))
    initialParams.conservativePass = true;
  if(!cfg.contains("fillDameBeforePass") && !cfg.contains("fillDameBeforePass0"))
    initialParams.fillDameBeforePass = true;

  const bool ponderingEnabled = cfg.getBool("ponderingEnabled");

  const enabled_t cleanupBeforePass = cfg.contains("cleanupBeforePass") ? cfg.getEnabled("cleanupBeforePass") : enabled_t::Auto;
  const enabled_t friendlyPass = cfg.contains("friendlyPass") ? cfg.getEnabled("friendlyPass") : enabled_t::Auto;
  if(cleanupBeforePass == enabled_t::True && friendlyPass == enabled_t::True)
    throw StringError("Cannot specify both cleanupBeforePass = true and friendlyPass = true at the same time");

  const bool allowResignation = cfg.contains("allowResignation") ? cfg.getBool("allowResignation") : false;
  const double resignThreshold = cfg.contains("allowResignation") ? cfg.getDouble("resignThreshold",-1.0,0.0) : -1.0; //Threshold on [-1,1], regardless of winLossUtilityFactor
  const int resignConsecTurns = cfg.contains("resignConsecTurns") ? cfg.getInt("resignConsecTurns",1,100) : 3;
  const double resignMinScoreDifference = cfg.contains("resignMinScoreDifference") ? cfg.getDouble("resignMinScoreDifference",0.0,1000.0) : -1e10;

  Setup::initializeSession(cfg);

  const double searchFactorWhenWinning = cfg.contains("searchFactorWhenWinning") ? cfg.getDouble("searchFactorWhenWinning",0.01,1.0) : 1.0;
  const double searchFactorWhenWinningThreshold = cfg.contains("searchFactorWhenWinningThreshold") ? cfg.getDouble("searchFactorWhenWinningThreshold",0.0,1.0) : 1.0;
  const bool ogsChatToStderr = cfg.contains("ogsChatToStderr") ? cfg.getBool("ogsChatToStderr") : false;
  const int analysisPVLen = cfg.contains("analysisPVLen") ? cfg.getInt("analysisPVLen",1,1000) : 13;
  const bool assumeMultipleStartingBlackMovesAreHandicap =
    cfg.contains("assumeMultipleStartingBlackMovesAreHandicap") ? cfg.getBool("assumeMultipleStartingBlackMovesAreHandicap") : true;
  const bool preventEncore = cfg.contains("preventCleanupPhase") ? cfg.getBool("preventCleanupPhase") : true;
  const double dynamicPlayoutDoublingAdvantageCapPerOppLead =
    cfg.contains("dynamicPlayoutDoublingAdvantageCapPerOppLead") ? cfg.getDouble("dynamicPlayoutDoublingAdvantageCapPerOppLead",0.0,0.5) : 0.045;
  double staticPlayoutDoublingAdvantage = initialParams.playoutDoublingAdvantage;
  const bool staticPDATakesPrecedence = cfg.contains("playoutDoublingAdvantage") && !cfg.contains("dynamicPlayoutDoublingAdvantageCapPerOppLead");
  const bool avoidMYTDaggerHack = cfg.contains("avoidMYTDaggerHack") ? cfg.getBool("avoidMYTDaggerHack") : false;
  const double normalAvoidRepeatedPatternUtility = initialParams.avoidRepeatedPatternUtility;
  const double handicapAvoidRepeatedPatternUtility = (cfg.contains("avoidRepeatedPatternUtility") || cfg.contains("avoidRepeatedPatternUtility0")) ?
    initialParams.avoidRepeatedPatternUtility : 0.005;

  const int defaultBoardXSize =
    cfg.contains("defaultBoardXSize") ? cfg.getInt("defaultBoardXSize",2,Board::MAX_LEN) :
    cfg.contains("defaultBoardSize") ? cfg.getInt("defaultBoardSize",2,Board::MAX_LEN) :
    -1;
  const int defaultBoardYSize =
    cfg.contains("defaultBoardYSize") ? cfg.getInt("defaultBoardYSize",2,Board::MAX_LEN) :
    cfg.contains("defaultBoardSize") ? cfg.getInt("defaultBoardSize",2,Board::MAX_LEN) :
    -1;
  const bool forDeterministicTesting =
    cfg.contains("forDeterministicTesting") ? cfg.getBool("forDeterministicTesting") : false;

  if(forDeterministicTesting)
    seedRand.init("forDeterministicTesting");

  const double genmoveWideRootNoise = initialParams.wideRootNoise;
  const double analysisWideRootNoise =
    cfg.contains("analysisWideRootNoise") ? cfg.getDouble("analysisWideRootNoise",0.0,5.0) : Setup::DEFAULT_ANALYSIS_WIDE_ROOT_NOISE;
  const bool analysisAntiMirror = initialParams.antiMirror;
  const bool genmoveAntiMirror =
    cfg.contains("genmoveAntiMirror") ? cfg.getBool("genmoveAntiMirror") : cfg.contains("antiMirror") ? cfg.getBool("antiMirror") : true;

  std::unique_ptr<PatternBonusTable> patternBonusTable = nullptr;
  {
    std::vector<std::unique_ptr<PatternBonusTable>> tables = Setup::loadAvoidSgfPatternBonusTables(cfg,logger);
    assert(tables.size() == 1);
    patternBonusTable = std::move(tables[0]);
  }

  Player perspective = Setup::parseReportAnalysisWinrates(cfg,C_EMPTY);

  GTPEngine* engine = new GTPEngine(
    nnModelFile,initialParams,initialRules,
    assumeMultipleStartingBlackMovesAreHandicap,preventEncore,
    dynamicPlayoutDoublingAdvantageCapPerOppLead,
    staticPlayoutDoublingAdvantage,staticPDATakesPrecedence,
    normalAvoidRepeatedPatternUtility, handicapAvoidRepeatedPatternUtility,
    avoidMYTDaggerHack,
    genmoveWideRootNoise,analysisWideRootNoise,
    genmoveAntiMirror,analysisAntiMirror,
    perspective,analysisPVLen,
    std::move(patternBonusTable)
  );
  engine->setOrResetBoardSize(cfg,logger,seedRand,defaultBoardXSize,defaultBoardYSize);

  //If nobody specified any time limit in any way, then assume a relatively fast time control
  if(!cfg.contains("maxPlayouts") && !cfg.contains("maxVisits") && !cfg.contains("maxTime")) {
    double mainTime = 1.0;
    double byoYomiTime = 5.0;
    int byoYomiPeriods = 5;
    TimeControls tc = TimeControls::canadianOrByoYomiTime(mainTime,byoYomiTime,byoYomiPeriods,1);
    engine->bTimeControls = tc;
    engine->wTimeControls = tc;
  }

  //Check for unused config keys
  cfg.warnUnusedKeys(cerr,&logger);

  logger.write("Loaded config " + cfg.getFileName());
  logger.write("Loaded model "+ nnModelFile);
  cmd.logOverrides(logger);
  logger.write("Model name: "+ (engine->nnEval == NULL ? string() : engine->nnEval->getInternalModelName()));
  logger.write("GTP ready, beginning main protocol loop");
  //Also check loggingToStderr so that we don't duplicate the message from the log file
  if(startupPrintMessageToStderr && !loggingToStderr) {
    cerr << "Loaded config " << cfg.getFileName() << endl;
    cerr << "Loaded model " << nnModelFile << endl;
    cerr << "Model name: "+ (engine->nnEval == NULL ? string() : engine->nnEval->getInternalModelName()) << endl;
    cerr << "GTP ready, beginning main protocol loop" << endl;
  }

  bool currentlyAnalyzing = false;
  string line;
  while(getline(cin,line)) {
    //Parse command, extracting out the command itself, the arguments, and any GTP id number for the command.
    string command;
    vector<string> pieces;
    bool hasId = false;
    int id = 0;
    {
      //Filter down to only "normal" ascii characters. Also excludes carrage returns.
      //Newlines are already handled by getline
      size_t newLen = 0;
      for(size_t i = 0; i < line.length(); i++)
        if(((int)line[i] >= 32 && (int)line[i] <= 126) || line[i] == '\t')
          line[newLen++] = line[i];

      line.erase(line.begin()+newLen, line.end());

      //Remove comments
      size_t commentPos = line.find("#");
      if(commentPos != string::npos)
        line = line.substr(0, commentPos);

      //Convert tabs to spaces
      for(size_t i = 0; i < line.length(); i++)
        if(line[i] == '\t')
          line[i] = ' ';

      line = Global::trim(line);

      //Upon any input line at all, stop any analysis and output a newline
      if(currentlyAnalyzing) {
        currentlyAnalyzing = false;
        engine->stopAndWait();
        cout << endl;
      }

      if(line.length() == 0)
        continue;

      if(logAllGTPCommunication)
        logger.write("Controller: " + line);

      //Parse id number of command, if present
      size_t digitPrefixLen = 0;
      while(digitPrefixLen < line.length() && Global::isDigit(line[digitPrefixLen]))
        digitPrefixLen++;
      if(digitPrefixLen > 0) {
        hasId = true;
        try {
          id = Global::parseDigits(line,0,digitPrefixLen);
        }
        catch(const IOError& e) {
          cout << "? GTP id '" << id << "' could not be parsed: " << e.what() << endl;
          continue;
        }
        line = line.substr(digitPrefixLen);
      }

      line = Global::trim(line);
      if(line.length() <= 0) {
        cout << "? empty command" << endl;
        continue;
      }

      pieces = Global::split(line,' ');
      for(size_t i = 0; i<pieces.size(); i++)
        pieces[i] = Global::trim(pieces[i]);
      assert(pieces.size() > 0);

      command = pieces[0];
      pieces.erase(pieces.begin());
    }

    bool responseIsError = false;
    bool suppressResponse = false;
    bool shouldQuitAfterResponse = false;
    bool maybeStartPondering = false;
    string response;

    if(command == "protocol_version") {
      response = "2";
    }

    else if(command == "name") {
      response = "KataGo";
    }

    else if(command == "version") {
      if(overrideVersion.size() > 0)
        response = overrideVersion;
      else
        response = Version::getKataGoVersion();
    }

    else if(command == "known_command") {
      if(pieces.size() != 1) {
        responseIsError = true;
        response = "Expected single argument for known_command but got '" + Global::concat(pieces," ") + "'";
      }
      else {
        if(std::find(knownCommands.begin(), knownCommands.end(), pieces[0]) != knownCommands.end())
          response = "true";
        else
          response = "false";
      }
    }

    else if(command == "list_commands") {
      for(size_t i = 0; i<knownCommands.size(); i++) {
        response += knownCommands[i];
        if(i < knownCommands.size()-1)
          response += "\n";
      }
    }

    else if(command == "quit") {
      shouldQuitAfterResponse = true;
      logger.write("Quit requested by controller");
    }

    else if(command == "boardsize" || command == "rectangular_boardsize") {
      int newXSize = 0;
      int newYSize = 0;
      bool suc = false;

      if(pieces.size() == 1) {
        if(contains(pieces[0],':')) {
          vector<string> subpieces = Global::split(pieces[0],':');
          if(subpieces.size() == 2 && Global::tryStringToInt(subpieces[0], newXSize) && Global::tryStringToInt(subpieces[1], newYSize))
            suc = true;
        }
        else {
          if(Global::tryStringToInt(pieces[0], newXSize)) {
            suc = true;
            newYSize = newXSize;
          }
        }
      }
      else if(pieces.size() == 2) {
        if(Global::tryStringToInt(pieces[0], newXSize) && Global::tryStringToInt(pieces[1], newYSize))
          suc = true;
      }

      if(!suc) {
        responseIsError = true;
        response = "Expected int argument for boardsize or pair of ints but got '" + Global::concat(pieces," ") + "'";
      }
      else if(newXSize < 2 || newYSize < 2) {
        responseIsError = true;
        response = "unacceptable size";
      }
      else if(newXSize > Board::MAX_LEN || newYSize > Board::MAX_LEN) {
        responseIsError = true;
        response = Global::strprintf("unacceptable size (Board::MAX_LEN is %d, consider increasing and recompiling)",(int)Board::MAX_LEN);
      }
      else {
        engine->setOrResetBoardSize(cfg,logger,seedRand,newXSize,newYSize);
      }
    }

    else if(command == "clear_board") {
      engine->clearBoard();
    }

    else if(command == "komi") {
      float newKomi = 0;
      if(pieces.size() != 1 || !Global::tryStringToFloat(pieces[0],newKomi)) {
        responseIsError = true;
        response = "Expected single float argument for komi but got '" + Global::concat(pieces," ") + "'";
      }
      //GTP spec says that we should accept any komi, but we're going to ignore that.
      else if(isnan(newKomi) || newKomi < Rules::MIN_USER_KOMI || newKomi > Rules::MAX_USER_KOMI) {
        responseIsError = true;
        response = "unacceptable komi";
      }
      else if(!Rules::komiIsIntOrHalfInt(newKomi)) {
        responseIsError = true;
        response = "komi must be an integer or half-integer";
      }
      else {
        if(isForcingKomi)
          newKomi = forcedKomi;
        engine->updateKomiIfNew(newKomi);
        //In case the controller tells us komi every move, restart pondering afterward.
        maybeStartPondering = engine->bot->getRootHist().moveHistory.size() > 0;
      }
    }

    else if(command == "get_komi") {
      response = Global::doubleToString(engine->getCurrentRules().komi);
    }

    else if(command == "kata-get-rules") {
      if(pieces.size() != 0) {
        response = "Expected no arguments for kata-get-rules but got '" + Global::concat(pieces," ") + "'";
      }
      else {
        response = engine->getCurrentRules().toJsonStringNoKomi();
      }
    }

    else if(command == "kata-set-rules") {
      string rest = Global::concat(pieces," ");
      bool parseSuccess = false;
      Rules newRules;
      try {
        newRules = Rules::parseRulesWithoutKomi(rest,engine->getCurrentRules().komi);
        parseSuccess = true;
      }
      catch(const StringError& err) {
        responseIsError = true;
        response = "Unknown rules '" + rest + "', " + err.what();
      }
      if(parseSuccess) {
        string error;
        bool suc = engine->setRulesNotIncludingKomi(newRules,error);
        if(!suc) {
          responseIsError = true;
          response = error;
        }
        logger.write("Changed rules to " + newRules.toStringNoKomiMaybeNice());
        if(!loggingToStderr)
          cerr << "Changed rules to " + newRules.toStringNoKomiMaybeNice() << endl;
      }
    }

    else if(command == "kata-set-rule") {
      if(pieces.size() != 2) {
        responseIsError = true;
        response = "Expected two arguments for kata-set-rule but got '" + Global::concat(pieces," ") + "'";
      }
      else {
        bool parseSuccess = false;
        Rules currentRules = engine->getCurrentRules();
        Rules newRules;
        try {
          newRules = Rules::updateRules(pieces[0], pieces[1], currentRules);
          parseSuccess = true;
        }
        catch(const StringError& err) {
          responseIsError = true;
          response = err.what();
        }
        if(parseSuccess) {
          string error;
          bool suc = engine->setRulesNotIncludingKomi(newRules,error);
          if(!suc) {
            responseIsError = true;
            response = error;
          }
          logger.write("Changed rules to " + newRules.toStringNoKomiMaybeNice());
          if(!loggingToStderr)
            cerr << "Changed rules to " + newRules.toStringNoKomiMaybeNice() << endl;
        }
      }
    }

    else if(command == "kgs-rules") {
      bool parseSuccess = false;
      Rules newRules;
      if(pieces.size() <= 0) {
        responseIsError = true;
        response = "Expected one argument kgs-rules";
      }
      else {
        string s = Global::toLower(Global::trim(pieces[0]));
        if(s == "chinese") {
          newRules = Rules::parseRulesWithoutKomi("chinese-kgs",engine->getCurrentRules().komi);
          parseSuccess = true;
        }
        else if(s == "aga") {
          newRules = Rules::parseRulesWithoutKomi("aga",engine->getCurrentRules().komi);
          parseSuccess = true;
        }
        else if(s == "new_zealand") {
          newRules = Rules::parseRulesWithoutKomi("new_zealand",engine->getCurrentRules().komi);
          parseSuccess = true;
        }
        else if(s == "japanese") {
          newRules = Rules::parseRulesWithoutKomi("japanese",engine->getCurrentRules().komi);
          parseSuccess = true;
        }
        else {
          responseIsError = true;
          response = "Unknown rules '" + s + "'";
        }
      }
      if(parseSuccess) {
        string error;
        bool suc = engine->setRulesNotIncludingKomi(newRules,error);
        if(!suc) {
          responseIsError = true;
          response = error;
        }
        logger.write("Changed rules to " + newRules.toStringNoKomiMaybeNice());
        if(!loggingToStderr)
          cerr << "Changed rules to " + newRules.toStringNoKomiMaybeNice() << endl;
      }
    }

    else if(command == "kata-list-params") {
      //For now, rootPolicyTemperature is hidden since it's not clear we want to support it
      response = "playoutDoublingAdvantage analysisWideRootNoise";
    }

    else if(command == "kata-get-param") {
      if(pieces.size() != 1) {
        responseIsError = true;
        response = "Expected one arguments for kata-get-param but got '" + Global::concat(pieces," ") + "'";
      }
      else {
        SearchParams params = engine->getParams();
        if(pieces[0] == "playoutDoublingAdvantage") {
          response = Global::doubleToString(engine->staticPlayoutDoublingAdvantage);
        }
        else if(pieces[0] == "rootPolicyTemperature") {
          response = Global::doubleToString(params.rootPolicyTemperature);
        }
        else if(pieces[0] == "analysisWideRootNoise") {
          response = Global::doubleToString(engine->analysisWideRootNoise);
        }
        else {
          responseIsError = true;
          response = "Invalid parameter";
        }
      }
    }

    else if(command == "kata-set-param") {
      if(pieces.size() != 2) {
        responseIsError = true;
        response = "Expected two arguments for kata-set-param but got '" + Global::concat(pieces," ") + "'";
      }
      else {
        int i;
        double d;
        if(pieces[0] == "playoutDoublingAdvantage") {
          if(Global::tryStringToDouble(pieces[1],d) && d >= -3.0 && d <= 3.0)
            engine->setStaticPlayoutDoublingAdvantage(d);
          else {
            responseIsError = true;
            response = "Invalid value for " + pieces[0] + ", must be float from -3.0 to 3.0";
          }
        }
        else if(pieces[0] == "rootPolicyTemperature") {
          if(Global::tryStringToDouble(pieces[1],d) && d >= 0.01 && d <= 100.0)
            engine->setRootPolicyTemperature(d);
          else {
            responseIsError = true;
            response = "Invalid value for " + pieces[0] + ", must be float from 0.01 to 100.0";
          }
        }
        else if(pieces[0] == "analysisWideRootNoise") {
          if(Global::tryStringToDouble(pieces[1],d) && d >= 0.0 && d <= 5.0)
            engine->setAnalysisWideRootNoise(d);
          else {
            responseIsError = true;
            response = "Invalid value for " + pieces[0] + ", must be float from 0.0 to 2.0";
          }
        }
        else if(pieces[0] == "numSearchThreads") {
          if(Global::tryStringToInt(pieces[1],i) && i >= 1 && i <= 1024)
            engine->setNumSearchThreads(i);
          else {
            responseIsError = true;
            response = "Invalid value for " + pieces[0] + ", must be integer from 1 to 1024";
          }
        }
        else {
          responseIsError = true;
          response = "Unknown or invalid parameter: " + pieces[0];
        }
      }
    }

    else if(command == "time_settings") {
      double mainTime;
      double byoYomiTime;
      int byoYomiStones;
      bool success = false;
      try {
        mainTime = parseTime(pieces,0,"main time");
        byoYomiTime = parseTime(pieces,1,"byo-yomi per-period time");
        byoYomiStones = parseByoYomiStones(pieces,2);
        success = true;
      }
      catch(const StringError& e) {
        responseIsError = true;
        response = e.what();
      }
      if(success) {
        TimeControls tc;
        //This means no time limits, according to gtp spec
        if(byoYomiStones == 0 && byoYomiTime > 0.0)
          tc = TimeControls();
        else if(byoYomiStones == 0)
          tc = TimeControls::absoluteTime(mainTime);
        else
          tc = TimeControls::canadianOrByoYomiTime(mainTime,byoYomiTime,1,byoYomiStones);
        engine->bTimeControls = tc;
        engine->wTimeControls = tc;
      }
    }

    else if(command == "kata-list_time_settings") {
      response = "none";
      response += " ";
      response += "absolute";
      response += " ";
      response += "byoyomi";
      response += " ";
      response += "canadian";
      response += " ";
      response += "fischer";
      response += " ";
      response += "fischer-capped";
    }

    else if(command == "kgs-time_settings" || command == "kata-time_settings") {
      if(pieces.size() < 1) {
        responseIsError = true;
        if(command == "kata-time_settings")
          response = "Expected 'none', 'absolute', 'byoyomi', 'canadian', 'fischer', or 'fischer-capped' as first argument for kata-time_settings";
        else
          response = "Expected 'none', 'absolute', 'byoyomi', or 'canadian' as first argument for kgs-time_settings";
      }
      else {
        string what = Global::toLower(Global::trim(pieces[0]));
        if(what == "none") {
          TimeControls tc = TimeControls();
          engine->bTimeControls = tc;
          engine->wTimeControls = tc;
        }
        else if(what == "absolute") {
          double mainTime;
          TimeControls tc;
          bool success = false;
          try {
            mainTime = parseTime(pieces,1,"main time");
            tc = TimeControls::absoluteTime(mainTime);
            success = true;
          }
          catch(const StringError& e) {
            responseIsError = true;
            response = e.what();
          }
          if(success) {
            engine->bTimeControls = tc;
            engine->wTimeControls = tc;
          }
        }
        else if(what == "canadian") {
          double mainTime;
          double byoYomiTime;
          int byoYomiStones;
          TimeControls tc;
          bool success = false;
          try {
            mainTime = parseTime(pieces,1,"main time");
            byoYomiTime = parseTime(pieces,2,"byo-yomi period time");
            byoYomiStones = parseByoYomiStones(pieces,3);
            //Use the same hack in time-settings - if somehow someone specifies positive overtime but 0 stones for it, intepret as no time control
            if(byoYomiStones == 0 && byoYomiTime > 0.0)
              tc = TimeControls();
            else if(byoYomiStones == 0)
              tc = TimeControls::absoluteTime(mainTime);
            else
              tc = TimeControls::canadianOrByoYomiTime(mainTime,byoYomiTime,1,byoYomiStones);
            success = true;
          }
          catch(const StringError& e) {
            responseIsError = true;
            response = e.what();
          }
          if(success) {
            engine->bTimeControls = tc;
            engine->wTimeControls = tc;
          }
        }
        else if(what == "byoyomi") {
          double mainTime;
          double byoYomiTime;
          int byoYomiPeriods;
          TimeControls tc;
          bool success = false;
          try {
            mainTime = parseTime(pieces,1,"main time");
            byoYomiTime = parseTime(pieces,2,"byo-yomi per-period time");
            byoYomiPeriods = parseByoYomiPeriods(pieces,3);
            if(byoYomiPeriods == 0)
              tc = TimeControls::absoluteTime(mainTime);
            else
              tc = TimeControls::canadianOrByoYomiTime(mainTime,byoYomiTime,byoYomiPeriods,1);
            success = true;
          }
          catch(const StringError& e) {
            responseIsError = true;
            response = e.what();
          }
          if(success) {
            engine->bTimeControls = tc;
            engine->wTimeControls = tc;
          }
        }
        else if(what == "fischer" && command == "kata-time_settings") {
          double mainTime;
          double increment;
          TimeControls tc;
          bool success = false;
          try {
            mainTime = parseTime(pieces,1,"main time");
            increment = parseTime(pieces,2,"increment time");
            tc = TimeControls::fischerTime(mainTime,increment);
            success = true;
          }
          catch(const StringError& e) {
            responseIsError = true;
            response = e.what();
          }
          if(success) {
            engine->bTimeControls = tc;
            engine->wTimeControls = tc;
          }
        }
        else if(what == "fischer-capped" && command == "kata-time_settings") {
          double mainTime;
          double increment;
          double mainTimeLimit;
          double maxTimePerMove;
          TimeControls tc;
          bool success = false;
          try {
            mainTime = parseTime(pieces,1,"main time");
            increment = parseTime(pieces,2,"increment time");
            mainTimeLimit = parseTimeAllowNegative(pieces,3,"main time limit");
            maxTimePerMove = parseTimeAllowNegative(pieces,4,"max time per move");
            if(mainTimeLimit < 0)
              mainTimeLimit = TimeControls::MAX_USER_INPUT_TIME;
            if(maxTimePerMove < 0)
              maxTimePerMove = TimeControls::MAX_USER_INPUT_TIME;
            tc = TimeControls::fischerCappedTime(mainTime,increment,mainTimeLimit,maxTimePerMove);
            success = true;
          }
          catch(const StringError& e) {
            responseIsError = true;
            response = e.what();
          }
          if(success) {
            engine->bTimeControls = tc;
            engine->wTimeControls = tc;
          }
        }
        else {
          responseIsError = true;
          if(command == "kata-time_settings")
            response = "Expected 'none', 'absolute', 'byoyomi', 'canadian', 'fischer', or 'fischer-capped' as first argument for kata-time_settings";
          else
            response = "Expected 'none', 'absolute', 'byoyomi', or 'canadian' as first argument for kgs-time_settings";
        }
      }
    }

    else if(command == "time_left") {
      Player pla;
      double time;
      int stones;
      if(pieces.size() != 3
         || !PlayerIO::tryParsePlayer(pieces[0],pla)
         || !Global::tryStringToDouble(pieces[1],time)
         || !Global::tryStringToInt(pieces[2],stones)
         ) {
        responseIsError = true;
        response = "Expected player and float time and int stones for time_left but got '" + Global::concat(pieces," ") + "'";
      }
      //Be slightly tolerant of negative time left
      else if(isnan(time) || time < -10.0 || time > TimeControls::MAX_USER_INPUT_TIME) {
        responseIsError = true;
        response = "invalid time";
      }
      else if(stones < 0 || stones > 100000) {
        responseIsError = true;
        response = "invalid stones";
      }
      else {
        TimeControls tc = pla == P_BLACK ? engine->bTimeControls : engine->wTimeControls;
        if(stones > 0 && tc.originalNumPeriods <= 0) {
          responseIsError = true;
          response = "stones left in period is > 0 but the time control used does not have any overtime periods";
        }
        else {
          //Main time
          if(stones == 0) {
            tc.mainTimeLeft = time;
            tc.inOvertime = false;
            tc.numPeriodsLeftIncludingCurrent = tc.originalNumPeriods;
            tc.numStonesLeftInPeriod = 0;
            tc.timeLeftInPeriod = 0;
          }
          else {
            //Hack for KGS byo-yomi - interpret num stones as periods instead
            if(tc.originalNumPeriods > 1 && tc.numStonesPerPeriod == 1) {
              tc.mainTimeLeft = 0.0;
              tc.inOvertime = true;
              tc.numPeriodsLeftIncludingCurrent = std::min(stones,tc.originalNumPeriods);
              tc.numStonesLeftInPeriod = 1;
              tc.timeLeftInPeriod = time;
            }
            //Normal canadian time interpertation of GTP
            else {
              tc.mainTimeLeft = 0.0;
              tc.inOvertime = true;
              tc.numPeriodsLeftIncludingCurrent = 1;
              tc.numStonesLeftInPeriod = std::min(stones,tc.numStonesPerPeriod);
              tc.timeLeftInPeriod = time;
            }
          }
          if(pla == P_BLACK)
            engine->bTimeControls = tc;
          else
            engine->wTimeControls = tc;

          //In case the controller tells us komi every move, restart pondering afterward.
          maybeStartPondering = engine->bot->getRootHist().moveHistory.size() > 0;
        }
      }
    }

    else if(command == "kata-debug-print-tc") {
      response += "Black "+ engine->bTimeControls.toDebugString(engine->bot->getRootBoard(),engine->bot->getRootHist(),initialParams.lagBuffer);
      response += "\n";
      response += "White "+ engine->wTimeControls.toDebugString(engine->bot->getRootBoard(),engine->bot->getRootHist(),initialParams.lagBuffer);
    }

    else if(command == "play") {
      Player pla;
      Loc loc;
      if(pieces.size() != 2) {
        responseIsError = true;
        response = "Expected two arguments for play but got '" + Global::concat(pieces," ") + "'";
      }
      else if(!PlayerIO::tryParsePlayer(pieces[0],pla)) {
        responseIsError = true;
        response = "Could not parse color: '" + pieces[0] + "'";
      }
      else if(!tryParseLoc(pieces[1],engine->bot->getRootBoard(),loc)) {
        responseIsError = true;
        response = "Could not parse vertex: '" + pieces[1] + "'";
      }
      else {
        bool suc = engine->play(loc,pla);
        if(!suc) {
          responseIsError = true;
          response = "illegal move";
        }
        maybeStartPondering = true;
      }
    }

    else if(command == "set_position") {
      if(pieces.size() % 2 != 0) {
        responseIsError = true;
        response = "Expected a space-separated sequence of <COLOR> <VERTEX> pairs but got '" + Global::concat(pieces," ") + "'";
      }
      else {
        vector<Move> initialStones;
        for(int i = 0; i<pieces.size(); i += 2) {
          Player pla;
          Loc loc;
          if(!PlayerIO::tryParsePlayer(pieces[i],pla)) {
            responseIsError = true;
            response = "Expected a space-separated sequence of <COLOR> <VERTEX> pairs but got '" + Global::concat(pieces," ") + "': ";
            response += "could not parse color: '" + pieces[0] + "'";
            break;
          }
          else if(!tryParseLoc(pieces[i+1],engine->bot->getRootBoard(),loc)) {
            responseIsError = true;
            response = "Expected a space-separated sequence of <COLOR> <VERTEX> pairs but got '" + Global::concat(pieces," ") + "': ";
            response += "Could not parse vertex: '" + pieces[1] + "'";
            break;
          }
          else if(loc == Board::PASS_LOC) {
            responseIsError = true;
            response = "Expected a space-separated sequence of <COLOR> <VERTEX> pairs but got '" + Global::concat(pieces," ") + "': ";
            response += "Could not parse vertex: '" + pieces[1] + "'";
            break;
          }
          initialStones.push_back(Move(loc,pla));
        }
        if(!responseIsError) {
          bool suc = engine->setPosition(initialStones);
          if(!suc) {
            responseIsError = true;
            response = "Illegal stone placements - overlapping stones or stones with no liberties?";
          }
          maybeStartPondering = false;
        }
      }
    }

    else if(command == "undo") {
      bool suc = engine->undo();
      if(!suc) {
        responseIsError = true;
        response = "cannot undo";
      }
    }

    else if(command == "genmove" || command == "genmove_debug" || command == "search_debug") {
      Player pla;
      if(pieces.size() != 1) {
        responseIsError = true;
        response = "Expected one argument for genmove but got '" + Global::concat(pieces," ") + "'";
      }
      else if(!PlayerIO::tryParsePlayer(pieces[0],pla)) {
        responseIsError = true;
        response = "Could not parse color: '" + pieces[0] + "'";
      }
      else {
        bool debug = command == "genmove_debug" || command == "search_debug";
        bool playChosenMove = command != "search_debug";

        engine->genMove(
          pla,
          logger,searchFactorWhenWinningThreshold,searchFactorWhenWinning,
          cleanupBeforePass,friendlyPass,ogsChatToStderr,
          allowResignation,resignThreshold,resignConsecTurns,resignMinScoreDifference,
          logSearchInfo,debug,playChosenMove,
          response,responseIsError,maybeStartPondering,
          GTPEngine::AnalyzeArgs()
        );
      }
    }

    else if(command == "genmove_analyze" || command == "lz-genmove_analyze" || command == "kata-genmove_analyze") {
      Player pla = engine->bot->getRootPla();
      bool parseFailed = false;
      GTPEngine::AnalyzeArgs args = parseAnalyzeCommand(command, pieces, pla, parseFailed, engine);
      if(parseFailed) {
        responseIsError = true;
        response = "Could not parse genmove_analyze arguments or arguments out of range: '" + Global::concat(pieces," ") + "'";
      }
      else {
        bool debug = false;
        bool playChosenMove = true;

        //Make sure the "equals" for GTP is printed out prior to the first analyze line, regardless of thread racing
        if(hasId)
          cout << "=" << Global::intToString(id) << endl;
        else
          cout << "=" << endl;

        engine->genMove(
          pla,
          logger,searchFactorWhenWinningThreshold,searchFactorWhenWinning,
          cleanupBeforePass,friendlyPass,ogsChatToStderr,
          allowResignation,resignThreshold,resignConsecTurns,resignMinScoreDifference,
          logSearchInfo,debug,playChosenMove,
          response,responseIsError,maybeStartPondering,
          args
        );
        //And manually handle the result as well. In case of error, don't report any play.
        suppressResponse = true;
        if(!responseIsError) {
          cout << response << endl;
          cout << endl;
        }
        else {
          cout << endl;
          if(!loggingToStderr)
            cerr << response << endl;
        }
      }
    }

    else if(command == "clear_cache") {
      engine->clearCache();
    }
    else if(command == "showboard") {
      ostringstream sout;
      engine->bot->getRootHist().printBasicInfo(sout, engine->bot->getRootBoard());
      //Filter out all double newlines, since double newline terminates GTP command responses
      string s = sout.str();
      string filtered;
      for(int i = 0; i<s.length(); i++) {
        if(i > 0 && s[i-1] == '\n' && s[i] == '\n')
          continue;
        filtered += s[i];
      }
      response = Global::trim(filtered);
    }

    else if(command == "fixed_handicap") {
      int n;
      if(pieces.size() != 1) {
        responseIsError = true;
        response = "Expected one argument for fixed_handicap but got '" + Global::concat(pieces," ") + "'";
      }
      else if(!Global::tryStringToInt(pieces[0],n)) {
        responseIsError = true;
        response = "Could not parse number of handicap stones: '" + pieces[0] + "'";
      }
      else if(n < 2) {
        responseIsError = true;
        response = "Number of handicap stones less than 2: '" + pieces[0] + "'";
      }
      else if(!engine->bot->getRootBoard().isEmpty()) {
        responseIsError = true;
        response = "Board is not empty";
      }
      else {
        engine->placeFixedHandicap(n,response,responseIsError);
      }
    }

    else if(command == "place_free_handicap") {
      int n;
      if(pieces.size() != 1) {
        responseIsError = true;
        response = "Expected one argument for place_free_handicap but got '" + Global::concat(pieces," ") + "'";
      }
      else if(!Global::tryStringToInt(pieces[0],n)) {
        responseIsError = true;
        response = "Could not parse number of handicap stones: '" + pieces[0] + "'";
      }
      else if(n < 2) {
        responseIsError = true;
        response = "Number of handicap stones less than 2: '" + pieces[0] + "'";
      }
      else if(!engine->bot->getRootBoard().isEmpty()) {
        responseIsError = true;
        response = "Board is not empty";
      }
      else {
        engine->placeFreeHandicap(n,response,responseIsError,seedRand);
      }
    }

    else if(command == "set_free_handicap") {
      if(!engine->bot->getRootBoard().isEmpty()) {
        responseIsError = true;
        response = "Board is not empty";
      }
      else {
        vector<Loc> locs;
        int xSize = engine->bot->getRootBoard().x_size;
        int ySize = engine->bot->getRootBoard().y_size;
        Board board(xSize,ySize);
        for(int i = 0; i<pieces.size(); i++) {
          Loc loc;
          bool suc = tryParseLoc(pieces[i],board,loc);
          if(!suc || loc == Board::PASS_LOC) {
            responseIsError = true;
            response = "Invalid handicap location: " + pieces[i];
          }
          locs.push_back(loc);
        }
        for(int i = 0; i<locs.size(); i++)
          board.setStone(locs[i],P_BLACK);

        Player pla = P_WHITE;
        BoardHistory hist(board,pla,engine->getCurrentRules(),0);
        hist.setInitialTurnNumber(board.numStonesOnBoard()); //Should give more accurate temperaure and time control behavior
        vector<Move> newMoveHistory;
        engine->setPositionAndRules(pla,board,hist,board,pla,newMoveHistory);
      }
    }

    else if(command == "final_score") {
      engine->stopAndWait();

      Player winner = C_EMPTY;
      double finalWhiteMinusBlackScore = 0.0;
      engine->computeAnticipatedWinnerAndScore(winner,finalWhiteMinusBlackScore);

      if(winner == C_EMPTY)
        response = "0";
      else if(winner == C_BLACK)
        response = "B+" + Global::strprintf("%.1f",-finalWhiteMinusBlackScore);
      else if(winner == C_WHITE)
        response = "W+" + Global::strprintf("%.1f",finalWhiteMinusBlackScore);
      else
        ASSERT_UNREACHABLE;
    }

    else if(command == "final_status_list") {
      int statusMode = 0;
      if(pieces.size() != 1) {
        responseIsError = true;
        response = "Expected one argument for final_status_list but got '" + Global::concat(pieces," ") + "'";
      }
      else {
        if(pieces[0] == "alive")
          statusMode = 0;
        else if(pieces[0] == "seki")
          statusMode = 1;
        else if(pieces[0] == "dead")
          statusMode = 2;
        else {
          responseIsError = true;
          response = "Argument to final_status_list must be 'alive' or 'seki' or 'dead'";
          statusMode = 3;
        }

        if(statusMode < 3) {
          vector<bool> isAlive = engine->computeAnticipatedStatuses();
          Board board = engine->bot->getRootBoard();
          vector<Loc> locsToReport;

          if(statusMode == 0) {
            for(int y = 0; y<board.y_size; y++) {
              for(int x = 0; x<board.x_size; x++) {
                Loc loc = Location::getLoc(x,y,board.x_size);
                if(board.colors[loc] != C_EMPTY && isAlive[loc])
                  locsToReport.push_back(loc);
              }
            }
          }
          if(statusMode == 2) {
            for(int y = 0; y<board.y_size; y++) {
              for(int x = 0; x<board.x_size; x++) {
                Loc loc = Location::getLoc(x,y,board.x_size);
                if(board.colors[loc] != C_EMPTY && !isAlive[loc])
                  locsToReport.push_back(loc);
              }
            }
          }

          response = "";
          for(int i = 0; i<locsToReport.size(); i++) {
            Loc loc = locsToReport[i];
            if(i > 0)
              response += " ";
            response += Location::toString(loc,board);
          }
        }
      }
    }

    else if(command == "loadsgf") {
      if(pieces.size() != 1 && pieces.size() != 2) {
        responseIsError = true;
        response = "Expected one or two arguments for loadsgf but got '" + Global::concat(pieces," ") + "'";
      }
      else {
        string filename = pieces[0];
        bool parseFailed = false;
        bool moveNumberSpecified = false;
        int moveNumber = 0;
        if(pieces.size() == 2) {
          bool suc = Global::tryStringToInt(pieces[1],moveNumber);
          if(!suc || moveNumber < 0 || moveNumber > 10000000)
            parseFailed = true;
          else {
            moveNumberSpecified = true;
          }
        }
        if(parseFailed) {
          responseIsError = true;
          response = "Invalid value for moveNumber for loadsgf";
        }
        else {
          Board sgfInitialBoard;
          Player sgfInitialNextPla;
          BoardHistory sgfInitialHist;
          Rules sgfRules;
          Board sgfBoard;
          Player sgfNextPla;
          BoardHistory sgfHist;

          bool sgfParseSuccess = false;
          CompactSgf* sgf = NULL;
          try {
            sgf = CompactSgf::loadFile(filename);

            if(sgf->moves.size() > 0x3FFFFFFF)
              throw StringError("Sgf has too many moves");
            if(!moveNumberSpecified || moveNumber > sgf->moves.size())
              moveNumber = (int)sgf->moves.size();

            sgfRules = sgf->getRulesOrWarn(
              engine->getCurrentRules(), //Use current rules as default
              [&logger](const string& msg) { logger.write(msg); cerr << msg << endl; }
            );
            if(engine->nnEval != NULL) {
              bool rulesWereSupported;
              Rules supportedRules = engine->nnEval->getSupportedRules(sgfRules,rulesWereSupported);
              if(!rulesWereSupported) {
                ostringstream out;
                out << "WARNING: Rules " << sgfRules.toJsonStringNoKomi()
                    << " from sgf not supported by neural net, using " << supportedRules.toJsonStringNoKomi() << " instead";
                logger.write(out.str());
                if(!loggingToStderr)
                  cerr << out.str() << endl;
                sgfRules = supportedRules;
              }
            }

            if(isForcingKomi)
              sgfRules.komi = forcedKomi;

            {
              //See if the rules differ, IGNORING komi differences
              Rules currentRules = engine->getCurrentRules();
              currentRules.komi = sgfRules.komi;
              if(sgfRules != currentRules) {
                ostringstream out;
                out << "Changing rules to " << sgfRules.toJsonStringNoKomi();
                logger.write(out.str());
                if(!loggingToStderr)
                  cerr << out.str() << endl;
              }
            }

            sgf->setupInitialBoardAndHist(sgfRules, sgfInitialBoard, sgfInitialNextPla, sgfInitialHist);
            sgfInitialHist.setInitialTurnNumber(sgfInitialBoard.numStonesOnBoard()); //Should give more accurate temperaure and time control behavior
            sgfBoard = sgfInitialBoard;
            sgfNextPla = sgfInitialNextPla;
            sgfHist = sgfInitialHist;
            sgf->playMovesTolerant(sgfBoard,sgfNextPla,sgfHist,moveNumber,preventEncore);

            delete sgf;
            sgf = NULL;
            sgfParseSuccess = true;
          }
          catch(const StringError& err) {
            delete sgf;
            sgf = NULL;
            responseIsError = true;
            response = "Could not load sgf: " + string(err.what());
          }
          catch(...) {
            delete sgf;
            sgf = NULL;
            responseIsError = true;
            response = "Cannot load file";
          }

          if(sgfParseSuccess) {
            if(sgfRules.komi != engine->getCurrentRules().komi) {
              ostringstream out;
              out << "Changing komi to " << sgfRules.komi;
              logger.write(out.str());
              if(!loggingToStderr)
                cerr << out.str() << endl;
            }
            engine->setPositionAndRules(sgfNextPla, sgfBoard, sgfHist, sgfInitialBoard, sgfInitialNextPla, sgfHist.moveHistory);
          }
        }
      }
    }

    else if(command == "printsgf") {
      if(pieces.size() != 0 && pieces.size() != 1) {
        responseIsError = true;
        response = "Expected zero or one argument for print but got '" + Global::concat(pieces," ") + "'";
      }
      else if(pieces.size() == 0 || pieces[0] == "-") {
        ostringstream out;
        WriteSgf::writeSgf(out,"","",engine->bot->getRootHist(),NULL,true,false);
        response = out.str();
      }
      else {
        ofstream out(pieces[0]);
        WriteSgf::writeSgf(out,"","",engine->bot->getRootHist(),NULL,true,false);
        out.close();
        response = "";
      }
    }

    else if(command == "analyze" || command == "lz-analyze" || command == "kata-analyze") {
      Player pla = engine->bot->getRootPla();
      bool parseFailed = false;
      GTPEngine::AnalyzeArgs args = parseAnalyzeCommand(command, pieces, pla, parseFailed, engine);

      if(parseFailed) {
        responseIsError = true;
        response = "Could not parse analyze arguments or arguments out of range: '" + Global::concat(pieces," ") + "'";
      }
      else {
        //Make sure the "equals" for GTP is printed out prior to the first analyze line, regardless of thread racing
        if(hasId)
          cout << "=" << Global::intToString(id) << endl;
        else
          cout << "=" << endl;

        engine->analyze(pla, args);

        //No response - currentlyAnalyzing will make sure we get a newline at the appropriate time, when stopped.
        suppressResponse = true;
        currentlyAnalyzing = true;
      }
    }

    else if(command == "kata-raw-nn") {
      int whichSymmetry = NNInputs::SYMMETRY_ALL;
      bool parsed = false;
      if(pieces.size() == 1) {
        string s = Global::trim(Global::toLower(pieces[0]));
        if(s == "all")
          parsed = true;
        else if(Global::tryStringToInt(s,whichSymmetry) && whichSymmetry >= 0 && whichSymmetry <= SymmetryHelpers::NUM_SYMMETRIES-1)
          parsed = true;
      }

      if(!parsed) {
        responseIsError = true;
        response = "Expected one argument 'all' or symmetry index [0-7] for kata-raw-nn but got '" + Global::concat(pieces," ") + "'";
      }
      else {
        response = engine->rawNN(whichSymmetry);
      }
    }


    else if(command == "cputime" || command == "gomill-cpu_time") {
      response = Global::doubleToString(engine->genmoveTimeSum);
    }

    else if(command == "stop") {
      //Stop any ongoing ponder or analysis
      engine->stopAndWait();
    }

    else {
      responseIsError = true;
      response = "unknown command";
    }


    //Postprocessing of response
    if(hasId)
      response = Global::intToString(id) + " " + response;
    else
      response = " " + response;

    if(responseIsError)
      response = "?" + response;
    else
      response = "=" + response;

    if(!suppressResponse) {
      cout << response << endl;
      cout << endl;
    }

    if(logAllGTPCommunication)
      logger.write(response);

    if(shouldQuitAfterResponse)
      break;

    if(maybeStartPondering && ponderingEnabled)
      engine->ponder();

  } //Close read loop

  delete engine;
  engine = NULL;
  NeuralNet::globalCleanup();
  ScoreValue::freeTables();

  logger.write("All cleaned up, quitting");
  return 0;
}
