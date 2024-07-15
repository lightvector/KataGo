#include "../core/global.h"
#include "../core/commandloop.h"
#include "../core/config_parser.h"
#include "../core/fileutils.h"
#include "../core/timer.h"
#include "../core/datetime.h"
#include "../core/makedir.h"
#include "../core/test.h"
#include "../dataio/sgf.h"
#include "../search/searchnode.h"
#include "../search/asyncbot.h"
#include "../search/patternbonustable.h"
#include "../program/setup.h"
#include "../program/playutils.h"
#include "../program/play.h"
#include "../tests/tests.h"
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
  "kata-get-models",
  "kata-get-param",
  "kata-set-param",
  "kata-list-params",
  "kgs-rules",

  "genmove",
  "kata-search", //Doesn't actually make the move
  "kata-search_cancellable", //Doesn't actually make the move, also any command or newline cancels the search.

  "genmove_debug", //Prints additional info to stderr
  "kata-search_debug", //Prints additional info to stderr, doesn't actually make the move

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
  "kata-search_analyze",  //Doesn't actually make the move
  "kata-search_analyze_cancellable",  //Doesn't actually make the move, also any command or newline cancels the search.
  // "analyze",
  "lz-analyze",
  "kata-analyze",

  //Display raw neural net evaluations
  "kata-raw-nn",
  "kata-raw-human-nn",

  //Misc other stuff
  "cputime",
  "gomill-cpu_time",
  "kata-benchmark",

  //Some debug commands
  "kata-debug-print-tc",
  "debug_moves",

  //Stop any ongoing ponder or analyze
  "stop",
};

static bool tryParseLoc(const string& s, const Board& b, Loc& loc) {
  return Location::tryOfString(s,b,loc);
}

//Filter out all double newlines, since double newline terminates GTP command responses
static string filterDoubleNewlines(const string& s) {
  string filtered;
  for(int i = 0; i<s.length(); i++) {
    if(i > 0 && s[i-1] == '\n' && s[i] == '\n')
      continue;
    filtered += s[i];
  }
  return filtered;
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
    double initialAdvantageInPoints = std::fabs(initialBlackAdvantageInPoints);
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
  const double resignMinScoreDifference,
  const double resignMinMovesPerBoardArea
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
  if(minTurnForResignation < resignMinMovesPerBoardArea * board.x_size * board.y_size)
    minTurnForResignation = (int)(resignMinMovesPerBoardArea * board.x_size * board.y_size);

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
  const string humanModelFile;
  const bool assumeMultipleStartingBlackMovesAreHandicap;
  const int analysisPVLen;
  const bool preventEncore;
  const bool autoAvoidPatterns;

  const double dynamicPlayoutDoublingAdvantageCapPerOppLead;
  bool staticPDATakesPrecedence;
  double normalAvoidRepeatedPatternUtility;
  double handicapAvoidRepeatedPatternUtility;

  NNEvaluator* nnEval;
  NNEvaluator* humanEval;
  AsyncBot* bot;
  Rules currentRules; //Should always be the same as the rules in bot, if bot is not NULL.

  //Stores the params we want to be using during genmoves or analysis
  SearchParams genmoveParams;
  SearchParams analysisParams;
  bool isGenmoveParams;

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
  std::unique_ptr<PatternBonusTable> patternBonusTable;

  double delayMoveScale;
  double delayMoveMax;

  Player perspective;

  Rand gtpRand;

  ClockTimer genmoveTimer;
  double genmoveTimeSum;
  std::atomic<int> genmoveExpectedId;

  //Positions during this game when genmove was called
  std::vector<Sgf::PositionSample> genmoveSamples;

  GTPEngine(
    const string& modelFile, const string& hModelFile,
    SearchParams initialGenmoveParams, SearchParams initialAnalysisParams,
    Rules initialRules,
    bool assumeMultiBlackHandicap, bool prevtEncore, bool autoPattern,
    double dynamicPDACapPerOppLead, bool staticPDAPrecedence,
    double normAvoidRepeatedPatternUtility, double hcapAvoidRepeatedPatternUtility,
    double delayScale, double delayMax,
    Player persp, int pvLen,
    std::unique_ptr<PatternBonusTable>&& pbTable
  )
    :nnModelFile(modelFile),
     humanModelFile(hModelFile),
     assumeMultipleStartingBlackMovesAreHandicap(assumeMultiBlackHandicap),
     analysisPVLen(pvLen),
     preventEncore(prevtEncore),
     autoAvoidPatterns(autoPattern),
     dynamicPlayoutDoublingAdvantageCapPerOppLead(dynamicPDACapPerOppLead),
     staticPDATakesPrecedence(staticPDAPrecedence),
     normalAvoidRepeatedPatternUtility(normAvoidRepeatedPatternUtility),
     handicapAvoidRepeatedPatternUtility(hcapAvoidRepeatedPatternUtility),
     nnEval(NULL),
     humanEval(NULL),
     bot(NULL),
     currentRules(initialRules),
     genmoveParams(initialGenmoveParams),
     analysisParams(initialAnalysisParams),
     isGenmoveParams(true),
     bTimeControls(),
     wTimeControls(),
     initialBoard(),
     initialPla(P_BLACK),
     moveHistory(),
     recentWinLossValues(),
     lastSearchFactor(1.0),
     desiredDynamicPDAForWhite(0.0),
     patternBonusTable(std::move(pbTable)),
     delayMoveScale(delayScale),
     delayMoveMax(delayMax),
     perspective(persp),
     gtpRand(),
     genmoveTimer(),
     genmoveTimeSum(0.0),
     genmoveExpectedId(0),
     genmoveSamples()
  {
  }

  ~GTPEngine() {
    stopAndWait();
    delete bot;
    delete nnEval;
    delete humanEval;
  }

  void stopAndWait() {
    // Invalidate any ongoing genmove
    int expectedSearchId = (genmoveExpectedId.load() + 1) & 0x3FFFFFFF;
    genmoveExpectedId.store(expectedSearchId);
    bot->stopAndWait();
  }

  Rules getCurrentRules() {
    return currentRules;
  }

  void clearStatsForNewGame() {
  }

  //Specify -1 for the sizes for a default
  void setOrResetBoardSize(ConfigParser& cfg, Logger& logger, Rand& seedRand, int boardXSize, int boardYSize, bool loggingToStderr) {
    bool wasDefault = false;
    if(boardXSize == -1 || boardYSize == -1) {
      boardXSize = Board::DEFAULT_LEN;
      boardYSize = Board::DEFAULT_LEN;
      wasDefault = true;
    }

    bool defaultRequireExactNNLen = true;
    int nnXLen = boardXSize;
    int nnYLen = boardYSize;

    if(cfg.contains("gtpForceMaxNNSize") && cfg.getBool("gtpForceMaxNNSize")) {
      defaultRequireExactNNLen = false;
      nnXLen = Board::MAX_LEN;
      nnYLen = Board::MAX_LEN;
    }

    //If the neural net is wrongly sized, we need to create or recreate it
    if(nnEval == NULL || !(nnXLen == nnEval->getNNXLen() && nnYLen == nnEval->getNNYLen())) {

      if(nnEval != NULL) {
        assert(bot != NULL);
        bot->stopAndWait();
        delete bot;
        delete nnEval;
        delete humanEval;
        bot = NULL;
        nnEval = NULL;
        humanEval = NULL;
        logger.write("Cleaned up old neural net and bot");
      }

      const int expectedConcurrentEvals = std::max(genmoveParams.numThreads, analysisParams.numThreads);
      const int defaultMaxBatchSize = std::max(8,((expectedConcurrentEvals+3)/4)*4);
      const bool disableFP16 = false;
      const string expectedSha256 = "";
      nnEval = Setup::initializeNNEvaluator(
        nnModelFile,nnModelFile,expectedSha256,cfg,logger,seedRand,expectedConcurrentEvals,
        nnXLen,nnYLen,defaultMaxBatchSize,defaultRequireExactNNLen,disableFP16,
        Setup::SETUP_FOR_GTP
      );
      logger.write("Loaded neural net with nnXLen " + Global::intToString(nnEval->getNNXLen()) + " nnYLen " + Global::intToString(nnEval->getNNYLen()));
      if(humanModelFile != "") {
        humanEval = Setup::initializeNNEvaluator(
          humanModelFile,humanModelFile,expectedSha256,cfg,logger,seedRand,expectedConcurrentEvals,
          nnXLen,nnYLen,defaultMaxBatchSize,defaultRequireExactNNLen,disableFP16,
          Setup::SETUP_FOR_GTP
        );
        logger.write("Loaded human SL net with nnXLen " + Global::intToString(humanEval->getNNXLen()) + " nnYLen " + Global::intToString(humanEval->getNNYLen()));
        if(!humanEval->requiresSGFMetadata()) {
          string warning;
          warning += "WARNING: Human model was not trained from SGF metadata to vary by rank! Did you pass the wrong model for -human-model?\n";
          logger.write(warning);
          if(!loggingToStderr)
            cerr << warning << endl;
        }
      }

      {
        bool rulesWereSupported;
        nnEval->getSupportedRules(currentRules,rulesWereSupported);
        if(!rulesWereSupported) {
          throw StringError("Rules " + currentRules.toJsonStringNoKomi() + " from config file " + cfg.getFileName() + " are NOT supported by neural net");
        }
      }
    }

    //On default setup, also override board size to whatever the neural net was initialized with
    //So that if the net was initalized smaller, we don't fail with a big board
    if(wasDefault) {
      boardXSize = nnEval->getNNXLen();
      boardYSize = nnEval->getNNYLen();
    }

    //If the bot is wrongly sized, we need to create or recreate the bot
    if(bot == NULL || bot->getRootBoard().x_size != boardXSize || bot->getRootBoard().y_size != boardYSize) {
      if(bot != NULL) {
        assert(bot != NULL);
        bot->stopAndWait();
        delete bot;
        bot = NULL;
        logger.write("Cleaned up old bot");
      }

      logger.write("Initializing board with boardXSize " + Global::intToString(boardXSize) + " boardYSize " + Global::intToString(boardYSize));
      if(!loggingToStderr)
        cerr << ("Initializing board with boardXSize " + Global::intToString(boardXSize) + " boardYSize " + Global::intToString(boardYSize)) << endl;

      string searchRandSeed;
      if(cfg.contains("searchRandSeed"))
        searchRandSeed = cfg.getString("searchRandSeed");
      else
        searchRandSeed = Global::uint64ToString(seedRand.nextUInt64());

      bot = new AsyncBot(genmoveParams, nnEval, humanEval, &logger, searchRandSeed);
      bot->setCopyOfExternalPatternBonusTable(patternBonusTable);

      Board board(boardXSize,boardYSize);
      Player pla = P_BLACK;
      BoardHistory hist(board,pla,currentRules,0);
      vector<Move> newMoveHistory;
      setPositionAndRules(pla,board,hist,board,pla,newMoveHistory);
      clearStatsForNewGame();
    }
  }

  void setPatternBonusTable(std::unique_ptr<PatternBonusTable>&& pbTable) {
    patternBonusTable = std::move(pbTable);
    if(bot != nullptr)
      bot->setCopyOfExternalPatternBonusTable(patternBonusTable);
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
    bool suc = board.setStonesFailIfNoLibs(initialStones);
    if(!suc)
      return false;

    //Sanity check
    for(int i = 0; i<initialStones.size(); i++) {
      if(board.colors[initialStones[i].loc] != initialStones[i].pla) {
        assert(false);
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

  struct GenmoveArgs {
    double searchFactorWhenWinningThreshold;
    double searchFactorWhenWinning;
    enabled_t cleanupBeforePass;
    enabled_t friendlyPass;
    bool ogsChatToStderr;
    bool allowResignation;
    double resignThreshold;
    int resignConsecTurns;
    double resignMinScoreDifference;
    double resignMinMovesPerBoardArea;
    bool logSearchInfo;
    bool logSearchInfoForChosenMove;
    bool debug;
  };

  struct AnalyzeArgs {
    bool analyzing = false;
    bool lz = false;
    bool kata = false;
    int minMoves = 0;
    int maxMoves = 10000000;
    bool showRootInfo = false;
    bool showOwnership = false;
    bool showOwnershipStdev = false;
    bool showMovesOwnership = false;
    bool showMovesOwnershipStdev = false;
    bool showPVVisits = false;
    bool showPVEdgeVisits = false;
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
          if(args.showPVEdgeVisits) {
            cout << " pvEdgeVisits ";
            if(preventEncore && data.pvContainsPass())
              data.writePVEdgeVisitsUpToPhaseEnd(cout,board,search->getRootHist(),search->getRootPla());
            else
              data.writePVEdgeVisits(cout);
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
        ReportedSearchValues rootVals;
        bool suc = search->getPrunedRootValues(rootVals);
        if(!suc)
          return;
        filterZeroVisitMoves(args,buf);
        if(buf.size() > args.maxMoves)
          buf.resize(args.maxMoves);
        if(buf.size() <= 0)
          return;
        const SearchNode* rootNode = search->getRootNode();

        vector<double> ownership, ownershipStdev;
        if(args.showOwnershipStdev) {
          tuple<vector<double>,vector<double>> ownershipAverageAndStdev;
          ownershipAverageAndStdev = search->getAverageAndStandardDeviationTreeOwnership();
          ownership = std::get<0>(ownershipAverageAndStdev);
          ownershipStdev = std::get<1>(ownershipAverageAndStdev);
        }
        else if(args.showOwnership) {
          ownership = search->getAverageTreeOwnership();
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
          // We report lead for scoreMean here so that a bunch of legacy tools that use KataGo use lead instead, which
          // is usually a better field for user applications. We report scoreMean instead as scoreSelfplay
          out << " scoreMean " << lead;
          out << " scoreStdev " << data.scoreStdev;
          out << " scoreLead " << lead;
          out << " scoreSelfplay " << scoreMean;
          out << " prior " << data.policyPrior;
          out << " lcb " << lcb;
          out << " utilityLcb " << utilityLcb;
          out << " weight " << data.weightSum;
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
          if(args.showPVEdgeVisits) {
            out << " pvEdgeVisits ";
            if(preventEncore && data.pvContainsPass())
              data.writePVEdgeVisitsUpToPhaseEnd(out,board,search->getRootHist(),search->getRootPla());
            else
              data.writePVEdgeVisits(out);
          }
          vector<double> movesOwnership, movesOwnershipStdev;
          if(args.showMovesOwnershipStdev) {
            tuple<vector<double>,vector<double>> movesOwnershipAverageAndStdev;
            movesOwnershipAverageAndStdev = search->getAverageAndStandardDeviationTreeOwnership(perspective,data.node,data.symmetry);
            movesOwnership = std::get<0>(movesOwnershipAverageAndStdev);
            movesOwnershipStdev = std::get<1>(movesOwnershipAverageAndStdev);

          }
          else if(args.showMovesOwnership) {
            movesOwnership = search->getAverageTreeOwnership(perspective,data.node,data.symmetry);
          }
          if(args.showMovesOwnership) {
            out << " ";

            out << "movesOwnership";
            int nnXLen = search->nnXLen;
            for(int y = 0; y<board.y_size; y++) {
              for(int x = 0; x<board.x_size; x++) {
                int pos = NNPos::xyToPos(x,y,nnXLen);
                out << " " << movesOwnership[pos]; // perspective already handled by getAverageAndStandardDeviationTreeOwnership
              }
            }
          }
          if(args.showMovesOwnershipStdev) {
            out << " ";

            out << "movesOwnershipStdev";
            int nnXLen = search->nnXLen;
            for(int y = 0; y<board.y_size; y++) {
              for(int x = 0; x<board.x_size; x++) {
                int pos = NNPos::xyToPos(x,y,nnXLen);
                out << " " << movesOwnershipStdev[pos];
              }
            }
          }
        }

        if(args.showRootInfo) {
          out << " rootInfo";
          double winrate = 0.5 * (1.0 + rootVals.winLossValue);
          double scoreMean = rootVals.expectedScore;
          double lead = rootVals.lead;
          double utility = rootVals.utility;
          if(perspective == P_BLACK || (perspective != P_BLACK && perspective != P_WHITE && pla == P_BLACK)) {
            winrate = 1.0 - winrate;
            scoreMean = -scoreMean;
            lead = -lead;
            utility = -utility;
          }
          out << " visits " << rootVals.visits;
          out << " utility " << utility;
          out << " winrate " << winrate;
          out << " scoreMean " << lead;
          out << " scoreStdev " << rootVals.expectedScoreStdev;
          out << " scoreLead " << lead;
          out << " scoreSelfplay " << scoreMean;
          out << " weight " << rootVals.weight;
          if(rootNode != NULL) {
            const NNOutput* nnOutput = rootNode->getNNOutput();
            if(nnOutput != NULL) {
              out << " rawStWrError " << nnOutput->shorttermWinlossError;
              out << " rawStScoreError " << nnOutput->shorttermScoreError;
              out << " rawVarTimeLeft " << nnOutput->varTimeLeft;
            }
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

        if(args.showOwnershipStdev) {
          out << " ";

          out << "ownershipStdev";
          int nnXLen = search->nnXLen;
          for(int y = 0; y<board.y_size; y++) {
            for(int x = 0; x<board.x_size; x++) {
              int pos = NNPos::xyToPos(x,y,nnXLen);
              out << " " << ownershipStdev[pos];
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
    Logger& logger,
    const GenmoveArgs& gargs,
    const AnalyzeArgs& args,
    bool playChosenMove,
    std::function<void(const string&, bool)> printGTPResponse,
    bool& maybeStartPondering
  ) {
    bool onMoveWasCalled = false;
    Loc genmoveMoveLoc = Board::NULL_LOC;
    auto onMove = [&genmoveMoveLoc,&onMoveWasCalled,this](Loc moveLoc, int searchId, Search* search) {
      (void)searchId;
      (void)search;
      onMoveWasCalled = true;
      genmoveMoveLoc = moveLoc;
    };
    launchGenMove(pla,gargs,args,onMove);
    bot->waitForSearchToEnd();
    testAssert(onMoveWasCalled);
    string response;
    bool responseIsError = false;
    Loc moveLocToPlay = Board::NULL_LOC;
    handleGenMoveResult(pla,bot->getSearchStopAndWait(),logger,gargs,args,genmoveMoveLoc,response,responseIsError,moveLocToPlay);
    printGTPResponse(response,responseIsError);
    if(moveLocToPlay != Board::NULL_LOC && playChosenMove) {
      bool suc = bot->makeMove(moveLocToPlay,pla,preventEncore);
      if(suc)
        moveHistory.push_back(Move(moveLocToPlay,pla));
      assert(suc);
      (void)suc; //Avoid warning when asserts are off

      maybeStartPondering = true;
    }
  }

  void genMoveCancellable(
    Player pla,
    Logger& logger,
    const GenmoveArgs& gargs,
    const AnalyzeArgs& args,
    std::function<void(const string&, bool)> printGTPResponse
  ) {
    // Make sure to capture things by value unless they're long-lived, since the callback needs to survive past the current scope.
    auto onMove = [pla,&logger,gargs,args,printGTPResponse,this](Loc moveLoc, int searchId, Search* search) {
      string response;
      bool responseIsError = false;
      Loc moveLocToPlay = Board::NULL_LOC;

      // Search invalidated before completion
      if(searchId != genmoveExpectedId.load()) {
        if(args.analyzing)
          response = "play cancelled";
        else
          response = "cancelled";
        printGTPResponse(response,responseIsError);
        return;
      }
      handleGenMoveResult(pla,search,logger,gargs,args,moveLoc,response,responseIsError,moveLocToPlay);
      printGTPResponse(response,responseIsError);
    };
    launchGenMove(pla,gargs,args,onMove);
  }

  void launchGenMove(
    Player pla,
    GenmoveArgs gargs,
    AnalyzeArgs args,
    std::function<void(Loc, int, Search*)> onMove
  ) {
    genmoveTimer.reset();

    nnEval->clearStats();
    if(humanEval != NULL)
      humanEval->clearStats();
    TimeControls tc = pla == P_BLACK ? bTimeControls : wTimeControls;

    if(!isGenmoveParams) {
      bot->setParams(genmoveParams);
      isGenmoveParams = true;
    }

    //Update dynamic PDA given whatever the most recent values are, if we're using dynamic
    updateDynamicPDA();

    SearchParams paramsToUse = genmoveParams;
    //Make sure we have the right parameters, in case someone updated params in the meantime.
    if(!staticPDATakesPrecedence) {
      double desiredDynamicPDA =
        (paramsToUse.playoutDoublingAdvantagePla == P_WHITE) ? desiredDynamicPDAForWhite :
        (paramsToUse.playoutDoublingAdvantagePla == P_BLACK) ? -desiredDynamicPDAForWhite :
        (paramsToUse.playoutDoublingAdvantagePla == C_EMPTY && pla == P_WHITE) ? desiredDynamicPDAForWhite :
        (paramsToUse.playoutDoublingAdvantagePla == C_EMPTY && pla == P_BLACK) ? -desiredDynamicPDAForWhite :
        (assert(false),0.0);

      paramsToUse.playoutDoublingAdvantage = desiredDynamicPDA;
    }

    {
      double avoidRepeatedPatternUtility = normalAvoidRepeatedPatternUtility;
      if(!args.analyzing) {
        double initialOppAdvantage = initialBlackAdvantage(bot->getRootHist()) * (pla == P_WHITE ? 1 : -1);
        if(initialOppAdvantage > getPointsThresholdForHandicapGame(getBoardSizeScaling(bot->getRootBoard())))
          avoidRepeatedPatternUtility = handicapAvoidRepeatedPatternUtility;
      }
      paramsToUse.avoidRepeatedPatternUtility = avoidRepeatedPatternUtility;
    }

    if(paramsToUse != bot->getParams())
      bot->setParams(paramsToUse);


    //Play faster when winning
    double searchFactor = PlayUtils::getSearchFactor(gargs.searchFactorWhenWinningThreshold,gargs.searchFactorWhenWinning,paramsToUse,recentWinLossValues,pla);
    lastSearchFactor = searchFactor;

    bot->setAvoidMoveUntilByLoc(args.avoidMoveUntilByLocBlack,args.avoidMoveUntilByLocWhite);

    //So that we can tell by the end of the search whether we still care for the result.
    int expectedSearchId = (genmoveExpectedId.load() + 1) & 0x3FFFFFFF;
    genmoveExpectedId.store(expectedSearchId);

    if(args.analyzing) {
      std::function<void(const Search* search)> callback = getAnalyzeCallback(pla,args);
      if(args.showOwnership || args.showOwnershipStdev || args.showMovesOwnership || args.showMovesOwnershipStdev)
        bot->setAlwaysIncludeOwnerMap(true);
      else
        bot->setAlwaysIncludeOwnerMap(false);

      //Make sure callback happens at least once
      auto onMoveWrapped = [onMove,callback](Loc moveLoc, int searchId, Search* search) {
        callback(search);
        onMove(moveLoc,searchId,search);
      };
      bot->genMoveAsyncAnalyze(pla, expectedSearchId, tc, searchFactor, onMoveWrapped, args.secondsPerReport, args.secondsPerReport, callback);
    }
    else {
      bot->genMoveAsync(pla,expectedSearchId,tc,searchFactor,onMove);
    }
  }

  void handleGenMoveResult(
    Player pla,
    Search* searchBot,
    Logger& logger,
    const GenmoveArgs& gargs,
    const AnalyzeArgs& args,
    Loc moveLoc,
    string& response, bool& responseIsError,
    Loc& moveLocToPlay
  ) {
    response = "";
    responseIsError = false;
    moveLocToPlay = Board::NULL_LOC;

    const Search* search = searchBot;

    bool isLegal = search->isLegalStrict(moveLoc,pla);
    if(moveLoc == Board::NULL_LOC || !isLegal) {
      responseIsError = true;
      response = "genmove returned null location or illegal move";
      ostringstream sout;
      sout << "genmove null location or illegal move!?!" << "\n";
      sout << search->getRootBoard() << "\n";
      sout << "Pla: " << PlayerIO::playerToString(pla) << "\n";
      sout << "MoveLoc: " << Location::toString(moveLoc,search->getRootBoard()) << "\n";
      logger.write(sout.str());
      genmoveTimeSum += genmoveTimer.getSeconds();
      return;
    }

    SearchNode* rootNode = search->rootNode;
    if(rootNode != NULL && delayMoveScale > 0.0 && delayMoveMax > 0.0) {
      int pos = search->getPos(moveLoc);
      const NNOutput* nnOutput = rootNode->getHumanOutput();
      nnOutput = nnOutput != NULL ? nnOutput : rootNode->getNNOutput();
      const float* policyProbs = nnOutput != NULL ? nnOutput->getPolicyProbsMaybeNoised() : NULL;
      if(policyProbs != NULL) {
        double prob = std::max(0.0,(double)policyProbs[pos]);
        double meanWait = 0.5 * delayMoveScale / (prob + 0.10);
        double waitTime = gtpRand.nextGamma(2.0) * meanWait / 2.0;
        waitTime = std::min(waitTime,delayMoveMax);
        waitTime = std::max(waitTime,0.0001);
        std::this_thread::sleep_for(std::chrono::duration<double>(waitTime));
      }
    }

    ReportedSearchValues values;
    double winLossValue;
    double lead;
    {
      values = search->getRootValuesRequireSuccess();
      winLossValue = values.winLossValue;
      lead = values.lead;
    }

    //Record data for resignation or adjusting handicap behavior ------------------------
    recentWinLossValues.push_back(winLossValue);

    //Decide whether we should resign---------------------
    bool resigned = gargs.allowResignation && shouldResign(
      search->getRootBoard(),search->getRootHist(),pla,recentWinLossValues,lead,
      gargs.resignThreshold,gargs.resignConsecTurns,gargs.resignMinScoreDifference,gargs.resignMinMovesPerBoardArea
    );

    //Snapshot the time NOW - all meaningful play-related computation time is done, the rest is just
    //output of various things.
    double timeTaken = genmoveTimer.getSeconds();
    genmoveTimeSum += timeTaken;

    //Chatting and logging ----------------------------

    const SearchParams& params = search->searchParams;

    if(gargs.ogsChatToStderr) {
      int64_t visits = search->getRootVisits();
      double winrate = 0.5 * (1.0 + (values.winValue - values.lossValue));
      double leadForPrinting = lead;
      //Print winrate from desired perspective
      if(perspective == P_BLACK || (perspective != P_BLACK && perspective != P_WHITE && pla == P_BLACK)) {
        winrate = 1.0 - winrate;
        leadForPrinting = -leadForPrinting;
      }
      cerr << "MALKOVICH:"
           << "Visits " << visits
           << " Winrate " << Global::strprintf("%.2f%%", winrate * 100.0)
           << " ScoreLead " << Global::strprintf("%.1f", leadForPrinting)
           << " ScoreStdev " << Global::strprintf("%.1f", values.expectedScoreStdev);
      if(params.playoutDoublingAdvantage != 0.0) {
        cerr << Global::strprintf(
          " (PDA %.2f)",
          search->getRootPla() == getOpp(params.playoutDoublingAdvantagePla) ?
          -params.playoutDoublingAdvantage : params.playoutDoublingAdvantage);
      }
      cerr << " PV ";
      search->printPVForMove(cerr,search->rootNode, moveLoc, analysisPVLen);
      cerr << endl;
    }

    if(gargs.logSearchInfo) {
      ostringstream sout;
      PlayUtils::printGenmoveLog(sout,search,nnEval,moveLoc,timeTaken,perspective,gargs.logSearchInfoForChosenMove);
      logger.write(sout.str());
    }
    if(gargs.debug) {
      PlayUtils::printGenmoveLog(cerr,search,nnEval,moveLoc,timeTaken,perspective,gargs.logSearchInfoForChosenMove);
    }

    //Hacks--------------------------------------------------
    //At least one of these hacks will use the bot to search stuff and clears its tree, so we apply them AFTER
    //all relevant logging and stuff.

    //Implement friendly pass - in area scoring rules other than tromp-taylor, maybe pass once there are no points
    //left to gain.
    int64_t numVisitsForFriendlyPass = 8 + std::min((int64_t)1000, std::min(params.maxVisits, params.maxPlayouts) / 10);
    moveLoc = PlayUtils::maybeFriendlyPass(gargs.cleanupBeforePass, gargs.friendlyPass, pla, moveLoc, searchBot, numVisitsForFriendlyPass);

    //Implement cleanupBeforePass hack - if the bot wants to pass, instead cleanup if there is something to clean
    //and we are in a ruleset where this is necessary or the user has configured it.
    moveLoc = PlayUtils::maybeCleanupBeforePass(gargs.cleanupBeforePass, gargs.friendlyPass, pla, moveLoc, bot);

    //Actual reporting of chosen move---------------------
    if(resigned)
      response = "resign";
    else
      response = Location::toString(moveLoc,search->getRootBoard());

    if(autoAvoidPatterns) {
      // Auto pattern expects moveless records using hintloc to contain the move.
      Sgf::PositionSample posSample;
      const BoardHistory& hist = search->getRootHist();
      posSample.board = search->getRootBoard();
      posSample.nextPla = pla;
      posSample.initialTurnNumber = hist.getCurrentTurnNumber();
      posSample.hintLoc = moveLoc;
      posSample.weight = 1.0;
      genmoveSamples.push_back(posSample);
    }

    if(!resigned) {
      moveLocToPlay = moveLoc;
    }

    if(args.analyzing) {
      response = "play " + response;
    }

    return;
  }

  void clearCache() {
    bot->clearSearch();
    nnEval->clearCache();
    if(humanEval != NULL)
      humanEval->clearCache();
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
    if(isGenmoveParams) {
      bot->setParams(analysisParams);
      isGenmoveParams = false;
    }

    std::function<void(const Search* search)> callback = getAnalyzeCallback(pla,args);
    bot->setAvoidMoveUntilByLoc(args.avoidMoveUntilByLocBlack,args.avoidMoveUntilByLocWhite);
    if(args.showOwnership || args.showOwnershipStdev || args.showMovesOwnership || args.showMovesOwnershipStdev)
      bot->setAlwaysIncludeOwnerMap(true);
    else
      bot->setAlwaysIncludeOwnerMap(false);

    double searchFactor = 1e40; //go basically forever
    bot->analyzeAsync(pla, searchFactor, args.secondsPerReport, args.secondsPerReport, callback);
  }

  void computeAnticipatedWinnerAndScore(Player& winner, double& finalWhiteMinusBlackScore) {
    stopAndWait();

    //No playoutDoublingAdvantage to avoid bias
    //Also never assume the game will end abruptly due to pass
    {
      SearchParams tmpParams = genmoveParams;
      tmpParams.playoutDoublingAdvantage = 0.0;
      tmpParams.conservativePass = true;
      tmpParams.humanSLChosenMoveProp = 0.0;
      tmpParams.humanSLRootExploreProbWeightful = 0.0;
      tmpParams.humanSLRootExploreProbWeightless = 0.0;
      tmpParams.humanSLPlaExploreProbWeightful = 0.0;
      tmpParams.humanSLPlaExploreProbWeightless = 0.0;
      tmpParams.humanSLOppExploreProbWeightful = 0.0;
      tmpParams.humanSLOppExploreProbWeightless = 0.0;
      tmpParams.antiMirror = false;
      tmpParams.avoidRepeatedPatternUtility = 0;
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
      int64_t numVisits = std::max(50, genmoveParams.numThreads * 10);
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
    bot->setParams(genmoveParams);
    isGenmoveParams = true;
  }

  vector<bool> computeAnticipatedStatuses() {
    stopAndWait();

    //No playoutDoublingAdvantage to avoid bias
    //Also never assume the game will end abruptly due to pass
    {
      SearchParams tmpParams = genmoveParams;
      tmpParams.playoutDoublingAdvantage = 0.0;
      tmpParams.conservativePass = true;
      tmpParams.humanSLChosenMoveProp = 0.0;
      tmpParams.humanSLRootExploreProbWeightful = 0.0;
      tmpParams.humanSLRootExploreProbWeightless = 0.0;
      tmpParams.humanSLPlaExploreProbWeightful = 0.0;
      tmpParams.humanSLPlaExploreProbWeightless = 0.0;
      tmpParams.humanSLOppExploreProbWeightful = 0.0;
      tmpParams.humanSLOppExploreProbWeightless = 0.0;
      tmpParams.antiMirror = false;
      tmpParams.avoidRepeatedPatternUtility = 0;
      bot->setParams(tmpParams);
    }

    //Make absolutely sure we can restore the bot's old state
    const Player oldPla = bot->getRootPla();
    const Board oldBoard = bot->getRootBoard();
    const BoardHistory oldHist = bot->getRootHist();

    Board board = bot->getRootBoard();
    BoardHistory hist = bot->getRootHist();
    Player pla = bot->getRootPla();

    int64_t numVisits = std::max(100, genmoveParams.numThreads * 20);
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
    bot->setParams(genmoveParams);
    isGenmoveParams = true;

    return isAlive;
  }

  string rawNNBrief(std::vector<Loc> branch, int whichSymmetry) {
    if(nnEval == NULL)
      return "";
    ostringstream out;

    Player pla = bot->getRootPla();
    Board board = bot->getRootBoard();
    BoardHistory hist = bot->getRootHist();

    Player prevPla = pla;
    Board prevBoard = board;
    BoardHistory prevHist = hist;
    Loc prevLoc = Board::NULL_LOC;

    for(Loc loc: branch) {
      prevPla = pla;
      prevBoard = board;
      prevHist = hist;
      prevLoc = loc;
      bool suc = hist.makeBoardMoveTolerant(board, loc, pla, false);
      if(!suc)
        return "illegal move sequence";
      pla = getOpp(pla);
    }

    string policyStr = "Policy: ";
    string wlStr = "White winloss: ";
    string leadStr = "White lead: ";

    for(int symmetry = 0; symmetry < SymmetryHelpers::NUM_SYMMETRIES; symmetry++) {
      if(whichSymmetry == NNInputs::SYMMETRY_ALL || whichSymmetry == symmetry) {
        {
          MiscNNInputParams nnInputParams;
          nnInputParams.playoutDoublingAdvantage =
            (analysisParams.playoutDoublingAdvantagePla == C_EMPTY || analysisParams.playoutDoublingAdvantagePla == pla) ?
            analysisParams.playoutDoublingAdvantage : -analysisParams.playoutDoublingAdvantage;
          nnInputParams.symmetry = symmetry;

          NNResultBuf buf;
          bool skipCache = true;
          bool includeOwnerMap = false;
          nnEval->evaluate(board,hist,pla,&analysisParams.humanSLProfile,nnInputParams,buf,skipCache,includeOwnerMap);

          NNOutput* nnOutput = buf.result.get();
          wlStr += Global::strprintf("%.2fc ", 100.0 * (nnOutput->whiteWinProb - nnOutput->whiteLossProb));
          leadStr += Global::strprintf("%.2f ", nnOutput->whiteLead);
        }
        if(prevLoc != Board::NULL_LOC) {
          MiscNNInputParams nnInputParams;
          nnInputParams.playoutDoublingAdvantage =
            (analysisParams.playoutDoublingAdvantagePla == C_EMPTY || analysisParams.playoutDoublingAdvantagePla == prevPla) ?
            analysisParams.playoutDoublingAdvantage : -analysisParams.playoutDoublingAdvantage;
          nnInputParams.symmetry = symmetry;

          NNResultBuf buf;
          bool skipCache = true;
          bool includeOwnerMap = false;
          nnEval->evaluate(prevBoard,prevHist,prevPla,&analysisParams.humanSLProfile,nnInputParams,buf,skipCache,includeOwnerMap);

          NNOutput* nnOutput = buf.result.get();
          int pos = NNPos::locToPos(prevLoc,board.x_size,nnOutput->nnXLen,nnOutput->nnYLen);
          policyStr += Global::strprintf("%.2f%% ", 100.0 * (nnOutput->policyProbs[pos]));
        }
      }
    }
    return Global::trim(policyStr + "\n" + wlStr + "\n" + leadStr);
  }

  string rawNN(int whichSymmetry, double policyOptimism, bool useHumanModel) {
    NNEvaluator* nnEvalToUse = useHumanModel ? humanEval : nnEval;
    if(nnEvalToUse == NULL)
      return "";
    ostringstream out;

    for(int symmetry = 0; symmetry < SymmetryHelpers::NUM_SYMMETRIES; symmetry++) {
      if(whichSymmetry == NNInputs::SYMMETRY_ALL || whichSymmetry == symmetry) {
        Board board = bot->getRootBoard();
        BoardHistory hist = bot->getRootHist();
        Player nextPla = bot->getRootPla();

        MiscNNInputParams nnInputParams;
        nnInputParams.playoutDoublingAdvantage =
          (analysisParams.playoutDoublingAdvantagePla == C_EMPTY || analysisParams.playoutDoublingAdvantagePla == nextPla) ?
          analysisParams.playoutDoublingAdvantage : -analysisParams.playoutDoublingAdvantage;
        nnInputParams.symmetry = symmetry;
        nnInputParams.policyOptimism = policyOptimism;
        NNResultBuf buf;
        bool skipCache = true;
        bool includeOwnerMap = true;
        nnEvalToUse->evaluate(board,hist,nextPla,&analysisParams.humanSLProfile,nnInputParams,buf,skipCache,includeOwnerMap);

        NNOutput* nnOutput = buf.result.get();
        out << "symmetry " << symmetry << endl;
        out << "whiteWin " << Global::strprintf("%.6f",nnOutput->whiteWinProb) << endl;
        out << "whiteLoss " << Global::strprintf("%.6f",nnOutput->whiteLossProb) << endl;
        out << "noResult " << Global::strprintf("%.6f",nnOutput->whiteNoResultProb) << endl;
        if(useHumanModel) {
          out << "whiteScore " << Global::strprintf("%.3f",nnOutput->whiteScoreMean) << endl;
          out << "whiteScoreSq " << Global::strprintf("%.3f",nnOutput->whiteScoreMeanSq) << endl;
        }
        else {
          out << "whiteLead " << Global::strprintf("%.3f",nnOutput->whiteLead) << endl;
          out << "whiteScoreSelfplay " << Global::strprintf("%.3f",nnOutput->whiteScoreMean) << endl;
          out << "whiteScoreSelfplaySq " << Global::strprintf("%.3f",nnOutput->whiteScoreMeanSq) << endl;
          out << "varTimeLeft " << Global::strprintf("%.3f",nnOutput->varTimeLeft) << endl;
        }
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

  const SearchParams& getGenmoveParams() {
    return genmoveParams;
  }

  void setGenmoveParamsIfChanged(const SearchParams& p) {
    if(genmoveParams != p) {
      genmoveParams = p;
      if(isGenmoveParams)
        bot->setParams(genmoveParams);
    }
  }

  const SearchParams& getAnalysisParams() {
    return analysisParams;
  }

  void setAnalysisParamsIfChanged(const SearchParams& p) {
    if(analysisParams != p) {
      analysisParams = p;
      if(!isGenmoveParams)
        bot->setParams(analysisParams);
    }
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
  bool isKata = (command == "kata-analyze" || command == "kata-genmove_analyze" || command == "kata-search_analyze" || command == "kata-search_analyze_cancellable");
  double lzAnalyzeInterval = TimeControls::UNLIMITED_TIME_DEFAULT;
  int minMoves = 0;
  int maxMoves = 10000000;
  bool showRootInfo = false;
  bool showOwnership = false;
  bool showOwnershipStdev = false;
  bool showMovesOwnership = false;
  bool showMovesOwnershipStdev = false;
  bool showPVVisits = false;
  bool showPVEdgeVisits = false;
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
  //ownershipStdev <bool whether to show ownershipStdev or not>
  //pvVisits <bool whether to show pvVisits or not>
  //pvEdgeVisits <bool whether to show pvEdgeVisits or not>

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
    else if(isKata && key == "rootInfo" && Global::tryStringToBool(value,showRootInfo)) {
      continue;
    }
    else if(isKata && key == "ownership" && Global::tryStringToBool(value,showOwnership)) {
      continue;
    }
    else if(isKata && key == "ownershipStdev" && Global::tryStringToBool(value,showOwnershipStdev)) {
      continue;
    }
    else if(isKata && key == "movesOwnership" && Global::tryStringToBool(value,showMovesOwnership)) {
      continue;
    }
    else if(isKata && key == "movesOwnershipStdev" && Global::tryStringToBool(value,showMovesOwnershipStdev)) {
      continue;
    }
    else if(isKata && key == "pvVisits" && Global::tryStringToBool(value,showPVVisits)) {
      continue;
    }
    else if(isKata && key == "pvEdgeVisits" && Global::tryStringToBool(value,showPVEdgeVisits)) {
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
  args.showRootInfo = showRootInfo;
  args.showOwnership = showOwnership;
  args.showOwnershipStdev = showOwnershipStdev;
  args.showMovesOwnership = showMovesOwnership;
  args.showMovesOwnershipStdev = showMovesOwnershipStdev;
  args.showPVVisits = showPVVisits;
  args.showPVEdgeVisits = showPVEdgeVisits;
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
  string humanModelFile;
  string overrideVersion;
  KataGoCommandLine cmd("Run KataGo main GTP engine for playing games or casual analysis.");
  try {
    cmd.addConfigFileArg(KataGoCommandLine::defaultGtpConfigFileName(),"gtp_example.cfg");
    cmd.addModelFileArg();
    cmd.addHumanModelFileArg();
    cmd.setShortUsageArgLimit();
    cmd.addOverrideConfigArg();

    TCLAP::ValueArg<string> overrideVersionArg("","override-version","Force KataGo to say a certain value in response to gtp version command",false,string(),"VERSION");
    cmd.add(overrideVersionArg);
    cmd.parseArgs(args);
    nnModelFile = cmd.getModelFile();
    humanModelFile = cmd.getHumanModelFile();
    overrideVersion = overrideVersionArg.getValue();

    cmd.getConfig(cfg);
  }
  catch (TCLAP::ArgException &e) {
    cerr << "Error: " << e.error() << " for argument " << e.argId() << endl;
    return 1;
  }

  Logger logger(&cfg);

  const bool logAllGTPCommunication = cfg.getBool("logAllGTPCommunication");
  const bool logSearchInfo = cfg.getBool("logSearchInfo");
  const bool logSearchInfoForChosenMove = cfg.contains("logSearchInfoForChosenMove") ? cfg.getBool("logSearchInfoForChosenMove") : false;

  bool startupPrintMessageToStderr = true;
  if(cfg.contains("startupPrintMessageToStderr"))
    startupPrintMessageToStderr = cfg.getBool("startupPrintMessageToStderr");

  logger.write("GTP Engine starting...");
  logger.write(Version::getKataGoVersionForHelp());
  //Also check loggingToStderr so that we don't duplicate the message from the log file
  if(startupPrintMessageToStderr && !logger.isLoggingToStderr()) {
    cerr << Version::getKataGoVersionForHelp() << endl;
  }

  //Defaults to 7.5 komi, gtp will generally override this
  const bool loadKomiFromCfg = false;
  Rules initialRules = Setup::loadSingleRules(cfg,loadKomiFromCfg);
  logger.write("Using " + initialRules.toStringNoKomiMaybeNice() + " rules initially, unless GTP/GUI overrides this");
  if(startupPrintMessageToStderr && !logger.isLoggingToStderr()) {
    cerr << "Using " + initialRules.toStringNoKomiMaybeNice() + " rules initially, unless GTP/GUI overrides this" << endl;
  }
  bool isForcingKomi = false;
  float forcedKomi = 0;
  if(cfg.contains("ignoreGTPAndForceKomi")) {
    isForcingKomi = true;
    forcedKomi = cfg.getFloat("ignoreGTPAndForceKomi", Rules::MIN_USER_KOMI, Rules::MAX_USER_KOMI);
    initialRules.komi = forcedKomi;
  }

  const bool hasHumanModel = humanModelFile != "";

  auto loadParams = [&hasHumanModel](ConfigParser& config, SearchParams& genmoveOut, SearchParams& analysisOut) {
    SearchParams params = Setup::loadSingleParams(config,Setup::SETUP_FOR_GTP,hasHumanModel);
    //Set a default for conservativePass that differs from matches or selfplay
    if(!config.contains("conservativePass"))
      params.conservativePass = true;
    if(!config.contains("fillDameBeforePass"))
      params.fillDameBeforePass = true;

    const double analysisWideRootNoise =
      config.contains("analysisWideRootNoise") ? config.getDouble("analysisWideRootNoise",0.0,5.0) : Setup::DEFAULT_ANALYSIS_WIDE_ROOT_NOISE;
    const double analysisIgnorePreRootHistory =
      config.contains("analysisIgnorePreRootHistory") ? config.getBool("analysisIgnorePreRootHistory") : Setup::DEFAULT_ANALYSIS_IGNORE_PRE_ROOT_HISTORY;
    const bool genmoveAntiMirror =
      config.contains("genmoveAntiMirror") ? config.getBool("genmoveAntiMirror") : config.contains("antiMirror") ? config.getBool("antiMirror") : true;

    genmoveOut = params;
    analysisOut = params;

    genmoveOut.antiMirror = genmoveAntiMirror;
    analysisOut.wideRootNoise = analysisWideRootNoise;
    analysisOut.ignorePreRootHistory = analysisIgnorePreRootHistory;
  };

  SearchParams initialGenmoveParams;
  SearchParams initialAnalysisParams;
  loadParams(cfg,initialGenmoveParams,initialAnalysisParams);
  logger.write("Using " + Global::intToString(initialGenmoveParams.numThreads) + " CPU thread(s) for search");

  const bool ponderingEnabled = cfg.contains("ponderingEnabled") ? cfg.getBool("ponderingEnabled") : false;

  const enabled_t cleanupBeforePass = cfg.contains("cleanupBeforePass") ? cfg.getEnabled("cleanupBeforePass") : enabled_t::Auto;
  const enabled_t friendlyPass = cfg.contains("friendlyPass") ? cfg.getEnabled("friendlyPass") : enabled_t::Auto;
  if(cleanupBeforePass == enabled_t::True && friendlyPass == enabled_t::True)
    throw StringError("Cannot specify both cleanupBeforePass = true and friendlyPass = true at the same time");

  const bool allowResignation = cfg.contains("allowResignation") ? cfg.getBool("allowResignation") : false;
  const double resignThreshold = cfg.contains("allowResignation") ? cfg.getDouble("resignThreshold",-1.0,0.0) : -1.0; //Threshold on [-1,1], regardless of winLossUtilityFactor
  const int resignConsecTurns = cfg.contains("resignConsecTurns") ? cfg.getInt("resignConsecTurns",1,100) : 3;
  const double resignMinScoreDifference = cfg.contains("resignMinScoreDifference") ? cfg.getDouble("resignMinScoreDifference",0.0,1000.0) : -1e10;
  const double resignMinMovesPerBoardArea = cfg.contains("resignMinMovesPerBoardArea") ? cfg.getDouble("resignMinMovesPerBoardArea",0.0,1.0) : 0.0;

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
  bool staticPDATakesPrecedence = cfg.contains("playoutDoublingAdvantage") && !cfg.contains("dynamicPlayoutDoublingAdvantageCapPerOppLead");
  const double normalAvoidRepeatedPatternUtility = initialGenmoveParams.avoidRepeatedPatternUtility;
  const double handicapAvoidRepeatedPatternUtility = cfg.contains("avoidRepeatedPatternUtility") ?
    initialGenmoveParams.avoidRepeatedPatternUtility : 0.005;
  const double delayMoveScale = cfg.contains("delayMoveScale") ? cfg.getDouble("delayMoveScale",0.0,10000.0) : 0.0;
  const double delayMoveMax = cfg.contains("delayMoveMax") ? cfg.getDouble("delayMoveMax",0.0,1000000.0) : 1000000.0;

  int defaultBoardXSize = -1;
  int defaultBoardYSize = -1;
  Setup::loadDefaultBoardXYSize(cfg,logger,defaultBoardXSize,defaultBoardYSize);

  const bool forDeterministicTesting =
    cfg.contains("forDeterministicTesting") ? cfg.getBool("forDeterministicTesting") : false;

  if(forDeterministicTesting)
    seedRand.init("forDeterministicTesting");

  std::unique_ptr<PatternBonusTable> patternBonusTable = nullptr;
  {
    std::vector<std::unique_ptr<PatternBonusTable>> tables = Setup::loadAvoidSgfPatternBonusTables(cfg,logger);
    assert(tables.size() == 1);
    patternBonusTable = std::move(tables[0]);
  }

  bool autoAvoidPatterns = false;
  {
    std::unique_ptr<PatternBonusTable> autoTable = Setup::loadAndPruneAutoPatternBonusTables(cfg,logger);
    if(autoTable != nullptr && patternBonusTable != nullptr)
      throw StringError("Providing both sgf avoid patterns and auto avoid patterns is not implemented right now");
    if(autoTable != nullptr) {
      autoAvoidPatterns = true;
      patternBonusTable = std::move(autoTable);
    }
  }
  // Toggled to true every time we save, toggled back to false once we do load.
  bool shouldReloadAutoAvoidPatterns = false;

  Player perspective = Setup::parseReportAnalysisWinrates(cfg,C_EMPTY);

  GTPEngine* engine = new GTPEngine(
    nnModelFile,humanModelFile,
    initialGenmoveParams,initialAnalysisParams,
    initialRules,
    assumeMultipleStartingBlackMovesAreHandicap,preventEncore,autoAvoidPatterns,
    dynamicPlayoutDoublingAdvantageCapPerOppLead,
    staticPDATakesPrecedence,
    normalAvoidRepeatedPatternUtility, handicapAvoidRepeatedPatternUtility,
    delayMoveScale,delayMoveMax,
    perspective,analysisPVLen,
    std::move(patternBonusTable)
  );
  engine->setOrResetBoardSize(cfg,logger,seedRand,defaultBoardXSize,defaultBoardYSize,logger.isLoggingToStderr());

  auto maybeSaveAvoidPatterns = [&](bool forceSave) {
    if(engine != NULL && autoAvoidPatterns) {
      int samplesPerSave = 200;
      if(cfg.contains("autoAvoidRepeatSaveChunkSize"))
        samplesPerSave = cfg.getInt("autoAvoidRepeatSaveChunkSize",1,10000);

      if(forceSave || engine->genmoveSamples.size() >= samplesPerSave) {
        bool suc = Setup::saveAutoPatternBonusData(engine->genmoveSamples, cfg, logger, seedRand);
        if(suc) {
          engine->genmoveSamples.clear();
          shouldReloadAutoAvoidPatterns = true;
        }
      }
    }
  };

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
  Setup::maybeWarnHumanSLParams(initialGenmoveParams,engine->nnEval,engine->humanEval,cerr,&logger);

  logger.write("Loaded config " + cfg.getFileName());
  logger.write("Loaded model " + nnModelFile);
  if(humanModelFile != "")
    logger.write("Loaded human SL model " + humanModelFile);
  cmd.logOverrides(logger);
  logger.write("Model name: "+ (engine->nnEval == NULL ? string() : engine->nnEval->getInternalModelName()));
  if(engine->humanEval != NULL)
    logger.write("Human SL model name: "+ (engine->humanEval->getInternalModelName()));
  logger.write("GTP ready, beginning main protocol loop");
  //Also check loggingToStderr so that we don't duplicate the message from the log file
  if(startupPrintMessageToStderr && !logger.isLoggingToStderr()) {
    cerr << "Loaded config " << cfg.getFileName() << endl;
    cerr << "Loaded model " << nnModelFile << endl;
    if(humanModelFile != "")
      cerr << "Loaded human SL model " << humanModelFile << endl;
    cerr << "Model name: "+ (engine->nnEval == NULL ? string() : engine->nnEval->getInternalModelName()) << endl;
    if(engine->humanEval != NULL)
      cerr << "Human SL model name: "+ (engine->humanEval->getInternalModelName()) << endl;
    cerr << "GTP ready, beginning main protocol loop" << endl;
  }

  if(humanModelFile != "" && !cfg.contains("humanSLProfile")) {
    logger.write("WARNING: Provided -human-model but humanSLProfile was not set in the config. The human SL model will not be used until it is set.");
    if(!logger.isLoggingToStderr())
      cerr << "WARNING: Provided -human-model but humanSLProfile was not set in the config. The human SL model will not be used until it is set." << endl;
  }

  bool currentlyGenmoving = false;
  bool currentlyAnalyzing = false;
  string line;
  while(getline(cin,line)) {
    //Parse command, extracting out the command itself, the arguments, and any GTP id number for the command.
    string command;
    vector<string> pieces;
    bool hasId = false;
    int id = 0;
    {
      line = CommandLoop::processSingleCommandLine(line);

      //Upon any input line at all, stop any analysis and output a newline
      //Only difference between analysis and genmove is that genmove handles its own
      //double newline in its onmove callback.
      if(currentlyAnalyzing) {
        currentlyAnalyzing = false;
        engine->stopAndWait();
        cout << endl;
      }
      if(currentlyGenmoving) {
        currentlyGenmoving = false;
        engine->stopAndWait();
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

    auto printGTPResponse = [hasId,id,&logger,logAllGTPCommunication](const string& response, bool responseIsError) {
      string postProcessed = response;
      if(hasId)
        postProcessed = Global::intToString(id) + " " + postProcessed;
      else
        postProcessed = " " + postProcessed;

      if(responseIsError)
        postProcessed = "?" + postProcessed;
      else
        postProcessed = "=" + postProcessed;

      cout << postProcessed << endl;
      cout << endl;

      if(logAllGTPCommunication)
        logger.write(postProcessed);
    };
    auto printGTPResponseHeader = [hasId,id,&logger,logAllGTPCommunication]() {
      if(hasId) {
        string s = "=" + Global::intToString(id);
        cout << s << endl;
        if(logAllGTPCommunication)
          logger.write(s);
      }
      else {
        cout << "=" << endl;
        if(logAllGTPCommunication)
          logger.write("=");
      }
    };

    auto printGTPResponseNoHeader = [hasId,id,&logger,logAllGTPCommunication](const string& response, bool responseIsError) {
      //Postprocessing of response in the case where we already printed the "=" and a newline ahead of time via printGTPResponseHeader.
      if(!responseIsError) {
        cout << response << endl;
        cout << endl;
      }
      else {
        cout << endl;
        if(!logger.isLoggingToStderr())
          cerr << response << endl;
      }
      if(logAllGTPCommunication)
        logger.write(response);
    };

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
      else {
        std::vector<string> parts;
        parts.push_back(Version::getKataGoVersion());
        if(engine->nnEval != NULL)
          parts.push_back(engine->nnEval->getAbbrevInternalModelName());
        if(engine->humanEval != NULL)
          parts.push_back(engine->humanEval->getAbbrevInternalModelName());
        response = Global::concat(parts,"+");
      }
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
      maybeSaveAvoidPatterns(true);
      shouldQuitAfterResponse = true;
      logger.write("Quit requested by controller");
    }

    else if(command == "boardsize" || command == "rectangular_boardsize") {
      maybeSaveAvoidPatterns(false);
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
        engine->setOrResetBoardSize(cfg,logger,seedRand,newXSize,newYSize,logger.isLoggingToStderr());
      }
    }

    else if(command == "clear_board") {
      maybeSaveAvoidPatterns(false);
      if(autoAvoidPatterns && shouldReloadAutoAvoidPatterns) {
        std::unique_ptr<PatternBonusTable> autoTable = Setup::loadAndPruneAutoPatternBonusTables(cfg,logger);
        engine->setPatternBonusTable(std::move(autoTable));
        shouldReloadAutoAvoidPatterns = false;
      }
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
        if(!logger.isLoggingToStderr())
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
          if(!logger.isLoggingToStderr())
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
        if(!logger.isLoggingToStderr())
          cerr << "Changed rules to " + newRules.toStringNoKomiMaybeNice() << endl;
      }
    }

    else if(command == "kata-list-params") {
      std::vector<string> paramsList;
      paramsList.push_back("analysisWideRootNoise");
      paramsList.push_back("analysisIgnorePreRootHistory");
      paramsList.push_back("genmoveAntiMirror");
      paramsList.push_back("antiMirror");
      paramsList.push_back("humanSLProfile");
      nlohmann::json params = engine->getGenmoveParams().changeableParametersToJson();
      for(auto& elt : params.items()) {
        paramsList.push_back(elt.key());
      }
      response = Global::concat(paramsList, " ");
    }

    else if(command == "kata-get-param") {
      if(pieces.size() != 1) {
        responseIsError = true;
        response = "Expected one arguments for kata-get-param but got '" + Global::concat(pieces," ") + "'";
      }
      else {
        const SearchParams& genmoveParams = engine->getGenmoveParams();
        const SearchParams& analysisParams = engine->getAnalysisParams();
        if(pieces[0] == "analysisWideRootNoise")
          response = Global::doubleToString(analysisParams.wideRootNoise);
        else if(pieces[0] == "analysisIgnorePreRootHistory")
          response = Global::boolToString(analysisParams.ignorePreRootHistory);
        else if(pieces[0] == "genmoveAntiMirror")
          response = Global::boolToString(genmoveParams.antiMirror);
        else if(pieces[0] == "antiMirror")
          response = Global::boolToString(analysisParams.antiMirror);
        else if(pieces[0] == "humanSLProfile") {
          response = cfg.contains("humanSLProfile") ? cfg.getString("humanSLProfile") : "";
        }
        else {
          nlohmann::json params = engine->getGenmoveParams().changeableParametersToJson();
          if(params.find(pieces[0]) == params.end()) {
            responseIsError = true;
            response = "Invalid parameter: " + pieces[0];
          }
          else {
            response = params[pieces[0]].dump();
          }
        }
      }
    }
    else if(command == "kata-get-models") {
      nlohmann::json modelsList = nlohmann::json::array();
      if(engine->nnEval != NULL) {
        nlohmann::json modelInfo;
        modelInfo["name"] = engine->nnEval->getModelName();
        modelInfo["internalName"] = engine->nnEval->getInternalModelName();
        modelInfo["maxBatchSize"] = engine->nnEval->getMaxBatchSize();
        modelInfo["usesHumanSLProfile"] = engine->nnEval->requiresSGFMetadata();
        modelInfo["version"] = engine->nnEval->getModelVersion();
        modelInfo["usingFP16"] = engine->nnEval->getUsingFP16Mode().toString();
        modelsList.push_back(modelInfo);
      }
      if(engine->humanEval != NULL) {
        nlohmann::json modelInfo;
        modelInfo["name"] = engine->humanEval->getModelName();
        modelInfo["internalName"] = engine->humanEval->getInternalModelName();
        modelInfo["maxBatchSize"] = engine->humanEval->getMaxBatchSize();
        modelInfo["usesHumanSLProfile"] = engine->humanEval->requiresSGFMetadata();
        modelInfo["version"] = engine->humanEval->getModelVersion();
        modelInfo["usingFP16"] = engine->humanEval->getUsingFP16Mode().toString();
        modelsList.push_back(modelInfo);
      }
      response = modelsList.dump();
    }
    else if(command == "kata-get-params") {
      const SearchParams& genmoveParams = engine->getGenmoveParams();
      const SearchParams& analysisParams = engine->getAnalysisParams();
      nlohmann::json params = engine->getGenmoveParams().changeableParametersToJson();
      params["analysisWideRootNoise"] = Global::doubleToString(analysisParams.wideRootNoise);
      params["analysisIgnorePreRootHistory"] = Global::boolToString(analysisParams.ignorePreRootHistory);
      params["genmoveAntiMirror"] = Global::boolToString(genmoveParams.antiMirror);
      params["antiMirror"] = Global::boolToString(analysisParams.antiMirror);
      params["humanSLProfile"] = cfg.contains("humanSLProfile") ? cfg.getString("humanSLProfile") : "";
      response = params.dump();
    }
    else if(command == "kata-set-param" || command == "kata-set-params") {
      std::map<string,string> overrideSettings;
      if(command == "kata-set-param") {
        if(pieces.size() != 1 && pieces.size() != 2) {
          responseIsError = true;
          response = "Expected one or two arguments for kata-set-param but got '" + Global::concat(pieces," ") + "'";
        }
        else if(pieces.size() == 1) {
          overrideSettings[pieces[0]] = "";
        }
        else {
          overrideSettings[pieces[0]] = pieces[1];
        }
      }
      else {
        if(pieces.size() < 1) {
          responseIsError = true;
          response = "Expected argument for kata-set-param but got '" + Global::concat(pieces," ") + "'";
        }
        else {
          try {
            string rejoined = Global::concat(pieces, " ", 0, pieces.size());
            nlohmann::json settings = nlohmann::json::parse(rejoined);
            if(!settings.is_object())
              throw StringError("Argument to kata-set-params must be a json object");

            for(auto it = settings.begin(); it != settings.end(); ++it) {
              overrideSettings[it.key()] = it.value().is_string() ? it.value().get<string>(): it.value().dump(); // always convert to string
            }
          }
          catch(const StringError& exception) {
            responseIsError = true;
            response = string("Could not set params: ") + exception.what();
          }
        }
      }

      if(!responseIsError) {
        try {
          // First validate that everything in here is used by checking it by itself
          {
            ConfigParser cleanCfg;
            cleanCfg.overrideKeys(overrideSettings);
            // Add required parameter so that it passes validation
            if(!cleanCfg.contains("numSearchThreads"))
              cleanCfg.overrideKey("numSearchThreads",Global::intToString(engine->getGenmoveParams().numThreads));

            SearchParams buf1;
            SearchParams buf2;
            loadParams(cleanCfg, buf1, buf2);

            // These parameters have a bit of special handling so we can't change them easily right now
            if(contains(overrideSettings,"dynamicPlayoutDoublingAdvantageCapPerOppLead")) throw StringError("Cannot be overridden in kata-set-param: dynamicPlayoutDoublingAdvantageCapPerOppLead");
            if(contains(overrideSettings,"avoidRepeatedPatternUtility")) throw StringError("Cannot be overridden in kata-set-param: avoidRepeatedPatternUtility");

            vector<string> unusedKeys = cleanCfg.unusedKeys();
            for(const string& unused: unusedKeys) {
              throw StringError("Unrecognized or non-overridable parameter in kata-set-params: " + unused);
            }
            ostringstream out;
            if(Setup::maybeWarnHumanSLParams(buf1,engine->nnEval,engine->humanEval,out,NULL)) {
              throw StringError(out.str());
            }
          }

          //Ignore any unused keys in the original config so far
          cfg.markAllKeysUsedWithPrefix("");
          // Now enshrine the new settings into the real config permanently, and parse.
          cfg.overrideKeys(overrideSettings);
          SearchParams genmoveParams;
          SearchParams analysisParams;
          loadParams(cfg,genmoveParams,analysisParams);

          SearchParams::failIfParamsDifferOnUnchangeableParameter(initialGenmoveParams,genmoveParams);
          SearchParams::failIfParamsDifferOnUnchangeableParameter(initialAnalysisParams,analysisParams);
          engine->setGenmoveParamsIfChanged(genmoveParams);
          engine->setAnalysisParamsIfChanged(analysisParams);
          staticPDATakesPrecedence = cfg.contains("playoutDoublingAdvantage") && !cfg.contains("dynamicPlayoutDoublingAdvantageCapPerOppLead");
          engine->staticPDATakesPrecedence = staticPDATakesPrecedence;
        }
        catch(const StringError& exception) {
          responseIsError = true;
          response = string("Could not set params: ") + exception.what();
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
      response += "Black "+ engine->bTimeControls.toDebugString(engine->bot->getRootBoard(),engine->bot->getRootHist(),engine->getGenmoveParams().lagBuffer);
      response += "\n";
      response += "White "+ engine->wTimeControls.toDebugString(engine->bot->getRootBoard(),engine->bot->getRootHist(),engine->getGenmoveParams().lagBuffer);
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
            response += "could not parse color: '" + pieces[i] + "'";
            break;
          }
          else if(!tryParseLoc(pieces[i+1],engine->bot->getRootBoard(),loc)) {
            responseIsError = true;
            response = "Expected a space-separated sequence of <COLOR> <VERTEX> pairs but got '" + Global::concat(pieces," ") + "': ";
            response += "could not parse vertex: '" + pieces[i+1] + "'";
            break;
          }
          else if(loc == Board::PASS_LOC) {
            responseIsError = true;
            response = "Expected a space-separated sequence of <COLOR> <VERTEX> pairs but got '" + Global::concat(pieces," ") + "': ";
            response += "could not parse vertex: '" + pieces[i+1] + "'";
            break;
          }
          initialStones.push_back(Move(loc,pla));
        }
        if(!responseIsError) {
          maybeSaveAvoidPatterns(false);
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

    else if(
      command == "genmove" ||
      command == "genmove_debug" ||
      command == "kata-search" ||
      command == "kata-search_cancellable" ||
      command == "kata-search_debug") {
      Player pla;
      if(pieces.size() != 1) {
        responseIsError = true;
        response = "Expected one argument for " + command + " but got '" + Global::concat(pieces," ") + "'";
      }
      else if(!PlayerIO::tryParsePlayer(pieces[0],pla)) {
        responseIsError = true;
        response = "Could not parse color: '" + pieces[0] + "'";
      }
      else {
        bool debug = command == "genmove_debug" || command == "kata-search_debug";
        bool playChosenMove = command == "genmove" || command == "genmove_debug";

        GTPEngine::GenmoveArgs gargs;
        gargs.searchFactorWhenWinningThreshold = searchFactorWhenWinningThreshold;
        gargs.searchFactorWhenWinning = searchFactorWhenWinning;
        gargs.cleanupBeforePass = cleanupBeforePass;
        gargs.friendlyPass = friendlyPass;
        gargs.ogsChatToStderr = ogsChatToStderr;
        gargs.allowResignation = allowResignation;
        gargs.resignThreshold = resignThreshold;
        gargs.resignConsecTurns = resignConsecTurns;
        gargs.resignMinScoreDifference = resignMinScoreDifference;
        gargs.resignMinMovesPerBoardArea = resignMinMovesPerBoardArea;
        gargs.logSearchInfo = logSearchInfo;
        gargs.logSearchInfoForChosenMove = logSearchInfoForChosenMove;
        gargs.debug = debug;

        if(command == "kata-search_cancellable") {
          engine->genMoveCancellable(
            pla,
            logger,
            gargs,
            GTPEngine::AnalyzeArgs(),
            printGTPResponse
          );
          suppressResponse = true; // genmove handles it manually by calling printGTPResponse
          currentlyGenmoving = true; // so that any newline will interrupt us
        }
        else {
          engine->genMove(
            pla,
            logger,
            gargs,
            GTPEngine::AnalyzeArgs(),
            playChosenMove,
            printGTPResponse,
            maybeStartPondering
          );
          suppressResponse = true; // genmove handles it manually by calling printGTPResponse
        }
      }
    }

    else if(
      command == "genmove_analyze" ||
      command == "lz-genmove_analyze" ||
      command == "kata-genmove_analyze" ||
      command == "kata-search_analyze" ||
      command == "kata-search_analyze_cancellable"
    ) {
      Player pla = engine->bot->getRootPla();
      bool parseFailed = false;
      GTPEngine::AnalyzeArgs analyzeArgs = parseAnalyzeCommand(command, pieces, pla, parseFailed, engine);
      if(parseFailed) {
        responseIsError = true;
        response = "Could not parse genmove_analyze arguments or arguments out of range: '" + Global::concat(pieces," ") + "'";
      }
      else {
        bool debug = false;
        bool playChosenMove = command == "genmove_analyze" || command == "lz-genmove_analyze" || command == "kata-genmove_analyze";

        GTPEngine::GenmoveArgs gargs;
        gargs.searchFactorWhenWinningThreshold = searchFactorWhenWinningThreshold;
        gargs.searchFactorWhenWinning = searchFactorWhenWinning;
        gargs.cleanupBeforePass = cleanupBeforePass;
        gargs.friendlyPass = friendlyPass;
        gargs.ogsChatToStderr = ogsChatToStderr;
        gargs.allowResignation = allowResignation;
        gargs.resignThreshold = resignThreshold;
        gargs.resignConsecTurns = resignConsecTurns;
        gargs.resignMinScoreDifference = resignMinScoreDifference;
        gargs.resignMinMovesPerBoardArea = resignMinMovesPerBoardArea;
        gargs.logSearchInfo = logSearchInfo;
        gargs.logSearchInfoForChosenMove = logSearchInfoForChosenMove;
        gargs.debug = debug;

        //Make sure the "equals" for GTP is printed out prior to the first analyze line, regardless of thread racing
        printGTPResponseHeader();

        if(command == "kata-search_analyze_cancellable") {
          engine->genMoveCancellable(
            pla,
            logger,
            gargs,
            analyzeArgs,
            printGTPResponseNoHeader
          );
          suppressResponse = true; // genmove handles it manually by calling printGTPResponse
          currentlyGenmoving = true; // so that any newline will interrupt us
        }
        else {
          engine->genMove(
            pla,
            logger,
            gargs,
            analyzeArgs,
            playChosenMove,
            printGTPResponseNoHeader,
            maybeStartPondering
          );
          suppressResponse = true; // genmove handles it manually by calling printGTPResponseNoHeader
        }
      }
    }

    else if(command == "clear_cache") {
      engine->clearCache();
    }
    else if(command == "showboard") {
      ostringstream sout;
      engine->bot->getRootHist().printBasicInfo(sout, engine->bot->getRootBoard());
      response = Global::trim(filterDoubleNewlines(sout.str()));
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
        maybeSaveAvoidPatterns(false);
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
        maybeSaveAvoidPatterns(false);
        engine->placeFreeHandicap(n,response,responseIsError,seedRand);
      }
    }

    else if(command == "set_free_handicap") {
      if(!engine->bot->getRootBoard().isEmpty()) {
        responseIsError = true;
        response = "Board is not empty";
      }
      else {
        vector<Move> locs;
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
          locs.push_back(Move(loc,P_BLACK));
        }
        bool suc = board.setStonesFailIfNoLibs(locs);
        if(!suc) {
          responseIsError = true;
          response = "Handicap placement is invalid";
        }
        else {
          maybeSaveAvoidPatterns(false);
          Player pla = P_WHITE;
          BoardHistory hist(board,pla,engine->getCurrentRules(),0);
          hist.setInitialTurnNumber(board.numStonesOnBoard()); //Should give more accurate temperaure and time control behavior
          vector<Move> newMoveHistory;
          engine->setPositionAndRules(pla,board,hist,board,pla,newMoveHistory);
        }
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
          moveNumber--;
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
                if(!logger.isLoggingToStderr())
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
                if(!logger.isLoggingToStderr())
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
              if(!logger.isLoggingToStderr())
                cerr << out.str() << endl;
            }
            maybeSaveAvoidPatterns(false);
            engine->setOrResetBoardSize(cfg,logger,seedRand,sgfBoard.x_size,sgfBoard.y_size,logger.isLoggingToStderr());
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
      else {
        auto writeSgfToStream = [&](ostream& out) {
          double overrideFinalScore = std::numeric_limits<double>::quiet_NaN();
          if(engine->bot->getRootHist().isGameFinished) {
            Player winner = C_EMPTY;
            double finalWhiteMinusBlackScore = 0.0;
            engine->computeAnticipatedWinnerAndScore(winner,finalWhiteMinusBlackScore);
            overrideFinalScore = finalWhiteMinusBlackScore;
          }
          WriteSgf::writeSgf(out,"","",engine->bot->getRootHist(),NULL,true,false,overrideFinalScore);
        };

        if(pieces.size() == 0 || pieces[0] == "-") {
          ostringstream out;
          writeSgfToStream(out);
          response = out.str();
        }
        else {
          ofstream out;
          if(FileUtils::tryOpen(out,pieces[0])) {
            writeSgfToStream(out);
            out.close();
            response = "";
          }
          else {
            responseIsError = true;
            response = "Could not open or write to file: " + pieces[0];
          }
        }
      }
    }

    else if(command == "analyze" || command == "lz-analyze" || command == "kata-analyze") {
      Player pla = engine->bot->getRootPla();
      bool parseFailed = false;
      GTPEngine::AnalyzeArgs analyzeArgs = parseAnalyzeCommand(command, pieces, pla, parseFailed, engine);

      if(parseFailed) {
        responseIsError = true;
        response = "Could not parse analyze arguments or arguments out of range: '" + Global::concat(pieces," ") + "'";
      }
      else {
        //Make sure the "equals" for GTP is printed out prior to the first analyze line, regardless of thread racing
        printGTPResponseHeader();

        engine->analyze(pla, analyzeArgs);

        //No response - currentlyAnalyzing will make sure we get a newline at the appropriate time, when stopped.
        suppressResponse = true;
        currentlyAnalyzing = true;
      }
    }

    else if(command == "kata-raw-nn") {
      int whichSymmetry = NNInputs::SYMMETRY_ALL;
      bool parsed = false;
      if(pieces.size() == 1 || pieces.size() == 2) {
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
        double policyOptimism = engine->getGenmoveParams().rootPolicyOptimism;
        if(pieces.size() == 2) {
          parsed = false;
          if(Global::tryStringToDouble(pieces[0],policyOptimism) && isnan(policyOptimism) && policyOptimism >= 0.0 && policyOptimism <= 1.0) {
            parsed = true;
          }
        }
        if(!parsed) {
          responseIsError = true;
          response = "Expected double from 0 to 1 for optimism but got '" + Global::concat(pieces," ") + "'";
        }
        else {
          const bool useHumanModel = false;
          response = engine->rawNN(whichSymmetry, policyOptimism, useHumanModel);
        }
      }
    }

    else if(command == "kata-raw-human-nn") {
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
        response = "Expected one argument 'all' or symmetry index [0-7] for kata-raw-human-nn but got '" + Global::concat(pieces," ") + "'";
      }
      else {
        if(engine->humanEval == NULL) {
          responseIsError = true;
          response = "Cannot run kata-raw-human-nn, -human-model was not provided";
        }
        else if(!(engine->getGenmoveParams().humanSLProfile.initialized || !engine->humanEval->requiresSGFMetadata())) {
          responseIsError = true;
          response = "Cannot run kata-raw-human-nn, humanSLProfile parameter was not set";
        }
        else {
          double policyOptimism = engine->getGenmoveParams().rootPolicyOptimism;
          const bool useHumanModel = true;
          response = engine->rawNN(whichSymmetry, policyOptimism, useHumanModel);
        }
      }
    }

    else if(command == "debug_moves") {
      PrintTreeOptions options;
      options = options.maxDepth(1);
      string printBranch;
      bool printRawStats = false;
      for(size_t i = 0; i<pieces.size(); i++) {
        if(pieces[i] == "rawstats") {
          printRawStats = true;
          continue;
        }
        if(i > 0)
          printBranch += " ";
        printBranch += pieces[i];
      }
      try {
        if(printBranch.length() > 0)
          options = options.onlyBranch(engine->bot->getRootBoard(),printBranch);
      }
      catch(const StringError& e) {
        (void)e;
        responseIsError = true;
        response = "Invalid move sequence";
      }
      if(!responseIsError) {
        Search* search = engine->bot->getSearchStopAndWait();
        ostringstream sout;

        Player pla = engine->bot->getRootPla();
        Board board = engine->bot->getRootBoard();
        BoardHistory hist = engine->bot->getRootHist();
        bool allLegal = true;
        for(Loc loc: options.branch_) {
          bool suc = hist.makeBoardMoveTolerant(board, loc, pla, false);
          if(!suc) {
            allLegal = false;
            break;
          }
          pla = getOpp(pla);
        }
        if(allLegal) {
          Board::printBoard(sout, board, Board::NULL_LOC, &hist.moveHistory);
        }
        search->printTree(sout, search->rootNode, options, perspective);
        if(printRawStats) {
          sout << engine->rawNNBrief(options.branch_, NNInputs::SYMMETRY_ALL);
        }
        response = filterDoubleNewlines(sout.str());
      }
    }
    else if(command == "cputime" || command == "gomill-cpu_time") {
      response = Global::doubleToString(engine->genmoveTimeSum);
    }

    else if(command == "kata-benchmark") {
      bool parsed = false;
      int64_t numVisits = 0;
      if(pieces.size() != 1) {
        responseIsError = true;
        response = "Expected one argument for kata-benchmark but got '" + Global::concat(pieces," ") + "'";
      }
      else {
        bool suc = Global::tryStringToInt64(pieces[0],numVisits);
        if(!suc) {
          responseIsError = true;
          response = "Could not parse number of visits: " + pieces[0];
        }
        parsed = true;
      }

      if(parsed) {
        engine->stopAndWait();

        int boardSizeX = engine->bot->getRootBoard().x_size;
        int boardSizeY = engine->bot->getRootBoard().y_size;
        if(boardSizeX != boardSizeY) {
          responseIsError = true;
          response =
            "Current board size is " + Global::intToString(boardSizeX) + "x" + Global::intToString(boardSizeY) +
            ", no built-in benchmarks for rectangular boards";
        }
        else {
          CompactSgf* sgf = NULL;
          try {
            string sgfData = TestCommon::getBenchmarkSGFData(boardSizeX);
            sgf = CompactSgf::parse(sgfData);
          }
          catch(const StringError& e) {
            responseIsError = true;
            response = e.what();
          }
          if(sgf != NULL) {
            const PlayUtils::BenchmarkResults* baseline = NULL;
            const double secondsPerGameMove = 1.0;
            const bool printElo = false;
            SearchParams params = engine->getGenmoveParams();
            params.maxTime = 1.0e20;
            params.maxPlayouts = ((int64_t)1) << 50;
            params.maxVisits = numVisits;
            //Make sure the "equals" for GTP is printed out prior to the benchmark line
            printGTPResponseHeader();

            try {
              PlayUtils::BenchmarkResults results = PlayUtils::benchmarkSearchOnPositionsAndPrint(
                params,
                sgf,
                10,
                engine->nnEval,
                baseline,
                secondsPerGameMove,
                printElo
              );
              (void)results;
            }
            catch(const StringError& e) {
              responseIsError = true;
              response = e.what();
              delete sgf;
              sgf = NULL;
            }
            if(sgf != NULL) {
              delete sgf;
              //Act of benchmarking will write to stdout with a newline at the end, so we just need one more newline ourselves
              //to complete GTP protocol.
              suppressResponse = true;
              cout << endl;
            }
          }
        }
      }
    }

    else if(command == "stop") {
      //Stop any ongoing ponder or analysis
      engine->stopAndWait();
    }

    else {
      responseIsError = true;
      response = "unknown command";
    }

    if(!suppressResponse)
      printGTPResponse(response,responseIsError);

    if(shouldQuitAfterResponse)
      break;

    if(maybeStartPondering && ponderingEnabled)
      engine->ponder();

  } //Close read loop

  // Interrupt stuff if we close stdout
  if(currentlyAnalyzing) {
    currentlyAnalyzing = false;
    engine->stopAndWait();
    cout << endl;
  }
  if(currentlyGenmoving) {
    currentlyGenmoving = false;
    engine->stopAndWait();
  }


  maybeSaveAvoidPatterns(true);
  delete engine;
  engine = NULL;
  NeuralNet::globalCleanup();
  ScoreValue::freeTables();

  logger.write("All cleaned up, quitting");
  return 0;
}
