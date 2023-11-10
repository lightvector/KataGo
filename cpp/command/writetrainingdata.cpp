#include "../core/global.h"
#include "../core/datetime.h"
#include "../core/fileutils.h"
#include "../core/makedir.h"
#include "../core/config_parser.h"
#include "../core/parallel.h"
#include "../dataio/sgf.h"
#include "../dataio/trainingwrite.h"
#include "../dataio/loadmodel.h"
#include "../dataio/files.h"
#include "../neuralnet/modelversion.h"
#include "../program/setup.h"
#include "../command/commandline.h"
#include "../main.h"

#include <chrono>
#include <csignal>

using namespace std;

static ValueTargets makeForcedWinnerValueTarget(Player winner) {
  ValueTargets targets;
  assert(winner == P_BLACK || winner == P_WHITE);
  targets.win = winner == P_WHITE ? 1.0f : 0.0f;
  targets.loss = winner == P_BLACK ? 1.0f : 0.0f;
  targets.noResult = 0.0f;
  targets.score = winner == P_WHITE ? 0.5f : -0.5f;
  targets.hasLead = false;
  targets.lead = 0.0f;
  return targets;
}

static ValueTargets makeWhiteValueTarget(
  const Board& board,
  const BoardHistory& hist,
  Player nextPla,
  double drawEquivalentWinsForWhite,
  Player playoutDoublingAdvantagePla,
  double playoutDoublingAdvantage,
  NNEvaluator* nnEval
) {
  ValueTargets targets;
  if(hist.isGameFinished) {
    if(hist.isNoResult) {
      targets.win = 0.0f;
      targets.loss = 0.0f;
      targets.noResult = 1.0f;
      targets.score = 0.0f;
      targets.hasLead = false;
      targets.lead = 0.0f;
    }
    else {
      BoardHistory copyHist(hist);
      copyHist.endAndScoreGameNow(board);
      targets.win = (float)ScoreValue::whiteWinsOfWinner(copyHist.winner, drawEquivalentWinsForWhite);
      targets.loss = 1.0f - targets.win;
      targets.noResult = 0.0f;
      targets.score = (float)ScoreValue::whiteScoreDrawAdjust(copyHist.finalWhiteMinusBlackScore,drawEquivalentWinsForWhite,hist);
      targets.hasLead = true;
      targets.lead = targets.score;
    }
    return targets;
  }

  NNResultBuf buf;
  bool skipCache = true;
  bool includeOwnerMap = false;
  MiscNNInputParams nnInputParams;
  nnInputParams.conservativePassAndIsRoot = true;
  nnInputParams.enablePassingHacks = true;
  nnInputParams.drawEquivalentWinsForWhite = drawEquivalentWinsForWhite;
  nnInputParams.playoutDoublingAdvantage = (playoutDoublingAdvantagePla == getOpp(nextPla) ? -playoutDoublingAdvantage : playoutDoublingAdvantage);
  Board copy(board);
  nnEval->evaluate(copy,hist,nextPla,nnInputParams,buf,skipCache,includeOwnerMap);

  targets.win = buf.result->whiteWinProb;
  targets.loss = buf.result->whiteLossProb;
  targets.noResult = buf.result->whiteNoResultProb;
  targets.score = buf.result->whiteScoreMean;
  targets.hasLead = true;
  targets.lead = buf.result->whiteLead;
  return targets;
}

static std::set<string> loadStrippedTxtFileLines(const string& filePath) {
  std::vector<string> lines = FileUtils::readFileLines(filePath,'\n');
  std::set<string> ret;
  for(const string& line: lines) {
    string s = Global::trim(line);
    if(s.size() > 0)
      ret.insert(s);
  }
  return ret;
}


int MainCmds::writetrainingdata(const vector<string>& args) {
  Board::initHash();
  ScoreValue::initTables();
  Rand seedRand;

  ConfigParser cfg;
  string nnModelFile;
  vector<string> sgfDirs;
  string noTrainUsersFile;
  string isBotUsersFile;
  string whatDataSource;
  size_t maxFilesToLoad;
  string outputDir;

  try {
    KataGoCommandLine cmd("Generate training data from sgfs.");
    cmd.addConfigFileArg("","");
    cmd.addModelFileArg();
    cmd.addOverrideConfigArg();

    TCLAP::MultiArg<string> sgfDirArg("","sgfdir","Directory of sgf files",false,"DIR");
    TCLAP::ValueArg<string> noTrainUsersArg("","no-train-users-file","Avoid training on these player's moves",true,string(),"TXTFILE");
    TCLAP::ValueArg<string> isBotUsersArg("","is-bot-users-file","Mark these usernames as bots",true,string(),"TXTFILE");
    TCLAP::ValueArg<string> whatDataSourceArg("","what-data-source","What data source",true,string(),"NAME");
    TCLAP::ValueArg<size_t> maxFilesToLoadArg("","max-files-to-load","Max sgf files to try to load",false,(size_t)10000000000000ULL,"NUM");
    TCLAP::ValueArg<string> outputDirArg("","output-dir","Dir to output files",true,string(),"DIR");

    cmd.add(sgfDirArg);
    cmd.add(noTrainUsersArg);
    cmd.add(isBotUsersArg);
    cmd.add(whatDataSourceArg);
    cmd.add(maxFilesToLoadArg);
    cmd.add(outputDirArg);

    cmd.parseArgs(args);

    nnModelFile = cmd.getModelFile();
    sgfDirs = sgfDirArg.getValue();
    noTrainUsersFile = noTrainUsersArg.getValue();
    isBotUsersFile = isBotUsersArg.getValue();
    whatDataSource = whatDataSourceArg.getValue();
    maxFilesToLoad = maxFilesToLoadArg.getValue();
    outputDir = outputDirArg.getValue();

    cmd.getConfig(cfg);
  }
  catch (TCLAP::ArgException &e) {
    cerr << "Error: " << e.error() << " for argument " << e.argId() << endl;
    return 1;
  }

  const bool logToStdout = true;
  const bool logToStderr = false;
  const bool logTimeStamp = true;
  Logger logger(nullptr, logToStdout, logToStderr, logTimeStamp);
  for(const string& arg: args)
    logger.write(string("Command: ") + arg);

  MakeDir::make(outputDir);
  const int numThreads = 16;

  const int dataBoardLen = cfg.getInt("dataBoardLen",3,37);
  const int maxRowsPerTrainFile = cfg.getInt("maxRowsPerTrainFile",1,100000000);
  const int minGameLenToWrite = cfg.getInt("minGameLenToWrite",1,100000000);

  static_assert(NNModelVersion::latestInputsVersionImplemented == 7, "");
  const int inputsVersion = 7;
  const int numBinaryChannels = NNInputs::NUM_FEATURES_SPATIAL_V7;
  const int numGlobalChannels = NNInputs::NUM_FEATURES_GLOBAL_V7;

  const std::set<string> noTrainUsers = loadStrippedTxtFileLines(noTrainUsersFile);
  const std::set<string> isBotUsers = loadStrippedTxtFileLines(isBotUsersFile);

  NNEvaluator* nnEval;
  {
    Setup::initializeSession(cfg);
    const int maxConcurrentEvals = numThreads * 2 + 16; // * 2 + 16 just to give plenty of headroom
    const int expectedConcurrentEvals = numThreads;
    const int defaultMaxBatchSize = std::max(8,((numThreads+3)/4)*4);
    const bool defaultRequireExactNNLen = false;
    const bool disableFP16 = false;
    const string expectedSha256 = "";
    nnEval = Setup::initializeNNEvaluator(
      nnModelFile,nnModelFile,expectedSha256,cfg,logger,seedRand,maxConcurrentEvals,expectedConcurrentEvals,
      NNPos::MAX_BOARD_LEN,NNPos::MAX_BOARD_LEN,defaultMaxBatchSize,defaultRequireExactNNLen,disableFP16,
      Setup::SETUP_FOR_ANALYSIS
    );
  }
  logger.write("Loaded neural net");

  string searchRandSeed = Global::uint64ToString(seedRand.nextUInt64());
  SearchParams params = SearchParams::basicDecentParams();
  params.maxVisits = 50;
  Search* search = new Search(params,nnEval,&logger,searchRandSeed);

  vector<string> sgfFiles;
  FileHelpers::collectSgfsFromDirsOrFiles(sgfDirs,sgfFiles);

  Setup::initializeSession(cfg);

  cfg.warnUnusedKeys(cerr,&logger);

  //Done loading!
  //------------------------------------------------------------------------------------
  std::atomic<int64_t> numSgfsDone(0);
  std::atomic<int64_t> numSgfErrors(0);

  auto reportSgfDone = [&](bool wasSuccess) {
    if(!wasSuccess)
      numSgfErrors.fetch_add(1);
    int64_t numErrors = numSgfErrors.load();
    int64_t numDone = numSgfsDone.fetch_add(1) + 1;

    if(numDone == sgfFiles.size() || numDone % 100 == 0) {
      logger.write(
        "Done " + Global::int64ToString(numDone) + " / " + Global::int64ToString(sgfFiles.size()) + " sgfs, " +
        string("errors ") + Global::int64ToString(numErrors)
      );
    }
  };

  std::vector<TrainingWriteBuffers*> threadDataBuffers;
  std::vector<Rand*> threadRands;
  for(int i = 0; i<numThreads; i++) {
    threadRands.push_back(new Rand());
    threadDataBuffers.push_back(
      new TrainingWriteBuffers(
        inputsVersion,
        maxRowsPerTrainFile,
        numBinaryChannels,
        numGlobalChannels,
        dataBoardLen,
        dataBoardLen
      )
    );
  }

  std::mutex statsLock;
  std::map<string,int64_t> gameCountByUsername;
  std::map<string,int64_t> gameCountByRank;
  std::map<string,int64_t> gameCountByTimeControl;
  std::map<string,int64_t> gameCountByRules;
  std::map<string,int64_t> gameCountByKomi;
  std::map<string,int64_t> gameCountByHandicap;
  std::map<string,int64_t> gameCountByResult;

  auto processSgf = [&](int threadIdx, size_t index) {
    const string& fileName = sgfFiles[index];
    std::unique_ptr<Sgf> sgfRaw = NULL;
    try {
      sgfRaw = std::unique_ptr<Sgf>(Sgf::loadFile(fileName));
    }
    catch(const StringError& e) {
      logger.write("Invalid SGF " + fileName + ": " + e.what());
      reportSgfDone(false);
      return;
    }
    std::unique_ptr<CompactSgf> sgf = NULL;
    try {
      sgf = std::make_unique<CompactSgf>(sgfRaw.get());
    }
    catch(const StringError& e) {
      logger.write("Invalid SGF " + fileName + ": " + e.what());
      reportSgfDone(false);
      return;
    }

    string sgfBUsername = sgfRaw->getRootPropertyWithDefault("PB", "");
    string sgfWUsername = sgfRaw->getRootPropertyWithDefault("PW", "");
    string sgfBRank = sgfRaw->getRootPropertyWithDefault("BR", "");
    string sgfWRank = sgfRaw->getRootPropertyWithDefault("WR", "");
    string sgfTimeControl = sgfRaw->getRootPropertyWithDefault("TC", "");
    string sgfRules = sgfRaw->getRootPropertyWithDefault("RU", "");
    string sgfKomi = sgfRaw->getRootPropertyWithDefault("KM", "");
    string sgfHandicap = sgfRaw->getRootPropertyWithDefault("HA", "");
    string sgfResult = sgfRaw->getRootPropertyWithDefault("RE", "");

    {
      std::lock_guard<std::mutex> lock(statsLock);
      gameCountByUsername[sgfBUsername] += 1;
      gameCountByUsername[sgfWUsername] += 1;
      gameCountByRank[sgfBRank] += 1;
      gameCountByRank[sgfWRank] += 1;
      gameCountByTimeControl[sgfTimeControl] += 1;
      gameCountByRules[sgfRules] += 1;
      gameCountByKomi[sgfKomi] += 1;
      gameCountByResult[sgfResult] += 1;
    }

    const bool shouldTrainB = noTrainUsers.find(sgfBUsername) == noTrainUsers.end();
    const bool shouldTrainW = noTrainUsers.find(sgfWUsername) == noTrainUsers.end();

    int sgfHandicapParsed = -1;
    if(sgfHandicap != "") {
      if(Global::tryStringToInt(sgfHandicap,sgfHandicapParsed) && sgfHandicapParsed >= 0 && sgfHandicapParsed <= 9) {
        // Yay
      }
      else {
        logger.write("Unable to parse handicap in sgf " + fileName + ": " + sgfHandicap);
        reportSgfDone(false);
        return;
      }
    }

    TrainingWriteBuffers* dataBuffer = threadDataBuffers[threadIdx];
    (void)dataBuffer;
    Rand& rand = *threadRands[threadIdx];

    Board board;
    Player nextPla;
    BoardHistory hist;
    Rules rules = sgf->getRulesOrFailAllowUnspecified(Rules::parseRules("chinese"));
    sgf->setupInitialBoardAndHist(rules, board, nextPla, hist);
    const vector<Move>& sgfMoves = sgf->moves;

    bool sgfGameEnded = false;
    bool sgfGameEndedByTime = false;
    bool sgfGameEndedByResign = false;
    bool sgfGameEndedByScore = false;
    bool sgfGameEndedByOther = false;
    Player sgfGameWinner = C_EMPTY;
    double sgfGameWhiteFinalScore = 0.0;

    if(sgfResult != "") {
      string s = Global::trim(Global::toLower(sgfResult));
      if(s == "b+r" || s == "black+r") {
        sgfGameEnded = true;
        sgfGameEndedByResign = true;
        sgfGameWinner = P_BLACK;
      }
      else if(s == "w+r" || s == "white+r") {
        sgfGameEnded = true;
        sgfGameEndedByResign = true;
        sgfGameWinner = P_WHITE;
      }
      else if(s == "b+t" || s == "black+t") {
        sgfGameEnded = true;
        sgfGameEndedByTime = true;
        sgfGameWinner = P_BLACK;
      }
      else if(s == "w+t" || s == "white+t") {
        sgfGameEnded = true;
        sgfGameEndedByTime = true;
        sgfGameWinner = P_WHITE;
      }
      else if(s == "draw" || s == "0" || s == "jigo") {
        sgfGameEnded = true;
        sgfGameEndedByScore = true;
        sgfGameWinner = C_EMPTY;
        sgfGameWhiteFinalScore = 0.0;
      }
      else if(
        (Global::isPrefix(s,"b+")
         && Global::tryStringToDouble(Global::chopPrefix(s,"b+"), sgfGameWhiteFinalScore))
        ||
        (Global::isPrefix(s,"black+")
         && Global::tryStringToDouble(Global::chopPrefix(s,"black+"), sgfGameWhiteFinalScore))
      ) {
        if(
          std::isfinite(sgfGameWhiteFinalScore)
          && std::abs(sgfGameWhiteFinalScore) < board.x_size * board.y_size
        ) {
          sgfGameEnded = true;
          sgfGameEndedByScore = true;
          sgfGameWinner = P_BLACK;
          sgfGameWhiteFinalScore *= -1;
          if(sgfGameWhiteFinalScore == 0.0)
            sgfGameWinner = C_EMPTY;
        }
        else {
          logger.write("Game ended with invalid score " + fileName + " result " + sgfResult);
          reportSgfDone(false);
          return;
        }
      }
      else if(
        (Global::isPrefix(s,"w+")
         && Global::tryStringToDouble(Global::chopPrefix(s,"w+"), sgfGameWhiteFinalScore))
        ||
        (Global::isPrefix(s,"white+")
         && Global::tryStringToDouble(Global::chopPrefix(s,"white+"), sgfGameWhiteFinalScore))
      ) {
        if(
          std::isfinite(sgfGameWhiteFinalScore)
          && std::abs(sgfGameWhiteFinalScore) < board.x_size * board.y_size
        ) {
          sgfGameEnded = true;
          sgfGameEndedByScore = true;
          sgfGameWinner = P_WHITE;
          sgfGameWhiteFinalScore *= 1;
          if(sgfGameWhiteFinalScore == 0.0)
            sgfGameWinner = C_EMPTY;
        }
        else {
          logger.write("Game ended with invalid score " + fileName + " result " + sgfResult);
          reportSgfDone(false);
          return;
        }
      }
      else {
        logger.write("Game ended with unknown result " + fileName + " result " + sgfResult);
        reportSgfDone(false);
        return;
      }
    }

    bool assumeMultipleStartingBlackMovesAreHandicap;
    int overrideNumHandicapStones = -1;
    if(whatDataSource == "ogs") {
      assumeMultipleStartingBlackMovesAreHandicap = false;
      if(sgfHandicapParsed >= 0)
        overrideNumHandicapStones = sgfHandicapParsed;
    }
    else {
      throw StringError("Unknown data source: " + whatDataSource);
    }

    //If there are any passes in the first 30 moves, start only after the latest such pass.
    int startGameAt = 0;
    for(size_t m = 0; m<sgfMoves.size() && m < 30; m++) {
      if(sgfMoves[m].loc == Board::PASS_LOC)
        startGameAt = m+1;
    }
    //If there are any passes in moves 30 to 50, reject, we're invalid.
    for(size_t m = 30; m<sgfMoves.size() && m < 50; m++) {
      if(sgfMoves[m].loc == Board::PASS_LOC) {
        logger.write("Pass during moves 30-50 in " + fileName);
        reportSgfDone(false);
        return;
      }
    }

    //If there are multiple black moves in a row to start, start at the last one.
    for(size_t m = 1; m<sgfMoves.size() && m < 15; m++) {
      if(sgfMoves[m].pla == P_BLACK && sgfMoves[m-1].pla == P_BLACK) {
        startGameAt = m;
      }
    }

    if(startGameAt >= sgfMoves.size()) {
      logger.write("Game too short overlaps with start in " + fileName + " turns " + Global::uint64ToString(sgfMoves.size()));
      reportSgfDone(false);
      return;
    }

    //Fastforward through initial bit before we start
    for(size_t m = 0; m<startGameAt; m++) {
      Move move = sgfMoves[m];

      //Quit out if according to our rules, we already finished the game, or we're somehow in a cleanup phase
      if(hist.isGameFinished || hist.encorePhase > 0) {
        logger.write("Game unexpectedly ended near start in " + fileName);
        reportSgfDone(false);
        return;
      }
      bool suc = hist.isLegal(board,move.loc,move.pla);
      if(!suc) {
        logger.write("Illegal move near start in " + fileName + " move " + Location::toString(move.loc, board.x_size, board.y_size));
        reportSgfDone(false);
        return;
      }
      bool preventEncore = false;
      hist.makeBoardMoveAssumeLegal(board,move.loc,move.pla,NULL,preventEncore);
    }

    //Use the actual first player of the game.
    nextPla = sgfMoves[startGameAt].pla;

    //Set up handicap behavior
    hist.setAssumeMultipleStartingBlackMovesAreHandicap(assumeMultipleStartingBlackMovesAreHandicap);
    hist.setOverrideNumHandicapStones(overrideNumHandicapStones);
    int numExtraBlack = hist.computeNumHandicapStones();

    const double drawEquivalentWinsForWhite = 0.5;
    const Player playoutDoublingAdvantagePla = C_EMPTY;
    const double playoutDoublingAdvantage = 0.0;

    vector<Board> boards;
    vector<BoardHistory> hists;
    vector<Player> nextPlas;
    vector<ValueTargets> whiteValueTargets;
    vector<Move> moves;

    for(size_t m = startGameAt; m<sgfMoves.size()+1; m++) {
      boards.push_back(board);
      hists.push_back(hist);
      nextPlas.push_back(nextPla);

      ValueTargets targets = makeWhiteValueTarget(
        board, hist, nextPla, drawEquivalentWinsForWhite, playoutDoublingAdvantagePla, playoutDoublingAdvantage, nnEval
      );
      whiteValueTargets.push_back(targets);

      if(m >= sgfMoves.size())
        break;
      //Quit out if according to our rules, we already finished the game, or we're somehow in a cleanup phase
      if(hist.isGameFinished || hist.encorePhase > 0)
        break;

      Move move = sgfMoves[m];
      moves.push_back(move);

      //Bad data checks
      if(move.pla != nextPla && m > 0) {
        logger.write("Bad SGF " + fileName + " early due to non-alternating players on turn " + Global::intToString(m));
        reportSgfDone(false);
        return;
      }
      bool suc = hist.isLegal(board,move.loc,move.pla);
      if(!suc) {
        logger.write("Illegal move in " + fileName + " turn " + Global::intToString(m) + " move " + Location::toString(move.loc, board.x_size, board.y_size));
        reportSgfDone(false);
        return;
      }
      bool preventEncore = false;
      hist.makeBoardMoveAssumeLegal(board,move.loc,move.pla,NULL,preventEncore);
      nextPla = getOpp(move.pla);
    }

    if(moves.size() < minGameLenToWrite) {
      logger.write("Game too short in " + fileName + " turns " + Global::uint64ToString(moves.size()));
      reportSgfDone(false);
      return;
    }

    // The turn that we will record data up to
    int endIdxForRecording = (int)moves.size();

    float valueTargetWeight = 0.0f;
    bool hasOwnershipTargets = false;
    bool hasForcedWinner = false; // Did we manually modify value targets to declare the winner?
    Color finalOwnership[Board::MAX_ARR_SIZE];
    Color finalFullArea[Board::MAX_ARR_SIZE];
    float finalWhiteScoring[Board::MAX_ARR_SIZE];
    std::fill(finalFullArea,finalFullArea+Board::MAX_ARR_SIZE,C_EMPTY);
    std::fill(finalOwnership,finalOwnership+Board::MAX_ARR_SIZE,C_EMPTY);
    std::fill(finalWhiteScoring,finalWhiteScoring+Board::MAX_ARR_SIZE,0.0f);

    // If this was a scored position, fill out the remaining turns with KataGo making moves, just to clean
    // up the position and carry out encore for JP-style rules.
    if(sgfGameEndedByScore) {
      bool gameFinishedProperly = false;
      for(int m = endIdxForRecording; m<endIdxForRecording+100; m++) {
        Loc moveLoc = search->runWholeSearchAndGetMove(nextPla);
        moves.push_back(Move(moveLoc,nextPla));

        bool preventEncore = false;
        hist.makeBoardMoveAssumeLegal(board,moveLoc,nextPla,NULL,preventEncore);
        nextPla = getOpp(nextPla);

        boards.push_back(board);
        hists.push_back(hist);
        nextPlas.push_back(nextPla);

        ValueTargets targets = makeWhiteValueTarget(
          board, hist, nextPla, drawEquivalentWinsForWhite, playoutDoublingAdvantagePla, playoutDoublingAdvantage, nnEval
        );
        whiteValueTargets.push_back(targets);

        if(hist.isGameFinished) {
          gameFinishedProperly = true;
          break;
        }
      }

      if(gameFinishedProperly) {
        // Does our result match the sgf recorded result?
        // If no, reduce weight a lot
        valueTargetWeight = (hist.winner == sgfGameWinner) ? 1.0f : 0.1f;

        // Ownership stuff!
        hasOwnershipTargets = true;
        hists[hists.size()-1].endAndScoreGameNow(board,finalOwnership);
        board.calculateArea(finalFullArea, true, true, true, hist.rules.multiStoneSuicideLegal);
        NNInputs::fillScoring(board,finalOwnership,hist.rules.taxRule == Rules::TAX_ALL,finalWhiteScoring);
      }
      else {
        logger.write("SGF " + fileName + " ended with score but it doesn't finish in a reasonable time");
        valueTargetWeight = 0.0f;
        hasOwnershipTargets = false;
      }
    }
    else if(sgfGameEndedByResign || sgfGameEndedByTime) {
      if(sgfGameWinner == P_WHITE) {
        valueTargetWeight = 2.0f * (float)std::max(0.0, whiteValueTargets[whiteValueTargets.size()-1].win - 0.5);

        // Assume white catches up in score at the same rate from turn 0 to 200 - does that leave white ahead?
        if(numExtraBlack > 0 && whiteValueTargets.size() < 200 && whiteValueTargets.size() > 30) {
          double change = whiteValueTargets[whiteValueTargets.size()-1].lead - whiteValueTargets[0].lead;
          double extrapolatedChange = change * (200.0 / whiteValueTargets.size());
          if(whiteValueTargets[0].lead + extrapolatedChange > 0.5) {
            // Black resigned while ahead, but white was catching up in a handicp game, so count full weight.
            valueTargetWeight = 1.0f;
          }
        }
        hasOwnershipTargets = false;
        hasForcedWinner = true;
        whiteValueTargets[whiteValueTargets.size()-1] = makeForcedWinnerValueTarget(P_WHITE);
      }
      else if(sgfGameWinner == P_BLACK) {
        // Player resigned when they were still ahead. Downweight and/or don't use.
        valueTargetWeight = 2.0f * (float)std::max(0.0, whiteValueTargets[whiteValueTargets.size()-1].loss - 0.5);
        hasOwnershipTargets = false;
        hasForcedWinner = true;
        whiteValueTargets[whiteValueTargets.size()-1] = makeForcedWinnerValueTarget(P_BLACK);
      }
      else {
        ASSERT_UNREACHABLE;
      }
      // For games ending by time, treat them similarly to resigns except use a lower and sharper weighting for outcome.
      // Square so that we require the winrate to have been more clear to accept such a loss, and further downeight by constant factor.
      valueTargetWeight = 0.5f * valueTargetWeight * valueTargetWeight;
    }
    else {
      valueTargetWeight = 0.0f;
      hasOwnershipTargets = false;
    }

    for(size_t m = 0; m<endIdxForRecording; m++) {
      int turnIdx = (int)m;
      float targetWeight = 1.0;
      int64_t unreducedNumVisits = 0;
      std::vector<PolicyTargetMove> policyTarget0;
      policyTarget0.push_back(PolicyTargetMove(moves[m].loc,1));
      bool hasPolicyTarget1 = false;
      std::vector<PolicyTargetMove> policyTarget1;
      if(m+1 < moves.size()) {
        hasPolicyTarget1 = true;
        policyTarget1.push_back(PolicyTargetMove(moves[m+1].loc,1));
      }
      const double policySurprise = 0;
      const double policyEntropy = 0;
      const double searchEntropy = 0;

      NNRawStats nnRawStats;
      nnRawStats.whiteWinLoss = 0;
      nnRawStats.whiteScoreMean = 0;
      nnRawStats.policyEntropy = 0;

      // If we manually hacked the td value, prevent an abrupt game end with a forced value from biasing it too much
      // by disabling the td targets if we get too close.
      float tdValueTargetWeight = valueTargetWeight;
      if(hasForcedWinner && whiteValueTargets.size() - m < 50) {
        tdValueTargetWeight = 0.0f;
      }

      const bool isSidePosition = false;
      const int numNeuralNetsBehindLatest = 0;
      const Hash128 gameHash;
      const std::vector<ChangedNeuralNet*> changedNeuralNets;
      const bool hitTurnLimit = false;
      const int mode = 0;

      if((nextPlas[m] == P_BLACK && shouldTrainB) || (nextPlas[m] == P_WHITE && shouldTrainW)) {
        dataBuffer->addRow(
          boards[m],
          hists[m],
          nextPlas[m],
          hists[0],
          hists[hists.size()-1],
          turnIdx,
          targetWeight,
          unreducedNumVisits,
          &policyTarget0,
          (hasPolicyTarget1 ? &policyTarget1 : NULL),
          policySurprise,
          policyEntropy,
          searchEntropy,
          whiteValueTargets,
          turnIdx,
          valueTargetWeight,
          tdValueTargetWeight,
          nnRawStats,
          &boards[boards.size()-1],
          (hasOwnershipTargets ? finalFullArea : NULL),
          (hasOwnershipTargets ? finalOwnership : NULL),
          (hasOwnershipTargets ? finalWhiteScoring : NULL),
          &boards,
          isSidePosition,
          numNeuralNetsBehindLatest,
          drawEquivalentWinsForWhite,
          playoutDoublingAdvantagePla,
          playoutDoublingAdvantage,
          gameHash,
          changedNeuralNets,
          hitTurnLimit,
          numExtraBlack,
          mode,
          rand
        );
      }
    }
    // void writeToZipFile(const std::string& fileName);
    reportSgfDone(true);
  };

  Parallel::iterRange(numThreads, std::min(maxFilesToLoad,sgfFiles.size()), std::function<void(int,size_t)>(processSgf));

  auto printGameCountsMap = [&](const std::map<string,int64_t> counts, const string& label) {
    logger.write("===================================================");
    logger.write("Counts by " + label);
    for(const auto& keyAndCount: counts) {
      logger.write(keyAndCount.first + ": " + Global::int64ToString(keyAndCount.second));
    }
    logger.write("===================================================");
  };
  printGameCountsMap(gameCountByUsername,"username");
  printGameCountsMap(gameCountByRank,"rank");
  printGameCountsMap(gameCountByTimeControl,"time control");
  printGameCountsMap(gameCountByRules,"rules");
  printGameCountsMap(gameCountByKomi,"komi");
  printGameCountsMap(gameCountByHandicap,"handicap");
  printGameCountsMap(gameCountByResult,"result");

  logger.write(nnEval->getModelFileName());
  logger.write("NN rows: " + Global::int64ToString(nnEval->numRowsProcessed()));
  logger.write("NN batches: " + Global::int64ToString(nnEval->numBatchesProcessed()));
  logger.write("NN avg batch size: " + Global::doubleToString(nnEval->averageProcessedBatchSize()));

  logger.write("All done");

  for(int i = 0; i<numThreads; i++)
    delete threadDataBuffers[i];

  delete nnEval;
  NeuralNet::globalCleanup();
  ScoreValue::freeTables();
  return 0;
}
