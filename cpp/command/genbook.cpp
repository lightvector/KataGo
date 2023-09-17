#include "../core/global.h"
#include "../core/makedir.h"
#include "../core/config_parser.h"
#include "../core/fileutils.h"
#include "../core/timer.h"
#include "../core/threadsafequeue.h"
#include "../dataio/poswriter.h"
#include "../dataio/sgf.h"
#include "../dataio/files.h"
#include "../book/book.h"
#include "../search/searchnode.h"
#include "../search/asyncbot.h"
#include "../program/setup.h"
#include "../program/playutils.h"
#include "../program/play.h"
#include "../command/commandline.h"
#include "../main.h"

#include <chrono>
#include <csignal>

//------------------------
#include "../core/using.h"
//------------------------

static std::atomic<bool> sigReceived(false);
static std::atomic<bool> shouldStop(false);
static void signalHandler(int signal)
{
  if(signal == SIGINT || signal == SIGTERM) {
    sigReceived.store(true);
    shouldStop.store(true);
  }
}

static double getMaxPolicy(float policyProbs[NNPos::MAX_NN_POLICY_SIZE]) {
  double maxPolicy = 0.0;
  for(int i = 0; i<NNPos::MAX_NN_POLICY_SIZE; i++)
    if(policyProbs[i] > maxPolicy)
      maxPolicy = policyProbs[i];
  return maxPolicy;
}

static void optimizeSymmetriesInplace(std::vector<SymBookNode>& nodes, Rand* rand, Logger& logger) {
  std::vector<std::unique_ptr<Board>> boards;
  {
    BoardHistory histBuf;
    std::vector<Loc> moveHistoryBuf;
    for(SymBookNode& node: nodes) {
      if(node.getBoardHistoryReachingHere(histBuf,moveHistoryBuf)) {
        boards.push_back(std::make_unique<Board>(histBuf.getRecentBoard(0)));
      }
      else {
        logger.write("WARNING: Failed to get board history reaching node, probably there is some bug");
        logger.write("BookHash of node optimizing symmetries: " + node.hash().toString());
        throw StringError("Terminating");
      }
    }
  }

  assert(nodes.size() < 0x7FFFFFFFU);
  std::vector<uint32_t> perm(nodes.size());
  if(rand != nullptr)
    rand->fillShuffledUIntRange(perm.size(), perm.data());
  else {
    for(size_t i = 0; i<perm.size(); i++)
      perm[i] = (uint32_t)i;
  }

  double diffBuf[SymmetryHelpers::NUM_SYMMETRIES];
  double similaritySumBuf[SymmetryHelpers::NUM_SYMMETRIES];
  const double maxDifferenceToReport = 12;

  // Iterate through all nodes in random order and for each one find its best symmetry
  std::vector<int> bestSymmetries(nodes.size());
  for(size_t i = 0; i<perm.size(); i++) {
    const Board& nodeBoard = *(boards[perm[i]]);

    std::fill(similaritySumBuf, similaritySumBuf+SymmetryHelpers::NUM_SYMMETRIES, 0.0);

    // Iterate over all previously symmetrized nodes, up to at most 100, and accumulate similarity
    for(size_t j = 0; j<i && j < 100; j++) {
      const Board& otherBoard = *(boards[perm[j]]);
      int otherBoardBestSymmetry = bestSymmetries[perm[j]];
      SymmetryHelpers::getSymmetryDifferences(nodeBoard, otherBoard, maxDifferenceToReport, diffBuf);
      for(int symmetry = 0; symmetry < SymmetryHelpers::NUM_SYMMETRIES; symmetry++) {
        // diffBuff[symmetry] has the similarity between nodeBoard * symmetry  and otherBoard.
        // Which is the same as the similarity between nodeboard * compose(symmetry,otherBoardBestSymmetry) and otherBoard * otherBoardBestSymmetry.
        // The latter is what we want, since that's what otherBoard will actually end up as after this whole function is done.
        // For similarity, use quadratic harmonic
        similaritySumBuf[SymmetryHelpers::compose(symmetry, otherBoardBestSymmetry)] += 1.0 / ((0.01 + diffBuf[symmetry]) * (0.01 + diffBuf[symmetry]));
      }
    }

    double bestSimilarity = similaritySumBuf[0];
    int bestSymmetry = 0;
    for(int symmetry = 1; symmetry < SymmetryHelpers::NUM_SYMMETRIES; symmetry++) {
      if(similaritySumBuf[symmetry] > bestSimilarity) {
        bestSimilarity = similaritySumBuf[symmetry];
        bestSymmetry = symmetry;
      }
    }
    bestSymmetries[perm[i]] = bestSymmetry;
  }

  for(size_t i = 0; i<nodes.size(); i++) {
    nodes[i] = nodes[i].applySymmetry(bestSymmetries[i]);
  }
}

static void maybeParseBonusFile(
  const std::string& bonusFile,
  int boardSizeX,
  int boardSizeY,
  Rules rules,
  int repBound,
  double bonusFileScale,
  Logger& logger,
  std::map<BookHash,double>& bonusByHash,
  std::map<BookHash,double>& expandBonusByHash,
  std::map<BookHash,double>& visitsRequiredByHash,
  std::map<BookHash,int>& branchRequiredByHash,
  Board& bonusInitialBoard,
  Player& bonusInitialPla
) {
  bonusInitialBoard = Board(boardSizeX,boardSizeY);
  bonusInitialPla = P_BLACK;
  if(bonusFile != "") {
    Sgf* sgf = Sgf::loadFile(bonusFile);
    bool flipIfPassOrWFirst = false;
    bool allowGameOver = false;
    Rand seedRand("bonusByHash");
    sgf->iterAllPositions(
      flipIfPassOrWFirst, allowGameOver, &seedRand, [&](Sgf::PositionSample& unusedSample, const BoardHistory& sgfHist, const string& comments) {
        (void)unusedSample;
        if(comments.size() > 0 && (
             comments.find("BONUS") != string::npos ||
             comments.find("EXPAND") != string::npos ||
             comments.find("VISITS") != string::npos ||
             comments.find("BRANCH") != string::npos
           )
        ) {
          BoardHistory hist(sgfHist.initialBoard, sgfHist.initialPla, rules, sgfHist.initialEncorePhase);
          Board board = hist.initialBoard;
          for(size_t i = 0; i<sgfHist.moveHistory.size(); i++) {
            bool suc = hist.makeBoardMoveTolerant(board, sgfHist.moveHistory[i].loc, sgfHist.moveHistory[i].pla);
            if(!suc)
              return;
          }

          auto parseCommand = [&comments,&board](const char* commandName, double& ret) {
            if(comments.find(commandName) != string::npos) {
              double bonus;
              try {
                vector<string> nextWords = Global::split(Global::trim(comments.substr(comments.find(commandName)+std::strlen(commandName))));
                if(nextWords.size() <= 0)
                  throw StringError("Could not parse " + string(commandName) + " value");
                bonus = Global::stringToDouble(nextWords[0]);
              }
              catch(const StringError& e) {
                cerr << board << endl;
                throw e;
              }
              ret = bonus;
              return true;
            }
            return false;
          };

          double ret = 0.0;
          BookHash hashRet;
          int symmetryToAlignRet;
          vector<int> symmetriesRet;
          if(parseCommand("BONUS",ret)) {
            if(!std::isfinite(ret) || ret < 0 || ret > 10000)
              throw StringError("Invalid BONUS: " + Global::doubleToString(ret));
            for(int bookVersion = 1; bookVersion <= Book::LATEST_BOOK_VERSION; bookVersion++) {
              BookHash::getHashAndSymmetry(hist, repBound, hashRet, symmetryToAlignRet, symmetriesRet, bookVersion);
              if(bonusByHash.find(hashRet) != bonusByHash.end())
                bonusByHash[hashRet] = std::max(bonusByHash[hashRet], ret * bonusFileScale);
              else
                bonusByHash[hashRet] = ret * bonusFileScale;
              logger.write("Adding bonus " + Global::doubleToString(ret * bonusFileScale) + " to hash " + hashRet.toString());
            }
          }

          if(parseCommand("EXPAND",ret)) {
            if(!std::isfinite(ret) || ret < 0 || ret > 10000)
              throw StringError("Invalid EXPAND: " + Global::doubleToString(ret));
            for(int bookVersion = 1; bookVersion <= Book::LATEST_BOOK_VERSION; bookVersion++) {
              BookHash::getHashAndSymmetry(hist, repBound, hashRet, symmetryToAlignRet, symmetriesRet, bookVersion);
              if(expandBonusByHash.find(hashRet) != expandBonusByHash.end())
                expandBonusByHash[hashRet] = std::max(expandBonusByHash[hashRet], ret * bonusFileScale);
              else
                expandBonusByHash[hashRet] = ret * bonusFileScale;
              logger.write("Adding expand bonus " + Global::doubleToString(ret * bonusFileScale) + " to hash " + hashRet.toString());
            }
          }

          if(parseCommand("VISITS",ret)) {
            if(!std::isfinite(ret) || ret < 0)
              throw StringError("Invalid VISITS: " + Global::doubleToString(ret));
            for(int bookVersion = 1; bookVersion <= Book::LATEST_BOOK_VERSION; bookVersion++) {
              BookHash::getHashAndSymmetry(hist, repBound, hashRet, symmetryToAlignRet, symmetriesRet, bookVersion);
              if(visitsRequiredByHash.find(hashRet) != visitsRequiredByHash.end())
                visitsRequiredByHash[hashRet] = std::max(visitsRequiredByHash[hashRet], ret * bonusFileScale);
              else
                visitsRequiredByHash[hashRet] = ret * bonusFileScale;
              logger.write("Adding required visits " + Global::doubleToString(ret * bonusFileScale) + " to hash " + hashRet.toString());
            }
          }

          if(parseCommand("BRANCH",ret)) {
            if(!std::isfinite(ret) || ret < 0 || ret > 200)
              throw StringError("Invalid BRANCH: " + Global::doubleToString(ret));
            for(int bookVersion = 1; bookVersion <= Book::LATEST_BOOK_VERSION; bookVersion++) {
              BookHash::getHashAndSymmetry(hist, repBound, hashRet, symmetryToAlignRet, symmetriesRet, bookVersion);
              if(branchRequiredByHash.find(hashRet) != branchRequiredByHash.end())
                branchRequiredByHash[hashRet] = std::max(branchRequiredByHash[hashRet], (int)ret);
              else
                branchRequiredByHash[hashRet] = (int)ret;
              logger.write("Adding required branching factor " + Global::intToString((int)ret) + " to hash " + hashRet.toString());
            }
          }

        }
      }
    );

    XYSize xySize = sgf->getXYSize();
    if(boardSizeX != xySize.x || boardSizeY != xySize.y)
      throw StringError("Board size in config does not match the board size of the bonus file");
    vector<Move> placements;
    sgf->getPlacements(placements,boardSizeX,boardSizeY);
    bool suc = bonusInitialBoard.setStonesFailIfNoLibs(placements);
    if(!suc)
      throw StringError("Invalid placements in sgf");
    bonusInitialPla = sgf->getFirstPlayerColor();
    delete sgf;
  }
}


int MainCmds::genbook(const vector<string>& args) {
  Board::initHash();
  ScoreValue::initTables();

  ConfigParser cfg;
  string modelFile;
  string htmlDir;
  string bookFile;
  string traceBookFile;
  string traceSgfFile;
  string logFile;
  string bonusFile;
  int numIterations;
  int saveEveryIterations;
  double traceBookMinVisits;
  bool allowChangingBookParams;
  bool htmlDevMode;
  double htmlMinVisits;
  try {
    KataGoCommandLine cmd("Generate opening book");
    cmd.addConfigFileArg("","",true);
    cmd.addModelFileArg();
    cmd.addOverrideConfigArg();

    TCLAP::ValueArg<string> htmlDirArg("","html-dir","HTML directory to export to, at the end of -num-iters",false,string(),"DIR");
    TCLAP::ValueArg<string> bookFileArg("","book-file","Book file to write to or continue expanding",true,string(),"FILE");
    TCLAP::ValueArg<string> traceBookFileArg("","trace-book-file","Other book file we should copy all the lines from",false,string(),"FILE");
    TCLAP::ValueArg<string> traceSgfFileArg("","trace-sgf-file","Other sgf file we should copy all the lines from",false,string(),"FILE");
    TCLAP::ValueArg<string> logFileArg("","log-file","Log file to write to",true,string(),"DIR");
    TCLAP::ValueArg<string> bonusFileArg("","bonus-file","SGF of bonuses marked",false,string(),"DIR");
    TCLAP::ValueArg<int> numIterationsArg("","num-iters","Number of iterations to expand book",true,0,"N");
    TCLAP::ValueArg<int> saveEveryIterationsArg("","save-every","Number of iterations per save to book file",true,0,"N");
    TCLAP::ValueArg<double> traceBookMinVisitsArg("","trace-book-min-visits","Require >= this many visits for copying from traceBookFile",false,0.0,"N");
    TCLAP::SwitchArg allowChangingBookParamsArg("","allow-changing-book-params","Allow changing book params");
    TCLAP::SwitchArg htmlDevModeArg("","html-dev-mode","Denser debug output for html");
    TCLAP::ValueArg<double> htmlMinVisitsArg("","html-min-visits","Require >= this many visits to export a position to html",false,0.0,"N");
    cmd.add(htmlDirArg);
    cmd.add(bookFileArg);
    cmd.add(traceBookFileArg);
    cmd.add(traceSgfFileArg);
    cmd.add(logFileArg);
    cmd.add(bonusFileArg);
    cmd.add(numIterationsArg);
    cmd.add(saveEveryIterationsArg);
    cmd.add(traceBookMinVisitsArg);
    cmd.add(allowChangingBookParamsArg);
    cmd.add(htmlDevModeArg);
    cmd.add(htmlMinVisitsArg);

    cmd.parseArgs(args);

    cmd.getConfig(cfg);
    modelFile = cmd.getModelFile();
    htmlDir = htmlDirArg.getValue();
    bookFile = bookFileArg.getValue();
    traceBookFile = traceBookFileArg.getValue();
    traceSgfFile = traceSgfFileArg.getValue();
    logFile = logFileArg.getValue();
    bonusFile = bonusFileArg.getValue();
    numIterations = numIterationsArg.getValue();
    saveEveryIterations = saveEveryIterationsArg.getValue();
    traceBookMinVisits = traceBookMinVisitsArg.getValue();
    allowChangingBookParams = allowChangingBookParamsArg.getValue();
    htmlDevMode = htmlDevModeArg.getValue();
    htmlMinVisits = htmlMinVisitsArg.getValue();
  }
  catch (TCLAP::ArgException &e) {
    cerr << "Error: " << e.error() << " for argument " << e.argId() << endl;
    return 1;
  }

  Rand rand;
  const bool logToStdoutDefault = true;
  Logger logger(&cfg, logToStdoutDefault);
  logger.addFile(logFile);

  const bool loadKomiFromCfg = true;
  Rules rules = Setup::loadSingleRules(cfg,loadKomiFromCfg);

  const SearchParams params = Setup::loadSingleParams(cfg,Setup::SETUP_FOR_GTP);

  const int boardSizeX = cfg.getInt("boardSizeX",2,Board::MAX_LEN);
  const int boardSizeY = cfg.getInt("boardSizeY",2,Board::MAX_LEN);
  const int repBound = cfg.getInt("repBound",3,1000);

  BookParams cfgParams = BookParams::loadFromCfg(cfg, params.maxVisits);

  const double bonusFileScale = cfg.contains("bonusFileScale") ? cfg.getDouble("bonusFileScale",0.0,1000000.0) : 1.0;

  const double randomizeParamsStdev = cfg.contains("randomizeParamsStdev") ? cfg.getDouble("randomizeParamsStdev",0.0,2.0) : 0.0;

  const bool logSearchInfo = cfg.getBool("logSearchInfo");
  const string rulesLabel = cfg.getString("rulesLabel");
  const string rulesLink = cfg.getString("rulesLink");

  const int64_t minTreeVisitsToRecord =
    cfg.contains("minTreeVisitsToRecord") ? cfg.getInt64("minTreeVisitsToRecord", (int64_t)1, (int64_t)1 << 50) : params.maxVisits;
  const int maxDepthToRecord =
    cfg.contains("maxDepthToRecord") ? cfg.getInt("maxDepthToRecord", 1, 100) : 1;
  const int64_t maxVisitsForLeaves =
    cfg.contains("maxVisitsForLeaves") ? cfg.getInt64("maxVisitsForLeaves", (int64_t)1, (int64_t)1 << 50) : (params.maxVisits+1) / 2;

  const int numGameThreads = cfg.getInt("numGameThreads",1,1000);
  const int numToExpandPerIteration = cfg.getInt("numToExpandPerIteration",1,10000000);

  std::map<BookHash,double> bonusByHash;
  std::map<BookHash,double> expandBonusByHash;
  std::map<BookHash,double> visitsRequiredByHash;
  std::map<BookHash,int> branchRequiredByHash;
  Board bonusInitialBoard;
  Player bonusInitialPla;

  maybeParseBonusFile(
    bonusFile,
    boardSizeX,
    boardSizeY,
    rules,
    repBound,
    bonusFileScale,
    logger,
    bonusByHash,
    expandBonusByHash,
    visitsRequiredByHash,
    branchRequiredByHash,
    bonusInitialBoard,
    bonusInitialPla
  );

  const double wideRootNoiseBookExplore = cfg.contains("wideRootNoiseBookExplore") ? cfg.getDouble("wideRootNoiseBookExplore",0.0,5.0) : params.wideRootNoise;
  const double cpuctExplorationLogBookExplore = cfg.contains("cpuctExplorationLogBookExplore") ? cfg.getDouble("cpuctExplorationLogBookExplore",0.0,10.0) : params.cpuctExplorationLog;
  NNEvaluator* nnEval;
  {
    Setup::initializeSession(cfg);
    const int maxConcurrentEvals = numGameThreads * params.numThreads * 2 + 16; // * 2 + 16 just to give plenty of headroom
    const int expectedConcurrentEvals = numGameThreads * params.numThreads;
    const int defaultMaxBatchSize = std::max(8,((numGameThreads * params.numThreads+3)/4)*4);
    const bool defaultRequireExactNNLen = true;
    const bool disableFP16 = false;
    const string expectedSha256 = "";
    nnEval = Setup::initializeNNEvaluator(
      modelFile,modelFile,expectedSha256,cfg,logger,rand,maxConcurrentEvals,expectedConcurrentEvals,
      boardSizeX,boardSizeY,defaultMaxBatchSize,defaultRequireExactNNLen,disableFP16,
      Setup::SETUP_FOR_ANALYSIS
    );
  }
  logger.write("Loaded neural net");

  vector<Search*> searches;
  for(int i = 0; i<numGameThreads; i++) {
    string searchRandSeed = Global::uint64ToString(rand.nextUInt64());
    searches.push_back(new Search(params, nnEval, &logger, searchRandSeed));
  }

  // Check for unused config keys
  cfg.warnUnusedKeys(cerr,&logger);

  if(htmlDir != "")
    MakeDir::make(htmlDir);

  Book* book;
  bool bookFileExists;
  {
    std::ifstream infile;
    bookFileExists = FileUtils::tryOpen(infile,bookFile);
  }
  if(bookFileExists) {
    book = Book::loadFromFile(bookFile);
    if(
      boardSizeX != book->getInitialHist().getRecentBoard(0).x_size ||
      boardSizeY != book->getInitialHist().getRecentBoard(0).y_size ||
      repBound != book->repBound ||
      rules != book->getInitialHist().rules
    ) {
      throw StringError("Book parameters do not match");
    }
    if(bonusFile != "") {
      if(!bonusInitialBoard.isEqualForTesting(book->getInitialHist().getRecentBoard(0), false, false))
        throw StringError(
          "Book initial board and initial board in bonus sgf file do not match\n" +
          Board::toStringSimple(book->getInitialHist().getRecentBoard(0),'\n') + "\n" +
          Board::toStringSimple(bonusInitialBoard,'\n')
        );
      if(bonusInitialPla != book->initialPla)
        throw StringError(
          "Book initial player and initial player in bonus sgf file do not match\n" +
          PlayerIO::playerToString(book->initialPla) + " book \n" +
          PlayerIO::playerToString(bonusInitialPla) + " bonus"
        );
    }

    if(!allowChangingBookParams) {
      BookParams existingBookParams = book->getParams();
      if(
        cfgParams.errorFactor != existingBookParams.errorFactor ||
        cfgParams.costPerMove != existingBookParams.costPerMove ||
        cfgParams.costPerUCBWinLossLoss != existingBookParams.costPerUCBWinLossLoss ||
        cfgParams.costPerUCBWinLossLossPow3 != existingBookParams.costPerUCBWinLossLossPow3 ||
        cfgParams.costPerUCBWinLossLossPow7 != existingBookParams.costPerUCBWinLossLossPow7 ||
        cfgParams.costPerUCBScoreLoss != existingBookParams.costPerUCBScoreLoss ||
        cfgParams.costPerLogPolicy != existingBookParams.costPerLogPolicy ||
        cfgParams.costPerMovesExpanded != existingBookParams.costPerMovesExpanded ||
        cfgParams.costPerSquaredMovesExpanded != existingBookParams.costPerSquaredMovesExpanded ||
        cfgParams.costWhenPassFavored != existingBookParams.costWhenPassFavored ||
        cfgParams.bonusPerWinLossError != existingBookParams.bonusPerWinLossError ||
        cfgParams.bonusPerScoreError != existingBookParams.bonusPerScoreError ||
        cfgParams.bonusPerSharpScoreDiscrepancy != existingBookParams.bonusPerSharpScoreDiscrepancy ||
        cfgParams.bonusPerExcessUnexpandedPolicy != existingBookParams.bonusPerExcessUnexpandedPolicy ||
        cfgParams.bonusPerUnexpandedBestWinLoss != existingBookParams.bonusPerUnexpandedBestWinLoss ||
        cfgParams.bonusForWLPV1 != existingBookParams.bonusForWLPV1 ||
        cfgParams.bonusForWLPV2 != existingBookParams.bonusForWLPV2 ||
        cfgParams.bonusForWLPVFinalProp != existingBookParams.bonusForWLPVFinalProp ||
        cfgParams.bonusForBiggestWLCost != existingBookParams.bonusForBiggestWLCost ||
        cfgParams.scoreLossCap != existingBookParams.scoreLossCap ||
        cfgParams.earlyBookCostReductionFactor != existingBookParams.earlyBookCostReductionFactor ||
        cfgParams.earlyBookCostReductionLambda != existingBookParams.earlyBookCostReductionLambda ||
        cfgParams.utilityPerScore != existingBookParams.utilityPerScore ||
        cfgParams.policyBoostSoftUtilityScale != existingBookParams.policyBoostSoftUtilityScale ||
        cfgParams.utilityPerPolicyForSorting != existingBookParams.utilityPerPolicyForSorting ||
        cfgParams.maxVisitsForReExpansion != existingBookParams.maxVisitsForReExpansion ||
        cfgParams.visitsScale != existingBookParams.visitsScale ||
        cfgParams.sharpScoreOutlierCap != existingBookParams.sharpScoreOutlierCap
      ) {
        throw StringError("Book parameters do not match");
      }
    }
    else {
      book->setParams(cfgParams);
    }
    logger.write("Loaded preexisting book with " + Global::uint64ToString(book->size()) + " nodes from " + bookFile);
    logger.write("Book version = " + Global::intToString(book->bookVersion));
  }
  else {
    {
      ostringstream bout;
      Board::printBoard(bout, bonusInitialBoard, Board::NULL_LOC, NULL);
      logger.write("Initializing new book with starting position:\n" + bout.str());
    }
    book = new Book(
      Book::LATEST_BOOK_VERSION,
      bonusInitialBoard,
      rules,
      bonusInitialPla,
      repBound,
      cfgParams
    );
    logger.write("Creating new book at " + bookFile);
    book->saveToFile(bookFile);
    ofstream out;
    FileUtils::open(out,bookFile + ".cfg");
    out << cfg.getContents() << endl;
    out.close();
  }

  if(traceBookFile.size() > 0 && traceSgfFile.size() > 0)
    throw StringError("Cannot trace book and sgf at the same time");

  Book* traceBook = NULL;
  if(traceBookFile.size() > 0) {
    if(numIterations > 0)
      throw StringError("Cannot specify iterations and trace book at the same time");
    traceBook = Book::loadFromFile(traceBookFile);
    traceBook->recomputeEverything();
    logger.write("Loaded trace book with " + Global::uint64ToString(traceBook->size()) + " nodes from " + traceBookFile);
    logger.write("traceBookMinVisits = " + Global::doubleToString(traceBookMinVisits));
  }

  book->setBonusByHash(bonusByHash);
  book->setExpandBonusByHash(expandBonusByHash);
  book->setVisitsRequiredByHash(visitsRequiredByHash);
  book->setBranchRequiredByHash(branchRequiredByHash);
  book->recomputeEverything();

  if(!std::atomic_is_lock_free(&shouldStop))
    throw StringError("shouldStop is not lock free, signal-quitting mechanism for terminating matches will NOT work!");
  std::signal(SIGINT, signalHandler);
  std::signal(SIGTERM, signalHandler);

  const PrintTreeOptions options;
  const Player perspective = P_WHITE;

  // ClockTimer timer;
  std::mutex bookMutex;

  // Avoid all moves that are currently in the book on this node,
  // unless allowReExpansion is true and this node qualifies for the visit threshold for allowReExpansion and
  // to re-search already searched moves freshly.
  // Mark avoidMoveUntilByLoc to be passed to search so that we only search new stuff.
  auto findNewMovesAlreadyLocked = [&](
    const BoardHistory& hist,
    ConstSymBookNode constNode,
    bool allowReExpansion,
    std::vector<int>& avoidMoveUntilByLoc,
    bool& isReExpansion
  ) {
    avoidMoveUntilByLoc = std::vector<int>(Board::MAX_ARR_SIZE,0);
    isReExpansion = allowReExpansion && constNode.canReExpand() && constNode.recursiveValues().visits <= book->getParams().maxVisitsForReExpansion;
    Player pla = hist.presumedNextMovePla;
    Board board = hist.getRecentBoard(0);
    bool hasAtLeastOneLegalNewMove = false;
    for(Loc moveLoc = 0; moveLoc < Board::MAX_ARR_SIZE; moveLoc++) {
      if(hist.isLegal(board,moveLoc,pla)) {
        if(!isReExpansion && constNode.isMoveInBook(moveLoc))
          avoidMoveUntilByLoc[moveLoc] = 1;
        else
          hasAtLeastOneLegalNewMove = true;
      }
    }
    return hasAtLeastOneLegalNewMove;
  };

  auto setParamsAndAvoidMoves = [&](Search* search, SearchParams thisParams, const std::vector<int>& avoidMoveUntilByLoc) {
    search->setParams(thisParams);
    search->setAvoidMoveUntilByLoc(avoidMoveUntilByLoc, avoidMoveUntilByLoc);
    search->setAvoidMoveUntilRescaleRoot(true);
  };

  auto setNodeThisValuesNoMoves = [&](SymBookNode node) {
    std::lock_guard<std::mutex> lock(bookMutex);
    BookValues& nodeValues = node.thisValuesNotInBook();
    if(node.pla() == P_WHITE) {
      nodeValues.winLossValue = -1e20;
      nodeValues.scoreMean = -1e20;
      nodeValues.sharpScoreMean =  -1e20;
    }
    else {
      nodeValues.winLossValue = 1e20;
      nodeValues.scoreMean = 1e20;
      nodeValues.sharpScoreMean =  1e20;
    }
    nodeValues.winLossError = 0.0;
    nodeValues.scoreError = 0.0;
    nodeValues.scoreStdev = 0.0;
    nodeValues.maxPolicy = 0.0;
    nodeValues.weight = 0.0;
    nodeValues.visits = 0.0;

    node.canExpand() = false;
  };

  auto setNodeThisValuesTerminal = [&](SymBookNode node, const BoardHistory& hist) {
    assert(hist.isGameFinished);

    std::lock_guard<std::mutex> lock(bookMutex);
    BookValues& nodeValues = node.thisValuesNotInBook();
    if(hist.isNoResult) {
      nodeValues.winLossValue = 0.0;
      nodeValues.scoreMean = 0.0;
      nodeValues.sharpScoreMean = 0.0;
    }
    else {
      if(hist.winner == P_WHITE) {
        assert(hist.finalWhiteMinusBlackScore > 0.0);
        nodeValues.winLossValue = 1.0;
      }
      else if(hist.winner == P_BLACK) {
        assert(hist.finalWhiteMinusBlackScore < 0.0);
        nodeValues.winLossValue = -1.0;
      }
      else {
        assert(hist.finalWhiteMinusBlackScore == 0.0);
        nodeValues.winLossValue = 0.0;
      }
      nodeValues.scoreMean = hist.finalWhiteMinusBlackScore;
      nodeValues.sharpScoreMean = hist.finalWhiteMinusBlackScore;
    }

    nodeValues.winLossError = 0.0;
    nodeValues.scoreError = 0.0;
    nodeValues.scoreStdev = 0.0;
    nodeValues.maxPolicy = 1.0;
    double visits = maxVisitsForLeaves;
    nodeValues.weight = visits;
    nodeValues.visits = visits;

    node.canExpand() = false;
  };

  auto setNodeThisValuesFromFinishedSearch = [&](
    SymBookNode node,
    Search* search,
    const SearchNode* searchNode,
    const Board& board,
    const BoardHistory& hist,
    const std::vector<int>& avoidMoveUntilByLoc
  ) {
    // Get root values
    ReportedSearchValues remainingSearchValues;
    bool getSuc = search->getPrunedNodeValues(searchNode,remainingSearchValues);
    // Something is bad if this is false, since we should be searching with positive visits
    // or otherwise this searchNode must be a terminal node with visits from a deeper search.
    assert(getSuc);
    (void)getSuc;
    double sharpScore = 0.0;
    // cout << "Calling sharpscore " << timer.getSeconds() << endl;
    getSuc = search->getSharpScore(searchNode,sharpScore);
    // cout << "Done sharpscore " << timer.getSeconds() << endl;
    assert(getSuc);
    (void)getSuc;

    // cout << "Calling shallowAvg " << timer.getSeconds() << endl;
    std::pair<double,double> errors = search->getShallowAverageShorttermWLAndScoreError(searchNode);
    // cout << "Done shallowAvg " << timer.getSeconds() << endl;

    // Use full symmetry for the policy for nodes we record for the book
    bool includeOwnerMap = false;
    // cout << "Calling full nn " << timer.getSeconds() << endl;
    std::shared_ptr<NNOutput> fullSymNNOutput = PlayUtils::getFullSymmetryNNOutput(board, hist, node.pla(), includeOwnerMap, search->nnEvaluator);
    float policyProbs[NNPos::MAX_NN_POLICY_SIZE];
    std::copy(fullSymNNOutput->policyProbs, fullSymNNOutput->policyProbs+NNPos::MAX_NN_POLICY_SIZE, policyProbs);
    // cout << "Done full nn " << timer.getSeconds() << endl;

    // Zero out all the policies for moves we already have, we want the max *remaining* policy
    if(avoidMoveUntilByLoc.size() > 0) {
      assert(avoidMoveUntilByLoc.size() == Board::MAX_ARR_SIZE);
      for(Loc loc = 0; loc<Board::MAX_ARR_SIZE; loc++) {
        if(avoidMoveUntilByLoc[loc] > 0) {
          int pos = search->getPos(loc);
          assert(pos >= 0 && pos < NNPos::MAX_NN_POLICY_SIZE);
          policyProbs[pos] = -1;
        }
      }
    }
    double maxPolicy = getMaxPolicy(policyProbs);
    assert(maxPolicy >= 0.0);

    // LOCK BOOK AND UPDATE -------------------------------------------------------
    std::lock_guard<std::mutex> lock(bookMutex);

    // Record those values to the book
    BookValues& nodeValues = node.thisValuesNotInBook();
    nodeValues.winLossValue = remainingSearchValues.winLossValue;
    nodeValues.scoreMean = remainingSearchValues.expectedScore;
    nodeValues.sharpScoreMean = sharpScore;
    nodeValues.winLossError = errors.first;
    nodeValues.scoreError = errors.second;
    nodeValues.scoreStdev = remainingSearchValues.expectedScoreStdev;

    nodeValues.maxPolicy = maxPolicy;
    nodeValues.weight = remainingSearchValues.weight;
    nodeValues.visits = (double)remainingSearchValues.visits;
  };


  // Perform a short search and update thisValuesNotInBook for a node
  auto searchAndUpdateNodeThisValues = [&](Search* search, SymBookNode node) {
    ConstSymBookNode constNode(node);
    BoardHistory hist;
    std::vector<int> symmetries;
    {
      std::lock_guard<std::mutex> lock(bookMutex);
      std::vector<Loc> moveHistory;
      bool suc = node.getBoardHistoryReachingHere(hist,moveHistory);
      if(!suc) {
        logger.write("WARNING: Failed to get board history reaching node when trying to export to trace book, probably there is some bug");
        logger.write("or else some hash collision or something else is wrong.");
        logger.write("BookHash of node unable to expand: " + node.hash().toString());
        throw StringError("Terminating since there's not a good way to put the book back into a good state with this node unupdated");
      }
      symmetries = constNode.getSymmetries();
    }

    Player pla = hist.presumedNextMovePla;
    Board board = hist.getRecentBoard(0);
    search->setPosition(pla,board,hist);
    search->setRootSymmetryPruningOnly(symmetries);

    // Directly set the values for a terminal position
    if(hist.isGameFinished) {
      setNodeThisValuesTerminal(node,hist);
      return;
    }

    std::vector<int> avoidMoveUntilByLoc;
    bool foundNewMoves;
    {
      const bool allowReExpansion = false;
      bool isReExpansion;
      std::lock_guard<std::mutex> lock(bookMutex);
      foundNewMoves = findNewMovesAlreadyLocked(hist,constNode,allowReExpansion,avoidMoveUntilByLoc,isReExpansion);
    }

    if(!foundNewMoves) {
      setNodeThisValuesNoMoves(node);
    }
    else {
      {
        SearchParams thisParams = params;
        thisParams.maxVisits = std::min(params.maxVisits, maxVisitsForLeaves);
        setParamsAndAvoidMoves(search,thisParams,avoidMoveUntilByLoc);
        // cout << "Search and update" << timer.getSeconds() << endl;
        search->runWholeSearch(search->rootPla);
        // cout << "Search and update done" << timer.getSeconds() << endl;
      }

      if(logSearchInfo) {
        std::lock_guard<std::mutex> lock(bookMutex);
        logger.write("Quick search on remaining moves");
        ostringstream out;
        search->printTree(out, search->rootNode, options, perspective);
        logger.write(out.str());
      }

      // Stick all the new values into the book node
      setNodeThisValuesFromFinishedSearch(node, search, search->getRootNode(), search->getRootBoard(), search->getRootHist(), avoidMoveUntilByLoc);
    }
  };

  auto addVariationToBookWithoutUpdate = [&](int gameThreadIdx, const BoardHistory& targetHist, std::set<BookHash>& nodesHashesToUpdate) {
    std::unique_lock<std::mutex> lock(bookMutex);

    Search* search = searches[gameThreadIdx];
    SymBookNode node = book->getRoot();
    BoardHistory hist = book->getInitialHist();
    Player pla = hist.presumedNextMovePla;
    Board board = hist.getRecentBoard(0);
    search->setPosition(pla,board,hist);

    // Run some basic error checking
    if(
      targetHist.initialBoard.pos_hash != board.pos_hash ||
      targetHist.initialBoard.ko_loc != board.ko_loc ||
      targetHist.initialPla != pla ||
      targetHist.initialEncorePhase != hist.initialEncorePhase
    ) {
      throw StringError("Target board history to add to book doesn't start from the same position");
    }
    assert(hist.moveHistory.size() == 0);

    for(auto& move: targetHist.moveHistory) {
      // Make sure we don't walk off the edge under this ruleset.
      if(hist.isGameFinished || hist.isPastNormalPhaseEnd) {
        logger.write("Skipping trace variation at this book hash " + node.hash().toString() + " since game over");
        node.canExpand() = false;
        break;
      }

      Loc moveLoc = move.loc;
      Player movePla = move.pla;
      if(movePla != pla)
        throw StringError("Target board history to add player got out of sync");
      if(movePla != node.pla())
        throw StringError("Target board history to add player got out of sync with node");

      // Illegal move, possibly due to rules mismatch between the books. In that case, we just stop where we are.
      if(!hist.isLegal(board,moveLoc,movePla)) {
        logger.write("Skipping trace variation at this book hash " + node.hash().toString() + " since illegal");
        break;
      }

      if(!node.isMoveInBook(moveLoc)) {
        // If this node in this book or under this ruleset is nonexpandable, then although we can
        // follow existing moves, we can't add any moves.
        if(!node.canExpand()) {
          logger.write("Skipping trace variation at this book hash " + node.hash().toString() + " since nonexpandable");
          break;
        }

        // UNLOCK for performing expensive symmetry computations
        lock.unlock();

        // To avoid oddities in positions where the rules mismatch, expand every move with a noticeably higher raw policy
        // Average all 8 symmetries
        const bool includeOwnerMap = false;
        std::shared_ptr<NNOutput> result = PlayUtils::getFullSymmetryNNOutput(board, hist, pla, includeOwnerMap, nnEval);
        const float* policyProbs = result->policyProbs;
        float moveLocPolicy = policyProbs[search->getPos(moveLoc)];
        assert(moveLocPolicy >= 0);
        vector<std::pair<Loc,float>> extraMoveLocsToExpand;
        for(int pos = 0; pos<NNPos::MAX_NN_POLICY_SIZE; pos++) {
          Loc loc = NNPos::posToLoc(pos, board.x_size, board.y_size, result->nnXLen, result->nnYLen);
          if(loc == Board::NULL_LOC || loc == moveLoc)
            continue;
          if(policyProbs[pos] > 0.0 && policyProbs[pos] > 1.5 * moveLocPolicy + 0.05f)
            extraMoveLocsToExpand.push_back(std::make_pair(loc,policyProbs[pos]));
        }
        std::sort(
          extraMoveLocsToExpand.begin(),
          extraMoveLocsToExpand.end(),
          [](std::pair<Loc,float>& p0, std::pair<Loc,float>& p1) {
            return p0.second > p1.second;
          }
        );

        // LOCK for going back to modifying the book and other shared state
        lock.lock();

        // We're adding moves to this node, so it needs update
        nodesHashesToUpdate.insert(node.hash());

        {
          // Possibly another thread added it, so we need to check again.
          if(!node.isMoveInBook(moveLoc)) {
            Board boardCopy = board;
            BoardHistory histCopy = hist;
            bool childIsTransposing;
            SymBookNode child = node.playAndAddMove(boardCopy,histCopy,moveLoc,moveLocPolicy,childIsTransposing);
            if(!child.isNull() && !childIsTransposing)
              nodesHashesToUpdate.insert(child.hash());
          }
        }
        for(std::pair<Loc,float>& extraMoveLocToExpand: extraMoveLocsToExpand) {
          // Possibly we added it via symmetry, or maybe even another thread, so we need to check again.
          if(!node.isMoveInBook(extraMoveLocToExpand.first)) {
            Board boardCopy = board;
            BoardHistory histCopy = hist;
            bool childIsTransposing;
            SymBookNode child = node.playAndAddMove(boardCopy,histCopy,extraMoveLocToExpand.first,extraMoveLocToExpand.second,childIsTransposing);
            if(!child.isNull() && !childIsTransposing)
              nodesHashesToUpdate.insert(child.hash());
          }
        }
      }

      assert(node.isMoveInBook(moveLoc));
      node = node.playMove(board,hist,moveLoc);
      assert(!node.isNull());
      pla = getOpp(pla);
    }
  };

  // Returns true if any child was added directly to this node (doesn't count recursive stuff).
  std::function<bool(
    Search*, const SearchNode*, SymBookNode,
    const Board&, const BoardHistory&, int,
    std::set<BookHash>&, std::set<BookHash>&,
    std::set<const SearchNode*>&
  )> expandFromSearchResultRecursively;
  expandFromSearchResultRecursively = [&](
    Search* search, const SearchNode* searchNode, SymBookNode node,
    const Board& board, const BoardHistory& hist, int maxDepth,
    std::set<BookHash>& nodesHashesToSearch, std::set<BookHash>& nodesHashesToUpdate,
    std::set<const SearchNode*>& searchNodesRecursedOn
  ) {
    // cout << "Entering expandFromSearchResultRecursively " << timer.getSeconds() << endl;

    if(maxDepth <= 0)
      return false;
    // Quit out immediately when handling transpositions in graph search
    if(searchNodesRecursedOn.find(searchNode) != searchNodesRecursedOn.end())
      return false;
    searchNodesRecursedOn.insert(searchNode);

    assert(searchNode != NULL);
    assert(searchNode->nextPla == node.pla());

    vector<Loc> locs;
    vector<double> playSelectionValues;
    const double scaleMaxToAtLeast = 0.0;
    const bool allowDirectPolicyMoves = false;
    bool suc = search->getPlaySelectionValues(*searchNode, locs, playSelectionValues, NULL, scaleMaxToAtLeast, allowDirectPolicyMoves);
    // Possible if this was a terminal node
    if(!suc)
      return false;

    // Find best move
    double bestValue = playSelectionValues[0];
    int bestIdx = 0;
    for(int i = 1; i<playSelectionValues.size(); i++) {
      if(playSelectionValues[i] > bestValue) {
        bestValue = playSelectionValues[i];
        bestIdx = i;
      }
    }
    Loc bestLoc = locs[bestIdx];

    ConstSearchNodeChildrenReference children = searchNode->getChildren();
    int numChildren = children.iterateAndCountChildren();

    const NNOutput* nnOutput = searchNode->getNNOutput();
    if(numChildren <= 0 || nnOutput == nullptr)
      return false;

    // Use full symmetry for the policy for nodes we record for the book
    bool includeOwnerMap = false;
    std::shared_ptr<NNOutput> fullSymNNOutput = PlayUtils::getFullSymmetryNNOutput(board, hist, node.pla(), includeOwnerMap, search->nnEvaluator);
    const float* policyProbs = fullSymNNOutput->policyProbs;

    bool anyRecursion = false;
    bool anythingAdded = false;
    // cout << "expandFromSearchResultRecursively begin loop over children " << timer.getSeconds() << endl;
    for(int i = 0; i<numChildren; i++) {
      const SearchNode* childSearchNode = children[i].getIfAllocated();
      Loc moveLoc = children[i].getMoveLoc();
      double rawPolicy = policyProbs[search->getPos(moveLoc)];
      int64_t childSearchVisits = childSearchNode->stats.visits.load(std::memory_order_acquire);

      // Add any child nodes that have enough visits or are the best move, if present.
      if(moveLoc == bestLoc || childSearchVisits >= maxVisitsForLeaves) {
        SymBookNode child;
        Board nextBoard = board;
        BoardHistory nextHist = hist;

        {
          std::unique_lock<std::mutex> lock(bookMutex);

          if(node.isMoveInBook(moveLoc)) {
            child = node.follow(moveLoc);
            if(!nextHist.isLegal(nextBoard,moveLoc,node.pla())) {
              logger.write("WARNING: Illegal move " + Location::toString(moveLoc, nextBoard));
              ostringstream debugOut;
              nextHist.printDebugInfo(debugOut,nextBoard);
              logger.write(debugOut.str());
              logger.write("BookHash of parent: " + node.hash().toString());
              logger.write("Marking node as done so we don't try to expand it again, but something is probably wrong.");
              node.canExpand() = false;
            }
            nextHist.makeBoardMoveAssumeLegal(nextBoard,moveLoc,node.pla(),nullptr);
            // Overwrite the child if has no moves yet and we searched it deeper
            if(child.numUniqueMovesInBook() == 0 && child.recursiveValues().visits < childSearchVisits) {
              // No longer need lock here, setNodeThisValuesFromFinishedSearch will lock on its own.
              lock.unlock();
              // Carefully use an empty vector for the avoidMoveUntilByLoc, since the child didn't avoid any moves.
              std::vector<int> childAvoidMoveUntilByLoc;
              setNodeThisValuesFromFinishedSearch(child, search, childSearchNode, nextBoard, nextHist, childAvoidMoveUntilByLoc);
            }

            // Top off the child with a new search if as a leaf it doesn't have enough and our search also doesn't have enough.
            if(child.numUniqueMovesInBook() == 0 && child.recursiveValues().visits < maxVisitsForLeaves)
              nodesHashesToSearch.insert(child.hash());
          }
          else {
            // Lock book to add the best child to the book
            bool childIsTransposing;
            {
              assert(!node.isMoveInBook(moveLoc));
              child = node.playAndAddMove(nextBoard, nextHist, moveLoc, rawPolicy, childIsTransposing);
              // Somehow child was illegal?
              if(child.isNull()) {
                logger.write("WARNING: Illegal move " + Location::toString(moveLoc, nextBoard));
                ostringstream debugOut;
                nextHist.printDebugInfo(debugOut,nextBoard);
                logger.write(debugOut.str());
                logger.write("BookHash of parent: " + node.hash().toString());
                logger.write("Marking node as done so we don't try to expand it again, but something is probably wrong.");
                node.canExpand() = false;
              }
              nodesHashesToUpdate.insert(child.hash());
              string moveHistoryStr;
              for(size_t j = search->rootHistory.moveHistory.size(); j<nextHist.moveHistory.size(); j++) {
                moveHistoryStr += Location::toString(nextHist.moveHistory[j].loc,board);
                moveHistoryStr += " ";
              }
              logger.write("Adding " + node.hash().toString() + " -> " + child.hash().toString() + " moves " + moveHistoryStr);
              // cout << "Adding " << timer.getSeconds() << endl;
              anythingAdded = true;
            }

            // Stick all the new values into the child node, UNLESS the child already had its own search (i.e. we're just transposing)
            // Unless the child is a leaf and we have more visits than it.
            if(!childIsTransposing || (child.numUniqueMovesInBook() == 0 && child.recursiveValues().visits < childSearchVisits)) {
              // No longer need lock here, setNodeThisValuesFromFinishedSearch will lock on its own.
              lock.unlock();
              // Carefully use an empty vector for the avoidMoveUntilByLoc, since the child didn't avoid any moves.
              std::vector<int> childAvoidMoveUntilByLoc;
              // cout << "Calling setNodeThisValuesFromFinishedSearch " << timer.getSeconds() << endl;
              setNodeThisValuesFromFinishedSearch(child, search, childSearchNode, nextBoard, nextHist, childAvoidMoveUntilByLoc);
              // cout << "Returned from setNodeThisValuesFromFinishedSearch " << timer.getSeconds() << endl;
            }

            // Top off the child with a new search if as a leaf it doesn't have enough and our search also doesn't have enough.
            if(child.numUniqueMovesInBook() == 0 && child.recursiveValues().visits < maxVisitsForLeaves)
              nodesHashesToSearch.insert(child.hash());
          }
        } // Release lock

        // Recursively record children with enough visits
        if(maxDepth > 0 && childSearchVisits >= minTreeVisitsToRecord && !nextHist.isGameFinished) {
          anyRecursion = true;
          // cout << "Calling expandFromSearchResultRecursively " << maxDepth << " " << childSearchVisits << " " << timer.getSeconds() << endl;
          expandFromSearchResultRecursively(
            search, childSearchNode, child, nextBoard, nextHist, maxDepth-1,
            nodesHashesToSearch, nodesHashesToUpdate, searchNodesRecursedOn
          );
          // cout << "Returned from expandFromSearchResultRecursively " << maxDepth << " " << childSearchVisits << " " << timer.getSeconds() << endl;
        }
      }
    }

    // This node's values need to be recomputed at the end if it changed or anything under it changed.
    if(anythingAdded || anyRecursion)
      nodesHashesToUpdate.insert(node.hash());

    // This node needs to be searched with its new avoid moves if any move was added to update its thisnodevalues.
    if(anythingAdded)
      nodesHashesToSearch.insert(node.hash());

    return anythingAdded;
  };

  auto expandNode = [&](int gameThreadIdx, SymBookNode node, std::vector<SymBookNode>& newAndChangedNodes) {
    ConstSymBookNode constNode(node);

    BoardHistory hist;
    std::vector<Loc> moveHistory;
    std::vector<int> symmetries;
    std::vector<double> winlossHistory;
    bool suc;
    {
      std::lock_guard<std::mutex> lock(bookMutex);
      suc = constNode.getBoardHistoryReachingHere(hist,moveHistory,winlossHistory);
      symmetries = constNode.getSymmetries();
    }

    if(!suc) {
      std::lock_guard<std::mutex> lock(bookMutex);
      logger.write("WARNING: Failed to get board history reaching node when trying to expand book, probably there is some bug");
      logger.write("or else some hash collision or something else is wrong.");
      logger.write("BookHash of node unable to expand: " + constNode.hash().toString());
      ostringstream movesOut;
      for(Loc move: moveHistory)
        movesOut << Location::toString(move,book->initialBoard) << " ";
      logger.write("Moves:");
      logger.write(movesOut.str());
      logger.write("Marking node as done so we don't try to expand it again, but something is probably wrong.");
      node.canExpand() = false;
      return;
    }

    // Book integrity check, only for later versions since older versions had a bug that gets them permanently with
    // hashes stuck to be bad.
    if(book->bookVersion >= 2) {
      BookHash hashRet;
      int symmetryToAlignRet;
      vector<int> symmetriesRet;
      BookHash::getHashAndSymmetry(hist, book->repBound, hashRet, symmetryToAlignRet, symmetriesRet, book->bookVersion);
      if(hashRet != node.hash()) {
        ostringstream out;
        Board board = hist.getRecentBoard(0);
        Board::printBoard(out, board, Board::NULL_LOC, NULL);
        for(Loc move: moveHistory)
          out << Location::toString(move,book->initialBoard) << " ";
        logger.write("Moves:");
        logger.write(out.str());
        throw StringError("Book failed integrity check, the node with hash " + node.hash().toString() + " when walked to has hash " + hashRet.toString());
      }
    }

    // Terminal node!
    // We ALLOW walking past the main phase of the game under this ruleset, to give the book the ability to
    // solve tactics in the cleanup phase of japanese rules if needed. So we only check isGameFinished instead of isPastNormalPhaseEnd.
    if(hist.isGameFinished) {
      std::lock_guard<std::mutex> lock(bookMutex);
      node.canExpand() = false;
      return;
    }

    Search* search = searches[gameThreadIdx];
    Player pla = hist.presumedNextMovePla;
    Board board = hist.getRecentBoard(0);
    search->setPosition(pla,board,hist);
    search->setRootSymmetryPruningOnly(symmetries);

    {
      ostringstream out;
      for(Move m: hist.moveHistory)
        out << Location::toString(m.loc,board) << " ";
      out << endl;
      for(double winLoss: winlossHistory)
        out << Global::strprintf("%2.0f", 100.0*(0.5 * (winLoss + 1.0))) << " ";
      out << endl;
      Board::printBoard(out, board, Board::NULL_LOC, &(hist.moveHistory));
      std::lock_guard<std::mutex> lock(bookMutex);
      logger.write("Expanding " + node.hash().toString() + " cost " + Global::doubleToString(node.totalExpansionCost()));
      logger.write(out.str());
    }

    std::vector<int> avoidMoveUntilByLoc;
    bool foundNewMoves;
    bool isReExpansion;
    {
      const bool allowReExpansion = true;
      std::lock_guard<std::mutex> lock(bookMutex);
      foundNewMoves = findNewMovesAlreadyLocked(hist,constNode,allowReExpansion,avoidMoveUntilByLoc,isReExpansion);
    }
    if(!foundNewMoves) {
      std::lock_guard<std::mutex> lock(bookMutex);
      node.canExpand() = false;
      return;
    }

    SearchParams thisParams = params;
    thisParams.wideRootNoise = wideRootNoiseBookExplore;
    thisParams.cpuctExplorationLog = cpuctExplorationLogBookExplore;
    setParamsAndAvoidMoves(search,thisParams,avoidMoveUntilByLoc);
    search->runWholeSearch(search->rootPla);


    if(shouldStop.load(std::memory_order_acquire))
      return;

    if(logSearchInfo) {
      std::lock_guard<std::mutex> lock(bookMutex);
      ostringstream out;
      search->printTree(out, search->rootNode, options, perspective);
      logger.write("Search result");
      logger.write(out.str());
    }

    // cout << "Beginning recurison " << timer.getSeconds() << endl;

    std::set<BookHash> nodesHashesToSearch;
    std::set<BookHash> nodesHashesToUpdate;
    std::set<const SearchNode*> searchNodesRecursedOn;
    bool anythingAdded = expandFromSearchResultRecursively(
      search, search->rootNode, node, board, hist, maxDepthToRecord,
      nodesHashesToSearch, nodesHashesToUpdate, searchNodesRecursedOn
    );

    // cout << "Ending recursion " << timer.getSeconds() << endl;

    // And immediately do a search to update each node we need to.
    {
      std::vector<SymBookNode> nodesToSearch;
      // Try to make all of the nodes be consistent in symmetry so that they can share cache.
      // Append the original position itself to the start so that it anchors the symmetries
      nodesToSearch.push_back(node);
      {
        std::lock_guard<std::mutex> lock(bookMutex);
        for(const BookHash& hash: nodesHashesToSearch) {
          SymBookNode nodeToSearch;
          nodeToSearch = book->getByHash(hash);
          nodesToSearch.push_back(nodeToSearch);
        }
      }
      optimizeSymmetriesInplace(nodesToSearch, NULL, logger);

      // Pop off the original position itself
      nodesToSearch.erase(nodesToSearch.begin());

      // cout << "Doing searches to update " << timer.getSeconds() << endl;
      for(SymBookNode nodeToSearch: nodesToSearch) {
        searchAndUpdateNodeThisValues(search,nodeToSearch);
      }
      // cout << "Done searches to update " << timer.getSeconds() << endl;
    }

    {
      std::lock_guard<std::mutex> lock(bookMutex);
      for(const BookHash& hash: nodesHashesToUpdate) {
        SymBookNode nodeToUpdate;
        nodeToUpdate = book->getByHash(hash);
        newAndChangedNodes.push_back(nodeToUpdate);
      }
    }

    // Only nodes that have never been expanded on their own (were added from another node's search) are allowed for reexpansion.
    node.canReExpand() = false;
    newAndChangedNodes.push_back(node);

    // Make sure to process the nodes to search and updates so the book is in a consistent state, before we do any quitting out.
    // On non-reexpansions, we expect to always add at least one new move to the book for this node.
    if(!anythingAdded && !isReExpansion) {
      std::lock_guard<std::mutex> lock(bookMutex);
      logger.write("WARNING: Could not expand since search obtained no new moves, despite earlier checks about legal moves existing not yet in book");
      logger.write("BookHash of node unable to expand: " + constNode.hash().toString());
      ostringstream debugOut;
      hist.printDebugInfo(debugOut,board);
      logger.write(debugOut.str());
      logger.write("Possibly this was simply due to a multi-step expansion of another search getting there first, so logging this but proceeding as normal");
      // logger.write("Marking node as done so we don't try to expand it again, but something is probably wrong.");
      // node.canExpand() = false;
    }

  };

  if(traceBook != NULL || traceSgfFile.size() > 0) {
    std::set<BookHash> nodesHashesToUpdate;

    if(traceBook != NULL) {
      ThreadSafeQueue<SymBookNode> positionsToTrace;
      std::vector<SymBookNode> allNodes = traceBook->getAllLeaves(traceBookMinVisits);
      std::atomic<int64_t> variationsAdded(0);
      auto loopAddingVariations = [&](int gameThreadIdx) {
        while(true) {
          if(shouldStop.load(std::memory_order_acquire))
            return;
          SymBookNode node;
          bool suc = positionsToTrace.tryPop(node);
          if(!suc)
            return;
          BoardHistory hist;
          std::vector<Loc> moveHistory;
          suc = node.getBoardHistoryReachingHere(hist, moveHistory);
          assert(suc);
          (void)suc;
          addVariationToBookWithoutUpdate(gameThreadIdx, hist, nodesHashesToUpdate);
          int64_t currentVariationsAdded = variationsAdded.fetch_add(1) + 1;
          if(currentVariationsAdded % 400 == 0) {
            logger.write(
              "Tracing book, currentVariationsAdded " +
              Global::int64ToString(currentVariationsAdded) + "/" + Global::uint64ToString(allNodes.size())
            );
          }
        }
      };

      for(SymBookNode node: allNodes)
        positionsToTrace.forcePush(node);
      vector<std::thread> threads;
      for(int gameThreadIdx = 0; gameThreadIdx<numGameThreads; gameThreadIdx++) {
        threads.push_back(std::thread(loopAddingVariations, gameThreadIdx));
      }
      for(int gameThreadIdx = 0; gameThreadIdx<numGameThreads; gameThreadIdx++) {
        threads[gameThreadIdx].join();
      }
      int64_t currentVariationsAdded = variationsAdded.load();
      logger.write(
        "Tracing book, currentVariationsAdded " +
        Global::int64ToString(currentVariationsAdded) + "/" + Global::uint64ToString(allNodes.size())
      );
    }
    else {
      assert(traceSgfFile.size() > 0);
      Sgf* sgf = Sgf::loadFile(traceSgfFile);
      bool flipIfPassOrWFirst = false;
      bool allowGameOver = false;
      Rand seedRand("bonusByHash");
      int64_t variationsAdded = 0;
      sgf->iterAllPositions(
        flipIfPassOrWFirst, allowGameOver, &seedRand, [&](Sgf::PositionSample& unusedSample, const BoardHistory& sgfHist, const string& comments) {
          (void)unusedSample;
          (void)comments;
          int gameThreadIdx = 0;
          addVariationToBookWithoutUpdate(gameThreadIdx, sgfHist, nodesHashesToUpdate);
          variationsAdded += 1;
          if(variationsAdded % 400 == 0) {
            logger.write(
              "Tracing sgf, variationsAdded " +
              Global::int64ToString(variationsAdded)
            );
          }
        }
      );
      logger.write(
        "Tracing sgf, variationsAdded " +
        Global::int64ToString(variationsAdded)
      );
      delete sgf;
    }

    {
      ThreadSafeQueue<BookHash> hashesToUpdate;
      std::atomic<int64_t> hashesUpdated(0);
      auto loopUpdatingHashes = [&](int gameThreadIdx) {
        while(true) {
          if(shouldStop.load(std::memory_order_acquire))
            return;
          BookHash hash;
          bool suc = hashesToUpdate.tryPop(hash);
          if(!suc)
            return;
          SymBookNode node;
          {
            std::lock_guard<std::mutex> lock(bookMutex);
            node = book->getByHash(hash);
            assert(!node.isNull());
          }
          Search* search = searches[gameThreadIdx];
          searchAndUpdateNodeThisValues(search, node);
          int64_t currentHashesUpdated = hashesUpdated.fetch_add(1) + 1;
          if(currentHashesUpdated % 100 == 0) {
            logger.write(
              "Updating book, currentHashesUpdated " +
              Global::int64ToString(currentHashesUpdated) + "/" + Global::uint64ToString(nodesHashesToUpdate.size())
            );
          }
        }
      };

      for(BookHash hash: nodesHashesToUpdate)
        hashesToUpdate.forcePush(hash);
      vector<std::thread> threads;
      for(int gameThreadIdx = 0; gameThreadIdx<numGameThreads; gameThreadIdx++) {
        threads.push_back(std::thread(loopUpdatingHashes, gameThreadIdx));
      }
      for(int gameThreadIdx = 0; gameThreadIdx<numGameThreads; gameThreadIdx++) {
        threads[gameThreadIdx].join();
      }
      int64_t currentHashesUpdated = hashesUpdated.load();
      logger.write(
        "Tracing book, currentHashesUpdated " +
        Global::int64ToString(currentHashesUpdated) + "/" + Global::uint64ToString(nodesHashesToUpdate.size())
      );
    }

    if(shouldStop.load(std::memory_order_acquire)) {
      logger.write("Trace book incomplete, exiting without saving");
      throw StringError("Trace book incomplete, exiting without saving");
    }

    logger.write("Recomputing recursive values for entire book");
    book->recomputeEverything();
  }
  else {
    ThreadSafeQueue<SymBookNode> positionsToSearch;

    for(int iteration = 0; iteration < numIterations; iteration++) {
      if(shouldStop.load(std::memory_order_acquire))
        break;

      if(iteration % saveEveryIterations == 0 && iteration != 0) {
        logger.write("SAVING TO FILE " + bookFile);
        book->setParams(cfgParams);
        book->saveToFile(bookFile);
        ofstream out;
        FileUtils::open(out, bookFile + ".cfg");
        out << cfg.getContents() << endl;
        out.close();
      }

      logger.write("BEGINNING BOOK EXPANSION ITERATION " + Global::intToString(iteration));

      if(randomizeParamsStdev > 0.0) {
        BookParams paramsCopy = cfgParams;
        paramsCopy.randomizeParams(rand, randomizeParamsStdev);
        book->setParams(paramsCopy);
        book->recomputeEverything();
        logger.write("Randomized params and recomputed costs");
      }

      std::vector<SymBookNode> nodesToExpand = book->getNextNToExpand(std::min(1+iteration/2,numToExpandPerIteration));
      // Try to make all of the expanded nodes be consistent in symmetry so that they can share cache, in case
      // many of them are for related board positions.
      optimizeSymmetriesInplace(nodesToExpand, &rand, logger);

      for(SymBookNode node: nodesToExpand) {
        bool suc = positionsToSearch.forcePush(node);
        assert(suc);
        (void)suc;
      }

      std::vector<SymBookNode> newAndChangedNodes = nodesToExpand;

      auto loopExpandingNodes = [&](int gameThreadIdx) {
        while(true) {
          if(shouldStop.load(std::memory_order_acquire))
            return;
          SymBookNode node;
          bool suc = positionsToSearch.tryPop(node);
          if(!suc)
            return;
          expandNode(gameThreadIdx, node, newAndChangedNodes);
        }
      };

      vector<std::thread> threads;
      for(int gameThreadIdx = 0; gameThreadIdx<numGameThreads; gameThreadIdx++) {
        threads.push_back(std::thread(loopExpandingNodes, gameThreadIdx));
      }
      for(int gameThreadIdx = 0; gameThreadIdx<numGameThreads; gameThreadIdx++) {
        threads[gameThreadIdx].join();
      }

      book->recompute(newAndChangedNodes);
      if(shouldStop.load(std::memory_order_acquire))
        break;
    }
  }

  if(traceBook != NULL || traceSgfFile.size() > 0 || numIterations > 0) {
    logger.write("SAVING TO FILE " + bookFile);
    book->setParams(cfgParams);
    book->saveToFile(bookFile);
    ofstream out;
    FileUtils::open(out, bookFile + ".cfg");
    out << cfg.getContents() << endl;
    out.close();
  }

  if(htmlDir != "") {
    logger.write("EXPORTING HTML TO " + htmlDir);
    int64_t numFilesWritten = book->exportToHtmlDir(htmlDir,rulesLabel,rulesLink,htmlDevMode,htmlMinVisits,logger);
    logger.write("Done exporting, exported " + Global::int64ToString(numFilesWritten) + " files");
  }

  for(int i = 0; i<numGameThreads; i++)
    delete searches[i];
  delete nnEval;
  delete book;
  delete traceBook;
  ScoreValue::freeTables();
  logger.write("DONE");
  return 0;
}

int MainCmds::writebook(const vector<string>& args) {
  Board::initHash();
  ScoreValue::initTables();

  ConfigParser cfg;
  string htmlDir;
  string bookFile;
  string bonusFile;
  bool htmlDevMode;
  double htmlMinVisits;
  try {
    KataGoCommandLine cmd("Generate opening book");
    cmd.addConfigFileArg("","",false);
    cmd.addOverrideConfigArg();

    TCLAP::ValueArg<string> htmlDirArg("","html-dir","HTML directory to export to, at the end of -num-iters",true,string(),"DIR");
    TCLAP::ValueArg<string> bookFileArg("","book-file","Book file to write to or continue expanding",true,string(),"FILE");
    TCLAP::ValueArg<string> bonusFileArg("","bonus-file","SGF of bonuses marked",false,string(),"DIR");
    TCLAP::SwitchArg htmlDevModeArg("","html-dev-mode","Denser debug output for html");
    TCLAP::ValueArg<double> htmlMinVisitsArg("","html-min-visits","Require >= this many visits to export a position to html",false,0.0,"N");
    cmd.add(htmlDirArg);
    cmd.add(bookFileArg);
    cmd.add(bonusFileArg);
    cmd.add(htmlDevModeArg);
    cmd.add(htmlMinVisitsArg);

    cmd.parseArgs(args);

    cmd.getConfigAllowEmpty(cfg);
    htmlDir = htmlDirArg.getValue();
    bookFile = bookFileArg.getValue();
    bonusFile = bonusFileArg.getValue();
    htmlDevMode = htmlDevModeArg.getValue();
    htmlMinVisits = htmlMinVisitsArg.getValue();
  }
  catch (TCLAP::ArgException &e) {
    cerr << "Error: " << e.error() << " for argument " << e.argId() << endl;
    return 1;
  }


  Rand rand;
  const bool logToStdoutDefault = true;
  Logger logger(&cfg, logToStdoutDefault);

  const string rulesLabel = cfg.getString("rulesLabel");
  const string rulesLink = cfg.getString("rulesLink");
  const double bonusFileScale = cfg.contains("bonusFileScale") ? cfg.getDouble("bonusFileScale",0.0,1000000.0) : 1.0;
  const SearchParams params = Setup::loadSingleParams(cfg,Setup::SETUP_FOR_GTP);
  BookParams cfgParams = BookParams::loadFromCfg(cfg, params.maxVisits);

  const bool loadKomiFromCfg = true;
  Rules rules = Setup::loadSingleRules(cfg,loadKomiFromCfg);
  const int boardSizeX = cfg.getInt("boardSizeX",2,Board::MAX_LEN);
  const int boardSizeY = cfg.getInt("boardSizeY",2,Board::MAX_LEN);
  const int repBound = cfg.getInt("repBound",3,1000);

  std::map<BookHash,double> bonusByHash;
  std::map<BookHash,double> expandBonusByHash;
  std::map<BookHash,double> visitsRequiredByHash;
  std::map<BookHash,int> branchRequiredByHash;
  Board bonusInitialBoard;
  Player bonusInitialPla;

  maybeParseBonusFile(
    bonusFile,
    boardSizeX,
    boardSizeY,
    rules,
    repBound,
    bonusFileScale,
    logger,
    bonusByHash,
    expandBonusByHash,
    visitsRequiredByHash,
    branchRequiredByHash,
    bonusInitialBoard,
    bonusInitialPla
  );

  // Check for unused config keys
  cfg.warnUnusedKeys(cerr,&logger);

  MakeDir::make(htmlDir);

  Book* book = Book::loadFromFile(bookFile);
  book->setParams(cfgParams);
  book->setBonusByHash(bonusByHash);
  book->setExpandBonusByHash(expandBonusByHash);
  book->setVisitsRequiredByHash(visitsRequiredByHash);
  book->setBranchRequiredByHash(branchRequiredByHash);
  book->recomputeEverything();

  logger.write("EXPORTING HTML TO " + htmlDir);
  int64_t numFilesWritten = book->exportToHtmlDir(htmlDir,rulesLabel,rulesLink,htmlDevMode,htmlMinVisits,logger);
  logger.write("Done exporting, exported " + Global::int64ToString(numFilesWritten) + " files");

  delete book;
  ScoreValue::freeTables();
  logger.write("DONE");
  return 0;
}

int MainCmds::checkbook(const vector<string>& args) {
  Board::initHash();
  ScoreValue::initTables();

  string bookFile;
  try {
    KataGoCommandLine cmd("Check integrity of opening book");

    TCLAP::ValueArg<string> bookFileArg("","book-file","Book file to write to or continue expanding",true,string(),"FILE");
    cmd.add(bookFileArg);

    cmd.parseArgs(args);

    bookFile = bookFileArg.getValue();
  }
  catch (TCLAP::ArgException &e) {
    cerr << "Error: " << e.error() << " for argument " << e.argId() << endl;
    return 1;
  }

  Rand rand;
  const bool logToStdout = true;
  const bool logToStderr = false;
  const bool logTime = false;
  Logger logger(nullptr, logToStdout, logToStderr, logTime);

  Book* book;
  {
    book = Book::loadFromFile(bookFile);
    logger.write("Loaded preexisting book with " + Global::uint64ToString(book->size()) + " nodes from " + bookFile);
    logger.write("Book version = " + Global::intToString(book->bookVersion));
  }

  const PrintTreeOptions options;

  auto testNode = [&](SymBookNode node) {
    ConstSymBookNode constNode(node);

    BoardHistory hist;
    std::vector<Loc> moveHistory;
    std::vector<int> symmetries;
    bool suc;
    {
      suc = constNode.getBoardHistoryReachingHere(hist,moveHistory);
      symmetries = constNode.getSymmetries();
    }

    if(!suc) {
      logger.write("WARNING: Failed to get board history reaching node, probably there is some bug");
      logger.write("or else some hash collision or something else is wrong.");
      logger.write("BookHash of node unable to expand: " + constNode.hash().toString());
      ostringstream out;
      Board board = hist.getRecentBoard(0);
      Board::printBoard(out, board, Board::NULL_LOC, NULL);
      for(Loc move: moveHistory)
        out << Location::toString(move,book->initialBoard) << " ";
      logger.write("Moves:");
      logger.write(out.str());
    }

    // Book integrity check
    {
      BookHash hashRet;
      int symmetryToAlignRet;
      vector<int> symmetriesRet;
      BookHash::getHashAndSymmetry(hist, book->repBound, hashRet, symmetryToAlignRet, symmetriesRet, book->bookVersion);
      if(hashRet != node.hash()) {
        logger.write("Book failed integrity check, the node with hash " + node.hash().toString() + " when walked to has hash " + hashRet.toString());
        ostringstream out;
        Board board = hist.getRecentBoard(0);
        Board::printBoard(out, board, Board::NULL_LOC, NULL);
        for(Loc move: moveHistory)
          out << Location::toString(move,book->initialBoard) << " ";
        logger.write("Moves:");
        logger.write(out.str());
      }
    }
  };

  std::vector<SymBookNode> allNodes = book->getAllNodes();
  logger.write("Checking book...");
  int64_t numNodesChecked = 0;
  for(SymBookNode node: allNodes) {
    testNode(node);
    numNodesChecked += 1;
    if(numNodesChecked % 10000 == 0)
      logger.write("Checked " + Global::int64ToString(numNodesChecked) + "/" + Global::int64ToString((int64_t)allNodes.size()) + " nodes");
  }

  delete book;
  ScoreValue::freeTables();
  logger.write("DONE");
  return 0;
}

int MainCmds::booktoposes(const vector<string>& args) {
  Board::initHash();
  ScoreValue::initTables();
  Rand seedRand;

  ConfigParser cfg;
  string modelFile;

  string outDir;
  string bookFile;
  int numThreads;
  int includeDepth;
  double includeVisits;
  int maxDepth;
  double minVisits;
  bool enableHints;
  double constantWeight;
  double depthWeight;
  double depthWeightScale;
  double policySurpriseWeight;
  double valueSurpriseWeight;
  double minWeight;
  try {
    KataGoCommandLine cmd("Dump startposes out of book");
    cmd.addConfigFileArg("","");
    cmd.addModelFileArg();

    cmd.addOverrideConfigArg();

    TCLAP::ValueArg<string> outDirArg("","out-dir","Directory to write poses",true,string(),"DIR");
    TCLAP::ValueArg<string> bookFileArg("","book-file","Book file to write to or continue expanding",true,string(),"FILE");
    TCLAP::ValueArg<int> numThreadsArg("","num-threads","Number of threads to use for processing",false,1,"N");
    TCLAP::ValueArg<int> includeDepthArg("","include-depth","Include positions up to this depth",false,-1,"DEPTH");
    TCLAP::ValueArg<double> includeVisitsArg("","include-visits","Include positions this many visits or more",false,1e300,"VISITS");
    TCLAP::ValueArg<int> maxDepthArg("","max-depth","Only include positions up to this depth",false,100000000,"DEPTH");
    TCLAP::ValueArg<double> minVisitsArg("","max-visits","Only include positions with this many visits or more",false,-1.0,"VISITS");
    TCLAP::SwitchArg enableHintsArg("","enable-hints","Hint the top book move");
    TCLAP::ValueArg<double> constantWeightArg("","constant-weight","How much weight to give each position as a fixed baseline",false,0.0,"FLOAT");
    TCLAP::ValueArg<double> depthWeightArg("","depth-weight","How much extra weight to give based on depth",false,0.0,"FLOAT");
    TCLAP::ValueArg<double> depthWeightScaleArg("","depth-weight-scale","Depth scale over which depth weight decays by a factor of e",false,1.0,"FLOAT");
    TCLAP::ValueArg<double> policySurpriseWeightArg("","policy-surprise-weight","How much weight to give each position per logit of policy surprise",false,0.0,"FLOAT");
    TCLAP::ValueArg<double> valueSurpriseWeightArg("","value-surprise-weight","How much weight to give each position per logit of value surprise",false,0.0,"FLOAT");
    TCLAP::ValueArg<double> minWeightArg("","min-weight","Only finally include positions with this much weight",false,0.0,"FLOAT");

    cmd.add(outDirArg);
    cmd.add(bookFileArg);
    cmd.add(numThreadsArg);
    cmd.add(includeDepthArg);
    cmd.add(includeVisitsArg);
    cmd.add(maxDepthArg);
    cmd.add(minVisitsArg);
    cmd.add(enableHintsArg);
    cmd.add(constantWeightArg);
    cmd.add(depthWeightArg);
    cmd.add(depthWeightScaleArg);
    cmd.add(policySurpriseWeightArg);
    cmd.add(valueSurpriseWeightArg);
    cmd.add(minWeightArg);

    cmd.parseArgs(args);

    modelFile = cmd.getModelFile();
    outDir = outDirArg.getValue();
    bookFile = bookFileArg.getValue();
    numThreads = numThreadsArg.getValue();
    includeDepth = includeDepthArg.getValue();
    includeVisits = includeVisitsArg.getValue();
    maxDepth = maxDepthArg.getValue();
    minVisits = minVisitsArg.getValue();
    enableHints = enableHintsArg.getValue();
    constantWeight = constantWeightArg.getValue();
    depthWeight = depthWeightArg.getValue();
    depthWeightScale = depthWeightScaleArg.getValue();
    policySurpriseWeight = policySurpriseWeightArg.getValue();
    valueSurpriseWeight = valueSurpriseWeightArg.getValue();
    minWeight = minWeightArg.getValue();

    cmd.getConfig(cfg);
  }
  catch (TCLAP::ArgException &e) {
    cerr << "Error: " << e.error() << " for argument " << e.argId() << endl;
    return 1;
  }

  const bool logToStdout = true;
  const bool logToStderr = false;
  const bool logTime = false;
  Logger logger(nullptr, logToStdout, logToStderr, logTime);

  Book* book;
  {
    book = Book::loadFromFile(bookFile);
    logger.write("Loaded preexisting book with " + Global::uint64ToString(book->size()) + " nodes from " + bookFile);
    logger.write("Book version = " + Global::intToString(book->bookVersion));
  }

  NNEvaluator* nnEval;
  {
    Setup::initializeSession(cfg);
    int maxConcurrentEvals = numThreads * 2 + 16; // * 2 + 16 just to give plenty of headroom
    int expectedConcurrentEvals = numThreads;
    int defaultMaxBatchSize = std::max(8,((numThreads+3)/4)*4);
    bool defaultRequireExactNNLen = true;
    bool disableFP16 = false;
    string expectedSha256 = "";
    nnEval = Setup::initializeNNEvaluator(
      modelFile,modelFile,expectedSha256,cfg,logger,seedRand,maxConcurrentEvals,expectedConcurrentEvals,
      book->initialBoard.x_size,book->initialBoard.y_size,defaultMaxBatchSize,defaultRequireExactNNLen,disableFP16,
      Setup::SETUP_FOR_GTP
    );
  }
  logger.write("Loaded neural net");

  cfg.warnUnusedKeys(cerr,&logger);

  std::map<BookHash,int> depthByHash;

  std::vector<ConstSymBookNode> nodesToExplore;
  std::vector<int> depthsToExplore;
  nodesToExplore.push_back(book->getRoot());
  depthsToExplore.push_back(0);

  logger.write("Beginning book sweep");
  int numNodesExplored = 0;
  while(nodesToExplore.size() > 0) {
    ConstSymBookNode node = nodesToExplore[nodesToExplore.size()-1];
    int depth = depthsToExplore[depthsToExplore.size()-1];
    nodesToExplore.pop_back();
    depthsToExplore.pop_back();

    if(depth > maxDepth)
      continue;
    if(node.recursiveValues().visits < minVisits)
      continue;
    if(depth > includeDepth && node.recursiveValues().visits < includeVisits)
      continue;

    BookHash hash = node.hash();
    {
      auto iter = depthByHash.find(hash);
      if(iter != depthByHash.end())
        if(depth >= iter->second)
          continue;
      depthByHash[hash] = depth;
    }

    std::vector<BookMove> moves = node.getUniqueMovesInBook();
    for(int i = (int)moves.size()-1; i >= 0; i--) {
      nodesToExplore.push_back(node.follow(moves[i].move));
      depthsToExplore.push_back(depth+1);
    }
    numNodesExplored += 1;
    if(numNodesExplored % 100000 == 0)
      logger.write("Num nodes explored: " + Global::intToString(numNodesExplored));
  }

  logger.write("Collected " + Global::intToString(depthByHash.size()) + " many positions in book to potentially use");

  PosWriter posWriter("bookposes.txt", outDir, 1, 0, 100000);
  posWriter.start();

  double utilityPerPolicyForSorting = book->getParams().utilityPerPolicyForSorting;

  std::mutex statsLock;
  int numPositionsProcessed = 0;
  int numPositionsWritten = 0;
  double totalWeightFromConstant = 0.0;
  double totalWeightFromDepth = 0.0;
  double totalWeightFromPolicySurprise = 0.0;
  double totalWeightFromValueSurpriseIrreducible = 0.0;
  double totalWeightFromValueSurpriseDivergence = 0.0;

  auto processPoses = [&](int threadIdx) {
    Rand rand;
    int counter = 0;
    for(auto iter = depthByHash.begin(); iter != depthByHash.end(); ++iter) {
      if(counter % numThreads != threadIdx) {
        counter += 1;
        continue;
      }
      counter += 1;

      BookHash hash = iter->first;
      int depth = iter->second;
      ConstSymBookNode node = book->getByHash(hash).applySymmetry(rand.nextInt(0,7));

      Player pla = node.pla();
      BoardHistory hist;
      std::vector<Loc> moveHistory;
      bool suc = node.getBoardHistoryReachingHere(hist,moveHistory);
      if(!suc) {
        logger.write("WARNING: Failed to get board history reaching node, probably there is some bug");
        logger.write("or else some hash collision or something else is wrong.");
        logger.write("BookHash of node unable to expand: " + node.hash().toString());

        ostringstream out;
        Board board = hist.getRecentBoard(0);
        Board::printBoard(out, board, Board::NULL_LOC, NULL);
        for(Loc move: moveHistory)
          out << Location::toString(move,book->initialBoard) << " ";
        logger.write("Moves:");
        logger.write(out.str());
        continue;
      }

      Sgf::PositionSample sample;
      sample.board = hist.getRecentBoard(5);
      for(int i = std::max(0,(int)hist.moveHistory.size()-5); i<hist.moveHistory.size(); i++)
        sample.moves.push_back(hist.moveHistory[i]);
      sample.nextPla = sample.moves.size() > 0 ? sample.moves[0].pla : pla;
      sample.initialTurnNumber = depth;
      sample.hintLoc = Board::NULL_LOC;

      std::vector<double> sortingValue;
      std::vector<BookMove> moves = node.getUniqueMovesInBook();
      for(int i = 0; i<moves.size(); i++) {
        ConstSymBookNode child = node.follow(moves[i].move);
        RecursiveBookValues values = child.recursiveValues();
        double plaFactor = pla == P_WHITE ? 1.0 : -1.0;
        double value = plaFactor * (values.winLossValue + values.sharpScoreMean * book->getParams().utilityPerScore * 0.5)
          + plaFactor * (pla == P_WHITE ? values.scoreLCB : values.scoreUCB) * 0.5 * book->getParams().utilityPerScore
          + utilityPerPolicyForSorting * (0.75 * moves[i].rawPolicy + 0.5 * log10(moves[i].rawPolicy + 0.0001)/4.0);
        sortingValue.push_back(value);
      }

      Loc bestMove = Board::NULL_LOC;
      if(sortingValue.size() > 0) {
        double bestSortingValue = -1e100;
        for(int i = 0; i<sortingValue.size(); i++) {
          if(sortingValue[i] > bestSortingValue) {
            bestSortingValue = sortingValue[i];
            bestMove = moves[i].move;
          }
        }
      }

      if(enableHints)
        sample.hintLoc = bestMove;

      double bookWLValue = node.recursiveValues().winLossValue;
      double bookWinChance = std::max(0.0, std::min(1.0, 0.5 * (bookWLValue + 1.0)));
      double bookLossChance = std::max(0.0, std::min(1.0, 0.5 * (-bookWLValue + 1.0)));

      double policySurprise = 0.0;
      double valueSurpriseIrreducible = 0.0;
      double valueSurpriseTotal = 0.0;
      Board board = hist.getRecentBoard(0);
      for(int sym = 0; sym<SymmetryHelpers::NUM_SYMMETRIES; sym++) {
        MiscNNInputParams nnInputParams;
        nnInputParams.symmetry = sym;
        NNResultBuf buf;
        bool skipCache = true; //Always ignore cache so that we use the desired symmetry
        bool includeOwnerMap = false;
        if(policySurpriseWeight > 0 || valueSurpriseWeight > 0)
          nnEval->evaluate(board,hist,pla,nnInputParams,buf,skipCache,includeOwnerMap);

        if(policySurpriseWeight > 0) {
          if(bestMove != Board::NULL_LOC) {
            double policyProb = buf.result->policyProbs[NNPos::locToPos(bestMove,board.x_size,nnEval->getNNXLen(),nnEval->getNNYLen())];
            assert(policyProb >= 0.0 && policyProb <= 1.0);
            policySurprise += -1.0 / (double)SymmetryHelpers::NUM_SYMMETRIES * log(policyProb + 1e-30);
          }
        }

        if(valueSurpriseWeight > 0) {
          double wlValue = (double)buf.result->whiteWinProb - (double)buf.result->whiteLossProb;
          double winChance = std::max(0.0, std::min(1.0, 0.5 * (wlValue + 1.0)));
          double lossChance = std::max(0.0, std::min(1.0, 0.5 * (-wlValue + 1.0)));

          valueSurpriseIrreducible += -1.0 / (double)SymmetryHelpers::NUM_SYMMETRIES * (
            bookWinChance * log(bookWinChance + 1e-30) + bookLossChance * log(bookLossChance + 1e-30)
          );
          valueSurpriseTotal += -1.0 / (double)SymmetryHelpers::NUM_SYMMETRIES * (
            bookWinChance * log(winChance + 1e-30) + bookLossChance * log(lossChance + 1e-30)
          );
        }
      }

      double weightFromConstant = constantWeight;
      double weightFromDepth = exp(-(double)depth / depthWeightScale) * depthWeight;
      double weightFromPolicySurprise = policySurprise * policySurpriseWeight;
      double weightFromValueSurpriseIrreducible = valueSurpriseIrreducible * valueSurpriseWeight;
      double weightFromValueSurpriseDivergence = (valueSurpriseTotal - valueSurpriseIrreducible) * valueSurpriseWeight;

      double weight = weightFromConstant + weightFromDepth + weightFromPolicySurprise + weightFromValueSurpriseIrreducible + weightFromValueSurpriseDivergence;
      sample.weight = weight;

      std::lock_guard<std::mutex> lock(statsLock);

      if(sample.weight >= minWeight) {
        posWriter.writePos(sample);

        totalWeightFromConstant += weightFromConstant;
        totalWeightFromDepth += weightFromDepth;
        totalWeightFromPolicySurprise += weightFromPolicySurprise;
        totalWeightFromValueSurpriseIrreducible += weightFromValueSurpriseIrreducible;
        totalWeightFromValueSurpriseDivergence += weightFromValueSurpriseDivergence;

        numPositionsWritten += 1;
      }

      numPositionsProcessed += 1;
      if(numPositionsProcessed % 20000 == 0)
        logger.write(
          "Num positions processed: " +
          Global::intToString(numPositionsProcessed) + "/" + Global::intToString(depthByHash.size()) + ", written " + Global::intToString(numPositionsWritten)
        );
    }
  };

  vector<std::thread> threads;
  for(int threadIdx = 0; threadIdx<numThreads; threadIdx++) {
    threads.push_back(std::thread(processPoses, threadIdx));
  }
  for(int threadIdx = 0; threadIdx<numThreads; threadIdx++) {
    threads[threadIdx].join();
  }
  threads.clear();

  posWriter.flushAndStop();

  logger.write("totalWeightFromConstant " + Global::doubleToString(totalWeightFromConstant));
  logger.write("totalWeightFromDepth " + Global::doubleToString(totalWeightFromDepth));
  logger.write("totalWeightFromPolicySurprise " + Global::doubleToString(totalWeightFromPolicySurprise));
  logger.write("totalWeightFromValueSurpriseIrreducible " + Global::doubleToString(totalWeightFromValueSurpriseIrreducible));
  logger.write("totalWeightFromValueSurpriseDivergence " + Global::doubleToString(totalWeightFromValueSurpriseDivergence));
  logger.write("numPositionsWritten " + Global::intToString(numPositionsWritten));

  delete book;
  ScoreValue::freeTables();
  logger.write("DONE");
  return 0;
}

