#include "../core/global.h"
#include "../core/makedir.h"
#include "../core/config_parser.h"
#include "../core/timer.h"
#include "../dataio/sgf.h"
#include "../dataio/files.h"
#include "../book/book.h"
#include "../search/asyncbot.h"
#include "../program/setup.h"
#include "../program/playutils.h"
#include "../program/play.h"
#include "../command/commandline.h"
#include "../main.h"

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

static double getMaxPolicy(float policyProbs[NNPos::MAX_NN_POLICY_SIZE]) {
  double maxPolicy = 0.0;
  for(int i = 0; i<NNPos::MAX_NN_POLICY_SIZE; i++)
    if(policyProbs[i] > maxPolicy)
      maxPolicy = policyProbs[i];
  return maxPolicy;
}


int MainCmds::genbook(int argc, const char* const* argv) {
  Board::initHash();
  ScoreValue::initTables();

  ConfigParser cfg;
  string modelFile;
  string htmlDir;
  string bookFile;
  string logFile;
  int numIterations;
  int numToExpandPerIteration;
  int saveEveryIterations;
  bool allowChangingBookParams;
  try {
    KataGoCommandLine cmd("View startposes");
    cmd.addConfigFileArg("","",true);
    cmd.addModelFileArg();
    cmd.addOverrideConfigArg();

    TCLAP::ValueArg<string> htmlDirArg("","html-dir","HTML directory to export to",false,string(),"DIR");
    TCLAP::ValueArg<string> bookFileArg("","book-file","Book file to write to or continue expanding",true,string(),"FILE");
    TCLAP::ValueArg<string> logFileArg("","log-file","Log file to write to",true,string(),"DIR");
    TCLAP::ValueArg<int> numIterationsArg("","num-iters","Number of iterations to expand book",true,0,"N");
    TCLAP::ValueArg<int> numToExpandPerIterationArg("","num-per-iter","Number of nodes per iteration",true,0,"N");
    TCLAP::ValueArg<int> saveEveryIterationsArg("","save-every","Number of iterations per save",true,0,"N");
    TCLAP::SwitchArg allowChangingBookParamsArg("","allow-changing-book-params","Allow changing book params");
    cmd.add(htmlDirArg);
    cmd.add(bookFileArg);
    cmd.add(logFileArg);
    cmd.add(numIterationsArg);
    cmd.add(numToExpandPerIterationArg);
    cmd.add(saveEveryIterationsArg);
    cmd.add(allowChangingBookParamsArg);

    cmd.parse(argc,argv);

    cmd.getConfig(cfg);
    modelFile = cmd.getModelFile();
    htmlDir = htmlDirArg.getValue();
    bookFile = bookFileArg.getValue();
    logFile = logFileArg.getValue();
    numIterations = numIterationsArg.getValue();
    numToExpandPerIteration = numToExpandPerIterationArg.getValue();
    saveEveryIterations = saveEveryIterationsArg.getValue();
    allowChangingBookParams = allowChangingBookParamsArg.getValue();
  }
  catch (TCLAP::ArgException &e) {
    cerr << "Error: " << e.error() << " for argument " << e.argId() << endl;
    return 1;
  }

  Rand rand;
  Logger logger;
  logger.setLogToStdout(true);
  logger.addFile(logFile);

  const bool loadKomiFromCfg = true;
  Rules rules = Setup::loadSingleRules(cfg,loadKomiFromCfg);

  const int boardSizeX = cfg.getInt("boardSizeX",2,Board::MAX_LEN);
  const int boardSizeY = cfg.getInt("boardSizeY",2,Board::MAX_LEN);
  const int repBound = cfg.getInt("repBound",3,1000);
  const double errorFactor = cfg.getDouble("errorFactor",0.0,100.0);
  const double costPerMove = cfg.getDouble("costPerMove",0.0,1000000.0);
  const double costPerUCBWinLossLoss = cfg.getDouble("costPerUCBWinLossLoss",0.0,1000000.0);
  const double costPerUCBScoreLoss = cfg.getDouble("costPerUCBScoreLoss",0.0,1000000.0);
  const double costPerLogPolicy = cfg.getDouble("costPerLogPolicy",0.0,1000000.0);
  const double costPerMovesExpanded = cfg.getDouble("costPerMovesExpanded",0.0,1000000.0);
  const double costPerSquaredMovesExpanded = cfg.getDouble("costPerSquaredMovesExpanded",0.0,1000000.0);
  const double costWhenPassFavored = cfg.getDouble("costWhenPassFavored",0.0,1000000.0);
  const double utilityPerScore = cfg.getDouble("utilityPerScore",0.0,1000000.0);
  const double policyBoostSoftUtilityScale = cfg.getDouble("policyBoostSoftUtilityScale",0.0,1000000.0);
  const double utilityPerPolicyForSorting = cfg.getDouble("utilityPerPolicyForSorting",0.0,1000000.0);
  const bool logSearchInfo = cfg.getBool("logSearchInfo");

  SearchParams params = Setup::loadSingleParams(cfg,Setup::SETUP_FOR_GTP);
  NNEvaluator* nnEval;
  {
    Setup::initializeSession(cfg);
    int maxConcurrentEvals = params.numThreads * 2 + 16; // * 2 + 16 just to give plenty of headroom
    int expectedConcurrentEvals = params.numThreads;
    int defaultMaxBatchSize = std::max(8,((params.numThreads+3)/4)*4);
    string expectedSha256 = "";
    nnEval = Setup::initializeNNEvaluator(
      modelFile,modelFile,expectedSha256,cfg,logger,rand,maxConcurrentEvals,expectedConcurrentEvals,
      boardSizeX,boardSizeY,defaultMaxBatchSize,
      Setup::SETUP_FOR_ANALYSIS
    );
  }
  logger.write("Loaded neural net");

  Search* search;
  {
    string searchRandSeed = Global::uint64ToString(rand.nextUInt64());
    search = new Search(params, nnEval, &logger, searchRandSeed);
  }

  // Check for unused config keys
  cfg.warnUnusedKeys(cerr,&logger);

  if(htmlDir != "")
    MakeDir::make(htmlDir);

  Book* book;
  bool bookFileExists;
  {
    std::ifstream infile(bookFile);
    bookFileExists = infile.good();
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
    if(!allowChangingBookParams) {
      if(
        errorFactor != book->getErrorFactor() ||
        costPerMove != book->getCostPerMove() ||
        costPerUCBWinLossLoss != book->getCostPerUCBWinLossLoss() ||
        costPerUCBScoreLoss != book->getCostPerUCBScoreLoss() ||
        costPerLogPolicy != book->getCostPerLogPolicy() ||
        costPerMovesExpanded != book->getCostPerMovesExpanded() ||
        costPerSquaredMovesExpanded != book->getCostPerSquaredMovesExpanded() ||
        costWhenPassFavored != book->getCostWhenPassFavored() ||
        utilityPerScore != book->getUtilityPerScore() ||
        policyBoostSoftUtilityScale != book->getPolicyBoostSoftUtilityScale() ||
        utilityPerPolicyForSorting != book->getUtilityPerPolicyForSorting()
      ) {
        throw StringError("Book parameters do not match");
      }
    }
    else {
      if(errorFactor != book->getErrorFactor()) { logger.write("Changing errorFactor from " + Global::doubleToString(book->getErrorFactor()) + " to " + Global::doubleToString(errorFactor)); book->setErrorFactor(errorFactor); }
      if(costPerMove != book->getCostPerMove()) { logger.write("Changing costPerMove from " + Global::doubleToString(book->getCostPerMove()) + " to " + Global::doubleToString(costPerMove)); book->setCostPerMove(costPerMove); }
      if(costPerUCBWinLossLoss != book->getCostPerUCBWinLossLoss()) { logger.write("Changing costPerUCBWinLossLoss from " + Global::doubleToString(book->getCostPerUCBWinLossLoss()) + " to " + Global::doubleToString(costPerUCBWinLossLoss)); book->setCostPerUCBWinLossLoss(costPerUCBWinLossLoss); }
      if(costPerUCBScoreLoss != book->getCostPerUCBScoreLoss()) { logger.write("Changing costPerUCBScoreLoss from " + Global::doubleToString(book->getCostPerUCBScoreLoss()) + " to " + Global::doubleToString(costPerUCBScoreLoss)); book->setCostPerUCBScoreLoss(costPerUCBScoreLoss); }
      if(costPerLogPolicy != book->getCostPerLogPolicy()) { logger.write("Changing costPerLogPolicy from " + Global::doubleToString(book->getCostPerLogPolicy()) + " to " + Global::doubleToString(costPerLogPolicy)); book->setCostPerLogPolicy(costPerLogPolicy); }
      if(costPerMovesExpanded != book->getCostPerMovesExpanded()) { logger.write("Changing costPerMovesExpanded from " + Global::doubleToString(book->getCostPerMovesExpanded()) + " to " + Global::doubleToString(costPerMovesExpanded)); book->setCostPerMovesExpanded(costPerMovesExpanded); }
      if(costPerSquaredMovesExpanded != book->getCostPerSquaredMovesExpanded()) { logger.write("Changing costPerSquaredMovesExpanded from " + Global::doubleToString(book->getCostPerSquaredMovesExpanded()) + " to " + Global::doubleToString(costPerSquaredMovesExpanded)); book->setCostPerSquaredMovesExpanded(costPerSquaredMovesExpanded); }
      if(costWhenPassFavored != book->getCostWhenPassFavored()) { logger.write("Changing costWhenPassFavored from " + Global::doubleToString(book->getCostWhenPassFavored()) + " to " + Global::doubleToString(costWhenPassFavored)); book->setCostWhenPassFavored(costWhenPassFavored); }
      if(utilityPerScore != book->getUtilityPerScore()) { logger.write("Changing utilityPerScore from " + Global::doubleToString(book->getUtilityPerScore()) + " to " + Global::doubleToString(utilityPerScore)); book->setUtilityPerScore(utilityPerScore); }
      if(policyBoostSoftUtilityScale != book->getPolicyBoostSoftUtilityScale()) { logger.write("Changing policyBoostSoftUtilityScale from " + Global::doubleToString(book->getPolicyBoostSoftUtilityScale()) + " to " + Global::doubleToString(policyBoostSoftUtilityScale)); book->setPolicyBoostSoftUtilityScale(policyBoostSoftUtilityScale); }
      if(utilityPerPolicyForSorting != book->getUtilityPerPolicyForSorting()) { logger.write("Changing utilityPerPolicyForSorting from " + Global::doubleToString(book->getUtilityPerPolicyForSorting()) + " to " + Global::doubleToString(utilityPerPolicyForSorting)); book->setUtilityPerPolicyForSorting(utilityPerPolicyForSorting); }
    }
    logger.write("Loaded preexisting book with " + Global::uint64ToString(book->size()) + " nodes from " + bookFile);
  }
  else {
    book = new Book(
      Board(boardSizeX,boardSizeY),
      rules,
      P_BLACK,
      repBound,
      errorFactor,
      costPerMove,
      costPerUCBWinLossLoss,
      costPerUCBScoreLoss,
      costPerLogPolicy,
      costPerMovesExpanded,
      costPerSquaredMovesExpanded,
      costWhenPassFavored,
      utilityPerScore,
      policyBoostSoftUtilityScale,
      utilityPerPolicyForSorting
    );
    logger.write("Creating new book at " + bookFile);
    book->saveToFile(bookFile);
    ofstream out(bookFile + ".cfg");
    out << cfg.getContents() << endl;
    out.close();
  }

  if(!std::atomic_is_lock_free(&shouldStop))
    throw StringError("shouldStop is not lock free, signal-quitting mechanism for terminating matches will NOT work!");
  std::signal(SIGINT, signalHandler);
  std::signal(SIGTERM, signalHandler);

  for(int iteration = 0; iteration < numIterations; iteration++) {
    if(shouldStop.load(std::memory_order_acquire))
      break;

    if(iteration % saveEveryIterations == 0 && iteration != 0) {
      logger.write("SAVING TO FILE " + bookFile);
      book->saveToFile(bookFile);
      ofstream out(bookFile + ".cfg");
      out << cfg.getContents() << endl;
      out.close();
    }

    logger.write("BEGINNING BOOK EXPANSION ITERATION " + Global::intToString(iteration));

    std::vector<SymBookNode> nodesToExpand = book->getNextNToExpand(std::min(1+iteration/2,numToExpandPerIteration));
    std::vector<SymBookNode> newAndChangedNodes = nodesToExpand;

    for(SymBookNode node: nodesToExpand) {
      logger.write("Expanding " + node.hash().toString());
      BoardHistory hist;
      std::vector<Loc> moveHistory;
      bool suc = node.getBoardHistoryReachingHere(hist,moveHistory);
      if(!suc) {
        logger.write("WARNING: Failed to get board history reaching node when trying to export to expand book, probably there is some bug");
        logger.write("or else some hash collision or something else is wrong.");
        logger.write("BookHash of node unable to expand: " + node.hash().toString());
        ostringstream movesOut;
        for(Loc move: moveHistory)
          movesOut << Location::toString(move,book->initialBoard) << " ";
        logger.write("Moves:");
        logger.write(movesOut.str());
        logger.write("Marking node as done so we don't try to expand it again, but something is probably wrong.");
        node.canExpand() = false;
        continue;
      }

      // Terminal node!
      if(hist.isGameFinished || hist.isPastNormalPhaseEnd || hist.encorePhase > 0) {
        node.canExpand() = false;
        continue;
      }

      Player pla = hist.presumedNextMovePla;
      Board board = hist.getRecentBoard(0);
      search->setPosition(pla,board,hist);
      search->setRootSymmetryPruningOnly(node.getSymmetries());

      {
        ostringstream out;
        Board::printBoard(out, board, Board::NULL_LOC, NULL);
        logger.write(out.str());
      }

      // Avoid all moves that are currently in the book on this node, only search new stuff.
      auto findNewMoves = [&](std::vector<int>& avoidMoveUntilByLoc) {
        avoidMoveUntilByLoc = std::vector<int>(Board::MAX_ARR_SIZE,0);
        bool hasAtLeastOneLegalNewMove = false;
        for(Loc moveLoc = 0; moveLoc < Board::MAX_ARR_SIZE; moveLoc++) {
          if(hist.isLegal(board,moveLoc,pla)) {
            if(node.isMoveInBook(moveLoc))
              avoidMoveUntilByLoc[moveLoc] = 1;
            else
              hasAtLeastOneLegalNewMove = true;
          }
        }
        return hasAtLeastOneLegalNewMove;
      };

      std::vector<int> avoidMoveUntilByLoc;
      if(!findNewMoves(avoidMoveUntilByLoc)) {
        node.canExpand() = false;
        continue;
      }

      search->setAvoidMoveUntilByLoc(avoidMoveUntilByLoc, avoidMoveUntilByLoc);

      const PrintTreeOptions options;
      const Player perspective = P_WHITE;
      const bool pondering = false;

      {
        double searchFactor = 1.0;
        std::atomic<bool> searchShouldStopNow(false);
        search->runWholeSearch(searchShouldStopNow, NULL, pondering, TimeControls(), searchFactor);
      }
      if(shouldStop.load(std::memory_order_acquire))
        break;

      Loc bestLoc = search->getChosenMoveLoc();
      if(bestLoc == Board::NULL_LOC) {
        logger.write("WARNING: Could not expand since search obtained no results, despite earlier checks about legal moves existing not yet in book");
        logger.write("BookHash of node unable to expand: " + node.hash().toString());
        ostringstream debugOut;
        hist.printDebugInfo(debugOut,board);
        logger.write(debugOut.str());
        logger.write("Marking node as done so we don't try to expand it again, but something is probably wrong.");
        node.canExpand() = false;
        continue;
      }
      assert(!node.isMoveInBook(bestLoc));

      if(logSearchInfo) {
        ostringstream out;
        search->printTree(out, search->rootNode, options, perspective);
        logger.write("Search result");
        logger.write(out.str());
      }

      // Record the values for the move determined to be best
      SymBookNode child;
      {
        float policyProbs[NNPos::MAX_NN_POLICY_SIZE];
        bool policySuc = search->getPolicy(policyProbs);
        assert(policySuc);
        (void)policySuc;

        // Add best child to book
        Board nextBoard = board;
        BoardHistory nextHist = hist;
        double rawPolicy = policyProbs[search->getPos(bestLoc)];
        child = node.playAndAddMove(nextBoard, nextHist, bestLoc, rawPolicy);
        // Somehow child was illegal?
        if(child.isNull()) {
          logger.write("WARNING: Illegal move " + Location::toString(bestLoc, nextBoard));
          ostringstream debugOut;
          nextHist.printDebugInfo(debugOut,nextBoard);
          logger.write(debugOut.str());
          logger.write("BookHash of parent: " + node.hash().toString());
          logger.write("Marking node as done so we don't try to expand it again, but something is probably wrong.");
          node.canExpand() = false;
          continue;
        }

        newAndChangedNodes.push_back(child);
        logger.write("Adding " + child.hash().toString() + " move " + Location::toString(bestLoc,board));
        BookValues& childValues = child.thisValuesNotInBook();

        // Find child node from search and its values
        const SearchNode* childSearchNode = search->getChildForMove(search->getRootNode(), bestLoc);
        ReportedSearchValues childSearchValues;
        bool getSuc = search->getPrunedNodeValues(childSearchNode,childSearchValues);
        assert(getSuc);
        (void)getSuc;

        // Record those values to the book
        childValues.winLossValue = childSearchValues.winLossValue;
        childValues.scoreMean = childSearchValues.expectedScore;
        childValues.lead = childSearchValues.lead;
        std::pair<double,double> errors = search->getAverageShorttermWLAndScoreError(childSearchNode);
        childValues.winLossError = errors.first;
        childValues.scoreError = errors.second;
        childValues.scoreStdev = childSearchValues.expectedScoreStdev;

        // Could return false if child is terminal, or otherwise has no nn eval.
        bool policySuc2 = search->getPolicy(childSearchNode, policyProbs);
        double maxPolicy = policySuc2 ? getMaxPolicy(policyProbs) : 1.0;
        assert(maxPolicy >= 0.0);
        childValues.maxPolicy = maxPolicy;
        childValues.weight = childSearchValues.weight;
        childValues.visits = childSearchValues.visits;
      }

      // Now that the child is in the book, see if there are more moves left
      // If there are none left, we need to set the thisValuesNotInBook appropriately.
      if(!findNewMoves(avoidMoveUntilByLoc)) {
        BookValues& nodeValues = node.thisValuesNotInBook();
        if(node.pla() == P_WHITE) {
          nodeValues.winLossValue = -1e20;
          nodeValues.scoreMean = -1e20;
          nodeValues.lead =  -1e20;
        }
        else {
          nodeValues.winLossValue = 1e20;
          nodeValues.scoreMean = 1e20;
          nodeValues.lead =  1e20;
        }
        nodeValues.winLossError = 0.0;
        nodeValues.scoreError = 0.0;
        nodeValues.scoreStdev = 0.0;
        nodeValues.maxPolicy = 0.0;
        nodeValues.weight = 0.0;
        nodeValues.visits = 0;

        node.canExpand() = false;
      }
      else {
        search->setAvoidMoveUntilByLoc(avoidMoveUntilByLoc, avoidMoveUntilByLoc);

        // Do a half-cost search on the remaining moves at the node
        {
          double searchFactor = 0.5;
          std::atomic<bool> searchShouldStopNow(false);
          search->runWholeSearch(searchShouldStopNow, NULL, pondering, TimeControls(), searchFactor);
        }
        if(shouldStop.load(std::memory_order_acquire))
          break;

        if(logSearchInfo) {
          logger.write("Quick search on remainimg moves");
          ostringstream out;
          search->printTree(out, search->rootNode, options, perspective);
          logger.write(out.str());
        }

        // Get root values
        ReportedSearchValues remainingSearchValues;
        bool getSuc = search->getPrunedRootValues(remainingSearchValues);
        // Something is bad if this is false, since we should be searching with positive visits and getting an nneval
        // and we know node is not a terminal node.
        assert(getSuc);
        (void)getSuc;

        // Record those values to the book
        BookValues& nodeValues = node.thisValuesNotInBook();
        nodeValues.winLossValue = remainingSearchValues.winLossValue;
        nodeValues.scoreMean = remainingSearchValues.expectedScore;
        nodeValues.lead = remainingSearchValues.lead;
        std::pair<double,double> errors = search->getAverageShorttermWLAndScoreError(search->getRootNode());
        nodeValues.winLossError = errors.first;
        nodeValues.scoreError = errors.second;
        nodeValues.scoreStdev = remainingSearchValues.expectedScoreStdev;

        // Just in case, handle failure case with policySuc2
        float policyProbs[NNPos::MAX_NN_POLICY_SIZE];
        bool policySuc2 = search->getPolicy(policyProbs);
        // Zero out all the policies for moves we already have, we want the max *remaining* policy
        for(Loc loc = 0; loc<Board::MAX_ARR_SIZE; loc++) {
          if(avoidMoveUntilByLoc[loc] > 0) {
            int pos = search->getPos(loc);
            assert(pos >= 0 && pos < NNPos::MAX_NN_POLICY_SIZE);
            policyProbs[pos] = -1;
          }
        }
        double maxPolicy = policySuc2 ? getMaxPolicy(policyProbs) : 1.0;
        assert(maxPolicy >= 0.0);
        nodeValues.maxPolicy = maxPolicy;
        nodeValues.weight = remainingSearchValues.weight;
        nodeValues.visits = remainingSearchValues.visits;
      }
    }

    if(shouldStop.load(std::memory_order_acquire))
      break;

    book->recompute(newAndChangedNodes);
  }

  if(numIterations > 0) {
    logger.write("SAVING TO FILE " + bookFile);
    book->saveToFile(bookFile);
    ofstream out(bookFile + ".cfg");
    out << cfg.getContents() << endl;
    out.close();
  }

  if(htmlDir != "") {
    logger.write("EXPORTING HTML TO " + htmlDir);
    book->exportToHtmlDir(htmlDir,logger);
  }

  delete search;
  delete nnEval;
  delete book;
  ScoreValue::freeTables();
  logger.write("DONE");
  return 0;
}

