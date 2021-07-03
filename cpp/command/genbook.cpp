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
  try {
    KataGoCommandLine cmd("View startposes");
    cmd.addConfigFileArg("","",true);
    cmd.addModelFileArg();
    cmd.addOverrideConfigArg();

    cmd.parse(argc,argv);

    cmd.getConfigAllowEmpty(cfg);
    if(cfg.getFileName() != "")
      modelFile = cmd.getModelFile();
  }
  catch (TCLAP::ArgException &e) {
    cerr << "Error: " << e.error() << " for argument " << e.argId() << endl;
    return 1;
  }

  Rand rand;
  Logger logger;
  logger.setLogToStdout(true);

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
  const double utilityPerScore = cfg.getDouble("utilityPerScore",0.0,1000000.0);

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

  Book* book = new Book(
    Board(boardSizeX,boardSizeY),
    rules,
    P_BLACK,
    repBound,
    errorFactor,
    costPerMove,
    costPerUCBWinLossLoss,
    costPerUCBScoreLoss,
    costPerLogPolicy,
    utilityPerScore
  );

  if(!std::atomic_is_lock_free(&shouldStop))
    throw StringError("shouldStop is not lock free, signal-quitting mechanism for terminating matches will NOT work!");
  std::signal(SIGINT, signalHandler);
  std::signal(SIGTERM, signalHandler);

  const int numToExpand = 2;
  for(int rep = 0; rep < 3; rep++) {
    if(shouldStop.load(std::memory_order_acquire))
      break;

    std::vector<SymBookNode> nodesToExpand = book->getNextNToExpand(numToExpand);
    std::vector<SymBookNode> newAndChangedNodes = nodesToExpand;

    for(SymBookNode node: nodesToExpand) {
      BoardHistory hist;
      bool suc = node.getBoardHistoryReachingHere(hist);
      assert(suc);
      (void)suc;

      // Terminal node!
      if(hist.isGameFinished || hist.isPastNormalPhaseEnd || hist.encorePhase > 0) {
        node.canExpand() = false;
        continue;
      }

      Player pla = hist.presumedNextMovePla;
      Board board = hist.getRecentBoard(0);
      search->setPosition(pla,board,hist);

      Board::printBoard(cout, board, Board::NULL_LOC, NULL);
      cout << endl;

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
      assert(!node.isMoveInBook(bestLoc));

      search->printTree(cout, search->rootNode, options, perspective);

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
        newAndChangedNodes.push_back(child);
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

        search->printTree(cout, search->rootNode, options, perspective);

        // Get root values
        ReportedSearchValues remainingSearchValues;
        bool getSuc = search->getPrunedRootValues(remainingSearchValues);
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

  book->exportToHtmlDir("tmpbook");

  delete search;
  delete nnEval;
  delete book;
  ScoreValue::freeTables();
  return 0;
}

