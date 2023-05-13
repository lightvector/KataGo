#include "../tests/tests.h"

#include <algorithm>
#include <iterator>
#include <iomanip>

#include "../core/fileutils.h"
#include "../book/book.h"
#include "../tests/testsearchcommon.h"

using namespace std;
using namespace TestCommon;
using namespace TestSearchCommon;

void Tests::runBookTests() {
  cout << "Running book tests" << endl;

  Board initialBoard(4,4);
  Rules rules = Rules::parseRules("japanese");
  Player initialPla = P_BLACK;

  int repBound = 9;
  double errorFactor = 1.05;
  double costPerMove = 0.53;
  double costPerUCBWinLossLoss = 2.04;
  double costPerUCBWinLossLossPow3 = 0.71;
  double costPerUCBWinLossLossPow7 = 0.67;
  double costPerUCBScoreLoss = 0.43;
  double costPerLogPolicy = 0.46;
  double costPerMovesExpanded = 0.230;
  double costPerSquaredMovesExpanded = 0.041;
  double costWhenPassFavored = 1.2;
  double bonusPerWinLossError = 0.32;
  double bonusPerScoreError = 0.51;
  double bonusPerSharpScoreDiscrepancy = 0.54;
  double bonusPerExcessUnexpandedPolicy = 1.14;
  double bonusPerUnexpandedBestWinLoss = 1.23;
  double bonusForWLPV1 = 0.14;
  double bonusForWLPV2 = 0.17;
  double bonusForBiggestWLCost = 0.59;
  double scoreLossCap = 0.95;
  double earlyBookCostReductionFactor = 0.52;
  double earlyBookCostReductionLambda = 0.33;
  double utilityPerScore = 0.11;
  double policyBoostSoftUtilityScale = 0.034;
  double utilityPerPolicyForSorting = 0.021;
  double maxVisitsForReExpansion = 25;
  double visitsScale = 50;
  double sharpScoreOutlierCap = 1.75;

  string testFileName = "./_test.katabook.tmp";

  Book* book = new Book(
    Book::LATEST_BOOK_VERSION,
    initialBoard,
    rules,
    initialPla,
    repBound,
    errorFactor,
    costPerMove,
    costPerUCBWinLossLoss,
    costPerUCBWinLossLossPow3,
    costPerUCBWinLossLossPow7,
    costPerUCBScoreLoss,
    costPerLogPolicy,
    costPerMovesExpanded,
    costPerSquaredMovesExpanded,
    costWhenPassFavored,
    bonusPerWinLossError,
    bonusPerScoreError,
    bonusPerSharpScoreDiscrepancy,
    bonusPerExcessUnexpandedPolicy,
    bonusPerUnexpandedBestWinLoss,
    bonusForWLPV1,
    bonusForWLPV2,
    bonusForBiggestWLCost,
    scoreLossCap,
    earlyBookCostReductionFactor,
    earlyBookCostReductionLambda,
    utilityPerScore,
    policyBoostSoftUtilityScale,
    utilityPerPolicyForSorting,
    maxVisitsForReExpansion,
    visitsScale,
    sharpScoreOutlierCap
  );

  Rand rand("runBookTests");
  std::vector<Loc> legalMovesBuf;
  
  for(int waveIdx = 0; waveIdx<35; waveIdx++) {
    cout << "Book size " << book->size() << endl;
    std::set<BookHash> nodesHashesToUpdate;
    
    for(int branchIdx = 0; branchIdx<10; branchIdx++) {
      SymBookNode node = book->getRoot();
      Board board(initialBoard);
      BoardHistory hist(board,initialPla,rules,0); 
      int branchLen = rand.nextUInt(10);
      
      for(int turnIdx = 0; turnIdx<branchLen; turnIdx++) {
        assert(!node.isNull());
        if(hist.isGameFinished || hist.encorePhase != 0)
          break;
        
        Player pla = hist.presumedNextMovePla;
        legalMovesBuf.clear();
        for(int y = 0; y<board.y_size; y++) {
          for(int x = 0; x<board.x_size; x++) {
            Loc loc = Location::getLoc(x,y,board.x_size);
            if(hist.isLegal(board,loc,pla))
              legalMovesBuf.push_back(loc);
          }
        }
        {
          Loc loc = Board::PASS_LOC;
          if(hist.isLegal(board,loc,pla))
            legalMovesBuf.push_back(loc);
        }
        
        Loc moveLoc = legalMovesBuf[rand.nextUInt((uint32_t)legalMovesBuf.size())];
        if(!node.isMoveInBook(moveLoc)) {
          bool childIsTransposing;
          double moveLocPolicy = rand.nextDouble();
          SymBookNode child = node.playAndAddMove(board,hist,moveLoc,moveLocPolicy,childIsTransposing);
          if(!child.isNull() && !childIsTransposing)
            nodesHashesToUpdate.insert(child.hash());
          nodesHashesToUpdate.insert(node.hash());
        }
        else {
          node = node.playMove(board,hist,moveLoc);
        }
      }
    }

    std::vector<SymBookNode> newAndChangedNodes;
    for(BookHash hash: nodesHashesToUpdate) {
      SymBookNode node = book->getByHash(hash);

      BookValues& nodeValues = node.thisValuesNotInBook();
      nodeValues.winLossValue = rand.nextDouble(-1,1);
      nodeValues.scoreMean = rand.nextGaussian();
      nodeValues.sharpScoreMean = rand.nextGaussian();
      nodeValues.winLossError = rand.nextExponential() * 0.1;
      nodeValues.scoreError = rand.nextExponential();
      nodeValues.scoreStdev = rand.nextExponential();

      nodeValues.maxPolicy = rand.nextExponential() * 0.1;
      nodeValues.weight = rand.nextDouble(10,20);
      nodeValues.visits = rand.nextDouble(10,20);

      newAndChangedNodes.push_back(node);
    }

    book->recompute(newAndChangedNodes);
    std::map<BookHash,double> costByHash;
    for(SymBookNode node: book->getAllNodes())
      costByHash[node.hash()] = node.totalExpansionCost();

    Book* loaded = NULL;
    if(waveIdx % 20 == 0) {
      book->saveToFile(testFileName);
      loaded = Book::loadFromFile(testFileName, sharpScoreOutlierCap);
    }
    
    book->recomputeEverything();
    for(SymBookNode node: book->getAllNodes()) {
      testAssert(abs(costByHash[node.hash()] - node.totalExpansionCost()) < 1e-3);
    }

    if(loaded != NULL) {
      for(SymBookNode node: loaded->getAllNodes()) {
        testAssert(abs(costByHash[node.hash()] - node.totalExpansionCost()) < 1e-3);
      }
      delete loaded;
    }
  }    

  FileUtils::tryRemoveFile(testFileName);
  cout << "Done" << endl;
}
