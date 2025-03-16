#include "../tests/tests.h"

#include <algorithm>
#include <iterator>
#include <iomanip>

#include "../core/fileutils.h"
#include "../core/makedir.h"
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
  BookParams params;
  params.errorFactor = 1.05;
  params.costPerMove = 0.53;
  params.costPerUCBWinLossLoss = 2.04;
  params.costPerUCBWinLossLossPow3 = 0.71;
  params.costPerUCBWinLossLossPow7 = 0.67;
  params.costPerUCBScoreLoss = 0.43;
  params.costPerLogPolicy = 0.46;
  params.costPerMovesExpanded = 0.230;
  params.costPerSquaredMovesExpanded = 0.041;
  params.costWhenPassFavored = 1.2;
  params.bonusPerWinLossError = 0.32;
  params.bonusPerScoreError = 0.51;
  params.bonusPerSharpScoreDiscrepancy = 0.54;
  params.bonusPerExcessUnexpandedPolicy = 1.14;
  params.bonusPerUnexpandedBestWinLoss = 1.23;
  params.bonusForWLPV1 = 0.14;
  params.bonusForWLPV2 = 0.17;
  params.bonusForBiggestWLCost = 0.59;
  params.scoreLossCap = 0.95;
  params.earlyBookCostReductionFactor = 0.52;
  params.earlyBookCostReductionLambda = 0.33;
  params.utilityPerScore = 0.11;
  params.policyBoostSoftUtilityScale = 0.034;
  params.utilityPerPolicyForSorting = 0.021;
  params.adjustedVisitsWLScale = 0.05;
  params.maxVisitsForReExpansion = 25;
  params.visitsScale = 50;
  params.sharpScoreOutlierCap = 1.75;

  string testFileName = "./_test.katabook.tmp";

  Book* book = new Book(
    Book::LATEST_BOOK_VERSION,
    initialBoard,
    rules,
    initialPla,
    repBound,
    params
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
          node = child;
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
      nodeValues.sharpScoreMeanRaw = rand.nextGaussian();
      nodeValues.sharpScoreMeanClamped = nodeValues.sharpScoreMeanRaw;
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
      loaded = Book::loadFromFile(testFileName);
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

  const bool logToStdout = false;
  const bool logToStderr = false;
  const bool logTime = false;
  Logger logger(nullptr, logToStdout, logToStderr, logTime);
  logger.addOStream(cout);
  MakeDir::make("./tests/");
  MakeDir::make("./tests/results/");
  MakeDir::make("./tests/results/bookhtml/");
  string htmlDir = "./tests/results/bookhtml/";
  string rulesLabel = "";
  string rulesLink = "";
  int64_t htmlMinVisits = 200;
  bool htmlDevMode = false;
  book->exportToHtmlDir(htmlDir,rulesLabel,rulesLink,htmlDevMode,htmlMinVisits,logger);

  delete book;
  FileUtils::tryRemoveFile(testFileName);
  cout << "Done" << endl;
}
