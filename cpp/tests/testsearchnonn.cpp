#include "../tests/tests.h"

#include <algorithm>
#include <cstdlib>
#include <iterator>
#include <iomanip>

#include "../core/fileutils.h"
#include "../dataio/files.h"
#include "../dataio/sgf.h"
#include "../neuralnet/nninputs.h"
#include "../search/asyncbot.h"
#include "../search/evalcache.h"
#include "../search/searchnode.h"
#include "../program/playutils.h"
#include "../program/setup.h"
#include "../tests/testsearchcommon.h"

using namespace std;
using namespace TestCommon;
using namespace TestSearchCommon;


void Tests::runNNLessSearchTests() {
  cout << "Running neuralnetless search tests" << endl;
  NeuralNet::globalInitialize();

  //Placeholder, doesn't actually do anything since we have debugSkipNeuralNet = true
  string modelFile = "/dev/null";

  const bool logToStdout = false;
  const bool logToStderr = false;
  const bool logTime = false;
  Logger logger(nullptr, logToStdout, logToStderr, logTime);
  logger.addOStream(cout);

  {
    cout << "===================================================================" << endl;
    cout << "Basic search with debugSkipNeuralNet and chosen move randomization" << endl;
    cout << "===================================================================" << endl;

    NNEvaluator* nnEval = startNNEval(modelFile,logger,"",NNPos::MAX_BOARD_LEN,NNPos::MAX_BOARD_LEN,0,true,false,false,true,false);
    SearchParams params;
    params.maxVisits = 100;
    Search* search = new Search(params, nnEval, &logger, "autoSearchRandSeed");
    Rules rules = Rules::getTrompTaylorish();
    TestSearchOptions opts;

    Board board = Board::parseBoard(9,9,R"%%(
.........
.........
..x..o...
.........
..x...o..
...o.....
..o.x.x..
.........
.........
)%%");
    Player nextPla = P_BLACK;
    BoardHistory hist(board,nextPla,rules,0);

    search->setPosition(nextPla,board,hist);
    search->runWholeSearch(nextPla);

    PrintTreeOptions options;
    options = options.maxDepth(1);
    search->printTree(cout, search->rootNode, options, P_WHITE);

    auto sampleChosenMoves = [&]() {
      std::map<Loc,int> moveLocsAndCounts;
      for(int i = 0; i<10000; i++) {
        Loc loc = search->getChosenMoveLoc();
        moveLocsAndCounts[loc] += 1;
      }
      vector<pair<Loc,int>> moveLocsAndCountsSorted;
      std::copy(moveLocsAndCounts.begin(),moveLocsAndCounts.end(),std::back_inserter(moveLocsAndCountsSorted));
      std::sort(moveLocsAndCountsSorted.begin(), moveLocsAndCountsSorted.end(), [](const pair<Loc,int>& a, const pair<Loc,int>& b) { return a.second > b.second; });

      for(int i = 0; i<moveLocsAndCountsSorted.size(); i++) {
        cout << Location::toString(moveLocsAndCountsSorted[i].first,board) << " " << moveLocsAndCountsSorted[i].second << endl;
      }
    };

    cout << "Chosen moves at temperature 0" << endl;
    sampleChosenMoves();

    {
      cout << "Chosen moves at temperature 1 but early temperature 0, when it's perfectly early" << endl;
      search->searchParams.chosenMoveTemperature = 1.0;
      search->searchParams.chosenMoveTemperatureEarly = 0.0;
      sampleChosenMoves();
    }

    {
      cout << "Chosen moves at temperature 1" << endl;
      search->searchParams.chosenMoveTemperature = 1.0;
      search->searchParams.chosenMoveTemperatureEarly = 1.0;
      sampleChosenMoves();
    }

    {
      cout << "Chosen moves at some intermediate temperature" << endl;
      //Ugly hack to artifically fill history. Breaks all sorts of invariants, but should work to
      //make the search htink there's some history to choose an intermediate temperature
      for(int i = 0; i<16; i++)
        search->rootHistory.moveHistory.emplace_back(Board::NULL_LOC,P_BLACK);

      search->searchParams.chosenMoveTemperature = 1.0;
      search->searchParams.chosenMoveTemperatureEarly = 0.0;
      search->searchParams.chosenMoveTemperatureHalflife = 16.0 * 19.0/9.0;
      sampleChosenMoves();
    }

    delete search;
    delete nnEval;
    cout << endl;
  }

  {
    cout << "===================================================================" << endl;
    cout << "Testing preservation of search tree across moves" << endl;
    cout << "===================================================================" << endl;

    NNEvaluator* nnEval = startNNEval(modelFile,logger,"",NNPos::MAX_BOARD_LEN,NNPos::MAX_BOARD_LEN,0,true,false,false,true,false);
    SearchParams params;
    params.maxVisits = 100;
    params.cpuctExploration *= 2;
    Search* search = new Search(params, nnEval, &logger, "autoSearchRandSeed");
    Rules rules = Rules::getTrompTaylorish();
    TestSearchOptions opts;

    Board board = Board::parseBoard(7,7,R"%%(
..xx...
xxxxxxx
.xx..xx
.xxoooo
xxxo...
ooooooo
...o...
)%%");
    Player nextPla = P_BLACK;
    BoardHistory hist(board,nextPla,rules,0);

    {
      //--------------------------------------
      cout << "First perform a basic search." << endl;

      search->setPosition(nextPla,board,hist);
      search->runWholeSearch(nextPla);

      ConstSearchNodeChildrenReference children = search->rootNode->getChildren();
      int childrenCapacity = children.getCapacity();
      testAssert(childrenCapacity > 1);

      //In theory nothing requires this, but it would be kind of crazy if this were false
      testAssert(children.iterateAndCountChildren() > 1);
      testAssert(children[1].getIfAllocated() != NULL);

      Loc locToDescend = children[1].getMoveLoc();

      PrintTreeOptions options;
      options = options.maxDepth(1);
      cout << search->rootBoard << endl;
      search->printTree(cout, search->rootNode, options, P_WHITE);
      search->printTree(cout, search->rootNode, options.onlyBranch(board,Location::toString(locToDescend,board)), P_WHITE);

      cout << endl;

      //--------------------------------------
      cout << "Next, make a move, and with no search, print the tree." << endl;

      search->makeMove(locToDescend,nextPla);
      nextPla = getOpp(nextPla);

      cout << search->rootBoard << endl;
      search->printTree(cout, search->rootNode, options, P_WHITE);
      cout << endl;

      //--------------------------------------
      cout << "Then continue the search to complete 100 visits." << endl;

      search->runWholeSearch(nextPla);
      search->printTree(cout, search->rootNode, options, P_WHITE);
      cout << endl;
    }

    delete search;
    delete nnEval;

    cout << endl;
  }


  {
    cout << "===================================================================" << endl;
    cout << "Testing pruning of search tree across moves due to root restrictions" << endl;
    cout << "===================================================================" << endl;

    Board board = Board::parseBoard(7,7,R"%%(
..xx...
xx.xxxx
x.xx.xx
.xxoooo
xxxo..x
ooooooo
o..oo.x
)%%");
    Player nextPla = P_BLACK;
    Rules rules = Rules::getTrompTaylorish();
    BoardHistory hist(board,nextPla,rules,0);
    hist.makeBoardMoveAssumeLegal(board,Location::ofString("B5",board),nextPla,NULL);
    nextPla = getOpp(nextPla);
    hist.makeBoardMoveAssumeLegal(board,Location::ofString("pass",board),nextPla,NULL);
    nextPla = getOpp(nextPla);
    hist.makeBoardMoveAssumeLegal(board,Location::ofString("C6",board),nextPla,NULL);
    nextPla = getOpp(nextPla);
    hist.makeBoardMoveAssumeLegal(board,Location::ofString("pass",board),nextPla,NULL);
    nextPla = getOpp(nextPla);
    hist.makeBoardMoveAssumeLegal(board,Location::ofString("G7",board),nextPla,NULL);
    nextPla = getOpp(nextPla);
    hist.makeBoardMoveAssumeLegal(board,Location::ofString("pass",board),nextPla,NULL);
    nextPla = getOpp(nextPla);
    hist.makeBoardMoveAssumeLegal(board,Location::ofString("F3",board),nextPla,NULL);
    nextPla = getOpp(nextPla);
    hist.makeBoardMoveAssumeLegal(board,Location::ofString("pass",board),nextPla,NULL);
    nextPla = getOpp(nextPla);

    auto hasSuicideRootMoves = [](const Search* search) {
      ConstSearchNodeChildrenReference children = search->rootNode->getChildren();
      int childrenCapacity = children.getCapacity();
      for(int i = 0; i<childrenCapacity; i++) {
        const SearchNode* child = children[i].getIfAllocated();
        if(child == NULL)
          break;
        if(search->rootBoard.isSuicide(children[i].getMoveLoc(),search->rootPla))
          return true;
      }
      return false;
    };
    auto hasPassAliveRootMoves = [](const Search* search) {
      ConstSearchNodeChildrenReference children = search->rootNode->getChildren();
      int childrenCapacity = children.getCapacity();
      for(int i = 0; i<childrenCapacity; i++) {
        const SearchNode* child = children[i].getIfAllocated();
        if(child == NULL)
          break;
        if(search->rootSafeArea[children[i].getMoveLoc()] != C_EMPTY)
          return true;
      }
      return false;
    };


    {
      cout << "First with no pruning" << endl;
      NNEvaluator* nnEval = startNNEval(modelFile,logger,"seed1",NNPos::MAX_BOARD_LEN,NNPos::MAX_BOARD_LEN,0,true,false,false,true,false);
      SearchParams params;
      params.maxVisits = 400;
      Search* search = new Search(params, nnEval, &logger, "autoSearchRandSeed3");
      TestSearchOptions opts;

      search->setPosition(nextPla,board,hist);
      search->runWholeSearch(nextPla);
      PrintTreeOptions options;
      options = options.maxDepth(1);
      cout << search->rootBoard << endl;
      search->printTree(cout, search->rootNode, options, P_WHITE);

      testAssert(hasSuicideRootMoves(search));

      delete search;
      delete nnEval;

      cout << endl;
    }

    {
      cout << "Next, with rootPruneUselessMoves" << endl;
      NNEvaluator* nnEval = startNNEval(modelFile,logger,"seed1",NNPos::MAX_BOARD_LEN,NNPos::MAX_BOARD_LEN,0,true,false,false,true,false);
      SearchParams params;
      params.maxVisits = 400;
      params.rootPruneUselessMoves = true;
      Search* search = new Search(params, nnEval, &logger, "autoSearchRandSeed3");
      TestSearchOptions opts;

      search->setPosition(nextPla,board,hist);
      search->runWholeSearch(nextPla);
      PrintTreeOptions options;
      options = options.maxDepth(1);
      cout << search->rootBoard << endl;
      search->printTree(cout, search->rootNode, options, P_WHITE);

      testAssert(!hasSuicideRootMoves(search));

      delete search;
      delete nnEval;

      cout << endl;
    }

    cout << "Progress the game, having black fill space while white passes..." << endl;
    hist.makeBoardMoveAssumeLegal(board,Location::ofString("A7",board),nextPla,NULL);
    nextPla = getOpp(nextPla);
    hist.makeBoardMoveAssumeLegal(board,Location::ofString("pass",board),nextPla,NULL);
    nextPla = getOpp(nextPla);
    hist.makeBoardMoveAssumeLegal(board,Location::ofString("E7",board),nextPla,NULL);
    nextPla = getOpp(nextPla);
    hist.makeBoardMoveAssumeLegal(board,Location::ofString("pass",board),nextPla,NULL);
    nextPla = getOpp(nextPla);
    hist.makeBoardMoveAssumeLegal(board,Location::ofString("F7",board),nextPla,NULL);
    nextPla = getOpp(nextPla);

    {
      cout << "Searching on the opponent, the move before" << endl;
      NNEvaluator* nnEval = startNNEval(modelFile,logger,"seed1b",NNPos::MAX_BOARD_LEN,NNPos::MAX_BOARD_LEN,0,true,false,false,true,false);
      SearchParams params;
      params.maxVisits = 400;
      params.rootPruneUselessMoves = true;
      Search* search = new Search(params, nnEval, &logger, "autoSearchRandSeed3");
      TestSearchOptions opts;

      search->setPosition(nextPla,board,hist);
      search->runWholeSearch(nextPla);
      PrintTreeOptions options;
      options = options.maxDepth(1);
      cout << search->rootBoard << endl;
      search->printTree(cout, search->rootNode, options, P_WHITE);
      search->printTree(cout, search->rootNode, options.onlyBranch(board,"pass"), P_WHITE);

      cout << endl;

      cout << "Now play forward the pass. The tree should still have useless suicides and also other moves in it" << endl;
      search->makeMove(Board::PASS_LOC,nextPla);
      testAssert(hasSuicideRootMoves(search));
      testAssert(hasPassAliveRootMoves(search));

      cout << search->rootBoard << endl;
      search->printTree(cout, search->rootNode, options, P_WHITE);

      cout << endl;

      cout << "But the moment we begin a search, it should no longer." << endl;
      search->beginSearch(false);
      testAssert(!hasSuicideRootMoves(search));
      testAssert(!hasPassAliveRootMoves(search));

      cout << search->rootBoard << endl;
      search->printTree(cout, search->rootNode, options, P_WHITE);

      cout << endl;

      cout << "Continue searching a bit more" << endl;
      search->runWholeSearch(getOpp(nextPla));

      cout << search->rootBoard << endl;
      search->printTree(cout, search->rootNode, options, P_WHITE);

      delete search;
      delete nnEval;
      cout << endl;
    }

  }

  {
    cout << "===================================================================" << endl;
    cout << "Testing search tree update near terminal positions" << endl;
    cout << "===================================================================" << endl;

    Board board = Board::parseBoard(7,7,R"%%(
x.xx.xx
xxx.xxx
xxxxxxx
xxxxxxx
ooooooo
ooooooo
o..o.oo
)%%");

    Player nextPla = P_WHITE;
    Rules rules = Rules::getTrompTaylorish();
    rules.multiStoneSuicideLegal = false;
    BoardHistory hist(board,nextPla,rules,0);

    {
      cout << "First with no pruning" << endl;
      NNEvaluator* nnEval = startNNEval(modelFile,logger,"seed1",NNPos::MAX_BOARD_LEN,NNPos::MAX_BOARD_LEN,0,true,false,false,true,false);
      SearchParams params;
      params.maxVisits = 400;
      params.dynamicScoreUtilityFactor = 0.5;
      params.useLcbForSelection = true;

      Search* search = new Search(params, nnEval, &logger, "autoSearchRandSeed3");
      TestSearchOptions opts;

      search->setPosition(nextPla,board,hist);
      search->runWholeSearch(nextPla);
      PrintTreeOptions options;
      options = options.maxDepth(1);
      options = options.printSqs(true);
      cout << search->rootBoard << endl;
      search->printTree(cout, search->rootNode, options, P_WHITE);

      cout << "Begin search is idempotent?" << endl;
      search->beginSearch(false);
      search->printTree(cout, search->rootNode, options, P_WHITE);
      search->makeMove(Location::ofString("B1",board),nextPla);
      search->printTree(cout, search->rootNode, options, P_WHITE);
      search->beginSearch(false);
      search->printTree(cout, search->rootNode, options, P_WHITE);

      delete search;
      delete nnEval;

      cout << endl;
    }
  }

  {
    cout << "===================================================================" << endl;
    cout << "Testing pruning of search tree at root node due to symmetries: empty board" << endl;
    cout << "===================================================================" << endl;

    NNEvaluator* nnEval = startNNEval(modelFile,logger,"",9,9,0,true,false,false,true,false);
    SearchParams params;
    params.maxVisits = 5000;
    params.cpuctExploration *= 4;
    params.rootSymmetryPruning = true;
    Search* search = new Search(params, nnEval, &logger, "autoSearchRandSeed");
    Rules rules = Rules::getTrompTaylorish();
    TestSearchOptions opts;

    Board board = Board::parseBoard(9,9,R"%%(
.........
.........
.........
.........
.........
.........
.........
.........
.........
)%%");
    Player nextPla = P_BLACK;
    BoardHistory hist(board,nextPla,rules,0);

    search->setPosition(nextPla,board,hist);
    search->runWholeSearch(nextPla);

    cout << search->rootBoard << endl;

    PrintTreeOptions options;
    options = options.maxDepth(1);
    search->printTree(cout, search->rootNode, options, P_WHITE);
    //Print the subtree under a move
    cout << "H8 subtree: " << endl;
    Loc locToDescend = Location::ofString("H8",search->rootBoard);
    search->printTree(cout, search->rootNode, options.onlyBranch(search->rootBoard,Location::toString(locToDescend,search->rootBoard)), P_WHITE);

    //Enumerate the tree and make sure every node is indeed hit exactly once in postorder.
    TestSearchCommon::verifyTreePostOrder(search,-1);

    //--------------------------------------
    cout << "Next, make a move, and with no search, print the tree." << endl;

    search->makeMove(locToDescend,nextPla);
    nextPla = getOpp(nextPla);

    cout << search->rootBoard << endl;
    search->printTree(cout, search->rootNode, options, P_WHITE);
    cout << endl;

    //Enumerate the tree and make sure every node is indeed hit exactly once in postorder.
    TestSearchCommon::verifyTreePostOrder(search,-1);

    //--------------------------------------
    cout << "Begin search but make no additional playouts, print the tree." << endl;
    const bool pondering = false;
    search->beginSearch(pondering);
    search->printTree(cout, search->rootNode, options, P_WHITE);
    cout << endl;

    //Enumerate the tree and make sure every node is indeed hit exactly once in postorder.
    TestSearchCommon::verifyTreePostOrder(search,-1);

    delete search;
    delete nnEval;
    cout << endl;
  }

  {
    cout << "===================================================================" << endl;
    cout << "Testing pruning of search tree at root node due to symmetries: asymmetry board" << endl;
    cout << "===================================================================" << endl;

    NNEvaluator* nnEval = startNNEval(modelFile,logger,"",9,9,0,true,false,false,true,false);
    SearchParams params;
    params.maxVisits = 2000;
    params.rootSymmetryPruning = true;
    Search* search = new Search(params, nnEval, &logger, "autoSearchRandSeed");
    Rules rules = Rules::getTrompTaylorish();
    TestSearchOptions opts;

    Board board = Board::parseBoard(9,9,R"%%(
.........
.........
.........
.........
.......O.
.........
.........
...X.....
.........
)%%");
    Player nextPla = P_BLACK;
    BoardHistory hist(board,nextPla,rules,0);

    search->setPosition(nextPla,board,hist);
    search->runWholeSearch(nextPla);

    cout << search->rootBoard << endl;

    PrintTreeOptions options;
    options = options.maxDepth(1);
    search->printTree(cout, search->rootNode, options, P_WHITE);

    delete search;
    delete nnEval;
    cout << endl;
  }

  {
    cout << "===================================================================" << endl;
    cout << "Testing pruning of search tree at root node due to symmetries, allowing only diagonal flips: empty board" << endl;
    cout << "===================================================================" << endl;

    NNEvaluator* nnEval = startNNEval(modelFile,logger,"",9,9,0,true,false,false,true,false);
    SearchParams params;
    params.maxVisits = 5000;
    params.rootSymmetryPruning = true;
    Search* search = new Search(params, nnEval, &logger, "autoSearchRandSeed");
    Rules rules = Rules::getTrompTaylorish();
    TestSearchOptions opts;

    Board board = Board::parseBoard(9,9,R"%%(
.........
.........
.........
.........
.........
.........
.........
.........
.........
)%%");
    Player nextPla = P_BLACK;
    BoardHistory hist(board,nextPla,rules,0);

    search->setPosition(nextPla,board,hist);
    search->setRootSymmetryPruningOnly({0,3,4,7});
    search->runWholeSearch(nextPla);

    cout << search->rootBoard << endl;

    PrintTreeOptions options;
    options = options.maxDepth(1);
    search->printTree(cout, search->rootNode, options, P_WHITE);

    delete search;
    delete nnEval;
    cout << endl;
  }

  {
    cout << "===================================================================" << endl;
    cout << "Testing pruning of search tree at root node due to symmetries, allowing only diagonal flips: only one diagonal possible" << endl;
    cout << "===================================================================" << endl;

    NNEvaluator* nnEval = startNNEval(modelFile,logger,"",9,9,0,true,false,false,true,false);
    SearchParams params;
    params.maxVisits = 5000;
    params.rootSymmetryPruning = true;
    Search* search = new Search(params, nnEval, &logger, "autoSearchRandSeed");
    Rules rules = Rules::getTrompTaylorish();
    TestSearchOptions opts;

    Board board = Board::parseBoard(9,9,R"%%(
.........
.........
.........
.........
....o....
.........
..x......
.........
.........
)%%");
    Player nextPla = P_BLACK;
    BoardHistory hist(board,nextPla,rules,0);

    search->setPosition(nextPla,board,hist);
    search->setRootSymmetryPruningOnly({0,3,4,7});
    search->runWholeSearch(nextPla);

    cout << search->rootBoard << endl;

    PrintTreeOptions options;
    options = options.maxDepth(1);
    search->printTree(cout, search->rootNode, options, P_WHITE);

    delete search;
    delete nnEval;
    cout << endl;
  }

  {
    cout << "===================================================================" << endl;
    cout << "Testing pruning of search tree at root node due to symmetries, allowing only diagonal flips: empty board, with avoid moves" << endl;
    cout << "===================================================================" << endl;

    NNEvaluator* nnEval = startNNEval(modelFile,logger,"",9,9,0,true,false,false,true,false);
    SearchParams params;
    params.maxVisits = 5000;
    params.rootSymmetryPruning = true;
    params.wideRootNoise = 0.05;
    Search* search = new Search(params, nnEval, &logger, "autoSearchRandSeed");
    Rules rules = Rules::getTrompTaylorish();
    TestSearchOptions opts;

    Board board = Board::parseBoard(9,9,R"%%(
.......xx
xx.....xx
.........
....x....
.........
.........
.....x...
xx......x
.......xx
)%%");

    vector<int> avoidMoveUntilByLoc(Board::MAX_ARR_SIZE);
    for(int y = 0; y < board.y_size; y++) {
      for(int x = 0; x < board.x_size; x++) {
        Loc loc = Location::getLoc(x,y,board.x_size);
        if(board.colors[loc] == C_BLACK) {
          avoidMoveUntilByLoc[loc] = 0;
          board.setStone(loc,C_EMPTY);
        }
        else
          avoidMoveUntilByLoc[loc] = (y % 2 == 0 ? 1 : 2);
      }
    }

    Player nextPla = P_BLACK;
    BoardHistory hist(board,nextPla,rules,0);

    search->setPosition(nextPla,board,hist);
    search->setRootSymmetryPruningOnly({0,3,4,7});
    search->setAvoidMoveUntilByLoc(avoidMoveUntilByLoc,avoidMoveUntilByLoc);
    search->runWholeSearch(nextPla);

    cout << search->rootBoard << endl;

    PrintTreeOptions options;
    options = options.maxDepth(1);
    search->printTree(cout, search->rootNode, options, P_WHITE);

    nlohmann::json json;
    Player perspective = P_WHITE;
    int analysisPVLen = 2;
    bool preventEncore = true;
    bool includePolicy = false;
    bool includeOwnership = false;
    bool includeOwnershipStdev = false;
    bool includeMovesOwnership = false;
    bool includeMovesOwnershipStdev = false;
    bool includePVVisits = false;
    bool includeNoResultValue = false;
    bool suc = search->getAnalysisJson(
      perspective, analysisPVLen, preventEncore,
      includePolicy, includeOwnership, includeOwnershipStdev, includeMovesOwnership, includeMovesOwnershipStdev, includePVVisits,
      includeNoResultValue,
      json
    );
    testAssert(suc);
    cout << json << endl;

    delete search;
    delete nnEval;
    cout << endl;
  }

  {
    cout << "===================================================================" << endl;
    cout << "Testing pruning of search tree at root node due to symmetries, allowing only diagonal flips: only one diagonal possible, with avoid moves" << endl;
    cout << "===================================================================" << endl;

    NNEvaluator* nnEval = startNNEval(modelFile,logger,"",9,9,0,true,false,false,true,false);
    SearchParams params;
    params.maxVisits = 5000;
    params.rootSymmetryPruning = true;
    params.wideRootNoise = 0.05;
    Search* search = new Search(params, nnEval, &logger, "autoSearchRandSeed");
    Rules rules = Rules::getTrompTaylorish();
    TestSearchOptions opts;

    Board board = Board::parseBoard(9,9,R"%%(
.......xx
xx.....xx
.........
....xo...
....o....
.........
.....x...
xx......x
.......xx
)%%");

    vector<int> avoidMoveUntilByLoc(Board::MAX_ARR_SIZE);
    for(int y = 0; y < board.y_size; y++) {
      for(int x = 0; x < board.x_size; x++) {
        Loc loc = Location::getLoc(x,y,board.x_size);
        if(board.colors[loc] == C_BLACK) {
          avoidMoveUntilByLoc[loc] = 0;
          board.setStone(loc,C_EMPTY);
        }
        else
          avoidMoveUntilByLoc[loc] = (y % 2 == 0 ? 1 : 2);
      }
    }

    Player nextPla = P_BLACK;
    BoardHistory hist(board,nextPla,rules,0);

    search->setPosition(nextPla,board,hist);
    search->setRootSymmetryPruningOnly({0,3,4,7});
    search->setAvoidMoveUntilByLoc(avoidMoveUntilByLoc,avoidMoveUntilByLoc);
    search->runWholeSearch(nextPla);

    cout << search->rootBoard << endl;

    PrintTreeOptions options;
    options = options.maxDepth(1);
    search->printTree(cout, search->rootNode, options, P_WHITE);

    nlohmann::json json;
    Player perspective = P_WHITE;
    int analysisPVLen = 2;
    bool preventEncore = true;
    bool includePolicy = false;
    bool includeOwnership = false;
    bool includeOwnershipStdev = false;
    bool includeMovesOwnership = false;
    bool includeMovesOwnershipStdev = false;
    bool includePVVisits = false;
    bool includeNoResultValue = false;
    bool suc = search->getAnalysisJson(
      perspective, analysisPVLen, preventEncore,
      includePolicy, includeOwnership, includeOwnershipStdev, includeMovesOwnership, includeMovesOwnershipStdev, includePVVisits,
      includeNoResultValue,
      json
    );
    testAssert(suc);
    cout << json << endl;

    delete search;
    delete nnEval;
    cout << endl;
  }

  {
    cout << "===================================================================" << endl;
    cout << "Testing pruning of search tree at root node, no symmetries, with avoid moves" << endl;
    cout << "===================================================================" << endl;

    NNEvaluator* nnEval = startNNEval(modelFile,logger,"",9,9,0,true,false,false,true,false);
    SearchParams params;
    params.maxVisits = 5000;
    params.rootSymmetryPruning = false;
    params.wideRootNoise = 0.05;
    Search* search = new Search(params, nnEval, &logger, "autoSearchRandSeed");
    Rules rules = Rules::getTrompTaylorish();
    TestSearchOptions opts;

    Board board = Board::parseBoard(9,9,R"%%(
.......xx
xx.....xx
.........
....xo...
....o....
.........
.....x...
xx......x
.......xx
)%%");

    vector<int> avoidMoveUntilByLoc(Board::MAX_ARR_SIZE);
    for(int y = 0; y < board.y_size; y++) {
      for(int x = 0; x < board.x_size; x++) {
        Loc loc = Location::getLoc(x,y,board.x_size);
        if(board.colors[loc] == C_BLACK) {
          avoidMoveUntilByLoc[loc] = 0;
          board.setStone(loc,C_EMPTY);
        }
        else
          avoidMoveUntilByLoc[loc] = (y % 2 == 0 ? 1 : 2);
      }
    }

    Player nextPla = P_BLACK;
    BoardHistory hist(board,nextPla,rules,0);

    search->setPosition(nextPla,board,hist);
    search->setRootSymmetryPruningOnly({0,3,4,7});
    search->setAvoidMoveUntilByLoc(avoidMoveUntilByLoc,avoidMoveUntilByLoc);
    search->runWholeSearch(nextPla);

    cout << search->rootBoard << endl;

    PrintTreeOptions options;
    options = options.maxDepth(1);
    search->printTree(cout, search->rootNode, options, P_WHITE);

    nlohmann::json json;
    Player perspective = P_WHITE;
    int analysisPVLen = 2;
    bool preventEncore = true;
    bool includePolicy = false;
    bool includeOwnership = false;
    bool includeOwnershipStdev = false;
    bool includeMovesOwnership = false;
    bool includeMovesOwnershipStdev = false;
    bool includePVVisits = false;
    bool includeNoResultValue = false;
    bool suc = search->getAnalysisJson(
      perspective, analysisPVLen, preventEncore,
      includePolicy, includeOwnership, includeOwnershipStdev, includeMovesOwnership, includeMovesOwnershipStdev, includePVVisits,
      includeNoResultValue,
      json
    );
    testAssert(suc);
    cout << json << endl;

    delete search;
    delete nnEval;
    cout << endl;
  }

  {
    cout << "===================================================================" << endl;
    cout << "Non-square board search" << endl;
    cout << "===================================================================" << endl;

    NNEvaluator* nnEval = startNNEval(modelFile,logger,"",7,17,0,true,false,false,true,false);
    SearchParams params;
    params.maxVisits = 100;
    Search* search = new Search(params, nnEval, &logger, "autoSearchRandSeed");
    Rules rules = Rules::getTrompTaylorish();
    TestSearchOptions opts;

    Board board = Board::parseBoard(7,17,R"%%(
.......
.......
..x.o..
.......
...o...
.......
.......
.......
.......
.......
...x...
.......
.......
..xx...
..oox..
....o..
.......
)%%");
    Player nextPla = P_BLACK;
    BoardHistory hist(board,nextPla,rules,0);

    search->setPosition(nextPla,board,hist);
    search->runWholeSearch(nextPla);

    cout << search->rootBoard << endl;

    PrintTreeOptions options;
    options = options.maxDepth(1);
    search->printTree(cout, search->rootNode, options, P_WHITE);

    delete search;
    delete nnEval;
    cout << endl;
  }


  {
    cout << "===================================================================" << endl;
    cout << "Visualize dirichlet noise" << endl;
    cout << "===================================================================" << endl;

    SearchParams params;
    params.rootNoiseEnabled = true;
    Rand rand("noiseVisualize");

    auto run = [&](int xSize, int ySize) {
      Board board(xSize,ySize);
      int nnXLen = 19;
      int nnYLen = 19;
      float sum = 0.0;
      int counter = 0;

      float origPolicyProbs[NNPos::MAX_NN_POLICY_SIZE];
      float policyProbs[NNPos::MAX_NN_POLICY_SIZE];
      std::fill(policyProbs,policyProbs+NNPos::MAX_NN_POLICY_SIZE,-1.0f);
      {
        for(int y = 0; y<board.y_size; y++) {
          for(int x = 0; x<board.x_size; x++) {
            int pos = NNPos::xyToPos(x,y,nnXLen);
            policyProbs[pos] = (float)pow(0.9,counter++);
            sum += policyProbs[pos];
          }
        }
        int pos = NNPos::locToPos(Board::PASS_LOC,board.x_size,nnXLen,nnYLen);
        policyProbs[pos] = (float)pow(0.9,counter++);
        sum += policyProbs[pos];

        for(int i = 0; i<NNPos::MAX_NN_POLICY_SIZE; i++) {
          if(policyProbs[i] >= 0.0)
            policyProbs[i] /= sum;
        }
      }

      std::copy(policyProbs,policyProbs+NNPos::MAX_NN_POLICY_SIZE,origPolicyProbs);
      Search::addDirichletNoise(params, rand, NNPos::MAX_NN_POLICY_SIZE, policyProbs);

      {
        for(int y = 0; y<board.y_size; y++) {
          for(int x = 0; x<board.x_size; x++) {
            int pos = NNPos::xyToPos(x,y,nnXLen);
            cout << Global::strprintf("%+6.2f ", 100.0*(policyProbs[pos] - origPolicyProbs[pos]));
          }
          cout << endl;
        }
        int pos = NNPos::locToPos(Board::PASS_LOC,board.x_size,nnXLen,nnYLen);
        cout << Global::strprintf("%+6.2f ", 100.0*(policyProbs[pos] - origPolicyProbs[pos]));
        cout << endl;
      }
    };

    run(19,19);
    run(11,7);
  }

  {
    cout << "===================================================================" << endl;
    cout << "Search tolerates moving past game end" << endl;
    cout << "===================================================================" << endl;

    NNEvaluator* nnEval = startNNEval(modelFile,logger,"",7,7,0,true,false,false,true,false);
    SearchParams params;
    params.maxVisits = 200;
    Search* search = new Search(params, nnEval, &logger, "autoSearchRandSeed");
    Search* search2 = new Search(params, nnEval, &logger, "autoSearchRandSeed");
    Search* search3 = new Search(params, nnEval, &logger, "autoSearchRandSeed");
    Rules rules = Rules::getTrompTaylorish();
    TestSearchOptions opts;
    PrintTreeOptions options;
    options = options.maxDepth(1);

    Board board = Board::parseBoard(7,7,R"%%(
.x.xo.o
xxxoooo
xxxxoo.
x.xo.oo
xxxoooo
xxxxooo
.xxxooo
)%%");
    Player nextPla = P_WHITE;
    BoardHistory hist(board,nextPla,rules,0);

    search->setPosition(nextPla,board,hist);
    search2->setPosition(nextPla,board,hist);
    search3->setPosition(nextPla,board,hist);

    search->makeMove(Location::ofString("C7",board),nextPla);
    search2->makeMove(Location::ofString("C7",board),nextPla);
    hist.makeBoardMoveAssumeLegal(board,Location::ofString("C7",board),nextPla,NULL);
    nextPla = getOpp(nextPla);
    search3->setPosition(nextPla,board,hist);

    search->makeMove(Location::ofString("pass",board),nextPla);
    search2->makeMove(Location::ofString("pass",board),nextPla);
    hist.makeBoardMoveAssumeLegal(board,Location::ofString("pass",board),nextPla,NULL);
    nextPla = getOpp(nextPla);
    search3->setPosition(nextPla,board,hist);
    board.checkConsistency();

    search2->runWholeSearch(nextPla);

    search->makeMove(Location::ofString("pass",board),nextPla);
    search2->makeMove(Location::ofString("pass",board),nextPla);
    board.checkConsistency();
    hist.makeBoardMoveAssumeLegal(board,Location::ofString("pass",board),nextPla,NULL);
    nextPla = getOpp(nextPla);
    search3->setPosition(nextPla,board,hist);

    testAssert(hist.isGameFinished);

    search->runWholeSearch(nextPla);
    search2->runWholeSearch(nextPla);
    search3->runWholeSearch(nextPla);

    hist.printDebugInfo(cout,board);
    cout << "Search made move after gameover" << endl;
    search->printTree(cout, search->rootNode, options, P_WHITE);
    cout << "Search made move (carrying tree over) after gameover" << endl;
    search2->printTree(cout, search2->rootNode, options, P_WHITE);
    cout << "Position was set after gameover" << endl;
    search3->printTree(cout, search3->rootNode, options, P_WHITE);

    cout << "Recapturing ko after two passes and supposed game over (violates superko)" << endl;
    search->makeMove(Location::ofString("D7",board),nextPla);
    search2->makeMove(Location::ofString("D7",board),nextPla);
    hist.makeBoardMoveAssumeLegal(board,Location::ofString("D7",board),nextPla,NULL);
    nextPla = getOpp(nextPla);
    search3->setPosition(nextPla,board,hist);

    search->runWholeSearch(nextPla);
    search2->runWholeSearch(nextPla);
    search3->runWholeSearch(nextPla);

    hist.printDebugInfo(cout,board);
    cout << "Search made move" << endl;
    search->printTree(cout, search->rootNode, options, P_WHITE);
    cout << "Search made move (carrying tree over)" << endl;
    search2->printTree(cout, search2->rootNode, options, P_WHITE);
    cout << "Position was set" << endl;
    search3->printTree(cout, search3->rootNode, options, P_WHITE);

    delete search;
    delete search2;
    delete search3;
    delete nnEval;
    cout << endl;
  }

  {
    cout << "===================================================================" << endl;
    cout << "Integrity of value bias, mem safety and updates" << endl;
    cout << "===================================================================" << endl;

    NNEvaluator* nnEval = startNNEval(modelFile,logger,"",7,7,0,true,false,false,true,false);
    SearchParams params;
    params.maxVisits = 500;
    params.subtreeValueBiasFactor = 0.5;
    params.chosenMoveTemperature = 0;
    Search* search = new Search(params, nnEval, &logger, "autoSearchRandSeed");
    Rules rules = Rules::getTrompTaylorish();
    Board board = Board::parseBoard(7,7,R"%%(
x.xxxx.
xxxooxx
xxxxox.
xxx.oxx
ooxoooo
o.oo.oo
.oooooo
)%%");
    Player nextPla = P_BLACK;
    BoardHistory hist(board,nextPla,rules,0);

    PrintTreeOptions options;
    options = options.maxDepth(1);

    search->setPosition(nextPla,board,hist);

    search->runWholeSearch(nextPla);
    cout << search->rootBoard << endl;
    search->printTree(cout, search->rootNode, options, P_WHITE);

    cout << "Making move" << endl;
    search->makeMove(search->getChosenMoveLoc(),nextPla);
    cout << search->rootBoard << endl;
    search->printTree(cout, search->rootNode, options, P_WHITE);

    cout << "Searching again" << endl;
    search->runWholeSearch(nextPla);
    cout << search->rootBoard << endl;
    search->printTree(cout, search->rootNode, options, P_WHITE);

    cout << "Making move" << endl;
    search->makeMove(search->getChosenMoveLoc(),nextPla);
    cout << search->rootBoard << endl;
    search->printTree(cout, search->rootNode, options, P_WHITE);

    cout << "Searching again" << endl;
    search->runWholeSearch(nextPla);
    cout << search->rootBoard << endl;
    search->printTree(cout, search->rootNode, options, P_WHITE);

    cout << "Making move" << endl;
    search->makeMove(search->getChosenMoveLoc(),nextPla);
    cout << search->rootBoard << endl;
    search->printTree(cout, search->rootNode, options, P_WHITE);


    delete search;
    delete nnEval;
    cout << endl;
  }

  {
    cout << "===================================================================" << endl;
    cout << "Value bias with ko" << endl;
    cout << "===================================================================" << endl;

    NNEvaluator* nnEval = startNNEval(modelFile,logger,"seeeed",14,14,0,true,false,false,true,false);
    SearchParams params;
    params.maxVisits = 2000;
    params.subtreeValueBiasFactor = 0.8;
    params.chosenMoveTemperature = 0;
    Search* search = new Search(params, nnEval, &logger, "seeeed");
    Rules rules = Rules::parseRules("japanese");
    Board board = Board::parseBoard(14,14,R"%%(
.oo.ox.xxxxxxx
o.oooxxxxxxxxx
xooxxxxxxxxxxx
.xxxxxxxxxxxxx
xxxxxxxooooxxx
xxxxxxxo.ox.x.
xxxxxxoooooxxx
xxxxxxo.oo.oxx
xxxxxxooooooxx
oxxxxxxxxxxxx.
.oooxxxxxxxxxx
oo.oxxxxxxxxoo
.ooooooooooo..
ooooo.oooooooo
)%%");
    Player nextPla = P_BLACK;
    BoardHistory hist(board,nextPla,rules,0);

    PrintTreeOptions options;
    options = options.maxDepth(1);

    search->setPosition(nextPla,board,hist);

    search->runWholeSearch(nextPla);
    cout << search->rootBoard << endl;
    search->printTree(cout, search->rootNode, options, P_WHITE);

    cout << "Making move" << endl;
    search->makeMove(search->getChosenMoveLoc(),nextPla);
    cout << search->rootBoard << endl;
    search->printTree(cout, search->rootNode, options, P_WHITE);

    cout << "Searching again" << endl;
    search->runWholeSearch(nextPla);
    cout << search->rootBoard << endl;
    search->printTree(cout, search->rootNode, options, P_WHITE);

    cout << "Making move" << endl;
    search->makeMove(search->getChosenMoveLoc(),nextPla);
    cout << search->rootBoard << endl;
    search->printTree(cout, search->rootNode, options, P_WHITE);

    cout << "Searching again" << endl;
    search->runWholeSearch(nextPla);
    cout << search->rootBoard << endl;
    search->printTree(cout, search->rootNode, options, P_WHITE);

    cout << "Making move" << endl;
    search->makeMove(search->getChosenMoveLoc(),nextPla);
    cout << search->rootBoard << endl;
    search->printTree(cout, search->rootNode, options, P_WHITE);


    delete search;
    delete nnEval;
    cout << endl;
  }

  {
    cout << "===================================================================" << endl;
    cout << "Analysis json" << endl;
    cout << "===================================================================" << endl;

    NNEvaluator* nnEval = startNNEval(modelFile,logger,"",9,9,0,true,false,false,true,false);
    SearchParams params;
    params.maxVisits = 10;
    params.subtreeValueBiasFactor = 0.5;
    params.chosenMoveTemperature = 0;
    Search* search = new Search(params, nnEval, &logger, "autoSearchRandSeed");
    search->setAlwaysIncludeOwnerMap(true);
    Rules rules = Rules::getTrompTaylorish();
    Board board = Board::parseBoard(7,7,R"%%(
.......
.......
.......
.......
.......
.......
.......
)%%");
    Player nextPla = P_BLACK;
    BoardHistory hist(board,nextPla,rules,0);

    PrintTreeOptions options;
    options = options.maxDepth(1);

    search->setPosition(nextPla,board,hist);
    search->runWholeSearch(nextPla);
    search->printTree(cout, search->rootNode, options, P_WHITE);
    nlohmann::json json;
    Player perspective = P_WHITE;
    int analysisPVLen = 2;
    bool preventEncore = true;
    bool includePolicy = true;
    bool includeOwnership = true;
    bool includeOwnershipStdev = false;
    bool includeMovesOwnership = false;
    bool includeMovesOwnershipStdev = false;
    bool includePVVisits = true;
    bool includeNoResultValue = false;
    bool suc = search->getAnalysisJson(
      perspective, analysisPVLen, preventEncore,
      includePolicy, includeOwnership, includeOwnershipStdev, includeMovesOwnership, includeMovesOwnershipStdev, includePVVisits,
      includeNoResultValue,
      json
    );
    testAssert(suc);
    cout << json << endl;

    delete search;
    delete nnEval;
  }

  {
    cout << "===================================================================" << endl;
    cout << "Analysis json with moves ownership and stdev" << endl;
    cout << "===================================================================" << endl;

    NNEvaluator* nnEval = startNNEval(modelFile,logger,"movesown",9,9,0,true,false,false,true,false);
    SearchParams params;
    params.maxVisits = 4;
    params.subtreeValueBiasFactor = 0.5;
    params.chosenMoveTemperature = 0;
    Search* search = new Search(params, nnEval, &logger, "autoSearchRandSeed");
    search->setAlwaysIncludeOwnerMap(true);
    Rules rules = Rules::getTrompTaylorish();
    Board board = Board::parseBoard(7,7,R"%%(
.......
.......
.......
.......
.......
.......
.......
)%%");
    Player nextPla = P_BLACK;
    BoardHistory hist(board,nextPla,rules,0);

    PrintTreeOptions options;
    options = options.maxDepth(1);

    search->setPosition(nextPla,board,hist);
    search->runWholeSearch(nextPla);
    search->printTree(cout, search->rootNode, options, P_WHITE);
    nlohmann::json json;
    Player perspective = P_WHITE;
    int analysisPVLen = 2;
    bool preventEncore = true;
    bool includePolicy = false;
    bool includeOwnership = true;
    bool includeOwnershipStdev = true;
    bool includeMovesOwnership = true;
    bool includeMovesOwnershipStdev = true;
    bool includePVVisits = false;
    bool includeNoResultValue = false;
    bool suc = search->getAnalysisJson(
      perspective, analysisPVLen, preventEncore,
      includePolicy, includeOwnership, includeOwnershipStdev, includeMovesOwnership, includeMovesOwnershipStdev, includePVVisits,
      includeNoResultValue,
      json
    );
    testAssert(suc);
    cout << json << endl;

    delete search;
    delete nnEval;
  }

  {
    cout << "===================================================================" << endl;
    cout << "Analysis json with moves ownership and stdev and symmetry" << endl;
    cout << "===================================================================" << endl;

    NNEvaluator* nnEval = startNNEval(modelFile,logger,"movesown",9,9,0,true,false,false,true,false);
    SearchParams params;
    params.maxVisits = 4;
    params.subtreeValueBiasFactor = 0.5;
    params.chosenMoveTemperature = 0;
    params.rootSymmetryPruning = true;
    Search* search = new Search(params, nnEval, &logger, "autoSearchRandSeed");
    search->setAlwaysIncludeOwnerMap(true);
    Rules rules = Rules::getTrompTaylorish();
    Board board = Board::parseBoard(7,7,R"%%(
.......
.......
.......
.......
.......
.......
.......
)%%");
    Player nextPla = P_BLACK;
    BoardHistory hist(board,nextPla,rules,0);

    PrintTreeOptions options;
    options = options.maxDepth(1);

    search->setPosition(nextPla,board,hist);
    search->runWholeSearch(nextPla);
    search->printTree(cout, search->rootNode, options, P_WHITE);
    nlohmann::json json;
    Player perspective = P_WHITE;
    int analysisPVLen = 2;
    bool preventEncore = true;
    bool includePolicy = false;
    bool includeOwnership = true;
    bool includeOwnershipStdev = true;
    bool includeMovesOwnership = true;
    bool includeMovesOwnershipStdev = true;
    bool includePVVisits = false;
    bool includeNoResultValue = false;
    bool suc = search->getAnalysisJson(
      perspective, analysisPVLen, preventEncore,
      includePolicy, includeOwnership, includeOwnershipStdev, includeMovesOwnership, includeMovesOwnershipStdev, includePVVisits,
      includeNoResultValue,
      json
    );
    testAssert(suc);
    cout << json << endl;

    delete search;
    delete nnEval;
  }

  {
    cout << "===================================================================" << endl;
    cout << "Analysis json 2" << endl;
    cout << "===================================================================" << endl;

    NNEvaluator* nnEval = startNNEval(modelFile,logger,"",9,9,0,true,false,false,true,false);
    SearchParams params;
    params.maxVisits = 10;
    params.subtreeValueBiasFactor = 0.5;
    params.chosenMoveTemperature = 0;
    Search* search = new Search(params, nnEval, &logger, "autoSearchRandSeed");
    search->setAlwaysIncludeOwnerMap(false);
    Rules rules = Rules::getTrompTaylorish();
    Board board = Board::parseBoard(9,6,R"%%(
.........
ooooooooo
oooxxxooo
..xxxxx..
xxx...xxx
xxxxxxxxx
)%%");
    Player nextPla = P_BLACK;
    BoardHistory hist(board,nextPla,rules,0);

    PrintTreeOptions options;
    options = options.maxDepth(1);

    search->setPosition(nextPla,board,hist);
    search->runWholeSearch(nextPla);
    search->printTree(cout, search->rootNode, options, P_WHITE);
    nlohmann::json json;
    Player perspective = P_WHITE;
    int analysisPVLen = 2;
    bool preventEncore = true;
    bool includePolicy = true;
    bool includeOwnership = false;
    bool includeOwnershipStdev = false;
    bool includeMovesOwnership = false;
    bool includeMovesOwnershipStdev = false;
    bool includePVVisits = false;
    bool includeNoResultValue = false;
    bool suc = search->getAnalysisJson(
      perspective, analysisPVLen, preventEncore,
      includePolicy, includeOwnership, includeOwnershipStdev, includeMovesOwnership, includeMovesOwnershipStdev, includePVVisits,
      includeNoResultValue,
      json
    );
    testAssert(suc);
    cout << json << endl;

    delete search;
    delete nnEval;
  }


  {
    cout << "===================================================================" << endl;
    cout << "Search results at 0, 1, 2 visits, and at terminal position" << endl;
    cout << "===================================================================" << endl;

    auto printResults = [](const Search* search, bool allowDirectPolicyMoves) {
      ReportedSearchValues values;
      cout << "getRootVisits " << search->getRootVisits() << endl;
      bool suc = search->getRootValues(values);
      cout << "getRootValues success: " << suc << endl;
      if(suc)
        cout << values.visits << " " << values.weight << " " << values.winLossValue << endl;

      suc = search->getPrunedRootValues(values);
      cout << "getPrunedRootValues success: " << suc << endl;
      if(suc)
        cout << values.visits << " " << values.weight << " " << values.winLossValue << endl;

      const SearchNode* node = search->getRootNode();
      if(node != NULL && (node = search->getChildForMove(node, Board::PASS_LOC)) != NULL)
      {
        suc = search->getNodeValues(node, values);
        cout << "getNodeValues for pass child success: " << suc << endl;
        if(suc)
          cout << values.visits << " " << values.weight << " " << values.winLossValue << endl;

        suc = search->getPrunedNodeValues(node, values);
        cout << "getPrunedNodeValues for pass child success: " << suc << endl;
        if(suc)
          cout << values.visits << " " << values.weight << " " << values.winLossValue << endl;
      }

      vector<double> playSelectionValues;
      vector<Loc> locs; // not used
      if(allowDirectPolicyMoves)
        suc = search->getPlaySelectionValues(locs,playSelectionValues,NULL,1.0);
      else {
        if(search->rootNode == NULL)
          suc = false;
        else
          suc = search->getPlaySelectionValues(*(search->rootNode),locs,playSelectionValues,NULL,1.0,allowDirectPolicyMoves);
      }
      cout << "getPlaySelectionValues success: " << suc << endl;
      if(suc) {
        for(size_t i = 0; i<playSelectionValues.size(); i++) {
          cout << Location::toString(locs[i],search->getRootBoard()) << " " << playSelectionValues[i] << endl;
        }
      }
      nlohmann::json json;
      Player perspective = P_WHITE;
      int analysisPVLen = 2;
      bool preventEncore = true;
      bool includePolicy = true;
      bool includeOwnership = false;
      bool includeOwnershipStdev = false;
      bool includeMovesOwnership = false;
      bool includeMovesOwnershipStdev = false;
      bool includePVVisits = true;
      bool includeNoResultValue = false;
      suc = search->getAnalysisJson(
        perspective, analysisPVLen, preventEncore,
        includePolicy, includeOwnership, includeOwnershipStdev, includeMovesOwnership, includeMovesOwnershipStdev, includePVVisits,
        includeNoResultValue,
        json
      );
      cout << "getAnalysisJson success: " << suc << endl;
      cout << json << endl;
    };

    NNEvaluator* nnEval = startNNEval(modelFile,logger,"",9,9,0,true,false,false,true,false);
    Rules rules = Rules::getTrompTaylorish();
    Board board = Board::parseBoard(7,7,R"%%(
.......
.......
..O....
....O..
..X.X..
.......
.......
)%%");
    Player nextPla = P_BLACK;
    BoardHistory hist(board,nextPla,rules,0);

    PrintTreeOptions options;
    options = options.maxDepth(1);

    {
      SearchParams params;
      params.maxVisits = 1;
      Search* search = new Search(params, nnEval, &logger, "autoSearchRandSeed");
      search->setPosition(nextPla,board,hist);
      search->beginSearch(false);

      cout << "Testing 0 visits allowDirectPolicyMoves false..." << endl;
      bool allowDirectPolicyMoves = false;
      printResults(search,allowDirectPolicyMoves);
      delete search;
    }
    {
      SearchParams params;
      params.maxVisits = 1;
      Search* search = new Search(params, nnEval, &logger, "autoSearchRandSeed");
      search->setPosition(nextPla,board,hist);
      search->beginSearch(false);

      cout << "Testing 0 visits allowDirectPolicyMoves true..." << endl;
      bool allowDirectPolicyMoves = true;
      printResults(search,allowDirectPolicyMoves);
      delete search;
    }
    {
      SearchParams params;
      params.maxVisits = 1;
      Search* search = new Search(params, nnEval, &logger, "autoSearchRandSeed");
      search->setPosition(nextPla,board,hist);
      search->runWholeSearch(nextPla);

      cout << "Testing 1 visits allowDirectPolicyMoves false..." << endl;
      bool allowDirectPolicyMoves = false;
      printResults(search,allowDirectPolicyMoves);
      delete search;
    }
    {
      SearchParams params;
      params.maxVisits = 1;
      Search* search = new Search(params, nnEval, &logger, "autoSearchRandSeed");
      search->setPosition(nextPla,board,hist);
      search->runWholeSearch(nextPla);

      cout << "Testing 1 visits allowDirectPolicyMoves true..." << endl;
      bool allowDirectPolicyMoves = true;
      printResults(search,allowDirectPolicyMoves);
      delete search;
    }
    {
      SearchParams params;
      params.maxVisits = 2;
      Search* search = new Search(params, nnEval, &logger, "autoSearchRandSeed");
      search->setPosition(nextPla,board,hist);
      search->runWholeSearch(nextPla);

      cout << "Testing 2 visits allowDirectPolicyMoves false..." << endl;
      bool allowDirectPolicyMoves = false;
      printResults(search,allowDirectPolicyMoves);
      delete search;
    }
    {
      SearchParams params;
      params.maxVisits = 2;
      Search* search = new Search(params, nnEval, &logger, "autoSearchRandSeed");
      search->setPosition(nextPla,board,hist);
      search->runWholeSearch(nextPla);

      cout << "Testing 2 visits allowDirectPolicyMoves true..." << endl;
      bool allowDirectPolicyMoves = true;
      printResults(search,allowDirectPolicyMoves);
      delete search;
    }
    {
      SearchParams params;
      params.maxVisits = 2;
      Search* search = new Search(params, nnEval, &logger, "autoSearchRandSeed");
      search->setPosition(nextPla,board,hist);
      search->makeMove(Board::PASS_LOC,P_BLACK);
      search->makeMove(Board::PASS_LOC,P_WHITE);
      search->runWholeSearch(nextPla);

      cout << "Testing 2 visits terminal position allowDirectPolicyMoves false..." << endl;
      bool allowDirectPolicyMoves = false;
      printResults(search,allowDirectPolicyMoves);
      delete search;
    }
    {
      SearchParams params;
      params.maxVisits = 2;
      Search* search = new Search(params, nnEval, &logger, "autoSearchRandSeed");
      search->setPosition(nextPla,board,hist);
      search->makeMove(Board::PASS_LOC,P_BLACK);
      search->makeMove(Board::PASS_LOC,P_WHITE);
      search->runWholeSearch(nextPla);

      cout << "Testing 2 visits terminal position allowDirectPolicyMoves true..." << endl;
      bool allowDirectPolicyMoves = true;
      printResults(search,allowDirectPolicyMoves);
      delete search;
    }
    {
      SearchParams params;
      params.maxVisits = 1000;
      Search* search = new Search(params, nnEval, &logger, "autoSearchRandSeed");
      search->setPosition(nextPla,board,hist);
      search->makeMove(Board::PASS_LOC,P_BLACK);
      search->setRootHintLoc(Board::PASS_LOC);
      search->runWholeSearch(P_WHITE);

      cout << "Testing 1000 visits just before terminal position allowDirectPolicyMoves false..." << endl;
      bool allowDirectPolicyMoves = false;
      printResults(search,allowDirectPolicyMoves);

      cout << "Testing 1000 visits just before terminal position, then playing the pass and having tree reuse. allowDirectPolicyMoves false..." << endl;
      search->makeMove(Board::PASS_LOC,P_WHITE);
      printResults(search,allowDirectPolicyMoves);

      delete search;
    }

    delete nnEval;
  }


  {
    cout << "===================================================================" << endl;
    cout << "Testing coherence of search tree recursive walking" << endl;
    cout << "===================================================================" << endl;

    NNEvaluator* nnEval = startNNEval(modelFile,logger,"",NNPos::MAX_BOARD_LEN,NNPos::MAX_BOARD_LEN,0,true,false,false,true,false);
    SearchParams params;
    params.maxVisits = 1000;
    params.dynamicScoreUtilityFactor = 3.0;
    Search* search = new Search(params, nnEval, &logger, "autoSearchRandSeed");
    Rules rules = Rules::getTrompTaylorish();
    TestSearchOptions opts;

    Board board = Board::parseBoard(7,7,R"%%(
x.xx.x.
xx.x.xx
xxx..xx
...ooo.
xxxo.o.
ooooooo
.o.oo.x
)%%");
    Player nextPla = P_BLACK;
    BoardHistory hist(board,nextPla,rules,0);

    {
      //--------------------------------------
      cout << "First perform a basic search." << endl;

      search->setPosition(nextPla,board,hist);
      search->runWholeSearch(nextPla);

      ConstSearchNodeChildrenReference children = search->rootNode->getChildren();
      int childrenCapacity = children.getCapacity();
      testAssert(childrenCapacity > 1);

      //In theory nothing requires this, but it would be kind of crazy if this were false
      testAssert(children.iterateAndCountChildren() > 1);
      testAssert(children[1].getIfAllocated() != NULL);

      Loc locToDescend = children[1].getMoveLoc();

      PrintTreeOptions options;
      options = options.maxDepth(1);
      cout << search->rootBoard << endl;
      search->printTree(cout, search->rootNode, options, P_WHITE);
      search->printTree(cout, search->rootNode, options.onlyBranch(board,Location::toString(locToDescend,board)), P_WHITE);

      cout << endl;

      //Enumerate the tree and make sure every node is indeed hit exactly once in postorder.
      TestSearchCommon::verifyTreePostOrder(search,-1);

      //--------------------------------------
      cout << "Next, make a move, and with no search, print the tree." << endl;

      search->makeMove(locToDescend,nextPla);
      nextPla = getOpp(nextPla);

      cout << search->rootBoard << endl;
      search->printTree(cout, search->rootNode, options, P_WHITE);
      cout << endl;

      //Enumerate the tree and make sure every node is indeed hit exactly once in postorder.
      TestSearchCommon::verifyTreePostOrder(search,-1);

      //--------------------------------------
      cout << "Begin search but make no additional playouts, print the tree." << endl;
      const bool pondering = false;
      search->beginSearch(pondering);
      search->printTree(cout, search->rootNode, options, P_WHITE);
      cout << endl;

      //Enumerate the tree and make sure every node is indeed hit exactly once in postorder.
      TestSearchCommon::verifyTreePostOrder(search,-1);
    }

    delete search;
    delete nnEval;
    cout << endl;
  }

  {
    cout << "===================================================================" << endl;
    cout << "Testing avoiding of all or almost all moves" << endl;
    cout << "===================================================================" << endl;

    SearchParams params;
    params.maxVisits = 100;
    Rules rules = Rules::getTrompTaylorish();
    TestSearchOptions opts;

    Board board = Board::parseBoard(9,5,R"%%(
xx..x..xx
xxxxxxxxx
....oxoxo
ooooooooo
oo..o..oo
)%%");

    {
      NNEvaluator* nnEval = startNNEval(modelFile,logger,"",9,5,0,true,false,false,true,false);
      Search* search = new Search(params, nnEval, &logger, "autoSearchRandSeed1234");
      cout << "Avoid all but 2 moves for both players, including passing" << endl;
      vector<int> avoidMoveUntilByLoc(Board::MAX_ARR_SIZE);
      for(int y = 0; y < board.y_size; y++) {
        for(int x = 0; x < board.x_size; x++) {
          Loc loc = Location::getLoc(x,y,board.x_size);
          if(x == 0 || x == 1) {
            avoidMoveUntilByLoc[loc] = 0;
          }
          else {
            avoidMoveUntilByLoc[loc] = 3;
          }
        }
      }
      avoidMoveUntilByLoc[Board::PASS_LOC] = 3;

      Player nextPla = P_WHITE;
      BoardHistory hist(board,nextPla,rules,0);

      search->setPosition(nextPla,board,hist);
      search->setAvoidMoveUntilByLoc(avoidMoveUntilByLoc,avoidMoveUntilByLoc);
      search->runWholeSearch(nextPla);

      cout << search->rootBoard << endl;
      PrintTreeOptions options;
      options = options.maxDepth(4);
      search->printTree(cout, search->rootNode, options, P_WHITE);

      nlohmann::json json;
      Player perspective = P_WHITE;
      int analysisPVLen = 2;
      bool preventEncore = true;
      bool includePolicy = false;
      bool includeOwnership = false;
      bool includeOwnershipStdev = false;
      bool includeMovesOwnership = false;
      bool includeMovesOwnershipStdev = false;
      bool includePVVisits = false;
      bool includeNoResultValue = false;
      bool suc = search->getAnalysisJson(
        perspective, analysisPVLen, preventEncore,
        includePolicy, includeOwnership, includeOwnershipStdev, includeMovesOwnership, includeMovesOwnershipStdev, includePVVisits,
        includeNoResultValue,
        json
      );
      testAssert(suc);
      cout << json << endl;

      //--------------------------------------
      cout << "Next, make a move, and with no search, print the tree." << endl;

      Loc locToDescend = Location::ofString("B3",search->rootBoard);
      search->makeMove(locToDescend,nextPla);
      nextPla = getOpp(nextPla);

      cout << search->rootBoard << endl;
      options = options.maxDepth(1);
      search->printTree(cout, search->rootNode, options, P_WHITE);
      cout << endl;

      //--------------------------------------
      cout << "Then continue the search to complete 100 visits." << endl;

      search->runWholeSearch(nextPla);
      options = options.maxDepth(1);
      search->printTree(cout, search->rootNode, options, P_WHITE);
      cout << endl;
      delete search;
      delete nnEval;
    }

    {
      NNEvaluator* nnEval = startNNEval(modelFile,logger,"",9,5,0,true,false,false,true,false);
      Search* search = new Search(params, nnEval, &logger, "autoSearchRandSeed1235");
      cout << "Avoid all moves for both players, including passing" << endl;
      vector<int> avoidMoveUntilByLoc(Board::MAX_ARR_SIZE);
      for(int y = 0; y < board.y_size; y++) {
        for(int x = 0; x < board.x_size; x++) {
          Loc loc = Location::getLoc(x,y,board.x_size);
          avoidMoveUntilByLoc[loc] = 3;
        }
      }
      avoidMoveUntilByLoc[Board::PASS_LOC] = 3;

      Player nextPla = P_WHITE;
      BoardHistory hist(board,nextPla,rules,0);

      search->setPosition(nextPla,board,hist);
      search->setAvoidMoveUntilByLoc(avoidMoveUntilByLoc,avoidMoveUntilByLoc);
      search->runWholeSearch(nextPla);

      cout << search->rootBoard << endl;
      PrintTreeOptions options;
      options = options.maxDepth(4);
      search->printTree(cout, search->rootNode, options, P_WHITE);

      nlohmann::json json;
      Player perspective = P_WHITE;
      int analysisPVLen = 2;
      bool preventEncore = true;
      bool includePolicy = false;
      bool includeOwnership = false;
      bool includeOwnershipStdev = false;
      bool includeMovesOwnership = false;
      bool includeMovesOwnershipStdev = false;
      bool includePVVisits = false;
      bool includeNoResultValue = false;
      bool suc = search->getAnalysisJson(
        perspective, analysisPVLen, preventEncore,
        includePolicy, includeOwnership, includeOwnershipStdev, includeMovesOwnership, includeMovesOwnershipStdev, includePVVisits,
        includeNoResultValue,
        json
      );
      testAssert(suc);
      cout << json << endl;
      delete search;
      delete nnEval;
    }

    {
      NNEvaluator* nnEval = startNNEval(modelFile,logger,"",9,5,0,true,false,false,true,false);
      Search* search = new Search(params, nnEval, &logger, "autoSearchRandSeed1236");
      cout << "Avoid all moves for black, including passing" << endl;
      vector<int> avoidMoveUntilByLoc(Board::MAX_ARR_SIZE);
      for(int y = 0; y < board.y_size; y++) {
        for(int x = 0; x < board.x_size; x++) {
          Loc loc = Location::getLoc(x,y,board.x_size);
          avoidMoveUntilByLoc[loc] = 10;
        }
      }
      avoidMoveUntilByLoc[Board::PASS_LOC] = 10;

      Player nextPla = P_WHITE;
      BoardHistory hist(board,nextPla,rules,0);

      search->setPosition(nextPla,board,hist);
      search->setAvoidMoveUntilByLoc(avoidMoveUntilByLoc,vector<int>());
      search->runWholeSearch(nextPla);

      cout << search->rootBoard << endl;
      PrintTreeOptions options;
      options = options.maxDepth(4);
      search->printTree(cout, search->rootNode, options, P_WHITE);

      nlohmann::json json;
      Player perspective = P_WHITE;
      int analysisPVLen = 2;
      bool preventEncore = true;
      bool includePolicy = false;
      bool includeOwnership = false;
      bool includeOwnershipStdev = false;
      bool includeMovesOwnership = false;
      bool includeMovesOwnershipStdev = false;
      bool includePVVisits = false;
      bool includeNoResultValue = false;
      bool suc = search->getAnalysisJson(
        perspective, analysisPVLen, preventEncore,
        includePolicy, includeOwnership, includeOwnershipStdev, includeMovesOwnership, includeMovesOwnershipStdev, includePVVisits,
        includeNoResultValue,
        json
      );
      testAssert(suc);
      cout << json << endl;
      delete search;
      delete nnEval;
    }
    cout << endl;
  }

  {
    cout << "===================================================================" << endl;
    cout << "Graph search, opening" << endl;
    cout << "===================================================================" << endl;

    NNEvaluator* nnEval = startNNEval(modelFile,logger,"",7,7,0,true,false,false,true,false);
    SearchParams params;
    params.maxVisits = 1000;
    params.subtreeValueBiasFactor = 0.5;
    params.useGraphSearch = true;
    params.chosenMoveTemperature = 0;
    Search* search = new Search(params, nnEval, &logger, "autoSearchRandSeed");
    Rules rules = Rules::getTrompTaylorish();
    Board board = Board::parseBoard(7,7,R"%%(
.......
.......
...o...
..ox...
..x....
.......
.......
)%%");
    Player nextPla = P_BLACK;
    BoardHistory hist(board,nextPla,rules,0);

    PrintTreeOptions options;
    options = options.maxDepth(1);

    search->setPosition(nextPla,board,hist);

    search->runWholeSearch(nextPla);
    cout << search->rootBoard << endl;
    search->printTree(cout, search->rootNode, options, P_WHITE);

    cout << "Making move" << endl;
    search->makeMove(search->getChosenMoveLoc(),nextPla);
    cout << search->rootBoard << endl;
    search->printTree(cout, search->rootNode, options, P_WHITE);

    cout << "Searching again" << endl;
    search->runWholeSearch(nextPla);
    cout << search->rootBoard << endl;
    search->printTree(cout, search->rootNode, options, P_WHITE);
    TestSearchCommon::verifyTreePostOrder(search,-1);

    cout << "Making move" << endl;
    search->makeMove(search->getChosenMoveLoc(),nextPla);
    cout << search->rootBoard << endl;
    search->printTree(cout, search->rootNode, options, P_WHITE);

    cout << "Searching again" << endl;
    search->runWholeSearch(nextPla);
    cout << search->rootBoard << endl;
    search->printTree(cout, search->rootNode, options, P_WHITE);

    cout << "Making move" << endl;
    search->makeMove(search->getChosenMoveLoc(),nextPla);
    cout << search->rootBoard << endl;
    search->printTree(cout, search->rootNode, options, P_WHITE);

    TestSearchCommon::verifyTreePostOrder(search,-1);

    delete search;
    delete nnEval;
    cout << endl;
  }

  {
    cout << "===================================================================" << endl;
    cout << "Graph search, 7x7 big fight" << endl;
    cout << "===================================================================" << endl;

    NNEvaluator* nnEval = startNNEval(modelFile,logger,"",7,7,0,true,false,false,true,false);
    SearchParams params;
    params.maxVisits = 1000;
    params.subtreeValueBiasFactor = 0.5;
    params.useGraphSearch = true;
    params.chosenMoveTemperature = 0;
    Search* search = new Search(params, nnEval, &logger, "autoSearchRandSeed");
    Rules rules = Rules::getTrompTaylorish();
    Board board = Board::parseBoard(7,7,R"%%(
.....o.
...oxox
..ooox.
.xoxxx.
.xxo.x.
..xooox
.......
)%%");
    Player nextPla = P_WHITE;
    BoardHistory hist(board,nextPla,rules,0);

    PrintTreeOptions options;
    options = options.maxDepth(1);

    search->setPosition(nextPla,board,hist);

    search->runWholeSearch(nextPla);
    cout << search->rootBoard << endl;
    search->printTree(cout, search->rootNode, options, P_WHITE);

    cout << "Making move" << endl;
    search->makeMove(search->getChosenMoveLoc(),nextPla);
    cout << search->rootBoard << endl;
    search->printTree(cout, search->rootNode, options, P_WHITE);

    cout << "Searching again" << endl;
    search->runWholeSearch(nextPla);
    cout << search->rootBoard << endl;
    search->printTree(cout, search->rootNode, options, P_WHITE);
    TestSearchCommon::verifyTreePostOrder(search,-1);

    cout << "Making move" << endl;
    search->makeMove(search->getChosenMoveLoc(),nextPla);
    cout << search->rootBoard << endl;
    search->printTree(cout, search->rootNode, options, P_WHITE);

    cout << "Searching again" << endl;
    search->runWholeSearch(nextPla);
    cout << search->rootBoard << endl;
    search->printTree(cout, search->rootNode, options, P_WHITE);

    cout << "Making move" << endl;
    search->makeMove(search->getChosenMoveLoc(),nextPla);
    cout << search->rootBoard << endl;
    search->printTree(cout, search->rootNode, options, P_WHITE);

    TestSearchCommon::verifyTreePostOrder(search,-1);

    delete search;
    delete nnEval;
    cout << endl;
  }

  {
    cout << "===================================================================" << endl;
    cout << "Graph search, 7x7 endgame kos" << endl;
    cout << "===================================================================" << endl;

    NNEvaluator* nnEval = startNNEval(modelFile,logger,"",7,7,0,true,false,false,true,false);
    SearchParams params;
    params.maxVisits = 1000;
    params.subtreeValueBiasFactor = 0.5;
    params.useGraphSearch = true;
    params.chosenMoveTemperature = 0;
    Search* search = new Search(params, nnEval, &logger, "autoSearchRandSeed");
    Rules rules = Rules::parseRules("japanese");
    Board board = Board::parseBoard(7,7,R"%%(
.o.x.x.
o.oxoxo
xoxxxo.
xx.xooo
xxxxox.
oxooox.
.ooox.x
)%%");
    Player nextPla = P_BLACK;
    BoardHistory hist(board,nextPla,rules,0);

    PrintTreeOptions options;
    options = options.maxDepth(1);

    search->setPosition(nextPla,board,hist);

    search->runWholeSearch(nextPla);
    cout << search->rootBoard << endl;
    search->printTree(cout, search->rootNode, options, P_WHITE);

    cout << "Making move" << endl;
    search->makeMove(search->getChosenMoveLoc(),nextPla);
    cout << search->rootBoard << endl;
    search->printTree(cout, search->rootNode, options, P_WHITE);

    cout << "Searching again" << endl;
    search->runWholeSearch(nextPla);
    cout << search->rootBoard << endl;
    search->printTree(cout, search->rootNode, options, P_WHITE);
    TestSearchCommon::verifyTreePostOrder(search,-1);

    cout << "Making move" << endl;
    search->makeMove(search->getChosenMoveLoc(),nextPla);
    cout << search->rootBoard << endl;
    search->printTree(cout, search->rootNode, options, P_WHITE);

    cout << "Searching again" << endl;
    search->runWholeSearch(nextPla);
    cout << search->rootBoard << endl;
    search->printTree(cout, search->rootNode, options, P_WHITE);

    cout << "Making move" << endl;
    search->makeMove(search->getChosenMoveLoc(),nextPla);
    cout << search->rootBoard << endl;
    search->printTree(cout, search->rootNode, options, P_WHITE);

    TestSearchCommon::verifyTreePostOrder(search,-1);

    delete search;
    delete nnEval;
    cout << endl;
  }

  {
    cout << "===================================================================" << endl;
    cout << "FPU parent weight by visited policy false" << endl;
    cout << "===================================================================" << endl;

    NNEvaluator* nnEval = startNNEval(modelFile,logger,"seeed",7,7,0,true,false,false,true,false);
    SearchParams params = SearchParams::forTestsV1();
    params.maxVisits = 1000;
    params.fpuParentWeightByVisitedPolicy = false;
    params.fpuParentWeightByVisitedPolicyPow = 1.0;

    Search* search = new Search(params, nnEval, &logger, "autoSearchRandSeed");
    Rules rules = Rules::parseRules("japanese");
    Board board = Board::parseBoard(7,7,R"%%(
.o.x.x.
o.oxoxo
xoxxxo.
xx.xooo
xxxxox.
oxooox.
.ooox.x
)%%");
    Player nextPla = P_BLACK;
    BoardHistory hist(board,nextPla,rules,0);

    PrintTreeOptions options;
    options = options.maxDepth(1);

    search->setPosition(nextPla,board,hist);

    search->runWholeSearch(nextPla);
    cout << search->rootBoard << endl;
    search->printTree(cout, search->rootNode, options, P_WHITE);

    delete search;
    delete nnEval;
    cout << endl;
  }

  {
    cout << "===================================================================" << endl;
    cout << "FPU parent weight by visited policy 1.0" << endl;
    cout << "===================================================================" << endl;

    NNEvaluator* nnEval = startNNEval(modelFile,logger,"seeed",7,7,0,true,false,false,true,false);
    SearchParams params = SearchParams::forTestsV1();
    params.maxVisits = 1000;
    params.fpuParentWeightByVisitedPolicy = true;
    params.fpuParentWeightByVisitedPolicyPow = 1.0;

    Search* search = new Search(params, nnEval, &logger, "autoSearchRandSeed");
    Rules rules = Rules::parseRules("japanese");
    Board board = Board::parseBoard(7,7,R"%%(
.o.x.x.
o.oxoxo
xoxxxo.
xx.xooo
xxxxox.
oxooox.
.ooox.x
)%%");
    Player nextPla = P_BLACK;
    BoardHistory hist(board,nextPla,rules,0);

    PrintTreeOptions options;
    options = options.maxDepth(1);

    search->setPosition(nextPla,board,hist);

    search->runWholeSearch(nextPla);
    cout << search->rootBoard << endl;
    search->printTree(cout, search->rootNode, options, P_WHITE);

    delete search;
    delete nnEval;
    cout << endl;
  }

  {
    cout << "===================================================================" << endl;
    cout << "FPU parent weight by visited policy 2.5" << endl;
    cout << "===================================================================" << endl;

    NNEvaluator* nnEval = startNNEval(modelFile,logger,"seeed",7,7,0,true,false,false,true,false);
    SearchParams params = SearchParams::forTestsV1();
    params.maxVisits = 1000;
    params.fpuParentWeightByVisitedPolicy = true;
    params.fpuParentWeightByVisitedPolicyPow = 2.5;

    Search* search = new Search(params, nnEval, &logger, "autoSearchRandSeed");
    Rules rules = Rules::parseRules("japanese");
    Board board = Board::parseBoard(7,7,R"%%(
.o.x.x.
o.oxoxo
xoxxxo.
xx.xooo
xxxxox.
oxooox.
.ooox.x
)%%");
    Player nextPla = P_BLACK;
    BoardHistory hist(board,nextPla,rules,0);

    PrintTreeOptions options;
    options = options.maxDepth(1);

    search->setPosition(nextPla,board,hist);

    search->runWholeSearch(nextPla);
    cout << search->rootBoard << endl;
    search->printTree(cout, search->rootNode, options, P_WHITE);

    delete search;
    delete nnEval;
    cout << endl;
  }

  {
    cout << "===================================================================" << endl;
    cout << "FPU parent weight by visited policy 0.5" << endl;
    cout << "===================================================================" << endl;

    NNEvaluator* nnEval = startNNEval(modelFile,logger,"seeed",7,7,0,true,false,false,true,false);
    SearchParams params = SearchParams::forTestsV1();
    params.maxVisits = 1000;
    params.fpuParentWeightByVisitedPolicy = true;
    params.fpuParentWeightByVisitedPolicyPow = 0.5;

    Search* search = new Search(params, nnEval, &logger, "autoSearchRandSeed");
    Rules rules = Rules::parseRules("japanese");
    Board board = Board::parseBoard(7,7,R"%%(
.o.x.x.
o.oxoxo
xoxxxo.
xx.xooo
xxxxox.
oxooox.
.ooox.x
)%%");
    Player nextPla = P_BLACK;
    BoardHistory hist(board,nextPla,rules,0);

    PrintTreeOptions options;
    options = options.maxDepth(1);

    search->setPosition(nextPla,board,hist);

    search->runWholeSearch(nextPla);
    cout << search->rootBoard << endl;
    search->printTree(cout, search->rootNode, options, P_WHITE);

    delete search;
    delete nnEval;
    cout << endl;
  }

  {
    cout << "===================================================================" << endl;
    cout << "Policy optimism with tree reuse" << endl;
    cout << "===================================================================" << endl;

    NNEvaluator* nnEval = startNNEval(modelFile,logger,"policyoptimismtreereuse",7,7,0,true,false,false,true,false);
    SearchParams params = SearchParams::forTestsV2();
    params.maxVisits = 100;
    params.rootPolicyOptimism = 0.43;
    params.policyOptimism = 0.71;
    SearchParams paramsLowVisits = params;
    paramsLowVisits.maxVisits = 8;

    Search* search = new Search(params, nnEval, &logger, "autoSearchRandSeeeeeed");
    Rules rules = Rules::parseRules("japanese");
    Board board = Board::parseBoard(5,5,R"%%(
.o.o.
ooooo
xxoxx
.xxx.
x.x.x
)%%");
    Player nextPla = P_BLACK;
    BoardHistory hist(board,nextPla,rules,0);
    PrintTreeOptions options;
    options = options.maxDepth(1);

    search->setPosition(nextPla,board,hist);

    cout << "Root position" << endl;
    search->runWholeSearch(nextPla);
    cout << search->rootBoard << endl;
    search->printTree(cout, search->rootNode, options, P_WHITE);
    search->rootNode->getNNOutput()->debugPrint(cout,search->rootBoard);
    testAssert(abs(search->rootNode->getNNOutput()->policyOptimismUsed - 0.43) < 0.00001);

    cout << "Make move and print" << endl;
    search->makeMove(search->getChosenMoveLoc(),nextPla);
    nextPla = getOpp(nextPla);
    search->printTree(cout, search->rootNode, options, P_WHITE);
    search->rootNode->getNNOutput()->debugPrint(cout,search->rootBoard);
    testAssert(abs(search->rootNode->getNNOutput()->policyOptimismUsed - 0.71) < 0.00001);

    cout << "Do search again" << endl;
    search->runWholeSearch(nextPla);
    cout << search->rootBoard << endl;
    search->printTree(cout, search->rootNode, options, P_WHITE);
    search->rootNode->getNNOutput()->debugPrint(cout,search->rootBoard);
    testAssert(abs(search->rootNode->getNNOutput()->policyOptimismUsed - 0.43) < 0.00001);

    cout << "Make move and print" << endl;
    search->makeMove(search->getChosenMoveLoc(),nextPla);
    nextPla = getOpp(nextPla);
    search->printTree(cout, search->rootNode, options, P_WHITE);
    search->rootNode->getNNOutput()->debugPrint(cout,search->rootBoard);
    testAssert(abs(search->rootNode->getNNOutput()->policyOptimismUsed - 0.71) < 0.00001);

    cout << "Do search again but with very low visits so the search already meets max visits" << endl;
    search->setParamsNoClearing(paramsLowVisits);
    search->runWholeSearch(nextPla);
    cout << search->rootBoard << endl;
    search->printTree(cout, search->rootNode, options, P_WHITE);
    search->rootNode->getNNOutput()->debugPrint(cout,search->rootBoard);
    testAssert(abs(search->rootNode->getNNOutput()->policyOptimismUsed - 0.43) < 0.00001);

    delete search;
    delete nnEval;
    cout << endl;
  }

  {
    cout << "===================================================================" << endl;
    cout << "Policy optimism with tree reuse, 0 root" << endl;
    cout << "===================================================================" << endl;

    NNEvaluator* nnEval = startNNEval(modelFile,logger,"policyoptimismtreereuse",7,7,0,true,false,false,true,false);
    SearchParams params = SearchParams::forTestsV2();
    params.maxVisits = 100;
    params.rootPolicyOptimism = 0.0;
    params.policyOptimism = 1.0;
    SearchParams paramsLowVisits = params;
    paramsLowVisits.maxVisits = 8;

    Search* search = new Search(params, nnEval, &logger, "autoSearchRandSeeeeeed");
    Rules rules = Rules::parseRules("japanese");
    Board board = Board::parseBoard(5,5,R"%%(
.o.o.
ooooo
xxoxx
.xxx.
x.x.x
)%%");
    Player nextPla = P_BLACK;
    BoardHistory hist(board,nextPla,rules,0);
    PrintTreeOptions options;
    options = options.maxDepth(1);

    search->setPosition(nextPla,board,hist);

    cout << "Root position" << endl;
    search->runWholeSearch(nextPla);
    cout << search->rootBoard << endl;
    search->printTree(cout, search->rootNode, options, P_WHITE);
    search->rootNode->getNNOutput()->debugPrint(cout,search->rootBoard);
    testAssert(abs(search->rootNode->getNNOutput()->policyOptimismUsed - 0.0) < 0.00001);

    cout << "Make move and print" << endl;
    search->makeMove(search->getChosenMoveLoc(),nextPla);
    nextPla = getOpp(nextPla);
    search->printTree(cout, search->rootNode, options, P_WHITE);
    search->rootNode->getNNOutput()->debugPrint(cout,search->rootBoard);
    testAssert(abs(search->rootNode->getNNOutput()->policyOptimismUsed - 1.0) < 0.00001);

    cout << "Do search again" << endl;
    search->runWholeSearch(nextPla);
    cout << search->rootBoard << endl;
    search->printTree(cout, search->rootNode, options, P_WHITE);
    search->rootNode->getNNOutput()->debugPrint(cout,search->rootBoard);
    testAssert(abs(search->rootNode->getNNOutput()->policyOptimismUsed - 0.0) < 0.00001);

    cout << "Make move and print" << endl;
    search->makeMove(search->getChosenMoveLoc(),nextPla);
    nextPla = getOpp(nextPla);
    search->printTree(cout, search->rootNode, options, P_WHITE);
    search->rootNode->getNNOutput()->debugPrint(cout,search->rootBoard);
    testAssert(abs(search->rootNode->getNNOutput()->policyOptimismUsed - 1.0) < 0.00001);

    cout << "Do search again but with very low visits so the search already meets max visits" << endl;
    search->setParamsNoClearing(paramsLowVisits);
    search->runWholeSearch(nextPla);
    cout << search->rootBoard << endl;
    search->printTree(cout, search->rootNode, options, P_WHITE);
    search->rootNode->getNNOutput()->debugPrint(cout,search->rootBoard);
    testAssert(abs(search->rootNode->getNNOutput()->policyOptimismUsed - 0.0) < 0.00001);

    delete search;
    delete nnEval;
    cout << endl;
  }

  {
    cout << "===================================================================" << endl;
    cout << "Zero node search" << endl;
    cout << "===================================================================" << endl;

    NNEvaluator* nnEval = startNNEval(modelFile,logger,"zeronodesearch",13,13,0,true,false,false,true,false);
    SearchParams params = SearchParams::forTestsV2();
    Search* search = new Search(params, nnEval, &logger, "autoSearchRandSeeeeeed");
    Rules rules = Rules::parseRules("japanese");
    Board board = Board::parseBoard(13,6,R"%%(
.............
.............
.....o.......
...x.........
.............
.............
)%%");
    Player nextPla = P_BLACK;
    BoardHistory hist(board,nextPla,rules,0);
    PrintTreeOptions options;
    options = options.maxDepth(1);

    search->setPosition(nextPla,board,hist);

    std::function<bool()> shouldStopEarly = []() noexcept {
      return true;
    };
    search->runWholeSearch(nextPla,&shouldStopEarly);
    cout << search->rootBoard << endl;
    search->printTree(cout, search->rootNode, options, P_WHITE);
    testAssert(search->rootNode->getNNOutput() == nullptr);
    cout << "Chosen move: " << Location::toString(search->getChosenMoveLoc(),board) << endl;

    delete search;
    delete nnEval;
    cout << endl;
  }

  {
    cout << "===================================================================" << endl;
    cout << "Chosen move probs" << endl;
    cout << "===================================================================" << endl;
    Rand rand;

    auto testRelativeProbs = [&](const std::vector<double>& relativeProbs, double temperature, double onlyBelowProb) {
      std::vector<double> processedRelProbsBuf(Board::MAX_ARR_SIZE);
      Search::chooseIndexWithTemperature(
        rand, relativeProbs.data(), (int)relativeProbs.size(), temperature, onlyBelowProb, processedRelProbsBuf.data()
      );
      cout << "Temperature " << temperature << " onlyBelowProb " << onlyBelowProb << endl;
      for (size_t i = 0; i < relativeProbs.size(); i++)
        cout << processedRelProbsBuf[i] << endl;
    };

    {
      const std::vector<double> relativeProbs({40,20,160,80,10,10});
      const double temperature = 1.0;
      const double onlyBelowProb = 1.0;
      testRelativeProbs(relativeProbs, temperature, onlyBelowProb);
    }
    {
      const std::vector<double> relativeProbs({40,20,160,80,10,10});
      const double temperature = 1.0;
      const double onlyBelowProb = 0.2;
      testRelativeProbs(relativeProbs, temperature, onlyBelowProb);
    }
    {
      const std::vector<double> relativeProbs({40,20,160,80,10,10});
      const double temperature = 0.5;
      const double onlyBelowProb = 1.0;
      testRelativeProbs(relativeProbs, temperature, onlyBelowProb);
    }
    {
      const std::vector<double> relativeProbs({40,20,160,80,10,10});
      const double temperature = 0.5;
      const double onlyBelowProb = 0.2;
      testRelativeProbs(relativeProbs, temperature, onlyBelowProb);
    }
    {
      const std::vector<double> relativeProbs({40,20,160,80,10,10});
      const double temperature = 2.0;
      const double onlyBelowProb = 0.2;
      testRelativeProbs(relativeProbs, temperature, onlyBelowProb);
    }
    {
      const std::vector<double> relativeProbs({40,20,160,80,10,10});
      const double temperature = 100000.0;
      const double onlyBelowProb = 0.2;
      testRelativeProbs(relativeProbs, temperature, onlyBelowProb);
    }
    {
      const std::vector<double> relativeProbs({40,20,160,80,10,10});
      const double temperature = 0.00001;
      const double onlyBelowProb = 0.2;
      testRelativeProbs(relativeProbs, temperature, onlyBelowProb);
    }
  }

  {
    cout << "===================================================================" << endl;
    cout << "Uninitialized search params" << endl;
    cout << "===================================================================" << endl;
    SearchParams params;
    params.printParams(cout);
    cout << endl;

    cout << "===================================================================" << endl;
    cout << "SearchParams forTestsV1" << endl;
    cout << "===================================================================" << endl;
    params = SearchParams::forTestsV1();
    params.printParams(cout);
    cout << endl;

    ConfigParser cfg;
    cfg.overrideKey("numSearchThreads","1");

    cout << "===================================================================" << endl;
    cout << "SearchParams for GTP" << endl;
    cout << "===================================================================" << endl;
    params = Setup::loadSingleParams(cfg, Setup::SETUP_FOR_GTP);
    params.printParams(cout);
    cout << endl;

    cout << "===================================================================" << endl;
    cout << "SearchParams for Analysis" << endl;
    cout << "===================================================================" << endl;
    params = Setup::loadSingleParams(cfg, Setup::SETUP_FOR_ANALYSIS);
    params.printParams(cout);
    cout << endl;

    cout << "===================================================================" << endl;
    cout << "SearchParams for Match" << endl;
    cout << "===================================================================" << endl;
    params = Setup::loadSingleParams(cfg, Setup::SETUP_FOR_MATCH);
    params.printParams(cout);
    cout << endl;

    cout << "===================================================================" << endl;
    cout << "SearchParams for Benchmark" << endl;
    cout << "===================================================================" << endl;
    params = Setup::loadSingleParams(cfg, Setup::SETUP_FOR_BENCHMARK);
    params.printParams(cout);
    cout << endl;

    cout << "===================================================================" << endl;
    cout << "SearchParams for Other" << endl;
    cout << "===================================================================" << endl;
    params = Setup::loadSingleParams(cfg, Setup::SETUP_FOR_OTHER);
    params.printParams(cout);
    cout << endl;

    cout << "===================================================================" << endl;
    cout << "SearchParams for Distributed" << endl;
    cout << "===================================================================" << endl;
    params = Setup::loadSingleParams(cfg, Setup::SETUP_FOR_DISTRIBUTED);
    params.printParams(cout);
    cout << endl;

  }

  {
    cout << "===================================================================" << endl;
    cout << "Eval cache keys depend on search params" << endl;
    cout << "===================================================================" << endl;

    NNEvaluator* nnEval = startNNEval(modelFile,logger,"evalcacheparams",7,7,0,true,false,false,true,false);

    SearchParams paramsA = SearchParams::forTestsV2(); //useGraphSearch = true
    paramsA.useEvalCache = true;
    paramsA.evalCacheMinVisits = 5;
    paramsA.maxVisits = 60;

    SearchParams paramsB = paramsA;
    paramsB.cpuctExploration = paramsA.cpuctExploration + 0.5;

    SearchParams paramsA2 = paramsA;

    testAssert(paramsA.getHash() == paramsA2.getHash());
    testAssert(paramsA.getHash() != paramsB.getHash());
    cout << "Equal params hash equal: " << (paramsA.getHash() == paramsA2.getHash()) << endl;
    cout << "Differing params hash differ: " << (paramsA.getHash() != paramsB.getHash()) << endl;

    Search* search = new Search(paramsA, nnEval, &logger, "evalcacheseed");
    Rules rules = Rules::parseRules("chinese");
    Board board = Board::parseBoard(7,7,R"%%(
.......
..x.o..
.......
..o.x..
.......
.......
.......
)%%");
    Player nextPla = P_BLACK;
    BoardHistory hist(board,nextPla,rules,0);

    // Search 1: params A.
    search->setPosition(nextPla,board,hist);
    search->runWholeSearch(nextPla);
    testAssert(search->evalCache != nullptr);
    const Hash128 rootGraphHash = search->rootGraphHash;
    const Hash128 keyA = rootGraphHash ^ paramsA.getHash();
    const Hash128 keyB = rootGraphHash ^ paramsB.getHash();
    std::shared_ptr<EvalCacheEntry> entryA = search->evalCache->find(keyA);
    testAssert(entryA != nullptr);
    testAssert(search->evalCache->find(keyB) == nullptr);
    cout << "After search A: A-slot populated, B-slot empty: "
         << (entryA != nullptr && search->evalCache->find(keyB) == nullptr) << endl;

    // Search 2: params B, same position. The eval cache persists across param changes.
    search->setParams(paramsB);
    search->setPosition(nextPla,board,hist);
    search->runWholeSearch(nextPla);
    testAssert(search->rootGraphHash == rootGraphHash);
    testAssert(keyA != keyB);
    testAssert(search->evalCache->find(keyB) != nullptr);
    testAssert(search->evalCache->find(keyA) == entryA);
    cout << "After search B: B-slot populated, A-slot unchanged, distinct slots: "
         << (search->evalCache->find(keyB) != nullptr
             && search->evalCache->find(keyA) == entryA
             && keyA != keyB) << endl;

    // Search 3: params identical to A. Must reuse the same key/slot as A.
    search->setParams(paramsA2);
    search->setPosition(nextPla,board,hist);
    search->runWholeSearch(nextPla);
    testAssert(search->rootGraphHash == rootGraphHash);
    const Hash128 keyA2 = rootGraphHash ^ paramsA2.getHash();
    testAssert(keyA2 == keyA);
    testAssert(search->evalCache->find(keyA2) != nullptr);
    cout << "After search A2 (identical to A): reuses A's slot: "
         << (keyA2 == keyA && search->evalCache->find(keyA2) != nullptr) << endl;

    delete search;
    delete nnEval;
    cout << endl;
  }

  {
    cout << "===================================================================" << endl;
    cout << "Board size distribution" << endl;
    cout << "===================================================================" << endl;
    ConfigParser cfg;
    cfg.overrideKey("koRules","SIMPLE");
    cfg.overrideKey("scoringRules","AREA");
    cfg.overrideKey("taxRules","SEKI");
    cfg.overrideKey("multiStoneSuicideLegals","false");
    cfg.overrideKey("hasButtons","false");
    cfg.overrideKey("bSizes","2,4,6,8");
    cfg.overrideKey("bSizeRelProbs","1,2,3,4");
    cfg.overrideKey("allowRectangleProb","0.3");
    cfg.overrideKey("komiAuto","true");
    GameInitializer gameInit(cfg, logger, "board size distribution random seed");

    std::map<std::pair<int,int>,int> boardSizeDistribution;
    for(int i = 0; i<100000; i++) {
      Board board;
      Player pla;
      BoardHistory hist;
      ExtraBlackAndKomi extraBlackAndKomi;
      OtherGameProperties otherGameProps;
      gameInit.createGame(board,pla,hist,extraBlackAndKomi,NULL,PlaySettings(),otherGameProps,NULL);
      boardSizeDistribution[std::make_pair(board.x_size,board.y_size)] += 1;
    }
    for(int x = 2; x<=8; x += 2) {
      for(int y = 2; y<=8; y += 2) {
        cout << x << "x" << y << " " << boardSizeDistribution[std::make_pair(x,y)] << endl;
      }
    }
  }

  NeuralNet::globalCleanup();
  cout << "Done" << endl;
}

namespace {

struct PersistentMCTSTestPosition {
  string sgfFile;
  int64_t turnIdx;
  Board board;
  Player nextPla;
  BoardHistory hist;
};

struct PersistentMCTSTestCase {
  string sgfFile;
  vector<PersistentMCTSTestPosition> positions;
  vector<int> rootSequence;
  vector<int> exportSteps;
};

struct PersistentMCTSRootSnapshot {
  int64_t visits;
  map<Loc,int64_t> edgeVisitsByLoc;
  ReportedSearchValues values;
  vector<double> ownership;
};

static string resolvePersistentMCTSTestPath(const string& path) {
  if(FileUtils::exists(path))
    return path;
  if(FileUtils::exists("cpp/" + path))
    return "cpp/" + path;
  if(FileUtils::exists("../" + path))
    return "../" + path;
  return path;
}

static SearchParams makePersistentMCTSTestParams(int64_t maxPlayouts) {
  SearchParams params;
  params.dynamicScoreUtilityFactor = 0.0;
  params.useGraphSearch = true;
  params.useEvalCache = false;
  params.useNoisePruning = false;
  params.useUncertainty = false;
  params.rootNoiseEnabled = false;
  params.rootSymmetryPruning = false;
  params.rootNumSymmetriesToSample = 1;
  params.rootDesiredPerChildVisitsCoeff = 0.0;
  params.rootPolicyOptimism = 0.0;
  params.policyOptimism = 0.0;
  params.wideRootNoise = 0.0;
  params.enablePassingHacks = false;
  params.enableMorePassingHacks = false;
  params.playoutDoublingAdvantage = 0.0;
  params.playoutDoublingAdvantagePla = C_EMPTY;
  params.subtreeValueBiasFactor = 0.0;
  params.numThreads = 1;
  params.minPlayoutsPerThread = 0.0;
  params.maxVisits = ((int64_t)1L << 50);
  params.maxPlayouts = maxPlayouts;
  params.maxTime = 1.0e20;
  params.searchFactorAfterOnePass = 1.0;
  params.searchFactorAfterTwoPass = 1.0;
  return params;
}

static bool persistentMCTSApproxEqual(double a, double b) {
  const double atol = 1e-4;
  const double rtol = 1e-3;
  return std::fabs(a-b) <= atol + rtol * std::max(std::fabs(a), std::fabs(b));
}

static void assertPersistentMCTSApprox(
  double actual,
  double expected,
  const string& label,
  const string& context
) {
  if(!persistentMCTSApproxEqual(actual,expected)) {
    cout << "Persistent MCTS mismatch at " << context << " for " << label
         << ": actual=" << actual << " expected=" << expected << endl;
    testAssert(false);
  }
}

static PersistentMCTSRootSnapshot capturePersistentMCTSRootSnapshot(Search* search) {
  PersistentMCTSRootSnapshot snapshot;
  snapshot.visits = search->getRootVisits();
  snapshot.values = search->getRootValuesRequireSuccess();
  snapshot.ownership = search->getAverageTreeOwnership();

  ConstSearchNodeChildrenReference children = search->rootNode->getChildren();
  int childrenCapacity = children.getCapacity();
  for(int i = 0; i<childrenCapacity; i++) {
    const SearchNode* child = children[i].getIfAllocated();
    if(child == NULL)
      break;
    Loc moveLoc = children[i].getMoveLoc();
    int64_t edgeVisits = children[i].getEdgeVisits();
    if(edgeVisits > 0)
      snapshot.edgeVisitsByLoc[moveLoc] = edgeVisits;
  }
  return snapshot;
}

static void comparePersistentMCTSRootSnapshots(
  const PersistentMCTSRootSnapshot& actual,
  const PersistentMCTSRootSnapshot& expected,
  const Board& board,
  const string& context
) {
  if(actual.visits != expected.visits) {
    cout << "Persistent MCTS visit mismatch at " << context
         << ": actual=" << actual.visits << " expected=" << expected.visits << endl;
    testAssert(false);
  }

  set<Loc> locs;
  for(const auto& entry: actual.edgeVisitsByLoc)
    locs.insert(entry.first);
  for(const auto& entry: expected.edgeVisitsByLoc)
    locs.insert(entry.first);
  double actualVisitDenom = std::max((int64_t)1, actual.visits);
  double expectedVisitDenom = std::max((int64_t)1, expected.visits);
  for(Loc loc: locs) {
    int64_t actualVisits = actual.edgeVisitsByLoc.count(loc) ? actual.edgeVisitsByLoc.at(loc) : 0;
    int64_t expectedVisits = expected.edgeVisitsByLoc.count(loc) ? expected.edgeVisitsByLoc.at(loc) : 0;
    double actualProp = (double)actualVisits / actualVisitDenom;
    double expectedProp = (double)expectedVisits / expectedVisitDenom;
    assertPersistentMCTSApprox(
      actualProp,
      expectedProp,
      string("edge distribution ") + Location::toString(loc,board),
      context
    );
  }

  assertPersistentMCTSApprox(actual.values.winValue, expected.values.winValue, "root winValue", context);
  assertPersistentMCTSApprox(actual.values.winLossValue, expected.values.winLossValue, "root winLossValue", context);
  assertPersistentMCTSApprox(actual.values.expectedScore, expected.values.expectedScore, "root expectedScore", context);
  assertPersistentMCTSApprox(actual.values.lead, expected.values.lead, "root lead", context);

  if(actual.ownership.size() != expected.ownership.size()) {
    cout << "Persistent MCTS ownership size mismatch at " << context
         << ": actual=" << actual.ownership.size() << " expected=" << expected.ownership.size() << endl;
    testAssert(false);
  }
  for(size_t i = 0; i<actual.ownership.size(); i++)
    assertPersistentMCTSApprox(actual.ownership[i], expected.ownership[i], "ownership", context);
}

static PersistentMCTSRootSnapshot runStandardPersistentMCTSReference(
  const SearchParams& baseParams,
  NNEvaluator* nnEval,
  Logger* logger,
  const PersistentMCTSTestPosition& pos,
  int64_t visits,
  int64_t numSearchesBegunBeforeRun
) {
  SearchParams params = baseParams;
  params.maxVisits = ((int64_t)1L << 50);
  params.maxPlayouts = visits;
  Search standard(params, nnEval, logger, "persistentMCTSSgfPersistentSeed");
  standard.setAlwaysIncludeOwnerMap(true);
  standard.numSearchesBegun = numSearchesBegunBeforeRun;
  standard.setPosition(pos.nextPla,pos.board,pos.hist);
  int64_t visitsRemaining = visits;
  while(visitsRemaining > 0) {
    int64_t chunk = std::min<int64_t>(64, visitsRemaining);
    standard.searchParams.maxPlayouts = chunk;
    standard.runWholeSearch(pos.nextPla);
    visitsRemaining -= chunk;
  }
  PersistentMCTSRootSnapshot snapshot = capturePersistentMCTSRootSnapshot(&standard);
  if(snapshot.visits != visits) {
    cout << "Standard MCTS reference did not hit requested visits for "
         << pos.sgfFile << " turn " << pos.turnIdx
         << ": requested=" << visits << " actual=" << snapshot.visits << endl;
    testAssert(false);
  }
  return snapshot;
}

static vector<string> collectPersistentMCTSSgfFiles() {
  vector<string> files;
  vector<string> fixedFiles = {
    "tests/data/sampletest/sampletest7x7.sgf",
    "tests/data/sampletest/sampletest9x9.sgf",
    "tests/data/humanslbigdiff.sgf",
    "tests/data/sampletest2/messy.sgf"
  };
  for(const string& path: fixedFiles) {
    string resolved = resolvePersistentMCTSTestPath(path);
    if(FileUtils::exists(resolved))
      files.push_back(resolved);
  }

  const char* includeExternalSgfs = std::getenv("KATAGO_PERSISTENT_MCTS_EXTERNAL_SGFS");
  if(includeExternalSgfs != NULL && string(includeExternalSgfs) == "1") {
    vector<string> dirs = {
      "../sgfs",
      "../../sgfs"
    };
    for(const string& dir: dirs) {
      if(FileUtils::exists(dir)) {
        vector<string> collected;
        FileHelpers::collectSgfsFromDirOrFile(dir,collected);
        files.insert(files.end(),collected.begin(),collected.end());
      }
    }
  }

  sort(files.begin(),files.end());
  files.erase(unique(files.begin(),files.end()),files.end());
  Rand rand("persistentMCTSSgfFileOrder");
  rand.shuffle(files);
  return files;
}

static bool makePersistentMCTSTestCase(
  const string& sgfFile,
  Rand& rand,
  PersistentMCTSTestCase& testCase
) {
  std::unique_ptr<CompactSgf> sgf;
  try {
    sgf = CompactSgf::loadFile(sgfFile);
  }
  catch(const StringError& e) {
    cout << "Skipping SGF for persistent MCTS test due to parse error: " << sgfFile << " " << e.what() << endl;
    return false;
  }
  if(sgf == nullptr || sgf->moves.size() <= 20)
    return false;
  if(sgf->xSize > NNPos::MAX_BOARD_LEN || sgf->ySize > NNPos::MAX_BOARD_LEN)
    return false;

  Rules rules = Rules::getTrompTaylorish();
  try {
    rules = sgf->getRulesOrFailAllowUnspecified(rules);
  }
  catch(const StringError& e) {
    cout << "Skipping SGF for persistent MCTS test due to rules error: " << sgfFile << " " << e.what() << endl;
    return false;
  }

  int64_t latestStart = (int64_t)sgf->moves.size() - 20;
  int64_t start = (int64_t)rand.nextUInt64((uint64_t)latestStart + 1);
  vector<uint32_t> offsets(20);
  for(uint32_t i = 0; i<offsets.size(); i++)
    offsets[i] = i;
  rand.shuffle(offsets);

  testCase.sgfFile = sgfFile;
  testCase.positions.clear();
  for(int i = 0; i<offsets.size() && testCase.positions.size() < 15; i++) {
    int64_t turnIdx = start + offsets[i];
    Board board;
    Player nextPla;
    BoardHistory hist;
    try {
      sgf->setupBoardAndHistTolerant(rules,board,nextPla,hist,turnIdx,false);
    }
    catch(const StringError& e) {
      continue;
    }
    if(hist.isGameFinished)
      continue;
    PersistentMCTSTestPosition pos;
    pos.sgfFile = sgfFile;
    pos.turnIdx = turnIdx;
    pos.board = board;
    pos.nextPla = nextPla;
    pos.hist = hist;
    testCase.positions.push_back(pos);
  }
  if(testCase.positions.size() < 15)
    return false;

  testCase.rootSequence.clear();
  vector<int> initialOrder(15);
  for(int i = 0; i<15; i++)
    initialOrder[i] = i;
  rand.shuffle(initialOrder);
  for(int i = 0; i<15; i++)
    testCase.rootSequence.push_back(initialOrder[i]);
  while(testCase.rootSequence.size() < 105)
    testCase.rootSequence.push_back((int)rand.nextUInt(15));

  vector<int> exportCandidates(105);
  for(int i = 0; i<105; i++)
    exportCandidates[i] = i;
  rand.shuffle(exportCandidates);
  testCase.exportSteps.assign(exportCandidates.begin(),exportCandidates.begin()+7);
  sort(testCase.exportSteps.begin(),testCase.exportSteps.end());
  return true;
}

static vector<PersistentMCTSTestCase> makePersistentMCTSTestCases(int numCases) {
  vector<PersistentMCTSTestCase> cases;
  vector<string> sgfFiles = collectPersistentMCTSSgfFiles();
  Rand rand("persistentMCTSSgfCaseSeed");
  for(const string& sgfFile: sgfFiles) {
    PersistentMCTSTestCase testCase;
    if(makePersistentMCTSTestCase(sgfFile,rand,testCase)) {
      cases.push_back(testCase);
      if(cases.size() >= (size_t)numCases)
        break;
    }
  }
  if(cases.size() < (size_t)numCases) {
    cout << "Only found " << cases.size() << " persistent MCTS SGF cases" << endl;
  }
  testAssert(cases.size() > 0);
  return cases;
}

static void runPersistentMCTSSgfCorrectnessTest(
  const SearchParams& baseParams,
  NNEvaluator* nnEval,
  Logger* logger
) {
  vector<PersistentMCTSTestCase> cases = makePersistentMCTSTestCases(2);
  for(size_t caseIdx = 0; caseIdx<cases.size(); caseIdx++) {
    const PersistentMCTSTestCase& testCase = cases[caseIdx];
    cout << "Persistent MCTS correctness SGF case " << caseIdx
         << ": " << testCase.sgfFile << endl;

    SearchParams params = baseParams;
    params.maxPlayouts = 64;
    Search persistent(params, nnEval, logger, "persistentMCTSSgfPersistentSeed");
    persistent.setAlwaysIncludeOwnerMap(true);
    persistent.setPersistentMCTSEnabled(true);

    size_t nextExportIdx = 0;
    for(int step = 0; step<testCase.rootSequence.size(); step++) {
      const PersistentMCTSTestPosition& pos = testCase.positions[testCase.rootSequence[step]];
      persistent.setPositionForMCTSPersistence(pos.nextPla,pos.board,pos.hist);
      persistent.searchParams.maxPlayouts = 64;
      int64_t numSearchesBegunBeforeRun = persistent.numSearchesBegun;
      persistent.runWholeSearch(pos.nextPla);

      PersistentMCTSRootSnapshot persistentSnapshot = capturePersistentMCTSRootSnapshot(&persistent);
      string context =
        testCase.sgfFile + " step " + Global::intToString(step) +
        " turn " + Global::int64ToString(pos.turnIdx);
      PersistentMCTSRootSnapshot standardSnapshot =
        runStandardPersistentMCTSReference(baseParams,nnEval,logger,pos,persistentSnapshot.visits,numSearchesBegunBeforeRun);
      comparePersistentMCTSRootSnapshots(persistentSnapshot,standardSnapshot,pos.board,context);

      if(nextExportIdx < testCase.exportSteps.size() && step == testCase.exportSteps[nextExportIdx]) {
        string exportPath =
          "/tmp/katago_persistent_mcts_sgf_" + Global::uint64ToString(caseIdx) +
          "_" + Global::intToString(step) + ".json";
        persistent.exportPersistentMCTS(exportPath);
        persistent.clearSearch();
        persistent.importPersistentMCTS(exportPath);
        PersistentMCTSRootSnapshot importedSnapshot = capturePersistentMCTSRootSnapshot(&persistent);
        comparePersistentMCTSRootSnapshots(importedSnapshot,persistentSnapshot,pos.board,context + " import");
        nextExportIdx++;
      }
    }
  }
}

static int64_t countPersistentMCTSExportNodes(const string& path) {
  ifstream in;
  FileUtils::open(in,path);
  nlohmann::json data;
  in >> data;
  if(!data.contains("nodes"))
    return 0;
  return (int64_t)data["nodes"].size();
}

static void runPersistentMCTSSgfStressTest(
  const SearchParams& baseParams,
  NNEvaluator* nnEval,
  Logger* logger
) {
  vector<PersistentMCTSTestCase> cases = makePersistentMCTSTestCases(1);
  const PersistentMCTSTestCase& testCase = cases[0];
  cout << "Persistent MCTS stress SGF case: " << testCase.sgfFile << endl;

  SearchParams params = baseParams;
  params.maxPlayouts = 1024;
  Search persistent(params, nnEval, logger, "persistentMCTSSgfStressSeed");
  persistent.setAlwaysIncludeOwnerMap(true);
  persistent.setPersistentMCTSEnabled(true);

  int64_t totalRequestedPlayouts = 0;
  size_t nextExportIdx = 0;
  for(int step = 0; step<testCase.rootSequence.size(); step++) {
    const PersistentMCTSTestPosition& pos = testCase.positions[testCase.rootSequence[step]];
    persistent.setPositionForMCTSPersistence(pos.nextPla,pos.board,pos.hist);
    persistent.searchParams.maxPlayouts = 1024;
    persistent.runWholeSearch(pos.nextPla);
    totalRequestedPlayouts += 1024;
    testAssert(persistent.getRootVisits() > 0);

    if(nextExportIdx < testCase.exportSteps.size() && step == testCase.exportSteps[nextExportIdx]) {
      string exportPath =
        "/tmp/katago_persistent_mcts_stress_" + Global::intToString(step) + ".json";
      PersistentMCTSRootSnapshot before = capturePersistentMCTSRootSnapshot(&persistent);
      persistent.exportPersistentMCTS(exportPath);
      persistent.clearSearch();
      persistent.importPersistentMCTS(exportPath);
      PersistentMCTSRootSnapshot after = capturePersistentMCTSRootSnapshot(&persistent);
      comparePersistentMCTSRootSnapshots(
        after,
        before,
        pos.board,
        testCase.sgfFile + " stress step " + Global::intToString(step)
      );
      nextExportIdx++;
    }
  }

  string finalExportPath = "/tmp/katago_persistent_mcts_stress_final.json";
  persistent.exportPersistentMCTS(finalExportPath);
  int64_t exportedNodes = countPersistentMCTSExportNodes(finalExportPath);
  int64_t reasonableNodeBound = totalRequestedPlayouts + (int64_t)testCase.positions.size() + 4096;
  cout << "Persistent MCTS stress nodes=" << exportedNodes
       << " requestedPlayouts=" << totalRequestedPlayouts
       << " bound=" << reasonableNodeBound << endl;
  testAssert(exportedNodes > 0);
  testAssert(exportedNodes <= reasonableNodeBound);
}

}

void Tests::runPersistentMCTSTests() {
  cout << "Running persistent MCTS tests" << endl;
  NeuralNet::globalInitialize();

  string modelFile = "/dev/null";
  const bool logToStdout = false;
  const bool logToStderr = false;
  const bool logTime = false;
  Logger logger(nullptr, logToStdout, logToStderr, logTime);

  NNEvaluator* nnEval = startNNEval(modelFile,logger,"persistent",NNPos::MAX_BOARD_LEN,NNPos::MAX_BOARD_LEN,0,true,false,false,true,false);

  SearchParams params = makePersistentMCTSTestParams(80);
  params.maxVisits = 1000000;
  params.maxPlayouts = 80;
  Search* search = new Search(params, nnEval, &logger, "persistentMCTSTestSeed");
  search->setPersistentMCTSEnabled(true);

  Rules rules = Rules::getTrompTaylorish();
  Board board = Board::parseBoard(7,7,R"%%(
.......
..x.o..
.......
...x...
.......
..o.x..
.......
)%%");
  Player nextPla = P_BLACK;
  BoardHistory hist(board,nextPla,rules,0);

  search->setPositionForMCTSPersistence(nextPla,board,hist);
  search->runWholeSearch(nextPla);

  auto collectRootEdges = [&]() {
    vector<pair<Loc,int64_t>> rootEdges;
    ConstSearchNodeChildrenReference children = search->rootNode->getChildren();
    int childrenCapacity = children.getCapacity();
    for(int i = 0; i<childrenCapacity; i++) {
      const SearchNode* child = children[i].getIfAllocated();
      if(child == NULL)
        break;
      Loc moveLoc = children[i].getMoveLoc();
      int64_t edgeVisits = children[i].getEdgeVisits();
      if(edgeVisits > 0)
        rootEdges.push_back(make_pair(moveLoc,edgeVisits));
    }
    return rootEdges;
  };
  auto getRootEdgeVisits = [&] (Loc loc) {
    ConstSearchNodeChildrenReference children = search->rootNode->getChildren();
    int childrenCapacity = children.getCapacity();
    for(int i = 0; i<childrenCapacity; i++) {
      const SearchNode* child = children[i].getIfAllocated();
      if(child == NULL)
        break;
      if(children[i].getMoveLoc() == loc)
        return children[i].getEdgeVisits();
    }
    return (int64_t)0;
  };
  auto assertRootEdgesMatch = [&] (const vector<pair<Loc,int64_t>>& expectedEdges) {
    for(const pair<Loc,int64_t>& expected: expectedEdges)
      testAssert(getRootEdgeVisits(expected.first) == expected.second);
  };

  vector<pair<Loc,int64_t>> rootEdgesBefore = collectRootEdges();
  Loc locToDescend = Board::NULL_LOC;
  int64_t locToDescendEdgeVisits = -1;
  for(const pair<Loc,int64_t>& edge: rootEdgesBefore) {
    if(edge.first != Board::PASS_LOC && edge.second > locToDescendEdgeVisits) {
      locToDescend = edge.first;
      locToDescendEdgeVisits = edge.second;
    }
  }
  testAssert(locToDescend != Board::NULL_LOC);
  testAssert(locToDescendEdgeVisits > 0);

  Board childBoard = board;
  BoardHistory childHist = hist;
  childHist.makeBoardMoveAssumeLegal(childBoard,locToDescend,nextPla,NULL);
  Player childPla = getOpp(nextPla);

  search->setPositionForMCTSPersistence(childPla,childBoard,childHist);
  int64_t inheritedChildRootVisits = search->getRootVisits();
  testAssert(inheritedChildRootVisits >= locToDescendEdgeVisits);

  search->searchParams.maxPlayouts = 200;
  search->runWholeSearch(childPla);
  testAssert(search->getRootVisits() > inheritedChildRootVisits);
  int64_t childRootVisitsAfterChildSearch = search->getRootVisits();

  search->setPositionForMCTSPersistence(nextPla,board,hist);
  assertRootEdgesMatch(rootEdgesBefore);

  search->searchParams.maxPlayouts = search->getRootVisits() + 240;
  search->runWholeSearch(nextPla);
  int64_t locToDescendEdgeVisitsAfterAncestorSearch = getRootEdgeVisits(locToDescend);
  testAssert(locToDescendEdgeVisitsAfterAncestorSearch > locToDescendEdgeVisits);

  search->setPositionForMCTSPersistence(childPla,childBoard,childHist);
  testAssert(search->getRootVisits() > childRootVisitsAfterChildSearch);

  search->setPositionForMCTSPersistence(nextPla,board,hist);
  vector<pair<Loc,int64_t>> rootEdgesBeforeExport = collectRootEdges();

  string exportPath = "/tmp/katago_persistent_mcts_test.json";
  search->exportPersistentMCTS(exportPath);
  search->clearSearch();
  search->importPersistentMCTS(exportPath);
  testAssert(search->getPersistentMCTSEnabled());
  testAssert(search->rootNode != NULL);
  assertRootEdgesMatch(rootEdgesBeforeExport);

  delete search;
  delete nnEval;

  NeuralNet::globalCleanup();
  cout << "Persistent MCTS tests passed" << endl;
}

void Tests::runPersistentMCTSStrictTests() {
  cout << "Running strict persistent MCTS tests" << endl;
  NeuralNet::globalInitialize();

  string modelFile = "/dev/null";
  const bool logToStdout = false;
  const bool logToStderr = false;
  const bool logTime = false;
  Logger logger(nullptr, logToStdout, logToStderr, logTime);

  NNEvaluator* nnEval = startNNEval(modelFile,logger,"persistent-strict",NNPos::MAX_BOARD_LEN,NNPos::MAX_BOARD_LEN,0,true,false,false,true,false);

  SearchParams sgfTestParams = makePersistentMCTSTestParams(64);
  runPersistentMCTSSgfCorrectnessTest(sgfTestParams, nnEval, &logger);
  runPersistentMCTSSgfStressTest(sgfTestParams, nnEval, &logger);

  delete nnEval;

  NeuralNet::globalCleanup();
  cout << "Strict persistent MCTS tests passed" << endl;
}
