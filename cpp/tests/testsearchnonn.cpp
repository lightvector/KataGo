#include "../tests/tests.h"

#include <algorithm>
#include <iterator>
#include <iomanip>

#include "../core/fileutils.h"
#include "../dataio/sgf.h"
#include "../neuralnet/nninputs.h"
#include "../search/asyncbot.h"
#include "../program/playutils.h"
#include "../tests/testsearchcommon.h"

using namespace std;
using namespace TestCommon;
using namespace TestSearchCommon;


void Tests::runNNLessSearchTests() {
  cout << "Running neuralnetless search tests" << endl;
  NeuralNet::globalInitialize();

  //Placeholder, doesn't actually do anything since we have debugSkipNeuralNet = true
  string modelFile = "/dev/null";

  Logger logger;
  logger.setLogToStdout(false);
  logger.setLogTime(false);
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
      std::sort(moveLocsAndCountsSorted.begin(), moveLocsAndCountsSorted.end(), [](pair<Loc,int> a, pair<Loc,int> b) { return a.second > b.second; });

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
        search->rootHistory.moveHistory.push_back(Move(Board::NULL_LOC,P_BLACK));

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

      //In theory nothing requires this, but it would be kind of crazy if this were false
      testAssert(search->rootNode->iterateAndCountChildren() > 1);
      int childrenCapacity;
      const SearchChildPointer* children = search->rootNode->getChildren(childrenCapacity);
      testAssert(childrenCapacity > 1);
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
      int childrenCapacity;
      const SearchChildPointer* children = search->rootNode->getChildren(childrenCapacity);
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
      int childrenCapacity;
      const SearchChildPointer* children = search->rootNode->getChildren(childrenCapacity);
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
    TestSearchCommon::verifyTreePostOrder(search);

    //--------------------------------------
    cout << "Next, make a move, and with no search, print the tree." << endl;

    search->makeMove(locToDescend,nextPla);
    nextPla = getOpp(nextPla);

    cout << search->rootBoard << endl;
    search->printTree(cout, search->rootNode, options, P_WHITE);
    cout << endl;

    //Enumerate the tree and make sure every node is indeed hit exactly once in postorder.
    TestSearchCommon::verifyTreePostOrder(search);

    //--------------------------------------
    cout << "Begin search but make no additional playouts, print the tree." << endl;
    const bool pondering = false;
    search->beginSearch(pondering);
    search->printTree(cout, search->rootNode, options, P_WHITE);
    cout << endl;

    //Enumerate the tree and make sure every node is indeed hit exactly once in postorder.
    TestSearchCommon::verifyTreePostOrder(search);

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
    int ownershipMinVisits = 1;
    bool preventEncore = true;
    bool includePolicy = false;
    bool includeOwnership = false;
    bool includeOwnershipStdev = false;
    bool includeMovesOwnership = false;
    bool includeMovesOwnershipStdev = false;
    bool includePVVisits = false;
    bool suc = search->getAnalysisJson(
      perspective, analysisPVLen, ownershipMinVisits, preventEncore,
      includePolicy, includeOwnership, includeOwnershipStdev, includeMovesOwnership, includeMovesOwnershipStdev, includePVVisits,
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
    int ownershipMinVisits = 1;
    bool preventEncore = true;
    bool includePolicy = false;
    bool includeOwnership = false;
    bool includeOwnershipStdev = false;
    bool includeMovesOwnership = false;
    bool includeMovesOwnershipStdev = false;
    bool includePVVisits = false;
    bool suc = search->getAnalysisJson(
      perspective, analysisPVLen, ownershipMinVisits, preventEncore,
      includePolicy, includeOwnership, includeOwnershipStdev, includeMovesOwnership, includeMovesOwnershipStdev, includePVVisits,
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
    int ownershipMinVisits = 1;
    bool preventEncore = true;
    bool includePolicy = false;
    bool includeOwnership = false;
    bool includeOwnershipStdev = false;
    bool includeMovesOwnership = false;
    bool includeMovesOwnershipStdev = false;
    bool includePVVisits = false;
    bool suc = search->getAnalysisJson(
      perspective, analysisPVLen, ownershipMinVisits, preventEncore,
      includePolicy, includeOwnership, includeOwnershipStdev, includeMovesOwnership, includeMovesOwnershipStdev, includePVVisits,
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

    assert(hist.isGameFinished);

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
    int ownershipMinVisits = 1;
    bool preventEncore = true;
    bool includePolicy = true;
    bool includeOwnership = true;
    bool includeOwnershipStdev = false;
    bool includeMovesOwnership = false;
    bool includeMovesOwnershipStdev = false;
    bool includePVVisits = true;
    bool suc = search->getAnalysisJson(
      perspective, analysisPVLen, ownershipMinVisits, preventEncore,
      includePolicy, includeOwnership, includeOwnershipStdev, includeMovesOwnership, includeMovesOwnershipStdev, includePVVisits,
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
    int ownershipMinVisits = 1;
    bool preventEncore = true;
    bool includePolicy = true;
    bool includeOwnership = false;
    bool includeOwnershipStdev = false;
    bool includeMovesOwnership = false;
    bool includeMovesOwnershipStdev = false;
    bool includePVVisits = false;
    bool suc = search->getAnalysisJson(
      perspective, analysisPVLen, ownershipMinVisits, preventEncore,
      includePolicy, includeOwnership, includeOwnershipStdev, includeMovesOwnership, includeMovesOwnershipStdev, includePVVisits,
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

    auto printResults = [](Search* search, bool allowDirectPolicyMoves) {
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
      int ownershipMinVisits = 1;
      bool preventEncore = true;
      bool includePolicy = true;
      bool includeOwnership = false;
      bool includeOwnershipStdev = false;
      bool includeMovesOwnership = false;
      bool includeMovesOwnershipStdev = false;
      bool includePVVisits = true;
      suc = search->getAnalysisJson(
        perspective, analysisPVLen, ownershipMinVisits, preventEncore,
        includePolicy, includeOwnership, includeOwnershipStdev, includeMovesOwnership, includeMovesOwnershipStdev, includePVVisits,
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

      //In theory nothing requires this, but it would be kind of crazy if this were false
      testAssert(search->rootNode->iterateAndCountChildren() > 1);
      int childrenCapacity;
      const SearchChildPointer* children = search->rootNode->getChildren(childrenCapacity);
      testAssert(childrenCapacity > 1);
      testAssert(children[1].getIfAllocated() != NULL);
      Loc locToDescend = children[1].getMoveLoc();

      PrintTreeOptions options;
      options = options.maxDepth(1);
      cout << search->rootBoard << endl;
      search->printTree(cout, search->rootNode, options, P_WHITE);
      search->printTree(cout, search->rootNode, options.onlyBranch(board,Location::toString(locToDescend,board)), P_WHITE);

      cout << endl;

      //Enumerate the tree and make sure every node is indeed hit exactly once in postorder.
      TestSearchCommon::verifyTreePostOrder(search);

      //--------------------------------------
      cout << "Next, make a move, and with no search, print the tree." << endl;

      search->makeMove(locToDescend,nextPla);
      nextPla = getOpp(nextPla);

      cout << search->rootBoard << endl;
      search->printTree(cout, search->rootNode, options, P_WHITE);
      cout << endl;

      //Enumerate the tree and make sure every node is indeed hit exactly once in postorder.
      TestSearchCommon::verifyTreePostOrder(search);

      //--------------------------------------
      cout << "Begin search but make no additional playouts, print the tree." << endl;
      const bool pondering = false;
      search->beginSearch(pondering);
      search->printTree(cout, search->rootNode, options, P_WHITE);
      cout << endl;

      //Enumerate the tree and make sure every node is indeed hit exactly once in postorder.
      TestSearchCommon::verifyTreePostOrder(search);
    }

    delete search;
    delete nnEval;
    cout << endl;
  }

  {
    cout << "===================================================================" << endl;
    cout << "Testing avoiding of all or almost all moves" << endl;
    cout << "===================================================================" << endl;

    NNEvaluator* nnEval = startNNEval(modelFile,logger,"",9,5,0,true,false,false,true,false);
    SearchParams params;
    params.maxVisits = 100;
    Search* search = new Search(params, nnEval, &logger, "autoSearchRandSeed1234");
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
      int ownershipMinVisits = 1;
      bool preventEncore = true;
      bool includePolicy = false;
      bool includeOwnership = false;
      bool includeOwnershipStdev = false;
      bool includeMovesOwnership = false;
      bool includeMovesOwnershipStdev = false;
      bool includePVVisits = false;
      bool suc = search->getAnalysisJson(
        perspective, analysisPVLen, ownershipMinVisits, preventEncore,
        includePolicy, includeOwnership, includeOwnershipStdev, includeMovesOwnership, includeMovesOwnershipStdev, includePVVisits,
        json
      );
      testAssert(suc);
      cout << json << endl;

      //--------------------------------------
      cout << "Next, make a move, and with no search, print the tree." << endl;

      Loc locToDescend = Location::ofString("B2",search->rootBoard);
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
    }

    {
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
      int ownershipMinVisits = 1;
      bool preventEncore = true;
      bool includePolicy = false;
      bool includeOwnership = false;
      bool includeOwnershipStdev = false;
      bool includeMovesOwnership = false;
      bool includeMovesOwnershipStdev = false;
      bool includePVVisits = false;
      bool suc = search->getAnalysisJson(
        perspective, analysisPVLen, ownershipMinVisits, preventEncore,
        includePolicy, includeOwnership, includeOwnershipStdev, includeMovesOwnership, includeMovesOwnershipStdev, includePVVisits,
        json
      );
      testAssert(suc);
      cout << json << endl;
    }

    {
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
      int ownershipMinVisits = 1;
      bool preventEncore = true;
      bool includePolicy = false;
      bool includeOwnership = false;
      bool includeOwnershipStdev = false;
      bool includeMovesOwnership = false;
      bool includeMovesOwnershipStdev = false;
      bool includePVVisits = false;
      bool suc = search->getAnalysisJson(
        perspective, analysisPVLen, ownershipMinVisits, preventEncore,
        includePolicy, includeOwnership, includeOwnershipStdev, includeMovesOwnership, includeMovesOwnershipStdev, includePVVisits,
        json
      );
      testAssert(suc);
      cout << json << endl;
    }

    delete search;
    delete nnEval;
    cout << endl;
  }

  NeuralNet::globalCleanup();
  cout << "Done" << endl;
}


