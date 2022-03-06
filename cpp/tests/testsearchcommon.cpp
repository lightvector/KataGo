#include "../tests/testsearchcommon.h"

#include "../dataio/sgf.h"
#include "../search/searchnode.h"
#include "../tests/tests.h"

//------------------------
#include "../core/using.h"
//------------------------
using namespace TestCommon;

string TestSearchCommon::getSearchRandSeed() {
  static int seedCounter = 0;
  return string("testSearchSeed") + Global::intToString(seedCounter++);
}

TestSearchCommon::TestSearchOptions::TestSearchOptions()
  :numMovesInARow(1),
   printRootPolicy(false),
   printOwnership(false),
   printEndingScoreValueBonus(false),
   printPlaySelectionValues(false),
   noClearBot(false),
   noClearCache(false),
   printMore(false),
   printMoreMoreMore(false),
   printAfterBegun(false),
   ignorePosition(false),
   printPostOrderNodeCount(false)
{}

void TestSearchCommon::printPolicyValueOwnership(const Board& board, const NNResultBuf& buf) {
  cout << board << endl;
  cout << endl;
  buf.result->debugPrint(cout,board);
}

void TestSearchCommon::printBasicStuffAfterSearch(const Board& board, const BoardHistory& hist, const Search* search, PrintTreeOptions options) {
  Board::printBoard(cout, board, Board::NULL_LOC, &(hist.moveHistory));
  cout << "Root visits: " << search->getRootVisits() << "\n";
  cout << "New playouts: " << search->lastSearchNumPlayouts << "\n";
  cout << "NN rows: " << search->nnEvaluator->numRowsProcessed() << endl;
  cout << "NN batches: " << search->nnEvaluator->numBatchesProcessed() << endl;
  cout << "NN avg batch size: " << search->nnEvaluator->averageProcessedBatchSize() << endl;
  cout << "PV: ";
  search->printPV(cout, search->rootNode, 25);
  cout << "\n";
  cout << "Tree:\n";
  search->printTree(cout, search->rootNode, options, P_WHITE);
}

void TestSearchCommon::runBotOnPosition(AsyncBot* bot, Board board, Player nextPla, BoardHistory hist, TestSearchOptions opts) {

  if(!opts.ignorePosition)
    bot->setPosition(nextPla,board,hist);

  PrintTreeOptions options;
  options = options.maxDepth(1);
  if(opts.printMoreMoreMore)
    options = options.maxDepth(20);
  else if(opts.printMore)
    options = options.minVisitsPropToExpand(0.1).maxDepth(2);
  if(opts.printOwnership)
    bot->setAlwaysIncludeOwnerMap(true);

  for(int i = 0; i<opts.numMovesInARow; i++) {

    Loc move;
    if(opts.printAfterBegun) {
      cout << "Just after begun" << endl;
      std::function<void()> onSearchBegun = [&]() {
        const Search* search = bot->getSearch();
        search->printTree(cout, search->rootNode, options, P_WHITE);
      };
      move = bot->genMoveSynchronous(nextPla,TimeControls(),1.0,onSearchBegun);
    }
    else {
      move = bot->genMoveSynchronous(nextPla,TimeControls());
    }
    const Search* search = bot->getSearch();

    printBasicStuffAfterSearch(board,hist,search,options);

    if(opts.printRootPolicy) {
      search->printRootPolicyMap(cout);
    }
    if(opts.printOwnership) {
      std::tuple<std::vector<double>,std::vector<double>> ownershipAndStdev = search->getAverageAndStandardDeviationTreeOwnership();
      std::vector<double> ownership = std::get<0>(ownershipAndStdev);
      std::vector<double> ownershipStdev = std::get<1>(ownershipAndStdev);
      for(int y = 0; y<board.y_size; y++) {
        for(int x = 0; x<board.x_size; x++) {
          int pos = NNPos::xyToPos(x,y,search->nnXLen);
          cout << Global::strprintf("%6.1f ", ownership[pos]*100);
        }
        cout << endl;
      }
      cout << endl;
      for(int y = 0; y<board.y_size; y++) {
        for(int x = 0; x<board.x_size; x++) {
          int pos = NNPos::xyToPos(x,y,search->nnXLen);
          cout << Global::strprintf("%6.1f ", ownershipStdev[pos]*100);
        }
        cout << endl;
      }
      cout << endl;
    }
    if(opts.printEndingScoreValueBonus) {
      search->printRootOwnershipMap(cout, P_WHITE);
      search->printRootEndingScoreValueBonus(cout);
    }
    if(opts.printPlaySelectionValues) {
      cout << "Play selection values" << endl;
      double scaleMaxToAtLeast = 10.0;
      vector<Loc> locsBuf;
      vector<double> playSelectionValuesBuf;
      bool success = search->getPlaySelectionValues(locsBuf,playSelectionValuesBuf,scaleMaxToAtLeast);
      testAssert(success);
      for(int j = 0; j<locsBuf.size(); j++) {
        cout << Location::toString(locsBuf[j],board) << " " << playSelectionValuesBuf[j] << endl;
      }
    }

    if(opts.printPostOrderNodeCount)
      verifyTreePostOrder(bot->getSearchStopAndWait(),-1);

    if(i < opts.numMovesInARow-1) {
      bot->makeMove(move, nextPla);
      hist.makeBoardMoveAssumeLegal(board,move,nextPla,NULL);
      cout << "Just after move" << endl;
      search->printTree(cout, search->rootNode, options, P_WHITE);
      nextPla = getOpp(nextPla);

      if(opts.printPostOrderNodeCount)
        verifyTreePostOrder(bot->getSearchStopAndWait(),-1);
    }
  }

  const Search* search = bot->getSearch();
  if(!opts.noClearCache) {
    search->nnEvaluator->clearCache();
    search->nnEvaluator->clearStats();
  }
  if(!opts.noClearBot)
    bot->clearSearch();
}

void TestSearchCommon::runBotOnSgf(AsyncBot* bot, const string& sgfStr, const Rules& defaultRules, int turnIdx, float overrideKomi, TestSearchOptions opts) {
  CompactSgf* sgf = CompactSgf::parse(sgfStr);

  Board board;
  Player nextPla;
  BoardHistory hist;
  Rules initialRules = sgf->getRulesOrFailAllowUnspecified(defaultRules);
  sgf->setupBoardAndHistAssumeLegal(initialRules, board, nextPla, hist, turnIdx);
  hist.setKomi(overrideKomi);
  runBotOnPosition(bot,board,nextPla,hist,opts);
  delete sgf;
}

NNEvaluator* TestSearchCommon::startNNEval(
  const string& modelFile, Logger& logger, const string& seed, int nnXLen, int nnYLen,
  int defaultSymmetry, bool inputsUseNHWC, bool useNHWC, bool useFP16, bool debugSkipNeuralNet,
  bool requireExactNNLen
) {
  vector<int> gpuIdxByServerThread = {0};
  int maxBatchSize = 16;
  int nnCacheSizePowerOfTwo = 16;
  int nnMutexPoolSizePowerOfTwo = 12;
  int maxConcurrentEvals = 1024;
  //bool debugSkipNeuralNet = false;
  bool openCLReTunePerBoardSize = false;
  const string& modelName = modelFile;
  const string openCLTunerFile = "";
  const string homeDataDirOverride = "";
  int numNNServerThreadsPerModel = 1;
  bool nnRandomize = false;
  string nnRandSeed = "runSearchTestsRandSeed"+seed;

  if(defaultSymmetry == -1) {
    nnRandomize = true;
    defaultSymmetry = 0;
  }

  string expectedSha256 = "";
  NNEvaluator* nnEval = new NNEvaluator(
    modelName,
    modelFile,
    expectedSha256,
    &logger,
    maxBatchSize,
    maxConcurrentEvals,
    nnXLen,
    nnYLen,
    requireExactNNLen,
    inputsUseNHWC,
    nnCacheSizePowerOfTwo,
    nnMutexPoolSizePowerOfTwo,
    debugSkipNeuralNet,
    openCLTunerFile,
    homeDataDirOverride,
    openCLReTunePerBoardSize,
    useFP16 ? enabled_t::True : enabled_t::False,
    useNHWC ? enabled_t::True : enabled_t::False,
    numNNServerThreadsPerModel,
    gpuIdxByServerThread,
    nnRandSeed,
    nnRandomize,
    defaultSymmetry
  );

  nnEval->spawnServerThreads();

  //Hack to get more consistent ordering of log messages spawned by nnEval threads with other output.
  std::this_thread::sleep_for(std::chrono::duration<double>(0.1));
  return nnEval;
}


void TestSearchCommon::verifyTreePostOrder(Search* search, int onlyRequireAtLeast) {
  //Enumerate the tree and make sure every node is indeed hit exactly once in postorder.
  std::vector<SearchNode*> nodes = search-> enumerateTreePostOrder();
  std::map<const SearchNode*,size_t> idxOfNode;
  for(size_t i = 0; i<nodes.size(); i++) {
    SearchNode* node = nodes[i];
    testAssert(node != NULL);
    idxOfNode[node] = i;
  }
  for(size_t i = 0; i<nodes.size(); i++) {
    int childrenCapacity;
    const SearchChildPointer* children = nodes[i]->getChildren(childrenCapacity);
    for(int j = 0; j<childrenCapacity; j++) {
      const SearchNode* child = children[j].getIfAllocated();
      if(child == NULL)
        break;
      testAssert(contains(idxOfNode,child));
      testAssert(idxOfNode[child] < i);
    }
  }
  if(onlyRequireAtLeast > 0) {
    if(nodes.size() >= onlyRequireAtLeast)
      cout << "Post order okay: " << "yes" << endl;
    else {
      cout << "Post order got too few nodes " << nodes.size() << " " << onlyRequireAtLeast << endl;
      testAssert(false);
    }
  }
  else{
    cout << "Post order okay: " << nodes.size() << endl;
  }
}
