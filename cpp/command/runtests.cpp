
#include <sstream>
#include "../core/global.h"
#include "../core/bsearch.h"
#include "../core/rand.h"
#include "../core/elo.h"
#include "../core/fancymath.h"
#include "../core/config_parser.h"
#include "../core/base64.h"
#include "../core/timer.h"
#include "../game/board.h"
#include "../game/rules.h"
#include "../game/boardhistory.h"
#include "../neuralnet/nninputs.h"
#include "../program/gtpconfig.h"
#include "../program/setup.h"
#include "../tests/tests.h"
#include "../tests/tinymodel.h"
#include "../command/commandline.h"
#include "../main.h"

using namespace std;

int MainCmds::runtests(int argc, const char* const* argv) {
  (void)argc;
  (void)argv;
  testAssert(sizeof(size_t) == 8);
  Board::initHash();
  ScoreValue::initTables();

  BSearch::runTests();
  Rand::runTests();
  FancyMath::runTests();
  ComputeElos::runTests();
  Base64::runTests();

  Tests::runBoardIOTests();
  Tests::runBoardBasicTests();

  Tests::runBoardAreaTests();

  Tests::runRulesTests();

  Tests::runBoardUndoTest();
  Tests::runBoardHandicapTest();
  Tests::runBoardStressTest();

  Tests::runSgfTests();
  Tests::runBasicSymmetryTests();
  Tests::runBoardSymmetryTests();

  ScoreValue::freeTables();

  cout << "All tests passed" << endl;
  return 0;
}

int MainCmds::runoutputtests(int argc, const char* const* argv) {
  (void)argc;
  (void)argv;
  Board::initHash();
  ScoreValue::initTables();

  Tests::runNNInputsV3V4Tests();
  Tests::runNNLessSearchTests();
  Tests::runTrainingWriteTests();
  Tests::runTimeControlsTests();
  Tests::runScoreTests();
  Tests::runNNSymmetryTests();
  Tests::runSgfFileTests();

  ScoreValue::freeTables();

  return 0;
}

int MainCmds::runsearchtests(int argc, const char* const* argv) {
  Board::initHash();
  ScoreValue::initTables();

  if(argc != 6) {
    cerr << "Must supply exactly five arguments: MODEL_FILE INPUTSNHWC CUDANHWC SYMMETRY FP16" << endl;
    return 1;
  }
  Tests::runSearchTests(
    string(argv[1]),
    Global::stringToBool(argv[2]),
    Global::stringToBool(argv[3]),
    Global::stringToInt(argv[4]),
    Global::stringToBool(argv[5])
  );

  ScoreValue::freeTables();

  return 0;
}

int MainCmds::runsearchtestsv3(int argc, const char* const* argv) {
  Board::initHash();
  ScoreValue::initTables();

  if(argc != 6) {
    cerr << "Must supply exactly five arguments: MODEL_FILE INPUTSNHWC CUDANHWC SYMMETRY FP16" << endl;
    return 1;
  }
  Tests::runSearchTestsV3(
    string(argv[1]),
    Global::stringToBool(argv[2]),
    Global::stringToBool(argv[3]),
    Global::stringToInt(argv[4]),
    Global::stringToBool(argv[5])
  );

  ScoreValue::freeTables();

  return 0;
}

int MainCmds::runsearchtestsv8(int argc, const char* const* argv) {
  Board::initHash();
  ScoreValue::initTables();

  if(argc != 5) {
    cerr << "Must supply exactly four arguments: MODEL_FILE INPUTSNHWC CUDANHWC FP16" << endl;
    return 1;
  }
  Tests::runSearchTestsV8(
    string(argv[1]),
    Global::stringToBool(argv[2]),
    Global::stringToBool(argv[3]),
    Global::stringToBool(argv[4])
  );

  ScoreValue::freeTables();

  return 0;
}

int MainCmds::runselfplayinittests(int argc, const char* const* argv) {
  if(argc != 2) {
    cerr << "Must supply exactly one argument: MODEL_FILE" << endl;
    return 1;
  }

  Board::initHash();
  ScoreValue::initTables();

  Tests::runSelfplayInitTestsWithNN(
    string(argv[1])
  );
  Tests::runMoreSelfplayTestsWithNN(
    string(argv[1])
  );

  ScoreValue::freeTables();

  return 0;
}

int MainCmds::runselfplayinitstattests(int argc, const char* const* argv) {
  if(argc != 2) {
    cerr << "Must supply exactly one argument: MODEL_FILE" << endl;
    return 1;
  }

  Board::initHash();
  ScoreValue::initTables();

  Tests::runSelfplayStatTestsWithNN(
    string(argv[1])
  );

  ScoreValue::freeTables();

  return 0;
}

int MainCmds::runsekitrainwritetests(int argc, const char* const* argv) {
  if(argc != 2) {
    cerr << "Must supply exactly one argument: MODEL_FILE" << endl;
    return 1;
  }

  Board::initHash();
  ScoreValue::initTables();

  Tests::runSekiTrainWriteTests(
    string(argv[1])
  );

  ScoreValue::freeTables();

  return 0;
}

int MainCmds::runnnlayertests(int argc, const char* const* argv) {
  (void)argc;
  (void)argv;
  Tests::runNNLayerTests();
  return 0;
}

int MainCmds::runnnontinyboardtest(int argc, const char* const* argv) {
  if(argc != 6) {
    cerr << "Must supply exactly five arguments: MODEL_FILE INPUTSNHWC CUDANHWC SYMMETRY FP16" << endl;
    return 1;
  }
  Board::initHash();
  ScoreValue::initTables();

  Tests::runNNOnTinyBoard(
    string(argv[1]),
    Global::stringToBool(argv[2]),
    Global::stringToBool(argv[3]),
    Global::stringToInt(argv[4]),
    Global::stringToBool(argv[5])
  );

  ScoreValue::freeTables();

  return 0;
}

int MainCmds::runnnsymmetriestest(int argc, const char* const* argv) {
  if(argc != 5) {
    cerr << "Must supply exactly four arguments: MODEL_FILE INPUTSNHWC CUDANHWC FP16" << endl;
    return 1;
  }
  Board::initHash();
  ScoreValue::initTables();

  Tests::runNNSymmetries(
    string(argv[1]),
    Global::stringToBool(argv[2]),
    Global::stringToBool(argv[3]),
    Global::stringToBool(argv[4])
  );

  ScoreValue::freeTables();

  return 0;
}

int MainCmds::runnnonmanyposestest(int argc, const char* const* argv) {
  if(argc != 6 && argc != 7) {
    cerr << "Must supply five or six arguments: MODEL_FILE INPUTSNHWC CUDANHWC SYMMETRY FP16 [COMPARISONFILE]" << endl;
    return 1;
  }
  Board::initHash();
  ScoreValue::initTables();

  if(argc == 6) {
    Tests::runNNOnManyPoses(
      string(argv[1]),
      Global::stringToBool(argv[2]),
      Global::stringToBool(argv[3]),
      Global::stringToInt(argv[4]),
      Global::stringToBool(argv[5]),
      ""
    );
  }
  else if(argc == 7) {
    Tests::runNNOnManyPoses(
      string(argv[1]),
      Global::stringToBool(argv[2]),
      Global::stringToBool(argv[3]),
      Global::stringToInt(argv[4]),
      Global::stringToBool(argv[5]),
      string(argv[6])
    );
  }

  ScoreValue::freeTables();

  return 0;
}

int MainCmds::runnnbatchingtest(int argc, const char* const* argv) {
  if(argc != 5) {
    cerr << "Must supply exactly four arguments: MODEL_FILE INPUTSNHWC CUDANHWC FP16" << endl;
    return 1;
  }
  Board::initHash();
  ScoreValue::initTables();

  Tests::runNNBatchingTest(
    string(argv[1]),
    Global::stringToBool(argv[2]),
    Global::stringToBool(argv[3]),
    Global::stringToBool(argv[4])
  );

  ScoreValue::freeTables();

  return 0;
}


int MainCmds::runownershiptests(int argc, const char* const* argv) {
  if(argc != 3) {
    cerr << "Must supply exactly two arguments: GTP_CONFIG MODEL_FILE" << endl;
    return 1;
  }
  Board::initHash();
  ScoreValue::initTables();

  Tests::runOwnershipTests(
    string(argv[1]),
    string(argv[2])
  );

  ScoreValue::freeTables();
  return 0;
}


int MainCmds::runtinynntests(int argc, const char* const* argv) {
  if(argc != 2) {
    cerr << "Must supply exactly one arguments: TMPDIR" << endl;
    return 1;
  }
  Board::initHash();
  ScoreValue::initTables();

  Logger logger;
  logger.setLogToStdout(true);
  logger.setLogTime(false);

  ConfigParser cfg;
  {
    //Dummy parameters
    int maxVisits = 500;
    int maxPlayouts = 500;
    double maxTime = 1.0;
    double maxPonderTime = 1.0;
    int nnCacheSizePowerOfTwo = 12;
    int nnMutexPoolSizePowerOfTwo = 10;
    int numSearchThreads = 3;
    string cfgStr = GTPConfig::makeConfig(
      Rules(),
      maxVisits,
      maxPlayouts,
      maxTime,
      maxPonderTime,
      std::vector<int>(),
      nnCacheSizePowerOfTwo,
      nnMutexPoolSizePowerOfTwo,
      numSearchThreads
    );
    istringstream in(cfgStr);
    cfg.initialize(in);
  }

  TinyModelTest::runTinyModelTest(
    string(argv[1]),
    logger,
    cfg
  );

  ScoreValue::freeTables();
  return 0;
}

int MainCmds::runbeginsearchspeedtest(int argc, const char* const* argv) {
  Board::initHash();
  ScoreValue::initTables();

  ConfigParser cfg;
  string modelFile;
  try {
    KataGoCommandLine cmd("Begin search speed test");
    cmd.addConfigFileArg("","");
    cmd.addModelFileArg();
    cmd.addOverrideConfigArg();

    cmd.parse(argc,argv);

    cmd.getConfig(cfg);
    modelFile = cmd.getModelFile();
  }
  catch (TCLAP::ArgException &e) {
    cerr << "Error: " << e.error() << " for argument " << e.argId() << endl;
    return 1;
  }

  Rand rand;
  Logger logger;
  logger.setLogToStdout(true);

  NNEvaluator* nnEval = NULL;
  const bool loadKomiFromCfg = false;
  Rules rules = Setup::loadSingleRules(cfg,loadKomiFromCfg);
  SearchParams params = Setup::loadSingleParams(cfg,Setup::SETUP_FOR_GTP);
  {
    Setup::initializeSession(cfg);
    int maxConcurrentEvals = params.numThreads * 2 + 16; // * 2 + 16 just to give plenty of headroom
    int expectedConcurrentEvals = params.numThreads;
    int defaultMaxBatchSize = std::max(8,((params.numThreads+3)/4)*4);
    string expectedSha256 = "";
    nnEval = Setup::initializeNNEvaluator(
      modelFile,modelFile,expectedSha256,cfg,logger,rand,maxConcurrentEvals,expectedConcurrentEvals,
      Board::MAX_LEN,Board::MAX_LEN,defaultMaxBatchSize,
      Setup::SETUP_FOR_GTP
    );
  }
  logger.write("Loaded neural net");
  string searchRandSeed = Global::uint64ToString(rand.nextUInt64());
  AsyncBot* bot = new AsyncBot(params, nnEval, &logger, searchRandSeed);

  Board board = Board::parseBoard(19,19,R"%%(
...................
...................
................o..
...x...........x...
...................
...................
...................
...................
...................
...................
...................
...................
...................
...................
...................
..ooo..........o...
..xx...............
.....x.............
...................
)%%");

  Player nextPla = P_BLACK;
  rules.komi = 6.5;
  BoardHistory hist(board,nextPla,rules,0);

  bot->setPosition(nextPla,board,hist);

  double time;
  ClockTimer timer;

  Loc moveLoc = bot->genMoveSynchronous(bot->getSearch()->rootPla,TimeControls());
  time = timer.getSeconds();

  Search* search = bot->getSearchStopAndWait();
  PrintTreeOptions options;
  Player perspective = P_WHITE;
  Board::printBoard(cout, board, Board::NULL_LOC, &(hist.moveHistory));
  search->printTree(cout, search->rootNode, options, perspective);

  cout << "Move: " << Location::toString(moveLoc,board) << endl;
  cout << "Time taken for search: " << time << endl;

  timer.reset();
  bot->makeMove(moveLoc,nextPla);
  nextPla = getOpp(nextPla);
  time = timer.getSeconds();
  cout << "Time taken for makeMove: " << time << endl;

  timer.reset();
  bool pondering = false;
  search->beginSearch(pondering);
  time = timer.getSeconds();
  cout << "Time taken for beginSearch: " << time << endl;

  timer.reset();
  // moveLoc = Location::ofString("S16",board);
  moveLoc = Location::ofString("A1",board);
  bot->makeMove(moveLoc,nextPla);
  time = timer.getSeconds();
  cout << "Time taken for makeMove that empties the tree: " << time << endl;
  cout << "Visits left: " << search->getRootVisits() << endl;

  timer.reset();
  pondering = false;
  search->beginSearch(pondering);
  time = timer.getSeconds();
  cout << "Time taken for beginSearch: " << time << endl;

  delete bot;
  delete nnEval;

  ScoreValue::freeTables();
  return 0;
}

int MainCmds::runsleeptest(int argc, const char* const* argv) {
  (void)argc;
  (void)argv;
  ClockTimer timer;
  {
    cout << "Attempting to sleep for 5 seconds" << endl;
    timer.reset();
    std::this_thread::sleep_for(std::chrono::duration<double>(5));
    double elapsed = timer.getSeconds();
    cout << "Time slept: " << elapsed << endl;
  }
  {
    cout << "Attempting to sleep for 1.5 seconds" << endl;
    timer.reset();
    std::this_thread::sleep_for(std::chrono::duration<double>(1.5));
    double elapsed = timer.getSeconds();
    cout << "Time slept: " << elapsed << endl;
  }
  {
    cout << "Attempting to sleep for 0.5 seconds" << endl;
    timer.reset();
    std::this_thread::sleep_for(std::chrono::duration<double>(0.5));
    double elapsed = timer.getSeconds();
    cout << "Time slept: " << elapsed << endl;
  }
  {
    cout << "Attempting to sleep for 0.05 seconds" << endl;
    timer.reset();
    std::this_thread::sleep_for(std::chrono::duration<double>(0.05));
    double elapsed = timer.getSeconds();
    cout << "Time slept: " << elapsed << endl;
  }
  {
    cout << "Attempting to sleep for 0.0 seconds" << endl;
    timer.reset();
    std::this_thread::sleep_for(std::chrono::duration<double>(0.0));
    double elapsed = timer.getSeconds();
    cout << "Time slept: " << elapsed << endl;
  }
  return 0;

}
