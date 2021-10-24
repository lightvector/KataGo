
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

int MainCmds::runtests(const vector<string>& args) {
  (void)args;
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

int MainCmds::runoutputtests(const vector<string>& args) {
  (void)args;
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

int MainCmds::runsearchtests(const vector<string>& args) {
  Board::initHash();
  ScoreValue::initTables();

  if(args.size() != 6) {
    cerr << "Must supply exactly five arguments: MODEL_FILE INPUTSNHWC CUDANHWC SYMMETRY FP16" << endl;
    return 1;
  }
  Tests::runSearchTests(
    args[1],
    Global::stringToBool(args[2]),
    Global::stringToBool(args[3]),
    Global::stringToInt(args[4]),
    Global::stringToBool(args[5])
  );

  ScoreValue::freeTables();

  return 0;
}

int MainCmds::runsearchtestsv3(const vector<string>& args) {
  Board::initHash();
  ScoreValue::initTables();

  if(args.size() != 6) {
    cerr << "Must supply exactly five arguments: MODEL_FILE INPUTSNHWC CUDANHWC SYMMETRY FP16" << endl;
    return 1;
  }
  Tests::runSearchTestsV3(
    args[1],
    Global::stringToBool(args[2]),
    Global::stringToBool(args[3]),
    Global::stringToInt(args[4]),
    Global::stringToBool(args[5])
  );

  ScoreValue::freeTables();

  return 0;
}

int MainCmds::runsearchtestsv8(const vector<string>& args) {
  Board::initHash();
  ScoreValue::initTables();

  if(args.size() != 5) {
    cerr << "Must supply exactly four arguments: MODEL_FILE INPUTSNHWC CUDANHWC FP16" << endl;
    return 1;
  }
  Tests::runSearchTestsV8(
    args[1],
    Global::stringToBool(args[2]),
    Global::stringToBool(args[3]),
    Global::stringToBool(args[4])
  );

  ScoreValue::freeTables();

  return 0;
}

int MainCmds::runselfplayinittests(const vector<string>& args) {
  if(args.size() != 2) {
    cerr << "Must supply exactly one argument: MODEL_FILE" << endl;
    return 1;
  }

  Board::initHash();
  ScoreValue::initTables();

  Tests::runSelfplayInitTestsWithNN(
    args[1]
  );
  Tests::runMoreSelfplayTestsWithNN(
    args[1]
  );

  ScoreValue::freeTables();

  return 0;
}

int MainCmds::runselfplayinitstattests(const vector<string>& args) {
  if(args.size() != 2) {
    cerr << "Must supply exactly one argument: MODEL_FILE" << endl;
    return 1;
  }

  Board::initHash();
  ScoreValue::initTables();

  Tests::runSelfplayStatTestsWithNN(
    args[1]
  );

  ScoreValue::freeTables();

  return 0;
}

int MainCmds::runsekitrainwritetests(const vector<string>& args) {
  if(args.size() != 2) {
    cerr << "Must supply exactly one argument: MODEL_FILE" << endl;
    return 1;
  }

  Board::initHash();
  ScoreValue::initTables();

  Tests::runSekiTrainWriteTests(
    args[1]
  );

  ScoreValue::freeTables();

  return 0;
}

int MainCmds::runnnlayertests(const vector<string>& args) {
  (void)args;
  Tests::runNNLayerTests();
  return 0;
}

int MainCmds::runnnontinyboardtest(const vector<string>& args) {
  if(args.size() != 6) {
    cerr << "Must supply exactly five arguments: MODEL_FILE INPUTSNHWC CUDANHWC SYMMETRY FP16" << endl;
    return 1;
  }
  Board::initHash();
  ScoreValue::initTables();

  Tests::runNNOnTinyBoard(
    args[1],
    Global::stringToBool(args[2]),
    Global::stringToBool(args[3]),
    Global::stringToInt(args[4]),
    Global::stringToBool(args[5])
  );

  ScoreValue::freeTables();

  return 0;
}

int MainCmds::runnnsymmetriestest(const vector<string>& args) {
  if(args.size() != 5) {
    cerr << "Must supply exactly four arguments: MODEL_FILE INPUTSNHWC CUDANHWC FP16" << endl;
    return 1;
  }
  Board::initHash();
  ScoreValue::initTables();

  Tests::runNNSymmetries(
    args[1],
    Global::stringToBool(args[2]),
    Global::stringToBool(args[3]),
    Global::stringToBool(args[4])
  );

  ScoreValue::freeTables();

  return 0;
}

int MainCmds::runnnonmanyposestest(const vector<string>& args) {
  if(args.size() != 6 && args.size() != 7) {
    cerr << "Must supply five or six arguments: MODEL_FILE INPUTSNHWC CUDANHWC SYMMETRY FP16 [COMPARISONFILE]" << endl;
    return 1;
  }
  Board::initHash();
  ScoreValue::initTables();

  if(args.size() == 6) {
    Tests::runNNOnManyPoses(
      args[1],
      Global::stringToBool(args[2]),
      Global::stringToBool(args[3]),
      Global::stringToInt(args[4]),
      Global::stringToBool(args[5]),
      ""
    );
  }
  else if(args.size() == 7) {
    Tests::runNNOnManyPoses(
      args[1],
      Global::stringToBool(args[2]),
      Global::stringToBool(args[3]),
      Global::stringToInt(args[4]),
      Global::stringToBool(args[5]),
      args[6]
    );
  }

  ScoreValue::freeTables();

  return 0;
}

int MainCmds::runnnbatchingtest(const vector<string>& args) {
  if(args.size() != 5) {
    cerr << "Must supply exactly four arguments: MODEL_FILE INPUTSNHWC CUDANHWC FP16" << endl;
    return 1;
  }
  Board::initHash();
  ScoreValue::initTables();

  Tests::runNNBatchingTest(
    args[1],
    Global::stringToBool(args[2]),
    Global::stringToBool(args[3]),
    Global::stringToBool(args[4])
  );

  ScoreValue::freeTables();

  return 0;
}


int MainCmds::runownershiptests(const vector<string>& args) {
  if(args.size() != 3) {
    cerr << "Must supply exactly two arguments: GTP_CONFIG MODEL_FILE" << endl;
    return 1;
  }
  Board::initHash();
  ScoreValue::initTables();

  Tests::runOwnershipTests(
    args[1],
    args[2]
  );

  ScoreValue::freeTables();
  return 0;
}


int MainCmds::runtinynntests(const vector<string>& args) {
  if(args.size() != 2) {
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

  const bool randFileName = false;
  TinyModelTest::runTinyModelTest(
    args[1],
    logger,
    cfg,
    randFileName
  );

  ScoreValue::freeTables();
  return 0;
}

int MainCmds::runnnevalcanarytests(const vector<string>& args) {
  if(args.size() != 4) {
    cerr << "Must supply exactly three arguments: GTP_CONFIG MODEL_FILE SYMMETRY" << endl;
    return 1;
  }
  const string& cfgFile = args[1];
  const string& modelFile = args[2];
  const int symmetry = Global::stringToInt(args[3]);

  Board::initHash();
  ScoreValue::initTables();

  Logger logger;
  logger.setLogToStdout(true);
  logger.setLogTime(false);

  ConfigParser cfg(cfgFile);
  Rand seedRand;

  NNEvaluator* nnEval;
  {
    Setup::initializeSession(cfg);
    int maxConcurrentEvals = 2;
    int expectedConcurrentEvals = 1;
    int defaultMaxBatchSize = 8;
    bool defaultRequireExactNNLen = false;
    string expectedSha256 = "";
    nnEval = Setup::initializeNNEvaluator(
      modelFile,modelFile,expectedSha256,cfg,logger,seedRand,maxConcurrentEvals,expectedConcurrentEvals,
      NNPos::MAX_BOARD_LEN,NNPos::MAX_BOARD_LEN,defaultMaxBatchSize,defaultRequireExactNNLen,
      Setup::SETUP_FOR_GTP
    );
  }

  bool print = true;
  Tests::runCanaryTests(nnEval,symmetry,print);
  delete nnEval;

  ScoreValue::freeTables();
  return 0;
}

int MainCmds::runbeginsearchspeedtest(const vector<string>& args) {
  Board::initHash();
  ScoreValue::initTables();

  ConfigParser cfg;
  string modelFile;
  try {
    KataGoCommandLine cmd("Begin search speed test");
    cmd.addConfigFileArg("","");
    cmd.addModelFileArg();
    cmd.addOverrideConfigArg();

    cmd.parseArgs(args);

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
    const int maxConcurrentEvals = params.numThreads * 2 + 16; // * 2 + 16 just to give plenty of headroom
    const int expectedConcurrentEvals = params.numThreads;
    const int defaultMaxBatchSize = std::max(8,((params.numThreads+3)/4)*4);
    const bool defaultRequireExactNNLen = false;
    const string expectedSha256 = "";
    nnEval = Setup::initializeNNEvaluator(
      modelFile,modelFile,expectedSha256,cfg,logger,rand,maxConcurrentEvals,expectedConcurrentEvals,
      Board::MAX_LEN,Board::MAX_LEN,defaultMaxBatchSize,defaultRequireExactNNLen,
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

int MainCmds::runsleeptest(const vector<string>& args) {
  (void)args;
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
