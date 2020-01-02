#include "../tests/tests.h"

#include "../dataio/trainingwrite.h"
#include "../dataio/sgf.h"
#include "../neuralnet/nneval.h"
#include "../program/play.h"

using namespace std;
using namespace TestCommon;

static NNEvaluator* startNNEval(
  const string& modelFile, const string& seed, Logger& logger,
  int defaultSymmetry, bool inputsUseNHWC, bool useNHWC, bool useFP16
) {
  const string& modelName = modelFile;
  vector<int> gpuIdxByServerThread = {0};
  vector<int> gpuIdxs = {0};
  int modelFileIdx = 0;
  int maxBatchSize = 16;
  int maxConcurrentEvals = 1024;
  int nnXLen = NNPos::MAX_BOARD_LEN;
  int nnYLen = NNPos::MAX_BOARD_LEN;
  bool requireExactNNLen = false;
  int nnCacheSizePowerOfTwo = 16;
  int nnMutexPoolSizePowerOfTwo = 12;
  bool debugSkipNeuralNet = modelFile == "/dev/null";
  float nnPolicyTemperature = 1.0;
  const string openCLTunerFile = "";
  bool openCLReTunePerBoardSize = false;
  NNEvaluator* nnEval = new NNEvaluator(
    modelName,
    modelFile,
    gpuIdxs,
    &logger,
    modelFileIdx,
    maxBatchSize,
    maxConcurrentEvals,
    nnXLen,
    nnYLen,
    requireExactNNLen,
    inputsUseNHWC,
    nnCacheSizePowerOfTwo,
    nnMutexPoolSizePowerOfTwo,
    debugSkipNeuralNet,
    nnPolicyTemperature,
    openCLTunerFile,
    openCLReTunePerBoardSize,
    useFP16,
    useNHWC
  );

  int numNNServerThreadsPerModel = 1;
  bool nnRandomize = false;

  nnEval->spawnServerThreads(
    numNNServerThreadsPerModel,
    nnRandomize,
    seed,
    defaultSymmetry,
    logger,
    gpuIdxByServerThread
  );

  return nnEval;
}

void Tests::runTrainingWriteTests() {
  cout << "Running training write tests" << endl;
  NeuralNet::globalInitialize();

  int maxRows = 256;
  double firstFileMinRandProp = 1.0;
  int debugOnlyWriteEvery = 5;

  Logger logger;
  logger.setLogToStdout(false);
  logger.setLogTime(false);
  logger.addOStream(cout);

  auto run = [&](
    const string& seedBase, const Rules& rules,
    double drawEquivalentWinsForWhite, int inputsVersion,
    int nnXLen, int nnYLen,
    int boardXLen, int boardYLen,
    bool cheapLongSgf
  ) {
    TrainingDataWriter dataWriter(&cout,inputsVersion, maxRows, firstFileMinRandProp, nnXLen, nnYLen, debugOnlyWriteEvery, seedBase+"dwriter");

    NNEvaluator* nnEval = startNNEval("/dev/null",seedBase+"nneval",logger,0,true,false,false);

    SearchParams params;
    params.maxVisits = cheapLongSgf ? 2 : 100;
    params.drawEquivalentWinsForWhite = drawEquivalentWinsForWhite;

    MatchPairer::BotSpec botSpec;
    botSpec.botIdx = 0;
    botSpec.botName = string("test");
    botSpec.nnEval = nnEval;
    botSpec.baseParams = params;

    Board initialBoard(boardXLen,boardYLen);
    Player initialPla = P_BLACK;
    int initialEncorePhase = 0;
    BoardHistory initialHist(initialBoard,initialPla,rules,initialEncorePhase);

    ExtraBlackAndKomi extraBlackAndKomi;
    extraBlackAndKomi.extraBlack = 0;
    extraBlackAndKomi.komiBase = rules.komi;
    extraBlackAndKomi.komi = rules.komi;
    bool doEndGameIfAllPassAlive = cheapLongSgf ? false : true;
    bool clearBotAfterSearch = true;
    int maxMovesPerGame = cheapLongSgf ? 200 : 40;
    vector<std::atomic<bool>*> stopConditions;
    FancyModes fancyModes;
    fancyModes.initGamesWithPolicy = true;
    fancyModes.forkSidePositionProb = 0.10;
    fancyModes.forSelfPlay = true;
    fancyModes.dataXLen = nnXLen;
    fancyModes.dataYLen = nnYLen;
    Rand rand(seedBase+"play");
    OtherGameProperties otherGameProps;
    FinishedGameData* gameData = Play::runGame(
      initialBoard,initialPla,initialHist,extraBlackAndKomi,
      botSpec,botSpec,
      seedBase+"search",
      doEndGameIfAllPassAlive, clearBotAfterSearch,
      logger, false, false,
      maxMovesPerGame, stopConditions,
      fancyModes, otherGameProps,
      rand,
      NULL
    );

    cout << "seedBase: " << seedBase << endl;
    cout << gameData->startHist.getRecentBoard(0) << endl;
    gameData->endHist.printDebugInfo(cout,gameData->endHist.getRecentBoard(0));
    cout << "Num captured black stones " << gameData->endHist.getRecentBoard(0).numBlackCaptures << endl;
    cout << "Num captured white stones " << gameData->endHist.getRecentBoard(0).numWhiteCaptures << endl;

    if(cheapLongSgf) {
      WriteSgf::writeSgf(cout,"Black","White",gameData->endHist,gameData);
    }
    else {
      dataWriter.writeGame(*gameData);
      dataWriter.flushIfNonempty();
    }

    delete gameData;
    delete nnEval;
    cout << endl;
  };

  int inputsVersion = 3;

  run("testtrainingwrite-tt",Rules::getTrompTaylorish(),0.5,inputsVersion,5,5,5,5,false);

  Rules rules;
  rules.koRule = Rules::KO_SIMPLE;
  rules.scoringRule = Rules::SCORING_TERRITORY;
  rules.multiStoneSuicideLegal = false;
  rules.taxRule = Rules::TAX_SEKI;
  rules.komi = 5;
  run("testtrainingwrite-jp",rules,0.5,inputsVersion,5,5,5,5,false);

  rules = Rules::getTrompTaylorish();
  rules.komi = 7;
  run("testtrainingwrite-gooddraws",rules,0.7,inputsVersion,5,5,5,5,false);

  inputsVersion = 5;
  run("testtrainingwrite-tt-v5",Rules::getTrompTaylorish(),0.5,inputsVersion,5,5,5,5,false);

  //Non-square v4
  inputsVersion = 4;
  run("testtrainingwrite-rect-v4",Rules::getTrompTaylorish(),0.5,inputsVersion,9,3,7,3,false);

  //V3 group taxing
  inputsVersion = 3;
  rules = Rules::getTrompTaylorish();
  rules.taxRule = Rules::TAX_ALL;
  run("testtrainingwrite-taxall-v3",rules,0.5,inputsVersion,5,5,5,5,false);
  run("testtrainingwrite-taxall-v3-a",rules,0.5,inputsVersion,5,5,5,5,false);
  run("testtrainingwrite-taxall-v3-c",rules,0.5,inputsVersion,5,5,5,5,false);

  //JP 3x3 game
  inputsVersion = 3;
  rules = Rules::getSimpleTerritory();
  run("testtrainingwrite-simpleterritory-sgf-c",rules,0.5,inputsVersion,9,1,9,1,true);

  NeuralNet::globalCleanup();
}


void Tests::runSelfplayInitTestsWithNN(const string& modelFile) {
  cout << "Running test for selfplay initialization with NN" << endl;
  NeuralNet::globalInitialize();

  int nnXLen = 11;
  int nnYLen = 11;

  Logger logger;
  logger.setLogToStdout(false);
  logger.setLogTime(false);
  logger.addOStream(cout);

  auto run = [&](
    const string& seedBase,
    const Rules& rules,
    double drawEquivalentWinsForWhite,
    int numExtraBlack,
    bool makeGameFairForEmptyBoard
  ) {
    NNEvaluator* nnEval = startNNEval(modelFile,seedBase+"nneval",logger,0,true,false,false);

    SearchParams params;
    params.maxVisits = 100;
    params.drawEquivalentWinsForWhite = drawEquivalentWinsForWhite;

    MatchPairer::BotSpec botSpec;
    botSpec.botIdx = 0;
    botSpec.botName = string("test");
    botSpec.nnEval = nnEval;
    botSpec.baseParams = params;

    Board initialBoard(11,11);
    Player initialPla = P_BLACK;
    int initialEncorePhase = 0;
    BoardHistory initialHist(initialBoard,initialPla,rules,initialEncorePhase);

    ExtraBlackAndKomi extraBlackAndKomi;
    extraBlackAndKomi.extraBlack = numExtraBlack;
    extraBlackAndKomi.komiBase = rules.komi;
    extraBlackAndKomi.komi = rules.komi;
    extraBlackAndKomi.makeGameFair = numExtraBlack > 0 && !makeGameFairForEmptyBoard;
    extraBlackAndKomi.makeGameFairForEmptyBoard = makeGameFairForEmptyBoard;

    bool doEndGameIfAllPassAlive = true;
    bool clearBotAfterSearch = true;
    int maxMovesPerGame = 1;
    vector<std::atomic<bool>*> stopConditions;
    FancyModes fancyModes;
    fancyModes.initGamesWithPolicy = true;
    fancyModes.forkSidePositionProb = 0.40;
    fancyModes.cheapSearchProb = 0.5;
    fancyModes.cheapSearchVisits = 20;
    fancyModes.cheapSearchTargetWeight = 0.123f;
    fancyModes.earlyForkGameProb = 0.5;
    fancyModes.earlyForkGameExpectedMoveProp = 0.05;
    fancyModes.forkGameMinChoices = 2;
    fancyModes.earlyForkGameMaxChoices = 2;
    fancyModes.compensateKomiVisits = 5;
    fancyModes.forSelfPlay = true;
    fancyModes.dataXLen = nnXLen;
    fancyModes.dataYLen = nnYLen;

    string searchRandSeed = seedBase+"search";
    Search* bot = new Search(botSpec.baseParams, botSpec.nnEval, searchRandSeed);

    Rand rand(seedBase+"play");
    OtherGameProperties otherGameProps;
    FinishedGameData* gameData = Play::runGame(
      initialBoard,initialPla,initialHist,extraBlackAndKomi,
      botSpec,botSpec,
      bot,bot,
      doEndGameIfAllPassAlive, clearBotAfterSearch,
      logger, false, false,
      maxMovesPerGame, stopConditions,
      fancyModes, otherGameProps,
      rand,
      NULL
    );

    ForkData forkData;
    Play::maybeForkGame(gameData,&forkData,fancyModes,rand,bot);

    cout << "====================================================================================================" << endl;
    cout << "====================================================================================================" << endl;
    cout << "====================================================================================================" << endl;
    cout << "seedBase: " << seedBase << endl;
    gameData->printDebug(cout);
    if(forkData.forks.size() > 0) {
      cout << "Forking to initial position " << PlayerIO::colorToChar(forkData.forks[0]->pla) << endl;
      cout << "Pre-fair komi " << forkData.forks[0]->hist.rules.komi << endl;
      Board board = forkData.forks[0]->board;
      BoardHistory hist = forkData.forks[0]->hist;
      Player pla = forkData.forks[0]->pla;
      Play::adjustKomiToEven(
        bot, bot, board, hist, pla,
        fancyModes.cheapSearchVisits, logger, OtherGameProperties(), rand
      );
      BoardHistory hist2 = forkData.forks[0]->hist;
      float oldKomi = hist2.rules.komi;
      double lead = Play::computeLead(
        bot, bot, board, hist2, pla,
        fancyModes.cheapSearchVisits, logger, OtherGameProperties()
      );
      cout << "Lead: " << lead << endl;
      hist.printDebugInfo(cout,board);
      testAssert(hist2.rules.komi == oldKomi);
    }
    delete gameData;
    delete bot;
    delete nnEval;
    cout << endl;
  };


  run("testselfplayinit0",Rules::getTrompTaylorish(),0.5,0,false);
  run("testselfplayinit1",Rules::getTrompTaylorish(),0.5,0,false);
  run("testselfplayinit2",Rules::getTrompTaylorish(),0.5,0,false);
  run("testselfplayinit3",Rules::getTrompTaylorish(),0.5,0,false);
  run("testselfplayinit4",Rules::getTrompTaylorish(),0.5,0,false);
  run("testselfplayinit5",Rules::getTrompTaylorish(),0.5,0,false);
  run("testselfplayinit6",Rules::getTrompTaylorish(),0.5,0,false);
  run("testselfplayinit7",Rules::getTrompTaylorish(),0.5,0,false);
  run("testselfplayinit8",Rules::getTrompTaylorish(),0.5,0,false);
  run("testselfplayinit9",Rules::getTrompTaylorish(),0.5,0,false);

  run("testselfplayinith1-0",Rules::getTrompTaylorish(),0.5,1,false);
  run("testselfplayinith1-1",Rules::getTrompTaylorish(),0.5,1,false);
  run("testselfplayinith1-2",Rules::getTrompTaylorish(),0.5,1,false);
  run("testselfplayinith1-3",Rules::getTrompTaylorish(),0.5,1,false);
  run("testselfplayinith1-4",Rules::getTrompTaylorish(),0.5,1,false);
  run("testselfplayinith1-5",Rules::getTrompTaylorish(),0.5,1,false);
  run("testselfplayinith1-6",Rules::getTrompTaylorish(),0.5,1,false);
  run("testselfplayinith1-7",Rules::getTrompTaylorish(),0.5,1,false);
  run("testselfplayinith1-8",Rules::getTrompTaylorish(),0.5,1,false);
  run("testselfplayinith1-9",Rules::getTrompTaylorish(),0.5,1,false);

  run("testselfplayinith2-0",Rules::getTrompTaylorish(),0.5,2,false);
  run("testselfplayinith2-1",Rules::getTrompTaylorish(),0.5,2,false);
  run("testselfplayinith2-2",Rules::getTrompTaylorish(),0.5,2,false);
  run("testselfplayinith2-3",Rules::getTrompTaylorish(),0.5,2,false);
  run("testselfplayinith2-4",Rules::getTrompTaylorish(),0.5,2,false);
  run("testselfplayinith2-5",Rules::getTrompTaylorish(),0.5,2,false);
  run("testselfplayinith2-6",Rules::getTrompTaylorish(),0.5,2,false);
  run("testselfplayinith2-7",Rules::getTrompTaylorish(),0.5,2,false);
  run("testselfplayinith2-8",Rules::getTrompTaylorish(),0.5,2,false);
  run("testselfplayinith2-9",Rules::getTrompTaylorish(),0.5,2,false);

  run("testselfplayinithE0",Rules::getTrompTaylorish(),0.5,0,true);
  run("testselfplayinithE1",Rules::getTrompTaylorish(),0.5,1,true);
  run("testselfplayinithE2",Rules::getTrompTaylorish(),0.5,2,true);

  Rules r = Rules::getTrompTaylorish();
  r.hasButton = true;
  run("testselfplayinit0button",r,0.5,0,false);
  run("testselfplayinit1button",r,0.5,0,false);
  run("testselfplayinith1-0button",r,0.5,1,false);
  run("testselfplayinith1-1button",r,0.5,1,false);
  run("testselfplayinith2-0button",r,0.5,2,false);
  run("testselfplayinith2-1button",r,0.5,2,false);


  NeuralNet::globalCleanup();
}

void Tests::runMoreSelfplayTestsWithNN(const string& modelFile) {
  cout << "Running more tests for selfplay" << endl;
  NeuralNet::globalInitialize();

  int nnXLen = 11;
  int nnYLen = 11;

  Logger logger;
  logger.setLogToStdout(false);
  logger.setLogTime(false);
  logger.addOStream(cout);

  auto run = [&](
    const string& seedBase,
    const Rules& rules,
    bool testAsym,
    bool testLead,
    bool testSurpriseWeight
  ) {
    NNEvaluator* nnEval = startNNEval(modelFile,seedBase+"nneval",logger,0,true,false,false);

    SearchParams params;
    params.maxVisits = 100;
    params.drawEquivalentWinsForWhite = 0.5;
    if(testLead) {
      params.chosenMoveTemperature = 1.0;
      params.chosenMoveTemperatureEarly = 1.0;
    }

    MatchPairer::BotSpec botSpec;
    botSpec.botIdx = 0;
    botSpec.botName = string("test");
    botSpec.nnEval = nnEval;
    botSpec.baseParams = params;

    Board initialBoard(11,11);
    Player initialPla = P_BLACK;
    int initialEncorePhase = 0;
    BoardHistory initialHist(initialBoard,initialPla,rules,initialEncorePhase);

    ExtraBlackAndKomi extraBlackAndKomi;
    extraBlackAndKomi.extraBlack = 0;
    extraBlackAndKomi.komiBase = rules.komi;
    extraBlackAndKomi.komi = rules.komi;
    extraBlackAndKomi.makeGameFair = false;
    extraBlackAndKomi.makeGameFairForEmptyBoard = false;

    bool doEndGameIfAllPassAlive = true;
    bool clearBotAfterSearch = true;
    int maxMovesPerGame = (testLead || testSurpriseWeight) ? 30 : 15;
    vector<std::atomic<bool>*> stopConditions;
    FancyModes fancyModes;
    fancyModes.initGamesWithPolicy = true;
    fancyModes.forkSidePositionProb = 0.0;
    fancyModes.cheapSearchProb = 0.5;
    fancyModes.cheapSearchVisits = 50;
    fancyModes.cheapSearchTargetWeight = 0.456f;
    fancyModes.compensateKomiVisits = 10;
    fancyModes.minAsymmetricCompensateKomiProb = 0.5;
    if(testLead)
      fancyModes.estimateLeadProb = 0.7;
    if(testSurpriseWeight)
      fancyModes.policySurpriseDataWeight = 0.8;

    fancyModes.forSelfPlay = true;
    fancyModes.dataXLen = nnXLen;
    fancyModes.dataYLen = nnYLen;

    string searchRandSeed = seedBase+"search";
    Search* bot = new Search(botSpec.baseParams, botSpec.nnEval, searchRandSeed);

    cout << "====================================================================================================" << endl;
    cout << "====================================================================================================" << endl;
    cout << "====================================================================================================" << endl;
    cout << "seedBase: " << seedBase << endl;

    Rand rand(seedBase+"play");
    OtherGameProperties otherGameProps;
    if(testAsym) {
      otherGameProps.playoutDoublingAdvantage = log(3.0) / log(2.0);
      otherGameProps.playoutDoublingAdvantagePla = P_WHITE;
    }
    bool logSearchInfo = testSurpriseWeight;
    FinishedGameData* gameData = Play::runGame(
      initialBoard,initialPla,initialHist,extraBlackAndKomi,
      botSpec,botSpec,
      bot,bot,
      doEndGameIfAllPassAlive, clearBotAfterSearch,
      logger, logSearchInfo, false,
      maxMovesPerGame, stopConditions,
      fancyModes, otherGameProps,
      rand,
      NULL
    );

    gameData->printDebug(cout);
    delete gameData;
    delete bot;
    delete nnEval;
    cout << endl;
  };


  run("testasym!",Rules::getTrompTaylorish(),true,false,false);
  run("test lead!",Rules::getTrompTaylorish(),false,true,false);
  Rules r = Rules::getTrompTaylorish();
  r.hasButton = true;
  run("test lead int button!",r,false,true,false);
  run("test surprise!",Rules::getTrompTaylorish(),false,false,true);


  //Test lead specifically on a final position
  auto testLeadOnBoard = [&](
    const string& seedBase,
    const Board& board,
    Rules rules,
    float komi
  ) {
    NNEvaluator* nnEval = startNNEval(modelFile,seedBase+"nneval",logger,0,true,false,false);
    SearchParams params;
    string searchRandSeed = seedBase+"search";
    Search* bot = new Search(params, nnEval, searchRandSeed);

    rules.komi = komi;
    Player pla = P_BLACK;
    BoardHistory hist(board,pla,rules,0);
    int compensateKomiVisits = 50;
    OtherGameProperties otherGameProps;
    double lead = Play::computeLead(bot,bot,board,hist,pla,compensateKomiVisits,logger,otherGameProps);
    assert(hist.rules.komi == komi);
    cout << board << endl;
    cout << "LEAD: " << lead << endl;
    delete bot;
    delete nnEval;
  };

  Rules rules = Rules::getTrompTaylorish();
  {
    Board board = Board::parseBoard(9,9,R"%%(
.........
.o...o...
.........
oooooooo.
xxxxxxxx.
.........
....x..x.
..x......
.........
)%%");
    testLeadOnBoard("basic9x9 lead", board, rules, 7.5f);
  }
  {
    Board board = Board::parseBoard(9,9,R"%%(
.........
.o...o...
.........
oooooooo.
xxxxxxxx.
.........
....x..x.
..x......
.........
)%%");
    testLeadOnBoard("basic9x9 lead komi + 0.5", board, rules, 8.0f);
  }
  {
    Board board = Board::parseBoard(9,9,R"%%(
.........
.o...o...
.........
oooooooo.
xxxxxxxx.
.........
....x..x.
..x......
.........
)%%");
    testLeadOnBoard("basic9x9 lead komi + 1", board, rules, 8.5f);
  }
  {
    Board board = Board::parseBoard(9,9,R"%%(
.........
.o...o...
.........
oooooooo.
xxxxxxxx.
.........
....x..x.
..x......
.........
)%%");
    testLeadOnBoard("basic9x9 lead komi + 1.5", board, rules, 9.0f);
  }
  {
    Board board = Board::parseBoard(9,9,R"%%(
.........
.o...o...
.........
oooooooo.
xxxxxxxx.
.........
....x..x.
..x......
.........
)%%");
    testLeadOnBoard("basic9x9 lead komi + 2.0", board, rules, 9.5f);
  }

  {
    Board board = Board::parseBoard(9,9,R"%%(
.x.......
ooooooooo
xxxxxxxxx
.........
.x..o.xo.
.ox.oxx..
.x.xxo.x.
.oxo.xxo.
.x.....x.
)%%");
    testLeadOnBoard("black crushing lead", board, rules, 7.5f);
  }
  {
    Board board = Board::parseBoard(9,9,R"%%(
.x.......
ooooooooo
xxxxxxxxx
.........
.x..o.xo.
.ox.oxx..
.x.xxo.x.
.oxo.xxo.
.x.....x.
)%%");
    testLeadOnBoard("black crushing lead komi + 0.5", board, rules, 8.0f);
  }
  {
    Board board = Board::parseBoard(9,9,R"%%(
.x.......
ooooooooo
xxxxxxxxx
.........
.x..o.xo.
.ox.oxx..
.x.xxo.x.
.oxo.xxo.
.x.....x.
)%%");
    testLeadOnBoard("black crushing lead komi + 1.0", board, rules, 8.5f);
  }
  {
    Board board = Board::parseBoard(9,9,R"%%(
.x.......
ooooooooo
xxxxxxxxx
.........
.x..o.xo.
.ox.oxx..
.x.xxo.x.
.oxo.xxo.
.x.....x.
)%%");
    testLeadOnBoard("black crushing lead komi + 1.5", board, rules, 9.0f);
  }
  {
    Board board = Board::parseBoard(9,9,R"%%(
.x.......
ooooooooo
xxxxxxxxx
.........
.x..o.xo.
.ox.oxx..
.x.xxo.x.
.oxo.xxo.
.x.....x.
)%%");
    testLeadOnBoard("black crushing lead komi + 2.0", board, rules, 9.5f);
  }

  Rules buttonRules = Rules::getTrompTaylorish();
  buttonRules.hasButton = true;
  {
    Board board = Board::parseBoard(9,9,R"%%(
.........
.o...o...
.........
oooooooo.
xxxxxxxx.
.........
....x..x.
..x......
.........
)%%");
    testLeadOnBoard("basic9x9 lead button", board, buttonRules, 7.5f);
  }
  {
    Board board = Board::parseBoard(9,9,R"%%(
.........
.o...o...
.........
oooooooo.
xxxxxxxx.
.........
....x..x.
..x......
.........
)%%");
    testLeadOnBoard("basic9x9 lead button komi + 0.5", board, buttonRules,  8.0f);
  }
  {
    Board board = Board::parseBoard(9,9,R"%%(
.........
.o...o...
.........
oooooooo.
xxxxxxxx.
.........
....x..x.
..x......
.........
)%%");
    testLeadOnBoard("basic9x9 lead button komi + 1", board, buttonRules,  8.5f);
  }
  {
    Board board = Board::parseBoard(9,9,R"%%(
.........
.o...o...
.........
oooooooo.
xxxxxxxx.
.........
....x..x.
..x......
.........
)%%");
    testLeadOnBoard("basic9x9 lead button komi + 1.5", board, buttonRules,  9.0f);
  }
  {
    Board board = Board::parseBoard(9,9,R"%%(
.........
.o...o...
.........
oooooooo.
xxxxxxxx.
.........
....x..x.
..x......
.........
)%%");
    testLeadOnBoard("basic9x9 lead button komi + 2.0", board, buttonRules,  9.5f);
  }

  {
    //Big giant test of certain fancyModes parts and game initialization
    FancyModes fancyModes;
    //Not testing these - covered by other tests
    fancyModes.initGamesWithPolicy = false;
    fancyModes.forkSidePositionProb = false;

    fancyModes.compensateKomiVisits = 20;
    fancyModes.fancyKomiVarying = true;

    fancyModes.sekiForkHack = true;
    fancyModes.forSelfPlay = true;
    fancyModes.dataXLen = 13;
    fancyModes.dataYLen = 13;

    NNEvaluator* nnEval = startNNEval(modelFile,"game init test nneval",logger,0,true,false,false);

    std::map<string,string> cfgParams({
        std::make_pair("maxMovesPerGame","5"),
        std::make_pair("logSearchInfo","false"),
        std::make_pair("logMoves","false"),
        std::make_pair("komiAuto","true"),
        std::make_pair("komiStdev","3.0"),
        std::make_pair("handicapProb","0.2"),
        std::make_pair("handicapCompensateKomiProb","1.0"),
        std::make_pair("forkCompensateKomiProb","1.0"),
        std::make_pair("komiBigStdevProb","0.15"),
        std::make_pair("komiBigStdev","40.0"),
        std::make_pair("drawRandRadius","0.5"),
        std::make_pair("noResultStdev","0.16"),

        std::make_pair("bSizes","10,11,12,13"),
        std::make_pair("bSizeRelProbs","1,2,2,1"),
        std::make_pair("koRules","SIMPLE,POSITIONAL,SITUATIONAL"),
        std::make_pair("scoringRules","AREA,TERRITORY"),
        std::make_pair("taxRules","NONE,NONE,SEKI,SEKI,ALL"),
        std::make_pair("multiStoneSuicideLegals","false,true"),
        std::make_pair("hasButtons","false,false,true"),
        std::make_pair("allowRectangleProb","0.2"),
    });

    ConfigParser cfg(cfgParams);
    MatchPairer::BotSpec botSpec;
    botSpec.botIdx = 0;
    botSpec.botName = modelFile;
    botSpec.nnEval = nnEval;
    botSpec.baseParams = SearchParams();
    botSpec.baseParams.maxVisits = 10;
    ForkData* forkData = new ForkData();

    GameRunner* gameRunner = new GameRunner(cfg, "game init test", fancyModes, logger);
    std::vector<std::atomic<bool>*> stopConditions;
    for(int i = 0; i<100; i++) {
      FinishedGameData* data = gameRunner->runGame(i, botSpec, botSpec, forkData, logger, stopConditions, NULL);
      cout << data->startHist.rules << endl;
      cout << "Start moves size " << data->startHist.moveHistory.size()
           << " Start pla " << PlayerIO::playerToString(data->startPla)
           << " XY " << data->startBoard.x_size << " " << data->startBoard.y_size
           << " Extra black " << data->numExtraBlack
           << " Draw equiv " << data->drawEquivalentWinsForWhite
           << " Modes " << data->mode << " " << data->modeMeta1 << " " << data->modeMeta2
           << " Forkstuff " << forkData->forks.size() << " " << forkData->sekiForks.size()
           << endl;
      delete data;
    }
    delete gameRunner;
    delete forkData;
    delete nnEval;
  }

  NeuralNet::globalCleanup();
}


void Tests::runSekiTrainWriteTests(const string& modelFile) {
  cout << "Running test for how a seki gets recorded" << endl;
  NeuralNet::globalInitialize();

  int nnXLen = 13;
  int nnYLen = 13;

  Logger logger;
  logger.setLogToStdout(false);
  logger.setLogTime(false);
  logger.addOStream(cout);

  auto run = [&](const string& sgfStr, const string& seedBase, const Rules& rules) {
    int inputsVersion = 6;
    int maxRows = 256;
    double firstFileMinRandProp = 1.0;
    int debugOnlyWriteEvery = 1000;
    TrainingDataWriter dataWriter(&cout,inputsVersion, maxRows, firstFileMinRandProp, nnXLen, nnYLen, debugOnlyWriteEvery, seedBase+"dwriter");

    NNEvaluator* nnEval = startNNEval(modelFile,seedBase+"nneval",logger,0,true,false,false);

    SearchParams params;
    params.maxVisits = 30;
    params.drawEquivalentWinsForWhite = 0.5;

    MatchPairer::BotSpec botSpec;
    botSpec.botIdx = 0;
    botSpec.botName = string("test");
    botSpec.nnEval = nnEval;
    botSpec.baseParams = params;

    CompactSgf* sgf = CompactSgf::parse(sgfStr);
    Board initialBoard;
    Player initialPla;
    BoardHistory initialHist;

    ExtraBlackAndKomi extraBlackAndKomi;
    extraBlackAndKomi.extraBlack = 0;
    extraBlackAndKomi.komiBase = rules.komi;
    extraBlackAndKomi.komi = rules.komi;
    int turnNumber = sgf->moves.size();
    sgf->setupBoardAndHistAssumeLegal(rules,initialBoard,initialPla,initialHist,turnNumber);

    bool doEndGameIfAllPassAlive = true;
    bool clearBotAfterSearch = true;
    int maxMovesPerGame = 1;
    vector<std::atomic<bool>*> stopConditions;
    FancyModes fancyModes;
    fancyModes.initGamesWithPolicy = false;
    fancyModes.forkSidePositionProb = 0;
    fancyModes.cheapSearchProb = 0;
    fancyModes.cheapSearchVisits = 0;
    fancyModes.cheapSearchTargetWeight = 0;
    fancyModes.earlyForkGameProb = 0;
    fancyModes.earlyForkGameExpectedMoveProp = 0;
    fancyModes.forkGameMinChoices = 2;
    fancyModes.earlyForkGameMaxChoices = 2;
    fancyModes.compensateKomiVisits = 5;
    fancyModes.forSelfPlay = true;
    fancyModes.dataXLen = nnXLen;
    fancyModes.dataYLen = nnYLen;

    string searchRandSeed = seedBase+"search";
    Search* bot = new Search(botSpec.baseParams, botSpec.nnEval, searchRandSeed);

    Rand rand(seedBase+"play");
    OtherGameProperties otherGameProps;
    FinishedGameData* gameData = Play::runGame(
      initialBoard,initialPla,initialHist,extraBlackAndKomi,
      botSpec,botSpec,
      bot,bot,
      doEndGameIfAllPassAlive, clearBotAfterSearch,
      logger, false, false,
      maxMovesPerGame, stopConditions,
      fancyModes, otherGameProps,
      rand,
      NULL
    );

    cout << "seedBase: " << seedBase << endl;
    gameData->endHist.printDebugInfo(cout,gameData->endHist.getRecentBoard(0));
    dataWriter.writeGame(*gameData);
    dataWriter.flushIfNonempty();
    delete gameData;
    delete bot;
    delete nnEval;
    delete sgf;
    cout << endl;
  };

  vector<Rules> ruless = {
    Rules(Rules::KO_SIMPLE, Rules::SCORING_AREA, Rules::TAX_NONE, false, false, Rules::WHB_ZERO, 0.0f),
    Rules(Rules::KO_SIMPLE, Rules::SCORING_TERRITORY, Rules::TAX_NONE, false, false, Rules::WHB_ZERO, 0.0f),
    Rules(Rules::KO_SIMPLE, Rules::SCORING_AREA, Rules::TAX_SEKI, false, false, Rules::WHB_ZERO, 0.0f),
    Rules(Rules::KO_SIMPLE, Rules::SCORING_TERRITORY, Rules::TAX_SEKI, false, false, Rules::WHB_ZERO, 0.0f),
    Rules(Rules::KO_SIMPLE, Rules::SCORING_AREA, Rules::TAX_ALL, false, false, Rules::WHB_ZERO, 0.0f),
    Rules(Rules::KO_SIMPLE, Rules::SCORING_TERRITORY, Rules::TAX_ALL, false, false, Rules::WHB_ZERO, 0.0f),
  };

  string sgfStr = "(;KM[0.0]PB[]SZ[13]PW[]AP[Sabaki:0.43.3]CA[UTF-8];B[aj];W[bi];B[bk];W[cj];B[cl];W[dk];B[dm];W[el];B[dl];W[ek];B[ck];W[dj];B[bj];W[ci];B[al];W[bm];B[fm];W[em];B[fl];W[ai];B[fk];W[dh];B[fj];W[bl];B[gi];W[eg];B[hh];W[ff];B[ig];W[ge];B[jf];W[hd];B[fi];W[di];B[gh];W[dg];B[hg];W[fe];B[ke];W[ic];B[ld];W[jb];B[fh];W[he];B[je];W[jc];B[kd];W[ja];B[md];W[la];B[mb];W[ka];B[mc];W[gc];B[jh];W[cc];B[kk];W[cf];B[jk];W[dc];B[ej];W[ei];B[eh];W[fg];B[gg];W[gf];B[hf];W[ie];B[if];W[id];B[jd];W[kc];B[lb];W[kb];B[lc])";

  for(int r = 0; r<ruless.size(); r++) {
    run(sgfStr,"abc",ruless[r]);
  }

  sgfStr = "(;FF[4]CA[UTF-8]AP[GoGui:1.4.9]SZ[13]KM[0];B[jj];W[kd];B[lc];W[kc];B[ld];W[ke];B[lb];W[kb];B[la];W[mb];B[le];W[kf];B[lf];W[lg];B[kg];W[lh];B[jg];W[mc];B[mf];W[md];B[ji];W[kk];B[jk];W[kj];B[jl];W[kl];B[ki];W[li];B[ie];W[hd];B[id];W[hc];B[he];W[ic];B[ge];W[fc];B[fk];W[ee];B[fh];W[dg];B[dk];W[ci];B[cb];W[cc];B[bc];W[cd];B[bd];W[db];B[bb];W[ce];B[aa];W[ck];B[dj];W[cj];B[ka];W[jb];B[ja];W[ia];B[mg];W[mh];B[kh];W[lk];B[be];W[bf];B[cf];W[bg];B[ca];W[da];B[dc];W[ec];B[dd];W[de];B[ei];W[ff];B[ml];W[mk];B[lm];W[km];B[mj];W[lj];B[jm];W[dl];B[el];W[cl];B[gf];W[mi];B[fg];W[eg];B[fe];W[ef];B[fd];W[ed];B[af];W[ag];B[ae];W[jf];B[if];W[em];B[fm];W[dm];B[di];W[dh];B[gd];W[gc];B[jd];W[jc];B[eh];W[je];B[df];W[cg];B[ib])";

  for(int r = 0; r<ruless.size(); r++) {
    run(sgfStr,"def",ruless[r]);
  }

  NeuralNet::globalCleanup();
}
