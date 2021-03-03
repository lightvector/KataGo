#include "../tests/tests.h"

#include "../dataio/trainingwrite.h"
#include "../dataio/sgf.h"
#include "../neuralnet/nneval.h"
#include "../program/playutils.h"
#include "../program/play.h"

using namespace std;
using namespace TestCommon;

static NNEvaluator* startNNEval(
  const string& modelFile, const string& seed, Logger& logger,
  int defaultSymmetry, bool inputsUseNHWC, bool useNHWC, bool useFP16
) {
  const string& modelName = modelFile;
  vector<int> gpuIdxByServerThread = {0};
  int maxBatchSize = 16;
  int maxConcurrentEvals = 1024;
  int nnXLen = NNPos::MAX_BOARD_LEN;
  int nnYLen = NNPos::MAX_BOARD_LEN;
  bool requireExactNNLen = false;
  int nnCacheSizePowerOfTwo = 16;
  int nnMutexPoolSizePowerOfTwo = 12;
  bool debugSkipNeuralNet = modelFile == "/dev/null";
  const string openCLTunerFile = "";
  const string homeDataDirOverride = "";
  bool openCLReTunePerBoardSize = false;
  int numNNServerThreadsPerModel = 1;
  bool nnRandomize = false;

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
    seed,
    nnRandomize,
    defaultSymmetry
  );

  nnEval->spawnServerThreads();
  return nnEval;
}

void Tests::runTrainingWriteTests() {
  cout << "Running training write tests" << endl;
  NeuralNet::globalInitialize();

  int maxRows = 256;
  double firstFileMinRandProp = 1.0;
  int debugOnlyWriteEvery = 5;

  Logger logger;
  logger.setLogToStdout(true);
  logger.setLogTime(false);

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
    auto shouldStop = []() { return false; };
    PlaySettings playSettings;
    playSettings.initGamesWithPolicy = true;
    playSettings.policyInitAreaProp = 0.04;
    playSettings.sidePositionProb = 0.10;
    playSettings.forSelfPlay = true;
    Rand rand(seedBase+"play");
    OtherGameProperties otherGameProps;
    FinishedGameData* gameData = Play::runGame(
      initialBoard,initialPla,initialHist,extraBlackAndKomi,
      botSpec,botSpec,
      seedBase+"search",
      doEndGameIfAllPassAlive, clearBotAfterSearch,
      logger, false, false,
      maxMovesPerGame, shouldStop,
      playSettings, otherGameProps,
      rand,
      nullptr,
      nullptr
    );

    cout << "seedBase: " << seedBase << endl;
    cout << gameData->startHist.getRecentBoard(0) << endl;
    gameData->endHist.printDebugInfo(cout,gameData->endHist.getRecentBoard(0));
    cout << "Num captured black stones " << gameData->endHist.getRecentBoard(0).numBlackCaptures << endl;
    cout << "Num captured white stones " << gameData->endHist.getRecentBoard(0).numWhiteCaptures << endl;

    if(cheapLongSgf) {
      WriteSgf::writeSgf(cout,"Black","White",gameData->endHist,gameData,false,false);
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

  Logger logger;
  logger.setLogToStdout(true);
  logger.setLogTime(false);

  NNEvaluator* nnEval = startNNEval(modelFile,"nneval",logger,0,true,false,false);

  auto run = [&](
    const string& seedBase,
    const Rules& rules,
    double drawEquivalentWinsForWhite,
    int numExtraBlack,
    bool makeGameFairForEmptyBoard
  ) {
    nnEval->clearCache();
    nnEval->clearStats();

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
    auto shouldStop = []() { return false; };
    PlaySettings playSettings;
    playSettings.initGamesWithPolicy = true;
    playSettings.policyInitAreaProp = 0.04;
    playSettings.sidePositionProb = 0.40;
    playSettings.cheapSearchProb = 0.5;
    playSettings.cheapSearchVisits = 20;
    playSettings.cheapSearchTargetWeight = 0.123f;
    playSettings.earlyForkGameProb = 0.5;
    playSettings.earlyForkGameExpectedMoveProp = 0.05;
    playSettings.forkGameMinChoices = 2;
    playSettings.earlyForkGameMaxChoices = 2;
    playSettings.compensateKomiVisits = 5;
    playSettings.forSelfPlay = true;

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
      maxMovesPerGame, shouldStop,
      playSettings, otherGameProps,
      rand,
      nullptr,
      nullptr
    );

    ForkData forkData;
    Play::maybeForkGame(gameData,&forkData,playSettings,rand,bot);

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
      PlayUtils::adjustKomiToEven(
        bot, bot, board, hist, pla,
        playSettings.cheapSearchVisits, logger, OtherGameProperties(), rand
      );
      BoardHistory hist2 = forkData.forks[0]->hist;
      float oldKomi = hist2.rules.komi;
      double lead = PlayUtils::computeLead(
        bot, bot, board, hist2, pla,
        playSettings.cheapSearchVisits, logger, OtherGameProperties()
      );
      cout << "Lead: " << lead << endl;
      hist.printDebugInfo(cout,board);
      testAssert(hist2.rules.komi == oldKomi);
    }
    delete gameData;
    delete bot;
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

  delete nnEval;
  NeuralNet::globalCleanup();
}

void Tests::runMoreSelfplayTestsWithNN(const string& modelFile) {
  cout << "Running more tests for selfplay" << endl;
  NeuralNet::globalInitialize();

  Logger logger;
  logger.setLogToStdout(true);
  logger.setLogTime(false);

  NNEvaluator* nnEval = startNNEval(modelFile,"nneval",logger,0,true,false,false);

  auto run = [&](
    const string& seedBase,
    const Rules& rules,
    bool testAsym,
    bool testLead,
    bool testPolicySurpriseWeight,
    bool testValueSurpriseWeight,
    bool testHint,
    bool testResign,
    bool testScaleDataWeight
  ) {
    nnEval->clearCache();
    nnEval->clearStats();

    SearchParams params;
    params.maxVisits = testResign ? 10 : 100;
    params.drawEquivalentWinsForWhite = 0.5;
    if(testLead) {
      params.chosenMoveTemperature = 1.0;
      params.chosenMoveTemperatureEarly = 1.0;
    }
    if(testHint) {
      //Triggers an old buggy codepath that has since been fixed, but left here as part of test
      params.rootPolicyTemperature = 1.000000000001;
    }

    MatchPairer::BotSpec botSpec;
    botSpec.botIdx = 0;
    botSpec.botName = string("test");
    botSpec.nnEval = nnEval;
    botSpec.baseParams = params;

    Board initialBoard(11,11);
    Player initialPla = P_BLACK;
    int initialEncorePhase = 0;
    if(testHint) {
      initialBoard = Board::parseBoard(11,11,R"%%(
...........
...........
..x...x....
........o..
...........
...........
...........
.......oo..
......oxx..
.....ox....
...........
)%%");

    }

    BoardHistory initialHist(initialBoard,initialPla,rules,initialEncorePhase);
    if(testHint)
      initialHist.setInitialTurnNumber(10);

    ExtraBlackAndKomi extraBlackAndKomi;
    extraBlackAndKomi.extraBlack = 0;
    extraBlackAndKomi.komiBase = rules.komi;
    extraBlackAndKomi.komi = rules.komi;
    extraBlackAndKomi.makeGameFair = false;
    extraBlackAndKomi.makeGameFairForEmptyBoard = false;

    bool doEndGameIfAllPassAlive = true;
    bool clearBotAfterSearch = true;
    int maxMovesPerGame = testResign ? 10000 : (testLead || testPolicySurpriseWeight || testValueSurpriseWeight) ? 30 : 15;
    auto shouldStop = []() { return false; };
    PlaySettings playSettings;
    playSettings.initGamesWithPolicy = true;
    playSettings.policyInitAreaProp = 0.04;
    playSettings.sidePositionProb = testScaleDataWeight ? 0.2 : 0.0;
    playSettings.cheapSearchProb = 0.5;
    playSettings.cheapSearchVisits = testResign ? 5 : 50;
    playSettings.cheapSearchTargetWeight = 0.456f;
    playSettings.compensateKomiVisits = 10;
    playSettings.minAsymmetricCompensateKomiProb = 0.5;
    if(testLead)
      playSettings.estimateLeadProb = 0.7;
    if(testPolicySurpriseWeight)
      playSettings.policySurpriseDataWeight = 0.8;
    if(testValueSurpriseWeight) {
      playSettings.valueSurpriseDataWeight = 0.15;
      playSettings.noResolveTargetWeights = true;
    }
    if(testResign) {
      playSettings.allowResignation = true;
      playSettings.resignThreshold = -0.9;
      playSettings.resignConsecTurns = 3;
    }
    if(testScaleDataWeight) {
      playSettings.scaleDataWeight = 4.0;
    }

    playSettings.forSelfPlay = !testResign;

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
    if(testHint) {
      otherGameProps.isHintPos = true;
      otherGameProps.hintTurn = initialHist.moveHistory.size();
      otherGameProps.hintPosHash = initialBoard.pos_hash;
      otherGameProps.hintLoc = Location::ofString("A1",initialBoard);
      otherGameProps.allowPolicyInit = false;
    }
    bool logSearchInfo = testPolicySurpriseWeight || testHint;
    FinishedGameData* gameData = Play::runGame(
      initialBoard,initialPla,initialHist,extraBlackAndKomi,
      botSpec,botSpec,
      bot,bot,
      doEndGameIfAllPassAlive, clearBotAfterSearch,
      logger, logSearchInfo, false,
      maxMovesPerGame, shouldStop,
      playSettings, otherGameProps,
      rand,
      nullptr,
      nullptr
    );
    if(testHint) {
      ForkData forkData;
      Play::maybeHintForkGame(gameData, &forkData, otherGameProps);
      cout << " Forkstuff " << forkData.forks.size() << " " << forkData.sekiForks.size() << endl;
      for(int i = 0; i<forkData.forks.size(); i++)
        cout << forkData.forks[0]->board << endl;
    }

    gameData->printDebug(cout);
    if(testResign) {
      WriteSgf::writeSgf(cout,"Black","White",gameData->endHist,gameData,false,false);
      cout << endl;
      WriteSgf::writeSgf(cout,"Black","White",gameData->endHist,gameData,false,true);
      cout << endl;
    }

    delete gameData;
    delete bot;
    cout << endl;
  };

  run("testasym!",Rules::getTrompTaylorish(),true,false,false,false,false,false,false);
  run("test lead!",Rules::getTrompTaylorish(),false,true,false,false,false,false,false);
  Rules r = Rules::getTrompTaylorish();
  r.hasButton = true;
  run("test lead int button!",r,false,true,false,false,false,false,false);
  run("test surprise!",Rules::getTrompTaylorish(),false,false,true,false,false,false,false);
  run("test value surprise!",Rules::getTrompTaylorish(),false,false,false,true,false,false,false);
  run("test hint!",Rules::getTrompTaylorish(),false,false,false,false,true,false,false);
  run("test resign!",Rules::getTrompTaylorish(),false,false,false,false,false,true,false);
  run("test scale data weight!",Rules::getTrompTaylorish(),false,false,false,false,false,false,true);

  //Test lead specifically on a final position
  auto testLeadOnBoard = [&](
    const string& seedBase,
    const Board& board,
    Rules rules,
    float komi
  ) {
    nnEval->clearCache();
    nnEval->clearStats();

    SearchParams params;
    string searchRandSeed = seedBase+"search";
    Search* bot = new Search(params, nnEval, searchRandSeed);

    rules.komi = komi;
    Player pla = P_BLACK;
    BoardHistory hist(board,pla,rules,0);
    int compensateKomiVisits = 50;
    OtherGameProperties otherGameProps;
    double lead = PlayUtils::computeLead(bot,bot,board,hist,pla,compensateKomiVisits,logger,otherGameProps);
    testAssert(hist.rules.komi == komi);
    cout << board << endl;
    cout << "LEAD: " << lead << endl;
    delete bot;
  };

  //MORE TESTING ----------------------------------------------------------------

  auto runMore = [&](
    const string& seedBase,
    const Rules& rules,
    bool testPolicySurpriseWeight,
    bool testValueSurpriseWeight,
    bool testScaleDataWeight,
    bool testSgf
  ) {
    nnEval->clearCache();
    nnEval->clearStats();

    SearchParams params;
    params.maxVisits = 100;
    params.drawEquivalentWinsForWhite = 0.5;

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
    int maxMovesPerGame = 20;
    auto shouldStop = []() { return false; };
    PlaySettings playSettings;
    playSettings.initGamesWithPolicy = true;
    playSettings.policyInitAreaProp = 0;
    playSettings.sidePositionProb = 0.2;
    playSettings.cheapSearchProb = 0.5;
    playSettings.cheapSearchVisits = 50;
    playSettings.cheapSearchTargetWeight = 0.0;
    playSettings.compensateKomiVisits = 10;
    playSettings.minAsymmetricCompensateKomiProb = 0.5;
    if(testPolicySurpriseWeight)
      playSettings.policySurpriseDataWeight = 0.5;
    if(testValueSurpriseWeight)
      playSettings.valueSurpriseDataWeight = 0.1;
    if(testScaleDataWeight)
      playSettings.scaleDataWeight = 1.5;

    playSettings.forSelfPlay = true;

    string searchRandSeed = seedBase+"search";
    Search* bot = new Search(botSpec.baseParams, botSpec.nnEval, searchRandSeed);

    cout << "====================================================================================================" << endl;
    cout << "====================================================================================================" << endl;
    cout << "====================================================================================================" << endl;
    cout << "seedBase: " << seedBase << endl;

    bool logSearchInfo = false;
    Rand rand(seedBase+"play");
    OtherGameProperties otherGameProps;
    FinishedGameData* gameData = Play::runGame(
      initialBoard,initialPla,initialHist,extraBlackAndKomi,
      botSpec,botSpec,
      bot,bot,
      doEndGameIfAllPassAlive, clearBotAfterSearch,
      logger, logSearchInfo, false,
      maxMovesPerGame, shouldStop,
      playSettings, otherGameProps,
      rand,
      nullptr,
      nullptr
    );
    gameData->printDebug(cout);
    if(testSgf) {
      WriteSgf::writeSgf(cout,"Black","White",gameData->endHist,gameData,false,false);
      cout << endl;
    }
    delete gameData;
    delete bot;
    cout << endl;
  };

  runMore("test policy surprise and scale together!",Rules::getTrompTaylorish(),true,false,true,false);
  runMore("test value surprise and scale together!",Rules::getTrompTaylorish(),false,true,true,false);
  runMore("test all three together!",Rules::getTrompTaylorish(),true,true,true,true);

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
    //Big giant test of certain playSettings parts and game initialization
    PlaySettings playSettings;
    //Not testing these - covered by other tests
    playSettings.initGamesWithPolicy = false;
    playSettings.sidePositionProb = 0;

    playSettings.compensateKomiVisits = 20;
    playSettings.fancyKomiVarying = true;

    playSettings.sekiForkHackProb = 0.04;
    playSettings.forSelfPlay = true;

    nnEval->clearCache();
    nnEval->clearStats();

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

    GameRunner* gameRunner = new GameRunner(cfg, "game init test game seed", playSettings, logger);
    auto shouldStop = []() { return false; };
    for(int i = 0; i<100; i++) {
      string seed = "game init test search seed:" + Global::int64ToString(i);
      FinishedGameData* data = gameRunner->runGame(seed, botSpec, botSpec, forkData, NULL, logger, shouldStop, nullptr, nullptr, false);
      cout << data->startHist.rules << endl;
      cout << "Start moves size " << data->startHist.moveHistory.size()
           << " Start pla " << PlayerIO::playerToString(data->startPla)
           << " XY " << data->startBoard.x_size << " " << data->startBoard.y_size
           << " Extra black " << data->numExtraBlack
           << " Draw equiv " << data->drawEquivalentWinsForWhite
           << " Mode " << data->mode
           << " BeganInEncorePhase " << data->beganInEncorePhase
           << " UsedInitialPosition " << data->usedInitialPosition
           << " Forkstuff " << forkData->forks.size() << " " << forkData->sekiForks.size()
           << endl;
      delete data;
    }
    delete gameRunner;
    delete forkData;
  }

  delete nnEval;
  NeuralNet::globalCleanup();
}


void Tests::runSekiTrainWriteTests(const string& modelFile) {
  cout << "Running test for how a seki gets recorded" << endl;
  NeuralNet::globalInitialize();

  int nnXLen = 13;
  int nnYLen = 13;

  Logger logger;
  logger.setLogToStdout(true);
  logger.setLogTime(false);

  NNEvaluator* nnEval = startNNEval(modelFile,"nneval",logger,0,true,false,false);

  auto run = [&](const string& sgfStr, const string& seedBase, const Rules& rules) {
    int inputsVersion = 6;
    int maxRows = 256;
    double firstFileMinRandProp = 1.0;
    int debugOnlyWriteEvery = 1000;
    TrainingDataWriter dataWriter(&cout,inputsVersion, maxRows, firstFileMinRandProp, nnXLen, nnYLen, debugOnlyWriteEvery, seedBase+"dwriter");

    nnEval->clearCache();
    nnEval->clearStats();

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
    int turnIdx = sgf->moves.size();
    sgf->setupBoardAndHistAssumeLegal(rules,initialBoard,initialPla,initialHist,turnIdx);

    bool doEndGameIfAllPassAlive = true;
    bool clearBotAfterSearch = true;
    int maxMovesPerGame = 1;
    auto shouldStop = []() { return false; };
    PlaySettings playSettings;
    playSettings.initGamesWithPolicy = false;
    playSettings.sidePositionProb = 0;
    playSettings.cheapSearchProb = 0;
    playSettings.cheapSearchVisits = 0;
    playSettings.cheapSearchTargetWeight = 0;
    playSettings.earlyForkGameProb = 0;
    playSettings.earlyForkGameExpectedMoveProp = 0;
    playSettings.forkGameMinChoices = 2;
    playSettings.earlyForkGameMaxChoices = 2;
    playSettings.compensateKomiVisits = 5;
    playSettings.forSelfPlay = true;

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
      maxMovesPerGame, shouldStop,
      playSettings, otherGameProps,
      rand,
      nullptr,
      nullptr
    );

    cout << "seedBase: " << seedBase << endl;
    gameData->endHist.printDebugInfo(cout,gameData->endHist.getRecentBoard(0));
    dataWriter.writeGame(*gameData);
    dataWriter.flushIfNonempty();
    delete gameData;
    delete bot;
    delete sgf;
    cout << endl;
  };

  vector<Rules> ruless = {
    Rules(Rules::KO_SIMPLE, Rules::SCORING_AREA, Rules::TAX_NONE, false, false, Rules::WHB_ZERO, false, 0.0f),
    Rules(Rules::KO_SIMPLE, Rules::SCORING_TERRITORY, Rules::TAX_NONE, false, false, Rules::WHB_ZERO, false, 0.0f),
    Rules(Rules::KO_SIMPLE, Rules::SCORING_AREA, Rules::TAX_SEKI, false, false, Rules::WHB_ZERO, false, 0.0f),
    Rules(Rules::KO_SIMPLE, Rules::SCORING_TERRITORY, Rules::TAX_SEKI, false, false, Rules::WHB_ZERO, false, 0.0f),
    Rules(Rules::KO_SIMPLE, Rules::SCORING_AREA, Rules::TAX_ALL, false, false, Rules::WHB_ZERO, false, 0.0f),
    Rules(Rules::KO_SIMPLE, Rules::SCORING_TERRITORY, Rules::TAX_ALL, false, false, Rules::WHB_ZERO, false, 0.0f),
  };

  string sgfStr = "(;KM[0.0]PB[]SZ[13]PW[]AP[Sabaki:0.43.3]CA[UTF-8];B[aj];W[bi];B[bk];W[cj];B[cl];W[dk];B[dm];W[el];B[dl];W[ek];B[ck];W[dj];B[bj];W[ci];B[al];W[bm];B[fm];W[em];B[fl];W[ai];B[fk];W[dh];B[fj];W[bl];B[gi];W[eg];B[hh];W[ff];B[ig];W[ge];B[jf];W[hd];B[fi];W[di];B[gh];W[dg];B[hg];W[fe];B[ke];W[ic];B[ld];W[jb];B[fh];W[he];B[je];W[jc];B[kd];W[ja];B[md];W[la];B[mb];W[ka];B[mc];W[gc];B[jh];W[cc];B[kk];W[cf];B[jk];W[dc];B[ej];W[ei];B[eh];W[fg];B[gg];W[gf];B[hf];W[ie];B[if];W[id];B[jd];W[kc];B[lb];W[kb];B[lc])";

  for(int r = 0; r<ruless.size(); r++) {
    run(sgfStr,"abc",ruless[r]);
  }

  sgfStr = "(;FF[4]CA[UTF-8]AP[GoGui:1.4.9]SZ[13]KM[0];B[jj];W[kd];B[lc];W[kc];B[ld];W[ke];B[lb];W[kb];B[la];W[mb];B[le];W[kf];B[lf];W[lg];B[kg];W[lh];B[jg];W[mc];B[mf];W[md];B[ji];W[kk];B[jk];W[kj];B[jl];W[kl];B[ki];W[li];B[ie];W[hd];B[id];W[hc];B[he];W[ic];B[ge];W[fc];B[fk];W[ee];B[fh];W[dg];B[dk];W[ci];B[cb];W[cc];B[bc];W[cd];B[bd];W[db];B[bb];W[ce];B[aa];W[ck];B[dj];W[cj];B[ka];W[jb];B[ja];W[ia];B[mg];W[mh];B[kh];W[lk];B[be];W[bf];B[cf];W[bg];B[ca];W[da];B[dc];W[ec];B[dd];W[de];B[ei];W[ff];B[ml];W[mk];B[lm];W[km];B[mj];W[lj];B[jm];W[dl];B[el];W[cl];B[gf];W[mi];B[fg];W[eg];B[fe];W[ef];B[fd];W[ed];B[af];W[ag];B[ae];W[jf];B[if];W[em];B[fm];W[dm];B[di];W[dh];B[gd];W[gc];B[jd];W[jc];B[eh];W[je];B[df];W[cg];B[ib])";

  for(int r = 0; r<ruless.size(); r++) {
    run(sgfStr,"def",ruless[r]);
  }

  {
    cout << "==============================================================" << endl;
    cout << "Also testing status logic inference!" << endl;
    SearchParams params;
    string searchRandSeed = "test statuses";
    Search* bot = new Search(params, nnEval, searchRandSeed);

    auto testStatuses = [&nnEval,&bot,&logger](const Board& board, const BoardHistory& hist, Player pla) {
      int numVisits = 50;
      vector<double> ownership = PlayUtils::computeOwnership(bot,board,hist,pla,numVisits,logger);
      vector<double> buf;
      vector<bool> isAlive = PlayUtils::computeAnticipatedStatusesWithOwnership(bot,board,hist,pla,numVisits,logger,buf);
      testAssert(bot->alwaysIncludeOwnerMap == false);
      cout << "Search assumes " << PlayerIO::playerToString(pla) << " first" << endl;
      cout << "Rules " << hist.rules << endl;
      cout << board << endl;
      for(int y = 0; y<board.y_size; y++) {
        for(int x = 0; x<board.x_size; x++) {
          Loc loc = Location::getLoc(x,y,board.x_size);
          if(board.colors[loc] == C_EMPTY)
            cout << ".";
          else
            cout << (isAlive[loc] ? "a" : "d");
        }
        cout << endl;
      }
      cout << endl;
      for(int y = 0; y<board.y_size; y++) {
        for(int x = 0; x<board.x_size; x++) {
          int pos = NNPos::xyToPos(x,y,bot->nnXLen);
          int ownershipValue = (int)round(100*ownership[pos]);
          string s;
          if(ownershipValue >= 99)
            s = "    W";
          else if(ownershipValue <= -99)
            s = "    B";
          else
            s = Global::strprintf(" %+4d", ownershipValue);
          cout << s;
        }
        cout << endl;
      }
      cout << endl;
    };

    {
      Board board = Board::parseBoard(9,9,R"%%(
.........
.o...o...
..x......
oooooooo.
xxxxxxxx.
ox.......
.ox.x..x.
oox.x....
.o.......
)%%");
      BoardHistory hist(board,P_BLACK,Rules::parseRules("tromp-taylor"),0);
      testStatuses(board,hist,P_BLACK);
      BoardHistory hist2(board,P_WHITE,Rules::parseRules("tromp-taylor"),0);
      testStatuses(board,hist2,P_WHITE);
    }
    //The neural net that we're using for this test actually produces a lot of nonsense because it doesn't
    //understand the seki. But that's okay, we'll just leave this test here anyways
    {
      Board board = Board::parseBoard(9,9,R"%%(
o.o.xxo.x
oooooxxxx
xxxxxoooo
....x....
xxx..o.o.
ooxxx.o..
.ooox.o..
xo.ox.xoo
.xxox.xx.
)%%");
      BoardHistory hist(board,P_WHITE,Rules::parseRules("tromp-taylor"),0);
      testStatuses(board,hist,P_WHITE);
      BoardHistory hist2(board,P_WHITE,Rules::parseRules("japanese"),0);
      testStatuses(board,hist2,P_WHITE);

    }

    delete bot;
    cout << "==============================================================" << endl;
  }

  delete nnEval;
  NeuralNet::globalCleanup();
}
