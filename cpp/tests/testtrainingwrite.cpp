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
    enabled_t::Auto,
    enabled_t::Auto,
    string(),
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
  bool inputsNHWC = true;
  bool useNHWC = false;
  TestCommon::overrideForBackends(inputsNHWC, useNHWC);

  cout << "Running training write tests" << endl;
  NeuralNet::globalInitialize();

  int maxRows = 256;
  double firstFileMinRandProp = 1.0;
  int debugOnlyWriteEvery = 5;

  const bool logToStdout = true;
  const bool logToStderr = false;
  const bool logTime = false;
  Logger logger(nullptr, logToStdout, logToStderr, logTime);

  auto run = [&](
    const string& seedBase, const Rules& rules,
    double drawEquivalentWinsForWhite, int inputsVersion,
    int nnXLen, int nnYLen,
    int boardXLen, int boardYLen,
    bool cheapLongSgf
  ) {
    TrainingDataWriter dataWriter(&cout,inputsVersion, maxRows, firstFileMinRandProp, nnXLen, nnYLen, debugOnlyWriteEvery, seedBase+"dwriter");

    NNEvaluator* nnEval = startNNEval("/dev/null",seedBase+"nneval",logger,0,inputsNHWC,useNHWC,false);

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
    extraBlackAndKomi.komiMean = rules.komi;
    extraBlackAndKomi.komiStdev = 0;
    bool doEndGameIfAllPassAlive = cheapLongSgf ? false : true;
    bool clearBotAfterSearch = true;
    int maxMovesPerGame = cheapLongSgf ? 200 : 40;
    auto shouldStop = []() noexcept { return false; };
    WaitableFlag* shouldPause = nullptr;
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
      shouldPause,
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

  //JP 9x1 game
  inputsVersion = 3;
  rules = Rules::getSimpleTerritory();
  run("testtrainingwrite-simpleterritory-sgf-c",rules,0.5,inputsVersion,9,1,9,1,true);

  NeuralNet::globalCleanup();
}


void Tests::runSelfplayInitTestsWithNN(const string& modelFile) {
  bool inputsNHWC = true;
  bool useNHWC = false;
  TestCommon::overrideForBackends(inputsNHWC, useNHWC);

  cout << "Running test for selfplay initialization with NN" << endl;
  NeuralNet::globalInitialize();

  const bool logToStdout = true;
  const bool logToStderr = false;
  const bool logTime = false;
  Logger logger(nullptr, logToStdout, logToStderr, logTime);

  NNEvaluator* nnEval = startNNEval(modelFile,"nneval",logger,0,inputsNHWC,useNHWC,false);

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
    extraBlackAndKomi.komiMean = rules.komi;
    extraBlackAndKomi.komiStdev = 0;
    extraBlackAndKomi.makeGameFair = numExtraBlack > 0 && !makeGameFairForEmptyBoard;
    extraBlackAndKomi.makeGameFairForEmptyBoard = makeGameFairForEmptyBoard;

    bool doEndGameIfAllPassAlive = true;
    bool clearBotAfterSearch = true;
    int maxMovesPerGame = 1;
    auto shouldStop = []() noexcept { return false; };
    WaitableFlag* shouldPause = nullptr;
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
    Search* bot = new Search(botSpec.baseParams, botSpec.nnEval, &logger, searchRandSeed);

    Rand rand(seedBase+"play");
    OtherGameProperties otherGameProps;
    FinishedGameData* gameData = Play::runGame(
      initialBoard,initialPla,initialHist,extraBlackAndKomi,
      botSpec,botSpec,
      bot,bot,
      doEndGameIfAllPassAlive, clearBotAfterSearch,
      logger, false, false,
      maxMovesPerGame, shouldStop,
      shouldPause,
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
        playSettings.cheapSearchVisits, OtherGameProperties(), rand
      );
      BoardHistory hist2 = forkData.forks[0]->hist;
      float oldKomi = hist2.rules.komi;
      double lead = PlayUtils::computeLead(
        bot, bot, board, hist2, pla,
        playSettings.cheapSearchVisits, OtherGameProperties()
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
  bool inputsNHWC = true;
  bool useNHWC = false;
  TestCommon::overrideForBackends(inputsNHWC, useNHWC);

  cout << "Running more tests for selfplay" << endl;
  NeuralNet::globalInitialize();

  const bool logToStdout = true;
  const bool logToStderr = false;
  const bool logTime = false;
  Logger logger(nullptr, logToStdout, logToStderr, logTime);

  NNEvaluator* nnEval = startNNEval(modelFile,"nneval",logger,0,inputsNHWC,useNHWC,false);

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
    extraBlackAndKomi.komiMean = rules.komi;
    extraBlackAndKomi.komiStdev = 0;
    extraBlackAndKomi.makeGameFair = false;
    extraBlackAndKomi.makeGameFairForEmptyBoard = false;

    bool doEndGameIfAllPassAlive = true;
    bool clearBotAfterSearch = true;
    int maxMovesPerGame = testResign ? 10000 : (testLead || testPolicySurpriseWeight || testValueSurpriseWeight) ? 30 : 15;
    auto shouldStop = []() noexcept { return false; };
    WaitableFlag* shouldPause = nullptr;
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
    Search* bot = new Search(botSpec.baseParams, botSpec.nnEval, &logger, searchRandSeed);

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
      otherGameProps.hintTurn = (int)initialHist.moveHistory.size();
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
      shouldPause,
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
        cout << forkData.forks[i]->board << endl;
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
    Search* bot = new Search(params, nnEval, &logger, searchRandSeed);

    rules.komi = komi;
    Player pla = P_BLACK;
    BoardHistory hist(board,pla,rules,0);
    int compensateKomiVisits = 50;
    OtherGameProperties otherGameProps;
    double lead = PlayUtils::computeLead(bot,bot,board,hist,pla,compensateKomiVisits,otherGameProps);
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
    extraBlackAndKomi.komiMean = rules.komi;
    extraBlackAndKomi.komiStdev = 0;
    extraBlackAndKomi.makeGameFair = false;
    extraBlackAndKomi.makeGameFairForEmptyBoard = false;

    bool doEndGameIfAllPassAlive = true;
    bool clearBotAfterSearch = true;
    int maxMovesPerGame = 20;
    auto shouldStop = []() noexcept { return false; };
    WaitableFlag* shouldPause = nullptr;
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
    Search* bot = new Search(botSpec.baseParams, botSpec.nnEval, &logger, searchRandSeed);

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
      shouldPause,
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
    auto shouldStop = []() noexcept { return false; };
    WaitableFlag* shouldPause = nullptr;
    for(int i = 0; i<100; i++) {
      string seed = "game init test search seed:" + Global::int64ToString(i);
      FinishedGameData* data = gameRunner->runGame(seed, botSpec, botSpec, forkData, NULL, logger, shouldStop, shouldPause, nullptr, nullptr, nullptr);
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


  {
    cout << "====================================================================================================" << endl;
    cout << "====================================================================================================" << endl;
    cout << "====================================================================================================" << endl;
    cout << "Running a 13x13 game in 5-move bursts with realistic visits and parameters to see training targets" << endl;

    nnEval->clearCache();
    nnEval->clearStats();

    string sgfData = TestCommon::getBenchmarkSGFData(13);
    CompactSgf* sgf = CompactSgf::parse(sgfData);

    SearchParams params = SearchParams::forTestsV1();
    params.rootNoiseEnabled = true;
    params.rootPolicyTemperatureEarly = 1.5;
    params.rootPolicyTemperature = 1.1;
    params.rootDesiredPerChildVisitsCoeff = 2.0;
    params.maxVisits = 800;
    params.drawEquivalentWinsForWhite = 0.5;

    MatchPairer::BotSpec botSpec;
    botSpec.botIdx = 0;
    botSpec.botName = string("test");
    botSpec.nnEval = nnEval;
    botSpec.baseParams = params;

    PlaySettings playSettings;
    playSettings.initGamesWithPolicy = false;
    playSettings.sidePositionProb = 0.0;
    playSettings.cheapSearchProb = 0.5;
    playSettings.cheapSearchVisits = 200;
    playSettings.cheapSearchTargetWeight = 0;
    playSettings.minAsymmetricCompensateKomiProb = 0.0;
    playSettings.compensateAfterPolicyInitProb = 0.0;
    playSettings.estimateLeadProb = 1.0;
    playSettings.earlyForkGameProb = 0.0;
    playSettings.forkGameProb = 0.0;

    playSettings.policySurpriseDataWeight = 0.5;
    playSettings.valueSurpriseDataWeight = 0.15;
    playSettings.noResolveTargetWeights = true;
    playSettings.allowResignation = false;
    playSettings.reduceVisits = false;
    playSettings.handicapAsymmetricPlayoutProb = 0;
    playSettings.normalAsymmetricPlayoutProb = 0;
    playSettings.sekiForkHackProb = 0;
    playSettings.fancyKomiVarying = false;

    playSettings.forSelfPlay = true;

    ExtraBlackAndKomi extraBlackAndKomi;
    extraBlackAndKomi.extraBlack = 0;
    extraBlackAndKomi.komiMean = rules.komi;
    extraBlackAndKomi.komiStdev = 0;
    extraBlackAndKomi.makeGameFair = false;
    extraBlackAndKomi.makeGameFairForEmptyBoard = false;

    vector<Move> moves = sgf->moves;

    Rules initialRules = Rules::parseRules("chinese");
    Board board;
    Player nextPla;
    BoardHistory hist;
    sgf->setupInitialBoardAndHist(initialRules, board, nextPla, hist);
    for(size_t i = 0; i<moves.size(); i++) {
      if(i % 10 == 0) {
        bool doEndGameIfAllPassAlive = true;
        bool clearBotAfterSearch = true;
        int maxMovesPerGame = 5;
        auto shouldStop = []() noexcept { return false; };
        WaitableFlag* shouldPause = nullptr;

        string searchRandSeed = "target testing" + Global::intToString((int)i);
        Search* bot = new Search(botSpec.baseParams, botSpec.nnEval, &logger, searchRandSeed);

        Rand rand(searchRandSeed + "rand");
        OtherGameProperties otherGameProps;
        bool logSearchInfo = false;
        FinishedGameData* gameData = Play::runGame(
          board,nextPla,hist,extraBlackAndKomi,
          botSpec,botSpec,
          bot,bot,
          doEndGameIfAllPassAlive, clearBotAfterSearch,
          logger, logSearchInfo, false,
          maxMovesPerGame, shouldStop,
          shouldPause,
          playSettings, otherGameProps,
          rand,
          nullptr,
          nullptr
        );
        gameData->printDebug(cout);
        delete gameData;
        delete bot;
      }

      bool suc = hist.makeBoardMoveTolerant(board,moves[i].loc,moves[i].pla);
      testAssert(suc);
      nextPla = getOpp(nextPla);
    }

    delete sgf;
  }

  {
    cout << "====================================================================================================" << endl;
    cout << "====================================================================================================" << endl;
    cout << "====================================================================================================" << endl;
    cout << "Testing turnnumber and early temperatures" << endl;

    int maxRows = 256;
    double firstFileMinRandProp = 1.0;
    int debugOnlyWriteEvery = 1;
    int inputsVersion = 7;

    SearchParams params = SearchParams::forTestsV2();
    params.rootNoiseEnabled = true;
    params.cpuctExploration = 6.0;
    params.useLcbForSelection = false;
    params.rootPolicyTemperatureEarly = 4.0;
    params.chosenMoveTemperatureEarly = 8.0;
    params.rootPolicyTemperature = 1.1;
    params.chosenMoveTemperature = 0.1;
    params.chosenMoveTemperatureHalflife = 20.0;
    params.maxVisits = 50;
    params.drawEquivalentWinsForWhite = 0.5;

    PlaySettings playSettings;
    playSettings.initGamesWithPolicy = false;
    playSettings.sidePositionProb = 0.0;
    playSettings.cheapSearchProb = 0.5;
    playSettings.cheapSearchVisits = 20;
    playSettings.cheapSearchTargetWeight = 0;
    playSettings.minAsymmetricCompensateKomiProb = 0.0;
    playSettings.compensateAfterPolicyInitProb = 0.0;
    playSettings.estimateLeadProb = 1.0;
    playSettings.earlyForkGameProb = 0.0;
    playSettings.forkGameProb = 0.0;

    playSettings.policySurpriseDataWeight = 0.0;
    playSettings.valueSurpriseDataWeight = 0.0;
    playSettings.noResolveTargetWeights = false;
    playSettings.allowResignation = false;
    playSettings.reduceVisits = false;
    playSettings.handicapAsymmetricPlayoutProb = 0;
    playSettings.normalAsymmetricPlayoutProb = 0;
    playSettings.sekiForkHackProb = 0;
    playSettings.fancyKomiVarying = false;

    playSettings.scaleDataWeight = 1.5;

    playSettings.forSelfPlay = true;

    std::map<string,string> cfgParams({
        std::make_pair("maxMovesPerGame","20"),
        std::make_pair("logSearchInfo","true"),
        std::make_pair("logMoves","false"),
        std::make_pair("komiAuto","true"),
        std::make_pair("komiStdev","2.0"),
        std::make_pair("handicapProb","0.0"),
        std::make_pair("handicapCompensateKomiProb","1.0"),
        std::make_pair("sgfCompensateKomiProb","1.0"),
        std::make_pair("forkCompensateKomiProb","1.0"),
        std::make_pair("komiBigStdevProb","0.0"),
        std::make_pair("komiBigStdev","20.0"),
        std::make_pair("drawRandRadius","0.5"),
        std::make_pair("noResultStdev","0.16"),

        std::make_pair("bSizes","9"),
        std::make_pair("bSizeRelProbs","1"),
        std::make_pair("koRules","SIMPLE"),
        std::make_pair("scoringRules","TERRITORY"),
        std::make_pair("taxRules","NONE"),
        std::make_pair("multiStoneSuicideLegals","false"),
        std::make_pair("hasButtons","false"),
        std::make_pair("allowRectangleProb","0.0"),
    });

    MatchPairer::BotSpec botSpec;
    botSpec.botIdx = 0;
    botSpec.botName = "testbot";
    botSpec.nnEval = nnEval;
    botSpec.baseParams = params;

    string seed = "seed-testing-temperature";

    {
      cout << "Turn number initial 0 selfplay with high temperatures" << endl;
      nnEval->clearCache();
      nnEval->clearStats();

      ConfigParser cfg(cfgParams);
      ForkData* forkData = new ForkData();
      GameRunner* gameRunner = new GameRunner(cfg, seed, playSettings, logger);
      auto shouldStop = []() noexcept { return false; };
      WaitableFlag* shouldPause = nullptr;
      TrainingDataWriter dataWriter(&cout,inputsVersion, maxRows, firstFileMinRandProp, 9, 9, debugOnlyWriteEvery, seed);

      Sgf::PositionSample startPosSample;
      startPosSample.board = Board(9,9);
      startPosSample.nextPla = P_BLACK;
      startPosSample.moves = std::vector<Move>();
      startPosSample.initialTurnNumber = 0;
      startPosSample.hintLoc = Board::NULL_LOC;
      startPosSample.weight = 10.0;
      startPosSample.trainingWeight = 0.35;

      FinishedGameData* data = gameRunner->runGame(seed, botSpec, botSpec, forkData, &startPosSample, logger, shouldStop, shouldPause, nullptr, nullptr, nullptr);
      data->printDebug(cout);

      dataWriter.writeGame(*data);
      dataWriter.flushIfNonempty();

      delete data;
      delete gameRunner;
      delete forkData;
    }
    {
      cout << "Turn number initial 40 selfplay with high early temperatures" << endl;
      nnEval->clearCache();
      nnEval->clearStats();

      ConfigParser cfg(cfgParams);
      ForkData* forkData = new ForkData();
      GameRunner* gameRunner = new GameRunner(cfg, seed, playSettings, logger);
      auto shouldStop = []() noexcept { return false; };
      WaitableFlag* shouldPause = nullptr;
      TrainingDataWriter dataWriter(&cout,inputsVersion, maxRows, firstFileMinRandProp, 9, 9, debugOnlyWriteEvery, seed);

      Sgf::PositionSample startPosSample;
      startPosSample.board = Board(9,9);
      startPosSample.nextPla = P_BLACK;
      startPosSample.moves = std::vector<Move>();
      startPosSample.initialTurnNumber = 40;
      startPosSample.hintLoc = Board::NULL_LOC;
      startPosSample.weight = 10.0;
      startPosSample.trainingWeight = 0.35;

      FinishedGameData* data = gameRunner->runGame(seed, botSpec, botSpec, forkData, &startPosSample, logger, shouldStop, shouldPause, nullptr, nullptr, nullptr);
      data->printDebug(cout);

      dataWriter.writeGame(*data);
      dataWriter.flushIfNonempty();

      delete data;
      delete gameRunner;
      delete forkData;
    }
  }


  {
    cout << "====================================================================================================" << endl;
    cout << "====================================================================================================" << endl;
    cout << "====================================================================================================" << endl;
    cout << "Testing no result" << endl;

    int maxRows = 256;
    double firstFileMinRandProp = 1.0;
    int debugOnlyWriteEvery = 1;
    int inputsVersion = 7;

    SearchParams params = SearchParams::forTestsV2();
    params.rootNoiseEnabled = true;
    params.chosenMoveTemperatureEarly = 0.1;
    params.chosenMoveTemperature = 0.1;
    params.maxVisits = 100;
    params.drawEquivalentWinsForWhite = 0.5;

    PlaySettings playSettings;
    playSettings.initGamesWithPolicy = false;
    playSettings.sidePositionProb = 0.3;
    playSettings.cheapSearchProb = 0.5;
    playSettings.cheapSearchVisits = 40;
    playSettings.cheapSearchTargetWeight = 0;
    playSettings.minAsymmetricCompensateKomiProb = 0.0;
    playSettings.compensateAfterPolicyInitProb = 0.0;
    playSettings.estimateLeadProb = 1.0;
    playSettings.earlyForkGameProb = 0.0;
    playSettings.forkGameProb = 0.0;

    playSettings.policySurpriseDataWeight = 0.0;
    playSettings.valueSurpriseDataWeight = 0.0;
    playSettings.noResolveTargetWeights = false;
    playSettings.allowResignation = false;
    playSettings.reduceVisits = false;
    playSettings.handicapAsymmetricPlayoutProb = 0;
    playSettings.normalAsymmetricPlayoutProb = 0;
    playSettings.sekiForkHackProb = 0;
    playSettings.fancyKomiVarying = false;

    playSettings.scaleDataWeight = 1.5;

    playSettings.forSelfPlay = true;

    std::map<string,string> cfgParams({
        std::make_pair("maxMovesPerGame","200"),
        std::make_pair("logSearchInfo","true"),
        std::make_pair("logMoves","false"),
        std::make_pair("komiAuto","true"),
        std::make_pair("komiStdev","3.0"),
        std::make_pair("handicapProb","0.0"),
        std::make_pair("handicapCompensateKomiProb","1.0"),
        std::make_pair("sgfCompensateKomiProb","1.0"),
        std::make_pair("forkCompensateKomiProb","1.0"),
        std::make_pair("komiBigStdevProb","0.0"),
        std::make_pair("komiBigStdev","20.0"),
        std::make_pair("drawRandRadius","0.5"),
        std::make_pair("noResultStdev","0.16"),

        std::make_pair("bSizes","9"),
        std::make_pair("bSizeRelProbs","1"),
        std::make_pair("koRules","SIMPLE"),
        std::make_pair("scoringRules","AREA"),
        std::make_pair("taxRules","SEKI"),
        std::make_pair("multiStoneSuicideLegals","false"),
        std::make_pair("hasButtons","false"),
        std::make_pair("allowRectangleProb","0.0"),
    });

    MatchPairer::BotSpec botSpec;
    botSpec.botIdx = 0;
    botSpec.botName = "testbot";
    botSpec.nnEval = nnEval;
    botSpec.baseParams = params;

    string seed = "seed-testing-temperature";

    {
      cout << "Turn number initial 0 selfplay with high temperatures" << endl;
      nnEval->clearCache();
      nnEval->clearStats();

      ConfigParser cfg(cfgParams);
      ForkData* forkData = new ForkData();
      GameRunner* gameRunner = new GameRunner(cfg, seed, playSettings, logger);
      auto shouldStop = []() noexcept { return false; };
      WaitableFlag* shouldPause = nullptr;
      TrainingDataWriter dataWriter(&cout,inputsVersion, maxRows, firstFileMinRandProp, 9, 9, debugOnlyWriteEvery, seed);

      Sgf::PositionSample startPosSample;
      startPosSample.board = Board::parseBoard(9,9,R"%%(
.........
......o..
xxxxxxx..
oooooox..
xo.o.oxx.
.xoxoxoo.
xxxxxxo..
.oooooox.
.........
)%%");
      startPosSample.nextPla = P_WHITE;
      startPosSample.moves = std::vector<Move>();
      startPosSample.initialTurnNumber = 0;
      startPosSample.hintLoc = Board::NULL_LOC;
      startPosSample.weight = 10.0;
      startPosSample.trainingWeight = 0.72;

      FinishedGameData* data = gameRunner->runGame(seed, botSpec, botSpec, forkData, &startPosSample, logger, shouldStop, shouldPause, nullptr, nullptr, nullptr);
      data->printDebug(cout);

      dataWriter.writeGame(*data);
      dataWriter.flushIfNonempty();

      delete data;
      delete gameRunner;
      delete forkData;
    }
  }

  delete nnEval;
  NeuralNet::globalCleanup();
}


void Tests::runSelfplayStatTestsWithNN(const string& modelFile) {
  bool inputsNHWC = true;
  bool useNHWC = false;
  TestCommon::overrideForBackends(inputsNHWC, useNHWC);

  cout << "Running 10b tests for selfplay" << endl;
  NeuralNet::globalInitialize();

  const bool logToStdout = true;
  const bool logToStderr = false;
  const bool logTime = false;
  Logger logger(nullptr, logToStdout, logToStderr, logTime);

  NNEvaluator* nnEval = startNNEval(modelFile,"nneval",logger,0,inputsNHWC,useNHWC,false);

  auto runStatTest = [&](const std::map<string,string>& cfgParams, PlaySettings playSettings, const Sgf::PositionSample* startPosSample, string name, int numSamples) {
    cout << "--------------------------------------------------------------------------------------" << endl;
    cout << name << endl;
    nnEval->clearCache();
    nnEval->clearStats();

    MatchPairer::BotSpec botSpec;
    botSpec.botIdx = 0;
    botSpec.botName = modelFile;
    botSpec.nnEval = nnEval;
    botSpec.baseParams = SearchParams();
    botSpec.baseParams.maxVisits = 10;

    ConfigParser cfg(cfgParams);
    ForkData* forkData = new ForkData();
    GameRunner* gameRunner = new GameRunner(cfg, "game init stattest1", playSettings, logger);
    auto shouldStop = []() noexcept { return false; };
    WaitableFlag* shouldPause = nullptr;

    std::map<float,int> komiDistribution;
    std::map<int,int> bStoneDistribution;
    std::map<int,int> wStoneDistribution;
    std::map<string,int> bSizeDistribution;
    std::map<int,int> turnDistribution;
    for(int i = 0; i<numSamples; i++) {
      string seed = name + Global::int64ToString(i);
      FinishedGameData* data = gameRunner->runGame(seed, botSpec, botSpec, forkData, startPosSample, logger, shouldStop, shouldPause, nullptr, nullptr, nullptr);
      komiDistribution[data->startHist.rules.komi] += 1;
      bStoneDistribution[data->startBoard.numPlaStonesOnBoard(P_BLACK)] += 1;
      wStoneDistribution[data->startBoard.numPlaStonesOnBoard(P_WHITE)] += 1;
      bSizeDistribution[Global::intToString(data->startBoard.x_size) + "x" + Global::intToString(data->startBoard.y_size)] += 1;
      turnDistribution[data->startHist.getCurrentTurnNumber()] += 1;
      delete data;
    }
    cout << "komiDistribution" << endl;
    for(auto iter = komiDistribution.begin(); iter != komiDistribution.end(); ++iter) {
      cout << Global::strprintf("%+5.1f",iter->first) << " " << iter->second << endl;
    }
    cout << "bStoneDistribution" << endl;
    for(auto iter = bStoneDistribution.begin(); iter != bStoneDistribution.end(); ++iter) {
      cout << Global::strprintf("%3d",iter->first) << " " << iter->second << endl;
    }
    cout << "wStoneDistribution" << endl;
    for(auto iter = wStoneDistribution.begin(); iter != wStoneDistribution.end(); ++iter) {
      cout << Global::strprintf("%3d",iter->first) << " " << iter->second << endl;
    }
    cout << "bSizeDistribution" << endl;
    for(auto iter = bSizeDistribution.begin(); iter != bSizeDistribution.end(); ++iter) {
      cout << iter->first << " " << iter->second << endl;
    }
    cout << "turnDistribution" << endl;
    for(auto iter = turnDistribution.begin(); iter != turnDistribution.end(); ++iter) {
      cout << iter->first << " " << iter->second << endl;
    }
    delete gameRunner;
    delete forkData;
  };

  {
    string name = "Game init test 19x19 mostly vanilla";
    //Statistical test of game initialization
    PlaySettings playSettings;
    playSettings.initGamesWithPolicy = true;
    playSettings.policyInitAreaProp = 0.0;
    playSettings.compensateAfterPolicyInitProb = 1.0;
    playSettings.sidePositionProb = 0;
    playSettings.compensateKomiVisits = 4;
    playSettings.estimateLeadProb = 0.0;
    playSettings.fancyKomiVarying = true;
    playSettings.sekiForkHackProb = 0.0;
    playSettings.earlyForkGameProb = 0.0;
    playSettings.forkGameMinChoices = 2;
    playSettings.earlyForkGameMaxChoices = 3;
    playSettings.forSelfPlay = true;

    std::map<string,string> cfgParams({
        std::make_pair("maxMovesPerGame","0"),
        std::make_pair("logSearchInfo","false"),
        std::make_pair("logMoves","false"),
        std::make_pair("komiAuto","true"),
        std::make_pair("komiStdev","2.0"),
        std::make_pair("handicapProb","0.0"),
        std::make_pair("handicapCompensateKomiProb","1.0"),
        std::make_pair("forkCompensateKomiProb","1.0"),
        std::make_pair("komiBigStdevProb","0.0"),
        std::make_pair("komiBigStdev","20.0"),
        std::make_pair("drawRandRadius","0.5"),
        std::make_pair("noResultStdev","0.16"),

        std::make_pair("bSizes","19"),
        std::make_pair("bSizeRelProbs","1"),
        std::make_pair("koRules","SIMPLE,POSITIONAL,SITUATIONAL"),
        std::make_pair("scoringRules","AREA"),
        std::make_pair("taxRules","NONE,NONE,SEKI,SEKI,ALL"),
        std::make_pair("multiStoneSuicideLegals","false,true"),
        std::make_pair("hasButtons","false,false,true"),
        std::make_pair("allowRectangleProb","0.0"),
    });
    runStatTest(cfgParams,playSettings,NULL,name,100);
  }

  {
    string name = "Game init test 19x19 mostly with territory rules";
    //Statistical test of game initialization
    PlaySettings playSettings;
    playSettings.initGamesWithPolicy = true;
    playSettings.policyInitAreaProp = 0.0;
    playSettings.compensateAfterPolicyInitProb = 1.0;
    playSettings.sidePositionProb = 0;
    playSettings.compensateKomiVisits = 4;
    playSettings.estimateLeadProb = 0.0;
    playSettings.fancyKomiVarying = true;
    playSettings.sekiForkHackProb = 0.0;
    playSettings.earlyForkGameProb = 0.0;
    playSettings.forkGameMinChoices = 2;
    playSettings.earlyForkGameMaxChoices = 3;
    playSettings.forSelfPlay = true;

    std::map<string,string> cfgParams({
        std::make_pair("maxMovesPerGame","0"),
        std::make_pair("logSearchInfo","false"),
        std::make_pair("logMoves","false"),
        std::make_pair("komiAuto","true"),
        std::make_pair("komiStdev","2.0"),
        std::make_pair("handicapProb","0.0"),
        std::make_pair("handicapCompensateKomiProb","1.0"),
        std::make_pair("forkCompensateKomiProb","1.0"),
        std::make_pair("komiBigStdevProb","0.0"),
        std::make_pair("komiBigStdev","20.0"),
        std::make_pair("drawRandRadius","0.5"),
        std::make_pair("noResultStdev","0.16"),

        std::make_pair("bSizes","19"),
        std::make_pair("bSizeRelProbs","1"),
        std::make_pair("koRules","SIMPLE,POSITIONAL,SITUATIONAL"),
        std::make_pair("scoringRules","AREA,TERRITORY"),
        std::make_pair("taxRules","NONE,NONE,SEKI,SEKI,ALL"),
        std::make_pair("multiStoneSuicideLegals","false,true"),
        std::make_pair("hasButtons","false,false,true"),
        std::make_pair("allowRectangleProb","0.0"),
    });
    runStatTest(cfgParams,playSettings,NULL,name,100);
  }

  {
    string name = "Game init test 19x19 komiBigStdev";
    //Statistical test of game initialization
    PlaySettings playSettings;
    playSettings.initGamesWithPolicy = true;
    playSettings.policyInitAreaProp = 0.0;
    playSettings.compensateAfterPolicyInitProb = 1.0;
    playSettings.sidePositionProb = 0;
    playSettings.compensateKomiVisits = 4;
    playSettings.estimateLeadProb = 0.0;
    playSettings.fancyKomiVarying = true;
    playSettings.sekiForkHackProb = 0.0;
    playSettings.earlyForkGameProb = 0.0;
    playSettings.forkGameMinChoices = 2;
    playSettings.earlyForkGameMaxChoices = 3;
    playSettings.forSelfPlay = true;

    std::map<string,string> cfgParams({
        std::make_pair("maxMovesPerGame","0"),
        std::make_pair("logSearchInfo","false"),
        std::make_pair("logMoves","false"),
        std::make_pair("komiAuto","true"),
        std::make_pair("komiStdev","2.0"),
        std::make_pair("handicapProb","0.0"),
        std::make_pair("handicapCompensateKomiProb","1.0"),
        std::make_pair("forkCompensateKomiProb","1.0"),
        std::make_pair("komiBigStdevProb","0.3"),
        std::make_pair("komiBigStdev","20.0"),
        std::make_pair("drawRandRadius","0.5"),
        std::make_pair("noResultStdev","0.16"),

        std::make_pair("bSizes","19"),
        std::make_pair("bSizeRelProbs","1"),
        std::make_pair("koRules","SIMPLE,POSITIONAL,SITUATIONAL"),
        std::make_pair("scoringRules","AREA"),
        std::make_pair("taxRules","NONE,NONE,SEKI,SEKI,ALL"),
        std::make_pair("multiStoneSuicideLegals","false,true"),
        std::make_pair("hasButtons","false,false,true"),
        std::make_pair("allowRectangleProb","0.0"),
    });
    runStatTest(cfgParams,playSettings,NULL,name,100);
  }

  {
    string name = "Game init test 9x9 komiBigStdev";
    //Statistical test of game initialization
    PlaySettings playSettings;
    playSettings.initGamesWithPolicy = true;
    playSettings.policyInitAreaProp = 0.0;
    playSettings.compensateAfterPolicyInitProb = 1.0;
    playSettings.sidePositionProb = 0;
    playSettings.compensateKomiVisits = 4;
    playSettings.estimateLeadProb = 0.0;
    playSettings.fancyKomiVarying = true;
    playSettings.sekiForkHackProb = 0.0;
    playSettings.earlyForkGameProb = 0.0;
    playSettings.forkGameMinChoices = 2;
    playSettings.earlyForkGameMaxChoices = 3;
    playSettings.forSelfPlay = true;

    std::map<string,string> cfgParams({
        std::make_pair("maxMovesPerGame","0"),
        std::make_pair("logSearchInfo","false"),
        std::make_pair("logMoves","false"),
        std::make_pair("komiAuto","true"),
        std::make_pair("komiStdev","2.0"),
        std::make_pair("handicapProb","0.0"),
        std::make_pair("handicapCompensateKomiProb","1.0"),
        std::make_pair("forkCompensateKomiProb","1.0"),
        std::make_pair("komiBigStdevProb","0.3"),
        std::make_pair("komiBigStdev","20.0"),
        std::make_pair("drawRandRadius","0.5"),
        std::make_pair("noResultStdev","0.16"),

        std::make_pair("bSizes","9"),
        std::make_pair("bSizeRelProbs","1"),
        std::make_pair("koRules","SIMPLE,POSITIONAL,SITUATIONAL"),
        std::make_pair("scoringRules","AREA"),
        std::make_pair("taxRules","NONE,NONE,SEKI,SEKI,ALL"),
        std::make_pair("multiStoneSuicideLegals","false,true"),
        std::make_pair("hasButtons","false,false,true"),
        std::make_pair("allowRectangleProb","0.0"),
    });
    runStatTest(cfgParams,playSettings,NULL,name,100);
  }

  {
    string name = "Game init test 19x19 policy init and make fair low temp";
    //Statistical test of game initialization
    PlaySettings playSettings;
    playSettings.initGamesWithPolicy = true;
    playSettings.policyInitAreaProp = 0.05;
    playSettings.policyInitAreaTemperature = 0.3;
    playSettings.compensateAfterPolicyInitProb = 1.0;
    playSettings.sidePositionProb = 0;
    playSettings.compensateKomiVisits = 4;
    playSettings.estimateLeadProb = 0.0;
    playSettings.fancyKomiVarying = true;
    playSettings.sekiForkHackProb = 0.0;
    playSettings.earlyForkGameProb = 0.0;
    playSettings.forkGameMinChoices = 2;
    playSettings.earlyForkGameMaxChoices = 3;
    playSettings.forSelfPlay = true;

    std::map<string,string> cfgParams({
        std::make_pair("maxMovesPerGame","0"),
        std::make_pair("logSearchInfo","false"),
        std::make_pair("logMoves","false"),
        std::make_pair("komiAuto","true"),
        std::make_pair("komiStdev","2.0"),
        std::make_pair("handicapProb","0.0"),
        std::make_pair("handicapCompensateKomiProb","1.0"),
        std::make_pair("forkCompensateKomiProb","1.0"),
        std::make_pair("komiBigStdevProb","0.0"),
        std::make_pair("komiBigStdev","20.0"),
        std::make_pair("drawRandRadius","0.5"),
        std::make_pair("noResultStdev","0.16"),

        std::make_pair("bSizes","19"),
        std::make_pair("bSizeRelProbs","1"),
        std::make_pair("koRules","SIMPLE,POSITIONAL,SITUATIONAL"),
        std::make_pair("scoringRules","AREA"),
        std::make_pair("taxRules","NONE,NONE,SEKI,SEKI,ALL"),
        std::make_pair("multiStoneSuicideLegals","false,true"),
        std::make_pair("hasButtons","false,false,true"),
        std::make_pair("allowRectangleProb","0.0"),
    });
    runStatTest(cfgParams,playSettings,NULL,name,100);
  }

  {
    string name = "Game init test 19x19 policy init and make fair";
    //Statistical test of game initialization
    PlaySettings playSettings;
    playSettings.initGamesWithPolicy = true;
    playSettings.policyInitAreaProp = 0.05;
    playSettings.compensateAfterPolicyInitProb = 1.0;
    playSettings.sidePositionProb = 0;
    playSettings.compensateKomiVisits = 4;
    playSettings.estimateLeadProb = 0.0;
    playSettings.fancyKomiVarying = true;
    playSettings.sekiForkHackProb = 0.0;
    playSettings.earlyForkGameProb = 0.0;
    playSettings.forkGameMinChoices = 2;
    playSettings.earlyForkGameMaxChoices = 3;
    playSettings.forSelfPlay = true;

    std::map<string,string> cfgParams({
        std::make_pair("maxMovesPerGame","0"),
        std::make_pair("logSearchInfo","false"),
        std::make_pair("logMoves","false"),
        std::make_pair("komiAuto","true"),
        std::make_pair("komiStdev","2.0"),
        std::make_pair("handicapProb","0.0"),
        std::make_pair("handicapCompensateKomiProb","1.0"),
        std::make_pair("forkCompensateKomiProb","1.0"),
        std::make_pair("komiBigStdevProb","0.0"),
        std::make_pair("komiBigStdev","20.0"),
        std::make_pair("drawRandRadius","0.5"),
        std::make_pair("noResultStdev","0.16"),

        std::make_pair("bSizes","19"),
        std::make_pair("bSizeRelProbs","1"),
        std::make_pair("koRules","SIMPLE,POSITIONAL,SITUATIONAL"),
        std::make_pair("scoringRules","AREA"),
        std::make_pair("taxRules","NONE,NONE,SEKI,SEKI,ALL"),
        std::make_pair("multiStoneSuicideLegals","false,true"),
        std::make_pair("hasButtons","false,false,true"),
        std::make_pair("allowRectangleProb","0.0"),
    });
    runStatTest(cfgParams,playSettings,NULL,name,100);
  }

  {
    string name = "Game init test 19x19 policy init and half make fair";
    //Statistical test of game initialization
    PlaySettings playSettings;
    playSettings.initGamesWithPolicy = true;
    playSettings.policyInitAreaProp = 0.05;
    playSettings.compensateAfterPolicyInitProb = 0.5;
    playSettings.sidePositionProb = 0;
    playSettings.compensateKomiVisits = 4;
    playSettings.estimateLeadProb = 0.0;
    playSettings.fancyKomiVarying = true;
    playSettings.sekiForkHackProb = 0.0;
    playSettings.earlyForkGameProb = 0.0;
    playSettings.forkGameMinChoices = 2;
    playSettings.earlyForkGameMaxChoices = 3;
    playSettings.forSelfPlay = true;

    std::map<string,string> cfgParams({
        std::make_pair("maxMovesPerGame","0"),
        std::make_pair("logSearchInfo","false"),
        std::make_pair("logMoves","false"),
        std::make_pair("komiAuto","true"),
        std::make_pair("komiStdev","2.0"),
        std::make_pair("handicapProb","0.0"),
        std::make_pair("handicapCompensateKomiProb","1.0"),
        std::make_pair("forkCompensateKomiProb","1.0"),
        std::make_pair("komiBigStdevProb","0.0"),
        std::make_pair("komiBigStdev","20.0"),
        std::make_pair("drawRandRadius","0.5"),
        std::make_pair("noResultStdev","0.16"),

        std::make_pair("bSizes","19"),
        std::make_pair("bSizeRelProbs","1"),
        std::make_pair("koRules","SIMPLE,POSITIONAL,SITUATIONAL"),
        std::make_pair("scoringRules","AREA"),
        std::make_pair("taxRules","NONE,NONE,SEKI,SEKI,ALL"),
        std::make_pair("multiStoneSuicideLegals","false,true"),
        std::make_pair("hasButtons","false,false,true"),
        std::make_pair("allowRectangleProb","0.0"),
    });
    runStatTest(cfgParams,playSettings,NULL,name,100);
  }

  {
    string name = "Game handcap bigger stdev";
    //Statistical test of game initialization
    PlaySettings playSettings;
    playSettings.initGamesWithPolicy = false;
    playSettings.policyInitAreaProp = 0.00;
    playSettings.compensateAfterPolicyInitProb = 0.0;
    playSettings.sidePositionProb = 0;
    playSettings.compensateKomiVisits = 4;
    playSettings.estimateLeadProb = 0.0;
    playSettings.fancyKomiVarying = true;
    playSettings.sekiForkHackProb = 0.0;
    playSettings.earlyForkGameProb = 0.0;
    playSettings.forkGameMinChoices = 2;
    playSettings.earlyForkGameMaxChoices = 3;
    playSettings.forSelfPlay = true;

    std::map<string,string> cfgParams({
        std::make_pair("maxMovesPerGame","0"),
        std::make_pair("logSearchInfo","false"),
        std::make_pair("logMoves","false"),
        std::make_pair("komiAuto","true"),
        std::make_pair("komiStdev","0.5"),
        std::make_pair("handicapProb","1.0"),
        std::make_pair("handicapCompensateKomiProb","0.5"),
        std::make_pair("forkCompensateKomiProb","1.0"),
        std::make_pair("komiBigStdevProb","0.50"),
        std::make_pair("komiBigStdev","5.0"),
        std::make_pair("komiBiggerStdevProb","0.10"),
        std::make_pair("komiBiggerStdev","100.0"),
        std::make_pair("drawRandRadius","0.5"),
        std::make_pair("noResultStdev","0.16"),

        std::make_pair("bSizes","14"),
        std::make_pair("bSizeRelProbs","1"),
        std::make_pair("koRules","SIMPLE,POSITIONAL,SITUATIONAL"),
        std::make_pair("scoringRules","AREA"),
        std::make_pair("taxRules","NONE,NONE,SEKI,SEKI,ALL"),
        std::make_pair("multiStoneSuicideLegals","false,true"),
        std::make_pair("hasButtons","false,false,true"),
        std::make_pair("allowRectangleProb","0.0"),
    });
    runStatTest(cfgParams,playSettings,NULL,name,100);
  }

  {
    string name = "Game handcap consistent";
    //Statistical test of game initialization
    PlaySettings playSettings;
    playSettings.initGamesWithPolicy = false;
    playSettings.policyInitAreaProp = 0.00;
    playSettings.compensateAfterPolicyInitProb = 0.0;
    playSettings.sidePositionProb = 0;
    playSettings.compensateKomiVisits = 4;
    playSettings.estimateLeadProb = 0.0;
    playSettings.fancyKomiVarying = true;
    playSettings.sekiForkHackProb = 0.0;
    playSettings.earlyForkGameProb = 0.0;
    playSettings.forkGameMinChoices = 2;
    playSettings.earlyForkGameMaxChoices = 3;
    playSettings.forSelfPlay = true;

    std::map<string,string> cfgParams({
        std::make_pair("maxMovesPerGame","0"),
        std::make_pair("logSearchInfo","false"),
        std::make_pair("logMoves","false"),
        std::make_pair("komiAuto","true"),
        std::make_pair("komiStdev","0.1"),
        std::make_pair("handicapProb","0.5"),
        std::make_pair("handicapCompensateKomiProb","1.0"),
        std::make_pair("forkCompensateKomiProb","1.0"),
        std::make_pair("komiBigStdevProb","0.0"),
        std::make_pair("komiBigStdev","5.0"),
        std::make_pair("komiBiggerStdevProb","0.0"),
        std::make_pair("komiBiggerStdev","100.0"),
        std::make_pair("sgfKomiInterpZeroProb","0.0"),
        std::make_pair("handicapKomiInterpZeroProb","0.0"),
        std::make_pair("drawRandRadius","0.5"),
        std::make_pair("noResultStdev","0.16"),

        std::make_pair("bSizes","14"),
        std::make_pair("bSizeRelProbs","1"),
        std::make_pair("koRules","SIMPLE,POSITIONAL,SITUATIONAL"),
        std::make_pair("scoringRules","AREA"),
        std::make_pair("taxRules","NONE,NONE,SEKI,SEKI,ALL"),
        std::make_pair("multiStoneSuicideLegals","false,true"),
        std::make_pair("hasButtons","false,false,true"),
        std::make_pair("allowRectangleProb","0.0"),
    });
    runStatTest(cfgParams,playSettings,NULL,name,100);
  }

  {
    string name = "Game handcap consistent interpZero";
    //Statistical test of game initialization
    PlaySettings playSettings;
    playSettings.initGamesWithPolicy = false;
    playSettings.policyInitAreaProp = 0.00;
    playSettings.compensateAfterPolicyInitProb = 0.0;
    playSettings.sidePositionProb = 0;
    playSettings.compensateKomiVisits = 4;
    playSettings.estimateLeadProb = 0.0;
    playSettings.fancyKomiVarying = true;
    playSettings.sekiForkHackProb = 0.0;
    playSettings.earlyForkGameProb = 0.0;
    playSettings.forkGameMinChoices = 2;
    playSettings.earlyForkGameMaxChoices = 3;
    playSettings.forSelfPlay = true;

    std::map<string,string> cfgParams({
        std::make_pair("maxMovesPerGame","0"),
        std::make_pair("logSearchInfo","false"),
        std::make_pair("logMoves","false"),
        std::make_pair("komiAuto","true"),
        std::make_pair("komiStdev","0.1"),
        std::make_pair("handicapProb","0.5"),
        std::make_pair("handicapCompensateKomiProb","1.0"),
        std::make_pair("forkCompensateKomiProb","1.0"),
        std::make_pair("komiBigStdevProb","0.0"),
        std::make_pair("komiBigStdev","5.0"),
        std::make_pair("komiBiggerStdevProb","0.0"),
        std::make_pair("komiBiggerStdev","100.0"),
        std::make_pair("sgfKomiInterpZeroProb","0.0"),
        std::make_pair("handicapKomiInterpZeroProb","0.5"),
        std::make_pair("drawRandRadius","0.5"),
        std::make_pair("noResultStdev","0.16"),

        std::make_pair("bSizes","14"),
        std::make_pair("bSizeRelProbs","1"),
        std::make_pair("koRules","SIMPLE,POSITIONAL,SITUATIONAL"),
        std::make_pair("scoringRules","AREA"),
        std::make_pair("taxRules","NONE,NONE,SEKI,SEKI,ALL"),
        std::make_pair("multiStoneSuicideLegals","false,true"),
        std::make_pair("hasButtons","false,false,true"),
        std::make_pair("allowRectangleProb","0.0"),
    });
    runStatTest(cfgParams,playSettings,NULL,name,100);
  }

  {
    string name = "Game init test 19x19 policy init and not make fair, low komi stdev";
    //Statistical test of game initialization
    PlaySettings playSettings;
    playSettings.initGamesWithPolicy = true;
    playSettings.policyInitAreaProp = 0.05;
    playSettings.compensateAfterPolicyInitProb = 0.0;
    playSettings.sidePositionProb = 0;
    playSettings.compensateKomiVisits = 4;
    playSettings.estimateLeadProb = 0.0;
    playSettings.fancyKomiVarying = false;
    playSettings.sekiForkHackProb = 0.0;
    playSettings.earlyForkGameProb = 0.0;
    playSettings.forkGameMinChoices = 2;
    playSettings.earlyForkGameMaxChoices = 3;
    playSettings.forSelfPlay = true;

    std::map<string,string> cfgParams({
        std::make_pair("maxMovesPerGame","0"),
        std::make_pair("logSearchInfo","false"),
        std::make_pair("logMoves","false"),
        std::make_pair("komiAuto","true"),
        std::make_pair("komiStdev","0.5"),
        std::make_pair("handicapProb","0.0"),
        std::make_pair("handicapCompensateKomiProb","1.0"),
        std::make_pair("forkCompensateKomiProb","1.0"),
        std::make_pair("komiBigStdevProb","0.0"),
        std::make_pair("komiBigStdev","20.0"),
        std::make_pair("drawRandRadius","0.5"),
        std::make_pair("noResultStdev","0.16"),

        std::make_pair("bSizes","19"),
        std::make_pair("bSizeRelProbs","1"),
        std::make_pair("koRules","SIMPLE,POSITIONAL,SITUATIONAL"),
        std::make_pair("scoringRules","AREA"),
        std::make_pair("taxRules","NONE,NONE,SEKI,SEKI,ALL"),
        std::make_pair("multiStoneSuicideLegals","false,true"),
        std::make_pair("hasButtons","false,false,true"),
        std::make_pair("allowRectangleProb","0.0"),
    });
    runStatTest(cfgParams,playSettings,NULL,name,100);
  }

  {
    string name = "Game init test 19x19 fork and half make fair, policy init not make fair, low komi stdev";
    //Statistical test of game initialization
    PlaySettings playSettings;
    playSettings.initGamesWithPolicy = true;
    playSettings.policyInitAreaProp = 0.05;
    playSettings.compensateAfterPolicyInitProb = 0.0;
    playSettings.sidePositionProb = 0;
    playSettings.compensateKomiVisits = 4;
    playSettings.estimateLeadProb = 0.0;
    playSettings.fancyKomiVarying = false;
    playSettings.sekiForkHackProb = 0.0;
    playSettings.earlyForkGameProb = 0.75;
    playSettings.earlyForkGameExpectedMoveProp = 0.01;
    playSettings.forkGameMinChoices = 2;
    playSettings.earlyForkGameMaxChoices = 3;
    playSettings.forSelfPlay = true;

    std::map<string,string> cfgParams({
        std::make_pair("maxMovesPerGame","0"),
        std::make_pair("logSearchInfo","false"),
        std::make_pair("logMoves","false"),
        std::make_pair("komiAuto","true"),
        std::make_pair("komiStdev","0.5"),
        std::make_pair("handicapProb","0.0"),
        std::make_pair("handicapCompensateKomiProb","1.0"),
        std::make_pair("forkCompensateKomiProb","0.5"),
        std::make_pair("komiBigStdevProb","0.0"),
        std::make_pair("komiBigStdev","20.0"),
        std::make_pair("drawRandRadius","0.5"),
        std::make_pair("noResultStdev","0.16"),

        std::make_pair("bSizes","19"),
        std::make_pair("bSizeRelProbs","1"),
        std::make_pair("koRules","SIMPLE,POSITIONAL,SITUATIONAL"),
        std::make_pair("scoringRules","AREA"),
        std::make_pair("taxRules","NONE,NONE,SEKI,SEKI,ALL"),
        std::make_pair("multiStoneSuicideLegals","false,true"),
        std::make_pair("hasButtons","false,false,true"),
        std::make_pair("allowRectangleProb","0.0"),
    });
    runStatTest(cfgParams,playSettings,NULL,name,100);
  }

  {
    string name = "Game init test 19x19 fork many choices and half make fair, policy init not make fair, low komi stdev";
    //Statistical test of game initialization
    PlaySettings playSettings;
    playSettings.initGamesWithPolicy = true;
    playSettings.policyInitAreaProp = 0.05;
    playSettings.compensateAfterPolicyInitProb = 0.0;
    playSettings.sidePositionProb = 0;
    playSettings.compensateKomiVisits = 4;
    playSettings.estimateLeadProb = 0.0;
    playSettings.fancyKomiVarying = false;
    playSettings.sekiForkHackProb = 0.0;
    playSettings.earlyForkGameProb = 0.75;
    playSettings.earlyForkGameExpectedMoveProp = 0.01;
    playSettings.forkGameMinChoices = 16;
    playSettings.earlyForkGameMaxChoices = 20;
    playSettings.forSelfPlay = true;

    std::map<string,string> cfgParams({
        std::make_pair("maxMovesPerGame","0"),
        std::make_pair("logSearchInfo","false"),
        std::make_pair("logMoves","false"),
        std::make_pair("komiAuto","true"),
        std::make_pair("komiStdev","0.5"),
        std::make_pair("handicapProb","0.0"),
        std::make_pair("handicapCompensateKomiProb","1.0"),
        std::make_pair("forkCompensateKomiProb","0.5"),
        std::make_pair("komiBigStdevProb","0.0"),
        std::make_pair("komiBigStdev","20.0"),
        std::make_pair("drawRandRadius","0.5"),
        std::make_pair("noResultStdev","0.16"),

        std::make_pair("bSizes","19"),
        std::make_pair("bSizeRelProbs","1"),
        std::make_pair("koRules","SIMPLE,POSITIONAL,SITUATIONAL"),
        std::make_pair("scoringRules","AREA"),
        std::make_pair("taxRules","NONE,NONE,SEKI,SEKI,ALL"),
        std::make_pair("multiStoneSuicideLegals","false,true"),
        std::make_pair("hasButtons","false,false,true"),
        std::make_pair("allowRectangleProb","0.0"),
    });
    runStatTest(cfgParams,playSettings,NULL,name,100);
  }

  {
    string name = "Game init test 19x19 fork many choices and half make fair, policy init not make fair, low komi stdev, fancy";
    //Statistical test of game initialization
    PlaySettings playSettings;
    playSettings.initGamesWithPolicy = true;
    playSettings.policyInitAreaProp = 0.05;
    playSettings.compensateAfterPolicyInitProb = 0.0;
    playSettings.sidePositionProb = 0;
    playSettings.compensateKomiVisits = 4;
    playSettings.estimateLeadProb = 0.0;
    playSettings.fancyKomiVarying = true;
    playSettings.sekiForkHackProb = 0.0;
    playSettings.earlyForkGameProb = 0.75;
    playSettings.earlyForkGameExpectedMoveProp = 0.01;
    playSettings.forkGameMinChoices = 16;
    playSettings.earlyForkGameMaxChoices = 20;
    playSettings.forSelfPlay = true;

    std::map<string,string> cfgParams({
        std::make_pair("maxMovesPerGame","0"),
        std::make_pair("logSearchInfo","false"),
        std::make_pair("logMoves","false"),
        std::make_pair("komiAuto","true"),
        std::make_pair("komiStdev","0.5"),
        std::make_pair("handicapProb","0.0"),
        std::make_pair("handicapCompensateKomiProb","1.0"),
        std::make_pair("forkCompensateKomiProb","0.5"),
        std::make_pair("komiBigStdevProb","0.0"),
        std::make_pair("komiBigStdev","20.0"),
        std::make_pair("drawRandRadius","0.5"),
        std::make_pair("noResultStdev","0.16"),

        std::make_pair("bSizes","19"),
        std::make_pair("bSizeRelProbs","1"),
        std::make_pair("koRules","SIMPLE,POSITIONAL,SITUATIONAL"),
        std::make_pair("scoringRules","AREA"),
        std::make_pair("taxRules","NONE,NONE,SEKI,SEKI,ALL"),
        std::make_pair("multiStoneSuicideLegals","false,true"),
        std::make_pair("hasButtons","false,false,true"),
        std::make_pair("allowRectangleProb","0.0"),
    });
    runStatTest(cfgParams,playSettings,NULL,name,100);
  }

  {
    string name = "Game init test 19x19 handicap";
    //Statistical test of game initialization
    PlaySettings playSettings;
    playSettings.initGamesWithPolicy = true;
    playSettings.policyInitAreaProp = 0.0;
    playSettings.compensateAfterPolicyInitProb = 0.0;
    playSettings.sidePositionProb = 0;
    playSettings.compensateKomiVisits = 4;
    playSettings.estimateLeadProb = 0.0;
    playSettings.fancyKomiVarying = false;
    playSettings.sekiForkHackProb = 0.0;
    playSettings.earlyForkGameProb = 0.0;
    playSettings.forkGameMinChoices = 2;
    playSettings.earlyForkGameMaxChoices = 3;
    playSettings.forSelfPlay = true;

    std::map<string,string> cfgParams({
        std::make_pair("maxMovesPerGame","0"),
        std::make_pair("logSearchInfo","false"),
        std::make_pair("logMoves","false"),
        std::make_pair("komiAuto","true"),
        std::make_pair("komiStdev","0.5"),
        std::make_pair("handicapProb","1.0"),
        std::make_pair("handicapCompensateKomiProb","0.5"),
        std::make_pair("forkCompensateKomiProb","1.0"),
        std::make_pair("komiBigStdevProb","0.0"),
        std::make_pair("komiBigStdev","20.0"),
        std::make_pair("drawRandRadius","0.5"),
        std::make_pair("noResultStdev","0.16"),

        std::make_pair("bSizes","19"),
        std::make_pair("bSizeRelProbs","1"),
        std::make_pair("koRules","SIMPLE,POSITIONAL,SITUATIONAL"),
        std::make_pair("scoringRules","AREA"),
        std::make_pair("taxRules","NONE,NONE,SEKI,SEKI,ALL"),
        std::make_pair("multiStoneSuicideLegals","false,true"),
        std::make_pair("hasButtons","false,false,true"),
        std::make_pair("allowRectangleProb","0.0"),
    });
    runStatTest(cfgParams,playSettings,NULL,name,100);
  }

  {
    string name = "Game init test 19x19 handicap low temp";
    //Statistical test of game initialization
    PlaySettings playSettings;
    playSettings.initGamesWithPolicy = true;
    playSettings.policyInitAreaProp = 0.0;
    playSettings.handicapTemperature = 0.5;
    playSettings.compensateAfterPolicyInitProb = 0.0;
    playSettings.sidePositionProb = 0;
    playSettings.compensateKomiVisits = 4;
    playSettings.estimateLeadProb = 0.0;
    playSettings.fancyKomiVarying = false;
    playSettings.sekiForkHackProb = 0.0;
    playSettings.earlyForkGameProb = 0.0;
    playSettings.forkGameMinChoices = 2;
    playSettings.earlyForkGameMaxChoices = 3;
    playSettings.forSelfPlay = true;

    std::map<string,string> cfgParams({
        std::make_pair("maxMovesPerGame","0"),
        std::make_pair("logSearchInfo","false"),
        std::make_pair("logMoves","false"),
        std::make_pair("komiAuto","true"),
        std::make_pair("komiStdev","0.5"),
        std::make_pair("handicapProb","1.0"),
        std::make_pair("handicapCompensateKomiProb","0.5"),
        std::make_pair("forkCompensateKomiProb","1.0"),
        std::make_pair("komiBigStdevProb","0.0"),
        std::make_pair("komiBigStdev","20.0"),
        std::make_pair("drawRandRadius","0.5"),
        std::make_pair("noResultStdev","0.16"),

        std::make_pair("bSizes","19"),
        std::make_pair("bSizeRelProbs","1"),
        std::make_pair("koRules","SIMPLE,POSITIONAL,SITUATIONAL"),
        std::make_pair("scoringRules","AREA"),
        std::make_pair("taxRules","NONE,NONE,SEKI,SEKI,ALL"),
        std::make_pair("multiStoneSuicideLegals","false,true"),
        std::make_pair("hasButtons","false,false,true"),
        std::make_pair("allowRectangleProb","0.0"),
    });
    runStatTest(cfgParams,playSettings,NULL,name,100);
  }

  {
    string name = "Game init test 19x19 handicap low temp fancy";
    //Statistical test of game initialization
    PlaySettings playSettings;
    playSettings.initGamesWithPolicy = true;
    playSettings.policyInitAreaProp = 0.0;
    playSettings.handicapTemperature = 0.5;
    playSettings.compensateAfterPolicyInitProb = 0.0;
    playSettings.sidePositionProb = 0;
    playSettings.compensateKomiVisits = 4;
    playSettings.estimateLeadProb = 0.0;
    playSettings.fancyKomiVarying = true;
    playSettings.sekiForkHackProb = 0.0;
    playSettings.earlyForkGameProb = 0.0;
    playSettings.forkGameMinChoices = 2;
    playSettings.earlyForkGameMaxChoices = 3;
    playSettings.forSelfPlay = true;

    std::map<string,string> cfgParams({
        std::make_pair("maxMovesPerGame","0"),
        std::make_pair("logSearchInfo","false"),
        std::make_pair("logMoves","false"),
        std::make_pair("komiAuto","true"),
        std::make_pair("komiStdev","0.5"),
        std::make_pair("handicapProb","1.0"),
        std::make_pair("handicapCompensateKomiProb","0.5"),
        std::make_pair("forkCompensateKomiProb","1.0"),
        std::make_pair("komiBigStdevProb","0.0"),
        std::make_pair("komiBigStdev","20.0"),
        std::make_pair("drawRandRadius","0.5"),
        std::make_pair("noResultStdev","0.16"),

        std::make_pair("bSizes","19"),
        std::make_pair("bSizeRelProbs","1"),
        std::make_pair("koRules","SIMPLE,POSITIONAL,SITUATIONAL"),
        std::make_pair("scoringRules","AREA"),
        std::make_pair("taxRules","NONE,NONE,SEKI,SEKI,ALL"),
        std::make_pair("multiStoneSuicideLegals","false,true"),
        std::make_pair("hasButtons","false,false,true"),
        std::make_pair("allowRectangleProb","0.0"),
    });
    runStatTest(cfgParams,playSettings,NULL,name,100);
  }

  {
    string name = "Game init test sgfpos empty board";
    //Statistical test of game initialization
    PlaySettings playSettings;
    playSettings.initGamesWithPolicy = true;
    playSettings.policyInitAreaProp = 0.05;
    playSettings.compensateAfterPolicyInitProb = 0.0;
    playSettings.sidePositionProb = 0;
    playSettings.compensateKomiVisits = 4;
    playSettings.estimateLeadProb = 0.0;
    playSettings.fancyKomiVarying = true;
    playSettings.sekiForkHackProb = 0.0;
    playSettings.earlyForkGameProb = 0.0;
    playSettings.forkGameMinChoices = 2;
    playSettings.earlyForkGameMaxChoices = 3;
    playSettings.forSelfPlay = true;

    Sgf::PositionSample startPosSample;
    startPosSample.board = Board(19,19);
    startPosSample.nextPla = P_BLACK;
    startPosSample.moves = std::vector<Move>();
    startPosSample.initialTurnNumber = 0;
    startPosSample.hintLoc = Board::NULL_LOC;
    startPosSample.weight = 1.0;

    std::map<string,string> cfgParams({
        std::make_pair("maxMovesPerGame","0"),
        std::make_pair("logSearchInfo","false"),
        std::make_pair("logMoves","false"),
        std::make_pair("komiAuto","true"),
        std::make_pair("komiStdev","2.0"),
        std::make_pair("handicapProb","0.0"),
        std::make_pair("handicapCompensateKomiProb","1.0"),
        std::make_pair("forkCompensateKomiProb","1.0"),
        std::make_pair("komiBigStdevProb","0.0"),
        std::make_pair("komiBigStdev","20.0"),
        std::make_pair("drawRandRadius","0.5"),
        std::make_pair("noResultStdev","0.16"),

        std::make_pair("bSizes","19"),
        std::make_pair("bSizeRelProbs","1"),
        std::make_pair("koRules","SIMPLE,POSITIONAL,SITUATIONAL"),
        std::make_pair("scoringRules","AREA"),
        std::make_pair("taxRules","NONE,NONE,SEKI,SEKI,ALL"),
        std::make_pair("multiStoneSuicideLegals","false,true"),
        std::make_pair("hasButtons","false,false,true"),
        std::make_pair("allowRectangleProb","0.0"),
    });
    runStatTest(cfgParams,playSettings,&startPosSample,name,100);
  }

  {
    string name = "Game init test sgfpos empty board with init";
    //Statistical test of game initialization
    PlaySettings playSettings;
    playSettings.initGamesWithPolicy = true;
    playSettings.policyInitAreaProp = 0.0;
    playSettings.startPosesPolicyInitAreaProp = 0.05;
    playSettings.compensateAfterPolicyInitProb = 0.0;
    playSettings.sidePositionProb = 0;
    playSettings.compensateKomiVisits = 4;
    playSettings.estimateLeadProb = 0.0;
    playSettings.fancyKomiVarying = true;
    playSettings.sekiForkHackProb = 0.0;
    playSettings.earlyForkGameProb = 0.0;
    playSettings.forkGameMinChoices = 2;
    playSettings.earlyForkGameMaxChoices = 3;
    playSettings.forSelfPlay = true;

    Sgf::PositionSample startPosSample;
    startPosSample.board = Board(19,19);
    startPosSample.nextPla = P_BLACK;
    startPosSample.moves = std::vector<Move>();
    startPosSample.initialTurnNumber = 0;
    startPosSample.hintLoc = Board::NULL_LOC;
    startPosSample.weight = 1.0;

    std::map<string,string> cfgParams({
        std::make_pair("maxMovesPerGame","0"),
        std::make_pair("logSearchInfo","false"),
        std::make_pair("logMoves","false"),
        std::make_pair("komiAuto","true"),
        std::make_pair("komiStdev","2.0"),
        std::make_pair("handicapProb","0.0"),
        std::make_pair("handicapCompensateKomiProb","1.0"),
        std::make_pair("forkCompensateKomiProb","1.0"),
        std::make_pair("komiBigStdevProb","0.0"),
        std::make_pair("komiBigStdev","20.0"),
        std::make_pair("drawRandRadius","0.5"),
        std::make_pair("noResultStdev","0.16"),

        std::make_pair("bSizes","19"),
        std::make_pair("bSizeRelProbs","1"),
        std::make_pair("koRules","SIMPLE,POSITIONAL,SITUATIONAL"),
        std::make_pair("scoringRules","AREA"),
        std::make_pair("taxRules","NONE,NONE,SEKI,SEKI,ALL"),
        std::make_pair("multiStoneSuicideLegals","false,true"),
        std::make_pair("hasButtons","false,false,true"),
        std::make_pair("allowRectangleProb","0.0"),
    });
    runStatTest(cfgParams,playSettings,&startPosSample,name,100);
  }

  {
    string name = "Game init test sgfpos nonempty board with init and different turn number";
    //Statistical test of game initialization
    PlaySettings playSettings;
    playSettings.initGamesWithPolicy = true;
    playSettings.policyInitAreaProp = 0.0;
    playSettings.startPosesPolicyInitAreaProp = 0.20;
    playSettings.compensateAfterPolicyInitProb = 0.0;
    playSettings.sidePositionProb = 0;
    playSettings.compensateKomiVisits = 4;
    playSettings.estimateLeadProb = 0.0;
    playSettings.fancyKomiVarying = true;
    playSettings.sekiForkHackProb = 0.0;
    playSettings.earlyForkGameProb = 0.0;
    playSettings.forkGameMinChoices = 2;
    playSettings.earlyForkGameMaxChoices = 3;
    playSettings.forSelfPlay = true;

    Sgf::PositionSample startPosSample;
    startPosSample.board = Board::parseBoard(9,9,R"%%(
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
    startPosSample.nextPla = P_BLACK;
    startPosSample.moves = std::vector<Move>({
        Move(Location::getLoc(8,3,9),P_BLACK),
      });
    startPosSample.initialTurnNumber = 50;
    startPosSample.hintLoc = Board::NULL_LOC;
    startPosSample.weight = 10.0;

    std::map<string,string> cfgParams({
        std::make_pair("maxMovesPerGame","0"),
        std::make_pair("logSearchInfo","false"),
        std::make_pair("logMoves","false"),
        std::make_pair("komiAuto","true"),
        std::make_pair("komiStdev","2.0"),
        std::make_pair("handicapProb","0.0"),
        std::make_pair("handicapCompensateKomiProb","1.0"),
        std::make_pair("forkCompensateKomiProb","1.0"),
        std::make_pair("komiBigStdevProb","0.0"),
        std::make_pair("komiBigStdev","20.0"),
        std::make_pair("drawRandRadius","0.5"),
        std::make_pair("noResultStdev","0.16"),

        std::make_pair("bSizes","9"),
        std::make_pair("bSizeRelProbs","1"),
        std::make_pair("koRules","SIMPLE,POSITIONAL,SITUATIONAL"),
        std::make_pair("scoringRules","AREA"),
        std::make_pair("taxRules","NONE,NONE,SEKI,SEKI,ALL"),
        std::make_pair("multiStoneSuicideLegals","false,true"),
        std::make_pair("hasButtons","false,false,true"),
        std::make_pair("allowRectangleProb","0.0"),
    });
    runStatTest(cfgParams,playSettings,&startPosSample,name,100);
  }

  {
    string name = "Game init test rectangles";
    //Statistical test of game initialization
    PlaySettings playSettings;
    playSettings.initGamesWithPolicy = true;
    playSettings.policyInitAreaProp = 0.0;
    playSettings.compensateAfterPolicyInitProb = 0.0;
    playSettings.sidePositionProb = 0;
    playSettings.compensateKomiVisits = 4;
    playSettings.estimateLeadProb = 0.0;
    playSettings.fancyKomiVarying = true;
    playSettings.sekiForkHackProb = 0.0;
    playSettings.earlyForkGameProb = 0.0;
    playSettings.forkGameMinChoices = 2;
    playSettings.earlyForkGameMaxChoices = 3;
    playSettings.forSelfPlay = true;

    std::map<string,string> cfgParams({
        std::make_pair("maxMovesPerGame","0"),
        std::make_pair("logSearchInfo","false"),
        std::make_pair("logMoves","false"),
        std::make_pair("komiAuto","true"),
        std::make_pair("komiStdev","2.0"),
        std::make_pair("handicapProb","0.0"),
        std::make_pair("handicapCompensateKomiProb","1.0"),
        std::make_pair("forkCompensateKomiProb","1.0"),
        std::make_pair("komiBigStdevProb","0.0"),
        std::make_pair("komiBigStdev","20.0"),
        std::make_pair("drawRandRadius","0.5"),
        std::make_pair("noResultStdev","0.16"),

        std::make_pair("bSizes","9,13,19"),
        std::make_pair("bSizeRelProbs","1,1,1"),
        std::make_pair("koRules","SIMPLE,POSITIONAL,SITUATIONAL"),
        std::make_pair("scoringRules","AREA"),
        std::make_pair("taxRules","NONE,NONE,SEKI,SEKI,ALL"),
        std::make_pair("multiStoneSuicideLegals","false,true"),
        std::make_pair("hasButtons","false,false,true"),
        std::make_pair("allowRectangleProb","0.1"),
    });
    runStatTest(cfgParams,playSettings,NULL,name,100);
  }
  {
    string name = "Game init test sgfpos black first with big black handicap, flipKomiProbWhenNoCompensate 0";
    //Statistical test of game initialization
    PlaySettings playSettings;
    playSettings.initGamesWithPolicy = false;
    playSettings.sidePositionProb = 0;
    playSettings.compensateKomiVisits = 4;
    playSettings.flipKomiProbWhenNoCompensate = 0.0;
    playSettings.estimateLeadProb = 0.0;
    playSettings.fancyKomiVarying = true;
    playSettings.sekiForkHackProb = 0.0;
    playSettings.earlyForkGameProb = 0.0;
    playSettings.forSelfPlay = true;

    Sgf::PositionSample startPosSample;
    startPosSample.board = Board(19,19);
    startPosSample.nextPla = P_BLACK;
    startPosSample.moves = std::vector<Move>({
        Move(Location::getLoc(3,3,19),P_BLACK),
        Move(Board::PASS_LOC,P_WHITE),
        Move(Location::getLoc(16,16,19),P_BLACK),
        Move(Board::PASS_LOC,P_WHITE),
        Move(Location::getLoc(3,16,19),P_BLACK),
        Move(Location::getLoc(16,3,19),P_WHITE),
      });
    startPosSample.initialTurnNumber = 0;
    startPosSample.hintLoc = Board::NULL_LOC;
    startPosSample.weight = 1.0;

    std::map<string,string> cfgParams({
        std::make_pair("maxMovesPerGame","0"),
        std::make_pair("logSearchInfo","false"),
        std::make_pair("logMoves","false"),
        std::make_pair("komiAuto","true"),
        std::make_pair("komiStdev","1.0"),
        std::make_pair("handicapProb","0.0"),
        std::make_pair("handicapCompensateKomiProb","1.0"),
        std::make_pair("forkCompensateKomiProb","1.0"),
        std::make_pair("sgfCompensateKomiProb","0.5"),
        std::make_pair("komiBigStdevProb","0.0"),
        std::make_pair("komiBigStdev","20.0"),
        std::make_pair("drawRandRadius","0.5"),
        std::make_pair("noResultStdev","0.16"),

        std::make_pair("bSizes","19"),
        std::make_pair("bSizeRelProbs","1"),
        std::make_pair("koRules","SIMPLE,POSITIONAL,SITUATIONAL"),
        std::make_pair("scoringRules","AREA"),
        std::make_pair("taxRules","NONE,NONE,SEKI,SEKI,ALL"),
        std::make_pair("multiStoneSuicideLegals","false,true"),
        std::make_pair("hasButtons","false,false,true"),
        std::make_pair("allowRectangleProb","0.0"),
    });
    runStatTest(cfgParams,playSettings,&startPosSample,name,100);
  }
  {
    string name = "Game init test sgfpos white first with big black handicap, flipKomiProbWhenNoCompensate 0";
    //Statistical test of game initialization
    PlaySettings playSettings;
    playSettings.initGamesWithPolicy = false;
    playSettings.sidePositionProb = 0;
    playSettings.compensateKomiVisits = 4;
    playSettings.flipKomiProbWhenNoCompensate = 0.0;
    playSettings.estimateLeadProb = 0.0;
    playSettings.fancyKomiVarying = true;
    playSettings.sekiForkHackProb = 0.0;
    playSettings.earlyForkGameProb = 0.0;
    playSettings.forSelfPlay = true;

    Sgf::PositionSample startPosSample;
    startPosSample.board = Board(19,19);
    startPosSample.nextPla = P_BLACK;
    startPosSample.moves = std::vector<Move>({
        Move(Location::getLoc(3,3,19),P_BLACK),
        Move(Board::PASS_LOC,P_WHITE),
        Move(Location::getLoc(16,16,19),P_BLACK),
        Move(Board::PASS_LOC,P_WHITE),
        Move(Location::getLoc(3,16,19),P_BLACK),
      });
    startPosSample.initialTurnNumber = 0;
    startPosSample.hintLoc = Board::NULL_LOC;
    startPosSample.weight = 1.0;

    std::map<string,string> cfgParams({
        std::make_pair("maxMovesPerGame","0"),
        std::make_pair("logSearchInfo","false"),
        std::make_pair("logMoves","false"),
        std::make_pair("komiAuto","true"),
        std::make_pair("komiStdev","1.0"),
        std::make_pair("handicapProb","0.0"),
        std::make_pair("handicapCompensateKomiProb","1.0"),
        std::make_pair("forkCompensateKomiProb","1.0"),
        std::make_pair("sgfCompensateKomiProb","0.5"),
        std::make_pair("komiBigStdevProb","0.0"),
        std::make_pair("komiBigStdev","20.0"),
        std::make_pair("drawRandRadius","0.5"),
        std::make_pair("noResultStdev","0.16"),

        std::make_pair("bSizes","19"),
        std::make_pair("bSizeRelProbs","1"),
        std::make_pair("koRules","SIMPLE,POSITIONAL,SITUATIONAL"),
        std::make_pair("scoringRules","AREA"),
        std::make_pair("taxRules","NONE,NONE,SEKI,SEKI,ALL"),
        std::make_pair("multiStoneSuicideLegals","false,true"),
        std::make_pair("hasButtons","false,false,true"),
        std::make_pair("allowRectangleProb","0.0"),
    });
    runStatTest(cfgParams,playSettings,&startPosSample,name,100);
  }
  {
    string name = "Game init test sgfpos black first with big black handicap, flipKomiProbWhenNoCompensate 0.25";
    //Statistical test of game initialization
    PlaySettings playSettings;
    playSettings.initGamesWithPolicy = false;
    playSettings.sidePositionProb = 0;
    playSettings.compensateKomiVisits = 4;
    playSettings.flipKomiProbWhenNoCompensate = 0.25;
    playSettings.estimateLeadProb = 0.0;
    playSettings.fancyKomiVarying = true;
    playSettings.sekiForkHackProb = 0.0;
    playSettings.earlyForkGameProb = 0.0;
    playSettings.forSelfPlay = true;

    Sgf::PositionSample startPosSample;
    startPosSample.board = Board(19,19);
    startPosSample.nextPla = P_BLACK;
    startPosSample.moves = std::vector<Move>({
        Move(Location::getLoc(3,3,19),P_BLACK),
        Move(Board::PASS_LOC,P_WHITE),
        Move(Location::getLoc(16,16,19),P_BLACK),
        Move(Board::PASS_LOC,P_WHITE),
        Move(Location::getLoc(3,16,19),P_BLACK),
        Move(Location::getLoc(16,3,19),P_WHITE),
      });
    startPosSample.initialTurnNumber = 0;
    startPosSample.hintLoc = Board::NULL_LOC;
    startPosSample.weight = 1.0;

    std::map<string,string> cfgParams({
        std::make_pair("maxMovesPerGame","0"),
        std::make_pair("logSearchInfo","false"),
        std::make_pair("logMoves","false"),
        std::make_pair("komiAuto","true"),
        std::make_pair("komiStdev","1.0"),
        std::make_pair("handicapProb","0.0"),
        std::make_pair("handicapCompensateKomiProb","1.0"),
        std::make_pair("forkCompensateKomiProb","1.0"),
        std::make_pair("sgfCompensateKomiProb","0.5"),
        std::make_pair("komiBigStdevProb","0.0"),
        std::make_pair("komiBigStdev","20.0"),
        std::make_pair("drawRandRadius","0.5"),
        std::make_pair("noResultStdev","0.16"),

        std::make_pair("bSizes","19"),
        std::make_pair("bSizeRelProbs","1"),
        std::make_pair("koRules","SIMPLE,POSITIONAL,SITUATIONAL"),
        std::make_pair("scoringRules","AREA"),
        std::make_pair("taxRules","NONE,NONE,SEKI,SEKI,ALL"),
        std::make_pair("multiStoneSuicideLegals","false,true"),
        std::make_pair("hasButtons","false,false,true"),
        std::make_pair("allowRectangleProb","0.0"),
    });
    runStatTest(cfgParams,playSettings,&startPosSample,name,100);
  }
  {
    string name = "Game init test sgfpos white first with big black handicap, flipKomiProbWhenNoCompensate 0.25";
    //Statistical test of game initialization
    PlaySettings playSettings;
    playSettings.initGamesWithPolicy = false;
    playSettings.sidePositionProb = 0;
    playSettings.compensateKomiVisits = 4;
    playSettings.flipKomiProbWhenNoCompensate = 0.25;
    playSettings.estimateLeadProb = 0.0;
    playSettings.fancyKomiVarying = true;
    playSettings.sekiForkHackProb = 0.0;
    playSettings.earlyForkGameProb = 0.0;
    playSettings.forSelfPlay = true;

    Sgf::PositionSample startPosSample;
    startPosSample.board = Board(19,19);
    startPosSample.nextPla = P_BLACK;
    startPosSample.moves = std::vector<Move>({
        Move(Location::getLoc(3,3,19),P_BLACK),
        Move(Board::PASS_LOC,P_WHITE),
        Move(Location::getLoc(16,16,19),P_BLACK),
        Move(Board::PASS_LOC,P_WHITE),
        Move(Location::getLoc(3,16,19),P_BLACK),
      });
    startPosSample.initialTurnNumber = 0;
    startPosSample.hintLoc = Board::NULL_LOC;
    startPosSample.weight = 1.0;

    std::map<string,string> cfgParams({
        std::make_pair("maxMovesPerGame","0"),
        std::make_pair("logSearchInfo","false"),
        std::make_pair("logMoves","false"),
        std::make_pair("komiAuto","true"),
        std::make_pair("komiStdev","1.0"),
        std::make_pair("handicapProb","0.0"),
        std::make_pair("handicapCompensateKomiProb","1.0"),
        std::make_pair("forkCompensateKomiProb","1.0"),
        std::make_pair("sgfCompensateKomiProb","0.5"),
        std::make_pair("komiBigStdevProb","0.0"),
        std::make_pair("komiBigStdev","20.0"),
        std::make_pair("drawRandRadius","0.5"),
        std::make_pair("noResultStdev","0.16"),

        std::make_pair("bSizes","19"),
        std::make_pair("bSizeRelProbs","1"),
        std::make_pair("koRules","SIMPLE,POSITIONAL,SITUATIONAL"),
        std::make_pair("scoringRules","AREA"),
        std::make_pair("taxRules","NONE,NONE,SEKI,SEKI,ALL"),
        std::make_pair("multiStoneSuicideLegals","false,true"),
        std::make_pair("hasButtons","false,false,true"),
        std::make_pair("allowRectangleProb","0.0"),
    });
    runStatTest(cfgParams,playSettings,&startPosSample,name,100);
  }

  {
    string name = "Game interpZero with whole board ownership";
    //Statistical test of game initialization
    PlaySettings playSettings;
    playSettings.initGamesWithPolicy = false;
    playSettings.policyInitAreaProp = 0.00;
    playSettings.compensateAfterPolicyInitProb = 0.0;
    playSettings.sidePositionProb = 0;
    playSettings.compensateKomiVisits = 4;
    playSettings.estimateLeadProb = 0.0;
    playSettings.fancyKomiVarying = true;
    playSettings.sekiForkHackProb = 0.0;
    playSettings.earlyForkGameProb = 0.0;
    playSettings.forkGameMinChoices = 2;
    playSettings.earlyForkGameMaxChoices = 3;
    playSettings.forSelfPlay = true;

    Sgf::PositionSample startPosSample;
    startPosSample.board = Board(19,19);
    startPosSample.nextPla = P_BLACK;
    startPosSample.moves = std::vector<Move>();
    startPosSample.initialTurnNumber = 0;
    startPosSample.hintLoc = Board::NULL_LOC;
    startPosSample.weight = 1.0;
    for(int y = 0; y<19; y++) {
      for(int x = 0; x<19; x++) {
        if((x + (y/5)) % 2 == 0) {
          startPosSample.board.setStone(Location::getLoc(x,y,startPosSample.board.x_size), C_BLACK);
        }
      }
    }

    std::map<string,string> cfgParams({
        std::make_pair("maxMovesPerGame","0"),
        std::make_pair("logSearchInfo","false"),
        std::make_pair("logMoves","false"),
        std::make_pair("komiAuto","true"),
        std::make_pair("komiStdev","0.1"),
        std::make_pair("handicapProb","0.5"),
        std::make_pair("handicapCompensateKomiProb","1.0"),
        std::make_pair("forkCompensateKomiProb","1.0"),
        std::make_pair("komiBigStdevProb","0.0"),
        std::make_pair("komiBigStdev","5.0"),
        std::make_pair("komiBiggerStdevProb","0.0"),
        std::make_pair("komiBiggerStdev","100.0"),
        std::make_pair("sgfKomiInterpZeroProb","0.5"),
        std::make_pair("drawRandRadius","0.5"),
        std::make_pair("noResultStdev","0.16"),

        std::make_pair("bSizes","19"),
        std::make_pair("bSizeRelProbs","1"),
        std::make_pair("koRules","SIMPLE,POSITIONAL,SITUATIONAL"),
        std::make_pair("scoringRules","AREA"),
        std::make_pair("taxRules","NONE,NONE,SEKI,SEKI,ALL"),
        std::make_pair("multiStoneSuicideLegals","false,true"),
        std::make_pair("hasButtons","false,false,true"),
        std::make_pair("allowRectangleProb","0.0"),
    });
    runStatTest(cfgParams,playSettings,&startPosSample,name,100);
  }

  delete nnEval;
  NeuralNet::globalCleanup();
}


void Tests::runSekiTrainWriteTests(const string& modelFile) {
  bool inputsNHWC = true;
  bool useNHWC = false;
  TestCommon::overrideForBackends(inputsNHWC, useNHWC);

  cout << "Running test for how a seki gets recorded" << endl;
  NeuralNet::globalInitialize();

  int nnXLen = 13;
  int nnYLen = 13;

  const bool logToStdout = true;
  const bool logToStderr = false;
  const bool logTime = false;
  Logger logger(nullptr, logToStdout, logToStderr, logTime);

  NNEvaluator* nnEval = startNNEval(modelFile,"nneval",logger,0,inputsNHWC,useNHWC,false);

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
    extraBlackAndKomi.komiMean = rules.komi;
    extraBlackAndKomi.komiStdev = 0;
    int turnIdx = (int)sgf->moves.size();
    sgf->setupBoardAndHistAssumeLegal(rules,initialBoard,initialPla,initialHist,turnIdx);

    bool doEndGameIfAllPassAlive = true;
    bool clearBotAfterSearch = true;
    int maxMovesPerGame = 1;
    auto shouldStop = []() noexcept { return false; };
    WaitableFlag* shouldPause = nullptr;
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
    Search* bot = new Search(botSpec.baseParams, botSpec.nnEval, &logger, searchRandSeed);

    Rand rand(seedBase+"play");
    OtherGameProperties otherGameProps;
    FinishedGameData* gameData = Play::runGame(
      initialBoard,initialPla,initialHist,extraBlackAndKomi,
      botSpec,botSpec,
      bot,bot,
      doEndGameIfAllPassAlive, clearBotAfterSearch,
      logger, false, false,
      maxMovesPerGame, shouldStop,
      shouldPause,
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
    Search* bot = new Search(params, nnEval, &logger, searchRandSeed);

    auto testStatuses = [&bot](const Board& board, const BoardHistory& hist, Player pla) {
      int numVisits = 50;
      vector<double> ownership = PlayUtils::computeOwnership(bot,board,hist,pla,numVisits);
      vector<double> buf;
      vector<bool> isAlive = PlayUtils::computeAnticipatedStatusesWithOwnership(bot,board,hist,pla,numVisits,buf);
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
  cout << "Done" << endl;
}
