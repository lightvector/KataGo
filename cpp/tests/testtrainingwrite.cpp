#include "../tests/tests.h"

#include "../dataio/trainingwrite.h"
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
    openCLReTunePerBoardSize
  );
  (void)inputsUseNHWC;

  int numNNServerThreadsPerModel = 1;
  bool nnRandomize = false;

  nnEval->spawnServerThreads(
    numNNServerThreadsPerModel,
    nnRandomize,
    seed,
    defaultSymmetry,
    logger,
    gpuIdxByServerThread,
    useFP16,
    useNHWC
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

  auto run = [&](const string& seedBase, const Rules& rules, double drawEquivalentWinsForWhite, int inputsVersion, int nnXLen, int nnYLen, int boardXLen, int boardYLen) {
    TrainingDataWriter dataWriter(&cout,inputsVersion, maxRows, firstFileMinRandProp, nnXLen, nnYLen, debugOnlyWriteEvery, seedBase+"dwriter");

    NNEvaluator* nnEval = startNNEval("/dev/null",seedBase+"nneval",logger,0,true,false,false);

    SearchParams params;
    params.maxVisits = 100;
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

    ExtraBlackAndKomi extraBlackAndKomi = ExtraBlackAndKomi(0,rules.komi,rules.komi);
    bool doEndGameIfAllPassAlive = true;
    bool clearBotAfterSearch = true;
    int maxMovesPerGame = 40;
    vector<std::atomic<bool>*> stopConditions;
    FancyModes fancyModes;
    fancyModes.initGamesWithPolicy = true;
    fancyModes.forkSidePositionProb = 0.10;
    fancyModes.forSelfPlay = true;
    fancyModes.dataXLen = nnXLen;
    fancyModes.dataYLen = nnYLen;
    Rand rand(seedBase+"play");
    FinishedGameData* gameData = Play::runGame(
      initialBoard,initialPla,initialHist,extraBlackAndKomi,
      botSpec,botSpec,
      seedBase+"search",
      doEndGameIfAllPassAlive, clearBotAfterSearch,
      logger, false, false,
      maxMovesPerGame, stopConditions,
      fancyModes, true,
      rand,
      NULL
    );

    cout << "seedBase: " << seedBase << endl;
    cout << gameData->startHist.getRecentBoard(0) << endl;
    gameData->endHist.printDebugInfo(cout,gameData->endHist.getRecentBoard(0));

    dataWriter.writeGame(*gameData);
    delete gameData;

    dataWriter.flushIfNonempty();

    delete nnEval;
    cout << endl;
  };

  int inputsVersion = 3;

  run("testtrainingwrite-tt",Rules::getTrompTaylorish(),0.5,inputsVersion,5,5,5,5);

  Rules rules;
  rules.koRule = Rules::KO_SIMPLE;
  rules.scoringRule = Rules::SCORING_TERRITORY;
  rules.multiStoneSuicideLegal = false;
  rules.komi = 5;
  run("testtrainingwrite-jp",rules,0.5,inputsVersion,5,5,5,5);

  rules = Rules::getTrompTaylorish();
  rules.komi = 7;
  run("testtrainingwrite-gooddraws",rules,0.7,inputsVersion,5,5,5,5);

  inputsVersion = 5;
  run("testtrainingwrite-tt-v5",Rules::getTrompTaylorish(),0.5,inputsVersion,5,5,5,5);

  //Non-square v4
  inputsVersion = 4;
  run("testtrainingwrite-rect-v4",Rules::getTrompTaylorish(),0.5,inputsVersion,9,3,7,3);


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

  auto run = [&](const string& seedBase, const Rules& rules, double drawEquivalentWinsForWhite, int numExtraBlack) {
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

    ExtraBlackAndKomi extraBlackAndKomi(numExtraBlack,rules.komi,rules.komi);

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
    fancyModes.earlyForkGameMinChoices = 2;
    fancyModes.earlyForkGameMaxChoices = 2;
    fancyModes.noCompensateKomiProb = 0.25;
    fancyModes.compensateKomiVisits = 5;
    fancyModes.forSelfPlay = true;
    fancyModes.dataXLen = nnXLen;
    fancyModes.dataYLen = nnYLen;

    string searchRandSeed = seedBase+"search";
    Search* bot = new Search(botSpec.baseParams, botSpec.nnEval, searchRandSeed);

    Rand rand(seedBase+"play");
    FinishedGameData* gameData = Play::runGame(
      initialBoard,initialPla,initialHist,extraBlackAndKomi,
      botSpec,botSpec,
      bot,bot,
      doEndGameIfAllPassAlive, clearBotAfterSearch,
      logger, false, false,
      maxMovesPerGame, stopConditions,
      fancyModes, true,
      rand,
      NULL
    );

    const InitialPosition* nextInitialPosition = NULL;
    Play::maybeForkGame(gameData,&nextInitialPosition,fancyModes,rand,bot,logger);

    cout << "====================================================================================================" << endl;
    cout << "====================================================================================================" << endl;
    cout << "====================================================================================================" << endl;
    cout << "seedBase: " << seedBase << endl;
    gameData->printDebug(cout);
    if(nextInitialPosition != NULL) {
      cout << "Forking to initial position " << colorToChar(nextInitialPosition->pla) << endl;
      nextInitialPosition->hist.printDebugInfo(cout,nextInitialPosition->board);
    }
    delete gameData;

    delete nnEval;
    cout << endl;
  };


  run("testselfplayinit0",Rules::getTrompTaylorish(),0.5,0);
  run("testselfplayinit1",Rules::getTrompTaylorish(),0.5,0);
  run("testselfplayinit2",Rules::getTrompTaylorish(),0.5,0);
  run("testselfplayinit3",Rules::getTrompTaylorish(),0.5,0);
  run("testselfplayinit4",Rules::getTrompTaylorish(),0.5,0);
  run("testselfplayinit5",Rules::getTrompTaylorish(),0.5,0);
  run("testselfplayinit6",Rules::getTrompTaylorish(),0.5,0);
  run("testselfplayinit7",Rules::getTrompTaylorish(),0.5,0);
  run("testselfplayinit8",Rules::getTrompTaylorish(),0.5,0);
  run("testselfplayinit9",Rules::getTrompTaylorish(),0.5,0);

  run("testselfplayinith1-0",Rules::getTrompTaylorish(),0.5,1);
  run("testselfplayinith1-1",Rules::getTrompTaylorish(),0.5,1);
  run("testselfplayinith1-2",Rules::getTrompTaylorish(),0.5,1);
  run("testselfplayinith1-3",Rules::getTrompTaylorish(),0.5,1);
  run("testselfplayinith1-4",Rules::getTrompTaylorish(),0.5,1);
  run("testselfplayinith1-5",Rules::getTrompTaylorish(),0.5,1);
  run("testselfplayinith1-6",Rules::getTrompTaylorish(),0.5,1);
  run("testselfplayinith1-7",Rules::getTrompTaylorish(),0.5,1);
  run("testselfplayinith1-8",Rules::getTrompTaylorish(),0.5,1);
  run("testselfplayinith1-9",Rules::getTrompTaylorish(),0.5,1);

  run("testselfplayinith2-0",Rules::getTrompTaylorish(),0.5,2);
  run("testselfplayinith2-1",Rules::getTrompTaylorish(),0.5,2);
  run("testselfplayinith2-2",Rules::getTrompTaylorish(),0.5,2);
  run("testselfplayinith2-3",Rules::getTrompTaylorish(),0.5,2);
  run("testselfplayinith2-4",Rules::getTrompTaylorish(),0.5,2);
  run("testselfplayinith2-5",Rules::getTrompTaylorish(),0.5,2);
  run("testselfplayinith2-6",Rules::getTrompTaylorish(),0.5,2);
  run("testselfplayinith2-7",Rules::getTrompTaylorish(),0.5,2);
  run("testselfplayinith2-8",Rules::getTrompTaylorish(),0.5,2);
  run("testselfplayinith2-9",Rules::getTrompTaylorish(),0.5,2);

  NeuralNet::globalCleanup();
}
