#include "../tests/tests.h"
using namespace TestCommon;

#include "../neuralnet/nneval.h"
#include "../dataio/trainingwrite.h"
#include "../program/play.h"

static NNEvaluator* startNNEval(
  const string& seed, Logger& logger,
  int defaultSymmetry, bool inputsUseNHWC, bool cudaUseNHWC, bool cudaUseFP16
) {
  //Placeholder, doesn't actually do anything since we have debugSkipNeuralNet = true
  string modelFile = "/dev/null";
  int modelFileIdx = 0;
  int maxBatchSize = 16;
  int maxConcurrentEvals = 1024;
  int posLen = NNPos::MAX_BOARD_LEN;
  bool requireExactPosLen = false;
  int nnCacheSizePowerOfTwo = 16;
  int nnMutexPoolSizePowerOfTwo = 12;
  bool debugSkipNeuralNet = true;
  double nnPolicyTemperature = 1.0;
  NNEvaluator* nnEval = new NNEvaluator(
    modelFile,
    modelFileIdx,
    maxBatchSize,
    maxConcurrentEvals,
    posLen,
    requireExactPosLen,
    inputsUseNHWC,
    nnCacheSizePowerOfTwo,
    nnMutexPoolSizePowerOfTwo,
    debugSkipNeuralNet,
    nnPolicyTemperature
  );
  (void)inputsUseNHWC;

  int numNNServerThreadsPerModel = 1;
  bool nnRandomize = false;
  vector<int> cudaGpuIdxByServerThread = {0};

  nnEval->spawnServerThreads(
    numNNServerThreadsPerModel,
    nnRandomize,
    seed,
    defaultSymmetry,
    logger,
    cudaGpuIdxByServerThread,
    cudaUseFP16,
    cudaUseNHWC
  );

  return nnEval;
}

void Tests::runTrainingWriteTests() {
  cout << "Running training write tests" << endl;
  string tensorflowGpuVisibleDeviceList = "";
  double tensorflowPerProcessGpuMemoryFraction = 0.3;
  NeuralNet::globalInitialize(tensorflowGpuVisibleDeviceList,tensorflowPerProcessGpuMemoryFraction);

  int inputsVersion = 3;
  int maxRows = 256;
  int posLen = 5;
  double firstFileMinRandProp = 1.0;
  int debugOnlyWriteEvery = 5;

  Logger logger;
  logger.setLogToStdout(false);
  logger.setLogTime(false);
  logger.addOStream(cout);

  auto run = [&](const string& seedBase, const Rules& rules, double drawEquivalentWinsForWhite) {
    TrainingDataWriter dataWriter(&cout,inputsVersion, maxRows, firstFileMinRandProp, posLen, debugOnlyWriteEvery, seedBase+"dwriter");

    NNEvaluator* nnEval = startNNEval(seedBase+"nneval",logger,0,true,false,false);

    SearchParams params;
    params.maxVisits = 100;
    params.drawEquivalentWinsForWhite = drawEquivalentWinsForWhite;

    MatchPairer::BotSpec botSpec;
    botSpec.botIdx = 0;
    botSpec.botName = string("test");
    botSpec.nnEval = nnEval;
    botSpec.baseParams = params;

    Board initialBoard(5,5);
    Player initialPla = P_BLACK;
    int initialEncorePhase = 0;
    BoardHistory initialHist(initialBoard,initialPla,rules,initialEncorePhase);

    int numExtraBlack = 0;
    bool doEndGameIfAllPassAlive = true;
    bool clearBotAfterSearch = true;
    int maxMovesPerGame = 40;
    vector<std::atomic<bool>*> stopConditions;
    FancyModes fancyModes;
    fancyModes.initGamesWithPolicy = true;
    fancyModes.forkSidePositionProb = 0.10;
    bool recordFullData = true;
    Rand rand(seedBase+"play");
    FinishedGameData* gameData = Play::runGame(
      initialBoard,initialPla,initialHist,numExtraBlack,
      botSpec,botSpec,
      seedBase+"search",
      doEndGameIfAllPassAlive, clearBotAfterSearch,
      logger, false, false,
      maxMovesPerGame, stopConditions,
      fancyModes, recordFullData, posLen,
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

  run("testtrainingwrite-tt",Rules::getTrompTaylorish(),0.5);

  Rules rules;
  rules.koRule = Rules::KO_SIMPLE;
  rules.scoringRule = Rules::SCORING_TERRITORY;
  rules.multiStoneSuicideLegal = false;
  rules.komi = 5;
  run("testtrainingwrite-jp",rules,0.5);

  rules = Rules::getTrompTaylorish();
  rules.komi = 7;
  run("testtrainingwrite-gooddraws",rules,0.7);

  NeuralNet::globalCleanup();
}
