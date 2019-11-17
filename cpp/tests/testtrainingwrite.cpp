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

    ExtraBlackAndKomi extraBlackAndKomi = ExtraBlackAndKomi(0,rules.komi,rules.komi);
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
      cout << "Forking to initial position " << PlayerIO::colorToChar(nextInitialPosition->pla) << endl;
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

    ExtraBlackAndKomi extraBlackAndKomi(0,rules.komi,rules.komi);
    int turnNumber = sgf->moves.size();
    sgf->setupBoardAndHist(rules,initialBoard,initialPla,initialHist,turnNumber);

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
    fancyModes.earlyForkGameMinChoices = 2;
    fancyModes.earlyForkGameMaxChoices = 2;
    fancyModes.noCompensateKomiProb = 0;
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

    cout << "seedBase: " << seedBase << endl;
    gameData->endHist.printDebugInfo(cout,gameData->endHist.getRecentBoard(0));
    dataWriter.writeGame(*gameData);
    dataWriter.flushIfNonempty();
    delete gameData;
    delete nnEval;
    delete sgf;
    cout << endl;
  };

  vector<Rules> ruless = {
    Rules(Rules::KO_SIMPLE, Rules::SCORING_AREA, Rules::TAX_NONE, false, 0.0f),
    Rules(Rules::KO_SIMPLE, Rules::SCORING_TERRITORY, Rules::TAX_NONE, false, 0.0f),
    Rules(Rules::KO_SIMPLE, Rules::SCORING_AREA, Rules::TAX_SEKI, false, 0.0f),
    Rules(Rules::KO_SIMPLE, Rules::SCORING_TERRITORY, Rules::TAX_SEKI, false, 0.0f),
    Rules(Rules::KO_SIMPLE, Rules::SCORING_AREA, Rules::TAX_ALL, false, 0.0f),
    Rules(Rules::KO_SIMPLE, Rules::SCORING_TERRITORY, Rules::TAX_ALL, false, 0.0f),
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
