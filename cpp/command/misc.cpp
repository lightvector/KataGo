#include "../core/global.h"
#include "../core/fileutils.h"
#include "../core/fancymath.h"
#include "../core/makedir.h"
#include "../core/config_parser.h"
#include "../core/parallel.h"
#include "../core/timer.h"
#include "../core/test.h"
#include "../dataio/sgf.h"
#include "../dataio/poswriter.h"
#include "../dataio/files.h"
#include "../search/asyncbot.h"
#include "../program/setup.h"
#include "../program/playutils.h"
#include "../program/play.h"
#include "../command/commandline.h"
#include "../tests/tests.h"
#include "../main.h"

#include <atomic>
#include <chrono>
#include <csignal>
#include <cmath>
#include <mutex>
#include <thread>

using namespace std;

static std::atomic<bool> sigReceived(false);
static void signalHandler(int signal)
{
  if(signal == SIGINT || signal == SIGTERM)
    sigReceived.store(true);
}

int MainCmds::printclockinfo(const vector<string>& args) {
  (void)args;
#ifdef OS_IS_WINDOWS
  cout << "Does nothing on windows, disabled" << endl;
#endif
#ifdef OS_IS_UNIX_OR_APPLE
  cout << "Tick unit in seconds: " << std::chrono::steady_clock::period::num << " / " <<  std::chrono::steady_clock::period::den << endl;
  cout << "Ticks since epoch: " << std::chrono::steady_clock::now().time_since_epoch().count() << endl;
#endif
  return 0;
}

int MainCmds::sampleinitializations(const vector<string>& args) {
  Board::initHash();
  ScoreValue::initTables();

  ConfigParser cfg;
  string modelFile;
  int numToGen;
  bool evaluate;
  try {
    KataGoCommandLine cmd("View startposes");
    cmd.addConfigFileArg("","");
    cmd.addModelFileArg();
    cmd.addOverrideConfigArg();

    TCLAP::ValueArg<int> numToGenArg("","num","Num to gen",false,1,"N");
    TCLAP::SwitchArg evaluateArg("","evaluate","Print out values and scores on the inited poses");
    cmd.add(numToGenArg);
    cmd.add(evaluateArg);
    cmd.parseArgs(args);
    numToGen = numToGenArg.getValue();
    evaluate = evaluateArg.getValue();

    cmd.getConfigAllowEmpty(cfg);
    if(cfg.getFileName() != "")
      modelFile = cmd.getModelFile();
  }
  catch (TCLAP::ArgException &e) {
    cerr << "Error: " << e.error() << " for argument " << e.argId() << endl;
    return 1;
  }

  Rand rand;

  const bool logToStdoutDefault = true;
  Logger logger(&cfg, logToStdoutDefault);

  NNEvaluator* nnEval = NULL;
  if(cfg.getFileName() != "") {
    SearchParams params = Setup::loadSingleParams(cfg,Setup::SETUP_FOR_GTP);
    {
      Setup::initializeSession(cfg);
      const int expectedConcurrentEvals = params.numThreads;
      const int defaultMaxBatchSize = std::max(8,((params.numThreads+3)/4)*4);
      const bool defaultRequireExactNNLen = false;
      const bool disableFP16 = false;
      const string expectedSha256 = "";
      nnEval = Setup::initializeNNEvaluator(
        modelFile,modelFile,expectedSha256,cfg,logger,rand,expectedConcurrentEvals,
        Board::MAX_LEN,Board::MAX_LEN,defaultMaxBatchSize,defaultRequireExactNNLen,disableFP16,
        Setup::SETUP_FOR_GTP
      );
    }
    logger.write("Loaded neural net");
  }

  AsyncBot* evalBot;
  {
    SearchParams params = Setup::loadSingleParams(cfg,Setup::SETUP_FOR_DISTRIBUTED);
    params.maxVisits = 20;
    params.numThreads = 1;
    string seed = Global::uint64ToString(rand.nextUInt64());
    evalBot = new AsyncBot(params, nnEval, &logger, seed);
  }

  //Play no moves in game, since we're sampling initializations
  cfg.overrideKey("maxMovesPerGame","0");

  const bool isDistributed = false;
  PlaySettings playSettings = PlaySettings::loadForSelfplay(cfg, isDistributed);
  GameRunner* gameRunner = new GameRunner(cfg, playSettings, logger);

  for(int i = 0; i<numToGen; i++) {
    string seed = Global::uint64ToString(rand.nextUInt64());
    MatchPairer::BotSpec botSpec;
    botSpec.botIdx = 0;
    botSpec.botName = "";
    botSpec.nnEval = nnEval;
    botSpec.baseParams = Setup::loadSingleParams(cfg,Setup::SETUP_FOR_DISTRIBUTED);

    FinishedGameData* data = gameRunner->runGame(
      seed,
      botSpec,
      botSpec,
      NULL,
      NULL,
      logger,
      nullptr,
      nullptr,
      nullptr,
      nullptr,
      nullptr
    );

    cout << data->startHist.rules.toString() << endl;
    Board::printBoard(cout, data->startBoard, Board::NULL_LOC, &(data->startHist.moveHistory));
    cout << endl;
    if(evaluate) {
      evalBot->setPosition(data->startPla, data->startBoard, data->startHist);
      evalBot->genMoveSynchronous(data->startPla,TimeControls());
      ReportedSearchValues values = evalBot->getSearchStopAndWait()->getRootValuesRequireSuccess();
      cout << "Winloss: " << values.winLossValue << endl;
      cout << "Lead: " << values.lead << endl;
    }

    delete data;
  }

  delete gameRunner;
  delete evalBot;
  if(nnEval != NULL)
    delete nnEval;

  ScoreValue::freeTables();
  return 0;
}

int MainCmds::evalrandominits(const vector<string>& args) {
  Board::initHash();
  ScoreValue::initTables();

  ConfigParser cfg;
  string modelFile;
  try {
    KataGoCommandLine cmd("Eval random inits");
    cmd.addConfigFileArg("","");
    cmd.addModelFileArg();
    cmd.addOverrideConfigArg();

    cmd.parseArgs(args);

    cmd.getConfigAllowEmpty(cfg);
    if(cfg.getFileName() != "")
      modelFile = cmd.getModelFile();
  }
  catch (TCLAP::ArgException &e) {
    cerr << "Error: " << e.error() << " for argument " << e.argId() << endl;
    return 1;
  }

  Rand rand;

  const bool logToStdoutDefault = true;
  Logger logger(&cfg, logToStdoutDefault);

  NNEvaluator* nnEval = NULL;
  if(cfg.getFileName() != "") {
    SearchParams params = Setup::loadSingleParams(cfg,Setup::SETUP_FOR_GTP);
    {
      Setup::initializeSession(cfg);
      const int expectedConcurrentEvals = params.numThreads;
      const int defaultMaxBatchSize = std::max(8,((params.numThreads+3)/4)*4);
      const bool defaultRequireExactNNLen = false;
      const bool disableFP16 = false;
      const string expectedSha256 = "";
      nnEval = Setup::initializeNNEvaluator(
        modelFile,modelFile,expectedSha256,cfg,logger,rand,expectedConcurrentEvals,
        Board::MAX_LEN,Board::MAX_LEN,defaultMaxBatchSize,defaultRequireExactNNLen,disableFP16,
        Setup::SETUP_FOR_GTP
      );
    }
    logger.write("Loaded neural net");
  }

  Search* evalBot;
  {
    SearchParams params = Setup::loadSingleParams(cfg,Setup::SETUP_FOR_DISTRIBUTED);
    params.maxVisits = 40;
    params.numThreads = 1;
    string seed = Global::uint64ToString(rand.nextUInt64());
    evalBot = new Search(params, nnEval, &logger, seed);
  }

  Rand gameRand;
  while(true) {
    Board board(19,19);
    Player pla = P_BLACK;
    Rules rules = Rules::parseRules("japanese");
    BoardHistory hist(board,pla,rules,0);
    int numInitialMovesToPlay = (int)gameRand.nextUInt(200);
    double temperature = 1.0;
    for(int i = 0; i<numInitialMovesToPlay; i++) {
      NNResultBuf buf;
      Loc loc = PlayUtils::getGameInitializationMove(evalBot, evalBot, board, hist, pla, buf, gameRand, temperature);

      assert(hist.isLegal(board,loc,pla));
      hist.makeBoardMoveAssumeLegal(board,loc,pla,NULL);
      pla = getOpp(pla);

      hist.endGameIfAllPassAlive(board);
      if(hist.isGameFinished)
        break;
    }

    evalBot->setPosition(pla,board,hist);
    evalBot->runWholeSearch(pla);
    ReportedSearchValues values = evalBot->getRootValuesRequireSuccess();
    cout << numInitialMovesToPlay << "," << values.winLossValue << "," << values.lead << endl;
  }
  delete evalBot;
  if(nnEval != NULL)
    delete nnEval;

  ScoreValue::freeTables();
  return 0;
}

int MainCmds::searchentropyanalysis(const vector<string>& args) {
  Board::initHash();
  ScoreValue::initTables();

  ConfigParser cfg;
  string modelFile;
  string boardSizeDataset;
  try {
    KataGoCommandLine cmd("Analyze search entropy across datasets");
    cmd.addConfigFileArg(KataGoCommandLine::defaultGtpConfigFileName(),"gtp_example.cfg");
    cmd.addModelFileArg();
    TCLAP::ValueArg<string> boardSizeDatasetArg("","boardsizedataset", "Dataset to analyze (9,13,19,10x14,rectangle)", true, "", "SIZE");
    cmd.add(boardSizeDatasetArg);
    cmd.addOverrideConfigArg();

    cmd.parseArgs(args);

    modelFile = cmd.getModelFile();
    boardSizeDataset = boardSizeDatasetArg.getValue();
    cmd.getConfig(cfg);
  }
  catch (TCLAP::ArgException &e) {
    cerr << "Error: " << e.error() << " for argument " << e.argId() << endl;
    return 1;
  }

  Rand rand;

  const bool logToStdoutDefault = true;
  Logger logger(&cfg, logToStdoutDefault);
  logger.write("Search Entropy Analysis");
  logger.write("Model: " + modelFile);
  logger.write("Dataset: " + boardSizeDataset);

  SearchParams params = Setup::loadSingleParams(cfg,Setup::SETUP_FOR_GTP);
  NNEvaluator* nnEval = NULL;
  {
    Setup::initializeSession(cfg);
    const int expectedConcurrentEvals = params.numThreads;
    const int defaultMaxBatchSize = std::max(8,((params.numThreads+3)/4)*4);
    const bool defaultRequireExactNNLen = false;
    const bool disableFP16 = false;
    const string expectedSha256 = "";
    nnEval = Setup::initializeNNEvaluator(
      modelFile,modelFile,expectedSha256,cfg,logger,rand,expectedConcurrentEvals,
      Board::MAX_LEN,Board::MAX_LEN,defaultMaxBatchSize,defaultRequireExactNNLen,disableFP16,
      Setup::SETUP_FOR_GTP
    );
  }
  logger.write("Loaded neural net");

  Search* bot;
  {
    SearchParams searchParams = Setup::loadSingleParams(cfg,Setup::SETUP_FOR_GTP);
    string seed = Global::uint64ToString(rand.nextUInt64());
    bot = new Search(searchParams, nnEval, &logger, seed);
  }

  vector<string> sgfData;
  if(boardSizeDataset == "9")
    sgfData = TestCommon::getMultiGameSize9Data();
  else if(boardSizeDataset == "13")
    sgfData = TestCommon::getMultiGameSize13Data();
  else if(boardSizeDataset == "19")
    sgfData = TestCommon::getMultiGameSize19Data();
  else if(boardSizeDataset == "10x14")
    sgfData = TestCommon::getMultiGameSize10x14Data();
  else if(boardSizeDataset == "rectangle")
    sgfData = TestCommon::getMultiGameRectangleData();
  else {
    throw StringError("Unknown dataset to test gpu error on: " + boardSizeDataset);
  }

  // Build list of all (sgfIdx, turnIdx) pairs, then deterministically shuffle
  vector<std::pair<int,int>> positions;
  {
    vector<std::unique_ptr<CompactSgf>> sgfObjs;
    for(const string& sgf : sgfData)
      sgfObjs.push_back(CompactSgf::parse(sgf));
    for(int sgfIdx = 0; sgfIdx < (int)sgfObjs.size(); sgfIdx++) {
      for(int turnIdx = 0; turnIdx < (int)sgfObjs[sgfIdx]->moves.size(); turnIdx++) {
        positions.push_back({sgfIdx, turnIdx});
      }
    }
  }
  {
    Rand shuffleRand("searchentropyanalysis_shuffle_seed");
    shuffleRand.shuffle(positions);
  }
  logger.write("Total positions to search: " + Global::intToString((int)positions.size()));

  vector<double> searchEntropies;
  vector<double> searchSurprises;
  int numPositions = 0;

  auto printRunningStats = [&]() {
    cout << "Searched numPositions: " << numPositions << endl;
    if(numPositions > 0) {
      double entropyMean = 0.0;
      for(double v : searchEntropies) entropyMean += v;
      entropyMean /= numPositions;
      double entropyVar = 0.0;
      for(double v : searchEntropies) { double d = v - entropyMean; entropyVar += d * d; }
      entropyVar /= numPositions;
      cout << "Mean search entropy: " << entropyMean << " standard deviation: " << sqrt(entropyVar) << endl;

      double surpriseMean = 0.0;
      for(double v : searchSurprises) surpriseMean += v;
      surpriseMean /= numPositions;
      double surpriseVar = 0.0;
      for(double v : searchSurprises) { double d = v - surpriseMean; surpriseVar += d * d; }
      surpriseVar /= numPositions;
      cout << "Mean search surprise: " << surpriseMean << " standard deviation: " << sqrt(surpriseVar) << endl;
    }
  };

  for(const auto& pos : positions) {
    int sgfIdx = pos.first;
    int turnIdx = pos.second;

    std::unique_ptr<CompactSgf> sgfObj = CompactSgf::parse(sgfData[sgfIdx]);

    Board board;
    Player pla;
    BoardHistory hist;
    Rules initialRules;
    sgfObj->setupInitialBoardAndHist(initialRules, board, pla, hist);

    for(int i = 0; i < turnIdx; i++) {
      Loc moveLoc = sgfObj->moves[i].loc;
      if(moveLoc != Board::NULL_LOC && hist.isLegal(board, moveLoc, pla)) {
        hist.makeBoardMoveAssumeLegal(board, moveLoc, pla, NULL);
        pla = getOpp(pla);
      }
    }

    if(!hist.isGameFinished) {
      bot->setPosition(pla, board, hist);
      bot->runWholeSearch(pla);

      double surprise, searchEntropy, policyEntropy;
      if(bot->getPolicySurpriseAndEntropy(surprise, searchEntropy, policyEntropy)) {
        searchEntropies.push_back(searchEntropy);
        searchSurprises.push_back(surprise);
        numPositions++;
        if(numPositions % 10 == 0)
          printRunningStats();
      }
    }
  }

  cout << modelFile << endl;
  cout << "Dataset: " << boardSizeDataset << endl;
  printRunningStats();

  delete bot;
  delete nnEval;
  ScoreValue::freeTables();
  return 0;
}

//One CSV row per recorded turn of the game, containing the stats that drive surprise-based selection of
//positions (policy surprise, value surprise) along with the per-game and per-turn covariates needed to
//analyze what they vary with.
static void writeSurpriseStatsRows(ostream& out, int64_t gameIdx, const FinishedGameData& data) {
  size_t numMoves = data.targetWeightByTurn.size();
  testAssert(data.hasFullData);
  testAssert(data.policySurpriseByTurn.size() == numMoves);
  testAssert(data.policyEntropyByTurn.size() == numMoves);
  testAssert(data.searchEntropyByTurn.size() == numMoves);
  testAssert(data.valueSurpriseByTurn.size() == numMoves);
  testAssert(data.wasCheapSearchByTurn.size() == numMoves);
  testAssert(data.nnRawStatsByTurn.size() == numMoves);
  testAssert(data.policyTargetsByTurn.size() == numMoves);
  testAssert(data.reanalysisByTurn.size() == numMoves);
  testAssert(data.whiteValueTargetsByTurn.size() == numMoves + 1);

  //Last entry of the value targets is the actual game outcome.
  const ValueTargets& outcome = data.whiteValueTargetsByTurn[numMoves];
  double pdaWhite =
    data.playoutDoublingAdvantagePla == P_WHITE ? data.playoutDoublingAdvantage :
    data.playoutDoublingAdvantagePla == P_BLACK ? -data.playoutDoublingAdvantage : 0.0;
  //Turn number of the first recorded turn, counting handicap/policy-init/fork setup moves and any initial
  //turn number offset of positions begun from sgfs, so that turn number is comparable across game modes.
  int64_t startTurnIdx = data.startHist.initialTurnNumber + (int64_t)data.startHist.moveHistory.size();

  for(size_t t = 0; t<numMoves; t++) {
    const ValueTargets& mcts = data.whiteValueTargetsByTurn[t];
    const NNRawStats& raw = data.nnRawStatsByTurn[t];
    const ReanalysisData& rean = data.reanalysisByTurn[t];
    out << gameIdx
        << "," << data.gameHash.toString()
        << "," << data.startBoard.x_size
        << "," << data.startBoard.y_size
        << "," << data.startHist.rules.komi
        << "," << pdaWhite
        << "," << data.numExtraBlack
        << "," << data.mode
        << "," << (data.hitTurnLimit ? 1 : 0)
        << "," << numMoves
        << "," << outcome.win
        << "," << outcome.loss
        << "," << outcome.noResult
        << "," << outcome.score
        << "," << (startTurnIdx + (int64_t)t)
        << "," << t
        << "," << (data.wasCheapSearchByTurn[t] ? 1 : 0)
        << "," << data.policyTargetsByTurn[t].unreducedNumVisits
        << "," << data.targetWeightByTurn[t]
        << "," << data.policySurpriseByTurn[t]
        << "," << data.valueSurpriseByTurn[t]
        << "," << data.policyEntropyByTurn[t]
        << "," << data.searchEntropyByTurn[t]
        << "," << raw.whiteWinLoss
        << "," << raw.whiteScoreMean
        << "," << mcts.win
        << "," << mcts.loss
        << "," << mcts.noResult
        << "," << mcts.score
        << "," << (rean.wasReanalyzed ? 1 : 0)
        << "," << rean.selectionPolicySurprise
        << "," << rean.selectionValueSurprise
        << "," << rean.originalNumVisits
        << "\n";
  }
}

int MainCmds::selfplaysurprisedump(const vector<string>& args) {
  Board::initHash();
  ScoreValue::initTables();

  ConfigParser cfg;
  string modelFile;
  string outputFile;
  int64_t numGamesTotal;
  bool reanalyzeAll;
  try {
    KataGoCommandLine cmd("Run selfplay games with a fixed model and dump per-position policy and value surprise stats to csv");
    cmd.addConfigFileArg("","selfplay config file");
    cmd.addModelFileArg();
    cmd.addOverrideConfigArg();

    TCLAP::ValueArg<string> outputArg("","output","Csv file to write one row per recorded turn to",true,string(),"FILE");
    TCLAP::ValueArg<int> numGamesArg("","num-games","Number of games to run",true,0,"N");
    TCLAP::SwitchArg reanalyzeAllArg("","reanalyze-all","Instead of disabling reanalysis, reanalyze every cheap-search position, so each such row records both the cheap search's (before) and the full search's (after) surprise stats");
    cmd.add(outputArg);
    cmd.add(numGamesArg);
    cmd.add(reanalyzeAllArg);
    cmd.parseArgs(args);
    outputFile = outputArg.getValue();
    numGamesTotal = numGamesArg.getValue();
    reanalyzeAll = reanalyzeAllArg.getValue();
    if(numGamesTotal <= 0)
      throw StringError("-num-games must be positive");

    cmd.getConfig(cfg);
    modelFile = cmd.getModelFile();
  }
  catch (TCLAP::ArgException &e) {
    cerr << "Error: " << e.error() << " for argument " << e.argId() << endl;
    return 1;
  }

  Rand seedRand;

  const bool logToStdoutDefault = true;
  Logger logger(&cfg, logToStdoutDefault);
  logger.write("Selfplay surprise stats dump");
  logger.write("Model: " + modelFile);

  const int numGameThreads = cfg.getInt("numGameThreads",1,16384);
  const string gameSeedBase = Global::uint64ToHexString(seedRand.nextUInt64());

  const SearchParams baseParams = Setup::loadSingleParams(cfg,Setup::SETUP_FOR_OTHER);
  const bool isDistributed = false;
  PlaySettings playSettings = PlaySettings::loadForSelfplay(cfg, isDistributed);
  if(reanalyzeAll) {
    //Reanalyze every cheap-search position, so that every cheap-search row records both the cheap search's
    //surprise stats (the ones that would drive reanalysis selection, preserved as origPolicySurprise and
    //origValueSurprise) and the full search's stats, without any selection bias in which positions get both.
    //Note that the post-reanalysis value surprise of a turn also absorbs the value changes of other reanalyzed
    //turns after it, via the smoothed forward-outcome track, so unlike policy surprise it is not a pure
    //cheap-vs-full comparison of this turn's search alone.
    logger.write("Reanalyzing all cheap-search positions to record before and after surprise stats");
    playSettings.useReanalyze = true;
    playSettings.reanalyzeProp = 1.0;
  }
  else {
    //Force no reanalysis, so that every turn's recorded stats are those of the search the game actually played
    //with - exactly the stats that would drive reanalysis selection - rather than partly overwritten by
    //reanalysis searches.
    if(playSettings.useReanalyze && playSettings.reanalyzeProp > 0.0)
      logger.write("Config has reanalyzeProp > 0, forcing it to 0 so that recorded stats are the pre-reanalysis ones");
    playSettings.reanalyzeProp = 0.0;
  }

  GameRunner* gameRunner = new GameRunner(cfg, playSettings, logger);

  const int minBoardXSizeUsed = gameRunner->getGameInitializer()->getMinBoardXSize();
  const int minBoardYSizeUsed = gameRunner->getGameInitializer()->getMinBoardYSize();
  const int maxBoardXSizeUsed = gameRunner->getGameInitializer()->getMaxBoardXSize();
  const int maxBoardYSizeUsed = gameRunner->getGameInitializer()->getMaxBoardYSize();

  Setup::initializeSession(cfg);
  NNEvaluator* nnEval;
  {
    const int expectedConcurrentEvals = cfg.getInt("numSearchThreads") * numGameThreads;
    const bool defaultRequireExactNNLen = minBoardXSizeUsed == maxBoardXSizeUsed && minBoardYSizeUsed == maxBoardYSizeUsed;
    const int defaultMaxBatchSize = -1;
    const bool disableFP16 = false;
    const string expectedSha256 = "";
    nnEval = Setup::initializeNNEvaluator(
      modelFile,modelFile,expectedSha256,cfg,logger,seedRand,expectedConcurrentEvals,
      maxBoardXSizeUsed,maxBoardYSizeUsed,defaultMaxBatchSize,defaultRequireExactNNLen,disableFP16,
      Setup::SETUP_FOR_OTHER
    );
  }
  logger.write("Loaded neural net");

  cfg.warnUnusedKeys(cerr,&logger);

  ofstream out;
  FileUtils::open(out,outputFile);
  out.precision(9);
  out << "gameIdx,gameHash,xSize,ySize,komi,pdaWhite,numExtraBlack,mode,hitTurnLimit,gameNumMoves"
      << ",outcomeWhiteWin,outcomeWhiteLoss,outcomeWhiteNoResult,outcomeWhiteScore"
      << ",turnIdx,turnAfterStart,wasCheapSearch,unreducedNumVisits,targetWeight"
      << ",policySurprise,valueSurprise,policyEntropy,searchEntropy"
      << ",rawNNWhiteWinLoss,rawNNWhiteScoreMean,mctsWhiteWin,mctsWhiteLoss,mctsWhiteNoResult,mctsWhiteScore"
      << ",wasReanalyzed,origPolicySurprise,origValueSurprise,origNumVisits"
      << "\n";

  if(!std::atomic_is_lock_free(&sigReceived))
    throw StringError("sigReceived is not lock free, signal-quitting mechanism will NOT work!");
  std::signal(SIGINT, signalHandler);
  std::signal(SIGTERM, signalHandler);

  std::mutex writeMutex;
  std::atomic<int64_t> numGamesStarted(0);
  int64_t numGamesWritten = 0;
  ForkData* forkData = new ForkData();

  auto gameLoop = [
    &gameRunner,&logger,&numGamesStarted,&forkData,numGamesTotal,&baseParams,&gameSeedBase,
    &nnEval,&writeMutex,&out,&numGamesWritten
  ](int threadIdx) {
    (void)threadIdx;
    auto shouldStopFunc = []() noexcept {
      return sigReceived.load();
    };
    Rand thisLoopSeedRand;
    while(true) {
      if(sigReceived.load())
        break;
      int64_t gameIdx = numGamesStarted.fetch_add(1,std::memory_order_acq_rel);
      if(gameIdx >= numGamesTotal)
        break;

      MatchPairer::BotSpec botSpec;
      botSpec.botIdx = 0;
      botSpec.botName = nnEval->getModelName();
      botSpec.nnEval = nnEval;
      botSpec.baseParams = baseParams;

      string seed = gameSeedBase + ":" + Global::uint64ToHexString(thisLoopSeedRand.nextUInt64());
      FinishedGameData* gameData = gameRunner->runGame(
        seed, botSpec, botSpec, forkData, NULL, logger,
        shouldStopFunc, nullptr, nullptr, nullptr, nullptr
      );
      //Happens when interrupted by the signal handler mid-game
      if(gameData == NULL)
        break;

      {
        std::lock_guard<std::mutex> lock(writeMutex);
        writeSurpriseStatsRows(out, gameIdx, *gameData);
        out.flush();
        numGamesWritten += 1;
        logger.write(
          "Finished game " + Global::int64ToString(gameIdx) + " (" +
          Global::int64ToString(numGamesWritten) + "/" + Global::int64ToString(numGamesTotal) + " written)"
        );
      }
      delete gameData;
    }
  };

  vector<std::thread> threads;
  for(int i = 0; i<numGameThreads; i++)
    threads.push_back(std::thread(gameLoop,i));
  for(int i = 0; i<numGameThreads; i++)
    threads[i].join();

  out.close();
  logger.write("Wrote stats for " + Global::int64ToString(numGamesWritten) + " games to " + outputFile);

  delete forkData;
  delete gameRunner;
  delete nnEval;
  ScoreValue::freeTables();
  return 0;
}
