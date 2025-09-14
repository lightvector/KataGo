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

#include <chrono>
#include <csignal>
#include <cmath>

using namespace std;

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

  vector<double> searchEntropies;
  vector<double> searchSurprises;
  int numPositions = 0;

  for(const string& sgf : sgfData) {
    CompactSgf* sgfObj = CompactSgf::parse(sgf);

    for(int turnIdx = 0; turnIdx < sgfObj->moves.size(); turnIdx++) {
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
            cout << "Searched numPositions: " << numPositions << endl;
        }
      }
    }
    delete sgfObj;
  }

  cout << modelFile << endl;
  cout << "Dataset: " << boardSizeDataset << endl;
  cout << "Num positions: " << numPositions << endl;

  {
    double mean = 0.0;
    for(double entropy : searchEntropies) {
      mean += entropy;
    }
    mean /= numPositions;

    double variance = 0.0;
    for(double entropy : searchEntropies) {
      double diff = entropy - mean;
      variance += diff * diff;
    }
    variance /= numPositions;
    double stdev = sqrt(variance);

    cout << "Mean search entropy: " << mean << endl;
    cout << "Standard deviation: " << stdev << endl;
  }
  {
    double mean = 0.0;
    for(double surprise : searchSurprises) {
      mean += surprise;
    }
    mean /= numPositions;

    double variance = 0.0;
    for(double surprise : searchSurprises) {
      double diff = surprise - mean;
      variance += diff * diff;
    }
    variance /= numPositions;
    double stdev = sqrt(variance);

    cout << "Mean search surprise: " << mean << endl;
    cout << "Standard deviation: " << stdev << endl;
  }

  delete bot;
  delete nnEval;
  ScoreValue::freeTables();
  return 0;
}
