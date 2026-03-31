//command/tuneparams.cpp
//KataGo hyperparameter tuning via QRS-Tune sequential optimization.
//Runs two bots (base reference vs experiment) for numTrials games,
//adapting experiment bot's PUCT parameters toward higher win rates.

#include "../core/global.h"
#include "../core/config_parser.h"
#include "../core/logger.h"
#include "../core/rand.h"
#include "../search/searchparams.h"
#include "../program/setup.h"
#include "../program/play.h"
#include "../program/playsettings.h"
#include "../command/commandline.h"
#include "../main.h"

#include "../qrstune/QRSOptimizer.h"

#include <vector>
#include <algorithm>
#include <csignal>

using namespace std;

static std::atomic<bool> sigReceived(false);
static std::atomic<bool> shouldStop(false);
static void signalHandler(int signal)
{
  if(signal == SIGINT || signal == SIGTERM) {
    sigReceived.store(true);
    shouldStop.store(true);
  }
}

//Number of dimensions = number of PUCT params being tuned
static const int nDims = 3;

static const char* paramNames[nDims] = {
  "cpuctExploration",
  "cpuctExplorationLog",
  "cpuctUtilityStdevPrior"
};

//Default search ranges (used when config keys are absent)
static const double qrsDefaultMins[nDims] = {0.5,  0.05, 0.1};
static const double qrsDefaultMaxs[nDims] = {2.0,  1.0,  0.8};

//Config keys for per-dimension search ranges
static const char* rangeMinKeys[nDims] = {
  "cpuctExplorationMin", "cpuctExplorationLogMin", "cpuctUtilityStdevPriorMin"
};
static const char* rangeMaxKeys[nDims] = {
  "cpuctExplorationMax", "cpuctExplorationLogMax", "cpuctUtilityStdevPriorMax"
};

//Map QRS-Tune normalized coordinate x in [-1,+1] to real PUCT value.
static double qrsDimToReal(int dim, double x, const double* mins, const double* maxs) {
  double center = (mins[dim] + maxs[dim]) * 0.5;
  double radius = (maxs[dim] - mins[dim]) * 0.5;
  return center + x * radius;
}

static void qrsToPUCT(
  const vector<double>& x,
  double& cpuctExploration,
  double& cpuctExplorationLog,
  double& cpuctUtilityStdevPrior,
  const double* mins, const double* maxs
) {
  cpuctExploration       = qrsDimToReal(0, x[0], mins, maxs);
  cpuctExplorationLog    = qrsDimToReal(1, x[1], mins, maxs);
  cpuctUtilityStdevPrior = qrsDimToReal(2, x[2], mins, maxs);
}

//Print ASCII-art regression curve for each PUCT dimension.
//For dimension d: fix all other dims at vBest, sweep d from -1 to +1.
static void printRegressionCurves(const QRSTune::QRSTuner& tuner,
                                   const vector<double>& vBest,
                                   const double* mins, const double* maxs,
                                   Logger& logger) {
  const int plotW = 60;
  const int plotH = 20;
  double bestWinRate = tuner.model().predict(vBest.data());

  for(int dim = 0; dim < nDims; dim++) {
    vector<string> canvas(plotH, string(plotW, ' '));

    int bestCol = (int)((vBest[dim] + 1.0) / 2.0 * (plotW - 1) + 0.5);
    bestCol = max(0, min(plotW - 1, bestCol));

    vector<double> xSlice(vBest);
    for(int col = 0; col < plotW; col++) {
      double t = -1.0 + 2.0 * col / (plotW - 1);
      xSlice[dim] = t;
      double winRate = tuner.model().predict(xSlice.data());

      int row = (int)((1.0 - winRate) * (plotH - 1) + 0.5);
      row = max(0, min(plotH - 1, row));
      canvas[row][col] = (col == bestCol) ? '*' : 'o';
    }

    double bestReal = qrsDimToReal(dim, vBest[dim], mins, maxs);
    logger.write("");
    logger.write(
      "[Dim " + Global::intToString(dim) + "] " + paramNames[dim] +
      "  (best QRS=" + Global::strprintf("%.3f", vBest[dim]) +
      " -> real=" + Global::strprintf("%.3f", bestReal) +
      ", est.winrate=" + Global::strprintf("%.3f", bestWinRate) + ")"
    );

    for(int row = 0; row < plotH; row++) {
      string label;
      if(row == 0)               label = "1.0 |";
      else if(row == plotH / 2) label = "0.5 |";
      else if(row == plotH - 1) label = "0.0 |";
      else                       label = "    |";
      logger.write(label + canvas[row]);
    }
    logger.write("    +" + string(plotW, '-'));

    {
      string line(plotW + 5, ' ');
      const int off = 5;
      auto place = [&](int col, const string& lbl) {
        int pos = off + col - (int)lbl.size() / 2;
        if(pos < 0) pos = 0;
        for(int i = 0; i < (int)lbl.size() && pos + i < (int)line.size(); i++)
          line[pos + i] = lbl[i];
      };
      place(0,          Global::strprintf("%.3f", qrsDimToReal(dim, -1.0, mins, maxs)));
      place(plotW / 2, Global::strprintf("%.3f", qrsDimToReal(dim,  0.0, mins, maxs)));
      place(plotW - 1, Global::strprintf("%.3f", qrsDimToReal(dim, +1.0, mins, maxs)));
      size_t last = line.find_last_not_of(' ');
      logger.write(line.substr(0, last + 1));
    }
  }
  logger.write("");
}

int MainCmds::tuneparams(const vector<string>& args) {
  Board::initHash();
  ScoreValue::initTables();
  Rand seedRand;

  ConfigParser cfg;
  string logFile;
  try {
    KataGoCommandLine cmd(
      "Tune KataGo hyperparameters using sequential optimization (QRS-Tune).\n"
      "Runs numTrials games between a fixed reference bot (bot0) and an\n"
      "experiment bot (bot1) whose PUCT parameters are adapted each trial."
    );
    cmd.addConfigFileArg("", "tune_params.cfg");

    TCLAP::ValueArg<string> logFileArg("", "log-file", "Log file to output to", false, string(), "FILE");
    cmd.add(logFileArg);
    cmd.setShortUsageArgLimit();
    cmd.addOverrideConfigArg();

    cmd.parseArgs(args);
    logFile = logFileArg.getValue();
    cmd.getConfig(cfg);
  }
  catch(TCLAP::ArgException& e) {
    cerr << "Error: " << e.error() << " for argument " << e.argId() << endl;
    return 1;
  }

  Logger logger(&cfg);
  logger.addFile(logFile);
  logger.write("tune-params starting...");
  logger.write(string("Git revision: ") + Version::getGitRevision());

  //Read tuning-specific config
  int numTrials = cfg.getInt("numTrials", 1, 100000);

  //Search ranges (configurable; defaults preserve prior behaviour)
  double qrsMins[nDims], qrsMaxs[nDims];
  for(int d = 0; d < nDims; d++) {
    qrsMins[d] = cfg.contains(rangeMinKeys[d])
                    ? cfg.getDouble(rangeMinKeys[d], -1e9, 1e9)
                    : qrsDefaultMins[d];
    qrsMaxs[d] = cfg.contains(rangeMaxKeys[d])
                    ? cfg.getDouble(rangeMaxKeys[d], -1e9, 1e9)
                    : qrsDefaultMaxs[d];
    if(qrsMins[d] >= qrsMaxs[d])
      throw StringError(
        string("tune-params: ") + rangeMinKeys[d] + " must be < " + rangeMaxKeys[d]);
  }
  logger.write(
    "QRS ranges: cpuctExploration=[" +
    Global::strprintf("%.4f", qrsMins[0]) + "," + Global::strprintf("%.4f", qrsMaxs[0]) +
    "] cpuctExplorationLog=[" +
    Global::strprintf("%.4f", qrsMins[1]) + "," + Global::strprintf("%.4f", qrsMaxs[1]) +
    "] cpuctUtilityStdevPrior=[" +
    Global::strprintf("%.4f", qrsMins[2]) + "," + Global::strprintf("%.4f", qrsMaxs[2]) + "]"
  );

  //Load search params for both bots
  vector<SearchParams> paramss = Setup::loadParams(cfg, Setup::SETUP_FOR_MATCH);
  if((int)paramss.size() < 2)
    throw StringError("tune-params: config must define numBots = 2 (bot0 = reference, bot1 = experiment)");

  //Model files
  string nnModelFile0 = cfg.getString("nnModelFile0");
  string nnModelFile1 = cfg.getString("nnModelFile1");
  vector<string> nnModelFiles = {nnModelFile0, nnModelFile1};

  //Game runner setup
  PlaySettings playSettings = PlaySettings::loadForMatch(cfg);
  GameRunner* gameRunner = new GameRunner(cfg, playSettings, logger);
  int maxBoardX = gameRunner->getGameInitializer()->getMaxBoardXSize();
  int maxBoardY = gameRunner->getGameInitializer()->getMaxBoardYSize();

  //Initialize neural net inference
  Setup::initializeSession(cfg);
  const int expectedConcurrentEvals = max(paramss[0].numThreads, paramss[1].numThreads);
  vector<string> expectedSha256s;
  vector<NNEvaluator*> nnEvals = Setup::initializeNNEvaluators(
    nnModelFiles, nnModelFiles, expectedSha256s,
    cfg, logger, seedRand,
    expectedConcurrentEvals,
    maxBoardX, maxBoardY,
    /*defaultMaxBatchSize=*/-1,
    /*defaultRequireExactNNLen=*/(maxBoardX == gameRunner->getGameInitializer()->getMinBoardXSize() &&
                                  maxBoardY == gameRunner->getGameInitializer()->getMinBoardYSize()),
    /*disableFP16=*/false,
    Setup::SETUP_FOR_MATCH
  );
  logger.write("Loaded neural nets");

  //Signal handling for graceful shutdown
  if(!std::atomic_is_lock_free(&shouldStop))
    throw StringError("shouldStop is not lock free, signal-quitting mechanism for terminating will NOT work!");
  std::signal(SIGINT, signalHandler);
  std::signal(SIGTERM, signalHandler);

  //QRS-Tune setup
  uint64_t qrsSeed = seedRand.nextUInt64();
  QRSTune::QRSTuner tuner(nDims, qrsSeed, numTrials);

  const string gameSeedBase = Global::uint64ToHexString(seedRand.nextUInt64());

  int wins = 0, losses = 0, draws = 0;

  logger.write("Starting " + Global::intToString(numTrials) + " tuning trials");

  for(int trial = 0; trial < numTrials; trial++) {
    vector<double> sample = tuner.nextSample();

    double cpuctExploration, cpuctExplorationLog, cpuctUtilityStdevPrior;
    qrsToPUCT(sample, cpuctExploration, cpuctExplorationLog, cpuctUtilityStdevPrior, qrsMins, qrsMaxs);

    SearchParams expParams = paramss[1];
    expParams.cpuctExploration       = cpuctExploration;
    expParams.cpuctExplorationLog    = cpuctExplorationLog;
    expParams.cpuctUtilityStdevPrior = cpuctUtilityStdevPrior;

    //Alternate colors to remove first-move advantage bias
    bool expIsBlack = (trial % 2 == 0);
    MatchPairer::BotSpec botSpecB, botSpecW;
    if(expIsBlack) {
      botSpecB.botIdx     = 1;
      botSpecB.botName    = "experiment";
      botSpecB.nnEval     = nnEvals[1];
      botSpecB.baseParams = expParams;
      botSpecW.botIdx     = 0;
      botSpecW.botName    = "base";
      botSpecW.nnEval     = nnEvals[0];
      botSpecW.baseParams = paramss[0];
    } else {
      botSpecB.botIdx     = 0;
      botSpecB.botName    = "base";
      botSpecB.nnEval     = nnEvals[0];
      botSpecB.baseParams = paramss[0];
      botSpecW.botIdx     = 1;
      botSpecW.botName    = "experiment";
      botSpecW.nnEval     = nnEvals[1];
      botSpecW.baseParams = expParams;
    }

    string seed = gameSeedBase + ":" + Global::intToString(trial);
    auto shouldStopFunc = []() noexcept { return shouldStop.load(); };

    FinishedGameData* gameData = gameRunner->runGame(
      seed, botSpecB, botSpecW,
      /*forkData=*/NULL,
      /*startPosSample=*/NULL,
      logger,
      shouldStopFunc,
      /*shouldPause=*/NULL,
      /*checkForNewNNEval=*/NULL,
      /*afterInitialization=*/NULL,
      /*onEachMove=*/NULL
    );

    double outcome = 0.5;
    if(gameData != NULL) {
      Player winner = gameData->endHist.winner;
      if(expIsBlack) {
        if(winner == P_BLACK)       { outcome = 1.0; wins++; }
        else if(winner == P_WHITE)  { outcome = 0.0; losses++; }
        else                        { outcome = 0.5; draws++; }
      } else {
        if(winner == P_WHITE)       { outcome = 1.0; wins++; }
        else if(winner == P_BLACK)  { outcome = 0.0; losses++; }
        else                        { outcome = 0.5; draws++; }
      }
      delete gameData;
    } else {
      draws++;
      logger.write("Warning: trial " + Global::intToString(trial) + " returned null game data");
    }

    tuner.addResult(sample, outcome);

    if(shouldStop.load())
      break;

    //Progress report every 100 trials
    if((trial + 1) % 100 == 0) {
      vector<double> vBest = tuner.bestCoords();
      double bE, bLog, bStdev;
      qrsToPUCT(vBest, bE, bLog, bStdev, qrsMins, qrsMaxs);
      logger.write(
        "Trial " + Global::intToString(trial + 1) + "/" + Global::intToString(numTrials) +
        " | W=" + Global::intToString(wins) + " L=" + Global::intToString(losses) + " D=" + Global::intToString(draws) +
        " | best: cpuctExploration=" + Global::doubleToString(bE) +
        " cpuctExplorationLog=" + Global::doubleToString(bLog) +
        " cpuctUtilityStdevPrior=" + Global::doubleToString(bStdev)
      );
    }
  }

  //Final result
  vector<double> vBest = tuner.bestCoords();
  double bestE, bestLog, bestStdev;
  qrsToPUCT(vBest, bestE, bestLog, bestStdev, qrsMins, qrsMaxs);

  logger.write("");
  logger.write("=== tune-params Results ===");
  logger.write(
    "Trials: " + Global::intToString(numTrials) +
    "  Wins: " + Global::intToString(wins) +
    "  Losses: " + Global::intToString(losses) +
    "  Draws: " + Global::intToString(draws)
  );
  logger.write("Best cpuctExploration       = " + Global::doubleToString(bestE));
  logger.write("Best cpuctExplorationLog    = " + Global::doubleToString(bestLog));
  logger.write("Best cpuctUtilityStdevPrior = " + Global::doubleToString(bestStdev));
  logger.write(
    "QRS raw coordinates: [" + Global::doubleToString(vBest[0]) + ", " +
    Global::doubleToString(vBest[1]) + ", " + Global::doubleToString(vBest[2]) + "]"
  );

  //ASCII-art regression curves (one per PUCT dimension)
  printRegressionCurves(tuner, vBest, qrsMins, qrsMaxs, logger);

  //Cleanup
  delete gameRunner;
  for(NNEvaluator* eval : nnEvals)
    delete eval;

  NeuralNet::globalCleanup();
  ScoreValue::freeTables();
  return 0;
}
