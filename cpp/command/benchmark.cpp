#include "../core/global.h"
#include "../core/config_parser.h"
#include "../core/timer.h"
#include "../dataio/sgf.h"
#include "../search/asyncbot.h"
#include "../program/setup.h"
#include "../program/playutils.h"
#include "../program/gtpconfig.h"
#include "../tests/tests.h"
#include "../command/commandline.h"
#include "../main.h"

#include <chrono>
#include <map>
#include <sstream>
#include <fstream>

#include <boost/filesystem.hpp>
namespace bfs = boost::filesystem;

using namespace std;

static NNEvaluator* createNNEval(int maxNumThreads, CompactSgf* sgf, const string& modelFile, Logger& logger, ConfigParser& cfg, const SearchParams& params);

static vector<PlayUtils::BenchmarkResults> doFixedTuneThreads(
  const SearchParams& params,
  const CompactSgf* sgf,
  int numPositionsPerGame,
  NNEvaluator*& nnEval,
  Logger& logger,
  double secondsPerGameMove,
  vector<int> numThreadsToTest,
  bool printElo
);
static vector<PlayUtils::BenchmarkResults> doAutoTuneThreads(
  const SearchParams& params,
  const CompactSgf* sgf,
  int numPositionsPerGame,
  NNEvaluator*& nnEval,
  Logger& logger,
  double secondsPerGameMove,
  std::function<void(int)> reallocateNNEvalWithEnoughBatchSize
);

#ifdef USE_EIGEN_BACKEND
static const int64_t defaultMaxVisits = 80;
#else
static const int64_t defaultMaxVisits = 800;
#endif

static constexpr double defaultSecondsPerGameMove = 5.0;
static const int ternarySearchInitialMax = 32;

int MainCmds::benchmark(int argc, const char* const* argv) {
  Board::initHash();
  ScoreValue::initTables();

  ConfigParser cfg;
  string modelFile;
  string sgfFile;
  int boardSize;
  int64_t maxVisits;
  vector<int> numThreadsToTest;
  int numPositionsPerGame;
  bool autoTuneThreads;
  int secondsPerGameMove;
  try {
    KataGoCommandLine cmd("Benchmark with gtp config to test speed with different numbers of threads.");
    cmd.addConfigFileArg(KataGoCommandLine::defaultGtpConfigFileName(),"gtp_example.cfg");
    cmd.addModelFileArg();
    TCLAP::ValueArg<long> visitsArg("v","visits","How many visits to use per search (default " + Global::int64ToString(defaultMaxVisits) + ")",false,(long)defaultMaxVisits,"VISITS");
    TCLAP::ValueArg<string> threadsArg("t","threads","Test these many threads, comma-separated, e.g. '4,8,12,16' ",false,"","THREADS");
    TCLAP::ValueArg<int> numPositionsPerGameArg("n","numpositions","How many positions to sample from a game (default 10)",false,10,"NUM");
    TCLAP::ValueArg<string> sgfFileArg("","sgf", "Optional game to sample positions from (default: uses a built-in-set of positions)",false,string(),"FILE");
    TCLAP::ValueArg<int> boardSizeArg("","boardsize", "Size of board to benchmark on (9-19), default 19",false,-1,"SIZE");
    TCLAP::SwitchArg autoTuneThreadsArg("s","tune","Automatically search for the optimal number of threads (default if not specifying specific numbers of threads)");
    TCLAP::ValueArg<double> secondsPerGameMoveArg("i","time","Typical amount of time per move spent while playing, in seconds (default " +
                                               Global::doubleToString(defaultSecondsPerGameMove) + ")",false,defaultSecondsPerGameMove,"SECONDS");
    cmd.add(visitsArg);
    cmd.add(threadsArg);
    cmd.add(numPositionsPerGameArg);

    cmd.setShortUsageArgLimit();

    cmd.addOverrideConfigArg();

    cmd.add(sgfFileArg);
    cmd.add(boardSizeArg);
    cmd.add(autoTuneThreadsArg);
    cmd.add(secondsPerGameMoveArg);
    cmd.parse(argc,argv);

    modelFile = cmd.getModelFile();
    sgfFile = sgfFileArg.getValue();
    boardSize = boardSizeArg.getValue();
    maxVisits = (int64_t)visitsArg.getValue();
    string desiredThreadsStr = threadsArg.getValue();
    numPositionsPerGame = numPositionsPerGameArg.getValue();
    autoTuneThreads = autoTuneThreadsArg.getValue();
    secondsPerGameMove = secondsPerGameMoveArg.getValue();

    if(boardSize != -1 && sgfFile != "")
      throw StringError("Cannot specify both -sgf and -boardsize at the same time");
    if(boardSize != -1 && (boardSize < 9 || boardSize > 19))
      throw StringError("Board size to test: invalid value " + Global::intToString(boardSize));
    if(maxVisits <= 1 || maxVisits >= 1000000000)
      throw StringError("Number of visits to use: invalid value " + Global::int64ToString(maxVisits));
    if(numPositionsPerGame <= 0 || numPositionsPerGame > 100000)
      throw StringError("Number of positions per game to use: invalid value " + Global::intToString(numPositionsPerGame));
    if(secondsPerGameMove <= 0 || secondsPerGameMove > 1000000)
      throw StringError("Number of seconds per game move to assume: invalid value " + Global::doubleToString(secondsPerGameMove));
    if(desiredThreadsStr != "" && autoTuneThreads)
      throw StringError("Cannot both automatically tune threads and specify fixed exact numbers of threads to test");

    //Apply default
    if(desiredThreadsStr == "")
      autoTuneThreads = true;

    if(!autoTuneThreads) {
      vector<string> desiredThreadsPieces = Global::split(desiredThreadsStr,',');
      for(int i = 0; i<desiredThreadsPieces.size(); i++) {
        string s = Global::trim(desiredThreadsPieces[i]);
        if(s == "")
          continue;
        int desiredThreads;
        bool suc = Global::tryStringToInt(s,desiredThreads);
        if(!suc || desiredThreads <= 0 || desiredThreads > 1024)
          throw StringError("Number of threads to use: invalid value: " + s);
        numThreadsToTest.push_back(desiredThreads);
      }

      if(numThreadsToTest.size() <= 0) {
        throw StringError("Must specify at least one valid value for -threads");
      }
    }

    cmd.getConfig(cfg);
  }
  catch (TCLAP::ArgException &e) {
    cerr << "Error: " << e.error() << " for argument " << e.argId() << endl;
    return 1;
  }

  CompactSgf* sgf;
  if(sgfFile != "") {
    sgf = CompactSgf::loadFile(sgfFile);
  }
  else {
    if(boardSize == -1)
      boardSize = 19;

    string sgfData = TestCommon::getBenchmarkSGFData(boardSize);
    sgf = CompactSgf::parse(sgfData);
  }

  Logger logger;
  logger.setLogToStdout(true);
  logger.write("Loading model and initializing benchmark...");

  SearchParams params = Setup::loadSingleParams(cfg);
  params.maxVisits = maxVisits;
  params.maxPlayouts = maxVisits;
  params.maxTime = 1e20;
  params.searchFactorAfterOnePass = 1.0;
  params.searchFactorAfterTwoPass = 1.0;

  Setup::initializeSession(cfg);

  if(cfg.contains("nnMaxBatchSize"))
    cout << "WARNING: Your nnMaxBatchSize is hardcoded to " + cfg.getString("nnMaxBatchSize") + ", ignoring it and assuming it is >= threads, for this benchmark." << endl;

  NNEvaluator* nnEval = NULL;
  auto reallocateNNEvalWithEnoughBatchSize = [&](int maxNumThreads) {
    if(nnEval != NULL)
      delete nnEval;
    nnEval = createNNEval(maxNumThreads, sgf, modelFile, logger, cfg, params);
  };

  if(!autoTuneThreads) {
    int maxThreads = 1;
    for(int i = 0; i<numThreadsToTest.size(); i++) {
      maxThreads = std::max(maxThreads,numThreadsToTest[i]);
    }
    reallocateNNEvalWithEnoughBatchSize(maxThreads);
  }
  else
    reallocateNNEvalWithEnoughBatchSize(ternarySearchInitialMax);

  logger.write("Loaded config " + cfg.getFileName());
  logger.write("Loaded model "+ modelFile);

  cout << endl;
  cout << "Testing using " << maxVisits << " visits." << endl;
  if(maxVisits == defaultMaxVisits) {
    cout << "  If you have a good GPU, you might increase this using \"-visits N\" to get more accurate results." << endl;
    cout << "  If you have a weak GPU and this is taking forever, you can decrease it instead to finish the benchmark faster." << endl;
  }

  cout << endl;

#ifdef USE_CUDA_BACKEND
  cout << "Your GTP config is currently set to cudaUseFP16 = " << nnEval->getUsingFP16Mode().toString()
       << " and cudaUseNHWC = " << nnEval->getUsingNHWCMode().toString() << endl;
  if(nnEval->getUsingFP16Mode() == enabled_t::False)
    cout << "If you have a strong GPU capable of FP16 tensor cores (e.g. RTX2080) setting these both to true may give a large performance boost." << endl;
#endif
#ifdef USE_OPENCL_BACKEND
  cout << "You are currently using the OpenCL version of KataGo." << endl;
  cout << "If you have a strong GPU capable of FP16 tensor cores (e.g. RTX2080), "
       << "using the Cuda version of KataGo instead may give a mild performance boost." << endl;
#endif
#ifdef USE_EIGEN_BACKEND
  cout << "You are currently using the Eigen (CPU) version of KataGo. Due to having no GPU, it may be slow." << endl;
#endif
  cout << endl;
  cout << "Your GTP config is currently set to use numSearchThreads = " << params.numThreads << endl;

  vector<PlayUtils::BenchmarkResults> results;
  if(!autoTuneThreads) {
    results = doFixedTuneThreads(params,sgf,numPositionsPerGame,nnEval,logger,secondsPerGameMove,numThreadsToTest,true);
  }
  else {
    results = doAutoTuneThreads(params,sgf,numPositionsPerGame,nnEval,logger,secondsPerGameMove,reallocateNNEvalWithEnoughBatchSize);
  }

  if(numThreadsToTest.size() > 1 || autoTuneThreads) {
    PlayUtils::BenchmarkResults::printEloComparison(results,secondsPerGameMove);

    cout << "If you care about performance, you may want to edit numSearchThreads in " << cfg.getFileName() << " based on the above results!" << endl;
    if(cfg.contains("nnMaxBatchSize"))
      cout << "WARNING: Your nnMaxBatchSize is hardcoded to " + cfg.getString("nnMaxBatchSize") + ", recommend deleting it and using the default (which this benchmark assumes)" << endl;
#ifdef USE_EIGEN_BACKEND
    if(cfg.contains("numNNServerThreadsPerModel")) {
      cout << "WARNING: Your numNNServerThreadsPerModel is hardcoded to " + cfg.getString("numNNServerThreadsPerModel") + ", consider deleting it and using the default (which this benchmark assumes when computing its performance stats)" << endl;
    }
#endif

    cout << "If you intend to do much longer searches, configure the seconds per game move you expect with the '-time' flag and benchmark again." << endl;
    cout << "If you intend to do short or fixed-visit searches, use lower numSearchThreads for better strength, high threads will weaken strength." << endl;

    cout << "If interested see also other notes about performance and mem usage in the top of " << cfg.getFileName() << endl;
    cout << endl;
  }

  delete nnEval;
  NeuralNet::globalCleanup();
  delete sgf;
  ScoreValue::freeTables();

  return 0;
}

static void warmStartNNEval(const CompactSgf* sgf, Logger& logger, const SearchParams& params, NNEvaluator* nnEval, Rand& seedRand) {
  Board board(sgf->xSize,sgf->ySize);
  Player nextPla = P_BLACK;
  BoardHistory hist(board,nextPla,Rules(),0);
  SearchParams thisParams = params;
  thisParams.numThreads = 1;
  thisParams.maxVisits = 5;
  thisParams.maxPlayouts = 5;
  thisParams.maxTime = 1e20;
  AsyncBot* bot = new AsyncBot(thisParams, nnEval, &logger, Global::uint64ToString(seedRand.nextUInt64()));
  bot->setPosition(nextPla,board,hist);
  bot->genMoveSynchronous(nextPla,TimeControls());
  delete bot;
}

static NNEvaluator* createNNEval(int maxNumThreads, CompactSgf* sgf, const string& modelFile, Logger& logger, ConfigParser& cfg, const SearchParams& params) {
  int maxConcurrentEvals = maxNumThreads * 2 + 16; // * 2 + 16 just to give plenty of headroom
  int defaultMaxBatchSize = std::max(8,((maxNumThreads+3)/4)*4);

  Rand seedRand;

  NNEvaluator* nnEval = Setup::initializeNNEvaluator(
    modelFile,modelFile,cfg,logger,seedRand,maxConcurrentEvals,
    sgf->xSize,sgf->ySize,defaultMaxBatchSize,
    Setup::SETUP_FOR_BENCHMARK
  );

  //Run on a sample position just to get any initialization and logs out of the way
  warmStartNNEval(sgf,logger,params,nnEval,seedRand);

  cout.flush();
  cerr.flush();
  //Sleep a bit to allow for nneval thread logs to finish
  std::this_thread::sleep_for(std::chrono::duration<double>(0.2));
  cout.flush();
  cerr.flush();
  cout << endl;

  return nnEval;
}

static void setNumThreads(SearchParams& params, NNEvaluator* nnEval, Logger& logger, int numThreads, const CompactSgf* sgf) {
  params.numThreads = numThreads;
#ifdef USE_EIGEN_BACKEND
  //Eigen is a little interesting in that by default, it sets numNNServerThreadsPerModel based on numSearchThreads
  //So, reset the number of threads in the nnEval each time we change the search numthreads
  logger.setLogToStdout(false);
  nnEval->killServerThreads();
  nnEval->setNumThreads(vector<int>(numThreads,-1));
  nnEval->spawnServerThreads();
  //Also since we killed and respawned all the threads, re-warm them
  Rand seedRand;
  warmStartNNEval(sgf,logger,params,nnEval,seedRand);
#else
  (void)nnEval;
  (void)logger;
  (void)numThreads;
  (void)sgf;
#endif
}

static vector<PlayUtils::BenchmarkResults> doFixedTuneThreads(
  const SearchParams& params,
  const CompactSgf* sgf,
  int numPositionsPerGame,
  NNEvaluator*& nnEval,
  Logger& logger,
  double secondsPerGameMove,
  vector<int> numThreadsToTest,
  bool printElo
) {
  vector<PlayUtils::BenchmarkResults> results;

  if(numThreadsToTest.size() > 1)
    cout << "Testing different numbers of threads: " << endl;

  for(int i = 0; i<numThreadsToTest.size(); i++) {
    const PlayUtils::BenchmarkResults* baseline = (i == 0) ? NULL : &results[0];
    SearchParams thisParams = params;
    setNumThreads(thisParams,nnEval,logger,numThreadsToTest[i],sgf);
    PlayUtils::BenchmarkResults result = PlayUtils::benchmarkSearchOnPositionsAndPrint(
      thisParams,
      sgf,
      numPositionsPerGame,
      nnEval,
      logger,
      baseline,
      secondsPerGameMove,
      printElo
    );
    results.push_back(result);
  }
  cout << endl;
  return results;
}

static vector<PlayUtils::BenchmarkResults> doAutoTuneThreads(
  const SearchParams& params,
  const CompactSgf* sgf,
  int numPositionsPerGame,
  NNEvaluator*& nnEval,
  Logger& logger,
  double secondsPerGameMove,
  std::function<void(int)> reallocateNNEvalWithEnoughBatchSize
) {
  vector<PlayUtils::BenchmarkResults> results;

  cout << "Automatically trying different numbers of threads to home in on the best: " << endl;
  cout << endl;

  map<int, PlayUtils::BenchmarkResults> resultCache; // key is threads

  auto getResult = [&](int numThreads) {
    if(resultCache.find(numThreads) == resultCache.end()) {
      const PlayUtils::BenchmarkResults* baseline = NULL;
      bool printElo = false;
      SearchParams thisParams = params;
      setNumThreads(thisParams,nnEval,logger,numThreads,sgf);
      PlayUtils::BenchmarkResults result = PlayUtils::benchmarkSearchOnPositionsAndPrint(
        thisParams,
        sgf,
        numPositionsPerGame,
        nnEval,
        logger,
        baseline,
        secondsPerGameMove,
        printElo
      );
      resultCache[numThreads] = result;
    }
    return resultCache[numThreads];
  };

  // There is a special ternary search on the integers that converges faster,
  // but since the function of threads -> elo is not perfectly convex (too noisy)
  // we will use the traditional ternary search.

  // Restrict to thread counts that are {1,2,3,4,5} * power of 2
  vector<int> possibleNumbersOfThreads;
  int twopow = 1;
  for(int i = 0; i < 20; i++) {
    // 5 * (2 ** 17) is way more than enough; 17 because we only add odd multiples to the vector, evens are just other powers of two.
    possibleNumbersOfThreads.push_back(twopow);
    possibleNumbersOfThreads.push_back(twopow * 3);
    possibleNumbersOfThreads.push_back(twopow * 5);
    twopow *= 2;
  }

  sort(possibleNumbersOfThreads.begin(), possibleNumbersOfThreads.end());

  int ternarySearchMin = 1;
  int ternarySearchMax = ternarySearchInitialMax;
  while(true) {
    reallocateNNEvalWithEnoughBatchSize(ternarySearchMax);
    cout << endl;

    int start = 0;
    int end = possibleNumbersOfThreads.size()-1;
    for(int i = 0; i < possibleNumbersOfThreads.size(); i++) {
      if(possibleNumbersOfThreads[i] < ternarySearchMin) {
        start = i + 1;
      }
      if(possibleNumbersOfThreads[i] > ternarySearchMax) {
        end = i - 1;
        break;
      }
    }
    if(start > end)
      start = end;

    cout << "Possible numbers of threads to test: ";
    for(int i = start; i <= end; i++) {
      cout << possibleNumbersOfThreads[i] << ", ";
    }
    cout << endl;
    cout << endl;

    while(start <= end) {
      int firstMid = start + (end - start) / 3;
      int secondMid = end - (end - start) / 3;

      double effect1 = getResult(possibleNumbersOfThreads[firstMid]).computeEloEffect(secondsPerGameMove);
      double effect2 = getResult(possibleNumbersOfThreads[secondMid]).computeEloEffect(secondsPerGameMove);
      if(effect1 < effect2)
        start = firstMid + 1;
      else
        end = secondMid - 1;
    }

    double bestElo = 0;
    int bestThreads = 0;

    results.clear();
    for(auto it : resultCache) {
      PlayUtils::BenchmarkResults result = it.second;
      double elo = result.computeEloEffect(secondsPerGameMove);
      results.push_back(result);

      if(elo > bestElo) {
        bestThreads = result.numThreads;
        bestElo = elo;
      }
    }

    // If our optimal thread count is in the top 2/3 of the maximum search limit, triple the search limit and repeat.
    if(3 * bestThreads > 2 * ternarySearchMax && ternarySearchMax < 5000) {
      ternarySearchMin = ternarySearchMax / 2;
      ternarySearchMax *= 3;
      cout << endl << endl << "Optimal number of threads is fairly high, tripling the search limit and trying again." << endl << endl;
      continue;
    }
    else {
      cout << endl << endl << "Ordered summary of results: " << endl << endl;
      for(int i = 0; i<results.size(); i++) {
        cout << results[i].toStringWithElo(i == 0 ? NULL : &results[0], secondsPerGameMove) << endl;
      }
      cout << endl;
      break;
    }
  }

  return results;
}


int MainCmds::genconfig(int argc, const char* const* argv, const char* firstCommand) {
  Board::initHash();
  ScoreValue::initTables();

  string outputFile;
  string modelFile;
  bool modelFileIsDefault;
  try {
    KataGoCommandLine cmd("Automatically generate and tune a new GTP config.");
    cmd.addModelFileArg();

    TCLAP::ValueArg<string> outputFileArg("","output","Path to write new config (default gtp.cfg)",false,string("gtp.cfg"),"FILE");
    cmd.add(outputFileArg);
    cmd.parse(argc,argv);

    outputFile = outputFileArg.getValue();
    modelFile = cmd.getModelFile();
    modelFileIsDefault = cmd.modelFileIsDefault();
  }
  catch (TCLAP::ArgException &e) {
    cerr << "Error: " << e.error() << " for argument " << e.argId() << endl;
    return 1;
  }

  auto promptAndParseInput = [](const string& prompt, std::function<void(const string&)> parse) {
    while(true) {
      try {
        cout << prompt << std::flush;
        string line;
        if(std::getline(std::cin, line)) {
          parse(Global::trim(line));
          break;
        }
        else {
          break;
        }
      }
      catch(const StringError& err) {
        string what = err.what();
        what = Global::trim(what);
        if(what.length() > 0)
          cout << err.what() << endl;
      }
    }
    if(!std::cin) {
      throw StringError("Stdin was closed - failing and not generating a config");
    }
  };

  auto parseYN = [](const string& line, bool& b) {
    string s = Global::toLower(Global::trim(line));
    if(s == "yes" || s == "y")
      b = true;
    else if(s == "no" || s == "n")
      b = false;
    else
      throw StringError("Please answer y or n");
  };

  if(bfs::exists(bfs::path(outputFile))) {
    bool b = false;
    promptAndParseInput("File " + outputFile + " already exists, okay to overwrite it with an entirely new config (y/n)?\n", [&](const string& line) { parseYN(line,b); });
    if(!b) {
      cout << "Please provide an alternate file path to output the generated config to via '-output NEWFILEPATH'" << endl;
      return 0;
    }
  }

  int boardSize = 19;
  string sgfData = TestCommon::getBenchmarkSGFData(boardSize);
  CompactSgf* sgf = CompactSgf::parse(sgfData);

  Rules configRules;
  int64_t configMaxVisits = ((int64_t)1) << 50;
  int64_t configMaxPlayouts = ((int64_t)1) << 50;
  double configMaxTime = 1e20;
  double configMaxPonderTime = -1.0;
  vector<int> configDeviceIdxs;
  int configNNCacheSizePowerOfTwo = 20;
  int configNNMutexPoolSizePowerOfTwo = 16;
  int configNumSearchThreads = 6;

  cout << endl;
  cout << "=========================================================================" << endl;
  cout << "RULES" << endl;

  {
    cout << endl;
    string prompt =
      "What rules should KataGo use by default for play and analysis?\n"
      "(chinese, japanese, korean, tromp-taylor, aga, chinese-ogs, new-zealand, bga, stone-scoring, aga-button):\n";
    promptAndParseInput(prompt, [&](const string& line) { configRules = Rules::parseRules(line); });
  }

  cout << endl;
  cout << "=========================================================================" << endl;
  cout << "SEARCH LIMITS" << endl;

  bool useSearchLimit = false;
  {
    cout << endl;
    string prompt =
      "When playing games, KataGo will always obey the time controls given by the GUI/tournament/match/online server.\n"
      "But you can specify an additional limit to make KataGo move much faster. This does NOT affect analysis/review,\n"
      "only affects playing games. Add a limit? (y/n) (default n):\n";
    promptAndParseInput(prompt, [&](const string& line) {
        if(line == "") useSearchLimit = false;
        else parseYN(line,useSearchLimit);
      });
  }

  if(!useSearchLimit) {
    cout << endl;
    string prompt =
      "NOTE: No limits configured for KataGo. KataGo will obey time controls provided by the GUI or server or match script\n"
      "but if they don't specify any, when playing games KataGo may think forever without moving. (press enter to continue)\n";
    promptAndParseInput(prompt, [&](const string& line) {
        (void)line;
      });
  }

  else {
    string whatLimit = "";
    {
      cout << endl;
      string prompt =
        "What to limit per move? Visits, playouts, or seconds?:\n";
      promptAndParseInput(prompt, [&](const string& line) {
          string s = Global::toLower(line);
          if(s == "visits" || s == "playouts" || s == "seconds") whatLimit = s;
          else if(s == "visit") whatLimit = "visits";
          else if(s == "playout") whatLimit = "playouts";
          else if(s == "second") whatLimit = "seconds";
          else throw StringError("Please specify one of \"visits\" or \"playouts\" or '\"seconds\"");
        });
    }

    if(whatLimit == "visits") {
      cout << endl;
      string prompt =
        "Specify max number of visits/move when playing games (doesn't affect analysis), leave blank for default (500):\n";
      promptAndParseInput(prompt, [&](const string& line) {
          if(line == "") configMaxVisits = 500;
          else {
            configMaxVisits = Global::stringToInt64(line);
            if(configMaxVisits < 1 || configMaxVisits > 1000000000)
              throw StringError("Must be between 1 and 1000000000");
          }
        });
    }
    else if(whatLimit == "playouts") {
      cout << endl;
      string prompt =
        "Specify max number of playouts/move when playing games (doesn't affect analysis), leave blank for default (300):\n";
      promptAndParseInput(prompt, [&](const string& line) {
          if(line == "") configMaxPlayouts = 300;
          else {
            configMaxPlayouts = Global::stringToInt64(line);
            if(configMaxPlayouts < 1 || configMaxPlayouts > 1000000000)
              throw StringError("Must be between 1 and 1000000000");
          }
        });
    }
    else if(whatLimit == "seconds") {
      cout << endl;
      string prompt =
        "Specify max time/move in seconds when playing games (doesn't affect analysis). Leave blank for default (10):\n";
      promptAndParseInput(prompt, [&](const string& line) {
          if(line == "") configMaxTime = 10.0;
          else {
            configMaxTime = Global::stringToDouble(line);
            if(isnan(configMaxTime) || configMaxTime <= 0 || configMaxTime >= 1.0e20)
              throw StringError("Must positive and less than 1e20");
          }
        });
    }
  }

  bool usePonder = false;
  {
    cout << endl;
    string prompt =
      "When playing games, KataGo can optionally ponder during the opponent's turn. This gives faster/stronger play\n"
      "in real games but should NOT be enabled if you are running tests with fixed limits (pondering may exceed those\n"
      "limits), or to avoid stealing the opponent's compute time when testing two bots on the same machine.\n"
      "Enable pondering? (y/n, default n):";
    promptAndParseInput(prompt, [&](const string& line) {
        if(line == "") usePonder = false;
        else parseYN(line,usePonder);
      });
  }

  if(usePonder) {
    cout << endl;
    string prompt =
      "Specify max num seconds KataGo should ponder during the opponent's turn. Leave blank for no limit:\n";
    promptAndParseInput(prompt, [&](const string& line) {
        if(line == "") configMaxPonderTime = 1.0e20;
        else {
          configMaxPonderTime = Global::stringToDouble(line);
          if(isnan(configMaxPonderTime) || configMaxPonderTime <= 0 || configMaxPonderTime >= 1.0e20)
            throw StringError("Must positive and less than 1e20");
        }
      });
  }

  cout << endl;
  cout << "=========================================================================" << endl;
  cout << "GPUS AND RAM" << endl;

#ifndef USE_EIGEN_BACKEND
  {
    cout << endl;
    cout << "Finding available GPU-like devices..." << endl;
    NeuralNet::printDevices();
    cout << endl;

    string prompt =
      "Specify devices/GPUs to use (for example \"0,1,2\" to use devices 0, 1, and 2). Leave blank for a default SINGLE-GPU config:\n";
    promptAndParseInput(prompt, [&](const string& line) {
        vector<string> pieces = Global::split(line,',');
        configDeviceIdxs.clear();
        for(size_t i = 0; i<pieces.size(); i++) {
          string piece = Global::trim(pieces[i]);
          int idx = Global::stringToInt(piece);
          if(idx < 0 || idx > 10000)
            throw StringError("Invalid device idx: " + Global::intToString(idx));
          configDeviceIdxs.push_back(idx);
        }
      });
  }
#endif

  {
    cout << endl;
    string prompt =
      "By default, KataGo will cache up to about 3GB of positions in memory (RAM), in addition to\n"
      "whatever the current search is using. Specify a max in GB or leave blank for default:\n";
    promptAndParseInput(prompt, [&](const string& line) {
        string s = Global::toLower(line);
        if(Global::isSuffix(s,"gb"))
          s = s.substr(0,s.length()-2);
        s = Global::trim(s);
        double approxGBLimit;
        if(s == "") approxGBLimit = 3.0;
        else {
          approxGBLimit = Global::stringToDouble(s);
          if(isnan(approxGBLimit) || approxGBLimit <= 0 || approxGBLimit >= 1000000.0)
            throw StringError("Must positive and less than 1000000");
        }
        approxGBLimit *= 1.00001;
        configNNCacheSizePowerOfTwo = 10; //Never set below this size
        while(configNNCacheSizePowerOfTwo < 48) {
          double memUsage = pow(2.0, configNNCacheSizePowerOfTwo) * 3000.0;
          if(memUsage * 2.0 > approxGBLimit * 1073741824.0)
            break;
          configNNCacheSizePowerOfTwo += 1;
        }
        configNNMutexPoolSizePowerOfTwo = configNNCacheSizePowerOfTwo - 4;
        if(configNNMutexPoolSizePowerOfTwo < 10)
          configNNMutexPoolSizePowerOfTwo = 10;
        if(configNNMutexPoolSizePowerOfTwo > 24)
          configNNMutexPoolSizePowerOfTwo = 24;
      });
  }

  cout << endl;
  cout << "=========================================================================" << endl;
  cout << "PERFORMANCE TUNING" << endl;

  bool skipThreadTuning = false;
  if(bfs::exists(bfs::path(outputFile))) {
    int oldConfigNumSearchThreads = -1;
    try {
      ConfigParser oldCfg(outputFile);
      oldConfigNumSearchThreads = oldCfg.getInt("numSearchThreads",1,4096);
    }
    catch(const StringError&) {
      cout << "NOTE: Overwritten config does not specify numSearchThreads or otherwise could not be parsed." << endl;
      cout << "Beginning performance tuning to set this." << endl;
    }
    if(oldConfigNumSearchThreads > 0) {
      promptAndParseInput(
        "Actually " + outputFile + " already exists, can skip performance tuning if desired and just use\nthe number of threads (" +
        Global::intToString(oldConfigNumSearchThreads) + ") "
        "already in that config (all other settings will still be overwritten).\nSkip performance tuning (y/n)?\n",
        [&](const string& line) { parseYN(line,skipThreadTuning); }
      );
      if(skipThreadTuning) {
        configNumSearchThreads = oldConfigNumSearchThreads;
      }
    }
  }

  string configFileContents;
  auto updateConfigContents = [&]() {
    configFileContents = GTPConfig::makeConfig(
      configRules,
      configMaxVisits,
      configMaxPlayouts,
      configMaxTime,
      configMaxPonderTime,
      configDeviceIdxs,
      configNNCacheSizePowerOfTwo,
      configNNMutexPoolSizePowerOfTwo,
      configNumSearchThreads
    );
  };
  updateConfigContents();

  if(!skipThreadTuning) {
    int64_t maxVisitsFromUser = -1;
    double secondsPerGameMove = defaultSecondsPerGameMove;
    {
      cout << endl;
      string prompt =
        "Specify number of visits to use test/tune performance with, leave blank for default based on GPU speed.\n"
        "Use large number for more accurate results, small if your GPU is old and this is taking forever:\n";
      promptAndParseInput(prompt, [&](const string& line) {
          if(line == "") maxVisitsFromUser = -1;
          else {
            maxVisitsFromUser = Global::stringToInt64(line);
            if(maxVisitsFromUser < 1 || maxVisitsFromUser > 1000000000)
              throw StringError("Must be between 1 and 1000000000");
          }
        });
    }

    {
      cout << endl;
      string prompt =
        "Specify number of seconds/move to optimize performance for (default " + Global::doubleToString(defaultSecondsPerGameMove) + "), leave blank for default:\n";
      promptAndParseInput(prompt, [&](const string& line) {
          if(line == "") secondsPerGameMove = defaultSecondsPerGameMove;
          else {
            secondsPerGameMove = Global::stringToDouble(line);
            if(isnan(secondsPerGameMove) || secondsPerGameMove <= 0 || secondsPerGameMove > 1000000)
              throw StringError("Must be between 0 and 1000000");
          }
        });
    }

    istringstream inConfig(configFileContents);
    ConfigParser cfg(inConfig);

    Logger logger;
    logger.setLogToStdout(true);
    logger.write("Loading model and initializing benchmark...");

    SearchParams params = Setup::loadSingleParams(cfg);
    params.maxVisits = defaultMaxVisits;
    params.maxPlayouts = defaultMaxVisits;
    params.maxTime = 1e20;
    params.searchFactorAfterOnePass = 1.0;
    params.searchFactorAfterTwoPass = 1.0;

    Setup::initializeSession(cfg);

    NNEvaluator* nnEval = NULL;
    auto reallocateNNEvalWithEnoughBatchSize = [&](int maxNumThreads) {
      if(nnEval != NULL)
        delete nnEval;
      nnEval = createNNEval(maxNumThreads, sgf, modelFile, logger, cfg, params);
    };
    cout << endl;

    int64_t maxVisits;
    if(maxVisitsFromUser > 0) {
      maxVisits = maxVisitsFromUser;
    }
    else {
      cout << "Running quick initial benchmark at 16 threads!" << endl;
      vector<int> numThreads = {16};
      reallocateNNEvalWithEnoughBatchSize(ternarySearchInitialMax);
      vector<PlayUtils::BenchmarkResults> results = doFixedTuneThreads(params,sgf,3,nnEval,logger,secondsPerGameMove,numThreads,false);
      double visitsPerSecond = results[0].totalVisits / (results[0].totalSeconds + 0.00001);
      //Make tests use about 2 seconds each
      maxVisits = (int64_t)round(2.0 * visitsPerSecond/100.0) * 100;
      if(maxVisits < 200) maxVisits = 200;
      if(maxVisits > 10000) maxVisits = 10000;
    }

    params.maxVisits = maxVisits;
    params.maxPlayouts = maxVisits;

    const int numPositionsPerGame = 10;

    cout << "=========================================================================" << endl;
    cout << "TUNING NOW" << endl;
    cout << "Tuning using " << maxVisits << " visits." << endl;

    vector<PlayUtils::BenchmarkResults> results;
    results = doAutoTuneThreads(params,sgf,numPositionsPerGame,nnEval,logger,secondsPerGameMove,reallocateNNEvalWithEnoughBatchSize);

    PlayUtils::BenchmarkResults::printEloComparison(results,secondsPerGameMove);
    int bestIdx = 0;
    for(int i = 1; i<results.size(); i++) {
      if(results[i].computeEloEffect(secondsPerGameMove) > results[bestIdx].computeEloEffect(secondsPerGameMove))
        bestIdx = i;
    }
    cout << "Using " << results[bestIdx].numThreads << " numSearchThreads!" << endl;

    configNumSearchThreads = results[bestIdx].numThreads;

    delete nnEval;
  }

  updateConfigContents();

  cout << endl;
  cout << "=========================================================================" << endl;
  cout << "DONE" << endl;
  cout << endl;
  cout << "Writing new config file to " << outputFile << endl;
  ofstream out(outputFile, ofstream::out | ofstream::trunc);
  out << configFileContents;
  out.close();

  cout << "You should be now able to run KataGo with this config via something like:" << endl;
  if(modelFileIsDefault)
    cout << firstCommand << " gtp -config '" << outputFile << "'" << endl;
  else
    cout << firstCommand << " gtp -model '" << modelFile << "' -config '" << outputFile << "'" << endl;
  cout << endl;

  cout << "Feel free to look at and edit the above config file further by hand in a txt editor." << endl;
  cout << "For more detailed notes about performance and what options in the config do, see:" << endl;
  cout << "https://github.com/lightvector/KataGo/blob/master/cpp/configs/gtp_example.cfg" << endl;
  cout << endl;

  NeuralNet::globalCleanup();
  delete sgf;
  ScoreValue::freeTables();

  return 0;
}
