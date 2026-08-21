#include "../core/global.h"
#include "../core/config_parser.h"
#include "../core/logger.h"
#include "../core/rand.h"
#include "../game/board.h"
#include "../neuralnet/nneval.h"
#include "../program/setup.h"
#include "../command/commandline.h"
#include "../main.h"

#include <iomanip>
#include <sstream>

using namespace std;

static string jsonEscape(const string& s) {
  ostringstream out;
  for(char c : s) {
    if(c == '"' || c == '\\')
      out << '\\' << c;
    else if(c == '\n')
      out << "\\n";
    else if(c == '\t')
      out << "\\t";
    else if(c == '\r')
      out << "\\r";
    else
      out << c;
  }
  return out.str();
}

int MainCmds::benchmarknn(const vector<string>& args) {
  Board::initHash();
  ScoreValue::initTables();
  Rand seedRand;

  ConfigParser cfg;
  string modelFile;
  int numIterations;
  int numWarmups;
  int batchSizeOverride;
  string boardSizesStr;
  bool requireExactNNLen;
  bool jsonOut;
  vector<int> boardSizes;

  try {
    KataGoCommandLine cmd(
      "Benchmark raw neural net forward throughput, without search. Uses numNNServerThreadsPerModel "
      "and per-thread GPU assignment from the config. Timing includes host-device transfers and "
      "output postprocessing, excludes input feature generation and search."
    );
    cmd.addConfigFileArg(KataGoCommandLine::defaultGtpConfigFileName(),"gtp_example.cfg");
    cmd.addModelFileArg();
    TCLAP::ValueArg<int> iterationsArg(
      "","iterations","Number of timed forward passes per NN server thread (default 200)",
      false,200,"N"
    );
    TCLAP::ValueArg<int> warmupArg(
      "","warmup","Untimed forward passes per NN server thread before timing (default 20)",
      false,20,"N"
    );
    TCLAP::ValueArg<int> batchSizeArg(
      "","batch-size","Batch size per NN server thread (default: nnMaxBatchSize from config, or 16)",
      false,-1,"N"
    );
    TCLAP::ValueArg<string> boardSizesArg(
      "","boardsize",
      "Board size, or comma-separated sizes cycled across the rows of each batch, e.g. 19 or 9,13,19 "
      "(default 19). The NN buffer size is the largest listed size.",
      false,"19","SIZES"
    );
    TCLAP::SwitchArg requireExactNNLenArg(
      "","require-exact-nnlen",
      "Run with requireExactNNLen (backend may skip mask handling). Needs a single board size."
    );
    TCLAP::SwitchArg jsonArg("","json","Print results as JSON",false);
    cmd.add(iterationsArg);
    cmd.add(warmupArg);
    cmd.add(batchSizeArg);
    cmd.add(boardSizesArg);
    cmd.add(requireExactNNLenArg);
    cmd.add(jsonArg);
    cmd.setShortUsageArgLimit();
    cmd.addOverrideConfigArg();

    cmd.parseArgs(args);

    modelFile = cmd.getModelFile();
    numIterations = iterationsArg.getValue();
    numWarmups = warmupArg.getValue();
    batchSizeOverride = batchSizeArg.getValue();
    boardSizesStr = boardSizesArg.getValue();
    requireExactNNLen = requireExactNNLenArg.getValue();
    jsonOut = jsonArg.getValue();
    cmd.getConfig(cfg);

    if(numIterations <= 0)
      throw StringError("benchmarknn: iterations must be positive");
    if(numWarmups < 0)
      throw StringError("benchmarknn: warmup must be nonnegative");
    for(const string& piece : Global::split(boardSizesStr,',')) {
      int bSize = Global::stringToInt(Global::trim(piece));
      if(bSize < 2 || bSize > Board::MAX_LEN)
        throw StringError("benchmarknn: invalid board size " + piece);
      boardSizes.push_back(bSize);
    }
    if(boardSizes.size() <= 0)
      throw StringError("benchmarknn: no board sizes specified");
    if(requireExactNNLen && boardSizes.size() > 1)
      throw StringError("benchmarknn: require-exact-nnlen needs a single board size");
  }
  catch(TCLAP::ArgException& e) {
    cerr << "Error: " << e.error() << " for argument " << e.argId() << endl;
    return 1;
  }

  // With -json, stdout must stay machine-readable, so route logs (including backend fallback
  // warnings that would otherwise be lost) to stderr instead.
  const bool logToStdout = !jsonOut;
  const bool logToStderr = jsonOut;
  Logger logger(NULL, logToStdout, logToStderr, false);
  logger.write("Version " + Version::getGitRevisionWithBackend());

  int nnLen = 0;
  for(int bSize : boardSizes)
    nnLen = std::max(nnLen, bSize);

  const string expectedSha256 = "";
  const int maxBatchSize =
    batchSizeOverride > 0 ? batchSizeOverride :
    cfg.contains("nnMaxBatchSize") ? cfg.getInt("nnMaxBatchSize",1,65536) :
    16;
  const int expectedConcurrentEvals = maxBatchSize;
  const bool disableFP16 = false;

  NNEvaluator* nnEval = NULL;
  try {
    nnEval = Setup::initializeNNEvaluator(
      modelFile,modelFile,expectedSha256,cfg,logger,seedRand,expectedConcurrentEvals,
      nnLen,nnLen,Setup::MaxBatchSizeRequest::explicitSize(maxBatchSize),requireExactNNLen,disableFP16,
      Setup::SETUP_FOR_BENCHMARKNN
    );

    NNEvalBenchmarkResult result = nnEval->benchmarkPureForward(numWarmups,numIterations,boardSizes);

    if(jsonOut) {
      ostringstream out;
      out << "{";
      out << "\"modelFile\":\"" << jsonEscape(nnEval->getModelFileName()) << "\",";
      out << "\"modelName\":\"" << jsonEscape(nnEval->getInternalModelName()) << "\",";
      out << "\"revision\":\"" << jsonEscape(Version::getGitRevisionWithBackend()) << "\",";
      out << "\"boardSizes\":[";
      for(size_t i = 0; i < boardSizes.size(); i++)
        out << (i > 0 ? "," : "") << boardSizes[i];
      out << "],";
      out << "\"requireExactNNLen\":" << (nnEval->getRequireExactNNLen() ? "true" : "false") << ",";
      out << "\"usingFP16\":" << (nnEval->isAnyThreadUsingFP16() ? "true" : "false") << ",";
      out << "\"batchSize\":" << result.batchSize << ",";
      out << "\"numThreads\":" << result.numThreads << ",";
      out << "\"numIterations\":" << result.numIterations << ",";
      out << "\"gpuIdxs\":[";
      bool first = true;
      for(int g : nnEval->getGpuIdxs()) {
        out << (first ? "" : ",") << g;
        first = false;
      }
      out << "],";
      out << setprecision(10);
      out << "\"perThreadMedianMs\":[";
      for(int i = 0; i < result.numThreads; i++)
        out << (i > 0 ? "," : "") << result.perThreadMedianSeconds[i] * 1000.0;
      out << "],";
      out << "\"perThreadNNEvalsPerSec\":[";
      for(int i = 0; i < result.numThreads; i++)
        out << (i > 0 ? "," : "") << result.perThreadNNEvalsPerSec[i];
      out << "],";
      out << "\"sumMedianNNEvalsPerSec\":" << result.sumMedianNNEvalsPerSec << ",";
      out << "\"actualWallSeconds\":" << result.actualWallSeconds << ",";
      out << "\"actualWallNNEvalsPerSec\":" << result.actualWallNNEvalsPerSec;
      out << "}";
      cout << out.str() << endl;
    }
    else {
      cout << "=== benchmarknn ===" << endl;
      cout << "model: " << nnEval->getModelFileName() << endl;
      cout << "internal model: " << nnEval->getInternalModelName() << endl;
      cout << "revision/backend: " << Version::getGitRevisionWithBackend() << endl;
      cout << "board sizes:";
      for(int bSize : boardSizes)
        cout << " " << bSize;
      cout << " (NN buffer " << nnLen << "x" << nnLen
           << (nnEval->getRequireExactNNLen() ? ", requireExactNNLen" : "") << ")" << endl;
      cout << "FP16: " << (nnEval->isAnyThreadUsingFP16() ? "true" : "false") << endl;
      cout << "batch size per thread: " << result.batchSize << endl;
      cout << "NN server threads: " << result.numThreads << endl;
      cout << "GPU indices:";
      for(int g : nnEval->getGpuIdxs())
        cout << " " << g;
      cout << endl;
      cout << "timed iterations per thread: " << result.numIterations << endl;
      for(int i = 0; i < result.numThreads; i++) {
        cout << "thread " << i << ": "
             << setprecision(6) << result.perThreadMedianSeconds[i] * 1000.0
             << " ms/batch median, " << setprecision(8) << result.perThreadNNEvalsPerSec[i]
             << " nnEval/s" << endl;
      }
      cout << "sum of per-thread median rates: "
           << setprecision(8) << result.sumMedianNNEvalsPerSec << " nnEval/s" << endl;
      cout << "wall time of timed region: "
           << setprecision(6) << result.actualWallSeconds << " s" << endl;
      cout << "overall throughput (wall): "
           << setprecision(8) << result.actualWallNNEvalsPerSec << " nnEval/s" << endl;
    }
  }
  catch(...) {
    delete nnEval;
    NeuralNet::globalCleanup();
    ScoreValue::freeTables();
    throw;
  }

  delete nnEval;
  NeuralNet::globalCleanup();
  ScoreValue::freeTables();
  return 0;
}
