#include "../core/global.h"
#include "../core/datetime.h"
#include "../core/fileutils.h"
#include "../core/makedir.h"
#include "../core/config_parser.h"
#include "../core/parallel.h"
#include "../dataio/sgf.h"
#include "../dataio/trainingwrite.h"
#include "../dataio/loadmodel.h"
#include "../dataio/files.h"
#include "../neuralnet/modelversion.h"
#include "../program/setup.h"
#include "../command/commandline.h"
#include "../main.h"

#include <chrono>
#include <csignal>

using namespace std;

int MainCmds::writetrainingdata(const vector<string>& args) {
  Board::initHash();
  ScoreValue::initTables();
  Rand seedRand;

  ConfigParser cfg;
  string nnModelFile;
  vector<string> sgfDirs;
  string outputDir;

  try {
    KataGoCommandLine cmd("Generate training data from sgfs.");
    cmd.addConfigFileArg("","");
    cmd.addModelFileArg();
    cmd.addOverrideConfigArg();

    TCLAP::MultiArg<string> sgfDirArg("","sgfdir","Directory of sgf files",false,"DIR");
    TCLAP::ValueArg<string> outputDirArg("","output-dir","Dir to output files",true,string(),"DIR");

    cmd.add(sgfDirArg);
    cmd.add(outputDirArg);

    cmd.parseArgs(args);

    nnModelFile = cmd.getModelFile();
    sgfDirs = sgfDirArg.getValue();
    outputDir = outputDirArg.getValue();

    cmd.getConfig(cfg);
  }
  catch (TCLAP::ArgException &e) {
    cerr << "Error: " << e.error() << " for argument " << e.argId() << endl;
    return 1;
  }

  const bool logToStdout = true;
  const bool logToStderr = false;
  const bool logTimeStamp = true;
  Logger logger(nullptr, logToStdout, logToStderr, logTimeStamp);
  for(const string& arg: args)
    logger.write(string("Command: ") + arg);

  MakeDir::make(outputDir);
  const int numThreads = 16;

  const int dataBoardLen = cfg.getInt("dataBoardLen",3,37);
  const int maxRowsPerTrainFile = cfg.getInt("maxRowsPerTrainFile",1,100000000);

  static_assert(NNModelVersion::latestInputsVersionImplemented == 7, "");
  const int inputsVersion = 7;
  const int numBinaryChannels = NNInputs::NUM_FEATURES_SPATIAL_V7;
  const int numGlobalChannels = NNInputs::NUM_FEATURES_GLOBAL_V7;

  NNEvaluator* nnEval;
  {
    Setup::initializeSession(cfg);
    const int expectedConcurrentEvals = numThreads;
    const int defaultMaxBatchSize = std::max(8,((numThreads+3)/4)*4);
    const bool defaultRequireExactNNLen = false;
    const bool disableFP16 = false;
    const string expectedSha256 = "";
    nnEval = Setup::initializeNNEvaluator(
      nnModelFile,nnModelFile,expectedSha256,cfg,logger,seedRand,expectedConcurrentEvals,
      NNPos::MAX_BOARD_LEN,NNPos::MAX_BOARD_LEN,defaultMaxBatchSize,defaultRequireExactNNLen,disableFP16,
      Setup::SETUP_FOR_ANALYSIS
    );
  }
  logger.write("Loaded neural net");

  vector<string> sgfFiles;
  FileHelpers::collectSgfsFromDirsOrFiles(sgfDirs,sgfFiles);

  Setup::initializeSession(cfg);

  cfg.warnUnusedKeys(cerr,&logger);

  //Done loading!
  //------------------------------------------------------------------------------------
  std::atomic<int64_t> numSgfsDone(0);
  std::atomic<int64_t> numSgfErrors(0);

  auto reportSgfDone = [&](bool wasSuccess) {
    if(!wasSuccess)
      numSgfErrors.fetch_add(1);
    int64_t numErrors = numSgfErrors.load();
    int64_t numDone = numSgfsDone.fetch_add(1) + 1;

    if(numDone == sgfFiles.size() || numDone % 100 == 0) {
      logger.write(
        "Done " + Global::int64ToString(numDone) + " / " + Global::int64ToString(sgfFiles.size()) + " sgfs, " +
        string("errors ") + Global::int64ToString(numErrors)
      );
    }
  };

  std::vector<TrainingWriteBuffers*> threadDataBuffers;
  for(int i = 0; i<numThreads; i++) {
    threadDataBuffers.push_back(
      new TrainingWriteBuffers(
        inputsVersion,
        maxRowsPerTrainFile,
        numBinaryChannels,
        numGlobalChannels,
        dataBoardLen,
        dataBoardLen
      )
    );
  }

  auto processSgf = [&](int threadIdx, size_t index) {
    const string& fileName = sgfFiles[index];
    Sgf* sgfRaw = NULL;
    try {
      sgfRaw = Sgf::loadFile(fileName);
    }
    catch(const StringError& e) {
      logger.write("Invalid SGF " + fileName + ": " + e.what());
      reportSgfDone(false);
      return;
    }
    CompactSgf* sgf = NULL;
    try {
      sgf = new CompactSgf(sgfRaw);
    }
    catch(const StringError& e) {
      logger.write("Invalid SGF " + fileName + ": " + e.what());
      reportSgfDone(false);
      delete sgfRaw;
      return;
    }

    TrainingWriteBuffers* dataBuffer = threadDataBuffers[threadIdx];
    (void)dataBuffer;

  // void clear();

  // void addRow(
  //   const Board& board, const BoardHistory& hist, Player nextPlayer,
  //   int turnAfterStart,
  //   float targetWeight,
  //   int64_t unreducedNumVisits,
  //   const std::vector<PolicyTargetMove>* policyTarget0, //can be null
  //   const std::vector<PolicyTargetMove>* policyTarget1, //can be null
  //   double policySurprise,
  //   double policyEntropy,
  //   double searchEntropy,
  //   const std::vector<ValueTargets>& whiteValueTargets,
  //   int whiteValueTargetsIdx, //index in whiteValueTargets corresponding to this turn.
  //   const NNRawStats& nnRawStats,
  //   const Board* finalBoard,
  //   Color* finalFullArea,
  //   Color* finalOwnership,
  //   float* finalWhiteScoring,
  //   const std::vector<Board>* posHistForFutureBoards, //can be null
  //   bool isSidePosition,
  //   int numNeuralNetsBehindLatest,
  //   const FinishedGameData& data,
  //   Rand& rand
  // );

  // void writeToZipFile(const std::string& fileName);


    reportSgfDone(true);
    delete sgf;
    delete sgfRaw;
  };

  Parallel::iterRange(numThreads, sgfFiles.size(), std::function<void(int,size_t)>(processSgf));

  logger.write(nnEval->getModelFileName());
  logger.write("NN rows: " + Global::int64ToString(nnEval->numRowsProcessed()));
  logger.write("NN batches: " + Global::int64ToString(nnEval->numBatchesProcessed()));
  logger.write("NN avg batch size: " + Global::doubleToString(nnEval->averageProcessedBatchSize()));

  logger.write("All done");

  for(int i = 0; i<numThreads; i++)
    delete threadDataBuffers[i];

  delete nnEval;
  NeuralNet::globalCleanup();
  ScoreValue::freeTables();
  return 0;
}
