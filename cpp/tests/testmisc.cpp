#include "../tests/tests.h"

#include "../core/fileutils.h"
#include "../command/commandline.h"
#include "../dataio/files.h"
#include "../dataio/loadmodel.h"
#include "../program/playutils.h"

#include <chrono>
#include <thread>

//------------------------
#include "../core/using.h"
//------------------------
using namespace TestCommon;

void Tests::runBenchmarkResultsTests() {
  PlayUtils::BenchmarkResults result;
  testAssert(result.getNNEvalsPerSecond() == 0.0);

  result.numNNEvals = 4321;
  result.totalSeconds = 2.0;
  testAssert(result.getNNEvalsPerSecond() == 2160.5);

  PlayUtils::BenchmarkResults moreEvalsButSlower;
  moreEvalsButSlower.numNNEvals = 5000;
  moreEvalsButSlower.totalSeconds = 4.0;
  PlayUtils::BenchmarkResults fewerEvalsButFaster;
  fewerEvalsButFaster.numNNEvals = 3000;
  fewerEvalsButFaster.totalSeconds = 2.0;
  testAssert(PlayUtils::BenchmarkResults::isBetterNNEvalsPerSecond(
    fewerEvalsButFaster,20,moreEvalsButSlower,18
  ));
  testAssert(!PlayUtils::BenchmarkResults::isBetterNNEvalsPerSecond(
    moreEvalsButSlower,18,fewerEvalsButFaster,20
  ));

  PlayUtils::BenchmarkResults equalRateMoreThreads;
  equalRateMoreThreads.numNNEvals = 3000;
  equalRateMoreThreads.totalSeconds = 2.0;
  equalRateMoreThreads.numThreads = 64;
  PlayUtils::BenchmarkResults equalRateFewerThreads = equalRateMoreThreads;
  equalRateFewerThreads.numThreads = 60;
  testAssert(PlayUtils::BenchmarkResults::isBetterNNEvalsPerSecond(
    equalRateMoreThreads,18,equalRateFewerThreads,20
  ));
  testAssert(PlayUtils::BenchmarkResults::isBetterNNEvalsPerSecond(
    equalRateFewerThreads,18,equalRateMoreThreads,18
  ));
  testAssert(!PlayUtils::BenchmarkResults::isBetterNNEvalsPerSecond(
    equalRateMoreThreads,18,equalRateFewerThreads,18
  ));

  const vector<int> parsed = KataGoCommandLine::parseCommaSeparatedUniqueInts(
    "18, 20,22",1,65536,"Test value"
  );
  testAssert(parsed == vector<int>({18,20,22}));
  const vector<string> invalidLists = {"", "18,,20", "0", "65537", "abc", "18,18"};
  for(const string& invalidList: invalidLists) {
    bool threw = false;
    try {
      (void)KataGoCommandLine::parseCommaSeparatedUniqueInts(
        invalidList,1,65536,"Test value"
      );
    }
    catch(const StringError&) {
      threw = true;
    }
    testAssert(threw);
  }

  testAssert(PlayUtils::getBenchmarkTuneMode(
    false,false,false,false,false,false,false
  ) == PlayUtils::BENCHMARK_TUNE_AUTO_THREADS);
  testAssert(PlayUtils::getBenchmarkTuneMode(
    false,true,false,false,false,false,false
  ) == PlayUtils::BENCHMARK_TUNE_FIXED_THREADS);
  testAssert(PlayUtils::getBenchmarkTuneMode(
    false,true,true,false,false,false,false
  ) == PlayUtils::BENCHMARK_TUNE_EXPLICIT_BATCH_GRID);
  testAssert(PlayUtils::getBenchmarkTuneMode(
    false,false,false,true,false,false,true
  ) == PlayUtils::BENCHMARK_TUNE_AUTO_BATCH);
  const vector<vector<bool>> invalidModes = {
    {true,false,false,true,false,false,false},
    {false,true,false,true,false,false,false},
    {false,false,true,true,false,false,false},
    {false,false,false,true,true,false,false},
    {false,false,false,false,false,false,true},
  };
  for(const vector<bool>& mode: invalidModes) {
    bool threw = false;
    try {
      (void)PlayUtils::getBenchmarkTuneMode(
        mode[0],mode[1],mode[2],mode[3],mode[4],mode[5],mode[6]
      );
    }
    catch(const StringError&) {
      threw = true;
    }
    testAssert(threw);
  }

  const vector<int> scoutThreads = PlayUtils::getAutoBatchScoutThreads(2,128);
  testAssert(scoutThreads.size() == 5);
  testAssert(std::find(scoutThreads.begin(),scoutThreads.end(),32) != scoutThreads.end());
  testAssert(std::find(scoutThreads.begin(),scoutThreads.end(),64) != scoutThreads.end());
  testAssert(std::find(scoutThreads.begin(),scoutThreads.end(),128) != scoutThreads.end());

  vector<PlayUtils::BenchmarkResults> nonMonotonicScout(4);
  for(int i = 0; i<nonMonotonicScout.size(); i++) {
    nonMonotonicScout[i].numThreads = 16 << i;
    nonMonotonicScout[i].numNNEvals = 100 + 10*i;
    nonMonotonicScout[i].totalSeconds = 1.0;
    nonMonotonicScout[i].avgBatchSize = 8+i;
  }
  nonMonotonicScout[1].numNNEvals = 300;
  nonMonotonicScout[1].avgBatchSize = 20.2;
  nonMonotonicScout[2].numNNEvals = 150;
  nonMonotonicScout[3].numNNEvals = 250;
  testAssert(PlayUtils::getAutoBatchScoutBatchSize(nonMonotonicScout,128) == 20);

  const vector<int> profileCandidates = PlayUtils::getAutoBatchProfileCandidates(128,20,6);
  testAssert(profileCandidates == vector<int>({128,20,19,21,18,22}));
  const vector<int> boundaryCandidates = PlayUtils::getAutoBatchProfileCandidates(3,1,10);
  testAssert(boundaryCandidates.size() == 3);
  testAssert(boundaryCandidates[0] == 3);
  testAssert(std::set<int>(boundaryCandidates.begin(),boundaryCandidates.end()).size() == boundaryCandidates.size());

  const vector<int> batch20Stencil = PlayUtils::getAutoBatchThreadStencil(20,2);
  testAssert(batch20Stencil == vector<int>({56,60,64}));

  vector<PlayUtils::BenchmarkResults> spikyProfile(3);
  spikyProfile[0].numThreads = 56;
  spikyProfile[0].numNNEvals = 100;
  spikyProfile[0].totalSeconds = 1.0;
  spikyProfile[1].numThreads = 60;
  spikyProfile[1].numNNEvals = 500;
  spikyProfile[1].totalSeconds = 1.0;
  spikyProfile[2].numThreads = 64;
  spikyProfile[2].numNNEvals = 110;
  spikyProfile[2].totalSeconds = 1.0;
  const PlayUtils::AutoBatchProfileResult spikySummary =
    PlayUtils::summarizeAutoBatchProfile(20,spikyProfile);
  testAssert(spikySummary.medianNNEvalsPerSecond == 110.0);
  testAssert(spikySummary.finalist.numThreads == 60);

  vector<PlayUtils::BenchmarkResults> confirmationA(2);
  confirmationA[0].numNNEvals = 100;
  confirmationA[0].totalSeconds = 1.0;
  confirmationA[1].numNNEvals = 100;
  confirmationA[1].totalSeconds = 9.0;
  vector<PlayUtils::BenchmarkResults> confirmationB(2);
  confirmationB[0].numNNEvals = 30;
  confirmationB[0].totalSeconds = 1.0;
  confirmationB[1].numNNEvals = 30;
  confirmationB[1].totalSeconds = 1.0;
  testAssert(
    0.5 * (confirmationA[0].getNNEvalsPerSecond() + confirmationA[1].getNNEvalsPerSecond()) >
    0.5 * (confirmationB[0].getNNEvalsPerSecond() + confirmationB[1].getNNEvalsPerSecond())
  );
  testAssert(PlayUtils::getPooledNNEvalsPerSecond(confirmationA) == 20.0);
  testAssert(PlayUtils::getPooledNNEvalsPerSecond(confirmationB) == 30.0);
  testAssert(PlayUtils::isBetterAutoBatchConfirmation(
    21,63,confirmationB,20,60,confirmationA
  ));
  testAssert(PlayUtils::isBetterAutoBatchConfirmation(
    19,64,confirmationB,20,60,confirmationB
  ));
  testAssert(PlayUtils::isBetterAutoBatchConfirmation(
    20,60,confirmationB,20,64,confirmationB
  ));

  PlayUtils::AutoBatchProfileResult equalMedianLargerBatch = spikySummary;
  equalMedianLargerBatch.maxBatchSize = 21;
  PlayUtils::AutoBatchProfileResult equalMedianSmallerBatch = spikySummary;
  equalMedianSmallerBatch.maxBatchSize = 20;
  testAssert(PlayUtils::isBetterAutoBatchProfile(
    equalMedianSmallerBatch,equalMedianLargerBatch
  ));

  testAssert(PlayUtils::getMaxUntestedBatchSizeGap(vector<int>({18,19,20,21,22,128}),128) == 105);

  cout << "testbenchmarkresults okay" << endl;
}

void Tests::runCollectFilesTests() {
  {
    vector<string> collected;
    cout << "Collecting sgfs from tests" << endl;
    FileHelpers::collectSgfsFromDir("tests", collected);
    std::sort(collected.begin(),collected.end());
    for(const string& s: collected) {
      cout << s << endl;
    }
  }
  {
    vector<string> collected;
    cout << "Collecting cfgs from tests" << endl;
    FileUtils::collectFiles("tests", [](const std::string& s) {return Global::isSuffix(s,".cfg");}, collected);
    std::sort(collected.begin(),collected.end());
    for(const string& s: collected) {
      cout << s << endl;
    }
  }
}

void Tests::runLoadModelTests() {
  bool logToStdoutDefault = true;
  bool logToStderrDefault = false;
  bool logTimeDefault = false;
  Logger logger(nullptr, logToStdoutDefault, logToStderrDefault, logTimeDefault);

  {
    string modelsDir = "tests/models/findLatestModelTest1";
    string modelName;
    string modelFile;
    string modelDir;
    time_t modelTime;
    bool suc = LoadModel::findLatestModel(modelsDir, logger, modelName, modelFile, modelDir, modelTime);
    testAssert(suc);
    cout << modelsDir << endl;
    cout << modelName << " " << modelFile << " " << modelDir << " " << modelTime << endl;
    testAssert(modelTime == 0);
    testAssert(modelName == "random");
    testAssert(modelDir == "/dev/null");
    testAssert(modelFile == "/dev/null");
  }

  {
    string modelsDir = "tests/models/findLatestModelTest2";
    string modelName;
    string modelFile;
    string modelDir;
    time_t modelTime;
    bool suc = LoadModel::findLatestModel(modelsDir, logger, modelName, modelFile, modelDir, modelTime);
    testAssert(suc);
    cout << modelsDir << endl;
    cout << modelName << " " << modelFile << " " << modelDir << endl;
    testAssert(modelTime > 0);
    testAssert(modelName == "abc.bin.gz");
    testAssert(modelDir == "tests/models/findLatestModelTest2" || modelDir == "tests\\models\\findLatestModelTest2");
    testAssert(modelFile == "tests/models/findLatestModelTest2/abc.bin.gz" || modelFile == "tests\\models\\findLatestModelTest2\\abc.bin.gz");
    testAssert(FileUtils::weaklyCanonical(modelDir) == FileUtils::weaklyCanonical(modelsDir));
    testAssert(Global::isPrefix(FileUtils::weaklyCanonical(modelDir), FileUtils::weaklyCanonical(modelsDir)));
  }


  {
    string modelsDir = "tests/models/findLatestModelTest3";
    string modelName;
    string modelFile;
    string modelDir;
    time_t modelTime;
    bool suc = LoadModel::findLatestModel(modelsDir, logger, modelName, modelFile, modelDir, modelTime);
    testAssert(suc);
    cout << modelsDir << endl;
    cout << modelName << " " << modelFile << " " << modelDir << endl;
    testAssert(modelTime > 0);
    testAssert(modelName == "def");
    testAssert(modelDir == "tests/models/findLatestModelTest3/def" || modelDir == "tests\\models\\findLatestModelTest3\\def");
    testAssert(modelFile == "tests/models/findLatestModelTest3/def/model.bin.gz" || modelFile == "tests\\models\\findLatestModelTest3\\def\\model.bin.gz");
    testAssert(FileUtils::weaklyCanonical(modelDir) != FileUtils::weaklyCanonical(modelsDir));
    testAssert(Global::isPrefix(FileUtils::weaklyCanonical(modelDir), FileUtils::weaklyCanonical(modelsDir)));
  }

  {
    LoadModel::setLastModifiedTimeToNow("tests/models/findLatestModelTest4/abc.bin.gz", logger);

    string modelsDir = "tests/models/findLatestModelTest4";
    string modelName;
    string modelFile;
    string modelDir;
    time_t modelTime;
    bool suc = LoadModel::findLatestModel(modelsDir, logger, modelName, modelFile, modelDir, modelTime);
    testAssert(suc);
    cout << modelsDir << endl;
    cout << modelName << " " << modelFile << " " << modelDir << endl;
    testAssert(modelTime > 0);
    testAssert(modelName == "abc.bin.gz");
    testAssert(modelDir == "tests/models/findLatestModelTest4" || modelDir == "tests\\models\\findLatestModelTest4");
    testAssert(modelFile == "tests/models/findLatestModelTest4/abc.bin.gz" || modelFile == "tests\\models\\findLatestModelTest4\\abc.bin.gz");
    testAssert(FileUtils::weaklyCanonical(modelDir) == FileUtils::weaklyCanonical(modelsDir));
    testAssert(Global::isPrefix(FileUtils::weaklyCanonical(modelDir), FileUtils::weaklyCanonical(modelsDir)));
  }
  std::this_thread::sleep_for(std::chrono::duration<double>(1.5));
  {
    LoadModel::setLastModifiedTimeToNow("tests/models/findLatestModelTest4/def/model.bin.gz", logger);

    string modelsDir = "tests/models/findLatestModelTest4";
    string modelName;
    string modelFile;
    string modelDir;
    time_t modelTime;
    bool suc = LoadModel::findLatestModel(modelsDir, logger, modelName, modelFile, modelDir, modelTime);
    testAssert(suc);
    cout << modelsDir << endl;
    cout << modelName << " " << modelFile << " " << modelDir << endl;
    testAssert(modelTime > 0);
    testAssert(modelName == "def");
    testAssert(modelDir == "tests/models/findLatestModelTest4/def" || modelDir == "tests\\models\\findLatestModelTest4\\def");
    testAssert(modelFile == "tests/models/findLatestModelTest4/def/model.bin.gz" || "tests\\models\\findLatestModelTest4\\def\\model.bin.gz");
    testAssert(FileUtils::weaklyCanonical(modelDir) != FileUtils::weaklyCanonical(modelsDir));
    testAssert(Global::isPrefix(FileUtils::weaklyCanonical(modelDir), FileUtils::weaklyCanonical(modelsDir)));
  }
  std::this_thread::sleep_for(std::chrono::duration<double>(1.5));
  {
    LoadModel::setLastModifiedTimeToNow("tests/models/findLatestModelTest4/def/ghi.bin.gz", logger);

    string modelsDir = "tests/models/findLatestModelTest4";
    string modelName;
    string modelFile;
    string modelDir;
    time_t modelTime;
    bool suc = LoadModel::findLatestModel(modelsDir, logger, modelName, modelFile, modelDir, modelTime);
    testAssert(suc);
    cout << modelsDir << endl;
    cout << modelName << " " << modelFile << " " << modelDir << endl;
    testAssert(modelTime > 0);
    testAssert(modelName == "ghi.bin.gz");
    testAssert(modelDir == "tests/models/findLatestModelTest4/def" || modelDir == "tests\\models\\findLatestModelTest4\\def");
    testAssert(modelFile == "tests/models/findLatestModelTest4/def/ghi.bin.gz" || modelFile == "tests\\models\\findLatestModelTest4\\def\\ghi.bin.gz");
    testAssert(FileUtils::weaklyCanonical(modelDir) != FileUtils::weaklyCanonical(modelsDir));
    testAssert(Global::isPrefix(FileUtils::weaklyCanonical(modelDir), FileUtils::weaklyCanonical(modelsDir)));
  }
  cout << "testloadmodel okay" << endl;
}
