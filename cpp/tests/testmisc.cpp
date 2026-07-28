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
