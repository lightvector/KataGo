#include "../tests/tests.h"

#include "../core/fileutils.h"
#include "../dataio/files.h"
#include "../dataio/loadmodel.h"
#include "../neuralnet/desc.h"

#include <chrono>
#include <sstream>
#include <thread>

//------------------------
#include "../core/using.h"
//------------------------
using namespace TestCommon;

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

  //A real v17 model file whose header slot after metaEncoderVersion is patched to 1 parses as
  //declaring preferPassAliveUnderSuicideRules - the engine side of the model-declaration handshake
  //that lets the alwaysComputePassAliveUnderSuicideRules auto mode turn on. The same slot layout is
  //what export_model_pytorch.py writes.
  {
    const string modelFile = "tests/models/b7c96h6kv3qk32v16tflrs-fson-bnh.bin.gz";
    string uncompressed;
    FileUtils::uncompressAndLoadFileIntoString(modelFile,"",uncompressed);

    size_t binStart = uncompressed.find("@BIN@");
    testAssert(binStart != string::npos);
    const string headerPrefix = uncompressed.substr(0,binStart);
    const string rest = uncompressed.substr(binStart);
    vector<string> tokens = Global::split(headerPrefix);
    //Header layout: name, version, numInputChannels, numInputGlobalChannels, 7 postprocess params,
    //metaEncoderVersion, preferPassAliveUnderSuicideRules, 6 unused option slots, then the trunk.
    testAssert(tokens.size() > 19);
    testAssert(tokens[1] == "17");
    testAssert(tokens[11] == "0");
    testAssert(tokens[12] == "0");
    testAssert(tokens[19] == "trunk");

    auto parseWithSlot = [&](const string& slotValue) {
      vector<string> patched = tokens;
      patched[12] = slotValue;
      string contents = Global::concat(patched," ") + " " + rest;
      std::istringstream in(contents);
      return ModelDesc(in,"",true);
    };

    {
      ModelDesc desc = parseWithSlot("0");
      testAssert(desc.modelVersion == 17);
      testAssert(!desc.preferPassAliveUnderSuicideRules);
    }
    {
      ModelDesc desc = parseWithSlot("1");
      testAssert(desc.modelVersion == 17);
      testAssert(desc.preferPassAliveUnderSuicideRules);
    }
    {
      bool threw = false;
      try {
        ModelDesc desc = parseWithSlot("2");
      }
      catch(const StringError&) {
        threw = true;
      }
      testAssert(threw);
    }
    cout << "model declaration parsing okay" << endl;
  }
}
