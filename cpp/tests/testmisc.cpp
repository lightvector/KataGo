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

  //A real v17 model file whose header slots after metaEncoderVersion are patched to 1 parse as
  //declaring preferPassAliveUnderSuicideRules and preferExcludeTerritoryAdjacentToAtari - the engine
  //side of the model-declaration handshake that lets the corresponding auto modes turn on. The same
  //slot layout is what export_model_pytorch.py writes.
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
    //metaEncoderVersion, preferPassAliveUnderSuicideRules, preferExcludeTerritoryAdjacentToAtari,
    //5 unused option slots, then the trunk.
    const size_t passAliveSlot = 12;
    const size_t excludeTerritorySlot = 13;
    testAssert(tokens.size() > 19);
    testAssert(tokens[1] == "17");
    testAssert(tokens[11] == "0");
    testAssert(tokens[passAliveSlot] == "0");
    testAssert(tokens[excludeTerritorySlot] == "0");
    testAssert(tokens[19] == "trunk");

    auto parseWithSlot = [&](size_t slot, const string& slotValue) {
      vector<string> patched = tokens;
      patched[slot] = slotValue;
      string contents = Global::concat(patched," ") + " " + rest;
      std::istringstream in(contents);
      return ModelDesc(in,"",true);
    };
    auto parseThrows = [&](size_t slot, const string& slotValue) {
      bool threw = false;
      try {
        ModelDesc desc = parseWithSlot(slot,slotValue);
      }
      catch(const StringError&) {
        threw = true;
      }
      return threw;
    };

    {
      ModelDesc desc = parseWithSlot(passAliveSlot,"0");
      testAssert(desc.modelVersion == 17);
      testAssert(!desc.preferPassAliveUnderSuicideRules);
      testAssert(!desc.preferExcludeTerritoryAdjacentToAtari);
    }
    {
      ModelDesc desc = parseWithSlot(passAliveSlot,"1");
      testAssert(desc.modelVersion == 17);
      testAssert(desc.preferPassAliveUnderSuicideRules);
      testAssert(!desc.preferExcludeTerritoryAdjacentToAtari);
    }
    {
      ModelDesc desc = parseWithSlot(excludeTerritorySlot,"1");
      testAssert(desc.modelVersion == 17);
      testAssert(!desc.preferPassAliveUnderSuicideRules);
      testAssert(desc.preferExcludeTerritoryAdjacentToAtari);
    }
    testAssert(parseThrows(passAliveSlot,"2"));
    testAssert(parseThrows(excludeTerritorySlot,"2"));
    cout << "model declaration parsing okay" << endl;
  }
}
