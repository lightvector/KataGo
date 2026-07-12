#include "../core/global.h"
#include "../core/fileutils.h"
#include "../core/rand.h"
#include "../dataio/sgf.h"
#include "../dataio/files.h"
#include "../dataio/numpywrite.h"
#include "../neuralnet/nninputs.h"
#include "../neuralnet/nninterface.h"
#include "../neuralnet/modelversion.h"
#include "../command/commandline.h"
#include "../main.h"

using namespace std;

// Dumps a calibration dataset (NN input feature tensors, sampled from real games) for offline
// post-training static quantization of an exported .onnx model -- e.g. via AMD's amd-quark
// (quark.onnx), to produce an INT8 QDQ model that the VitisAI EP can actually accelerate on NPU
// (the VitisAI EP falls back to CPU for FP32 ops, so quantization is a prerequisite for NPU
// offload, not an optimization). See Compiling.md's ONNX/VitisAI section.
//
// Deliberately reuses NNInputs::fillRowV7 -- the exact same feature-encoding code path used at
// real inference time -- rather than any separate/reimplemented extraction logic, so calibration
// data is guaranteed to match what the network actually sees at inference. Does NOT load any
// neural net compute backend (GPU/NPU/CPU session) -- feature extraction only depends on the
// model's channel counts (from ModelDesc, no session needed), not on a working backend. This
// keeps the tool fast and usable on any machine, independent of any particular execution
// provider's availability or quirks.
int MainCmds::dumpcalibrationdata(const vector<string>& args) {
  Board::initHash();

  string modelFile;
  vector<string> sgfFilesFromCmdline;
  vector<string> sgfDirs;
  vector<string> sgfsDirs;
  string outputFile;
  int nnXLen;
  int nnYLen;
  int positionsPerGame;
  int maxPositions;
  string randSeed;

  try {
    KataGoCommandLine cmd(
      "Dump a calibration dataset of NN input tensors (spatial+global features) sampled from sgf "
      "games, for offline INT8 quantization of an exported ONNX model (e.g. via AMD's amd-quark, "
      "for the VitisAI/NPU execution provider). Does not require a GPU/NPU or a config file -- "
      "purely extracts board features from the model's channel counts."
    );
    cmd.addModelFileArg();

    TCLAP::MultiArg<string> sgfArg("","sgf","Sgf file",false,"SGF");
    TCLAP::MultiArg<string> sgfDirArg("","sgfdir","Directory of sgf files",false,"DIR");
    TCLAP::MultiArg<string> sgfsDirArg("","sgfsdir","Directory of sgfs (multi-game-per-file) files",false,"DIR");
    TCLAP::ValueArg<string> outputArg("o","output","Output npz file path",true,string(),"FILE");
    TCLAP::ValueArg<int> xLenArg("x","xlen","Board x size baked into calibration tensors",false,19,"N");
    TCLAP::ValueArg<int> yLenArg("y","ylen","Board y size baked into calibration tensors",false,19,"N");
    TCLAP::ValueArg<int> positionsPerGameArg(
      "","positions-per-game","How many positions to sample per game, evenly spaced across its move range",false,8,"N"
    );
    TCLAP::ValueArg<int> maxPositionsArg("","max-positions","Stop after collecting this many total positions",false,2000,"N");
    TCLAP::ValueArg<string> randSeedArg("","rand-seed","Random seed for shuffling game order",false,string(),"SEED");

    cmd.add(sgfArg);
    cmd.add(sgfDirArg);
    cmd.add(sgfsDirArg);
    cmd.add(outputArg);
    cmd.add(xLenArg);
    cmd.add(yLenArg);
    cmd.add(positionsPerGameArg);
    cmd.add(maxPositionsArg);
    cmd.add(randSeedArg);
    cmd.parseArgs(args);

    modelFile = cmd.getModelFile();
    sgfFilesFromCmdline = sgfArg.getValue();
    sgfDirs = sgfDirArg.getValue();
    sgfsDirs = sgfsDirArg.getValue();
    outputFile = outputArg.getValue();
    nnXLen = xLenArg.getValue();
    nnYLen = yLenArg.getValue();
    positionsPerGame = positionsPerGameArg.getValue();
    maxPositions = maxPositionsArg.getValue();
    randSeed = randSeedArg.getValue();
  }
  catch(TCLAP::ArgException& e) {
    cerr << "Error: " << e.error() << " for argument " << e.argId() << endl;
    return 1;
  }

  if(nnXLen < 2 || nnXLen > NNPos::MAX_BOARD_LEN || nnYLen < 2 || nnYLen > NNPos::MAX_BOARD_LEN)
    throw StringError("Invalid board size for dumpcalibrationdata");
  if(positionsPerGame <= 0)
    throw StringError("positions-per-game must be positive");
  if(maxPositions <= 0)
    throw StringError("max-positions must be positive");
  if(randSeed.size() <= 0)
    randSeed = Global::uint64ToString(Rand().nextUInt64());

  // Load just the model's channel-count metadata -- no compute backend/session is created.
  const string expectedSha256 = "";
  std::unique_ptr<LoadedModel, void(*)(LoadedModel*)> loadedModel(
    NeuralNet::loadModelFile(modelFile, expectedSha256),
    NeuralNet::freeLoadedModel
  );
  const ModelDesc& modelDesc = NeuralNet::getModelDesc(loadedModel.get());
  int numSpatialFeatures = modelDesc.numInputChannels;
  int numGlobalFeatures = modelDesc.numInputGlobalChannels;
  if(numSpatialFeatures != NNInputs::NUM_FEATURES_SPATIAL_V7 || numGlobalFeatures != NNInputs::NUM_FEATURES_GLOBAL_V7) {
    throw StringError(
      "dumpcalibrationdata only supports the current V7 feature set (spatial=" +
      Global::intToString(NNInputs::NUM_FEATURES_SPATIAL_V7) + ", global=" +
      Global::intToString(NNInputs::NUM_FEATURES_GLOBAL_V7) + "), but the model reports spatial=" +
      Global::intToString(numSpatialFeatures) + " global=" + Global::intToString(numGlobalFeatures)
    );
  }
  if(modelDesc.numInputMetaChannels > 0) {
    cerr << "Warning: model has " << modelDesc.numInputMetaChannels
         << " metadata input channels; dumpcalibrationdata does not populate these (left as zero)." << endl;
  }

  // Collect sgf files --------------------------------------------------------
  vector<string> sgfFiles;
  FileHelpers::collectSgfsFromDirsOrFiles(sgfDirs,sgfFiles);
  for(const string& s: sgfFilesFromCmdline)
    sgfFiles.push_back(s);

  vector<string> sgfsFiles;
  FileHelpers::collectMultiSgfsFromDirsOrFiles(sgfsDirs,sgfsFiles);

  if(sgfFiles.size() <= 0 && sgfsFiles.size() <= 0)
    throw StringError("No sgf files specified or found (use -sgf, -sgfdir, and/or -sgfsdir)");

  vector<std::unique_ptr<CompactSgf>> compactSgfs;
  for(const string& f : sgfFiles) {
    try {
      compactSgfs.push_back(CompactSgf::loadFile(f));
    }
    catch(const std::exception& e) {
      cerr << "Warning: skipping sgf that failed to load: " << f << " (" << e.what() << ")" << endl;
    }
  }
  for(const string& f : sgfsFiles) {
    vector<std::unique_ptr<Sgf>> sgfs;
    try {
      sgfs = Sgf::loadSgfsFile(f);
    }
    catch(const std::exception& e) {
      cerr << "Warning: skipping sgfs file that failed to load: " << f << " (" << e.what() << ")" << endl;
      continue;
    }
    for(auto& sgf : sgfs) {
      try {
        compactSgfs.push_back(std::unique_ptr<CompactSgf>(new CompactSgf(std::move(*sgf))));
      }
      catch(const std::exception& e) {
        cerr << "Warning: skipping one game within " << f << " that failed to compact (" << e.what() << ")" << endl;
      }
    }
  }

  if(compactSgfs.size() <= 0)
    throw StringError("No sgf games successfully loaded");

  cout << "Loaded " << compactSgfs.size() << " games." << endl;

  // Shuffle game order deterministically so a max-positions cutoff doesn't bias toward
  // whatever happens to sort first (e.g. alphabetically-early filenames).
  Rand rand(randSeed);
  rand.shuffle(compactSgfs);

  // Sample positions -----------------------------------------------------------
  int64_t maxRows = (int64_t)maxPositions;
  NumpyBuffer<float> binaryInputNCHW(std::vector<int64_t>({maxRows,(int64_t)numSpatialFeatures,(int64_t)nnYLen,(int64_t)nnXLen}));
  NumpyBuffer<float> globalInputNC(std::vector<int64_t>({maxRows,(int64_t)numGlobalFeatures}));

  int64_t numRowsWritten = 0;
  int numGamesUsed = 0;
  MiscNNInputParams nnInputParams;
  const bool inputsUseNHWC = false;

  for(size_t gameIdx = 0; gameIdx < compactSgfs.size() && numRowsWritten < maxRows; gameIdx++) {
    CompactSgf* sgf = compactSgfs[gameIdx].get();
    int64_t numMoves = (int64_t)sgf->moves.size();
    if(numMoves <= 0)
      continue;
    if(sgf->xSize != nnXLen || sgf->ySize != nnYLen)
      continue; // Skip games whose board size doesn't match the requested calibration board size

    Rules initialRules;
    try {
      initialRules = sgf->getRulesOrWarn(
        Rules::getTrompTaylorish(),
        [](const string& msg) { (void)msg; } // Suppress per-game rules warnings; too noisy over many games.
      );
    }
    catch(const std::exception& e) {
      cerr << "Warning: skipping game with unparseable rules: " << e.what() << endl;
      continue;
    }

    bool usedAnyPositionThisGame = false;
    for(int i = 0; i < positionsPerGame && numRowsWritten < maxRows; i++) {
      // Evenly space sampled move numbers across the game, skewed away from move 0 (near-empty
      // boards contribute little to calibration) and away from the very last move.
      int64_t moveNum = (numMoves * (int64_t)(i+1)) / (positionsPerGame + 1);
      if(moveNum < 0) moveNum = 0;
      if(moveNum > numMoves) moveNum = numMoves;

      Board board;
      Player nextPla;
      BoardHistory hist;
      bool ok = true;
      try {
        sgf->setupInitialBoardAndHist(initialRules, board, nextPla, hist);
        sgf->playMovesTolerant(board, nextPla, hist, moveNum, false);
      }
      catch(const std::exception&) {
        ok = false; // Skip this position (e.g. an illegal move somewhere in the replay)
      }
      if(!ok)
        continue;

      float* spatialRow = binaryInputNCHW.data + numRowsWritten * (int64_t)numSpatialFeatures * nnXLen * nnYLen;
      float* globalRow = globalInputNC.data + numRowsWritten * (int64_t)numGlobalFeatures;
      NNInputs::fillRowV7(board, hist, nextPla, nnInputParams, nnXLen, nnYLen, inputsUseNHWC, spatialRow, globalRow);

      numRowsWritten++;
      usedAnyPositionThisGame = true;
    }
    if(usedAnyPositionThisGame) {
      numGamesUsed++;
      if(numGamesUsed % 200 == 0)
        cout << "..." << numRowsWritten << " positions collected from " << numGamesUsed << " games" << endl;
    }
  }

  if(numRowsWritten <= 0)
    throw StringError("Collected zero calibration positions (check board size matches your sgfs, and that games have moves)");

  cout << "Collected " << numRowsWritten << " calibration positions from " << numGamesUsed << " games." << endl;

  uint64_t spatialBytes = binaryInputNCHW.prepareHeaderWithNumRows(numRowsWritten);
  uint64_t globalBytes = globalInputNC.prepareHeaderWithNumRows(numRowsWritten);

  ZipFile zipFile(outputFile);
  zipFile.writeBuffer("binaryInputNCHW", binaryInputNCHW.dataIncludingHeader, spatialBytes);
  zipFile.writeBuffer("globalInputNC", globalInputNC.dataIncludingHeader, globalBytes);
  zipFile.close();

  cout << "Wrote " << numRowsWritten << " rows to " << outputFile << endl;
  cout << "  binaryInputNCHW shape: [" << numRowsWritten << ", " << numSpatialFeatures << ", " << nnYLen << ", " << nnXLen << "]" << endl;
  cout << "  globalInputNC shape: [" << numRowsWritten << ", " << numGlobalFeatures << "]" << endl;
  cout << "Note: channel 0 of binaryInputNCHW is the on-board mask (per KataGo convention), matching" << endl;
  cout << "the InputSpatial tensor consumed by the exported ONNX graph (see exportonnx)." << endl;

  return 0;
}
