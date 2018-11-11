
#include "../dataio/trainingwrite.h"

ValueTargets::ValueTargets()
  :win(0),
   loss(0),
   noResult(0),
   scoreValue(0),
   score(0),
   mctsUtility1(0),
   mctsUtility4(0),
   mctsUtility16(0),
   mctsUtility64(0),
   mctsUtility256(0)
{}
ValueTargets::~ValueTargets()
{}

//-------------------------------------------------------------------------------------

SidePosition::SidePosition()
  :board(),
   hist(),
   pla(P_BLACK),
   policyTarget(),
   whiteValueTargets()
{}

SidePosition::SidePosition(const Board& b, const BoardHistory& h, Player p)
  :board(b),
   hist(h),
   pla(p),
   policyTarget(),
   whiteValueTargets()
{}

SidePosition::~SidePosition()
{}

//-------------------------------------------------------------------------------------

FinishedGameData::FinishedGameData()
  :bName(),
   wName(),
   bIdx(0),
   wIdx(0),

   preStartBoard(),
   startBoard(),
   startHist(),
   endHist(),
   startPla(P_BLACK),
   gameHash(),

   drawEquivalentWinsForWhite(0.0),
   hitTurnLimit(false),

   firstTrainingTurn(0),
   mode(0),
   modeMeta1(0),
   modeMeta2(0),

   hasFullData(false),
   posLen(-1),
   policyTargetsByTurn(),
   whiteValueTargetsByTurn(),
   finalWhiteOwnership(NULL),

   sidePositions()
{
}

FinishedGameData::~FinishedGameData() {
  for(int i = 0; i<policyTargetsByTurn.size(); i++)
    delete policyTargetsByTurn[i];

  if(finalWhiteOwnership != NULL)
    delete[] finalWhiteOwnership;

  for(int i = 0; i<sidePositions.size(); i++)
    delete sidePositions[i];
}


//-------------------------------------------------------------------------------------


//Don't forget to update everything else in the header file and the code below too if changing any of these
//And update the python code
static const int POLICY_TARGET_NUM_CHANNELS = 1;
static const int GLOBAL_TARGET_NUM_CHANNELS = 46;
static const int VALUE_SPATIAL_TARGET_NUM_CHANNELS = 1;

TrainingWriteBuffers::TrainingWriteBuffers(int iVersion, int maxRws, int numBChannels, int numFChannels, int pLen)
  :inputsVersion(iVersion),
   maxRows(maxRws),
   numBinaryChannels(numBChannels),
   numGlobalChannels(numFChannels),
   posLen(pLen),
   packedBoardArea((pLen*pLen + 7)/8),
   curRows(0),
   binaryInputNCHWUnpacked(NULL),
   binaryInputNCHWPacked({maxRws, numBChannels, packedBoardArea}),
   globalInputNC({maxRws, numFChannels}),
   policyTargetsNCMove({maxRws, POLICY_TARGET_NUM_CHANNELS, NNPos::getPolicySize(pLen)}),
   globalTargetsNC({maxRws, GLOBAL_TARGET_NUM_CHANNELS}),
   valueTargetsNCHW({maxRws, VALUE_SPATIAL_TARGET_NUM_CHANNELS, pLen, pLen})
{
  binaryInputNCHWUnpacked = new float[numBChannels * pLen * pLen];
}

TrainingWriteBuffers::~TrainingWriteBuffers()
{
  delete[] binaryInputNCHWUnpacked;
}

void TrainingWriteBuffers::clear() {
  curRows = 0;
}

//Copy floats that are all 0-1 into bits, packing 8 to a byte, big-endian-style within each byte.
static void packBits(const float* binaryFloats, int len, uint8_t* bits) {
  for(int i = 0; i < len; i += 8) {
    if(i + 8 <= len) {
      bits[i >> 3] =
        ((uint8_t)binaryFloats[i + 0] << 7) |
        ((uint8_t)binaryFloats[i + 1] << 6) |
        ((uint8_t)binaryFloats[i + 2] << 5) |
        ((uint8_t)binaryFloats[i + 3] << 4) |
        ((uint8_t)binaryFloats[i + 4] << 3) |
        ((uint8_t)binaryFloats[i + 5] << 2) |
        ((uint8_t)binaryFloats[i + 6] << 1) |
        ((uint8_t)binaryFloats[i + 7] << 0);
    }
    else {
      bits[i >> 3] = 0;
      for(int di = 0; i + di < len; di++) {
        bits[i >> 3] |= ((uint8_t)binaryFloats[i + di] << (7-di));
      }
    }
  }
}

static void zeroPolicyTarget(int policySize, int16_t* target) {
  for(int pos = 0; pos<policySize; pos++)
    target[pos] = 0;
}

//Copy playouts into target, expanding out the sparse representation into a full plane.
static void fillPolicyTarget(const vector<PolicyTargetMove>& policyTargetMoves, int policySize, int posLen, int boardXSize, int16_t* target) {
  zeroPolicyTarget(policySize,target);
  size_t size = policyTargetMoves.size();
  for(size_t i = 0; i<size; i++) {
    const PolicyTargetMove& move = policyTargetMoves[i];
    int pos = NNPos::locToPos(move.loc, boardXSize, posLen);
    assert(pos >= 0 && pos < policySize);
    target[pos] = move.policyTarget;
  }
}

static float fsq(float x) {
  return x * x;
}

static void fillValueTDTargets(const vector<ValueTargets>& whiteValueTargetsByTurn, int idx, Player nextPlayer, float nowFactor, float* buf) {
  double winValue = 0.0;
  double lossValue = 0.0;
  double noResultValue = 0.0;
  double scoreValue = 0.0;

  double weightLeft = 1.0;
  for(int i = idx; i<whiteValueTargetsByTurn.size(); i++) {
    double weightNow;
    if(i == whiteValueTargetsByTurn.size() - 1) {
      weightNow = weightLeft;
      weightLeft = 0.0;
    }
    else {
      weightNow = weightLeft * nowFactor;
      weightLeft *= (1.0 - nowFactor);
    }

    //Training rows need things from the perspective of the player to move, so we flip as appropriate.
    const ValueTargets& targets = whiteValueTargetsByTurn[i];
    winValue += weightNow * (nextPlayer == P_WHITE ? targets.win : targets.loss);
    lossValue += weightNow * (nextPlayer == P_WHITE ? targets.loss : targets.win);
    noResultValue = weightNow * targets.noResult;
    scoreValue = weightNow * (nextPlayer == P_WHITE ? targets.scoreValue : -targets.scoreValue);
  }
  buf[0] = (float)winValue;
  buf[1] = (float)lossValue;
  buf[2] = (float)noResultValue;
  buf[3] = (float)scoreValue;
}

void TrainingWriteBuffers::addRow(
  const Board& board, const BoardHistory& hist, Player nextPlayer,
  int turnNumberAfterStart,
  const vector<PolicyTargetMove>* policyTarget0, //can be null
  const vector<ValueTargets>& whiteValueTargets,
  int whiteValueTargetsIdx, //index in whiteValueTargets corresponding to this turn.
  int8_t* finalWhiteOwnership,
  bool isSidePosition,
  const FinishedGameData& data,
  Rand& rand
) {
  if(inputsVersion < 3 || inputsVersion > 3)
    throw StringError("Training write buffers: Does not support input version: " + Global::intToString(inputsVersion));

  int posArea = posLen*posLen;
  assert(data.posLen == posLen);
  assert(data.hasFullData);

  {
    bool inputsUseNHWC = false;
    float* rowBin = binaryInputNCHWUnpacked;
    float* rowGlobal = globalInputNC.data + curRows * numGlobalChannels;
    if(inputsVersion == 3) {
      assert(NNInputs::NUM_FEATURES_BIN_V3 == numBinaryChannels);
      assert(NNInputs::NUM_FEATURES_GLOBAL_V3 == numGlobalChannels);
      NNInputs::fillRowV3(board, hist, nextPlayer, data.drawEquivalentWinsForWhite, posLen, inputsUseNHWC, rowBin, rowGlobal);
    }
    else
      assert(false);

    //Pack bools bitwise into uint8_t
    uint8_t* rowBinPacked = binaryInputNCHWPacked.data + curRows * numBinaryChannels * packedBoardArea;
    for(int c = 0; c<numBinaryChannels; c++)
      packBits(rowBin + c * posArea, posArea, rowBinPacked + c * packedBoardArea);
  }

  //Vector for global targets and metadata
  float* rowGlobal = globalTargetsNC.data + curRows * GLOBAL_TARGET_NUM_CHANNELS;

  //Fill policy
  int policySize = NNPos::getPolicySize(posLen);
  int16_t* rowPolicy = policyTargetsNCMove.data + curRows * POLICY_TARGET_NUM_CHANNELS * policySize;

  if(policyTarget0 != NULL) {
    fillPolicyTarget(*policyTarget0, policySize, posLen, board.x_size, rowPolicy + 0 * policySize);
    rowGlobal[25] = 1.0f;
  }
  else {
    zeroPolicyTarget(policySize, rowPolicy + 0 * policySize);
    rowGlobal[25] = 0.0f;
  }

  //Unused
  rowGlobal[27] = 0.0f;

  //Fill td-like value targets
  assert(whiteValueTargetsIdx >= 0 && whiteValueTargetsIdx < whiteValueTargets.size());
  fillValueTDTargets(whiteValueTargets, whiteValueTargetsIdx, nextPlayer, 0.0, rowGlobal);
  fillValueTDTargets(whiteValueTargets, whiteValueTargetsIdx, nextPlayer, 1.0/36.0, rowGlobal+4);
  fillValueTDTargets(whiteValueTargets, whiteValueTargetsIdx, nextPlayer, 1.0/12.0, rowGlobal+8);
  fillValueTDTargets(whiteValueTargets, whiteValueTargetsIdx, nextPlayer, 1.0/4.0, rowGlobal+12);
  fillValueTDTargets(whiteValueTargets, whiteValueTargetsIdx, nextPlayer, 1.0, rowGlobal+16);

  //Fill short-term variance info
  const ValueTargets& thisTargets = whiteValueTargets[whiteValueTargetsIdx];
  rowGlobal[21] = fsq(thisTargets.mctsUtility4 - thisTargets.mctsUtility1);
  rowGlobal[22] = fsq(thisTargets.mctsUtility16 - thisTargets.mctsUtility4);
  rowGlobal[23] = fsq(thisTargets.mctsUtility64 - thisTargets.mctsUtility16);
  rowGlobal[24] = fsq(thisTargets.mctsUtility256 - thisTargets.mctsUtility64);

  //Fill in whether we should use history or not
  bool useHist0 = rand.nextDouble() < 0.98;
  bool useHist1 = useHist0 && rand.nextDouble() < 0.98;
  bool useHist2 = useHist1 && rand.nextDouble() < 0.98;
  bool useHist3 = useHist2 && rand.nextDouble() < 0.98;
  bool useHist4 = useHist3 && rand.nextDouble() < 0.98;
  rowGlobal[28] = useHist0 ? 1.0f : 0.0f;
  rowGlobal[29] = useHist1 ? 1.0f : 0.0f;
  rowGlobal[30] = useHist2 ? 1.0f : 0.0f;
  rowGlobal[31] = useHist3 ? 1.0f : 0.0f;
  rowGlobal[32] = useHist4 ? 1.0f : 0.0f;

  //Fill in hash of game
  Hash128 gameHash = data.gameHash;
  rowGlobal[33] = (float)(gameHash.hash0 & 0x3FFFFF);
  rowGlobal[34] = (float)((gameHash.hash0 >> 22) & 0x3FFFFF);
  rowGlobal[35] = (float)((gameHash.hash0 >> 44) & 0xFFFFF);
  rowGlobal[36] = (float)(gameHash.hash1 & 0x3FFFFF);
  rowGlobal[37] = (float)((gameHash.hash1 >> 22) & 0x3FFFFF);
  rowGlobal[38] = (float)((gameHash.hash1 >> 44) & 0xFFFFF);

  //Some misc metadata
  rowGlobal[39] = turnNumberAfterStart;
  rowGlobal[40] = data.hitTurnLimit ? 1.0f : 0.0f;

  //Metadata about how the game was initialized
  rowGlobal[41] = data.firstTrainingTurn;
  rowGlobal[42] = data.mode;
  rowGlobal[43] = data.modeMeta1;
  rowGlobal[44] = data.modeMeta2;
  rowGlobal[45] = isSidePosition ? 1.0f : 0.0f;

  assert(46 == GLOBAL_TARGET_NUM_CHANNELS);

  int8_t* rowOwnership = valueTargetsNCHW.data + curRows * VALUE_SPATIAL_TARGET_NUM_CHANNELS * posArea;
  if(finalWhiteOwnership == NULL || (data.endHist.isGameFinished && data.endHist.isNoResult)) {
    rowGlobal[26] = 0.0f;
    rowGlobal[20] = 0.0f;
    for(int i = 0; i<posArea; i++)
      rowOwnership[i] = 0.0f;
  }
  else {
    rowGlobal[26] = 1.0f;
    //Fill score info
    const ValueTargets& lastTargets = whiteValueTargets[whiteValueTargets.size()-1];
    rowGlobal[20] = nextPlayer == P_WHITE ? lastTargets.score : -lastTargets.score;

    //Fill ownership info
    for(int i = 0; i<posArea; i++) {
      assert(data.finalWhiteOwnership[i] == 0 || data.finalWhiteOwnership[i] == 1 || data.finalWhiteOwnership[i] == -1);
      //Training rows need things from the perspective of the player to move, so we flip as appropriate.
      rowOwnership[i] = (nextPlayer == P_WHITE ? data.finalWhiteOwnership[i] : -data.finalWhiteOwnership[i]);
    }
  }

  curRows++;
}

void TrainingWriteBuffers::writeToZipFile(const string& fileName) {
  ZipFile zipFile(fileName);

  uint64_t numBytes;

  numBytes = binaryInputNCHWPacked.prepareHeaderWithNumRows(curRows);
  zipFile.writeBuffer("binaryInputNCHWPacked", binaryInputNCHWPacked.dataIncludingHeader, numBytes);

  numBytes = globalInputNC.prepareHeaderWithNumRows(curRows);
  zipFile.writeBuffer("globalInputNC", globalInputNC.dataIncludingHeader, numBytes);

  numBytes = policyTargetsNCMove.prepareHeaderWithNumRows(curRows);
  zipFile.writeBuffer("policyTargetsNCMove", policyTargetsNCMove.dataIncludingHeader, numBytes);

  numBytes = globalTargetsNC.prepareHeaderWithNumRows(curRows);
  zipFile.writeBuffer("globalTargetsNC", globalTargetsNC.dataIncludingHeader, numBytes);

  numBytes = valueTargetsNCHW.prepareHeaderWithNumRows(curRows);
  zipFile.writeBuffer("valueTargetsNCHW", valueTargetsNCHW.dataIncludingHeader, numBytes);

  zipFile.close();
}



//-------------------------------------------------------------------------------------


TrainingDataWriter::TrainingDataWriter(const string& outDir, int iVersion, int maxRowsPerFile, int posLen)
  :outputDir(outDir),inputsVersion(iVersion),rand(),writeBuffers(NULL)
{
  int numBinaryChannels;
  int numGlobalChannels;
  //Note that this inputsVersion is for data writing, it might be different than the inputsVersion used
  //to feed into a model during selfplay
  if(inputsVersion < 3 || inputsVersion > 3)
    throw StringError("TrainingDataWriter: Unsupported inputs version: " + Global::intToString(inputsVersion));
  else if(inputsVersion == 3) {
    numBinaryChannels = NNInputs::NUM_FEATURES_BIN_V3;
    numGlobalChannels = NNInputs::NUM_FEATURES_GLOBAL_V3;
  }

  writeBuffers = new TrainingWriteBuffers(inputsVersion, maxRowsPerFile, numBinaryChannels, numGlobalChannels, posLen);
}

TrainingDataWriter::~TrainingDataWriter()
{
  delete writeBuffers;
}

void TrainingDataWriter::writeAndClearIfFull() {
  if(writeBuffers->curRows >= writeBuffers->maxRows) {
    string filename = outputDir + "/" + Global::uint64ToHexString(rand.nextUInt64()) + ".npz";
    string tmpFilename = filename + ".tmp";
    writeBuffers->writeToZipFile(tmpFilename);
    writeBuffers->clear();
    std::rename(tmpFilename.c_str(),filename.c_str());
  }
}

void TrainingDataWriter::close() {
  if(writeBuffers->curRows > 0) {
    string filename = outputDir + "/" + Global::uint64ToHexString(rand.nextUInt64()) + ".npz";
    string tmpFilename = filename + ".tmp";
    writeBuffers->writeToZipFile(tmpFilename);
    writeBuffers->clear();
    std::rename(tmpFilename.c_str(),filename.c_str());
  }
}

void TrainingDataWriter::writeGame(const FinishedGameData& data) {
  int numMoves = data.endHist.moveHistory.size() - data.startHist.moveHistory.size();
  assert(numMoves >= 0);
  assert(data.policyTargetsByTurn.size() == numMoves);
  assert(data.whiteValueTargetsByTurn.size() == numMoves+1);

  Board board(data.startBoard);
  BoardHistory hist(data.startHist);
  Player nextPlayer = data.startPla;

  //Some sanity checks
  {
    const ValueTargets& lastTargets = data.whiteValueTargetsByTurn[data.whiteValueTargetsByTurn.size()-1];
    if(!data.endHist.isGameFinished)
      assert(data.hitTurnLimit);
    else if(hist.isNoResult)
      assert(lastTargets.win == 0.0f && lastTargets.loss == 0.0f && lastTargets.noResult == 1.0f);
    else if(data.endHist.winner == P_BLACK)
      assert(lastTargets.win == 0.0f && lastTargets.loss == 1.0f && lastTargets.noResult == 0.0f);
    else if(data.endHist.winner == P_WHITE)
      assert(lastTargets.win == 1.0f && lastTargets.loss == 0.0f && lastTargets.noResult == 0.0f);
    else
      assert(lastTargets.noResult == 0.0f);

    assert(data.finalWhiteOwnership != NULL);
  }

  //Write main game rows
  for(int turnNumberAfterStart = 0; turnNumberAfterStart<numMoves; turnNumberAfterStart++) {
    const vector<PolicyTargetMove>* policyTarget0 = data.policyTargetsByTurn[turnNumberAfterStart];
    bool isSidePosition = false;
    writeBuffers->addRow(
      board,hist,nextPlayer,
      turnNumberAfterStart,
      policyTarget0,
      data.whiteValueTargetsByTurn,
      turnNumberAfterStart,
      data.finalWhiteOwnership,
      isSidePosition,
      data,
      rand
    );
    writeAndClearIfFull();

    Move move = data.endHist.moveHistory[turnNumberAfterStart + data.startHist.moveHistory.size()];
    assert(move.pla == nextPlayer);
    assert(hist.isLegal(board,move.loc,move.pla));
    hist.makeBoardMoveAssumeLegal(board, move.loc, move.pla, NULL);
    nextPlayer = getOpp(nextPlayer);
  }

  //Write side rows
  vector<ValueTargets> whiteValueTargetsBuf(1);
  for(int i = 0; i<data.sidePositions.size(); i++) {
    SidePosition* sp = data.sidePositions[i];
    int turnNumberAfterStart = sp->hist.moveHistory.size() - data.startHist.moveHistory.size();
    assert(turnNumberAfterStart > 0);
    whiteValueTargetsBuf[0] = sp->whiteValueTargets;
    bool isSidePosition = true;
    writeBuffers->addRow(
      sp->board,sp->hist,sp->pla,
      turnNumberAfterStart,
      &(sp->policyTarget),
      whiteValueTargetsBuf,
      0,
      NULL,
      isSidePosition,
      data,
      rand
    );

  }

}


