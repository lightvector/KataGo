
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

//Don't forget to update everything else in the header file and the code below too if changing any of these
static const int POLICY_TARGET_NUM_CHANNELS = 3;
static const int FLOAT_TARGET_NUM_CHANNELS = 43;
static const int VALUE_SPATIAL_TARGET_NUM_CHANNELS = 1;

TrainingWriteBuffers::TrainingWriteBuffers(int iVersion, int maxRws, int numBChannels, int numFChannels, int pLen)
  :inputsVersion(iVersion),
   maxRows(maxRws),
   numBinaryChannels(numBChannels),
   numFloatChannels(numFChannels),
   posLen(pLen),
   packedBoardArea((pLen*pLen + 7)/8),
   curRows(0),
   binaryInputNCHWUnpacked(NULL),
   binaryInputNCHWPacked({maxRws, numBChannels, packedBoardArea}),
   floatInputNC({maxRws, numFChannels}),
   policyTargetsNCMove({maxRws, POLICY_TARGET_NUM_CHANNELS, NNPos::getPolicySize(pLen)}),
   floatTargetsNC({maxRws, FLOAT_TARGET_NUM_CHANNELS}),
   valueTargetsNCHW({maxRws, VALUE_SPATIAL_TARGET_NUM_CHANNELS, pLen, pLen})
{
  binaryInputNCHWUnpacked = new bool[numBChannels * pLen * pLen];
}

TrainingWriteBuffers::~TrainingWriteBuffers()
{
  delete[] binaryInputNCHWUnpacked;
}

void TrainingWriteBuffers::clear() {
  curRows = 0;
}

//Copy bools into bits, packing 8 bools to a byte, big-endian-style within each byte.
static void packBits(const bool* bools, int len, uint8_t* bits) {
  for(int i = 0; i < len; i += 8) {
    if(i + 8 <= len) {
      bits[i >> 3] =
        ((uint8_t)bools[i + 0] << 7) |
        ((uint8_t)bools[i + 1] << 6) |
        ((uint8_t)bools[i + 2] << 5) |
        ((uint8_t)bools[i + 3] << 4) |
        ((uint8_t)bools[i + 4] << 3) |
        ((uint8_t)bools[i + 5] << 2) |
        ((uint8_t)bools[i + 6] << 1) |
        ((uint8_t)bools[i + 7] << 0);
    }
    else {
      bits[i >> 3] = 0;
      for(int di = 0; i + di < len; di++) {
        bits[i >> 3] |= ((uint8_t)bools[i + di] << (7-di));
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

static void fillValueTDTargets(const vector<ValueTargets>& whiteValueTargetsByTurn, int turnNumber, Player nextPlayer, float nowFactor, float* buf) {
  double winValue = 0.0;
  double lossValue = 0.0;
  double noResultValue = 0.0;
  double scoreValue = 0.0;

  double weightLeft = 1.0;
  for(int i = turnNumber; i<whiteValueTargetsByTurn.size(); i++) {
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
  const Board& board, const BoardHistory& hist, Player nextPlayer, double drawEquivalentWinsForWhite,
  int turnNumber,
  const vector<PolicyTargetMove>* policyTarget0, //can be null
  const vector<PolicyTargetMove>* policyTarget1, //can be null
  const vector<PolicyTargetMove>* policyTarget2, //can be null
  const vector<ValueTargets>& whiteValueTargetsByTurn,
  const int8_t* finalOwnership,
  Hash128 gameHash,
  int mode, int modeMeta1, int modeMeta2,
  Rand& rand
) {
  if(inputsVersion < 3 || inputsVersion > 3)
    throw StringError("Training write buffers: Does not support input version: " + Global::intToString(inputsVersion));

  int posArea = posLen*posLen;

  {
    bool inputsUseNHWC = false;
    bool* rowBin = binaryInputNCHWUnpacked;
    float* rowFloat = floatInputNC.data + curRows * numFloatChannels;
    if(inputsVersion == 3) {
      assert(NNInputs::NUM_FEATURES_BIN_V3 == numBinaryChannels);
      assert(NNInputs::NUM_FEATURES_FLOAT_V3 == numFloatChannels);
      NNInputs::fillRowV3(board, hist, nextPlayer, drawEquivalentWinsForWhite, posLen, inputsUseNHWC, rowBin, rowFloat);
    }
    else
      assert(false);

    //Pack bools bitwise into uint8_t
    uint8_t* rowBinPacked = binaryInputNCHWPacked.data + curRows * numBinaryChannels * packedBoardArea;
    for(int c = 0; c<numBinaryChannels; c++)
      packBits(rowBin + c * posArea, posArea, rowBinPacked + c * packedBoardArea);
  }

  //Vector for global targets and metadata
  float* rowFloat = floatTargetsNC.data + curRows * FLOAT_TARGET_NUM_CHANNELS;

  //Fill policy
  int policySize = NNPos::getPolicySize(posLen);
  int16_t* rowPolicy = policyTargetsNCMove.data + curRows * POLICY_TARGET_NUM_CHANNELS * policySize;

  if(policyTarget0 != NULL) {
    fillPolicyTarget(*policyTarget0, policySize, posLen, board.x_size, rowPolicy + 0 * policySize);
    rowFloat[25] = 1.0f;
  }
  else {
    zeroPolicyTarget(policySize, rowPolicy + 0 * policySize);
    rowFloat[25] = 0.0f;
  }

  if(policyTarget1 != NULL) {
    fillPolicyTarget(*policyTarget1, policySize, posLen, board.x_size, rowPolicy + 1 * policySize);
    rowFloat[26] = 1.0f;
  }
  else {
    zeroPolicyTarget(policySize, rowPolicy + 1 * policySize);
    rowFloat[26] = 0.0f;
  }

  if(policyTarget2 != NULL) {
    fillPolicyTarget(*policyTarget2, policySize, posLen, board.x_size, rowPolicy + 2 * policySize);
    rowFloat[27] = 1.0f;
  }
  else {
    zeroPolicyTarget(policySize, rowPolicy + 2 * policySize);
    rowFloat[27] = 0.0f;
  }

  //Fill td-like value targets
  assert(turnNumber >= 0 && turnNumber < whiteValueTargetsByTurn.size());
  fillValueTDTargets(whiteValueTargetsByTurn, turnNumber, nextPlayer, 0.0, rowFloat);
  fillValueTDTargets(whiteValueTargetsByTurn, turnNumber, nextPlayer, 1.0/36.0, rowFloat+4);
  fillValueTDTargets(whiteValueTargetsByTurn, turnNumber, nextPlayer, 1.0/12.0, rowFloat+8);
  fillValueTDTargets(whiteValueTargetsByTurn, turnNumber, nextPlayer, 1.0/4.0, rowFloat+12);
  fillValueTDTargets(whiteValueTargetsByTurn, turnNumber, nextPlayer, 1.0, rowFloat+16);

  //Fill score info
  const ValueTargets& lastTargets = whiteValueTargetsByTurn[whiteValueTargetsByTurn.size()-1];
  rowFloat[20] = nextPlayer == P_WHITE ? lastTargets.score : -lastTargets.score;

  //Fill short-term variance info
  const ValueTargets& thisTargets = whiteValueTargetsByTurn[turnNumber];
  rowFloat[21] = fsq(thisTargets.mctsUtility4 - thisTargets.mctsUtility1);
  rowFloat[22] = fsq(thisTargets.mctsUtility16 - thisTargets.mctsUtility4);
  rowFloat[23] = fsq(thisTargets.mctsUtility64 - thisTargets.mctsUtility16);
  rowFloat[24] = fsq(thisTargets.mctsUtility256 - thisTargets.mctsUtility64);

  //Fill in whether we should use history or not
  rowFloat[28] = rand.nextDouble() < 0.95;
  rowFloat[29] = rowFloat[28] && rand.nextDouble() < 0.95;
  rowFloat[30] = rowFloat[29] && rand.nextDouble() < 0.95;
  rowFloat[31] = rowFloat[30] && rand.nextDouble() < 0.95;
  rowFloat[32] = rowFloat[31] && rand.nextDouble() < 0.95;

  //Fill in hash of game
  rowFloat[33] = (float)(gameHash.hash0 & 0x3FFFFF);
  rowFloat[34] = (float)((gameHash.hash0 >> 22) & 0x3FFFFF);
  rowFloat[35] = (float)((gameHash.hash0 >> 44) & 0xFFFFF);
  rowFloat[36] = (float)(gameHash.hash1 & 0x3FFFFF);
  rowFloat[37] = (float)((gameHash.hash1 >> 22) & 0x3FFFFF);
  rowFloat[38] = (float)((gameHash.hash1 >> 44) & 0xFFFFF);

  rowFloat[39] = turnNumber;

  //Metadata about how the game was initialized
  rowFloat[40] = mode;
  rowFloat[41] = modeMeta1;
  rowFloat[42] = modeMeta2;

  assert(43 == FLOAT_TARGET_NUM_CHANNELS);

  int8_t* rowOwnership = valueTargetsNCHW.data + curRows * VALUE_SPATIAL_TARGET_NUM_CHANNELS * posArea;
  for(int i = 0; i<posArea; i++) {
    assert(finalOwnership[i] == 0 || finalOwnership[i] == 1 || finalOwnership[i] == -1);
    rowOwnership[i] = finalOwnership[i];
  }

  curRows++;
}

void TrainingWriteBuffers::writeToZipFile(const string& fileName) {
  ZipFile zipFile(fileName);

  uint64_t numBytes;

  numBytes = binaryInputNCHWPacked.prepareHeaderWithNumRows(curRows);
  zipFile.writeBuffer("binaryInputNCHWPacked", binaryInputNCHWPacked.dataIncludingHeader, numBytes);

  numBytes = floatInputNC.prepareHeaderWithNumRows(curRows);
  zipFile.writeBuffer("floatInputNC", floatInputNC.dataIncludingHeader, numBytes);

  numBytes = policyTargetsNCMove.prepareHeaderWithNumRows(curRows);
  zipFile.writeBuffer("policyTargetsNCMove", policyTargetsNCMove.dataIncludingHeader, numBytes);

  numBytes = floatTargetsNC.prepareHeaderWithNumRows(curRows);
  zipFile.writeBuffer("floatTargetsNC", floatTargetsNC.dataIncludingHeader, numBytes);

  numBytes = valueTargetsNCHW.prepareHeaderWithNumRows(curRows);
  zipFile.writeBuffer("valueTargetsNCHW", valueTargetsNCHW.dataIncludingHeader, numBytes);

  zipFile.close();
}



FinishedGameData::FinishedGameData(int pLen, double drawEquivForWhite)
  : startBoard(),
    startHist(),
    endHist(),
    startPla(P_BLACK),
    gameHash(),
    moves(),
    policyTargetsByTurn(),
    whiteValueTargetsByTurn(),
    finalOwnership(NULL),
    drawEquivalentWinsForWhite(drawEquivForWhite),
    posLen(pLen),
    mode(0),
    modeMeta1(0),
    modeMeta2(0)
{
  finalOwnership = new int8_t[posLen*posLen];
  for(int i = 0; i<posLen*posLen; i++)
    finalOwnership[i] = 0;
}

FinishedGameData::~FinishedGameData() {
  for(int i = 0; i<policyTargetsByTurn.size(); i++)
    delete policyTargetsByTurn[i];

  delete[] finalOwnership;
}



TrainingDataWriter::TrainingDataWriter(const string& outDir, int iVersion, int maxRowsPerFile, int posLen)
  :outputDir(outDir),inputsVersion(iVersion),rand(),writeBuffers(NULL)
{
  int numBinaryChannels;
  int numFloatChannels;
  if(inputsVersion < 3 || inputsVersion > 3)
    throw StringError("TrainingDataWriter: Unsupported inputs version: " + Global::intToString(inputsVersion));
  else if(inputsVersion == 3) {
    numBinaryChannels = NNInputs::NUM_FEATURES_BIN_V3;
    numFloatChannels = NNInputs::NUM_FEATURES_FLOAT_V3;
  }

  writeBuffers = new TrainingWriteBuffers(inputsVersion, maxRowsPerFile, numBinaryChannels, numFloatChannels, posLen);
}

TrainingDataWriter::~TrainingDataWriter()
{
  delete writeBuffers;
}

void TrainingDataWriter::writeAndClearIfFull() {
  if(writeBuffers->curRows >= writeBuffers->maxRows) {
    writeBuffers->writeToZipFile(outputDir + "/" + Global::uint64ToHexString(rand.nextUInt64()) + ".npz");
    writeBuffers->clear();
  }
}

void TrainingDataWriter::close() {
  if(writeBuffers->curRows > 0) {
    writeBuffers->writeToZipFile(outputDir + "/" + Global::uint64ToHexString(rand.nextUInt64()) + ".npz");
    writeBuffers->clear();
  }
}

void TrainingDataWriter::writeGame(const FinishedGameData& data) {
  int numMoves = data.moves.size();
  assert(data.policyTargetsByTurn.size() == numMoves);
  assert(data.whiteValueTargetsByTurn.size() == numMoves+1);

  Board board(data.startBoard);
  BoardHistory hist(data.startHist);
  Player nextPlayer = data.startPla;

  for(int turnNumber = 0; turnNumber<numMoves; turnNumber++) {
    const vector<PolicyTargetMove>* policyTarget0 = data.policyTargetsByTurn[turnNumber];
    const vector<PolicyTargetMove>* policyTarget1 =
      (turnNumber >= data.policyTargetsByTurn.size() - 1) ? NULL : data.policyTargetsByTurn[turnNumber+1];
    const vector<PolicyTargetMove>* policyTarget2 =
      (turnNumber >= data.policyTargetsByTurn.size() - 2) ? NULL : data.policyTargetsByTurn[turnNumber+2];

    writeBuffers->addRow(
      board,hist,nextPlayer,data.drawEquivalentWinsForWhite,
      turnNumber,
      policyTarget0,
      policyTarget1,
      policyTarget2,
      data.whiteValueTargetsByTurn,
      data.finalOwnership,
      data.gameHash,
      data.mode,data.modeMeta1,data.modeMeta2,
      rand
    );
    writeAndClearIfFull();

    Move move = data.moves[turnNumber];
    assert(move.pla == nextPlayer);
    hist.makeBoardMoveAssumeLegal(board, move.loc, move.pla, NULL);
    nextPlayer = getOpp(nextPlayer);
  }

}


