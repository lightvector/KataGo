
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
static const int VALUE_TARGET_NUM_CHANNELS = 25;
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
   policyTargetsNCPos({maxRws, POLICY_TARGET_NUM_CHANNELS, NNPos::getPolicySize(pLen)}),
   policyTargetWeightsNC({maxRws, POLICY_TARGET_NUM_CHANNELS}),
   valueTargetsNC({maxRws, VALUE_TARGET_NUM_CHANNELS}),
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
      weightNow = weightLeft * nowFactor;
      weightLeft *= (1.0 - nowFactor);
    }
    else {
      weightNow = weightLeft;
      weightLeft = 0.0;
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
  const int16_t* finalOwnership
) {
  if(inputsVersion < 3 || inputsVersion > 3)
    throw StringError("Training write buffers: Does not support input version: " + Global::intToString(inputsVersion));

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
  int posArea = posLen*posLen;
  uint8_t* rowBinPacked = binaryInputNCHWPacked.data + curRows * numBinaryChannels * packedBoardArea;
  for(int c = 0; c<numBinaryChannels; c++)
    packBits(rowBin + c * posArea, posArea, rowBinPacked + c * packedBoardArea);

  //Fill policy
  int policySize = NNPos::getPolicySize(posLen);
  int16_t* rowPolicy = policyTargetsNCPos.data + curRows * POLICY_TARGET_NUM_CHANNELS * policySize;
  float* rowPolicyWeight = policyTargetWeightsNC.data + curRows * POLICY_TARGET_NUM_CHANNELS + 0;

  if(policyTarget0 != NULL) {
    fillPolicyTarget(*policyTarget0, policySize, posLen, board.x_size, rowPolicy + 0 * policySize);
    rowPolicyWeight[0] = 1.0f;
  }
  else {
    zeroPolicyTarget(policySize, rowPolicy + 0 * policySize);
    rowPolicyWeight[0] = 0.0f;
  }

  if(policyTarget1 != NULL) {
    fillPolicyTarget(*policyTarget1, policySize, posLen, board.x_size, rowPolicy + 1 * policySize);
    rowPolicyWeight[1] = 1.0f;
  }
  else {
    zeroPolicyTarget(policySize, rowPolicy + 1 * policySize);
    rowPolicyWeight[1] = 0.0f;
  }

  if(policyTarget2 != NULL) {
    fillPolicyTarget(*policyTarget2, policySize, posLen, board.x_size, rowPolicy + 2 * policySize);
    rowPolicyWeight[2] = 1.0f;
  }
  else {
    zeroPolicyTarget(policySize, rowPolicy + 2 * policySize);
    rowPolicyWeight[2] = 0.0f;
  }

  //Fill value
  assert(turnNumber >= 0 && turnNumber < whiteValueTargetsByTurn.size());
  float* rowValue = valueTargetsNC.data + curRows * VALUE_TARGET_NUM_CHANNELS;

  //td-like targets
  fillValueTDTargets(whiteValueTargetsByTurn, turnNumber, nextPlayer, 0.0, rowValue);
  fillValueTDTargets(whiteValueTargetsByTurn, turnNumber, nextPlayer, 1.0/36.0, rowValue+4);
  fillValueTDTargets(whiteValueTargetsByTurn, turnNumber, nextPlayer, 1.0/12.0, rowValue+8);
  fillValueTDTargets(whiteValueTargetsByTurn, turnNumber, nextPlayer, 1.0/4.0, rowValue+12);
  fillValueTDTargets(whiteValueTargetsByTurn, turnNumber, nextPlayer, 1.0, rowValue+16);

  const ValueTargets& lastTargets = whiteValueTargetsByTurn[whiteValueTargetsByTurn.size()-1];
  rowValue[20] = nextPlayer == P_WHITE ? lastTargets.score : -lastTargets.score;

  const ValueTargets& thisTargets = whiteValueTargetsByTurn[turnNumber];
  rowValue[21] = fsq(thisTargets.mctsUtility4 - thisTargets.mctsUtility1);
  rowValue[22] = fsq(thisTargets.mctsUtility16 - thisTargets.mctsUtility4);
  rowValue[23] = fsq(thisTargets.mctsUtility64 - thisTargets.mctsUtility16);
  rowValue[24] = fsq(thisTargets.mctsUtility256 - thisTargets.mctsUtility64);
  assert(25 == VALUE_TARGET_NUM_CHANNELS);

  int16_t* rowOwnership = valueTargetsNCHW.data + curRows * VALUE_SPATIAL_TARGET_NUM_CHANNELS * posArea;
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

  numBytes = policyTargetsNCPos.prepareHeaderWithNumRows(curRows);
  zipFile.writeBuffer("policyTargetsNCPos", policyTargetsNCPos.dataIncludingHeader, numBytes);

  numBytes = policyTargetWeightsNC.prepareHeaderWithNumRows(curRows);
  zipFile.writeBuffer("policyTargetWeightsNC", policyTargetWeightsNC.dataIncludingHeader, numBytes);

  numBytes = valueTargetsNC.prepareHeaderWithNumRows(curRows);
  zipFile.writeBuffer("valueTargetsNC", valueTargetsNC.dataIncludingHeader, numBytes);

  numBytes = valueTargetsNCHW.prepareHeaderWithNumRows(curRows);
  zipFile.writeBuffer("valueTargetsNCHW", valueTargetsNCHW.dataIncludingHeader, numBytes);

  zipFile.close();
}



FinishedGameData::FinishedGameData(int pLen, double drawEquivForWhite)
  : startBoard(),
    startHist(),
    startPla(P_BLACK),
    moves(),
    policyTargetsByTurn(),
    whiteValueTargetsByTurn(),
    finalOwnership(NULL),
    drawEquivalentWinsForWhite(drawEquivForWhite),
    posLen(pLen)
{
  finalOwnership = new int16_t[posLen*posLen];
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
      data.finalOwnership
    );
    writeAndClearIfFull();

    Move move = data.moves[turnNumber];
    assert(move.pla == nextPlayer);
    hist.makeBoardMoveAssumeLegal(board, move.loc, move.pla, NULL);
    nextPlayer = getOpp(nextPlayer);
  }

}


