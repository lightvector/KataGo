#include "../dataio/trainingwrite.h"

#include "../core/fileutils.h"
#include "../neuralnet/modelversion.h"

using namespace std;

ValueTargets::ValueTargets()
  :win(0),
   loss(0),
   noResult(0),
   score(0),
   hasLead(false),
   lead(0)
{}
ValueTargets::~ValueTargets()
{}

//-------------------------------------------------------------------------------------

SidePosition::SidePosition()
  :board(),
   hist(),
   pla(P_BLACK),
   unreducedNumVisits(),
   policyTarget(),
   policySurprise(),
   policyEntropy(),
   searchEntropy(),
   whiteValueTargets(),
   whiteQValueTargets(),
   targetWeight(),
   targetWeightUnrounded(),
   numNeuralNetChangesSoFar(),
   playoutDoublingAdvantagePla(C_EMPTY),
   playoutDoublingAdvantage(0.0)
{}

SidePosition::SidePosition(const Board& b, const BoardHistory& h, Player p, int numNNChangesSoFar)
  :board(b),
   hist(h),
   pla(p),
   unreducedNumVisits(),
   policyTarget(),
   policySurprise(),
   policyEntropy(),
   searchEntropy(),
   whiteValueTargets(),
   whiteQValueTargets(),
   targetWeight(1.0f),
   targetWeightUnrounded(1.0f),
   numNeuralNetChangesSoFar(numNNChangesSoFar),
   playoutDoublingAdvantagePla(C_EMPTY),
   playoutDoublingAdvantage(0.0)
{}

SidePosition::~SidePosition()
{}

//-------------------------------------------------------------------------------------

FinishedGameData::FinishedGameData()
  :bName(),
   wName(),
   bIdx(0),
   wIdx(0),

   startBoard(),
   startHist(),
   endHist(),
   startPla(P_BLACK),
   gameHash(),

   drawEquivalentWinsForWhite(0.0),
   playoutDoublingAdvantagePla(P_BLACK),
   playoutDoublingAdvantage(0.0),

   hitTurnLimit(false),

   numExtraBlack(0),
   mode(0),
   beganInEncorePhase(0),
   usedInitialPosition(0),

   hasFullData(false),
   targetWeightByTurn(),
   targetWeightByTurnUnrounded(),
   policyTargetsByTurn(),
   whiteValueTargetsByTurn(),
   whiteQValueTargetsByTurn(),
   nnRawStatsByTurn(),
   finalFullArea(NULL),
   finalOwnership(NULL),
   finalSekiAreas(NULL),
   finalWhiteScoring(NULL),

   trainingWeight(1.0),

   sidePositions(),
   changedNeuralNets(),
   bTimeUsed(0.0),
   wTimeUsed(0.0),
   bMoveCount(0),
   wMoveCount(0)
{
}

FinishedGameData::~FinishedGameData() {
  for(size_t i = 0; i<policyTargetsByTurn.size(); i++)
    delete policyTargetsByTurn[i].policyTargets;

  if(finalFullArea != NULL)
    delete[] finalFullArea;
  if(finalOwnership != NULL)
    delete[] finalOwnership;
  if(finalSekiAreas != NULL)
    delete[] finalSekiAreas;
  if(finalWhiteScoring != NULL)
    delete[] finalWhiteScoring;

  for(size_t i = 0; i<sidePositions.size(); i++)
    delete sidePositions[i];

  for(size_t i = 0; i<changedNeuralNets.size(); i++)
    delete changedNeuralNets[i];
}

void FinishedGameData::printDebug(ostream& out) const {
  out << "bName " << bName << endl;
  out << "wName " << wName << endl;
  out << "bIdx " << bIdx << endl;
  out << "wIdx " << wIdx << endl;
  out << "startPla " << PlayerIO::colorToChar(startPla) << endl;
  out << "start" << endl;
  startHist.printDebugInfo(out,startBoard);
  out << "end" << endl;
  endHist.printDebugInfo(out,endHist.getRecentBoard(0));
  out << "gameHash " << gameHash << endl;
  out << "hitTurnLimit " << hitTurnLimit << endl;
  out << "numExtraBlack " << numExtraBlack << endl;
  out << "mode " << mode << endl;
  out << "beganInEncorePhase " << beganInEncorePhase << endl;
  out << "usedInitialPosition " << usedInitialPosition << endl;
  out << "hasFullData " << hasFullData << endl;
  for(int i = 0; i<targetWeightByTurn.size(); i++)
    out << "targetWeightByTurn " << i << " " << targetWeightByTurn[i] << " " << "unrounded" << " " << targetWeightByTurnUnrounded[i] << endl;
  for(int i = 0; i<policyTargetsByTurn.size(); i++) {
    out << "policyTargetsByTurn " << i << " ";
    out << "unreducedNumVisits " << policyTargetsByTurn[i].unreducedNumVisits << " ";
    if(policyTargetsByTurn[i].policyTargets != NULL) {
      const vector<PolicyTargetMove>& target = *(policyTargetsByTurn[i].policyTargets);
      for(int j = 0; j<target.size(); j++)
        out << Location::toString(target[j].loc,startBoard) << " " << target[j].policyTarget << " ";
    }
    out << endl;
  }
  for (int i = 0; i < policySurpriseByTurn.size(); i++)
    out << "policySurpriseByTurn " << i << " " << policySurpriseByTurn[i] << endl;
  for (int i = 0; i < policyEntropyByTurn.size(); i++)
    out << "policyEntropyByTurn " << i << " " << policyEntropyByTurn[i] << endl;
  for (int i = 0; i < searchEntropyByTurn.size(); i++)
    out << "searchEntropyByTurn " << i << " " << searchEntropyByTurn[i] << endl;

  for(int i = 0; i<whiteValueTargetsByTurn.size(); i++) {
    out << "whiteValueTargetsByTurn " << i << " ";
    out << whiteValueTargetsByTurn[i].win << " ";
    out << whiteValueTargetsByTurn[i].loss << " ";
    out << whiteValueTargetsByTurn[i].noResult << " ";
    out << whiteValueTargetsByTurn[i].score << " ";
    if(whiteValueTargetsByTurn[i].hasLead)
      out << whiteValueTargetsByTurn[i].lead << " ";
    else
      out << "-" << " ";
    out << endl;
  }

  for(int i = 0; i<whiteQValueTargetsByTurn.size(); i++) {
    out << "whiteQValueTargetsByTurn " << i << " ";
    const vector<QValueTargetMove>& target = whiteQValueTargetsByTurn[i].targets;
    for(int j = 0; j<target.size(); j++)
      out << Location::toString(target[j].loc,startBoard) << " " << target[j].winLoss << " " << target[j].score << " " << target[j].visits << " ";
    out << endl;
  }

  for(int i = 0; i<nnRawStatsByTurn.size(); i++) {
    out << "Raw Stats " << nnRawStatsByTurn[i].whiteWinLoss << " " << nnRawStatsByTurn[i].whiteScoreMean << " " << nnRawStatsByTurn[i].policyEntropy << endl;
  }
  if(finalFullArea != NULL) {
    for(int y = 0; y<startBoard.y_size; y++) {
      for(int x = 0; x<startBoard.x_size; x++) {
        Loc loc = Location::getLoc(x,y,startBoard.x_size);
        out << PlayerIO::colorToChar(finalFullArea[loc]);
      }
      out << endl;
    }
  }
  if(finalOwnership != NULL) {
    for(int y = 0; y<startBoard.y_size; y++) {
      for(int x = 0; x<startBoard.x_size; x++) {
        Loc loc = Location::getLoc(x,y,startBoard.x_size);
        out << PlayerIO::colorToChar(finalOwnership[loc]);
      }
      out << endl;
    }
  }
  if(finalSekiAreas != NULL) {
    for(int y = 0; y<startBoard.y_size; y++) {
      for(int x = 0; x<startBoard.x_size; x++) {
        Loc loc = Location::getLoc(x,y,startBoard.x_size);
        out << (int)finalSekiAreas[loc];
      }
      out << endl;
    }
  }
  if(finalWhiteScoring != NULL) {
    for(int y = 0; y<startBoard.y_size; y++) {
      for(int x = 0; x<startBoard.x_size; x++) {
        Loc loc = Location::getLoc(x,y,startBoard.x_size);
        out << Global::strprintf(" %.3f",finalWhiteScoring[loc]);
      }
      out << endl;
    }
  }
  out << "trainingWeight " << trainingWeight << endl;
  for(int i = 0; i<sidePositions.size(); i++) {
    SidePosition* sp = sidePositions[i];
    out << "Side position " << i << endl;
    out << "targetWeight " << sp->targetWeight << " " << "unrounded" << " " << sp->targetWeightUnrounded << endl;
    sp->hist.printDebugInfo(out,sp->board);
    {
      out << "Side position policyTarget ";
      out << "unreducedNumVisits " << sp->unreducedNumVisits << " ";
      const vector<PolicyTargetMove>& target = sp->policyTarget;
      for(int j = 0; j<target.size(); j++)
        out << Location::toString(target[j].loc,startBoard) << " " << target[j].policyTarget << " ";
      out << endl;
    }
    out << "Side position whiteValueTargets ";
    out << sp->whiteValueTargets.win << " ";
    out << sp->whiteValueTargets.loss << " ";
    out << sp->whiteValueTargets.noResult << " ";
    out << sp->whiteValueTargets.score << " ";
    if(sp->whiteValueTargets.hasLead)
      out << sp->whiteValueTargets.lead << " ";
    else
      out << "-" << " ";
    out << endl;
    {
      out << "Side position whiteQValueTargets ";
      const vector<QValueTargetMove>& target = sp->whiteQValueTargets.targets;
      for(int j = 0; j<target.size(); j++)
        out << Location::toString(target[j].loc,startBoard) << " " << target[j].winLoss << " " << target[j].score << " " << target[j].visits << " ";
      out << endl;
    }
  }
}

//-------------------------------------------------------------------------------------


//Don't forget to update everything else in the header file and the code below too if changing any of these
//And update the python code
static const int POLICY_TARGET_NUM_CHANNELS = 2;
static const int GLOBAL_TARGET_NUM_CHANNELS = 64;
static const int VALUE_SPATIAL_TARGET_NUM_CHANNELS = 5;
static const int QVALUE_SPATIAL_TARGET_NUM_CHANNELS = 3;

TrainingWriteBuffers::TrainingWriteBuffers(int iVersion, int maxRws, int numBChannels, int numFChannels, int xLen, int yLen, bool includeMetadata)
  :inputsVersion(iVersion),
   maxRows(maxRws),
   numBinaryChannels(numBChannels),
   numGlobalChannels(numFChannels),
   dataXLen(xLen),
   dataYLen(yLen),
   packedBoardArea((xLen*yLen + 7)/8),
   hasMetadataInput(includeMetadata),
   curRows(0),
   binaryInputNCHWUnpacked(NULL),
   binaryInputNCHWPacked({maxRws, numBChannels, packedBoardArea}),
   globalInputNC({maxRws, numFChannels}),
   policyTargetsNCMove({maxRws, POLICY_TARGET_NUM_CHANNELS, NNPos::getPolicySize(xLen,yLen)}),
   globalTargetsNC({maxRws, GLOBAL_TARGET_NUM_CHANNELS}),
   scoreDistrN({maxRws, xLen*yLen*2+NNPos::EXTRA_SCORE_DISTR_RADIUS*2}),
   valueTargetsNCHW({maxRws, VALUE_SPATIAL_TARGET_NUM_CHANNELS, yLen, xLen}),
   qValueTargetsNCMove({maxRws, QVALUE_SPATIAL_TARGET_NUM_CHANNELS, NNPos::getPolicySize(xLen,yLen)}),
   metadataInputNC({(includeMetadata ? maxRws : 1), SGFMetadata::METADATA_INPUT_NUM_CHANNELS})
{
  binaryInputNCHWUnpacked = new float[numBChannels * xLen * yLen];
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

static void uniformPolicyTarget(int policySize, int16_t* target) {
  for(int pos = 0; pos<policySize; pos++)
    target[pos] = 1;
}

//Copy playouts into target, expanding out the sparse representation into a full plane.
static void fillPolicyTarget(const vector<PolicyTargetMove>& policyTargetMoves, int policySize, int dataXLen, int dataYLen, int boardXSize, int16_t* target) {
  zeroPolicyTarget(policySize,target);
  size_t size = policyTargetMoves.size();
  for(size_t i = 0; i<size; i++) {
    const PolicyTargetMove& move = policyTargetMoves[i];
    int pos = NNPos::locToPos(move.loc, boardXSize, dataXLen, dataYLen);
    assert(pos >= 0 && pos < policySize);
    target[pos] = move.policyTarget;
  }
}

//Clamps a value to integer in [-120,120] to pack down to 8 bits.
//Randomizes to make sure the expectation is exactly correct.
static int8_t clampToRadius120(float x, Rand& rand) {
  //We need to pack this down to 8 bits, so map into [-120,120].
  //Randomize to ensure the expectation is exactly correct.
  int low = (int)floor(x);
  int high = low+1;
  if(low < -120) return -120;
  if(high > 120) return 120;

  float lambda = (float)(x-low);
  if(lambda == 0.0f) return (int8_t)low;
  else return (int8_t)(rand.nextBool(lambda) ? high : low);
}
static int16_t clampToRadius32000(float x, Rand& rand) {
  //We need to pack this down to 16 bits, so clamp into an integer [-32000,32000].
  //Randomize to ensure the expectation is exactly correct.
  int low = (int)floor(x);
  int high = low+1;
  if(low < -32000) return -32000;
  if(high > 32000) return 32000;

  float lambda = (float)(x-low);
  if(lambda == 0.0f) return (int16_t)low;
  else return (int16_t)(rand.nextBool(lambda) ? high : low);
}

static void fillQValueTarget(const vector<QValueTargetMove>& whiteQValueTargets, Player nextPlayer, int policySize, int dataXLen, int dataYLen, int boardXSize, int16_t* cPosTarget, Rand& rand) {
  for(int i = 0; i < QVALUE_SPATIAL_TARGET_NUM_CHANNELS * policySize; i++) {
    cPosTarget[i] = 0;
  }

  float scoreTargetCap = (float)(NNPos::MAX_BOARD_AREA + NNPos::EXTRA_SCORE_DISTR_RADIUS);

  size_t size = whiteQValueTargets.size();
  for(size_t i = 0; i<size; i++) {
    const QValueTargetMove& entry = whiteQValueTargets[i];
    int pos = NNPos::locToPos(entry.loc, boardXSize, dataXLen, dataYLen);
    assert(pos >= 0 && pos < policySize);

    float winLoss = nextPlayer == P_WHITE ? entry.winLoss : -entry.winLoss;
    float score = nextPlayer == P_WHITE ? entry.score : -entry.score;
    if(score > scoreTargetCap)
      score = scoreTargetCap;
    if(score < -scoreTargetCap)
      score = -scoreTargetCap;

    cPosTarget[pos] = clampToRadius32000(winLoss*32000.0f,rand);
    cPosTarget[pos+policySize] = clampToRadius32000(score*60.0f,rand);
    cPosTarget[pos+policySize*2] = (int16_t)(std::max((int64_t)0,std::min(entry.visits,(int64_t)32000)));
  }
}

static void fillValueTDTargets(const vector<ValueTargets>& whiteValueTargetsByTurn, int idx, Player nextPlayer, double nowFactor, float* buf) {
  double winValue = 0.0;
  double lossValue = 0.0;
  double noResultValue = 0.0;
  double score = 0.0;

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
    noResultValue += weightNow * targets.noResult;
    score += weightNow * (nextPlayer == P_WHITE ? targets.score : -targets.score);
  }
  double scoreTargetCap = NNPos::MAX_BOARD_AREA + NNPos::EXTRA_SCORE_DISTR_RADIUS;
  if(score > scoreTargetCap)
    score = scoreTargetCap;
  if(score < -scoreTargetCap)
    score = -scoreTargetCap;

  buf[0] = (float)winValue;
  buf[1] = (float)lossValue;
  buf[2] = (float)noResultValue;
  buf[3] = (float)score;
}

void TrainingWriteBuffers::addRow(
  const Board& board, const BoardHistory& hist, Player nextPlayer,
  const BoardHistory& startHist,
  const BoardHistory& actualGameEndHist,
  int turnIdx,
  float targetWeight,
  int64_t unreducedNumVisits,
  const vector<PolicyTargetMove>* policyTarget0, //can be null
  const vector<PolicyTargetMove>* policyTarget1, //can be null
  double policySurprise,
  double policyEntropy,
  double searchEntropy,
  const vector<ValueTargets>& whiteValueTargets,
  const vector<QValueTargets>& whiteQValueTargets,
  int whiteValueTargetsIdx, //index in whiteValueTargets corresponding to this turn.
  float valueTargetWeight,
  float tdValueTargetWeight,
  float leadTargetWeightFactor,
  const NNRawStats& nnRawStats,
  const Board* finalBoard,
  Color* finalFullArea,
  Color* finalOwnership,
  float* finalWhiteScoring,
  const vector<Board>* posHistForFutureBoards, //can be null
  bool isSidePosition,
  int numNeuralNetsBehindLatest,
  double drawEquivalentWinsForWhite,
  Player playoutDoublingAdvantagePla,
  double playoutDoublingAdvantage,
  Hash128 gameHash,
  const std::vector<ChangedNeuralNet*>& changedNeuralNets,
  bool hitTurnLimit,
  int numExtraBlack,
  int mode,
  SGFMetadata* sgfMeta,
  Rand& rand
) {
  static_assert(NNModelVersion::latestInputsVersionImplemented == 7, "");
  if(inputsVersion < 3 || inputsVersion > 7)
    throw StringError("Training write buffers: Does not support input version: " + Global::intToString(inputsVersion));

  int posArea = dataXLen*dataYLen;
  assert(curRows < maxRows);

  {
    MiscNNInputParams nnInputParams;
    nnInputParams.drawEquivalentWinsForWhite = drawEquivalentWinsForWhite;
    //Note: this is coordinated with the fact that selfplay does not use this feature on side positions
    if(!isSidePosition)
      nnInputParams.playoutDoublingAdvantage = getOpp(nextPlayer) == playoutDoublingAdvantagePla ? -playoutDoublingAdvantage : playoutDoublingAdvantage;
    else {
      assert(playoutDoublingAdvantagePla == C_EMPTY);
      assert(playoutDoublingAdvantage == 0.0);
    }

    bool inputsUseNHWC = false;
    float* rowBin = binaryInputNCHWUnpacked;
    float* rowGlobal = globalInputNC.data + curRows * numGlobalChannels;
    static_assert(NNModelVersion::latestInputsVersionImplemented == 7, "");
    if(inputsVersion == 3) {
      assert(NNInputs::NUM_FEATURES_SPATIAL_V3 == numBinaryChannels);
      assert(NNInputs::NUM_FEATURES_GLOBAL_V3 == numGlobalChannels);
      NNInputs::fillRowV3(board, hist, nextPlayer, nnInputParams, dataXLen, dataYLen, inputsUseNHWC, rowBin, rowGlobal);
    }
    else if(inputsVersion == 4) {
      assert(NNInputs::NUM_FEATURES_SPATIAL_V4 == numBinaryChannels);
      assert(NNInputs::NUM_FEATURES_GLOBAL_V4 == numGlobalChannels);
      NNInputs::fillRowV4(board, hist, nextPlayer, nnInputParams, dataXLen, dataYLen, inputsUseNHWC, rowBin, rowGlobal);
    }
    else if(inputsVersion == 5) {
      assert(NNInputs::NUM_FEATURES_SPATIAL_V5 == numBinaryChannels);
      assert(NNInputs::NUM_FEATURES_GLOBAL_V5 == numGlobalChannels);
      NNInputs::fillRowV5(board, hist, nextPlayer, nnInputParams, dataXLen, dataYLen, inputsUseNHWC, rowBin, rowGlobal);
    }
    else if(inputsVersion == 6) {
      assert(NNInputs::NUM_FEATURES_SPATIAL_V6 == numBinaryChannels);
      assert(NNInputs::NUM_FEATURES_GLOBAL_V6 == numGlobalChannels);
      NNInputs::fillRowV6(board, hist, nextPlayer, nnInputParams, dataXLen, dataYLen, inputsUseNHWC, rowBin, rowGlobal);
    }
    else if(inputsVersion == 7) {
      assert(NNInputs::NUM_FEATURES_SPATIAL_V7 == numBinaryChannels);
      assert(NNInputs::NUM_FEATURES_GLOBAL_V7 == numGlobalChannels);
      NNInputs::fillRowV7(board, hist, nextPlayer, nnInputParams, dataXLen, dataYLen, inputsUseNHWC, rowBin, rowGlobal);
    }
    else
      ASSERT_UNREACHABLE;

    //Pack bools bitwise into uint8_t
    uint8_t* rowBinPacked = binaryInputNCHWPacked.data + curRows * numBinaryChannels * packedBoardArea;
    for(int c = 0; c<numBinaryChannels; c++)
      packBits(rowBin + c * posArea, posArea, rowBinPacked + c * packedBoardArea);
  }

  //Vector for global targets and metadata
  float* rowGlobal = globalTargetsNC.data + curRows * GLOBAL_TARGET_NUM_CHANNELS;

  //Target weight for the whole row
  rowGlobal[25] = targetWeight;

  //Fill policy
  const int policySize = NNPos::getPolicySize(dataXLen,dataYLen);
  int16_t* rowPolicy = policyTargetsNCMove.data + curRows * POLICY_TARGET_NUM_CHANNELS * policySize;

  if(policyTarget0 != NULL) {
    fillPolicyTarget(*policyTarget0, policySize, dataXLen, dataYLen, board.x_size, rowPolicy + 0 * policySize);
    rowGlobal[26] = 1.0f;
  }
  else {
    uniformPolicyTarget(policySize, rowPolicy + 0 * policySize);
    rowGlobal[26] = 0.0f;
  }

  if(policyTarget1 != NULL) {
    fillPolicyTarget(*policyTarget1, policySize, dataXLen, dataYLen, board.x_size, rowPolicy + 1 * policySize);
    rowGlobal[28] = 1.0f;
  }
  else {
    uniformPolicyTarget(policySize, rowPolicy + 1 * policySize);
    rowGlobal[28] = 0.0f;
  }

  //Fill td-like value targets
  int boardArea = board.x_size * board.y_size;
  assert(whiteValueTargetsIdx >= 0 && whiteValueTargetsIdx < whiteValueTargets.size());
  fillValueTDTargets(whiteValueTargets, whiteValueTargetsIdx, nextPlayer, 0.0, rowGlobal);
  //These three constants used to be 'nicer' numbers 0.18, 0.06, 0.02, but we screwed up the functional form
  //by omitting the "1.0 +" at the front (breaks scaling to small board sizes), so when we fixed this we also
  //decreased the other numbers slightly to try to maximally limit the impact of the fix on the numerical values
  //on the actual board sizes 9-19, since it would be costly to retest.
  fillValueTDTargets(whiteValueTargets, whiteValueTargetsIdx, nextPlayer, 1.0/(1.0 + boardArea * 0.176), rowGlobal+4);
  fillValueTDTargets(whiteValueTargets, whiteValueTargetsIdx, nextPlayer, 1.0/(1.0 + boardArea * 0.056), rowGlobal+8);
  fillValueTDTargets(whiteValueTargets, whiteValueTargetsIdx, nextPlayer, 1.0/(1.0 + boardArea * 0.016), rowGlobal+12);
  fillValueTDTargets(whiteValueTargets, whiteValueTargetsIdx, nextPlayer, 1.0, rowGlobal+16);

  //Lead
  rowGlobal[21] = 0.0f;
  rowGlobal[29] = 0.0f;
  const ValueTargets& thisTargets = whiteValueTargets[whiteValueTargetsIdx];
  //If the actual game ended in a no-result, we don't use lead for any position during the game
  //including side positions, just in case.
  if(thisTargets.hasLead && !(actualGameEndHist.isGameFinished && actualGameEndHist.isNoResult)) {
    //Flip based on next player for training
    float lead = nextPlayer == P_WHITE ? thisTargets.lead : -thisTargets.lead;
    float scoreTargetCap = NNPos::MAX_BOARD_AREA + NNPos::EXTRA_SCORE_DISTR_RADIUS;
    if(lead > scoreTargetCap)
      lead = scoreTargetCap;
    if(lead < -scoreTargetCap)
      lead = -scoreTargetCap;

    rowGlobal[21] = lead;
    //Lead weight scales by how much we trust value in general
    rowGlobal[29] = valueTargetWeight * leadTargetWeightFactor;
  }

  //Expected time of arrival of winloss variance, in turns
  {
    double sum = 0.0;
    for(int i = whiteValueTargetsIdx+1; i<whiteValueTargets.size(); i++) {
      int turnsFromNow = i-whiteValueTargetsIdx;
      const ValueTargets& prevTargets = whiteValueTargets[i-1];
      const ValueTargets& targets = whiteValueTargets[i];
      double prevWL = prevTargets.win - prevTargets.loss;
      double nextWL = targets.win - targets.loss;
      double variance = (nextWL - prevWL) * (nextWL - prevWL);
      sum += turnsFromNow * variance;
    }
    rowGlobal[22] = (float)sum;
  }

  //Unused
  rowGlobal[23] = 0.0f;
  rowGlobal[24] = (float)(1.0f - tdValueTargetWeight);
  rowGlobal[30] = (float)policySurprise;
  rowGlobal[31] = (float)policyEntropy;
  rowGlobal[32] = (float)searchEntropy;
  // Value weight
  rowGlobal[35] = (float)(1.0f - valueTargetWeight);

  //Fill in whether we should use history or not
  bool useHist0 = rand.nextDouble() < 0.98;
  bool useHist1 = useHist0 && rand.nextDouble() < 0.98;
  bool useHist2 = useHist1 && rand.nextDouble() < 0.98;
  bool useHist3 = useHist2 && rand.nextDouble() < 0.98;
  bool useHist4 = useHist3 && rand.nextDouble() < 0.98;
  rowGlobal[36] = useHist0 ? 1.0f : 0.0f;
  rowGlobal[37] = useHist1 ? 1.0f : 0.0f;
  rowGlobal[38] = useHist2 ? 1.0f : 0.0f;
  rowGlobal[39] = useHist3 ? 1.0f : 0.0f;
  rowGlobal[40] = useHist4 ? 1.0f : 0.0f;

  //Fill in hash of game
  rowGlobal[41] = (float)(gameHash.hash0 & 0x3FFFFF);
  rowGlobal[42] = (float)((gameHash.hash0 >> 22) & 0x3FFFFF);
  rowGlobal[43] = (float)((gameHash.hash0 >> 44) & 0xFFFFF);
  rowGlobal[44] = (float)(gameHash.hash1 & 0x3FFFFF);
  rowGlobal[45] = (float)((gameHash.hash1 >> 22) & 0x3FFFFF);
  rowGlobal[46] = (float)((gameHash.hash1 >> 44) & 0xFFFFF);

  //Various other data
  rowGlobal[47] = hist.currentSelfKomi(nextPlayer,drawEquivalentWinsForWhite);
  rowGlobal[48] = (hist.encorePhase == 2 || hist.rules.scoringRule == Rules::SCORING_AREA) ? 1.0f : 0.0f;

  //Earlier neural net metadata
  rowGlobal[49] = changedNeuralNets.size() > 0 ? 1.0f : 0.0f;
  rowGlobal[50] = (float)numNeuralNetsBehindLatest;

  //Some misc metadata
  rowGlobal[51] = (float)turnIdx;
  rowGlobal[52] = hitTurnLimit ? 1.0f : 0.0f;
  rowGlobal[53] = (float)startHist.moveHistory.size();
  rowGlobal[54] = (float)numExtraBlack;

  //Metadata about how the game was initialized
  rowGlobal[55] = (float)mode;
  rowGlobal[56] = (float)hist.initialTurnNumber;

  //Some stats
  rowGlobal[57] = (float)(nextPlayer == P_WHITE ? nnRawStats.whiteWinLoss : -nnRawStats.whiteWinLoss);
  rowGlobal[58] = (float)(nextPlayer == P_WHITE ? nnRawStats.whiteScoreMean : -nnRawStats.whiteScoreMean);
  rowGlobal[59] = (float)nnRawStats.policyEntropy;

  //Original number of visits
  rowGlobal[60] = (float)unreducedNumVisits;

  //Bonus points
  if(!isSidePosition) {
    //Possibly this should count whiteHandicapBonusScore too, but in selfplay this never changes
    //after the start of a game
    float whiteBonusPoints = actualGameEndHist.whiteBonusScore - hist.whiteBonusScore;
    float selfBonusPoints = (nextPlayer == P_WHITE ? whiteBonusPoints : -whiteBonusPoints);
    //Note: we have a lot of data where this isn't reliable for side positions
    rowGlobal[61] = selfBonusPoints != 0 ? selfBonusPoints : 0.0f; //Conditional avoids negative zero
  }
  else {
    rowGlobal[61] = 0.0f;
  }

  //Game finished
  rowGlobal[62] = (!isSidePosition && actualGameEndHist.isGameFinished && !hitTurnLimit) ? 1.0f : 0.0f;

  //Version
  rowGlobal[63] = 2.0f;

  assert(64 == GLOBAL_TARGET_NUM_CHANNELS);

  int scoreDistrLen = posArea*2 + NNPos::EXTRA_SCORE_DISTR_RADIUS*2;
  int scoreDistrMid = posArea + NNPos::EXTRA_SCORE_DISTR_RADIUS;
  int8_t* rowScoreDistr = scoreDistrN.data + curRows * scoreDistrLen;
  int8_t* rowOwnership = valueTargetsNCHW.data + curRows * VALUE_SPATIAL_TARGET_NUM_CHANNELS * posArea;

  if(finalOwnership == NULL || (actualGameEndHist.isGameFinished && actualGameEndHist.isNoResult)) {
    rowGlobal[27] = 0.0f;
    rowGlobal[20] = 0.0f;
    for(int i = 0; i<posArea*2; i++)
      rowOwnership[i] = 0;
    for(int i = 0; i<scoreDistrLen; i++)
      rowScoreDistr[i] = 0;
    //Dummy value, to make sure it still sums to 100
    rowScoreDistr[scoreDistrMid-1] = 50;
    rowScoreDistr[scoreDistrMid] = 50;
  }
  else {
    assert(finalFullArea != NULL);
    assert(finalBoard != NULL);

    //Ownership weight scales by value weight
    rowGlobal[27] = valueTargetWeight;
    //Fill score info
    const ValueTargets& lastTargets = whiteValueTargets[whiteValueTargets.size()-1];
    float score = nextPlayer == P_WHITE ? lastTargets.score : -lastTargets.score;
    rowGlobal[20] = score;

    //Fill with zeros in case the buffers differ in size
    for(int i = 0; i<posArea*2; i++)
      rowOwnership[i] = 0;

    //Fill ownership info
    Player opp = getOpp(nextPlayer);
    for(int y = 0; y<board.y_size; y++) {
      for(int x = 0; x<board.x_size; x++) {
        int pos = NNPos::xyToPos(x,y,dataXLen);
        Loc loc = Location::getLoc(x,y,board.x_size);
        if(finalOwnership[loc] == nextPlayer) rowOwnership[pos] = 1;
        else if(finalOwnership[loc] == opp) rowOwnership[pos] = -1;
        //Mark full area points that ended up not being owned
        if(finalFullArea[loc] != C_EMPTY && finalOwnership[loc] == C_EMPTY)
          rowOwnership[pos+posArea] = (finalFullArea[loc] == nextPlayer ? 1 : -1);
      }
    }

    //Fill score vector "onehot"-like
    for(int i = 0; i<scoreDistrLen; i++)
      rowScoreDistr[i] = 0;
    int centerScore = (int)round(score);
    int lowerIdx = centerScore+scoreDistrMid-1;
    int upperIdx = centerScore+scoreDistrMid;
    if(upperIdx <= 0)
      rowScoreDistr[0] = 100;
    else if(lowerIdx >= scoreDistrLen-1)
      rowScoreDistr[scoreDistrLen] = 100;
    else {
      float lambda = score - (centerScore-0.5f);
      int upperProp = (int)round(lambda*100.0f);
      rowScoreDistr[lowerIdx] = 100-upperProp;
      rowScoreDistr[upperIdx] = upperProp;
    }
  }

  if(posHistForFutureBoards == NULL) {
    rowGlobal[33] = 0.0f;
    for(int i = 0; i<posArea; i++) {
      rowOwnership[i+posArea*2] = 0;
      rowOwnership[i+posArea*3] = 0;
    }
  }
  else {
    const vector<Board>& boards = *posHistForFutureBoards;
    assert(boards.size() == whiteValueTargets.size());
    assert(boards.size() > 0);

    // Future position weight
    rowGlobal[33] = 1.0f;
    int endIdx = (int)boards.size()-1;
    const Board& board2 = boards[std::min(whiteValueTargetsIdx+8,endIdx)];
    const Board& board3 = boards[std::min(whiteValueTargetsIdx+32,endIdx)];
    assert(board2.y_size == board.y_size && board2.x_size == board.x_size);
    assert(board3.y_size == board.y_size && board3.x_size == board.x_size);

    for(int i = 0; i<posArea; i++) {
      rowOwnership[i+posArea*2] = 0;
      rowOwnership[i+posArea*3] = 0;
    }
    Player pla = nextPlayer;
    Player opp = getOpp(nextPlayer);
    for(int y = 0; y<board.y_size; y++) {
      for(int x = 0; x<board.x_size; x++) {
        int pos = NNPos::xyToPos(x,y,dataXLen);
        Loc loc = Location::getLoc(x,y,board.x_size);
        if(board2.colors[loc] == pla) rowOwnership[pos+posArea*2] = 1;
        else if(board2.colors[loc] == opp) rowOwnership[pos+posArea*2] = -1;
        if(board3.colors[loc] == pla) rowOwnership[pos+posArea*3] = 1;
        else if(board3.colors[loc] == opp) rowOwnership[pos+posArea*3] = -1;
      }
    }
  }


  if(finalWhiteScoring == NULL || (actualGameEndHist.isGameFinished && actualGameEndHist.isNoResult)) {
    rowGlobal[34] = 0.0f;
    for(int i = 0; i<posArea; i++) {
      rowOwnership[i+posArea*4] = 0;
    }
  }
  else {
    // Scoring weight scales with value weight
    rowGlobal[34] = valueTargetWeight;
    //Fill with zeros in case the buffers differ in size
    for(int i = 0; i<posArea; i++) {
      rowOwnership[i+posArea*4] = 0;
    }

    for(int y = 0; y<board.y_size; y++) {
      for(int x = 0; x<board.x_size; x++) {
        int pos = NNPos::xyToPos(x,y,dataXLen);
        Loc loc = Location::getLoc(x,y,board.x_size);
        float scoring = (nextPlayer == P_WHITE ? finalWhiteScoring[loc] : -finalWhiteScoring[loc]);
        assert(scoring <= 1.0f && scoring >= -1.0f);
        rowOwnership[pos+posArea*4] = clampToRadius120(scoring*120.0f,rand);
      }
    }
  }


  //Q values
  {
    assert(whiteValueTargetsIdx < whiteQValueTargets.size());
    int16_t* rowQValues = qValueTargetsNCMove.data + curRows * QVALUE_SPATIAL_TARGET_NUM_CHANNELS * policySize;
    fillQValueTarget(whiteQValueTargets[whiteValueTargetsIdx].targets, nextPlayer, policySize, dataXLen, dataYLen, board.x_size, rowQValues, rand);
  }

  if(hasMetadataInput) {
    assert(sgfMeta != NULL);
    float* rowMetadata = metadataInputNC.data + curRows * SGFMetadata::METADATA_INPUT_NUM_CHANNELS;
    SGFMetadata::fillMetadataRow(sgfMeta, rowMetadata, nextPlayer, board.x_size * board.y_size);
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

  numBytes = scoreDistrN.prepareHeaderWithNumRows(curRows);
  zipFile.writeBuffer("scoreDistrN", scoreDistrN.dataIncludingHeader, numBytes);

  numBytes = valueTargetsNCHW.prepareHeaderWithNumRows(curRows);
  zipFile.writeBuffer("valueTargetsNCHW", valueTargetsNCHW.dataIncludingHeader, numBytes);

  numBytes = qValueTargetsNCMove.prepareHeaderWithNumRows(curRows);
  zipFile.writeBuffer("qValueTargetsNCMove", qValueTargetsNCMove.dataIncludingHeader, numBytes);

  if(hasMetadataInput) {
    numBytes = metadataInputNC.prepareHeaderWithNumRows(curRows);
    zipFile.writeBuffer("metadataInputNC", metadataInputNC.dataIncludingHeader, numBytes);
  }

  zipFile.close();
}

void TrainingWriteBuffers::writeToTextOstream(ostream& out) {
  int64_t len;

  auto printHeader = [&out](const char* dataIncludingHeader) {
    //In actuality our headers aren't that long, so we cut it off at half the total header bytes
    for(int i = 0; i<10; i++)
      out << (int)dataIncludingHeader[i] << " ";
    for(int i = 10; i<NumpyBuffer<int>::TOTAL_HEADER_BYTES/2; i++)
      out << dataIncludingHeader[i];
    out << endl;
  };

  out << "binaryInputNCHWPacked" << endl;
  binaryInputNCHWPacked.prepareHeaderWithNumRows(curRows);
  char buf[32];
  printHeader((const char*)binaryInputNCHWPacked.dataIncludingHeader);
  len = binaryInputNCHWPacked.getActualDataLen(curRows);
  for(int i = 0; i<len; i++) {
    sprintf(buf,"%02X",binaryInputNCHWPacked.data[i]);
    out << buf;
    if((i+1) % (len/curRows) == 0) out << endl;
  }
  out << endl;

  out << "globalInputNC" << endl;
  globalInputNC.prepareHeaderWithNumRows(curRows);
  printHeader((const char*)globalInputNC.dataIncludingHeader);
  len = globalInputNC.getActualDataLen(curRows);
  for(int i = 0; i<len; i++) {
    out << globalInputNC.data[i] << " ";
    if((i+1) % (len/curRows) == 0) out << endl;
  }
  out << endl;

  out << "policyTargetsNCMove" << endl;
  policyTargetsNCMove.prepareHeaderWithNumRows(curRows);
  printHeader((const char*)policyTargetsNCMove.dataIncludingHeader);
  len = policyTargetsNCMove.getActualDataLen(curRows);
  for(int i = 0; i<len; i++) {
    out << policyTargetsNCMove.data[i] << " ";
    if((i+1) % (len/curRows) == 0) out << endl;
  }
  out << endl;

  out << "globalTargetsNC" << endl;
  globalTargetsNC.prepareHeaderWithNumRows(curRows);
  printHeader((const char*)globalTargetsNC.dataIncludingHeader);
  len = globalTargetsNC.getActualDataLen(curRows);
  for(int i = 0; i<len; i++) {
    out << globalTargetsNC.data[i] << " ";
    if((i+1) % (len/curRows) == 0) out << endl;
  }
  out << endl;

  out << "scoreDistrN" << endl;
  scoreDistrN.prepareHeaderWithNumRows(curRows);
  printHeader((const char*)scoreDistrN.dataIncludingHeader);
  len = scoreDistrN.getActualDataLen(curRows);
  for(int i = 0; i<len; i++) {
    out << (int)scoreDistrN.data[i] << " ";
    if((i+1) % (len/curRows) == 0) out << endl;
  }
  out << endl;

  out << "valueTargetsNCHW" << endl;
  valueTargetsNCHW.prepareHeaderWithNumRows(curRows);
  printHeader((const char*)valueTargetsNCHW.dataIncludingHeader);
  len = valueTargetsNCHW.getActualDataLen(curRows);
  for(int i = 0; i<len; i++) {
    out << (int)valueTargetsNCHW.data[i] << " ";
    if((i+1) % (len/curRows) == 0) out << endl;
  }
  out << endl;

  out << "qValueTargetsNCMove" << endl;
  qValueTargetsNCMove.prepareHeaderWithNumRows(curRows);
  printHeader((const char*)qValueTargetsNCMove.dataIncludingHeader);
  len = qValueTargetsNCMove.getActualDataLen(curRows);
  for(int i = 0; i<len; i++) {
    out << (int)qValueTargetsNCMove.data[i] << " ";
    if((i+1) % (len/curRows) == 0) out << endl;
  }
  out << endl;

  if(hasMetadataInput) {
    out << "metadataInputNC" << endl;
    metadataInputNC.prepareHeaderWithNumRows(curRows);
    printHeader((const char*)metadataInputNC.dataIncludingHeader);
    len = metadataInputNC.getActualDataLen(curRows);
    for(int i = 0; i<len; i++) {
      out << (int)metadataInputNC.data[i] << " ";
      if((i+1) % (len/curRows) == 0) out << endl;
    }
    out << endl;
  }
}

//-------------------------------------------------------------------------------------

TrainingDataWriter::TrainingDataWriter(const string& outDir, int iVersion, int maxRowsPerFile, double firstFileMinRandProp, int dataXLen, int dataYLen, const string& randSeed)
  : TrainingDataWriter(outDir,NULL,iVersion,maxRowsPerFile,firstFileMinRandProp,dataXLen,dataYLen,1,randSeed)
{}
TrainingDataWriter::TrainingDataWriter(ostream* dbgOut, int iVersion, int maxRowsPerFile, double firstFileMinRandProp, int dataXLen, int dataYLen, int onlyEvery, const string& randSeed)
  : TrainingDataWriter(string(),dbgOut,iVersion,maxRowsPerFile,firstFileMinRandProp,dataXLen,dataYLen,onlyEvery,randSeed)
{}

TrainingDataWriter::TrainingDataWriter(const string& outDir, ostream* dbgOut, int iVersion, int maxRowsPerFile, double firstFileMinRandProp, int dataXLen, int dataYLen, int onlyEvery, const string& randSeed)
  :outputDir(outDir),inputsVersion(iVersion),rand(randSeed),writeBuffers(NULL),debugOut(dbgOut),debugOnlyWriteEvery(onlyEvery),rowCount(0)
{
  int numBinaryChannels;
  int numGlobalChannels;
  //Note that this inputsVersion is for data writing, it might be different than the inputsVersion used
  //to feed into a model during selfplay
  static_assert(NNModelVersion::latestInputsVersionImplemented == 7, "");
  if(inputsVersion == 3) {
    numBinaryChannels = NNInputs::NUM_FEATURES_SPATIAL_V3;
    numGlobalChannels = NNInputs::NUM_FEATURES_GLOBAL_V3;
  }
  else if(inputsVersion == 4) {
    numBinaryChannels = NNInputs::NUM_FEATURES_SPATIAL_V4;
    numGlobalChannels = NNInputs::NUM_FEATURES_GLOBAL_V4;
  }
  else if(inputsVersion == 5) {
    numBinaryChannels = NNInputs::NUM_FEATURES_SPATIAL_V5;
    numGlobalChannels = NNInputs::NUM_FEATURES_GLOBAL_V5;
  }
  else if(inputsVersion == 6) {
    numBinaryChannels = NNInputs::NUM_FEATURES_SPATIAL_V6;
    numGlobalChannels = NNInputs::NUM_FEATURES_GLOBAL_V6;
  }
  else if(inputsVersion == 7) {
    numBinaryChannels = NNInputs::NUM_FEATURES_SPATIAL_V7;
    numGlobalChannels = NNInputs::NUM_FEATURES_GLOBAL_V7;
  }
  else {
    throw StringError("TrainingDataWriter: Unsupported inputs version: " + Global::intToString(inputsVersion));
  }

  const bool hasMetadataInput = false;
  writeBuffers = new TrainingWriteBuffers(
    inputsVersion,
    maxRowsPerFile,
    numBinaryChannels,
    numGlobalChannels,
    dataXLen,
    dataYLen,
    hasMetadataInput
  );

  if(firstFileMinRandProp < 0 || firstFileMinRandProp > 1)
    throw StringError("TrainingDataWriter: firstFileMinRandProp not in [0,1]: " + Global::doubleToString(firstFileMinRandProp));
  isFirstFile = true;
  if(firstFileMinRandProp >= 1.0)
    firstFileMaxRows = maxRowsPerFile;
  else
    firstFileMaxRows = maxRowsPerFile - (int)(maxRowsPerFile * (1.0-firstFileMinRandProp) * rand.nextDouble());
}



TrainingDataWriter::~TrainingDataWriter()
{
  delete writeBuffers;
}

bool TrainingDataWriter::isEmpty() const {
  return writeBuffers->curRows <= 0;
}
int64_t TrainingDataWriter::numRowsInBuffer() const {
  return writeBuffers->curRows;
}

void TrainingDataWriter::writeAndClearIfFull() {
  if(writeBuffers->curRows >= writeBuffers->maxRows || (isFirstFile && writeBuffers->curRows >= firstFileMaxRows)) {
    flushIfNonempty();
  }
}



void TrainingDataWriter::flushIfNonempty() {
  string resultingFilename;
  flushIfNonempty(resultingFilename);
}

bool TrainingDataWriter::flushIfNonempty(string& resultingFilename) {
  if(writeBuffers->curRows <= 0)
    return false;

  isFirstFile = false;

  if(debugOut != NULL) {
    writeBuffers->writeToTextOstream(*debugOut);
    writeBuffers->clear();
    resultingFilename = "";
  }
  else {
    resultingFilename = outputDir + "/" + Global::uint64ToHexString(rand.nextUInt64()) + ".npz";
    string tmpFilename = resultingFilename + ".tmp";
    writeBuffers->writeToZipFile(tmpFilename);
    writeBuffers->clear();
    FileUtils::rename(tmpFilename,resultingFilename);
  }
  return true;
}

void TrainingDataWriter::writeGame(const FinishedGameData& data) {
  int numMoves = (int)(data.endHist.moveHistory.size() - data.startHist.moveHistory.size());
  assert(numMoves >= 0);
  assert(data.startHist.moveHistory.size() <= data.endHist.moveHistory.size());
  assert(data.endHist.moveHistory.size() <= 100000000);
  assert(data.targetWeightByTurn.size() == numMoves);
  assert(data.targetWeightByTurnUnrounded.size() == numMoves);
  assert(data.policyTargetsByTurn.size() == numMoves);
  assert(data.policySurpriseByTurn.size() == numMoves);
  assert(data.policyEntropyByTurn.size() == numMoves);
  assert(data.searchEntropyByTurn.size() == numMoves);
  assert(data.whiteValueTargetsByTurn.size() == numMoves+1);
  assert(data.whiteQValueTargetsByTurn.size() == numMoves);
  assert(data.nnRawStatsByTurn.size() == numMoves);

  //Some sanity checks
  #ifndef NDEBUG
  {
    const ValueTargets& lastTargets = data.whiteValueTargetsByTurn[data.whiteValueTargetsByTurn.size()-1];
    if(!data.endHist.isGameFinished)
      assert(data.hitTurnLimit);
    else if(data.endHist.isNoResult)
      assert(lastTargets.win == 0.0f && lastTargets.loss == 0.0f && lastTargets.noResult == 1.0f);
    else if(data.endHist.winner == P_BLACK)
      assert(lastTargets.win == 0.0f && lastTargets.loss == 1.0f && lastTargets.noResult == 0.0f);
    else if(data.endHist.winner == P_WHITE)
      assert(lastTargets.win == 1.0f && lastTargets.loss == 0.0f && lastTargets.noResult == 0.0f);
    else
      assert(lastTargets.noResult == 0.0f);

    assert(data.finalFullArea != NULL);
    assert(data.finalOwnership != NULL);
    assert(data.finalSekiAreas != NULL);
    assert(data.finalWhiteScoring != NULL);
    assert(!data.endHist.isResignation);
  }
  #endif

  //Play out all the moves in a single pass first to compute all the future board states
  vector<Board> posHistForFutureBoards;
  {
    Board board(data.startBoard);
    BoardHistory hist(data.startHist);
    Player nextPlayer = data.startPla;
    posHistForFutureBoards.push_back(board);

    int startTurnIdx = (int)data.startHist.moveHistory.size();
    for(int turnAfterStart = 0; turnAfterStart<numMoves; turnAfterStart++) {
      int turnIdx = turnAfterStart + startTurnIdx;

      Move move = data.endHist.moveHistory[turnIdx];
      assert(move.pla == nextPlayer);
      assert(hist.isLegal(board,move.loc,move.pla));
      hist.makeBoardMoveAssumeLegal(board, move.loc, move.pla, NULL);
      nextPlayer = getOpp(nextPlayer);

      posHistForFutureBoards.push_back(board);
    }
  }

  assert(data.hasFullData);

  Board board(data.startBoard);
  BoardHistory hist(data.startHist);
  Player nextPlayer = data.startPla;

  //Write main game rows
  int startTurnIdx = (int)data.startHist.moveHistory.size();
  for(int turnAfterStart = 0; turnAfterStart<numMoves; turnAfterStart++) {
    double targetWeight = data.targetWeightByTurn[turnAfterStart];
    int turnIdx = turnAfterStart + startTurnIdx;

    int64_t unreducedNumVisits = data.policyTargetsByTurn[turnAfterStart].unreducedNumVisits;
    const vector<PolicyTargetMove>* policyTarget0 = data.policyTargetsByTurn[turnAfterStart].policyTargets;
    const vector<PolicyTargetMove>* policyTarget1 = (turnAfterStart + 1 < numMoves) ? data.policyTargetsByTurn[turnAfterStart+1].policyTargets : NULL;
    bool isSidePosition = false;
    float valueTargetWeight = 1.0f;
    float tdValueTargetWeight = 1.0f;
    float leadTargetWeightFactor = 1.0f;

    int numNeuralNetsBehindLatest = 0;
    for(int i = 0; i<data.changedNeuralNets.size(); i++) {
      if(data.changedNeuralNets[i]->turnIdx > turnIdx) {
        numNeuralNetsBehindLatest = (int)data.changedNeuralNets.size()-i;
        break;
      }
    }

    while(targetWeight > 0.0) {
      if(targetWeight >= 1.0 || rand.nextBool(targetWeight)) {
        if(debugOut == NULL || rowCount % debugOnlyWriteEvery == 0) {
          writeBuffers->addRow(
            board,hist,nextPlayer,
            data.startHist,
            data.endHist,
            turnIdx,
            (float)data.trainingWeight,
            unreducedNumVisits,
            policyTarget0,
            policyTarget1,
            data.policySurpriseByTurn[turnAfterStart],
            data.policyEntropyByTurn[turnAfterStart],
            data.searchEntropyByTurn[turnAfterStart],
            data.whiteValueTargetsByTurn,
            data.whiteQValueTargetsByTurn,
            turnAfterStart,
            valueTargetWeight,
            tdValueTargetWeight,
            leadTargetWeightFactor,
            data.nnRawStatsByTurn[turnAfterStart],
            &(data.endHist.getRecentBoard(0)),
            data.finalFullArea,
            data.finalOwnership,
            data.finalWhiteScoring,
            &posHistForFutureBoards,
            isSidePosition,
            numNeuralNetsBehindLatest,
            data.drawEquivalentWinsForWhite,
            data.playoutDoublingAdvantagePla,
            data.playoutDoublingAdvantage,
            data.gameHash,
            data.changedNeuralNets,
            data.hitTurnLimit,
            data.numExtraBlack,
            data.mode,
            NULL,
            rand
          );
          writeAndClearIfFull();
        }
        rowCount++;
      }
      targetWeight -= 1.0;
    }


    Move move = data.endHist.moveHistory[turnIdx];
    assert(move.pla == nextPlayer);
    assert(hist.isLegal(board,move.loc,move.pla));
    hist.makeBoardMoveAssumeLegal(board, move.loc, move.pla, NULL);
    nextPlayer = getOpp(nextPlayer);
  }

  //Write side rows
  vector<ValueTargets> whiteValueTargetsBuf(1);
  vector<QValueTargets> whiteQValueTargetsBuf(1);
  for(int i = 0; i<data.sidePositions.size(); i++) {
    SidePosition* sp = data.sidePositions[i];

    double targetWeight = sp->targetWeight;
    while(targetWeight > 0.0) {
      if(targetWeight >= 1.0 || rand.nextBool(targetWeight)) {
        if(debugOut == NULL || rowCount % debugOnlyWriteEvery == 0) {

          int turnIdx = (int)sp->hist.moveHistory.size();
          assert(turnIdx >= data.startHist.moveHistory.size());
          whiteValueTargetsBuf[0] = sp->whiteValueTargets;
          whiteQValueTargetsBuf[0] = sp->whiteQValueTargets;
          bool isSidePosition = true;
          int numNeuralNetsBehindLatest = (int)data.changedNeuralNets.size() - sp->numNeuralNetChangesSoFar;
          float valueTargetWeight = 1.0f;
          float tdValueTargetWeight = 1.0f;
          float leadTargetWeightFactor = 1.0f;

          writeBuffers->addRow(
            sp->board,sp->hist,sp->pla,
            data.startHist,
            data.endHist, // actual game ending hist, even for side position
            turnIdx,
            (float)data.trainingWeight,
            sp->unreducedNumVisits,
            &(sp->policyTarget),
            NULL,
            sp->policySurprise,
            sp->policyEntropy,
            sp->searchEntropy,
            whiteValueTargetsBuf,
            whiteQValueTargetsBuf,
            0,
            valueTargetWeight,
            tdValueTargetWeight,
            leadTargetWeightFactor,
            sp->nnRawStats,
            NULL,
            NULL,
            NULL,
            NULL,
            NULL,
            isSidePosition,
            numNeuralNetsBehindLatest,
            data.drawEquivalentWinsForWhite,
            sp->playoutDoublingAdvantagePla,
            sp->playoutDoublingAdvantage,
            data.gameHash,
            data.changedNeuralNets,
            data.hitTurnLimit, // actual game hit turn limit
            data.numExtraBlack,
            data.mode,
            NULL,
            rand
          );
          writeAndClearIfFull();
        }
        rowCount++;
      }
      targetWeight -= 1.0;
    }

  }

}
