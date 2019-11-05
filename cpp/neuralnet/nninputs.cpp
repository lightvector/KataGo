#include "../neuralnet/nninputs.h"

using namespace std;

int NNPos::xyToPos(int x, int y, int nnXLen) {
  return y * nnXLen + x;
}
int NNPos::locToPos(Loc loc, int boardXSize, int nnXLen, int nnYLen) {
  if(loc == Board::PASS_LOC)
    return nnXLen * nnYLen;
  else if(loc == Board::NULL_LOC)
    return nnXLen * (nnYLen + 1);
  return Location::getY(loc,boardXSize) * nnXLen + Location::getX(loc,boardXSize);
}
Loc NNPos::posToLoc(int pos, int boardXSize, int boardYSize, int nnXLen, int nnYLen) {
  if(pos == nnXLen * nnYLen)
    return Board::PASS_LOC;
  int x = pos % nnXLen;
  int y = pos / nnXLen;
  if(x < 0 || x >= boardXSize || y < 0 || y >= boardYSize)
    return Board::NULL_LOC;
  return Location::getLoc(x,y,boardXSize);
}

bool NNPos::isPassPos(int pos, int nnXLen, int nnYLen) {
  return pos == nnXLen * nnYLen;
}

int NNPos::getPolicySize(int nnXLen, int nnYLen) {
  return nnXLen * nnYLen + 1;
}

//-----------------------------------------------------------------------------------------------------------
//-----------------------------------------------------------------------------------------------------------

const Hash128 MiscNNInputParams::ZOBRIST_CONSERVATIVE_PASS =
  Hash128(0x0c2b96f4b8ae2da9ULL, 0x5a14dee208fec0edULL);

//-----------------------------------------------------------------------------------------------------------
//-----------------------------------------------------------------------------------------------------------

double ScoreValue::whiteWinsOfWinner(Player winner, double drawEquivalentWinsForWhite) {
  if(winner == P_WHITE)
    return 1.0;
  else if(winner == P_BLACK)
    return 0.0;

  assert(winner == C_EMPTY);
  return drawEquivalentWinsForWhite;
}

static const double twoOverPi = 0.63661977236758134308;
static const double piOverTwo = 1.57079632679489661923;

double ScoreValue::whiteScoreDrawAdjust(double finalWhiteMinusBlackScore, double drawEquivalentWinsForWhite, const BoardHistory& hist) {
  return finalWhiteMinusBlackScore + hist.whiteKomiAdjustmentForDraws(drawEquivalentWinsForWhite);
}

double ScoreValue::whiteScoreValueOfScoreSmooth(double finalWhiteMinusBlackScore, double center, double scale, double drawEquivalentWinsForWhite, const Board& b, const BoardHistory& hist) {
  double adjustedScore = finalWhiteMinusBlackScore + hist.whiteKomiAdjustmentForDraws(drawEquivalentWinsForWhite) - center;
  if(b.x_size == b.y_size)
    return atan(adjustedScore / (scale*b.x_size)) * twoOverPi;
  else
    return atan(adjustedScore / (scale*sqrt(b.x_size*b.y_size))) * twoOverPi;
}

double ScoreValue::whiteScoreValueOfScoreSmoothNoDrawAdjust(double finalWhiteMinusBlackScore, double center, double scale, const Board& b) {
  double adjustedScore = finalWhiteMinusBlackScore - center;
  if(b.x_size == b.y_size)
    return atan(adjustedScore / (scale*b.x_size)) * twoOverPi;
  else
    return atan(adjustedScore / (scale*sqrt(b.x_size*b.y_size))) * twoOverPi;
}

static double inverse_atan(double x) {
  if(x >= piOverTwo - 1e-6) return 1e6;
  if(x <= -piOverTwo + 1e-6) return -1e6;
  return tan(x);
}

double ScoreValue::approxWhiteScoreOfScoreValueSmooth(double scoreValue, double center, double scale, const Board& b) {
  assert(scoreValue >= -1 && scoreValue <= 1);
  double scoreUnscaled = inverse_atan(scoreValue*piOverTwo);
  if(b.x_size == b.y_size)
    return scoreUnscaled * (scale*b.x_size) + center;
  else
    return scoreUnscaled * (scale*sqrt(b.x_size*b.y_size)) + center;
}

double ScoreValue::whiteScoreMeanSqOfScoreGridded(double finalWhiteMinusBlackScore, double drawEquivalentWinsForWhite, const BoardHistory& hist) {
  bool komiIsInteger = ((int)hist.rules.komi == hist.rules.komi);
  if(!komiIsInteger)
    return finalWhiteMinusBlackScore * finalWhiteMinusBlackScore;

  assert((int)finalWhiteMinusBlackScore == finalWhiteMinusBlackScore);
  double lower = finalWhiteMinusBlackScore - 0.5;
  double upper = finalWhiteMinusBlackScore + 0.5;
  double lowerSq = lower * lower;
  double upperSq = upper * upper;

  return lowerSq + (upperSq - lowerSq) * drawEquivalentWinsForWhite;
}


static bool scoreValueTablesInitialized = false;
static double* expectedSVTable = NULL;
static const int svTableAssumedBSize = NNPos::MAX_BOARD_LEN;
static const int svTableMeanRadius = svTableAssumedBSize*svTableAssumedBSize + NNPos::EXTRA_SCORE_DISTR_RADIUS;
static const int svTableMeanLen = svTableMeanRadius*2;
static const int svTableStdevLen = svTableAssumedBSize*svTableAssumedBSize + NNPos::EXTRA_SCORE_DISTR_RADIUS;

void ScoreValue::freeTables() {
  if(scoreValueTablesInitialized) {
    delete[] expectedSVTable;
    expectedSVTable = NULL;
    scoreValueTablesInitialized = false;
  }
}

void ScoreValue::initTables() {
  assert(!scoreValueTablesInitialized);
  expectedSVTable = new double[svTableMeanLen*svTableStdevLen];

  //Precompute normal PDF
  const int stepsPerUnit = 10; //Must be divisible by 2. This is both the number of segments that we divide points into, and that we divide stdevs into
  const int boundStdevs = 5;
  int minStdevSteps = -boundStdevs*stepsPerUnit;
  int maxStdevSteps = boundStdevs*stepsPerUnit;
  double* normalPDF = new double[(maxStdevSteps-minStdevSteps)+1];
  for(int i = minStdevSteps; i <= maxStdevSteps; i++) {
    double xInStdevs = (double)i / stepsPerUnit;
    double w = exp(-0.5 * xInStdevs * xInStdevs);
    normalPDF[i-minStdevSteps] = w;
  }
  //Precompute scorevalue at increments of 1/stepsPerUnit points
  Board board(svTableAssumedBSize,svTableAssumedBSize);
  int minSVSteps = - (svTableMeanRadius*stepsPerUnit + stepsPerUnit/2 + boundStdevs * svTableStdevLen * stepsPerUnit);
  int maxSVSteps = -minSVSteps;
  double* svPrecomp = new double[(maxSVSteps-minSVSteps)+1];
  for(int i = minSVSteps; i <= maxSVSteps; i++) {
    double mean = (double)i / stepsPerUnit;
    double sv = whiteScoreValueOfScoreSmoothNoDrawAdjust(mean, 0.0, 1.0, board);
    svPrecomp[i-minSVSteps] = sv;
  }

  //Perform numeric integration
  for(int meanIdx = 0; meanIdx < svTableMeanLen; meanIdx++) {
    int meanSteps = (meanIdx - svTableMeanRadius) * stepsPerUnit - stepsPerUnit/2;
    for(int stdevIdx = 0; stdevIdx < svTableStdevLen; stdevIdx++) {
      double wSum = 0.0;
      double wsvSum = 0.0;
      for(int i = minStdevSteps; i <= maxStdevSteps; i++) {
        int xSteps = meanSteps + stdevIdx * i;
        double w = normalPDF[i-minStdevSteps];
        assert(xSteps >= minSVSteps && xSteps <= maxSVSteps);
        double sv = svPrecomp[xSteps-minSVSteps];
        wSum += w;
        wsvSum += w*sv;
      }
      expectedSVTable[meanIdx*svTableStdevLen + stdevIdx] = wsvSum / wSum;
    }
  }

  delete[] normalPDF;
  delete[] svPrecomp;
  scoreValueTablesInitialized = true;
}

double ScoreValue::expectedWhiteScoreValue(double whiteScoreMean, double whiteScoreStdev, double center, double scale, const Board& b) {
  assert(scoreValueTablesInitialized);

  double scaleFactor;
  if(b.x_size == b.y_size)
    scaleFactor = (double)svTableAssumedBSize / (scale * b.x_size);
  else
    scaleFactor = (double)svTableAssumedBSize / (scale * sqrt(b.x_size*b.y_size));

  double meanScaled = (whiteScoreMean - center) * scaleFactor;
  double stdevScaled = whiteScoreStdev * scaleFactor;

  double meanRounded = round(meanScaled);
  double stdevFloored = floor(stdevScaled);
  int meanIdx0 = (int)meanRounded + svTableMeanRadius;
  int stdevIdx0 = (int)stdevFloored;
  int meanIdx1 = meanIdx0+1;
  int stdevIdx1 = stdevIdx0+1;

  if(meanIdx0 < 0) { meanIdx0 = 0; meanIdx1 = 0; }
  if(meanIdx1 >= svTableMeanLen) { meanIdx0 = svTableMeanLen-1; meanIdx1 = svTableMeanLen-1; }
  assert(stdevIdx0 >= 0);
  if(stdevIdx1 >= svTableStdevLen) { stdevIdx0 = svTableStdevLen-1; stdevIdx1 = svTableStdevLen-1; }

  double lambdaMean = meanScaled - meanRounded + 0.5;
  double lambdaStdev = stdevScaled - stdevFloored;

  double a00 = expectedSVTable[meanIdx0*svTableStdevLen + stdevIdx0];
  double a01 = expectedSVTable[meanIdx0*svTableStdevLen + stdevIdx1];
  double a10 = expectedSVTable[meanIdx1*svTableStdevLen + stdevIdx0];
  double a11 = expectedSVTable[meanIdx1*svTableStdevLen + stdevIdx1];

  double b0 = a00 + lambdaStdev*(a01-a00);
  double b1 = a10 + lambdaStdev*(a11-a10);
  return b0 + lambdaMean*(b1-b0);
}

//-----------------------------------------------------------------------------------------------------------
//-----------------------------------------------------------------------------------------------------------

void NNInputs::fillOwnership(
  const Board& board,
  const Color* area,
  bool groupTax,
  int nnXLen,
  int nnYLen,
  float* ownership
) {
  if(!groupTax) {
    std::fill(ownership, ownership + nnXLen*nnYLen, 0.0f);
    for(int y = 0; y<board.y_size; y++) {
      for(int x = 0; x<board.x_size; x++) {
        Loc loc = Location::getLoc(x,y,board.x_size);
        int pos = NNPos::locToPos(loc,board.x_size,nnXLen,nnYLen);
        Color areaColor = area[loc];
        if(areaColor == P_BLACK)
          ownership[pos] = -1.0f;
        else if(areaColor == P_WHITE)
          ownership[pos] = 1.0f;
        else {
          assert(areaColor == C_EMPTY);
          ownership[pos] = 0;
        }
      }
    }
  }
  else {
    bool visited[Board::MAX_ARR_SIZE];
    Loc queue[Board::MAX_ARR_SIZE];

    std::fill(visited, visited + Board::MAX_ARR_SIZE, false);
    std::fill(ownership, ownership + nnXLen*nnYLen, 0.0f);
    for(int y = 0; y<board.y_size; y++) {
      for(int x = 0; x<board.x_size; x++) {
        Loc loc = Location::getLoc(x,y,board.x_size);
        if(visited[loc])
          continue;
        Color areaColor = area[loc];
        if(areaColor == P_BLACK || areaColor == P_WHITE) {
          float fullValue = areaColor == P_WHITE ? 1.0f : -1.0f;
          int queueHead = 0;
          int queueTail = 1;
          queue[0] = loc;
          visited[loc] = true;

          //First, count how many empty or opp locations there are
          int territoryCount = 0;
          while(queueHead < queueTail) {
            Loc next = queue[queueHead];
            queueHead++;
            if(board.colors[next] != areaColor)
              territoryCount++;
            //Push adjacent locations on to queue
            for(int i = 0; i<4; i++) {
              Loc adj = next + board.adj_offsets[i];
              if(area[adj] == areaColor && !visited[adj]) {
                queue[queueTail] = adj;
                queueTail++;
                visited[adj] = true;
              }
            }
          }

          //Then, actually fill values
          float territoryValue = territoryCount <= 2 ? 0.0f : fullValue * (territoryCount - 2.0f) / territoryCount;
          for(int j = 0; j<queueTail; j++) {
            Loc next = queue[j];
            queueHead++;
            int pos = NNPos::locToPos(next,board.x_size,nnXLen,nnYLen);
            if(board.colors[next] != areaColor)
              ownership[pos] = territoryValue;
            else
              ownership[pos] = fullValue;
          }
        }
        else {
          assert(areaColor == C_EMPTY);
          int pos = NNPos::locToPos(loc,board.x_size,nnXLen,nnYLen);
          ownership[pos] = 0;
        }
      }
    }
  }
}


//-----------------------------------------------------------------------------------------------------------
//-----------------------------------------------------------------------------------------------------------


NNOutput::NNOutput()
  :whiteOwnerMap(NULL),noisedPolicyProbs(NULL)
{}
NNOutput::NNOutput(const NNOutput& other) {
  nnHash = other.nnHash;
  whiteWinProb = other.whiteWinProb;
  whiteLossProb = other.whiteLossProb;
  whiteNoResultProb = other.whiteNoResultProb;
  whiteScoreMean = other.whiteScoreMean;
  whiteScoreMeanSq = other.whiteScoreMeanSq;

  nnXLen = other.nnXLen;
  nnYLen = other.nnYLen;
  if(other.whiteOwnerMap != NULL) {
    whiteOwnerMap = new float[nnXLen * nnYLen];
    std::copy(other.whiteOwnerMap, other.whiteOwnerMap + nnXLen * nnYLen, whiteOwnerMap);
  }
  else
    whiteOwnerMap = NULL;

  if(other.noisedPolicyProbs != NULL) {
    noisedPolicyProbs = new float[NNPos::MAX_NN_POLICY_SIZE];
    std::copy(other.noisedPolicyProbs, other.noisedPolicyProbs + NNPos::MAX_NN_POLICY_SIZE, noisedPolicyProbs);
  }
  else
    noisedPolicyProbs = NULL;

  std::copy(other.policyProbs, other.policyProbs+NNPos::MAX_NN_POLICY_SIZE, policyProbs);
}

NNOutput& NNOutput::operator=(const NNOutput& other) {
  if(&other == this)
    return *this;
  nnHash = other.nnHash;
  whiteWinProb = other.whiteWinProb;
  whiteLossProb = other.whiteLossProb;
  whiteNoResultProb = other.whiteNoResultProb;
  whiteScoreMean = other.whiteScoreMean;
  whiteScoreMeanSq = other.whiteScoreMeanSq;

  nnXLen = other.nnXLen;
  nnYLen = other.nnYLen;
  if(whiteOwnerMap != NULL)
    delete[] whiteOwnerMap;
  if(other.whiteOwnerMap != NULL) {
    whiteOwnerMap = new float[nnXLen * nnYLen];
    std::copy(other.whiteOwnerMap, other.whiteOwnerMap + nnXLen * nnYLen, whiteOwnerMap);
  }
  else
    whiteOwnerMap = NULL;
  if(noisedPolicyProbs != NULL)
    delete[] noisedPolicyProbs;
  if(other.noisedPolicyProbs != NULL) {
    noisedPolicyProbs = new float[NNPos::MAX_NN_POLICY_SIZE];
    std::copy(other.noisedPolicyProbs, other.noisedPolicyProbs + NNPos::MAX_NN_POLICY_SIZE, noisedPolicyProbs);
  }
  else
    noisedPolicyProbs = NULL;

  std::copy(other.policyProbs, other.policyProbs+NNPos::MAX_NN_POLICY_SIZE, policyProbs);

  return *this;
}


NNOutput::~NNOutput() {
  if(whiteOwnerMap != NULL) {
    delete[] whiteOwnerMap;
    whiteOwnerMap = NULL;
  }
  if(noisedPolicyProbs != NULL) {
    delete[] noisedPolicyProbs;
    noisedPolicyProbs = NULL;
  }
}


void NNOutput::debugPrint(ostream& out, const Board& board) {
  out << "Win " << Global::strprintf("%.2fc",whiteWinProb*100) << endl;
  out << "Loss " << Global::strprintf("%.2fc",whiteLossProb*100) << endl;
  out << "NoResult " << Global::strprintf("%.2fc",whiteNoResultProb*100) << endl;
  out << "ScoreMean " << Global::strprintf("%.1f",whiteScoreMean) << endl;
  out << "ScoreMeanSq " << Global::strprintf("%.1f",whiteScoreMeanSq) << endl;

  out << "Policy" << endl;
  for(int y = 0; y<board.y_size; y++) {
    for(int x = 0; x<board.x_size; x++) {
      int pos = NNPos::xyToPos(x,y,nnXLen);
      float prob = policyProbs[pos];
      if(prob < 0)
        out << "   - ";
      else
        out << Global::strprintf("%4d ", (int)round(prob * 1000));
    }
    out << endl;
  }

  if(whiteOwnerMap != NULL) {
    for(int y = 0; y<board.y_size; y++) {
      for(int x = 0; x<board.x_size; x++) {
        int pos = NNPos::xyToPos(x,y,nnXLen);
        float whiteOwn = whiteOwnerMap[pos];
        out << Global::strprintf("%5d ", (int)round(whiteOwn * 1000));
      }
      out << endl;
    }
    out << endl;
  }
}


//-------------------------------------------------------------------------------------------------------------


static void setRowBinV3(float* rowBin, int pos, int feature, float value, int posStride, int featureStride) {
  rowBin[pos * posStride + feature * featureStride] = value;
}
static void setRowBinV4(float* rowBin, int pos, int feature, float value, int posStride, int featureStride) {
  rowBin[pos * posStride + feature * featureStride] = value;
}
static void setRowBinV5(float* rowBin, int pos, int feature, float value, int posStride, int featureStride) {
  rowBin[pos * posStride + feature * featureStride] = value;
}
static void setRowBinV6(float* rowBin, int pos, int feature, float value, int posStride, int featureStride) {
  rowBin[pos * posStride + feature * featureStride] = value;
}


//Calls f on each location that is part of an inescapable atari, or a group that can be put into inescapable atari
static void iterLadders(const Board& board, int nnXLen, std::function<void(Loc,int,const vector<Loc>&)> f) {
  int xSize = board.x_size;
  int ySize = board.y_size;

  Loc chainHeadsSolved[Board::MAX_PLAY_SIZE];
  bool chainHeadsSolvedValue[Board::MAX_PLAY_SIZE];
  int numChainHeadsSolved = 0;
  Board copy(board);
  vector<Loc> buf;
  vector<Loc> workingMoves;

  for(int y = 0; y<ySize; y++) {
    for(int x = 0; x<xSize; x++) {
      int pos = NNPos::xyToPos(x,y,nnXLen);
      Loc loc = Location::getLoc(x,y,xSize);
      Color stone = board.colors[loc];
      if(stone == P_BLACK || stone == P_WHITE) {
        int libs = board.getNumLiberties(loc);
        if(libs == 1 || libs == 2) {
          bool alreadySolved = false;
          Loc head = board.chain_head[loc];
          for(int i = 0; i<numChainHeadsSolved; i++) {
            if(chainHeadsSolved[i] == head) {
              alreadySolved = true;
              if(chainHeadsSolvedValue[i]) {
                workingMoves.clear();
                f(loc,pos,workingMoves);
              }
              break;
            }
          }
          if(!alreadySolved) {
            //Perform search on copy so as not to mess up tracking of solved heads
            bool laddered;
            if(libs == 1)
              laddered = copy.searchIsLadderCaptured(loc,true,buf);
            else {
              workingMoves.clear();
              laddered = copy.searchIsLadderCapturedAttackerFirst2Libs(loc,buf,workingMoves);
            }

            chainHeadsSolved[numChainHeadsSolved] = head;
            chainHeadsSolvedValue[numChainHeadsSolved] = laddered;
            numChainHeadsSolved++;
            if(laddered)
              f(loc,pos,workingMoves);
          }
        }
      }
    }
  }
}

//Currently does NOT depend on history (except for marking ko-illegal spots)
Hash128 NNInputs::getHash(
  const Board& board, const BoardHistory& hist, Player nextPlayer,
  const MiscNNInputParams& nnInputParams
) {
  int xSize = board.x_size;
  int ySize = board.y_size;

  //Note that board.pos_hash also incorporates the size of the board.
  Hash128 hash = board.pos_hash;
  hash ^= Board::ZOBRIST_PLAYER_HASH[nextPlayer];

  assert(hist.encorePhase >= 0 && hist.encorePhase <= 2);
  hash ^= Board::ZOBRIST_ENCORE_HASH[hist.encorePhase];

  if(hist.encorePhase == 0) {
    if(board.ko_loc != Board::NULL_LOC)
      hash ^= Board::ZOBRIST_KO_LOC_HASH[board.ko_loc];
    for(int y = 0; y<ySize; y++) {
      for(int x = 0; x<xSize; x++) {
        Loc loc = Location::getLoc(x,y,xSize);
        if(hist.superKoBanned[loc] && loc != board.ko_loc)
          hash ^= Board::ZOBRIST_KO_LOC_HASH[loc];
      }
    }
  }
  else {
    for(int y = 0; y<ySize; y++) {
      for(int x = 0; x<xSize; x++) {
        Loc loc = Location::getLoc(x,y,xSize);
        if(hist.superKoBanned[loc])
          hash ^= Board::ZOBRIST_KO_LOC_HASH[loc];
        if(hist.blackKoProhibited[loc])
          hash ^= Board::ZOBRIST_KO_MARK_HASH[loc][P_BLACK];
        if(hist.whiteKoProhibited[loc])
          hash ^= Board::ZOBRIST_KO_MARK_HASH[loc][P_WHITE];
      }
    }
  }

  float selfKomi = hist.currentSelfKomi(nextPlayer,nnInputParams.drawEquivalentWinsForWhite);

  //Discretize the komi for the purpose of matching hash, so that extremely close effective komi we just reuse nn cache hits
  int64_t komiDiscretized = (int64_t)(selfKomi*256.0f);
  uint64_t komiHash = Hash::murmurMix((uint64_t)komiDiscretized);
  hash.hash0 ^= komiHash;
  hash.hash1 ^= Hash::basicLCong(komiHash);

  //Fold in the ko, scoring, and suicide rules
  hash ^= Rules::ZOBRIST_KO_RULE_HASH[hist.rules.koRule];
  hash ^= Rules::ZOBRIST_SCORING_RULE_HASH[hist.rules.scoringRule];
  hash ^= Rules::ZOBRIST_TAX_RULE_HASH[hist.rules.taxRule];
  if(hist.rules.multiStoneSuicideLegal)
    hash ^= Rules::ZOBRIST_MULTI_STONE_SUICIDE_HASH;

  //Fold in whether a pass ends this phase
  bool passEndsPhase = hist.passWouldEndPhase(board,nextPlayer);
  if(passEndsPhase) {
    hash ^= Board::ZOBRIST_PASS_ENDS_PHASE;
    //And in the case that a pass ends the phase, conservativePass also affects the result
    if(nnInputParams.conservativePass)
      hash ^= MiscNNInputParams::ZOBRIST_CONSERVATIVE_PASS;
  }
  //Fold in whether the game is over or not, since this affects how we compute input features
  //but is not a function necessarily of previous hashed values.
  //If the history is in a weird prolonged state, also treat it similarly.
  if(hist.isGameFinished || hist.isPastNormalPhaseEnd)
    hash ^= Board::ZOBRIST_GAME_IS_OVER;

  return hash;
}

//===========================================================================================
//INPUTSVERSION 3
//===========================================================================================

void NNInputs::fillRowV3(
  const Board& board, const BoardHistory& hist, Player nextPlayer,
  const MiscNNInputParams& nnInputParams,
  int nnXLen, int nnYLen, bool useNHWC, float* rowBin, float* rowGlobal
) {
  assert(nnXLen <= NNPos::MAX_BOARD_LEN);
  assert(nnYLen <= NNPos::MAX_BOARD_LEN);
  assert(board.x_size <= nnXLen);
  assert(board.y_size <= nnYLen);
  std::fill(rowBin,rowBin+NUM_FEATURES_SPATIAL_V3*nnXLen*nnYLen,false);
  std::fill(rowGlobal,rowGlobal+NUM_FEATURES_GLOBAL_V3,0.0f);

  Player pla = nextPlayer;
  Player opp = getOpp(pla);
  int xSize = board.x_size;
  int ySize = board.y_size;

  int featureStride;
  int posStride;
  if(useNHWC) {
    featureStride = 1;
    posStride = NNInputs::NUM_FEATURES_SPATIAL_V3;
  }
  else {
    featureStride = nnXLen * nnYLen;
    posStride = 1;
  }

  for(int y = 0; y<ySize; y++) {
    for(int x = 0; x<xSize; x++) {
      int pos = NNPos::xyToPos(x,y,nnXLen);
      Loc loc = Location::getLoc(x,y,xSize);

      //Feature 0 - on board
      setRowBinV3(rowBin,pos,0, 1.0f, posStride, featureStride);

      Color stone = board.colors[loc];

      //Features 1,2 - pla,opp stone
      //Features 3,4,5 - 1,2,3 libs
      if(stone == pla)
        setRowBinV3(rowBin,pos,1, 1.0f, posStride, featureStride);
      else if(stone == opp)
        setRowBinV3(rowBin,pos,2, 1.0f, posStride, featureStride);

      if(stone == pla || stone == opp) {
        int libs = board.getNumLiberties(loc);
        if(libs == 1) setRowBinV3(rowBin,pos,3, 1.0f, posStride, featureStride);
        else if(libs == 2) setRowBinV3(rowBin,pos,4, 1.0f, posStride, featureStride);
        else if(libs == 3) setRowBinV3(rowBin,pos,5, 1.0f, posStride, featureStride);
      }
    }
  }

  //Feature 6 - ko-ban locations, including possibly superko.
  if(hist.encorePhase == 0) {
    if(board.ko_loc != Board::NULL_LOC) {
      int pos = NNPos::locToPos(board.ko_loc,xSize,nnXLen,nnYLen);
      setRowBinV3(rowBin,pos,6, 1.0f, posStride, featureStride);
    }
    for(int y = 0; y<ySize; y++) {
      for(int x = 0; x<xSize; x++) {
        Loc loc = Location::getLoc(x,y,xSize);
        if(hist.superKoBanned[loc] && loc != board.ko_loc) {
          int pos = NNPos::locToPos(loc,xSize,nnXLen,nnYLen);
          setRowBinV3(rowBin,pos,6, 1.0f, posStride, featureStride);
        }
      }
    }
  }
  else {
    //Feature 6,7,8 - in the encore, no-second-ko-capture locations, encore ko prohibitions where we have to pass for ko
    for(int y = 0; y<ySize; y++) {
      for(int x = 0; x<xSize; x++) {
        Loc loc = Location::getLoc(x,y,xSize);
        int pos = NNPos::locToPos(loc,xSize,nnXLen,nnYLen);
        if(hist.superKoBanned[loc])
          setRowBinV3(rowBin,pos,6, 1.0f, posStride, featureStride);
        if((pla == P_BLACK && hist.blackKoProhibited[loc]) || (pla == P_WHITE && hist.whiteKoProhibited[loc]))
          setRowBinV3(rowBin,pos,7, 1.0f, posStride, featureStride);
        if((pla == P_BLACK && hist.whiteKoProhibited[loc]) || (pla == P_WHITE && hist.blackKoProhibited[loc]))
          setRowBinV3(rowBin,pos,8, 1.0f, posStride, featureStride);
      }
    }
  }

  //Hide history from the net if a pass would end things and we're behaving as if a pass won't.
  //Or if the game is in fact over right now!
  bool hideHistory =
    hist.isGameFinished ||
    hist.isPastNormalPhaseEnd ||
    (nnInputParams.conservativePass && hist.passWouldEndGame(board,nextPlayer));

  //Features 9,10,11,12,13
  if(!hideHistory) {
    const vector<Move>& moveHistory = hist.moveHistory;
    size_t moveHistoryLen = moveHistory.size();
    if(moveHistoryLen >= 1 && moveHistory[moveHistoryLen-1].pla == opp) {
      Loc prev1Loc = moveHistory[moveHistoryLen-1].loc;
      if(prev1Loc == Board::PASS_LOC)
        rowGlobal[0] = 1.0;
      else if(prev1Loc != Board::NULL_LOC) {
        int pos = NNPos::locToPos(prev1Loc,xSize,nnXLen,nnYLen);
        setRowBinV3(rowBin,pos,9, 1.0f, posStride, featureStride);
      }
      if(moveHistoryLen >= 2 && moveHistory[moveHistoryLen-2].pla == pla) {
        Loc prev2Loc = moveHistory[moveHistoryLen-2].loc;
        if(prev2Loc == Board::PASS_LOC)
          rowGlobal[1] = 1.0;
        else if(prev2Loc != Board::NULL_LOC) {
          int pos = NNPos::locToPos(prev2Loc,xSize,nnXLen,nnYLen);
          setRowBinV3(rowBin,pos,10, 1.0f, posStride, featureStride);
        }
        if(moveHistoryLen >= 3 && moveHistory[moveHistoryLen-3].pla == opp) {
          Loc prev3Loc = moveHistory[moveHistoryLen-3].loc;
          if(prev3Loc == Board::PASS_LOC)
            rowGlobal[2] = 1.0;
          else if(prev3Loc != Board::NULL_LOC) {
            int pos = NNPos::locToPos(prev3Loc,xSize,nnXLen,nnYLen);
            setRowBinV3(rowBin,pos,11, 1.0f, posStride, featureStride);
          }
          if(moveHistoryLen >= 4 && moveHistory[moveHistoryLen-4].pla == pla) {
            Loc prev4Loc = moveHistory[moveHistoryLen-4].loc;
            if(prev4Loc == Board::PASS_LOC)
              rowGlobal[3] = 1.0;
            else if(prev4Loc != Board::NULL_LOC) {
              int pos = NNPos::locToPos(prev4Loc,xSize,nnXLen,nnYLen);
              setRowBinV3(rowBin,pos,12, 1.0f, posStride, featureStride);
            }
            if(moveHistoryLen >= 5 && moveHistory[moveHistoryLen-5].pla == opp) {
              Loc prev5Loc = moveHistory[moveHistoryLen-5].loc;
              if(prev5Loc == Board::PASS_LOC)
                rowGlobal[4] = 1.0;
              else if(prev5Loc != Board::NULL_LOC) {
                int pos = NNPos::locToPos(prev5Loc,xSize,nnXLen,nnYLen);
                setRowBinV3(rowBin,pos,13, 1.0f, posStride, featureStride);
              }
            }
          }
        }
      }
    }
  }

  //Ladder features 14,15,16,17
  auto addLadderFeature = [&board,xSize,nnXLen,nnYLen,posStride,featureStride,rowBin,opp](Loc loc, int pos, const vector<Loc>& workingMoves){
    assert(board.colors[loc] == P_BLACK || board.colors[loc] == P_WHITE);
    assert(pos >= 0 && pos < NNPos::MAX_BOARD_AREA);
    setRowBinV3(rowBin,pos,14, 1.0f, posStride, featureStride);
    if(board.colors[loc] == opp && board.getNumLiberties(loc) > 1) {
      for(size_t j = 0; j < workingMoves.size(); j++) {
        int workingPos = NNPos::locToPos(workingMoves[j],xSize,nnXLen,nnYLen);
        setRowBinV3(rowBin,workingPos,17, 1.0f, posStride, featureStride);
      }
    }
  };

  iterLadders(board, nnXLen, addLadderFeature);

  const Board& prevBoard = hideHistory ? board : hist.getRecentBoard(1);
  auto addPrevLadderFeature = [&prevBoard,posStride,featureStride,rowBin](Loc loc, int pos, const vector<Loc>& workingMoves){
    (void)workingMoves;
    (void)loc;
    assert(prevBoard.colors[loc] == P_BLACK || prevBoard.colors[loc] == P_WHITE);
    assert(pos >= 0 && pos < NNPos::MAX_BOARD_AREA);
    setRowBinV3(rowBin,pos,15, 1.0f, posStride, featureStride);
  };
  iterLadders(prevBoard, nnXLen, addPrevLadderFeature);

  const Board& prevPrevBoard = hideHistory ? board : hist.getRecentBoard(2);
  auto addPrevPrevLadderFeature = [&prevPrevBoard,posStride,featureStride,rowBin](Loc loc, int pos, const vector<Loc>& workingMoves){
    (void)workingMoves;
    (void)loc;
    assert(prevPrevBoard.colors[loc] == P_BLACK || prevPrevBoard.colors[loc] == P_WHITE);
    assert(pos >= 0 && pos < NNPos::MAX_BOARD_AREA);
    setRowBinV3(rowBin,pos,16, 1.0f, posStride, featureStride);
  };
  iterLadders(prevPrevBoard, nnXLen, addPrevPrevLadderFeature);

  //Features 18,19 - current territory
  Color area[Board::MAX_ARR_SIZE];
  bool nonPassAliveStones;
  bool safeBigTerritories;
  bool unsafeBigTerritories;
  if(hist.rules.scoringRule == Rules::SCORING_AREA) {
    nonPassAliveStones = true;
    safeBigTerritories = true;
    unsafeBigTerritories = true;
  }
  else if(hist.rules.scoringRule == Rules::SCORING_TERRITORY) {
    nonPassAliveStones = false;
    safeBigTerritories = true;
    unsafeBigTerritories = false;
  }
  else {
    ASSERT_UNREACHABLE;
  }
  board.calculateArea(area,nonPassAliveStones,safeBigTerritories,unsafeBigTerritories,hist.rules.multiStoneSuicideLegal);

  for(int y = 0; y<ySize; y++) {
    for(int x = 0; x<xSize; x++) {
      Loc loc = Location::getLoc(x,y,xSize);
      int pos = NNPos::locToPos(loc,xSize,nnXLen,nnYLen);
      if(area[loc] == pla)
        setRowBinV3(rowBin,pos,18, 1.0f, posStride, featureStride);
      else if(area[loc] == opp)
        setRowBinV3(rowBin,pos,19, 1.0f, posStride, featureStride);
    }
  }

  //Features 20, 21 - second encore starting stones
  if(hist.encorePhase >= 2) {
    for(int y = 0; y<ySize; y++) {
      for(int x = 0; x<xSize; x++) {
        Loc loc = Location::getLoc(x,y,xSize);
        int pos = NNPos::locToPos(loc,xSize,nnXLen,nnYLen);
        if(hist.secondEncoreStartColors[loc] == pla)
          setRowBinV3(rowBin,pos,20, 1.0f, posStride, featureStride);
        else if(hist.secondEncoreStartColors[loc] == opp)
          setRowBinV3(rowBin,pos,21, 1.0f, posStride, featureStride);
      }
    }
  }


  //Global features.
  //The first 5 of them were set already above to flag which of the past 5 moves were passes.

  //Komi and any score adjustments
  float selfKomi = hist.currentSelfKomi(nextPlayer,nnInputParams.drawEquivalentWinsForWhite);
  float bArea = xSize * ySize;
  //Bound komi just in case
  if(selfKomi > bArea+1.0f)
    selfKomi = bArea+1.0f;
  if(selfKomi < -bArea-1.0f)
    selfKomi = -bArea-1.0f;
  rowGlobal[5] = selfKomi/15.0f;

  //Ko rule
  if(hist.rules.koRule == Rules::KO_SIMPLE) {}
  else if(hist.rules.koRule == Rules::KO_POSITIONAL || hist.rules.koRule == Rules::KO_SPIGHT) {
    rowGlobal[6] = 1.0f;
    rowGlobal[7] = 0.5f;
  }
  else if(hist.rules.koRule == Rules::KO_SITUATIONAL) {
    rowGlobal[6] = 1.0f;
    rowGlobal[7] = -0.5f;
  }
  else
    ASSERT_UNREACHABLE;

  //Suicide
  if(hist.rules.multiStoneSuicideLegal)
    rowGlobal[8] = 1.0f;

  //Scoring
  if(hist.rules.scoringRule == Rules::SCORING_AREA) {}
  else if(hist.rules.scoringRule == Rules::SCORING_TERRITORY)
    rowGlobal[9] = 1.0f;
  else
    ASSERT_UNREACHABLE;

  //Encore phase
  if(hist.encorePhase > 0)
    rowGlobal[10] = 1.0f;
  if(hist.encorePhase > 1)
    rowGlobal[11] = 1.0f;

  //Does a pass end the current phase given the ruleset and history?
  bool passWouldEndPhase = hideHistory ? false : hist.passWouldEndPhase(board,nextPlayer);
  rowGlobal[12] = passWouldEndPhase ? 1.0f : 0.0f;

  //Provide parity information about the board size and komi
  //This comes from the following observation:
  //From white's perspective:
  //Komi = 0.0 - Draw possible
  //Komi = 0.5 - Win the games we would have drawn with komi 0.0
  //Komi = 1.0 - Usually no difference from komi 0.5
  //Komi = 1.5 - Usually no difference from komi 0.5
  //Komi = 2.0 - Draw possible
  //If we were to assign an "effective goodness" to these komis in order it would look like
  //0 1 1 1 2 3 3 3 4 5 5 5 6 ...
  //since when away from the right parity, increasing the komi doesn't help us except in cases of seki with odd numbers of dame.
  //If we were to add 0.5 times a vector like:
  //0 -1 0 1 0 -1 0 1 0 -1 0 ...
  //Then this would become a linear function and hopefully easier for a neural net to learn.
  //We expect that this is hard for a neural net to learn since it depends on the parity of the board size
  //and is very "xor"like.
  //So we provide it as an input.
  //Since we are using a model where games are jittered by 0.5 (see BoardHistory::whiteKomiAdjustmentForDraws)
  //in theory right thing to first order to provide should be a triangular wave with a period of 2 komi points:
  //  ../\........
  //  ./..\.......
  //  /....\..../.
  //  ......\../..
  //  .......\/...
  //The upsloping part of the wave is centered around the komi value where you could draw
  //since komi is extra valuable when it turns losses into draws into wins, peaking at the komi value where you could draw + 0.5.
  //It's downsloping around the komi value where you can't draw, since the marginal komi there is nearly useless, not causing you to win
  //more games except in case of odd-dame seki.

  if(hist.rules.scoringRule == Rules::SCORING_AREA || hist.encorePhase >= 2) {
    bool boardAreaIsEven = (xSize*ySize) % 2 == 0;

    //What is the parity of the komi values that can produce jigos?
    bool drawableKomisAreEven = boardAreaIsEven;

    //Find the difference between the komi viewed from our perspective and the nearest drawable komi below it.
    float komiFloor;
    if(drawableKomisAreEven)
      komiFloor = floor(selfKomi / 2.0f) * 2.0f;
    else
      komiFloor = floor((selfKomi-1.0f) / 2.0f) * 2.0f + 1.0f;

    //Cap just in case we have floating point weirdness
    float delta = selfKomi - komiFloor;
    assert(delta >= -0.0001f);
    assert(delta <= 2.0001f);
    if(delta < 0.0f)
      delta = 0.0f;
    if(delta > 2.0f)
      delta = 2.0f;

    //Create the triangle wave based on the difference
    float wave;
    if(delta < 0.5f)
      wave = delta;
    else if(delta < 1.5f)
      wave = 1.0f-delta;
    else
      wave = delta-2.0f;

    //NOTE: If ever changing which feature this is, must also update index in model.py where we multiply it into the scorebelief parity vector
    rowGlobal[13] = wave;
  }

}


//===========================================================================================
//INPUTSVERSION 4
//===========================================================================================

void NNInputs::fillRowV4(
  const Board& board, const BoardHistory& hist, Player nextPlayer,
  const MiscNNInputParams& nnInputParams,
  int nnXLen, int nnYLen, bool useNHWC, float* rowBin, float* rowGlobal
) {
  assert(nnXLen <= NNPos::MAX_BOARD_LEN);
  assert(nnYLen <= NNPos::MAX_BOARD_LEN);
  assert(board.x_size <= nnXLen);
  assert(board.y_size <= nnYLen);
  std::fill(rowBin,rowBin+NUM_FEATURES_SPATIAL_V4*nnXLen*nnYLen,false);
  std::fill(rowGlobal,rowGlobal+NUM_FEATURES_GLOBAL_V4,0.0f);

  Player pla = nextPlayer;
  Player opp = getOpp(pla);
  int xSize = board.x_size;
  int ySize = board.y_size;

  int featureStride;
  int posStride;
  if(useNHWC) {
    featureStride = 1;
    posStride = NNInputs::NUM_FEATURES_SPATIAL_V4;
  }
  else {
    featureStride = nnXLen * nnYLen;
    posStride = 1;
  }

  for(int y = 0; y<ySize; y++) {
    for(int x = 0; x<xSize; x++) {
      int pos = NNPos::xyToPos(x,y,nnXLen);
      Loc loc = Location::getLoc(x,y,xSize);

      //Feature 0 - on board
      setRowBinV4(rowBin,pos,0, 1.0f, posStride, featureStride);

      Color stone = board.colors[loc];

      //Features 1,2 - pla,opp stone
      //Features 3,4,5 - 1,2,3 libs
      if(stone == pla)
        setRowBinV4(rowBin,pos,1, 1.0f, posStride, featureStride);
      else if(stone == opp)
        setRowBinV4(rowBin,pos,2, 1.0f, posStride, featureStride);

      if(stone == pla || stone == opp) {
        int libs = board.getNumLiberties(loc);
        if(libs == 1) setRowBinV4(rowBin,pos,3, 1.0f, posStride, featureStride);
        else if(libs == 2) setRowBinV4(rowBin,pos,4, 1.0f, posStride, featureStride);
        else if(libs == 3) setRowBinV4(rowBin,pos,5, 1.0f, posStride, featureStride);
      }
    }
  }

  //Feature 6 - ko-ban locations, including possibly superko.
  if(hist.encorePhase == 0) {
    if(board.ko_loc != Board::NULL_LOC) {
      int pos = NNPos::locToPos(board.ko_loc,xSize,nnXLen,nnYLen);
      setRowBinV4(rowBin,pos,6, 1.0f, posStride, featureStride);
    }
    for(int y = 0; y<ySize; y++) {
      for(int x = 0; x<xSize; x++) {
        Loc loc = Location::getLoc(x,y,xSize);
        if(hist.superKoBanned[loc] && loc != board.ko_loc) {
          int pos = NNPos::locToPos(loc,xSize,nnXLen,nnYLen);
          setRowBinV4(rowBin,pos,6, 1.0f, posStride, featureStride);
        }
      }
    }
  }
  else {
    //Feature 6,7,8 - in the encore, no-second-ko-capture locations, encore ko prohibitions where we have to pass for ko
    for(int y = 0; y<ySize; y++) {
      for(int x = 0; x<xSize; x++) {
        Loc loc = Location::getLoc(x,y,xSize);
        int pos = NNPos::locToPos(loc,xSize,nnXLen,nnYLen);
        if(hist.superKoBanned[loc])
          setRowBinV4(rowBin,pos,6, 1.0f, posStride, featureStride);
        if((pla == P_BLACK && hist.blackKoProhibited[loc]) || (pla == P_WHITE && hist.whiteKoProhibited[loc]))
          setRowBinV4(rowBin,pos,7, 1.0f, posStride, featureStride);
        if((pla == P_BLACK && hist.whiteKoProhibited[loc]) || (pla == P_WHITE && hist.blackKoProhibited[loc]))
          setRowBinV4(rowBin,pos,8, 1.0f, posStride, featureStride);
      }
    }
  }

  //Hide history from the net if a pass would end things and we're behaving as if a pass won't.
  //Or if the game is in fact over right now!
  bool hideHistory =
    hist.isGameFinished ||
    hist.isPastNormalPhaseEnd ||
    (nnInputParams.conservativePass && hist.passWouldEndGame(board,nextPlayer));

  //Features 9,10,11,12,13
  if(!hideHistory) {
    const vector<Move>& moveHistory = hist.moveHistory;
    size_t moveHistoryLen = moveHistory.size();
    if(moveHistoryLen >= 1 && moveHistory[moveHistoryLen-1].pla == opp) {
      Loc prev1Loc = moveHistory[moveHistoryLen-1].loc;
      if(prev1Loc == Board::PASS_LOC)
        rowGlobal[0] = 1.0;
      else if(prev1Loc != Board::NULL_LOC) {
        int pos = NNPos::locToPos(prev1Loc,xSize,nnXLen,nnYLen);
        setRowBinV4(rowBin,pos,9, 1.0f, posStride, featureStride);
      }
      if(moveHistoryLen >= 2 && moveHistory[moveHistoryLen-2].pla == pla) {
        Loc prev2Loc = moveHistory[moveHistoryLen-2].loc;
        if(prev2Loc == Board::PASS_LOC)
          rowGlobal[1] = 1.0;
        else if(prev2Loc != Board::NULL_LOC) {
          int pos = NNPos::locToPos(prev2Loc,xSize,nnXLen,nnYLen);
          setRowBinV4(rowBin,pos,10, 1.0f, posStride, featureStride);
        }
        if(moveHistoryLen >= 3 && moveHistory[moveHistoryLen-3].pla == opp) {
          Loc prev3Loc = moveHistory[moveHistoryLen-3].loc;
          if(prev3Loc == Board::PASS_LOC)
            rowGlobal[2] = 1.0;
          else if(prev3Loc != Board::NULL_LOC) {
            int pos = NNPos::locToPos(prev3Loc,xSize,nnXLen,nnYLen);
            setRowBinV4(rowBin,pos,11, 1.0f, posStride, featureStride);
          }
          if(moveHistoryLen >= 4 && moveHistory[moveHistoryLen-4].pla == pla) {
            Loc prev4Loc = moveHistory[moveHistoryLen-4].loc;
            if(prev4Loc == Board::PASS_LOC)
              rowGlobal[3] = 1.0;
            else if(prev4Loc != Board::NULL_LOC) {
              int pos = NNPos::locToPos(prev4Loc,xSize,nnXLen,nnYLen);
              setRowBinV4(rowBin,pos,12, 1.0f, posStride, featureStride);
            }
            if(moveHistoryLen >= 5 && moveHistory[moveHistoryLen-5].pla == opp) {
              Loc prev5Loc = moveHistory[moveHistoryLen-5].loc;
              if(prev5Loc == Board::PASS_LOC)
                rowGlobal[4] = 1.0;
              else if(prev5Loc != Board::NULL_LOC) {
                int pos = NNPos::locToPos(prev5Loc,xSize,nnXLen,nnYLen);
                setRowBinV4(rowBin,pos,13, 1.0f, posStride, featureStride);
              }
            }
          }
        }
      }
    }
  }

  //Ladder features 14,15,16,17
  auto addLadderFeature = [&board,xSize,nnXLen,nnYLen,posStride,featureStride,rowBin,opp](Loc loc, int pos, const vector<Loc>& workingMoves){
    assert(board.colors[loc] == P_BLACK || board.colors[loc] == P_WHITE);
    assert(pos >= 0 && pos < NNPos::MAX_BOARD_AREA);
    setRowBinV4(rowBin,pos,14, 1.0f, posStride, featureStride);
    if(board.colors[loc] == opp && board.getNumLiberties(loc) > 1) {
      for(size_t j = 0; j < workingMoves.size(); j++) {
        int workingPos = NNPos::locToPos(workingMoves[j],xSize,nnXLen,nnYLen);
        setRowBinV4(rowBin,workingPos,17, 1.0f, posStride, featureStride);
      }
    }
  };

  iterLadders(board, nnXLen, addLadderFeature);

  const Board& prevBoard = hideHistory ? board : hist.getRecentBoard(1);
  auto addPrevLadderFeature = [&prevBoard,posStride,featureStride,rowBin](Loc loc, int pos, const vector<Loc>& workingMoves){
    (void)workingMoves;
    (void)loc;
    assert(prevBoard.colors[loc] == P_BLACK || prevBoard.colors[loc] == P_WHITE);
    assert(pos >= 0 && pos < NNPos::MAX_BOARD_AREA);
    setRowBinV4(rowBin,pos,15, 1.0f, posStride, featureStride);
  };
  iterLadders(prevBoard, nnXLen, addPrevLadderFeature);

  const Board& prevPrevBoard = hideHistory ? board : hist.getRecentBoard(2);
  auto addPrevPrevLadderFeature = [&prevPrevBoard,posStride,featureStride,rowBin](Loc loc, int pos, const vector<Loc>& workingMoves){
    (void)workingMoves;
    (void)loc;
    assert(prevPrevBoard.colors[loc] == P_BLACK || prevPrevBoard.colors[loc] == P_WHITE);
    assert(pos >= 0 && pos < NNPos::MAX_BOARD_AREA);
    setRowBinV4(rowBin,pos,16, 1.0f, posStride, featureStride);
  };
  iterLadders(prevPrevBoard, nnXLen, addPrevPrevLadderFeature);

  //Features 18,19 - pass alive territory and stones
  Color area[Board::MAX_ARR_SIZE];
  {
    bool nonPassAliveStones = false;
    bool safeBigTerritories = true;
    bool unsafeBigTerritories = false;
    board.calculateArea(area,nonPassAliveStones,safeBigTerritories,unsafeBigTerritories,hist.rules.multiStoneSuicideLegal);
  }

  for(int y = 0; y<ySize; y++) {
    for(int x = 0; x<xSize; x++) {
      Loc loc = Location::getLoc(x,y,xSize);
      int pos = NNPos::locToPos(loc,xSize,nnXLen,nnYLen);
      if(area[loc] == pla)
        setRowBinV4(rowBin,pos,18, 1.0f, posStride, featureStride);
      else if(area[loc] == opp)
        setRowBinV4(rowBin,pos,19, 1.0f, posStride, featureStride);
    }
  }

  //Features 20, 21 - second encore starting stones
  if(hist.encorePhase >= 2) {
    for(int y = 0; y<ySize; y++) {
      for(int x = 0; x<xSize; x++) {
        Loc loc = Location::getLoc(x,y,xSize);
        int pos = NNPos::locToPos(loc,xSize,nnXLen,nnYLen);
        if(hist.secondEncoreStartColors[loc] == pla)
          setRowBinV4(rowBin,pos,20, 1.0f, posStride, featureStride);
        else if(hist.secondEncoreStartColors[loc] == opp)
          setRowBinV4(rowBin,pos,21, 1.0f, posStride, featureStride);
      }
    }
  }


  //Global features.
  //The first 5 of them were set already above to flag which of the past 5 moves were passes.

  //Komi and any score adjustments
  float selfKomi = hist.currentSelfKomi(nextPlayer,nnInputParams.drawEquivalentWinsForWhite);
  float bArea = xSize * ySize;
  //Bound komi just in case
  if(selfKomi > bArea+1.0f)
    selfKomi = bArea+1.0f;
  if(selfKomi < -bArea-1.0f)
    selfKomi = -bArea-1.0f;
  rowGlobal[5] = selfKomi/15.0f;

  //Ko rule
  if(hist.rules.koRule == Rules::KO_SIMPLE) {}
  else if(hist.rules.koRule == Rules::KO_POSITIONAL || hist.rules.koRule == Rules::KO_SPIGHT) {
    rowGlobal[6] = 1.0f;
    rowGlobal[7] = 0.5f;
  }
  else if(hist.rules.koRule == Rules::KO_SITUATIONAL) {
    rowGlobal[6] = 1.0f;
    rowGlobal[7] = -0.5f;
  }
  else
    ASSERT_UNREACHABLE;

  //Suicide
  if(hist.rules.multiStoneSuicideLegal)
    rowGlobal[8] = 1.0f;

  //Scoring
  if(hist.rules.scoringRule == Rules::SCORING_AREA) {}
  else if(hist.rules.scoringRule == Rules::SCORING_TERRITORY)
    rowGlobal[9] = 1.0f;
  else
    ASSERT_UNREACHABLE;

  //Encore phase
  if(hist.encorePhase > 0)
    rowGlobal[10] = 1.0f;
  if(hist.encorePhase > 1)
    rowGlobal[11] = 1.0f;

  //Does a pass end the current phase given the ruleset and history?
  bool passWouldEndPhase = hideHistory ? false : hist.passWouldEndPhase(board,nextPlayer);
  rowGlobal[12] = passWouldEndPhase ? 1.0f : 0.0f;

  //Provide parity information about the board size and komi
  //This comes from the following observation:
  //From white's perspective:
  //Komi = 0.0 - Draw possible
  //Komi = 0.5 - Win the games we would have drawn with komi 0.0
  //Komi = 1.0 - Usually no difference from komi 0.5
  //Komi = 1.5 - Usually no difference from komi 0.5
  //Komi = 2.0 - Draw possible
  //If we were to assign an "effective goodness" to these komis in order it would look like
  //0 1 1 1 2 3 3 3 4 5 5 5 6 ...
  //since when away from the right parity, increasing the komi doesn't help us except in cases of seki with odd numbers of dame.
  //If we were to add 0.5 times a vector like:
  //0 -1 0 1 0 -1 0 1 0 -1 0 ...
  //Then this would become a linear function and hopefully easier for a neural net to learn.
  //We expect that this is hard for a neural net to learn since it depends on the parity of the board size
  //and is very "xor"like.
  //So we provide it as an input.
  //Since we are using a model where games are jittered by 0.5 (see BoardHistory::whiteKomiAdjustmentForDraws)
  //in theory right thing to first order to provide should be a triangular wave with a period of 2 komi points:
  //  ../\........
  //  ./..\.......
  //  /....\..../.
  //  ......\../..
  //  .......\/...
  //The upsloping part of the wave is centered around the komi value where you could draw
  //since komi is extra valuable when it turns losses into draws into wins, peaking at the komi value where you could draw + 0.5.
  //It's downsloping around the komi value where you can't draw, since the marginal komi there is nearly useless, not causing you to win
  //more games except in case of odd-dame seki.

  if(hist.rules.scoringRule == Rules::SCORING_AREA || hist.encorePhase >= 2) {
    bool boardAreaIsEven = (xSize*ySize) % 2 == 0;

    //What is the parity of the komi values that can produce jigos?
    bool drawableKomisAreEven = boardAreaIsEven;

    //Find the difference between the komi viewed from our perspective and the nearest drawable komi below it.
    float komiFloor;
    if(drawableKomisAreEven)
      komiFloor = floor(selfKomi / 2.0f) * 2.0f;
    else
      komiFloor = floor((selfKomi-1.0f) / 2.0f) * 2.0f + 1.0f;

    //Cap just in case we have floating point weirdness
    float delta = selfKomi - komiFloor;
    assert(delta >= -0.0001f);
    assert(delta <= 2.0001f);
    if(delta < 0.0f)
      delta = 0.0f;
    if(delta > 2.0f)
      delta = 2.0f;

    //Create the triangle wave based on the difference
    float wave;
    if(delta < 0.5f)
      wave = delta;
    else if(delta < 1.5f)
      wave = 1.0f-delta;
    else
      wave = delta-2.0f;

    //NOTE: If ever changing which feature this is, must also update index in model.py where we multiply it into the scorebelief parity vector
    rowGlobal[13] = wave;
  }

}



//===========================================================================================
//INPUTSVERSION 5
//===========================================================================================

void NNInputs::fillRowV5(
  const Board& board, const BoardHistory& hist, Player nextPlayer,
  const MiscNNInputParams& nnInputParams,
  int nnXLen, int nnYLen, bool useNHWC, float* rowBin, float* rowGlobal
) {
  assert(nnXLen <= NNPos::MAX_BOARD_LEN);
  assert(nnYLen <= NNPos::MAX_BOARD_LEN);
  assert(board.x_size <= nnXLen);
  assert(board.y_size <= nnYLen);
  std::fill(rowBin,rowBin+NUM_FEATURES_SPATIAL_V5*nnXLen*nnYLen,false);
  std::fill(rowGlobal,rowGlobal+NUM_FEATURES_GLOBAL_V5,0.0f);

  Player pla = nextPlayer;
  Player opp = getOpp(pla);
  int xSize = board.x_size;
  int ySize = board.y_size;

  int featureStride;
  int posStride;
  if(useNHWC) {
    featureStride = 1;
    posStride = NNInputs::NUM_FEATURES_SPATIAL_V5;
  }
  else {
    featureStride = nnXLen * nnYLen;
    posStride = 1;
  }

  for(int y = 0; y<ySize; y++) {
    for(int x = 0; x<xSize; x++) {
      int pos = NNPos::xyToPos(x,y,nnXLen);
      Loc loc = Location::getLoc(x,y,xSize);

      //Feature 0 - on board
      setRowBinV5(rowBin,pos,0, 1.0f, posStride, featureStride);

      Color stone = board.colors[loc];

      //Features 1,2 - pla,opp stone
      if(stone == pla)
        setRowBinV5(rowBin,pos,1, 1.0f, posStride, featureStride);
      else if(stone == opp)
        setRowBinV5(rowBin,pos,2, 1.0f, posStride, featureStride);
    }
  }

  //Feature 3 - ko-ban locations, including possibly superko.
  if(hist.encorePhase == 0) {
    if(board.ko_loc != Board::NULL_LOC) {
      int pos = NNPos::locToPos(board.ko_loc,xSize,nnXLen,nnYLen);
      setRowBinV5(rowBin,pos,3, 1.0f, posStride, featureStride);
    }
    for(int y = 0; y<ySize; y++) {
      for(int x = 0; x<xSize; x++) {
        Loc loc = Location::getLoc(x,y,xSize);
        if(hist.superKoBanned[loc] && loc != board.ko_loc) {
          int pos = NNPos::locToPos(loc,xSize,nnXLen,nnYLen);
          setRowBinV5(rowBin,pos,3, 1.0f, posStride, featureStride);
        }
      }
    }
  }
  else {
    //Feature 3,4,5 - in the encore, no-second-ko-capture locations, encore ko prohibitions where we have to pass for ko
    for(int y = 0; y<ySize; y++) {
      for(int x = 0; x<xSize; x++) {
        Loc loc = Location::getLoc(x,y,xSize);
        int pos = NNPos::locToPos(loc,xSize,nnXLen,nnYLen);
        if(hist.superKoBanned[loc])
          setRowBinV5(rowBin,pos,3, 1.0f, posStride, featureStride);
        if((pla == P_BLACK && hist.blackKoProhibited[loc]) || (pla == P_WHITE && hist.whiteKoProhibited[loc]))
          setRowBinV5(rowBin,pos,4, 1.0f, posStride, featureStride);
        if((pla == P_BLACK && hist.whiteKoProhibited[loc]) || (pla == P_WHITE && hist.blackKoProhibited[loc]))
          setRowBinV5(rowBin,pos,5, 1.0f, posStride, featureStride);
      }
    }
  }

  //Hide history from the net if a pass would end things and we're behaving as if a pass won't.
  //Or if the game is in fact over right now!
  bool hideHistory =
    hist.isGameFinished ||
    hist.isPastNormalPhaseEnd ||
    (nnInputParams.conservativePass && hist.passWouldEndGame(board,nextPlayer));

  //Features 6,7,8,9,10
  if(!hideHistory) {
    const vector<Move>& moveHistory = hist.moveHistory;
    size_t moveHistoryLen = moveHistory.size();
    if(moveHistoryLen >= 1 && moveHistory[moveHistoryLen-1].pla == opp) {
      Loc prev1Loc = moveHistory[moveHistoryLen-1].loc;
      if(prev1Loc == Board::PASS_LOC)
        rowGlobal[0] = 1.0;
      else if(prev1Loc != Board::NULL_LOC) {
        int pos = NNPos::locToPos(prev1Loc,xSize,nnXLen,nnYLen);
        setRowBinV5(rowBin,pos,6, 1.0f, posStride, featureStride);
      }
      if(moveHistoryLen >= 2 && moveHistory[moveHistoryLen-2].pla == pla) {
        Loc prev2Loc = moveHistory[moveHistoryLen-2].loc;
        if(prev2Loc == Board::PASS_LOC)
          rowGlobal[1] = 1.0;
        else if(prev2Loc != Board::NULL_LOC) {
          int pos = NNPos::locToPos(prev2Loc,xSize,nnXLen,nnYLen);
          setRowBinV5(rowBin,pos,7, 1.0f, posStride, featureStride);
        }
        if(moveHistoryLen >= 3 && moveHistory[moveHistoryLen-3].pla == opp) {
          Loc prev3Loc = moveHistory[moveHistoryLen-3].loc;
          if(prev3Loc == Board::PASS_LOC)
            rowGlobal[2] = 1.0;
          else if(prev3Loc != Board::NULL_LOC) {
            int pos = NNPos::locToPos(prev3Loc,xSize,nnXLen,nnYLen);
            setRowBinV5(rowBin,pos,8, 1.0f, posStride, featureStride);
          }
          if(moveHistoryLen >= 4 && moveHistory[moveHistoryLen-4].pla == pla) {
            Loc prev4Loc = moveHistory[moveHistoryLen-4].loc;
            if(prev4Loc == Board::PASS_LOC)
              rowGlobal[3] = 1.0;
            else if(prev4Loc != Board::NULL_LOC) {
              int pos = NNPos::locToPos(prev4Loc,xSize,nnXLen,nnYLen);
              setRowBinV5(rowBin,pos,9, 1.0f, posStride, featureStride);
            }
            if(moveHistoryLen >= 5 && moveHistory[moveHistoryLen-5].pla == opp) {
              Loc prev5Loc = moveHistory[moveHistoryLen-5].loc;
              if(prev5Loc == Board::PASS_LOC)
                rowGlobal[4] = 1.0;
              else if(prev5Loc != Board::NULL_LOC) {
                int pos = NNPos::locToPos(prev5Loc,xSize,nnXLen,nnYLen);
                setRowBinV5(rowBin,pos,10, 1.0f, posStride, featureStride);
              }
            }
          }
        }
      }
    }
  }

  //Features 11, 12 - second encore starting stones
  if(hist.encorePhase >= 2) {
    for(int y = 0; y<ySize; y++) {
      for(int x = 0; x<xSize; x++) {
        Loc loc = Location::getLoc(x,y,xSize);
        int pos = NNPos::locToPos(loc,xSize,nnXLen,nnYLen);
        if(hist.secondEncoreStartColors[loc] == pla)
          setRowBinV5(rowBin,pos,11, 1.0f, posStride, featureStride);
        else if(hist.secondEncoreStartColors[loc] == opp)
          setRowBinV5(rowBin,pos,12, 1.0f, posStride, featureStride);
      }
    }
  }


  //Global features.
  //The first 5 of them were set already above to flag which of the past 5 moves were passes.

  //Komi and any score adjustments
  float selfKomi = hist.currentSelfKomi(nextPlayer,nnInputParams.drawEquivalentWinsForWhite);
  float bArea = xSize * ySize;
  //Bound komi just in case
  if(selfKomi > bArea+1.0f)
    selfKomi = bArea+1.0f;
  if(selfKomi < -bArea-1.0f)
    selfKomi = -bArea-1.0f;
  rowGlobal[5] = selfKomi/15.0f;

  //Ko rule
  if(hist.rules.koRule == Rules::KO_SIMPLE) {}
  else if(hist.rules.koRule == Rules::KO_POSITIONAL || hist.rules.koRule == Rules::KO_SPIGHT) {
    rowGlobal[6] = 1.0f;
    rowGlobal[7] = 0.5f;
  }
  else if(hist.rules.koRule == Rules::KO_SITUATIONAL) {
    rowGlobal[6] = 1.0f;
    rowGlobal[7] = -0.5f;
  }
  else
    ASSERT_UNREACHABLE;

  //Suicide
  if(hist.rules.multiStoneSuicideLegal)
    rowGlobal[8] = 1.0f;

  //Scoring
  if(hist.rules.scoringRule == Rules::SCORING_AREA) {}
  else if(hist.rules.scoringRule == Rules::SCORING_TERRITORY)
    rowGlobal[9] = 1.0f;
  else
    ASSERT_UNREACHABLE;

  //Encore phase
  if(hist.encorePhase > 0)
    rowGlobal[10] = 1.0f;
  if(hist.encorePhase > 1)
    rowGlobal[11] = 1.0f;

}

//===========================================================================================
//INPUTSVERSION 6
//===========================================================================================


void NNInputs::fillRowV6(
  const Board& board, const BoardHistory& hist, Player nextPlayer,
  const MiscNNInputParams& nnInputParams,
  int nnXLen, int nnYLen, bool useNHWC, float* rowBin, float* rowGlobal
) {
  assert(nnXLen <= NNPos::MAX_BOARD_LEN);
  assert(nnYLen <= NNPos::MAX_BOARD_LEN);
  assert(board.x_size <= nnXLen);
  assert(board.y_size <= nnYLen);
  std::fill(rowBin,rowBin+NUM_FEATURES_SPATIAL_V6*nnXLen*nnYLen,false);
  std::fill(rowGlobal,rowGlobal+NUM_FEATURES_GLOBAL_V6,0.0f);

  Player pla = nextPlayer;
  Player opp = getOpp(pla);
  int xSize = board.x_size;
  int ySize = board.y_size;

  int featureStride;
  int posStride;
  if(useNHWC) {
    featureStride = 1;
    posStride = NNInputs::NUM_FEATURES_SPATIAL_V6;
  }
  else {
    featureStride = nnXLen * nnYLen;
    posStride = 1;
  }

  for(int y = 0; y<ySize; y++) {
    for(int x = 0; x<xSize; x++) {
      int pos = NNPos::xyToPos(x,y,nnXLen);
      Loc loc = Location::getLoc(x,y,xSize);

      //Feature 0 - on board
      setRowBinV6(rowBin,pos,0, 1.0f, posStride, featureStride);

      Color stone = board.colors[loc];

      //Features 1,2 - pla,opp stone
      //Features 3,4,5 - 1,2,3 libs
      if(stone == pla)
        setRowBinV6(rowBin,pos,1, 1.0f, posStride, featureStride);
      else if(stone == opp)
        setRowBinV6(rowBin,pos,2, 1.0f, posStride, featureStride);

      if(stone == pla || stone == opp) {
        int libs = board.getNumLiberties(loc);
        if(libs == 1) setRowBinV6(rowBin,pos,3, 1.0f, posStride, featureStride);
        else if(libs == 2) setRowBinV6(rowBin,pos,4, 1.0f, posStride, featureStride);
        else if(libs == 3) setRowBinV6(rowBin,pos,5, 1.0f, posStride, featureStride);
      }
    }
  }

  //Feature 6 - ko-ban locations, including possibly superko.
  if(hist.encorePhase == 0) {
    if(board.ko_loc != Board::NULL_LOC) {
      int pos = NNPos::locToPos(board.ko_loc,xSize,nnXLen,nnYLen);
      setRowBinV6(rowBin,pos,6, 1.0f, posStride, featureStride);
    }
    for(int y = 0; y<ySize; y++) {
      for(int x = 0; x<xSize; x++) {
        Loc loc = Location::getLoc(x,y,xSize);
        if(hist.superKoBanned[loc] && loc != board.ko_loc) {
          int pos = NNPos::locToPos(loc,xSize,nnXLen,nnYLen);
          setRowBinV6(rowBin,pos,6, 1.0f, posStride, featureStride);
        }
      }
    }
  }
  else {
    //Feature 6,7,8 - in the encore, no-second-ko-capture locations, encore ko prohibitions where we have to pass for ko
    for(int y = 0; y<ySize; y++) {
      for(int x = 0; x<xSize; x++) {
        Loc loc = Location::getLoc(x,y,xSize);
        int pos = NNPos::locToPos(loc,xSize,nnXLen,nnYLen);
        if(hist.superKoBanned[loc])
          setRowBinV6(rowBin,pos,6, 1.0f, posStride, featureStride);
        if((pla == P_BLACK && hist.blackKoProhibited[loc]) || (pla == P_WHITE && hist.whiteKoProhibited[loc]))
          setRowBinV6(rowBin,pos,7, 1.0f, posStride, featureStride);
        if((pla == P_BLACK && hist.whiteKoProhibited[loc]) || (pla == P_WHITE && hist.blackKoProhibited[loc]))
          setRowBinV6(rowBin,pos,8, 1.0f, posStride, featureStride);
      }
    }
  }

  //Hide history from the net if a pass would end things and we're behaving as if a pass won't.
  //Or if the game is in fact over right now!
  bool hideHistory =
    hist.isGameFinished ||
    hist.isPastNormalPhaseEnd ||
    (nnInputParams.conservativePass && hist.passWouldEndGame(board,nextPlayer));

  //Features 9,10,11,12,13
  if(!hideHistory) {
    const vector<Move>& moveHistory = hist.moveHistory;
    size_t moveHistoryLen = moveHistory.size();
    //Also effectively wipe history as we change phase
    assert(moveHistoryLen >= hist.numTurnsThisPhase);
    int numTurnsThisPhase = hist.numTurnsThisPhase;

    if(numTurnsThisPhase >= 1 && moveHistory[moveHistoryLen-1].pla == opp) {
      Loc prev1Loc = moveHistory[moveHistoryLen-1].loc;
      if(prev1Loc == Board::PASS_LOC)
        rowGlobal[0] = 1.0;
      else if(prev1Loc != Board::NULL_LOC) {
        int pos = NNPos::locToPos(prev1Loc,xSize,nnXLen,nnYLen);
        setRowBinV6(rowBin,pos,9, 1.0f, posStride, featureStride);
      }
      if(numTurnsThisPhase >= 2 && moveHistory[moveHistoryLen-2].pla == pla) {
        Loc prev2Loc = moveHistory[moveHistoryLen-2].loc;
        if(prev2Loc == Board::PASS_LOC)
          rowGlobal[1] = 1.0;
        else if(prev2Loc != Board::NULL_LOC) {
          int pos = NNPos::locToPos(prev2Loc,xSize,nnXLen,nnYLen);
          setRowBinV6(rowBin,pos,10, 1.0f, posStride, featureStride);
        }
        if(numTurnsThisPhase >= 3 && moveHistory[moveHistoryLen-3].pla == opp) {
          Loc prev3Loc = moveHistory[moveHistoryLen-3].loc;
          if(prev3Loc == Board::PASS_LOC)
            rowGlobal[2] = 1.0;
          else if(prev3Loc != Board::NULL_LOC) {
            int pos = NNPos::locToPos(prev3Loc,xSize,nnXLen,nnYLen);
            setRowBinV6(rowBin,pos,11, 1.0f, posStride, featureStride);
          }
          if(numTurnsThisPhase >= 4 && moveHistory[moveHistoryLen-4].pla == pla) {
            Loc prev4Loc = moveHistory[moveHistoryLen-4].loc;
            if(prev4Loc == Board::PASS_LOC)
              rowGlobal[3] = 1.0;
            else if(prev4Loc != Board::NULL_LOC) {
              int pos = NNPos::locToPos(prev4Loc,xSize,nnXLen,nnYLen);
              setRowBinV6(rowBin,pos,12, 1.0f, posStride, featureStride);
            }
            if(numTurnsThisPhase >= 5 && moveHistory[moveHistoryLen-5].pla == opp) {
              Loc prev5Loc = moveHistory[moveHistoryLen-5].loc;
              if(prev5Loc == Board::PASS_LOC)
                rowGlobal[4] = 1.0;
              else if(prev5Loc != Board::NULL_LOC) {
                int pos = NNPos::locToPos(prev5Loc,xSize,nnXLen,nnYLen);
                setRowBinV6(rowBin,pos,13, 1.0f, posStride, featureStride);
              }
            }
          }
        }
      }
    }
  }

  //Ladder features 14,15,16,17
  auto addLadderFeature = [&board,xSize,nnXLen,nnYLen,posStride,featureStride,rowBin,opp](Loc loc, int pos, const vector<Loc>& workingMoves){
    assert(board.colors[loc] == P_BLACK || board.colors[loc] == P_WHITE);
    assert(pos >= 0 && pos < NNPos::MAX_BOARD_AREA);
    setRowBinV6(rowBin,pos,14, 1.0f, posStride, featureStride);
    if(board.colors[loc] == opp && board.getNumLiberties(loc) > 1) {
      for(size_t j = 0; j < workingMoves.size(); j++) {
        int workingPos = NNPos::locToPos(workingMoves[j],xSize,nnXLen,nnYLen);
        setRowBinV6(rowBin,workingPos,17, 1.0f, posStride, featureStride);
      }
    }
  };

  iterLadders(board, nnXLen, addLadderFeature);

  const Board& prevBoard = hideHistory ? board : hist.getRecentBoard(1);
  auto addPrevLadderFeature = [&prevBoard,posStride,featureStride,rowBin](Loc loc, int pos, const vector<Loc>& workingMoves){
    (void)workingMoves;
    (void)loc;
    assert(prevBoard.colors[loc] == P_BLACK || prevBoard.colors[loc] == P_WHITE);
    assert(pos >= 0 && pos < NNPos::MAX_BOARD_AREA);
    setRowBinV6(rowBin,pos,15, 1.0f, posStride, featureStride);
  };
  iterLadders(prevBoard, nnXLen, addPrevLadderFeature);

  const Board& prevPrevBoard = hideHistory ? board : hist.getRecentBoard(2);
  auto addPrevPrevLadderFeature = [&prevPrevBoard,posStride,featureStride,rowBin](Loc loc, int pos, const vector<Loc>& workingMoves){
    (void)workingMoves;
    (void)loc;
    assert(prevPrevBoard.colors[loc] == P_BLACK || prevPrevBoard.colors[loc] == P_WHITE);
    assert(pos >= 0 && pos < NNPos::MAX_BOARD_AREA);
    setRowBinV6(rowBin,pos,16, 1.0f, posStride, featureStride);
  };
  iterLadders(prevPrevBoard, nnXLen, addPrevPrevLadderFeature);

  //Features 18,19 - current territory, not counting group tax
  Color area[Board::MAX_ARR_SIZE];
  bool hasAreaFeature = false;
  if(hist.rules.scoringRule == Rules::SCORING_AREA && hist.rules.taxRule == Rules::TAX_NONE) {
    hasAreaFeature = true;
    bool nonPassAliveStones = true;
    bool safeBigTerritories = true;
    bool unsafeBigTerritories = true;
    board.calculateArea(area,nonPassAliveStones,safeBigTerritories,unsafeBigTerritories,hist.rules.multiStoneSuicideLegal);
  }
  else {
    bool keepTerritories = false;
    bool keepStones = false;
    int whiteMinusBlackNonDameTouchingRegionCount = 0;
    if(hist.rules.scoringRule == Rules::SCORING_AREA && (hist.rules.taxRule == Rules::TAX_SEKI || hist.rules.taxRule == Rules::TAX_ALL)) {
      hasAreaFeature = true;
      keepTerritories = false;
      keepStones = true;
    }
    else if(hist.rules.scoringRule == Rules::SCORING_TERRITORY && hist.rules.taxRule == Rules::TAX_NONE) {
      //Territory scoring omits feature until we reach the stage where scoring matters
      if(hist.encorePhase >= 2) {
        hasAreaFeature = true;
        keepTerritories = true;
        keepStones = false;
      }
    }
    else if(hist.rules.scoringRule == Rules::SCORING_TERRITORY && (hist.rules.taxRule == Rules::TAX_SEKI || hist.rules.taxRule == Rules::TAX_ALL)) {
      //Territory scoring omits feature until we reach the stage where scoring matters
      if(hist.encorePhase >= 2) {
        hasAreaFeature = true;
        keepTerritories = false;
        keepStones = false;
      }
    }
    else {
      ASSERT_UNREACHABLE;
    }

    if(hasAreaFeature) {
      board.calculateNonDameTouchingArea(
        area,whiteMinusBlackNonDameTouchingRegionCount,
        keepTerritories,
        keepStones,
        hist.rules.multiStoneSuicideLegal
      );
    }
  }

  if(hasAreaFeature) {
    for(int y = 0; y<ySize; y++) {
      for(int x = 0; x<xSize; x++) {
        Loc loc = Location::getLoc(x,y,xSize);
        int pos = NNPos::locToPos(loc,xSize,nnXLen,nnYLen);
        if(area[loc] == pla)
          setRowBinV6(rowBin,pos,18, 1.0f, posStride, featureStride);
        else if(area[loc] == opp)
          setRowBinV6(rowBin,pos,19, 1.0f, posStride, featureStride);
        else {
          if(hist.rules.scoringRule == Rules::SCORING_TERRITORY) {
            //Also we must be in the second encore phase, based on the logic above.
            if(board.colors[loc] == pla && hist.secondEncoreStartColors[loc] == pla)
              setRowBinV6(rowBin,pos,18, 1.0f, posStride, featureStride);
            else if(board.colors[loc] == opp && hist.secondEncoreStartColors[loc] == opp)
              setRowBinV6(rowBin,pos,19, 1.0f, posStride, featureStride);
          }
        }
      }
    }
  }

  //Features 20, 21 - second encore starting stones
  if(hist.encorePhase >= 2) {
    for(int y = 0; y<ySize; y++) {
      for(int x = 0; x<xSize; x++) {
        Loc loc = Location::getLoc(x,y,xSize);
        int pos = NNPos::locToPos(loc,xSize,nnXLen,nnYLen);
        if(hist.secondEncoreStartColors[loc] == pla)
          setRowBinV6(rowBin,pos,20, 1.0f, posStride, featureStride);
        else if(hist.secondEncoreStartColors[loc] == opp)
          setRowBinV6(rowBin,pos,21, 1.0f, posStride, featureStride);
      }
    }
  }


  //Global features.
  //The first 5 of them were set already above to flag which of the past 5 moves were passes.

  //Komi and any score adjustments
  float selfKomi = hist.currentSelfKomi(nextPlayer,nnInputParams.drawEquivalentWinsForWhite);
  float bArea = xSize * ySize;
  //Bound komi just in case
  if(selfKomi > bArea+1.0f)
    selfKomi = bArea+1.0f;
  if(selfKomi < -bArea-1.0f)
    selfKomi = -bArea-1.0f;
  rowGlobal[5] = selfKomi/20.0f;

  //Ko rule
  if(hist.rules.koRule == Rules::KO_SIMPLE) {}
  else if(hist.rules.koRule == Rules::KO_POSITIONAL || hist.rules.koRule == Rules::KO_SPIGHT) {
    rowGlobal[6] = 1.0f;
    rowGlobal[7] = 0.5f;
  }
  else if(hist.rules.koRule == Rules::KO_SITUATIONAL) {
    rowGlobal[6] = 1.0f;
    rowGlobal[7] = -0.5f;
  }
  else
    ASSERT_UNREACHABLE;

  //Suicide
  if(hist.rules.multiStoneSuicideLegal)
    rowGlobal[8] = 1.0f;

  //Scoring
  if(hist.rules.scoringRule == Rules::SCORING_AREA) {}
  else if(hist.rules.scoringRule == Rules::SCORING_TERRITORY)
    rowGlobal[9] = 1.0f;
  else
    ASSERT_UNREACHABLE;
  //Tax
  if(hist.rules.taxRule == Rules::TAX_NONE) {}
  else if(hist.rules.taxRule == Rules::TAX_SEKI)
    rowGlobal[10] = 1.0f;
  else if(hist.rules.taxRule == Rules::TAX_ALL) {
    rowGlobal[10] = 1.0f;
    rowGlobal[11] = 1.0f;
  }
  else
    ASSERT_UNREACHABLE;

  //Encore phase
  if(hist.encorePhase > 0)
    rowGlobal[12] = 1.0f;
  if(hist.encorePhase > 1)
    rowGlobal[13] = 1.0f;

  //Does a pass end the current phase given the ruleset and history?
  bool passWouldEndPhase = hideHistory ? false : hist.passWouldEndPhase(board,nextPlayer);
  rowGlobal[14] = passWouldEndPhase ? 1.0f : 0.0f;

  //Provide parity information about the board size and komi
  //This comes from the following observation:
  //From white's perspective:
  //Komi = 0.0 - Draw possible
  //Komi = 0.5 - Win the games we would have drawn with komi 0.0
  //Komi = 1.0 - Usually no difference from komi 0.5
  //Komi = 1.5 - Usually no difference from komi 0.5
  //Komi = 2.0 - Draw possible
  //If we were to assign an "effective goodness" to these komis in order it would look like
  //0 1 1 1 2 3 3 3 4 5 5 5 6 ...
  //since when away from the right parity, increasing the komi doesn't help us except in cases of seki with odd numbers of dame.
  //If we were to add 0.5 times a vector like:
  //0 -1 0 1 0 -1 0 1 0 -1 0 ...
  //Then this would become a linear function and hopefully easier for a neural net to learn.
  //We expect that this is hard for a neural net to learn since it depends on the parity of the board size
  //and is very "xor"like.
  //So we provide it as an input.
  //Since we are using a model where games are jittered by 0.5 (see BoardHistory::whiteKomiAdjustmentForDraws)
  //in theory right thing to first order to provide should be a triangular wave with a period of 2 komi points:
  //  ../\........
  //  ./..\.......
  //  /....\..../.
  //  ......\../..
  //  .......\/...
  //The upsloping part of the wave is centered around the komi value where you could draw
  //since komi is extra valuable when it turns losses into draws into wins, peaking at the komi value where you could draw + 0.5.
  //It's downsloping around the komi value where you can't draw, since the marginal komi there is nearly useless, not causing you to win
  //more games except in case of odd-dame seki.

  if(hist.rules.scoringRule == Rules::SCORING_AREA || hist.encorePhase >= 2) {
    bool boardAreaIsEven = (xSize*ySize) % 2 == 0;

    //What is the parity of the komi values that can produce jigos?
    bool drawableKomisAreEven = boardAreaIsEven;

    //Find the difference between the komi viewed from our perspective and the nearest drawable komi below it.
    float komiFloor;
    if(drawableKomisAreEven)
      komiFloor = floor(selfKomi / 2.0f) * 2.0f;
    else
      komiFloor = floor((selfKomi-1.0f) / 2.0f) * 2.0f + 1.0f;

    //Cap just in case we have floating point weirdness
    float delta = selfKomi - komiFloor;
    assert(delta >= -0.0001f);
    assert(delta <= 2.0001f);
    if(delta < 0.0f)
      delta = 0.0f;
    if(delta > 2.0f)
      delta = 2.0f;

    //Create the triangle wave based on the difference
    float wave;
    if(delta < 0.5f)
      wave = delta;
    else if(delta < 1.5f)
      wave = 1.0f-delta;
    else
      wave = delta-2.0f;

    //NOTE: If ever changing which feature this is, must also update index in model.py where we multiply it into the scorebelief parity vector
    rowGlobal[15] = wave;
  }

}
