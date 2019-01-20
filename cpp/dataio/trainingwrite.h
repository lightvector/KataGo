#ifndef TRAINING_WRITE_H
#define TRAINING_WRITE_H

#include "../neuralnet/nninputs.h"
#include "../neuralnet/nninterface.h"
#include "../dataio/numpywrite.h"

STRUCT_NAMED_PAIR(Loc,loc,int16_t,policyTarget,PolicyTargetMove);
STRUCT_NAMED_PAIR(vector<PolicyTargetMove>*,policyTargets,int64_t,unreducedNumVisits,PolicyTarget);

struct ValueTargets {
  //As usual, these are from the perspective of white.
  float win;
  float loss;
  float noResult;
  float score;

  bool hasMctsUtility;
  float mctsUtility1;
  float mctsUtility4;
  float mctsUtility16;
  float mctsUtility64;
  float mctsUtility256;
  ValueTargets();
  ~ValueTargets();
};

struct SidePosition {
  Board board;
  BoardHistory hist;
  Player pla;
  int64_t unreducedNumVisits;
  vector<PolicyTargetMove> policyTarget;
  ValueTargets whiteValueTargets;
  float targetWeight;
  int numNeuralNetChangesSoFar; //Number of neural net changes this game before the creation of this side position

  SidePosition();
  SidePosition(const Board& board, const BoardHistory& hist, Player pla, int numNeuralNetChangesSoFar);
  ~SidePosition();
};

STRUCT_NAMED_PAIR(string,name,int,turnNumber,ChangedNeuralNet);

struct FinishedGameData {
  string bName;
  string wName;
  int bIdx;
  int wIdx;

  Board startBoard; //Board as of the end of startHist, beginning of training period
  BoardHistory startHist; //Board history as of start of training period
  BoardHistory endHist; //Board history as of end of training period
  Player startPla; //Player to move as of end of startHist.
  Hash128 gameHash;

  double drawEquivalentWinsForWhite;
  bool hitTurnLimit;

  //Metadata about how the game was initialized
  int numExtraBlack;
  int mode;
  int modeMeta1;
  int modeMeta2;

  //If false, then we don't have these below vectors and ownership information
  bool hasFullData;
  int posLen;
  vector<float> targetWeightByTurn;
  vector<PolicyTarget> policyTargetsByTurn;
  vector<ValueTargets> whiteValueTargetsByTurn;
  int8_t* finalWhiteOwnership;

  vector<SidePosition*> sidePositions;
  vector<ChangedNeuralNet*> changedNeuralNets;

  FinishedGameData();
  ~FinishedGameData();

  void printDebug(ostream& out) const;
};

struct TrainingWriteBuffers {
  int inputsVersion;
  int maxRows;
  int numBinaryChannels;
  int numGlobalChannels;
  int posLen;
  int packedBoardArea;

  int curRows;
  float* binaryInputNCHWUnpacked;

  //Input feature planes that have spatial extent, all of which happen to be binary.
  //Packed bitwise, with each (HW) zero-padded to a round byte.
  //Within each byte, bits are packed bigendianwise, since that's what numpy's unpackbits will expect.
  NumpyBuffer<uint8_t> binaryInputNCHWPacked;
  //Input features that are global.
  NumpyBuffer<float> globalInputNC;

  //Policy targets
  //Shape is [N,C,Pos]. Almost NCHW, except we have a Pos of length, e.g. 362, due to the pass input, instead of 19x19.
  //Contains number of visits, possibly with a subtraction.
  //Channel i will still be a dummy probability distribution (not all zero) if weight 0
  //C0: Policy target this turn.
  //C1: Policy target next turn.
  NumpyBuffer<int16_t> policyTargetsNCMove;

  //Value targets and other metadata, from the perspective of the player to move
  //C0-3: Categorial game result, win,loss,noresult, and also score. Draw is encoded as some blend of win and loss based on drawEquivalentWinsForWhite.
  //C4-7: MCTS win-loss-noresult estimate td-like target, lambda = 35/36, nowFactor = 1/36
  //C8-11: MCTS win-loss-noresult estimate td-like target, lambda = 11/12, nowFactor = 1/12
  //C12-15: MCTS win-loss-noresult estimate td-like target, lambda = 3/4, nowFactor = 1/4
  //C16-19: MCTS win-loss-noresult estimate td-like target, lambda = 0, nowFactor = 1 (no-temporal-averaging MCTS search result)

  //C20: Actual final score, from the perspective of the player to move, adjusted for draw utility, zero if C27 is zero.
  //C21: MCTS utility variance, 1->4 visits
  //C22: MCTS utility variance, 4->16 visits
  //C23: MCTS utility variance, 16->64 visits
  //C24: MCTS utility variance, 64->256 visits

  //C25 Weight multiplier for row as a whole

  //C26 Weight assigned to the policy target
  //C27 Weight assigned to the final board ownership target and score distr and bonus score targets. Most training rows will have this be 1, some will be 0.
  //C28: Weight assigned to the utilityvariance target
  //C29: Weight assigned to the next move policy target
  //C30: Unused

  //C31-35: Precomputed mask values indicating if we should use historical moves 1-5, if we desire random history masking.
  //1 means use, 0 means don't use.

  //C36-41: 128-bit hash identifying the game which should also be output in the SGF data.
  //Split into chunks of 22, 22, 20, 22, 22, 20 bits, little-endian style (since floats have > 22 bits of precision).

  //C42: Komi, adjusted for draw utility and points costed or paid so far, from the perspective of the player to move.
  //C43: 1 if we're in an area-scoring-like phase of the game (area scoring or second encore territory scoring)

  //C44: 1 if an earlier neural net started this game, compared to the latest in this data file.
  //C45: If positive, an earlier neural net was playing this specific move, compared to the latest in this data file.

  //C46: Turn number of the game, zero-indexed.
  //C47: Did this game end via hitting turn limit?
  //C48: First turn of this game that was selfplay for training rather than initialization (e.g. handicap stones, random init of the starting board pos)
  //C49: Number of extra moves black got at the start (i.e. handicap games)

  //C50-51: Game type, game typesource metadata
  // 0 = normal self-play game. C51 unused
  // 1 = encore-training game. C51 is the starting encore phase
  //C52: 0 = normal, 1 = whole game was forked with an experimental move in the opening
  //C53: 0 = normal, 1 = training sample was an isolated side position forked off of main game
  //C54: Unused
  //C55: Number of visits in the search generating this row, prior to any reduction.

  NumpyBuffer<float> globalTargetsNC;

  //Score target
  //Indices correspond to scores, from (-posLen^2-EXTRA_SCORE_DISTR_RADIUS)-0.5 to (posLen^2+EXTRA_SCORE_DISTR_RADIUS)+0.5,
  //making 2*posLen^2+2*EXTRA_SCORE_DISTR_RADIUS indices in total.
  //Index of the actual score is labeled with 100, the rest labeled with 0, from the perspective of the player to move.
  //Except in case of integer komi, the value can be split between two adjacent labels based on value of draw.
  //Arbitrary if C26 has weight 0.
  NumpyBuffer<int8_t> scoreDistrN;
  //Ranges from -30 to 30, 61 indices in total. Index of the number of bonus points the player to move will get onward from this point in the game
  //is labeled with 1, the rest labeled with 0, from the perspective of the player to move.
  NumpyBuffer<int8_t> selfBonusScoreN;

  //Spatial value-related targets
  //C0 - Final board ownership (-1,0,1), from the perspective of the player to move. All 0 if C26 has weight 0.
  NumpyBuffer<int8_t> valueTargetsNCHW;

  TrainingWriteBuffers(int inputsVersion, int maxRows, int numBinaryChannels, int numGlobalChannels, int posLen);
  ~TrainingWriteBuffers();

  TrainingWriteBuffers(const TrainingWriteBuffers&) = delete;
  TrainingWriteBuffers& operator=(const TrainingWriteBuffers&) = delete;

  void clear();

  void addRow(
    const Board& board, const BoardHistory& hist, Player nextPlayer,
    int turnNumberAfterStart,
    float targetWeight,
    int64_t unreducedNumVisits,
    const vector<PolicyTargetMove>* policyTarget0, //can be null
    const vector<PolicyTargetMove>* policyTarget1, //can be null
    const vector<ValueTargets>& whiteValueTargets,
    int whiteValueTargetsIdx, //index in whiteValueTargets corresponding to this turn.
    int8_t* finalWhiteOwnership,
    bool isSidePosition,
    int numNeuralNetsBehindLatest,
    const FinishedGameData& data,
    Rand& rand
  );

  void writeToZipFile(const string& fileName);
  void writeToTextOstream(ostream& out);

};

class TrainingDataWriter {
 public:
  TrainingDataWriter(const string& outputDir, int inputsVersion, int maxRowsPerFile, double firstFileMinRandProp, int posLen, const string& randSeed);
  TrainingDataWriter(ostream* debugOut, int inputsVersion, int maxRowsPerFile, double firstFileMinRandProp, int posLen, int onlyWriteEvery, const string& randSeed);
  TrainingDataWriter(const string& outputDir, ostream* debugOut, int inputsVersion, int maxRowsPerFile, double firstFileMinRandProp, int posLen, int onlyWriteEvery, const string& randSeed);
  ~TrainingDataWriter();

  void writeGame(const FinishedGameData& data);
  void flushIfNonempty();

 private:
  string outputDir;
  int inputsVersion;
  Rand rand;
  TrainingWriteBuffers* writeBuffers;

  ostream* debugOut;
  int debugOnlyWriteEvery;
  int64_t rowCount;

  bool isFirstFile;
  int firstFileMaxRows;

  void writeAndClearIfFull();

};


#endif
