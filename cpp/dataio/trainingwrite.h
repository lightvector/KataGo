#ifndef DATAIO_TRAINING_WRITE_H_
#define DATAIO_TRAINING_WRITE_H_

#include "../dataio/numpywrite.h"
#include "../neuralnet/nninputs.h"
#include "../neuralnet/nninterface.h"

STRUCT_NAMED_PAIR(Loc,loc,int16_t,policyTarget,PolicyTargetMove);
STRUCT_NAMED_PAIR(std::vector<PolicyTargetMove>*,policyTargets,int64_t,unreducedNumVisits,PolicyTarget);

struct ValueTargets {
  //As usual, these are from the perspective of white.
  float win;
  float loss;
  float noResult;
  float score;
  bool hasLead;
  float lead;

  ValueTargets();
  ~ValueTargets();
};

struct SidePosition {
  Board board;
  BoardHistory hist;
  Player pla;
  int64_t unreducedNumVisits;
  std::vector<PolicyTargetMove> policyTarget;
  ValueTargets whiteValueTargets;
  float targetWeight;
  int numNeuralNetChangesSoFar; //Number of neural net changes this game before the creation of this side position

  SidePosition();
  SidePosition(const Board& board, const BoardHistory& hist, Player pla, int numNeuralNetChangesSoFar);
  ~SidePosition();
};

STRUCT_NAMED_PAIR(std::string,name,int,turnNumber,ChangedNeuralNet);

struct FinishedGameData {
  std::string bName;
  std::string wName;
  int bIdx;
  int wIdx;

  Board startBoard; //Board as of the end of startHist, beginning of training period
  BoardHistory startHist; //Board history as of start of training period
  BoardHistory endHist; //Board history as of end of training period
  Player startPla; //Player to move as of end of startHist.
  Hash128 gameHash;

  double drawEquivalentWinsForWhite;
  Player playoutDoublingAdvantagePla;
  double playoutDoublingAdvantage;
  bool hitTurnLimit;

  //Metadata about how the game was initialized
  int numExtraBlack;
  int mode;
  int beganInEncorePhase;
  int usedInitialPosition;

  //If false, then we don't have these below vectors and ownership information
  bool hasFullData;
  std::vector<float> targetWeightByTurn;
  std::vector<PolicyTarget> policyTargetsByTurn;
  std::vector<ValueTargets> whiteValueTargetsByTurn; //Except this one, we may have some of
  Color* finalFullArea;
  Color* finalOwnership;
  bool* finalSekiAreas;
  float* finalWhiteScoring;

  std::vector<SidePosition*> sidePositions;
  std::vector<ChangedNeuralNet*> changedNeuralNets;

  static constexpr int NUM_MODES = 6;
  static constexpr int MODE_NORMAL = 0;
  static constexpr int MODE_CLEANUP_TRAINING = 1;
  static constexpr int MODE_FORK = 2;
  static constexpr int MODE_HANDICAP = 3;
  static constexpr int MODE_SGFPOS = 4;
  static constexpr int MODE_HINTPOS = 5;

  FinishedGameData();
  ~FinishedGameData();

  void printDebug(std::ostream& out) const;
};

struct TrainingWriteBuffers {
  int inputsVersion;
  int maxRows;
  int numBinaryChannels;
  int numGlobalChannels;
  int dataXLen;
  int dataYLen;
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
  //C4-7: MCTS win-loss-noresult estimate td-like target, lambda = 1 - 1/(1 + boardArea * 0.176)
  //C8-11: MCTS win-loss-noresult estimate td-like target, lambda = 1 - 1/(1 + boardArea * 0.056)
  //C12-15: MCTS win-loss-noresult estimate td-like target, lambda = 1 - 1/(1 + boardArea * 0.016)
  //C16-19: MCTS win-loss-noresult estimate td-like target, lambda = 0 (so, actually just the immediate MCTS result).

  //C20: Actual final score, from the perspective of the player to move, adjusted for draw utility, zero if C27 is zero.
  //C21: Lead in points, number of points to make the game fair, zero if C29 is zero.
  //C22: Expected arrival time of WL variance.
  //C23-24: Unused

  //C25 Weight multiplier for row as a whole

  //C26: Weight assigned to the policy target
  //C27: Weight assigned to the final board ownership target and score distr targets. Most training rows will have this be 1, some will be 0.
  //C28: Weight assigned to the next move policy target
  //C29: Weight assigned to the lead target
  //C30-32: Unused
  //C33: Weight assigned to the future position targets valueTargetsNCHW C1-C2
  //C34: Weight assigned to the area/territory target valueTargetsNCHW C4
  //C35: Unused

  //C36-40: Precomputed mask values indicating if we should use historical moves 1-5, if we desire random history masking.
  //1 means use, 0 means don't use.

  //C41-46: 128-bit hash identifying the game, different rows from the same game share the same value.
  //Split into chunks of 22, 22, 20, 22, 22, 20 bits, little-endian style (since floats have > 22 bits of precision).

  //C47: Komi, adjusted for draw utility and points costed or paid so far, from the perspective of the player to move.
  //C48: 1 if we're in an area-scoring-like phase of the game (area scoring or second encore territory scoring)

  //C49: 1 if an earlier neural net started this game, compared to the latest in this data file.
  //C50: If positive, an earlier neural net was playing this specific move, compared to the latest in this data file.

  //C51: Turn number of the game, zero-indexed.
  //C52: Did this game end via hitting turn limit?
  //C53: First turn of this game that was selfplay for training rather than initialization (e.g. handicap stones, random init of the starting board pos)
  //C54: Number of extra moves black got at the start (i.e. handicap games)

  //C55-56: Game type, how the game was initialized
  //0 = normal self-play game.
  //1 = cleanup-phase-training game. C56 is the starting encore phase
  //2 = fork from another self-play game.
  //3 = handicap game
  //4 = sampled from an external SGF position (e.g. human data or other bots).
  //5 = sampled from a hint position (e.g. blindspot training).

  //C57: Contains some data but is actually redundant with C55.
  //C58: Contains some data but is actually redundant with C55.
  //C59: Unused
  //C60: Number of visits in the search generating this row, prior to any reduction.
  //C61: Number of bonus points the player to move will get onward from this point in the game
  //C62: Unused
  //C63: Unused

  NumpyBuffer<float> globalTargetsNC;

  //Score target
  //Indices correspond to scores, from (-dataXLen*dataYLen-EXTRA_SCORE_DISTR_RADIUS)-0.5 to (dataXLen*dataYLen+EXTRA_SCORE_DISTR_RADIUS)+0.5,
  //making 2*dataXLen*dataYLen+2*EXTRA_SCORE_DISTR_RADIUS indices in total.
  //Index of the actual score is labeled with 100, the rest labeled with 0, from the perspective of the player to move.
  //Except in case of integer komi, the value can be split between two adjacent labels based on value of draw.
  //Arbitrary if C27 has weight 0.
  NumpyBuffer<int8_t> scoreDistrN;

  //Spatial value-related targets
  //C0: Final board ownership [-1,1], from the perspective of the player to move. All 0 if C27 has weight 0.
  //C1: Difference between ownership and naive area (such as due to seki). All 0 if C27 has weight 0.
  //C2-3: Future board position a certain number of turns in the future. All 0 if C33 has weight 0.
  //C4: Final board area/territory [-120,120]. All 0 if C34 has weight 0. Unlike ownership, takes into account group tax and scoring rules.
  NumpyBuffer<int8_t> valueTargetsNCHW;

  TrainingWriteBuffers(int inputsVersion, int maxRows, int numBinaryChannels, int numGlobalChannels, int dataXLen, int dataYLen);
  ~TrainingWriteBuffers();

  TrainingWriteBuffers(const TrainingWriteBuffers&) = delete;
  TrainingWriteBuffers& operator=(const TrainingWriteBuffers&) = delete;

  void clear();

  void addRow(
    const Board& board, const BoardHistory& hist, Player nextPlayer,
    int turnNumberAfterStart,
    float targetWeight,
    int64_t unreducedNumVisits,
    const std::vector<PolicyTargetMove>* policyTarget0, //can be null
    const std::vector<PolicyTargetMove>* policyTarget1, //can be null
    const std::vector<ValueTargets>& whiteValueTargets,
    int whiteValueTargetsIdx, //index in whiteValueTargets corresponding to this turn.
    const Board* finalBoard,
    Color* finalFullArea,
    Color* finalOwnership,
    float* finalWhiteScoring,
    const std::vector<Board>* posHistForFutureBoards, //can be null
    bool isSidePosition,
    int numNeuralNetsBehindLatest,
    const FinishedGameData& data,
    Rand& rand
  );

  void writeToZipFile(const std::string& fileName);
  void writeToTextOstream(std::ostream& out);

};

class TrainingDataWriter {
 public:
  TrainingDataWriter(const std::string& outputDir, int inputsVersion, int maxRowsPerFile, double firstFileMinRandProp, int dataXLen, int dataYLen, const std::string& randSeed);
  TrainingDataWriter(std::ostream* debugOut, int inputsVersion, int maxRowsPerFile, double firstFileMinRandProp, int dataXLen, int dataYLen, int onlyWriteEvery, const std::string& randSeed);
  TrainingDataWriter(const std::string& outputDir, std::ostream* debugOut, int inputsVersion, int maxRowsPerFile, double firstFileMinRandProp, int dataXLen, int dataYLen, int onlyWriteEvery, const std::string& randSeed);
  ~TrainingDataWriter();

  void writeGame(const FinishedGameData& data);
  void flushIfNonempty();
  bool flushIfNonempty(std::string& resultingFilename);

  bool isEmpty() const;
  int64_t numRowsInBuffer() const;

 private:
  std::string outputDir;
  int inputsVersion;
  Rand rand;
  TrainingWriteBuffers* writeBuffers;

  std::ostream* debugOut;
  int debugOnlyWriteEvery;
  int64_t rowCount;

  bool isFirstFile;
  int firstFileMaxRows;

  void writeAndClearIfFull();

};


#endif  // DATAIO_TRAININGWRITE_H_
