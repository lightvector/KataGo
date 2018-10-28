#ifndef TRAINING_WRITE_H
#define TRAINING_WRITE_H

#include "../neuralnet/nninputs.h"
#include "../neuralnet/nninterface.h"
#include "../dataio/numpywrite.h"

STRUCT_NAMED_PAIR(Loc,loc,int16_t,policyTarget,PolicyTargetMove);
struct ValueTargets {
  //As usual, these are from the perspective of white.
  float win;
  float loss;
  float noResult;
  float scoreValue;
  float score;
  float mctsUtility1;
  float mctsUtility4;
  float mctsUtility16;
  float mctsUtility64;
  float mctsUtility256;
  ValueTargets();
  ~ValueTargets();
};

struct FinishedGameData {
  Board startBoard;
  BoardHistory startHist;
  BoardHistory endHist;
  Player startPla;
  Hash128 gameHash;

  //This vector MIGHT be shorter than the list of moves in startHist, because there might be moves in
  //startHist for context that we don't actually want to record as part of this game for training data.
  vector<Move> moves;
  vector<vector<PolicyTargetMove>*> policyTargetsByTurn;
  vector<ValueTargets> whiteValueTargetsByTurn;
  int8_t* finalOwnership;
  double drawEquivalentWinsForWhite;

  int posLen;
  bool hitTurnLimit;

  //Metadata about how the game was initialized
  int firstTrainingTurn;
  int mode;
  int modeMeta1;
  int modeMeta2;

  FinishedGameData(int posLen, double drawEquivalentWinsForWhite);
  ~FinishedGameData();

};

struct TrainingWriteBuffers {
  int inputsVersion;
  int maxRows;
  int numBinaryChannels;
  int numFloatChannels;
  int posLen;
  int packedBoardArea;

  int curRows;
  bool* binaryInputNCHWUnpacked;

  //Input feature planes that have spatial extent, all of which happen to be binary.
  //Packed bitwise, with each (HW) zero-padded to a round byte.
  //Within each byte, bits are packed bigendianwise, since that's what numpy's unpackbits will expect.
  NumpyBuffer<uint8_t> binaryInputNCHWPacked;
  //Input features that are global.
  NumpyBuffer<float> floatInputNC;

  //Policy targets
  //Shape is [N,C,Pos]. Almost NCHW, except we have a Pos of length, e.g. 362, due to the pass input, instead of 19x19.
  //Contains number of visits, possibly with a subtraction.
  //Channel i will still be a dummy probability distribution (not all zero) if weight 0
  //C0: Policy target this turn.
  NumpyBuffer<int16_t> policyTargetsNCMove;

  //Value targets and other metadata
  //C0-3: Categorial game result, win,loss,noresult, and also score utility. Draw is encoded as some blend of win and loss based on drawEquivalentWinsForWhite.
  //C4-7: MCTS win-loss-noresult estimate td-like target, lambda = 35/36
  //C8-11: MCTS win-loss-noresult estimate td-like target, lambda = 11/12
  //C12-15: MCTS win-loss-noresult estimate td-like target, lambda = 3/4
  //C16-19: MCTS win-loss-noresult estimate td-like target, lambda = 0

  //C20: Actual score
  //C21: MCTS utility variance, 1->4 visits
  //C22: MCTS utility variance, 4->16 visits
  //C23: MCTS utility variance, 16->64 visits
  //C24: MCTS utility variance, 64->256 visits

  //C25 Weight assigned to the policy target
  //Currently always 1.0, But it is also conceivable that maybe some training rows will lack a policy target
  //so users should be robust to that.
  //C26,27 Unused

  //C28-32: Precomputed mask values indicating if we should use historical moves 1-5, if we desire random history masking.
  //1 means use, 0 means don't use.

  //C33-38: 128-bit hash identifying the game which should also be output in the SGF data.
  //Split into chunks of 22, 22, 20, 22, 22, 20 bits, little-endian style (since floats have > 22 bits of precision).

  //C39: Turn number, zero-indexed
  //C40: Did this game end via hitting turn limit?
  //C41: First turn of this game that was proper selfplay for training rather than being initialization
  //C42-44: Game type, game typesource metadata
  // 0 = normal self-play game. C43,C44 unused
  // 1 = encore-training game. C43 is the starting encore phase, C44 unused
  NumpyBuffer<float> floatTargetsNC;

  //Spatial value-related targets
  //C0 - Final board ownership (-1,0,1). All 0 if no result.
  NumpyBuffer<int8_t> valueTargetsNCHW;

  TrainingWriteBuffers(int inputsVersion, int maxRows, int numBinaryChannels, int numFloatChannels, int posLen);
  ~TrainingWriteBuffers();

  TrainingWriteBuffers(const TrainingWriteBuffers&) = delete;
  TrainingWriteBuffers& operator=(const TrainingWriteBuffers&) = delete;

  void clear();

  void addRow(
    const Board& board, const BoardHistory& hist, Player nextPlayer, double drawEquivalentWinsForWhite,
    int turnNumber,
    const vector<PolicyTargetMove>* policyTarget0, //can be null
    const FinishedGameData& data,
    Rand& rand
  );

  void writeToZipFile(const string& fileName);

};

class TrainingDataWriter {
 public:
  TrainingDataWriter(const string& outputDir, int inputsVersion, int maxRowsPerFile, int posLen);
  ~TrainingDataWriter();

  void writeGame(const FinishedGameData& data);
  void close();

 private:
  string outputDir;
  int inputsVersion;
  Rand rand;
  TrainingWriteBuffers* writeBuffers;

  void writeAndClearIfFull();

};


#endif
