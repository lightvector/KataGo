#ifndef TRAINING_WRITE_H
#define TRAINING_WRITE_H

#include "../neuralnet/nninputs.h"
#include "../neuralnet/nninterface.h"
#include "../dataio/numpywrite.h"

STRUCT_NAMED_PAIR(Loc,loc,int16_t,playouts,PolicyTargetMove);
struct ValueTargets {
  float win;
  float loss;
  float noResult;
  float scoreUtility;
  float score;
  float mctsUtility1;
  float mctsUtility4;
  float mctsUtility16;
  float mctsUtility64;
  float mctsUtility256;
  ValueTargets();
  ~ValueTargets();
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
  NumpyBuffer<float> floatInputNC;

  //Shape is [N,C,Pos]. Almost NCHW, except we have a Pos of length, e.g. 362, due to the pass input, instead of 19x19.
  //Contains number of visits, possibly with a subtraction.
  //Channel i Will be a dummy probability distribution (not all zero) if policyTargetWeightsNC[i] == 0
  //C0: Policy target this turn.
  //C1: Policy target next turn.
  //C2: Policy target next next turn.
  NumpyBuffer<int16_t> policyTargetsNCPos;
  //Weight assigned to the given policy target.
  //Generally 1.0, except sometimes the game has ended and therefore there no policy target for C1 or C2,
  //so we have 0 weight in those cases. It is also conceivable that maybe some training rows will lack a C0 policy target
  //so users should be robust to that.
  NumpyBuffer<float> policyTargetWeightsNC;

  //Value-related targets
  //C0-3: Categorial game result, win,loss,noresult, and also score utility. Draw is encoded as some blend of win and loss based on drawUtilityForWhite.
  //C4-7: MCTS win-loss-noresult estimate td-like target, lambda = 35/36
  //C8-11: MCTS win-loss-noresult estimate td-like target, lambda = 11/12
  //C12-15: MCTS win-loss-noresult estimate td-like target, lambda = 3/4
  //C16-19: MCTS win-loss-noresult estimate td-like target, lambda = 0

  //C20: Actual score
  //C21: MCTS utility variance, 1->4 visits
  //C22: MCTS utility variance, 4->16 visits
  //C23: MCTS utility variance, 16->64 visits
  //C24: MCTS utility variance, 64->256 visits
  //C25: Action-value weight
  NumpyBuffer<float> valueTargetsNC;

  //Spatial value-related targets
  //C0 - Final board ownership (-1,0,1). All 0 if no result.
  //C1 - Action-value head.
  NumpyBuffer<int16_t> valueTargetsNCHW;

  TrainingWriteBuffers(int inputsVersion, int maxRows, int numBinaryChannels, int numFloatChannels, int posLen);
  ~TrainingWriteBuffers();

  TrainingWriteBuffers(const TrainingWriteBuffers&) = delete;
  TrainingWriteBuffers& operator=(const TrainingWriteBuffers&) = delete;

  void clear();

  void addRow(
    const Board& board, const BoardHistory& hist, Player nextPlayer, double drawUtilityForWhite,
    int turnNumber,
    const vector<PolicyTargetMove>* policyTarget0, //can be null
    const vector<PolicyTargetMove>* policyTarget1, //can be null
    const vector<PolicyTargetMove>* policyTarget2, //can be null
    const vector<ValueTargets>& whiteValueTargetsByTurn,
    const int16_t* finalOwnership,
    const float* actionValueTarget //can be null
  );

  void writeToZipFile(const string& fileName);

};

struct FinishedGameData {
  Board startBoard;
  BoardHistory startHist;
  Player startPla;
  vector<Move> moves;
  vector<vector<PolicyTargetMove>*> policyTargetsByTurn;
  vector<ValueTargets> whiteValueTargetsByTurn;
  vector<float*> actionValueTargetByTurn;
  int16_t* finalOwnership;
  double drawUtilityForWhite;

  FinishedGameData(Board startBoard, BoardHistory startHist, Player startPla, int posLen, double drawUtilityForWhite);
  ~FinishedGameData();

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
