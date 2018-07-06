#include "../core/global.h"
#include "../neuralnet/nninputs.h"

int NNPos::getOffset(int bSize) {
  return (MAX_BOARD_LEN - bSize)/2;
}
int NNPos::xyToPos(int x, int y, int offset) {
  return (y + offset) * MAX_BOARD_LEN + (x + offset);
}
int NNPos::locToPos(Loc loc, int bSize, int offset) {
  if(loc == Board::PASS_LOC)
    return MAX_BOARD_LEN * MAX_BOARD_LEN;
  else if(loc == Board::NULL_LOC)
    return MAX_BOARD_LEN * (MAX_BOARD_LEN + 1);
  return (Location::getY(loc,bSize) + offset) * MAX_BOARD_LEN + (Location::getX(loc,bSize) + offset);
}
Loc NNPos::posToLoc(int pos, int bSize, int offset) {
  if(pos == MAX_BOARD_LEN * MAX_BOARD_LEN)
    return Board::PASS_LOC;
  int x = pos % MAX_BOARD_LEN - offset;
  int y = pos / MAX_BOARD_LEN - offset;
  if(x < 0 || x >= bSize || y < 0 || y >= bSize)
    return Board::NULL_LOC;
  return Location::getLoc(x,y,bSize);
}

static void setRow(float* row, int pos, int feature, float value) {
  row[pos * NNInputs::NUM_FEATURES + feature] = value;
}

//Calls f on each location that is part of an inescapable atari, or a group that can be put into inescapable atari
static void iterLadders(const Board& board, std::function<void(Loc,int,const vector<Loc>&)> f) {
  int bSize = board.x_size;
  int offset = NNPos::getOffset(bSize);

  Loc chainHeadsSolved[bSize*bSize];
  bool chainHeadsSolvedValue[bSize*bSize];
  int numChainHeadsSolved = 0;
  Board copy(board);
  vector<Loc> buf;
  vector<Loc> workingMoves;

  for(int y = 0; y<bSize; y++) {
    for(int x = 0; x<bSize; x++) {
      int pos = NNPos::xyToPos(x,y,offset);
      Loc loc = Location::getLoc(x,y,bSize);
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

// //Calls f on each location that is part of an inescapable atari, or a group that can be put into inescapable atari
// static void iterWouldBeLadder(const Board& board, Player pla, std::function<void(Loc,int)> f) {
//   Player opp = getOpp(pla);
//   int bSize = board.x_size;
//   int offset = NNPos::getOffset(bSize);

//   Board copy(board);
//   vector<Loc> buf;

//   for(int y = 0; y<bSize; y++) {
//     for(int x = 0; x<bSize; x++) {
//       int pos = NNPos::xyToPos(x,y,offset);
//       Loc loc = Location::getLoc(x,y,bSize);
//       Color stone = board.colors[loc];
//       if(stone == C_EMPTY && board.getNumLibertiesAfterPlay(loc,pla,3) == 2) {

//       }
//     }
//   }
// }


Hash128 NNInputs::getHash(
    const Board& board, const vector<Move>& moveHistory, int moveHistoryLen,
    Player nextPlayer, float selfKomi
) {
  Hash128 hash = board.pos_hash;
  hash ^= Board::ZOBRIST_PLAYER_HASH[nextPlayer];

  //TODO should this incorporate history?
  (void)moveHistory;
  (void)moveHistoryLen;

  //TODO incorporate superko
  if(board.ko_loc != Board::NULL_LOC)
    hash ^= Board::ZOBRIST_KO_LOC_HASH[board.ko_loc];

  int64_t komiX2 = (int64_t)(selfKomi*2.0f);
  uint64_t komiHash = Hash::murmurMix((uint64_t)komiX2);
  hash.hash0 ^= komiHash;
  hash.hash1 ^= Hash::basicLCong(komiHash);

  return hash;
}


void NNInputs::fillRow(
  const Board& board, const vector<Move>& moveHistory, int moveHistoryLen,
  Player nextPlayer, float selfKomi, float* row
) {
  assert(board.x_size == board.y_size);
  assert(moveHistoryLen <= moveHistory.size());

  Player pla = nextPlayer;
  Player opp = getOpp(pla);
  int bSize = board.x_size;

  int offset = NNPos::getOffset(bSize);

  for(int y = 0; y<bSize; y++) {
    for(int x = 0; x<bSize; x++) {
      int pos = NNPos::xyToPos(x,y,offset);
      Loc loc = Location::getLoc(x,y,bSize);

      //Feature 0 - on board
      setRow(row,pos,0, 1.0f);
      //Feature 18 - komi/15 from self perspective
      setRow(row,pos,18, selfKomi/15.0f);

      Color stone = board.colors[loc];

      //Features 1,2 - pla,opp stone
      //Features 3,4,5 and 6,7,8 - pla 1,2,3 libs and opp 1,2,3 libs.
      if(stone == pla) {
        setRow(row,pos,1, 1.0f);
        int libs = board.getNumLiberties(loc);
        if(libs == 1) setRow(row,pos,3, 1.0f);
        else if(libs == 2) setRow(row,pos,4, 1.0f);
        else if(libs == 3) setRow(row,pos,5, 1.0f);
      }
      else if(stone == opp) {
        setRow(row,pos,2, 1.0f);
        int libs = board.getNumLiberties(loc);
        if(libs == 1) setRow(row,pos,6, 1.0f);
        else if(libs == 2) setRow(row,pos,7, 1.0f);
        else if(libs == 3) setRow(row,pos,8, 1.0f);
      }
    }
  }

  //TODO superko
  //Feature 9 - simple ko location
  if(board.ko_loc != Board::NULL_LOC) {
    int pos = NNPos::locToPos(board.ko_loc,bSize,offset);
    setRow(row,pos,9, 1.0f);
  }

  //Features 10,11,12,13,14
  if(moveHistoryLen >= 1 && moveHistory[moveHistoryLen-1].pla == opp) {
    Loc prev1Loc = moveHistory[moveHistoryLen-1].loc;
    if(prev1Loc != Board::PASS_LOC && prev1Loc != Board::NULL_LOC) {
      int pos = NNPos::locToPos(prev1Loc,bSize,offset);
      setRow(row,pos,10, 1.0f);
    }
    if(moveHistoryLen >= 2 && moveHistory[moveHistoryLen-2].pla == pla) {
      Loc prev2Loc = moveHistory[moveHistoryLen-2].loc;
      if(prev2Loc != Board::PASS_LOC && prev2Loc != Board::NULL_LOC) {
        int pos = NNPos::locToPos(prev2Loc,bSize,offset);
        setRow(row,pos,11, 1.0f);
      }
      if(moveHistoryLen >= 3 && moveHistory[moveHistoryLen-3].pla == opp) {
        Loc prev3Loc = moveHistory[moveHistoryLen-3].loc;
        if(prev3Loc != Board::PASS_LOC && prev3Loc != Board::NULL_LOC) {
          int pos = NNPos::locToPos(prev3Loc,bSize,offset);
          setRow(row,pos,12, 1.0f);
        }
        if(moveHistoryLen >= 4 && moveHistory[moveHistoryLen-4].pla == pla) {
          Loc prev4Loc = moveHistory[moveHistoryLen-4].loc;
          if(prev4Loc != Board::PASS_LOC && prev4Loc != Board::NULL_LOC) {
            int pos = NNPos::locToPos(prev4Loc,bSize,offset);
            setRow(row,pos,13, 1.0f);
          }
          if(moveHistoryLen >= 5 && moveHistory[moveHistoryLen-5].pla == opp) {
            Loc prev5Loc = moveHistory[moveHistoryLen-5].loc;
            if(prev5Loc != Board::PASS_LOC && prev5Loc != Board::NULL_LOC) {
              int pos = NNPos::locToPos(prev5Loc,bSize,offset);
              setRow(row,pos,14, 1.0f);
            }
          }
        }
      }
    }
  }

  //Ladder features 15,16,17
  auto addLadderFeature = [&board,bSize,offset,row](Loc loc, int pos, const vector<Loc>& workingMoves){
    assert(board.colors[loc] == P_BLACK || board.colors[loc] == P_WHITE);
    int libs = board.getNumLiberties(loc);
    if(libs == 1)
      setRow(row,pos,15,1.0);
    else {
      setRow(row,pos,16,1.0);
      for(size_t j = 0; j < workingMoves.size(); j++) {
        int workingPos = NNPos::locToPos(workingMoves[j],bSize,offset);
        setRow(row,workingPos,17,1.0);
      }
    }
  };
  iterLadders(board, addLadderFeature);
}
