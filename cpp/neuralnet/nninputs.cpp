
#include "../neuralnet/nninputs.h"

int NNPos::xyToPos(int x, int y, int posLen) {
  return y * posLen + x;
}
int NNPos::locToPos(Loc loc, int boardXSize, int posLen) {
  if(loc == Board::PASS_LOC)
    return posLen * posLen;
  else if(loc == Board::NULL_LOC)
    return posLen * (posLen + 1);
  return Location::getY(loc,boardXSize) * posLen + Location::getX(loc,boardXSize);
}
Loc NNPos::posToLoc(int pos, int boardXSize, int boardYSize, int posLen) {
  if(pos == posLen * posLen)
    return Board::PASS_LOC;
  int x = pos % posLen;
  int y = pos / posLen;
  if(x < 0 || x >= boardXSize || y < 0 || y >= boardYSize)
    return Board::NULL_LOC;
  return Location::getLoc(x,y,boardXSize);
}

bool NNPos::isPassPos(int pos, int posLen) {
  return pos == posLen * posLen;
}

int NNPos::getPolicySize(int posLen) {
  return posLen * posLen + 1;
}

NNOutput::NNOutput() {}
NNOutput::NNOutput(const NNOutput& other) {
  nnHash = other.nnHash;
  whiteValue = other.whiteValue;
  std::copy(other.policyProbs, other.policyProbs+NNPos::MAX_NN_POLICY_SIZE, policyProbs);
}

double NNOutput::whiteValueOfWinner(Player winner, double drawValue) {
  if(winner == P_WHITE)
    return 1.0;
  else if(winner == P_BLACK)
    return -1.0;
  return drawValue;
}

double NNOutput::whiteValueOfScore(double finalWhiteMinusBlackScore, const Board& b) {
  if(b.x_size == b.y_size)
    return tanh(finalWhiteMinusBlackScore / (2*b.x_size));
  else
    return tanh(finalWhiteMinusBlackScore / (2*sqrt(b.x_size*b.y_size)));
}



static void setRowV0(float* row, int pos, int feature, float value, int posStride, int featureStride) {
  row[pos * posStride + feature * featureStride] = value;
}
static void setRowV1(float* row, int pos, int feature, float value, int posStride, int featureStride) {
  row[pos * posStride + feature * featureStride] = value;
}
static void setRowV2(float* row, int pos, int feature, float value, int posStride, int featureStride) {
  row[pos * posStride + feature * featureStride] = value;
}
static void setRowBinV3(bool* rowBin, int pos, int feature, bool value, int posStride, int featureStride) {
  rowBin[pos * posStride + feature * featureStride] = value;
}


//Calls f on each location that is part of an inescapable atari, or a group that can be put into inescapable atari
static void iterLadders(const Board& board, int posLen, std::function<void(Loc,int,const vector<Loc>&)> f) {
  int xSize = board.x_size;
  int ySize = board.y_size;

  Loc chainHeadsSolved[xSize*ySize];
  bool chainHeadsSolvedValue[xSize*ySize];
  int numChainHeadsSolved = 0;
  Board copy(board);
  vector<Loc> buf;
  vector<Loc> workingMoves;

  for(int y = 0; y<ySize; y++) {
    for(int x = 0; x<xSize; x++) {
      int pos = NNPos::xyToPos(x,y,posLen);
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


Hash128 NNInputs::getHashV0(
  const Board& board, const vector<Move>& moveHistory, int moveHistoryLen,
  Player nextPlayer, float selfKomi
) {
  Hash128 hash = board.pos_hash;
  hash ^= Board::ZOBRIST_PLAYER_HASH[nextPlayer];

  (void)moveHistory;
  (void)moveHistoryLen;

  if(board.ko_loc != Board::NULL_LOC)
    hash ^= Board::ZOBRIST_KO_LOC_HASH[board.ko_loc];

  int64_t komiX2 = (int64_t)(selfKomi*2.0f);
  uint64_t komiHash = Hash::murmurMix((uint64_t)komiX2);
  hash.hash0 ^= komiHash;
  hash.hash1 ^= Hash::basicLCong(komiHash);

  return hash;
}


void NNInputs::fillRowV0(
  const Board& board, const vector<Move>& moveHistory, int moveHistoryLen,
  Player nextPlayer, float selfKomi, int posLen, bool useNHWC, float* row
) {
  assert(moveHistoryLen <= moveHistory.size());
  std::fill(row,row+ROW_SIZE_V0,0.0f);

  Player pla = nextPlayer;
  Player opp = getOpp(pla);
  int xSize = board.x_size;
  int ySize = board.y_size;

  int featureStride;
  int posStride;
  if(useNHWC) {
    featureStride = 1;
    posStride = NNInputs::NUM_FEATURES_V0;
  }
  else {
    featureStride = posLen * posLen;
    posStride = 1;
  }

  assert(xSize <= posLen);
  assert(ySize <= posLen);

  for(int y = 0; y<ySize; y++) {
    for(int x = 0; x<xSize; x++) {
      int pos = NNPos::xyToPos(x,y,posLen);
      Loc loc = Location::getLoc(x,y,xSize);

      //Feature 0 - on board
      setRowV0(row,pos,0, 1.0f, posStride, featureStride);
      //Feature 18 - komi/15 from self perspective
      setRowV0(row,pos,18, selfKomi/15.0f, posStride, featureStride);

      Color stone = board.colors[loc];

      //Features 1,2 - pla,opp stone
      //Features 3,4,5 and 6,7,8 - pla 1,2,3 libs and opp 1,2,3 libs.
      if(stone == pla) {
        setRowV0(row,pos,1, 1.0f, posStride, featureStride);
        int libs = board.getNumLiberties(loc);
        if(libs == 1) setRowV0(row,pos,3, 1.0f, posStride, featureStride);
        else if(libs == 2) setRowV0(row,pos,4, 1.0f, posStride, featureStride);
        else if(libs == 3) setRowV0(row,pos,5, 1.0f, posStride, featureStride);
      }
      else if(stone == opp) {
        setRowV0(row,pos,2, 1.0f, posStride, featureStride);
        int libs = board.getNumLiberties(loc);
        if(libs == 1) setRowV0(row,pos,6, 1.0f, posStride, featureStride);
        else if(libs == 2) setRowV0(row,pos,7, 1.0f, posStride, featureStride);
        else if(libs == 3) setRowV0(row,pos,8, 1.0f, posStride, featureStride);
      }
    }
  }

  //Feature 9 - simple ko location
  if(board.ko_loc != Board::NULL_LOC) {
    int pos = NNPos::locToPos(board.ko_loc,xSize,posLen);
    setRowV0(row,pos,9, 1.0f, posStride, featureStride);
  }

  //Features 10,11,12,13,14
  if(moveHistoryLen >= 1 && moveHistory[moveHistoryLen-1].pla == opp) {
    Loc prev1Loc = moveHistory[moveHistoryLen-1].loc;
    if(prev1Loc != Board::PASS_LOC && prev1Loc != Board::NULL_LOC) {
      int pos = NNPos::locToPos(prev1Loc,xSize,posLen);
      setRowV0(row,pos,10, 1.0f, posStride, featureStride);
    }
    if(moveHistoryLen >= 2 && moveHistory[moveHistoryLen-2].pla == pla) {
      Loc prev2Loc = moveHistory[moveHistoryLen-2].loc;
      if(prev2Loc != Board::PASS_LOC && prev2Loc != Board::NULL_LOC) {
        int pos = NNPos::locToPos(prev2Loc,xSize,posLen);
        setRowV0(row,pos,11, 1.0f, posStride, featureStride);
      }
      if(moveHistoryLen >= 3 && moveHistory[moveHistoryLen-3].pla == opp) {
        Loc prev3Loc = moveHistory[moveHistoryLen-3].loc;
        if(prev3Loc != Board::PASS_LOC && prev3Loc != Board::NULL_LOC) {
          int pos = NNPos::locToPos(prev3Loc,xSize,posLen);
          setRowV0(row,pos,12, 1.0f, posStride, featureStride);
        }
        if(moveHistoryLen >= 4 && moveHistory[moveHistoryLen-4].pla == pla) {
          Loc prev4Loc = moveHistory[moveHistoryLen-4].loc;
          if(prev4Loc != Board::PASS_LOC && prev4Loc != Board::NULL_LOC) {
            int pos = NNPos::locToPos(prev4Loc,xSize,posLen);
            setRowV0(row,pos,13, 1.0f, posStride, featureStride);
          }
          if(moveHistoryLen >= 5 && moveHistory[moveHistoryLen-5].pla == opp) {
            Loc prev5Loc = moveHistory[moveHistoryLen-5].loc;
            if(prev5Loc != Board::PASS_LOC && prev5Loc != Board::NULL_LOC) {
              int pos = NNPos::locToPos(prev5Loc,xSize,posLen);
              setRowV0(row,pos,14, 1.0f, posStride, featureStride);
            }
          }
        }
      }
    }
  }

  //Ladder features 15,16,17
  auto addLadderFeature = [&board,xSize,posLen,posStride,featureStride,row](Loc loc, int pos, const vector<Loc>& workingMoves){
    assert(board.colors[loc] == P_BLACK || board.colors[loc] == P_WHITE);
    int libs = board.getNumLiberties(loc);
    if(libs == 1)
      setRowV0(row,pos,15,1.0, posStride, featureStride);
    else {
      setRowV0(row,pos,16,1.0, posStride, featureStride);
      for(size_t j = 0; j < workingMoves.size(); j++) {
        int workingPos = NNPos::locToPos(workingMoves[j],xSize,posLen);
        setRowV0(row,workingPos,17,1.0, posStride, featureStride);
      }
    }
  };
  iterLadders(board, posLen, addLadderFeature);
}


//===========================================================================================

//Currently does NOT depend on history (except for marking ko-illegal spots)
Hash128 NNInputs::getHashV1(
  const Board& board, const BoardHistory& hist, Player nextPlayer
) {
  int xSize = board.x_size;
  int ySize = board.y_size;

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

  float selfKomi = hist.currentSelfKomi(nextPlayer,0.0);
  int64_t komiX2 = (int64_t)(selfKomi*2.0f);
  uint64_t komiHash = Hash::murmurMix((uint64_t)komiX2);
  hash.hash0 ^= komiHash;
  hash.hash1 ^= Hash::basicLCong(komiHash);

  return hash;
}


void NNInputs::fillRowV1(
  const Board& board, const BoardHistory& hist, Player nextPlayer,
  int posLen, bool useNHWC, float* row
) {
  assert(posLen <= NNPos::MAX_BOARD_LEN);
  assert(board.x_size <= posLen);
  assert(board.y_size <= posLen);
  std::fill(row,row+ROW_SIZE_V1,0.0f);

  Player pla = nextPlayer;
  Player opp = getOpp(pla);
  int xSize = board.x_size;
  int ySize = board.y_size;

  int featureStride;
  int posStride;
  if(useNHWC) {
    featureStride = 1;
    posStride = NNInputs::NUM_FEATURES_V1;
  }
  else {
    featureStride = posLen * posLen;
    posStride = 1;
  }

  float selfKomi = hist.currentSelfKomi(nextPlayer,0.0);

  for(int y = 0; y<ySize; y++) {
    for(int x = 0; x<xSize; x++) {
      int pos = NNPos::xyToPos(x,y,posLen);
      Loc loc = Location::getLoc(x,y,xSize);

      //Feature 0 - on board
      setRowV1(row,pos,0, 1.0f, posStride, featureStride);
      //Feature 18 - komi/15 from self perspective
      setRowV1(row,pos,18, selfKomi/15.0f, posStride, featureStride);

      Color stone = board.colors[loc];

      //Features 1,2 - pla,opp stone
      //Features 3,4,5 and 6,7,8 - pla 1,2,3 libs and opp 1,2,3 libs.
      if(stone == pla) {
        setRowV1(row,pos,1, 1.0f, posStride, featureStride);
        int libs = board.getNumLiberties(loc);
        if(libs == 1) setRowV1(row,pos,3, 1.0f, posStride, featureStride);
        else if(libs == 2) setRowV1(row,pos,4, 1.0f, posStride, featureStride);
        else if(libs == 3) setRowV1(row,pos,5, 1.0f, posStride, featureStride);
      }
      else if(stone == opp) {
        setRowV1(row,pos,2, 1.0f, posStride, featureStride);
        int libs = board.getNumLiberties(loc);
        if(libs == 1) setRowV1(row,pos,6, 1.0f, posStride, featureStride);
        else if(libs == 2) setRowV1(row,pos,7, 1.0f, posStride, featureStride);
        else if(libs == 3) setRowV1(row,pos,8, 1.0f, posStride, featureStride);
      }
    }
  }

  //Feature 9 - ko-ban locations, including possibly superko
  if(hist.encorePhase == 0) {
    if(board.ko_loc != Board::NULL_LOC) {
      int pos = NNPos::locToPos(board.ko_loc,xSize,posLen);
      setRowV1(row,pos,9, 1.0f, posStride, featureStride);
    }
    for(int y = 0; y<ySize; y++) {
      for(int x = 0; x<xSize; x++) {
        Loc loc = Location::getLoc(x,y,xSize);
        if(hist.superKoBanned[loc] && loc != board.ko_loc) {
          int pos = NNPos::locToPos(loc,xSize,posLen);
          setRowV1(row,pos,9, 1.0f, posStride, featureStride);
        }
      }
    }
  }
  else {
    for(int y = 0; y<ySize; y++) {
      for(int x = 0; x<xSize; x++) {
        Loc loc = Location::getLoc(x,y,xSize);
        if(hist.superKoBanned[loc]) {
          int pos = NNPos::locToPos(loc,xSize,posLen);
          setRowV1(row,pos,9, 1.0f, posStride, featureStride);
        }
      }
    }
  }

  //Features 10,11,12,13,14
  const vector<Move>& moveHistory = hist.moveHistory;
  size_t moveHistoryLen = moveHistory.size();
  if(moveHistoryLen >= 1 && moveHistory[moveHistoryLen-1].pla == opp) {
    Loc prev1Loc = moveHistory[moveHistoryLen-1].loc;
    if(prev1Loc != Board::PASS_LOC && prev1Loc != Board::NULL_LOC) {
      int pos = NNPos::locToPos(prev1Loc,xSize,posLen);
      setRowV1(row,pos,10, 1.0f, posStride, featureStride);
    }
    if(moveHistoryLen >= 2 && moveHistory[moveHistoryLen-2].pla == pla) {
      Loc prev2Loc = moveHistory[moveHistoryLen-2].loc;
      if(prev2Loc != Board::PASS_LOC && prev2Loc != Board::NULL_LOC) {
        int pos = NNPos::locToPos(prev2Loc,xSize,posLen);
        setRowV1(row,pos,11, 1.0f, posStride, featureStride);
      }
      if(moveHistoryLen >= 3 && moveHistory[moveHistoryLen-3].pla == opp) {
        Loc prev3Loc = moveHistory[moveHistoryLen-3].loc;
        if(prev3Loc != Board::PASS_LOC && prev3Loc != Board::NULL_LOC) {
          int pos = NNPos::locToPos(prev3Loc,xSize,posLen);
          setRowV1(row,pos,12, 1.0f, posStride, featureStride);
        }
        if(moveHistoryLen >= 4 && moveHistory[moveHistoryLen-4].pla == pla) {
          Loc prev4Loc = moveHistory[moveHistoryLen-4].loc;
          if(prev4Loc != Board::PASS_LOC && prev4Loc != Board::NULL_LOC) {
            int pos = NNPos::locToPos(prev4Loc,xSize,posLen);
            setRowV1(row,pos,13, 1.0f, posStride, featureStride);
          }
          if(moveHistoryLen >= 5 && moveHistory[moveHistoryLen-5].pla == opp) {
            Loc prev5Loc = moveHistory[moveHistoryLen-5].loc;
            if(prev5Loc != Board::PASS_LOC && prev5Loc != Board::NULL_LOC) {
              int pos = NNPos::locToPos(prev5Loc,xSize,posLen);
              setRowV1(row,pos,14, 1.0f, posStride, featureStride);
            }
          }
        }
      }
    }
  }

  //Ladder features 15,16,17
  auto addLadderFeature = [&board,xSize,posLen,posStride,featureStride,row](Loc loc, int pos, const vector<Loc>& workingMoves){
    assert(board.colors[loc] == P_BLACK || board.colors[loc] == P_WHITE);
    int libs = board.getNumLiberties(loc);
    if(libs == 1)
      setRowV1(row,pos,15,1.0, posStride, featureStride);
    else {
      setRowV1(row,pos,16,1.0, posStride, featureStride);
      for(size_t j = 0; j < workingMoves.size(); j++) {
        int workingPos = NNPos::locToPos(workingMoves[j],xSize,posLen);
        setRowV1(row,workingPos,17,1.0, posStride, featureStride);
      }
    }
  };
  iterLadders(board, posLen, addLadderFeature);
}




//===========================================================================================

//Currently does NOT depend on history (except for marking ko-illegal spots)
Hash128 NNInputs::getHashV2(
  const Board& board, const BoardHistory& hist, Player nextPlayer
) {
  int xSize = board.x_size;
  int ySize = board.y_size;

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

  float selfKomi = hist.currentSelfKomi(nextPlayer,0.0);
  int64_t komiX2 = (int64_t)(selfKomi*2.0f);
  uint64_t komiHash = Hash::murmurMix((uint64_t)komiX2);
  hash.hash0 ^= komiHash;
  hash.hash1 ^= Hash::basicLCong(komiHash);

  return hash;
}


void NNInputs::fillRowV2(
  const Board& board, const BoardHistory& hist, Player nextPlayer,
  int posLen, bool useNHWC, float* row
) {
  assert(posLen <= NNPos::MAX_BOARD_LEN);
  assert(board.x_size <= posLen);
  assert(board.y_size <= posLen);
  std::fill(row,row+ROW_SIZE_V2,0.0f);

  Player pla = nextPlayer;
  Player opp = getOpp(pla);
  int xSize = board.x_size;
  int ySize = board.y_size;

  int featureStride;
  int posStride;
  if(useNHWC) {
    featureStride = 1;
    posStride = NNInputs::NUM_FEATURES_V2;
  }
  else {
    featureStride = posLen * posLen;
    posStride = 1;
  }

  float selfKomi = hist.currentSelfKomi(nextPlayer,0.0);

  for(int y = 0; y<ySize; y++) {
    for(int x = 0; x<xSize; x++) {
      int pos = NNPos::xyToPos(x,y,posLen);
      Loc loc = Location::getLoc(x,y,xSize);

      //Feature 0 - on board
      setRowV2(row,pos,0, 1.0f, posStride, featureStride);
      //Feature 16 - komi/15 from self perspective
      setRowV2(row,pos,16, selfKomi/15.0f, posStride, featureStride);

      Color stone = board.colors[loc];

      //Features 1,2 - pla,opp stone
      //Features 3,4,5 - 1,2,3 libs
      if(stone == pla)
        setRowV2(row,pos,1, 1.0f, posStride, featureStride);
      else if(stone == opp)
        setRowV2(row,pos,2, 1.0f, posStride, featureStride);

      if(stone == pla || stone == opp) {
        int libs = board.getNumLiberties(loc);
        if(libs == 1) setRowV2(row,pos,3, 1.0f, posStride, featureStride);
        else if(libs == 2) setRowV2(row,pos,4, 1.0f, posStride, featureStride);
        else if(libs == 3) setRowV2(row,pos,5, 1.0f, posStride, featureStride);
      }
    }
  }

  //Feature 6 - ko-ban locations, including possibly superko. Or in the encore, no-second-ko-capture locations
  if(hist.encorePhase == 0) {
    if(board.ko_loc != Board::NULL_LOC) {
      int pos = NNPos::locToPos(board.ko_loc,xSize,posLen);
      setRowV2(row,pos,6, 1.0f, posStride, featureStride);
    }
    for(int y = 0; y<ySize; y++) {
      for(int x = 0; x<xSize; x++) {
        Loc loc = Location::getLoc(x,y,xSize);
        if(hist.superKoBanned[loc] && loc != board.ko_loc) {
          int pos = NNPos::locToPos(loc,xSize,posLen);
          setRowV2(row,pos,6, 1.0f, posStride, featureStride);
        }
      }
    }
  }
  else {
    for(int y = 0; y<ySize; y++) {
      for(int x = 0; x<xSize; x++) {
        Loc loc = Location::getLoc(x,y,xSize);
        if(hist.superKoBanned[loc]) {
          int pos = NNPos::locToPos(loc,xSize,posLen);
          setRowV2(row,pos,6, 1.0f, posStride, featureStride);
        }
      }
    }
  }

  //Features 7,8,9,10,11
  const vector<Move>& moveHistory = hist.moveHistory;
  size_t moveHistoryLen = moveHistory.size();
  if(moveHistoryLen >= 1 && moveHistory[moveHistoryLen-1].pla == opp) {
    Loc prev1Loc = moveHistory[moveHistoryLen-1].loc;
    if(prev1Loc != Board::PASS_LOC && prev1Loc != Board::NULL_LOC) {
      int pos = NNPos::locToPos(prev1Loc,xSize,posLen);
      setRowV2(row,pos,7, 1.0f, posStride, featureStride);
    }
    if(moveHistoryLen >= 2 && moveHistory[moveHistoryLen-2].pla == pla) {
      Loc prev2Loc = moveHistory[moveHistoryLen-2].loc;
      if(prev2Loc != Board::PASS_LOC && prev2Loc != Board::NULL_LOC) {
        int pos = NNPos::locToPos(prev2Loc,xSize,posLen);
        setRowV2(row,pos,8, 1.0f, posStride, featureStride);
      }
      if(moveHistoryLen >= 3 && moveHistory[moveHistoryLen-3].pla == opp) {
        Loc prev3Loc = moveHistory[moveHistoryLen-3].loc;
        if(prev3Loc != Board::PASS_LOC && prev3Loc != Board::NULL_LOC) {
          int pos = NNPos::locToPos(prev3Loc,xSize,posLen);
          setRowV2(row,pos,9, 1.0f, posStride, featureStride);
        }
        if(moveHistoryLen >= 4 && moveHistory[moveHistoryLen-4].pla == pla) {
          Loc prev4Loc = moveHistory[moveHistoryLen-4].loc;
          if(prev4Loc != Board::PASS_LOC && prev4Loc != Board::NULL_LOC) {
            int pos = NNPos::locToPos(prev4Loc,xSize,posLen);
            setRowV2(row,pos,10, 1.0f, posStride, featureStride);
          }
          if(moveHistoryLen >= 5 && moveHistory[moveHistoryLen-5].pla == opp) {
            Loc prev5Loc = moveHistory[moveHistoryLen-5].loc;
            if(prev5Loc != Board::PASS_LOC && prev5Loc != Board::NULL_LOC) {
              int pos = NNPos::locToPos(prev5Loc,xSize,posLen);
              setRowV2(row,pos,11, 1.0f, posStride, featureStride);
            }
          }
        }
      }
    }
  }

  //Ladder features 12,13,14,15
  auto addLadderFeature = [&board,xSize,posLen,posStride,featureStride,row,opp](Loc loc, int pos, const vector<Loc>& workingMoves){
    assert(board.colors[loc] == P_BLACK || board.colors[loc] == P_WHITE);
    assert(pos >= 0 && pos < NNPos::MAX_BOARD_AREA);
    setRowV2(row,pos,12,1.0, posStride, featureStride);
    if(board.colors[loc] == opp && board.getNumLiberties(loc) > 1) {
      for(size_t j = 0; j < workingMoves.size(); j++) {
        int workingPos = NNPos::locToPos(workingMoves[j],xSize,posLen);
        setRowV2(row,workingPos,15,1.0, posStride, featureStride);
      }
    }
  };

  iterLadders(board, posLen, addLadderFeature);

  const Board& prevBoard = hist.getRecentBoard(1);
  auto addPrevLadderFeature = [&prevBoard,posStride,featureStride,row](Loc loc, int pos, const vector<Loc>& workingMoves){
    (void)workingMoves;
    (void)loc;
    assert(prevBoard.colors[loc] == P_BLACK || prevBoard.colors[loc] == P_WHITE);
    assert(pos >= 0 && pos < NNPos::MAX_BOARD_AREA);
    setRowV2(row,pos,13,1.0, posStride, featureStride);
  };
  iterLadders(prevBoard, posLen, addPrevLadderFeature);

  const Board& prevPrevBoard = hist.getRecentBoard(2);
  auto addPrevPrevLadderFeature = [&prevPrevBoard,posStride,featureStride,row](Loc loc, int pos, const vector<Loc>& workingMoves){
    (void)workingMoves;
    (void)loc;
    assert(prevPrevBoard.colors[loc] == P_BLACK || prevPrevBoard.colors[loc] == P_WHITE);
    assert(pos >= 0 && pos < NNPos::MAX_BOARD_AREA);
    setRowV2(row,pos,14,1.0, posStride, featureStride);
  };
  iterLadders(prevPrevBoard, posLen, addPrevPrevLadderFeature);

}



//===========================================================================================

//Currently does NOT depend on history (except for marking ko-illegal spots)
Hash128 NNInputs::getHashV3(
  const Board& board, const BoardHistory& hist, Player nextPlayer,
  double drawUtilityForWhite
) {
  int xSize = board.x_size;
  int ySize = board.y_size;

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

  //TODO incorporate pass ends phase into hash, if we do end up using pass ends phase.

  //For the purpose of computing the hash, discretize drawUtilityForWhite down to units of 1/16
  double drawUtilityForWhiteDiscretized = round(drawUtilityForWhite*16.0)/16.0;
  float selfKomi = hist.currentSelfKomi(nextPlayer,drawUtilityForWhiteDiscretized);
  int64_t komiDiscretized = (int64_t)(selfKomi*256.0f);
  uint64_t komiHash = Hash::murmurMix((uint64_t)komiDiscretized);
  hash.hash0 ^= komiHash;
  hash.hash1 ^= Hash::basicLCong(komiHash);

  //Fold in the ko, scoring, and suicide rules
  hash ^= Rules::ZOBRIST_KO_RULE_HASH[hist.rules.koRule];
  hash ^= Rules::ZOBRIST_SCORING_RULE_HASH[hist.rules.scoringRule];
  if(hist.rules.multiStoneSuicideLegal)
    hash ^= Rules::ZOBRIST_MULTI_STONE_SUICIDE_HASH;

  return hash;
}


void NNInputs::fillRowV3(
  const Board& board, const BoardHistory& hist, Player nextPlayer,
  double drawUtilityForWhite, int posLen, bool useNHWC, bool* rowBin, float* rowFloat
) {
  assert(posLen <= NNPos::MAX_BOARD_LEN);
  assert(board.x_size <= posLen);
  assert(board.y_size <= posLen);
  std::fill(rowBin,rowBin+ROW_SIZE_BIN_V3,false);
  std::fill(rowFloat,rowFloat+ROW_SIZE_FLOAT_V3,0.0f);

  Player pla = nextPlayer;
  Player opp = getOpp(pla);
  int xSize = board.x_size;
  int ySize = board.y_size;

  int featureStride;
  int posStride;
  if(useNHWC) {
    featureStride = 1;
    posStride = NNInputs::NUM_FEATURES_BIN_V3;
  }
  else {
    featureStride = posLen * posLen;
    posStride = 1;
  }

  for(int y = 0; y<ySize; y++) {
    for(int x = 0; x<xSize; x++) {
      int pos = NNPos::xyToPos(x,y,posLen);
      Loc loc = Location::getLoc(x,y,xSize);

      //Feature 0 - on board
      setRowBinV3(rowBin,pos,0, true, posStride, featureStride);

      Color stone = board.colors[loc];

      //Features 1,2 - pla,opp stone
      //Features 3,4,5 - 1,2,3 libs
      if(stone == pla)
        setRowBinV3(rowBin,pos,1, true, posStride, featureStride);
      else if(stone == opp)
        setRowBinV3(rowBin,pos,2, true, posStride, featureStride);

      if(stone == pla || stone == opp) {
        int libs = board.getNumLiberties(loc);
        if(libs == 1) setRowBinV3(rowBin,pos,3, true, posStride, featureStride);
        else if(libs == 2) setRowBinV3(rowBin,pos,4, true, posStride, featureStride);
        else if(libs == 3) setRowBinV3(rowBin,pos,5, true, posStride, featureStride);
      }
    }
  }

  //Feature 6 - ko-ban locations, including possibly superko.
  if(hist.encorePhase == 0) {
    if(board.ko_loc != Board::NULL_LOC) {
      int pos = NNPos::locToPos(board.ko_loc,xSize,posLen);
      setRowBinV3(rowBin,pos,6, true, posStride, featureStride);
    }
    for(int y = 0; y<ySize; y++) {
      for(int x = 0; x<xSize; x++) {
        Loc loc = Location::getLoc(x,y,xSize);
        if(hist.superKoBanned[loc] && loc != board.ko_loc) {
          int pos = NNPos::locToPos(loc,xSize,posLen);
          setRowBinV3(rowBin,pos,6, true, posStride, featureStride);
        }
      }
    }
  }
  else {
    //Feature 6,7,8 - in the encore, no-second-ko-capture locations, encore ko prohibitions where we have to pass for ko
    for(int y = 0; y<ySize; y++) {
      for(int x = 0; x<xSize; x++) {
        Loc loc = Location::getLoc(x,y,xSize);
        int pos = NNPos::locToPos(loc,xSize,posLen);
        if(hist.superKoBanned[loc])
          setRowBinV3(rowBin,pos,6, true, posStride, featureStride);
        if(hist.blackKoProhibited[loc])
          setRowBinV3(rowBin,pos,7, true, posStride, featureStride);
        if(hist.whiteKoProhibited[loc])
          setRowBinV3(rowBin,pos,8, true, posStride, featureStride);
      }
    }
  }

  //TODO Pass can be encoded as -1/128 on every spot.
  //Features 9,10,11,12,13
  const vector<Move>& moveHistory = hist.moveHistory;
  size_t moveHistoryLen = moveHistory.size();
  if(moveHistoryLen >= 1 && moveHistory[moveHistoryLen-1].pla == opp) {
    Loc prev1Loc = moveHistory[moveHistoryLen-1].loc;
    if(prev1Loc != Board::PASS_LOC && prev1Loc != Board::NULL_LOC) {
      int pos = NNPos::locToPos(prev1Loc,xSize,posLen);
      setRowBinV3(rowBin,pos,9, true, posStride, featureStride);
    }
    if(moveHistoryLen >= 2 && moveHistory[moveHistoryLen-2].pla == pla) {
      Loc prev2Loc = moveHistory[moveHistoryLen-2].loc;
      if(prev2Loc != Board::PASS_LOC && prev2Loc != Board::NULL_LOC) {
        int pos = NNPos::locToPos(prev2Loc,xSize,posLen);
        setRowBinV3(rowBin,pos,10, true, posStride, featureStride);
      }
      if(moveHistoryLen >= 3 && moveHistory[moveHistoryLen-3].pla == opp) {
        Loc prev3Loc = moveHistory[moveHistoryLen-3].loc;
        if(prev3Loc != Board::PASS_LOC && prev3Loc != Board::NULL_LOC) {
          int pos = NNPos::locToPos(prev3Loc,xSize,posLen);
          setRowBinV3(rowBin,pos,11, true, posStride, featureStride);
        }
        if(moveHistoryLen >= 4 && moveHistory[moveHistoryLen-4].pla == pla) {
          Loc prev4Loc = moveHistory[moveHistoryLen-4].loc;
          if(prev4Loc != Board::PASS_LOC && prev4Loc != Board::NULL_LOC) {
            int pos = NNPos::locToPos(prev4Loc,xSize,posLen);
            setRowBinV3(rowBin,pos,12, true, posStride, featureStride);
          }
          if(moveHistoryLen >= 5 && moveHistory[moveHistoryLen-5].pla == opp) {
            Loc prev5Loc = moveHistory[moveHistoryLen-5].loc;
            if(prev5Loc != Board::PASS_LOC && prev5Loc != Board::NULL_LOC) {
              int pos = NNPos::locToPos(prev5Loc,xSize,posLen);
              setRowBinV3(rowBin,pos,13, true, posStride, featureStride);
            }
          }
        }
      }
    }
  }

  //Ladder features 14,15,16,17
  auto addLadderFeature = [&board,xSize,posLen,posStride,featureStride,rowBin,opp](Loc loc, int pos, const vector<Loc>& workingMoves){
    assert(board.colors[loc] == P_BLACK || board.colors[loc] == P_WHITE);
    assert(pos >= 0 && pos < NNPos::MAX_BOARD_AREA);
    setRowBinV3(rowBin,pos,14, true, posStride, featureStride);
    if(board.colors[loc] == opp && board.getNumLiberties(loc) > 1) {
      for(size_t j = 0; j < workingMoves.size(); j++) {
        int workingPos = NNPos::locToPos(workingMoves[j],xSize,posLen);
        setRowBinV3(rowBin,workingPos,17, true, posStride, featureStride);
      }
    }
  };

  iterLadders(board, posLen, addLadderFeature);

  const Board& prevBoard = hist.getRecentBoard(1);
  auto addPrevLadderFeature = [&prevBoard,posStride,featureStride,rowBin](Loc loc, int pos, const vector<Loc>& workingMoves){
    (void)workingMoves;
    (void)loc;
    assert(prevBoard.colors[loc] == P_BLACK || prevBoard.colors[loc] == P_WHITE);
    assert(pos >= 0 && pos < NNPos::MAX_BOARD_AREA);
    setRowBinV3(rowBin,pos,15, true, posStride, featureStride);
  };
  iterLadders(prevBoard, posLen, addPrevLadderFeature);

  const Board& prevPrevBoard = hist.getRecentBoard(2);
  auto addPrevPrevLadderFeature = [&prevPrevBoard,posStride,featureStride,rowBin](Loc loc, int pos, const vector<Loc>& workingMoves){
    (void)workingMoves;
    (void)loc;
    assert(prevPrevBoard.colors[loc] == P_BLACK || prevPrevBoard.colors[loc] == P_WHITE);
    assert(pos >= 0 && pos < NNPos::MAX_BOARD_AREA);
    setRowBinV3(rowBin,pos,16, true, posStride, featureStride);
  };
  iterLadders(prevPrevBoard, posLen, addPrevPrevLadderFeature);

  //Features 18,19 - current territory
  Color area[Board::MAX_ARR_SIZE];
  if(hist.rules.scoringRule == Rules::SCORING_AREA) {
    bool nonPassAliveStones = true;
    bool safeBigTerritories = true;
    bool unsafeBigTerritories = true;
    board.calculateArea(area,nonPassAliveStones,safeBigTerritories,unsafeBigTerritories,hist.rules.multiStoneSuicideLegal);
  }
  else if(hist.rules.scoringRule == Rules::SCORING_TERRITORY) {
    bool nonPassAliveStones = false;
    bool safeBigTerritories = true;
    bool unsafeBigTerritories = false;
    board.calculateArea(area,nonPassAliveStones,safeBigTerritories,unsafeBigTerritories,hist.rules.multiStoneSuicideLegal);
  }
  else {
    assert(false);
  }

  for(int y = 0; y<ySize; y++) {
    for(int x = 0; x<xSize; x++) {
      Loc loc = Location::getLoc(x,y,xSize);
      int pos = NNPos::locToPos(loc,xSize,posLen);
      if(area[loc] == pla)
        setRowBinV3(rowBin,pos,18, true, posStride, featureStride);
      else if(area[loc] == opp)
        setRowBinV3(rowBin,pos,19, true, posStride, featureStride);
    }
  }


  //Floating point features
  float selfKomi = hist.currentSelfKomi(nextPlayer,drawUtilityForWhite);
  rowFloat[0] = selfKomi/15.0f;

  if(hist.rules.koRule == Rules::KO_SIMPLE) {}
  else if(hist.rules.koRule == Rules::KO_POSITIONAL || hist.rules.koRule == Rules::KO_SPIGHT) {
    rowFloat[1] = 1.0f;
    rowFloat[2] = 0.5f;
  }
  else if(hist.rules.koRule == Rules::KO_SITUATIONAL) {
    rowFloat[1] = 1.0f;
    rowFloat[2] = -0.5f;
  }
  else
    assert(false);

  if(hist.rules.multiStoneSuicideLegal)
    rowFloat[3] = 1.0f;

  if(hist.rules.scoringRule == Rules::SCORING_AREA) {}
  else if(hist.rules.scoringRule == Rules::SCORING_TERRITORY)
    rowFloat[4] = 1.0f;
  else
    assert(false);

  if(hist.encorePhase > 0)
    rowFloat[5] = 1.0f;

  if(hist.encorePhase > 1)
    rowFloat[6] = 1.0f;

  rowFloat[7] = sqrt((float)(xSize*ySize));

  bool komiIsInteger = ((int)hist.rules.komi == hist.rules.komi);
  if(!komiIsInteger) {
    bool boardAreaIsEven = (xSize*ySize) % 2 == 0;
    bool komiIsBelowEven = (((int)floor(hist.rules.komi + 1.0f) % 2) + 2) % 2 == 0;
    //TODO think about this and make sure this is right
    rowFloat[8] = ((boardAreaIsEven == komiIsBelowEven) == (pla == P_WHITE)) ? -0.5f : 0.5f;
  }

}
