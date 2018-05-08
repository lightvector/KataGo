#include "core/global.h"
#include "core/rand.h"
#include "fastboard.h"
#include "sgf.h"
#include "datapool.h"
#include <fstream>

#include <H5Cpp.h>
using namespace H5;

#define TCLAP_NAMESTARTSTRING "-" //Use single dashes for all flags
#include <tclap/CmdLine.h>

//Data and feature row parameters
static const int maxBoardSize = 19;
static const int numFeatures = 26;
static const int numRecentBoards = 6; //For recent captures

//Different segments of the data row
static const int inputStart = 0;
static const int inputLen = maxBoardSize * maxBoardSize * numFeatures;

static const int targetStart = inputStart + inputLen;
static const int targetLen = maxBoardSize * maxBoardSize;

static const int ladderTargetStart = targetStart + targetLen;
static const int ladderTargetLen = 0;
// static const int ladderTargetLen = maxBoardSize * maxBoardSize;

static const int targetWeightsStart = ladderTargetStart + ladderTargetLen;
static const int targetWeightsLen = 1;

static const int rankStart = targetWeightsStart + targetWeightsLen;
static const int rankLenGoGoD = 1; //pro
static const int rankLenKGS = 9; //1d-9d
static const int rankLenFox = 17 + 9; //17k-9d
static const int rankLenOGSPre2014 = 19 + 9; //19k-9d

static const int rankStartGoGoD = 0;
static const int rankStartKGS = rankLenGoGoD;
static const int rankStartFox = rankLenGoGoD + rankLenKGS;
static const int rankStartOGSPre2014 = rankLenGoGoD + rankLenKGS + rankLenFox;
static const int rankLen = rankLenGoGoD + rankLenKGS + rankLenFox + rankLenOGSPre2014;

static const int sideStart = rankStart + rankLen;
static const int sideLen = 1;

static const int turnNumberStart = sideStart + sideLen;
static const int turnNumberLen = 2;

static const int recentCapturesStart = turnNumberStart + turnNumberLen;
static const int recentCapturesLen = maxBoardSize * maxBoardSize;

static const int nextMovesStart = recentCapturesStart + recentCapturesLen;
static const int nextMovesLen = 7;

static const int sgfHashStart = nextMovesStart + nextMovesLen;
static const int sgfHashLen = 8;

static const int totalRowLen = sgfHashStart + sgfHashLen;

//HDF5 parameters
static const int chunkHeight = 6000;
static const int deflateLevel = 6;
static const int h5Dimension = 2;

//SGF sources
static const int NUM_SOURCES = 5;
static const int SOURCE_GOGOD = 0;
static const int SOURCE_KGS = 1;
static const int SOURCE_FOX = 2;
static const int SOURCE_OGSPre2014 = 3;
static const int SOURCE_UNKNOWN = 4;
static bool emittedSourceWarningYet = false;
static int parseSource(const string& fileName) {
  if(fileName.find("GoGoD") != string::npos)
    return SOURCE_GOGOD;
  else if(fileName.find("/KGS/") != string::npos || fileName.find("/KGS4d/") != string::npos)
    return SOURCE_KGS;
  else if(fileName.find("FoxGo") != string::npos)
    return SOURCE_FOX;
  else if(fileName.find("OGSPre2014") != string::npos)
    return SOURCE_OGSPre2014;
  else {
    if(!emittedSourceWarningYet) {
      cerr << "Note: unknown source for sgf " << fileName << endl;
      cerr << "There is some hardcoded logic for applying different filter conditions for known data sources (e.g. KGS, GoGoD, etc). If you would like to do filtering of your own, you can manually modify the parseSource function in write.cpp and/or add appropriate sources for your data, and add whatever conditions you like at appropriate points in the rest of write.cpp." << endl;
      cerr << "Suppressing further warnings for unknown sgf sources" << endl;
      emittedSourceWarningYet = true;
    }
  }
}

//When doing fancy conditions (cmdline flag -fancy-conditions), randomly keep games from source only with this prob
static const double sourceGameFancyProb[NUM_SOURCES] = {
  1.00, /* GoGoD */
  1.00, /* KGS */
  0.15, /* FOX */ //Fox dataset is enormously large, only keep some of the games to prevent it from dwarfing all others in training and using lots of memory when writing
  1.00, /* OGS */
  1.00, /* Unknown */
};

//When doing fancy conditions (cmdline flag -fancy-conditions), randomly keep training instances from source only with this prob
//These numbers are tuned to try to balance the number of games in the training set coming from each different rank of player
//on each different server.
static const double rankOneHotFancyProb[rankLen] = {
  1.00, /* GoGoD */
  0.30, 0.30, 0.20, 0.10, 0.20, 0.10, 0.20, 0.50, 1.00, /* KGS */

  0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.20, /* FOX 17k-10k */
  0.15, 0.15, 0.15, 0.15, 0.15, /* FOX 9k-5k */
  0.14, 0.125, 0.080, 0.060, /* FOX 4k-1k */
  0.040, 0.030, 0.025, 0.040, 0.060, /* FOX 1d-5d */
  0.140, 0.350, 0.800, 0.400, /* FOX 6d-9d */

  0.80, 0.80, 0.80, 0.80,  /* OGS 19k-16k */
  0.80, 0.80, 0.80, 0.80, 0.80,  /* OGS 15k-11k */
  0.80, 0.80, 0.80, 0.80, 0.80,  /* OGS 10k-6k */
  0.80, 1.00, 1.00, 1.00, 1.00,  /* OGS 5k-1k */
  1.00, 1.00, 1.00, 1.00, 1.00,  /* OGS 1d-5d */
  1.00, 1.00, 1.00, 1.00,        /* OGS 6d-9d */
};

//Each row contains a one-hot segment that indicates the rank of the player that made this move, differentiated by
//rank since ranks mean different things on different servers.
//Computes the index of the one-hot entry to fill (or -1 indicating to fill none of them)
static int computeRankOneHot(int source, int rank) {
  int rankOneHot = -1; //Fill nothing by default for unknown sources or ranks that are out-of-range
  if(source == SOURCE_GOGOD)
    rankOneHot = rankStartGoGoD;
  else if(source == SOURCE_KGS && rank >= 0 && rank <= 8)
    rankOneHot = rankStartKGS + rank;
  else if(source == SOURCE_FOX && rank >= -17 && rank <= 8)
    rankOneHot = rankStartFox + 17 + rank;
  else if(source == SOURCE_OGSPre2014 && rank >= -19 && rank <= 8)
    rankOneHot = rankStartOGSPre2014 + 19 + rank;

  assert(rankOneHot >= -1 && rankOneHot < rankLen);
  return rankOneHot;
}


static int xyToTensorPos(int x, int y, int offset) {
  return (y+offset) * maxBoardSize + (x+offset);
}
static int locToTensorPos(Loc loc, int bSize, int offset) {
  if(loc == FastBoard::PASS_LOC)
    return (bSize + offset) * maxBoardSize + (bSize + offset);
  return (Location::getY(loc,bSize) + offset) * maxBoardSize + (Location::getX(loc,bSize) + offset);
}

static void setRow(float* row, int pos, int feature, float value) {
  row[pos*numFeatures + feature] = value;
}

static const int TARGET_NEXT_MOVE_AND_LADDER = 0;

//Calls f on each location that is part of an inescapable atari, or a group that can be put into inescapable atari
static void iterLadders(const FastBoard& board, std::function<void(Loc,int,const vector<Loc>&)> f) {
  int bSize = board.x_size;
  int offset = (maxBoardSize - bSize) / 2;

  Loc chainHeadsSolved[bSize*bSize];
  bool chainHeadsSolvedValue[bSize*bSize];
  int numChainHeadsSolved = 0;
  FastBoard copy(board);
  vector<Loc> buf;
  vector<Loc> workingMoves;

  for(int y = 0; y<bSize; y++) {
    for(int x = 0; x<bSize; x++) {
      int pos = xyToTensorPos(x,y,offset);
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
// static void iterWouldBeLadder(const FastBoard& board, Player pla, std::function<void(Loc,int)> f) {
//   Player opp = getEnemy(pla);
//   int bSize = board.x_size;
//   int offset = (maxBoardSize - bSize) / 2;

//   FastBoard copy(board);
//   vector<Loc> buf;

//   for(int y = 0; y<bSize; y++) {
//     for(int x = 0; x<bSize; x++) {
//       int pos = xyToTensorPos(x,y,offset);
//       Loc loc = Location::getLoc(x,y,bSize);
//       Color stone = board.colors[loc];
//       if(stone == C_EMPTY && board.getNumLibertiesAfterPlay(loc,pla,3) == 2) {

//       }
//     }
//   }
// }

static void fillRow(const vector<FastBoard>& recentBoards, const vector<Move>& moves, int nextMoveIdx, int target, int rankOneHot, Hash128 sgfHash, float* row, Rand& rand, bool alwaysHistory) {
  const FastBoard& board = recentBoards[0];

  assert(board.x_size == board.y_size);
  assert(nextMoveIdx < moves.size());

  Player pla = moves[nextMoveIdx].pla;
  Player opp = getEnemy(pla);
  int bSize = board.x_size;
  int offset = (maxBoardSize - bSize) / 2;

  for(int y = 0; y<bSize; y++) {
    for(int x = 0; x<bSize; x++) {
      int pos = xyToTensorPos(x,y,offset);
      Loc loc = Location::getLoc(x,y,bSize);

      //Feature 0 - on board
      setRow(row,pos,0, 1.0);

      Color stone = board.colors[loc];

      //Features 1,2 - pla,opp stone
      //Features 3,4,5,6 and 7,8,9,10 - pla 1,2,3,4 libs and opp 1,2,3,4 libs.
      if(stone == pla) {
        setRow(row,pos,1, 1.0);
        int libs = board.getNumLiberties(loc);
        if(libs == 1) setRow(row,pos,3, 1.0);
        else if(libs == 2) setRow(row,pos,4, 1.0);
        else if(libs == 3) setRow(row,pos,5, 1.0);
        else if(libs == 4) setRow(row,pos,6, 1.0);
      }
      else if(stone == opp) {
        setRow(row,pos,2, 1.0);
        int libs = board.getNumLiberties(loc);
        if(libs == 1) setRow(row,pos,7, 1.0);
        else if(libs == 2) setRow(row,pos,8, 1.0);
        else if(libs == 3) setRow(row,pos,9, 1.0);
        else if(libs == 4) setRow(row,pos,10, 1.0);
      }

      if(stone == pla || stone == opp) {}
      else {
        //Feature 11,12,13 - 1, 2, 3 liberties after own play.
        //Feature 14,15,16 - 1, 2, 3 liberties after opponent play
        int plaLibAfterPlay = board.getNumLibertiesAfterPlay(loc,pla,4);
        int oppLibAfterPlay = board.getNumLibertiesAfterPlay(loc,opp,4);
        if(plaLibAfterPlay == 1)      setRow(row,pos,11, 1.0);
        else if(plaLibAfterPlay == 2) setRow(row,pos,12, 1.0);
        else if(plaLibAfterPlay == 3) setRow(row,pos,13, 1.0);

        if(oppLibAfterPlay == 1)      setRow(row,pos,14, 1.0);
        else if(oppLibAfterPlay == 2) setRow(row,pos,15, 1.0);
        else if(oppLibAfterPlay == 3) setRow(row,pos,16, 1.0);
      }
    }
  }

  //Feature 17 - simple ko location
  if(board.ko_loc != FastBoard::NULL_LOC) {
    int pos = locToTensorPos(board.ko_loc,bSize,offset);
    setRow(row,pos,17, 1.0);
  }

  //Probabilistically include prev move features
  //Features 18,19,20,21,22
  bool includePrev1 = alwaysHistory || rand.nextDouble() < 0.9;
  bool includePrev2 = alwaysHistory || (includePrev1 && rand.nextDouble() < 0.95);
  bool includePrev3 = alwaysHistory || (includePrev2 && rand.nextDouble() < 0.95);
  bool includePrev4 = alwaysHistory || (includePrev3 && rand.nextDouble() < 0.98);
  bool includePrev5 = alwaysHistory || (includePrev4 && rand.nextDouble() < 0.98);

  if(nextMoveIdx >= 1 && moves[nextMoveIdx-1].pla == opp && includePrev1) {
    Loc prev1Loc = moves[nextMoveIdx-1].loc;
    if(prev1Loc != FastBoard::PASS_LOC) {
      int pos = locToTensorPos(prev1Loc,bSize,offset);
      setRow(row,pos,18, 1.0);
    }
    if(nextMoveIdx >= 2 && moves[nextMoveIdx-2].pla == pla && includePrev2) {
      Loc prev2Loc = moves[nextMoveIdx-2].loc;
      if(prev2Loc != FastBoard::PASS_LOC) {
        int pos = locToTensorPos(prev2Loc,bSize,offset);
        setRow(row,pos,19, 1.0);
      }
      if(nextMoveIdx >= 3 && moves[nextMoveIdx-3].pla == opp && includePrev3) {
        Loc prev3Loc = moves[nextMoveIdx-3].loc;
        if(prev3Loc != FastBoard::PASS_LOC) {
          int pos = locToTensorPos(prev3Loc,bSize,offset);
          setRow(row,pos,20, 1.0);
        }
        if(nextMoveIdx >= 4 && moves[nextMoveIdx-4].pla == pla && includePrev4) {
          Loc prev4Loc = moves[nextMoveIdx-4].loc;
          if(prev4Loc != FastBoard::PASS_LOC) {
            int pos = locToTensorPos(prev4Loc,bSize,offset);
            setRow(row,pos,21, 1.0);
          }
          if(nextMoveIdx >= 5 && moves[nextMoveIdx-5].pla == opp && includePrev5) {
            Loc prev5Loc = moves[nextMoveIdx-5].loc;
            if(prev5Loc != FastBoard::PASS_LOC) {
              int pos = locToTensorPos(prev5Loc,bSize,offset);
              setRow(row,pos,22, 1.0);
            }
          }
        }
      }
    }
  }

  //Ladder features 23,24,25
  auto addLadderFeature = [&board,bSize,offset,row](Loc loc, int pos, const vector<Loc>& workingMoves){
    assert(board.colors[loc] == P_BLACK || board.colors[loc] == P_WHITE);
    int libs = board.getNumLiberties(loc);
    if(libs == 1)
      setRow(row,pos,23,1.0);
    else {
      setRow(row,pos,24,1.0);
      for(size_t j = 0; j < workingMoves.size(); j++) {
        int workingPos = locToTensorPos(workingMoves[j],bSize,offset);
        setRow(row,workingPos,25,1.0);
      }
    }
  };
  iterLadders(board, addLadderFeature);


  if(target == TARGET_NEXT_MOVE_AND_LADDER) {
    //Next move target
    Loc nextMoveLoc = moves[nextMoveIdx].loc;
    assert(nextMoveLoc != FastBoard::PASS_LOC);
    int nextMovePos = locToTensorPos(nextMoveLoc,bSize,offset);
    row[targetStart + nextMovePos] = 1.0;

    //Ladder target
    // auto addLadderTarget = [&board,row](Loc loc, int pos, const vector<Loc>& workingMoves){
    //   (void)workingMoves;
    //   assert(board.colors[loc] == P_BLACK || board.colors[loc] == P_WHITE);
    //   row[ladderTargetStart + pos] = 1.0;
    // };
    // iterLadders(board, addLadderTarget);
  }

  //Weight of the row, currently always 1.0
  row[targetWeightsStart] = 1.0;

  //One-hot indicating rank
  if(rankOneHot != -1)
    row[rankStart + rankOneHot] = 1.0;

  //Indicate the side to move, black = 0, white = 1
  if(pla == P_BLACK)
    row[sideStart] = 0.0;
  else
    row[sideStart] = 1.0;

  //Record what turn out of what turn it is
  row[turnNumberStart] = nextMoveIdx;
  row[turnNumberStart+1] = moves.size();

  //Record recent captures, by marking any positions where stones vanished between one board and the next
  for(int i = (int)recentBoards.size()-1; i >= 0; i--) {
    const FastBoard& b = recentBoards[i];
    const FastBoard& bPrev = recentBoards[i+1];
    for(int y = 0; y<bSize; y++) {
      for(int x = 0; x<bSize; x++) {
        Loc loc = Location::getLoc(x,y,bSize);
        if(b.colors[loc] == C_EMPTY && bPrev.colors[loc] != C_EMPTY) {
          int pos = xyToTensorPos(x,y,offset);
          row[recentCapturesStart+pos] = i+1;
        }
      }
    }
  }

  //Record next moves
  for(int i = 0; i<nextMovesLen; i++) {
    int idx = nextMoveIdx + i;
    if(idx >= moves.size())
      row[nextMovesStart+i] = locToTensorPos(FastBoard::PASS_LOC,bSize,offset);
    else {
      row[nextMovesStart+i] = locToTensorPos(moves[idx].loc,bSize,offset);
    }
  }

  //Record 16-bit chunks of sgf hash, so that later we can identify where this training example came from
  row[sgfHashStart+0] = (float)((sgfHash.hash1 >> 0) & 0xFFFF);
  row[sgfHashStart+1] = (float)((sgfHash.hash1 >> 16) & 0xFFFF);
  row[sgfHashStart+2] = (float)((sgfHash.hash1 >> 32) & 0xFFFF);
  row[sgfHashStart+3] = (float)((sgfHash.hash1 >> 48) & 0xFFFF);
  row[sgfHashStart+4] = (float)((sgfHash.hash0 >> 0) & 0xFFFF);
  row[sgfHashStart+5] = (float)((sgfHash.hash0 >> 16) & 0xFFFF);
  row[sgfHashStart+6] = (float)((sgfHash.hash0 >> 32) & 0xFFFF);
  row[sgfHashStart+7] = (float)((sgfHash.hash0 >> 48) & 0xFFFF);
}

static uint64_t parseHex64(const string& str) {
  assert(str.length() == 16);
  uint64_t x = 0;
  for(int i = 0; i<16; i++) {
    x *= 16;
    if(str[i] >= '0' && str[i] <= '9')
      x += str[i] - '0';
    else if(str[i] >= 'a' && str[i] <= 'f')
      x += str[i] - 'a' + 10;
    else if(str[i] >= 'A' && str[i] <= 'F')
      x += str[i] - 'A' + 10;
    else
      assert(false);
  }
  return x;
}

static int parseSource(const CompactSgf* sgf) {
  return parseSource(sgf->fileName);
}

static int parseHandicap(const string& handicap) {
  int h;
  bool suc = Global::tryStringToInt(handicap,h);
  if(!suc)
    throw IOError("Unknown handicap: " + handicap);
  return h;
}

static const int RANK_UNRANKED = -1000;

//2 kyu = -2, 1 kyu = -1, 1 dan = 0, 2 dan = 1, ...  higher is stronger, pros are assumed to be 9d.
static int parseRank(const string& rank, bool isGoGoD) {
  string r = Global::toLower(rank);

  //Special case parsings
  if(isGoGoD) {
    if(r == "meijin" || r == "kisei" || r == "insei" || r == "judan" || r == "holder")
      return 8;
  }

  if(r.length() < 2 || r.length() > 7)
    throw IOError("Could not parse rank: " + rank);

  int n = 0;
  bool isK = false;
  bool isD = false;
  bool isP = false;
  bool isA = false;
  bool isAK = false;
  if(r.length() == 2) {
    if(r[1] != 'k' && r[1] != 'd' && r[1] != 'p' && r[1] != 'a')
      throw IOError("Could not parse rank: " + rank);
    if(!Global::isDigits(r,0,1))
      throw IOError("Could not parse rank: " + rank);
    n = Global::parseDigits(r,0,1);
    isK = r[1] == 'k';
    isD = r[1] == 'd';
    isP = r[1] == 'p';
    isA = r[1] == 'a'; //a few GoGoD records use 'a' to represent amateur dan
  }
  else if(r.length() == 3) {
    if(r[2] != 'k' && r[2] != 'd' && r[2] != 'p' && r[2] != 'a')
      throw IOError("Could not parse rank: " + rank);
    if(!Global::isDigits(r,0,2))
      throw IOError("Could not parse rank: " + rank);
    n = Global::parseDigits(r,0,2);
    isK = r[2] == 'k';
    isD = r[2] == 'd';
    isP = r[2] == 'p';
    isA = r[2] == 'a'; //a few GoGoD records use 'a' to represent amateur dan
  }
  else if(r.length() == 4) {
    //UTF-8 for 级(kyu/grade)
    if(r[1] == '\xE7' && r[2] == '\xBA' && r[3] == '\xA7') {
      if(!Global::isDigits(r,0,1))
        throw IOError("Could not parse rank: " + rank);
      isK = true;
      n = Global::parseDigits(r,0,1);
    }
    //UTF-8 for 段(dan)
    else if(r[1] == '\xE6' && r[2] == '\xAE' && r[3] == '\xB5') {
      if(!Global::isDigits(r,0,1))
        throw IOError("Could not parse rank: " + rank);
      isD = true;
      n = Global::parseDigits(r,0,1);
    }
    else
      throw IOError("Could not parse rank: " + rank);
  }
  else if(r.length() == 5) {
    //UTF-8 for 级(kyu/grade)
    if(r[2] == '\xE7' && r[3] == '\xBA' && r[4] == '\xA7') {
      if(!Global::isDigits(r,0,2))
        throw IOError("Could not parse rank: " + rank);
      isK = true;
      n = Global::parseDigits(r,0,2);
    }
    //UTF-8 for 段(dan)
    else if(r[2] == '\xE6' && r[3] == '\xAE' && r[4] == '\xB5') {
      //FoxGo labels pro ranks like P6<chinese character for duan> for 6p
      if(r[0] == 'p' && Global::isDigits(r,1,2)) {
        isP = true;
        n = Global::parseDigits(r,1,2);
      }
      else {
        if(!Global::isDigits(r,0,2))
          throw IOError("Could not parse rank: " + rank);
        isD = true;
        n = Global::parseDigits(r,0,2);
      }
    }
    else
      throw IOError("Could not parse rank: " + rank);
  }
  else if(r.length() == 6) {
    //GoGoD often labels ranks like "6d ama"
    if(r[1] == 'd' && r[2] == ' ' && r[3] == 'a' && r[4] == 'm' && r[5] == 'a' && Global::isDigits(r,0,1)) {
      isA = true;
      n = Global::parseDigits(r,0,1);
    }
    //GoGoD often labels ranks like "1k ama"
    else if(r[1] == 'k' && r[2] == ' ' && r[3] == 'a' && r[4] == 'm' && r[5] == 'a' && Global::isDigits(r,0,1)) {
      isAK = true;
      n = Global::parseDigits(r,0,1);
    }
    else
      throw IOError("Could not parse rank: " + rank);
  }
  //GoGoD has rengos between various pros
  else if(r.length() == 7) {
    if(r[1] == 'd' && r[2] == ' ' && r[3] == '&' && r[4] == ' ' && r[6] == 'd' &&
       Global::isDigits(r,0,1) && Global::isDigits(r,5,6))
    {
      isD = true;
      n = std::min(Global::parseDigits(r,0,1),Global::parseDigits(r,5,6));
    }
    else
      throw IOError("Could not parse rank: " + rank);
  }
  else {
    throw IOError("Could not parse rank: " + rank);
  }

  if(isGoGoD) {
    if(isA)
      return n >= 9 ? 8 : n-1;
    //Treat GoGoD games as all 9d a large number of pros are labeled e.g. "3d" indicating 3 *professional* dan something like "3p".
    //There are some games involving genuinely amateur dan players, but it's basically impossible to tell from the rank whether it's
    //amateur or pro.
    else if(isD)
      return 8;
    else if(isP)
      return 8;
    //Even kyu games can refer to the old korean kyu which is actually quite strong. We go ahead and exclude everything
    //that's worse than 3k though, and anything else we label as 8d.
    else if(isK)
      return n >= 3 ? -n : 7;
    else if(isAK)
      return -n;
    else {
      throw IOError("Could not parse rank: " + rank);
    }
  }
  else {
    if(isK)
      return -n;
    else if(isD)
      return n >= 9 ? 8 : n-1;
    //Treat all professional dan ranks as 9d amateur
    else if(isP)
      return 8;
    else {
      assert(false);
      return 0;
    }
  }
}

struct Stats {
  size_t count;
  map<int,int64_t> countBySource;
  map<int,int64_t> countByRank;
  map<int,int64_t> countByOppRank;
  map<string,int64_t> countByUser;
  map<int,int64_t> countByHandicap;

  Stats()
    :count(),countBySource(),countByRank(),countByOppRank(),countByUser(),countByHandicap() {

  }

  void print() {
    cout << "Count: " << count << endl;
    cout << "Sources:" << endl;
    for(auto const& kv: countBySource) {
      cout << kv.first << " " << kv.second << endl;
    }
    cout << "Ranks:" << endl;
    for(auto const& kv: countByRank) {
      cout << kv.first << " " << kv.second << endl;
    }
    cout << "OppRanks:" << endl;
    for(auto const& kv: countByOppRank) {
      cout << kv.first << " " << kv.second << endl;
    }
    cout << "Handicap:" << endl;
    for(auto const& kv: countByHandicap) {
      cout << kv.first << " " << kv.second << endl;
    }
    cout << "Major Users:" << endl;
    for(auto const& kv: countByUser) {
      if(kv.second > count / 2000)
        cout << kv.first << " " << kv.second << endl;
    }
  }
};

static void iterSgfMoves(
  CompactSgf* sgf,
  //board,source,rank,oppRank,user,handicap,moves,index within moves
  std::function<void(const vector<FastBoard>&,int,int,int,const string&,int,const string&,const vector<Move>&,int,Hash128)> f
) {
  int bSize;
  int source;
  int wRank;
  int bRank;
  string wUser;
  string bUser;
  int handicap;
  string date;
  const vector<Move>* placementsBuf = NULL;
  const vector<Move>* movesBuf = NULL;
  try {
    bSize = sgf->bSize;
    const SgfNode& root = sgf->rootNode;

    source = parseSource(sgf);

    if(source == SOURCE_GOGOD) {
      //By default, assume pro rank in GoGod if not specified
      wRank = 8;
      bRank = 8;
      bool isGoGoD = true;
      try {
        if(root.hasProperty("WR"))
          wRank = parseRank(root.getSingleProperty("WR"),isGoGoD);
      }
      catch(const IOError &e) {
        cout << "Warning: " << sgf->fileName << ": " << e.message << endl;
      }
      try {
        if(root.hasProperty("BR"))
          bRank = parseRank(root.getSingleProperty("BR"),isGoGoD);
      }
      catch(const IOError &e) {
        cout << "Warning: " << sgf->fileName << ": " << e.message << endl;
      }
    }
    else {
      wRank = RANK_UNRANKED;
      bRank = RANK_UNRANKED;
      bool isGoGoD = false;
      if(root.hasProperty("WR"))
        wRank = parseRank(root.getSingleProperty("WR"),isGoGoD);
      if(root.hasProperty("BR"))
        bRank = parseRank(root.getSingleProperty("BR"),isGoGoD);
    }

    wUser = root.getSingleProperty("PW");
    bUser = root.getSingleProperty("PB");

    handicap = 0;
    if(root.hasProperty("HA"))
      handicap = parseHandicap(root.getSingleProperty("HA"));

    if(root.hasProperty("DT"))
      date = root.getSingleProperty("DT");

    //Apply some filters
    if(bSize != 19)
      return;

    placementsBuf = &(sgf->placements);
    movesBuf = &(sgf->moves);

    //OGS has a ton of garbage, for OGS require a minimum length
    //to try to filter out random demos and problems and such
    if(source == SOURCE_OGSPre2014) {
      if(movesBuf->size() < 40)
        return;
    }
  }
  catch(const IOError &e) {
    cout << "Skipping sgf file: " << sgf->fileName << ": " << e.message << endl;
    return;
  }

  const vector<Move>& placements = *placementsBuf;
  const vector<Move>& moves = *movesBuf;

  FastBoard initialBoard(bSize);
  for(int j = 0; j<placements.size(); j++) {
    Move m = placements[j];
    bool suc = initialBoard.setStone(m.loc,m.pla);
    if(!suc) {
      cout << sgf->fileName << endl;
      cout << ("Illegal stone placement " + Global::intToString(j)) << endl;
      cout << initialBoard << endl;
      return;
    }
  }

  //If there are multiple black moves in a row, then make them all right now.
  //Sometimes sgfs break the standard and do handicap setup in this way.
  int j = 0;
  if(moves.size() > 1 && moves[0].pla == P_BLACK && moves[1].pla == P_BLACK) {
    for(; j<moves.size(); j++) {
      Move m = moves[j];
      if(m.pla != P_BLACK)
        break;
      bool suc = initialBoard.playMove(m.loc,m.pla);
      if(!suc) {
        cout << sgf->fileName << endl;
        cout << ("Illegal move! " + Global::intToString(j)) << endl;
        cout << initialBoard << endl;
      }
    }
  }

  vector<FastBoard> recentBoards;
  for(int i = 0; i<numRecentBoards; i++)
    recentBoards.push_back(initialBoard);

  Player prevPla = C_EMPTY;
  for(; j<moves.size(); j++) {
    Move m = moves[j];

    //Forbid consecutive moves by the same player
    if(m.pla == prevPla) {
      //Multiple-consecutive-move-by-same-player issues are super-common on FoxGo, so don't print on Fox
      //Not actually sure how this happens. It's a large number of games, but still only a tiny percentage,
      //and it often happens well into the middle of the game, and definitely before the end of the game.
      if(source != SOURCE_FOX) {
        cout << sgf->fileName << endl;
        cout << ("Multiple moves in a row by same player at " + Global::intToString(j)) << endl;
        cout << recentBoards[0] << endl;
      }
      //Terminate reading from the game in this case
      break;
    }

    int rank = m.pla == P_WHITE ? wRank : bRank;
    int oppRank = m.pla == P_WHITE ? bRank : wRank;
    const string& user = m.pla == P_WHITE ? wUser : bUser;
    f(recentBoards,source,rank,oppRank,user,handicap,date,moves,j,sgf->hash);

    for(int dj = 0; dj<numRecentBoards && j-dj >= 0; dj++) {
      Move mv = moves[j-dj];
      bool suc = recentBoards[dj].playMove(mv.loc,mv.pla);
      if(!suc) {
        cout << sgf->fileName << endl;
        cout << ("Illegal move! " + Global::intToString(j)) << endl;
        cout << recentBoards[dj] << endl;
        break;
      }
    }

    prevPla = m.pla;
  }

  return;
}

static void iterSgfsMoves(
  vector<CompactSgf*> sgfs,
  uint64_t shardSeed, int numShards,
  const size_t& numMovesUsed, const size_t& curDataSetRow,
  //source,rank,user,handicap,date,moves,index within moves,
  std::function<void(const vector<FastBoard>&,int,int,int,const string&,int,const string&,const vector<Move>&,int,Hash128)> f
) {

  size_t numMovesItered = 0;
  size_t numMovesIteredOrSkipped = 0;

  for(int shard = 0; shard < numShards; shard++) {
    Rand shardRand(shardSeed);

    std::function<void(const vector<FastBoard>&,int,int,int,const string&,int,const string&,const vector<Move>&,int,Hash128)> g =
      [f,shard,numShards,&shardRand,&numMovesIteredOrSkipped,&numMovesItered](
        const vector<FastBoard>& recentBoards, int source, int rank, int oppRank, const string& user, int handicap, const string& date,
        const vector<Move>& moves, int moveIdx, Hash128 sgfHash
      ) {
      //Only use this move if it's within our shard.
      numMovesIteredOrSkipped++;
      if(numShards <= 1 || shard == shardRand.nextUInt(numShards)) {
        numMovesItered++;
        f(recentBoards,source,rank,oppRank,user,handicap,date,moves,moveIdx,sgfHash);
      }
    };

    for(int i = 0; i<sgfs.size(); i++) {
      if(i % 5000 == 0)
        cout << "Shard " << shard << " "
             << "processed " << i << "/" << sgfs.size() << " sgfs, "
             << "itered " << numMovesItered << " moves, "
             << "used " << numMovesUsed << " moves, "
             << "written " << curDataSetRow << " rows..." << endl;

      iterSgfMoves(sgfs[i],g);
    }
  }

  assert(numMovesIteredOrSkipped == numMovesItered * numShards);
  cout << "Over all shards, numMovesItered = " << numMovesItered
       << " numMovesIteredOrSkipped = " << numMovesIteredOrSkipped
       << " numMovesItered*numShards = " << (numMovesItered * numShards) << endl;
}

static void maybeUseRow(
  const vector<FastBoard>& recentBoards, int source, int rank, int oppRank, const string& user, int handicap,
  const string& date, const vector<Move>& movesBuf, int moveIdx, Hash128 sgfHash,
  DataPool& dataPool,
  Rand& rand, double keepProb, int minRank, int minOppRank, int maxHandicap, int target,
  bool alwaysHistory,
  const set<string>& excludeUsers, bool fancyConditions, double fancyPosKeepFactor,
  set<Hash>& posHashes, Stats& total, Stats& used
) {
  //TODO also filter out games that are > 85% identical hashes to another game
  //For now, only generate training rows for non-passes
  //Also only use moves by this player if that player meets rank threshold
  if(movesBuf[moveIdx].loc != FastBoard::PASS_LOC &&
     rank >= minRank &&
     oppRank >= minOppRank &&
     handicap <= maxHandicap &&
     !contains(excludeUsers,user)
  ) {
    int rankOneHot = computeRankOneHot(source,rank);
    bool canUse = true;

    //Apply special filtering for when we want to make a rank-balanced training set
    if(fancyConditions) {
      //Require that we have a good rank
      if(rankOneHot < 0)
        canUse = false;
      //Some ranks have too many games, filter them down
      if(rand.nextDouble() >= rankOneHotFancyProb[rankOneHot] * fancyPosKeepFactor)
        canUse = false;
      //No handicap games from GoGoD since they're less likely to be pro-level
      if(source == SOURCE_GOGOD && handicap >= 2)
        canUse = false;
      //No kyu moves from GoGoD, no amateur moves that are too weak
      if(source == SOURCE_GOGOD && rank < 4)
        canUse = false;
      //OGS had a major rank shift in 2014, only use games before
      if(source == SOURCE_OGSPre2014) {
        if(date.size() != 10)
          canUse = false;
        //Find year and month of date in format yyyy-mm-dd
        else if(!Global::isDigits(date,0,4) || !Global::isDigits(date,5,7))
          canUse = false;
        else {
          int year = Global::parseDigits(date,0,4);
          int month = Global::parseDigits(date,5,7);
          if((year >= 1990 && year <= 2013) || (year == 2014 && month <= 3))
          {} //good
          else
            canUse = false;
        }
      }
      //Fox Go has a bunch of games by usernameless people. Are they guests? Anyways let's filter that.
      if(source == SOURCE_FOX) {
        if(user.length() <= 0 || user == " ")
          canUse = false;
      }
    }

    if(canUse) {
      float* newRow = NULL;
      if(keepProb >= 1.0 || (rand.nextDouble() < keepProb))
        newRow = dataPool.addNewRow(rand);

      if(newRow != NULL) {
        assert(recentBoards.size() > 0);
        fillRow(recentBoards,movesBuf,moveIdx,target,rankOneHot,sgfHash,newRow,rand,alwaysHistory);
        posHashes.insert(recentBoards[0].pos_hash);

        used.count += 1;
        used.countBySource[source] += 1;
        used.countByRank[rank] += 1;
        used.countByOppRank[oppRank] += 1;
        used.countByUser[user] += 1;
        used.countByHandicap[handicap] += 1;
      }
    }
  }

  total.count += 1;
  total.countBySource[source] += 1;
  total.countByRank[rank] += 1;
  total.countByOppRank[oppRank] += 1;
  total.countByUser[user] += 1;
  total.countByHandicap[handicap] += 1;
}

static void processSgfs(
  vector<CompactSgf*> sgfs, DataSet* dataSet,
  size_t poolSize,
  uint64_t shardSeed, int numShards,
  Rand& rand, double keepProb,
  int minRank, int minOppRank, int maxHandicap, int target,
  bool alwaysHistory,
  const set<string>& excludeUsers, bool fancyConditions, double fancyPosKeepFactor,
  set<Hash>& posHashes, Stats& total, Stats& used
) {
  size_t curDataSetRow = 0;
  std::function<void(const float*,size_t)> writeRow = [&curDataSetRow,&dataSet](const float* rows, size_t numRows) {
    hsize_t newDims[h5Dimension] = {curDataSetRow+numRows,totalRowLen};
    dataSet->extend(newDims);
    DataSpace fileSpace = dataSet->getSpace();
    hsize_t memDims[h5Dimension] = {numRows,totalRowLen};
    DataSpace memSpace(h5Dimension,memDims);
    hsize_t start[h5Dimension] = {curDataSetRow,0};
    hsize_t count[h5Dimension] = {numRows,totalRowLen};
    fileSpace.selectHyperslab(H5S_SELECT_SET, count, start);
    dataSet->write(rows, PredType::NATIVE_FLOAT, memSpace, fileSpace);
    curDataSetRow += numRows;
  };

  DataPool dataPool(totalRowLen,poolSize,chunkHeight,writeRow);

  std::function<void(const vector<FastBoard>&,int,int,int,const string&,int,const string&,const vector<Move>&,int,Hash128)> f =
    [&dataPool,&rand,keepProb,minRank,minOppRank,maxHandicap,target,&excludeUsers,fancyConditions,fancyPosKeepFactor,alwaysHistory,&posHashes,&total,&used](
      const vector<FastBoard>& recentBoards, int source, int rank, int oppRank, const string& user, int handicap, const string& date,
      const vector<Move>& moves, int moveIdx, Hash128 sgfHash
    ) {
    maybeUseRow(
      recentBoards,source,rank,oppRank,user,handicap,date,moves,moveIdx,sgfHash,
      dataPool,rand,keepProb,minRank,minOppRank,maxHandicap,target,
      alwaysHistory,
      excludeUsers,fancyConditions,fancyPosKeepFactor,
      posHashes,total,used
    );
  };

  iterSgfsMoves(
    sgfs,
    shardSeed,numShards,
    used.count,curDataSetRow,
    f
  );

  cout << "Emptying pool" << endl;
  dataPool.finishAndWritePool(rand);
}



int main(int argc, const char* argv[]) {
  assert(sizeof(size_t) == 8);
  FastBoard::initHash();

//   string s =
// ". . . . . O O O O . . . . . . O O X ."
// ". . . . X X O X O O . . . . . O X . X"
// ". . . X X O O X X . O O . X . O X X ."
// ". . X X . X X . . O . X O . . O X . X"
// ". X O O O X . X O . . O . O . O X . O"
// "X X X O O X . X O . X X O X O O X . O"
// ". X O O O X . X . O . . X . O X X X ."
// "X O O . O X O X X O . X X O . O . . ."
// ". X X O . O X X O X X . . X . O X X X"
// "X . X O O O O O O X . . . . . O O O ."
// ". X O O O X . O X X . . . . X X O . ."
// ". X O X . X . O O X X . X X . X O . ."
// "X . X . X . . O X X O O O O X X . . ."
// "X X O X X X . O . X X O . . O . X X ."
// "X O O X . O O . X . . X O . O O X O ."
// "O . O X O . O . X O . X O . O * O O ."
// ". O O X . O O X X X X O O O X O O . ."
// ". O X X X O O X O O O O O . . . . . ."
// ". O . . . O X X . . . . . . . . . . ."
// ;

//   FastBoard testBoard(19);

//   int next = -1;
//   for(int y = 0; y<19; y++) {
//     for(int x = 0; x < 19; x++) {
//       next += 1;
//       while(s[next] != '.' && s[next] != '*' && s[next] != 'O' && s[next] != 'X')
//         next += 1;
//       if(s[next] == 'O')
//         testBoard.setStone(Location::getLoc(x,y,19),P_WHITE);
//       if(s[next] == 'X')
//         testBoard.setStone(Location::getLoc(x,y,19),P_BLACK);
//     }
//   }

//   cout << testBoard << endl;
//   FastBoard testCopy(testBoard);
//   vector<Loc> buf;
//   cout << testCopy << endl;
//   cout << testCopy.searchIsLadderCaptured(Location::getLoc(11,4,19),true,buf) << endl;
//   cout << testCopy.searchIsLadderCaptured(Location::getLoc(6,7,19),true,buf) << endl;
//   return 0;

  cout << "Command: ";
  for(int i = 0; i<argc; i++)
    cout << argv[i] << " ";
  cout << endl;

  vector<string> gamesDirs;
  string outputFile;
  string onlyFilesFile;
  string excludeFilesFile;
  vector<string> excludeHashesFiles;
  size_t poolSize;
  int trainShards;
  double valGameProb;
  double keepTrainProb;
  double keepValProb;
  int minRank;
  int minOppRank;
  int maxHandicap;
  int target;
  bool alwaysHistory;
  bool fancyConditions;
  double fancyGameKeepFactor;
  double fancyPosKeepFactor;
  vector<string> excludeUsersFiles;

  try {
    TCLAP::CmdLine cmd("Sgf->HDF5 data writer", ' ', "1.0",true);
    TCLAP::MultiArg<string> gamesdirArg("","gamesdir","Directory of sgf files",true,"DIR");
    TCLAP::ValueArg<string> outputArg("","output","H5 file to write",true,string(),"FILE");
    TCLAP::ValueArg<string> onlyFilesArg("","only-files","Specify a list of files to filter to, one per line in a txt file",false,string(),"FILEOFFILES");
    TCLAP::ValueArg<string> excludeFilesArg("","exclude-files","Specify a list of files to filter out, one per line in a txt file",false,string(),"FILEOFFILES");
    TCLAP::MultiArg<string> excludeHashesArg("","exclude-hashes","Specify a list of hashes to filter out, one per line in a txt file",false,"FILEOF(HASH,HASH)");
    TCLAP::ValueArg<size_t> poolSizeArg("","pool-size","Pool size for shuffling rows",true,(size_t)0,"SIZE");
    TCLAP::ValueArg<int>    trainShardsArg("","train-shards","Make this many passes processing 1/N of the data each time",true,0,"INT");
    TCLAP::ValueArg<double> valGameProbArg("","val-game-prob","Probability of using a game for validation instead of train",true,0.0,"PROB");
    TCLAP::ValueArg<double> keepTrainProbArg("","keep-train-prob","Probability per-move of keeping a move in the train set",false,1.0,"PROB");
    TCLAP::ValueArg<double> keepValProbArg("","keep-val-prob","Probability per-move of keeping a move in the val set",false,1.0,"PROB");
    TCLAP::ValueArg<int>    minRankArg("","min-rank","Min rank to use a player's move",false,-10000,"RANK");
    TCLAP::ValueArg<int>    minOppRankArg("","min-opp-rank","Min rank of opp to use a player's move",false,-10000,"RANK");
    TCLAP::ValueArg<int>    maxHandicapArg("","max-handicap","Max handicap of game to use a player's move",false,9,"HCAP");
    TCLAP::ValueArg<string> targetArg("","target","What should be predicted? Currently only option is nextmove",false,string("nextmove"),"TARGET");
    TCLAP::SwitchArg        alwaysHistoryArg("","always-history","Always include history",false);
    TCLAP::SwitchArg        fancyConditionsArg("","fancy-conditions","Fancy filtering for rank balancing",false);
    TCLAP::ValueArg<double> fancyGameKeepFactorArg("","fancy-game-keep-factor","Multiply fancy game keep prob by this",false,1.0,"PROB");
    TCLAP::ValueArg<double> fancyPosKeepFactorArg("","fancy-pos-keep-factor","Multiply fancy pos keep prob by this",false,1.0,"PROB");
    TCLAP::MultiArg<string> excludeUsersArg("","exclude-users","File of users to exclude, one per line",false,"FILE");
    cmd.add(gamesdirArg);
    cmd.add(outputArg);
    cmd.add(onlyFilesArg);
    cmd.add(excludeFilesArg);
    cmd.add(excludeHashesArg);
    cmd.add(poolSizeArg);
    cmd.add(trainShardsArg);
    cmd.add(valGameProbArg);
    cmd.add(keepTrainProbArg);
    cmd.add(keepValProbArg);
    cmd.add(minRankArg);
    cmd.add(minOppRankArg);
    cmd.add(maxHandicapArg);
    cmd.add(targetArg);
    cmd.add(alwaysHistoryArg);
    cmd.add(fancyConditionsArg);
    cmd.add(fancyGameKeepFactorArg);
    cmd.add(fancyPosKeepFactorArg);
    cmd.add(excludeUsersArg);
    cmd.parse(argc,argv);
    gamesDirs = gamesdirArg.getValue();
    outputFile = outputArg.getValue();
    onlyFilesFile = onlyFilesArg.getValue();
    excludeFilesFile = excludeFilesArg.getValue();
    excludeHashesFiles = excludeHashesArg.getValue();
    poolSize = poolSizeArg.getValue();
    trainShards = trainShardsArg.getValue();
    valGameProb = valGameProbArg.getValue();
    keepTrainProb = keepTrainProbArg.getValue();
    keepValProb = keepValProbArg.getValue();
    minRank = minRankArg.getValue();
    minOppRank = minOppRankArg.getValue();
    maxHandicap = maxHandicapArg.getValue();
    alwaysHistory = alwaysHistoryArg.getValue();
    fancyConditions = fancyConditionsArg.getValue();
    fancyGameKeepFactor = fancyGameKeepFactorArg.getValue();
    fancyPosKeepFactor = fancyPosKeepFactorArg.getValue();
    excludeUsersFiles = excludeUsersArg.getValue();

    if(targetArg.getValue() == "nextmove")
      target = TARGET_NEXT_MOVE_AND_LADDER;
    else
      throw IOError("Must specify target nextmove or... actually no other options right now");
  }
  catch (TCLAP::ArgException &e) {
    cerr << "Error: " << e.error() << " for argument " << e.argId() << std::endl;
    return 1;
  }

  set<string> excludeUsers;
  for(size_t i = 0; i < excludeUsersFiles.size(); i++) {
    const string& file = excludeUsersFiles[i];
    vector<string> users = Global::readFileLines(file,'\n');
    for(size_t j = 0; j < users.size(); j++) {
      const string& user = Global::trim(Global::stripComments(users[j]));
      excludeUsers.insert(user);
    }
  }

  bool onlyFilesProvided = false;
  set<string> onlyFiles;
  if(onlyFilesFile.length() > 0) {
    onlyFilesProvided = true;
    vector<string> files = Global::readFileLines(onlyFilesFile,'\n');
    for(size_t j = 0; j < files.size(); j++) {
      const string& file = Global::trim(Global::stripComments(files[j]));
      onlyFiles.insert(file);
    }
  }

  bool excludeFilesProvided = false;
  set<string> excludeFiles;
  if(excludeFilesFile.length() > 0) {
    excludeFilesProvided = true;
    vector<string> files = Global::readFileLines(excludeFilesFile,'\n');
    for(size_t j = 0; j < files.size(); j++) {
      const string& file = Global::trim(Global::stripComments(files[j]));
      excludeFiles.insert(file);
    }
  }

  bool excludeHashesProvided = false;
  set<Hash128> excludeHashes;
  for(int i = 0; i<excludeHashesFiles.size(); i++) {
    const string& excludeHashesFile = excludeHashesFiles[i];
    excludeHashesProvided = true;
    vector<string> hashes = Global::readFileLines(excludeHashesFile,'\n');
    for(size_t j = 0; j < hashes.size(); j++) {
      const string& hash128 = Global::trim(Global::stripComments(hashes[j]));
      if(hash128.length() <= 0)
        continue;
      if(hash128.length() != 32)
        throw IOError("Could not parse hashpair in exclude hashes file: " + hash128);

      uint64_t hash0 = parseHex64(hash128.substr(0,16));
      uint64_t hash1 = parseHex64(hash128.substr(16,16));
      excludeHashes.insert(Hash128(hash0,hash1));
    }
  }

  //Print some stats-----------------------------------------------------------------
  cout << "maxBoardSize " << maxBoardSize << endl;
  cout << "numFeatures " << numFeatures << endl;
  cout << "inputLen " << inputLen << endl;
  cout << "targetLen " << targetLen << endl;
  cout << "ladderTargetLen " << ladderTargetLen << endl;
  cout << "targetWeightsLen " << targetWeightsLen << endl;
  cout << "rankLen " << rankLen << endl;
  cout << "totalRowLen " << totalRowLen << endl;
  cout << "chunkHeight " << chunkHeight << endl;
  cout << "deflateLevel " << deflateLevel << endl;
  cout << "poolSize " << poolSize << endl;
  cout << "trainShards " << trainShards << endl;
  cout << "valGameProb " << valGameProb << endl;
  cout << "keepTrainProb " << keepTrainProb << endl;
  cout << "keepValProb " << keepValProb << endl;
  cout << "minRank " << minRank << endl;
  cout << "minOppRank " << minOppRank << endl;
  cout << "maxHandicap " << maxHandicap << endl;
  cout << "target " << target << endl;
  cout << "alwaysHistory " << alwaysHistory << endl;
  cout << "fancyConditions " << fancyConditions << endl;
  cout << "fancyGameKeepFactor " << fancyGameKeepFactor << endl;
  cout << "fancyPosKeepFactor " << fancyPosKeepFactor << endl;

  cout << endl;
  cout << "Excluding users:" << endl;
  for(const string& s: excludeUsers) {
    cout << s << endl;
  }
  if(onlyFilesProvided) {
    cout << "Filtering to only " << onlyFiles.size() << " files (or whatever subset is present)";
  }
  if(excludeFilesProvided) {
    cout << "Filtering to exclude " << excludeFiles.size() << " files (if present)";
  }
  cout << endl;

  //Collect SGF files-----------------------------------------------------------------
  const string suffix = ".sgf";
  auto filter = [&suffix](const string& name) {
    return Global::isSuffix(name,suffix);
  };

  vector<string> files;
  for(int i = 0; i<gamesDirs.size(); i++)
    Global::collectFiles(gamesDirs[i], filter, files);
  cout << "Found " << files.size() << " sgf files!" << endl;

  cout << "Opening h5 file..." << endl;
  H5File* h5File = new H5File(H5std_string(outputFile), H5F_ACC_TRUNC);
  hsize_t maxDims[h5Dimension] = {H5S_UNLIMITED, totalRowLen};
  hsize_t chunkDims[h5Dimension] = {chunkHeight, totalRowLen};
  hsize_t initFileDims[h5Dimension] = {0, totalRowLen};

  DSetCreatPropList dataSetProps;
  dataSetProps.setChunk(h5Dimension,chunkDims);
  dataSetProps.setDeflate(deflateLevel);

  //Load, filter and randomize SGFS----------------------------------------------------------
  Rand rand;

  //Filter if filtering by file
  if(onlyFilesProvided) {
    int kept = 0;
    for(int i = 0; i<files.size(); i++) {
      if(contains(onlyFiles,files[i])) {
        if(i != kept)
          std::swap(files[i],files[kept]);
        kept++;
      }
    }
    files.resize(kept);
    cout << "Kept " << files.size() << " sgf files after filtering by onlyFiles!" << endl;
  }
  if(excludeFilesProvided) {
    int kept = 0;
    for(int i = 0; i<files.size(); i++) {
      if(!contains(excludeFiles,files[i])) {
        if(i != kept)
          std::swap(files[i],files[kept]);
        kept++;
      }
    }
    files.resize(kept);
    cout << "Kept " << files.size() << " sgf files after filtering by excludeFiles!" << endl;
  }

  //Filter if doing fancy stuff
  if(fancyConditions) {
    int kept = 0;
    for(int i = 0; i<files.size(); i++) {
      int source = parseSource(files[i]);
      double prob = sourceGameFancyProb[source] * fancyGameKeepFactor;
      bool keep = prob >= 1.0 || rand.nextDouble() < prob;
      if(keep) {
        if(i != kept)
          std::swap(files[i],files[kept]);
        kept++;
      }
    }
    files.resize(kept);
    cout << "Kept " << files.size() << " sgf files after filtering by fancy source!" << endl;
  }

  cout << "Loading SGFS..." << endl;
  vector<CompactSgf*> sgfs = CompactSgf::loadFiles(files);

  // for(int i = 0; i<sgfs.size(); i++) {
  //   if(sgfs[i]->hash[0] == 0x1a94b16410ae6be0ULL ||
  //      sgfs[i]->hash[0] == 0xcd09562bc06cd9bbULL ||
  //      sgfs[i]->hash[0] == 0x2233404f2631382ULL ||
  //      sgfs[i]->hash[0] == 0x76c1a947cde45a9eULL ||
  //      sgfs[i]->hash[0] == 0xe471f1dbcf146ff1ULL ||
  //      sgfs[i]->hash[0] == 0x810d5d87c8dce0cbULL ||
  //      sgfs[i]->hash[0] == 0xa7fcf6c55ee16e30ULL ||
  //      sgfs[i]->hash[0] == 0x594b712635a2692dULL ||
  //      sgfs[i]->hash[0] == 0xa8cd003f20241f33ULL ||
  //      sgfs[i]->hash[0] == 0xe069b39c42b26fe3ULL ||
  //      sgfs[i]->hash[0] == 0xf4b683e270916e8eULL ||
  //      sgfs[i]->hash[0] == 0x9510377733620c6dULL ||
  //      sgfs[i]->hash[0] == 0xe8c22c39775764d2ULL)
  //     cout << sgfs[i]->fileName << endl;
  // }

  if(excludeHashesProvided) {
    int kept = 0;
    for(int i = 0; i<sgfs.size(); i++) {
      if(!contains(excludeHashes,sgfs[i]->hash)) {
        if(i != kept)
          std::swap(sgfs[i],sgfs[kept]);
        kept++;
      }
      else {
        delete sgfs[i];
      }
    }
    sgfs.resize(kept);
    cout << "Kept " << sgfs.size() << " sgf files after filtering by excludeHashes!" << endl;
  }

  //Shuffle sgfs
  cout << "Shuffling SGFS..." << endl;
  for(int i = 1; i<sgfs.size(); i++) {
    int r = rand.nextUInt(i+1);
    CompactSgf* tmp = sgfs[i];
    sgfs[i] = sgfs[r];
    sgfs[r] = tmp;
  }

  //Split into train and val
  vector<CompactSgf*> trainSgfs;
  vector<CompactSgf*> valSgfs;
  for(int i = 0; i<sgfs.size(); i++) {
    if(rand.nextDouble() < valGameProb)
      valSgfs.push_back(sgfs[i]);
    else
      trainSgfs.push_back(sgfs[i]);
  }
  //Clear sgfs via swap with enpty factor
  vector<CompactSgf*>().swap(sgfs);

  //Process SGFS to make rows----------------------------------------------------------
  uint64_t trainShardSeed = rand.nextUInt64();
  uint64_t valShardSeed = rand.nextUInt64();

  cout << "Generating TRAINING set..." << endl;
  H5std_string trainSetName("train");
  DataSet* trainDataSet = new DataSet(h5File->createDataSet(trainSetName, PredType::IEEE_F32LE, DataSpace(h5Dimension,initFileDims,maxDims), dataSetProps));
  set<Hash> trainPosHashes;
  Stats trainTotalStats;
  Stats trainUsedStats;
  processSgfs(
    trainSgfs,trainDataSet,
    poolSize,
    trainShardSeed, trainShards,
    rand, keepTrainProb,
    minRank, minOppRank, maxHandicap, target,
    alwaysHistory,
    excludeUsers, fancyConditions, fancyPosKeepFactor,
    trainPosHashes, trainTotalStats, trainUsedStats
  );
  delete trainDataSet;

  cout << "Generating VALIDATION set..." << endl;
  H5std_string valSetName("val");
  DataSet* valDataSet = new DataSet(h5File->createDataSet(valSetName, PredType::IEEE_F32LE, DataSpace(h5Dimension,initFileDims,maxDims), dataSetProps));
  set<Hash> valPosHashes;
  Stats valTotalStats;
  Stats valUsedStats;
  processSgfs(
    valSgfs,valDataSet,
    poolSize,
    valShardSeed, trainShards,
    rand, keepValProb,
    minRank, minOppRank, maxHandicap, target,
    alwaysHistory,
    excludeUsers, fancyConditions, fancyPosKeepFactor,
    valPosHashes, valTotalStats, valUsedStats
  );
  delete valDataSet;

  //Close the h5 file
  delete h5File;

  //Record names of all the sgf files
  ofstream trainNames;
  trainNames.open(outputFile + ".train.txt");
  for(int i = 0; i<trainSgfs.size(); i++) {
    trainNames << trainSgfs[i]->fileName << "\n";
  }
  trainNames.close();
  ofstream valNames;
  valNames.open(outputFile + ".val.txt");
  for(int i = 0; i<valSgfs.size(); i++) {
    valNames << valSgfs[i]->fileName << "\n";
  }
  valNames.close();

  cout << "Done" << endl;

  cout << "TRAIN TOTAL------------------------------------" << endl;
  trainTotalStats.print();
  cout << "TRAIN USED------------------------------------" << endl;
  cout << trainPosHashes.size() << " unique pos hashes used" << endl;
  trainUsedStats.print();

  cout << "VAL TOTAL------------------------------------" << endl;
  valTotalStats.print();
  cout << "VAL USED------------------------------------" << endl;
  cout << valPosHashes.size() << " unique pos hashes used" << endl;
  valUsedStats.print();

  //Cleanup----------------------------------------------------------------------------
  for(int i = 0; i<sgfs.size(); i++) {
    delete sgfs[i];
  }
  cout << "Everything cleaned up" << endl;

  return 0;
}
