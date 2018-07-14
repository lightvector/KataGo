
#include <zstr/src/zstr.hpp>
#include <cstdlib>

#include "../dataio/lzparse.h"

LZSample::LZSample()
  :emptyBoard(19,19,true),plaStones(),oppStones(),sideStr(),policyStr(),resultStr()
{}

LZSample::~LZSample()
{}

static void getLine(istream& in, string& buf) {
  std::getline(in,buf);
  size_t len = buf.length();
  if(len > 0 && buf[len-1] == '\r') //just in case
    buf.pop_back();
}

static void setStone(Color* stones, int tensorPos, Color stone) {
  int x = tensorPos % 19;
  int y = tensorPos / 19;
  stones[Location::getLoc(x,y,19)] = stone;
}

static int parseHexChar(char c) {
  int d;
  if(c >= '0' && c <= '9')
    d = c-'0';
  else if (c >= 'a' && c <= 'f')
    d = c-'a'+10;
  else
    assert(false);
  return d;
}

static void decodeStones(const string& linePla, const string& lineOpp, Color* stones, Player pla)
{
  assert(linePla.length() == 91);
  assert(lineOpp.length() == 91);
  Player opp = getOpp(pla);
  //The first 90 characters are a hex-encoding of the first 360 points
  for(int i = 0; i<90; i++) {
    int dPla = parseHexChar(linePla[i]);
    int dOpp = parseHexChar(lineOpp[i]);

    Color stone;
    if(dPla & 0x8) stone = pla; else if(dOpp & 0x8) stone = opp; else stone = C_EMPTY;
    setStone(stones,i*4+0,stone);
    if(dPla & 0x4) stone = pla; else if(dOpp & 0x4) stone = opp; else stone = C_EMPTY;
    setStone(stones,i*4+1,stone);
    if(dPla & 0x2) stone = pla; else if(dOpp & 0x2) stone = opp; else stone = C_EMPTY;
    setStone(stones,i*4+2,stone);
    if(dPla & 0x1) stone = pla; else if(dOpp & 0x1) stone = opp; else stone = C_EMPTY;
    setStone(stones,i*4+3,stone);
  }
  //The last character is either 0 or 1
  {
    int cPla = linePla[90];
    int cOpp = lineOpp[90];

    assert(cPla == '1' || cPla == '0');
    assert(cOpp == '1' || cOpp == '0');

    Color stone;
    if(cPla == '1') stone = pla; else if(cOpp == '1') stone = opp; else stone = C_EMPTY;
    setStone(stones,90*4,stone);
  }
}

static Move inferMove(Color* board, Color* prev, Player whoMoved, Color stones[8][Board::MAX_ARR_SIZE], int stonesIdx, const short adj_offsets[8]) {
  //Search to find if there is a stone of the player who moved that is newly placed
  for(int y = 0; y<19; y++) {
    for(int x = 0; x<19; x++) {
      Loc loc = Location::getLoc(x,y,19);
      if(board[loc] == whoMoved && prev[loc] != whoMoved)
        return Move(loc,whoMoved);
    }
  }

  //Search to find if there as a suicide
  for(int y = 0; y<19; y++) {
    for(int x = 0; x<19; x++) {
      Loc loc = Location::getLoc(x,y,19);
      if(board[loc] != whoMoved && prev[loc] == whoMoved) {
        //Look around for an empty spot to play, next to where our stones vanished.
        for(int i = 0; i < 4; i++)
        {
          Loc adj = loc + adj_offsets[i];
          if(prev[adj] == C_EMPTY)
            return Move(adj,whoMoved);
        }
      }
    }
  }

  //Otherwise it must have been a pass
  for(int y = 0; y<19; y++) {
    for(int x = 0; x<19; x++) {
      Loc loc = Location::getLoc(x,y,19);
      if(board[loc] != prev[loc]) {
        for(int i = 0; i<8; i++) {
          for(int y2 = 0; y2<19; y2++) {
            for(int x2 = 0; x2<19; x2++) {
              Loc loc2 = Location::getLoc(x2,y2,19);
              assert(stones[i][loc2] >= 0 && stones[i][loc2] <= 2);
              cout << (stones[i][loc2] == 0 ? '.' : stones[i][loc2] == 1 ? 'X' : 'O');
            }
            cout << endl;
          }
          cout << endl;
        }
        cout << "Problem getting to index " << stonesIdx << " from " << (stonesIdx+1) << endl;
        throw IOError(string("Bad leela zero board consistency"));
      }
    }
  }
  return Move(Board::PASS_LOC,whoMoved);
}

void LZSample::iterSamples(
  const string& gzippedFile,
  std::function<void(const LZSample&,const string&,int)> f
) {
  LZSample sample;
  zstr::ifstream in(gzippedFile);

  int sampleCount = 0;
  while(in.good()) {
    //First 8 lines are pla stones, second 8 lines are opp stones
    //Most recent states are first
    for(int i = 0; i<8; i++) {
      getLine(in,sample.plaStones[i]);
    }
    for(int i = 0; i<8; i++) {
      getLine(in,sample.oppStones[i]);
    }

    if(!in.good())
      break;

    //Next line is which color, 0 = black, 1 = white
    getLine(in,sample.sideStr);
    assert(sample.sideStr.length() == 1);

    //Next we have 362 floats indicating moves
    getLine(in,sample.policyStr);

    //Next we have one line indicating whether the current player won or lost (+1 or -1).
    getLine(in,sample.resultStr);

    f(sample,gzippedFile,sampleCount);
    sampleCount++;
  }
}

void LZSample::parse(
  vector<Board>& boards,
  vector<Move>& moves,
  float policyTarget[362],
  Player& nextPlayer,
  Player& winner
) const {
  if(boards.size() != 8)
    boards.resize(8);
  if(moves.size() != 8)
    moves.resize(8);

  Player pla;
  if(sideStr[0] == '0')
    pla = P_BLACK;
  else if(sideStr[0] == '1')
    pla = P_WHITE;
  else
    assert(false);
  Player opp = getOpp(pla);

  //Parse all stones
  Color stones[8][Board::MAX_ARR_SIZE];
  for(int i = 0; i<8; i++)
    decodeStones(plaStones[i], oppStones[i], stones[i], pla);

  //Infer the moves based on the stones
  for(int i = 0; i<7; i++)
  {
    Color* board = stones[i];
    Color* prev = stones[i+1];
    Player whoMoved = (i % 2 == 0) ? opp : pla;
    Move move = inferMove(board,prev,whoMoved,stones,i,emptyBoard.adj_offsets);
    moves[7-i-1] = move;
  }

  //Generate the boards from the stones and the moves
  boards[7] = emptyBoard;
  for(int y = 0; y<19; y++) {
    for(int x = 0; x<19; x++) {
      Loc loc = Location::getLoc(x,y,19);
      boards[7].setStone(loc,stones[7][loc]);
    }
  }
  for(int i = 6; i>=0; i--)
  {
    boards[i] = boards[i+1];
    Move move = moves[7-i-1];
    bool suc = boards[i].playMove(move.loc,move.pla);
    if(!suc)
      throw IOError(string("Leela zero illegal implied move"));
  }

  {
    const char* start = policyStr.c_str();
    const char* s = start;
    char* end = NULL;

    float maxProb = 0;
    int maxI = 0;
    for(int i = 0; i<362; i++) {
      float prob = std::strtod(s,&end);
      policyTarget[i] = prob;
      s = end;
      if(prob > maxProb) {
        maxProb = prob;
        maxI = i;
      }
    }
    assert(end == start + policyStr.length());

    //Fill in the "next" move to be the argmax of the policyTarget
    if(maxI == 361)
      moves[7] = Move(Board::PASS_LOC,pla);
    else {
      int x = maxI % 19;
      int y = maxI / 19;
      moves[7] = Move(Location::getLoc(x,y,19),pla);
    }
  }

  if(resultStr == "1")
    winner = pla;
  else if(resultStr == "-1")
    winner = opp;
  else
    assert(false);

  nextPlayer = pla;
}

