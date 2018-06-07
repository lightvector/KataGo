
#include "core/global.h"
#include "fastboard.h"
#include "sgf.h"
#include "lzparse.h"
#include <zstr/src/zstr.hpp>
#include <cstdlib>

LZSample::LZSample()
  :boards(),moves(),probs(),next(C_BLACK),winner(C_EMPTY)
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
  Player opp = getEnemy(pla);
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

static Move inferMove(Color* board, Color* prev, Player whoMoved, short adj_offsets[8]) {
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
      assert(board[loc] == prev[loc]);
    }
  }
  return Move(FastBoard::PASS_LOC,whoMoved);
}

void LZSample::iterSamples(
  const string& gzippedFile,
  std::function<void(const LZSample&)> f
) {
  FastBoard emptyBoard;

  LZSample sample;
  for(int i = 0; i<8; i++)
    sample.boards.push_back(emptyBoard);
  for(int i = 0; i<8; i++)
    sample.moves.push_back(Move(FastBoard::NULL_LOC,C_EMPTY));

  zstr::ifstream in(gzippedFile);

  string plaStones[8];
  string oppStones[8];
  string buf;
  while(in.good()) {
    //First 8 lines are pla stones, second 8 lines are opp stones
    //Most recent states are first
    for(int i = 0; i<8; i++) {
      getLine(in,plaStones[i]);
    }
    for(int i = 0; i<8; i++) {
      getLine(in,oppStones[i]);
    }

    if(!in.good())
      break;

    //Next line is which color, 0 = black, 1 = white
    getLine(in,buf);
    assert(buf.length() == 1);
    Player pla;
    if(buf[0] == '0')
      pla = P_BLACK;
    else if(buf[0] == '1')
      pla = P_WHITE;
    else
      assert(false);
    Player opp = getEnemy(pla);

    //Parse all stones
    Color stones[8][FastBoard::MAX_ARR_SIZE];
    for(int i = 0; i<8; i++)
      decodeStones(plaStones[i], oppStones[i], stones[i], pla);

    //Infer the moves based on the stones
    for(int i = 0; i<7; i++)
    {
      Color* board = stones[i];
      Color* prev = stones[i+1];
      Player whoMoved = (i % 2 == 0) ? opp : pla;
      Move move = inferMove(board,prev,whoMoved,emptyBoard.adj_offsets);
      sample.moves[7-i-1] = move;
    }

    //Generate the boards from the stones and the moves
    sample.boards[7] = emptyBoard;
    for(int y = 0; y<19; y++) {
      for(int x = 0; x<19; x++) {
        Loc loc = Location::getLoc(x,y,19);
        sample.boards[7].setStone(loc,stones[7][loc]);
      }
    }
    for(int i = 6; i>=0; i--)
    {
      sample.boards[i] = sample.boards[i+1];
      Move move = sample.moves[7-i-1];
      bool suc = sample.boards[i].playMove(move.loc,move.pla);
      assert(suc);
    }

    //Next we have 362 floats indicating moves
    getLine(in,buf);
    {
      const char* start = buf.c_str();
      const char* s = start;
      char* end = NULL;

      float maxProb = 0;
      int maxI = 0;
      for(int i = 0; i<362; i++) {
        float prob = std::strtod(s,&end);
        sample.probs[i] = prob;
        s = end;
        if(prob > maxProb) {
          maxProb = prob;
          maxI = i;
        }
      }
      assert(end == start + buf.length());

      //Fill in the "next" move to be the argmax of the probs
      if(maxI == 361)
        sample.moves[7] = Move(FastBoard::PASS_LOC,pla);
      else {
        int x = maxI % 19;
        int y = maxI / 19;
        sample.moves[7] = Move(Location::getLoc(x,y,19),pla);
      }
    }

    //Next we have one line indicating whether the current player won or lost (+1 or -1).
    getLine(in,buf);
    if(buf == "1")
      sample.winner = pla;
    else if(buf == "-1")
      sample.winner = opp;
    else
      assert(false);

    sample.next = pla;

    f(sample);
  }
}

