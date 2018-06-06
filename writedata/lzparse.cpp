
#include "core/global.h"
#include "fastboard.h"
#include "sgf.h"
#include "lzparse.h"
#include <zstr/src/zstr.hpp>

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

static void setStone(FastBoard& board, int tensorPos, Player pla) {
  int x = tensorPos % 19;
  int y = tensorPos / 19;
  board.setStone(Location::getLoc(x,y,19),pla);
}

static void decodeStones(const string& line, FastBoard& board, Player pla)
{
  assert(line.length() == 91);
  //The first 90 characters are a hex-encoding of the first 360 points
  for(int i = 0; i<90; i++) {
    char c = line[i];
    int d;
    if(c >= '0' && c <= '9')
      d = c-'0';
    else if (c >= 'a' && c <= 'f')
      d = c-'a'+10;
    else
      assert(false);

    if(d & 0x1) setStone(board,i*4+3,pla);
    if(d & 0x2) setStone(board,i*4+2,pla);
    if(d & 0x4) setStone(board,i*4+1,pla);
    if(d & 0x8) setStone(board,i*4+0,pla);
  }
  //The last character is either 0 or 1
  {
    char c = line[90];
    if(c == '0') {}
    else if(c == '1')
      setStone(board,90*4,pla);
    else
      assert(false);
  }
}

static Move inferMove(FastBoard& board, FastBoard& prev, Player whoMoved) {
  //Search to find if there is a stone of the player who moved that is newly placed
  for(int y = 0; y<19; y++) {
    for(int x = 0; x<19; x++) {
      Loc loc = Location::getLoc(x,y,19);
      if(board.colors[loc] == whoMoved && prev.colors[loc] != whoMoved)
        return Move(loc,whoMoved);
    }
  }

  //Search to find if there as a suicide
  for(int y = 0; y<19; y++) {
    for(int x = 0; x<19; x++) {
      Loc loc = Location::getLoc(x,y,19);
      if(board.colors[loc] != whoMoved && prev.colors[loc] == whoMoved) {
        //Look around for a suicidal spot to play
        for(int i = 0; i < 4; i++)
        {
          Loc adj = loc + prev.adj_offsets[i];
          if(prev.colors[adj] == C_EMPTY && prev.isSuicide(adj,whoMoved))
            return Move(adj,whoMoved);
        }
      }
    }
  }

  //Otherwise it must have been a pass
  for(int y = 0; y<19; y++) {
    for(int x = 0; x<19; x++) {
      Loc loc = Location::getLoc(x,y,19);
      assert(board.colors[loc] == prev.colors[loc]);
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
  for(int i = 0; i<7; i++)
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

    //Generate the boards
    for(int i = 0; i<8; i++)
    {
      sample.boards[i] = emptyBoard;
      decodeStones(plaStones[i], sample.boards[i], pla);
      decodeStones(oppStones[i], sample.boards[i], opp);
    }

    //Infer the previous moves
    for(int i = 0; i<7; i++)
    {
      FastBoard& board = sample.boards[i];
      FastBoard& prev = sample.boards[i+1];
      Player whoMoved = (i % 2 == 0) ? opp : pla;
      Move move = inferMove(board,prev,whoMoved);
      sample.moves[7-i-1] = move;
    }

    //Next we have 362 floats indicating moves
    getLine(in,buf);
    {
      istringstream moveIn(buf);
      for(int i = 0; i<362; i++) {
        moveIn >> sample.probs[i];
        if(!moveIn)
          assert(false);
      }
      float dummy;
      assert(!(moveIn >> dummy));
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

