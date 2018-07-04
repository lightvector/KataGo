#ifndef LZPARSE_H_
#define LZPARSE_H_

#include "core/global.h"
#include "game/board.h"
#include "sgf.h"

struct LZSample {
  Board emptyBoard;
  string plaStones[8];
  string oppStones[8];
  string sideStr;
  string policyStr;
  string resultStr;

  LZSample();
  ~LZSample();

  static void iterSamples(
    const string& gzippedFile,
    std::function<void(const LZSample&,const string&,int)> f
  );

  void parse(
    vector<Board>& boards, //Index 0 is the most recent
    vector<Move>& moves, //Index 0 is the least recent, index (len-2) is the last move made, index (len-1) is the next move.
    float probs[362], //Indexed by y*19+x as usual
    Player& next,
    Player& winner
  ) const;


};

#endif
