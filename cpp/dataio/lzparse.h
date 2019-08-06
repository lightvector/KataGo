#ifndef DATAIO_LZPARSE_H_
#define DATAIO_LZPARSE_H_

#include "../core/global.h"
#include "../dataio/sgf.h"
#include "../game/board.h"

struct LZSample {
  Board emptyBoard;
  std::string plaStones[8];
  std::string oppStones[8];
  std::string sideStr;
  std::string policyStr;
  std::string resultStr;

  LZSample();
  ~LZSample();

  static void iterSamples(
    const std::string& gzippedFile,
    std::function<void(const LZSample&,const std::string&,int)> f
  );

  void parse(
    Board& board,
    BoardHistory& hist,
    std::vector<Move>& moves, //Index 0 is the least recent, index (len-2) is the last move made, index (len-1) is the next move.
    float probs[362], //Indexed by y*19+x as usual
    Player& next,
    Player& winner
  ) const;
};

#endif  // DATAIO_LZPARSE_H_
