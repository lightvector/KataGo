#ifndef LZPARSE_H_
#define LZPARSE_H_

#include "core/global.h"
#include "fastboard.h"
#include "sgf.h"

struct LZSample {
  vector<FastBoard> boards; //Index 0 is the most recent
  vector<Move> moves; //Index 0 is the least recent, index (len-1) is the last move made.
  float probs[362]; //Indexed by y*19+x as usual
  Player winner;

  LZSample();
  ~LZSample();

  static void iterSamples(
    const string& gzippedFile,
    std::function<void(const LZSample&)> f
  );

};

#endif
