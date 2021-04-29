#ifndef SEARCH_LOCALPATTERN_H
#define SEARCH_LOCALPATTERN_H

#include "../core/global.h"
#include "../core/hash.h"
#include "../core/rand.h"
#include "../game/board.h"

struct LocalPatternHasher {
  int xSize;
  int ySize;
  std::vector<Hash128> zobristLocalPattern;
  std::vector<Hash128> zobristPla;
  std::vector<Hash128> zobristAtari;

  LocalPatternHasher();
  ~LocalPatternHasher();

  void init(int xSize, int ySize, Rand& rand);

  Hash128 getHash(const Board& board, Loc loc, Player pla) const;
};

#endif //SEARCH_LOCALPATTERN_H
