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

  //Returns the hash that would occur if symmetry were applied to both board and loc.
  //So basically, the only thing that changes is the zobrist indexing.
  Hash128 getHashWithSym(const Board& board, Loc loc, Player pla, int symmetry, bool flipColors) const;
};

#endif //SEARCH_LOCALPATTERN_H
