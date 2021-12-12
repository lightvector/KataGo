#ifndef GAME_GRAPHHASH_H_
#define GAME_GRAPHHASH_H_

#include "../game/boardhistory.h"

namespace GraphHash {
  //Hash taking into account all state relevant for move legality, the rules, and some immediate other info like effect of passing.
  //Does NOT take into account more complex history beyond immediate ko and superko bans.
  Hash128 getStateHash(const BoardHistory& hist, Player nextPlayer, double drawEquivalentWinsForWhite);

  //Call this AFTER making a move, to update a hash suitable for superko-safe tranpositions, given the previous graph hash.
  //Will guard against cycles up to repBound in size and possibly some slightly larger cycles.
  Hash128 getGraphHash(Hash128 prevGraphHash, const BoardHistory& hist, Player nextPlayer, int repBound, double drawEquivalentWinsForWhite);

  //Compute graph hash from scratch by replaying the whole history.
  Hash128 getGraphHashFromScratch(const BoardHistory& hist, Player nextPlayer, int repBound, double drawEquivalentWinsForWhite);
}

#endif
