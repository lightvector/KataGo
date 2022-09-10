#ifndef SEARCH_SUBTREEVALUEBIASTABLE_H
#define SEARCH_SUBTREEVALUEBIASTABLE_H

#include "../core/global.h"
#include "../core/hash.h"
#include "../core/multithread.h"
#include "../game/board.h"
#include "../search/mutexpool.h"

struct Search;
struct SubtreeValueBiasTable;

struct SubtreeValueBiasEntry {
  double deltaUtilitySum = 0.0;
  double weightSum = 0.0;
  mutable std::atomic_flag entryLock = ATOMIC_FLAG_INIT;
};

struct SubtreeValueBiasHandle {
  //Protected under the entryLock in subtreeValueBiasTableEntry
  //Used only if subtreeValueBiasTableEntry is not nullptr.
  //During search, subtreeValueBiasTableEntry itself is set upon creation of the node and remains constant
  //thereafter, making it safe to access without synchronization.
  double lastSubtreeValueBiasDeltaSum;
  double lastSubtreeValueBiasWeight;
  std::shared_ptr<SubtreeValueBiasEntry> entry;

  //Used for retrieving search parameters
  const SubtreeValueBiasTable* table;

  SubtreeValueBiasHandle();
  ~SubtreeValueBiasHandle();
  void clear();

  // Get the average bias value averaged across all nodes with this entry
  double getValue() const;
  // Update this node's contribution to the bias among all nodes with this entry,
  // and get the new average bias value
  double updateValue(double newDeltaSumThisNode, double newWeightThisNode);

  SubtreeValueBiasHandle& operator=(const SubtreeValueBiasHandle&) = delete;
  SubtreeValueBiasHandle(const SubtreeValueBiasHandle&) = delete;
};

struct SubtreeValueBiasTable {
  std::vector<std::map<Hash128,std::shared_ptr<SubtreeValueBiasEntry>>> entries;
  MutexPool* mutexPool;
  const Search* const search;
  bool freePropEnabled;

  SubtreeValueBiasTable(int32_t numShards, const Search* search);
  ~SubtreeValueBiasTable();

  // Enable or disable the logic of SubtreeValueBiasHandle where it subtracts out its contribution
  // to the bias when the handle is freed.
  void setFreePropEnabled();
  void setFreePropDisabled();

  // ASSUMES there is no concurrent multithreading of this table or any of its entries,
  // and that all past mutations on this table or any of its entries are now visible to this thread.
  void clearUnusedSynchronous();

  // The board specified here is expected to be the board BEFORE the move is played.
  void get(
    SubtreeValueBiasHandle& buf,
    Player pla,
    Loc parentPrevMoveLoc,
    Loc prevMoveLoc,
    const Board& prevBoard
  );
};

#endif


