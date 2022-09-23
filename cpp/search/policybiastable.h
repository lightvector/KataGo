#ifndef SEARCH_POLICYBIASTABLE_H
#define SEARCH_POLICYBIASTABLE_H

#include "../core/global.h"
#include "../core/hash.h"
#include "../core/multithread.h"
#include "../game/boardhistory.h"
#include "../search/mutexpool.h"

struct Search;
struct PolicyBiasTable;

struct PolicyBiasEntry {
  std::atomic<double> average;
  double weightSum;
  mutable std::atomic_flag entryLock = ATOMIC_FLAG_INIT;

  inline PolicyBiasEntry(): average(0.0), weightSum(0.0) {
  }
};

struct PolicyBiasHandle {
  double lastSum;
  double lastWeight;
  int lastPos;

  std::vector<std::shared_ptr<PolicyBiasEntry>> entries;

  //Used for retrieving search parameters
  const PolicyBiasTable* table;

  PolicyBiasHandle();
  ~PolicyBiasHandle();
  void clear();

  // Update this node's contribution to the sum among all nodes with this entry,
  void updateValue(double newSumThisNode, double newWeightThisNode, int pos);

  PolicyBiasHandle& operator=(const PolicyBiasHandle&) = delete;
  PolicyBiasHandle(const PolicyBiasHandle&) = delete;

private:
  void revertUpdates(double freeProp);
};



struct PolicyBiasTable {
  std::vector<std::map<Hash128,std::shared_ptr<PolicyBiasEntry>>> entries;
  MutexPool* mutexPool;
  const Search* const search;
  bool freePropEnabled;

  int expectedNNXLen;
  int expectedNNYLen;

  PolicyBiasTable(const Search* search);
  ~PolicyBiasTable();

  // Enable or disable the logic of PolicyBiasHandle where it subtracts out its contribution
  // to the bias when the handle is freed.
  void setFreePropEnabled();
  void setFreePropDisabled();

  // ASSUMES there is no concurrent multithreading of this table or any of its entries,
  // and that all past mutations on this table or any of its entries are now visible to this thread.
  void clearUnusedSynchronous();

  // Set the nnXLen and nnYLen expected by this table for the policy bias entry vectors returned
  // Mostly for safety and error checking.
  void setNNLenAndAssertEmptySynchronous(int nnXLen, int nnYLen);


  void get(
    PolicyBiasHandle& buf,
    Player pla,
    Loc prevMoveLoc,
    int nnXLen,
    int nnYLen,
    const Board& board,
    const BoardHistory& hist
  );
};

#endif


