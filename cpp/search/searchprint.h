#ifndef SEARCHPRINT_H
#define SEARCHPRINT_H
#include "../game/board.h"

struct PrintTreeOptions {
  PrintTreeOptions();

  PrintTreeOptions& maxDepth(int);
  PrintTreeOptions& maxChildrenToShow(int);
  PrintTreeOptions& minVisitsToShow(uint64_t);
  PrintTreeOptions& minVisitsToExpand(uint64_t);
  PrintTreeOptions& minVisitsPropToShow(double);
  PrintTreeOptions& minVisitsPropToExpand(double);
  PrintTreeOptions& onlyBranch(const Board& board, const string& moves);

  int maxDepth_;
  int maxChildrenToShow_;
  uint64_t minVisitsToShow_;
  uint64_t minVisitsToExpand_;
  double minVisitsPropToShow_;
  double minVisitsPropToExpand_;
  vector<Loc> branch_;
};

inline PrintTreeOptions::PrintTreeOptions()
  :maxDepth_(1),
   maxChildrenToShow_(100000),
   minVisitsToShow_(1),
   minVisitsToExpand_(1),
   minVisitsPropToShow_(0.0),
   minVisitsPropToExpand_(0.0),
   branch_()
{}

inline PrintTreeOptions& PrintTreeOptions::maxDepth(int d) { maxDepth_ = d; return *this;}
inline PrintTreeOptions& PrintTreeOptions::maxChildrenToShow(int c) { maxChildrenToShow_ = c; return *this;}
inline PrintTreeOptions& PrintTreeOptions::minVisitsToShow(uint64_t v) { minVisitsToShow_ = v; return *this;}
inline PrintTreeOptions& PrintTreeOptions::minVisitsToExpand(uint64_t v) { minVisitsToExpand_ = v; return *this;}
inline PrintTreeOptions& PrintTreeOptions::minVisitsPropToShow(double p) { minVisitsPropToShow_ = p; return *this;}
inline PrintTreeOptions& PrintTreeOptions::minVisitsPropToExpand(double p) { minVisitsPropToExpand_ = p; return *this;}
inline PrintTreeOptions& PrintTreeOptions::onlyBranch(const Board& board, const string& moves) {
  branch_ = Location::parseSequence(moves,board);
  return *this;
}

#endif
