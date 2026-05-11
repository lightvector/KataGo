#ifndef SEARCH_SEARCHPRINT_H_
#define SEARCH_SEARCHPRINT_H_

#include "../game/board.h"

struct PrintTreeOptions {
  PrintTreeOptions();

  PrintTreeOptions maxDepth(int) const;
  PrintTreeOptions maxChildrenToShow(int) const;
  PrintTreeOptions minVisitsToShow(int64_t) const;
  PrintTreeOptions minVisitsToExpand(int64_t) const;
  PrintTreeOptions minVisitsPropToShow(double) const;
  PrintTreeOptions minVisitsPropToExpand(double) const;
  PrintTreeOptions printSqs(bool) const;
  PrintTreeOptions printAvgShorttermError(bool) const;
  PrintTreeOptions onlyBranch(const Board& board, const std::string& moves) const;
  PrintTreeOptions alsoBranch(const Board& board, const std::string& moves) const;

  int maxDepth_;
  int maxChildrenToShow_;
  int64_t minVisitsToShow_;
  int64_t minVisitsToExpand_;
  double minVisitsPropToShow_;
  double minVisitsPropToExpand_;
  int maxPVDepth_;
  bool printRawNN_;
  bool printSqs_;
  bool printAvgShorttermError_;
  std::vector<Loc> branch_;
  bool alsoBranch_;
};

inline PrintTreeOptions::PrintTreeOptions()
  :maxDepth_(1),
   maxChildrenToShow_(100000),
   minVisitsToShow_(0),
   minVisitsToExpand_(1),
   minVisitsPropToShow_(0.0),
   minVisitsPropToExpand_(0.0),
   maxPVDepth_(7),
   printRawNN_(false),
   printSqs_(false),
   printAvgShorttermError_(false),
   branch_(),
   alsoBranch_(false)
{}

inline PrintTreeOptions PrintTreeOptions::maxDepth(int d) const { PrintTreeOptions other = *this; other.maxDepth_ = d; return other;}
inline PrintTreeOptions PrintTreeOptions::maxChildrenToShow(int c) const { PrintTreeOptions other = *this; other.maxChildrenToShow_ = c; return other;}
inline PrintTreeOptions PrintTreeOptions::minVisitsToShow(int64_t v) const { PrintTreeOptions other = *this; other.minVisitsToShow_ = v; return other;}
inline PrintTreeOptions PrintTreeOptions::minVisitsToExpand(int64_t v) const { PrintTreeOptions other = *this; other.minVisitsToExpand_ = v; return other;}
inline PrintTreeOptions PrintTreeOptions::minVisitsPropToShow(double p) const { PrintTreeOptions other = *this; other.minVisitsPropToShow_ = p; return other;}
inline PrintTreeOptions PrintTreeOptions::minVisitsPropToExpand(double p) const { PrintTreeOptions other = *this; other.minVisitsPropToExpand_ = p; return other;}
inline PrintTreeOptions PrintTreeOptions::printSqs(bool b) const { PrintTreeOptions other = *this; other.printSqs_ = b; return other;}
inline PrintTreeOptions PrintTreeOptions::printAvgShorttermError(bool b) const { PrintTreeOptions other = *this; other.printAvgShorttermError_ = b; return other;}
inline PrintTreeOptions PrintTreeOptions::onlyBranch(const Board& board, const std::string& moves) const {
  PrintTreeOptions other = *this; other.branch_ = Location::parseSequence(moves,board);
  return other;
}
inline PrintTreeOptions PrintTreeOptions::alsoBranch(const Board& board, const std::string& moves) const {
  PrintTreeOptions other = *this; other.branch_ = Location::parseSequence(moves,board);
  other.alsoBranch_ = true;
  return other;
}

#endif  // SEARCH_SEARCHPRINT_H_
