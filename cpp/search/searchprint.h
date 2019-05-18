#ifndef SEARCH_SEARCHPRINT_H_
#define SEARCH_SEARCHPRINT_H_

#include "../game/board.h"

struct PrintTreeOptions {
  PrintTreeOptions();

  PrintTreeOptions maxDepth(int);
  PrintTreeOptions maxChildrenToShow(int);
  PrintTreeOptions minVisitsToShow(int64_t);
  PrintTreeOptions minVisitsToExpand(int64_t);
  PrintTreeOptions minVisitsPropToShow(double);
  PrintTreeOptions minVisitsPropToExpand(double);
  PrintTreeOptions printSqs(bool);
  PrintTreeOptions onlyBranch(const Board& board, const std::string& moves);

  int maxDepth_;
  int maxChildrenToShow_;
  int64_t minVisitsToShow_;
  int64_t minVisitsToExpand_;
  double minVisitsPropToShow_;
  double minVisitsPropToExpand_;
  int maxPVDepth_;
  bool printRawNN_;
  bool printSqs_;
  std::vector<Loc> branch_;
};

inline PrintTreeOptions::PrintTreeOptions()
  :maxDepth_(1),
   maxChildrenToShow_(100000),
   minVisitsToShow_(1),
   minVisitsToExpand_(1),
   minVisitsPropToShow_(0.0),
   minVisitsPropToExpand_(0.0),
   maxPVDepth_(7),
   printRawNN_(false),
   printSqs_(false),
   branch_()
{}

inline PrintTreeOptions PrintTreeOptions::maxDepth(int d) { PrintTreeOptions other = *this; other.maxDepth_ = d; return other;}
inline PrintTreeOptions PrintTreeOptions::maxChildrenToShow(int c) { PrintTreeOptions other = *this; other.maxChildrenToShow_ = c; return other;}
inline PrintTreeOptions PrintTreeOptions::minVisitsToShow(int64_t v) { PrintTreeOptions other = *this; other.minVisitsToShow_ = v; return other;}
inline PrintTreeOptions PrintTreeOptions::minVisitsToExpand(int64_t v) { PrintTreeOptions other = *this; other.minVisitsToExpand_ = v; return other;}
inline PrintTreeOptions PrintTreeOptions::minVisitsPropToShow(double p) { PrintTreeOptions other = *this; other.minVisitsPropToShow_ = p; return other;}
inline PrintTreeOptions PrintTreeOptions::minVisitsPropToExpand(double p) { PrintTreeOptions other = *this; other.minVisitsPropToExpand_ = p; return other;}
inline PrintTreeOptions PrintTreeOptions::printSqs(bool b) { PrintTreeOptions other = *this; other.printSqs_ = b; return other;}
inline PrintTreeOptions PrintTreeOptions::onlyBranch(const Board& board, const std::string& moves) {
  PrintTreeOptions other = *this; other.branch_ = Location::parseSequence(moves,board);
  return other;
}

#endif  // SEARCH_SEARCHPRINT_H_
