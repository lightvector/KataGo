#ifndef SEARCH_ANALYSISDATA_H_
#define SEARCH_ANALYSISDATA_H_

#include "../game/board.h"
#include "../game/boardhistory.h"

struct SearchNode;

struct AnalysisData {
  //Utilities and scores should all be from white's perspective
  Loc move;
  int64_t numVisits;
  double playSelectionValue; //Similar units to visits, but might have LCB adjustments
  double lcb; //In units of utility
  double radius; //In units of utility
  double utility; //From -1 to 1 or -1.25 to -1.25 or other similar bounds, depending on score utility
  double resultUtility; //Utility from winloss result
  double scoreUtility; //Utility from score. Summing with resultUtility gives utility.
  double winLossValue; //From -1 to 1
  double policyPrior; //From 0 to 1
  double scoreMean; //In units of points
  double scoreStdev; //In units of points
  double ess; //Effective sample size taking into account weighting, could be somewhat smaller than visits
  double weightFactor; //Due to child value weighting
  int order; //Preference order of the moves, 0 is best
  std::vector<Loc> pv;

  const SearchNode* node; //ONLY valid so long as search is not cleared

  AnalysisData();
  AnalysisData(const AnalysisData& other);
  AnalysisData(AnalysisData&& other) noexcept;
  ~AnalysisData();

  AnalysisData& operator=(const AnalysisData& other);
  AnalysisData& operator=(AnalysisData&& other) noexcept;

  bool pvContainsPass() const;
  void writePV(std::ostream& out, const Board& board) const;
  void writePVUpToPhaseEnd(std::ostream& out, const Board& initialBoard, const BoardHistory& initialHist, Player initialPla) const;
};

bool operator<(const AnalysisData& a0, const AnalysisData& a1);


#endif  // SEARCH_ANALYSISDATA_H_
