#ifndef ANALYSISDATA_H
#define ANALYSISDATA_H

#include "../game/board.h"

struct SearchNode;

struct AnalysisData {
  Loc move;
  int64_t numVisits;
  double playSelectionValue;
  double lcb;
  double radius;
  double utility;
  double resultUtility;
  double scoreUtility;
  double winLossValue;
  double policyPrior;
  double scoreMean;
  double scoreStdev;
  double ess;
  double weightFactor;
  int order;
  vector<Loc> pv;

  const SearchNode* node; //ONLY valid so long as search is not cleared

  AnalysisData();
  AnalysisData(const AnalysisData& other);
  AnalysisData(AnalysisData&& other);
  ~AnalysisData();

  AnalysisData& operator=(const AnalysisData& other);
  AnalysisData& operator=(AnalysisData&& other);

};

bool operator<(const AnalysisData& a0, const AnalysisData& a1);


#endif
