#ifndef ANALYSISDATA_H
#define ANALYSISDATA_H

#include "../game/board.h"

struct AnalysisData {
  Loc move;
  int64_t numVisits;
  double utility;
  double resultUtility;
  double scoreUtility;
  double winLossValue;
  double policyPrior;
  double scoreMean;
  double scoreStdev;
  double ess;
  int order;
  vector<Loc> pv;

  AnalysisData();
  AnalysisData(const AnalysisData& other);
  AnalysisData(AnalysisData&& other);
  ~AnalysisData();

  AnalysisData& operator=(const AnalysisData& other);
  AnalysisData& operator=(AnalysisData&& other);

};

bool operator<(const AnalysisData& a0, const AnalysisData& a1);


#endif
