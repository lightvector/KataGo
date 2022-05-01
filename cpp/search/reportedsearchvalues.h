#ifndef SEARCH_REPORTEDSEARCHVALUES_H_
#define SEARCH_REPORTEDSEARCHVALUES_H_

#include "../core/global.h"

struct Search;

struct ReportedSearchValues {
  double winValue;
  double lossValue;
  double noResultValue;
  double staticScoreValue;
  double dynamicScoreValue;
  double expectedScore;
  double expectedScoreStdev;
  double lead;
  double winLossValue;
  double utility;
  double weight;
  int64_t visits;

  ReportedSearchValues();
  ReportedSearchValues(
    const Search& search,
    double winLossValueAvg,
    double noResultValueAvg,
    double scoreMeanAvg,
    double scoreMeanSqAvg,
    double leadAvg,
    double utilityAvg,
    double totalWeight,
    int64_t totalVisits
  );
  ~ReportedSearchValues();
};

#endif
