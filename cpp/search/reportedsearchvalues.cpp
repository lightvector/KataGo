#include "../search/reportedsearchvalues.h"

#include "../neuralnet/nninputs.h"
#include "../search/search.h"

ReportedSearchValues::ReportedSearchValues()
{}
ReportedSearchValues::~ReportedSearchValues()
{}
ReportedSearchValues::ReportedSearchValues(
  const Search& search,
  double winLossValueAvg,
  double noResultValueAvg,
  double scoreMeanAvg,
  double scoreMeanSqAvg,
  double leadAvg,
  double utilityAvg,
  double totalWeight,
  int64_t totalVisits
) {
  winLossValue = winLossValueAvg;
  noResultValue = noResultValueAvg;
  double scoreMean = scoreMeanAvg;
  double scoreMeanSq = scoreMeanSqAvg;
  double scoreStdev = ScoreValue::getScoreStdev(scoreMean,scoreMeanSq);
  staticScoreValue = ScoreValue::expectedWhiteScoreValue(scoreMean,scoreStdev,0.0,2.0,search.rootBoard);
  dynamicScoreValue = ScoreValue::expectedWhiteScoreValue(scoreMean,scoreStdev,search.recentScoreCenter,search.searchParams.dynamicScoreCenterScale,search.rootBoard);
  expectedScore = scoreMean;
  expectedScoreStdev = scoreStdev;
  lead = leadAvg;
  utility = utilityAvg;

  //Clamp. Due to tiny floating point errors, these could be outside range.
  if(winLossValue < -1.0) winLossValue = -1.0;
  if(winLossValue > 1.0) winLossValue = 1.0;
  if(noResultValue < 0.0) noResultValue = 0.0;
  if(noResultValue > 1.0-std::fabs(winLossValue)) noResultValue = 1.0-std::fabs(winLossValue);

  winValue = 0.5 * (winLossValue + (1.0 - noResultValue));
  lossValue = 0.5 * (-winLossValue + (1.0 - noResultValue));

  //Handle float imprecision
  if(winValue < 0.0) winValue = 0.0;
  if(winValue > 1.0) winValue = 1.0;
  if(lossValue < 0.0) lossValue = 0.0;
  if(lossValue > 1.0) lossValue = 1.0;

  weight = totalWeight;
  visits = totalVisits;
}
