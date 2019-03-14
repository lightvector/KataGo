#include "../search/analysisdata.h"

AnalysisData::AnalysisData()
  :move(Board::NULL_LOC),
   numVisits(0),
   utility(0.0),
   winLossValue(0.0),
   policyPrior(0.0),
   scoreMean(0.0),
   scoreStdev(0.0),
   order(0),
   pv()
{}

AnalysisData::AnalysisData(const AnalysisData& other)
  :move(other.move),
   numVisits(other.numVisits),
   utility(other.utility),
   winLossValue(other.winLossValue),
   policyPrior(other.policyPrior),
   scoreMean(other.scoreMean),
   scoreStdev(other.scoreStdev),
   order(other.order),
   pv(other.pv)
{}

AnalysisData::AnalysisData(AnalysisData&& other)
  :move(other.move),
   numVisits(other.numVisits),
   utility(other.utility),
   winLossValue(other.winLossValue),
   policyPrior(other.policyPrior),
   scoreMean(other.scoreMean),
   scoreStdev(other.scoreStdev),
   order(other.order),
   pv(std::move(other.pv))
{}

AnalysisData::~AnalysisData()
{}

AnalysisData& AnalysisData::operator=(const AnalysisData& other) {
  if(this == &other)
    return *this;
  move = other.move;
  numVisits = other.numVisits;
  utility = other.utility;
  winLossValue = other.winLossValue;
  policyPrior = other.policyPrior;
  scoreMean = other.scoreMean;
  scoreStdev = other.scoreStdev;
  order = other.order;
  pv = other.pv;
  return *this;
}

AnalysisData& AnalysisData::operator=(AnalysisData&& other) {
  if(this == &other)
    return *this;
  move = other.move;
  numVisits = other.numVisits;
  utility = other.utility;
  winLossValue = other.winLossValue;
  policyPrior = other.policyPrior;
  scoreMean = other.scoreMean;
  scoreStdev = other.scoreStdev;
  order = other.order;
  pv = std::move(other.pv);
  return *this;
}

bool operator<(const AnalysisData& a0, const AnalysisData& a1) {
  if(a0.numVisits > a1.numVisits)
    return true;
  else if(a0.numVisits < a1.numVisits)
    return false;
  else if(a0.utility > a1.utility)
    return true;
  else if(a0.utility < a1.utility)
    return false;
  else
    return a0.policyPrior > a1.policyPrior;
}
