#include "../search/analysisdata.h"

AnalysisData::AnalysisData()
  :move(Board::NULL_LOC),
   numVisits(0),
   playSelectionValue(0.0),
   lcb(0.0),
   radius(0.0),
   utility(0.0),
   resultUtility(0.0),
   scoreUtility(0.0),
   winLossValue(0.0),
   policyPrior(0.0),
   scoreMean(0.0),
   scoreStdev(0.0),
   lead(0.0),
   ess(0.0),
   weightFactor(0.0),
   weightSum(0.0),
   weightSqSum(0.0),
   utilitySqAvg(0.0),
   scoreMeanSqAvg(0.0),
   order(0),
   isSymmetryOf(Board::NULL_LOC),
   symmetry(0),
   pv(),
   pvVisits(),
   pvEdgeVisits(),
   node(NULL)
{}

AnalysisData::AnalysisData(const AnalysisData& other)
  :move(other.move),
   numVisits(other.numVisits),
   playSelectionValue(other.playSelectionValue),
   lcb(other.lcb),
   radius(other.radius),
   utility(other.utility),
   resultUtility(other.resultUtility),
   scoreUtility(other.scoreUtility),
   winLossValue(other.winLossValue),
   policyPrior(other.policyPrior),
   scoreMean(other.scoreMean),
   scoreStdev(other.scoreStdev),
   lead(other.lead),
   ess(other.ess),
   weightFactor(other.weightFactor),
   weightSum(other.weightSum),
   weightSqSum(other.weightSqSum),
   utilitySqAvg(other.utilitySqAvg),
   scoreMeanSqAvg(other.scoreMeanSqAvg),
   order(other.order),
   isSymmetryOf(other.isSymmetryOf),
   symmetry(other.symmetry),
   pv(other.pv),
   pvVisits(other.pvVisits),
   pvEdgeVisits(other.pvEdgeVisits),
   node(other.node)
{}

AnalysisData::AnalysisData(AnalysisData&& other) noexcept
  :move(other.move),
   numVisits(other.numVisits),
   playSelectionValue(other.playSelectionValue),
   lcb(other.lcb),
   radius(other.radius),
   utility(other.utility),
   resultUtility(other.resultUtility),
   scoreUtility(other.scoreUtility),
   winLossValue(other.winLossValue),
   policyPrior(other.policyPrior),
   scoreMean(other.scoreMean),
   scoreStdev(other.scoreStdev),
   lead(other.lead),
   ess(other.ess),
   weightFactor(other.weightFactor),
   weightSum(other.weightSum),
   weightSqSum(other.weightSqSum),
   utilitySqAvg(other.utilitySqAvg),
   scoreMeanSqAvg(other.scoreMeanSqAvg),
   order(other.order),
   isSymmetryOf(other.isSymmetryOf),
   symmetry(other.symmetry),
   pv(std::move(other.pv)),
   pvVisits(std::move(other.pvVisits)),
   pvEdgeVisits(std::move(other.pvEdgeVisits)),
   node(other.node)
{}

AnalysisData::~AnalysisData()
{}

AnalysisData& AnalysisData::operator=(const AnalysisData& other) {
  if(this == &other)
    return *this;
  move = other.move;
  numVisits = other.numVisits;
  playSelectionValue = other.playSelectionValue;
  lcb = other.lcb;
  radius = other.radius;
  utility = other.utility;
  resultUtility = other.resultUtility;
  scoreUtility = other.scoreUtility;
  winLossValue = other.winLossValue;
  policyPrior = other.policyPrior;
  scoreMean = other.scoreMean;
  scoreStdev = other.scoreStdev;
  lead = other.lead;
  ess = other.ess;
  weightFactor = other.weightFactor;
  weightSum = other.weightSum;
  weightSqSum = other.weightSqSum;
  utilitySqAvg = other.utilitySqAvg;
  scoreMeanSqAvg = other.scoreMeanSqAvg;
  order = other.order;
  isSymmetryOf = other.isSymmetryOf;
  symmetry = other.symmetry;
  pv = other.pv;
  pvVisits = other.pvVisits;
  pvEdgeVisits = other.pvEdgeVisits;
  node = other.node;
  return *this;
}

AnalysisData& AnalysisData::operator=(AnalysisData&& other) noexcept {
  if(this == &other)
    return *this;
  move = other.move;
  numVisits = other.numVisits;
  playSelectionValue = other.playSelectionValue;
  lcb = other.lcb;
  radius = other.radius;
  utility = other.utility;
  resultUtility = other.resultUtility;
  scoreUtility = other.scoreUtility;
  winLossValue = other.winLossValue;
  policyPrior = other.policyPrior;
  scoreMean = other.scoreMean;
  scoreStdev = other.scoreStdev;
  lead = other.lead;
  ess = other.ess;
  weightFactor = other.weightFactor;
  weightSum = other.weightSum;
  weightSqSum = other.weightSqSum;
  utilitySqAvg = other.utilitySqAvg;
  scoreMeanSqAvg = other.scoreMeanSqAvg;
  order = other.order;
  isSymmetryOf = other.isSymmetryOf;
  symmetry = other.symmetry;
  pv = std::move(other.pv);
  pvVisits = std::move(other.pvVisits);
  pvEdgeVisits = std::move(other.pvEdgeVisits);
  node = other.node;
  return *this;
}

bool operator<(const AnalysisData& a0, const AnalysisData& a1) {
  // Sort all 0-visit moves to the end.
  if(a0.numVisits > 0 && a1.numVisits == 0)
    return true;
  if(a1.numVisits > 0 && a0.numVisits == 0)
    return false;
  // Then sort by play selection value, the normal value for chosing moves to play.
  if(a0.playSelectionValue > a1.playSelectionValue)
    return true;
  if(a0.playSelectionValue < a1.playSelectionValue)
    return false;
  // Then by visits
  if(a0.numVisits > a1.numVisits)
    return true;
  if(a0.numVisits < a1.numVisits)
    return false;
  // Then just by raw policy
  else
    return a0.policyPrior > a1.policyPrior;
}

bool AnalysisData::pvContainsPass() const {
  for(int i = 0; i<pv.size(); i++)
    if(pv[i] == Board::PASS_LOC)
      return true;
  return false;
}

void AnalysisData::writePV(std::ostream& out, const Board& board) const {
  for(int j = 0; j<pv.size(); j++) {
    if(j > 0)
      out << " ";
    out << Location::toString(pv[j],board);
  }
}

void AnalysisData::writePVVisits(std::ostream& out) const {
  for(int j = 0; j<pvVisits.size(); j++) {
    if(j > 0)
      out << " ";
    out << pvVisits[j];
  }
}
void AnalysisData::writePVEdgeVisits(std::ostream& out) const {
  for(int j = 0; j<pvEdgeVisits.size(); j++) {
    if(j > 0)
      out << " ";
    out << pvEdgeVisits[j];
  }
}

int AnalysisData::getPVLenUpToPhaseEnd(const Board& initialBoard, const BoardHistory& initialHist, Player initialPla) const {
  Board board(initialBoard);
  BoardHistory hist(initialHist);
  Player nextPla = initialPla;
  int j;
  for(j = 0; j<pv.size(); j++) {
    hist.makeBoardMoveAssumeLegal(board,pv[j],nextPla,NULL);
    nextPla = getOpp(nextPla);
    if(hist.encorePhase != initialHist.encorePhase)
      break;
  }
  return j;
}

void AnalysisData::writePVUpToPhaseEnd(std::ostream& out, const Board& initialBoard, const BoardHistory& initialHist, Player initialPla) const {
  Board board(initialBoard);
  BoardHistory hist(initialHist);
  Player nextPla = initialPla;
  for(int j = 0; j<pv.size(); j++) {
    if(j > 0)
      out << " ";
    out << Location::toString(pv[j],board);

    hist.makeBoardMoveAssumeLegal(board,pv[j],nextPla,NULL);
    nextPla = getOpp(nextPla);
    if(hist.encorePhase != initialHist.encorePhase)
      break;
  }
}

void AnalysisData::writePVVisitsUpToPhaseEnd(std::ostream& out, const Board& initialBoard, const BoardHistory& initialHist, Player initialPla) const {
  Board board(initialBoard);
  BoardHistory hist(initialHist);
  Player nextPla = initialPla;
  assert(pv.size() == pvVisits.size());
  for(int j = 0; j<pv.size(); j++) {
    if(j > 0)
      out << " ";
    out << pvVisits[j];

    hist.makeBoardMoveAssumeLegal(board,pv[j],nextPla,NULL);
    nextPla = getOpp(nextPla);
    if(hist.encorePhase != initialHist.encorePhase)
      break;
  }
}

void AnalysisData::writePVEdgeVisitsUpToPhaseEnd(std::ostream& out, const Board& initialBoard, const BoardHistory& initialHist, Player initialPla) const {
  Board board(initialBoard);
  BoardHistory hist(initialHist);
  Player nextPla = initialPla;
  assert(pv.size() == pvEdgeVisits.size());
  for(int j = 0; j<pv.size(); j++) {
    if(j > 0)
      out << " ";
    out << pvEdgeVisits[j];

    hist.makeBoardMoveAssumeLegal(board,pv[j],nextPla,NULL);
    nextPla = getOpp(nextPla);
    if(hist.encorePhase != initialHist.encorePhase)
      break;
  }
}
