#include "../search/search.h"

#include "../search/searchnode.h"

//------------------------
#include "../core/using.h"
//------------------------


double Search::numVisitsNeededToBeNonFutile(double maxVisitsMoveVisits) {
  double requiredVisits = searchParams.futileVisitsThreshold * maxVisitsMoveVisits;
  //In the case where we're playing high temperature, also require that we can't get to more than a 1:100 odds of playing the move.
  double chosenMoveTemperature = interpolateEarly(
    searchParams.chosenMoveTemperatureHalflife, searchParams.chosenMoveTemperatureEarly, searchParams.chosenMoveTemperature
  );
  if(chosenMoveTemperature < 1e-3)
    return requiredVisits;
  double requiredVisitsDueToTemp = maxVisitsMoveVisits * pow(0.01, chosenMoveTemperature);
  return std::min(requiredVisits, requiredVisitsDueToTemp);
}

double Search::computeUpperBoundVisitsLeftDueToTime(
  int64_t rootVisits, double timeUsed, double plannedTimeLimit
) {
  if(rootVisits <= 1)
    return 1e30;
  double timeThoughtSoFar = effectiveSearchTimeCarriedOver + timeUsed;
  double timeLeftPlanned = plannedTimeLimit - timeUsed;
  //Require at least a tenth of a second of search to begin to trust an estimate of visits/time.
  if(timeThoughtSoFar < 0.1)
    return 1e30;

  double proportionOfTimeThoughtLeft = timeLeftPlanned / timeThoughtSoFar;
  return ceil(proportionOfTimeThoughtLeft * rootVisits + searchParams.numThreads-1);
}

double Search::recomputeSearchTimeLimit(
  const TimeControls& tc, double timeUsed, double searchFactor, int64_t rootVisits
) {
  double tcMin;
  double tcRec;
  double tcMax;
  tc.getTime(rootBoard,rootHistory,searchParams.lagBuffer,tcMin,tcRec,tcMax);

  tcRec *= searchParams.overallocateTimeFactor;

  if(searchParams.midgameTimeFactor != 1.0) {
    double boardAreaScale = rootBoard.x_size * rootBoard.y_size / 361.0;
    double presumedTurnNumber = (double)rootHistory.getCurrentTurnNumber();
    if(presumedTurnNumber < 0) presumedTurnNumber = 0;

    double midGameWeight;
    if(presumedTurnNumber < searchParams.midgameTurnPeakTime * boardAreaScale)
      midGameWeight = presumedTurnNumber / (searchParams.midgameTurnPeakTime * boardAreaScale);
    else
      midGameWeight = exp(
        -(presumedTurnNumber - searchParams.midgameTurnPeakTime * boardAreaScale) /
        (searchParams.endgameTurnTimeDecay * boardAreaScale)
      );
    if(midGameWeight < 0)
      midGameWeight = 0;
    if(midGameWeight > 1)
      midGameWeight = 1;

    tcRec *= 1.0 + midGameWeight * (searchParams.midgameTimeFactor - 1.0);
  }

  if(searchParams.obviousMovesTimeFactor < 1.0) {
    double surprise = 0.0;
    double searchEntropy = 0.0;
    double policyEntropy = 0.0;
    bool suc = getPolicySurpriseAndEntropy(surprise, searchEntropy, policyEntropy);
    if(suc) {
      //If the original policy was confident and the surprise is low, then this is probably an "obvious" move.
      double obviousnessByEntropy = exp(-policyEntropy/searchParams.obviousMovesPolicyEntropyTolerance);
      double obviousnessBySurprise = exp(-surprise/searchParams.obviousMovesPolicySurpriseTolerance);
      double obviousnessWeight = std::min(obviousnessByEntropy, obviousnessBySurprise);
      tcRec *= 1.0 + obviousnessWeight * (searchParams.obviousMovesTimeFactor - 1.0);
    }
  }

  if(tcRec > 1e-20) {
    double remainingTimeNeeded = tcRec - effectiveSearchTimeCarriedOver;
    double remainingTimeNeededFactor = remainingTimeNeeded/tcRec;
    //TODO this is a bit conservative relative to old behavior, it might be of slightly detrimental value, needs testing.
    //Apply softplus so that we still do a tiny bit of search even in the presence of variable search time instead of instamoving,
    //there are some benefits from root-level search due to broader root exploration and the cost is small, also we may be over
    //counting the ponder benefit if search is faster on this node than on the previous turn.
    tcRec = tcRec * std::min(1.0, log(1.0+exp(remainingTimeNeededFactor * 6.0)) / 6.0);
  }

  //Make sure we're not wasting time
  tcRec = tc.roundUpTimeLimitIfNeeded(searchParams.lagBuffer,timeUsed,tcRec);
  if(tcRec > tcMax) tcRec = tcMax;

  //After rounding up time, check if with our planned rounded time, anything is futile to search
  if(searchParams.futileVisitsThreshold > 0) {
    double upperBoundVisitsLeftDueToTime = computeUpperBoundVisitsLeftDueToTime(rootVisits, timeUsed, tcRec);
    if(upperBoundVisitsLeftDueToTime < searchParams.futileVisitsThreshold * rootVisits) {
      vector<Loc> locs;
      vector<double> playSelectionValues;
      vector<double> visitCounts;
      bool suc = getPlaySelectionValues(locs, playSelectionValues, &visitCounts, 1.0);
      if(suc && playSelectionValues.size() > 0) {
        //This may fail to hold if we have no actual visits and play selections are being pulled from stuff like raw policy
        if(playSelectionValues.size() == visitCounts.size()) {
          int numMoves = (int)playSelectionValues.size();
          int maxVisitsIdx = 0;
          int bestMoveIdx = 0;
          for(int i = 1; i<numMoves; i++) {
            if(playSelectionValues[i] > playSelectionValues[bestMoveIdx])
              bestMoveIdx = i;
            if(visitCounts[i] > visitCounts[maxVisitsIdx])
              maxVisitsIdx = i;
          }
          if(maxVisitsIdx == bestMoveIdx) {
            double requiredVisits = numVisitsNeededToBeNonFutile(visitCounts[maxVisitsIdx]);
            bool foundPossibleAlternativeMove = false;
            for(int i = 0; i<numMoves; i++) {
              if(i == bestMoveIdx)
                continue;
              if(visitCounts[i] + upperBoundVisitsLeftDueToTime >= requiredVisits) {
                foundPossibleAlternativeMove = true;
                break;
              }
            }
            if(!foundPossibleAlternativeMove) {
              //We should stop search now - set our desired thinking to very slightly smaller than what we used.
              tcRec = timeUsed * (1.0 - (1e-10));
            }
          }
        }
      }
    }
  }

  //Make sure we're not wasting time, even after considering that we might want to stop early
  tcRec = tc.roundUpTimeLimitIfNeeded(searchParams.lagBuffer,timeUsed,tcRec);
  if(tcRec > tcMax) tcRec = tcMax;

  //Apply caps and search factor
  //Since searchFactor is mainly used for friendliness (like, play faster after many passes)
  //we allow it to violate the min time.
  if(tcRec < tcMin) tcRec = tcMin;
  tcRec *= searchFactor;
  if(tcRec > tcMax) tcRec = tcMax;

  return tcRec;
}
