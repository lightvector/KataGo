#include "../search/search.h"

#include "../search/searchnode.h"

//------------------------
#include "../core/using.h"
//------------------------



void Search::addLeafValue(
  SearchNode& node,
  double winLossValue,
  double noResultValue,
  double scoreMean,
  double scoreMeanSq,
  double lead,
  double weight,
  bool isTerminal,
  bool assumeNoExistingWeight
) {
  double utility =
    getResultUtility(winLossValue, noResultValue)
    + getScoreUtility(scoreMean, scoreMeanSq);

  if(searchParams.subtreeValueBiasFactor != 0 && !isTerminal && node.subtreeValueBiasTableEntry != nullptr) {
    SubtreeValueBiasEntry& entry = *(node.subtreeValueBiasTableEntry);
    while(entry.entryLock.test_and_set(std::memory_order_acquire));
    double newEntryDeltaUtilitySum = entry.deltaUtilitySum;
    double newEntryWeightSum = entry.weightSum;
    entry.entryLock.clear(std::memory_order_release);
    //This is the amount of the direct evaluation of this node that we are going to bias towards the table entry
    const double biasFactor = searchParams.subtreeValueBiasFactor;
    if(newEntryWeightSum > 0.001)
      utility += biasFactor * newEntryDeltaUtilitySum / newEntryWeightSum;
  }

  utility += getPatternBonus(node.patternBonusHash,getOpp(node.nextPla));

  double utilitySq = utility * utility;
  double weightSq = weight * weight;

  if(assumeNoExistingWeight) {
    while(node.statsLock.test_and_set(std::memory_order_acquire));
    node.stats.winLossValueAvg.store(winLossValue,std::memory_order_release);
    node.stats.noResultValueAvg.store(noResultValue,std::memory_order_release);
    node.stats.scoreMeanAvg.store(scoreMean,std::memory_order_release);
    node.stats.scoreMeanSqAvg.store(scoreMeanSq,std::memory_order_release);
    node.stats.leadAvg.store(lead,std::memory_order_release);
    node.stats.utilityAvg.store(utility,std::memory_order_release);
    node.stats.utilitySqAvg.store(utilitySq,std::memory_order_release);
    node.stats.weightSqSum.store(weightSq,std::memory_order_release);
    node.stats.weightSum.store(weight,std::memory_order_release);
    int64_t oldVisits = node.stats.visits.fetch_add(1,std::memory_order_release);
    node.statsLock.clear(std::memory_order_release);
    assert(oldVisits == 0);
  }
  else {
    while(node.statsLock.test_and_set(std::memory_order_acquire));
    double oldWeightSum = node.stats.weightSum.load(std::memory_order_relaxed);
    double newWeightSum = oldWeightSum + weight;

    node.stats.winLossValueAvg.store((node.stats.winLossValueAvg.load(std::memory_order_relaxed) * oldWeightSum + winLossValue * weight)/newWeightSum,std::memory_order_release);
    node.stats.noResultValueAvg.store((node.stats.noResultValueAvg.load(std::memory_order_relaxed) * oldWeightSum + noResultValue * weight)/newWeightSum,std::memory_order_release);
    node.stats.scoreMeanAvg.store((node.stats.scoreMeanAvg.load(std::memory_order_relaxed) * oldWeightSum + scoreMean * weight)/newWeightSum,std::memory_order_release);
    node.stats.scoreMeanSqAvg.store((node.stats.scoreMeanSqAvg.load(std::memory_order_relaxed) * oldWeightSum + scoreMeanSq * weight)/newWeightSum,std::memory_order_release);
    node.stats.leadAvg.store((node.stats.leadAvg.load(std::memory_order_relaxed) * oldWeightSum + lead * weight)/newWeightSum,std::memory_order_release);
    node.stats.utilityAvg.store((node.stats.utilityAvg.load(std::memory_order_relaxed) * oldWeightSum + utility * weight)/newWeightSum,std::memory_order_release);
    node.stats.utilitySqAvg.store((node.stats.utilitySqAvg.load(std::memory_order_relaxed) * oldWeightSum + utilitySq * weight)/newWeightSum,std::memory_order_release);
    node.stats.weightSqSum.store(node.stats.weightSqSum.load(std::memory_order_relaxed) + weightSq,std::memory_order_release);
    node.stats.weightSum.store(newWeightSum,std::memory_order_release);
    node.stats.visits.fetch_add(1,std::memory_order_release);
    node.statsLock.clear(std::memory_order_release);
  }
}

void Search::addCurrentNNOutputAsLeafValue(SearchNode& node, bool assumeNoExistingWeight) {
  const NNOutput* nnOutput = node.getNNOutput();
  assert(nnOutput != NULL);
  //Values in the search are from the perspective of white positive always
  double winProb = (double)nnOutput->whiteWinProb;
  double lossProb = (double)nnOutput->whiteLossProb;
  double noResultProb = (double)nnOutput->whiteNoResultProb;
  double scoreMean = (double)nnOutput->whiteScoreMean;
  double scoreMeanSq = (double)nnOutput->whiteScoreMeanSq;
  double lead = (double)nnOutput->whiteLead;
  double weight = computeWeightFromNNOutput(nnOutput);
  addLeafValue(node,winProb-lossProb,noResultProb,scoreMean,scoreMeanSq,lead,weight,false,assumeNoExistingWeight);
}

double Search::computeWeightFromNNOutput(const NNOutput* nnOutput) const {
  if(!searchParams.useUncertainty)
    return 1.0;
  if(!nnEvaluator->supportsShorttermError())
    return 1.0;

  double scoreMean = (double)nnOutput->whiteScoreMean;
  double utilityUncertaintyWL = searchParams.winLossUtilityFactor * nnOutput->shorttermWinlossError;
  double utilityUncertaintyScore = getApproxScoreUtilityDerivative(scoreMean) * nnOutput->shorttermScoreError;
  double utilityUncertainty = utilityUncertaintyWL + utilityUncertaintyScore;

  double poweredUncertainty;
  if(searchParams.uncertaintyExponent == 1.0)
    poweredUncertainty = utilityUncertainty;
  else if(searchParams.uncertaintyExponent == 0.5)
    poweredUncertainty = sqrt(utilityUncertainty);
  else
    poweredUncertainty = pow(utilityUncertainty, searchParams.uncertaintyExponent);

  double baselineUncertainty = searchParams.uncertaintyCoeff / searchParams.uncertaintyMaxWeight;
  double weight = searchParams.uncertaintyCoeff / (poweredUncertainty + baselineUncertainty);
  return weight;
}


void Search::updateStatsAfterPlayout(SearchNode& node, SearchThread& thread, bool isRoot) {
  //The thread that grabs a 0 from this peforms the recomputation of stats.
  int32_t oldDirtyCounter = node.dirtyCounter.fetch_add(1,std::memory_order_acq_rel);
  assert(oldDirtyCounter >= 0);
  //If we atomically grab a nonzero, then we know another thread must already be doing the work, so we can skip the update ourselves.
  if(oldDirtyCounter > 0)
    return;
  int32_t numVisitsCompleted = 1;
  while(true) {
    //Perform update
    recomputeNodeStats(node,thread,numVisitsCompleted,isRoot);
    //Now attempt to undo the counter
    oldDirtyCounter = node.dirtyCounter.fetch_add(-numVisitsCompleted,std::memory_order_acq_rel);
    int32_t newDirtyCounter = oldDirtyCounter - numVisitsCompleted;
    //If no other threads incremented it in the meantime, so our decrement hits zero, we're done.
    if(newDirtyCounter <= 0) {
      assert(newDirtyCounter == 0);
      break;
    }
    //Otherwise, more threads incremented this more in the meantime. So we need to loop again and add their visits, recomputing again.
    numVisitsCompleted = newDirtyCounter;
    continue;
  }
}

//Recompute all the stats of this node based on its children, except its visits and virtual losses, which are not child-dependent and
//are updated in the manner specified.
//Assumes this node has an nnOutput
void Search::recomputeNodeStats(SearchNode& node, SearchThread& thread, int numVisitsToAdd, bool isRoot) {
  //Find all children and compute weighting of the children based on their values
  vector<MoreNodeStats>& statsBuf = thread.statsBuf;
  int numGoodChildren = 0;

  int childrenCapacity;
  const SearchChildPointer* children = node.getChildren(childrenCapacity);
  double origTotalChildWeight = 0.0;
  for(int i = 0; i<childrenCapacity; i++) {
    const SearchNode* child = children[i].getIfAllocated();
    if(child == NULL)
      break;
    MoreNodeStats& stats = statsBuf[numGoodChildren];

    Loc moveLoc = children[i].getMoveLocRelaxed();
    int64_t edgeVisits = children[i].getEdgeVisits();
    stats.stats = NodeStats(child->stats);

    if(stats.stats.visits <= 0 || stats.stats.weightSum <= 0.0 || edgeVisits <= 0)
      continue;

    double childWeight = stats.stats.weightSum * ((double)edgeVisits / (double)stats.stats.visits);
    double childUtility = stats.stats.utilityAvg;
    stats.selfUtility = node.nextPla == P_WHITE ? childUtility : -childUtility;
    stats.weightAdjusted = childWeight;
    stats.prevMoveLoc = moveLoc;

    origTotalChildWeight += stats.weightAdjusted;
    numGoodChildren++;
  }

  //Always tracks the sum of statsBuf[i].weightAdjusted across the children.
  double currentTotalChildWeight = origTotalChildWeight;

  if(searchParams.useNoisePruning && numGoodChildren > 0 && !(searchParams.antiMirror && mirroringPla != C_EMPTY)) {
    double policyProbsBuf[NNPos::MAX_NN_POLICY_SIZE];
    {
      const NNOutput* nnOutput = node.getNNOutput();
      assert(nnOutput != NULL);
      const float* policyProbs = nnOutput->getPolicyProbsMaybeNoised();
      for(int i = 0; i<numGoodChildren; i++)
        policyProbsBuf[i] = std::max(1e-30, (double)policyProbs[getPos(statsBuf[i].prevMoveLoc)]);
    }
    currentTotalChildWeight = pruneNoiseWeight(statsBuf, numGoodChildren, currentTotalChildWeight, policyProbsBuf);
  }

  {
    double amountToSubtract = 0.0;
    double amountToPrune = 0.0;
    if(isRoot && searchParams.rootNoiseEnabled && !searchParams.useNoisePruning) {
      double maxChildWeight = 0.0;
      for(int i = 0; i<numGoodChildren; i++) {
        if(statsBuf[i].weightAdjusted > maxChildWeight)
          maxChildWeight = statsBuf[i].weightAdjusted;
      }
      amountToSubtract = std::min(searchParams.chosenMoveSubtract, maxChildWeight/64.0);
      amountToPrune = std::min(searchParams.chosenMovePrune, maxChildWeight/64.0);
    }

    downweightBadChildrenAndNormalizeWeight(
      numGoodChildren, currentTotalChildWeight, currentTotalChildWeight,
      amountToSubtract, amountToPrune, statsBuf
    );
  }

  double winLossValueSum = 0.0;
  double noResultValueSum = 0.0;
  double scoreMeanSum = 0.0;
  double scoreMeanSqSum = 0.0;
  double leadSum = 0.0;
  double utilitySum = 0.0;
  double utilitySqSum = 0.0;
  double weightSqSum = 0.0;
  double weightSum = currentTotalChildWeight;
  for(int i = 0; i<numGoodChildren; i++) {
    const NodeStats& stats = statsBuf[i].stats;

    double desiredWeight = statsBuf[i].weightAdjusted;
    double weightScaling = desiredWeight / stats.weightSum;

    winLossValueSum += desiredWeight * stats.winLossValueAvg;
    noResultValueSum += desiredWeight * stats.noResultValueAvg;
    scoreMeanSum += desiredWeight * stats.scoreMeanAvg;
    scoreMeanSqSum += desiredWeight * stats.scoreMeanSqAvg;
    leadSum += desiredWeight * stats.leadAvg;
    utilitySum += desiredWeight * stats.utilityAvg;
    utilitySqSum += desiredWeight * stats.utilitySqAvg;
    weightSqSum += weightScaling * weightScaling * stats.weightSqSum;
  }

  //Also add in the direct evaluation of this node.
  {
    const NNOutput* nnOutput = node.getNNOutput();
    assert(nnOutput != NULL);
    double winProb = (double)nnOutput->whiteWinProb;
    double lossProb = (double)nnOutput->whiteLossProb;
    double noResultProb = (double)nnOutput->whiteNoResultProb;
    double scoreMean = (double)nnOutput->whiteScoreMean;
    double scoreMeanSq = (double)nnOutput->whiteScoreMeanSq;
    double lead = (double)nnOutput->whiteLead;
    double utility =
      getResultUtility(winProb-lossProb, noResultProb)
      + getScoreUtility(scoreMean, scoreMeanSq);

    if(searchParams.subtreeValueBiasFactor != 0 && node.subtreeValueBiasTableEntry != nullptr) {
      SubtreeValueBiasEntry& entry = *(node.subtreeValueBiasTableEntry);

      double newEntryDeltaUtilitySum;
      double newEntryWeightSum;

      if(currentTotalChildWeight > 1e-10) {
        double utilityChildren = utilitySum / currentTotalChildWeight;
        double subtreeValueBiasWeight = pow(origTotalChildWeight, searchParams.subtreeValueBiasWeightExponent);
        double subtreeValueBiasDeltaSum = (utilityChildren - utility) * subtreeValueBiasWeight;

        while(entry.entryLock.test_and_set(std::memory_order_acquire));
        entry.deltaUtilitySum += subtreeValueBiasDeltaSum - node.lastSubtreeValueBiasDeltaSum;
        entry.weightSum += subtreeValueBiasWeight - node.lastSubtreeValueBiasWeight;
        newEntryDeltaUtilitySum = entry.deltaUtilitySum;
        newEntryWeightSum = entry.weightSum;
        node.lastSubtreeValueBiasDeltaSum = subtreeValueBiasDeltaSum;
        node.lastSubtreeValueBiasWeight = subtreeValueBiasWeight;
        entry.entryLock.clear(std::memory_order_release);
      }
      else {
        while(entry.entryLock.test_and_set(std::memory_order_acquire));
        newEntryDeltaUtilitySum = entry.deltaUtilitySum;
        newEntryWeightSum = entry.weightSum;
        entry.entryLock.clear(std::memory_order_release);
      }

      //This is the amount of the direct evaluation of this node that we are going to bias towards the table entry
      const double biasFactor = searchParams.subtreeValueBiasFactor;
      if(newEntryWeightSum > 0.001)
        utility += biasFactor * newEntryDeltaUtilitySum / newEntryWeightSum;
      //This is the amount by which we need to scale desiredSelfWeight such that if the table entry were actually equal to
      //the current difference between the direct eval and the children, we would perform a no-op... unless a noop is actually impossible
      //Then we just take what we can get.
      //desiredSelfWeight *= weightSum / (1.0-biasFactor) / std::max(0.001, (weightSum + desiredSelfWeight - desiredSelfWeight / (1.0-biasFactor)));
    }

    double weight = computeWeightFromNNOutput(nnOutput);
    winLossValueSum += (winProb - lossProb) * weight;
    noResultValueSum += noResultProb * weight;
    scoreMeanSum += scoreMean * weight;
    scoreMeanSqSum += scoreMeanSq * weight;
    leadSum += lead * weight;
    utilitySum += utility * weight;
    utilitySqSum += utility * utility * weight;
    weightSqSum += weight * weight;
    weightSum += weight;
  }

  double winLossValueAvg = winLossValueSum / weightSum;
  double noResultValueAvg = noResultValueSum / weightSum;
  double scoreMeanAvg = scoreMeanSum / weightSum;
  double scoreMeanSqAvg = scoreMeanSqSum / weightSum;
  double leadAvg = leadSum / weightSum;
  double utilityAvg = utilitySum / weightSum;
  double utilitySqAvg = utilitySqSum / weightSum;

  double oldUtilityAvg = utilityAvg;
  utilityAvg += getPatternBonus(node.patternBonusHash,getOpp(node.nextPla));
  utilitySqAvg = utilitySqAvg + (utilityAvg * utilityAvg - oldUtilityAvg * oldUtilityAvg);

  //TODO statslock may be unnecessary now with the dirtyCounter mechanism?
  while(node.statsLock.test_and_set(std::memory_order_acquire));
  node.stats.winLossValueAvg.store(winLossValueAvg,std::memory_order_release);
  node.stats.noResultValueAvg.store(noResultValueAvg,std::memory_order_release);
  node.stats.scoreMeanAvg.store(scoreMeanAvg,std::memory_order_release);
  node.stats.scoreMeanSqAvg.store(scoreMeanSqAvg,std::memory_order_release);
  node.stats.leadAvg.store(leadAvg,std::memory_order_release);
  node.stats.utilityAvg.store(utilityAvg,std::memory_order_release);
  node.stats.utilitySqAvg.store(utilitySqAvg,std::memory_order_release);
  node.stats.weightSqSum.store(weightSqSum,std::memory_order_release);
  node.stats.weightSum.store(weightSum,std::memory_order_release);
  node.stats.visits.fetch_add(numVisitsToAdd,std::memory_order_release);
  node.statsLock.clear(std::memory_order_release);
}
