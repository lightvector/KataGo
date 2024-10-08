#include "../search/search.h"

#include "../search/searchnode.h"

//------------------------
#include "../core/using.h"
//------------------------

static double cpuctExploration(double totalChildWeight, const SearchParams& searchParams) {
  return searchParams.cpuctExploration +
    searchParams.cpuctExplorationLog * log((totalChildWeight + searchParams.cpuctExplorationBase) / searchParams.cpuctExplorationBase);
}

static double cpuctExplorationHuman(double totalChildWeight, const SearchParams& searchParams) {
  return searchParams.humanSLCpuctExploration + searchParams.humanSLCpuctPermanent * sqrt(totalChildWeight);
}

//Tiny constant to add to numerator of puct formula to make it positive
//even when visits = 0.
static constexpr double TOTALCHILDWEIGHT_PUCT_OFFSET = 0.01;

double Search::getExploreScaling(
  double totalChildWeight, double parentUtilityStdevFactor
) const {
  return
    cpuctExploration(totalChildWeight, searchParams)
    * sqrt(totalChildWeight + TOTALCHILDWEIGHT_PUCT_OFFSET)
    * parentUtilityStdevFactor;
}
double Search::getExploreScalingHuman(
  double totalChildWeight
) const {
  return
    cpuctExplorationHuman(totalChildWeight, searchParams)
    * sqrt(totalChildWeight + TOTALCHILDWEIGHT_PUCT_OFFSET);
}

double Search::getExploreSelectionValue(
  double exploreScaling,
  double nnPolicyProb,
  double childWeight,
  double childUtility,
  Player pla
) const {
  if(nnPolicyProb < 0)
    return POLICY_ILLEGAL_SELECTION_VALUE;

  double exploreComponent = exploreScaling * nnPolicyProb / (1.0 + childWeight);

  //At the last moment, adjust value to be from the player's perspective, so that players prefer values in their favor
  //rather than in white's favor
  double valueComponent = pla == P_WHITE ? childUtility : -childUtility;
  return exploreComponent + valueComponent;
}

//Return the childWeight that would make Search::getExploreSelectionValue return the given explore selection value.
//Or return 0, if it would be less than 0.
double Search::getExploreSelectionValueInverse(
  double exploreSelectionValue,
  double exploreScaling,
  double nnPolicyProb,
  double childUtility,
  Player pla
) const {
  if(nnPolicyProb < 0)
    return 0;
  double valueComponent = pla == P_WHITE ? childUtility : -childUtility;

  double exploreComponent = exploreSelectionValue - valueComponent;
  double exploreComponentScaling = exploreScaling * nnPolicyProb;

  //Guard against float weirdness
  if(exploreComponent <= 0)
    return 1e100;

  double childWeight = exploreComponentScaling / exploreComponent - 1;
  if(childWeight < 0)
    childWeight = 0;
  return childWeight;
}

static void maybeApplyWideRootNoise(
  double& childUtility,
  float& nnPolicyProb,
  const SearchParams& searchParams,
  SearchThread* thread,
  const SearchNode& parent
) {
  //For very large wideRootNoise, go ahead and also smooth out the policy
  nnPolicyProb = (float)pow(nnPolicyProb, 1.0 / (4.0*searchParams.wideRootNoise + 1.0));
  if(thread->rand.nextBool(0.5)) {
    double bonus = searchParams.wideRootNoise * std::fabs(thread->rand.nextGaussian());
    if(parent.nextPla == P_WHITE)
      childUtility += bonus;
    else
      childUtility -= bonus;
  }
}


double Search::getExploreSelectionValueOfChild(
  const SearchNode& parent, const float* parentPolicyProbs, const SearchNode* child,
  Loc moveLoc,
  double exploreScaling,
  double totalChildWeight, int64_t childEdgeVisits, double fpuValue,
  double parentUtility, double parentWeightPerVisit,
  bool isDuringSearch, bool antiMirror, double maxChildWeight,
  bool countEdgeVisit,
  SearchThread* thread
) const {
  (void)parentUtility;
  int movePos = getPos(moveLoc);
  float nnPolicyProb = parentPolicyProbs[movePos];

  int32_t childVirtualLosses = child->virtualLosses.load(std::memory_order_acquire);
  int64_t childVisits = child->stats.visits.load(std::memory_order_acquire);
  double utilityAvg = child->stats.utilityAvg.load(std::memory_order_acquire);
  double scoreMeanAvg = child->stats.scoreMeanAvg.load(std::memory_order_acquire);
  double scoreMeanSqAvg = child->stats.scoreMeanSqAvg.load(std::memory_order_acquire);
  double childWeight;
  if(countEdgeVisit)
    childWeight = child->stats.getChildWeight(childEdgeVisits,childVisits);
  else
    childWeight = child->stats.weightSum.load(std::memory_order_acquire);

  //It's possible that childVisits is actually 0 here with multithreading because we're visiting this node while a child has
  //been expanded but its thread not yet finished its first visit.
  //It's also possible that we observe childWeight <= 0 even though childVisits >= due to multithreading, the two could
  //be out of sync briefly since they are separate atomics.
  double childUtility;
  if(childVisits <= 0 || childWeight <= 0.0)
    childUtility = fpuValue;
  else {
    childUtility = utilityAvg;

    //Tiny adjustment for passing
    double endingScoreBonus = getEndingWhiteScoreBonus(parent,moveLoc);
    if(endingScoreBonus != 0)
      childUtility += getScoreUtilityDiff(scoreMeanAvg, scoreMeanSqAvg, endingScoreBonus);
  }

  //Virtual losses to direct threads down different paths
  if(childVirtualLosses > 0) {
    double virtualLossWeight = childVirtualLosses * searchParams.numVirtualLossesPerThread;

    double utilityRadius = searchParams.winLossUtilityFactor + searchParams.staticScoreUtilityFactor + searchParams.dynamicScoreUtilityFactor;
    double virtualLossUtility = (parent.nextPla == P_WHITE ? -utilityRadius : utilityRadius);
    double virtualLossWeightFrac = (double)virtualLossWeight / (virtualLossWeight + std::max(0.25,childWeight));
    childUtility = childUtility + (virtualLossUtility - childUtility) * virtualLossWeightFrac;
    childWeight += virtualLossWeight;
  }

  if(isDuringSearch && (&parent == rootNode) && countEdgeVisit) {
    //Futile visits pruning - skip this move if the amount of time we have left to search is too small, assuming
    //its average weight per visit is maintained.
    //We use childVisits rather than childEdgeVisits for the final estimate since when childEdgeVisits < childVisits, adding new visits is instant.
    if(searchParams.futileVisitsThreshold > 0) {
      double requiredWeight = searchParams.futileVisitsThreshold * maxChildWeight;
      //Avoid divide by 0 by adding a prior equal to the parent's weight per visit
      double averageVisitsPerWeight = (childEdgeVisits + 1.0) / (childWeight + parentWeightPerVisit);
      double estimatedRequiredVisits = requiredWeight * averageVisitsPerWeight;
      if(childVisits + thread->upperBoundVisitsLeft < estimatedRequiredVisits)
        return FUTILE_VISITS_PRUNE_VALUE;
    }
    //Hack to get the root to funnel more visits down child branches
    if(searchParams.rootDesiredPerChildVisitsCoeff > 0.0) {
      if(nnPolicyProb > 0 && childWeight < sqrt(nnPolicyProb * totalChildWeight * searchParams.rootDesiredPerChildVisitsCoeff)) {
        return 1e20;
      }
    }
    //Hack for hintloc - must search this move almost as often as the most searched move
    if(rootHintLoc != Board::NULL_LOC && moveLoc == rootHintLoc) {
      double averageWeightPerVisit = (childWeight + parentWeightPerVisit) / (childVisits + 1.0);
      ConstSearchNodeChildrenReference children = parent.getChildren();
      int childrenCapacity = children.getCapacity();
      for(int i = 0; i<childrenCapacity; i++) {
        const SearchChildPointer& childPointer = children[i];
        const SearchNode* c = childPointer.getIfAllocated();
        if(c == NULL)
          break;
        int64_t cEdgeVisits = childPointer.getEdgeVisits();
        double cWeight = c->stats.getChildWeight(cEdgeVisits);
        if(childWeight + averageWeightPerVisit < cWeight * 0.8)
          return 1e20;
      }
    }

    if(searchParams.wideRootNoise > 0.0 && nnPolicyProb >= 0) {
      maybeApplyWideRootNoise(childUtility, nnPolicyProb, searchParams, thread, parent);
    }
  }
  if(isDuringSearch && antiMirror && nnPolicyProb >= 0 && countEdgeVisit) {
    maybeApplyAntiMirrorPolicy(nnPolicyProb, moveLoc, parentPolicyProbs, parent.nextPla, thread);
    maybeApplyAntiMirrorForcedExplore(childUtility, parentUtility, moveLoc, parentPolicyProbs, childWeight, totalChildWeight, parent.nextPla, thread, parent);
  }

  return getExploreSelectionValue(exploreScaling,nnPolicyProb,childWeight,childUtility,parent.nextPla);
}

double Search::getNewExploreSelectionValue(
  const SearchNode& parent,
  double exploreScaling,
  float nnPolicyProb,
  double fpuValue,
  double parentWeightPerVisit,
  double maxChildWeight,
  bool countEdgeVisit,
  SearchThread* thread
) const {
  double childWeight = 0;
  double childUtility = fpuValue;
  if(&parent == rootNode && countEdgeVisit) {
    //Futile visits pruning - skip this move if the amount of time we have left to search is too small
    if(searchParams.futileVisitsThreshold > 0) {
      //Avoid divide by 0 by adding a prior equal to the parent's weight per visit
      double averageVisitsPerWeight = 1.0 / parentWeightPerVisit;
      double requiredWeight = searchParams.futileVisitsThreshold * maxChildWeight;
      double estimatedRequiredVisits = requiredWeight * averageVisitsPerWeight;
      if(thread->upperBoundVisitsLeft < estimatedRequiredVisits)
        return FUTILE_VISITS_PRUNE_VALUE;
    }
    if(searchParams.wideRootNoise > 0.0) {
      maybeApplyWideRootNoise(childUtility, nnPolicyProb, searchParams, thread, parent);
    }
  }
  return getExploreSelectionValue(exploreScaling,nnPolicyProb,childWeight,childUtility,parent.nextPla);
}

double Search::getReducedPlaySelectionWeight(
  const SearchNode& parent, const float* parentPolicyProbs, const SearchNode* child,
  Loc moveLoc,
  double exploreScaling,
  int64_t childEdgeVisits,
  double bestChildExploreSelectionValue
) const {
  assert(&parent == rootNode);
  int movePos = getPos(moveLoc);
  float nnPolicyProb = parentPolicyProbs[movePos];

  int64_t childVisits = child->stats.visits.load(std::memory_order_acquire);
  double scoreMeanAvg = child->stats.scoreMeanAvg.load(std::memory_order_acquire);
  double scoreMeanSqAvg = child->stats.scoreMeanSqAvg.load(std::memory_order_acquire);
  double utilityAvg = child->stats.utilityAvg.load(std::memory_order_acquire);
  double childWeight = child->stats.getChildWeight(childEdgeVisits,childVisits);

  //Child visits may be 0 if this function is called in a multithreaded context, such as during live analysis
  //Child weight may also be 0 if it's out of sync.
  if(childVisits <= 0 || childWeight <= 0.0)
    return 0;

  //Tiny adjustment for passing
  double endingScoreBonus = getEndingWhiteScoreBonus(parent,moveLoc);
  double childUtility = utilityAvg;
  if(endingScoreBonus != 0)
    childUtility += getScoreUtilityDiff(scoreMeanAvg, scoreMeanSqAvg, endingScoreBonus);

  double childWeightWeRetrospectivelyWanted = getExploreSelectionValueInverse(
    bestChildExploreSelectionValue, exploreScaling, nnPolicyProb, childUtility, parent.nextPla
  );
  if(childWeight > childWeightWeRetrospectivelyWanted)
    return childWeightWeRetrospectivelyWanted;
  return childWeight;
}

double Search::getFpuValueForChildrenAssumeVisited(
  const SearchNode& node, Player pla, bool isRoot, double policyProbMassVisited,
  double& parentUtility, double& parentWeightPerVisit, double& parentUtilityStdevFactor
) const {
  int64_t visits = node.stats.visits.load(std::memory_order_acquire);
  double weightSum = node.stats.weightSum.load(std::memory_order_acquire);
  double utilityAvg = node.stats.utilityAvg.load(std::memory_order_acquire);
  double utilitySqAvg = node.stats.utilitySqAvg.load(std::memory_order_acquire);

  assert(visits > 0);
  assert(weightSum > 0.0);
  parentWeightPerVisit = weightSum / visits;
  parentUtility = utilityAvg;
  double variancePrior = searchParams.cpuctUtilityStdevPrior * searchParams.cpuctUtilityStdevPrior;
  double variancePriorWeight = searchParams.cpuctUtilityStdevPriorWeight;
  double parentUtilityStdev;
  if(visits <= 0 || weightSum <= 1)
    parentUtilityStdev = searchParams.cpuctUtilityStdevPrior;
  else {
    double utilitySq = parentUtility * parentUtility;
    //Make sure we're robust to numerical precision issues or threading desync of these values, so we don't observe negative variance
    if(utilitySqAvg < utilitySq)
      utilitySqAvg = utilitySq;
    parentUtilityStdev = sqrt(
      std::max(
        0.0,
        ((utilitySq + variancePrior) * variancePriorWeight + utilitySqAvg * weightSum)
        / (variancePriorWeight + weightSum - 1.0)
        - utilitySq
      )
    );
  }
  parentUtilityStdevFactor = 1.0 + searchParams.cpuctUtilityStdevScale * (parentUtilityStdev / searchParams.cpuctUtilityStdevPrior - 1.0);

  double parentUtilityForFPU = parentUtility;
  if(searchParams.fpuParentWeightByVisitedPolicy) {
    double avgWeight = std::min(1.0, pow(policyProbMassVisited, searchParams.fpuParentWeightByVisitedPolicyPow));
    parentUtilityForFPU = avgWeight * parentUtility + (1.0 - avgWeight) * getUtilityFromNN(*(node.getNNOutput()));
  }
  else if(searchParams.fpuParentWeight > 0.0) {
    parentUtilityForFPU = searchParams.fpuParentWeight * getUtilityFromNN(*(node.getNNOutput())) + (1.0 - searchParams.fpuParentWeight) * parentUtility;
  }

  double fpuValue;
  {
    double fpuReductionMax = isRoot ? searchParams.rootFpuReductionMax : searchParams.fpuReductionMax;
    double fpuLossProp = isRoot ? searchParams.rootFpuLossProp : searchParams.fpuLossProp;
    double utilityRadius = searchParams.winLossUtilityFactor + searchParams.staticScoreUtilityFactor + searchParams.dynamicScoreUtilityFactor;

    double reduction = fpuReductionMax * sqrt(policyProbMassVisited);
    fpuValue = pla == P_WHITE ? parentUtilityForFPU - reduction : parentUtilityForFPU + reduction;
    double lossValue = pla == P_WHITE ? -utilityRadius : utilityRadius;
    fpuValue = fpuValue + (lossValue - fpuValue) * fpuLossProp;
  }

  return fpuValue;
}


void Search::selectBestChildToDescend(
  SearchThread& thread, const SearchNode& node, SearchNodeState nodeState,
  int& numChildrenFound, int& bestChildIdx, Loc& bestChildMoveLoc, bool& countEdgeVisit,
  bool isRoot) const
{
  assert(thread.pla == node.nextPla);

  double maxSelectionValue = POLICY_ILLEGAL_SELECTION_VALUE;
  bestChildIdx = -1;
  bestChildMoveLoc = Board::NULL_LOC;
  countEdgeVisit = true;

  ConstSearchNodeChildrenReference children = node.getChildren(nodeState);
  int childrenCapacity = children.getCapacity();

  double policyProbMassVisited = 0.0;
  double maxChildWeight = 0.0;
  double totalChildWeight = 0.0;
  const NNOutput* nnOutput = node.getNNOutput();
  assert(nnOutput != NULL);
  const float* policyProbs = nnOutput->getPolicyProbsMaybeNoised();
  for(int i = 0; i<childrenCapacity; i++) {
    const SearchChildPointer& childPointer = children[i];
    const SearchNode* child = childPointer.getIfAllocated();
    if(child == NULL)
      break;
    Loc moveLoc = childPointer.getMoveLocRelaxed();
    int movePos = getPos(moveLoc);
    float nnPolicyProb = policyProbs[movePos];
    if(nnPolicyProb < 0)
      continue;
    policyProbMassVisited += nnPolicyProb;

    int64_t edgeVisits = childPointer.getEdgeVisits();
    double childWeight = child->stats.getChildWeight(edgeVisits);

    totalChildWeight += childWeight;
    if(childWeight > maxChildWeight)
      maxChildWeight = childWeight;
  }

  bool useHumanSL = false;
  if(humanEvaluator != NULL &&
     (searchParams.humanSLProfile.initialized || !humanEvaluator->requiresSGFMetadata())
  ) {
    const NNOutput* humanOutput = node.getHumanOutput();
    if(humanOutput != NULL) {
      double weightlessProb;
      double weightfulProb;
      if(isRoot) {
        weightlessProb = searchParams.humanSLRootExploreProbWeightless;
        weightfulProb = searchParams.humanSLRootExploreProbWeightful;
      }
      else if(thread.pla == rootPla) {
        weightlessProb = searchParams.humanSLPlaExploreProbWeightless;
        weightfulProb = searchParams.humanSLPlaExploreProbWeightful;
      }
      else {
        weightlessProb = searchParams.humanSLOppExploreProbWeightless;
        weightfulProb = searchParams.humanSLOppExploreProbWeightful;
      }

      double totalHumanProb = weightlessProb + weightfulProb;
      if(totalHumanProb > 0.0) {
        double r = thread.rand.nextDouble();
        if(r < weightlessProb) {
          useHumanSL = true;
          countEdgeVisit = false;
        }
        else if(r < totalHumanProb) {
          useHumanSL = true;
        }
      }
    }

    // Swap out policy and also recompute policy prob mass visited
    if(useHumanSL) {
      nnOutput = humanOutput;
      policyProbMassVisited = 0.0;
      assert(nnOutput != NULL);
      policyProbs = nnOutput->getPolicyProbsMaybeNoised();
      for(int i = 0; i<childrenCapacity; i++) {
        const SearchChildPointer& childPointer = children[i];
        const SearchNode* child = childPointer.getIfAllocated();
        if(child == NULL)
          break;
        Loc moveLoc = childPointer.getMoveLocRelaxed();
        int movePos = getPos(moveLoc);
        float nnPolicyProb = policyProbs[movePos];
        if(nnPolicyProb < 0)
          continue;
        policyProbMassVisited += nnPolicyProb;
      }
    }
  }

  //Probability mass should not sum to more than 1, giving a generous allowance
  //for floating point error.
  assert(policyProbMassVisited <= 1.0001);

  //If we're doing a weightless visit, then we should redo PUCT to operate on child node weight, not child edge weight
  if(!countEdgeVisit) {
    totalChildWeight = 0.0;
    maxChildWeight = 0.0;
    for(int i = 0; i<childrenCapacity; i++) {
      const SearchChildPointer& childPointer = children[i];
      const SearchNode* child = childPointer.getIfAllocated();
      if(child == NULL)
        break;
      double childWeight = child->stats.weightSum.load(std::memory_order_acquire);
      totalChildWeight += childWeight;
      if(childWeight > maxChildWeight)
        maxChildWeight = childWeight;
    }
  }

  //First play urgency
  double parentUtility;
  double parentWeightPerVisit;
  double parentUtilityStdevFactor;
  double fpuValue = getFpuValueForChildrenAssumeVisited(
    node, thread.pla, isRoot, policyProbMassVisited,
    parentUtility, parentWeightPerVisit, parentUtilityStdevFactor
  );

  bool posesWithChildBuf[NNPos::MAX_NN_POLICY_SIZE] = { }; // Initialize all to false
  bool antiMirror = searchParams.antiMirror && mirroringPla != C_EMPTY && isMirroringSinceSearchStart(thread.history,0);

  double exploreScaling;
  if(useHumanSL)
    exploreScaling = getExploreScalingHuman(totalChildWeight);
  else
    exploreScaling = getExploreScaling(totalChildWeight, parentUtilityStdevFactor);

  //Try all existing children
  //Also count how many children we actually find
  numChildrenFound = 0;
  for(int i = 0; i<childrenCapacity; i++) {
    const SearchChildPointer& childPointer = children[i];
    const SearchNode* child = childPointer.getIfAllocated();
    if(child == NULL)
      break;
    numChildrenFound++;
    int64_t childEdgeVisits = childPointer.getEdgeVisits();

    Loc moveLoc = childPointer.getMoveLocRelaxed();
    bool isDuringSearch = true;
    double selectionValue = getExploreSelectionValueOfChild(
      node,policyProbs,child,
      moveLoc,
      exploreScaling,
      totalChildWeight,childEdgeVisits,fpuValue,
      parentUtility,parentWeightPerVisit,
      isDuringSearch,antiMirror,maxChildWeight,
      countEdgeVisit,
      &thread
    );
    if(selectionValue > maxSelectionValue) {
      // if(child->state.load(std::memory_order_seq_cst) == SearchNode::STATE_EVALUATING) {
      //   selectionValue -= EVALUATING_SELECTION_VALUE_PENALTY;
      //   if(isRoot && child->prevMoveLoc == Location::ofString("K4",thread.board)) {
      //     out << "ouch" << "\n";
      //   }
      // }
      maxSelectionValue = selectionValue;
      bestChildIdx = i;
      bestChildMoveLoc = moveLoc;
    }

    posesWithChildBuf[getPos(moveLoc)] = true;
  }

  const std::vector<int>& avoidMoveUntilByLoc = thread.pla == P_BLACK ? avoidMoveUntilByLocBlack : avoidMoveUntilByLocWhite;

  //Try all the things in the eval cache that are moves we haven't visited yet.
  if(searchParams.useEvalCache && searchParams.useGraphSearch && node.evalCacheEntry != NULL && mirroringPla == C_EMPTY) {
    for(const auto& pair: node.evalCacheEntry->firstExploreEvals) {
      Loc moveLoc = pair.first;
      int movePos = getPos(moveLoc);
      bool alreadyTried = posesWithChildBuf[movePos];
      if(alreadyTried)
        continue;

      //Special logic for the root
      if(isRoot) {
        assert(thread.board.pos_hash == rootBoard.pos_hash);
        assert(thread.pla == rootPla);
        if(!isAllowedRootMove(moveLoc))
          continue;
      }
      if(avoidMoveUntilByLoc.size() > 0) {
        assert(avoidMoveUntilByLoc.size() >= Board::MAX_ARR_SIZE);
        int untilDepth = avoidMoveUntilByLoc[moveLoc];
        if(thread.history.moveHistory.size() - rootHistory.moveHistory.size() < untilDepth)
          continue;
      }

      //Quit immediately for illegal moves
      float nnPolicyProb = policyProbs[movePos];
      if(nnPolicyProb < 0)
        continue;

      FirstExploreEval eval = pair.second;
      double cacheAvgUtility =
        getResultUtility(eval.avgWinLoss,0.0)
        + getScoreUtility(eval.avgScoreMean, eval.avgScoreMean * eval.avgScoreMean);

      double selectionValue = getNewExploreSelectionValue(
        node,
        exploreScaling,
        nnPolicyProb,cacheAvgUtility,
        parentWeightPerVisit,
        maxChildWeight,
        countEdgeVisit,
        &thread
      );
      if(selectionValue > maxSelectionValue) {
        maxSelectionValue = selectionValue;
        bestChildIdx = numChildrenFound;
        bestChildMoveLoc = moveLoc;
      }
    }
  }

  //Try the new child with the best policy value
  Loc bestNewMoveLoc = Board::NULL_LOC;
  float bestNewNNPolicyProb = -1.0f;
  for(int movePos = 0; movePos<policySize; movePos++) {
    bool alreadyTried = posesWithChildBuf[movePos];
    if(alreadyTried)
      continue;

    Loc moveLoc = NNPos::posToLoc(movePos,thread.board.x_size,thread.board.y_size,nnXLen,nnYLen);
    if(moveLoc == Board::NULL_LOC)
      continue;

    //Special logic for the root
    if(isRoot) {
      assert(thread.board.pos_hash == rootBoard.pos_hash);
      assert(thread.pla == rootPla);
      if(!isAllowedRootMove(moveLoc))
        continue;
    }
    if(avoidMoveUntilByLoc.size() > 0) {
      assert(avoidMoveUntilByLoc.size() >= Board::MAX_ARR_SIZE);
      int untilDepth = avoidMoveUntilByLoc[moveLoc];
      if(thread.history.moveHistory.size() - rootHistory.moveHistory.size() < untilDepth)
        continue;
    }

    //Quit immediately for illegal moves
    float nnPolicyProb = policyProbs[movePos];
    if(nnPolicyProb < 0)
      continue;

    if(antiMirror) {
      maybeApplyAntiMirrorPolicy(nnPolicyProb, moveLoc, policyProbs, node.nextPla, &thread);
    }

    if(nnPolicyProb > bestNewNNPolicyProb) {
      bestNewNNPolicyProb = nnPolicyProb;
      bestNewMoveLoc = moveLoc;
    }
  }
  if(bestNewMoveLoc != Board::NULL_LOC) {
    double selectionValue = getNewExploreSelectionValue(
      node,
      exploreScaling,
      bestNewNNPolicyProb,fpuValue,
      parentWeightPerVisit,
      maxChildWeight,
      countEdgeVisit,
      &thread
    );
    if(selectionValue > maxSelectionValue) {
      maxSelectionValue = selectionValue;
      bestChildIdx = numChildrenFound;
      bestChildMoveLoc = bestNewMoveLoc;
    }
  }
}
