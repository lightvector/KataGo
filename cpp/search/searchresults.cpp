
//-------------------------------------------------------------------------------------
//This file contains various functions for extracting stats and results from the search, choosing a move, etc
//-------------------------------------------------------------------------------------

#include "../search/search.h"

#include <cinttypes>

#include "../program/playutils.h"
#include "../search/searchnode.h"

using namespace std;
using nlohmann::json;

int64_t Search::getRootVisits() const {
  if(rootNode == NULL)
    return 0;
  int64_t n = rootNode->stats.visits.load(std::memory_order_acquire);
  return n;
}

bool Search::getPlaySelectionValues(
  vector<Loc>& locs, vector<double>& playSelectionValues, double scaleMaxToAtLeast
) const {
  if(rootNode == NULL) {
    locs.clear();
    playSelectionValues.clear();
    return false;
  }
  const bool allowDirectPolicyMoves = true;
  return getPlaySelectionValues(*rootNode, locs, playSelectionValues, NULL, scaleMaxToAtLeast, allowDirectPolicyMoves);
}

bool Search::getPlaySelectionValues(
  vector<Loc>& locs, vector<double>& playSelectionValues, vector<double>* retVisitCounts, double scaleMaxToAtLeast
) const {
  if(rootNode == NULL) {
    locs.clear();
    playSelectionValues.clear();
    if(retVisitCounts != NULL)
      retVisitCounts->clear();
    return false;
  }
  const bool allowDirectPolicyMoves = true;
  return getPlaySelectionValues(*rootNode, locs, playSelectionValues, retVisitCounts, scaleMaxToAtLeast, allowDirectPolicyMoves);
}

bool Search::getPlaySelectionValues(
  const SearchNode& node,
  vector<Loc>& locs, vector<double>& playSelectionValues, vector<double>* retVisitCounts, double scaleMaxToAtLeast,
  bool allowDirectPolicyMoves
) const {
  double lcbBuf[NNPos::MAX_NN_POLICY_SIZE];
  double radiusBuf[NNPos::MAX_NN_POLICY_SIZE];
  bool result = getPlaySelectionValues(
    node,locs,playSelectionValues,retVisitCounts,scaleMaxToAtLeast,allowDirectPolicyMoves,
    false,false,lcbBuf,radiusBuf
  );
  return result;
}

bool Search::getPlaySelectionValues(
  const SearchNode& node,
  vector<Loc>& locs, vector<double>& playSelectionValues, vector<double>* retVisitCounts, double scaleMaxToAtLeast,
  bool allowDirectPolicyMoves, bool alwaysComputeLcb, bool neverUseLcb,
  //Note: lcbBuf is signed from the player to move's perspective
  double lcbBuf[NNPos::MAX_NN_POLICY_SIZE], double radiusBuf[NNPos::MAX_NN_POLICY_SIZE]
) const {
  locs.clear();
  playSelectionValues.clear();
  if(retVisitCounts != NULL)
    retVisitCounts->clear();

  const NNOutput* nnOutput = node.getNNOutput();
  const float* policyProbs = nnOutput != NULL ? nnOutput->getPolicyProbsMaybeNoised() : NULL;

  double totalChildWeight = 0.0;
  const bool suppressPass = shouldSuppressPass(&node);

  //Store up basic weights
  ConstSearchNodeChildrenReference children = node.getChildren();
  const int childrenCapacity = children.getCapacity();
  for(int i = 0; i<childrenCapacity; i++) {
    const SearchChildPointer& childPointer = children[i];
    const SearchNode* child = childPointer.getIfAllocated();
    if(child == NULL)
      break;
    Loc moveLoc = childPointer.getMoveLocRelaxed();

    int64_t edgeVisits = childPointer.getEdgeVisits();
    double childWeight = child->stats.getChildWeight(edgeVisits);

    locs.push_back(moveLoc);
    totalChildWeight += childWeight;

    // If the move appears to be outright illegal in policy probs, zero out the selection value.
    // Also if we're suppressing passes.
    // We always push a value on to playSelectionValues even if that value is 0,
    // because some callers rely on this to line up with the raw indices in the children array of the node.
    if((suppressPass && moveLoc == Board::PASS_LOC) || policyProbs[getPos(moveLoc)] < 0) {
      playSelectionValues.push_back(0.0);
      if(retVisitCounts != NULL)
        (*retVisitCounts).push_back(0.0);
    }
    else {
      playSelectionValues.push_back((double)childWeight);
      if(retVisitCounts != NULL)
        (*retVisitCounts).push_back((double)edgeVisits);
    }
  }

  int numChildren = (int)playSelectionValues.size();

  //Find the best child before LCB for pruning. Intended to be the most stably explored child.
  //This is the most weighted child, except with a tiny adjustment so that
  //at very low playouts, variable child weights and discretization doesn't do crazy things.
  int nonLCBBestIdx = 0;
  double nonLCBBestChildWeight = -1e30;
  {
    double maxGoodness = -1e30;
    for(int i = 0; i<numChildren; i++) {
      double weight = playSelectionValues[i];
      const SearchChildPointer& childPointer = children[i];
      double edgeVisits = childPointer.getEdgeVisits();
      Loc moveLoc = childPointer.getMoveLocRelaxed();
      double policyProb = policyProbs[getPos(moveLoc)];

      //Small weight on raw policy, and discount one visit's worth of weight since the most recent
      //visit could be overweighted.
      double g = weight * std::max(0.0,edgeVisits-1.0) / std::max(1.0, edgeVisits) + 2.0 * policyProb;
      if(g > maxGoodness) {
        maxGoodness = g;
        nonLCBBestChildWeight = weight;
        nonLCBBestIdx = i;
      }
    }
  }

  //Possibly reduce weight on children that we spend too many visits on in retrospect
  if(&node == rootNode && numChildren > 0) {
    const SearchChildPointer& bestChildPointer = children[nonLCBBestIdx];
    const SearchNode* bestChild = bestChildPointer.getIfAllocated();
    int64_t bestChildEdgeVisits = bestChildPointer.getEdgeVisits();
    Loc bestMoveLoc = bestChildPointer.getMoveLocRelaxed();
    assert(bestChild != NULL);
    const bool isRoot = true;
    const double policyProbMassVisited = 1.0; //doesn't matter, since fpu value computed from it isn't used here
    double parentUtility;
    double parentWeightPerVisit;
    double parentUtilityStdevFactor;
    double fpuValue = getFpuValueForChildrenAssumeVisited(
      node, rootPla, isRoot, policyProbMassVisited,
      parentUtility, parentWeightPerVisit, parentUtilityStdevFactor
    );

    bool isDuringSearch = false;

    double exploreScaling = getExploreScaling(totalChildWeight, parentUtilityStdevFactor);

    assert(nnOutput != NULL);
    const bool countEdgeVisit = true;
    double bestChildExploreSelectionValue = getExploreSelectionValueOfChild(
      node,policyProbs,bestChild,
      bestMoveLoc,
      exploreScaling,
      totalChildWeight,bestChildEdgeVisits,fpuValue,
      parentUtility,parentWeightPerVisit,
      isDuringSearch,false,nonLCBBestChildWeight,
      countEdgeVisit,
      NULL
    );

    for(int i = 0; i<numChildren; i++) {
      const SearchChildPointer& childPointer = children[i];
      const SearchNode* child = childPointer.getIfAllocated();
      Loc moveLoc = childPointer.getMoveLocRelaxed();
      if(suppressPass && moveLoc == Board::PASS_LOC) {
        playSelectionValues[i] = 0;
        continue;
      }
      if(i != nonLCBBestIdx) {
        int64_t edgeVisits = childPointer.getEdgeVisits();
        double reduced = getReducedPlaySelectionWeight(
          node, policyProbs, child,
          moveLoc,
          exploreScaling,
          edgeVisits,
          bestChildExploreSelectionValue
        );
        playSelectionValues[i] = ceil(reduced);
      }
    }
  }

  //Now compute play selection values taking into account LCB
  if(!neverUseLcb && (alwaysComputeLcb || (searchParams.useLcbForSelection && numChildren > 0))) {
    double bestLcb = -1e10;
    int bestLcbIndex = -1;
    for(int i = 0; i<numChildren; i++) {
      const SearchChildPointer& childPointer = children[i];
      const SearchNode* child = childPointer.getIfAllocated();
      int64_t edgeVisits = childPointer.getEdgeVisits();
      Loc moveLoc = childPointer.getMoveLocRelaxed();
      getSelfUtilityLCBAndRadius(node,child,edgeVisits,moveLoc,lcbBuf[i],radiusBuf[i]);
      //Check if this node is eligible to be considered for best LCB
      double weight = playSelectionValues[i];
      if(weight > 0 && weight >= searchParams.minVisitPropForLCB * nonLCBBestChildWeight) {
        if(lcbBuf[i] > bestLcb) {
          bestLcb = lcbBuf[i];
          bestLcbIndex = i;
        }
      }
    }

    if(searchParams.useLcbForSelection && numChildren > 0 && (searchParams.useNonBuggyLcb ? (bestLcbIndex >= 0) : (bestLcbIndex > 0))) {
      //Best LCB move gets a bonus that ensures it is large enough relative to every other child
      double adjustedWeight = playSelectionValues[bestLcbIndex];
      for(int i = 0; i<numChildren; i++) {
        if(i != bestLcbIndex) {
          double excessValue = bestLcb - lcbBuf[i];
          //This move is actually worse lcb than some other move, it's just that the other
          //move failed its checks for having enough minimum weight. So don't actually
          //try to compute how much better this one is than that one, because it's not better.
          if(excessValue < 0)
            continue;

          double radius = radiusBuf[i];
          //How many times wider would the radius have to be before the lcb would be worse?
          //Add adjust the denom so that we cannot possibly gain more than a factor of 5, just as a guard
          double radiusFactor = (radius + excessValue) / (radius + 0.20 * excessValue);

          //That factor, squared, is the number of "weight" more that we should pretend we have, for
          //the purpose of selection, since normally stdev is proportional to 1/weight^2.
          double lbound = radiusFactor * radiusFactor * playSelectionValues[i];
          if(lbound > adjustedWeight)
            adjustedWeight = lbound;
        }
      }
      playSelectionValues[bestLcbIndex] = adjustedWeight;
    }
  }

  auto isOkayRawPolicyMoveAtRoot = [&](Loc moveLoc, double policyProb, bool obeyAllowedRootMove) {
    if(!rootHistory.isLegal(rootBoard,moveLoc,rootPla) || policyProb < 0 || (obeyAllowedRootMove && !isAllowedRootMove(moveLoc)))
      return false;
    const std::vector<int>& avoidMoveUntilByLoc = rootPla == P_BLACK ? avoidMoveUntilByLocBlack : avoidMoveUntilByLocWhite;
    if(avoidMoveUntilByLoc.size() > 0) {
      assert(avoidMoveUntilByLoc.size() >= Board::MAX_ARR_SIZE);
      int untilDepth = avoidMoveUntilByLoc[moveLoc];
      if(untilDepth > 0)
        return false;
    }
    return true;
  };

  //If we have no children, then use the policy net directly. Only for the root, though, if calling this on any subtree
  //then just require that we have children, for implementation simplicity (since it requires that we have a board and a boardhistory too)
  //(and we also use isAllowedRootMove and avoidMoveUntilByLoc)
  if(numChildren == 0) {
    if(nnOutput == NULL || &node != rootNode || !allowDirectPolicyMoves)
      return false;

    bool obeyAllowedRootMove = true;
    while(true) {
      for(int movePos = 0; movePos<policySize; movePos++) {
        Loc moveLoc = NNPos::posToLoc(movePos,rootBoard.x_size,rootBoard.y_size,nnXLen,nnYLen);
        double policyProb = policyProbs[movePos];
        if(!isOkayRawPolicyMoveAtRoot(moveLoc,policyProb,obeyAllowedRootMove))
          continue;
        if(suppressPass && moveLoc == Board::PASS_LOC)
          policyProb = 0.0;
        locs.push_back(moveLoc);
        playSelectionValues.push_back(policyProb);
        numChildren++;
      }
      //Still no children? Then at this point just ignore isAllowedRootMove.
      if(numChildren == 0 && obeyAllowedRootMove) {
        obeyAllowedRootMove = false;
        continue;
      }
      break;
    }
  }

  //Might happen absurdly rarely if we both have no children and don't properly have an nnOutput
  //but have a hash collision or something so we "found" an nnOutput anyways.
  //Could also happen if we have avoidMoveUntilByLoc pruning all the allowed moves.
  if(numChildren == 0)
    return false;

  double maxValue = 0.0;
  for(int i = 0; i<numChildren; i++) {
    if(playSelectionValues[i] > maxValue)
      maxValue = playSelectionValues[i];
  }

  if(maxValue <= 1e-50) {
    //If we reached this point we have nonzero many children but the children are all weightless.
    //In that case, at least set each one to be weighted by its policy.
    for(int i = 0; i<numChildren; i++) {
      playSelectionValues[i] = std::max(0.0,(double)policyProbs[getPos(locs[i])]);
    }
    //Recompute max
    for(int i = 0; i<numChildren; i++) {
      if(playSelectionValues[i] > maxValue)
        maxValue = playSelectionValues[i];
    }
    if(maxValue <= 1e-50) {
      return false;
    }
  }

  //Sanity check - if somehow we had more than this, something must have overflowed or gone wrong
  assert(maxValue < 1e40);

  double amountToSubtract = std::min(searchParams.chosenMoveSubtract, maxValue/64.0);
  double amountToPrune = std::min(searchParams.chosenMovePrune, maxValue/64.0);
  for(int i = 0; i<numChildren; i++) {
    if(playSelectionValues[i] < amountToPrune)
      playSelectionValues[i] = 0.0;
    else {
      playSelectionValues[i] -= amountToSubtract;
      if(playSelectionValues[i] <= 0.0)
        playSelectionValues[i] = 0.0;
    }
  }

  // Average in human policy
  if(humanEvaluator != NULL &&
     (searchParams.humanSLProfile.initialized || !humanEvaluator->requiresSGFMetadata()) &&
     searchParams.humanSLChosenMoveProp > 0.0
  ) {
    const NNOutput* humanOutput = node.getHumanOutput();
    const float* humanProbs = humanOutput != NULL ? humanOutput->getPolicyProbsMaybeNoised() : NULL;
    if(humanProbs != NULL) {
      // First, take a pass to just fill out all the legal/allowed moves into the play selection values, if allowed, and if at root.
      if(&node == rootNode && allowDirectPolicyMoves) {
        std::set<Loc> locsSet(locs.begin(),locs.end());
        for(int movePos = 0; movePos<policySize; movePos++) {
          Loc moveLoc = NNPos::posToLoc(movePos,rootBoard.x_size,rootBoard.y_size,nnXLen,nnYLen);
          double humanProb = humanProbs[movePos];
          const bool obeyAllowedRootMove = true;
          if(!isOkayRawPolicyMoveAtRoot(moveLoc,humanProb,obeyAllowedRootMove))
            continue;
          if(contains(locsSet,moveLoc))
            continue;
          locs.push_back(moveLoc);
          locsSet.insert(moveLoc);
          playSelectionValues.push_back(0.0); // Pushing zeros since we're just filling in
          numChildren++;
        }
      }

      // Grab utility on the moves we have utilities for.
      std::map<Loc,double> shiftedPolicy;
      std::map<Loc,double> selfUtilities;
      double selfUtilityMax = -1e10;
      double selfUtilitySum = 0.0;
      for(int i = 0; i<childrenCapacity; i++) {
        const SearchChildPointer& childPointer = children[i];
        const SearchNode* child = childPointer.getIfAllocated();
        if(child == NULL)
          break;
        Loc moveLoc = childPointer.getMoveLocRelaxed();
        double humanProb = humanProbs[getPos(moveLoc)];
        if((suppressPass && moveLoc == Board::PASS_LOC) || humanProb < 0)
          humanProb = 0.0;

        shiftedPolicy[moveLoc] = humanProb;
        selfUtilities[moveLoc] = (rootPla == P_WHITE ? 1 : -1) * child->stats.utilityAvg.load(std::memory_order_acquire);
        selfUtilityMax = std::max(selfUtilityMax, selfUtilities[moveLoc]);
        selfUtilitySum += selfUtilities[moveLoc];
      }
      // Straight linear average. Use this to complete the remaining utilities, i.e. for fpu
      double selfUtilityAvg = selfUtilitySum / std::max((size_t)1, selfUtilities.size());
      selfUtilityMax = std::max(selfUtilityMax, selfUtilityAvg); // In case of 0 size
      for(Loc loc: locs) {
        if(!contains(shiftedPolicy,loc)) {
          double humanProb = humanProbs[getPos(loc)];
          if((suppressPass && loc == Board::PASS_LOC) || humanProb < 0)
            humanProb = 0.0;
          shiftedPolicy[loc] = humanProb;
          selfUtilities[loc] = selfUtilityAvg;
        }
      }
      // Perform shift
      for(Loc loc: locs)
        shiftedPolicy[loc] *= exp((selfUtilities[loc] - selfUtilityMax)/searchParams.humanSLChosenMovePiklLambda);

      double shiftedPolicySum = 0.0;
      for(Loc loc: locs)
        shiftedPolicySum += shiftedPolicy[loc];

      // Renormalize and average in to current play selection values, scaling up to the current sum scale of playSelectionValues.
      if(shiftedPolicySum > 0.0) {
        for(Loc loc: locs)
          shiftedPolicy[loc] /= shiftedPolicySum;

        double playSelectionValueSum = 0.0;
        double playSelectionValueNonPassSum = 0.0;
        for(int i = 0; i<numChildren; i++) {
          playSelectionValueSum += playSelectionValues[i];
          if(locs[i] != Board::PASS_LOC)
            playSelectionValueNonPassSum += playSelectionValues[i];
        }

        if(searchParams.humanSLChosenMoveIgnorePass) {
          double shiftedPolicyNonPassSum = 0.0;
          for(Loc loc: locs) {
            if(loc != Board::PASS_LOC)
              shiftedPolicyNonPassSum += shiftedPolicy[loc];
          }
          if(shiftedPolicyNonPassSum > 0.0) {
            for(Loc loc: locs) {
              if(loc != Board::PASS_LOC)
                shiftedPolicy[loc] = shiftedPolicy[loc] / shiftedPolicyNonPassSum * playSelectionValueNonPassSum / playSelectionValueSum;
              else
                shiftedPolicy[loc] = (playSelectionValueSum - playSelectionValueNonPassSum) / playSelectionValueSum;
            }
          }
        }

        for(int i = 0; i<numChildren; i++) {
          playSelectionValues[i] += searchParams.humanSLChosenMoveProp * (playSelectionValueSum * shiftedPolicy[locs[i]] - playSelectionValues[i]);
        }
      }
    }
  }

  maxValue = 0.0;
  for(int i = 0; i<numChildren; i++) {
    if(playSelectionValues[i] > maxValue)
      maxValue = playSelectionValues[i];
  }
  assert(maxValue > 0.0);
  if(maxValue < scaleMaxToAtLeast) {
    for(int i = 0; i<numChildren; i++) {
      playSelectionValues[i] *= scaleMaxToAtLeast / maxValue;
    }
  }

  return true;
}


bool Search::getRootValues(ReportedSearchValues& values) const {
  return getNodeValues(rootNode,values);
}

ReportedSearchValues Search::getRootValuesRequireSuccess() const {
  ReportedSearchValues values;
  if(rootNode == NULL)
    throw StringError("Bug? Bot search root was null");
  bool success = getNodeValues(rootNode,values);
  if(!success)
    throw StringError("Bug? Bot search returned no root values");
  return values;
}

bool Search::getRootRawNNValues(ReportedSearchValues& values) const {
  if(rootNode == NULL)
    return false;
  return getNodeRawNNValues(*rootNode,values);
}

ReportedSearchValues Search::getRootRawNNValuesRequireSuccess() const {
  ReportedSearchValues values;
  if(rootNode == NULL)
    throw StringError("Bug? Bot search root was null");
  bool success = getNodeRawNNValues(*rootNode,values);
  if(!success)
    throw StringError("Bug? Bot search returned no root values");
  return values;
}

bool Search::getNodeRawNNValues(const SearchNode& node, ReportedSearchValues& values) const {
  const NNOutput* nnOutput = node.getNNOutput();
  if(nnOutput == NULL)
    return false;

  values.winValue = nnOutput->whiteWinProb;
  values.lossValue = nnOutput->whiteLossProb;
  values.noResultValue = nnOutput->whiteNoResultProb;

  double scoreMean = nnOutput->whiteScoreMean;
  double scoreMeanSq = nnOutput->whiteScoreMeanSq;
  double scoreStdev = ScoreValue::getScoreStdev(scoreMean,scoreMeanSq);
  double sqrtBoardArea = rootBoard.sqrtBoardArea();
  values.staticScoreValue = ScoreValue::expectedWhiteScoreValue(scoreMean,scoreStdev,0.0,2.0, sqrtBoardArea);
  values.dynamicScoreValue = ScoreValue::expectedWhiteScoreValue(scoreMean,scoreStdev,recentScoreCenter,searchParams.dynamicScoreCenterScale, sqrtBoardArea);
  values.expectedScore = scoreMean;
  values.expectedScoreStdev = scoreStdev;
  values.lead = nnOutput->whiteLead;

  //Sanity check
  assert(values.winValue >= 0.0);
  assert(values.lossValue >= 0.0);
  assert(values.noResultValue >= 0.0);
  assert(values.winValue + values.lossValue + values.noResultValue < 1.001);

  double winLossValue = values.winValue - values.lossValue;
  if(winLossValue > 1.0) winLossValue = 1.0;
  if(winLossValue < -1.0) winLossValue = -1.0;
  values.winLossValue = winLossValue;

  values.weight = computeWeightFromNNOutput(nnOutput);
  values.visits = 1;

  return true;
}


bool Search::getNodeValues(const SearchNode* node, ReportedSearchValues& values) const {
  if(node == NULL)
    return false;
  int64_t visits = node->stats.visits.load(std::memory_order_acquire);
  double weightSum = node->stats.weightSum.load(std::memory_order_acquire);
  double winLossValueAvg = node->stats.winLossValueAvg.load(std::memory_order_acquire);
  double noResultValueAvg = node->stats.noResultValueAvg.load(std::memory_order_acquire);
  double scoreMeanAvg = node->stats.scoreMeanAvg.load(std::memory_order_acquire);
  double scoreMeanSqAvg = node->stats.scoreMeanSqAvg.load(std::memory_order_acquire);
  double leadAvg = node->stats.leadAvg.load(std::memory_order_acquire);
  double utilityAvg = node->stats.utilityAvg.load(std::memory_order_acquire);

  if(weightSum <= 0.0)
    return false;
  assert(visits >= 0);
  if(node == rootNode) {
    //For terminal nodes, we may have no nnoutput and yet we have legitimate visits and terminal evals.
    //But for the root, the root is never treated as a terminal node and always gets an nneval, so if
    //it has visits and weight, it has an nnoutput unless something has gone wrong.
    const NNOutput* nnOutput = node->getNNOutput();
    assert(nnOutput != NULL);
    (void)nnOutput;
  }

  values = ReportedSearchValues(
    *this,
    winLossValueAvg,
    noResultValueAvg,
    scoreMeanAvg,
    scoreMeanSqAvg,
    leadAvg,
    utilityAvg,
    weightSum,
    visits
  );
  return true;
}

const SearchNode* Search::getRootNode() const {
  return rootNode;
}
const SearchNode* Search::getChildForMove(const SearchNode* node, Loc moveLoc) const {
  if(node == NULL)
    return NULL;
  ConstSearchNodeChildrenReference children = node->getChildren();
  int childrenCapacity = children.getCapacity();
  for(int i = 0; i<childrenCapacity; i++) {
    const SearchChildPointer& childPointer = children[i];
    const SearchNode* child = childPointer.getIfAllocated();
    if(child == NULL)
      break;
    Loc childMoveLoc = childPointer.getMoveLocRelaxed();
    if(moveLoc == childMoveLoc)
      return child;
  }
  return NULL;
}

Loc Search::getChosenMoveLoc() {
  if(rootNode == NULL)
    return Board::NULL_LOC;

  vector<Loc> locs;
  vector<double> playSelectionValues;
  bool suc = getPlaySelectionValues(locs,playSelectionValues,0.0);
  if(!suc)
    return Board::NULL_LOC;

  assert(locs.size() == playSelectionValues.size());

  double temperature = interpolateEarly(
    searchParams.chosenMoveTemperatureHalflife, searchParams.chosenMoveTemperatureEarly, searchParams.chosenMoveTemperature
  );

  uint32_t idxChosen = chooseIndexWithTemperature(
    nonSearchRand,
    playSelectionValues.data(),
    (int)playSelectionValues.size(),
    temperature,
    searchParams.chosenMoveTemperatureOnlyBelowProb,
    NULL
  );
  return locs[idxChosen];
}


bool Search::getPolicy(float policyProbs[NNPos::MAX_NN_POLICY_SIZE]) const {
  return getPolicy(rootNode, policyProbs);
}
bool Search::getPolicy(const SearchNode* node, float policyProbs[NNPos::MAX_NN_POLICY_SIZE]) const {
  if(node == NULL)
    return false;
  const NNOutput* nnOutput = node->getNNOutput();
  if(nnOutput == NULL)
    return false;

  std::copy(nnOutput->policyProbs, nnOutput->policyProbs+NNPos::MAX_NN_POLICY_SIZE, policyProbs);
  return true;
}


//Safe to call concurrently with search
double Search::getPolicySurprise() const {
  double surprise = 0.0;
  double searchEntropy = 0.0;
  double policyEntropy = 0.0;
  if(getPolicySurpriseAndEntropy(surprise,searchEntropy,policyEntropy))
    return surprise;
  return 0.0;
}

bool Search::getPolicySurpriseAndEntropy(double& surpriseRet, double& searchEntropyRet, double& policyEntropyRet) const {
  return getPolicySurpriseAndEntropy(surpriseRet, searchEntropyRet, policyEntropyRet, rootNode);
}

//Safe to call concurrently with search
bool Search::getPolicySurpriseAndEntropy(double& surpriseRet, double& searchEntropyRet, double& policyEntropyRet, const SearchNode* node) const {
  if(node == NULL)
    return false;
  const NNOutput* nnOutput = node->getNNOutput();
  if(nnOutput == NULL)
    return false;

  vector<Loc> locs;
  vector<double> playSelectionValues;
  const bool allowDirectPolicyMoves = true;
  const bool alwaysComputeLcb = false;
  double lcbBuf[NNPos::MAX_NN_POLICY_SIZE];
  double radiusBuf[NNPos::MAX_NN_POLICY_SIZE];
  bool suc = getPlaySelectionValues(
    *node,locs,playSelectionValues,NULL,1.0,allowDirectPolicyMoves,alwaysComputeLcb,false,lcbBuf,radiusBuf
  );
  if(!suc)
    return false;

  float policyProbsFromNNBuf[NNPos::MAX_NN_POLICY_SIZE];
  {
    const float* policyProbsFromNN = nnOutput->getPolicyProbsMaybeNoised();
    std::copy(policyProbsFromNN, policyProbsFromNN+NNPos::MAX_NN_POLICY_SIZE, policyProbsFromNNBuf);
  }

  double sumPlaySelectionValues = 0.0;
  for(size_t i = 0; i < playSelectionValues.size(); i++)
    sumPlaySelectionValues += playSelectionValues[i];

  double surprise = 0.0;
  double searchEntropy = 0.0;
  for(size_t i = 0; i < playSelectionValues.size(); i++) {
    int pos = getPos(locs[i]);
    double policy = std::max((double)policyProbsFromNNBuf[pos],1e-100);
    double target = playSelectionValues[i] / sumPlaySelectionValues;
    if(target > 1e-100) {
      double logTarget = log(target);
      double logPolicy = log(policy);
      surprise += target * (logTarget - logPolicy);
      searchEntropy += -target * logTarget;
    }
  }

  double policyEntropy = 0.0;
  for(int pos = 0; pos<NNPos::MAX_NN_POLICY_SIZE; pos++) {
    double policy = policyProbsFromNNBuf[pos];
    if(policy > 1e-100) {
      policyEntropy += -policy * log(policy);
    }
  }

  //Just in case, guard against float imprecision
  if(surprise < 0.0)
    surprise = 0.0;
  if(searchEntropy < 0.0)
    searchEntropy = 0.0;
  if(policyEntropy < 0.0)
    policyEntropy = 0.0;

  surpriseRet = surprise;
  searchEntropyRet = searchEntropy;
  policyEntropyRet = policyEntropy;

  return true;
}

void Search::printRootOwnershipMap(ostream& out, Player perspective) const {
  if(rootNode == NULL)
    return;
  const NNOutput* nnOutput = rootNode->getNNOutput();
  if(nnOutput == NULL)
    return;
  if(nnOutput->whiteOwnerMap == NULL)
    return;

  Player perspectiveToUse = (perspective != P_BLACK && perspective != P_WHITE) ? rootPla : perspective;
  double perspectiveFactor = perspectiveToUse == P_BLACK ? -1.0 : 1.0;

  for(int y = 0; y<rootBoard.y_size; y++) {
    for(int x = 0; x<rootBoard.x_size; x++) {
      int pos = NNPos::xyToPos(x,y,nnOutput->nnXLen);
      out << Global::strprintf("%6.1f ", perspectiveFactor * nnOutput->whiteOwnerMap[pos]*100);
    }
    out << endl;
  }
  out << endl;
}

void Search::printRootPolicyMap(ostream& out) const {
  if(rootNode == NULL)
    return;
  const NNOutput* nnOutput = rootNode->getNNOutput();
  if(nnOutput == NULL)
    return;

  const float* policyProbs = nnOutput->getPolicyProbsMaybeNoised();
  for(int y = 0; y<rootBoard.y_size; y++) {
    for(int x = 0; x<rootBoard.x_size; x++) {
      int pos = NNPos::xyToPos(x,y,nnOutput->nnXLen);
      out << Global::strprintf("%6.1f ", policyProbs[pos]*100);
    }
    out << endl;
  }
  out << endl;
}

void Search::printRootEndingScoreValueBonus(ostream& out) const {
  if(rootNode == NULL)
    return;
  const NNOutput* nnOutput = rootNode->getNNOutput();
  if(nnOutput == NULL)
    return;
  if(nnOutput->whiteOwnerMap == NULL)
    return;

  ConstSearchNodeChildrenReference children = rootNode->getChildren();
  int childrenCapacity = children.getCapacity();
  for(int i = 0; i<childrenCapacity; i++) {
    const SearchNode* child = children[i].getIfAllocated();
    if(child == NULL)
      break;

    int64_t edgeVisits = children[i].getEdgeVisits();
    Loc moveLoc = children[i].getMoveLocRelaxed();
    int64_t childVisits = child->stats.visits.load(std::memory_order_acquire);
    double scoreMeanAvg = child->stats.scoreMeanAvg.load(std::memory_order_acquire);
    double scoreMeanSqAvg = child->stats.scoreMeanSqAvg.load(std::memory_order_acquire);
    double utilityAvg = child->stats.utilityAvg.load(std::memory_order_acquire);

    double utilityNoBonus = utilityAvg;
    double endingScoreBonus = getEndingWhiteScoreBonus(*rootNode,moveLoc);
    double utilityDiff = getScoreUtilityDiff(scoreMeanAvg, scoreMeanSqAvg, endingScoreBonus);
    double utilityWithBonus = utilityNoBonus + utilityDiff;

    out << Location::toString(moveLoc,rootBoard) << " " << Global::strprintf(
      "visits %d edgeVisits %d utilityNoBonus %.2fc utilityWithBonus %.2fc endingScoreBonus %.2f",
      childVisits, edgeVisits, utilityNoBonus*100, utilityWithBonus*100, endingScoreBonus
    );
    out << endl;
  }
}

void Search::appendPV(
  vector<Loc>& buf,
  vector<int64_t>& visitsBuf,
  vector<int64_t>& edgeVisitsBuf,
  vector<Loc>& scratchLocs,
  vector<double>& scratchValues,
  const SearchNode* node,
  int maxDepth
) const {
  appendPVForMove(buf,visitsBuf,edgeVisitsBuf,scratchLocs,scratchValues,node,Board::NULL_LOC,maxDepth);
}

void Search::appendPVForMove(
  vector<Loc>& buf,
  vector<int64_t>& visitsBuf,
  vector<int64_t>& edgeVisitsBuf,
  vector<Loc>& scratchLocs,
  vector<double>& scratchValues,
  const SearchNode* node,
  Loc move,
  int maxDepth
) const {
  if(node == NULL)
    return;

  for(int depth = 0; depth < maxDepth; depth++) {
    const bool allowDirectPolicyMoves = true;
    bool success = getPlaySelectionValues(*node, scratchLocs, scratchValues, NULL, 1.0, allowDirectPolicyMoves);
    if(!success)
      return;

    double maxSelectionValue = POLICY_ILLEGAL_SELECTION_VALUE;
    int bestChildIdx = -1;
    Loc bestChildMoveLoc = Board::NULL_LOC;

    for(int i = 0; i<scratchValues.size(); i++) {
      Loc moveLoc = scratchLocs[i];
      double selectionValue = scratchValues[i];

      if(depth == 0 && moveLoc == move) {
        maxSelectionValue = selectionValue;
        bestChildIdx = i;
        bestChildMoveLoc = moveLoc;
        break;
      }

      if(selectionValue > maxSelectionValue) {
        maxSelectionValue = selectionValue;
        bestChildIdx = i;
        bestChildMoveLoc = moveLoc;
      }
    }

    if(bestChildIdx < 0 || bestChildMoveLoc == Board::NULL_LOC)
      return;
    if(depth == 0 && move != Board::NULL_LOC && bestChildMoveLoc != move)
      return;

    ConstSearchNodeChildrenReference children = node->getChildren();
    int childrenCapacity = children.getCapacity();
    //Direct policy move
    if(bestChildIdx >= childrenCapacity) {
      buf.push_back(bestChildMoveLoc);
      visitsBuf.push_back(0);
      edgeVisitsBuf.push_back(0);
      return;
    }
    const SearchNode* child = children[bestChildIdx].getIfAllocated();
    //Direct policy move
    if(child == NULL) {
      buf.push_back(bestChildMoveLoc);
      visitsBuf.push_back(0);
      edgeVisitsBuf.push_back(0);
      return;
    }

    node = child;

    int64_t visits = node->stats.visits.load(std::memory_order_acquire);
    int64_t edgeVisits = children[bestChildIdx].getEdgeVisits();

    buf.push_back(bestChildMoveLoc);
    visitsBuf.push_back(visits);
    edgeVisitsBuf.push_back(edgeVisits);
  }
}


void Search::printPV(ostream& out, const SearchNode* n, int maxDepth) const {
  vector<Loc> buf;
  vector<int64_t> visitsBuf;
  vector<int64_t> edgeVisitsBuf;
  vector<Loc> scratchLocs;
  vector<double> scratchValues;
  appendPV(buf,visitsBuf,edgeVisitsBuf,scratchLocs,scratchValues,n,maxDepth);
  printPV(out,buf);
}

void Search::printPV(ostream& out, const vector<Loc>& buf) const {
  bool printedAnything = false;
  for(int i = 0; i<buf.size(); i++) {
    if(printedAnything)
      out << " ";
    if(buf[i] == Board::NULL_LOC)
      continue;
    out << Location::toString(buf[i],rootBoard);
    printedAnything = true;
  }
}

//Child should NOT be locked.
AnalysisData Search::getAnalysisDataOfSingleChild(
  const SearchNode* child, int64_t edgeVisits, vector<Loc>& scratchLocs, vector<double>& scratchValues,
  Loc move, double policyProb, double fpuValue, double parentUtility, double parentWinLossValue,
  double parentScoreMean, double parentScoreStdev, double parentLead, int maxPVDepth
) const {
  int64_t childVisits = 0;
  double winLossValueAvg = 0.0;
  double noResultValueAvg = 0.0;
  double scoreMeanAvg = 0.0;
  double scoreMeanSqAvg = 0.0;
  double leadAvg = 0.0;
  double utilityAvg = 0.0;
  double utilitySqAvg = 0.0;
  double weightSum = 0.0;
  double weightSqSum = 0.0;
  double childWeightSum = 0.0;

  if(child != NULL) {
    childVisits = child->stats.visits.load(std::memory_order_acquire);
    winLossValueAvg = child->stats.winLossValueAvg.load(std::memory_order_acquire);
    noResultValueAvg = child->stats.noResultValueAvg.load(std::memory_order_acquire);
    scoreMeanAvg = child->stats.scoreMeanAvg.load(std::memory_order_acquire);
    scoreMeanSqAvg = child->stats.scoreMeanSqAvg.load(std::memory_order_acquire);
    leadAvg = child->stats.leadAvg.load(std::memory_order_acquire);
    utilityAvg = child->stats.utilityAvg.load(std::memory_order_acquire);
    utilitySqAvg = child->stats.utilitySqAvg.load(std::memory_order_acquire);
    weightSum = child->stats.getChildWeight(edgeVisits,childVisits);
    weightSqSum = child->stats.getChildWeightSq(edgeVisits,childVisits);
    childWeightSum = child->stats.weightSum.load(std::memory_order_acquire);
  }

  AnalysisData data;
  data.move = move;
  data.numVisits = edgeVisits;
  if(childVisits <= 0 || childWeightSum <= 1e-30) {
    data.utility = fpuValue;
    data.scoreUtility = getScoreUtility(parentScoreMean,parentScoreMean*parentScoreMean+parentScoreStdev*parentScoreStdev);
    data.resultUtility = fpuValue - data.scoreUtility;
    data.winLossValue = searchParams.winLossUtilityFactor == 1.0 ? parentWinLossValue + (fpuValue - parentUtility) : 0.0;
    // Make sure winloss values due to FPU don't go out of bounds for purposes of reporting to UI
    if(data.winLossValue < -1.0)
      data.winLossValue = -1.0;
    if(data.winLossValue > 1.0)
      data.winLossValue = 1.0;
    data.scoreMean = parentScoreMean;
    data.scoreStdev = parentScoreStdev;
    data.lead = parentLead;
    data.ess = 0.0;
    data.weightSum = 0.0;
    data.weightSqSum = 0.0;
    data.utilitySqAvg = data.utility * data.utility;
    data.scoreMeanSqAvg = parentScoreMean * parentScoreMean + parentScoreStdev * parentScoreStdev;
    data.childVisits = childVisits;
    data.childWeightSum = childWeightSum;
  }
  else {
    data.utility = utilityAvg;
    data.resultUtility = getResultUtility(winLossValueAvg, noResultValueAvg);
    data.scoreUtility = getScoreUtility(scoreMeanAvg, scoreMeanSqAvg);
    data.winLossValue = winLossValueAvg;
    data.scoreMean = scoreMeanAvg;
    data.scoreStdev = ScoreValue::getScoreStdev(scoreMeanAvg,scoreMeanSqAvg);
    data.lead = leadAvg;
    data.ess = weightSum * weightSum / std::max(1e-8,weightSqSum);
    data.weightSum = weightSum;
    data.weightSqSum = weightSqSum;
    data.utilitySqAvg = utilitySqAvg;
    data.scoreMeanSqAvg = scoreMeanSqAvg;
    data.childVisits = childVisits;
    data.childWeightSum = childWeightSum;
  }

  data.policyPrior = policyProb;
  data.order = 0;

  data.pv.clear();
  data.pv.push_back(move);
  data.pvVisits.clear();
  data.pvVisits.push_back(childVisits);
  data.pvEdgeVisits.clear();
  data.pvEdgeVisits.push_back(edgeVisits);
  appendPV(data.pv, data.pvVisits, data.pvEdgeVisits, scratchLocs, scratchValues, child, maxPVDepth);

  data.node = child;

  return data;
}

void Search::getAnalysisData(
  vector<AnalysisData>& buf,int minMovesToTryToGet, bool includeWeightFactors, int maxPVDepth, bool duplicateForSymmetries
) const {
  buf.clear();
  if(rootNode == NULL)
    return;
  getAnalysisData(*rootNode, buf, minMovesToTryToGet, includeWeightFactors, maxPVDepth, duplicateForSymmetries);
}

void Search::getAnalysisData(
  const SearchNode& node, vector<AnalysisData>& buf, int minMovesToTryToGet, bool includeWeightFactors, int maxPVDepth, bool duplicateForSymmetries
) const {
  buf.clear();
  vector<const SearchNode*> children;
  vector<int64_t> childrenEdgeVisits;
  vector<Loc> childrenMoveLocs;
  children.reserve(rootBoard.x_size * rootBoard.y_size + 1);
  childrenEdgeVisits.reserve(rootBoard.x_size * rootBoard.y_size + 1);
  childrenMoveLocs.reserve(rootBoard.x_size * rootBoard.y_size + 1);

  int numChildren;
  vector<Loc> scratchLocs;
  vector<double> scratchValues;
  double lcbBuf[NNPos::MAX_NN_POLICY_SIZE];
  double radiusBuf[NNPos::MAX_NN_POLICY_SIZE];
  float policyProbs[NNPos::MAX_NN_POLICY_SIZE];
  {
    ConstSearchNodeChildrenReference childrenArr = node.getChildren();
    int childrenCapacity = childrenArr.getCapacity();
    for(int i = 0; i<childrenCapacity; i++) {
      const SearchChildPointer& childPointer = childrenArr[i];
      const SearchNode* child = childPointer.getIfAllocated();
      if(child == NULL)
        break;
      children.push_back(child);
      childrenEdgeVisits.push_back(childPointer.getEdgeVisits());
      childrenMoveLocs.push_back(childPointer.getMoveLocRelaxed());
    }
    numChildren = (int)children.size();

    if(numChildren <= 0)
      return;
    assert(numChildren <= NNPos::MAX_NN_POLICY_SIZE);

    const bool alwaysComputeLcb = true;
    bool gotPlaySelectionValues = getPlaySelectionValues(node, scratchLocs, scratchValues, NULL, 1.0, false, alwaysComputeLcb, false, lcbBuf, radiusBuf);

    // No play selection values - then fill with values consistent with all 0 visits.
    // We want it to be possible to get analysis data even when all visits are weightless.
    if(!gotPlaySelectionValues) {
      for(int i = 0; i<numChildren; i++) {
        scratchLocs.push_back(childrenMoveLocs[i]);
        scratchValues.push_back(0.0);
      }
      double lcbBufValue;
      double radiusBufValue;
      getSelfUtilityLCBAndRadiusZeroVisits(lcbBufValue,radiusBufValue);
      std::fill(lcbBuf,lcbBuf+numChildren,lcbBufValue);
      std::fill(radiusBuf,radiusBuf+numChildren,radiusBufValue);
    }

    const NNOutput* nnOutput = node.getNNOutput();
    const float* policyProbsFromNN = nnOutput->getPolicyProbsMaybeNoised();
    for(int i = 0; i<NNPos::MAX_NN_POLICY_SIZE; i++)
      policyProbs[i] = policyProbsFromNN[i];
  }

  //Copy to make sure we keep these values so we can reuse scratch later for PV
  vector<double> playSelectionValues = scratchValues;

  double policyProbMassVisited = 0.0;
  {
    for(int i = 0; i<numChildren; i++) {
      policyProbMassVisited += std::max(0.0, (double)policyProbs[getPos(childrenMoveLocs[i])]);
    }
    //Probability mass should not sum to more than 1, giving a generous allowance
    //for floating point error.
    assert(policyProbMassVisited <= 1.0001);
  }

  double parentWinLossValue;
  double parentScoreMean;
  double parentScoreStdev;
  double parentLead;
  {
    double weightSum = node.stats.weightSum.load(std::memory_order_acquire);
    double winLossValueAvg = node.stats.winLossValueAvg.load(std::memory_order_acquire);
    double scoreMeanAvg = node.stats.scoreMeanAvg.load(std::memory_order_acquire);
    double scoreMeanSqAvg = node.stats.scoreMeanSqAvg.load(std::memory_order_acquire);
    double leadAvg = node.stats.leadAvg.load(std::memory_order_acquire);
    assert(weightSum > 0.0);

    parentWinLossValue = winLossValueAvg;
    parentScoreMean = scoreMeanAvg;
    parentScoreStdev = ScoreValue::getScoreStdev(parentScoreMean,scoreMeanSqAvg);
    parentLead = leadAvg;
  }

  double parentUtility;
  double parentWeightPerVisit;
  double parentUtilityStdevFactor;
  double fpuValue = getFpuValueForChildrenAssumeVisited(
    node, node.nextPla, true, policyProbMassVisited,
    parentUtility, parentWeightPerVisit, parentUtilityStdevFactor
  );

  vector<MoreNodeStats> statsBuf(numChildren);
  for(int i = 0; i<numChildren; i++) {
    const SearchNode* child = children[i];
    int64_t edgeVisits = childrenEdgeVisits[i];
    Loc moveLoc = childrenMoveLocs[i];
    double policyProb = policyProbs[getPos(moveLoc)];
    AnalysisData data = getAnalysisDataOfSingleChild(
      child, edgeVisits, scratchLocs, scratchValues, moveLoc, policyProb, fpuValue, parentUtility, parentWinLossValue,
      parentScoreMean, parentScoreStdev, parentLead, maxPVDepth
    );
    data.playSelectionValue = playSelectionValues[i];
    //Make sure data.lcb is from white's perspective, for consistency with everything else
    //In lcbBuf, it's from self perspective, unlike values at nodes.
    data.lcb = node.nextPla == P_BLACK ? -lcbBuf[i] : lcbBuf[i];
    data.radius = radiusBuf[i];
    buf.push_back(data);

    if(includeWeightFactors) {
      MoreNodeStats& stats = statsBuf[i];
      stats.stats = NodeStats(child->stats);
      stats.selfUtility = node.nextPla == P_WHITE ? data.utility : -data.utility;
      stats.weightAdjusted = stats.stats.getChildWeight(edgeVisits);
      stats.prevMoveLoc = moveLoc;
    }
  }

  //Find all children and compute weighting of the children based on their values
  if(includeWeightFactors) {
    double totalChildWeight = 0.0;
    for(int i = 0; i<numChildren; i++) {
      totalChildWeight += statsBuf[i].weightAdjusted;
    }
    if(searchParams.useNoisePruning) {
      double policyProbsBuf[NNPos::MAX_NN_POLICY_SIZE];
      for(int i = 0; i<numChildren; i++)
        policyProbsBuf[i] = std::max(1e-30, (double)policyProbs[getPos(statsBuf[i].prevMoveLoc)]);
      totalChildWeight = pruneNoiseWeight(statsBuf, numChildren, totalChildWeight, policyProbsBuf);
    }
    double amountToSubtract = 0.0;
    double amountToPrune = 0.0;
    downweightBadChildrenAndNormalizeWeight(
      numChildren, totalChildWeight, totalChildWeight,
      amountToSubtract, amountToPrune, statsBuf
    );
    for(int i = 0; i<numChildren; i++)
      buf[i].weightFactor = statsBuf[i].weightAdjusted;
  }

  //Fill the rest of the moves directly from policy
  if(numChildren < minMovesToTryToGet) {
    //A bit inefficient, but no big deal
    for(int i = 0; i<minMovesToTryToGet - numChildren; i++) {
      int bestPos = -1;
      double bestPolicy = -1.0;
      for(int pos = 0; pos<NNPos::MAX_NN_POLICY_SIZE; pos++) {
        if(policyProbs[pos] < bestPolicy)
          continue;

        bool alreadyUsed = false;
        for(int j = 0; j<buf.size(); j++) {
          if(getPos(buf[j].move) == pos) {
            alreadyUsed = true;
            break;
          }
        }
        if(alreadyUsed)
          continue;

        bestPos = pos;
        bestPolicy = policyProbs[pos];
      }
      if(bestPos < 0 || bestPolicy < 0.0)
        break;

      Loc bestMove = NNPos::posToLoc(bestPos,rootBoard.x_size,rootBoard.y_size,nnXLen,nnYLen);
      AnalysisData data = getAnalysisDataOfSingleChild(
        NULL, 0, scratchLocs, scratchValues, bestMove, bestPolicy, fpuValue, parentUtility, parentWinLossValue,
        parentScoreMean, parentScoreStdev, parentLead, maxPVDepth
      );
      buf.push_back(data);
    }
  }
  std::stable_sort(buf.begin(),buf.end());

  if(duplicateForSymmetries && searchParams.rootSymmetryPruning && rootSymmetries.size() > 1) {
    vector<AnalysisData> newBuf;
    std::set<Loc> isDone;
    for(int i = 0; i<buf.size(); i++) {
      const AnalysisData& data = buf[i];
      for(int symmetry : rootSymmetries) {
        Loc symMove = SymmetryHelpers::getSymLoc(data.move, rootBoard, symmetry);
        if(contains(isDone,symMove))
          continue;
        const std::vector<int>& avoidMoveUntilByLoc = rootPla == P_BLACK ? avoidMoveUntilByLocBlack : avoidMoveUntilByLocWhite;
        if(avoidMoveUntilByLoc.size() > 0 && avoidMoveUntilByLoc[symMove] > 0)
          continue;

        isDone.insert(symMove);
        newBuf.push_back(data);
        //Replace the fields that need to be adjusted for symmetry
        AnalysisData& newData = newBuf.back();
        newData.move = symMove;
        if(symmetry != 0)
          newData.isSymmetryOf = data.move;
        newData.symmetry = symmetry;
        for(int j = 0; j<newData.pv.size(); j++)
          newData.pv[j] = SymmetryHelpers::getSymLoc(newData.pv[j], rootBoard, symmetry);
      }
    }
    buf = std::move(newBuf);
  }

  for(int i = 0; i<buf.size(); i++)
    buf[i].order = i;
}

void Search::printPVForMove(ostream& out, const SearchNode* n, Loc move, int maxDepth) const {
  vector<Loc> buf;
  vector<int64_t> visitsBuf;
  vector<int64_t> edgeVisitsBuf;
  vector<Loc> scratchLocs;
  vector<double> scratchValues;
  appendPVForMove(buf,visitsBuf,edgeVisitsBuf,scratchLocs,scratchValues,n,move,maxDepth);
  for(int i = 0; i<buf.size(); i++) {
    if(i > 0)
      out << " ";
    out << Location::toString(buf[i],rootBoard);
  }
}

void Search::printTree(ostream& out, const SearchNode* node, PrintTreeOptions options, Player perspective) const {
  if(node == NULL)
    return;
  string prefix;
  AnalysisData data;
  {
    vector<Loc> scratchLocs;
    vector<double> scratchValues;
    //Use dummy values for parent
    double policyProb = NAN;
    double fpuValue = 0;
    double parentUtility = 0;
    double parentWinLossValue = 0;
    double parentScoreMean = 0;
    double parentScoreStdev = 0;
    double parentLead = 0;
    //Since we don't have an edge from another parent we are following, we just use the visits on the node itself as the edge visits.
    int64_t edgeVisits = node->stats.visits.load(std::memory_order_acquire);
    data = getAnalysisDataOfSingleChild(
      node, edgeVisits, scratchLocs, scratchValues,
      Board::NULL_LOC, policyProb, fpuValue, parentUtility, parentWinLossValue,
      parentScoreMean, parentScoreStdev, parentLead, options.maxPVDepth_
    );
    data.weightFactor = NAN;
  }
  perspective = (perspective != P_BLACK && perspective != P_WHITE) ? node->nextPla : perspective;
  printTreeHelper(out, node, options, prefix, 0, 0, data, perspective);
}

void Search::printTreeHelper(
  ostream& out, const SearchNode* n, const PrintTreeOptions& options,
  string& prefix, int64_t origVisits, int depth, const AnalysisData& data, Player perspective
) const {
  if(n == NULL)
    return;

  const SearchNode& node = *n;

  Player perspectiveToUse = (perspective != P_BLACK && perspective != P_WHITE) ? n->nextPla : perspective;
  double perspectiveFactor = perspectiveToUse == P_BLACK ? -1.0 : 1.0;

  if(depth == 0)
    origVisits = data.numVisits;

  //Output for this node
  {
    out << prefix;
    char buf[128];

    out << ": ";

    if(data.childVisits > 0) {
      sprintf(buf,"T %6.2fc ",(perspectiveFactor * data.utility * 100.0));
      out << buf;
      sprintf(buf,"W %6.2fc ",(perspectiveFactor * data.resultUtility * 100.0));
      out << buf;
      sprintf(buf,"S %6.2fc (%+5.1f L %+5.1f) ",
              perspectiveFactor * data.scoreUtility * 100.0,
              perspectiveFactor * data.scoreMean,
              perspectiveFactor * data.lead
      );
      out << buf;
    }

    // bool hasNNValue = false;
    // double nnResultValue;
    // double nnTotalValue;
    // lock.lock();
    // if(node.nnOutput != nullptr) {
    //   nnResultValue = getResultUtilityFromNN(*node.nnOutput);
    //   nnTotalValue = getUtilityFromNN(*node.nnOutput);
    //   hasNNValue = true;
    // }
    // lock.unlock();

    // if(hasNNValue) {
    //   sprintf(buf,"VW %6.2fc VS %6.2fc ", nnResultValue * 100.0, (nnTotalValue - nnResultValue) * 100.0);
    //   out << buf;
    // }
    // else {
    //   sprintf(buf,"VW ---.--c VS ---.--c ");
    //   out << buf;
    // }

    if(depth > 0 && !isnan(data.lcb)) {
      sprintf(buf,"LCB %7.2fc ", perspectiveFactor * data.lcb * 100.0);
      out << buf;
    }

    if(!isnan(data.policyPrior)) {
      sprintf(buf,"P %5.2f%% ", data.policyPrior * 100.0);
      out << buf;
    }
    if(!isnan(data.weightFactor)) {
      sprintf(buf,"WF %5.1f ", data.weightFactor);
      out << buf;
    }
    if(data.playSelectionValue >= 0 && depth > 0) {
      sprintf(buf,"PSV %7.0f ", data.playSelectionValue);
      out << buf;
    }

    if(options.printSqs_) {
      sprintf(buf,"SMSQ %5.1f USQ %7.5f W %6.2f WSQ %8.2f ", data.scoreMeanSqAvg, data.utilitySqAvg, data.weightSum, data.weightSqSum);
      out << buf;
    }

    if(options.printAvgShorttermError_) {
      std::pair<double,double> wlAndScoreError = getShallowAverageShorttermWLAndScoreError(&node);
      sprintf(buf,"STWL %6.2fc STS %5.1f ", wlAndScoreError.first * 100.0, wlAndScoreError.second);
      out << buf;
    }

    // Using child visits here instead of edge visits because edge visits is at least
    // semi-reflected in WF and PSV.
    sprintf(buf,"N %7" PRIu64 "  --  ", data.childVisits);
    out << buf;

    printPV(out, data.pv);
    out << endl;
  }

  if(depth >= options.branch_.size()) {
    if(depth >= options.maxDepth_ + options.branch_.size())
      return;
    if(data.numVisits < options.minVisitsToExpand_)
      return;
    if((double)data.numVisits < origVisits * options.minVisitsPropToExpand_)
      return;
  }
  if((options.alsoBranch_ && depth == 0) || (!options.alsoBranch_ && depth == options.branch_.size())) {
    out << "---" << PlayerIO::playerToString(node.nextPla) << "(" << (node.nextPla == perspectiveToUse ? "^" : "v") << ")---" << endl;
  }

  vector<AnalysisData> analysisData;
  bool duplicateForSymmetries = false;
  getAnalysisData(node,analysisData,0,true,options.maxPVDepth_,duplicateForSymmetries);

  int numChildren = (int)analysisData.size();

  //Apply filtering conditions, but include children that don't match the filtering condition
  //but where there are children afterward that do, in case we ever use something more complex
  //than plain visits as a filter criterion. Do this by finding the last child that we want as the threshold.
  int lastIdxWithEnoughVisits = numChildren-1;
  while(true) {
    if(lastIdxWithEnoughVisits <= 0)
      break;

    int64_t childVisits = analysisData[lastIdxWithEnoughVisits].numVisits;
    bool hasEnoughVisits = childVisits >= options.minVisitsToShow_
      && (double)childVisits >= origVisits * options.minVisitsPropToShow_;
    if(hasEnoughVisits)
      break;
    lastIdxWithEnoughVisits--;
  }

  int numChildrenToRecurseOn = numChildren;
  if(options.maxChildrenToShow_ < numChildrenToRecurseOn)
    numChildrenToRecurseOn = options.maxChildrenToShow_;
  if(lastIdxWithEnoughVisits+1 < numChildrenToRecurseOn)
    numChildrenToRecurseOn = lastIdxWithEnoughVisits+1;


  for(int i = 0; i<numChildren; i++) {
    const SearchNode* child = analysisData[i].node;
    Loc moveLoc = analysisData[i].move;

    if((depth >= options.branch_.size() && i < numChildrenToRecurseOn) ||
       (depth < options.branch_.size() && moveLoc == options.branch_[depth]) ||
       (depth < options.branch_.size() && options.alsoBranch_ && i < numChildrenToRecurseOn)
    ) {
      size_t oldLen = prefix.length();
      string locStr = Location::toString(moveLoc,rootBoard);
      if(locStr == "pass")
        prefix += "pss";
      else
        prefix += locStr;
      prefix += " ";
      while(prefix.length() < oldLen+4)
        prefix += " ";
      int nextDepth = depth+1;
      if(depth < options.branch_.size() && moveLoc != options.branch_[depth])
        nextDepth = (int)options.branch_.size() + 1;
      printTreeHelper(out,child,options,prefix,origVisits,nextDepth,analysisData[i], perspective);
      prefix.erase(oldLen);
    }
  }
}


std::pair<double,double> Search::getShallowAverageShorttermWLAndScoreError(const SearchNode* node) const {
  if(node == NULL)
    node = rootNode;
  if(node == NULL)
    return std::make_pair(0.0,0.0);
  if(!nnEvaluator->supportsShorttermError())
    return std::make_pair(-1.0,-1.0);
  std::unordered_set<const SearchNode*> graphPath;
  double policyProbsBuf[NNPos::MAX_NN_POLICY_SIZE];
  double wlError = 0.0;
  double scoreError = 0.0;
  // Stop deepening when we hit a node whose proportion in the final average would be less than this.
  // Sublinear in visits so that the cost of this grows more slowly than overall search depth.
  int64_t visits = node->stats.visits.load(std::memory_order_acquire);
  double minProp = 0.25 / pow(std::max(1.0,(double)visits),0.625);
  double desiredProp = 1.0;
  getShallowAverageShorttermWLAndScoreErrorHelper(
    node,
    graphPath,
    policyProbsBuf,
    minProp,
    desiredProp,
    wlError,
    scoreError
  );
  return std::make_pair(wlError,scoreError);
}

void Search::getShallowAverageShorttermWLAndScoreErrorHelper(
  const SearchNode* node,
  std::unordered_set<const SearchNode*>& graphPath,
  double policyProbsBuf[NNPos::MAX_NN_POLICY_SIZE],
  double minProp,
  double desiredProp,
  double& wlError,
  double& scoreError
) const {
  const NNOutput* nnOutput = node->getNNOutput();
  if(nnOutput == NULL) {
    // Accumulate nothing. This will be correct for terminal nodes, which have no uncertainty.
    // Not quite correct for multithreading, but no big deal, this value isn't used for anything critical
    // and currently isn't called while multithreaded. Yay code debt.
    return;
  }

  if(desiredProp < minProp) {
    // We don't track the average errors on nodes, so just use the error of this node's raw nn output.
    wlError += desiredProp * nnOutput->shorttermWinlossError;
    scoreError += desiredProp * nnOutput->shorttermScoreError;
    return;
  }

  std::pair<std::unordered_set<const SearchNode*>::iterator,bool> result = graphPath.insert(node);
  // No insertion, node was already there, this means we hit a cycle in the graph
  if(!result.second) {
    //Just treat it as base case and immediately terminate.
    wlError += desiredProp * nnOutput->shorttermWinlossError;
    scoreError += desiredProp * nnOutput->shorttermScoreError;
    return;
  }

  ConstSearchNodeChildrenReference children = node->getChildren();
  int childrenCapacity = children.getCapacity();

  vector<MoreNodeStats> statsBuf;
  for(int i = 0; i<childrenCapacity; i++) {
    const SearchChildPointer& childPointer = children[i];
    const SearchNode* child = childPointer.getIfAllocated();
    if(child == NULL)
      break;
    int64_t edgeVisits = childPointer.getEdgeVisits();
    Loc moveLoc = childPointer.getMoveLocRelaxed();
    MoreNodeStats stats;
    stats.stats = NodeStats(child->stats);
    stats.selfUtility = node->nextPla == P_WHITE ? stats.stats.utilityAvg : -stats.stats.utilityAvg;
    stats.weightAdjusted = stats.stats.getChildWeight(edgeVisits);
    stats.prevMoveLoc = moveLoc;
    statsBuf.push_back(stats);
  }
  int numChildren = (int)statsBuf.size();

  // Find all children and compute weighting of the children based on their values
  {
    double totalChildWeight = 0.0;
    for(int i = 0; i<numChildren; i++) {
      totalChildWeight += statsBuf[i].weightAdjusted;
    }
    const float* policyProbs = nnOutput->getPolicyProbsMaybeNoised();
    if(searchParams.useNoisePruning) {
      for(int i = 0; i<numChildren; i++)
        policyProbsBuf[i] = std::max(1e-30, (double)policyProbs[getPos(statsBuf[i].prevMoveLoc)]);
      totalChildWeight = pruneNoiseWeight(statsBuf, numChildren, totalChildWeight, policyProbsBuf);
    }
    double amountToSubtract = 0.0;
    double amountToPrune = 0.0;
    downweightBadChildrenAndNormalizeWeight(
      numChildren, totalChildWeight, totalChildWeight,
      amountToSubtract, amountToPrune, statsBuf
    );
  }

  //What we actually weight the children by for averaging
  double relativeChildrenWeightSum = 0.0;
  //What the weights of the children sum to from the search.
  double childrenWeightSum = 0;
  for(int i = 0; i<numChildren; i++) {
    double childWeight = statsBuf[i].weightAdjusted;
    relativeChildrenWeightSum += childWeight;
    childrenWeightSum += childWeight;
  }
  double parentNNWeight = computeWeightFromNNOutput(nnOutput);
  parentNNWeight = std::max(parentNNWeight,1e-10);
  double desiredPropFromChildren = desiredProp * childrenWeightSum / (childrenWeightSum + parentNNWeight);
  double selfProp = desiredProp * parentNNWeight / (childrenWeightSum + parentNNWeight);

  // In multithreading we may sometimes have children but with no weight at all yet, in that case just use parent alone.
  if(desiredPropFromChildren <= 0.0 || relativeChildrenWeightSum <= 0.0) {
    selfProp += desiredPropFromChildren;
  }
  else {

    for(int i = 0; i<numChildren; i++) {
      const SearchNode* child = children[i].getIfAllocated();
      assert(child != NULL);
      double childWeight = statsBuf[i].weightAdjusted;
      double desiredPropFromChild = childWeight / relativeChildrenWeightSum * desiredPropFromChildren;
      getShallowAverageShorttermWLAndScoreErrorHelper(child,graphPath,policyProbsBuf,minProp,desiredPropFromChild,wlError,scoreError);
    }
  }

  graphPath.erase(node);

  // Also add in the direct evaluation of this node.
  {
    wlError += selfProp * nnOutput->shorttermWinlossError;
    scoreError += selfProp * nnOutput->shorttermScoreError;
  }
}

bool Search::getSharpScore(const SearchNode* node, double& ret) const {
  if(node == NULL)
    node = rootNode;
  if(node == NULL)
    return false;

  int64_t visits = node->stats.visits.load(std::memory_order_acquire);
  // Stop deepening when we hit a node whose proportion in the final average would be less than this.
  // Sublinear in visits so that the cost of this grows more slowly than overall search depth.
  double minProp = 0.25 / pow(std::max(1.0,(double)visits),0.5);
  double desiredProp = 1.0;

  // Store initial value so we can start accumulating
  ret = 0.0;

  std::unordered_set<const SearchNode*> graphPath;

  double policyProbsBuf[NNPos::MAX_NN_POLICY_SIZE];
  if(node != rootNode) {
    return getSharpScoreHelper(node,graphPath,policyProbsBuf,minProp,desiredProp,ret);
  }

  const NNOutput* nnOutput = node->getNNOutput();
  if(nnOutput == NULL)
    return false;

  vector<double> playSelectionValues;
  vector<Loc> locs; // not used
  const bool allowDirectPolicyMoves = false;
  const bool alwaysComputeLcb = false;
  const bool neverUseLcb = true;
  bool suc = getPlaySelectionValues(*node,locs,playSelectionValues,NULL,1.0,allowDirectPolicyMoves,alwaysComputeLcb,neverUseLcb,NULL,NULL);
  // If there are no children, or otherwise values could not be computed, then fall back to the normal case
  if(!suc) {
    ReportedSearchValues values;
    if(getNodeValues(node,values)) {
      ret = values.expectedScore;
      return true;
    }
    return false;
  }

  int numChildren = (int)playSelectionValues.size();

  ConstSearchNodeChildrenReference children = node->getChildren();

  //What we actually weight the children by for averaging sharp score, sharper than the plain weight.
  double relativeChildrenWeightSum = 0.0;
  //What the weights of the children sum to from the search.
  double childrenWeightSum = 0;
  for(int i = 0; i<numChildren; i++) {
    double childWeight = playSelectionValues[i];
    relativeChildrenWeightSum += childWeight * childWeight * childWeight;
    childrenWeightSum += childWeight;
  }
  double parentNNWeight = computeWeightFromNNOutput(nnOutput);
  parentNNWeight = std::max(parentNNWeight,1e-10);
  double desiredPropFromChildren = desiredProp * childrenWeightSum / (childrenWeightSum + parentNNWeight);
  double selfProp = desiredProp * parentNNWeight / (childrenWeightSum + parentNNWeight);

  // In multithreading we may sometimes have children but with no weight at all yet, in that case just use parent alone.
  if(desiredPropFromChildren <= 0.0 || relativeChildrenWeightSum <= 0.0) {
    selfProp += desiredPropFromChildren;
  }
  else {

    graphPath.insert(node);

    for(int i = 0; i<numChildren; i++) {
      const SearchNode* child = children[i].getIfAllocated();
      assert(child != NULL);
      double childWeight = playSelectionValues[i];
      double desiredPropFromChild = childWeight * childWeight * childWeight / relativeChildrenWeightSum * desiredPropFromChildren;
      bool accumulated = getSharpScoreHelper(child,graphPath,policyProbsBuf,minProp,desiredPropFromChild,ret);
      if(!accumulated)
        selfProp += desiredPropFromChild;
    }

    graphPath.erase(node);

  }

  // Also add in the direct evaluation of this node.
  {
    double scoreMean = (double)nnOutput->whiteScoreMean;
    // cout << "Accumulating " << scoreMean << " " << selfProp << endl;
    ret += scoreMean * selfProp;
  }
  return true;
}


bool Search::getSharpScoreHelper(
  const SearchNode* node,
  std::unordered_set<const SearchNode*>& graphPath,
  double policyProbsBuf[NNPos::MAX_NN_POLICY_SIZE],
  double minProp,
  double desiredProp,
  double& ret
) const {
  const NNOutput* nnOutput = node->getNNOutput();
  if(nnOutput == NULL || desiredProp < minProp) {
    NodeStats stats = NodeStats(node->stats);
    if(stats.visits <= 0)
      return false;
    // cout << "Accumulating " << stats.scoreMeanAvg << " " << desiredProp << endl;
    ret += stats.scoreMeanAvg * desiredProp;
    return true;
  }

  ConstSearchNodeChildrenReference children = node->getChildren();
  int childrenCapacity = children.getCapacity();

  if(childrenCapacity <= 0) {
    double scoreMean = (double)nnOutput->whiteScoreMean;
    // cout << "Accumulating " << scoreMean << " " << desiredProp << endl;
    ret += scoreMean * desiredProp;
    return true;
  }

  std::pair<std::unordered_set<const SearchNode*>::iterator,bool> result = graphPath.insert(node);
  // No insertion, node was already there, this means we hit a cycle in the graph
  if(!result.second) {
    // Just treat it as base case and immediately terminate.
    double scoreMean = (double)nnOutput->whiteScoreMean;
    // cout << "Accumulating " << scoreMean << " " << desiredProp << endl;
    ret += scoreMean * desiredProp;
    return true;
  }

  vector<MoreNodeStats> statsBuf;
  for(int i = 0; i<childrenCapacity; i++) {
    const SearchChildPointer& childPointer = children[i];
    const SearchNode* child = childPointer.getIfAllocated();
    if(child == NULL)
      break;
    int64_t edgeVisits = childPointer.getEdgeVisits();
    Loc moveLoc = childPointer.getMoveLocRelaxed();
    MoreNodeStats stats;
    stats.stats = NodeStats(child->stats);
    stats.selfUtility = node->nextPla == P_WHITE ? stats.stats.utilityAvg : -stats.stats.utilityAvg;
    stats.weightAdjusted = stats.stats.getChildWeight(edgeVisits);
    stats.prevMoveLoc = moveLoc;
    statsBuf.push_back(stats);
  }
  int numChildren = (int)statsBuf.size();

  //Find all children and compute weighting of the children based on their values
  {
    double totalChildWeight = 0.0;
    for(int i = 0; i<numChildren; i++) {
      totalChildWeight += statsBuf[i].weightAdjusted;
    }
    const float* policyProbs = nnOutput->getPolicyProbsMaybeNoised();
    if(searchParams.useNoisePruning) {
      for(int i = 0; i<numChildren; i++)
        policyProbsBuf[i] = std::max(1e-30, (double)policyProbs[getPos(statsBuf[i].prevMoveLoc)]);
      totalChildWeight = pruneNoiseWeight(statsBuf, numChildren, totalChildWeight, policyProbsBuf);
    }
    double amountToSubtract = 0.0;
    double amountToPrune = 0.0;
    downweightBadChildrenAndNormalizeWeight(
      numChildren, totalChildWeight, totalChildWeight,
      amountToSubtract, amountToPrune, statsBuf
    );
  }

  // What we actually weight the children by for averaging sharp score, sharper than the plain weight.
  double relativeChildrenWeightSum = 0.0;
  // What the weights of the children sum to from the search.
  double childrenWeightSum = 0;
  for(int i = 0; i<numChildren; i++) {
    if(statsBuf[i].stats.visits <= 0)
      continue;
    double childWeight = statsBuf[i].weightAdjusted;
    relativeChildrenWeightSum += childWeight * childWeight * childWeight;
    childrenWeightSum += childWeight;
  }
  double parentNNWeight = computeWeightFromNNOutput(nnOutput);
  parentNNWeight = std::max(parentNNWeight,1e-10);
  double desiredPropFromChildren = desiredProp * childrenWeightSum / (childrenWeightSum + parentNNWeight);
  double selfProp = desiredProp * parentNNWeight / (childrenWeightSum + parentNNWeight);

  // In multithreading we may sometimes have children but with no weight at all yet, in that case just use parent alone.
  if(desiredPropFromChildren <= 0.0 || relativeChildrenWeightSum <= 0.0) {
    selfProp += desiredPropFromChildren;
  }
  else {

    for(int i = 0; i<numChildren; i++) {
      const SearchNode* child = children[i].getIfAllocated();
      assert(child != NULL);
      double childWeight = statsBuf[i].weightAdjusted;
      double desiredPropFromChild = childWeight * childWeight * childWeight / relativeChildrenWeightSum * desiredPropFromChildren;
      bool accumulated = getSharpScoreHelper(child,graphPath,policyProbsBuf,minProp,desiredPropFromChild,ret);
      if(!accumulated)
        selfProp += desiredPropFromChild;
    }
  }

  graphPath.erase(node);

  // Also add in the direct evaluation of this node.
  {
    double scoreMean = (double)nnOutput->whiteScoreMean;
    // cout << "Accumulating " << scoreMean << " " << selfProp << endl;
    ret += scoreMean * selfProp;
  }
  return true;
}

vector<double> Search::getAverageTreeOwnership(const SearchNode* node) const {
  if(node == NULL)
    node = rootNode;
  if(!alwaysIncludeOwnerMap)
    throw StringError("Called Search::getAverageTreeOwnership when alwaysIncludeOwnerMap is false");
  vector<double> vec(nnXLen*nnYLen,0.0);
  auto accumulate = [&vec,this](float* ownership, double selfProp){
    for (int pos = 0; pos < nnXLen*nnYLen; pos++)
      vec[pos] += selfProp * ownership[pos];
  };
  int64_t visits = node->stats.visits.load(std::memory_order_acquire);
  //Stop deepening when we hit a node whose proportion in the final average would be less than this.
  //Sublinear in visits so that the cost of this grows more slowly than overall search depth.
  double minProp = 0.5 / pow(std::max(1.0,(double)visits),0.75);
  //Entirely drop a node with weight less than this
  double pruneProp = minProp * 0.01;
  std::unordered_set<const SearchNode*> graphPath;
  traverseTreeForOwnership(minProp,pruneProp,1.0,node,graphPath,accumulate);
  return vec;
}

std::pair<vector<double>,vector<double>> Search::getAverageAndStandardDeviationTreeOwnership(const SearchNode* node) const {
  if(node == NULL)
    node = rootNode;
  vector<double> average(nnXLen*nnYLen,0.0);
  vector<double> stdev(nnXLen*nnYLen,0.0);
  auto accumulate = [&average,&stdev,this](float* ownership, double selfProp) {
    for (int pos = 0; pos < nnXLen*nnYLen; pos++) {
      const double value = ownership[pos];
      average[pos] += selfProp * value;
      stdev[pos] += selfProp * value * value;
    }
  };
  int64_t visits = node->stats.visits.load(std::memory_order_acquire);
  // Stop deepening when we hit a node whose proportion in the final average would be less than this.
  // Sublinear in visits so that the cost of this grows more slowly than overall search depth.
  double minProp = 0.5 / pow(std::max(1.0,(double)visits),0.75);
  // Entirely drop a node with weight less than this
  double pruneProp = minProp * 0.01;
  std::unordered_set<const SearchNode*> graphPath;
  traverseTreeForOwnership(minProp,pruneProp,1.0,node,graphPath,accumulate);
  for(int pos = 0; pos<nnXLen*nnYLen; pos++) {
    const double avg = average[pos];
    stdev[pos] = sqrt(max(stdev[pos] - avg * avg, 0.0));
  }
  return std::make_pair(average, stdev);
}

// Returns true if anything was accumulated, false otherwise.
template<typename Func>
bool Search::traverseTreeForOwnership(
  double minProp,
  double pruneProp,
  double desiredProp,
  const SearchNode* node,
  std::unordered_set<const SearchNode*>& graphPath,
  Func& accumulate
) const {
  if(node == NULL)
    return false;

  const NNOutput* nnOutput = node->getNNOutput();
  if(nnOutput == NULL)
    return false;

  // Base case
  if(desiredProp < minProp) {
    float* ownerMap = nnOutput->whiteOwnerMap;
    assert(ownerMap != NULL);
    accumulate(ownerMap, desiredProp);
    return true;
  }

  ConstSearchNodeChildrenReference children = node->getChildren();
  int childrenCapacity = children.getCapacity();

  if(childrenCapacity <= 0) {
    float* ownerMap = nnOutput->whiteOwnerMap;
    assert(ownerMap != NULL);
    accumulate(ownerMap, desiredProp);
    return true;
  }

  std::pair<std::unordered_set<const SearchNode*>::iterator,bool> result = graphPath.insert(node);
  // No insertion, node was already there, this means we hit a cycle in the graph
  if(!result.second) {
    //Just treat it as base case and immediately terminate.
    float* ownerMap = nnOutput->whiteOwnerMap;
    assert(ownerMap != NULL);
    accumulate(ownerMap, desiredProp);
    return true;
  }

  double selfProp;
  double parentNNWeight = computeWeightFromNNOutput(nnOutput);
  if(childrenCapacity <= SearchChildrenSizes::SIZE0TOTAL) {
    double childWeightBuf[SearchChildrenSizes::SIZE0TOTAL];
    selfProp = traverseTreeForOwnershipChildren(
      minProp, pruneProp, desiredProp, parentNNWeight, children, childWeightBuf, childrenCapacity, graphPath, accumulate
    );
  }
  else {
    vector<double> childWeightBuf(childrenCapacity);
    selfProp = traverseTreeForOwnershipChildren(
      minProp, pruneProp, desiredProp, parentNNWeight, children, &childWeightBuf[0], childrenCapacity, graphPath, accumulate
    );
  }

  graphPath.erase(node);

  float* ownerMap = nnOutput->whiteOwnerMap;
  assert(ownerMap != NULL);
  accumulate(ownerMap, selfProp);
  return true;
}

// Returns the prop that the parent node should be weighted.
// Not guaranteed to be <= the parent's weightsum due to multithreading.
template<typename Func>
double Search::traverseTreeForOwnershipChildren(
  double minProp,
  double pruneProp,
  double desiredProp,
  double parentNNWeight,
  ConstSearchNodeChildrenReference children,
  double* childWeightBuf,
  int childrenCapacity,
  std::unordered_set<const SearchNode*>& graphPath,
  Func& accumulate
) const {
  int numChildren = 0;
  for(int i = 0; i<childrenCapacity; i++) {
    const SearchChildPointer& childPointer = children[i];
    const SearchNode* child = childPointer.getIfAllocated();
    if(child == NULL)
      break;
    int64_t edgeVisits = childPointer.getEdgeVisits();
    double childWeight = child->stats.getChildWeight(edgeVisits);
    childWeightBuf[i] = childWeight;
    numChildren += 1;
  }

  // What we actually weight the children by for averaging ownership, sharper than the plain weight.
  double relativeChildrenWeightSum = 0.0;
  // What the weights of the children sum to from the search.
  double childrenWeightSum = 0;
  for(int i = 0; i<numChildren; i++) {
    double childWeight = childWeightBuf[i];
    relativeChildrenWeightSum += (double)childWeight * childWeight;
    childrenWeightSum += childWeight;
  }

  // Just in case
  parentNNWeight = std::max(parentNNWeight,1e-10);
  double desiredPropFromChildren = desiredProp * childrenWeightSum / (childrenWeightSum + parentNNWeight);
  double selfProp = desiredProp * parentNNWeight / (childrenWeightSum + parentNNWeight);

  // Recurse
  // In multithreading we may sometimes have children but with no weight at all yet, in that case just use parent alone.
  if(desiredPropFromChildren <= 0.0 || relativeChildrenWeightSum <= 0.0) {
    selfProp += desiredPropFromChildren;
  }
  else {
    for(int i = 0; i<numChildren; i++) {
      double childWeight = childWeightBuf[i];
      const SearchNode* child = children[i].getIfAllocated();
      assert(child != NULL);
      double desiredPropFromChild = (double)childWeight * childWeight / relativeChildrenWeightSum * desiredPropFromChildren;
      if(desiredPropFromChild < pruneProp)
        selfProp += desiredPropFromChild;
      else {
        bool accumulated = traverseTreeForOwnership(minProp,pruneProp,desiredPropFromChild,child,graphPath,accumulate);
        if(!accumulated)
          selfProp += desiredPropFromChild;
      }
    }
  }

  return selfProp;
}

std::vector<double> Search::getAverageTreeOwnership(
  const Player perspective,
  const SearchNode* node,
  int symmetry
) const {
  const vector<double> ownership = getAverageTreeOwnership(node);
  const Board& board = rootBoard;
  vector<double> ownershipToOutput(board.y_size * board.x_size, 0.0);

  for(int y = 0; y < board.y_size; y++) {
    for(int x = 0; x < board.x_size; x++) {
      int pos = NNPos::xyToPos(x, y, nnXLen);
      Loc symLoc = SymmetryHelpers::getSymLoc(x, y, board, symmetry);
      int symPos = Location::getY(symLoc, board.x_size) * board.x_size + Location::getX(symLoc, board.x_size);
      assert(symPos >= 0 && symPos < board.y_size * board.x_size);

      double o;
      if(perspective == P_BLACK || (perspective != P_BLACK && perspective != P_WHITE && rootPla == P_BLACK))
        o = -ownership[pos];
      else
        o = ownership[pos];
      // Round to 10^-6 to limit the size of output.
      // No guarantees that the serializer actually outputs something of this length rather than longer due to float wonkiness, but it should usually be true.
      ownershipToOutput[symPos] = Global::roundStatic(o, 1000000.0);
    }
  }
  return ownershipToOutput;
}

std::pair<std::vector<double>,std::vector<double>> Search::getAverageAndStandardDeviationTreeOwnership(
  const Player perspective,
  const SearchNode* node,
  int symmetry
) const {
  const std::pair<vector<double>,vector<double>> ownershipAverageAndStdev = getAverageAndStandardDeviationTreeOwnership(node);
  const Board& board = rootBoard;
  const vector<double>& ownership = std::get<0>(ownershipAverageAndStdev);
  const vector<double>& ownershipStdev = std::get<1>(ownershipAverageAndStdev);
  vector<double> ownershipToOutput(board.y_size * board.x_size, 0.0);
  vector<double> ownershipStdevToOutput(board.y_size * board.x_size, 0.0);

  for(int y = 0; y < board.y_size; y++) {
    for(int x = 0; x < board.x_size; x++) {
      int pos = NNPos::xyToPos(x, y, nnXLen);
      Loc symLoc = SymmetryHelpers::getSymLoc(x, y, board, symmetry);
      int symPos = Location::getY(symLoc, board.x_size) * board.x_size + Location::getX(symLoc, board.x_size);
      assert(symPos >= 0 && symPos < board.y_size * board.x_size);

      double o;
      if(perspective == P_BLACK || (perspective != P_BLACK && perspective != P_WHITE && rootPla == P_BLACK))
        o = -ownership[pos];
      else
        o = ownership[pos];
      // Round to 10^-6 to limit the size of output.
      // No guarantees that the serializer actually outputs something of this length rather than longer due to float wonkiness, but it should usually be true.
      ownershipToOutput[symPos] = Global::roundStatic(o, 1000000.0);
      ownershipStdevToOutput[symPos] = Global::roundStatic(ownershipStdev[pos], 1000000.0);
    }
  }
  return std::make_pair(ownershipToOutput, ownershipStdevToOutput);
}


bool Search::getAnalysisJson(
  const Player perspective,
  int analysisPVLen,
  bool preventEncore,
  bool includePolicy,
  bool includeOwnership,
  bool includeOwnershipStdev,
  bool includeMovesOwnership,
  bool includeMovesOwnershipStdev,
  bool includePVVisits,
  bool includeQValues,
  json& ret
) const {
  vector<AnalysisData> buf;
  static constexpr int minMoves = 0;
  static constexpr int OUTPUT_PRECISION = 8;

  const Board& board = rootBoard;
  const BoardHistory& hist = rootHistory;
  bool duplicateForSymmetries = true;
  getAnalysisData(buf, minMoves, false, analysisPVLen, duplicateForSymmetries);

  const NNOutput* nnOutput = NULL;
  const NNOutput* humanOutput = NULL;
  if(rootNode != NULL) {
    nnOutput = rootNode->getNNOutput();
    humanOutput = rootNode->getHumanOutput();
  }

  // Stats for all the individual moves
  json moveInfos = json::array();
  for(int i = 0; i < buf.size(); i++) {
    const AnalysisData& data = buf[i];
    double winrate = 0.5 * (1.0 + data.winLossValue);
    double utility = data.utility;
    double lcb = PlayUtils::getHackedLCBForWinrate(this, data, rootPla);
    double utilityLcb = data.lcb;
    double scoreMean = data.scoreMean;
    double lead = data.lead;
    if(perspective == P_BLACK || (perspective != P_BLACK && perspective != P_WHITE && rootPla == P_BLACK)) {
      winrate = 1.0 - winrate;
      lcb = 1.0 - lcb;
      utility = -utility;
      scoreMean = -scoreMean;
      lead = -lead;
      utilityLcb = -utilityLcb;
    }

    json moveInfo;
    moveInfo["move"] = Location::toString(data.move, board);
    moveInfo["visits"] = data.childVisits;
    moveInfo["weight"] = Global::roundDynamic(data.childWeightSum,OUTPUT_PRECISION);
    moveInfo["utility"] = Global::roundDynamic(utility,OUTPUT_PRECISION);
    moveInfo["winrate"] = Global::roundDynamic(winrate,OUTPUT_PRECISION);
    // We report lead for scoreMean here so that a bunch of legacy tools that use KataGo use lead instead, which
    // is usually a better field for user applications. We report scoreMean instead as scoreSelfplay
    moveInfo["scoreMean"] = Global::roundDynamic(lead,OUTPUT_PRECISION);
    moveInfo["scoreSelfplay"] = Global::roundDynamic(scoreMean,OUTPUT_PRECISION);
    moveInfo["scoreLead"] = Global::roundDynamic(lead,OUTPUT_PRECISION);
    moveInfo["scoreStdev"] = Global::roundDynamic(data.scoreStdev,OUTPUT_PRECISION);
    moveInfo["prior"] = Global::roundDynamic(data.policyPrior,OUTPUT_PRECISION);
    if(humanOutput != NULL)
      moveInfo["humanPrior"] = Global::roundDynamic(std::max(0.0,(double)humanOutput->policyProbs[getPos(data.move)]),OUTPUT_PRECISION);
    moveInfo["lcb"] = Global::roundDynamic(lcb,OUTPUT_PRECISION);
    moveInfo["utilityLcb"] = Global::roundDynamic(utilityLcb,OUTPUT_PRECISION);
    moveInfo["order"] = data.order;
    if(data.isSymmetryOf != Board::NULL_LOC)
      moveInfo["isSymmetryOf"] = Location::toString(data.isSymmetryOf, board);
    moveInfo["edgeVisits"] = data.numVisits;
    moveInfo["edgeWeight"] = Global::roundDynamic(data.weightSum,OUTPUT_PRECISION);
    moveInfo["playSelectionValue"] = Global::roundDynamic(data.playSelectionValue,OUTPUT_PRECISION);

    json pv = json::array();
    int pvLen =
      (preventEncore && data.pvContainsPass()) ? data.getPVLenUpToPhaseEnd(board, hist, rootPla) : (int)data.pv.size();
    for(int j = 0; j < pvLen; j++)
      pv.push_back(Location::toString(data.pv[j], board));
    moveInfo["pv"] = pv;

    if(includePVVisits) {
      assert(data.pvVisits.size() >= pvLen);
      json pvVisits = json::array();
      for(int j = 0; j < pvLen; j++)
        pvVisits.push_back(data.pvVisits[j]);
      moveInfo["pvVisits"] = pvVisits;

      assert(data.pvEdgeVisits.size() >= pvLen);
      json pvEdgeVisits = json::array();
      for(int j = 0; j < pvLen; j++)
        pvEdgeVisits.push_back(data.pvEdgeVisits[j]);
      moveInfo["pvEdgeVisits"] = pvEdgeVisits;
    }

    if(includeMovesOwnership && includeMovesOwnershipStdev) {
      std::pair<std::vector<double>,std::vector<double>> ownershipAndStdev = getAverageAndStandardDeviationTreeOwnership(perspective, data.node, data.symmetry);
      moveInfo["ownership"] = json(ownershipAndStdev.first);
      moveInfo["ownershipStdev"] = json(ownershipAndStdev.second);
    }
    else if(includeMovesOwnershipStdev) {
      std::pair<std::vector<double>,std::vector<double>> ownershipAndStdev = getAverageAndStandardDeviationTreeOwnership(perspective, data.node, data.symmetry);
      moveInfo["ownershipStdev"] = json(ownershipAndStdev.second);
    }
    else if(includeMovesOwnership) {
      moveInfo["ownership"] = json(getAverageTreeOwnership(perspective, data.node, data.symmetry));
    }

    moveInfos.push_back(moveInfo);
  }
  ret["moveInfos"] = moveInfos;

  // Stats for root position
  {
    ReportedSearchValues rootVals;
    bool suc = getPrunedRootValues(rootVals);
    if(!suc)
      return false;

    double winloss = rootVals.winLossValue;
    double scoreMean = rootVals.expectedScore;
    double lead = rootVals.lead;
    double utility = rootVals.utility;
    double flipFactor = (perspective == P_BLACK || (perspective != P_BLACK && perspective != P_WHITE && rootPla == P_BLACK)) ? -1.0 : 1.0;

    json rootInfo;
    rootInfo["visits"] = rootVals.visits;
    rootInfo["weight"] = rootVals.weight;
    rootInfo["winrate"] = Global::roundDynamic(0.5 + 0.5*winloss*flipFactor,OUTPUT_PRECISION);
    rootInfo["scoreSelfplay"] = Global::roundDynamic(scoreMean*flipFactor,OUTPUT_PRECISION);
    rootInfo["scoreLead"] = Global::roundDynamic(lead*flipFactor,OUTPUT_PRECISION);
    rootInfo["scoreStdev"] = Global::roundDynamic(rootVals.expectedScoreStdev,OUTPUT_PRECISION);
    rootInfo["utility"] = Global::roundDynamic(utility*flipFactor,OUTPUT_PRECISION);

    if(nnOutput != NULL) {
      rootInfo["rawWinrate"] = Global::roundDynamic(0.5 + 0.5*(nnOutput->whiteWinProb - nnOutput->whiteLossProb)*flipFactor,OUTPUT_PRECISION);
      rootInfo["rawLead"] = Global::roundDynamic(nnOutput->whiteLead*flipFactor,OUTPUT_PRECISION);
      rootInfo["rawScoreSelfplay"] = Global::roundDynamic(nnOutput->whiteScoreMean*flipFactor,OUTPUT_PRECISION);
      double wsm = nnOutput->whiteScoreMean;
      rootInfo["rawScoreSelfplayStdev"] = Global::roundDynamic(sqrt(std::max(0.0, nnOutput->whiteScoreMeanSq - wsm*wsm)),OUTPUT_PRECISION);
      rootInfo["rawNoResultProb"] = Global::roundDynamic(nnOutput->whiteNoResultProb,OUTPUT_PRECISION);
      rootInfo["rawStWrError"] = Global::roundDynamic(nnOutput->shorttermWinlossError * 0.5,OUTPUT_PRECISION);
      rootInfo["rawStScoreError"] = Global::roundDynamic(nnOutput->shorttermScoreError,OUTPUT_PRECISION);
      rootInfo["rawVarTimeLeft"] = Global::roundDynamic(nnOutput->varTimeLeft,OUTPUT_PRECISION);

      if(includeQValues && nnOutput->whiteQWinloss != NULL) {
        json thisSideQWinloss = json::array();
        json thisSideQScore = json::array();
        for(int y = 0; y < board.y_size; y++) {
          for(int x = 0; x < board.x_size; x++) {
            int pos = NNPos::xyToPos(x, y, nnXLen);
            if(nnOutput->policyProbs[pos] < 0) {
              thisSideQWinloss.push_back(nullptr);
              thisSideQScore.push_back(nullptr);
            }
            else {
              thisSideQWinloss.push_back(Global::roundDynamic(nnOutput->whiteQWinloss[pos]*flipFactor,OUTPUT_PRECISION));
              thisSideQScore.push_back(Global::roundDynamic(nnOutput->whiteQScore[pos]*flipFactor,OUTPUT_PRECISION));
            }
          }
        }
        int pos = NNPos::locToPos(Board::PASS_LOC, board.x_size, nnXLen, nnYLen);
        if(nnOutput->policyProbs[pos] < 0) {
          thisSideQWinloss.push_back(nullptr);
          thisSideQScore.push_back(nullptr);
        }
        else{
          thisSideQWinloss.push_back(Global::roundDynamic(nnOutput->whiteQWinloss[pos]*flipFactor,OUTPUT_PRECISION));
          thisSideQScore.push_back(Global::roundDynamic(nnOutput->whiteQScore[pos]*flipFactor,OUTPUT_PRECISION));
        }
      }
    }
    if(humanOutput != NULL) {
      rootInfo["humanWinrate"] = Global::roundDynamic(0.5 + 0.5*(humanOutput->whiteWinProb - humanOutput->whiteLossProb)*flipFactor,OUTPUT_PRECISION);
      rootInfo["humanScoreMean"] = Global::roundDynamic(humanOutput->whiteScoreMean*flipFactor,OUTPUT_PRECISION);
      double wsm = humanOutput->whiteScoreMean;
      rootInfo["humanScoreStdev"] = Global::roundDynamic(sqrt(std::max(0.0, humanOutput->whiteScoreMeanSq - wsm*wsm)),OUTPUT_PRECISION);
      rootInfo["humanStWrError"] = Global::roundDynamic(humanOutput->shorttermWinlossError * 0.5,OUTPUT_PRECISION);
      rootInfo["humanStScoreError"] = Global::roundDynamic(humanOutput->shorttermScoreError,OUTPUT_PRECISION);
    }

    Hash128 thisHash;
    Hash128 symHash;
    for(int symmetry = 0; symmetry < SymmetryHelpers::NUM_SYMMETRIES; symmetry++) {
      Board symBoard = SymmetryHelpers::getSymBoard(board,symmetry);
      Hash128 hash = symBoard.getSitHashWithSimpleKo(rootPla);
      if(symmetry == 0) {
        thisHash = hash;
        symHash = hash;
      }
      else {
        if(hash < symHash)
          symHash = hash;
      }
    }
    rootInfo["thisHash"] = Global::uint64ToHexString(thisHash.hash1) + Global::uint64ToHexString(thisHash.hash0);
    rootInfo["symHash"] = Global::uint64ToHexString(symHash.hash1) + Global::uint64ToHexString(symHash.hash0);
    rootInfo["currentPlayer"] = PlayerIO::playerToStringShort(rootPla);

    ret["rootInfo"] = rootInfo;
  }

  // Raw policy prior
  if(includePolicy) {
    {
      float policyProbs[NNPos::MAX_NN_POLICY_SIZE];
      bool suc = getPolicy(policyProbs);
      if(!suc)
        return false;
      json policy = json::array();
      for(int y = 0; y < board.y_size; y++) {
        for(int x = 0; x < board.x_size; x++) {
          int pos = NNPos::xyToPos(x, y, nnXLen);
          policy.push_back(Global::roundDynamic(policyProbs[pos],OUTPUT_PRECISION));
        }
      }

      int pos = NNPos::locToPos(Board::PASS_LOC, board.x_size, nnXLen, nnYLen);
      policy.push_back(Global::roundDynamic(policyProbs[pos],OUTPUT_PRECISION));
      ret["policy"] = policy;
    }

    if(humanOutput != NULL) {
      const float* policyProbs = humanOutput->getPolicyProbsMaybeNoised();
      json policy = json::array();
      for(int y = 0; y < board.y_size; y++) {
        for(int x = 0; x < board.x_size; x++) {
          int pos = NNPos::xyToPos(x, y, nnXLen);
          policy.push_back(Global::roundDynamic(policyProbs[pos],OUTPUT_PRECISION));
        }
      }
      int pos = NNPos::locToPos(Board::PASS_LOC, board.x_size, nnXLen, nnYLen);
      policy.push_back(Global::roundDynamic(policyProbs[pos],OUTPUT_PRECISION));
      ret["humanPolicy"] = policy;
    }
  }

  // Average tree ownership
  if(includeOwnership && includeOwnershipStdev) {
    int symmetry = 0;
    std::pair<std::vector<double>,std::vector<double>> ownershipAndStdev = getAverageAndStandardDeviationTreeOwnership(perspective, rootNode, symmetry);
    ret["ownership"] = json(ownershipAndStdev.first);
    ret["ownershipStdev"] = json(ownershipAndStdev.second);
  }
  else if(includeOwnershipStdev) {
    int symmetry = 0;
    std::pair<std::vector<double>,std::vector<double>> ownershipAndStdev = getAverageAndStandardDeviationTreeOwnership(perspective, rootNode, symmetry);
    ret["ownershipStdev"] = json(ownershipAndStdev.second);
  }
  else if(includeOwnership) {
    int symmetry = 0;
    ret["ownership"] = json(getAverageTreeOwnership(perspective, rootNode, symmetry));
  }

  return true;
}

//Compute all the stats of the node based on its children, pruning weights such that they are as expected
//based on policy and utility. This is used to give accurate rootInfo even with a lot of wide root noise
bool Search::getPrunedRootValues(ReportedSearchValues& values) const {
  return getPrunedNodeValues(rootNode,values);
}

bool Search::getPrunedNodeValues(const SearchNode* nodePtr, ReportedSearchValues& values) const {
  if(nodePtr == NULL)
    return false;
  const SearchNode& node = *nodePtr;

  ConstSearchNodeChildrenReference children = node.getChildren();
  int childrenCapacity = children.getCapacity();

  vector<double> playSelectionValues;
  vector<Loc> locs; // not used
  const bool allowDirectPolicyMoves = false;
  const bool alwaysComputeLcb = false;
  const bool neverUseLcb = true;
  bool suc = getPlaySelectionValues(node,locs,playSelectionValues,NULL,1.0,allowDirectPolicyMoves,alwaysComputeLcb,neverUseLcb,NULL,NULL);
  //If there are no children, or otherwise values could not be computed,
  //then fall back to the normal case and just listen to the values on the node rather than trying
  //to recompute things.
  if(!suc) {
    return getNodeValues(nodePtr,values);
  }

  double winLossValueSum = 0.0;
  double noResultValueSum = 0.0;
  double scoreMeanSum = 0.0;
  double scoreMeanSqSum = 0.0;
  double leadSum = 0.0;
  double utilitySum = 0.0;
  double utilitySqSum = 0.0;
  double weightSum = 0.0;
  double weightSqSum = 0.0;
  for(int i = 0; i<childrenCapacity; i++) {
    const SearchChildPointer& childPointer = children[i];
    const SearchNode* child = childPointer.getIfAllocated();
    if(child == NULL)
      break;
    int64_t edgeVisits = childPointer.getEdgeVisits();
    NodeStats stats = NodeStats(child->stats);

    if(stats.visits <= 0 || stats.weightSum <= 0.0 || edgeVisits <= 0)
      continue;
    double weight = playSelectionValues[i];
    double weightScaling = weight / stats.weightSum;
    winLossValueSum += weight * stats.winLossValueAvg;
    noResultValueSum += weight * stats.noResultValueAvg;
    scoreMeanSum += weight * stats.scoreMeanAvg;
    scoreMeanSqSum += weight * stats.scoreMeanSqAvg;
    leadSum += weight * stats.leadAvg;
    utilitySum += weight * stats.utilityAvg;
    utilitySqSum += weight * stats.utilitySqAvg;
    weightSqSum += weightScaling * weightScaling * stats.weightSqSum;
    weightSum += weight;
  }

  //Also add in the direct evaluation of this node.
  {
    const NNOutput* nnOutput = node.getNNOutput();
    //If somehow the nnOutput is still null here, skip
    if(nnOutput == NULL)
      return false;
    double winProb = (double)nnOutput->whiteWinProb;
    double lossProb = (double)nnOutput->whiteLossProb;
    double noResultProb = (double)nnOutput->whiteNoResultProb;
    double scoreMean = (double)nnOutput->whiteScoreMean;
    double scoreMeanSq = (double)nnOutput->whiteScoreMeanSq;
    double lead = (double)nnOutput->whiteLead;
    double utility =
      getResultUtility(winProb-lossProb, noResultProb)
      + getScoreUtility(scoreMean, scoreMeanSq);

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
  values = ReportedSearchValues(
    *this,
    winLossValueSum / weightSum,
    noResultValueSum / weightSum,
    scoreMeanSum / weightSum,
    scoreMeanSqSum / weightSum,
    leadSum / weightSum,
    utilitySum / weightSum,
    node.stats.weightSum.load(std::memory_order_acquire),
    node.stats.visits.load(std::memory_order_acquire)
  );
  return true;
}
