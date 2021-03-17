
//-------------------------------------------------------------------------------------
//This file contains various functions for extracting stats and results from the search, choosing a move, etc
//-------------------------------------------------------------------------------------

#include "../search/search.h"

#include <inttypes.h>

#include "../core/fancymath.h"
#include "../program/playutils.h"

using namespace std;
using nlohmann::json;

static const int64_t MIN_VISITS_FOR_LCB = 3;

bool Search::getPlaySelectionValues(
  vector<Loc>& locs, vector<double>& playSelectionValues, double scaleMaxToAtLeast
) const {
  if(rootNode == NULL) {
    locs.clear();
    playSelectionValues.clear();
    return false;
  }
  bool allowDirectPolicyMoves = true;
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
  bool allowDirectPolicyMoves = true;
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
    false,lcbBuf,radiusBuf
  );
  return result;
}

bool Search::getPlaySelectionValues(
  const SearchNode& node,
  vector<Loc>& locs, vector<double>& playSelectionValues, vector<double>* retVisitCounts, double scaleMaxToAtLeast,
  bool allowDirectPolicyMoves, bool alwaysComputeLcb,
  //Note: lcbBuf is signed from the player to move's perspective
  double lcbBuf[NNPos::MAX_NN_POLICY_SIZE], double radiusBuf[NNPos::MAX_NN_POLICY_SIZE]
) const {
  locs.clear();
  playSelectionValues.clear();
  if(retVisitCounts != NULL)
    retVisitCounts->clear();

  double totalChildWeight = 0.0;
  double maxChildWeight = 0.0;
  const bool suppressPass = shouldSuppressPass(&node);

  //Store up basic weights
  int childrenCapacity;
  const SearchChildPointer* children = node.getChildren(childrenCapacity);
  for(int i = 0; i<childrenCapacity; i++) {
    const SearchNode* child = children[i].getIfAllocated();
    if(child == NULL)
      break;
    Loc moveLoc = child->prevMoveLoc;

    while(child->statsLock.test_and_set(std::memory_order_acquire));
    int64_t childVisits = child->stats.visits;
    double childWeight = child->stats.weightSum;
    child->statsLock.clear(std::memory_order_release);

    locs.push_back(moveLoc);
    totalChildWeight += childWeight;
    if(childWeight > maxChildWeight)
      maxChildWeight = childWeight;
    if(suppressPass && moveLoc == Board::PASS_LOC) {
      playSelectionValues.push_back(0.0);
      if(retVisitCounts != NULL)
        (*retVisitCounts).push_back(0.0);
    }
    else {
      playSelectionValues.push_back((double)childWeight);
      if(retVisitCounts != NULL)
        (*retVisitCounts).push_back((double)childVisits);
    }
  }

  int numChildren = playSelectionValues.size();

  //Find the best child by weight
  int mostWeightedIdx = 0;
  double mostWeightedChildWeight = -1e30;
  for(int i = 0; i<numChildren; i++) {
    double value = playSelectionValues[i];
    if(value > mostWeightedChildWeight) {
      mostWeightedChildWeight = value;
      mostWeightedIdx = i;
    }
  }

  //Possibly reduce weight on children that we spend too many visits on in retrospect
  if(&node == rootNode && numChildren > 0) {

    const SearchNode* bestChild = children[mostWeightedIdx].getIfAllocated();
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

    const NNOutput* nnOutput = node.getNNOutput();
    assert(nnOutput != NULL);
    const float* policyProbs = nnOutput->getPolicyProbsMaybeNoised();
    double bestChildExploreSelectionValue = getExploreSelectionValue(
      node,policyProbs,bestChild,totalChildWeight,fpuValue,
      parentUtility,parentWeightPerVisit,parentUtilityStdevFactor,
      isDuringSearch,maxChildWeight,NULL
    );

    for(int i = 0; i<numChildren; i++) {
      const SearchNode* child = children[i].getIfAllocated();
      if(suppressPass && child->prevMoveLoc == Board::PASS_LOC) {
        playSelectionValues[i] = 0;
        continue;
      }
      if(i != mostWeightedIdx) {
        double reduced = getReducedPlaySelectionWeight(
          node, policyProbs, child,
          totalChildWeight, parentUtilityStdevFactor, bestChildExploreSelectionValue
        );
        playSelectionValues[i] = (int64_t)ceil(reduced);
      }
    }
  }

  //Now compute play selection values taking into account LCB
  if(alwaysComputeLcb || (searchParams.useLcbForSelection && numChildren > 0)) {
    double bestLcb = -1e10;
    int bestLcbIndex = -1;
    for(int i = 0; i<numChildren; i++) {
      const SearchNode* child = children[i].getIfAllocated();
      getSelfUtilityLCBAndRadius(node,child,lcbBuf[i],radiusBuf[i]);
      //Check if this node is eligible to be considered for best LCB
      double weight = playSelectionValues[i];
      if(weight >= MIN_VISITS_FOR_LCB && weight >= searchParams.minVisitPropForLCB * mostWeightedChildWeight) {
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

  const NNOutput* nnOutput = node.getNNOutput();

  //If we have no children, then use the policy net directly. Only for the root, though, if calling this on any subtree
  //then just require that we have children, for implementation simplicity (since it requires that we have a board and a boardhistory too)
  //(and we also use isAllowedRootMove)
  if(numChildren == 0) {
    if(nnOutput == NULL || &node != rootNode || !allowDirectPolicyMoves)
      return false;

    bool obeyAllowedRootMove = true;
    while(true) {
      for(int movePos = 0; movePos<policySize; movePos++) {
        Loc moveLoc = NNPos::posToLoc(movePos,rootBoard.x_size,rootBoard.y_size,nnXLen,nnYLen);
        const float* policyProbs = nnOutput->getPolicyProbsMaybeNoised();
        double policyProb = policyProbs[movePos];
        if(!rootHistory.isLegal(rootBoard,moveLoc,rootPla) || policyProb < 0 || (obeyAllowedRootMove && !isAllowedRootMove(moveLoc)))
          continue;
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
  if(numChildren == 0)
    return false;

  double maxValue = 0.0;
  for(int i = 0; i<numChildren; i++) {
    if(playSelectionValues[i] > maxValue)
      maxValue = playSelectionValues[i];
  }

  if(maxValue <= 1e-50)
    return false;

  //Sanity check - if somehow we had more than this, something must have overflowed or gone wrong
  assert(maxValue < 1e40);

  double amountToSubtract = std::min(searchParams.chosenMoveSubtract, maxValue/64.0);
  double amountToPrune = std::min(searchParams.chosenMovePrune, maxValue/64.0);
  double newMaxValue = maxValue - amountToSubtract;
  for(int i = 0; i<numChildren; i++) {
    if(playSelectionValues[i] < amountToPrune)
      playSelectionValues[i] = 0.0;
    else {
      playSelectionValues[i] -= amountToSubtract;
      if(playSelectionValues[i] <= 0.0)
        playSelectionValues[i] = 0.0;
    }
  }

  assert(newMaxValue > 0.0);

  if(newMaxValue < scaleMaxToAtLeast) {
    for(int i = 0; i<numChildren; i++) {
      playSelectionValues[i] *= scaleMaxToAtLeast / newMaxValue;
    }
  }

  return true;
}

void Search::maybeRecomputeNormToTApproxTable() {
  if(normToTApproxZ <= 0.0 || normToTApproxZ != searchParams.lcbStdevs || normToTApproxTable.size() <= 0) {
    normToTApproxZ = searchParams.lcbStdevs;
    normToTApproxTable.clear();
    for(int i = 0; i < 512; i++)
      normToTApproxTable.push_back(FancyMath::normToTApprox(normToTApproxZ,(double)(i+MIN_VISITS_FOR_LCB)));
  }
}

double Search::getNormToTApproxForLCB(int64_t numVisits) const {
  int64_t idx = numVisits-MIN_VISITS_FOR_LCB;
  assert(idx >= 0);
  if(idx >= normToTApproxTable.size())
    idx = normToTApproxTable.size()-1;
  return normToTApproxTable[idx];
}

//Parent must be locked
void Search::getSelfUtilityLCBAndRadius(const SearchNode& parent, const SearchNode* child, double& lcbBuf, double& radiusBuf) const {
  while(child->statsLock.test_and_set(std::memory_order_acquire));
  double utilitySum = child->stats.utilitySum;
  double utilitySqSum = child->stats.utilitySqSum;
  double scoreMeanSum = child->stats.scoreMeanSum;
  double scoreMeanSqSum = child->stats.scoreMeanSqSum;
  double weightSum = child->stats.weightSum;
  double weightSqSum = child->stats.weightSqSum;
  child->statsLock.clear(std::memory_order_release);

  radiusBuf = 2.0 * (searchParams.winLossUtilityFactor + searchParams.staticScoreUtilityFactor + searchParams.dynamicScoreUtilityFactor);
  lcbBuf = -radiusBuf;
  if(weightSum <= 0.0)
    return;

  assert(weightSqSum > 0.0);
  double ess = weightSum * weightSum / weightSqSum;
  int64_t essInt = (int64_t)round(ess);
  if(essInt < MIN_VISITS_FOR_LCB)
    return;

  double utilityNoBonus = utilitySum / weightSum;
  double endingScoreBonus = getEndingWhiteScoreBonus(parent,child);
  double utilityDiff = getScoreUtilityDiff(scoreMeanSum, scoreMeanSqSum, weightSum, endingScoreBonus);
  double utilityWithBonus = utilityNoBonus + utilityDiff;
  double selfUtility = parent.nextPla == P_WHITE ? utilityWithBonus : -utilityWithBonus;

  double utilityVariance = std::max(1e-8, utilitySqSum/weightSum - utilityNoBonus * utilityNoBonus);
  double estimateStdev = sqrt(utilityVariance / ess);
  double radius = estimateStdev * getNormToTApproxForLCB(essInt);

  lcbBuf = selfUtility - radius;
  radiusBuf = radius;
}

bool Search::getRootValues(ReportedSearchValues& values) const {
  if(rootNode == NULL)
    return false;
  return getNodeValues(*rootNode,values);
}

ReportedSearchValues Search::getRootValuesRequireSuccess() const {
  ReportedSearchValues values;
  if(rootNode == NULL)
    throw StringError("Bug? Bot search root was null");
  bool success = getNodeValues(*rootNode,values);
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
  double scoreStdev = getScoreStdev(scoreMean,scoreMeanSq);
  values.staticScoreValue = ScoreValue::expectedWhiteScoreValue(scoreMean,scoreStdev,0.0,2.0,rootBoard);
  values.dynamicScoreValue = ScoreValue::expectedWhiteScoreValue(scoreMean,scoreStdev,recentScoreCenter,searchParams.dynamicScoreCenterScale,rootBoard);
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

  values.visits = 1;

  return true;
}


bool Search::getNodeValues(const SearchNode& node, ReportedSearchValues& values) const {
  const NNOutput* nnOutput = node.getNNOutput();
  if(nnOutput == NULL)
    return false;

  while(node.statsLock.test_and_set(std::memory_order_acquire));
  double winValueSum = node.stats.winValueSum;
  double noResultValueSum = node.stats.noResultValueSum;
  double scoreMeanSum = node.stats.scoreMeanSum;
  double scoreMeanSqSum = node.stats.scoreMeanSqSum;
  double leadSum = node.stats.leadSum;
  double weightSum = node.stats.weightSum;
  double utilitySum = node.stats.utilitySum;
  int64_t visits = node.stats.visits;

  node.statsLock.clear(std::memory_order_release);

  if(weightSum <= 0.0)
    return false;

  values.winValue = winValueSum / weightSum;
  values.lossValue = (weightSum - winValueSum - noResultValueSum) / weightSum;
  values.noResultValue = noResultValueSum / weightSum;
  double scoreMean = scoreMeanSum / weightSum;
  double scoreMeanSq = scoreMeanSqSum / weightSum;
  double scoreStdev = getScoreStdev(scoreMean,scoreMeanSq);
  values.staticScoreValue = ScoreValue::expectedWhiteScoreValue(scoreMean,scoreStdev,0.0,2.0,rootBoard);
  values.dynamicScoreValue = ScoreValue::expectedWhiteScoreValue(scoreMean,scoreStdev,recentScoreCenter,searchParams.dynamicScoreCenterScale,rootBoard);
  values.expectedScore = scoreMean;
  values.expectedScoreStdev = scoreStdev;
  values.lead = leadSum / weightSum;
  values.utility = utilitySum / weightSum;

  //Perform a little normalization - due to tiny floating point errors, winValue and lossValue could be outside [0,1].
  //(particularly lossValue, as it was produced by subtractions from weightSum that could have lost precision).
  if(values.winValue < 0.0) values.winValue = 0.0;
  if(values.lossValue < 0.0) values.lossValue = 0.0;
  if(values.noResultValue < 0.0) values.noResultValue = 0.0;
  double sum = values.winValue + values.lossValue + values.noResultValue;
  assert(sum > 0.9 && sum < 1.1); //If it's wrong by more than this, we have a bigger bug somewhere
  values.winValue /= sum;
  values.lossValue /= sum;
  values.noResultValue /= sum;

  double winLossValue = values.winValue - values.lossValue;
  assert(winLossValue > -1.01 && winLossValue < 1.01); //Sanity check, but allow generously for float imprecision
  if(winLossValue > 1.0) winLossValue = 1.0;
  if(winLossValue < -1.0) winLossValue = -1.0;
  values.winLossValue = winLossValue;
  values.visits = visits;

  return true;
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

  uint32_t idxChosen = chooseIndexWithTemperature(nonSearchRand, playSelectionValues.data(), playSelectionValues.size(), temperature);
  return locs[idxChosen];
}

//Hack to encourage well-behaved dame filling behavior under territory scoring
bool Search::shouldSuppressPass(const SearchNode* n) const {
  if(!searchParams.fillDameBeforePass || n == NULL || n != rootNode)
    return false;
  if(rootHistory.rules.scoringRule != Rules::SCORING_TERRITORY || rootHistory.encorePhase > 0)
    return false;

  const SearchNode& node = *n;
  const NNOutput* nnOutput = node.getNNOutput();
  if(nnOutput == NULL)
    return false;
  if(nnOutput->whiteOwnerMap == NULL)
    return false;
  assert(nnOutput->nnXLen == nnXLen);
  assert(nnOutput->nnYLen == nnYLen);
  const float* whiteOwnerMap = nnOutput->whiteOwnerMap;

  //Find the pass move
  const SearchNode* passNode = NULL;

  int childrenCapacity;
  const SearchChildPointer* children = node.getChildren(childrenCapacity);
  for(int i = 0; i<childrenCapacity; i++) {
    const SearchNode* child = children[i].getIfAllocated();
    if(child == NULL)
      break;
    Loc moveLoc = child->prevMoveLoc;
    if(moveLoc == Board::PASS_LOC) {
      passNode = child;
      break;
    }
  }
  if(passNode == NULL)
    return false;

  double passWeight;
  double passUtility;
  double passScoreMean;
  double passLead;
  {
    while(node.statsLock.test_and_set(std::memory_order_acquire));
    int64_t numVisits = node.stats.visits;
    double utilitySum = node.stats.utilitySum;
    double scoreMeanSum = node.stats.scoreMeanSum;
    double leadSum = node.stats.leadSum;
    double weightSum = node.stats.weightSum;
    node.statsLock.clear(std::memory_order_release);

    if(numVisits <= 0 || weightSum <= 1e-10)
      return false;
    passWeight = weightSum;
    passUtility = utilitySum / weightSum;
    passScoreMean = scoreMeanSum / weightSum;
    passLead = leadSum / weightSum;
  }

  const double extreme = 0.95;

  //Suppress pass if we find a move that is not a spot that the opponent almost certainly owns
  //or that is adjacent to a pla owned spot, and is not greatly worse than pass.
  for(int i = 0; i<childrenCapacity; i++) {
    const SearchNode* child = children[i].getIfAllocated();
    if(child == NULL)
      break;
    Loc moveLoc = child->prevMoveLoc;
    if(moveLoc == Board::PASS_LOC)
      continue;
    int pos = NNPos::locToPos(moveLoc,rootBoard.x_size,nnXLen,nnYLen);
    double plaOwnership = rootPla == P_WHITE ? whiteOwnerMap[pos] : -whiteOwnerMap[pos];
    bool oppOwned = plaOwnership < -extreme;
    bool adjToPlaOwned = false;
    for(int j = 0; j<4; j++) {
      Loc adj = moveLoc + rootBoard.adj_offsets[j];
      int adjPos = NNPos::locToPos(adj,rootBoard.x_size,nnXLen,nnYLen);
      double adjPlaOwnership = rootPla == P_WHITE ? whiteOwnerMap[adjPos] : -whiteOwnerMap[adjPos];
      if(adjPlaOwnership > extreme) {
        adjToPlaOwned = true;
        break;
      }
    }
    if(oppOwned && !adjToPlaOwned)
      continue;

    while(child->statsLock.test_and_set(std::memory_order_acquire));
    int64_t numVisits = child->stats.visits;
    double utilitySum = child->stats.utilitySum;
    double scoreMeanSum = child->stats.scoreMeanSum;
    double leadSum = child->stats.leadSum;
    double weightSum = child->stats.weightSum;
    child->statsLock.clear(std::memory_order_release);

    //Too few visits - reject move
    if((numVisits <= 500 && weightSum <= 2 * sqrt(passWeight)) || weightSum <= 1e-10)
      continue;

    double utility = utilitySum / weightSum;
    double scoreMean = scoreMeanSum / weightSum;
    double lead = leadSum / weightSum;

    if(rootPla == P_WHITE
       && utility > passUtility - 0.1
       && scoreMean > passScoreMean - 0.5
       && lead > passLead - 0.5)
      return true;
    if(rootPla == P_BLACK
       && utility < passUtility + 0.1
       && scoreMean < passScoreMean + 0.5
       && lead < passLead + 0.5)
      return true;
  }
  return false;
}

bool Search::getPolicy(float policyProbs[NNPos::MAX_NN_POLICY_SIZE]) const {
  if(rootNode == NULL)
    return false;
  const NNOutput* nnOutput = rootNode->getNNOutput();
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

//Safe to call concurrently with search
bool Search::getPolicySurpriseAndEntropy(double& surpriseRet, double& searchEntropyRet, double& policyEntropyRet) const {
  if(rootNode == NULL)
    return false;
  const NNOutput* nnOutput = rootNode->getNNOutput();
  if(nnOutput == NULL)
    return false;

  vector<Loc> locs;
  vector<double> playSelectionValues;
  bool allowDirectPolicyMoves = true;
  bool alwaysComputeLcb = false;
  double lcbBuf[NNPos::MAX_NN_POLICY_SIZE];
  double radiusBuf[NNPos::MAX_NN_POLICY_SIZE];
  bool suc = getPlaySelectionValues(
    *rootNode,locs,playSelectionValues,NULL,1.0,allowDirectPolicyMoves,alwaysComputeLcb,lcbBuf,radiusBuf
  );
  if(!suc)
    return false;

  float policyProbsFromNNBuf[NNPos::MAX_NN_POLICY_SIZE];
  {
    const float* policyProbsFromNN = nnOutput->getPolicyProbsMaybeNoised();
    std::copy(policyProbsFromNN, policyProbsFromNN+NNPos::MAX_NN_POLICY_SIZE, policyProbsFromNNBuf);
  }

  double sumPlaySelectionValues = 0.0;
  for(int i = 0; i<playSelectionValues.size(); i++)
    sumPlaySelectionValues += playSelectionValues[i];

  double surprise = 0.0;
  double searchEntropy = 0.0;
  for(int i = 0; i<playSelectionValues.size(); i++) {
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

  int childrenCapacity;
  const SearchChildPointer* children = rootNode->getChildren(childrenCapacity);
  for(int i = 0; i<childrenCapacity; i++) {
    const SearchNode* child = children[i].getIfAllocated();
    if(child == NULL)
      break;

    while(child->statsLock.test_and_set(std::memory_order_acquire));
    int64_t childVisits = child->stats.visits;
    double utilitySum = child->stats.utilitySum;
    double scoreMeanSum = child->stats.scoreMeanSum;
    double scoreMeanSqSum = child->stats.scoreMeanSqSum;
    double weightSum = child->stats.weightSum;
    child->statsLock.clear(std::memory_order_release);

    double utilityNoBonus = utilitySum / weightSum;
    double endingScoreBonus = getEndingWhiteScoreBonus(*rootNode,child);
    double utilityDiff = getScoreUtilityDiff(scoreMeanSum, scoreMeanSqSum, weightSum, endingScoreBonus);
    double utilityWithBonus = utilityNoBonus + utilityDiff;

    out << Location::toString(child->prevMoveLoc,rootBoard) << " " << Global::strprintf(
      "visits %d utilityNoBonus %.2fc utilityWithBonus %.2fc endingScoreBonus %.2f",
      childVisits, utilityNoBonus*100, utilityWithBonus*100, endingScoreBonus
    );
    out << endl;
  }
}

void Search::appendPV(vector<Loc>& buf, vector<int64_t>& visitsBuf, vector<Loc>& scratchLocs, vector<double>& scratchValues, const SearchNode* node, int maxDepth) const {
  appendPVForMove(buf,visitsBuf,scratchLocs,scratchValues,node,Board::NULL_LOC,maxDepth);
}

void Search::appendPVForMove(vector<Loc>& buf, vector<int64_t>& visitsBuf, vector<Loc>& scratchLocs, vector<double>& scratchValues, const SearchNode* node, Loc move, int maxDepth) const {
  if(node == NULL)
    return;

  for(int depth = 0; depth < maxDepth; depth++) {
    bool success = getPlaySelectionValues(*node, scratchLocs, scratchValues, NULL, 1.0, false);
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

    int childrenCapacity;
    const SearchChildPointer* children = node->getChildren(childrenCapacity);
    assert(bestChildIdx <= childrenCapacity);
    assert(scratchValues.size() <= childrenCapacity);

    const SearchNode* child = children[bestChildIdx].getIfAllocated();
    assert(child != NULL);
    node = child;

    while(node->statsLock.test_and_set(std::memory_order_acquire));
    int64_t visits = node->stats.visits;
    node->statsLock.clear(std::memory_order_release);

    buf.push_back(bestChildMoveLoc);
    visitsBuf.push_back(visits);
  }
}


void Search::printPV(ostream& out, const SearchNode* n, int maxDepth) const {
  vector<Loc> buf;
  vector<int64_t> visitsBuf;
  vector<Loc> scratchLocs;
  vector<double> scratchValues;
  appendPV(buf,visitsBuf,scratchLocs,scratchValues,n,maxDepth);
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
  const SearchNode* child, vector<Loc>& scratchLocs, vector<double>& scratchValues,
  Loc move, double policyProb, double fpuValue, double parentUtility, double parentWinLossValue,
  double parentScoreMean, double parentScoreStdev, double parentLead, int maxPVDepth
) const {
  int64_t numVisits = 0;
  double winValueSum = 0.0;
  double noResultValueSum = 0.0;
  double scoreMeanSum = 0.0;
  double scoreMeanSqSum = 0.0;
  double leadSum = 0.0;
  double weightSum = 0.0;
  double weightSqSum = 0.0;
  double utilitySum = 0.0;

  if(child != NULL) {
    while(child->statsLock.test_and_set(std::memory_order_acquire));
    numVisits = child->stats.visits;
    winValueSum = child->stats.winValueSum;
    noResultValueSum = child->stats.noResultValueSum;
    scoreMeanSum = child->stats.scoreMeanSum;
    scoreMeanSqSum = child->stats.scoreMeanSqSum;
    leadSum = child->stats.leadSum;
    weightSum = child->stats.weightSum;
    weightSqSum = child->stats.weightSqSum;
    utilitySum = child->stats.utilitySum;
    child->statsLock.clear(std::memory_order_release);
  }

  AnalysisData data;
  data.move = move;
  data.numVisits = numVisits;
  if(weightSum <= 1e-30) {
    data.utility = fpuValue;
    data.scoreUtility = getScoreUtility(parentScoreMean,parentScoreMean*parentScoreMean+parentScoreStdev*parentScoreStdev,1.0);
    data.resultUtility = fpuValue - data.scoreUtility;
    data.winLossValue = searchParams.winLossUtilityFactor == 1.0 ? parentWinLossValue + (fpuValue - parentUtility) : 0.0;
    data.scoreMean = parentScoreMean;
    data.scoreStdev = parentScoreStdev;
    data.lead = parentLead;
    data.ess = 0.0;
  }
  else {
    double winValue = winValueSum / weightSum;
    double lossValue = (weightSum - winValueSum - noResultValueSum) / weightSum;
    double noResultValue = noResultValueSum / weightSum;
    double scoreMean = scoreMeanSum / weightSum;
    double scoreMeanSq = scoreMeanSqSum / weightSum;
    double lead = leadSum / weightSum;

    data.utility = utilitySum / weightSum;
    data.resultUtility = getResultUtility(winValue, noResultValue);
    data.scoreUtility = data.utility - data.resultUtility;
    data.winLossValue = winValue - lossValue;
    data.scoreMean = scoreMean;
    data.scoreStdev = getScoreStdev(scoreMean,scoreMeanSq);
    data.lead = lead;
    data.ess = weightSum * weightSum / weightSqSum;
  }

  data.policyPrior = policyProb;
  data.order = 0;

  data.pv.clear();
  data.pv.push_back(move);
  data.pvVisits.clear();
  data.pvVisits.push_back(numVisits);
  appendPV(data.pv, data.pvVisits, scratchLocs, scratchValues, child, maxPVDepth);

  data.node = child;

  return data;
}

void Search::getAnalysisData(
  vector<AnalysisData>& buf,int minMovesToTryToGet, bool includeWeightFactors, int maxPVDepth
) const {
  buf.clear();
  if(rootNode == NULL)
    return;
  getAnalysisData(*rootNode, buf, minMovesToTryToGet, includeWeightFactors, maxPVDepth);
}

void Search::getAnalysisData(
  const SearchNode& node, vector<AnalysisData>& buf, int minMovesToTryToGet, bool includeWeightFactors, int maxPVDepth
) const {
  buf.clear();
  vector<const SearchNode*> children;
  children.reserve(rootBoard.x_size * rootBoard.y_size + 1);

  int numChildren;
  vector<Loc> scratchLocs;
  vector<double> scratchValues;
  double lcbBuf[NNPos::MAX_NN_POLICY_SIZE];
  double radiusBuf[NNPos::MAX_NN_POLICY_SIZE];
  float policyProbs[NNPos::MAX_NN_POLICY_SIZE];
  {
    int childrenCapacity;
    const SearchChildPointer* childrenArr = node.getChildren(childrenCapacity);
    for(int i = 0; i<childrenCapacity; i++) {
      const SearchNode* child = childrenArr[i].getIfAllocated();
      if(child == NULL)
        break;
      children.push_back(child);
    }
    numChildren = children.size();

    if(numChildren <= 0)
      return;
    assert(numChildren <= NNPos::MAX_NN_POLICY_SIZE);

    bool alwaysComputeLcb = true;
    bool success = getPlaySelectionValues(node, scratchLocs, scratchValues, NULL, 1.0, false, alwaysComputeLcb, lcbBuf, radiusBuf);
    if(!success)
      return;

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
      const SearchNode* child = children[i];
      policyProbMassVisited += policyProbs[getPos(child->prevMoveLoc)];
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
    while(node.statsLock.test_and_set(std::memory_order_acquire));
    double winValueSum = node.stats.winValueSum;
    double noResultValueSum = node.stats.noResultValueSum;
    double scoreMeanSum = node.stats.scoreMeanSum;
    double scoreMeanSqSum = node.stats.scoreMeanSqSum;
    double leadSum = node.stats.leadSum;
    double weightSum = node.stats.weightSum;
    node.statsLock.clear(std::memory_order_release);
    assert(weightSum > 0.0);

    double winValue = winValueSum / weightSum;
    double lossValue = (weightSum - winValueSum - noResultValueSum) / weightSum;

    parentWinLossValue = winValue - lossValue;
    parentScoreMean = scoreMeanSum / weightSum;
    double scoreMeanSq = scoreMeanSqSum / weightSum;
    parentScoreStdev = getScoreStdev(parentScoreMean,scoreMeanSq);
    parentLead = leadSum / weightSum;
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
    double policyProb = policyProbs[getPos(child->prevMoveLoc)];
    AnalysisData data = getAnalysisDataOfSingleChild(
      child, scratchLocs, scratchValues, child->prevMoveLoc, policyProb, fpuValue, parentUtility, parentWinLossValue,
      parentScoreMean, parentScoreStdev, parentLead, maxPVDepth
    );
    data.playSelectionValue = playSelectionValues[i];
    //Make sure data.lcb is from white's perspective, for consistency with everything else
    //In lcbBuf, it's from self perspective, unlike values at nodes.
    data.lcb = node.nextPla == P_BLACK ? -lcbBuf[i] : lcbBuf[i];
    data.radius = radiusBuf[i];
    buf.push_back(data);

    MoreNodeStats& stats = statsBuf[i];
    while(child->statsLock.test_and_set(std::memory_order_acquire));
    stats.stats = child->stats;
    child->statsLock.clear(std::memory_order_release);
    stats.selfUtility = node.nextPla == P_WHITE ? data.utility : -data.utility;
    stats.weightAdjusted = stats.stats.weightSum;
    stats.prevMoveLoc = child->prevMoveLoc;
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
        NULL, scratchLocs, scratchValues, bestMove, bestPolicy, fpuValue, parentUtility, parentWinLossValue,
        parentScoreMean, parentScoreStdev, parentLead, maxPVDepth
      );
      buf.push_back(data);
    }
  }
  std::stable_sort(buf.begin(),buf.end());

  for(int i = 0; i<buf.size(); i++)
    buf[i].order = i;
}

void Search::printPVForMove(ostream& out, const SearchNode* n, Loc move, int maxDepth) const {
  vector<Loc> buf;
  vector<int64_t> visitsBuf;
  vector<Loc> scratchLocs;
  vector<double> scratchValues;
  appendPVForMove(buf,visitsBuf,scratchLocs,scratchValues,n,move,maxDepth);
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
    data = getAnalysisDataOfSingleChild(
      node, scratchLocs, scratchValues,
      (node == rootNode ? Board::NULL_LOC : node->prevMoveLoc), policyProb, fpuValue, parentUtility, parentWinLossValue,
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

    if(data.numVisits > 0) {
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
      while(node.statsLock.test_and_set(std::memory_order_acquire));
      double scoreMeanSqSum = node.stats.scoreMeanSqSum;
      double utilitySqSum = node.stats.utilitySqSum;
      double weightSum = node.stats.weightSum;
      double weightSqSum = node.stats.weightSqSum;
      node.statsLock.clear(std::memory_order_release);
      sprintf(buf,"SMSQ %5.1f USQ %7.5f W %6.2f WSQ %8.2f ", scoreMeanSqSum/weightSum, utilitySqSum/weightSum, weightSum, weightSqSum);
      out << buf;
    }

    sprintf(buf,"N %7" PRIu64 "  --  ", data.numVisits);
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
  if(depth == options.branch_.size()) {
    out << "---" << PlayerIO::playerToString(node.nextPla) << "(" << (node.nextPla == perspectiveToUse ? "^" : "v") << ")---" << endl;
  }

  vector<AnalysisData> analysisData;
  getAnalysisData(node,analysisData,0,true,options.maxPVDepth_);

  int numChildren = analysisData.size();

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
    Loc moveLoc = child->prevMoveLoc;

    if((depth >= options.branch_.size() && i < numChildrenToRecurseOn) ||
       (depth < options.branch_.size() && moveLoc == options.branch_[depth]))
    {
      size_t oldLen = prefix.length();
      string locStr = Location::toString(moveLoc,rootBoard);
      if(locStr == "pass")
        prefix += "pss";
      else
        prefix += locStr;
      prefix += " ";
      while(prefix.length() < oldLen+4)
        prefix += " ";
      printTreeHelper(
        out,child,options,prefix,origVisits,depth+1,analysisData[i], perspective);
      prefix.erase(oldLen);
    }
  }
}


vector<double> Search::getAverageTreeOwnership(double minWeight, const SearchNode* node) const {
  if(node == NULL)
    node = rootNode;
  if(!alwaysIncludeOwnerMap)
    throw StringError("Called Search::getAverageTreeOwnership when alwaysIncludeOwnerMap is false");
  vector<double> vec(nnXLen*nnYLen,0.0);
  getAverageTreeOwnershipHelper(vec,minWeight,1.0,node);
  return vec;
}

double Search::getAverageTreeOwnershipHelper(vector<double>& accum, double minWeight, double desiredWeight, const SearchNode* node) const {
  if(node == NULL)
    return 0;

  const NNOutput* nnOutput = node->getNNOutput();
  if(nnOutput == NULL)
    return 0;

  int childrenCapacity;
  const SearchChildPointer* children = node->getChildren(childrenCapacity);

  vector<double> childWeightBuf(childrenCapacity);
  double thisNodeWeight = computeWeightFromNNOutput(nnOutput);
  int numChildren = 0;
  for(int i = 0; i<childrenCapacity; i++) {
    const SearchNode* child = children[i].getIfAllocated();
    if(child == NULL)
      break;
    while(child->statsLock.test_and_set(std::memory_order_acquire));
    double childWeight = child->stats.weightSum;
    child->statsLock.clear(std::memory_order_release);
    childWeightBuf[i] = childWeight;
    numChildren += 1;
  }

  double relativeChildrenWeightSum = 0.0;
  double usedChildrenWeightSum = 0;
  for(int i = 0; i<numChildren; i++) {
    double childWeight = childWeightBuf[i];
    if(childWeight < minWeight)
      continue;
    relativeChildrenWeightSum += (double)childWeight * childWeight;
    usedChildrenWeightSum += childWeight;
  }

  double desiredWeightFromChildren = desiredWeight * usedChildrenWeightSum / (usedChildrenWeightSum + thisNodeWeight);

  //Recurse
  double actualWeightFromChildren = 0.0;
  for(int i = 0; i<numChildren; i++) {
    double childWeight = childWeightBuf[i];
    if(childWeight < minWeight)
      continue;
    const SearchNode* child = children[i].getIfAllocated();
    assert(child != NULL);
    double desiredWeightFromChild = (double)childWeight * childWeight / relativeChildrenWeightSum * desiredWeightFromChildren;
    actualWeightFromChildren += getAverageTreeOwnershipHelper(accum,minWeight,desiredWeightFromChild,child);
  }

  double selfWeight = desiredWeight - actualWeightFromChildren;
  float* ownerMap = nnOutput->whiteOwnerMap;
  assert(ownerMap != NULL);
  for(int pos = 0; pos<nnXLen*nnYLen; pos++)
    accum[pos] += selfWeight * ownerMap[pos];

  return desiredWeight;
}

json Search::getJsonOwnershipMap(const Player pla, const Player perspective, const Board& board, const SearchNode* node, double ownershipMinWeight) const {
  vector<double> ownership = getAverageTreeOwnership(ownershipMinWeight, node);
  json ownerships = json::array();
  for(int y = 0; y < board.y_size; y++) {
    for(int x = 0; x < board.x_size; x++) {
      int pos = NNPos::xyToPos(x, y, nnXLen);
      double o;
      if(perspective == P_BLACK || (perspective != P_BLACK && perspective != P_WHITE && pla == P_BLACK))
        o = -ownership[pos];
      else
        o = ownership[pos];
      ownerships.push_back(o);
    }
  }
  return ownerships;
}

bool Search::getAnalysisJson(
  const Player perspective,
  const Board& board,
  const BoardHistory& hist,
  int analysisPVLen,
  double ownershipMinWeight,
  bool preventEncore,
  bool includePolicy,
  bool includeOwnership,
  bool includeMovesOwnership,
  bool includePVVisits,
  json& ret
) const {
  vector<AnalysisData> buf;
  static constexpr int minMoves = 0;

  getAnalysisData(buf, minMoves, false, analysisPVLen);

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
    moveInfo["visits"] = data.numVisits;
    moveInfo["utility"] = utility;
    moveInfo["winrate"] = winrate;
    moveInfo["scoreMean"] = lead;
    moveInfo["scoreSelfplay"] = scoreMean;
    moveInfo["scoreLead"] = lead;
    moveInfo["scoreStdev"] = data.scoreStdev;
    moveInfo["prior"] = data.policyPrior;
    moveInfo["lcb"] = lcb;
    moveInfo["utilityLcb"] = utilityLcb;
    moveInfo["order"] = data.order;

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
    }

    if(includeMovesOwnership)
      moveInfo["ownership"] = getJsonOwnershipMap(rootPla, perspective, board, data.node, ownershipMinWeight);
    moveInfos.push_back(moveInfo);
  }
  ret["moveInfos"] = moveInfos;

  // Stats for root position
  {
    ReportedSearchValues rootVals;
    bool suc = getRootValues(rootVals);
    if(!suc)
      return false;

    double winrate = 0.5 * (1.0 + rootVals.winLossValue);
    double scoreMean = rootVals.expectedScore;
    double lead = rootVals.lead;
    double utility = rootVals.utility;

    if(perspective == P_BLACK || (perspective != P_BLACK && perspective != P_WHITE && rootPla == P_BLACK)) {
      winrate = 1.0 - winrate;
      scoreMean = -scoreMean;
      lead = -lead;
      utility = -utility;
    }

    json rootInfo;
    rootInfo["visits"] = rootVals.visits;
    rootInfo["winrate"] = winrate;
    rootInfo["scoreSelfplay"] = scoreMean;
    rootInfo["scoreLead"] = lead;
    rootInfo["scoreStdev"] = rootVals.expectedScoreStdev;
    rootInfo["utility"] = utility;

    Hash128 thisHash;
    Hash128 symHash;
    for(int symmetry = 0; symmetry < 8; symmetry++) {
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

    ret["rootInfo"] = rootInfo;
  }
  // Raw policy prior
  if(includePolicy) {
    float policyProbs[NNPos::MAX_NN_POLICY_SIZE];
    bool suc = getPolicy(policyProbs);
    if(!suc)
      return false;
    json policy = json::array();
    for(int y = 0; y < board.y_size; y++) {
      for(int x = 0; x < board.x_size; x++) {
        int pos = NNPos::xyToPos(x, y, nnXLen);
        policy.push_back(policyProbs[pos]);
      }
    }

    int passPos = NNPos::locToPos(Board::PASS_LOC, board.x_size, nnXLen, nnYLen);
    policy.push_back(policyProbs[passPos]);
    ret["policy"] = policy;
  }
  // Average tree ownership
  if(includeOwnership)
    ret["ownership"] = getJsonOwnershipMap(rootPla, perspective, board, rootNode, ownershipMinWeight);
  return true;
}
