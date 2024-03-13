#include "../search/search.h"

#include "../core/fancymath.h"
#include "../search/searchnode.h"
#include "../search/patternbonustable.h"

//------------------------
#include "../core/using.h"
//------------------------

uint32_t Search::chooseIndexWithTemperature(Rand& rand, const double* relativeProbs, int numRelativeProbs, double temperature) {
  assert(numRelativeProbs > 0);
  assert(numRelativeProbs <= Board::MAX_ARR_SIZE); //We're just doing this on the stack
  double processedRelProbs[Board::MAX_ARR_SIZE];

  double maxValue = 0.0;
  for(int i = 0; i<numRelativeProbs; i++) {
    if(relativeProbs[i] > maxValue)
      maxValue = relativeProbs[i];
  }
  assert(maxValue > 0.0);

  //Temperature so close to 0 that we just calculate the max directly
  if(temperature <= 1.0e-4) {
    double bestProb = relativeProbs[0];
    int bestIdx = 0;
    for(int i = 1; i<numRelativeProbs; i++) {
      if(relativeProbs[i] > bestProb) {
        bestProb = relativeProbs[i];
        bestIdx = i;
      }
    }
    return bestIdx;
  }
  //Actual temperature
  else {
    double logMaxValue = log(maxValue);
    double sum = 0.0;
    for(int i = 0; i<numRelativeProbs; i++) {
      //Numerically stable way to raise to power and normalize
      processedRelProbs[i] = relativeProbs[i] <= 0.0 ? 0.0 : exp((log(relativeProbs[i]) - logMaxValue) / temperature);
      sum += processedRelProbs[i];
    }
    assert(sum > 0.0);
    uint32_t idxChosen = rand.nextUInt(processedRelProbs,numRelativeProbs);
    return idxChosen;
  }
}

void Search::computeDirichletAlphaDistribution(int policySize, const float* policyProbs, double* alphaDistr) {
  int legalCount = 0;
  for(int i = 0; i<policySize; i++) {
    if(policyProbs[i] >= 0)
      legalCount += 1;
  }

  if(legalCount <= 0)
    throw StringError("computeDirichletAlphaDistribution: No move with nonnegative policy value - can't even pass?");

  //We're going to generate a gamma draw on each move with alphas that sum up to searchParams.rootDirichletNoiseTotalConcentration.
  //Half of the alpha weight are uniform.
  //The other half are shaped based on the log of the existing policy.
  double logPolicySum = 0.0;
  for(int i = 0; i<policySize; i++) {
    if(policyProbs[i] >= 0) {
      alphaDistr[i] = log(std::min(0.01, (double)policyProbs[i]) + 1e-20);
      logPolicySum += alphaDistr[i];
    }
  }
  double logPolicyMean = logPolicySum / legalCount;
  double alphaPropSum = 0.0;
  for(int i = 0; i<policySize; i++) {
    if(policyProbs[i] >= 0) {
      alphaDistr[i] = std::max(0.0, alphaDistr[i] - logPolicyMean);
      alphaPropSum += alphaDistr[i];
    }
  }
  double uniformProb = 1.0 / legalCount;
  if(alphaPropSum <= 0.0) {
    for(int i = 0; i<policySize; i++) {
      if(policyProbs[i] >= 0)
        alphaDistr[i] = uniformProb;
    }
  }
  else {
    for(int i = 0; i<policySize; i++) {
      if(policyProbs[i] >= 0)
        alphaDistr[i] = 0.5 * (alphaDistr[i] / alphaPropSum + uniformProb);
    }
  }
}

void Search::addDirichletNoise(const SearchParams& searchParams, Rand& rand, int policySize, float* policyProbs) {
  double r[NNPos::MAX_NN_POLICY_SIZE];
  Search::computeDirichletAlphaDistribution(policySize, policyProbs, r);

  //r now contains the proportions with which we would like to split the alpha
  //The total of the alphas is searchParams.rootDirichletNoiseTotalConcentration
  //Generate gamma draw on each move
  double rSum = 0.0;
  for(int i = 0; i<policySize; i++) {
    if(policyProbs[i] >= 0) {
      r[i] = rand.nextGamma(r[i] * searchParams.rootDirichletNoiseTotalConcentration);
      rSum += r[i];
    }
    else
      r[i] = 0.0;
  }

  //Normalized gamma draws -> dirichlet noise
  for(int i = 0; i<policySize; i++)
    r[i] /= rSum;

  //At this point, r[i] contains a dirichlet distribution draw, so add it into the nnOutput.
  for(int i = 0; i<policySize; i++) {
    if(policyProbs[i] >= 0) {
      double weight = searchParams.rootDirichletNoiseWeight;
      policyProbs[i] = (float)(r[i] * weight + policyProbs[i] * (1.0-weight));
    }
  }
}


std::shared_ptr<NNOutput>* Search::maybeAddPolicyNoiseAndTemp(SearchThread& thread, bool isRoot, NNOutput* oldNNOutput) const {
  if(!isRoot)
    return NULL;
  if(!searchParams.rootNoiseEnabled &&
     searchParams.rootPolicyTemperature == 1.0 &&
     searchParams.rootPolicyTemperatureEarly == 1.0 &&
     rootHintLoc == Board::NULL_LOC &&
     !avoidMoveUntilRescaleRoot
  )
    return NULL;
  if(oldNNOutput == NULL)
    return NULL;
  if(oldNNOutput->noisedPolicyProbs != NULL)
    return NULL;

  //Copy nnOutput as we're about to modify its policy to add noise or temperature
  std::shared_ptr<NNOutput>* newNNOutputSharedPtr = new std::shared_ptr<NNOutput>(new NNOutput(*oldNNOutput));
  NNOutput* newNNOutput = newNNOutputSharedPtr->get();

  float* noisedPolicyProbs = new float[NNPos::MAX_NN_POLICY_SIZE];
  newNNOutput->noisedPolicyProbs = noisedPolicyProbs;
  std::copy(newNNOutput->policyProbs, newNNOutput->policyProbs + NNPos::MAX_NN_POLICY_SIZE, noisedPolicyProbs);

  if(searchParams.rootPolicyTemperature != 1.0 || searchParams.rootPolicyTemperatureEarly != 1.0) {
    double rootPolicyTemperature = interpolateEarly(
      searchParams.chosenMoveTemperatureHalflife, searchParams.rootPolicyTemperatureEarly, searchParams.rootPolicyTemperature
    );

    double maxValue = 0.0;
    for(int i = 0; i<policySize; i++) {
      double prob = noisedPolicyProbs[i];
      if(prob > maxValue)
        maxValue = prob;
    }
    assert(maxValue > 0.0);

    double logMaxValue = log(maxValue);
    double invTemp = 1.0 / rootPolicyTemperature;
    double sum = 0.0;

    for(int i = 0; i<policySize; i++) {
      if(noisedPolicyProbs[i] > 0) {
        //Numerically stable way to raise to power and normalize
        float p = (float)exp((log((double)noisedPolicyProbs[i]) - logMaxValue) * invTemp);
        noisedPolicyProbs[i] = p;
        sum += p;
      }
    }
    assert(sum > 0.0);
    for(int i = 0; i<policySize; i++) {
      if(noisedPolicyProbs[i] >= 0) {
        noisedPolicyProbs[i] = (float)(noisedPolicyProbs[i] / sum);
      }
    }
  }

  if(searchParams.rootNoiseEnabled) {
    addDirichletNoise(searchParams, thread.rand, policySize, noisedPolicyProbs);
  }

  if(avoidMoveUntilRescaleRoot) {
    const std::vector<int>& avoidMoveUntilByLoc = rootPla == P_BLACK ? avoidMoveUntilByLocBlack : avoidMoveUntilByLocWhite;
    if(avoidMoveUntilByLoc.size() > 0) {
      assert(avoidMoveUntilByLoc.size() >= Board::MAX_ARR_SIZE);
      double policySum = 0.0;
      for(Loc loc = 0; loc<Board::MAX_ARR_SIZE; loc++) {
        if((rootBoard.isOnBoard(loc) || loc == Board::PASS_LOC) && avoidMoveUntilByLoc[loc] <= 0) {
          int pos = getPos(loc);
          if(noisedPolicyProbs[pos] > 0) {
            policySum += noisedPolicyProbs[pos];
          }
        }
      }
      if(policySum > 0.0) {
        for(int i = 0; i<policySize; i++) {
          if(noisedPolicyProbs[i] > 0) {
            noisedPolicyProbs[i] = (float)(noisedPolicyProbs[i] / policySum);
          }
        }
      }
    }
  }

  //Move a small amount of policy to the hint move, around the same level that noising it would achieve
  if(rootHintLoc != Board::NULL_LOC) {
    const float propToMove = 0.02f;
    int pos = getPos(rootHintLoc);
    if(noisedPolicyProbs[pos] >= 0) {
      double amountToMove = 0.0;
      for(int i = 0; i<policySize; i++) {
        if(noisedPolicyProbs[i] >= 0) {
          amountToMove += noisedPolicyProbs[i] * propToMove;
          noisedPolicyProbs[i] *= (1.0f-propToMove);
        }
      }
      noisedPolicyProbs[pos] += (float)amountToMove;
    }
  }

  return newNNOutputSharedPtr;
}




double Search::getResultUtility(double winLossValue, double noResultValue) const {
  return (
    winLossValue * searchParams.winLossUtilityFactor +
    noResultValue * searchParams.noResultUtilityForWhite
  );
}

double Search::getResultUtilityFromNN(const NNOutput& nnOutput) const {
  return (
    (nnOutput.whiteWinProb - nnOutput.whiteLossProb) * searchParams.winLossUtilityFactor +
    nnOutput.whiteNoResultProb * searchParams.noResultUtilityForWhite
  );
}

double Search::getScoreUtility(double scoreMeanAvg, double scoreMeanSqAvg) const {
  double scoreMean = scoreMeanAvg;
  double scoreMeanSq = scoreMeanSqAvg;
  double scoreStdev = ScoreValue::getScoreStdev(scoreMean, scoreMeanSq);
  double sqrtBoardArea = rootBoard.sqrtBoardArea();
  double staticScoreValue = ScoreValue::expectedWhiteScoreValue(scoreMean,scoreStdev,0.0,2.0, sqrtBoardArea);
  double dynamicScoreValue = ScoreValue::expectedWhiteScoreValue(scoreMean,scoreStdev,recentScoreCenter,searchParams.dynamicScoreCenterScale, sqrtBoardArea);
  return staticScoreValue * searchParams.staticScoreUtilityFactor + dynamicScoreValue * searchParams.dynamicScoreUtilityFactor;
}

double Search::getScoreUtilityDiff(double scoreMeanAvg, double scoreMeanSqAvg, double delta) const {
  double scoreMean = scoreMeanAvg;
  double scoreMeanSq = scoreMeanSqAvg;
  double scoreStdev = ScoreValue::getScoreStdev(scoreMean, scoreMeanSq);
  double sqrtBoardArea = rootBoard.sqrtBoardArea();
  double staticScoreValueDiff =
    ScoreValue::expectedWhiteScoreValue(scoreMean + delta,scoreStdev,0.0,2.0, sqrtBoardArea)
    -ScoreValue::expectedWhiteScoreValue(scoreMean,scoreStdev,0.0,2.0, sqrtBoardArea);
  double dynamicScoreValueDiff =
    ScoreValue::expectedWhiteScoreValue(scoreMean + delta,scoreStdev,recentScoreCenter,searchParams.dynamicScoreCenterScale, sqrtBoardArea)
    -ScoreValue::expectedWhiteScoreValue(scoreMean,scoreStdev,recentScoreCenter,searchParams.dynamicScoreCenterScale, sqrtBoardArea);
  return staticScoreValueDiff * searchParams.staticScoreUtilityFactor + dynamicScoreValueDiff * searchParams.dynamicScoreUtilityFactor;
}

//Ignores scoreMeanSq's effect on the utility, since that's complicated
double Search::getApproxScoreUtilityDerivative(double scoreMean) const {
  double sqrtBoardArea = rootBoard.sqrtBoardArea();
  double staticScoreValueDerivative = ScoreValue::whiteDScoreValueDScoreSmoothNoDrawAdjust(scoreMean,0.0,2.0, sqrtBoardArea);
  double dynamicScoreValueDerivative = ScoreValue::whiteDScoreValueDScoreSmoothNoDrawAdjust(scoreMean,recentScoreCenter,searchParams.dynamicScoreCenterScale, sqrtBoardArea);
  return staticScoreValueDerivative * searchParams.staticScoreUtilityFactor + dynamicScoreValueDerivative * searchParams.dynamicScoreUtilityFactor;
}


double Search::getUtilityFromNN(const NNOutput& nnOutput) const {
  double resultUtility = getResultUtilityFromNN(nnOutput);
  return resultUtility + getScoreUtility(nnOutput.whiteScoreMean, nnOutput.whiteScoreMeanSq);
}


bool Search::isAllowedRootMove(Loc moveLoc) const {
  assert(moveLoc == Board::PASS_LOC || rootBoard.isOnBoard(moveLoc));

  //A bad situation that can happen that unnecessarily prolongs training games is where one player
  //repeatedly passes and the other side repeatedly fills the opponent's space and/or suicides over and over.
  //To mitigate some of this and save computation, we make it so that at the root, if the last four moves by the opponent
  //were passes, we will never play a move in either player's pass-alive area. In theory this could prune
  //a good move in situations like https://senseis.xmp.net/?1EyeFlaw, but this should be extraordinarly rare,
  if(searchParams.rootPruneUselessMoves &&
     rootHistory.moveHistory.size() > 0 &&
     moveLoc != Board::PASS_LOC
  ) {
    size_t lastIdx = rootHistory.moveHistory.size()-1;
    Player opp = getOpp(rootPla);
    if(lastIdx >= 6 &&
       rootHistory.moveHistory[lastIdx-0].loc == Board::PASS_LOC &&
       rootHistory.moveHistory[lastIdx-2].loc == Board::PASS_LOC &&
       rootHistory.moveHistory[lastIdx-4].loc == Board::PASS_LOC &&
       rootHistory.moveHistory[lastIdx-6].loc == Board::PASS_LOC &&
       rootHistory.moveHistory[lastIdx-0].pla == opp &&
       rootHistory.moveHistory[lastIdx-2].pla == opp &&
       rootHistory.moveHistory[lastIdx-4].pla == opp &&
       rootHistory.moveHistory[lastIdx-6].pla == opp &&
       (rootSafeArea[moveLoc] == opp || rootSafeArea[moveLoc] == rootPla))
      return false;
  }

  if(searchParams.rootSymmetryPruning && moveLoc != Board::PASS_LOC && rootSymDupLoc[moveLoc]) {
    return false;
  }

  return true;
}

double Search::getPatternBonus(Hash128 patternBonusHash, Player prevMovePla) const {
  if(patternBonusTable == NULL || prevMovePla != plaThatSearchIsFor)
    return 0;
  return patternBonusTable->get(patternBonusHash).utilityBonus;
}


double Search::getEndingWhiteScoreBonus(const SearchNode& parent, Loc moveLoc) const {
  if(&parent != rootNode || moveLoc == Board::NULL_LOC)
    return 0.0;

  const NNOutput* nnOutput = parent.getNNOutput();
  if(nnOutput == NULL || nnOutput->whiteOwnerMap == NULL)
    return 0.0;

  bool isAreaIsh = rootHistory.rules.scoringRule == Rules::SCORING_AREA
    || (rootHistory.rules.scoringRule == Rules::SCORING_TERRITORY && rootHistory.encorePhase >= 2);
  assert(nnOutput->nnXLen == nnXLen);
  assert(nnOutput->nnYLen == nnYLen);
  float* whiteOwnerMap = nnOutput->whiteOwnerMap;

  const double extreme = 0.95;
  const double tail = 0.05;

  //Extra points from the perspective of the root player
  double extraRootPoints = 0.0;
  if(isAreaIsh) {
    //Areaish scoring - in an effort to keep the game short and slightly discourage pointless territory filling at the end
    //discourage any move that, except in case of ko, is either:
    // * On a spot that the opponent almost surely owns, unless it captures stones.
    // * On a spot that the player almost surely owns and it is not adjacent to opponent stones and is not a connection of non-pass-alive groups.
    //These conditions should still make it so that "cleanup" and dame-filling moves are not discouraged.
    // * When playing button go, very slightly discourage passing - so that if there are an even number of dame, filling a dame is still favored over passing.
    if(moveLoc != Board::PASS_LOC && rootBoard.ko_loc == Board::NULL_LOC) {
      int pos = NNPos::locToPos(moveLoc,rootBoard.x_size,nnXLen,nnYLen);
      double plaOwnership = rootPla == P_WHITE ? whiteOwnerMap[pos] : -whiteOwnerMap[pos];
      if(plaOwnership <= -extreme) {
        if(!rootBoard.wouldBeCapture(moveLoc,rootPla))
          extraRootPoints -= searchParams.rootEndingBonusPoints * ((-extreme - plaOwnership) / tail);
      }
      else if(plaOwnership >= extreme) {
        if(!rootBoard.isAdjacentToPla(moveLoc,getOpp(rootPla)) &&
           !rootBoard.isNonPassAliveSelfConnection(moveLoc,rootPla,rootSafeArea)) {
          extraRootPoints -= searchParams.rootEndingBonusPoints * ((plaOwnership - extreme) / tail);
        }
      }
    }
    if(moveLoc == Board::PASS_LOC && rootHistory.hasButton) {
      extraRootPoints -= searchParams.rootEndingBonusPoints * 0.5;
    }
  }
  else {
    //Territorish scoring - slightly encourage dame-filling by discouraging passing, so that the player will try to do everything
    //non-point-losing first, like filling dame.
    //Human japanese rules often "want" you to fill the dame so this is a cosmetic adjustment to encourage the neural
    //net to learn to do so in the main phase rather than waiting until the encore.
    //But cosmetically, it's also not great if we just encourage useless threat moves in the opponent's territory to prolong the game.
    //So also discourage those moves except in cases of ko. Also similar to area scoring just to be symmetrical, discourage moves on spots
    //that the player almost surely owns that are not adjacent to opponent stones and are not a connection of non-pass-alive groups.
    if(moveLoc == Board::PASS_LOC)
      extraRootPoints -= searchParams.rootEndingBonusPoints * (2.0/3.0);
    else if(rootBoard.ko_loc == Board::NULL_LOC) {
      int pos = NNPos::locToPos(moveLoc,rootBoard.x_size,nnXLen,nnYLen);
      double plaOwnership = rootPla == P_WHITE ? whiteOwnerMap[pos] : -whiteOwnerMap[pos];
      if(plaOwnership <= -extreme)
        extraRootPoints -= searchParams.rootEndingBonusPoints * ((-extreme - plaOwnership) / tail);
      else if(plaOwnership >= extreme) {
        if(!rootBoard.isAdjacentToPla(moveLoc,getOpp(rootPla)) &&
           !rootBoard.isNonPassAliveSelfConnection(moveLoc,rootPla,rootSafeArea)) {
          extraRootPoints -= searchParams.rootEndingBonusPoints * ((plaOwnership - extreme) / tail);
        }
      }
    }
  }

  if(rootPla == P_WHITE)
    return extraRootPoints;
  else
    return -extraRootPoints;
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
  int64_t passEdgeVisits = 0;

  ConstSearchNodeChildrenReference children = node.getChildren();
  int childrenCapacity = children.getCapacity();
  for(int i = 0; i<childrenCapacity; i++) {
    const SearchChildPointer& childPointer = children[i];
    const SearchNode* child = childPointer.getIfAllocated();
    if(child == NULL)
      break;
    Loc moveLoc = childPointer.getMoveLocRelaxed();
    if(moveLoc == Board::PASS_LOC) {
      passNode = child;
      passEdgeVisits = childPointer.getEdgeVisits();
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
    int64_t passVisits = passNode->stats.visits.load(std::memory_order_acquire);
    double scoreMeanAvg = passNode->stats.scoreMeanAvg.load(std::memory_order_acquire);
    double leadAvg = passNode->stats.leadAvg.load(std::memory_order_acquire);
    double utilityAvg = passNode->stats.utilityAvg.load(std::memory_order_acquire);
    double childWeight = passNode->stats.getChildWeight(passEdgeVisits,passVisits);

    if(passVisits <= 0 || childWeight <= 1e-10)
      return false;
    passWeight = childWeight;
    passUtility = utilityAvg;
    passScoreMean = scoreMeanAvg;
    passLead = leadAvg;
  }

  const double extreme = 0.95;

  //Suppress pass if we find a move that is not a spot that the opponent almost certainly owns
  //or that is adjacent to a pla owned spot, and is not greatly worse than pass.
  for(int i = 0; i<childrenCapacity; i++) {
    const SearchChildPointer& childPointer = children[i];
    const SearchNode* child = childPointer.getIfAllocated();
    if(child == NULL)
      break;
    Loc moveLoc = childPointer.getMoveLocRelaxed();
    if(moveLoc == Board::PASS_LOC)
      continue;
    int pos = NNPos::locToPos(moveLoc,rootBoard.x_size,nnXLen,nnYLen);
    double plaOwnership = rootPla == P_WHITE ? whiteOwnerMap[pos] : -whiteOwnerMap[pos];
    bool oppOwned = plaOwnership < -extreme;
    bool adjToPlaOwned = false;
    for(int j = 0; j<4; j++) {
      Loc adj = moveLoc + rootBoard.adj_offsets[j];
      if(rootBoard.isOnBoard(adj)) {
        int adjPos = NNPos::locToPos(adj,rootBoard.x_size,nnXLen,nnYLen);
        double adjPlaOwnership = rootPla == P_WHITE ? whiteOwnerMap[adjPos] : -whiteOwnerMap[adjPos];
        if(adjPlaOwnership > extreme) {
          adjToPlaOwned = true;
          break;
        }
      }
    }
    if(oppOwned && !adjToPlaOwned)
      continue;

    int64_t edgeVisits = childPointer.getEdgeVisits();

    double scoreMeanAvg = child->stats.scoreMeanAvg.load(std::memory_order_acquire);
    double leadAvg = child->stats.leadAvg.load(std::memory_order_acquire);
    double utilityAvg = child->stats.utilityAvg.load(std::memory_order_acquire);
    double childWeight = child->stats.getChildWeight(edgeVisits);

    //Too few visits - reject move
    if((edgeVisits <= 500 && childWeight <= 2 * sqrt(passWeight)) || childWeight <= 1e-10)
      continue;

    double utility = utilityAvg;
    double scoreMean = scoreMeanAvg;
    double lead = leadAvg;

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

double Search::interpolateEarly(double halflife, double earlyValue, double value) const {
  double rawHalflives = (double)rootHistory.getCurrentTurnNumber() / halflife;
  double halflives = rawHalflives * 19.0 / sqrt(rootBoard.x_size*rootBoard.y_size);
  return value + (earlyValue - value) * pow(0.5, halflives);
}

void Search::getSelfUtilityLCBAndRadius(const SearchNode& parent, const SearchNode* child, int64_t edgeVisits, Loc moveLoc, double& lcbBuf, double& radiusBuf) const {
  int64_t childVisits = child->stats.visits.load(std::memory_order_acquire);
  double scoreMeanAvg = child->stats.scoreMeanAvg.load(std::memory_order_acquire);
  double scoreMeanSqAvg = child->stats.scoreMeanSqAvg.load(std::memory_order_acquire);
  double utilityAvg = child->stats.utilityAvg.load(std::memory_order_acquire);
  double utilitySqAvg = child->stats.utilitySqAvg.load(std::memory_order_acquire);
  double weightSum = child->stats.getChildWeight(edgeVisits,childVisits);
  double weightSqSum = child->stats.getChildWeightSq(edgeVisits,childVisits);

  // Max radius of the entire utility range
  double utilityRangeRadius = searchParams.winLossUtilityFactor + searchParams.staticScoreUtilityFactor + searchParams.dynamicScoreUtilityFactor;
  radiusBuf = 2.0 * utilityRangeRadius * searchParams.lcbStdevs;
  lcbBuf = -radiusBuf;
  if(childVisits <= 0 || weightSum <= 0.0 || weightSqSum <= 0.0)
    return;

  // Effective sample size for weighted data
  double ess = weightSum * weightSum / weightSqSum;

  // To behave well at low playouts, we'd like a variance approximation that makes sense even with very small sample sizes.
  // We'd like to avoid using a T distribution approximation because we actually know a bound on the scale of the utilities
  // involved, namely utilityRangeRadius. So instead add a prior with a small weight that the variance is the largest it can be.
  // This should give a relatively smooth scaling that works for small discrete samples but diminishes for larger playouts.
  double priorWeight = weightSum / (ess * ess * ess);
  utilitySqAvg = std::max(utilitySqAvg, utilityAvg * utilityAvg + 1e-8);
  utilitySqAvg = (utilitySqAvg * weightSum + (utilitySqAvg + utilityRangeRadius * utilityRangeRadius) * priorWeight) / (weightSum + priorWeight);
  weightSum += priorWeight;
  weightSqSum += priorWeight*priorWeight;

  // Recompute effective sample size now that we have the prior
  ess = weightSum * weightSum / weightSqSum;

  double endingScoreBonus = getEndingWhiteScoreBonus(parent,moveLoc);
  double utilityDiff = getScoreUtilityDiff(scoreMeanAvg, scoreMeanSqAvg, endingScoreBonus);
  double utilityWithBonus = utilityAvg + utilityDiff;
  double selfUtility = parent.nextPla == P_WHITE ? utilityWithBonus : -utilityWithBonus;

  double utilityVariance = utilitySqAvg - utilityAvg * utilityAvg;
  double estimateStdev = sqrt(utilityVariance / ess);
  double radius = estimateStdev * searchParams.lcbStdevs;

  lcbBuf = selfUtility - radius;
  radiusBuf = radius;
}
