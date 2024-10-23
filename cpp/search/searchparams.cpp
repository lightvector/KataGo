#include "../search/searchparams.h"

#include "../external/nlohmann_json/json.hpp"

using nlohmann::json;

//Default search params
//The intent is that the are good default guesses for values of the parameters,
//with deterministic behavior (no noise, no randomization) and no bounds (unbounded time and visits).
//They are not necessarily the best parameters though, and have been kept mostly fixed over time even as things
//have changed to preserve the behavior of tests.
SearchParams::SearchParams()
  :winLossUtilityFactor(1.0),
   staticScoreUtilityFactor(0.3),
   dynamicScoreUtilityFactor(0.0),
   dynamicScoreCenterZeroWeight(0.0),
   dynamicScoreCenterScale(1.0),
   noResultUtilityForWhite(0.0),
   drawEquivalentWinsForWhite(0.5),
   cpuctExploration(1.0),
   cpuctExplorationLog(0.0),
   cpuctExplorationBase(500),
   cpuctUtilityStdevPrior(0.25),
   cpuctUtilityStdevPriorWeight(1.0),
   cpuctUtilityStdevScale(0.0),
   fpuReductionMax(0.2),
   fpuLossProp(0.0),
   fpuParentWeightByVisitedPolicy(false),
   fpuParentWeightByVisitedPolicyPow(1.0),
   fpuParentWeight(0.0),
   policyOptimism(0.0),
   valueWeightExponent(0.5),
   useNoisePruning(false),
   noisePruneUtilityScale(0.15),
   noisePruningCap(1e50),
   useUncertainty(false),
   uncertaintyCoeff(0.2),
   uncertaintyExponent(1.0),
   uncertaintyMaxWeight(8.0),
   useGraphSearch(false),
   graphSearchRepBound(11),
   graphSearchCatchUpLeakProb(0.0),
   //graphSearchCatchUpProp(0.0),
   rootNoiseEnabled(false),
   rootDirichletNoiseTotalConcentration(10.83),
   rootDirichletNoiseWeight(0.25),
   rootPolicyTemperature(1.0),
   rootPolicyTemperatureEarly(1.0),
   rootFpuReductionMax(0.2),
   rootFpuLossProp(0.0),
   rootNumSymmetriesToSample(1),
   rootSymmetryPruning(false),
   rootDesiredPerChildVisitsCoeff(0.0),
   rootPolicyOptimism(0.0),
   chosenMoveTemperature(0.0),
   chosenMoveTemperatureEarly(0.0),
   chosenMoveTemperatureHalflife(19),
   chosenMoveTemperatureOnlyBelowProb(1.0),
   chosenMoveSubtract(0.0),
   chosenMovePrune(1.0),
   useLcbForSelection(false),
   lcbStdevs(4.0),
   minVisitPropForLCB(0.05),
   useNonBuggyLcb(false),
   rootEndingBonusPoints(0.0),
   rootPruneUselessMoves(false),
   conservativePass(false),
   fillDameBeforePass(false),
   avoidMYTDaggerHackPla(C_EMPTY),
   wideRootNoise(0.0),
   enablePassingHacks(false),
   playoutDoublingAdvantage(0.0),
   playoutDoublingAdvantagePla(C_EMPTY),
   avoidRepeatedPatternUtility(0.0),
   nnPolicyTemperature(1.0f),
   antiMirror(false),
   ignorePreRootHistory(false),
   ignoreAllHistory(false),
   subtreeValueBiasFactor(0.0),
   subtreeValueBiasTableNumShards(65536),
   subtreeValueBiasFreeProp(0.8),
   subtreeValueBiasWeightExponent(0.5),
   nodeTableShardsPowerOfTwo(16),
   numVirtualLossesPerThread(3.0),
   numThreads(1),
   minPlayoutsPerThread(0.0),
   maxVisits(((int64_t)1) << 50),
   maxPlayouts(((int64_t)1) << 50),
   maxTime(1.0e20),
   maxVisitsPondering(((int64_t)1) << 50),
   maxPlayoutsPondering(((int64_t)1) << 50),
   maxTimePondering(1.0e20),
   lagBuffer(0.0),
   searchFactorAfterOnePass(1.0),
   searchFactorAfterTwoPass(1.0),
   treeReuseCarryOverTimeFactor(0.0),
   overallocateTimeFactor(1.0),
   midgameTimeFactor(1.0),
   midgameTurnPeakTime(130.0),
   endgameTurnTimeDecay(100.0),
   obviousMovesTimeFactor(1.0),
   obviousMovesPolicyEntropyTolerance(0.30),
   obviousMovesPolicySurpriseTolerance(0.15),
   futileVisitsThreshold(0.0),
   humanSLProfile(),
   humanSLCpuctExploration(1.0),
   humanSLCpuctPermanent(0.0),
   humanSLRootExploreProbWeightless(0.0),
   humanSLRootExploreProbWeightful(0.0),
   humanSLPlaExploreProbWeightless(0.0),
   humanSLPlaExploreProbWeightful(0.0),
   humanSLOppExploreProbWeightless(0.0),
   humanSLOppExploreProbWeightful(0.0),
   humanSLChosenMoveProp(0.0),
   humanSLChosenMoveIgnorePass(false),
   humanSLChosenMovePiklLambda(1000000000.0)
{}

SearchParams::~SearchParams()
{}

bool SearchParams::operator==(const SearchParams& other) const {
  return (
    winLossUtilityFactor == other.winLossUtilityFactor &&
    staticScoreUtilityFactor == other.staticScoreUtilityFactor &&
    dynamicScoreUtilityFactor == other.dynamicScoreUtilityFactor &&
    dynamicScoreCenterZeroWeight == other.dynamicScoreCenterZeroWeight &&
    dynamicScoreCenterScale == other.dynamicScoreCenterScale &&
    noResultUtilityForWhite == other.noResultUtilityForWhite &&
    drawEquivalentWinsForWhite == other.drawEquivalentWinsForWhite &&

    cpuctExploration == other.cpuctExploration &&
    cpuctExplorationLog == other.cpuctExplorationLog &&
    cpuctExplorationBase == other.cpuctExplorationBase &&

    cpuctUtilityStdevPrior == other.cpuctUtilityStdevPrior &&
    cpuctUtilityStdevPriorWeight == other.cpuctUtilityStdevPriorWeight &&
    cpuctUtilityStdevScale == other.cpuctUtilityStdevScale &&

    fpuReductionMax == other.fpuReductionMax &&
    fpuLossProp == other.fpuLossProp &&

    fpuParentWeightByVisitedPolicy == other.fpuParentWeightByVisitedPolicy &&
    fpuParentWeightByVisitedPolicyPow == other.fpuParentWeightByVisitedPolicyPow &&
    fpuParentWeight == other.fpuParentWeight &&

    policyOptimism == other.policyOptimism &&

    valueWeightExponent == other.valueWeightExponent &&
    useNoisePruning == other.useNoisePruning &&
    noisePruneUtilityScale == other.noisePruneUtilityScale &&
    noisePruningCap == other.noisePruningCap &&

    useUncertainty == other.useUncertainty &&
    uncertaintyCoeff == other.uncertaintyCoeff &&
    uncertaintyExponent == other.uncertaintyExponent &&
    uncertaintyMaxWeight == other.uncertaintyMaxWeight &&

    useGraphSearch == other.useGraphSearch &&
    graphSearchRepBound == other.graphSearchRepBound &&
    graphSearchCatchUpLeakProb == other.graphSearchCatchUpLeakProb &&

    rootNoiseEnabled == other.rootNoiseEnabled &&
    rootDirichletNoiseTotalConcentration == other.rootDirichletNoiseTotalConcentration &&
    rootDirichletNoiseWeight == other.rootDirichletNoiseWeight &&

    rootPolicyTemperature == other.rootPolicyTemperature &&
    rootPolicyTemperatureEarly == other.rootPolicyTemperatureEarly &&
    rootFpuReductionMax == other.rootFpuReductionMax &&
    rootFpuLossProp == other.rootFpuLossProp &&
    rootNumSymmetriesToSample == other.rootNumSymmetriesToSample &&
    rootSymmetryPruning == other.rootSymmetryPruning &&
    rootDesiredPerChildVisitsCoeff == other.rootDesiredPerChildVisitsCoeff &&

    rootPolicyOptimism == other.rootPolicyOptimism &&

    chosenMoveTemperature == other.chosenMoveTemperature &&
    chosenMoveTemperatureEarly == other.chosenMoveTemperatureEarly &&
    chosenMoveTemperatureHalflife == other.chosenMoveTemperatureHalflife &&

    chosenMoveTemperatureOnlyBelowProb == other.chosenMoveTemperatureOnlyBelowProb &&
    chosenMoveSubtract == other.chosenMoveSubtract &&
    chosenMovePrune == other.chosenMovePrune &&

    useLcbForSelection == other.useLcbForSelection &&
    lcbStdevs == other.lcbStdevs &&
    minVisitPropForLCB == other.minVisitPropForLCB &&
    useNonBuggyLcb == other.useNonBuggyLcb &&

    rootEndingBonusPoints == other.rootEndingBonusPoints &&
    rootPruneUselessMoves == other.rootPruneUselessMoves &&
    conservativePass == other.conservativePass &&
    fillDameBeforePass == other.fillDameBeforePass &&
    avoidMYTDaggerHackPla == other.avoidMYTDaggerHackPla &&
    wideRootNoise == other.wideRootNoise &&
    enablePassingHacks == other.enablePassingHacks &&

    playoutDoublingAdvantage == other.playoutDoublingAdvantage &&
    playoutDoublingAdvantagePla == other.playoutDoublingAdvantagePla &&

    avoidRepeatedPatternUtility == other.avoidRepeatedPatternUtility &&

    nnPolicyTemperature == other.nnPolicyTemperature &&
    antiMirror == other.antiMirror &&

    ignorePreRootHistory == other.ignorePreRootHistory &&
    ignoreAllHistory == other.ignoreAllHistory &&

    subtreeValueBiasFactor == other.subtreeValueBiasFactor &&
    subtreeValueBiasTableNumShards == other.subtreeValueBiasTableNumShards &&
    subtreeValueBiasFreeProp == other.subtreeValueBiasFreeProp &&
    subtreeValueBiasWeightExponent == other.subtreeValueBiasWeightExponent &&

    nodeTableShardsPowerOfTwo == other.nodeTableShardsPowerOfTwo &&
    numVirtualLossesPerThread == other.numVirtualLossesPerThread &&

    numThreads == other.numThreads &&
    minPlayoutsPerThread == other.minPlayoutsPerThread &&
    maxVisits == other.maxVisits &&
    maxPlayouts == other.maxPlayouts &&
    maxTime == other.maxTime &&

    maxVisitsPondering == other.maxVisitsPondering &&
    maxPlayoutsPondering == other.maxPlayoutsPondering &&
    maxTimePondering == other.maxTimePondering &&

    lagBuffer == other.lagBuffer &&

    searchFactorAfterOnePass == other.searchFactorAfterOnePass &&
    searchFactorAfterTwoPass == other.searchFactorAfterTwoPass &&

    treeReuseCarryOverTimeFactor == other.treeReuseCarryOverTimeFactor &&
    overallocateTimeFactor == other.overallocateTimeFactor &&
    midgameTimeFactor == other.midgameTimeFactor &&
    midgameTurnPeakTime == other.midgameTurnPeakTime &&
    endgameTurnTimeDecay == other.endgameTurnTimeDecay &&
    obviousMovesTimeFactor == other.obviousMovesTimeFactor &&
    obviousMovesPolicyEntropyTolerance == other.obviousMovesPolicyEntropyTolerance &&
    obviousMovesPolicySurpriseTolerance == other.obviousMovesPolicySurpriseTolerance &&

    futileVisitsThreshold == other.futileVisitsThreshold &&

    humanSLProfile == other.humanSLProfile &&
    humanSLCpuctExploration == other.humanSLCpuctExploration &&
    humanSLCpuctPermanent == other.humanSLCpuctPermanent &&
    humanSLRootExploreProbWeightless == other.humanSLRootExploreProbWeightless &&
    humanSLRootExploreProbWeightful == other.humanSLRootExploreProbWeightful &&
    humanSLPlaExploreProbWeightless == other.humanSLPlaExploreProbWeightless &&
    humanSLPlaExploreProbWeightful == other.humanSLPlaExploreProbWeightful &&
    humanSLOppExploreProbWeightless == other.humanSLOppExploreProbWeightless &&
    humanSLOppExploreProbWeightful == other.humanSLOppExploreProbWeightful &&

    humanSLChosenMoveProp == other.humanSLChosenMoveProp &&
    humanSLChosenMoveIgnorePass == other.humanSLChosenMoveIgnorePass &&
    humanSLChosenMovePiklLambda == other.humanSLChosenMovePiklLambda
  );
}

bool SearchParams::operator!=(const SearchParams& other) const {
  return !(*this == other);
}


SearchParams SearchParams::forTestsV1() {
  SearchParams params;
  params.staticScoreUtilityFactor = 0.1;
  params.dynamicScoreUtilityFactor = 0.3;
  params.dynamicScoreCenterZeroWeight = 0.2;
  params.dynamicScoreCenterScale = 0.75;
  params.cpuctExploration = 0.9;
  params.cpuctExplorationLog = 0.4;
  params.rootFpuReductionMax = 0.1;
  params.rootPolicyTemperatureEarly = 1.2;
  params.rootPolicyTemperature = 1.1;
  params.useLcbForSelection = true;
  params.lcbStdevs = 5;
  params.minVisitPropForLCB = 0.15;
  params.rootEndingBonusPoints = 0.5;
  params.rootPruneUselessMoves = true;
  params.conservativePass = true;
  params.useNonBuggyLcb = true;
  return params;
}

SearchParams SearchParams::forTestsV2() {
  SearchParams params;
  params.staticScoreUtilityFactor = 0.1;
  params.dynamicScoreUtilityFactor = 0.3;
  params.dynamicScoreCenterZeroWeight = 0.2;
  params.dynamicScoreCenterScale = 0.75;
  params.cpuctExploration = 0.9;
  params.cpuctExplorationLog = 0.4;
  params.rootFpuReductionMax = 0.1;
  params.rootPolicyTemperatureEarly = 1.2;
  params.rootPolicyTemperature = 1.1;
  params.useLcbForSelection = true;
  params.lcbStdevs = 5;
  params.minVisitPropForLCB = 0.15;
  params.rootEndingBonusPoints = 0.5;
  params.rootPruneUselessMoves = true;
  params.conservativePass = true;
  params.useNonBuggyLcb = true;
  params.useGraphSearch = true;
  params.fpuParentWeightByVisitedPolicy = true;
  params.valueWeightExponent = 0.25;
  params.useNoisePruning = true;
  params.useUncertainty = true;
  params.uncertaintyCoeff = 0.25;
  params.uncertaintyExponent = 1.0;
  params.uncertaintyMaxWeight = 8.0;
  params.cpuctUtilityStdevPrior = 0.40;
  params.cpuctUtilityStdevPriorWeight = 2.0;
  params.cpuctUtilityStdevScale = 0.85;
  params.fillDameBeforePass = true;
  params.subtreeValueBiasFactor = 0.45;
  params.subtreeValueBiasFreeProp = 0.8;
  params.subtreeValueBiasWeightExponent = 0.85;
  return params;
}

SearchParams SearchParams::basicDecentParams() {
  SearchParams params;
  params.staticScoreUtilityFactor = 0.1;
  params.dynamicScoreUtilityFactor = 0.3;
  params.dynamicScoreCenterZeroWeight = 0.2;
  params.dynamicScoreCenterScale = 0.75;
  params.cpuctExploration = 1.0;
  params.cpuctExplorationLog = 0.45;
  params.rootFpuReductionMax = 0.1;
  params.rootPolicyTemperatureEarly = 1.0;
  params.rootPolicyTemperature = 1.0;
  params.useLcbForSelection = true;
  params.lcbStdevs = 5;
  params.minVisitPropForLCB = 0.20;
  params.rootEndingBonusPoints = 0.5;
  params.rootPruneUselessMoves = true;
  params.conservativePass = true;
  params.enablePassingHacks = true;
  params.useNonBuggyLcb = true;
  params.useGraphSearch = true;
  params.fpuParentWeightByVisitedPolicy = true;
  params.valueWeightExponent = 0.25;
  params.useNoisePruning = true;
  params.useUncertainty = true;
  params.uncertaintyCoeff = 0.25;
  params.uncertaintyExponent = 1.0;
  params.uncertaintyMaxWeight = 8.0;
  params.cpuctUtilityStdevPrior = 0.40;
  params.cpuctUtilityStdevPriorWeight = 2.0;
  params.cpuctUtilityStdevScale = 0.85;
  params.fillDameBeforePass = true;
  params.subtreeValueBiasFactor = 0.45;
  params.subtreeValueBiasFreeProp = 0.8;
  params.subtreeValueBiasWeightExponent = 0.85;
  return params;
}

void SearchParams::failIfParamsDifferOnUnchangeableParameter(const SearchParams& initial, const SearchParams& dynamic) {
  if(dynamic.nodeTableShardsPowerOfTwo != initial.nodeTableShardsPowerOfTwo) {
    throw StringError("Cannot change nodeTableShardsPowerOfTwo after initialization");
  }
}

json SearchParams::changeableParametersToJson() const {
  json ret;
  ret["winLossUtilityFactor"] = winLossUtilityFactor;
  ret["staticScoreUtilityFactor"] = staticScoreUtilityFactor;
  ret["dynamicScoreUtilityFactor"] = dynamicScoreUtilityFactor;
  ret["dynamicScoreCenterZeroWeight"] = dynamicScoreCenterZeroWeight;
  ret["dynamicScoreCenterScale"] = dynamicScoreCenterScale;
  ret["noResultUtilityForWhite"] = noResultUtilityForWhite;
  ret["drawEquivalentWinsForWhite"] = drawEquivalentWinsForWhite;

  ret["cpuctExploration"] = cpuctExploration;
  ret["cpuctExplorationLog"] = cpuctExplorationLog;
  ret["cpuctExplorationBase"] = cpuctExplorationBase;

  ret["cpuctUtilityStdevPrior"] = cpuctUtilityStdevPrior;
  ret["cpuctUtilityStdevPriorWeight"] = cpuctUtilityStdevPriorWeight;
  ret["cpuctUtilityStdevScale"] = cpuctUtilityStdevScale;

  ret["fpuReductionMax"] = fpuReductionMax;
  ret["fpuLossProp"] = fpuLossProp;

  ret["fpuParentWeightByVisitedPolicy"] = fpuParentWeightByVisitedPolicy;
  ret["fpuParentWeightByVisitedPolicyPow"] = fpuParentWeightByVisitedPolicyPow;
  ret["fpuParentWeight"] = fpuParentWeight;

  ret["policyOptimism"] = policyOptimism;

  ret["valueWeightExponent"] = valueWeightExponent;
  ret["useNoisePruning"] = useNoisePruning;
  ret["noisePruneUtilityScale"] = noisePruneUtilityScale;
  ret["noisePruningCap"] = noisePruningCap;

  ret["useUncertainty"] = useUncertainty;
  ret["uncertaintyCoeff"] = uncertaintyCoeff;
  ret["uncertaintyExponent"] = uncertaintyExponent;
  ret["uncertaintyMaxWeight"] = uncertaintyMaxWeight;

  ret["useGraphSearch"] = useGraphSearch;
  ret["graphSearchRepBound"] = graphSearchRepBound;
  ret["graphSearchCatchUpLeakProb"] = graphSearchCatchUpLeakProb;

  ret["rootNoiseEnabled"] = rootNoiseEnabled;
  ret["rootDirichletNoiseTotalConcentration"] = rootDirichletNoiseTotalConcentration;
  ret["rootDirichletNoiseWeight"] = rootDirichletNoiseWeight;

  ret["rootPolicyTemperature"] = rootPolicyTemperature;
  ret["rootPolicyTemperatureEarly"] = rootPolicyTemperatureEarly;
  ret["rootFpuReductionMax"] = rootFpuReductionMax;
  ret["rootFpuLossProp"] = rootFpuLossProp;
  ret["rootNumSymmetriesToSample"] = rootNumSymmetriesToSample;
  ret["rootSymmetryPruning"] = rootSymmetryPruning;
  ret["rootDesiredPerChildVisitsCoeff"] = rootDesiredPerChildVisitsCoeff;

  ret["rootPolicyOptimism"] = rootPolicyOptimism;

  ret["chosenMoveTemperature"] = chosenMoveTemperature;
  ret["chosenMoveTemperatureEarly"] = chosenMoveTemperatureEarly;
  ret["chosenMoveTemperatureHalflife"] = chosenMoveTemperatureHalflife;
  ret["chosenMoveTemperatureOnlyBelowProb"] = chosenMoveTemperatureOnlyBelowProb;

  ret["chosenMoveSubtract"] = chosenMoveSubtract;
  ret["chosenMovePrune"] = chosenMovePrune;
  ret["useLcbForSelection"] = useLcbForSelection;

  ret["lcbStdevs"] = lcbStdevs;
  ret["minVisitPropForLCB"] = minVisitPropForLCB;
  ret["useNonBuggyLcb"] = useNonBuggyLcb;
  ret["rootEndingBonusPoints"] = rootEndingBonusPoints;

  ret["rootPruneUselessMoves"] = rootPruneUselessMoves;
  ret["conservativePass"] = conservativePass;
  ret["fillDameBeforePass"] = fillDameBeforePass;
  // Unused
  // ret["avoidMYTDaggerHackPla"] = PlayerIO::playerToStringShort(avoidMYTDaggerHackPla);
  ret["wideRootNoise"] = wideRootNoise;
  ret["enablePassingHacks"] = enablePassingHacks;

  // Special handling in GTP
  ret["playoutDoublingAdvantage"] = playoutDoublingAdvantage;
  ret["playoutDoublingAdvantagePla"] = PlayerIO::playerToStringShort(playoutDoublingAdvantagePla);

  // Special handling in GTP
  // ret["avoidRepeatedPatternUtility"] = avoidRepeatedPatternUtility;

  ret["nnPolicyTemperature"] = nnPolicyTemperature;
  // Special handling in GTP
  // ret["antiMirror"] = antiMirror;

  ret["ignorePreRootHistory"] = ignorePreRootHistory;
  ret["ignoreAllHistory"] = ignoreAllHistory;

  ret["subtreeValueBiasFactor"] = subtreeValueBiasFactor;
  ret["subtreeValueBiasTableNumShards"] = subtreeValueBiasTableNumShards;
  ret["subtreeValueBiasFreeProp"] = subtreeValueBiasFreeProp;
  ret["subtreeValueBiasWeightExponent"] = subtreeValueBiasWeightExponent;

  // ret["nodeTableShardsPowerOfTwo"] = nodeTableShardsPowerOfTwo;
  ret["numVirtualLossesPerThread"] = numVirtualLossesPerThread;

  ret["numSearchThreads"] = numThreads; // NOTE: different name since that's how setup.cpp loads it
  ret["minPlayoutsPerThread"] = minPlayoutsPerThread;
  ret["maxVisits"] = maxVisits;
  ret["maxPlayouts"] = maxPlayouts;
  ret["maxTime"] = maxTime;

  ret["maxVisitsPondering"] = maxVisitsPondering;
  ret["maxPlayoutsPondering"] = maxPlayoutsPondering;
  ret["maxTimePondering"] = maxTimePondering;

  ret["lagBuffer"] = lagBuffer;

  ret["searchFactorAfterOnePass"] = searchFactorAfterOnePass;
  ret["searchFactorAfterTwoPass"] = searchFactorAfterTwoPass;

  ret["treeReuseCarryOverTimeFactor"] = treeReuseCarryOverTimeFactor;
  ret["overallocateTimeFactor"] = overallocateTimeFactor;
  ret["midgameTimeFactor"] = midgameTimeFactor;
  ret["midgameTurnPeakTime"] = midgameTurnPeakTime;
  ret["endgameTurnTimeDecay"] = endgameTurnTimeDecay;
  ret["obviousMovesTimeFactor"] = obviousMovesTimeFactor;
  ret["obviousMovesPolicyEntropyTolerance"] = obviousMovesPolicyEntropyTolerance;
  ret["obviousMovesPolicySurpriseTolerance"] = obviousMovesPolicySurpriseTolerance;

  ret["futileVisitsThreshold"] = futileVisitsThreshold;

  ret["humanSLCpuctExploration"] = humanSLCpuctExploration;
  ret["humanSLCpuctPermanent"] = humanSLCpuctPermanent;

  ret["humanSLRootExploreProbWeightless"] = humanSLRootExploreProbWeightless;
  ret["humanSLRootExploreProbWeightful"] = humanSLRootExploreProbWeightful;
  ret["humanSLPlaExploreProbWeightless"] = humanSLPlaExploreProbWeightless;
  ret["humanSLPlaExploreProbWeightful"] = humanSLPlaExploreProbWeightful;
  ret["humanSLOppExploreProbWeightless"] = humanSLOppExploreProbWeightless;
  ret["humanSLOppExploreProbWeightful"] = humanSLOppExploreProbWeightful;

  ret["humanSLChosenMoveProp"] = humanSLChosenMoveProp;
  ret["humanSLChosenMoveIgnorePass"] = humanSLChosenMoveIgnorePass;
  ret["humanSLChosenMovePiklLambda"] = humanSLChosenMovePiklLambda;

  return ret;
}

#define PRINTPARAM(PARAMNAME) out << #PARAMNAME << ": " << PARAMNAME << std::endl;
void SearchParams::printParams(std::ostream& out) const {


  PRINTPARAM(winLossUtilityFactor);
  PRINTPARAM(staticScoreUtilityFactor);
  PRINTPARAM(dynamicScoreUtilityFactor);
  PRINTPARAM(dynamicScoreCenterZeroWeight);
  PRINTPARAM(dynamicScoreCenterScale);
  PRINTPARAM(noResultUtilityForWhite);
  PRINTPARAM(drawEquivalentWinsForWhite);

  PRINTPARAM(cpuctExploration);
  PRINTPARAM(cpuctExplorationLog);
  PRINTPARAM(cpuctExplorationBase);

  PRINTPARAM(cpuctUtilityStdevPrior);
  PRINTPARAM(cpuctUtilityStdevPriorWeight);
  PRINTPARAM(cpuctUtilityStdevScale);

  PRINTPARAM(fpuReductionMax);
  PRINTPARAM(fpuLossProp);

  PRINTPARAM(fpuParentWeightByVisitedPolicy);
  PRINTPARAM(fpuParentWeightByVisitedPolicyPow);
  PRINTPARAM(fpuParentWeight);

  PRINTPARAM(policyOptimism);

  PRINTPARAM(valueWeightExponent);
  PRINTPARAM(useNoisePruning);
  PRINTPARAM(noisePruneUtilityScale);
  PRINTPARAM(noisePruningCap);


  PRINTPARAM(useUncertainty);
  PRINTPARAM(uncertaintyCoeff);
  PRINTPARAM(uncertaintyExponent);
  PRINTPARAM(uncertaintyMaxWeight);


  PRINTPARAM(useGraphSearch);
  PRINTPARAM(graphSearchRepBound);
  PRINTPARAM(graphSearchCatchUpLeakProb);



  PRINTPARAM(rootNoiseEnabled);
  PRINTPARAM(rootDirichletNoiseTotalConcentration);
  PRINTPARAM(rootDirichletNoiseWeight);

  PRINTPARAM(rootPolicyTemperature);
  PRINTPARAM(rootPolicyTemperatureEarly);
  PRINTPARAM(rootFpuReductionMax);
  PRINTPARAM(rootFpuLossProp);
  PRINTPARAM(rootNumSymmetriesToSample);
  PRINTPARAM(rootSymmetryPruning);

  PRINTPARAM(rootDesiredPerChildVisitsCoeff);

  PRINTPARAM(rootPolicyOptimism);

  PRINTPARAM(chosenMoveTemperature);
  PRINTPARAM(chosenMoveTemperatureEarly);
  PRINTPARAM(chosenMoveTemperatureHalflife);
  PRINTPARAM(chosenMoveTemperatureOnlyBelowProb);
  PRINTPARAM(chosenMoveSubtract);
  PRINTPARAM(chosenMovePrune);

  PRINTPARAM(useLcbForSelection);
  PRINTPARAM(lcbStdevs);
  PRINTPARAM(minVisitPropForLCB);
  PRINTPARAM(useNonBuggyLcb);


  PRINTPARAM(rootEndingBonusPoints);
  PRINTPARAM(rootPruneUselessMoves);
  PRINTPARAM(conservativePass);
  PRINTPARAM(fillDameBeforePass);
  std::cout << "avoidMYTDaggerHackPla" << ": " << (int)avoidMYTDaggerHackPla << std::endl;
  PRINTPARAM(wideRootNoise);
  PRINTPARAM(enablePassingHacks);

  PRINTPARAM(playoutDoublingAdvantage);
  std::cout << "playoutDoublingAdvantagePla" << ": " << (int)playoutDoublingAdvantagePla << std::endl;

  PRINTPARAM(avoidRepeatedPatternUtility);

  PRINTPARAM(nnPolicyTemperature);
  PRINTPARAM(antiMirror);

  PRINTPARAM(ignorePreRootHistory);
  PRINTPARAM(ignoreAllHistory);

  PRINTPARAM(subtreeValueBiasFactor);
  PRINTPARAM(subtreeValueBiasTableNumShards);
  PRINTPARAM(subtreeValueBiasFreeProp);
  PRINTPARAM(subtreeValueBiasWeightExponent);


  PRINTPARAM(nodeTableShardsPowerOfTwo);
  PRINTPARAM(numVirtualLossesPerThread);


  PRINTPARAM(numThreads);
  PRINTPARAM(minPlayoutsPerThread);
  PRINTPARAM(maxVisits);
  PRINTPARAM(maxPlayouts);
  PRINTPARAM(maxTime);


  PRINTPARAM(maxVisitsPondering);
  PRINTPARAM(maxPlayoutsPondering);
  PRINTPARAM(maxTimePondering);


  PRINTPARAM(lagBuffer);


  PRINTPARAM(searchFactorAfterOnePass);
  PRINTPARAM(searchFactorAfterTwoPass);


  PRINTPARAM(treeReuseCarryOverTimeFactor);
  PRINTPARAM(overallocateTimeFactor);
  PRINTPARAM(midgameTimeFactor);
  PRINTPARAM(midgameTurnPeakTime);
  PRINTPARAM(endgameTurnTimeDecay);
  PRINTPARAM(obviousMovesTimeFactor);
  PRINTPARAM(obviousMovesPolicyEntropyTolerance);
  PRINTPARAM(obviousMovesPolicySurpriseTolerance);

  PRINTPARAM(futileVisitsThreshold);


  PRINTPARAM(humanSLCpuctExploration);
  PRINTPARAM(humanSLCpuctPermanent);
  PRINTPARAM(humanSLRootExploreProbWeightless);
  PRINTPARAM(humanSLRootExploreProbWeightful);
  PRINTPARAM(humanSLPlaExploreProbWeightless);
  PRINTPARAM(humanSLPlaExploreProbWeightful);
  PRINTPARAM(humanSLOppExploreProbWeightless);
  PRINTPARAM(humanSLOppExploreProbWeightful);
  PRINTPARAM(humanSLChosenMoveProp);
  PRINTPARAM(humanSLChosenMoveIgnorePass);
  PRINTPARAM(humanSLChosenMovePiklLambda);

}
