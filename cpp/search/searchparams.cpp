#include "../search/searchparams.h"

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
   fpuReductionMax(0.2),
   fpuLossProp(0.0),
   fpuParentWeight(0.0),
   parentValueWeightFactor(1.0),
   valueWeightExponent(0.5),
   rootNoiseEnabled(false),
   rootDirichletNoiseTotalConcentration(10.83),
   rootDirichletNoiseWeight(0.25),
   rootPolicyTemperature(1.0),
   rootPolicyTemperatureEarly(1.0),
   rootFpuReductionMax(0.2),
   rootFpuLossProp(0.0),
   rootNumSymmetriesToSample(1),
   rootDesiredPerChildVisitsCoeff(0.0),
   chosenMoveTemperature(0.0),
   chosenMoveTemperatureEarly(0.0),
   chosenMoveTemperatureHalflife(19),
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
   playoutDoublingAdvantage(0.0),
   playoutDoublingAdvantagePla(C_EMPTY),
   nnPolicyTemperature(1.0f),
   antiMirror(false),
   subtreeValueBiasFactor(0.0),
   subtreeValueBiasTableNumShards(65536),
   subtreeValueBiasFreeProp(0.8),
   subtreeValueBiasWeightExponent(0.5),
   mutexPoolSize(8192),
   numVirtualLossesPerThread(3),
   numThreads(1),
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
   futileVisitsThreshold(0.0)
{}

SearchParams::~SearchParams()
{}

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

void SearchParams::failIfParamsDifferOnUnchangeableParameter(const SearchParams& initial, const SearchParams& dynamic) {
  if(dynamic.numThreads > initial.numThreads) {
    throw StringError("Cannot increase number of search threads after initialization since this is used to initialize neural net buffer capacity");
  }
  if(dynamic.mutexPoolSize != initial.mutexPoolSize) {
    throw StringError("Cannot change mutex pool size after initialization");
  }
}
