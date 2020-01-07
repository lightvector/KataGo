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
   fpuUseParentAverage(true),
   valueWeightExponent(0.5),
   visitsExponent(1.0),
   scaleParentWeight(true),
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
   rootEndingBonusPoints(0.0),
   rootPruneUselessMoves(false),
   conservativePass(false),
   fillDameBeforePass(false),
   localExplore(false),
   playoutDoublingAdvantage(0.0),
   playoutDoublingAdvantagePla(C_EMPTY),
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
   searchFactorAfterTwoPass(1.0)
{}

SearchParams::~SearchParams()
{}
