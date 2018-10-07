#include "../search/searchparams.h"

//Default search params
//The intent is that the are good default guesses for good values of the parameters,
//with deterministic behavior (no noise, no randomization) and no bounds (unbounded time and visits).
//Currently, utility is entirely win-loss.
SearchParams::SearchParams()
  :winLossUtilityFactor(1.0),
   scoreUtilityFactor(0.0),
   noResultUtilityForWhite(0.0),
   drawUtilityForWhite(0.0),
   cpuctExploration(1.0),
   fpuReductionMax(0.2),
   fpuUseParentAverage(true),
   valueWeightExponent(0.5),
   visitsExponent(1.0),
   rootNoiseEnabled(false),
   rootDirichletNoiseTotalConcentration(10.83),
   rootDirichletNoiseWeight(0.25),
   chosenMoveTemperature(0.0),
   chosenMoveTemperatureEarly(0.0),
   chosenMoveTemperatureHalflife(16),
   chosenMoveSubtract(1.0),
   mutexPoolSize(8192),
   numVirtualLossesPerThread(3),
   numThreads(1),
   maxVisits(((uint64_t)1) << 63),
   maxPlayouts(((uint64_t)1) << 63),
   maxTime(1.0e20)
{}

SearchParams::~SearchParams()
{}
