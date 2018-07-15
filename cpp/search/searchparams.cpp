#include "../search/searchparams.h"

SearchParams::SearchParams()
  :winLossUtilityFactor(1.0),
   scoreUtilityFactor(0.08),
   noResultUtilityForWhite(0.0),
   cpuctExploration(1.6),
   fpuReductionMax(0.5),
   rootNoiseEnabled(false),
   rootDirichletNoiseTotalConcentration(10.0),
   rootDirichletNoiseWeight(0.25),
   randSeed(""), //Empty string is still a random seed
   mutexPoolSize(8192),
   numThreads(1)
{}

SearchParams::~SearchParams()
{}
