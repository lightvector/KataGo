#ifndef SEARCH_PARAMS_H
#define SEARCH_PARAMS_H

struct SearchParams {
  double winLossUtilityFactor;
  double scoreUtilityFactor;
  
  double noResultUtilityForWhite;

  double cpuctExploration;
  double fpuReductionMax;

  bool rootNoiseEnabled;
  double rootDirichletNoiseTotalConcentration;
  double rootDirichletNoiseWeight;
  
  SearchParams();
  ~SearchParams();
};

#endif
