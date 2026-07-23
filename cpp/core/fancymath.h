#ifndef CORE_FANCYMATH_H_
#define CORE_FANCYMATH_H_

#include "../core/global.h"

namespace FancyMath {
  //For large or extreme values these might not be too accurate, use GSL or Boost for more accuracy
  double beta(double a, double b);
  double logbeta(double a, double b);
  double incompleteBeta(double x, double a, double b);
  double regularizedIncompleteBeta(double x, double a, double b);

  double evaluateContinuedFraction(const std::function<double(int)>& numer, const std::function<double(int)>& denom, double tolerance, int maxTerms);

  //For large or extreme values these might not be too accurate, use GSL or Boost for more accuracy
  double tdistpdf(double x, double degreesOfFreedom);
  double tdistcdf(double x, double degreesOfFreedom);
  double betapdf(double x, double a, double b);
  double betacdf(double x, double a, double b);

  //Given z, compute and return an approximation for the value t
  //such that the probability that a draw from StudentT(degreesOfFreedom) > t
  //is the same as a probability that a draw from StandardNormal() > z
  double normToTApprox(double z, double degreesOfFreedom);

  //predProb is scaled into the range [epsilon,1.0-epsilon].
  double binaryCrossEntropy(double predProb, double targetProb, double epsilon);

  //Wilson score 95% two-tailed confidence interval for binomial proportion.
  //Draws should be counted as 0.5 wins before calling.
  void wilsonCI95(double wins, double n, double& lo, double& hi);

  //One-tailed p-value for H0: winrate=0.5 vs H1: winrate>0.5, using normal approximation.
  //Small values indicate the first player wins significantly more than 50%.
  double oneTailedPValue(double wins, double n);

  void runTests();
}


#endif  // CORE_FANCYMATH_H_
