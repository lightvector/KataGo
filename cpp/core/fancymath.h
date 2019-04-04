#ifndef FANCYMATH
#define FANCYMATH

#include "../core/global.h"

namespace FancyMath {
  //For large or extreme values these might not be too accurate, use GSL or Boost for more accuracy
  double beta(double a, double b);
  double logbeta(double a, double b);
  double incompleteBeta(double x, double a, double b);
  double regularizedIncompleteBeta(double x, double a, double b);

  double evaluateContinuedFraction(const function<double(int)>& numer, const function<double(int)>& denom, double tolerance, int maxTerms);

  //For large or extreme values these might not be too accurate, use GSL or Boost for more accuracy
  double tdistpdf(double x, double degreesOfFreedom);
  double tdistcdf(double x, double degreesOfFreedom);
  double betapdf(double x, double a, double b);
  double betacdf(double x, double a, double b);

  //Given z, compute and return an approximation for the value t
  //such that the probability that a draw from StudentT(degreesOfFreedom) > t
  //is the same as a probability that a draw from StandardNormal() > z
  double normToTApprox(double z, double degreesOfFreedom);
  
  void runTests();
}


#endif
